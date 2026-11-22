import base64
import io
import re
import difflib

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

MAX_FILE_MB = 40
MAX_PAGES = 30
RENDER_DPI = 120
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WORD_EXTS = {".docx"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt", ".md", ".csv", ".html", ".rtf"}

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)

_ocr_engine = None


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def ocr_ndarray(img_array):
    """Returns list of (line_text, [x0,y0,x1,y1])."""
    result, _ = get_ocr()(img_array)
    out = []
    for box_pts, text, _score in (result or []):
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        out.append((text, [min(xs), min(ys), max(xs), max(ys)]))
    return out


def gemini_ocr(img):
    """Vision-LLM OCR via Gemini. Returns (line_text, box) per line where box is
    [x0,y0,x1,y1] in image pixels, or None when no key/failure."""
    import json
    import os
    import urllib.request
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    body = {
        "contents": [{"parts": [
            {"text": ('Transcribe ALL text in this document exactly — every word, '
                      'number and punctuation mark must be preserved verbatim. '
                      'Respond with JSON only: {"lines": [{"text": "...", '
                      '"box": [ymin, xmin, ymax, xmax]}]} where box coordinates '
                      'are normalized 0-1000 relative to image size.')},
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(buf.getvalue()).decode()}}]}],
        "generationConfig": {"temperature": 0,
                             "response_mime_type": "application/json"},
    }
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-3.1-flash-lite:generateContent?key={key}")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        lines = json.loads(text)["lines"]
        out = []
        for ln in lines:
            ymin, xmin, ymax, xmax = ln["box"]
            out.append((ln["text"], [xmin / 1000 * img.width,
                                     ymin / 1000 * img.height,
                                     xmax / 1000 * img.width,
                                     ymax / 1000 * img.height]))
        return out
    except Exception:
        return None


def ocr_pil(img):
    """OCR with 2x upscale for small images — tiny punctuation like trailing
    periods is often lost at low resolution. Boxes are mapped back to the
    original pixel space."""
    g = gemini_ocr(img)
    if g:
        return g
    import numpy as np
    from PIL import Image
    img = img.convert("RGB")
    scale = 1.0
    if img.width < 2000:
        scale = min(2.0, 2000.0 / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    lines = ocr_ndarray(np.array(img))
    if scale != 1.0:
        lines = [(t, [v / scale for v in b]) for t, b in lines]
    return lines


def b64_png(pil_or_bytes):
    from PIL import Image
    if isinstance(pil_or_bytes, bytes):
        return base64.b64encode(pil_or_bytes).decode()
    buf = io.BytesIO()
    pil_or_bytes.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def apportion_pieces(text, geom):
    """Split a line-level box across words proportionally by character span,
    so highlights land on the differing word instead of the whole line."""
    out = []
    for m in re.finditer(r"\S+|\s+", text):
        t = m.group(0)
        if t.isspace() or geom is None:
            out.append((t, None))
            continue
        x0, y0, x1, y1 = geom[1]
        span = x1 - x0
        sub = [x0 + span * m.start() / len(text),
               y0,
               x0 + span * m.end() / len(text),
               y1]
        out.append((t, (geom[0], sub)))
    return out


def extract_pdf(file_bytes):
    """Returns (pieces, pages). pieces = [(text, box_or_None)] where box is
    (page_index, [x0,y0,x1,y1]) in rendered-image pixels."""
    import pymupdf
    pieces, pages = [], []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for pno, page in enumerate(doc):
            if pno >= MAX_PAGES:
                break
            words = page.get_text("words")
            pix = page.get_pixmap(dpi=RENDER_DPI)
            scale = pix.width / page.rect.width
            pages.append({"image": b64_png(pix.tobytes("png")),
                          "w": pix.width, "h": pix.height, "boxes": []})
            if pieces:
                pieces.append(("\n\n", None))
            if words:
                cur_line = None
                for w in sorted(words, key=lambda w: (w[5], w[6], w[7])):
                    line_id = (w[5], w[6])
                    if cur_line is not None:
                        pieces.append((" " if line_id == cur_line else "\n", None))
                    cur_line = line_id
                    box = [w[0] * scale, w[1] * scale, w[2] * scale, w[3] * scale]
                    pieces.append((w[4], (pno, box)))
            else:
                # scanned page -> OCR on the rendered pixels
                from PIL import Image
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                for i, (text, box) in enumerate(ocr_pil(img)):
                    if i:
                        pieces.append(("\n", None))
                    pieces.extend(apportion_pieces(text, (pno, box)))
    return pieces, pages


def extract_docx(file_bytes):
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return [("\n".join(parts), None)], []


def extract_image(file_bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes))
    pages = [{"image": b64_png(img.convert("RGB")),
              "w": img.width, "h": img.height, "boxes": []}]
    pieces = []
    for i, (text, box) in enumerate(ocr_pil(img)):
        if i:
            pieces.append(("\n", None))
        pieces.extend(apportion_pieces(text, (0, box)))
    return pieces, pages


def extract(filename, file_bytes):
    ext = filename.lower().rsplit(".", 1)
    ext = "." + ext[-1] if len(ext) == 2 else ""
    if ext in PDF_EXTS:
        return *extract_pdf(file_bytes), "pdf"
    if ext in WORD_EXTS:
        pieces, pages = extract_docx(file_bytes)
        return pieces, pages, "word"
    if ext in TEXT_EXTS:
        text = file_bytes.decode("utf-8", errors="replace")
        return [(text, None)], [], "text"
    if ext in IMAGE_EXTS or not ext:
        pieces, pages = extract_image(file_bytes)
        return pieces, pages, "image (OCR)"
    raise ValueError(f"Format tidak didukung: {ext or filename}")


def normalize_text(t):
    return re.sub(r"\s+", " ", t).strip()


def tokenize(pieces):
    tokens, geom = [], []
    for text, box in pieces:
        for t in TOKEN_RE.findall(text):
            tokens.append(t)
            geom.append(box if not t.isspace() else None)
    return tokens, geom


@app.post("/api/compare")
def compare():
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    if not f1 or not f2:
        return jsonify({"error": "Upload dua file dulu (Dokumen 1 = acuan, Dokumen 2 = yang dicek)."}), 400
    try:
        pieces1, pages1, kind1 = extract(f1.filename, f1.read())
        pieces2, pages2, kind2 = extract(f2.filename, f2.read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {e}"}), 500

    norm = request.form.get("normalize") == "1"
    t1, g1 = tokenize(pieces1)
    t2, g2 = tokenize(pieces2)
    if norm:
        keep = [i for i, t in enumerate(t1) if not t.isspace()]
        t1n = [t1[i] for i in keep]
        t1n.insert(0, " ")
        g1n = [g1[i] for i in keep]
        g1n.insert(0, None)
        keep2 = [i for i, t in enumerate(t2) if not t.isspace()]
        t2n = [t2[i] for i in keep2]
        t2n.insert(0, " ")
        g2n = [g2[i] for i in keep2]
        g2n.insert(0, None)
        t1, g1, t2, g2 = t1n, g1n, t2n, g2n
        text1 = " ".join(t for t in t1 if not t.isspace())
        text2 = " ".join(t for t in t2 if not t.isspace())
    else:
        text1 = "".join(t1)
        text2 = "".join(t2)

    # run diff, collecting page boxes per side
    sm = difflib.SequenceMatcher(a=t1, b=t2, autojunk=False)
    chunks, stats = [], {"equal": 0, "replace": 0, "insert": 0, "delete": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        stats[tag] += 1
        chunks.append({"tag": tag, "a": "".join(t1[i1:i2]), "b": "".join(t2[j1:j2])})
        if tag == "equal":
            continue
        for g in g1[i1:i2]:
            if g and g[0] < len(pages1):
                pages1[g[0]]["boxes"].append(g[1])
        for g in g2[j1:j2]:
            if g and g[0] < len(pages2):
                pages2[g[0]]["boxes"].append(g[1])

    stats["changed_regions"] = stats["replace"] + stats["insert"] + stats["delete"]
    return jsonify({
        "file1": {"name": f1.filename, "type": kind1, "chars": len(text1)},
        "file2": {"name": f2.filename, "type": kind2, "chars": len(text2)},
        "text1": text1, "text2": text2,
        "chunks": chunks,
        "stats": stats,
        "identical": text1 == text2,
        "pages1": pages1, "pages2": pages2,
    })


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
