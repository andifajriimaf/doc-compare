import io
import re
import difflib
import unicodedata

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

MAX_FILE_MB = 40
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WORD_EXTS = {".docx"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt", ".md", ".csv", ".html", ".rtf"}

_ocr_engine = None


def ocr_image(path):
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    result, _ = _ocr_engine(path)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def extract_pdf(file_bytes):
    import pymupdf
    out = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                out.append(text)
            else:
                # scanned page -> OCR the rendered page
                pix = page.get_pixmap(dpi=200)
                out.append(ocr_image_from_bytes(pix.tobytes("png")))
    return "\n".join(out)


def ocr_image_from_bytes(img_bytes):
    import numpy as np
    from PIL import Image
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    result, _ = _ocr_engine(img)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def extract_docx(file_bytes):
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def normalize_text(t):
    # collapse every whitespace run to a single space -> line-break/layout
    # differences between formats don't count; punctuation still must match
    return re.sub(r"\s+", " ", t).strip()


def extract_text(filename, file_bytes):
    ext = filename.lower().rsplit(".", 1)
    ext = "." + ext[-1] if len(ext) == 2 else ""
    if ext in PDF_EXTS:
        return extract_pdf(file_bytes), "pdf"
    if ext in WORD_EXTS:
        return extract_docx(file_bytes), "word"
    if ext in TEXT_EXTS:
        return file_bytes.decode("utf-8", errors="replace"), "text"
    if ext in IMAGE_EXTS or not ext:
        return ocr_image_from_bytes(file_bytes), "image (OCR)"
    raise ValueError(f"Format tidak didukung: {ext or filename}")


TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def diff_texts(text1, text2):
    """Token-level diff: every word, punctuation mark, and whitespace run
    is a token, so punctuation differences are caught exactly."""
    t1 = TOKEN_RE.findall(text1)
    t2 = TOKEN_RE.findall(text2)
    sm = difflib.SequenceMatcher(a=t1, b=t2, autojunk=False)
    chunks = []
    stats = {"equal": 0, "replace": 0, "insert": 0, "delete": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        stats[tag] += 1
        chunks.append({
            "tag": tag,
            "a": "".join(t1[i1:i2]),
            "b": "".join(t2[j1:j2]),
            "a_tokens": t1[i1:i2],
            "b_tokens": t2[j1:j2],
        })
    total = stats["replace"] + stats["insert"] + stats["delete"]
    return chunks, {"changed_regions": total, **stats}


@app.post("/api/compare")
def compare():
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    if not f1 or not f2:
        return jsonify({"error": "Upload dua file dulu (Dokumen 1 = acuan, Dokumen 2 = yang dicek)."}), 400
    try:
        text1, kind1 = extract_text(f1.filename, f1.read())
        text2, kind2 = extract_text(f2.filename, f2.read())
        if request.form.get("normalize") == "1":
            text1 = normalize_text(text1)
            text2 = normalize_text(text2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {e}"}), 500

    chunks, stats = diff_texts(text1, text2)
    return jsonify({
        "file1": {"name": f1.filename, "type": kind1, "chars": len(text1)},
        "file2": {"name": f2.filename, "type": kind2, "chars": len(text2)},
        "text1": text1,
        "text2": text2,
        "chunks": chunks,
        "stats": stats,
        "identical": text1 == text2,
    })


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
