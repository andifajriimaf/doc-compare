import base64
import io
import re
import difflib
import os
import threading

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
    import time
    import urllib.error
    data = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            # overload / rate limit is usually temporary
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                print(f"gemini_ocr: HTTP {e.code}", flush=True)
                return None
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"gemini_ocr: {type(e).__name__}", flush=True)
            return None
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        lines = json.loads(text)["lines"]
        out = []
        for ln in lines:
            box = ln["box"]
            while len(box) == 1 and isinstance(box[0], list):
                box = box[0]
            ymin, xmin, ymax, xmax = box
            out.append((ln["text"], [xmin / 1000 * img.width,
                                     ymin / 1000 * img.height,
                                     xmax / 1000 * img.width,
                                     ymax / 1000 * img.height]))
        return out
    except Exception as e:
        print(f"gemini_ocr: bad response ({type(e).__name__})", flush=True)
        return None


def local_ocr(img):
    """RapidOCR with 2x upscale for small images — tiny punctuation like
    trailing periods is often lost at low resolution. Boxes are mapped back
    to the original pixel space."""
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


def split_words(text, box):
    """Split a line box across its words proportionally by character span."""
    out = []
    x0, y0, x1, y1 = box
    for m in re.finditer(r"\S+", text):
        out.append((m.group(0), [x0 + (x1 - x0) * m.start() / len(text), y0,
                                 x0 + (x1 - x0) * m.end() / len(text), y1]))
    return out


def char_w(ch):
    """Rough relative glyph width in a proportional font."""
    if ch.isspace():
        return 0.55
    if ch in "ilI.,:;'|!j":
        return 0.45
    if ch in "ftr()-\"/[]":
        return 0.6
    if ch in "mwMW":
        return 1.5
    if ch.isupper():
        return 1.25
    return 1.0


def local_rows(l_lines):
    """Group detector segments into visual text rows, left to right, and
    expand each row to per-character geometry."""
    rows = []
    for text, box in sorted(l_lines, key=lambda l: (l[1][1] + l[1][3]) / 2):
        cy, h = (box[1] + box[3]) / 2, box[3] - box[1]
        if rows and abs(rows[-1]["cy"] - cy) < 0.5 * max(h, rows[-1]["h"]):
            rows[-1]["segs"].append((text, box))
        else:
            rows.append({"cy": cy, "h": h, "segs": [(text, box)]})
    for row in rows:
        row["chars"], row["geo"] = [], []
        for text, (x0, y0, x1, y1) in sorted(row["segs"], key=lambda sg: sg[1][0]):
            widths = [char_w(ch) for ch in text]
            total, acc = sum(widths) or 1.0, 0.0
            for ch, cw in zip(text, widths):
                if not ch.isspace():
                    row["chars"].append(ch.lower())
                    row["geo"].append((x0 + (x1 - x0) * acc / total, y0,
                                       x0 + (x1 - x0) * (acc + cw) / total, y1))
                acc += cw
    return rows


def word_spans(text):
    """Glyph-width-weighted (start, end) fractions of each word in a line."""
    widths = [char_w(ch) for ch in text]
    total = sum(widths) or 1.0
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.group(0), sum(widths[:m.start()]) / total,
                      sum(widths[:m.end()]) / total))
    return spans


def ink_bands(img, x0, x1):
    """Vertical extents of text rows between x0 and x1, from pixel ink."""
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.int16)
    x0, x1 = max(0, int(x0)), min(g.shape[1], int(x1))
    if x1 - x0 < 4:
        return []
    strip = g[:, x0:x1]
    ink = (np.abs(strip - int(np.median(g))) > 45).mean(axis=1) > 0.01
    bands, start = [], None
    for y, on in enumerate(list(ink) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= 4:
                bands.append((start, y))
            start = None
    return bands


def snap_to_local(g_lines, l_lines, img):
    """Gemini reads text best but its boxes drift; the local detector reads
    worse but its boxes sit on the pixels. Match every Gemini line to the
    detector row(s) carrying the same characters and take the geometry there.
    Words the detector could not confirm are placed from their neighbours,
    from an unclaimed detector row, or from the pixel rows themselves."""
    rows = local_rows(l_lines)
    lines, fits, claimed = [], [], set()
    for text, gbox in g_lines:
        gx0, gy0, gx1, gy1 = gbox
        # [word, gemini box, snapped box, start frac, end frac]
        words = [[w, [gx0 + (gx1 - gx0) * p0, gy0, gx0 + (gx1 - gx0) * p1, gy1],
                  None, p0, p1] for w, p0, p1 in word_spans(text)]
        gcy = (gy0 + gy1) / 2
        used = set()
        for _ in range(3):  # a Gemini line may span several detector rows
            todo = [i for i, w in enumerate(words) if w[2] is None]
            if not todo:
                break
            chars, owner = [], []
            for i in todo:
                for ch in words[i][0]:
                    chars.append(ch.lower())
                    owner.append(i)
            best = None
            for ri, row in enumerate(rows):
                if ri in used:
                    continue
                blocks = difflib.SequenceMatcher(
                    None, chars, row["chars"], autojunk=False).get_matching_blocks()
                n = sum(bl.size for bl in blocks)
                score = n - 2.0 * abs(row["cy"] - gcy) / img.height
                if best is None or score > best[0]:
                    best = (score, ri, blocks, n)
            # a row sharing only a stray word with this line is not its row
            if best is None or best[3] < 1 or best[3] < 0.4 * len(chars):
                break
            _, ri, blocks, _ = best
            used.add(ri)
            hits = {}
            for bl in blocks:
                for k in range(bl.size):
                    hits.setdefault(owner[bl.a + k], []).append(rows[ri]["geo"][bl.b + k])
            for i, hs in hits.items():
                w = words[i][0]
                # drop stray single-character matches far from the word
                mid = sorted((h[0] + h[2]) / 2 for h in hs)[len(hs) // 2]
                cw = sorted(h[2] - h[0] for h in hs)[len(hs) // 2]
                reach = max(len(w), 2) * cw * 1.3
                hs = [h for h in hs if abs((h[0] + h[2]) / 2 - mid) <= reach]
                if len(hs) * 2 > len(w) or len(w) == 1:
                    words[i][2] = [min(h[0] for h in hs), min(h[1] for h in hs),
                                   max(h[2] for h in hs), max(h[3] for h in hs)]
                    fits.append((words[i][1], words[i][2]))
                    claimed.add(ri)
        lines.append(words)

    def fit(idx):
        xs = [g[idx] for g, _ in fits]
        ys = [l[idx] for _, l in fits]
        n = len(xs)
        if n < 2:
            return 1.0, 0.0
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        if var < 1e-6:
            return 1.0, my - mx
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
        return a, my - a * mx
    coef = [fit(i) for i in range(4)]

    def drift(box):
        return [coef[i][0] * box[i] + coef[i][1] for i in range(4)]

    def row_box(row):
        g = row["geo"]
        return [min(c[0] for c in g), min(c[1] for c in g),
                max(c[2] for c in g), max(c[3] for c in g)]

    out = []
    for (text, gbox), words in zip(g_lines, lines):
        done = [w for w in words if w[2] is not None]
        if done and len(done) < len(words):
            # place the rest relative to confirmed words of the same line
            for w in words:
                if w[2] is not None:
                    continue
                ref = min(done, key=lambda d: abs(d[3] - w[3]))
                pts = [(d[3], d[2][0]) for d in done if abs(d[2][1] - ref[2][1]) < 3]
                pts += [(d[4], d[2][2]) for d in done if abs(d[2][1] - ref[2][1]) < 3]
                mp = sum(p for p, _ in pts) / len(pts)
                mx = sum(x for _, x in pts) / len(pts)
                var = sum((p - mp) ** 2 for p, _ in pts)
                a = sum((p - mp) * (x - mx) for p, x in pts) / var if var > 1e-9 else 0
                if a <= 0:
                    a = (gbox[2] - gbox[0]) * coef[0][0]
                w[2] = [mx + a * (w[3] - mp), ref[2][1],
                        mx + a * (w[4] - mp), ref[2][3]]
        elif not done and words:
            est = drift(gbox)
            ecy = (est[1] + est[3]) / 2
            free = [(abs(r["cy"] - ecy), ri) for ri, r in enumerate(rows)
                    if ri not in claimed and r["geo"]
                    and abs(r["cy"] - ecy) < 1.5 * max(r["h"], est[3] - est[1])]
            if free:
                ri = min(free)[1]
                claimed.add(ri)
                x0, y0, x1, y1 = row_box(rows[ri])
            else:
                x0, y0, x1, y1 = est
                taken = [row_box(rows[ri]) for ri in claimed]
                bands = [b for b in ink_bands(img, x0, x1)
                         if not any(min(b[1], t[3]) - max(b[0], t[1])
                                    > 0.5 * (b[1] - b[0]) for t in taken)]
                if bands:
                    y0, y1 = min(bands, key=lambda b: abs((b[0] + b[1]) / 2 - ecy))
            for w in words:
                w[2] = [x0 + (x1 - x0) * w[3], y0, x0 + (x1 - x0) * w[4], y1]
        out.append([(w[0], w[2]) for w in words])
    return out


def ocr_pil(img):
    """Returns a list of lines, each a list of (word, [x0,y0,x1,y1])."""
    local = local_ocr(img)
    g = gemini_ocr(img)
    if not g and os.environ.get("GEMINI_API_KEY"):
        OCR_STATE.fallback = True
    if g:
        return snap_to_local(g, local, img) if local else [split_words(t, b) for t, b in g]
    return [split_words(t, b) for t, b in local]


def line_pieces(lines, pno):
    pieces = []
    for i, words in enumerate(lines):
        if i:
            pieces.append(("\n", None))
        for j, (w, box) in enumerate(words):
            if j:
                pieces.append((" ", None))
            pieces.append((w, (pno, box)))
    return pieces


def b64_png(pil_or_bytes):
    from PIL import Image
    if isinstance(pil_or_bytes, bytes):
        return base64.b64encode(pil_or_bytes).decode()
    buf = io.BytesIO()
    pil_or_bytes.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


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
                pieces.extend(line_pieces(ocr_pil(img), pno))
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
    return line_pieces(ocr_pil(img), 0), pages


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


OCR_STATE = threading.local()

PUNCT_NAMES = {
    ".": "titik", ",": "koma", ":": "titik dua", ";": "titik koma",
    "(": "kurung buka", ")": "kurung tutup", "\"": "tanda kutip",
    "\u201c": "tanda kutip", "\u201d": "tanda kutip", "'": "tanda petik",
    "-": "tanda hubung", "/": "garis miring", "?": "tanda tanya",
    "!": "tanda seru", "&": "tanda dan",
}


def describe_chars(chars):
    names = []
    for ch in chars:
        if ch.isspace():
            n = "enter" if ch == "\n" else "spasi"
        else:
            n = f"{PUNCT_NAMES.get(ch, 'tanda')} ({ch})"
        if n not in names:
            names.append(n)
    return ", ".join(names)


def explain(tag, a, b):
    """Plain-language description of one difference (Dok 1 = acuan)."""
    q = lambda t: "\u00ab" + t.strip().replace("\n", " ") + "\u00bb"
    simple = lambda t: all(not c.isalnum() for c in t)
    a, b = a.strip() or a, b.strip() or b
    if tag == "insert":
        if simple(b):
            return f"Dok 2 kelebihan {describe_chars(b)}, di Dok 1 tidak ada"
        return f"Dok 2 ada tambahan teks {q(b)} yang tidak ada di Dok 1"
    if tag == "delete":
        if simple(a):
            return f"Dok 2 kurang {describe_chars(a)} yang ada di Dok 1"
        return f"Teks {q(a)} ada di Dok 1 tapi hilang di Dok 2"
    if not a.strip() and not b.strip():
        return (f"Beda spasi/pindah baris: Dok 1 pakai {describe_chars(a)}, "
                f"Dok 2 pakai {describe_chars(b)}")
    if a.lower() == b.lower():
        return f"Beda huruf besar/kecil: Dok 1 {q(a)}, Dok 2 {q(b)}"
    gone, added, words = [], [], False
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if t == "equal":
            continue
        if not simple(a[i1:i2]) or not simple(b[j1:j2]):
            words = True
            break
        gone += a[i1:i2]
        added += b[j1:j2]
    if words:
        return f"Teks beda: Dok 1 {q(a)}, Dok 2 {q(b)}"
    parts = []
    if gone:
        parts.append(f"Dok 2 kurang {describe_chars(gone)}")
    if added:
        parts.append(f"Dok 2 kelebihan {describe_chars(added)}")
    return "Kata sama, tapi " + " dan ".join(parts) + f": Dok 1 {q(a)}, Dok 2 {q(b)}"


def normalize_text(t):
    return re.sub(r"\s+", " ", t).strip()


def tokenize(pieces):
    tokens, geom = [], []
    for text, box in pieces:
        for t in TOKEN_RE.findall(text):
            tokens.append(t)
            geom.append(box if not t.isspace() else None)
    return tokens, geom


HEADING_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*[.)])\s*(\S.*)$")


def token_points(tokens):
    """For each token, the numbered heading ("1. JUDUL") it falls under."""
    points, cur, i = [], "", 0
    for line in "".join(tokens).split("\n"):
        m = HEADING_RE.match(line)
        if m:
            cur = f"{m.group(1)} {m.group(2).strip()}"[:90]
        end = len(line) + 1
        while i < len(tokens) and end > 0:
            points.append(cur)
            end -= len(tokens[i])
            i += 1
    points += [cur] * (len(tokens) - len(points))
    return points


@app.post("/api/compare")
def compare():
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")
    if not f1 or not f2:
        return jsonify({"error": "Upload dua file dulu (Dokumen 1 = acuan, Dokumen 2 = yang dicek)."}), 400
    try:
        OCR_STATE.fallback = False
        pieces1, pages1, kind1 = extract(f1.filename, f1.read())
        fb1, OCR_STATE.fallback = OCR_STATE.fallback, False
        pieces2, pages2, kind2 = extract(f2.filename, f2.read())
        fb2 = OCR_STATE.fallback
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {e}"}), 500

    norm = request.form.get("normalize") == "1"
    t1, g1 = tokenize(pieces1)
    t2, g2 = tokenize(pieces2)
    p1 = token_points(t1)
    d1, d2 = t1, t2
    if norm:
        sp = lambda t, i: (" " if i and t[i - 1].isspace() else "") + t[i]
        d1 = [""] + [sp(t1, i) for i, t in enumerate(t1) if not t.isspace()]
        d2 = [""] + [sp(t2, i) for i, t in enumerate(t2) if not t.isspace()]
        keep = [i for i, t in enumerate(t1) if not t.isspace()]
        t1n = [t1[i] for i in keep]
        t1n.insert(0, " ")
        g1n = [g1[i] for i in keep]
        g1n.insert(0, None)
        p1 = [""] + [p1[i] for i in keep]
        keep2 = [i for i, t in enumerate(t2) if not t.isspace()]
        t2n = [t2[i] for i in keep2]
        t2n.insert(0, " ")
        g2n = [g2[i] for i in keep2]
        g2n.insert(0, None)
        t1, g1, t2, g2 = t1n, g1n, t2n, g2n
        text1 = "".join(d1).strip()
        text2 = "".join(d2).strip()
    else:
        text1 = "".join(t1)
        text2 = "".join(t2)

    # run diff, collecting page boxes per side
    sm = difflib.SequenceMatcher(a=t1, b=t2, autojunk=False)
    chunks, stats = [], {"equal": 0, "replace": 0, "insert": 0, "delete": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        stats[tag] += 1
        chunks.append({"tag": tag, "a": "".join(d1[i1:i2]), "b": "".join(d2[j1:j2])})
        if tag == "equal":
            continue
        chunks[-1]["point"] = p1[min(i1, len(p1) - 1)] if p1 else ""
        chunks[-1]["note"] = explain(tag, chunks[-1]["a"], chunks[-1]["b"])
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
        "identical": stats["changed_regions"] == 0,
        "pages1": pages1, "pages2": pages2,
        "ocr_fallback": [n for n, fb in ((1, fb1), (2, fb2)) if fb],
    })


@app.post("/api/export")
def export_xlsx():
    """Excel report: one defect per row, plus a summary sheet."""
    from flask import send_file
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    d = request.get_json(force=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Defect"
    ws.append(["No", "Poin", "Keterangan", "Teks di Dok 1 (acuan)", "Teks di Dok 2"])
    n = 0
    for c in d.get("chunks", []):
        if c["tag"] != "equal":
            n += 1
            ws.append([n, c.get("point", ""), c.get("note", ""),
                       c.get("a", "").strip(), c.get("b", "").strip()])
    for col, w in zip("ABCDE", (6, 36, 60, 40, 40)):
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="374151")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"

    sm = wb.create_sheet("Ringkasan")
    st = d.get("stats", {})
    for k, v in [
        ("Dokumen 1 (acuan)", d.get("file1", {}).get("name", "")),
        ("Dokumen 2", d.get("file2", {}).get("name", "")),
        ("Hasil", "IDENTIK" if d.get("identical") else "BERBEDA"),
        ("Jumlah defect", n),
        ("Diganti", st.get("replace", 0)),
        ("Ditambah di Dok 2", st.get("insert", 0)),
        ("Hilang di Dok 2", st.get("delete", 0)),
    ]:
        sm.append([k, v])
    sm.column_dimensions["A"].width = 24
    sm.column_dimensions["B"].width = 60
    for cell in sm["A"]:
        cell.font = Font(bold=True)

    name = re.sub(r'[\\/:*?"<>|]+', "_", (d.get("filename") or "").strip()) or "hasil-compare"
    if not name.lower().endswith(".xlsx"):
        name += ".xlsx"
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=name,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
