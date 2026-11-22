# Kompare Dokumen

Aplikasi QA untuk membandingkan dua dokumen secara ketat — Dokumen 1 adalah acuan,
Dokumen 2 harus sama persis (teks + tanda baca).

## Input (3 cara per dokumen)
- Drag & drop
- Klik untuk pilih file
- Ctrl+V paste screenshot (di-OCR) atau teks

## Format
PDF (termasuk scan → OCR), Word .docx, image (PNG/JPG/WebP/BMP/TIFF → OCR), TXT/CSV/HTML.
Boleh campur antar format.

## Jalankan
```bash
pip install -r requirements.txt
python3 app.py   # http://localhost:8000
```

## Fitur
- Diff token-level (kata + tanda baca + spasi), highlight merah/hijau/kuning
- Toggle "Abaikan beda spasi/enter" untuk perbandingan lintas format
- Lompat antar-perbedaan, view side-by-side atau list perbedaan
