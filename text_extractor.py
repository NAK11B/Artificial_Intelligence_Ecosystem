# text_extractor.py
"""
Extract text from a local PDF and write it to Selected_Document.txt (UTF-8).
- Tries PyPDF2 first, falls back to pdfminer.six if needed.
- Collapses extra whitespace.
- Prints success/failure and returns full text when run directly.
"""

import re
import sys
from pathlib import Path
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text

OUTPUT_PATH = Path("Selected_Document.txt")
# Use your actual filename (placed in repo root)
DEFAULT_PDF_PATH = Path("wuhan_military_games.pdf")

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_with_pypdf2(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t:
            pages.append(t)
    return "\n\n".join(pages)

def _extract_with_pdfminer(pdf_path: Path) -> str:
    try:
        return pdfminer_extract_text(str(pdf_path)) or ""
    except Exception:
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    p = Path(pdf_path)
    if not p.exists():
        print(f"[ERROR] PDF not found: {p.resolve()}")
        return ""

    text = _extract_with_pypdf2(p)
    if len(text.strip()) < 20:
        fallback = _extract_with_pdfminer(p)
        if len(fallback.strip()) > len(text.strip()):
            text = fallback

    # Clean & write
    lines = [_collapse_ws(x) for x in text.splitlines()]
    cleaned = "\n\n".join([ln for ln in lines if ln])

    try:
        OUTPUT_PATH.write_text(cleaned, encoding="utf-8")
        if cleaned:
            print(f"[OK] Wrote {len(cleaned):,} chars to {OUTPUT_PATH.resolve()}")
        else:
            print("[WARN] Extracted text is empty; file written but blank.")
    except Exception as e:
        print(f"[ERROR] Could not write {OUTPUT_PATH.name}: {e}")
        return ""

    return cleaned

def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_PDF_PATH)
    print(f"[INFO] Using PDF: {Path(pdf).resolve()}")
    text = extract_text_from_pdf(pdf)
    print(f"[INFO] Extracted chars: {len(text)}")
    print(f"[INFO] Wrote: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
