from __future__ import annotations

from pathlib import Path


def check_pymupdf(pdf_path: Path) -> None:
    import fitz

    with fitz.open(pdf_path) as doc:
        print(f"Opened {pdf_path.name} with {len(doc)} pages.")


if __name__ == "__main__":
    sample = Path("sample.pdf")
    if not sample.exists():
        print("sample.pdf not found.")
    else:
        check_pymupdf(sample)
