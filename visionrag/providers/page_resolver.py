from __future__ import annotations

from io import BytesIO

from PIL import Image

from visionrag.providers.s3_client import S3Client


class PageResolver:
    def __init__(self, s3_client: S3Client, dpi: int = 200) -> None:
        self._s3_client = s3_client
        self._dpi = dpi

    def fetch_pdf(self, bucket: str, key: str) -> bytes:
        return self._s3_client.fetch_pdf_bytes(bucket=bucket, key=key)

    def render_pages(self, pdf_bytes: bytes, page_numbers: list[int]) -> dict[int, Image.Image]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for rendering pages.") from exc

        rendered: dict[int, Image.Image] = {}
        zoom = self._dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_number in sorted(set(page_numbers)):
                if page_number < 1 or page_number > len(doc):
                    continue
                page = doc[page_number - 1]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                rendered[page_number] = image

        return rendered

