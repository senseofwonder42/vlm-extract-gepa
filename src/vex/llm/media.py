"""Turn document files into images the VLM can read.

PDFs are rendered page by page with PyMuPDF at a configurable DPI (page
rotation flags in the PDF are honoured by the renderer). PNG and JPEG files
are sent as-is: re-encoding a scan would only add artefacts.
"""

import base64
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from loguru import logger

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@dataclass(frozen=True)
class PageImage:
    """One page image.

    Attributes:
        data: Encoded image bytes.
        mime: `image/png` or `image/jpeg`.
    """

    data: bytes
    mime: str

    def content_part(self) -> dict:
        """Return the OpenAI-style `image_url` content part with a data URL."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{encoded}"}}


def load_pages(path: Path, dpi: int, max_pages: int) -> list[PageImage]:
    """Load a document as a list of page images.

    Args:
        path: PDF, PNG or JPEG file.
        dpi: Rendering resolution for PDFs.
        max_pages: Keep at most this many pages (the first ones).

    Returns:
        Page images in document order.

    Raises:
        ValueError: If the file type is unsupported or the PDF has no pages.
    """
    suffix = path.suffix.lower()
    if suffix in _MIME_BY_SUFFIX:
        return [PageImage(data=path.read_bytes(), mime=_MIME_BY_SUFFIX[suffix])]
    if suffix != ".pdf":
        raise ValueError(f"Unsupported document type: {path.name}")

    with pymupdf.open(path) as pdf:
        if pdf.page_count == 0:
            raise ValueError(f"PDF has no pages: {path.name}")
        if pdf.page_count > max_pages:
            logger.warning("{}: {} pages, keeping the first {}", path.name, pdf.page_count, max_pages)
        return [
            PageImage(data=page.get_pixmap(dpi=dpi).tobytes("png"), mime="image/png")
            for page in list(pdf)[:max_pages]
        ]
