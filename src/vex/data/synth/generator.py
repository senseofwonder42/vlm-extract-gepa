"""Render synthetic certificates: Jinja2 HTML → PDF (PyMuPDF Story) → optional scan.

Renderer choice (docs/PLAN.md §2.10): PyMuPDF's `Story` renders an HTML/CSS
subset to PDF. It is already a dependency, needs no system library or browser,
and is byte-deterministic once the PDF metadata is pinned. Its CSS support is
limited (no flex/grid, table columns need pixel widths), which form-like
certificates can live with.

Output mix per document (seeded):
- clean digital PDFs (like a Mémo downloaded from a customer area);
- scans (`synth.scan_fraction`, default 65%): each page rasterized, degraded by `augment`, saved as JPEG
  (one page) or as a PDF of JPEG pages (two pages).

Every file is listed in `synthetic_meta` with its generator values. Those
values are for tests and sanity checks only, never labels.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import jinja2
import markupsafe
import numpy as np
import pymupdf
from loguru import logger
from PIL import Image

from vex.data.synth.augment import augment, sample_params
from vex.data.synth.logos import draw_logo
from vex.data.synth.values import SampledDocument, sample_document

PAGE_BREAK = "<!--PAGEBREAK-->"
_TEMPLATES = jinja2.Environment(
    loader=jinja2.PackageLoader("vex.data.synth", "templates"),
    autoescape=True,
    undefined=jinja2.StrictUndefined,
)
# Values that must never wrap across lines (a plate split at its hyphen is
# not something real certificates do, and it breaks the meta sanity checks).
_TEMPLATES.filters["nobr"] = lambda value: markupsafe.Markup(
    f'<span style="white-space: nowrap">{markupsafe.escape(value)}</span>'
)
_FIXED_METADATA = {
    "producer": "vex-synth",
    "creator": "vex-synth",
    "creationDate": "D:20250101000000Z",
    "modDate": "D:20250101000000Z",
}
_MARGIN = 36


@dataclass(frozen=True)
class GeneratedFile:
    """One generated document file.

    Attributes:
        path: Written file.
        meta: Line written to synthetic_meta.
    """

    path: Path
    meta: dict


def render_pdf(sample: SampledDocument) -> bytes:
    """Render a sampled document to deterministic PDF bytes.

    Args:
        sample: Sampled document.

    Returns:
        PDF bytes; identical inputs give identical bytes.
    """
    html = _TEMPLATES.get_template(f"{sample.layout}.html.j2").render(**sample.display)
    archive = pymupdf.Archive()
    archive.add(draw_logo(sample.display["insurer"].shape, sample.display["insurer"].color), "logo.png")
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page_rect = pymupdf.paper_rect("a4")
    content_rect = page_rect + (_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)
    for chunk in html.split(PAGE_BREAK):
        story = pymupdf.Story(html=chunk, archive=archive)
        more = True
        while more:
            device = writer.begin_page(page_rect)
            more, _ = story.place(content_rect)
            story.draw(device)
            writer.end_page()
    writer.close()
    with pymupdf.open("pdf", buffer.getvalue()) as pdf:
        pdf.set_metadata(_FIXED_METADATA)
        return pdf.tobytes(garbage=3, deflate=True, no_new_id=True)


def _scan(pdf_bytes: bytes, rng: np.random.Generator) -> tuple[bytes, str, list[dict]]:
    dpi = int(rng.integers(130, 201))
    pages, params_used = [], []
    with pymupdf.open("pdf", pdf_bytes) as pdf:
        for page in pdf:
            image = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png")))
            params = sample_params(rng)
            pages.append(augment(image, params, rng))
            params_used.append({"dpi": dpi, **params.as_dict()})
    if len(pages) == 1:
        return pages[0], ".jpg", params_used
    with pymupdf.open() as out:
        for jpeg in pages:
            with pymupdf.open(stream=jpeg, filetype="jpeg") as img_doc:
                rect = img_doc[0].rect
                page = out.new_page(width=rect.width, height=rect.height)
                page.insert_image(rect, stream=jpeg)
        out.set_metadata(_FIXED_METADATA)
        return out.tobytes(garbage=3, deflate=True, no_new_id=True), ".pdf", params_used


def generate_one(index: int, seed: int, out_dir: Path, scan_fraction: float = 0.65) -> GeneratedFile:
    """Generate document `index` and write it to `out_dir`.

    Args:
        index: Document index.
        seed: Generator seed.
        out_dir: Destination directory (`data/raw/synthetic`).
        scan_fraction: Probability that the document is degraded as a scan.

    Returns:
        The written file and its meta line.
    """
    sample = sample_document(index, seed)
    rng = np.random.default_rng([seed, index])
    pdf_bytes = render_pdf(sample)
    if rng.random() >= scan_fraction:
        data, suffix, scan = pdf_bytes, ".pdf", None
    else:
        data, suffix, scan = _scan(pdf_bytes, rng)
    path = out_dir / f"syn_{index:04d}_{sample.layout}{suffix}"
    path.write_bytes(data)
    meta = {
        "file": path.name,
        "layout": sample.layout,
        "seed": seed,
        "index": index,
        "pages": sample.pages,
        "n_vehicles": len(sample.display["vehicles"]),
        "scan": scan,
        "values": sample.meta,
    }
    return GeneratedFile(path=path, meta=meta)


def generate(
    n: int, seed: int, out_dir: Path, meta_path: Path, scan_fraction: float = 0.65, overwrite: bool = False
) -> list[GeneratedFile]:
    """Generate `n` synthetic documents and their synthetic_meta file.

    Args:
        n: Number of documents.
        seed: Generator seed.
        out_dir: Destination directory for the documents.
        meta_path: synthetic_meta JSONL file (rewritten).
        scan_fraction: Probability that a document is degraded as a scan.
        overwrite: Delete existing `syn_*` files first; otherwise refuse to
            write into a directory that already holds some.

    Returns:
        The generated files.

    Raises:
        FileExistsError: If `out_dir` already holds synthetic files and
            `overwrite` is False.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("syn_*"))
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already holds {len(existing)} synthetic files; pass --overwrite")
    for path in existing:
        path.unlink()
    files = [generate_one(index, seed, out_dir, scan_fraction) for index in range(n)]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(f.meta, ensure_ascii=False) for f in files]
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Generated {} synthetic documents in {} (meta: {})", n, out_dir, meta_path)
    return files
