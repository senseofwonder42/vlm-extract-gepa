"""Scan-style augmentation: rotation, blur, contrast, noise, JPEG artefacts.

Each transform draws its strength from a seeded numpy generator, so the same
seed always yields the same bytes. `ScanParams` is stored in synthetic_meta
to make every augmented file explainable.

The final JPEG is encoded by PyMuPDF, not Pillow: Pillow 10.0.1's bundled
libjpeg-turbo produced different bytes (and slightly different decoded pixels)
for identical input across processes on Apple Silicon, which would change
`doc_id = sha256(file)` between regenerations. MuPDF's encoder was byte-stable
(verified 2026-10-17; see tests/test_synth.py::test_generation_is_identical_across_processes).
"""

from dataclasses import asdict, dataclass

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class ScanParams:
    """Augmentation applied to one page.

    Attributes:
        angle: Rotation in degrees (counter-clockwise); large values simulate
            pages scanned sideways or upside down.
        blur_radius: Gaussian blur radius in pixels.
        contrast: Contrast factor (1 = unchanged).
        brightness: Brightness factor (1 = unchanged).
        noise_sigma: Std-dev of additive Gaussian noise (0-255 scale).
        jpeg_quality: JPEG quality of the final encoding.
    """

    angle: float
    blur_radius: float
    contrast: float
    brightness: float
    noise_sigma: float
    jpeg_quality: int

    def as_dict(self) -> dict:
        """Return the parameters as a plain dict."""
        return asdict(self)


def sample_params(rng: np.random.Generator) -> ScanParams:
    """Draw augmentation parameters.

    About 8% of pages are rotated by 90°/180°/270°; the rest get a small skew.

    Args:
        rng: Seeded generator.

    Returns:
        The parameters.
    """
    if rng.random() < 0.08:
        angle = float(rng.choice([90.0, 180.0, 270.0]))
    else:
        angle = float(np.round(rng.uniform(-4.0, 4.0), 2))
    return ScanParams(
        angle=angle,
        blur_radius=float(np.round(rng.uniform(0.0, 1.3), 2)),
        contrast=float(np.round(rng.uniform(0.7, 1.25), 2)),
        brightness=float(np.round(rng.uniform(0.85, 1.1), 2)),
        noise_sigma=float(np.round(rng.uniform(0.0, 12.0), 1)),
        jpeg_quality=int(rng.integers(30, 86)),
    )


def augment(image: Image.Image, params: ScanParams, rng: np.random.Generator) -> bytes:
    """Apply scan-style degradation and return JPEG bytes.

    Args:
        image: Clean page image.
        params: Augmentation parameters.
        rng: Seeded generator (used for the noise pattern).

    Returns:
        JPEG-encoded degraded page.
    """
    page = image.convert("L" if rng.random() < 0.4 else "RGB")
    page = page.rotate(params.angle, expand=True, fillcolor="white" if page.mode == "RGB" else 255,
                       resample=Image.Resampling.BICUBIC)
    if params.blur_radius > 0:
        page = page.filter(ImageFilter.GaussianBlur(params.blur_radius))
    page = ImageEnhance.Contrast(page).enhance(params.contrast)
    page = ImageEnhance.Brightness(page).enhance(params.brightness)
    if params.noise_sigma > 0:
        pixels = np.asarray(page, dtype=np.float32)
        pixels = pixels + rng.normal(0.0, params.noise_sigma, size=pixels.shape)
        page = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    colorspace = pymupdf.csGRAY if page.mode == "L" else pymupdf.csRGB
    pixmap = pymupdf.Pixmap(colorspace, page.width, page.height, page.tobytes(), False)
    return pixmap.tobytes("jpeg", jpg_quality=params.jpeg_quality)
