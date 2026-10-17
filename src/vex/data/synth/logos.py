"""Procedurally drawn logos for the fictional insurers.

Logos are plain geometric marks with no text (the insurer name is printed
next to them by the template), so they cannot resemble a real brand's
wordmark and need no font file.
"""

import io

from PIL import Image, ImageDraw

_SIZE = 160


def draw_logo(shape: str, color: str) -> bytes:
    """Draw a logo as PNG bytes.

    Args:
        shape: `circle`, `square`, `triangle`, `diamond` or `shield`.
        color: Fill colour (hex).

    Returns:
        PNG-encoded image.

    Raises:
        ValueError: On an unknown shape.
    """
    image = Image.new("RGB", (_SIZE, _SIZE), "white")
    draw = ImageDraw.Draw(image)
    m, s = 10, _SIZE - 10
    if shape == "circle":
        draw.ellipse((m, m, s, s), fill=color)
        draw.ellipse((m + 35, m + 35, s - 35, s - 35), fill="white")
    elif shape == "square":
        draw.rectangle((m, m, s, s), fill=color)
        draw.rectangle((m + 30, m + 30, s - 30, s - 30), outline="white", width=12)
    elif shape == "triangle":
        draw.polygon([(_SIZE // 2, m), (s, s), (m, s)], fill=color)
    elif shape == "diamond":
        draw.polygon([(_SIZE // 2, m), (s, _SIZE // 2), (_SIZE // 2, s), (m, _SIZE // 2)], fill=color)
    elif shape == "shield":
        draw.polygon([(m, m), (s, m), (s, _SIZE // 2), (_SIZE // 2, s), (m, _SIZE // 2)], fill=color)
        draw.line([(_SIZE // 2, m + 15), (_SIZE // 2, s - 25)], fill="white", width=10)
    else:
        raise ValueError(f"Unknown logo shape: {shape}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
