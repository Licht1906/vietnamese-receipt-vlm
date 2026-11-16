"""Image helpers shared by inference and training."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: str | Path) -> Image.Image:
    """Load an image, apply EXIF rotation (phone photos) and convert to RGB."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side is at most max_side. Never upscales."""
    width, height = image.size
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def to_data_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 95) -> str:
    """Encode a PIL image as a base64 data URI for OpenAI-style chat messages."""
    buffer = io.BytesIO()
    save_kwargs = {"quality": quality} if fmt.upper() == "JPEG" else {}
    image.save(buffer, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"
