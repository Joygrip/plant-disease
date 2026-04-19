"""API-level inference wrapper — validates image bytes, then delegates to plant_disease.inference."""

import io
import logging

import torch.nn as nn
from PIL import Image, UnidentifiedImageError

from plant_disease.inference import run_inference as _run_inference

from api.exceptions import ImageTooSmallError, InvalidImageError

logger = logging.getLogger(__name__)

_MIN_DIMENSION = 50
_WARN_DIMENSION = 4000


def format_display_name(raw: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato \u2014 Late blight'"""
    return raw.replace("___", " \u2014 ").replace("_", " ")


def run_inference(
    image_bytes: bytes,
    model: nn.Module,
    class_names: list[str],
    top_k: int = 3,
) -> dict:
    """
    Run inference from raw image bytes.

    Returns the standard inference dict plus image_size.
    Raises PlantAIException subclasses on bad input.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("PIL could not decode the bytes") from exc
    except Exception as exc:
        raise InvalidImageError(str(exc)) from exc

    width, height = pil_image.size
    if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
        raise ImageTooSmallError(width, height)
    if width > _WARN_DIMENSION or height > _WARN_DIMENSION:
        logger.warning("Large image %dx%d submitted — will be resized to 224x224", width, height)

    result = _run_inference(pil_image, model, class_names, top_k)
    result["image_size"] = {"width": width, "height": height}
    return result
