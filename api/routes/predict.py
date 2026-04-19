import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from api.config import settings
from api.exceptions import (
    FileTooLargeError,
    InvalidContentTypeError,
    ModelNotLoadedError,
    UnknownModelError,
)
from api.inference import format_display_name, run_inference

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_TOP_K = 10


@router.post("/predict", tags=["inference"])
async def predict(
    request: Request,
    image: Annotated[UploadFile, File(description="Leaf image (JPEG/PNG)")],
    top_k: Annotated[int, Query(ge=1, description="Number of top predictions (max 10)")] = 3,
    model: Annotated[Optional[str], Form(description="Model name: baseline or mobilenet_v2")] = None,
) -> JSONResponse:
    """Run plant disease classification on an uploaded leaf image."""
    models: dict = getattr(request.app.state, "models", {})
    class_names_map: dict = getattr(request.app.state, "class_names", {})

    if not models:
        raise ModelNotLoadedError()

    # Resolve requested model name
    requested = model or settings.default_model
    if requested not in models:
        if model is not None:
            raise UnknownModelError(requested)
        # default not available — fall back to first loaded
        requested = next(iter(models))

    # Validate content type
    content_type = image.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise InvalidContentTypeError(content_type)

    # Read and check file size
    image_bytes = await image.read()
    if len(image_bytes) > settings.max_upload_bytes:
        raise FileTooLargeError(settings.max_upload_mb)

    effective_top_k = min(top_k, _MAX_TOP_K)
    loaded_model = models[requested]
    class_names = class_names_map[requested]

    result = run_inference(image_bytes, loaded_model, class_names, top_k=effective_top_k)

    confidence = result["confidence"]
    logger.info(
        "inference model=%s class=%s confidence=%.4f ms=%.1f",
        requested,
        result["class_name"],
        confidence,
        result["inference_ms"],
    )

    top_k_entries = [
        {
            "class": entry["class"],
            "display": format_display_name(entry["class"]),
            "probability": entry["confidence"],
        }
        for entry in result["top_k"]
    ]

    return JSONResponse({
        "prediction": result["class_name"],
        "prediction_display": format_display_name(result["class_name"]),
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "top_k": top_k_entries,
        "inference_ms": result["inference_ms"],
        "model": requested,
        "image_size": result["image_size"],
    })
