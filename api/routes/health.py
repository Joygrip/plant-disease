import datetime
from pathlib import Path

import torch
from fastapi import APIRouter

from api.config import settings

router = APIRouter()


@router.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness check — does not require the model to be loaded."""
    models_dir = settings.models_dir
    ckpt = models_dir / "mobilenet_v2_best.pt"

    if ckpt.exists():
        mtime = datetime.datetime.fromtimestamp(
            ckpt.stat().st_mtime, tz=datetime.timezone.utc
        )
        age = str(datetime.datetime.now(tz=datetime.timezone.utc) - mtime).split(".")[0]
    else:
        age = "unknown"

    return {
        "status": "ok",
        "model": settings.default_model,
        "checkpoint_age": age,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
