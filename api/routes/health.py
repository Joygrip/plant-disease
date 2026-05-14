import datetime
import time

import torch
from fastapi import APIRouter, Request

from api.config import settings

router = APIRouter()


@router.get("/health", tags=["ops"])
async def health(request: Request) -> dict:
    """Liveness check — reports loaded models and uptime."""
    models_dir = settings.models_dir
    ckpt = models_dir / "mobilenet_v2_best.pt"

    if ckpt.exists():
        mtime = datetime.datetime.fromtimestamp(
            ckpt.stat().st_mtime, tz=datetime.timezone.utc
        )
        age = str(datetime.datetime.now(tz=datetime.timezone.utc) - mtime).split(".")[0]
    else:
        age = "unknown"

    loaded_models = list(getattr(request.app.state, "models", {}).keys())
    startup_time = getattr(request.app.state, "startup_time", None)
    uptime = round(time.time() - startup_time, 1) if startup_time else 0.0

    status = "ok" if loaded_models else "starting"

    return {
        "status": status,
        "model": settings.default_model,
        "models_loaded": loaded_models,
        "default_model": settings.default_model,
        "checkpoint_age": age,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "uptime_seconds": uptime,
    }
