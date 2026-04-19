import logging
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.exceptions import InternalError, PlantAIException, plant_ai_exception_handler
from api.routes.classes import router as classes_router
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from plant_disease.inference import load_checkpoint

    app.state.models = {}
    app.state.class_names = {}

    device = torch.device("cpu")
    models_dir = settings.models_dir

    checkpoints = {
        "mobilenet_v2": models_dir / "mobilenet_v2_best.pt",
        "baseline": models_dir / "baseline_best.pt",
    }

    for model_name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found, skipping: %s", ckpt_path)
            continue
        try:
            model, class_names = load_checkpoint(
                ckpt_path,
                device,
                warn_callback=lambda msg, n=model_name: logger.warning("%s: %s", n, msg),
            )
            app.state.models[model_name] = model
            app.state.class_names[model_name] = class_names
            logger.info("Loaded model '%s' from %s", model_name, ckpt_path)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", model_name, exc, exc_info=True)

    if not app.state.models:
        raise RuntimeError(
            f"No model checkpoints found in {models_dir} — cannot start API"
        )

    logger.info("Startup complete. Loaded models: %s", list(app.state.models.keys()))
    yield
    # shutdown: nothing to clean up for PyTorch CPU models


app = FastAPI(
    title="Plant Disease Classification API",
    version="0.1.0",
    description="Classify plant leaf diseases from images using CNN and MobileNetV2 models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.add_exception_handler(PlantAIException, plant_ai_exception_handler)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    err = InternalError()
    return JSONResponse(
        status_code=500,
        content={"error": {"code": err.code, "message": err.message, "detail": ""}},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s %s %d %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(classes_router)
