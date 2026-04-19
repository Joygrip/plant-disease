from fastapi import APIRouter

from plant_disease import config
from api.inference import format_display_name

router = APIRouter()

# Computed once at import time — immutable, cache-friendly.
_CLASSES = [
    {"raw": name, "display": format_display_name(name)}
    for name in config.CLASS_NAMES
]


@router.get("/classes", tags=["inference"])
async def get_classes() -> dict:
    """Return all 38 recognisable plant disease classes with human-readable display names."""
    return {"classes": _CLASSES}
