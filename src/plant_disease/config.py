"""Central configuration: paths, hyperparameters, class list."""

import os
import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # plant-disease/
DATA_ROOT = ROOT / "data" / "New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
SPLITS_FILE = ROOT / "data" / "splits.json"

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Image / normalisation
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # MobileNetV2-compatible
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Training defaults  (overridable from CLI)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
# Windows multiprocessing uses spawn (not fork) — 4 workers is reliable there.
# On Linux, scale with CPU count up to 8.
NUM_WORKERS = 4 if platform.system() == "Windows" else min(8, (os.cpu_count() or 4) // 2)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
SEED = 42

# ---------------------------------------------------------------------------
# Class names — sorted for stable index assignment
# 38 classes from vipoooool/new-plant-diseases-dataset
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)
assert NUM_CLASSES == 38, f"Expected 38 classes, got {NUM_CLASSES}"
