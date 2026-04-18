"""
Carve the validation folder into val/test splits (50/50, stratified, deterministic).

Writes data/splits.json — paths are relative to the valid/ directory so the
file is portable even if the dataset is re-downloaded to a different location.

Run once before training:
    python scripts/prepare_splits.py
"""

import json
import random
import sys
from pathlib import Path

# Make the src package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import config


def main(seed: int = config.SEED) -> None:
    if not config.VALID_DIR.exists():
        print(f"ERROR: valid/ directory not found at {config.VALID_DIR}")
        print("Download the dataset first and place it under data/")
        sys.exit(1)

    rng = random.Random(seed)

    val_paths: list[str] = []
    test_paths: list[str] = []

    class_dirs = sorted(d for d in config.VALID_DIR.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"ERROR: No class subdirectories found in {config.VALID_DIR}")
        sys.exit(1)

    total = 0
    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            print(f"  WARNING: No images in {class_dir.name}, skipping")
            continue

        rng.shuffle(images)
        mid = len(images) // 2
        val_imgs = images[:mid]
        test_imgs = images[mid:]

        for img in val_imgs:
            val_paths.append(str(img.relative_to(config.VALID_DIR)))
        for img in test_imgs:
            test_paths.append(str(img.relative_to(config.VALID_DIR)))

        total += len(images)
        print(f"  {class_dir.name}: {len(val_imgs)} val / {len(test_imgs)} test")

    splits = {"val": val_paths, "test": test_paths, "seed": seed}
    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\nDone. {len(val_paths)} val, {len(test_paths)} test ({total} total)")
    print(f"Splits written to {config.SPLITS_FILE}")


if __name__ == "__main__":
    main()
