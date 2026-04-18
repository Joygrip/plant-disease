"""
Sanity-check the dataset: class counts, image shapes, sample display.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --split train
    python scripts/inspect_dataset.py --split valid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import config


def count_split(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        n = sum(
            1 for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        counts[class_dir.name] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both")
    args = parser.parse_args()

    splits_to_check = []
    if args.split in ("train", "both"):
        splits_to_check.append(("train", config.TRAIN_DIR))
    if args.split in ("valid", "both"):
        splits_to_check.append(("valid", config.VALID_DIR))

    for split_name, split_dir in splits_to_check:
        if not split_dir.exists():
            print(f"  {split_name}: directory not found at {split_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {split_name.upper()}  —  {split_dir}")
        print(f"{'='*60}")
        counts = count_split(split_dir)

        if not counts:
            print("  No class directories found.")
            continue

        total = sum(counts.values())
        known = set(config.CLASS_NAMES)
        for cls, n in counts.items():
            flag = "" if cls in known else "  ← UNKNOWN CLASS"
            print(f"  {cls:<55}  {n:>6}{flag}")

        missing = known - set(counts)
        if missing:
            print(f"\n  WARNING: {len(missing)} expected classes not found:")
            for m in sorted(missing):
                print(f"    - {m}")

        print(f"\n  Total images : {total:,}")
        print(f"  Total classes: {len(counts)}")


if __name__ == "__main__":
    main()
