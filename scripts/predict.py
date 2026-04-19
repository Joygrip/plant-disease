"""
Single-image inference tool.

Usage:
    python scripts/predict.py models/baseline_best.pt path/to/leaf.jpg
    python scripts/predict.py models/baseline_best.pt path/to/leaf.jpg --top-k 5
    python scripts/predict.py models/baseline_best.pt path/to/leaf.jpg --gpu
    python scripts/predict.py models/baseline_best.pt path/to/leaf.jpg --json

Exit codes:
    0  success
    1  bad input (image missing, not an image, checkpoint missing)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

from plant_disease.inference import load_checkpoint as _lib_load_checkpoint
from plant_disease.inference import run_inference


# ---------------------------------------------------------------------------
# Model loading (CLI wrapper — translates exceptions to sys.exit(1))
# ---------------------------------------------------------------------------

def _load_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    """
    Return (model, class_names) from a checkpoint, printing errors to stderr
    and exiting with code 1 on failure.
    """
    try:
        return _lib_load_checkpoint(
            checkpoint,
            device,
            warn_callback=lambda msg: print(f"WARNING: {msg}", file=sys.stderr),
        )
    except FileNotFoundError:
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: cannot load checkpoint {checkpoint}: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core inference (importable for tests and future API use)
# ---------------------------------------------------------------------------

def predict(
    image_path: Path,
    checkpoint: Path,
    device: torch.device,
    top_k: int = 3,
) -> dict:
    """
    Run inference on one image and return a structured result.

    Return value::

        {
            "class_name":   str,    # top-1 predicted class
            "confidence":   float,  # top-1 softmax probability (0–1)
            "top_k": [
                {"class": str, "confidence": float},
                ...                 # length == top_k, sorted descending
            ],
            "inference_ms": float,  # wall-clock time for the forward pass
        }

    Calls sys.exit(1) on bad input so the CLI exits cleanly.
    """
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        print(f"ERROR: not a recognised image file: {image_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: cannot open {image_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    model, class_names = _load_checkpoint(checkpoint, device)
    return run_inference(img, model, class_names, top_k)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_result(result: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"Prediction : {result['class_name']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")
    print(f"Inference  : {result['inference_ms']:.1f} ms")
    print()
    k = len(result["top_k"])
    print(f"Top-{k}:")
    for rank, entry in enumerate(result["top_k"], 1):
        print(f"  {rank}. {entry['class']:<55}  {entry['confidence'] * 100:5.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict plant disease from a single leaf image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/predict.py models/baseline_best.pt leaf.jpg\n"
            "  python scripts/predict.py models/baseline_best.pt leaf.jpg --top-k 5\n"
            "  python scripts/predict.py models/baseline_best.pt leaf.jpg --json\n"
        ),
    )
    p.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file")
    p.add_argument("image",      type=Path, help="Path to leaf image (JPEG / PNG)")
    p.add_argument(
        "--top-k", type=int, default=3, metavar="K",
        help="Number of top predictions to show (default: 3)",
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use GPU for inference (default: CPU)",
    )
    p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print raw JSON instead of formatted text",
    )
    return p.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

    args = parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print(
                "WARNING: --gpu requested but CUDA not available, using CPU",
                file=sys.stderr,
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    result = predict(args.image, args.checkpoint, device, top_k=args.top_k)
    _print_result(result, as_json=args.as_json)


if __name__ == "__main__":
    main()
