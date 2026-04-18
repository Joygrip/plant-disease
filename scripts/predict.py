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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

from plant_disease import config
from plant_disease.data import _eval_transform
from plant_disease.models import build_model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    """
    Return (model, class_names) from a checkpoint and its sidecar _meta.json.

    If no metadata file exists, model type is inferred from the filename and
    class names fall back to config.CLASS_NAMES.
    """
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    meta_path = checkpoint.with_name(checkpoint.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_name  = meta["model"]
        class_names = meta.get("class_names", config.CLASS_NAMES)
    else:
        model_name  = "baseline" if "baseline" in checkpoint.stem else "mobilenet_v2"
        class_names = config.CLASS_NAMES
        print(
            f"WARNING: no metadata at {meta_path} — "
            f"inferring model type as '{model_name}'",
            file=sys.stderr,
        )

    model = build_model(model_name, num_classes=len(class_names))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, class_names


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

    tf     = _eval_transform()
    tensor = tf(img).unsqueeze(0).to(device)          # (1, 3, 224, 224)

    model, class_names = _load_checkpoint(checkpoint, device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    probs   = torch.softmax(logits, dim=1)[0]         # (num_classes,)
    k       = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(k)

    top_list = [
        {"class": class_names[idx.item()], "confidence": prob.item()}
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "class_name":   top_list[0]["class"],
        "confidence":   top_list[0]["confidence"],
        "top_k":        top_list,
        "inference_ms": round(inference_ms, 2),
    }


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
