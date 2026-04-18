"""
Subprocess worker for CPU inference latency measurement.

Called by compare_models.py. Runs in a clean subprocess with a fixed thread
count so results are reproducible across machines. Prints a single JSON line
to stdout.

Usage (internal — do not call directly):
    python scripts/_latency_worker.py \\
        --checkpoint models/baseline_best.pt \\
        --model baseline \\
        --num-classes 38 \\
        --num-threads 1 \\
        --warmup 50 \\
        --runs 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from plant_disease import config
from plant_disease.models import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--num-classes", type=int, default=38)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    model = build_model(args.model, num_classes=args.num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.eval()

    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy)

    times_ms = []
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(dummy)
            times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    result = {
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "mean_ms":   float(np.mean(arr)),
        "runs":      args.runs,
        "warmup":    args.warmup,
        "num_threads": args.num_threads,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
