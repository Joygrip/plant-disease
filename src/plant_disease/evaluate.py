"""
Evaluate a saved checkpoint on the test split.

Usage:
    python -m plant_disease.evaluate --checkpoint models/baseline_best.pt
    python -m plant_disease.evaluate --checkpoint models/baseline_best.pt --output-dir outputs/eval
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from plant_disease import config
from plant_disease.data import get_dataloaders
from plant_disease.models import build_model
from plant_disease.utils import get_device, get_logger, seed_everything


# ---------------------------------------------------------------------------
# Core evaluation (model-agnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return (all_labels, all_preds) for the full dataloader."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def compute_top_confused_pairs(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
    top_k: int = 10,
) -> list[tuple[str, str, int]]:
    """Return top-k (true_class, pred_class, count) off-diagonal pairs."""
    cm = confusion_matrix(labels, preds)
    np.fill_diagonal(cm, 0)  # zero diagonal, keep misclassifications only
    flat = cm.flatten()
    top_indices = flat.argsort()[::-1][:top_k]
    pairs = []
    n = len(class_names)
    for idx in top_indices:
        true_cls = idx // n
        pred_cls = idx % n
        count = flat[idx]
        if count == 0:
            break
        pairs.append((class_names[true_cls], class_names[pred_cls], int(count)))
    return pairs


def save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds)
    # Normalise per row (true class) so colours reflect error rate, not class size
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        cm_norm,
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="lightgray",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a plant disease classifier checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt state dict")
    p.add_argument("--meta", type=Path, default=None, help="Path to _meta.json (auto-inferred if omitted)")
    p.add_argument("--model", choices=["baseline", "mobilenet_v2"], default="baseline")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("evaluate")

    # Load metadata to get class names (fall back to config if no meta file)
    meta_path = args.meta or args.checkpoint.with_name(
        args.checkpoint.stem + "_meta.json"
    )
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        class_names: list[str] = meta["class_names"]
    else:
        logger.warning("No meta file found — using class names from config")
        class_names = config.CLASS_NAMES

    # Build and load model
    if args.model == "baseline":
        model = build_model(num_classes=len(class_names))
    else:
        raise ValueError(f"Unknown model: {args.model}")

    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    labels, preds = run_evaluation(model, test_loader, device)

    # --- Overall accuracy ---
    accuracy = sum(l == p for l, p in zip(labels, preds)) / len(labels)
    logger.info(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # --- Per-class report ---
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    report_path = args.output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Classification report → {report_path}")
    print(report)

    # --- Confusion matrix PNG ---
    cm_path = args.output_dir / "confusion_matrix.png"
    save_confusion_matrix(labels, preds, class_names, cm_path)
    logger.info(f"Confusion matrix → {cm_path}")

    # --- Top-10 confused pairs ---
    pairs = compute_top_confused_pairs(labels, preds, class_names, top_k=10)
    confused_path = args.output_dir / "top_confused_pairs.txt"
    lines = ["Top-10 most confused class pairs\n", "=" * 60 + "\n"]
    for rank, (true_cls, pred_cls, count) in enumerate(pairs, 1):
        lines.append(f"  {rank:2d}. {count:4d}×  {true_cls}  →  {pred_cls}\n")
    confused_path.write_text("".join(lines), encoding="utf-8")
    logger.info(f"Confused pairs → {confused_path}")
    print("".join(lines))

    # --- Summary JSON ---
    summary = {
        "checkpoint": str(args.checkpoint),
        "test_accuracy": accuracy,
        "num_test_samples": len(labels),
        "top_confused_pairs": [
            {"true": t, "pred": p, "count": c} for t, p, c in pairs
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
