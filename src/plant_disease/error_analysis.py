"""
Thesis-grade error analysis for a trained checkpoint.

Usage:
    python -m plant_disease.error_analysis models/mobilenet_v2_best.pt
    python -m plant_disease.error_analysis models/baseline_best.pt --output-dir reports/error_analysis/baseline/

Outputs (all in --output-dir):
    confusion_pairs.json          — structured top-10 confused-pair analysis
    misclass_<pair>.png           — 6-image gallery for each of top-5 pairs
    confidence_histogram.png      — correct vs incorrect prediction confidence
    per_species.csv               — per-species accuracy rollup
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from plant_disease import config
from plant_disease.data import get_dataloaders
from plant_disease.models import build_model
from plant_disease.utils import get_device, seed_everything

IMAGENET_MEAN = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD  = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Inference — collect labels, preds, confidences
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """
    Return (labels, preds, max_softmax_confidence) for the full loader.
    Order matches loader.dataset.samples.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds:  list[int] = []
    all_confs:  list[float] = []

    for imgs, labels in tqdm(loader, desc="Inference", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_labels, all_preds, all_confs


# ---------------------------------------------------------------------------
# Species helpers
# ---------------------------------------------------------------------------

def _species(class_name: str) -> str:
    return class_name.split("___")[0]


def _same_species(a: str, b: str) -> bool:
    return _species(a) == _species(b)


# ---------------------------------------------------------------------------
# a. Confusion-pair deep dive
# ---------------------------------------------------------------------------

def analyze_confused_pairs(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Return a list of dicts for the top_k most confused pairs (bidirectional total).

    Each dict has:
        true_class, pred_class, count_ab, count_ba, total,
        symmetric (bool), same_species (bool)
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    n = len(class_names)

    # Collect all (A,B) pairs with A < B and at least one direction non-zero
    seen: set[tuple[int, int]] = set()
    pairs = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            cnt_ab = int(cm[a, b])
            cnt_ba = int(cm[b, a])
            total = cnt_ab + cnt_ba
            if total == 0:
                continue
            pairs.append((total, a, b, cnt_ab, cnt_ba))

    pairs.sort(reverse=True)

    result = []
    for total, a, b, cnt_ab, cnt_ba in pairs[:top_k]:
        hi, lo = max(cnt_ab, cnt_ba), min(cnt_ab, cnt_ba)
        symmetric = (lo / hi >= 0.5) if hi > 0 else True
        result.append({
            "true_class":   class_names[a],
            "pred_class":   class_names[b],
            "count_ab":     cnt_ab,
            "count_ba":     cnt_ba,
            "total":        total,
            "symmetric":    symmetric,
            "same_species": _same_species(class_names[a], class_names[b]),
        })
    return result


# ---------------------------------------------------------------------------
# b. Misclassification gallery
# ---------------------------------------------------------------------------

def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Inverse ImageNet normalization → numpy HWC uint8."""
    img = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _load_image_tensor(path: Path) -> torch.Tensor:
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return tf(Image.open(path).convert("RGB"))


def save_misclass_gallery(
    pair: dict,
    labels: list[int],
    preds: list[int],
    confs: list[float],
    class_names: list[str],
    dataset_samples: list[tuple[Path, int]],
    output_dir: Path,
    n_images: int = 6,
) -> None:
    cls_a = class_names.index(pair["true_class"])
    cls_b = class_names.index(pair["pred_class"])

    # Indices where true=A, pred=B (A→B confusions)
    indices = [
        i for i, (l, p) in enumerate(zip(labels, preds))
        if l == cls_a and p == cls_b
    ][:n_images]

    if not indices:
        return

    cols = min(3, len(indices))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for ax in axes.flatten():
        ax.axis("off")

    for pos, idx in enumerate(indices):
        r, c = divmod(pos, cols)
        img_path = dataset_samples[idx][0]
        try:
            tensor = _load_image_tensor(img_path)
            img_np = _denorm(tensor)
        except Exception:
            continue
        ax = axes[r, c]
        ax.imshow(img_np)
        ax.set_title(
            f"Pred: {pair['pred_class'].split('___')[-1]}\nConf: {confs[idx]:.2f}",
            fontsize=7,
        )
        ax.axis("off")

    pair_slug = f"{pair['true_class']}__vs__{pair['pred_class']}".replace(" ", "_")[:80]
    fig.suptitle(
        f"True: {pair['true_class']}\nMisclassified as: {pair['pred_class']}",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = output_dir / f"misclass_{pair_slug}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# c. Confidence histogram
# ---------------------------------------------------------------------------

def save_confidence_histogram(
    labels: list[int],
    preds: list[int],
    confs: list[float],
    output_dir: Path,
) -> dict:
    correct_confs = [c for l, p, c in zip(labels, preds, confs) if l == p]
    wrong_confs   = [c for l, p, c in zip(labels, preds, confs) if l != p]

    mean_correct = float(np.mean(correct_confs)) if correct_confs else 0.0
    mean_wrong   = float(np.mean(wrong_confs))   if wrong_confs   else 0.0
    high_conf_wrong = sum(1 for c in wrong_confs if c > 0.9)
    frac_high_conf_wrong = high_conf_wrong / len(wrong_confs) if wrong_confs else 0.0

    print(f"  Mean confidence (correct): {mean_correct:.4f}")
    print(f"  Mean confidence (wrong):   {mean_wrong:.4f}")
    print(
        f"  Wrong & conf>0.9: {high_conf_wrong}/{len(wrong_confs)} "
        f"({frac_high_conf_wrong*100:.1f}%) — dangerous errors"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(correct_confs, bins=bins, alpha=0.6, label=f"Correct (n={len(correct_confs):,})",
            color="steelblue", density=True)
    ax.hist(wrong_confs,   bins=bins, alpha=0.6, label=f"Wrong   (n={len(wrong_confs):,})",
            color="tomato",    density=True)
    ax.axvline(0.9, color="red", linestyle="--", linewidth=1, label="conf=0.9 threshold")
    ax.set_xlabel("Max softmax confidence")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence: Correct vs Incorrect")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)

    return {
        "mean_confidence_correct": mean_correct,
        "mean_confidence_wrong":   mean_wrong,
        "n_wrong_high_confidence": high_conf_wrong,
        "frac_wrong_high_confidence": frac_high_conf_wrong,
    }


# ---------------------------------------------------------------------------
# d. Per-species rollup
# ---------------------------------------------------------------------------

def save_per_species_rollup(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
    output_dir: Path,
) -> list[dict]:
    # Map each class index to its species
    species_of = [_species(cn) for cn in class_names]
    all_species = sorted(set(species_of))

    rows: list[dict] = []
    for sp in all_species:
        sp_class_indices = [i for i, s in enumerate(species_of) if s == sp]
        # Samples belonging to this species
        sp_mask = [l in sp_class_indices for l in labels]
        sp_labels = [l for l, m in zip(labels, sp_mask) if m]
        sp_preds  = [p for p, m in zip(preds,  sp_mask) if m]
        n_images  = len(sp_labels)

        if n_images == 0:
            continue

        # Species-level accuracy: predicted species == true species
        sp_correct = [species_of[p] == sp for p in sp_preds]
        species_acc = sum(sp_correct) / n_images

        # Disease-within-species accuracy: exact class match | predicted species == true species
        within = [(l == p) for l, p, sc in zip(sp_labels, sp_preds, sp_correct) if sc]
        disease_acc = sum(within) / len(within) if within else float("nan")

        rows.append({
            "species":          sp,
            "num_classes":      len(sp_class_indices),
            "num_test_images":  n_images,
            "species_accuracy": round(species_acc, 4),
            "disease_accuracy": round(disease_acc, 4) if not math.isnan(disease_acc) else "",
        })

    out_path = output_dir / "per_species.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["species", "num_classes", "num_test_images",
                        "species_accuracy", "disease_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Per-species rollup → {out_path}")
    return rows


# ---------------------------------------------------------------------------
# Main analysis function (importable for testing and generate_results_doc.py)
# ---------------------------------------------------------------------------

def run_error_analysis(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    output_dir: Path,
    model_name: str = "model",
) -> dict:
    """
    Run the full error analysis suite.  Returns a summary dict.
    output_dir is created if it doesn't exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    print("\n[1/4] Running inference on test set…")
    labels, preds, confs = _run_inference(model, test_loader, device)

    print("\n[2/4] Confusion-pair deep dive…")
    pairs = analyze_confused_pairs(labels, preds, class_names, top_k=10)

    # Print summary table
    print(f"\n  {'Pair':<70}  {'A→B':>5}  {'B→A':>5}  {'Sym':>5}  {'Same sp':>7}")
    print("  " + "-" * 100)
    for p in pairs:
        sym_flag = "Y" if p["symmetric"]    else "N"
        sp_flag  = "Y" if p["same_species"] else "N"
        label = f"{p['true_class']} → {p['pred_class']}"
        print(f"  {label:<70}  {p['count_ab']:>5}  {p['count_ba']:>5}  {sym_flag:>5}  {sp_flag:>7}")

    confusion_pairs_path = output_dir / "confusion_pairs.json"
    with open(confusion_pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    print("\n[3/4] Misclassification galleries (top 5 pairs)…")
    dataset_samples = [(Path(s[0]), s[1]) for s in test_loader.dataset.samples]
    for pair in pairs[:5]:
        save_misclass_gallery(
            pair, labels, preds, confs, class_names, dataset_samples, output_dir,
        )
        slug = f"{pair['true_class']} → {pair['pred_class']}"
        print(f"  Gallery: {slug}")

    print("\n[4/4] Confidence histogram & per-species rollup…")
    conf_stats = save_confidence_histogram(labels, preds, confs, output_dir)
    species_rows = save_per_species_rollup(labels, preds, class_names, output_dir)

    # Same-species confusion summary
    n_same_sp  = sum(1 for p in pairs if p["same_species"])
    n_cross_sp = len(pairs) - n_same_sp
    print(f"\n  Top-10 confused pairs: {n_same_sp} within-species, {n_cross_sp} cross-species")

    summary = {
        "model": model_name,
        "num_test_samples": len(labels),
        "overall_accuracy": sum(l == p for l, p in zip(labels, preds)) / len(labels),
        "top_confused_pairs": pairs,
        "same_species_confusion_pairs": n_same_sp,
        "cross_species_confusion_pairs": n_cross_sp,
        "confidence_stats": conf_stats,
        "per_species_rows": species_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error analysis for a trained checkpoint")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--model", choices=["baseline", "mobilenet_v2"], default=None,
                   help="Model architecture (inferred from meta if omitted)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    meta_path = args.checkpoint.with_name(args.checkpoint.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_name  = meta["model"]
        class_names = meta.get("class_names", config.CLASS_NAMES)
    else:
        model_name  = args.model or ("baseline" if "baseline" in args.checkpoint.stem else "mobilenet_v2")
        class_names = config.CLASS_NAMES

    if args.model:
        model_name = args.model

    output_dir = args.output_dir or (
        Path("reports") / "error_analysis" / model_name
    )

    model = build_model(model_name, num_classes=len(class_names))
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)

    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    summary = run_error_analysis(model, test_loader, class_names, output_dir, model_name)

    print(f"\nOverall accuracy: {summary['overall_accuracy']:.4f}")
    print(f"All outputs written to {output_dir}/")


if __name__ == "__main__":
    main()
