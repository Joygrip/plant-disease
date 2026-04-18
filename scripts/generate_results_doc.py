"""
Generate reports/ml_results.md by assembling existing run artifacts.

Must be run AFTER:
    1. train.py       → models/<model>_best_meta.json, models/<model>_run_info.json,
                         models/<model>_metrics.csv
    2. evaluate.py    → outputs/eval/<model>/summary.json,
                         outputs/eval/<model>/classification_report.txt
    3. compare_models.py → reports/baseline_vs_mobilenet.md,
                            reports/latency_stats.json
    4. error_analysis → reports/error_analysis/mobilenet_v2/summary.json,
                         reports/error_analysis/mobilenet_v2/per_species.csv
    5. plot_training_curves.py → reports/training_curves/*.png

Idempotent — safe to re-run, always overwrites reports/ml_results.md.

Usage:
    python scripts/generate_results_doc.py \\
        models/baseline_best.pt \\
        models/mobilenet_v2_best.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import config

ROOT        = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR  = ROOT / "models"


# ---------------------------------------------------------------------------
# Artifact readers — return [TBD] gracefully when file is absent
# ---------------------------------------------------------------------------

TBD = "*[TO BE FILLED AFTER TRAINING]*"


def _read_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else TBD


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(value, fmt=".4f", fallback=TBD):
    if not value and value != 0:
        return fallback
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Dataset statistics (from splits.json)
# ---------------------------------------------------------------------------

def _dataset_stats() -> dict:
    splits_path = ROOT / "data" / "splits.json"
    if not splits_path.exists():
        return {"train": TBD, "val": TBD, "test": TBD}
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    return {
        "val":  len(splits["val"]),
        "test": len(splits["test"]),
        "train": TBD,  # train size requires scanning disk
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_setup(dataset_stats: dict) -> str:
    lines = ["## Experimental Setup\n"]
    lines.append("### Dataset")
    lines.append("- Source: `vipoooool/new-plant-diseases-dataset` (Kaggle)")
    lines.append("- 38 classes (species × disease combinations)")
    val  = dataset_stats['val']
    test = dataset_stats['test']
    val_str  = f"{val:,}"  if isinstance(val,  (int, float)) else str(val)
    test_str = f"{test:,}" if isinstance(test, (int, float)) else str(test)
    lines.append(f"- Train: ~70,000 images · Val: {val_str} · Test: {test_str}")
    lines.append("- Split: `valid/` split 50/50 stratified by class (seed=42) into val + test\n")

    lines.append("### Preprocessing")
    lines.append(f"- Resize to {config.IMAGE_SIZE}×{config.IMAGE_SIZE} px")
    lines.append(f"- Normalize with ImageNet stats:")
    lines.append(f"  - mean = {config.IMAGENET_MEAN}")
    lines.append(f"  - std  = {config.IMAGENET_STD}\n")

    lines.append("### Augmentation (train only)")
    lines.append("- `RandomHorizontalFlip()`")
    lines.append("- `RandomRotation(15°)`")
    lines.append("- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)`\n")

    lines.append("### Framework")
    lines.append("- PyTorch · torchvision · scikit-learn")
    lines.append("- Mixed precision training (`torch.amp`)")
    lines.append("- Optimizer: AdamW · Schedule: cosine LR · Early stopping (patience=5)\n")
    return "\n".join(lines)


def _section_model(
    model_name: str,
    ckpt_path: Path,
    eval_output_dir: Path,
) -> str:
    meta     = _read_json(MODELS_DIR / f"{model_name}_best_meta.json")
    run_info = _read_json(MODELS_DIR / f"{model_name}_run_info.json")
    eval_sum = _read_json(eval_output_dir / "summary.json")
    cls_rpt  = _read_text(eval_output_dir / "classification_report.txt")

    display = "Baseline CNN" if model_name == "baseline" else "MobileNetV2"
    lines = [f"## {display}\n"]

    # Architecture
    lines.append("### Architecture")
    if model_name == "baseline":
        lines.append("4 × ConvBlock (Conv2d → BN → ReLU → MaxPool2d)")
        lines.append("→ Global Average Pooling → Linear(256, 38)")
        lines.append("~1.2 M parameters")
    else:
        lines.append("MobileNetV2 backbone (IMAGENET1K_V2 weights)")
        lines.append("→ Dropout(0.2) → Linear(1280, 38)")
        lines.append("~2.27 M parameters")
        lines.append("Two-stage fine-tuning: 5 epochs frozen backbone + 25 epochs top-3 layers")
    lines.append("")

    # Hardware & timing
    lines.append("### Training Hardware & Timing")
    gpu   = run_info.get("gpu_name", TBD)
    torch_v = run_info.get("torch_version", TBD)
    py_v  = run_info.get("python_version", TBD)
    t_s   = run_info.get("total_training_time_s")
    t_min = f"{t_s/60:.1f} min" if t_s else TBD
    ep_t  = run_info.get("mean_epoch_time_s", TBD)
    ep_r  = run_info.get("epochs_run", TBD)
    lines.append(f"- GPU: {gpu}")
    lines.append(f"- torch {torch_v} · Python {py_v}")
    lines.append(f"- Total training time: {t_min}")
    lines.append(f"- Mean epoch time: {ep_t} s · Epochs run: {ep_r}")
    git = run_info.get("git_commit", TBD)
    lines.append(f"- Git commit: `{git}`\n")

    # Best checkpoint
    best_ep    = meta.get("epoch", TBD)
    best_stage = meta.get("best_stage", TBD)
    best_val   = _fmt(meta.get("val_acc"), ".4f")
    lines.append("### Best Checkpoint")
    lines.append(f"- Best epoch: {best_ep} (stage {best_stage})")
    lines.append(f"- Val accuracy: {best_val}\n")

    # Test metrics
    test_acc = _fmt(eval_sum.get("test_accuracy"), ".4f")
    lines.append("### Test Metrics")
    lines.append(f"- Test accuracy: {test_acc}\n")

    # Training curves
    curves_img = REPORTS_DIR / "training_curves" / f"{model_name}.png"
    rel_path = curves_img.relative_to(REPORTS_DIR) if curves_img.exists() else None
    if rel_path:
        lines.append(f"![Training curves]({rel_path})\n")
    else:
        lines.append(f"*Training curves: run `plot_training_curves.py` to generate*\n")

    # Classification report
    lines.append("### Per-class Classification Report\n")
    lines.append("```")
    lines.append(cls_rpt if cls_rpt != TBD else TBD)
    lines.append("```\n")

    return "\n".join(lines)


def _section_comparison() -> str:
    cmp_path = REPORTS_DIR / "baseline_vs_mobilenet.md"
    if cmp_path.exists():
        # Embed the comparison report as-is (it's already well-structured markdown)
        content = _read_text(cmp_path)
        return f"## Model Comparison\n\n*From `scripts/compare_models.py`*\n\n{content}\n"
    return "## Model Comparison\n\n" + TBD + "\n"


def _section_error_analysis(model_name: str = "mobilenet_v2") -> str:
    ea_dir = REPORTS_DIR / "error_analysis" / model_name
    summary = _read_json(ea_dir / "summary.json")
    species_rows = _read_csv_rows(ea_dir / "per_species.csv")
    pairs = summary.get("top_confused_pairs", [])

    display = "Baseline CNN" if model_name == "baseline" else "MobileNetV2"
    lines = [f"## Error Analysis ({display})\n"]

    # Confusion-pair breakdown
    lines.append("### Top Confused Class Pairs\n")
    if pairs:
        same_sp  = sum(1 for p in pairs if p.get("same_species"))
        cross_sp = len(pairs) - same_sp
        lines.append(
            f"Of the top 10 confused pairs: **{same_sp} within-species** "
            f"(harder; model confuses diseases on the same plant) and "
            f"**{cross_sp} cross-species** (potentially more concerning).\n"
        )
        lines.append("| # | True class | Predicted as | A→B | B→A | Symmetric | Same species |")
        lines.append("|---|-----------|-------------|-----|-----|-----------|-------------|")
        for i, p in enumerate(pairs, 1):
            sym  = "Y" if p.get("symmetric")    else "N"
            sp   = "Y" if p.get("same_species") else "N"
            lines.append(
                f"| {i} | {p['true_class']} | {p['pred_class']} "
                f"| {p['count_ab']} | {p['count_ba']} | {sym} | {sp} |"
            )
        lines.append("")
    else:
        lines.append(TBD + "\n")

    # Misclassification galleries
    lines.append("### Misclassification Galleries\n")
    gallery_files = sorted(ea_dir.glob("misclass_*.png")) if ea_dir.exists() else []
    if gallery_files:
        for f in gallery_files:
            rel = f.relative_to(REPORTS_DIR)
            lines.append(f"![{f.stem}]({rel})\n")
    else:
        lines.append(TBD + "\n")

    # Confidence histogram
    lines.append("### Confidence Calibration\n")
    hist_path = ea_dir / "confidence_histogram.png"
    if hist_path.exists():
        rel = hist_path.relative_to(REPORTS_DIR)
        lines.append(f"![Confidence histogram]({rel})\n")
        cs = summary.get("confidence_stats", {})
        lines.append(
            f"- Mean confidence when **correct**: {_fmt(cs.get('mean_confidence_correct'))}\n"
            f"- Mean confidence when **wrong**:   {_fmt(cs.get('mean_confidence_wrong'))}\n"
            f"- Wrong predictions with conf > 0.9 (dangerous errors): "
            f"{cs.get('n_wrong_high_confidence', TBD)} "
            f"({_fmt(cs.get('frac_wrong_high_confidence', 0), '.1%')})\n"
        )
    else:
        lines.append(TBD + "\n")

    # Per-species table
    lines.append("### Per-species Accuracy Rollup\n")
    if species_rows:
        lines.append("| Species | # Classes | # Test images | Species acc | Disease acc |")
        lines.append("|---------|-----------|--------------|-------------|-------------|")
        for row in species_rows:
            lines.append(
                f"| {row['species']} | {row['num_classes']} | {row['num_test_images']} "
                f"| {row['species_accuracy']} | {row.get('disease_accuracy', '')} |"
            )
        lines.append("")
    else:
        lines.append(TBD + "\n")

    return "\n".join(lines)


def _section_limitations(model_name: str = "mobilenet_v2") -> str:
    ea_dir  = REPORTS_DIR / "error_analysis" / model_name
    summary = _read_json(ea_dir / "summary.json")
    pairs   = summary.get("top_confused_pairs", [])
    cs      = summary.get("confidence_stats", {})

    lines = ["## Limitations\n"]
    lines.append(
        "- **Lab-condition dataset**: all images are from controlled conditions. "
        "Real-world field photos have background clutter, variable lighting, and partial "
        "occlusion. Expect a generalization gap.\n"
    )

    if pairs:
        n_cross = sum(1 for p in pairs if not p.get("same_species"))
        if n_cross > 0:
            lines.append(
                f"- **Cross-species confusions**: {n_cross} of the top-10 confused pairs "
                f"involve different plant species — these should be rare and indicate the "
                f"model still makes gross errors in some cases.\n"
            )

    frac = cs.get("frac_wrong_high_confidence")
    if frac:
        lines.append(
            f"- **Overconfident errors**: {frac*100:.1f}% of wrong predictions have "
            f"softmax confidence > 0.9. These are the most dangerous failures for an "
            f"end-user application — the model is wrong but appears certain.\n"
        )

    lines.append(
        "- **No temporal or geographic diversity**: the dataset does not cover "
        "seasonal variation or regional disease strains.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level document assembly
# ---------------------------------------------------------------------------

def generate_results_doc(
    baseline_ckpt: Path,
    mobilenet_ckpt: Path,
    baseline_eval_dir: Path,
    mobilenet_eval_dir: Path,
) -> str:
    dataset_stats = _dataset_stats()

    parts = [
        "# ML Results\n",
        "_Generated by `scripts/generate_results_doc.py`. Re-run to update._\n",
        _section_setup(dataset_stats),
        _section_model("baseline",     baseline_ckpt,  baseline_eval_dir),
        _section_model("mobilenet_v2", mobilenet_ckpt, mobilenet_eval_dir),
        _section_comparison(),
        _section_error_analysis("mobilenet_v2"),
        _section_limitations("mobilenet_v2"),
    ]
    return "\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble ml_results.md from run artifacts")
    p.add_argument("baseline_checkpoint",    type=Path)
    p.add_argument("mobilenet_checkpoint",   type=Path)
    p.add_argument("--baseline-eval-dir",    type=Path,
                   default=ROOT / "outputs" / "eval" / "baseline")
    p.add_argument("--mobilenet-eval-dir",   type=Path,
                   default=ROOT / "outputs" / "eval" / "mobilenet_v2")
    p.add_argument("--output",               type=Path,
                   default=REPORTS_DIR / "ml_results.md")
    return p.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    doc = generate_results_doc(
        baseline_ckpt=args.baseline_checkpoint,
        mobilenet_ckpt=args.mobilenet_checkpoint,
        baseline_eval_dir=args.baseline_eval_dir,
        mobilenet_eval_dir=args.mobilenet_eval_dir,
    )
    args.output.write_text(doc, encoding="utf-8")
    print(f"Results document written → {args.output}")


if __name__ == "__main__":
    main()
