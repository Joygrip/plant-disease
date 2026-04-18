"""
Generate a side-by-side comparison report for the thesis.

Usage:
    python scripts/compare_models.py \\
        models/baseline_best.pt \\
        models/mobilenet_v2_best.pt

    python scripts/compare_models.py \\
        models/baseline_best.pt \\
        models/mobilenet_v2_best.pt \\
        --cpu-threads 1 \\
        --latency-warmup 50 \\
        --latency-runs 1000

Output: reports/baseline_vs_mobilenet.md  (overwritten on each run)
Requires: dataset + splits.json (same as evaluate.py)
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import config
from plant_disease.data import get_dataloaders
from plant_disease.evaluate import compute_top_confused_pairs, run_evaluation
from plant_disease.models import build_model
from plant_disease.utils import count_parameters, get_device, seed_everything


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
WORKER = Path(__file__).parent / "_latency_worker.py"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path, device: torch.device):
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")
    if not meta_path.exists():
        print(f"WARNING: no meta file for {ckpt_path.name} — inferring model name")
        model_name = "baseline" if "baseline" in ckpt_path.stem else "mobilenet_v2"
        class_names = config.CLASS_NAMES
        meta: dict = {}
    else:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        model_name = meta["model"]
        class_names = meta.get("class_names", config.CLASS_NAMES)

    model = build_model(model_name, num_classes=len(class_names))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    return model, meta, model_name, class_names


# ---------------------------------------------------------------------------
# Latency benchmark — runs in a subprocess with fixed thread count
# ---------------------------------------------------------------------------

def _measure_latency(
    ckpt_path: Path,
    model_name: str,
    num_classes: int,
    *,
    num_threads: int,
    warmup: int,
    runs: int,
) -> dict:
    """
    Run the latency worker in a subprocess and return the stats dict.

    Subprocess isolation ensures torch.set_num_threads() doesn't bleed into
    the main process and results are reproducible across machines.
    """
    cmd = [
        sys.executable, str(WORKER),
        "--checkpoint", str(ckpt_path),
        "--model", model_name,
        "--num-classes", str(num_classes),
        "--num-threads", str(num_threads),
        "--warmup", str(warmup),
        "--runs", str(runs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Latency worker failed for {model_name}:\n{result.stderr}"
        )
    return json.loads(result.stdout.strip())


# ---------------------------------------------------------------------------
# Per-class F1 helpers (importable)
# ---------------------------------------------------------------------------

def _per_class_f1(labels, preds, class_names: list[str]) -> dict[str, float]:
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    return {cls: report[cls]["f1-score"] for cls in class_names}


def _macro_f1(labels, preds, class_names: list[str]) -> float:
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    return report["macro avg"]["f1-score"]


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _build_report(
    names: list[str],
    metas: list[dict],
    total_params: list[int],
    test_accs: list[float],
    macro_f1s: list[float],
    latency_stats: list[dict],
    per_class_f1s: list[dict[str, float]],
    confused_pairs: list[list[tuple[str, str, int]]],
    class_names: list[str],
    machine_info: str,
) -> str:
    lines: list[str] = []
    lines.append("# Baseline CNN vs MobileNetV2 — Comparison Report\n")
    lines.append("*Generated automatically by `scripts/compare_models.py`*\n")
    lines.append(f"**Machine:** {machine_info}\n")

    # Summary table
    lines.append("## Summary\n")
    header = "| Metric | " + " | ".join(names) + " |"
    sep    = "|--------|" + "|".join(["--------"] * len(names)) + "|"
    lines.append(header)
    lines.append(sep)

    def row(label: str, values: list) -> str:
        return "| " + label + " | " + " | ".join(str(v) for v in values) + " |"

    lines.append(row("Total parameters", [f"{p:,}" for p in total_params]))
    lines.append(row("Test accuracy", [f"{a:.4f} ({a*100:.2f}%)" for a in test_accs]))
    lines.append(row("Macro F1", [f"{f:.4f}" for f in macro_f1s]))
    lines.append(row(
        "CPU latency — median (ms)",
        [f"{s['median_ms']:.2f}" for s in latency_stats],
    ))
    lines.append(row(
        "CPU latency — p95 (ms)",
        [f"{s['p95_ms']:.2f}" for s in latency_stats],
    ))
    lines.append(row(
        "CPU latency — mean (ms)",
        [f"{s['mean_ms']:.2f}" for s in latency_stats],
    ))

    for i, (name, meta) in enumerate(zip(names, metas)):
        if meta:
            best_epoch = meta.get("epoch", "—")
            best_stage = meta.get("best_stage", "—")
            val_acc    = meta.get("val_acc", 0.0)
            lines.append(row(
                f"Best epoch ({name})",
                ["—" if j != i else f"{best_epoch} (stage {best_stage})"
                 for j in range(len(names))],
            ))
            lines.append(row(
                f"Best val acc ({name})",
                ["—" if j != i else f"{val_acc:.4f}" for j in range(len(names))],
            ))

    lines.append("")

    # Per-class F1 table
    lines.append("## Per-class F1 (sorted by baseline F1 ascending)\n")
    lines.append("Δ = MobileNetV2 F1 − Baseline F1\n")
    col_headers = ["Class"] + [f"F1 ({n})" for n in names]
    if len(names) == 2:
        col_headers.append("Δ")
    lines.append("| " + " | ".join(col_headers) + " |")
    lines.append("|" + "|".join(["--------"] * len(col_headers)) + "|")

    baseline_f1 = per_class_f1s[0]
    for cls in sorted(class_names, key=lambda c: baseline_f1.get(c, 0.0)):
        f1_vals = [f"{d.get(cls, 0.0):.4f}" for d in per_class_f1s]
        row_parts = [cls] + f1_vals
        if len(names) == 2:
            delta = per_class_f1s[1].get(cls, 0.0) - per_class_f1s[0].get(cls, 0.0)
            sign = "+" if delta >= 0 else ""
            row_parts.append(f"{sign}{delta:.4f}")
        lines.append("| " + " | ".join(row_parts) + " |")

    lines.append("")

    # Top confused pairs
    lines.append("## Top 10 Confused Class Pairs\n")
    for name, pairs in zip(names, confused_pairs):
        lines.append(f"### {name}\n")
        lines.append("| Rank | Count | True class | Predicted as |")
        lines.append("|------|-------|-----------|-------------|")
        for rank, (true_cls, pred_cls, count) in enumerate(pairs, 1):
            lines.append(f"| {rank} | {count} | {true_cls} | {pred_cls} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two model checkpoints")
    p.add_argument("checkpoint_a", type=Path, help="First checkpoint (baseline)")
    p.add_argument("checkpoint_b", type=Path, help="Second checkpoint (MobileNetV2)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="CPU threads for latency worker (default 1 for reproducibility)")
    p.add_argument("--latency-warmup", type=int, default=50,
                   help="Warmup passes before timing (discarded)")
    p.add_argument("--latency-runs", type=int, default=1000,
                   help="Timed passes for latency measurement")
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

    machine_info = (
        f"{platform.processor()} · "
        f"{torch.get_num_threads()} logical cores · "
        f"torch {torch.__version__}"
    )
    print(f"Machine: {machine_info}")

    checkpoints = [args.checkpoint_a, args.checkpoint_b]
    models, metas, names, class_name_lists = [], [], [], []
    for ckpt in checkpoints:
        model, meta, name, class_names = _load_checkpoint(ckpt, device)
        models.append(model)
        metas.append(meta)
        names.append(name)
        class_name_lists.append(class_names)
        print(f"Loaded {name} from {ckpt.name}")

    class_names = class_name_lists[0]
    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, augment=False,
    )

    total_params, test_accs, macro_f1s = [], [], []
    per_class_f1s, confused_pairs, latency_stats = [], [], []

    for ckpt, model, name in zip(checkpoints, models, names):
        print(f"\n=== Evaluating {name} ===")
        total_params.append(count_parameters(model))

        labels, preds = run_evaluation(model, test_loader, device)
        acc = sum(l == p for l, p in zip(labels, preds)) / len(labels)
        test_accs.append(acc)
        macro_f1s.append(_macro_f1(labels, preds, class_names))
        per_class_f1s.append(_per_class_f1(labels, preds, class_names))
        confused_pairs.append(compute_top_confused_pairs(labels, preds, class_names, top_k=10))
        print(f"  Test accuracy : {acc:.4f}   Macro F1: {macro_f1s[-1]:.4f}")

        print(
            f"  Measuring CPU latency "
            f"(warmup={args.latency_warmup}, runs={args.latency_runs}, "
            f"threads={args.cpu_threads})…"
        )
        stats = _measure_latency(
            ckpt, name, len(class_names),
            num_threads=args.cpu_threads,
            warmup=args.latency_warmup,
            runs=args.latency_runs,
        )
        latency_stats.append(stats)
        print(
            f"  Latency median={stats['median_ms']:.2f} ms  "
            f"p95={stats['p95_ms']:.2f} ms  "
            f"mean={stats['mean_ms']:.2f} ms"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "baseline_vs_mobilenet.md"
    report = _build_report(
        names=names,
        metas=metas,
        total_params=total_params,
        test_accs=test_accs,
        macro_f1s=macro_f1s,
        latency_stats=latency_stats,
        per_class_f1s=per_class_f1s,
        confused_pairs=confused_pairs,
        class_names=class_names,
        machine_info=machine_info,
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written → {report_path}")

    # Also write raw latency stats as JSON for generate_results_doc.py to read
    stats_path = REPORTS_DIR / "latency_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(zip(names, latency_stats)), f, indent=2)

    # Write per-model summary.json so generate_results_doc.py finds test accuracy
    # without requiring a separate evaluate.py run.
    outputs_root = Path(__file__).resolve().parents[1] / "outputs" / "eval"
    for name, acc, ckpt in zip(names, test_accs, checkpoints):
        eval_dir = outputs_root / name
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary = {"checkpoint": str(ckpt), "test_accuracy": acc}
        (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
