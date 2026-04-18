"""
Plot training curves from one or more metrics CSV files.

Usage:
    # Single model
    python scripts/plot_training_curves.py models/baseline_metrics.csv

    # Compare two models (overlay)
    python scripts/plot_training_curves.py \\
        models/baseline_metrics.csv \\
        models/mobilenet_v2_metrics.csv

Outputs:
    reports/training_curves/<model_name>.png   (one per CSV)
    reports/training_curves/comparison.png     (when multiple CSVs given)
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "training_curves"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> dict[str, list]:
    """Return dict of column_name → list of values (numeric where possible)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    data: dict[str, list] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                data[k].append(float(v))
            except (ValueError, TypeError):
                data[k].append(v)
    return data


def _stage_transitions(data: dict) -> list[int]:
    """Return epoch numbers where the 'stage' column changes from '1' to '2'."""
    stages = data.get("stage", [])
    transitions = []
    for i in range(1, len(stages)):
        if str(stages[i - 1]) == "1" and str(stages[i]) == "2":
            epochs = data.get("epoch", [])
            transitions.append(int(epochs[i]) if epochs else i + 1)
    return transitions


def _model_label(csv_path: Path) -> str:
    """Derive a short display label from the filename."""
    stem = csv_path.stem  # e.g. baseline_metrics → baseline
    return stem.replace("_metrics", "").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Single-model plot
# ---------------------------------------------------------------------------

def plot_single(data: dict, label: str, transitions: list[int], out_path: Path) -> None:
    epochs      = np.array(data["epoch"])
    train_loss  = np.array(data["train_loss"])
    val_loss    = np.array(data["val_loss"])
    val_acc     = np.array(data["val_acc"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"Training Curves — {label}", fontsize=13)

    # Loss plot
    ax1.plot(epochs, train_loss, label="Train loss", color="steelblue")
    ax1.plot(epochs, val_loss,   label="Val loss",   color="tomato")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, val_acc * 100, label="Val accuracy", color="seagreen")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Stage transition lines
    for tr_epoch in transitions:
        for ax in (ax1, ax2):
            ax.axvline(
                tr_epoch - 0.5, color="purple", linestyle="--",
                linewidth=1.2, label="Stage 1→2" if ax is ax1 else None,
            )
        ax1.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Multi-model comparison overlay
# ---------------------------------------------------------------------------

def plot_comparison(
    datasets: list[dict],
    labels: list[str],
    transitions_per_model: list[list[int]],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Training Curves — Model Comparison", fontsize=13)

    # Colour cycle: 2 colours per model (loss pair)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0

    for i, (data, label) in enumerate(zip(datasets, labels)):
        epochs     = np.array(data["epoch"])
        train_loss = np.array(data["train_loss"])
        val_loss   = np.array(data["val_loss"])
        val_acc    = np.array(data["val_acc"])

        c_train = colours[ci % len(colours)]
        c_val   = colours[(ci + 1) % len(colours)]
        ci += 2

        ax1.plot(epochs, train_loss, color=c_train, linestyle="--",  alpha=0.7,
                 label=f"{label} train")
        ax1.plot(epochs, val_loss,   color=c_val,   linestyle="-",
                 label=f"{label} val")
        ax2.plot(epochs, val_acc * 100, color=c_val, label=label)

        # Stage transition for this model
        for tr_epoch in transitions_per_model[i]:
            for ax in (ax1, ax2):
                ax.axvline(
                    tr_epoch - 0.5, color=c_val, linestyle=":",
                    linewidth=1.0, alpha=0.8,
                )

    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves from metrics CSV files")
    p.add_argument("csvs", type=Path, nargs="+", help="One or more metrics CSV files")
    p.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    return p.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

    args = parse_args()

    datasets, labels, transitions_list = [], [], []

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping")
            continue
        data = _load_csv(csv_path)
        label = _model_label(csv_path)
        transitions = _stage_transitions(data)
        datasets.append(data)
        labels.append(label)
        transitions_list.append(transitions)

        out_path = args.output_dir / f"{csv_path.stem.replace('_metrics', '')}.png"
        print(f"Plotting {label}…")
        plot_single(data, label, transitions, out_path)

    if len(datasets) >= 2:
        comp_path = args.output_dir / "comparison.png"
        print("Plotting comparison…")
        plot_comparison(datasets, labels, transitions_list, comp_path)


if __name__ == "__main__":
    main()
