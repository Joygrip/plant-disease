"""
Pre-flight validation — run before committing to full training.

Catches data-pipeline, model, normalization, and gradient-flow bugs.

Exit codes:
    0 — all checks passed
    1 — warnings present, no failures
    2 — at least one check failed

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --batch-size 32 --num-workers 0
    python scripts/preflight.py --allow-cpu   # skip CUDA requirement (CI / Mac)
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torchvision import transforms

from plant_disease import config
from plant_disease.data import _eval_transform, _train_transform, get_dataloaders
from plant_disease.models import build_model

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "preflight"

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

_warnings: list[str] = []
_failures: list[str] = []


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    _warnings.append(msg)


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    _failures.append(msg)


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Check: CUDA availability
# ---------------------------------------------------------------------------

def check_cuda(allow_cpu: bool) -> None:
    _section("CUDA / device")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1024 ** 3
        total_gb = total_bytes / 1024 ** 3
        _pass(f"CUDA available: {device_name} (CUDA {cuda_ver})")
        _pass(f"Free VRAM: {free_gb:.1f} GB / {total_gb:.1f} GB")
        if free_gb < 4.0:
            _warn(
                f"Free VRAM {free_gb:.1f} GB < 4 GB — training may OOM with "
                f"default batch size. Consider reducing --batch-size."
            )
    else:
        msg = (
            "CUDA not available. Either install the CUDA build of torch "
            "(see README Troubleshooting) or pass --allow-cpu to proceed "
            "with CPU-only training (will be very slow)."
        )
        if allow_cpu:
            _warn("CUDA not available — running CPU-only (--allow-cpu passed)")
        else:
            _fail(msg)


# ---------------------------------------------------------------------------
# Check: ImageNet normalization constants
# ---------------------------------------------------------------------------

def check_normalization_constants() -> None:
    _section("Normalization constants")
    expected_mean = (0.485, 0.456, 0.406)
    expected_std  = (0.229, 0.224, 0.225)

    for name, tf_fn in [("train", _train_transform), ("eval", _eval_transform)]:
        tf = tf_fn()
        norm_layers = [t for t in tf.transforms if isinstance(t, transforms.Normalize)]
        if not norm_layers:
            _fail(f"{name} transform has no Normalize layer")
            continue
        norm = norm_layers[0]
        mean_ok = all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(norm.mean, expected_mean))
        std_ok  = all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(norm.std,  expected_std))
        if mean_ok and std_ok:
            _pass(f"{name} transform: mean={tuple(norm.mean)}, std={tuple(norm.std)}")
        else:
            _fail(
                f"{name} transform normalization mismatch\n"
                f"    mean got={tuple(norm.mean)} expected={expected_mean}\n"
                f"    std  got={tuple(norm.std)}  expected={expected_std}"
            )

    # Verify config constants match
    if config.IMAGENET_MEAN == expected_mean and config.IMAGENET_STD == expected_std:
        _pass(f"config constants: mean={config.IMAGENET_MEAN}, std={config.IMAGENET_STD}")
    else:
        _fail(f"config.IMAGENET_MEAN/STD do not match expected ImageNet values")


# ---------------------------------------------------------------------------
# Check: data loading (skipped if dataset absent)
# ---------------------------------------------------------------------------

def check_data_loading(batch_size: int, num_workers: int) -> dict | None:
    _section("Data loading")

    if not config.TRAIN_DIR.exists():
        _warn(f"Dataset not found at {config.TRAIN_DIR} — skipping data checks")
        return None

    if not config.SPLITS_FILE.exists():
        _warn(f"splits.json not found — run prepare_splits.py first. Skipping data checks.")
        return None

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size, num_workers=num_workers, augment=True
        )
    except Exception as e:
        _fail(f"get_dataloaders() raised: {e}")
        return None

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    first_batches = {}

    for split_name, loader in loaders.items():
        try:
            imgs, labels = next(iter(loader))
        except Exception as e:
            _fail(f"{split_name} loader: failed to get first batch: {e}")
            continue

        # Shape check
        B, C, H, W = imgs.shape
        if C == 3 and H == config.IMAGE_SIZE and W == config.IMAGE_SIZE:
            _pass(f"{split_name}: shape {tuple(imgs.shape)}, dtype={imgs.dtype}")
        else:
            _fail(f"{split_name}: unexpected shape {tuple(imgs.shape)} — expected (B,3,{config.IMAGE_SIZE},{config.IMAGE_SIZE})")

        # dtype
        if imgs.dtype != torch.float32:
            _fail(f"{split_name}: expected float32, got {imgs.dtype}")

        # Value range — after ImageNet normalization, roughly [-3, 3], definitely not 0–255
        vmin, vmax = imgs.min().item(), imgs.max().item()
        if vmin >= -4.0 and vmax <= 4.0:
            _pass(f"{split_name}: value range [{vmin:.2f}, {vmax:.2f}] (normalized ✓)")
        else:
            _fail(f"{split_name}: value range [{vmin:.2f}, {vmax:.2f}] — looks un-normalized (0–255 range?)")

        # Label range
        lmin, lmax = labels.min().item(), labels.max().item()
        if 0 <= lmin and lmax <= 37:
            _pass(f"{split_name}: label range [{lmin}, {lmax}] ⊆ [0, 37] ✓")
        else:
            _fail(f"{split_name}: label range [{lmin}, {lmax}] — expected [0, 37]")

        first_batches[split_name] = (imgs, labels)

    _pass(f"train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}  test={len(test_loader.dataset):,} images")
    return first_batches


# ---------------------------------------------------------------------------
# Check: augmented sample dump
# ---------------------------------------------------------------------------

def check_augmentation_dump(first_batches: dict | None) -> None:
    _section("Augmentation visual dump")

    if first_batches is None or "train" not in first_batches:
        _warn("Skipping augmentation dump — no train batch available")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        _warn("matplotlib not available — skipping augmentation dump")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    imgs, labels = first_batches["train"]
    n = min(16, imgs.size(0))

    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(n):
        img = (imgs[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(config.CLASS_NAMES[labels[i].item()].replace("___", "\n"), fontsize=6)
        axes[i].axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    out_path = REPORTS_DIR / "augmented_samples.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    _pass(f"Augmented samples saved → {out_path}")


# ---------------------------------------------------------------------------
# Check: model forward pass (runs on CPU without dataset)
# ---------------------------------------------------------------------------

def check_model_forward(batch_size: int, first_batches: dict | None) -> dict:
    _section("Model forward passes")
    device = torch.device("cpu")  # always CPU for this check

    models_built = {}
    for model_name in ["baseline", "mobilenet_v2"]:
        try:
            model = build_model(model_name, num_classes=38).to(device).eval()
        except Exception as e:
            _fail(f"{model_name}: build_model() raised: {e}")
            continue

        # Use a real batch if available, otherwise random
        if first_batches and "train" in first_batches:
            imgs = first_batches["train"][0][:min(batch_size, 4)].to(device)
        else:
            imgs = torch.randn(min(batch_size, 4), 3, 224, 224)

        try:
            with torch.no_grad():
                logits = model(imgs)
        except Exception as e:
            _fail(f"{model_name}: forward pass raised: {e}")
            continue

        if logits.shape != (imgs.size(0), 38):
            _fail(f"{model_name}: output shape {tuple(logits.shape)} != ({imgs.size(0)}, 38)")
        elif torch.isnan(logits).all():
            _fail(f"{model_name}: all logits are NaN")
        elif (logits == logits[0]).all():
            _fail(f"{model_name}: all logits are identical — degenerate output")
        else:
            _pass(f"{model_name}: shape={tuple(logits.shape)}, "
                  f"range=[{logits.min():.3f}, {logits.max():.3f}]")
            models_built[model_name] = model

    return models_built


# ---------------------------------------------------------------------------
# Check: MobileNetV2 gradient flow
# ---------------------------------------------------------------------------

def check_gradient_flow(models_built: dict) -> None:
    _section("MobileNetV2 gradient flow (frozen backbone)")

    if "mobilenet_v2" not in models_built:
        _warn("MobileNetV2 not built — skipping gradient check")
        return

    model = models_built["mobilenet_v2"].train()
    model.freeze_backbone()

    imgs   = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 38, (4,))
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    optimizer.zero_grad()
    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()

    frozen_with_grad    = []
    trainable_no_grad   = []
    n_frozen = n_trainable = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            n_frozen += 1
            if p.grad is not None:
                frozen_with_grad.append(name)
        else:
            n_trainable += 1
            if p.grad is None:
                trainable_no_grad.append(name)

    _pass(f"{n_frozen} frozen params (no grad), {n_trainable} trainable params (grad present)")

    if frozen_with_grad:
        _fail(f"{len(frozen_with_grad)} frozen param(s) have gradients: {frozen_with_grad[:3]}...")
    else:
        _pass("No frozen params received gradients ✓")

    if trainable_no_grad:
        _fail(f"{len(trainable_no_grad)} trainable param(s) have None grad: {trainable_no_grad[:3]}...")
    else:
        _pass("All trainable params have gradients ✓")


# ---------------------------------------------------------------------------
# Check: GPU memory (skipped if no CUDA)
# ---------------------------------------------------------------------------

def check_gpu_memory(batch_size: int) -> None:
    _section("GPU memory estimate")

    if not torch.cuda.is_available():
        _warn("No CUDA device — skipping GPU memory check")
        return

    device = torch.device("cuda")
    model = build_model("mobilenet_v2", num_classes=38).to(device).train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    imgs   = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 38, (batch_size,), device=device)

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(imgs)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    gpu_name = torch.cuda.get_device_name(0)

    msg = f"Peak GPU memory: {peak_mb:.0f} MB on {gpu_name} (batch_size={batch_size})"
    if peak_mb > 6 * 1024:
        _warn(f"{msg} — exceeds 6 GB: AMP may not be engaged or batch is too large")
    else:
        _pass(msg)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-flight validation before training")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default 0 for compatibility)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Treat missing CUDA as a warning, not a failure")
    return p.parse_args()


def main() -> None:
    # Reconfigure stdout to UTF-8 so ✓/⚠/✗ render on Windows cp1252 consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    print("\n" + "="*60)
    print("  PLANT DISEASE PREFLIGHT CHECK")
    print("="*60)

    check_cuda(args.allow_cpu)
    check_normalization_constants()
    first_batches = check_data_loading(args.batch_size, args.num_workers)
    check_augmentation_dump(first_batches)
    models_built = check_model_forward(args.batch_size, first_batches)
    check_gradient_flow(models_built)
    check_gpu_memory(args.batch_size)

    # Summary
    print(f"\n{'='*60}")
    n_pass = sum([
        1 for _ in range(1)  # at least normalization ran
    ])
    if _failures:
        print(f"  RESULT: {len(_failures)} FAILURE(S), {len(_warnings)} WARNING(S)")
        for f in _failures:
            print(f"    ✗ {f}")
        sys.exit(2)
    elif _warnings:
        print(f"  RESULT: PASSED WITH {len(_warnings)} WARNING(S)")
        for w in _warnings:
            print(f"    ⚠ {w}")
        sys.exit(1)
    else:
        print("  RESULT: ALL CHECKS PASSED ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
