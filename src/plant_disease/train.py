"""
Training entry point.

Usage:
    python -m plant_disease.train --config configs/baseline.json
    python -m plant_disease.train --config configs/mobilenet_v2.json
    python -m plant_disease.train --model baseline --epochs 30 --batch-size 64
    python -m plant_disease.train --help
"""

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from plant_disease import config
from plant_disease.data import get_dataloaders
from plant_disease.models import build_model
from plant_disease.utils import CSVLogger, count_parameters, get_device, get_logger, seed_everything


# ---------------------------------------------------------------------------
# CLI  (config file → defaults; CLI flags override)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    cfg: dict = {}
    if pre_args.config is not None:
        with open(pre_args.config) as f:
            cfg = json.load(f)

    p = argparse.ArgumentParser(description="Train a plant disease classifier")
    p.add_argument("--config", type=Path, default=None, help="JSON config file (flags override)")
    p.add_argument("--model", choices=["baseline", "mobilenet_v2"],
                   default=cfg.get("model", "baseline"))
    p.add_argument("--epochs", type=int, default=cfg.get("epochs", config.EPOCHS))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", config.BATCH_SIZE))
    p.add_argument("--lr", type=float, default=cfg.get("lr", config.LEARNING_RATE))
    p.add_argument("--weight-decay", type=float,
                   default=cfg.get("weight_decay", config.WEIGHT_DECAY))
    p.add_argument("--num-workers", type=int, default=cfg.get("num_workers", config.NUM_WORKERS))
    p.add_argument("--output-dir", type=Path,
                   default=Path(cfg.get("output_dir", str(config.MODELS_DIR))))
    p.add_argument("--seed", type=int, default=cfg.get("seed", config.SEED))
    p.add_argument("--no-aug", action="store_true", help="Disable train augmentation")
    p.add_argument("--smoke-test", action="store_true",
                   help="Subsample to 100 train + 50 val images, force 1 epoch (CI sanity check)")
    p.add_argument("--finetune-schedule", choices=["single-stage", "two-stage"],
                   default=cfg.get("finetune_schedule", "single-stage"))
    p.add_argument("--stage1-epochs", type=int, default=cfg.get("stage1_epochs", 5))
    p.add_argument("--stage2-lr", type=float, default=cfg.get("stage2_lr", 1e-4))
    p.add_argument("--unfreeze-blocks", type=int, default=cfg.get("unfreeze_blocks", 3))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Optimizer / scheduler factory
# ---------------------------------------------------------------------------

def _build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    num_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    output_dir: Path,
    model_name: str,
    args: argparse.Namespace,
    epoch: int,
    val_acc: float,
    stage: str = "1",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{model_name}_best.pt")
    meta = {
        "model": model_name,
        "epoch": epoch,
        "best_stage": stage,
        "val_acc": val_acc,
        "class_names": config.CLASS_NAMES,
        "num_classes": config.NUM_CLASSES,
        "input_size": config.IMAGE_SIZE,
        "imagenet_mean": list(config.IMAGENET_MEAN),
        "imagenet_std": list(config.IMAGENET_STD),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "finetune_schedule": args.finetune_schedule,
            "stage1_epochs": args.stage1_epochs,
            "stage2_lr": args.stage2_lr,
            "unfreeze_blocks": args.unfreeze_blocks,
        },
    }
    with open(output_dir / f"{model_name}_best_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Run-info writer
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    try:
        ref = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=config.ROOT,
        )
        dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            capture_output=True, cwd=config.ROOT,
        )
        commit = ref.stdout.strip()
        return commit + (" (dirty)" if dirty.returncode != 0 else "")
    except Exception:
        return "unavailable"


def _write_run_info(
    output_dir: Path,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
    total_time_s: float,
    epochs_run: int,
    best_epoch: int,
    best_val_acc: float,
) -> None:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        peak_gpu_mb = torch.cuda.max_memory_allocated(0) / 1024 ** 2
    else:
        gpu_name = "CPU"
        peak_gpu_mb = None

    run_info = {
        "model": model_name,
        "device": str(device),
        "gpu_name": gpu_name,
        "peak_gpu_memory_mb": round(peak_gpu_mb, 1) if peak_gpu_mb is not None else None,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "total_training_time_s": round(total_time_s, 1),
        "mean_epoch_time_s": round(total_time_s / max(epochs_run, 1), 1),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "git_commit": _get_git_commit(),
        "config": vars(args),
    }
    with open(output_dir / f"{model_name}_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Training phase — single shared loop used by both stages
# ---------------------------------------------------------------------------

def _train_phase(
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch_range: range,
    total_epochs: int,
    stage: str,
    patience: int,
    best_val_acc: float,
    patience_counter: int,
    logger,
    csv_logger: CSVLogger,
    output_dir: Path,
    model_name: str,
    args: argparse.Namespace,
) -> tuple[float, int, bool, int, int]:
    """
    Run one training phase.

    Returns (best_val_acc, patience_counter, stopped_early, last_epoch, best_epoch).
    best_epoch is -1 if no improvement occurred in this phase.
    """
    stopped_early = False
    last_epoch = epoch_range.start
    best_epoch = -1

    for epoch in epoch_range:
        last_epoch = epoch
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d}/{total_epochs}  stage={stage}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.0f}s)"
        )
        csv_logger.log({
            "epoch": epoch,
            "stage": stage,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "elapsed_s": f"{elapsed:.1f}",
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(model, output_dir, model_name, args, epoch, val_acc, stage)
            logger.info(f"  ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={patience}, stage={stage})"
                )
                stopped_early = True
                break

    return best_val_acc, patience_counter, stopped_early, last_epoch, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", args.output_dir / f"{args.model}_train.log")
    csv_logger = CSVLogger(
        args.output_dir / f"{args.model}_metrics.csv",
        ["epoch", "stage", "train_loss", "val_loss", "val_acc", "elapsed_s"],
    )

    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_aug,
    )

    if args.smoke_test:
        rng = random.Random(args.seed)
        n_train = min(100, len(train_loader.dataset))
        n_val = min(50, len(val_loader.dataset))
        loader_kw = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_loader = DataLoader(
            Subset(train_loader.dataset, rng.sample(range(len(train_loader.dataset)), n_train)),
            shuffle=True, **loader_kw,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, rng.sample(range(len(val_loader.dataset)), n_val)),
            shuffle=False, **loader_kw,
        )
        args.epochs = 1
        args.finetune_schedule = "single-stage"
        logger.info(f"Smoke-test mode: {n_train} train / {n_val} val images, 1 epoch, single-stage")

    model = build_model(args.model, num_classes=config.NUM_CLASSES).to(device)
    logger.info(f"Total parameters   : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    t_start = time.time()

    phase_kwargs = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        scaler=scaler,
        device=device,
        total_epochs=args.epochs,
        patience=config.EARLY_STOP_PATIENCE,
        logger=logger,
        csv_logger=csv_logger,
        output_dir=args.output_dir,
        model_name=args.model,
        args=args,
    )

    if args.finetune_schedule == "single-stage":
        optimizer, scheduler = _build_optimizer_and_scheduler(
            model, args.lr, args.weight_decay, args.epochs
        )
        logger.info(f"Trainable parameters: {count_parameters(model):,}")
        best_val_acc, _, _, last_epoch, best_epoch = _train_phase(
            **phase_kwargs,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_range=range(1, args.epochs + 1),
            stage="1",
            best_val_acc=0.0,
            patience_counter=0,
        )
        epochs_run = last_epoch

    else:  # two-stage
        s1_epochs = args.stage1_epochs
        s2_epochs = args.epochs - s1_epochs
        if s2_epochs < 1:
            raise ValueError(
                f"--stage1-epochs ({s1_epochs}) must be less than --epochs ({args.epochs})"
            )

        model.freeze_backbone()
        logger.info(f"Stage 1: frozen backbone. Trainable: {count_parameters(model):,}")
        opt1, sch1 = _build_optimizer_and_scheduler(model, args.lr, args.weight_decay, s1_epochs)
        best_val_acc, _, stopped, last_epoch_s1, best_epoch_s1 = _train_phase(
            **phase_kwargs,
            optimizer=opt1,
            scheduler=sch1,
            epoch_range=range(1, s1_epochs + 1),
            stage="1",
            best_val_acc=0.0,
            patience_counter=0,
        )
        epochs_run = last_epoch_s1
        best_epoch = best_epoch_s1

        if not stopped:
            model.unfreeze_top_blocks(args.unfreeze_blocks)
            logger.info(
                f"Stage 2: unfroze top {args.unfreeze_blocks} feature layers. "
                f"Trainable: {count_parameters(model):,}"
            )
            opt2, sch2 = _build_optimizer_and_scheduler(
                model, args.stage2_lr, args.weight_decay, s2_epochs
            )
            best_val_acc, _, _, last_epoch_s2, best_epoch_s2 = _train_phase(
                **phase_kwargs,
                optimizer=opt2,
                scheduler=sch2,
                epoch_range=range(s1_epochs + 1, args.epochs + 1),
                stage="2",
                best_val_acc=best_val_acc,
                patience_counter=0,
            )
            epochs_run = last_epoch_s2
            if best_epoch_s2 != -1:
                best_epoch = best_epoch_s2

    total_time = time.time() - t_start
    logger.info(
        f"Training complete. Best val_acc={best_val_acc:.4f}. "
        f"Total time: {total_time / 60:.1f} min"
    )

    _write_run_info(
        output_dir=args.output_dir,
        model_name=args.model,
        args=args,
        device=device,
        total_time_s=total_time,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
    )
    logger.info(f"Run info written → {args.output_dir / (args.model + '_run_info.json')}")


if __name__ == "__main__":
    main()
