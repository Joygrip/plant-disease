"""Dataset, transforms, and DataLoader factories."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from plant_disease import config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlantDiseaseDataset(Dataset):
    """Loads images from an explicit list of (path, label_index) pairs."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------

def _load_train_samples(class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name, idx in class_to_idx.items():
        class_dir = config.TRAIN_DIR / class_name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((img_path, idx))
    return samples


def _load_split_samples(
    split_key: str,
    class_to_idx: dict[str, int],
) -> list[tuple[Path, int]]:
    """Load val or test samples from the splits index file."""
    if not config.SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Splits file not found: {config.SPLITS_FILE}\n"
            "Run  python scripts/prepare_splits.py  first."
        )
    with open(config.SPLITS_FILE, encoding="utf-8") as f:
        splits = json.load(f)

    samples: list[tuple[Path, int]] = []
    for rel_path in splits[split_key]:
        # rel_path is relative to config.VALID_DIR
        abs_path = config.VALID_DIR / rel_path
        class_name = Path(rel_path).parent.name
        if class_name not in class_to_idx:
            raise KeyError(f"Unknown class '{class_name}' in splits file")
        samples.append((abs_path, class_to_idx[class_name]))
    return samples


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_class_to_idx() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(config.CLASS_NAMES)}


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    augment: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    class_to_idx = get_class_to_idx()

    train_samples = _load_train_samples(class_to_idx)
    val_samples = _load_split_samples("val", class_to_idx)
    test_samples = _load_split_samples("test", class_to_idx)

    train_tf = _train_transform() if augment else _eval_transform()

    train_ds = PlantDiseaseDataset(train_samples, train_tf)
    val_ds = PlantDiseaseDataset(val_samples, _eval_transform())
    test_ds = PlantDiseaseDataset(test_samples, _eval_transform())

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
