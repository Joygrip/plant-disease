r"""
Download the plant-disease dataset from Kaggle.

Authentication — use ONE of the following options (checked in this order):

  Option A — KAGGLE_API_TOKEN env var (recommended, new Kaggle token format):
    The KGAT-prefixed token is introspected server-side; no username needed.
    Windows PowerShell (persistent):
      [Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_...", "User")
    macOS/Linux (~/.bashrc or ~/.zshrc):
      export KAGGLE_API_TOKEN="KGAT_..."

  Option B — KAGGLE_USERNAME + KAGGLE_KEY env vars (legacy API key pair):
    Windows PowerShell (persistent):
      [Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", "your_username", "User")
      [Environment]::SetEnvironmentVariable("KAGGLE_KEY", "your_key_hex", "User")
    macOS/Linux:
      export KAGGLE_USERNAME="your_username"
      export KAGGLE_KEY="your_key_hex"

  Option C — kaggle.json file (legacy file):
    Windows: %USERPROFILE%\.kaggle\kaggle.json
    macOS/Linux: ~/.kaggle/kaggle.json  (chmod 600)
    Download from: https://www.kaggle.com/settings -> API -> Create New Token

  If both Option A and another option are present, Option A wins (kaggle
  library tries access-token auth before legacy-key auth).

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --data-dir path/to/data
    python scripts/download_dataset.py --force    # re-download if present
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "vipoooool/new-plant-diseases-dataset"
EXPECTED_SUBDIR = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
EXPECTED_SPLITS = ["train", "valid"]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_NO_AUTH_MSG = """\
[ERROR] Kaggle authentication not configured. Use one of:

  Option A (recommended — new KGAT token, no username needed):
    Set environment variable KAGGLE_API_TOKEN to your KGAT_... token.
    Windows PowerShell (persistent, close+reopen shell after):
      [Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_...", "User")
    macOS/Linux (add to ~/.bashrc or ~/.zshrc):
      export KAGGLE_API_TOKEN="KGAT_..."
    Generate token at: https://www.kaggle.com/settings -> API -> Create New Token

  Option B (legacy key pair):
    Set both KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    Windows PowerShell (persistent):
      [Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", "your_username", "User")
      [Environment]::SetEnvironmentVariable("KAGGLE_KEY", "your_key_hex", "User")

  Option C (legacy file):
    Place kaggle.json at:
      Windows: %USERPROFILE%\\.kaggle\\kaggle.json
      macOS/Linux: ~/.kaggle/kaggle.json  (chmod 600)
    Download from: https://www.kaggle.com/settings -> API -> Create New Token
"""


def _detect_auth() -> str:
    """Return a string describing the auth source found, or exit with an error.

    Returns one of: "token_env", "legacy_env", "legacy_file".
    Precedence: KAGGLE_API_TOKEN > KAGGLE_USERNAME+KAGGLE_KEY > kaggle.json.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "token_env"

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy_env"

    # Check both Windows and Unix default locations.
    candidates = [
        Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".kaggle" / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    if any(p.exists() for p in candidates):
        return "legacy_file"

    print(_NO_AUTH_MSG, file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the plant-disease Kaggle dataset")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Directory to extract into (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the dataset already exists")
    return p.parse_args()


def _download(data_dir: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        if e.name == "kaggle" or (e.name and e.name.startswith("kaggle.")):
            print(
                "[ERROR] kaggle package not installed. Run:\n"
                "  uv pip install kaggle\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET_SLUG} into {data_dir} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(data_dir), quiet=False)

    candidates = list(data_dir.glob("*.zip"))
    if not candidates:
        print("[ERROR] No zip file found after download.", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def _unzip(zip_path: Path, data_dir: Path) -> None:
    print(f"Extracting {zip_path.name} ...")
    
    extract_dir = str(data_dir.resolve())
    if os.name == "nt" and not extract_dir.startswith("\\\\?\\"):
        extract_dir = "\\\\?\\" + extract_dir
        
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()
    print("Zip removed.")


def _verify_structure(dataset_dir: Path) -> None:
    print("\nVerifying directory structure ...")
    for split in EXPECTED_SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            print(
                f"[ERROR] Expected directory not found after unzip: {split_dir}\n"
                "The dataset vendor may have changed the zip layout. "
                "Check the Kaggle page and update EXPECTED_SPLITS / EXPECTED_SUBDIR.",
                file=sys.stderr,
            )
            sys.exit(1)

    for split in EXPECTED_SPLITS:
        split_dir = dataset_dir / split
        found_classes = {d.name for d in split_dir.iterdir() if d.is_dir()}
        expected_classes = set(CLASS_NAMES)
        missing = expected_classes - found_classes
        extra = found_classes - expected_classes
        if missing:
            print(f"[WARN] {split}/: missing expected classes: {sorted(missing)[:5]}...")
        if extra:
            print(f"[WARN] {split}/: unexpected extra classes: {sorted(extra)[:5]}...")
        if not missing and not extra:
            print(f"  [OK] {split}/: all 38 classes present")


def _print_counts(dataset_dir: Path) -> None:
    print("\nFile counts per class:\n")
    img_exts = {".jpg", ".jpeg", ".png"}
    header = f"{'Class':<55} {'train':>7} {'valid':>7}"
    print(header)
    print("-" * len(header))

    for cls in CLASS_NAMES:
        counts = {}
        for split in EXPECTED_SPLITS:
            class_dir = dataset_dir / split / cls
            if class_dir.is_dir():
                counts[split] = sum(
                    1 for f in class_dir.iterdir() if f.suffix.lower() in img_exts
                )
            else:
                counts[split] = 0
        print(f"{cls:<55} {counts.get('train', 0):>7} {counts.get('valid', 0):>7}")


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.data_dir / EXPECTED_SUBDIR

    if dataset_dir.exists() and not args.force:
        print(f"Dataset already present at {dataset_dir}")
        print("Pass --force to re-download.")
        _verify_structure(dataset_dir)
        _print_counts(dataset_dir)
        return

    auth_source = _detect_auth()
    print(f"Kaggle auth: {auth_source}")
    zip_path = _download(args.data_dir)
    _unzip(zip_path, args.data_dir)
    _verify_structure(dataset_dir)
    _print_counts(dataset_dir)
    print(f"\nDataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
