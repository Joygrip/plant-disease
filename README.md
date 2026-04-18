# Plant Disease Classification

A PyTorch pipeline for identifying plant diseases from leaf images, built as a diploma project. Two models are compared: a lightweight baseline CNN (~1.2 M parameters) and MobileNetV2 (~2.3 M parameters with 38-class head) trained via two-stage transfer learning.

## Setup

### Prerequisites

- **Python 3.13** — not 3.14. PyTorch has no CUDA wheels for cp314 yet; 3.13 (cp313) has full CUDA support on the pytorch.org index since torch 2.5.
- **uv** package manager — `pip install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **NVIDIA GPU with CUDA 12.8 drivers** — verify with `nvidia-smi`. CPU training works but is ~20× slower. Older drivers (CUDA 12.4 or 12.6) also work; see [Troubleshooting](#troubleshooting) for the index override.
- **Kaggle account + API token** — needed for `download_dataset.py` (see below).

### Install

```bash
git clone <repo-url>
cd plant-disease

uv venv --python 3.13

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

uv pip install -e ".[dev]"
```

`uv` will automatically pull the CUDA 12.8 PyTorch wheels on Windows and Linux (configured in `pyproject.toml` via `[tool.uv.sources]`). macOS gets the default CPU wheel.

cu128 is the current channel for Python 3.13 + Windows — pytorch.org stopped shipping Windows cu121 wheels after torch 2.4.x. If your GPU requires older CUDA drivers, edit the index URL in `pyproject.toml`:
```toml
[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu126"   # or cu124 for older drivers
```

### Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Should print `True <your GPU name>`. If it prints `False`, see [Troubleshooting](#troubleshooting).

### Download dataset

The script accepts three auth methods, checked in this order of precedence:

#### Option A — `KAGGLE_API_TOKEN` env var (recommended)

This is the token format Kaggle's current UI issues (`KGAT_...`). The token
is self-contained — no username variable needed; the kaggle library resolves
the username server-side via token introspection.

**Windows PowerShell** (persistent, survives reboots — close and reopen the shell after):
```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_...", "User")
```

**macOS/Linux** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export KAGGLE_API_TOKEN="KGAT_..."
```

Generate a token at [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**.

#### Option B — `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars (legacy key pair)

**Windows PowerShell** (persistent):
```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", "your_username", "User")
[Environment]::SetEnvironmentVariable("KAGGLE_KEY", "your_key_hex", "User")
```

**macOS/Linux**:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key_hex"
```

#### Option C — `kaggle.json` file (legacy file)

Save the file to:
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- macOS/Linux: `~/.kaggle/kaggle.json` then `chmod 600 ~/.kaggle/kaggle.json`

Download from [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**.

#### Run the download

```bash
python scripts/download_dataset.py
```

Downloads [`vipoooool/new-plant-diseases-dataset`](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), unzips into `data/`, prints per-class file counts. Idempotent — re-running when the dataset exists is a no-op.

Expected layout after download:

```
data/
└── New Plant Diseases Dataset(Augmented)/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   └── ...  (38 classes, ~70 k images)
    └── valid/
        ├── Apple___Apple_scab/
        └── ...  (38 classes, ~17.5 k images)
```

### Validate environment

```bash
python scripts/preflight.py
```

Must exit 0 before training. Checks CUDA, ImageNet normalisation, data loading, model forward passes, gradient flow, and GPU memory.

### Prepare splits

```bash
python scripts/prepare_splits.py
```

Carves `valid/` 50/50 stratified into val + test. Writes `data/splits.json`. Run once after downloading.

### Train

```bash
python -m plant_disease.train --config configs/baseline.json
python -m plant_disease.train --config configs/mobilenet_v2.json
```

Override any value from the CLI:

```bash
python -m plant_disease.train --config configs/baseline.json --epochs 10 --batch-size 32
```

Best checkpoint → `models/<model>_best.pt` + `models/<model>_best_meta.json`.  
Per-epoch metrics → `models/<model>_metrics.csv`.

### Evaluate

```bash
python -m plant_disease.evaluate \
    --checkpoint models/baseline_best.pt \
    --model baseline \
    --output-dir outputs/eval/baseline/

python -m plant_disease.evaluate \
    --checkpoint models/mobilenet_v2_best.pt \
    --model mobilenet_v2 \
    --output-dir outputs/eval/mobilenet_v2/
```

### Compare models

```bash
python scripts/compare_models.py \
    models/baseline_best.pt \
    models/mobilenet_v2_best.pt
```

Output: `reports/baseline_vs_mobilenet.md`

### Expected results

| Model | Val accuracy | Test accuracy |
|-------|-------------|--------------|
| Baseline CNN | ~93–96% | ~93–95% |
| MobileNetV2 | ~97–99% | ~97–98% |

### Run tests

```bash
pytest                      # all fast tests
pytest -m "not slow"        # skip slow tests explicitly
```

Tests run without the dataset. Integration tests that need real data are skipped automatically when `data/` is absent.

### One-command sanity check

```bash
python scripts/smoke_test.py
```

Runs the fast test suite, preflight (CPU-safe), and a 1-epoch mini-training run on 100 images. Use this after any environment change to confirm the whole pipeline is wired up.

## Troubleshooting

### `torch.cuda.is_available()` is False on Windows

You got the CPU wheel. Force the CUDA build:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Then re-verify with the one-liner above.

### `nvidia-smi` shows the driver but CUDA is still unavailable

Make sure the installed CUDA toolkit version matches the wheel. CUDA 12.8 wheels require driver ≥ 570.00. If your driver is older, switch the index URL in `pyproject.toml` to `/cu126` (driver ≥ 560.00) or `/cu124` (driver ≥ 550.54) and re-run `uv pip install -e ".[dev]"`.

### `RuntimeError: CUDA error: device-side assert triggered`

Usually a label index out of range. Confirm `prepare_splits.py` ran successfully and the dataset has exactly 38 classes.

### Accidentally used Python 3.14?

Python 3.14 (cp314) has no CUDA PyTorch wheels yet — `torch.cuda.is_available()` will return `False` even with a working GPU, and `uv pip install torch` will silently fall back to a CPU-only build.

Recreate the venv with the correct version:

```bash
deactivate
rmdir /s /q .venv          # Windows
# rm -rf .venv             # macOS/Linux
uv venv --python 3.13
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

Then verify: `python --version` should print `Python 3.13.x`.

### Windows: DataLoader hangs with `num_workers > 0`

All training entry points are guarded with `if __name__ == "__main__"` — this is required for Windows multiprocessing (`spawn` start method). Do not call `get_dataloaders()` at module top-level in any new script.

## Directory layout

```
plant-disease/
├── configs/
│   ├── baseline.json              ← training preset for baseline CNN
│   └── mobilenet_v2.json          ← training preset for MobileNetV2
├── data/                          ← gitignored
│   ├── New Plant Diseases Dataset(Augmented)/
│   └── splits.json
├── models/                        ← gitignored
│   ├── baseline_best.pt
│   ├── baseline_best_meta.json
│   ├── baseline_metrics.csv
│   ├── mobilenet_v2_best.pt
│   └── mobilenet_v2_best_meta.json
├── reports/
│   └── baseline_vs_mobilenet.md   ← generated by compare_models.py
├── scripts/
│   ├── download_dataset.py        ← Kaggle download + verify
│   ├── prepare_splits.py
│   ├── inspect_dataset.py
│   ├── compare_models.py
│   ├── preflight.py
│   └── smoke_test.py              ← one-command sanity check
├── src/plant_disease/
│   ├── config.py                  ← paths, hyperparams, class list
│   ├── data.py                    ← Dataset + DataLoader factories
│   ├── train.py                   ← training loop (CLI, two-stage, --smoke-test)
│   ├── evaluate.py                ← model-agnostic evaluation + reports
│   ├── utils.py                   ← seeding, logging, metrics
│   └── models/
│       ├── __init__.py            ← build_model() factory
│       ├── baseline_cnn.py        ← 4-block CNN, ~1.2 M params
│       └── mobilenet_v2.py        ← MobileNetV2 classifier, ~3.4 M params
└── tests/
    ├── test_data.py
    ├── test_mobilenet.py
    ├── test_module_import.py      ← confirms scripts import cleanly (no side-effects)
    ├── test_preflight.py
    └── test_training_config.py
```
