#test
# Plant Disease Classification

Diploma project. Two models compared: a small baseline CNN (~1.2 M params) and MobileNetV2 (~2.3 M params) with two-stage transfer learning, both trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (38 classes, ~70 k training images).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

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

`uv` pulls CUDA 12.8 PyTorch wheels automatically on Windows/Linux. If your GPU needs older CUDA drivers, change the index in `pyproject.toml`:

```toml
[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu126"   # or cu124
```

### Dataset

You need a Kaggle API token. Three ways to provide it:

**Option A — `KAGGLE_API_TOKEN` env var** (the new `KGAT_...` token from kaggle.com/settings):

```powershell
# Windows PowerShell (persistent):
[Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_...", "User")
```
```bash
# macOS/Linux (~/.bashrc or ~/.zshrc):
export KAGGLE_API_TOKEN="KGAT_..."
```

**Option B — `KAGGLE_USERNAME` + `KAGGLE_KEY`** (legacy key pair from the JSON file):

```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", "your_username", "User")
[Environment]::SetEnvironmentVariable("KAGGLE_KEY", "your_key_hex", "User")
```

**Option C — `kaggle.json` file** at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS).

Then run:

```bash
python scripts/download_dataset.py
```

### Prepare splits

```bash
python scripts/prepare_splits.py
```

Splits `valid/` 50/50 stratified into val + test. Run once after downloading.

### Validate environment

```bash
python scripts/preflight.py
```

Checks CUDA, data loading, model forward passes, gradient flow, GPU memory.

## Training

```bash
python -m plant_disease.train --config configs/baseline.json
python -m plant_disease.train --config configs/mobilenet_v2.json
```

CLI flags override the config:

```bash
python -m plant_disease.train --config configs/baseline.json --epochs 10 --batch-size 32
```

Best checkpoint → `models/<model>_best.pt` + `models/<model>_best_meta.json`.

## Evaluation

```bash
python -m plant_disease.evaluate \
    --checkpoint models/baseline_best.pt \
    --output-dir outputs/eval/baseline/

python -m plant_disease.evaluate \
    --checkpoint models/mobilenet_v2_best.pt \
    --output-dir outputs/eval/mobilenet_v2/
```

## Compare models

```bash
python scripts/compare_models.py \
    models/baseline_best.pt \
    models/mobilenet_v2_best.pt
```

Output: `reports/baseline_vs_mobilenet.md`

## Results

| Model | Test accuracy |
|-------|--------------|
| Baseline CNN | 99.70% |
| MobileNetV2 | 99.68% |

## Tests

```bash
pytest
```

Tests run without the dataset — data-dependent tests skip automatically when `data/` is absent.

### Quick sanity check

```bash
python scripts/smoke_test.py
```

Runs tests, preflight, and a 1-epoch mini-training run on 100 images.

## Predict

```bash
python scripts/predict.py models/mobilenet_v2_best.pt path/to/leaf.jpg
python scripts/predict.py models/mobilenet_v2_best.pt path/to/leaf.jpg --top-k 5
```

## Running the API

Install dependencies (includes FastAPI, uvicorn, pydantic-settings):

```bash
uv pip install -e ".[dev]"
```

Start the backend from the `plant-disease/` directory:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

Test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Predict (curl)
curl -X POST http://localhost:8000/predict \
  -F "image=@path/to/leaf.jpg" \
  -F "model=mobilenet_v2"

# Predict (PowerShell)
Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method POST `
  -Form @{ image = Get-Item path\to\leaf.jpg; model = "mobilenet_v2" }

# List classes
curl http://localhost:8000/classes
```

### API environment variables

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `./models` | Directory containing `.pt` checkpoint files |
| `MAX_UPLOAD_MB` | `10` | Maximum accepted image size in MB |
| `DEFAULT_MODEL` | `mobilenet_v2` | Model used when `model` form field is omitted |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Python logging level |

Override via env var or a `.env` file in `plant-disease/`.

### API tests

```bash
pytest tests/test_api_health.py tests/test_api_predict.py tests/test_api_classes.py -v
```

Add `-m "not slow"` to skip tests that require real checkpoints and dataset images (the default).

## Troubleshooting

**`torch.cuda.is_available()` returns False on Windows** — you got the CPU wheel. Force the CUDA build:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

**Driver version** — CUDA 12.8 wheels require driver ≥ 570.00. Switch to `/cu126` (≥ 560.00) or `/cu124` (≥ 550.54) if needed.

**`RuntimeError: CUDA error: device-side assert triggered`** — usually a label index out of range. Check that `prepare_splits.py` ran and the dataset has exactly 38 classes.

**DataLoader hangs on Windows with `num_workers > 0`** — all entry points use `if __name__ == "__main__"` guards as required for the `spawn` start method. Don't call `get_dataloaders()` at module level in new scripts.

## Layout

```
plant-disease/
├── configs/
│   ├── baseline.json
│   └── mobilenet_v2.json
├── data/                          ← gitignored
│   ├── New Plant Diseases Dataset(Augmented)/
│   └── splits.json
├── models/                        ← gitignored
├── reports/
│   └── baseline_vs_mobilenet.md
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_splits.py
│   ├── compare_models.py
│   ├── predict.py
│   ├── preflight.py
│   └── smoke_test.py
├── src/plant_disease/
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── utils.py
│   └── models/
│       ├── baseline_cnn.py
│       └── mobilenet_v2.py
└── tests/
```
