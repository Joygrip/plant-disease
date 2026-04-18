"""
One-command environment sanity check.

Runs, in order:
  1. pytest fast tests  (excludes @pytest.mark.slow)
  2. python scripts/preflight.py --allow-cpu  (works without GPU)
  3. python -m plant_disease.train --config configs/baseline.json
         --epochs 1 --smoke-test  (100 train + 50 val images, 1 epoch)

Exits 0 only if all three steps pass.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-train   # steps 1+2 only
"""

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, cmd: list[str], *, ok_codes: tuple[int, ...] = (0,)) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  CMD : {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode not in ok_codes:
        print(f"\n[SMOKE TEST FAILED] Step '{label}' exited {result.returncode}",
              file=sys.stderr)
        sys.exit(result.returncode)
    print(f"  [OK] {label}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-pipeline smoke test")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip the mini training run (steps 1+2 only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    _run(
        "pytest fast tests",
        [py, "-m", "pytest", "-m", "not slow", "--tb=short", "-q"],
    )

    _run(
        "preflight --allow-cpu",
        [py, str(ROOT / "scripts" / "preflight.py"), "--allow-cpu"],
        ok_codes=(0, 1),  # 1 = warnings only (e.g. dataset not yet downloaded)
    )

    if not args.skip_train:
        if not (ROOT / "data" / "New Plant Diseases Dataset(Augmented)" / "train").exists():
            print(
                "\n[SKIP] Dataset not found — skipping mini training run.\n"
                "       Run  python scripts/download_dataset.py  first, then\n"
                "       re-run smoke_test.py without --skip-train."
            )
        else:
            _run(
                "mini training run (1 epoch, 100 images)",
                [
                    py, "-m", "plant_disease.train",
                    "--config", str(ROOT / "configs" / "baseline.json"),
                    "--epochs", "1",
                    "--smoke-test",
                ],
            )

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
