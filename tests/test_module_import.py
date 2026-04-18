"""
Confirm that entry-point modules can be imported as libraries without
triggering any top-level side effects (no DataLoader spawning, no sys.exit).

This guards the Windows multiprocessing requirement: DataLoader with
num_workers > 0 on Windows requires all DataLoader creation to happen
inside `if __name__ == "__main__"` blocks — never at module top-level.
"""

import importlib
import sys
from pathlib import Path

import pytest

# Make sure scripts/ is importable as modules
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _import(module_name: str) -> None:
    """Import (or re-use cached) module; any exception is a test failure."""
    importlib.import_module(module_name)


def test_import_plant_disease_train() -> None:
    _import("plant_disease.train")


def test_import_plant_disease_evaluate() -> None:
    _import("plant_disease.evaluate")


def test_import_plant_disease_config() -> None:
    _import("plant_disease.config")


def test_import_plant_disease_data() -> None:
    _import("plant_disease.data")


def test_import_preflight_script() -> None:
    _import("preflight")


def test_import_compare_models_script() -> None:
    _import("compare_models")
