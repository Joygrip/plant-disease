"""
Tests that Unicode characters used across the project survive Windows cp1252.

Characters in use:
  ✓  U+2713  checkpoint-saved log line (train.py)
  —  U+2014  em-dash in log lines (train.py, evaluate.py)
  →  U+2192  arrow used in report text (compare_models.py, evaluate.py,
              error_analysis.py, generate_results_doc.py)
  Δ  U+0394  delta column header (compare_models.py per-class F1 table)

Root causes on Windows:
  • sys.stdout / sys.stderr default to cp1252 → reconfigure() in main()
  • logging.FileHandler defaults to locale encoding → encoding="utf-8" in
    get_logger()
  • Path.write_text() and open(..., "w") default to locale encoding →
    encoding="utf-8" on every text write call
"""

import io
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Verify every problematic character exists in the relevant source files
# ---------------------------------------------------------------------------

def test_train_source_contains_checkmark_and_em_dash():
    src = (Path(__file__).resolve().parents[1] / "src" / "plant_disease" / "train.py")
    text = src.read_text(encoding="utf-8")
    assert "✓" in text, "train.py must log ✓ on new-best checkpoint"
    assert "—" in text, "train.py must log — in log messages"


def test_compare_models_source_contains_arrow_and_delta():
    src = Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py"
    text = src.read_text(encoding="utf-8")
    assert "→" in text, "compare_models.py report contains → arrows"
    assert "Δ" in text, "compare_models.py report contains Δ delta column"


def test_evaluate_source_contains_arrow():
    src = Path(__file__).resolve().parents[1] / "src" / "plant_disease" / "evaluate.py"
    text = src.read_text(encoding="utf-8")
    assert "→" in text, "evaluate.py uses → in confused-pair output"


def test_generate_results_doc_source_contains_arrow():
    src = Path(__file__).resolve().parents[1] / "scripts" / "generate_results_doc.py"
    text = src.read_text(encoding="utf-8")
    assert "→" in text, "generate_results_doc.py embeds → in report text"


# ---------------------------------------------------------------------------
# All write_text() and open("w") calls carry encoding="utf-8"
# ---------------------------------------------------------------------------

_WRITE_TEXT_FILES = [
    "scripts/compare_models.py",
    "scripts/generate_results_doc.py",
    "src/plant_disease/evaluate.py",
]

_READ_TEXT_FILES = [
    "scripts/generate_results_doc.py",
]

_OPEN_WRITE_FILES = [
    "src/plant_disease/error_analysis.py",
    "src/plant_disease/train.py",
    "src/plant_disease/utils.py",
    "scripts/compare_models.py",
    "scripts/prepare_splits.py",
]

_OPEN_READ_FILES = [
    "scripts/generate_results_doc.py",
    "scripts/compare_models.py",
    "scripts/plot_training_curves.py",
    "scripts/predict.py",
    "src/plant_disease/evaluate.py",
    "src/plant_disease/data.py",
    "src/plant_disease/train.py",
    "src/plant_disease/error_analysis.py",
]

ROOT = Path(__file__).resolve().parents[1]


def _source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_no_write_text_without_encoding():
    """Every write_text() call in production code must specify encoding='utf-8'."""
    for rel in _WRITE_TEXT_FILES:
        text = _source(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            if "write_text(" in line and "encoding" not in line:
                raise AssertionError(
                    f"{rel}:{lineno} — write_text() is missing encoding='utf-8':\n  {line.strip()}"
                )


def test_no_open_write_without_encoding():
    """Every open(..., 'w') or open(..., 'a') call must specify encoding='utf-8'."""
    for rel in _OPEN_WRITE_FILES:
        text = _source(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if 'open(' in stripped and ('"w"' in stripped or '"a"' in stripped):
                if "encoding" not in stripped:
                    raise AssertionError(
                        f"{rel}:{lineno} — open() write/append is missing encoding='utf-8':\n  {stripped}"
                    )


def test_no_read_text_without_encoding():
    """Every read_text() call in production code must specify encoding='utf-8'."""
    for rel in _READ_TEXT_FILES:
        text = _source(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # match only method calls (.read_text()), not _read_text() helper calls
            if ".read_text(" in stripped and "encoding" not in stripped:
                raise AssertionError(
                    f"{rel}:{lineno} — read_text() is missing encoding='utf-8':\n  {stripped}"
                )


def test_no_open_read_without_encoding():
    """Every open() used for reading must specify encoding='utf-8'.

    Matches open() calls that are NOT binary (no 'b' mode) and do not yet
    have an encoding= keyword.  Image.open() and csv.writer() binary modes
    are excluded because they contain 'b' or are in the binary-mode list.
    """
    for rel in _OPEN_READ_FILES:
        text = _source(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if "open(" not in stripped:
                continue
            # skip binary-mode opens (rb, wb, ab, etc.)
            if '"rb"' in stripped or "'rb'" in stripped:
                continue
            # skip write/append opens (covered by the write test)
            if '"w"' in stripped or '"a"' in stripped:
                continue
            # skip Image.open() — PIL binary read, no encoding
            if "Image.open(" in stripped:
                continue
            # must have explicit encoding for all remaining text-mode opens
            if "encoding" not in stripped:
                raise AssertionError(
                    f"{rel}:{lineno} — open() read is missing encoding='utf-8':\n  {stripped}"
                )


def test_file_handler_uses_utf8_encoding():
    """get_logger() must pass encoding='utf-8' to FileHandler."""
    text = _source("src/plant_disease/utils.py")
    assert 'FileHandler(log_file, encoding="utf-8")' in text, (
        "utils.py FileHandler must use encoding='utf-8' to log → and ✓ on Windows"
    )


# ---------------------------------------------------------------------------
# Encoding behaviour: cp1252 stream raises, reconfigured utf-8 stream does not
# ---------------------------------------------------------------------------

def test_cp1252_stream_raises_on_checkmark():
    """Baseline sanity: a raw cp1252 stream cannot encode ✓."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("✓")
        stream.flush()


def test_reconfigured_utf8_stream_handles_checkmark():
    """After reconfigure(utf-8) the same stream writes ✓ without error."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    stream.reconfigure(encoding="utf-8")
    stream.write("✓ New best val_acc=0.9972 \u2014 checkpoint saved")
    stream.flush()
    assert "✓" in buf.getvalue().decode("utf-8")


def test_reconfigured_utf8_stream_handles_em_dash():
    """The em-dash — (U+2014) used in log messages also survives reconfigure."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    stream.reconfigure(encoding="utf-8")
    stream.write("\u2014")
    stream.flush()
    assert "\u2014" in buf.getvalue().decode("utf-8")


def test_cp1252_raises_on_arrow():
    """→ (U+2192) used in report text cannot be encoded by cp1252."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("→")
        stream.flush()


def test_cp1252_raises_on_delta():
    """Δ (U+0394) used in F1 delta column cannot be encoded by cp1252."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("Δ")
        stream.flush()


def test_utf8_file_write_round_trips_all_special_chars(tmp_path):
    """write_text(encoding='utf-8') must preserve ✓ — → Δ on any platform."""
    content = "✓ checkpoint saved — val_acc=0.99 → Tomato___Late_blight Δ=+0.02"
    out = tmp_path / "report.txt"
    out.write_text(content, encoding="utf-8")
    assert out.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# Logger round-trip: ✓ written to a StringIO via StreamHandler → no error
# ---------------------------------------------------------------------------

def test_logger_writes_checkmark_to_stringio():
    """A StreamHandler targeting a StringIO can log ✓ and — without raising."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("test_train_unicode")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("  \u2713 New best val_acc=0.9972 \u2014 checkpoint saved")
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "\u2713" in output
    assert "\u2014" in output


# ---------------------------------------------------------------------------
# main() source contains the reconfigure calls
# ---------------------------------------------------------------------------

def test_main_source_contains_reconfigure_calls():
    """Confirm main() reconfigures both stdout and stderr to utf-8."""
    src = _source("src/plant_disease/train.py")
    assert 'sys.stdout.reconfigure(encoding="utf-8")' in src, (
        "main() must call sys.stdout.reconfigure(encoding='utf-8')"
    )
    assert 'sys.stderr.reconfigure(encoding="utf-8")' in src, (
        "main() must call sys.stderr.reconfigure(encoding='utf-8')"
    )
