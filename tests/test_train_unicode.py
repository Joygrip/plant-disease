"""
Tests that train.py handles Unicode characters (✓, —) without encoding errors.

The checkpoint log line contains these chars; on Windows the default console
encoding is cp1252 which can't encode them. main() reconfigures stdout/stderr
to utf-8 before any logging happens.
"""

import io
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Verify the problematic characters exist in the source
# ---------------------------------------------------------------------------

def test_checkpoint_log_contains_checkmark():
    """The 'New best' log line in train.py must contain ✓ and —."""
    src = (Path(__file__).resolve().parents[1] / "src" / "plant_disease" / "train.py")
    text = src.read_text(encoding="utf-8")
    assert "✓" in text, "train.py must log ✓ on new-best checkpoint"
    assert "—" in text, "train.py must log — on new-best checkpoint"


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
    import inspect
    from plant_disease import train

    src = inspect.getsource(train.main)
    assert 'sys.stdout.reconfigure(encoding="utf-8")' in src, (
        "main() must call sys.stdout.reconfigure(encoding='utf-8')"
    )
    assert 'sys.stderr.reconfigure(encoding="utf-8")' in src, (
        "main() must call sys.stderr.reconfigure(encoding='utf-8')"
    )
