"""
Tests for Kaggle auth detection in scripts/download_dataset.py.

Key facts verified against kaggle library source (kaggle_api_extended.py):
  - KAGGLE_API_TOKEN alone is sufficient for KGAT tokens; the library
    introspects the username server-side. No KAGGLE_USERNAME needed.
  - KAGGLE_USERNAME + KAGGLE_KEY together activate the legacy env path.
  - kaggle.json at the default location activates the legacy file path.
  - If both KAGGLE_API_TOKEN and another method are present, KAGGLE_API_TOKEN
    wins (the library tries access-token auth before legacy-key auth).

No real Kaggle API calls are made — KaggleApi.authenticate() is
patched out in tests that exercise _download().
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_dataset as dd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_env(*keys: str):
    """Context manager: temporarily remove env vars, restore on exit."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    return _ctx()


# ---------------------------------------------------------------------------
# _detect_auth — Option A: KAGGLE_API_TOKEN only
# ---------------------------------------------------------------------------

def test_token_env_alone_is_sufficient():
    """KAGGLE_API_TOKEN alone → 'token_env' (no username var needed)."""
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        os.environ["KAGGLE_API_TOKEN"] = "KGAT_abc123"
        result = dd._detect_auth()
    assert result == "token_env"


def test_token_env_takes_precedence_over_json():
    """When both KAGGLE_API_TOKEN and kaggle.json are present, token wins."""
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=True),  # simulate json present
    ):
        os.environ["KAGGLE_API_TOKEN"] = "KGAT_abc123"
        result = dd._detect_auth()
    assert result == "token_env"


def test_token_env_takes_precedence_over_legacy_env():
    """When both KAGGLE_API_TOKEN and KAGGLE_USERNAME+KEY are set, token wins."""
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        os.environ["KAGGLE_API_TOKEN"] = "KGAT_abc123"
        os.environ["KAGGLE_USERNAME"] = "myuser"
        os.environ["KAGGLE_KEY"] = "deadbeef"
        result = dd._detect_auth()
    assert result == "token_env"


# ---------------------------------------------------------------------------
# _detect_auth — Option B: KAGGLE_USERNAME + KAGGLE_KEY
# ---------------------------------------------------------------------------

def test_legacy_env_when_both_username_and_key_set():
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        os.environ["KAGGLE_USERNAME"] = "myuser"
        os.environ["KAGGLE_KEY"] = "deadbeef"
        result = dd._detect_auth()
    assert result == "legacy_env"


def test_legacy_env_requires_both_vars():
    """Only KAGGLE_USERNAME without KAGGLE_KEY must not return 'legacy_env'."""
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        os.environ["KAGGLE_USERNAME"] = "myuser"
        # KAGGLE_KEY intentionally absent
        with pytest.raises(SystemExit):
            dd._detect_auth()


def test_legacy_env_requires_both_vars_key_only():
    """Only KAGGLE_KEY without KAGGLE_USERNAME must not return 'legacy_env'."""
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        os.environ["KAGGLE_KEY"] = "deadbeef"
        # KAGGLE_USERNAME intentionally absent
        with pytest.raises(SystemExit):
            dd._detect_auth()


# ---------------------------------------------------------------------------
# _detect_auth — Option C: kaggle.json file
# ---------------------------------------------------------------------------

def test_legacy_file_when_json_exists(tmp_path):
    """kaggle.json at the default location → 'legacy_file'."""
    fake_json = tmp_path / "kaggle.json"
    fake_json.write_text('{"username":"u","key":"k"}')

    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        # Patch Path.exists to return True only for paths ending in kaggle.json
        patch.object(
            dd.Path,
            "exists",
            lambda self: str(self).endswith("kaggle.json"),
        ),
    ):
        result = dd._detect_auth()
    assert result == "legacy_file"


# ---------------------------------------------------------------------------
# _detect_auth — no credentials at all
# ---------------------------------------------------------------------------

def test_no_auth_exits_with_error():
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        with pytest.raises(SystemExit) as exc_info:
            dd._detect_auth()
    assert exc_info.value.code == 1


def test_no_auth_error_message_mentions_both_options(capsys):
    with (
        _clean_env("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
        patch.object(dd.Path, "exists", return_value=False),
    ):
        with pytest.raises(SystemExit):
            dd._detect_auth()
    captured = capsys.readouterr()
    assert "KAGGLE_API_TOKEN" in captured.err
    assert "KAGGLE_USERNAME" in captured.err
    assert "kaggle.json" in captured.err


# ---------------------------------------------------------------------------
# _download — calls KaggleApi.authenticate() (mocked)
# ---------------------------------------------------------------------------

def test_download_calls_authenticate(tmp_path):
    """_download() must call api.authenticate() before any API method."""
    fake_zip = tmp_path / "new-plant-diseases-dataset.zip"
    fake_zip.write_bytes(b"PK")  # minimal non-empty file for glob to find

    mock_api = MagicMock()
    mock_cls = MagicMock(return_value=mock_api)

    # Patch the class in its defining module (imported locally inside _download).
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", mock_cls):
        result = dd._download(tmp_path)

    mock_api.authenticate.assert_called_once()
    mock_api.dataset_download_files.assert_called_once_with(
        dd.DATASET_SLUG, path=str(tmp_path), quiet=False
    )
    assert result == fake_zip
