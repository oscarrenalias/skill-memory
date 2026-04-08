"""Tests for memory.py _ensure_model() (bead B-2ad17234).

Coverage:
  (1) first-call download succeeds + SHA256 passes → returns model path
  (2) subsequent call when file already exists → no-op (no download)
  (3) SHA256 mismatch → file is unlinked + sys.exit(1)
  (4) network failure → file is unlinked (if partial) + sys.exit(1)
"""
import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

MEMORY_PY = Path(__file__).parent.parent / "memory.py"
_VENV_PATH = str(MEMORY_PY.parent / ".venv")


def _load_memory_module(tmp_model_path: Path):
    """Load memory.py as a fresh module with bootstrap and model path patched.

    Returns the module object with _MODEL_PATH replaced by tmp_model_path so
    we can control whether the file exists without touching the real cache.
    """
    # Patch sys.prefix so _bootstrap() is a no-op
    with mock.patch("sys.prefix", _VENV_PATH):
        spec = importlib.util.spec_from_file_location("memory_test_instance", str(MEMORY_PY))
        mod = importlib.util.module_from_spec(spec)
        # Insert into sys.modules under a unique name so imports don't collide
        sys.modules["memory_test_instance"] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop("memory_test_instance", None)

    # Redirect the module's _MODEL_PATH to our temp location
    mod._MODEL_PATH = tmp_model_path
    return mod


def _fake_sha256(content: bytes) -> str:
    """Return the SHA256 hex digest of content."""
    return hashlib.sha256(content).hexdigest()


class TestEnsureModelNoOp(unittest.TestCase):
    """(2) If the model file already exists, _ensure_model returns immediately."""

    def test_existing_file_returns_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"
            # Create a dummy file to simulate an already-downloaded model
            model_path.write_bytes(b"pretend-model-data")

            mod = _load_memory_module(model_path)

            # urlretrieve must NOT be called
            with mock.patch("urllib.request.urlretrieve") as mock_dl:
                result = mod._ensure_model()

            mock_dl.assert_not_called()
            self.assertEqual(result, model_path)

    def test_existing_file_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"
            model_path.write_bytes(b"pretend-model-data")

            mod = _load_memory_module(model_path)

            with mock.patch("urllib.request.urlretrieve"):
                mod._ensure_model()

            self.assertTrue(model_path.exists(), "Existing model file was unexpectedly deleted")


class TestEnsureModelFirstDownload(unittest.TestCase):
    """(1) First-call download succeeds and SHA256 is verified."""

    def test_download_and_sha_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"
            # model_path does NOT exist yet
            self.assertFalse(model_path.exists())

            fake_content = b"fake-onnx-model-bytes"
            expected_sha = _fake_sha256(fake_content)

            mod = _load_memory_module(model_path)
            # Override the SHA256 constant so our fake content matches
            mod._MODEL_SHA256 = expected_sha

            def _fake_urlretrieve(url, dest):
                Path(dest).write_bytes(fake_content)

            with mock.patch("urllib.request.urlretrieve", side_effect=_fake_urlretrieve):
                result = mod._ensure_model()

            self.assertEqual(result, model_path)
            self.assertTrue(model_path.exists(), "Model file was not created by download")

    def test_download_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "deep" / "nested" / "model.onnx"
            self.assertFalse(model_path.parent.exists())

            fake_content = b"data"
            expected_sha = _fake_sha256(fake_content)

            mod = _load_memory_module(model_path)
            mod._MODEL_SHA256 = expected_sha

            def _fake_urlretrieve(url, dest):
                Path(dest).write_bytes(fake_content)

            with mock.patch("urllib.request.urlretrieve", side_effect=_fake_urlretrieve):
                mod._ensure_model()

            self.assertTrue(model_path.parent.exists(), "Parent directories were not created")


class TestEnsureModelShaMismatch(unittest.TestCase):
    """(3) SHA256 mismatch → file unlinked, sys.exit(1) raised."""

    def test_sha_mismatch_calls_sys_exit_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            mod = _load_memory_module(model_path)
            # Set expected SHA to something that will never match our fake content
            mod._MODEL_SHA256 = "a" * 64

            def _fake_urlretrieve(url, dest):
                Path(dest).write_bytes(b"wrong-content")

            with mock.patch("urllib.request.urlretrieve", side_effect=_fake_urlretrieve):
                with self.assertRaises(SystemExit) as cm:
                    mod._ensure_model()

            self.assertEqual(cm.exception.code, 1)

    def test_sha_mismatch_unlinks_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            mod = _load_memory_module(model_path)
            mod._MODEL_SHA256 = "b" * 64

            def _fake_urlretrieve(url, dest):
                Path(dest).write_bytes(b"bad-content")

            with mock.patch("urllib.request.urlretrieve", side_effect=_fake_urlretrieve):
                try:
                    mod._ensure_model()
                except SystemExit:
                    pass

            self.assertFalse(
                model_path.exists(),
                "Partial/corrupt model file was not removed after SHA256 mismatch",
            )


class TestEnsureModelNetworkFailure(unittest.TestCase):
    """(4) Network failure (non-SystemExit exception) → file unlinked, sys.exit(1) raised."""

    def test_network_error_calls_sys_exit_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            mod = _load_memory_module(model_path)

            def _raise_network_error(url, dest):
                raise OSError("connection refused")

            with mock.patch("urllib.request.urlretrieve", side_effect=_raise_network_error):
                with self.assertRaises(SystemExit) as cm:
                    mod._ensure_model()

            self.assertEqual(cm.exception.code, 1)

    def test_network_error_unlinks_partial_file(self):
        """If urlretrieve writes a partial file before failing, it must be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            mod = _load_memory_module(model_path)

            def _partial_then_fail(url, dest):
                # Write partial content to simulate interrupted download
                Path(dest).write_bytes(b"\x00\x01partial")
                raise OSError("network timeout")

            with mock.patch("urllib.request.urlretrieve", side_effect=_partial_then_fail):
                try:
                    mod._ensure_model()
                except SystemExit:
                    pass

            self.assertFalse(
                model_path.exists(),
                "Partial download file was not cleaned up after network error",
            )

    def test_network_error_no_partial_file_ok(self):
        """If urlretrieve fails before creating any file, no error should occur in cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            mod = _load_memory_module(model_path)

            def _fail_immediately(url, dest):
                # Never creates the file
                raise ConnectionError("DNS failure")

            with mock.patch("urllib.request.urlretrieve", side_effect=_fail_immediately):
                with self.assertRaises(SystemExit) as cm:
                    mod._ensure_model()

            self.assertEqual(cm.exception.code, 1)
            self.assertFalse(model_path.exists())


if __name__ == "__main__":
    unittest.main()
