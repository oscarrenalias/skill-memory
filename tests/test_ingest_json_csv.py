"""Tests for _ingest_json, _ingest_csv, and cmd_ingest dispatch in memory.py.

Covers:
  - JSON string-array branch
  - JSON object-array branch (with content key)
  - JSON object missing 'content' key → SystemExit(1)
  - CSV happy path — column text extracted; other columns → metadata
  - CSV missing --column flag → SystemExit(1)
  - Unknown file extension → SystemExit(1)
"""
import os
import sys
import json
import csv
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Bootstrap memory.py with heavy deps mocked so no venv/model is needed.
# ---------------------------------------------------------------------------
os.execv = lambda *a, **kw: None  # prevent venv re-exec

with mock.patch.dict('sys.modules', {
    'numpy': mock.MagicMock(),
    'sqlite_vec': mock.MagicMock(),
    'onnxruntime': mock.MagicMock(),
}):
    with mock.patch('os.execv', lambda *a, **kw: None):
        import importlib
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("memory_jc", os.path.join(repo_root, "memory.py"))
        _mem = importlib.util.module_from_spec(spec)
        sys.modules['memory_jc'] = _mem
        try:
            spec.loader.exec_module(_mem)
        except Exception:
            pass

_ingest_json = _mem._ingest_json
_ingest_csv = _mem._ingest_csv
cmd_ingest = _mem.cmd_ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed(texts):
    """Return a deterministic unit-vector array stub (shape: N×384)."""
    import numpy as real_np
    arr = real_np.ones((len(texts), 384), dtype=real_np.float32)
    norms = real_np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / norms).astype(real_np.float32)


class _BaseIngestTest(unittest.TestCase):
    """Set up a temp dir with a temp DB path for each test."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        # Patch _embed and all DB-touching helpers so tests run without
        # onnxruntime or sqlite_vec installed in the test environment.
        self._patches = [
            mock.patch.object(_mem, '_embed', side_effect=_fake_embed),
            mock.patch.object(_mem, '_auto_init', return_value=None),
            mock.patch.object(_mem, '_open_db', return_value=mock.MagicMock()),
            mock.patch.object(_mem, '_insert_chunk', return_value=None),
        ]
        for p in self._patches:
            p.start()
        self.mock_insert = _mem._insert_chunk

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _write_json(self, payload):
        path = Path(self.tmp_dir) / "data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_csv(self, rows, fieldnames):
        path = Path(self.tmp_dir) / "data.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path


# ---------------------------------------------------------------------------
# JSON tests
# ---------------------------------------------------------------------------

class TestIngestJson(_BaseIngestTest):

    def test_string_array_adds_one_row_per_string(self):
        """JSON array of strings: each string becomes a memory row."""
        path = self._write_json(["hello", "world", "foo"])
        _ingest_json(path, source="test-src", db_path=self.db_path)
        self.assertEqual(self.mock_insert.call_count, 3)
        # Each call should have the string as the content arg
        contents = [c.args[1] for c in self.mock_insert.call_args_list]
        self.assertIn("hello", contents)
        self.assertIn("world", contents)
        self.assertIn("foo", contents)

    def test_string_array_uses_provided_source(self):
        """Source tag flows through to _insert_chunk for string elements."""
        path = self._write_json(["memory text"])
        _ingest_json(path, source="my-source", db_path=self.db_path)
        call = self.mock_insert.call_args_list[0]
        self.assertEqual(call.args[2], "my-source")

    def test_object_array_uses_content_key(self):
        """JSON array of objects: 'content' key is the memory text."""
        path = self._write_json([{"content": "object content"}])
        _ingest_json(path, source="src", db_path=self.db_path)
        self.assertEqual(self.mock_insert.call_count, 1)
        self.assertEqual(self.mock_insert.call_args.args[1], "object content")

    def test_object_array_metadata_passed_through(self):
        """JSON object 'metadata' dict is forwarded to _insert_chunk."""
        path = self._write_json([{"content": "text", "metadata": {"tag": "test"}}])
        _ingest_json(path, source="src", db_path=self.db_path)
        meta_arg = self.mock_insert.call_args.args[3]
        self.assertEqual(meta_arg, {"tag": "test"})

    def test_object_array_per_row_source_overrides_default(self):
        """A 'source' key on a JSON object overrides the command-level source."""
        path = self._write_json([{"content": "text", "source": "row-src"}])
        _ingest_json(path, source="default-src", db_path=self.db_path)
        self.assertEqual(self.mock_insert.call_args.args[2], "row-src")

    def test_missing_content_key_exits_1(self):
        """Object element without 'content' key causes SystemExit(1)."""
        path = self._write_json([{"text": "no content key"}])
        with self.assertRaises(SystemExit) as ctx:
            _ingest_json(path, source="src", db_path=self.db_path)
        self.assertEqual(ctx.exception.code, 1)

    def test_empty_array_adds_zero_rows(self):
        """Empty JSON array ingests zero rows without error."""
        path = self._write_json([])
        _ingest_json(path, source="src", db_path=self.db_path)
        self.assertEqual(self.mock_insert.call_count, 0)

    def test_non_array_top_level_exits_1(self):
        """A top-level JSON object (not array) causes SystemExit(1)."""
        path = self._write_json({"content": "not an array"})
        with self.assertRaises(SystemExit) as ctx:
            _ingest_json(path, source="src", db_path=self.db_path)
        self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------

class TestIngestCsv(_BaseIngestTest):

    def test_happy_path_adds_one_row_per_data_row(self):
        """CSV with --column extracts text from specified column."""
        path = self._write_csv(
            [{"text": "hello csv", "tag": "a"}, {"text": "world csv", "tag": "b"}],
            fieldnames=["text", "tag"],
        )
        _ingest_csv(path, column="text", source="csv-src", db_path=self.db_path)
        self.assertEqual(self.mock_insert.call_count, 2)
        contents = [c.args[1] for c in self.mock_insert.call_args_list]
        self.assertIn("hello csv", contents)
        self.assertIn("world csv", contents)

    def test_other_columns_become_metadata(self):
        """Non-text columns appear in the metadata dict passed to _insert_chunk."""
        path = self._write_csv(
            [{"text": "content here", "author": "alice", "year": "2024"}],
            fieldnames=["text", "author", "year"],
        )
        _ingest_csv(path, column="text", source="src", db_path=self.db_path)
        meta_arg = self.mock_insert.call_args.args[3]
        self.assertEqual(meta_arg.get("author"), "alice")
        self.assertEqual(meta_arg.get("year"), "2024")
        self.assertNotIn("text", meta_arg)

    def test_column_not_found_exits_1(self):
        """Specifying a non-existent column name causes SystemExit(1)."""
        path = self._write_csv([{"text": "hi"}], fieldnames=["text"])
        with self.assertRaises(SystemExit) as ctx:
            _ingest_csv(path, column="nonexistent", source="src", db_path=self.db_path)
        self.assertEqual(ctx.exception.code, 1)

    def test_source_tag_propagated(self):
        """Source tag is forwarded to _insert_chunk for CSV rows."""
        path = self._write_csv([{"text": "data"}], fieldnames=["text"])
        _ingest_csv(path, column="text", source="csv-tag", db_path=self.db_path)
        self.assertEqual(self.mock_insert.call_args.args[2], "csv-tag")


# ---------------------------------------------------------------------------
# cmd_ingest dispatch tests (unknown extension)
# ---------------------------------------------------------------------------

class TestCmdIngestDispatch(_BaseIngestTest):

    def _make_args(self, file, column=None, source=None, chunk_size=1000, overlap=100, db=None):
        return SimpleNamespace(
            file=str(file),
            column=column,
            source=source,
            chunk_size=chunk_size,
            overlap=overlap,
            db=str(db) if db else None,
        )

    def test_unknown_extension_exits_1(self):
        """Unsupported file extension causes SystemExit(1)."""
        path = Path(self.tmp_dir) / "data.xyz"
        path.write_text("some content", encoding="utf-8")
        args = self._make_args(path)
        with self.assertRaises(SystemExit) as ctx:
            cmd_ingest(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_csv_without_column_flag_exits_1(self):
        """CSV ingest without --column causes SystemExit(1)."""
        path = self._write_csv([{"text": "hi"}], fieldnames=["text"])
        args = self._make_args(path, column=None)
        with self.assertRaises(SystemExit) as ctx:
            cmd_ingest(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_missing_file_exits_1(self):
        """Non-existent file path causes SystemExit(1)."""
        args = self._make_args(Path(self.tmp_dir) / "missing.json")
        with self.assertRaises(SystemExit) as ctx:
            cmd_ingest(args)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
