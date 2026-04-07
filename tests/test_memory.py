"""Tests for memory.py: core CRUD operations and CLI command handlers.

Test classes:
    TestInit    — idempotent init, tables exist after init
    TestAdd     — returns ID, row present in both tables, auto-init
    TestSearch  — most-similar text returned, --limit, --json output
    TestDelete  — removes from both tables; unknown ID exits 1
    TestList    — newest-first, --source filter, --limit 0 returns all
    TestStats   — populated DB; missing DB exits 0

Integration tests (real ONNX model) are guarded by
    AGENT_MEMORY_INTEGRATION_TESTS=1
"""

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Import memory.py, neutralising _bootstrap() side-effects:
#   - subprocess.check_call  (would create a venv + pip-install)
#   - os.execv               (would re-exec Python inside the venv)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

_MEMORY_AVAILABLE = False
_mem = None
_import_exc = None

try:
    import importlib.util as _ilu

    with mock.patch("os.execv", lambda *a, **kw: None), \
         mock.patch("subprocess.check_call", lambda *a, **kw: None):
        _spec = _ilu.spec_from_file_location(
            "memory", os.path.join(_REPO_ROOT, "memory.py")
        )
        _mem = _ilu.module_from_spec(_spec)
        sys.modules.setdefault("memory", _mem)
        _spec.loader.exec_module(_mem)

    _MEMORY_AVAILABLE = True
except Exception as _exc:
    _import_exc = _exc

_SKIP_MSG = (
    f"memory.py or its dependencies not importable: {_import_exc}"
    if not _MEMORY_AVAILABLE
    else ""
)


# ---------------------------------------------------------------------------
# Deterministic mock embedding
# ---------------------------------------------------------------------------

def _mock_embed(texts):
    """Return a deterministic unit-vector array of shape (N, 384).

    The same text always produces the same vector (hash-seeded RNG), so
    searching with the exact stored text yields distance ≈ 0 for that entry.
    """
    import numpy as np

    rows = []
    for text in texts:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384).astype(np.float32)
        v = v / np.linalg.norm(v)
        rows.append(v)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _TempDirTest(unittest.TestCase):
    """Base: each test runs in a fresh temp dir; DB path via AGENT_MEMORY_DB."""

    def setUp(self):
        self._orig_dir = os.getcwd()
        self._tmpdir = tempfile.mkdtemp()
        os.chdir(self._tmpdir)
        self._db_file = os.path.join(self._tmpdir, "test_memories.db")
        os.environ["AGENT_MEMORY_DB"] = self._db_file

    def tearDown(self):
        os.chdir(self._orig_dir)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        os.environ.pop("AGENT_MEMORY_DB", None)

    def _args(self, **kwargs):
        """Build a Namespace for CLI command handlers."""
        return argparse.Namespace(db=None, **kwargs)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestInit(_TempDirTest):

    def test_init_creates_db_file(self):
        _mem.cmd_init(self._args())
        self.assertTrue(os.path.exists(self._db_file))

    def test_init_is_idempotent(self):
        _mem.cmd_init(self._args())
        _mem.cmd_init(self._args())  # must not raise
        self.assertTrue(os.path.exists(self._db_file))

    def test_memories_table_exists(self):
        _mem.cmd_init(self._args())
        con = sqlite3.connect(self._db_file)
        try:
            names = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        finally:
            con.close()
        self.assertIn("memories", names)

    def test_memories_vec_table_exists(self):
        """memories_vec (virtual) appears in sqlite_master after init."""
        _mem.cmd_init(self._args())
        con = sqlite3.connect(self._db_file)
        try:
            # virtual tables show up with type 'table' in sqlite_master
            names = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master"
                ).fetchall()
            }
        finally:
            con.close()
        self.assertIn("memories_vec", names)


# ---------------------------------------------------------------------------
# TestAdd
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestAdd(_TempDirTest):

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def test_add_prints_id(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_add(self._args(text="Hello world", source=None, meta=None))
            output = out.getvalue()
        self.assertIn("Added", output)
        parts = output.strip().split()
        self.assertEqual(len(parts), 2)
        self.assertRegex(parts[1], r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

    def test_add_row_in_memories(self):
        _mem.cmd_add(self._args(text="test content", source="test-src", meta=None))
        con = sqlite3.connect(self._db_file)
        try:
            row = con.execute("SELECT content, source FROM memories").fetchone()
        finally:
            con.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "test content")
        self.assertEqual(row[1], "test-src")

    def test_add_row_in_memories_vec(self):
        _mem.cmd_add(self._args(text="vector test", source=None, meta=None))
        con = _mem._open_db(Path(self._db_file))
        try:
            count_mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            count_vec = con.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        finally:
            con.close()
        self.assertEqual(count_mem, 1)
        self.assertEqual(count_vec, 1)

    def test_add_auto_inits_db(self):
        """cmd_add creates the DB automatically when it does not exist."""
        self.assertFalse(os.path.exists(self._db_file))
        _mem.cmd_add(self._args(text="auto init", source=None, meta=None))
        self.assertTrue(os.path.exists(self._db_file))


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestSearch(_TempDirTest):

    TEXT_A = "the quick brown fox"
    TEXT_B = "lazy dog sleeps soundly"
    TEXT_C = "python programming language"

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()
        _mem.cmd_init(self._args())
        for text in (self.TEXT_A, self.TEXT_B, self.TEXT_C):
            _mem.cmd_add(self._args(text=text, source="test", meta=None))

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def test_most_similar_text_returned_first(self):
        """Exact-match query should rank the matching memory first (distance ≈ 0)."""
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_search(
                self._args(query=self.TEXT_A, limit=3, threshold=None, source=None, json=False)
            )
            output = out.getvalue()
        first_line = output.strip().splitlines()[0]
        self.assertIn(self.TEXT_A[:30], first_line)

    def test_limit_respected(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_search(
                self._args(query=self.TEXT_A, limit=1, threshold=None, source=None, json=False)
            )
            lines = [l for l in out.getvalue().strip().splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)

    def test_json_output_is_valid_json(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_search(
                self._args(query=self.TEXT_A, limit=3, threshold=None, source=None, json=True)
            )
            output = out.getvalue()
        parsed = json.loads(output)  # raises ValueError if invalid JSON
        self.assertIsInstance(parsed, list)
        if parsed:
            self.assertIn("id", parsed[0])
            self.assertIn("content", parsed[0])
            self.assertIn("distance", parsed[0])


# ---------------------------------------------------------------------------
# TestDelete
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestDelete(_TempDirTest):

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()
        _mem.cmd_init(self._args())

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def _add_and_get_id(self, text="deletable memory"):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_add(self._args(text=text, source=None, meta=None))
            output = out.getvalue()
        return output.strip().split()[-1]

    def test_delete_removes_from_memories(self):
        mem_id = self._add_and_get_id()
        with patch("sys.stdout", new_callable=io.StringIO):
            _mem.cmd_delete(self._args(id=mem_id))
        con = sqlite3.connect(self._db_file)
        try:
            row = con.execute("SELECT id FROM memories WHERE id = ?", (mem_id,)).fetchone()
        finally:
            con.close()
        self.assertIsNone(row)

    def test_delete_removes_from_memories_vec(self):
        mem_id = self._add_and_get_id()
        con_pre = _mem._open_db(Path(self._db_file))
        try:
            row = con_pre.execute(
                "SELECT rowid FROM memories WHERE id = ?", (mem_id,)
            ).fetchone()
            self.assertIsNotNone(row)
            stored_rowid = row[0]
        finally:
            con_pre.close()

        with patch("sys.stdout", new_callable=io.StringIO):
            _mem.cmd_delete(self._args(id=mem_id))

        # After deletion, memories_vec count should be 0
        con_post = _mem._open_db(Path(self._db_file))
        try:
            vec_count = con_post.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        finally:
            con_post.close()
        self.assertEqual(vec_count, 0)

    def test_delete_unknown_id_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            _mem.cmd_delete(self._args(id="00000000-0000-0000-0000-000000000000"))
        self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# TestList
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestList(_TempDirTest):

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()
        _mem.cmd_init(self._args())

        # Insert directly with controlled timestamps to guarantee ordering.
        con = sqlite3.connect(self._db_file)
        try:
            rows = [
                ("id-first",  "first memory",  "src-a", "{}", "2024-01-01T00:00:01Z"),
                ("id-second", "second memory", "src-b", "{}", "2024-01-01T00:00:02Z"),
                ("id-third",  "third memory",  "src-a", "{}", "2024-01-01T00:00:03Z"),
            ]
            con.executemany(
                "INSERT INTO memories (id, content, source, metadata, created_at) VALUES (?,?,?,?,?)",
                rows,
            )
            con.commit()
        finally:
            con.close()

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def test_newest_first(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_list(self._args(limit=20, source=None, json=True))
            rows = json.loads(out.getvalue())
        contents = [r["content"] for r in rows]
        self.assertGreater(contents.index("first memory"), contents.index("third memory"),
                           "third memory (newer) should appear before first memory (older)")

    def test_source_filter(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_list(self._args(limit=20, source="src-a", json=True))
            rows = json.loads(out.getvalue())
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["source"], "src-a")

    def test_limit_0_returns_all(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_list(self._args(limit=0, source=None, json=True))
            rows = json.loads(out.getvalue())
        self.assertEqual(len(rows), 3)

    def test_limit_1_returns_one(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_list(self._args(limit=1, source=None, json=True))
            rows = json.loads(out.getvalue())
        self.assertEqual(len(rows), 1)


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestStats(_TempDirTest):

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def test_stats_on_populated_db(self):
        _mem.cmd_init(self._args())
        _mem.cmd_add(self._args(text="stat test memory", source=None, meta=None))
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_stats(self._args())
            output = out.getvalue()
        self.assertIn("Memories:", output)
        self.assertIn("1", output)

    def test_stats_missing_db_exits_0(self):
        """cmd_stats on a missing DB should exit 0, not raise an error."""
        self.assertFalse(os.path.exists(self._db_file))
        with self.assertRaises(SystemExit) as ctx:
            _mem.cmd_stats(self._args())
        self.assertEqual(ctx.exception.code, 0)


# ---------------------------------------------------------------------------
# Integration tests (real ONNX model — skipped unless explicitly enabled)
# ---------------------------------------------------------------------------

_RUN_INTEGRATION = os.environ.get("AGENT_MEMORY_INTEGRATION_TESTS") == "1"


@unittest.skipUnless(
    _RUN_INTEGRATION and _MEMORY_AVAILABLE,
    "Set AGENT_MEMORY_INTEGRATION_TESTS=1 to run integration tests",
)
class TestSearchIntegration(_TempDirTest):
    """End-to-end semantic search using the real bge-small-en-v1.5 ONNX model."""

    def test_semantic_similarity(self):
        """Semantically similar texts should rank above unrelated texts."""
        _mem.cmd_init(self._args())
        _mem.cmd_add(self._args(text="The cat sat on the mat.", source=None, meta=None))
        _mem.cmd_add(self._args(text="Python is a programming language.", source=None, meta=None))

        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_search(
                self._args(
                    query="A feline rested on a rug.",
                    limit=2,
                    threshold=None,
                    source=None,
                    json=True,
                )
            )
            results = json.loads(out.getvalue())

        self.assertTrue(results, "Expected at least one search result")
        self.assertIn("cat", results[0]["content"])


if __name__ == "__main__":
    unittest.main()
