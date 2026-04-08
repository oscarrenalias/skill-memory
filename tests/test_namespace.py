"""Tests for --namespace flag: partition isolation and default behaviour.

Coverage:
  (1) add to namespace A; list in namespace A returns it
  (2) add to namespace A; list in namespace B does NOT return it
  (3) add to namespace A; search in namespace B does NOT return it
  (4) default namespace ('default') is used when --namespace is omitted
  (5) stats shows per-namespace counts
"""

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Import memory.py, neutralising _bootstrap() side-effects.
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
# Deterministic mock embedding (same helper as test_memory.py)
# ---------------------------------------------------------------------------

def _mock_embed(texts):
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
        return argparse.Namespace(db=None, **kwargs)


# ---------------------------------------------------------------------------
# Namespace isolation tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestNamespaceIsolation(_TempDirTest):

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()
        _mem.cmd_init(self._args())

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def _add(self, text, namespace="default", source=None):
        _mem.cmd_add(self._args(text=text, source=source, meta=None, namespace=namespace))

    def _list(self, namespace="default", source=None):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_list(self._args(limit=0, source=source, json=True, namespace=namespace))
            return json.loads(out.getvalue())

    def _search(self, query, namespace="default", limit=5):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_search(
                self._args(
                    query=query,
                    limit=limit,
                    threshold=None,
                    source=None,
                    json=True,
                    namespace=namespace,
                )
            )
            return json.loads(out.getvalue())

    # (1) Memory added to namespace A is visible when listing namespace A.
    def test_list_returns_memory_in_same_namespace(self):
        self._add("alpha memory content", namespace="alpha")
        rows = self._list(namespace="alpha")
        contents = [r["content"] for r in rows]
        self.assertIn("alpha memory content", contents)

    # (2) Memory added to namespace A is NOT visible when listing namespace B.
    def test_list_isolates_across_namespaces(self):
        self._add("secret alpha content", namespace="alpha")
        rows = self._list(namespace="beta")
        contents = [r["content"] for r in rows]
        self.assertNotIn("secret alpha content", contents)

    # (3) Memory added to namespace A is NOT returned when searching namespace B.
    def test_search_isolates_across_namespaces(self):
        self._add("secret alpha content", namespace="alpha")
        # beta namespace is empty; search should return no results
        results = self._search("secret alpha content", namespace="beta")
        contents = [r["content"] for r in results]
        self.assertNotIn("secret alpha content", contents)

    # (4) Default namespace is 'default' when not specified.
    def test_default_namespace_is_default(self):
        # Add without explicit namespace — should land in 'default'
        _mem.cmd_add(self._args(text="default ns content", source=None, meta=None))
        rows = self._list(namespace="default")
        contents = [r["content"] for r in rows]
        self.assertIn("default ns content", contents)

    # (4b) Memory in 'default' namespace is not visible in another namespace.
    def test_default_namespace_not_visible_in_other(self):
        _mem.cmd_add(self._args(text="default ns content", source=None, meta=None))
        rows = self._list(namespace="other")
        contents = [r["content"] for r in rows]
        self.assertNotIn("default ns content", contents)

    # Additional: two namespaces can coexist; each sees only its own memories.
    def test_two_namespaces_coexist(self):
        self._add("memory for alpha", namespace="alpha")
        self._add("memory for beta", namespace="beta")

        alpha_rows = self._list(namespace="alpha")
        beta_rows = self._list(namespace="beta")

        alpha_contents = [r["content"] for r in alpha_rows]
        beta_contents = [r["content"] for r in beta_rows]

        self.assertIn("memory for alpha", alpha_contents)
        self.assertNotIn("memory for beta", alpha_contents)

        self.assertIn("memory for beta", beta_contents)
        self.assertNotIn("memory for alpha", beta_contents)


# ---------------------------------------------------------------------------
# Namespace schema tests (no embeddings needed)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestNamespaceSchema(_TempDirTest):

    def test_namespace_column_exists_after_init(self):
        """After init, the memories table has a namespace column."""
        _mem.cmd_init(self._args())
        con = sqlite3.connect(self._db_file)
        try:
            cols = {row[1] for row in con.execute("PRAGMA table_info(memories)").fetchall()}
        finally:
            con.close()
        self.assertIn("namespace", cols)

    def test_migration_adds_namespace_column_to_existing_db(self):
        """init on a DB created without namespace column adds the column idempotently."""
        # Create a DB without the namespace column (pre-migration schema)
        con = sqlite3.connect(self._db_file)
        try:
            con.execute(
                "CREATE TABLE memories "
                "(id TEXT NOT NULL UNIQUE, content TEXT NOT NULL, source TEXT, "
                "metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)"
            )
            con.commit()
        finally:
            con.close()

        # Running init should migrate (add namespace column) without error
        _mem.cmd_init(self._args())

        con = sqlite3.connect(self._db_file)
        try:
            cols = {row[1] for row in con.execute("PRAGMA table_info(memories)").fetchall()}
        finally:
            con.close()
        self.assertIn("namespace", cols)

    def test_migration_is_idempotent(self):
        """Running init twice on a migrated DB does not raise."""
        _mem.cmd_init(self._args())
        _mem.cmd_init(self._args())  # must not raise

    def test_existing_rows_default_to_default_namespace(self):
        """Rows inserted without namespace get 'default' via column DEFAULT."""
        _mem.cmd_init(self._args())
        con = sqlite3.connect(self._db_file)
        try:
            con.execute(
                "INSERT INTO memories (id, content, source, metadata, created_at) "
                "VALUES ('x', 'old content', NULL, '{}', '2024-01-01T00:00:00Z')"
            )
            con.commit()
            ns = con.execute(
                "SELECT namespace FROM memories WHERE id = 'x'"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(ns, "default")


# ---------------------------------------------------------------------------
# Stats per-namespace test
# ---------------------------------------------------------------------------

@unittest.skipUnless(_MEMORY_AVAILABLE, _SKIP_MSG)
class TestStatsNamespace(_TempDirTest):

    def setUp(self):
        super().setUp()
        self._embed_patch = patch.object(_mem, "_embed", side_effect=_mock_embed)
        self._embed_patch.start()

    def tearDown(self):
        self._embed_patch.stop()
        super().tearDown()

    def test_stats_shows_namespace_breakdown(self):
        _mem.cmd_init(self._args())
        _mem.cmd_add(self._args(text="alpha memory", source=None, meta=None, namespace="alpha"))
        _mem.cmd_add(self._args(text="beta memory", source=None, meta=None, namespace="beta"))
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            _mem.cmd_stats(self._args())
            output = out.getvalue()
        self.assertIn("alpha", output)
        self.assertIn("beta", output)
        self.assertIn("Namespaces:", output)


if __name__ == "__main__":
    unittest.main()
