"""Integration tests for memory.py search, delete, list, and stats commands.

Tests run inside the managed .venv (invoked via venv's pytest binary).

The _embed function is mocked via patch.object(mem, "_embed") rather than the
string form "memory._embed" because the module is loaded via importlib and is
NOT automatically registered in sys.modules under the name "memory".
"""
import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_SKILL_DIR = Path(__file__).parent
spec = importlib.util.spec_from_file_location("memory", Path(__file__).parent / ".apm/skills/skill-memory/memory.py")
mem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mem)


def _make_vec(seed: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1e6))
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def _similar_vecs():
    """Two close unit vectors with L2 distance ~0.74 (small perturbation then renorm)."""
    base = _make_vec(1.0)
    perturbed = base + np.random.default_rng(42).standard_normal(384).astype(np.float32) * 0.05
    perturbed /= np.linalg.norm(perturbed)
    return base, perturbed


def _insert_memory(db_path, content, source, embedding, created_at=None):
    import uuid
    from datetime import datetime, timezone
    mem_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    con = mem._open_db(db_path)
    try:
        with con:
            cur = con.execute(
                "INSERT INTO memories (id, content, source, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (mem_id, content, source, "{}", created_at),
            )
            rowid = cur.lastrowid
            con.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, mem._serialize_vec(embedding)),
            )
    finally:
        con.close()
    return mem_id


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        mem._init_db(self.db_path)
        self.vec_a, self.vec_b = _similar_vecs()
        self.vec_c = _make_vec(99.0)
        self.id_a = _insert_memory(self.db_path, "cats are great pets", "notes", self.vec_a)
        self.id_b = _insert_memory(self.db_path, "cats love to play", "notes", self.vec_b)
        self.id_c = _insert_memory(self.db_path, "quantum computing", "science", self.vec_c)

    def tearDown(self):
        os.unlink(self.db_path)

    def _search(self, query_vec, **kwargs):
        import io; from contextlib import redirect_stdout
        # query attribute required by cmd_search; actual text is intercepted by the mock
        defaults = dict(db=str(self.db_path), query="test", limit=5, threshold=None, source=None, json=False)
        defaults.update(kwargs)
        # patch.object patches mem directly so cmd_search uses the mock regardless of sys.modules
        with patch.object(mem, "_embed", return_value=np.array([query_vec])):
            buf = io.StringIO()
            with redirect_stdout(buf):
                mem.cmd_search(_Args(**defaults))
            return buf.getvalue()

    def test_returns_results(self):
        self.assertIn(self.id_a, self._search(self.vec_a))

    def test_limit(self):
        lines = [l for l in self._search(self.vec_a, limit=1).strip().splitlines() if l]
        self.assertEqual(len(lines), 1)

    def test_threshold_filters(self):
        # Empirical distances: id_a~=0.0, id_b~=0.74, id_c~=1.43 from vec_a
        # Use tight threshold that keeps id_a/id_b but drops id_c
        out_all = self._search(self.vec_a, limit=5, threshold=1.1)
        self.assertIn(self.id_a, out_all)
        self.assertNotIn(self.id_c, out_all)
        # Zero threshold should exclude id_b and id_c (only exact match at distance 0 passes)
        out_exact = self._search(self.vec_a, limit=5, threshold=0.001)
        self.assertIn(self.id_a, out_exact)
        self.assertNotIn(self.id_b, out_exact)
        self.assertNotIn(self.id_c, out_exact)

    def test_source_filter(self):
        out = self._search(self.vec_a, limit=5, source="science")
        self.assertNotIn(self.id_a, out)
        self.assertIn(self.id_c, out)

    def test_json_output(self):
        import io; from contextlib import redirect_stdout
        buf = io.StringIO()
        with patch.object(mem, "_embed", return_value=np.array([self.vec_a])):
            with redirect_stdout(buf):
                mem.cmd_search(_Args(db=str(self.db_path), query="test", limit=5,
                                     threshold=None, source=None, json=True))
        data = json.loads(buf.getvalue())
        self.assertIsInstance(data, list)
        for r in data:
            for key in ("id", "content", "source", "metadata", "created_at", "distance"):
                self.assertIn(key, r)

    def test_nonexistent_db_exits_0(self):
        import io; from contextlib import redirect_stdout
        with patch.object(mem, "_embed", return_value=np.array([self.vec_a])):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    mem.cmd_search(_Args(db="/tmp/no_agmem.db", query="test", limit=5,
                                         threshold=None, source=None, json=False))
        self.assertEqual(ctx.exception.code, 0)


class TestDelete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        mem._init_db(self.db_path)
        self.mem_id = _insert_memory(self.db_path, "dogs are loyal", "test", _make_vec(2.0))

    def tearDown(self):
        os.unlink(self.db_path)

    def _count(self, table):
        # memories_vec requires sqlite_vec extension
        if table == "memories_vec":
            con = mem._open_db(self.db_path)
        else:
            con = sqlite3.connect(self.db_path)
        try:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            con.close()

    def test_removes_from_both_tables(self):
        import io; from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            mem.cmd_delete(_Args(db=str(self.db_path), id=self.mem_id))
        self.assertEqual(self._count("memories"), 0)
        self.assertEqual(self._count("memories_vec"), 0)

    def test_prints_confirmation(self):
        import io; from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            mem.cmd_delete(_Args(db=str(self.db_path), id=self.mem_id))
        self.assertIn(self.mem_id, buf.getvalue())

    def test_nonexistent_id_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            mem.cmd_delete(_Args(db=str(self.db_path), id="00000000-0000-0000-0000-000000000000"))
        self.assertEqual(ctx.exception.code, 1)

    def test_nonexistent_db_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            mem.cmd_delete(_Args(db="/tmp/no_delete_agmem.db", id="some-id"))
        self.assertEqual(ctx.exception.code, 1)

    def test_deleted_id_absent_from_list(self):
        import io; from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            mem.cmd_delete(_Args(db=str(self.db_path), id=self.mem_id))
        buf = io.StringIO()
        with redirect_stdout(buf):
            mem.cmd_list(_Args(db=str(self.db_path), limit=20, source=None, json=False))
        self.assertNotIn(self.mem_id, buf.getvalue())


class TestList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        mem._init_db(self.db_path)
        self.ids = []
        # Explicit timestamps spread 1 second apart to guarantee deterministic ordering
        base_ts = "2026-01-01T00:00:0{}Z"
        for i in range(5):
            src = "tagA" if i % 2 == 0 else "tagB"
            ts = base_ts.format(i)
            mid = _insert_memory(self.db_path, f"memory number {i}", src, _make_vec(float(i)),
                                  created_at=ts)
            self.ids.append((mid, src))

    def tearDown(self):
        os.unlink(self.db_path)

    def _list(self, **kwargs):
        import io; from contextlib import redirect_stdout
        defaults = dict(db=str(self.db_path), limit=20, source=None, json=False)
        defaults.update(kwargs)
        buf = io.StringIO()
        with redirect_stdout(buf):
            mem.cmd_list(_Args(**defaults))
        return buf.getvalue()

    def test_returns_all(self):
        out = self._list()
        for mid, _ in self.ids:
            self.assertIn(mid, out)

    def test_newest_first(self):
        # ids[4] has timestamp ...04Z (newest), ids[0] has ...00Z (oldest)
        out = self._list()
        self.assertLess(out.index(self.ids[-1][0]), out.index(self.ids[0][0]))

    def test_source_filter(self):
        out = self._list(source="tagA")
        for mid, src in self.ids:
            if src == "tagA":
                self.assertIn(mid, out)
            else:
                self.assertNotIn(mid, out)

    def test_limit(self):
        lines = [l for l in self._list(limit=2).strip().splitlines() if l]
        self.assertEqual(len(lines), 2)

    def test_limit_zero_returns_all(self):
        out = self._list(limit=0)
        for mid, _ in self.ids:
            self.assertIn(mid, out)

    def test_json_output(self):
        import io; from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            mem.cmd_list(_Args(db=str(self.db_path), limit=20, source=None, json=True))
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), len(self.ids))
        for r in data:
            for key in ("id", "content", "source", "metadata", "created_at"):
                self.assertIn(key, r)
        self.assertNotIn("distance", data[0])

    def test_nonexistent_db_exits_0(self):
        import io; from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                mem.cmd_list(_Args(db="/tmp/no_list_agmem.db", limit=20, source=None, json=False))
        self.assertEqual(ctx.exception.code, 0)


class TestStats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        mem._init_db(self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def _stats(self, db=None):
        import io; from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            mem.cmd_stats(_Args(db=db or str(self.db_path)))
        return buf.getvalue()

    def test_empty_db(self):
        out = self._stats()
        self.assertIn("Memories:", out)
        self.assertIn("DB path:", out)
        self.assertIn("DB size:", out)

    def test_populated_db(self):
        _insert_memory(self.db_path, "test", "src", _make_vec(3.0))
        self.assertIn("Memories:  1", self._stats())

    def test_nonexistent_db_exits_0(self):
        import io; from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                mem.cmd_stats(_Args(db="/tmp/no_stats_agmem.db"))
        self.assertEqual(ctx.exception.code, 0)

    def test_shows_db_path(self):
        self.assertIn(str(self.db_path), self._stats())


if __name__ == "__main__":
    unittest.main()
