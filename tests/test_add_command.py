"""Integration tests for memory.py add command (bead B-a4f7357a).

Coverage:
  (1) add prints 'Added <uuid-v4>'
  (2) two add calls produce distinct UUIDs
  (3) add inserts rows in both memories and memories_vec tables
  (4) auto-init: add works without a prior init call
  (5) --source and --meta are persisted to the DB
  (6) transaction rollback: memories unchanged when memories_vec insert fails
  (7) SHA256 failure path: exit code 1, partial file deleted
  (8) embedding shape (1, 384) and L2 norm ≈ 1.0 (venv-gated)
"""
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

MEMORY_PY = Path(__file__).parent.parent / "memory.py"
_VENV_PY = MEMORY_PY.parent / ".venv" / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
)
_VENV_PATH = str(MEMORY_PY.parent / ".venv")
_venv_ready = _VENV_PY.exists()

UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRow:
    """Minimal ndarray-like row returned by the stub embedder."""
    def __init__(self, dim=384):
        import struct
        # single unit vector: first component = 1.0, rest 0.0
        self._data = struct.pack(f"<{dim}f", 1.0, *([0.0] * (dim - 1)))

    def tobytes(self) -> bytes:
        return self._data


class _FakeEmbeddings:
    def __init__(self, n: int = 1, dim: int = 384):
        self._rows = [_FakeRow(dim) for _ in range(n)]

    def __getitem__(self, idx):
        return self._rows[idx]


# Inline source that patches _bootstrap, _init_db (full tables), and _embed
_STUB_FULL = textwrap.dedent(f"""\
    import sys
    sys.prefix = {_VENV_PATH!r}

    import importlib.util, pathlib, sqlite3, struct

    def _stub_init_db(db_path):
        db_path = pathlib.Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS memories "
            "(id TEXT NOT NULL UNIQUE, content TEXT NOT NULL, source TEXT, "
            "metadata TEXT NOT NULL DEFAULT '{{}}', created_at TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS memories_vec "
            "(rowid INTEGER PRIMARY KEY, embedding BLOB);"
        )
        conn.commit()
        return conn

    class _FakeRow:
        def __init__(self, dim=384):
            self._data = struct.pack(f'<{{dim}}f', 1.0, *([0.0] * (dim - 1)))
        def tobytes(self): return self._data

    class _FakeEmbeddings:
        def __init__(self, n=1): self._rows = [_FakeRow() for _ in range(n)]
        def __getitem__(self, i): return self._rows[i]

    def _stub_embed(texts): return _FakeEmbeddings(len(texts))

    _spec = importlib.util.spec_from_file_location('memory', {str(MEMORY_PY)!r})
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['memory'] = _mod
    _spec.loader.exec_module(_mod)
    _mod._init_db = _stub_init_db
    _mod._embed = _stub_embed
""")

# Stub that creates ONLY memories (no memories_vec) — used for rollback test
_STUB_NO_VEC = textwrap.dedent(f"""\
    import sys
    sys.prefix = {_VENV_PATH!r}

    import importlib.util, pathlib, sqlite3, struct

    def _stub_init_db_no_vec(db_path):
        db_path = pathlib.Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories "
            "(id TEXT NOT NULL UNIQUE, content TEXT NOT NULL, source TEXT, "
            "metadata TEXT NOT NULL DEFAULT '{{}}', created_at TEXT NOT NULL)"
        )
        conn.commit()
        return conn

    class _FakeRow:
        def __init__(self): self._data = struct.pack('<384f', *([0.0] * 384))
        def tobytes(self): return self._data

    class _FakeEmbeddings:
        def __init__(self, n=1): self._rows = [_FakeRow() for _ in range(n)]
        def __getitem__(self, i): return self._rows[i]

    def _stub_embed(texts): return _FakeEmbeddings(len(texts))

    _spec = importlib.util.spec_from_file_location('memory', {str(MEMORY_PY)!r})
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['memory'] = _mod
    _spec.loader.exec_module(_mod)
    _mod._init_db = _stub_init_db_no_vec
    _mod._embed = _stub_embed
""")


def _run_add(extra_args=(), env=None, stub_src=None):
    """Run 'memory.py add' via subprocess with the given stub prefix."""
    run_env = os.environ.copy()
    run_env.pop("AGENT_MEMORY_DB", None)
    if env:
        run_env.update(env)

    src = (stub_src or _STUB_FULL) + "_mod.main()\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(src)
        script_path = fh.name
    try:
        return subprocess.run(
            [sys.executable, script_path, "add", *extra_args],
            capture_output=True,
            text=True,
            env=run_env,
        )
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# (1) add prints 'Added <uuid-v4>'
# ---------------------------------------------------------------------------

class TestAddOutput(unittest.TestCase):
    def test_add_prints_added_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(["hello world", "--db", str(db_path)])
            self.assertEqual(
                result.returncode, 0,
                f"add exited {result.returncode}:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertTrue(
                result.stdout.strip().startswith("Added "),
                f"Expected 'Added ...' in stdout; got: {result.stdout!r}",
            )

    def test_add_output_is_uuid_v4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(["hello world", "--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            line = result.stdout.strip()
            self.assertTrue(line.startswith("Added "), f"Unexpected output: {line!r}")
            uid = line[len("Added "):]
            self.assertRegex(
                uid, UUID_V4_RE,
                f"Output ID {uid!r} is not a valid UUID v4",
            )


# ---------------------------------------------------------------------------
# (2) two add calls produce distinct UUIDs
# ---------------------------------------------------------------------------

class TestAddDistinctUUIDs(unittest.TestCase):
    def test_two_adds_produce_distinct_uuids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            uids = []
            for text in ("first memory", "second memory"):
                result = _run_add([text, "--db", str(db_path)])
                self.assertEqual(
                    result.returncode, 0,
                    f"add failed:\nstdout={result.stdout}\nstderr={result.stderr}",
                )
                uid = result.stdout.strip()[len("Added "):]
                self.assertRegex(uid, UUID_V4_RE)
                uids.append(uid)
            self.assertNotEqual(
                uids[0], uids[1],
                f"Both add calls produced the same UUID: {uids[0]!r}",
            )


# ---------------------------------------------------------------------------
# (3) add inserts rows in both tables
# ---------------------------------------------------------------------------

class TestAddDualTableRows(unittest.TestCase):
    def test_add_inserts_row_in_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(["test content", "--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("SELECT id, content FROM memories").fetchall()
            conn.close()
            self.assertEqual(len(rows), 1, f"Expected 1 row in memories; got {len(rows)}")
            self.assertEqual(rows[0][1], "test content")

    def test_add_inserts_row_in_memories_vec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(["test content", "--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
            conn.close()
            self.assertEqual(count, 1, f"Expected 1 row in memories_vec; got {count}")

    def test_two_adds_produce_two_rows_in_each_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            for text in ("first", "second"):
                result = _run_add([text, "--db", str(db_path)])
                self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            vec_count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
            conn.close()
            self.assertEqual(mem_count, 2, f"Expected 2 rows in memories; got {mem_count}")
            self.assertEqual(vec_count, 2, f"Expected 2 rows in memories_vec; got {vec_count}")


# ---------------------------------------------------------------------------
# (4) auto-init: add without prior init
# ---------------------------------------------------------------------------

class TestAddAutoInit(unittest.TestCase):
    def test_add_auto_creates_db_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "auto.db"
            self.assertFalse(db_path.exists(), "DB should not exist before add")
            result = _run_add(["auto init test", "--db", str(db_path)])
            self.assertEqual(
                result.returncode, 0,
                f"add failed:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertTrue(db_path.exists(), "DB file should have been created by add auto-init")

    def test_add_auto_creates_memories_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "auto.db"
            result = _run_add(["auto init test", "--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.close()
            self.assertIn("memories", tables)


# ---------------------------------------------------------------------------
# (5) --source and --meta are persisted
# ---------------------------------------------------------------------------

class TestAddSourceMeta(unittest.TestCase):
    def test_source_is_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(
                ["some content", "--source", "test-tag", "--db", str(db_path)]
            )
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT source FROM memories").fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "test-tag", f"source mismatch: {row[0]!r}")

    def test_meta_is_stored_as_json(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(
                ["some content", "--meta", "key1=value1", "--meta", "key2=value2",
                 "--db", str(db_path)]
            )
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT metadata FROM memories").fetchone()
            conn.close()
            self.assertIsNotNone(row)
            meta = json.loads(row[0])
            self.assertEqual(meta.get("key1"), "value1", f"key1 missing/wrong in {meta!r}")
            self.assertEqual(meta.get("key2"), "value2", f"key2 missing/wrong in {meta!r}")

    def test_source_none_when_not_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(["no source", "--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT source FROM memories").fetchone()
            conn.close()
            self.assertIsNone(row[0], f"Expected NULL source; got {row[0]!r}")

    def test_empty_meta_stored_as_empty_object(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = _run_add(["no meta", "--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT metadata FROM memories").fetchone()
            conn.close()
            self.assertIsNotNone(row)
            meta = json.loads(row[0])
            self.assertEqual(meta, {}, f"Expected empty dict; got {meta!r}")


# ---------------------------------------------------------------------------
# (6) Transaction rollback: memories unchanged when memories_vec insert fails
# ---------------------------------------------------------------------------

class TestAddTransactionRollback(unittest.TestCase):
    def test_rollback_leaves_memories_empty(self):
        """
        Use a stub that creates ONLY memories (no memories_vec). The second
        INSERT in cmd_add will raise OperationalError, which must roll back
        the first INSERT under the `with conn:` transaction block.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rollback.db"
            result = _run_add(
                ["rollback test", "--db", str(db_path)],
                stub_src=_STUB_NO_VEC,
            )
            # Expect non-zero exit (vec insert will fail)
            self.assertNotEqual(
                result.returncode, 0,
                "Expected failure when memories_vec table is missing",
            )
            # Verify the memories table exists but is empty (rolled back)
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            conn.close()
            self.assertEqual(
                count, 0,
                f"Expected 0 rows in memories after rollback; got {count}",
            )


# ---------------------------------------------------------------------------
# (7) SHA256 failure path: exit code 1, partial file deleted
# ---------------------------------------------------------------------------

class TestSHA256FailurePath(unittest.TestCase):
    def test_sha256_mismatch_exits_1_and_deletes_file(self):
        """
        Stub urlretrieve to write a file with wrong content so SHA256 fails.
        Expect exit code 1 and the partial file to be absent.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_model_path = Path(tmpdir) / "model.onnx"
            sha256_stub = textwrap.dedent(f"""\
                import sys
                sys.prefix = {_VENV_PATH!r}

                import importlib.util, pathlib, urllib.request

                _fake_model_path = pathlib.Path({str(fake_model_path)!r})

                _original_urlretrieve = urllib.request.urlretrieve

                def _fake_urlretrieve(url, dst):
                    dst = pathlib.Path(dst)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(b"definitely not the real model")

                urllib.request.urlretrieve = _fake_urlretrieve

                _spec = importlib.util.spec_from_file_location('memory', {str(MEMORY_PY)!r})
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules['memory'] = _mod
                _spec.loader.exec_module(_mod)

                # Override _MODEL_PATH after loading so _ensure_model writes to our temp dir
                _mod._MODEL_PATH = _fake_model_path

                _mod._ensure_model()
            """)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as fh:
                fh.write(sha256_stub)
                script_path = fh.name

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                )
            finally:
                os.unlink(script_path)

            self.assertEqual(
                result.returncode, 1,
                f"Expected exit code 1 on SHA256 mismatch; got {result.returncode}\n"
                f"stderr: {result.stderr}",
            )
            self.assertFalse(
                fake_model_path.exists(),
                f"Partial model file should have been deleted but still exists at {fake_model_path}",
            )
            self.assertIn("SHA256", result.stderr, f"Expected SHA256 mention in stderr: {result.stderr!r}")


# ---------------------------------------------------------------------------
# (8) Embedding shape and L2 norm (requires managed venv)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_venv_ready, "managed venv not set up — skipping embedding integration tests")
class TestEmbeddingShape(unittest.TestCase):
    def test_embed_returns_shape_1_384(self):
        script = textwrap.dedent(f"""\
            import sys, importlib.util
            _spec = importlib.util.spec_from_file_location('memory', {str(MEMORY_PY)!r})
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules['memory'] = _mod
            _spec.loader.exec_module(_mod)
            import numpy as np
            result = _mod._embed(["hello world"])
            assert result.shape == (1, 384), f"Expected shape (1, 384), got {{result.shape}}"
            norm = np.linalg.norm(result[0])
            assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ≈ 1.0, got {{norm}}"
            print("OK")
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
            fh.write(script)
            script_path = fh.name
        try:
            result = subprocess.run(
                [str(_VENV_PY), script_path],
                capture_output=True,
                text=True,
            )
        finally:
            os.unlink(script_path)
        self.assertEqual(
            result.returncode, 0,
            f"Embedding test failed:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
