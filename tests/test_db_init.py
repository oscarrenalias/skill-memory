"""Tests for memory.py DB schema and init command (bead B-27a23ff7).

Coverage:
  (1) init creates DB at default path
  (2) init is idempotent (safe to run twice)
  (3) --db flag overrides path
  (4) AGENT_MEMORY_DB env var overrides path
  (5) output matches exact string 'Initialised memory DB at <path>'
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MEMORY_PY = Path(__file__).parent.parent / ".apm/skills/skill-memory/memory.py"
_VENV_PY = MEMORY_PY.parent / ".venv" / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VENV_PATH = str(MEMORY_PY.parent / ".venv")


def _run_init(extra_args=(), env=None, use_stub=True):
    """Run 'memory.py init' via subprocess.

    use_stub=True (default): replaces _init_db with a lightweight stub that
    creates the DB file and the 'memories' table without requiring sqlite_vec.
    This allows path-resolution and output-format tests to run without the
    managed venv.

    use_stub=False: uses the real _init_db. Requires sqlite_vec to be
    installed in the managed venv.
    """
    run_env = os.environ.copy()
    # Always clear AGENT_MEMORY_DB first so tests start from a clean slate
    run_env.pop("AGENT_MEMORY_DB", None)
    if env:
        run_env.update(env)

    if use_stub:
        # Write a small wrapper script to a temp file to avoid `-c` one-liner
        # limitations (Python forbids `def` after `;` in a compound statement).
        script_src = f"""\
import sys
sys.prefix = {_VENV_PATH!r}

import importlib.util
import pathlib
import sqlite3

def _stub_init_db(db_path):
    \"\"\"Minimal stub: creates DB + memories table without sqlite_vec.\"\"\"
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories "
        "(id TEXT NOT NULL UNIQUE, content TEXT NOT NULL, source TEXT, "
        "metadata TEXT NOT NULL DEFAULT '{{}}', created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

_spec = importlib.util.spec_from_file_location('memory', {str(MEMORY_PY)!r})
_mod = importlib.util.module_from_spec(_spec)
sys.modules['memory'] = _mod
_spec.loader.exec_module(_mod)
_mod._init_db = _stub_init_db
_mod.main()
"""
        import tempfile as _tf
        with _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
            fh.write(script_src)
            script_path = fh.name
        try:
            return subprocess.run(
                [sys.executable, script_path, "init", *extra_args],
                capture_output=True,
                text=True,
                env=run_env,
            )
        finally:
            os.unlink(script_path)
    else:
        return subprocess.run(
            [str(_VENV_PY), str(MEMORY_PY), "init", *extra_args],
            capture_output=True,
            text=True,
            env=run_env,
        )


# ---------------------------------------------------------------------------
# (3) --db flag overrides path
# ---------------------------------------------------------------------------

class TestDbFlagOverridesPath(unittest.TestCase):
    def test_db_flag_sets_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "custom.db"
            result = _run_init(extra_args=["--db", str(db_path)])
            self.assertEqual(
                result.returncode, 0,
                f"init exited {result.returncode}:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            expected_output = f"Initialised memory DB at {db_path}"
            self.assertIn(
                expected_output,
                result.stdout,
                f"Expected '{expected_output}' in stdout; got:\n{result.stdout}",
            )

    def test_db_flag_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "test.db"
            result = _run_init(extra_args=["--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            self.assertTrue(db_path.exists(), f"DB file not created at {db_path}")


# ---------------------------------------------------------------------------
# (4) AGENT_MEMORY_DB env var overrides path
# ---------------------------------------------------------------------------

class TestEnvVarOverridesPath(unittest.TestCase):
    def test_env_var_sets_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "env.db"
            result = _run_init(env={"AGENT_MEMORY_DB": str(db_path)})
            self.assertEqual(
                result.returncode, 0,
                f"init exited {result.returncode}:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            expected_output = f"Initialised memory DB at {db_path}"
            self.assertIn(
                expected_output,
                result.stdout,
                f"Expected '{expected_output}' in stdout; got:\n{result.stdout}",
            )

    def test_env_var_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "env.db"
            result = _run_init(env={"AGENT_MEMORY_DB": str(db_path)})
            self.assertEqual(result.returncode, 0)
            self.assertTrue(db_path.exists(), f"DB file not created at {db_path}")

    def test_db_flag_takes_precedence_over_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = Path(tmpdir) / "flag.db"
            env_path = Path(tmpdir) / "env.db"
            result = _run_init(
                extra_args=["--db", str(flag_path)],
                env={"AGENT_MEMORY_DB": str(env_path)},
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn(str(flag_path), result.stdout)
            self.assertNotIn(str(env_path), result.stdout)


# ---------------------------------------------------------------------------
# (5) Output matches exact string 'Initialised memory DB at <path>'
# ---------------------------------------------------------------------------

class TestInitOutputFormat(unittest.TestCase):
    def test_exact_output_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "output_test.db"
            result = _run_init(extra_args=["--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            lines = result.stdout.strip().splitlines()
            self.assertEqual(
                len(lines), 1,
                f"Expected exactly one output line; got {len(lines)}: {lines!r}",
            )
            self.assertEqual(
                lines[0],
                f"Initialised memory DB at {db_path}",
                f"Output line did not match expected format: {lines[0]!r}",
            )

    def test_no_unexpected_stderr_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stderr_test.db"
            result = _run_init(extra_args=["--db", str(db_path)])
            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                result.stderr.strip(), "",
                f"Unexpected stderr on success: {result.stderr!r}",
            )


# ---------------------------------------------------------------------------
# (1) init creates DB — integration tests (require sqlite_vec in managed venv)
# ---------------------------------------------------------------------------

_venv_ready = _VENV_PY.exists()


@unittest.skipUnless(_venv_ready, "managed venv not set up — skipping integration tests")
class TestInitCreatesDb(unittest.TestCase):
    def test_init_creates_db_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "new.db"
            result = subprocess.run(
                [str(_VENV_PY), str(MEMORY_PY), "init", "--db", str(db_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"init failed:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertTrue(db_path.exists(), "DB file was not created")

    def test_init_creates_memories_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "schema.db"
            result = subprocess.run(
                [str(_VENV_PY), str(MEMORY_PY), "init", "--db", str(db_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.close()
            self.assertIn("memories", tables, f"'memories' table not found; got {tables}")

    def test_init_creates_memories_vec_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "vec.db"
            result = subprocess.run(
                [str(_VENV_PY), str(MEMORY_PY), "init", "--db", str(db_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            conn = sqlite3.connect(str(db_path))
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.close()
            self.assertIn(
                "memories_vec", tables,
                f"'memories_vec' virtual table not found; got {tables}",
            )


# ---------------------------------------------------------------------------
# (2) init is idempotent — integration tests (require sqlite_vec in managed venv)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_venv_ready, "managed venv not set up — skipping integration tests")
class TestInitIdempotent(unittest.TestCase):
    def test_init_twice_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "idempotent.db"
            for _ in range(2):
                result = subprocess.run(
                    [str(_VENV_PY), str(MEMORY_PY), "init", "--db", str(db_path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"init failed on repeat run:\nstdout={result.stdout}\nstderr={result.stderr}",
                )

    def test_init_twice_produces_same_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "idempotent2.db"
            outputs = []
            for _ in range(2):
                result = subprocess.run(
                    [str(_VENV_PY), str(MEMORY_PY), "init", "--db", str(db_path)],
                    capture_output=True,
                    text=True,
                )
                outputs.append(result.stdout.strip())
            self.assertEqual(outputs[0], outputs[1], "Output changed between first and second init")
            self.assertEqual(
                outputs[0],
                f"Initialised memory DB at {db_path}",
            )

    def test_init_twice_does_not_duplicate_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "no_dupe.db"
            for _ in range(2):
                subprocess.run(
                    [str(_VENV_PY), str(MEMORY_PY), "init", "--db", str(db_path)],
                    capture_output=True,
                    text=True,
                )
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            conn.close()
            self.assertEqual(count, 0, "Unexpected rows in memories after double init")


# ---------------------------------------------------------------------------
# (2-stub) Idempotency via stub (no venv required)
# ---------------------------------------------------------------------------

class TestInitIdempotentStub(unittest.TestCase):
    """Verify idempotency at the subprocess/output level without sqlite_vec."""

    def test_init_twice_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "idem.db"
            for i in range(2):
                result = _run_init(extra_args=["--db", str(db_path)])
                self.assertEqual(
                    result.returncode, 0,
                    f"Run {i + 1} failed:\nstdout={result.stdout}\nstderr={result.stderr}",
                )

    def test_init_twice_same_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "idem2.db"
            outputs = []
            for _ in range(2):
                result = _run_init(extra_args=["--db", str(db_path)])
                outputs.append(result.stdout.strip())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], f"Initialised memory DB at {db_path}")


if __name__ == "__main__":
    unittest.main()
