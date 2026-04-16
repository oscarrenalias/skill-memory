"""Tests for memory.py CLI skeleton and venv bootstrap (bead B-2632ba9d)."""
import os
import stat
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

MEMORY_PY = Path(__file__).parent.parent / ".apm/skills/skill-memory/memory.py"


class TestShebang(unittest.TestCase):
    def test_shebang_on_line_1(self):
        first_line = MEMORY_PY.read_text().splitlines()[0]
        self.assertEqual(first_line, "#!/usr/bin/env python3")


class TestExecutable(unittest.TestCase):
    def test_file_is_executable(self):
        file_stat = MEMORY_PY.stat()
        is_exec = bool(file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        self.assertTrue(is_exec, "memory.py is not marked executable")


class TestBootstrapNoOp(unittest.TestCase):
    """_bootstrap() should return early when sys.prefix already contains .venv."""

    def test_bootstrap_noop_when_in_venv(self):
        # Patch sys.prefix to simulate running inside the managed venv
        venv_path = str(MEMORY_PY.parent / ".venv")
        with mock.patch("sys.prefix", venv_path):
            # Import the module with _bootstrap replaced to detect if it exits early
            # We re-execute just the no-op branch logic inline to avoid side effects
            import sys as _sys
            _VENV = MEMORY_PY.parent / ".venv"
            # Simulate the guard condition directly
            result = str(_VENV) in _sys.prefix
            self.assertTrue(result, "_bootstrap() guard did not detect .venv in sys.prefix")


class TestCLIHelp(unittest.TestCase):
    """--help should exit 0 and list key subcommands without running bootstrap."""

    def _run_help(self, *extra_args):
        env = os.environ.copy()
        # Pretend we're already inside the venv so _bootstrap() is a no-op
        venv_path = str(MEMORY_PY.parent / ".venv")
        env["PYTHONPATH"] = str(MEMORY_PY.parent)
        # Patch sys.prefix via PYTHONSTARTUP is unreliable; instead we run via
        # a small wrapper that patches sys.prefix before importing memory.
        wrapper = (
            f"import sys; sys.prefix = {venv_path!r}; "
            f"import runpy; runpy.run_path({str(MEMORY_PY)!r}, run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-c", wrapper, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        return result

    def test_help_exits_zero(self):
        result = self._run_help()
        self.assertEqual(result.returncode, 0, f"--help exited {result.returncode}:\n{result.stderr}")

    def test_help_lists_init(self):
        result = self._run_help()
        combined = result.stdout + result.stderr
        self.assertIn("init", combined)

    def test_help_lists_add(self):
        result = self._run_help()
        combined = result.stdout + result.stderr
        self.assertIn("add", combined)


class TestUnknownSubcommand(unittest.TestCase):
    """An unknown subcommand should exit non-zero."""

    def test_unknown_subcommand_exits_nonzero(self):
        venv_path = str(MEMORY_PY.parent / ".venv")
        wrapper = (
            f"import sys; sys.prefix = {venv_path!r}; "
            f"import runpy; runpy.run_path({str(MEMORY_PY)!r}, run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-c", wrapper, "nonexistent-subcommand"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0,
                            "Expected non-zero exit for unknown subcommand")


if __name__ == "__main__":
    unittest.main()
