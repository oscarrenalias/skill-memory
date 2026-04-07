#!/usr/bin/env python3
"""agent-memory CLI — long-term semantic memory for agents."""
import sys
import os
import subprocess
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_VENV = _SKILL_DIR / ".venv"
_DEPS = [
    "onnxruntime>=1.17.0",
    "sqlite-vec>=0.1.6",
    "tokenizers>=0.19.0",
    "numpy>=1.24.0",
]


def _bootstrap() -> None:
    if str(_VENV) in sys.prefix:
        return  # already running inside the managed venv
    venv_py = _VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not venv_py.exists():
        print("agent-memory: first run, installing dependencies...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "venv", str(_VENV)])
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "--quiet", *_DEPS])
    os.execv(str(venv_py), [str(venv_py)] + sys.argv)


_bootstrap()

import argparse
import sqlite3


_DEFAULT_DB = Path.home() / ".local" / "share" / "agent-memory" / "memories.db"


def _resolve_db(db_arg):
    """Resolve DB path: CLI flag > env var > default."""
    if db_arg:
        return Path(db_arg)
    env = os.environ.get("AGENT_MEMORY_DB")
    if env:
        return Path(env)
    return _DEFAULT_DB


def _format_size(num_bytes):
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def cmd_stats(args):
    db_path = _resolve_db(args.db)

    if not db_path.exists():
        print(f"No memory DB found at {db_path}")
        sys.exit(0)

    size_bytes = os.path.getsize(db_path)

    con = sqlite3.connect(db_path)
    try:
        (count,) = con.execute("SELECT COUNT(*) FROM memories").fetchone()
    finally:
        con.close()

    print(f"DB path:   {db_path}")
    print(f"Memories:  {count}")
    print(f"DB size:   {_format_size(size_bytes)}")


def main():
    parser = argparse.ArgumentParser(
        prog="memory.py",
        description="agent-memory — long-term semantic memory for agents",
    )
    parser.add_argument("--db", metavar="PATH", help="Override DB path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show DB statistics")
    stats_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
