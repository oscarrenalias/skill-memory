#!/usr/bin/env python3
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
        print("agent-memory: first run, installing dependencies…", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "venv", str(_VENV)])
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "--quiet", *_DEPS])
    os.execv(str(venv_py), [str(venv_py)] + sys.argv)


_bootstrap()

import argparse  # noqa: E402 — must come after bootstrap
import json
import sqlite3
import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path.home() / ".local" / "share" / "agent-memory" / "memories.db"


def _resolve_db(args: argparse.Namespace) -> Path:
    """Resolve DB path: --db flag > AGENT_MEMORY_DB env var > default."""
    raw = getattr(args, "db", None) or os.environ.get("AGENT_MEMORY_DB") or str(_DEFAULT_DB)
    return Path(raw)


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

_DDL_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT    NOT NULL UNIQUE,
    content    TEXT    NOT NULL,
    source     TEXT,
    metadata   TEXT    NOT NULL DEFAULT '{}',
    created_at TEXT    NOT NULL
);
"""

_DDL_MEMORIES_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    embedding float[384]
);
"""


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the DB, load sqlite-vec, create tables. Returns open connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Load sqlite-vec extension
    try:
        import sqlite_vec  # noqa: PLC0415
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        print(f"error: could not load sqlite-vec extension: {exc}", file=sys.stderr)
        sys.exit(1)
    conn.executescript(_DDL_MEMORIES + _DDL_MEMORIES_VEC)
    conn.commit()
    return conn


def _auto_init(args: argparse.Namespace) -> tuple[sqlite3.Connection, Path]:
    """Auto-initialise the DB and return (connection, db_path)."""
    db_path = _resolve_db(args)
    conn = _init_db(db_path)
    return conn, db_path


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    db_path = _resolve_db(args)
    _init_db(db_path)
    print(f"Initialised memory DB at {db_path}")


def cmd_add(args: argparse.Namespace) -> None:
    raise NotImplementedError("add not yet implemented")


def cmd_search(args: argparse.Namespace) -> None:
    raise NotImplementedError("search not yet implemented")


def cmd_delete(args: argparse.Namespace) -> None:
    raise NotImplementedError("delete not yet implemented")


def cmd_list(args: argparse.Namespace) -> None:
    raise NotImplementedError("list not yet implemented")


def cmd_stats(args: argparse.Namespace) -> None:
    db_path = _resolve_db(args)
    if not db_path.exists():
        print(f"No memory DB found at {db_path}")
        return
    conn = _init_db(db_path)
    (count,) = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    size_bytes = db_path.stat().st_size
    if size_bytes >= 1_048_576:
        size_str = f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes} B"
    print(f"DB path:   {db_path}")
    print(f"Memories:  {count}")
    print(f"DB size:   {size_str}")
    conn.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="memory.py",
        description="agent-memory: semantic long-term memory for Claude agents",
    )
    parser.add_argument("--db", metavar="PATH", help="Override DB path")
    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")
    sub.required = True

    # init
    p_init = sub.add_parser("init", help="Initialise the memory database")
    p_init.add_argument("--db", metavar="PATH", help="Override DB path")
    p_init.set_defaults(func=cmd_init)

    # add
    p_add = sub.add_parser("add", help="Add a memory entry")
    p_add.add_argument("text", help="Text content to store")
    p_add.add_argument("--source", metavar="TAG", help="Optional origin tag or file path")
    p_add.add_argument("--meta", metavar="KEY=VALUE", action="append", default=[], help="Arbitrary metadata (repeatable)")
    p_add.add_argument("--db", metavar="PATH", help="Override DB path")
    p_add.set_defaults(func=cmd_add)

    # search
    p_search = sub.add_parser("search", help="Semantic search over memories")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--limit", type=int, default=5, metavar="N", help="Max results (default: 5)")
    p_search.add_argument("--threshold", type=float, metavar="F", help="Max distance filter")
    p_search.add_argument("--source", metavar="TAG", help="Filter by source tag")
    p_search.add_argument("--json", action="store_true", help="Output as JSON array")
    p_search.add_argument("--db", metavar="PATH", help="Override DB path")
    p_search.set_defaults(func=cmd_search)

    # delete
    p_delete = sub.add_parser("delete", help="Delete a memory by ID")
    p_delete.add_argument("id", help="Memory UUID to delete")
    p_delete.add_argument("--db", metavar="PATH", help="Override DB path")
    p_delete.set_defaults(func=cmd_delete)

    # list
    p_list = sub.add_parser("list", help="List stored memories")
    p_list.add_argument("--limit", type=int, default=20, metavar="N", help="Max results (default: 20; 0 = all)")
    p_list.add_argument("--source", metavar="TAG", help="Filter by source tag")
    p_list.add_argument("--json", action="store_true", help="Output as JSON array")
    p_list.add_argument("--db", metavar="PATH", help="Override DB path")
    p_list.set_defaults(func=cmd_list)

    # stats
    p_stats = sub.add_parser("stats", help="Show DB statistics")
    p_stats.add_argument("--db", metavar="PATH", help="Override DB path")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
