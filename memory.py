#!/usr/bin/env python3
import sys
import os
import subprocess
import hashlib
import urllib.request
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
# Model bootstrap
# ---------------------------------------------------------------------------

_MODEL_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
_MODEL_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
_MODEL_PATH = Path.home() / ".cache" / "agent-memory" / "bge-small-en-v1.5" / "model.onnx"


def _sha256_of(path: Path) -> str:
    """Compute SHA256 hex digest of a file in streaming fashion."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model() -> Path:
    """Download the ONNX model if absent; verify SHA256 and clean up on failure."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading bge-small-en-v1.5 model…", file=sys.stderr)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        if _MODEL_SHA256:
            actual = _sha256_of(_MODEL_PATH)
            if actual != _MODEL_SHA256:
                _MODEL_PATH.unlink(missing_ok=True)
                print(
                    f"error: model SHA256 mismatch\n  expected: {_MODEL_SHA256}\n  got:      {actual}",
                    file=sys.stderr,
                )
                sys.exit(1)
    except SystemExit:
        raise
    except Exception as exc:
        if _MODEL_PATH.exists():
            _MODEL_PATH.unlink()
        print(f"error: failed to download model: {exc}", file=sys.stderr)
        sys.exit(1)
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------

_ort_session = None
_embed_tokenizer = None


def _get_ort_session():
    """Return the module-level ONNX InferenceSession singleton, creating it if needed."""
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort  # noqa: PLC0415
        model_path = _ensure_model()
        _ort_session = ort.InferenceSession(str(model_path))
    return _ort_session


def _get_embed_tokenizer():
    """Return the module-level tokenizer singleton, creating it if needed."""
    global _embed_tokenizer
    if _embed_tokenizer is None:
        from tokenizers import Tokenizer  # noqa: PLC0415
        tokenizer_path = _SKILL_DIR / "assets" / "tokenizer.json"
        tok = Tokenizer.from_file(str(tokenizer_path))
        tok.enable_truncation(max_length=512)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        _embed_tokenizer = tok
    return _embed_tokenizer


def _embed(texts: list) -> "np.ndarray":
    """Embed texts using bge-small-en-v1.5 ONNX model.

    Returns float32 ndarray of shape (N, 384), L2-normalised (CLS token pooling).
    The ONNX session and tokenizer are module-level singletons loaded once per process.
    """
    import numpy as np  # noqa: PLC0415

    tokenizer = _get_embed_tokenizer()
    session = _get_ort_session()

    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
    token_type_ids = np.array([enc.type_ids for enc in encodings], dtype=np.int64)

    outputs = session.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    last_hidden_state = outputs[0]  # shape (N, seq_len, 384)
    cls_embeddings = last_hidden_state[:, 0, :]  # CLS token pooling -> (N, 384)

    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    return (cls_embeddings / norms).astype(np.float32)


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
    conn, _ = _auto_init(args)
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Parse --meta KEY=VALUE pairs into a dict
    meta: dict = {}
    for kv in args.meta:
        if "=" not in kv:
            print(f"error: --meta value must be KEY=VALUE, got: {kv!r}", file=sys.stderr)
            sys.exit(1)
        k, v = kv.split("=", 1)
        meta[k] = v

    # Embed the text
    embeddings = _embed([args.text])
    vec_bytes = embeddings[0].tobytes()

    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO memories (id, content, source, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, args.text, args.source, json.dumps(meta), now),
            )
            rowid = cur.lastrowid
            conn.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, vec_bytes),
            )
    finally:
        conn.close()

    print(f"Added {memory_id}")


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
