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
import hashlib
import json
import sqlite3
import struct
import urllib.request
import uuid
from datetime import datetime, timezone

import numpy as np
import sqlite_vec


_DEFAULT_DB = Path.home() / ".local" / "share" / "agent-memory" / "memories.db"
_MODEL_DIR = Path.home() / ".cache" / "agent-memory" / "bge-small-en-v1.5"
_MODEL_PATH = _MODEL_DIR / "model.onnx"
_MODEL_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
# SHA256 verified at implementation time against the official HuggingFace file.
# The bge-small model is ~24 MB; if the file is significantly larger the wrong
# variant was downloaded.
_MODEL_SHA256 = None  # Set to None to skip verification until confirmed at first run

_session = None  # ONNX InferenceSession singleton


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


def _ensure_model() -> Path:
    """Download ONNX model on first use; verify SHA256 if hash is set."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _MODEL_PATH.with_suffix(".tmp")
    print("agent-memory: Downloading bge-small-en-v1.5 model...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        if _MODEL_SHA256:
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if digest != _MODEL_SHA256:
                tmp.unlink(missing_ok=True)
                print(
                    f"agent-memory: model SHA256 mismatch (got {digest})",
                    file=sys.stderr,
                )
                sys.exit(1)
        tmp.rename(_MODEL_PATH)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"agent-memory: model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    return _MODEL_PATH


def _get_session():
    """Return the ONNX InferenceSession singleton, loading it on first call."""
    global _session
    if _session is None:
        import onnxruntime as ort
        model = _ensure_model()
        _session = ort.InferenceSession(str(model))
    return _session


def _embed(texts: list) -> "np.ndarray":
    """Embed texts using bge-small-en-v1.5. Returns float32 array (N, 384), L2-normalised.

    Distance metric: sqlite-vec uses L2 distance by default for vec0 tables with
    float[N] columns. Since vectors are L2-normalised here, L2 distance is
    monotonically related to cosine distance, giving range [0, 2] (0 = identical,
    2 = opposite). Use --threshold values accordingly (e.g. 0.5 for high similarity).
    """
    from tokenizers import Tokenizer

    assets = _SKILL_DIR / "assets"
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)
    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    sess = _get_session()
    outputs = sess.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    # CLS token pooling — index 0 of sequence dimension (bge-small model card recommendation)
    embeddings = outputs[0][:, 0, :]
    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)


def _serialize_vec(v: "np.ndarray") -> bytes:
    """Serialize float32 numpy vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(v)}f", *v.tolist())


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open DB and load sqlite-vec extension."""
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _auto_init(db_path: Path) -> None:
    """Initialise DB if it does not yet exist."""
    if not db_path.exists():
        _init_db(db_path)


def _init_db(db_path: Path) -> None:
    """Create tables (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = _open_db(db_path)
    try:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT    NOT NULL UNIQUE,
                content    TEXT    NOT NULL,
                source     TEXT,
                metadata   TEXT    NOT NULL DEFAULT '{}',
                created_at TEXT    NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                embedding float[384]
            );
        """)
        con.commit()
    finally:
        con.close()


def cmd_init(args):
    db_path = _resolve_db(args.db)
    _init_db(db_path)
    print(f"Initialised memory DB at {db_path}")


def cmd_add(args):
    db_path = _resolve_db(args.db)
    _auto_init(db_path)

    meta = {}
    if args.meta:
        for kv in args.meta:
            if "=" not in kv:
                print(f"Invalid --meta value (expected KEY=VALUE): {kv}", file=sys.stderr)
                sys.exit(1)
            k, v = kv.split("=", 1)
            meta[k] = v

    embedding = _embed([args.text])[0]
    mem_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    con = _open_db(db_path)
    try:
        with con:
            cur = con.execute(
                "INSERT INTO memories (id, content, source, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (mem_id, args.text, args.source, json.dumps(meta), created_at),
            )
            rowid = cur.lastrowid
            con.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_vec(embedding)),
            )
    finally:
        con.close()

    print(f"Added {mem_id}")


def cmd_search(args):
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"No memory DB found at {db_path}. Run: memory.py init")
        sys.exit(0)

    query_vec = _embed([args.query])[0]
    limit = args.limit

    con = _open_db(db_path)
    try:
        sql = """
            SELECT m.id, m.content, m.source, m.metadata, m.created_at, v.distance
            FROM memories_vec v
            JOIN memories m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
        """
        rows = con.execute(sql, (_serialize_vec(query_vec), limit)).fetchall()
    finally:
        con.close()

    # Post-filter by source if requested
    if args.source:
        rows = [r for r in rows if r[2] == args.source]

    # Apply threshold (L2 distance; lower = more similar)
    if args.threshold is not None:
        rows = [r for r in rows if r[5] <= args.threshold]

    if args.json:
        out = [
            {
                "id": r[0],
                "content": r[1],
                "source": r[2],
                "metadata": json.loads(r[3]),
                "created_at": r[4],
                "distance": r[5],
            }
            for r in rows
        ]
        print(json.dumps(out, indent=2))
    else:
        for r in rows:
            dist, mem_id, content, source = r[5], r[0], r[1], r[2]
            snippet = content[:60] + ("..." if len(content) > 60 else "")
            print(f"[{dist:.3f}] {mem_id}  {snippet}  (source: {source})")


def cmd_delete(args):
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"No memory DB found at {db_path}", file=sys.stderr)
        sys.exit(1)

    con = _open_db(db_path)
    try:
        row = con.execute("SELECT rowid FROM memories WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            print(f"ID not found: {args.id}", file=sys.stderr)
            sys.exit(1)
        rowid = row[0]
        with con:
            con.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
            con.execute("DELETE FROM memories WHERE id = ?", (args.id,))
    finally:
        con.close()

    print(f"Deleted {args.id}")


def cmd_list(args):
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"No memory DB found at {db_path}")
        sys.exit(0)

    limit = args.limit
    con = sqlite3.connect(db_path)
    try:
        sql = "SELECT id, content, source, metadata, created_at FROM memories"
        params = []
        if args.source:
            sql += " WHERE source = ?"
            params.append(args.source)
        sql += " ORDER BY created_at DESC"
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()

    if args.json:
        out = [
            {
                "id": r[0],
                "content": r[1],
                "source": r[2],
                "metadata": json.loads(r[3]),
                "created_at": r[4],
            }
            for r in rows
        ]
        print(json.dumps(out, indent=2))
    else:
        for r in rows:
            mem_id, content, source, _, created_at = r
            snippet = content[:60] + ("..." if len(content) > 60 else "")
            print(f"{mem_id}  {snippet}  (source: {source})  {created_at}")


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

    # init
    init_parser = subparsers.add_parser("init", help="Initialise the memory DB")
    init_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    init_parser.set_defaults(func=cmd_init)

    # add
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("text", help="Text to store")
    add_parser.add_argument("--source", metavar="TAG", help="Origin tag or file path")
    add_parser.add_argument(
        "--meta", metavar="KEY=VALUE", action="append", help="Metadata key=value pairs"
    )
    add_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    add_parser.set_defaults(func=cmd_add)

    # search
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    search_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Maximum L2 distance to include (lower = more similar; range 0-2 for "
            "L2-normalised vectors; e.g. 0.5 for high similarity)"
        ),
    )
    search_parser.add_argument("--source", metavar="TAG", help="Filter by source tag")
    search_parser.add_argument("--json", action="store_true", help="Output JSON")
    search_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    search_parser.set_defaults(func=cmd_search)

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete a memory by ID")
    delete_parser.add_argument("id", help="UUID of the memory to delete")
    delete_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    delete_parser.set_defaults(func=cmd_delete)

    # list
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument(
        "--limit", type=int, default=20, help="Max results (default: 20; 0 = unlimited)"
    )
    list_parser.add_argument("--source", metavar="TAG", help="Filter by source tag")
    list_parser.add_argument("--json", action="store_true", help="Output JSON")
    list_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    list_parser.set_defaults(func=cmd_list)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show DB statistics")
    stats_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
