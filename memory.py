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
import re
import sqlite3
import struct
import urllib.request
import uuid
from datetime import datetime, timezone

import numpy as np
import sqlite_vec


_DEFAULT_DB = _SKILL_DIR / "memories.db"
_MODEL_DIR = Path.home() / ".cache" / "agent-memory" / "bge-small-en-v1.5"
_MODEL_PATH = _MODEL_DIR / "model.onnx"
_MODEL_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
_MODEL_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"

_ort_session = None  # ONNX InferenceSession singleton
_embed_tokenizer = None  # tokenizers.Tokenizer singleton


def _resolve_db(db_arg):
    if db_arg:
        return Path(db_arg)
    env = os.environ.get("AGENT_MEMORY_DB")
    if env:
        return Path(env)
    return _DEFAULT_DB


def _resolve_namespace(namespace_arg):
    return namespace_arg if namespace_arg else "default"


def _format_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MODEL_PATH.with_suffix(".tmp")
    print("agent-memory: Downloading bge-small-en-v1.5 model...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        if _MODEL_SHA256:
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if digest != _MODEL_SHA256:
                tmp.unlink(missing_ok=True)
                print(f"agent-memory: model SHA256 mismatch (got {digest})", file=sys.stderr)
                sys.exit(1)
        tmp.rename(_MODEL_PATH)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"agent-memory: model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    return _MODEL_PATH


def _get_session():
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort
        model = _ensure_model()
        _ort_session = ort.InferenceSession(str(model))
    return _ort_session


def _embed(texts: list) -> "np.ndarray":
    """Embed texts using bge-small-en-v1.5. Returns float32 array (N, 384), L2-normalised."""
    global _embed_tokenizer
    if _embed_tokenizer is None:
        from tokenizers import Tokenizer
        assets = _SKILL_DIR / "assets"
        _embed_tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
        _embed_tokenizer.enable_padding()
        _embed_tokenizer.enable_truncation(max_length=512)

    encodings = _embed_tokenizer.encode_batch(texts)
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
    # CLS token pooling (bge-small model card recommendation)
    embeddings = outputs[0][:, 0, :]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)


def _serialize_vec(v) -> bytes:
    """Serialize float32 vector to bytes for sqlite-vec."""
    if hasattr(v, 'tobytes'):
        return v.tobytes()
    return struct.pack(f"{len(v)}f", *v.tolist())


def _open_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _auto_init(db_path: Path) -> None:
    if not db_path.exists():
        _init_db(db_path)


def _init_db(db_path: Path) -> None:
    """Create tables (idempotent). Migrates existing DBs to add the namespace column."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = _open_db(db_path)
    try:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT    NOT NULL UNIQUE,
                content    TEXT    NOT NULL,
                source     TEXT,
                metadata   TEXT    NOT NULL DEFAULT '{}',
                created_at TEXT    NOT NULL,
                namespace  TEXT    NOT NULL DEFAULT 'default'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                embedding float[384]
            );
        """)
        con.commit()
        # Migrate existing DBs that predate the namespace column.
        try:
            con.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
            con.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists — new DB or already migrated.
    finally:
        con.close()


def _insert_chunk(
    con: sqlite3.Connection,
    content: str,
    source: str,
    metadata: dict,
    namespace: str = "default",
) -> None:
    """Insert a single chunk into memories and memories_vec within an open connection."""
    embedding = _embed([content])[0]
    mem_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = con.execute(
        "INSERT INTO memories (id, content, source, metadata, created_at, namespace)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (mem_id, content, source, json.dumps(metadata), created_at, namespace),
    )
    rowid = cur.lastrowid
    con.execute(
        "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, _serialize_vec(embedding)),
    )


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list:
    """Split text into sentences at .!? followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _chunk_paragraphs(paragraphs: list, chunk_size: int, overlap: int) -> list:
    """Merge paragraphs into chunks up to chunk_size; split oversized ones at sentence
    boundaries. Returns list of chunk strings."""
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            # Flush current buffer first
            if current.strip():
                chunks.append(current.strip())
                current = ""
            # Split oversized paragraph at sentence boundaries
            sentences = _split_sentences(para)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + len(sent) + 1 > chunk_size:
                    chunks.append(buf.strip())
                    buf = (buf[-overlap:].strip() + " " + sent).strip() if overlap > 0 else sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if buf.strip():
                chunks.append(buf.strip())
        else:
            if current and len(current) + len(para) + 2 > chunk_size:
                chunks.append(current.strip())
                current = (current[-overlap:].strip() + "\n\n" + para).strip() if overlap > 0 else para
            else:
                current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _chunk_txt(text: str, chunk_size: int, overlap: int) -> list:
    """Chunk plain-text on blank-line paragraph boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)
    return _chunk_paragraphs(paragraphs, chunk_size, overlap)


def _chunk_md(text: str, chunk_size: int, overlap: int) -> list:
    """Chunk markdown. Level-2 headings (## ...) are hard boundaries; heading text
    is carried into the following chunk as its first line."""
    # Split on ## headings
    sections = re.split(r'(?m)^(## .+)$', text)

    # sections: [pre_heading_text, heading, body, heading, body, ...]
    segments = []
    if sections[0].strip():
        segments.append(("", sections[0]))
    for i in range(1, len(sections), 2):
        heading = sections[i]
        body = sections[i + 1] if i + 1 < len(sections) else ""
        segments.append((heading, body))

    chunks = []
    for heading_prefix, body in segments:
        full_text = (heading_prefix + "\n\n" + body).strip() if heading_prefix else body.strip()
        if not full_text:
            continue
        paragraphs = re.split(r'\n\s*\n', full_text)
        section_chunks = _chunk_paragraphs(paragraphs, chunk_size, overlap)
        chunks.extend(section_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    db_path = _resolve_db(args.db)
    _init_db(db_path)
    print(f"Initialised memory DB at {db_path}")


def cmd_add(args):
    db_path = _resolve_db(args.db)
    namespace = _resolve_namespace(getattr(args, "namespace", None))
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
                "INSERT INTO memories (id, content, source, metadata, created_at, namespace)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, args.text, args.source, json.dumps(meta), created_at, namespace),
            )
            rowid = cur.lastrowid
            con.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_vec(embedding)),
            )
    finally:
        con.close()

    print(f"Added {mem_id}")


def _ingest_text_chunks(file_path, ext, chunk_size, overlap, source, db_path, namespace="default"):
    """Shared logic for .txt and .md ingestion."""
    text = file_path.read_text(encoding="utf-8")

    if ext == ".txt":
        raw_chunks = _chunk_txt(text, chunk_size, overlap)
    else:  # .md
        raw_chunks = _chunk_md(text, chunk_size, overlap)

    MIN_CHUNK_LEN = 10
    chunks = []
    skipped = 0
    for chunk in raw_chunks:
        if len(chunk.strip()) < MIN_CHUNK_LEN:
            skipped += 1
        else:
            chunks.append(chunk)

    _auto_init(db_path)
    con = _open_db(db_path)
    added = 0
    try:
        with con:
            for chunk in chunks:
                _insert_chunk(con, chunk, source, {}, namespace)
                added += 1
    finally:
        con.close()

    print(f"Ingesting {file_path.name}\u2026 {added} chunks added ({skipped} skipped).", file=sys.stderr)


def _ingest_json(file_path, source, db_path, namespace="default"):
    """Ingest a JSON file. Expects a top-level array of strings or objects."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {file_path.name}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(
            f"Invalid JSON format in {file_path.name}: expected a top-level array, got {type(data).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    records = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            records.append({"content": item, "source": source, "metadata": {}})
        elif isinstance(item, dict):
            if "content" not in item:
                print(
                    f"Invalid JSON format in {file_path.name}: element at index {idx} is missing required 'content' key",
                    file=sys.stderr,
                )
                sys.exit(1)
            content = item["content"]
            if not isinstance(content, str):
                print(
                    f"Invalid JSON format in {file_path.name}: element at index {idx} 'content' must be a string",
                    file=sys.stderr,
                )
                sys.exit(1)
            meta = item.get("metadata", {})
            if not isinstance(meta, dict):
                print(
                    f"Invalid JSON format in {file_path.name}: element at index {idx} 'metadata' must be an object",
                    file=sys.stderr,
                )
                sys.exit(1)
            row_source = item.get("source", source)
            records.append({"content": content, "source": row_source, "metadata": meta})
        else:
            print(
                f"Invalid JSON format in {file_path.name}: element at index {idx} must be a string or object, got {type(item).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)

    MIN_CHUNK_LEN = 10
    _auto_init(db_path)
    con = _open_db(db_path)
    added = 0
    skipped = 0
    try:
        with con:
            for rec in records:
                if len(rec["content"].strip()) < MIN_CHUNK_LEN:
                    skipped += 1
                    continue
                _insert_chunk(con, rec["content"], rec["source"], rec["metadata"], namespace)
                added += 1
    finally:
        con.close()

    print(f"Ingesting {file_path.name}\u2026 {added} rows added ({skipped} skipped).", file=sys.stderr)


def _ingest_csv(file_path, column, source, db_path, namespace="default"):
    """Ingest a CSV file. --column specifies the text column; others become metadata."""
    import csv

    with file_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            print(f"CSV file {file_path.name} appears to be empty", file=sys.stderr)
            sys.exit(1)
        fieldnames = list(reader.fieldnames)
        if column not in fieldnames:
            print(
                f"Column {column!r} not found in {file_path.name}. Available columns: {fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)
        meta_columns = [f for f in fieldnames if f != column]
        rows = list(reader)

    MIN_CHUNK_LEN = 10
    _auto_init(db_path)
    con = _open_db(db_path)
    added = 0
    skipped = 0
    try:
        with con:
            for row in rows:
                content = row[column]
                if len(content.strip()) < MIN_CHUNK_LEN:
                    skipped += 1
                    continue
                metadata = {col: row[col] for col in meta_columns}
                _insert_chunk(con, content, source, metadata, namespace)
                added += 1
    finally:
        con.close()

    print(f"Ingesting {file_path.name}\u2026 {added} rows added ({skipped} skipped).", file=sys.stderr)


def cmd_ingest(args):
    """Ingest a file by chunking it and inserting each chunk into the memory DB."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    ext = file_path.suffix.lower()
    source = args.source if args.source else file_path.name
    db_path = _resolve_db(args.db)
    namespace = _resolve_namespace(getattr(args, "namespace", None))

    if ext in (".txt", ".md"):
        _ingest_text_chunks(file_path, ext, args.chunk_size, args.overlap, source, db_path, namespace)
    elif ext == ".json":
        _ingest_json(file_path, source, db_path, namespace)
    elif ext == ".csv":
        if not args.column:
            print(
                f"--column NAME is required when ingesting a CSV file (specifies which column holds the text)",
                file=sys.stderr,
            )
            sys.exit(1)
        _ingest_csv(file_path, args.column, source, db_path, namespace)
    else:
        print(
            f"Unsupported format: {ext!r}. Supported formats: .txt, .md, .json, .csv",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_search(args):
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"No memory DB found at {db_path}. Run: memory.py init")
        sys.exit(0)

    namespace = _resolve_namespace(getattr(args, "namespace", None))
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
              AND m.namespace = ?
            ORDER BY v.distance
        """
        rows = con.execute(sql, (_serialize_vec(query_vec), limit, namespace)).fetchall()
    finally:
        con.close()

    if args.source:
        rows = [r for r in rows if r[2] == args.source]

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
    elif getattr(args, 'long', False):
        for r in rows:
            dist, mem_id, content, source = r[5], r[0], r[1], r[2]
            print(f"--- [{dist:.3f}] {mem_id}  (source: {source}) ---")
            print(content)
            print()
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

    namespace = _resolve_namespace(getattr(args, "namespace", None))
    limit = args.limit
    con = sqlite3.connect(db_path)
    try:
        conditions = ["namespace = ?"]
        params = [namespace]
        if args.source:
            conditions.append("source = ?")
            params.append(args.source)
        sql = "SELECT id, content, source, metadata, created_at FROM memories"
        sql += " WHERE " + " AND ".join(conditions)
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
    elif getattr(args, 'long', False):
        for r in rows:
            mem_id, content, source, _, created_at = r
            print(f"--- {mem_id}  (source: {source})  {created_at} ---")
            print(content)
            print()
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
        ns_rows = con.execute(
            "SELECT namespace, COUNT(*) FROM memories GROUP BY namespace ORDER BY namespace"
        ).fetchall()
    finally:
        con.close()

    print(f"DB path:   {db_path}")
    print(f"Memories:  {count}")
    print(f"DB size:   {_format_size(size_bytes)}")
    if ns_rows:
        print("Namespaces:")
        for ns, ns_count in ns_rows:
            print(f"  {ns}: {ns_count}")


def main():
    parser = argparse.ArgumentParser(
        prog="memory.py",
        description="agent-memory — long-term semantic memory for agents",
    )
    parser.add_argument("--db", metavar="PATH", help="Override DB path")
    parser.add_argument(
        "--namespace",
        metavar="NAME",
        default="default",
        help="Memory namespace to scope reads and writes (default: 'default')",
    )

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

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file into memory")
    ingest_parser.add_argument("file", help="Path to file to ingest (.txt, .md, .json, or .csv)")
    ingest_parser.add_argument(
        "--source", metavar="TAG",
        help="Source tag (defaults to file basename)"
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=1000, metavar="N",
        help="Target chunk size in characters (default: 1000)"
    )
    ingest_parser.add_argument(
        "--overlap", type=int, default=100, metavar="N",
        help="Character overlap between adjacent chunks (default: 100)"
    )
    ingest_parser.add_argument(
        "--column", metavar="NAME",
        help="CSV only: column name containing the text to ingest"
    )
    ingest_parser.add_argument("--db", metavar="PATH", help="Override DB path")
    ingest_parser.set_defaults(func=cmd_ingest)

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
    search_parser.add_argument(
        "--long", action="store_true",
        help="Print full chunk content for each result, separated by delimiter lines"
    )
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
    list_parser.add_argument(
        "--long", action="store_true",
        help="Print full chunk content for each result, separated by delimiter lines"
    )
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
