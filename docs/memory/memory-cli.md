---
name: memory.py CLI
description: Reference for the agent-memory CLI entry point — invocation, bootstrap, and implemented subcommands
type: project
---

# memory.py CLI

`memory.py` is the single-file entry point for the `agent-memory` skill. It manages its own Python virtualenv and requires only a standard Python 3.11+ installation.

## Invocation

```
./memory.py <subcommand> [options]
# or
python3 /path/to/memory.py <subcommand> [options]
```

The file must be executable (`chmod +x memory.py`). The shebang (`#!/usr/bin/env python3`) handles interpreter lookup.

## Bootstrap

On the first run, `memory.py` transparently creates a `.venv/` directory next to itself and installs four dependencies:

- `onnxruntime>=1.17.0`
- `sqlite-vec>=0.1.6`
- `tokenizers>=0.19.0`
- `numpy>=1.24.0`

After installation the process re-execs itself under the managed interpreter. This is a one-time cost (~30 s on a typical connection). A message is printed to `stderr` during the first-run install:

```
agent-memory: first run, installing dependencies…
```

`.apm/skills/skill-memory/.venv/` is listed in `.gitignore` and must never be committed.
The virtualenv is created next to `memory.py` inside `.apm/skills/skill-memory/`.

## Model bootstrap

The first subcommand that requires embeddings (e.g. `add`) triggers an automatic model download if the ONNX file is absent.

- **Model:** `BAAI/bge-small-en-v1.5` (ONNX variant, ~24 MB)
- **Cache path:** `~/.cache/agent-memory/bge-small-en-v1.5/model.onnx`
- **Integrity check:** SHA256 is verified after download. On mismatch the partial file is deleted and the process exits with code 1 and a descriptive error.
- **Network failure:** The partial file is deleted and the process exits with code 1.

A progress message is printed to `stderr` during the download:

```
Downloading bge-small-en-v1.5 model…
```

This is a one-time cost. Subsequent invocations skip the download if the file already exists at the cache path.

## Embedding pipeline

Embeddings are produced by the `_embed(texts: list) -> np.ndarray` function using the `BAAI/bge-small-en-v1.5` ONNX model.

### Singleton pattern

Two module-level singletons are loaded once per process and reused across all calls:

- **`_get_ort_session()`** — creates the `onnxruntime.InferenceSession` on first call using the model path returned by `_ensure_model()`.
- **`_get_embed_tokenizer()`** — loads the `tokenizers.Tokenizer` from `assets/tokenizer.json` (bundled next to `memory.py`) on first call, with `max_length=512` truncation and zero-padding enabled.

### Tokenization

Input texts are tokenized in batch via `tokenizer.encode_batch(texts)`. The tokenizer produces three int64 arrays:

| Array | ONNX input name |
|---|---|
| `input_ids` | `input_ids` |
| `attention_mask` | `attention_mask` |
| `type_ids` | `token_type_ids` |

### Inference and pooling

The ONNX session is run with the three arrays above. The single output node `last_hidden_state` has shape `(N, seq_len, 384)`. The **CLS token** (index 0 of the sequence dimension) is selected:

```
cls_embeddings = last_hidden_state[:, 0, :]  # shape (N, 384)
```

This follows the official recommendation from the bge-small-en-v1.5 model card.

### L2 normalisation

Each embedding vector is L2-normalised before returning:

```python
norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
return (cls_embeddings / norms).astype(np.float32)
```

The returned array has shape `(N, 384)` and dtype `float32`. Each row has L2 norm ≈ 1.0, making cosine similarity equivalent to dot product — which is what `sqlite-vec`'s `vec0` index uses for nearest-neighbour search.

## DB path resolution

Every subcommand resolves the database path in this order (highest priority first):

| Priority | Source |
|---|---|
| 1 | `--db <path>` CLI flag |
| 2 | `AGENT_MEMORY_DB` environment variable |
| 3 | `~/.local/share/agent-memory/memories.db` (default) |

## Subcommands

### `init`

```
memory.py init [--db PATH]
```

Creates the memory database and both tables. Idempotent — safe to run more than once.

Output on success:

```
Initialised memory DB at <path>
```

### `stats`

```
memory.py stats [--db PATH]
```

Prints the DB path, total memory count, and file size. If no DB exists at the resolved path, prints a not-found message and exits cleanly.

Example output:

```
DB path:   /home/user/.local/share/agent-memory/memories.db
Memories:  42
DB size:   1.3 MB
```

### `add`

```
memory.py add <text> [--source TAG] [--meta KEY=VALUE ...] [--db PATH]
```

Stores a new memory entry. Auto-initialises the database if it does not yet exist.

| Argument | Description |
|---|---|
| `<text>` | Text content to store (required) |
| `--source TAG` | Optional origin tag or file path |
| `--meta KEY=VALUE` | Arbitrary metadata key/value pair (repeatable) |
| `--db PATH` | Override DB path |

The text is embedded via the `bge-small-en-v1.5` ONNX model (model downloaded on first use — see [Model bootstrap](#model-bootstrap)). Both the `memories` and `memories_vec` rows are written in a single transaction.

Output on success:

```
Added <uuid>
```

### Stub subcommands (not yet implemented)

The following subcommands are defined in the parser and will be implemented in subsequent beads. Invoking them currently raises `NotImplementedError`.

| Subcommand | Purpose |
|---|---|
| `search <query> [--limit N] [--threshold F] [--source TAG] [--json] [--db PATH]` | Semantic similarity search over stored memories |
| `list [--limit N] [--source TAG] [--json] [--db PATH]` | List stored memories |
| `delete <id> [--db PATH]` | Delete a memory by UUID |

## DB schema

Two tables are created by `init`:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT    NOT NULL UNIQUE,   -- UUID v4 string
    content    TEXT    NOT NULL,
    source     TEXT,                      -- optional origin tag / file path
    metadata   TEXT    NOT NULL DEFAULT '{}',  -- arbitrary JSON blob
    created_at TEXT    NOT NULL           -- ISO-8601 UTC
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    embedding float[384]
);
```

`memories_vec` requires the `sqlite-vec` extension, which is loaded automatically by the bootstrap.
