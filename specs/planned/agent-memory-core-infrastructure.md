---
name: "agent-memory: core infrastructure"
id: spec-a536cb5e
description: "SQLite DB schema, ONNX embedding pipeline, model bootstrap, and init/add commands"
dependencies: []
priority: high
complexity: high
status: planned
tags:
- agent-memory
- infrastructure
- embeddings
- sqlite
scope:
  in: "|"
  out: "|"
feature_root_id: null
---

# agent-memory: core infrastructure

## Objective

Build the foundational layer of the `agent-memory` Claude Code skill: a Python CLI (`memory.py`) backed by SQLite + sqlite-vec for vector storage and BAAI/bge-small-en-v1.5 via ONNX runtime for embeddings. This spec covers the DB schema, embedding pipeline, model bootstrap, and the `init` and `add` commands. All subsequent specs depend on this one.

## Problems to Fix

1. There is no standalone, distributable long-term memory primitive for agents in this project. Agent beads currently cannot persist knowledge across sessions or share learned context with other agents.
2. Existing "memory" in this project is limited to append-only markdown files (`docs/memory/`), which are not semantically searchable.

## Changes

### 1. Directory layout

Create `.claude/skills/agent-memory/` with:

```
.claude/skills/agent-memory/
├── memory.py            # CLI entry point (executable, shebang #!/usr/bin/env python3)
├── .gitignore           # ignores .venv/
└── assets/              # vendored tokenizer files (small, committed to git)
    ├── tokenizer.json        # ~711 KB — full fast-tokenizer vocab + config
    ├── tokenizer_config.json # ~366 B
    ├── vocab.txt             # ~232 KB — legacy vocab, needed for fallback
    └── special_tokens_map.json # ~125 B
```

All four tokenizer files are sourced from `BAAI/bge-small-en-v1.5` on HuggingFace. The ONNX model is NOT committed. It is downloaded at runtime (see §4). There is no `pyproject.toml` — dependencies are managed by the bootstrap (see §2).

### 2. Dependency bootstrap

`memory.py` manages its own isolated virtualenv at `.venv/` relative to the script file. This requires nothing beyond a standard Python 3.11+ installation — no `uv`, no `pip install` by the caller.

The bootstrap runs at the very top of `memory.py`, before any third-party imports:

```python
#!/usr/bin/env python3
import sys, os, subprocess
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
```

`memory.py` must be marked executable (`chmod +x`). Invocation is `./memory.py <subcommand>` or equivalently `python3 /path/to/memory.py <subcommand>` — the agent running the skill can use whichever form is convenient. The shebang (`#!/usr/bin/env python3`) handles the initial interpreter lookup.

On first run, two things happen sequentially before any command executes:
1. `.venv/` is created and deps are installed (~30s, pip download)
2. The ONNX model is downloaded if absent (~24 MB, see §4)

Both are one-time costs. A message is printed to stderr for each so the caller knows why startup is slow.

`.venv/` must be listed in `.claude/skills/agent-memory/.gitignore` so it is never committed.

### 3. DB schema

Two tables created by `memory.py init`:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT    NOT NULL UNIQUE,   -- UUID v4 string
    content    TEXT    NOT NULL,
    source     TEXT,                      -- optional origin tag / file path
    metadata   TEXT    NOT NULL DEFAULT '{}',  -- arbitrary JSON blob
    created_at TEXT    NOT NULL           -- ISO-8601 UTC
    -- implicit INTEGER rowid used to join with memories_vec
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    embedding float[384]
);
```

DB path resolution order (highest priority first):
1. `--db <path>` CLI flag
2. `AGENT_MEMORY_DB` environment variable
3. `~/.local/share/agent-memory/memories.db` (default)

`init` must be idempotent (safe to run more than once). It should also run automatically before any write command if the DB file does not yet exist.

### 4. Model bootstrap (`_ensure_model()`)

On first use, download the ONNX model to `~/.cache/agent-memory/bge-small-en-v1.5/model.onnx`:

- Source URL: `https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx`
- Verify SHA256 against a hardcoded constant after download
- Print a one-line progress message to stderr during download (`Downloading bge-small-en-v1.5 model…`)
- If download or verification fails, exit with a clear error message and delete the partial file

> **⚠ PENDING DECISION — see §Pending Decisions #1:** The SHA256 of the exact model file must be confirmed before hardcoding.

### 5. Embedding pipeline (`_embed(texts: list[str]) -> np.ndarray`)

```python
def _embed(texts: list[str]) -> np.ndarray:
    # Returns float32 array of shape (N, 384), L2-normalised
```

Steps:
1. Load tokenizer from `assets/tokenizer.json` (relative to `memory.py`)
2. Tokenize with `max_length=512`, `padding=True`, `truncation=True`
3. Run `onnxruntime.InferenceSession` on the downloaded model; input node names are `input_ids`, `attention_mask`, `token_type_ids`; output node name is `last_hidden_state` (shape `[batch, seq_len, 384]`)
4. Pool strategy: **CLS token** — take `output[:, 0, :]` (index 0 of the sequence dimension). This is the official recommendation from the bge-small-en-v1.5 model card.
5. L2-normalise each vector: `v / np.linalg.norm(v, axis=1, keepdims=True)`

The ONNX session is module-level singleton (loaded once per process invocation). Do not reload on every call.

### 6. `init` command

```
memory.py init [--db PATH]
```

- Creates parent directories if needed
- Creates the two tables (idempotent)
- Prints: `Initialised memory DB at <path>`

### 7. `add` command

```
memory.py add <text> [--source TAG] [--meta KEY=VALUE ...] [--db PATH]
```

- Auto-inits DB if not yet initialised
- Embeds `text` via `_embed([text])`
- Generates a UUID v4 `id`
- Inserts into `memories` (get `lastrowid`) then inserts into `memories_vec` with matching rowid
- Prints: `Added <id>`

Both inserts must be wrapped in a single transaction; roll back on any failure.

## Files to Add / Modify

| File | Change |
|---|---|
| `.claude/skills/agent-memory/memory.py` | New — executable CLI with shebang, bootstrap, init and add commands |
| `.claude/skills/agent-memory/.gitignore` | New — ignores `.venv/` |
| `.claude/skills/agent-memory/assets/tokenizer.json` | New — vendored BGE fast tokenizer (~711 KB) |
| `.claude/skills/agent-memory/assets/tokenizer_config.json` | New — vendored BGE tokenizer config (~366 B) |
| `.claude/skills/agent-memory/assets/vocab.txt` | New — vendored BGE legacy vocab (~232 KB) |
| `.claude/skills/agent-memory/assets/special_tokens_map.json` | New — vendored BGE special tokens (~125 B) |

## Acceptance Criteria

- `./memory.py init` creates `memories` and `memories_vec` tables; running it twice does not raise an error.
- `./memory.py add "hello world"` prints an ID in the form `Added <uuid>`.
- A second `add` of different text produces a different UUID and a row in both DB tables.
- Running `add` without a prior `init` auto-initialises the DB.
- `_embed(["hello"])` returns an ndarray of shape `(1, 384)` with L2 norm ≈ 1.0 (CLS token, L2-normalised).
- If `model.onnx` is absent, the first `add` triggers a download with a progress message on stderr.
- If the downloaded model fails SHA256 verification, the command exits with code 1 and a descriptive error.
- All DB writes use a single transaction; a simulated failure after the `memories` insert leaves the DB unchanged.
- `--db /tmp/test.db` and `AGENT_MEMORY_DB=/tmp/test.db` both override the default path.

## Pending Decisions

1. **Model SHA256** *(partially resolved — verification required at implementation time)*: The implementing agent must compute the SHA256 of the downloaded `model.onnx` file before hardcoding it. Do not rely on any pre-reported hash without verification. The expected file size is ~24 MB (bge-*small*); if the downloaded file is significantly larger, the wrong model variant has been downloaded. Compute the hash with:
   ```bash
   curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx -o model.onnx && sha256sum model.onnx
   ```
   Then hardcode the resulting hash as `_MODEL_SHA256` in `memory.py`.

2. ~~**Pooling strategy**~~: **Resolved — CLS token pooling.** Use `last_hidden_state[:, 0, :]` followed by L2 normalisation. Confirmed by the official bge-small-en-v1.5 model card.

3. ~~**Tokenizer asset source**~~: **Resolved.** Vendor four files from `BAAI/bge-small-en-v1.5` on HuggingFace: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`. The implementing agent downloads and commits these during setup.
