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

`.venv/` is listed in `.gitignore` and must never be committed.

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

### Stub subcommands (not yet implemented)

The following subcommands are defined in the parser and will be implemented in subsequent beads. Invoking them currently raises `NotImplementedError`.

| Subcommand | Purpose |
|---|---|
| `add <text> [--source TAG] [--meta KEY=VALUE ...] [--db PATH]` | Store a new memory with optional metadata |
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
