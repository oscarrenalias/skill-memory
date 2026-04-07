---
name: "agent-memory: query and CRUD operations"
id: spec-9f1b7155
description: "search, delete, list, and stats commands for the agent-memory CLI"
dependencies:
- spec-a536cb5e
priority: high
complexity: medium
status: planned
tags:
- agent-memory
- search
- crud
- sqlite-vec
scope:
  in: "|"
  out: "|"
feature_root_id: null
---

# agent-memory: query and CRUD operations

## Objective

Add `search`, `delete`, `list`, and `stats` commands to `memory.py`, completing the interactive memory operations. These commands depend on the DB schema, embedding pipeline, and `add` command delivered by Spec 1 (`spec-a536cb5e`).

## Problems to Fix

1. After Spec 1, memories can be written but not retrieved, browsed, or removed. The skill is unusable until read operations exist.

## Changes

### 1. `search` command

```
memory.py search <query> [--limit N] [--threshold F] [--source TAG] [--json] [--db PATH]
```

- Embed `query` via `_embed([query])` (reuse pipeline from Spec 1)
- Query `memories_vec` for the `--limit` nearest neighbours (default: 5)
- Join with `memories` on rowid to retrieve content, source, metadata, created_at
- Apply `--threshold` as a maximum L2 distance filter (default: no threshold; range 0–2 for L2-normalised vectors; lower = more similar)
- Apply `--source` as a Python post-filter on ANN results (pre-filtering is not supported by sqlite-vec 0.1.x)

**Default output** (one result per line):
```
[0.142] <uuid>  The quick brown fox…  (source: notes)
```

**JSON output** (`--json`): array of objects:
```json
[
  {
    "id": "<uuid>",
    "content": "The quick brown fox…",
    "source": "notes",
    "metadata": {},
    "created_at": "2026-04-07T10:00:00Z",
    "distance": 0.142
  }
]
```

### 2. `delete` command

```
memory.py delete <id> [--db PATH]
```

- Resolve `id` to a rowid via `SELECT rowid FROM memories WHERE id = ?`
- Delete from `memories_vec` by rowid, then from `memories` by id (in a single transaction)
- Print: `Deleted <id>` on success
- Exit 1 with error message on stderr if ID not found or DB does not exist

**Status: implemented.** Both `memories_vec` and `memories` rows are removed atomically. The rowid lookup ensures the vector index stays consistent with the metadata table.

### 3. `list` command

```
memory.py list [--limit N] [--source TAG] [--json] [--db PATH]
```

- Select from `memories` ordered by `created_at DESC`
- `--limit` defaults to 20; pass 0 for unlimited
- `--source` filters by exact match on the `source` column
- Default output: one row per line — `<uuid>  <truncated content 60 chars>  (source: …)  <created_at>`
- `--json`: array of objects with all columns (no `distance` field)

### 4. `stats` command

```
memory.py stats [--db PATH]
```

Prints:
```
DB path:   ~/.local/share/agent-memory/memories.db
Memories:  42
DB size:   1.2 MB
```

- Row count from `SELECT COUNT(*) FROM memories`
- DB size from `os.path.getsize`
- If DB does not exist, print a short message and exit 0 (not an error)

## Files to Modify

| File | Change |
|---|---|
| `memory.py` | Add search, delete, list, stats subcommand handlers |

No new files are introduced by this spec.

## Acceptance Criteria

- `memory.py add "cats are great"` followed by `memory.py search "cats"` returns at least one result containing the added text with a distance < 0.5.
- `memory.py search "cats" --limit 3` returns at most 3 results.
- `memory.py search "cats" --json` outputs valid JSON with the schema described above.
- `memory.py delete <id>` removes the row from both `memories` and `memories_vec`; a subsequent `memory.py list` does not include the deleted id.
- `memory.py delete <nonexistent-id>` exits with code 1 and prints an error to stderr.
- `memory.py list` returns results newest-first.
- `memory.py list --source mytag` returns only rows where source = "mytag".
- `memory.py stats` prints DB path, row count, and file size without errors when at least one memory exists.
- `memory.py stats` on a non-existent DB prints a descriptive message and exits 0.

## Resolved Decisions

1. **sqlite-vec distance metric** — **Resolved (L2 / Euclidean).** `vec0` tables with `float[N]` columns use L2 distance in sqlite-vec ≥0.1.6. Vectors are L2-normalised by `_embed`, so distance values fall in [0, 2] (0 = identical, 2 = maximally opposite). Use `--threshold ≤0.5` for high similarity, `≤1.0` for moderate. This is documented in the `_embed` and `_init_db` docstrings in `memory.py`.

2. **Source filter placement** — **Resolved (post-filter).** sqlite-vec 0.1.x `vec0` tables do not support additional column predicates in the `WHERE` clause alongside the `MATCH` constraint. Source filtering is applied as a Python post-filter after the ANN results are retrieved (see `cmd_search`). This is documented in the `_init_db` docstring in `memory.py`.
