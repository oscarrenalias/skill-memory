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
- Apply `--threshold` as a maximum distance filter (default: no threshold; lower distance = more similar)
- Apply `--source` to filter by the `source` column before or after the ANN search

> **⚠ PENDING DECISION — see §Pending Decisions #1:** sqlite-vec distance metric (L2 vs cosine) and the expected value range for the threshold flag must be confirmed so the `--threshold` default and help text are meaningful.

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
- Delete from `memories_vec` by rowid, then from `memories` by id
- Both deletes in a single transaction
- Print: `Deleted <id>` on success; exit 1 with error if ID not found

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

## Pending Decisions

1. **sqlite-vec distance metric**: `vec0` returns a `distance` column whose semantics (L2, cosine, inner product) depend on how the virtual table was created and the sqlite-vec version. The spec currently uses the default (likely L2). The implementing agent must confirm which metric is in use for the version of sqlite-vec declared in `pyproject.toml`, document it in a comment in `memory.py`, and set the `--threshold` help text accordingly. If cosine distance is available and preferred (range 0–2 for normalised vectors, lower = more similar), the spec should be updated before implementation.

2. **Source filter placement**: For the `search` command, filtering by `--source` can happen either (a) as a post-filter on ANN results (fast, may miss relevant items if limit is small) or (b) as a pre-filter using a row id allow-list before the ANN query. sqlite-vec's support for pre-filtering constraints should be checked for the declared version. *Implementing agent should use post-filter unless pre-filter is straightforward and well-supported.*
