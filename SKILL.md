---
name: agent-memory
description: Long-term semantic memory for agents — store, search, and retrieve text memories across sessions using SQLite and BAAI/bge-small-en-v1.5 embeddings.
tools: Bash
user-invocable: true
---

# agent-memory

Long-term semantic memory for agents. Memories are stored in a local SQLite database with vector search backed by `sqlite-vec` and BAAI/bge-small-en-v1.5 ONNX embeddings.

## Getting Started

`memory.py` lives at the repo root alongside this `SKILL.md`. Run it with:

```bash
python3 memory.py <command> [options]
```

On first run, `memory.py` bootstraps a local `.venv` with its dependencies (`onnxruntime`, `sqlite-vec`, `tokenizers`, `numpy`). The embedding model is downloaded to `~/.cache/agent-memory/bge-small-en-v1.5/` on first use. The default DB lives at `memories.db` in the same directory as `memory.py` (i.e. the repo root) and is auto-initialised on first write — no explicit `init` needed.

## When to Use This Skill

| Moment | Action |
|--------|--------|
| Bead start | `search` for relevant prior context before reading the codebase |
| Bead end | `add` notable findings, design decisions, or gotchas |
| After completing a feature | `ingest` the spec or implementation notes for future retrieval |

## Namespaces

All commands respect a `--namespace NAME` flag (default: `'default'`). Namespaces partition memories within a single DB so that different agents, projects, or concerns can share one file without interference.

```bash
python3 memory.py --namespace myproject add "TEXT"
python3 memory.py --namespace myproject search "QUERY"
python3 memory.py --namespace myproject list
```

`--namespace` is a **top-level flag** — it must come before the subcommand name. Memories written under one namespace are invisible when reading under a different namespace.

| Scenario | Recommendation |
|----------|----------------|
| Single agent / single project | Use the default namespace |
| Multiple agents sharing a DB | Assign each agent its own namespace |
| Temporary scratch space | Use a short-lived namespace and delete when done |

## Commands Reference

### `init` — Initialise the memory DB

Creates the `memories` and `memories_vec` tables. Safe to run multiple times (idempotent).

```bash
python3 memory.py [--namespace NAME] init [--db PATH]
```

### `add` — Add a single memory

```bash
python3 memory.py [--namespace NAME] add "TEXT" [--source TAG] [--meta KEY=VALUE ...] [--db PATH]
```

Use for short, standalone facts or observations discovered during a bead. `--source` is an optional tag (e.g. a bead ID or filename) for later filtering.

### `ingest` — Bulk-ingest a file

```bash
python3 memory.py [--namespace NAME] ingest FILE [--source TAG] [--chunk-size N] [--overlap N] [--column NAME] [--db PATH]
```

Reads `FILE`, splits it into chunks, and inserts each chunk. Use after completing a feature to seed the memory store from documentation or notes.

**Supported formats:**

| Extension | Chunking strategy |
|-----------|-------------------|
| `.txt` | Split on blank-line paragraph boundaries; merge short paragraphs up to `--chunk-size`; split oversized paragraphs at sentence boundaries |
| `.md` | Same as `.txt` but `## ` headings are hard boundaries; heading text is carried into the following chunk |
| `.json` | Top-level array of strings or `{"content": "...", "metadata": {}, "source": "..."}` objects |
| `.csv` | `--column NAME` selects the text column; all other columns become chunk metadata |

Defaults: `--chunk-size 1000` (characters), `--overlap 100`. Chunks shorter than 10 characters are silently skipped; the summary line reports the count.

Progress output goes to stderr:
```
Ingesting notes.md… 12 chunks added (0 skipped).
```

### `search` — Semantic search

```bash
python3 memory.py [--namespace NAME] search "QUERY" [--limit N] [--threshold F] [--source TAG] [--json] [--db PATH]
```

Returns the top-`N` memories closest to the query vector (default: 5). `--threshold` is the maximum L2 distance to include (0–2; lower = more similar). `--json` outputs a JSON array.

### `list` — List memories

```bash
python3 memory.py [--namespace NAME] list [--limit N] [--source TAG] [--json] [--db PATH]
```

Lists memories newest-first (default limit: 20; `--limit 0` for all). Filter by `--source` to scope to a specific file or bead.

### `delete` — Delete a memory

```bash
python3 memory.py [--namespace NAME] delete UUID [--db PATH]
```

Removes the memory and its embedding vector by UUID.

### `stats` — Show DB statistics

```bash
python3 memory.py [--namespace NAME] stats [--db PATH]
```

Prints the DB path, total memory count, and file size.

## DB Path Conventions

| Scenario | Recommendation |
|----------|----------------|
| Personal / cross-project context | Use the default (`memories.db` next to `memory.py`) |
| Project-local context | Pass `--db .agent-memory.db` or set `AGENT_MEMORY_DB=.agent-memory.db` |
| CI / ephemeral environments | Set `AGENT_MEMORY_DB` to a temp path |

Prefer the shared default DB for general knowledge that spans projects. Use a project-local DB when memories are scoped to a single repository and should not bleed into other workspaces.

## What NOT to Store

- Bead-specific or ephemeral task state
- Information that belongs in `CLAUDE.md` or guardrail templates
- Git history, recent changes, or file contents that are trivially readable from the repo
- Details already captured in a spec or design document

Mirror the guidance in `docs/memory/conventions.md`: store only what is project-wide, reusable, and likely to change your approach in a future bead if you had known it upfront.
