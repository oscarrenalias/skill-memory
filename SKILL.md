---
name: agent-memory
description: Persist and retrieve knowledge across agent sessions. Use at bead start to search for prior context; use at bead end to store findings, decisions, and gotchas worth remembering in future tasks.
tools: Bash
user-invocable: true
---

# agent-memory

This skill exists to break the amnesia of single-session agents. Without it, every bead starts from scratch — rediscovering the same patterns, re-learning the same gotchas, and repeating the same mistakes. With it, agents can accumulate project-wide knowledge that improves their judgment in future beads.

Use this skill to surface relevant context before you start work, and to write down anything you learned that you'd want to know next time.

## What NOT to Store

These are not worth persisting — they are easily recovered, quickly stale, or belong elsewhere:

- File contents or code snippets readable from the repo
- Git history, recent changes, or who-changed-what (use `git log` instead)
- Bead-specific task state or in-progress work
- Information already captured in a spec or design document
- Details that belong in `CLAUDE.md` or agent guardrail templates

**Good memory:** A recurring merge conflict pattern and how to resolve it. A non-obvious project convention not documented anywhere. A design decision and its rationale.

**Bad memory:** The current contents of `memory.py`. The list of PRs merged this week. The steps you took to complete a bead.

The guiding test: *would knowing this upfront change my approach to a future bead?* If yes, store it. If no, skip it.

## When to Use This Skill

### At bead start — search before reading code

Run a search before diving into the codebase. Prior agents may have stored context that reframes the problem:

```bash
python3 memory.py --namespace takt search "merge conflict resolution"
python3 memory.py --namespace takt search "namespace conventions"
```

**Interpreting results:** A strong match is a result whose text directly addresses your question — same pattern, same decision space, same file area. Treat distance scores below ~0.8 as potentially relevant; above ~1.2 as noise. Skim the top 3–5 results rather than acting on a single hit. If results look unrelated, your query may be too specific — try broader terms.

### At bead end — write what you'd want to know next time

After completing work, add memories for:
- Non-obvious conventions you discovered (e.g. "always use `--namespace` before the subcommand")
- Design decisions and why they were made
- Patterns that caused errors and how to avoid them
- Anything that surprised you and would surprise the next agent

```bash
python3 memory.py --namespace takt add "The --namespace flag must come before the subcommand name, not after it. Wrong: memory.py add --namespace foo. Right: memory.py --namespace foo add." --source B-abc12def
```

Keep entries focused and self-contained. Each memory should make sense on its own without surrounding context.

## Namespace Conventions for takt Agents

Use the `--namespace` flag to partition memories by feature or agent type. This prevents unrelated contexts from polluting search results.

| Scenario | Recommended namespace |
|----------|-----------------------|
| Project-wide agent knowledge | `takt` or the project name |
| Feature-scoped context | The feature root bead ID (e.g. `B-fc273260`) |
| Agent-type-specific patterns | `developer`, `tester`, `planner`, etc. |

Use one namespace per feature or per agent type — don't mix concerns in the same namespace. Within a feature, all sub-beads should share the feature root bead ID as namespace so they can see each other's findings.

```bash
# Write under feature namespace
python3 memory.py --namespace B-fc273260 add "TEXT" --source B-fc273260-docs

# Search the feature namespace
python3 memory.py --namespace B-fc273260 search "QUERY"

# Write project-wide knowledge
python3 memory.py --namespace takt add "TEXT"
```

## Getting Started

Run `python3 memory.py <command> [options]` from the repo root. The DB auto-initialises on first write.

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
