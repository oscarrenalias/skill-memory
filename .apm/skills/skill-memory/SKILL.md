---
name: agent-memory
description: General-purpose long-term memory for agents. Store and retrieve any information worth remembering across sessions — business context, client knowledge, research findings, decisions, discovered facts, or anything else that would be useful to recall later. Supports text-based information, can ingest files (markdown, CSV, plain text), and is searchable by semantic similarity.
tools: Bash
user-invocable: true
---

# agent-memory

General-purpose long-term memory for agents. Use it to store any information worth keeping between sessions — there is no restriction on topic or domain. Business context, technical findings, meeting notes, research, decisions, warnings, preferences — if it would be useful to recall later, it belongs here. Provides a simple interface for adding text-based memories with optional metadata and searching them by semantic similarity. Search before starting work; write when you finish.

## Example Use Cases

The scope is intentionally broad and not limited to software development. Here are examples across different domains:

- **Client and stakeholder knowledge** — A client's stated priorities going into a project; a stakeholder who prefers summaries over detail; known constraints a decision-maker has mentioned before.
- **Business context** — Key figures from a business case; a market assumption underpinning a strategy; a budget constraint that ruled out a particular direction.
- **Meetings and conversations** — Outcomes and action items from a client meeting; a commitment made verbally that isn't written down elsewhere; context behind a decision that was reached in discussion.
- **Research and analysis** — Findings from a competitive analysis; a source that turned out to be unreliable; conclusions from a spike or exploratory investigation.
- **Preferences and working styles** — How a particular person or team prefers to receive information; a communication style that worked well; a format that was explicitly rejected.
- **Decisions and their rationale** — Why one option was chosen over another; constraints that shaped the outcome; assumptions that were made at the time.
- **Recurring patterns and warnings** — A mistake that keeps being repeated and how to avoid it; something that looks straightforward but has a hidden catch.
- **Domain and subject matter knowledge** — A regulatory requirement relevant to a sector; a term that means something specific in a given context; a process quirk particular to an organisation.

## What NOT to Store

These are not worth persisting — they are easily recovered, quickly stale, or belong elsewhere:

- Verbatim content from documents or files you can read directly
- Transient state or in-progress work from the current session
- Information already captured in a formal, maintained document
- Step-by-step logs of what you did (that's a task record, not a memory)
- Raw data that hasn't been interpreted into a finding or conclusion

**Good memory:** A client's budget constraint that ruled out a vendor. A decision made in a meeting and the reasoning behind it. A term that means something specific in this organisation. A recurring mistake and how to avoid it.

**Bad memory:** A verbatim transcript of a document you can read directly. A list of tasks completed this session. Information that is already captured in a formal document and won't change.

The guiding test: *would knowing this upfront change my approach to a future task?* If yes, store it. If no, skip it.

## When to Use This Skill

### When starting: search before doing work

Run a search before diving into a codebase, or starting an activity. Prior agents may have stored context that reframes the problem:

```bash
python3 .apm/skills/skill-memory/memory.py search "merge conflict resolution"
```

Content can also be added under namespaces (e.g. `architecture`) to compartmentalise it and keep it scoped to that area. For example, if you're working on a performance-related task, search the `architecture` namespace for any relevant design decisions:

```bash
python3 .apm/skills/skill-memory/memory.py --namespace "architecture" search "performance requirements"
```

If no results come up, try broadening your query or searching without a namespace to see if related context was stored elsewhere. If no namespace is specified, the `default` namespace is used.

**Interpreting results:** A strong match is a result whose text directly addresses your question — same pattern, same decision space, same file area. Treat distance scores below ~0.8 as potentially relevant; above ~1.2 as noise. Skim the top 3–5 results rather than acting on a single hit. If results look unrelated, your query may be too specific — try broader terms.

### When wrapping up: write what you'd want to know next time

After completing work, add memories for:

- Non-obvious things you discovered
- Design decisions and why they were made
- Patterns that caused errors and how to avoid them
- Anything that surprised you and would surprise the next agent
- Any type of information that would change how you'd approach a similar task in the future

```bash
python3 .apm/skills/skill-memory/memory.py --namespace lessons-learned add "The --namespace flag must come before the subcommand name, not after it. Wrong: memory.py add --namespace foo. Right: memory.py --namespace foo add." --source B-abc12def
```

Keep entries focused and self-contained. Each memory should make sense on its own without surrounding context.

## Namespace Conventions

Use the `--namespace` flag to partition memories by feature or agent type. This prevents unrelated contexts from polluting search results.

Namespaces are arbitrary strings, including alphanumeric characters, hyphens, and underscores matching this regexp: [a-zA-Z0-9_-]. Here are some recommended conventions:

| Scenario | Recommended namespace |
|----------|-----------------------|
| Agent-type-specific patterns | `developer`, `tester`, `planner`, etc. |
| Feature-specific context | `feature-x`, `feature-x`, etc. |
| General project knowledge | `project`, `general`, or no namespace (defaults to `default`) |

Use one namespace per feature or per agent type — don't mix concerns in the same namespace. 

```bash
# Write under feature namespace
python3 .apm/skills/skill-memory/memory.py --namespace feature-1234 add "TEXT"

# Search the feature namespace
python3 .apm/skills/skill-memory/memory.py --namespace feature-1234 search "QUERY"

# Write project-wide knowledge
python3 .apm/skills/skill-memory/memory.py --namespace project add "TEXT"
```

## Getting Started

Run `python3 .apm/skills/skill-memory/memory.py <command> [options]` from the repository root. The DB auto-initialises on first write.

**First run**: `.apm/skills/skill-memory/memory.py` automatically creates a `.venv` in the skill directory and installs its dependencies (`onnxruntime`, `sqlite-vec`, `tokenizers`, `numpy`). This requires internet access and takes about a minute. Subsequent runs are instant.

**If the first run fails** (network timeout, interrupted install): the `.venv` may be left in a partial state. Delete it and retry:

```bash
rm -rf .apm/skills/skill-memory/.venv && python3 .apm/skills/skill-memory/memory.py --help
```

**Python requirement**: `.apm/skills/skill-memory/memory.py` checks at startup whether your Python supports SQLite extension-loading. If it doesn't, the process exits immediately with instructions — follow the message shown and do not attempt to work around it.

## Commands Reference

### `init` — Initialise the memory DB

Creates the `memories` and `memories_vec` tables. Safe to run multiple times (idempotent).

```bash
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] init [--db PATH]
```

### `add` — Add a single memory

```bash
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] add "TEXT" [--source TAG] [--meta KEY=VALUE ...] [--db PATH]
```

Use for short, standalone facts or observations discovered work. `--source` is an optional tag (e.g., a filename) for later filtering.

### `ingest` — Bulk-ingest a file

```bash
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] ingest FILE [--source TAG] [--chunk-size N] [--overlap N] [--column NAME] [--db PATH]
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
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] search "QUERY" [--limit N] [--threshold F] [--source TAG] [--json] [--db PATH]
```

Returns the top-`N` memories closest to the query vector (default: 5). `--threshold` is the maximum L2 distance to include (0–2; lower = more similar). `--json` outputs a JSON array.

### `list` — List memories

```bash
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] list [--limit N] [--source TAG] [--json] [--db PATH]
```

Lists memories newest-first (default limit: 20; `--limit 0` for all). Filter by `--source` to scope to a specific file, for example.

### `delete` — Delete a memory

```bash
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] delete UUID [--db PATH]
```

Removes the memory and its embedding vector by UUID.

### `stats` — Show DB statistics

```bash
python3 .apm/skills/skill-memory/memory.py [--namespace NAME] stats [--db PATH]
```

Prints the DB path, total memory count, and file size.

## DB Path Conventions

| Scenario | Recommendation |
|----------|----------------|
| Personal / cross-project context | Use the default (`memories.db` next to `.apm/skills/skill-memory/memory.py`) |
| Project-local context | Pass `--db .agent-memory.db` or set `AGENT_MEMORY_DB=.agent-memory.db` |
| CI / ephemeral environments | Set `AGENT_MEMORY_DB` to a temp path |

Prefer the shared default DB for general knowledge that spans projects. Use a project-local DB when memories are scoped to a single repository and should not bleed into other workspaces.
