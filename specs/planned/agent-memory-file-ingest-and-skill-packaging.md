---
name: "agent-memory: file ingest and skill packaging"
id: spec-7d29a065
description: "ingest command, SKILL.md agent guidance, apm.yml manifest, and test suite"
dependencies: []
priority: medium
complexity: medium
status: planned
tags:
- agent-memory
- ingest
- skill-packaging
- apm
- tests
scope:
  in: "|"
  out: "|"
feature_root_id: null
---

# agent-memory: file ingest and skill packaging

## Objective

Complete the `agent-memory` skill by adding bulk file ingestion, the distributable skill manifest (`SKILL.md` + `apm.yml`), and a test suite. After this spec, the skill is ready to be installed via APM from a git URL and used by agents in any takt-orchestrated project.

## Problems to Fix

1. After Specs 1 and 2, adding memories is limited to one text at a time. Agents need to be able to seed the memory store from existing files (documentation, notes, prior context).
2. The skill has no distribution packaging (SKILL.md, apm.yml) so it cannot be installed by other projects.
3. There are no automated tests; the implementation cannot be validated without manual inspection.

## Changes

### 1. `ingest` command

```
memory.py ingest <file> [--source TAG] [--chunk-size N] [--overlap N] [--db PATH]
```

Reads `<file>`, splits it into chunks, and calls the same insert logic as `add` for each chunk.

`--source` defaults to the file's basename if not provided.
`--chunk-size` is the target chunk size in characters (default: 1000).
`--overlap` is the character overlap between adjacent chunks (default: 100).

> **⚠ PENDING DECISION — see §Pending Decisions #1:** Whether chunk-size is measured in characters or tokens. Characters are simpler and do not require loading the tokenizer just for chunking; tokens are more precise for the model. Spec currently specifies characters.

**Supported formats:**

| Extension | Chunking strategy |
|---|---|
| `.txt` | Split on blank-line paragraph boundaries; merge short paragraphs up to `--chunk-size`; split oversized paragraphs at sentence boundaries |
| `.md` | Same as `.txt` but treat level-2 headings (`## `) as hard chunk boundaries regardless of size, keeping the heading text in the following chunk |
| `.json` | Expect a JSON array. Each element is either a string (used as content) or an object with a `"content"` key (required) and optional `"metadata"` object and `"source"` string. Array of objects with neither `"content"` nor string elements → exit 1 with a descriptive error. |
| `.csv` | Require `--column NAME` flag specifying which column contains the text. All other columns are merged into the chunk's metadata JSON. First row is treated as a header. |

> **⚠ PENDING DECISION — see §Pending Decisions #2:** Whether to support other extensions (e.g. `.rst`, `.html`) in a future spec or raise a clear "unsupported format" error now. Spec currently raises an error for unknown extensions.

Progress output to stderr:
```
Ingesting notes.md… 12 chunks added (0 skipped).
```

Chunks shorter than 10 characters (after stripping whitespace) are silently skipped. The skipped count is included in the summary line.

### 2. SKILL.md

Create `SKILL.md` at the repo root with frontmatter and agent-facing guidance.

**Frontmatter:**

```yaml
---
name: agent-memory
description: Long-term semantic memory for agents — store, search, and retrieve text memories across sessions using SQLite and BAAI/bge-small-en-v1.5 embeddings.
tools: Bash
user-invocable: true
---
```

> **⚠ PENDING DECISION — see §Pending Decisions #3:** Whether `user-invocable` should be `true` (exposes the skill as a slash command for human users) or `false` (agent-only). Setting it `true` allows humans to use `/agent-memory` interactively; `false` restricts it to agent automation contexts.

**Body sections** (agent guidance):

- **Getting Started**: how to locate `memory.py` (same pattern as `spec.py` in `skill-spec-management/SKILL.md`); auto-init behaviour
- **When to use this skill**: at bead start (`search` for relevant prior context); at bead end (`add` notable findings); after completing a feature (`ingest` the spec or implementation notes)
- **Commands reference**: one paragraph per command with the full CLI signature and a one-sentence description of when to use it
- **DB path conventions**: default location, how to use `--db` for project-local DBs, when to prefer per-project vs shared DBs
- **What NOT to store**: ephemeral task state, bead-specific details, information already in CLAUDE.md (mirrors the memory guidance in `docs/memory/conventions.md`)

### 3. apm.yml

```yaml
name: agent-memory
version: 0.1.0
description: Long-term semantic memory for agents via SQLite + sqlite-vec + BAAI/bge-small-en-v1.5 ONNX embeddings
author: Renalias, Oscar
dependencies:
  apm: []
  mcp: []
scripts: {}
```

### 4. Test suite

Create `tests/__init__.py` (empty) and `tests/test_memory.py` at the repo root.

Use the same `_TempDirTest` base-class pattern as `skill-spec-management/tests/test_spec.py`: each test `chdir`s into a fresh `tempfile.mkdtemp()`.

Override the DB path via `AGENT_MEMORY_DB` env var pointing to a temp file, so tests do not touch the user's real DB.

**Test classes and coverage:**

| Class | What it tests |
|---|---|
| `TestInit` | init is idempotent; tables exist after init |
| `TestAdd` | add returns an ID; row present in both tables; auto-init on first add |
| `TestSearch` | search returns the most similar text; --limit respected; --json output is valid |
| `TestDelete` | delete removes from both tables; delete of unknown ID exits 1 |
| `TestList` | list returns newest-first; --source filter; --limit 0 returns all |
| `TestStats` | stats on populated DB; stats on missing DB exits 0 |
| `TestIngestTxt` | paragraph chunking; short chunks skipped |
| `TestIngestMd` | heading boundaries respected |
| `TestIngestJson` | string array; object array with content key; invalid input exits 1 |
| `TestIngestCsv` | --column flag; other columns become metadata |

Tests that require the embedding model (all tests involving actual embedding) must either:
- Use a real model download (integration test, slow), OR
- Mock `_embed()` to return a deterministic random unit vector of shape `(N, 384)`

> **⚠ PENDING DECISION — see §Pending Decisions #4:** Whether the test suite should mock `_embed` (fast, no network) or use the real model (slow, requires download, higher confidence). A pragmatic split: mock by default; one integration test class guarded by an env var `AGENT_MEMORY_INTEGRATION_TESTS=1`.

## Files to Add / Modify

| File | Change |
|---|---|
| `memory.py` | Add ingest subcommand handler |
| `SKILL.md` | New — agent-facing skill documentation |
| `apm.yml` | New — APM package manifest |
| `tests/__init__.py` | New — empty package marker |
| `tests/test_memory.py` | New — full test suite |

## Acceptance Criteria

- `memory.py ingest notes.txt` adds at least one chunk and prints a summary line to stderr.
- A `.txt` file with 3 paragraphs produces 3 separate memory rows (assuming each paragraph is under `--chunk-size`).
- A `.md` file with a `## Section` heading followed by body text produces a chunk that starts with the heading text.
- `memory.py ingest data.json` with `[{"content": "hello"}, {"content": "world"}]` adds exactly 2 rows.
- `memory.py ingest data.csv --column text` adds one row per CSV data row; other columns appear in the metadata JSON.
- Chunks shorter than 10 characters are skipped; the summary line reports the skipped count.
- `memory.py ingest unknown.xyz` exits with code 1 and a "unsupported format" error.
- `python3 -m unittest discover -s tests/` passes with all tests green (mocked embed).
- `SKILL.md` exists with valid frontmatter and a "When to use this skill" section.
- `apm.yml` is valid YAML and contains `name`, `version`, `description`, `author`, `dependencies`, `scripts`.

## Pending Decisions

1. **Chunk-size unit — characters vs tokens**: Character-based chunking is simpler and avoids loading the tokenizer at ingest time. Token-based is more precise (model max is 512 tokens). The spec currently specifies characters. *Decision owner: engineer implementing Spec 3. If tokens are chosen, update `--chunk-size` help text and default to 400 tokens.*

2. **Unsupported file formats**: Whether `ingest` should fail immediately on unsupported extensions or attempt to read the file as plain text. The spec currently exits 1 with a clear error. *This is a low-stakes decision; the implementing agent may use their judgement and document it in a comment.*

3. **`user-invocable` flag in SKILL.md**: `true` makes the skill accessible as a human-facing slash command (`/agent-memory`); `false` restricts it to agent automation. Setting `true` is consistent with `skill-spec-management`. *Recommend `true` unless there is a specific reason to restrict human access.*

4. **Test embedding strategy**: The spec recommends mocking `_embed` for unit tests and using a real model for integration tests behind an env var flag. If the model download is unreliable in CI environments, the integration test class should be skipped rather than fail. *Implementing agent should confirm this is acceptable or propose an alternative.*
