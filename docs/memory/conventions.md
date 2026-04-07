---
name: Conventions
description: Project conventions for bead orchestration
type: project
---

# Conventions

## Bead IDs

Bead IDs use the format `B-{8 hex chars}`. Child beads append suffixes:
`B-abc12def-test`, `B-abc12def-review`, `B-abc12def-docs`.

## Running Commands

All commands must be run from the project root. Never run commands from inside a
worktree unless the bead assignment explicitly requires it.

## Memory Append-Only Rule

New memory entries are appended; existing entries are never edited in place unless
explicitly correcting an error. This preserves the audit trail.

## Feature Branches

Each feature has a dedicated branch `feature/{feature-root-id-lowercase}` and a
worktree at `.takt/worktrees/{feature-root-id}`.

## Bead Lifecycle

Beads move through: `open` → `ready` → `in_progress` → `done` | `blocked` | `handed_off`.
Only the scheduler transitions beads out of `in_progress`. Do not manually mark a
developer bead `done` — use `takt merge` after work is complete.

## Running the Test Suite

Run the full test suite from the project root:

```bash
python3 -m unittest discover -s tests/
```

All tests mock `_embed()` with a deterministic unit-vector function so no model download
is required. Integration tests that exercise the real ONNX embedding model are guarded by
an environment variable and skipped by default:

```bash
AGENT_MEMORY_INTEGRATION_TESTS=1 python3 -m unittest discover -s tests/
```

Test classes and their coverage:

| Class | Coverage |
|---|---|
| `TestInit` | idempotent init; both tables exist after init |
| `TestAdd` | returns UUID; row in both `memories` and `memories_vec`; auto-init |
| `TestSearch` | most-similar text ranked first; `--limit`; `--json` output |
| `TestDelete` | removes from both tables; unknown ID exits 1 |
| `TestList` | newest-first order; `--source` filter; `--limit 0` returns all |
| `TestStats` | populated DB output; missing DB exits 0 |
| `TestIngestTxt` | paragraph chunking; short chunks skipped |
| `TestIngestMd` | `## ` heading boundaries; heading text in chunk |
| `TestIngestJson` | string array; object array with `content` key; invalid input exits 1 |
| `TestIngestCsv` | `--column` flag selects text column; other columns become metadata |
