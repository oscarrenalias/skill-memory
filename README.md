# agent-memory

Long-term semantic memory for agents — store, search, and retrieve text memories across sessions using SQLite and BAAI/bge-small-en-v1.5 ONNX embeddings.

## Installation

Install via APM (recommended):

```bash
apm install --t claude oscarrenalias/skill-agent-memory#v0.1.4
```

Please replace v0.1.4 with the right version, check the [GitHub releases page](https://github.com/oscarrenalias/skill-memory/tags) to check the latest available version.

Alternatively, clone manually and add to your Claude Code skills path.

Upon loading the skill, the model will initialize its own dependencies (it's self-contained but it will pull libraries) as well as initialize the local database.

## Usage

The main interface is `memory.py`:

```bash
python3 memory.py <command> [options]
```

### Common commands

These should not be required by human operators as the agent will drive everything through the skill. They are only provided here for reference.

```bash
# Store a memory
python3 memory.py add "The auth service uses RS256 JWTs, not HS256." --source myproject

# Search memories
python3 memory.py search "authentication token format" --limit 5

# Ingest a file (splits into chunks automatically)
python3 memory.py ingest notes.md --source myproject

# List recent memories
python3 memory.py list --limit 20

# Delete a memory by UUID
python3 memory.py delete <uuid>

# Show DB stats
python3 memory.py stats
```

### DB location

| Scenario | Path |
|----------|------|
| Default (cross-project) | `~/.local/share/agent-memory/memories.db` |
| Project-local | `--db .agent-memory.db` or `AGENT_MEMORY_DB=.agent-memory.db` |
| CI / ephemeral | Set `AGENT_MEMORY_DB` to a temp path |

Use the default for general knowledge that spans projects. Use a project-local DB for memories scoped to a single repository.

### What to store

Store project-wide, reusable facts that would change your approach in a future session if you had known them upfront. Do **not** store ephemeral task state, git history, or content trivially readable from the repo. See `docs/memory/conventions.md` for full guidance.

---

## Developer Guide

### Repo layout

```
memory.py          # Main CLI entry point (bootstraps .venv on first run)
SKILL.md           # APM skill manifest and usage reference
apm.yml            # APM package metadata
docs/memory/       # conventions.md, known-issues.md, memory-cli.md
tests/             # pytest test suite
specs/             # Takt feature specs
templates/         # Takt agent guardrail templates
```

### Running tests

```bash
.venv/bin/pytest
```

The project manages its own virtualenv at `.venv/`. Use `.venv/bin/pytest` rather than a system-level `pytest`.

### How specs and beads work

This project uses [takt](https://github.com/oscar-renalias/takt) for feature orchestration. Each feature is described in a spec under `specs/`, then broken into **beads** — atomic units of work assigned to planner, developer, tester, or documentation agents. Beads run in isolated git worktrees under `.takt/worktrees/` and are merged back to the feature branch on completion.

To add a new feature, create a spec in `specs/` and run the takt scheduler to generate beads.
