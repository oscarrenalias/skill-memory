# agent-memory

Long-term semantic memory for agents — store, search, and retrieve text memories across sessions using SQLite and BAAI/bge-small-en-v1.5 ONNX embeddings.

## Installation

### Via APM (recommended)

```bash
apm install --t claude oscarrenalias/skill-memory
```

Check the [GitHub releases page](https://github.com/oscarrenalias/skill-memory/releases) for the latest version.

### Via zip (manual)

Download the latest `skill-memory-vX.Y.Z.zip` from the [GitHub releases page](https://github.com/oscarrenalias/skill-memory/releases), then unzip into your agent skills directory:

```bash
unzip skill-memory-vX.Y.Z.zip -d /path/to/your/skills/
```

The zip contains a single `skill-memory/` directory — unzipping it directly into your skills folder gives the correct layout with no extra steps.

Upon loading the skill, the model initialises its own dependencies (self-contained, pulls libraries on first run) and creates the local database automatically.

### Python requirement

`memory.py` requires a Python build compiled with SQLite extension-loading support. The script checks this at startup and exits with instructions if the requirement is not met. If you see the error, install Homebrew Python on macOS:

```bash
brew install python
```

Then re-run `python3 memory.py --help` from the skill directory to re-bootstrap the `.venv` against the new Python.

## Usage

When loaded as a skill, the agent drives all commands — no manual invocation is needed. The CLI is documented here for reference and debugging. Commands below assume you are running from the skill directory, or replace `memory.py` with the full path to it.

```bash
python3 memory.py <command> [options]
```

### Common commands

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
| Default (cross-project) | `memories.db` next to `memory.py` |
| Project-local | `--db .agent-memory.db` or `AGENT_MEMORY_DB=.agent-memory.db` |
| CI / ephemeral | Set `AGENT_MEMORY_DB` to a temp path |

Use the default for general knowledge that spans projects. Use a project-local DB for memories scoped to a single repository.

### What to store

Store project-wide, reusable facts that would change your approach in a future session if you had known them upfront. Do **not** store ephemeral task state, git history, or content trivially readable from the repo.

---

## Developer Guide

### Repo layout

```
.apm/skills/skill-memory/
  memory.py        # Main CLI entry point (bootstraps .venv on first run)
  SKILL.md         # Skill manifest and usage reference
  apm.yml          # Skill-level APM package metadata
  assets/          # ONNX tokenizer model files (vocab, config, tokenizer.json)
README.md          # This file (repo root)
apm.yml            # Repo-level APM manifest with compilation.exclude (repo root)
tests/             # pytest test suite
specs/             # Takt feature specs
templates/         # Takt agent guardrail templates
```

### Running tests

```bash
.apm/skills/skill-memory/.venv/bin/pytest tests/ test_commands.py
```

Bootstrap the venv first if it doesn't exist: `python3 .apm/skills/skill-memory/memory.py --help`. The project manages its own virtualenv at `.apm/skills/skill-memory/.venv/`.

### How specs and beads work

This project uses [takt](https://github.com/oscar-renalias/takt) for feature orchestration. Each feature is described in a spec under `specs/`, then broken into **beads** — atomic units of work assigned to planner, developer, tester, or documentation agents. Beads run in isolated git worktrees under `.takt/worktrees/` and are merged back to the feature branch on completion.

To add a new feature, create a spec in `specs/` and run the takt scheduler to generate beads.
