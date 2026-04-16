---
name: Restructure repo layout — isolate skill under .apm/skills/skill-memory
id: spec-073a7f9e
description: "Move skill artefacts (memory.py, SKILL.md, apm.yml, assets/) from repo root to .apm/skills/skill-memory/ so APM only installs the skill, not the full repo scaffolding"
dependencies: null
priority: high
complexity: null
status: done
tags:
- refactor
- repo-structure
scope:
  in: null
  out: null
feature_root_id: B-5c40fb01
---
# Restructure repo layout — isolate skill under .apm/skills/skill-memory

## Objective

The reference layout (see `oscarrenalias/skill-office`) places each skill under `.apm/skills/<skill-name>/`, with a repo-level `apm.yml` that uses `compilation.exclude` to strip scaffolding dirs from distribution. This means APM only installs the skill itself — not the takt pipeline data, specs, templates, tests, or CI config.

Currently, `memory.py`, `SKILL.md`, `apm.yml`, and `assets/` all live at the repo root, causing APM to install the entire repository.

## Changes

### 1. Move skill artefacts (use `git mv`)

```
memory.py   → .apm/skills/skill-memory/memory.py
SKILL.md    → .apm/skills/skill-memory/SKILL.md
apm.yml     → .apm/skills/skill-memory/apm.yml
assets/     → .apm/skills/skill-memory/assets/
README.md   → .apm/skills/skill-memory/README.md
```

`memory.py` uses `_SKILL_DIR = Path(__file__).parent` for all relative paths (`.venv`, `memories.db`, `assets/`), so no changes to `memory.py` internals are needed — all paths resolve correctly at the new location.

### 2. Replace root `apm.yml`

Create a new root `apm.yml` as the repo-level manifest (the moved file becomes the skill-level manifest):

```yaml
name: skill-memory
version: 0.1.0
description: Long-term semantic memory for agents via SQLite + sqlite-vec + BAAI/bge-small-en-v1.5 ONNX embeddings
author: "Renalias, Oscar"
dependencies:
  apm: []
  mcp: []
scripts: {}
compilation:
  exclude:
    - "specs/**"
    - "tests/**"
    - ".takt/**"
    - ".claude/**"
    - ".agents/**"
    - "templates/**"
    - "docs/**"
```

### 3. Update test path constants

All test files compute the path to `memory.py` as `Path(__file__).parent.parent / "memory.py"` (from `tests/`) or `Path(__file__).parent / "memory.py"` (from repo root). Update each to the new location.

Files to update — change only the path constant, not test logic:

| File | Current constant | New constant |
|------|-----------------|--------------|
| `tests/test_add_command.py` line 23 | `parent.parent / "memory.py"` | `parent.parent / ".apm/skills/skill-memory/memory.py"` |
| `tests/test_db_init.py` line 18 | same pattern | same fix |
| `tests/test_embed.py` line 23 | same pattern | same fix |
| `tests/test_ensure_model.py` line 18 | same pattern | same fix |
| `tests/test_ingest_chunking.py` line 34 | `os.path.join(repo_root, "memory.py")` | `os.path.join(repo_root, ".apm/skills/skill-memory/memory.py")` |
| `tests/test_ingest_json_csv.py` line 34 | same pattern | same fix |
| `tests/test_memory.py` line 48 | `os.path.join(_REPO_ROOT, "memory.py")` | `os.path.join(_REPO_ROOT, ".apm/skills/skill-memory/memory.py")` |
| `tests/test_memory_cli.py` line 11 | `parent.parent / "memory.py"` | same fix |
| `tests/test_namespace.py` line 41 | `os.path.join(_REPO_ROOT, "memory.py")` | same fix |
| `tests/test_tokenizer_assets.py` line 9 | `parent.parent / "assets"` | `parent.parent / ".apm/skills/skill-memory/assets"` |
| `test_commands.py` line 21 | `_SKILL_DIR / "memory.py"` | `Path(__file__).parent / ".apm/skills/skill-memory/memory.py"` |

### 4. Update `.github/workflows/test.yml`

Three lines change:

```yaml
# Cache key (line 16)
key: venv-${{ hashFiles('.apm/skills/skill-memory/memory.py') }}

# Bootstrap step (line 19-20)
run: |
  python .apm/skills/skill-memory/memory.py --help
  .apm/skills/skill-memory/.venv/bin/pip install --quiet pytest

# Test step (line 22)
run: .apm/skills/skill-memory/.venv/bin/pytest tests/ test_commands.py
```

### 5. Update `.gitignore`

Add entries for the new locations (keep or remove root-level ones):

```
.apm/skills/skill-memory/.venv/
.apm/skills/skill-memory/memories.db
.apm/skills/skill-memory/__pycache__/
```

## Acceptance Criteria

1. `python3 .apm/skills/skill-memory/memory.py --help` runs successfully and bootstraps `.venv` inside `.apm/skills/skill-memory/`
2. `python3 .apm/skills/skill-memory/memory.py stats` returns DB stats without error
3. `memory.py` does **not** exist at the repo root
4. `assets/` does **not** exist at the repo root
5. Root `apm.yml` has a `compilation.exclude` section
6. `.apm/skills/skill-memory/apm.yml` exists with the skill-level metadata
7. All tests pass: `.apm/skills/skill-memory/.venv/bin/pytest tests/ test_commands.py`

## Pending Decisions

None.
