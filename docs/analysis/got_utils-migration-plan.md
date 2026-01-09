# Migration Plan: got_utils.py → cortical.got

**Goal:** Enable `python -m cortical.got` as CLI entry point, eliminate scripts/got_utils.py

---

## Current State (3365 lines in got_utils.py)

### Classes to Move

| Class | Lines | Target Location | Notes |
|-------|-------|-----------------|-------|
| `GoTBackendFactory` | 619-650 (~30) | `cortical/got/factory.py` | Simple factory |
| `TransactionalGoTAdapter` | 656-3058 (~2400) | `cortical/got/adapter.py` | Main manager wrapper |

### Utility Functions to Move

| Function | Lines | Target | Notes |
|----------|-------|--------|-------|
| `_get_auto_committer` | 227-240 | `cortical/got/git_integration.py` | Git auto-commit |
| `_build_descriptive_commit_message` | 242-342 | `cortical/got/git_integration.py` | Commit messages |
| `_generic_commit_message` | 344-349 | `cortical/got/git_integration.py` | |
| `got_auto_commit` | 351-420 | `cortical/got/git_integration.py` | |
| `_got_auto_push` | 422-504 | `cortical/got/git_integration.py` | |
| `get_current_branch` | 506-517 | `cortical/got/git_integration.py` | |
| `generate_session_id` | 519-550 | `cortical/utils/` (already there?) | |
| `has_task_reference` | 552-563 | `cortical/got/git_integration.py` | |
| `extract_commit_type` | 565-579 | `cortical/got/git_integration.py` | |
| `suggest_task_category` | 581-594 | `cortical/got/git_integration.py` | |
| `generate_task_title_from_commit` | 596-613 | `cortical/got/git_integration.py` | |
| `format_task_table` | 3060-3085 | `cortical/got/cli/shared.py` | Already has shared.py |
| `format_sprint_status` | 3087-3130 | `cortical/got/cli/shared.py` | |
| `suggest_command` | 3144-3163 | `cortical/got/cli/shared.py` | |
| `cmd_compact` | 3168-3177 | `cortical/got/cli/query.py` | Deprecated cmd |
| `print_command_suggestion` | 3179-3195 | `cortical/got/cli/shared.py` | |

### CLI Entry Point

| Function | Lines | Target | Notes |
|----------|-------|--------|-------|
| `main()` | 3197-3317 | `cortical/got/__main__.py` | Entry point |
| `_run_with_auto_commit` | 3319-3365 | `cortical/got/__main__.py` | Wrapper |

---

## Target Structure

```
cortical/got/
├── __init__.py          # Exports
├── __main__.py          # NEW: CLI entry point (main, _run_with_auto_commit)
├── adapter.py           # NEW: TransactionalGoTAdapter class
├── factory.py           # NEW: GoTBackendFactory
├── git_integration.py   # NEW: Auto-commit, git helpers
├── api.py               # Existing: GoTManager
├── cli/
│   ├── __init__.py
│   ├── shared.py        # Add: format_*, suggest_*, print_command_suggestion
│   ├── task.py
│   ├── sprint.py
│   ├── handoff.py
│   ├── query.py         # Add: cmd_compact
│   └── ...
└── ...
```

---

## Migration Steps

### Phase 1: Create New Modules
1. Create `cortical/got/adapter.py` with `TransactionalGoTAdapter`
2. Create `cortical/got/factory.py` with `GoTBackendFactory`
3. Create `cortical/got/git_integration.py` with git utilities
4. Create `cortical/got/__main__.py` with CLI entry point

### Phase 2: Update CLI Shared
1. Move `format_task_table`, `format_sprint_status` to `cli/shared.py`
2. Move `suggest_command`, `print_command_suggestion` to `cli/shared.py`

### Phase 3: Update Imports
1. Update `cortical/got/__init__.py` to export new classes
2. Update CLI modules to import from new locations

### Phase 4: Create Shim
1. Reduce `scripts/got_utils.py` to thin wrapper:
```python
#!/usr/bin/env python3
"""Legacy entry point - use 'python -m cortical.got' instead."""
from cortical.got.__main__ import main
if __name__ == "__main__":
    main()
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing scripts | Keep got_utils.py as shim |
| Import cycles | Careful module organization |
| Large adapter class | Consider splitting by domain |

---

## Line Count Estimates

| New File | Estimated Lines |
|----------|-----------------|
| adapter.py | ~2400 |
| factory.py | ~50 |
| git_integration.py | ~400 |
| __main__.py | ~200 |
| shared.py additions | ~150 |

**Total new code:** ~3200 lines (same as current, just reorganized)
**got_utils.py after:** ~10 lines (shim only)
