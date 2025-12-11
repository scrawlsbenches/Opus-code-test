# CLI Wrapper Guide

A quick reference for the `cortical.cli_wrapper` module - designed for AI assistants working on code.

## Design Philosophy

**Quiet by default, powerful when needed.**

- No emoji, no unsolicited suggestions
- Git context is opt-in, not automatic
- Data is available when you ask, silent otherwise

## Quick Reference

### Simple Command (90% of cases)

```python
from cortical.cli_wrapper import run

result = run("pytest tests/")
if result.success:
    # continue
else:
    print(result.stderr)
```

### With Git Context

```python
result = run("git status", git=True)
print(result.git.branch)
print(result.git.modified_files)
```

### Session Tracking

```python
from cortical.cli_wrapper import Session

with Session() as s:
    s.run("pytest tests/")
    s.run("git add -A")
    s.run("git commit -m 'fix'")

    if s.should_reindex():
        # Corpus was modified, consider re-indexing
        s.run("python scripts/index_codebase.py -i")

    print(s.all_passed)  # True/False
    print(s.summary())   # Dict with stats
```

### Compound Commands

```python
from cortical.cli_wrapper import test_then_commit, commit_and_push, sync_with_main

# Only commit if tests pass
ok, results = test_then_commit(message="Fix auth bug")

# Add + commit + push in one call
ok, _ = commit_and_push("Quick fix")

# Fetch + rebase on main
ok, _ = sync_with_main()
```

### Task Checkpointing (for context switching)

```python
from cortical.cli_wrapper import TaskCheckpoint

checkpoint = TaskCheckpoint()

# Before switching tasks, save context
checkpoint.save("feature-auth", {
    'branch': 'feature/auth',
    'notes': 'Need to add token refresh',
    'files': ['auth.py', 'test_auth.py'],
})

# Later, resume
ctx = checkpoint.load("feature-auth")
print(ctx['notes'])  # "Need to add token refresh"

# See all checkpoints
checkpoint.list_tasks()  # ['feature-auth', 'bugfix-123']
```

### Hooks (for automation)

```python
from cortical.cli_wrapper import CLIWrapper

wrapper = CLIWrapper()

@wrapper.on_success("pytest")
def after_tests(result):
    print(f"Tests passed in {result.duration:.1f}s")

@wrapper.on_error()
def on_any_failure(result):
    # Log, alert, etc.
    pass
```

## ExecutionContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Did the command succeed? |
| `exit_code` | int | Process exit code |
| `stdout` | str | Standard output |
| `stderr` | str | Standard error |
| `duration` | float | Execution time in seconds |
| `command` | List[str] | Command that was run |
| `command_str` | str | Command as string |
| `git` | GitContext | Git info (if `git=True`) |
| `task_type` | str | Classified type (test, commit, etc.) |

## GitContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_repo` | bool | Inside a git repo? |
| `branch` | str | Current branch name |
| `commit_hash` | str | Short commit hash |
| `is_dirty` | bool | Uncommitted changes? |
| `modified_files` | List[str] | Modified but unstaged |
| `staged_files` | List[str] | Staged for commit |
| `untracked_files` | List[str] | Not tracked by git |

## When to Use What

| Situation | Use |
|-----------|-----|
| Run one command, check result | `run()` |
| Need git branch/status | `run(..., git=True)` |
| Multiple related commands | `Session()` |
| Test before committing | `test_then_commit()` |
| Quick commit + push | `commit_and_push()` |
| Stay updated with main | `sync_with_main()` |
| Switching between tasks | `TaskCheckpoint` |
| Custom automation | `CLIWrapper` + hooks |

## Key Design Decisions

1. **`git=False` by default** - Avoids subprocess overhead when you don't need git info
2. **No global state** - Each `run()` is independent; use `Session` for stateful tracking
3. **Compound commands return `(bool, List[results])`** - Always know if it worked and what happened
4. **Hooks are opt-in** - Register them explicitly, no surprise callbacks
5. **Checkpoints are JSON files** - Human-readable, git-friendly, easy to inspect

## File Locations

- **Core module**: `cortical/cli_wrapper.py`
- **Tests**: `tests/test_cli_wrapper.py` (60 tests)
- **Example script**: `scripts/cli_wrappers.py`
