# GoT (Graph of Thought) Process Safety

Task: T-20251221-114358-2b0b - Add thread safety to GoT graph operations

## Overview

The GoT system uses file-based locking to ensure process-safe operations on the graph. This prevents data corruption when multiple processes (or Claude Code sessions) attempt to modify the graph concurrently.

## Locking Mechanism

### ProcessLock Class

Located in `scripts/got_utils.py`, the `ProcessLock` class provides:

```python
from got_utils import ProcessLock

lock = ProcessLock(
    lock_path=Path(".got/.got.lock"),
    stale_timeout=3600.0,  # 1 hour
    reentrant=True         # Allow same process to re-acquire
)

with lock:
    # Protected operations here
    pass
```

### Features

| Feature | Description |
|---------|-------------|
| **File-based locking** | Uses `fcntl.flock()` for POSIX-compliant exclusive locks |
| **PID tracking** | Lock file contains PID and timestamp for debugging |
| **Stale lock detection** | Automatically recovers from locks held by dead processes |
| **Timeout support** | `acquire(timeout=N)` for non-blocking attempts |
| **Reentrant option** | Same process can acquire lock multiple times |
| **Context manager** | `with lock:` syntax for automatic release |
| **Exception safety** | Lock released even if exception occurs |
| **Corrupted file recovery** | Handles malformed lock files gracefully |

### Stale Lock Recovery

A lock is considered stale if:
1. The owning process (PID in lock file) no longer exists
2. The lock is older than `stale_timeout` (default: 1 hour)

Stale locks are automatically acquired by the next process.

## Protected Operations

The following `GoTProjectManager` methods are protected by the lock:

| Method | Description |
|--------|-------------|
| `create_task()` | Create a new task node |
| `update_task_status()` | Update task status |
| `complete_task()` | Mark task as complete |
| `block_task()` | Block a task with reason |
| `delete_task()` | Delete a task (with safety checks) |
| `add_dependency()` | Add DEPENDS_ON edge |
| `add_blocks()` | Add BLOCKS edge |
| `log_decision()` | Log a decision node |

### Read Operations

Read operations (`get_task()`, `list_tasks()`, `get_decisions()`, etc.) are **NOT** locked. This follows a readers-writer pattern where:
- Multiple readers can proceed concurrently
- Writers (mutations) are serialized via the lock

This is safe because:
- The event log is append-only (atomic writes via `atomic_append()`)
- Graph state is rebuilt from events on load
- Readers see a consistent snapshot at load time

## Usage Examples

### Multiple Claude Code Sessions

When multiple sessions work on the same repository:

```
Session A                     Session B
    │                             │
    ├─ create_task()             │
    │   └─ acquires lock         │
    │   └─ creates task          ├─ create_task()
    │   └─ releases lock         │   └─ waits for lock
    │                             │   └─ acquires lock
    │                             │   └─ creates task
    │                             │   └─ releases lock
```

### Parallel Agent Orchestration

When using Director pattern with sub-agents:

```python
# Director spawns multiple agents
# Each agent uses the same .got directory
# Locking ensures safe concurrent access

# Agent 1
manager = GoTProjectManager(got_dir=".got")
manager.update_task_status(task_id, "in_progress")  # Locked

# Agent 2 (concurrent)
manager2 = GoTProjectManager(got_dir=".got")
manager2.complete_task(other_task_id)  # Waits for lock, then proceeds
```

## Lock File Location

The lock file is created at: `{got_dir}/.got.lock`

Default: `.got/.got.lock`

## Error Handling

### Lock Acquisition Failure

If lock cannot be acquired within timeout:
- `acquire(timeout=N)` returns `False`
- Context manager raises exception after default timeout

### Dead Lock Recovery

If a process crashes while holding the lock:
- Next process detects the PID no longer exists
- Lock is automatically stolen after timeout
- No manual intervention required

### Corrupted Lock File

If lock file contains garbage:
- File is deleted and re-created
- Operation proceeds normally

## Testing

Tests are in `tests/unit/test_process_lock.py`:

| Test Class | Coverage |
|------------|----------|
| `TestProcessLockBasics` | Lock creation, PID tracking, release |
| `TestProcessLockTimeout` | Timeout behavior |
| `TestStaleLockRecovery` | Dead process detection |
| `TestProcessLockReentrancy` | Re-entrant acquisition |
| `TestProcessLockErrorHandling` | Exception safety |
| `TestGoTManagerLocking` | Integration with GoTProjectManager |
| `TestMultiProcessSafety` | Cross-process locking |

Run tests:
```bash
python -m pytest tests/unit/test_process_lock.py -v
```

## Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Full support | Uses `fcntl.flock()` |
| macOS | Full support | Uses `fcntl.flock()` |
| Windows | Limited | `fcntl` not available, falls back to no-op |

## Best Practices

1. **Use GoTProjectManager methods** - Don't modify files directly
2. **Don't hold locks long** - Keep operations atomic
3. **Handle lock failures gracefully** - Check return values or catch exceptions
4. **Use reentrant locks for nested calls** - Default is reentrant=True

## Related Files

- `scripts/got_utils.py` - ProcessLock class and GoTProjectManager
- `tests/unit/test_process_lock.py` - Unit tests
- `.got/.got.lock` - Lock file (auto-created)
- `.got/events/` - Event log (append-only, atomic writes)
