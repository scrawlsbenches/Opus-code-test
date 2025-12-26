# Knowledge Transfer: GoT Thread Safety Race Condition Fixes

**Date:** 2025-12-26
**Session:** X4EON
**Branch:** `claude/accept-handoff-X4EON`

## Summary

Fixed race conditions in GoT concurrent operations that caused "No such file or directory" errors when multiple threads tried to update `.tmp` files simultaneously.

## Root Cause Analysis

### The Problem

Both `VersionedStore._save_version()` and `QueryIndexManager._atomic_write_json()` used atomic file operations:
1. Write to `.tmp` file
2. Fsync
3. Rename `.tmp` to final file

When multiple threads ran concurrently:
- Thread A writes to `_version.tmp`
- Thread B overwrites `_version.tmp`
- Thread A tries to rename `_version.tmp` → **File not found** (already renamed by B)

### Why ProcessLock Wasn't Enough

`ProcessLock` uses `fcntl.flock()` which provides **process-level** locking only. Within the same process, all threads share the same file descriptor table, so multiple threads can all "hold" the same flock simultaneously.

```
Process boundary
├── Thread A: flock(fd) → SUCCESS (process now holds lock)
├── Thread B: flock(fd) → SUCCESS (process already holds lock!)
└── Both threads proceed concurrently → RACE CONDITION
```

## The Fix

Added `threading.Lock` for intra-process thread safety while keeping `ProcessLock` for inter-process safety:

### VersionedStore (cortical/got/versioned_store.py)

```python
# In __init__:
self._version_thread_lock = threading.Lock()
self._version_lock = ProcessLock(self.store_dir / ".version.lock", reentrant=False)

# In _save_version():
with self._version_thread_lock:      # Thread safety
    with self._version_lock:          # Process safety
        # Atomic write operations
```

### QueryIndexManager (cortical/got/indexer.py)

```python
# In __init__:
self._write_lock = threading.Lock()

# In _atomic_write_json():
with self._write_lock:
    # Atomic write operations
```

## Tests Enabled

Previously 7 tests were skipped due to these race conditions. All now pass:

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_got_index_concurrent.py` | 6 | ✅ All passing |
| `test_got_index_integration.py` | 9 | ✅ All passing |

**Total: 15 concurrent tests passing**

## Commits

| Commit | Description |
|--------|-------------|
| `584c310e` | fix(got): Add threading.Lock to VersionedStore._save_version() |
| `037fe8a0` | fix(got): Add threading.Lock to QueryIndexManager._atomic_write_json() |

## Tasks Completed

- **T-20251226-132353-68a469de**: Fix VersionedStore race condition in _version.tmp
- **T-20251226-191308-26fc6a6c**: Fix QueryIndexManager race condition in _atomic_write_json

## Key Insight for Future Reference

**When using `fcntl.flock()` (via ProcessLock) in multi-threaded code, always add a `threading.Lock` wrapper.** The flock provides process isolation but not thread isolation.

This pattern should be applied anywhere atomic file operations are performed with potential concurrent access:
1. Check if code can be called from multiple threads
2. If yes, wrap with `threading.Lock`
3. Keep `ProcessLock` for cross-process safety

## Files Modified

- `cortical/got/versioned_store.py` - Added `_version_thread_lock`
- `cortical/got/indexer.py` - Added `_write_lock`
- `tests/unit/test_got_index_concurrent.py` - Removed skip markers, updated docstring
- `tests/unit/test_got_index_integration.py` - Removed skip marker

## Verification Checklist for Next Agent

- [ ] Run `python -m pytest tests/unit/test_got_index_concurrent.py -v` - should see 6 passed
- [ ] Run `python -m pytest tests/unit/test_got_index_integration.py -v` - should see 9 passed
- [ ] Run `python -m pytest tests/smoke/ -q` - should see 18 passed
- [ ] Run `python scripts/got_utils.py validate` - should show HEALTHY
- [ ] Review commits 584c310e and 037fe8a0 for correctness
