# Knowledge Transfer: Concurrency & Crash-Safety Fixes Session

**Date:** 2025-12-26
**Session ID:** T2HnB (claude/accept-handoff-T2HnB)
**Agent:** opus-session
**Tags:** `concurrency`, `file-locking`, `thread-safety`, `crash-safety`, `got`, `race-conditions`

---

## Executive Summary

This session completed all critical and high-priority concurrency/crash-safety tasks from handoff `H-20251224-215434-555b0663`. Five fixes were implemented addressing race conditions at both process and thread levels, plus crash-safety for index writes.

**Commits Made:**
1. `da5d05e3` - versioned_store history locking
2. `1d644343` - ML collector JSONL locking  
3. `499c7561` - cache threading.Lock
4. `e28e0f26` - atomic index writes

---

## Problem Context

The GoT (Graph of Thought) system had multiple concurrency vulnerabilities:

1. **Process-level races**: Multiple Claude Code sessions or parallel agents could corrupt shared files
2. **Thread-level races**: LRU cache operations had check-then-modify patterns vulnerable to race conditions
3. **Crash-safety gaps**: Index writes weren't atomic - crashes during write could corrupt indexes

---

## Fixes Implemented

### 1. WAL File Locking (`cortical/got/wal.py`)

**Problem:** Multiple processes appending to WAL file could interleave writes, corrupting JSONL format.

**Solution:** Added `ProcessLock` around the `log()` method.

```python
from cortical.utils.locking import ProcessLock

# In __init__:
self._wal_lock = ProcessLock(self.wal_dir / ".wal.lock")

# In log():
def log(self, tx_id: str, operation: str, data: Dict[str, Any]) -> int:
    with self._wal_lock:
        seq = self._next_seq()
        entry = TransactionWALEntry(...)
        with open(self.wal_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict(), separators=(',', ':')) + '\n')
            f.flush()
            if self.durability == DurabilityMode.PARANOID:
                os.fsync(f.fileno())
        return seq
```

**Note:** Had to remove fsync from ProcessLock's holder info write to fix test `test_relaxed_mode_never_fsyncs_wal`. The holder info is just metadata for stale detection, not critical data.

---

### 2. Versioned Store History Locking (`cortical/got/versioned_store.py`)

**Problem:** `_save_to_history()` appends to JSONL files without locking - concurrent writes corrupt history.

**Solution:** Added `ProcessLock` for history directory.

```python
from cortical.utils.locking import ProcessLock

# In __init__:
self._history_lock = ProcessLock(self.history_dir / ".history.lock")

# In _save_to_history():
with self._history_lock:
    with open(history_path, 'a', encoding='utf-8') as f:
        json.dump(history_entry, f, sort_keys=True)
        f.write('\n')
```

---

### 3. ML Collector JSONL Locking (`scripts/ml_collector/persistence.py`)

**Problem:** `save_commit_lite()` and `save_session_lite()` have check-then-act race:
- Check if entry exists in file
- If not, append to file
- Two processes could both check, both find "not exists", both append (duplicate!)

**Solution:** Used existing `file_lock()` context manager to wrap the entire check-and-write operation.

```python
# Lock to prevent check-then-act race condition
with file_lock(COMMITS_LITE_FILE):
    # Legacy: Check if this commit hash already exists in the file (idempotent)
    if COMMITS_LITE_FILE.exists():
        with open(COMMITS_LITE_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        if existing.get("hash") == context.hash:
                            return COMMITS_LITE_FILE  # Already recorded
                    except json.JSONDecodeError:
                        continue

    # Append to JSONL file (one JSON object per line)
    with open(COMMITS_LITE_FILE, 'a') as f:
        f.write(json.dumps(lite_data, separators=(',', ':')) + '\n')

    # Also write to CALI for O(1) lookups
    if cali_put('commit', context.hash, lite_data):
        logger.debug(f"CALI: stored commit {context.hash[:8]}")
```

Same pattern applied to `save_session_lite()`.

---

### 4. Cache Thread-Safety (`cortical/got/api.py`)

**Problem:** LRU cache has check-modify-write race at lines 164-166:
```python
if entity_id in self._cache_access_order:
    self._cache_access_order.remove(entity_id)  # ValueError if another thread removed!
self._cache_access_order.append(entity_id)
```

**Solution:** Added `threading.Lock` and wrapped all cache operations.

```python
import threading

# In __init__:
self._cache_lock = threading.Lock()

# In _cache_get():
with self._cache_lock:
    entity = self._entity_cache.get(entity_id)
    if entity is None:
        return None
    # ... TTL check, LRU update ...
    return entity

# In _cache_set():
with self._cache_lock:
    # LRU eviction, entity storage, LRU update
    
# In _cache_invalidate():
with self._cache_lock:
    self._cache_invalidate_locked(entity_id)
    
# Helper for use within locked context (avoids deadlock):
def _cache_invalidate_locked(self, entity_id: str) -> None:
    self._entity_cache.pop(entity_id, None)
    self._cache_timestamps.pop(entity_id, None)
    if entity_id in self._cache_access_order:
        self._cache_access_order.remove(entity_id)
```

**Key insight:** Created `_cache_invalidate_locked()` helper because `_cache_get()` calls invalidate from within the lock - calling `_cache_invalidate()` would deadlock.

---

### 5. Atomic Index Writes (`cortical/got/indexer.py`)

**Problem:** `_save_indexes()` writes multiple JSON files directly - crash during write corrupts indexes.

**Solution:** Added `_atomic_write_json()` helper using temp file + `os.replace()`.

```python
def _atomic_write_json(self, filepath: Path, data: Any) -> None:
    """
    Write JSON atomically using temp file + os.replace().
    If a crash occurs during write, the original file remains intact.
    """
    temp_file = filepath.with_suffix(".json.tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename - either succeeds completely or fails completely
        os.replace(temp_file, filepath)
    except IOError as e:
        logger.error(f"Failed to save {filepath.name}: {e}")
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
```

---

## Patterns Reference

### Process-Level Locking (Cross-Session Safety)
Use `ProcessLock` from `cortical/utils/locking.py`:
```python
from cortical.utils.locking import ProcessLock

lock = ProcessLock(path / ".lockfile")
with lock:
    # Critical section - only one process at a time
```

### Thread-Level Locking (Same-Process Safety)
Use `threading.Lock`:
```python
import threading

self._lock = threading.Lock()
with self._lock:
    # Critical section - only one thread at a time
```

### Atomic File Writes (Crash Safety)
Use temp file + `os.replace()`:
```python
temp_file = filepath.with_suffix(".tmp")
with open(temp_file, "w") as f:
    f.write(data)
    f.flush()
    os.fsync(f.fileno())
os.replace(temp_file, filepath)  # Atomic on POSIX
```

---

## CLI Improvements Also Made

During handoff acceptance, discovered and fixed missing CLI commands:

1. **Added `handoff show` command** (`cortical/got/cli/handoff.py`)
   - Shows full handoff details including instructions and context

2. **Added `--limit/-n` flag** to all list commands:
   - `task list --limit 10`
   - `decision list --limit 5`
   - `handoff list --limit 5`
   - `sprint list --limit 5`
   - `epic list --limit 5`

3. **Added `decision show` command** (`cortical/got/cli/decision.py`)
   - Shows full decision details including rationale and alternatives

---

## Remaining Tasks

High priority tasks still pending:

| Task ID | Title | Priority |
|---------|-------|----------|
| T-20251226-112725-6e0cec19 | Integrate QueryIndexManager | high |
| T-20251226-112810-f4d8650c | Add concurrent access tests | high |
| T-20251226-114231-675e2be0 | Analyze GoT merge conflict root causes | high |

Medium priority:
| Task ID | Title | Priority |
|---------|-------|----------|
| T-20251226-112824-d50defff | Add performance tests for indexer | medium |
| T-20251226-114231-a640305e | Design semantic versioning for GoT | medium |
| T-20251223-003108-9872bb95 | Multi-branch layer inheritance | medium |

---

## Testing Notes

- All 325 GoT tests pass after changes
- All 343 ML collector tests pass
- Test `test_relaxed_mode_never_fsyncs_wal` required fix - ProcessLock was doing fsync on holder info

---

## Files Modified

| File | Changes |
|------|---------|
| `cortical/got/wal.py` | Added ProcessLock import, lock init, wrapped log() |
| `cortical/got/versioned_store.py` | Added ProcessLock import, lock init, wrapped _save_to_history() |
| `cortical/got/api.py` | Added threading import, lock init, wrapped all cache methods |
| `cortical/got/indexer.py` | Added _atomic_write_json() helper, refactored _save_indexes() |
| `scripts/ml_collector/persistence.py` | Wrapped save_commit_lite/save_session_lite with file_lock() |
| `cortical/utils/locking.py` | Removed unnecessary fsync from holder info write |
| `cortical/got/cli/handoff.py` | Added show command, --limit flag |
| `cortical/got/cli/task.py` | Added --limit flag |
| `cortical/got/cli/decision.py` | Added show command, --limit flag |
| `cortical/got/cli/sprint.py` | Added --limit flag to sprint and epic list |

---

## Verification Commands

```bash
# Run GoT tests
python -m pytest tests/unit/test_got*.py tests/integration/test_got*.py -q

# Run ML tests
python -m pytest tests/unit/test_ml*.py tests/integration/test_ml*.py -q

# Validate GoT state
python scripts/got_utils.py validate

# Check task status
python scripts/got_utils.py dashboard
```

---

## Session Metrics

- **Tasks Completed:** 5 (3 critical, 2 high)
- **Files Modified:** 10
- **Tests Passing:** 668+ (325 GoT + 343 ML)
- **Commits:** 4 (excluding earlier CLI fixes)
- **Branch:** claude/accept-handoff-T2HnB

---

*Generated: 2025-12-26*
