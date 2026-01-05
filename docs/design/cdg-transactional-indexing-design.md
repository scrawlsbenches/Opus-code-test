# CDG Transactional Indexing Design Document

*Author: Engineering Consultation Session*
*Date: 2026-01-05*
*Status: REVISED v2.0 (Post-Critique)*

---

## Executive Summary

This document describes the architectural changes required to achieve atomic index updates with entity operations in GoT. The core problem is that indexes are updated OUTSIDE transaction boundaries, creating a race condition where concurrent operations can observe inconsistent state.

**Critical Design Decision:** After rigorous critique from architecture, correctness, testability, and implementation experts, the approach has been revised:

> ~~Original: Add callback infrastructure to CDG~~
> **Revised: Solve at GoT layer, preserving CDG domain-agnosticism**

The revised approach moves index update coordination to `TransactionContext` in GoT, where it belongs architecturally. This eliminates the layering violation where foundation layer (CDG) would know about application-specific concerns (indexes).

---

## Problem Analysis

### Current Architecture (BROKEN)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT FLOW                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Thread A: Delete Task                    Thread B: Query Index          │
│  ─────────────────────────                ──────────────────────         │
│  1. tx.begin()                                                           │
│  2. tx_manager.delete(task_id)                                           │
│  3. tx_manager.commit()  ─────────────►  Task file deleted               │
│     └── Lock released                                                    │
│                                                                          │
│  ◄─── RACE WINDOW ───────────────────►   4. index.lookup("pending")     │
│                                             └── Index still has task_id  │
│  5. index_manager.remove_task()             └── try to load task file    │
│  6. index_manager.save()                       └── FileNotFoundError!    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Evidence from Code

**Location of Race Window (api.py:2126-2177):**
```python
# Line 2126: Transaction commits, lock released
result = self.tx_manager.commit(self.tx)

# Lines 2138-2140: Index updated AFTER commit, outside lock
if self._got_manager is not None and self._task_changes:
    self._apply_index_updates()

# Line 2177: Index persisted to disk
self._got_manager._index_manager.save()
```

### Secondary Issue: TOCTOU Race in Storage

Additionally, `storage.py:195-198` has a Time-Of-Check-Time-Of-Use race:
```python
if not path.exists():    # Check
    return None
# RACE WINDOW HERE
wrapper = self._read_and_verify(path)  # Use - can raise FileNotFoundError
```

This must also be addressed to fully eliminate the concurrent delete issue.

---

## Design Requirements

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Index updates MUST occur atomically within transaction boundaries |
| FR-2 | Index state MUST be consistent with entity state at all times |
| FR-3 | Index updates MUST be recoverable on crash |
| FR-4 | Solution MUST preserve CDG domain-agnosticism (no GoT knowledge in CDG) |
| FR-5 | Existing GoT API MUST remain unchanged (backward compatible) |

### Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | No external dependencies (Sovereignty Principle) |
| NFR-2 | Index lookup performance: O(1) maintained |
| NFR-3 | Transaction commit overhead: measurable, acceptable trade-off |
| NFR-4 | Must use existing Container/DI patterns |

---

## REVISED Solution: GoT-Layer Lock Coordination

### Architecture Overview

The revised approach solves the problem in `TransactionContext` (GoT layer) by explicitly coordinating with CDG's lock, rather than adding callback infrastructure to CDG.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REVISED FLOW (FIXED)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TransactionContext.__exit__():                                          │
│  ─────────────────────────────────                                       │
│  1. Build index updates from _task_changes (preparation, no I/O)         │
│  2. Acquire tx_manager.lock explicitly                                   │
│     ┌─ LOCK HELD ─────────────────────────────────────────────────┐     │
│     │ 3. tx_manager.commit() [commits entity changes]              │     │
│     │ 4. index_manager.apply_updates() [in-memory updates]         │     │
│     │ 5. index_manager.save() [persist to disk]                    │     │
│     └───────────────────────────────────────────────────────────────┘     │
│  6. Lock released                                                        │
│                                                                          │
│  Thread B: Query Index                                                   │
│  ──────────────────────                                                  │
│  Can only execute AFTER step 6 - index is already updated               │
│  No race window exists                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **CDG Remains Domain-Agnostic**: No changes to CDG. It doesn't know about indexes.
2. **Lock Coordination at GoT Layer**: `TransactionContext` manages the broader atomicity.
3. **Explicit Lock Acquisition**: Uses the reentrant lock that CDG already provides.
4. **Error Handling with Retry**: Index failures are retried before escalating.

---

## Detailed Implementation

### Change 1: TransactionContext Modification (api.py)

```python
class TransactionContext:
    """
    Context manager for transactional operations.

    REVISED: Index updates now occur within the transaction lock to ensure
    atomicity of entity changes + index updates.
    """

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.tx is None:
            return False

        if exc_type is not None:
            self.tx_manager.rollback(self.tx, reason="exception")
            return False

        if self.read_only:
            self.tx_manager.rollback(self.tx, reason="read_only")
            return False

        # REVISED: Coordinate index updates within transaction lock
        # Phase 1: Prepare index updates (no I/O, fast)
        index_updates = self._prepare_index_updates()

        # Phase 2: Execute within explicit lock scope
        with self.tx_manager.lock:
            # Commit entity changes (this re-acquires the same lock - it's reentrant)
            result = self.tx_manager.commit(self.tx)

            if not result.success:
                raise TransactionError(
                    f"Transaction commit failed: {result.reason}",
                    conflicts=result.conflicts
                )

            # Invalidate cache for all written entities
            if self._got_manager is not None and self.tx.write_set:
                written_ids = list(self.tx.write_set.keys())
                self._got_manager._cache_invalidate_many(written_ids)

            # CRITICAL: Apply index updates WITHIN the lock
            if self._got_manager is not None and index_updates:
                self._apply_index_updates_atomic(index_updates)

        # Lock released here - index is guaranteed consistent
        return False

    def _prepare_index_updates(self) -> Dict[str, Any]:
        """
        Prepare index updates without I/O (fast, outside lock).

        Returns a dictionary describing what needs to change in the index.
        """
        if not self._task_changes:
            return {}

        updates = {
            'deletes': [],
            'creates': [],
            'modifications': []
        }

        for task_id, changes in self._task_changes.items():
            if changes.get('is_delete'):
                updates['deletes'].append(task_id)
            elif changes.get('is_create'):
                task = self.tx.write_set.get(task_id)
                if task and isinstance(task, Task):
                    updates['creates'].append({
                        'id': task_id,
                        'status': task.status,
                        'priority': task.priority
                    })
            else:
                task = self.tx.write_set.get(task_id)
                if task and isinstance(task, Task):
                    updates['modifications'].append({
                        'id': task_id,
                        'old_status': changes.get('old_status'),
                        'new_status': task.status,
                        'old_priority': changes.get('old_priority'),
                        'new_priority': task.priority
                    })

        return updates

    def _apply_index_updates_atomic(self, updates: Dict[str, Any]) -> None:
        """
        Apply index updates with retry logic.

        MUST be called within transaction lock.
        Raises TransactionError if all retries fail.
        """
        if self._got_manager is None or self._got_manager._index_manager is None:
            return

        index_manager = self._got_manager._index_manager
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Apply in-memory updates
                for task_id in updates.get('deletes', []):
                    index_manager.remove_task(task_id)

                for task_data in updates.get('creates', []):
                    index_manager.index_task(
                        task_data['id'],
                        status=task_data['status'],
                        priority=task_data['priority']
                    )

                for mod in updates.get('modifications', []):
                    index_manager.update_task(
                        mod['id'],
                        old_status=mod['old_status'],
                        new_status=mod['new_status'],
                        old_priority=mod['old_priority'],
                        new_priority=mod['new_priority']
                    )

                # Persist to disk
                if not index_manager.save():
                    raise IOError("Index save failed")

                return  # Success

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Index update failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.01 * (2 ** attempt))  # Exponential backoff

        # All retries exhausted - this is a critical failure
        logger.error(
            f"Index update failed after {max_retries} retries: {last_error}. "
            "Marking index as corrupt for rebuild on next access."
        )
        # Mark index as needing rebuild rather than leaving inconsistent
        index_manager.mark_needs_rebuild()
        raise TransactionError(
            f"Index update failed after {max_retries} retries: {last_error}"
        )
```

### Change 2: Storage TOCTOU Fix (storage.py)

```python
def read(self, entity_id: str) -> Optional[Entity]:
    """
    Read an entity by ID.

    REVISED: Handle FileNotFoundError gracefully to eliminate TOCTOU race.
    """
    path = self._entity_path(entity_id)
    try:
        wrapper = self._read_and_verify(path)
        return self.entity_factory(wrapper["data"])
    except FileNotFoundError:
        # File was deleted between check and read - treat as not found
        # This is expected during concurrent delete + read operations
        return None
    except (json.JSONDecodeError, KeyError) as e:
        raise CorruptionError(f"Entity {entity_id} is corrupted: {e}")
```

### Change 3: Index Manager Enhancements (indexer.py)

```python
class QueryIndexManager:
    """Manages query indexes for fast lookups."""

    def __init__(self, got_dir: Path):
        # ... existing init ...
        self._needs_rebuild = False
        self._rebuild_lock = threading.Lock()

    def mark_needs_rebuild(self) -> None:
        """Mark index as needing rebuild due to consistency failure."""
        with self._rebuild_lock:
            self._needs_rebuild = True
            # Write flag to disk for crash recovery
            flag_file = self._index_dir / ".needs_rebuild"
            flag_file.touch()
            logger.warning("Index marked for rebuild")

    def check_and_rebuild_if_needed(self, task_loader: Callable[[], List[Task]]) -> None:
        """
        Check if rebuild is needed and perform it.

        Called on startup or before queries to ensure consistency.
        """
        flag_file = self._index_dir / ".needs_rebuild"

        with self._rebuild_lock:
            if self._needs_rebuild or flag_file.exists():
                logger.info("Rebuilding index from entities...")
                tasks = task_loader()
                self.rebuild_all(tasks)
                self._needs_rebuild = False
                if flag_file.exists():
                    flag_file.unlink()
                logger.info(f"Index rebuilt with {len(tasks)} tasks")
```

### Change 4: Crash Recovery Enhancement (api.py)

```python
class GoTManager:
    """Main GoT API facade."""

    def __init__(self, got_dir: Path, ...):
        # ... existing init ...

        # Check for index consistency after WAL recovery
        if self._index_manager:
            self._index_manager.check_and_rebuild_if_needed(
                task_loader=self._load_all_tasks
            )

    def _load_all_tasks(self) -> List[Task]:
        """Load all tasks from storage for index rebuild."""
        tasks = []
        for entity_id in self._task_store.list_ids():
            task = self._task_store.read(entity_id)
            if task:
                tasks.append(task)
        return tasks
```

---

## Rejected Alternative: CDG Callbacks

The original design proposed adding `PostCommitCallback` to CDG. This was rejected for:

1. **Layering Violation**: CDG (foundation) would know about GoT (application) concerns
2. **Error Handling Complexity**: Silent failures create worse inconsistency
3. **DI Pattern Mismatch**: Callback registration doesn't fit container model
4. **Over-Engineering**: The problem can be solved more simply at GoT layer

See Appendix A for the full critique that led to this rejection.

---

## Implementation Plan

### Phase 1: Storage TOCTOU Fix (Low Risk)

**Files:** `cortical/cdg/storage.py`

**Changes:** Catch `FileNotFoundError` in `read()`, return `None`

**Tests:**
- Unit test for concurrent read during delete
- Verify no FileNotFoundError propagates

### Phase 2: Index Manager Enhancements (Low Risk)

**Files:** `cortical/got/indexer.py`

**Changes:**
- Add `mark_needs_rebuild()` method
- Add `check_and_rebuild_if_needed()` method
- Add `.needs_rebuild` flag file

**Tests:**
- Unit test for rebuild flag persistence
- Unit test for rebuild from entities

### Phase 3: TransactionContext Modification (Medium Risk)

**Files:** `cortical/got/api.py`

**Changes:**
- Add `_prepare_index_updates()` method
- Modify `__exit__` to use explicit lock coordination
- Add `_apply_index_updates_atomic()` with retry logic

**Tests:**
- Behavioral test: `test_index_consistent_under_concurrent_deletes`
- Regression test: Verify `test_create_update_delete_in_parallel` passes
- Unit test: Verify lock is held during index update

### Phase 4: Crash Recovery (Low Risk)

**Files:** `cortical/got/api.py`

**Changes:** Add `check_and_rebuild_if_needed()` call in GoTManager.__init__

**Tests:**
- Integration test: Crash simulation with index rebuild

---

## Testing Strategy (REVISED)

### Critical Tests

#### Test 1: Lock Held During Index Update

```python
def test_lock_held_during_index_update(fresh_got_manager):
    """Verify index updates occur within transaction lock."""
    import threading

    callback_executed = threading.Event()
    lock_held = []

    # Monkey-patch index_manager.save to check lock state
    original_save = fresh_got_manager._index_manager.save
    def instrumented_save():
        callback_executed.set()
        lock_held.append(fresh_got_manager._tx_manager.lock.locked())
        time.sleep(0.05)  # Hold briefly for external verification
        return original_save()

    fresh_got_manager._index_manager.save = instrumented_save

    # External thread tries to acquire lock during save
    lock_acquired = []
    def try_acquire():
        callback_executed.wait(timeout=1.0)
        acquired = fresh_got_manager._tx_manager.lock.acquire(timeout=0.01)
        lock_acquired.append(acquired)
        if acquired:
            fresh_got_manager._tx_manager.lock.release()

    thread = threading.Thread(target=try_acquire)
    thread.start()

    # Create task (triggers index save)
    fresh_got_manager.create_task("Test Task")

    thread.join()

    assert lock_held[0] is True, "Lock must be held during index save"
    assert lock_acquired[0] is False, "External thread should not acquire lock"
```

#### Test 2: Concurrent Delete Race Condition (Regression)

```python
@pytest.mark.parametrize("concurrency", [2, 4, 8])
def test_no_file_not_found_on_concurrent_delete_and_query(fresh_got_manager, concurrency):
    """
    Regression test for race condition that caused FileNotFoundError.

    Before fix: Index lookup returns deleted task ID → FileNotFoundError
    After fix: Index updated atomically with delete → no stale references
    """
    import threading

    # Pre-create tasks
    tasks_per_thread = 5
    all_tasks = []
    for i in range(concurrency * tasks_per_thread):
        task = fresh_got_manager.create_task(f"Task-{i}", status="pending")
        all_tasks.append(task)

    # Distribute tasks
    batches = [
        all_tasks[i * tasks_per_thread:(i + 1) * tasks_per_thread]
        for i in range(concurrency)
    ]

    errors = []
    errors_lock = threading.Lock()
    barrier = threading.Barrier(concurrency)

    def delete_and_query(batch):
        try:
            barrier.wait()  # Synchronize start
            for task in batch:
                fresh_got_manager.delete_task(task.id, force=True)
                # Query immediately - should NOT return deleted task
                results = fresh_got_manager.query_api.find_tasks(status="pending")
                if task.id in [r.id for r in results]:
                    with errors_lock:
                        errors.append(f"Deleted task {task.id} still in index")
        except FileNotFoundError as e:
            with errors_lock:
                errors.append(f"FileNotFoundError (THE BUG): {e}")
        except Exception as e:
            with errors_lock:
                errors.append(f"Unexpected: {e}")

    threads = [
        threading.Thread(target=delete_and_query, args=(batch,))
        for batch in batches
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Race condition detected: {errors}"
```

#### Test 3: Index Rebuild After Failure

```python
def test_index_rebuilt_after_save_failure(fresh_got_manager, monkeypatch):
    """Index is marked for rebuild if save fails, and rebuilt on next access."""
    # Create task
    task = fresh_got_manager.create_task("Test", status="pending")

    # Force save failure
    def failing_save():
        raise IOError("Disk full")

    monkeypatch.setattr(
        fresh_got_manager._index_manager,
        'save',
        failing_save
    )

    # Try to create another task - should fail but mark rebuild
    with pytest.raises(TransactionError):
        fresh_got_manager.create_task("Test2", status="pending")

    # Verify rebuild flag set
    flag_file = fresh_got_manager._index_manager._index_dir / ".needs_rebuild"
    assert flag_file.exists()

    # Restore save and query
    monkeypatch.undo()
    fresh_got_manager._index_manager.check_and_rebuild_if_needed(
        task_loader=fresh_got_manager._load_all_tasks
    )

    # Verify index is consistent
    results = fresh_got_manager.query_api.find_tasks(status="pending")
    assert task.id in [r.id for r in results]
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Lock hold time increases | High | Medium | Accept trade-off; measure impact |
| Retry logic causes delay | Low | Low | Cap at 3 retries with backoff |
| Index rebuild on startup | Low | Medium | Only happens after crash+failure |
| Reentrant lock edge cases | Low | High | Lock is already reentrant; tested |

---

## Performance Considerations

The revised design increases lock hold time by including index I/O within the lock. Expected impact:

| Operation | Before | After | Delta |
|-----------|--------|-------|-------|
| Commit latency | ~5ms | ~15-25ms | +10-20ms |
| Throughput | ~100 tx/s | ~40-60 tx/s | -40-60% |

**Recommendation:** Accept this trade-off for correctness. Throughput can be improved later by:
1. Batching index writes (don't fsync on every commit)
2. Background index persistence (mark dirty, sync periodically)
3. More efficient serialization (msgpack vs JSON)

---

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Should callbacks be in CDG? | NO - Solved at GoT layer instead |
| Should callback failures abort? | YES - With retry and rebuild fallback |
| Should indexes be WAL-logged? | NO - Rebuild from entities on crash |
| What about TOCTOU in storage? | FIXED - Catch FileNotFoundError |

---

## Appendix A: Critique Summary

The original design received critique from four perspectives:

**Architecture Critique:**
- ❌ Layering violation: CDG would know about GoT concerns
- ❌ DI pattern mismatch: Callbacks don't fit container model
- ✓ Recommendation: Solve at GoT layer, preserve CDG domain-agnosticism

**Correctness Critique:**
- ❌ Callback failure handling creates worse inconsistency
- ❌ Crash recovery undefined
- ❌ TOCTOU race in storage.py not addressed
- ✓ Recommendation: Retry + rebuild, fix storage race

**Testability Critique:**
- ❌ Lock-within-callback test flawed
- ❌ Concurrent test has logic errors
- ❌ Missing failure mode tests
- ✓ Recommendation: Use thread coordination for lock tests

**Implementation Critique:**
- ❌ Type inconsistencies (set[str] vs Set[str])
- ❌ Performance impact not measured
- ✓ Recommendation: Accept performance trade-off for correctness

---

## Appendix B: Lock Hierarchy Documentation

**IMPORTANT:** Add to CLAUDE.md under "Critical Bugs":

```markdown
### Lock Ordering Invariant

When working with concurrent access to GoT:

1. `CDGTransactionManager.lock` (ProcessLock, reentrant)
2. `QueryIndexManager._write_lock` (threading.Lock)

**Invariant:** If acquiring both locks, ALWAYS acquire transaction lock first.

Violation causes deadlock. The revised TransactionContext design enforces
this by performing index updates within the transaction lock scope.
```

---

*Document Version: 2.0 REVISED*
*Based on critique from: Architecture, Correctness, Testability, Implementation experts*
*Status: Ready for Implementation Review*
