# Tiered Locking: Practical Implementation Guide

This guide provides step-by-step implementation of tiered locking patterns for the Cortical GoT system.

---

## Phase 1: Hierarchical Lock Manager

### Step 1.1: Define Lock Hierarchy

```python
# cortical/got/lock_hierarchy.py

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
from pathlib import Path


class LockLevel(Enum):
    """Lock hierarchy levels in strict order."""
    DATABASE = 0      # Must acquire first
    ENTITY_TYPE = 1   # Must acquire second (if acquiring beyond DB)
    RECORD_ID = 2     # Must acquire last


@dataclass
class LockRequest:
    """Request to acquire locks at specific hierarchy levels."""

    levels: List[str]  # ['database'] or ['entity_type', 'tasks'] or ['record_id', 'task_123']
    timeout_ms: int = 5000
    read_only: bool = False

    def validate(self) -> bool:
        """Ensure locks are requested in correct order."""
        valid_orders = [
            ['database'],
            ['entity_type'],
            ['record_id'],
            ['database', 'entity_type'],
            ['database', 'record_id'],
            ['entity_type', 'record_id'],
            ['database', 'entity_type', 'record_id'],
        ]
        return self.levels in valid_orders


class HierarchicalLockManager:
    """
    Manages locks at multiple granularity levels.

    Guarantees deadlock-free operation through strict lock ordering:
    DATABASE → ENTITY_TYPE → RECORD_ID

    Example usage:

        mgr = HierarchicalLockManager(Path(".got"))

        # Lock single task row
        with mgr.acquire(['record_id', 'task_abc'], timeout_ms=5000):
            task = store.get('task_abc')
            task.status = 'completed'
            store.put('task_abc', task)
    """

    def __init__(self, got_dir: Path):
        self.got_dir = got_dir
        self.got_dir.mkdir(parents=True, exist_ok=True)

        # Database-level lock (protects entire .got directory)
        from cortical.utils.locking import ProcessLock
        self._db_lock = ProcessLock(self.got_dir / ".got_db.lock", reentrant=True)

        # Entity-type level locks (one per entity type)
        self._entity_locks: Dict[str, ProcessLock] = {}

        # Record-level locks (lazily created)
        self._record_locks: Dict[str, ProcessLock] = {}

        # Thread-local lock stack for deadlock detection
        import threading
        self._local = threading.local()

    def _get_lock_stack(self) -> List[str]:
        """Get this thread's lock acquisition stack."""
        if not hasattr(self._local, 'lock_stack'):
            self._local.lock_stack = []
        return self._local.lock_stack

    def acquire(
        self,
        levels: List[str],
        timeout_ms: int = 5000,
        read_only: bool = False
    ) -> '_LockContext':
        """
        Acquire locks at specified hierarchy levels.

        Args:
            levels: Lock hierarchy path, e.g.:
                ['database']                    # Database lock
                ['entity_type', 'tasks']        # Table lock
                ['record_id', 'task_123']       # Row lock
            timeout_ms: Timeout in milliseconds
            read_only: If True, use shared locks (not yet implemented)

        Returns:
            Context manager that releases locks on exit

        Raises:
            ValueError: If lock order is invalid
            DeadlockDetected: If circular wait detected
            LockTimeout: If lock not acquired within timeout
        """
        request = LockRequest(levels=levels, timeout_ms=timeout_ms, read_only=read_only)

        if not request.validate():
            raise ValueError(f"Invalid lock order: {levels}")

        # Check for circular wait (deadlock detection)
        self._check_deadlock(levels)

        return _LockContext(self, levels, timeout_ms)

    def _check_deadlock(self, requested_levels: List[str]) -> None:
        """
        Detect circular wait (deadlock).

        Deadlock = lock ordering violation.
        Example:
            Thread A: holds entity_type:tasks, wants record_id:task_123
            Thread B: holds record_id:task_456, wants entity_type:tasks
            → Circular wait, deadlock!

        This is prevented by GLOBAL ORDERING.
        """
        stack = self._get_lock_stack()

        # All locks in stack must come BEFORE requested locks in global order
        lock_order = ['database', 'entity_type', 'record_id']

        if stack:
            last_level = stack[-1]
            requested_level = requested_levels[0]

            last_idx = lock_order.index(last_level)
            requested_idx = lock_order.index(requested_level)

            if requested_idx <= last_idx:
                raise DeadlockDetected(
                    f"Lock ordering violation: "
                    f"Have {last_level}, requesting {requested_level}"
                )

    def _acquire_db_lock(self, timeout_ms: float) -> None:
        """Acquire database-level lock."""
        if not self._db_lock.acquire(timeout=timeout_ms / 1000.0):
            raise LockTimeout("Database lock acquisition timed out")

    def _acquire_entity_lock(self, entity_type: str, timeout_ms: float) -> None:
        """Acquire entity-type level lock."""
        if entity_type not in self._entity_locks:
            self._entity_locks[entity_type] = ProcessLock(
                self.got_dir / f".lock_entity_{entity_type}"
            )

        lock = self._entity_locks[entity_type]
        if not lock.acquire(timeout=timeout_ms / 1000.0):
            raise LockTimeout(f"Entity lock acquisition timed out for {entity_type}")

    def _acquire_record_lock(self, record_id: str, timeout_ms: float) -> ProcessLock:
        """Acquire record-level lock."""
        if record_id not in self._record_locks:
            self._record_locks[record_id] = ProcessLock(
                self.got_dir / f".lock_record_{record_id}"
            )

        lock = self._record_locks[record_id]
        if not lock.acquire(timeout=timeout_ms / 1000.0):
            raise LockTimeout(f"Record lock acquisition timed out for {record_id}")

        return lock

    def _release_locks(self, acquired_locks: List) -> None:
        """Release locks in reverse order (LIFO)."""
        for lock in reversed(acquired_locks):
            if lock:
                lock.release()


class _LockContext:
    """Context manager for holding locks."""

    def __init__(
        self,
        manager: HierarchicalLockManager,
        levels: List[str],
        timeout_ms: int
    ):
        self.manager = manager
        self.levels = levels
        self.timeout_ms = timeout_ms
        self.acquired_locks: List = []

    def __enter__(self) -> '_LockContext':
        """Acquire locks in hierarchy order."""
        stack = self.manager._get_lock_stack()
        stack.append(self.levels[0])  # Track for deadlock detection

        try:
            # Acquire locks in order: database → entity_type → record_id
            if 'database' in self.levels:
                self.manager._acquire_db_lock(self.timeout_ms)
                self.acquired_locks.append(self.manager._db_lock)

            if 'entity_type' in self.levels:
                entity_type = self.levels[self.levels.index('entity_type') + 1]
                self.manager._acquire_entity_lock(entity_type, self.timeout_ms)
                self.acquired_locks.append(self.manager._entity_locks[entity_type])

            if 'record_id' in self.levels:
                record_id = self.levels[self.levels.index('record_id') + 1]
                lock = self.manager._acquire_record_lock(record_id, self.timeout_ms)
                self.acquired_locks.append(lock)

            return self
        except Exception:
            # Release what we've acquired so far
            self.manager._release_locks(self.acquired_locks)
            stack.pop()
            raise

    def __exit__(self, *args) -> None:
        """Release locks in reverse order."""
        stack = self.manager._get_lock_stack()
        if stack:
            stack.pop()

        self.manager._release_locks(self.acquired_locks)


# Custom exceptions

class DeadlockDetected(Exception):
    """Deadlock detected via lock ordering violation."""
    pass


class LockTimeout(Exception):
    """Lock acquisition timed out."""
    pass
```

### Step 1.2: Integrate with Transaction Manager

```python
# cortical/got/tx_manager_v2.py (modified)

from pathlib import Path
from .lock_hierarchy import HierarchicalLockManager


class TransactionManager:
    """
    Enhanced TransactionManager with hierarchical locking.

    Changes from v1:
    - Replaces database-level ProcessLock with HierarchicalLockManager
    - Supports row-level locking for independent concurrent operations
    - Maintains deadlock-free guarantees via lock ordering
    """

    def __init__(self, got_dir: Path, durability="balanced"):
        self.got_dir = Path(got_dir)

        # Replace old: self.lock = ProcessLock(...)
        # With new: hierarchical manager
        self.lock_mgr = HierarchicalLockManager(self.got_dir)

        # Rest of initialization unchanged
        from .versioned_store import VersionedStore
        self.store = VersionedStore(self.got_dir / "entities")
        self.durability = durability

    def get(self, entity_id: str) -> 'Entity':
        """
        Get entity with automatic lock based on entity type.

        For reads, acquire minimal lock (record-level only).
        """
        # Parse entity_id to get type: "task_123" → "task"
        entity_type = entity_id.split('_')[0] if '_' in entity_id else 'unknown'

        # Row-level lock for reads (allows concurrent reads)
        with self.lock_mgr.acquire(['record_id', entity_id], read_only=True):
            return self.store.get(entity_id)

    def put(self, entity_id: str, entity: 'Entity') -> None:
        """
        Put entity with write lock.

        Acquires row-level lock to prevent concurrent modifications.
        """
        entity_type = entity_id.split('_')[0] if '_' in entity_id else 'unknown'

        # Row-level lock for writes
        with self.lock_mgr.acquire(['record_id', entity_id], timeout_ms=5000):
            self.store.put(entity_id, entity)
            self._log_write(entity_id, entity)

    def query(self, entity_type: str, **filters) -> List['Entity']:
        """
        Query entities by type with table-level lock.

        Uses entity-type lock since this is a bulk operation.
        """
        # Table-level lock for bulk operations
        with self.lock_mgr.acquire(
            ['entity_type', entity_type],
            timeout_ms=10000  # Bulk ops may take longer
        ):
            return self.store.query(entity_type, **filters)

    def batch_update(self, updates: Dict[str, 'Entity']) -> None:
        """
        Update multiple entities atomically.

        Acquires row locks in sorted order to prevent deadlock.
        """
        # Sort entity IDs to ensure consistent lock ordering
        sorted_ids = sorted(updates.keys())

        # Acquire row locks in order (prevents deadlock)
        contexts = []
        try:
            for entity_id in sorted_ids:
                ctx = self.lock_mgr.acquire(['record_id', entity_id])
                ctx.__enter__()
                contexts.append(ctx)

            # All locks held, perform updates
            for entity_id, entity in updates.items():
                self.store.put(entity_id, entity)

        finally:
            # Release in reverse order
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)
```

### Step 1.3: Testing Hierarchical Locks

```python
# tests/integration/test_hierarchical_locks.py

import unittest
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from cortical.got.lock_hierarchy import HierarchicalLockManager, DeadlockDetected


class TestHierarchicalLocks(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.got_dir = Path(self.temp_dir.name)
        self.mgr = HierarchicalLockManager(self.got_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_lock_acquisition(self):
        """Verify single lock can be acquired and released."""
        with self.mgr.acquire(['database']):
            pass  # Lock held during context
        # Lock released here

    def test_hierarchical_lock_ordering(self):
        """Verify locks must be acquired in strict order."""
        # Valid: database → entity_type
        with self.mgr.acquire(['database']):
            with self.mgr.acquire(['entity_type', 'tasks']):
                pass

        # Invalid: entity_type → database (would deadlock)
        with self.mgr.acquire(['entity_type', 'tasks']):
            with self.assertRaises(DeadlockDetected):
                with self.mgr.acquire(['database']):
                    pass

    def test_concurrent_row_locks(self):
        """Verify multiple threads can hold different row locks simultaneously."""
        results = []

        def update_task(task_id: str):
            with self.mgr.acquire(['record_id', task_id], timeout_ms=5000):
                time.sleep(0.1)  # Simulate work
                results.append(task_id)

        # Update 10 different tasks concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(update_task, f'task_{i}')
                for i in range(10)
            ]
            for future in futures:
                future.result()

        # All tasks should complete (concurrent, not blocked)
        self.assertEqual(len(results), 10)

    def test_deadlock_prevention(self):
        """Verify circular wait is prevented."""
        # Thread A: database → entity_type
        # Thread B: entity_type → database (would deadlock, but prevented)

        def thread_a():
            with self.mgr.acquire(['database']):
                time.sleep(0.1)
                with self.mgr.acquire(['entity_type', 'tasks']):
                    return True

        def thread_b():
            with self.mgr.acquire(['entity_type', 'tasks']):
                time.sleep(0.1)
                # This will be prevented by deadlock detection
                with self.mgr.acquire(['database']):
                    return True

        # Only thread_a completes; thread_b gets DeadlockDetected
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(thread_a)
            future_b = executor.submit(thread_b)

            # Thread A should succeed
            self.assertTrue(future_a.result())

            # Thread B should raise DeadlockDetected
            with self.assertRaises(DeadlockDetected):
                future_b.result()


if __name__ == '__main__':
    unittest.main()
```

---

## Phase 2: Tier Classification

### Step 2.1: Define Tiers

```python
# cortical/got/tiers.py

from enum import Enum
from dataclasses import dataclass
from typing import Set, Tuple, Optional


class ConsistencyTier(Enum):
    """Data classification by consistency requirements."""

    CRITICAL = "critical"          # ACID: pessimistic, fsync
    IMPORTANT = "important"        # Strong eventual: optimistic
    BEST_EFFORT = "best_effort"    # Eventual: lock-free


@dataclass
class TierConfig:
    """Configuration for each consistency tier."""

    name: str
    fsync: bool                     # Sync to disk immediately?
    lock_strategy: str              # 'pessimistic', 'optimistic', 'none'
    isolation_level: str            # 'serializable', 'snapshot', 'eventual'
    retry_limit: int                # Max retries on conflict
    timeout_ms: int                 # Lock timeout


TIER_CONFIGS = {
    ConsistencyTier.CRITICAL: TierConfig(
        name="Critical (ACID)",
        fsync=True,
        lock_strategy='pessimistic',
        isolation_level='serializable',
        retry_limit=3,
        timeout_ms=5000,
    ),
    ConsistencyTier.IMPORTANT: TierConfig(
        name="Important (Strong Eventual)",
        fsync=False,
        lock_strategy='optimistic',
        isolation_level='snapshot',
        retry_limit=10,
        timeout_ms=1000,
    ),
    ConsistencyTier.BEST_EFFORT: TierConfig(
        name="Best-Effort (Eventual)",
        fsync=False,
        lock_strategy='none',
        isolation_level='eventual',
        retry_limit=0,
        timeout_ms=10,
    ),
}


class TierManager:
    """
    Manages data classification into consistency tiers.

    Classification rules:
        CRITICAL: task.{id,status}, decision.{id,content}, edge.edge_type
        IMPORTANT: task.{priority,tags}, edge.weight, indices
        BEST_EFFORT: ml_metrics.*, observability.*, debug.*
    """

    def __init__(self):
        # (entity_type, field_name) → ConsistencyTier
        self._tier_mapping: Dict[Tuple[str, str], ConsistencyTier] = {
            # CRITICAL TIER: ACID required
            ('task', 'id'): ConsistencyTier.CRITICAL,
            ('task', 'status'): ConsistencyTier.CRITICAL,
            ('task', 'created_at'): ConsistencyTier.CRITICAL,

            ('decision', 'id'): ConsistencyTier.CRITICAL,
            ('decision', 'content'): ConsistencyTier.CRITICAL,

            ('edge', 'id'): ConsistencyTier.CRITICAL,
            ('edge', 'edge_type'): ConsistencyTier.CRITICAL,
            ('edge', 'from_id'): ConsistencyTier.CRITICAL,
            ('edge', 'to_id'): ConsistencyTier.CRITICAL,

            ('sprint', 'id'): ConsistencyTier.CRITICAL,
            ('epic', 'id'): ConsistencyTier.CRITICAL,

            # IMPORTANT TIER: Strong eventual consistency
            ('task', 'priority'): ConsistencyTier.IMPORTANT,
            ('task', 'tags'): ConsistencyTier.IMPORTANT,
            ('task', 'dependencies'): ConsistencyTier.IMPORTANT,

            ('edge', 'weight'): ConsistencyTier.IMPORTANT,
            ('edge', 'confidence'): ConsistencyTier.IMPORTANT,

            # BEST_EFFORT TIER: Lock-free
            ('metrics', 'prediction_count'): ConsistencyTier.BEST_EFFORT,
            ('metrics', 'cache_hits'): ConsistencyTier.BEST_EFFORT,
            ('observability', 'timing_histogram'): ConsistencyTier.BEST_EFFORT,
            ('debug', 'trace_frames'): ConsistencyTier.BEST_EFFORT,
        }

    def classify(
        self,
        entity_type: str,
        field_name: str
    ) -> ConsistencyTier:
        """
        Classify a field into a consistency tier.

        Args:
            entity_type: Type of entity (task, decision, edge, etc.)
            field_name: Name of field

        Returns:
            ConsistencyTier for this field
        """
        # Exact match
        key = (entity_type, field_name)
        if key in self._tier_mapping:
            return self._tier_mapping[key]

        # Wildcard match (e.g., "metrics.*" for any metrics field)
        if entity_type in ('metrics', 'ml_metrics'):
            return ConsistencyTier.BEST_EFFORT

        if entity_type == 'observability' or field_name.startswith('ml_'):
            return ConsistencyTier.BEST_EFFORT

        if field_name.startswith('debug_'):
            return ConsistencyTier.BEST_EFFORT

        # Default to IMPORTANT for unknown fields
        return ConsistencyTier.IMPORTANT

    def get_config(self, tier: ConsistencyTier) -> TierConfig:
        """Get configuration for a tier."""
        return TIER_CONFIGS[tier]

    def should_fsync(self, tier: ConsistencyTier) -> bool:
        """Should writes to this tier be synced to disk immediately?"""
        return self.get_config(tier).fsync

    def get_lock_strategy(self, tier: ConsistencyTier) -> Optional[str]:
        """Get locking strategy for tier."""
        return self.get_config(tier).lock_strategy

    def get_retry_limit(self, tier: ConsistencyTier) -> int:
        """Get max retries for tier."""
        return self.get_config(tier).retry_limit
```

### Step 2.2: Tier-Aware Transaction

```python
# cortical/got/tiered_transaction.py

from .tiers import TierManager, ConsistencyTier
from .lock_hierarchy import HierarchicalLockManager


class TieredTransaction:
    """
    Transaction with tier-aware consistency guarantees.

    Uses appropriate locking strategy based on data criticality:
    - CRITICAL: pessimistic locks + fsync
    - IMPORTANT: optimistic locking + async fsync
    - BEST_EFFORT: lock-free append
    """

    def __init__(
        self,
        tx_id: str,
        tx_manager,
        tier_manager: TierManager,
        lock_mgr: HierarchicalLockManager
    ):
        self.tx_id = tx_id
        self.tx_manager = tx_manager
        self.tier_manager = tier_manager
        self.lock_mgr = lock_mgr

        self.writes: Dict[str, Dict[str, any]] = {}  # entity_id → field updates
        self.critical_locks: Dict[str, any] = {}     # Pessimistic locks held
        self.important_versions: Dict[str, int] = {} # Version tracking for optimistic

    def write(
        self,
        entity_id: str,
        entity_type: str,
        field_name: str,
        value: any
    ) -> None:
        """
        Write field with tier-appropriate strategy.

        Selects locking approach based on field criticality.
        """
        tier = self.tier_manager.classify(entity_type, field_name)
        config = self.tier_manager.get_config(tier)

        if tier == ConsistencyTier.CRITICAL:
            # CRITICAL: Pessimistic lock + immediate write
            self._write_critical(entity_id, field_name, value, config)

        elif tier == ConsistencyTier.IMPORTANT:
            # IMPORTANT: Optimistic lock + version tracking
            self._write_important(entity_id, field_name, value, config)

        else:  # BEST_EFFORT
            # BEST_EFFORT: Lock-free append
            self._write_best_effort(entity_id, field_name, value)

    def _write_critical(
        self,
        entity_id: str,
        field_name: str,
        value: any,
        config
    ) -> None:
        """
        Write CRITICAL field with pessimistic locking.

        1. Acquire row lock (pessimistic)
        2. Write immediately
        3. Hold lock until commit
        """
        if entity_id not in self.critical_locks:
            # Acquire lock if not already held
            lock_ctx = self.lock_mgr.acquire(
                ['record_id', entity_id],
                timeout_ms=config.timeout_ms
            )
            lock_ctx.__enter__()
            self.critical_locks[entity_id] = lock_ctx

        # Buffer write
        if entity_id not in self.writes:
            self.writes[entity_id] = {'tier': 'critical', 'updates': {}}

        self.writes[entity_id]['updates'][field_name] = value

    def _write_important(
        self,
        entity_id: str,
        field_name: str,
        value: any,
        config
    ) -> None:
        """
        Write IMPORTANT field with optimistic locking.

        1. Read current version
        2. Buffer write
        3. Check version at commit
        """
        # Track version for conflict detection
        if entity_id not in self.important_versions:
            entity = self.tx_manager.get(entity_id)
            self.important_versions[entity_id] = entity.version

        # Buffer write
        if entity_id not in self.writes:
            self.writes[entity_id] = {'tier': 'important', 'updates': {}}

        self.writes[entity_id]['updates'][field_name] = value

    def _write_best_effort(
        self,
        entity_id: str,
        field_name: str,
        value: any
    ) -> None:
        """
        Write BEST_EFFORT field with lock-free append.

        Writes directly to metrics log (no buffering).
        """
        self.tx_manager.append_to_log(
            log_type='metrics',
            entity_id=entity_id,
            field=field_name,
            value=value,
            timestamp=datetime.now().isoformat()
        )

    def commit(self) -> bool:
        """
        Commit transaction with tier-aware semantics.

        Returns:
            True if committed, False if conflict (optimistic tier)
        """
        try:
            # Commit CRITICAL writes (already have locks)
            for entity_id, write_info in self.writes.items():
                if write_info['tier'] == 'critical':
                    entity = self.tx_manager.get(entity_id)
                    for field, value in write_info['updates'].items():
                        setattr(entity, field, value)
                    self.tx_manager.put(entity_id, entity, fsync=True)

            # Commit IMPORTANT writes (check versions)
            for entity_id, write_info in self.writes.items():
                if write_info['tier'] == 'important':
                    entity = self.tx_manager.get(entity_id)

                    # Check for conflict
                    if entity.version != self.important_versions.get(entity_id):
                        raise VersionConflictError(f"Conflict on {entity_id}")

                    for field, value in write_info['updates'].items():
                        setattr(entity, field, value)
                    self.tx_manager.put(entity_id, entity, fsync=False)

            # BEST_EFFORT writes already committed to log
            # (no action needed here)

            return True

        except VersionConflictError:
            # Rollback transaction
            self.abort()
            return False

        finally:
            # Release critical locks
            for lock_ctx in self.critical_locks.values():
                lock_ctx.__exit__(None, None, None)

    def abort(self) -> None:
        """Abort transaction and release locks."""
        self.writes.clear()
        for lock_ctx in self.critical_locks.values():
            lock_ctx.__exit__(None, None, None)
        self.critical_locks.clear()


class VersionConflictError(Exception):
    """Version conflict detected in optimistic locking."""
    pass
```

---

## Phase 3: Usage Examples

### Example 1: Update Task Status (CRITICAL)

```python
# This update uses pessimistic locking (CRITICAL tier)

with manager.transaction() as tx:
    task = tx.read('task_123')

    # Validating state machine (important for correctness)
    if task.status not in ('pending', 'in_progress'):
        raise StateError("Invalid transition")

    # Status change is CRITICAL - uses pessimistic lock
    tx.write(
        entity_id='task_123',
        entity_type='task',
        field_name='status',
        value='completed'
    )

    # Pessimistic lock acquired here, held until commit
    # If conflict: Lock held, prevents conflict
    # Commit: Releases lock

# Latency: ~50ms (lock: 10ms, write: 40ms)
```

### Example 2: Update Task Priority (IMPORTANT)

```python
# This update uses optimistic locking (IMPORTANT tier)

with manager.transaction() as tx:
    task = tx.read('task_123')

    # Priority change is IMPORTANT - uses optimistic locking
    tx.write(
        entity_id='task_123',
        entity_type='task',
        field_name='priority',
        value='high'
    )

    # No lock acquired (optimistic)
    # Version tracked for conflict detection
    # Commit checks: is version still the same?

# Latency: ~10ms (no lock, direct write)
# Conflict recovery: Retry (up to 10 times)
```

### Example 3: Log ML Metric (BEST_EFFORT)

```python
# This update uses lock-free append (BEST_EFFORT tier)

with manager.transaction() as tx:
    # Metric logging is BEST_EFFORT - lock-free
    tx.write(
        entity_id='session_abc123',
        entity_type='ml_metrics',
        field_name='prediction_count',
        value=42
    )

    # No lock acquired
    # Write already committed to metrics log
    # Loss acceptable (lossy)

# Latency: <1ms (no lock, append-only)
# No retries, no conflicts, no recovery
```

---

## Phase 4: Testing Tier Behavior

```python
# tests/integration/test_tiered_transactions.py

class TestTieredTransactions(unittest.TestCase):

    def test_critical_tier_uses_pessimistic_locks(self):
        """Verify CRITICAL fields use pessimistic locking."""
        # Track lock acquisitions
        lock_acquisitions = []

        original_acquire = self.lock_mgr.acquire
        def track_acquire(levels, **kwargs):
            lock_acquisitions.append(('pessimistic', levels))
            return original_acquire(levels, **kwargs)

        self.lock_mgr.acquire = track_acquire

        with manager.transaction() as tx:
            tx.write('task_123', 'task', 'status', 'completed')  # CRITICAL
            tx.commit()

        # Should see lock acquisition for record_id:task_123
        self.assertTrue(any('task_123' in str(x) for x in lock_acquisitions))

    def test_important_tier_no_locks_on_write(self):
        """Verify IMPORTANT fields don't lock on write."""
        lock_acquisitions = []

        original_acquire = self.lock_mgr.acquire
        def track_acquire(levels, **kwargs):
            lock_acquisitions.append(levels)
            return original_acquire(levels, **kwargs)

        self.lock_mgr.acquire = track_acquire

        with manager.transaction() as tx:
            tx.write('task_123', 'task', 'priority', 'high')  # IMPORTANT
            tx.commit()

        # Should NOT see lock acquisition (optimistic)
        self.assertEqual(len(lock_acquisitions), 0)

    def test_best_effort_concurrent_writes(self):
        """Verify BEST_EFFORT tier allows unlimited concurrent writes."""
        import threading

        results = []
        errors = []

        def log_metrics(session_id: str):
            try:
                with manager.transaction() as tx:
                    tx.write(
                        f'session_{session_id}',
                        'ml_metrics',
                        'prediction_count',
                        42
                    )
                results.append(session_id)
            except Exception as e:
                errors.append((session_id, str(e)))

        # 100 concurrent metric logs
        threads = [
            threading.Thread(target=log_metrics, args=(f"{i:03d}",))
            for i in range(100)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed without blocking
        self.assertEqual(len(results), 100)
        self.assertEqual(len(errors), 0)
```

---

## Summary: Implementation Checklist

- [ ] **Phase 1**: Implement `HierarchicalLockManager` (1 week)
  - [ ] Define lock levels and hierarchy
  - [ ] Implement deadlock detection
  - [ ] Add tests for lock ordering
  - [ ] Integrate with `TransactionManager`

- [ ] **Phase 2**: Implement tier classification (1 week)
  - [ ] Define `ConsistencyTier` enum
  - [ ] Create `TierManager` with classification rules
  - [ ] Implement `TieredTransaction` class
  - [ ] Add tests for tier behavior

- [ ] **Phase 3**: Update GoT operations (1 week)
  - [ ] Update task create/update methods
  - [ ] Update decision operations
  - [ ] Update edge operations
  - [ ] Update query operations

- [ ] **Phase 4**: Performance validation (1 week)
  - [ ] Benchmark single operations
  - [ ] Benchmark concurrent operations
  - [ ] Profile lock contention
  - [ ] Load test (100 concurrent agents)

Total: 4-5 weeks for complete implementation.

Expected improvements:
- Single operation: 500ms → 50ms (10x)
- 10 concurrent operations: 5s → 500ms (10x)
- Lock contention: 80% → 10% (8x reduction)
- Transaction abort rate: 5% → <1%

