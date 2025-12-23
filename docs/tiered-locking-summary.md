# Tiered Locking: Complete Summary

This document provides a quick reference for understanding and implementing tiered locking in the Cortical GoT system.

---

## The Problem

Your system has two conflicting requirements:

| Requirement | GoT (Tasks/Decisions) | ML Data (Metrics) |
|---|---|---|
| **Consistency** | ACID (must never lose) | Eventual (lossy acceptable) |
| **Latency** | ~100ms acceptable | <1ms required |
| **Throughput** | ~100 ops/sec sufficient | 10,000+ ops/sec needed |
| **Locking** | Pessimistic (safety) | Lock-free (speed) |

**Current Architecture Problem:** Database-level lock treats both the same way → ML logging is severely throttled (95 metrics/sec instead of 95,000).

---

## The Solution: Tiered Locking

Apply different locking strategies to different data classes:

```
┌─────────────────────────────────────────────────────────────┐
│                   TIERED ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TIER 1: CRITICAL (ACID)                                    │
│  ├─ Data: task.id, task.status, decision.id, edge.type    │
│  ├─ Locking: Pessimistic (row-level via hierarchy)         │
│  ├─ WAL: Synchronous fsync per transaction                 │
│  ├─ Isolation: Serializable (snapshot isolation)           │
│  └─ Throughput: ~100 ops/sec, latency ~50ms               │
│                                                              │
│  TIER 2: IMPORTANT (Strong Eventual Consistency)            │
│  ├─ Data: task.priority, edge.weight, indices              │
│  ├─ Locking: Optimistic (version-based retry)              │
│  ├─ WAL: Async fsync (batched every 5s)                    │
│  ├─ Isolation: Snapshot isolation                          │
│  └─ Throughput: ~500 ops/sec, latency ~10ms              │
│                                                              │
│  TIER 3: BEST-EFFORT (Eventual Consistency)                 │
│  ├─ Data: ML metrics, performance counters, debug info     │
│  ├─ Locking: None (lock-free, content-addressable)         │
│  ├─ Storage: CALI (session-based logs, no WAL)             │
│  ├─ Isolation: Eventual consistency                        │
│  └─ Throughput: 95,000+ ops/sec, latency <1ms            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Patterns

### Pattern 1: Hierarchical Locking

Replace database-level lock with row-level locks:

```python
# BEFORE: Database-level lock (everything serialized)
with global_lock:
    update_task()
    update_metrics()
    update_edge()
    # All sequential, blocking

# AFTER: Row-level locks (independent operations parallel)
with lock_mgr.acquire(['record_id', 'task_123']):
    update_task()  # Blocks only this task

with lock_mgr.acquire(['record_id', 'metric_abc']):
    update_metrics()  # Can run in parallel!
```

**Expected improvement:** 10x throughput increase (database-level was bottleneck).

### Pattern 2: Optimistic Locking

Use versioning for data without strict consistency requirements:

```python
# CRITICAL: Pessimistic (hold lock before write)
with lock_mgr.acquire(['record_id', entity_id]):
    entity = store.get(entity_id)  # Read under lock
    entity.status = 'completed'
    store.put(entity_id, entity)

# IMPORTANT: Optimistic (check version at commit)
entity = store.get(entity_id)
original_version = entity.version

entity.priority = 'high'

# Check if anyone else modified it
current = store.get(entity_id)
if current.version != original_version:
    retry()  # Version changed, retry
else:
    store.put(entity_id, entity)
```

**Expected improvement:** 3-5x faster for read-heavy workloads (no lock overhead).

### Pattern 3: Lock-Free Append-Only

Use content-addressable storage for best-effort data:

```python
# NO LOCKS: Multiple threads write to different files in parallel
# Each session gets unique file: 2025-12-17_10-30-45_abc123_metrics.jsonl
# Another session: 2025-12-17_10-30-45_def456_metrics.jsonl
# No collision, no lock needed!

store.put('metric', 'session_abc_metric_001', {
    'model': 'file_prediction',
    'accuracy': 0.95,
    'timestamp': '2025-12-17T10:30:45Z',
})
# Direct write, no lock, no WAL, <1ms latency ✓
```

**Expected improvement:** 100x throughput increase (from serialized to parallel).

---

## Implementation Roadmap

### Week 1: Hierarchical Locks

```python
# cortical/got/lock_hierarchy.py (NEW)

lock_mgr = HierarchicalLockManager(got_dir)

# Row-level lock for single entity
with lock_mgr.acquire(['record_id', 'task_123']):
    task = store.get('task_123')
    task.status = 'completed'
    store.put('task_123', task)

# Table-level lock for bulk operations
with lock_mgr.acquire(['entity_type', 'tasks']):
    for task in query_all_tasks():
        process(task)
```

**Benefit:** Reduce contention from database-level to row-level (~10x improvement).

### Week 2-3: Tier Classification

```python
# cortical/got/tiers.py (NEW)

tier_mgr = TierManager()

# Classify data
tier = tier_mgr.classify('task', 'status')  # → CRITICAL
tier = tier_mgr.classify('task', 'priority')  # → IMPORTANT
tier = tier_mgr.classify('metrics', 'cache_hits')  # → BEST_EFFORT

# Apply tier-appropriate locking
if tier == ConsistencyTier.CRITICAL:
    use_pessimistic_lock()  # Safety first
elif tier == ConsistencyTier.IMPORTANT:
    use_optimistic_lock()  # Performance balanced
else:
    use_lock_free_append()  # Maximum speed
```

**Benefit:** Optimal locking strategy per data type (~5-10x improvement per tier).

### Week 4: Update GoT Operations

```python
# Integrate tiered transactions into existing GoT methods

class GoTManager:
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update task with tier-aware consistency."""
        with self.tiered_tx() as tx:
            for field, value in updates.items():
                # Each field uses appropriate tier
                tx.write('task', task_id, field, value)
            return tx.commit()

    def log_metric(self, metric_name: str, value: float) -> None:
        """Log metric with lock-free append."""
        # Automatically uses BEST_EFFORT tier (lock-free)
        with self.tiered_tx() as tx:
            tx.write('metrics', self.session_id, metric_name, value)
```

**Benefit:** Transparent tier enforcement throughout codebase.

---

## Performance Expectations

### Before Tiered Locking

```
Operation                    Latency      Throughput       Contention
─────────────────────────────────────────────────────────────────────
Single task update           500ms        ~2 ops/sec       N/A
10 concurrent updates        5+ seconds   Serialized       80% wait
100 metric logs             >30 seconds   95 logs/sec      System blocked
```

### After Tiered Locking

```
Operation                    Latency      Throughput       Contention
─────────────────────────────────────────────────────────────────────
Single task update (CRIT)    50ms         ~20 ops/sec      Per-entity
Single priority update (IMP) 10ms         ~100 ops/sec     Retry-based
100 metric logs (BE)         <1ms each    95K logs/sec     Zero (parallel)
10 concurrent updates        500ms        10x better       <10% wait
```

### Improvement Ratios

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Task update p99 latency | 500ms | 50ms | **10x** |
| Concurrent throughput | Serial | 10x | **10x** |
| ML metrics throughput | 95/sec | 95K/sec | **1000x** |
| Lock contention | 80% | <10% | **8x** |
| Transaction abort rate | 5% | <1% | **5x** |

---

## Critical Implementation Details

### 1. Lock Ordering (Prevents Deadlock)

**Always acquire in this order:**
```
DATABASE → ENTITY_TYPE → RECORD_ID
```

**Never reverse the order** (causes circular wait):
```python
# WRONG - deadlock!
with lock_mgr.acquire(['record_id', 'entity_123']):
    with lock_mgr.acquire(['entity_type', 'tasks']):  # Circular wait!
        pass

# CORRECT - safe
with lock_mgr.acquire(['entity_type', 'tasks']):
    with lock_mgr.acquire(['record_id', 'entity_123']):  # Safe
        pass
```

### 2. Fsync Strategy by Tier

```python
# CRITICAL: Always sync to disk
store.put(entity_id, entity, fsync=True)  # Wait for disk

# IMPORTANT: Async flush (batched)
store.put(entity_id, entity, fsync=False)  # Return immediately

# BEST_EFFORT: Never sync (memory overflow to disk)
log_append(entity_id, entity)  # Append-only, no fsync
```

### 3. Conflict Detection (Optimistic Tier)

Track version when reading, check at commit:

```python
# Read phase
entity = store.get(entity_id)
original_version = entity.version

# Modify
entity.priority = 'high'

# Commit phase: check version unchanged
current = store.get(entity_id)
if current.version != original_version:
    # Someone else modified it! Retry
    raise VersionConflict("Retry operation")
```

---

## Codebase Integration Points

| File | Change | Benefit |
|---|---|---|
| `cortical/got/lock_hierarchy.py` | NEW: HierarchicalLockManager | Replace database-level lock |
| `cortical/got/tiers.py` | NEW: TierManager + ConsistencyTier enum | Classify data by tier |
| `cortical/got/tiered_transaction.py` | NEW: TieredTransaction | Tier-aware transaction semantics |
| `cortical/got/tx_manager.py` | MODIFY: Use hierarchical locks | Integrate lock hierarchy |
| `cortical/got/api.py` | MODIFY: Use tiered transactions | Transparent tier enforcement |
| `cortical/ml_storage.py` | ALREADY DONE: CALI storage | Already lock-free! |
| `cortical/utils/locking.py` | KEEP: ProcessLock still needed | Basis for hierarchical locks |

---

## Common Pitfalls & Solutions

### Pitfall 1: Deadlock from Lock Order Violation

**Problem:** Threads acquire locks in different orders → circular wait.

**Solution:** Enforce global lock ordering via HierarchicalLockManager.

```python
# Prevented by design
with lock_mgr.acquire(['entity_type', 'tasks']):  # First
    with lock_mgr.acquire(['record_id', 'entity_123']):  # Second (safe)
        # Never violates order
```

### Pitfall 2: Lost Updates with Optimistic Locking

**Problem:** Check version, then another thread modifies, we overwrite their change.

**Solution:** Atomic compare-and-swap at commit time:

```python
# NOT SAFE:
if current.version == original_version:
    current.value = new_value
    store.put(entity_id, current)
    # Another thread could modify between check and put!

# SAFE (atomic):
success = store.put_if_version(
    entity_id,
    updated_entity,
    expected_version=original_version
)
if not success:
    raise VersionConflict("Retry")
```

### Pitfall 3: Forgetting to Release Locks

**Solution:** Always use context managers:

```python
# WRONG - if exception occurs, lock never released
lock = lock_mgr.acquire(['record_id', 'task_123'])
update_task()
lock.release()

# CORRECT - lock released even on exception
with lock_mgr.acquire(['record_id', 'task_123']):
    update_task()  # Lock released automatically
```

### Pitfall 4: Holding Locks Too Long

**Problem:** Lock held during I/O → blocks other threads.

**Solution:** Minimize time in lock, do computation outside:

```python
# WRONG - lock held for 100ms computation
with lock_mgr.acquire(['record_id', 'task_123']):
    heavy_computation()  # 100ms
    task.result = result
    store.put('task_123', task)

# CORRECT - lock held only for store
result = heavy_computation()  # Outside lock, ~100ms
with lock_mgr.acquire(['record_id', 'task_123']):
    task.result = result
    store.put('task_123', task)  # Only ~1ms under lock
```

---

## Testing Strategy

### 1. Unit Tests (Lock Behavior)

```python
# tests/integration/test_hierarchical_locks.py

class TestLocks(unittest.TestCase):
    def test_concurrent_row_locks_parallel(self):
        """Multiple threads can hold different row locks."""
        # Spawn 10 threads, each updating different task
        # Should complete in ~100ms (parallel), not 1000ms (serial)

    def test_deadlock_prevention(self):
        """Circular wait is prevented."""
        # Try to acquire locks in reverse order
        # Should raise DeadlockDetected
```

### 2. Integration Tests (Tier Behavior)

```python
# tests/integration/test_tiered_transactions.py

class TestTiers(unittest.TestCase):
    def test_critical_tier_uses_pessimistic_lock(self):
        """CRITICAL fields use pessimistic locking."""
        # Track lock acquisitions
        # Verify task.status acquisition is pessimistic

    def test_important_tier_uses_optimistic_lock(self):
        """IMPORTANT fields use optimistic locking."""
        # No lock acquired on write
        # Version checked at commit

    def test_best_effort_tier_is_lock_free(self):
        """BEST_EFFORT fields use lock-free append."""
        # No lock acquired
        # No fsync
        # Throughput > 10K ops/sec
```

### 3. Performance Tests (Benchmarks)

```bash
# Run YCSB benchmarks
python benchmarks/run_benchmarks.py --workload a --agents 10

# Compare before/after
python benchmarks/run_benchmarks.py --compare

# Profile lock contention
python benchmarks/profile_locks.py --duration 60
```

---

## Documentation Structure

This research includes four documents:

| Document | Focus | Audience |
|---|---|---|
| **tiered-locking-patterns.md** | Theory & principles | Architects, researchers |
| **tiered-locking-implementation.md** | Step-by-step code | Developers |
| **tiered-locking-benchmarks.md** | Performance modeling | Performance engineers |
| **tiered-locking-summary.md** | Quick reference | Everyone |

---

## Success Criteria

Implementation is complete when:

- [ ] Lock contention < 10% (was 80%)
- [ ] Single task update latency p99 < 100ms (was 500ms)
- [ ] 10 concurrent task updates complete in <1s (was 5s)
- [ ] ML metric throughput > 10K ops/sec (was 95)
- [ ] Transaction abort rate < 2% (was 5%)
- [ ] All tests pass with 100% coverage
- [ ] No deadlocks detected under stress test (100 agents)

---

## Next Steps

1. **Read** tiered-locking-patterns.md (theoretical foundation)
2. **Implement** Phase 1 from tiered-locking-implementation.md (hierarchical locks)
3. **Benchmark** using tiered-locking-benchmarks.md (establish baseline)
4. **Implement** Phase 2 (tier classification)
5. **Test** with tiered-locking-benchmarks.md (verify improvements)
6. **Deploy** to production with monitoring

Total effort: **4-5 weeks** for 10-15x throughput improvement.

---

## References

Key papers and resources:

- **Concurrency Control (Database Theory):**
  - "Designing Data-Intensive Applications" (Martin Kleppmann) - Chapter 7
  - "Database Internals" (Alex Petrov) - Chapters 11-12

- **Lock-Free Algorithms:**
  - "The Art of Multiprocessor Programming" (Herlihy & Shavit) - Chapters 5-6
  - "Lock-Free Programming" (Herb Sutter) - Concurrency patterns

- **Practical Implementations:**
  - PostgreSQL MVCC: `src/backend/access/transam/`
  - CockroachDB Distributed Transactions: `pkg/storage/txn/`
  - DynamoDB Eventual Consistency: Amazon DynamoDB paper (2007)

---

## Questions?

Refer to the detailed documents:
- **How does hierarchical locking work?** → tiered-locking-patterns.md, Part 1
- **How do I implement this?** → tiered-locking-implementation.md
- **What performance should I expect?** → tiered-locking-benchmarks.md
- **How do I debug a deadlock?** → tiered-locking-patterns.md, Appendix D
- **What are the trade-offs?** → tiered-locking-patterns.md, Part 2

