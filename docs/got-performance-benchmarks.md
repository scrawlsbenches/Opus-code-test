# GoT Transactional System - Performance Benchmarks

**Date:** 2025-12-21
**System:** Linux 4.4.0
**Python:** 3.x

## Executive Summary

The GoT transactional system now supports **three durability modes** to balance reliability vs performance:

| Mode | Use Case | Speedup vs PARANOID |
|------|----------|---------------------|
| **PARANOID** | Debugging, critical data | Baseline |
| **BALANCED** | Normal operation (default) | 2-2.5x |
| **RELAXED** | Maximum performance | 6-10x |

## Durability Modes Comparison

### Latency (milliseconds)

| Operation | PARANOID | BALANCED | RELAXED |
|-----------|----------|----------|---------|
| Create task (avg) | 33.78 | 17.24 | 5.71 |
| Create task (min) | 30.82 | 16.06 | 5.06 |
| Create task (max) | 35.99 | 19.90 | 6.99 |
| Read task (avg) | 10.72 | 0.84 | 0.96 |
| Transaction (5 ops) | 67.11 | 36.46 | 9.30 |
| Bulk 100 tasks | 909.16 | 383.73 | 94.24 |
| Per-task in bulk | 9.09 | 3.84 | 0.94 |

### Throughput (ops/second)

| Operation | PARANOID | BALANCED | RELAXED |
|-----------|----------|----------|---------|
| Single creates | 29.6 | 58.0 | 175.2 |
| Bulk creates | 110.0 | 260.6 | 1061.2 |
| Reads | 93.3 | 1193.2 | 1046.7 |

### Speedup vs PARANOID

| Operation | PARANOID | BALANCED | RELAXED |
|-----------|----------|----------|---------|
| Single creates | 1.0x | **2.0x** | **5.9x** |
| Bulk 100 tasks | 1.0x | **2.4x** | **9.6x** |
| Reads | 1.0x | **12.8x** | **11.2x** |

## Mode Selection Guide

### PARANOID Mode
```python
manager = GoTManager(".got", durability=DurabilityMode.PARANOID)
```
- fsync on EVERY operation (WAL entry, entity file, version file)
- **Use when:** Debugging reliability issues, untrusted environment, critical data
- **Guarantees:** Zero data loss even on power failure mid-operation
- **Performance:** ~30 ops/s single, ~110 ops/s bulk

### BALANCED Mode (Default)
```python
manager = GoTManager(".got", durability=DurabilityMode.BALANCED)
# OR simply:
manager = GoTManager(".got")  # Defaults to BALANCED
```
- fsync only on transaction COMMIT
- **Use when:** Normal operation, confident in system stability
- **Guarantees:** Committed transactions survive power loss
- **Risk:** Uncommitted transaction might lose WAL entries (rolled back anyway)
- **Performance:** ~58 ops/s single, ~261 ops/s bulk

### RELAXED Mode
```python
manager = GoTManager(".got", durability=DurabilityMode.RELAXED)
```
- No fsync calls, rely on OS buffer cache
- **Use when:** Maximum performance, acceptable risk, frequent git saves
- **Guarantees:** Survives process crash (data in kernel buffer)
- **Risk:** Power loss within OS flush window (~5-30s) loses uncommitted data
- **Performance:** ~175 ops/s single, ~1061 ops/s bulk

## Legacy Benchmark Results (PARANOID Mode)

### Operation Latencies

| Operation | Avg (ms) | Min (ms) | Max (ms) | Notes |
|-----------|----------|----------|----------|-------|
| Manager init | 2.69 | - | - | One-time startup cost |
| Create task (single) | 27.75 | 26.24 | 31.52 | Includes WAL + fsync |
| Create task (bulk/100) | 8.77 | - | - | Per-task in single TX |
| Read task | 9.70 | 9.09 | 10.31 | Includes checksum verify |
| Update task | 28.08 | - | - | Similar to create |
| Transaction (5 ops) | 58.54 | - | - | Multi-op atomic commit |
| Create decision | 25.08 | - | - | Similar to task |
| Recovery check | 11.04 | - | - | WAL scan |
| Integrity check | 5.06 | - | - | Checksum verification |
| Bulk 100 tasks | 877.61 | - | - | All in single TX |

### Throughput (PARANOID)

| Mode | Operations/Second |
|------|-------------------|
| Single task creates | 36 ops/s |
| Bulk task creates | 114 ops/s |
| Task reads | 103 ops/s |

### Time Distribution

```
Total benchmark time: 1,887 ms

fsync calls:     1,406 ms (74.5%)  ← Durability cost
WAL logging:       123 ms (6.5%)
JSON serialize:     89 ms (4.7%)
Checksum compute:   67 ms (3.6%)
Other:             202 ms (10.7%)
```

## Why fsync Dominates

Every write operation triggers multiple `fsync` calls:

1. **WAL sequence file** - Increment and persist sequence number
2. **WAL log entry** - Append and sync the operation record
3. **Entity file** - Write and sync the actual data
4. **Version file** - Update and sync global version

This ensures:
- **No data loss on crash** - All committed data survives power failure
- **Consistent recovery** - WAL can replay incomplete transactions
- **Checksum integrity** - Corruption is always detected

## Performance Implications

### For GoT Scripts (Typical Usage)

| Scenario | Expected Performance |
|----------|---------------------|
| Create 10 tasks | ~280ms |
| Create 100 tasks (single TX) | ~880ms |
| Read 50 tasks | ~485ms |
| Full integrity check | ~5ms per 100 entities |

### Recommendations

1. **Use transactions for bulk operations** - 3x faster per-task
2. **Batch related changes** - One TX for parent + children + edges
3. **Read operations are fast** - Don't cache unless profiling shows need
4. **Recovery is cheap** - ~11ms to check if needed

## Comparison to Alternatives

| System | Create Task | Durability |
|--------|-------------|------------|
| GoT TX (this) | 28ms | Full (fsync + WAL) |
| SQLite WAL mode | ~5ms | Full |
| JSON file (no fsync) | ~1ms | None (data loss risk) |
| In-memory only | ~0.01ms | None |

The ~28ms per operation is the cost of **guaranteed durability** without external dependencies.

## Acceptable Performance Thresholds

For GoT script usage, these latencies are acceptable:

| Operation | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Single task create | <100ms | 28ms | ✅ OK |
| Bulk 100 tasks | <2000ms | 878ms | ✅ OK |
| Task read | <50ms | 10ms | ✅ OK |
| Recovery check | <100ms | 11ms | ✅ OK |

## Bottleneck Analysis

The `fsync` calls are in:
- `wal.py:69-70` - Sequence file sync
- `wal.py:108-109` - WAL entry sync
- `versioned_store.py:201` - Entity file sync
- `versioned_store.py:384` - Version file sync

**These are NOT bugs** - they are required for ACID durability. Removing them would cause data loss on crash.

## Future Optimizations (If Needed)

If performance becomes critical:

1. **Group commit** - Batch multiple transactions into one fsync (reduces durability window)
2. **Async fsync** - Background sync with in-memory buffer (adds complexity)
3. **WAL compaction** - Periodic cleanup of old WAL entries

Current performance is adequate for GoT script workloads (hundreds of tasks, not millions).
