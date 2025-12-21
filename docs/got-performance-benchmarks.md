# GoT Transactional System - Performance Benchmarks

**Date:** 2025-12-21
**System:** Linux 4.4.0
**Python:** 3.x

## Executive Summary

The GoT transactional system prioritizes **reliability over speed**. The `fsync` calls that ensure data durability account for 74.5% of execution time. This is intentional - we trade throughput for guaranteed crash recovery.

## Benchmark Results

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

### Throughput

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
