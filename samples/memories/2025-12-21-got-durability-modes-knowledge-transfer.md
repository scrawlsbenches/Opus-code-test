# Knowledge Transfer: GoT Configurable Durability Modes

**Date:** 2025-12-21
**Author:** Claude (Opus 4.5)
**Branch:** `claude/test-task-workflow-gRXq7`

---

## Context

The GoT transactional system currently uses maximum durability (`fsync` on every operation), which accounts for **74.5% of execution time**. This is over-engineered for task tracking data where:
- Data is recoverable (recreate tasks in 30 seconds)
- Power loss is rare on dev machines
- Process crashes are the main concern, not hardware failures

User requested configurable durability to balance reliability vs performance based on use case.

---

## Design: Three Durability Modes

### Mode 1: `paranoid` (Current Behavior)
**Use when:** Debugging reliability issues, critical data, untrusted environment

```
fsync on:
├── Every WAL sequence increment
├── Every WAL log entry
├── Every entity file write
└── Every version file update

Performance: ~36 ops/s (single), ~114 ops/s (bulk)
Guarantees: Zero data loss even on power failure mid-operation
```

### Mode 2: `balanced` (Recommended Default)
**Use when:** Normal operation, confident in system reliability

```
fsync on:
├── Transaction COMMIT only (WAL)
└── Batch entity files at commit

Performance: Expected ~150-200 ops/s
Guarantees: Committed transactions survive power loss
Risk: Uncommitted transaction might lose WAL entries (rolled back anyway)
```

### Mode 3: `relaxed` (OS Handles It)
**Use when:** Maximum performance, acceptable risk, frequent saves to git

```
fsync on:
└── Never (rely on OS buffer cache)

Performance: Expected ~500+ ops/s
Guarantees: Survives process crash (data in kernel buffer)
Risk: Power loss within OS flush window (~5-30s) loses data
```

---

## Implementation Plan

### Files to Modify

1. **`cortical/got/config.py`** (NEW)
   - Create `DurabilityMode` enum
   - Create `GoTConfig` dataclass with durability setting

2. **`cortical/got/wal.py`**
   - Accept durability mode in constructor
   - Conditional fsync in `log()` method
   - Add `fsync_now()` method for explicit sync

3. **`cortical/got/versioned_store.py`**
   - Accept durability mode in constructor
   - Conditional fsync in `_fsync_file()`
   - Batch fsync support for `apply_writes()`

4. **`cortical/got/tx_manager.py`**
   - Pass durability mode to WAL and store
   - Call explicit fsync on commit for `balanced` mode

5. **`cortical/got/api.py`**
   - Accept `durability` parameter in `GoTManager.__init__()`
   - Default to `balanced` mode

### API Changes

```python
from cortical.got import GoTManager, DurabilityMode

# Paranoid mode (current behavior)
manager = GoTManager(".got", durability=DurabilityMode.PARANOID)

# Balanced mode (recommended)
manager = GoTManager(".got", durability=DurabilityMode.BALANCED)

# Relaxed mode (fastest)
manager = GoTManager(".got", durability=DurabilityMode.RELAXED)

# Default is BALANCED
manager = GoTManager(".got")  # Uses BALANCED
```

### Backward Compatibility

- Existing code without `durability` parameter gets `BALANCED` (slight behavior change)
- To preserve exact current behavior, use `DurabilityMode.PARANOID`
- All tests should pass with any durability mode (correctness unchanged)

---

## Test Strategy

### Unit Tests
- Test each durability mode initializes correctly
- Test fsync is called/not called based on mode
- Test explicit `fsync_now()` works in all modes

### Integration Tests
- Test crash recovery works in all modes
- Test committed data survives in PARANOID and BALANCED
- Test data consistency after simulated crashes

### Performance Tests
- Benchmark all three modes with identical workload
- Compare: single ops, bulk ops, mixed read/write
- Document results in `docs/got-performance-benchmarks.md`

---

## Constraints for Sub-Agents

### MUST:
- Keep all existing tests passing
- Add durability parameter without breaking existing API
- Document fsync behavior in docstrings
- Use `DurabilityMode` enum (not strings)

### MUST NOT:
- Change transaction semantics
- Remove fsync entirely from PARANOID mode
- Break crash recovery in any mode
- Change file formats or WAL structure

### SHOULD:
- Default to BALANCED mode
- Log durability mode on manager init (debug level)
- Keep changes minimal and focused

---

## Expected Performance Targets

| Mode | Single Create | Bulk 100 | Improvement |
|------|---------------|----------|-------------|
| PARANOID | 28ms | 878ms | Baseline |
| BALANCED | 10-15ms | 200-300ms | ~3x |
| RELAXED | 2-5ms | 50-100ms | ~10x |

---

## Current State Summary

### Completed (This Session)
- ✅ Lock timeout with retry and stale lock recovery
- ✅ Orphan entity auto-repair in recovery
- ✅ Query API (find_tasks, get_blockers, etc.)
- ✅ Performance benchmarks documented
- ✅ 260 tests passing

### Files Already Modified
- `cortical/got/tx_manager.py` - Lock timeout
- `cortical/got/recovery.py` - Orphan repair
- `cortical/got/api.py` - Query API
- `cortical/got/__init__.py` - Exports

### Test Count
- 223 unit tests (got/)
- 22 integration tests
- 15 migration tests

---

## Handoff Notes

1. **Start with config.py** - Define the enum and config first
2. **Modify WAL second** - Most fsync calls are here
3. **Modify store third** - Entity file fsyncs
4. **Wire up in tx_manager** - Pass config through
5. **Expose in api.py last** - User-facing API

The changes are surgical - each file has specific fsync calls to make conditional. Don't refactor unrelated code.

---

## Questions for Future Sessions

1. Should we add a `manager.force_sync()` method for explicit durability checkpoints?
2. Should BALANCED fsync on every N commits instead of every commit?
3. Should we expose durability mode in CLI tools?
