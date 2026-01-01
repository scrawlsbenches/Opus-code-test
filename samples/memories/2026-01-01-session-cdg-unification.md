# Session Knowledge Transfer: 2026-01-01 CDG Unification

**Date:** 2026-01-01
**Session:** Cortical Distributed Graph (CDG) and GoT Unification
**Branch:** `claude/distributed-git-graph-8iuIR`
**Previous Session:** `2025-12-31-session-cdg-bootstrap.md`

---

## Summary

This session completed the unification of GoT and CDG, transforming CDG from a specification into a fully configurable storage layer with transaction management, write-ahead logging, and crash recovery. GoT now delegates all transaction operations to CDG, achieving the "best of both worlds" goal. All 75+ behavioral tests pass.

**Key Achievement:** CDG is now a configurable storage layer that can serve different use cases (GoT workloads, simple storage, high-performance ephemeral) through configuration presets.

---

## What Was Accomplished

### 1. CDGConfig Enhancement

Extended CDGConfig with comprehensive configuration options:

```python
# New enums added:
class IsolationLevel(Enum):
    SNAPSHOT = "snapshot"
    READ_COMMITTED = "read_committed"

class RecoveryMode(Enum):
    NONE = "none"       # No recovery (ephemeral)
    CHECKSUM = "checksum"  # Verify checksums only
    FULL = "full"       # WAL replay + orphans + checksums

class OrphanStrategy(Enum):
    FAIL = "fail"       # Error on orphans
    DELETE = "delete"   # Remove orphans
    REPAIR = "repair"   # Adopt orphans via WAL
```

**Configuration presets created:**
- `CDGConfig.for_got()` - Full ACID with WAL and recovery
- `CDGConfig.for_simple_storage()` - Basic storage, no transactions
- `CDGConfig.for_high_performance()` - Maximum speed, ephemeral

**File:** `/home/user/Opus-code-test/cortical/cdg/config.py` (+175 lines)

### 2. CDGWALManager Implementation

Lifted WALManager from GoT into CDG with configuration integration:

```python
class CDGWALManager:
    def __init__(self, wal_dir: Path, config: CDGConfig): ...

    # Transaction lifecycle logging
    def log_tx_begin(self, tx_id: str, snapshot_version: int) -> int: ...
    def log_write(self, tx_id: str, entity_id: str, old_v: int, new_v: int) -> int: ...
    def log_tx_prepare(self, tx_id: str) -> int: ...
    def log_tx_commit(self, tx_id: str, version: int) -> int: ...
    def log_tx_abort(self, tx_id: str, reason: str) -> int: ...
    def log_tx_rollback(self, tx_id: str, reason: str) -> int: ...

    # Recovery support
    def replay() -> List[Dict[str, Any]]: ...
    def get_incomplete_transactions() -> List[Dict[str, Any]]: ...
```

**Key features:**
- JSONL format with checksums (TransactionWALEntry)
- Archive/truncate support
- Durability mode integration (fsync behavior)
- Process-safe via ProcessLock

**File:** `/home/user/Opus-code-test/cortical/cdg/wal.py` (+467 lines)

### 3. CDGTransactionManager Implementation

Created full ACID transaction orchestration:

```python
class CDGTransactionManager:
    def __init__(self, store_dir: Path, config: CDGConfig, entity_factory: Optional[EntityFactory] = None): ...

    def begin(self) -> Transaction: ...
    def read(self, tx: Transaction, entity_id: str) -> Optional[Entity]: ...
    def write(self, tx: Transaction, entity: Entity) -> None: ...
    def commit(self, tx: Transaction) -> CommitResult: ...
    def rollback(self, tx: Transaction, reason: str = "explicit") -> None: ...
    def recover(self) -> RecoveryResult: ...
```

**ACID guarantees:**
- **Atomicity:** All writes in transaction succeed or all fail
- **Consistency:** Checksums verify data integrity
- **Isolation:** Snapshot isolation via read_at_version
- **Durability:** WAL + fsync (mode-dependent)

**Conflict detection:** Optimistic locking via read_set version comparison

**File:** `/home/user/Opus-code-test/cortical/cdg/transaction_manager.py` (+410 lines)

### 4. CDGRecoveryManager Implementation

Created configurable crash recovery:

```python
class CDGRecoveryManager:
    def __init__(self, store_dir: Path, config: CDGConfig, entity_factory: Optional[EntityFactory] = None): ...

    def needs_recovery(self) -> bool: ...
    def recover(self) -> RecoveryResult: ...
    def verify_store_integrity(self) -> List[str]: ...  # Corrupted entity IDs
    def verify_wal_integrity(self) -> int: ...  # Corrupted entry count
    def rollback_incomplete_transactions(self) -> List[str]: ...
    def detect_orphaned_entities(self) -> List[str]: ...
    def repair_orphans(self, strategy: OrphanStrategy = None) -> RepairResult: ...
```

**Recovery cascade (FULL mode):**
1. Rollback incomplete transactions (from WAL)
2. Detect orphaned entities (disk vs WAL comparison)
3. Repair orphans (based on OrphanStrategy)
4. Verify entity checksums
5. Verify WAL integrity
6. Rebuild indexes (via callback)

**File:** `/home/user/Opus-code-test/cortical/cdg/recovery.py` (+632 lines)

### 5. GoT Integration

Modified GoT's TransactionManager to delegate to CDG:

```python
# cortical/got/tx_manager.py
class TransactionManager:
    def __init__(self, got_dir: Path, durability: DurabilityMode = ...):
        # Create CDG components with GoT's directory structure
        cdg_config = CDGConfig.for_got()
        self.store = CDGStore(got_dir / "entities", entity_factory=_got_entity_factory)
        self.wal = CDGWALManager(got_dir / "wal", cdg_config)

        # Create CDGTransactionManager and delegate
        self._cdg_tx = CDGTransactionManager(...)

    def begin(self) -> Transaction:
        return self._cdg_tx.begin()

    def read(self, tx, entity_id):
        return self._cdg_tx.read(tx, entity_id)

    # ... all methods delegate to self._cdg_tx
```

**Key preservation:**
- GoT's public API unchanged
- Entity type dispatch works (Task, Decision, Sprint types preserved)
- Directory structure unchanged for backward compatibility
- All 75+ GoT tests pass

**Files modified:**
- `/home/user/Opus-code-test/cortical/got/tx_manager.py`
- `/home/user/Opus-code-test/cortical/got/transaction.py`

---

## Architecture After Unification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                                    │
│  (GoT CLI, ThoughtGraph, KnowledgeGraph)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                     GOT TRANSACTION API                                  │
│  TransactionManager (thin wrapper → delegates to CDG)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                     CDG TRANSACTION LAYER                                │
│  CDGTransactionManager                                                   │
│  ├── begin/read/write/commit/rollback                                   │
│  ├── Snapshot isolation (read_at_version)                               │
│  └── Optimistic locking (read_set conflict detection)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                     CDG WAL LAYER                                        │
│  CDGWALManager                                                           │
│  ├── TX_BEGIN, WRITE, TX_COMMIT logging                                 │
│  ├── JSONL format with checksums                                        │
│  └── Archive/truncate support                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                     CDG RECOVERY LAYER                                   │
│  CDGRecoveryManager                                                      │
│  ├── RecoveryMode: NONE, CHECKSUM, FULL                                 │
│  ├── OrphanStrategy: FAIL, DELETE, REPAIR                               │
│  └── Auto-recovery on startup                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                     CDG STORAGE LAYER                                    │
│  CDGStore                                                                │
│  ├── Entity JSON files with checksums                                   │
│  ├── History files for MVCC (snapshot isolation)                        │
│  └── Thread + process safety via locks                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| **Configuration presets** | Different use cases need different trade-offs | Single config (too rigid), per-feature flags (too granular) |
| **WAL as optional** | Simple apps don't need crash recovery overhead | Always-on (slow), never (unsafe) |
| **Orphan REPAIR default** | Preserve data from git-tracked files | DELETE (data loss), FAIL (blocks startup) |
| **Snapshot isolation only** | Simplest correct isolation level | READ_COMMITTED (more complex), Serializable (overkill) |
| **GoT delegates entirely** | Single source of truth for transactions | Partial delegation (complexity), keep separate (duplication) |
| **Entity factory pattern** | GoT needs Task/Decision types, CDG is generic | Hardcode GoT types in CDG (violates separation) |

---

## Problems Encountered & Solutions

### Problem 1: Circular Import Between Transaction Manager and Recovery

**Symptom:** Import error when CDGTransactionManager tried to import CDGRecoveryManager.

**Root Cause:** CDGTransactionManager.recover() needed CDGRecoveryManager, but both were importing each other.

**Solution:** Used lazy import inside recover() method:
```python
def recover(self) -> "RecoveryResult":
    from .recovery import CDGRecoveryManager, RecoveryResult
    recovery_mgr = CDGRecoveryManager(self.store_dir, self.config)
    return recovery_mgr.recover()
```

### Problem 2: Entity Type Dispatch in CDG

**Symptom:** Reads returning base `Entity` instead of `Task`, `Decision`, etc.

**Root Cause:** CDGStore is generic and doesn't know about GoT types.

**Solution:** Entity factory pattern:
```python
def _got_entity_factory(data: dict) -> Entity:
    entity_type = data.get("entity_type")
    if entity_type == "task":
        return Task.from_dict(data)
    elif entity_type == "decision":
        return Decision.from_dict(data)
    # ... etc

# Pass to CDGStore
store = CDGStore(path, entity_factory=_got_entity_factory)
```

### Problem 3: Directory Structure Compatibility

**Symptom:** GoT expected `entities/` subdirectory, CDG was using root.

**Root Cause:** Different directory conventions between GoT and CDG.

**Solution:** GoT's TransactionManager manually creates CDG components with correct paths:
```python
self.store = CDGStore(got_dir / "entities", ...)
self.wal = CDGWALManager(got_dir / "wal", ...)
```

---

## Technical Insights

### 1. Configuration Presets Enable Flexibility

The preset pattern allows same code with different behaviors:

```python
# Full ACID (GoT default)
config = CDGConfig.for_got()
# transactions=True, wal=True, recovery=FULL

# Simple storage (cache, ephemeral data)
config = CDGConfig.for_simple_storage()
# transactions=False, wal=False, recovery=CHECKSUM

# Maximum performance (benchmarks, throwaway data)
config = CDGConfig.for_high_performance()
# transactions=False, wal=False, recovery=NONE, durability=FAST
```

### 2. WAL Entry Format Unchanged

CDG uses the same WAL entry format as GoT (TransactionWALEntry from cortical.wal):

```json
{
    "seq": 1,
    "ts": "2026-01-01T01:00:00.000000+00:00",
    "tx": "TX-20260101-010000-abc12345",
    "op": "TX_BEGIN",
    "data": {"snapshot": 42},
    "checksum": "abcd1234efgh5678"
}
```

This ensures:
- Existing WAL files remain readable
- Recovery works across the migration
- No data format migration needed

### 3. Orphan Detection Algorithm

Orphan detection compares disk entities with WAL records:

```python
# 1. Get all entity IDs from disk
disk_entities = {f.stem for f in store_dir.glob("*.json")
                 if not f.name.startswith("_")}

# 2. Get all entity IDs from WAL (WRITE operations + ADOPTED entries)
wal_entities = {entry["data"]["entity_id"]
                for entry in wal.replay()
                if entry["op"] in ("WRITE", "ADOPTED")}

# 3. Orphans = disk entities not in WAL
orphans = disk_entities - wal_entities
```

### 4. Recovery Mode Decision Tree

```
Is data ephemeral/cacheable?
  YES → RecoveryMode.NONE (fastest startup)
  NO  → Does the app use transactions?
          YES → RecoveryMode.FULL (WAL replay + orphans + checksums)
          NO  → RecoveryMode.CHECKSUM (just verify data integrity)
```

---

## Test Results

| Test Suite | Result |
|------------|--------|
| Smoke tests | **18 passed** |
| GoT workflow tests | **26 passed** (2 skipped) |
| Graph persistence tests | **31 passed** |
| **Total** | **75 passed, 2 skipped, 0 failed** |

---

## Files Created/Modified

### New Files (1,809 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `cortical/cdg/wal.py` | 467 | CDGWALManager - write-ahead logging |
| `cortical/cdg/transaction_manager.py` | 410 | CDGTransactionManager - ACID transactions |
| `cortical/cdg/recovery.py` | 632 | CDGRecoveryManager - crash recovery |

### Modified Files

| File | Changes | Purpose |
|------|---------|---------|
| `cortical/cdg/config.py` | +175 lines | New enums and presets |
| `cortical/cdg/__init__.py` | +32 lines | New exports |
| `cortical/got/tx_manager.py` | Refactored | Delegate to CDG |
| `cortical/got/transaction.py` | Replaced | Re-export from CDG |

---

## Commits

| Hash | Message |
|------|---------|
| `ff93d07b` | fix: Add thread/process safety to CDGStore and fix indexer field names |
| `bdef0e14` | feat: Unify GoT and CDG with configurable transaction/WAL/recovery layers |

---

## Context for Next Session

### Current State

**Completed:**
- CDG is a fully configurable storage layer
- GoT delegates all transactions to CDG
- All tests pass
- Thread/process safety implemented
- Crash recovery with configurable strategies

**Future Work:**
- Partition support (currently partition_count=1)
- Query layer (fluent API with partition hints)
- Index integration (currently via callback)
- Performance benchmarks

### Suggested Next Steps

1. **Add CDG unit specifications** (`tests/unit/specifications/cdg/`)
   - `wal_spec.py` - WAL entry format, checksums
   - `transaction_manager_spec.py` - ACID guarantees
   - `recovery_spec.py` - Recovery cascade

2. **Performance characterization**
   - Baseline latency measurements
   - WAL overhead analysis
   - Recovery time analysis

3. **ThoughtGraph migration**
   - Use `CDGConfig.for_simple_storage()`
   - Or enable transactions if needed

### Files to Review

**CDG entry points:**
1. `/home/user/Opus-code-test/cortical/cdg/__init__.py` - Public API
2. `/home/user/Opus-code-test/cortical/cdg/config.py` - Configuration options
3. `/home/user/Opus-code-test/cortical/cdg/transaction_manager.py` - ACID transactions

**GoT integration:**
4. `/home/user/Opus-code-test/cortical/got/tx_manager.py` - How GoT uses CDG

---

## Lessons Learned

### 1. Configuration Presets > Feature Flags

Instead of exposing many boolean flags, provide named presets that represent common use cases. Users can still customize, but presets capture intent.

### 2. Delegation > Duplication

GoT's TransactionManager became a thin wrapper that delegates to CDG. This avoided duplicating transaction logic and ensures single source of truth.

### 3. Entity Factory Pattern for Type Dispatch

Generic storage (CDG) + domain types (Task, Decision) = entity factory pattern. CDG doesn't know about GoT types, but GoT can teach it via factory function.

### 4. Lazy Imports Break Cycles

When modules have circular dependencies, use lazy imports inside methods rather than at module level.

### 5. Director Pattern for Complex Work

The 6-phase orchestration with parallel agents where possible and sequential where needed completed this major refactoring efficiently.

---

## Tags

`cdg`, `got-unification`, `transactions`, `wal`, `recovery`, `configuration`, `phase-2-complete`, `acid`, `crash-recovery`, `director-orchestration`

---

*Session completed: 2026-01-01 | Branch: claude/distributed-git-graph-8iuIR | CDG Unification: Complete*
