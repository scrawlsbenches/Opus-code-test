# Code Duplication Analysis Report

## Executive Summary

Found **MAJOR DUPLICATIONS** across three areas:
1. **ProcessLock** class duplicated between `cortical/got/tx_manager.py` and `scripts/got_utils.py`
2. **THREE separate WAL implementations** serving similar purposes
3. **ID generation functions** duplicated between `cortical/got/api.py` and `scripts/got_utils.py`
4. **Multiple Conflict-related classes** with overlapping functionality

---

## 1. ProcessLock Class - SIGNIFICANT DUPLICATION

### Files:
- `/home/user/Opus-code-test/cortical/got/tx_manager.py` (lines 38-257)
- `/home/user/Opus-code-test/scripts/got_utils.py` (lines 317-530)

### Duplication Level: ~90% similar functionality

**Both implementations provide:**
- File-based process locking using fcntl.flock()
- Stale lock detection via PID checking (os.kill(pid, 0))
- Timeout support with retry logic
- Reentrant lock support
- Context manager interface (__enter__/__exit__)
- Lock holder metadata (PID, timestamp)

**Key differences:**
- `cortical/got/tx_manager.py`: Uses thread lock, exponential backoff (10ms-500ms)
- `scripts/got_utils.py`: Uses fixed 50ms poll interval, has stale_timeout parameter

**Recommendation:** Consolidate into a single implementation in `cortical/got/tx_manager.py` and have `scripts/got_utils.py` import it.

---

## 2. WAL (Write-Ahead Log) - THREE IMPLEMENTATIONS

### Files:
1. **`/home/user/Opus-code-test/cortical/wal.py`** (720 lines)
   - For: Cortical Text Processor persistence
   - Classes: WALWriter, WALEntry, SnapshotManager, WALRecovery
   - Features: Document operations, compute phases, staleness tracking

2. **`/home/user/Opus-code-test/cortical/got/wal.py`** (322 lines)
   - For: GoT transactional system
   - Classes: WALManager
   - Features: Transaction logging (TX_BEGIN, WRITE, TX_COMMIT)

3. **`/home/user/Opus-code-test/cortical/reasoning/graph_persistence.py`** (565-1132)
   - For: Graph of Thought persistence
   - Classes: GraphWAL, GraphWALEntry
   - Features: Graph operations (add_node, add_edge, remove_node)
   - **IMPORTS from cortical.wal** (WALWriter, WALEntry, SnapshotManager)

**Duplication Assessment:**
- GraphWAL wraps/extends WALEntry from cortical.wal (good reuse)
- WALManager (cortical/got/wal.py) is independent and duplicates concepts from cortical.wal
- All three use similar patterns: JSONL format, checksums (SHA256[:16]), fsync for durability

**Recommendation:**
- Keep cortical.wal as base implementation
- Refactor cortical/got/wal.py to extend/use cortical.wal instead of reimplementing
- GraphWAL is already doing this correctly (imports from cortical.wal)

---

## 3. ID Generation Functions - NEAR DUPLICATES

### Files:
- `/home/user/Opus-code-test/cortical/got/api.py` (lines 40-67)
- `/home/user/Opus-code-test/scripts/got_utils.py` (lines 102-136)

**Duplicated functions:**

#### generate_task_id()
- `cortical/got/api.py`: Returns `T-YYYYMMDD-HHMMSS-XXXXXXXX` (8 hex chars, secrets.token_hex(4))
- `scripts/got_utils.py`: Returns `task:T-YYYYMMDD-HHMMSS-XXXX` (4 hex chars, os.urandom(2), with "task:" prefix)

#### generate_decision_id()
- `cortical/got/api.py`: Returns `D-YYYYMMDD-HHMMSS-XXXXXXXX`
- `scripts/got_utils.py`: Returns `decision:D-YYYYMMDD-HHMMSS-XXXX` (with "decision:" prefix)

**Recommendation:** Consolidate in cortical/got/api.py, import in scripts/got_utils.py. Decide on one ID format.

---

## 4. Conflict Classes - MULTIPLE IMPLEMENTATIONS

### Found in:
- `cortical/got/conflict.py`: ConflictStrategy, SyncConflict, ConflictResolver
- `cortical/got/tx_manager.py`: Conflict (line 261)
- `cortical/got/errors.py`: ConflictError
- `cortical/reasoning/collaboration.py`: ConflictType, ConflictEvent, ConflictDetail
- `cortical/reasoning/context_pool.py`: ConflictResolutionStrategy

**Assessment:** These serve different purposes:
- GoT conflicts: Sync/transaction conflicts in versioned store
- Reasoning conflicts: Agent collaboration conflicts
- Context pool conflicts: Context finding conflicts

**Recommendation:** Not true duplicates - domain-specific. Consider namespace clarification or shared base class if patterns converge.

---

## 5. Checksum Computation - SIMILAR PATTERNS

### Found in:
- `cortical/got/checksums.py`: compute_checksum(data) - standalone function
- `cortical/got/types.py`: Entity.compute_checksum() - method
- `cortical/reasoning/graph_persistence.py`: GraphWALEntry._compute_checksum() - method
- `cortical/wal.py`: WALEntry._compute_checksum() - method
- `cortical/reasoning/context_pool.py`: _compute_checksum() - method

**All use:** SHA256 hash, [:16] truncation, JSON serialization with sort_keys=True

**Recommendation:** Extract to shared utility in cortical/got/checksums.py, import everywhere. Single source of truth for checksum algorithm.

---

## 6. Task Management Methods - SIMILAR SIGNATURES

### Classes with similar task methods:
- `cortical/got/api.py`: GoTManager.create_task, get_task, update_task, delete_task
- `cortical/got/api.py`: TransactionContext.create_task, get_task, update_task (lines 611-664)
- `cortical/got/protocol.py`: GoTBackend protocol (interface definition)
- `scripts/got_utils.py`: TransactionalGoTAdapter.create_task, get_task, update_task, delete_task (lines 1629-1738)
- `scripts/got_utils.py`: GoTProjectManager.create_task, get_task, update_task_status, delete_task (lines 2507-2933)

**Assessment:** These are layers/adapters, not duplicates:
- GoTBackend: Protocol/interface
- GoTManager: Implementation using GoTBackend
- TransactionContext: Transactional wrapper around GoTManager
- TransactionalGoTAdapter: Adapter bridging old ThoughtGraph API to new GoT system
- GoTProjectManager: High-level project management using GoT

**Recommendation:** This is acceptable layering. Document the architecture to clarify relationships.

---

## 7. ThoughtGraph/ThoughtNode - NO DUPLICATION FOUND

**Checked:**
- `cortical/reasoning/thought_graph.py`: ThoughtGraph class (definitive implementation)
- `cortical/reasoning/graph_of_thought.py`: ThoughtNode, ThoughtEdge, NodeType, EdgeType
- `scripts/got_utils.py`: **IMPORTS** from cortical.reasoning (line 34)

**Result:** No duplication - scripts correctly imports from cortical.reasoning.

---

## Summary of Duplications

| Item | Severity | Files | Recommendation |
|------|----------|-------|----------------|
| ProcessLock | HIGH | cortical/got/tx_manager.py, scripts/got_utils.py | Consolidate to cortical/got/tx_manager.py |
| WAL implementations | HIGH | cortical/wal.py, cortical/got/wal.py, graph_persistence.py | Refactor got/wal.py to use cortical.wal |
| ID generation | MEDIUM | cortical/got/api.py, scripts/got_utils.py | Move to cortical/got/api.py, import |
| Checksum functions | MEDIUM | 5 locations | Extract to cortical/got/checksums.py |
| Conflict classes | LOW | Multiple files | Domain-specific, document relationships |
| Task methods | LOW | Multiple files | Acceptable layering |

---

## Recommendations

### Immediate (High Priority):
1. **Consolidate ProcessLock**: Remove duplication by having scripts/got_utils.py import from cortical/got/tx_manager.py
2. **Unify WAL**: Refactor cortical/got/wal.py to extend cortical.wal instead of reimplementing

### Short-term (Medium Priority):
3. **Centralize ID generation**: Move to cortical/got/api.py, standardize format
4. **Extract checksum utility**: Single implementation in cortical/got/checksums.py

### Long-term (Low Priority):
5. **Document architecture**: Clarify the relationship between task management layers
6. **Consider shared Conflict base**: If patterns converge across domains

---

## Files Analyzed

**cortical/got/** (14 files, ~3,200 lines):
- versioned_store.py, api.py, wal.py, errors.py, conflict.py
- checksums.py, tx_manager.py, config.py, recovery.py, types.py
- sync.py, protocol.py, transaction.py, __init__.py

**cortical/reasoning/** (18 files, ~20,561 lines):
- graph_persistence.py (2,017 lines)
- thought_graph.py, graph_of_thought.py, cognitive_loop.py
- collaboration.py, verification.py, crisis_manager.py
- + 11 other files

**scripts/**:
- got_utils.py (5,319 lines)

**cortical/**:
- wal.py (720 lines)

---

*Analysis Date: 2025-12-22*
*Total Files Reviewed: 34*
*Total Lines: ~30,000+*
