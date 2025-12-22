# GoT Architecture Forensics: Branch History Analysis

**Date:** 2025-12-22
**Analysis Type:** Git History & Architecture Investigation
**Focus:** cortical/got/ and cortical/reasoning/ evolution

---

## Executive Summary

The Graph-of-Thought (GoT) system has **two parallel implementations** that evolved over time:

1. **Event-sourced backend** (cortical.reasoning.thought_graph + .got/)
2. **Transactional backend** (cortical.got + .got-tx/)

Currently, the system uses the **event-sourced backend**. The transactional backend was developed as an alternative but is not actively deployed (.got-tx/ directory doesn't exist).

**Key Finding:** This is **NOT a merge conflict situation**. This is a deliberate architectural evolution with dual backend support and a factory pattern for backend selection.

---

## Timeline of Evolution

### Phase 1: Event-Sourced Foundation (Sprint 15)

**First Commit:** `933280c0` - "feat: Implement GoT project management CLI (Sprint 15)"

```
cortical/reasoning/
├── thought_graph.py        # Graph structure
├── graph_of_thought.py     # NodeType, EdgeType enums
└── graph_persistence.py    # WAL, snapshots

scripts/got_utils.py (initial)
├── Uses cortical.reasoning.thought_graph
├── Stores data in .got/ directory
└── Event-sourced with append-only event log
```

**Storage Location:** `.got/events/*.jsonl` (git-tracked event logs)

### Phase 2: Reasoning Framework Expansion

**Commits:** `ea1a33b8` through `c25ef650`

```
cortical/reasoning/ grew to include:
├── cognitive_loop.py       # QAPV cycle
├── verification.py         # Multi-level verification
├── crisis_manager.py       # Failure handling
├── collaboration.py        # Parallel agents
└── graph_persistence.py    # Enhanced WAL/snapshots
```

**Purpose:** GoT became part of a larger reasoning framework.

### Phase 3: Transactional Backend Created

**First Commit:** `2feccff2` - "feat(got): Implement ACID-compliant transaction layer"

```
cortical/got/                # NEW PACKAGE
├── __init__.py
├── transaction.py          # ACID transactions
├── tx_manager.py           # TransactionManager
├── versioned_store.py      # File-based storage
├── wal.py                  # Write-ahead log
├── types.py                # Task, Decision, Edge
├── errors.py               # GoTError hierarchy
└── checksums.py            # Data integrity
```

**Storage Location:** `.got-tx/` (NOT currently in use)

**Why Created:**
- ACID guarantees for concurrent access
- Conflict detection and resolution
- Multi-agent coordination
- Snapshot isolation

### Phase 4: Dual Backend Integration

**Key Commit:** `1c688af5` - "feat(got): Integrate transactional backend into got_utils.py"

```python
# scripts/got_utils.py gained:
from cortical.got.api import GoTManager as TxGoTManager

# Auto-detection logic
USE_TX_BACKEND = os.environ.get("GOT_USE_TX", "").lower() in ("1", "true", "yes")
if not USE_TX_BACKEND and TX_BACKEND_AVAILABLE:
    # Auto-detect: if .got-tx exists and has entities, use it
    USE_TX_BACKEND = (GOT_TX_DIR / "entities").exists()
```

**Result:** got_utils.py can use **either** backend.

### Phase 5: Migration Issues Discovery

**Commits with fixes:**
- `472c9342` - "fix(got): Fix edge migration - recover 301 edges from event log"
- `31900161` - "fix(got): Fix migration data loss and dashboard metrics"
- `3a8cdd94` - "fix(got): Fix edge loading in graph property for validate/dashboard"
- `dc359d8d` - "fix(got): Add decision migration and graph loading"

**Problem:** Migration from event-sourced → transactional had data integrity issues.

### Phase 6: Technical Debt Remediation

**Key Commit:** `54c1dd88` - "refactor(got): Address technical debt with protocol and factory patterns"

**Changes:**
1. **ID Format Standardization**
   - Old: `task:T-20251221-030434-3529`
   - New: `T-20251221-030434-3529`
   - Removed 40+ prefix manipulation calls

2. **GoTBackend Protocol** (`cortical/got/protocol.py`)
   - 23 methods defining backend contract
   - Both backends implement this protocol
   - Type-safe backend switching

3. **GoTBackendFactory** (`scripts/got_utils.py`)
   - Centralized backend creation
   - Auto-detection from env var or filesystem
   - New `--backend` CLI flag

---

## Current Architecture

### File Locations

| Component | Path | Lines | Purpose |
|-----------|------|-------|---------|
| CLI Wrapper | `scripts/got_utils.py` | 5,319 | Massive CLI with all commands |
| Event-Sourced Backend | `cortical/reasoning/thought_graph.py` | 41,038 | ThoughtGraph implementation |
| Transactional Backend | `cortical/got/api.py` | 747 | GoTManager API |
| Backend Protocol | `cortical/got/protocol.py` | 353 | Interface definition |

### Data Storage

| Backend | Directory | Status |
|---------|-----------|--------|
| Event-sourced | `.got/` | **ACTIVE** (exists on filesystem) |
| Transactional | `.got-tx/` | **INACTIVE** (doesn't exist) |

### Backend Selection Flow

```
┌─────────────────────────────────────────────────────────────┐
│ CLI Entry Point (scripts/got_utils.py main())               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ GoTBackendFactory.create()                                   │
│   1. Check GOT_BACKEND env var                              │
│   2. Check --backend CLI flag                               │
│   3. Auto-detect: if .got-tx/entities exists → transactional│
│   4. Default: event-sourced                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ GoTProjectManager    │    │ TransactionalGoT     │
│ (event-sourced)      │    │ Adapter              │
│                      │    │                      │
│ Uses:                │    │ Uses:                │
│ - ThoughtGraph       │    │ - TxGoTManager       │
│ - .got/events/       │    │ - .got-tx/entities/  │
│ - Append-only log    │    │ - ACID transactions  │
└──────────────────────┘    └──────────────────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
              Both implement GoTBackend Protocol
```

---

## Duplicate Code Analysis

### ID Generation (Duplicate)

**Location 1:** `scripts/got_utils.py:102-139`
```python
def generate_task_id() -> str:
    """Generate unique task ID: task:T-YYYYMMDD-HHMMSS-XXXX"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(2).hex()
    return f"task:T-{timestamp}-{suffix}"
```

**Location 2:** `cortical/got/api.py:40-52`
```python
def generate_task_id() -> str:
    """
    Generate unique task ID.
    Format: T-YYYYMMDD-HHMMSS-XXXXXXXX where XXXXXXXX is random hex.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4)  # 8 hex chars
    return f"T-{timestamp}-{random_suffix}"
```

**Differences:**
- got_utils.py adds `task:` prefix (legacy)
- cortical/got uses `secrets.token_hex(4)` (8 chars) vs `os.urandom(2).hex()` (4 chars)
- cortical/got uses UTC timezone

**Impact:** IDs generated by different backends are **incompatible** due to suffix length difference.

### Other Potential Duplication

**Backend Adapters:** got_utils.py has 5,319 lines including:
- TransactionalGoTAdapter class (~500 lines)
- GoTProjectManager class (~2,000 lines)
- CLI command handlers (~2,000 lines)
- Utility functions (~800 lines)

**Recommendation:** Extract shared logic into cortical.got.common module.

---

## Migration Issues Found

### 1. Edge Migration Data Loss
**Commit:** `472c9342`
**Problem:** 301 edges were lost during migration
**Root Cause:** Edge relation stored in `type` field, not `rel` field
**Fix:** Updated migration to use correct field name

### 2. Dashboard Metrics Issues
**Commit:** `31900161`
**Problem:** Node count didn't match dashboard (370 vs 389)
**Root Cause:** `decision.create` is separate event type from `node.create`
**Fix:** Include both event types when counting

### 3. Edge Loading in Dashboard
**Commit:** `3a8cdd94`
**Problem:** Edges not showing in dashboard
**Root Cause:** Graph property not loading edges
**Fix:** Fixed edge loading in graph property getter

### 4. Decision Migration
**Commit:** `dc359d8d`
**Problem:** Decisions not migrated properly
**Fix:** Added decision migration and graph loading support

---

## Evidence of Merge Conflicts?

**Search Results:**
```bash
git log --merges --oneline -- cortical/got/ cortical/reasoning/
# Result: (empty) - NO MERGE COMMITS
```

**Conclusion:** There were **NO merge conflicts** in GoT-related code.

The dual implementation is **intentional architectural design**, not the result of failed merges or parallel branches creating conflicts.

---

## Branch Activity Analysis

### Branches That Touched GoT Code

```bash
git log --all --oneline --graph --decorate -- cortical/got/ scripts/got_utils.py | head -80
```

**Pattern:** All commits are linear on feature branches, then merged to main via PR.

**No evidence of:**
- Force pushes (`git log` shows clean date ordering)
- History rewrites (no unusual gaps or date inversions)
- "Ours vs theirs" merge resolutions (no merge commits in the logs)
- Abandoned refactoring attempts (all commits are part of completed features)

---

## Relationship Between Components

### got_utils.py vs cortical/got/

| Aspect | got_utils.py | cortical/got/ |
|--------|--------------|---------------|
| **Creation Date** | Sprint 15 (earlier) | Dec 2025 (later) |
| **Original Backend** | cortical.reasoning.thought_graph | Self-contained |
| **Lines of Code** | 5,319 | 747 (api.py) |
| **Purpose** | CLI wrapper + business logic | Library API |
| **Storage** | .got/ (event-sourced) | .got-tx/ (transactional) |
| **Current Use** | **ACTIVE** | **INACTIVE** |

**Timeline:**
1. got_utils.py created first, using cortical.reasoning
2. cortical/got/ created later as standalone transactional backend
3. got_utils.py refactored to support **both** backends via adapter pattern

---

## Why Two Backends?

### Event-Sourced Backend (cortical.reasoning.thought_graph)

**Pros:**
- Git-friendly (append-only .jsonl files)
- Full audit trail
- Simple to understand and debug
- No locking needed (append-only)

**Cons:**
- No ACID guarantees
- Slow for large graphs (O(n) event replay)
- No conflict resolution
- Hard to query efficiently

### Transactional Backend (cortical.got)

**Pros:**
- ACID guarantees
- Snapshot isolation
- Conflict detection and resolution
- Optimized for concurrent access
- Checksums for data integrity

**Cons:**
- More complex implementation
- Binary storage (not as git-friendly)
- Requires lock management
- Heavier weight

**Decision:** Keep both, let users choose via `--backend` flag.

---

## Technical Debt Identified

### 1. ID Generation Duplication ✅ CRITICAL

**Problem:** Two different implementations with incompatible formats.

**Impact:**
- IDs from event-sourced backend: 4-char suffix
- IDs from transactional backend: 8-char suffix
- Migration between backends may fail

**Recommendation:**
```python
# Move to cortical.got.id_generation
def generate_task_id() -> str:
    """Canonical task ID generator."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4)  # 8 hex chars
    return f"T-{timestamp}-{random_suffix}"
```

### 2. Massive got_utils.py ✅ HIGH

**Problem:** 5,319 lines is too large for maintainability.

**Recommendation:** Split into:
- `scripts/got_cli.py` - CLI entry point (~500 lines)
- `cortical/got/adapters.py` - Backend adapters (~1,000 lines)
- `cortical/got/commands.py` - Command handlers (~2,000 lines)
- `cortical/got/queries.py` - Query language (~500 lines)

### 3. Backend Auto-Detection Logic ⚠️ MEDIUM

**Problem:** Auto-detection based on filesystem state is fragile.

```python
# Current logic
USE_TX_BACKEND = (GOT_TX_DIR / "entities").exists()
```

**Issue:** What if .got-tx/ exists but is corrupted or incomplete?

**Recommendation:** Add `.got-backend` marker file:
```json
{
  "backend": "event-sourced",
  "version": "1.0.0",
  "created": "2025-12-22T00:00:00Z"
}
```

### 4. Migration Testing Gaps ⚠️ MEDIUM

**Evidence:** Multiple fix commits for migration issues (472c9342, 31900161, 3a8cdd94)

**Recommendation:** Add comprehensive migration test suite:
- Round-trip migration (event → tx → event)
- Large graph migration (1000+ nodes)
- Edge case handling (orphaned nodes, circular deps)
- Data integrity verification

---

## Recommendations

### Immediate Actions

1. **Consolidate ID Generation** (1-2 hours)
   - Create `cortical/got/id_generation.py`
   - Move canonical implementation there
   - Update both backends to use it
   - Add tests for ID format consistency

2. **Add Backend Marker File** (30 minutes)
   - Create `.got-backend` file on initialization
   - Check marker before auto-detection
   - Prevent silent backend switching

3. **Document Backend Selection** (1 hour)
   - Update CLAUDE.md with backend comparison table
   - Add troubleshooting guide for backend selection
   - Document migration process

### Medium-Term Refactoring

4. **Split got_utils.py** (4-8 hours)
   - Extract adapters to cortical/got/adapters.py
   - Extract commands to cortical/got/commands.py
   - Keep CLI entry point minimal
   - Update imports across codebase

5. **Add Migration Test Suite** (4-6 hours)
   - Comprehensive round-trip tests
   - Large graph tests (1000+ nodes)
   - Edge case coverage
   - Performance benchmarks

### Long-Term Considerations

6. **Choose Primary Backend** (Research + Decision)
   - Evaluate production usage patterns
   - Measure performance differences
   - Survey user preferences
   - Make a strategic decision on default backend

7. **Deprecation Plan** (If needed)
   - If one backend is clearly superior, deprecate the other
   - Provide migration tools
   - Maintain backward compatibility for 2-3 releases

---

## Summary

**What This Analysis Found:**

1. ✅ **NO merge conflicts** - Clean git history
2. ✅ **Intentional dual backend** - Not a bug, it's a feature
3. ✅ **Protocol-based abstraction** - Well-designed factory pattern
4. ⚠️ **ID generation duplication** - Critical to fix
5. ⚠️ **Migration data loss issues** - Already fixed, but need tests
6. ⚠️ **got_utils.py is massive** - Needs refactoring

**Current Status:**
- Event-sourced backend is **ACTIVE** and working
- Transactional backend is **INACTIVE** but fully implemented
- Backend switching works via `--backend` flag
- Migration has had issues but fixes are in place

**Next Steps:**
1. Fix ID generation duplication
2. Add backend marker file
3. Improve migration test coverage
4. Consider refactoring got_utils.py

---

**Analysis completed:** 2025-12-22
**Git history reviewed:** 450+ commits
**Files analyzed:** 15+ files across cortical/got/, cortical/reasoning/, scripts/
