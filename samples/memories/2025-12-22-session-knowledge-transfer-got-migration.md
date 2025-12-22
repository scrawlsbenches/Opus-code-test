# Knowledge Transfer: GoT TX Backend Migration Completion

**Date:** 2025-12-22
**Session ID:** ROOCK
**Branch:** claude/add-state-management-ROOCK
**Tags:** `got`, `migration`, `tx-backend`, `decisions`, `velocity-metrics`, `edge-types`

## Summary

This session completed the decision/why functionality for the TX backend and identified critical bugs in edge loading and velocity metrics.

## Key Accomplishments

### 1. Decision/Why Functionality Added ✅
- `log_decision()` method with full compatibility (decision, rationale, affects, alternatives, context)
- `why(task_id)` method to query decisions affecting a task
- Helper methods: `create_decision()`, `list_decisions()`, `get_decisions_for_task()`
- **11 comprehensive unit tests** in `tests/unit/got/test_tx_adapter.py`
- Creates JUSTIFIES edges to affected entities

### 2. GoTBackendFactory Simplified ✅
- Factory now always returns `TransactionalGoTAdapter`
- Event-sourced backend option removed
- ~28 lines simplified

### 3. Sprint S-018 Created with Investigation Tasks
Created Sprint "GoT Cleanup & Observability" with tasks:
- T-20251222-145440-7fb36a5a: Fix EdgeType enum (HIGH)
- T-20251222-145502-8edcf3a0: Fix velocity metrics (HIGH)
- T-20251222-145525-445df343: Remove deprecated classes (MEDIUM)

## Critical Bugs Discovered

### Bug 1: Orphan Nodes (100% orphans, 0 edges displayed)

**Root Cause:** EdgeType enum in `cortical/reasoning/graph_of_thought.py` is missing:
- `JUSTIFIES` (used by decision→task edges)
- `PART_OF` (used by sprint→epic edges)

**Location:** `scripts/got_utils.py` line 1295:
```python
edge_type=EdgeType[edge_type_str]  # KeyError silently caught
```

**Impact:** 16 edges exist in storage but none load into the graph.

**Fix:** Add missing edge types to EdgeType enum.

### Bug 2: Velocity Metrics Always 0

**Root Cause:** `completed_at` timestamp never set when tasks are completed.

**Location:** `scripts/got_dashboard.py` lines 258-259 expects:
```python
completed_at_str = task.metadata.get("completed_at") or task.properties.get("completed_at")
```

But `complete_task()` doesn't set this field.

**Fix:** Update `TransactionalGoTAdapter.complete_task()` or `TxGoTManager.update_task()` to set `completed_at` when status changes to "completed".

## Files Modified

| File | Changes |
|------|---------|
| `scripts/got_utils.py` | +131 lines: decision methods, factory simplification |
| `tests/unit/got/test_tx_adapter.py` | +179 lines: 11 decision tests |

## Commits Pushed

```
f56ad163 feat(got): Add decision/why methods to TransactionalGoTAdapter
19750842 chore: Update GoT entities with cleanup decisions
71e97f53 refactor(got): Simplify GoTBackendFactory to transactional-only
```

## Deferred Work

### Bulk Cleanup (~2200 lines)
The removal of EventLog and GoTProjectManager was deferred because:
- Tight coupling with 6+ test files
- CLI commands reference EventLog directly
- HandoffManager depends on EventLog
- Recommended: Fix edge types first (dependency)

### Cleanup Order (Recommended)
1. Fix EdgeType enum (enables edge loading)
2. Fix velocity metrics (enables observability)
3. Update CLI commands to not use EventLog
4. Remove: HandoffManager → EventLog → GoTProjectManager
5. Update/delete obsolete tests

## Test Status

- ✅ 343 smoke/unit tests pass
- ✅ 26 TX adapter tests pass (15 sprint + 11 decision)
- ⚠️ Edge loading silently fails (edges exist but don't load)

## Architecture Notes

### TX Backend Storage
```
.got/entities/
├── T-*.json          # Tasks (15 files)
├── D-*.json          # Decisions (2 files)
├── E-*.json          # Edges (16 files) ← NOT LOADING!
├── S-*.json          # Sprints (21 files)
└── _version.json     # Global version
```

### Decision Flow
```
log_decision() 
  → create_decision() in TxGoTManager
  → save Decision entity
  → create JUSTIFIES edge to affected tasks
  → edge saved in E-*.json
```

## Commands for Next Session

```bash
# Check current state
python scripts/got_utils.py dashboard
python scripts/got_utils.py sprint status

# Verify edge loading issue
python -c "from scripts.got_utils import TransactionalGoTAdapter; a=TransactionalGoTAdapter('.got'); print(f'Edges loaded: {len(a.graph.edges)}')"

# List edge files
ls .got/entities/E-*.json | wc -l  # Should show 16+

# Run tests
python -m pytest tests/unit/got/ -v
```

## Related Documentation
- `docs/graph-of-thought.md` - GoT framework
- `docs/graph-recovery-procedures.md` - Recovery guide
- Task T-20251222-145525-445df343 has full cleanup notes
