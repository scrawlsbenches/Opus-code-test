# Knowledge Transfer: Deprecated Class Removal (EventLog, GoTProjectManager)

**Date:** 2025-12-24
**Session:** claude/accept-handoff-next-session-ZjS2N
**Task:** T-20251222-145525-445df343

## Summary

Completed the migration from event-sourced backend to TX (transactional) backend by removing ~2,500 lines of deprecated code.

## What Was Removed

| Class | Lines | Purpose | Replacement |
|-------|-------|---------|-------------|
| `EventLog` | ~762 | Append-only event log persistence | TX entity files in `.got/entities/` |
| `HandoffManager` | ~114 | Handoff operations via EventLog | Methods in `TransactionalGoTAdapter` |
| `GoTProjectManager` | ~1,473 | Main API using event-sourced backend | `TransactionalGoTAdapter` |
| `TaskMigrator` | ~90 | Migration from file-based tasks | No longer needed |
| `atomic_append` | ~43 | Helper for EventLog | Not needed |

## Architecture Change

### Before (Event-Sourced)
```
User action → EventLog.log_event() → .got/events/session-xxx.jsonl
Startup → EventLog.rebuild_graph_from_events() → ThoughtGraph (slow)
```

### After (Transactional)
```
User action → GoTManager.create_task() → .got/entities/T-xxx.json
Startup → GoTManager.load() → Direct entity loading (fast)
```

## Files Modified

### Core Changes
- `scripts/got_utils.py` - Removed 4 classes, ~2,500 lines deleted

### CLI Updates (type hints)
- `cortical/got/cli/backup.py` - `GoTProjectManager` → `TransactionalGoTAdapter`
- `cortical/got/cli/decision.py`
- `cortical/got/cli/handoff.py`
- `cortical/got/cli/query.py`
- `cortical/got/cli/sprint.py`
- `cortical/got/cli/task.py`

### Deprecated Commands
- `cmd_migrate` - Now shows deprecation message
- `cmd_compact` - Now shows deprecation message

### Tests Removed
- `tests/regression/test_got_edge_rebuild.py` - Tested EventLog replay
- `tests/unit/test_got_query.py` - Tested GoTProjectManager (covered by behavioral tests)

### Tests Updated
- `tests/unit/test_cli_backup.py` - Updated TestCmdMigrate
- `tests/unit/test_cli_query.py` - Updated TestCmdCompact
- `tests/unit/test_process_lock.py` - Use GoTBackendFactory.create()

## Key Fixes Made During Migration

1. **`TransactionalGoTAdapter.block_task()`** - Fixed to set `blocked_reason` in properties and create BLOCKS edge

2. **`TransactionalGoTAdapter.get_task_dependencies()`** - Fixed to look for DEPENDS_ON edges instead of calling get_blockers()

3. **`TransactionalGoTAdapter.get_all_relationships()`** - Fixed edge type comparison (uppercase: "BLOCKS", "DEPENDS_ON")

4. **`TransactionalGoTAdapter.create_task()`** - Fixed to add CONTAINS edge when sprint_id provided

5. **API compatibility** - Changed `blocker_id` parameter to `blocked_by` in tests

## Test Results

- **8,764 tests passed**, 12 skipped
- All GoT CLI tests pass
- All behavioral workflow tests pass
- Reasoning demo works correctly

## What Still Works

- All `got` CLI commands (task, sprint, epic, decision, handoff, query, etc.)
- Handoff primitives (initiate, accept, complete, list)
- Query language ("what blocks X", "path from A to B", etc.)
- Dashboard and validation
- Reasoning framework (QAPV cycles, verification, parallel coordination)

## Important: Don't Reintroduce

1. **Event-sourced persistence** - TX backend is now the only option
2. **EventLog class** - Use GoTManager from `cortical/got/api.py`
3. **GoTProjectManager** - Use `GoTBackendFactory.create()` or `TransactionalGoTAdapter`
4. **Direct graph manipulation in tests** - Use public API methods

## Related Files

- `cortical/got/api.py` - GoTManager (TX backend implementation)
- `cortical/got/types.py` - Task, Decision, Edge types
- `scripts/got_utils.py` - TransactionalGoTAdapter, GoTBackendFactory

## Commits

- `4333dd08` - refactor(got): Remove deprecated EventLog and GoTProjectManager classes (~2,500 lines)
