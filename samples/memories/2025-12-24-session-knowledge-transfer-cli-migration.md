# Knowledge Transfer: CLI Migration to TX Backend (2025-12-24)

**Session ID:** claude/accept-handoff-next-session-ZjS2N
**Previous Session:** claude/got-handoff-command-0tOum (via handoff H-20251224-055528-be8b)

---

## Executive Summary

This session completed the CLI migration from EventLog/HandoffManager to TX backend manager methods, unblocking task T-20251222-145525-445df343 (Remove deprecated EventLog and GoTProjectManager classes).

---

## What Was Done

### 1. CLI Handoff Commands Migration

**File:** `cortical/got/cli/handoff.py`

Updated all 4 handoff command handlers to use TX backend manager methods:

| Function | Old Pattern | New Pattern |
|----------|-------------|-------------|
| `cmd_handoff_initiate` | `HandoffManager(EventLog(...)).initiate_handoff()` | `manager.initiate_handoff()` |
| `cmd_handoff_accept` | `HandoffManager(...).accept_handoff()` | `manager.accept_handoff()` |
| `cmd_handoff_complete` | `HandoffManager(...).complete_handoff()` | `manager.complete_handoff()` |
| `cmd_handoff_list` | `HandoffManager.load_handoffs_from_events()` | `manager.list_handoffs()` |

### 2. Validation Command Fix

**File:** `cortical/got/cli/query.py`

- **Removed false edge discrepancy check** - Was comparing event log edge count vs TX entity edge count, producing false positives
- **Added entity file counting** - Now directly counts `.got/entities/{T,E,D,H}-*.json` files
- **Fixed status sorting** - Added `str()` conversion to handle mocked properties

### 3. Test Updates

| File | Changes |
|------|---------|
| `tests/unit/test_cli_handoff.py` | Completely rewritten with TX backend mocks |
| `tests/unit/test_got_cli.py` | Updated mock_manager fixture to use MagicMock instead of spec=GoTProjectManager; added handoff method defaults |
| `tests/unit/test_cli_query.py` | Removed edge discrepancy tests; updated validate tests to mock got_dir |

### 4. Task Status Update

**Task T-20251222-145525-445df343** (Remove deprecated EventLog and GoTProjectManager classes):
- Status changed from `blocked` to `in_progress`
- Ready for the actual code removal (~2,200 lines)

---

## Test Results

```
8,893 passed, 10 skipped, 17 deselected
Coverage: 88% (close to 89% baseline)
```

All handoff tests pass (30 total across two test files).

---

## Key Technical Details

### Mock Fixture Pattern for TX Backend

When testing CLI commands that use TX backend methods, use this pattern:

```python
@pytest.fixture
def mock_manager(temp_got_dir):
    """Create a mock with TX backend methods."""
    manager = MagicMock()  # NOT Mock(spec=GoTProjectManager)
    manager.got_dir = temp_got_dir

    # TX backend handoff methods
    manager.initiate_handoff.return_value = "H-20251220-120000-abc123"
    manager.accept_handoff.return_value = True
    manager.complete_handoff.return_value = True
    manager.list_handoffs.return_value = []

    return manager
```

### Why Not Use spec=GoTProjectManager

The `GoTProjectManager` class doesn't have the new handoff methods - those are on `TransactionalGoTAdapter`. Using `spec=GoTProjectManager` would cause `AttributeError` when accessing `manager.initiate_handoff`.

---

## Files Modified

| File | Lines Changed |
|------|---------------|
| `cortical/got/cli/handoff.py` | +42/-16 |
| `cortical/got/cli/query.py` | +68/-52 |
| `scripts/got_utils.py` | +91/-72 |
| `tests/unit/test_cli_handoff.py` | +220/-395 |
| `tests/unit/test_cli_query.py` | +75/-75 |
| `tests/unit/test_got_cli.py` | +80/-72 |

---

## Commit

```
3a8e329a refactor(got): Migrate CLI commands to use TX backend manager methods
```

Pushed to: `origin/claude/accept-handoff-next-session-ZjS2N`

---

## Next Steps for T-20251222-145525-445df343

The task to remove deprecated code is now unblocked. Here's the recommended approach:

1. **Remove HandoffManager** (~114 lines in `scripts/got_utils.py`)
   - Already unused by CLI commands

2. **Remove EventLog** (~762 lines in `scripts/got_utils.py`)
   - Check for any remaining usages first
   - `cmd_compact` and `cmd_migrate_events` may still reference it

3. **Remove GoTProjectManager** (~1,473 lines in `scripts/got_utils.py`)
   - Replace with TransactionalGoTAdapter where needed
   - Update tests that still use it

4. **Update remaining tests**
   - `tests/behavioral/test_got_workflow.py`
   - `tests/regression/test_got_edge_rebuild.py`

---

## Lessons Learned

1. **Check which file the CLI actually uses** - The CLI routes through `cortical/got/cli/*.py` modules, not the handlers in `scripts/got_utils.py`

2. **Don't use spec= with evolving interfaces** - When a mock needs to support methods from multiple classes, use `MagicMock()` without spec

3. **Validate tests after modifying validation logic** - Tests that check for specific output strings will fail if the output format changes

---

## Handoff Ready

This session is ready for handoff. The next agent should:
1. Review this document
2. Continue with T-20251222-145525-445df343 (remove deprecated code)
3. Or work on other tasks as directed by the user

---

**Related Decisions:**
- D-20251224-052658-40924db3 (Migration gap audit from previous session)

**Related Handoffs:**
- H-20251224-055528-be8b (accepted at session start)
