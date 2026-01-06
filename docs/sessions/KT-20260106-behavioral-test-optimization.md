# Knowledge Transfer: Behavioral Test Optimization Session

**Date:** 2026-01-06
**Branch:** `claude/setup-pytest-coverage-x89p2`
**Session Focus:** Investigating behavioral test failures and design issues

---

## Executive Summary

This session investigated behavioral test failures revealed by the InMemoryFileSystem migration. The user explicitly asked to "slow down" and treat these as design breadcrumbs rather than just fixing tests with `use_memory=False`.

**Key Outcomes:**
1. Deleted obsolete `test_legacy_query_uses_expression_system.py` (redundant with `test_agent_uses_natural_query_expressions.py`)
2. Fixed `knowledge_transfer_stories.py` to use in-memory storage (removed file-checking assertions)
3. Fixed `delete_sprint()` - was using direct file operations, now uses transactional API
4. Fixed `get_decision()` - was using direct file operations, now uses transactional API
5. Fixed `CDGStore.__init__` - cache initialization order bug (cache must init before recovery)
6. Added `delete_sprint()` and `get_sprint_tasks()` methods to `TransactionContext`

---

## Changes Made (Uncommitted)

### 1. `cortical/cdg/storage.py`
**Cache initialization order fix:**
```python
# BEFORE (broken): Recovery called before cache initialized
self._version = self._load_version()
self._recover_pending_history()  # ← Calls read() which needs cache!
self._cache_enabled = cache_enabled  # ← Too late!

# AFTER (fixed): Cache initialized before recovery
self._version = self._load_version()
self._cache_enabled = cache_enabled  # ← Initialize first
self._cache: Dict[str, Entity] = {}
# ... other cache fields ...
self._recover_pending_history()  # ← Now safe to call
```

### 2. `cortical/got/api.py`

**a) `GoTManager.get_decision()` fix:**
```python
# BEFORE (broken with in-memory):
def get_decision(self, decision_id: str) -> Optional[Decision]:
    entities_dir = self.got_dir / "entities"
    decision_file = entities_dir / f"{decision_id}.json"
    if not decision_file.exists():  # ← Direct file check!
        return None
    # ...

# AFTER (works with in-memory):
def get_decision(self, decision_id: str) -> Optional[Decision]:
    with self.transaction(read_only=True) as tx:
        return tx.get_decision(decision_id)
```

**b) `GoTManager.delete_sprint()` fix:**
```python
# BEFORE (broken with in-memory):
def delete_sprint(self, sprint_id: str, force: bool = False) -> None:
    # ... validation ...
    sprint_file = entities_dir / f"{sprint_id}.json"
    if sprint_file.exists():
        sprint_file.unlink()  # ← Direct file deletion!

# AFTER (works with in-memory):
def delete_sprint(self, sprint_id: str, force: bool = False) -> None:
    with self.transaction() as tx:
        tx.delete_sprint(sprint_id, force=force)
```

**c) Added to `TransactionContext`:**
- `delete_sprint(sprint_id, force=False)` - Transactional sprint deletion
- `get_sprint_tasks(sprint_id)` - Get tasks in sprint via CONTAINS edges

### 3. `tests/behavioral/knowledge_transfer_stories.py`
- Changed fixture to use `use_memory=True`
- Removed file-checking assertions (lines 389-392, 428-432)
- Now verifies behavior through API return values, not file existence

### 4. Deleted: `tests/behavioral/test_legacy_query_uses_expression_system.py`
- Redundant with `test_agent_uses_natural_query_expressions.py`
- Had design flaw: created separate `TransactionalGoTAdapter` with its own manager

---

## Design Issues Identified

### Issue 1: Inconsistent API Patterns
Some `GoTManager` methods use transactional API (`get_task`, `get_sprint`), while others used direct file access (`get_decision` before fix, `delete_sprint` before fix).

**Pattern to follow:**
```python
# For read operations:
def get_X(self, x_id: str) -> Optional[X]:
    with self.transaction(read_only=True) as tx:
        return tx.get_X(x_id)

# For write operations:
def delete_X(self, x_id: str, force: bool = False) -> None:
    with self.transaction() as tx:
        tx.delete_X(x_id, force=force)
```

### Issue 2: Tests Checking Implementation Details
Several tests checked for file existence instead of using API methods:
```python
# BAD: Tests implementation detail
assert (entities_dir / f"{entity_id}.json").exists()

# GOOD: Tests behavior
assert manager.get_entity(entity_id) is not None
```

### Issue 3: Pre-existing Transaction Test Failures
Three tests fail consistently (pre-existing bugs, not from this session):
- `cdg_transaction_stories.py::test_scenario_conflict_causes_no_writes_to_persist`
- `cdg_transaction_stories.py::test_scenario_concurrent_conflicting_writes_one_wins`
- `test_got_transactional_behavioral.py::test_scenario_read_only_transaction_discards_writes`

These are transaction conflict resolution bugs unrelated to our changes.

---

## Current Test Status

**Behavioral tests run time:** ~2+ minutes (user reports too slow)

**Fixed failures:**
- `test_cdg_history_integrity.py` - 6 tests (cache init order)
- `test_cdg_crash_recovery.py` - 2 tests (cache init order)
- `test_graph_traversal_functions.py::test_scenario_type_of_returns_decision_for_decision` (get_decision fix)
- `test_developer_manages_sprints.py::test_scenario_force_delete_removes_sprint_and_edges` (delete_sprint fix)
- `knowledge_transfer_stories.py` - all 20 tests now pass with in-memory

**Pre-existing failures (not from this session):**
- 3 transaction conflict tests (see Issue 3 above)

---

## Files to Review

| File | Status | Notes |
|------|--------|-------|
| `cortical/cdg/storage.py` | Modified | Cache init order fix |
| `cortical/got/api.py` | Modified | get_decision, delete_sprint, TransactionContext methods |
| `tests/behavioral/knowledge_transfer_stories.py` | Modified | In-memory, removed file checks |
| `tests/behavioral/test_legacy_query_uses_expression_system.py` | DELETED | Obsolete |

---

## Next Steps

1. **Commit current changes** - The fixes are working but uncommitted
2. **Investigate slow tests** - Behavioral tests taking 2+ minutes, need profiling
3. **Fix pre-existing transaction bugs** - 3 failing tests in transaction conflict handling
4. **Audit other GoTManager methods** - Check for other direct file access patterns that should use transactions

---

## Commands to Verify State

```bash
# Check current changes
git status
git diff cortical/cdg/storage.py
git diff cortical/got/api.py

# Run quick verification
python -m pytest tests/behavioral/test_developer_manages_sprints.py -v --tb=short
python -m pytest tests/behavioral/knowledge_transfer_stories.py -v --tb=short
python -m pytest tests/behavioral/test_graph_traversal_functions.py::TestTypeOfFunction -v --tb=short

# Run crash recovery tests (verifies cache init fix)
python -m pytest tests/behavioral/test_cdg_crash_recovery.py tests/behavioral/test_cdg_history_integrity.py -v --tb=short

# Pre-existing failures (expected to fail)
python -m pytest tests/behavioral/cdg_transaction_stories.py::TestDeveloperCommitsAtomicChanges::test_scenario_conflict_causes_no_writes_to_persist -v
```

---

## Key Insight

The user's directive to "slow down" and investigate tests as design breadcrumbs was valuable. What appeared to be simple "add `use_memory=False`" fixes revealed:

1. **`delete_sprint` not using transactions** - Major bug that would have caused data loss
2. **`get_decision` using direct file access** - Broken with in-memory storage
3. **Test files checking implementation details** - Tests should verify behavior, not file existence
4. **Obsolete test file** - `test_legacy_query_uses_expression_system.py` was redundant

The pattern: **Tests that fail with in-memory storage often reveal real bugs in production code, not just test issues.**
