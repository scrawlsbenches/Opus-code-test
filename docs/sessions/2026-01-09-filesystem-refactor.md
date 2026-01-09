# Session Scratchpad: FileSystem Refactoring

**Date:** 2026-01-09
**Branch:** claude/review-previous-work-aqvaF
**Handoff:** H-20260109-101844-be46a87c (accepted)

## SESSION OVERRIDES

None currently.

## Current Task

Refactoring CDG to make FileSystem a required first-class dependency that encapsulates `base_dir`.

## Completed

1. Added `base_dir` property to FileSystem Protocol
2. Updated `RealFileSystem(base_dir)` - required parameter
3. Updated `InMemoryFileSystem(base_dir)` - required parameter
4. Updated `CDGStore(filesystem)` - filesystem is first required parameter
5. Updated `CDGTransactionManager(filesystem)` - same
6. Updated `CDGRecoveryManager(filesystem)` - same
7. Updated `bootstrap.py` to create FileSystem with `entities_dir`
8. Updated `got_module.py` and `cdg_module.py` to use new API
9. Fixed smoke tests (34/34 pass)

## COMPLETED - All Files Updated

### Session 1 - Updated:
- [x] tests/fixtures/test_bootstrap.py
- [x] tests/behavioral/test_cdg_schema_stories.py
- [x] tests/behavioral/test_audit_reasoning_stories.py
- [x] tests/behavioral/test_cdg_store_caching_stories.py
- [x] tests/unit/cdg/test_cdg_durability.py
- [x] tests/behavioral/cdg_recovery_stories.py
- [x] tests/behavioral/test_got_transactional_behavioral.py

### Session 2 - Updated:
- [x] tests/behavioral/cdg_transaction_stories.py (2 usages)
- [x] tests/behavioral/test_cdg_crash_recovery.py (all CDGTransactionManager, CDGStore, CDGRecoveryManager usages)
- [x] tests/performance/contracts/test_cdg_contract.py (all CDGTransactionManager, CDGRecoveryManager usages)
- [x] tests/behavioral/test_container_di_stories.py (updated test expectation for new API)
- [x] tests/behavioral/test_cdg_history_integrity.py (all CDGStore usages)
- [x] cortical/core/modules/cdg_module.py (fixed duplicate filesystem argument)

## Design Discussions (Pending)

### Are entity_factory and index_manager needed?

**Question**: As database designers, do we actually need these classes based on:
1. What the code is actually doing
2. What is needed for the system to function well
3. Theoretical database design principles

**To investigate when we return to this**:
- What does entity_factory actually do? Is it just JSON deserialization?
- What does index_manager do? Is it essential for core storage operations?
- Could these be handled differently (e.g., schema-based, convention-based)?
- Are they adding unnecessary complexity?

### Test Pattern Concerns

**Issue raised**: Tests may be using `RealFileSystem` when they should use:
- The bootstrap container with child containers for test isolation
- `InMemoryFileSystem` for faster tests
- Existing conftest.py fixtures

**To investigate**: Review test bootstrap pattern before making more changes.

## Design Decisions Made

1. **entity_factory is Optional** - Has `default_entity_factory` fallback
2. **index_manager is Optional** - Indexing is skipped when None

## Original Issue

Started from investigating `test_scenario_cdg_transaction_manager_requires_injection` which led to discovering the architectural flaw: CDGStore/CDGTransactionManager were taking both `store_dir` AND `filesystem` separately, violating single responsibility.

## Recovery Instructions

If context is lost:
1. Run `python -m pytest tests/smoke/ -v` - should pass (34/34)
2. Run `python -m pytest tests/behavioral/ -v --tb=no | grep -E "FAILED|ERROR"` - shows remaining work
3. Fix tests by updating to new API: `InMemoryFileSystem(base_dir)` and `CDGStore(filesystem=fs)`
