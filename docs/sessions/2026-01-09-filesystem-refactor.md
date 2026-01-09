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

## Remaining Work

Update test files using old API:

### Files needing `InMemoryFileSystem(base_dir)`:
- [ ] tests/fixtures/test_bootstrap.py
- [ ] tests/behavioral/test_cdg_schema_stories.py
- [ ] tests/behavioral/test_audit_reasoning_stories.py
- [ ] tests/behavioral/test_cdg_store_caching_stories.py

### Files needing `CDGStore(filesystem=...)`:
- [ ] tests/unit/cdg/test_cdg_durability.py
- [ ] tests/behavioral/cdg_recovery_stories.py (~15 usages)
- [ ] tests/behavioral/test_got_transactional_behavioral.py
- [ ] tests/behavioral/test_cdg_schema_stories.py

### Other potential usages:
- [ ] CDGTransactionManager old API usages
- [ ] CDGRecoveryManager old API usages

## Design Decisions

1. **entity_factory is Optional** - Has `default_entity_factory` fallback
2. **index_manager is Optional** - Indexing is skipped when None

## Original Issue

Started from investigating `test_scenario_cdg_transaction_manager_requires_injection` which led to discovering the architectural flaw: CDGStore/CDGTransactionManager were taking both `store_dir` AND `filesystem` separately, violating single responsibility.

## Recovery Instructions

If context is lost:
1. Run `python -m pytest tests/smoke/ -v` - should pass (34/34)
2. Run `python -m pytest tests/behavioral/ -v --tb=no | grep -E "FAILED|ERROR"` - shows remaining work
3. Fix tests by updating to new API: `InMemoryFileSystem(base_dir)` and `CDGStore(filesystem=fs)`
