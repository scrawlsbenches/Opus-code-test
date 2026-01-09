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

**Root Cause - CLAUDE.md Gap:**

The "Available Fixtures" section (lines 1385-1392) only documents CorticalTextProcessor fixtures:
- `small_processor`, `shared_processor`, `fresh_processor`, `small_corpus_docs`

**MISSING from CLAUDE.md** - The GoT/CDG fixtures that exist in conftest.py:
- `tmp_got_dir` - temporary directory
- `fresh_tx_manager` / `memory_tx_manager` - TransactionManager via container
- `fresh_got_manager` / `memory_got_manager` - GoTManager via container
- `memory_container` - full container access
- Helper functions: `_create_tx_manager()`, `_create_got_manager()`, `_create_container()`

The "Search Before Creating" section mentions conftest.py but doesn't explicitly say:
> "When writing CDG/GoT tests, use fixtures. DO NOT directly instantiate CDGStore, CDGTransactionManager."

**Action needed**: Update CLAUDE.md to document GoT/CDG fixtures explicitly.

### CLAUDE.md Restructuring Analysis

**User Question**: Is CLAUDE.md overwhelming with too much detail?

**Answer**: Yes. The paradox is more detail ≠ better guidance. Information overload causes skimming, which is why the GoT/CDG fixture documentation was missed even though conftest.py itself has excellent comments.

**Can it be split into multiple documents?**

Yes. Proposed structure:

| Document | Content | When Loaded |
|----------|---------|-------------|
| `CLAUDE.md` | Core principles, workflow, red flags | Always (session start) |
| `docs/fixtures.md` | Test fixtures by component | Read when writing tests |
| `docs/cli-reference.md` | GoT CLI commands | Read when using CLI |
| `docs/architecture.md` | Seven pillars, module details | Read when exploring |

**The critical missing principle** (add to CLAUDE.md):
> "When writing tests for any component, FIRST check `tests/conftest.py` for existing fixtures. DO NOT directly instantiate managers, stores, or transaction managers."

**Note**: conftest.py (lines 166-181) already has excellent documentation:
```
# BREAKING CHANGE (2026-01-04): TransactionManager and GoTManager now require
# dependency injection. Direct instantiation is prohibited.
#
# USE THESE FIXTURES - DO NOT CREATE MANAGERS DIRECTLY IN TESTS
```

The issue isn't missing documentation—it's that CLAUDE.md doesn't point to where the documentation lives.

### Script to Fix Tests

**TODO**: Write a script using git history to:
1. Find my changes that used `RealFileSystem` directly
2. Revert those specific patterns
3. Replace with proper fixture usage (`fresh_tx_manager`, `_create_container()`, etc.)

This should be done one test file at a time, verifying each works.

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
