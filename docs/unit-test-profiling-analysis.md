# Unit Test Profiling Analysis

**Date:** 2025-12-26
**Branch:** `claude/profile-unit-tests-WUbZi`
**Updated:** After merging latest from main (includes concurrent test fixes)

## Executive Summary

Profiled 8,913 unit tests completing in **101.15s** (average ~11ms per test). Overall test performance is excellent. All tests now pass after recent fixes for concurrent test race conditions. However, several tests are misplaced and should be moved to more appropriate test categories.

## Test Suite Overview

| Metric | Value |
|--------|-------|
| Total tests | 8,913 |
| Total time | 101.15s |
| Average per test | ~11ms |
| Failed tests | 0 ✅ |
| Skipped tests | 2 (optional dependencies) |
| Largest test files | `test_graph_persistence.py` (3,052 lines), `test_moe_foundation.py` (3,005 lines) |

## Slow Tests Identified

### 1. Performance Tests in Unit Tests (SHOULD MOVE)

**Location:** `tests/unit/got/test_query_builder.py::TestQueryPerformance`

| Test | Setup Time | Issue |
|------|------------|-------|
| `test_explain_query` | 3.72s | Creates 100 tasks in fixture |
| `test_query_with_index_hint` | 3.66s | Creates 100 tasks in fixture |
| `test_lazy_iteration` | 3.50s | Creates 100 tasks in fixture |

**Problem:** These tests use a `large_manager` fixture that creates 100 tasks, which is appropriate for performance testing but not unit testing.

**Recommendation:** Move `TestQueryPerformance` class to `tests/performance/test_got_query_perf.py` (which already exists and could be extended).

---

### 2. Integration Tests in Unit Tests (SHOULD MOVE)

**a) File I/O Integration**

| Test | Time | Issue |
|------|------|-------|
| `tests/unit/test_prism_slm.py::TestIntegration::test_train_on_corpus_files` | 1.57s | Reads actual files from `samples/` directory |

**Problem:** This test reads real corpus files, making it an integration test, not a unit test.

**Recommendation:** Move to `tests/integration/` or mock the file reading.

---

**b) Subprocess Tests**

| Test | Time | Issue |
|------|------|-------|
| `tests/unit/test_process_lock.py::TestMultiProcessSafety::test_subprocess_respects_lock` | 0.78s | Spawns actual subprocess |

**Problem:** Subprocess spawning is an integration concern.

**Recommendation:** Move to `tests/integration/` or mark with `@pytest.mark.integration`.

---

### 3. Timing-Dependent Tests (SHOULD MARK AS SLOW)

These tests use `time.sleep()` and test timing behavior:

| Test | Time | Sleep Duration |
|------|------|----------------|
| `test_scheduler_runs_consolidation` | 1.50s | `time.sleep(1.5)` |
| `test_commit_on_save_debounced_with_auto_push` | 0.70s | `time.sleep(0.7)` |
| `test_cleanup_cancels_timer` | 0.50s | Timer-based |
| `test_valid_lock_not_removed` | 0.50s | `time.sleep()` |
| `test_exponential_backoff` | 0.50s | Timeout behavior |

**Problem:** These tests are inherently slow because they test time-based behavior.

**Recommendation:**
- Mark with `@pytest.mark.slow`
- Consider using `freezegun` or time mocking where possible
- Keep in unit tests but exclude from `make test-fast`

---

### 4. Large Fixture Setup Costs

Tests with expensive fixtures creating many entities:

| Fixture | Location | Entities Created | Setup Time |
|---------|----------|------------------|------------|
| `large_manager` | `test_query_builder.py` | 100 tasks | ~3.5s |
| Path finder fixtures | `test_path_finder.py` | Multiple tasks/edges | 0.5-0.9s |
| Query metrics fixtures | `test_query_metrics.py` | 10+ tasks | 0.4-0.5s |

**Recommendation:** These are appropriate if testing performance, but should use smaller datasets (10-20 entities) for unit tests.

---

### 5. Concurrent Tests (FIXED ✅)

The following tests were previously flaky but have been fixed in recent commits:

| Test | Status | Fix |
|------|--------|-----|
| `test_handles_concurrent_recovery_attempts` | ✅ Fixed | Added lock RuntimeError handling |
| `test_rapid_status_toggles` | ✅ Fixed | Improved thread synchronization |
| `test_parallel_index_reads_during_writes` | ✅ Fixed | Fixed race condition |

**Note:** These tests now pass reliably. They test concurrent behavior and are appropriately placed in unit tests since they verify thread-safety of core components.

---

## Recommendations Summary

### Move to `tests/performance/`

```
tests/unit/got/test_query_builder.py::TestQueryPerformance
  → tests/performance/test_got_query_perf.py (extend existing)
```

### Move to `tests/integration/`

```
tests/unit/test_prism_slm.py::TestIntegration::test_train_on_corpus_files
  → tests/integration/test_prism_slm_integration.py

tests/unit/test_process_lock.py::TestMultiProcessSafety
  → tests/integration/test_process_lock_integration.py
```

### Mark with `@pytest.mark.slow`

```python
# In tests/unit/test_consolidation.py
@pytest.mark.slow
def test_scheduler_runs_consolidation(self, hive, cortex):
    ...

# In tests/unit/test_graph_persistence.py
@pytest.mark.slow
def test_commit_on_save_debounced_with_auto_push(self, mock_run):
    ...
```

---

## Test Category Guidelines

For future reference, use these guidelines for test placement:

| Category | Criteria | Example |
|----------|----------|---------|
| **unit/** | Fast (<100ms), isolated, mocked dependencies | Testing a single function's logic |
| **integration/** | Multiple components, real I/O, subprocesses | Testing file persistence, subprocess communication |
| **performance/** | Large datasets, timing benchmarks | Testing query performance with 100+ entities |
| **behavioral/** | User workflows, quality metrics | Testing search result quality |
| **regression/** | Bug-specific reproduction | Preventing specific bug from recurring |

---

## Impact Analysis

Moving/marking these tests would:

1. **Speed up `make test-fast`** by ~12-15s (removing slow tests from default run)
2. **Improve CI reliability** by isolating flaky concurrent tests
3. **Better organize test categories** for clearer purpose
4. **Enable parallel test runs** by identifying tests that can't run concurrently

---

## Next Steps

1. [ ] Move `TestQueryPerformance` to `tests/performance/`
2. [ ] Move `test_train_on_corpus_files` to `tests/integration/`
3. [ ] Move `TestMultiProcessSafety` to `tests/integration/`
4. [ ] Add `@pytest.mark.slow` to timing-dependent tests
5. [x] ~~Fix thread safety bugs in concurrent tests~~ (Fixed in main)
6. [ ] Update `make test-fast` to exclude `slow` marker
