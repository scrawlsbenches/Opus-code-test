# Knowledge Transfer: Test Performance Infrastructure Improvements

**Date:** 2025-12-27
**Session:** Test Infrastructure Optimization
**Branch:** `claude/setup-handoff-validation-SltvC`

## Summary

This session identified and fixed critical test performance issues that were wasting developer time. The root cause was function-scoped fixtures recreating expensive GoTManager instances for each test, plus no auto-exclusion of performance tests from default runs.

## Key Findings

### Problem 1: Fixture Recreation Waste (75s wasted)

`tests/performance/test_got_query_perf.py` had a function-scoped fixture that created 100 tasks with disk I/O for **each of 14 tests**:

```
Before: 14 tests × 5s setup each = 75s just in fixture setup
After:  1 class-scoped fixture × 5s = 5s total
```

### Problem 2: Performance Tests Not Excluded (143 tests, ~90s)

The `pyproject.toml` had `addopts = "-m 'not optional and not slow'"` but performance tests weren't marked `@slow`. Only 12 tests had `@pytest.mark.slow`.

### Problem 3: No Shared GoT Fixtures

Every test creating a GoTManager did it from scratch. No reusable fixtures existed in `conftest.py`.

### Problem 4: No Documentation

CLAUDE.md had no guidance on fixture scope or test performance budgets.

## Changes Made

### 1. Added Shared GoT Fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def fresh_got_manager(tmp_path):
    """Function-scoped for tests that modify state."""

@pytest.fixture(scope="class")
def got_manager_with_sample_tasks(tmp_path_factory):
    """Class-scoped with 20 tasks - shared across class."""

@pytest.fixture(scope="class")
def got_manager_large(tmp_path_factory):
    """Class-scoped with 100 tasks for perf tests."""
```

### 2. Auto-Mark Performance Tests as Slow (`tests/conftest.py`)

```python
elif '/performance/' in test_path:
    item.add_marker(pytest.mark.performance)
    item.add_marker(pytest.mark.slow)  # NEW: Excluded by default
```

### 3. Refactored Performance Tests

- `test_got_query_perf.py`: Changed to use `got_manager_large` (class-scoped)
- `test_query_api_perf.py`: Changed to use `got_manager_large` (class-scoped)

### 4. Added CLAUDE.md Guidelines

New sections added:
- **GoT Fixtures table** with scope documentation
- **Test Performance Guidelines** with budgets
- **Fixture Scope Decision Tree** (ASCII art)
- **Common Mistakes table** with fixes

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Performance test setup | ~75s | ~15s | **60s saved** |
| Default dev run | 5:47 | 4:17 | **90s saved** |
| Full suite | 7:06 | 5:44 | **82s saved** |

### Developer Workflow Now

| Task | Command | Time |
|------|---------|------|
| Quick sanity | `make test-smoke` | **3s** |
| After changes | `pytest tests/unit/ tests/smoke/` | **1:30** |
| Default run | `pytest tests/` | **4:17** |
| Full suite | `pytest tests/ -m ""` | **5:44** |

## Root Cause Analysis

**Why did this keep happening?**

1. No performance budget documented
2. No shared fixtures to reuse
3. Copy-paste propagated bad patterns
4. No timing check in dev workflow

**The fix isn't just code - it's documentation.** The CLAUDE.md decision tree now makes the right choice obvious.

## Commits

1. `5a6ee58c` - perf(tests): Add shared GoT fixtures and test performance guidelines
2. `76c3795c` - perf(tests): Auto-mark performance tests as slow

## Future Work

1. **Parallel test execution** (`pytest -n auto`) - could halve times
2. **Mark behavioral tests slow** - 200+ tests use `shared_processor`
3. **Test sharding for CI** - split across workers

## Key Lessons for Future Agents

1. **Always check fixture scope** - function scope recreates per test
2. **Use `--durations=10`** before committing new tests
3. **Check `tests/conftest.py`** for shared fixtures before creating new ones
4. **Performance tests should be auto-excluded** from dev runs
5. **Document patterns** in CLAUDE.md to prevent regression

## Files Modified

- `tests/conftest.py` - Added GoT fixtures, auto-mark performance as slow
- `tests/performance/test_got_query_perf.py` - Use class-scoped fixtures
- `tests/performance/test_query_api_perf.py` - Use class-scoped fixtures
- `CLAUDE.md` - Added Test Performance Guidelines section

## Validation

Handoff `H-20251227-113751-b59fd87e` was completed with:
- 139 new Query API tests passed
- 10,280 total tests passed
- 0 regressions
- Performance stable across 3 runs
