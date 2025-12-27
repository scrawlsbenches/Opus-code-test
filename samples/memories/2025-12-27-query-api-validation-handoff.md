# Handoff: Query API Test Validation

**From:** Session `claude/evaluate-query-language-usability-BUe5R`
**Date:** 2025-12-27
**Status:** Ready for validation
**Priority:** High - these tests protect the Query API

---

## What Was Done

Added 139 tests across 6 files to protect the Query API. Also fixed test reliability issues and updated CLAUDE.md with common mistakes.

### Files Created
```
tests/unit/got/test_pattern_matcher.py          # 36 tests
tests/unit/got/test_query_api_contracts.py      # 38 tests
tests/unit/got/test_query_api_invariants.py     # 28 tests
tests/behavioral/test_query_api_workflows.py    # 15 tests
tests/performance/test_query_api_perf.py        # 12 tests
tests/integration/test_query_api_integration.py # 10 tests
```

### Files Modified
- `CLAUDE.md` - Added 4 "Common Mistakes" entries for Query API
- `samples/memories/2025-12-27-query-api-testing-knowledge-transfer.md` - Knowledge transfer

---

## Validation Checklist

### 1. Run All New Tests
```bash
python -m pytest tests/unit/got/test_pattern_matcher.py \
  tests/unit/got/test_query_api_contracts.py \
  tests/unit/got/test_query_api_invariants.py \
  tests/behavioral/test_query_api_workflows.py \
  tests/performance/test_query_api_perf.py \
  tests/integration/test_query_api_integration.py -v
```

**Expected:** 139 passed, 2 skipped (hypothesis tests skip without `pip install hypothesis`)

### 2. Verify No Regressions in Full Test Suite
```bash
python -m pytest tests/ -q --tb=no
```

**Expected:** All existing tests still pass. Total should be ~3000+ tests.

### 3. Verify Tests Actually Test the Right Things

Spot-check that these key patterns are tested correctly:

| Pattern | Test File | What to Verify |
|---------|-----------|----------------|
| `Pattern().outgoing()` | `test_query_api_contracts.py:209-234` | Uses `outgoing()`, not `edge()` |
| `PathFinder.explain()` | `test_query_api_contracts.py:88-92` | No arguments passed |
| `match.bindings["A"].id` | `test_query_api_workflows.py:405-406` | Extracts `.id` from Task |
| `result.paths` | `test_query_api_integration.py:173-175` | Uses `.paths` attribute |

### 4. Verify Performance Tests Won't Flake on CI

Check the timing thresholds are reasonable:

```bash
grep -n "avg_ms < " tests/performance/test_query_api_perf.py
```

**Expected:** All thresholds should be `50` (not `10`).

```bash
grep -n "assert.*result1 == result2" tests/performance/test_query_api_perf.py
```

**Expected:** Direction method tests use behavioral equivalence, not timing.

### 5. Verify CLAUDE.md Updates

```bash
grep -A5 "DON'T use Pattern.edge" CLAUDE.md
grep -A5 "DON'T pass arguments to PathFinder.explain" CLAUDE.md
grep -A5 "DON'T assume PatternMatch bindings are IDs" CLAUDE.md
grep -A5 "DON'T forget PathFinder.all_paths" CLAUDE.md
```

**Expected:** All 4 patterns documented with WRONG/CORRECT examples.

### 6. Run with Coverage

```bash
python -m coverage run -m pytest tests/unit/got/test_pattern_matcher.py \
  tests/unit/got/test_query_api_contracts.py \
  tests/unit/got/test_query_api_invariants.py -q
python -m coverage report --include="cortical/got/*"
```

**Expected:** Query API modules should have good coverage (check `pattern_matcher.py`, `query_builder.py`, `graph_walker.py`, `path_finder.py`).

---

## Known Issues / Accepted Trade-offs

| Issue | Decision | Rationale |
|-------|----------|-----------|
| 2 tests skip without hypothesis | Acceptable | Graceful fallback, CI has hypothesis |
| 50ms timing thresholds (not 10ms) | Intentional | Prevents flaky failures on slow CI |
| Direction tests use behavioral equivalence | Intentional | Micro-benchmark timing too noisy |

---

## If Validation Fails

### Tests Fail
1. Check if API signatures changed (read error messages carefully)
2. Verify imports in `cortical/got/__init__.py` match test expectations
3. Check knowledge transfer doc for API patterns

### Timing Tests Flake
1. Increase thresholds further (100ms is still acceptable for explain())
2. Or skip timing tests with `@pytest.mark.skip(reason="timing too variable")`

### Coverage Low
1. Check if new Query API methods lack test coverage
2. Add targeted unit tests for uncovered branches

---

## Success Criteria

✅ All 139 new tests pass
✅ No regressions in existing tests
✅ Performance tests don't flake (run 3x to verify)
✅ CLAUDE.md patterns are accurate
✅ Coverage on Query API modules is reasonable (>80%)

---

## Commands to Run Validation

```bash
# Quick validation (2 min)
python -m pytest tests/unit/got/test_pattern_matcher.py \
  tests/unit/got/test_query_api_contracts.py \
  tests/unit/got/test_query_api_invariants.py \
  tests/behavioral/test_query_api_workflows.py \
  tests/performance/test_query_api_perf.py \
  tests/integration/test_query_api_integration.py -v

# Full validation with coverage (5 min)
python -m coverage run -m pytest tests/ -q && \
python -m coverage report --include="cortical/got/*"

# Stability check - run 3x
for i in 1 2 3; do
  echo "=== Run $i ==="
  python -m pytest tests/performance/test_query_api_perf.py -q
done
```

---

## After Validation

If all checks pass:
1. Mark this handoff as validated
2. Consider creating a PR if on a feature branch
3. Update any tracking tasks as complete

If issues found:
1. Document the issue
2. Fix or create a task for follow-up
3. Re-run validation
