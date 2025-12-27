# Knowledge Transfer: Query API Testing Session

**Date:** 2025-12-27
**Branch:** `claude/evaluate-query-language-usability-BUe5R`
**Tags:** `query-api`, `testing`, `got`, `pattern-matcher`, `graph-walker`, `path-finder`

## Session Summary

Added comprehensive test coverage for the Query API (139 tests across 6 files). This document captures key learnings for future sessions.

## Critical API Patterns Learned

### 1. Pattern Edge Direction Methods

**Pattern uses `outgoing()`/`incoming()`, not `edge()`:**
```python
# WRONG
pattern = Pattern().node("A").edge("A", "B", "DEPENDS_ON").node("B")

# CORRECT
pattern = Pattern().node("A").outgoing("DEPENDS_ON").node("B")
```

The `edge()` method exists but has a different signature (2-3 args, not 4).

### 2. Direction Methods Belong to Different Classes

| Class | Direction Methods | Scope |
|-------|-------------------|-------|
| `GraphWalker` | `outgoing()`, `incoming()`, `both()` | Global (on Walker) |
| `Pattern` | `outgoing()`, `incoming()`, `both()` | Per-edge (on Pattern) |
| `PathFinder` | `directed()` | Global (on Finder) |

**PatternMatcher does NOT have direction methods** - those are on Pattern.

### 3. PathFinder.explain() Takes No Arguments

```python
# WRONG
plan = finder.explain(source_id, target_id)

# CORRECT - describes finder configuration, not a specific search
plan = finder.explain()
```

### 4. PatternMatch Bindings Contain Task Objects

```python
match = matcher.find(pattern)[0]
task = match.bindings["A"]  # This is a Task object, NOT a string ID

# To get the ID:
task_id = match.bindings["A"].id
```

### 5. PathFinder.all_paths() Returns PathSearchResult

```python
result = finder.all_paths(source, target)
# Access paths via .paths attribute:
for path in result.paths:
    print(path)
# Check truncation:
if result.truncated:
    print(f"Found {result.paths_found} paths, showing {len(result.paths)}")
```

### 6. PatternMatch Only Takes `bindings` Parameter

```python
# WRONG
PatternMatch(bindings={"A": task}, edges=[])

# CORRECT
PatternMatch(bindings={"A": task})
```

## Test Files Created

| File | Tests | Purpose |
|------|-------|---------|
| `tests/unit/got/test_pattern_matcher.py` | 36 | PatternSearchResult, PatternPlan, explain() |
| `tests/unit/got/test_query_api_contracts.py` | 38 | API consistency across all 4 query classes |
| `tests/unit/got/test_query_api_invariants.py` | 28 | Property-based invariants |
| `tests/behavioral/test_query_api_workflows.py` | 15 | User workflow scenarios |
| `tests/performance/test_query_api_perf.py` | 12 | Performance and overhead tests |
| `tests/integration/test_query_api_integration.py` | 10 | Multi-tool integration |

## Design Decisions Made

### Performance Test Approach

**Decision:** Use behavioral equivalence tests instead of timing comparisons for alias methods.

**Rationale:** Micro-benchmarks are too noisy for CI. A 2.78x timing ratio doesn't indicate a bug when both methods are aliases for the same operation.

**Implementation:**
```python
# Instead of timing comparison:
result1 = walker.directed().visit(...).run()
result2 = walker.outgoing().visit(...).run()
assert result1 == result2  # Behavioral equivalence
```

### Timing Thresholds

**Decision:** Use 50ms threshold instead of 10ms for `explain()` methods.

**Rationale:** Slow CI servers can have 5x variance on fast operations. 50ms is still fast enough to catch real regressions while avoiding flaky failures.

## Files Modified

- `CLAUDE.md`: Added 4 new "Common Mistakes" entries for Query API patterns
- `tests/performance/test_query_api_perf.py`: Fixed flaky timing tests

## Commits

1. `730993dc` - test(got): Add comprehensive Query API test coverage (139 tests)
2. `2ebc6aa4` - fix(tests): Improve Query API test reliability and add docs

## Future Considerations

1. **Hypothesis tests**: 2 tests skip without hypothesis. Consider making them run with manual fallback.
2. **Pattern.edge() signature**: Consider documenting what `edge()` actually does vs `outgoing()`/`incoming()`.
3. **PathSearchResult consistency**: Consider if `all_paths()` should support list-like interface directly.

## Related Documentation

- `docs/got-query-guide.md` - Query API user guide
- `cortical/got/__init__.py` - Exports for Query, GraphWalker, PathFinder, PatternMatcher
