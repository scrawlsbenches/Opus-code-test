# Knowledge Transfer: Performance Tuning Session

**Date:** 2025-12-29
**Session ID:** pulIU
**Branch:** `claude/accept-got-handoff-pulIU`
**Handoff:** H-20251229-181947-a2ed6ed7 (completed)

---

## Summary

Completed scientific performance tuning achieving:
- **245x improvement** in fast search
- **22% faster** corpus load
- **51% faster** compute_all (index build)

Also established scaling benchmarks for capacity planning.

---

## Optimizations Applied

### 1. Fast Search O(n) Removal (245x improvement)

**Problem:** `fast_find_documents` was iterating over ALL documents (O(n)) to find name-only matches, negating the performance benefit of candidate filtering.

**Root cause:** Lines 211-224 in `cortical/query/search.py`:
```python
# OLD - O(n) scan of ALL 1199 documents per query
for doc_col in layer3.minicolumns.values():
    doc_name_tokens = set(tokenizer.tokenize(doc_id.replace('_', ' ')))
    ...
```

**Fix:** Removed the O(n) loop. Documents with content matches get name boosts via `_apply_document_name_boost()`.

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| `fast_speedup` | 0.139x (7x slower) | 34.08x (34x faster) |
| Improvement | - | **245x** |

**Trade-off:** Pure name-only matches (no content match) not included in fast search. Use `find_documents_for_query` for comprehensive search.

**Commit:** `e9c183a1`

---

### 2. Lazy Edge Loading (22% faster load)

**Problem:** Corpus load took ~40s. Profiling revealed 17.5s spent creating 4.1M Edge objects from JSON.

**Root cause:** `Minicolumn.from_dict()` immediately converted all typed_connections to Edge objects:
```python
# OLD - 4.1M Edge.from_dict() calls during load
col.typed_connections = {
    target_id: Edge.from_dict(edge_data)
    for target_id, edge_data in typed_conn_data.items()
}
```

**Fix:** Implemented lazy loading:
1. Store raw dict in `_typed_connections_raw` during load
2. Convert to Edge objects only on first property access
3. Most searches never trigger conversion (don't use edges)

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| Load time | ~40s | ~31s |
| Improvement | - | **22%** |

**Commit:** `a9256211`

---

### 3. In-Place Edge Updates (51% faster compute_all)

**Problem:** `compute_all` took 40.2s on 200 docs. Profiling showed `add_lateral_connections_batch` taking 10.2s with 85K calls.

**Root cause:** Every weight update created a new Edge object:
```python
# OLD - creates new object every time
typed[target_id] = Edge(
    target_id=target_id,
    weight=existing.weight + weight,  # New object just to update weight!
    ...
)
```

**Fix:** Modify Edge fields in place (Edge is a mutable dataclass):
```python
# NEW - modify in place
typed[target_id].weight += weight
```

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| compute_all (200 docs) | 40.2s | 19.7s |
| Improvement | - | **51%** |

**Commit:** `be5c40f0`

---

## Scaling Analysis

Established scaling characteristics for capacity planning:

| Corpus Size | Load Time | Search Time | Notes |
|-------------|-----------|-------------|-------|
| 1,200 docs | 35s | 0.3ms | Current |
| 2,400 docs | ~69s | 0.3ms | 2x growth |
| 6,000 docs | ~3 min | 0.3ms | 5x growth |
| 12,000 docs | ~6 min | 0.3ms | 10x growth |

**Key insights:**
- **Search: O(query_terms)** - constant regardless of corpus size ✅
- **Load: O(n)** - linear with documents ⚠️
- **JSON parsing: 72% of load time** - the scaling bottleneck

**Thresholds:**
- <5K docs: Current approach works fine
- 5-10K docs: Load becomes noticeable (~3-5 min)
- >10K docs: Need binary format or streaming load

**Commit:** `39a0eaa2`

---

## Files Modified

| File | Change |
|------|--------|
| `cortical/query/search.py` | Removed O(n) document scan |
| `cortical/minicolumn.py` | Implemented lazy Edge loading |
| `tests/unit/test_query_search.py` | Updated test for new behavior |
| `benchmarks/results/baseline-real-corpus.json` | Full benchmark baseline |
| `benchmarks/results/scaling-analysis.json` | Scaling projections |

---

## Scientific Method Applied

| Step | Action | Result |
|------|--------|--------|
| 1. Baseline | Run full benchmark suite | `fast_speedup=0.139x`, load=40s |
| 2. Profile | cProfile load operation | Found 4.1M Edge objects, O(n) scan |
| 3. Hypothesis | Removing O(n) → faster search | Predicted 2-3x improvement |
| 4. ONE change | Remove O(n) loop | 245x improvement! |
| 5. Verify | Re-run benchmarks | Confirmed |
| 6. Repeat | Profile load | Found Edge creation bottleneck |
| 7. ONE change | Lazy Edge loading | 22% faster load |
| 8. Verify | Re-run load tests | Confirmed |

---

## Remaining Bottlenecks

| Operation | Bottleneck | Time | Solution |
|-----------|------------|------|----------|
| Load | JSON parsing | 24.7s (72%) | Binary format (MessagePack, Protobuf) |
| Load | Minicolumn creation | 6.6s | Possible with `__slots__` optimization |
| Search | None | 0.3ms | Already O(query_terms) |

---

## Commands for Next Session

```bash
# Verify current state
python scripts/got_utils.py task show T-20251229-181927-f8a5b6aa

# Run scaling benchmark
python -c "..." # See benchmarks/results/scaling-analysis.json

# Test search performance
python scripts/search_codebase.py "query" --top 5

# Profile load if investigating further
python -c "import cProfile; ..."
```

---

## Regression Protection

Performance regression guards added to prevent reintroducing these bugs:

| Optimization | Test | Guard Type |
|-------------|------|------------|
| Fast search O(n) removal | `test_fast_search_must_be_faster_than_standard` | Timing comparison |
| Fast search O(n) removal | `test_no_all_documents_iteration_in_fast_search` | Code pattern check |
| Lazy Edge loading | `test_minicolumn_from_dict_defers_edge_creation` | State inspection |
| In-place Edge updates | `test_add_lateral_connection_updates_in_place` | Object identity check |
| In-place Edge updates | `test_add_lateral_connections_batch_updates_in_place` | Object identity check |

**File:** `tests/regression/test_perf_regressions.py`
**Commit:** `4594f5e3`

These run automatically in CI as part of the regression test suite.

---

## Tests Status

- **687+ tests passing** after all changes
- All minicolumn, persistence, and query tests green
- 5 new regression tests for performance guards

---

*Generated: 2025-12-29*
*Updated: 2025-12-29 (added regression protection)*
