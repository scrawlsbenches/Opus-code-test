# Knowledge Transfer: Fast Search Optimization (245x Improvement)

**Date:** 2025-12-29
**Session:** Performance tuning - fast_find_documents optimization
**Branch:** `claude/accept-got-handoff-pulIU`
**Handoff:** H-20251229-181947-a2ed6ed7

---

## Summary

Successfully optimized `fast_find_documents` achieving a **245x performance improvement** by removing an O(n) document scan that was negating the candidate filtering optimization.

---

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `fast_speedup` | 0.139x (7x slower) | 34.08x (34x faster) | **245x** |
| Standard search | 0.428ms | 0.428ms | (unchanged) |
| Fast search | 3.571ms | 0.013ms | **275x faster** |

---

## Root Cause Analysis

### The Problem

`fast_find_documents` in `cortical/query/search.py` had an O(n) loop (lines 211-224):

```python
# OLD CODE - O(n) scan of ALL documents
if doc_name_boost > 1.0:
    layer3 = layers.get(CorticalLayer.DOCUMENTS)
    if layer3:
        for doc_col in layer3.minicolumns.values():  # O(n) - ALL 1199 docs!
            doc_id = doc_col.content
            doc_name_tokens = set(tokenizer.tokenize(doc_id.replace('_', ' ')))  # Expensive!
            ...
```

This loop:
1. Iterated over ALL 1199 documents per query
2. Tokenized each document name per query
3. Negated the benefit of candidate filtering

### The Fix

1. **Removed the O(n) pre-scan loop** - Documents with content matches still get name boosts via `_apply_document_name_boost()`
2. **Passed layer3 to helper function** - Enables potential caching of name tokens
3. **Documented trade-off** - Pure name-only matches (no content match) are not included in fast search

---

## Scientific Method Applied

| Step | Action | Result |
|------|--------|--------|
| 1. Baseline | Run full benchmark suite | `fast_speedup=0.139x` identified |
| 2. Analyze | Read code, find O(n) loop | Lines 211-224 in search.py |
| 3. Hypothesis | Removing O(n) will restore performance | Predicted ~2-3x faster |
| 4. ONE change | Remove the loop, add comment | Commit e9c183a1 |
| 5. Verify | Re-run benchmark | 34.08x speedup achieved! |

---

## Files Modified

| File | Change |
|------|--------|
| `cortical/query/search.py` | Removed O(n) loop, added layer3 to _apply_document_name_boost |
| `tests/unit/test_query_search.py` | Updated test to document new behavior |
| `benchmarks/results/baseline-real-corpus.json` | Baseline benchmark results |

---

## Trade-off Accepted

**Pure name-only matches are not included in fast search.**

- Documents with content matches → get name boosts ✅
- Documents with ONLY name matches (no content) → not found by fast search ❌

**Recommendation:** Use `find_documents_for_query` if you need comprehensive search including name-only matches.

---

## Future Optimizations (Not Addressed)

From baseline benchmarks, other potential improvements:

| Metric | Value | Notes |
|--------|-------|-------|
| `save_corpus_ms` | 92,010ms | 92s to save - potential for streaming |
| `load_corpus_ms` | 41,265ms | 41s to load - potential for lazy loading |
| `bigram_connections_ms` | 1,163ms | Already optimized, reasonable for corpus size |

---

## Commit Details

```
Commit: e9c183a1
Message: perf(query): Remove O(n) document scan from fast_find_documents
Branch: claude/accept-got-handoff-pulIU
Tests: 687 passed
```

---

*Generated: 2025-12-29*
