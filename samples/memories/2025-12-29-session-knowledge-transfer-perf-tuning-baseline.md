# Knowledge Transfer: Performance Tuning Baseline Session

**Date:** 2025-12-29
**Session:** Scientific performance tuning approach
**Branch:** `claude/fix-got-handoff-api-czz09`

---

## Summary

Started a scientific approach to performance tuning. Established corpus indexing but was interrupted before completing baseline benchmarks. Key insight: **measure first, optimize second**.

---

## What Was Accomplished

### 1. Corpus Indexed Successfully

```
Corpus Statistics:
  Documents: 1198
  Tokens (Layer 0): 50639
  Bigrams (Layer 1): 570027
  Concepts (Layer 2): 0
  Semantic relations: 0

Phase breakdown:
  Discovering files: 0.12s
  Indexing files: 31.05s (1198 files, 415,718 lines)
  Computing analysis (fast mode): 57.35s
  Saving corpus: 90.19s
  Total: 178.80s
```

Corpus saved to: `corpus_dev.json/`

### 2. Synthetic Corpus Benchmarks (25 docs)

| Metric | Value | Notes |
|--------|-------|-------|
| `bigram_connections_ms` | 122.7ms | 83% of compute time |
| `pagerank_ms` | 17.3ms | Fast |
| `tfidf_ms` | 0.7ms | Very fast |
| `fast_search.speedup` | 1.175x | Fast search IS faster on small corpus |
| `cache_speedup` | 2.09x | Warm cache helps |

### 3. Previous Agent's Optimization (merged)

The previous agent (H-20251229-142212-fac4c081) optimized `compute_bigram_connections`:
- Pair canonicalization: 1.39x faster
- Early bailout for maxed bigrams: 62.7% wasted work eliminated
- Result: **12.9% faster, 14.8% better throughput**

---

## Key Findings

### Potential Issue Identified (NOT YET VERIFIED)

In `cortical/query/search.py` lines 213-224, `fast_find_documents` has O(n) complexity:

```python
if doc_name_boost > 1.0:
    layer3 = layers.get(CorticalLayer.DOCUMENTS)
    if layer3:
        for doc_col in layer3.minicolumns.values():  # O(N) - ALL documents!
            doc_name_tokens = set(tokenizer.tokenize(...))  # Tokenize EVERY name!
```

**Hypothesis:** This explains why fast_search is 7x slower on large corpus (per previous knowledge transfer). But this is UNVERIFIED - need baseline data first.

### Scientific Method Reminder

The user correctly cautioned: **Don't affect measurements before establishing baseline.**

Proper approach:
1. Establish baseline (measure current state with no changes)
2. Identify bottleneck (use data, not guesses)
3. Form hypothesis (predict improvement)
4. Make ONE change (isolate variable)
5. Measure again (validate)
6. Document (record findings)

---

## What Needs To Be Done

### Immediate Next Steps

1. **Run full benchmark suite on real corpus**
   ```bash
   python -m benchmarks.corpus.runner --all --use-corpus corpus_dev.json --output benchmarks/results/baseline.json
   ```

2. **Analyze baseline results** - Look for actual bottlenecks in the data

3. **Focus on these benchmarks specifically:**
   - `fast_search_comparison` - Verify the 7x slowdown claim
   - `compute_all_phases` - See where time goes on large corpus
   - `bigram_connections` - Check if recent optimization helped

4. **Form hypothesis AFTER seeing data** - Not before

### Commands Ready To Run

```bash
# Full benchmark (takes ~5 minutes)
python -m benchmarks.corpus.runner --all --use-corpus corpus_dev.json --output benchmarks/results/baseline.json

# Quick check of specific benchmark
python -m benchmarks.corpus.runner --benchmark fast_search_comparison --use-corpus corpus_dev.json

# Profile compute_all phases
python -m benchmarks.corpus.runner --benchmark compute_all_phases --use-corpus corpus_dev.json
```

---

## Files Modified This Session

None - we were establishing baseline, not making changes.

---

## Context From Previous Sessions

From `samples/memories/2025-12-29-session-knowledge-transfer-corpus-benchmarks.md`:

| Metric | Synthetic (25 docs) | Real (1194 docs) | Notes |
|--------|---------------------|------------------|-------|
| `fast_search.speedup` | 0.76x | 0.14x | **7x slower on large corpus** |
| `graph_boosted_search.overhead` | +500% | -27% | Faster on large corpus |

---

## Session Metrics

- **Duration:** ~10 minutes (interrupted)
- **Commits:** 0 (baseline only)
- **Corpus indexed:** Yes (corpus_dev.json/)
- **Baseline benchmarks:** Started but interrupted

---

*Generated: 2025-12-29*
