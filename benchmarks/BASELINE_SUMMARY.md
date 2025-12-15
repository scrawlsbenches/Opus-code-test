# TF-IDF Baseline Performance Summary

**Date:** 2025-12-15
**Algorithm:** TF-IDF (current implementation)
**Purpose:** Baseline measurements before BM25 migration

## Executive Summary

| Metric | Synthetic (100 docs) | Real Corpus (150 docs) |
|--------|---------------------|------------------------|
| Score Computation | 0.72ms | 16.30ms |
| Search Latency (mean) | 0.15ms | 0.37ms |
| Search Throughput | 6,507 QPS | ~2,700 QPS |
| Scaling Complexity | O(n) | O(n) |

## Detailed Benchmarks

### 1. Score Computation Time

How long it takes to compute TF-IDF scores for all terms.

| Corpus Size | Vocabulary | Time (ms) | Per-Doc (ms) | Per-Term (us) |
|-------------|------------|-----------|--------------|---------------|
| 25 docs | 64 terms | 0.20 | 0.008 | 3.13 |
| 50 docs | 64 terms | 0.38 | 0.008 | 5.99 |
| 100 docs | 64 terms | 0.72 | 0.007 | 11.17 |
| 200 docs | 64 terms | 1.42 | 0.007 | 22.17 |
| **150 docs (real)** | **11,862 terms** | **16.30** | **0.109** | **1.37** |

**Observations:**
- Computation scales linearly with corpus size (O(n))
- Per-document cost is stable (~0.007ms for synthetic, ~0.1ms for real)
- Real corpus has 185x more vocabulary, hence higher absolute time

### 2. Search Query Latency

Time to execute a search query and return ranked results.

**Synthetic Corpus (100 docs, 64 terms):**
| Metric | Value |
|--------|-------|
| Mean Latency | 0.15ms |
| Median Latency | 0.15ms |
| P95 Latency | 0.18ms |
| Max Latency | 0.18ms |
| Throughput | 6,507 QPS |

**Real Corpus (150 docs, 11,862 terms):**
| Query | Latency (ms) | Results |
|-------|--------------|---------|
| pagerank algorithm | 0.38 | 5 |
| tfidf computation | 0.30 | 5 |
| lateral connections | 0.31 | 5 |
| query expansion | 0.46 | 5 |
| document search | 0.52 | 5 |
| minicolumn layer | 0.40 | 5 |
| semantic relations | 0.37 | 5 |
| louvain clustering | 0.20 | 5 |
| **Mean** | **0.37** | - |

### 3. Search Relevance Quality

Using domain-based relevance (documents from same domain should rank higher).

| Query | Domain | P@1 | P@3 | MRR | Term Recall |
|-------|--------|-----|-----|-----|-------------|
| neural network training | ml | 0.00 | 0.00 | 0.11 | 0.80 |
| database query optimization | db | 1.00 | 1.00 | 1.00 | 0.80 |
| process memory management | sys | 1.00 | 1.00 | 1.00 | 0.80 |
| api authentication | web | 1.00 | 1.00 | 1.00 | 0.80 |
| **Mean** | - | **0.75** | **0.75** | **0.78** | **0.80** |

**Known Issue:** "neural network training" query performs poorly - first relevant result at rank 9.

### 4. Memory Footprint

Memory used by TF-IDF score storage.

| Corpus Size | TF-IDF Entries | Memory (KB) | Bytes/Entry |
|-------------|----------------|-------------|-------------|
| 25 docs | 995 | 51.8 | 53.4 |
| 50 docs | 1,988 | 99.1 | 51.0 |
| 100 docs | 4,010 | 193.7 | 49.5 |
| 200 docs | 8,043 | 398.6 | 50.8 |

**Memory scales linearly** at ~50 bytes per (term, document) entry.

### 5. Scaling Behavior

Log-log regression to estimate computational complexity.

| Docs | Time (ms) |
|------|-----------|
| 10 | 0.08 |
| 25 | 0.16 |
| 50 | 0.33 |
| 100 | 0.62 |
| 150 | 0.99 |
| 200 | 1.30 |

**Scaling Exponent:** 0.94 (close to 1.0)
**Estimated Complexity:** O(n) - linear scaling confirmed

## Targets for BM25

Based on these baselines, BM25 should achieve:

| Metric | Baseline | Target | Notes |
|--------|----------|--------|-------|
| Compute Time | 16.3ms | < 20ms | Allow 20% overhead for length normalization |
| Search Latency | 0.37ms | < 0.5ms | Same or better |
| Mean P@3 | 0.75 | > 0.80 | Improved relevance expected |
| Mean MRR | 0.78 | > 0.85 | Better first-result ranking |
| Memory | 50 bytes/entry | < 60 bytes | Small overhead for doc lengths OK |
| Complexity | O(n) | O(n) | Must maintain linear scaling |

## Files

- `baseline_tfidf.json` - Synthetic corpus benchmarks
- `baseline_tfidf_real.json` - Real corpus benchmarks

## How to Compare After BM25 Implementation

```bash
# Run benchmarks with BM25
python scripts/benchmark_scoring.py --output benchmarks/after_bm25.json

# Compare results
python scripts/benchmark_scoring.py --compare benchmarks/baseline_tfidf.json benchmarks/after_bm25.json
```

## Notes

1. **Synthetic corpus** has fixed 64-term vocabulary due to deterministic generation
2. **Real corpus** (150 files, 11,862 terms) is more representative of actual usage
3. **Query expansion** uses PageRank + lateral connections, not TF-IDF directly
4. **Term recall** measures how many expected terms appear in query expansion
