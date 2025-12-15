# Scoring Algorithm Performance Comparison

**Date:** 2025-12-15
**Algorithms Compared:** TF-IDF vs BM25
**Default Algorithm:** BM25 (as of this commit)

## Executive Summary

### TF-IDF vs BM25 Comparison

| Metric | TF-IDF | BM25 | Change |
|--------|--------|------|--------|
| Score Computation (100 docs) | 0.72ms | 1.26ms | +75% |
| Search Latency | 0.15ms | 0.15ms | +0.8% |
| Mean P@3 | 0.75 | 0.75 | 0% |
| Mean MRR | 0.78 | 0.78 | 0% |
| Scaling Complexity | O(n) | O(n) | Same |

### Key Findings

1. **BM25 computation is ~60-90% slower** than TF-IDF due to length normalization overhead
2. **Search latency is virtually identical** (< 1% difference)
3. **Relevance metrics are the same** on the synthetic corpus (uniform document lengths)
4. **Both algorithms scale linearly** with corpus size

### Why Use BM25 Despite Slower Computation?

1. **Term frequency saturation**: Prevents single repeated terms from dominating scores
2. **Length normalization**: Fair comparison across documents of different sizes
3. **Industry standard**: Used by Elasticsearch, Lucene, and most modern search engines
4. **Real-world relevance**: Benefits appear with variable document lengths

## Detailed Benchmarks

### Score Computation Time

| Corpus Size | TF-IDF (ms) | BM25 (ms) | Overhead |
|-------------|-------------|-----------|----------|
| 25 docs | 0.20 | 0.33 | +65% |
| 50 docs | 0.38 | 0.62 | +63% |
| 100 docs | 0.72 | 1.26 | +75% |
| 200 docs | 1.42 | 2.73 | +92% |
| **Real (150 docs)** | **16.3** | **25.6** | **+57%** |

**Note:** The overhead comes from the length normalization calculation in BM25.

### Search Query Latency

Both algorithms have nearly identical search latency:

| Algorithm | Mean Latency | Throughput |
|-----------|--------------|------------|
| TF-IDF | 0.15ms | 6,507 QPS |
| BM25 | 0.15ms | 6,374 QPS |

Search uses pre-computed scores, so the algorithm choice doesn't affect query time.

### Search Relevance Quality

On the synthetic corpus (uniform document lengths):

| Metric | TF-IDF | BM25 |
|--------|--------|------|
| Mean P@1 | 0.75 | 0.75 |
| Mean P@3 | 0.75 | 0.75 |
| Mean MRR | 0.78 | 0.78 |
| Term Recall | 0.80 | 0.80 |

**Note:** Relevance is identical on synthetic corpus because documents have uniform lengths. BM25's benefits appear with variable document lengths.

### Memory Footprint

| Corpus Size | TF-IDF (KB) | BM25 (KB) |
|-------------|-------------|-----------|
| 100 docs | 193.7 | 193.8 |
| 200 docs | 398.6 | 398.7 |

Memory usage is essentially identical. The doc_lengths dictionary adds negligible overhead.

### Scaling Behavior

| Algorithm | Scaling Exponent | Complexity |
|-----------|-----------------|------------|
| TF-IDF | 0.94 | O(n) |
| BM25 | 0.96 | O(n) |

Both algorithms maintain linear scaling with corpus size.

## Configuration

BM25 is now the default. To switch algorithms:

```python
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig

# Use BM25 (default)
processor = CorticalTextProcessor()

# Use TF-IDF
config = CorticalConfig(scoring_algorithm='tfidf')
processor = CorticalTextProcessor(config=config)

# Tune BM25 parameters
config = CorticalConfig(
    scoring_algorithm='bm25',
    bm25_k1=1.5,  # Term frequency saturation (default: 1.2)
    bm25_b=0.75   # Length normalization (default: 0.75)
)
```

### BM25 Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `bm25_k1` | 1.2 | 0-3 | Term frequency saturation. Higher = more weight to term frequency |
| `bm25_b` | 0.75 | 0-1 | Length normalization. 0 = none, 1 = full |

## Files

- `baseline_tfidf.json` - TF-IDF benchmark results
- `baseline_tfidf_real.json` - TF-IDF real corpus results
- `after_bm25.json` - BM25 benchmark results

## How to Run Benchmarks

```bash
# Run with current default algorithm (BM25)
python scripts/benchmark_scoring.py --output benchmarks/current.json

# Run with specific algorithm
python scripts/benchmark_scoring.py --algorithm tfidf --output benchmarks/tfidf.json
python scripts/benchmark_scoring.py --algorithm bm25 --output benchmarks/bm25.json

# Compare two benchmark runs
python scripts/benchmark_scoring.py --compare benchmarks/baseline_tfidf.json benchmarks/after_bm25.json
```

## Implementation Notes

### What Changed

1. **config.py**: Added `scoring_algorithm`, `bm25_k1`, `bm25_b` parameters
2. **analysis.py**: Added `compute_bm25()` and `_bm25_core()` functions
3. **processor/core.py**: Added `doc_lengths` and `avg_doc_length` tracking
4. **processor/documents.py**: Track document lengths during processing
5. **processor/compute.py**: `compute_tfidf()` now respects `scoring_algorithm` config
6. **processor/persistence_api.py**: Save/restore document lengths

### BM25 Formula

```
BM25(t, d) = IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))

Where:
- IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
- tf(t,d) = term frequency in document
- |d| = document length (in tokens)
- avgdl = average document length
- k1 = term frequency saturation parameter
- b = length normalization parameter
```

### Backward Compatibility

- Scores are stored in the same `col.tfidf` and `col.tfidf_per_doc` fields
- All existing search functions work unchanged
- Old pickle files are compatible (doc_lengths are recomputed on load)
