# Performance Benchmarks

This document provides real performance numbers for the Cortical Text Processor, measured on representative workloads.

**Test Environment:**
- Platform: Linux 4.4.0
- Python: 3.x
- Date: December 2025

---

## Quick Reference

| Operation | Small Corpus (25 docs) | Real Codebase (539 files) |
|-----------|------------------------|---------------------------|
| Document processing | 0.31 ms/doc | ~20 ms/doc |
| compute_all() | 141 ms | ~130 s |
| Standard search | 0.13 ms | ~2 ms |
| Fast search | 0.06 ms | ~1 ms |
| Passage retrieval | 0.36 ms | ~100 ms |

---

## Small Corpus Benchmarks (25 Documents)

### Corpus Profile
| Metric | Value |
|--------|-------|
| Documents | 25 |
| Tokens (Layer 0) | 460 |
| Bigrams (Layer 1) | 592 |
| Concepts (Layer 2) | 31 |
| L0 Connections | 3,323 |

### Document Processing

| Metric | Time |
|--------|------|
| Total (25 docs) | 7.8 ms |
| Average per document | 0.31 ms |
| Min per document | 0.22 ms |
| Max per document | 0.58 ms |

### Compute Phases

Individual phase timings reveal where processing time is spent:

| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| bigram_connections | 106.9 | 68.6% |
| concept_clusters | 14.8 | 9.5% |
| pagerank | 13.1 | 8.4% |
| activation_propagation | 12.4 | 8.0% |
| graph_embeddings | 4.7 | 3.0% |
| doc_connections | 3.5 | 2.2% |
| tfidf | 0.4 | 0.2% |
| **TOTAL** | **155.8** | **100%** |

Full `compute_all()` pipeline: **140.9 ms**

### Query Operations

| Operation | Avg (ms) | Min (ms) | Max (ms) |
|-----------|----------|----------|----------|
| Standard search | 0.13 | 0.06 | 0.25 |
| Fast search | 0.06 | 0.05 | 0.08 |
| Query expansion | 0.05 | 0.03 | 0.07 |
| Passage retrieval | 0.36 | 0.23 | 0.62 |

**Fast search speedup: 2.0x** over standard search

### Persistence

| Operation | Time | Size |
|-----------|------|------|
| Save | 137.1 ms | 1.33 MB |
| Load | 45.4 ms | - |

---

## Real Codebase Benchmarks (539 Files)

Benchmarked using the full codebase including `cortical/`, `scripts/`, `tests/`, `docs/`, and `samples/` directories with recursive subdirectory indexing.

### Corpus Profile
| Metric | Value |
|--------|-------|
| Files | 539 |
| Lines | ~136,000 |
| Tokens (Layer 0) | 27,306 |
| Bigrams (Layer 1) | 237,901 |
| Concepts (Layer 2) | 35 |
| L0 Connections | 1,189,953 |

### Compute Phase Timings (December 2025)

Profiled on full codebase with `scripts/profile_full_analysis.py`:

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| bigram_connections | 55.41 | 42.6% |
| semantics | 32.46 | 24.9% |
| louvain | 17.97 | 13.8% |
| pagerank | 15.70 | 12.1% |
| activation | 5.04 | 3.9% |
| doc_connections | 3.46 | 2.7% |
| tfidf | 0.10 | 0.1% |
| **TOTAL** | **130.14** | **100%** |

### Processing Times

| Operation | Time |
|-----------|------|
| Document processing | ~11 s |
| compute_all() | ~130 s |

### Persistence (JSON format)

| Operation | Time |
|-----------|------|
| Save | ~34 s |
| Load | ~8 s |

---

## Scaling Characteristics

How performance scales with corpus size:

| Documents | Tokens | Process (ms) | Compute (ms) | Query (ms) |
|-----------|--------|--------------|--------------|------------|
| 10 | 36 | 11.9 | 4.5 | 0.04 |
| 25 | 36 | 26.6 | 5.2 | 0.05 |
| 50 | 36 | 53.3 | 7.8 | 0.07 |
| 100 | 36 | 109.0 | 17.3 | 0.13 |
| 200 | 36 | 215.3 | 44.9 | 0.22 |

**Observations:**
- Document processing scales linearly (O(n))
- Query time scales sub-linearly due to indexed lookups
- Compute time scales slightly super-linearly due to connection building

---

## Performance Thresholds

These thresholds are used in CI to catch performance regressions:

| Operation | Threshold | Notes |
|-----------|-----------|-------|
| compute_all() (25 docs) | 5,000 ms | Generous for CI variability |
| propagate_activation | 1,000 ms | Per-phase limit |
| compute_importance | 1,000 ms | PageRank phase |
| compute_tfidf | 1,000 ms | TF-IDF phase |
| compute_bigram_connections | 2,000 ms | Most expensive phase |
| build_concept_clusters | 2,000 ms | Louvain clustering |
| compute_graph_embeddings | 2,000 ms | Embedding phase |
| Single query | 200 ms | Interactive use |
| Query expansion | 100 ms | Per expansion |
| Passage retrieval | 500 ms | Including chunking |

---

## Bottleneck Analysis

### Primary Bottleneck: Bigram Connections

On both small and large corpora, `compute_bigram_connections` dominates at **42.6%** of compute time. Uses batch processing to reduce method call overhead.

**Mitigation parameters:**
```python
CorticalConfig(
    max_bigrams_per_term=100,   # Limits per-term connections
    max_bigrams_per_doc=500,    # Limits per-document processing
)
```

### Secondary Bottleneck: Semantic Extraction

On larger corpora, semantic extraction is the second largest phase at **24.9%** due to similarity computation across term pairs.

**Mitigation parameters:**
```python
CorticalConfig(
    max_similarity_pairs=100000,  # Cap total similarity computations
    min_context_keys=3,           # Require minimum context overlap
)
```

### Optimized: Document Connections

Previously the dominant bottleneck at 53.2% (138.74s), now only **2.7%** (3.46s) after Sprint 8 optimization that changed from O(n²·m) to O(m·d²) complexity.

### Lessons Learned

From profiling sessions (documented in `samples/performance_profiling_process.txt`):

1. **Profile before optimizing** - The obvious bottleneck (Louvain clustering) was actually fast; real culprits were connections and semantics
2. **O(n²) patterns explode** - Common terms like "self" creating millions of pairs
3. **Invert the loop** - Iterating tokens once and accumulating document pairs is faster than checking all pairs against all tokens
4. **Parameter limits are essential** - Trading completeness for tractability

---

## Running Benchmarks

### Quick Performance Check
```bash
# Run performance tests (without coverage)
python -m pytest tests/performance/ -v --no-cov
```

### Profile Full Analysis
```bash
# Profile all compute phases
python scripts/profile_full_analysis.py

# Profile specific phase
python scripts/profile_full_analysis.py --phase louvain
python scripts/profile_full_analysis.py --phase semantics
python scripts/profile_full_analysis.py --phase bigram

# With custom timeout
python scripts/profile_full_analysis.py --timeout 60
```

### Custom Benchmarks

```python
from cortical import CorticalTextProcessor
import time

processor = CorticalTextProcessor(enable_metrics=True)

# Process documents
for doc_id, content in documents.items():
    processor.process_document(doc_id, content)

# Run compute phases
processor.compute_all(verbose=True)  # Shows phase timings

# Get metrics summary
print(processor.get_metrics_summary())
```

---

## Optimizing for Your Use Case

### High-Throughput Document Processing

```python
# Use batch processing
processor.add_documents_batch(doc_dict)  # More efficient than individual adds

# Skip expensive phases if not needed
processor.compute_tfidf()  # Just TF-IDF
processor.compute_importance()  # Just PageRank
# Skip: bigram_connections, concept_clusters, semantics
```

### Low-Latency Search

```python
# Use fast search for interactive queries
results = processor.fast_find_documents(query, top_n=10)

# Pre-build search index for repeated queries
index = processor.build_search_index()
results = processor.search_with_index(query, index)

# Cache query expansions
expanded = processor.expand_query_cached(query)
```

### Memory-Constrained Environments

```python
# Use streaming document processing
for doc_id, content in large_corpus:
    processor.add_document_incremental(doc_id, content, recompute='none')

# Compute in batches
processor.compute_tfidf()  # Lightweight
processor.compute_importance()  # Moderate
# Defer expensive operations
```

### Git-Friendly Collaborative Indexing

```bash
# Use chunk-based storage for team workflows
python scripts/index_codebase.py --incremental --use-chunks

# Periodic compaction
python scripts/index_codebase.py --compact --use-chunks
```

---

## Historical Performance Fixes

| Date | Issue | Before | After | Fix |
|------|-------|--------|-------|-----|
| 2025-12-18 | doc_connections O(n²·m) | 138.74s | 3.46s | **Inverted loop: O(m·d²) - 40x faster** |
| 2025-12-18 | bigram_connections overhead | 4.69M calls | 128K calls | Batched connection updates |
| 2025-12-11 | bigram_connections timeout | 20.85s timeout | 10.79s | `max_bigrams_per_term=100` |
| 2025-12-11 | semantics timeout | 30.05s timeout | 5.56s | `max_similarity_pairs=100000` |
| 2025-12-11 | louvain (not a bottleneck) | 2.2s | 2.2s | No change needed |

### Sprint 8 Optimizations (2025-12-18)

**1. Bigram Connections - Batch Processing**

The `compute_bigram_connections` function was optimized to use batch connection updates:

```python
# Old: 4.69M individual calls with cache invalidation
b1.add_lateral_connection(b2.id, weight)

# New: Accumulate then batch (128K batch calls)
pending_connections[b1.id][b2.id] += weight
# ... at end:
bigram.add_lateral_connections_batch(dict(connections))
```

**2. Document Connections - Loop Inversion**

The `compute_document_connections` function was optimized from O(n²·m) to O(m·d²):

```python
# Old: For each doc pair, check ALL tokens (4B iterations for 539 docs)
for doc1 in docs:
    for doc2 in docs:
        for token in all_tokens:  # 27,306 tokens!
            if doc1 in token.docs and doc2 in token.docs: ...

# New: Iterate tokens once, accumulate document pairs
for token in all_tokens:
    for doc1 in token.docs:
        for doc2 in token.docs:  # Only docs sharing this token
            pair_weights[(doc1, doc2)] += token.tfidf
```

Result: **40x speedup** (138.74s → 3.46s), total compute_all() reduced by **50%**.

---

## Notes

- All timings measured without coverage instrumentation (coverage adds ~10x overhead)
- Timings will vary based on hardware, Python version, and corpus characteristics
- Query performance depends heavily on corpus vocabulary and connection density
- Persistence times are I/O bound and vary with storage medium
