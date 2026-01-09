# CorticalTextProcessor Reference

> **Gate**: Working with the processor API? Read this file.

---

## Scoring Algorithms

The processor supports multiple scoring algorithms for term weighting:

### BM25 (Default)

BM25 (Best Match 25) is the default scoring algorithm, optimized for code search:

```python
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig

# BM25 with default parameters (recommended)
config = CorticalConfig(scoring_algorithm='bm25')

# Tune BM25 parameters if needed
config = CorticalConfig(
    scoring_algorithm='bm25',
    bm25_k1=1.2,  # Term frequency saturation (0.0-3.0, default 1.2)
    bm25_b=0.75   # Length normalization (0.0-1.0, default 0.75)
)
processor = CorticalTextProcessor(config=config)
```

**Parameters:**
- `bm25_k1`: Controls term frequency saturation. Higher values give more weight to term frequency.
- `bm25_b`: Controls document length normalization. Set to 0.0 to disable length normalization.

### TF-IDF (Legacy)

```python
config = CorticalConfig(scoring_algorithm='tfidf')
```

### Graph-Boosted Search (GB-BM25)

A hybrid search combining BM25 with graph signals:

```python
# Standard search (uses BM25 under the hood)
results = processor.find_documents_for_query("query")

# Graph-boosted search (adds PageRank + proximity signals)
results = processor.graph_boosted_search(
    "query",
    pagerank_weight=0.3,   # Weight for term importance (0-1)
    proximity_weight=0.2   # Weight for connected terms (0-1)
)
```

**GB-BM25 combines:**
1. BM25 base score (term relevance)
2. PageRank boost (important terms rank higher)
3. Proximity boost (connected query terms boost documents)
4. Coverage boost (documents matching more terms rank higher)

---

## Performance Considerations

1. **Use `get_by_id()` for ID lookups** - O(1) vs O(n) iteration
2. **Batch document additions** with `add_documents_batch()` for bulk imports
3. **Use incremental updates** with `add_document_incremental()` for live systems
4. **Cache query expansions** when processing multiple similar queries
5. **Pre-compute chunks** in `find_passages_batch()` to avoid redundant work
6. **Use `fast_find_documents()`** for ~2-3x faster search on large corpora
7. **Pre-build index** with `build_search_index()` for fastest repeated queries
8. **Watch for O(n²) patterns** in loops over connections—use limits like `max_bigrams_per_term`
9. **Use `graph_boosted_search()`** for hybrid scoring with PageRank signals

---

## Code Search Capabilities

### Code-Aware Tokenization
```python
# Enable identifier splitting for code search
tokenizer = Tokenizer(split_identifiers=True)
tokens = tokenizer.tokenize("getUserCredentials")
# ['getusercredentials', 'get', 'user', 'credentials']
```

### Programming Concept Expansion
```python
# Expand queries with programming synonyms (get/fetch/load)
results = processor.expand_query("fetch data", use_code_concepts=True)
# Or use the convenience method
results = processor.expand_query_for_code("fetch data")
```

### Intent-Based Search
```python
# Parse natural language queries
parsed = processor.parse_intent_query("where do we handle authentication?")
# {'intent': 'location', 'action': 'handle', 'subject': 'authentication', ...}

# Search with intent understanding
results = processor.search_by_intent("how do we validate input?")
```

### Semantic Fingerprinting
```python
# Compare code similarity
fp1 = processor.get_fingerprint(code_block_1)
fp2 = processor.get_fingerprint(code_block_2)
comparison = processor.compare_fingerprints(fp1, fp2)
explanation = processor.explain_similarity(fp1, fp2)
```

### Fast Search
```python
# Fast document search (~2-3x faster)
results = processor.fast_find_documents("authentication")

# Pre-built index for fastest search
index = processor.build_search_index()
results = processor.search_with_index("query", index)
```

---

## Debugging Tips

### Inspecting Layer State
```python
processor = CorticalTextProcessor()
processor.process_document("test", "Neural networks process data.")
processor.compute_all()

# Check layer sizes
for layer_enum, layer in processor.layers.items():
    print(f"{layer_enum.name}: {layer.column_count()} minicolumns")

# Inspect a specific minicolumn
col = processor.layers[CorticalLayer.TOKENS].get_minicolumn("neural")
print(f"PageRank: {col.pagerank}")
print(f"TF-IDF: {col.tfidf}")
print(f"Connections: {len(col.lateral_connections)}")
print(f"Documents: {col.document_ids}")
```

### Tracing Query Expansion
```python
expanded = processor.expand_query("neural networks", max_expansions=10)
for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    print(f"  {term}: {weight:.3f}")
```

### Checking Semantic Relations
```python
processor.extract_corpus_semantics()
for t1, rel, t2, weight in processor.semantic_relations[:10]:
    print(f"{t1} --{rel}--> {t2} ({weight:.2f})")
```

### Profiling Performance
```bash
# Profile full analysis phases with timeout detection
python scripts/profile_full_analysis.py

# This reveals which phases are slow and helps identify O(n²) bottlenecks
```

---

## Observability and Metrics

The processor includes built-in observability features for tracking performance.

### Enable metrics collection
```python
processor = CorticalTextProcessor(enable_metrics=True)
processor.process_document("doc1", "Neural networks process data.")
processor.compute_all()
processor.find_documents_for_query("neural networks")
print(processor.get_metrics_summary())
```

### Access metrics programmatically
```python
metrics = processor.get_metrics()

if "compute_all" in metrics:
    stats = metrics["compute_all"]
    print(f"Average: {stats['avg_ms']:.2f}ms")
    print(f"Count: {stats['count']}")

if "query_cache_hits" in metrics:
    hits = metrics["query_cache_hits"]["count"]
    misses = metrics["query_cache_misses"]["count"]
    hit_rate = hits / (hits + misses) * 100
    print(f"Cache hit rate: {hit_rate:.1f}%")
```

### Automatically timed operations
- `compute_all()` and all compute phases (PageRank, TF-IDF, clustering, etc.)
- `process_document()` with doc_id context
- `find_documents_for_query()` with query context
- `save()` operations
- Query cache hits/misses via `expand_query_cached()`

### Control metrics collection
```python
processor.disable_metrics()   # Disable temporarily
processor.enable_metrics()    # Re-enable
processor.reset_metrics()     # Reset all metrics
processor.record_metric("api_calls", 10)  # Custom metrics
```

### Demo
```bash
python examples/observability_demo.py
```
