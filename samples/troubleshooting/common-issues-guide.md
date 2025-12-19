# Troubleshooting Guide

Solutions to common issues when using the Cortical Text Processor.

## Search Quality Issues

### Problem: Irrelevant Results

**Symptoms:** Search returns documents that don't match the intent.

**Causes and Solutions:**

1. **Query too broad**
   ```python
   # Bad: Single generic term
   results = processor.find_documents_for_query("error")

   # Better: Add context
   results = processor.find_documents_for_query("authentication error handling")
   ```

2. **Over-expansion diluting relevance**
   ```python
   # Check what's being searched
   expanded = processor.expand_query("database", max_expansions=20)
   print(expanded)  # Too many expansions?

   # Limit expansions
   expanded = processor.expand_query("database", max_expansions=5)
   ```

3. **Missing domain vocabulary**
   ```python
   # The processor doesn't know your domain terms
   # Solution: Add custom synonyms via code_concepts
   from cortical.code_concepts import CodeConceptExpander

   expander = CodeConceptExpander()
   expander.add_synonym_group(['api', 'endpoint', 'route', 'handler'])
   ```

### Problem: Missing Expected Results

**Symptoms:** Document should match but doesn't appear in results.

**Diagnostic steps:**

```python
# 1. Check if document is indexed
doc_ids = processor.get_document_ids()
print("my_doc" in doc_ids)

# 2. Check term presence
layer0 = processor.get_layer(CorticalLayer.TOKENS)
col = layer0.get_minicolumn("expected_term")
if col:
    print(f"Term found in docs: {col.document_ids}")

# 3. Check query expansion
expanded = processor.expand_query("my query")
print(f"Searching for: {list(expanded.keys())}")

# 4. Verify computation state
stale = processor.get_stale_computations()
if stale:
    print(f"Recompute needed: {stale}")
    processor.compute_all()
```

### Problem: Test Files Ranking Higher Than Implementation

**Symptoms:** Searching for "authentication" returns `test_auth.py` before `auth.py`.

**Solution:**
```python
# Use test file penalty
results = processor.find_documents_for_query(
    "authentication",
    test_file_penalty=0.5  # Reduce test file scores
)
```

## Performance Issues

### Problem: Slow Initial Indexing

**Symptoms:** `compute_all()` takes minutes on large corpus.

**Diagnostic:**
```python
import time

# Time each phase
start = time.time()
processor.compute_tfidf()
print(f"TF-IDF: {time.time() - start:.2f}s")

start = time.time()
processor.compute_importance()
print(f"PageRank: {time.time() - start:.2f}s")

start = time.time()
processor.compute_bigram_connections()
print(f"Bigrams: {time.time() - start:.2f}s")
```

**Common culprits:**

1. **Bigram explosion** - Common terms create millions of pairs
   ```python
   config = CorticalConfig(
       max_bigrams_per_term=100,    # Limit per-term bigrams
       max_bigrams_per_doc=500,     # Limit per-document bigrams
   )
   ```

2. **Semantic extraction** - Too many similarity pairs
   ```python
   processor.extract_corpus_semantics(
       max_similarity_pairs=100000,  # Limit pairs
       min_context_keys=3,           # Require more evidence
   )
   ```

### Problem: Slow Queries

**Symptoms:** Each query takes >100ms.

**Solutions:**

1. **Use fast search**
   ```python
   # ~2-3x faster, slightly reduced expansion
   results = processor.fast_find_documents("query")
   ```

2. **Pre-build search index**
   ```python
   # Build once
   index = processor.build_search_index()

   # Query many times (fastest)
   results = processor.search_with_index("query", index)
   ```

3. **Cache query expansions**
   ```python
   # For repeated similar queries
   expanded = processor.expand_query_cached("query")
   ```

### Problem: High Memory Usage

**Symptoms:** Process uses gigabytes of RAM.

**Diagnostic:**
```python
import sys

def get_size(obj):
    """Approximate object size in MB."""
    return sys.getsizeof(obj) / (1024 * 1024)

for layer_enum, layer in processor.layers.items():
    print(f"{layer_enum.name}: {layer.column_count()} columns")
```

**Solutions:**

1. **Use chunk-based storage** for large corpora
2. **Remove unused documents** after analysis
3. **Limit bigram/concept generation** via config

## Persistence Issues

### Problem: Load Fails with Version Error

**Symptoms:** `ValueError: Unsupported state version`

**Cause:** Saved with older/newer version of the library.

**Solution:**
```python
# Check saved version
import json

with open("corpus_state/metadata.json") as f:
    metadata = json.load(f)
    print(f"Saved version: {metadata.get('state_version')}")
    print(f"Current version: {STATE_VERSION}")

# Migration may be needed
# See docs/persistence-migration.md
```

### Problem: Pickle Security Warning

**Symptoms:** `DeprecationWarning: Pickle format is deprecated`

**Solution:** Migrate to JSON format
```python
# Load old pickle
processor = CorticalTextProcessor.load("old_corpus.pkl")

# Save as JSON (secure, recommended)
processor.save("new_corpus")  # Creates directory with JSON files
```

### Problem: Save Takes Too Long

**Symptoms:** `save()` takes minutes.

**Solution:** Use incremental chunk-based storage
```bash
python scripts/index_codebase.py --incremental --use-chunks
```

## Staleness Issues

### Problem: Outdated Scores After Adding Documents

**Symptoms:** New documents don't appear in search results.

**Cause:** Computations are stale after document changes.

**Diagnostic:**
```python
stale = processor.get_stale_computations()
print(f"Stale: {stale}")
# Output: {'tfidf', 'pagerank', 'concepts', ...}
```

**Solutions:**

1. **Use incremental updates**
   ```python
   # Updates TF-IDF automatically
   processor.add_document_incremental(doc_id, text)
   ```

2. **Recompute specific phases**
   ```python
   processor.compute_tfidf()      # Just TF-IDF
   processor.compute_importance() # Just PageRank
   ```

3. **Full recomputation**
   ```python
   processor.compute_all()  # Only recomputes stale phases
   ```

## Configuration Issues

### Problem: BM25 Results Seem Wrong

**Symptoms:** Long documents always rank higher (or lower).

**Cause:** BM25 parameters not tuned for your corpus.

**Solution:**
```python
# Default parameters
config = CorticalConfig(
    scoring_algorithm='bm25',
    bm25_k1=1.2,  # Term frequency saturation (try 0.5-2.0)
    bm25_b=0.75,  # Length normalization (try 0.0-1.0)
)

# For very short documents (e.g., code snippets)
config = CorticalConfig(
    bm25_k1=1.5,  # Higher saturation
    bm25_b=0.3,   # Less length penalty
)

# For long documents (e.g., articles)
config = CorticalConfig(
    bm25_k1=1.0,  # Lower saturation
    bm25_b=0.9,   # More length normalization
)
```

### Problem: Too Many Stopwords Removed

**Symptoms:** Important query terms being filtered.

**Solution:**
```python
from cortical.tokenizer import Tokenizer

# Check what's being removed
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("the API")
print(tokens)  # ['api'] - "the" removed

# Add custom stopwords or allowlist
tokenizer = Tokenizer(
    additional_stopwords=['fig', 'table'],  # Domain-specific
    stopword_allowlist=['the']              # Keep if needed
)
```

## Debug Techniques

### Verbose Query Analysis

```python
def debug_query(processor, query: str):
    """Full diagnostic on a query."""
    print(f"Query: {query}")
    print("-" * 50)

    # Expansion
    expanded = processor.expand_query(query)
    print("Expanded terms:")
    for term, weight in sorted(expanded.items(), key=lambda x: -x[1])[:10]:
        print(f"  {term}: {weight:.3f}")

    # Results with explanation
    results = processor.find_documents_for_query(query, top_n=5, explain=True)
    print("\nResults:")
    for doc_id, score in results:
        coverage = processor.get_query_coverage(query, doc_id)
        print(f"  {doc_id}: {score:.3f}")
        print(f"    Matched: {coverage.get('matched_terms', [])}")

debug_query(processor, "authentication")
```

### Minicolumn Inspection

```python
def inspect_term(processor, term: str):
    """Inspect a term's minicolumn."""
    layer0 = processor.get_layer(CorticalLayer.TOKENS)
    col = layer0.get_minicolumn(term)

    if not col:
        print(f"Term '{term}' not in corpus")
        return

    print(f"Term: {term}")
    print(f"  PageRank: {col.pagerank:.6f}")
    print(f"  TF-IDF: {col.tfidf:.4f}")
    print(f"  Documents: {len(col.document_ids)}")
    print(f"  Lateral connections: {len(col.lateral_connections)}")

    # Top connections
    sorted_conn = sorted(col.lateral_connections.items(), key=lambda x: -x[1])
    print("  Top connected terms:")
    for other_id, weight in sorted_conn[:5]:
        print(f"    {other_id}: {weight:.3f}")

inspect_term(processor, "neural")
```

## Getting Help

1. Check the [[query-optimization-guide.md]] for query formulation tips
2. Review [[graph-algorithms-primer.md]] for algorithm details
3. Read the docstrings: `help(processor.find_documents_for_query)`
4. Enable metrics: `processor = CorticalTextProcessor(enable_metrics=True)`
