# Cortical Text Processor Cookbook

A practical guide to common patterns and recipes for using the Cortical Text Processor effectively.

---

## Table of Contents

1. [Document Processing Patterns](#document-processing-patterns)
2. [Search Optimization Recipes](#search-optimization-recipes)
3. [Corpus Maintenance Patterns](#corpus-maintenance-patterns)
4. [Query Expansion Tuning](#query-expansion-tuning)
5. [Clustering Configuration](#clustering-configuration)
6. [Performance Optimization](#performance-optimization)
7. [RAG Integration Patterns](#rag-integration-patterns)

---

## Document Processing Patterns

### Recipe 1: Batch Processing (Recommended)

**When to use:** Adding multiple documents at once (initial corpus loading, bulk imports).

```python
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()

# Prepare documents as list of (doc_id, content, metadata) tuples
documents = [
    ("doc1", "Neural networks process information.", {"source": "book1"}),
    ("doc2", "Deep learning enables pattern recognition.", {"source": "book1"}),
    ("doc3", "Machine learning algorithms learn from data.", {"source": "book2"}),
]

# Add all documents and recompute once
stats = processor.add_documents_batch(
    documents,
    recompute='full',  # 'full', 'tfidf', or 'none'
    verbose=True
)

print(f"Added {stats['documents_added']} documents")
```

**Expected outcome:**
- Single recomputation pass instead of per-document recomputation
- ~3-5x faster than calling `process_document()` in a loop

**Recomputation options:**
- `recompute='full'`: Slowest, most accurate (includes all graph algorithms)
- `recompute='tfidf'`: Fast, good for search quality
- `recompute='none'`: Fastest, but computations marked stale

---

### Recipe 2: Incremental Updates (Live Systems)

**When to use:** Adding documents to an already-built corpus (RAG systems, streaming data).

```python
# Start with existing corpus
processor = CorticalTextProcessor.load("corpus.pkl")

# Add new document without full recomputation
processor.add_document_incremental(
    "new_doc",
    "New document content.",
    metadata={"timestamp": "2025-12-10"},
    recompute='tfidf'  # Only recompute TF-IDF for search quality
)

# Later: full recomputation when needed
processor.recompute(level='full', verbose=True)
```

---

### Recipe 3: Document Removal

**When to use:** Delete outdated documents, remove duplicates.

```python
# Remove single document
result = processor.remove_document("old_doc", verbose=True)
print(f"Tokens affected: {result['tokens_affected']}")

# Remove multiple documents efficiently
doc_ids_to_remove = ["old_doc1", "old_doc2", "old_doc3"]
result = processor.remove_documents_batch(
    doc_ids_to_remove,
    recompute='tfidf',
    verbose=True
)
```

---

## Search Optimization Recipes

### Recipe 4: Choosing the Right Search Method

**Decision tree:**

```
Searching repeatedly on same corpus?
├─ YES → fast_find_documents() or build_search_index()
└─ NO  → find_documents_for_query()

Need text passages for RAG?
├─ YES → find_passages_for_query()
└─ NO  → find_documents_for_query()

Large corpus (1000+ docs)?
└─ YES → fast_find_documents() for ~2-3x speedup
```

---

### Recipe 5: Fast Document Search

**When to use:** Large corpora, need sub-100ms response time.

```python
# Fast search with candidate filtering
results = processor.fast_find_documents(
    "neural networks",
    top_n=5,
    candidate_multiplier=3,  # 5 * 3 = 15 candidates examined
    use_code_concepts=True   # Enable for code search
)

for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")
```

**Tuning `candidate_multiplier`:**
- `1`: Aggressive (may miss relevant documents)
- `3`: Balanced (recommended)
- `5`: Conservative (slower but higher recall)

---

### Recipe 6: Pre-Built Search Index (Fastest)

**When to use:** Repeated searching on stable corpus.

```python
# Build index once
index = processor.build_search_index()

# Use for fast searches
queries = ["neural networks", "machine learning", "deep learning"]
for query in queries:
    results = processor.search_with_index(query, index, top_n=5)
    print(f"{query}: {len(results)} results")
```

**Note:** Rebuild index after `add_documents_batch()` or `remove_document()`.

---

### Recipe 7: Passage Retrieval for RAG

**When to use:** Building retrieval-augmented generation systems.

```python
results = processor.find_passages_for_query(
    "neural network training",
    top_n=5,
    chunk_size=512,      # Characters per chunk
    overlap=128,         # Overlap between chunks
    use_expansion=True
)

# Results: (passage_text, doc_id, start_char, end_char, score)
for passage, doc_id, start, end, score in results:
    print(f"[{doc_id}:{start}-{end}] Score: {score:.3f}")
    print(passage[:100] + "...")
```

**Chunk size tuning:**
- `256`: Small, precise passages
- `512`: Balanced (recommended)
- `1024`: Large, more context

---

## Corpus Maintenance Patterns

### Recipe 8: Detecting Stale Computations

**When to use:** Understand what needs recomputation after changes.

```python
# Check what's stale
stale = processor.get_stale_computations()
print(f"Stale: {stale}")

if 'tfidf' in stale:
    print("TF-IDF scores are outdated - search quality affected")
    processor.compute_tfidf(verbose=True)

if 'pagerank' in stale:
    print("PageRank scores are outdated")
    processor.compute_importance(verbose=True)
```

---

### Recipe 9: Save and Load Corpus

**When to use:** Persist trained corpus for deployment.

```python
# Build and save
processor = CorticalTextProcessor()
processor.add_documents_batch(documents, recompute='full')
processor.save("production_corpus.pkl", verbose=True)

# Load in production
loaded = CorticalTextProcessor.load("production_corpus.pkl")
results = loaded.find_documents_for_query("query")
```

---

## Query Expansion Tuning

### Recipe 10: Understanding Expansion

```python
# See what expansion adds
expanded = processor.expand_query("neural", max_expansions=10)

print("Original term: neural")
print("Expanded with:")
for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    if term != "neural":
        print(f"  {term}: {weight:.3f}")
```

**Expansion sources:**
- **Lateral connections** (0.6x): Terms appearing near query term
- **Concept membership** (0.4x): Terms in same semantic cluster
- **Code concepts** (0.6x): Programming synonyms (get/fetch/load)

---

### Recipe 11: Tuning Expansion Parameters

```python
# Conservative expansion (higher precision)
conservative = processor.expand_query(
    "neural networks",
    max_expansions=3,
    use_variants=False
)

# Aggressive expansion (higher recall)
aggressive = processor.expand_query(
    "neural networks",
    max_expansions=20,
    use_variants=True,
    use_code_concepts=True
)

# Balanced (recommended)
balanced = processor.expand_query(
    "neural networks",
    max_expansions=10,
    use_variants=True
)
```

---

### Recipe 12: Multi-Hop Expansion

**When to use:** Discover distantly related terms through semantic relations.

```python
# Extract semantic relations first
processor.extract_corpus_semantics()

# Multi-hop expansion
expanded = processor.expand_query_multihop(
    "neural",
    max_hops=2,         # Follow 2 relation hops
    max_expansions=15,
    decay_factor=0.5    # Weight decreases per hop
)
```

---

## Clustering Configuration

### Recipe 13: Tuning Cluster Strictness

```python
# Strict clustering (more separate clusters)
processor.compute_all(
    build_concepts=True,
    cluster_strictness=1.0,
    bridge_weight=0.0
)

# Loose clustering (fewer, larger clusters)
processor.compute_all(
    build_concepts=True,
    cluster_strictness=0.5,
    bridge_weight=0.3
)
```

**Strictness guide:**
- `1.0`: Strict (more clusters, stronger topic separation)
- `0.5`: Balanced (recommended)
- `0.0`: Loose (fewer clusters, more topic mixing)

**Bridge weight effects:**
- `0.0`: No synthetic connections (isolated topics)
- `0.1-0.3`: Light bridging (recommended)
- `0.5+`: Strong bridging (may create spurious links)

---

## Performance Optimization

### Recipe 14: Profiling Corpus Size

```python
summary = processor.get_corpus_summary()

print(f"Documents: {summary['documents']}")
print(f"Total columns: {summary['total_columns']}")
print(f"Layer breakdown:")
print(f"  Tokens: {summary['layer_stats'].get(0, {}).get('minicolumns', 0)}")
print(f"  Bigrams: {summary['layer_stats'].get(1, {}).get('minicolumns', 0)}")

# Optimization strategy
if summary['documents'] < 100:
    print("Small corpus: use standard methods")
elif summary['documents'] < 1000:
    print("Medium corpus: consider fast_find_documents()")
else:
    print("Large corpus: use search index")
```

---

### Recipe 15: Query Cache Management

```python
# Enable query caching
processor.set_query_cache_size(100)

# Cached expansion (instant for repeated queries)
results1 = processor.expand_query_cached("neural networks")
results2 = processor.expand_query_cached("neural networks")  # From cache

# Clear cache when corpus changes
processor.clear_query_cache()
```

---

## RAG Integration Patterns

### Recipe 16: Simple RAG Backend

```python
def rag_retrieve(processor, query: str, top_n: int = 5) -> str:
    """Retrieve context for RAG system."""
    passages = processor.find_passages_for_query(
        query,
        top_n=top_n,
        chunk_size=512,
        overlap=128
    )

    context = "Context from knowledge base:\n\n"
    for passage, doc_id, _, _, score in passages:
        context += f"[{doc_id}] {passage}\n\n"

    return context

# Use in RAG loop
context = rag_retrieve(processor, "How do neural networks learn?")
# Pass to LLM with question
```

---

### Recipe 17: Multi-Stage RAG Ranking

**When to use:** Maximum quality ranking combining multiple signals.

```python
results = processor.multi_stage_rank(
    "neural networks",
    top_n=5,
    chunk_size=512,
    concept_boost=0.3  # Weight for concept relevance
)

for passage, doc_id, start, end, score, stages in results:
    print(f"[{doc_id}] Final: {score:.3f}")
    print(f"  Concept: {stages['concept_score']:.3f}")
    print(f"  Document: {stages['doc_score']:.3f}")
    print(f"  Passage: {stages['chunk_score']:.3f}")
```

---

## Quick Reference

| Task | Best Method |
|------|-------------|
| Multiple documents | `add_documents_batch()` |
| Incremental updates | `add_document_incremental()` |
| Document removal | `remove_documents_batch()` |
| General search | `find_documents_for_query()` |
| Large corpus search | `fast_find_documents()` |
| Repeated searches | `build_search_index()` |
| RAG passages | `find_passages_for_query()` |
| High-quality RAG | `multi_stage_rank()` |
| Query debugging | `expand_query()` |
| Intent search | `search_by_intent()` |

---

## Troubleshooting

### No Results Found

```python
# Check if query terms exist
layer0 = processor.get_layer(CorticalLayer.TOKENS)
for term in processor.tokenizer.tokenize(query):
    if not layer0.get_minicolumn(term):
        print(f"'{term}' not in corpus")

# Try with expansion
results = processor.find_documents_for_query(query, use_expansion=True)
```

### Search is Slow

```python
# Use fast search
results = processor.fast_find_documents(query, top_n=5)

# Or build index
index = processor.build_search_index()
results = processor.search_with_index(query, index)
```

### Stale Results

```python
# Check and recompute
stale = processor.get_stale_computations()
if stale:
    processor.recompute(level='full')
```

---

*See also: [Query Guide](query-guide.md) for detailed query formulation, [Claude Usage Guide](claude-usage.md) for AI agent usage.*
