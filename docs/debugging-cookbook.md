# Debugging Cookbook

A practical guide for debugging common issues with the Cortical Text Processor.

---

## Table of Contents

1. [Inspecting Layer State](#inspecting-layer-state)
2. [Tracing Query Expansion](#tracing-query-expansion)
3. [Debugging Ranking Issues](#debugging-ranking-issues)
4. [Profiling Performance](#profiling-performance)
5. [Common Error Messages and Solutions](#common-error-messages-and-solutions)
6. [Search Troubleshooting](#search-troubleshooting)
7. [Data Integrity Checks](#data-integrity-checks)

---

## Inspecting Layer State

### Problem: Need to understand what's in the processor

**Basic layer inspection:**

```python
from cortical import CorticalTextProcessor, CorticalLayer

processor = CorticalTextProcessor()
processor.process_document("test", "Neural networks process data.")
processor.compute_all()

# Check layer sizes
for layer_enum, layer in processor.layers.items():
    print(f"{layer_enum.name}: {layer.column_count()} minicolumns")
```

**Expected output:**
```
TOKENS: 4
BIGRAMS: 3
CONCEPTS: 1
DOCUMENTS: 1
```

### Problem: Inspect a specific term

```python
from cortical import CorticalLayer

layer0 = processor.get_layer(CorticalLayer.TOKENS)
col = layer0.get_minicolumn("neural")

if col:
    print(f"Term: {col.content}")
    print(f"PageRank: {col.pagerank:.4f}")
    print(f"TF-IDF: {col.tfidf:.4f}")
    print(f"Connections: {len(col.lateral_connections)}")
    print(f"Documents: {col.document_ids}")
    print(f"Activation: {col.activation}")
else:
    print("Term 'neural' not found in corpus")
```

### Problem: Find top terms by PageRank

```python
layer0 = processor.get_layer(CorticalLayer.TOKENS)

# Get top 20 PageRank terms
top_terms = sorted(
    [(col.content, col.pagerank) for col in layer0],
    key=lambda x: -x[1]
)[:20]

print("Top PageRank terms:")
for term, score in top_terms:
    print(f"  {term}: {score:.4f}")
```

### Problem: Check document metadata

```python
# Get all document IDs
doc_ids = list(processor.documents.keys())
print(f"Total documents: {len(doc_ids)}")

# Inspect specific document
doc_id = doc_ids[0]
metadata = processor.get_document_metadata(doc_id)
print(f"\nDocument: {doc_id}")
print(f"Metadata: {metadata}")

# Get document content
content = processor.documents.get(doc_id)
print(f"Content preview: {content[:200]}...")
```

### Problem: Inspect concept clusters

```python
layer2 = processor.get_layer(CorticalLayer.CONCEPTS)
layer0 = processor.get_layer(CorticalLayer.TOKENS)

print(f"Total concept clusters: {layer2.column_count()}")

for concept_col in layer2:
    print(f"\nConcept: {concept_col.content}")

    # Get member terms
    member_terms = []
    for token_id in concept_col.feedforward_connections:
        token_col = layer0.get_by_id(token_id)
        if token_col:
            member_terms.append(token_col.content)

    print(f"  Members ({len(member_terms)}): {', '.join(member_terms[:10])}")
    if len(member_terms) > 10:
        print(f"  ... and {len(member_terms) - 10} more")
```

---

## Tracing Query Expansion

### Problem: Query returns no results or wrong results

**Step 1: Check if query terms exist in corpus**

```python
query = "neural networks"
tokens = processor.tokenizer.tokenize(query)
layer0 = processor.get_layer(CorticalLayer.TOKENS)

print(f"Query: {query}")
print(f"Tokenized: {tokens}")
print("\nTerm existence:")
for term in tokens:
    col = layer0.get_minicolumn(term)
    if col:
        print(f"  ✓ '{term}' found - {len(col.document_ids)} docs")
    else:
        print(f"  ✗ '{term}' NOT in corpus")
```

**Step 2: Trace query expansion**

```python
expanded = processor.expand_query(query, max_expansions=10)

print(f"\nExpanded query terms:")
for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    marker = "★" if term in tokens else " "
    print(f"  {marker} {term}: {weight:.3f}")
```

**Step 3: Test with different expansion settings**

```python
# Conservative expansion
conservative = processor.expand_query(
    query,
    max_expansions=3,
    use_variants=False
)
print(f"Conservative: {len(conservative)} terms")

# Aggressive expansion
aggressive = processor.expand_query(
    query,
    max_expansions=20,
    use_variants=True,
    use_code_concepts=True
)
print(f"Aggressive: {len(aggressive)} terms")

# Compare results
conservative_results = processor.find_documents_for_query(
    query,
    max_expansions=3
)
aggressive_results = processor.find_documents_for_query(
    query,
    max_expansions=20
)
print(f"\nConservative: {len(conservative_results)} results")
print(f"Aggressive: {len(aggressive_results)} results")
```

### Problem: Understanding why a term expanded to others

```python
term = "neural"
layer0 = processor.get_layer(CorticalLayer.TOKENS)
col = layer0.get_minicolumn(term)

if col:
    print(f"Expansion sources for '{term}':")

    # Lateral connections
    print("\nLateral connections (co-occurrence):")
    for neighbor_id, weight in sorted(
        col.lateral_connections.items(),
        key=lambda x: -x[1]
    )[:10]:
        neighbor = layer0.get_by_id(neighbor_id)
        if neighbor:
            print(f"  {neighbor.content}: {weight:.2f}")

    # Typed connections (semantic relations)
    print("\nSemantic relations:")
    for neighbor_id, edge in list(col.typed_connections.items())[:10]:
        neighbor = layer0.get_by_id(neighbor_id)
        if neighbor:
            print(f"  {edge.relation_type}: {neighbor.content} (conf: {edge.confidence:.2f})")
```

---

## Debugging Ranking Issues

### Problem: Wrong documents ranking highly

**Step 1: Check document TF-IDF scores**

```python
query = "neural networks"
tokens = processor.tokenizer.tokenize(query)
layer0 = processor.get_layer(CorticalLayer.TOKENS)

# Get TF-IDF for each document
doc_scores = {}
for doc_id in processor.documents.keys():
    score = 0.0
    for term in tokens:
        col = layer0.get_minicolumn(term)
        if col:
            score += col.tfidf_per_doc.get(doc_id, 0.0)
    doc_scores[doc_id] = score

# Show top scoring documents
print("Top documents by TF-IDF:")
for doc_id, score in sorted(doc_scores.items(), key=lambda x: -x[1])[:10]:
    print(f"  {doc_id}: {score:.3f}")
```

**Step 2: Compare with search results**

```python
results = processor.find_documents_for_query(query, top_n=10)

print("\nActual search results:")
for doc_id, score in results:
    tfidf_score = doc_scores.get(doc_id, 0.0)
    print(f"  {doc_id}: {score:.3f} (base TF-IDF: {tfidf_score:.3f})")
```

### Problem: Test files ranking higher than implementations

**Check if test file penalty is working:**

```python
# Find documents
results_default = processor.find_documents_for_query(
    "neural network implementation",
    top_n=10
)

results_with_penalty = processor.find_documents_for_query(
    "neural network implementation",
    top_n=10,
    test_file_penalty=0.5  # Penalize test files
)

print("Default results:")
for doc_id, score in results_default:
    is_test = 'test' in doc_id.lower()
    marker = "[TEST]" if is_test else ""
    print(f"  {doc_id}: {score:.3f} {marker}")

print("\nWith test file penalty:")
for doc_id, score in results_with_penalty:
    is_test = 'test' in doc_id.lower()
    marker = "[TEST]" if is_test else ""
    print(f"  {doc_id}: {score:.3f} {marker}")
```

### Problem: Understanding passage ranking

```python
passages = processor.find_passages_for_query(
    "neural network training",
    top_n=5,
    chunk_size=200,
    overlap=50
)

print("Passage ranking breakdown:")
for i, (text, doc_id, start, end, score) in enumerate(passages):
    print(f"\n{i+1}. [{doc_id}:{start}-{end}] Score: {score:.3f}")
    print(f"   Text: {text[:100]}...")

    # Check which query terms appear in passage
    tokens = processor.tokenizer.tokenize("neural network training")
    text_lower = text.lower()
    matches = [t for t in tokens if t in text_lower]
    print(f"   Matched terms: {matches}")
```

---

## Profiling Performance

### Problem: `compute_all()` is slow

**Use the profiling script:**

```bash
python scripts/profile_full_analysis.py
```

This shows timing for each phase:
- `tfidf`: TF-IDF computation
- `bigram_connections`: Bigram lateral connections
- `pagerank`: PageRank importance
- `semantics`: Semantic relation extraction
- `louvain`: Concept clustering

**Identify the bottleneck:**

```python
import time
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()
# ... add documents ...

# Time each phase
phases = {
    'tfidf': lambda: processor.compute_tfidf(verbose=False),
    'bigram_connections': lambda: processor.compute_bigram_connections(verbose=False),
    'pagerank': lambda: processor.compute_importance(verbose=False),
    'doc_connections': lambda: processor.compute_document_connections(verbose=False),
    'semantics': lambda: processor.extract_corpus_semantics(verbose=False),
    'concepts': lambda: processor.build_concept_clusters(verbose=False),
}

for phase, func in phases.items():
    start = time.time()
    func()
    duration = time.time() - start
    print(f"{phase}: {duration:.2f}s")
```

### Problem: Search is slow

**Option 1: Use fast search**

```python
import time

query = "neural networks"

# Standard search
start = time.time()
results1 = processor.find_documents_for_query(query, top_n=5)
time1 = time.time() - start

# Fast search
start = time.time()
results2 = processor.fast_find_documents(query, top_n=5)
time2 = time.time() - start

print(f"Standard search: {time1:.4f}s")
print(f"Fast search: {time2:.4f}s")
print(f"Speedup: {time1/time2:.1f}x")
```

**Option 2: Pre-build search index**

```python
# Build index once
start = time.time()
index = processor.build_search_index()
build_time = time.time() - start
print(f"Index build time: {build_time:.2f}s")

# Fast searches
queries = ["neural networks", "machine learning", "deep learning"]
start = time.time()
for query in queries:
    results = processor.search_with_index(query, index, top_n=5)
total_time = time.time() - start

print(f"3 searches: {total_time:.4f}s ({total_time/3*1000:.2f}ms each)")
```

### Problem: Memory usage too high

**Check corpus size:**

```python
summary = processor.get_corpus_summary()

print(f"Documents: {summary['documents']}")
print(f"Total minicolumns: {summary['total_columns']}")
print(f"\nLayer breakdown:")
for layer_num, stats in summary['layer_stats'].items():
    print(f"  Layer {layer_num}: {stats['minicolumns']} minicolumns")

# Check connection counts
layer0 = processor.get_layer(CorticalLayer.TOKENS)
total_connections = sum(
    len(col.lateral_connections) for col in layer0
)
print(f"\nTotal lateral connections: {total_connections}")
```

---

## Common Error Messages and Solutions

### Error: `ValueError: Query is empty after tokenization`

**Cause:** Query contains only stop words or special characters.

**Solution:**

```python
# Check what tokenization produces
query = "the and or"
tokens = processor.tokenizer.tokenize(query)
print(f"Tokens: {tokens}")  # Empty list - all stop words

# Use meaningful terms
query = "neural network"
tokens = processor.tokenizer.tokenize(query)
print(f"Tokens: {tokens}")  # ['neural', 'network']
```

### Error: `KeyError` when accessing layer

**Cause:** Layer doesn't exist or hasn't been built.

**Solution:**

```python
from cortical import CorticalLayer

# Check if concepts layer exists
try:
    layer2 = processor.get_layer(CorticalLayer.CONCEPTS)
    print(f"Concepts: {layer2.column_count()}")
except KeyError:
    print("Concepts layer not built - run build_concept_clusters()")
    processor.build_concept_clusters()
```

### Error: Stale computation warnings

**Cause:** Computations are outdated after adding documents.

**Solution:**

```python
# Check what's stale
stale = processor.get_stale_computations()
print(f"Stale computations: {stale}")

# Recompute as needed
if 'tfidf' in stale:
    processor.compute_tfidf()
if 'pagerank' in stale:
    processor.compute_importance()

# Or recompute everything
processor.compute_all()
```

### Error: `AttributeError: 'NoneType' object has no attribute...`

**Cause:** Term lookup returned `None`.

**Solution:**

```python
# Wrong - may crash if term doesn't exist
term = "nonexistent"
col = layer0.get_minicolumn(term)
print(col.pagerank)  # AttributeError if col is None

# Correct - check for None
col = layer0.get_minicolumn(term)
if col:
    print(f"{term}: {col.pagerank}")
else:
    print(f"'{term}' not found in corpus")
```

---

## Search Troubleshooting

### Problem: No results for valid query

**Checklist:**

1. **Are query terms in the corpus?**
   ```python
   tokens = processor.tokenizer.tokenize("your query")
   layer0 = processor.get_layer(CorticalLayer.TOKENS)
   for term in tokens:
       exists = layer0.get_minicolumn(term) is not None
       print(f"{term}: {exists}")
   ```

2. **Try with query expansion:**
   ```python
   results = processor.find_documents_for_query(
       "your query",
       use_expansion=True,
       max_expansions=20
   )
   ```

3. **Check if TF-IDF is computed:**
   ```python
   if processor.is_stale(processor.COMP_TFIDF):
       print("TF-IDF is stale - recomputing")
       processor.compute_tfidf()
   ```

### Problem: Too many irrelevant results

**Solution 1: Reduce expansion**

```python
# More conservative expansion
results = processor.find_documents_for_query(
    query,
    max_expansions=3,
    use_variants=False
)
```

**Solution 2: Increase top_n to find where relevance drops**

```python
results = processor.find_documents_for_query(query, top_n=20)
for i, (doc_id, score) in enumerate(results):
    print(f"{i+1}. {doc_id}: {score:.3f}")
```

### Problem: Passage retrieval not finding relevant chunks

**Debug chunk boundaries:**

```python
from cortical.query import chunk_text

doc_id = "problem_doc"
content = processor.documents.get(doc_id, "")

chunks = list(chunk_text(content, chunk_size=200, overlap=50))
print(f"Document '{doc_id}' chunked into {len(chunks)} pieces:")
for i, (text, start, end) in enumerate(chunks[:5]):
    print(f"\n{i+1}. [{start}-{end}]:")
    print(f"   {text[:100]}...")
```

---

## Data Integrity Checks

### Problem: Verify corpus consistency

**Check document-term consistency:**

```python
layer0 = processor.get_layer(CorticalLayer.TOKENS)
layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)

# Verify all documents in layer 3 exist
print("Checking document layer consistency...")
for doc_col in layer3:
    doc_id = doc_col.content
    if doc_id not in processor.documents:
        print(f"  ERROR: Document {doc_id} in layer but not in processor.documents")

# Verify token-document references
print("\nChecking token-document references...")
errors = 0
for col in layer0:
    for doc_id in col.document_ids:
        if doc_id not in processor.documents:
            print(f"  ERROR: Term '{col.content}' references missing doc {doc_id}")
            errors += 1

print(f"Found {errors} errors")
```

### Problem: Check for orphaned minicolumns

```python
layer0 = processor.get_layer(CorticalLayer.TOKENS)

orphans = []
for col in layer0:
    if not col.document_ids:
        orphans.append(col.content)

if orphans:
    print(f"Found {len(orphans)} orphaned terms:")
    print(orphans[:20])
else:
    print("No orphaned terms found")
```

### Problem: Verify PageRank scores

```python
layer0 = processor.get_layer(CorticalLayer.TOKENS)

# PageRank should sum to ~1.0
total_pr = sum(col.pagerank for col in layer0)
print(f"Total PageRank: {total_pr:.4f} (should be ~1.0)")

# Check for zero PageRank (shouldn't happen after compute_importance)
zero_pr = [col.content for col in layer0 if col.pagerank == 0.0]
if zero_pr:
    print(f"Warning: {len(zero_pr)} terms with PageRank = 0")
```

---

## Quick Debugging Checklist

When something goes wrong:

1. **Check corpus is loaded:**
   ```python
   print(f"Documents: {len(processor.documents)}")
   ```

2. **Check computations are fresh:**
   ```python
   stale = processor.get_stale_computations()
   if stale: processor.compute_all()
   ```

3. **Inspect query tokenization:**
   ```python
   tokens = processor.tokenizer.tokenize(query)
   print(f"Tokenized: {tokens}")
   ```

4. **Check term existence:**
   ```python
   layer0 = processor.get_layer(CorticalLayer.TOKENS)
   for term in tokens:
       print(f"{term}: {layer0.get_minicolumn(term) is not None}")
   ```

5. **Test with minimal example:**
   ```python
   test = CorticalTextProcessor()
   test.process_document("test", "neural networks process data")
   test.compute_all()
   results = test.find_documents_for_query("neural")
   print(results)
   ```

---

## Advanced Debugging

### Enable verbose output

```python
# During processing
processor.process_document("doc1", "text", verbose=True)

# During computation
processor.compute_all(verbose=True)

# During search
results = processor.find_documents_for_query(
    "query",
    top_n=5,
    verbose=True
)
```

### Export for external inspection

```python
# Export to JSON for manual inspection
processor.export_to_json("debug_export.json")

# Load in external tool or Python
import json
with open("debug_export.json") as f:
    data = json.load(f)
    print(json.dumps(data['layers'][0][:5], indent=2))
```

### Compare two processors

```python
# Useful for testing changes
p1 = CorticalTextProcessor()
p2 = CorticalTextProcessor()

# Process with different configs
p1.process_document("doc1", "text")
p2.process_document("doc1", "text")

p1.compute_all()
p2.compute_all()

# Compare results
r1 = p1.find_documents_for_query("query")
r2 = p2.find_documents_for_query("query")

print(f"Processor 1: {r1}")
print(f"Processor 2: {r2}")
```

---

*See also: [Cookbook](cookbook.md) for usage patterns, [Query Guide](query-guide.md) for search optimization.*
