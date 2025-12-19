# Query Optimization Guide

Techniques for getting the best results from the Cortical Text Processor.

## Understanding Query Expansion

When you submit a query, it gets expanded before searching:

```python
# Original query
query = "database performance"

# After expansion
expanded = {
    "database": 1.0,      # Original term
    "db": 0.8,            # Abbreviation
    "sql": 0.6,           # Related technology
    "query": 0.5,         # Co-occurring term
    "performance": 1.0,   # Original term
    "speed": 0.7,         # Synonym
    "optimization": 0.6,  # Related concept
    "latency": 0.5,       # Related metric
}
```

## Query Formulation Tips

### Be Specific

```python
# Too broad - many irrelevant results
processor.find_documents_for_query("error")

# Better - more context
processor.find_documents_for_query("authentication error handling")
```

### Use Domain Terms

```python
# Generic
processor.find_documents_for_query("speed up code")

# Domain-specific (will match technical discussions)
processor.find_documents_for_query("optimize performance bottleneck")
```

### Combine Concepts

```python
# Single concept
processor.find_documents_for_query("testing")

# Multiple related concepts (better precision)
processor.find_documents_for_query("unit testing mocking assertions")
```

## Search Methods Comparison

### Standard Search

```python
results = processor.find_documents_for_query(
    query_text="neural networks",
    top_n=10
)
```
- Uses query expansion
- BM25 scoring
- Good for general use

### Fast Search

```python
results = processor.fast_find_documents(
    query_text="neural networks",
    top_n=10
)
```
- 2-3x faster
- Reduced expansion
- Good for autocomplete, live search

### Graph-Boosted Search

```python
results = processor.graph_boosted_search(
    query_text="neural networks",
    pagerank_weight=0.3,
    proximity_weight=0.2
)
```
- Combines BM25 with graph signals
- Considers term importance (PageRank)
- Boosts documents with connected terms
- Best for comprehensive search

### Intent-Based Search

```python
results = processor.search_by_intent(
    "where do we handle authentication?"
)
```
- Parses natural language queries
- Understands question types (where, how, what)
- Maps to relevant code locations

## Passage Retrieval for RAG

For retrieval-augmented generation, use passage search:

```python
passages = processor.find_passages_for_query(
    query_text="explain the login process",
    top_n=5,
    chunk_size=200,
    chunk_overlap=50
)

for doc_id, text, start, end, score in passages:
    print(f"[{doc_id}:{start}-{end}] {text[:100]}...")
```

### Chunk Size Guidelines

| Use Case | Chunk Size | Overlap |
|----------|------------|---------|
| Short answers | 100-150 | 25 |
| Explanations | 200-300 | 50 |
| Full context | 400-500 | 100 |
| Code snippets | 150-200 | 30 |

## Performance Optimization

### Build Search Index

For repeated queries, pre-build the index:

```python
# Build once
index = processor.build_search_index()

# Use many times (fastest)
results = processor.search_with_index("query", index)
```

### Batch Processing

```python
# Instead of multiple single queries
for query in queries:
    results.append(processor.find_documents_for_query(query))

# Use batch (more efficient)
all_results = processor.find_documents_batch(queries, top_n=5)
```

### Query Caching

```python
# Enable caching for repeated expansions
expanded = processor.expand_query_cached(query_text)
```

## Debugging Queries

### See Query Expansion

```python
expanded = processor.expand_query(
    query_text="machine learning",
    max_expansions=20
)
for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    print(f"  {term}: {weight:.3f}")
```

### Understand Scoring

```python
# Get detailed scoring breakdown
results = processor.find_documents_for_query(
    query_text="authentication",
    top_n=5,
    explain=True  # Returns scoring details
)
```

### Check Term Coverage

```python
# See which query terms matched each document
for doc_id, score in results:
    coverage = processor.get_query_coverage(query_text, doc_id)
    print(f"{doc_id}: {coverage['matched_terms']}")
```

## Common Pitfalls

### Over-Expansion

```python
# Too many expansions dilute relevance
processor.expand_query(query, max_expansions=100)  # Too many

# Reasonable limit
processor.expand_query(query, max_expansions=10)   # Better
```

### Ignoring Stopwords

```python
# Query dominated by stopwords
"the process of the authentication"  # Bad

# Content words only
"authentication process"  # Better
```

### Single-Word Queries

```python
# Too ambiguous
processor.find_documents_for_query("run")  # What kind of run?

# Add context
processor.find_documents_for_query("run tests pytest")  # Clear intent
```

## Summary

1. **Be specific** - Add context to narrow results
2. **Use domain terms** - Technical vocabulary matches better
3. **Choose the right method** - Fast vs comprehensive vs intent
4. **Tune chunk sizes** - Match your RAG use case
5. **Cache and batch** - For production performance
6. **Debug with expansion** - Understand what's being searched
