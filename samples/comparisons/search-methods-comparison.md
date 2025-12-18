# Search Methods Comparison

A detailed comparison of search methods available in the Cortical Text Processor.

## Overview

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| Standard Search | Medium | High | General use |
| Fast Search | Fast | Medium | Autocomplete, live search |
| Graph-Boosted | Slower | Highest | Comprehensive results |
| Intent-Based | Medium | Variable | Natural language queries |
| Passage Search | Slow | High | RAG applications |

## Standard Search

```python
results = processor.find_documents_for_query("authentication", top_n=10)
```

### How It Works

1. **Tokenize query** - Split into terms, apply stemming
2. **Expand query** - Add synonyms and related terms from graph
3. **Score documents** - BM25 scoring for each expanded term
4. **Rank and return** - Sort by combined score

### Strengths

- Balanced speed and quality
- Good query expansion
- Handles synonyms well
- Predictable behavior

### Weaknesses

- Doesn't consider term importance (PageRank)
- Doesn't boost connected documents
- May miss non-obvious matches

### When to Use

- Most search use cases
- When consistent behavior is needed
- Interactive search with moderate latency tolerance

## Fast Search

```python
results = processor.fast_find_documents("authentication", top_n=10)
```

### How It Works

1. **Tokenize query** - Same as standard
2. **Limited expansion** - Only top-weighted expansions
3. **Optimized scoring** - Simplified BM25 calculation
4. **Early termination** - Stop once top_n confident results found

### Strengths

- 2-3x faster than standard
- Good for real-time applications
- Acceptable quality for most queries

### Weaknesses

- Reduced recall (fewer expansions)
- May miss edge case matches
- Less thorough ranking

### When to Use

- Autocomplete suggestions
- Live search-as-you-type
- High-throughput applications
- When latency < 20ms required

## Graph-Boosted Search (GB-BM25)

```python
results = processor.graph_boosted_search(
    "authentication",
    pagerank_weight=0.3,
    proximity_weight=0.2
)
```

### How It Works

1. **Standard search** - Base BM25 scores
2. **PageRank boost** - Documents with high-PageRank terms rank higher
3. **Proximity boost** - Documents with connected query terms get bonus
4. **Coverage boost** - Documents matching more expanded terms rank higher

### Score Composition

```
final_score = bm25_score * (1 + pagerank_boost + proximity_boost + coverage_boost)

where:
  pagerank_boost = pagerank_weight * avg(matched_term_pageranks)
  proximity_boost = proximity_weight * connected_term_bonus
  coverage_boost = (matched_terms / total_expanded_terms)
```

### Strengths

- Best overall relevance
- Surfaces important documents
- Rewards comprehensive matches
- Leverages graph structure

### Weaknesses

- Slowest method
- Requires computed PageRank
- More complex tuning
- May over-boost central terms

### When to Use

- Important searches where quality matters
- Research and exploration
- Finding authoritative documents
- When latency is not critical

## Intent-Based Search

```python
results = processor.search_by_intent("where do we handle authentication?")
```

### How It Works

1. **Parse intent** - Identify question type (where, how, what, why)
2. **Extract entities** - Find action verbs and subjects
3. **Map to patterns** - Match intent to code patterns
4. **Targeted search** - Search with intent-specific boosting

### Intent Types

| Question | Intent | Boost Pattern |
|----------|--------|---------------|
| "where" | location | File paths, class definitions |
| "how" | implementation | Functions, methods |
| "what" | definition | Classes, types, constants |
| "why" | reasoning | Comments, documentation |
| "when" | timing | Events, triggers, conditions |

### Strengths

- Natural language queries
- Understands user intent
- Maps questions to code locations
- Good for exploration

### Weaknesses

- Limited intent vocabulary
- May misparse complex questions
- Requires well-structured corpus
- Less predictable than keyword search

### When to Use

- Conversational interfaces
- Code exploration
- New developer onboarding
- "I don't know what to search for"

## Passage Search

```python
passages = processor.find_passages_for_query(
    "explain authentication flow",
    top_n=5,
    chunk_size=200,
    chunk_overlap=50
)
```

### How It Works

1. **Find documents** - Standard search for matching documents
2. **Chunk documents** - Split into overlapping passages
3. **Score passages** - Re-rank by passage-level relevance
4. **Return with context** - Include passage boundaries

### Strengths

- Precise answer location
- Perfect for RAG pipelines
- Avoids returning entire documents
- Controllable chunk size

### Weaknesses

- Slowest method (chunking overhead)
- May split important context
- Chunk size tuning required
- More memory intensive

### When to Use

- RAG (Retrieval-Augmented Generation)
- Question answering systems
- When precise snippets needed
- Documentation search

## Benchmark Results

Testing on 150 document corpus, 100 queries:

| Method | Avg Latency | P95 Latency | MRR | Recall@10 |
|--------|-------------|-------------|-----|-----------|
| Standard | 25ms | 45ms | 0.72 | 0.85 |
| Fast | 8ms | 15ms | 0.65 | 0.78 |
| Graph-Boosted | 42ms | 78ms | 0.81 | 0.91 |
| Intent | 31ms | 55ms | 0.68 | 0.82 |
| Passage | 85ms | 150ms | 0.76 | 0.88 |

## Decision Flowchart

```
                           Need search?
                               │
                    ┌──────────┴──────────┐
                    │                      │
              Latency critical?      Quality critical?
                    │                      │
              ┌─────┴─────┐          ┌─────┴─────┐
              │           │          │           │
            Yes          No        Yes          No
              │           │          │           │
        Fast Search   Standard   Graph-Boosted  │
                                               │
                                     Natural language?
                                          │
                                    ┌─────┴─────┐
                                   Yes          No
                                    │           │
                              Intent-Based   Standard
```

## Combining Methods

For best results, combine methods based on context:

```python
def smart_search(processor, query: str, context: dict):
    """Choose search method based on context."""

    # Autocomplete
    if context.get('autocomplete'):
        return processor.fast_find_documents(query, top_n=5)

    # Natural language question
    if query.endswith('?') or query.startswith(('how', 'where', 'what', 'why')):
        return processor.search_by_intent(query)

    # RAG pipeline
    if context.get('need_passages'):
        return processor.find_passages_for_query(query)

    # Important search
    if context.get('thorough'):
        return processor.graph_boosted_search(query)

    # Default
    return processor.find_documents_for_query(query)
```

## Related Topics

- [[query-optimization-guide.md]] - Writing effective queries
- [[graph-algorithms-primer.md]] - Understanding BM25 and PageRank
- [[troubleshooting/common-issues-guide.md]] - Fixing search quality issues
