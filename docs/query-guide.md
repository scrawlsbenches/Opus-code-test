# Query Guide

A comprehensive guide to formulating effective search queries and understanding how the query system works internally.

---

## Table of Contents

1. [How Queries Work Internally](#how-queries-work-internally)
2. [Query Syntax and Patterns](#query-syntax-and-patterns)
3. [Understanding Query Expansion](#understanding-query-expansion)
4. [Single-Word vs Multi-Word Queries](#single-word-vs-multi-word-queries)
5. [Code Patterns vs Concept Searches](#code-patterns-vs-concept-searches)
6. [Intent-Based Queries](#intent-based-queries)
7. [Interpreting Relevance Scores](#interpreting-relevance-scores)
8. [When Queries Fail](#when-queries-fail)
9. [Advanced Techniques](#advanced-techniques)

---

## How Queries Work Internally

### The Query Pipeline

When you submit a query, the system performs a multi-stage pipeline:

```
Query Text
    |
[1. Tokenization] -> Split into words, remove stop words
    |
[2. Term Matching] -> Look up terms in token layer
    |
[3. Expansion] -> Add related terms via lateral connections
    |
[4. Document Scoring] -> TF-IDF weighting
    |
[5. Ranking] -> Sort by relevance score
    |
Results
```

### Stage 1: Tokenization

Your query is tokenized using the same rules as document processing:

```python
# "neural networks process data" becomes:
["neural", "networks", "process", "data"]

# Stop words are removed: "the", "a", "in", "of", "is"
# Short words (< 3 characters) are removed
```

**Key points:**
- Tokenization is **case-insensitive**
- Punctuation is removed
- Words shorter than 3 characters are filtered

### Stage 2: Term Matching

The system looks up each query token in Layer 0:

```
Token         Found?   Status
"neural"      YES      Exact match in corpus
"networks"    YES      Exact match in corpus
```

If a token doesn't exist, the system tries **word variants**:
- Stemmed versions
- Plural forms
- Common aliases

### Stage 3: Expansion

The query is expanded using three methods:

**Method A: Lateral Connections (Default)**
- Terms co-occurring with query terms
- Weights: connection strength x neighbor PageRank x 0.6

**Method B: Concept Clustering**
- Terms in same semantic cluster
- Weights: concept PageRank x member PageRank x 0.4

**Method C: Code Concepts (Optional)**
- Programming synonyms (get/fetch/load)
- Only enabled with `use_code_concepts=True`

### Stage 4: Document Scoring

```
doc_score = sum(term_weight x tfidf_per_doc)
            for each term in expanded_query

where:
  term_weight = original terms: 1.0
              = expanded terms: 0.3-0.8
```

---

## Query Syntax and Patterns

The system uses **simple, natural language-based syntax**. No special operators needed.

### Basic Patterns

| Pattern | Example | Effect |
|---------|---------|--------|
| **Single word** | `neural` | Search term and related concepts |
| **Multi-word** | `neural networks` | All words must match (AND logic) |
| **Question words** | `where authentication` | Intent-based search |
| **Action verbs** | `how validate input` | Parse action + subject |

### What Doesn't Work

```python
# NOT supported:
"neural" OR "learning"         # No boolean operators
"neural*"                      # No wildcards
"exact phrase match"           # No phrase searching
```

### How Multi-Word Queries Work

Multi-word queries use **AND logic** at document level:

```
Query: "neural networks"

Step 1: Find docs with "neural"   -> [doc1, doc3, doc5]
Step 2: Find docs with "networks" -> [doc1, doc3, doc6]
Step 3: Intersection              -> [doc1, doc3]
Step 4: Rank by combined score
```

---

## Understanding Query Expansion

Query expansion is **the core secret** to finding relevant results even when your query doesn't exactly match.

### How Expansion Works

Given query `"fetch user"`:

```
Original Terms (weight 1.0):
  - fetch
  - user

Lateral Connection Expansion:

Neighbors of "fetch":
  - get: 0.45
  - load: 0.42
  - data: 0.38

Neighbors of "user":
  - profile: 0.52
  - account: 0.48
  - authenticate: 0.35

Final Query Terms:
{
  "fetch": 1.0,        # Original
  "user": 1.0,         # Original
  "get": 0.45,         # Expansion
  "profile": 0.52,     # Expansion
  ...
}
```

### Controlling Expansion

```python
# With lateral connections only
results = processor.find_documents_for_query(
    "neural networks",
    use_expansion=True,
    use_semantic=False
)

# No expansion (exact match)
results = processor.find_documents_for_query(
    "neural networks",
    use_expansion=False
)

# Code-specific expansion
results = processor.expand_query_for_code("fetch user credentials")
```

### Debugging Expansion

```python
expanded = processor.expand_query("neural networks", max_expansions=10)

for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    print(f"  {term}: {weight:.3f}")
```

---

## Single-Word vs Multi-Word Queries

### Single-Word Queries

**Advantages:**
- Faster execution
- Broader matching
- Better for exploratory search

**Disadvantages:**
- May return less relevant results if term is ambiguous

```python
Query: "learning"
# Finds all documents with "learning" and related terms
```

### Multi-Word Queries

**Advantages:**
- More specific results (AND logic)
- Provides disambiguation context

**Disadvantages:**
- Harder to match (all terms must exist)

```python
Query: "machine learning"
# Returns only docs with BOTH terms
```

### Strategy: Combining Both

```python
# Broad search first
broad = processor.find_documents_for_query("learning", top_n=20)

# Narrow with multi-word
narrow = processor.find_documents_for_query("machine learning", top_n=5)

# Use narrow if available, fall back to broad
results = narrow if narrow else broad
```

---

## Code Patterns vs Concept Searches

### Concept Searches (General Text)

Best for finding semantic topics:

```python
processor.find_documents_for_query("authentication")
processor.find_documents_for_query("neural networks")
```

Uses:
- Lateral connections
- Concept clusters
- Natural language semantics

### Code Pattern Searches

Best for finding implementations:

```python
processor.expand_query_for_code("get user credentials")
processor.expand_query_for_code("validate input")
```

Uses:
- Code concept groups (get/fetch/load)
- Programming keywords
- Identifier splitting

### When to Use Each

| Type | Use Case | Method |
|------|----------|--------|
| **Concept** | Find ideas, topics | `find_documents_for_query()` |
| **Code** | Find implementations | `expand_query_for_code()` |
| **Intent** | Find by action | `search_by_intent()` |
| **Passage** | Find specific text | `find_passages_for_query()` |

---

## Intent-Based Queries

Intent-based queries use **natural language patterns** to understand what you're looking for.

### Supported Question Words

| Word | Intent | Example |
|------|--------|---------|
| **where** | location | "where do we handle authentication?" |
| **how** | implementation | "how does validation work?" |
| **what** | definition | "what is a concept cluster?" |
| **why** | rationale | "why do we use PageRank?" |
| **when** | lifecycle | "when do we compute TF-IDF?" |

### How Intent Parsing Works

```
Query: "where do we handle authentication?"

Step 1: Detect "where" -> intent = "location"
Step 2: Extract content words -> handle, authentication
Step 3: Identify action verb -> "handle"
Step 4: Identify subject -> "authentication"
Step 5: Build expanded terms
Step 6: Search with weighted terms
```

### Using Intent Queries

```python
results = processor.search_by_intent("where do we validate input?", top_n=5)

parsed = processor.parse_intent_query("how does PageRank work?")
# {
#   'action': 'work',
#   'subject': 'pagerank',
#   'intent': 'implementation',
#   'expanded_terms': ['work', 'pagerank', 'rank', ...]
# }
```

---

## Interpreting Relevance Scores

### Score Meaning

```
Score > 0.80   Very relevant - high confidence match
Score 0.50-0.80  Relevant - good match
Score 0.25-0.50  Somewhat relevant - weak connection
Score < 0.25   Marginally relevant
```

### How Scores Are Calculated

```python
# TF-IDF Score:
tf_idf = (term_count_in_doc / total_terms) x log(total_docs / docs_with_term)

# Query Score:
doc_score = sum(term_weight x term_tfidf_per_doc)
```

### Factors Affecting Scores

1. **Term Frequency (TF):** More occurrences = higher score
2. **Inverse Document Frequency (IDF):** Rarer terms = higher weight
3. **Query Term Weight:** Original (1.0) vs expansion (0.3-0.6)
4. **Concept overlap:** Documents in same cluster score higher

---

## When Queries Fail

### Problem 1: No Results Found

**Diagnosis:**
```python
layer0 = processor.get_layer(CorticalLayer.TOKENS)
for term in processor.tokenizer.tokenize(query):
    if not layer0.get_minicolumn(term):
        print(f"{term}: NOT FOUND")
```

**Solutions:**
1. Try variant forms: `"getUserData"` -> `"get user data"`
2. Enable code splitting in tokenizer
3. Use related concepts instead

### Problem 2: Wrong Documents Returned

**Diagnosis:**
```python
expanded = processor.expand_query("authentication")
# Check for unexpected expansion terms
```

**Solutions:**
1. Use multi-word queries for specificity
2. Disable expansion: `use_expansion=False`
3. Use intent-based search

### Problem 3: Missing Relevant Documents

**Solutions:**
1. Enable semantic expansion:
   ```python
   processor.extract_corpus_semantics()
   results = processor.find_documents_for_query(
       query,
       use_semantic=True
   )
   ```

2. Use multi-hop expansion:
   ```python
   expanded = processor.expand_query_multihop(query, max_hops=2)
   ```

### Problem 4: Slow Queries

**Solutions:**
1. Use `fast_find_documents()`
2. Pre-build search index
3. Use narrower queries

---

## Advanced Techniques

### Technique 1: Multi-Hop Expansion

```python
processor.extract_corpus_semantics()

expanded = processor.expand_query_multihop(
    "neural",
    max_hops=2,
    max_expansions=15
)

# Hop 0: neural
# Hop 1: networks, learning, brain
# Hop 2: deep (via learning), cortex (via brain)
```

### Technique 2: Passage Retrieval (RAG)

```python
results = processor.find_passages_for_query(
    "neural network training",
    top_n=5,
    chunk_size=512,
    overlap=128
)

for passage, doc_id, start, end, score in results:
    print(f"[{doc_id}:{start}-{end}] Score: {score:.3f}")
    print(passage)
```

### Technique 3: Multi-Stage Ranking

```python
results = processor.multi_stage_rank(
    "neural networks",
    top_n=5,
    concept_boost=0.3
)

for passage, doc_id, start, end, score, stages in results:
    print(f"Concept: {stages['concept_score']:.3f}")
    print(f"Document: {stages['doc_score']:.3f}")
    print(f"Passage: {stages['chunk_score']:.3f}")
```

### Technique 4: Batch Queries

```python
queries = ["neural networks", "machine learning", "deep learning"]
results = processor.find_documents_batch(queries, top_n=5)
# ~2-3x faster for multiple queries
```

---

## Quick Reference

### Common Methods

```python
# Basic search
processor.find_documents_for_query(query, top_n=5)

# Fast search (large corpora)
processor.fast_find_documents(query, top_n=5)

# Intent-based
processor.search_by_intent("where do we...?", top_n=5)

# Passages (RAG)
processor.find_passages_for_query(query, top_n=5, chunk_size=512)

# Code-specific
processor.expand_query_for_code(query)

# Multi-hop
processor.expand_query_multihop(query, max_hops=2)

# Batch queries
processor.find_documents_batch(queries, top_n=5)

# Debug expansion
processor.expand_query(query, max_expansions=10)
```

### Query Tips

1. **Start simple** - Single keywords first
2. **Add specificity** - Multi-word if needed
3. **Use intent words** - "where", "how", "what"
4. **Check expansion** - See what terms are added
5. **Trust the system** - Expansion finds related terms

---

*For practical recipes, see [Cookbook](cookbook.md). For Claude-specific usage, see [Claude Usage Guide](claude-usage.md).*
