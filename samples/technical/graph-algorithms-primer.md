# Graph Algorithms Primer

A practical guide to the graph algorithms used in the Cortical Text Processor.

## PageRank

PageRank measures the importance of nodes in a graph based on the structure of incoming links. Originally developed by Google for ranking web pages, we adapt it for term importance.

### How It Works

1. **Initialize** all nodes with equal probability (1/N)
2. **Iterate** until convergence:
   - Each node distributes its rank to neighbors
   - Apply damping factor (typically 0.85)
   - Add random jump probability
3. **Converge** when rank changes fall below threshold

### Mathematical Formula

```
PR(A) = (1-d)/N + d * Σ(PR(Ti)/C(Ti))
```

Where:
- `d` = damping factor (0.85)
- `N` = total number of nodes
- `Ti` = nodes linking to A
- `C(Ti)` = outbound link count from Ti

### In Cortical Text Processor

We apply PageRank to the term co-occurrence graph:
- **Nodes** = terms (tokens, bigrams)
- **Edges** = co-occurrence relationships
- **Result** = term importance scores

High PageRank terms are semantically central to the corpus.

## TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF measures how important a term is to a document within a corpus.

### Components

**Term Frequency (TF):**
```
TF(t,d) = count(t in d) / total_terms(d)
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
```

Where `df(t)` = number of documents containing term t.

**Combined:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### Intuition

- Terms appearing frequently in one document but rarely across corpus get high scores
- Common terms (the, is, a) get low scores due to high document frequency
- Distinctive terms that characterize a document get boosted

## BM25 (Best Match 25)

BM25 improves on TF-IDF with saturation and length normalization.

### Formula

```
BM25(q,d) = Σ IDF(qi) × (f(qi,d) × (k1 + 1)) / (f(qi,d) + k1 × (1 - b + b × |d|/avgdl))
```

### Parameters

- `k1` (1.2-2.0): Controls term frequency saturation
- `b` (0.75): Controls document length normalization

### Why BM25 is Better

1. **Saturation**: Additional term occurrences have diminishing returns
2. **Length normalization**: Long documents don't unfairly dominate
3. **Tunable**: Parameters can be adjusted for domain

## Louvain Community Detection

Louvain algorithm finds communities (clusters) in graphs by maximizing modularity.

### Phases

1. **Local optimization**: Move nodes to neighboring communities if it increases modularity
2. **Aggregation**: Treat communities as single nodes
3. **Repeat**: Until no improvement possible

### Modularity

```
Q = (1/2m) × Σ [Aij - (ki×kj)/(2m)] × δ(ci, cj)
```

Measures how much the graph deviates from random edge distribution.

### In Text Processing

We use Louvain to:
- Group related terms into concept clusters
- Identify topic boundaries
- Build the Layer 2 (CONCEPTS) hierarchy

## Practical Considerations

### Performance

| Algorithm | Complexity | Typical Runtime |
|-----------|------------|-----------------|
| PageRank | O(E × iterations) | ~100ms for 10K terms |
| TF-IDF | O(N × D) | ~50ms for 100 docs |
| BM25 | O(Q × D) | ~10ms per query |
| Louvain | O(N log N) | ~500ms for 10K terms |

### When to Use What

- **PageRank**: Finding important/central terms
- **TF-IDF**: Document retrieval, keyword extraction
- **BM25**: Search ranking (default in Cortical)
- **Louvain**: Topic modeling, concept extraction

## References

- Brin & Page (1998): "The Anatomy of a Large-Scale Hypertextual Web Search Engine"
- Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework: BM25 and Beyond"
- Blondel et al. (2008): "Fast unfolding of communities in large networks"
