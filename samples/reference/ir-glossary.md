# Information Retrieval Glossary

Key terms and concepts used in the Cortical Text Processor and information retrieval systems.

## Core Concepts

### BM25 (Best Match 25)
A probabilistic ranking function that improves on TF-IDF by adding term frequency saturation and document length normalization. The default scoring algorithm in the Cortical Text Processor.

**Parameters:**
- `k1`: Controls term frequency saturation (typically 1.2)
- `b`: Controls document length normalization (typically 0.75)

### Corpus
A collection of documents being analyzed. In the Cortical Text Processor, documents are added via `process_document()` and form the basis for all analysis.

### Document Frequency (DF)
The number of documents in a corpus that contain a specific term. Used in IDF calculation to identify distinctive terms.

### Inverse Document Frequency (IDF)
A measure of how rare a term is across the corpus. Calculated as `log(N/df)` where N is total documents and df is document frequency. Rare terms have high IDF.

### Term Frequency (TF)
The number of times a term appears in a specific document, often normalized by document length.

### TF-IDF
A weighting scheme that combines term frequency and inverse document frequency. Terms that appear frequently in one document but rarely across the corpus get high TF-IDF scores.

## Graph Algorithms

### PageRank
An algorithm that measures the importance of nodes in a graph based on link structure. Originally developed for web page ranking, adapted here for term importance in co-occurrence graphs.

**Key concepts:**
- Damping factor (typically 0.85)
- Random surfer model
- Iterative convergence

### Louvain Algorithm
A community detection algorithm that finds clusters by maximizing modularity. Used to identify concept clusters (semantic topics) in the term graph.

### Modularity
A measure of community structure in networks. High modularity indicates dense connections within communities and sparse connections between them.

## Semantic Concepts

### Co-occurrence
When two terms appear together in the same context (document, sentence, or window). Frequent co-occurrence suggests semantic relatedness.

### Hebbian Learning
"Neurons that fire together wire together." In this system, terms that co-occur frequently develop stronger lateral connections.

### Lateral Connections
Weighted connections between terms at the same layer based on co-occurrence. These enable query expansion and semantic similarity.

### Query Expansion
The process of adding related terms to a search query to improve recall. For example, expanding "car" to include "vehicle", "automobile".

### Semantic Similarity
A measure of how related two terms are in meaning, not just spelling. Can be computed from co-occurrence patterns, graph distance, or vector similarity.

### Synonym
Words with the same or similar meanings. The system learns synonyms from co-occurrence patterns rather than using predefined dictionaries.

## Architecture Terms

### Hierarchical Layer
One level of the 4-layer processing hierarchy:
- Layer 0 (TOKENS): Individual words
- Layer 1 (BIGRAMS): Word pairs
- Layer 2 (CONCEPTS): Semantic clusters
- Layer 3 (DOCUMENTS): Full documents

### Minicolumn
The basic unit of representation. Contains a term/concept, its connections, statistics (TF-IDF, PageRank), and document associations.

### Feedforward Connections
Connections from lower layers to higher layers (tokens → bigrams → concepts → documents). Represent abstraction.

### Feedback Connections
Connections from higher layers to lower layers. Enable top-down context to influence lower-level processing.

### Typed Connections
Connections with semantic labels (e.g., "is-a", "has-part", "related-to"). Enable structured knowledge representation.

## Search Concepts

### Precision
The fraction of retrieved documents that are relevant. High precision = few false positives.

`Precision = Relevant ∩ Retrieved / Retrieved`

### Recall
The fraction of relevant documents that are retrieved. High recall = few false negatives.

`Recall = Relevant ∩ Retrieved / Relevant`

### F1 Score
The harmonic mean of precision and recall. Balances both metrics.

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

### Mean Reciprocal Rank (MRR)
Average of reciprocal ranks of the first relevant result. If first relevant is at position 3, reciprocal rank is 1/3.

### Precision@K
Precision calculated over only the top K results. P@5 measures precision in top 5.

### Recall@K
Fraction of all relevant documents found in top K results.

## Indexing Concepts

### Inverted Index
A data structure mapping terms to the documents containing them. Enables fast lookup of "which documents contain term X?"

### Posting List
The list of documents (and positions) where a term appears. Part of an inverted index.

### Stemming
Reducing words to their root form. "running", "runs", "ran" → "run". Improves recall but may hurt precision.

### Stop Words
Common words (the, is, a, an) that are often removed from indexing because they add little semantic value.

### Tokenization
Breaking text into individual tokens (usually words). Handles punctuation, case normalization, and splitting.

## Persistence Concepts

### State Version
A version number stored with saved state to handle format changes between library versions.

### Chunk-Based Storage
A git-friendly storage format where changes are saved as append-only JSON files, avoiding merge conflicts.

### Compaction
Consolidating multiple chunk files into a single file, similar to git garbage collection.

## Performance Concepts

### Staleness
A computation is "stale" when underlying data has changed and results may be outdated. The processor tracks staleness per computation type.

### Incremental Update
Updating only what changed rather than recomputing everything. Much faster for adding single documents.

### Query Caching
Storing expanded query results to avoid recomputing expansion for repeated queries.

## Formulas Quick Reference

| Metric | Formula |
|--------|---------|
| TF | `count(t,d) / len(d)` |
| IDF | `log(N / df(t))` |
| TF-IDF | `TF × IDF` |
| BM25 | `IDF × (TF × (k1+1)) / (TF + k1 × (1-b+b×len/avglen))` |
| PageRank | `(1-d)/N + d × Σ(PR(i)/out(i))` |
| Modularity | `(1/2m) × Σ[Aij - kikj/2m] × δ(ci,cj)` |
| Cosine Similarity | `(A·B) / (‖A‖ × ‖B‖)` |
| Jaccard Similarity | `|A ∩ B| / |A ∪ B|` |

## See Also

- [[graph-algorithms-primer.md]] - Detailed algorithm explanations
- [[semantic-search-explained.md]] - How semantic search works
- [[query-optimization-guide.md]] - Practical query tips
