---
title: "Graph Algorithms"
generated: "2025-12-16T17:26:23.834319Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/analysis/clustering.py"
  - "/home/user/Opus-code-test/cortical/analysis/quality.py"
  - "/home/user/Opus-code-test/cortical/analysis/pagerank.py"
  - "/home/user/Opus-code-test/cortical/analysis/__init__.py"
  - "/home/user/Opus-code-test/cortical/analysis/tfidf.py"
  - "/home/user/Opus-code-test/cortical/analysis/activation.py"
  - "/home/user/Opus-code-test/cortical/analysis/connections.py"
  - "/home/user/Opus-code-test/cortical/analysis/utils.py"
tags:
  - architecture
  - modules
  - analysis
---

# Graph Algorithms

Graph algorithms for computing importance, relevance, and clusters.

## Modules

- **__init__.py**: Analysis Module
- **activation.py**: Activation propagation algorithm.
- **clustering.py**: Clustering algorithms for community detection.
- **connections.py**: Connection building algorithms for network layers.
- **pagerank.py**: PageRank algorithms for importance scoring.
- **quality.py**: Clustering quality metrics.
- **tfidf.py**: TF-IDF and BM25 scoring algorithms.
- **utils.py**: Utility functions and classes for analysis algorithms.


## __init__.py

Analysis Module
===============

Graph analysis algorithms for the cortical network.

Contains implementations of:
- PageRank for importance scoring
- TF-IDF for term weighting
- Louvain community det...


### Dependencies

**Standard Library:**

- `activation.propagate_activation`
- `clustering._louvain_core`
- `clustering.build_concept_clusters`
- `clustering.cluster_by_label_propagation`
- `clustering.cluster_by_louvain`
- ... and 22 more



## activation.py

Activation propagation algorithm.

Contains:
- propagate_activation: Spread activation through the network layers


### Functions

#### propagate_activation

```python
propagate_activation(layers: Dict[CorticalLayer, HierarchicalLayer], iterations: int = 3, decay: float = 0.8, lateral_weight: float = 0.3) -> None
```

Propagate activation through the network.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`



## clustering.py

Clustering algorithms for community detection.

Contains:
- cluster_by_louvain: Louvain modularity optimization (recommended)
- cluster_by_label_propagation: Label propagation clustering (legacy)
- bu...


### Functions

#### cluster_by_label_propagation

```python
cluster_by_label_propagation(layer: HierarchicalLayer, min_cluster_size: int = 3, max_iterations: int = 20, cluster_strictness: float = 1.0, bridge_weight: float = 0.0) -> Dict[int, List[str]]
```

Cluster minicolumns using label propagation.

#### cluster_by_louvain

```python
cluster_by_louvain(layer: HierarchicalLayer, min_cluster_size: int = 3, resolution: float = 1.0, max_iterations: int = 10) -> Dict[int, List[str]]
```

Cluster minicolumns using Louvain community detection.

#### build_concept_clusters

```python
build_concept_clusters(layers: Dict[CorticalLayer, HierarchicalLayer], clusters: Dict[int, List[str]], doc_vote_threshold: float = 0.1) -> None
```

Build concept layer from token clusters.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`
- `typing.List`
- ... and 2 more



## connections.py

Connection building algorithms for network layers.

Contains:
- compute_document_connections: Build document-to-document similarity connections
- compute_bigram_connections: Build lateral connections ...


### Functions

#### compute_concept_connections

```python
compute_concept_connections(layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: List[Tuple[str, str, str, float]] = None, min_shared_docs: int = 1, min_jaccard: float = 0.1, use_member_semantics: bool = False, use_embedding_similarity: bool = False, embedding_threshold: float = 0.3, embeddings: Dict[str, List[float]] = None) -> Dict[str, Any]
```

Build lateral connections between concepts in Layer 2.

#### compute_bigram_connections

```python
compute_bigram_connections(layers: Dict[CorticalLayer, HierarchicalLayer], min_shared_docs: int = 1, component_weight: float = 0.5, chain_weight: float = 0.7, cooccurrence_weight: float = 0.3, max_bigrams_per_term: int = 100, max_bigrams_per_doc: int = 500, max_connections_per_bigram: int = 50) -> Dict[str, Any]
```

Build lateral connections between bigrams in Layer 1.

#### compute_document_connections

```python
compute_document_connections(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], min_shared_terms: int = 3) -> None
```

Build lateral connections between documents.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `minicolumn.Minicolumn`
- `typing.Any`
- ... and 5 more



## pagerank.py

PageRank algorithms for importance scoring.

Contains:
- compute_pagerank: Standard PageRank for a single layer
- compute_semantic_pagerank: PageRank with semantic relation weighting
- compute_hierarc...


### Functions

#### compute_pagerank

```python
compute_pagerank(layer: HierarchicalLayer, damping: float = 0.85, iterations: int = 20, tolerance: float = 1e-06) -> Dict[str, float]
```

Compute PageRank scores for minicolumns in a layer.

#### compute_semantic_pagerank

```python
compute_semantic_pagerank(layer: HierarchicalLayer, semantic_relations: List[Tuple[str, str, str, float]], relation_weights: Optional[Dict[str, float]] = None, damping: float = 0.85, iterations: int = 20, tolerance: float = 1e-06) -> Dict[str, Any]
```

Compute PageRank with semantic relation type weighting.

#### compute_hierarchical_pagerank

```python
compute_hierarchical_pagerank(layers: Dict[CorticalLayer, HierarchicalLayer], layer_iterations: int = 10, global_iterations: int = 5, damping: float = 0.85, cross_layer_damping: float = 0.7, tolerance: float = 0.0001) -> Dict[str, Any]
```

Compute PageRank with cross-layer propagation.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.RELATION_WEIGHTS`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Any`
- ... and 4 more



## quality.py

Clustering quality metrics.

Contains:
- compute_clustering_quality: Comprehensive quality evaluation (modularity, silhouette, balance)
- _compute_modularity: Modularity Q metric
- _compute_silhouette...


### Functions

#### compute_clustering_quality

```python
compute_clustering_quality(layers: Dict[CorticalLayer, HierarchicalLayer], sample_size: int = 500) -> Dict[str, Any]
```

Compute clustering quality metrics for the concept layer.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `random`
- `typing.Any`
- ... and 3 more



## tfidf.py

TF-IDF and BM25 scoring algorithms.

Contains:
- compute_tfidf: Traditional TF-IDF scoring
- compute_bm25: Okapi BM25 scoring with length normalization
- _tfidf_core: Pure TF-IDF algorithm for unit te...


### Functions

#### compute_tfidf

```python
compute_tfidf(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> None
```

Compute TF-IDF scores for tokens.

#### compute_bm25

```python
compute_bm25(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], doc_lengths: Dict[str, int], avg_doc_length: float, k1: float = 1.2, b: float = 0.75) -> None
```

Compute BM25 scores for tokens.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- `typing.Dict`
- `typing.Tuple`



## utils.py

Utility functions and classes for analysis algorithms.

Contains:
- SparseMatrix: Zero-dependency sparse matrix for bigram connections
- Similarity functions: cosine_similarity, _doc_similarity, _vect...


### Classes

#### SparseMatrix

Simple sparse matrix implementation using dictionary of keys (DOK) format.

**Methods:**

- `set`
- `get`
- `multiply_transpose`
- `get_nonzero`

### Functions

#### cosine_similarity

```python
cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float
```

Compute cosine similarity between two sparse vectors.

#### SparseMatrix.set

```python
SparseMatrix.set(self, row: int, col: int, value: float) -> None
```

Set value at (row, col).

#### SparseMatrix.get

```python
SparseMatrix.get(self, row: int, col: int) -> float
```

Get value at (row, col).

#### SparseMatrix.multiply_transpose

```python
SparseMatrix.multiply_transpose(self) -> 'SparseMatrix'
```

Multiply this matrix by its transpose: M * M^T

#### SparseMatrix.get_nonzero

```python
SparseMatrix.get_nonzero(self) -> List[Tuple[int, int, float]]
```

Get all non-zero entries.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `math`
- `typing.Dict`
- `typing.List`
- `typing.Tuple`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
