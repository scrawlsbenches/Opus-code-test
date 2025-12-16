# The Cortical Chronicles

*A Self-Documenting Living Book*

---

This document is automatically generated from the Cortical Text Processor codebase.
It consolidates all book chapters into a single markdown file for offline reading,
PDF generation, or direct viewing on GitHub.

---

## Table of Contents

### [Preface](#preface)

- [How This Book Works](#how-this-book-works)

### [Foundations: Core Algorithms](#foundations-core-algorithms)

- [BM25/TF-IDF — Distinctiveness Scoring](#bm25tf-idf-distinctiveness-scoring)
- [Graph-Boosted Search (GB-BM25) — Hybrid Ranking](#graph-boosted-search-gb-bm25-hybrid-ranking)
- [Louvain Community Detection — Concept Discovery](#louvain-community-detection-concept-discovery)
- [PageRank — Importance Discovery](#pagerank-importance-discovery)
- [Query Expansion — Semantic Bridging](#query-expansion-semantic-bridging)
- [Semantic Relation Extraction — Knowledge Graph Construction](#semantic-relation-extraction-knowledge-graph-construction)

### [Architecture: System Design](#architecture-system-design)

- [Architecture Overview](#architecture-overview)
- [Graph Algorithms](#graph-algorithms)
- [Configuration](#configuration)
- [Data Structures](#data-structures)
- [NLP Components](#nlp-components)
- [Observability](#observability)
- [Persistence Layer](#persistence-layer)
- [Core Processor](#core-processor)
- [Search & Retrieval](#search-retrieval)
- [Utilities](#utilities)

### [Decisions: ADRs](#decisions-adrs)

- [Decisions Chapter](#decisions-chapter)

### [Evolution: Project History](#evolution-project-history)

- [Bug Fixes and Lessons](#bug-fixes-and-lessons)
- [Feature Evolution](#feature-evolution)
- [Refactorings and Architecture Evolution](#refactorings-and-architecture-evolution)
- [Test](#test)
- [Project Timeline](#project-timeline)

### [Future: Roadmap](#future-roadmap)

- [Future Chapter](#future-chapter)

---

# Preface

## How This Book Works

> *"The best documentation is the kind that writes itself."*

## Overview

The Cortical Chronicles is a **self-documenting book**. It uses the Cortical Text Processor—the very system it documents—to generate its own content. This creates a fascinating recursive property: the book understands itself through the same algorithms it explains.

## The Generation Process

```
┌─────────────────────────────────────────────────────────────┐
│                    BOOK GENERATION                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Source Files           Generators           Chapters        │
│  ───────────           ──────────           ────────        │
│                                                              │
│  docs/VISION.md    →   AlgorithmGen    →   01-foundations/  │
│  cortical/*.ai_meta →  ModuleDocGen    →   02-architecture/ │
│  samples/decisions/ →  DecisionGen     →   03-decisions/    │
│  git log           →   NarrativeGen    →   04-evolution/    │
│  tasks/            →   RoadmapGen      →   05-future/       │
│                                                              │
│                    ↓                                         │
│              search-index.json                               │
│                    ↓                                         │
│               index.html (searchable)                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Chapter Types

### 01-foundations/
Algorithm deep-dives extracted from `docs/VISION.md`. Each algorithm (PageRank, BM25, Louvain, etc.) gets its own chapter with:
- Purpose and intuition
- Mathematical formulation
- Implementation details
- Why it matters for code search

### 02-architecture/
Module documentation generated from `.ai_meta` files. Includes:
- Module purpose and dependencies
- Key functions and classes
- Mermaid dependency graphs

### 03-decisions/
Architecture Decision Records from `samples/decisions/`. Documents the "why" behind design choices.

### 04-evolution/
A narrative of project history generated from git commits. Transforms raw commit logs into a readable story of how the project evolved.

### 05-future/
Roadmap and vision from task files and `VISION.md`. Shows where the project is heading.

## The Self-Reference Loop

Here's what makes this book special:

1. **The processor indexes its own code** → Creates a semantic graph
2. **The generators query that graph** → Find relevant content
3. **The book explains those algorithms** → Reader understands the system
4. **The system processes those explanations** → Understands itself better

This isn't just cute—it's a powerful test of the system's capabilities. If the Cortical Text Processor can understand and explain itself, it can understand any codebase.

## Regenerating the Book

The book regenerates automatically on every push to `main`:

```bash
# Manual regeneration
python scripts/generate_book.py

# Generate specific chapter
python scripts/generate_book.py --chapter foundations

# Preview without writing
python scripts/generate_book.py --dry-run
```

## Searching the Book

The book includes a semantic search interface. Open `index.html` to:
- Search by keyword or concept
- Browse by chapter
- Follow cross-references

The search uses the same algorithms described in the book—query expansion, BM25 scoring, PageRank boosting.

## See Also

- [Algorithm Analysis](../01-foundations/index.md) - Deep dive into the algorithms
- [Architecture](../02-architecture/index.md) - How the code is organized
- [Source: generate_book.py](../../scripts/generate_book.py) - The generation script

## Source Files

This chapter was written manually as the seed for the book. Future chapters are auto-generated from:
- `scripts/generate_book.py` - The orchestrator
- `docs/VISION.md:185-430` - Algorithm documentation source

---

*This chapter is part of [The Cortical Chronicles](../README.md),
a self-documenting book generated by the Cortical Text Processor.*

---

# Foundations: Core Algorithms

## BM25/TF-IDF — Distinctiveness Scoring

**Purpose:** Score how well a term distinguishes a specific document from the rest of the corpus.

**Implementation:** `cortical/analysis/tfidf.py`

**BM25 Formula (Default):**
```
BM25(t, d) = IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

Where:
- `IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)` — Inverse document frequency with smoothing
- `tf(t,d)` — Term frequency in document d
- `k1 = 1.2` — Term frequency saturation (diminishing returns after ~12 occurrences)
- `b = 0.75` — Length normalization factor

**Why BM25 Over TF-IDF:**
- Non-negative IDF even for terms appearing in most documents
- Length normalization prevents long files from unfairly dominating
- Term frequency saturation models realistic relevance (saying "API" 100 times doesn't make a doc 100× more relevant than saying it once)

**Dual Storage Strategy:**
- **Global TF-IDF** (`col.tfidf`): Term importance to entire corpus
- **Per-Document TF-IDF** (`col.tfidf_per_doc[doc_id]`): Term importance within specific document

This dual approach allows:
- Fast corpus-wide importance filtering
- Accurate per-document relevance scoring for search

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Graph-Boosted Search (GB-BM25) — Hybrid Ranking

**Purpose:** Combine BM25 relevance with graph structure signals.

**Implementation:** `cortical/query/search.py:425-564`

**Scoring Formula:**
```
final_score = (0.5 × normalized_bm25) + (0.3 × normalized_pagerank) + (0.2 × normalized_proximity)
            × coverage_multiplier (0.5 to 1.5)
```

**Three Signal Sources:**

1. **BM25 Base Score (50%):**
   - Standard term frequency × inverse document frequency
   - Per-document scoring using `col.tfidf_per_doc`

2. **PageRank Boost (30%):**
   - Sum of matched term PageRanks
   - Rewards documents containing important terms

3. **Proximity Boost (20%):**
   - For each pair of original query terms:
     - Check if they're connected in the co-occurrence graph
     - If connected, boost documents containing both
   - Rewards documents where query terms appear together

**Coverage Multiplier:**
- Documents matching 1/5 query terms: 0.7× multiplier
- Documents matching all 5 query terms: 1.5× multiplier
- Prevents documents matching one rare term from outranking documents matching many terms

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Louvain Community Detection — Concept Discovery

**Purpose:** Discover semantic clusters (concepts) from the term co-occurrence graph.

**Implementation:** `cortical/analysis/clustering.py`

**Two-Phase Algorithm:**

**Phase 1 — Local Optimization:**
```
for each node:
    find neighboring communities
    calculate modularity gain for moving to each
    move to best community if gain > 0
repeat until no nodes move
```

**Phase 2 — Network Aggregation:**
```
collapse each community into a single super-node
edges between communities become edges between super-nodes
repeat Phase 1 on the aggregated network
```

**Modularity Formula:**
```
Q = (1/2m) × Σ [A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)
```

The algorithm optimizes Q, which measures how much edge weight falls within communities versus what would be expected by random chance.

**Resolution Parameter:**
- `resolution = 1.0` (default): Balanced clusters, ~32 concepts
- `resolution = 0.5`: Coarse clusters, ~38 concepts (max cluster 64% of tokens)
- `resolution = 2.0`: Fine-grained clusters, ~79 concepts (max cluster 4.2% of tokens)

**Concept Naming:**
```python
top_members = sorted(cluster_members, key=lambda m: m.pagerank, reverse=True)[:3]
concept_name = '/'.join(top_members)  # e.g., "neural/learning/networks"
```

**Why This Matters:**
- Enables concept-level search ("find documents about authentication")
- Reduces dimensionality while preserving semantic structure
- Creates Layer 2 (Concepts) that bridges raw terms and documents

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## PageRank — Importance Discovery

**Purpose:** Identify which terms matter most in the corpus, independent of raw frequency.

**Implementation:** `cortical/analysis/pagerank.py`

**How It Works:**
```
importance[term] = (1 - damping) / N + damping × Σ (neighbor_importance × edge_weight / neighbor_outgoing_sum)
```

The algorithm iteratively propagates importance through the term co-occurrence graph. Terms that are referenced by many important terms become important themselves—a recursive definition that converges to stable values.

**Key Parameters:**
- `damping = 0.85`: The probability of following a link vs. jumping to a random node
- `tolerance = 1e-6`: Convergence threshold (stops when no term changes by more than this)
- `max_iterations = 20`: Upper bound on iterations

**Three Variants:**
1. **Standard PageRank**: Applied to Layer 0 (tokens) and Layer 1 (bigrams)
2. **Semantic PageRank**: Adjusts edge weights by relation type (IsA connections count 1.5× more than CoOccurs)
3. **Hierarchical PageRank**: Propagates importance across all 4 layers with separate cross-layer damping

**Why This Matters for Code Search:**
- Common utility functions referenced everywhere get high PageRank
- Core abstractions that everything depends on surface naturally
- Prevents over-emphasis on boilerplate code that appears frequently but isn't semantically central

**Performance:** O(iterations × edges), typically 100-500ms for 10K tokens with early convergence usually at 5-10 iterations.

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Query Expansion — Semantic Bridging

**Purpose:** Transform literal query terms into semantically enriched term sets.

**Implementation:** `cortical/query/expansion.py`

**Three Expansion Methods:**

1. **Lateral Connection Expansion:**
   - Follow co-occurrence edges from query terms
   - Score: `edge_weight × neighbor_score × 0.6`
   - Takes top 5 neighbors per query term

2. **Concept Cluster Membership:**
   - Find concepts containing query terms
   - Add other cluster members as expansions
   - Score: `concept.pagerank × member.pagerank × 0.4`

3. **Code Concept Synonyms:**
   - Programming-specific synonym groups (get/fetch/load, create/make/build)
   - Limited to 3 synonyms per term to prevent drift

**Multi-Hop Inference:**
```
Query: "neural"
  Hop 0: neural (1.0)
  Hop 1: networks (0.4), learning (0.35)
  Hop 2: deep (0.098) — via learning with decay
```

Chain validity is scored by relation type pairs:
- `(IsA, IsA)`: 1.0 — fully transitive (dog→animal→living_thing)
- `(RelatedTo, RelatedTo)`: 0.6 — weaker transitivity
- `(Antonym, Antonym)`: 0.3 — double negation, avoid

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Semantic Relation Extraction — Knowledge Graph Construction

**Purpose:** Extract typed relationships (IsA, PartOf, Causes) from document text.

**Implementation:** `cortical/semantics.py`

**Pattern-Based Extraction:**
24 regex patterns detect 10+ relation types:
```python
r'(\w+)\s+(?:is|are)\s+(?:a|an)\s+(?:type\s+of\s+)?(\w+)' → IsA (0.9 confidence)
r'(\w+)\s+(?:is|are)\s+(?:a\s+)?part\s+of' → PartOf (0.95 confidence)
r'(\w+)\s+(?:causes|leads?\s+to)' → Causes (0.9 confidence)
```

**Semantic Retrofitting:**
Blends co-occurrence weights with semantic relation knowledge:
```
new_weight = α × original_weight + (1-α) × semantic_target_weight
```
With α = 0.3, semantic signals dominate (70%) while preserving some corpus statistics (30%).

**Relation Weight Multipliers:**
| Relation | Weight | Semantics |
|----------|--------|-----------|
| SameAs | 2.0 | Strongest synonymy |
| IsA | 1.5 | Hypernymy |
| PartOf | 1.3 | Meronymy |
| RelatedTo | 0.8 | Generic |
| Antonym | -0.5 | Opposition |

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

# Architecture: System Design

## Architecture Overview

This section documents the architecture of the Cortical Text Processor through automatically extracted module metadata.

## Statistics

- **Total Modules**: 45
- **Module Groups**: 9
- **Classes**: 50
- **Functions**: 422

## Module Groups

### [Analysis](mod-analysis.md)

8 modules:

- `__init__.py`
- `activation.py`
- `clustering.py`
- `connections.py`
- `pagerank.py`
- `quality.py`
- `tfidf.py`
- `utils.py`

### [Configuration](mod-configuration.md)

3 modules:

- `config.py`
- `constants.py`
- `validation.py`

### [Data Structures](mod-data-structures.md)

3 modules:

- `layers.py`
- `minicolumn.py`
- `types.py`

### [Nlp](mod-nlp.md)

3 modules:

- `embeddings.py`
- `semantics.py`
- `tokenizer.py`

### [Observability](mod-observability.md)

3 modules:

- `observability.py`
- `progress.py`
- `results.py`

### [Persistence](mod-persistence.md)

3 modules:

- `chunk_index.py`
- `persistence.py`
- `state_storage.py`

### [Processor](mod-processor.md)

6 modules:

- `__init__.py`
- `compute.py`
- `core.py`
- `documents.py`
- `introspection.py`
- `persistence_api.py`

### [Query](mod-query.md)

8 modules:

- `__init__.py`
- `chunking.py`
- `definitions.py`
- `expansion.py`
- `intent.py`
- `passages.py`
- `ranking.py`
- `search.py`

### [Utilities](mod-utilities.md)

8 modules:

- `cli_wrapper.py`
- `code_concepts.py`
- `diff.py`
- `fingerprint.py`
- `fluent.py`
- `gaps.py`
- `mcp_server.py`
- `patterns.py`

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Graph Algorithms

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

---

## Configuration

Configuration management and validation.

## Modules

- **config.py**: Configuration Module
- **constants.py**: Centralized constants for the Cortical Text Processor.
- **validation.py**: Validation Module


## config.py

Configuration Module
====================

Centralized configuration for the Cortical Text Processor.

This module provides a dataclass-based configuration system that allows
users to customize algori...


### Classes

#### CorticalConfig

Configuration settings for the Cortical Text Processor.

**Methods:**

- `copy`
- `to_dict`
- `from_dict`

### Functions

#### get_default_config

```python
get_default_config() -> CorticalConfig
```

Get a new instance of the default configuration.

#### CorticalConfig.copy

```python
CorticalConfig.copy(self) -> 'CorticalConfig'
```

Create a copy of this configuration.

#### CorticalConfig.to_dict

```python
CorticalConfig.to_dict(self) -> Dict
```

Convert configuration to a dictionary for serialization.

#### CorticalConfig.from_dict

```python
CorticalConfig.from_dict(cls, data: Dict) -> 'CorticalConfig'
```

Create configuration from a dictionary.

### Dependencies

**Standard Library:**

- `dataclasses.dataclass`
- `dataclasses.field`
- `math`
- `typing.Dict`
- `typing.FrozenSet`
- ... and 1 more



## constants.py

Centralized constants for the Cortical Text Processor.

This module provides a single source of truth for constants used across
multiple modules, preventing drift and inconsistencies.

Task #96: Centr...


### Dependencies

**Standard Library:**

- `typing.Dict`
- `typing.FrozenSet`



## validation.py

Validation Module
=================

Input validation utilities and decorators for the Cortical Text Processor.

This module provides reusable validators and decorators to ensure
parameters are valid ...


### Functions

#### validate_non_empty_string

```python
validate_non_empty_string(value: Any, param_name: str) -> None
```

Validate that a value is a non-empty string.

#### validate_positive_int

```python
validate_positive_int(value: Any, param_name: str) -> None
```

Validate that a value is a positive integer.

#### validate_non_negative_int

```python
validate_non_negative_int(value: Any, param_name: str) -> None
```

Validate that a value is a non-negative integer.

#### validate_range

```python
validate_range(value: Any, param_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None, inclusive: bool = True) -> None
```

Validate that a numeric value is within a specified range.

#### validate_params

```python
validate_params(**validators: Callable[[Any], None]) -> Callable[[F], F]
```

Decorator to validate function parameters.

#### marks_stale

```python
marks_stale(*computation_types: str) -> Callable[[F], F]
```

Decorator to mark computations as stale after method execution.

#### marks_fresh

```python
marks_fresh(*computation_types: str) -> Callable[[F], F]
```

Decorator to mark computations as fresh after method execution.

### Dependencies

**Standard Library:**

- `functools.wraps`
- `inspect`
- `typing.Any`
- `typing.Callable`
- `typing.Optional`
- ... and 2 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Data Structures

Fundamental data structures used throughout the system.

## Modules

- **layers.py**: Layers Module
- **minicolumn.py**: Minicolumn Module
- **types.py**: Type Aliases for the Cortical Text Processor.


## layers.py

Layers Module
=============

Defines the hierarchical layer structure inspired by the visual cortex.

The neocortex processes information through a hierarchy of layers,
each extracting progressively m...


### Classes

#### CorticalLayer

Enumeration of cortical processing layers.

**Methods:**

- `description`
- `analogy`

#### HierarchicalLayer

A layer in the cortical hierarchy containing minicolumns.

**Methods:**

- `get_or_create_minicolumn`
- `get_minicolumn`
- `get_by_id`
- `remove_minicolumn`
- `column_count`
- `total_connections`
- `average_activation`
- `activation_range`
- `sparsity`
- `top_by_pagerank`
- `top_by_tfidf`
- `top_by_activation`
- `to_dict`
- `from_dict`

### Functions

#### CorticalLayer.description

```python
CorticalLayer.description(self) -> str
```

Human-readable description of this layer.

#### CorticalLayer.analogy

```python
CorticalLayer.analogy(self) -> str
```

Visual cortex analogy for this layer.

#### HierarchicalLayer.get_or_create_minicolumn

```python
HierarchicalLayer.get_or_create_minicolumn(self, content: str) -> Minicolumn
```

Get existing minicolumn or create new one.

#### HierarchicalLayer.get_minicolumn

```python
HierarchicalLayer.get_minicolumn(self, content: str) -> Optional[Minicolumn]
```

Get a minicolumn by content, or None if not found.

#### HierarchicalLayer.get_by_id

```python
HierarchicalLayer.get_by_id(self, col_id: str) -> Optional[Minicolumn]
```

Get a minicolumn by its ID in O(1) time.

#### HierarchicalLayer.remove_minicolumn

```python
HierarchicalLayer.remove_minicolumn(self, content: str) -> bool
```

Remove a minicolumn from this layer.

#### HierarchicalLayer.column_count

```python
HierarchicalLayer.column_count(self) -> int
```

Return the number of minicolumns in this layer.

#### HierarchicalLayer.total_connections

```python
HierarchicalLayer.total_connections(self) -> int
```

Return total number of lateral connections in this layer.

#### HierarchicalLayer.average_activation

```python
HierarchicalLayer.average_activation(self) -> float
```

Calculate average activation across all minicolumns.

#### HierarchicalLayer.activation_range

```python
HierarchicalLayer.activation_range(self) -> tuple
```

Return (min, max) activation values.

#### HierarchicalLayer.sparsity

```python
HierarchicalLayer.sparsity(self, threshold_fraction: float = 0.5) -> float
```

Calculate sparsity (fraction of columns with below-average activation).

#### HierarchicalLayer.top_by_pagerank

```python
HierarchicalLayer.top_by_pagerank(self, n: int = 10) -> list
```

Get top minicolumns by PageRank score.

#### HierarchicalLayer.top_by_tfidf

```python
HierarchicalLayer.top_by_tfidf(self, n: int = 10) -> list
```

Get top minicolumns by TF-IDF score.

#### HierarchicalLayer.top_by_activation

```python
HierarchicalLayer.top_by_activation(self, n: int = 10) -> list
```

Get top minicolumns by activation level.

#### HierarchicalLayer.to_dict

```python
HierarchicalLayer.to_dict(self) -> Dict
```

Convert layer to dictionary for serialization.

#### HierarchicalLayer.from_dict

```python
HierarchicalLayer.from_dict(cls, data: Dict) -> 'HierarchicalLayer'
```

Create a layer from dictionary representation.

### Dependencies

**Standard Library:**

- `enum.IntEnum`
- `minicolumn.Minicolumn`
- `typing.Dict`
- `typing.Iterator`
- `typing.Optional`



## minicolumn.py

Minicolumn Module
=================

Core data structure representing a cortical minicolumn.

In the neocortex, minicolumns are vertical structures containing
~80-100 neurons that respond to similar f...


### Classes

#### Edge

Typed edge with metadata for ConceptNet-style graph representation.

**Methods:**

- `to_dict`
- `from_dict`

#### Minicolumn

A minicolumn represents a single concept/feature at a given hierarchy level.

**Methods:**

- `lateral_connections`
- `lateral_connections`
- `add_lateral_connection`
- `add_lateral_connections_batch`
- `set_lateral_connection_weight`
- `add_typed_connection`
- `get_typed_connection`
- `get_connections_by_type`
- `get_connections_by_source`
- `add_feedforward_connection`
- `add_feedback_connection`
- `connection_count`
- `top_connections`
- `to_dict`
- `from_dict`

### Functions

#### Edge.to_dict

```python
Edge.to_dict(self) -> Dict
```

Convert to dictionary for serialization.

#### Edge.from_dict

```python
Edge.from_dict(cls, data: Dict) -> 'Edge'
```

Create an Edge from dictionary representation.

#### Minicolumn.lateral_connections

```python
Minicolumn.lateral_connections(self, value: Dict[str, float]) -> None
```

Set lateral connections from a dictionary (for deserialization).

#### Minicolumn.add_lateral_connection

```python
Minicolumn.add_lateral_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a lateral connection to another column.

#### Minicolumn.add_lateral_connections_batch

```python
Minicolumn.add_lateral_connections_batch(self, connections: Dict[str, float]) -> None
```

Add or strengthen multiple lateral connections at once.

#### Minicolumn.set_lateral_connection_weight

```python
Minicolumn.set_lateral_connection_weight(self, target_id: str, weight: float) -> None
```

Set the weight of a lateral connection directly (not additive).

#### Minicolumn.add_typed_connection

```python
Minicolumn.add_typed_connection(self, target_id: str, weight: float = 1.0, relation_type: str = 'co_occurrence', confidence: float = 1.0, source: str = 'corpus') -> None
```

Add or update a typed connection with metadata.

#### Minicolumn.get_typed_connection

```python
Minicolumn.get_typed_connection(self, target_id: str) -> Optional[Edge]
```

Get a typed connection by target ID.

#### Minicolumn.get_connections_by_type

```python
Minicolumn.get_connections_by_type(self, relation_type: str) -> List[Edge]
```

Get all typed connections with a specific relation type.

#### Minicolumn.get_connections_by_source

```python
Minicolumn.get_connections_by_source(self, source: str) -> List[Edge]
```

Get all typed connections from a specific source.

#### Minicolumn.add_feedforward_connection

```python
Minicolumn.add_feedforward_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a feedforward connection to a lower layer column.

#### Minicolumn.add_feedback_connection

```python
Minicolumn.add_feedback_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a feedback connection to a higher layer column.

#### Minicolumn.connection_count

```python
Minicolumn.connection_count(self) -> int
```

Return the number of lateral connections.

#### Minicolumn.top_connections

```python
Minicolumn.top_connections(self, n: int = 5) -> list
```

Get the strongest lateral connections.

#### Minicolumn.to_dict

```python
Minicolumn.to_dict(self) -> Dict
```

Convert to dictionary for serialization.

#### Minicolumn.from_dict

```python
Minicolumn.from_dict(cls, data: Dict) -> 'Minicolumn'
```

Create a minicolumn from dictionary representation.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Dict`
- `typing.List`
- ... and 2 more



## types.py

Type Aliases for the Cortical Text Processor.

This module provides type aliases for complex return types used throughout
the library, making function signatures more readable and maintainable.

Task ...


### Dependencies

**Standard Library:**

- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## NLP Components

Natural language processing components for tokenization and semantics.

## Modules

- **embeddings.py**: Embeddings Module
- **semantics.py**: Semantics Module
- **tokenizer.py**: Tokenizer Module


## embeddings.py

Embeddings Module
=================

Graph-based embeddings for the cortical network.

Implements three methods for computing term embeddings from the
connection graph structure:
1. Adjacency: Direct ...


### Functions

#### compute_graph_embeddings

```python
compute_graph_embeddings(layers: Dict[CorticalLayer, HierarchicalLayer], dimensions: int = 64, method: str = 'adjacency', max_terms: Optional[int] = None) -> Tuple[Dict[str, List[float]], Dict[str, Any]]
```

Compute embeddings for tokens based on graph structure.

#### embedding_similarity

```python
embedding_similarity(embeddings: Dict[str, List[float]], term1: str, term2: str) -> float
```

Compute cosine similarity between two term embeddings.

#### find_similar_by_embedding

```python
find_similar_by_embedding(embeddings: Dict[str, List[float]], term: str, top_n: int = 10) -> List[Tuple[str, float]]
```

Find terms most similar to a given term by embedding.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- `random`
- ... and 5 more



## semantics.py

Semantics Module
================

Corpus-derived semantic relations and retrofitting.

Extracts semantic relationships from co-occurrence patterns,
then uses them to adjust connection weights (retrof...


### Functions

#### extract_pattern_relations

```python
extract_pattern_relations(documents: Dict[str, str], valid_terms: Set[str], min_confidence: float = 0.5) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations using pattern matching on document text.

#### get_pattern_statistics

```python
get_pattern_statistics(relations: List[Tuple[str, str, str, float]]) -> Dict[str, Any]
```

Get statistics about extracted pattern-based relations.

#### extract_corpus_semantics

```python
extract_corpus_semantics(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], tokenizer, window_size: int = 5, min_cooccurrence: int = 2, use_pattern_extraction: bool = True, min_pattern_confidence: float = 0.6, max_similarity_pairs: int = 100000, min_context_keys: int = 3) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations from corpus co-occurrence patterns.

#### retrofit_connections

```python
retrofit_connections(layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: List[Tuple[str, str, str, float]], iterations: int = 10, alpha: float = 0.3) -> Dict[str, Any]
```

Retrofit lateral connections using semantic relations.

#### retrofit_embeddings

```python
retrofit_embeddings(embeddings: Dict[str, List[float]], semantic_relations: List[Tuple[str, str, str, float]], iterations: int = 10, alpha: float = 0.4) -> Dict[str, Any]
```

Retrofit embeddings using semantic relations.

#### get_relation_type_weight

```python
get_relation_type_weight(relation_type: str) -> float
```

Get the weight for a relation type.

#### build_isa_hierarchy

```python
build_isa_hierarchy(semantic_relations: List[Tuple[str, str, str, float]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
```

Build IsA parent-child hierarchy from semantic relations.

#### get_ancestors

```python
get_ancestors(term: str, parents: Dict[str, Set[str]], max_depth: int = 10) -> Dict[str, int]
```

Get all ancestors of a term with their depth in the hierarchy.

#### get_descendants

```python
get_descendants(term: str, children: Dict[str, Set[str]], max_depth: int = 10) -> Dict[str, int]
```

Get all descendants of a term with their depth in the hierarchy.

#### inherit_properties

```python
inherit_properties(semantic_relations: List[Tuple[str, str, str, float]], decay_factor: float = 0.7, max_depth: int = 5) -> Dict[str, Dict[str, Tuple[float, str, int]]]
```

Compute inherited properties for all terms based on IsA hierarchy.

#### compute_property_similarity

```python
compute_property_similarity(term1: str, term2: str, inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]], direct_properties: Optional[Dict[str, Dict[str, float]]] = None) -> float
```

Compute similarity between terms based on shared properties (direct + inherited).

#### apply_inheritance_to_connections

```python
apply_inheritance_to_connections(layers: Dict[CorticalLayer, HierarchicalLayer], inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]], boost_factor: float = 0.3) -> Dict[str, Any]
```

Boost lateral connections between terms that share inherited properties.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.RELATION_WEIGHTS`
- `copy`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 10 more



## tokenizer.py

Tokenizer Module
================

Text tokenization with stemming and word variant support.

Like early visual processing, the tokenizer extracts basic features
(words) from raw input, filtering nois...


### Classes

#### Tokenizer

Text tokenizer with stemming and word variant support.

**Methods:**

- `tokenize`
- `extract_ngrams`
- `stem`
- `get_word_variants`
- `add_word_mapping`

### Functions

#### split_identifier

```python
split_identifier(identifier: str) -> List[str]
```

Split a code identifier into component words.

#### Tokenizer.tokenize

```python
Tokenizer.tokenize(self, text: str, split_identifiers: Optional[bool] = None) -> List[str]
```

Extract tokens from text.

#### Tokenizer.extract_ngrams

```python
Tokenizer.extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]
```

Extract n-grams from token list.

#### Tokenizer.stem

```python
Tokenizer.stem(self, word: str) -> str
```

Apply simple suffix stripping (Porter-lite stemming).

#### Tokenizer.get_word_variants

```python
Tokenizer.get_word_variants(self, word: str) -> List[str]
```

Get related words/variants for query expansion.

#### Tokenizer.add_word_mapping

```python
Tokenizer.add_word_mapping(self, word: str, variants: List[str]) -> None
```

Add a custom word mapping for query expansion.

### Dependencies

**Standard Library:**

- `re`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Set`
- ... and 1 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Observability

Metrics collection and progress tracking.

## Modules

- **observability.py**: Observability Module
- **progress.py**: Progress reporting infrastructure for long-running operations.
- **results.py**: Result Dataclasses for Cortical Text Processor


## observability.py

Observability Module
====================

Provides timing hooks, metrics collection, and trace context for monitoring
the Cortical Text Processor's performance and operations.

This module follows th...


### Classes

#### MetricsCollector

Collects and aggregates timing and count metrics for operations.

**Methods:**

- `record_timing`
- `record_count`
- `get_operation_stats`
- `get_all_stats`
- `get_trace`
- `reset`
- `enable`
- `disable`
- `trace_context`
- `get_summary`

#### TraceContext

Context for request tracing across operations.

**Methods:**

- `elapsed_ms`

### Functions

#### timed

```python
timed(operation_name: Optional[str] = None, include_args: bool = False)
```

Decorator for timing method calls and recording to metrics.

#### measure_time

```python
measure_time(func: Callable) -> Callable
```

Simple timing decorator that logs execution time.

#### get_global_metrics

```python
get_global_metrics() -> MetricsCollector
```

Get the global metrics collector instance.

#### enable_global_metrics

```python
enable_global_metrics() -> None
```

Enable global metrics collection.

#### disable_global_metrics

```python
disable_global_metrics() -> None
```

Disable global metrics collection.

#### reset_global_metrics

```python
reset_global_metrics() -> None
```

Reset global metrics.

#### MetricsCollector.record_timing

```python
MetricsCollector.record_timing(self, operation: str, duration_ms: float, trace_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None
```

Record a timing measurement for an operation.

#### MetricsCollector.record_count

```python
MetricsCollector.record_count(self, metric_name: str, count: int = 1) -> None
```

Record a simple count metric.

#### MetricsCollector.get_operation_stats

```python
MetricsCollector.get_operation_stats(self, operation: str) -> Dict[str, Any]
```

Get statistics for a specific operation.

#### MetricsCollector.get_all_stats

```python
MetricsCollector.get_all_stats(self) -> Dict[str, Dict[str, Any]]
```

Get statistics for all operations.

#### MetricsCollector.get_trace

```python
MetricsCollector.get_trace(self, trace_id: str) -> List[tuple]
```

Get all operations recorded for a trace ID.

#### MetricsCollector.reset

```python
MetricsCollector.reset(self) -> None
```

Clear all collected metrics.

#### MetricsCollector.enable

```python
MetricsCollector.enable(self) -> None
```

Enable metrics collection.

#### MetricsCollector.disable

```python
MetricsCollector.disable(self) -> None
```

Disable metrics collection.

#### MetricsCollector.trace_context

```python
MetricsCollector.trace_context(self, trace_id: str)
```

Context manager for tracing a block of operations.

#### MetricsCollector.get_summary

```python
MetricsCollector.get_summary(self) -> str
```

Get a human-readable summary of all metrics.

#### TraceContext.elapsed_ms

```python
TraceContext.elapsed_ms(self) -> float
```

Get elapsed time since trace started in milliseconds.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `contextlib.contextmanager`
- `functools`
- `logging`
- `time`
- ... and 5 more



## progress.py

Progress reporting infrastructure for long-running operations.

This module provides a flexible progress reporting system that supports:
- Console output with nice formatting
- Custom callbacks for in...


### Classes

#### ProgressReporter

Protocol for progress reporters.

**Methods:**

- `update`
- `complete`

#### ConsoleProgressReporter

Console-based progress reporter with nice formatting.

**Methods:**

- `update`
- `complete`

#### CallbackProgressReporter

Progress reporter that calls a custom callback function.

**Methods:**

- `update`
- `complete`

#### SilentProgressReporter

No-op progress reporter for silent operation.

**Methods:**

- `update`
- `complete`

#### MultiPhaseProgress

Helper for tracking progress across multiple sequential phases.

**Methods:**

- `start_phase`
- `update`
- `complete_phase`
- `overall_progress`

### Functions

#### ProgressReporter.update

```python
ProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Update progress for a specific phase.

#### ProgressReporter.complete

```python
ProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Mark a phase as complete.

#### ConsoleProgressReporter.update

```python
ConsoleProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Update progress display.

#### ConsoleProgressReporter.complete

```python
ConsoleProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Mark phase as complete and move to new line.

#### CallbackProgressReporter.update

```python
CallbackProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Call callback with progress update.

#### CallbackProgressReporter.complete

```python
CallbackProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Call callback with completion notification.

#### SilentProgressReporter.update

```python
SilentProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Do nothing.

#### SilentProgressReporter.complete

```python
SilentProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Do nothing.

#### MultiPhaseProgress.start_phase

```python
MultiPhaseProgress.start_phase(self, phase: str) -> None
```

Start a new phase.

#### MultiPhaseProgress.update

```python
MultiPhaseProgress.update(self, percent: float, message: Optional[str] = None) -> None
```

Update progress within current phase.

#### MultiPhaseProgress.complete_phase

```python
MultiPhaseProgress.complete_phase(self, message: Optional[str] = None) -> None
```

Mark current phase as complete.

#### MultiPhaseProgress.overall_progress

```python
MultiPhaseProgress.overall_progress(self) -> float
```

Get overall progress across all phases (0-100).

### Dependencies

**Standard Library:**

- `abc.ABC`
- `abc.abstractmethod`
- `sys`
- `time`
- `typing.Any`
- ... and 4 more



## results.py

Result Dataclasses for Cortical Text Processor
===============================================

Strongly-typed result containers for query operations that provide
IDE autocomplete and type checking su...


### Classes

#### DocumentMatch

A document search result with relevance score.

**Methods:**

- `to_dict`
- `to_tuple`
- `from_tuple`
- `from_dict`

#### PassageMatch

A passage retrieval result with text, location, and relevance score.

**Methods:**

- `to_dict`
- `to_tuple`
- `location`
- `length`
- `from_tuple`
- `from_dict`

#### QueryResult

Complete query result with matches and metadata.

**Methods:**

- `to_dict`
- `top_match`
- `match_count`
- `average_score`
- `from_dict`

### Functions

#### convert_document_matches

```python
convert_document_matches(results: List[tuple], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[DocumentMatch]
```

Convert list of (doc_id, score) tuples to DocumentMatch objects.

#### convert_passage_matches

```python
convert_passage_matches(results: List[tuple], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[PassageMatch]
```

Convert list of (doc_id, text, start, end, score) tuples to PassageMatch objects.

#### DocumentMatch.to_dict

```python
DocumentMatch.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### DocumentMatch.to_tuple

```python
DocumentMatch.to_tuple(self) -> tuple
```

Convert to tuple format (doc_id, score).

#### DocumentMatch.from_tuple

```python
DocumentMatch.from_tuple(cls, doc_id: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'DocumentMatch'
```

Create from tuple format (doc_id, score).

#### DocumentMatch.from_dict

```python
DocumentMatch.from_dict(cls, data: Dict[str, Any]) -> 'DocumentMatch'
```

Create from dictionary.

#### PassageMatch.to_dict

```python
PassageMatch.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### PassageMatch.to_tuple

```python
PassageMatch.to_tuple(self) -> tuple
```

Convert to tuple format (doc_id, text, start, end, score).

#### PassageMatch.location

```python
PassageMatch.location(self) -> str
```

Get citation-style location string.

#### PassageMatch.length

```python
PassageMatch.length(self) -> int
```

Get passage length in characters.

#### PassageMatch.from_tuple

```python
PassageMatch.from_tuple(cls, doc_id: str, text: str, start: int, end: int, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'PassageMatch'
```

Create from tuple format (doc_id, text, start, end, score).

#### PassageMatch.from_dict

```python
PassageMatch.from_dict(cls, data: Dict[str, Any]) -> 'PassageMatch'
```

Create from dictionary.

#### QueryResult.to_dict

```python
QueryResult.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary with nested match dicts.

#### QueryResult.top_match

```python
QueryResult.top_match(self) -> Union[DocumentMatch, PassageMatch, None]
```

Get the highest-scoring match.

#### QueryResult.match_count

```python
QueryResult.match_count(self) -> int
```

Get number of matches.

#### QueryResult.average_score

```python
QueryResult.average_score(self) -> float
```

Get average relevance score across all matches.

#### QueryResult.from_dict

```python
QueryResult.from_dict(cls, data: Dict[str, Any]) -> 'QueryResult'
```

Create from dictionary.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Any`
- `typing.Dict`
- ... and 3 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Persistence Layer

Save and load functionality for maintaining processor state.

## Modules

- **chunk_index.py**: Chunk-based indexing for git-compatible corpus storage.
- **persistence.py**: Persistence Module
- **state_storage.py**: Git-friendly State Storage Module


## chunk_index.py

Chunk-based indexing for git-compatible corpus storage.

This module provides append-only, time-stamped JSON chunks that can be
safely committed to git without merge conflicts. Each indexing session
c...


### Classes

#### ChunkOperation

A single operation in a chunk (add, modify, or delete).

**Methods:**

- `to_dict`
- `from_dict`

#### Chunk

A chunk containing operations from a single indexing session.

**Methods:**

- `to_dict`
- `from_dict`
- `get_filename`

#### ChunkWriter

Writes indexing session changes to timestamped JSON chunks.

**Methods:**

- `add_document`
- `modify_document`
- `delete_document`
- `has_operations`
- `save`

#### ChunkLoader

Loads and combines chunks to rebuild document state.

**Methods:**

- `get_chunk_files`
- `load_chunk`
- `load_all`
- `get_documents`
- `get_mtimes`
- `get_metadata`
- `get_chunks`
- `compute_hash`
- `is_cache_valid`
- `save_cache_hash`
- `get_stats`

#### ChunkCompactor

Compacts multiple chunk files into a single file.

**Methods:**

- `compact`

### Functions

#### get_changes_from_manifest

```python
get_changes_from_manifest(current_files: Dict[str, float], manifest: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]
```

Compare current files to manifest to find changes.

#### ChunkOperation.to_dict

```python
ChunkOperation.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### ChunkOperation.from_dict

```python
ChunkOperation.from_dict(cls, d: Dict[str, Any]) -> 'ChunkOperation'
```

Create from dictionary.

#### Chunk.to_dict

```python
Chunk.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### Chunk.from_dict

```python
Chunk.from_dict(cls, d: Dict[str, Any]) -> 'Chunk'
```

Create from dictionary.

#### Chunk.get_filename

```python
Chunk.get_filename(self) -> str
```

Generate filename for this chunk.

#### ChunkWriter.add_document

```python
ChunkWriter.add_document(self, doc_id: str, content: str, mtime: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
```

Record an add operation.

#### ChunkWriter.modify_document

```python
ChunkWriter.modify_document(self, doc_id: str, content: str, mtime: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
```

Record a modify operation.

#### ChunkWriter.delete_document

```python
ChunkWriter.delete_document(self, doc_id: str)
```

Record a delete operation.

#### ChunkWriter.has_operations

```python
ChunkWriter.has_operations(self) -> bool
```

Check if any operations were recorded.

#### ChunkWriter.save

```python
ChunkWriter.save(self, warn_size_kb: int = DEFAULT_WARN_SIZE_KB) -> Optional[Path]
```

Save chunk to file.

#### ChunkLoader.get_chunk_files

```python
ChunkLoader.get_chunk_files(self) -> List[Path]
```

Get all chunk files sorted by timestamp.

#### ChunkLoader.load_chunk

```python
ChunkLoader.load_chunk(self, filepath: Path) -> Chunk
```

Load a single chunk file.

#### ChunkLoader.load_all

```python
ChunkLoader.load_all(self) -> Dict[str, str]
```

Load all chunks and replay operations to get current document state.

#### ChunkLoader.get_documents

```python
ChunkLoader.get_documents(self) -> Dict[str, str]
```

Get loaded documents (calls load_all if needed).

#### ChunkLoader.get_mtimes

```python
ChunkLoader.get_mtimes(self) -> Dict[str, float]
```

Get document modification times.

#### ChunkLoader.get_metadata

```python
ChunkLoader.get_metadata(self) -> Dict[str, Dict[str, Any]]
```

Get document metadata (doc_type, headings, etc.).

#### ChunkLoader.get_chunks

```python
ChunkLoader.get_chunks(self) -> List[Chunk]
```

Get loaded chunks.

#### ChunkLoader.compute_hash

```python
ChunkLoader.compute_hash(self) -> str
```

Compute hash of current document state.

#### ChunkLoader.is_cache_valid

```python
ChunkLoader.is_cache_valid(self, cache_path: str, cache_hash_path: Optional[str] = None) -> bool
```

Check if pkl cache is valid for current chunk state.

#### ChunkLoader.save_cache_hash

```python
ChunkLoader.save_cache_hash(self, cache_path: str, cache_hash_path: Optional[str] = None)
```

Save current document hash for cache validation.

#### ChunkLoader.get_stats

```python
ChunkLoader.get_stats(self) -> Dict[str, Any]
```

Get statistics about loaded chunks.

#### ChunkCompactor.compact

```python
ChunkCompactor.compact(self, before: Optional[str] = None, keep_recent: int = 0, dry_run: bool = False) -> Dict[str, Any]
```

Compact chunks into a single chunk.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `hashlib`
- ... and 11 more



## persistence.py

Persistence Module
==================

Save and load functionality for the cortical processor.

Supports:
- Pickle serialization for full state
- JSON export for graph visualization
- Incremental upda...


### Classes

#### SignatureVerificationError

Raised when HMAC signature verification fails.

### Functions

#### save_processor

```python
save_processor(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, embeddings: Optional[Dict[str, list]] = None, semantic_relations: Optional[list] = None, metadata: Optional[Dict] = None, verbose: bool = True, format: str = 'pickle', signing_key: Optional[bytes] = None) -> None
```

Save processor state to a file.

#### load_processor

```python
load_processor(filepath: str, verbose: bool = True, format: Optional[str] = None, verify_key: Optional[bytes] = None) -> tuple
```

Load processor state from a file.

#### export_graph_json

```python
export_graph_json(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], layer_filter: Optional[CorticalLayer] = None, min_weight: float = 0.0, max_nodes: int = 500, verbose: bool = True) -> Dict
```

Export graph structure as JSON for visualization.

#### export_embeddings_json

```python
export_embeddings_json(filepath: str, embeddings: Dict[str, list], metadata: Optional[Dict] = None) -> None
```

Export embeddings as JSON.

#### load_embeddings_json

```python
load_embeddings_json(filepath: str) -> Dict[str, list]
```

Load embeddings from JSON.

#### export_semantic_relations_json

```python
export_semantic_relations_json(filepath: str, relations: list) -> None
```

Export semantic relations as JSON.

#### load_semantic_relations_json

```python
load_semantic_relations_json(filepath: str) -> list
```

Load semantic relations from JSON.

#### get_state_summary

```python
get_state_summary(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> Dict
```

Get a summary of the current processor state.

#### export_conceptnet_json

```python
export_conceptnet_json(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: Optional[list] = None, include_cross_layer: bool = True, include_typed_edges: bool = True, min_weight: float = 0.0, min_confidence: float = 0.0, max_nodes_per_layer: int = 100, verbose: bool = True) -> Dict[str, Any]
```

Export ConceptNet-style graph for visualization.

### Dependencies

**Standard Library:**

- `hashlib`
- `hmac`
- `json`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 9 more



## state_storage.py

Git-friendly State Storage Module
=================================

Replaces pickle-based persistence with JSON files that:
- Can be diff'd and reviewed in git
- Won't cause merge conflicts
- Support...


### Classes

#### StateManifest

Manifest file tracking state version and component checksums.

**Methods:**

- `to_dict`
- `from_dict`
- `update_checksum`

#### StateWriter

Writes processor state to git-friendly JSON files.

**Methods:**

- `save_layer`
- `save_documents`
- `save_semantic_relations`
- `save_embeddings`
- `save_manifest`
- `save_all`

#### StateLoader

Loads processor state from git-friendly JSON files.

**Methods:**

- `exists`
- `load_manifest`
- `validate_checksum`
- `load_layer`
- `load_documents`
- `load_semantic_relations`
- `load_embeddings`
- `load_all`
- `get_stats`

### Functions

#### migrate_pkl_to_json

```python
migrate_pkl_to_json(pkl_path: str, json_dir: str, verbose: bool = True) -> bool
```

Migrate a pickle file to git-friendly JSON format.

#### StateManifest.to_dict

```python
StateManifest.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### StateManifest.from_dict

```python
StateManifest.from_dict(cls, data: Dict[str, Any]) -> 'StateManifest'
```

Create manifest from dictionary.

#### StateManifest.update_checksum

```python
StateManifest.update_checksum(self, component: str, content: str) -> bool
```

Update checksum for a component.

#### StateWriter.save_layer

```python
StateWriter.save_layer(self, layer: HierarchicalLayer, force: bool = False) -> bool
```

Save a single layer to its JSON file.

#### StateWriter.save_documents

```python
StateWriter.save_documents(self, documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, force: bool = False) -> bool
```

Save documents and metadata.

#### StateWriter.save_semantic_relations

```python
StateWriter.save_semantic_relations(self, relations: List[Tuple], force: bool = False) -> bool
```

Save semantic relations.

#### StateWriter.save_embeddings

```python
StateWriter.save_embeddings(self, embeddings: Dict[str, List[float]], force: bool = False) -> bool
```

Save graph embeddings.

#### StateWriter.save_manifest

```python
StateWriter.save_manifest(self) -> None
```

Save the manifest file.

#### StateWriter.save_all

```python
StateWriter.save_all(self, layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, embeddings: Optional[Dict[str, List[float]]] = None, semantic_relations: Optional[List[Tuple]] = None, stale_computations: Optional[Set[str]] = None, force: bool = False, verbose: bool = True) -> Dict[str, bool]
```

Save all processor state.

#### StateLoader.exists

```python
StateLoader.exists(self) -> bool
```

Check if state directory exists and has manifest.

#### StateLoader.load_manifest

```python
StateLoader.load_manifest(self) -> StateManifest
```

Load the manifest file.

#### StateLoader.validate_checksum

```python
StateLoader.validate_checksum(self, component: str, filepath: Path) -> bool
```

Validate a component's checksum.

#### StateLoader.load_layer

```python
StateLoader.load_layer(self, level: int) -> HierarchicalLayer
```

Load a single layer.

#### StateLoader.load_documents

```python
StateLoader.load_documents(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]
```

Load documents and metadata.

#### StateLoader.load_semantic_relations

```python
StateLoader.load_semantic_relations(self) -> List[Tuple]
```

Load semantic relations.

#### StateLoader.load_embeddings

```python
StateLoader.load_embeddings(self) -> Dict[str, List[float]]
```

Load graph embeddings.

#### StateLoader.load_all

```python
StateLoader.load_all(self, validate: bool = True, verbose: bool = True) -> Tuple[Dict[CorticalLayer, HierarchicalLayer], Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, List[float]], List[Tuple], Dict[str, Any]]
```

Load all processor state.

#### StateLoader.get_stats

```python
StateLoader.get_stats(self) -> Dict[str, Any]
```

Get statistics about stored state without loading everything.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `hashlib`
- ... and 13 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Core Processor

The core processor orchestrates all text processing operations.

## Modules

- **__init__.py**: Cortical Text Processor - Main processor package.
- **compute.py**: Compute methods: analysis, clustering, embeddings, semantic extraction.
- **core.py**: Core processor functionality: initialization, staleness tracking, and layer management.
- **documents.py**: Document management: processing, adding, removing, and metadata handling.
- **introspection.py**: Introspection: state inspection, fingerprints, gaps, and summaries.
- **persistence_api.py**: Persistence API: save, load, export, and migration methods.


## __init__.py

Cortical Text Processor - Main processor package.

This package splits the monolithic processor.py into focused modules:
- core.py: Initialization, staleness tracking, layer management
- documents.py:...


### Classes

#### CorticalTextProcessor

Neocortex-inspired text processing system.

### Dependencies

**Standard Library:**

- `compute.ComputeMixin`
- `core.CoreMixin`
- `documents.DocumentsMixin`
- `introspection.IntrospectionMixin`
- `persistence_api.PersistenceMixin`
- ... and 1 more



## compute.py

Compute methods: analysis, clustering, embeddings, semantic extraction.

This module contains all methods that perform computational analysis on the corpus,
including PageRank, TF-IDF, clustering, and...


### Classes

#### ComputeMixin

Mixin providing computation functionality.

**Methods:**

- `recompute`
- `compute_all`
- `resume_from_checkpoint`
- `propagate_activation`
- `compute_importance`
- `compute_semantic_importance`
- `compute_hierarchical_importance`
- `compute_tfidf`
- `compute_bm25`
- `compute_document_connections`
- `compute_bigram_connections`
- `build_concept_clusters`
- `compute_clustering_quality`
- `compute_concept_connections`
- `extract_corpus_semantics`
- `extract_pattern_relations`
- `retrofit_connections`
- `compute_property_inheritance`
- `compute_property_similarity`
- `compute_graph_embeddings`
- `retrofit_embeddings`
- `embedding_similarity`
- `find_similar_by_embedding`

### Functions

#### ComputeMixin.recompute

```python
ComputeMixin.recompute(self, level: str = 'stale', verbose: bool = True) -> Dict[str, bool]
```

Recompute specified analysis levels.

#### ComputeMixin.compute_all

```python
ComputeMixin.compute_all(self, verbose: bool = True, build_concepts: bool = True, pagerank_method: str = 'standard', connection_strategy: str = 'document_overlap', cluster_strictness: float = 1.0, bridge_weight: float = 0.0, progress_callback: Optional[ProgressReporter] = None, show_progress: bool = False, checkpoint_dir: Optional[str] = None, resume: bool = False) -> Dict[str, Any]
```

Run all computation steps.

#### ComputeMixin.resume_from_checkpoint

```python
ComputeMixin.resume_from_checkpoint(cls, checkpoint_dir: str, config: Optional['CorticalConfig'] = None, verbose: bool = True) -> 'CorticalTextProcessor'
```

Resume processing from a checkpoint directory.

#### ComputeMixin.propagate_activation

```python
ComputeMixin.propagate_activation(self, iterations: int = 3, decay: float = 0.8, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_importance

```python
ComputeMixin.compute_importance(self, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_semantic_importance

```python
ComputeMixin.compute_semantic_importance(self, relation_weights: Optional[Dict[str, float]] = None, verbose: bool = True) -> Dict[str, Any]
```

Compute PageRank with semantic relation weighting.

#### ComputeMixin.compute_hierarchical_importance

```python
ComputeMixin.compute_hierarchical_importance(self, layer_iterations: int = 10, global_iterations: int = 5, cross_layer_damping: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]
```

Compute PageRank with cross-layer propagation.

#### ComputeMixin.compute_tfidf

```python
ComputeMixin.compute_tfidf(self, verbose: bool = True) -> None
```

Compute document relevance scores using the configured algorithm.

#### ComputeMixin.compute_bm25

```python
ComputeMixin.compute_bm25(self, k1: float = None, b: float = None, verbose: bool = True) -> None
```

Compute BM25 scores for document relevance ranking.

#### ComputeMixin.compute_document_connections

```python
ComputeMixin.compute_document_connections(self, min_shared_terms: int = 3, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_bigram_connections

```python
ComputeMixin.compute_bigram_connections(self, min_shared_docs: int = 1, component_weight: float = 0.5, chain_weight: float = 0.7, cooccurrence_weight: float = 0.3, max_bigrams_per_term: int = 100, max_bigrams_per_doc: int = 500, max_connections_per_bigram: int = 50, verbose: bool = True) -> Dict[str, Any]
```

Build lateral connections between bigrams based on shared components and co-occurrence.

#### ComputeMixin.build_concept_clusters

```python
ComputeMixin.build_concept_clusters(self, min_cluster_size: Optional[int] = None, clustering_method: str = 'louvain', cluster_strictness: Optional[float] = None, bridge_weight: float = 0.0, resolution: Optional[float] = None, verbose: bool = True) -> Dict[int, List[str]]
```

Build concept clusters from token layer.

#### ComputeMixin.compute_clustering_quality

```python
ComputeMixin.compute_clustering_quality(self, sample_size: int = 500) -> Dict[str, Any]
```

Compute clustering quality metrics for the concept layer.

#### ComputeMixin.compute_concept_connections

```python
ComputeMixin.compute_concept_connections(self, use_semantics: bool = True, min_shared_docs: int = 1, min_jaccard: float = 0.1, use_member_semantics: bool = False, use_embedding_similarity: bool = False, embedding_threshold: float = 0.3, verbose: bool = True) -> Dict[str, Any]
```

Build lateral connections between concepts based on document overlap and semantics.

#### ComputeMixin.extract_corpus_semantics

```python
ComputeMixin.extract_corpus_semantics(self, use_pattern_extraction: bool = True, min_pattern_confidence: float = 0.6, max_similarity_pairs: int = 100000, min_context_keys: int = 3, verbose: bool = True) -> int
```

Extract semantic relations from the corpus.

#### ComputeMixin.extract_pattern_relations

```python
ComputeMixin.extract_pattern_relations(self, min_confidence: float = 0.6, verbose: bool = True) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations using pattern matching only.

#### ComputeMixin.retrofit_connections

```python
ComputeMixin.retrofit_connections(self, iterations: int = 10, alpha: float = 0.3, verbose: bool = True) -> Dict
```

None

#### ComputeMixin.compute_property_inheritance

```python
ComputeMixin.compute_property_inheritance(self, decay_factor: float = 0.7, max_depth: int = 5, apply_to_connections: bool = True, boost_factor: float = 0.3, verbose: bool = True) -> Dict[str, Any]
```

Compute property inheritance based on IsA hierarchy.

#### ComputeMixin.compute_property_similarity

```python
ComputeMixin.compute_property_similarity(self, term1: str, term2: str) -> float
```

Compute similarity between terms based on shared properties.

#### ComputeMixin.compute_graph_embeddings

```python
ComputeMixin.compute_graph_embeddings(self, dimensions: int = 64, method: str = 'fast', max_terms: Optional[int] = None, verbose: bool = True) -> Dict
```

Compute graph embeddings for tokens.

#### ComputeMixin.retrofit_embeddings

```python
ComputeMixin.retrofit_embeddings(self, iterations: int = 10, alpha: float = 0.4, verbose: bool = True) -> Dict
```

None

#### ComputeMixin.embedding_similarity

```python
ComputeMixin.embedding_similarity(self, term1: str, term2: str) -> float
```

None

#### ComputeMixin.find_similar_by_embedding

```python
ComputeMixin.find_similar_by_embedding(self, term: str, top_n: int = 10) -> List[Tuple[str, float]]
```

None

### Dependencies

**Standard Library:**

- `datetime.datetime`
- `json`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- ... and 11 more

**Local Imports:**

- `.analysis`
- `.embeddings`
- `.semantics`



## core.py

Core processor functionality: initialization, staleness tracking, and layer management.

This module contains the base class definition and core infrastructure that all
other processor mixins depend o...


### Classes

#### CoreMixin

Core mixin providing initialization and staleness tracking.

**Methods:**

- `is_stale`
- `get_stale_computations`
- `get_layer`
- `get_metrics`
- `get_metrics_summary`
- `reset_metrics`
- `enable_metrics`
- `disable_metrics`
- `record_metric`

### Functions

#### CoreMixin.is_stale

```python
CoreMixin.is_stale(self, computation_type: str) -> bool
```

Check if a specific computation is stale.

#### CoreMixin.get_stale_computations

```python
CoreMixin.get_stale_computations(self) -> set
```

Get the set of computations that are currently stale.

#### CoreMixin.get_layer

```python
CoreMixin.get_layer(self, layer: CorticalLayer) -> HierarchicalLayer
```

Get a specific layer by enum.

#### CoreMixin.get_metrics

```python
CoreMixin.get_metrics(self) -> Dict[str, Dict[str, Any]]
```

Get all collected metrics.

#### CoreMixin.get_metrics_summary

```python
CoreMixin.get_metrics_summary(self) -> str
```

Get a human-readable summary of all metrics.

#### CoreMixin.reset_metrics

```python
CoreMixin.reset_metrics(self) -> None
```

Clear all collected metrics.

#### CoreMixin.enable_metrics

```python
CoreMixin.enable_metrics(self) -> None
```

Enable metrics collection.

#### CoreMixin.disable_metrics

```python
CoreMixin.disable_metrics(self) -> None
```

Disable metrics collection.

#### CoreMixin.record_metric

```python
CoreMixin.record_metric(self, metric_name: str, count: int = 1) -> None
```

Record a custom count metric.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `logging`
- `minicolumn.Minicolumn`
- ... and 5 more



## documents.py

Document management: processing, adding, removing, and metadata handling.

This module contains all methods related to managing documents in the corpus.


### Classes

#### DocumentsMixin

Mixin providing document management functionality.

**Methods:**

- `process_document`
- `set_document_metadata`
- `get_document_metadata`
- `get_all_document_metadata`
- `add_document_incremental`
- `add_documents_batch`
- `remove_document`
- `remove_documents_batch`

### Functions

#### DocumentsMixin.process_document

```python
DocumentsMixin.process_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, int]
```

Process a document and add it to the corpus.

#### DocumentsMixin.set_document_metadata

```python
DocumentsMixin.set_document_metadata(self, doc_id: str, **kwargs) -> None
```

Set or update metadata for a document.

#### DocumentsMixin.get_document_metadata

```python
DocumentsMixin.get_document_metadata(self, doc_id: str) -> Dict[str, Any]
```

Get metadata for a document.

#### DocumentsMixin.get_all_document_metadata

```python
DocumentsMixin.get_all_document_metadata(self) -> Dict[str, Dict[str, Any]]
```

Get metadata for all documents.

#### DocumentsMixin.add_document_incremental

```python
DocumentsMixin.add_document_incremental(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, recompute: str = 'tfidf') -> Dict[str, int]
```

Add a document with selective recomputation for efficiency.

#### DocumentsMixin.add_documents_batch

```python
DocumentsMixin.add_documents_batch(self, documents: List[Tuple[str, str, Optional[Dict[str, Any]]]], recompute: str = 'full', verbose: bool = True) -> Dict[str, Any]
```

Add multiple documents with a single recomputation.

#### DocumentsMixin.remove_document

```python
DocumentsMixin.remove_document(self, doc_id: str, verbose: bool = False) -> Dict[str, Any]
```

Remove a document from the corpus.

#### DocumentsMixin.remove_documents_batch

```python
DocumentsMixin.remove_documents_batch(self, doc_ids: List[str], recompute: str = 'none', verbose: bool = True) -> Dict[str, Any]
```

Remove multiple documents efficiently with single recomputation.

### Dependencies

**Standard Library:**

- `copy`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- `typing.Any`
- ... and 4 more



## introspection.py

Introspection: state inspection, fingerprints, gaps, and summaries.

This module contains methods for examining the processor state and
comparing texts/documents.


### Classes

#### IntrospectionMixin

Mixin providing introspection functionality.

**Methods:**

- `get_document_signature`
- `get_corpus_summary`
- `analyze_knowledge_gaps`
- `detect_anomalies`
- `get_fingerprint`
- `compare_fingerprints`
- `explain_fingerprint`
- `explain_similarity`
- `find_similar_texts`
- `compare_with`
- `compare_documents`
- `what_changed`
- `summarize_document`
- `detect_patterns`
- `detect_patterns_in_corpus`
- `get_pattern_summary`
- `get_corpus_pattern_statistics`
- `format_pattern_report`
- `list_available_patterns`
- `list_pattern_categories`

### Functions

#### IntrospectionMixin.get_document_signature

```python
IntrospectionMixin.get_document_signature(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]
```

Get the top-n TF-IDF terms for a document.

#### IntrospectionMixin.get_corpus_summary

```python
IntrospectionMixin.get_corpus_summary(self) -> Dict
```

Get summary statistics about the corpus.

#### IntrospectionMixin.analyze_knowledge_gaps

```python
IntrospectionMixin.analyze_knowledge_gaps(self) -> Dict
```

Analyze the corpus for knowledge gaps.

#### IntrospectionMixin.detect_anomalies

```python
IntrospectionMixin.detect_anomalies(self, threshold: float = 0.3) -> List[Dict]
```

Detect anomalous patterns in the corpus.

#### IntrospectionMixin.get_fingerprint

```python
IntrospectionMixin.get_fingerprint(self, text: str, top_n: int = 20) -> Dict
```

Compute the semantic fingerprint of a text.

#### IntrospectionMixin.compare_fingerprints

```python
IntrospectionMixin.compare_fingerprints(self, fp1: Dict, fp2: Dict) -> Dict
```

Compare two fingerprints and compute similarity metrics.

#### IntrospectionMixin.explain_fingerprint

```python
IntrospectionMixin.explain_fingerprint(self, fp: Dict, top_n: int = 10) -> Dict
```

Generate a human-readable explanation of a fingerprint.

#### IntrospectionMixin.explain_similarity

```python
IntrospectionMixin.explain_similarity(self, fp1: Dict, fp2: Dict) -> str
```

Generate a human-readable explanation of fingerprint similarity.

#### IntrospectionMixin.find_similar_texts

```python
IntrospectionMixin.find_similar_texts(self, text: str, candidates: List[Tuple[str, str]], top_n: int = 5) -> List[Tuple[str, float, Dict]]
```

Find texts most similar to the given text.

#### IntrospectionMixin.compare_with

```python
IntrospectionMixin.compare_with(self, other: 'CorticalTextProcessor', top_movers: int = 20, min_pagerank_delta: float = 0.0001) -> 'diff_module.SemanticDiff'
```

Compare this processor state with another to find semantic differences.

#### IntrospectionMixin.compare_documents

```python
IntrospectionMixin.compare_documents(self, doc_id_1: str, doc_id_2: str) -> Dict
```

Compare two documents within this corpus.

#### IntrospectionMixin.what_changed

```python
IntrospectionMixin.what_changed(self, old_content: str, new_content: str) -> Dict
```

Compare two text contents to show what changed semantically.

#### IntrospectionMixin.summarize_document

```python
IntrospectionMixin.summarize_document(self, doc_id: str, num_sentences: int = 3) -> str
```

Generate a summary of a document using extractive summarization.

#### IntrospectionMixin.detect_patterns

```python
IntrospectionMixin.detect_patterns(self, doc_id: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]
```

Detect programming patterns in a specific document.

#### IntrospectionMixin.detect_patterns_in_corpus

```python
IntrospectionMixin.detect_patterns_in_corpus(self, patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]
```

Detect patterns across all documents in the corpus.

#### IntrospectionMixin.get_pattern_summary

```python
IntrospectionMixin.get_pattern_summary(self, doc_id: str) -> Dict[str, int]
```

Get a summary of pattern occurrences in a document.

#### IntrospectionMixin.get_corpus_pattern_statistics

```python
IntrospectionMixin.get_corpus_pattern_statistics(self) -> Dict[str, Any]
```

Get pattern statistics across the entire corpus.

#### IntrospectionMixin.format_pattern_report

```python
IntrospectionMixin.format_pattern_report(self, doc_id: str, show_lines: bool = False) -> str
```

Format pattern detection results as a human-readable report.

#### IntrospectionMixin.list_available_patterns

```python
IntrospectionMixin.list_available_patterns(self) -> List[str]
```

List all available pattern names that can be detected.

#### IntrospectionMixin.list_pattern_categories

```python
IntrospectionMixin.list_pattern_categories(self) -> List[str]
```

List all pattern categories.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `logging`
- `re`
- `typing.Any`
- `typing.Dict`
- ... and 4 more

**Local Imports:**

- `.fingerprint`
- `.gaps`
- `.patterns`
- `.persistence`



## persistence_api.py

Persistence API: save, load, export, and migration methods.

This module contains all methods related to saving and loading processor state.


### Classes

#### PersistenceMixin

Mixin providing persistence functionality.

**Methods:**

- `save`
- `load`
- `save_json`
- `load_json`
- `migrate_to_json`
- `export_graph`
- `export_conceptnet_json`

### Functions

#### PersistenceMixin.save

```python
PersistenceMixin.save(self, filepath: str, verbose: bool = True, signing_key: Optional[bytes] = None) -> None
```

Save processor state to a file.

#### PersistenceMixin.load

```python
PersistenceMixin.load(cls, filepath: str, verbose: bool = True, verify_key: Optional[bytes] = None) -> 'CorticalTextProcessor'
```

Load processor state from a file.

#### PersistenceMixin.save_json

```python
PersistenceMixin.save_json(self, state_dir: str, force: bool = False, verbose: bool = True) -> Dict[str, bool]
```

Save processor state to git-friendly JSON format.

#### PersistenceMixin.load_json

```python
PersistenceMixin.load_json(cls, state_dir: str, config: Optional[CorticalConfig] = None, verbose: bool = True) -> 'CorticalTextProcessor'
```

Load processor from git-friendly JSON format.

#### PersistenceMixin.migrate_to_json

```python
PersistenceMixin.migrate_to_json(self, pkl_path: str, json_dir: str, verbose: bool = True) -> bool
```

Migrate existing pickle file to git-friendly JSON format.

#### PersistenceMixin.export_graph

```python
PersistenceMixin.export_graph(self, filepath: str, layer: Optional[CorticalLayer] = None, max_nodes: int = 500) -> Dict
```

Export graph to JSON for visualization.

#### PersistenceMixin.export_conceptnet_json

```python
PersistenceMixin.export_conceptnet_json(self, filepath: str, include_cross_layer: bool = True, include_typed_edges: bool = True, min_weight: float = 0.0, min_confidence: float = 0.0, max_nodes_per_layer: int = 100, verbose: bool = True) -> Dict[str, Any]
```

Export ConceptNet-style graph for visualization.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- `typing.Any`
- ... and 3 more

**Local Imports:**

- `.persistence`
- `.state_storage`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Search & Retrieval

Search and retrieval components for finding relevant documents and passages.

## Modules

- **__init__.py**: Query Module
- **chunking.py**: Chunking Module
- **definitions.py**: Definition Search Module
- **expansion.py**: Query Expansion Module
- **intent.py**: Intent Query Module
- **passages.py**: Passage Retrieval Module
- **ranking.py**: Ranking Module
- **search.py**: Document Search Module


## __init__.py

Query Module
============

Query expansion and search functionality.

This package provides methods for expanding queries using lateral connections,
concept clusters, and word variants, then searching...


### Dependencies

**Standard Library:**

- `analogy.complete_analogy`
- `analogy.complete_analogy_simple`
- `analogy.find_relation_between`
- `analogy.find_terms_with_relation`
- `chunking.CODE_BOUNDARY_PATTERN`
- ... and 49 more



## chunking.py

Chunking Module
==============

Functions for splitting documents into chunks for passage retrieval.

This module provides:
- Fixed-size text chunking with overlap
- Code-aware chunking aligned to sem...


### Functions

#### create_chunks

```python
create_chunks(text: str, chunk_size: int = 512, overlap: int = 128) -> List[Tuple[str, int, int]]
```

Split text into overlapping chunks.

#### find_code_boundaries

```python
find_code_boundaries(text: str) -> List[int]
```

Find semantic boundaries in code (class/function definitions, decorators).

#### create_code_aware_chunks

```python
create_code_aware_chunks(text: str, target_size: int = 512, min_size: int = 100, max_size: int = 1024) -> List[Tuple[str, int, int]]
```

Create chunks aligned to code structure boundaries.

#### is_code_file

```python
is_code_file(doc_id: str) -> bool
```

Determine if a document is a code file based on its path/extension.

#### precompute_term_cols

```python
precompute_term_cols(query_terms: Dict[str, float], layer0: HierarchicalLayer) -> Dict[str, 'Minicolumn']
```

Pre-compute minicolumn lookups for query terms.

#### score_chunk_fast

```python
score_chunk_fast(chunk_tokens: List[str], query_terms: Dict[str, float], term_cols: Dict[str, 'Minicolumn'], doc_id: Optional[str] = None) -> float
```

Fast chunk scoring using pre-computed minicolumn lookups.

#### score_chunk

```python
score_chunk(chunk_text: str, query_terms: Dict[str, float], layer0: HierarchicalLayer, tokenizer: Tokenizer, doc_id: Optional[str] = None) -> float
```

Score a chunk against query terms using TF-IDF.

### Dependencies

**Standard Library:**

- `layers.HierarchicalLayer`
- `re`
- `tokenizer.Tokenizer`
- `typing.Dict`
- `typing.List`
- ... and 3 more



## definitions.py

Definition Search Module
========================

Functions for finding and boosting code definitions (classes, functions, methods).

This module handles:
- Detection of definition-seeking queries ("...


### Classes

#### DefinitionQuery

Info about a definition-seeking query.

### Functions

#### is_definition_query

```python
is_definition_query(query_text: str) -> Tuple[bool, Optional[str], Optional[str]]
```

Detect if a query is looking for a code definition.

#### find_definition_in_text

```python
find_definition_in_text(text: str, identifier: str, def_type: str, context_chars: int = 500) -> Optional[Tuple[str, int, int]]
```

Find a definition in source text and extract surrounding context.

#### find_definition_passages

```python
find_definition_passages(query_text: str, documents: Dict[str, str], context_chars: int = 500, boost: float = DEFINITION_BOOST) -> List[Tuple[str, str, int, int, float]]
```

Find definition passages for a definition query.

#### detect_definition_query

```python
detect_definition_query(query_text: str) -> DefinitionQuery
```

Detect if a query is searching for a code definition.

#### apply_definition_boost

```python
apply_definition_boost(passages: List[Tuple[str, str, int, int, float]], query_text: str, boost_factor: float = 3.0) -> List[Tuple[str, str, int, int, float]]
```

Boost passages that contain actual code definitions matching the query.

#### is_test_file

```python
is_test_file(doc_id: str) -> bool
```

Detect if a document ID represents a test file.

#### boost_definition_documents

```python
boost_definition_documents(doc_results: List[Tuple[str, float]], query_text: str, documents: Dict[str, str], boost_factor: float = 2.0, test_with_definition_penalty: float = 0.5, test_without_definition_penalty: float = 0.7) -> List[Tuple[str, float]]
```

Boost documents that contain the actual definition being searched for.

### Dependencies

**Standard Library:**

- `re`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 2 more



## expansion.py

Query Expansion Module
=====================

Functions for expanding query terms using lateral connections,
semantic relations, and code concept synonyms.

This module provides:
- Basic query expansi...


### Functions

#### score_relation_path

```python
score_relation_path(path: List[str]) -> float
```

Score a relation path by its semantic coherence.

#### expand_query

```python
expand_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, max_expansions: int = 10, use_lateral: bool = True, use_concepts: bool = True, use_variants: bool = True, use_code_concepts: bool = False, filter_code_stop_words: bool = False, tfidf_weight: float = 0.7, max_expansion_weight: float = 2.0) -> Dict[str, float]
```

Expand a query using lateral connections and concept clusters.

#### expand_query_semantic

```python
expand_query_semantic(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, semantic_relations: List[Tuple[str, str, str, float]], max_expansions: int = 10) -> Dict[str, float]
```

Expand query using semantic relations extracted from corpus.

#### expand_query_multihop

```python
expand_query_multihop(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, semantic_relations: List[Tuple[str, str, str, float]], max_hops: int = 2, max_expansions: int = 15, decay_factor: float = 0.5, min_path_score: float = 0.2) -> Dict[str, float]
```

Expand query using multi-hop semantic inference.

#### get_expanded_query_terms

```python
get_expanded_query_terms(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, max_expansions: int = 5, semantic_discount: float = 0.8, filter_code_stop_words: bool = False) -> Dict[str, float]
```

Get expanded query terms with optional semantic expansion.

### Dependencies

**Standard Library:**

- `code_concepts.expand_code_concepts`
- `collections.defaultdict`
- `config.DEFAULT_CHAIN_VALIDITY`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 6 more



## intent.py

Intent Query Module
==================

Intent-based query understanding for natural language code search.

This module handles:
- Parsing natural language queries to extract intent (where, how, what,...


### Classes

#### ParsedIntent

Structured representation of a parsed query intent.

### Functions

#### parse_intent_query

```python
parse_intent_query(query_text: str) -> ParsedIntent
```

Parse a natural language query to extract intent and searchable terms.

#### search_by_intent

```python
search_by_intent(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: 'Tokenizer', top_n: int = 5) -> List[Tuple[str, float, ParsedIntent]]
```

Search the corpus using intent-based query understanding.

### Dependencies

**Standard Library:**

- `code_concepts.get_related_terms`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`
- ... and 4 more



## passages.py

Passage Retrieval Module
========================

Functions for retrieving relevant passages from documents.

This module provides:
- Passage retrieval for RAG systems
- Batch passage retrieval
- Int...


### Functions

#### find_passages_for_query

```python
find_passages_for_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, use_expansion: bool = True, doc_filter: Optional[List[str]] = None, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, use_definition_search: bool = True, definition_boost: float = DEFINITION_BOOST, apply_doc_boost: bool = True, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, auto_detect_intent: bool = True, prefer_docs: bool = False, custom_boosts: Optional[Dict[str, float]] = None, use_code_aware_chunks: bool = True, filter_code_stop_words: bool = True, test_file_penalty: float = 0.8) -> List[Tuple[str, str, int, int, float]]
```

Find text passages most relevant to a query.

#### find_documents_batch

```python
find_documents_batch(queries: List[str], layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[List[Tuple[str, float]]]
```

Find documents for multiple queries efficiently.

#### find_passages_batch

```python
find_passages_batch(queries: List[str], layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, use_expansion: bool = True, doc_filter: Optional[List[str]] = None, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[List[Tuple[str, str, int, int, float]]]
```

Find passages for multiple queries efficiently.

### Dependencies

**Standard Library:**

- `chunking.CODE_BOUNDARY_PATTERN`
- `chunking.create_chunks`
- `chunking.create_code_aware_chunks`
- `chunking.find_code_boundaries`
- `chunking.is_code_file`
- ... and 18 more



## ranking.py

Ranking Module
=============

Multi-stage ranking and document type boosting for search results.

This module provides:
- Document type boosting (docs, code, tests)
- Conceptual vs implementation quer...


### Functions

#### is_conceptual_query

```python
is_conceptual_query(query_text: str) -> bool
```

Determine if a query is conceptual (should boost documentation).

#### get_doc_type_boost

```python
get_doc_type_boost(doc_id: str, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, custom_boosts: Optional[Dict[str, float]] = None) -> float
```

Get the boost factor for a document based on its type.

#### apply_doc_type_boost

```python
apply_doc_type_boost(results: List[Tuple[str, float]], doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, boost_docs: bool = True, custom_boosts: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]
```

Apply document type boosting to search results.

#### find_documents_with_boost

```python
find_documents_with_boost(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, auto_detect_intent: bool = True, prefer_docs: bool = False, custom_boosts: Optional[Dict[str, float]] = None, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, float]]
```

Find documents with optional document-type boosting.

#### find_relevant_concepts

```python
find_relevant_concepts(query_terms: Dict[str, float], layers: Dict[CorticalLayer, HierarchicalLayer], top_n: int = 5) -> List[Tuple[str, float, set]]
```

Stage 1: Find concepts relevant to query terms.

#### multi_stage_rank

```python
multi_stage_rank(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, concept_boost: float = 0.3, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, str, int, int, float, Dict[str, float]]]
```

Multi-stage ranking pipeline for improved RAG performance.

#### multi_stage_rank_documents

```python
multi_stage_rank_documents(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, concept_boost: float = 0.3, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, float, Dict[str, float]]]
```

Multi-stage ranking for documents (without chunk scoring).

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.CONCEPTUAL_KEYWORDS`
- `constants.DOC_TYPE_BOOSTS`
- `constants.IMPLEMENTATION_KEYWORDS`
- `expansion.get_expanded_query_terms`
- ... and 9 more



## search.py

Document Search Module
=====================

Functions for searching and retrieving documents from the corpus.

This module provides:
- Basic document search using TF-IDF scoring
- Fast document sear...


### Functions

#### find_documents_for_query

```python
find_documents_for_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, doc_name_boost: float = 2.0, filter_code_stop_words: bool = True, test_file_penalty: float = 0.8) -> List[Tuple[str, float]]
```

Find documents most relevant to a query using TF-IDF and optional expansion.

#### fast_find_documents

```python
fast_find_documents(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, candidate_multiplier: int = 3, use_code_concepts: bool = True, doc_name_boost: float = 2.0) -> List[Tuple[str, float]]
```

Fast document search using candidate filtering.

#### build_document_index

```python
build_document_index(layers: Dict[CorticalLayer, HierarchicalLayer]) -> Dict[str, Dict[str, float]]
```

Build an optimized inverted index for fast querying.

#### search_with_index

```python
search_with_index(query_text: str, index: Dict[str, Dict[str, float]], tokenizer: Tokenizer, top_n: int = 5) -> List[Tuple[str, float]]
```

Search using a pre-built inverted index.

#### query_with_spreading_activation

```python
query_with_spreading_activation(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 10, max_expansions: int = 8) -> List[Tuple[str, float]]
```

Query with automatic expansion using spreading activation.

#### find_related_documents

```python
find_related_documents(doc_id: str, layers: Dict[CorticalLayer, HierarchicalLayer]) -> List[Tuple[str, float]]
```

Find documents related to a given document via lateral connections.

#### graph_boosted_search

```python
graph_boosted_search(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, pagerank_weight: float = 0.3, proximity_weight: float = 0.2, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None) -> List[Tuple[str, float]]
```

Graph-Boosted BM25 (GB-BM25): Hybrid scoring combining BM25 with graph signals.

### Dependencies

**Standard Library:**

- `code_concepts.get_related_terms`
- `collections.defaultdict`
- `expansion.expand_query`
- `expansion.get_expanded_query_terms`
- `layers.CorticalLayer`
- ... and 6 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Utilities

Utility modules supporting various features.

## Modules

- **cli_wrapper.py**: CLI wrapper framework for collecting context and triggering actions.
- **code_concepts.py**: Code Concepts Module
- **diff.py**: Semantic Diff Module
- **fingerprint.py**: Fingerprint Module
- **fluent.py**: Fluent API for CorticalTextProcessor - chainable method interface.
- **gaps.py**: Gaps Module
- **mcp_server.py**: MCP (Model Context Protocol) Server for Cortical Text Processor.
- **patterns.py**: Code Pattern Detection Module


## cli_wrapper.py

CLI wrapper framework for collecting context and triggering actions.

Design philosophy: QUIET BY DEFAULT, POWERFUL WHEN NEEDED.

Most of the time you just want to run a command and check if it worked...


### Classes

#### GitContext

Git repository context information.

**Methods:**

- `collect`
- `to_dict`

#### ExecutionContext

Complete context for a CLI command execution.

**Methods:**

- `to_dict`
- `to_json`
- `summary`

#### HookType

Types of hooks that can be registered.

#### HookRegistry

Registry for CLI execution hooks.

**Methods:**

- `register`
- `register_pre`
- `register_post`
- `register_success`
- `register_error`
- `get_hooks`
- `trigger`

#### CLIWrapper

Wrapper for CLI command execution with context collection and hooks.

**Methods:**

- `run`
- `on_success`
- `on_error`
- `on_complete`

#### TaskCompletionManager

Manager for task completion triggers and context window management.

**Methods:**

- `on_task_complete`
- `on_any_complete`
- `handle_completion`
- `get_session_summary`
- `should_trigger_reindex`

#### ContextWindowManager

Manages context window state based on CLI execution history.

**Methods:**

- `add_execution`
- `add_file_read`
- `get_recent_files`
- `get_context_summary`
- `suggest_pruning`

#### Session

Track a sequence of commands as a session.

**Methods:**

- `run`
- `should_reindex`
- `summary`
- `results`
- `success_rate`
- `all_passed`
- `modified_files`

#### TaskCheckpoint

Save/restore context state when switching between tasks.

**Methods:**

- `save`
- `load`
- `list_tasks`
- `delete`
- `summarize`

### Functions

#### create_wrapper_with_completion_manager

```python
create_wrapper_with_completion_manager() -> Tuple[CLIWrapper, TaskCompletionManager]
```

Create a CLIWrapper with an attached TaskCompletionManager.

#### run_with_context

```python
run_with_context(command: Union[str, List[str]], **kwargs) -> ExecutionContext
```

Convenience function to run a command with full context collection.

#### run

```python
run(command: Union[str, List[str]], git: bool = False, timeout: Optional[float] = None, cwd: Optional[str] = None) -> ExecutionContext
```

Run a command. That's it.

#### test_then_commit

```python
test_then_commit(test_cmd: Union[str, List[str]] = 'python -m unittest discover -s tests', message: str = 'Update', add_all: bool = True) -> Tuple[bool, List[ExecutionContext]]
```

Run tests, commit only if they pass.

#### commit_and_push

```python
commit_and_push(message: str, add_all: bool = True, branch: Optional[str] = None) -> Tuple[bool, List[ExecutionContext]]
```

Add, commit, and push in one go.

#### sync_with_main

```python
sync_with_main(main_branch: str = 'main') -> Tuple[bool, List[ExecutionContext]]
```

Fetch and rebase current branch on main.

#### GitContext.collect

```python
GitContext.collect(cls, cwd: Optional[str] = None) -> 'GitContext'
```

Collect git context from current directory.

#### GitContext.to_dict

```python
GitContext.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### ExecutionContext.to_dict

```python
ExecutionContext.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for serialization.

#### ExecutionContext.to_json

```python
ExecutionContext.to_json(self, indent: int = 2) -> str
```

Convert to JSON string.

#### ExecutionContext.summary

```python
ExecutionContext.summary(self) -> str
```

Return a concise summary string.

#### HookRegistry.register

```python
HookRegistry.register(self, hook_type: HookType, callback: HookCallback, pattern: Optional[str] = None) -> None
```

Register a hook callback.

#### HookRegistry.register_pre

```python
HookRegistry.register_pre(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for pre-execution hooks.

#### HookRegistry.register_post

```python
HookRegistry.register_post(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for post-execution hooks.

#### HookRegistry.register_success

```python
HookRegistry.register_success(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for success hooks.

#### HookRegistry.register_error

```python
HookRegistry.register_error(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for error hooks.

#### HookRegistry.get_hooks

```python
HookRegistry.get_hooks(self, hook_type: HookType, command: List[str]) -> List[HookCallback]
```

Get all hooks that should be triggered for a command.

#### HookRegistry.trigger

```python
HookRegistry.trigger(self, hook_type: HookType, context: ExecutionContext) -> None
```

Trigger all matching hooks.

#### CLIWrapper.run

```python
CLIWrapper.run(self, command: Union[str, List[str]], cwd: Optional[str] = None, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None, **kwargs) -> ExecutionContext
```

Execute a command with context collection and hooks.

#### CLIWrapper.on_success

```python
CLIWrapper.on_success(self, pattern: Optional[str] = None)
```

Decorator to register a success hook.

#### CLIWrapper.on_error

```python
CLIWrapper.on_error(self, pattern: Optional[str] = None)
```

Decorator to register an error hook.

#### CLIWrapper.on_complete

```python
CLIWrapper.on_complete(self, pattern: Optional[str] = None)
```

Decorator to register a completion hook (success or failure).

#### TaskCompletionManager.on_task_complete

```python
TaskCompletionManager.on_task_complete(self, task_type: str, callback: HookCallback) -> None
```

Register a callback for when a specific task type completes.

#### TaskCompletionManager.on_any_complete

```python
TaskCompletionManager.on_any_complete(self, callback: HookCallback) -> None
```

Register a callback for any task completion.

#### TaskCompletionManager.handle_completion

```python
TaskCompletionManager.handle_completion(self, context: ExecutionContext) -> None
```

Handle task completion and trigger appropriate callbacks.

#### TaskCompletionManager.get_session_summary

```python
TaskCompletionManager.get_session_summary(self) -> Dict[str, Any]
```

Get summary of all tasks completed in this session.

#### TaskCompletionManager.should_trigger_reindex

```python
TaskCompletionManager.should_trigger_reindex(self) -> bool
```

Determine if corpus should be re-indexed based on session activity.

#### ContextWindowManager.add_execution

```python
ContextWindowManager.add_execution(self, context: ExecutionContext) -> None
```

Add an execution to the context window.

#### ContextWindowManager.add_file_read

```python
ContextWindowManager.add_file_read(self, filepath: str) -> None
```

Track that a file was read.

#### ContextWindowManager.get_recent_files

```python
ContextWindowManager.get_recent_files(self, limit: int = 10) -> List[str]
```

Get most recently accessed files.

#### ContextWindowManager.get_context_summary

```python
ContextWindowManager.get_context_summary(self) -> Dict[str, Any]
```

Get a summary of current context window state.

#### ContextWindowManager.suggest_pruning

```python
ContextWindowManager.suggest_pruning(self) -> List[str]
```

Suggest files that could be pruned from context.

#### Session.run

```python
Session.run(self, command: Union[str, List[str]], **kwargs) -> ExecutionContext
```

Run a command within this session.

#### Session.should_reindex

```python
Session.should_reindex(self) -> bool
```

Check if corpus re-indexing is recommended based on session activity.

#### Session.summary

```python
Session.summary(self) -> Dict[str, Any]
```

Get a summary of this session's activity.

#### Session.results

```python
Session.results(self) -> List[ExecutionContext]
```

All command results from this session.

#### Session.success_rate

```python
Session.success_rate(self) -> float
```

Fraction of commands that succeeded (0.0 to 1.0).

#### Session.all_passed

```python
Session.all_passed(self) -> bool
```

True if all commands in this session succeeded.

#### Session.modified_files

```python
Session.modified_files(self) -> List[str]
```

List of files modified during this session (from git context).

#### TaskCheckpoint.save

```python
TaskCheckpoint.save(self, task_name: str, context: Dict[str, Any]) -> None
```

Save context for a task.

#### TaskCheckpoint.load

```python
TaskCheckpoint.load(self, task_name: str) -> Optional[Dict[str, Any]]
```

Load context for a task. Returns None if not found.

#### TaskCheckpoint.list_tasks

```python
TaskCheckpoint.list_tasks(self) -> List[str]
```

List all saved task checkpoints.

#### TaskCheckpoint.delete

```python
TaskCheckpoint.delete(self, task_name: str) -> bool
```

Delete a checkpoint. Returns True if deleted.

#### TaskCheckpoint.summarize

```python
TaskCheckpoint.summarize(self, task_name: str) -> Optional[str]
```

Get a one-line summary of a task checkpoint.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `enum.Enum`
- ... and 15 more



## code_concepts.py

Code Concepts Module
====================

Programming concept groups for semantic code search.

Maps common programming synonyms and related terms to enable
intent-based code retrieval. When a develo...


### Functions

#### get_related_terms

```python
get_related_terms(term: str, max_terms: int = 5) -> List[str]
```

Get programming terms related to the given term.

#### expand_code_concepts

```python
expand_code_concepts(terms: List[str], max_expansions_per_term: int = 3, weight: float = 0.6) -> Dict[str, float]
```

Expand a list of terms using code concept groups.

#### get_concept_group

```python
get_concept_group(term: str) -> List[str]
```

Get the concept group names a term belongs to.

#### list_concept_groups

```python
list_concept_groups() -> List[str]
```

List all available concept group names.

#### get_group_terms

```python
get_group_terms(group_name: str) -> List[str]
```

Get all terms in a concept group.

### Dependencies

**Standard Library:**

- `typing.Dict`
- `typing.FrozenSet`
- `typing.List`
- `typing.Set`



## diff.py

Semantic Diff Module
====================

Provides "What Changed?" functionality for comparing:
- Two versions of a document
- Two processor states
- Before/after states of a corpus

This goes beyond...


### Classes

#### TermChange

Represents a change to a term/concept.

**Methods:**

- `pagerank_delta`
- `tfidf_delta`
- `documents_added`
- `documents_removed`

#### RelationChange

Represents a change to a semantic relation.

#### ClusterChange

Represents a change to concept clustering.

#### SemanticDiff

Complete semantic diff between two states.

**Methods:**

- `summary`
- `to_dict`

### Functions

#### compare_processors

```python
compare_processors(old_processor: 'CorticalTextProcessor', new_processor: 'CorticalTextProcessor', top_movers: int = 20, min_pagerank_delta: float = 0.0001) -> SemanticDiff
```

Compare two processor states to find semantic differences.

#### compare_documents

```python
compare_documents(processor: 'CorticalTextProcessor', doc_id_old: str, doc_id_new: str) -> Dict[str, Any]
```

Compare two documents within the same corpus.

#### what_changed

```python
what_changed(processor: 'CorticalTextProcessor', old_content: str, new_content: str, temp_doc_prefix: str = '_diff_temp_') -> Dict[str, Any]
```

Compare two text contents to show what changed semantically.

#### TermChange.pagerank_delta

```python
TermChange.pagerank_delta(self) -> Optional[float]
```

Change in PageRank importance.

#### TermChange.tfidf_delta

```python
TermChange.tfidf_delta(self) -> Optional[float]
```

Change in TF-IDF score.

#### TermChange.documents_added

```python
TermChange.documents_added(self) -> Set[str]
```

Documents where this term newly appears.

#### TermChange.documents_removed

```python
TermChange.documents_removed(self) -> Set[str]
```

Documents where this term no longer appears.

#### SemanticDiff.summary

```python
SemanticDiff.summary(self) -> str
```

Generate a human-readable summary of changes.

#### SemanticDiff.to_dict

```python
SemanticDiff.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for serialization.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 7 more



## fingerprint.py

Fingerprint Module
==================

Semantic fingerprinting for code comparison and similarity analysis.

A fingerprint is an interpretable representation of a text's semantic
content, including te...


### Classes

#### SemanticFingerprint

Structured representation of a text's semantic fingerprint.

### Functions

#### compute_fingerprint

```python
compute_fingerprint(text: str, tokenizer: Tokenizer, layers: Optional[Dict[CorticalLayer, HierarchicalLayer]] = None, top_n: int = 20) -> SemanticFingerprint
```

Compute the semantic fingerprint of a text.

#### compare_fingerprints

```python
compare_fingerprints(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> Dict[str, Any]
```

Compare two fingerprints and compute similarity metrics.

#### explain_fingerprint

```python
explain_fingerprint(fp: SemanticFingerprint, top_n: int = 10) -> Dict[str, Any]
```

Generate a human-readable explanation of a fingerprint.

#### explain_similarity

```python
explain_similarity(fp1: SemanticFingerprint, fp2: SemanticFingerprint, comparison: Optional[Dict[str, Any]] = None) -> str
```

Generate a human-readable explanation of why two fingerprints are similar.

### Dependencies

**Standard Library:**

- `code_concepts.get_concept_group`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- ... and 7 more



## fluent.py

Fluent API for CorticalTextProcessor - chainable method interface.

Example:
    from cortical import FluentProcessor

    # Simple usage
    results = (FluentProcessor()
        .add_document("doc1",...


### Classes

#### FluentProcessor

Fluent/chainable API wrapper for CorticalTextProcessor.

**Methods:**

- `from_existing`
- `from_files`
- `from_directory`
- `load`
- `add_document`
- `add_documents`
- `with_config`
- `with_tokenizer`
- `build`
- `save`
- `search`
- `fast_search`
- `search_passages`
- `expand`
- `processor`
- `is_built`

### Functions

#### FluentProcessor.from_existing

```python
FluentProcessor.from_existing(cls, processor: CorticalTextProcessor) -> 'FluentProcessor'
```

Create a FluentProcessor from an existing CorticalTextProcessor.

#### FluentProcessor.from_files

```python
FluentProcessor.from_files(cls, file_paths: List[Union[str, Path]], tokenizer: Optional[Tokenizer] = None, config: Optional[CorticalConfig] = None) -> 'FluentProcessor'
```

Create a processor from a list of files.

#### FluentProcessor.from_directory

```python
FluentProcessor.from_directory(cls, directory: Union[str, Path], pattern: str = '*.txt', recursive: bool = False, tokenizer: Optional[Tokenizer] = None, config: Optional[CorticalConfig] = None) -> 'FluentProcessor'
```

Create a processor from all files in a directory.

#### FluentProcessor.load

```python
FluentProcessor.load(cls, path: Union[str, Path]) -> 'FluentProcessor'
```

Load a processor from a saved file.

#### FluentProcessor.add_document

```python
FluentProcessor.add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> 'FluentProcessor'
```

Add a document to the processor (chainable).

#### FluentProcessor.add_documents

```python
FluentProcessor.add_documents(self, documents: Union[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str, Dict]]]) -> 'FluentProcessor'
```

Add multiple documents at once (chainable).

#### FluentProcessor.with_config

```python
FluentProcessor.with_config(self, config: CorticalConfig) -> 'FluentProcessor'
```

Set configuration (chainable).

#### FluentProcessor.with_tokenizer

```python
FluentProcessor.with_tokenizer(self, tokenizer: Tokenizer) -> 'FluentProcessor'
```

Set custom tokenizer (chainable).

#### FluentProcessor.build

```python
FluentProcessor.build(self, verbose: bool = True, build_concepts: bool = True, pagerank_method: str = 'standard', connection_strategy: str = 'document_overlap', cluster_strictness: float = 1.0, bridge_weight: float = 0.0, show_progress: bool = False) -> 'FluentProcessor'
```

Build the processor by computing all analysis phases (chainable).

#### FluentProcessor.save

```python
FluentProcessor.save(self, path: Union[str, Path]) -> 'FluentProcessor'
```

Save the processor to disk (chainable).

#### FluentProcessor.search

```python
FluentProcessor.search(self, query: str, top_n: int = 5, use_expansion: bool = True, use_semantic: bool = True) -> List[Tuple[str, float]]
```

Search for documents matching the query.

#### FluentProcessor.fast_search

```python
FluentProcessor.fast_search(self, query: str, top_n: int = 5, candidate_multiplier: int = 3, use_code_concepts: bool = True) -> List[Tuple[str, float]]
```

Fast document search with pre-filtering.

#### FluentProcessor.search_passages

```python
FluentProcessor.search_passages(self, query: str, top_n: int = 5, chunk_size: Optional[int] = None, overlap: Optional[int] = None, use_expansion: bool = True) -> List[Tuple[str, str, int, int, float]]
```

Search for passage chunks matching the query.

#### FluentProcessor.expand

```python
FluentProcessor.expand(self, query: str, max_expansions: Optional[int] = None, use_variants: bool = True, use_code_concepts: bool = False) -> Dict[str, float]
```

Expand a query with related terms.

#### FluentProcessor.processor

```python
FluentProcessor.processor(self) -> CorticalTextProcessor
```

Access the underlying CorticalTextProcessor instance.

#### FluentProcessor.is_built

```python
FluentProcessor.is_built(self) -> bool
```

Check if the processor has been built.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `os`
- `pathlib.Path`
- `processor.CorticalTextProcessor`
- `tokenizer.Tokenizer`
- ... and 6 more



## gaps.py

Gaps Module
===========

Knowledge gap detection and anomaly analysis.

Identifies:
- Isolated documents that don't connect well to the corpus
- Weakly covered topics (few documents)
- Bridge opportun...


### Functions

#### analyze_knowledge_gaps

```python
analyze_knowledge_gaps(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> Dict
```

Analyze the corpus to identify potential knowledge gaps.

#### detect_anomalies

```python
detect_anomalies(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], threshold: float = 0.3) -> List[Dict]
```

Detect documents that don't fit well with the rest of the corpus.

### Dependencies

**Standard Library:**

- `analysis.cosine_similarity`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- ... and 5 more



## mcp_server.py

MCP (Model Context Protocol) Server for Cortical Text Processor.

Provides an MCP server interface for AI agents to integrate with the
Cortical Text Processor, enabling semantic search, query expansio...


### Classes

#### CorticalMCPServer

MCP Server wrapper for CorticalTextProcessor.

**Methods:**

- `run`

### Functions

#### create_mcp_server

```python
create_mcp_server(corpus_path: Optional[str] = None, config: Optional[CorticalConfig] = None) -> CorticalMCPServer
```

Create a Cortical MCP Server instance.

#### main

```python
main()
```

Main entry point for running the MCP server from command line.

#### CorticalMCPServer.run

```python
CorticalMCPServer.run(self, transport: str = 'stdio')
```

Run the MCP server.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `logging`
- `mcp.server.FastMCP`
- `os`
- `pathlib.Path`
- ... and 5 more



## patterns.py

Code Pattern Detection Module
==============================

Detects common programming patterns in indexed code.

Identifies design patterns, idioms, and code structures including:
- Singleton patte...


### Functions

#### detect_patterns_in_text

```python
detect_patterns_in_text(text: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]
```

Detect programming patterns in a text string.

#### detect_patterns_in_documents

```python
detect_patterns_in_documents(documents: Dict[str, str], patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]
```

Detect patterns across multiple documents.

#### get_pattern_summary

```python
get_pattern_summary(pattern_results: Dict[str, List[int]]) -> Dict[str, int]
```

Summarize pattern detection results by counting occurrences.

#### get_patterns_by_category

```python
get_patterns_by_category(pattern_results: Dict[str, List[int]]) -> Dict[str, Dict[str, int]]
```

Group pattern results by category.

#### get_pattern_description

```python
get_pattern_description(pattern_name: str) -> Optional[str]
```

Get the description for a pattern.

#### get_pattern_category

```python
get_pattern_category(pattern_name: str) -> Optional[str]
```

Get the category for a pattern.

#### list_all_patterns

```python
list_all_patterns() -> List[str]
```

List all available pattern names.

#### list_patterns_by_category

```python
list_patterns_by_category(category: str) -> List[str]
```

List all patterns in a specific category.

#### list_all_categories

```python
list_all_categories() -> List[str]
```

List all pattern categories.

#### format_pattern_report

```python
format_pattern_report(pattern_results: Dict[str, List[int]], show_lines: bool = False) -> str
```

Format pattern detection results as a human-readable report.

#### get_corpus_pattern_statistics

```python
get_corpus_pattern_statistics(doc_patterns: Dict[str, Dict[str, List[int]]]) -> Dict[str, any]
```

Compute statistics across all documents.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `re`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 2 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

# Decisions: ADRs

## Decisions

*This chapter will be auto-generated in a future update.*

---

# Evolution: Project History

## Bug Fixes and Lessons

*What broke, how we fixed it, and what we learned.*

---

## Overview

**14 bugs** have been identified and resolved. Each fix taught us something about the system.

## Bug Fix History

### Archive ML session after transcript processing (T-003 16f3)

**Commit:** `59072c8`  
**Date:** 2025-12-16  
**Files Changed:** scripts/ml_data_collector.py  

### Update CSV truncation test for new defaults (input=500, output=2000)

**Commit:** `ca94a01`  
**Date:** 2025-12-16  

### Fix ML data collection milestone counting and add session/action capture

**Commit:** `273baef`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-15/chat-20251216-121720-30c3c1.json, .git-ml/chats/2025-12-16/chat-20251216-121720-01077d.json, .git-ml/chats/2025-12-16/chat-20251216-121720-306450.json, .git-ml/chats/2025-12-16/chat-20251216-121720-5ef95b.json, .git-ml/chats/2025-12-16/chat-20251216-121720-8a1e7b.json  
*(and 6 more)*  

### Address critical ML data collection and prediction issues

**Commit:** `fead1c1`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json, .git-ml/chats/2025-12-16/chat-20251216-115057-3617f9.json, .git-ml/chats/2025-12-16/chat-20251216-115057-9502fd.json, .git-ml/chats/2025-12-16/chat-20251216-115057-cbbe64.json, .git-ml/chats/2025-12-16/chat-20251216-115057-f65b7a.json  
*(and 4 more)*  

### Add missing imports in validate command

**Commit:** `172ad8f`  
**Date:** 2025-12-16  
**Files Changed:** scripts/ml_data_collector.py  

### Clean up gitignore pattern for .git-ml/commits/

**Commit:** `a65d54f`  
**Date:** 2025-12-16  
**Files Changed:** .gitignore  

### Prevent infinite commit loop in ML data collection hooks

**Commit:** `66ad656`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-16/chat-20251216-004054-78b531.json, .git-ml/tracked/commits.jsonl, scripts/ml_data_collector.py  

### Correct hook format in settings.local.json

**Commit:** `19ac02a`  
**Date:** 2025-12-16  
**Files Changed:** .claude/settings.local.json  

### Use filename-based sorting for deterministic session ordering

**Commit:** `61d502d`  
**Date:** 2025-12-15  

### Increase ID suffix length to prevent collisions

**Commit:** `8ac4b6b`  
**Date:** 2025-12-15  

### Add import guards for optional test dependencies

**Commit:** `91ffb04`  
**Date:** 2025-12-15  

### Make session file sorting stable for deterministic ordering

**Commit:** `7433b36`  
**Date:** 2025-12-15  

### Cap query expansion weights to prevent term domination

**Commit:** `fecd6dc`  
**Date:** 2025-12-15  

### Add YAML frontmatter to slash commands for discovery

**Commit:** `5b52da2`  
**Date:** 2025-12-15

---

## Feature Evolution

*How the Cortical Text Processor gained its capabilities.*

---

## Overview

The system has evolved through **22 feature additions**. Below is the narrative of how each capability came to be.

## Documentation Capabilities

### Add CI workflow and documentation (Wave 4)

**Commit:** `940fdf2`  
**Date:** 2025-12-16  
**Files Modified:** 5  

### Add animated GIF visualizations to README

**Commit:** `b4d7c82`  
**Date:** 2025-12-15  

## Search Capabilities

### Add search integration and web interface (Wave 3)

**Commit:** `0022466`  
**Date:** 2025-12-16  
**Files Modified:** 11  

### Add security concept group and TF-IDF weighted query expansion

**Commit:** `af3a7e0`  
**Date:** 2025-12-15  

### Add comprehensive test coverage for query and analysis modules

**Commit:** `70a4b1b`  
**Date:** 2025-12-15  

## Other Capabilities

### Add content generators for Cortical Chronicles (Wave 2)

**Commit:** `3022110`  
**Date:** 2025-12-16  
**Files Modified:** 23  

### Add Cortical Chronicles book infrastructure (Wave 1)

**Commit:** `c730057`  
**Date:** 2025-12-16  
**Files Modified:** 13  

### Batch task distribution implementation via Director orchestration

**Commit:** `4f915c3`  
**Date:** 2025-12-16  
**Files Modified:** 8  

### Add orchestration extraction for director sub-agent tracking

**Commit:** `4eaeb37`  
**Date:** 2025-12-15  

### Add stunning animated ASCII codebase visualizer

**Commit:** `e085a0b`  
**Date:** 2025-12-15  

### Add ASCII art codebase visualization script

**Commit:** `43aae33`  
**Date:** 2025-12-15  

### Complete legacy task system migration

**Commit:** `33dc8b2`  
**Date:** 2025-12-15  

### Add director orchestration execution tracking system

**Commit:** `4976c58`  
**Date:** 2025-12-15  

## Ml Capabilities

### Add file existence filter to ML predictions

**Commit:** `3cab2ba`  
**Date:** 2025-12-16  
**Files Modified:** 1  

### Add ML file prediction model

**Commit:** `ac549dd`  
**Date:** 2025-12-16  
**Files Modified:** 2  

### Add chunked storage for git-friendly ML data

**Commit:** `0754540`  
**Date:** 2025-12-16  
**Files Modified:** 4  

### Add ML stats report to CI pipeline

**Commit:** `3e05a70`  
**Date:** 2025-12-16  
**Files Modified:** 9  

## Data Capabilities

### Add git-tracked JSONL storage for orchestration data

**Commit:** `fb30e38`  
**Date:** 2025-12-15  

### Add lightweight commit data for ephemeral environments

**Commit:** `89d6aa5`  
**Date:** 2025-12-15  
**Files Modified:** 475  

## Testing Capabilities

### Add comprehensive test coverage for Wave 4 modules (FINAL)

**Commit:** `73d6da8`  
**Date:** 2025-12-15  

### Add comprehensive test coverage for Wave 3 modules

**Commit:** `036f830`  
**Date:** 2025-12-15  

### Add comprehensive test coverage for Wave 2 modules

**Commit:** `5a6bb26`  
**Date:** 2025-12-15

---

## Refactorings and Architecture Evolution

*How the codebase structure improved over time.*

---

## Overview

The codebase has undergone **3 refactorings**. Each improved code quality, maintainability, or performance.

## Refactoring History

### Remove unused protobuf serialization (T-013 f0ff)

**Commit:** `d7a98ae`  
**Date:** 2025-12-16  
**Changes:** +100/-1460 lines  
**Scope:** 6 files affected  

### Split large files exceeding 25000 token limit

**Commit:** `21ec5ea`  
**Date:** 2025-12-15  

### Consolidate ML data to single JSONL files

**Commit:** `205fe34`  
**Date:** 2025-12-15  
**Changes:** +658/-12208 lines  
**Scope:** 486 files affected

---

## Timeline

---

## December 2025

### Week of Dec 15

- **2025-12-16**: feat: Add book
- **2025-12-16**: docs: Add vision

---

## Project Timeline

*A chronological journey through the Cortical Text Processor's development.*

---

## December 2025

### Week of Dec 15

- **2025-12-16**: feat: Add CI workflow and documentation (Wave 4)
- **2025-12-16**: feat: Add search integration and web interface (Wave 3)
- **2025-12-16**: feat: Add content generators for Cortical Chronicles (Wave 2)
- **2025-12-16**: feat: Add Cortical Chronicles book infrastructure (Wave 1)
- **2025-12-16**: docs: Add deep algorithm analysis and author reflections to VISION.md
- **2025-12-16**: docs: Add product vision document for legacy feature roadmap
- **2025-12-16**: refactor: Remove unused protobuf serialization (T-013 f0ff)
- **2025-12-16**: fix: Archive ML session after transcript processing (T-003 16f3)
- **2025-12-16**: fix: Update CSV truncation test for new defaults (input=500, output=2000)
- **2025-12-16**: feat: Batch task distribution implementation via Director orchestration

---

# Future: Roadmap

## Future

*This chapter will be auto-generated in a future update.*

---

---

## About This Book

**The Cortical Chronicles** is a self-documenting book generated by the Cortical Text Processor.
It documents its own architecture, algorithms, and evolution through automated extraction
of code metadata, git history, and architectural decision records.

### How to Regenerate

```bash
# Generate individual chapters
python scripts/generate_book.py

# Generate consolidated markdown
python scripts/generate_book.py --markdown

# Force regeneration (ignore cache)
python scripts/generate_book.py --markdown --force
```

### Source Code

The source code and generation scripts are available at the project repository.
