# System Architecture

This document describes the 4-layer hierarchical architecture of the Cortical Text Processor. The design is inspired by visual cortex organization, processing text at increasing levels of abstraction.

## Layer Overview

```
Layer 3 (DOCUMENTS)  ← Full documents        [IT analogy: objects]
    ↑↓
Layer 2 (CONCEPTS)   ← Semantic clusters     [V4 analogy: shapes]
    ↑↓
Layer 1 (BIGRAMS)    ← Word pairs            [V2 analogy: patterns]
    ↑↓
Layer 0 (TOKENS)     ← Individual words      [V1 analogy: edges]
```

Information flows both upward (abstraction) and downward (grounding) through feedforward and feedback connections.

---

## Core Data Structures

### CorticalLayer Enum

**Location:** `layers.py:21-56`

```python
class CorticalLayer(Enum):
    TOKENS = 0      # Individual words
    BIGRAMS = 1     # Word pairs
    CONCEPTS = 2    # Semantic clusters
    DOCUMENTS = 3   # Full documents
```

### HierarchicalLayer

**Location:** `layers.py:59-273`

Container for minicolumns at each layer:

```python
class HierarchicalLayer:
    layer_type: CorticalLayer
    minicolumns: Dict[str, Minicolumn]  # content → minicolumn
    _id_index: Dict[str, str]           # id → content (O(1) lookup)
```

**Key Methods:**
- `get_or_create_minicolumn(content)` - Create or retrieve minicolumn
- `get_minicolumn(content)` - Retrieve by content
- `get_by_id(col_id)` - O(1) lookup by ID (critical for performance)
- `column_count()` - Number of minicolumns

### Minicolumn

**Location:** `minicolumn.py:56-357`

The fundamental unit of representation:

```python
class Minicolumn:
    # Identity
    id: str              # "L0_neural", "L1_neural networks"
    content: str         # "neural", "neural networks"
    layer: int           # 0, 1, 2, or 3

    # Statistics
    activation: float           # Neural activation level
    occurrence_count: int       # Total occurrences in corpus
    pagerank: float            # Importance score
    tfidf: float               # Global TF-IDF weight
    tfidf_per_doc: Dict[str, float]  # Per-document TF-IDF

    # Document association
    document_ids: Set[str]     # Which documents contain this
    doc_occurrence_counts: Dict[str, int]  # Occurrences per document

    # Connections (see Connection Types below)
    lateral_connections: Dict[str, float]
    typed_connections: Dict[str, Edge]
    feedforward_connections: Dict[str, float]
    feedback_connections: Dict[str, float]

    # Clustering
    cluster_id: Optional[int]  # For Layer 0 tokens
```

**ID Pattern:** `f"L{layer}_{content}"`
- Token: `"L0_neural"`
- Bigram: `"L1_neural networks"`
- Concept: `"L2_neural/networks/learning"`
- Document: `"L3_doc_001"`

### Edge

**Location:** `minicolumn.py:16-53`

Typed connection with metadata (ConceptNet-style):

```python
@dataclass
class Edge:
    target_id: str                      # "L0_network"
    weight: float = 1.0                 # Connection strength
    relation_type: str = 'co_occurrence'  # 'IsA', 'PartOf', etc.
    confidence: float = 1.0             # [0.0, 1.0]
    source: str = 'corpus'              # 'corpus', 'semantic', 'inferred'
```

---

## Connection Types

### 1. Lateral Connections

**Within-layer** associations from co-occurrence.

```python
minicolumn.lateral_connections: Dict[str, float]
# {"L0_networks": 0.8, "L0_learning": 0.5}
```

- **Layer 0:** Tokens appearing near each other in text
- **Layer 1:** Bigrams sharing components or co-occurring
- **Layer 2:** Concepts with overlapping documents or semantics
- **Layer 3:** Documents sharing vocabulary

### 2. Typed Connections

**Within-layer** with semantic metadata.

```python
minicolumn.typed_connections: Dict[str, Edge]
# {"L0_animal": Edge(weight=0.9, relation_type='IsA', confidence=0.95)}
```

Used for ConceptNet-style reasoning with relation types.

### 3. Feedforward Connections

**Downward** links to components (higher → lower layer).

```python
minicolumn.feedforward_connections: Dict[str, float]
```

- Bigram → component tokens: `"neural networks" → ["neural", "networks"]`
- Concept → member tokens: `"neural/networks/learning" → [member tokens]`
- Document → contained tokens: `"doc1" → [all tokens in doc1]`

### 4. Feedback Connections

**Upward** links to containers (lower → higher layer).

```python
minicolumn.feedback_connections: Dict[str, float]
```

- Token → containing bigrams: `"neural" → ["neural networks", "neural processing"]`
- Token → containing concepts: `"neural" → ["neural/networks/learning"]`
- Token → containing documents: `"neural" → ["doc1", "doc2"]`

---

## Data Flow

### Document Processing

**Location:** `processor.py:54-137`

When a document is processed:

```
INPUT: "Neural networks process data."

1. TOKENIZATION
   → ["neural", "networks", "process", "data"]
   → Create Layer 0 minicolumns

2. DOCUMENT-TOKEN CONNECTIONS
   → doc.feedforward_connections["L0_neural"] = 1.0
   → token.feedback_connections["L3_doc1"] = 1.0

3. LATERAL TOKEN CONNECTIONS
   → "neural" ↔ "networks" (co-occurrence)
   → "networks" ↔ "process" (co-occurrence)

4. BIGRAM EXTRACTION
   → ["neural networks", "networks process", "process data"]
   → Create Layer 1 minicolumns

5. BIGRAM-TOKEN CONNECTIONS
   → bigram.feedforward_connections["L0_neural"] = 1.0
   → token.feedback_connections["L1_neural networks"] = 1.0
```

**Important:** Bigrams use SPACE separators: `"neural networks"`, not `"neural_networks"`.

### Network Computation

**Location:** `processor.py:452-596` (`compute_all()`)

After processing documents, compute the full network:

```
1. ACTIVATION PROPAGATION
   → Spread activation through connections
   → Simulates information flow

2. PAGERANK
   → Compute importance for Layer 0 and Layer 1
   → Options: standard, semantic, hierarchical

3. TF-IDF
   → Compute term weights for Layer 0
   → Both global and per-document variants

4. DOCUMENT CONNECTIONS
   → Connect Layer 3 documents by shared vocabulary
   → Weight by sum of shared term TF-IDF scores

5. BIGRAM CONNECTIONS
   → Connect Layer 1 bigrams by:
     - Shared components ("neural networks" ↔ "neural processing")
     - Chain patterns ("machine learning" ↔ "learning algorithms")
     - Document co-occurrence

6. CONCEPT CLUSTERING
   → Run label propagation on Layer 0
   → Create Layer 2 concepts from clusters
   → Connect concepts to member tokens

7. CONCEPT CONNECTIONS
   → Connect Layer 2 concepts by:
     - Document overlap (Jaccard similarity)
     - Semantic relations between members
     - Embedding similarity (optional)
```

### Query Flow

**Location:** `query.py`

When a query is executed:

```
INPUT: "neural networks"

1. TOKENIZE QUERY
   → ["neural", "networks"]

2. EXPAND QUERY
   → Add related terms from lateral connections
   → Add terms from concept clusters
   → Result: {"neural": 1.0, "networks": 1.0, "learning": 0.5, ...}

3. SCORE DOCUMENTS
   → For each document, sum term scores:
     score = Σ(term_weight × token.tfidf_per_doc[doc_id])

4. RANK AND RETURN
   → Sort documents by score
   → Return top_n results
```

---

## Layer Details

### Layer 0: Tokens

**Purpose:** Represent individual words after tokenization.

**Content:** Lowercase stemmed words (stop words removed).

**Connections:**
- Lateral: Co-occurring tokens within window
- Feedback: Containing bigrams, concepts, documents
- Feedforward: None (lowest layer)

**Key Fields:**
- `occurrence_count`: Total times seen in corpus
- `document_ids`: Set of documents containing token
- `pagerank`: Importance score
- `tfidf`: Global TF-IDF weight
- `cluster_id`: Assigned concept cluster

### Layer 1: Bigrams

**Purpose:** Represent word pairs for phrase-level patterns.

**Content:** Space-separated word pairs: `"neural networks"`.

**Connections:**
- Lateral: Bigrams sharing components or co-occurring
- Feedforward: Component tokens
- Feedback: None typically (no Layer 2 → Layer 1 direct)

**Key Fields:**
- Same as Layer 0
- Bigrams inherit properties from component tokens

### Layer 2: Concepts

**Purpose:** Represent semantic topic clusters.

**Content:** Named by top members: `"neural/networks/learning"`.

**Connections:**
- Lateral: Concepts with overlapping documents or semantics
- Feedforward: Member tokens
- Feedback: None typically

**Creation:** Built by `build_concept_clusters()` using label propagation on Layer 0 tokens.

### Layer 3: Documents

**Purpose:** Represent full documents in the corpus.

**Content:** Document ID string.

**Connections:**
- Lateral: Documents sharing vocabulary
- Feedforward: All tokens in document
- Feedback: None (highest layer)

**Key Fields:**
- `document_ids`: Contains only self
- `occurrence_count`: 1 (single document)

---

## Performance Patterns

### O(1) ID Lookups

**Critical:** Always use `layer.get_by_id(col_id)` instead of iterating:

```python
# WRONG - O(n):
for col in layer.minicolumns.values():
    if col.id == target_id:
        neighbor = col

# RIGHT - O(1):
neighbor = layer.get_by_id(target_id)
```

Used throughout `analysis.py` and `query.py`.

### Staleness Tracking

**Location:** `processor.py:49`

```python
self._stale_computations: set
```

Tracks which computations need rerunning after corpus changes:
- `COMP_TFIDF`
- `COMP_PAGERANK`
- `COMP_ACTIVATION`
- `COMP_DOC_CONNECTIONS`
- `COMP_BIGRAM_CONNECTIONS`
- `COMP_CONCEPTS`

### Query Caching

**Location:** `processor.py:51-52`

```python
self._query_expansion_cache: Dict[str, Dict[str, float]]
self._query_cache_max_size: int = 100
```

LRU cache for query expansion results. Cleared after `compute_all()`.

---

## File Reference

| Component | File | Lines |
|-----------|------|-------|
| CorticalLayer enum | `layers.py` | 21-56 |
| HierarchicalLayer | `layers.py` | 59-273 |
| Minicolumn | `minicolumn.py` | 56-357 |
| Edge | `minicolumn.py` | 16-53 |
| process_document() | `processor.py` | 54-137 |
| compute_all() | `processor.py` | 452-596 |
| Tokenizer | `tokenizer.py` | Full file |

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: DOCUMENTS                        │
│  ┌─────────┐    ┌─────────┐                                 │
│  │  doc1   │←──→│  doc2   │  (lateral: shared vocab)        │
│  └────┬────┘    └────┬────┘                                 │
│       │              │      (feedforward: contained tokens) │
└───────┼──────────────┼──────────────────────────────────────┘
        ↓              ↓
┌───────┼──────────────┼──────────────────────────────────────┐
│       │   Layer 2: CONCEPTS                                 │
│  ┌────┴────┐    ┌────┴────┐                                │
│  │ concept1│←──→│ concept2│  (lateral: doc overlap)        │
│  └────┬────┘    └────┬────┘                                │
│       │              │      (feedforward: member tokens)    │
└───────┼──────────────┼──────────────────────────────────────┘
        ↓              ↓
┌───────┼──────────────┼──────────────────────────────────────┐
│       │   Layer 1: BIGRAMS                                  │
│  ┌────┴──────┐  ┌────┴──────┐                              │
│  │neural     │←→│networks   │  (lateral: shared component) │
│  │networks   │  │process    │                              │
│  └────┬──────┘  └────┬──────┘                              │
│       │              │      (feedforward: component tokens) │
└───────┼──────────────┼──────────────────────────────────────┘
        ↓              ↓
┌───────┼──────────────┼──────────────────────────────────────┐
│       │   Layer 0: TOKENS                                   │
│  ┌────┴────┐ ┌──────┐ ┌────┴────┐ ┌────────┐              │
│  │ neural  │←→│networks│←→│ process │←→│  data  │           │
│  └─────────┘ └──────┘ └─────────┘ └────────┘              │
│              (lateral: co-occurrence within window)         │
└─────────────────────────────────────────────────────────────┘
```
