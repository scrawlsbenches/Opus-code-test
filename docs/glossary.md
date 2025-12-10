# Glossary

This glossary defines terminology used throughout the Cortical Text Processor codebase. Terms are organized by category for easy reference.

---

## Core Data Structures

### Minicolumn
The fundamental unit of representation at each layer. Named after cortical minicolumns in neuroscience, but implemented as a data structure holding connections, statistics, and metadata.

**Location:** `minicolumn.py:56-357`

**Fields:**
- `id`: Unique identifier (e.g., "L0_neural")
- `content`: The actual content (word, bigram, concept name, or doc_id)
- `layer`: Layer number (0-3)
- Various connection dictionaries and statistics

### Edge
A typed connection with metadata, used for ConceptNet-style semantic edges.

**Location:** `minicolumn.py:16-53`

**Fields:**
- `target_id`: Target minicolumn ID
- `weight`: Connection strength
- `relation_type`: Semantic type ('IsA', 'PartOf', 'CoOccurs', etc.)
- `confidence`: Reliability score [0.0, 1.0]
- `source`: Origin ('corpus', 'semantic', 'inferred')

### HierarchicalLayer
Container that holds all minicolumns at a specific layer level.

**Location:** `layers.py:59-273`

**Key Features:**
- `minicolumns` dict maps content to Minicolumn objects
- `_id_index` provides O(1) lookup by minicolumn ID
- Methods: `get_or_create_minicolumn()`, `get_by_id()`, `column_count()`

### CorticalLayer
Enumeration defining the 4 processing layers.

**Location:** `layers.py:21-56`

```
TOKENS = 0      # Individual words
BIGRAMS = 1     # Word pairs
CONCEPTS = 2    # Semantic clusters
DOCUMENTS = 3   # Full documents
```

---

## Connection Types

### Lateral Connections
**Within-layer** connections between minicolumns at the same level. Built from co-occurrence patterns (tokens appearing near each other in text).

**Storage:** `minicolumn.lateral_connections: Dict[str, float]`

**Use:** Query expansion, PageRank computation, spreading activation.

### Typed Connections
**Within-layer** connections with semantic metadata. Store relation type, confidence, and source information.

**Storage:** `minicolumn.typed_connections: Dict[str, Edge]`

**Use:** Semantic PageRank, ConceptNet-style reasoning.

### Feedforward Connections
**Cross-layer** connections pointing downward (higher layer → lower layer). Connect containers to their components.

**Storage:** `minicolumn.feedforward_connections: Dict[str, float]`

**Examples:**
- Bigram → component tokens
- Concept → member tokens
- Document → contained tokens

### Feedback Connections
**Cross-layer** connections pointing upward (lower layer → higher layer). Connect components to their containers.

**Storage:** `minicolumn.feedback_connections: Dict[str, float]`

**Examples:**
- Token → containing bigrams
- Token → containing concepts
- Token → containing documents

---

## Algorithms

### PageRank
Graph algorithm measuring importance based on connection structure. Terms connected to other important terms receive higher scores.

**Formula:** `PR(i) = (1-d)/n + d × Σ(PR(j) × w(j→i) / out(j))`

**Location:** `analysis.py:22-95`

**Variants:**
- Standard PageRank: Equal edge weights
- Semantic PageRank: Weights edges by relation type
- Hierarchical PageRank: Propagates across layers

### TF-IDF
Term Frequency - Inverse Document Frequency. Measures how distinctive a term is to documents in the corpus.

**Formula:** `TF-IDF = log(1 + count) × log(num_docs / doc_frequency)`

**Location:** `analysis.py:394-433`

**Variants:**
- Global: Uses total corpus occurrence (`col.tfidf`)
- Per-document: Uses document-specific count (`col.tfidf_per_doc[doc_id]`)

### Label Propagation
Community detection algorithm for clustering. Tokens adopt the most common label among their neighbors, causing related tokens to converge to the same cluster.

**Location:** `analysis.py:502-636`

**Parameters:**
- `cluster_strictness`: Higher = more separate clusters
- `bridge_weight`: Synthetic inter-document connections

### Damping Factor
PageRank parameter (default 0.85) representing probability of following a link vs. random jump. Lower damping = more randomness in importance distribution.

### Query Expansion
Process of adding related terms to a search query based on lateral connections, concept membership, or semantic relations.

**Location:** `query.py:55-176`

### Spreading Activation
Information propagation through connections. Activation starts at query terms and spreads to connected nodes, simulating neural activation patterns.

---

## Semantic Relations

### IsA
Hypernym/hyponym relationship. "A dog IsA animal" means dog is a type of animal.

**Weight:** 1.5 (highest)

### PartOf
Meronym/holonym relationship. "Wheel PartOf car" means wheel is a component of car.

**Weight:** 1.3

### HasA / HasProperty
Property or component ownership. "Dog HasProperty loyal" or "Dog HasA tail".

**Weight:** 1.2

### SimilarTo
Similarity without hierarchy. "Dog SimilarTo cat" - both are pets/animals.

**Weight:** 1.4

### RelatedTo
General association from co-occurrence. Default relation type.

**Weight:** 1.0

### CoOccurs
Statistical co-occurrence in text. Lower confidence than explicit relations.

**Weight:** 0.8

### Causes
Causal relationship. "Rain Causes floods".

**Weight:** 1.1

### UsedFor
Functional purpose. "Hammer UsedFor nailing".

**Weight:** 1.0

### Antonym
Opposition/contrast. "Big Antonym small".

**Weight:** 0.3 (penalized)

### DerivedFrom
Morphological or etymological derivation.

**Weight:** 1.2

---

## Processing Concepts

### Tokenization
Breaking text into individual word tokens. Includes lowercasing, stop word removal, and optional stemming.

**Location:** `tokenizer.py`

### Bigram
A pair of consecutive tokens. Stored with SPACE separator: "neural networks" (not underscore).

**Location:** `tokenizer.py:303-316`

### Concept Cluster
Group of semantically related tokens discovered through label propagation. Becomes a minicolumn in Layer 2.

### Corpus
The collection of all documents processed by the system.

### Retrofitting
Post-processing that adjusts lateral connection weights to align with semantic relations. Blends co-occurrence patterns with semantic knowledge.

**Location:** `semantics.py:378-476`

---

## Architecture Concepts

### 4-Layer Hierarchy
The core architecture organizing text at increasing abstraction levels:
- Layer 0: TOKENS (words)
- Layer 1: BIGRAMS (word pairs)
- Layer 2: CONCEPTS (topic clusters)
- Layer 3: DOCUMENTS (full texts)

### Cortical Metaphor
The naming convention draws from neuroscience (V1→V2→V4→IT visual cortex pathway) but implementations are standard IR algorithms, not neural models.

### Staleness Tracking
System for knowing which computations need rerunning after corpus changes. Prevents unnecessary recomputation.

**Location:** `processor.py:49`

---

## Search Concepts

### Intent Parsing
Extracting user intent from natural language queries. Maps question words to intent types (where→location, how→implementation).

**Location:** `query.py:179-284`

### Multi-hop Expansion
Query expansion through chains of semantic relations. Finds terms 2+ hops away through valid relation paths.

**Location:** `query.py:407-531`

### Chunk
A segment of document text for passage retrieval. Created with configurable size and overlap.

**Location:** `query.py:937-978`

### Inverted Index
Pre-computed mapping from terms to containing documents. Enables fast candidate filtering.

**Location:** `query.py` (fast search functions)

---

## Code Concepts

### Programming Concept Groups
Collections of synonymous programming terms. "get", "fetch", "load", "retrieve" are grouped together.

**Location:** `code_concepts.py`

### Code-Aware Tokenization
Tokenization that splits identifiers: `getUserName` → `["getusername", "get", "user", "name"]`.

**Location:** `tokenizer.py` (split_identifiers parameter)

### Semantic Fingerprint
Vector representation of a text's semantic content for similarity comparison.

**Location:** `fingerprint.py`

---

## Performance Concepts

### O(1) ID Lookup
Using `layer.get_by_id(col_id)` instead of iterating minicolumns. Critical for algorithm performance.

### Query Cache
LRU cache storing query expansion results to avoid recomputation for repeated queries.

**Location:** `processor.py:51-52`

### Batch Processing
Processing multiple queries or documents together to amortize overhead.

**Functions:** `find_documents_batch()`, `find_passages_batch()`, `add_documents_batch()`

---

## File Locations Quick Reference

| Term | Primary File |
|------|--------------|
| Minicolumn | `minicolumn.py` |
| Edge | `minicolumn.py` |
| HierarchicalLayer | `layers.py` |
| CorticalLayer | `layers.py` |
| PageRank | `analysis.py` |
| TF-IDF | `analysis.py` |
| Label Propagation | `analysis.py` |
| Query Expansion | `query.py` |
| Relation Extraction | `semantics.py` |
| Retrofitting | `semantics.py` |
| Tokenization | `tokenizer.py` |
| Fingerprint | `fingerprint.py` |
| Code Concepts | `code_concepts.py` |
