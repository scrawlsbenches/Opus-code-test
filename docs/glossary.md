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

**Location:** `query/expansion.py`

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

**Location:** `query/intent.py`

### Multi-hop Expansion
Query expansion through chains of semantic relations. Finds terms 2+ hops away through valid relation paths.

**Location:** `query/expansion.py`

### Chunk
A segment of document text for passage retrieval. Created with configurable size and overlap.

**Location:** `query/chunking.py`

### Inverted Index
Pre-computed mapping from terms to containing documents. Enables fast candidate filtering.

**Location:** `query/search.py`

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

## Hubris MoE System

### MicroExpert
Base class for specialized prediction experts. Each expert focuses on a specific aspect of coding tasks (files, tests, errors, workflows).

**Location:** `scripts/hubris/micro_expert.py`

**Subclasses:**
- `FileExpert` - Predicts files to modify
- `TestExpert` - Predicts tests to run
- `ErrorDiagnosisExpert` - Diagnoses errors
- `EpisodeExpert` - Learns workflow patterns

### ExpertPrediction
Standardized output format from any expert. Contains ranked items with confidence scores.

**Fields:**
- `expert_id`: Which expert made the prediction
- `expert_type`: Category of expert (file, test, error, episode)
- `items`: List of (item, confidence) tuples
- `metadata`: Context about how prediction was made

### Credit System
Value-based economy where experts earn/lose credits based on prediction accuracy. Higher credits = more influence in ensemble voting.

**Location:** `scripts/hubris/credit_account.py`, `scripts/hubris/credit_router.py`

### CreditLedger
Central registry of all expert credit accounts. Manages balances, transactions, and inter-expert transfers.

**Key Methods:**
- `get_or_create_account()`: Get or initialize expert account
- `get_top_experts()`: Rank experts by balance
- `transfer()`: Move credits between accounts

### CreditAccount
Individual expert's credit balance and transaction history. Starts at 100 credits.

**Fields:**
- `balance`: Current credit balance
- `transactions`: History of credits/debits

### ValueSignal
Feedback from real-world outcomes that triggers credit updates. Generated when predictions resolve.

**Types:**
- `positive`: Prediction was correct
- `negative`: Prediction was wrong
- `neutral`: Inconclusive

**Location:** `scripts/hubris/value_signal.py`

### Cold-Start Mode
Initial state when experts haven't learned yet (all have default 100 credits). Predictions show low confidence and ML fallback is offered.

**Detection:** `is_cold_start()` in `hubris_cli.py`

### Calibration
Analysis of prediction confidence vs actual accuracy. Well-calibrated experts have confidence that matches their hit rate.

**Location:** `scripts/hubris/calibration_tracker.py`

### ECE (Expected Calibration Error)
Average gap between predicted confidence and actual accuracy across confidence bins. Lower is better.

**Formula:** `ECE = Σ(bin_size/total) × |accuracy - confidence|`

**Interpretation:**
- < 0.05: Excellent
- < 0.10: Good
- < 0.15: Acceptable
- ≥ 0.15: Needs attention

### MCE (Max Calibration Error)
Worst calibration gap in any single bin. Identifies where predictions are most unreliable.

### Brier Score
Mean squared error of probabilistic predictions. Combines calibration and discrimination.

**Formula:** `Brier = (1/n) × Σ(confidence - outcome)²`

### Staking
Mechanism for experts to bet credits on high-confidence predictions. Higher multipliers = higher risk/reward.

**Strategies:**
- CONSERVATIVE (1.0x)
- MODERATE (1.5x)
- AGGRESSIVE (2.0x)
- YOLO (3.0x)

**Location:** `scripts/hubris/staking.py`

### Confidence Boost
Bonus applied to high-credit experts (>150 credits) during aggregation. Up to +10% confidence.

### Temperature (Credit Routing)
Softmax temperature controlling weight distribution. Lower = sharper distinctions, higher = more democratic.

### Disagreement Score
Measure of how much experts disagree on predictions. High disagreement reduces ensemble confidence.

---

## File Locations Quick Reference

| Term | Primary File |
|------|--------------|
| Minicolumn | `cortical/minicolumn.py` |
| Edge | `cortical/minicolumn.py` |
| HierarchicalLayer | `cortical/layers.py` |
| CorticalLayer | `cortical/layers.py` |
| PageRank | `cortical/analysis.py` |
| TF-IDF | `cortical/analysis.py` |
| Label Propagation | `cortical/analysis.py` |
| Query Expansion | `cortical/query/expansion.py` |
| Relation Extraction | `cortical/semantics.py` |
| Retrofitting | `cortical/semantics.py` |
| Tokenization | `cortical/tokenizer.py` |
| Fingerprint | `cortical/fingerprint.py` |
| Code Concepts | `cortical/code_concepts.py` |
| MicroExpert | `scripts/hubris/micro_expert.py` |
| FileExpert | `scripts/hubris/experts/file_expert.py` |
| TestExpert | `scripts/hubris/experts/test_expert.py` |
| CreditLedger | `scripts/hubris/credit_account.py` |
| CreditRouter | `scripts/hubris/credit_router.py` |
| ValueSignal | `scripts/hubris/value_signal.py` |
| CalibrationTracker | `scripts/hubris/calibration_tracker.py` |
| Staking | `scripts/hubris/staking.py` |
| Hubris CLI | `scripts/hubris_cli.py` |
