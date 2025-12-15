# Core Algorithms

This document describes the information retrieval algorithms implemented in the Cortical Text Processor. These algorithms work together to build a semantic understanding of text corpora.

## Overview

The system uses standard IR algorithms with a hierarchical, layered architecture:

| Algorithm | Purpose | Primary File |
|-----------|---------|--------------|
| PageRank | Importance scoring | `analysis.py:22-95` |
| TF-IDF | Term weighting | `analysis.py:394-433` |
| Label Propagation | Concept clustering | `analysis.py:502-636` |
| Query Expansion | Semantic search | `query/expansion.py` |
| Relation Extraction | Knowledge building | `semantics.py:109-186` |

---

## PageRank - Importance Scoring

PageRank measures term importance based on network structure. Terms connected to other important terms receive higher scores.

### Standard PageRank

**Location:** `analysis.py:22-95`

**Algorithm:**
1. Initialize each term with equal PageRank: `1.0 / n`
2. Iterate until convergence:
   ```
   PR(i) = (1 - damping) / n + damping * Σ(PR(j) * weight(j→i) / outgoing(j))
   ```
3. Stop when max change < tolerance (1e-6) or after 20 iterations

**Parameters:**
- `damping`: 0.85 (probability of following links vs random jump)
- `iterations`: 20 maximum
- `tolerance`: 1e-6 convergence threshold

**Use Case:** Identify the most important terms in the corpus based on how they connect to other important terms.

### Semantic PageRank

**Location:** `analysis.py:113-235`

Enhances PageRank by weighting edges according to semantic relation types.

**Relation Weights:**
```
IsA: 1.5           (hypernym relationships are strong)
SimilarTo: 1.4     (similarity is important)
PartOf: 1.3        (part-whole relationships)
HasProperty: 1.2   (property associations)
DerivedFrom: 1.2   (morphological derivation)
Causes: 1.1        (causal relationships)
RelatedTo: 1.0     (general association - baseline)
UsedFor: 1.0       (functional relationships)
CoOccurs: 0.8      (basic co-occurrence - lower weight)
Antonym: 0.3       (opposing concepts - penalized)
```

**Use Case:** When semantic relations have been extracted, use semantic PageRank for importance that respects relationship types.

### Hierarchical PageRank

**Location:** `analysis.py:238-391`

Propagates importance across all 4 layers bidirectionally:
- Upward: tokens → bigrams → concepts → documents
- Downward: documents → concepts → bigrams → tokens

**Algorithm:**
1. Compute local PageRank within each layer
2. Propagate scores upward via `feedback_connections`
3. Propagate scores downward via `feedforward_connections`
4. Normalize within each layer
5. Repeat until convergence

**Parameters:**
- `layer_iterations`: 10 (within-layer iterations)
- `global_iterations`: 5 (cross-layer iterations)
- `cross_layer_damping`: 0.7 (damping at layer boundaries)

**Use Case:** When you want importance to flow through the full hierarchy, enabling documents to boost their constituent terms and vice versa.

---

## TF-IDF - Term Weighting

**Location:** `analysis.py:394-433`

TF-IDF (Term Frequency - Inverse Document Frequency) measures how distinctive a term is to the corpus.

**Formula:**
```
TF-IDF = TF × IDF
TF = log(1 + occurrence_count)
IDF = log(num_documents / document_frequency)
```

**Two Variants:**

1. **Global TF-IDF** (`col.tfidf`):
   - Uses total corpus occurrence count
   - Good for corpus-wide term importance

2. **Per-Document TF-IDF** (`col.tfidf_per_doc[doc_id]`):
   - Uses occurrence count within specific document
   - Better for document-specific relevance scoring

**Important:** Always use `tfidf_per_doc[doc_id]` for per-document scoring. The global `tfidf` field uses total occurrence count across all documents.

---

## Label Propagation - Concept Clustering

**Location:** `analysis.py:502-636`

Label propagation is a semi-supervised community detection algorithm that clusters tokens into semantic concepts.

**Algorithm:**
1. Each token starts with a unique label
2. Iterate up to 20 times:
   - Count neighbor labels weighted by connection strength
   - Adopt most common label if it exceeds change threshold
3. Group tokens by final label into clusters
4. Filter clusters smaller than `min_cluster_size`

**Parameters:**
- `cluster_strictness` (0.0-1.0): Higher = more separate clusters
- `bridge_weight` (0.0-1.0): Synthetic connections between documents
- `min_cluster_size`: Minimum tokens per cluster (default 3)

**Concept Creation:**
After clustering, each cluster becomes a concept in Layer 2:
- Named after top 3 members by PageRank: `"neural/networks/learning"`
- Connected bidirectionally to member tokens
- Aggregates member properties (documents, activation, pagerank)

---

## Query Expansion

### Basic Expansion

**Location:** `query/expansion.py`

Expands query terms to find semantically related words.

**Three Expansion Methods:**

1. **Lateral Connections** - Direct word associations from co-occurrence
   - Score: `connection_weight × neighbor_pagerank × 0.6`

2. **Concept Clusters** - Words from same semantic category
   - Score: `concept_pagerank × member_pagerank × 0.4`

3. **Code Concepts** - Programming synonyms (optional)
   - Example: "get" → "fetch", "load", "retrieve"
   - Score: `0.6`

### Multi-Hop Expansion

**Location:** `query/expansion.py`

Finds related terms through transitive relation chains.

**Example Chains:**
- `"dog" → IsA → "animal" → HasProperty → "living"`
- `"car" → PartOf → "engine" → UsedFor → "transportation"`

**Chain Validity Scoring:**
Not all relation chains are equally valid:
```
(IsA, IsA): 1.0           - Fully transitive hypernymy
(IsA, HasProperty): 0.9   - Property inheritance
(RelatedTo, RelatedTo): 0.6 - Weak association
(Antonym, Antonym): 0.3   - Double negation, unreliable
```

**Parameters:**
- `max_hops`: Maximum chain depth (default 2)
- `decay_factor`: Weight decay per hop (default 0.5)
- `min_path_score`: Minimum chain validity (default 0.2)

### Intent-Based Query Parsing

**Location:** `query/intent.py`

Parses natural language queries to extract intent.

**Intent Types:**
- `"where"` → `location` (find file/function location)
- `"how"` → `implementation` (find implementation details)
- `"what"` → `definition` (find definitions)
- `"why"` → `rationale` (find explanations/comments)
- `"when"` → `lifecycle` (find lifecycle events)

**Example:**
```
Input: "where do we handle authentication?"
Output: ParsedIntent(
    action='handle',
    subject='authentication',
    intent='location',
    expanded_terms=['handle', 'manage', 'authentication', 'auth', ...]
)
```

---

## Relation Extraction

### Pattern-Based Extraction

**Location:** `semantics.py:109-186`

Extracts semantic relations from text using regex patterns.

**Relation Types:**
- **IsA**: "dogs are animals", "a kind of"
- **HasA**: "dogs have ears", "contains"
- **PartOf**: "wheel is part of car"
- **UsedFor**: "hammer is used for nailing"
- **Causes**: "rain causes floods"
- **CapableOf**: "dog can bark"
- **AtLocation**: "found in", "lives in"
- **HasProperty**: "dog is loyal"
- **Antonym**: "big vs small", "opposite of"
- **DerivedFrom**: "comes from"

Each pattern has a confidence score (0.5-0.95) based on how reliable it is.

### Co-occurrence Relations

**Location:** `semantics.py:251-292`

Extracts relations from statistical co-occurrence.

**Algorithm:**
1. Count term pairs within sliding window (5 tokens)
2. Compute PMI (Pointwise Mutual Information):
   ```
   PMI = log((co-occurrence + 1) / (expected + 1))
   expected = (count_term1 × count_term2) / corpus_size
   ```
3. Create `CoOccurs` relations for high-PMI pairs

### Similarity Relations

**Location:** `semantics.py:294-363`

Finds similar terms based on context vectors.

**Algorithm:**
1. Build context vectors: what words appear near each term
2. Compute cosine similarity between context vectors
3. Create `SimilarTo` relations for pairs with similarity > 0.3

---

## Retrofitting

**Location:** `semantics.py:378-476`

Adjusts connection weights to align with semantic relations.

**Algorithm:**
1. Store original lateral connection weights
2. Build semantic neighbor lookup
3. Iterate 10 times:
   - Blend original and semantic weights:
     ```
     new_weight = alpha × original + (1 - alpha) × semantic
     ```
   - Add new semantic connections that didn't exist

**Parameter:**
- `alpha`: 0.3 (mostly semantic, some original)

**Use Case:** If "dog" and "cat" aren't connected by co-occurrence but both have "IsA animal" relation, retrofitting strengthens their connection.

---

## Performance Optimizations

| Optimization | Location | Benefit |
|--------------|----------|---------|
| O(1) ID lookups | `layer.get_by_id()` | Avoid O(n) iteration |
| Query cache | `expand_query_cached()` | Skip repeated expansions |
| Pre-computed lookups | `precompute_term_cols()` | Faster chunk scoring |
| Fast search | `fast_find_documents()` | 2-3x faster via candidate filtering |
| Inverted index | `build_document_index()` | Fastest repeated queries |

---

## Quick Reference

**When to use which algorithm:**

| Goal | Algorithm | Method |
|------|-----------|--------|
| Find important terms | PageRank | `compute_pagerank()` |
| Respect semantic relations | Semantic PageRank | `compute_semantic_importance()` |
| Cross-layer importance | Hierarchical PageRank | `compute_hierarchical_importance()` |
| Term distinctiveness | TF-IDF | `compute_tfidf()` |
| Group related terms | Label Propagation | `build_concept_clusters()` |
| Expand search queries | Query Expansion | `expand_query()` |
| Find distant relations | Multi-hop Expansion | `expand_query_multihop()` |
| Extract knowledge | Relation Extraction | `extract_corpus_semantics()` |
| Improve connections | Retrofitting | `retrofit_connections()` |
