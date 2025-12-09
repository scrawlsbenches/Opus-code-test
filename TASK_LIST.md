# Task List: Bug Fixes & RAG Enhancements

This document tracks bug fixes and feature enhancements for the Cortical Text Processor.

**Last Updated:** 2025-12-09
**Status:** Bug fixes complete | RAG enhancements planned

---

## Critical Priority

### 1. Fix Per-Document TF-IDF Calculation Bug

**File:** `cortical/analysis.py`
**Line:** 131
**Status:** [x] Completed

**Problem:**
The per-document term frequency calculation was incorrect. The code always returned 1.

**Solution Applied:**
1. Added `doc_occurrence_counts: Dict[str, int]` field to `Minicolumn` class
2. Updated `processor.py` to track per-document token occurrences during document processing
3. Fixed TF-IDF calculation to use actual occurrence counts: `col.doc_occurrence_counts.get(doc_id, 1)`

**Files Modified:**
- `cortical/minicolumn.py` - Added new field and serialization support
- `cortical/processor.py` - Track occurrences per document
- `cortical/analysis.py` - Use actual counts in TF-IDF calculation

---

## High Priority

### 2. Add ID-to-Minicolumn Lookup Optimization

**Files:** `cortical/layers.py`, `cortical/analysis.py`, `cortical/query.py`
**Status:** [x] Completed

**Problem:**
Multiple O(n) linear searches occurred when looking up minicolumns by ID.

**Solution Applied:**
1. Added `_id_index: Dict[str, str]` secondary index to `HierarchicalLayer`
2. Added `get_by_id()` method for O(1) lookups
3. Updated `from_dict()` to rebuild index when loading
4. Replaced all linear searches with `get_by_id()` calls

**Files Modified:**
- `cortical/layers.py` - Added `_id_index` and `get_by_id()` method
- `cortical/analysis.py` - Updated PageRank, activation propagation, label propagation
- `cortical/query.py` - Updated query expansion, spreading activation, related documents

**Performance Impact:** Graph algorithms improved from O(n²) to O(n)

---

## Medium Priority

### 3. Fix Type Annotation Errors

**File:** `cortical/semantics.py`
**Lines:** 153, 248
**Status:** [x] Completed

**Solution Applied:**
1. Added `Any` to imports
2. Changed `any` to `Any` on both lines

---

### 4. Remove Unused Import

**File:** `cortical/analysis.py`
**Line:** 16
**Status:** [x] Completed

**Solution Applied:**
Removed `Counter` from the import statement.

---

### 5. Fix Unconditional Print in Export Function

**File:** `cortical/persistence.py`
**Lines:** 175-176
**Status:** [x] Completed

**Solution Applied:**
1. Added `verbose: bool = True` parameter to `export_graph_json()`
2. Wrapped print statements in `if verbose:` conditional

---

## Low Priority

### 6. Add Missing Test Coverage

**Files:** `tests/test_embeddings.py`, `tests/test_semantics.py`, `tests/test_gaps.py`, `tests/test_analysis.py`, `tests/test_persistence.py`
**Status:** [x] Completed

**Tests Added:**

**test_embeddings.py (15 tests):**
- `test_compute_graph_embeddings_adjacency`
- `test_compute_graph_embeddings_random_walk`
- `test_compute_graph_embeddings_spectral`
- `test_compute_graph_embeddings_invalid_method`
- `test_embedding_similarity`
- `test_embedding_similarity_self`
- `test_embedding_similarity_missing_term`
- `test_find_similar_by_embedding`
- `test_find_similar_by_embedding_missing_term`
- `test_embedding_dimensions`
- `test_embedding_normalization`
- `test_empty_layer_embeddings`

**test_semantics.py (12 tests):**
- `test_extract_corpus_semantics`
- `test_extract_corpus_semantics_cooccurs`
- `test_retrofit_connections`
- `test_retrofit_connections_affects_weights`
- `test_retrofit_embeddings`
- `test_get_relation_type_weight`
- `test_relation_weights_constant`
- `test_empty_corpus_semantics`
- `test_retrofit_empty_relations`
- `test_larger_window_more_relations`

**test_gaps.py (15 tests):**
- `test_analyze_knowledge_gaps_structure`
- `test_analyze_knowledge_gaps_summary`
- `test_analyze_knowledge_gaps_isolated_documents`
- `test_analyze_knowledge_gaps_weak_topics`
- `test_analyze_knowledge_gaps_coverage_score`
- `test_detect_anomalies_structure`
- `test_detect_anomalies_reasons`
- `test_detect_anomalies_sorted`
- `test_detect_anomalies_threshold`
- `test_empty_corpus_gaps`
- `test_single_document_gaps`
- `test_single_document_anomalies`
- `test_bridge_opportunities_format`

**test_analysis.py (17 tests):**
- `test_pagerank_empty_layer`
- `test_pagerank_single_node`
- `test_pagerank_multiple_nodes`
- `test_pagerank_convergence`
- `test_tfidf_empty_corpus`
- `test_tfidf_single_document`
- `test_tfidf_multiple_documents`
- `test_tfidf_per_document`
- `test_propagation_empty_layers`
- `test_propagation_preserves_activation`
- `test_clustering_empty_layer`
- `test_clustering_returns_dict`
- `test_clustering_min_size`
- `test_build_concept_clusters`
- `test_document_connections`
- `test_cosine_similarity` (5 sub-tests)
- `test_get_by_id_returns_correct_minicolumn`
- `test_get_by_id_returns_none_for_missing`

**test_persistence.py (12 tests):**
- `test_save_and_load`
- `test_save_load_preserves_id_index`
- `test_save_load_preserves_doc_occurrence_counts`
- `test_save_load_empty_processor`
- `test_export_graph_json`
- `test_export_graph_json_layer_filter`
- `test_export_graph_json_min_weight`
- `test_export_graph_json_max_nodes`
- `test_export_graph_json_verbose_false`
- `test_export_embeddings_json`
- `test_export_embeddings_json_with_metadata`
- `test_get_state_summary`
- `test_get_state_summary_empty`

**Test Coverage Summary:**
- Previous: 39 tests
- Added: 70 new tests
- **Total: 109 tests (all passing)**

---

### 7. Document Magic Numbers

**File:** `cortical/gaps.py`
**Lines:** 62, 76, 99
**Status:** [ ] Deferred

**Note:** This task remains as a future enhancement. The magic numbers are functional but could benefit from documentation or configuration options.

---

---

# RAG System Enhancements

The following tasks are required to transform the Cortical Text Processor into a production-ready RAG (Retrieval-Augmented Generation) system.

---

## RAG Critical Priority

### 8. Implement Chunk-Level Retrieval

**Files:** `cortical/processor.py`, `cortical/query.py`
**Status:** [x] Completed

**Problem:**
Current retrieval returns only document IDs and scores. RAG systems need actual text passages with position information for context windows and citations.

**Current Behavior:**
```python
results = processor.find_documents_for_query("neural networks")
# Returns: [("doc1", 3.47), ("doc2", 2.15)]  # Just IDs!
```

**Required Behavior:**
```python
results = processor.find_passages_for_query("neural networks")
# Returns: [
#   ("Neural networks process information...", "doc1", 1500, 2000, 3.47),
#   (text, doc_id, start_char, end_char, score)
# ]
```

**Implementation Steps:**
1. Add `find_passages_for_query()` method to `processor.py`
2. Add `_create_chunks()` helper for splitting documents with overlap
3. Add `_score_tokens()` helper for chunk-level scoring
4. Add corresponding function to `query.py` for standalone use
5. Support configurable `chunk_size` (default 512) and `overlap` (default 128)

**Files to Modify:**
- `cortical/processor.py` - Add new methods (~50 lines)
- `cortical/query.py` - Add standalone function (~40 lines)
- `tests/test_processor.py` - Add tests for chunk retrieval
- `tests/test_query.py` - Add tests for passage finding

---

### 9. Add Document Metadata Support

**Files:** `cortical/processor.py`, `cortical/persistence.py`
**Status:** [x] Completed

**Problem:**
No way to store or retrieve document metadata (source URL, timestamp, author, etc.). RAG systems need this for proper citations and filtering.

**Current Data Model:**
```python
self.documents: Dict[str, str] = {}  # Only doc_id → text
```

**Required Data Model:**
```python
self.documents: Dict[str, str] = {}
self.document_metadata: Dict[str, Dict[str, Any]] = {}
# Stores: source, timestamp, author, category, custom fields
```

**Implementation Steps:**
1. Add `document_metadata` dict to `CorticalTextProcessor.__init__()`
2. Modify `process_document()` to accept optional `metadata` parameter
3. Add `set_document_metadata()` and `get_document_metadata()` methods
4. Update `persistence.py` to save/load metadata
5. Increment state version to `2.1`

**Files to Modify:**
- `cortical/processor.py` - Add metadata storage and methods
- `cortical/persistence.py` - Update save/load functions
- `tests/test_persistence.py` - Add metadata persistence tests

---

## RAG High Priority

### 10. Activate Layer 2 (Concept Clustering) by Default

**Files:** `cortical/processor.py`, `cortical/query.py`
**Status:** [x] Completed

**Problem:**
Layer 2 (Concepts) has clustering code but is never populated automatically. This layer could enable topic-based filtering and hierarchical search.

**Current Behavior:**
- `compute_all()` does NOT call `build_concept_clusters()`
- Layer 2 remains empty with 0 minicolumns
- Query expansion code checks Layer 2 but finds nothing

**Implementation Steps:**
1. Add `build_concepts: bool = True` parameter to `compute_all()`
2. Call `build_concept_clusters()` when enabled
3. Update query expansion to use concepts when available
4. Add concept-based document filtering option

**Files to Modify:**
- `cortical/processor.py` - Update `compute_all()` (~10 lines)
- `cortical/query.py` - Enhance expansion logic (~20 lines)

---

### 11. Integrate Semantic Relations into Retrieval

**Files:** `cortical/query.py`
**Status:** [x] Completed

**Problem:**
`semantics.py` extracts relations (IsA, PartOf, RelatedTo, etc.) but they're only used for retrofitting embeddings, not for query expansion or retrieval.

**Current State:**
- `expand_query_semantic()` exists in `query.py` (lines 127-174)
- This function is NEVER called by `find_documents_for_query()`
- Semantic relations are computed but ignored during search

**Implementation Steps:**
1. Add `use_semantic: bool = True` parameter to `find_documents_for_query()`
2. Call `expand_query_semantic()` when semantic relations exist
3. Combine lateral connection expansion with semantic expansion
4. Weight semantic expansions appropriately

**Files to Modify:**
- `cortical/query.py` - Integrate semantic expansion (~15 lines)

---

### 12. Persist Full Computed State

**Files:** `cortical/persistence.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Embeddings, semantic relations, and concept clusters are not saved. Loading a model requires expensive recomputation.

**Currently Saved:**
- Layers (tokens, bigrams, documents)
- Document text
- Generic metadata

**NOT Saved (must recompute):**
- `semantic_relations` - extracted IsA, PartOf, etc.
- `embeddings` - graph embeddings for all terms
- Concept clusters in Layer 2

**Implementation Steps:**
1. Add `semantic_relations` to save state
2. Add `embeddings` to save state
3. Update `load_processor()` to restore these fields
4. Increment state version to `2.1`
5. Handle backward compatibility with v2.0 files

**Files to Modify:**
- `cortical/persistence.py` - Update save/load (~30 lines)
- `cortical/processor.py` - Update save/load methods

---

## RAG Medium Priority

### 13. Fix Remaining Type Annotation

**File:** `cortical/embeddings.py`
**Line:** 26
**Status:** [x] Completed

**Problem:**
```python
# Current (incorrect):
) -> Tuple[Dict[str, List[float]], Dict[str, any]]:

# Should be:
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
```

**Implementation:** Single line fix, add `Any` to imports.

---

### 14. Optimize Spectral Embeddings Lookup

**File:** `cortical/embeddings.py`
**Lines:** 151-156
**Status:** [x] Completed

**Problem:**
Spectral embeddings use O(n) linear search instead of O(1) `get_by_id()`:
```python
# Current (slow):
for t, c in layer.minicolumns.items():
    if c.id == neighbor_id:
        ...

# Should use:
neighbor = layer.get_by_id(neighbor_id)
```

---

### 15. Add Incremental Document Indexing

**File:** `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Adding a document requires calling `compute_all()` which recomputes everything. For RAG systems with frequent updates, this is inefficient.

**Solution Applied:**
1. Added staleness tracking with `_stale_computations` set and computation type constants
2. Added `add_document_incremental()` method with selectable recomputation levels:
   - `'none'`: Just add document, mark computations stale (fastest)
   - `'tfidf'`: Recompute TF-IDF only (good for search)
   - `'full'`: Run full `compute_all()` (most accurate)
3. Added `add_documents_batch()` for efficient batch additions with single recomputation
4. Added `recompute()` method with levels: `'stale'`, `'tfidf'`, `'full'`
5. Added helper methods: `is_stale()`, `get_stale_computations()`, `_mark_fresh()`, `_mark_all_stale()`

**Files Modified:**
- `cortical/processor.py` - Added incremental indexing methods (~200 lines)
- `tests/test_processor.py` - Added 15 tests for incremental indexing

**Usage Examples:**
```python
# Quick incremental update (TF-IDF only)
processor.add_document_incremental("new_doc", "content", recompute='tfidf')

# Batch add with deferred recomputation
processor.add_document_incremental("doc1", "content1", recompute='none')
processor.add_document_incremental("doc2", "content2", recompute='none')
processor.recompute(level='full')  # Single recomputation for batch

# Efficient batch API
docs = [("doc1", "content1", {"source": "web"}), ("doc2", "content2", None)]
processor.add_documents_batch(docs, recompute='full')
```

---

## RAG Low Priority

### 16. Document Magic Numbers in Gap Detection

**File:** `cortical/gaps.py`
**Lines:** 62, 76, 99
**Status:** [ ] Deferred (carried over)

**Magic Numbers:**
- `avg_sim < 0.02` - isolation threshold
- `tfidf > 0.005` - weak topic threshold
- `0.005 < sim < 0.03` - bridge opportunity range

**Implementation:** Add docstrings or make configurable parameters.

---

### 17. Add Multi-Stage Ranking Pipeline

**Files:** `cortical/query.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Current ranking is flat (Token TF-IDF → Document Score). Better RAG performance with staged ranking.

**Solution Applied:**
Implemented a 4-stage ranking pipeline:

1. **Stage 1 (Concepts):** Find relevant concepts from Layer 2 clusters, score by query term overlap
2. **Stage 2 (Documents):** Rank documents using combined concept + TF-IDF scores
3. **Stage 3 (Chunks):** Score passages within top documents using chunk-level TF-IDF
4. **Stage 4 (Rerank):** Combine all signals (chunk 50%, TF-IDF 30%, concept 20%) for final scoring

**Files Modified:**
- `cortical/query.py` - Added `find_relevant_concepts()`, `multi_stage_rank()`, `multi_stage_rank_documents()` (~300 lines)
- `cortical/processor.py` - Added processor wrapper methods (~90 lines)
- `tests/test_processor.py` - Added 15 tests for multi-stage ranking

**Usage Examples:**
```python
# Full 4-stage ranking (passages with stage breakdown)
results = processor.multi_stage_rank("neural networks", top_n=5, concept_boost=0.3)
for passage, doc_id, start, end, score, stages in results:
    print(f"[{doc_id}] Score: {score:.3f}")
    print(f"  Concept: {stages['concept_score']:.3f}")
    print(f"  Doc: {stages['doc_score']:.3f}")
    print(f"  Chunk: {stages['chunk_score']:.3f}")

# Document-level ranking (stages 1-2 only)
results = processor.multi_stage_rank_documents("neural networks", top_n=3)
for doc_id, score, stages in results:
    print(f"{doc_id}: {score:.3f} (concept: {stages['concept_score']:.3f})")
```

---

### 18. Add Batch Query API

**Files:** `cortical/query.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
No efficient way to run multiple queries. Each query repeats tokenization and expansion.

**Solution Applied:**
1. Added `find_documents_batch()` function to `query.py` with expansion caching
2. Added `find_passages_batch()` function to `query.py` with chunk pre-computation
3. Added corresponding methods to `CorticalTextProcessor`
4. Both functions share tokenization and expansion caches across queries

**Files Modified:**
- `cortical/query.py` - Added batch query functions (~180 lines)
- `cortical/processor.py` - Added processor wrapper methods (~90 lines)
- `tests/test_processor.py` - Added 14 tests for batch query functionality

**Usage Examples:**
```python
# Batch document search
queries = ["neural networks", "machine learning", "data processing"]
results = processor.find_documents_batch(queries, top_n=3)
for query, docs in zip(queries, results):
    print(f"{query}: {[doc_id for doc_id, _ in docs]}")

# Batch passage search (for RAG)
results = processor.find_passages_batch(queries, top_n=5, chunk_size=512)
for query, passages in zip(queries, results):
    print(f"{query}: {len(passages)} passages found")
```

---

---

# ConceptNet-Enhanced PageRank

The following tasks implement a ConceptNet-like enhanced PageRank algorithm that leverages semantic relations, cross-layer connections, and typed edge weights for improved concept importance scoring.

---

## ConceptNet Critical Priority

### 19. Build Cross-Layer Feedforward Connections

**Files:** `cortical/analysis.py`, `cortical/processor.py`, `cortical/minicolumn.py`
**Status:** [x] Completed

**Problem:**
Layers were isolated - concepts didn't connect back to their member tokens, and bigrams didn't link to component unigrams. This broke the hierarchical flow needed for cross-layer PageRank.

**Solution Applied:**
1. Added `feedforward_connections: Dict[str, float]` to Minicolumn (weighted links to lower layer)
2. Added `feedback_connections: Dict[str, float]` to Minicolumn (weighted links to higher layer)
3. Added helper methods: `add_feedforward_connection()`, `add_feedback_connection()`
4. Updated bigram creation to link to component tokens with weight 1.0 per occurrence
5. Updated document processing to create bidirectional doc↔token connections
6. Updated concept creation to link to member tokens weighted by normalized PageRank
7. Updated `to_dict()`/`from_dict()` for persistence

**Files Modified:**
- `cortical/minicolumn.py` - Added connection fields and helper methods (~50 lines)
- `cortical/processor.py` - Populate feedforward/feedback during document processing
- `cortical/analysis.py` - Updated `build_concept_clusters()` to create weighted links
- `tests/test_processor.py` - Added 12 tests for cross-layer connections

**Connection Types:**
```python
# Bigram → Tokens (weight by occurrence count)
bigram.feedforward_connections["L0_neural"] = 2.0  # seen twice

# Token → Bigrams (feedback)
token.feedback_connections["L1_neural_networks"] = 2.0

# Document → Tokens (weight by term frequency)
doc.feedforward_connections["L0_neural"] = 3.0  # appears 3 times

# Concept → Tokens (weight by normalized PageRank)
concept.feedforward_connections["L0_neural"] = 1.0  # highest PR
concept.feedforward_connections["L0_networks"] = 0.7  # lower PR
```

---

### 20. Add Concept-Level Lateral Connections

**Files:** `cortical/analysis.py`
**Status:** [ ] Pending

**Problem:**
Layer 2 (Concepts) has 0 lateral connections. Concept clusters should connect to each other based on:
- Shared documents (concepts appearing in same docs)
- Semantic overlap (member tokens with semantic relations)
- Co-activation patterns

**Implementation Steps:**
1. Add `compute_concept_connections()` function to `analysis.py`
2. Connect concepts sharing documents (weighted by Jaccard similarity of doc sets)
3. Connect concepts with semantically related member tokens
4. Weight connections by relation type (IsA > PartOf > RelatedTo)
5. Call from `compute_all()` after `build_concept_clusters()`

**Example Output:**
```python
# Concept "neural/networks/learning" connects to:
#   → "knowledge/graph/semantic" (weight: 0.73, reason: shared docs + RelatedTo)
#   → "data/processing/systems" (weight: 0.45, reason: shared docs)
```

---

### 21. Add Bigram Lateral Connections

**Files:** `cortical/analysis.py`, `cortical/processor.py`
**Status:** [ ] Pending

**Problem:**
Layer 1 (Bigrams) has 0 lateral connections. Bigrams should connect when they:
- Share a component term ("neural_networks" ↔ "neural_processing")
- Co-occur in the same documents
- Form chains ("machine_learning" ↔ "learning_algorithms")

**Implementation Steps:**
1. Add `compute_bigram_connections()` function to `analysis.py`
2. Connect bigrams sharing a term (left or right component)
3. Weight by co-occurrence count and component position
4. Add document co-occurrence bonus
5. Call from `compute_all()` after bigram layer is populated

---

## ConceptNet High Priority

### 22. Implement Relation-Weighted PageRank

**Files:** `cortical/analysis.py`
**Status:** [ ] Pending

**Problem:**
Current PageRank treats all `lateral_connections` equally. ConceptNet-style PageRank should weight edges by semantic relation type.

**Current PageRank:**
```python
for target_id, weight in col.lateral_connections.items():
    # All weights treated the same
```

**Enhanced PageRank:**
```python
RELATION_WEIGHTS = {
    'IsA': 1.5,        # Hypernym relationships are strong
    'PartOf': 1.3,     # Meronym relationships
    'HasProperty': 1.2,
    'RelatedTo': 1.0,  # Default co-occurrence
    'Antonym': 0.5,    # Opposing concepts
}
```

**Implementation Steps:**
1. Add `relation_type: str` field to connection edges (or use separate dict)
2. Create `compute_semantic_pagerank()` function
3. Apply relation-type multipliers during propagation
4. Use semantic relations from `extract_corpus_semantics()` to label edges
5. Add `pagerank_method` parameter to `compute_all()`: 'standard' | 'semantic'

---

### 23. Implement Cross-Layer PageRank Propagation

**Files:** `cortical/analysis.py`
**Status:** [ ] Pending

**Problem:**
PageRank only flows within a single layer. Importance should propagate across layers:
- Important tokens boost their bigrams
- Important bigrams boost their concepts
- Important concepts boost their documents (and vice versa)

**Implementation Steps:**
1. Add `compute_hierarchical_pagerank()` function
2. Iterate: compute layer-local PageRank, then propagate to adjacent layers
3. Use `feedforward_connections` and `feedback_connections` for cross-layer flow
4. Apply damping factor at layer boundaries (e.g., 0.7)
5. Converge when cross-layer changes are minimal

**Algorithm:**
```
for iteration in range(max_iterations):
    for layer in [TOKENS, BIGRAMS, CONCEPTS, DOCUMENTS]:
        compute_local_pagerank(layer)
    propagate_up(TOKENS → BIGRAMS → CONCEPTS → DOCUMENTS)
    propagate_down(DOCUMENTS → CONCEPTS → BIGRAMS → TOKENS)
    if converged: break
```

---

### 24. Add Typed Edge Storage

**Files:** `cortical/minicolumn.py`, `cortical/analysis.py`
**Status:** [ ] Pending

**Problem:**
`lateral_connections` only stores `{target_id: weight}`. ConceptNet-style graphs need edge metadata: relation type, confidence, source.

**Current:**
```python
lateral_connections: Dict[str, float] = {}  # {id: weight}
```

**Enhanced:**
```python
@dataclass
class Edge:
    target_id: str
    weight: float
    relation_type: str = 'co_occurrence'
    confidence: float = 1.0
    source: str = 'corpus'  # 'corpus', 'semantic', 'inferred'

typed_connections: Dict[str, Edge] = {}
```

**Implementation Steps:**
1. Create `Edge` dataclass in `minicolumn.py`
2. Add `typed_connections` field alongside `lateral_connections` (backward compat)
3. Update connection-building code to populate edge metadata
4. Update persistence to save/load typed connections
5. Migrate algorithms to use typed connections when available

---

## ConceptNet Medium Priority

### 25. Implement Multi-Hop Semantic Inference

**Files:** `cortical/query.py`, `cortical/semantics.py`
**Status:** [ ] Pending

**Problem:**
Query expansion only follows single-hop connections. ConceptNet enables multi-hop inference:
- "dog" → IsA → "animal" → HasProperty → "living"
- "car" → PartOf → "engine" → UsedFor → "transportation"

**Implementation Steps:**
1. Add `expand_query_multihop()` function to `query.py`
2. Follow relation chains up to `max_hops` (default: 2)
3. Decay weight by hop distance: `weight *= 0.5 ** hop`
4. Filter by relation path validity (IsA chains are good, random walks less so)
5. Use for enhanced document retrieval

**Example:**
```python
expand_query_multihop("neural", max_hops=2)
# Hop 1: networks (co-occur), learning (co-occur), brain (RelatedTo)
# Hop 2: deep (via learning), cortex (via brain), AI (via networks)
```

---

### 26. Add Relation Path Scoring

**Files:** `cortical/semantics.py`
**Status:** [ ] Pending

**Problem:**
Not all relation paths are equally valid for inference. Need to score paths by semantic coherence.

**Valid Paths:**
- IsA → IsA (transitive hypernymy): "poodle" → "dog" → "animal" ✓
- PartOf → HasA (part inheritance): "wheel" → "car" → "engine" ✓
- RelatedTo → RelatedTo (association): loose but acceptable

**Invalid Paths:**
- Antonym → IsA: contradictory
- Random oscillation: low confidence

**Implementation Steps:**
1. Create `VALID_RELATION_CHAINS` matrix defining allowed transitions
2. Add `score_relation_path()` function
3. Penalize invalid transitions, reward coherent chains
4. Use in multi-hop expansion to filter low-quality paths

---

### 27. Implement Concept Inheritance

**Files:** `cortical/analysis.py`, `cortical/semantics.py`
**Status:** [ ] Pending

**Problem:**
IsA relations should enable property inheritance. If "dog IsA animal" and "animal HasProperty living", then "dog" should inherit "living".

**Implementation Steps:**
1. Build IsA hierarchy from semantic relations
2. Add `inherit_properties()` function
3. Propagate properties down IsA chains with decay
4. Store inherited properties separately from direct properties
5. Use inherited properties in similarity calculations

---

## ConceptNet Low Priority

### 28. Add Commonsense Relation Extraction

**Files:** `cortical/semantics.py`
**Status:** [ ] Pending

**Problem:**
Current relation extraction is limited to co-occurrence patterns. Could extract richer relations:
- "X is a type of Y" → IsA
- "X contains Y" → HasA
- "X is used for Y" → UsedFor
- "X causes Y" → Causes

**Implementation Steps:**
1. Add pattern-based relation extraction
2. Create regex/rule patterns for common relation expressions
3. Extract during `extract_corpus_semantics()`
4. Store relation type with confidence score
5. Weight by pattern specificity

---

### 29. Visualize ConceptNet-Style Graph

**Files:** `cortical/persistence.py`
**Status:** [ ] Pending

**Problem:**
Current `export_graph_json()` doesn't distinguish edge types or layers. Need ConceptNet-style visualization export.

**Implementation Steps:**
1. Add `export_conceptnet_json()` function
2. Include edge relation types and confidence
3. Color-code by layer (tokens=blue, bigrams=green, concepts=orange, docs=red)
4. Export in format compatible with graph visualization tools (D3.js, Cytoscape)
5. Include cross-layer edges

---

### 30. Add Analogy Completion

**Files:** `cortical/query.py`
**Status:** [ ] Pending

**Problem:**
ConceptNet enables analogy completion: "king is to queen as man is to ?" → "woman"
This requires relation-aware vector arithmetic.

**Implementation Steps:**
1. Add `complete_analogy(a, b, c)` function
2. Find relation between a→b
3. Apply same relation from c to find d
4. Use graph embeddings + relation type matching
5. Return top candidates with confidence

**Example:**
```python
complete_analogy("neural", "networks", "knowledge")
# → "graphs" (both form compound technical terms)
```

---

## Summary

| Priority | Task | Status | Category |
|----------|------|--------|----------|
| Critical | Fix TF-IDF per-doc calculation | ✅ Completed | Bug Fix |
| High | Add ID lookup optimization | ✅ Completed | Bug Fix |
| Medium | Fix type annotations (semantics.py) | ✅ Completed | Bug Fix |
| Medium | Remove unused import | ✅ Completed | Bug Fix |
| Medium | Add verbose parameter | ✅ Completed | Bug Fix |
| Low | Add test coverage | ✅ Completed | Bug Fix |
| **Critical** | **Implement chunk-level retrieval** | ✅ Completed | **RAG** |
| **Critical** | **Add document metadata support** | ✅ Completed | **RAG** |
| **High** | **Activate Layer 2 concepts** | ✅ Completed | **RAG** |
| **High** | **Integrate semantic relations** | ✅ Completed | **RAG** |
| **High** | **Persist full computed state** | ✅ Completed | **RAG** |
| Medium | Fix type annotation (embeddings.py) | ✅ Completed | Bug Fix |
| Medium | Optimize spectral embeddings | ✅ Completed | Performance |
| Medium | Add incremental indexing | ✅ Completed | RAG |
| Low | Document magic numbers | ⏳ Deferred | Documentation |
| Low | Multi-stage ranking pipeline | ✅ Completed | RAG |
| Low | Batch query API | ✅ Completed | RAG |
| **Critical** | **Build cross-layer feedforward connections** | ✅ Completed | **ConceptNet** |
| **Critical** | **Add concept-level lateral connections** | ⏳ Pending | **ConceptNet** |
| **Critical** | **Add bigram lateral connections** | ⏳ Pending | **ConceptNet** |
| **High** | **Implement relation-weighted PageRank** | ⏳ Pending | **ConceptNet** |
| **High** | **Implement cross-layer PageRank propagation** | ⏳ Pending | **ConceptNet** |
| **High** | **Add typed edge storage** | ⏳ Pending | **ConceptNet** |
| Medium | Implement multi-hop semantic inference | ⏳ Pending | ConceptNet |
| Medium | Add relation path scoring | ⏳ Pending | ConceptNet |
| Medium | Implement concept inheritance | ⏳ Pending | ConceptNet |
| Low | Add commonsense relation extraction | ⏳ Pending | ConceptNet |
| Low | Visualize ConceptNet-style graph | ⏳ Pending | ConceptNet |
| Low | Add analogy completion | ⏳ Pending | ConceptNet |

**Bug Fix Completion:** 7/7 tasks (100%)
**RAG Enhancement Completion:** 8/8 tasks (100%)
**ConceptNet Enhancement Completion:** 1/12 tasks (8%)

---

## Test Results

```
Ran 185 tests in 0.147s
OK
```

All tests passing as of 2025-12-09.

---

*Updated from code review on 2025-12-09*
