# Task List: Bug Fixes & RAG Enhancements

This document tracks bug fixes and feature enhancements for the Cortical Text Processor.

**Last Updated:** 2025-12-11
**Status:** Bug fixes complete | Developer experience enhancements planned

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
**Status:** [x] Completed

**Solution Applied:**
Added documented constants at module level with detailed explanations:
- `ISOLATION_THRESHOLD = 0.02` - Documents below this avg cosine similarity are isolated
- `WELL_CONNECTED_THRESHOLD = 0.03` - Documents above this are well-integrated
- `WEAK_TOPIC_TFIDF_THRESHOLD = 0.005` - Terms above this TF-IDF are significant topics
- `BRIDGE_SIMILARITY_MIN = 0.005` and `BRIDGE_SIMILARITY_MAX = 0.03` - Range for bridging candidates

Each constant includes documentation of typical ranges and usage guidance.

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

**Files:** `cortical/analysis.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Layer 2 (Concepts) had 0 lateral connections. Concept clusters should connect to each other based on shared documents and semantic overlap.

**Solution Applied:**
1. Added `compute_concept_connections()` function to `analysis.py`
2. Connects concepts by Jaccard similarity of document sets
3. Optionally boosts weights using semantic relations between member tokens
4. Relation type weighting: IsA (1.5) > PartOf (1.3) > HasProperty (1.2) > RelatedTo (1.0)
5. Called from `compute_all()` after `build_concept_clusters()`
6. Added `compute_concept_connections()` method to processor with parameters

**Files Modified:**
- `cortical/analysis.py` - Added `compute_concept_connections()` (~110 lines)
- `cortical/processor.py` - Added processor wrapper method, integrated into `compute_all()`
- `tests/test_processor.py` - Added 8 tests for concept connections

**Usage:**
```python
# Automatic in compute_all()
processor.compute_all()  # Calls compute_concept_connections() automatically

# Manual with options
stats = processor.compute_concept_connections(
    use_semantics=True,    # Boost weights with semantic relations
    min_shared_docs=1,     # Minimum shared documents
    min_jaccard=0.1        # Minimum Jaccard similarity
)
```

---

### 21. Add Bigram Lateral Connections

**Files:** `cortical/analysis.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Layer 1 (Bigrams) has 0 lateral connections. Bigrams should connect when they:
- Share a component term ("neural_networks" ↔ "neural_processing")
- Co-occur in the same documents
- Form chains ("machine_learning" ↔ "learning_algorithms")

**Solution Applied:**
1. Added `compute_bigram_connections()` function to `analysis.py` (~140 lines)
2. Connects bigrams sharing left component (e.g., "neural_networks" ↔ "neural_processing")
3. Connects bigrams sharing right component (e.g., "deep_learning" ↔ "machine_learning")
4. Connects chain bigrams where right of one = left of other ("machine_learning" ↔ "learning_algorithms")
5. Adds document co-occurrence connections weighted by Jaccard similarity
6. Added configurable weights: `component_weight=0.5`, `chain_weight=0.7`, `cooccurrence_weight=0.3`
7. Added `compute_bigram_connections()` method to processor with full docstring
8. Integrated into `compute_all()` pipeline
9. Added `COMP_BIGRAM_CONNECTIONS` staleness tracking constant
10. Updated `recompute()` method to handle bigram connections

**Files Modified:**
- `cortical/analysis.py` - Added `compute_bigram_connections()` (~140 lines)
- `cortical/processor.py` - Added wrapper method and integrated into `compute_all()`
- `tests/test_processor.py` - Added 11 tests for bigram connections

**Usage:**
```python
# Automatic in compute_all()
processor.compute_all()  # Calls compute_bigram_connections() automatically

# Manual with options
stats = processor.compute_bigram_connections(
    component_weight=0.5,  # Weight for shared component connections
    chain_weight=0.7,      # Weight for chain connections
    cooccurrence_weight=0.3,  # Weight for document co-occurrence
    verbose=True
)
print(f"Created {stats['connections_created']} bigram connections")
print(f"  Component: {stats['component_connections']}")
print(f"  Chain: {stats['chain_connections']}")
print(f"  Co-occurrence: {stats['cooccurrence_connections']}")
```

---

## ConceptNet High Priority

### 22. Implement Relation-Weighted PageRank

**Files:** `cortical/analysis.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Current PageRank treats all `lateral_connections` equally. ConceptNet-style PageRank should weight edges by semantic relation type.

**Solution Applied:**
1. Added `RELATION_WEIGHTS` constant to `analysis.py` with default weights:
   - IsA: 1.5, PartOf: 1.3, HasProperty: 1.2, SimilarTo: 1.4, RelatedTo: 1.0
   - Causes: 1.1, UsedFor: 1.0, CoOccurs: 0.8, Antonym: 0.3, DerivedFrom: 1.2
2. Created `compute_semantic_pagerank()` function (~120 lines):
   - Builds semantic relation lookup from (term1, term2) pairs
   - Applies relation-type multipliers to edge weights
   - Returns stats: pagerank scores, iterations_run, edges_with_relations
3. Added `compute_semantic_importance()` method to processor:
   - Falls back to standard PageRank if no semantic relations
   - Applies semantic PageRank to both token and bigram layers
   - Returns comprehensive statistics
4. Updated `compute_all()` with `pagerank_method` parameter:
   - 'standard': Traditional PageRank (default)
   - 'semantic': ConceptNet-style with relation weighting
   - Automatically extracts semantic relations if needed

**Files Modified:**
- `cortical/analysis.py` - Added `RELATION_WEIGHTS` and `compute_semantic_pagerank()` (~130 lines)
- `cortical/processor.py` - Added `compute_semantic_importance()`, updated `compute_all()`
- `tests/test_processor.py` - Added 9 tests for semantic PageRank

**Usage:**
```python
# Use semantic PageRank via compute_all
processor.compute_all(pagerank_method='semantic')

# Or call directly
processor.extract_corpus_semantics()
stats = processor.compute_semantic_importance()
print(f"Found {stats['total_edges_with_relations']} semantic edges")

# Custom relation weights
custom_weights = {'IsA': 2.0, 'CoOccurs': 0.5}
processor.compute_semantic_importance(relation_weights=custom_weights)
```

---

### 23. Implement Cross-Layer PageRank Propagation

**Files:** `cortical/analysis.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
PageRank only flows within a single layer. Importance should propagate across layers:
- Important tokens boost their bigrams
- Important bigrams boost their concepts
- Important concepts boost their documents (and vice versa)

**Solution Applied:**
1. Added `compute_hierarchical_pagerank()` function to `analysis.py` (~150 lines):
   - Computes local PageRank within each layer
   - Propagates scores up via feedback_connections (tokens → bigrams → concepts → documents)
   - Propagates scores down via feedforward_connections (documents → concepts → bigrams → tokens)
   - Normalizes PageRank within each layer after propagation
   - Converges when cross-layer changes are minimal
2. Added `compute_hierarchical_importance()` method to processor
3. Updated `compute_all()` with `pagerank_method='hierarchical'` option
4. Returns detailed statistics: iterations_run, converged, per-layer stats

**Files Modified:**
- `cortical/analysis.py` - Added `compute_hierarchical_pagerank()` (~150 lines)
- `cortical/processor.py` - Added `compute_hierarchical_importance()`, updated `compute_all()`
- `tests/test_processor.py` - Added 9 tests for hierarchical PageRank

**Usage:**
```python
# Use hierarchical PageRank via compute_all
processor.compute_all(pagerank_method='hierarchical')

# Or call directly with custom parameters
stats = processor.compute_hierarchical_importance(
    layer_iterations=10,      # Iterations for intra-layer PageRank
    global_iterations=5,      # Iterations for cross-layer propagation
    cross_layer_damping=0.7   # Damping at layer boundaries
)
print(f"Converged: {stats['converged']} in {stats['iterations_run']} iterations")
for layer, info in stats['layer_stats'].items():
    print(f"  {layer}: {info['nodes']} nodes, max PR={info['max_pagerank']:.4f}")
```

---

### 24. Add Typed Edge Storage

**Files:** `cortical/minicolumn.py`, `cortical/__init__.py`
**Status:** [x] Completed

**Problem:**
`lateral_connections` only stores `{target_id: weight}`. ConceptNet-style graphs need edge metadata: relation type, confidence, source.

**Solution Applied:**
1. Created `Edge` dataclass in `minicolumn.py` with:
   - `target_id`: Target minicolumn ID
   - `weight`: Connection strength (accumulates)
   - `relation_type`: Semantic type ('co_occurrence', 'IsA', 'PartOf', etc.)
   - `confidence`: Confidence score (0.0 to 1.0)
   - `source`: Origin ('corpus', 'semantic', 'inferred')
2. Added `typed_connections: Dict[str, Edge]` field to Minicolumn
3. Implemented `add_typed_connection()` with intelligent merging:
   - Weights accumulate
   - Specific relation types override 'co_occurrence'
   - Higher confidence is kept
   - Source priority: inferred > semantic > corpus
4. Added query methods:
   - `get_typed_connection(target_id)` - Get single edge
   - `get_connections_by_type(relation_type)` - Filter by relation
   - `get_connections_by_source(source)` - Filter by source
5. Updated `to_dict()` and `from_dict()` for persistence
6. Exported `Edge` class from package

**Files Modified:**
- `cortical/minicolumn.py` - Added Edge dataclass and typed_connections
- `cortical/__init__.py` - Export Edge class
- `tests/test_layers.py` - Added 15 tests for Edge and typed connections

**Usage:**
```python
from cortical import Minicolumn, Edge

col = Minicolumn("L0_test", "test", 0)

# Add typed connections
col.add_typed_connection("L0_network", 0.8, relation_type='RelatedTo')
col.add_typed_connection("L0_brain", 0.5, relation_type='IsA', source='semantic')

# Query by type
is_a_edges = col.get_connections_by_type('IsA')
semantic_edges = col.get_connections_by_source('semantic')

# Get single edge
edge = col.get_typed_connection("L0_network")
print(f"{edge.relation_type}: {edge.weight} ({edge.confidence})")
```

---

## ConceptNet Medium Priority

### 25. Implement Multi-Hop Semantic Inference

**Files:** `cortical/query.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Query expansion only follows single-hop connections. ConceptNet enables multi-hop inference:
- "dog" → IsA → "animal" → HasProperty → "living"
- "car" → PartOf → "engine" → UsedFor → "transportation"

**Solution Applied:**
1. Added `VALID_RELATION_CHAINS` constant to `query.py` defining valid relation chain patterns with validity scores:
   - Transitive hierarchies: IsA→IsA (1.0), PartOf→PartOf (1.0), IsA→HasProperty (0.9)
   - Association chains: RelatedTo→RelatedTo (0.6), SimilarTo→SimilarTo (0.7)
   - Causal chains: Causes→Causes (0.8), Causes→HasProperty (0.7)
   - Invalid chains: Antonym→IsA (0.1) - contradictory
2. Added `score_relation_path()` function to compute path validity scores
3. Added `expand_query_multihop()` function (~90 lines) implementing:
   - BFS-style expansion with hop tracking
   - Weight decay by hop distance: `weight *= decay_factor ** hop`
   - Path validity filtering with `min_path_score` threshold
   - Configurable parameters: `max_hops`, `max_expansions`, `decay_factor`, `min_path_score`
4. Added `expand_query_multihop()` method to processor with fallback to regular expansion

**Files Modified:**
- `cortical/query.py` - Added `VALID_RELATION_CHAINS`, `score_relation_path()`, `expand_query_multihop()` (~150 lines)
- `cortical/processor.py` - Added processor wrapper method (~45 lines)
- `tests/test_processor.py` - Added 18 tests for multi-hop inference and path scoring

**Usage:**
```python
# Extract semantic relations first
processor.extract_corpus_semantics()

# Multi-hop expansion (finds 2-hop away terms)
expanded = processor.expand_query_multihop("neural", max_hops=2)
# Hop 1: networks (co-occur), learning (co-occur), brain (RelatedTo)
# Hop 2: deep (via learning), cortex (via brain), AI (via networks)

# Custom parameters
expanded = processor.expand_query_multihop(
    "neural",
    max_hops=3,           # Follow up to 3 hops
    decay_factor=0.6,     # Slower weight decay
    min_path_score=0.3,   # Filter low-validity paths
    max_expansions=20     # More expansion terms
)
```

---

### 26. Add Relation Path Scoring

**Files:** `cortical/query.py`
**Status:** [x] Completed (implemented with Task 25)

**Problem:**
Not all relation paths are equally valid for inference. Need to score paths by semantic coherence.

**Valid Paths:**
- IsA → IsA (transitive hypernymy): "poodle" → "dog" → "animal" ✓
- PartOf → HasA (part inheritance): "wheel" → "car" → "engine" ✓
- RelatedTo → RelatedTo (association): loose but acceptable

**Invalid Paths:**
- Antonym → IsA: contradictory
- Random oscillation: low confidence

**Solution Applied (with Task 25):**
1. Added `VALID_RELATION_CHAINS` dict defining allowed transitions with validity scores:
   - Transitive: (IsA, IsA)=1.0, (PartOf, PartOf)=1.0
   - Property inheritance: (IsA, HasProperty)=0.9, (PartOf, HasProperty)=0.8
   - Association: (RelatedTo, RelatedTo)=0.6, (SimilarTo, SimilarTo)=0.7
   - Invalid: (Antonym, IsA)=0.1
2. Added `score_relation_path()` function that multiplies consecutive pair validities
3. Default validity score of 0.4 for unknown relation pairs
4. Integrated into `expand_query_multihop()` with `min_path_score` parameter

**Files Modified:**
- `cortical/query.py` - Added constants and scoring function (~50 lines)
- `tests/test_processor.py` - Added 7 tests in `TestMultiHopPathScoring` class

**Usage:**
```python
from cortical.query import score_relation_path, VALID_RELATION_CHAINS

# Score a relation path
score = score_relation_path(['IsA', 'IsA'])  # 1.0 (transitive)
score = score_relation_path(['IsA', 'HasProperty'])  # 0.9 (property inheritance)
score = score_relation_path(['Antonym', 'IsA'])  # 0.1 (contradictory)

# Check valid chain patterns
print(VALID_RELATION_CHAINS[('IsA', 'IsA')])  # 1.0
```

---

### 27. Implement Concept Inheritance

**Files:** `cortical/semantics.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
IsA relations should enable property inheritance. If "dog IsA animal" and "animal HasProperty living", then "dog" should inherit "living".

**Solution Applied:**
1. Added `build_isa_hierarchy()` function to extract parent-child relationships from IsA relations
2. Added `get_ancestors()` and `get_descendants()` functions for hierarchy traversal with depth tracking
3. Added `inherit_properties()` function that:
   - Extracts direct properties from HasProperty, HasA, CapableOf, AtLocation, UsedFor relations
   - Propagates properties down IsA chains with configurable decay factor
   - Returns mapping of term → {property: (weight, source_ancestor, depth)}
4. Added `compute_property_similarity()` for weighted Jaccard similarity based on shared properties
5. Added `apply_inheritance_to_connections()` to boost lateral connections for shared inherited properties
6. Added processor wrapper methods: `compute_property_inheritance()` and `compute_property_similarity()`

**Files Modified:**
- `cortical/semantics.py` - Added 6 new functions (~280 lines)
- `cortical/processor.py` - Added 2 processor wrapper methods (~80 lines)
- `tests/test_semantics.py` - Added 23 tests across 7 new test classes

**Usage:**
```python
# Compute property inheritance
processor.extract_corpus_semantics()
stats = processor.compute_property_inheritance(
    decay_factor=0.7,      # Weight decay per level
    max_depth=5,           # Maximum inheritance depth
    apply_to_connections=True,  # Boost lateral connections
    boost_factor=0.3       # Boost weight for shared properties
)

# Check inherited properties for a term
inherited = stats['inherited']
if 'dog' in inherited:
    for prop, (weight, source, depth) in inherited['dog'].items():
        print(f"  {prop}: {weight:.2f} (from {source}, depth {depth})")

# Compute similarity based on shared properties
sim = processor.compute_property_similarity("dog", "cat")
```

---

## ConceptNet Low Priority

### 28. Add Commonsense Relation Extraction

**Files:** `cortical/semantics.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Current relation extraction is limited to co-occurrence patterns. Could extract richer relations:
- "X is a type of Y" → IsA
- "X contains Y" → HasA
- "X is used for Y" → UsedFor
- "X causes Y" → Causes

**Solution Applied:**
1. Added `RELATION_PATTERNS` constant with 30+ regex patterns covering:
   - IsA patterns: "X is a type of Y", "X is a kind of Y", "X belongs to Y"
   - HasA patterns: "X has Y", "X contains Y", "X consists of Y"
   - PartOf patterns: "X is part of Y", "X is a component of Y"
   - UsedFor patterns: "X is used for Y", "X helps Y", "X enables Y"
   - Causes patterns: "X causes Y", "X leads to Y", "X produces Y"
   - CapableOf patterns: "X can Y", "X is able to Y"
   - AtLocation patterns: "X is found in Y", "X lives in Y"
   - HasProperty patterns: "X is Y" (with context)
   - Antonym patterns: "X is opposite of Y"
   - DerivedFrom patterns: "X comes from Y"
   - DefinedBy patterns: "X means Y"
2. Added `extract_pattern_relations()` function with filtering for:
   - Invalid terms (not in corpus)
   - Stopwords
   - Self-relations
   - Duplicate relations
3. Added `get_pattern_statistics()` for relation type analysis
4. Updated `extract_corpus_semantics()` with `use_pattern_extraction` parameter
5. Added processor method `extract_pattern_relations()` for direct access

**Files Modified:**
- `cortical/semantics.py` - Added `RELATION_PATTERNS`, `extract_pattern_relations()`, `get_pattern_statistics()` (~180 lines)
- `cortical/processor.py` - Updated `extract_corpus_semantics()`, added `extract_pattern_relations()` (~70 lines)
- `tests/test_semantics.py` - Added 16 tests for pattern extraction

**Usage:**
```python
# Automatic pattern extraction during semantic extraction
processor.extract_corpus_semantics(
    use_pattern_extraction=True,    # Enabled by default
    min_pattern_confidence=0.6      # Minimum confidence threshold
)

# Direct pattern extraction
relations = processor.extract_pattern_relations(min_confidence=0.5)
for t1, rel_type, t2, confidence in relations:
    print(f"{t1} --{rel_type}--> {t2} ({confidence:.2f})")
```

---

### 29. Visualize ConceptNet-Style Graph

**Files:** `cortical/persistence.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
Current `export_graph_json()` doesn't distinguish edge types or layers. Need ConceptNet-style visualization export.

**Solution Applied:**
1. Added `LAYER_COLORS` constant with color codes for each layer:
   - Tokens: Royal Blue (#4169E1)
   - Bigrams: Forest Green (#228B22)
   - Concepts: Dark Orange (#FF8C00)
   - Documents: Crimson (#DC143C)
2. Added `LAYER_NAMES` constant for display names
3. Added `export_conceptnet_json()` function (~200 lines) with:
   - Color-coded nodes by layer with layer_name
   - Typed edges with relation_type, confidence, and source_type
   - Cross-layer edges (feedforward/feedback)
   - Relation-based edge colors
   - D3.js/Cytoscape/Gephi-compatible format
4. Added `_get_relation_color()` helper with 16 relation type colors
5. Added `_count_edge_types()` and `_count_relation_types()` helpers
6. Added processor wrapper method `export_conceptnet_json()`

**Files Modified:**
- `cortical/persistence.py` - Added constants and export function (~270 lines)
- `cortical/processor.py` - Added processor wrapper method (~50 lines)
- `tests/test_persistence.py` - Added 13 tests for ConceptNet export

**Usage:**
```python
# Export ConceptNet-style graph
processor.extract_corpus_semantics(verbose=False)
graph = processor.export_conceptnet_json(
    "graph.json",
    include_cross_layer=True,     # Include feedforward/feedback edges
    include_typed_edges=True,     # Include typed_connections
    min_weight=0.0,               # Minimum edge weight
    max_nodes_per_layer=100       # Limit nodes per layer
)

# Open graph.json in D3.js, Cytoscape.js, or Gephi for visualization
```

---

### 30. Add Analogy Completion

**Files:** `cortical/query.py`, `cortical/processor.py`
**Status:** [x] Completed

**Problem:**
ConceptNet enables analogy completion: "king is to queen as man is to ?" → "woman"
This requires relation-aware vector arithmetic.

**Solution Applied:**
1. Added `find_relation_between()` helper function to find semantic relations between two terms
2. Added `find_terms_with_relation()` helper to find terms connected by a specific relation type
3. Added `complete_analogy()` function (~120 lines) with three strategies:
   - **Relation matching**: Find a→b relation, apply to c
   - **Vector arithmetic**: Use embeddings for a - b + c ≈ d
   - **Pattern matching**: Use co-occurrence patterns as fallback
4. Added `complete_analogy_simple()` lightweight version using only co-occurrence patterns
5. Added processor wrapper methods: `complete_analogy()` and `complete_analogy_simple()`

**Files Modified:**
- `cortical/query.py` - Added helper functions and analogy completion (~180 lines)
- `cortical/processor.py` - Added processor wrapper methods (~60 lines)
- `tests/test_processor.py` - Added 14 tests for analogy completion

**Usage:**
```python
# Full analogy completion with multiple strategies
results = processor.complete_analogy("king", "queen", "man", top_n=5)
for term, score, method in results:
    print(f"  {term}: {score:.3f} ({method})")

# Simple version (co-occurrence only)
results = processor.complete_analogy_simple("neural", "networks", "knowledge")
for term, score in results:
    print(f"  {term}: {score:.3f}")

# Control which strategies to use
results = processor.complete_analogy(
    "neural", "networks", "knowledge",
    use_embeddings=True,   # Enable vector arithmetic
    use_relations=True     # Enable relation matching
)
```

---

## Actionable Tasks (2025-12-10)

The following tasks were identified during comprehensive code review and are prioritized for implementation:

---

### 47. Dog-Food the System During Development

**Files:** New `scripts/index_codebase.py`, `scripts/search_codebase.py`, `.claude/skills/`
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Goal:**
Use the Cortical Text Processor to index and search its own codebase during development.

**Solution Applied:**
1. Created `scripts/index_codebase.py`:
   - Indexes all 19 Python files in `cortical/` and `tests/`
   - Indexes 4 documentation files (CLAUDE.md, TASK_LIST.md, README.md, KNOWLEDGE_TRANSFER.md)
   - Saves indexed corpus to `corpus_dev.pkl` (23 documents, ~15,600 lines)
   - Computes semantic PageRank, TF-IDF, concepts, and semantic relations

2. Created `scripts/search_codebase.py`:
   - Loads indexed corpus and performs semantic search
   - Returns file:line references for each result
   - Supports `--top N`, `--verbose`, `--expand`, `--interactive` options
   - Interactive mode with `/expand`, `/concepts`, `/stats`, `/quit` commands

3. Created Claude Skills in `.claude/skills/`:
   - `codebase-search/SKILL.md` - Search skill for finding code patterns
   - `corpus-indexer/SKILL.md` - Indexing skill for updating corpus

4. Updated `CLAUDE.md` with Dog-Fooding section documenting usage

**Example Usage:**
```bash
python scripts/index_codebase.py
python scripts/search_codebase.py "PageRank algorithm" --top 3
python scripts/search_codebase.py "bigram separator" --expand
python scripts/search_codebase.py --interactive
```

**Success Criteria:** All met
- Can find relevant code when searching for concepts
- Passages include accurate file:line references (e.g., `cortical/analysis.py:127`)
- System handles its own codebase without errors
- Identified usability issue: return value order in find_passages_for_query (fixed)

---

### 37. Create Dedicated Query Module Tests

**File:** `tests/test_query.py` (new file)
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Problem:**
`cortical/query.py` (1,503 lines, 20+ functions) has NO dedicated test file. Functions are tested only indirectly through `test_processor.py`.

**Functions Needing Coverage:**
- `expand_query_multihop()` - Multi-hop inference with relation chains
- `score_relation_path()` - Relation path validation
- `get_expanded_query_terms()` - Helper for all expansion methods
- `find_relevant_concepts()` - Concept filtering for RAG
- `find_relation_between()` and `find_terms_with_relation()` - Relation discovery
- Chunking and batch operations

**Deliverable:** Create `tests/test_query.py` with 30+ unit tests.

**Solution Applied:**
Created `tests/test_query.py` with 48 comprehensive tests covering:
- `TestScoreRelationPath` (4 tests) - Relation path validation
- `TestCreateChunks` (4 tests) - Text chunking
- `TestFindRelationBetween` (4 tests) - Relation discovery
- `TestFindTermsWithRelation` (4 tests) - Term relation lookup
- `TestExpandQuery` (4 tests) - Basic query expansion
- `TestExpandQueryMultihop` (4 tests) - Multi-hop expansion
- `TestGetExpandedQueryTerms` (3 tests) - Unified expansion helper
- `TestFindDocumentsForQuery` (4 tests) - Document retrieval
- `TestFindDocumentsBatch` (3 tests) - Batch document retrieval
- `TestFindPassagesForQuery` (2 tests) - Passage retrieval
- `TestFindRelevantConcepts` (2 tests) - Concept filtering
- `TestCompleteAnalogy` (3 tests) - Analogy completion
- `TestQueryWithSpreadingActivation` (2 tests) - Activation search
- `TestScoreChunk` (3 tests) - Chunk scoring
- `TestEdgeCases` (2 tests) - Edge case handling

Test count increased from 340 to 388.

---

### 38. Add Input Validation to Public API

**Files:** `cortical/processor.py`
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Problem:**
Public API methods silently accept invalid inputs, leading to confusing behavior.

**Solution Applied:**
Added input validation to 4 key public API methods:

1. **`process_document()`** - Validates doc_id (non-empty string) and content (non-empty string)
2. **`find_documents_for_query()`** - Validates query_text (non-empty string) and top_n (positive int)
3. **`complete_analogy()`** - Validates all 3 terms (non-empty strings) and top_n (positive int)
4. **`add_documents_batch()`** - Validates documents list format, doc_id/content types, and recompute level

All methods now raise `ValueError` with descriptive messages for invalid input.

**Tests Added:** 20 new tests in `TestInputValidation` class covering:
- Empty/None/non-string doc_id
- Empty/whitespace-only/non-string content
- Empty/whitespace-only query_text
- Invalid top_n values (0, negative)
- Invalid document batch formats
- Valid input acceptance

Test count increased from 388 to 408.

---

### 39. Move Inline Imports to Module Top

**Files:** `cortical/processor.py:161`, `cortical/semantics.py:493`
**Status:** [x] Completed (2025-12-10)
**Priority:** Low

**Problem:**
`import copy` statements inside methods pollute namespaces and impact readability.

**Solution Applied:**
Moved `import copy` to module-level imports in both files.

---

### 40. Add Parameter Range Validation

**Files:** Multiple
**Status:** [x] Completed (2025-12-10)
**Priority:** Medium

**Problem:**
No validation for invalid parameter ranges.

**Solution Applied:**
Added validation to key functions:
- `compute_pagerank()`: damping must be in range (0, 1)
- `compute_semantic_pagerank()`: damping must be in range (0, 1)
- `compute_hierarchical_pagerank()`: damping and cross_layer_damping must be in range (0, 1)
- `retrofit_connections()`: alpha must be in range [0, 1]
- `retrofit_embeddings()`: alpha must be in range (0, 1]
- `create_chunks()`: chunk_size > 0, overlap >= 0, overlap < chunk_size

Added 9 new tests for parameter validation.

---

### 41. Create Configuration Dataclass

**Files:** New `cortical/config.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** Medium

**Problem:**
Magic numbers scattered across modules with no central configuration:
- `gaps.py`: ISOLATION_THRESHOLD=0.02, WELL_CONNECTED_THRESHOLD=0.03
- `query.py`: VALID_RELATION_CHAINS (15 entries)
- `analysis.py`: damping=0.85, iterations=20, tolerance=1e-6

**Solution Applied:**
1. Created `cortical/config.py` with `CorticalConfig` dataclass
2. Centralized all magic numbers and defaults:
   - PageRank: damping, iterations, tolerance
   - Clustering: min_cluster_size, cluster_strictness
   - Gap detection: isolation_threshold, well_connected_threshold, etc.
   - Chunking: chunk_size, chunk_overlap
   - Query expansion: max_query_expansions, semantic_expansion_discount
   - Bigram connections: component_weight, chain_weight, cooccurrence_weight
   - Concept connections: min_shared_docs, min_jaccard, embedding_threshold
   - Multi-hop: max_hops, decay_factor, min_path_score
   - Property inheritance: decay_factor, max_depth, boost_factor
   - Relation weights dictionary
3. Added validation in `__post_init__()` for all parameters
4. Added `copy()`, `to_dict()`, and `from_dict()` methods
5. Moved `VALID_RELATION_CHAINS` to config module
6. Updated `__init__.py` to export new classes
7. Created 29 tests in `tests/test_config.py`

**Usage:**
```python
from cortical import CorticalTextProcessor, CorticalConfig

# Custom configuration
config = CorticalConfig(
    pagerank_damping=0.9,
    min_cluster_size=5,
    isolation_threshold=0.03
)
processor = CorticalTextProcessor(config=config)
```

---

### 42. Add Simple Query Language Support

**File:** `cortical/query.py`
**Status:** [ ] Not Started
**Priority:** Low

**Problem:**
Only natural language queries supported. No structured filtering.

**Solution:** Add minimal syntax:
- `"term1 AND term2"` - require both terms
- `"term1 OR term2"` - either term
- `"-term1"` - exclude term
- `"term1"` (quoted) - exact match

---

### 43. Optimize Chunk Scoring Performance

**File:** `cortical/query.py:590-630`
**Status:** [x] Completed (2025-12-10)
**Priority:** Medium

**Problem:**
`score_chunk()` tokenizes chunk text every call with no caching.

**Solution Applied:**
1. Added `precompute_term_cols()` to cache minicolumn lookups for query terms
2. Added `score_chunk_fast()` for optimized scoring with pre-computed lookups
3. Updated `find_passages_for_query()` to use fast scoring
4. Updated `find_passages_batch()` to use fast scoring

Added 4 new tests for optimization functions.

---

### 44. Remove Deprecated feedforward_sources

**Files:** `cortical/minicolumn.py:117`, `analysis.py:457`, `query.py:105`
**Status:** [ ] Not Started
**Priority:** Low

**Problem:**
`feedforward_sources` is marked deprecated but still used in 4+ locations.

**Solution:** Migrate all usages to `feedforward_connections` and remove deprecated attribute.

---

### 45. Add LRU Cache for Query Results

**File:** `cortical/processor.py`
**Status:** [x] Completed (2025-12-10)
**Priority:** Medium

**Problem:**
Every query re-expands terms and rescores documents. Repeated queries (common in RAG loops) are slow.

**Solution Applied:**
1. Added `_query_expansion_cache` dict and `_query_cache_max_size` to processor
2. Added `expand_query_cached()` method with cache lookup and LRU-style eviction
3. Added `clear_query_cache()` to manually invalidate cache
4. Added `set_query_cache_size()` to configure cache size
5. Auto-invalidate cache on `compute_all()` since corpus state changes

Added 8 new tests for cache functionality.

---

### 46. Standardize Return Types with Dataclasses

**File:** `cortical/query.py`
**Status:** [ ] Not Started
**Priority:** Low

**Problem:**
Inconsistent return types across query functions:
- `find_documents_for_query()` → `List[Tuple[str, float]]`
- `find_passages_for_query()` → `List[Tuple[str, str, int, int, float]]`
- `complete_analogy()` → `List[Tuple[str, float, str]]`

**Solution:**
```python
@dataclass
class DocumentMatch:
    doc_id: str
    score: float

@dataclass
class PassageMatch:
    doc_id: str
    text: str
    start: int
    end: int
    score: float
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
| Low | Document magic numbers | ✅ Completed | Documentation |
| Low | Multi-stage ranking pipeline | ✅ Completed | RAG |
| Low | Batch query API | ✅ Completed | RAG |
| **Critical** | **Build cross-layer feedforward connections** | ✅ Completed | **ConceptNet** |
| **Critical** | **Add concept-level lateral connections** | ✅ Completed | **ConceptNet** |
| **Critical** | **Add bigram lateral connections** | ✅ Completed | **ConceptNet** |
| **High** | **Implement relation-weighted PageRank** | ✅ Completed | **ConceptNet** |
| **High** | **Implement cross-layer PageRank propagation** | ✅ Completed | **ConceptNet** |
| **High** | **Add typed edge storage** | ✅ Completed | **ConceptNet** |
| Medium | Implement multi-hop semantic inference | ✅ Completed | ConceptNet |
| Medium | Add relation path scoring | ✅ Completed | ConceptNet |
| Medium | Implement concept inheritance | ✅ Completed | ConceptNet |
| Low | Add commonsense relation extraction | ✅ Completed | ConceptNet |
| Low | Visualize ConceptNet-style graph | ✅ Completed | ConceptNet |
| Low | Add analogy completion | ✅ Completed | ConceptNet |

**Bug Fix Completion:** 8/8 tasks (100%)
**RAG Enhancement Completion:** 8/8 tasks (100%)
**ConceptNet Enhancement Completion:** 12/12 tasks (100%)

---

## Test Results

```
Ran 408 tests in 0.336s
OK
```

All tests passing as of 2025-12-10.

---

## Layer 2 Connection Improvements (2025-12-10)

### Problem Statement

Layer 2 (Concept Layer/V4) shows 0 connections when documents cover diverse topics because:
- Label propagation creates topic-specific clusters
- Concepts inherit only their members' documents
- Connection filter requires shared documents (Jaccard ≥ 0.1)
- No document overlap → no connections

### Task L2-1: Add Configurable Connection Thresholds ✅ COMPLETED
**File:** `cortical/analysis.py` (lines 614-812)

- [x] Add `min_shared_docs=0` option to allow connections without document overlap
- [x] Add `min_jaccard=0.0` option to disable Jaccard filtering
- [x] Expose these parameters in `CorticalTextProcessor.compute_concept_connections()`
- [x] Update docstrings to explain threshold behavior
- [x] Add tests for edge cases (zero thresholds, negative values)

### Task L2-2: Connect Concepts via Semantic Relations ✅ COMPLETED
**File:** `cortical/analysis.py`

- [x] Add new connection method that links concepts when their member tokens have semantic relations
- [x] For each concept pair, check if any (token1, relation, token2) exists in semantic_relations
- [x] Weight connections by number of semantic links between members
- [x] Make this work independently of document overlap
- [x] Add `use_member_semantics=True` parameter to `compute_concept_connections()`
- [x] Add tests verifying semantic-based connections

### Task L2-3: Connect Concepts via Shared Vocabulary/Embeddings ✅ COMPLETED
**File:** `cortical/analysis.py`

- [x] Add connection method based on embedding similarity between concept centroids
- [x] Compute concept centroid as average of member token embeddings
- [x] Connect concepts with cosine similarity above threshold
- [x] Add `use_embedding_similarity=True` and `embedding_threshold=0.3` parameters
- [x] Falls back gracefully if embeddings not computed
- [x] Add tests for embedding-based connections

### Task L2-4: Improve Clustering to Reduce Topic Isolation ✅ COMPLETED
**File:** `cortical/analysis.py` (lines 482-616)

- [x] Add `cluster_strictness` parameter to label propagation (0.0-1.0)
- [x] Lower strictness = more cross-topic token mixing in clusters
- [x] Add `bridge_weight` parameter for inter-document token bridging
- [x] Add tests for different strictness levels and bridging

### Task L2-5: Integration and API Updates ✅ COMPLETED
**File:** `cortical/processor.py`

- [x] Update `compute_all()` to accept connection strategy parameters
- [x] Add `connection_strategy` enum: 'document_overlap', 'semantic', 'embedding', 'hybrid'
- [x] 'hybrid' combines all three methods with configurable weights
- [x] Add documentation in CLAUDE.md
- [x] Add 6 new tests for compute_all strategies

**Success Criteria:** ✅ ALL MET
- Layer 2 shows meaningful connections even with diverse document topics
- User can choose connection strategy based on their use case
- All existing tests continue to pass (337 tests)
- New tests cover the added functionality (17 new tests added)

---

## Code Quality Improvements (2025-12-10)

### Query Expansion Helper Refactoring

**File:** `cortical/query.py`
**Status:** [x] Completed

**Problem:**
Query expansion logic (expand + semantic merge) was duplicated in 6 functions:
- `find_documents_for_query()`
- `find_passages_for_query()`
- `find_documents_batch()`
- `find_passages_batch()`
- `multi_stage_rank()`
- `multi_stage_rank_documents()`

**Solution Applied:**
Added `get_expanded_query_terms()` helper function (~60 lines) that consolidates:
- Lateral connection expansion via `expand_query()`
- Semantic relation expansion via `expand_query_semantic()`
- Merging of expansion results with appropriate weighting
- Configurable parameters: `max_expansions`, `semantic_discount`

All 6 functions now use this helper, reducing code duplication by ~100 lines.

---

*Updated from code review on 2025-12-10*

---

## Critical Bug Fixes (2025-12-10)

The following critical bugs were identified during code review and must be fixed:

### 34. Fix Bigram Separator Mismatch in Analogy Completion

**File:** `cortical/query.py`
**Lines:** 1442-1468
**Status:** [x] Completed (2025-12-10)
**Priority:** Critical

**Problem:**
The `complete_analogy_simple()` function uses underscore separators for bigram lookup and parsing, but bigrams are stored with **space** separators (defined in `tokenizer.py:179`).

**Affected Code:**
```python
# Line 1442-1443: WRONG - uses underscore
ab_bigram = f"{term_a}_{term_b}"  # Creates "neural_networks"
ba_bigram = f"{term_b}_{term_a}"

# Line 1452: WRONG - splits by underscore
parts = bigram.split('_')

# But bigrams are stored with spaces (tokenizer.py:179):
# ' '.join(tokens[i:i+n])  # Creates "neural networks"
```

**Impact:**
- The bigram pattern matching strategy in `complete_analogy_simple()` is completely non-functional
- `ab_col` and `ba_col` will always be `None` because "neural_networks" doesn't exist in the corpus
- The `parts` split will never produce valid component extraction

**Solution:**
```python
# Line 1442-1443: Should use space
ab_bigram = f"{term_a} {term_b}"  # Correct: "neural networks"
ba_bigram = f"{term_b} {term_a}"

# Line 1452: Should split by space
parts = bigram.split(' ')
```

**Files to Modify:**
- `cortical/query.py` - Fix separator in lines 1442, 1443, 1452
- `tests/test_processor.py` - Add tests for bigram-based analogy completion

---

### 35. Fix Bigram Separator Mismatch in Bigram Connections

**File:** `cortical/analysis.py`
**Line:** 927
**Status:** [x] Completed (2025-12-10)
**Priority:** Critical

**Problem:**
The `compute_bigram_connections()` function splits bigram content by underscore, but bigrams are stored with **space** separators.

**Affected Code:**
```python
# Line 927: WRONG - splits by underscore
for bigram in bigrams:
    parts = bigram.content.split('_')
    if len(parts) == 2:
        left_index[parts[0]].append(bigram)
        right_index[parts[1]].append(bigram)

# But bigrams are stored with spaces:
# "neural networks" not "neural_networks"
```

**Impact:**
- `left_index` and `right_index` dictionaries are never populated
- Component-sharing connections (e.g., "neural networks" ↔ "neural processing") are never created
- Chain connections (e.g., "machine learning" ↔ "learning algorithms") are never created
- Only document co-occurrence connections work correctly

**Solution:**
```python
# Line 927: Should split by space
parts = bigram.content.split(' ')
```

**Verification:**
After fixing, the `compute_bigram_connections()` stats should show non-zero values for:
- `component_connections`
- `chain_connections`

Currently these are always 0 due to the bug.

**Files to Modify:**
- `cortical/analysis.py` - Fix separator in line 927
- `tests/test_analysis.py` - Add tests verifying component/chain connections work

---

---

## New Task Summary (2025-12-10)

| # | Priority | Task | Status | Category |
|---|----------|------|--------|----------|
| 34 | **Critical** | Fix bigram separator in analogy completion | ✅ Completed | Bug Fix |
| 35 | **Critical** | Fix bigram separator in bigram connections | ✅ Completed | Bug Fix |
| 47 | **High** | Dog-food the system during development | ✅ Completed | Validation |
| 37 | **High** | Create dedicated query module tests | ✅ Completed | Testing |
| 38 | **High** | Add input validation to public API | ✅ Completed | Code Quality |
| 39 | Low | Move inline imports to module top | ✅ Completed | Code Quality |
| 40 | Medium | Add parameter range validation | ✅ Completed | Code Quality |
| 41 | Medium | Create configuration dataclass | [ ] Not Started | Architecture |
| 42 | Low | Add simple query language support | [ ] Not Started | Feature |
| 43 | Medium | Optimize chunk scoring performance | ✅ Completed | Performance |
| 44 | Low | Remove deprecated feedforward_sources | [ ] Not Started | Cleanup |
| 45 | Medium | Add LRU cache for query results | ✅ Completed | Performance |
| 46 | Low | Standardize return types with dataclasses | [ ] Not Started | API |

**Completed:** 9/13 tasks
**High Priority Remaining:** 0 tasks
**Medium Priority Remaining:** 1 task (#41)
**Low Priority Remaining:** 3 tasks (#42, #44, #46)

**Total Tests:** 593 (all passing)

---

### 57. Add Incremental Codebase Indexing

**Files:** `scripts/index_codebase.py`, `cortical/processor.py`, `cortical/layers.py`, `tests/test_incremental_indexing.py`
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Problem:**
The codebase indexer had to rebuild the entire corpus on every run, even for small changes. This was slow and inefficient for iterative development.

**Solution Applied:**
1. Added manifest file (`corpus_dev.manifest.json`) to track file modification times
2. Added `--incremental` flag to only re-index changed files
3. Added `--status` flag to show what would change without indexing
4. Added `--force` flag to force full rebuild
5. Added `remove_document()` and `remove_documents_batch()` methods to processor
6. Added `remove_minicolumn()` method to HierarchicalLayer
7. Added robust progress tracking with `ProgressTracker` class
8. Added phase timing and logging support (`--log FILE`)
9. Added timeout support (`--timeout N`)
10. Added fast mode (default) that skips slow bigram connections

**Performance Fix:**
Identified that `compute_bigram_connections()` has O(n²) complexity with large corpora (26,000+ bigrams), causing hangs. Fast mode skips this operation:
- Before: >10 minutes (hung)
- After: ~2.3 seconds

**Files Modified:**
- `scripts/index_codebase.py` - Complete rewrite with incremental support (~840 lines)
- `cortical/processor.py` - Added `remove_document()`, `remove_documents_batch()` (~160 lines)
- `cortical/layers.py` - Added `remove_minicolumn()` (~20 lines)
- `tests/test_incremental_indexing.py` - 47 comprehensive tests
- `.claude/skills/corpus-indexer/SKILL.md` - Updated documentation
- `CLAUDE.md` - Updated Dog-Fooding section

**Usage:**
```bash
# Fast incremental update (~1-2s)
python scripts/index_codebase.py --incremental

# Check what would change
python scripts/index_codebase.py --status

# Full rebuild with logging
python scripts/index_codebase.py --force --log index.log

# With timeout safeguard
python scripts/index_codebase.py --timeout 60
```

---

## Intent-Based Code Search Enhancements

The following tasks enhance the system's ability to understand developer intent and retrieve code by meaning rather than exact keyword matching.

---

### 48. Add Code-Aware Tokenization

**Files:** `cortical/tokenizer.py`, `tests/test_tokenizer.py`
**Status:** [x] Completed
**Priority:** High

**Problem:**
Current tokenizer treats code like prose. It doesn't understand that `getUserCredentials`, `get_user_credentials`, and `fetch user credentials` are semantically equivalent.

**Solution Applied:**
1. Added `split_identifier()` function to break camelCase, PascalCase, underscore_style, and CONSTANT_STYLE
2. Added `PROGRAMMING_KEYWORDS` constant for common code terms (function, class, def, get, set, etc.)
3. Added `split_identifiers` parameter to `Tokenizer.__init__()` and `tokenize()` method
4. Tokens include both original identifier and split components when enabled
5. Split parts don't duplicate already-seen tokens, preserving proper bigram extraction

**Example:**
```python
tokenizer = Tokenizer(split_identifiers=True)
tokens = tokenizer.tokenize("getUserCredentials")
# ['getusercredentials', 'get', 'user', 'credentials']
```

**Tests Added:**
- 8 tests for `split_identifier()` function (camelCase, PascalCase, underscore_style, acronyms)
- 8 tests for code-aware tokenization (splitting, stop word filtering, min length, deduplication)

---

### 49. Add Synonym/Concept Mapping for Code Patterns

**Files:** `cortical/code_concepts.py`, `cortical/query.py`, `cortical/processor.py`
**Status:** [x] Completed
**Priority:** High

**Problem:**
The system doesn't know that "fetch", "get", "retrieve", "load" are often interchangeable in code contexts, or that "auth", "authentication", "credentials", "login" form a concept cluster.

**Solution Applied:**
1. Created `cortical/code_concepts.py` with 16 programming concept groups
2. Added `expand_code_concepts()` function for query expansion
3. Integrated with `expand_query()` via `use_code_concepts` parameter
4. Added `expand_query_for_code()` convenience method to processor
5. Added 33 tests in `tests/test_code_concepts.py`

**Concept Groups Implemented:**
- retrieval, storage, deletion, auth, error, validation
- transform, network, database, async, config, logging
- testing, file, iteration, lifecycle, events

---

### 50. Add Intent-Based Query Understanding

**Files:** `cortical/query.py`, `cortical/processor.py`, `tests/test_intent_query.py`
**Status:** [x] Completed
**Priority:** High

**Problem:**
Natural language queries like "where do we handle authentication?" aren't decomposed into searchable intents.

**Solution Applied:**
1. Added `parse_intent_query()` to extract action + subject + intent + expanded terms
2. Added `ParsedIntent` TypedDict for structured results
3. Added `QUESTION_INTENTS` mapping (where→location, how→implementation, what→definition, why→rationale, when→lifecycle)
4. Added `ACTION_VERBS` frozenset with 50+ common programming verbs
5. Added `search_by_intent()` for intent-aware document search
6. Added processor wrapper methods
7. Added 24 tests in `tests/test_intent_query.py`

**Example:**
```python
parse_intent_query("where do we handle authentication?")
# Returns: {
#   'action': 'handle',
#   'subject': 'authentication',
#   'intent': 'location',
#   'question_word': 'where',
#   'expanded_terms': ['handle', 'authentication', 'auth', 'login', ...]
# }
```

---

### 51. Add Fingerprint Export API

**Files:** `cortical/fingerprint.py`, `cortical/processor.py`, `tests/test_fingerprint.py`
**Status:** [x] Completed
**Priority:** Medium

**Problem:**
No way to export or compare the semantic representation of code blocks.

**Solution Applied:**
1. Created `cortical/fingerprint.py` with `SemanticFingerprint` TypedDict
2. Added `compute_fingerprint()` returning terms, concepts, bigrams, top_terms
3. Added `compare_fingerprints()` for cosine similarity scoring
4. Added `explain_fingerprint()` showing top contributing terms and concepts
5. Added `explain_similarity()` for human-readable explanations
6. Added processor methods: `get_fingerprint()`, `compare_fingerprints()`, `explain_fingerprint()`, `explain_similarity()`, `find_similar_texts()`
7. Added 24 tests in `tests/test_fingerprint.py`

**Use Cases:**
- Compare similarity between functions
- Find duplicate/similar code blocks
- Explain why two code blocks are related

---

### 52. Optimize Query-to-Corpus Comparison

**Files:** `cortical/query.py`, `cortical/processor.py`, `scripts/search_codebase.py`
**Status:** [x] Completed
**Priority:** Medium

**Problem:**
Each query recomputes expansions and scores against all documents. For interactive use, this should be faster.

**Solution Applied:**
1. Added `fast_find_documents()` using candidate pre-filtering
2. Added `build_document_index()` for pre-computed inverted index
3. Added `search_with_index()` for fastest cached search
4. Added processor wrappers: `fast_find_documents()`, `build_search_index()`, `search_with_index()`
5. Added `--fast` flag to search_codebase.py script
6. Added 20 tests in `tests/test_query_optimization.py`

**Performance:**
- `fast_find_documents()`: ~2-3x faster than full search
- `search_with_index()`: Fastest when index is cached

---

## Intelligence Documentation

The following tasks create self-describing documentation that improves the system's ability to understand itself when indexed. This creates a feedback loop: better documentation → better semantic search → better AI understanding of the codebase.

---

### 53. Create Algorithm Intelligence Documentation

**File:** New `docs/algorithms.md`
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Goal:**
Document the core IR algorithms in a way that helps semantic search understand what each algorithm does, when to use it, and how components relate.

**Content:**
- PageRank explanation with use cases
- TF-IDF calculation and per-document vs global variants
- Label propagation for concept clustering
- Co-occurrence counting ("Hebbian learning" metaphor)
- Relation extraction patterns
- Query expansion strategies

---

### 54. Create Architecture Intelligence Documentation

**File:** New `docs/architecture.md`
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Goal:**
Document the 4-layer architecture and data flow in searchable prose that helps answer "where is X handled?" and "how does X work?" queries.

**Content:**
- Layer 0 (Tokens): Word-level processing
- Layer 1 (Bigrams): Phrase patterns
- Layer 2 (Concepts): Topic clusters
- Layer 3 (Documents): Full document representations
- Cross-layer connections (feedforward/feedback)
- Minicolumn data structure

---

### 55. Create Pattern Glossary

**File:** New `docs/glossary.md`
**Status:** [x] Completed (2025-12-10)
**Priority:** Medium

**Goal:**
Define terminology used throughout the codebase so searches for concepts find relevant definitions.

**Terms:**
- Minicolumn, Edge, HierarchicalLayer
- Lateral connections, typed connections
- Feedforward/feedback connections
- PageRank, TF-IDF, damping factor
- Semantic relations (IsA, PartOf, etc.)
- Query expansion, spreading activation

---

### 56. Create Usage Patterns Documentation

**File:** New `docs/patterns.md`
**Status:** [x] Completed (2025-12-11)
**Priority:** Medium

**Goal:**
Document common usage patterns and code examples that help answer "how do I..." queries.

**Solution Applied:**
Created `docs/patterns.md` with 15 usage patterns covering:

1. **Code Search Patterns** (Patterns 1-4):
   - Code-aware tokenization with identifier splitting
   - Programming concept expansion
   - Intent-based code search
   - Combined code search

2. **Fingerprint Comparison** (Patterns 5-8):
   - Basic fingerprinting
   - Explain similarity
   - Find similar code blocks
   - Code deduplication

3. **Intent-Based Querying** (Patterns 9-10):
   - Query intent detection
   - Intent-aware search

4. **Document Type Boosting** (Patterns 11-12):
   - Boost documentation
   - Search with type filtering

5. **Configuration Patterns** (Patterns 13-15):
   - Custom configuration
   - Save and restore configuration
   - Domain-specific configurations

Note: Basic document processing, RAG retrieval, batch operations, and incremental
updates are already covered in `docs/cookbook.md`.

---

### 58. Git-Compatible Chunk-Based Indexing

**Files:** `scripts/index_codebase.py`, `cortical/chunk_index.py` (new), `tests/test_chunk_indexing.py` (new)
**Status:** [x] Completed (2025-12-10)
**Priority:** High

**Problem:**
The current pkl-based index cannot be tracked in git (binary, merge conflicts). This prevents sharing indexed state across branches and team members, requiring full rebuilds.

**Solution:**
Implement append-only, time-stamped JSON chunks that can be safely committed to git and merged without conflicts.

**Architecture:**
```
corpus_chunks/                        # Tracked in git
├── 2025-12-10_21-53-45_a1b2.json    # Session 1 changes
├── 2025-12-10_22-15-30_c3d4.json    # Session 2 changes
└── 2025-12-10_23-00-00_e5f6.json    # Session 3 changes

corpus_dev.pkl                        # NOT tracked (local cache)
```

**Chunk Format:**
```json
{
  "timestamp": "2025-12-10T21:53:45",
  "session_id": "a1b2c3d4",
  "branch": "feature-x",
  "operations": [
    {"op": "add", "doc_id": "docs/new.md", "content": "...", "mtime": 1234567890},
    {"op": "modify", "doc_id": "query.py", "content": "...", "mtime": 1234567891},
    {"op": "delete", "doc_id": "old.md"}
  ]
}
```

**Implementation Tasks:**
1. [x] Create `ChunkWriter` class - save session changes as timestamped JSON
2. [x] Create `ChunkLoader` class - combine chunks on startup (later timestamps win)
3. [x] Add cache validator - check if pkl matches combined chunk hash
4. [x] Add `ChunkCompactor` class - merge old chunks into single file
5. [x] Update CLI with `--use-chunks` flag
6. [x] Handle deletions with tombstones
7. [x] Add `.gitignore` entry for `corpus_dev.pkl` (keep chunks tracked)
8. [x] Add comprehensive tests (84 new tests)

**Startup Flow:**
```
1. Load all chunk files (sorted by timestamp)
2. Replay operations → build document set
   - Later timestamps win for conflicts
   - Deletes remove documents
3. Check if pkl cache is valid (hash of combined docs)
   - Valid: load pkl (fast)
   - Invalid: recompute analysis (~2s)
```

**Benefits:**
- No merge conflicts (unique timestamp+session names)
- Shared indexed state across team/branches
- Fast startup when cache valid
- Git-friendly (small JSON, append-only)
- Periodic compaction like `git gc`

**Usage (planned):**
```bash
# Index with chunks (creates timestamped JSON)
python scripts/index_codebase.py --incremental --use-chunks

# Compact old chunks
python scripts/index_codebase.py --compact --before 2025-12-01

# Status including chunk info
python scripts/index_codebase.py --status --use-chunks
```

---

## Code Review Findings (2025-12-10)

The following tasks were identified during code review of PR #23 (Git-Compatible Chunk-Based Indexing):

---

### 59. Rename TimeoutError to Avoid Built-in Shadowing

**Files:** `scripts/index_codebase.py`
**Status:** [x] Completed (2025-12-11) - Implemented as part of Task #60
**Priority:** Low

**Problem:**
The `TimeoutError` class shadows Python's built-in `TimeoutError` (introduced in Python 3.3). This could cause confusion and unexpected behavior when catching timeout exceptions.

**Solution Applied:**
Renamed the custom exception to `IndexingTimeoutError`:
```python
class IndexingTimeoutError(Exception):
    """Raised when indexing exceeds the timeout."""
    pass
```

---

### 60. Add Windows Compatibility for Timeout Handler

**Files:** `scripts/index_codebase.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** Medium

**Problem:**
The timeout handler uses `signal.SIGALRM` which is Unix-only. This will raise an `AttributeError` on Windows systems.

**Solution Applied:**
Added cross-platform timeout implementation:

1. **Platform detection**: Added `_IS_WINDOWS = platform.system() == 'Windows'`

2. **Windows implementation**: Uses `threading.Timer` with `threading.Event`:
   ```python
   timer = threading.Timer(seconds, timeout_callback)
   timer.daemon = True
   timer.start()
   # Check timed_out.is_set() after operations
   ```

3. **Unix implementation**: Continues using `signal.SIGALRM` (unchanged behavior)

4. **Limitation documented**: Windows implementation cannot interrupt blocking I/O operations

5. **Also addressed Task #59**: Renamed `TimeoutError` to `IndexingTimeoutError` to avoid shadowing the built-in

---

### 61. Add Chunk Size Warning for Large Chunks

**Files:** `cortical/chunk_index.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** Low

**Problem:**
Large chunk files in git can bloat repository history. There's no warning when chunks exceed a reasonable size threshold.

**Solution Applied:**
1. Added `DEFAULT_WARN_SIZE_KB = 1024` constant (1MB default threshold)
2. Added `warn_size_kb` parameter to `ChunkWriter.save()` method
3. Added warning emission when chunk file exceeds threshold
4. Warning includes helpful message suggesting `--compact`
5. Warning can be disabled by passing `warn_size_kb=0`
6. Added 3 tests for warning functionality

**Files Modified:**
- `cortical/chunk_index.py` - Added warning logic
- `tests/test_chunk_indexing.py` - Added 3 tests

**Usage:**
```python
writer = ChunkWriter('corpus_chunks')
writer.add_document('large_doc', 'x' * 2_000_000)

# Default: warn if > 1MB
writer.save()  # May emit warning

# Custom threshold
writer.save(warn_size_kb=500)  # Warn if > 500KB

# Disable warning
writer.save(warn_size_kb=0)  # Never warn
```

---

### 62. Add Chunk Compaction Documentation

**Files:** `CLAUDE.md`, `.claude/skills/corpus-indexer/SKILL.md`
**Status:** [x] Completed (2025-12-11)
**Priority:** Low

**Problem:**
The `--compact` feature is implemented but not documented in CLAUDE.md or the corpus-indexer skill.

**Solution Applied:**
Added comprehensive compaction documentation to both files covering:

1. **When to compact:**
   - After 10+ chunk files accumulate
   - When size warnings appear
   - Before merging branches
   - To clean up deleted entries

2. **How compaction works:**
   - Reads chunks in timestamp order
   - Replays operations (later timestamps win)
   - Creates single compacted chunk
   - Removes old chunk files
   - Preserves valid cache

3. **Example commands:**
   - `--compact --use-chunks` for full compaction
   - `--compact --before DATE --use-chunks` for date-based

4. **Recommended frequency:**
   - Weekly for active development
   - Monthly for maintenance
   - Before major releases

**Files Modified:**
- `CLAUDE.md` - Added "Chunk Compaction" section
- `.claude/skills/corpus-indexer/SKILL.md` - Added compaction section

---

## Dog-Fooding Findings (2025-12-10)

The following issues were identified during a dog-fooding session reviewing the docs folder and testing search quality.

---

### 65. Add Document Metadata to Chunk-Based Indexing (Prerequisite)

**Files:** `cortical/chunk_index.py`, `scripts/index_codebase.py`, `tests/test_chunk_indexing.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** High
**Blocks:** #63

**Problem:**
Chunk-based indexing loses all document metadata, making it impossible to implement document-type boosting for search.

**Solution Applied:**
1. Added `metadata` field to `ChunkOperation` dataclass
2. Updated `ChunkWriter.add_document()` and `modify_document()` to accept metadata
3. Updated `ChunkLoader` to track `_metadata` dict with `get_metadata()` method
4. Updated `ChunkCompactor.compact()` to preserve metadata during compaction
5. Added `_extract_file_metadata()` helper to extract:
   - `doc_type`: 'code', 'test', 'docs', or 'root_docs'
   - `headings`: List of markdown section headings (## and ###)
   - `language`: 'python' or 'markdown'
   - `function_count`, `class_count` for Python files
6. Updated `index_with_chunks()` to extract and use metadata
7. Added 12 new tests in `TestChunkMetadata` class

**Files Modified:**
- `cortical/chunk_index.py` - Added metadata support to ChunkOperation, ChunkWriter, ChunkLoader, ChunkCompactor
- `scripts/index_codebase.py` - Added `extract_markdown_headings()`, `get_doc_type()`, `_extract_file_metadata()`
- `tests/test_chunk_indexing.py` - Added 12 tests for metadata functionality

**This enables Task #63** - with metadata, we can now boost docs in search.

---

### 63. Improve Search Ranking for Documentation Files

**Files:** `cortical/query.py`, `scripts/search_codebase.py`, `tests/test_query.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** High
**Depends on:** #65

**Problem:**
When searching for conceptual terms like "PageRank algorithm" or "4-layer architecture", the search returns code implementations (processor.py) instead of documentation files (docs/algorithms.md, docs/architecture.md) that explicitly explain these concepts.

**Solution Applied:**

1. **Document-type boosting** - Added `DOC_TYPE_BOOSTS` dict with boost factors:
   - `docs/` folder: 1.5x
   - Root-level .md: 1.3x
   - Code files: 1.0x
   - Test files: 0.8x

2. **Query intent detection** - Added `is_conceptual_query()` function:
   - Detects conceptual keywords: "what", "explain", "describe", "architecture", etc.
   - Detects implementation keywords: "where", "implement", "function", etc.
   - Auto-boosts docs for conceptual queries

3. **New search function** - `find_documents_with_boost()`:
   - `auto_detect_intent=True` - Automatically boost docs for conceptual queries
   - `prefer_docs=True` - Always boost documentation
   - `custom_boosts` - Override boost factors

4. **Search script updates**:
   - Added `--prefer-docs` flag to always boost documentation
   - Added `--no-boost` flag to disable boosting (raw TF-IDF)
   - Shows document type indicator in results: `[DOCS]`, `[CODE]`, `[TEST]`
   - Shows query intent detection: "(Query type: conceptual/implementation)"

5. **Processor methods added**:
   - `processor.find_documents_with_boost()` - Search with doc-type boosting
   - `processor.is_conceptual_query()` - Check if query is conceptual

6. **Tests added** - 17 new tests in `TestDocTypeBoost` and `TestDocTypeBoostIntegration`

**Usage:**
```bash
# Auto-detect intent (default)
python scripts/search_codebase.py "what is PageRank"
# (Query type: conceptual) → docs boosted

# Force docs preference
python scripts/search_codebase.py "PageRank" --prefer-docs

# Disable boosting (raw TF-IDF)
python scripts/search_codebase.py "PageRank" --no-boost
```

---

### 64. Add Document Type Indicator to Search Results

**Files:** `scripts/search_codebase.py`
**Status:** [x] Completed (2025-12-11) - Implemented as part of Task #63
**Priority:** Low

**Problem:**
Search results show file paths but don't indicate document type at a glance. Users can't quickly distinguish documentation from code without reading the path.

**Solution Applied:**
Added document type labels to search results output:
- `[CODE]` - Code files (.py)
- `[DOCS]` - Documentation in docs/ folder
- `[DOC]` - Root-level markdown files
- `[TEST]` - Test files in tests/

**New Output:**
```
[1] [CODE] cortical/processor.py:578
    Score: 1.904

[2] [DOCS] docs/algorithms.md:19
    Score: 1.856
```

**Implementation:**
```python
def get_doc_type(doc_id: str) -> str:
    if doc_id.endswith('.md'):
        return 'DOCS'
    elif doc_id.startswith('tests/'):
        return 'TEST'
    else:
        return 'CODE'
```

---

### 66. Add Doc-Type Boosting to Passage-Level Search

**Files:** `cortical/query.py`, `cortical/processor.py`, `tests/test_query.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** Medium

**Problem:**
Document-level search correctly applies doc-type boosting (CLAUDE.md ranks #3 for "chunk compaction"), but passage-level search (`find_passages_for_query`) returns raw TF-IDF scores without boosting. This causes code snippets with keyword matches to rank higher than documentation passages for conceptual queries.

**Solution Applied:**
1. Added `apply_doc_boost` parameter to `find_passages_for_query()` (default True)
2. Added `auto_detect_intent` parameter to auto-boost docs for conceptual queries (default True)
3. Added `prefer_docs` parameter to always boost documentation (default False)
4. Added `custom_boosts` parameter for custom boost factors
5. Passage scores are multiplied by doc-type boost factor when appropriate
6. Definition passages also receive doc-type boost
7. Added processor wrappers with same parameters

**Files Modified:**
- `cortical/query.py` - Extended find_passages_for_query with boost parameters
- `cortical/processor.py` - Updated processor wrapper
- `tests/test_query.py` - Added 6 new tests

**Usage:**
```python
# Auto-detect conceptual queries and boost docs (default)
results = processor.find_passages_for_query("what is PageRank algorithm")

# Force docs preference
results = processor.find_passages_for_query("PageRank", prefer_docs=True)

# Disable boosting (raw TF-IDF)
results = processor.find_passages_for_query("PageRank", apply_doc_boost=False)

# Custom boost factors
results = processor.find_passages_for_query(
    "query",
    custom_boosts={'docs': 2.0, 'code': 0.8, 'test': 0.5}
)
```

---

## Dog-Fooding Summary

| # | Priority | Task | Status | Category |
|---|----------|------|--------|----------|
| 65 | High | Add document metadata to chunk indexing | [x] Completed | Infrastructure |
| 63 | High | Improve search ranking for docs | [x] Completed | Search Quality |
| 64 | Low | Add document type indicator | [x] Completed | UX |
| 66 | Medium | Add doc-type boost to passage search | [x] Completed | Search Quality |

**Dependency Chain:** #65 → #63 → #64 (all complete), #66 extends this work

**Status Update (2025-12-11):**
- Document-level search now correctly boosts documentation for conceptual queries
- Passage-level search still needs boosting (#66) - docs found but code ranks higher

---

## Code Review Summary (PR #23)

| # | Priority | Task | Status | Category |
|---|----------|------|--------|----------|
| 59 | Low | Rename TimeoutError to avoid shadowing | [x] Completed | Code Quality |
| 60 | Medium | Add Windows compatibility for timeout | [x] Completed | Compatibility |
| 61 | Low | Add chunk size warning | [x] Completed | UX |
| 62 | Low | Add chunk compaction documentation | [x] Completed | Documentation |

**Test Results:** 691 tests passing (including 32 new tests)

---

## Actionable Tasks Summary (Updated 2025-12-11)

| # | Priority | Task | Status | Category |
|---|----------|------|--------|----------|
| 41 | Medium | Create Configuration Dataclass | [x] Completed | Code Quality |
| 56 | Medium | Create Usage Patterns Documentation | [x] Completed | Documentation |
| 66 | Medium | Add doc-type boost to passage search | [x] Completed | Search Quality |
| 42 | Low | Add Simple Query Language Support | [ ] Not Started | Feature |
| 44 | Low | Remove Deprecated feedforward_sources | [ ] Not Started | Code Quality |
| 46 | Low | Standardize Return Types with Dataclasses | [ ] Not Started | Code Quality |

---

*Updated 2025-12-11*

---

# Developer Experience Enhancements

These tasks focus on making the Cortical Text Processor genuinely enjoyable to use for day-to-day development work.

---

## Showcase Improvements

### 67. Fix O(n) Lookup in Showcase find_concept_associations

**File:** `showcase.py`
**Lines:** 213-218
**Status:** [x] Completed
**Priority:** Low

**Problem:**
The `find_concept_associations` method iterates all minicolumns to find neighbor content:
```python
for c in layer0.minicolumns.values():
    if c.id == neighbor_id:
        # found it
```

This is O(n) when we have O(1) `get_by_id()` available.

**Solution:**
```python
neighbor = layer0.get_by_id(neighbor_id)
if neighbor:
    # use neighbor.content
```

---

### 68. Add Code-Specific Features to Showcase

**File:** `showcase.py`
**Status:** [x] Completed
**Priority:** Medium

**Problem:**
The showcase demonstrates general IR capabilities but not code-specific features documented in CLAUDE.md:
- `expand_query_for_code()` - programming-aware query expansion
- `search_by_intent()` - natural language intent queries
- `get_fingerprint()` / `compare_fingerprints()` - code similarity
- `is_conceptual_query()` - query type detection

**Solution:**
Add new demonstration sections:
1. **Code Query Expansion** - show how "fetch data" expands to include "get", "load", "retrieve"
2. **Intent-Based Search** - demonstrate "where do we handle errors?" style queries
3. **Code Fingerprinting** - compare two similar functions and explain their similarity
4. **Query Intent Detection** - show how system distinguishes "what is PageRank" vs "compute pagerank"

---

### 69. Add Passage-Level Search Demo to Showcase

**File:** `showcase.py`
**Status:** [x] Completed
**Priority:** Medium

**Problem:**
`find_passages_for_query()` is the key RAG capability for retrieving relevant code snippets, but it's not demonstrated. This is arguably the most useful feature for LLM integration.

**Solution:**
Add "RAG DEMONSTRATION" section showing:
1. Query → relevant passages with file:line references
2. How passage chunking works
3. Overlap handling for context preservation
4. Use case: feeding context to an LLM

---

### 70. Add Performance Timing to Showcase

**File:** `showcase.py`
**Status:** [x] Completed
**Priority:** Low

**Problem:**
No timing information shown. Users can't gauge performance characteristics.

**Solution:**
Add timing for key operations:
- Document processing time
- `compute_all()` time
- Query expansion time
- Document search time
- Passage retrieval time

---

## Code Index Improvements

### 71. Enable Code-Aware Tokenization in Index

**File:** `scripts/index_codebase.py`
**Status:** [x] Completed
**Priority:** High

**Problem:**
The indexer uses default tokenization which doesn't split identifiers. Searching for "user" won't find `getUserCredentials` or `user_credentials`.

**Solution:**
Enable `split_identifiers=True` when creating the processor:
```python
processor = CorticalTextProcessor(
    tokenizer_config={'split_identifiers': True}
)
```

Or configure per-document based on file type (.py files get identifier splitting).

**Impact:** Much better code search - "auth" would find `authenticate`, `AuthService`, `user_auth`, etc.

---

### 72. Use Programming Query Expansion in Search

**File:** `scripts/search_codebase.py`
**Status:** [x] Completed
**Priority:** High

**Problem:**
Search uses `expand_query()` but not `expand_query_for_code()`. Programming synonyms aren't utilized.

**Solution:**
```python
# In search_codebase.py
if is_code_query(query):  # Detect if searching for code patterns
    expanded = processor.expand_query_for_code(query)
else:
    expanded = processor.expand_query(query)
```

**Impact:** "get data" would expand to include "fetch", "load", "retrieve", "read" variations.

---

### 73. Add "Find Similar Code" Command

**Files:** `scripts/search_codebase.py`
**Status:** [x] Completed
**Priority:** Medium

**Problem:**
No way to find code similar to a given snippet or function. Fingerprinting exists but isn't exposed.

**Solution Applied:**
1. Added `find_similar_code()` function (~60 lines) implementing:
   - Parses file:line references to extract target text from indexed documents
   - Falls back to raw text comparison for direct code input
   - Uses `get_fingerprint()` and `compare_fingerprints()` for similarity scoring
   - Chunks documents and compares fingerprints against target
   - Returns top-N similar passages with scores and shared terms
2. Added `display_similar_results()` function for formatted output
3. Added `--similar-to` / `-s` argument to CLI
4. Added tests in `tests/test_search_codebase.py` (5 tests)

**Usage:**
```bash
# Find code similar to a file:line reference
python scripts/search_codebase.py --similar-to cortical/processor.py:100

# Find code similar to raw text
python scripts/search_codebase.py -s "def compute_score(items, weights)"

# With more results
python scripts/search_codebase.py --similar-to processor.py:50 --top 10
```

**Files Modified:**
- `scripts/search_codebase.py` - Added `find_similar_code()`, `display_similar_results()`, CLI argument (~120 lines)
- `tests/test_search_codebase.py` - New file with 23 tests for search functions

---

## Creative Developer Experience Features

### 74. Add "Explain This Code" Command

**Files:** `scripts/explain_code.py` (new)
**Status:** [ ] Not Started
**Priority:** Medium

**Problem:**
When jumping into unfamiliar code, it's hard to understand what it does and how it fits into the larger system.

**Solution:**
Create `explain_code.py` that uses semantic analysis to explain code:
```bash
python scripts/explain_code.py cortical/analysis.py:compute_pagerank

# Output:
# Function: compute_pagerank
# Purpose: Computes importance scores for tokens using iterative graph algorithm
# 
# Key Concepts: pagerank, damping, convergence, graph, centrality
# Related Files:
#   - cortical/processor.py:789 (calls this function)
#   - tests/test_analysis.py:45 (tests this function)
#   - CLAUDE.md:142 (documents this feature)
# 
# Similar Functions:
#   - compute_tfidf (same file) - also computes term importance
#   - compute_importance (processor.py) - wrapper method
```

**Implementation:**
1. Parse target location
2. Get semantic fingerprint
3. Find related documents (callers, tests, docs)
4. Find similar code patterns
5. Extract key concepts from local context

---

### 75. Add "What Changed?" Semantic Diff

**Files:** `scripts/what_changed.py` (new)
**Status:** [ ] Not Started
**Priority:** Medium

**Problem:**
Git diff shows line-by-line changes but doesn't explain semantic impact. Hard to review large changes.

**Solution:**
Create semantic diff tool:
```bash
python scripts/what_changed.py HEAD~5..HEAD

# Output:
# Semantic Summary of Changes (5 commits)
# 
# New Capabilities:
#   - Doc-type boosting for search results
#   - Chunk-based indexing for git compatibility
# 
# Modified Behaviors:
#   - Query expansion now considers document type
#   - Passage search has new boosting parameter
# 
# Files Most Affected:
#   - cortical/query.py (3 new functions, 2 modified)
#   - scripts/search_codebase.py (new --prefer-docs flag)
# 
# Concepts Impacted: search, ranking, documentation, indexing
```

**Implementation:**
1. Get changed files from git
2. Re-index changed files
3. Compare fingerprints before/after
4. Identify new concepts, modified concepts, removed concepts
5. Generate natural language summary

---

### 76. Add "Suggest Related Files" Feature

**Files:** `scripts/related_files.py` (new), integration with editor
**Status:** [ ] Not Started
**Priority:** Medium

**Problem:**
When editing a file, you often need to update related files (tests, docs, callers). Easy to miss something.

**Solution:**
```bash
python scripts/related_files.py cortical/query.py

# Output:
# Files related to cortical/query.py:
# 
# Tests (should update if changing behavior):
#   - tests/test_processor.py (47 references)
#   - tests/test_query.py (if exists)
# 
# Documentation (should update if changing API):
#   - CLAUDE.md (references: expand_query, find_documents)
#   - docs/usage-patterns.md (examples using query functions)
# 
# Callers (may be affected):
#   - cortical/processor.py (imports 12 functions)
#   - scripts/search_codebase.py (uses find_documents)
# 
# Similar Files (might need same changes):
#   - cortical/analysis.py (similar structure, shared patterns)
```

**Implementation:**
1. Find all files that import/reference target
2. Find test files that test target
3. Find docs that mention target functions
4. Find files with similar fingerprints
5. Rank by relevance

---

### 77. Add Interactive "Ask the Codebase" Mode

**Files:** `scripts/ask_codebase.py` (new)
**Status:** [x] Completed
**Priority:** High

**Problem:**
Current search returns passages but doesn't synthesize answers. You have to read multiple results.

**Solution:**
Create conversational interface that uses RAG to answer questions:
```bash
python scripts/ask_codebase.py

Ask> How does query expansion work?

Based on cortical/query.py:234-298 and CLAUDE.md:156:

Query expansion works by finding semantically related terms to add to a search:

1. Tokenizes the query into individual terms
2. For each term, finds lateral connections (co-occurring terms)
3. Weights expansions by connection strength and PageRank
4. Returns expanded query as term→weight dictionary

Key parameters:
- max_expansions: limit number of added terms (default: 10)
- use_semantic: include typed semantic relations (default: True)

Sources:
- cortical/query.py:234 (get_expanded_query_terms)
- cortical/query.py:298 (expand_query)
- CLAUDE.md:156 (Quick Reference)

Ask> What's different about expand_query_for_code?
...
```

**Implementation:**
1. Take natural language question
2. Detect intent (conceptual vs implementation)
3. Retrieve relevant passages with boosting
4. Synthesize answer from passages (or format for LLM)
5. Include source references

---

### 78. Add Code Pattern Detection

**Files:** `cortical/patterns.py` (new), `scripts/find_patterns.py` (new)
**Status:** [ ] Not Started
**Priority:** Low

**Problem:**
Hard to find all instances of a pattern (e.g., "all functions that iterate over minicolumns").

**Solution:**
```bash
# Find all places that iterate minicolumns (potential O(n) → O(1) optimization)
python scripts/find_patterns.py "for.*in.*minicolumns"

# Find all TF-IDF calculations
python scripts/find_patterns.py --semantic "tfidf calculation"

# Find potential bugs: linear search where O(1) exists
python scripts/find_patterns.py --smell "linear-search-with-index"
```

**Implementation:**
1. Regex patterns for syntactic search
2. Semantic patterns using fingerprint similarity
3. "Smell" patterns for common anti-patterns
4. Report with file:line references

---

### 79. Add Corpus Health Dashboard

**Files:** `scripts/corpus_health.py` (new)
**Status:** [ ] Not Started
**Priority:** Low

**Problem:**
No visibility into corpus quality - are docs well-connected? Any orphaned files? Coverage gaps?

**Solution:**
```bash
python scripts/corpus_health.py

# Output:
# ═══════════════════════════════════════════════════
#            CORPUS HEALTH REPORT
# ═══════════════════════════════════════════════════
# 
# Overall Health: 87% (Good)
# 
# Coverage:
#   ✓ 45/47 Python files indexed
#   ✓ 8/8 documentation files indexed
#   ✗ 2 files not indexed: __init__.py, __pycache__
# 
# Connectivity:
#   ✓ Average doc connections: 12.3
#   ✓ No orphaned documents
#   ⚠ 3 weakly connected: test_*.py files
# 
# Documentation Quality:
#   ✓ All public functions mentioned in docs
#   ⚠ 5 functions lack docstrings
#   ✗ TASK_LIST.md references deleted function
# 
# Index Freshness:
#   ✓ Corpus updated 2 minutes ago
#   ✓ All files current
# 
# Recommendations:
#   1. Add tests for cortical/chunk_index.py (0% coverage)
#   2. Document new functions in query.py:get_doc_type_boost
#   3. Update TASK_LIST.md references
```

---

### 80. Add "Learning Mode" for New Contributors

**Files:** `scripts/learn_codebase.py` (new)
**Status:** [ ] Not Started
**Priority:** Low

**Problem:**
New contributors struggle to understand unfamiliar codebases. Where to start? What's important?

**Solution:**
Interactive learning mode that guides exploration:
```bash
python scripts/learn_codebase.py

Welcome to the Cortical Text Processor codebase!

This codebase implements a hierarchical text analysis system inspired by
how the visual cortex processes information.

Would you like to:
1. Start with the architecture overview
2. Explore a specific feature
3. See the most important files
4. Take a guided tour

> 1

ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════

The system has 4 layers, like visual cortex V1→IT:

  Layer 0 (Tokens)   → Individual words
  Layer 1 (Bigrams)  → Word pairs
  Layer 2 (Concepts) → Semantic clusters
  Layer 3 (Documents)→ Full documents

Key files to understand:
  1. cortical/processor.py - Main API (start here)
  2. cortical/minicolumn.py - Core data structure
  3. cortical/analysis.py - Graph algorithms

[Press Enter to explore processor.py, or type a question]
> How does PageRank work here?

[Retrieves and explains relevant passages...]
```

---

### 81. Fix Tokenizer Underscore-Prefixed Identifier Bug

**File:** `cortical/tokenizer.py`
**Line:** 265
**Status:** [x] Completed
**Priority:** High
**Category:** Code Search

**Problem:**
The tokenizer regex `r'\b[a-zA-Z][a-zA-Z0-9_]*\b'` requires tokens to start with a letter, which causes Python dunder methods (`__init__`, `__slots__`, `__str__`) and private variables (`_id_index`, `_cache`) to be completely unsearchable in code search.

**Found via dog-fooding:** Searching for `__slots__` returns zero results even though it exists in the codebase.

**Root Cause:**
```python
# Line 265 - requires first char to be a letter
raw_tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text)
```

**Solution:**
1. Modify regex to capture underscore-prefixed identifiers:
   ```python
   raw_tokens = re.findall(r'\b_*[a-zA-Z][a-zA-Z0-9_]*\b', text)
   ```
2. Ensure dunder methods are preserved (not filtered as stop words)
3. Add `'init'`, `'str'`, `'repr'` etc. to `PROGRAMMING_KEYWORDS` if not present
4. Add tests for underscore-prefixed identifier tokenization

**Impact:**
- High for code search use case
- Python-specific identifiers currently invisible to search
- Affects private methods, dunder methods, internal variables

**Files to Modify:**
- `cortical/tokenizer.py` - Fix regex pattern
- `tests/test_tokenizer.py` - Add tests for underscore identifiers

---

### 84. Add Direct Definition Pattern Search for Code Search

**Files:** `cortical/query.py`, `cortical/processor.py`, `tests/test_query.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** High
**Category:** Code Search

**Problem:**
When searching for "class Minicolumn", the passage containing the actual class definition (`class Minicolumn:`) scores low because TF-IDF favors chunks with more query term matches. The definition chunk is mostly docstring text with few matching terms.

**Found via dog-fooding:** Even with document-level boosting, the actual class definition often doesn't appear in top results.

**Solution Applied:**
1. Added `is_definition_query()` function to detect definition queries (class/def/function/method patterns)
2. Added `find_definition_in_text()` to search for definition patterns in source code
3. Added `find_definition_passages()` to extract high-scoring passages from definitions
4. Updated `find_passages_for_query()` with `use_definition_search` parameter (default True)
5. Definition passages are injected with high boost score (5.0) before regular passages
6. Test files receive a penalty (0.6x) so source files rank higher
7. Added processor wrapper methods: `is_definition_query()`, `find_definition_passages()`

**Files Modified:**
- `cortical/query.py` - Added definition search functions (~160 lines)
- `cortical/processor.py` - Added processor wrappers (~65 lines)
- `tests/test_query.py` - Added 19 new tests

**Usage:**
```python
# Definition search is enabled by default
results = processor.find_passages_for_query("class Minicolumn")
# Returns passage containing actual class definition first

# Disable if needed
results = processor.find_passages_for_query("class Minicolumn", use_definition_search=False)

# Check if query is a definition query
is_def, def_type, name = processor.is_definition_query("def compute_pagerank")
# (True, 'function', 'compute_pagerank')
```

---

### 85. Improve Test File vs Source File Ranking

**Files:** `cortical/query.py`
**Status:** [x] Completed (2025-12-11) - Addressed by Tasks #84 and #66
**Priority:** Medium
**Category:** Code Search

**Problem:**
Test files often rank higher than source files because they mention class/function names more frequently (in test method names, assertions, etc.). This is counterintuitive for users looking for implementations.

**Found via dog-fooding:** Searching "class Minicolumn" returns test_layers.py results before minicolumn.py results.

**Solution Applied:**
This is now addressed by two mechanisms:

1. **Definition search (Task #84)** applies 0.6x penalty to test files for definition queries like "class Minicolumn" and "def compute_pagerank"

2. **Doc-type boost (Task #66)** applies 0.8x boost to test files via `DOC_TYPE_BOOSTS`. This is applied for:
   - Conceptual queries (auto-detected when `auto_detect_intent=True`)
   - When `prefer_docs=True` is set
   - Custom boosts via `custom_boosts` parameter

**Test file detection:**
```python
is_test = (doc_id.startswith('tests/') or '_test' in doc_id or 'test_' in doc_id)
```

**Result:**
For "class Minicolumn" queries, the actual class definition in `minicolumn.py` now ranks higher than test mentions in `test_layers.py`

---

### 86. Add Semantic Chunk Boundaries for Code

**Files:** `cortical/query.py`, `cortical/processor.py`, `tests/test_query.py`
**Status:** [x] Completed (2025-12-11)
**Priority:** Medium
**Category:** Code Search

**Problem:**
Current chunking uses fixed character boundaries which can split code mid-function or mid-class. This creates passages that lack context and score poorly.

**Found via dog-fooding:** The chunk containing `class Minicolumn:` starts mid-way through the previous function's code.

**Solution Applied:**
1. Added `find_code_boundaries()` to detect class/function/decorator boundaries
2. Added `create_code_aware_chunks()` to split at semantic boundaries
3. Added `is_code_file()` to detect code files by extension
4. Added `use_code_aware_chunks` parameter to `find_passages_for_query()` (default True)
5. Code files automatically use semantic chunking, other files use fixed chunking

**Files Modified:**
- `cortical/query.py` - Added code-aware chunking functions (~150 lines)
- `cortical/processor.py` - Updated processor wrapper
- `tests/test_query.py` - Added 15 new tests

**Supported boundaries:**
- Class definitions (`class Foo:`)
- Function/method definitions (`def bar():`, `async def baz():`)
- Decorators (`@decorator`)
- Comment separators (`# ---` or `# ===`)
- Blank line sequences

**Usage:**
```python
# Enabled by default for code files
results = processor.find_passages_for_query("class Minicolumn")

# Disable if needed
results = processor.find_passages_for_query("class Minicolumn", use_code_aware_chunks=False)
```

---

## Summary Table

| # | Priority | Task | Status | Category |
|---|----------|------|--------|----------|
| 67 | Low | Fix O(n) lookup in showcase | ✓ Done | Showcase |
| 68 | Medium | Add code-specific features to showcase | ✓ Done | Showcase |
| 69 | Medium | Add passage-level search demo | ✓ Done | Showcase |
| 70 | Low | Add performance timing to showcase | ✓ Done | Showcase |
| 71 | High | Enable code-aware tokenization in index | ✓ Done | Code Index |
| 72 | High | Use programming query expansion in search | ✓ Done | Code Index |
| 73 | Medium | Add "Find Similar Code" command | | Code Index |
| 74 | Medium | Add "Explain This Code" command | | Developer Experience |
| 75 | Medium | Add "What Changed?" semantic diff | | Developer Experience |
| 76 | Medium | Add "Suggest Related Files" feature | | Developer Experience |
| 77 | High | Add interactive "Ask the Codebase" mode | ✓ Done | Developer Experience |
| 78 | Low | Add code pattern detection | | Developer Experience |
| 79 | Low | Add corpus health dashboard | | Developer Experience |
| 80 | Low | Add "Learning Mode" for new contributors | | Developer Experience |
| 81 | High | Fix tokenizer underscore-prefixed identifiers | ✓ Done | Code Search |
| 82 | High | Add code stop words filter for query expansion | ✓ Done | Code Search |
| 83 | Medium | Add definition-aware boosting for class/def queries | ✓ Done | Code Search |
| 84 | High | Add direct definition pattern search | ✓ Done | Code Search |
| 85 | Medium | Improve test file vs source file ranking | ✓ Done | Code Search |
| 86 | Medium | Add semantic chunk boundaries for code | ✓ Done | Code Search |

---

*Added 2025-12-11, completions updated 2025-12-11*
