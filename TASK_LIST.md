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

## Code Review Concerns

The following concerns were identified during code review and should be addressed in future iterations:

### 31. Consider Splitting processor.py

**File:** `cortical/processor.py`
**Status:** [ ] Future Enhancement
**Priority:** Low

**Concern:**
The `processor.py` file has grown to 800+ lines with the addition of incremental indexing, batch APIs, and multi-stage ranking. Consider splitting into smaller modules:
- `processor_core.py` - Core document processing
- `processor_batch.py` - Batch operations (add_documents_batch, find_*_batch)
- `processor_incremental.py` - Incremental indexing and staleness tracking

---

### 32. Semantic Lookup Memory Optimization

**File:** `cortical/analysis.py`
**Function:** `compute_concept_connections()`
**Status:** [ ] Future Enhancement
**Priority:** Medium

**Concern:**
The semantic lookup builds a double-nested dictionary (`Dict[str, Dict[str, Tuple[str, float]]]`) which stores relations in both directions. For large semantic relation sets (10K+ relations), this could consume significant memory.

**Potential Solution:**
- Use a single direction and check both orderings at lookup time
- Or use a frozenset key: `{(t1, t2): (relation, weight)}`

---

### 33. Tune Semantic Bonus Cap

**File:** `cortical/analysis.py`
**Line:** ~408
**Status:** [ ] Future Enhancement
**Priority:** Low

**Concern:**
Semantic bonus is capped at 50% boost (`min(avg_semantic, 0.5)`). This is a reasonable default but may benefit from:
- Making it a configurable parameter
- Empirical testing on different corpus types

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
| Low | Add analogy completion | ⏳ Pending | ConceptNet |

**Bug Fix Completion:** 7/7 tasks (100%)
**RAG Enhancement Completion:** 8/8 tasks (100%)
**ConceptNet Enhancement Completion:** 11/12 tasks (92%)

---

## Test Results

```
Ran 307 tests in 0.275s
OK
```

All tests passing as of 2025-12-10.

---

*Updated from code review on 2025-12-10*
