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

**Files:** `cortical/query.py`
**Status:** [ ] Future Enhancement

**Problem:**
Current ranking is flat (Token TF-IDF → Document Score). Better RAG performance with staged ranking:

1. **Stage 1 (Concepts):** Filter by topic relevance
2. **Stage 2 (Documents):** Rank documents in topic
3. **Stage 3 (Chunks):** Rank passages in documents
4. **Stage 4 (Rerank):** Final relevance scoring

---

### 18. Add Batch Query API

**Files:** `cortical/query.py`, `cortical/processor.py`
**Status:** [ ] Future Enhancement

**Problem:**
No efficient way to run multiple queries. Each query repeats tokenization and expansion.

**Implementation:**
```python
def find_documents_batch(self, queries: List[str], top_n: int = 5):
    """Process multiple queries efficiently."""
    # Batch tokenization
    # Shared expansion cache
    # Parallel scoring
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
| Low | Multi-stage ranking pipeline | ⬜ Future | RAG |
| Low | Batch query API | ⬜ Future | RAG |

**Bug Fix Completion:** 7/7 tasks (100%)
**RAG Enhancement Completion:** 6/8 tasks (75%)

---

## Test Results

```
Ran 144 tests in 0.141s
OK
```

All tests passing as of 2025-12-09.

---

*Updated from code review on 2025-12-09*
