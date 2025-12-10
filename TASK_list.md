# Task List: Layer 2 Connection Improvements

## Problem Statement

Layer 2 (Concept Layer/V4) shows 0 connections when documents cover diverse topics because:
- Label propagation creates topic-specific clusters
- Concepts inherit only their members' documents
- Connection filter requires shared documents (Jaccard ≥ 0.1)
- No document overlap → no connections

## Tasks

### Task 1: Add Configurable Connection Thresholds ✅ COMPLETED
**File:** `cortical/analysis.py` (lines 614-812)

- [x] Add `min_shared_docs=0` option to allow connections without document overlap
- [x] Add `min_jaccard=0.0` option to disable Jaccard filtering
- [x] Expose these parameters in `CorticalTextProcessor.compute_concept_connections()`
- [x] Update docstrings to explain threshold behavior
- [x] Add tests for edge cases (zero thresholds, negative values)

### Task 2: Connect Concepts via Semantic Relations ✅ COMPLETED
**File:** `cortical/analysis.py`

- [x] Add new connection method that links concepts when their member tokens have semantic relations
- [x] For each concept pair, check if any (token1, relation, token2) exists in semantic_relations
- [x] Weight connections by number of semantic links between members
- [x] Make this work independently of document overlap
- [x] Add `use_member_semantics=True` parameter to `compute_concept_connections()`
- [x] Add tests verifying semantic-based connections

### Task 3: Connect Concepts via Shared Vocabulary/Embeddings ✅ COMPLETED
**File:** `cortical/analysis.py`

- [x] Add connection method based on embedding similarity between concept centroids
- [x] Compute concept centroid as average of member token embeddings
- [x] Connect concepts with cosine similarity above threshold
- [x] Add `use_embedding_similarity=True` and `embedding_threshold=0.3` parameters
- [x] Falls back gracefully if embeddings not computed
- [x] Add tests for embedding-based connections

### Task 4: Improve Clustering to Reduce Topic Isolation
**File:** `cortical/analysis.py` (lines 482-553)

- [ ] Add `cluster_strictness` parameter to label propagation (0.0-1.0)
- [ ] Lower strictness = more cross-topic token mixing in clusters
- [ ] Consider adding inter-document token bridging before clustering
- [ ] Add tests for different strictness levels

### Task 5: Integration and API Updates
**File:** `cortical/processor.py`

- [ ] Update `compute_all()` to accept connection strategy parameters
- [ ] Add `connection_strategy` enum: 'document_overlap', 'semantic', 'embedding', 'hybrid'
- [ ] 'hybrid' combines all three methods with configurable weights
- [ ] Update showcase.py to demonstrate different strategies
- [ ] Add documentation in CLAUDE.md

## Priority Order

1. **Task 1** (Quick win - just parameter changes)
2. **Task 2** (High value - semantic relations already exist)
3. **Task 3** (Medium - requires embeddings computed first)
4. **Task 4** (Lower priority - more invasive change)
5. **Task 5** (Final - ties everything together)

## Success Criteria

- Layer 2 shows meaningful connections even with diverse document topics
- User can choose connection strategy based on their use case
- All existing tests continue to pass
- New tests cover the added functionality
