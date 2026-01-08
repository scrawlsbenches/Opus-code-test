# SKG Method Mapping to Existing Cortical Modules

**Date**: 2025-12-30
**Purpose**: Map SemanticKnowledgeGraph methods to equivalent existing implementations

---

## Overview

The SemanticKnowledgeGraph (SKG) implements several algorithms that already exist in `cortical/`. This document maps each SKG method to its existing counterpart, enabling future refactoring to delegate rather than duplicate.

---

## Algorithm Mappings

### PageRank

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `_compute_pagerank()` | `cortical/analysis/pagerank.py` | **DELEGATE** |

**SKG Implementation** (`knowledge_graph.py:454-488`):
- Basic PageRank with damping factor
- Fixed iteration count
- No convergence check

**Existing Implementation** (`cortical/analysis/pagerank.py`):
- `compute_pagerank()` - Standard PageRank for single layer
- `compute_semantic_pagerank()` - PageRank with semantic relation weighting
- `compute_hierarchical_pagerank()` - Cross-layer propagation
- `_pagerank_iterate()` - Core loop with convergence tolerance

**Gap Analysis**:
- Existing has semantic weighting (uses `RELATION_WEIGHTS`)
- Existing has hierarchical cross-layer propagation
- Existing has convergence tolerance
- SKG is simpler but misses these features

**Migration Path**:
```python
# Before (SKG)
self._compute_pagerank()

# After (delegate)
from cortical.analysis.pagerank import compute_semantic_pagerank
pagerank_scores = compute_semantic_pagerank(self._layers, ...)
for node_id, score in pagerank_scores.items():
    self._nodes[node_id].pagerank = score
```

---

### TF-IDF / BM25

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `_compute_tfidf()` | `cortical/analysis/tfidf.py` | **DELEGATE** |
| `_bm25_score()` | `cortical/analysis/tfidf.py` | **DELEGATE** |

**SKG Implementation** (`knowledge_graph.py:490-644`):
- Basic TF-IDF with log IDF
- BM25 with k1=1.5, b=0.75

**Existing Implementation** (`cortical/analysis/tfidf.py`):
- `compute_tfidf()` - Standard TF-IDF
- `compute_bm25()` - BM25 with configurable parameters
- `parallel_tfidf()` - Parallelized for large corpora
- `parallel_bm25()` - Parallelized BM25

**Gap Analysis**:
- Existing has parallel versions for performance
- Existing integrates with layer structure
- Existing has more configuration options

---

### Query Expansion

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `search()` expansion | `cortical/query/expansion.py` | **DELEGATE** |
| `search_multihop()` | `cortical/query/expansion.py` | **DELEGATE** |

**SKG Implementation** (`knowledge_graph.py:516-551`):
- Simple lateral connection expansion
- Basic token matching

**Existing Implementation** (`cortical/query/expansion.py`):
- `expand_query()` - Full expansion with lateral, semantic, variants
- `expand_query_semantic()` - Semantic relation-based expansion
- `expand_query_multihop()` - Multi-hop inference through relation chains
- `score_relation_path()` - Semantic coherence scoring
- `VALID_RELATION_CHAINS` - Relation chain validity matrix

**Gap Analysis**:
- Existing has sophisticated relation chain scoring
- Existing supports code concept expansion
- Existing has TF-IDF weighting for expansions
- SKG expansion is very basic

**Migration Path**:
```python
# Before (SKG)
if expand_query:
    for token in query_tokens:
        # basic expansion...

# After (delegate)
from cortical.query.expansion import expand_query
expanded_terms = expand_query(
    query_text=query,
    layers=self._layers,
    tokenizer=_tokenizer,
    use_lateral=True,
    use_concepts=True,
)
```

---

### Search / Retrieval

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `search()` | `cortical/query/search.py` | **DELEGATE** |
| `spread_activation()` | `cortical/query/search.py` | **CONSIDER** |

**SKG Implementation**:
- Basic TF-IDF scoring
- Simple query expansion
- Manual document matching

**Existing Implementation** (`cortical/query/search.py`):
- `find_documents_for_query()` - Full document search
- `fast_find_documents()` - Optimized with candidate filtering
- `query_with_spreading_activation()` - Spreading activation search
- `graph_boosted_search()` - PageRank-boosted results
- Freshness decay for temporal relevance

**Gap Analysis**:
- Existing has candidate filtering for performance
- Existing has freshness decay
- Existing integrates with ranking module
- SKG search is functional but basic

---

### Spreading Activation

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `spread_activation()` | `cortical/analysis/activation.py` | **KEEP** (different purpose) |

**SKG Implementation** (`knowledge_graph.py:646-697`):
- Returns activation dictionary
- Configurable hops and decay

**Existing Implementation** (`cortical/analysis/activation.py`):
- `propagate_activation()` - Updates minicolumn activations in-place
- Works with layer structure

**Gap Analysis**:
- Different return semantics (dict vs in-place)
- SKG version is suitable for query-time use
- Existing is for batch propagation

---

### Ranking

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `search()` ranking | `cortical/query/ranking.py` | **DELEGATE** |

**SKG Implementation**:
- Combined TF-IDF + PageRank scoring
- Simple sort

**Existing Implementation** (`cortical/query/ranking.py`):
- `multi_stage_rank()` - Conceptual + implementation ranking
- `apply_doc_type_boost()` - Boost by document type
- `find_relevant_concepts()` - Concept-based relevance

---

## Persistence Mappings

| SKG Need | Existing Module | Recommendation |
|----------|-----------------|----------------|
| Save graph state | `cortical/state_storage.py` | **INTEGRATE** |
| Fault tolerance | `cortical/wal.py` | **INTEGRATE** |
| Crash recovery | `cortical/wal.py` | **INTEGRATE** |

**Existing Implementation** (`cortical/state_storage.py`):
- `StateWriter` - Save processor state to JSON files
- `StateLoader` - Load processor state
- Git-friendly format (diffable, no merge conflicts)
- Layer-by-layer storage

**Existing Implementation** (`cortical/wal.py`):
- `WALWriter` - Append-only operation log
- `WALEntry` - Log entry with checksum
- `SnapshotManager` - Periodic snapshots
- `WALRecovery` - Crash recovery

**Migration Path**:
```python
class SemanticKnowledgeGraph:
    def __init__(self, ..., persistence_dir: Optional[str] = None):
        if persistence_dir:
            self._wal = WALWriter(persistence_dir)
            self._state_writer = StateWriter(persistence_dir)

    def add_document(self, doc_id, content, ...):
        # ... existing code ...
        if self._wal:
            self._wal.append(WALEntry(
                operation="add_document",
                payload={"doc_id": doc_id, "content_hash": hash(content)}
            ))

    def save(self):
        if self._state_writer:
            self._state_writer.save_graph(self)

    @classmethod
    def load(cls, persistence_dir: str):
        loader = StateLoader(persistence_dir)
        # ... reconstruct graph ...
```

---

## Semantic Relations Mappings

| SKG Method | Existing Module | Recommendation |
|------------|-----------------|----------------|
| `_extract_semantic_relations()` | `cortical/semantics.py` | **DELEGATE** |

**SKG Implementation** (`knowledge_graph.py:401-452`):
- Regex-based IsA and PartOf extraction
- Simple pattern matching

**Existing Implementation** (`cortical/semantics.py`):
- `extract_pattern_relations()` - Comprehensive pattern extraction
- `extract_cooccurrence_relations()` - Statistical co-occurrence
- `retrofit_weights()` - Adjust weights based on relations
- `compute_semantic_coherence()` - Quality measurement

---

## Integration Adapters vs Real Systems

| SKG Adapter | Real System | Status |
|-------------|-------------|--------|
| `CELAdapter` | `cortical/cel/` | Adapter mirrors API |
| `GoTAdapter` | `cortical/got/` | Adapter mirrors API |
| `WovenMindAdapter` | `cortical/reasoning/woven_mind.py` | Adapter mirrors API |
| `PRISMAdapter` | `cortical/reasoning/prism_*.py` | Adapter mirrors API |
| `SparkSLMAdapter` | `cortical/spark/` | Adapter mirrors API |

**Current State**: Adapters are standalone implementations that can be swapped for real systems.

**Migration Path** (example for WovenMind):
```python
# In SemanticKnowledgeGraph.__init__()
if self._enable_woven_mind:
    if use_real_systems:
        from cortical.reasoning.woven_mind import WovenMind
        self._woven_mind = WovenMind()
    else:
        from .integrations import WovenMindAdapter
        self._woven_mind_adapter = WovenMindAdapter()
```

---

## Priority Matrix

| Integration | Priority | Complexity | Value |
|-------------|----------|------------|-------|
| Query Expansion | **HIGH** | Low | High - Better search quality |
| PageRank | Medium | Low | Medium - Already works |
| TF-IDF/BM25 | Medium | Low | Medium - Already works |
| Persistence | **HIGH** | Medium | High - Data durability |
| Ranking | Low | Low | Low - Current is adequate |
| Semantics | Low | Medium | Medium - Current is basic but ok |

---

## Recommended Phased Approach

### Phase 1 (Immediate) - COMPLETED
1. ✅ Integrate `cortical/query/expansion.py` for query expansion
   - Added `use_real_expansion=True` parameter to SKG
   - `search()` now delegates to `expand_query()` from expansion module
2. ✅ Add persistence via `cortical/wal.py` and `cortical/state_storage.py`
   - Added `persistence_dir` parameter to SKG
   - WAL logging on `add_document()` and `remove_document()`
   - Added `save()`, `load()`, and `replay_wal()` methods

### Phase 2 (When Needed)
3. Delegate PageRank to `cortical/analysis/pagerank.py`
4. Delegate TF-IDF/BM25 to `cortical/analysis/tfidf.py`
5. Use `cortical/query/ranking.py` for multi-stage ranking

### Phase 3 (Optional)
6. Wire adapters to real CEL, GoT, WovenMind, PRISM, SparkSLM
7. Add `cortical/semantics.py` for richer relation extraction
8. Integrate observability from `cortical/observability.py`

---

## Summary

The SKG has functional implementations that work. The existing `cortical/` modules offer:
- More sophisticated algorithms (hierarchical PageRank, semantic expansion)
- Better performance (parallel processing, candidate filtering)
- Production features (persistence, crash recovery)

The recommended approach is **incremental delegation** - keep the SKG interface stable while swapping in real implementations behind the scenes.
