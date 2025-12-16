---
title: "Core Processor"
generated: "2025-12-16T17:02:01.493976Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/processor/persistence_api.py"
  - "/home/user/Opus-code-test/cortical/processor/documents.py"
  - "/home/user/Opus-code-test/cortical/processor/__init__.py"
  - "/home/user/Opus-code-test/cortical/processor/core.py"
  - "/home/user/Opus-code-test/cortical/processor/introspection.py"
  - "/home/user/Opus-code-test/cortical/processor/compute.py"
tags:
  - architecture
  - modules
  - processor
---

# Core Processor

The core processor orchestrates all text processing operations.

## Modules

- **__init__.py**: Cortical Text Processor - Main processor package.
- **compute.py**: Compute methods: analysis, clustering, embeddings, semantic extraction.
- **core.py**: Core processor functionality: initialization, staleness tracking, and layer management.
- **documents.py**: Document management: processing, adding, removing, and metadata handling.
- **introspection.py**: Introspection: state inspection, fingerprints, gaps, and summaries.
- **persistence_api.py**: Persistence API: save, load, export, and migration methods.


## __init__.py

Cortical Text Processor - Main processor package.

This package splits the monolithic processor.py into focused modules:
- core.py: Initialization, staleness tracking, layer management
- documents.py:...


### Classes

#### CorticalTextProcessor

Neocortex-inspired text processing system.

### Dependencies

**Standard Library:**

- `compute.ComputeMixin`
- `core.CoreMixin`
- `documents.DocumentsMixin`
- `introspection.IntrospectionMixin`
- `persistence_api.PersistenceMixin`
- ... and 1 more



## compute.py

Compute methods: analysis, clustering, embeddings, semantic extraction.

This module contains all methods that perform computational analysis on the corpus,
including PageRank, TF-IDF, clustering, and...


### Classes

#### ComputeMixin

Mixin providing computation functionality.

**Methods:**

- `recompute`
- `compute_all`
- `resume_from_checkpoint`
- `propagate_activation`
- `compute_importance`
- `compute_semantic_importance`
- `compute_hierarchical_importance`
- `compute_tfidf`
- `compute_bm25`
- `compute_document_connections`
- `compute_bigram_connections`
- `build_concept_clusters`
- `compute_clustering_quality`
- `compute_concept_connections`
- `extract_corpus_semantics`
- `extract_pattern_relations`
- `retrofit_connections`
- `compute_property_inheritance`
- `compute_property_similarity`
- `compute_graph_embeddings`
- `retrofit_embeddings`
- `embedding_similarity`
- `find_similar_by_embedding`

### Functions

#### ComputeMixin.recompute

```python
ComputeMixin.recompute(self, level: str = 'stale', verbose: bool = True) -> Dict[str, bool]
```

Recompute specified analysis levels.

#### ComputeMixin.compute_all

```python
ComputeMixin.compute_all(self, verbose: bool = True, build_concepts: bool = True, pagerank_method: str = 'standard', connection_strategy: str = 'document_overlap', cluster_strictness: float = 1.0, bridge_weight: float = 0.0, progress_callback: Optional[ProgressReporter] = None, show_progress: bool = False, checkpoint_dir: Optional[str] = None, resume: bool = False) -> Dict[str, Any]
```

Run all computation steps.

#### ComputeMixin.resume_from_checkpoint

```python
ComputeMixin.resume_from_checkpoint(cls, checkpoint_dir: str, config: Optional['CorticalConfig'] = None, verbose: bool = True) -> 'CorticalTextProcessor'
```

Resume processing from a checkpoint directory.

#### ComputeMixin.propagate_activation

```python
ComputeMixin.propagate_activation(self, iterations: int = 3, decay: float = 0.8, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_importance

```python
ComputeMixin.compute_importance(self, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_semantic_importance

```python
ComputeMixin.compute_semantic_importance(self, relation_weights: Optional[Dict[str, float]] = None, verbose: bool = True) -> Dict[str, Any]
```

Compute PageRank with semantic relation weighting.

#### ComputeMixin.compute_hierarchical_importance

```python
ComputeMixin.compute_hierarchical_importance(self, layer_iterations: int = 10, global_iterations: int = 5, cross_layer_damping: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]
```

Compute PageRank with cross-layer propagation.

#### ComputeMixin.compute_tfidf

```python
ComputeMixin.compute_tfidf(self, verbose: bool = True) -> None
```

Compute document relevance scores using the configured algorithm.

#### ComputeMixin.compute_bm25

```python
ComputeMixin.compute_bm25(self, k1: float = None, b: float = None, verbose: bool = True) -> None
```

Compute BM25 scores for document relevance ranking.

#### ComputeMixin.compute_document_connections

```python
ComputeMixin.compute_document_connections(self, min_shared_terms: int = 3, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_bigram_connections

```python
ComputeMixin.compute_bigram_connections(self, min_shared_docs: int = 1, component_weight: float = 0.5, chain_weight: float = 0.7, cooccurrence_weight: float = 0.3, max_bigrams_per_term: int = 100, max_bigrams_per_doc: int = 500, max_connections_per_bigram: int = 50, verbose: bool = True) -> Dict[str, Any]
```

Build lateral connections between bigrams based on shared components and co-occurrence.

#### ComputeMixin.build_concept_clusters

```python
ComputeMixin.build_concept_clusters(self, min_cluster_size: Optional[int] = None, clustering_method: str = 'louvain', cluster_strictness: Optional[float] = None, bridge_weight: float = 0.0, resolution: Optional[float] = None, verbose: bool = True) -> Dict[int, List[str]]
```

Build concept clusters from token layer.

#### ComputeMixin.compute_clustering_quality

```python
ComputeMixin.compute_clustering_quality(self, sample_size: int = 500) -> Dict[str, Any]
```

Compute clustering quality metrics for the concept layer.

#### ComputeMixin.compute_concept_connections

```python
ComputeMixin.compute_concept_connections(self, use_semantics: bool = True, min_shared_docs: int = 1, min_jaccard: float = 0.1, use_member_semantics: bool = False, use_embedding_similarity: bool = False, embedding_threshold: float = 0.3, verbose: bool = True) -> Dict[str, Any]
```

Build lateral connections between concepts based on document overlap and semantics.

#### ComputeMixin.extract_corpus_semantics

```python
ComputeMixin.extract_corpus_semantics(self, use_pattern_extraction: bool = True, min_pattern_confidence: float = 0.6, max_similarity_pairs: int = 100000, min_context_keys: int = 3, verbose: bool = True) -> int
```

Extract semantic relations from the corpus.

#### ComputeMixin.extract_pattern_relations

```python
ComputeMixin.extract_pattern_relations(self, min_confidence: float = 0.6, verbose: bool = True) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations using pattern matching only.

#### ComputeMixin.retrofit_connections

```python
ComputeMixin.retrofit_connections(self, iterations: int = 10, alpha: float = 0.3, verbose: bool = True) -> Dict
```

None

#### ComputeMixin.compute_property_inheritance

```python
ComputeMixin.compute_property_inheritance(self, decay_factor: float = 0.7, max_depth: int = 5, apply_to_connections: bool = True, boost_factor: float = 0.3, verbose: bool = True) -> Dict[str, Any]
```

Compute property inheritance based on IsA hierarchy.

#### ComputeMixin.compute_property_similarity

```python
ComputeMixin.compute_property_similarity(self, term1: str, term2: str) -> float
```

Compute similarity between terms based on shared properties.

#### ComputeMixin.compute_graph_embeddings

```python
ComputeMixin.compute_graph_embeddings(self, dimensions: int = 64, method: str = 'fast', max_terms: Optional[int] = None, verbose: bool = True) -> Dict
```

Compute graph embeddings for tokens.

#### ComputeMixin.retrofit_embeddings

```python
ComputeMixin.retrofit_embeddings(self, iterations: int = 10, alpha: float = 0.4, verbose: bool = True) -> Dict
```

None

#### ComputeMixin.embedding_similarity

```python
ComputeMixin.embedding_similarity(self, term1: str, term2: str) -> float
```

None

#### ComputeMixin.find_similar_by_embedding

```python
ComputeMixin.find_similar_by_embedding(self, term: str, top_n: int = 10) -> List[Tuple[str, float]]
```

None

### Dependencies

**Standard Library:**

- `datetime.datetime`
- `json`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- ... and 11 more

**Local Imports:**

- `.analysis`
- `.embeddings`
- `.semantics`



## core.py

Core processor functionality: initialization, staleness tracking, and layer management.

This module contains the base class definition and core infrastructure that all
other processor mixins depend o...


### Classes

#### CoreMixin

Core mixin providing initialization and staleness tracking.

**Methods:**

- `is_stale`
- `get_stale_computations`
- `get_layer`
- `get_metrics`
- `get_metrics_summary`
- `reset_metrics`
- `enable_metrics`
- `disable_metrics`
- `record_metric`

### Functions

#### CoreMixin.is_stale

```python
CoreMixin.is_stale(self, computation_type: str) -> bool
```

Check if a specific computation is stale.

#### CoreMixin.get_stale_computations

```python
CoreMixin.get_stale_computations(self) -> set
```

Get the set of computations that are currently stale.

#### CoreMixin.get_layer

```python
CoreMixin.get_layer(self, layer: CorticalLayer) -> HierarchicalLayer
```

Get a specific layer by enum.

#### CoreMixin.get_metrics

```python
CoreMixin.get_metrics(self) -> Dict[str, Dict[str, Any]]
```

Get all collected metrics.

#### CoreMixin.get_metrics_summary

```python
CoreMixin.get_metrics_summary(self) -> str
```

Get a human-readable summary of all metrics.

#### CoreMixin.reset_metrics

```python
CoreMixin.reset_metrics(self) -> None
```

Clear all collected metrics.

#### CoreMixin.enable_metrics

```python
CoreMixin.enable_metrics(self) -> None
```

Enable metrics collection.

#### CoreMixin.disable_metrics

```python
CoreMixin.disable_metrics(self) -> None
```

Disable metrics collection.

#### CoreMixin.record_metric

```python
CoreMixin.record_metric(self, metric_name: str, count: int = 1) -> None
```

Record a custom count metric.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `logging`
- `minicolumn.Minicolumn`
- ... and 5 more



## documents.py

Document management: processing, adding, removing, and metadata handling.

This module contains all methods related to managing documents in the corpus.


### Classes

#### DocumentsMixin

Mixin providing document management functionality.

**Methods:**

- `process_document`
- `set_document_metadata`
- `get_document_metadata`
- `get_all_document_metadata`
- `add_document_incremental`
- `add_documents_batch`
- `remove_document`
- `remove_documents_batch`

### Functions

#### DocumentsMixin.process_document

```python
DocumentsMixin.process_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, int]
```

Process a document and add it to the corpus.

#### DocumentsMixin.set_document_metadata

```python
DocumentsMixin.set_document_metadata(self, doc_id: str, **kwargs) -> None
```

Set or update metadata for a document.

#### DocumentsMixin.get_document_metadata

```python
DocumentsMixin.get_document_metadata(self, doc_id: str) -> Dict[str, Any]
```

Get metadata for a document.

#### DocumentsMixin.get_all_document_metadata

```python
DocumentsMixin.get_all_document_metadata(self) -> Dict[str, Dict[str, Any]]
```

Get metadata for all documents.

#### DocumentsMixin.add_document_incremental

```python
DocumentsMixin.add_document_incremental(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, recompute: str = 'tfidf') -> Dict[str, int]
```

Add a document with selective recomputation for efficiency.

#### DocumentsMixin.add_documents_batch

```python
DocumentsMixin.add_documents_batch(self, documents: List[Tuple[str, str, Optional[Dict[str, Any]]]], recompute: str = 'full', verbose: bool = True) -> Dict[str, Any]
```

Add multiple documents with a single recomputation.

#### DocumentsMixin.remove_document

```python
DocumentsMixin.remove_document(self, doc_id: str, verbose: bool = False) -> Dict[str, Any]
```

Remove a document from the corpus.

#### DocumentsMixin.remove_documents_batch

```python
DocumentsMixin.remove_documents_batch(self, doc_ids: List[str], recompute: str = 'none', verbose: bool = True) -> Dict[str, Any]
```

Remove multiple documents efficiently with single recomputation.

### Dependencies

**Standard Library:**

- `copy`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- `typing.Any`
- ... and 4 more



## introspection.py

Introspection: state inspection, fingerprints, gaps, and summaries.

This module contains methods for examining the processor state and
comparing texts/documents.


### Classes

#### IntrospectionMixin

Mixin providing introspection functionality.

**Methods:**

- `get_document_signature`
- `get_corpus_summary`
- `analyze_knowledge_gaps`
- `detect_anomalies`
- `get_fingerprint`
- `compare_fingerprints`
- `explain_fingerprint`
- `explain_similarity`
- `find_similar_texts`
- `compare_with`
- `compare_documents`
- `what_changed`
- `summarize_document`
- `detect_patterns`
- `detect_patterns_in_corpus`
- `get_pattern_summary`
- `get_corpus_pattern_statistics`
- `format_pattern_report`
- `list_available_patterns`
- `list_pattern_categories`

### Functions

#### IntrospectionMixin.get_document_signature

```python
IntrospectionMixin.get_document_signature(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]
```

Get the top-n TF-IDF terms for a document.

#### IntrospectionMixin.get_corpus_summary

```python
IntrospectionMixin.get_corpus_summary(self) -> Dict
```

Get summary statistics about the corpus.

#### IntrospectionMixin.analyze_knowledge_gaps

```python
IntrospectionMixin.analyze_knowledge_gaps(self) -> Dict
```

Analyze the corpus for knowledge gaps.

#### IntrospectionMixin.detect_anomalies

```python
IntrospectionMixin.detect_anomalies(self, threshold: float = 0.3) -> List[Dict]
```

Detect anomalous patterns in the corpus.

#### IntrospectionMixin.get_fingerprint

```python
IntrospectionMixin.get_fingerprint(self, text: str, top_n: int = 20) -> Dict
```

Compute the semantic fingerprint of a text.

#### IntrospectionMixin.compare_fingerprints

```python
IntrospectionMixin.compare_fingerprints(self, fp1: Dict, fp2: Dict) -> Dict
```

Compare two fingerprints and compute similarity metrics.

#### IntrospectionMixin.explain_fingerprint

```python
IntrospectionMixin.explain_fingerprint(self, fp: Dict, top_n: int = 10) -> Dict
```

Generate a human-readable explanation of a fingerprint.

#### IntrospectionMixin.explain_similarity

```python
IntrospectionMixin.explain_similarity(self, fp1: Dict, fp2: Dict) -> str
```

Generate a human-readable explanation of fingerprint similarity.

#### IntrospectionMixin.find_similar_texts

```python
IntrospectionMixin.find_similar_texts(self, text: str, candidates: List[Tuple[str, str]], top_n: int = 5) -> List[Tuple[str, float, Dict]]
```

Find texts most similar to the given text.

#### IntrospectionMixin.compare_with

```python
IntrospectionMixin.compare_with(self, other: 'CorticalTextProcessor', top_movers: int = 20, min_pagerank_delta: float = 0.0001) -> 'diff_module.SemanticDiff'
```

Compare this processor state with another to find semantic differences.

#### IntrospectionMixin.compare_documents

```python
IntrospectionMixin.compare_documents(self, doc_id_1: str, doc_id_2: str) -> Dict
```

Compare two documents within this corpus.

#### IntrospectionMixin.what_changed

```python
IntrospectionMixin.what_changed(self, old_content: str, new_content: str) -> Dict
```

Compare two text contents to show what changed semantically.

#### IntrospectionMixin.summarize_document

```python
IntrospectionMixin.summarize_document(self, doc_id: str, num_sentences: int = 3) -> str
```

Generate a summary of a document using extractive summarization.

#### IntrospectionMixin.detect_patterns

```python
IntrospectionMixin.detect_patterns(self, doc_id: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]
```

Detect programming patterns in a specific document.

#### IntrospectionMixin.detect_patterns_in_corpus

```python
IntrospectionMixin.detect_patterns_in_corpus(self, patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]
```

Detect patterns across all documents in the corpus.

#### IntrospectionMixin.get_pattern_summary

```python
IntrospectionMixin.get_pattern_summary(self, doc_id: str) -> Dict[str, int]
```

Get a summary of pattern occurrences in a document.

#### IntrospectionMixin.get_corpus_pattern_statistics

```python
IntrospectionMixin.get_corpus_pattern_statistics(self) -> Dict[str, Any]
```

Get pattern statistics across the entire corpus.

#### IntrospectionMixin.format_pattern_report

```python
IntrospectionMixin.format_pattern_report(self, doc_id: str, show_lines: bool = False) -> str
```

Format pattern detection results as a human-readable report.

#### IntrospectionMixin.list_available_patterns

```python
IntrospectionMixin.list_available_patterns(self) -> List[str]
```

List all available pattern names that can be detected.

#### IntrospectionMixin.list_pattern_categories

```python
IntrospectionMixin.list_pattern_categories(self) -> List[str]
```

List all pattern categories.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `logging`
- `re`
- `typing.Any`
- `typing.Dict`
- ... and 4 more

**Local Imports:**

- `.fingerprint`
- `.gaps`
- `.patterns`
- `.persistence`



## persistence_api.py

Persistence API: save, load, export, and migration methods.

This module contains all methods related to saving and loading processor state.


### Classes

#### PersistenceMixin

Mixin providing persistence functionality.

**Methods:**

- `save`
- `load`
- `save_json`
- `load_json`
- `migrate_to_json`
- `export_graph`
- `export_conceptnet_json`

### Functions

#### PersistenceMixin.save

```python
PersistenceMixin.save(self, filepath: str, verbose: bool = True, signing_key: Optional[bytes] = None) -> None
```

Save processor state to a file.

#### PersistenceMixin.load

```python
PersistenceMixin.load(cls, filepath: str, verbose: bool = True, verify_key: Optional[bytes] = None) -> 'CorticalTextProcessor'
```

Load processor state from a file.

#### PersistenceMixin.save_json

```python
PersistenceMixin.save_json(self, state_dir: str, force: bool = False, verbose: bool = True) -> Dict[str, bool]
```

Save processor state to git-friendly JSON format.

#### PersistenceMixin.load_json

```python
PersistenceMixin.load_json(cls, state_dir: str, config: Optional[CorticalConfig] = None, verbose: bool = True) -> 'CorticalTextProcessor'
```

Load processor from git-friendly JSON format.

#### PersistenceMixin.migrate_to_json

```python
PersistenceMixin.migrate_to_json(self, pkl_path: str, json_dir: str, verbose: bool = True) -> bool
```

Migrate existing pickle file to git-friendly JSON format.

#### PersistenceMixin.export_graph

```python
PersistenceMixin.export_graph(self, filepath: str, layer: Optional[CorticalLayer] = None, max_nodes: int = 500) -> Dict
```

Export graph to JSON for visualization.

#### PersistenceMixin.export_conceptnet_json

```python
PersistenceMixin.export_conceptnet_json(self, filepath: str, include_cross_layer: bool = True, include_typed_edges: bool = True, min_weight: float = 0.0, min_confidence: float = 0.0, max_nodes_per_layer: int = 100, verbose: bool = True) -> Dict[str, Any]
```

Export ConceptNet-style graph for visualization.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- `typing.Any`
- ... and 3 more

**Local Imports:**

- `.persistence`
- `.state_storage`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
