---
title: "Search & Retrieval"
generated: "2025-12-16T17:02:01.493418Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/query/ranking.py"
  - "/home/user/Opus-code-test/cortical/query/intent.py"
  - "/home/user/Opus-code-test/cortical/query/passages.py"
  - "/home/user/Opus-code-test/cortical/query/__init__.py"
  - "/home/user/Opus-code-test/cortical/query/search.py"
  - "/home/user/Opus-code-test/cortical/query/definitions.py"
  - "/home/user/Opus-code-test/cortical/query/chunking.py"
  - "/home/user/Opus-code-test/cortical/query/expansion.py"
tags:
  - architecture
  - modules
  - query
---

# Search & Retrieval

Search and retrieval components for finding relevant documents and passages.

## Modules

- **__init__.py**: Query Module
- **chunking.py**: Chunking Module
- **definitions.py**: Definition Search Module
- **expansion.py**: Query Expansion Module
- **intent.py**: Intent Query Module
- **passages.py**: Passage Retrieval Module
- **ranking.py**: Ranking Module
- **search.py**: Document Search Module


## __init__.py

Query Module
============

Query expansion and search functionality.

This package provides methods for expanding queries using lateral connections,
concept clusters, and word variants, then searching...


### Dependencies

**Standard Library:**

- `analogy.complete_analogy`
- `analogy.complete_analogy_simple`
- `analogy.find_relation_between`
- `analogy.find_terms_with_relation`
- `chunking.CODE_BOUNDARY_PATTERN`
- ... and 49 more



## chunking.py

Chunking Module
==============

Functions for splitting documents into chunks for passage retrieval.

This module provides:
- Fixed-size text chunking with overlap
- Code-aware chunking aligned to sem...


### Functions

#### create_chunks

```python
create_chunks(text: str, chunk_size: int = 512, overlap: int = 128) -> List[Tuple[str, int, int]]
```

Split text into overlapping chunks.

#### find_code_boundaries

```python
find_code_boundaries(text: str) -> List[int]
```

Find semantic boundaries in code (class/function definitions, decorators).

#### create_code_aware_chunks

```python
create_code_aware_chunks(text: str, target_size: int = 512, min_size: int = 100, max_size: int = 1024) -> List[Tuple[str, int, int]]
```

Create chunks aligned to code structure boundaries.

#### is_code_file

```python
is_code_file(doc_id: str) -> bool
```

Determine if a document is a code file based on its path/extension.

#### precompute_term_cols

```python
precompute_term_cols(query_terms: Dict[str, float], layer0: HierarchicalLayer) -> Dict[str, 'Minicolumn']
```

Pre-compute minicolumn lookups for query terms.

#### score_chunk_fast

```python
score_chunk_fast(chunk_tokens: List[str], query_terms: Dict[str, float], term_cols: Dict[str, 'Minicolumn'], doc_id: Optional[str] = None) -> float
```

Fast chunk scoring using pre-computed minicolumn lookups.

#### score_chunk

```python
score_chunk(chunk_text: str, query_terms: Dict[str, float], layer0: HierarchicalLayer, tokenizer: Tokenizer, doc_id: Optional[str] = None) -> float
```

Score a chunk against query terms using TF-IDF.

### Dependencies

**Standard Library:**

- `layers.HierarchicalLayer`
- `re`
- `tokenizer.Tokenizer`
- `typing.Dict`
- `typing.List`
- ... and 3 more



## definitions.py

Definition Search Module
========================

Functions for finding and boosting code definitions (classes, functions, methods).

This module handles:
- Detection of definition-seeking queries ("...


### Classes

#### DefinitionQuery

Info about a definition-seeking query.

### Functions

#### is_definition_query

```python
is_definition_query(query_text: str) -> Tuple[bool, Optional[str], Optional[str]]
```

Detect if a query is looking for a code definition.

#### find_definition_in_text

```python
find_definition_in_text(text: str, identifier: str, def_type: str, context_chars: int = 500) -> Optional[Tuple[str, int, int]]
```

Find a definition in source text and extract surrounding context.

#### find_definition_passages

```python
find_definition_passages(query_text: str, documents: Dict[str, str], context_chars: int = 500, boost: float = DEFINITION_BOOST) -> List[Tuple[str, str, int, int, float]]
```

Find definition passages for a definition query.

#### detect_definition_query

```python
detect_definition_query(query_text: str) -> DefinitionQuery
```

Detect if a query is searching for a code definition.

#### apply_definition_boost

```python
apply_definition_boost(passages: List[Tuple[str, str, int, int, float]], query_text: str, boost_factor: float = 3.0) -> List[Tuple[str, str, int, int, float]]
```

Boost passages that contain actual code definitions matching the query.

#### is_test_file

```python
is_test_file(doc_id: str) -> bool
```

Detect if a document ID represents a test file.

#### boost_definition_documents

```python
boost_definition_documents(doc_results: List[Tuple[str, float]], query_text: str, documents: Dict[str, str], boost_factor: float = 2.0, test_with_definition_penalty: float = 0.5, test_without_definition_penalty: float = 0.7) -> List[Tuple[str, float]]
```

Boost documents that contain the actual definition being searched for.

### Dependencies

**Standard Library:**

- `re`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 2 more



## expansion.py

Query Expansion Module
=====================

Functions for expanding query terms using lateral connections,
semantic relations, and code concept synonyms.

This module provides:
- Basic query expansi...


### Functions

#### score_relation_path

```python
score_relation_path(path: List[str]) -> float
```

Score a relation path by its semantic coherence.

#### expand_query

```python
expand_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, max_expansions: int = 10, use_lateral: bool = True, use_concepts: bool = True, use_variants: bool = True, use_code_concepts: bool = False, filter_code_stop_words: bool = False, tfidf_weight: float = 0.7, max_expansion_weight: float = 2.0) -> Dict[str, float]
```

Expand a query using lateral connections and concept clusters.

#### expand_query_semantic

```python
expand_query_semantic(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, semantic_relations: List[Tuple[str, str, str, float]], max_expansions: int = 10) -> Dict[str, float]
```

Expand query using semantic relations extracted from corpus.

#### expand_query_multihop

```python
expand_query_multihop(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, semantic_relations: List[Tuple[str, str, str, float]], max_hops: int = 2, max_expansions: int = 15, decay_factor: float = 0.5, min_path_score: float = 0.2) -> Dict[str, float]
```

Expand query using multi-hop semantic inference.

#### get_expanded_query_terms

```python
get_expanded_query_terms(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, max_expansions: int = 5, semantic_discount: float = 0.8, filter_code_stop_words: bool = False) -> Dict[str, float]
```

Get expanded query terms with optional semantic expansion.

### Dependencies

**Standard Library:**

- `code_concepts.expand_code_concepts`
- `collections.defaultdict`
- `config.DEFAULT_CHAIN_VALIDITY`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 6 more



## intent.py

Intent Query Module
==================

Intent-based query understanding for natural language code search.

This module handles:
- Parsing natural language queries to extract intent (where, how, what,...


### Classes

#### ParsedIntent

Structured representation of a parsed query intent.

### Functions

#### parse_intent_query

```python
parse_intent_query(query_text: str) -> ParsedIntent
```

Parse a natural language query to extract intent and searchable terms.

#### search_by_intent

```python
search_by_intent(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: 'Tokenizer', top_n: int = 5) -> List[Tuple[str, float, ParsedIntent]]
```

Search the corpus using intent-based query understanding.

### Dependencies

**Standard Library:**

- `code_concepts.get_related_terms`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`
- ... and 4 more



## passages.py

Passage Retrieval Module
========================

Functions for retrieving relevant passages from documents.

This module provides:
- Passage retrieval for RAG systems
- Batch passage retrieval
- Int...


### Functions

#### find_passages_for_query

```python
find_passages_for_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, use_expansion: bool = True, doc_filter: Optional[List[str]] = None, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, use_definition_search: bool = True, definition_boost: float = DEFINITION_BOOST, apply_doc_boost: bool = True, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, auto_detect_intent: bool = True, prefer_docs: bool = False, custom_boosts: Optional[Dict[str, float]] = None, use_code_aware_chunks: bool = True, filter_code_stop_words: bool = True, test_file_penalty: float = 0.8) -> List[Tuple[str, str, int, int, float]]
```

Find text passages most relevant to a query.

#### find_documents_batch

```python
find_documents_batch(queries: List[str], layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[List[Tuple[str, float]]]
```

Find documents for multiple queries efficiently.

#### find_passages_batch

```python
find_passages_batch(queries: List[str], layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, use_expansion: bool = True, doc_filter: Optional[List[str]] = None, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[List[Tuple[str, str, int, int, float]]]
```

Find passages for multiple queries efficiently.

### Dependencies

**Standard Library:**

- `chunking.CODE_BOUNDARY_PATTERN`
- `chunking.create_chunks`
- `chunking.create_code_aware_chunks`
- `chunking.find_code_boundaries`
- `chunking.is_code_file`
- ... and 18 more



## ranking.py

Ranking Module
=============

Multi-stage ranking and document type boosting for search results.

This module provides:
- Document type boosting (docs, code, tests)
- Conceptual vs implementation quer...


### Functions

#### is_conceptual_query

```python
is_conceptual_query(query_text: str) -> bool
```

Determine if a query is conceptual (should boost documentation).

#### get_doc_type_boost

```python
get_doc_type_boost(doc_id: str, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, custom_boosts: Optional[Dict[str, float]] = None) -> float
```

Get the boost factor for a document based on its type.

#### apply_doc_type_boost

```python
apply_doc_type_boost(results: List[Tuple[str, float]], doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, boost_docs: bool = True, custom_boosts: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]
```

Apply document type boosting to search results.

#### find_documents_with_boost

```python
find_documents_with_boost(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, auto_detect_intent: bool = True, prefer_docs: bool = False, custom_boosts: Optional[Dict[str, float]] = None, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, float]]
```

Find documents with optional document-type boosting.

#### find_relevant_concepts

```python
find_relevant_concepts(query_terms: Dict[str, float], layers: Dict[CorticalLayer, HierarchicalLayer], top_n: int = 5) -> List[Tuple[str, float, set]]
```

Stage 1: Find concepts relevant to query terms.

#### multi_stage_rank

```python
multi_stage_rank(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, concept_boost: float = 0.3, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, str, int, int, float, Dict[str, float]]]
```

Multi-stage ranking pipeline for improved RAG performance.

#### multi_stage_rank_documents

```python
multi_stage_rank_documents(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, concept_boost: float = 0.3, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, float, Dict[str, float]]]
```

Multi-stage ranking for documents (without chunk scoring).

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.CONCEPTUAL_KEYWORDS`
- `constants.DOC_TYPE_BOOSTS`
- `constants.IMPLEMENTATION_KEYWORDS`
- `expansion.get_expanded_query_terms`
- ... and 9 more



## search.py

Document Search Module
=====================

Functions for searching and retrieving documents from the corpus.

This module provides:
- Basic document search using TF-IDF scoring
- Fast document sear...


### Functions

#### find_documents_for_query

```python
find_documents_for_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, doc_name_boost: float = 2.0, filter_code_stop_words: bool = True, test_file_penalty: float = 0.8) -> List[Tuple[str, float]]
```

Find documents most relevant to a query using TF-IDF and optional expansion.

#### fast_find_documents

```python
fast_find_documents(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, candidate_multiplier: int = 3, use_code_concepts: bool = True, doc_name_boost: float = 2.0) -> List[Tuple[str, float]]
```

Fast document search using candidate filtering.

#### build_document_index

```python
build_document_index(layers: Dict[CorticalLayer, HierarchicalLayer]) -> Dict[str, Dict[str, float]]
```

Build an optimized inverted index for fast querying.

#### search_with_index

```python
search_with_index(query_text: str, index: Dict[str, Dict[str, float]], tokenizer: Tokenizer, top_n: int = 5) -> List[Tuple[str, float]]
```

Search using a pre-built inverted index.

#### query_with_spreading_activation

```python
query_with_spreading_activation(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 10, max_expansions: int = 8) -> List[Tuple[str, float]]
```

Query with automatic expansion using spreading activation.

#### find_related_documents

```python
find_related_documents(doc_id: str, layers: Dict[CorticalLayer, HierarchicalLayer]) -> List[Tuple[str, float]]
```

Find documents related to a given document via lateral connections.

#### graph_boosted_search

```python
graph_boosted_search(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, pagerank_weight: float = 0.3, proximity_weight: float = 0.2, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None) -> List[Tuple[str, float]]
```

Graph-Boosted BM25 (GB-BM25): Hybrid scoring combining BM25 with graph signals.

### Dependencies

**Standard Library:**

- `code_concepts.get_related_terms`
- `collections.defaultdict`
- `expansion.expand_query`
- `expansion.get_expanded_query_terms`
- `layers.CorticalLayer`
- ... and 6 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
