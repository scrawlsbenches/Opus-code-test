---
title: "NLP Components"
generated: "2025-12-16T17:26:23.831847Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/semantics.py"
  - "/home/user/Opus-code-test/cortical/embeddings.py"
  - "/home/user/Opus-code-test/cortical/tokenizer.py"
tags:
  - architecture
  - modules
  - nlp
---

# NLP Components

Natural language processing components for tokenization and semantics.

## Modules

- **embeddings.py**: Embeddings Module
- **semantics.py**: Semantics Module
- **tokenizer.py**: Tokenizer Module


## embeddings.py

Embeddings Module
=================

Graph-based embeddings for the cortical network.

Implements three methods for computing term embeddings from the
connection graph structure:
1. Adjacency: Direct ...


### Functions

#### compute_graph_embeddings

```python
compute_graph_embeddings(layers: Dict[CorticalLayer, HierarchicalLayer], dimensions: int = 64, method: str = 'adjacency', max_terms: Optional[int] = None) -> Tuple[Dict[str, List[float]], Dict[str, Any]]
```

Compute embeddings for tokens based on graph structure.

#### embedding_similarity

```python
embedding_similarity(embeddings: Dict[str, List[float]], term1: str, term2: str) -> float
```

Compute cosine similarity between two term embeddings.

#### find_similar_by_embedding

```python
find_similar_by_embedding(embeddings: Dict[str, List[float]], term: str, top_n: int = 10) -> List[Tuple[str, float]]
```

Find terms most similar to a given term by embedding.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- `random`
- ... and 5 more



## semantics.py

Semantics Module
================

Corpus-derived semantic relations and retrofitting.

Extracts semantic relationships from co-occurrence patterns,
then uses them to adjust connection weights (retrof...


### Functions

#### extract_pattern_relations

```python
extract_pattern_relations(documents: Dict[str, str], valid_terms: Set[str], min_confidence: float = 0.5) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations using pattern matching on document text.

#### get_pattern_statistics

```python
get_pattern_statistics(relations: List[Tuple[str, str, str, float]]) -> Dict[str, Any]
```

Get statistics about extracted pattern-based relations.

#### extract_corpus_semantics

```python
extract_corpus_semantics(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], tokenizer, window_size: int = 5, min_cooccurrence: int = 2, use_pattern_extraction: bool = True, min_pattern_confidence: float = 0.6, max_similarity_pairs: int = 100000, min_context_keys: int = 3) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations from corpus co-occurrence patterns.

#### retrofit_connections

```python
retrofit_connections(layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: List[Tuple[str, str, str, float]], iterations: int = 10, alpha: float = 0.3) -> Dict[str, Any]
```

Retrofit lateral connections using semantic relations.

#### retrofit_embeddings

```python
retrofit_embeddings(embeddings: Dict[str, List[float]], semantic_relations: List[Tuple[str, str, str, float]], iterations: int = 10, alpha: float = 0.4) -> Dict[str, Any]
```

Retrofit embeddings using semantic relations.

#### get_relation_type_weight

```python
get_relation_type_weight(relation_type: str) -> float
```

Get the weight for a relation type.

#### build_isa_hierarchy

```python
build_isa_hierarchy(semantic_relations: List[Tuple[str, str, str, float]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
```

Build IsA parent-child hierarchy from semantic relations.

#### get_ancestors

```python
get_ancestors(term: str, parents: Dict[str, Set[str]], max_depth: int = 10) -> Dict[str, int]
```

Get all ancestors of a term with their depth in the hierarchy.

#### get_descendants

```python
get_descendants(term: str, children: Dict[str, Set[str]], max_depth: int = 10) -> Dict[str, int]
```

Get all descendants of a term with their depth in the hierarchy.

#### inherit_properties

```python
inherit_properties(semantic_relations: List[Tuple[str, str, str, float]], decay_factor: float = 0.7, max_depth: int = 5) -> Dict[str, Dict[str, Tuple[float, str, int]]]
```

Compute inherited properties for all terms based on IsA hierarchy.

#### compute_property_similarity

```python
compute_property_similarity(term1: str, term2: str, inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]], direct_properties: Optional[Dict[str, Dict[str, float]]] = None) -> float
```

Compute similarity between terms based on shared properties (direct + inherited).

#### apply_inheritance_to_connections

```python
apply_inheritance_to_connections(layers: Dict[CorticalLayer, HierarchicalLayer], inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]], boost_factor: float = 0.3) -> Dict[str, Any]
```

Boost lateral connections between terms that share inherited properties.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.RELATION_WEIGHTS`
- `copy`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 10 more



## tokenizer.py

Tokenizer Module
================

Text tokenization with stemming and word variant support.

Like early visual processing, the tokenizer extracts basic features
(words) from raw input, filtering nois...


### Classes

#### Tokenizer

Text tokenizer with stemming and word variant support.

**Methods:**

- `tokenize`
- `extract_ngrams`
- `stem`
- `get_word_variants`
- `add_word_mapping`

### Functions

#### split_identifier

```python
split_identifier(identifier: str) -> List[str]
```

Split a code identifier into component words.

#### Tokenizer.tokenize

```python
Tokenizer.tokenize(self, text: str, split_identifiers: Optional[bool] = None) -> List[str]
```

Extract tokens from text.

#### Tokenizer.extract_ngrams

```python
Tokenizer.extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]
```

Extract n-grams from token list.

#### Tokenizer.stem

```python
Tokenizer.stem(self, word: str) -> str
```

Apply simple suffix stripping (Porter-lite stemming).

#### Tokenizer.get_word_variants

```python
Tokenizer.get_word_variants(self, word: str) -> List[str]
```

Get related words/variants for query expansion.

#### Tokenizer.add_word_mapping

```python
Tokenizer.add_word_mapping(self, word: str, variants: List[str]) -> None
```

Add a custom word mapping for query expansion.

### Dependencies

**Standard Library:**

- `re`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Set`
- ... and 1 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
