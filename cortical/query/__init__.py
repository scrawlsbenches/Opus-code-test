"""
Query Module
============

Query expansion and search functionality.

This package provides methods for expanding queries using lateral connections,
concept clusters, and word variants, then searching the corpus using TF-IDF
and graph-based scoring.

The package is organized into focused submodules:
- expansion: Query term expansion (lateral, semantic, multihop)
- search: Document search functions
- passages: Passage retrieval for RAG
- chunking: Text chunking functions
- intent: Intent-based query understanding
- definitions: Definition search and boosting
- ranking: Multi-stage ranking and doc type boosting
- analogy: Analogy completion and relation discovery

All public functions are re-exported here for backward compatibility.
"""

# Import all public symbols for backward compatibility
# Existing code using `from cortical.query import ...` will continue to work

# Intent types and parsing
from .intent import (
    ParsedIntent,
    QUESTION_INTENTS,
    ACTION_VERBS,
    parse_intent_query,
    search_by_intent,
)

# Definition search
from .definitions import (
    DefinitionQuery,
    DEFINITION_QUERY_PATTERNS,
    DEFINITION_SOURCE_PATTERNS,
    DEFINITION_BOOST,
    is_definition_query,
    find_definition_in_text,
    find_definition_passages,
    detect_definition_query,
    apply_definition_boost,
    is_test_file,
    boost_definition_documents,
)

# Query expansion
from .expansion import (
    VALID_RELATION_CHAINS,
    score_relation_path,
    expand_query,
    expand_query_semantic,
    expand_query_multihop,
    get_expanded_query_terms,
)

# Document search
from .search import (
    find_documents_for_query,
    fast_find_documents,
    build_document_index,
    search_with_index,
    query_with_spreading_activation,
    find_related_documents,
)

# Document type boosting and ranking
from .ranking import (
    DOC_TYPE_BOOSTS,
    CONCEPTUAL_KEYWORDS,
    IMPLEMENTATION_KEYWORDS,
    is_conceptual_query,
    get_doc_type_boost,
    apply_doc_type_boost,
    find_documents_with_boost,
    find_relevant_concepts,
    multi_stage_rank,
    multi_stage_rank_documents,
)

# Chunking functions
from .chunking import (
    CODE_BOUNDARY_PATTERN,
    create_chunks,
    find_code_boundaries,
    create_code_aware_chunks,
    is_code_file,
    precompute_term_cols,
    score_chunk_fast,
    score_chunk,
)

# Passage retrieval
from .passages import (
    find_passages_for_query,
    find_documents_batch,
    find_passages_batch,
)

# Analogy completion and semantic relations
from .analogy import (
    find_relation_between,
    find_terms_with_relation,
    complete_analogy,
    complete_analogy_simple,
)


__all__ = [
    # Intent
    'ParsedIntent',
    'QUESTION_INTENTS',
    'ACTION_VERBS',
    'parse_intent_query',
    'search_by_intent',
    # Definitions
    'DefinitionQuery',
    'DEFINITION_QUERY_PATTERNS',
    'DEFINITION_SOURCE_PATTERNS',
    'DEFINITION_BOOST',
    'is_definition_query',
    'find_definition_in_text',
    'find_definition_passages',
    'detect_definition_query',
    'apply_definition_boost',
    'is_test_file',
    'boost_definition_documents',
    # Expansion
    'VALID_RELATION_CHAINS',
    'score_relation_path',
    'expand_query',
    'expand_query_semantic',
    'expand_query_multihop',
    'get_expanded_query_terms',
    # Search
    'find_documents_for_query',
    'fast_find_documents',
    'build_document_index',
    'search_with_index',
    'query_with_spreading_activation',
    'find_related_documents',
    # Ranking
    'DOC_TYPE_BOOSTS',
    'CONCEPTUAL_KEYWORDS',
    'IMPLEMENTATION_KEYWORDS',
    'is_conceptual_query',
    'get_doc_type_boost',
    'apply_doc_type_boost',
    'find_documents_with_boost',
    'find_relevant_concepts',
    'multi_stage_rank',
    'multi_stage_rank_documents',
    # Passages
    'CODE_BOUNDARY_PATTERN',
    'create_chunks',
    'find_code_boundaries',
    'create_code_aware_chunks',
    'is_code_file',
    'precompute_term_cols',
    'score_chunk_fast',
    'score_chunk',
    'find_passages_for_query',
    'find_documents_batch',
    'find_passages_batch',
    # Analogy
    'find_relation_between',
    'find_terms_with_relation',
    'complete_analogy',
    'complete_analogy_simple',
]
