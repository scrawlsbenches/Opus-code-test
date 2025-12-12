"""
Type Aliases for the Cortical Text Processor.

This module provides type aliases for complex return types used throughout
the library, making function signatures more readable and maintainable.

Task #114: Add type aliases for complex types

Usage:
    from cortical.types import DocumentScore, PassageResult, SemanticRelation

Example:
    def find_documents(query: str) -> DocumentResults:
        ...
        return results  # List of (doc_id, score) tuples
"""

from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# SCORE TYPES
# =============================================================================

# Basic score tuple: (item_id, score)
DocumentScore = Tuple[str, float]
"""A (doc_id, score) tuple representing a document with its relevance score."""

TermScore = Tuple[str, float]
"""A (term, score) tuple representing a term with its importance score."""

# Result lists
DocumentResults = List[DocumentScore]
"""List of (doc_id, score) tuples, typically sorted by relevance."""

TermResults = List[TermScore]
"""List of (term, score) tuples, typically sorted by importance."""


# =============================================================================
# PASSAGE TYPES
# =============================================================================

PassageResult = Tuple[str, float, str]
"""A (doc_id, score, passage_text) tuple for chunk-level retrieval."""

PassageResults = List[PassageResult]
"""List of (doc_id, score, passage_text) tuples for RAG applications."""

# Passage with position information
PassageWithPosition = Tuple[str, str, int, int, float]
"""A (doc_id, passage_text, start_char, end_char, score) tuple."""

PassageWithPositionResults = List[PassageWithPosition]
"""List of passages with character position information."""

# Passage with expanded terms
PassageWithExpansion = Tuple[str, str, int, int, float, Dict[str, float]]
"""A (doc_id, passage_text, start, end, score, expanded_terms) tuple."""

PassageWithExpansionResults = List[PassageWithExpansion]
"""List of passages with the query expansion used to find them."""


# =============================================================================
# SEMANTIC RELATION TYPES
# =============================================================================

SemanticRelation = Tuple[str, str, str, float]
"""A (term1, relation_type, term2, confidence) semantic relation tuple.

Example: ('dog', 'IsA', 'animal', 0.95)
"""

SemanticRelations = List[SemanticRelation]
"""List of semantic relation tuples extracted from the corpus."""


# =============================================================================
# EMBEDDING TYPES
# =============================================================================

EmbeddingVector = List[float]
"""A dense vector representation of a term."""

EmbeddingDict = Dict[str, EmbeddingVector]
"""Dictionary mapping terms to their embedding vectors."""


# =============================================================================
# METADATA TYPES
# =============================================================================

DocumentMetadata = Dict[str, Any]
"""Arbitrary metadata associated with a document."""

AllDocumentMetadata = Dict[str, DocumentMetadata]
"""Dictionary mapping doc_ids to their metadata."""


# =============================================================================
# GRAPH TYPES
# =============================================================================

ConnectionWeight = float
"""Weight of a connection between minicolumns."""

ConnectionMap = Dict[str, ConnectionWeight]
"""Dictionary mapping target_ids to connection weights."""

IncomingConnections = Dict[str, List[Tuple[str, float]]]
"""Dictionary mapping node_ids to list of (source_id, weight) incoming edges."""


# =============================================================================
# INTENT QUERY TYPES
# =============================================================================

IntentResult = Tuple[str, float, Dict[str, Any]]
"""A (doc_id, score, intent_info) tuple from intent-based search."""

IntentResults = List[IntentResult]
"""List of intent-based search results with metadata."""


# =============================================================================
# BATCH TYPES
# =============================================================================

DocumentInput = Tuple[str, str, Optional[Dict[str, Any]]]
"""A (doc_id, content, metadata) tuple for batch document processing."""

DocumentBatch = List[DocumentInput]
"""List of documents to process in batch."""

BatchResults = List[DocumentResults]
"""Results from batch query processing - one DocumentResults per query."""

BatchPassageResults = List[PassageWithPositionResults]
"""Results from batch passage retrieval - one PassageResults per query."""


# =============================================================================
# SEARCH INDEX TYPES
# =============================================================================

SearchIndex = Dict[str, Dict[str, float]]
"""Pre-built search index mapping terms to doc_id -> score dictionaries."""

TermDocScores = Dict[str, float]
"""Dictionary mapping doc_ids to scores for a single term."""


# =============================================================================
# CLUSTER TYPES
# =============================================================================

ClusterAssignments = Dict[str, str]
"""Dictionary mapping term/node content to cluster_id."""

ClusterQuality = Dict[str, Any]
"""Dictionary with clustering quality metrics (modularity, silhouette, etc.)."""
