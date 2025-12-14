"""
Document Search Module
=====================

Functions for searching and retrieving documents from the corpus.

This module provides:
- Basic document search using TF-IDF scoring
- Fast document search with candidate filtering
- Pre-built search index for repeated queries
- Spreading activation search
- Related document discovery
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..tokenizer import Tokenizer
from ..code_concepts import get_related_terms

from .expansion import expand_query, get_expanded_query_terms


def find_documents_for_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True,
    doc_name_boost: float = 2.0,
    filter_code_stop_words: bool = True,
    test_file_penalty: float = 0.8
) -> List[Tuple[str, float]]:
    """
    Find documents most relevant to a query using TF-IDF and optional expansion.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of documents to return
        use_expansion: Whether to expand query terms using lateral connections
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion (if available)
        doc_name_boost: Multiplier for documents whose name matches query terms (default 2.0)
        filter_code_stop_words: Filter ubiquitous code tokens (self, def, return)
                                from expansion candidates. Reduces noise in code search. (default True)
        test_file_penalty: Multiplier for test files to rank them lower (default 0.8).
                           Set to 1.0 to disable penalty.

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer3 = layers[CorticalLayer.DOCUMENTS]

    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic,
        filter_code_stop_words=filter_code_stop_words
    )

    # Score each document
    doc_scores: Dict[str, float] = defaultdict(float)

    for term, term_weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            for doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_scores[doc_id] += tfidf * term_weight

    # Boost documents whose name matches query terms
    if doc_name_boost > 1.0 and doc_scores:
        query_tokens = set(tokenizer.tokenize(query_text))
        max_score = max(doc_scores.values()) if doc_scores else 0.0

        # First pass: identify exact and partial matches
        exact_matches = []
        partial_matches = []

        for doc_id in doc_scores:
            # Use cached tokenized name if available, otherwise tokenize on-the-fly
            doc_col = layer3.get_by_id(f"L3_{doc_id}")
            if doc_col and hasattr(doc_col, 'name_tokens') and doc_col.name_tokens is not None:
                doc_name_tokens = doc_col.name_tokens
            else:
                # Fallback for old data without cached tokens (or mock objects in tests)
                doc_name_tokens = set(tokenizer.tokenize(doc_id.replace('_', ' ')))
            # Count how many query tokens appear in doc name
            matches = len(query_tokens & doc_name_tokens)
            if matches > 0:
                match_ratio = matches / len(query_tokens) if query_tokens else 0

                if match_ratio == 1.0:
                    exact_matches.append(doc_id)
                else:
                    partial_matches.append((doc_id, match_ratio))

        # Apply boosts:
        # - Exact matches: ensure they rank above all non-exact matches
        # - Partial matches: proportional boost
        for doc_id in exact_matches:
            # For exact matches, add max_score to ensure they rank first
            # This guarantees exact match beats all other documents
            doc_scores[doc_id] += max_score * doc_name_boost

        for doc_id, match_ratio in partial_matches:
            # Partial matches use proportional boost
            boost = 1 + (doc_name_boost - 1) * match_ratio
            doc_scores[doc_id] *= boost

    # Apply test file penalty to reduce test file ranking
    if test_file_penalty < 1.0:
        for doc_id in list(doc_scores.keys()):
            # Detect test files by path patterns
            if (doc_id.startswith('tests/') or
                doc_id.startswith('test_') or
                '/test_' in doc_id or
                '/tests/' in doc_id):
                doc_scores[doc_id] *= test_file_penalty

    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
    return sorted_docs[:top_n]


def fast_find_documents(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    candidate_multiplier: int = 3,
    use_code_concepts: bool = True,
    doc_name_boost: float = 2.0
) -> List[Tuple[str, float]]:
    """
    Fast document search using candidate filtering.

    Optimizes search by:
    1. Using set intersection to find candidate documents
    2. Only scoring top candidates fully
    3. Using code concept expansion for better recall

    This is ~2-3x faster than full search on large corpora while
    maintaining similar result quality.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return
        candidate_multiplier: Multiplier for candidate set size
        use_code_concepts: Whether to use code concept expansion
        doc_name_boost: Multiplier for documents whose name matches query terms (default 2.0)

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Tokenize query
    tokens = tokenizer.tokenize(query_text)
    if not tokens:
        return []

    query_tokens = set(tokens)

    # Phase 1: Find candidate documents (fast set operations)
    # Get documents containing ANY query term
    candidate_docs: Dict[str, int] = defaultdict(int)  # doc_id -> match count

    for token in tokens:
        col = layer0.get_minicolumn(token)
        if col:
            for doc_id in col.document_ids:
                candidate_docs[doc_id] += 1

    # If no candidates, try code concept expansion for recall
    if not candidate_docs and use_code_concepts:
        for token in tokens:
            related = get_related_terms(token, max_terms=3)
            for related_term in related:
                col = layer0.get_minicolumn(related_term)
                if col:
                    for doc_id in col.document_ids:
                        candidate_docs[doc_id] += 0.5  # Lower weight for expansion

    # Add documents whose names match query terms to candidates
    # This ensures exact name matches are considered even if content doesn't match
    if doc_name_boost > 1.0:
        layer3 = layers.get(CorticalLayer.DOCUMENTS)
        if layer3:
            for doc_col in layer3.minicolumns.values():
                doc_id = doc_col.content
                doc_name_tokens = set(tokenizer.tokenize(doc_id.replace('_', ' ')))
                matches = len(query_tokens & doc_name_tokens)
                if matches > 0:
                    # Ensure name-matching docs are in candidates
                    # High initial score to prioritize them
                    if doc_id not in candidate_docs:
                        candidate_docs[doc_id] = matches * 2

    if not candidate_docs:
        return []

    # Rank candidates by match count first (fast pre-filter)
    sorted_candidates = sorted(
        candidate_docs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Take top N * multiplier candidates for full scoring
    max_candidates = top_n * candidate_multiplier
    top_candidates = sorted_candidates[:max_candidates]

    # Phase 2: Full scoring only on top candidates
    doc_scores: Dict[str, float] = {}

    for doc_id, match_count in top_candidates:
        score = 0.0
        for token in tokens:
            col = layer0.get_minicolumn(token)
            if col and doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                score += tfidf

        # Boost by match coverage
        coverage_boost = match_count / len(tokens)
        score *= (1 + 0.5 * coverage_boost)

        doc_scores[doc_id] = score

    # Apply document name boost after all scores calculated
    if doc_name_boost > 1.0 and doc_scores:
        max_score = max(doc_scores.values())
        exact_matches = []
        partial_matches = []

        for doc_id in doc_scores:
            doc_name_tokens = set(tokenizer.tokenize(doc_id.replace('_', ' ')))
            matches = len(query_tokens & doc_name_tokens)
            if matches > 0:
                match_ratio = matches / len(query_tokens)

                if match_ratio == 1.0:
                    exact_matches.append(doc_id)
                else:
                    partial_matches.append((doc_id, match_ratio))

        # Exact matches get additive boost to ensure top ranking
        for doc_id in exact_matches:
            doc_scores[doc_id] += max_score * doc_name_boost

        # Partial matches get multiplicative boost
        for doc_id, match_ratio in partial_matches:
            boost = 1 + (doc_name_boost - 1) * match_ratio
            doc_scores[doc_id] *= boost

    # Return top results
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


def build_document_index(
    layers: Dict[CorticalLayer, HierarchicalLayer]
) -> Dict[str, Dict[str, float]]:
    """
    Build an optimized inverted index for fast querying.

    Creates a term -> {doc_id: score} mapping that can be used
    for fast set operations during search.

    Args:
        layers: Dictionary of layers

    Returns:
        Dict mapping terms to {doc_id: tfidf_score} dicts
    """
    layer0 = layers.get(CorticalLayer.TOKENS)
    if not layer0:
        return {}

    index: Dict[str, Dict[str, float]] = {}

    for col in layer0.minicolumns.values():
        term = col.content
        term_index: Dict[str, float] = {}

        for doc_id in col.document_ids:
            tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
            term_index[doc_id] = tfidf

        if term_index:
            index[term] = term_index

    return index


def search_with_index(
    query_text: str,
    index: Dict[str, Dict[str, float]],
    tokenizer: Tokenizer,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Search using a pre-built inverted index.

    This is the fastest search method when the index is cached.

    Args:
        query_text: Search query
        index: Pre-built index from build_document_index()
        tokenizer: Tokenizer instance
        top_n: Number of results to return

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    tokens = tokenizer.tokenize(query_text)
    if not tokens:
        return []

    doc_scores: Dict[str, float] = defaultdict(float)

    for token in tokens:
        if token in index:
            for doc_id, score in index[token].items():
                doc_scores[doc_id] += score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


def query_with_spreading_activation(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 10,
    max_expansions: int = 8
) -> List[Tuple[str, float]]:
    """
    Query with automatic expansion using spreading activation.

    This is like the brain's spreading activation during memory retrieval:
    a cue activates not just direct matches but semantically related concepts.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return
        max_expansions: How many expansion terms to add

    Returns:
        List of (concept, score) tuples ranked by relevance
    """
    expanded_terms = expand_query(
        query_text, layers, tokenizer,
        max_expansions=max_expansions
    )

    if not expanded_terms:
        return []

    layer0 = layers[CorticalLayer.TOKENS]
    activated: Dict[str, float] = {}

    # Activate based on expanded query
    for term, term_weight in expanded_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            # Direct activation
            score = col.pagerank * col.activation * term_weight
            activated[col.content] = activated.get(col.content, 0) + score

            # Spread to neighbors using O(1) ID lookup
            for neighbor_id, conn_weight in col.lateral_connections.items():
                neighbor = layer0.get_by_id(neighbor_id)
                if neighbor:
                    spread_score = neighbor.pagerank * conn_weight * term_weight * 0.3
                    activated[neighbor.content] = activated.get(neighbor.content, 0) + spread_score

    sorted_concepts = sorted(activated.items(), key=lambda x: -x[1])
    return sorted_concepts[:top_n]


def find_related_documents(
    doc_id: str,
    layers: Dict[CorticalLayer, HierarchicalLayer]
) -> List[Tuple[str, float]]:
    """
    Find documents related to a given document via lateral connections.

    Args:
        doc_id: Source document ID
        layers: Dictionary of layers

    Returns:
        List of (doc_id, weight) tuples for related documents
    """
    layer3 = layers.get(CorticalLayer.DOCUMENTS)
    if not layer3:
        return []

    col = layer3.get_minicolumn(doc_id)
    if not col:
        return []

    related = []
    for neighbor_id, weight in col.lateral_connections.items():
        # Use O(1) ID lookup instead of linear search
        neighbor = layer3.get_by_id(neighbor_id)
        if neighbor:
            related.append((neighbor.content, weight))

    return sorted(related, key=lambda x: -x[1])
