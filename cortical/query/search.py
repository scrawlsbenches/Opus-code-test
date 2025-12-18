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


def _apply_document_name_boost(
    doc_scores: Dict[str, float],
    query_tokens: set,
    tokenizer: 'Tokenizer',
    doc_name_boost: float,
    layer3: Optional['HierarchicalLayer'] = None
) -> None:
    """
    Apply document name matching boost to scores in-place.

    Identifies exact and partial matches between query tokens and document names,
    then applies multiplicative (partial) or additive (exact) boosts.

    Args:
        doc_scores: Dictionary of doc_id -> score (modified in-place)
        query_tokens: Set of query tokens to match
        tokenizer: Tokenizer instance for tokenizing document names
        doc_name_boost: Boost factor (only applied if > 1.0)
        layer3: Optional document layer for cached name tokens
    """
    if doc_name_boost <= 1.0 or not doc_scores or not query_tokens:
        return

    max_score = max(doc_scores.values())
    exact_matches = []
    partial_matches = []

    for doc_id in doc_scores:
        # Try to get cached tokens from layer3 if available
        doc_name_tokens = None
        if layer3:
            doc_col = layer3.get_by_id(f"L3_{doc_id}")
            if doc_col and hasattr(doc_col, 'name_tokens') and doc_col.name_tokens is not None:
                doc_name_tokens = doc_col.name_tokens

        if doc_name_tokens is None:
            doc_name_tokens = set(tokenizer.tokenize(doc_id.replace('_', ' ')))

        matches = len(query_tokens & doc_name_tokens)
        if matches > 0:
            match_ratio = matches / len(query_tokens)
            if match_ratio == 1.0:
                exact_matches.append(doc_id)
            else:
                partial_matches.append((doc_id, match_ratio))

    # Apply exact match additive boost
    for doc_id in exact_matches:
        doc_scores[doc_id] += max_score * doc_name_boost

    # Apply partial match multiplicative boost
    for doc_id, match_ratio in partial_matches:
        boost = 1 + (doc_name_boost - 1) * match_ratio
        doc_scores[doc_id] *= boost


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
    if doc_name_boost > 1.0:
        query_tokens = set(tokenizer.tokenize(query_text))
        _apply_document_name_boost(doc_scores, query_tokens, tokenizer, doc_name_boost, layer3)

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
    _apply_document_name_boost(doc_scores, query_tokens, tokenizer, doc_name_boost)

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


def graph_boosted_search(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    pagerank_weight: float = 0.3,
    proximity_weight: float = 0.2,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None
) -> List[Tuple[str, float]]:
    """
    Graph-Boosted BM25 (GB-BM25): Hybrid scoring combining BM25 with graph signals.

    This creative algorithm combines multiple signals:
    1. BM25/TF-IDF base score (term relevance)
    2. PageRank boost (matched term importance)
    3. Proximity boost (query terms connected in graph)
    4. Coverage boost (documents with more unique query term matches)

    This approach is designed for code search where:
    - Important functions/classes should rank higher (PageRank)
    - Related concepts should boost each other (graph connections)
    - Comprehensive matches beat partial matches (coverage)

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of results to return
        pagerank_weight: Weight for PageRank boost (0-1, default 0.3)
        proximity_weight: Weight for term proximity boost (0-1, default 0.2)
        use_expansion: Whether to use query expansion
        semantic_relations: Optional semantic relations for expansion

    Returns:
        List of (doc_id, score) tuples ranked by combined relevance

    Raises:
        ValueError: If pagerank_weight or proximity_weight not in [0.0, 1.0]
    """
    # Validate weight parameters
    if not (0.0 <= pagerank_weight <= 1.0):
        raise ValueError(f"pagerank_weight must be in [0.0, 1.0], got {pagerank_weight}")
    if not (0.0 <= proximity_weight <= 1.0):
        raise ValueError(f"proximity_weight must be in [0.0, 1.0], got {proximity_weight}")

    layer0 = layers[CorticalLayer.TOKENS]
    layer3 = layers[CorticalLayer.DOCUMENTS]

    # Get expanded query terms
    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=True
    )

    if not query_terms:
        return []

    # Phase 1: Compute base BM25/TF-IDF scores per document
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_term_matches: Dict[str, set] = defaultdict(set)  # Track unique term matches
    doc_pagerank_sum: Dict[str, float] = defaultdict(float)  # Sum of PageRank for matched terms

    for term, term_weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            # Get term's PageRank importance
            term_pagerank = getattr(col, 'pagerank', 0.0) or 0.0

            for doc_id in col.document_ids:
                # Base BM25/TF-IDF score
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_scores[doc_id] += tfidf * term_weight

                # Track term match for coverage
                doc_term_matches[doc_id].add(term)

                # Accumulate PageRank boost
                doc_pagerank_sum[doc_id] += term_pagerank * term_weight

    if not doc_scores:
        return []

    # Phase 2: Compute proximity boost using lateral connections
    # Boost documents where query terms are connected in the graph
    proximity_scores: Dict[str, float] = defaultdict(float)

    original_tokens = tokenizer.tokenize(query_text)
    if len(original_tokens) > 1:
        # Check if query terms have lateral connections to each other
        for i, t1 in enumerate(original_tokens):
            col1 = layer0.get_minicolumn(t1)
            if not col1:
                continue

            for t2 in original_tokens[i+1:]:
                col2 = layer0.get_minicolumn(t2)
                if not col2:
                    continue

                # Check for connection between terms
                conn_weight = col1.lateral_connections.get(col2.id, 0.0)
                if conn_weight > 0:
                    # Boost documents containing both terms
                    shared_docs = col1.document_ids & col2.document_ids
                    for doc_id in shared_docs:
                        proximity_scores[doc_id] += conn_weight

    # Phase 3: Combine all signals
    max_base_score = max(doc_scores.values()) if doc_scores else 1.0
    max_pagerank = max(doc_pagerank_sum.values()) if doc_pagerank_sum else 1.0
    max_proximity = max(proximity_scores.values()) if proximity_scores else 1.0

    final_scores: Dict[str, float] = {}
    num_query_terms = len(set(original_tokens))

    for doc_id, base_score in doc_scores.items():
        # Normalize base score
        norm_base = base_score / max_base_score if max_base_score > 0 else 0

        # Normalize PageRank boost
        pagerank_boost = doc_pagerank_sum.get(doc_id, 0.0)
        norm_pagerank = pagerank_boost / max_pagerank if max_pagerank > 0 else 0

        # Normalize proximity boost
        prox_boost = proximity_scores.get(doc_id, 0.0)
        norm_proximity = prox_boost / max_proximity if max_proximity > 0 else 0

        # Coverage boost: reward documents matching more unique query terms
        coverage = len(doc_term_matches.get(doc_id, set())) / num_query_terms if num_query_terms > 0 else 0

        # Combine signals with weights
        # Base score dominates, with boosts from graph signals
        combined = (
            (1 - pagerank_weight - proximity_weight) * norm_base +
            pagerank_weight * norm_pagerank +
            proximity_weight * norm_proximity
        )

        # Apply coverage multiplier (0.5 to 1.5 range)
        coverage_mult = 0.5 + coverage

        # Final score preserves relative magnitude
        final_scores[doc_id] = combined * coverage_mult * max_base_score

    sorted_docs = sorted(final_scores.items(), key=lambda x: -x[1])
    return sorted_docs[:top_n]
