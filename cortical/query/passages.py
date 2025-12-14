"""
Passage Retrieval Module
========================

Functions for retrieving relevant passages from documents.

This module provides:
- Passage retrieval for RAG systems
- Batch passage retrieval
- Integration with chunking and scoring

Chunking functions are in the chunking module.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..tokenizer import Tokenizer

from .expansion import get_expanded_query_terms
from .search import find_documents_for_query
from .definitions import find_definition_passages, DEFINITION_BOOST
from .ranking import get_doc_type_boost, is_conceptual_query
from .chunking import (
    create_chunks,
    create_code_aware_chunks,
    is_code_file,
    precompute_term_cols,
    score_chunk_fast,
    score_chunk,
    # Re-export for backward compatibility
    CODE_BOUNDARY_PATTERN,
    find_code_boundaries,
)


def find_passages_for_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    documents: Dict[str, str],
    top_n: int = 5,
    chunk_size: int = 512,
    overlap: int = 128,
    use_expansion: bool = True,
    doc_filter: Optional[List[str]] = None,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True,
    use_definition_search: bool = True,
    definition_boost: float = DEFINITION_BOOST,
    apply_doc_boost: bool = True,
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    auto_detect_intent: bool = True,
    prefer_docs: bool = False,
    custom_boosts: Optional[Dict[str, float]] = None,
    use_code_aware_chunks: bool = True,
    filter_code_stop_words: bool = True,
    test_file_penalty: float = 0.8
) -> List[Tuple[str, str, int, int, float]]:
    """
    Find text passages most relevant to a query.

    This is the key function for RAG systems - instead of returning document IDs,
    it returns actual text passages with position information for citations.

    For definition queries (e.g., "class Minicolumn", "def compute_pagerank"),
    this function will directly search for the definition pattern and inject
    those results with a high score, ensuring definitions appear in top results.

    For conceptual queries (e.g., "what is PageRank", "explain architecture"),
    documentation passages are boosted to appear higher in results when
    auto_detect_intent=True.

    For code files, semantic chunk boundaries can be used to align chunks
    with class/function definitions rather than fixed character positions.

    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        documents: Dict mapping doc_id to document text
        top_n: Number of passages to return
        chunk_size: Size of each chunk in characters (default 512)
        overlap: Overlap between chunks in characters (default 128)
        use_expansion: Whether to expand query terms
        doc_filter: Optional list of doc_ids to restrict search to
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion (if available)
        use_definition_search: Whether to search for definition patterns (default True)
        definition_boost: Score boost for definition matches (default 5.0)
        apply_doc_boost: Whether to apply document-type boosting (default True)
        doc_metadata: Optional metadata dict {doc_id: {doc_type: ..., ...}}
        auto_detect_intent: Auto-detect conceptual queries and boost docs (default True)
        prefer_docs: Always boost documentation regardless of query type (default False)
        custom_boosts: Optional custom boost factors for doc types
        use_code_aware_chunks: Use semantic boundaries for code files (default True)
        filter_code_stop_words: Filter ubiquitous code tokens (self, def, return)
                                from expansion. Reduces noise in code search. (default True)
        test_file_penalty: Multiplier for test files to rank them lower (default 0.8).
                           Set to 1.0 to disable penalty.

    Returns:
        List of (passage_text, doc_id, start_char, end_char, score) tuples
        ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Determine if we should apply doc-type boosting
    should_boost = apply_doc_boost and (
        prefer_docs or (auto_detect_intent and is_conceptual_query(query_text))
    )

    # Check for definition query and find definition passages
    definition_passages: List[Tuple[str, str, int, int, float]] = []
    if use_definition_search:
        docs_to_search = documents
        if doc_filter:
            docs_to_search = {k: v for k, v in documents.items() if k in doc_filter}
        definition_passages = find_definition_passages(
            query_text, docs_to_search, chunk_size, definition_boost
        )

    # Get expanded query terms
    query_terms = get_expanded_query_terms(
        query_text, layers, tokenizer,
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic,
        filter_code_stop_words=filter_code_stop_words
    )

    if not query_terms and not definition_passages:
        return []

    # If we only have definition results, apply boosting and return
    if not query_terms:
        if should_boost:
            definition_passages = [
                (p[0], p[1], p[2], p[3], p[4] * get_doc_type_boost(p[1], doc_metadata, custom_boosts))
                for p in definition_passages
            ]
            definition_passages.sort(key=lambda x: -x[4])
        return definition_passages[:top_n]

    # Pre-compute minicolumn lookups for query terms (optimization)
    term_cols = precompute_term_cols(query_terms, layer0)

    # Get candidate documents
    if doc_filter:
        # Use provided filter directly as candidates (caller may have pre-boosted)
        # Assign dummy scores since we'll re-score passages anyway
        doc_scores = [(doc_id, 1.0) for doc_id in doc_filter if doc_id in documents]
    else:
        # No filter - get candidates via document search
        doc_scores = find_documents_for_query(
            query_text, layers, tokenizer,
            top_n=min(len(documents), top_n * 3),
            use_expansion=use_expansion,
            semantic_relations=semantic_relations,
            use_semantic=use_semantic,
            filter_code_stop_words=filter_code_stop_words,
            test_file_penalty=test_file_penalty
        )

    # Score passages within candidate documents
    passages: List[Tuple[str, str, int, int, float]] = []

    # Track definition passage locations to avoid duplicates
    def_locations = {(p[1], p[2], p[3]) for p in definition_passages}

    for doc_id, doc_score in doc_scores:
        if doc_id not in documents:
            continue

        text = documents[doc_id]

        # Use code-aware chunking for code files if enabled
        if use_code_aware_chunks and is_code_file(doc_id):
            chunks = create_code_aware_chunks(
                text,
                target_size=chunk_size,
                min_size=max(50, chunk_size // 4),
                max_size=chunk_size * 2
            )
        else:
            chunks = create_chunks(text, chunk_size, overlap)

        # Pre-compute doc-type boost for this document
        doc_type_boost = get_doc_type_boost(doc_id, doc_metadata, custom_boosts) if should_boost else 1.0

        for chunk_text, start_char, end_char in chunks:
            # Skip if this overlaps with a definition passage
            if (doc_id, start_char, end_char) in def_locations:
                continue

            # Use fast scoring with pre-computed lookups
            chunk_tokens = tokenizer.tokenize(chunk_text)
            chunk_score = score_chunk_fast(
                chunk_tokens, query_terms, term_cols, doc_id
            )
            # Combine chunk score with document score for final ranking
            combined_score = chunk_score * (1 + doc_score * 0.1)

            # Apply document-type boost
            combined_score *= doc_type_boost

            passages.append((
                chunk_text,
                doc_id,
                start_char,
                end_char,
                combined_score
            ))

    # Apply doc-type boost to definition passages too
    if should_boost:
        definition_passages = [
            (p[0], p[1], p[2], p[3], p[4] * get_doc_type_boost(p[1], doc_metadata, custom_boosts))
            for p in definition_passages
        ]

    # Combine definition passages with regular passages
    all_passages = definition_passages + passages

    # Sort by score and return top passages
    all_passages.sort(key=lambda x: x[4], reverse=True)
    return all_passages[:top_n]


def find_documents_batch(
    queries: List[str],
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[List[Tuple[str, float]]]:
    """
    Find documents for multiple queries efficiently.

    More efficient than calling find_documents_for_query() multiple times
    because it shares tokenization and expansion caching across queries.

    Args:
        queries: List of search query strings
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of documents to return per query
        use_expansion: Whether to expand query terms
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion

    Returns:
        List of results, one per query. Each result is a list of (doc_id, score) tuples.

    Example:
        >>> queries = ["neural networks", "machine learning", "data processing"]
        >>> results = find_documents_batch(queries, layers, tokenizer, top_n=3)
        >>> for query, docs in zip(queries, results):
        ...     print(f"{query}: {[doc_id for doc_id, _ in docs]}")
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Cache for expanded query terms to avoid redundant computation
    expansion_cache: Dict[str, Dict[str, float]] = {}

    all_results: List[List[Tuple[str, float]]] = []

    for query_text in queries:
        # Check cache first for expansion
        if query_text in expansion_cache:
            query_terms = expansion_cache[query_text]
        else:
            query_terms = get_expanded_query_terms(
                query_text, layers, tokenizer,
                use_expansion=use_expansion,
                semantic_relations=semantic_relations,
                use_semantic=use_semantic
            )
            expansion_cache[query_text] = query_terms

        # Score documents
        doc_scores: Dict[str, float] = defaultdict(float)
        for term, term_weight in query_terms.items():
            col = layer0.get_minicolumn(term)
            if col:
                for doc_id in col.document_ids:
                    tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                    doc_scores[doc_id] += tfidf * term_weight

        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        all_results.append(sorted_docs[:top_n])

    return all_results


def find_passages_batch(
    queries: List[str],
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    documents: Dict[str, str],
    top_n: int = 5,
    chunk_size: int = 512,
    overlap: int = 128,
    use_expansion: bool = True,
    doc_filter: Optional[List[str]] = None,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
) -> List[List[Tuple[str, str, int, int, float]]]:
    """
    Find passages for multiple queries efficiently.

    More efficient than calling find_passages_for_query() multiple times
    because it shares chunk computation and expansion caching across queries.

    Args:
        queries: List of search query strings
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        documents: Dict mapping doc_id to document text
        top_n: Number of passages to return per query
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        use_expansion: Whether to expand query terms
        doc_filter: Optional list of doc_ids to restrict search to
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion

    Returns:
        List of results, one per query. Each result is a list of
        (passage_text, doc_id, start_char, end_char, score) tuples.

    Example:
        >>> queries = ["neural networks", "deep learning"]
        >>> results = find_passages_batch(queries, layers, tokenizer, documents)
        >>> for query, passages in zip(queries, results):
        ...     print(f"{query}: {len(passages)} passages found")
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Pre-compute chunks for all documents to avoid redundant chunking
    doc_chunks_cache: Dict[str, List[Tuple[str, int, int]]] = {}
    for doc_id, text in documents.items():
        if doc_filter and doc_id not in doc_filter:
            continue
        doc_chunks_cache[doc_id] = create_chunks(text, chunk_size, overlap)

    # Cache for expanded query terms
    expansion_cache: Dict[str, Dict[str, float]] = {}

    all_results: List[List[Tuple[str, str, int, int, float]]] = []

    for query_text in queries:
        # Get expanded query terms (with caching)
        if query_text in expansion_cache:
            query_terms = expansion_cache[query_text]
        else:
            query_terms = get_expanded_query_terms(
                query_text, layers, tokenizer,
                use_expansion=use_expansion,
                semantic_relations=semantic_relations,
                use_semantic=use_semantic
            )
            expansion_cache[query_text] = query_terms

        if not query_terms:
            all_results.append([])
            continue

        # Pre-compute minicolumn lookups for query terms (optimization)
        term_cols = precompute_term_cols(query_terms, layer0)

        # Get candidate documents
        doc_scores = find_documents_for_query(
            query_text, layers, tokenizer,
            top_n=min(len(documents), top_n * 3),
            use_expansion=use_expansion,
            semantic_relations=semantic_relations,
            use_semantic=use_semantic
        )

        # Apply document filter
        if doc_filter:
            doc_scores = [(doc_id, score) for doc_id, score in doc_scores if doc_id in doc_filter]

        # Score passages using cached chunks and fast scoring
        passages: List[Tuple[str, str, int, int, float]] = []

        for doc_id, doc_score in doc_scores:
            if doc_id not in doc_chunks_cache:
                continue

            for chunk_text, start_char, end_char in doc_chunks_cache[doc_id]:
                # Use fast scoring with pre-computed lookups
                chunk_tokens = tokenizer.tokenize(chunk_text)
                chunk_score = score_chunk_fast(
                    chunk_tokens, query_terms, term_cols, doc_id
                )
                combined_score = chunk_score * (1 + doc_score * 0.1)
                passages.append((chunk_text, doc_id, start_char, end_char, combined_score))

        passages.sort(key=lambda x: x[4], reverse=True)
        all_results.append(passages[:top_n])

    return all_results
