"""
Query Module
============

Query expansion and search functionality.

Provides methods for expanding queries using lateral connections,
concept clusters, and word variants, then searching the corpus
using TF-IDF and graph-based scoring.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .tokenizer import Tokenizer


def expand_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    max_expansions: int = 10,
    use_lateral: bool = True,
    use_concepts: bool = True,
    use_variants: bool = True
) -> Dict[str, float]:
    """
    Expand a query using lateral connections and concept clusters.
    
    This mimics how the brain retrieves related memories when given a cue:
    - Lateral connections: direct word associations (like priming)
    - Concept clusters: semantic category membership
    - Word variants: stemming and synonym mapping
    
    Args:
        query_text: Original query string
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        max_expansions: Maximum number of expansion terms to add
        use_lateral: Include terms from lateral connections
        use_concepts: Include terms from concept clusters
        use_variants: Try word variants when direct match fails
        
    Returns:
        Dict mapping terms to weights (original terms get weight 1.0)
    """
    tokens = tokenizer.tokenize(query_text)
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers.get(CorticalLayer.CONCEPTS)
    
    # Start with original terms at full weight
    expanded: Dict[str, float] = {}
    unmatched_tokens = []
    
    for token in tokens:
        col = layer0.get_minicolumn(token)
        if col:
            expanded[token] = 1.0
        else:
            unmatched_tokens.append(token)
    
    # Try to match unmatched tokens using variants
    if use_variants and unmatched_tokens:
        for token in unmatched_tokens:
            variants = tokenizer.get_word_variants(token)
            for variant in variants:
                col = layer0.get_minicolumn(variant)
                if col and variant not in expanded:
                    expanded[variant] = 0.8
                    break
    
    if not expanded:
        return expanded
    
    candidate_expansions: Dict[str, float] = defaultdict(float)
    
    # Method 1: Lateral connections (direct associations)
    if use_lateral:
        for token in list(expanded.keys()):
            col = layer0.get_minicolumn(token)
            if col:
                sorted_neighbors = sorted(
                    col.lateral_connections.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for neighbor_id, weight in sorted_neighbors:
                    # Use O(1) ID lookup instead of linear search
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor and neighbor.content not in expanded:
                        score = weight * neighbor.pagerank * 0.6
                        candidate_expansions[neighbor.content] = max(
                            candidate_expansions[neighbor.content], score
                        )
    
    # Method 2: Concept cluster membership
    if use_concepts and layer2 and layer2.column_count() > 0:
        for token in list(expanded.keys()):
            col = layer0.get_minicolumn(token)
            if col:
                for concept in layer2.minicolumns.values():
                    if col.id in concept.feedforward_sources:
                        for member_id in concept.feedforward_sources:
                            # Use O(1) ID lookup instead of linear search
                            member = layer0.get_by_id(member_id)
                            if member and member.content not in expanded:
                                score = concept.pagerank * member.pagerank * 0.4
                                candidate_expansions[member.content] = max(
                                    candidate_expansions[member.content], score
                                )
    
    # Select top expansions
    sorted_candidates = sorted(
        candidate_expansions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_expansions]
    
    for term, score in sorted_candidates:
        expanded[term] = score
    
    return expanded


def expand_query_semantic(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    semantic_relations: List[Tuple[str, str, str, float]],
    max_expansions: int = 10
) -> Dict[str, float]:
    """
    Expand query using semantic relations extracted from corpus.
    
    Args:
        query_text: Original query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        semantic_relations: List of (term1, relation, term2, weight) tuples
        max_expansions: Maximum expansions
        
    Returns:
        Dict mapping terms to weights
    """
    tokens = tokenizer.tokenize(query_text)
    layer0 = layers[CorticalLayer.TOKENS]
    
    # Build semantic neighbor lookup
    neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for t1, relation, t2, weight in semantic_relations:
        neighbors[t1].append((t2, weight))
        neighbors[t2].append((t1, weight))
    
    # Start with original terms
    expanded = {t: 1.0 for t in tokens if layer0.get_minicolumn(t)}
    
    if not expanded:
        return expanded
    
    # Add semantic neighbors
    candidates: Dict[str, float] = defaultdict(float)
    for token in list(expanded.keys()):
        for neighbor, weight in neighbors.get(token, []):
            if neighbor not in expanded and layer0.get_minicolumn(neighbor):
                candidates[neighbor] = max(candidates[neighbor], weight * 0.7)
    
    # Take top candidates
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    for term, score in sorted_candidates[:max_expansions]:
        expanded[term] = score
    
    return expanded


def find_documents_for_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    top_n: int = 5,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True
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

    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    if use_expansion:
        # Start with lateral connection expansion
        query_terms = expand_query(query_text, layers, tokenizer, max_expansions=5)

        # Add semantic expansion if available
        if use_semantic and semantic_relations:
            semantic_terms = expand_query_semantic(
                query_text, layers, tokenizer, semantic_relations, max_expansions=5
            )
            # Merge semantic expansions (don't override stronger weights)
            for term, weight in semantic_terms.items():
                if term not in query_terms:
                    query_terms[term] = weight * 0.8  # Slightly discount semantic expansions
                else:
                    # Take the max weight
                    query_terms[term] = max(query_terms[term], weight * 0.8)
    else:
        tokens = tokenizer.tokenize(query_text)
        query_terms = {t: 1.0 for t in tokens}

    # Score each document
    doc_scores: Dict[str, float] = defaultdict(float)

    for term, term_weight in query_terms.items():
        col = layer0.get_minicolumn(term)
        if col:
            for doc_id in col.document_ids:
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                doc_scores[doc_id] += tfidf * term_weight

    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
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


def create_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: Document text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if not text:
        return []

    chunks = []
    stride = max(1, chunk_size - overlap)
    text_len = len(text)

    for start in range(0, text_len, stride):
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))

        if end >= text_len:
            break

    return chunks


def score_chunk(
    chunk_text: str,
    query_terms: Dict[str, float],
    layer0: HierarchicalLayer,
    tokenizer: Tokenizer,
    doc_id: Optional[str] = None
) -> float:
    """
    Score a chunk against query terms using TF-IDF.

    Args:
        chunk_text: Text of the chunk
        query_terms: Dict mapping query terms to weights
        layer0: Token layer for TF-IDF lookups
        tokenizer: Tokenizer instance
        doc_id: Optional document ID for per-document TF-IDF

    Returns:
        Relevance score for the chunk
    """
    chunk_tokens = tokenizer.tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0

    # Count token occurrences in chunk
    token_counts: Dict[str, int] = {}
    for token in chunk_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    score = 0.0
    for term, term_weight in query_terms.items():
        if term in token_counts:
            col = layer0.get_minicolumn(term)
            if col:
                # Use per-document TF-IDF if available, otherwise global
                tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf) if doc_id else col.tfidf
                # Weight by occurrence in chunk and query weight
                score += tfidf * token_counts[term] * term_weight

    # Normalize by chunk length to avoid bias toward longer chunks
    return score / len(chunk_tokens) if chunk_tokens else 0.0


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
    use_semantic: bool = True
) -> List[Tuple[str, str, int, int, float]]:
    """
    Find text passages most relevant to a query.

    This is the key function for RAG systems - instead of returning document IDs,
    it returns actual text passages with position information for citations.

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

    Returns:
        List of (passage_text, doc_id, start_char, end_char, score) tuples
        ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Get expanded query terms
    if use_expansion:
        query_terms = expand_query(query_text, layers, tokenizer, max_expansions=5)
        # Add semantic expansion if available
        if use_semantic and semantic_relations:
            semantic_terms = expand_query_semantic(
                query_text, layers, tokenizer, semantic_relations, max_expansions=5
            )
            for term, weight in semantic_terms.items():
                if term not in query_terms:
                    query_terms[term] = weight * 0.8
                else:
                    query_terms[term] = max(query_terms[term], weight * 0.8)
    else:
        tokens = tokenizer.tokenize(query_text)
        query_terms = {t: 1.0 for t in tokens}

    if not query_terms:
        return []

    # First, get candidate documents (more than we need, since we'll rank passages)
    doc_scores = find_documents_for_query(
        query_text, layers, tokenizer,
        top_n=min(len(documents), top_n * 3),
        use_expansion=use_expansion,
        semantic_relations=semantic_relations,
        use_semantic=use_semantic
    )

    # Apply document filter if provided
    if doc_filter:
        doc_scores = [(doc_id, score) for doc_id, score in doc_scores if doc_id in doc_filter]

    # Score passages within candidate documents
    passages: List[Tuple[str, str, int, int, float]] = []

    for doc_id, doc_score in doc_scores:
        if doc_id not in documents:
            continue

        text = documents[doc_id]
        chunks = create_chunks(text, chunk_size, overlap)

        for chunk_text, start_char, end_char in chunks:
            chunk_score = score_chunk(
                chunk_text, query_terms, layer0, tokenizer, doc_id
            )
            # Combine chunk score with document score for final ranking
            combined_score = chunk_score * (1 + doc_score * 0.1)

            passages.append((
                chunk_text,
                doc_id,
                start_char,
                end_char,
                combined_score
            ))

    # Sort by score and return top passages
    passages.sort(key=lambda x: x[4], reverse=True)
    return passages[:top_n]


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
        if use_expansion:
            if query_text in expansion_cache:
                query_terms = expansion_cache[query_text]
            else:
                query_terms = expand_query(query_text, layers, tokenizer, max_expansions=5)
                if use_semantic and semantic_relations:
                    semantic_terms = expand_query_semantic(
                        query_text, layers, tokenizer, semantic_relations, max_expansions=5
                    )
                    for term, weight in semantic_terms.items():
                        if term not in query_terms:
                            query_terms[term] = weight * 0.8
                        else:
                            query_terms[term] = max(query_terms[term], weight * 0.8)
                expansion_cache[query_text] = query_terms
        else:
            tokens = tokenizer.tokenize(query_text)
            query_terms = {t: 1.0 for t in tokens}

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
        if use_expansion:
            if query_text in expansion_cache:
                query_terms = expansion_cache[query_text]
            else:
                query_terms = expand_query(query_text, layers, tokenizer, max_expansions=5)
                if use_semantic and semantic_relations:
                    semantic_terms = expand_query_semantic(
                        query_text, layers, tokenizer, semantic_relations, max_expansions=5
                    )
                    for term, weight in semantic_terms.items():
                        if term not in query_terms:
                            query_terms[term] = weight * 0.8
                        else:
                            query_terms[term] = max(query_terms[term], weight * 0.8)
                expansion_cache[query_text] = query_terms
        else:
            tokens = tokenizer.tokenize(query_text)
            query_terms = {t: 1.0 for t in tokens}

        if not query_terms:
            all_results.append([])
            continue

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

        # Score passages using cached chunks
        passages: List[Tuple[str, str, int, int, float]] = []

        for doc_id, doc_score in doc_scores:
            if doc_id not in doc_chunks_cache:
                continue

            for chunk_text, start_char, end_char in doc_chunks_cache[doc_id]:
                chunk_score = score_chunk(
                    chunk_text, query_terms, layer0, tokenizer, doc_id
                )
                combined_score = chunk_score * (1 + doc_score * 0.1)
                passages.append((chunk_text, doc_id, start_char, end_char, combined_score))

        passages.sort(key=lambda x: x[4], reverse=True)
        all_results.append(passages[:top_n])

    return all_results
