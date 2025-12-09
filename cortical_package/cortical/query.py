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
                    if neighbor_id in layer0.minicolumns:
                        neighbor = layer0.minicolumns[neighbor_id]
                        if neighbor.content not in expanded:
                            score = weight * neighbor.pagerank * 0.6
                            candidate_expansions[neighbor.content] = max(
                                candidate_expansions[neighbor.content], score
                            )
                    else:
                        # Look up by ID
                        for c in layer0.minicolumns.values():
                            if c.id == neighbor_id and c.content not in expanded:
                                score = weight * c.pagerank * 0.6
                                candidate_expansions[c.content] = max(
                                    candidate_expansions[c.content], score
                                )
                                break
    
    # Method 2: Concept cluster membership
    if use_concepts and layer2 and layer2.column_count() > 0:
        for token in list(expanded.keys()):
            col = layer0.get_minicolumn(token)
            if col:
                for concept in layer2.minicolumns.values():
                    if col.id in concept.feedforward_sources:
                        for member_id in concept.feedforward_sources:
                            if member_id in layer0.minicolumns:
                                member = layer0.minicolumns[member_id]
                                if member.content not in expanded:
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
    use_expansion: bool = True
) -> List[Tuple[str, float]]:
    """
    Find documents most relevant to a query using TF-IDF and optional expansion.
    
    Args:
        query_text: Search query
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        top_n: Number of documents to return
        use_expansion: Whether to expand query terms
        
    Returns:
        List of (doc_id, score) tuples ranked by relevance
    """
    layer0 = layers[CorticalLayer.TOKENS]
    
    if use_expansion:
        query_terms = expand_query(query_text, layers, tokenizer, max_expansions=5)
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
            
            # Spread to neighbors
            for neighbor_id, conn_weight in col.lateral_connections.items():
                if neighbor_id in layer0.minicolumns:
                    neighbor = layer0.minicolumns[neighbor_id]
                    spread_score = neighbor.pagerank * conn_weight * term_weight * 0.3
                    activated[neighbor.content] = activated.get(neighbor.content, 0) + spread_score
                else:
                    for c in layer0.minicolumns.values():
                        if c.id == neighbor_id:
                            spread_score = c.pagerank * conn_weight * term_weight * 0.3
                            activated[c.content] = activated.get(c.content, 0) + spread_score
                            break
    
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
        if neighbor_id in layer3.minicolumns:
            neighbor = layer3.minicolumns[neighbor_id]
            related.append((neighbor.content, weight))
        else:
            for c in layer3.minicolumns.values():
                if c.id == neighbor_id:
                    related.append((c.content, weight))
                    break
    
    return sorted(related, key=lambda x: -x[1])
