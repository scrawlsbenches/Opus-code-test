"""
Query Expansion Module
=====================

Functions for expanding query terms using lateral connections,
semantic relations, and code concept synonyms.

This module provides:
- Basic query expansion using lateral connections
- Semantic relation-based expansion
- Multi-hop inference through relation chains
- Code concept expansion (programming synonyms)
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..tokenizer import Tokenizer, CODE_EXPANSION_STOP_WORDS
from ..code_concepts import expand_code_concepts
from ..config import DEFAULT_CHAIN_VALIDITY


# Valid relation chain patterns for multi-hop inference
# Key: (relation1, relation2) -> validity score (0.0 = invalid, 1.0 = fully valid)
VALID_RELATION_CHAINS = {
    # Transitive hierarchies
    ('IsA', 'IsA'): 1.0,           # dog IsA animal IsA living_thing
    ('PartOf', 'PartOf'): 1.0,     # wheel PartOf car PartOf vehicle
    ('IsA', 'HasProperty'): 0.9,   # dog IsA animal HasProperty alive
    ('PartOf', 'HasProperty'): 0.8,  # wheel PartOf car HasProperty fast

    # Association chains
    ('RelatedTo', 'RelatedTo'): 0.6,
    ('SimilarTo', 'SimilarTo'): 0.7,
    ('CoOccurs', 'CoOccurs'): 0.5,
    ('RelatedTo', 'IsA'): 0.7,
    ('RelatedTo', 'SimilarTo'): 0.7,

    # Causal chains
    ('Causes', 'Causes'): 0.8,
    ('Causes', 'HasProperty'): 0.7,

    # Derivation chains
    ('DerivedFrom', 'DerivedFrom'): 0.8,
    ('DerivedFrom', 'IsA'): 0.7,

    # Usage chains
    ('UsedFor', 'UsedFor'): 0.6,
    ('UsedFor', 'RelatedTo'): 0.5,

    # Antonym - generally invalid for chaining
    ('Antonym', 'Antonym'): 0.3,   # Double negation, weak
    ('Antonym', 'IsA'): 0.1,       # Contradictory
}


def score_relation_path(path: List[str]) -> float:
    """
    Score a relation path by its semantic coherence.

    Args:
        path: List of relation types traversed (e.g., ['IsA', 'HasProperty'])

    Returns:
        Score from 0.0 (invalid) to 1.0 (fully valid)
    """
    if not path:
        return 1.0
    if len(path) == 1:
        return 1.0

    # Compute score as product of consecutive pair validities
    total_score = 1.0
    for i in range(len(path) - 1):
        pair = (path[i], path[i + 1])
        # Check both orderings
        pair_score = VALID_RELATION_CHAINS.get(pair, DEFAULT_CHAIN_VALIDITY)
        total_score *= pair_score

    return total_score


def expand_query(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    max_expansions: int = 10,
    use_lateral: bool = True,
    use_concepts: bool = True,
    use_variants: bool = True,
    use_code_concepts: bool = False,
    filter_code_stop_words: bool = False,
    tfidf_weight: float = 0.7,
    max_expansion_weight: float = 2.0
) -> Dict[str, float]:
    """
    Expand a query using lateral connections and concept clusters.

    This mimics how the brain retrieves related memories when given a cue:
    - Lateral connections: direct word associations (like priming)
    - Concept clusters: semantic category membership
    - Word variants: stemming and synonym mapping
    - Code concepts: programming synonym groups (get/fetch/load)

    Args:
        query_text: Original query string
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        max_expansions: Maximum number of expansion terms to add
        use_lateral: Include terms from lateral connections
        use_concepts: Include terms from concept clusters
        use_variants: Try word variants when direct match fails
        use_code_concepts: Include programming synonym expansions
        filter_code_stop_words: Filter ubiquitous code tokens (self, cls, etc.)
                                from expansion candidates. Useful for code search.
        tfidf_weight: Weight for TF-IDF vs PageRank in lateral expansion scoring.
                      Range [0.0, 1.0]. Default 0.7 favors distinctive terms (TF-IDF).
                      0.0 = use only PageRank (well-connected terms)
                      1.0 = use only TF-IDF (distinctive terms)
        max_expansion_weight: Maximum weight for expanded terms relative to original
                              terms. Prevents single expanded terms from dominating
                              search results. Default 2.0 means expanded terms can
                              have at most 2x the weight of original terms.

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
                # Weight neighbors by TF-IDF * co-occurrence, not just co-occurrence
                # This prioritizes distinctive terms over common ones
                neighbors_with_scores = []
                for neighbor_id, cooccur_weight in col.lateral_connections.items():
                    neighbor = layer0.get_by_id(neighbor_id)
                    if neighbor:
                        # Combine TF-IDF distinctiveness with co-occurrence strength
                        selection_score = cooccur_weight * (neighbor.tfidf + 0.1)
                        neighbors_with_scores.append((neighbor, cooccur_weight, selection_score))

                sorted_neighbors = sorted(
                    neighbors_with_scores,
                    key=lambda x: x[2],  # Sort by TF-IDF-weighted score
                    reverse=True
                )[:5]

                for neighbor, weight, _ in sorted_neighbors:
                    if neighbor.content not in expanded:
                        # Combine TF-IDF (distinctiveness) and PageRank (importance)
                        # tfidf_weight controls the balance between the two
                        term_score = (neighbor.tfidf * tfidf_weight +
                                      neighbor.pagerank * (1.0 - tfidf_weight))
                        score = weight * term_score * 0.6
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

    # Method 3: Code concept groups (programming synonyms)
    if use_code_concepts:
        code_expansions = expand_code_concepts(
            list(expanded.keys()),
            max_expansions_per_term=3,
            weight=0.6
        )
        for term, weight in code_expansions.items():
            if term not in expanded:
                candidate_expansions[term] = max(
                    candidate_expansions[term], weight
                )

    # Filter out ubiquitous code tokens if requested
    if filter_code_stop_words:
        candidate_expansions = {
            term: score for term, score in candidate_expansions.items()
            if term not in CODE_EXPANSION_STOP_WORDS
        }

    # Cap expansion weights to prevent single terms from dominating
    # Max weight is relative to the highest weight of original terms
    if candidate_expansions and max_expansion_weight > 0:
        max_original_weight = max(expanded.values()) if expanded else 1.0
        weight_cap = max_original_weight * max_expansion_weight
        candidate_expansions = {
            term: min(score, weight_cap)
            for term, score in candidate_expansions.items()
        }

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


def expand_query_multihop(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    semantic_relations: List[Tuple[str, str, str, float]],
    max_hops: int = 2,
    max_expansions: int = 15,
    decay_factor: float = 0.5,
    min_path_score: float = 0.2
) -> Dict[str, float]:
    """
    Expand query using multi-hop semantic inference.

    Unlike single-hop expansion that only follows direct connections,
    this follows relation chains to discover semantically related terms
    through transitive relationships.

    Example inference chains:
        "dog" -> IsA -> "animal" -> HasProperty -> "living"
        "car" -> PartOf -> "engine" -> UsedFor -> "transportation"

    Args:
        query_text: Original query string
        layers: Dictionary of layers (needs TOKENS)
        tokenizer: Tokenizer instance
        semantic_relations: List of (term1, relation, term2, weight) tuples
        max_hops: Maximum number of relation hops (default: 2)
        max_expansions: Maximum expansion terms to return
        decay_factor: Weight decay per hop (default: 0.5, so hop2 = 0.25)
        min_path_score: Minimum path validity score to include (default: 0.2)

    Returns:
        Dict mapping terms to weights (original terms get weight 1.0,
        expansions get decayed weights based on hop distance and path validity)

    Example:
        >>> expanded = expand_query_multihop("neural", layers, tokenizer, relations)
        >>> # Hop 1: networks (co-occur), learning (co-occur), brain (RelatedTo)
        >>> # Hop 2: deep (via learning), cortex (via brain), AI (via networks)
    """
    tokens = tokenizer.tokenize(query_text)
    layer0 = layers[CorticalLayer.TOKENS]

    # Start with original terms at full weight
    expanded: Dict[str, float] = {}
    for token in tokens:
        if layer0.get_minicolumn(token):
            expanded[token] = 1.0

    if not expanded or not semantic_relations:
        return expanded

    # Build bidirectional neighbor lookup with relation types
    # neighbors[term] = [(neighbor, relation_type, weight), ...]
    neighbors: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for t1, relation, t2, weight in semantic_relations:
        neighbors[t1].append((t2, relation, weight))
        neighbors[t2].append((t1, relation, weight))

    # Track expansions with their hop distance, weight, and relation path
    # (term, weight, hop, relation_path)
    candidates: Dict[str, Tuple[float, int, List[str]]] = {}

    # BFS-style expansion with hop tracking
    # frontier: [(term, current_weight, hop_count, relation_path)]
    frontier: List[Tuple[str, float, int, List[str]]] = [
        (term, 1.0, 0, []) for term in expanded.keys()
    ]

    visited_at_hop: Dict[str, int] = {term: 0 for term in expanded.keys()}

    while frontier:
        current_term, current_weight, hop, path = frontier.pop(0)

        if hop >= max_hops:
            continue

        next_hop = hop + 1

        for neighbor, relation, rel_weight in neighbors.get(current_term, []):
            # Skip if already in original query terms
            if neighbor in expanded:
                continue

            # Check if term exists in corpus
            if not layer0.get_minicolumn(neighbor):
                continue

            # Skip if we've visited this term at an earlier or equal hop
            if neighbor in visited_at_hop and visited_at_hop[neighbor] <= next_hop:
                continue

            # Compute new path and its validity
            new_path = path + [relation]
            path_score = score_relation_path(new_path)

            if path_score < min_path_score:
                continue

            # Compute weight with decay and path validity
            # weight = base_weight * relation_weight * decay^hop * path_validity
            hop_decay = decay_factor ** next_hop
            new_weight = current_weight * rel_weight * hop_decay * path_score

            # Update candidate if this path gives higher weight
            if neighbor not in candidates or candidates[neighbor][0] < new_weight:
                candidates[neighbor] = (new_weight, next_hop, new_path)
                visited_at_hop[neighbor] = next_hop

                # Add to frontier for further expansion
                if next_hop < max_hops:
                    frontier.append((neighbor, new_weight, next_hop, new_path))

    # Sort candidates by weight and take top expansions
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: x[1][0],  # Sort by weight
        reverse=True
    )[:max_expansions]

    # Add to expanded dict
    for term, (weight, hop, path) in sorted_candidates:
        expanded[term] = weight

    return expanded


def get_expanded_query_terms(
    query_text: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: Tokenizer,
    use_expansion: bool = True,
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    use_semantic: bool = True,
    max_expansions: int = 5,
    semantic_discount: float = 0.8,
    filter_code_stop_words: bool = False
) -> Dict[str, float]:
    """
    Get expanded query terms with optional semantic expansion.

    This is a helper function that consolidates query expansion logic used
    by multiple search functions. It handles:
    - Lateral connection expansion via expand_query()
    - Semantic relation expansion via expand_query_semantic()
    - Merging of expansion results with appropriate weighting

    Args:
        query_text: Original query string
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        use_expansion: Whether to expand query terms using lateral connections
        semantic_relations: Optional list of semantic relations for expansion
        use_semantic: Whether to use semantic relations for expansion
        max_expansions: Maximum expansion terms per method (default 5)
        semantic_discount: Weight multiplier for semantic expansions (default 0.8)
        filter_code_stop_words: Filter ubiquitous code tokens (self, cls, etc.)
                                from expansion candidates. Useful for code search.

    Returns:
        Dict mapping terms to weights (original terms get weight 1.0,
        expansions get lower weights based on connection strength)

    Example:
        >>> terms = get_expanded_query_terms("neural networks", layers, tokenizer)
        >>> # Returns: {'neural': 1.0, 'networks': 1.0, 'deep': 0.3, 'learning': 0.25, ...}
    """
    if use_expansion:
        # Start with lateral connection expansion
        query_terms = expand_query(
            query_text, layers, tokenizer,
            max_expansions=max_expansions,
            filter_code_stop_words=filter_code_stop_words
        )

        # Add semantic expansion if available
        if use_semantic and semantic_relations:
            semantic_terms = expand_query_semantic(
                query_text, layers, tokenizer, semantic_relations, max_expansions=max_expansions
            )
            # Merge semantic expansions (don't override stronger weights)
            for term, weight in semantic_terms.items():
                if term not in query_terms:
                    query_terms[term] = weight * semantic_discount
                else:
                    # Take the max weight
                    query_terms[term] = max(query_terms[term], weight * semantic_discount)
    else:
        tokens = tokenizer.tokenize(query_text)
        query_terms = {t: 1.0 for t in tokens}

    return query_terms
