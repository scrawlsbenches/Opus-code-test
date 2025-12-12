"""
Analogy Completion Module
========================

Functions for analogy completion and semantic relation discovery.

This module provides:
- Analogy completion (a:b::c:?)
- Relation discovery between terms
- Semantic relation navigation
"""

from typing import Dict, List, Tuple, Optional

from ..layers import CorticalLayer, HierarchicalLayer


def find_relation_between(
    term_a: str,
    term_b: str,
    semantic_relations: List[Tuple[str, str, str, float]]
) -> List[Tuple[str, float]]:
    """
    Find semantic relations between two terms.

    Args:
        term_a: Source term
        term_b: Target term
        semantic_relations: List of (t1, relation, t2, weight) tuples

    Returns:
        List of (relation_type, weight) tuples
    """
    relations = []
    for t1, rel_type, t2, weight in semantic_relations:
        if t1 == term_a and t2 == term_b:
            relations.append((rel_type, weight))
        elif t2 == term_a and t1 == term_b:
            # Reverse direction
            relations.append((rel_type, weight * 0.9))  # Slight penalty for reverse

    return sorted(relations, key=lambda x: x[1], reverse=True)


def find_terms_with_relation(
    term: str,
    relation_type: str,
    semantic_relations: List[Tuple[str, str, str, float]],
    direction: str = 'forward'
) -> List[Tuple[str, float]]:
    """
    Find terms connected to a given term by a specific relation type.

    Args:
        term: Source term
        relation_type: Type of relation to follow
        semantic_relations: List of (t1, relation, t2, weight) tuples
        direction: 'forward' (term->x) or 'backward' (x->term)

    Returns:
        List of (target_term, weight) tuples
    """
    results = []
    for t1, rel_type, t2, weight in semantic_relations:
        if rel_type != relation_type:
            continue

        if direction == 'forward' and t1 == term:
            results.append((t2, weight))
        elif direction == 'backward' and t2 == term:
            results.append((t1, weight))

    return sorted(results, key=lambda x: x[1], reverse=True)


def complete_analogy(
    term_a: str,
    term_b: str,
    term_c: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    semantic_relations: List[Tuple[str, str, str, float]],
    embeddings: Optional[Dict[str, List[float]]] = None,
    top_n: int = 5,
    use_embeddings: bool = True,
    use_relations: bool = True
) -> List[Tuple[str, float, str]]:
    """
    Complete an analogy: "a is to b as c is to ?"

    Uses multiple strategies to find the best completion:
    1. Relation matching: Find what relation connects a->b, then find terms with
       the same relation from c
    2. Vector arithmetic: Use embeddings to compute d = c + (b - a)
    3. Pattern matching: Find terms that co-occur with c similar to how b co-occurs with a

    Example:
        "neural" is to "networks" as "knowledge" is to ?
        -> "graphs" (both form compound technical terms with similar structure)

    Args:
        term_a: First term of the known pair
        term_b: Second term of the known pair
        term_c: First term of the query pair
        layers: Dictionary of layers
        semantic_relations: List of (t1, relation, t2, weight) tuples
        embeddings: Optional graph embeddings for vector arithmetic
        top_n: Number of candidates to return
        use_embeddings: Whether to use embedding-based completion
        use_relations: Whether to use relation-based completion

    Returns:
        List of (candidate_term, confidence, method) tuples, where method describes
        which approach found this candidate ('relation', 'embedding', 'pattern')
    """
    layer0 = layers[CorticalLayer.TOKENS]
    candidates: Dict[str, Tuple[float, str]] = {}  # term -> (score, method)

    # Check that terms exist
    if not layer0.get_minicolumn(term_a) or not layer0.get_minicolumn(term_b):
        return []
    if not layer0.get_minicolumn(term_c):
        return []

    # Strategy 1: Relation-based completion
    if use_relations and semantic_relations:
        # Find relation between a and b
        relations_ab = find_relation_between(term_a, term_b, semantic_relations)

        for rel_type, rel_weight in relations_ab:
            # Find terms with same relation from c
            c_targets = find_terms_with_relation(
                term_c, rel_type, semantic_relations, direction='forward'
            )

            for target, target_weight in c_targets:
                # Don't include the input terms
                if target in {term_a, term_b, term_c}:
                    continue

                score = rel_weight * target_weight
                if target not in candidates or candidates[target][0] < score:
                    candidates[target] = (score, f'relation:{rel_type}')

    # Strategy 2: Embedding-based completion (vector arithmetic)
    if use_embeddings and embeddings:
        if term_a in embeddings and term_b in embeddings and term_c in embeddings:
            vec_a = embeddings[term_a]
            vec_b = embeddings[term_b]
            vec_c = embeddings[term_c]

            # d = c + (b - a)  (the analogy vector)
            vec_d = [
                c + (b - a)
                for a, b, c in zip(vec_a, vec_b, vec_c)
            ]

            # Find nearest terms to vec_d
            best_matches = []
            for term, vec in embeddings.items():
                if term in {term_a, term_b, term_c}:
                    continue

                # Cosine similarity
                dot = sum(d * v for d, v in zip(vec_d, vec))
                mag_d = sum(d * d for d in vec_d) ** 0.5
                mag_v = sum(v * v for v in vec) ** 0.5

                if mag_d > 0 and mag_v > 0:
                    similarity = dot / (mag_d * mag_v)
                    best_matches.append((term, similarity))

            # Sort by similarity and add to candidates
            best_matches.sort(key=lambda x: x[1], reverse=True)
            for term, sim in best_matches[:top_n * 2]:
                if sim > 0.5:  # Only include reasonably similar terms
                    if term not in candidates or candidates[term][0] < sim:
                        candidates[term] = (sim, 'embedding')

    # Strategy 3: Pattern matching (co-occurrence structure)
    col_a = layer0.get_minicolumn(term_a)
    col_b = layer0.get_minicolumn(term_b)
    col_c = layer0.get_minicolumn(term_c)

    if col_a and col_b and col_c:
        # Find terms that relate to c similarly to how b relates to a
        # I.e., if b co-occurs strongly with a, find terms that co-occur strongly with c

        a_neighbors = set(col_a.lateral_connections.keys())
        c_neighbors = set(col_c.lateral_connections.keys())

        # Look at c's neighbors that aren't a's neighbors (new context)
        for neighbor_id in c_neighbors:
            neighbor = layer0.get_by_id(neighbor_id)
            if not neighbor:
                continue

            term = neighbor.content
            if term in {term_a, term_b, term_c}:
                continue

            # Score based on how similar the neighbor's connection to c is
            # compared to b's connection to a
            c_weight = col_c.lateral_connections.get(neighbor_id, 0)
            b_to_a_weight = col_a.lateral_connections.get(col_b.id, 0)

            if c_weight > 0 and b_to_a_weight > 0:
                # The term should have similar connection strength pattern
                score = min(c_weight, b_to_a_weight) * 0.5
                if score > 0.1:
                    if term not in candidates or candidates[term][0] < score:
                        candidates[term] = (score, 'pattern')

    # Sort and return top candidates
    results = [
        (term, score, method)
        for term, (score, method) in candidates.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_n]


def complete_analogy_simple(
    term_a: str,
    term_b: str,
    term_c: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    tokenizer: 'Tokenizer',
    semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Simplified analogy completion using only term relationships.

    A lighter version of complete_analogy that doesn't require embeddings.
    Uses bigram patterns and co-occurrence to find analogies.

    Example:
        "neural" is to "networks" as "knowledge" is to ?
        -> Looks for terms that form similar bigrams with "knowledge"

    Args:
        term_a: First term of the known pair
        term_b: Second term of the known pair
        term_c: First term of the query pair
        layers: Dictionary of layers
        tokenizer: Tokenizer instance
        semantic_relations: Optional semantic relations
        top_n: Number of candidates to return

    Returns:
        List of (candidate_term, confidence) tuples
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer1 = layers.get(CorticalLayer.BIGRAMS)

    candidates: Dict[str, float] = {}

    col_a = layer0.get_minicolumn(term_a)
    col_b = layer0.get_minicolumn(term_b)
    col_c = layer0.get_minicolumn(term_c)

    if not col_a or not col_b or not col_c:
        return []

    # Strategy 1: Bigram pattern matching
    if layer1:
        # Find bigrams containing "a b" pattern (bigrams use space separators)
        ab_bigram = f"{term_a} {term_b}"
        ba_bigram = f"{term_b} {term_a}"

        ab_col = layer1.get_minicolumn(ab_bigram)
        ba_col = layer1.get_minicolumn(ba_bigram)

        # If "a b" is a bigram, look for "c ?" bigrams
        if ab_col or ba_col:
            for bigram_col in layer1.minicolumns.values():
                bigram = bigram_col.content
                parts = bigram.split(' ')
                if len(parts) != 2:
                    continue

                first, second = parts

                # Look for bigrams starting with c
                if first == term_c and second not in {term_a, term_b, term_c}:
                    score = bigram_col.pagerank * 0.8
                    if second not in candidates or candidates[second] < score:
                        candidates[second] = score

                # Look for bigrams ending with c
                if second == term_c and first not in {term_a, term_b, term_c}:
                    score = bigram_col.pagerank * 0.6
                    if first not in candidates or candidates[first] < score:
                        candidates[first] = score

    # Strategy 2: Co-occurrence similarity
    # Find terms that co-occur with c like b co-occurs with a
    a_neighbors = col_a.lateral_connections
    c_neighbors = col_c.lateral_connections

    for neighbor_id, c_weight in c_neighbors.items():
        neighbor = layer0.get_by_id(neighbor_id)
        if not neighbor:
            continue

        term = neighbor.content
        if term in {term_a, term_b, term_c}:
            continue

        # Check if this term has similar connection pattern
        score = c_weight * 0.3
        if score > 0.05:
            candidates[term] = candidates.get(term, 0) + score

    # Strategy 3: Semantic relations (if available)
    if semantic_relations:
        relations_ab = find_relation_between(term_a, term_b, semantic_relations)
        for rel_type, rel_weight in relations_ab[:2]:  # Top 2 relations
            c_targets = find_terms_with_relation(
                term_c, rel_type, semantic_relations, direction='forward'
            )
            for target, target_weight in c_targets[:3]:  # Top 3 targets
                if target not in {term_a, term_b, term_c}:
                    score = rel_weight * target_weight
                    candidates[target] = candidates.get(target, 0) + score

    # Sort and return
    results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return results[:top_n]
