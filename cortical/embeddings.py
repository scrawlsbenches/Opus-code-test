"""
Embeddings Module
=================

Graph-based embeddings for the cortical network.

Implements three methods for computing term embeddings from the
connection graph structure:
1. Adjacency: Direct connection weights to landmark nodes
2. Random Walk: DeepWalk-inspired walk co-occurrence
3. Spectral: Graph Laplacian eigenvector approximation
"""

import math
import random
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer


def compute_graph_embeddings(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    dimensions: int = 64,
    method: str = 'adjacency',
    max_terms: Optional[int] = None
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Compute embeddings for tokens based on graph structure.

    Args:
        layers: Dictionary of layers (needs TOKENS)
        dimensions: Number of embedding dimensions
        method: 'adjacency', 'random_walk', 'spectral', or 'fast'
        max_terms: If set, only compute embeddings for top N terms by PageRank.
                   This significantly speeds up computation for large corpora.
                   Recommended: 1000-2000 for large corpora (5000+ tokens).

    Returns:
        Tuple of (embeddings dict, statistics dict)
    """
    layer0 = layers[CorticalLayer.TOKENS]

    # Sample top terms if max_terms is specified
    if max_terms is not None and max_terms < layer0.column_count():
        sorted_cols = sorted(layer0.minicolumns.values(), key=lambda c: c.pagerank, reverse=True)
        sampled_terms = {col.content for col in sorted_cols[:max_terms]}
    else:
        sampled_terms = None

    if method == 'fast':
        # Fast direct adjacency without multi-hop propagation
        embeddings = _fast_adjacency_embeddings(layer0, dimensions, sampled_terms)
    elif method == 'tfidf':
        # TF-IDF based embeddings (best for semantic similarity)
        embeddings = _tfidf_embeddings(layer0, dimensions, sampled_terms)
    elif method == 'adjacency':
        embeddings = _adjacency_embeddings(layer0, dimensions, sampled_terms)
    elif method == 'random_walk':
        embeddings = _random_walk_embeddings(layer0, dimensions, sampled_terms)
    elif method == 'spectral':
        embeddings = _spectral_embeddings(layer0, dimensions, sampled_terms)
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    stats = {
        'method': method,
        'dimensions': dimensions,
        'terms_embedded': len(embeddings),
        'max_terms': max_terms,
        'sampled': max_terms is not None and max_terms < layer0.column_count()
    }

    return embeddings, stats


def _fast_adjacency_embeddings(
    layer: HierarchicalLayer,
    dimensions: int,
    sampled_terms: Optional[set] = None,
    use_idf_weighting: bool = True
) -> Dict[str, List[float]]:
    """
    Fast direct adjacency embeddings without multi-hop propagation.

    Much faster than full adjacency but less expressive. Good for large corpora
    where speed is more important than embedding quality.

    Args:
        layer: Layer to compute embeddings for
        dimensions: Number of embedding dimensions (= number of landmarks)
        sampled_terms: If set, only compute embeddings for these terms
        use_idf_weighting: If True, weight connections by IDF of the target term.
                          This down-weights connections to very common terms,
                          improving embedding quality for diverse corpora.
    """
    embeddings: Dict[str, List[float]] = {}

    sorted_cols = sorted(layer.minicolumns.values(), key=lambda c: c.pagerank, reverse=True)
    landmarks = sorted_cols[:dimensions]
    landmark_ids = {lm.id: i for i, lm in enumerate(landmarks)}

    # Compute IDF weights for landmarks if enabled
    # IDF = log(N / df) where df = number of documents containing the term
    total_docs = len(set(doc_id for col in layer.minicolumns.values() for doc_id in col.document_ids))
    landmark_idf = {}
    if use_idf_weighting and total_docs > 0:
        for lm in landmarks:
            doc_freq = max(1, len(lm.document_ids))
            # Using smoothed IDF: log((N + 1) / (df + 1)) + 1
            landmark_idf[lm.id] = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
    else:
        # No weighting - all landmarks have weight 1.0
        for lm in landmarks:
            landmark_idf[lm.id] = 1.0

    cols_to_process = layer.minicolumns.values()
    if sampled_terms is not None:
        cols_to_process = [c for c in cols_to_process if c.content in sampled_terms]

    for col in cols_to_process:
        vec = [0.0] * dimensions

        # Direct connections only, weighted by landmark IDF
        for lm_id, lm_idx in landmark_ids.items():
            if lm_id in col.lateral_connections:
                raw_weight = col.lateral_connections[lm_id]
                idf_weight = landmark_idf.get(lm_id, 1.0)
                vec[lm_idx] = raw_weight * idf_weight

        # Normalize
        mag = math.sqrt(sum(v*v for v in vec)) + 1e-10
        embeddings[col.content] = [v / mag for v in vec]

    return embeddings


def _tfidf_embeddings(
    layer: HierarchicalLayer,
    dimensions: int,
    sampled_terms: Optional[set] = None
) -> Dict[str, List[float]]:
    """
    TF-IDF based embeddings using document distribution as feature space.

    Each term's embedding is its TF-IDF scores across documents. This produces
    embeddings where semantically similar terms (those appearing in similar
    documents) have high cosine similarity.

    This method is generally better for semantic similarity than graph-based
    methods because:
    1. Terms appearing in similar documents are likely semantically related
    2. TF-IDF naturally down-weights common terms
    3. Embeddings are dense (no sparse landmark issues)

    Args:
        layer: Layer to compute embeddings for
        dimensions: Maximum number of document dimensions (uses top N docs by size)
        sampled_terms: If set, only compute embeddings for these terms
    """
    embeddings: Dict[str, List[float]] = {}

    # Get all documents and sort by document "importance" (term count)
    all_docs = set()
    doc_term_count = defaultdict(int)
    for col in layer.minicolumns.values():
        for doc_id in col.document_ids:
            all_docs.add(doc_id)
            doc_term_count[doc_id] += 1

    # Use top N documents as dimensions (by term coverage)
    sorted_docs = sorted(all_docs, key=lambda d: -doc_term_count[d])
    doc_dims = sorted_docs[:dimensions]
    doc_to_idx = {doc: i for i, doc in enumerate(doc_dims)}

    cols_to_process = layer.minicolumns.values()
    if sampled_terms is not None:
        cols_to_process = [c for c in cols_to_process if c.content in sampled_terms]

    for col in cols_to_process:
        vec = [0.0] * len(doc_dims)

        # Fill in TF-IDF values for documents in our dimension space
        for doc_id, tfidf_score in col.tfidf_per_doc.items():
            if doc_id in doc_to_idx:
                vec[doc_to_idx[doc_id]] = tfidf_score

        # Normalize
        mag = math.sqrt(sum(v*v for v in vec)) + 1e-10
        embeddings[col.content] = [v / mag for v in vec]

    return embeddings


def _adjacency_embeddings(
    layer: HierarchicalLayer,
    dimensions: int,
    sampled_terms: Optional[set] = None,
    propagation_steps: int = 2,
    damping: float = 0.5
) -> Dict[str, List[float]]:
    """
    Compute embeddings using multi-hop adjacency to landmark nodes.

    Improves over simple direct adjacency by propagating through the graph,
    which handles sparse graphs better and produces more meaningful embeddings.

    Args:
        layer: Layer to compute embeddings for
        dimensions: Number of embedding dimensions (= number of landmarks)
        sampled_terms: If set, only compute embeddings for these terms
        propagation_steps: Number of propagation steps (default 2)
        damping: Weight decay per step (default 0.5)
    """
    embeddings: Dict[str, List[float]] = {}

    sorted_cols = sorted(layer.minicolumns.values(), key=lambda c: c.pagerank, reverse=True)
    landmarks = sorted_cols[:dimensions]
    landmark_ids = {lm.id: i for i, lm in enumerate(landmarks)}

    # Build adjacency lookup for efficient propagation
    id_to_col = {col.id: col for col in layer.minicolumns.values()}

    cols_to_process = layer.minicolumns.values()
    if sampled_terms is not None:
        cols_to_process = [c for c in cols_to_process if c.content in sampled_terms]

    for col in cols_to_process:
        vec = [0.0] * dimensions

        # Direct connections (weight = 1.0)
        for lm_id, lm_idx in landmark_ids.items():
            if lm_id in col.lateral_connections:
                vec[lm_idx] += col.lateral_connections[lm_id]

        # Multi-hop propagation: reach landmarks through neighbors
        current_weight = damping
        frontier = list(col.lateral_connections.items())
        visited = {col.id}

        for step in range(propagation_steps):
            next_frontier = []
            for neighbor_id, edge_weight in frontier:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor = id_to_col.get(neighbor_id)
                if not neighbor:
                    continue

                # Check if this neighbor connects to any landmark
                for lm_id, lm_idx in landmark_ids.items():
                    if lm_id in neighbor.lateral_connections:
                        # Add propagated weight (damped by distance)
                        vec[lm_idx] += edge_weight * neighbor.lateral_connections[lm_id] * current_weight

                # Add neighbor's neighbors to next frontier
                for next_id, next_weight in neighbor.lateral_connections.items():
                    if next_id not in visited:
                        next_frontier.append((next_id, edge_weight * next_weight * current_weight))

            frontier = next_frontier
            current_weight *= damping

        # Normalize
        mag = math.sqrt(sum(v*v for v in vec)) + 1e-10
        embeddings[col.content] = [v / mag for v in vec]

    return embeddings


def _random_walk_embeddings(
    layer: HierarchicalLayer,
    dimensions: int,
    sampled_terms: Optional[set] = None,
    walks_per_node: int = 10,
    walk_length: int = 40,
    window_size: int = 5
) -> Dict[str, List[float]]:
    """Compute embeddings using random walks (DeepWalk-inspired)."""
    embeddings: Dict[str, List[float]] = {}
    id_to_term = {col.id: col.content for col in layer.minicolumns.values()}
    cooccurrence: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Only walk from sampled terms if specified
    cols_to_walk = layer.minicolumns.values()
    if sampled_terms is not None:
        cols_to_walk = [c for c in cols_to_walk if c.content in sampled_terms]

    for col in cols_to_walk:
        for _ in range(walks_per_node):
            walk = _weighted_random_walk(col, layer, walk_length, id_to_term)
            for i, term in enumerate(walk):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        cooccurrence[term][walk[j]] += 1.0

    sorted_cols = sorted(layer.minicolumns.values(), key=lambda c: c.pagerank, reverse=True)
    landmarks = [c.content for c in sorted_cols[:dimensions]]

    terms_to_embed = layer.minicolumns.keys() if sampled_terms is None else sampled_terms
    for term in terms_to_embed:
        if term in layer.minicolumns:
            vec = [cooccurrence[term].get(lm, 0) for lm in landmarks]
            mag = math.sqrt(sum(v*v for v in vec)) + 1e-10
            embeddings[term] = [v / mag for v in vec]

    return embeddings


def _weighted_random_walk(start_col, layer: HierarchicalLayer, length: int, id_to_term: Dict[str, str]) -> List[str]:
    """Perform a weighted random walk from a starting column."""
    walk = [start_col.content]
    current = start_col
    
    for _ in range(length - 1):
        if not current.lateral_connections:
            break
        neighbors = list(current.lateral_connections.items())
        total_weight = sum(w for _, w in neighbors)
        if total_weight == 0:
            break
        
        r = random.random() * total_weight
        cumsum = 0.0
        next_id = neighbors[0][0]
        for neighbor_id, weight in neighbors:
            cumsum += weight
            if cumsum >= r:
                next_id = neighbor_id
                break
        
        next_term = id_to_term.get(next_id)
        if next_term and next_term in layer.minicolumns:
            current = layer.minicolumns[next_term]
            walk.append(next_term)
        else:
            break
    
    return walk


def _spectral_embeddings(
    layer: HierarchicalLayer,
    dimensions: int,
    sampled_terms: Optional[set] = None,
    iterations: int = 50
) -> Dict[str, List[float]]:
    """Compute embeddings using spectral methods (graph Laplacian).

    Note: This is inherently O(dimensions × iterations × n²) so it's slow for large graphs.
    When sampled_terms is provided, only those terms get embeddings but the full graph
    structure is still used for computation.
    """
    import warnings

    embeddings: Dict[str, List[float]] = {}

    # If sampling, use only sampled terms for the graph
    if sampled_terms is not None:
        terms = [t for t in layer.minicolumns.keys() if t in sampled_terms]
    else:
        terms = list(layer.minicolumns.keys())

    n = len(terms)
    if n == 0:
        return embeddings

    # Warn about slow computation on large graphs
    if n > 5000:
        warnings.warn(
            f"Spectral embeddings with {n} terms will be slow (O(n²) complexity). "
            f"Consider using max_terms parameter or 'fast'/'tfidf' method instead.",
            RuntimeWarning,
            stacklevel=3  # Points to compute_graph_embeddings() caller
        )

    term_to_idx = {t: i for i, t in enumerate(terms)}
    adjacency: Dict[int, Dict[int, float]] = defaultdict(dict)
    degrees = [0.0] * n

    for term in terms:
        col = layer.minicolumns[term]
        i = term_to_idx[term]
        for neighbor_id, weight in col.lateral_connections.items():
            neighbor = layer.get_by_id(neighbor_id)
            if neighbor and neighbor.content in term_to_idx:
                j = term_to_idx[neighbor.content]
                adjacency[i][j] = weight
                degrees[i] += weight

    degrees = [d if d > 0 else 1.0 for d in degrees]
    actual_dims = min(dimensions, n)
    vectors = []

    for d in range(actual_dims):
        vec = [random.gauss(0, 1) for _ in range(n)]
        for prev in vectors:
            dot = sum(v * p for v, p in zip(vec, prev))
            vec = [v - dot * p for v, p in zip(vec, prev)]
        mag = math.sqrt(sum(v*v for v in vec)) + 1e-10
        vec = [v / mag for v in vec]

        for _ in range(iterations):
            new_vec = [0.0] * n
            for i in range(n):
                for j, weight in adjacency[i].items():
                    norm_weight = weight / math.sqrt(degrees[i] * degrees[j])
                    new_vec[i] -= norm_weight * vec[j]
                new_vec[i] += vec[i]

            for prev in vectors:
                dot = sum(v * p for v, p in zip(new_vec, prev))
                new_vec = [v - dot * p for v, p in zip(new_vec, prev)]
            mag = math.sqrt(sum(v*v for v in new_vec)) + 1e-10
            vec = [v / mag for v in new_vec]

        vectors.append(vec)

    for term in terms:
        i = term_to_idx[term]
        embeddings[term] = [vectors[d][i] if d < len(vectors) else 0.0 for d in range(dimensions)]
    
    return embeddings


def embedding_similarity(embeddings: Dict[str, List[float]], term1: str, term2: str) -> float:
    """Compute cosine similarity between two term embeddings."""
    if term1 not in embeddings or term2 not in embeddings:
        return 0.0
    vec1, vec2 = embeddings[term1], embeddings[term2]
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


def find_similar_by_embedding(embeddings: Dict[str, List[float]], term: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Find terms most similar to a given term by embedding."""
    if term not in embeddings:
        return []
    similarities = [(t, embedding_similarity(embeddings, term, t)) for t in embeddings if t != term]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
