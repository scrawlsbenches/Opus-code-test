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
    method: str = 'adjacency'
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Compute embeddings for tokens based on graph structure.
    
    Args:
        layers: Dictionary of layers (needs TOKENS)
        dimensions: Number of embedding dimensions
        method: 'adjacency', 'random_walk', or 'spectral'
        
    Returns:
        Tuple of (embeddings dict, statistics dict)
    """
    layer0 = layers[CorticalLayer.TOKENS]
    
    if method == 'adjacency':
        embeddings = _adjacency_embeddings(layer0, dimensions)
    elif method == 'random_walk':
        embeddings = _random_walk_embeddings(layer0, dimensions)
    elif method == 'spectral':
        embeddings = _spectral_embeddings(layer0, dimensions)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    stats = {
        'method': method,
        'dimensions': dimensions,
        'terms_embedded': len(embeddings)
    }
    
    return embeddings, stats


def _adjacency_embeddings(layer: HierarchicalLayer, dimensions: int) -> Dict[str, List[float]]:
    """Compute embeddings using adjacency to landmark nodes."""
    embeddings: Dict[str, List[float]] = {}
    
    sorted_cols = sorted(layer.minicolumns.values(), key=lambda c: c.pagerank, reverse=True)
    landmarks = sorted_cols[:dimensions]
    
    for col in layer.minicolumns.values():
        vec = [col.lateral_connections.get(lm.id, 0) for lm in landmarks]
        mag = math.sqrt(sum(v*v for v in vec)) + 1e-10
        embeddings[col.content] = [v / mag for v in vec]
    
    return embeddings


def _random_walk_embeddings(
    layer: HierarchicalLayer,
    dimensions: int,
    walks_per_node: int = 10,
    walk_length: int = 40,
    window_size: int = 5
) -> Dict[str, List[float]]:
    """Compute embeddings using random walks (DeepWalk-inspired)."""
    embeddings: Dict[str, List[float]] = {}
    id_to_term = {col.id: col.content for col in layer.minicolumns.values()}
    cooccurrence: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    for col in layer.minicolumns.values():
        for _ in range(walks_per_node):
            walk = _weighted_random_walk(col, layer, walk_length, id_to_term)
            for i, term in enumerate(walk):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        cooccurrence[term][walk[j]] += 1.0
    
    sorted_cols = sorted(layer.minicolumns.values(), key=lambda c: c.pagerank, reverse=True)
    landmarks = [c.content for c in sorted_cols[:dimensions]]
    
    for term in layer.minicolumns:
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


def _spectral_embeddings(layer: HierarchicalLayer, dimensions: int, iterations: int = 100) -> Dict[str, List[float]]:
    """Compute embeddings using spectral methods (graph Laplacian)."""
    embeddings: Dict[str, List[float]] = {}
    terms = list(layer.minicolumns.keys())
    n = len(terms)
    if n == 0:
        return embeddings
    
    term_to_idx = {t: i for i, t in enumerate(terms)}
    adjacency: Dict[int, Dict[int, float]] = defaultdict(dict)
    degrees = [0.0] * n
    
    for term, col in layer.minicolumns.items():
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
