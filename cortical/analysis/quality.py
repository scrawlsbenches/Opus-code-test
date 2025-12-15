"""
Clustering quality metrics.

Contains:
- compute_clustering_quality: Comprehensive quality evaluation (modularity, silhouette, balance)
- _compute_modularity: Modularity Q metric
- _compute_silhouette: Silhouette score for cluster coherence
- _compute_cluster_balance: Gini coefficient for size balance
- _generate_quality_assessment: Human-readable quality interpretation
- _modularity_core: Pure modularity algorithm for unit testing
- _silhouette_core: Pure silhouette algorithm for unit testing
"""

from typing import Dict, List, Any
from collections import defaultdict
import random

from ..layers import CorticalLayer, HierarchicalLayer
from .utils import _doc_similarity


def _modularity_core(
    adjacency: Dict[str, Dict[str, float]],
    community: Dict[str, int]
) -> float:
    """
    Compute modularity Q for a given community assignment.

    Modularity measures the density of connections within communities
    compared to connections between communities.

    Q = (1/2m) * Σ [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

    Args:
        adjacency: Adjacency dict mapping node to {neighbor: weight}
        community: Dict mapping node to community_id

    Returns:
        Modularity score between -0.5 and 1 (typically 0 to 0.7)
        - Q > 0.3: Good community structure
        - Q > 0.5: Strong community structure

    Example:
        >>> adj = {"a": {"b": 1.0}, "b": {"a": 1.0}, "c": {"d": 1.0}, "d": {"c": 1.0}}
        >>> comm = {"a": 0, "b": 0, "c": 1, "d": 1}
        >>> q = _modularity_core(adj, comm)
        >>> assert q > 0.3  # Good separation
    """
    nodes = list(adjacency.keys())
    if not nodes:
        return 0.0

    # Compute m (total edge weight / 2)
    total_weight = sum(
        sum(neighbors.values())
        for neighbors in adjacency.values()
    ) / 2.0

    if total_weight == 0:
        return 0.0

    m = total_weight

    # Compute degree of each node
    k = {node: sum(adjacency[node].values()) for node in nodes}

    # Compute modularity
    q = 0.0
    for i in nodes:
        for j, weight in adjacency[i].items():
            if j in community:
                if community[i] == community[j]:
                    q += weight - (k[i] * k[j]) / (2 * m)

    return q / (2 * m)


def _silhouette_core(
    distances: Dict[str, Dict[str, float]],
    labels: Dict[str, int]
) -> float:
    """
    Compute silhouette score for a clustering.

    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Range is -1 to 1, higher is better.

    Args:
        distances: Distance matrix as dict of dicts: distances[i][j] = distance from i to j
        labels: Dict mapping node to cluster_id

    Returns:
        Average silhouette score across all nodes (-1 to 1)
        - > 0.5: Strong clustering
        - 0.25-0.5: Reasonable clustering
        - < 0.25: Weak or no structure

    Example:
        >>> # Two tight clusters far apart
        >>> distances = {
        ...     "a": {"b": 0.1, "c": 0.9, "d": 0.9},
        ...     "b": {"a": 0.1, "c": 0.9, "d": 0.9},
        ...     "c": {"a": 0.9, "b": 0.9, "d": 0.1},
        ...     "d": {"a": 0.9, "b": 0.9, "c": 0.1}
        ... }
        >>> labels = {"a": 0, "b": 0, "c": 1, "d": 1}
        >>> s = _silhouette_core(distances, labels)
        >>> assert s > 0.5  # Strong clustering
    """
    if not labels or len(set(labels.values())) < 2:
        return 0.0

    nodes = list(labels.keys())
    silhouettes = []

    # Group nodes by cluster
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, cluster in labels.items():
        clusters[cluster].append(node)

    for node in nodes:
        my_cluster = labels[node]
        my_cluster_nodes = [n for n in clusters[my_cluster] if n != node]

        # a = average distance to nodes in same cluster
        if my_cluster_nodes:
            a = sum(distances.get(node, {}).get(other, 0.0) for other in my_cluster_nodes)
            a /= len(my_cluster_nodes)
        else:
            a = 0.0

        # b = minimum average distance to nodes in any other cluster
        b = float('inf')
        for other_cluster, other_nodes in clusters.items():
            if other_cluster != my_cluster and other_nodes:
                avg_dist = sum(
                    distances.get(node, {}).get(other, 0.0)
                    for other in other_nodes
                ) / len(other_nodes)
                b = min(b, avg_dist)

        if b == float('inf'):
            b = 0.0

        # Silhouette for this node
        if max(a, b) > 0:
            s = (b - a) / max(a, b)
        else:
            s = 0.0

        silhouettes.append(s)

    return sum(silhouettes) / len(silhouettes) if silhouettes else 0.0


def compute_clustering_quality(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    sample_size: int = 500
) -> Dict[str, Any]:
    """
    Compute clustering quality metrics for the concept layer.

    Calculates modularity, silhouette score, and balance (Gini coefficient)
    to evaluate how well the clustering algorithm has performed.

    Args:
        layers: Dictionary of hierarchical layers
        sample_size: Max number of tokens to sample for silhouette calculation
                    (full calculation is O(n²), sampling keeps it tractable)

    Returns:
        Dictionary with:
        - modularity: float (-1 to 1, higher is better, >0.3 is good)
        - silhouette: float (-1 to 1, higher is better, >0.25 is reasonable)
        - balance: float (0 to 1, 0 = perfectly balanced, 1 = all in one cluster)
        - num_clusters: int
        - quality_assessment: str (interpretation of the metrics)

    Example:
        >>> quality = compute_clustering_quality(processor.layers)
        >>> print(f"Modularity: {quality['modularity']:.3f}")
        >>> print(quality['quality_assessment'])
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers[CorticalLayer.CONCEPTS]

    num_clusters = layer2.column_count()

    if layer0.column_count() == 0 or num_clusters == 0:
        return {
            'modularity': 0.0,
            'silhouette': 0.0,
            'balance': 1.0,
            'num_clusters': 0,
            'quality_assessment': 'No clusters to evaluate'
        }

    # Compute all metrics
    modularity = _compute_modularity(layer0, layer2)
    silhouette = _compute_silhouette(layer0, layer2, sample_size)
    balance = _compute_cluster_balance(layer2)

    # Generate quality assessment
    assessment = _generate_quality_assessment(modularity, silhouette, balance, num_clusters)

    return {
        'modularity': modularity,
        'silhouette': silhouette,
        'balance': balance,
        'num_clusters': num_clusters,
        'quality_assessment': assessment
    }


def _compute_modularity(
    layer0: HierarchicalLayer,
    layer2: HierarchicalLayer
) -> float:
    """
    Compute the modularity Q of the current clustering.

    Modularity measures the density of connections within clusters
    compared to connections between clusters.

    Q = (1/2m) * Σ [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

    where:
    - m = total edge weight
    - A_ij = edge weight between i and j
    - k_i = degree of node i
    - δ(c_i, c_j) = 1 if nodes i and j are in the same community

    Returns:
        Modularity score between -0.5 and 1 (typically 0 to 0.7)
        - Q > 0.3: Good community structure
        - Q > 0.5: Strong community structure
    """
    # Build token -> cluster mapping
    token_to_cluster: Dict[str, str] = {}
    for cluster_col in layer2.minicolumns.values():
        cluster_id = cluster_col.content
        for token_id in cluster_col.feedforward_connections:
            token_col = layer0.get_by_id(token_id)
            if token_col:
                token_to_cluster[token_col.content] = cluster_id

    # Compute total edge weight m
    total_weight = 0.0
    for col in layer0.minicolumns.values():
        for _, weight in col.lateral_connections.items():
            total_weight += weight

    m = total_weight / 2.0  # Each edge counted twice

    if m == 0:
        return 0.0

    # Compute node degrees k
    degrees: Dict[str, float] = {}
    for content, col in layer0.minicolumns.items():
        degrees[content] = sum(col.lateral_connections.values())

    # Compute modularity Q
    q = 0.0
    for content, col in layer0.minicolumns.items():
        c_i = token_to_cluster.get(content)
        if c_i is None:
            continue

        k_i = degrees.get(content, 0.0)

        for neighbor_id, weight in col.lateral_connections.items():
            neighbor_col = layer0.get_by_id(neighbor_id)
            if neighbor_col is None:
                continue

            neighbor_content = neighbor_col.content
            c_j = token_to_cluster.get(neighbor_content)
            if c_j is None:
                continue

            k_j = degrees.get(neighbor_content, 0.0)

            # δ(c_i, c_j) - same cluster indicator
            if c_i == c_j:
                # A_ij - k_i*k_j/(2m)
                q += weight - (k_i * k_j) / (2 * m)

    return q / (2 * m)


def _compute_silhouette(
    layer0: HierarchicalLayer,
    layer2: HierarchicalLayer,
    sample_size: int = 500
) -> float:
    """
    Compute silhouette score for the clustering.

    For each token, silhouette measures how similar it is to its own cluster
    compared to the nearest other cluster.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    where:
    - a(i) = mean distance to other points in same cluster
    - b(i) = mean distance to points in nearest cluster

    For our representation, distance = 1 - document_cooccurrence_similarity
    where similarity is based on shared documents (Jaccard on document sets).
    This produces more meaningful silhouette scores than connection-based
    similarity because tokens in the same semantic cluster tend to appear
    in the same documents.

    Returns:
        Average silhouette score between -1 and 1
        - s > 0.5: Strong cluster structure
        - s > 0.25: Reasonable structure
        - s < 0: Poor clustering
    """
    if layer2.column_count() < 2:
        return 0.0  # Need at least 2 clusters

    # Build cluster membership
    token_to_cluster: Dict[str, str] = {}
    cluster_tokens: Dict[str, List[str]] = defaultdict(list)

    for cluster_col in layer2.minicolumns.values():
        cluster_id = cluster_col.content
        for token_id in cluster_col.feedforward_connections:
            token_col = layer0.get_by_id(token_id)
            if token_col:
                token_to_cluster[token_col.content] = cluster_id
                cluster_tokens[cluster_id].append(token_col.content)

    # Skip clusters with < 2 tokens
    valid_clusters = {k: v for k, v in cluster_tokens.items() if len(v) >= 2}
    if len(valid_clusters) < 2:
        return 0.0

    # Get all tokens in valid clusters
    all_tokens = []
    for tokens in valid_clusters.values():
        all_tokens.extend(tokens)

    if len(all_tokens) == 0:
        return 0.0

    # Sample if too many tokens (silhouette is O(n²))
    if len(all_tokens) > sample_size:
        all_tokens = random.sample(all_tokens, sample_size)

    # Build document sets for sampled tokens
    # Document set: frozenset of document IDs for this token
    token_docs: Dict[str, frozenset] = {}
    for token in all_tokens:
        col = layer0.get_minicolumn(token)
        if col and col.document_ids:
            token_docs[token] = frozenset(col.document_ids)

    # Compute silhouette for each token
    silhouette_sum = 0.0
    count = 0

    for token in all_tokens:
        if token not in token_to_cluster or token not in token_docs:
            continue

        my_cluster = token_to_cluster[token]
        my_docs = token_docs[token]

        if my_cluster not in valid_clusters or not my_docs:
            continue

        # a(i): mean distance to same-cluster tokens
        same_cluster = [t for t in valid_clusters[my_cluster] if t != token and t in token_docs]
        if not same_cluster:
            continue

        a_i = 0.0
        for other in same_cluster:
            sim = _doc_similarity(my_docs, token_docs[other])
            a_i += 1.0 - sim  # Distance = 1 - similarity
        a_i /= len(same_cluster)

        # b(i): mean distance to nearest other cluster
        b_i = float('inf')
        for other_cluster, other_tokens in valid_clusters.items():
            if other_cluster == my_cluster:
                continue

            other_tokens_filtered = [t for t in other_tokens if t in token_docs]
            if not other_tokens_filtered:
                continue

            cluster_dist = 0.0
            for other in other_tokens_filtered:
                sim = _doc_similarity(my_docs, token_docs[other])
                cluster_dist += 1.0 - sim
            cluster_dist /= len(other_tokens_filtered)

            b_i = min(b_i, cluster_dist)

        if b_i == float('inf'):
            continue

        # Silhouette coefficient
        max_ab = max(a_i, b_i)
        if max_ab > 0:
            s_i = (b_i - a_i) / max_ab
            silhouette_sum += s_i
            count += 1

    return silhouette_sum / count if count > 0 else 0.0


def _compute_cluster_balance(layer2: HierarchicalLayer) -> float:
    """
    Compute Gini coefficient for cluster size balance.

    Returns:
        Gini coefficient (0 = perfectly balanced, 1 = all in one cluster)
    """
    cluster_sizes = [
        len(col.feedforward_connections)
        for col in layer2.minicolumns.values()
    ]

    if not cluster_sizes or len(cluster_sizes) == 1:
        return 1.0

    sorted_sizes = sorted(cluster_sizes)
    n = len(sorted_sizes)
    total = sum(sorted_sizes)

    if total == 0:
        return 1.0

    # Standard Gini calculation:
    # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    weighted_sum = sum((i + 1) * size for i, size in enumerate(sorted_sizes))
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

    return max(0.0, min(1.0, gini))


def _generate_quality_assessment(
    modularity: float,
    silhouette: float,
    balance: float,
    num_clusters: int
) -> str:
    """
    Generate a human-readable assessment of clustering quality.

    Note on metric interpretation:
    - Modularity measures graph edge density within clusters (Louvain's objective)
    - Silhouette measures document co-occurrence similarity (semantic coherence)
    - These metrics measure different things: high modularity with low silhouette
      is normal for graph-based clustering of text, as tokens that co-occur in
      sentences don't necessarily appear in the same documents.
    """
    parts = []

    # Modularity assessment (primary metric for Louvain clustering)
    if modularity >= 0.5:
        parts.append(f"Strong community structure (modularity {modularity:.2f})")
    elif modularity >= 0.3:
        parts.append(f"Good community structure (modularity {modularity:.2f})")
    elif modularity >= 0.1:
        parts.append(f"Weak community structure (modularity {modularity:.2f})")
    else:
        parts.append(f"No clear community structure (modularity {modularity:.2f})")

    # Silhouette assessment (measures document co-occurrence, not graph structure)
    # Negative values are typical for graph-based clustering of diverse corpora
    # because sentence co-occurrence != document co-occurrence
    if silhouette >= 0.25:
        parts.append(f"strong topic coherence (silhouette {silhouette:.2f})")
    elif silhouette >= 0.1:
        parts.append(f"moderate topic coherence (silhouette {silhouette:.2f})")
    elif silhouette >= -0.1:
        parts.append(f"typical graph clustering (silhouette {silhouette:.2f})")
    else:
        parts.append(f"diverse clusters (silhouette {silhouette:.2f})")

    # Balance assessment
    if balance <= 0.3:
        parts.append("well-balanced sizes")
    elif balance <= 0.5:
        parts.append("moderately balanced sizes")
    else:
        parts.append("imbalanced sizes (some clusters dominate)")

    return f"{num_clusters} clusters with {parts[0]}, {parts[1]}, {parts[2]}"
