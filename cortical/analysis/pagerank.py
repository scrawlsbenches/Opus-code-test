"""
PageRank algorithms for importance scoring.

Contains:
- compute_pagerank: Standard PageRank for a single layer
- compute_semantic_pagerank: PageRank with semantic relation weighting
- compute_hierarchical_pagerank: Cross-layer PageRank propagation
- _pagerank_core: Pure algorithm for unit testing
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..constants import RELATION_WEIGHTS


def _pagerank_iterate(
    nodes: List[str],
    incoming: Dict[str, List[Tuple[str, float]]],
    outgoing_sum: Dict[str, float],
    pagerank: Dict[str, float],
    damping: float,
    n: int,
    iterations: int,
    tolerance: float
) -> Tuple[Dict[str, float], int]:
    """
    Core PageRank iteration loop.

    Extracted for reuse across different PageRank variants.

    Args:
        nodes: List of node IDs to compute PageRank for
        incoming: Map of node_id -> list of (source_id, weight) tuples
        outgoing_sum: Map of node_id -> sum of outgoing edge weights
        pagerank: Initial PageRank values
        damping: Damping factor (probability of following links)
        n: Number of nodes (for teleportation probability)
        iterations: Maximum iterations
        tolerance: Convergence threshold

    Returns:
        Tuple of (final_pagerank_dict, iterations_run)
    """
    iterations_run = 0

    for iteration in range(iterations):
        iterations_run = iteration + 1
        new_pagerank = {}
        max_diff = 0.0

        for node in nodes:
            incoming_sum = 0.0
            for source_id, weight in incoming.get(node, []):
                if source_id in pagerank and outgoing_sum.get(source_id, 0) > 0:
                    incoming_sum += pagerank[source_id] * weight / outgoing_sum[source_id]

            new_rank = (1 - damping) / n + damping * incoming_sum
            new_pagerank[node] = new_rank
            max_diff = max(max_diff, abs(new_rank - pagerank.get(node, 0)))

        pagerank = new_pagerank

        if max_diff < tolerance:
            break

    return pagerank, iterations_run


def _pagerank_core(
    graph: Dict[str, List[Tuple[str, float]]],
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Pure PageRank algorithm on a graph.

    This core function takes primitive types and can be unit tested without
    needing HierarchicalLayer objects.

    Args:
        graph: Adjacency list mapping node_id to list of (target_id, weight) tuples.
               Each entry represents outgoing edges from that node.
        damping: Damping factor (probability of following links), must be in (0, 1)
        iterations: Maximum number of iterations
        tolerance: Convergence threshold

    Returns:
        Dictionary mapping node_id to PageRank score

    Example:
        >>> graph = {
        ...     "a": [("b", 1.0)],
        ...     "b": [("a", 1.0), ("c", 1.0)],
        ...     "c": [("a", 1.0)]
        ... }
        >>> ranks = _pagerank_core(graph)
        >>> assert ranks["a"] > ranks["c"]  # "a" has more incoming links
    """
    # O(iterations * edges) where edges = total number of connections in the graph
    n = len(graph)
    if n == 0:
        return {}

    nodes = list(graph.keys())

    # Initialize PageRank uniformly
    pagerank = {node: 1.0 / n for node in nodes}

    # Build incoming links map and outgoing sums
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    outgoing_sum: Dict[str, float] = defaultdict(float)

    for source, edges in graph.items():
        for target, weight in edges:
            if target in graph:  # Only count edges to nodes in the graph
                incoming[target].append((source, weight))
                outgoing_sum[source] += weight

    # Run PageRank iteration
    pagerank, _ = _pagerank_iterate(
        nodes=nodes,
        incoming=incoming,
        outgoing_sum=outgoing_sum,
        pagerank=pagerank,
        damping=damping,
        n=n,
        iterations=iterations,
        tolerance=tolerance
    )

    return pagerank


def compute_pagerank(
    layer: HierarchicalLayer,
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Compute PageRank scores for minicolumns in a layer.

    PageRank measures importance based on connection structure.
    Highly connected columns that are connected to other important
    columns receive higher scores.

    Args:
        layer: The layer to compute PageRank for
        damping: Damping factor (probability of following links)
        iterations: Maximum number of iterations
        tolerance: Convergence threshold

    Returns:
        Dictionary mapping column IDs to PageRank scores

    Raises:
        ValueError: If damping is not in range (0, 1)
    """
    # O(iterations * n * avg_degree) where n = number of minicolumns, avg_degree = avg connections per column
    # Typical: O(20 * n * d) where d is average number of lateral connections
    if not (0 < damping < 1):
        raise ValueError(f"damping must be between 0 and 1, got {damping}")

    n = len(layer.minicolumns)
    if n == 0:
        return {}

    # Initialize PageRank uniformly
    pagerank = {col.id: 1.0 / n for col in layer.minicolumns.values()}

    # Build incoming links map
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    outgoing_sum: Dict[str, float] = defaultdict(float)

    for col in layer.minicolumns.values():
        for target_id, weight in col.lateral_connections.items():
            # Use O(1) lookup via get_by_id instead of O(n) linear search
            if layer.get_by_id(target_id) is not None:
                incoming[target_id].append((col.id, weight))
                outgoing_sum[col.id] += weight

    # Run PageRank iteration
    nodes = list(pagerank.keys())
    pagerank, _ = _pagerank_iterate(
        nodes=nodes,
        incoming=incoming,
        outgoing_sum=outgoing_sum,
        pagerank=pagerank,
        damping=damping,
        n=n,
        iterations=iterations,
        tolerance=tolerance
    )

    # Update minicolumn pagerank values
    for col in layer.minicolumns.values():
        col.pagerank = pagerank.get(col.id, 1.0 / n)

    return pagerank


def compute_semantic_pagerank(
    layer: HierarchicalLayer,
    semantic_relations: List[Tuple[str, str, str, float]],
    relation_weights: Optional[Dict[str, float]] = None,
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute PageRank with semantic relation type weighting.

    This ConceptNet-style PageRank applies different multipliers based on
    the semantic relation type between nodes. For example, IsA relationships
    are weighted more heavily than simple co-occurrence.

    Args:
        layer: The layer to compute PageRank for
        semantic_relations: List of (term1, relation, term2, weight) tuples
        relation_weights: Optional custom relation weights dict. If None, uses defaults.
        damping: Damping factor (probability of following links)
        iterations: Maximum number of iterations
        tolerance: Convergence threshold

    Returns:
        Dict containing:
        - pagerank: Dict mapping column IDs to PageRank scores
        - iterations_run: Number of iterations until convergence
        - edges_with_relations: Number of edges that had semantic relation info

    Example:
        >>> relations = [("neural", "RelatedTo", "networks", 0.8)]
        >>> result = compute_semantic_pagerank(layer, relations)
        >>> print(f"PageRank converged in {result['iterations_run']} iterations")

    Raises:
        ValueError: If damping is not in range (0, 1)
    """
    if not (0 < damping < 1):
        raise ValueError(f"damping must be between 0 and 1, got {damping}")

    n = len(layer.minicolumns)
    if n == 0:
        return {'pagerank': {}, 'iterations_run': 0, 'edges_with_relations': 0}

    # Use default weights if not provided
    weights = relation_weights or RELATION_WEIGHTS

    # Build semantic relation lookup: (term1, term2) -> (relation_type, weight)
    semantic_lookup: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for t1, relation, t2, rel_weight in semantic_relations:
        # Store in both directions for undirected lookup
        semantic_lookup[(t1, t2)] = (relation, rel_weight)
        semantic_lookup[(t2, t1)] = (relation, rel_weight)

    # Initialize PageRank uniformly
    pagerank = {col.id: 1.0 / n for col in layer.minicolumns.values()}

    # Build incoming links map with relation-weighted edges
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    outgoing_sum: Dict[str, float] = defaultdict(float)
    edges_with_relations = 0

    # Build content -> id mapping for semantic lookup
    content_to_id: Dict[str, str] = {}
    for col in layer.minicolumns.values():
        content_to_id[col.content] = col.id

    for col in layer.minicolumns.values():
        for target_id, base_weight in col.lateral_connections.items():
            target = layer.get_by_id(target_id)
            if target is None:
                continue

            # Check if there's a semantic relation between these terms
            lookup_key = (col.content, target.content)
            if lookup_key in semantic_lookup:
                relation_type, rel_weight = semantic_lookup[lookup_key]
                # Apply relation type multiplier
                type_multiplier = weights.get(relation_type, 1.0)
                # Combined weight: base_weight * relation_weight * type_multiplier
                adjusted_weight = base_weight * rel_weight * type_multiplier
                edges_with_relations += 1
            else:
                # No semantic relation, use base weight
                adjusted_weight = base_weight

            incoming[target_id].append((col.id, adjusted_weight))
            outgoing_sum[col.id] += adjusted_weight

    # Run PageRank iteration
    nodes = list(pagerank.keys())
    pagerank, iterations_run = _pagerank_iterate(
        nodes=nodes,
        incoming=incoming,
        outgoing_sum=outgoing_sum,
        pagerank=pagerank,
        damping=damping,
        n=n,
        iterations=iterations,
        tolerance=tolerance
    )

    # Update minicolumn pagerank values
    for col in layer.minicolumns.values():
        col.pagerank = pagerank.get(col.id, 1.0 / n)

    return {
        'pagerank': pagerank,
        'iterations_run': iterations_run,
        'edges_with_relations': edges_with_relations
    }


def compute_hierarchical_pagerank(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    layer_iterations: int = 10,
    global_iterations: int = 5,
    damping: float = 0.85,
    cross_layer_damping: float = 0.7,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Compute PageRank with cross-layer propagation.

    This hierarchical PageRank allows importance to flow between layers:
    - Upward: tokens → bigrams → concepts → documents
    - Downward: documents → concepts → bigrams → tokens

    The algorithm alternates between:
    1. Computing local PageRank within each layer
    2. Propagating scores up the hierarchy (via feedback_connections)
    3. Propagating scores down the hierarchy (via feedforward_connections)

    Args:
        layers: Dictionary of all layers
        layer_iterations: Max iterations for intra-layer PageRank
        global_iterations: Max iterations for cross-layer propagation
        damping: Damping factor for intra-layer PageRank
        cross_layer_damping: Damping factor for cross-layer propagation (default 0.7)
        tolerance: Convergence threshold for global iterations

    Returns:
        Dict containing:
        - iterations_run: Number of global iterations
        - converged: Whether the algorithm converged
        - layer_stats: Per-layer statistics

    Example:
        >>> result = compute_hierarchical_pagerank(layers)
        >>> print(f"Converged in {result['iterations_run']} iterations")

    Raises:
        ValueError: If damping or cross_layer_damping is not in range (0, 1)
    """
    if not (0 < damping < 1):
        raise ValueError(f"damping must be between 0 and 1, got {damping}")
    if not (0 < cross_layer_damping < 1):
        raise ValueError(f"cross_layer_damping must be between 0 and 1, got {cross_layer_damping}")

    # Define layer order for propagation
    layer_order = [
        CorticalLayer.TOKENS,
        CorticalLayer.BIGRAMS,
        CorticalLayer.CONCEPTS,
        CorticalLayer.DOCUMENTS
    ]

    # Filter to only existing layers with minicolumns
    active_layers = [l for l in layer_order if l in layers and layers[l].column_count() > 0]

    if not active_layers:
        return {'iterations_run': 0, 'converged': True, 'layer_stats': {}}

    # Store previous PageRank values for convergence check
    prev_pageranks: Dict[CorticalLayer, Dict[str, float]] = {}

    iterations_run = 0
    converged = False

    for global_iter in range(global_iterations):
        iterations_run = global_iter + 1
        max_global_diff = 0.0

        # Step 1: Compute local PageRank for each layer
        for layer_enum in active_layers:
            layer = layers[layer_enum]
            compute_pagerank(layer, damping=damping, iterations=layer_iterations, tolerance=tolerance)

        # Step 2: Propagate up (tokens → bigrams → concepts → documents)
        for i in range(len(active_layers) - 1):
            lower_layer_enum = active_layers[i]
            upper_layer_enum = active_layers[i + 1]
            lower_layer = layers[lower_layer_enum]
            upper_layer = layers[upper_layer_enum]

            # Propagate from lower to upper via feedback connections
            for col in lower_layer.minicolumns.values():
                if not col.feedback_connections:
                    continue

                for target_id, weight in col.feedback_connections.items():
                    target = upper_layer.get_by_id(target_id)
                    if target:
                        # Boost upper layer node based on lower layer importance
                        boost = col.pagerank * weight * cross_layer_damping
                        target.pagerank += boost

        # Step 3: Propagate down (documents → concepts → bigrams → tokens)
        for i in range(len(active_layers) - 1, 0, -1):
            upper_layer_enum = active_layers[i]
            lower_layer_enum = active_layers[i - 1]
            upper_layer = layers[upper_layer_enum]
            lower_layer = layers[lower_layer_enum]

            # Propagate from upper to lower via feedforward connections
            for col in upper_layer.minicolumns.values():
                if not col.feedforward_connections:
                    continue

                for target_id, weight in col.feedforward_connections.items():
                    target = lower_layer.get_by_id(target_id)
                    if target:
                        # Boost lower layer node based on upper layer importance
                        boost = col.pagerank * weight * cross_layer_damping
                        target.pagerank += boost

        # Normalize PageRank within each layer
        for layer_enum in active_layers:
            layer = layers[layer_enum]
            total = sum(col.pagerank for col in layer.minicolumns.values())
            if total > 0:
                for col in layer.minicolumns.values():
                    col.pagerank /= total

        # Check convergence
        for layer_enum in active_layers:
            layer = layers[layer_enum]
            current_pr = {col.id: col.pagerank for col in layer.minicolumns.values()}

            if layer_enum in prev_pageranks:
                for col_id, pr in current_pr.items():
                    prev_pr = prev_pageranks[layer_enum].get(col_id, 0)
                    max_global_diff = max(max_global_diff, abs(pr - prev_pr))

            prev_pageranks[layer_enum] = current_pr

        if max_global_diff < tolerance and global_iter > 0:
            converged = True
            break

    # Collect layer statistics
    layer_stats = {}
    for layer_enum in active_layers:
        layer = layers[layer_enum]
        pageranks = [col.pagerank for col in layer.minicolumns.values()]
        layer_stats[layer_enum.name] = {
            'nodes': len(pageranks),
            'max_pagerank': max(pageranks) if pageranks else 0,
            'min_pagerank': min(pageranks) if pageranks else 0,
            'avg_pagerank': sum(pageranks) / len(pageranks) if pageranks else 0
        }

    return {
        'iterations_run': iterations_run,
        'converged': converged,
        'layer_stats': layer_stats
    }
