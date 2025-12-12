"""
Analysis Module
===============

Graph analysis algorithms for the cortical network.

Contains implementations of:
- PageRank for importance scoring
- TF-IDF for term weighting
- Louvain community detection for clustering (recommended)
- Label propagation for clustering (legacy, for backward compatibility)
- Activation propagation for information flow
"""

import math
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn
from .constants import RELATION_WEIGHTS


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

    # Iterate until convergence
    for iteration in range(iterations):
        new_pagerank = {}
        max_diff = 0.0

        for col in layer.minicolumns.values():
            # Sum of weighted incoming PageRank
            incoming_sum = 0.0
            for source_id, weight in incoming[col.id]:
                if source_id in pagerank and outgoing_sum[source_id] > 0:
                    incoming_sum += pagerank[source_id] * weight / outgoing_sum[source_id]

            # Apply damping
            new_rank = (1 - damping) / n + damping * incoming_sum
            new_pagerank[col.id] = new_rank

            max_diff = max(max_diff, abs(new_rank - pagerank.get(col.id, 0)))

        pagerank = new_pagerank

        if max_diff < tolerance:
            break

    # Update minicolumn pagerank values
    for col in layer.minicolumns.values():
        col.pagerank = pagerank.get(col.id, 1.0 / n)

    return pagerank


# RELATION_WEIGHTS imported from constants.py


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

    # Iterate until convergence
    iterations_run = 0
    for iteration in range(iterations):
        iterations_run = iteration + 1
        new_pagerank = {}
        max_diff = 0.0

        for col in layer.minicolumns.values():
            # Sum of weighted incoming PageRank
            incoming_sum = 0.0
            for source_id, weight in incoming[col.id]:
                if source_id in pagerank and outgoing_sum[source_id] > 0:
                    incoming_sum += pagerank[source_id] * weight / outgoing_sum[source_id]

            # Apply damping
            new_rank = (1 - damping) / n + damping * incoming_sum
            new_pagerank[col.id] = new_rank

            max_diff = max(max_diff, abs(new_rank - pagerank.get(col.id, 0)))

        pagerank = new_pagerank

        if max_diff < tolerance:
            break

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
            compute_pagerank(layer, damping=damping, iterations=layer_iterations, tolerance=1e-6)

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


def compute_tfidf(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str]
) -> None:
    """
    Compute TF-IDF scores for tokens.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) measures
    how distinctive a term is to the corpus. High TF-IDF terms are
    both frequent in their documents and rare across the corpus.
    
    Args:
        layers: Dictionary of layers (needs TOKENS layer)
        documents: Dictionary mapping doc_id to content
    """
    layer0 = layers[CorticalLayer.TOKENS]
    num_docs = len(documents)
    
    if num_docs == 0:
        return
    
    for col in layer0.minicolumns.values():
        # Document frequency
        df = len(col.document_ids)
        
        if df > 0:
            # Inverse document frequency
            idf = math.log(num_docs / df)
            
            # Term frequency (normalized by occurrence count)
            tf = math.log1p(col.occurrence_count)
            
            # TF-IDF
            col.tfidf = tf * idf
            
            # Per-document TF-IDF using actual occurrence counts
            for doc_id in col.document_ids:
                # Get actual term frequency in this document
                doc_tf = col.doc_occurrence_counts.get(doc_id, 1)
                col.tfidf_per_doc[doc_id] = math.log1p(doc_tf) * idf


def propagate_activation(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    iterations: int = 3,
    decay: float = 0.8,
    lateral_weight: float = 0.3
) -> None:
    """
    Propagate activation through the network.
    
    This simulates how information flows through cortical layers:
    - Activation spreads to connected columns (lateral)
    - Activation flows up the hierarchy (feedforward)
    - Activation decays over time
    
    Args:
        layers: Dictionary of all layers
        iterations: Number of propagation iterations
        decay: How much activation decays per iteration
        lateral_weight: Weight for lateral spreading
    """
    for _ in range(iterations):
        # Store new activations
        new_activations: Dict[str, float] = {}
        
        # Process each layer
        for layer_enum in CorticalLayer:
            if layer_enum not in layers:
                continue
            layer = layers[layer_enum]
            
            for col in layer.minicolumns.values():
                # Start with decayed current activation
                new_act = col.activation * decay
                
                # Add lateral input using O(1) ID lookup
                for neighbor_id, weight in col.lateral_connections.items():
                    neighbor = layer.get_by_id(neighbor_id)
                    if neighbor:
                        new_act += neighbor.activation * weight * lateral_weight
                
                # Add feedforward input using O(1) ID lookup
                for source_id in col.feedforward_sources:
                    # Find source in lower layers
                    for lower_enum in CorticalLayer:
                        if lower_enum >= layer_enum:
                            break
                        if lower_enum not in layers:
                            continue
                        lower_layer = layers[lower_enum]
                        source = lower_layer.get_by_id(source_id)
                        if source:
                            new_act += source.activation * 0.5
                            break
                
                new_activations[col.id] = new_act
        
        # Apply new activations
        for layer_enum in CorticalLayer:
            if layer_enum not in layers:
                continue
            layer = layers[layer_enum]
            for col in layer.minicolumns.values():
                if col.id in new_activations:
                    col.activation = new_activations[col.id]


def cluster_by_label_propagation(
    layer: HierarchicalLayer,
    min_cluster_size: int = 3,
    max_iterations: int = 20,
    cluster_strictness: float = 1.0,
    bridge_weight: float = 0.0
) -> Dict[int, List[str]]:
    """
    Cluster minicolumns using label propagation.

    Label propagation is a semi-supervised community detection
    algorithm. Each node adopts the most common label among its
    neighbors, causing labels to propagate through densely
    connected regions.

    Args:
        layer: Layer to cluster
        min_cluster_size: Minimum nodes per cluster
        max_iterations: Maximum iterations
        cluster_strictness: Controls clustering aggressiveness (0.0-1.0).
            - 1.0 (default): Strict clustering, topics stay separate
            - 0.5: Moderate mixing, allows some cross-topic clustering
            - 0.0: Minimal clustering, most tokens group together
            Lower values create fewer, larger clusters.
        bridge_weight: Weight for synthetic inter-document connections (0.0-1.0).
            When > 0, adds weak connections between tokens that appear in
            different documents, helping bridge topic-isolated clusters.
            - 0.0 (default): No bridging
            - 0.3: Light bridging
            - 0.7: Strong bridging

    Returns:
        Dictionary mapping cluster_id to list of column contents
    """
    # Clamp parameters to valid range
    cluster_strictness = max(0.0, min(1.0, cluster_strictness))
    bridge_weight = max(0.0, min(1.0, bridge_weight))

    # Initialize each node with unique label
    labels = {col.content: i for i, col in enumerate(layer.minicolumns.values())}

    # Get column list for shuffling
    columns = list(layer.minicolumns.keys())

    # Build augmented connection weights (includes optional bridging)
    augmented_connections: Dict[str, Dict[str, float]] = defaultdict(dict)

    for content in columns:
        col = layer.minicolumns[content]
        for neighbor_id, weight in col.lateral_connections.items():
            neighbor = layer.get_by_id(neighbor_id)
            if neighbor:
                augmented_connections[content][neighbor.content] = weight

    # Add synthetic bridge connections between documents if requested
    if bridge_weight > 0:
        # Group tokens by document
        doc_tokens: Dict[str, List[str]] = defaultdict(list)
        for content in columns:
            col = layer.minicolumns[content]
            for doc_id in col.document_ids:
                doc_tokens[doc_id].append(content)

        # Create weak connections between tokens from different documents
        doc_ids = list(doc_tokens.keys())
        for i, doc1 in enumerate(doc_ids):
            for doc2 in doc_ids[i+1:]:
                tokens1 = doc_tokens[doc1]
                tokens2 = doc_tokens[doc2]
                # Connect a sample of tokens to avoid O(n²) explosion
                sample_size = min(5, len(tokens1), len(tokens2))
                for t1 in tokens1[:sample_size]:
                    for t2 in tokens2[:sample_size]:
                        if t1 != t2:
                            # Add weak bidirectional bridge
                            current = augmented_connections[t1].get(t2, 0)
                            augmented_connections[t1][t2] = current + bridge_weight * 0.5
                            current = augmented_connections[t2].get(t1, 0)
                            augmented_connections[t2][t1] = current + bridge_weight * 0.5

    # Calculate label change threshold based on strictness
    # Higher strictness = requires stronger evidence to change label
    # This means higher strictness → higher threshold → more clusters (topics stay separate)
    change_threshold = cluster_strictness * 0.3

    for iteration in range(max_iterations):
        changed = False

        # Process in order (could shuffle for better results)
        for content in columns:
            # Count neighbor labels weighted by connection strength
            label_weights: Dict[int, float] = defaultdict(float)

            for neighbor_content, weight in augmented_connections[content].items():
                if neighbor_content in labels:
                    label_weights[labels[neighbor_content]] += weight

            # Apply strictness: current label gets a bonus based on strictness
            current_label = labels[content]
            if current_label in label_weights and cluster_strictness > 0.0:
                # Higher strictness = stronger bias toward current label (resist change)
                label_weights[current_label] *= (1 + cluster_strictness * 2)

            # Adopt most common label
            if label_weights:
                best_label, best_weight = max(label_weights.items(), key=lambda x: x[1])
                current_weight = label_weights.get(current_label, 0)

                # Only change if the improvement exceeds threshold
                if best_label != current_label:
                    if current_weight == 0 or (best_weight / max(current_weight, 0.001) - 1) > change_threshold:
                        labels[content] = best_label
                        changed = True

        if not changed:
            break

    # Build clusters
    clusters: Dict[int, List[str]] = defaultdict(list)
    for content, label in labels.items():
        clusters[label].append(content)

    # Filter by minimum size
    filtered = {
        label: members
        for label, members in clusters.items()
        if len(members) >= min_cluster_size
    }

    # Update cluster_id on minicolumns
    for label, members in filtered.items():
        for content in members:
            if content in layer.minicolumns:
                layer.minicolumns[content].cluster_id = label

    return filtered


def cluster_by_louvain(
    layer: HierarchicalLayer,
    min_cluster_size: int = 3,
    resolution: float = 1.0,
    max_iterations: int = 10
) -> Dict[int, List[str]]:
    """
    Cluster minicolumns using Louvain community detection.

    Louvain is a modularity optimization algorithm that finds communities
    by iteratively improving modularity. Unlike label propagation, it
    handles dense graphs well and produces meaningful clusters.

    The algorithm works in two phases:
    1. Local optimization: Move nodes to communities that maximize modularity
    2. Network aggregation: Merge communities into super-nodes and repeat

    Args:
        layer: Layer to cluster
        min_cluster_size: Minimum nodes per cluster (clusters below this
            size are filtered from the result)
        resolution: Resolution parameter for modularity (default 1.0).
            - Higher values (>1.0): More, smaller clusters
            - Lower values (<1.0): Fewer, larger clusters
        max_iterations: Maximum number of optimization passes (default 10)

    Returns:
        Dictionary mapping cluster_id to list of column contents

    Example:
        >>> clusters = cluster_by_louvain(layer0, min_cluster_size=3)
        >>> print(f"Found {len(clusters)} clusters")

    Note:
        This is a zero-dependency implementation of the Louvain algorithm.
        For very large graphs (>100k nodes), consider using optimized
        implementations from networkx or igraph.
    """
    columns = list(layer.minicolumns.keys())
    n = len(columns)

    if n == 0:
        return {}

    # Build adjacency structure from layer connections
    # content -> {neighbor_content: weight}
    adjacency: Dict[str, Dict[str, float]] = {c: {} for c in columns}
    total_weight = 0.0

    for content in columns:
        col = layer.minicolumns[content]
        for neighbor_id, weight in col.lateral_connections.items():
            neighbor = layer.get_by_id(neighbor_id)
            if neighbor and neighbor.content in adjacency:
                adjacency[content][neighbor.content] = weight
                total_weight += weight

    # m = total edge weight (each edge counted once)
    # Since adjacency is bidirectional, total_weight counts each edge twice
    m = total_weight / 2.0

    if m == 0:
        # No connections - each node is its own cluster
        clusters = {i: [content] for i, content in enumerate(columns)}
        return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

    # Initialize: each node in its own community
    # community[content] = community_id
    community: Dict[str, int] = {content: i for i, content in enumerate(columns)}

    # Precompute node degrees (sum of edge weights)
    # k[content] = sum of weights attached to content
    k: Dict[str, float] = {}
    for content in columns:
        k[content] = sum(adjacency[content].values())

    # Cache community degree sums for O(1) lookup instead of O(n) per node
    # sigma_tot[community_id] = sum of degrees of nodes in that community
    sigma_tot: Dict[int, float] = {i: k[content] for i, content in enumerate(columns)}

    def compute_modularity_gain(
        node: str,
        target_community: int,
        node_community_weights: Dict[int, float]
    ) -> float:
        """
        Compute modularity gain from moving node to target_community.

        Uses the formula:
        ΔQ = [k_i,in / m - resolution * k_i * Σ_tot / (2m²)]

        where:
        - k_i = degree of node i
        - k_i,in = sum of edge weights from node i to nodes in target community
        - Σ_tot = sum of degrees of all nodes in target community
        """
        k_i = k[node]
        k_i_in = node_community_weights.get(target_community, 0.0)

        # Use cached sigma_tot value (O(1) instead of O(n))
        target_sigma_tot = sigma_tot.get(target_community, 0.0)

        # If node is already in target_community, exclude its contribution
        if community[node] == target_community:
            target_sigma_tot -= k_i

        if m == 0:
            return 0.0

        # Modularity gain with resolution parameter
        gain = k_i_in / m - resolution * k_i * target_sigma_tot / (2 * m * m)
        return gain

    def phase1() -> bool:
        """
        Local optimization phase.

        For each node, try moving it to each neighbor's community.
        Move to the community that gives maximum modularity gain.

        Returns:
            True if any node was moved, False if converged
        """
        nonlocal sigma_tot  # Allow updating the cached sigma_tot

        improved = True
        any_moved = False

        while improved:
            improved = False

            for node in columns:
                current_comm = community[node]

                # Compute weights to each neighboring community
                # comm_weights[community_id] = sum of edge weights to that community
                comm_weights: Dict[int, float] = {}
                for neighbor, weight in adjacency[node].items():
                    neighbor_comm = community[neighbor]
                    comm_weights[neighbor_comm] = comm_weights.get(neighbor_comm, 0.0) + weight

                # Find best community to move to
                best_comm = current_comm
                best_gain = 0.0

                # Check current community first (to stay if no improvement)
                for target_comm in comm_weights:
                    if target_comm == current_comm:
                        continue

                    gain = compute_modularity_gain(node, target_comm, comm_weights)
                    # Also compute "loss" from leaving current community
                    loss = compute_modularity_gain(node, current_comm, comm_weights)
                    net_gain = gain - loss

                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_comm = target_comm

                # Move node if there's improvement
                if best_comm != current_comm:
                    # Update sigma_tot cache: remove from old, add to new
                    k_node = k[node]
                    sigma_tot[current_comm] = sigma_tot.get(current_comm, 0.0) - k_node
                    sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + k_node

                    community[node] = best_comm
                    improved = True
                    any_moved = True

        return any_moved

    def phase2() -> Tuple[
        Dict[str, Dict[str, float]],  # new adjacency
        Dict[str, int],  # new community mapping
        Dict[str, float],  # new k values
        Dict[int, float],  # new sigma_tot
        float,  # new m value
        Dict[int, List[str]]  # community -> original nodes
    ]:
        """
        Network aggregation phase.

        Merge nodes in the same community into super-nodes.
        Create new graph where edge weight between super-nodes is
        sum of edges between their constituent nodes.

        Returns:
            New adjacency, community mapping, k values, sigma_tot, m, and community members
        """
        # Get unique communities
        unique_comms = set(community.values())

        # Map old community IDs to new sequential IDs
        comm_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_comms))}

        # Track which original nodes belong to each super-node
        comm_members: Dict[int, List[str]] = {i: [] for i in range(len(unique_comms))}
        for node, comm in community.items():
            new_comm = comm_map[comm]
            comm_members[new_comm].append(node)

        # Build new adjacency between super-nodes
        new_adj: Dict[str, Dict[str, float]] = {}
        new_m = 0.0

        for new_comm in range(len(unique_comms)):
            new_adj[str(new_comm)] = {}

        for node, neighbors in adjacency.items():
            node_new_comm = comm_map[community[node]]
            for neighbor, weight in neighbors.items():
                neighbor_new_comm = comm_map[community[neighbor]]
                if node_new_comm != neighbor_new_comm:
                    # Edge between different communities
                    key = str(neighbor_new_comm)
                    node_key = str(node_new_comm)
                    new_adj[node_key][key] = new_adj[node_key].get(key, 0.0) + weight
                    new_m += weight

        new_m /= 2.0  # Each edge counted twice

        # New community mapping (each super-node starts in its own community)
        new_community = {str(i): i for i in range(len(unique_comms))}

        # New k values (degree of each super-node)
        new_k: Dict[str, float] = {}
        for new_comm in range(len(unique_comms)):
            # Sum of all degrees of constituent nodes
            new_k[str(new_comm)] = sum(k[node] for node in comm_members[new_comm])

        # New sigma_tot (each super-node starts in its own community, so sigma_tot = k)
        new_sigma_tot: Dict[int, float] = {i: new_k[str(i)] for i in range(len(unique_comms))}

        return new_adj, new_community, new_k, new_sigma_tot, new_m, comm_members

    # Main Louvain loop
    # Track the hierarchy of community memberships
    community_hierarchy: List[Dict[int, List[str]]] = []

    for iteration in range(max_iterations):
        # Phase 1: Local optimization
        moved = phase1()

        if not moved and iteration > 0:
            # Converged
            break

        # Check if we've reduced to a single community
        unique_comms = set(community.values())
        if len(unique_comms) <= 1:
            break

        # Phase 2: Network aggregation
        adjacency, new_community, k, sigma_tot, m, members = phase2()
        community_hierarchy.append(members)

        # Update columns list for new super-nodes
        columns = list(adjacency.keys())
        community = new_community

        if m == 0:
            break

    # Reconstruct final communities by unwinding the hierarchy
    # Start with the final community assignment
    final_communities: Dict[int, Set[str]] = {}

    if community_hierarchy:
        # We have hierarchy - unwind it
        # Start with last level
        for super_node, comm in community.items():
            if comm not in final_communities:
                final_communities[comm] = set()

            # Trace back through hierarchy to get original nodes
            current_members = {super_node}

            for level_members in reversed(community_hierarchy):
                new_members: Set[str] = set()
                for member in current_members:
                    member_int = int(member)
                    if member_int in level_members:
                        new_members.update(level_members[member_int])
                    else:
                        # Member is an original node
                        new_members.add(member)
                current_members = new_members

            final_communities[comm].update(current_members)
    else:
        # No hierarchy - use direct community assignment
        for node, comm in community.items():
            if comm not in final_communities:
                final_communities[comm] = set()
            final_communities[comm].add(node)

    # Convert to expected format and filter by size
    result: Dict[int, List[str]] = {}
    cluster_id = 0
    for comm, members in final_communities.items():
        # Filter out numeric super-node IDs, keep only original content strings
        original_members = [m for m in members if m in layer.minicolumns]
        if len(original_members) >= min_cluster_size:
            result[cluster_id] = original_members
            cluster_id += 1

    # Update cluster_id on minicolumns
    for label, members in result.items():
        for content in members:
            if content in layer.minicolumns:
                layer.minicolumns[content].cluster_id = label

    return result


def build_concept_clusters(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    clusters: Dict[int, List[str]]
) -> None:
    """
    Build concept layer from token clusters.
    
    Creates Layer 2 (Concepts) minicolumns from clustered tokens.
    Each concept is named after its most important members.
    
    Args:
        layers: Dictionary of all layers
        clusters: Cluster dictionary from label propagation
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers[CorticalLayer.CONCEPTS]
    
    for cluster_id, members in clusters.items():
        if len(members) < 2:
            continue
        
        # Get member columns and sort by PageRank
        member_cols = []
        for m in members:
            col = layer0.get_minicolumn(m)
            if col:
                member_cols.append(col)
        
        if not member_cols:
            continue
        
        member_cols.sort(key=lambda c: c.pagerank, reverse=True)
        
        # Name concept after top members
        top_names = [c.content for c in member_cols[:3]]
        concept_name = '/'.join(top_names)
        
        # Create concept minicolumn
        concept = layer2.get_or_create_minicolumn(concept_name)
        concept.cluster_id = cluster_id
        
        # Aggregate properties from members with weighted connections
        max_pagerank = max(c.pagerank for c in member_cols) if member_cols else 1.0
        for col in member_cols:
            concept.feedforward_sources.add(col.id)
            concept.document_ids.update(col.document_ids)
            concept.activation += col.activation * 0.5
            concept.occurrence_count += col.occurrence_count
            # Weighted feedforward: concept → token (weight by normalized PageRank)
            weight = col.pagerank / max_pagerank if max_pagerank > 0 else 1.0
            concept.add_feedforward_connection(col.id, weight)
            # Weighted feedback: token → concept (weight by normalized PageRank)
            col.add_feedback_connection(concept.id, weight)

        # Set PageRank as average of members
        concept.pagerank = sum(c.pagerank for c in member_cols) / len(member_cols)


def compute_concept_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    semantic_relations: List[Tuple[str, str, str, float]] = None,
    min_shared_docs: int = 1,
    min_jaccard: float = 0.1,
    use_member_semantics: bool = False,
    use_embedding_similarity: bool = False,
    embedding_threshold: float = 0.3,
    embeddings: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """
    Build lateral connections between concepts in Layer 2.

    Concepts are connected based on:
    1. Shared documents (Jaccard similarity of document sets)
    2. Semantic relations between member tokens (if provided)
    3. Semantic relations between members independent of docs (use_member_semantics)
    4. Embedding similarity of concept centroids (use_embedding_similarity)

    Args:
        layers: Dictionary of all layers
        semantic_relations: Optional list of (term1, relation, term2, weight) tuples
        min_shared_docs: Minimum shared documents for connection (0 to disable filter)
        min_jaccard: Minimum Jaccard similarity threshold (0.0 to disable filter)
        use_member_semantics: Connect concepts via semantic relations between members,
                              even without document overlap
        use_embedding_similarity: Connect concepts via embedding similarity of centroids
        embedding_threshold: Minimum cosine similarity for embedding-based connections
        embeddings: Term embeddings dict (required if use_embedding_similarity=True)

    Returns:
        Statistics about connections created
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers[CorticalLayer.CONCEPTS]

    if layer2.column_count() == 0:
        return {
            'connections_created': 0,
            'concepts': 0,
            'doc_overlap_connections': 0,
            'semantic_connections': 0,
            'embedding_connections': 0
        }

    concepts = list(layer2.minicolumns.values())
    connections_created = 0
    doc_overlap_connections = 0
    semantic_connections = 0
    embedding_connections = 0

    # Build semantic relation lookup for faster access
    semantic_lookup: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)
    if semantic_relations:
        for t1, relation, t2, weight in semantic_relations:
            # Store relation in both directions
            semantic_lookup[t1][t2] = (relation, weight)
            semantic_lookup[t2][t1] = (relation, weight)

    # Relation type weights for scoring
    relation_weights = {
        'IsA': 1.5,
        'PartOf': 1.3,
        'HasProperty': 1.2,
        'RelatedTo': 1.0,
        'Antonym': 0.3,
    }

    # Pre-compute member tokens for each concept (used by multiple strategies)
    concept_members: Dict[str, Set[str]] = {}
    for concept in concepts:
        members = set()
        for token_id in concept.feedforward_connections:
            token = layer0.get_by_id(token_id)
            if token:
                members.add(token.content)
        concept_members[concept.id] = members

    # Pre-compute concept centroids if using embedding similarity
    concept_centroids: Dict[str, List[float]] = {}
    if use_embedding_similarity and embeddings:
        for concept in concepts:
            members = concept_members[concept.id]
            member_embeddings = [embeddings[m] for m in members if m in embeddings]
            if member_embeddings:
                dim = len(member_embeddings[0])
                centroid = [0.0] * dim
                for emb in member_embeddings:
                    for j, v in enumerate(emb):
                        centroid[j] += v
                for j in range(dim):
                    centroid[j] /= len(member_embeddings)
                concept_centroids[concept.id] = centroid

    # Track which pairs have been connected to avoid duplicates
    connected_pairs: Set[Tuple[str, str]] = set()

    def add_connection(c1: Minicolumn, c2: Minicolumn, weight: float) -> bool:
        """Add bidirectional connection if not already connected."""
        pair = tuple(sorted([c1.id, c2.id]))
        if pair in connected_pairs:
            # Already connected, strengthen existing connection
            c1.add_lateral_connection(c2.id, weight)
            c2.add_lateral_connection(c1.id, weight)
            return False
        connected_pairs.add(pair)
        c1.add_lateral_connection(c2.id, weight)
        c2.add_lateral_connection(c1.id, weight)
        return True

    # Compare all concept pairs
    for i, concept1 in enumerate(concepts):
        docs1 = concept1.document_ids
        members1 = concept_members[concept1.id]

        for concept2 in concepts[i+1:]:
            docs2 = concept2.document_ids
            members2 = concept_members[concept2.id]

            # Strategy 1: Document overlap (traditional method)
            shared_docs = docs1 & docs2
            union_docs = docs1 | docs2
            jaccard = len(shared_docs) / len(union_docs) if union_docs else 0

            passes_doc_filter = (
                len(shared_docs) >= min_shared_docs and jaccard >= min_jaccard
            )

            if passes_doc_filter:
                # Base weight from document overlap
                weight = jaccard

                # Add semantic relation bonus if available
                if semantic_relations:
                    semantic_bonus = 0.0
                    relation_count = 0
                    for m1 in members1:
                        if m1 in semantic_lookup:
                            for m2 in members2:
                                if m2 in semantic_lookup[m1]:
                                    relation, rel_weight = semantic_lookup[m1][m2]
                                    rel_multiplier = relation_weights.get(relation, 1.0)
                                    semantic_bonus += rel_weight * rel_multiplier
                                    relation_count += 1

                    # Normalize and add semantic bonus (max 50% boost)
                    if relation_count > 0:
                        avg_semantic = semantic_bonus / relation_count
                        weight *= (1 + min(avg_semantic, 0.5))

                if add_connection(concept1, concept2, weight):
                    connections_created += 1
                    doc_overlap_connections += 1

            # Strategy 2: Member semantic relations (independent of document overlap)
            if use_member_semantics and semantic_relations and not passes_doc_filter:
                semantic_score = 0.0
                relation_count = 0
                for m1 in members1:
                    if m1 in semantic_lookup:
                        for m2 in members2:
                            if m2 in semantic_lookup[m1]:
                                relation, rel_weight = semantic_lookup[m1][m2]
                                rel_multiplier = relation_weights.get(relation, 1.0)
                                semantic_score += rel_weight * rel_multiplier
                                relation_count += 1

                if relation_count > 0:
                    # Normalize by number of relations found
                    avg_score = semantic_score / relation_count
                    # Scale to reasonable weight range (0.1 - 0.8)
                    weight = min(0.1 + avg_score * 0.3, 0.8)
                    if add_connection(concept1, concept2, weight):
                        connections_created += 1
                        semantic_connections += 1

            # Strategy 3: Embedding similarity (independent of document overlap)
            if use_embedding_similarity and embeddings and not passes_doc_filter:
                if concept1.id in concept_centroids and concept2.id in concept_centroids:
                    centroid1 = concept_centroids[concept1.id]
                    centroid2 = concept_centroids[concept2.id]
                    sim = cosine_similarity(
                        {str(i): v for i, v in enumerate(centroid1)},
                        {str(i): v for i, v in enumerate(centroid2)}
                    )
                    if sim >= embedding_threshold:
                        # Scale similarity to connection weight
                        weight = sim * 0.7  # Scale down slightly
                        if add_connection(concept1, concept2, weight):
                            connections_created += 1
                            embedding_connections += 1

    return {
        'connections_created': connections_created,
        'concepts': len(concepts),
        'doc_overlap_connections': doc_overlap_connections,
        'semantic_connections': semantic_connections,
        'embedding_connections': embedding_connections
    }


def compute_bigram_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    min_shared_docs: int = 1,
    component_weight: float = 0.5,
    chain_weight: float = 0.7,
    cooccurrence_weight: float = 0.3,
    max_bigrams_per_term: int = 100,
    max_bigrams_per_doc: int = 500
) -> Dict[str, Any]:
    """
    Build lateral connections between bigrams in Layer 1.

    Bigrams are connected based on:
    1. Shared component terms ("neural_networks" ↔ "neural_processing")
    2. Document co-occurrence (appear in same documents)
    3. Chains ("machine_learning" ↔ "learning_algorithms" where right=left)

    Args:
        layers: Dictionary of all layers
        min_shared_docs: Minimum shared documents for co-occurrence connection
        component_weight: Weight for shared component connections (default 0.5)
        chain_weight: Weight for chain connections (default 0.7)
        cooccurrence_weight: Weight for document co-occurrence (default 0.3)
        max_bigrams_per_term: Skip terms appearing in more than this many bigrams
            to avoid O(n²) explosion from common terms like "self", "return" (default 100)
        max_bigrams_per_doc: Skip documents with more than this many bigrams for
            co-occurrence connections to avoid O(n²) explosion (default 500)

    Returns:
        Statistics about connections created:
        - connections_created: Total bidirectional connections
        - component_connections: Connections from shared components
        - chain_connections: Connections from chains
        - cooccurrence_connections: Connections from document co-occurrence
        - skipped_common_terms: Number of terms skipped due to max_bigrams_per_term
        - skipped_large_docs: Number of docs skipped due to max_bigrams_per_doc
    """
    layer1 = layers[CorticalLayer.BIGRAMS]

    if layer1.column_count() == 0:
        return {
            'connections_created': 0,
            'bigrams': 0,
            'component_connections': 0,
            'chain_connections': 0,
            'cooccurrence_connections': 0,
            'skipped_common_terms': 0,
            'skipped_large_docs': 0
        }

    bigrams = list(layer1.minicolumns.values())

    # Build indexes for efficient lookup
    # left_component_index: {"neural": [bigram1, bigram2, ...]}
    # right_component_index: {"networks": [bigram1, bigram3, ...]}
    # Note: Bigrams use space separators (e.g., "neural networks")
    left_index: Dict[str, List[Minicolumn]] = defaultdict(list)
    right_index: Dict[str, List[Minicolumn]] = defaultdict(list)

    for bigram in bigrams:
        parts = bigram.content.split(' ')
        if len(parts) == 2:
            left_index[parts[0]].append(bigram)
            right_index[parts[1]].append(bigram)

    # Track connection types for statistics
    component_connections = 0
    chain_connections = 0
    cooccurrence_connections = 0

    # Track which pairs we've already connected (avoid duplicates)
    connected_pairs: Set[Tuple[str, str]] = set()

    def add_connection(b1: Minicolumn, b2: Minicolumn, weight: float, conn_type: str) -> bool:
        """Add bidirectional connection if not already connected."""
        nonlocal component_connections, chain_connections, cooccurrence_connections

        pair = tuple(sorted([b1.id, b2.id]))
        if pair in connected_pairs:
            # Already connected, just strengthen the connection
            b1.add_lateral_connection(b2.id, weight)
            b2.add_lateral_connection(b1.id, weight)
            return False

        connected_pairs.add(pair)
        b1.add_lateral_connection(b2.id, weight)
        b2.add_lateral_connection(b1.id, weight)

        if conn_type == 'component':
            component_connections += 1
        elif conn_type == 'chain':
            chain_connections += 1
        elif conn_type == 'cooccurrence':
            cooccurrence_connections += 1

        return True

    # Track skipped common terms for statistics
    skipped_common_terms = 0

    # 1. Connect bigrams sharing a component
    # Left component matches: "neural_networks" ↔ "neural_processing"
    for component, bigram_list in left_index.items():
        # Skip overly common terms to avoid O(n²) explosion
        if len(bigram_list) > max_bigrams_per_term:
            skipped_common_terms += 1
            continue
        for i, b1 in enumerate(bigram_list):
            for b2 in bigram_list[i+1:]:
                # Weight by component's PageRank importance (if available)
                weight = component_weight
                add_connection(b1, b2, weight, 'component')

    # Right component matches: "deep_learning" ↔ "machine_learning"
    for component, bigram_list in right_index.items():
        # Skip overly common terms to avoid O(n²) explosion
        if len(bigram_list) > max_bigrams_per_term:
            skipped_common_terms += 1
            continue
        for i, b1 in enumerate(bigram_list):
            for b2 in bigram_list[i+1:]:
                weight = component_weight
                add_connection(b1, b2, weight, 'component')

    # 2. Connect chain bigrams (right of one = left of other)
    # "machine_learning" ↔ "learning_algorithms"
    for term in left_index:
        if term in right_index:
            # Skip overly common terms
            if len(left_index[term]) > max_bigrams_per_term or len(right_index[term]) > max_bigrams_per_term:
                continue
            # term appears as right component in some bigrams and left in others
            for b_left in right_index[term]:  # ends with term
                for b_right in left_index[term]:  # starts with term
                    if b_left.id != b_right.id:
                        add_connection(b_left, b_right, chain_weight, 'chain')

    # 3. Connect bigrams that co-occur in the same documents
    # Use inverted index for O(d * b²) instead of O(n²) where d=docs, b=bigrams per doc
    doc_to_bigrams: Dict[str, List[Minicolumn]] = defaultdict(list)
    for bigram in bigrams:
        for doc_id in bigram.document_ids:
            doc_to_bigrams[doc_id].append(bigram)

    # Track pairs we've already processed to avoid duplicate work
    cooccur_processed: Set[Tuple[str, str]] = set()
    skipped_large_docs = 0

    for doc_id, doc_bigrams in doc_to_bigrams.items():
        # Skip documents with too many bigrams to avoid O(n²) explosion
        if len(doc_bigrams) > max_bigrams_per_doc:
            skipped_large_docs += 1
            continue

        # Only compare bigrams within the same document
        for i, b1 in enumerate(doc_bigrams):
            docs1 = b1.document_ids
            for b2 in doc_bigrams[i+1:]:
                # Skip if already processed this pair
                pair_key = tuple(sorted([b1.id, b2.id]))
                if pair_key in cooccur_processed:
                    continue
                cooccur_processed.add(pair_key)

                docs2 = b2.document_ids
                shared_docs = docs1 & docs2
                if len(shared_docs) >= min_shared_docs:
                    # Weight by Jaccard similarity of document sets
                    jaccard = len(shared_docs) / len(docs1 | docs2)
                    weight = cooccurrence_weight * jaccard
                    add_connection(b1, b2, weight, 'cooccurrence')

    return {
        'connections_created': len(connected_pairs),
        'bigrams': len(bigrams),
        'component_connections': component_connections,
        'chain_connections': chain_connections,
        'cooccurrence_connections': cooccurrence_connections,
        'skipped_common_terms': skipped_common_terms,
        'skipped_large_docs': skipped_large_docs
    }


def compute_document_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    min_shared_terms: int = 3
) -> None:
    """
    Build lateral connections between documents.
    
    Documents are connected based on shared vocabulary,
    weighted by TF-IDF scores of shared terms.
    
    Args:
        layers: Dictionary of all layers
        documents: Dictionary of documents
        min_shared_terms: Minimum shared terms for connection
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer3 = layers[CorticalLayer.DOCUMENTS]
    
    doc_ids = list(documents.keys())
    
    for i, doc1 in enumerate(doc_ids):
        col1 = layer3.get_minicolumn(doc1)
        if not col1:
            col1 = layer3.get_or_create_minicolumn(doc1)
        
        for doc2 in doc_ids[i+1:]:
            col2 = layer3.get_minicolumn(doc2)
            if not col2:
                col2 = layer3.get_or_create_minicolumn(doc2)
            
            # Find shared terms
            shared_weight = 0.0
            shared_count = 0
            
            for token_col in layer0.minicolumns.values():
                if doc1 in token_col.document_ids and doc2 in token_col.document_ids:
                    # Weight by TF-IDF
                    weight = token_col.tfidf
                    shared_weight += weight
                    shared_count += 1
            
            if shared_count >= min_shared_terms:
                col1.add_lateral_connection(col2.id, shared_weight)
                col2.add_lateral_connection(col1.id, shared_weight)


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.

    Args:
        vec1: First vector as dict of term -> weight
        vec2: Second vector as dict of term -> weight

    Returns:
        Cosine similarity between 0 and 1
    """
    # Find common keys
    common = set(vec1.keys()) & set(vec2.keys())

    if not common:
        return 0.0

    # Compute dot product
    dot = sum(vec1[k] * vec2[k] for k in common)

    # Compute magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


# =============================================================================
# CLUSTERING QUALITY METRICS (Task #125)
# =============================================================================


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
    import random
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


def _doc_similarity(docs1: frozenset, docs2: frozenset) -> float:
    """
    Compute Jaccard similarity between two document sets.

    Args:
        docs1: Frozenset of document IDs for first token
        docs2: Frozenset of document IDs for second token

    Returns:
        Jaccard similarity: |intersection| / |union|
    """
    if not docs1 or not docs2:
        return 0.0

    intersection = len(docs1 & docs2)
    union = len(docs1 | docs2)

    return intersection / union if union > 0 else 0.0


def _vector_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute similarity between two connection vectors.

    Uses Jaccard-style similarity based on shared connections.
    """
    if not vec1 or not vec2:
        return 0.0

    keys1 = set(vec1.keys())
    keys2 = set(vec2.keys())

    intersection = keys1 & keys2
    union = keys1 | keys2

    if not union:
        return 0.0

    # Weighted Jaccard: sum of mins / sum of maxes
    min_sum = 0.0
    max_sum = 0.0

    for key in union:
        v1 = vec1.get(key, 0.0)
        v2 = vec2.get(key, 0.0)
        min_sum += min(v1, v2)
        max_sum += max(v1, v2)

    return min_sum / max_sum if max_sum > 0 else 0.0


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
