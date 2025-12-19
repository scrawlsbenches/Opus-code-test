"""
Clustering algorithms for community detection.

Contains:
- cluster_by_louvain: Louvain modularity optimization (recommended)
- cluster_by_label_propagation: Label propagation clustering (legacy)
- build_concept_clusters: Build Layer 2 concepts from token clusters
- _louvain_core: Pure Louvain algorithm for unit testing
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer


def _louvain_core(
    adjacency: Dict[str, Dict[str, float]],
    resolution: float = 1.0,
    max_iterations: int = 10
) -> Dict[str, int]:
    """
    Pure Louvain community detection algorithm.

    This core function takes primitive types and can be unit tested without
    needing HierarchicalLayer objects.

    Args:
        adjacency: Adjacency dict mapping node to {neighbor: weight}.
                  Graph should be undirected (if A->B exists, B->A should too).
        resolution: Resolution parameter for modularity (default 1.0).
                   Higher = more, smaller clusters. Lower = fewer, larger clusters.
        max_iterations: Maximum optimization passes

    Returns:
        Dictionary mapping node to community_id (integer)

    Example:
        >>> adj = {
        ...     "a": {"b": 1.0, "c": 1.0},
        ...     "b": {"a": 1.0, "c": 1.0},
        ...     "c": {"a": 1.0, "b": 1.0},
        ...     "d": {"e": 1.0},
        ...     "e": {"d": 1.0}
        ... }
        >>> communities = _louvain_core(adj)
        >>> assert communities["a"] == communities["b"] == communities["c"]
        >>> assert communities["d"] == communities["e"]
        >>> assert communities["a"] != communities["d"]  # Two separate communities
    """
    # O(n * edges) per iteration, typically converges in O(log n) iterations
    # Overall: O(n * edges * log n) typical case
    nodes = list(adjacency.keys())
    n = len(nodes)

    if n == 0:
        return {}

    # Compute total edge weight
    total_weight = sum(
        sum(neighbors.values())
        for neighbors in adjacency.values()
    ) / 2.0  # Divided by 2 because undirected graph counts each edge twice

    if total_weight == 0:
        # No connections - each node is its own community
        return {node: i for i, node in enumerate(nodes)}

    # Initialize: each node in its own community
    community = {node: i for i, node in enumerate(nodes)}

    # Precompute node degrees
    k = {node: sum(adjacency[node].values()) for node in nodes}

    # Cache community degree sums
    sigma_tot = {i: k[node] for i, node in enumerate(nodes)}

    m = total_weight

    def compute_modularity_gain(node: str, target_comm: int) -> float:
        """Compute modularity gain from moving node to target community."""
        k_i = k[node]

        # Sum of edge weights from node to nodes in target community
        k_i_in = sum(
            weight for neighbor, weight in adjacency[node].items()
            if community.get(neighbor) == target_comm
        )

        sigma = sigma_tot.get(target_comm, 0.0)

        # Modularity gain formula with resolution parameter
        delta_q = (k_i_in / m) - resolution * (sigma * k_i) / (2 * m * m)
        return delta_q

    # Optimization loop
    for _ in range(max_iterations):
        moved = False

        for node in nodes:
            current_comm = community[node]
            k_i = k[node]

            # Remove node from its community temporarily
            sigma_tot[current_comm] -= k_i

            # Find best community
            best_comm = current_comm
            best_gain = 0.0

            # Get neighboring communities
            neighbor_comms = set(
                community[neighbor]
                for neighbor in adjacency[node]
                if neighbor in community
            )
            neighbor_comms.add(current_comm)

            for target_comm in neighbor_comms:
                gain = compute_modularity_gain(node, target_comm)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = target_comm

            # Move to best community
            community[node] = best_comm
            sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + k_i

            if best_comm != current_comm:
                moved = True

        if not moved:
            break

    # Renumber communities to be contiguous
    unique_comms = sorted(set(community.values()))
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    return {node: comm_map[comm] for node, comm in community.items()}


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
    clusters: Dict[int, List[str]],
    doc_vote_threshold: float = 0.1
) -> None:
    """
    Build concept layer from token clusters.

    Creates Layer 2 (Concepts) minicolumns from clustered tokens.
    Each concept is named after its most important members.

    Args:
        layers: Dictionary of all layers
        clusters: Cluster dictionary from label propagation
        doc_vote_threshold: Minimum fraction of cluster members that must
            contain a document for it to be assigned to the concept.
            Default 0.1 (10%) prevents high-frequency tokens from causing
            every concept to contain every document.
    """
    # O(num_clusters * avg_members_per_cluster * (log(avg_members) + avg_docs))
    # Dominated by sorting members by PageRank: O(total_tokens * log(avg_cluster_size))
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

        # Count document votes across cluster members
        # A document is assigned to the concept only if enough members contain it
        doc_votes: Dict[str, int] = {}
        for col in member_cols:
            for doc_id in col.document_ids:
                doc_votes[doc_id] = doc_votes.get(doc_id, 0) + 1

        # Calculate vote threshold (minimum votes needed)
        min_votes = max(1, int(len(member_cols) * doc_vote_threshold))

        # Aggregate properties from members with weighted connections
        max_pagerank = max(c.pagerank for c in member_cols) if member_cols else 1.0
        for col in member_cols:
            concept.feedforward_sources.add(col.id)
            concept.activation += col.activation * 0.5
            concept.occurrence_count += col.occurrence_count
            # Weighted feedforward: concept → token (weight by normalized PageRank)
            weight = col.pagerank / max_pagerank if max_pagerank > 0 else 1.0
            concept.add_feedforward_connection(col.id, weight)
            # Weighted feedback: token → concept (weight by normalized PageRank)
            col.add_feedback_connection(concept.id, weight)

        # Assign documents that meet the vote threshold
        for doc_id, votes in doc_votes.items():
            if votes >= min_votes:
                concept.document_ids.add(doc_id)

        # Set PageRank as average of members
        concept.pagerank = sum(c.pagerank for c in member_cols) / len(member_cols)
