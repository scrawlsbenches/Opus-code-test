"""
Graph Algorithms: Composable algorithm mixins for BaseGraph.

This module provides algorithm mixins that can be added to any
BaseGraph subclass through multiple inheritance. Each mixin adds
a specific set of algorithms.

Available Mixins:
- PageRankMixin: PageRank centrality algorithm
- ClusteringMixin: Community detection (label propagation)
- SpreadingActivationMixin: Activation spreading through graph

Usage:
    class MyGraph(BaseGraph[MyNode, MyEdge], PageRankMixin, ClusteringMixin):
        ...

    graph = MyGraph()
    pagerank = graph.compute_pagerank()
    clusters = graph.label_propagation()

Design Philosophy:
    Mixins allow selective composition of algorithms. A lightweight
    graph doesn't need PageRank overhead. A reasoning graph might
    need spreading activation but not clustering. Pick what you need.

See docs/base-graph-design.md for architecture details.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


class PageRankMixin:
    """
    Mixin providing PageRank centrality algorithm.

    PageRank computes the relative importance of nodes based on
    the link structure. Nodes linked by many important nodes
    are themselves important.

    Performance: O(E × iterations) where E = edge count

    Example:
        class MyGraph(BaseGraph, PageRankMixin):
            pass

        graph = MyGraph()
        # ... add nodes and edges ...
        scores = graph.compute_pagerank(damping=0.85, iterations=20)
    """

    def compute_pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 20,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes.

        Algorithm:
            PR(A) = (1-d)/N + d * Σ(PR(B)/L(B)) for all B linking to A
            where d = damping factor, N = node count, L(B) = out-links from B

        Args:
            damping: Damping factor (0-1, typically 0.85)
                - Higher = more weight on link structure
                - Lower = more uniform distribution
            iterations: Maximum iterations
            tolerance: Convergence threshold (early termination)

        Returns:
            Dict mapping node_id to PageRank score (sums to 1.0)

        Performance:
            O(E × iterations) where E = number of edges
        """
        n = self.node_count
        if n == 0:
            return {}

        # Initialize uniform distribution
        pr: Dict[str, float] = {node.id: 1.0 / n for node in self.nodes}

        # Build outgoing edge count
        out_degree: Dict[str, int] = {
            node.id: len(self.edges_from(node.id)) for node in self.nodes
        }

        for _ in range(iterations):
            new_pr: Dict[str, float] = {}
            max_diff = 0.0

            for node in self.nodes:
                # Sum of PR contributions from incoming edges
                incoming_sum = 0.0
                for edge in self.edges_to(node.id):
                    source_out = out_degree[edge.source_id]
                    if source_out > 0:
                        incoming_sum += pr[edge.source_id] / source_out

                new_pr[node.id] = (1 - damping) / n + damping * incoming_sum
                max_diff = max(max_diff, abs(new_pr[node.id] - pr[node.id]))

            pr = new_pr

            # Early termination on convergence
            if max_diff < tolerance:
                break

        return pr

    def pagerank_top_n(
        self,
        n: int = 10,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Get top N nodes by PageRank score.

        Args:
            n: Number of top nodes to return
            damping: PageRank damping factor
            iterations: Maximum iterations

        Returns:
            List of (node_id, score) tuples sorted by score descending
        """
        scores = self.compute_pagerank(damping, iterations)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n]


class ClusteringMixin:
    """
    Mixin providing clustering/community detection algorithms.

    Label propagation is a fast, scalable algorithm for finding
    communities in graphs. Each node takes the most common label
    among its neighbors.

    Performance: O(E × iterations) where E = edge count

    Example:
        class MyGraph(BaseGraph, ClusteringMixin):
            pass

        graph = MyGraph()
        # ... add nodes and edges ...
        clusters = graph.label_propagation()
        # {node_id: cluster_id, ...}
    """

    def label_propagation(
        self,
        max_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Community detection using label propagation.

        Algorithm:
            1. Initialize each node with unique label
            2. In random order, each node adopts most common neighbor label
            3. Repeat until convergence or max_iterations

        Args:
            max_iterations: Maximum iterations before stopping
            seed: Random seed for reproducibility

        Returns:
            Dict mapping node_id to cluster_id
        """
        if seed is not None:
            random.seed(seed)

        # Initialize: each node in its own cluster
        labels: Dict[str, int] = {node.id: i for i, node in enumerate(self.nodes)}

        if self.node_count == 0:
            return labels

        node_ids = list(labels.keys())

        for _ in range(max_iterations):
            changed = False
            random.shuffle(node_ids)

            for node_id in node_ids:
                # Count neighbor labels (weighted by edge weight if available)
                neighbor_labels: Dict[int, float] = {}

                for neighbor_id in self.neighbors(node_id, "both"):
                    label = labels[neighbor_id]

                    # Try to get edge weight
                    edge = self.get_edge(node_id, neighbor_id, "")
                    if edge is None:
                        edge = self.get_edge(neighbor_id, node_id, "")
                    weight = edge.weight if edge else 1.0

                    neighbor_labels[label] = neighbor_labels.get(label, 0) + weight

                if neighbor_labels:
                    # Assign most common neighbor label
                    best_label = max(neighbor_labels, key=neighbor_labels.get)
                    if labels[node_id] != best_label:
                        labels[node_id] = best_label
                        changed = True

            if not changed:
                break

        return labels

    def get_clusters(
        self,
        max_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[int, List[str]]:
        """
        Get clusters as dict mapping cluster_id to list of node_ids.

        Args:
            max_iterations: Maximum iterations for label propagation
            seed: Random seed for reproducibility

        Returns:
            Dict mapping cluster_id to list of node_ids in that cluster
        """
        labels = self.label_propagation(max_iterations, seed)

        clusters: Dict[int, List[str]] = {}
        for node_id, cluster_id in labels.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_id)

        return clusters

    def modularity(self, labels: Dict[str, int]) -> float:
        """
        Compute modularity score for a given clustering.

        Modularity measures the quality of a division into communities.
        Higher values indicate better community structure.

        Args:
            labels: Dict mapping node_id to cluster_id

        Returns:
            Modularity score in range [-0.5, 1.0]
        """
        m = self.edge_count  # Total edges
        if m == 0:
            return 0.0

        # Compute sum of edges within communities - expected edges
        q = 0.0

        for node_i in self.nodes:
            for node_j in self.nodes:
                if labels[node_i.id] != labels[node_j.id]:
                    continue

                # Actual edge weight
                edge = self.get_edge(node_i.id, node_j.id, "")
                a_ij = edge.weight if edge else 0.0

                # Expected edge weight (null model)
                k_i = self.degree(node_i.id, "both")
                k_j = self.degree(node_j.id, "both")
                expected = (k_i * k_j) / (2 * m)

                q += (a_ij - expected)

        return q / (2 * m)


class SpreadingActivationMixin:
    """
    Mixin providing spreading activation algorithm.

    Spreading activation simulates how neural activation spreads
    through a network. Starting from a source node, activation
    propagates along edges with decay.

    Used for:
    - Priming (activate related concepts)
    - Relevance ranking (distance from query)
    - Attention mechanisms

    Example:
        class MyGraph(BaseGraph, SpreadingActivationMixin):
            pass

        graph = MyGraph()
        # ... add nodes and edges ...
        activations = graph.spread_activation("concept_A", max_hops=3)
    """

    def spread_activation(
        self,
        source_id: str,
        initial_activation: float = 1.0,
        decay: float = 0.5,
        max_hops: int = 3,
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        Spread activation from source through the graph.

        Algorithm:
            1. Source node gets initial_activation
            2. For each hop, activation spreads to neighbors
            3. Activation = parent_activation × edge_weight × decay^hop
            4. Stop when below threshold or max_hops reached

        Args:
            source_id: Starting node ID
            initial_activation: Starting activation level (default 1.0)
            decay: Decay factor per hop (0-1, default 0.5)
            max_hops: Maximum propagation distance
            threshold: Minimum activation to propagate

        Returns:
            Dict mapping node_id to activation level
        """
        if not self.has_node(source_id):
            raise ValueError(f"Source node '{source_id}' not found")

        activations: Dict[str, float] = {source_id: initial_activation}
        frontier = [source_id]

        for hop in range(max_hops):
            current_decay = decay ** (hop + 1)
            next_frontier: List[str] = []

            for node_id in frontier:
                parent_activation = activations.get(node_id, 0)

                for edge in self.edges_from(node_id):
                    target = edge.target_id
                    new_activation = parent_activation * edge.weight * current_decay

                    if new_activation < threshold:
                        continue

                    # Take maximum activation (allows multiple paths)
                    activations[target] = max(
                        activations.get(target, 0),
                        new_activation,
                    )

                    if target not in next_frontier:
                        next_frontier.append(target)

            frontier = next_frontier

            if not frontier:
                break

        return activations

    def multi_source_activation(
        self,
        sources: Dict[str, float],
        decay: float = 0.5,
        max_hops: int = 3,
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        Spread activation from multiple sources simultaneously.

        Args:
            sources: Dict mapping source_id to initial activation
            decay: Decay factor per hop
            max_hops: Maximum propagation distance
            threshold: Minimum activation to propagate

        Returns:
            Dict mapping node_id to combined activation level
        """
        combined: Dict[str, float] = {}

        for source_id, initial_activation in sources.items():
            activations = self.spread_activation(
                source_id,
                initial_activation,
                decay,
                max_hops,
                threshold,
            )

            for node_id, activation in activations.items():
                combined[node_id] = max(combined.get(node_id, 0), activation)

        return combined


class CentralityMixin:
    """
    Mixin providing additional centrality measures.

    Centrality measures identify important nodes in different ways:
    - Degree centrality: Number of connections
    - Betweenness centrality: Bridge nodes on shortest paths
    - Closeness centrality: Average distance to all nodes
    """

    def degree_centrality(self, direction: str = "both") -> Dict[str, float]:
        """
        Compute degree centrality for all nodes.

        Degree centrality = node_degree / (n - 1)
        where n = total nodes

        Args:
            direction: "out", "in", or "both"

        Returns:
            Dict mapping node_id to centrality score (0-1)
        """
        n = self.node_count
        if n <= 1:
            return {node.id: 0.0 for node in self.nodes}

        max_degree = n - 1
        return {
            node.id: self.degree(node.id, direction) / max_degree
            for node in self.nodes
        }

    def closeness_centrality(self) -> Dict[str, float]:
        """
        Compute closeness centrality for all nodes.

        Closeness centrality = (n - 1) / sum(shortest_path_lengths)
        Nodes that are closer to all others have higher closeness.

        Returns:
            Dict mapping node_id to centrality score (0-1)
        """
        n = self.node_count
        if n <= 1:
            return {node.id: 0.0 for node in self.nodes}

        centrality: Dict[str, float] = {}

        for node in self.nodes:
            # BFS to find shortest paths to all other nodes
            distances = self._bfs_distances(node.id)
            total_distance = sum(distances.values())

            if total_distance > 0 and len(distances) > 1:
                centrality[node.id] = (len(distances) - 1) / total_distance
            else:
                centrality[node.id] = 0.0

        return centrality

    def _bfs_distances(self, start_id: str) -> Dict[str, int]:
        """Compute shortest path distances from start to all reachable nodes."""
        from collections import deque

        distances: Dict[str, int] = {start_id: 0}
        queue: deque = deque([start_id])

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            for neighbor_id in self.neighbors(current, "both"):
                if neighbor_id not in distances:
                    distances[neighbor_id] = current_dist + 1
                    queue.append(neighbor_id)

        return distances
