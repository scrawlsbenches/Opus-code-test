"""
Concrete Graph Implementations built on BaseGraph.

This module provides ready-to-use graph implementations for common
use cases. These can be used directly or as templates for custom
graph types.

Available Implementations:
- SimpleGraph: Lightweight general-purpose graph
- DAGGraph: Directed acyclic graph with cycle prevention
- WeightedGraph: Graph with weighted edges for pathfinding

Usage:
    from cortical.graph import SimpleGraph

    graph = SimpleGraph()
    graph.add_node("A", content="Node A")
    graph.add_node("B", content="Node B")
    graph.add_edge("A", "B", edge_type="connects")

See docs/base-graph-design.md for architecture details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseGraph
from .protocols import NodeBase, EdgeBase
from .algorithms import (
    PageRankMixin,
    ClusteringMixin,
    SpreadingActivationMixin,
    CentralityMixin,
)


# =============================================================================
# Simple Node/Edge for general use
# =============================================================================


@dataclass
class SimpleNode(NodeBase):
    """
    Simple node with minimal overhead.

    Inherits all fields from NodeBase:
    - id, node_type, content, properties, metadata, created_at, modified_at

    Use this for general-purpose graphs where you don't need
    domain-specific node attributes.
    """
    pass


@dataclass
class SimpleEdge(EdgeBase):
    """
    Simple edge with minimal overhead.

    Inherits all fields from EdgeBase:
    - source_id, target_id, edge_type, weight, bidirectional, properties, created_at

    Use this for general-purpose graphs where you don't need
    domain-specific edge attributes.
    """
    pass


# =============================================================================
# SimpleGraph: General-purpose graph
# =============================================================================


class SimpleGraph(
    BaseGraph[SimpleNode, SimpleEdge],
    PageRankMixin,
    ClusteringMixin,
    SpreadingActivationMixin,
    CentralityMixin,
):
    """
    General-purpose graph implementation.

    A complete, ready-to-use graph with all standard algorithms:
    - PageRank centrality
    - Community detection (label propagation)
    - Spreading activation
    - Degree/closeness centrality

    Example:
        graph = SimpleGraph()

        # Add nodes
        graph.add_node("A", content="Concept A", node_type="concept")
        graph.add_node("B", content="Concept B", node_type="concept")
        graph.add_node("C", content="Concept C", node_type="concept")

        # Add edges
        graph.add_edge("A", "B", edge_type="related", weight=0.8)
        graph.add_edge("B", "C", edge_type="related", weight=0.6)

        # Compute algorithms
        pagerank = graph.compute_pagerank()
        clusters = graph.label_propagation()
        activations = graph.spread_activation("A")

        # Traverse
        path = graph.shortest_path("A", "C")
        neighbors = graph.neighbors("B", direction="both")
    """

    def _create_node(self, id: str, **kwargs: Any) -> SimpleNode:
        """Create a SimpleNode."""
        return SimpleNode(
            id=id,
            node_type=kwargs.get("node_type", ""),
            content=kwargs.get("content", ""),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
            created_at=kwargs.get("created_at", datetime.now()),
            modified_at=kwargs.get("modified_at", datetime.now()),
        )

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs: Any,
    ) -> SimpleEdge:
        """Create a SimpleEdge."""
        return SimpleEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=kwargs.get("weight", 1.0),
            bidirectional=kwargs.get("bidirectional", False),
            properties=kwargs.get("properties", {}),
            created_at=kwargs.get("created_at", datetime.now()),
        )


# =============================================================================
# DAGGraph: Directed Acyclic Graph
# =============================================================================


class DAGGraph(BaseGraph[SimpleNode, SimpleEdge]):
    """
    Directed Acyclic Graph with automatic cycle prevention.

    Any attempt to add an edge that would create a cycle is rejected.
    Useful for dependency graphs, task scheduling, etc.

    Example:
        dag = DAGGraph()
        dag.add_node("design", content="Design phase")
        dag.add_node("implement", content="Implementation")
        dag.add_node("test", content="Testing")

        # Dependencies: design -> implement -> test
        dag.add_edge("design", "implement")
        dag.add_edge("implement", "test")

        # This would create a cycle, so it raises ValueError
        # dag.add_edge("test", "design")  # Raises ValueError

        # Topological sort for scheduling
        order = dag.topological_sort()  # ["design", "implement", "test"]
    """

    def _create_node(self, id: str, **kwargs: Any) -> SimpleNode:
        """Create a SimpleNode."""
        return SimpleNode(
            id=id,
            node_type=kwargs.get("node_type", ""),
            content=kwargs.get("content", ""),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
        )

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs: Any,
    ) -> SimpleEdge:
        """Create a SimpleEdge."""
        return SimpleEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=kwargs.get("weight", 1.0),
            bidirectional=False,  # DAG edges are always directed
            properties=kwargs.get("properties", {}),
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        weight: float = 1.0,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> SimpleEdge:
        """
        Add an edge, rejecting if it would create a cycle.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            weight: Edge weight
            bidirectional: Ignored (always False for DAG)
            **kwargs: Additional edge attributes

        Returns:
            The created edge

        Raises:
            ValueError: If edge would create a cycle
        """
        # Check for self-loop
        if source_id == target_id:
            raise ValueError(f"Self-loops not allowed: {source_id}")

        # Check if edge already exists (idempotent)
        existing = self.get_edge(source_id, target_id, edge_type)
        if existing is not None:
            return existing

        # Check if adding this edge would create a cycle
        # A cycle would be created if there's already a path from target to source
        if self._has_path(target_id, source_id):
            raise ValueError(
                f"Edge {source_id} -> {target_id} would create a cycle"
            )

        return super().add_edge(
            source_id, target_id, edge_type, weight, False, **kwargs
        )

    def _has_path(self, from_id: str, to_id: str) -> bool:
        """Check if there's a path from from_id to to_id."""
        if not self.has_node(from_id) or not self.has_node(to_id):
            return False

        if from_id == to_id:
            return True

        visited: Set[str] = set()
        stack = [from_id]

        while stack:
            current = stack.pop()
            if current == to_id:
                return True

            if current in visited:
                continue

            visited.add(current)

            for neighbor_id in self.neighbors(current, "out"):
                if neighbor_id not in visited:
                    stack.append(neighbor_id)

        return False

    def blocked_by(self, node_id: str) -> Set[str]:
        """
        Get all nodes that must complete before this one (transitive).

        Args:
            node_id: The node to check

        Returns:
            Set of all predecessor node IDs
        """
        if not self.has_node(node_id):
            return set()

        result: Set[str] = set()
        visited: Set[str] = set()

        # Start with direct predecessors
        stack = list(self.neighbors(node_id, "in"))

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            result.add(current)

            for predecessor_id in self.neighbors(current, "in"):
                if predecessor_id not in visited:
                    stack.append(predecessor_id)

        return result

    def blocks(self, node_id: str) -> Set[str]:
        """
        Get all nodes waiting on this one (transitive).

        Args:
            node_id: The node to check

        Returns:
            Set of all successor node IDs
        """
        if not self.has_node(node_id):
            return set()

        result: Set[str] = set()
        visited: Set[str] = set()

        stack = list(self.neighbors(node_id, "out"))

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            result.add(current)

            for successor_id in self.neighbors(current, "out"):
                if successor_id not in visited:
                    stack.append(successor_id)

        return result

    def ready_tasks(self, completed: Set[str]) -> Set[str]:
        """
        Get tasks that are ready to start given completed tasks.

        A task is ready if all its dependencies are completed.

        Args:
            completed: Set of completed task IDs

        Returns:
            Set of task IDs that can now be started

        Performance: O(V + E) where V = nodes, E = edges
            Optimized with direct storage access and early-exit checks.
        """
        ready: Set[str] = set()

        # Direct access to storage edge index for O(1) lookup per node
        edges_by_target = getattr(self._storage, '_edges_by_target', None)

        if edges_by_target is not None:
            # Fast path: direct index access (InMemoryGraphStorage)
            for node in self._storage.all_nodes():
                node_id = node.id
                incoming_edges = edges_by_target.get(node_id)

                if incoming_edges is None or len(incoming_edges) == 0:
                    # No dependencies - always ready
                    ready.add(node_id)
                else:
                    # Check if all dependencies are completed (early exit on first miss)
                    is_ready = True
                    for edge in incoming_edges:
                        if edge.source_id not in completed:
                            is_ready = False
                            break
                    if is_ready:
                        ready.add(node_id)
        else:
            # Fallback: use neighbors() for other storage backends
            for node in self.nodes:
                dependencies = set(self.neighbors(node.id, "in"))
                if dependencies.issubset(completed):
                    ready.add(node.id)

        return ready


# =============================================================================
# WeightedGraph: For weighted path algorithms
# =============================================================================


@dataclass
class WeightedEdge(EdgeBase):
    """
    Edge with cost attribute for weighted pathfinding.

    In addition to weight (similarity/strength), cost represents
    the traversal cost for algorithms like Dijkstra.
    """
    cost: float = 1.0


class WeightedGraph(BaseGraph[SimpleNode, WeightedEdge]):
    """
    Graph with weighted edges for pathfinding algorithms.

    Provides Dijkstra's algorithm for weighted shortest paths.

    Example:
        graph = WeightedGraph()
        graph.add_node("A", content="Start")
        graph.add_node("B", content="Via B")
        graph.add_node("C", content="Via C")
        graph.add_node("D", content="End")

        graph.add_edge("A", "B", cost=1)
        graph.add_edge("A", "C", cost=4)
        graph.add_edge("B", "D", cost=2)
        graph.add_edge("C", "D", cost=1)

        # Find weighted shortest path
        path, cost = graph.dijkstra("A", "D")
        # path = ["A", "B", "D"], cost = 3
    """

    def _create_node(self, id: str, **kwargs: Any) -> SimpleNode:
        """Create a SimpleNode."""
        return SimpleNode(
            id=id,
            node_type=kwargs.get("node_type", ""),
            content=kwargs.get("content", ""),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
        )

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs: Any,
    ) -> WeightedEdge:
        """Create a WeightedEdge."""
        return WeightedEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=kwargs.get("weight", 1.0),
            bidirectional=kwargs.get("bidirectional", False),
            properties=kwargs.get("properties", {}),
            cost=kwargs.get("cost", 1.0),
        )

    def dijkstra(
        self,
        from_id: str,
        to_id: str,
    ) -> Tuple[Optional[List[str]], float]:
        """
        Find shortest weighted path using Dijkstra's algorithm.

        Args:
            from_id: Starting node ID
            to_id: Target node ID

        Returns:
            Tuple of (path, total_cost) or (None, inf) if no path

        Performance:
            O((V + E) log V) with binary heap
        """
        import heapq

        if not self.has_node(from_id) or not self.has_node(to_id):
            return None, float("inf")

        if from_id == to_id:
            return [from_id], 0.0

        # Distance and predecessor tracking
        distances: Dict[str, float] = {from_id: 0.0}
        predecessors: Dict[str, str] = {}

        # Priority queue: (distance, node_id)
        heap: List[Tuple[float, str]] = [(0.0, from_id)]
        visited: Set[str] = set()

        while heap:
            current_dist, current_id = heapq.heappop(heap)

            if current_id in visited:
                continue

            visited.add(current_id)

            if current_id == to_id:
                break

            for edge in self.edges_from(current_id):
                neighbor_id = edge.target_id

                if neighbor_id in visited:
                    continue

                new_dist = current_dist + edge.cost

                if new_dist < distances.get(neighbor_id, float("inf")):
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(heap, (new_dist, neighbor_id))

        # Reconstruct path
        if to_id not in distances:
            return None, float("inf")

        path = [to_id]
        current = to_id
        while current in predecessors:
            current = predecessors[current]
            path.append(current)

        path.reverse()
        return path, distances[to_id]

    def all_pairs_shortest_paths(self) -> Dict[str, Dict[str, float]]:
        """
        Compute shortest path costs between all pairs of nodes.

        Returns:
            Nested dict: distances[from_id][to_id] = cost

        Performance:
            O(V × (V + E) log V) using Dijkstra from each node
        """
        distances: Dict[str, Dict[str, float]] = {}

        for node in self.nodes:
            distances[node.id] = {}
            _, costs = self._dijkstra_all(node.id)
            distances[node.id] = costs

        return distances

    def _dijkstra_all(self, from_id: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Run Dijkstra from a single source to all nodes."""
        import heapq

        distances: Dict[str, float] = {from_id: 0.0}
        predecessors: Dict[str, str] = {}

        heap: List[Tuple[float, str]] = [(0.0, from_id)]
        visited: Set[str] = set()

        while heap:
            current_dist, current_id = heapq.heappop(heap)

            if current_id in visited:
                continue

            visited.add(current_id)

            for edge in self.edges_from(current_id):
                neighbor_id = edge.target_id

                if neighbor_id in visited:
                    continue

                new_dist = current_dist + edge.cost

                if new_dist < distances.get(neighbor_id, float("inf")):
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(heap, (new_dist, neighbor_id))

        return predecessors, distances
