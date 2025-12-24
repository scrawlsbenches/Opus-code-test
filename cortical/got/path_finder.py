"""
Path Finder for Graph of Thought.

Implements efficient path finding algorithms:
- BFS for shortest path (unweighted graphs)
- DFS for all paths enumeration
- Edge type filtering
- Path length constraints

Example:
    # Find shortest path
    path = PathFinder(manager).shortest_path(start_id, end_id)

    # Find all paths with constraints
    paths = (
        PathFinder(manager)
        .via_edges("DEPENDS_ON", "BLOCKS")
        .max_length(5)
        .all_paths(start_id, end_id)
    )
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .api import GoTManager


@dataclass
class PathResult:
    """Result of path finding with metadata."""
    path: List[str]
    length: int
    edge_types: List[str]


class PathFinder:
    """
    Fluent path finder for GoT graphs.

    Uses BFS for shortest path and DFS for all paths.
    Supports edge type filtering and length constraints.
    """

    def __init__(self, manager: GoTManager):
        """Initialize path finder with GoT manager."""
        self._manager = manager
        self._edge_types: Optional[List[str]] = None
        self._max_len: Optional[int] = None
        self._bidirectional: bool = True  # Follow edges in both directions

    def via_edges(self, *edge_types: str) -> "PathFinder":
        """
        Only consider edges of specified types.

        Args:
            *edge_types: Edge types to follow

        Returns:
            Self for chaining
        """
        self._edge_types = list(edge_types)
        return self

    def max_length(self, length: int) -> "PathFinder":
        """
        Set maximum path length (number of nodes).

        Args:
            length: Maximum path length

        Returns:
            Self for chaining
        """
        self._max_len = length
        return self

    def directed(self) -> "PathFinder":
        """
        Only follow edges in their source->target direction.

        By default, edges are treated as undirected.

        Returns:
            Self for chaining
        """
        self._bidirectional = False
        return self

    def shortest_path(
        self,
        from_id: str,
        to_id: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.

        Args:
            from_id: Starting node ID
            to_id: Target node ID

        Returns:
            List of node IDs in path, or None if no path exists
        """
        if from_id == to_id:
            return [from_id]

        adjacency = self._build_adjacency()

        # BFS
        queue = deque([(from_id, [from_id])])
        visited = {from_id}

        while queue:
            current, path = queue.popleft()

            # Check max length
            if self._max_len is not None and len(path) >= self._max_len:
                continue

            # Check neighbors
            for neighbor in adjacency.get(current, []):
                if neighbor == to_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def all_paths(
        self,
        from_id: str,
        to_id: str
    ) -> List[List[str]]:
        """
        Find all paths between two nodes using DFS.

        Warning: Can be expensive for large graphs with many paths.

        Args:
            from_id: Starting node ID
            to_id: Target node ID

        Returns:
            List of paths (each path is list of node IDs)
        """
        if from_id == to_id:
            return [[from_id]]

        adjacency = self._build_adjacency()
        paths: List[List[str]] = []

        self._dfs_all_paths(
            current=from_id,
            target=to_id,
            path=[from_id],
            visited={from_id},
            adjacency=adjacency,
            paths=paths
        )

        return paths

    def path_exists(self, from_id: str, to_id: str) -> bool:
        """
        Check if any path exists between two nodes.

        More efficient than shortest_path when you only need existence.

        Args:
            from_id: Starting node ID
            to_id: Target node ID

        Returns:
            True if path exists
        """
        return self.shortest_path(from_id, to_id) is not None

    def reachable_from(self, node_id: str) -> Set[str]:
        """
        Find all nodes reachable from a given node.

        Args:
            node_id: Starting node ID

        Returns:
            Set of reachable node IDs
        """
        adjacency = self._build_adjacency()
        reachable = set()

        self._dfs_reachable(node_id, adjacency, reachable)

        return reachable

    def connected_components(self) -> List[Set[str]]:
        """
        Find all connected components in the graph.

        Returns:
            List of sets, each containing node IDs in a component
        """
        adjacency = self._build_adjacency()
        all_nodes = self._get_all_node_ids()

        components = []
        visited = set()

        for node in all_nodes:
            if node not in visited:
                component = set()
                self._dfs_reachable(node, adjacency, component)
                visited.update(component)
                components.append(component)

        return components

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adjacency: Dict[str, List[str]] = {}

        for edge in self._manager.list_edges():
            # Check edge type filter
            if self._edge_types and edge.edge_type not in self._edge_types:
                continue

            # Add source -> target
            if edge.source_id not in adjacency:
                adjacency[edge.source_id] = []
            adjacency[edge.source_id].append(edge.target_id)

            # Add target -> source (bidirectional)
            if self._bidirectional:
                if edge.target_id not in adjacency:
                    adjacency[edge.target_id] = []
                adjacency[edge.target_id].append(edge.source_id)

        return adjacency

    def _get_all_node_ids(self) -> Set[str]:
        """Get all node IDs in the graph."""
        nodes = set()

        for task in self._manager.list_all_tasks():
            nodes.add(task.id)

        for sprint in self._manager.list_sprints():
            nodes.add(sprint.id)

        for decision in self._manager.list_decisions():
            nodes.add(decision.id)

        return nodes

    def _dfs_all_paths(
        self,
        current: str,
        target: str,
        path: List[str],
        visited: Set[str],
        adjacency: Dict[str, List[str]],
        paths: List[List[str]]
    ) -> None:
        """DFS to find all paths."""
        # Check max length
        if self._max_len is not None and len(path) >= self._max_len:
            return

        for neighbor in adjacency.get(current, []):
            if neighbor == target:
                paths.append(path + [neighbor])
            elif neighbor not in visited:
                visited.add(neighbor)
                self._dfs_all_paths(
                    neighbor, target, path + [neighbor],
                    visited, adjacency, paths
                )
                visited.remove(neighbor)

    def _dfs_reachable(
        self,
        node: str,
        adjacency: Dict[str, List[str]],
        reachable: Set[str]
    ) -> None:
        """DFS to find all reachable nodes."""
        if node in reachable:
            return

        reachable.add(node)

        for neighbor in adjacency.get(node, []):
            self._dfs_reachable(neighbor, adjacency, reachable)
