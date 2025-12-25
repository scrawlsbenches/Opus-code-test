"""
Path Finder for Graph of Thought.

This module provides path-finding algorithms for analyzing relationships
between entities in the GoT graph. Useful for:
- Dependency chain analysis
- Finding blocking relationships
- Detecting connection paths between tasks

ALGORITHMS
----------
shortest_path (BFS):
    Finds the shortest path between two nodes using Breadth-First Search.
    Optimal for unweighted graphs where all edges have equal cost.
    Time complexity: O(V + E) where V=vertices, E=edges

all_paths (DFS):
    Finds ALL possible paths between two nodes using Depth-First Search.
    WARNING: Can be expensive for densely connected graphs!
    Use max_length() to limit search depth.

EDGE DIRECTION
--------------
By default, edges are treated as BIDIRECTIONAL. This means:
    A --DEPENDS_ON--> B
Can be traversed in either direction for path finding.

Use .directed() to only follow source->target direction.

USAGE EXAMPLES
--------------

Find shortest dependency path:
    >>> path = PathFinder(manager).shortest_path(task_a, task_b)
    >>> if path:
    ...     print(f"Path length: {len(path)}")
    ...     print(" -> ".join(path))

Find all paths (with safety limit):
    >>> paths = PathFinder(manager).max_length(5).all_paths(start, end)
    >>> print(f"Found {len(paths)} paths")

Check if two tasks are connected:
    >>> if PathFinder(manager).path_exists(task_a, task_b):
    ...     print("Tasks are connected!")

Find all reachable nodes:
    >>> reachable = PathFinder(manager).reachable_from(start_task)
    >>> print(f"{len(reachable)} tasks in dependency chain")

Find disconnected components:
    >>> components = PathFinder(manager).connected_components()
    >>> print(f"{len(components)} isolated groups")

PERFORMANCE NOTES
-----------------
- shortest_path is O(V+E) - safe for large graphs
- all_paths can be O(V!) in worst case - USE max_length!
- path_exists stops at first path - more efficient than shortest_path
- reachable_from uses DFS - memory efficient
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

    SAFETY: all_paths() has defaults to prevent exponential blowup:
    - max_paths=100: Stop after finding 100 paths
    - max_length=10: Don't explore paths longer than 10 nodes

    Override with .max_paths(None) or .max_length(None) if needed,
    but be aware of the O(V!) worst-case complexity.
    """

    # Default limits to prevent exponential blowup
    DEFAULT_MAX_PATHS = 100
    DEFAULT_MAX_LENGTH = 10

    def __init__(self, manager: GoTManager):
        """Initialize path finder with GoT manager."""
        self._manager = manager
        self._edge_types: Optional[List[str]] = None
        self._max_len: Optional[int] = None
        self._max_paths: Optional[int] = None
        self._bidirectional: bool = True  # Follow edges in both directions
        self._use_defaults: bool = True  # Apply safety defaults

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

    def max_length(self, length: Optional[int]) -> "PathFinder":
        """
        Set maximum path length (number of nodes).

        For all_paths(), the default is 10 to prevent exponential blowup.
        Pass None to remove the limit (use with caution!).

        Args:
            length: Maximum path length, or None to remove limit

        Returns:
            Self for chaining
        """
        self._max_len = length
        self._use_defaults = False  # Explicit setting overrides defaults
        return self

    def max_paths(self, count: Optional[int]) -> "PathFinder":
        """
        Set maximum number of paths to find in all_paths().

        For all_paths(), the default is 100 to prevent exponential blowup.
        Pass None to remove the limit (use with caution!).

        Args:
            count: Maximum number of paths, or None to remove limit

        Returns:
            Self for chaining
        """
        self._max_paths = count
        self._use_defaults = False  # Explicit setting overrides defaults
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

        SAFETY: Has default limits to prevent exponential blowup:
        - max_paths=100: Stop after finding 100 paths
        - max_length=10: Don't explore paths longer than 10 nodes

        Use .max_paths(N) or .max_length(N) to override.
        Use .max_paths(None) or .max_length(None) to remove limits.

        Args:
            from_id: Starting node ID
            to_id: Target node ID

        Returns:
            List of paths (each path is list of node IDs)
        """
        if from_id == to_id:
            return [[from_id]]

        # Apply safety defaults if no explicit limits set
        effective_max_len = self._max_len
        effective_max_paths = self._max_paths

        if self._use_defaults:
            if effective_max_len is None:
                effective_max_len = self.DEFAULT_MAX_LENGTH
            if effective_max_paths is None:
                effective_max_paths = self.DEFAULT_MAX_PATHS

        adjacency = self._build_adjacency()
        paths: List[List[str]] = []

        self._dfs_all_paths(
            current=from_id,
            target=to_id,
            path=[from_id],
            visited={from_id},
            adjacency=adjacency,
            paths=paths,
            max_len=effective_max_len,
            max_paths=effective_max_paths
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
        paths: List[List[str]],
        max_len: Optional[int] = None,
        max_paths: Optional[int] = None
    ) -> bool:
        """
        DFS to find all paths.

        Args:
            current: Current node in traversal
            target: Target node to find
            path: Current path being explored
            visited: Set of visited nodes in current path
            adjacency: Adjacency list
            paths: Accumulator for found paths
            max_len: Maximum path length (None = no limit)
            max_paths: Maximum number of paths to find (None = no limit)

        Returns:
            True if should continue searching, False if max_paths reached
        """
        # Check max length
        if max_len is not None and len(path) >= max_len:
            return True  # Continue searching other branches

        for neighbor in adjacency.get(current, []):
            # Check if we've found enough paths
            if max_paths is not None and len(paths) >= max_paths:
                return False  # Stop searching

            if neighbor == target:
                paths.append(path + [neighbor])
                # Check limit after adding path
                if max_paths is not None and len(paths) >= max_paths:
                    return False
            elif neighbor not in visited:
                visited.add(neighbor)
                should_continue = self._dfs_all_paths(
                    neighbor, target, path + [neighbor],
                    visited, adjacency, paths, max_len, max_paths
                )
                visited.remove(neighbor)
                if not should_continue:
                    return False

        return True  # Continue searching

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
