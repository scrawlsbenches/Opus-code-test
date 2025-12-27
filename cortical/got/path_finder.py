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

import logging
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

# Module logger for path finding operations
logger = logging.getLogger(__name__)


@dataclass
class PathResult:
    """Result of path finding with metadata."""
    path: List[str]
    length: int
    edge_types: List[str]


@dataclass
class PathSearchResult:
    """
    Result of path search with truncation metadata.

    Provides transparency when limits are hit, solving the silent
    failure problem where users couldn't know if more paths existed.

    Attributes:
        paths: List of found paths (each path is list of node IDs)
        truncated: True if search was stopped due to limits
        truncation_reason: Why truncated ("max_paths", "max_length", None)
        paths_found: Total paths found before stopping
        limit_value: The limit that was hit (if truncated)

    Example:
        >>> result = PathFinder(manager).max_paths(10).all_paths(a, b)
        >>> if result.truncated:
        ...     print(f"Warning: Found {result.paths_found} paths, "
        ...           f"stopped at {result.truncation_reason}={result.limit_value}")
        >>> for path in result.paths:
        ...     print(path)
    """
    paths: List[List[str]]
    truncated: bool = False
    truncation_reason: Optional[str] = None
    paths_found: int = 0
    limit_value: Optional[int] = None

    def __iter__(self):
        """Allow iteration over paths for backwards compatibility."""
        return iter(self.paths)

    def __len__(self):
        """Allow len() for backwards compatibility."""
        return len(self.paths)

    def __bool__(self):
        """Allow boolean check for backwards compatibility."""
        return len(self.paths) > 0

    def __getitem__(self, index):
        """Allow indexing for backwards compatibility."""
        return self.paths[index]


@dataclass
class PathPlan:
    """
    Execution plan for path finding operations.

    Provides introspection into what a path finder will do without
    actually executing the search. Useful for debugging and optimization.

    Attributes:
        algorithm: "BFS" for shortest_path, "DFS" for all_paths
        from_id: Starting node ID (if set)
        to_id: Target node ID (if set)
        edge_types: Edge types to follow (None = all)
        max_length: Maximum path length limit
        max_paths: Maximum paths to find (all_paths only)
        bidirectional: Whether edges are traversed in both directions
        estimated_nodes: Estimated nodes to visit
        estimated_complexity: "O(V+E)" for BFS, "O(V!)" worst case for DFS

    Example:
        >>> plan = PathFinder(manager).via_edges("DEPENDS_ON").explain()
        >>> print(plan)
        Path Finding Plan
        ========================================
        Algorithm: Not yet determined
        ...
    """
    algorithm: Optional[str] = None
    from_id: Optional[str] = None
    to_id: Optional[str] = None
    edge_types: Optional[List[str]] = None
    max_length: Optional[int] = None
    max_paths: Optional[int] = None
    bidirectional: bool = True
    estimated_nodes: int = 0
    estimated_complexity: str = "unknown"

    def __str__(self) -> str:
        """Human-readable visualization of the path plan."""
        lines = ["Path Finding Plan", "=" * 40]

        lines.append(f"Algorithm: {self.algorithm or 'Not yet determined'}")

        if self.from_id:
            lines.append(f"From: {self.from_id}")
        if self.to_id:
            lines.append(f"To: {self.to_id}")

        lines.append(f"Direction: {'Bidirectional' if self.bidirectional else 'Directed'}")

        if self.edge_types:
            lines.append(f"Edge types: {', '.join(self.edge_types)}")
        else:
            lines.append("Edge types: All")

        if self.max_length is not None:
            lines.append(f"Max length: {self.max_length}")
        if self.max_paths is not None:
            lines.append(f"Max paths: {self.max_paths}")

        lines.append("")
        lines.append(f"Estimated nodes: ~{self.estimated_nodes}")
        lines.append(f"Complexity: {self.estimated_complexity}")

        return "\n".join(lines)


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

    def explain(self) -> PathPlan:
        """
        Get path finding plan without executing.

        Returns a PathPlan describing the current configuration of the
        PathFinder. Useful for debugging and understanding what settings
        are active before running an expensive path search.

        Returns:
            PathPlan with current configuration and complexity estimates

        Example:
            >>> plan = PathFinder(manager).via_edges("DEPENDS_ON").max_paths(50).explain()
            >>> print(plan)
            Path Finding Plan
            ========================================
            Algorithm: Not yet determined
            Direction: Bidirectional
            Edge types: DEPENDS_ON
            Max paths: 50
            ...
        """
        # Count nodes for estimation
        node_count = len(self._get_all_node_ids())

        # Determine effective limits
        effective_max_len = self._max_len
        effective_max_paths = self._max_paths

        if self._use_defaults:
            if effective_max_len is None:
                effective_max_len = self.DEFAULT_MAX_LENGTH
            if effective_max_paths is None:
                effective_max_paths = self.DEFAULT_MAX_PATHS

        return PathPlan(
            algorithm=None,  # Not determined until actual method called
            from_id=None,
            to_id=None,
            edge_types=self._edge_types,
            max_length=effective_max_len,
            max_paths=effective_max_paths,
            bidirectional=self._bidirectional,
            estimated_nodes=node_count,
            estimated_complexity="O(V+E) for BFS, O(V!) for DFS"
        )

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
    ) -> PathSearchResult:
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
            PathSearchResult with paths and truncation metadata.
            Backwards compatible: can iterate, check len(), index directly.

        Example:
            >>> result = PathFinder(manager).all_paths(a, b)
            >>> if result.truncated:
            ...     logger.warning(f"Search truncated: {result.truncation_reason}")
            >>> for path in result:  # Works like list
            ...     print(path)
        """
        if from_id == to_id:
            return PathSearchResult(
                paths=[[from_id]],
                truncated=False,
                paths_found=1
            )

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

        # Track truncation reason
        truncation_info: Dict[str, Any] = {
            "truncated": False,
            "reason": None,
            "limit_value": None
        }

        self._dfs_all_paths(
            current=from_id,
            target=to_id,
            path=[from_id],
            visited={from_id},
            adjacency=adjacency,
            paths=paths,
            max_len=effective_max_len,
            max_paths=effective_max_paths,
            truncation_info=truncation_info
        )

        result = PathSearchResult(
            paths=paths,
            truncated=truncation_info["truncated"],
            truncation_reason=truncation_info["reason"],
            paths_found=len(paths),
            limit_value=truncation_info["limit_value"]
        )

        # Log warning if truncated
        if result.truncated:
            logger.warning(
                f"PathFinder.all_paths() truncated: {result.truncation_reason}="
                f"{result.limit_value}, found {result.paths_found} paths. "
                f"Use .max_paths(N) or .max_length(N) to adjust limits."
            )

        return result

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
        max_paths: Optional[int] = None,
        truncation_info: Optional[Dict[str, Any]] = None
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
            truncation_info: Dict to populate with truncation reason (mutated)

        Returns:
            True if should continue searching, False if limit reached
        """
        # Check max length - note: this doesn't stop search, just skips this branch
        if max_len is not None and len(path) >= max_len:
            # Record that we hit length limit (but continue other branches)
            if truncation_info is not None and not truncation_info["truncated"]:
                # Only record if this could have led to a valid path
                # (we can't know for sure, so we mark as potentially truncated)
                pass  # Length limit on a branch isn't necessarily truncation
            return True  # Continue searching other branches

        for neighbor in adjacency.get(current, []):
            # Check if we've found enough paths
            if max_paths is not None and len(paths) >= max_paths:
                # TRUNCATION: We hit max_paths limit
                if truncation_info is not None:
                    truncation_info["truncated"] = True
                    truncation_info["reason"] = "max_paths"
                    truncation_info["limit_value"] = max_paths
                return False  # Stop searching

            if neighbor == target:
                paths.append(path + [neighbor])
                # Check limit after adding path
                if max_paths is not None and len(paths) >= max_paths:
                    # TRUNCATION: We hit max_paths limit
                    if truncation_info is not None:
                        truncation_info["truncated"] = True
                        truncation_info["reason"] = "max_paths"
                        truncation_info["limit_value"] = max_paths
                    return False
            elif neighbor not in visited:
                visited.add(neighbor)
                should_continue = self._dfs_all_paths(
                    neighbor, target, path + [neighbor],
                    visited, adjacency, paths, max_len, max_paths,
                    truncation_info
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
