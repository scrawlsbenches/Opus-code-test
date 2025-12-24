"""
Graph Walker with Visitor Pattern for Graph of Thought.

Provides flexible graph traversal with:
- BFS and DFS traversal strategies
- Visitor pattern for node processing
- Filtering and depth limiting
- Edge type filtering
- Directional traversal (forward/reverse)

Example:
    # Count tasks by status while traversing
    def count_by_status(node, acc):
        status = node.properties.get("status", "unknown")
        acc[status] = acc.get(status, 0) + 1
        return acc

    result = (
        GraphWalker(manager)
        .starting_from(root_task_id)
        .bfs()
        .filter(lambda n: n.node_type == "task")
        .max_depth(3)
        .visit(count_by_status, initial={})
        .run()
    )
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    TypeVar,
)

from .api import GoTManager
from .types import Task, Edge


T = TypeVar("T")


class TraversalStrategy(Enum):
    """Graph traversal strategy."""
    BFS = auto()
    DFS = auto()


@dataclass
class TraversalNode:
    """Node with traversal metadata."""
    node: Any
    depth: int
    path: List[str] = field(default_factory=list)


class GraphWalker:
    """
    Fluent graph walker with visitor pattern.

    Supports BFS and DFS traversal with flexible filtering,
    depth limiting, and edge type constraints.
    """

    def __init__(self, manager: GoTManager):
        """Initialize walker with GoT manager."""
        self._manager = manager
        self._start_id: Optional[str] = None
        self._strategy: TraversalStrategy = TraversalStrategy.BFS
        self._visitor: Optional[Callable[[Any, T], T]] = None
        self._initial: T = None
        self._filter_fn: Optional[Callable[[Any], bool]] = None
        self._max_depth: Optional[int] = None
        self._edge_types: Optional[List[str]] = None
        self._reverse_direction: bool = False
        self._bidirectional: bool = True  # Follow edges in both directions by default
        self._visited: Set[str] = set()

    def starting_from(self, node_id: str) -> "GraphWalker":
        """
        Set the starting node for traversal.

        Args:
            node_id: ID of node to start from

        Returns:
            Self for chaining
        """
        self._start_id = node_id
        return self

    def bfs(self) -> "GraphWalker":
        """
        Use breadth-first search traversal.

        Returns:
            Self for chaining
        """
        self._strategy = TraversalStrategy.BFS
        return self

    def dfs(self) -> "GraphWalker":
        """
        Use depth-first search traversal.

        Returns:
            Self for chaining
        """
        self._strategy = TraversalStrategy.DFS
        return self

    def visit(
        self,
        visitor: Callable[[Any, T], T],
        initial: T = None
    ) -> "GraphWalker":
        """
        Set visitor function to call at each node.

        The visitor receives (node, accumulator) and returns new accumulator.

        Args:
            visitor: Function to call at each node
            initial: Initial accumulator value

        Returns:
            Self for chaining
        """
        self._visitor = visitor
        self._initial = initial
        return self

    def filter(self, predicate: Callable[[Any], bool]) -> "GraphWalker":
        """
        Filter nodes during traversal.

        Nodes that don't match the predicate are not visited.

        Args:
            predicate: Function returning True for nodes to include

        Returns:
            Self for chaining
        """
        self._filter_fn = predicate
        return self

    def max_depth(self, depth: int) -> "GraphWalker":
        """
        Limit maximum traversal depth.

        Args:
            depth: Maximum depth (0 = start node only)

        Returns:
            Self for chaining
        """
        self._max_depth = depth
        return self

    def follow(self, *edge_types: str) -> "GraphWalker":
        """
        Only follow edges of specified types.

        Args:
            *edge_types: Edge types to follow (e.g., "DEPENDS_ON")

        Returns:
            Self for chaining
        """
        self._edge_types = list(edge_types)
        return self

    def reverse(self) -> "GraphWalker":
        """
        Follow edges in reverse direction only.

        For "A -DEPENDS_ON-> B", normal traversal from A finds B.
        Reverse traversal from B finds A.

        Returns:
            Self for chaining
        """
        self._reverse_direction = True
        self._bidirectional = False
        return self

    def directed(self) -> "GraphWalker":
        """
        Only follow edges in their source->target direction.

        By default, edges are treated as undirected (bidirectional).

        Returns:
            Self for chaining
        """
        self._bidirectional = False
        return self

    def run(self) -> T:
        """
        Execute the traversal and return result.

        Returns:
            Final accumulator value from visitor
        """
        if self._start_id is None:
            return self._initial

        # Build node lookup
        node_map = self._build_node_map()

        # Build adjacency list
        adjacency = self._build_adjacency()

        # Get start node
        start_node = node_map.get(self._start_id)
        if start_node is None:
            return self._initial

        # Initialize accumulator
        acc = self._initial

        # Reset visited set
        self._visited = set()

        # Execute traversal
        if self._strategy == TraversalStrategy.BFS:
            acc = self._bfs_traverse(start_node, node_map, adjacency, acc)
        else:
            acc = self._dfs_traverse(start_node, node_map, adjacency, acc, 0)

        return acc

    def iter(self) -> Generator[Any, None, None]:
        """
        Iterate over visited nodes.

        Yields:
            Nodes in traversal order
        """
        if self._start_id is None:
            return

        node_map = self._build_node_map()
        adjacency = self._build_adjacency()

        start_node = node_map.get(self._start_id)
        if start_node is None:
            return

        self._visited = set()

        if self._strategy == TraversalStrategy.BFS:
            yield from self._bfs_iterate(start_node, node_map, adjacency)
        else:
            yield from self._dfs_iterate(start_node, node_map, adjacency, 0)

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _build_node_map(self) -> Dict[str, Any]:
        """Build mapping of node IDs to nodes."""
        node_map = {}

        # Add all tasks
        for task in self._manager.list_all_tasks():
            # Create a simple object with id and properties for compatibility
            node_map[task.id] = task

        # Add sprints
        for sprint in self._manager.list_sprints():
            node_map[sprint.id] = sprint

        # Add decisions
        for decision in self._manager.list_decisions():
            node_map[decision.id] = decision

        return node_map

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adjacency: Dict[str, List[str]] = {}

        for edge in self._manager.list_edges():
            # Check edge type filter
            if self._edge_types and edge.edge_type not in self._edge_types:
                continue

            if self._bidirectional:
                # Bidirectional: add both directions
                if edge.source_id not in adjacency:
                    adjacency[edge.source_id] = []
                adjacency[edge.source_id].append(edge.target_id)

                if edge.target_id not in adjacency:
                    adjacency[edge.target_id] = []
                adjacency[edge.target_id].append(edge.source_id)
            elif self._reverse_direction:
                # Reverse: target -> source
                if edge.target_id not in adjacency:
                    adjacency[edge.target_id] = []
                adjacency[edge.target_id].append(edge.source_id)
            else:
                # Normal: source -> target
                if edge.source_id not in adjacency:
                    adjacency[edge.source_id] = []
                adjacency[edge.source_id].append(edge.target_id)

        return adjacency

    def _should_visit(self, node: Any) -> bool:
        """Check if node should be visited."""
        if self._filter_fn is not None:
            return self._filter_fn(node)
        return True

    def _bfs_traverse(
        self,
        start: Any,
        node_map: Dict[str, Any],
        adjacency: Dict[str, List[str]],
        acc: T
    ) -> T:
        """BFS traversal with visitor."""
        queue = deque([(start, 0)])  # (node, depth)
        self._visited.add(start.id)

        while queue:
            node, depth = queue.popleft()

            # Check depth limit
            if self._max_depth is not None and depth > self._max_depth:
                continue

            # Check filter
            if not self._should_visit(node):
                continue

            # Visit node
            if self._visitor:
                acc = self._visitor(node, acc)

            # Queue neighbors
            for neighbor_id in adjacency.get(node.id, []):
                if neighbor_id not in self._visited:
                    self._visited.add(neighbor_id)
                    neighbor = node_map.get(neighbor_id)
                    if neighbor:
                        queue.append((neighbor, depth + 1))

        return acc

    def _bfs_iterate(
        self,
        start: Any,
        node_map: Dict[str, Any],
        adjacency: Dict[str, List[str]]
    ) -> Generator[Any, None, None]:
        """BFS iteration yielding nodes."""
        queue = deque([(start, 0)])
        self._visited.add(start.id)

        while queue:
            node, depth = queue.popleft()

            if self._max_depth is not None and depth > self._max_depth:
                continue

            if not self._should_visit(node):
                continue

            yield node

            for neighbor_id in adjacency.get(node.id, []):
                if neighbor_id not in self._visited:
                    self._visited.add(neighbor_id)
                    neighbor = node_map.get(neighbor_id)
                    if neighbor:
                        queue.append((neighbor, depth + 1))

    def _dfs_traverse(
        self,
        node: Any,
        node_map: Dict[str, Any],
        adjacency: Dict[str, List[str]],
        acc: T,
        depth: int
    ) -> T:
        """DFS traversal with visitor."""
        # Check depth limit
        if self._max_depth is not None and depth > self._max_depth:
            return acc

        # Mark visited
        self._visited.add(node.id)

        # Check filter
        if not self._should_visit(node):
            return acc

        # Visit node
        if self._visitor:
            acc = self._visitor(node, acc)

        # Recurse to neighbors
        for neighbor_id in adjacency.get(node.id, []):
            if neighbor_id not in self._visited:
                neighbor = node_map.get(neighbor_id)
                if neighbor:
                    acc = self._dfs_traverse(
                        neighbor, node_map, adjacency, acc, depth + 1
                    )

        return acc

    def _dfs_iterate(
        self,
        node: Any,
        node_map: Dict[str, Any],
        adjacency: Dict[str, List[str]],
        depth: int
    ) -> Generator[Any, None, None]:
        """DFS iteration yielding nodes."""
        if self._max_depth is not None and depth > self._max_depth:
            return

        self._visited.add(node.id)

        if not self._should_visit(node):
            return

        yield node

        for neighbor_id in adjacency.get(node.id, []):
            if neighbor_id not in self._visited:
                neighbor = node_map.get(neighbor_id)
                if neighbor:
                    yield from self._dfs_iterate(
                        neighbor, node_map, adjacency, depth + 1
                    )
