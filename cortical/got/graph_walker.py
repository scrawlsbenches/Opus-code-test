"""
Graph Walker with Visitor Pattern for Graph of Thought.

This module provides flexible graph traversal using the classic Visitor pattern.
It's designed for tasks like dependency analysis, impact assessment, and
collecting statistics across connected nodes.

TRAVERSAL STRATEGIES
--------------------
- BFS (Breadth-First): Visits nodes level by level. Best for finding
  shortest paths or when you need to process nearby nodes first.

- DFS (Depth-First): Goes deep before backtracking. Best for exploring
  complete branches or when order doesn't matter (uses less memory).

EDGE DIRECTION
--------------
By default, edges are treated as BIDIRECTIONAL (undirected graph).
This is usually what you want for "find all connected nodes".

Direction methods (consistent with PatternMatcher):
- .outgoing(): Only follow source->target direction (same as .directed())
- .incoming(): Only follow target->source direction (same as .reverse())
- .both(): Follow both directions (default, explicit reset)

Legacy methods (still supported):
- .directed(): Only follow source->target direction
- .reverse(): Only follow target->source direction

VISITOR PATTERN
---------------
The visitor receives (node, accumulator) and returns updated accumulator.
This allows stateful traversal without external mutable state.

USAGE EXAMPLES
--------------

Count tasks by status:
    >>> def count_by_status(node, acc):
    ...     acc[node.status] = acc.get(node.status, 0) + 1
    ...     return acc
    >>> result = GraphWalker(manager).starting_from(task_id).bfs() \\
    ...     .visit(count_by_status, initial={}).run()
    {"pending": 3, "completed": 2}

Find all connected task IDs:
    >>> ids = []
    >>> GraphWalker(manager).starting_from(task_id).bfs() \\
    ...     .visit(lambda n, acc: acc + [n.id], initial=[]).run()

Traverse only DEPENDS_ON edges up to depth 3:
    >>> GraphWalker(manager).starting_from(task_id) \\
    ...     .follow("DEPENDS_ON").max_depth(3).bfs().visit(collector).run()

Find what blocks a task (reverse traversal):
    >>> GraphWalker(manager).starting_from(task_id) \\
    ...     .follow("BLOCKS").reverse().bfs().visit(collector).run()

PERFORMANCE NOTES
-----------------
- BFS uses a queue (deque) for O(1) enqueue/dequeue
- DFS uses recursion with explicit visited set to prevent cycles
- Bidirectional doubles the edge lookups but finds all connections
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


@dataclass
class WalkerPlan:
    """
    Execution plan for graph walking operations.

    Provides introspection into what a graph walker will do without
    actually executing the traversal. Useful for debugging and optimization.

    Attributes:
        strategy: "BFS" or "DFS"
        start_id: Starting node ID
        max_depth: Maximum traversal depth (None = unlimited)
        edge_types: Edge types to follow (None = all)
        has_filter: Whether a node filter is configured
        has_visitor: Whether a visitor function is configured
        reverse_direction: Whether traversing in reverse direction
        estimated_nodes: Estimated nodes in the graph

    Example:
        >>> plan = GraphWalker(manager).starting_from(node_id).bfs().explain()
        >>> print(plan)
        Walker Plan
        ========================================
        Strategy: BFS
        Start: T-123
        ...
    """
    strategy: str
    start_id: Optional[str] = None
    max_depth: Optional[int] = None
    edge_types: Optional[List[str]] = None
    has_filter: bool = False
    has_visitor: bool = False
    reverse_direction: bool = False
    estimated_nodes: int = 0

    def __str__(self) -> str:
        """Human-readable visualization of the walker plan."""
        lines = ["Walker Plan", "=" * 40]

        lines.append(f"Strategy: {self.strategy}")

        if self.start_id:
            lines.append(f"Start: {self.start_id}")
        else:
            lines.append("Start: Not configured")

        if self.max_depth is not None:
            lines.append(f"Max depth: {self.max_depth}")
        else:
            lines.append("Max depth: Unlimited")

        if self.edge_types:
            lines.append(f"Edge types: {', '.join(self.edge_types)}")
        else:
            lines.append("Edge types: All")

        lines.append(f"Direction: {'Reverse' if self.reverse_direction else 'Forward'}")
        lines.append(f"Has filter: {self.has_filter}")
        lines.append(f"Has visitor: {self.has_visitor}")

        lines.append("")
        lines.append(f"Estimated nodes: ~{self.estimated_nodes}")

        return "\n".join(lines)


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

    def outgoing(self) -> "GraphWalker":
        """
        Only follow outgoing edges (source->target direction).

        Alias for directed(). Consistent with PatternMatcher.outgoing().

        Example:
            >>> # Find what this task depends on (follow DEPENDS_ON outward)
            >>> GraphWalker(manager).starting_from(task_id) \\
            ...     .follow("DEPENDS_ON").outgoing().bfs().visit(fn).run()

        Returns:
            Self for chaining
        """
        return self.directed()

    def incoming(self) -> "GraphWalker":
        """
        Only follow incoming edges (target->source direction).

        Alias for reverse(). Consistent with PatternMatcher.incoming().

        Example:
            >>> # Find what depends on this task (follow DEPENDS_ON edges pointing here)
            >>> GraphWalker(manager).starting_from(task_id) \\
            ...     .follow("DEPENDS_ON").incoming().bfs().visit(fn).run()

        Returns:
            Self for chaining
        """
        return self.reverse()

    def both(self) -> "GraphWalker":
        """
        Follow edges in both directions (bidirectional).

        This is the default behavior. Useful for resetting after
        calling directed()/outgoing()/incoming()/reverse().

        Consistent with PatternMatcher.both().

        Returns:
            Self for chaining
        """
        self._bidirectional = True
        self._reverse_direction = False
        return self

    def explain(self) -> WalkerPlan:
        """
        Get walker execution plan without executing.

        Returns a WalkerPlan describing the current configuration of the
        GraphWalker. Useful for debugging and understanding what settings
        are active before running an expensive traversal.

        Returns:
            WalkerPlan with current configuration

        Example:
            >>> plan = (GraphWalker(manager)
            ...     .starting_from(task_id)
            ...     .bfs()
            ...     .max_depth(5)
            ...     .explain())
            >>> print(plan)
            Walker Plan
            ========================================
            Strategy: BFS
            Start: T-123
            Max depth: 5
            ...
        """
        # Count nodes for estimation
        node_map = self._build_node_map()

        return WalkerPlan(
            strategy=self._strategy.name,
            start_id=self._start_id,
            max_depth=self._max_depth,
            edge_types=self._edge_types,
            has_filter=self._filter_fn is not None,
            has_visitor=self._visitor is not None,
            reverse_direction=self._reverse_direction,
            estimated_nodes=len(node_map)
        )

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
