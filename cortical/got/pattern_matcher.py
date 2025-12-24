"""
Pattern Matcher for Graph of Thought.

Implements subgraph pattern matching for finding structural patterns:
- Chain patterns (A -> B -> C)
- Star patterns (hub connected to many nodes)
- Custom constraint matching

Example:
    # Find dependency chains
    pattern = (
        Pattern()
        .node("a", type="task", status="pending")
        .edge("DEPENDS_ON", direction="incoming")
        .node("b", type="task")
        .edge("DEPENDS_ON", direction="incoming")
        .node("c", type="task")
    )

    matches = PatternMatcher(manager).find(pattern)
    for match in matches:
        print(f"Chain: {match['a'].id} <- {match['b'].id} <- {match['c'].id}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
)

from .api import GoTManager


@dataclass
class NodeConstraint:
    """Constraint for a pattern node."""
    name: str
    node_type: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeConstraint:
    """Constraint for a pattern edge."""
    edge_type: str
    direction: str = "outgoing"  # incoming, outgoing, any


class Pattern:
    """
    Fluent pattern builder for subgraph matching.

    Build patterns by chaining node() and edge() calls:

        Pattern()
        .node("a", type="task")
        .edge("DEPENDS_ON")
        .node("b", type="task")
    """

    def __init__(self):
        """Initialize empty pattern."""
        self._nodes: List[NodeConstraint] = []
        self._edges: List[EdgeConstraint] = []
        self._last_was_node: bool = False

    def node(
        self,
        name: str,
        type: Optional[str] = None,
        **constraints
    ) -> "Pattern":
        """
        Add a node to the pattern.

        Args:
            name: Variable name for the node
            type: Required node type (task, sprint, decision)
            **constraints: Field constraints (status="pending", priority="high")

        Returns:
            Self for chaining
        """
        self._nodes.append(NodeConstraint(
            name=name,
            node_type=type,
            constraints=constraints
        ))
        self._last_was_node = True
        return self

    def edge(
        self,
        edge_type: str,
        direction: str = "outgoing"
    ) -> "Pattern":
        """
        Add an edge constraint between nodes.

        Args:
            edge_type: Type of edge (DEPENDS_ON, CONTAINS, etc.)
            direction: "incoming", "outgoing", or "any"

        Returns:
            Self for chaining
        """
        if not self._last_was_node:
            raise ValueError("Edge must follow a node")

        self._edges.append(EdgeConstraint(
            edge_type=edge_type,
            direction=direction
        ))
        self._last_was_node = False
        return self

    @property
    def node_count(self) -> int:
        """Number of nodes in pattern."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in pattern."""
        return len(self._edges)

    def get_nodes(self) -> List[NodeConstraint]:
        """Get node constraints."""
        return self._nodes.copy()

    def get_edges(self) -> List[EdgeConstraint]:
        """Get edge constraints."""
        return self._edges.copy()


@dataclass
class PatternMatch:
    """A single pattern match result."""
    bindings: Dict[str, Any]  # name -> node

    def __getitem__(self, key: str) -> Any:
        return self.bindings[key]

    def keys(self):
        return self.bindings.keys()

    def values(self):
        return self.bindings.values()

    def items(self):
        return self.bindings.items()


class PatternMatcher:
    """
    Find subgraph patterns in GoT graph.

    Uses backtracking search with constraint propagation
    for efficient pattern matching.
    """

    def __init__(self, manager: GoTManager):
        """Initialize matcher with GoT manager."""
        self._manager = manager
        self._max_results: Optional[int] = None

    def limit(self, n: int) -> "PatternMatcher":
        """
        Limit number of matches to find.

        Args:
            n: Maximum number of matches

        Returns:
            Self for chaining
        """
        self._max_results = n
        return self

    def find(self, pattern: Pattern) -> List[PatternMatch]:
        """
        Find all matches for the pattern.

        Args:
            pattern: Pattern to match

        Returns:
            List of matches, each mapping node names to actual nodes
        """
        if pattern.node_count == 0:
            return []

        # Build graph data structures
        nodes = self._get_all_nodes()
        edges = self._build_edge_index()

        # Find matches using backtracking
        matches = []
        self._backtrack(
            pattern=pattern,
            node_index=0,
            bindings={},
            nodes=nodes,
            edges=edges,
            matches=matches
        )

        return matches

    def find_first(self, pattern: Pattern) -> Optional[PatternMatch]:
        """
        Find first match for the pattern.

        Args:
            pattern: Pattern to match

        Returns:
            First match or None
        """
        self._max_results = 1
        matches = self.find(pattern)
        return matches[0] if matches else None

    def count(self, pattern: Pattern) -> int:
        """
        Count matches for the pattern.

        Args:
            pattern: Pattern to match

        Returns:
            Number of matches
        """
        return len(self.find(pattern))

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _get_all_nodes(self) -> Dict[str, Any]:
        """Get all nodes indexed by ID."""
        nodes = {}

        for task in self._manager.list_all_tasks():
            nodes[task.id] = task

        for sprint in self._manager.list_sprints():
            nodes[sprint.id] = sprint

        for decision in self._manager.list_decisions():
            nodes[decision.id] = decision

        return nodes

    def _build_edge_index(self) -> Dict[Tuple[str, str, str], bool]:
        """
        Build edge index for fast lookup.

        Returns dict mapping (source_id, edge_type, target_id) -> True
        """
        index = {}

        for edge in self._manager.list_edges():
            # Index both directions
            index[(edge.source_id, edge.edge_type, edge.target_id)] = True
            # For "any" direction queries
            index[(edge.target_id, f"~{edge.edge_type}", edge.source_id)] = True

        return index

    def _get_node_type(self, node: Any) -> str:
        """Get type string for a node."""
        # Check by ID prefix
        if hasattr(node, 'id'):
            if node.id.startswith('T-'):
                return 'task'
            elif node.id.startswith('S-'):
                return 'sprint'
            elif node.id.startswith('D-'):
                return 'decision'
            elif node.id.startswith('H-'):
                return 'handoff'
        return 'unknown'

    def _matches_constraint(
        self,
        node: Any,
        constraint: NodeConstraint
    ) -> bool:
        """Check if node matches constraint."""
        # Check node type
        if constraint.node_type:
            node_type = self._get_node_type(node)
            if node_type != constraint.node_type:
                return False

        # Check field constraints
        for field, expected in constraint.constraints.items():
            # Try direct attribute
            value = getattr(node, field, None)
            # Try properties dict
            if value is None and hasattr(node, 'properties'):
                value = node.properties.get(field)

            if value != expected:
                return False

        return True

    def _edge_exists(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        direction: str,
        edges: Dict[Tuple[str, str, str], bool]
    ) -> bool:
        """Check if edge exists matching direction constraint."""
        if direction == "outgoing":
            # from_id -> to_id
            return (from_id, edge_type, to_id) in edges
        elif direction == "incoming":
            # to_id -> from_id (edge points to from_id)
            return (to_id, edge_type, from_id) in edges
        else:  # any
            return (
                (from_id, edge_type, to_id) in edges or
                (to_id, edge_type, from_id) in edges
            )

    def _backtrack(
        self,
        pattern: Pattern,
        node_index: int,
        bindings: Dict[str, Any],
        nodes: Dict[str, Any],
        edges: Dict[Tuple[str, str, str], bool],
        matches: List[PatternMatch]
    ) -> None:
        """Backtracking search for pattern matches."""
        # Check result limit
        if self._max_results and len(matches) >= self._max_results:
            return

        # All nodes matched - found a complete match
        if node_index >= pattern.node_count:
            matches.append(PatternMatch(bindings=bindings.copy()))
            return

        # Get current node constraint
        node_constraint = pattern.get_nodes()[node_index]

        # Get edge constraint to previous node (if any)
        edge_constraint = None
        if node_index > 0:
            edge_constraint = pattern.get_edges()[node_index - 1]
            prev_constraint = pattern.get_nodes()[node_index - 1]
            prev_node = bindings[prev_constraint.name]

        # Try each candidate node
        for node_id, node in nodes.items():
            # Skip if already bound
            if node_id in {n.id for n in bindings.values() if hasattr(n, 'id')}:
                continue

            # Check node constraint
            if not self._matches_constraint(node, node_constraint):
                continue

            # Check edge constraint to previous node
            if edge_constraint:
                if not self._edge_exists(
                    prev_node.id,
                    node_id,
                    edge_constraint.edge_type,
                    edge_constraint.direction,
                    edges
                ):
                    continue

            # Bind and recurse
            bindings[node_constraint.name] = node
            self._backtrack(
                pattern, node_index + 1, bindings,
                nodes, edges, matches
            )
            del bindings[node_constraint.name]
