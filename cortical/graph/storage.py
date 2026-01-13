"""
Graph Storage Backends: Pluggable persistence for BaseGraph.

This module provides the storage layer that BaseGraph uses for
node and edge persistence. Different backends offer different
performance/durability tradeoffs.

Backends:
- InMemoryGraphStorage: Fast, no persistence (default)
- FileGraphStorage: JSON file-based, git-friendly (planned)
- TransactionalGraphStorage: WAL-based, ACID compliant (planned)

Design Philosophy:
    Storage is injected into BaseGraph via dependency injection,
    allowing tests to use fast in-memory storage while production
    uses durable file-based storage.

Example:
    # Default in-memory storage
    graph = MyGraph()

    # Custom storage
    storage = InMemoryGraphStorage()
    graph = MyGraph(storage=storage)

See docs/base-graph-design.md for architecture details.
"""

from __future__ import annotations

from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .protocols import NodeBase, EdgeBase

# Type variables for generic storage
N = TypeVar("N", bound=NodeBase)
E = TypeVar("E", bound=EdgeBase)


@runtime_checkable
class GraphStorage(Protocol[N, E]):
    """
    Protocol defining the storage backend interface.

    Any class implementing these methods can be used as a BaseGraph
    storage backend. This enables swapping storage implementations
    without changing graph code.

    Thread Safety:
        Implementations should document their thread-safety guarantees.
        InMemoryGraphStorage is NOT thread-safe by default.
    """

    # Node operations
    def add_node(self, node: N) -> None:
        """Add a node to storage. Overwrites if exists."""
        ...

    def get_node(self, node_id: str) -> Optional[N]:
        """Get a node by ID, or None if not found."""
        ...

    def remove_node(self, node_id: str) -> Optional[N]:
        """Remove a node and return it, or None if not found."""
        ...

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        ...

    def all_nodes(self) -> Iterator[N]:
        """Iterate over all nodes."""
        ...

    def node_count(self) -> int:
        """Return total number of nodes."""
        ...

    # Edge operations
    def add_edge(self, edge: E) -> None:
        """Add an edge to storage. Handles indexing."""
        ...

    def get_edge(
        self, source_id: str, target_id: str, edge_type: str
    ) -> Optional[E]:
        """Get a specific edge, or None if not found."""
        ...

    def remove_edge(
        self, source_id: str, target_id: str, edge_type: str
    ) -> bool:
        """Remove an edge. Returns True if removed, False if not found."""
        ...

    def edges_from(self, node_id: str) -> List[E]:
        """Get all outgoing edges from a node."""
        ...

    def edges_to(self, node_id: str) -> List[E]:
        """Get all incoming edges to a node."""
        ...

    def all_edges(self) -> Iterator[E]:
        """Iterate over all edges."""
        ...

    def edge_count(self) -> int:
        """Return total number of edges."""
        ...

    # Bulk operations
    def clear(self) -> None:
        """Remove all nodes and edges."""
        ...


class InMemoryGraphStorage(Generic[N, E]):
    """
    High-performance in-memory storage with O(1) lookups.

    This is the default storage backend for BaseGraph. It maintains
    three indexes for fast access:
    - _nodes: Dict[str, N] for O(1) node lookup by ID
    - _edges_by_source: Dict[str, List[E]] for O(1) outgoing edges
    - _edges_by_target: Dict[str, List[E]] for O(1) incoming edges

    Performance Characteristics:
    - add_node: O(1)
    - get_node: O(1)
    - remove_node: O(E) where E = edges connected to node
    - add_edge: O(1) amortized
    - get_edge: O(k) where k = edges from source
    - edges_from/to: O(1)

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        wrap operations in a lock or use a thread-safe storage backend.

    Memory Usage:
        ~3x edge count due to three index structures.
        For very large graphs, consider a database-backed storage.

    Example:
        storage = InMemoryGraphStorage()
        storage.add_node(NodeBase(id="A", content="Node A"))
        storage.add_edge(EdgeBase(source_id="A", target_id="B"))
        print(storage.node_count())  # 1 (B not added as node)
    """

    def __init__(self) -> None:
        """Initialize empty storage with indexes."""
        self._nodes: Dict[str, N] = {}
        self._edges: List[E] = []
        self._edges_by_source: Dict[str, List[E]] = {}
        self._edges_by_target: Dict[str, List[E]] = {}

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, node: N) -> None:
        """
        Add a node to storage.

        If a node with the same ID already exists, it is overwritten.

        Args:
            node: The node to add

        Time Complexity: O(1)
        """
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[N]:
        """
        Get a node by ID.

        Args:
            node_id: The node ID to look up

        Returns:
            The node if found, None otherwise

        Time Complexity: O(1)
        """
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> Optional[N]:
        """
        Remove a node and all connected edges.

        Args:
            node_id: The node ID to remove

        Returns:
            The removed node if found, None otherwise

        Time Complexity: O(E) where E = number of connected edges
        """
        node = self._nodes.pop(node_id, None)
        if node is None:
            return None

        # Remove edges connected to this node
        edges_to_remove = [
            e for e in self._edges
            if e.source_id == node_id or e.target_id == node_id
        ]

        for edge in edges_to_remove:
            self._edges.remove(edge)

        # Clean up source index
        if node_id in self._edges_by_source:
            del self._edges_by_source[node_id]

        # Clean up target index
        if node_id in self._edges_by_target:
            del self._edges_by_target[node_id]

        # Remove node from other nodes' edge lists
        for source_id, edges in list(self._edges_by_source.items()):
            self._edges_by_source[source_id] = [
                e for e in edges if e.target_id != node_id
            ]

        for target_id, edges in list(self._edges_by_target.items()):
            self._edges_by_target[target_id] = [
                e for e in edges if e.source_id != node_id
            ]

        return node

    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists.

        Args:
            node_id: The node ID to check

        Returns:
            True if node exists, False otherwise

        Time Complexity: O(1)
        """
        return node_id in self._nodes

    def all_nodes(self) -> Iterator[N]:
        """
        Iterate over all nodes.

        Yields:
            All nodes in storage

        Time Complexity: O(N) total for full iteration
        """
        return iter(self._nodes.values())

    def node_count(self) -> int:
        """
        Return total number of nodes.

        Returns:
            Number of nodes in storage

        Time Complexity: O(1)
        """
        return len(self._nodes)

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(self, edge: E) -> None:
        """
        Add an edge to storage.

        The edge is indexed by both source and target for fast lookups.
        If bidirectional=True, a reverse edge is also added.

        Args:
            edge: The edge to add

        Time Complexity: O(1) amortized
        """
        self._edges.append(edge)

        # Index by source
        if edge.source_id not in self._edges_by_source:
            self._edges_by_source[edge.source_id] = []
        self._edges_by_source[edge.source_id].append(edge)

        # Index by target
        if edge.target_id not in self._edges_by_target:
            self._edges_by_target[edge.target_id] = []
        self._edges_by_target[edge.target_id].append(edge)

        # Handle bidirectional edges by adding reverse
        if getattr(edge, 'bidirectional', False):
            # Create reverse edge (need to handle different edge types)
            reverse_data = {
                'source_id': edge.target_id,
                'target_id': edge.source_id,
                'edge_type': edge.edge_type,
                'weight': edge.weight,
                'bidirectional': False,  # Prevent infinite recursion
            }

            # Copy properties if present
            if hasattr(edge, 'properties'):
                reverse_data['properties'] = edge.properties

            # Create reverse edge of same type
            reverse = type(edge)(**reverse_data)

            self._edges.append(reverse)

            # Index reverse edge
            if reverse.source_id not in self._edges_by_source:
                self._edges_by_source[reverse.source_id] = []
            self._edges_by_source[reverse.source_id].append(reverse)

            if reverse.target_id not in self._edges_by_target:
                self._edges_by_target[reverse.target_id] = []
            self._edges_by_target[reverse.target_id].append(reverse)

    def get_edge(
        self, source_id: str, target_id: str, edge_type: str
    ) -> Optional[E]:
        """
        Get a specific edge by source, target, and type.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type string

        Returns:
            The edge if found, None otherwise

        Time Complexity: O(k) where k = edges from source
        """
        for edge in self._edges_by_source.get(source_id, []):
            if edge.target_id == target_id and edge.edge_type == edge_type:
                return edge
        return None

    def remove_edge(
        self, source_id: str, target_id: str, edge_type: str
    ) -> bool:
        """
        Remove a specific edge.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type string

        Returns:
            True if edge was found and removed, False otherwise

        Time Complexity: O(E) in worst case
        """
        edge = self.get_edge(source_id, target_id, edge_type)
        if edge is None:
            return False

        # Remove from main list
        self._edges.remove(edge)

        # Remove from source index
        if source_id in self._edges_by_source:
            self._edges_by_source[source_id] = [
                e for e in self._edges_by_source[source_id]
                if not (e.target_id == target_id and e.edge_type == edge_type)
            ]

        # Remove from target index
        if target_id in self._edges_by_target:
            self._edges_by_target[target_id] = [
                e for e in self._edges_by_target[target_id]
                if not (e.source_id == source_id and e.edge_type == edge_type)
            ]

        return True

    def edges_from(self, node_id: str) -> List[E]:
        """
        Get all outgoing edges from a node.

        Args:
            node_id: The source node ID

        Returns:
            List of edges with source_id == node_id

        Time Complexity: O(1) for lookup, O(k) for copy where k = edge count
        """
        return list(self._edges_by_source.get(node_id, []))

    def edges_to(self, node_id: str) -> List[E]:
        """
        Get all incoming edges to a node.

        Args:
            node_id: The target node ID

        Returns:
            List of edges with target_id == node_id

        Time Complexity: O(1) for lookup, O(k) for copy where k = edge count
        """
        return list(self._edges_by_target.get(node_id, []))

    def all_edges(self) -> Iterator[E]:
        """
        Iterate over all edges.

        Yields:
            All edges in storage

        Time Complexity: O(E) total for full iteration
        """
        return iter(self._edges)

    def edge_count(self) -> int:
        """
        Return total number of edges.

        Returns:
            Number of edges in storage

        Time Complexity: O(1)
        """
        return len(self._edges)

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def clear(self) -> None:
        """
        Remove all nodes and edges.

        Time Complexity: O(1)
        """
        self._nodes.clear()
        self._edges.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()

    # =========================================================================
    # Additional Utilities
    # =========================================================================

    def edges_between(self, node_id_a: str, node_id_b: str) -> List[E]:
        """
        Get all edges between two nodes (in either direction).

        Args:
            node_id_a: First node ID
            node_id_b: Second node ID

        Returns:
            List of edges connecting the two nodes
        """
        result = []

        # A -> B edges
        for edge in self._edges_by_source.get(node_id_a, []):
            if edge.target_id == node_id_b:
                result.append(edge)

        # B -> A edges
        for edge in self._edges_by_source.get(node_id_b, []):
            if edge.target_id == node_id_a:
                result.append(edge)

        return result

    def edges_of_type(self, edge_type: str) -> List[E]:
        """
        Get all edges of a specific type.

        Args:
            edge_type: The edge type to filter by

        Returns:
            List of edges with matching edge_type

        Time Complexity: O(E)
        """
        return [e for e in self._edges if e.edge_type == edge_type]

    def nodes_of_type(self, node_type: str) -> List[N]:
        """
        Get all nodes of a specific type.

        Args:
            node_type: The node type to filter by

        Returns:
            List of nodes with matching node_type

        Time Complexity: O(N)
        """
        return [
            n for n in self._nodes.values()
            if getattr(n, 'node_type', '') == node_type
        ]
