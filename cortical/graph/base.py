"""
BaseGraph: Abstract base class for all graph implementations.

This module provides the core graph abstraction that all Cortical
graph implementations can inherit from. It provides:

- CRUD operations for nodes and edges
- BFS/DFS traversal with visitor pattern
- Common graph algorithms (shortest path, cycles, components)
- Serialization support

Subclasses must implement two factory methods:
- _create_node(): Factory for domain-specific node types
- _create_edge(): Factory for domain-specific edge types

Example:
    class MyGraph(BaseGraph[MyNode, MyEdge]):
        def _create_node(self, id: str, **kwargs) -> MyNode:
            return MyNode(id=id, **kwargs)

        def _create_edge(self, source_id: str, target_id: str, **kwargs) -> MyEdge:
            return MyEdge(source_id=source_id, target_id=target_id, **kwargs)

    graph = MyGraph()
    graph.add_node("A", content="Node A")
    graph.add_node("B", content="Node B")
    graph.add_edge("A", "B", edge_type="connects")

See docs/base-graph-design.md for architecture details.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .protocols import NodeBase, EdgeBase
from .storage import GraphStorage, InMemoryGraphStorage

# Type variables for generic graph
N = TypeVar("N", bound=NodeBase)
E = TypeVar("E", bound=EdgeBase)
T = TypeVar("T")  # Accumulator type for visitors


class BaseGraph(Generic[N, E], ABC):
    """
    Abstract base class for all graph implementations.

    Provides the common interface and default implementations for:
    - Node CRUD operations
    - Edge CRUD operations
    - Traversal (BFS, DFS)
    - Path finding
    - Cycle detection
    - Connected components
    - Serialization

    Type Parameters:
        N: Node type (must satisfy NodeBase contract)
        E: Edge type (must satisfy EdgeBase contract)

    Attributes:
        _storage: The storage backend (injected via constructor)

    Subclass Requirements:
        Subclasses MUST implement:
        - _create_node(id, **kwargs) -> N
        - _create_edge(source_id, target_id, **kwargs) -> E

    Thread Safety:
        BaseGraph itself is not thread-safe. Thread safety depends on
        the storage backend used. For concurrent access, use a
        thread-safe storage backend or external synchronization.
    """

    def __init__(self, storage: Optional[GraphStorage[N, E]] = None) -> None:
        """
        Initialize graph with optional storage backend.

        Args:
            storage: Storage backend (defaults to InMemoryGraphStorage)
        """
        self._storage: GraphStorage[N, E] = storage or InMemoryGraphStorage()

    # =========================================================================
    # Abstract Factory Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _create_node(self, id: str, **kwargs: Any) -> N:
        """
        Factory method for creating domain-specific nodes.

        Subclasses override this to create their specific node types
        (ThoughtNode, GraphNode, Atom, etc.)

        Args:
            id: Unique node identifier
            **kwargs: Node-specific attributes

        Returns:
            A new node instance of type N
        """
        ...

    @abstractmethod
    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs: Any,
    ) -> E:
        """
        Factory method for creating domain-specific edges.

        Subclasses override this to create their specific edge types
        (ThoughtEdge, GraphEdge, SynapticEdge, etc.)

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            **kwargs: Edge-specific attributes

        Returns:
            A new edge instance of type E
        """
        ...

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(
        self,
        node_id: str,
        node_type: str = "",
        content: str = "",
        **kwargs: Any,
    ) -> N:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            node_type: Type/category of the node
            content: Primary content or description
            **kwargs: Additional node-specific attributes

        Returns:
            The created node

        Raises:
            ValueError: If node_id already exists
        """
        if self._storage.has_node(node_id):
            raise ValueError(f"Node '{node_id}' already exists")

        node = self._create_node(
            id=node_id,
            node_type=node_type,
            content=content,
            **kwargs,
        )
        self._storage.add_node(node)
        return node

    def get_node(self, node_id: str) -> Optional[N]:
        """
        Get a node by ID.

        Args:
            node_id: The node ID to look up

        Returns:
            The node if found, None otherwise
        """
        return self._storage.get_node(node_id)

    def remove_node(self, node_id: str) -> Optional[N]:
        """
        Remove a node and all its connected edges.

        Args:
            node_id: The node ID to remove

        Returns:
            The removed node if found, None otherwise
        """
        return self._storage.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists.

        Args:
            node_id: The node ID to check

        Returns:
            True if node exists, False otherwise
        """
        return self._storage.has_node(node_id)

    def get_or_create_node(
        self,
        node_id: str,
        node_type: str = "",
        content: str = "",
        **kwargs: Any,
    ) -> Tuple[N, bool]:
        """
        Get an existing node or create it if it doesn't exist.

        Args:
            node_id: Unique identifier for the node
            node_type: Type/category (used if creating)
            content: Content (used if creating)
            **kwargs: Additional attributes (used if creating)

        Returns:
            Tuple of (node, created) where created is True if node was created
        """
        existing = self._storage.get_node(node_id)
        if existing is not None:
            return existing, False

        node = self._create_node(
            id=node_id,
            node_type=node_type,
            content=content,
            **kwargs,
        )
        self._storage.add_node(node)
        return node, True

    @property
    def nodes(self) -> Iterator[N]:
        """Iterate over all nodes."""
        return self._storage.all_nodes()

    @property
    def node_count(self) -> int:
        """Return number of nodes."""
        return self._storage.node_count()

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        weight: float = 1.0,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> E:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            weight: Edge weight (0.0 to 1.0)
            bidirectional: Whether edge goes both ways
            **kwargs: Additional edge-specific attributes

        Returns:
            The created edge

        Raises:
            ValueError: If either node doesn't exist
        """
        if not self._storage.has_node(source_id):
            raise ValueError(f"Source node '{source_id}' not found")
        if not self._storage.has_node(target_id):
            raise ValueError(f"Target node '{target_id}' not found")

        edge = self._create_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            bidirectional=bidirectional,
            **kwargs,
        )
        self._storage.add_edge(edge)
        return edge

    def get_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
    ) -> Optional[E]:
        """
        Get a specific edge by source, target, and type.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type (empty string matches first edge)

        Returns:
            The edge if found, None otherwise
        """
        return self._storage.get_edge(source_id, target_id, edge_type)

    def remove_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
    ) -> bool:
        """
        Remove a specific edge.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type

        Returns:
            True if edge was found and removed, False otherwise
        """
        return self._storage.remove_edge(source_id, target_id, edge_type)

    def edges_from(self, node_id: str) -> List[E]:
        """
        Get all outgoing edges from a node.

        Args:
            node_id: The source node ID

        Returns:
            List of edges with source_id == node_id
        """
        return self._storage.edges_from(node_id)

    def edges_to(self, node_id: str) -> List[E]:
        """
        Get all incoming edges to a node.

        Args:
            node_id: The target node ID

        Returns:
            List of edges with target_id == node_id
        """
        return self._storage.edges_to(node_id)

    @property
    def edges(self) -> Iterator[E]:
        """Iterate over all edges."""
        return self._storage.all_edges()

    @property
    def edge_count(self) -> int:
        """Return number of edges."""
        return self._storage.edge_count()

    # =========================================================================
    # Neighbor Operations
    # =========================================================================

    def neighbors(
        self,
        node_id: str,
        direction: str = "out",
    ) -> List[str]:
        """
        Get neighbor node IDs.

        Args:
            node_id: Node to get neighbors for
            direction: "out" (outgoing), "in" (incoming), or "both"

        Returns:
            List of neighbor node IDs
        """
        neighbors: List[str] = []

        if direction in ("out", "both"):
            for edge in self._storage.edges_from(node_id):
                neighbors.append(edge.target_id)

        if direction in ("in", "both"):
            for edge in self._storage.edges_to(node_id):
                neighbors.append(edge.source_id)

        return neighbors

    def degree(self, node_id: str, direction: str = "both") -> int:
        """
        Get node degree (number of connections).

        Args:
            node_id: The node ID
            direction: "out", "in", or "both"

        Returns:
            Number of connections in the specified direction
        """
        return len(self.neighbors(node_id, direction))

    # =========================================================================
    # Traversal Operations
    # =========================================================================

    def bfs(
        self,
        start_id: str,
        visitor: Optional[Callable[[N, T], T]] = None,
        initial: Optional[T] = None,
        max_depth: Optional[int] = None,
        edge_filter: Optional[Callable[[E], bool]] = None,
        node_filter: Optional[Callable[[N], bool]] = None,
        direction: str = "out",
    ) -> Union[List[str], T]:
        """
        Breadth-first search traversal.

        Args:
            start_id: Starting node ID
            visitor: Optional function (node, accumulator) -> accumulator
            initial: Initial accumulator value
            max_depth: Maximum traversal depth (None = unlimited)
            edge_filter: Optional predicate to filter edges
            node_filter: Optional predicate to filter nodes
            direction: "out", "in", or "both"

        Returns:
            If visitor provided: Final accumulator value
            Otherwise: List of node IDs in BFS order

        Raises:
            ValueError: If start node doesn't exist
        """
        start_node = self._storage.get_node(start_id)
        if start_node is None:
            raise ValueError(f"Start node '{start_id}' not found")

        visited: Set[str] = {start_id}
        queue: deque[Tuple[N, int]] = deque([(start_node, 0)])
        result: List[str] = []
        acc = initial

        while queue:
            node, depth = queue.popleft()

            # Check depth limit
            if max_depth is not None and depth > max_depth:
                continue

            # Check node filter
            if node_filter is not None and not node_filter(node):
                continue

            # Visit node
            result.append(node.id)
            if visitor is not None:
                acc = visitor(node, acc)

            # Get edges based on direction
            edges: List[E] = []
            if direction in ("out", "both"):
                edges.extend(self._storage.edges_from(node.id))
            if direction in ("in", "both"):
                edges.extend(self._storage.edges_to(node.id))

            # Process edges
            for edge in edges:
                # Check edge filter
                if edge_filter is not None and not edge_filter(edge):
                    continue

                # Determine neighbor
                neighbor_id = (
                    edge.target_id if edge.source_id == node.id else edge.source_id
                )

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor = self._storage.get_node(neighbor_id)
                    if neighbor is not None:
                        queue.append((neighbor, depth + 1))

        return acc if visitor is not None else result

    def dfs(
        self,
        start_id: str,
        visitor: Optional[Callable[[N, T], T]] = None,
        initial: Optional[T] = None,
        max_depth: Optional[int] = None,
        edge_filter: Optional[Callable[[E], bool]] = None,
        node_filter: Optional[Callable[[N], bool]] = None,
        direction: str = "out",
    ) -> Union[List[str], T]:
        """
        Depth-first search traversal.

        Same parameters as bfs().

        Returns:
            If visitor provided: Final accumulator value
            Otherwise: List of node IDs in DFS order

        Raises:
            ValueError: If start node doesn't exist
        """
        start_node = self._storage.get_node(start_id)
        if start_node is None:
            raise ValueError(f"Start node '{start_id}' not found")

        visited: Set[str] = set()
        result: List[str] = []
        acc = initial

        def recurse(node: N, depth: int) -> None:
            nonlocal acc

            if node.id in visited:
                return
            if max_depth is not None and depth > max_depth:
                return
            if node_filter is not None and not node_filter(node):
                return

            visited.add(node.id)
            result.append(node.id)

            if visitor is not None:
                acc = visitor(node, acc)

            # Get edges based on direction
            edges: List[E] = []
            if direction in ("out", "both"):
                edges.extend(self._storage.edges_from(node.id))
            if direction in ("in", "both"):
                edges.extend(self._storage.edges_to(node.id))

            for edge in edges:
                if edge_filter is not None and not edge_filter(edge):
                    continue

                neighbor_id = (
                    edge.target_id if edge.source_id == node.id else edge.source_id
                )
                neighbor = self._storage.get_node(neighbor_id)
                if neighbor is not None:
                    recurse(neighbor, depth + 1)

        recurse(start_node, 0)
        return acc if visitor is not None else result

    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        direction: str = "out",
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.

        Args:
            from_id: Starting node ID
            to_id: Target node ID
            direction: "out", "in", or "both"

        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        if from_id == to_id:
            return [from_id]

        if not self.has_node(from_id) or not self.has_node(to_id):
            return None

        visited: Set[str] = {from_id}
        queue: deque[Tuple[str, List[str]]] = deque([(from_id, [from_id])])

        while queue:
            node_id, path = queue.popleft()

            for neighbor_id in self.neighbors(node_id, direction):
                if neighbor_id == to_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs_cycle(node_id: str, path: List[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor_id in self.neighbors(node_id, "out"):
                if neighbor_id not in visited:
                    dfs_cycle(neighbor_id, path)
                elif neighbor_id in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor_id)
                    cycles.append(path[cycle_start:] + [neighbor_id])

            path.pop()
            rec_stack.remove(node_id)

        for node in self.nodes:
            if node.id not in visited:
                dfs_cycle(node.id, [])

        return cycles

    def has_cycle(self) -> bool:
        """
        Check if graph contains any cycle.

        More efficient than find_cycles() when you only need to know
        if a cycle exists.

        Returns:
            True if graph contains a cycle, False otherwise
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle_from(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor_id in self.neighbors(node_id, "out"):
                if neighbor_id not in visited:
                    if has_cycle_from(neighbor_id):
                        return True
                elif neighbor_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node in self.nodes:
            if node.id not in visited:
                if has_cycle_from(node.id):
                    return True
        return False

    def connected_components(self) -> List[Set[str]]:
        """
        Find all connected components (treating graph as undirected).

        Returns:
            List of sets, where each set contains node IDs in a component
        """
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for node in self.nodes:
            if node.id not in visited:
                component: Set[str] = set()
                queue: deque[str] = deque([node.id])

                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue

                    visited.add(current)
                    component.add(current)

                    for neighbor_id in self.neighbors(current, "both"):
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)

                components.append(component)

        return components

    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order (Kahn's algorithm).

        For a DAG, returns an ordering where for every edge A->B,
        A appears before B.

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph contains a cycle
        """
        # Compute in-degrees
        in_degree: Dict[str, int] = {node.id: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1

        # Initialize heap with zero in-degree nodes (deterministic ordering)
        heap = [node_id for node_id, degree in in_degree.items() if degree == 0]
        heapq.heapify(heap)

        result: List[str] = []

        while heap:
            current = heapq.heappop(heap)
            result.append(current)

            for neighbor_id in self.neighbors(current, "out"):
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    heapq.heappush(heap, neighbor_id)

        if len(result) != self.node_count:
            raise ValueError("Graph contains a cycle")

        return result

    def find_roots(self) -> Set[str]:
        """
        Find nodes with no incoming edges.

        Returns:
            Set of node IDs with in-degree = 0
        """
        return {
            node.id
            for node in self.nodes
            if len(self._storage.edges_to(node.id)) == 0
        }

    def find_leaves(self) -> Set[str]:
        """
        Find nodes with no outgoing edges.

        Returns:
            Set of node IDs with out-degree = 0
        """
        return {
            node.id
            for node in self.nodes
            if len(self._storage.edges_from(node.id)) == 0
        }

    def find_hubs(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Find nodes with highest degree.

        Args:
            top_n: Number of top hubs to return

        Returns:
            List of (node_id, degree) tuples sorted by degree descending
        """
        degrees = [(node.id, self.degree(node.id)) for node in self.nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_n]

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to dictionary.

        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        return {
            "nodes": [self._node_to_dict(n) for n in self.nodes],
            "edges": [self._edge_to_dict(e) for e in self.edges],
        }

    def _node_to_dict(self, node: N) -> Dict[str, Any]:
        """
        Serialize a node to dictionary.

        Override for custom node serialization.
        """
        if hasattr(node, "to_dict"):
            return node.to_dict()
        return {
            "id": node.id,
            "node_type": getattr(node, "node_type", ""),
            "content": getattr(node, "content", ""),
            "properties": getattr(node, "properties", {}),
        }

    def _edge_to_dict(self, edge: E) -> Dict[str, Any]:
        """
        Serialize an edge to dictionary.

        Override for custom edge serialization.
        """
        if hasattr(edge, "to_dict"):
            return edge.to_dict()
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        **kwargs: Any,
    ) -> "BaseGraph[N, E]":
        """
        Deserialize graph from dictionary.

        Args:
            data: Dictionary with 'nodes' and 'edges' lists
            **kwargs: Additional arguments passed to constructor

        Returns:
            New graph instance
        """
        graph = cls(**kwargs)

        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            graph.add_node(node_id, **node_data)

        for edge_data in data.get("edges", []):
            graph.add_edge(**edge_data)

        return graph

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._storage.clear()

    def copy(self) -> "BaseGraph[N, E]":
        """
        Create a shallow copy of the graph.

        Returns:
            New graph with same nodes and edges
        """
        new_graph = type(self)()
        for node in self.nodes:
            new_graph._storage.add_node(node)
        for edge in self.edges:
            new_graph._storage.add_edge(edge)
        return new_graph

    def subgraph(self, node_ids: Set[str]) -> "BaseGraph[N, E]":
        """
        Create a subgraph containing only specified nodes.

        Args:
            node_ids: Set of node IDs to include

        Returns:
            New graph containing only specified nodes and edges between them
        """
        new_graph = type(self)()

        for node in self.nodes:
            if node.id in node_ids:
                new_graph._storage.add_node(node)

        for edge in self.edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                new_graph._storage.add_edge(edge)

        return new_graph

    def __len__(self) -> int:
        """Return number of nodes."""
        return self.node_count

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists."""
        return self.has_node(node_id)

    def __iter__(self) -> Iterator[N]:
        """Iterate over nodes."""
        return self.nodes

    def __repr__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}(nodes={self.node_count}, edges={self.edge_count})"
