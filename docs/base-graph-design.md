# Base Graph Class Design

**Status**: Implemented (PR #283 merged 2026-01-13)
**Author**: Claude
**Date**: 2026-01-13
**Task**: Unify all graph implementations under a common base

---

## Executive Summary

This document describes the unified `BaseGraph` architecture that consolidates graph implementations in the Cortical codebase. The design provides:

1. **Protocol-based node/edge abstractions** - Type-safe, extensible contracts
2. **Pluggable storage backends** - In-memory, file-based, or database
3. **Composable algorithm mixins** - DFS, BFS, PageRank, etc.
4. **Full backward compatibility** - Existing graphs can adopt incrementally

---

## Current State: Graph Implementations Inventory

| Graph Class | Location | Node Type | Edge Type | Storage | Special Features |
|-------------|----------|-----------|-----------|---------|------------------|
| `ThoughtGraph` | reasoning/thought_graph.py | ThoughtNode | ThoughtEdge | Dict | Clusters, visualization |
| `SemanticKnowledgeGraph` | graph/knowledge_graph.py | GraphNode | GraphEdge | Dict + WAL | PageRank, BM25, layers |
| `CognitiveGraph` | cognitive/graph.py | Atom | Atom (links) | Pluggable | Hypergraph, attention |
| `TaskDAG` | audits/algorithms/dag.py | str | Set[str] | DAGGraph | ✅ Migrated 2026-01-13 |
| `SynapticMemoryGraph` | reasoning/prism_got.py | ThoughtNode | SynapticEdge | Dict | Hebbian learning |
| `CausalGraph` | reasoning/prism_causal.py | str | CausalEdge | Dict | Evidence tracking |
| `PLNGraph` | reasoning/prism_pln.py | HiveNode | HiveEdge | Dict | Probabilistic logic |
| `TransitionGraph` | reasoning/prism_slm.py | str | Dict | Dict | Markov transitions |
| `Minicolumn` hierarchy | minicolumn.py, layers.py | Minicolumn | Edge | Dict | Cortical layers |
| `GraphWalker` | got/graph_walker.py | Any | Edge | External | Visitor pattern |
| CDG entities | cdg/types.py | Entity | Edge | Versioned | ACID transactions |

### Common Patterns Identified

All implementations share these core concepts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SHARED GRAPH CONCEPTS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  NODES                           EDGES                                   │
│  ─────                           ─────                                   │
│  • Unique ID (str)               • Source ID                            │
│  • Type/category                 • Target ID                            │
│  • Content/payload               • Type/relationship                    │
│  • Properties (dict)             • Weight (float)                       │
│  • Metadata (timestamps)         • Confidence (float)                   │
│                                  • Bidirectional flag                   │
│                                                                          │
│  STORAGE                         TRAVERSAL                               │
│  ───────                         ─────────                               │
│  • Node lookup by ID             • BFS                                  │
│  • Edge index by source          • DFS                                  │
│  • Edge index by target          • Shortest path                        │
│  • Adjacency lists               • Cycle detection                      │
│                                                                          │
│  ALGORITHMS                      PERSISTENCE                             │
│  ──────────                      ───────────                             │
│  • PageRank                      • JSON serialization                   │
│  • Connected components          • to_dict() / from_dict()              │
│  • Topological sort              • Checksum verification                │
│  • Bridge/articulation points    • WAL support (optional)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture

### Layer 1: Protocols (Contracts)

```python
# cortical/graph/protocols.py
from __future__ import annotations
from typing import Protocol, TypeVar, Generic, Dict, List, Set, Optional, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime

NID = TypeVar('NID', bound=str)  # Node ID type
EID = TypeVar('EID', bound=str)  # Edge ID type


@dataclass
class NodeBase:
    """
    Minimal node contract that all node types must satisfy.

    Subclasses can add domain-specific fields (activation, pagerank, etc.)
    while maintaining compatibility with BaseGraph operations.
    """
    id: str
    node_type: str = ""
    content: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeBase):
            return NotImplemented
        return self.id == other.id


@dataclass
class EdgeBase:
    """
    Minimal edge contract that all edge types must satisfy.

    Subclasses can add domain-specific fields (confidence, temporal_decay, etc.)
    while maintaining compatibility with BaseGraph operations.
    """
    source_id: str
    target_id: str
    edge_type: str = ""
    weight: float = 1.0
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def id(self) -> str:
        """Generate deterministic edge ID from components."""
        return f"E-{self.source_id}-{self.target_id}-{self.edge_type}"

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeBase):
            return NotImplemented
        return (self.source_id == other.source_id and
                self.target_id == other.target_id and
                self.edge_type == other.edge_type)


class NodeProtocol(Protocol):
    """Protocol for node-like objects."""
    @property
    def id(self) -> str: ...
    @property
    def node_type(self) -> str: ...


class EdgeProtocol(Protocol):
    """Protocol for edge-like objects."""
    @property
    def source_id(self) -> str: ...
    @property
    def target_id(self) -> str: ...
    @property
    def edge_type(self) -> str: ...
    @property
    def weight(self) -> float: ...
```

### Layer 2: Storage Backends

```python
# cortical/graph/storage.py
from typing import Protocol, TypeVar, Generic, Dict, List, Set, Optional, Iterator
from abc import ABC, abstractmethod

N = TypeVar('N', bound=NodeBase)
E = TypeVar('E', bound=EdgeBase)


class GraphStorage(Protocol[N, E]):
    """
    Protocol for graph storage backends.

    Implementations:
    - InMemoryGraphStorage: Dict-based, fastest, no persistence
    - FileGraphStorage: JSON file-based, git-friendly
    - TransactionalGraphStorage: WAL-based, ACID compliant
    """

    # Node operations
    def add_node(self, node: N) -> None: ...
    def get_node(self, node_id: str) -> Optional[N]: ...
    def remove_node(self, node_id: str) -> Optional[N]: ...
    def has_node(self, node_id: str) -> bool: ...
    def all_nodes(self) -> Iterator[N]: ...
    def node_count(self) -> int: ...

    # Edge operations
    def add_edge(self, edge: E) -> None: ...
    def get_edge(self, source_id: str, target_id: str, edge_type: str) -> Optional[E]: ...
    def remove_edge(self, source_id: str, target_id: str, edge_type: str) -> bool: ...
    def edges_from(self, node_id: str) -> List[E]: ...
    def edges_to(self, node_id: str) -> List[E]: ...
    def all_edges(self) -> Iterator[E]: ...
    def edge_count(self) -> int: ...

    # Bulk operations
    def clear(self) -> None: ...


class InMemoryGraphStorage(Generic[N, E]):
    """
    High-performance in-memory storage with O(1) lookups.

    Uses three indexes for fast access:
    - nodes: Dict[str, N] for node lookup
    - edges_by_source: Dict[str, List[E]] for outgoing edges
    - edges_by_target: Dict[str, List[E]] for incoming edges
    """

    def __init__(self):
        self._nodes: Dict[str, N] = {}
        self._edges: List[E] = []
        self._edges_by_source: Dict[str, List[E]] = {}
        self._edges_by_target: Dict[str, List[E]] = {}

    def add_node(self, node: N) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[N]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> Optional[N]:
        node = self._nodes.pop(node_id, None)
        if node:
            # Remove associated edges
            self._edges = [e for e in self._edges
                         if e.source_id != node_id and e.target_id != node_id]
            self._edges_by_source.pop(node_id, None)
            self._edges_by_target.pop(node_id, None)
            # Clean up references in other nodes' edge lists
            for edges in self._edges_by_source.values():
                edges[:] = [e for e in edges if e.target_id != node_id]
            for edges in self._edges_by_target.values():
                edges[:] = [e for e in edges if e.source_id != node_id]
        return node

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def all_nodes(self) -> Iterator[N]:
        return iter(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    def add_edge(self, edge: E) -> None:
        self._edges.append(edge)

        # Index by source
        if edge.source_id not in self._edges_by_source:
            self._edges_by_source[edge.source_id] = []
        self._edges_by_source[edge.source_id].append(edge)

        # Index by target
        if edge.target_id not in self._edges_by_target:
            self._edges_by_target[edge.target_id] = []
        self._edges_by_target[edge.target_id].append(edge)

        # Handle bidirectional edges
        if edge.bidirectional:
            reverse = type(edge)(
                source_id=edge.target_id,
                target_id=edge.source_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                bidirectional=False,  # Prevent infinite recursion
            )
            self._edges.append(reverse)
            if edge.target_id not in self._edges_by_source:
                self._edges_by_source[edge.target_id] = []
            self._edges_by_source[edge.target_id].append(reverse)
            if edge.source_id not in self._edges_by_target:
                self._edges_by_target[edge.source_id] = []
            self._edges_by_target[edge.source_id].append(reverse)

    def get_edge(self, source_id: str, target_id: str, edge_type: str) -> Optional[E]:
        for edge in self._edges_by_source.get(source_id, []):
            if edge.target_id == target_id and edge.edge_type == edge_type:
                return edge
        return None

    def remove_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        edge = self.get_edge(source_id, target_id, edge_type)
        if edge:
            self._edges.remove(edge)
            self._edges_by_source[source_id].remove(edge)
            self._edges_by_target[target_id].remove(edge)
            return True
        return False

    def edges_from(self, node_id: str) -> List[E]:
        return list(self._edges_by_source.get(node_id, []))

    def edges_to(self, node_id: str) -> List[E]:
        return list(self._edges_by_target.get(node_id, []))

    def all_edges(self) -> Iterator[E]:
        return iter(self._edges)

    def edge_count(self) -> int:
        return len(self._edges)

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()
```

### Layer 3: BaseGraph Abstract Class

```python
# cortical/graph/base.py
from __future__ import annotations
from typing import (
    TypeVar, Generic, Dict, List, Set, Optional, Any,
    Iterator, Callable, Tuple, Union
)
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from .protocols import NodeBase, EdgeBase, NodeProtocol, EdgeProtocol
from .storage import GraphStorage, InMemoryGraphStorage

N = TypeVar('N', bound=NodeBase)
E = TypeVar('E', bound=EdgeBase)
T = TypeVar('T')  # Accumulator type for visitors


class BaseGraph(Generic[N, E], ABC):
    """
    Abstract base class for all graph implementations.

    Provides:
    - CRUD operations for nodes and edges
    - BFS/DFS traversal with visitor pattern
    - Common graph algorithms (shortest path, connected components)
    - Serialization support

    Subclasses must implement:
    - _create_node(): Factory for domain-specific node types
    - _create_edge(): Factory for domain-specific edge types

    Example:
        class MyGraph(BaseGraph[MyNode, MyEdge]):
            def _create_node(self, id, **kwargs) -> MyNode:
                return MyNode(id=id, **kwargs)

            def _create_edge(self, source_id, target_id, **kwargs) -> MyEdge:
                return MyEdge(source_id=source_id, target_id=target_id, **kwargs)
    """

    def __init__(self, storage: Optional[GraphStorage[N, E]] = None):
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
    def _create_node(self, id: str, **kwargs) -> N:
        """
        Factory method for creating domain-specific nodes.

        Subclasses override this to create their specific node types
        (ThoughtNode, GraphNode, Atom, etc.)
        """
        ...

    @abstractmethod
    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs
    ) -> E:
        """
        Factory method for creating domain-specific edges.

        Subclasses override this to create their specific edge types
        (ThoughtEdge, GraphEdge, SynapticEdge, etc.)
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
        **kwargs
    ) -> N:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier
            node_type: Type/category of node
            content: Node content/payload
            **kwargs: Additional node-specific attributes

        Returns:
            The created node

        Raises:
            ValueError: If node_id already exists
        """
        if self._storage.has_node(node_id):
            raise ValueError(f"Node {node_id} already exists")

        node = self._create_node(
            id=node_id,
            node_type=node_type,
            content=content,
            **kwargs
        )
        self._storage.add_node(node)
        return node

    def get_node(self, node_id: str) -> Optional[N]:
        """Get a node by ID, or None if not found."""
        return self._storage.get_node(node_id)

    def remove_node(self, node_id: str) -> Optional[N]:
        """
        Remove a node and all its edges.

        Returns:
            The removed node, or None if not found
        """
        return self._storage.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return self._storage.has_node(node_id)

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
        **kwargs
    ) -> E:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            weight: Edge weight (0.0-1.0)
            bidirectional: Whether edge goes both ways
            **kwargs: Additional edge-specific attributes

        Returns:
            The created edge

        Raises:
            ValueError: If either node doesn't exist
        """
        if not self._storage.has_node(source_id):
            raise ValueError(f"Source node {source_id} not found")
        if not self._storage.has_node(target_id):
            raise ValueError(f"Target node {target_id} not found")

        edge = self._create_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            bidirectional=bidirectional,
            **kwargs
        )
        self._storage.add_edge(edge)
        return edge

    def get_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = ""
    ) -> Optional[E]:
        """Get a specific edge, or None if not found."""
        return self._storage.get_edge(source_id, target_id, edge_type)

    def remove_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = ""
    ) -> bool:
        """Remove an edge. Returns True if removed, False if not found."""
        return self._storage.remove_edge(source_id, target_id, edge_type)

    def edges_from(self, node_id: str) -> List[E]:
        """Get all outgoing edges from a node."""
        return self._storage.edges_from(node_id)

    def edges_to(self, node_id: str) -> List[E]:
        """Get all incoming edges to a node."""
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

    def neighbors(self, node_id: str, direction: str = "out") -> List[str]:
        """
        Get neighbor node IDs.

        Args:
            node_id: Node to get neighbors for
            direction: "out" (outgoing), "in" (incoming), or "both"

        Returns:
            List of neighbor node IDs
        """
        neighbors = []

        if direction in ("out", "both"):
            for edge in self._storage.edges_from(node_id):
                neighbors.append(edge.target_id)

        if direction in ("in", "both"):
            for edge in self._storage.edges_to(node_id):
                neighbors.append(edge.source_id)

        return neighbors

    def degree(self, node_id: str, direction: str = "both") -> int:
        """Get node degree (number of connections)."""
        return len(self.neighbors(node_id, direction))

    # =========================================================================
    # Traversal Operations
    # =========================================================================

    def bfs(
        self,
        start_id: str,
        visitor: Optional[Callable[[N, T], T]] = None,
        initial: T = None,
        max_depth: Optional[int] = None,
        edge_filter: Optional[Callable[[E], bool]] = None,
        node_filter: Optional[Callable[[N], bool]] = None,
        direction: str = "out"
    ) -> Union[List[str], T]:
        """
        Breadth-first search traversal.

        Args:
            start_id: Starting node ID
            visitor: Optional function (node, accumulator) -> accumulator
            initial: Initial accumulator value
            max_depth: Maximum traversal depth
            edge_filter: Filter edges to follow
            node_filter: Filter nodes to visit
            direction: "out", "in", or "both"

        Returns:
            If visitor provided: Final accumulator value
            Otherwise: List of node IDs in BFS order
        """
        start_node = self._storage.get_node(start_id)
        if start_node is None:
            raise ValueError(f"Start node {start_id} not found")

        visited: Set[str] = {start_id}
        queue: deque = deque([(start_node, 0)])
        result: List[str] = [start_id]
        acc = initial

        while queue:
            node, depth = queue.popleft()

            if max_depth is not None and depth > max_depth:
                continue

            if node_filter is not None and not node_filter(node):
                continue

            if visitor is not None:
                acc = visitor(node, acc)

            # Get neighbors based on direction
            edges = []
            if direction in ("out", "both"):
                edges.extend(self._storage.edges_from(node.id))
            if direction in ("in", "both"):
                edges.extend(self._storage.edges_to(node.id))

            for edge in edges:
                if edge_filter is not None and not edge_filter(edge):
                    continue

                neighbor_id = (edge.target_id if edge.source_id == node.id
                              else edge.source_id)

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor = self._storage.get_node(neighbor_id)
                    if neighbor:
                        result.append(neighbor_id)
                        queue.append((neighbor, depth + 1))

        return acc if visitor else result

    def dfs(
        self,
        start_id: str,
        visitor: Optional[Callable[[N, T], T]] = None,
        initial: T = None,
        max_depth: Optional[int] = None,
        edge_filter: Optional[Callable[[E], bool]] = None,
        node_filter: Optional[Callable[[N], bool]] = None,
        direction: str = "out"
    ) -> Union[List[str], T]:
        """
        Depth-first search traversal.

        Same parameters as bfs().
        """
        start_node = self._storage.get_node(start_id)
        if start_node is None:
            raise ValueError(f"Start node {start_id} not found")

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

            edges = []
            if direction in ("out", "both"):
                edges.extend(self._storage.edges_from(node.id))
            if direction in ("in", "both"):
                edges.extend(self._storage.edges_to(node.id))

            for edge in edges:
                if edge_filter is not None and not edge_filter(edge):
                    continue

                neighbor_id = (edge.target_id if edge.source_id == node.id
                              else edge.source_id)
                neighbor = self._storage.get_node(neighbor_id)
                if neighbor:
                    recurse(neighbor, depth + 1)

        recurse(start_node, 0)
        return acc if visitor else result

    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        direction: str = "out"
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.

        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        if from_id == to_id:
            return [from_id]

        if not self.has_node(from_id) or not self.has_node(to_id):
            return None

        visited: Set[str] = {from_id}
        queue: deque = deque([(from_id, [from_id])])

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
        """Find all cycles in the graph."""
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
        """Check if graph contains any cycle."""
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
        """Find all connected components (treating graph as undirected)."""
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for node in self.nodes:
            if node.id not in visited:
                component: Set[str] = set()
                queue = deque([node.id])

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

        Raises:
            ValueError: If graph contains a cycle
        """
        import heapq

        # Compute in-degrees
        in_degree: Dict[str, int] = {node.id: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1

        # Initialize heap with zero in-degree nodes
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
        """Find nodes with no incoming edges."""
        return {node.id for node in self.nodes
                if len(self._storage.edges_to(node.id)) == 0}

    def find_leaves(self) -> Set[str]:
        """Find nodes with no outgoing edges."""
        return {node.id for node in self.nodes
                if len(self._storage.edges_from(node.id)) == 0}

    def find_hubs(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Find nodes with highest degree."""
        degrees = [(node.id, self.degree(node.id)) for node in self.nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_n]

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [self._node_to_dict(n) for n in self.nodes],
            "edges": [self._edge_to_dict(e) for e in self.edges],
        }

    def _node_to_dict(self, node: N) -> Dict[str, Any]:
        """Serialize a node. Override for custom node types."""
        if hasattr(node, 'to_dict'):
            return node.to_dict()
        return {
            "id": node.id,
            "node_type": getattr(node, 'node_type', ''),
            "content": getattr(node, 'content', ''),
            "properties": getattr(node, 'properties', {}),
        }

    def _edge_to_dict(self, edge: E) -> Dict[str, Any]:
        """Serialize an edge. Override for custom edge types."""
        if hasattr(edge, 'to_dict'):
            return edge.to_dict()
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "BaseGraph":
        """Deserialize graph from dictionary."""
        graph = cls(**kwargs)

        for node_data in data.get("nodes", []):
            graph.add_node(**node_data)

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
        """Create a shallow copy of the graph."""
        new_graph = type(self)()
        for node in self.nodes:
            new_graph._storage.add_node(node)
        for edge in self.edges:
            new_graph._storage.add_edge(edge)
        return new_graph

    def subgraph(self, node_ids: Set[str]) -> "BaseGraph[N, E]":
        """Create a subgraph containing only specified nodes."""
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
```

### Layer 4: Algorithm Mixins

```python
# cortical/graph/algorithms.py
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC


class PageRankMixin:
    """
    Mixin providing PageRank algorithm.

    Usage:
        class MyGraph(BaseGraph, PageRankMixin):
            pass
    """

    def compute_pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 20,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes.

        Optimized O(E × iterations) implementation with early termination.

        Args:
            damping: Damping factor (0-1, typically 0.85)
            iterations: Maximum iterations
            tolerance: Convergence threshold

        Returns:
            Dict mapping node_id to PageRank score
        """
        n = self.node_count
        if n == 0:
            return {}

        # Initialize uniform distribution
        pr = {node.id: 1.0 / n for node in self.nodes}

        # Build outgoing edge count
        out_degree = {node.id: len(self.edges_from(node.id)) for node in self.nodes}

        for _ in range(iterations):
            new_pr = {}
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


class ClusteringMixin:
    """
    Mixin providing clustering algorithms.
    """

    def label_propagation(
        self,
        max_iterations: int = 100
    ) -> Dict[str, int]:
        """
        Community detection using label propagation.

        Returns:
            Dict mapping node_id to cluster_id
        """
        import random

        # Initialize: each node in its own cluster
        labels = {node.id: i for i, node in enumerate(self.nodes)}

        node_ids = list(labels.keys())

        for _ in range(max_iterations):
            changed = False
            random.shuffle(node_ids)

            for node_id in node_ids:
                # Count neighbor labels
                neighbor_labels: Dict[int, float] = {}

                for neighbor_id in self.neighbors(node_id, "both"):
                    label = labels[neighbor_id]
                    # Weight by edge weight if available
                    edge = self.get_edge(node_id, neighbor_id, "")
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


class SpreadingActivationMixin:
    """
    Mixin providing spreading activation algorithm.
    """

    def spread_activation(
        self,
        source_id: str,
        initial_activation: float = 1.0,
        decay: float = 0.5,
        max_hops: int = 3
    ) -> Dict[str, float]:
        """
        Spread activation from source through the graph.

        Args:
            source_id: Starting node
            initial_activation: Starting activation level
            decay: Decay factor per hop (0-1)
            max_hops: Maximum propagation distance

        Returns:
            Dict mapping node_id to activation level
        """
        activations = {source_id: initial_activation}
        frontier = [source_id]

        for hop in range(max_hops):
            current_decay = decay ** (hop + 1)
            next_frontier = []

            for node_id in frontier:
                for edge in self.edges_from(node_id):
                    target = edge.target_id
                    new_activation = activations.get(node_id, 0) * edge.weight * current_decay
                    activations[target] = max(activations.get(target, 0), new_activation)
                    if target not in next_frontier:
                        next_frontier.append(target)

            frontier = next_frontier

        return activations
```

### Layer 5: Concrete Implementations

```python
# cortical/graph/implementations.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from .base import BaseGraph
from .protocols import NodeBase, EdgeBase
from .algorithms import PageRankMixin, ClusteringMixin, SpreadingActivationMixin


# Example: Simple Graph (for general use)
@dataclass
class SimpleNode(NodeBase):
    """Simple node with minimal fields."""
    pass


@dataclass
class SimpleEdge(EdgeBase):
    """Simple edge with minimal fields."""
    pass


class SimpleGraph(BaseGraph[SimpleNode, SimpleEdge], PageRankMixin):
    """
    Simple graph implementation for general use.

    Example:
        graph = SimpleGraph()
        graph.add_node("A", content="Node A")
        graph.add_node("B", content="Node B")
        graph.add_edge("A", "B", edge_type="connects")
    """

    def _create_node(self, id: str, **kwargs) -> SimpleNode:
        return SimpleNode(id=id, **kwargs)

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs
    ) -> SimpleEdge:
        return SimpleEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            **kwargs
        )


# Example: Adapting ThoughtGraph to use BaseGraph
@dataclass
class ThoughtNodeV2(NodeBase):
    """ThoughtNode compatible with BaseGraph."""
    # Inherit: id, node_type, content, properties, metadata, created_at, modified_at
    # ThoughtGraph-specific extensions can be added here
    pass


@dataclass
class ThoughtEdgeV2(EdgeBase):
    """ThoughtEdge compatible with BaseGraph."""
    # Inherit: source_id, target_id, edge_type, weight, bidirectional, properties, created_at
    confidence: float = 1.0


class ThoughtGraphV2(
    BaseGraph[ThoughtNodeV2, ThoughtEdgeV2],
    PageRankMixin,
    ClusteringMixin,
    SpreadingActivationMixin
):
    """
    ThoughtGraph reimplemented on BaseGraph foundation.

    Provides all ThoughtGraph functionality plus:
    - Pluggable storage backends
    - Standardized algorithms
    - Better serialization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clusters: Dict[str, set] = {}

    def _create_node(self, id: str, **kwargs) -> ThoughtNodeV2:
        return ThoughtNodeV2(id=id, **kwargs)

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        confidence: float = 1.0,
        **kwargs
    ) -> ThoughtEdgeV2:
        return ThoughtEdgeV2(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            **kwargs
        )

    # ThoughtGraph-specific methods
    def add_cluster(self, cluster_id: str, node_ids: set) -> None:
        """Add a cluster of nodes."""
        self._clusters[cluster_id] = node_ids

    def get_cluster(self, node_id: str) -> Optional[str]:
        """Get cluster containing a node."""
        for cluster_id, nodes in self._clusters.items():
            if node_id in nodes:
                return cluster_id
        return None

    # Visualization (from original ThoughtGraph)
    def to_mermaid(self) -> str:
        """Export to Mermaid diagram format."""
        lines = ["graph TD"]

        for node in self.nodes:
            content = node.content[:30].replace('"', "'")
            lines.append(f'    {node.id}["{content}"]')

        for edge in self.edges:
            arrow = "<-->" if edge.bidirectional else "-->"
            label = edge.edge_type.replace("_", " ")
            lines.append(f"    {edge.source_id} {arrow}|{label}| {edge.target_id}")

        return "\n".join(lines)
```

---

## Migration Strategy

### Phase 1: Foundation (Non-Breaking)

1. Create `cortical/graph/` package with:
   - `protocols.py` - NodeBase, EdgeBase, protocols
   - `storage.py` - InMemoryGraphStorage, FileGraphStorage
   - `base.py` - BaseGraph abstract class
   - `algorithms.py` - PageRankMixin, ClusteringMixin, etc.

2. Add tests in `tests/unit/test_base_graph.py`

3. No changes to existing graph classes

### Phase 2: Adapter Layer

1. Create adapters that wrap existing graphs:
   ```python
   class ThoughtGraphAdapter(BaseGraph):
       """Adapts existing ThoughtGraph to BaseGraph interface."""

       def __init__(self, thought_graph: ThoughtGraph):
           self._wrapped = thought_graph

       def _create_node(self, id, **kwargs):
           return self._wrapped.add_node(id, **kwargs)
   ```

2. Test adapters pass all BaseGraph tests

### Phase 3: Incremental Migration

#### First Target: TaskDAG → DAGGraph ✅ COMPLETED (2026-01-13)

**Location:** `cortical/audits/algorithms/dag.py` (173 lines, down from 459)

**Migration Summary:**
- TaskDAG now wraps DAGGraph internally (composition over inheritance)
- All 11 built-in tests pass
- Full backward compatibility maintained
- Lines reduced: **-286 lines** (62% reduction)

**Performance validation (before migration):**
| Scale | Nodes  | TaskDAG (old) | DAGGraph | Ratio | Status |
|-------|--------|---------------|----------|-------|--------|
| 1x    | 1000   | 0.052ms       | 0.176ms  | 3.35x | Constant factor |
| 8x    | 8000   | 0.874ms       | 2.390ms  | 2.74x | Stable |
| 16x   | 16000  | 2.091ms       | 5.427ms  | 2.60x | Stable |
| 32x   | 32000  | 5.084ms       | 16.888ms | 3.32x | Acceptable |

**Key optimization:** `DAGGraph.ready_tasks()` was optimized from 7.26x overhead to ~3x by direct storage access (`_edges_by_target` index).

**API compatibility layer:**
```python
# cortical/audits/algorithms/dag.py
from cortical.graph.implementations import DAGGraph

class TaskDAG:
    """Wraps DAGGraph with task-oriented API."""

    def __init__(self):
        self._graph = DAGGraph()

    def add_task(self, task_id: str) -> None:
        if not self._graph.has_node(task_id):
            self._graph.add_node(task_id)

    def add_dependency(self, from_task: str, to_task: str) -> bool:
        try:
            self._graph.add_edge(from_task, to_task)
            return True
        except ValueError:
            return False  # Cycle or self-loop detected

    # ... delegates to DAGGraph for blocked_by, blocks, ready_tasks, etc.
```

#### Subsequent Migrations

2. `ThoughtGraph` -> Subclass `BaseGraph` with custom `ThoughtNode`/`ThoughtEdge`
3. `CausalGraph` -> Subclass `BaseGraph` with evidence tracking
4. `SemanticKnowledgeGraph` -> Complex (has layers, WAL) - defer
5. `CognitiveGraph` -> Keep separate (hypergraph, different paradigm)

### Phase 4: Deprecation

1. Mark old implementations as deprecated
2. Update documentation
3. Remove old implementations after transition period

---

## Performance Considerations

### Time Complexity

| Operation | BaseGraph | Current ThoughtGraph | Notes |
|-----------|-----------|---------------------|-------|
| Add node | O(1) | O(1) | Same |
| Get node | O(1) | O(1) | Same |
| Add edge | O(1) | O(1) | Same |
| Get neighbors | O(k) | O(k) | k = degree |
| BFS/DFS | O(V+E) | O(V+E) | Same |
| Shortest path | O(V+E) | O(V+E) | Same |
| PageRank | O(E×I) | O(E×I) | I = iterations |
| Find cycles | O(V+E) | O(V+E) | Same |

### Space Complexity

| Storage | Nodes | Edges | Index Overhead |
|---------|-------|-------|----------------|
| Current ThoughtGraph | O(V) | O(E) | O(E) for edge indices |
| BaseGraph | O(V) | O(E) | O(E) for edge indices |
| With bidirectional | O(V) | O(2E) | O(2E) |

### Memory Optimization

1. **Lazy loading**: Edges can be loaded on-demand
2. **Compression**: Edge indices can use compact representations
3. **Pooling**: Reuse edge type strings via interning

---

## Design Decisions

### Why Protocol + Abstract Class?

**Protocols** (duck typing contracts) allow any object with the right attributes to be used as a node or edge. This enables:
- Gradual migration without changing existing dataclasses
- External libraries to provide compatible types
- Runtime flexibility

**Abstract class** (BaseGraph) provides:
- Default implementations of common operations
- Template method pattern for customization points
- Consistent API across all graph types

### Why Pluggable Storage?

Different use cases need different storage characteristics:

| Use Case | Storage Backend | Rationale |
|----------|-----------------|-----------|
| Unit tests | InMemoryGraphStorage | Fast, no I/O |
| Development | FileGraphStorage | Git-friendly, debuggable |
| Production | TransactionalGraphStorage | ACID, crash recovery |
| Large graphs | PartitionedGraphStorage | Horizontal scaling |

### Why Mixins for Algorithms?

- **Composability**: Pick only the algorithms you need
- **Testability**: Test algorithms independently
- **Extensibility**: Add new algorithms without modifying BaseGraph
- **Performance**: Don't pay for unused code

---

## Future Extensions

### Planned Enhancements

1. **Weighted shortest path** (Dijkstra's algorithm)
2. **Betweenness centrality**
3. **Graph isomorphism detection**
4. **Streaming graph updates**
5. **Distributed graph storage**

### Extension Points

```python
class CustomGraph(BaseGraph):
    """Example of extending BaseGraph."""

    def _create_node(self, id, **kwargs):
        # Custom node creation with validation
        ...

    def _create_edge(self, source_id, target_id, **kwargs):
        # Custom edge creation with constraints
        ...

    def add_edge(self, source_id, target_id, **kwargs):
        # Override to add custom behavior (e.g., cycle prevention)
        if self._would_create_cycle(source_id, target_id):
            return None
        return super().add_edge(source_id, target_id, **kwargs)
```

---

## Graph Selection Guide: BaseGraph vs CognitiveGraph

The codebase has two primary graph architectures serving different purposes:

### When to Use BaseGraph (cortical/graph/)

Use `BaseGraph` and its implementations when you need:

| Use Case | Recommended Class |
|----------|-------------------|
| General-purpose graph operations | `SimpleGraph` |
| Task dependencies with cycle prevention | `DAGGraph` |
| Weighted shortest path (routing, costs) | `WeightedGraph` |
| Domain-specific nodes/edges | Subclass `BaseGraph` |
| PageRank, clustering, centrality | Any with algorithm mixins |

**Characteristics:**
- Traditional graph model (nodes + edges as separate types)
- Composable algorithm mixins (opt-in features)
- Pluggable storage backends via DI
- Type-safe with generics: `BaseGraph[NodeType, EdgeType]`
- Serializable to dict/JSON

**Example:**
```python
from cortical.graph import SimpleGraph

graph = SimpleGraph()
graph.add_node("A", content="Start")
graph.add_node("B", content="End")
graph.add_edge("A", "B", edge_type="CONNECTS", weight=0.9)
path = graph.shortest_path("A", "B")
```

### When to Use CognitiveGraph (cortical/cognitive/)

Use `CognitiveGraph` when you need:

| Use Case | Why CognitiveGraph |
|----------|-------------------|
| Meta-reasoning (statements about statements) | Links are atoms that can be linked to |
| Probabilistic truth values | TruthValue with strength + confidence |
| Attention dynamics | STI (short-term importance) spreading |
| Hebbian learning | Links strengthen with co-activation |
| NLP word associations | WORD atoms with SIMILARITY links |
| Code-to-concept bridging | REFERS_TO links between words and code |

**Characteristics:**
- Hypergraph model (links ARE atoms, can point to other links)
- Probabilistic logic (strength, confidence per atom)
- Content-addressed (same content = same atom)
- Attention economy (STI decay and spreading)
- Bio-inspired (Hebbian learning, attention)

**Example:**
```python
from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue

graph = CognitiveGraph()

# Create concepts
cat = graph.node("cat", AtomType.CONCEPT)
animal = graph.node("animal", AtomType.CONCEPT)

# Create relationship (link is an atom)
inheritance = graph.link(AtomType.INHERITANCE, [cat, animal],
                         tv=TruthValue(0.95, 0.9))

# Meta-reasoning: "John believes cats are animals"
john = graph.node("john", AtomType.PERSON)
belief = graph.link(AtomType.BELIEVES, [john, inheritance])
```

### Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     WHICH GRAPH SHOULD I USE?                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Need to make statements ABOUT relationships?                           │
│  ├─ YES → CognitiveGraph (hypergraph semantics)                         │
│  └─ NO  → BaseGraph                                                     │
│                                                                          │
│  Need probabilistic truth values?                                       │
│  ├─ YES → CognitiveGraph (TruthValue with strength/confidence)          │
│  └─ NO  → BaseGraph (boolean existence)                                 │
│                                                                          │
│  Need attention dynamics / Hebbian learning?                            │
│  ├─ YES → CognitiveGraph (STI, attention spreading)                     │
│  └─ NO  → BaseGraph                                                     │
│                                                                          │
│  Need cycle prevention (DAG)?                                           │
│  ├─ YES → DAGGraph                                                      │
│  └─ NO  → Continue                                                      │
│                                                                          │
│  Need weighted shortest paths?                                          │
│  ├─ YES → WeightedGraph                                                 │
│  └─ NO  → SimpleGraph                                                   │
│                                                                          │
│  Need custom node/edge types with standard algorithms?                  │
│  └─ Subclass BaseGraph with algorithm mixins                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Can They Interoperate?

Currently, `BaseGraph` and `CognitiveGraph` are **separate implementations** with different storage backends:

- `BaseGraph` uses `InMemoryGraphStorage` (cortical/graph/storage.py)
- `CognitiveGraph` uses `InMemoryStorage` (cortical/cognitive/graph.py)

**Future integration possibilities:**
1. Adapter to expose CognitiveGraph as BaseGraph for algorithm reuse
2. Shared storage protocol with type conversion
3. Bridge layer for cross-graph queries

For now, choose based on your requirements and use one consistently within a component.

---

## Appendix: File Structure

```
cortical/graph/
├── __init__.py           # Public API exports
├── protocols.py          # NodeBase, EdgeBase, NodeProtocol, EdgeProtocol
├── storage.py            # GraphStorage protocol, InMemoryGraphStorage
├── base.py               # BaseGraph abstract class
├── algorithms.py         # PageRankMixin, ClusteringMixin, etc.
├── implementations.py    # SimpleGraph, concrete implementations
├── walker.py             # Fluent graph walker (from got/graph_walker.py)
└── visualization.py      # Mermaid, DOT, ASCII export

tests/unit/
├── test_base_graph.py           # BaseGraph tests
├── test_graph_storage.py        # Storage backend tests
├── test_graph_algorithms.py     # Algorithm mixin tests
└── test_graph_implementations.py # Concrete implementation tests
```

---

## References

- [cortical/reasoning/thought_graph.py](../cortical/reasoning/thought_graph.py) - Current ThoughtGraph
- [cortical/graph/knowledge_graph.py](../cortical/graph/knowledge_graph.py) - SemanticKnowledgeGraph
- [cortical/cognitive/graph.py](../cortical/cognitive/graph.py) - CognitiveGraph
- [cortical/audits/algorithms/dag.py](../cortical/audits/algorithms/dag.py) - TaskDAG
- [cortical/got/graph_walker.py](../cortical/got/graph_walker.py) - GraphWalker
