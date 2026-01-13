"""
Graph Protocols: Type-safe contracts for nodes and edges.

This module defines the minimal contracts that all graph node and edge types
must satisfy. Using protocols (structural subtyping) allows gradual migration
without changing existing dataclass definitions.

Key Types:
- NodeBase: Minimal dataclass for graph nodes
- EdgeBase: Minimal dataclass for graph edges
- NodeProtocol: Protocol for duck-typed node compatibility
- EdgeProtocol: Protocol for duck-typed edge compatibility

Design Philosophy:
    We use both dataclasses (NodeBase, EdgeBase) and protocols (NodeProtocol,
    EdgeProtocol) to support two use cases:

    1. New code inherits from NodeBase/EdgeBase for consistency
    2. Existing code can satisfy NodeProtocol/EdgeProtocol without changes

Example:
    # Option 1: Inherit from base classes
    @dataclass
    class MyNode(NodeBase):
        activation: float = 0.0  # Add domain-specific field

    # Option 2: Use existing class that satisfies protocol
    @dataclass
    class LegacyNode:
        id: str  # Has required attribute, satisfies NodeProtocol
        node_type: str = ""

See docs/base-graph-design.md for architecture details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Protocol, runtime_checkable


@dataclass
class NodeBase:
    """
    Minimal node contract that all node types should satisfy.

    This is the recommended base class for new node types. It provides
    the essential fields that BaseGraph operations depend on.

    Subclasses can add domain-specific fields (activation, pagerank,
    tfidf, cluster_id, etc.) while maintaining compatibility.

    Attributes:
        id: Unique identifier for the node
        node_type: Type/category string (e.g., "concept", "task", "document")
        content: Primary content or description
        properties: Flexible key-value store for domain-specific data
        metadata: Additional metadata (tags, author, source, etc.)
        created_at: When the node was created
        modified_at: When the node was last modified

    Example:
        @dataclass
        class ConceptNode(NodeBase):
            activation: float = 0.0
            pagerank: float = 0.0
            cluster_id: Optional[int] = None
    """

    id: str
    node_type: str = ""
    content: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Hash based on ID for use in sets and as dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on ID."""
        if not isinstance(other, NodeBase):
            return NotImplemented
        return self.id == other.id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "content": self.content,
            "properties": self.properties,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeBase":
        """Deserialize node from dictionary."""
        created_at = data.get("created_at")
        modified_at = data.get("modified_at")

        return cls(
            id=data["id"],
            node_type=data.get("node_type", ""),
            content=data.get("content", ""),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(created_at)
                if isinstance(created_at, str)
                else created_at or datetime.now()
            ),
            modified_at=(
                datetime.fromisoformat(modified_at)
                if isinstance(modified_at, str)
                else modified_at or datetime.now()
            ),
        )


@dataclass
class EdgeBase:
    """
    Minimal edge contract that all edge types should satisfy.

    This is the recommended base class for new edge types. It provides
    the essential fields that BaseGraph operations depend on.

    Subclasses can add domain-specific fields (confidence, temporal_decay,
    evidence_refs, etc.) while maintaining compatibility.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Relationship type (e.g., "DEPENDS_ON", "SIMILAR", "IS_A")
        weight: Edge weight for algorithms (0.0 to 1.0, default 1.0)
        bidirectional: Whether the edge goes both ways
        properties: Flexible key-value store for domain-specific data
        created_at: When the edge was created

    Example:
        @dataclass
        class SynapticEdge(EdgeBase):
            confidence: float = 1.0
            temporal_decay: float = 0.0
            last_activated: Optional[datetime] = None
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
        """Hash based on source, target, and type for uniqueness."""
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        """Equality based on source, target, and type."""
        if not isinstance(other, EdgeBase):
            return NotImplemented
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.edge_type == other.edge_type
        )

    def __post_init__(self) -> None:
        """Validate edge attributes."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be in [0.0, 1.0], got {self.weight}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "properties": self.properties,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeBase":
        """Deserialize edge from dictionary."""
        created_at = data.get("created_at")

        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data.get("edge_type", ""),
            weight=data.get("weight", 1.0),
            bidirectional=data.get("bidirectional", False),
            properties=data.get("properties", {}),
            created_at=(
                datetime.fromisoformat(created_at)
                if isinstance(created_at, str)
                else created_at or datetime.now()
            ),
        )

    def reverse(self) -> "EdgeBase":
        """Create a reversed copy of this edge (swap source and target)."""
        return EdgeBase(
            source_id=self.target_id,
            target_id=self.source_id,
            edge_type=self.edge_type,
            weight=self.weight,
            bidirectional=False,  # Reversed edge is not bidirectional
            properties=self.properties.copy(),
        )


@runtime_checkable
class NodeProtocol(Protocol):
    """
    Protocol for node-like objects (structural subtyping).

    Any object with these attributes can be used as a node in BaseGraph,
    even if it doesn't inherit from NodeBase. This enables gradual
    migration of existing node types.

    Example:
        # This class satisfies NodeProtocol without inheriting NodeBase
        @dataclass
        class LegacyNode:
            id: str
            node_type: str = ""

        node = LegacyNode(id="N1")
        assert isinstance(node, NodeProtocol)  # True!
    """

    @property
    def id(self) -> str:
        """Unique identifier for the node."""
        ...

    @property
    def node_type(self) -> str:
        """Type/category of the node."""
        ...


@runtime_checkable
class EdgeProtocol(Protocol):
    """
    Protocol for edge-like objects (structural subtyping).

    Any object with these attributes can be used as an edge in BaseGraph,
    even if it doesn't inherit from EdgeBase. This enables gradual
    migration of existing edge types.

    Example:
        # This class satisfies EdgeProtocol without inheriting EdgeBase
        @dataclass
        class LegacyEdge:
            source_id: str
            target_id: str
            edge_type: str = ""
            weight: float = 1.0

        edge = LegacyEdge(source_id="A", target_id="B")
        assert isinstance(edge, EdgeProtocol)  # True!
    """

    @property
    def source_id(self) -> str:
        """ID of the source node."""
        ...

    @property
    def target_id(self) -> str:
        """ID of the target node."""
        ...

    @property
    def edge_type(self) -> str:
        """Type/category of the relationship."""
        ...

    @property
    def weight(self) -> float:
        """Edge weight for weighted algorithms."""
        ...
