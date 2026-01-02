"""
Core entity types for the Cortical Distributed Graph.

Provides the foundational data structures for graph storage:
- Entity: Base class for all graph nodes
- Node: Alias for Entity (CDG terminology)
- Edge: Relationship between entities

These types are lifted from GoT (Graph of Thoughts) with CDG extensions
for partition awareness and additional metadata.

Design Principles:
- Immutable after creation (modify via new versions)
- JSON-serializable for persistence
- Checksum-verifiable for integrity
- Partition-aware for distributed storage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from cortical.utils.checksums import compute_checksum
from .errors import ValidationError


# Valid edge types - single source of truth
# These are the default CDG edge types. Domain-specific graphs
# can extend this set via schema configuration.
VALID_EDGE_TYPES = frozenset({
    # Core relationship types
    'DEPENDS_ON',    # A depends on B
    'BLOCKS',        # A blocks B
    'CONTAINS',      # A contains B (hierarchical)
    'RELATES_TO',    # General relationship
    'REQUIRES',      # Hard requirement
    'IMPLEMENTS',    # A implements B
    'SUPERSEDES',    # A replaces B
    'DERIVED_FROM',  # A derived from B
    # Hierarchical relationships
    'PARENT_OF',     # Hierarchical parent
    'CHILD_OF',      # Hierarchical child
    'PART_OF',       # Component of larger entity
    # Reference and semantic relationships
    'REFERENCES',    # Soft reference
    'CONTRADICTS',   # Conflicting entities
    'JUSTIFIES',     # A justifies B
    'MOTIVATES',     # A motivates B
    'CAUSED_BY',     # A was caused by B
    # Workflow relationships
    'TRANSFERS',     # A transfers to B
    'PRODUCES',      # A produces B
    'DOCUMENTED_BY', # A is documented by B
})


@dataclass
class Entity:
    """
    Base class for all versioned entities in CDG.

    Provides common fields for optimistic locking, timestamps, checksums,
    and partition routing. All entities are JSON-serializable.

    Lifted from GoT with CDG extensions:
    - partition_key: Optional hint for partition routing
    - properties: Flexible key-value store for domain data

    Attributes:
        id: Unique entity identifier (e.g., "E-001", "task-123")
        entity_type: Type discriminator (e.g., "task", "document")
        version: Monotonic version for optimistic locking
        created_at: ISO 8601 creation timestamp
        modified_at: ISO 8601 last modification timestamp
        partition_key: Optional partition routing hint
        properties: Flexible properties for domain-specific data

    Example:
        entity = Entity(
            id="DOC-001",
            entity_type="document",
            properties={"title": "Design Doc", "status": "draft"}
        )
    """

    id: str
    entity_type: str = ""
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # CDG extension: partition hint
    partition_key: Optional[str] = None

    # CDG extension: flexible properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize entity to JSON-serializable dictionary.

        Returns:
            Dictionary containing all entity fields
        """
        result = {
            "id": self.id,
            "entity_type": self.entity_type,
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }
        if self.partition_key is not None:
            result["partition_key"] = self.partition_key
        if self.properties:
            result["properties"] = self.properties
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Entity:
        """
        Deserialize entity from dictionary.

        Args:
            data: Dictionary containing entity fields

        Returns:
            New Entity instance
        """
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            partition_key=data.get("partition_key"),
            properties=data.get("properties", {}),
        )

    def compute_checksum(self) -> str:
        """
        Compute SHA256 checksum of entity data.

        Returns:
            First 16 characters of hex digest
        """
        return compute_checksum(self.to_dict())

    def bump_version(self) -> None:
        """Increment version and update modified_at timestamp."""
        self.version += 1
        self.modified_at = datetime.now(timezone.utc).isoformat()

    def with_property(self, key: str, value: Any) -> "Entity":
        """
        Return new entity with additional property.

        This is a convenience method for immutable-style updates.

        Args:
            key: Property key
            value: Property value

        Returns:
            New Entity with updated properties
        """
        new_props = {**self.properties, key: value}
        return Entity(
            id=self.id,
            entity_type=self.entity_type,
            version=self.version,
            created_at=self.created_at,
            modified_at=datetime.now(timezone.utc).isoformat(),
            partition_key=self.partition_key,
            properties=new_props,
        )


# Node is an alias for Entity in CDG terminology
Node = Entity


@dataclass
class Edge:
    """
    Relationship between two entities in CDG.

    Edges connect entities with typed relationships and optional
    weight/confidence scores for weighted graph algorithms.

    Lifted from GoT with CDG extensions:
    - created_at/modified_at timestamps
    - properties for flexible metadata

    Attributes:
        id: Unique edge identifier (auto-generated if empty)
        source_id: ID of source entity
        target_id: ID of target entity
        edge_type: Relationship type (must be in VALID_EDGE_TYPES)
        weight: Edge weight for algorithms (0.0 to 1.0)
        confidence: Confidence score (0.0 to 1.0)
        version: Version for optimistic locking
        created_at: ISO 8601 creation timestamp
        modified_at: ISO 8601 last modification timestamp
        properties: Flexible properties for domain-specific data

    Example:
        edge = Edge(
            source_id="TASK-001",
            target_id="TASK-002",
            edge_type="DEPENDS_ON",
            weight=0.8,
            confidence=0.95
        )
    """

    id: str = ""
    source_id: str = ""
    target_id: str = ""
    edge_type: str = ""
    weight: float = 1.0
    confidence: float = 1.0
    version: int = 1

    # CDG extension: timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # CDG extension: flexible properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge fields and auto-generate ID if needed."""
        # Validate edge_type against allowed values
        if self.edge_type and self.edge_type not in VALID_EDGE_TYPES:
            raise ValidationError(
                f"Invalid edge_type: '{self.edge_type}'. "
                f"Must be one of: {sorted(VALID_EDGE_TYPES)}",
                edge_type=self.edge_type,
                valid_types=sorted(VALID_EDGE_TYPES)
            )

        # Auto-generate ID if not provided or empty
        if not self.id and self.source_id and self.target_id and self.edge_type:
            self.id = f"E-{self.source_id}-{self.target_id}-{self.edge_type}"

        # Validate weight bounds
        if not (0.0 <= self.weight <= 1.0):
            raise ValidationError(
                f"Edge weight must be in [0.0, 1.0], got {self.weight}",
                weight=self.weight
            )

        # Validate confidence bounds
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError(
                f"Edge confidence must be in [0.0, 1.0], got {self.confidence}",
                confidence=self.confidence
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize edge to JSON-serializable dictionary.

        Returns:
            Dictionary containing all edge fields
        """
        result = {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }
        if self.properties:
            result["properties"] = self.properties
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        """
        Deserialize edge from dictionary.

        Args:
            data: Dictionary containing edge fields

        Returns:
            New Edge instance
        """
        return cls(
            id=data.get("id", ""),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            edge_type=data.get("edge_type", ""),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            properties=data.get("properties", {}),
        )

    def compute_checksum(self) -> str:
        """
        Compute SHA256 checksum of edge data.

        Returns:
            First 16 characters of hex digest
        """
        return compute_checksum(self.to_dict())

    def bump_version(self) -> None:
        """Increment version and update modified_at timestamp."""
        self.version += 1
        self.modified_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_bidirectional(self) -> bool:
        """
        Check if edge type is typically bidirectional.

        Bidirectional edges like RELATES_TO should be traversable
        in both directions with equal weight.

        Returns:
            True if edge type is bidirectional
        """
        bidirectional_types = {'RELATES_TO', 'CONTRADICTS'}
        return self.edge_type in bidirectional_types

    def reverse(self) -> "Edge":
        """
        Create reversed edge (swap source and target).

        Useful for bidirectional traversal. The reversed edge
        gets a new auto-generated ID.

        Returns:
            New Edge with source/target swapped
        """
        return Edge(
            id="",  # Will auto-generate
            source_id=self.target_id,
            target_id=self.source_id,
            edge_type=self.edge_type,
            weight=self.weight,
            confidence=self.confidence,
            version=1,
            properties=self.properties.copy(),
        )
