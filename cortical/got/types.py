"""
Entity types for GoT (Graph of Thought) transactional system.

Provides base Entity class and concrete entity types (Task, Decision, Edge)
with versioning, checksums, and JSON serialization support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List

from .checksums import compute_checksum
from .errors import ValidationError


@dataclass
class Entity:
    """
    Base class for all versioned entities in the GoT system.

    Provides common fields for optimistic locking, timestamps, and checksums.
    All entities must be JSON-serializable.
    """

    id: str
    entity_type: str = ""
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
            entity_type=data["entity_type"],
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
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


@dataclass
class Task(Entity):
    """
    Task entity representing a work item in the GoT system.

    Tasks track status, priority, and arbitrary properties for workflow management.
    """

    title: str = ""
    status: str = "pending"
    priority: str = "medium"
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task fields after initialization."""
        self.entity_type = "task"
        valid_statuses = {"pending", "in_progress", "completed", "blocked"}
        if self.status not in valid_statuses:
            raise ValidationError(
                f"Invalid status '{self.status}'",
                valid_statuses=list(valid_statuses)
            )
        valid_priorities = {"low", "medium", "high", "critical"}
        if self.priority not in valid_priorities:
            raise ValidationError(
                f"Invalid priority '{self.priority}'",
                valid_priorities=list(valid_priorities)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        result = super().to_dict()
        result.update({
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "description": self.description,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Deserialize task from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "task"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            title=data.get("title", ""),
            status=data.get("status", "pending"),
            priority=data.get("priority", "medium"),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Decision(Entity):
    """
    Decision entity representing a logged choice with rationale.

    Decisions capture why something was decided and which entities are affected.
    """

    title: str = ""
    rationale: str = ""
    affects: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set entity type after initialization."""
        self.entity_type = "decision"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision to dictionary."""
        result = super().to_dict()
        result.update({
            "title": self.title,
            "rationale": self.rationale,
            "affects": self.affects,
            "properties": self.properties,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Decision:
        """Deserialize decision from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "decision"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            title=data.get("title", ""),
            rationale=data.get("rationale", ""),
            affects=data.get("affects", []),
            properties=data.get("properties", {}),
        )


@dataclass
class Edge(Entity):
    """
    Edge entity representing a relationship between two entities.

    Edges connect entities with typed relationships (DEPENDS_ON, BLOCKS, etc.)
    and optional weight/confidence scores.
    """

    source_id: str = ""
    target_id: str = ""
    edge_type: str = ""
    weight: float = 1.0
    confidence: float = 1.0

    def __post_init__(self):
        """Validate edge fields and auto-generate ID if needed."""
        self.entity_type = "edge"

        # Auto-generate ID if not provided or empty
        if not self.id:
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
        """Serialize edge to dictionary."""
        result = super().to_dict()
        result.update({
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "confidence": self.confidence,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        """Deserialize edge from dictionary."""
        return cls(
            id=data.get("id", ""),  # Allow empty for auto-generation
            entity_type=data.get("entity_type", "edge"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            edge_type=data.get("edge_type", ""),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
        )
