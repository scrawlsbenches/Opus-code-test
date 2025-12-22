"""
Entity types for GoT (Graph of Thought) transactional system.

Provides base Entity class and concrete entity types (Task, Decision, Edge)
with versioning, checksums, and JSON serialization support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List

from cortical.utils.checksums import compute_checksum
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
class Sprint(Entity):
    """
    Sprint entity for time-boxed work periods.

    Sprints organize work into fixed time periods with goals, isolation,
    and session tracking. Each sprint can belong to an epic.
    """

    title: str = ""
    status: str = "available"
    epic_id: str = ""
    number: int = 0
    session_id: str = ""
    isolation: List[str] = field(default_factory=list)
    goals: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate sprint fields after initialization."""
        self.entity_type = "sprint"
        valid_statuses = {"available", "in_progress", "completed", "blocked"}
        if self.status not in valid_statuses:
            raise ValidationError(
                f"Invalid status '{self.status}'",
                valid_statuses=list(valid_statuses)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sprint to dictionary."""
        result = super().to_dict()
        result.update({
            "title": self.title,
            "status": self.status,
            "epic_id": self.epic_id,
            "number": self.number,
            "session_id": self.session_id,
            "isolation": self.isolation,
            "goals": self.goals,
            "notes": self.notes,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Sprint:
        """Deserialize sprint from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "sprint"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            title=data.get("title", ""),
            status=data.get("status", "available"),
            epic_id=data.get("epic_id", ""),
            number=data.get("number", 0),
            session_id=data.get("session_id", ""),
            isolation=data.get("isolation", []),
            goals=data.get("goals", []),
            notes=data.get("notes", []),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Epic(Entity):
    """
    Epic entity for large initiatives spanning multiple sprints.

    Epics organize work into high-level goals with phases, tracking
    progress across multiple sprints and work periods.
    """

    title: str = ""
    status: str = "active"
    phase: int = 1
    phases: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate epic fields after initialization."""
        self.entity_type = "epic"
        valid_statuses = {"active", "completed", "on_hold"}
        if self.status not in valid_statuses:
            raise ValidationError(
                f"Invalid status '{self.status}'",
                valid_statuses=list(valid_statuses)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize epic to dictionary."""
        result = super().to_dict()
        result.update({
            "title": self.title,
            "status": self.status,
            "phase": self.phase,
            "phases": self.phases,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Epic:
        """Deserialize epic from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "epic"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            title=data.get("title", ""),
            status=data.get("status", "active"),
            phase=data.get("phase", 1),
            phases=data.get("phases", []),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
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
