"""
Entity types for GoT (Graph of Thought) transactional system.

Provides base Entity class and concrete entity types (Task, Decision, Edge)
with versioning, checksums, and JSON serialization support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, TypedDict

from cortical.utils.checksums import compute_checksum
from .errors import ValidationError


# =============================================================================
# TYPED DICT SCHEMAS FOR NESTED STRUCTURES
# =============================================================================
# These TypedDicts define the expected structure of Dict[str, Any] fields
# used in entity types. They enable type checking and documentation while
# remaining backward-compatible with existing untyped dictionaries.


class TaskProperties(TypedDict, total=False):
    """
    Expected structure for Task.properties field.

    All fields are optional (total=False) for backward compatibility.

    Example:
        properties = TaskProperties(
            category="feature",
            tags=["api", "core"],
            estimated_hours=4.0
        )
    """
    category: str           # Task category: "feature", "bugfix", "chore", etc.
    retrospective: str      # Completion notes / lessons learned
    tags: List[str]         # Classification tags
    estimated_hours: float  # Time estimate in hours
    sprint_id: str          # Sprint this task belongs to
    epic_id: str            # Epic this task is part of


class TaskMetadata(TypedDict, total=False):
    """
    Expected structure for Task.metadata field.

    Stores operational metadata like timestamps and attribution.

    Example:
        metadata = TaskMetadata(
            completed_at="2025-12-26T10:00:00+00:00",
            created_by="agent-abc123"
        )
    """
    completed_at: str       # ISO 8601 completion timestamp
    started_at: str         # ISO 8601 start timestamp
    updated_at: str         # ISO 8601 last update timestamp
    created_by: str         # Agent/user ID who created the task
    completed_by: str       # Agent/user ID who completed the task
    blocked_reason: str     # Why task is blocked (if status="blocked")
    blocked_by: List[str]   # Task IDs blocking this task


class SprintGoal(TypedDict, total=False):
    """
    Structure for individual goals in Sprint.goals list.

    Goals track sprint objectives with completion status.

    Example:
        goal = SprintGoal(
            description="Implement edge type validation",
            completed=True,
            progress=1.0
        )
    """
    description: str        # Goal description (required in practice)
    completed: bool         # Whether goal is complete
    progress: float         # Progress as 0.0-1.0 (optional)
    priority: str           # Goal priority: "high", "medium", "low"


class SprintMetadata(TypedDict, total=False):
    """
    Expected structure for Sprint.metadata field.

    Stores sprint-level operational metadata.
    """
    started_at: str         # ISO 8601 start timestamp
    completed_at: str       # ISO 8601 completion timestamp
    velocity: float         # Story points or task count completed
    notes: str              # Sprint retrospective notes


class EpicPhase(TypedDict, total=False):
    """
    Structure for phases in Epic.phases list.

    Phases organize epic work into sequential stages.

    Example:
        phase = EpicPhase(
            name="Foundation",
            status="completed",
            sprints=["S-018", "S-019"]
        )
    """
    name: str               # Phase name (required in practice)
    status: str             # "pending", "in_progress", "completed"
    description: str        # Phase description
    sprints: List[str]      # Sprint IDs in this phase
    order: int              # Phase sequence number


class EpicMetadata(TypedDict, total=False):
    """
    Expected structure for Epic.metadata field.
    """
    started_at: str         # ISO 8601 start timestamp
    completed_at: str       # ISO 8601 completion timestamp
    owner: str              # Primary owner/responsible agent


class HandoffContext(TypedDict, total=False):
    """
    Structure for Handoff.context field.

    Context provides necessary information for the target agent.

    Example:
        context = HandoffContext(
            current_branch="claude/feature-x",
            files_modified=["api.py", "tests/test_api.py"],
            sprint_id="S-026"
        )
    """
    current_branch: str         # Git branch to work on
    files_modified: List[str]   # Files changed so far
    files_to_review: List[str]  # Files needing attention
    sprint_id: str              # Associated sprint
    epic_id: str                # Associated epic
    blockers: List[str]         # Current blocking issues
    dependencies: List[str]     # Task IDs this depends on
    notes: str                  # Free-form context notes
    session_id: str             # Source session ID


class HandoffResult(TypedDict, total=False):
    """
    Structure for Handoff.result field.

    Returned by the target agent upon completion.

    Example:
        result = HandoffResult(
            success=True,
            tasks_completed=["T-001", "T-002"],
            summary="Completed edge validation implementation"
        )
    """
    success: bool               # Whether handoff work succeeded
    tasks_completed: List[str]  # Task IDs completed
    tasks_created: List[str]    # New task IDs created
    commits: List[str]          # Commit hashes made
    files_modified: List[str]   # Files that were changed
    summary: str                # Summary of work done
    blockers: List[str]         # Remaining blockers (if any)
    next_steps: List[str]       # Suggested next actions


class DocumentMetadata(TypedDict, total=False):
    """
    Expected structure for Document.metadata field.
    """
    author: str             # Document author
    reviewers: List[str]    # Document reviewers
    status: str             # "draft", "review", "published"
    version: str            # Document version string


# Type aliases for common nested list types
SprintGoals = List[SprintGoal]
EpicPhases = List[EpicPhase]


# Valid edge types - single source of truth
# This set is re-exported in entity_schemas.py for schema validation
VALID_EDGE_TYPES = frozenset({
    # Core relationship types
    'DEPENDS_ON',    # Task A depends on Task B
    'BLOCKS',        # Task A blocks Task B
    'CONTAINS',      # Sprint contains Task, Epic contains Sprint
    'RELATES_TO',    # General relationship
    'REQUIRES',      # Hard requirement
    'IMPLEMENTS',    # Task implements Decision
    'SUPERSEDES',    # Entity replaces another
    'DERIVED_FROM',  # Entity derived from another
    # Hierarchical relationships
    'PARENT_OF',     # Hierarchical parent
    'CHILD_OF',      # Hierarchical child
    'PART_OF',       # Component of larger entity (Sprint PART_OF Epic)
    # Reference and semantic relationships
    'REFERENCES',    # Soft reference
    'CONTRADICTS',   # Conflicting entities
    'JUSTIFIES',     # Decision justifies Task
    'MOTIVATES',     # Entity motivates another
    # Workflow relationships
    'TRANSFERS',     # Task transfers to Handoff
    'PRODUCES',      # Task produces Document/Artifact
    'DOCUMENTED_BY', # Task/Decision is documented by Document (inverse of PRODUCES)
})


# Valid entity types - single source of truth for deserialization
VALID_ENTITY_TYPES = frozenset({
    'task',
    'decision',
    'edge',
    'sprint',
    'epic',
    'handoff',
    'claudemd_layer',
    'claudemd_version',
    'persona_profile',
    'team',
    'document',
})


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

        # Normalize edge_type to uppercase for consistency
        if self.edge_type:
            self.edge_type = self.edge_type.upper()

        # Validate edge_type against allowed values
        if self.edge_type and self.edge_type not in VALID_EDGE_TYPES:
            raise ValidationError(
                f"Invalid edge_type: '{self.edge_type}'. "
                f"Must be one of: {sorted(VALID_EDGE_TYPES)}",
                edge_type=self.edge_type,
                valid_types=sorted(VALID_EDGE_TYPES)
            )

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


@dataclass
class Handoff(Entity):
    """
    Handoff entity representing an agent-to-agent work transfer.

    Handoffs track the lifecycle of transferring work from one agent to
    another, including context, instructions, and completion status.

    Status lifecycle: initiated → accepted → completed
                                ↘ rejected
    """

    source_agent: str = ""
    target_agent: str = ""
    task_id: str = ""
    status: str = "initiated"
    instructions: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    initiated_at: str = ""
    accepted_at: str = ""
    completed_at: str = ""
    rejected_at: str = ""
    reject_reason: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate handoff fields after initialization."""
        self.entity_type = "handoff"
        valid_statuses = {"initiated", "accepted", "completed", "rejected"}
        if self.status not in valid_statuses:
            raise ValidationError(
                f"Invalid status '{self.status}'",
                valid_statuses=list(valid_statuses)
            )
        # Auto-set initiated_at if not provided
        if not self.initiated_at:
            self.initiated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize handoff to dictionary."""
        result = super().to_dict()
        result.update({
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "task_id": self.task_id,
            "status": self.status,
            "instructions": self.instructions,
            "context": self.context,
            "result": self.result,
            "artifacts": self.artifacts,
            "initiated_at": self.initiated_at,
            "accepted_at": self.accepted_at,
            "completed_at": self.completed_at,
            "rejected_at": self.rejected_at,
            "reject_reason": self.reject_reason,
            "properties": self.properties,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Handoff":
        """Deserialize handoff from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "handoff"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            source_agent=data.get("source_agent", ""),
            target_agent=data.get("target_agent", ""),
            task_id=data.get("task_id", ""),
            status=data.get("status", "initiated"),
            instructions=data.get("instructions", ""),
            context=data.get("context", {}),
            result=data.get("result", {}),
            artifacts=data.get("artifacts", []),
            initiated_at=data.get("initiated_at", ""),
            accepted_at=data.get("accepted_at", ""),
            completed_at=data.get("completed_at", ""),
            rejected_at=data.get("rejected_at", ""),
            reject_reason=data.get("reject_reason", ""),
            properties=data.get("properties", {}),
        )


@dataclass
class ClaudeMdLayer(Entity):
    """
    CLAUDE.md layer content entity.

    Stores markdown content for a specific layer of the CLAUDE.md file,
    with freshness tracking and inclusion rules for context-aware generation.
    """

    # Core fields
    layer_type: str = ""          # "core", "operational", "contextual", "persona", "ephemeral"
    layer_number: int = 0         # 0-4
    section_id: str = ""          # e.g., "architecture", "query-module"
    title: str = ""               # Human-readable title
    content: str = ""             # Markdown content

    # Freshness tracking
    freshness_status: str = "fresh"  # "fresh", "stale", "regenerating"
    freshness_decay_days: int = 0    # 0 = never decay
    last_regenerated: str = ""       # ISO 8601 timestamp
    regeneration_trigger: str = ""   # What caused last regeneration

    # Inclusion rules
    inclusion_rule: str = "always"   # "always", "context", "user_pref"
    context_modules: List[str] = field(default_factory=list)  # For context-based inclusion
    context_branches: List[str] = field(default_factory=list)  # Branch patterns

    # Versioning
    content_hash: str = ""        # SHA256 of content (first 16 chars)
    version_number: int = 1       # Content version

    # Metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate layer fields after initialization."""
        self.entity_type = "claudemd_layer"

        # Validate layer_type
        valid_layer_types = {"core", "operational", "contextual", "persona", "ephemeral", ""}
        if self.layer_type and self.layer_type not in valid_layer_types:
            raise ValidationError(
                f"Invalid layer_type '{self.layer_type}'",
                valid_types=list(valid_layer_types - {""})
            )

        # Validate freshness_status
        valid_freshness = {"fresh", "stale", "regenerating"}
        if self.freshness_status not in valid_freshness:
            raise ValidationError(
                f"Invalid freshness_status '{self.freshness_status}'",
                valid_statuses=list(valid_freshness)
            )

        # Validate inclusion_rule
        valid_inclusion = {"always", "context", "user_pref"}
        if self.inclusion_rule not in valid_inclusion:
            raise ValidationError(
                f"Invalid inclusion_rule '{self.inclusion_rule}'",
                valid_rules=list(valid_inclusion)
            )

        # Validate layer_number
        if not (0 <= self.layer_number <= 4):
            raise ValidationError(
                f"Invalid layer_number {self.layer_number}, must be 0-4",
                layer_number=self.layer_number
            )

    def compute_content_hash(self) -> str:
        """Compute hash of content for change detection."""
        import hashlib
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def is_stale(self, current_date: datetime = None) -> bool:
        """Check if layer content is stale based on decay rules."""
        if self.freshness_decay_days == 0:
            return False  # Never decays
        if not self.last_regenerated:
            return True
        from datetime import timedelta
        current = current_date or datetime.now(timezone.utc)
        # Parse ISO timestamp
        if isinstance(self.last_regenerated, str):
            regen_date = datetime.fromisoformat(self.last_regenerated.replace('Z', '+00:00'))
        else:
            regen_date = self.last_regenerated
        return (current - regen_date).days > self.freshness_decay_days

    def mark_stale(self, reason: str = "") -> None:
        """Mark layer as needing regeneration."""
        self.freshness_status = "stale"
        if reason:
            self.regeneration_trigger = reason
        self.bump_version()

    def mark_fresh(self) -> None:
        """Mark layer as freshly regenerated."""
        self.freshness_status = "fresh"
        self.last_regenerated = datetime.now(timezone.utc).isoformat()
        self.content_hash = self.compute_content_hash()
        self.bump_version()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer to dictionary."""
        result = super().to_dict()
        result.update({
            "layer_type": self.layer_type,
            "layer_number": self.layer_number,
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "freshness_status": self.freshness_status,
            "freshness_decay_days": self.freshness_decay_days,
            "last_regenerated": self.last_regenerated,
            "regeneration_trigger": self.regeneration_trigger,
            "inclusion_rule": self.inclusion_rule,
            "context_modules": self.context_modules,
            "context_branches": self.context_branches,
            "content_hash": self.content_hash,
            "version_number": self.version_number,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaudeMdLayer":
        """Deserialize layer from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "claudemd_layer"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            layer_type=data.get("layer_type", ""),
            layer_number=data.get("layer_number", 0),
            section_id=data.get("section_id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            freshness_status=data.get("freshness_status", "fresh"),
            freshness_decay_days=data.get("freshness_decay_days", 0),
            last_regenerated=data.get("last_regenerated", ""),
            regeneration_trigger=data.get("regeneration_trigger", ""),
            inclusion_rule=data.get("inclusion_rule", "always"),
            context_modules=data.get("context_modules", []),
            context_branches=data.get("context_branches", []),
            content_hash=data.get("content_hash", ""),
            version_number=data.get("version_number", 1),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ClaudeMdVersion(Entity):
    """
    Version snapshot for CLAUDE.md layer evolution tracking.

    Stores a point-in-time snapshot of layer content with change metadata,
    enabling persona versioning and rollback capabilities.
    """

    layer_id: str = ""            # Reference to ClaudeMdLayer
    version_number: int = 1       # Version number (increments per layer)
    content_snapshot: str = ""    # Full content at this version
    content_hash: str = ""        # Hash of content_snapshot

    # Change tracking
    change_rationale: str = ""    # Why this version was created
    changed_by: str = ""          # Agent/user who made change
    changed_sections: List[str] = field(default_factory=list)  # Section IDs changed

    # Diff info
    additions: int = 0            # Lines added
    deletions: int = 0            # Lines removed

    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set entity type and auto-generate ID if needed."""
        self.entity_type = "claudemd_version"

        # Auto-generate ID if not provided
        if not self.id and self.layer_id:
            self.id = f"CMV-{self.layer_id}-v{self.version_number}"

    def compute_content_hash(self) -> str:
        """Compute hash of snapshot content."""
        import hashlib
        return hashlib.sha256(self.content_snapshot.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize version to dictionary."""
        result = super().to_dict()
        result.update({
            "layer_id": self.layer_id,
            "version_number": self.version_number,
            "content_snapshot": self.content_snapshot,
            "content_hash": self.content_hash,
            "change_rationale": self.change_rationale,
            "changed_by": self.changed_by,
            "changed_sections": self.changed_sections,
            "additions": self.additions,
            "deletions": self.deletions,
            "properties": self.properties,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaudeMdVersion":
        """Deserialize version from dictionary."""
        return cls(
            id=data.get("id", ""),  # Allow empty for auto-generation
            entity_type=data.get("entity_type", "claudemd_version"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            layer_id=data.get("layer_id", ""),
            version_number=data.get("version_number", 1),
            content_snapshot=data.get("content_snapshot", ""),
            content_hash=data.get("content_hash", ""),
            change_rationale=data.get("change_rationale", ""),
            changed_by=data.get("changed_by", ""),
            changed_sections=data.get("changed_sections", []),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            properties=data.get("properties", {}),
        )


@dataclass
class Team(Entity):
    """
    Team or organizational unit for knowledge scoping.

    Enables hierarchical knowledge inheritance where child teams
    inherit layers and settings from parent teams.

    Example:
        team = Team(
            id="TEAM-eng-backend",
            name="Backend Engineering",
            parent_team_id="TEAM-engineering",
            branch_patterns=["feature/*", "dev"],
            module_scope=["query", "analysis", "persistence"]
        )
    """

    # Identity
    name: str = ""                    # "Backend Engineering"
    description: str = ""             # Team description

    # Hierarchy
    parent_team_id: str = ""          # Parent team for inheritance

    # Scope - what this team works on
    branch_patterns: List[str] = field(default_factory=list)
    # Git branch patterns this team owns, e.g., ["feature/*", "dev", "hotfix/*"]

    module_scope: List[str] = field(default_factory=list)
    # Modules this team is responsible for, e.g., ["query", "analysis"]

    # Team-specific layers
    layer_ids: List[str] = field(default_factory=list)
    # ClaudeMdLayer IDs specific to this team

    # Team members (persona profile IDs)
    member_profiles: List[str] = field(default_factory=list)

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)
    # Team-specific settings like default_freshness_days, required_sections, etc.

    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set entity type."""
        self.entity_type = "team"

    def is_in_scope(self, module: str) -> bool:
        """
        Check if a module is in this team's scope.

        Args:
            module: Module name to check

        Returns:
            True if module is in scope (or scope is empty = all modules)
        """
        if not self.module_scope:
            return True  # Empty scope means all modules
        return module in self.module_scope

    def matches_branch(self, branch: str) -> bool:
        """
        Check if a branch matches this team's patterns.

        Args:
            branch: Branch name to check

        Returns:
            True if branch matches any pattern
        """
        import fnmatch

        if not self.branch_patterns:
            return True  # Empty patterns means all branches

        for pattern in self.branch_patterns:
            if fnmatch.fnmatch(branch, pattern):
                return True
        return False

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a team setting with optional default.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize team to dictionary."""
        result = super().to_dict()
        result.update({
            "name": self.name,
            "description": self.description,
            "parent_team_id": self.parent_team_id,
            "branch_patterns": self.branch_patterns,
            "module_scope": self.module_scope,
            "layer_ids": self.layer_ids,
            "member_profiles": self.member_profiles,
            "settings": self.settings,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Team":
        """Deserialize team from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "team"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            name=data.get("name", ""),
            description=data.get("description", ""),
            parent_team_id=data.get("parent_team_id", ""),
            branch_patterns=data.get("branch_patterns", []),
            module_scope=data.get("module_scope", []),
            layer_ids=data.get("layer_ids", []),
            member_profiles=data.get("member_profiles", []),
            settings=data.get("settings", {}),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PersonaProfile(Entity):
    """
    Profile defining layer preferences for a specific role.

    Enables teams of personas with inherited knowledge and
    role-specific CLAUDE.md generation.

    Example:
        profile = PersonaProfile(
            id="PP-backend-dev",
            name="Backend Developer",
            role="developer",
            team_id="TEAM-engineering-backend",
            layer_preferences={"include_spark": False, "include_query": True},
            default_branch="dev"
        )
    """

    # Identity
    name: str = ""                    # "Senior Backend Developer"
    role: str = ""                    # "developer", "qa", "devops", "marketing", "manager"
    team_id: str = ""                 # Reference to Team entity

    # Layer preferences (section_id -> include boolean)
    layer_preferences: Dict[str, bool] = field(default_factory=dict)
    # e.g., {"spark-module": False, "query-module": True, "marketing": False}

    # Inheritance
    inherits_from: str = ""           # Parent profile ID for preference inheritance

    # Context defaults
    default_branch: str = ""          # Default branch context ("dev", "qa", "prod")
    default_modules: List[str] = field(default_factory=list)
    # Modules this persona typically works with

    # Customization
    custom_layers: List[str] = field(default_factory=list)
    # Additional layer IDs always included for this persona

    excluded_layers: List[str] = field(default_factory=list)
    # Layer IDs always excluded for this persona

    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate persona profile fields."""
        self.entity_type = "persona_profile"

        # Validate role if provided
        valid_roles = {"developer", "qa", "devops", "marketing", "manager", "analyst", "designer", ""}
        if self.role and self.role not in valid_roles:
            raise ValidationError(
                f"Invalid role '{self.role}'",
                valid_roles=list(valid_roles - {""})
            )

    def should_include_layer(self, section_id: str) -> bool:
        """
        Check if a layer should be included for this persona.

        Args:
            section_id: The section ID to check

        Returns:
            True if layer should be included, False otherwise
        """
        # Check explicit exclusions first
        if section_id in self.excluded_layers:
            return False

        # Check explicit inclusions
        if section_id in self.custom_layers:
            return True

        # Check preferences
        if section_id in self.layer_preferences:
            return self.layer_preferences[section_id]

        # Default to include
        return True

    def get_effective_preferences(self, parent_profile: "PersonaProfile" = None) -> Dict[str, bool]:
        """
        Get effective layer preferences with inheritance.

        Args:
            parent_profile: Parent profile to inherit from

        Returns:
            Merged preferences dict
        """
        if parent_profile is None:
            return self.layer_preferences.copy()

        # Start with parent preferences
        effective = parent_profile.layer_preferences.copy()

        # Override with this profile's preferences
        effective.update(self.layer_preferences)

        return effective

    def to_dict(self) -> Dict[str, Any]:
        """Serialize persona profile to dictionary."""
        result = super().to_dict()
        result.update({
            "name": self.name,
            "role": self.role,
            "team_id": self.team_id,
            "layer_preferences": self.layer_preferences,
            "inherits_from": self.inherits_from,
            "default_branch": self.default_branch,
            "default_modules": self.default_modules,
            "custom_layers": self.custom_layers,
            "excluded_layers": self.excluded_layers,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaProfile":
        """Deserialize persona profile from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "persona_profile"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            name=data.get("name", ""),
            role=data.get("role", ""),
            team_id=data.get("team_id", ""),
            layer_preferences=data.get("layer_preferences", {}),
            inherits_from=data.get("inherits_from", ""),
            default_branch=data.get("default_branch", ""),
            default_modules=data.get("default_modules", []),
            custom_layers=data.get("custom_layers", []),
            excluded_layers=data.get("excluded_layers", []),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document(Entity):
    """
    Document entity representing a tracked documentation file.

    Documents can be linked to tasks, decisions, and sprints via edges,
    enabling traceability between work items and their documentation.

    Edge types for documents:
        - DOCUMENTED_BY: Task/Decision → Document (task is documented by doc)
        - PRODUCES: Task → Document (task produces/creates doc)
        - REFERENCES: Any → Document (entity references doc)

    Example:
        doc = Document(
            id="DOC-docs-architecture-md",
            path="docs/architecture.md",
            title="Architecture Overview",
            doc_type="architecture",
            tags=["core", "design"]
        )
    """

    # Core fields
    path: str = ""                    # Relative path from repo root
    title: str = ""                   # Human-readable title
    doc_type: str = "general"         # architecture, design, memory, decision, api, guide

    # Content tracking
    content_hash: str = ""            # SHA256 hash of content (first 16 chars)
    line_count: int = 0               # Number of lines
    word_count: int = 0               # Approximate word count

    # Staleness detection
    last_file_modified: str = ""      # ISO timestamp of file mtime
    last_verified: str = ""           # When we last checked the file
    is_stale: bool = False            # True if file changed since last_verified

    # Organization
    tags: List[str] = field(default_factory=list)
    category: str = ""                # Parent category (e.g., "api", "guides")

    # Linkage tracking (cached for quick access)
    linked_task_ids: List[str] = field(default_factory=list)
    linked_decision_ids: List[str] = field(default_factory=list)

    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate document fields after initialization."""
        self.entity_type = "document"

        # Validate doc_type
        valid_doc_types = {
            "general", "architecture", "design", "memory", "decision",
            "api", "guide", "research", "knowledge-transfer", ""
        }
        if self.doc_type and self.doc_type not in valid_doc_types:
            raise ValidationError(
                f"Invalid doc_type '{self.doc_type}'",
                valid_types=list(valid_doc_types - {""})
            )

    def compute_content_hash(self, content: str) -> str:
        """Compute hash of document content."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def update_from_file(self, content: str, file_mtime: str) -> bool:
        """
        Update document tracking from file content.

        Args:
            content: File content
            file_mtime: File modification time (ISO format)

        Returns:
            True if content changed, False otherwise
        """
        new_hash = self.compute_content_hash(content)
        changed = new_hash != self.content_hash

        self.content_hash = new_hash
        self.last_file_modified = file_mtime
        self.last_verified = datetime.now(timezone.utc).isoformat()
        self.line_count = content.count('\n') + 1
        self.word_count = len(content.split())
        self.is_stale = False

        if changed:
            self.bump_version()

        return changed

    def mark_stale(self) -> None:
        """Mark document as needing re-verification."""
        self.is_stale = True
        self.bump_version()

    @classmethod
    def id_from_path(cls, path: str) -> str:
        """
        Generate document ID from file path.

        Args:
            path: File path (e.g., "docs/architecture.md")

        Returns:
            Document ID (e.g., "DOC-docs-architecture-md")
        """
        # Normalize path and convert to ID-safe format
        safe_path = path.replace("/", "-").replace(".", "-").replace("_", "-")
        return f"DOC-{safe_path}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to dictionary."""
        result = super().to_dict()
        result.update({
            "path": self.path,
            "title": self.title,
            "doc_type": self.doc_type,
            "content_hash": self.content_hash,
            "line_count": self.line_count,
            "word_count": self.word_count,
            "last_file_modified": self.last_file_modified,
            "last_verified": self.last_verified,
            "is_stale": self.is_stale,
            "tags": self.tags,
            "category": self.category,
            "linked_task_ids": self.linked_task_ids,
            "linked_decision_ids": self.linked_decision_ids,
            "properties": self.properties,
            "metadata": self.metadata,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Deserialize document from dictionary."""
        return cls(
            id=data["id"],
            entity_type=data.get("entity_type", "document"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            modified_at=data.get("modified_at", datetime.now(timezone.utc).isoformat()),
            path=data.get("path", ""),
            title=data.get("title", ""),
            doc_type=data.get("doc_type", "general"),
            content_hash=data.get("content_hash", ""),
            line_count=data.get("line_count", 0),
            word_count=data.get("word_count", 0),
            last_file_modified=data.get("last_file_modified", ""),
            last_verified=data.get("last_verified", ""),
            is_stale=data.get("is_stale", False),
            tags=data.get("tags", []),
            category=data.get("category", ""),
            linked_task_ids=data.get("linked_task_ids", []),
            linked_decision_ids=data.get("linked_decision_ids", []),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )
