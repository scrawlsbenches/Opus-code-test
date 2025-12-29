"""
Temporal reference types for the Cognitive Event Lattice.

References enable self-reference without paradox by always specifying
WHEN something is referenced, not just WHAT.

Key Insight:
    "The system" doesn't exist as a static thing.
    Only "the system at event E" is a concrete, stable reference.

Reference Types:
    MerkleRoot: Content hash of an event (immutable identifier)
    EventHorizon: A point in the event DAG (for temporal queries)
    TemporalReference: Reference to entity at specific horizon
    DeferredReference: Reference resolved at execution time
    CausalLink: Edge in the causal DAG

The Double Helix in References:
    WISDOM: TemporalReference (what we know at a point)
    SANITY: DeferredReference (handling unknown future state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from .protocols import CognitiveLattice


T = TypeVar('T')


class ReferenceMode(Enum):
    """How a reference should be resolved."""

    SNAPSHOT = auto()    # Fixed at creation time
    DEFERRED = auto()    # Resolved at execution time
    FLOATING = auto()    # Always resolves to latest
    VERSIONED = auto()   # Specific entity version


@dataclass(frozen=True)
class MerkleRoot:
    """
    Content-addressed identifier for an event.

    A MerkleRoot is the SHA256 hash of an event's content.
    It is immutable and globally unique for that content.

    Properties:
        - Deterministic: Same content = same root
        - Verifiable: Can recompute to verify integrity
        - Immutable: Content cannot change without changing root

    Used for:
        - Event IDs
        - Causal parent references
        - Integrity verification
    """

    value: str

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"MerkleRoot({self.value[:16]}...)"

    @property
    def short(self) -> str:
        """Short form for display (first 8 chars)."""
        return self.value[:8]

    def matches(self, other: 'MerkleRoot') -> bool:
        """Check if roots match."""
        return self.value == other.value


@dataclass(frozen=True)
class EventHorizon:
    """
    A point in the event DAG representing "state as of here".

    The EventHorizon is used for temporal queries - asking
    "what was the state at this point in the event history?"

    Unlike MerkleRoot (which identifies a single event), an
    EventHorizon represents the accumulated state up to and
    including that event.

    Properties:
        event_id: The event marking this horizon
        is_head: Whether this is a current branch head

    Example:
        horizon = lattice.current_horizon
        old_task = lattice.materialize("T-xxx", at=horizon)
        # Even after more events, old_task stays the same
    """

    event_id: str
    is_head: bool = False

    def __str__(self) -> str:
        head_marker = " (HEAD)" if self.is_head else ""
        return f"@{self.event_id[:8]}{head_marker}"

    def __repr__(self) -> str:
        return f"EventHorizon({self.event_id[:16]}..., head={self.is_head})"

    @property
    def merkle_root(self) -> MerkleRoot:
        """Get as MerkleRoot."""
        return MerkleRoot(self.event_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'event_id': self.event_id,
            'is_head': self.is_head,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventHorizon':
        """Deserialize from storage."""
        return cls(
            event_id=data['event_id'],
            is_head=data.get('is_head', False),
        )


@dataclass(frozen=True)
class TemporalReference(Generic[T]):
    """
    Reference to an entity at a specific point in time.

    This is the key to self-reference without paradox:
    instead of referencing "the entity", we reference
    "the entity as it was at event E".

    The entity may have changed since then, but this
    reference always resolves to the same state.

    Attributes:
        entity_id: The entity being referenced
        horizon: The event horizon (point in time)
        entity_type: Type hint for the entity

    Example:
        # Create task that references current storage format
        task = Intention(
            title="Optimize storage",
            references=[
                TemporalReference(
                    entity_id="storage_format",
                    horizon=lattice.current_horizon,
                )
            ],
        )
        # Later, even after storage changes:
        old_format = task.references[0].resolve(lattice)
        # Returns format as it was when task was created
    """

    entity_id: str
    horizon: EventHorizon
    entity_type: Optional[str] = None

    def resolve(self, lattice: 'CognitiveLattice') -> Optional[T]:
        """
        Resolve the reference to an entity.

        Args:
            lattice: The cognitive lattice to resolve against

        Returns:
            The entity as it existed at the horizon, or None
        """
        return lattice.materializer.materialize(
            self.entity_id,
            at=self.horizon,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'entity_id': self.entity_id,
            'horizon': self.horizon.to_dict(),
            'entity_type': self.entity_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalReference':
        """Deserialize from storage."""
        return cls(
            entity_id=data['entity_id'],
            horizon=EventHorizon.from_dict(data['horizon']),
            entity_type=data.get('entity_type'),
        )


@dataclass
class DeferredReference(Generic[T]):
    """
    Reference that is resolved at execution time, not creation time.

    Used when a task depends on another task completing first.
    The reference will be resolved AFTER dependencies complete,
    capturing the state at that moment.

    Lifecycle:
        1. Created with mode=DEFERRED, horizon=None
        2. Dependencies complete
        3. resolve_now() called, captures horizon
        4. Now behaves like TemporalReference

    Attributes:
        entity_id: The entity to reference
        after: Events/intentions that must complete first
        mode: Current resolution mode
        resolved_horizon: Captured horizon (after resolution)

    Example:
        # Create task that references storage AFTER previous task completes
        ref = DeferredReference(
            entity_id="storage_format",
            after=["previous_task_intention_id"],
        )
        # Later, when previous_task completes:
        ref.resolve_now(lattice)
        # Now ref.resolved_horizon captures the post-task state
    """

    entity_id: str
    after: list[str] = field(default_factory=list)
    mode: ReferenceMode = ReferenceMode.DEFERRED
    resolved_horizon: Optional[EventHorizon] = None
    entity_type: Optional[str] = None

    def is_resolved(self) -> bool:
        """Check if this reference has been resolved."""
        return self.resolved_horizon is not None

    def resolve_now(self, lattice: 'CognitiveLattice') -> None:
        """
        Capture the current horizon as the resolution point.

        This should be called when all dependencies in `after` have completed.

        Args:
            lattice: The cognitive lattice

        Raises:
            ValueError: If already resolved
        """
        if self.is_resolved():
            raise ValueError("Reference already resolved")

        self.resolved_horizon = lattice.current_horizon
        self.mode = ReferenceMode.SNAPSHOT

    def resolve(self, lattice: 'CognitiveLattice') -> Optional[T]:
        """
        Resolve the reference to an entity.

        Args:
            lattice: The cognitive lattice

        Returns:
            The entity at the resolved horizon

        Raises:
            ValueError: If not yet resolved
        """
        if not self.is_resolved():
            raise ValueError(
                "Deferred reference not yet resolved. "
                "Call resolve_now() after dependencies complete."
            )

        return lattice.materializer.materialize(
            self.entity_id,
            at=self.resolved_horizon,
        )

    def to_temporal(self) -> TemporalReference[T]:
        """
        Convert to a TemporalReference after resolution.

        Returns:
            Equivalent TemporalReference

        Raises:
            ValueError: If not yet resolved
        """
        if not self.is_resolved():
            raise ValueError("Cannot convert unresolved deferred reference")

        return TemporalReference(
            entity_id=self.entity_id,
            horizon=self.resolved_horizon,
            entity_type=self.entity_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'entity_id': self.entity_id,
            'after': self.after,
            'mode': self.mode.name,
            'resolved_horizon': (
                self.resolved_horizon.to_dict()
                if self.resolved_horizon else None
            ),
            'entity_type': self.entity_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeferredReference':
        """Deserialize from storage."""
        resolved = data.get('resolved_horizon')
        return cls(
            entity_id=data['entity_id'],
            after=data.get('after', []),
            mode=ReferenceMode[data.get('mode', 'DEFERRED')],
            resolved_horizon=(
                EventHorizon.from_dict(resolved) if resolved else None
            ),
            entity_type=data.get('entity_type'),
        )


@dataclass(frozen=True)
class CausalLink:
    """
    A directed edge in the causal DAG.

    CausalLinks connect events in cause-effect relationships.
    They form the structure of the event DAG.

    Types of causal relationships:
        PARENT: Direct causal dependency (event B caused by A)
        MERGE: Combining multiple branches (multiple parents)
        SUPERSEDES: One event replaces another (compaction)

    Attributes:
        from_event: Source event (cause)
        to_event: Target event (effect)
        link_type: Type of causal relationship
    """

    from_event: str  # MerkleRoot as string
    to_event: str    # MerkleRoot as string
    link_type: str = "PARENT"

    def __str__(self) -> str:
        return f"{self.from_event[:8]} -{self.link_type}-> {self.to_event[:8]}"

    @property
    def from_root(self) -> MerkleRoot:
        return MerkleRoot(self.from_event)

    @property
    def to_root(self) -> MerkleRoot:
        return MerkleRoot(self.to_event)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'from_event': self.from_event,
            'to_event': self.to_event,
            'link_type': self.link_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalLink':
        """Deserialize from storage."""
        return cls(
            from_event=data['from_event'],
            to_event=data['to_event'],
            link_type=data.get('link_type', 'PARENT'),
        )


@dataclass
class ReferenceSet:
    """
    Collection of references with resolution tracking.

    Used when an intention references multiple entities,
    some with immediate snapshots and some deferred.

    Provides bulk operations and resolution status tracking.
    """

    temporal: list[TemporalReference] = field(default_factory=list)
    deferred: list[DeferredReference] = field(default_factory=list)

    def add_snapshot(
        self,
        entity_id: str,
        horizon: EventHorizon,
        entity_type: Optional[str] = None,
    ) -> None:
        """Add a snapshot reference (resolved now)."""
        self.temporal.append(TemporalReference(
            entity_id=entity_id,
            horizon=horizon,
            entity_type=entity_type,
        ))

    def add_deferred(
        self,
        entity_id: str,
        after: list[str],
        entity_type: Optional[str] = None,
    ) -> None:
        """Add a deferred reference (resolved later)."""
        self.deferred.append(DeferredReference(
            entity_id=entity_id,
            after=after,
            entity_type=entity_type,
        ))

    def all_resolved(self) -> bool:
        """Check if all deferred references are resolved."""
        return all(ref.is_resolved() for ref in self.deferred)

    def pending_dependencies(self) -> set[str]:
        """Get all pending dependency IDs."""
        deps = set()
        for ref in self.deferred:
            if not ref.is_resolved():
                deps.update(ref.after)
        return deps

    def resolve_all(self, lattice: 'CognitiveLattice') -> list:
        """
        Resolve all references and return entities.

        Raises:
            ValueError: If any deferred reference is not yet resolved
        """
        entities = []

        for ref in self.temporal:
            entity = ref.resolve(lattice)
            if entity is not None:
                entities.append(entity)

        for ref in self.deferred:
            entity = ref.resolve(lattice)
            if entity is not None:
                entities.append(entity)

        return entities

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'temporal': [r.to_dict() for r in self.temporal],
            'deferred': [r.to_dict() for r in self.deferred],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceSet':
        """Deserialize from storage."""
        return cls(
            temporal=[
                TemporalReference.from_dict(r)
                for r in data.get('temporal', [])
            ],
            deferred=[
                DeferredReference.from_dict(r)
                for r in data.get('deferred', [])
            ],
        )
