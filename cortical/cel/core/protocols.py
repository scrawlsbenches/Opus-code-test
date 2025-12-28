"""
Protocol definitions for the Cognitive Event Lattice.

These protocols define the contracts that implementations must fulfill.
Using Protocol (structural subtyping) allows for maximum flexibility -
any class with the right methods is compatible, enabling easy mocking,
testing, and future backend swaps without code changes.

Design Philosophy:
    "Program to an interface, not an implementation."

    Every component in CEL depends only on these protocols, never on
    concrete implementations. This enables:
    - Swapping storage backends (file, memory, S3, SQLite)
    - Testing with mocks
    - Gradual migration to new implementations
    - Plugin architectures

The Double Helix in Protocols:
    WISDOM protocols: EventStore, Materializer, SemanticIndex, EventReducer
    SANITY protocols: HealthMonitor, MigrationEngine, CompactionStrategy
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from .events import CognitiveEvent, EventType
from .references import EventHorizon, MerkleRoot, TemporalReference


# Type variables for generic protocols
T = TypeVar('T')
E = TypeVar('E', bound='CognitiveEvent')


# =============================================================================
# WISDOM STRAND PROTOCOLS
# =============================================================================


@runtime_checkable
class EventStore(Protocol):
    """
    Protocol for append-only event storage.

    The EventStore is the source of truth - immutable, append-only,
    content-addressed. All other state is derived from events.

    Implementations:
        - FileSystemEventStore: JSONL files in .got/dag/
        - MemoryEventStore: In-memory for testing
        - SQLiteEventStore: Future - for larger deployments
        - S3EventStore: Future - for distributed systems

    Invariants:
        - Events are immutable once stored
        - Event IDs are content hashes (deterministic)
        - Causal order is preserved (parents before children)
        - append() is the only write operation
    """

    @abstractmethod
    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """
        Append an event to the store.

        Args:
            event: The event to append (must have valid causal_parents)

        Returns:
            MerkleRoot of the event (content hash)

        Raises:
            CausalViolationError: If causal parents don't exist
            DuplicateEventError: If event already exists (idempotent)
        """
        ...

    @abstractmethod
    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """
        Retrieve an event by ID (content hash).

        Args:
            event_id: The Merkle root / content hash

        Returns:
            The event, or None if not found
        """
        ...

    @abstractmethod
    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
        event_types: Optional[Sequence[EventType]] = None,
    ) -> Iterator[CognitiveEvent]:
        """
        Iterate events in causal order.

        Args:
            from_event: Start after this event (exclusive)
            to_event: Stop at this event (inclusive)
            event_types: Filter by event types

        Yields:
            Events in causal order (parents before children)
        """
        ...

    @abstractmethod
    def heads(self) -> List[MerkleRoot]:
        """
        Get current branch heads (events with no children).

        Returns:
            List of MerkleRoots representing branch heads
        """
        ...

    @abstractmethod
    def latest(self) -> Optional[MerkleRoot]:
        """
        Get the latest event's Merkle root.

        For single-branch operation, this is the current horizon.
        For multi-branch, returns the "main" branch head.

        Returns:
            MerkleRoot of latest event, or None if empty
        """
        ...

    @abstractmethod
    def ancestors(self, event_id: str, depth: int = -1) -> Iterator[CognitiveEvent]:
        """
        Iterate ancestors of an event (causal history).

        Args:
            event_id: Starting event
            depth: Maximum depth (-1 for unlimited)

        Yields:
            Ancestor events in reverse causal order
        """
        ...

    @abstractmethod
    def descendants(self, event_id: str) -> Iterator[CognitiveEvent]:
        """
        Iterate descendants of an event (causal future).

        Args:
            event_id: Starting event

        Yields:
            Descendant events in causal order
        """
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """Total number of events in the store."""
        ...


@runtime_checkable
class Materializer(Protocol[T]):
    """
    Protocol for materializing entities from events.

    The Materializer is responsible for "folding" events into entity state.
    It maintains the current view of entities while preserving the ability
    to reconstruct past states.

    Key Insight:
        Entities don't exist in the EventStore - only events do.
        Entities are materialized (computed) from events on demand.
        This enables temporal queries: "What was entity X at time T?"

    Implementations:
        - CachingMaterializer: Maintains hot cache of recent entities
        - LazyMaterializer: Computes on demand, no caching
        - SnapshottingMaterializer: Periodic snapshots for faster startup
    """

    @abstractmethod
    def materialize(
        self,
        entity_id: str,
        at: Optional[EventHorizon] = None,
    ) -> Optional[T]:
        """
        Materialize an entity at a specific point in time.

        Args:
            entity_id: The entity to materialize
            at: Event horizon (None = current state)

        Returns:
            The materialized entity, or None if it doesn't exist at that point
        """
        ...

    @abstractmethod
    def materialize_many(
        self,
        entity_ids: Sequence[str],
        at: Optional[EventHorizon] = None,
    ) -> Dict[str, T]:
        """
        Batch materialize multiple entities.

        Args:
            entity_ids: Entities to materialize
            at: Event horizon (None = current state)

        Returns:
            Dict mapping entity_id to materialized entity (missing = not found)
        """
        ...

    @abstractmethod
    def invalidate(self, entity_id: str) -> None:
        """
        Invalidate cached materialization for an entity.

        Args:
            entity_id: Entity to invalidate
        """
        ...

    @abstractmethod
    def invalidate_all(self) -> None:
        """Invalidate all cached materializations."""
        ...

    @abstractmethod
    def register_reducer(
        self,
        entity_type: str,
        reducer: 'EventReducer[T]',
    ) -> None:
        """
        Register a reducer for an entity type.

        Args:
            entity_type: The type of entity (e.g., 'task', 'decision')
            reducer: Function that folds events into entity state
        """
        ...


class EventReducer(Protocol[T]):
    """
    Protocol for reducing events into entity state.

    An EventReducer is a pure function that takes an optional previous state
    and an event, and returns the new state. This is similar to Redux reducers
    or fold/reduce in functional programming.

    Example:
        def task_reducer(state: Optional[Task], event: CognitiveEvent) -> Task:
            if event.event_type == EventType.INTENTION:
                return Task.from_intention(event)
            elif event.event_type == EventType.FULFILLMENT:
                return state._replace(status='completed')
            return state
    """

    @abstractmethod
    def __call__(
        self,
        state: Optional[T],
        event: CognitiveEvent,
    ) -> Optional[T]:
        """
        Apply an event to the current state.

        Args:
            state: Current entity state (None if entity doesn't exist yet)
            event: Event to apply

        Returns:
            New entity state (None if entity was deleted)
        """
        ...

    @property
    @abstractmethod
    def entity_type(self) -> str:
        """The entity type this reducer handles."""
        ...


@runtime_checkable
class SemanticIndex(Protocol):
    """
    Protocol for semantic/probabilistic indexing.

    The SemanticIndex provides fast, approximate answers to semantic queries:
    - "Does concept X exist?" (bloom filter)
    - "What's similar to X?" (embeddings)
    - "Find documents containing X" (inverted index)

    Design Trade-off:
        These structures trade exactness for speed. The bloom filter
        may have false positives (but never false negatives). Embeddings
        are approximate by nature. The inverted index is exact but may
        be stale if not updated.

    Implementations:
        - BloomSemanticIndex: Fast existence checks
        - EmbeddingSemanticIndex: Vector similarity
        - HybridSemanticIndex: Combines both
    """

    @abstractmethod
    def index_event(self, event: CognitiveEvent) -> None:
        """
        Update indexes for a new event.

        Args:
            event: Event to index
        """
        ...

    @abstractmethod
    def probably_contains(self, concept: str) -> bool:
        """
        Fast probabilistic check for concept existence.

        Args:
            concept: Concept to check

        Returns:
            True if concept probably exists (may be false positive)
            False if concept definitely doesn't exist
        """
        ...

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for events/entities matching a query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching event/entity IDs
        """
        ...

    @abstractmethod
    def similar_to(self, entity_id: str, limit: int = 10) -> List[tuple[str, float]]:
        """
        Find entities similar to a given entity.

        Args:
            entity_id: Reference entity
            limit: Maximum results

        Returns:
            List of (entity_id, similarity_score) tuples
        """
        ...

    @abstractmethod
    def rebuild(self, event_store: EventStore) -> None:
        """
        Rebuild indexes from scratch using event store.

        Args:
            event_store: Source of truth for rebuilding
        """
        ...


# =============================================================================
# SANITY STRAND PROTOCOLS
# =============================================================================


@dataclass
class HealthMetrics:
    """Snapshot of system health metrics."""

    timestamp: datetime
    event_count: int
    entity_count: int
    cache_hit_rate: float
    bloom_false_positive_rate: float
    stale_references_count: int
    orphaned_events_count: int
    storage_bytes: int
    compaction_benefit_estimate: float  # 0.0 to 1.0

    def needs_compaction(self, threshold: float = 0.3) -> bool:
        """Check if compaction is recommended."""
        return self.compaction_benefit_estimate >= threshold

    def is_healthy(self) -> bool:
        """Overall health check."""
        return (
            self.stale_references_count == 0 and
            self.orphaned_events_count == 0 and
            self.bloom_false_positive_rate < 0.1
        )


@runtime_checkable
class HealthMonitor(Protocol):
    """
    Protocol for system health monitoring.

    The HealthMonitor watches the system and can trigger self-maintenance.
    This is the self-awareness capability - the system knowing when
    it needs healing.

    Implementations:
        - PassiveHealthMonitor: Reports metrics, no auto-action
        - ActiveHealthMonitor: Can create maintenance tasks
        - ScheduledHealthMonitor: Periodic health checks
    """

    @abstractmethod
    def check(self) -> HealthMetrics:
        """
        Perform health check and return metrics.

        Returns:
            Current health metrics
        """
        ...

    @abstractmethod
    def subscribe(
        self,
        event: str,
        callback: Callable[[HealthMetrics], None],
    ) -> Callable[[], None]:
        """
        Subscribe to health events.

        Args:
            event: Event type ('compaction_needed', 'unhealthy', etc.)
            callback: Function to call when event occurs

        Returns:
            Unsubscribe function
        """
        ...

    @abstractmethod
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a custom metric.

        Args:
            name: Metric name
            value: Metric value
        """
        ...

    @abstractmethod
    def get_history(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[HealthMetrics]:
        """
        Get historical health metrics.

        Args:
            since: Start time (None = beginning)
            limit: Maximum records

        Returns:
            List of historical metrics, newest first
        """
        ...


@dataclass
class MigrationPlan:
    """Plan for a data migration."""

    id: str
    name: str
    description: str
    from_version: str
    to_version: str
    steps: List['MigrationStep']
    reversible: bool
    estimated_duration_seconds: float

    def is_compatible_with(self, current_version: str) -> bool:
        """Check if this migration can be applied."""
        return current_version == self.from_version


@dataclass
class MigrationStep:
    """Single step in a migration."""

    name: str
    description: str
    transform: Callable[[CognitiveEvent], CognitiveEvent]
    rollback: Optional[Callable[[CognitiveEvent], CognitiveEvent]]


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    from_version: str
    to_version: str
    events_migrated: int
    events_failed: int
    duration_seconds: float
    errors: List[str]


@runtime_checkable
class MigrationEngine(Protocol):
    """
    Protocol for data migration.

    The MigrationEngine handles schema evolution without data loss.
    Because events are immutable, migration works by:
    1. Reading old events
    2. Transforming to new schema
    3. Writing new events with migration metadata

    The old events remain (for audit), but are marked as superseded.

    Key Principle:
        Migrations transform INTERPRETATIONS, not raw events.
        The event "user clicked button" doesn't change meaning,
        but how we structure the data around it might.

    Implementations:
        - InPlaceMigrationEngine: Transforms during read
        - CopyMigrationEngine: Creates new event stream
        - LazyMigrationEngine: Migrates on first access
    """

    @abstractmethod
    def current_version(self) -> str:
        """Get current schema version."""
        ...

    @abstractmethod
    def available_migrations(self) -> List[MigrationPlan]:
        """List available migrations from current version."""
        ...

    @abstractmethod
    def plan_migration(self, to_version: str) -> Optional[MigrationPlan]:
        """
        Plan migration to a specific version.

        Args:
            to_version: Target version

        Returns:
            Migration plan, or None if no path exists
        """
        ...

    @abstractmethod
    def execute(self, plan: MigrationPlan) -> MigrationResult:
        """
        Execute a migration plan.

        Args:
            plan: The migration plan to execute

        Returns:
            Result of the migration
        """
        ...

    @abstractmethod
    def rollback(self, plan: MigrationPlan) -> MigrationResult:
        """
        Rollback a migration.

        Args:
            plan: The migration to rollback

        Returns:
            Result of the rollback

        Raises:
            IrreversibleMigrationError: If migration cannot be rolled back
        """
        ...

    @abstractmethod
    def register_migration(self, plan: MigrationPlan) -> None:
        """
        Register a new migration plan.

        Args:
            plan: Migration plan to register
        """
        ...


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    success: bool
    events_before: int
    events_after: int
    bytes_before: int
    bytes_after: int
    duration_seconds: float
    preserved_merkle_root: MerkleRoot  # Root is preserved for verification


@runtime_checkable
class CompactionStrategy(Protocol):
    """
    Protocol for event compaction.

    Compaction reduces storage by combining events while preserving
    the same materialized state. This is semantic compression:
    we care about MEANING, not raw event count.

    Example:
        Events: [set x=1, set x=2, set x=3]
        Compacted: [set x=3]  # Same final state

    Key Invariant:
        materialize(events) == materialize(compact(events))

    Implementations:
        - LastWriteWinsCompaction: Keep latest for each key
        - SnapshotCompaction: Replace history with snapshot
        - SemanticCompaction: Preserve semantically significant events
    """

    @abstractmethod
    def should_compact(self, metrics: HealthMetrics) -> bool:
        """
        Decide if compaction should run.

        Args:
            metrics: Current health metrics

        Returns:
            True if compaction is recommended
        """
        ...

    @abstractmethod
    def compact(
        self,
        events: Iterator[CognitiveEvent],
        preserve_after: Optional[datetime] = None,
    ) -> Iterator[CognitiveEvent]:
        """
        Compact a sequence of events.

        Args:
            events: Events to compact (in causal order)
            preserve_after: Don't compact events after this time

        Yields:
            Compacted events (subset of input)
        """
        ...

    @abstractmethod
    def estimate_benefit(self, event_store: EventStore) -> float:
        """
        Estimate compaction benefit (0.0 to 1.0).

        Args:
            event_store: Store to analyze

        Returns:
            Estimated reduction ratio (0.3 = 30% smaller after compaction)
        """
        ...


# =============================================================================
# COMPOSITE PROTOCOLS
# =============================================================================


@runtime_checkable
class CognitiveLattice(Protocol):
    """
    The complete Cognitive Event Lattice interface.

    This combines both strands (Wisdom + Sanity) into a unified API.
    The lattice is the primary user-facing interface.
    """

    # Wisdom strand
    @property
    @abstractmethod
    def events(self) -> EventStore:
        """Access to event store."""
        ...

    @property
    @abstractmethod
    def materializer(self) -> Materializer:
        """Access to materializer."""
        ...

    @property
    @abstractmethod
    def semantic(self) -> SemanticIndex:
        """Access to semantic index."""
        ...

    # Sanity strand
    @property
    @abstractmethod
    def health(self) -> HealthMonitor:
        """Access to health monitor."""
        ...

    @property
    @abstractmethod
    def migration(self) -> MigrationEngine:
        """Access to migration engine."""
        ...

    @property
    @abstractmethod
    def compaction(self) -> CompactionStrategy:
        """Access to compaction strategy."""
        ...

    # Unified operations
    @abstractmethod
    def intend(
        self,
        title: str,
        **kwargs,
    ) -> 'Intention':
        """
        Create an intention (task) with temporal reference.

        Args:
            title: Intention title
            **kwargs: Additional fields (references_system_at, after, etc.)

        Returns:
            Created Intention
        """
        ...

    @abstractmethod
    def observe(self, content: Dict[str, Any]) -> 'Observation':
        """
        Record an observation.

        Args:
            content: Observation content

        Returns:
            Created Observation
        """
        ...

    @abstractmethod
    def fulfill(self, intention_id: str, result: Dict[str, Any]) -> 'Fulfillment':
        """
        Mark an intention as fulfilled.

        Args:
            intention_id: Intention to fulfill
            result: Result data

        Returns:
            Created Fulfillment event
        """
        ...

    @property
    @abstractmethod
    def current_horizon(self) -> EventHorizon:
        """Get current event horizon (latest Merkle root)."""
        ...

    @abstractmethod
    def at(self, horizon: EventHorizon) -> 'CognitiveLattice':
        """
        Create a view of the lattice at a specific horizon.

        Args:
            horizon: Event horizon to view at

        Returns:
            Lattice view frozen at that horizon
        """
        ...
