"""
Schema migration for the Cognitive Event Lattice.

The migration system enables the lattice to evolve its schema
over time without losing historical data. This is essential
for long-lived systems where requirements change.

Key Insight:
    Because events are immutable, we don't modify old events.
    Instead, migrations work by:
    1. Reading old events through adapters
    2. Writing new events in the new format
    3. Keeping old events for audit/history

Migration Strategies:
    - Lazy: Transform on read (adapters)
    - Eager: Transform all at once (batch migration)
    - Hybrid: Lazy with background eager migration

Design Pattern:
    Migrations are themselves events. When a migration runs,
    it creates Compaction events that link old events to new
    representations. This maintains the causal chain.

This module implements Level 5 of the CEL architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ..core.events import CognitiveEvent, Compaction, EventType
from ..core.protocols import EventStore, MigrationEngine
from ..core.references import MerkleRoot


class MigrationStatus(Enum):
    """Status of a migration."""

    PENDING = auto()      # Not yet started
    IN_PROGRESS = auto()  # Currently running
    COMPLETED = auto()    # Successfully finished
    FAILED = auto()       # Failed with error
    ROLLED_BACK = auto()  # Reverted


@dataclass
class MigrationStep:
    """
    A single step in a migration plan.

    Each step transforms events of a specific type or pattern.
    Steps can be applied lazily (on read) or eagerly (batch).

    Attributes:
        name: Human-readable step name
        description: What this step does
        event_filter: Predicate to select events to transform
        transform: Function to transform event content
        version_from: Source schema version
        version_to: Target schema version
    """

    name: str
    description: str
    event_filter: Callable[[CognitiveEvent], bool]
    transform: Callable[[Dict[str, Any]], Dict[str, Any]]
    version_from: str
    version_to: str

    def applies_to(self, event: CognitiveEvent) -> bool:
        """Check if this step applies to an event."""
        return self.event_filter(event)

    def apply(self, event: CognitiveEvent) -> CognitiveEvent:
        """
        Apply transformation to an event.

        Note: This creates a new event with transformed content.
        The original event is preserved.
        """
        new_content = self.transform(event.content.copy())

        # Add migration metadata
        new_content['_migrated_from'] = event.id
        new_content['_migration_step'] = self.name
        new_content['_schema_version'] = self.version_to

        # Create new event with same type but new content
        # Note: This will have a different Merkle root
        return CognitiveEvent(
            timestamp=event.timestamp,
            event_type=event.event_type,
            causal_parents=event.causal_parents + (event.id,),
            content=new_content,
            concepts=event.concepts,
        )


@dataclass
class MigrationPlan:
    """
    A complete migration plan with ordered steps.

    Plans are versioned and can be applied incrementally.
    Each plan tracks its own progress.

    Attributes:
        name: Plan identifier
        description: What this migration accomplishes
        steps: Ordered list of migration steps
        created_at: When the plan was created
        dry_run: If True, don't persist changes
    """

    name: str
    description: str
    steps: List[MigrationStep]
    created_at: datetime = field(default_factory=datetime.now)
    dry_run: bool = False

    # Progress tracking
    _status: MigrationStatus = MigrationStatus.PENDING
    _current_step: int = 0
    _events_processed: int = 0
    _errors: List[str] = field(default_factory=list)

    @property
    def status(self) -> MigrationStatus:
        return self._status

    @property
    def progress(self) -> float:
        """Return progress as percentage (0-100)."""
        if not self.steps:
            return 100.0
        return (self._current_step / len(self.steps)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan metadata."""
        return {
            'name': self.name,
            'description': self.description,
            'steps': [s.name for s in self.steps],
            'created_at': self.created_at.isoformat(),
            'status': self._status.name,
            'progress': self.progress,
            'events_processed': self._events_processed,
            'errors': self._errors,
        }


class SchemaMigrationEngine:
    """
    Engine for executing schema migrations.

    Implements the MigrationEngine protocol with support for:
    - Lazy migration (transform on read via adapters)
    - Eager migration (batch transform all events)
    - Rollback capability (via causal links)
    - Progress tracking and resumability

    Implements: MigrationEngine protocol
    """

    def __init__(
        self,
        event_store: EventStore,
        target_store: Optional[EventStore] = None,
    ):
        """
        Initialize the migration engine.

        Args:
            event_store: Source event store
            target_store: Destination store (None = same store)
        """
        self._source = event_store
        self._target = target_store or event_store
        self._plans: Dict[str, MigrationPlan] = {}
        self._adapters: Dict[str, MigrationStep] = {}

        # Migration history
        self._history: List[Dict[str, Any]] = []

    def register_plan(self, plan: MigrationPlan) -> None:
        """Register a migration plan."""
        self._plans[plan.name] = plan

    def register_adapter(self, version: str, step: MigrationStep) -> None:
        """
        Register a lazy migration adapter.

        Adapters transform events on-read rather than batch migrating.
        This is useful for gradual migrations.

        Args:
            version: Schema version this adapter handles
            step: The migration step to apply
        """
        self._adapters[version] = step

    def migrate(
        self,
        plan_name: str,
        batch_size: int = 100,
    ) -> MigrationPlan:
        """
        Execute a migration plan.

        Args:
            plan_name: Name of registered plan to execute
            batch_size: Events to process per batch

        Returns:
            The executed plan with status updated

        Raises:
            KeyError: If plan not found
        """
        if plan_name not in self._plans:
            raise KeyError(f"Migration plan not found: {plan_name}")

        plan = self._plans[plan_name]
        plan._status = MigrationStatus.IN_PROGRESS

        try:
            self._execute_plan(plan, batch_size)
            plan._status = MigrationStatus.COMPLETED
        except Exception as e:
            plan._errors.append(str(e))
            plan._status = MigrationStatus.FAILED

        # Record in history
        self._history.append(plan.to_dict())

        return plan

    def _execute_plan(self, plan: MigrationPlan, batch_size: int) -> None:
        """Execute migration plan steps."""
        for step_idx, step in enumerate(plan.steps):
            plan._current_step = step_idx

            # Collect events for this step
            batch: List[CognitiveEvent] = []

            for event in self._source.iterate():
                if step.applies_to(event):
                    batch.append(event)

                    if len(batch) >= batch_size:
                        self._process_batch(batch, step, plan)
                        batch = []

            # Process remaining
            if batch:
                self._process_batch(batch, step, plan)

        plan._current_step = len(plan.steps)

    def _process_batch(
        self,
        events: List[CognitiveEvent],
        step: MigrationStep,
        plan: MigrationPlan,
    ) -> None:
        """Process a batch of events through a migration step."""
        for event in events:
            if plan.dry_run:
                # Just count, don't transform
                plan._events_processed += 1
                continue

            try:
                # Transform the event
                new_event = step.apply(event)

                # Append to target store
                self._target.append(new_event)

                # Create compaction event linking old to new
                compaction = Compaction(
                    compressed_events=(event.id,),
                    snapshot={
                        'migration': plan.name,
                        'step': step.name,
                        'original_id': event.id,
                        'new_id': new_event.id,
                    },
                    preserved_merkle_root=new_event.id,  # Link to migrated event
                )
                self._target.append(compaction)

                plan._events_processed += 1

            except Exception as e:
                plan._errors.append(f"Event {event.id}: {e}")

    def adapt_event(self, event: CognitiveEvent) -> CognitiveEvent:
        """
        Apply lazy migration to an event.

        This is called when reading events through adapters.
        It transforms old-format events to current format.

        Args:
            event: The event to potentially transform

        Returns:
            Transformed event (or original if no adapter applies)
        """
        # Check schema version in content
        version = event.content.get('_schema_version', 'v1')

        if version in self._adapters:
            adapter = self._adapters[version]
            if adapter.applies_to(event):
                return adapter.apply(event)

        return event

    def can_migrate(self, from_version: str, to_version: str) -> bool:
        """
        Check if a migration path exists.

        Args:
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            True if migration is possible
        """
        # Build migration graph from registered plans
        migration_graph: Dict[str, List[str]] = {}

        for plan in self._plans.values():
            for step in plan.steps:
                if step.version_from not in migration_graph:
                    migration_graph[step.version_from] = []
                migration_graph[step.version_from].append(step.version_to)

        # BFS to find path
        if from_version == to_version:
            return True

        visited = {from_version}
        queue = [from_version]

        while queue:
            current = queue.pop(0)
            for next_version in migration_graph.get(current, []):
                if next_version == to_version:
                    return True
                if next_version not in visited:
                    visited.add(next_version)
                    queue.append(next_version)

        return False

    def get_migration_path(
        self,
        from_version: str,
        to_version: str,
    ) -> List[MigrationStep]:
        """
        Get ordered steps to migrate between versions.

        Args:
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            List of migration steps in order

        Raises:
            ValueError: If no path exists
        """
        if from_version == to_version:
            return []

        # Build step lookup
        step_lookup: Dict[Tuple[str, str], MigrationStep] = {}
        graph: Dict[str, List[str]] = {}

        for plan in self._plans.values():
            for step in plan.steps:
                key = (step.version_from, step.version_to)
                step_lookup[key] = step

                if step.version_from not in graph:
                    graph[step.version_from] = []
                graph[step.version_from].append(step.version_to)

        # BFS to find shortest path
        visited = {from_version}
        queue = [(from_version, [])]

        while queue:
            current, path = queue.pop(0)

            for next_version in graph.get(current, []):
                step = step_lookup[(current, next_version)]
                new_path = path + [step]

                if next_version == to_version:
                    return new_path

                if next_version not in visited:
                    visited.add(next_version)
                    queue.append((next_version, new_path))

        raise ValueError(
            f"No migration path from {from_version} to {to_version}"
        )

    def current_version(self) -> str:
        """
        Determine current schema version from events.

        Returns the most recent schema version found in events.
        """
        latest_version = 'v1'  # Default

        for event in self._source.iterate():
            version = event.content.get('_schema_version', 'v1')
            # Simple string comparison - could be more sophisticated
            if version > latest_version:
                latest_version = version

        return latest_version

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        return self._history.copy()


# =============================================================================
# COMMON MIGRATION HELPERS
# =============================================================================


def rename_field(
    old_name: str,
    new_name: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a transform function that renames a field."""
    def transform(content: Dict[str, Any]) -> Dict[str, Any]:
        if old_name in content:
            content[new_name] = content.pop(old_name)
        return content
    return transform


def add_field(
    field_name: str,
    default_value: Any,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a transform function that adds a field with default."""
    def transform(content: Dict[str, Any]) -> Dict[str, Any]:
        if field_name not in content:
            content[field_name] = default_value
        return content
    return transform


def remove_field(
    field_name: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a transform function that removes a field."""
    def transform(content: Dict[str, Any]) -> Dict[str, Any]:
        content.pop(field_name, None)
        return content
    return transform


def transform_field(
    field_name: str,
    transform_fn: Callable[[Any], Any],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a transform function that modifies a field value."""
    def transform(content: Dict[str, Any]) -> Dict[str, Any]:
        if field_name in content:
            content[field_name] = transform_fn(content[field_name])
        return content
    return transform


def compose_transforms(
    *transforms: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Compose multiple transform functions."""
    def combined(content: Dict[str, Any]) -> Dict[str, Any]:
        for t in transforms:
            content = t(content)
        return content
    return combined


def by_event_type(*types: EventType) -> Callable[[CognitiveEvent], bool]:
    """Create a filter for specific event types."""
    type_set = set(types)
    return lambda event: event.event_type in type_set


def by_content_field(
    field: str,
    value: Any = None,
    exists: bool = True,
) -> Callable[[CognitiveEvent], bool]:
    """Create a filter for events with specific content fields."""
    def filter_fn(event: CognitiveEvent) -> bool:
        has_field = field in event.content
        if not exists:
            return not has_field
        if value is not None:
            return has_field and event.content[field] == value
        return has_field
    return filter_fn
