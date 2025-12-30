"""
Event-to-Entity materialization for the Cognitive Event Lattice.

The Materializer is responsible for "folding" events into entity state.
Entities don't exist in storage - only events do. Entities are
computed projections of events.

Key Insight:
    materialize(entity_id, at=horizon) = fold(events[:horizon], reducer)

This enables:
    - Temporal queries: "What was X at time T?"
    - Self-reference without paradox: Reference snapshots, not live state
    - Caching: Computed entities can be cached and invalidated

Design Pattern:
    Reducers are pure functions: (state, event) -> new_state
    This is similar to Redux reducers or functional fold/reduce.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Sequence,
    TypeVar,
)

from ..core.events import CognitiveEvent, EventType
from ..core.protocols import EventReducer, EventStore
from ..core.references import EventHorizon


T = TypeVar('T')
E = TypeVar('E')


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the materialization cache."""

    entity: T
    horizon: str  # Event ID at which this was materialized
    materialized_at: datetime
    access_count: int = 0

    def is_valid_at(self, horizon: str) -> bool:
        """Check if this cache entry is valid for the given horizon."""
        return self.horizon == horizon


class EntityReducerRegistry:
    """
    Registry of entity reducers by type.

    Reducers are registered for each entity type (task, decision, etc.)
    and are used to fold events into entity state.

    Example:
        registry = EntityReducerRegistry()

        @registry.register('task')
        def task_reducer(state: Optional[Task], event: CognitiveEvent) -> Optional[Task]:
            if event.event_type == EventType.INTENTION:
                return Task.from_intention(event)
            elif event.event_type == EventType.FULFILLMENT:
                return state._replace(status='completed')
            return state
    """

    def __init__(self):
        self._reducers: Dict[str, EventReducer] = {}

    def register(self, entity_type: str) -> Callable[[Callable], Callable]:
        """
        Decorator to register a reducer for an entity type.

        Args:
            entity_type: Type of entity (e.g., 'task', 'decision')

        Returns:
            Decorator function
        """
        def decorator(func: Callable[[Optional[T], CognitiveEvent], Optional[T]]) -> Callable:
            self._reducers[entity_type] = _FunctionReducer(entity_type, func)
            return func
        return decorator

    def add(self, reducer: EventReducer) -> None:
        """Add a reducer to the registry."""
        self._reducers[reducer.entity_type] = reducer

    def get(self, entity_type: str) -> Optional[EventReducer]:
        """Get reducer for an entity type."""
        return self._reducers.get(entity_type)

    def reduce(
        self,
        entity_type: str,
        state: Optional[T],
        event: CognitiveEvent,
    ) -> Optional[T]:
        """
        Apply reducer for entity type.

        Args:
            entity_type: Type of entity
            state: Current state (None if new)
            event: Event to apply

        Returns:
            New state

        Raises:
            KeyError: If no reducer registered for type
        """
        reducer = self._reducers.get(entity_type)
        if reducer is None:
            raise KeyError(f"No reducer registered for entity type: {entity_type}")
        return reducer(state, event)


@dataclass
class _FunctionReducer(Generic[T]):
    """Wrapper to make a function implement EventReducer protocol."""

    _entity_type: str
    _func: Callable[[Optional[T], CognitiveEvent], Optional[T]]

    def __call__(self, state: Optional[T], event: CognitiveEvent) -> Optional[T]:
        return self._func(state, event)

    @property
    def entity_type(self) -> str:
        return self._entity_type


class CachingMaterializer(Generic[T]):
    """
    Materializer with LRU caching.

    Maintains a cache of recently materialized entities to avoid
    re-computing from events on every access.

    Cache Invalidation:
        - Explicit: invalidate(entity_id) or invalidate_all()
        - TTL-based: Entries expire after configured time
        - LRU eviction: Least recently used entries removed when full

    Thread Safety:
        This class is thread-safe. Internal locking protects cache
        operations during concurrent access.

    Implements: Materializer protocol
    """

    def __init__(
        self,
        event_store: EventStore,
        reducer_registry: EntityReducerRegistry,
        cache_size: int = 1000,
        cache_ttl_seconds: Optional[float] = None,
        entity_index: Optional[Any] = None,
    ):
        """
        Initialize the materializer.

        Args:
            event_store: Source of events
            reducer_registry: Registry of entity reducers
            cache_size: Maximum cache entries
            cache_ttl_seconds: Cache TTL (None = no expiry)
            entity_index: Optional EntityIndex for O(1) entity lookups.
                         If provided, materialization uses indexed lookups
                         instead of scanning all events (469x speedup at 100K scale).
        """
        self._store = event_store
        self._reducers = reducer_registry
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl_seconds
        self._entity_index = entity_index

        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: list[str] = []
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._index_lookups = 0
        self._scan_lookups = 0

    def _cache_key(self, entity_id: str, horizon: Optional[EventHorizon]) -> str:
        """Generate cache key for entity at horizon."""
        horizon_id = horizon.event_id if horizon else "HEAD"
        return f"{entity_id}@{horizon_id}"

    def _evict_lru(self) -> None:
        """Evict least recently used entries if cache is full."""
        while len(self._cache) >= self._cache_size and self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)

    def _check_ttl(self, entry: CacheEntry[T]) -> bool:
        """Check if cache entry is still valid (not expired)."""
        if self._cache_ttl is None:
            return True
        age = (datetime.now() - entry.materialized_at).total_seconds()
        return age < self._cache_ttl

    def materialize(
        self,
        entity_id: str,
        at: Optional[EventHorizon] = None,
    ) -> Optional[T]:
        """
        Materialize an entity at a specific point in time.

        Args:
            entity_id: Entity to materialize
            at: Event horizon (None = current state)

        Returns:
            Materialized entity, or None if doesn't exist
        """
        with self._lock:
            cache_key = self._cache_key(entity_id, at)

            # Check cache
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if self._check_ttl(entry):
                    self._hits += 1
                    entry.access_count += 1
                    # Update access order
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
                    self._access_order.append(cache_key)
                    return entry.entity

            self._misses += 1

        # Materialize from events (outside lock for performance)
        entity = self._materialize_from_events(entity_id, at)

        with self._lock:
            # Cache result
            self._evict_lru()

            horizon_id = at.event_id if at else (
                self._store.latest().value if self._store.latest() else "EMPTY"
            )
            self._cache[cache_key] = CacheEntry(
                entity=entity,
                horizon=horizon_id,
                materialized_at=datetime.now(),
            )
            self._access_order.append(cache_key)

        return entity

    def _materialize_from_events(
        self,
        entity_id: str,
        at: Optional[EventHorizon] = None,
    ) -> Optional[T]:
        """
        Materialize entity by folding events.

        This is the core materialization logic - iterate events
        and apply reducers to build up entity state.

        Performance:
            - With EntityIndex: O(entity_events) - only events for this entity
            - Without EntityIndex: O(all_events) - scans all events
        """
        # Determine entity type from ID prefix
        entity_type = self._entity_type_from_id(entity_id)
        if entity_type is None:
            return None

        reducer = self._reducers.get(entity_type)
        if reducer is None:
            return None

        # Fold events up to horizon
        state: Optional[T] = None
        to_event_id = at.event_id if at else None

        # Use EntityIndex for O(1) lookup if available
        if self._entity_index is not None:
            self._index_lookups += 1
            # Get only events for this entity (O(1) lookup!)
            event_ids = self._entity_index.events_for(entity_id, until=to_event_id)
            for event_id in event_ids:
                event = self._store.get(event_id)
                if event is not None:
                    state = reducer(state, event)
        else:
            # Fallback: scan all events (O(n))
            self._scan_lookups += 1
            for event in self._store.iterate(to_event=to_event_id):
                # Check if this event affects our entity
                if self._event_affects_entity(event, entity_id):
                    state = reducer(state, event)

        return state

    def _entity_type_from_id(self, entity_id: str) -> Optional[str]:
        """Infer entity type from ID prefix."""
        prefixes = {
            'T-': 'task',
            'D-': 'decision',
            'S-': 'sprint',
            'E-': 'edge',
            'H-': 'handoff',
            'EPIC-': 'epic',
            'CML-': 'claudemd_layer',
            'DOC-': 'document',
        }
        for prefix, entity_type in prefixes.items():
            if entity_id.startswith(prefix):
                return entity_type
        return None

    def _event_affects_entity(self, event: CognitiveEvent, entity_id: str) -> bool:
        """Check if an event affects a specific entity."""
        content = event.content

        # Direct entity reference
        if content.get('entity_id') == entity_id:
            return True

        # Intention creating this entity
        if event.event_type == EventType.INTENTION:
            # Check if this intention's ID matches (for tasks created as intentions)
            if content.get('id') == entity_id:
                return True

        # Fulfillment of intention
        if event.event_type == EventType.FULFILLMENT:
            if content.get('intention_id') == entity_id:
                return True

        # Invalidation
        if event.event_type == EventType.INVALIDATION:
            if content.get('entity_id') == entity_id:
                return True

        return False

    def materialize_many(
        self,
        entity_ids: Sequence[str],
        at: Optional[EventHorizon] = None,
    ) -> Dict[str, T]:
        """Batch materialize multiple entities."""
        results = {}
        for entity_id in entity_ids:
            entity = self.materialize(entity_id, at)
            if entity is not None:
                results[entity_id] = entity
        return results

    def invalidate(self, entity_id: str) -> None:
        """Invalidate cached materialization for an entity."""
        with self._lock:
            # Remove all cache entries for this entity (at any horizon)
            keys_to_remove = [
                k for k in self._cache.keys()
                if k.startswith(f"{entity_id}@")
            ]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                if key in self._access_order:
                    self._access_order.remove(key)

    def invalidate_all(self) -> None:
        """Invalidate all cached materializations."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def register_reducer(
        self,
        entity_type: str,
        reducer: EventReducer[T],
    ) -> None:
        """Register a reducer for an entity type."""
        self._reducers.add(reducer)

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            lookup_total = self._index_lookups + self._scan_lookups
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total if total > 0 else 0.0,
                'size': len(self._cache),
                'max_size': self._cache_size,
                # Materialization lookup stats
                'index_lookups': self._index_lookups,
                'scan_lookups': self._scan_lookups,
                'index_ratio': self._index_lookups / lookup_total if lookup_total > 0 else 0.0,
                'has_entity_index': self._entity_index is not None,
            }


# =============================================================================
# DEFAULT REDUCERS
# =============================================================================


def default_reducer_registry() -> EntityReducerRegistry:
    """
    Create registry with default reducers for standard entity types.

    This provides basic reducers that work with the standard
    CognitiveEvent structure. Custom reducers can be added for
    more sophisticated materialization logic.
    """
    registry = EntityReducerRegistry()

    @registry.register('task')
    def task_reducer(state: Optional[Dict], event: CognitiveEvent) -> Optional[Dict]:
        """Reduce events into task state."""
        if event.event_type == EventType.INTENTION:
            # Create new task from intention
            return {
                'id': event.id,  # Use event ID as entity ID
                'entity_type': 'task',
                'title': event.content.get('title', ''),
                'status': 'pending',
                'priority': event.content.get('priority', 'medium'),
                'category': event.content.get('category', 'feature'),
                'description': event.content.get('description', ''),
                'created_at': event.timestamp,
                'modified_at': event.timestamp,
                'version': 1,
            }

        if event.event_type == EventType.FULFILLMENT and state is not None:
            # Mark task as completed
            return {
                **state,
                'status': 'completed',
                'modified_at': event.timestamp,
                'version': state.get('version', 1) + 1,
                'result': event.content.get('result', {}),
            }

        if event.event_type == EventType.INVALIDATION:
            # Task was invalidated (deleted)
            return None

        return state

    @registry.register('decision')
    def decision_reducer(state: Optional[Dict], event: CognitiveEvent) -> Optional[Dict]:
        """Reduce events into decision state."""
        if event.event_type == EventType.INTENTION:
            # Decisions are recorded as a special category of intention
            if event.content.get('category') == 'decision':
                return {
                    'id': event.id,
                    'entity_type': 'decision',
                    'title': event.content.get('title', ''),
                    'rationale': event.content.get('description', ''),
                    'created_at': event.timestamp,
                    'version': 1,
                }

        if event.event_type == EventType.INVALIDATION:
            return None

        return state

    return registry
