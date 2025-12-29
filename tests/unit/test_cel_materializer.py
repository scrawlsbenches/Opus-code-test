"""
Unit tests for CEL Materializer (wisdom/materializer.py).

Tests the event-to-entity materialization system.
This module has 21% coverage - target is 70%+.
"""

import pytest
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from cortical.cel.wisdom.materializer import (
    CacheEntry,
    EntityReducerRegistry,
    CachingMaterializer,
    default_reducer_registry,
    _FunctionReducer,
)
from cortical.cel.core.events import CognitiveEvent, EventType
from cortical.cel.core.references import EventHorizon, MerkleRoot


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_event_store():
    """Create a mock event store."""
    store = MagicMock()
    store.latest.return_value = MerkleRoot("latest-event-id")
    store.iterate.return_value = iter([])
    return store


@pytest.fixture
def sample_intention_event():
    """Create a sample INTENTION event (like creating a task)."""
    return CognitiveEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=EventType.INTENTION,
        causal_parents=(),
        content={
            'entity_id': 'T-20251229-001',
            'title': 'Test Task',
            'priority': 'high',
            'category': 'feature',
            'description': 'A test task',
        },
        concepts=('task', 'test'),
    )


@pytest.fixture
def sample_fulfillment_event():
    """Create a sample FULFILLMENT event (like completing a task)."""
    return CognitiveEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=EventType.FULFILLMENT,
        causal_parents=('intention-id',),
        content={
            'entity_id': 'T-20251229-001',
            'intention_id': 'T-20251229-001',
            'result': {'success': True},
        },
        concepts=('task', 'completed'),
    )


@pytest.fixture
def sample_invalidation_event():
    """Create a sample INVALIDATION event (like deleting a task)."""
    return CognitiveEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=EventType.INVALIDATION,
        causal_parents=(),
        content={
            'entity_id': 'T-20251229-001',
            'reason': 'Deleted by user',
        },
        concepts=('task', 'deleted'),
    )


@pytest.fixture
def registry_with_task_reducer():
    """Create a registry with a simple task reducer."""
    registry = EntityReducerRegistry()

    @registry.register('task')
    def task_reducer(state: Optional[Dict], event: CognitiveEvent) -> Optional[Dict]:
        if event.event_type == EventType.INTENTION:
            return {
                'id': event.content.get('entity_id', event.id),
                'title': event.content.get('title', ''),
                'status': 'pending',
            }
        if event.event_type == EventType.FULFILLMENT and state:
            return {**state, 'status': 'completed'}
        if event.event_type == EventType.INVALIDATION:
            return None
        return state

    return registry


# =============================================================================
# TEST: CacheEntry
# =============================================================================

class TestCacheEntry:
    """Test cache entry dataclass."""

    def test_create_cache_entry(self):
        """Can create a cache entry."""
        entry = CacheEntry(
            entity={'id': 'test', 'title': 'Test'},
            horizon='event-123',
            materialized_at=datetime.now(),
        )

        assert entry.entity == {'id': 'test', 'title': 'Test'}
        assert entry.horizon == 'event-123'
        assert entry.access_count == 0

    def test_is_valid_at_matching_horizon(self):
        """Entry is valid when horizon matches."""
        entry = CacheEntry(
            entity={'id': 'test'},
            horizon='event-123',
            materialized_at=datetime.now(),
        )

        assert entry.is_valid_at('event-123') is True

    def test_is_valid_at_different_horizon(self):
        """Entry is invalid when horizon differs."""
        entry = CacheEntry(
            entity={'id': 'test'},
            horizon='event-123',
            materialized_at=datetime.now(),
        )

        assert entry.is_valid_at('event-456') is False

    def test_access_count_increments(self):
        """Access count can be incremented."""
        entry = CacheEntry(
            entity={'id': 'test'},
            horizon='event-123',
            materialized_at=datetime.now(),
        )

        assert entry.access_count == 0
        entry.access_count += 1
        assert entry.access_count == 1


# =============================================================================
# TEST: EntityReducerRegistry
# =============================================================================

class TestEntityReducerRegistry:
    """Test reducer registry."""

    def test_create_empty_registry(self):
        """Can create empty registry."""
        registry = EntityReducerRegistry()

        assert registry.get('task') is None

    def test_register_with_decorator(self):
        """Can register reducer with decorator."""
        registry = EntityReducerRegistry()

        @registry.register('task')
        def task_reducer(state, event):
            return {'status': 'reduced'}

        assert registry.get('task') is not None

    def test_registered_reducer_is_callable(self):
        """Registered reducer can be called."""
        registry = EntityReducerRegistry()

        @registry.register('task')
        def task_reducer(state, event):
            return {'status': 'reduced'}

        reducer = registry.get('task')
        result = reducer(None, MagicMock())

        assert result == {'status': 'reduced'}

    def test_add_reducer_directly(self):
        """Can add reducer directly without decorator."""
        registry = EntityReducerRegistry()
        reducer = _FunctionReducer('task', lambda s, e: {'added': True})

        registry.add(reducer)

        assert registry.get('task') is not None

    def test_reduce_applies_reducer(self):
        """Reduce method applies the registered reducer."""
        registry = EntityReducerRegistry()

        @registry.register('task')
        def task_reducer(state, event):
            return {'title': event.content.get('title', '')}

        mock_event = MagicMock()
        mock_event.content = {'title': 'Test Task'}

        result = registry.reduce('task', None, mock_event)

        assert result == {'title': 'Test Task'}

    def test_reduce_raises_for_unknown_type(self):
        """Reduce raises KeyError for unknown entity type."""
        registry = EntityReducerRegistry()

        with pytest.raises(KeyError, match="No reducer registered"):
            registry.reduce('unknown', None, MagicMock())

    def test_reducer_receives_state(self):
        """Reducer receives current state."""
        registry = EntityReducerRegistry()

        @registry.register('task')
        def task_reducer(state, event):
            if state is None:
                return {'count': 1}
            return {'count': state['count'] + 1}

        mock_event = MagicMock()

        # First call - no state
        result1 = registry.reduce('task', None, mock_event)
        assert result1['count'] == 1

        # Second call - with state
        result2 = registry.reduce('task', result1, mock_event)
        assert result2['count'] == 2


# =============================================================================
# TEST: _FunctionReducer
# =============================================================================

class TestFunctionReducer:
    """Test function reducer wrapper."""

    def test_wraps_function(self):
        """Wraps a function as a reducer."""
        def my_reducer(state, event):
            return {'wrapped': True}

        wrapper = _FunctionReducer('task', my_reducer)

        assert wrapper.entity_type == 'task'
        assert wrapper(None, MagicMock()) == {'wrapped': True}

    def test_entity_type_property(self):
        """Entity type property returns correct value."""
        wrapper = _FunctionReducer('decision', lambda s, e: None)

        assert wrapper.entity_type == 'decision'

    def test_passes_state_and_event(self):
        """Passes both state and event to wrapped function."""
        calls = []

        def tracking_reducer(state, event):
            calls.append((state, event))
            return state

        wrapper = _FunctionReducer('task', tracking_reducer)
        mock_event = MagicMock()

        wrapper({'id': 'test'}, mock_event)

        assert len(calls) == 1
        assert calls[0][0] == {'id': 'test'}
        assert calls[0][1] is mock_event


# =============================================================================
# TEST: CachingMaterializer - Basic
# =============================================================================

class TestCachingMaterializerBasic:
    """Test basic materializer functionality."""

    def test_create_materializer(self, mock_event_store, registry_with_task_reducer):
        """Can create a materializer."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._cache_size == 1000  # default
        assert materializer._cache_ttl is None

    def test_create_with_custom_cache_size(self, mock_event_store, registry_with_task_reducer):
        """Can create with custom cache size."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
            cache_size=100,
        )

        assert materializer._cache_size == 100

    def test_create_with_ttl(self, mock_event_store, registry_with_task_reducer):
        """Can create with cache TTL."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
            cache_ttl_seconds=60.0,
        )

        assert materializer._cache_ttl == 60.0


# =============================================================================
# TEST: CachingMaterializer - Entity Type Detection
# =============================================================================

class TestCachingMaterializerEntityTypes:
    """Test entity type detection from ID prefixes."""

    def test_task_prefix(self, mock_event_store, registry_with_task_reducer):
        """T- prefix maps to task."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('T-20251229-001') == 'task'

    def test_decision_prefix(self, mock_event_store, registry_with_task_reducer):
        """D- prefix maps to decision."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('D-20251229-001') == 'decision'

    def test_sprint_prefix(self, mock_event_store, registry_with_task_reducer):
        """S- prefix maps to sprint."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('S-20251229-001') == 'sprint'

    def test_handoff_prefix(self, mock_event_store, registry_with_task_reducer):
        """H- prefix maps to handoff."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('H-20251229-001') == 'handoff'

    def test_epic_prefix(self, mock_event_store, registry_with_task_reducer):
        """EPIC- prefix maps to epic."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('EPIC-migration') == 'epic'

    def test_document_prefix(self, mock_event_store, registry_with_task_reducer):
        """DOC- prefix maps to document."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('DOC-guide') == 'document'

    def test_unknown_prefix_returns_none(self, mock_event_store, registry_with_task_reducer):
        """Unknown prefix returns None."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        assert materializer._entity_type_from_id('UNKNOWN-id') is None


# =============================================================================
# TEST: CachingMaterializer - Event Affects Entity
# =============================================================================

class TestCachingMaterializerEventAffectsEntity:
    """Test event-to-entity matching logic."""

    def test_direct_entity_id_match(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Event with matching entity_id affects entity."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer._event_affects_entity(
            sample_intention_event,
            'T-20251229-001'
        )

        assert result is True

    def test_no_match_returns_false(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Event not matching entity returns False."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer._event_affects_entity(
            sample_intention_event,
            'T-different-id'
        )

        assert result is False

    def test_fulfillment_matches_intention_id(
        self, mock_event_store, registry_with_task_reducer, sample_fulfillment_event
    ):
        """Fulfillment event matches by intention_id."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer._event_affects_entity(
            sample_fulfillment_event,
            'T-20251229-001'
        )

        assert result is True

    def test_invalidation_matches_entity_id(
        self, mock_event_store, registry_with_task_reducer, sample_invalidation_event
    ):
        """Invalidation event matches by entity_id."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer._event_affects_entity(
            sample_invalidation_event,
            'T-20251229-001'
        )

        assert result is True


# =============================================================================
# TEST: CachingMaterializer - Materialization
# =============================================================================

class TestCachingMaterializerMaterialization:
    """Test entity materialization."""

    def test_materialize_returns_none_for_no_events(
        self, mock_event_store, registry_with_task_reducer
    ):
        """Returns None when no events affect entity."""
        mock_event_store.iterate.return_value = iter([])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer.materialize('T-20251229-001')

        assert result is None

    def test_materialize_folds_events(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Materializes by folding matching events."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer.materialize('T-20251229-001')

        assert result is not None
        assert result['title'] == 'Test Task'
        assert result['status'] == 'pending'

    def test_materialize_applies_multiple_events(
        self, mock_event_store, registry_with_task_reducer,
        sample_intention_event, sample_fulfillment_event
    ):
        """Applies multiple events in sequence."""
        mock_event_store.iterate.return_value = iter([
            sample_intention_event,
            sample_fulfillment_event,
        ])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer.materialize('T-20251229-001')

        assert result is not None
        assert result['status'] == 'completed'

    def test_materialize_returns_none_for_unknown_type(
        self, mock_event_store, registry_with_task_reducer
    ):
        """Returns None for unknown entity type."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        result = materializer.materialize('UNKNOWN-type-id')

        assert result is None


# =============================================================================
# TEST: CachingMaterializer - Caching
# =============================================================================

class TestCachingMaterializerCaching:
    """Test caching behavior."""

    def test_second_access_is_cache_hit(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Second access returns cached value."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        # First access
        materializer.materialize('T-20251229-001')
        stats1 = materializer.cache_stats

        # Reset mock for second access
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        # Second access should be cache hit
        materializer.materialize('T-20251229-001')
        stats2 = materializer.cache_stats

        assert stats2['hits'] == stats1['hits'] + 1

    def test_cache_stats_tracks_hits_and_misses(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Cache stats track hits and misses."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        stats_initial = materializer.cache_stats
        assert stats_initial['hits'] == 0
        assert stats_initial['misses'] == 0

        # First access (miss)
        materializer.materialize('T-20251229-001')
        stats_after_miss = materializer.cache_stats
        assert stats_after_miss['misses'] == 1

    def test_cache_size_respected(
        self, mock_event_store, registry_with_task_reducer
    ):
        """Cache evicts when size exceeded."""
        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
            cache_size=2,
        )

        # Create events for different entities
        def make_event(entity_id):
            return CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.INTENTION,
                causal_parents=(),
                content={'entity_id': entity_id, 'title': f'Task {entity_id}'},
                concepts=('task',),
            )

        # Materialize 3 entities (cache size is 2)
        for i in range(3):
            mock_event_store.iterate.return_value = iter([make_event(f'T-{i}')])
            materializer.materialize(f'T-{i}')

        # Cache should not exceed size
        assert materializer.cache_stats['size'] <= 2

    def test_invalidate_removes_from_cache(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Invalidate removes entity from cache."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        # Materialize and cache
        materializer.materialize('T-20251229-001')
        assert materializer.cache_stats['size'] >= 1

        # Invalidate
        materializer.invalidate('T-20251229-001')

        # Cache should be smaller
        # Note: The key format includes @horizon, so check access order
        assert 'T-20251229-001@' not in str(materializer._cache.keys())

    def test_invalidate_all_clears_cache(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Invalidate_all clears entire cache."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        materializer.materialize('T-20251229-001')
        materializer.invalidate_all()

        assert materializer.cache_stats['size'] == 0


# =============================================================================
# TEST: CachingMaterializer - TTL
# =============================================================================

class TestCachingMaterializerTTL:
    """Test TTL-based cache expiration."""

    def test_expired_entry_triggers_rematerialization(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Expired cache entry causes rematerialization."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
            cache_ttl_seconds=0.01,  # Very short TTL
        )

        # First access
        materializer.materialize('T-20251229-001')

        # Wait for TTL to expire
        time.sleep(0.02)

        # Reset mock for second access
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        # Second access should be a miss (entry expired)
        materializer.materialize('T-20251229-001')

        # Should have 2 misses (both were cache misses)
        assert materializer.cache_stats['misses'] == 2

    def test_non_expired_entry_is_cache_hit(
        self, mock_event_store, registry_with_task_reducer, sample_intention_event
    ):
        """Non-expired cache entry is a hit."""
        mock_event_store.iterate.return_value = iter([sample_intention_event])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
            cache_ttl_seconds=60.0,  # Long TTL
        )

        # First access
        materializer.materialize('T-20251229-001')

        # Second access (should be hit)
        materializer.materialize('T-20251229-001')

        assert materializer.cache_stats['hits'] == 1


# =============================================================================
# TEST: CachingMaterializer - Batch Materialization
# =============================================================================

class TestCachingMaterializerBatch:
    """Test batch materialization."""

    def test_materialize_many_returns_dict(
        self, mock_event_store, registry_with_task_reducer
    ):
        """Materialize_many returns dict of entities."""
        def make_event(entity_id):
            return CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.INTENTION,
                causal_parents=(),
                content={'entity_id': entity_id, 'title': f'Task {entity_id}'},
                concepts=('task',),
            )

        mock_event_store.iterate.side_effect = [
            iter([make_event('T-1')]),
            iter([make_event('T-2')]),
        ]

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        results = materializer.materialize_many(['T-1', 'T-2'])

        assert len(results) == 2
        assert 'T-1' in results
        assert 'T-2' in results

    def test_materialize_many_excludes_none_results(
        self, mock_event_store, registry_with_task_reducer
    ):
        """Materialize_many excludes None results."""
        mock_event_store.iterate.side_effect = [
            iter([]),  # No events for first entity
            iter([CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.INTENTION,
                causal_parents=(),
                content={'entity_id': 'T-2', 'title': 'Task 2'},
                concepts=('task',),
            )]),
        ]

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        results = materializer.materialize_many(['T-1', 'T-2'])

        assert len(results) == 1
        assert 'T-1' not in results
        assert 'T-2' in results


# =============================================================================
# TEST: CachingMaterializer - Thread Safety
# =============================================================================

class TestCachingMaterializerThreadSafety:
    """Test thread safety of materializer."""

    def test_concurrent_materializations(
        self, mock_event_store, registry_with_task_reducer
    ):
        """Concurrent materializations don't corrupt cache."""
        def make_event(entity_id):
            return CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.INTENTION,
                causal_parents=(),
                content={'entity_id': entity_id, 'title': f'Task {entity_id}'},
                concepts=('task',),
            )

        # Allow multiple calls
        mock_event_store.iterate.side_effect = lambda **kwargs: iter([
            make_event('T-concurrent')
        ])

        materializer = CachingMaterializer(
            event_store=mock_event_store,
            reducer_registry=registry_with_task_reducer,
        )

        results = []
        errors = []

        def materialize_task():
            try:
                result = materializer.materialize('T-concurrent')
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=materialize_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is not None for r in results)


# =============================================================================
# TEST: default_reducer_registry
# =============================================================================

class TestDefaultReducerRegistry:
    """Test the default reducer registry factory."""

    def test_creates_registry_with_task_reducer(self):
        """Default registry has task reducer."""
        registry = default_reducer_registry()

        assert registry.get('task') is not None

    def test_creates_registry_with_decision_reducer(self):
        """Default registry has decision reducer."""
        registry = default_reducer_registry()

        assert registry.get('decision') is not None

    def test_task_reducer_creates_from_intention(self):
        """Task reducer creates task from INTENTION event."""
        registry = default_reducer_registry()

        event = CognitiveEvent(
            timestamp='2025-12-29T12:00:00Z',
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={
                'title': 'New Task',
                'priority': 'high',
                'category': 'feature',
            },
            concepts=('task',),
        )

        result = registry.reduce('task', None, event)

        assert result is not None
        assert result['title'] == 'New Task'
        assert result['priority'] == 'high'
        assert result['status'] == 'pending'

    def test_task_reducer_completes_on_fulfillment(self):
        """Task reducer completes task on FULFILLMENT event."""
        registry = default_reducer_registry()

        state = {
            'id': 'test',
            'title': 'Test',
            'status': 'pending',
            'version': 1,
        }

        event = CognitiveEvent(
            timestamp='2025-12-29T13:00:00Z',
            event_type=EventType.FULFILLMENT,
            causal_parents=(),
            content={'result': {'success': True}},
            concepts=('completed',),
        )

        result = registry.reduce('task', state, event)

        assert result is not None
        assert result['status'] == 'completed'
        assert result['version'] == 2

    def test_task_reducer_invalidation_returns_none(self):
        """Task reducer returns None on INVALIDATION."""
        registry = default_reducer_registry()

        state = {'id': 'test', 'title': 'Test'}

        event = CognitiveEvent(
            timestamp='2025-12-29T14:00:00Z',
            event_type=EventType.INVALIDATION,
            causal_parents=(),
            content={},
            concepts=(),
        )

        result = registry.reduce('task', state, event)

        assert result is None

    def test_decision_reducer_creates_from_intention_with_category(self):
        """Decision reducer creates decision from categorized INTENTION."""
        registry = default_reducer_registry()

        event = CognitiveEvent(
            timestamp='2025-12-29T12:00:00Z',
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={
                'title': 'Use event sourcing',
                'category': 'decision',
                'description': 'For better auditability',
            },
            concepts=('decision',),
        )

        result = registry.reduce('decision', None, event)

        assert result is not None
        assert result['title'] == 'Use event sourcing'
        assert result['rationale'] == 'For better auditability'

    def test_decision_reducer_ignores_non_decision_intention(self):
        """Decision reducer ignores INTENTION without decision category."""
        registry = default_reducer_registry()

        event = CognitiveEvent(
            timestamp='2025-12-29T12:00:00Z',
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={
                'title': 'Not a decision',
                'category': 'task',
            },
            concepts=('task',),
        )

        result = registry.reduce('decision', None, event)

        assert result is None
