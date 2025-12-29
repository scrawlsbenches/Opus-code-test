"""
Tests for CEL Performance Optimization Components.

Tests the high-performance components in cortical.cel.performance:
- EntityIndex
- ConceptIndex
- TemporalIndex
- OptimizedDAG
- SnapshotManager
- StreamingEventStore
"""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Import the modules we're testing
from cortical.cel.performance.entity_index import (
    EntityIndex,
    ConceptIndex,
    TemporalIndex,
)
from cortical.cel.performance.optimized_dag import (
    OptimizedDAG,
    HeapTopologicalSort,
    CausalViolationError,
    DuplicateEventError,
)
from cortical.cel.performance.snapshots import (
    SnapshotManager,
    SnapshotConfig,
    Snapshot,
    SnapshotMetadata,
)
from cortical.cel.core.events import CognitiveEvent, EventType
from cortical.cel.core.references import EventHorizon


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_events() -> List[CognitiveEvent]:
    """Create a list of sample events for testing."""
    events = []
    for i in range(100):
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={
                'entity_id': f'entity_{i % 10}',
                'data': f'test_data_{i}',
            },
            concepts=(f'concept_{i % 5}', 'test'),
        )
        events.append(event)
    return events


@pytest.fixture
def entity_index(sample_events: List[CognitiveEvent]) -> EntityIndex:
    """Create an entity index with sample events."""
    index = EntityIndex()
    for event in sample_events:
        index.on_event(event)
    return index


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# ENTITY INDEX TESTS
# =============================================================================

class TestEntityIndex:
    """Tests for EntityIndex."""

    def test_empty_index(self):
        """Empty index should return empty results."""
        index = EntityIndex()
        assert index.events_for("nonexistent") == []
        assert not index.entity_exists("nonexistent")
        assert index.entity_count() == 0

    def test_index_single_event(self):
        """Single event should be indexed correctly."""
        index = EntityIndex()
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'entity_id': 'entity_1'},
            concepts=(),
        )
        index.on_event(event)

        assert index.entity_exists('entity_1')
        assert len(index.events_for('entity_1')) == 1
        assert index.event_count('entity_1') == 1

    def test_index_multiple_events_same_entity(self, sample_events):
        """Multiple events for same entity should all be indexed."""
        index = EntityIndex()
        for event in sample_events:
            index.on_event(event)

        # entity_0 should have 10 events (every 10th from 100)
        events = index.events_for('entity_0')
        assert len(events) == 10

    def test_events_in_chronological_order(self):
        """Events should be returned in chronological order."""
        index = EntityIndex()

        timestamps = [
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z",
        ]

        for ts in timestamps:
            event = CognitiveEvent(
                timestamp=ts,
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'entity_id': 'entity_1'},
                concepts=(),
            )
            index.on_event(event)

        events = index.events_for('entity_1')
        assert len(events) == 3
        # Should be in order since we inserted in order

    def test_entities_affected_by(self, entity_index, sample_events):
        """Should find entities affected by an event."""
        event = sample_events[0]
        entity_id = event.content.get('entity_id')

        affected = entity_index.entities_affected_by(event.id)
        assert entity_id in affected

    def test_time_filtered_queries(self):
        """Should support time-filtered queries."""
        index = EntityIndex()

        # Add events at different times
        for i in range(10):
            ts = f"2025-01-{i+1:02d}T00:00:00Z"
            event = CognitiveEvent(
                timestamp=ts,
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'entity_id': 'entity_1'},
                concepts=(),
            )
            index.on_event(event)

        # Query with time filter
        events = index.events_for(
            'entity_1',
            since="2025-01-05T00:00:00Z",
            until="2025-01-08T00:00:00Z",
        )
        assert len(events) == 4  # Jan 5, 6, 7, 8

    def test_clear(self, entity_index):
        """Clear should empty the index."""
        assert entity_index.entity_count() > 0
        entity_index.clear()
        assert entity_index.entity_count() == 0

    def test_save_and_load(self, entity_index, temp_dir):
        """Index should be persistable."""
        path = temp_dir / "entity_index.json"
        entity_index.save(path)

        loaded = EntityIndex.load(path)
        assert loaded.entity_count() == entity_index.entity_count()


# =============================================================================
# CONCEPT INDEX TESTS
# =============================================================================

class TestConceptIndex:
    """Tests for ConceptIndex."""

    def test_empty_index(self):
        """Empty index should work correctly."""
        index = ConceptIndex()
        assert not index.probably_has("nonexistent")
        assert index.events_for("nonexistent") == set()
        assert index.concept_count == 0

    def test_index_concepts(self, sample_events):
        """Concepts should be indexed correctly."""
        index = ConceptIndex()
        for event in sample_events:
            index.on_event(event)

        # 5 unique concepts (concept_0 through concept_4) plus 'test'
        assert index.concept_count == 6

    def test_bloom_filter(self, sample_events):
        """Bloom filter should work for existing concepts."""
        index = ConceptIndex()
        for event in sample_events:
            index.on_event(event)

        # Should return True for existing concepts
        assert index.probably_has('concept_0')
        assert index.probably_has('test')

        # Should return False for non-existent (with possible false positives)
        # We can't guarantee no false positives, but most should be False
        false_positive_count = sum(
            1 for i in range(100)
            if index.probably_has(f'definitely_not_a_concept_{i}')
        )
        assert false_positive_count < 10  # Expect <10% false positives

    def test_events_for_all(self, sample_events):
        """Should find events matching ALL concepts."""
        index = ConceptIndex()
        for event in sample_events:
            index.on_event(event)

        # All sample events have 'test' concept
        all_test = index.events_for_all(['test'])
        assert len(all_test) == 100

        # Events with both concept_0 and test
        both = index.events_for_all(['concept_0', 'test'])
        assert len(both) == 20  # Every 5th event has concept_0

    def test_events_for_any(self, sample_events):
        """Should find events matching ANY concept."""
        index = ConceptIndex()
        for event in sample_events:
            index.on_event(event)

        # Events with concept_0 OR concept_1
        any_match = index.events_for_any(['concept_0', 'concept_1'])
        assert len(any_match) == 40  # 20 each


# =============================================================================
# TEMPORAL INDEX TESTS
# =============================================================================

class TestTemporalIndex:
    """Tests for TemporalIndex."""

    def test_empty_index(self):
        """Empty index should work correctly."""
        index = TemporalIndex()
        assert index.event_count == 0
        assert index.time_range == (None, None)
        assert index.events_in_range() == []

    def test_range_queries(self):
        """Range queries should return correct events."""
        index = TemporalIndex()

        # Add events at different times
        for i in range(10):
            ts = f"2025-01-{i+1:02d}T00:00:00Z"
            event = CognitiveEvent(
                timestamp=ts,
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'index': i},
                concepts=(),
            )
            index.on_event(event)

        # Query range
        events = index.events_in_range(
            start="2025-01-03T00:00:00Z",
            end="2025-01-07T00:00:00Z",
        )
        assert len(events) == 5

    def test_events_before(self):
        """Should get events before a timestamp."""
        index = TemporalIndex()

        for i in range(10):
            ts = f"2025-01-{i+1:02d}T00:00:00Z"
            event = CognitiveEvent(
                timestamp=ts,
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'index': i},
                concepts=(),
            )
            index.on_event(event)

        events = index.events_before("2025-01-05T00:00:00Z", limit=10)
        assert len(events) == 4  # Jan 1-4

    def test_time_range(self):
        """Time range should reflect min/max timestamps."""
        index = TemporalIndex()

        event1 = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={},
            concepts=(),
        )
        event2 = CognitiveEvent(
            timestamp="2025-12-31T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={},
            concepts=(),
        )
        index.on_event(event1)
        index.on_event(event2)

        earliest, latest = index.time_range
        assert earliest == "2025-01-01T00:00:00Z"
        assert latest == "2025-12-31T00:00:00Z"


# =============================================================================
# OPTIMIZED DAG TESTS
# =============================================================================

class TestOptimizedDAG:
    """Tests for OptimizedDAG."""

    def test_empty_dag(self):
        """Empty DAG should work correctly."""
        dag = OptimizedDAG()
        assert dag.count == 0
        assert dag.get_heads() == []
        assert dag.get_latest() is None

    def test_add_single_event(self):
        """Single event should be added correctly."""
        dag = OptimizedDAG()
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={},
            concepts=(),
        )
        root = dag.add(event)

        assert dag.count == 1
        assert dag.contains(event.id)
        assert len(dag.get_heads()) == 1

    def test_causal_chain(self):
        """Events with causal dependencies should form a chain."""
        dag = OptimizedDAG()

        events = []
        prev_id = None
        for i in range(5):
            parents = (prev_id,) if prev_id else ()
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=parents,
                content={'index': i},
                concepts=(),
            )
            dag.add(event, verify_parents=False)
            events.append(event)
            prev_id = event.id

        assert dag.count == 5
        # Only the last event should be a head
        heads = dag.get_heads()
        assert len(heads) == 1

    def test_duplicate_event_error(self):
        """Adding duplicate event should raise error."""
        dag = OptimizedDAG()
        event = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'fixed': True},
            concepts=(),
        )
        dag.add(event)

        with pytest.raises(DuplicateEventError):
            dag.add(event)

    def test_causal_order_iteration(self):
        """causal_order() should yield events in topological order."""
        dag = OptimizedDAG()

        # Create chain: A -> B -> C
        event_a = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'name': 'A'},
            concepts=(),
        )
        dag.add(event_a, verify_parents=False)

        event_b = CognitiveEvent(
            timestamp="2025-01-02T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(event_a.id,),
            content={'name': 'B'},
            concepts=(),
        )
        dag.add(event_b, verify_parents=False)

        event_c = CognitiveEvent(
            timestamp="2025-01-03T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(event_b.id,),
            content={'name': 'C'},
            concepts=(),
        )
        dag.add(event_c, verify_parents=False)

        # Causal order should be A, B, C
        order = list(dag.causal_order())
        assert len(order) == 3
        assert order[0].content['name'] == 'A'
        assert order[1].content['name'] == 'B'
        assert order[2].content['name'] == 'C'

    def test_ancestors_and_descendants(self):
        """Should find ancestors and descendants correctly."""
        dag = OptimizedDAG()

        # Create: Root -> [A, B] -> C
        root = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'name': 'root'},
            concepts=(),
        )
        dag.add(root, verify_parents=False)

        event_a = CognitiveEvent(
            timestamp="2025-01-02T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(root.id,),
            content={'name': 'A'},
            concepts=(),
        )
        dag.add(event_a, verify_parents=False)

        event_b = CognitiveEvent(
            timestamp="2025-01-02T00:00:01Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(root.id,),
            content={'name': 'B'},
            concepts=(),
        )
        dag.add(event_b, verify_parents=False)

        # Ancestors of A should include root
        ancestors = list(dag.ancestors(event_a.id))
        assert len(ancestors) == 1
        assert ancestors[0].id == root.id

        # Descendants of root should include A and B
        descendants = list(dag.descendants(root.id))
        assert len(descendants) == 2

    def test_verify_integrity(self):
        """Integrity check should pass for valid DAG."""
        dag = OptimizedDAG()

        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={},
            concepts=(),
        )
        dag.add(event)

        errors = dag.verify_integrity()
        assert errors == []


# =============================================================================
# SNAPSHOT MANAGER TESTS
# =============================================================================

class TestSnapshotManager:
    """Tests for SnapshotManager."""

    def test_empty_manager(self, temp_dir):
        """Empty manager should work correctly."""
        manager = SnapshotManager(temp_dir)
        assert manager.snapshot_count == 0
        assert manager.load_latest() is None
        assert manager.list_snapshots() == []

    def test_create_and_load_snapshot(self, temp_dir):
        """Should create and load snapshots."""
        manager = SnapshotManager(temp_dir)

        horizon = EventHorizon(event_id="test_event_123")
        entity_index = {
            'entity_1': ['event_1', 'event_2'],
            'entity_2': ['event_3'],
        }

        metadata = manager.create_snapshot(
            horizon=horizon,
            event_count=100,
            entity_index=entity_index,
        )

        assert metadata.snapshot_type == 'full'
        assert metadata.event_count == 100
        assert manager.snapshot_count == 1

        # Load snapshot
        loaded = manager.load_latest()
        assert loaded is not None
        assert loaded.metadata.event_horizon == "test_event_123"
        assert len(loaded.entity_index) == 2

    def test_snapshot_retention(self, temp_dir):
        """Old snapshots should be cleaned up."""
        config = SnapshotConfig(retention_count=2)
        manager = SnapshotManager(temp_dir, config)

        # Create 5 snapshots
        for i in range(5):
            horizon = EventHorizon(event_id=f"event_{i}")
            manager.create_snapshot(
                horizon=horizon,
                event_count=i * 100,
                entity_index={},
            )

        # Only 2 should remain (retention_count=2)
        assert manager.snapshot_count == 2

    def test_should_snapshot(self, temp_dir):
        """should_snapshot should trigger at correct intervals."""
        config = SnapshotConfig(full_interval=100, delta_interval=10)
        manager = SnapshotManager(temp_dir, config)

        assert manager.should_snapshot(0) == 'none'
        assert manager.should_snapshot(50) == 'none'
        assert manager.should_snapshot(100) == 'full'
        assert manager.should_snapshot(200) == 'full'

    def test_compressed_snapshots(self, temp_dir):
        """Compressed snapshots should work."""
        config = SnapshotConfig(compress=True)
        manager = SnapshotManager(temp_dir, config)

        horizon = EventHorizon(event_id="test_compressed")
        manager.create_snapshot(
            horizon=horizon,
            event_count=100,
            entity_index={'entity_1': ['e1', 'e2'] * 100},  # Make it big
        )

        loaded = manager.load_latest()
        assert loaded is not None
        assert loaded.metadata.compressed == True


# =============================================================================
# HEAP TOPOLOGICAL SORT TESTS
# =============================================================================

class TestHeapTopologicalSort:
    """Tests for HeapTopologicalSort."""

    def test_empty_sort(self):
        """Empty graph should produce empty result."""
        sorter = HeapTopologicalSort({}, {})
        result = list(sorter)
        assert result == []

    def test_single_event(self):
        """Single event should be yielded."""
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={},
            concepts=(),
        )
        events = {event.id: event}
        children = {}

        sorter = HeapTopologicalSort(events, children)
        result = list(sorter)

        assert len(result) == 1
        assert result[0].id == event.id

    def test_chain_ordering(self):
        """Chain should be yielded in correct order."""
        # Create A -> B -> C
        event_a = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'name': 'A'},
            concepts=(),
        )
        event_b = CognitiveEvent(
            timestamp="2025-01-02T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(event_a.id,),
            content={'name': 'B'},
            concepts=(),
        )
        event_c = CognitiveEvent(
            timestamp="2025-01-03T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(event_b.id,),
            content={'name': 'C'},
            concepts=(),
        )

        events = {
            event_a.id: event_a,
            event_b.id: event_b,
            event_c.id: event_c,
        }
        children = {
            event_a.id: {event_b.id},
            event_b.id: {event_c.id},
        }

        sorter = HeapTopologicalSort(events, children)
        result = list(sorter)

        assert len(result) == 3
        assert result[0].content['name'] == 'A'
        assert result[1].content['name'] == 'B'
        assert result[2].content['name'] == 'C'

    def test_concurrent_events_ordered_by_timestamp(self):
        """Concurrent events should be ordered by timestamp."""
        # Create root -> [A, B] where A and B have same depth
        root = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'name': 'root'},
            concepts=(),
        )
        event_a = CognitiveEvent(
            timestamp="2025-01-02T00:00:00Z",  # Earlier
            event_type=EventType.OBSERVATION,
            causal_parents=(root.id,),
            content={'name': 'A'},
            concepts=(),
        )
        event_b = CognitiveEvent(
            timestamp="2025-01-03T00:00:00Z",  # Later
            event_type=EventType.OBSERVATION,
            causal_parents=(root.id,),
            content={'name': 'B'},
            concepts=(),
        )

        events = {
            root.id: root,
            event_a.id: event_a,
            event_b.id: event_b,
        }
        children = {
            root.id: {event_a.id, event_b.id},
        }

        sorter = HeapTopologicalSort(events, children)
        result = list(sorter)

        assert len(result) == 3
        assert result[0].content['name'] == 'root'
        # A should come before B (earlier timestamp)
        assert result[1].content['name'] == 'A'
        assert result[2].content['name'] == 'B'


# =============================================================================
# STREAMING STORE TESTS
# =============================================================================

class TestStreamingEventStore:
    """Tests for StreamingEventStore."""

    @pytest.fixture
    def temp_store_dir(self, tmp_path):
        """Create a temporary directory for the store."""
        store_dir = tmp_path / "streaming_store"
        store_dir.mkdir()
        return store_dir

    def test_empty_store(self, temp_store_dir):
        """Empty store should have count 0."""
        from cortical.cel.performance.streaming_store import (
            StreamingEventStore,
            StoreConfig,
        )

        store = StreamingEventStore(temp_store_dir, StoreConfig())
        assert store.count == 0

    def test_append_and_get(self, temp_store_dir):
        """Should be able to append and retrieve events."""
        from cortical.cel.performance.streaming_store import (
            StreamingEventStore,
            StoreConfig,
        )

        store = StreamingEventStore(temp_store_dir, StoreConfig(batch_size=10))

        # Create and append event
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'test': 'data', 'entity_id': 'E-001'},
            concepts=('test',),
        )
        store.append(event)
        store.flush()

        # Retrieve event
        retrieved = store.get(event.id)
        assert retrieved is not None
        assert retrieved.id == event.id
        assert retrieved.content['test'] == 'data'

    def test_append_multiple_and_count(self, temp_store_dir):
        """Should correctly count multiple events."""
        from cortical.cel.performance.streaming_store import (
            StreamingEventStore,
            StoreConfig,
        )

        store = StreamingEventStore(temp_store_dir, StoreConfig(batch_size=5))

        # Append multiple events
        for i in range(10):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'index': i, 'entity_id': f'E-{i % 3}'},
                concepts=(f'concept_{i % 5}',),
            )
            store.append(event)

        store.flush()
        assert store.count == 10

    def test_events_for_entity(self, temp_store_dir):
        """Should query events by entity."""
        from cortical.cel.performance.streaming_store import (
            StreamingEventStore,
            StoreConfig,
        )

        store = StreamingEventStore(temp_store_dir, StoreConfig())

        # Create events for different entities
        entity_1_events = []
        for i in range(5):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'entity_id': 'entity_1', 'seq': i},
                concepts=(),
            )
            store.append(event)
            entity_1_events.append(event.id)

        for i in range(3):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'entity_id': 'entity_2', 'seq': i},
                concepts=(),
            )
            store.append(event)

        store.flush()

        # Query entity_1
        results = store.events_for_entity('entity_1')
        assert len(results) == 5

    def test_lru_cache(self, temp_store_dir):
        """LRU cache should work correctly."""
        from cortical.cel.performance.streaming_store import LRUCache

        cache = LRUCache(max_size=3)

        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)

        assert cache.get('a') == 1
        assert cache.get('b') == 2
        assert len(cache) == 3

        # Adding 4th item should evict oldest (c, since a and b were accessed)
        cache.put('d', 4)
        assert len(cache) == 3
        assert cache.get('c') is None  # Evicted
        assert cache.get('d') == 4

    def test_event_index(self, temp_store_dir):
        """EventIndex should track event locations."""
        from cortical.cel.performance.streaming_store import EventIndex

        index = EventIndex()

        # Add some events using the correct method
        index.add('e1', 'seg1', 0)
        index.add('e2', 'seg1', 1)
        index.add('e3', 'seg2', 0)

        assert index.count == 3
        assert index.get_location('e1') == ('seg1', 0)
        assert index.get_location('e2') == ('seg1', 1)
        assert index.get_location('e3') == ('seg2', 0)
        assert index.get_location('nonexistent') is None

        # Check segment events
        seg1_events = index.get_segment_events('seg1')
        assert 'e1' in seg1_events
        assert 'e2' in seg1_events

    def test_batching_writer(self, temp_store_dir):
        """BatchingWriter should batch writes correctly."""
        from cortical.cel.performance.streaming_store import BatchingWriter, StoreConfig

        flushed_events = []

        def on_flush(events):
            flushed_events.extend(events)

        config = StoreConfig(batch_size=5)
        writer = BatchingWriter(temp_store_dir, config, on_flush)

        events = []
        for i in range(12):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'index': i},
                concepts=(),
            )
            writer.write(event)
            events.append(event)

        writer.flush()

        # All events should have been flushed
        assert len(flushed_events) == 12
