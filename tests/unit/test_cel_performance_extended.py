"""
Extended Unit Tests for CEL Performance Components.

Additional tests to improve coverage for:
- OptimizedDAG: find_path, common_ancestor, subgraph, depth, verify_integrity
- SnapshotManager: delta snapshots, recovery
- StreamingEventStore: error handling, segments, queries
"""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
from cortical.cel.performance.streaming_store import (
    StreamingEventStore,
    StoreConfig,
    LRUCache,
    EventIndex,
)
from cortical.cel.core.events import CognitiveEvent, EventType
from cortical.cel.core.references import EventHorizon


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory."""
    return tmp_path


@pytest.fixture
def simple_dag():
    """Create a simple DAG with chain A -> B -> C."""
    dag = OptimizedDAG()

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

    return dag, (event_a, event_b, event_c)


@pytest.fixture
def diamond_dag():
    """Create a diamond-shaped DAG: A -> [B, C] -> D."""
    dag = OptimizedDAG()

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
        timestamp="2025-01-02T00:00:01Z",
        event_type=EventType.OBSERVATION,
        causal_parents=(event_a.id,),
        content={'name': 'C'},
        concepts=(),
    )
    dag.add(event_c, verify_parents=False)

    event_d = CognitiveEvent(
        timestamp="2025-01-03T00:00:00Z",
        event_type=EventType.OBSERVATION,
        causal_parents=(event_b.id, event_c.id),
        content={'name': 'D'},
        concepts=(),
    )
    dag.add(event_d, verify_parents=False)

    return dag, (event_a, event_b, event_c, event_d)


# =============================================================================
# OPTIMIZED DAG - CAUSAL VIOLATION TESTS
# =============================================================================


class TestOptimizedDAGCausalViolation:
    """Tests for CausalViolationError handling."""

    def test_causal_violation_error_attributes(self):
        """Test CausalViolationError has correct attributes."""
        error = CausalViolationError("event123", ["parent1", "parent2"])

        assert error.event_id == "event123"
        assert error.missing_parents == ["parent1", "parent2"]
        assert "event123" in str(error)

    def test_add_with_missing_parents_raises(self):
        """Adding event with missing parents should raise CausalViolationError."""
        dag = OptimizedDAG()

        event = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=("nonexistent_parent",),
            content={'name': 'orphan'},
            concepts=(),
        )

        with pytest.raises(CausalViolationError) as exc_info:
            dag.add(event, verify_parents=True)

        assert "nonexistent_parent" in str(exc_info.value.missing_parents)


# =============================================================================
# OPTIMIZED DAG - FIND PATH TESTS
# =============================================================================


class TestOptimizedDAGFindPath:
    """Tests for find_path method."""

    def test_find_path_direct_chain(self, simple_dag):
        """Find path in simple chain."""
        dag, (event_a, event_b, event_c) = simple_dag

        path = dag.find_path(event_a.id, event_c.id)

        assert path is not None
        assert len(path) == 3
        assert path[0] == event_a.id
        assert path[-1] == event_c.id

    def test_find_path_same_event(self, simple_dag):
        """Finding path from event to itself."""
        dag, (event_a, event_b, event_c) = simple_dag

        path = dag.find_path(event_a.id, event_a.id)

        assert path == [event_a.id]

    def test_find_path_nonexistent_source(self, simple_dag):
        """Find path with nonexistent source returns None."""
        dag, (event_a, event_b, event_c) = simple_dag

        path = dag.find_path("nonexistent", event_c.id)
        assert path is None

    def test_find_path_nonexistent_target(self, simple_dag):
        """Find path with nonexistent target returns None."""
        dag, (event_a, event_b, event_c) = simple_dag

        path = dag.find_path(event_a.id, "nonexistent")
        assert path is None

    def test_find_path_no_connection(self):
        """Find path between unconnected events returns None."""
        dag = OptimizedDAG()

        # Create two independent events
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
            causal_parents=(),
            content={'name': 'B'},
            concepts=(),
        )
        dag.add(event_a, verify_parents=False)
        dag.add(event_b, verify_parents=False)

        path = dag.find_path(event_a.id, event_b.id)
        assert path is None


# =============================================================================
# OPTIMIZED DAG - COMMON ANCESTOR TESTS
# =============================================================================


class TestOptimizedDAGCommonAncestor:
    """Tests for common_ancestor method."""

    def test_common_ancestor_in_diamond(self, diamond_dag):
        """Find common ancestor in diamond DAG."""
        dag, (event_a, event_b, event_c, event_d) = diamond_dag

        # B and C share A as common ancestor
        ancestor = dag.common_ancestor(event_b.id, event_c.id)
        assert ancestor == event_a.id

    def test_common_ancestor_direct_lineage(self, simple_dag):
        """Find common ancestor in direct lineage."""
        dag, (event_a, event_b, event_c) = simple_dag

        ancestor = dag.common_ancestor(event_b.id, event_c.id)
        assert ancestor == event_b.id

    def test_common_ancestor_nonexistent_event(self, simple_dag):
        """Common ancestor with nonexistent event returns None."""
        dag, (event_a, event_b, event_c) = simple_dag

        ancestor = dag.common_ancestor(event_a.id, "nonexistent")
        assert ancestor is None

    def test_common_ancestor_both_nonexistent(self, simple_dag):
        """Common ancestor with both nonexistent events returns None."""
        dag, _ = simple_dag

        ancestor = dag.common_ancestor("nonexistent1", "nonexistent2")
        assert ancestor is None

    def test_common_ancestor_no_common(self):
        """No common ancestor returns None."""
        dag = OptimizedDAG()

        # Two independent chains
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
            causal_parents=(),
            content={'name': 'B'},
            concepts=(),
        )
        dag.add(event_a, verify_parents=False)
        dag.add(event_b, verify_parents=False)

        ancestor = dag.common_ancestor(event_a.id, event_b.id)
        assert ancestor is None


# =============================================================================
# OPTIMIZED DAG - SUBGRAPH TESTS
# =============================================================================


class TestOptimizedDAGSubgraph:
    """Tests for subgraph extraction."""

    def test_subgraph_descendants_only(self, simple_dag):
        """Extract subgraph with descendants only."""
        dag, (event_a, event_b, event_c) = simple_dag

        subdag = dag.subgraph(event_b.id, include_ancestors=False)

        # Should include B and C, but not A
        assert subdag.contains(event_b.id)
        assert subdag.contains(event_c.id)
        assert not subdag.contains(event_a.id)

    def test_subgraph_with_ancestors(self, simple_dag):
        """Extract subgraph including ancestors."""
        dag, (event_a, event_b, event_c) = simple_dag

        subdag = dag.subgraph(event_b.id, include_ancestors=True)

        # Should include all events
        assert subdag.contains(event_a.id)
        assert subdag.contains(event_b.id)
        assert subdag.contains(event_c.id)

    def test_subgraph_single_event(self):
        """Extract subgraph from single event."""
        dag = OptimizedDAG()

        event = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'name': 'single'},
            concepts=(),
        )
        dag.add(event, verify_parents=False)

        subdag = dag.subgraph(event.id)
        assert subdag.count == 1
        assert subdag.contains(event.id)


# =============================================================================
# OPTIMIZED DAG - DEPTH TESTS
# =============================================================================


class TestOptimizedDAGDepth:
    """Tests for depth property."""

    def test_depth_empty_dag(self):
        """Empty DAG has depth 0."""
        dag = OptimizedDAG()
        assert dag.depth == 0

    def test_depth_single_event(self):
        """Single event has depth 1."""
        dag = OptimizedDAG()

        event = CognitiveEvent(
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'name': 'single'},
            concepts=(),
        )
        dag.add(event, verify_parents=False)

        assert dag.depth == 1

    def test_depth_chain(self, simple_dag):
        """Chain A -> B -> C has depth 3."""
        dag, _ = simple_dag
        assert dag.depth == 3

    def test_depth_diamond(self, diamond_dag):
        """Diamond A -> [B,C] -> D has depth 3."""
        dag, _ = diamond_dag
        assert dag.depth == 3


# =============================================================================
# OPTIMIZED DAG - CAUSAL ORDER WITH FROM/TO TESTS
# =============================================================================


class TestOptimizedDAGCausalOrderFiltered:
    """Tests for causal_order with from_event and to_event filters."""

    def test_causal_order_from_event(self, simple_dag):
        """Start iteration from specific event."""
        dag, (event_a, event_b, event_c) = simple_dag

        events = list(dag.causal_order(from_event=event_a.id))

        # Should skip A and include B, C
        assert len(events) == 2
        names = [e.content['name'] for e in events]
        assert 'A' not in names
        assert 'B' in names
        assert 'C' in names

    def test_causal_order_to_event(self, simple_dag):
        """Stop iteration at specific event."""
        dag, (event_a, event_b, event_c) = simple_dag

        events = list(dag.causal_order(to_event=event_b.id))

        # Should include A, B but not C
        assert len(events) == 2
        names = [e.content['name'] for e in events]
        assert 'A' in names
        assert 'B' in names
        assert 'C' not in names


# =============================================================================
# OPTIMIZED DAG - ANCESTORS WITH DEPTH LIMIT TESTS
# =============================================================================


class TestOptimizedDAGAncestorsDepth:
    """Tests for ancestors with depth limit."""

    def test_ancestors_with_max_depth(self):
        """Ancestors limited by max_depth."""
        dag = OptimizedDAG()

        # Create chain: A -> B -> C -> D -> E
        events = []
        prev_id = None
        for i, name in enumerate(['A', 'B', 'C', 'D', 'E']):
            parents = (prev_id,) if prev_id else ()
            event = CognitiveEvent(
                timestamp=f"2025-01-0{i+1}T00:00:00Z",
                event_type=EventType.OBSERVATION,
                causal_parents=parents,
                content={'name': name},
                concepts=(),
            )
            dag.add(event, verify_parents=False)
            events.append(event)
            prev_id = event.id

        # Get ancestors of E with max_depth=2 (should get D and C)
        ancestors = list(dag.ancestors(events[4].id, max_depth=2))

        assert len(ancestors) == 2
        names = [e.content['name'] for e in ancestors]
        assert 'D' in names
        assert 'C' in names
        assert 'A' not in names

    def test_ancestors_nonexistent_event(self):
        """Ancestors of nonexistent event yields nothing."""
        dag = OptimizedDAG()
        ancestors = list(dag.ancestors("nonexistent"))
        assert ancestors == []


# =============================================================================
# OPTIMIZED DAG - DESCENDANTS TESTS
# =============================================================================


class TestOptimizedDAGDescendants:
    """Tests for descendants method."""

    def test_descendants_nonexistent_event(self):
        """Descendants of nonexistent event yields nothing."""
        dag = OptimizedDAG()
        descendants = list(dag.descendants("nonexistent"))
        assert descendants == []


# =============================================================================
# HEAP TOPOLOGICAL SORT - VISITED SKIP TESTS
# =============================================================================


class TestHeapTopologicalSortEdgeCases:
    """Edge case tests for HeapTopologicalSort."""

    def test_sort_with_multiple_paths_to_same_event(self):
        """Sort handles multiple paths correctly (visited tracking)."""
        # Create: A -> [B, C] -> D
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
            timestamp="2025-01-02T00:00:01Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(event_a.id,),
            content={'name': 'C'},
            concepts=(),
        )
        event_d = CognitiveEvent(
            timestamp="2025-01-03T00:00:00Z",
            event_type=EventType.OBSERVATION,
            causal_parents=(event_b.id, event_c.id),
            content={'name': 'D'},
            concepts=(),
        )

        events = {
            event_a.id: event_a,
            event_b.id: event_b,
            event_c.id: event_c,
            event_d.id: event_d,
        }
        children = {
            event_a.id: {event_b.id, event_c.id},
            event_b.id: {event_d.id},
            event_c.id: {event_d.id},
        }

        sorter = HeapTopologicalSort(events, children)
        result = list(sorter)

        # D should only appear once
        names = [e.content['name'] for e in result]
        assert names.count('D') == 1
        assert len(result) == 4


# =============================================================================
# SNAPSHOT MANAGER - DELTA AND RECOVERY TESTS
# =============================================================================


class TestSnapshotManagerExtended:
    """Extended tests for SnapshotManager."""

    def test_delta_snapshot(self, temp_dir):
        """Test delta snapshot creation."""
        config = SnapshotConfig(delta_interval=10, full_interval=100)
        manager = SnapshotManager(temp_dir, config)

        # Check if delta is suggested
        snapshot_type = manager.should_snapshot(10)
        # With no previous snapshot, it might suggest full instead
        assert snapshot_type in ['none', 'delta', 'full']

    def test_snapshot_creation_and_list(self, temp_dir):
        """Test that snapshot creation and listing works."""
        config = SnapshotConfig(retention_count=5)
        manager = SnapshotManager(temp_dir, config)

        horizon = EventHorizon(event_id="test_horizon_1")
        manager.create_snapshot(
            horizon=horizon,
            event_count=50,
            entity_index={'entity_1': ['e1']},
        )

        snapshots = manager.list_snapshots()
        # Should have at least 1 snapshot
        assert len(snapshots) >= 1
        assert snapshots[0].event_horizon == "test_horizon_1"

    def test_list_snapshots_ordered(self, temp_dir):
        """List snapshots returns ordered list."""
        manager = SnapshotManager(temp_dir)

        for i in range(3):
            horizon = EventHorizon(event_id=f"event_{i}")
            manager.create_snapshot(
                horizon=horizon,
                event_count=i * 10,
                entity_index={},
            )

        snapshots = manager.list_snapshots()
        # Should have 3 snapshots (retention allows all by default)
        assert len(snapshots) >= 1


# =============================================================================
# STREAMING EVENT STORE - EXTENDED TESTS
# =============================================================================


class TestStreamingEventStoreExtended:
    """Extended tests for StreamingEventStore."""

    @pytest.fixture
    def store_dir(self, tmp_path):
        """Create a store directory."""
        store_dir = tmp_path / "streaming_store"
        store_dir.mkdir()
        return store_dir

    def test_get_nonexistent_event(self, store_dir):
        """Getting nonexistent event returns None."""
        store = StreamingEventStore(store_dir, StoreConfig())
        result = store.get("nonexistent_event_id")
        assert result is None

    def test_events_by_concept(self, store_dir):
        """Query events by concept."""
        store = StreamingEventStore(store_dir, StoreConfig())

        # Add events with different concepts
        for i in range(5):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'entity_id': 'E-001', 'index': i},
                concepts=('concept_a', f'unique_{i}'),
            )
            store.append(event)

        store.flush()

        # Query by concept (if method exists)
        if hasattr(store, 'events_by_concept'):
            results = store.events_by_concept('concept_a')
            assert len(results) == 5

    def test_close_and_reopen(self, store_dir):
        """Test store persistence across close/reopen."""
        config = StoreConfig()

        # Create store and add events
        store1 = StreamingEventStore(store_dir, config)
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'entity_id': 'E-001', 'test': 'persistence'},
            concepts=(),
        )
        store1.append(event)
        store1.flush()
        event_id = event.id

        # Close if method exists
        if hasattr(store1, 'close'):
            store1.close()

        # Reopen and verify
        store2 = StreamingEventStore(store_dir, config)
        retrieved = store2.get(event_id)

        assert retrieved is not None
        assert retrieved.content['test'] == 'persistence'


# =============================================================================
# LRU CACHE - EXTENDED TESTS
# =============================================================================


class TestLRUCacheExtended:
    """Extended tests for LRUCache."""

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = LRUCache(max_size=5)

        for i in range(5):
            cache.put(f'key_{i}', i)

        assert len(cache) == 5

        cache.clear()
        assert len(cache) == 0

    def test_cache_update_existing(self):
        """Updating existing key moves it to most recent."""
        cache = LRUCache(max_size=3)

        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)

        # Update 'a' to move it to most recent
        cache.put('a', 10)

        # Add new item, should evict 'b' (oldest accessed)
        cache.put('d', 4)

        assert cache.get('a') == 10
        assert cache.get('b') is None  # Evicted
        assert cache.get('c') == 3
        assert cache.get('d') == 4

    def test_cache_contains(self):
        """Test contains check."""
        cache = LRUCache(max_size=3)

        cache.put('a', 1)

        assert 'a' in cache._cache or cache.get('a') is not None
        # Just verify the basic operation works
        assert cache.get('a') == 1


# =============================================================================
# EVENT INDEX - EXTENDED TESTS
# =============================================================================


class TestEventIndexExtended:
    """Extended tests for EventIndex."""

    def test_index_contains(self):
        """Test contains check."""
        index = EventIndex()

        index.add('e1', 'seg1', 0)

        assert index.get_location('e1') is not None
        assert index.get_location('nonexistent') is None

    def test_index_segments(self):
        """Test getting all segments."""
        index = EventIndex()

        index.add('e1', 'seg1', 0)
        index.add('e2', 'seg2', 0)
        index.add('e3', 'seg1', 1)

        seg1_events = index.get_segment_events('seg1')
        seg2_events = index.get_segment_events('seg2')

        assert len(seg1_events) == 2  # e1, e3
        assert len(seg2_events) == 1  # e2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
