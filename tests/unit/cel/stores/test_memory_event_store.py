"""
Tests for MemoryEventStore.

Tests the in-memory CEL EventStore implementation.
"""

import pytest
from datetime import datetime, timezone, timedelta

from cortical.cel.stores import MemoryEventStore
from cortical.cel.core.events import Observation, MetaCognition, EventType
from cortical.cel.core.references import MerkleRoot


class TestMemoryEventStoreBasics:
    """Basic operations."""

    def test_empty_store(self):
        """Empty store has no events."""
        store = MemoryEventStore()
        assert store.count == 0
        assert len(store) == 0
        assert store.latest() is None
        assert store.heads() == []

    def test_append_returns_merkle_root(self):
        """Appending an event returns its Merkle root."""
        store = MemoryEventStore()
        event = Observation(
            content={'type': 'test'},
            concepts=('test',),
        )
        root = store.append(event)

        assert isinstance(root, MerkleRoot)
        assert root.value == event.id

    def test_append_is_idempotent(self):
        """Appending same event twice doesn't duplicate."""
        store = MemoryEventStore()
        event = Observation(
            content={'type': 'test'},
            concepts=('test',),
        )
        root1 = store.append(event)
        root2 = store.append(event)

        assert root1.value == root2.value
        assert store.count == 1

    def test_get_returns_event(self):
        """Can retrieve appended event by ID."""
        store = MemoryEventStore()
        event = Observation(
            content={'type': 'test', 'value': 42},
            concepts=('test',),
        )
        root = store.append(event)

        retrieved = store.get(root.value)
        assert retrieved is not None
        assert retrieved.content['value'] == 42

    def test_get_missing_returns_none(self):
        """Getting non-existent event returns None."""
        store = MemoryEventStore()
        assert store.get('nonexistent') is None

    def test_contains(self):
        """Can check if event exists."""
        store = MemoryEventStore()
        event = Observation(content={'type': 'test'}, concepts=('test',))
        root = store.append(event)

        assert root.value in store
        assert 'nonexistent' not in store


class TestMemoryEventStoreIteration:
    """Iteration and querying."""

    def test_iterate_all_events(self):
        """Can iterate all events in order."""
        store = MemoryEventStore()

        events = []
        for i in range(5):
            event = Observation(
                content={'index': i},
                concepts=('test',),
            )
            store.append(event)
            events.append(event)

        retrieved = list(store.iterate())
        assert len(retrieved) == 5
        for i, event in enumerate(retrieved):
            assert event.content['index'] == i

    def test_iterate_with_from_event(self):
        """Can iterate starting after a specific event."""
        store = MemoryEventStore()

        roots = []
        for i in range(5):
            event = Observation(content={'index': i}, concepts=('test',))
            roots.append(store.append(event))

        # Start after event 2 (index 2)
        retrieved = list(store.iterate(from_event=roots[2].value))
        assert len(retrieved) == 2  # Events 3 and 4
        assert retrieved[0].content['index'] == 3
        assert retrieved[1].content['index'] == 4

    def test_iterate_with_to_event(self):
        """Can iterate up to a specific event."""
        store = MemoryEventStore()

        roots = []
        for i in range(5):
            event = Observation(content={'index': i}, concepts=('test',))
            roots.append(store.append(event))

        # Stop at event 2 (index 2)
        retrieved = list(store.iterate(to_event=roots[2].value))
        assert len(retrieved) == 3  # Events 0, 1, 2
        assert retrieved[-1].content['index'] == 2

    def test_iterate_with_event_types(self):
        """Can filter by event type."""
        store = MemoryEventStore()

        # Add mixed event types
        obs = Observation(content={'type': 'obs'}, concepts=('test',))
        meta = MetaCognition(
            observation_type='test',
            metrics={},
            conclusions=[],
            actions_triggered=[],
        )
        store.append(obs)
        store.append(meta)

        # Filter to observations only
        observations = list(store.iterate(event_types=[EventType.OBSERVATION]))
        assert len(observations) == 1
        assert observations[0].event_type == EventType.OBSERVATION

        # Filter to metacognition only
        metacognitions = list(store.iterate(event_types=[EventType.METACOGNITION]))
        assert len(metacognitions) == 1
        assert metacognitions[0].event_type == EventType.METACOGNITION


class TestMemoryEventStoreHeadsAndHorizon:
    """Head and horizon operations."""

    def test_single_event_is_head(self):
        """Single event is a head."""
        store = MemoryEventStore()
        event = Observation(content={'type': 'test'}, concepts=('test',))
        root = store.append(event)

        heads = store.heads()
        assert len(heads) == 1
        assert heads[0].value == root.value

    def test_latest_returns_most_recent(self):
        """Latest returns most recently appended event."""
        store = MemoryEventStore()

        for i in range(5):
            event = Observation(content={'index': i}, concepts=('test',))
            root = store.append(event)

        latest = store.latest()
        assert latest is not None
        retrieved = store.get(latest.value)
        assert retrieved.content['index'] == 4

    def test_horizon_returns_latest(self):
        """Horizon points to latest event."""
        store = MemoryEventStore()
        event = Observation(content={'type': 'test'}, concepts=('test',))
        root = store.append(event)

        horizon = store.horizon()
        assert horizon.event_id == root.value
        assert horizon.is_head is True

    def test_horizon_genesis_when_empty(self):
        """Empty store horizon is GENESIS."""
        store = MemoryEventStore()
        horizon = store.horizon()
        assert horizon.event_id == "GENESIS"
        assert horizon.is_head is True


class TestMemoryEventStoreTimeRange:
    """Time-based queries."""

    def test_events_in_range(self):
        """Can query events by time range."""
        store = MemoryEventStore()

        # Add events (they'll have slightly different timestamps)
        for i in range(3):
            event = Observation(content={'index': i}, concepts=('test',))
            store.append(event)

        # Query all events (wide range)
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=1)
        events = list(store.events_in_range(start_time=start, end_time=end))
        assert len(events) == 3


class TestMemoryEventStoreStats:
    """Statistics and utility methods."""

    def test_stats_returns_counts(self):
        """Stats returns event counts by type."""
        store = MemoryEventStore()

        # Add mixed events
        for i in range(3):
            store.append(Observation(content={'i': i}, concepts=('test',)))
        store.append(MetaCognition(
            observation_type='test',
            metrics={},
            conclusions=[],
            actions_triggered=[],
        ))

        stats = store.stats
        assert stats['event_count'] == 4
        assert stats['events_by_type']['OBSERVATION'] == 3
        assert stats['events_by_type']['METACOGNITION'] == 1

    def test_clear_removes_all_events(self):
        """Clear removes all events."""
        store = MemoryEventStore()

        for i in range(5):
            store.append(Observation(content={'i': i}, concepts=('test',)))

        assert store.count == 5
        cleared = store.clear()
        assert cleared == 5
        assert store.count == 0
        assert store.latest() is None

    def test_repr(self):
        """Has useful repr."""
        store = MemoryEventStore()
        store.append(Observation(content={}, concepts=('test',)))

        repr_str = repr(store)
        assert 'MemoryEventStore' in repr_str
        assert 'events=1' in repr_str


# =============================================================================
# CEL INTEGRATION TESTS - LatticeBuilder
# =============================================================================


class TestMemoryEventStoreLatticeIntegration:
    """Test MemoryEventStore works with CEL's LatticeBuilder."""

    def test_lattice_builder_with_memory_store(self):
        """LatticeBuilder.with_storage(MemoryEventStore) works."""
        from cortical.cel.container import LatticeBuilder

        lattice = (
            LatticeBuilder()
            .with_storage(MemoryEventStore)
            .build()
        )

        assert lattice is not None
        assert lattice.event_store is not None
        assert isinstance(lattice.event_store, MemoryEventStore)

    def test_lattice_append_and_retrieve(self):
        """Can append and retrieve events through lattice."""
        from cortical.cel.container import LatticeBuilder

        lattice = (
            LatticeBuilder()
            .with_storage(MemoryEventStore)
            .build()
        )

        event = Observation(
            content={'type': 'test', 'value': 42},
            concepts=('integration', 'test'),
        )
        root = lattice.event_store.append(event)

        retrieved = lattice.event_store.get(root.value)
        assert retrieved is not None
        assert retrieved.content['value'] == 42

    def test_lattice_current_horizon(self):
        """Lattice horizon updates as events are added."""
        from cortical.cel.container import LatticeBuilder

        lattice = (
            LatticeBuilder()
            .with_storage(MemoryEventStore)
            .build()
        )

        # Empty store has GENESIS horizon
        horizon = lattice.current_horizon
        assert horizon.event_id == "GENESIS"

        # Add event
        event = Observation(content={'test': 1}, concepts=('test',))
        root = lattice.event_store.append(event)

        # Horizon should now point to the event
        horizon = lattice.current_horizon
        assert horizon.event_id == root.value
        assert horizon.is_head is True

    def test_lattice_with_materializer(self):
        """MemoryEventStore works with CachingMaterializer."""
        from cortical.cel.container import Container, CognitiveLatticeImpl
        from cortical.cel.core.protocols import EventStore, Materializer
        from cortical.cel.wisdom.materializer import (
            CachingMaterializer,
            default_reducer_registry,
        )

        # Create store and materializer manually (LatticeBuilder doesn't
        # auto-wire reducer_registry)
        store = MemoryEventStore()
        reducers = default_reducer_registry()
        materializer = CachingMaterializer(
            event_store=store,
            reducer_registry=reducers,
        )

        container = Container()
        container.register_instance(EventStore, store)
        container.register_instance(Materializer, materializer)

        lattice = CognitiveLatticeImpl(container)

        assert lattice.event_store is not None
        assert lattice.materializer is not None

        # Add events and verify materializer can process them
        for i in range(5):
            event = Observation(
                content={'entity_id': f'TEST-{i}', 'value': i},
                concepts=('test',),
            )
            lattice.event_store.append(event)

        # Materializer should be able to iterate the events
        assert lattice.event_store.count == 5

    def test_container_register_instance(self):
        """Can register MemoryEventStore instance directly."""
        from cortical.cel.container import Container, CognitiveLatticeImpl
        from cortical.cel.core.protocols import EventStore

        store = MemoryEventStore()

        # Pre-populate with data
        event = Observation(content={'pre': 'existing'}, concepts=('test',))
        store.append(event)

        container = Container()
        container.register_instance(EventStore, store)

        lattice = CognitiveLatticeImpl(container)

        # Should have our pre-populated event
        assert lattice.event_store.count == 1
        assert lattice.event_store is store

    def test_event_store_protocol_compliance(self):
        """MemoryEventStore satisfies EventStore protocol."""
        from cortical.cel.core.protocols import EventStore

        store = MemoryEventStore()

        # Verify protocol compliance via isinstance (runtime_checkable)
        assert isinstance(store, EventStore)

    def test_causal_chain_through_lattice(self):
        """Causal relationships work through lattice."""
        from cortical.cel.container import LatticeBuilder

        lattice = (
            LatticeBuilder()
            .with_storage(MemoryEventStore)
            .build()
        )

        # Create causal chain: root -> child -> grandchild
        root = Observation(content={'level': 0}, concepts=('root',))
        root_hash = lattice.event_store.append(root)

        child = Observation(
            content={'level': 1},
            concepts=('child',),
            causal_parents=[root_hash.value],
        )
        child_hash = lattice.event_store.append(child)

        grandchild = Observation(
            content={'level': 2},
            concepts=('grandchild',),
            causal_parents=[child_hash.value],
        )
        grandchild_hash = lattice.event_store.append(grandchild)

        # Verify ancestors
        ancestors = list(lattice.event_store.ancestors(grandchild_hash.value))
        ancestor_levels = [a.content['level'] for a in ancestors]
        assert 1 in ancestor_levels  # child
        assert 0 in ancestor_levels  # root

        # Verify descendants
        descendants = list(lattice.event_store.descendants(root_hash.value))
        descendant_levels = [d.content['level'] for d in descendants]
        assert 1 in descendant_levels  # child
        assert 2 in descendant_levels  # grandchild

    def test_iterate_with_event_type_filter(self):
        """Event type filtering works through lattice."""
        from cortical.cel.container import LatticeBuilder

        lattice = (
            LatticeBuilder()
            .with_storage(MemoryEventStore)
            .build()
        )

        # Add mixed event types
        obs = Observation(content={'type': 'obs'}, concepts=('test',))
        meta = MetaCognition(
            observation_type='self_check',
            metrics={'health': 1.0},
            conclusions=['all good'],
            actions_triggered=[],
        )

        lattice.event_store.append(obs)
        lattice.event_store.append(meta)

        # Filter by type
        observations = list(lattice.event_store.iterate(
            event_types=[EventType.OBSERVATION]
        ))
        assert len(observations) == 1
        assert observations[0].event_type == EventType.OBSERVATION

        metacognitions = list(lattice.event_store.iterate(
            event_types=[EventType.METACOGNITION]
        ))
        assert len(metacognitions) == 1
        assert metacognitions[0].event_type == EventType.METACOGNITION
