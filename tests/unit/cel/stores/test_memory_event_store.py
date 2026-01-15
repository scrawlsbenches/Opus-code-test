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
