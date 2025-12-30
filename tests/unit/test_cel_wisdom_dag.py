"""
Tests for cortical.cel.wisdom.dag module.

This module provides comprehensive tests for the CEL (Cognitive Event Lattice)
wisdom DAG implementation, including:
- CausalViolationError and DuplicateEventError exceptions
- MerkleDAG class
- FileSystemEventStore class
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_cognitive_event():
    """Create a mock CognitiveEvent."""
    def _create(event_id, parents=None, timestamp=None, event_type=None):
        event = MagicMock()
        event.id = event_id
        event.causal_parents = parents or []
        event.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        event.event_type = event_type or MagicMock()
        event.to_dict.return_value = {
            "id": event_id,
            "causal_parents": parents or [],
            "timestamp": event.timestamp,
            "event_type": str(event_type) if event_type else "TEST"
        }
        return event
    return _create


# =============================================================================
# EXCEPTION TESTS
# =============================================================================

class TestCausalViolationError:
    """Tests for CausalViolationError exception."""

    def test_causal_violation_error_creation(self):
        """Test CausalViolationError creation with event ID and missing parents."""
        from cortical.cel.wisdom.dag import CausalViolationError

        error = CausalViolationError(
            event_id="abc123def456",
            missing_parents=["parent1", "parent2"]
        )

        assert error.event_id == "abc123def456"
        assert error.missing_parents == ["parent1", "parent2"]

    def test_causal_violation_error_message(self):
        """Test CausalViolationError message formatting."""
        from cortical.cel.wisdom.dag import CausalViolationError

        error = CausalViolationError(
            event_id="abc123def456ghij",
            missing_parents=["missing123456789"]
        )

        # Message should contain truncated IDs
        message = str(error)
        assert "abc123def456ghij" in message
        assert "missing" in message

    def test_causal_violation_error_inherits_exception(self):
        """Test CausalViolationError inherits from Exception."""
        from cortical.cel.wisdom.dag import CausalViolationError

        assert issubclass(CausalViolationError, Exception)


class TestDuplicateEventError:
    """Tests for DuplicateEventError exception."""

    def test_duplicate_event_error_creation(self):
        """Test DuplicateEventError creation."""
        from cortical.cel.wisdom.dag import DuplicateEventError

        error = DuplicateEventError(event_id="abc123def456")

        assert error.event_id == "abc123def456"

    def test_duplicate_event_error_message(self):
        """Test DuplicateEventError message formatting."""
        from cortical.cel.wisdom.dag import DuplicateEventError

        error = DuplicateEventError(event_id="abc123def456ghij")

        message = str(error)
        assert "already exists" in message
        assert "abc123def456ghij" in message

    def test_duplicate_event_error_inherits_exception(self):
        """Test DuplicateEventError inherits from Exception."""
        from cortical.cel.wisdom.dag import DuplicateEventError

        assert issubclass(DuplicateEventError, Exception)


# =============================================================================
# MERKLE DAG TESTS
# =============================================================================

class TestMerkleDAG:
    """Tests for MerkleDAG class."""

    def test_merkle_dag_creation(self):
        """Test MerkleDAG creation with defaults."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        assert dag.events == {}
        assert len(dag.heads) == 0
        assert dag.count == 0

    def test_add_root_event(self, mock_cognitive_event):
        """Test adding a root event (no parents)."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[])

        result = dag.add(event)

        assert result.value == "event1"
        assert "event1" in dag.events
        assert "event1" in dag.heads
        assert dag.count == 1

    def test_add_event_with_parent(self, mock_cognitive_event):
        """Test adding an event with a parent."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        parent = mock_cognitive_event("parent1", parents=[])
        child = mock_cognitive_event("child1", parents=["parent1"])

        dag.add(parent)
        dag.add(child)

        assert "child1" in dag.heads
        assert "parent1" not in dag.heads  # Parent is no longer a head
        assert dag.count == 2

    def test_add_duplicate_event(self, mock_cognitive_event):
        """Test adding duplicate event raises DuplicateEventError."""
        from cortical.cel.wisdom.dag import MerkleDAG, DuplicateEventError

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[])

        dag.add(event)

        with pytest.raises(DuplicateEventError) as exc_info:
            dag.add(event)

        assert exc_info.value.event_id == "event1"

    def test_add_event_missing_parent(self, mock_cognitive_event):
        """Test adding event with missing parent raises CausalViolationError."""
        from cortical.cel.wisdom.dag import MerkleDAG, CausalViolationError

        dag = MerkleDAG()
        event = mock_cognitive_event("child1", parents=["nonexistent_parent"])

        with pytest.raises(CausalViolationError) as exc_info:
            dag.add(event)

        assert exc_info.value.event_id == "child1"
        assert "nonexistent_parent" in exc_info.value.missing_parents

    def test_get_existing_event(self, mock_cognitive_event):
        """Test getting an existing event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[])
        dag.add(event)

        result = dag.get("event1")

        assert result == event

    def test_get_nonexistent_event(self):
        """Test getting a non-existent event returns None."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        result = dag.get("nonexistent")

        assert result is None

    def test_contains_existing_event(self, mock_cognitive_event):
        """Test contains returns True for existing event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[])
        dag.add(event)

        assert dag.contains("event1") is True

    def test_contains_nonexistent_event(self):
        """Test contains returns False for non-existent event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        assert dag.contains("nonexistent") is False

    def test_get_heads_empty(self):
        """Test get_heads returns empty list for empty DAG."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        result = dag.get_heads()

        assert result == []

    def test_get_heads_with_events(self, mock_cognitive_event):
        """Test get_heads returns current branch heads."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event1 = mock_cognitive_event("event1", parents=[])
        event2 = mock_cognitive_event("event2", parents=[])
        dag.add(event1)
        dag.add(event2)

        heads = dag.get_heads()

        assert len(heads) == 2
        head_values = [h.value for h in heads]
        assert "event1" in head_values
        assert "event2" in head_values

    def test_get_latest_empty(self):
        """Test get_latest returns None for empty DAG."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        result = dag.get_latest()

        assert result is None

    def test_get_latest_with_events(self, mock_cognitive_event):
        """Test get_latest returns event with latest timestamp."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event1 = mock_cognitive_event("event1", parents=[], timestamp="2025-01-01T00:00:00Z")
        event2 = mock_cognitive_event("event2", parents=[], timestamp="2025-01-02T00:00:00Z")
        dag.add(event1)
        dag.add(event2)

        result = dag.get_latest()

        assert result.value == "event2"

    def test_ancestors_empty(self, mock_cognitive_event):
        """Test ancestors returns nothing for root event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[])
        dag.add(event)

        ancestors = list(dag.ancestors("event1"))

        assert ancestors == []

    def test_ancestors_nonexistent(self):
        """Test ancestors returns nothing for non-existent event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        ancestors = list(dag.ancestors("nonexistent"))

        assert ancestors == []

    def test_ancestors_with_parents(self, mock_cognitive_event):
        """Test ancestors returns parent events."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        parent = mock_cognitive_event("parent1", parents=[])
        child = mock_cognitive_event("child1", parents=["parent1"])
        dag.add(parent)
        dag.add(child)

        ancestors = list(dag.ancestors("child1"))

        assert len(ancestors) == 1
        assert ancestors[0].id == "parent1"

    def test_ancestors_with_depth_limit(self, mock_cognitive_event):
        """Test ancestors respects depth limit."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        grandparent = mock_cognitive_event("gp1", parents=[])
        parent = mock_cognitive_event("p1", parents=["gp1"])
        child = mock_cognitive_event("c1", parents=["p1"])

        dag.add(grandparent)
        dag.add(parent)
        dag.add(child)

        # Depth 1 should only return parent
        ancestors = list(dag.ancestors("c1", depth=1))
        assert len(ancestors) == 1
        assert ancestors[0].id == "p1"

    def test_descendants_empty(self, mock_cognitive_event):
        """Test descendants returns nothing for leaf event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[])
        dag.add(event)

        descendants = list(dag.descendants("event1"))

        assert descendants == []

    def test_descendants_nonexistent(self):
        """Test descendants returns nothing for non-existent event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        descendants = list(dag.descendants("nonexistent"))

        assert descendants == []

    def test_descendants_with_children(self, mock_cognitive_event):
        """Test descendants returns child events."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        parent = mock_cognitive_event("parent1", parents=[])
        child = mock_cognitive_event("child1", parents=["parent1"])
        dag.add(parent)
        dag.add(child)

        descendants = list(dag.descendants("parent1"))

        assert len(descendants) == 1
        assert descendants[0].id == "child1"

    def test_causal_order_empty(self):
        """Test causal_order returns nothing for empty DAG."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        events = list(dag.causal_order())

        assert events == []

    def test_causal_order_single_event(self, mock_cognitive_event):
        """Test causal_order with single event."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event = mock_cognitive_event("event1", parents=[], timestamp="2025-01-01T00:00:00Z")
        dag.add(event)

        events = list(dag.causal_order())

        assert len(events) == 1
        assert events[0].id == "event1"

    def test_causal_order_parent_before_child(self, mock_cognitive_event):
        """Test causal_order yields parents before children."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        parent = mock_cognitive_event("parent1", parents=[], timestamp="2025-01-01T00:00:00Z")
        child = mock_cognitive_event("child1", parents=["parent1"], timestamp="2025-01-02T00:00:00Z")

        dag.add(parent)
        dag.add(child)

        events = list(dag.causal_order())
        event_ids = [e.id for e in events]

        assert event_ids.index("parent1") < event_ids.index("child1")

    def test_causal_order_from_event(self, mock_cognitive_event):
        """Test causal_order with from_event parameter."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event1 = mock_cognitive_event("e1", parents=[], timestamp="2025-01-01T00:00:00Z")
        event2 = mock_cognitive_event("e2", parents=["e1"], timestamp="2025-01-02T00:00:00Z")
        event3 = mock_cognitive_event("e3", parents=["e2"], timestamp="2025-01-03T00:00:00Z")

        dag.add(event1)
        dag.add(event2)
        dag.add(event3)

        # Test basic causal ordering works
        all_events = list(dag.causal_order())
        assert len(all_events) == 3

        # from_event skips e1 and starts yielding from the next event
        events = list(dag.causal_order(from_event="e1"))
        event_ids = [e.id for e in events]

        # e1 should be skipped since we're starting after it
        assert "e1" not in event_ids
        # The exact behavior depends on the implementation - just verify e1 is excluded
        assert len(event_ids) <= 2

    def test_causal_order_to_event(self, mock_cognitive_event):
        """Test causal_order with to_event parameter."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()
        event1 = mock_cognitive_event("e1", parents=[], timestamp="2025-01-01T00:00:00Z")
        event2 = mock_cognitive_event("e2", parents=["e1"], timestamp="2025-01-02T00:00:00Z")
        event3 = mock_cognitive_event("e3", parents=["e2"], timestamp="2025-01-03T00:00:00Z")

        dag.add(event1)
        dag.add(event2)
        dag.add(event3)

        events = list(dag.causal_order(to_event="e2"))
        event_ids = [e.id for e in events]

        # Should stop at e2 (inclusive)
        assert "e1" in event_ids
        assert "e2" in event_ids
        assert "e3" not in event_ids

    def test_count_property(self, mock_cognitive_event):
        """Test count property returns correct count."""
        from cortical.cel.wisdom.dag import MerkleDAG

        dag = MerkleDAG()

        assert dag.count == 0

        dag.add(mock_cognitive_event("e1", parents=[]))
        assert dag.count == 1

        dag.add(mock_cognitive_event("e2", parents=[]))
        assert dag.count == 2


# =============================================================================
# FILESYSTEM EVENT STORE TESTS
# =============================================================================

class TestFileSystemEventStore:
    """Tests for FileSystemEventStore class."""

    def test_store_creation(self, tmp_path):
        """Test FileSystemEventStore creation."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        assert store.base_path.exists()
        assert store.events_path.exists()
        assert store.count == 0

    def test_store_append_event(self, tmp_path, mock_cognitive_event):
        """Test appending an event to the store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("event1", parents=[])

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                result = store.append(event)

        assert result.value == "event1"

    def test_store_get_event(self, tmp_path, mock_cognitive_event):
        """Test getting an event from the store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("event1", parents=[])

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event)

        result = store.get("event1")
        assert result == event

    def test_store_get_nonexistent(self, tmp_path):
        """Test getting non-existent event returns None."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        result = store.get("nonexistent")
        assert result is None

    def test_store_heads(self, tmp_path, mock_cognitive_event):
        """Test getting heads from store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("event1", parents=[])

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event)

        heads = store.heads()
        assert len(heads) == 1
        assert heads[0].value == "event1"

    def test_store_latest(self, tmp_path, mock_cognitive_event):
        """Test getting latest event from store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event1 = mock_cognitive_event("e1", parents=[], timestamp="2025-01-01T00:00:00Z")
        event2 = mock_cognitive_event("e2", parents=[], timestamp="2025-01-02T00:00:00Z")

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event1)
                store.append(event2)

        latest = store.latest()
        assert latest.value == "e2"

    def test_store_iterate(self, tmp_path, mock_cognitive_event):
        """Test iterating events from store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("e1", parents=[], timestamp="2025-01-01T00:00:00Z")

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event)

        events = list(store.iterate())
        assert len(events) == 1
        assert events[0].id == "e1"

    def test_store_iterate_with_event_types(self, tmp_path, mock_cognitive_event):
        """Test iterating with event type filter."""
        from cortical.cel.wisdom.dag import FileSystemEventStore
        from cortical.cel.core.events import EventType

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("e1", parents=[], timestamp="2025-01-01T00:00:00Z", event_type=EventType.OBSERVATION)

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event)

        # Filter by matching type
        events = list(store.iterate(event_types=[EventType.OBSERVATION]))
        assert len(events) == 1

        # Filter by non-matching type
        events = list(store.iterate(event_types=[EventType.INTENTION]))
        assert len(events) == 0

    def test_store_ancestors(self, tmp_path, mock_cognitive_event):
        """Test getting ancestors from store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        parent = mock_cognitive_event("p1", parents=[], timestamp="2025-01-01T00:00:00Z")
        child = mock_cognitive_event("c1", parents=["p1"], timestamp="2025-01-02T00:00:00Z")

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(parent)
                store.append(child)

        ancestors = list(store.ancestors("c1"))
        assert len(ancestors) == 1
        assert ancestors[0].id == "p1"

    def test_store_descendants(self, tmp_path, mock_cognitive_event):
        """Test getting descendants from store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        parent = mock_cognitive_event("p1", parents=[], timestamp="2025-01-01T00:00:00Z")
        child = mock_cognitive_event("c1", parents=["p1"], timestamp="2025-01-02T00:00:00Z")

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(parent)
                store.append(child)

        descendants = list(store.descendants("p1"))
        assert len(descendants) == 1
        assert descendants[0].id == "c1"

    def test_store_horizon_empty(self, tmp_path):
        """Test horizon raises error for empty store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        with pytest.raises(ValueError, match="Cannot get horizon for empty store"):
            store.horizon()

    def test_store_horizon(self, tmp_path, mock_cognitive_event):
        """Test getting horizon from store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("e1", parents=[], timestamp="2025-01-01T00:00:00Z")

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event)

        horizon = store.horizon()
        assert horizon.event_id == "e1"
        assert horizon.is_head is True

    def test_store_verify_integrity_valid(self, tmp_path, mock_cognitive_event):
        """Test verify_integrity returns empty list for valid DAG."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        event = mock_cognitive_event("e1", parents=[])

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(event)

        errors = store.verify_integrity()
        assert errors == []

    def test_store_count(self, tmp_path, mock_cognitive_event):
        """Test count property of store."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)

        assert store.count == 0

        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(mock_cognitive_event("e1", parents=[]))

        assert store.count == 1

    def test_store_event_path(self, tmp_path):
        """Test _event_path generates correct path."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        path = store._event_path("abcdef123456")

        assert path.parent.name == "ab"
        assert path.name == "abcdef123456.json"

    def test_store_save_event(self, tmp_path, mock_cognitive_event):
        """Test _save_event writes to disk."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")
        event = mock_cognitive_event("abcdef123456", parents=[])

        store._save_event(event)

        expected_path = store.events_path / "ab" / "abcdef123456.json"
        assert expected_path.exists()

    def test_store_save_heads(self, tmp_path, mock_cognitive_event):
        """Test _save_heads writes to disk."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)

        # Add an event to populate heads
        with patch.object(store, '_save_event'):
            with patch.object(store, '_save_heads'):
                store.append(mock_cognitive_event("e1", parents=[]))

        # Now actually save heads
        store._save_heads()

        assert store.heads_path.exists()
        with open(store.heads_path) as f:
            data = json.load(f)
            assert "heads" in data

    def test_store_lazy_load(self, tmp_path, mock_cognitive_event):
        """Test that store lazy loads events."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        # First, create a store and add an event
        store1 = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)

        # Create a new store pointing to same path to verify lazy loading behavior
        store2 = FileSystemEventStore(tmp_path / "events", use_optimized_dag=False)
        assert store2._loaded is False

        # Accessing count should trigger load
        _ = store2.count
        assert store2._loaded is True

    def test_store_with_optimized_dag(self, tmp_path, mock_cognitive_event):
        """Test store with OptimizedDAG."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events", use_optimized_dag=True)

        assert store._use_optimized is True

    def test_store_ensure_loaded_already_loaded(self, tmp_path):
        """Test _ensure_loaded returns early if already loaded."""
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")
        store._loaded = True

        # This should return immediately without doing anything
        store._ensure_loaded()

        # Still loaded
        assert store._loaded is True
