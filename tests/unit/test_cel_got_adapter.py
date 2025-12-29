"""
Unit tests for CEL-GoT adapter (adapters/got.py).

Tests the bridge between GoT entities and CEL events.
This module has 18% coverage - target is 70%+.
"""

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from cortical.cel.adapters.got import (
    GoTEventAdapter,
    GoTEntityAdapter,
    GotBridgeEventStore,
    create_got_bridge,
    GOT_AVAILABLE,
)
from cortical.cel.core.events import (
    CognitiveEvent,
    EventType,
    Observation,
    Intention,
)
from cortical.cel.core.references import MerkleRoot

# Skip all tests if GoT types not available
pytestmark = pytest.mark.skipif(
    not GOT_AVAILABLE,
    reason="GoT types not available"
)


# Import GoT types
if GOT_AVAILABLE:
    from cortical.got.types import (
        Task,
        Decision,
        Sprint,
        Epic,
        Edge,
        Handoff,
        Document,
        Entity,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_task():
    """Create a sample pending task."""
    return Task(
        id="T-20251229-120000-abc12345",
        title="Implement CEL bridge",
        status="pending",
        priority="high",
        description="Bridge GoT to CEL for migration",
        properties={"category": "feature", "tags": ["cel", "migration"]},
        metadata={"created_by": "test-agent"},
    )


@pytest.fixture
def completed_task():
    """Create a completed task."""
    return Task(
        id="T-20251229-110000-def67890",
        title="Add unit tests",
        status="completed",
        priority="medium",
        description="Add comprehensive unit tests",
        properties={"category": "test"},
        metadata={"completed_at": "2025-12-29T12:00:00Z"},
    )


@pytest.fixture
def blocked_task():
    """Create a blocked task."""
    return Task(
        id="T-20251229-100000-ghi11111",
        title="Deploy to production",
        status="blocked",
        priority="critical",
        description="Blocked by testing",
        properties={},
        metadata={"blocked_reason": "Waiting for CI"},
    )


@pytest.fixture
def sample_decision():
    """Create a sample decision."""
    return Decision(
        id="D-20251229-090000-jkl22222",
        title="Use event sourcing for CEL",
        rationale="Immutable events provide better auditability",
        affects=["architecture", "persistence"],
        properties={"status": "accepted"},
    )


@pytest.fixture
def sample_sprint():
    """Create a sample sprint."""
    return Sprint(
        id="S-20251229-080000-mno33333",
        title="Sprint 30: CEL Migration",
        status="in_progress",
        epic_id="EPIC-cel-migration",
        number=30,
        goals=[{"description": "Complete bridge tests", "completed": False}],
        notes=["Focus on adapter coverage"],
        properties={},
        metadata={},
    )


@pytest.fixture
def sample_epic():
    """Create a sample epic."""
    return Epic(
        id="EPIC-cel-migration",
        title="CEL Migration Epic",
        status="active",
        phase=2,
        phases=[
            {"name": "Phase 1: Foundation", "status": "completed"},
            {"name": "Phase 2: Bridge", "status": "in_progress"},
        ],
        properties={},
        metadata={},
    )


@pytest.fixture
def sample_handoff():
    """Create a sample handoff."""
    return Handoff(
        id="H-20251229-070000-pqr44444",
        source_agent="main-agent",
        target_agent="specialist-agent",
        task_id="T-20251229-120000-abc12345",
        status="completed",
        instructions="Continue CEL integration work",
        context={"branch": "claude/feature-cel"},
        result={"success": True},
        properties={},
    )


@pytest.fixture
def sample_document():
    """Create a sample document."""
    return Document(
        id="DOC-cel-guide",
        path="docs/cel-guide.md",
        title="CEL Integration Guide",
        doc_type="guide",  # Valid doc_types: general, architecture, design, memory, decision, api, guide, research, knowledge-transfer
        tags=["cel", "guide", "architecture"],
        category="documentation",
        properties={},
        metadata={"author": "test-agent"},
    )


@pytest.fixture
def temp_got_dir(tmp_path):
    """Create a temporary GoT directory with sample entities."""
    got_path = tmp_path / ".got"
    entities_path = got_path / "entities"
    entities_path.mkdir(parents=True)

    # Create sample task file
    task_data = {
        "_checksum": "abc123",
        "_written_at": "2025-12-29T12:00:00Z",
        "data": {
            "id": "T-test-001",
            "entity_type": "task",
            "title": "Test task",
            "status": "pending",
            "priority": "medium",
            "description": "A test task",
            "properties": {},
            "metadata": {},
            "version": 1,
            "created_at": "2025-12-29T12:00:00Z",
            "modified_at": "2025-12-29T12:00:00Z",
        }
    }
    with open(entities_path / "T-test-001.json", "w") as f:
        json.dump(task_data, f)

    # Create sample decision file
    decision_data = {
        "_checksum": "def456",
        "_written_at": "2025-12-29T12:00:00Z",
        "data": {
            "id": "D-test-001",
            "entity_type": "decision",
            "title": "Test decision",
            "rationale": "For testing",
            "affects": [],
            "properties": {},
            "version": 1,
            "created_at": "2025-12-29T12:00:00Z",
            "modified_at": "2025-12-29T12:00:00Z",
        }
    }
    with open(entities_path / "D-test-001.json", "w") as f:
        json.dump(decision_data, f)

    return got_path


# =============================================================================
# TEST: GoTEventAdapter.entity_to_event()
# =============================================================================

class TestGoTEventAdapterEntityToEvent:
    """Test converting GoT entities to CEL events."""

    def test_pending_task_becomes_intention(self, sample_task):
        """Pending task converts to INTENTION event type."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert event.event_type == EventType.INTENTION
        assert event.content['entity_id'] == sample_task.id
        assert event.content['entity_type'] == 'task'
        assert event.content['title'] == sample_task.title
        assert event.content['status'] == 'pending'
        assert event.content['priority'] == 'high'

    def test_completed_task_becomes_fulfillment(self, completed_task):
        """Completed task converts to FULFILLMENT event type."""
        event = GoTEventAdapter.entity_to_event(completed_task)

        assert event.event_type == EventType.FULFILLMENT
        assert event.content['entity_id'] == completed_task.id
        assert event.content['status'] == 'completed'

    def test_blocked_task_becomes_observation(self, blocked_task):
        """Blocked task converts to OBSERVATION event type."""
        event = GoTEventAdapter.entity_to_event(blocked_task)

        assert event.event_type == EventType.OBSERVATION
        assert event.content['entity_id'] == blocked_task.id
        assert event.content['status'] == 'blocked'

    def test_task_concepts_extracted_from_title(self, sample_task):
        """Concepts are extracted from task title."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        # Title words should be in concepts (up to 5)
        assert 'implement' in event.concepts
        assert 'cel' in event.concepts
        assert 'bridge' in event.concepts

    def test_task_concepts_include_category(self, sample_task):
        """Category from properties is added to concepts."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert 'feature' in event.concepts

    def test_task_concepts_include_tags(self, sample_task):
        """Tags from properties are added to concepts."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert 'cel' in event.concepts
        assert 'migration' in event.concepts

    def test_task_preserves_got_version(self, sample_task):
        """GoT version is preserved in event content."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert event.content['got_version'] == sample_task.version

    def test_task_preserves_metadata(self, sample_task):
        """Task metadata is preserved in event content."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert event.content['metadata'] == sample_task.metadata

    def test_decision_becomes_observation(self, sample_decision):
        """Decision converts to OBSERVATION event type."""
        event = GoTEventAdapter.entity_to_event(sample_decision)

        assert event.event_type == EventType.OBSERVATION
        assert event.content['entity_type'] == 'decision'
        assert event.content['category'] == 'decision'
        assert event.content['rationale'] == sample_decision.rationale

    def test_decision_concepts_include_decision_keyword(self, sample_decision):
        """Decision events always include 'decision' in concepts."""
        event = GoTEventAdapter.entity_to_event(sample_decision)

        assert 'decision' in event.concepts

    def test_sprint_becomes_observation(self, sample_sprint):
        """Sprint converts to OBSERVATION event type."""
        event = GoTEventAdapter.entity_to_event(sample_sprint)

        assert event.event_type == EventType.OBSERVATION
        assert event.content['entity_type'] == 'sprint'
        assert event.content['category'] == 'sprint'
        assert event.content['number'] == 30
        assert event.content['epic_id'] == sample_sprint.epic_id

    def test_sprint_concepts_include_sprint_keyword(self, sample_sprint):
        """Sprint events always include 'sprint' in concepts."""
        event = GoTEventAdapter.entity_to_event(sample_sprint)

        assert 'sprint' in event.concepts

    def test_sprint_with_epic_includes_epic_concept(self, sample_sprint):
        """Sprint with epic_id includes 'epic' in concepts."""
        event = GoTEventAdapter.entity_to_event(sample_sprint)

        assert 'epic' in event.concepts

    def test_epic_becomes_observation(self, sample_epic):
        """Epic converts to OBSERVATION event type."""
        event = GoTEventAdapter.entity_to_event(sample_epic)

        assert event.event_type == EventType.OBSERVATION
        assert event.content['entity_type'] == 'epic'
        assert event.content['category'] == 'epic'
        assert event.content['phase'] == 2
        assert len(event.content['phases']) == 2

    def test_epic_concepts_include_epic_keyword(self, sample_epic):
        """Epic events always include 'epic' in concepts."""
        event = GoTEventAdapter.entity_to_event(sample_epic)

        assert 'epic' in event.concepts

    def test_handoff_becomes_observation(self, sample_handoff):
        """Handoff converts to OBSERVATION event type."""
        event = GoTEventAdapter.entity_to_event(sample_handoff)

        assert event.event_type == EventType.OBSERVATION
        assert event.content['entity_type'] == 'handoff'
        assert event.content['category'] == 'handoff'
        assert event.content['source_agent'] == 'main-agent'
        assert event.content['target_agent'] == 'specialist-agent'

    def test_completed_handoff_includes_completed_concept(self, sample_handoff):
        """Completed handoff includes 'completed' in concepts."""
        event = GoTEventAdapter.entity_to_event(sample_handoff)

        assert 'completed' in event.concepts
        assert 'handoff' in event.concepts
        assert 'agent-coordination' in event.concepts

    def test_document_becomes_observation(self, sample_document):
        """Document converts to OBSERVATION event type."""
        event = GoTEventAdapter.entity_to_event(sample_document)

        assert event.event_type == EventType.OBSERVATION
        assert event.content['entity_type'] == 'document'
        assert event.content['path'] == sample_document.path
        assert event.content['doc_type'] == 'guide'

    def test_document_concepts_include_tags(self, sample_document):
        """Document concepts include its tags."""
        event = GoTEventAdapter.entity_to_event(sample_document)

        assert 'document' in event.concepts
        assert 'guide' in event.concepts  # doc_type
        assert 'cel' in event.concepts  # first tag

    def test_event_has_timestamp_from_entity(self, sample_task):
        """Event timestamp comes from entity modified_at."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert event.timestamp == sample_task.modified_at

    def test_event_has_empty_causal_parents(self, sample_task):
        """Initial conversion has no causal parents (filled by store)."""
        event = GoTEventAdapter.entity_to_event(sample_task)

        assert event.causal_parents == ()


# =============================================================================
# TEST: GoTEntityAdapter.event_to_entity()
# =============================================================================

class TestGoTEntityAdapterEventToEntity:
    """Test converting CEL events back to GoT entities."""

    def test_task_event_converts_to_task(self, sample_task):
        """Task event roundtrips back to Task entity."""
        event = GoTEventAdapter.entity_to_event(sample_task)
        entity = GoTEntityAdapter.event_to_entity(event)

        assert isinstance(entity, Task)
        assert entity.id == sample_task.id
        assert entity.title == sample_task.title
        assert entity.status == sample_task.status
        assert entity.priority == sample_task.priority

    def test_decision_event_converts_to_decision(self, sample_decision):
        """Decision event roundtrips back to Decision entity."""
        event = GoTEventAdapter.entity_to_event(sample_decision)
        entity = GoTEntityAdapter.event_to_entity(event)

        assert isinstance(entity, Decision)
        assert entity.id == sample_decision.id
        assert entity.title == sample_decision.title
        assert entity.rationale == sample_decision.rationale

    def test_sprint_event_converts_to_sprint(self, sample_sprint):
        """Sprint event roundtrips back to Sprint entity."""
        event = GoTEventAdapter.entity_to_event(sample_sprint)
        entity = GoTEntityAdapter.event_to_entity(event)

        assert isinstance(entity, Sprint)
        assert entity.id == sample_sprint.id
        assert entity.title == sample_sprint.title
        assert entity.number == sample_sprint.number

    def test_epic_event_converts_to_epic(self, sample_epic):
        """Epic event roundtrips back to Epic entity."""
        event = GoTEventAdapter.entity_to_event(sample_epic)
        entity = GoTEntityAdapter.event_to_entity(event)

        assert isinstance(entity, Epic)
        assert entity.id == sample_epic.id
        assert entity.title == sample_epic.title
        assert entity.phase == sample_epic.phase

    def test_handoff_event_converts_to_handoff(self, sample_handoff):
        """Handoff event roundtrips back to Handoff entity."""
        event = GoTEventAdapter.entity_to_event(sample_handoff)
        entity = GoTEntityAdapter.event_to_entity(event)

        assert isinstance(entity, Handoff)
        assert entity.id == sample_handoff.id
        assert entity.source_agent == sample_handoff.source_agent
        assert entity.target_agent == sample_handoff.target_agent

    def test_document_event_converts_to_document(self, sample_document):
        """Document event roundtrips back to Document entity."""
        event = GoTEventAdapter.entity_to_event(sample_document)
        entity = GoTEntityAdapter.event_to_entity(event)

        assert isinstance(entity, Document)
        assert entity.id == sample_document.id
        assert entity.path == sample_document.path
        assert entity.doc_type == sample_document.doc_type

    def test_event_without_entity_type_returns_none(self):
        """Event without entity_type in content returns None."""
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={"some": "data"},
            concepts=("test",),
        )
        entity = GoTEntityAdapter.event_to_entity(event)

        assert entity is None

    def test_unknown_entity_type_returns_none(self):
        """Event with unknown entity_type returns None."""
        event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={"entity_type": "unknown_type"},
            concepts=("test",),
        )
        entity = GoTEntityAdapter.event_to_entity(event)

        assert entity is None


# =============================================================================
# TEST: GotBridgeEventStore
# =============================================================================

class TestGotBridgeEventStore:
    """Test the bridge event store."""

    def test_init_with_got_path(self, tmp_path):
        """Store initializes with GoT path."""
        got_path = tmp_path / ".got"
        got_path.mkdir()

        store = GotBridgeEventStore(got_path=got_path)

        assert store._got_path == got_path
        assert store._write_got is True
        assert store._write_cel is True

    def test_init_with_cel_path(self, tmp_path):
        """Store initializes with CEL path."""
        got_path = tmp_path / ".got"
        cel_path = tmp_path / ".cel"
        got_path.mkdir()

        store = GotBridgeEventStore(got_path=got_path, cel_path=cel_path)

        assert store._cel_path == cel_path

    def test_load_entities_from_got(self, temp_got_dir):
        """Store loads entities from GoT directory."""
        store = GotBridgeEventStore(got_path=temp_got_dir)
        store._ensure_loaded()

        # Should have loaded 2 entities (task and decision)
        assert len(store._events) == 2
        assert store._loaded is True

    def test_get_event_by_id(self, temp_got_dir):
        """Can retrieve event by ID after loading."""
        store = GotBridgeEventStore(got_path=temp_got_dir)

        # Get an event - this triggers loading
        events = list(store.iterate())
        assert len(events) > 0

        event_id = events[0].id
        retrieved = store.get(event_id)
        assert retrieved is not None
        assert retrieved.id == event_id

    def test_get_nonexistent_event_returns_none(self, temp_got_dir):
        """Getting nonexistent event returns None."""
        store = GotBridgeEventStore(got_path=temp_got_dir)

        result = store.get("nonexistent-id")
        assert result is None

    def test_contains_for_existing_event(self, temp_got_dir):
        """Contains returns True for existing event."""
        store = GotBridgeEventStore(got_path=temp_got_dir)

        events = list(store.iterate())
        event_id = events[0].id

        assert store.contains(event_id) is True

    def test_contains_for_nonexistent_event(self, temp_got_dir):
        """Contains returns False for nonexistent event."""
        store = GotBridgeEventStore(got_path=temp_got_dir)

        assert store.contains("nonexistent-id") is False

    def test_iterate_yields_all_events(self, temp_got_dir):
        """Iterate yields all loaded events."""
        store = GotBridgeEventStore(got_path=temp_got_dir)

        events = list(store.iterate())
        assert len(events) == 2  # task and decision

    def test_heads_returns_recent_events(self, temp_got_dir):
        """Heads returns most recent events as MerkleRoots."""
        store = GotBridgeEventStore(got_path=temp_got_dir)
        store._ensure_loaded()

        heads = store.heads()
        assert len(heads) > 0
        assert all(isinstance(h, MerkleRoot) for h in heads)

    def test_latest_returns_most_recent(self, temp_got_dir):
        """Latest returns most recent event."""
        store = GotBridgeEventStore(got_path=temp_got_dir)
        store._ensure_loaded()

        latest = store.latest()
        assert latest is not None
        assert isinstance(latest, MerkleRoot)

    def test_latest_on_empty_store_returns_none(self, tmp_path):
        """Latest returns None on empty store."""
        got_path = tmp_path / ".got"
        got_path.mkdir()

        store = GotBridgeEventStore(got_path=got_path)

        assert store.latest() is None

    def test_append_adds_event(self, temp_got_dir):
        """Append adds event to store."""
        store = GotBridgeEventStore(
            got_path=temp_got_dir,
            write_to_got=False,  # Don't write to disk
            write_to_cel=False,
        )
        store._ensure_loaded()
        initial_count = len(store._events)

        new_event = CognitiveEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={"test": "data", "entity_type": "task"},
            concepts=("test",),
        )

        root = store.append(new_event)

        assert len(store._events) == initial_count + 1
        assert isinstance(root, MerkleRoot)
        assert store.contains(new_event.id)

    def test_append_writes_to_got_when_enabled(self, temp_got_dir, sample_task):
        """Append writes to GoT when write_to_got is True."""
        store = GotBridgeEventStore(
            got_path=temp_got_dir,
            write_to_got=True,
            write_to_cel=False,
        )

        event = GoTEventAdapter.entity_to_event(sample_task)
        store.append(event)

        # Check file was created
        entity_file = temp_got_dir / "entities" / f"{sample_task.id}.json"
        assert entity_file.exists()

    def test_append_writes_to_cel_when_enabled(self, temp_got_dir, tmp_path, sample_task):
        """Append writes to CEL when write_to_cel is True."""
        cel_path = tmp_path / ".cel"
        store = GotBridgeEventStore(
            got_path=temp_got_dir,
            cel_path=cel_path,
            write_to_got=False,
            write_to_cel=True,
        )

        event = GoTEventAdapter.entity_to_event(sample_task)
        store.append(event)

        # Check file was created
        events_path = cel_path / "events"
        assert events_path.exists()
        event_files = list(events_path.glob("*.json"))
        assert len(event_files) == 1

    def test_stats_returns_counts(self, temp_got_dir):
        """Stats returns entity counts and configuration."""
        store = GotBridgeEventStore(got_path=temp_got_dir)

        stats = store.stats

        assert 'total_events' in stats
        assert stats['total_events'] == 2
        assert 'entity_types' in stats
        assert 'task' in stats['entity_types']
        assert 'decision' in stats['entity_types']

    def test_empty_entities_dir_handled(self, tmp_path):
        """Empty entities directory is handled gracefully."""
        got_path = tmp_path / ".got"
        entities_path = got_path / "entities"
        entities_path.mkdir(parents=True)

        store = GotBridgeEventStore(got_path=got_path)
        events = list(store.iterate())

        assert len(events) == 0

    def test_nonexistent_got_path_handled(self, tmp_path):
        """Nonexistent GoT path is handled gracefully."""
        got_path = tmp_path / ".got"  # Does not exist

        store = GotBridgeEventStore(got_path=got_path)
        events = list(store.iterate())

        assert len(events) == 0


# =============================================================================
# TEST: create_got_bridge() Factory
# =============================================================================

class TestCreateGotBridge:
    """Test the factory function for creating bridges."""

    def test_default_creates_both_mode(self, tmp_path):
        """Default creates bridge with both writes enabled."""
        got_path = tmp_path / ".got"
        got_path.mkdir()

        bridge = create_got_bridge(got_path=got_path)

        assert bridge._write_got is True
        assert bridge._write_cel is True

    def test_got_only_mode(self, tmp_path):
        """got_only mode only writes to GoT."""
        got_path = tmp_path / ".got"
        got_path.mkdir()

        bridge = create_got_bridge(got_path=got_path, write_mode='got_only')

        assert bridge._write_got is True
        assert bridge._write_cel is False

    def test_cel_only_mode(self, tmp_path):
        """cel_only mode only writes to CEL."""
        got_path = tmp_path / ".got"
        got_path.mkdir()

        bridge = create_got_bridge(got_path=got_path, write_mode='cel_only')

        assert bridge._write_got is False
        assert bridge._write_cel is True

    def test_read_only_mode(self, tmp_path):
        """read_only mode doesn't write anywhere."""
        got_path = tmp_path / ".got"
        got_path.mkdir()

        bridge = create_got_bridge(got_path=got_path, write_mode='read_only')

        assert bridge._write_got is False
        assert bridge._write_cel is False

    def test_with_cel_path(self, tmp_path):
        """Factory accepts cel_path parameter."""
        got_path = tmp_path / ".got"
        cel_path = tmp_path / ".cel"
        got_path.mkdir()

        bridge = create_got_bridge(got_path=got_path, cel_path=cel_path)

        assert bridge._cel_path == cel_path


# =============================================================================
# TEST: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_task_without_title_handled(self):
        """Task without title uses empty concepts."""
        task = Task(
            id="T-no-title",
            title="",
            status="pending",
            priority="medium",
        )

        event = GoTEventAdapter.entity_to_event(task)

        assert event.content['title'] == ""
        # Should not crash with empty title

    def test_sprint_without_epic_no_epic_concept(self):
        """Sprint without epic_id doesn't add 'epic' concept."""
        sprint = Sprint(
            id="S-no-epic",
            title="Test Sprint",
            status="available",
            epic_id="",
            number=1,
            goals=[],
            notes=[],
            properties={},
            metadata={},
        )

        event = GoTEventAdapter.entity_to_event(sprint)

        assert 'epic' not in event.concepts

    def test_malformed_json_file_skipped(self, tmp_path):
        """Malformed JSON files are skipped during load."""
        got_path = tmp_path / ".got"
        entities_path = got_path / "entities"
        entities_path.mkdir(parents=True)

        # Create malformed file
        with open(entities_path / "bad.json", "w") as f:
            f.write("{ not valid json }")

        store = GotBridgeEventStore(got_path=got_path)
        events = list(store.iterate())

        # Should not crash, just skip the bad file
        assert len(events) == 0

    def test_missing_entity_type_in_file_skipped(self, tmp_path):
        """Files without entity_type are skipped."""
        got_path = tmp_path / ".got"
        entities_path = got_path / "entities"
        entities_path.mkdir(parents=True)

        # Create file without entity_type
        with open(entities_path / "no-type.json", "w") as f:
            json.dump({"data": {"id": "test", "title": "No type"}}, f)

        store = GotBridgeEventStore(got_path=got_path)
        events = list(store.iterate())

        assert len(events) == 0

    def test_invalid_entity_type_skipped(self, tmp_path):
        """Files with invalid entity_type are skipped."""
        got_path = tmp_path / ".got"
        entities_path = got_path / "entities"
        entities_path.mkdir(parents=True)

        # Create file with invalid entity_type
        data = {
            "data": {
                "id": "test",
                "entity_type": "not_a_valid_type",
            }
        }
        with open(entities_path / "invalid-type.json", "w") as f:
            json.dump(data, f)

        store = GotBridgeEventStore(got_path=got_path)
        events = list(store.iterate())

        assert len(events) == 0

    def test_iterate_with_from_event(self, temp_got_dir):
        """Iterate respects from_event parameter."""
        store = GotBridgeEventStore(got_path=temp_got_dir)
        events = list(store.iterate())

        if len(events) >= 2:
            # Start from second event
            from_id = events[0].id
            filtered = list(store.iterate(from_event=from_id))

            # Should not include the first event
            assert events[0] not in filtered

    def test_iterate_with_to_event(self, temp_got_dir):
        """Iterate respects to_event parameter."""
        store = GotBridgeEventStore(got_path=temp_got_dir)
        events = list(store.iterate())

        if len(events) >= 2:
            # Stop at first event
            to_id = events[0].id
            filtered = list(store.iterate(to_event=to_id))

            # Should include at most one event
            assert len(filtered) <= 1
