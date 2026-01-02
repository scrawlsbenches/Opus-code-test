"""
Behavioral tests for Cognitive Event Lattice (CEL) event sourcing workflows.

Epic: Event-Sourced Cognitive Architecture

As a cognitive system developer,
I want to store immutable events that form a causal DAG,
So that I can materialize entity state at any point in time without paradox.

These tests demonstrate the core event sourcing behaviors:
1. Events are immutable truth, entities are derived
2. Content-addressed storage prevents conflicts
3. Temporal references enable self-reference without paradox
4. Materialization computes entity state from event history
5. Time travel queries reconstruct past states
6. Semantic indexing enables fast concept-based retrieval
7. The system monitors its own health

Testing Philosophy (Metus):
- Scenarios test behaviors, not implementation
- Given-When-Then format tells the story
- Tests serve as living documentation
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# =============================================================================
# MINIMAL CEL IMPLEMENTATION FOR BEHAVIORAL TESTING
# Built from first principles, no external dependencies
# =============================================================================

class EventType(Enum):
    """Types of cognitive events."""
    OBSERVATION = auto()
    INTENTION = auto()
    FULFILLMENT = auto()
    META_COGNITION = auto()


@dataclass(frozen=True)
class CognitiveEvent:
    """
    Immutable record of something that happened.

    Content-addressed: ID is hash of content, enabling natural deduplication.
    """
    timestamp: str
    event_type: EventType
    causal_parents: Tuple[str, ...]
    content: Dict[str, Any]
    concepts: Tuple[str, ...]

    @property
    def id(self) -> str:
        """Content-addressed ID (Merkle root)."""
        data = {
            'timestamp': self.timestamp,
            'event_type': self.event_type.name,
            'causal_parents': list(self.causal_parents),
            'content': self.content,
            'concepts': list(self.concepts),
        }
        content_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class EventHorizon:
    """A specific point in the event DAG for temporal queries."""
    event_id: str
    is_head: bool = False


@dataclass
class MaterializedTask:
    """Task entity materialized from events."""
    id: str
    title: str
    status: str
    description: str = ""
    created_at: str = ""
    completed_at: str = ""
    version: int = 0


class InMemoryEventStore:
    """Event store built from first principles for testing."""

    def __init__(self):
        self._events: Dict[str, CognitiveEvent] = {}
        self._order: List[str] = []
        self._children: Dict[str, List[str]] = defaultdict(list)

    def append(self, event: CognitiveEvent) -> str:
        """Append event, return its content-addressed ID."""
        event_id = event.id

        if event_id in self._events:
            # Content-addressed deduplication
            return event_id

        self._events[event_id] = event
        self._order.append(event_id)

        for parent_id in event.causal_parents:
            self._children[parent_id].append(event_id)

        return event_id

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        return self._events.get(event_id)

    def iterate_until(self, horizon: EventHorizon) -> List[CognitiveEvent]:
        """Iterate events up to (and including) the horizon."""
        events = []
        for eid in self._order:
            events.append(self._events[eid])
            if eid == horizon.event_id:
                break
        return events

    def latest(self) -> Optional[str]:
        """Get most recent event ID."""
        return self._order[-1] if self._order else None

    @property
    def count(self) -> int:
        return len(self._events)


class SemanticIndex:
    """Concept-based index built from first principles."""

    def __init__(self):
        self._concept_to_events: Dict[str, Set[str]] = defaultdict(set)

    def index_event(self, event: CognitiveEvent) -> None:
        """Index an event's concepts."""
        for concept in event.concepts:
            self._concept_to_events[concept].add(event.id)

    def search(self, query: str) -> Set[str]:
        """Search for events matching query terms."""
        terms = query.lower().split()
        if not terms:
            return set()

        # Intersection of all term matches
        result = None
        for term in terms:
            matches = self._concept_to_events.get(term, set())
            if result is None:
                result = matches.copy()
            else:
                result &= matches

        return result if result else set()


class EntityMaterializer:
    """Materializes entities from event history."""

    def __init__(self, store: InMemoryEventStore):
        self._store = store

    def materialize_task(
        self,
        entity_id: str,
        at: Optional[EventHorizon] = None
    ) -> Optional[MaterializedTask]:
        """
        Materialize a task as of a specific horizon.

        Entities don't exist in storage - they are COMPUTED from events.
        This enables time-travel queries.
        """
        state = None

        if at:
            events = self._store.iterate_until(at)
        else:
            events = [self._store.get(eid) for eid in self._store._order]

        for event in events:
            if event.content.get('entity_id') != entity_id:
                continue
            if event.content.get('entity_type') != 'task':
                continue

            if state is None:
                # First event creates the task
                state = MaterializedTask(
                    id=entity_id,
                    title=event.content.get('title', ''),
                    status=event.content.get('status', 'pending'),
                    description=event.content.get('description', ''),
                    created_at=event.timestamp,
                    version=1,
                )
            else:
                # Subsequent events update it
                if 'title' in event.content:
                    state.title = event.content['title']
                if 'status' in event.content:
                    state.status = event.content['status']
                if 'description' in event.content:
                    state.description = event.content['description']
                if state.status == 'completed' and not state.completed_at:
                    state.completed_at = event.timestamp
                state.version += 1

        return state


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def event_store():
    """Provide a clean event store for each test."""
    return InMemoryEventStore()


@pytest.fixture
def semantic_index():
    """Provide a clean semantic index for each test."""
    return SemanticIndex()


@pytest.fixture
def materializer(event_store):
    """Provide an entity materializer."""
    return EntityMaterializer(event_store)


# =============================================================================
# BEHAVIORAL SCENARIOS
# =============================================================================

class TestDeveloperBuildsEventSourcedSystem:
    """
    Epic: Event Sourcing Foundation

    As a cognitive system developer,
    I want events to be the single source of truth,
    So that entities are derived consistently from immutable history.
    """

    def test_scenario_events_are_immutable_truth(self, event_store):
        """
        Scenario: Events are immutable and append-only

        Given an empty event store
        When I append a cognitive event
        Then the event is stored immutably
        And I receive its content-addressed ID
        """
        # Given an empty event store
        assert event_store.count == 0

        # When I append a cognitive event
        event = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'System initialized'},
            concepts=('system', 'initialization'),
        )
        event_id = event_store.append(event)

        # Then the event is stored immutably
        assert event_store.count == 1
        stored_event = event_store.get(event_id)
        assert stored_event.content == event.content
        assert stored_event.event_type == EventType.OBSERVATION

        # And I receive its content-addressed ID
        assert len(event_id) == 16  # SHA256 truncated to 16 chars
        assert event_id == event.id

    def test_scenario_entities_are_computed_from_events(self, event_store, materializer):
        """
        Scenario: Entities are materialized from event streams

        Given a series of task-related events
        When I materialize the task entity
        Then the task state is computed from the event history
        And the materialized entity reflects all event updates
        """
        # Given a series of task-related events

        # Event 1: Task created
        event1 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={
                'entity_id': 'task-001',
                'entity_type': 'task',
                'title': 'Build event store',
                'status': 'pending',
                'description': 'Implement core event storage infrastructure',
            },
            concepts=('task', 'implementation'),
        )
        event_store.append(event1)

        # Event 2: Task started
        event2 = CognitiveEvent(
            timestamp="2024-12-30T11:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(event1.id,),
            content={
                'entity_id': 'task-001',
                'entity_type': 'task',
                'status': 'in_progress',
            },
            concepts=('task', 'update'),
        )
        event_store.append(event2)

        # When I materialize the task entity
        task = materializer.materialize_task('task-001')

        # Then the task state is computed from the event history
        assert task is not None
        assert task.id == 'task-001'

        # And the materialized entity reflects all event updates
        assert task.title == 'Build event store'
        assert task.status == 'in_progress'
        assert task.description == 'Implement core event storage infrastructure'
        assert task.version == 2  # Two events applied

    def test_scenario_content_addressed_ids_prevent_conflicts(self, event_store):
        """
        Scenario: Same content produces same ID (natural deduplication)

        Given two events with identical content
        When I append both events to the store
        Then they produce the same content-addressed ID
        And the store contains only one event (deduplication)
        """
        # Given two events with identical content
        event1 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Hello, world!'},
            concepts=('greeting',),
        )

        event2 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Hello, world!'},
            concepts=('greeting',),
        )

        # When I append both events to the store
        id1 = event_store.append(event1)
        id2 = event_store.append(event2)

        # Then they produce the same content-addressed ID
        assert id1 == id2
        assert event1.id == event2.id

        # And the store contains only one event (deduplication)
        assert event_store.count == 1


class TestDeveloperQueriesPastStates:
    """
    Epic: Time Travel Queries

    As a cognitive system developer,
    I want to query entity state at any point in time,
    So that I can understand what the system knew when it made decisions.
    """

    def test_scenario_materialize_entity_at_past_horizon(self, event_store, materializer):
        """
        Scenario: Query entity state as it was at a specific point in time

        Given a task that transitions through multiple states
        When I capture a horizon during the 'in_progress' state
        And the task later completes
        Then I can materialize the task at the past horizon
        And it shows the 'in_progress' state, not the completed state
        """
        # Given a task that transitions through multiple states

        # Event 1: Task created (pending)
        event1 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={
                'entity_id': 'task-time-travel',
                'entity_type': 'task',
                'title': 'Test time travel',
                'status': 'pending',
            },
            concepts=('task',),
        )
        event_store.append(event1)

        # Event 2: Task started (in_progress)
        event2 = CognitiveEvent(
            timestamp="2024-12-30T11:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(event1.id,),
            content={
                'entity_id': 'task-time-travel',
                'entity_type': 'task',
                'status': 'in_progress',
            },
            concepts=('task',),
        )
        event_store.append(event2)

        # When I capture a horizon during the 'in_progress' state
        horizon_in_progress = EventHorizon(event_id=event2.id)

        # And the task later completes
        event3 = CognitiveEvent(
            timestamp="2024-12-30T12:00:00",
            event_type=EventType.FULFILLMENT,
            causal_parents=(event2.id,),
            content={
                'entity_id': 'task-time-travel',
                'entity_type': 'task',
                'status': 'completed',
            },
            concepts=('task', 'completed'),
        )
        event_store.append(event3)

        # Then I can materialize the task at the past horizon
        task_then = materializer.materialize_task('task-time-travel', at=horizon_in_progress)
        task_now = materializer.materialize_task('task-time-travel')

        # And it shows the 'in_progress' state, not the completed state
        assert task_then.status == 'in_progress'
        assert task_then.completed_at == ''

        # While current state is completed
        assert task_now.status == 'completed'
        assert task_now.completed_at != ''

    def test_scenario_temporal_reference_enables_stable_self_reference(self, event_store):
        """
        Scenario: Reference system state at specific time without paradox

        Given I capture the current horizon before creating a task
        When I create a task that references that horizon
        And the system continues to evolve with new events
        Then the task's temporal reference remains stable
        And points to the system state before the task existed
        """
        # Given I capture the current horizon before creating a task
        genesis_event = CognitiveEvent(
            timestamp="2024-12-30T09:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'System genesis'},
            concepts=('system',),
        )
        genesis_id = event_store.append(genesis_event)

        horizon_before_task = EventHorizon(event_id=genesis_id)

        # When I create a task that references that horizon
        task_event = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.INTENTION,
            causal_parents=(genesis_id,),
            content={
                'entity_id': 'task-self-ref',
                'entity_type': 'task',
                'title': 'Implement self-referencing task',
                'status': 'pending',
                'temporal_reference': {
                    'entity_id': 'task-self-ref',
                    'horizon_id': horizon_before_task.event_id,
                }
            },
            concepts=('task', 'self-reference'),
        )
        task_id = event_store.append(task_event)

        # And the system continues to evolve with new events
        followup_event = CognitiveEvent(
            timestamp="2024-12-30T11:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(task_id,),
            content={'message': 'System continues to evolve'},
            concepts=('system',),
        )
        event_store.append(followup_event)

        # Then the task's temporal reference remains stable
        stored_task = event_store.get(task_id)
        ref_horizon_id = stored_task.content['temporal_reference']['horizon_id']

        assert ref_horizon_id == genesis_id
        # And points to the system state before the task existed
        assert ref_horizon_id != task_id


class TestDeveloperSearchesBySemantics:
    """
    Epic: Semantic Concept Search

    As a cognitive system developer,
    I want to search events by conceptual tags,
    So that I can find related events without knowing exact IDs.
    """

    def test_scenario_events_are_indexed_by_concepts(self, event_store, semantic_index):
        """
        Scenario: Find events by concept tags

        Given multiple events tagged with different concepts
        When I index all events
        And I search for a specific concept
        Then I retrieve all events tagged with that concept
        """
        # Given multiple events tagged with different concepts
        event1 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Task created'},
            concepts=('task', 'creation'),
        )

        event2 = CognitiveEvent(
            timestamp="2024-12-30T11:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(event1.id,),
            content={'message': 'Task updated'},
            concepts=('task', 'update'),
        )

        event3 = CognitiveEvent(
            timestamp="2024-12-30T12:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(event2.id,),
            content={'message': 'Decision made'},
            concepts=('decision', 'architecture'),
        )

        event_store.append(event1)
        event_store.append(event2)
        event_store.append(event3)

        # When I index all events
        semantic_index.index_event(event1)
        semantic_index.index_event(event2)
        semantic_index.index_event(event3)

        # And I search for a specific concept
        task_events = semantic_index.search('task')

        # Then I retrieve all events tagged with that concept
        assert len(task_events) == 2
        assert event1.id in task_events
        assert event2.id in task_events
        assert event3.id not in task_events

    def test_scenario_multi_term_search_uses_intersection(self, semantic_index):
        """
        Scenario: Search with multiple terms finds events matching all terms

        Given events with various concept combinations
        When I search for multiple terms
        Then I get events that match ALL terms (intersection)
        """
        # Given events with various concept combinations
        event1 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Task and storage'},
            concepts=('task', 'storage'),
        )

        event2 = CognitiveEvent(
            timestamp="2024-12-30T11:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Only task'},
            concepts=('task', 'other'),
        )

        event3 = CognitiveEvent(
            timestamp="2024-12-30T12:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Only storage'},
            concepts=('storage', 'other'),
        )

        semantic_index.index_event(event1)
        semantic_index.index_event(event2)
        semantic_index.index_event(event3)

        # When I search for multiple terms
        results = semantic_index.search('task storage')

        # Then I get events that match ALL terms (intersection)
        assert len(results) == 1
        assert event1.id in results


class TestDeveloperBuildsEventDAG:
    """
    Epic: Causal Event Graph

    As a cognitive system developer,
    I want events to reference their causal parents,
    So that I can understand dependencies and event relationships.
    """

    def test_scenario_events_form_causal_chain(self, event_store):
        """
        Scenario: Events form a directed acyclic graph through parent references

        Given a genesis event with no parents
        When I create subsequent events that reference their parents
        Then each event maintains its causal lineage
        And I can traverse the DAG by following parent references
        """
        # Given a genesis event with no parents
        genesis = CognitiveEvent(
            timestamp="2024-12-30T09:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Genesis'},
            concepts=('system',),
        )
        genesis_id = event_store.append(genesis)

        # When I create subsequent events that reference their parents
        child1 = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(genesis_id,),
            content={'message': 'First child'},
            concepts=('system',),
        )
        child1_id = event_store.append(child1)

        child2 = CognitiveEvent(
            timestamp="2024-12-30T11:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(child1_id,),
            content={'message': 'Second child'},
            concepts=('system',),
        )
        child2_id = event_store.append(child2)

        # Then each event maintains its causal lineage
        genesis_retrieved = event_store.get(genesis_id)
        child1_retrieved = event_store.get(child1_id)
        child2_retrieved = event_store.get(child2_id)

        assert len(genesis_retrieved.causal_parents) == 0
        assert genesis_id in child1_retrieved.causal_parents
        assert child1_id in child2_retrieved.causal_parents

        # And I can traverse the DAG by following parent references
        assert event_store._children[genesis_id][0] == child1_id
        assert event_store._children[child1_id][0] == child2_id


class TestSystemMonitorsOwnHealth:
    """
    Epic: Self-Monitoring and Meta-Cognition

    As a cognitive system,
    I want to monitor my own health and state,
    So that I can detect issues and request maintenance.
    """

    def test_scenario_system_records_meta_cognition_events(self, event_store):
        """
        Scenario: System observes itself through meta-cognition events

        Given a system performing normal operations
        When the system observes its own state
        Then it creates a meta-cognition event
        And the event captures health metrics and observations
        """
        # Given a system performing normal operations
        normal_event = CognitiveEvent(
            timestamp="2024-12-30T10:00:00",
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'message': 'Normal operation'},
            concepts=('system',),
        )
        event_store.append(normal_event)

        # When the system observes its own state
        meta_event = CognitiveEvent(
            timestamp="2024-12-30T10:01:00",
            event_type=EventType.META_COGNITION,
            causal_parents=(normal_event.id,),
            content={
                'observation_type': 'self_check',
                'observation': 'Performed health check',
                'event_count': event_store.count,
                'health_status': 'HEALTHY',
            },
            concepts=('meta', 'self-observation'),
        )
        meta_id = event_store.append(meta_event)

        # Then it creates a meta-cognition event
        stored_meta = event_store.get(meta_id)
        assert stored_meta.event_type == EventType.META_COGNITION

        # And the event captures health metrics and observations
        assert stored_meta.content['observation_type'] == 'self_check'
        assert 'event_count' in stored_meta.content
        assert 'health_status' in stored_meta.content

    def test_scenario_event_count_triggers_maintenance_recommendation(self):
        """
        Scenario: High event count triggers compaction recommendation

        Given an event store with many events
        When the system checks its own health
        Then it detects the high event count
        And recommends compaction to maintain performance
        """
        # This scenario demonstrates self-maintenance behavior
        # The actual health monitoring logic would be in a separate component

        # Given an event store with many events
        event_count = 1500

        # When the system checks its own health
        # (Simplified logic for demonstration)
        issues = []
        recommendations = []

        if event_count > 1000:
            issues.append(f"High event count: {event_count}")
            recommendations.append("Consider running compaction")

        # Then it detects the high event count
        assert len(issues) > 0
        assert "High event count" in issues[0]

        # And recommends compaction to maintain performance
        assert "compaction" in recommendations[0].lower()
