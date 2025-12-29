"""
Behavioral tests for CEL Compaction strategies.

User stories test the compaction system from an end-user perspective,
focusing on reducing storage while preserving semantic meaning.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import MagicMock

from cortical.cel.sanity.compaction import (
    CompactionResult,
    BaseCompactor,
    TimeWindowCompactor,
    SemanticCompactor,
    CausalChainCompactor,
    create_compaction_schedule,
    estimate_compaction_savings,
)
from cortical.cel.core.events import CognitiveEvent, EventType
from cortical.cel.core.references import MerkleRoot


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_event_store():
    """Create a mock event store."""
    store = MagicMock()
    store.iterate.return_value = iter([])
    store.append.return_value = MerkleRoot("new-event-id")
    return store


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    now = datetime.now(timezone.utc)
    return [
        CognitiveEvent(
            timestamp=(now - timedelta(days=10)).isoformat(),
            event_type=EventType.INTENTION,
            causal_parents=(),
            content={'entity_id': 'task-1', 'title': 'Task 1'},
            concepts=('task', 'feature'),
        ),
        CognitiveEvent(
            timestamp=(now - timedelta(days=10, hours=1)).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'entity_id': 'task-1', 'update': 'Progress'},
            concepts=('task', 'update'),
        ),
        CognitiveEvent(
            timestamp=(now - timedelta(days=10, hours=2)).isoformat(),
            event_type=EventType.FULFILLMENT,
            causal_parents=(),
            content={'entity_id': 'task-1', 'result': 'Done'},
            concepts=('task', 'completed'),
        ),
    ]


@pytest.fixture
def old_events():
    """Create events that are old enough for compaction (>7 days)."""
    old_time = datetime.now(timezone.utc) - timedelta(days=30)
    events = []
    for i in range(10):
        events.append(CognitiveEvent(
            timestamp=(old_time + timedelta(hours=i)).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'entity_id': 'task-old', 'update': f'Update {i}'},
            concepts=('task', 'update'),
        ))
    return events


# =============================================================================
# USER STORY: CompactionResult
# =============================================================================

class TestCompactionResultBehavior:
    """
    User Story: As a system administrator, I want to understand
    the results of compaction operations, so I can verify storage
    savings and track what was removed.
    """

    def test_result_tracks_original_count(self):
        """Result reports how many events existed before compaction."""
        result = CompactionResult(
            original_count=100,
            compacted_count=50,
            events_removed=['e1', 'e2'],
            events_created=['c1'],
        )

        assert result.original_count == 100

    def test_result_tracks_compacted_count(self):
        """Result reports how many events remain after compaction."""
        result = CompactionResult(
            original_count=100,
            compacted_count=50,
            events_removed=['e1', 'e2'],
            events_created=['c1'],
        )

        assert result.compacted_count == 50

    def test_compression_ratio_calculated(self):
        """Result calculates compression ratio (compacted/original)."""
        result = CompactionResult(
            original_count=100,
            compacted_count=50,
            events_removed=[],
            events_created=[],
        )

        assert result.compression_ratio == 0.5  # 50/100

    def test_compression_ratio_handles_zero(self):
        """Compression ratio is 1.0 when original count is 0."""
        result = CompactionResult(
            original_count=0,
            compacted_count=0,
            events_removed=[],
            events_created=[],
        )

        assert result.compression_ratio == 1.0

    def test_duration_computed_when_complete(self):
        """Duration is computed when completed_at is set."""
        start = datetime.now()
        result = CompactionResult(
            original_count=100,
            compacted_count=50,
            events_removed=[],
            events_created=[],
            started_at=start,
        )
        result.completed_at = start + timedelta(seconds=10)

        assert result.duration is not None
        assert result.duration.total_seconds() == 10

    def test_duration_none_when_incomplete(self):
        """Duration is None when compaction is incomplete."""
        result = CompactionResult(
            original_count=100,
            compacted_count=0,
            events_removed=[],
            events_created=[],
        )

        assert result.duration is None

    def test_serialization_includes_all_fields(self):
        """to_dict() includes all relevant information."""
        result = CompactionResult(
            original_count=100,
            compacted_count=50,
            events_removed=['e1', 'e2', 'e3'],
            events_created=['c1'],
            bytes_saved=5000,
        )
        result.completed_at = datetime.now()

        data = result.to_dict()

        assert 'original_count' in data
        assert 'compacted_count' in data
        assert 'compression_ratio' in data
        assert 'events_removed' in data  # Count, not list
        assert 'bytes_saved' in data
        assert 'started_at' in data
        assert 'completed_at' in data


# =============================================================================
# USER STORY: TimeWindowCompactor
# =============================================================================

class TestTimeWindowCompactorBehavior:
    """
    User Story: As a storage manager, I want to compact old events
    within time windows, so I can reduce storage while preserving
    the final state of each entity.
    """

    def test_compactor_respects_min_age(self, mock_event_store):
        """Only events older than min_age are considered for compaction."""
        # Create event that's only 1 day old (min_age is 7 days by default)
        recent_event = CognitiveEvent(
            timestamp=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content={'entity_id': 'task-1'},
            concepts=('task',),
        )
        mock_event_store.iterate.return_value = iter([recent_event])

        compactor = TimeWindowCompactor(mock_event_store)
        groups = compactor.identify_compactable()

        assert len(groups) == 0  # Recent event not compactable

    def test_compactor_groups_by_entity_and_window(self, mock_event_store, old_events):
        """Events are grouped by entity ID and time window."""
        mock_event_store.iterate.return_value = iter(old_events)

        compactor = TimeWindowCompactor(
            mock_event_store,
            window_size=timedelta(hours=24),
            min_age=timedelta(days=7),
        )
        groups = compactor.identify_compactable()

        # All events are for same entity within 24h window
        assert len(groups) >= 1

    def test_compactor_preserves_marked_events(self, mock_event_store, old_events):
        """Events marked as preserved are not compacted."""
        mock_event_store.iterate.return_value = iter(old_events)

        compactor = TimeWindowCompactor(mock_event_store, min_age=timedelta(days=1))
        compactor.preserve(old_events[0].id)

        assert compactor.is_preserved(old_events[0].id)

    def test_compact_group_keeps_last_event(self, mock_event_store, old_events):
        """Compacting a group keeps the last event's state."""
        compactor = TimeWindowCompactor(mock_event_store)
        compacted, removed_ids = compactor.compact_group(old_events)

        # Last event should not be in removed list
        last_event_id = old_events[-1].id
        assert last_event_id not in removed_ids

    def test_compact_group_creates_summary(self, mock_event_store, old_events):
        """Compacted event contains summary information."""
        compactor = TimeWindowCompactor(mock_event_store)
        compacted, _ = compactor.compact_group(old_events)

        # Compaction events store summary in 'snapshot' field
        snapshot = compacted.content['snapshot']
        assert snapshot['compaction_type'] == 'time_window'
        assert 'event_count' in snapshot
        assert 'time_span' in snapshot

    def test_should_compact_recommends_when_many_old_events(self, mock_event_store):
        """should_compact() returns True when many old events exist."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        events = [
            CognitiveEvent(
                timestamp=old_time,
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=(),
            )
            for _ in range(150)  # More than 100 threshold
        ]
        mock_event_store.iterate.return_value = iter(events)

        compactor = TimeWindowCompactor(mock_event_store, min_age=timedelta(days=7))

        assert compactor.should_compact() is True


# =============================================================================
# USER STORY: SemanticCompactor
# =============================================================================

class TestSemanticCompactorBehavior:
    """
    User Story: As a knowledge engineer, I want to merge semantically
    similar events, so I can reduce redundancy while preserving
    unique information.
    """

    def test_compactor_groups_by_concept_similarity(self, mock_event_store):
        """Events with similar concepts are grouped together."""
        events = [
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'topic': 'auth'},
                concepts=('authentication', 'security', 'login'),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'topic': 'auth2'},
                concepts=('authentication', 'security', 'password'),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'topic': 'auth3'},
                concepts=('authentication', 'security', 'session'),
            ),
        ]
        mock_event_store.iterate.return_value = iter(events)

        compactor = SemanticCompactor(
            mock_event_store,
            similarity_threshold=0.5,  # 2/4 = 0.5 overlap
            min_group_size=2,
        )
        groups = compactor.identify_compactable()

        # Events share 2 concepts, should be grouped
        assert len(groups) >= 1

    def test_compactor_respects_similarity_threshold(self, mock_event_store):
        """Events below threshold are not grouped."""
        events = [
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('apple', 'banana', 'cherry'),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('dog', 'cat', 'bird'),  # Completely different
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('car', 'bus', 'train'),  # Completely different
            ),
        ]
        mock_event_store.iterate.return_value = iter(events)

        compactor = SemanticCompactor(
            mock_event_store,
            similarity_threshold=0.8,
            min_group_size=2,
        )
        groups = compactor.identify_compactable()

        # No groups should form with 80% similarity threshold
        assert len(groups) == 0

    def test_compact_group_picks_representative(self, mock_event_store):
        """Compaction picks the event with most concepts as representative."""
        events = [
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'id': 'few-concepts'},
                concepts=('a', 'b'),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'id': 'many-concepts'},
                concepts=('a', 'b', 'c', 'd', 'e'),  # Most concepts
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'id': 'medium-concepts'},
                concepts=('a', 'b', 'c'),
            ),
        ]

        compactor = SemanticCompactor(mock_event_store)
        compacted, removed_ids = compactor.compact_group(events)

        # Event with most concepts should be representative
        snapshot = compacted.content['snapshot']
        assert snapshot['representative_id'] == events[1].id

    def test_compact_group_preserves_all_concepts(self, mock_event_store):
        """Compaction preserves all unique concepts from the group."""
        events = [
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('concept1', 'shared'),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('concept2', 'shared'),
            ),
        ]

        compactor = SemanticCompactor(mock_event_store)
        compacted, _ = compactor.compact_group(events)

        # Access concepts from snapshot
        snapshot = compacted.content['snapshot']
        all_concepts = set(snapshot['all_concepts'])
        assert 'concept1' in all_concepts
        assert 'concept2' in all_concepts
        assert 'shared' in all_concepts

    def test_jaccard_similarity_calculation(self, mock_event_store):
        """Jaccard similarity is correctly calculated."""
        # Jaccard = intersection / union
        set1 = {'a', 'b', 'c'}
        set2 = {'b', 'c', 'd'}
        # intersection = {b, c} = 2
        # union = {a, b, c, d} = 4
        # Jaccard = 2/4 = 0.5

        similarity = SemanticCompactor._jaccard_similarity(set1, set2)
        assert similarity == 0.5

    def test_jaccard_empty_sets(self, mock_event_store):
        """Jaccard similarity of empty sets is 0."""
        similarity = SemanticCompactor._jaccard_similarity(set(), set())
        assert similarity == 0.0


# =============================================================================
# USER STORY: CausalChainCompactor
# =============================================================================

class TestCausalChainCompactorBehavior:
    """
    User Story: As a system architect, I want to flatten long
    causal chains, so I can reduce traversal depth while
    preserving the relationship between start and end events.
    """

    def test_compactor_identifies_long_chains(self, mock_event_store):
        """Chains longer than max_chain_length are identified."""
        # Create a chain: e1 -> e2 -> e3 -> e4 -> e5 -> e6
        events = []
        prev_id = None
        for i in range(6):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(prev_id,) if prev_id else (),
                content={'step': i},
                concepts=('chain',),
            )
            events.append(event)
            prev_id = event.id

        mock_event_store.iterate.return_value = iter(events)

        compactor = CausalChainCompactor(mock_event_store, max_chain_length=3)
        groups = compactor.identify_compactable()

        # Chain of 6 > max of 3, should be identified
        assert len(groups) >= 1

    def test_compactor_preserves_endpoints(self, mock_event_store):
        """Compaction keeps first and last events in chain."""
        events = []
        prev_id = None
        for i in range(5):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(prev_id,) if prev_id else (),
                content={'step': i},
                concepts=('chain',),
            )
            events.append(event)
            prev_id = event.id

        compactor = CausalChainCompactor(mock_event_store)
        compacted, removed_ids = compactor.compact_group(events)

        # First and last should not be removed
        assert events[0].id not in removed_ids
        assert events[-1].id not in removed_ids
        # Intermediate should be removed
        assert events[1].id in removed_ids
        assert events[2].id in removed_ids
        assert events[3].id in removed_ids

    def test_compact_group_creates_chain_summary(self, mock_event_store):
        """Compacted event contains chain summary."""
        events = []
        prev_id = None
        for i in range(4):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(prev_id,) if prev_id else (),
                content={'step': i},
                concepts=('chain',),
            )
            events.append(event)
            prev_id = event.id

        compactor = CausalChainCompactor(mock_event_store)
        compacted, _ = compactor.compact_group(events)

        # Access chain summary from snapshot
        snapshot = compacted.content['snapshot']
        assert snapshot['compaction_type'] == 'causal_chain'
        assert snapshot['chain_start'] == events[0].id
        assert snapshot['chain_end'] == events[-1].id
        assert snapshot['chain_length'] == 4

    def test_short_chains_not_compacted(self, mock_event_store):
        """Chains shorter than threshold are not compacted."""
        # Create short chain: e1 -> e2 -> e3
        events = []
        prev_id = None
        for i in range(3):
            event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(prev_id,) if prev_id else (),
                content={'step': i},
                concepts=('chain',),
            )
            events.append(event)
            prev_id = event.id

        mock_event_store.iterate.return_value = iter(events)

        compactor = CausalChainCompactor(mock_event_store, max_chain_length=5)
        groups = compactor.identify_compactable()

        # Chain of 3 <= max of 5, should not be compacted
        assert len(groups) == 0


# =============================================================================
# USER STORY: Compaction Utilities
# =============================================================================

class TestCompactionUtilitiesBehavior:
    """
    User Story: As an operator, I want utilities to help me
    understand when and how to run compaction, so I can
    optimize storage efficiently.
    """

    def test_create_compaction_schedule_returns_list(self, mock_event_store):
        """create_compaction_schedule returns ordered compactor list."""
        mock_event_store.iterate.return_value = iter([])

        schedule = create_compaction_schedule(mock_event_store)

        assert isinstance(schedule, list)
        # Each item is (name, compactor) tuple
        for item in schedule:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_estimate_savings_provides_useful_metrics(self, mock_event_store):
        """estimate_compaction_savings provides actionable metrics."""
        events = [
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('shared', 'unique1'),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={},
                concepts=('shared', 'unique2'),
            ),
        ]
        mock_event_store.iterate.return_value = iter(events)

        savings = estimate_compaction_savings(mock_event_store)

        assert 'total_events' in savings
        assert 'unique_concepts' in savings
        assert 'duplicate_concept_refs' in savings
        assert 'estimated_savings_percent' in savings

    def test_estimate_savings_handles_empty_store(self, mock_event_store):
        """Estimate works on empty store without error."""
        mock_event_store.iterate.return_value = iter([])

        savings = estimate_compaction_savings(mock_event_store)

        assert savings['total_events'] == 0
        assert savings['unique_concepts'] == 0


# =============================================================================
# USER STORY: BaseCompactor Contract
# =============================================================================

class TestBaseCompactorContract:
    """
    User Story: As a developer extending compaction, I want a
    clear contract for implementing custom compactors.
    """

    def test_preserve_marks_events_as_non_compactable(self, mock_event_store):
        """Preserved events are not included in compaction."""
        # Create a concrete implementation for testing
        class TestCompactor(BaseCompactor):
            def identify_compactable(self):
                return []

            def compact_group(self, events):
                return events[0], []

        compactor = TestCompactor(mock_event_store)
        compactor.preserve('important-event-id')

        assert compactor.is_preserved('important-event-id')
        assert not compactor.is_preserved('other-event-id')

    def test_compact_filters_preserved_from_groups(self, mock_event_store):
        """compact() filters out preserved events from groups."""
        # Need 3 events so after filtering 1, we still have 2 (min for compaction)
        events = [
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'id': 'e1'},
                concepts=(),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'id': 'e2'},
                concepts=(),
            ),
            CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=(),
                content={'id': 'e3'},
                concepts=(),
            ),
        ]

        class TestCompactor(BaseCompactor):
            def __init__(self, store, test_events):
                super().__init__(store)
                self._test_events = test_events

            def identify_compactable(self):
                return [self._test_events]

            def compact_group(self, evts):
                # Track what we received
                self.received_events = evts
                # Return a Compaction event for proper behavior
                from cortical.cel.core.events import Compaction
                compacted = Compaction(
                    compressed_events=tuple(e.id for e in evts),
                    snapshot={'test': True},
                    preserved_merkle_root=evts[-1].id,
                )
                return compacted, [e.id for e in evts[:-1]]

        mock_event_store.iterate.return_value = iter(events)
        mock_event_store.append.return_value = MagicMock(value='new-root')
        compactor = TestCompactor(mock_event_store, events)
        compactor.preserve(events[0].id)

        compactor.compact()

        # Should only receive non-preserved events (e2 and e3)
        assert len(compactor.received_events) == 2
        received_ids = {e.id for e in compactor.received_events}
        assert events[0].id not in received_ids
        assert events[1].id in received_ids
        assert events[2].id in received_ids
