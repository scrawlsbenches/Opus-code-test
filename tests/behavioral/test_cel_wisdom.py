"""
Behavioral tests for CEL Wisdom Layer (actual modules).

These tests exercise the real cortical/cel/wisdom/* modules with
user story format for clarity and coverage improvement.

User stories describe:
- WHO needs the feature
- WHAT they need to do
- WHY they need it

Coverage targets:
- wisdom/semantic.py (BloomFilter, SemanticIndex)
- wisdom/dag.py (MerkleDAG, CausalViolationError)
- core/events.py (CognitiveEvent, Observation, Intention)
- core/references.py (MerkleRoot, EventHorizon, TemporalReference)
"""

from __future__ import annotations

import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import pytest

# Import actual CEL modules
from cortical.cel.core.events import (
    CognitiveEvent,
    EventType,
    Observation,
    Intention,
    Fulfillment,
)
from cortical.cel.core.references import (
    MerkleRoot,
    EventHorizon,
    TemporalReference,
    DeferredReference,
    CausalLink,
    ReferenceMode,
    ReferenceSet,
)
from cortical.cel.wisdom.semantic import BloomFilter, BloomSemanticIndex, InvertedIndex
from cortical.cel.wisdom.dag import (
    MerkleDAG,
    CausalViolationError,
    DuplicateEventError,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def bloom_filter():
    """Create a bloom filter with default settings."""
    return BloomFilter(expected_elements=1000, fp_rate=0.01)


@pytest.fixture
def merkle_dag():
    """Create an empty MerkleDAG."""
    return MerkleDAG()


@pytest.fixture
def sample_observation():
    """Create a sample observation event."""
    return Observation(
        content={"type": "test", "value": 42},
        concepts=["test", "observation"],
    )


@pytest.fixture
def sample_intention():
    """Create a sample intention event."""
    return Intention(
        title="Test task creation",
        priority="high",
        category="feature",
        concepts=["task", "creation"],
    )


# =============================================================================
# USER STORY: BLOOM FILTER FOR FAST MEMBERSHIP TESTING
# =============================================================================

class TestBloomFilterBehavior:
    """
    User Story: As a semantic indexer, I want to quickly check if a concept
    MIGHT exist in my index, so I can avoid expensive lookups for
    definitely-not-present items.

    Acceptance Criteria:
    - Can add items to the filter
    - Returns True for items that were added
    - May return True for items NOT added (false positive)
    - NEVER returns False for items that were added (no false negatives)
    """

    def test_added_items_are_found(self, bloom_filter):
        """Items added to filter are always found."""
        concepts = ["neural", "network", "deep_learning", "attention"]

        for concept in concepts:
            bloom_filter.add(concept)

        # All added items MUST be found (no false negatives)
        # BloomFilter uses .contains() or 'in' operator
        for concept in concepts:
            assert bloom_filter.contains(concept), \
                f"False negative: {concept} was added but not found"

    def test_in_operator_works(self, bloom_filter):
        """The 'in' operator works for membership testing."""
        bloom_filter.add("exists")

        assert "exists" in bloom_filter
        # Note: may still return True for non-existent (false positive)
        # but definitely returns True for existing items

    def test_filter_handles_many_items(self, bloom_filter):
        """Filter works correctly with many items."""
        n_items = 500

        for i in range(n_items):
            bloom_filter.add(f"concept_{i}")

        # All items must be found
        for i in range(n_items):
            assert bloom_filter.contains(f"concept_{i}")

        # Check count property (not method)
        assert bloom_filter.count == n_items

    def test_false_positive_rate_is_bounded(self):
        """False positive rate stays within configured bounds."""
        fp_rate = 0.05  # 5% target
        n_elements = 1000

        bf = BloomFilter(expected_elements=n_elements, fp_rate=fp_rate)

        # Add elements
        for i in range(n_elements):
            bf.add(f"exists_{i}")

        # Test for false positives on non-existent items
        false_positives = 0
        n_tests = 10000

        for i in range(n_tests):
            if bf.contains(f"not_exists_{i}"):
                false_positives += 1

        actual_fp_rate = false_positives / n_tests

        # Allow 3x tolerance for statistical variation
        assert actual_fp_rate < fp_rate * 3, \
            f"False positive rate {actual_fp_rate:.3f} exceeds 3x target {fp_rate}"

    def test_empty_filter_returns_false(self):
        """Empty filter returns False for all queries."""
        bf = BloomFilter()
        assert not bf.contains("anything")
        assert not bf.contains("")
        assert bf.count == 0

    def test_filter_size_computed_from_parameters(self):
        """Filter size is computed optimally from parameters."""
        bf_small = BloomFilter(expected_elements=100, fp_rate=0.1)
        bf_large = BloomFilter(expected_elements=10000, fp_rate=0.001)

        # Larger expected elements or lower FP rate = larger filter
        assert bf_large.size > bf_small.size

    def test_filter_serialization(self, bloom_filter):
        """Bloom filter can be serialized to bytes."""
        bloom_filter.add("test")

        data = bloom_filter.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_filter_deserialization(self):
        """Bloom filter can be deserialized from bytes."""
        original = BloomFilter(expected_elements=100, fp_rate=0.01)
        original.add("preserved")

        data = original.to_bytes()
        restored = BloomFilter.from_bytes(data, 100, 0.01)

        assert restored.contains("preserved")

    def test_estimated_fp_rate(self, bloom_filter):
        """Can estimate current false positive rate."""
        # Empty filter
        assert bloom_filter.estimated_fp_rate == 0.0

        # After adding items
        for i in range(100):
            bloom_filter.add(f"item_{i}")

        rate = bloom_filter.estimated_fp_rate
        assert 0 < rate < 1


# =============================================================================
# USER STORY: MERKLE DAG FOR CAUSAL EVENT ORDERING
# =============================================================================

class TestMerkleDAGBehavior:
    """
    User Story: As a reasoning system, I want to store events in a
    causally-ordered DAG, so I can trace why things happened and
    ensure consistency.

    Acceptance Criteria:
    - Events can be added to the DAG
    - Events get content-addressed IDs
    - Parent events must exist before children
    - Can traverse ancestors and descendants
    - Duplicate events are handled gracefully
    """

    def test_add_event_returns_merkle_root(self, merkle_dag, sample_observation):
        """Adding event returns its content-addressed ID."""
        root = merkle_dag.add(sample_observation)

        assert isinstance(root, MerkleRoot)
        assert len(root.value) == 64  # SHA256 hex
        assert root.value == sample_observation.id

    def test_same_event_produces_same_id(self, merkle_dag):
        """Content-addressed: same content = same ID."""
        event1 = Observation(
            content={"data": "test"},
            timestamp="2025-01-01T00:00:00+00:00",
        )
        event2 = Observation(
            content={"data": "test"},
            timestamp="2025-01-01T00:00:00+00:00",
        )

        assert event1.id == event2.id

    def test_event_retrieval_by_id(self, merkle_dag, sample_observation):
        """Events can be retrieved by their ID."""
        root = merkle_dag.add(sample_observation)

        retrieved = merkle_dag.get(root.value)
        assert retrieved is not None
        assert retrieved.id == sample_observation.id

    def test_causal_parents_must_exist(self, merkle_dag, sample_observation):
        """Events cannot reference non-existent parents."""
        # Create event with fake parent
        child = Observation(
            content={"child": True},
            causal_parents=["nonexistent_parent_id"],
        )

        with pytest.raises(CausalViolationError) as exc_info:
            merkle_dag.add(child)

        assert "nonexistent_parent_id" in exc_info.value.missing_parents

    def test_valid_causal_chain(self, merkle_dag):
        """Events with valid parents are accepted."""
        parent = Observation(content={"step": 1})
        parent_root = merkle_dag.add(parent)

        child = Observation(
            content={"step": 2},
            causal_parents=[parent_root.value],
        )
        child_root = merkle_dag.add(child)

        # Verify chain
        retrieved_child = merkle_dag.get(child_root.value)
        assert parent_root.value in retrieved_child.causal_parents

    def test_ancestor_traversal(self, merkle_dag):
        """Can find all ancestors of an event."""
        # Create chain: e1 <- e2 <- e3
        e1 = Observation(content={"gen": 1})
        r1 = merkle_dag.add(e1)

        e2 = Observation(content={"gen": 2}, causal_parents=[r1.value])
        r2 = merkle_dag.add(e2)

        e3 = Observation(content={"gen": 3}, causal_parents=[r2.value])
        r3 = merkle_dag.add(e3)

        # ancestors() yields CognitiveEvent objects
        ancestors = list(merkle_dag.ancestors(r3.value))
        ancestor_ids = {a.id for a in ancestors}

        assert r1.value in ancestor_ids
        assert r2.value in ancestor_ids
        assert r3.value not in ancestor_ids  # Not ancestor of itself

    def test_ancestor_with_depth_limit(self, merkle_dag):
        """Ancestor traversal respects depth limit."""
        # Create chain: e1 <- e2 <- e3 <- e4
        e1 = Observation(content={"gen": 1})
        r1 = merkle_dag.add(e1)

        e2 = Observation(content={"gen": 2}, causal_parents=[r1.value])
        r2 = merkle_dag.add(e2)

        e3 = Observation(content={"gen": 3}, causal_parents=[r2.value])
        r3 = merkle_dag.add(e3)

        e4 = Observation(content={"gen": 4}, causal_parents=[r3.value])
        r4 = merkle_dag.add(e4)

        # Depth 1 should only get e3
        shallow = list(merkle_dag.ancestors(r4.value, depth=1))
        assert len(shallow) == 1
        assert shallow[0].id == r3.value

    def test_descendant_traversal(self, merkle_dag):
        """Can find all descendants of an event."""
        # Create tree: e1 -> (e2, e3)
        e1 = Observation(content={"root": True})
        r1 = merkle_dag.add(e1)

        e2 = Observation(content={"branch": "left"}, causal_parents=[r1.value])
        r2 = merkle_dag.add(e2)

        e3 = Observation(content={"branch": "right"}, causal_parents=[r1.value])
        r3 = merkle_dag.add(e3)

        # descendants() yields CognitiveEvent objects
        descendants = list(merkle_dag.descendants(r1.value))
        desc_ids = {d.id for d in descendants}

        assert r2.value in desc_ids
        assert r3.value in desc_ids
        assert r1.value not in desc_ids

    def test_head_tracking(self, merkle_dag):
        """DAG tracks current heads (events with no children)."""
        e1 = Observation(content={"first": True})
        r1 = merkle_dag.add(e1)

        # e1 is the only head (heads is a set of strings)
        assert r1.value in merkle_dag.heads

        e2 = Observation(content={"second": True}, causal_parents=[r1.value])
        r2 = merkle_dag.add(e2)

        # Now e2 is head, e1 is not
        assert r2.value in merkle_dag.heads
        assert r1.value not in merkle_dag.heads

    def test_get_heads_returns_merkle_roots(self, merkle_dag):
        """get_heads() returns list of MerkleRoot objects."""
        e1 = Observation(content={"head": 1})
        merkle_dag.add(e1)

        heads = merkle_dag.get_heads()
        assert len(heads) == 1
        assert isinstance(heads[0], MerkleRoot)

    def test_get_latest_returns_most_recent(self, merkle_dag):
        """get_latest() returns most recent head by timestamp."""
        e1 = Observation(content={"first": True}, timestamp="2025-01-01T00:00:00+00:00")
        merkle_dag.add(e1)

        e2 = Observation(content={"second": True}, timestamp="2025-01-01T01:00:00+00:00")
        merkle_dag.add(e2)

        latest = merkle_dag.get_latest()
        assert latest is not None
        assert latest.value == e2.id

    def test_contains_method(self, merkle_dag, sample_observation):
        """contains() checks if event exists in DAG."""
        assert not merkle_dag.contains("nonexistent")

        root = merkle_dag.add(sample_observation)
        assert merkle_dag.contains(root.value)

    def test_duplicate_event_raises_error(self, merkle_dag, sample_observation):
        """Adding same event twice raises DuplicateEventError."""
        merkle_dag.add(sample_observation)

        with pytest.raises(DuplicateEventError):
            merkle_dag.add(sample_observation)

    def test_count_property(self, merkle_dag):
        """count property returns number of events."""
        assert merkle_dag.count == 0

        e1 = Observation(content={"a": 1})
        merkle_dag.add(e1)
        assert merkle_dag.count == 1

        e2 = Observation(content={"b": 2})
        merkle_dag.add(e2)
        assert merkle_dag.count == 2

    def test_causal_order_iteration(self, merkle_dag):
        """causal_order() yields events in topological order."""
        # Create: e1 -> e2 -> e3
        e1 = Observation(content={"step": 1})
        r1 = merkle_dag.add(e1)

        e2 = Observation(content={"step": 2}, causal_parents=[r1.value])
        r2 = merkle_dag.add(e2)

        e3 = Observation(content={"step": 3}, causal_parents=[r2.value])
        merkle_dag.add(e3)

        ordered = list(merkle_dag.causal_order())

        # Parents must come before children
        ids = [e.id for e in ordered]
        assert ids.index(e1.id) < ids.index(e2.id) < ids.index(e3.id)


# =============================================================================
# USER STORY: COGNITIVE EVENTS FOR STATE CHANGES
# =============================================================================

class TestCognitiveEventBehavior:
    """
    User Story: As a cognitive system, I want to record state changes
    as typed, immutable events, so I can reconstruct history and
    reason about causality.

    Acceptance Criteria:
    - Events are immutable after creation
    - Events have typed variants (Observation, Intention, etc.)
    - Events can be serialized/deserialized
    - Events capture concepts for semantic indexing
    """

    def test_observation_captures_external_events(self):
        """Observation events record external happenings."""
        obs = Observation(
            content={
                "type": "file_change",
                "path": "/src/main.py",
                "action": "modified",
            },
            concepts=["file", "source_code", "modification"],
        )

        assert obs.event_type == EventType.OBSERVATION
        assert "file_change" == obs.content["type"]
        assert "file" in obs.concepts

    def test_intention_captures_goals(self):
        """Intention events record things that should happen."""
        intent = Intention(
            title="Complete task T-001",
            priority="high",
            category="feature",
            concepts=["task", "completion"],
        )

        assert intent.event_type == EventType.INTENTION
        assert intent.title == "Complete task T-001"
        assert intent.priority == "high"
        assert intent.category == "feature"

    def test_intention_extracts_concepts_from_title(self):
        """Intention auto-extracts concepts from title if none provided."""
        intent = Intention(title="Implement neural network layer")

        # Should extract concepts from title (excluding stop words)
        assert len(intent.concepts) > 0

    def test_fulfillment_completes_intentions(self):
        """Fulfillment events mark intentions as done."""
        intent = Intention(title="Test task")
        intent_id = intent.id

        fulfill = Fulfillment(
            intention_id=intent_id,
            result={"success": True},
        )

        assert fulfill.event_type == EventType.FULFILLMENT
        assert fulfill.intention_id == intent_id
        # Fulfilled intention is automatically a causal parent
        assert intent_id in fulfill.causal_parents

    def test_fulfillment_was_successful_property(self):
        """Fulfillment tracks success status."""
        intent = Intention(title="Test task")

        success = Fulfillment(intention_id=intent.id, result={"success": True})
        assert success.was_successful

        failure = Fulfillment(intention_id=intent.id, result={"success": False})
        assert not failure.was_successful

    def test_events_are_immutable(self, sample_observation):
        """Events cannot be modified after creation."""
        with pytest.raises((TypeError, AttributeError)):
            sample_observation.content = {"modified": True}

    def test_event_serialization_roundtrip(self, sample_observation):
        """Events can be serialized and deserialized."""
        data = sample_observation.to_dict()

        # Verify dict contains expected fields
        assert "id" in data
        assert "timestamp" in data
        assert "event_type" in data
        assert "content" in data
        assert "concepts" in data

        # Roundtrip
        restored = CognitiveEvent.from_dict(data)
        assert restored.id == sample_observation.id
        assert restored.content == sample_observation.content

    def test_event_id_is_content_addressed(self, sample_observation):
        """Event ID is deterministic hash of content."""
        id1 = sample_observation.id
        id2 = sample_observation.id  # Should be cached

        assert id1 == id2
        assert len(id1) == 64  # SHA256 hex

    def test_concepts_extracted_for_indexing(self):
        """Events expose concepts for semantic indexing."""
        obs = Observation(
            content={"data": "test"},
            concepts=["neural", "network", "deep_learning"],
        )

        assert len(obs.concepts) == 3
        assert "neural" in obs.concepts
        assert isinstance(obs.concepts, tuple)  # Immutable

    def test_event_merkle_root_property(self, sample_observation):
        """Events expose ID as MerkleRoot type."""
        root = sample_observation.merkle_root
        assert isinstance(root, MerkleRoot)
        assert root.value == sample_observation.id

    def test_event_horizon_property(self, sample_observation):
        """Events expose horizon for temporal queries."""
        horizon = sample_observation.horizon
        assert isinstance(horizon, EventHorizon)
        assert horizon.event_id == sample_observation.id

    def test_event_with_parent(self, sample_observation):
        """Events can be created with additional parent."""
        parent_id = "parent123"
        child = sample_observation.with_parent(parent_id)

        assert parent_id in child.causal_parents
        # Original unchanged
        assert parent_id not in sample_observation.causal_parents


# =============================================================================
# USER STORY: TEMPORAL REFERENCES FOR STABLE REASONING
# =============================================================================

class TestTemporalReferenceBehavior:
    """
    User Story: As a reasoning agent, I want to reference entity state
    at a specific point in time, so my reasoning remains stable even
    if the entity changes later.

    Acceptance Criteria:
    - Can create reference to entity at specific horizon
    - Reference contains entity ID and horizon
    - MerkleRoot provides identity verification
    """

    def test_merkle_root_identity(self):
        """MerkleRoot provides content-based identity."""
        root = MerkleRoot("abc123def456")

        assert str(root) == "abc123def456"
        assert root.short == "abc123de"
        assert root.matches(MerkleRoot("abc123def456"))
        assert not root.matches(MerkleRoot("different"))

    def test_event_horizon_as_time_marker(self):
        """EventHorizon marks a point in the event stream."""
        horizon = EventHorizon(event_id="event_123")

        assert horizon.event_id == "event_123"
        assert not horizon.is_head

    def test_event_horizon_head_marking(self):
        """EventHorizon can be marked as branch head."""
        head = EventHorizon(event_id="head_event", is_head=True)
        assert head.is_head
        assert "(HEAD)" in str(head)

    def test_event_horizon_serialization(self):
        """EventHorizon can be serialized."""
        horizon = EventHorizon(event_id="test_123", is_head=True)
        data = horizon.to_dict()

        assert data["event_id"] == "test_123"
        assert data["is_head"] == True

        restored = EventHorizon.from_dict(data)
        assert restored.event_id == horizon.event_id
        assert restored.is_head == horizon.is_head

    def test_temporal_reference_captures_snapshot(self):
        """TemporalReference captures entity at specific time."""
        horizon = EventHorizon(event_id="event_at_creation")
        ref = TemporalReference(
            entity_id="task_001",
            horizon=horizon,
            entity_type="task",
        )

        assert ref.entity_id == "task_001"
        assert ref.horizon.event_id == "event_at_creation"
        assert ref.entity_type == "task"

    def test_temporal_reference_serialization(self):
        """TemporalReference can be serialized."""
        horizon = EventHorizon(event_id="event_123")
        ref = TemporalReference(
            entity_id="entity_456",
            horizon=horizon,
            entity_type="task",
        )

        data = ref.to_dict()
        assert data["entity_id"] == "entity_456"
        assert "horizon" in data

        restored = TemporalReference.from_dict(data)
        assert restored.entity_id == ref.entity_id
        assert restored.horizon.event_id == ref.horizon.event_id

    def test_causal_link_connects_events(self):
        """CausalLink represents edge in causal graph."""
        link = CausalLink(
            from_event="parent_event_id",
            to_event="child_event_id",
            link_type="PARENT",
        )

        assert link.from_event == "parent_event_id"
        assert link.to_event == "child_event_id"
        assert link.link_type == "PARENT"

    def test_causal_link_root_properties(self):
        """CausalLink exposes MerkleRoot accessors."""
        link = CausalLink(from_event="from_abc", to_event="to_xyz")

        assert link.from_root.value == "from_abc"
        assert link.to_root.value == "to_xyz"

    def test_deferred_reference_lifecycle(self):
        """DeferredReference resolves after dependencies."""
        ref = DeferredReference(
            entity_id="config",
            after=["task_1", "task_2"],
        )

        assert not ref.is_resolved()
        assert ref.mode == ReferenceMode.DEFERRED

    def test_reference_set_management(self):
        """ReferenceSet tracks multiple references."""
        ref_set = ReferenceSet()

        horizon = EventHorizon(event_id="now")
        ref_set.add_snapshot("entity_1", horizon, "task")
        ref_set.add_deferred("entity_2", ["dep_1"], "config")

        assert len(ref_set.temporal) == 1
        assert len(ref_set.deferred) == 1
        assert not ref_set.all_resolved()  # Deferred not yet resolved

        deps = ref_set.pending_dependencies()
        assert "dep_1" in deps


# =============================================================================
# USER STORY: SEMANTIC INDEX FOR CONCEPT SEARCH
# =============================================================================

class TestSemanticIndexBehavior:
    """
    User Story: As a query system, I want to quickly find events
    by concept, so I can answer semantic queries efficiently.

    Acceptance Criteria:
    - Events are indexed by their concepts
    - Can search for events by concept
    - Bloom filter provides fast negative check
    - Inverted index provides actual event IDs
    """

    def test_index_event_by_concepts(self):
        """Events are indexed by their concepts."""
        index = BloomSemanticIndex()

        event = Observation(
            content={"data": "test"},
            concepts=["neural", "network"],
        )

        index.index_event(event)

        # probably_contains uses bloom filter
        assert index.probably_contains("neural")
        assert index.probably_contains("network")

    def test_search_returns_matching_events(self):
        """Search returns all events with matching concept."""
        index = BloomSemanticIndex()

        events = [
            Observation(content={"i": i}, concepts=["common", f"unique_{i}"])
            for i in range(5)
        ]

        for event in events:
            index.index_event(event)

        # Search uses terms extracted from query
        # Note: search() extracts words from query text
        results = index.search("common")
        assert len(results) == 5

    def test_bloom_filter_fast_negative(self):
        """Bloom filter quickly rejects definitely-not-present."""
        index = BloomSemanticIndex()

        event = Observation(content={"x": 1}, concepts=["exists"])
        index.index_event(event)

        # probably_contains uses bloom filter for fast check
        assert not index.probably_contains("definitely_not_present")
        assert index.probably_contains("exists")

    def test_inverted_index_direct_usage(self):
        """InvertedIndex can be used directly for exact matching."""
        index = InvertedIndex()

        # Add entries
        index.add("concept_a", "event_1")
        index.add("concept_a", "event_2")
        index.add("concept_b", "event_2")

        # search() returns set of event IDs
        results_a = index.search("concept_a")
        results_b = index.search("concept_b")

        assert "event_1" in results_a
        assert "event_2" in results_a
        assert "event_2" in results_b
        assert "event_1" not in results_b

    def test_inverted_index_search_all(self):
        """InvertedIndex supports multi-term search."""
        index = InvertedIndex()

        index.add("term_a", "event_1")
        index.add("term_b", "event_1")
        index.add("term_a", "event_2")

        # OR search (any term matches)
        results_or = index.search_all(["term_a", "term_b"], require_all=False)
        assert "event_1" in results_or
        assert "event_2" in results_or

        # AND search (all terms required)
        results_and = index.search_all(["term_a", "term_b"], require_all=True)
        assert "event_1" in results_and
        assert "event_2" not in results_and

    def test_inverted_index_remove_event(self):
        """InvertedIndex can remove event's terms."""
        index = InvertedIndex()

        index.add("term", "event_1")
        index.add("term", "event_2")

        index.remove_event("event_1")

        results = index.search("term")
        assert "event_1" not in results
        assert "event_2" in results

    def test_semantic_index_stats(self):
        """Semantic index provides statistics."""
        index = BloomSemanticIndex()

        for i in range(10):
            event = Observation(content={"i": i}, concepts=[f"concept_{i}"])
            index.index_event(event)

        stats = index.stats
        assert stats["event_count"] == 10
        assert stats["term_count"] >= 10

    def test_similar_to_finds_related(self):
        """similar_to() finds entities with overlapping terms."""
        index = BloomSemanticIndex()

        # Create events with overlapping concepts
        e1 = Observation(content={"id": 1}, concepts=["neural", "network", "deep"])
        e2 = Observation(content={"id": 2}, concepts=["neural", "network", "shallow"])
        e3 = Observation(content={"id": 3}, concepts=["completely", "different"])

        for e in [e1, e2, e3]:
            index.index_event(e)

        similar = index.similar_to(e1.id, limit=5)

        # e2 should be similar (shares neural, network)
        similar_ids = [s[0] for s in similar]
        assert e2.id in similar_ids


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """TDD-style tests for edge cases and boundary conditions."""

    def test_empty_concepts_allowed(self):
        """Events can have no concepts."""
        event = Observation(content={"data": "test"}, concepts=[])
        assert len(event.concepts) == 0

    def test_very_long_concept_names(self):
        """Long concept names are handled."""
        long_concept = "x" * 1000
        bf = BloomFilter()
        bf.add(long_concept)
        assert bf.contains(long_concept)

    def test_unicode_concepts(self):
        """Unicode in concepts works correctly."""
        bf = BloomFilter()
        concepts = ["日本語", "emoji🎉", "Ελληνικά"]

        for c in concepts:
            bf.add(c)

        for c in concepts:
            assert bf.contains(c)

    def test_empty_dag_operations(self, merkle_dag):
        """Empty DAG handles queries gracefully."""
        assert merkle_dag.get("nonexistent") is None
        assert list(merkle_dag.ancestors("nonexistent")) == []
        assert list(merkle_dag.descendants("nonexistent")) == []
        assert len(merkle_dag.heads) == 0
        assert merkle_dag.get_latest() is None

    def test_dag_with_multiple_roots(self, merkle_dag):
        """DAG can have multiple root events (no parents)."""
        e1 = Observation(content={"root": 1})
        e2 = Observation(content={"root": 2})

        r1 = merkle_dag.add(e1)
        r2 = merkle_dag.add(e2)

        # Both are heads (no children yet)
        assert r1.value in merkle_dag.heads
        assert r2.value in merkle_dag.heads

    def test_event_with_multiple_parents(self, merkle_dag):
        """Events can have multiple causal parents (merge)."""
        e1 = Observation(content={"branch": "left"})
        e2 = Observation(content={"branch": "right"})

        r1 = merkle_dag.add(e1)
        r2 = merkle_dag.add(e2)

        # Merge event with both parents
        merge = Observation(
            content={"merge": True},
            causal_parents=[r1.value, r2.value],
        )
        rm = merkle_dag.add(merge)

        retrieved = merkle_dag.get(rm.value)
        assert len(retrieved.causal_parents) == 2
        assert r1.value in retrieved.causal_parents
        assert r2.value in retrieved.causal_parents

    def test_empty_inverted_index(self):
        """Empty inverted index returns empty results."""
        index = InvertedIndex()

        assert index.search("anything") == set()
        assert index.term_count == 0
        assert index.event_count == 0

    def test_inverted_index_serialization(self):
        """InvertedIndex can be serialized and restored."""
        index = InvertedIndex()
        index.add("term", "event_1")
        index.add("term", "event_2")

        data = index.to_dict()
        restored = InvertedIndex.from_dict(data)

        assert restored.search("term") == {"event_1", "event_2"}
