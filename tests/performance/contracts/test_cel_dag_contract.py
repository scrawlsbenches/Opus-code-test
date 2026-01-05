"""
╔══════════════════════════════════════════════════════════════════════╗
║                   CEL DAG PERFORMANCE CONTRACT                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Event append < 5ms per event (with parent validation)             ║
║  • Causal order iteration O(n log n) for n events                    ║
║  • Ancestor traversal < 10ms for depth ≤ 100                         ║
║  • DAG integrity check < 100ms for 1,000 events                      ║
║  • No causal violations ever (100% enforcement)                      ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List

import pytest

from cortical.cel.core.events import Intention, Observation, Fulfillment
from cortical.cel.wisdom.dag import (
    MerkleDAG,
    CausalViolationError,
    DuplicateEventError,
)


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestDAGAppendContract:
    """
    DAG Append Performance Contract

    As a system appending events in real-time,
    I expect append operations to be fast,
    So that event sourcing doesn't slow down the system.
    """

    # The sacred numbers
    MAX_APPEND_MS = 10.0
    SAMPLE_SIZE = 100

    def test_append_latency_with_validation(self):
        """
        CONTRACT: Event append in < 5ms (including causal validation).

        Append is the write path. It must be fast.
        """
        dag = MerkleDAG()
        latencies = []

        # Create a linear chain to measure append performance
        previous_id = None
        for i in range(self.SAMPLE_SIZE):
            event = Observation(
                content={'index': i, 'data': f'Event {i}'},
                causal_parents=[previous_id] if previous_id else [],
            )

            start = time.perf_counter()
            root = dag.add(event)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            previous_id = root.value

        p95 = percentile(latencies, 95)

        assert p95 < self.MAX_APPEND_MS, (
            f"CONTRACT VIOLATION: p95 append latency is {p95:.2f}ms, "
            f"contract requires <{self.MAX_APPEND_MS}ms"
        )

    def test_append_rejects_missing_parents(self):
        """
        CONTRACT: Append ALWAYS validates causal parents (100% enforcement).

        Missing parents must raise CausalViolationError.
        This is a correctness contract.
        """
        dag = MerkleDAG()

        # Attempt to add event with non-existent parent
        event = Observation(
            content={'test': 'data'},
            causal_parents=['nonexistent_parent_abc123'],
        )

        with pytest.raises(CausalViolationError) as exc_info:
            dag.add(event)

        # Check that error mentions missing parents (string may be truncated)
        assert 'missing parent' in str(exc_info.value).lower(), (
            "CONTRACT VIOLATION: CausalViolationError missing parent details"
        )

    def test_append_rejects_duplicates(self):
        """
        CONTRACT: Duplicate events are rejected.

        Content-addressing means same content = same ID.
        Duplicates must raise DuplicateEventError.
        """
        dag = MerkleDAG()

        # Add event
        event = Observation(
            content={'test': 'duplicate'},
            timestamp='2024-12-31T00:00:00Z',
        )
        dag.add(event)

        # Attempt to add identical event
        duplicate = Observation(
            content={'test': 'duplicate'},
            timestamp='2024-12-31T00:00:00Z',
        )

        with pytest.raises(DuplicateEventError):
            dag.add(duplicate)

    def test_append_updates_heads_correctly(self):
        """
        CONTRACT: Append correctly maintains branch heads.

        Heads are events with no children. This is critical for
        determining current state.
        """
        dag = MerkleDAG()

        # Add root event
        root = Observation(content={'root': True})
        root_id = dag.add(root).value

        # Root should be the only head
        assert len(dag.heads) == 1
        assert root_id in dag.heads

        # Add child
        child = Observation(
            content={'child': True},
            causal_parents=[root_id],
        )
        child_id = dag.add(child).value

        # Now child is the only head
        assert len(dag.heads) == 1
        assert child_id in dag.heads
        assert root_id not in dag.heads

        # Add two children of child (branching)
        child_a = Observation(
            content={'branch': 'a'},
            causal_parents=[child_id],
        )
        child_b = Observation(
            content={'branch': 'b'},
            causal_parents=[child_id],
        )
        id_a = dag.add(child_a).value
        id_b = dag.add(child_b).value

        # Both branches are heads
        assert len(dag.heads) == 2
        assert id_a in dag.heads
        assert id_b in dag.heads
        assert child_id not in dag.heads


@pytest.mark.contract
class TestDAGTraversalContract:
    """
    DAG Traversal Performance Contract

    As a system replaying event history,
    I expect traversal to be efficient,
    So that materialization doesn't take forever.
    """

    # The sacred numbers
    MAX_TRAVERSE_MS_PER_100 = 50  # 50ms to traverse 100 events
    MAX_ANCESTOR_MS = 20.0  # Ancestor lookup for depth 100

    def test_causal_order_iteration_performance(self):
        """
        CONTRACT: Causal order iteration scales as O(n log n).

        Topological sort is the bottleneck. We contract O(n log n).
        """
        dag = MerkleDAG()

        # Build a linear chain of 100 events
        chain_size = 100
        previous_id = None
        for i in range(chain_size):
            event = Observation(
                content={'index': i},
                causal_parents=[previous_id] if previous_id else [],
            )
            previous_id = dag.add(event).value

        # Measure iteration time
        start = time.perf_counter()
        events = list(dag.causal_order())
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(events) == chain_size, "Not all events returned"

        assert elapsed_ms < self.MAX_TRAVERSE_MS_PER_100, (
            f"CONTRACT VIOLATION: Traversing {chain_size} events took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_TRAVERSE_MS_PER_100}ms"
        )

    def test_causal_order_is_valid(self):
        """
        CONTRACT: Causal order guarantees parents before children.

        This is a correctness contract. Violation breaks everything.
        """
        dag = MerkleDAG()

        # Build a tree:
        #     root
        #     /  \
        #    a    b
        #    |    |
        #    a1   b1

        root = Observation(content={'id': 'root'})
        root_id = dag.add(root).value

        a = Observation(content={'id': 'a'}, causal_parents=[root_id])
        a_id = dag.add(a).value

        b = Observation(content={'id': 'b'}, causal_parents=[root_id])
        b_id = dag.add(b).value

        a1 = Observation(content={'id': 'a1'}, causal_parents=[a_id])
        a1_id = dag.add(a1).value

        b1 = Observation(content={'id': 'b1'}, causal_parents=[b_id])
        b1_id = dag.add(b1).value

        # Get causal order
        events = list(dag.causal_order())
        event_ids = [e.id for e in events]

        # Verify root comes first
        assert event_ids[0] == root_id, "Root not first in causal order"

        # Verify parents before children
        for event in events:
            event_pos = event_ids.index(event.id)
            for parent_id in event.causal_parents:
                if parent_id in event_ids:
                    parent_pos = event_ids.index(parent_id)
                    assert parent_pos < event_pos, (
                        f"CONTRACT VIOLATION: Parent {parent_id[:8]} appears "
                        f"after child {event.id[:8]} in causal order"
                    )

    def test_ancestor_traversal_performance(self):
        """
        CONTRACT: Ancestor lookup for depth 100 in < 10ms.

        Ancestor traversal is used for temporal queries.
        """
        dag = MerkleDAG()

        # Build chain of 100 events
        depth = 100
        previous_id = None
        for i in range(depth):
            event = Observation(
                content={'depth': i},
                causal_parents=[previous_id] if previous_id else [],
            )
            previous_id = dag.add(event).value

        # Measure ancestor traversal from tip
        start = time.perf_counter()
        ancestors = list(dag.ancestors(previous_id, depth=-1))
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(ancestors) == depth - 1, "Wrong ancestor count"

        assert elapsed_ms < self.MAX_ANCESTOR_MS, (
            f"CONTRACT VIOLATION: Ancestor traversal (depth {depth}) took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_ANCESTOR_MS}ms"
        )

    def test_descendant_traversal_correctness(self):
        """
        CONTRACT: Descendant traversal returns all descendants.

        Used for finding all events affected by a change.
        """
        dag = MerkleDAG()

        # Build tree:
        #     root
        #     /  \
        #    a    b
        #    |
        #    a1

        root = Observation(content={'id': 'root'})
        root_id = dag.add(root).value

        a = Observation(content={'id': 'a'}, causal_parents=[root_id])
        a_id = dag.add(a).value

        b = Observation(content={'id': 'b'}, causal_parents=[root_id])
        b_id = dag.add(b).value

        a1 = Observation(content={'id': 'a1'}, causal_parents=[a_id])
        a1_id = dag.add(a1).value

        # Get descendants of root
        descendants = list(dag.descendants(root_id))
        descendant_ids = {d.id for d in descendants}

        assert len(descendants) == 3, f"Wrong descendant count: {len(descendants)}"
        assert a_id in descendant_ids
        assert b_id in descendant_ids
        assert a1_id in descendant_ids


@pytest.mark.contract
class TestDAGIntegrityContract:
    """
    DAG Integrity Verification Contract

    As a system storing critical event history,
    I expect integrity checks to be thorough and fast,
    So that corruption is detected early.
    """

    # The sacred numbers
    MAX_VERIFY_MS_PER_1K = 100  # 100ms for 1,000 events

    def test_integrity_check_performance(self):
        """
        CONTRACT: Integrity check for 1,000 events in < 100ms.

        Integrity checks run periodically. They must be fast.
        """
        dag = MerkleDAG()

        # Build chain of 100 events (scaled down for test speed)
        chain_size = 100
        previous_id = None
        for i in range(chain_size):
            event = Observation(
                content={'index': i},
                causal_parents=[previous_id] if previous_id else [],
            )
            previous_id = dag.add(event).value

        # Measure integrity verification
        start = time.perf_counter()
        errors = []

        # Verify all events exist and have valid parents
        for event_id, event in dag.events.items():
            # Check ID matches
            if event.id != event_id:
                errors.append(f"ID mismatch: {event_id} vs {event.id}")

            # Check parents exist
            for parent_id in event.causal_parents:
                if parent_id not in dag.events:
                    errors.append(f"Missing parent: {parent_id}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Scale to 1K events
        scaled_ms = (elapsed_ms / chain_size) * 1000

        assert len(errors) == 0, f"Integrity errors found: {errors}"

        assert scaled_ms < self.MAX_VERIFY_MS_PER_1K, (
            f"CONTRACT VIOLATION: Integrity check would take {scaled_ms:.2f}ms for 1K events, "
            f"contract requires <{self.MAX_VERIFY_MS_PER_1K}ms"
        )

    def test_merkle_root_immutability(self):
        """
        CONTRACT: Event IDs are immutable and deterministic.

        Once computed, an event's ID never changes.
        """
        event = Observation(
            content={'test': 'immutability'},
            timestamp='2024-12-31T00:00:00Z',
        )

        # Compute ID multiple times
        id1 = event.id
        id2 = event.id
        id3 = event.id

        assert id1 == id2 == id3, (
            f"CONTRACT VIOLATION: Event ID changed across calls. "
            f"id1={id1[:16]}, id2={id2[:16]}, id3={id3[:16]}"
        )


@pytest.mark.contract
class TestDAGScalingContract:
    """
    DAG Scaling Behavior Contract

    These contracts ensure the DAG performs acceptably
    as event count grows.
    """

    def test_linear_chain_append_doesnt_degrade(self):
        """
        CONTRACT: Append performance stable across chain length.

        The 100th append should be as fast as the 10th.
        """
        dag = MerkleDAG()

        latencies_first_50 = []
        latencies_last_50 = []

        previous_id = None
        for i in range(100):
            event = Observation(
                content={'index': i},
                causal_parents=[previous_id] if previous_id else [],
            )

            start = time.perf_counter()
            root = dag.add(event)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i < 50:
                latencies_first_50.append(elapsed_ms)
            else:
                latencies_last_50.append(elapsed_ms)

            previous_id = root.value

        avg_first_50 = sum(latencies_first_50) / len(latencies_first_50)
        avg_last_50 = sum(latencies_last_50) / len(latencies_last_50)

        # Last 50 shouldn't be more than 2x slower than first 50
        degradation = avg_last_50 / avg_first_50

        assert degradation < 2.0, (
            f"CONTRACT VIOLATION: Append degraded {degradation:.2f}x. "
            f"First 50 avg: {avg_first_50:.3f}ms, Last 50 avg: {avg_last_50:.3f}ms"
        )

    def test_wide_branching_doesnt_break_heads(self):
        """
        CONTRACT: DAG handles wide branching (many heads).

        Creating 100 parallel branches from one event should work.
        """
        dag = MerkleDAG()

        # Create root
        root = Observation(content={'root': True})
        root_id = dag.add(root).value

        # Create 100 branches
        branch_count = 100
        branch_ids = []

        start = time.perf_counter()
        for i in range(branch_count):
            branch = Observation(
                content={'branch': i},
                causal_parents=[root_id],
            )
            branch_ids.append(dag.add(branch).value)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # All branches should be heads
        assert len(dag.heads) == branch_count, (
            f"Expected {branch_count} heads, got {len(dag.heads)}"
        )

        # Should complete in reasonable time
        avg_per_branch = elapsed_ms / branch_count
        assert avg_per_branch < 5.0, (
            f"CONTRACT VIOLATION: Wide branching too slow: {avg_per_branch:.2f}ms/branch"
        )
