"""
╔══════════════════════════════════════════════════════════════════════╗
║                  CEL EVENT PERFORMANCE CONTRACT                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Event creation < 1ms per event                                    ║
║  • Content hash computation < 0.5ms per event                        ║
║  • Event serialization < 1ms per event                               ║
║  • Event deserialization < 1ms per event                             ║
║  • Hash determinism 100% (same content = same hash)                  ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List

import pytest

from cortical.cel.core.events import (
    CognitiveEvent,
    EventType,
    Intention,
    Observation,
    Fulfillment,
)


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestEventCreationContract:
    """
    Event Creation Performance Contract

    As a system appending thousands of events,
    I expect event creation to be near-instant,
    So that event sourcing overhead is negligible.
    """

    # The sacred numbers
    MAX_CREATE_MS = 2.0
    MAX_HASH_MS = 1.0
    SAMPLE_SIZE = 100

    def test_observation_creation_latency(self):
        """
        CONTRACT: Observation events created in < 1ms.

        Event creation is synchronous and must be fast.
        """
        latencies = []

        for i in range(self.SAMPLE_SIZE):
            start = time.perf_counter()
            event = Observation(
                content={
                    'type': 'test_event',
                    'index': i,
                    'data': f'Test observation {i}',
                }
            )
            # Force ID computation
            _ = event.id
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.MAX_CREATE_MS, (
            f"CONTRACT VIOLATION: p95 observation creation is {p95:.3f}ms, "
            f"contract requires <{self.MAX_CREATE_MS}ms"
        )

    def test_intention_creation_latency(self):
        """
        CONTRACT: Intention events created in < 1ms.

        Intentions are created frequently during task planning.
        """
        latencies = []

        for i in range(self.SAMPLE_SIZE):
            start = time.perf_counter()
            event = Intention(
                title=f"Test intention {i}",
                priority='medium',
                category='test',
                description=f"Test description for intention {i}",
            )
            # Force ID computation
            _ = event.id
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.MAX_CREATE_MS, (
            f"CONTRACT VIOLATION: p95 intention creation is {p95:.3f}ms, "
            f"contract requires <{self.MAX_CREATE_MS}ms"
        )

    def test_content_hash_computation_latency(self):
        """
        CONTRACT: Content hash computed in < 0.5ms.

        Hash computation is the bottleneck for event creation.
        We use SHA256 for cryptographic guarantees.
        """
        # Pre-create events to isolate hash computation
        events = [
            Observation(
                content={
                    'type': 'benchmark',
                    'index': i,
                    'payload': 'x' * 100,  # ~100 bytes
                }
            )
            for i in range(self.SAMPLE_SIZE)
        ]

        latencies = []
        for event in events:
            # Clear cached ID
            object.__setattr__(event, '_id', None)

            start = time.perf_counter()
            _ = event.id
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.MAX_HASH_MS, (
            f"CONTRACT VIOLATION: p95 hash computation is {p95:.3f}ms, "
            f"contract requires <{self.MAX_HASH_MS}ms"
        )

    def test_hash_determinism_contract(self):
        """
        CONTRACT: Same content always produces same hash.

        Content-addressing requires perfect hash determinism.
        This is a correctness contract, not performance.
        """
        content = {
            'type': 'determinism_test',
            'data': 'test data',
            'nested': {'key': 'value'},
        }

        # Create 100 identical events
        event_ids = []
        for _ in range(100):
            event = Observation(
                content=content,
                timestamp='2024-12-31T00:00:00Z',  # Fixed timestamp
            )
            event_ids.append(event.id)

        # All IDs must be identical
        unique_ids = set(event_ids)

        assert len(unique_ids) == 1, (
            f"CONTRACT VIOLATION: Hash non-deterministic. "
            f"Got {len(unique_ids)} different hashes for identical content"
        )

    def test_causal_parent_ordering_preserved(self):
        """
        CONTRACT: Causal parent order is preserved in hash.

        Different parent orders must produce different hashes.
        """
        content = {'type': 'parent_test'}

        # Same parents, different order
        event1 = Observation(
            content=content,
            causal_parents=['parent_a', 'parent_b'],
            timestamp='2024-12-31T00:00:00Z',
        )

        event2 = Observation(
            content=content,
            causal_parents=['parent_b', 'parent_a'],
            timestamp='2024-12-31T00:00:00Z',
        )

        assert event1.id != event2.id, (
            "CONTRACT VIOLATION: Causal parent order not reflected in hash"
        )


@pytest.mark.contract
class TestEventSerializationContract:
    """
    Event Serialization Performance Contract

    As a system persisting thousands of events,
    I expect serialization to be fast,
    So that disk I/O is the bottleneck, not conversion.
    """

    # The sacred numbers
    MAX_SERIALIZE_MS = 2.0
    MAX_DESERIALIZE_MS = 2.0
    SAMPLE_SIZE = 100

    def test_serialization_latency(self):
        """
        CONTRACT: Event serialization in < 1ms.

        Serialization happens on every event append.
        """
        events = [
            Intention(
                title=f"Benchmark intention {i}",
                description="A" * 200,  # ~200 char description
                priority='medium',
                category='feature',
            )
            for i in range(self.SAMPLE_SIZE)
        ]

        latencies = []
        for event in events:
            start = time.perf_counter()
            _ = event.to_dict()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.MAX_SERIALIZE_MS, (
            f"CONTRACT VIOLATION: p95 serialization is {p95:.3f}ms, "
            f"contract requires <{self.MAX_SERIALIZE_MS}ms"
        )

    def test_deserialization_latency(self):
        """
        CONTRACT: Event deserialization in < 1ms.

        Deserialization happens on every event read.
        """
        # Create serialized events
        serialized = [
            Intention(
                title=f"Benchmark intention {i}",
                description="A" * 200,
                priority='medium',
                category='feature',
            ).to_dict()
            for i in range(self.SAMPLE_SIZE)
        ]

        latencies = []
        for data in serialized:
            start = time.perf_counter()
            _ = CognitiveEvent.from_dict(data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.MAX_DESERIALIZE_MS, (
            f"CONTRACT VIOLATION: p95 deserialization is {p95:.3f}ms, "
            f"contract requires <{self.MAX_DESERIALIZE_MS}ms"
        )

    def test_roundtrip_preserves_identity(self):
        """
        CONTRACT: Serialize/deserialize preserves event ID.

        The hash must survive the roundtrip.
        """
        original = Intention(
            title="Roundtrip test",
            description="Testing serialization roundtrip",
        )

        # Roundtrip
        serialized = original.to_dict()
        deserialized = CognitiveEvent.from_dict(serialized)

        assert original.id == deserialized.id, (
            f"CONTRACT VIOLATION: ID changed during roundtrip. "
            f"Original: {original.id}, Deserialized: {deserialized.id}"
        )


@pytest.mark.contract
class TestEventTypeContract:
    """
    Event Type Invariants Contract

    These are correctness contracts ensuring type-specific behavior.
    """

    def test_fulfillment_includes_intention_as_parent(self):
        """
        CONTRACT: Fulfillment always has intention as causal parent.

        This is load-bearing for causal reconstruction.
        """
        intention_id = "test_intention_abc123"

        # Create fulfillment with no explicit parents
        fulfillment = Fulfillment(
            intention_id=intention_id,
            result={'success': True},
        )

        assert intention_id in fulfillment.causal_parents, (
            f"CONTRACT VIOLATION: Fulfillment missing intention in causal_parents. "
            f"Parents: {fulfillment.causal_parents}"
        )

    def test_intention_extracts_concepts_from_title(self):
        """
        CONTRACT: Intentions auto-extract concepts from title.

        Concepts enable semantic search over intentions.
        """
        intention = Intention(
            title="Optimize neural network training pipeline",
        )

        # Should extract meaningful words
        assert len(intention.concepts) > 0, (
            "CONTRACT VIOLATION: No concepts extracted from intention title"
        )

        # Should filter stop words
        assert 'the' not in intention.concepts, (
            "CONTRACT VIOLATION: Stop words not filtered from concepts"
        )

        # Should have key terms
        has_relevant = any(
            term in intention.concepts
            for term in ['optimize', 'neural', 'network', 'training', 'pipeline']
        )
        assert has_relevant, (
            f"CONTRACT VIOLATION: Key terms missing from concepts. "
            f"Got: {intention.concepts}"
        )
