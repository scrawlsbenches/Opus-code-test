"""
╔══════════════════════════════════════════════════════════════════════╗
║               ROUTING & COORDINATION PERFORMANCE CONTRACT             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Attention routing decision < 50ms                                 ║
║  • Mode selection (FAST/SLOW) < 20ms                                 ║
║  • Context pool publish operation < 10ms                             ║
║  • Context pool query operation < 15ms                               ║
║  • Conflict detection completes < 50ms for 1000 findings             ║
║  • TTL-based pruning < 100ms for 10000 findings                      ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List

import pytest

from cortical.reasoning.attention_router import AttentionRouter, AttentionRouterConfig
from cortical.reasoning.loom import Loom, ThinkingMode
from cortical.reasoning.loom_hive import LoomHiveConnector
from cortical.reasoning.loom_cortex import LoomCortexConnector
from cortical.reasoning.context_pool import ContextPool, ConflictResolutionStrategy


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestAttentionRoutingContract:
    """
    Attention Routing Performance Contract

    As a cognitive system processing inputs,
    I expect routing decisions to be instantaneous,
    So that mode switching never introduces perceptible latency.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    ROUTING_DECISION_MS = 50
    MODE_SELECTION_MS = 20
    SAMPLE_SIZE = 50

    def test_routing_latency_honored(self):
        """
        CONTRACT: Routing decisions complete in under 50ms.

        Fast mode selection is critical for real-time processing.
        """
        # Setup
        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()
        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Train the hive with some data
        hive.train("neural networks process data efficiently")
        hive.train("machine learning algorithms optimize performance")

        # Measure routing latency
        contexts = [
            ["neural", "networks"],
            ["machine", "learning"],
            ["data", "processing"],
            ["algorithm", "optimization"],
        ]

        latencies = []
        for _ in range(self.SAMPLE_SIZE):
            context = contexts[_ % len(contexts)]
            start = time.perf_counter()
            router.route(context, mode=ThinkingMode.FAST)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.ROUTING_DECISION_MS, (
            f"CONTRACT VIOLATION: p95 routing latency is {p95:.1f}ms, "
            f"contract requires <{self.ROUTING_DECISION_MS}ms"
        )

    def test_mode_selection_latency_honored(self):
        """
        CONTRACT: Auto mode selection completes in under 20ms.

        Surprise detection and mode switching must be near-instant.
        """
        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()
        router = AttentionRouter(
            loom=loom, hive=hive, cortex=cortex,
            config=AttentionRouterConfig(auto_switch=True)
        )

        # Train
        hive.train("neural networks and deep learning systems")

        # Measure auto mode selection
        latencies = []
        for _ in range(self.SAMPLE_SIZE):
            start = time.perf_counter()
            # Auto mode selection (mode=None)
            router.route(["neural", "networks"], mode=None)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p50 = percentile(latencies, 50)

        assert p50 < self.MODE_SELECTION_MS, (
            f"CONTRACT VIOLATION: p50 mode selection is {p50:.1f}ms, "
            f"contract requires <{self.MODE_SELECTION_MS}ms"
        )

    def test_dual_routing_completes_quickly(self):
        """
        CONTRACT: Dual routing (both FAST+SLOW) completes in reasonable time.

        Even when processing through both systems, latency must be bounded.
        """
        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()
        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Train both systems
        text = "neural network machine learning algorithm"
        hive.train(text)
        for _ in range(5):
            cortex.process_slow(["neural", "network"])

        # Measure dual routing
        start = time.perf_counter()
        result = router.route_both(["neural", "network"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Dual routing should be < 150ms (generous budget for both systems)
        assert elapsed_ms < 150, (
            f"CONTRACT VIOLATION: Dual routing took {elapsed_ms:.1f}ms, "
            f"contract requires <150ms"
        )

        # Verify it actually produced results
        assert len(result.fast_result) > 0 or len(result.slow_result) > 0


@pytest.mark.contract
class TestContextPoolContract:
    """
    Context Pool Performance Contract

    As a multi-agent system sharing findings,
    I expect publish/query operations to be fast,
    So that coordination overhead never becomes a bottleneck.
    """

    # The sacred numbers
    PUBLISH_LATENCY_MS = 10
    QUERY_LATENCY_MS = 15
    CONFLICT_DETECTION_MS = 50
    TTL_PRUNING_MS = 100

    def test_publish_latency_honored(self):
        """
        CONTRACT: Publishing findings completes in under 10ms.

        Fast publishing enables high-throughput multi-agent coordination.
        """
        pool = ContextPool()

        latencies = []
        for i in range(100):
            start = time.perf_counter()
            pool.publish(
                topic="test_finding",
                content=f"Finding {i} from custom analysis engine",
                source_agent=f"agent_{i % 5}",
                confidence=0.8
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.PUBLISH_LATENCY_MS, (
            f"CONTRACT VIOLATION: p95 publish latency is {p95:.1f}ms, "
            f"contract requires <{self.PUBLISH_LATENCY_MS}ms"
        )

    def test_query_latency_honored(self):
        """
        CONTRACT: Querying findings completes in under 15ms.

        Fast queries enable agents to quickly discover shared knowledge.
        """
        pool = ContextPool()

        # Pre-populate with findings
        for i in range(500):
            pool.publish(
                topic=f"topic_{i % 10}",
                content=f"Finding {i}",
                source_agent=f"agent_{i % 5}",
                confidence=0.7
            )

        # Measure query latency
        latencies = []
        for i in range(50):
            topic = f"topic_{i % 10}"
            start = time.perf_counter()
            results = pool.query(topic)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.QUERY_LATENCY_MS, (
            f"CONTRACT VIOLATION: p95 query latency is {p95:.1f}ms, "
            f"contract requires <{self.QUERY_LATENCY_MS}ms"
        )

    def test_conflict_detection_bounded(self):
        """
        CONTRACT: Conflict detection completes in under 50ms for 1000 findings.

        Conflict detection must scale to handle real multi-agent scenarios.
        """
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

        # Publish findings that will conflict
        for i in range(1000):
            pool.publish(
                topic="architecture_decision",
                content=f"Use custom approach {i % 3} for data storage",
                source_agent=f"agent_{i % 10}",
                confidence=0.8
            )

        # Measure conflict detection
        start = time.perf_counter()
        conflicts = pool.get_conflicts()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.CONFLICT_DETECTION_MS, (
            f"CONTRACT VIOLATION: Conflict detection took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.CONFLICT_DETECTION_MS}ms"
        )

    def test_ttl_pruning_scales(self):
        """
        CONTRACT: TTL-based pruning completes in under 100ms for 10000 findings.

        Pruning expired findings must not block the pool.
        """
        pool = ContextPool(ttl_seconds=1.0)

        # Publish many findings
        for i in range(10000):
            pool.publish(
                topic=f"topic_{i % 100}",
                content=f"Ephemeral finding {i}",
                source_agent=f"agent_{i % 50}",
                confidence=0.5
            )

        # Wait for expiration
        time.sleep(1.1)

        # Measure pruning via query (triggers _prune_expired)
        start = time.perf_counter()
        pool.query_all()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.TTL_PRUNING_MS, (
            f"CONTRACT VIOLATION: TTL pruning took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.TTL_PRUNING_MS}ms"
        )

    def test_count_operation_is_fast(self):
        """
        CONTRACT: Counting findings is O(1) or very fast.

        Count operations should not iterate through all findings.
        """
        pool = ContextPool()

        # Add many findings
        for i in range(5000):
            pool.publish(
                topic=f"topic_{i % 20}",
                content=f"Finding {i}",
                source_agent="agent_test",
                confidence=0.7
            )

        # Measure count operation
        start = time.perf_counter()
        total = pool.count()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Count should be near-instant
        assert elapsed_ms < 5, (
            f"CONTRACT VIOLATION: Count took {elapsed_ms:.1f}ms, "
            f"should be near-instant (< 5ms)"
        )
        assert total == 5000, "Count returned incorrect value"
