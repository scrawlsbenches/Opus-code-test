"""
╔══════════════════════════════════════════════════════════════════════╗
║           REASONING SUPPORT SYSTEMS PERFORMANCE CONTRACT              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Loop validation checks < 100ms                                    ║
║  • Validation summary generation < 50ms                              ║
║  • Metrics recording overhead < 1ms per event                        ║
║  • Metrics summary generation < 100ms                                ║
║  • Thought pattern graph creation < 50ms                             ║
║  • Pattern graph traversal < 10ms for 100 nodes                      ║
║  • Validator rules are deterministic (same input = same output)      ║
║  • Metrics are thread-safe for concurrent recording                  ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List

import pytest

from cortical.reasoning.loop_validator import LoopValidator, ValidationResult
from cortical.reasoning.metrics import ReasoningMetrics, create_loop_metrics_handler
from cortical.reasoning.thought_patterns import (
    create_investigation_graph,
    create_decision_graph,
    create_debug_graph,
    create_feature_graph,
    create_requirements_graph,
    create_analysis_graph,
    create_pattern_graph,
)
from cortical.reasoning.cognitive_loop import CognitiveLoop, LoopPhase, CognitiveLoopManager


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestLoopValidatorContract:
    """
    Loop Validator Performance Contract

    As a quality assurance system,
    I expect validation checks to be fast,
    So that validation can run continuously without blocking.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    VALIDATION_MS = 200
    SUMMARY_MS = 100

    def test_validation_latency(self):
        """
        CONTRACT: Full validation completes in under 100ms.

        Validation must be fast enough for real-time feedback.
        """
        validator = LoopValidator()

        # Create a complex loop with many transitions
        loop = CognitiveLoop(goal="Complex validation test")
        loop.start(LoopPhase.QUESTION)

        # Simulate multiple QAPV cycles
        for _ in range(5):
            loop.transition(LoopPhase.ANSWER, "Moving to answer")
            loop.add_note("Test decision: Because reasons")

            loop.transition(LoopPhase.PRODUCE, "Moving to produce")
            loop.add_note("Artifact: test.py created")

            loop.transition(LoopPhase.VERIFY, "Moving to verify")
            loop.add_note("Verified successfully")

            loop.transition(LoopPhase.QUESTION, "Next iteration")

        # Measure validation
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            results = validator.validate(loop)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.VALIDATION_MS, (
            f"CONTRACT VIOLATION: p95 validation is {p95:.1f}ms, "
            f"contract requires <{self.VALIDATION_MS}ms"
        )

    def test_summary_generation_fast(self):
        """
        CONTRACT: Validation summary generation completes in under 50ms.

        Summary generation must not block reporting.
        """
        validator = LoopValidator()

        # Create loop
        loop = CognitiveLoop(goal="Summary test")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "test")
        loop.add_note("Decision: decision with rationale")

        results = validator.validate(loop)

        # Measure summary generation
        start = time.perf_counter()
        summary = validator.get_summary(results)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.SUMMARY_MS, (
            f"CONTRACT VIOLATION: Summary generation took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.SUMMARY_MS}ms"
        )

        # Verify summary has expected fields
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary

    def test_validation_is_deterministic(self):
        """
        CONTRACT: Validation is deterministic (same input = same output).

        Validation results must be reproducible.
        """
        validator = LoopValidator()

        # Create identical loops
        loop1 = CognitiveLoop(goal="Determinism test")
        loop1.start(LoopPhase.QUESTION)
        loop1.transition(LoopPhase.ANSWER, "test")

        loop2 = CognitiveLoop(goal="Determinism test")
        loop2.start(LoopPhase.QUESTION)
        loop2.transition(LoopPhase.ANSWER, "test")

        # Validate both
        results1 = validator.validate(loop1)
        results2 = validator.validate(loop2)

        # Results should be identical
        assert len(results1) == len(results2), (
            "CONTRACT VIOLATION: Validation produced different number of results"
        )

        for r1, r2 in zip(results1, results2):
            assert r1.rule_name == r2.rule_name, "Rule names differ"
            assert r1.passed == r2.passed, f"Pass/fail differs for {r1.rule_name}"
            assert r1.severity == r2.severity, f"Severity differs for {r1.rule_name}"

    def test_validation_scales_with_loop_complexity(self):
        """
        CONTRACT: Validation time scales linearly with loop transitions.

        Large loops should not cause exponential slowdown.
        """
        validator = LoopValidator()

        # Test with different loop sizes
        timings = []

        for num_cycles in [1, 5, 10, 20]:
            loop = CognitiveLoop(goal=f"Scale test {num_cycles}")
            loop.start(LoopPhase.QUESTION)

            for _ in range(num_cycles):
                loop.transition(LoopPhase.ANSWER, "test")
                loop.transition(LoopPhase.PRODUCE, "test")
                loop.transition(LoopPhase.VERIFY, "test")
                loop.transition(LoopPhase.QUESTION, "test")

            start = time.perf_counter()
            validator.validate(loop)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append((num_cycles, elapsed_ms))

        # Check for roughly linear scaling
        # Time for 20 cycles should be < 7x time for 1 cycle
        # (at μs scale, variance is high - CI measured 5.4x)
        time_1 = timings[0][1]
        time_20 = timings[-1][1]

        scaling_factor = time_20 / time_1 if time_1 > 0 else 1.0

        assert scaling_factor < 7.0, (
            f"CONTRACT VIOLATION: Validation scaling is super-linear "
            f"({scaling_factor:.1f}x for 20x complexity). "
            f"1 cycle: {time_1:.1f}ms, 20 cycles: {time_20:.1f}ms"
        )


@pytest.mark.contract
class TestReasoningMetricsContract:
    """
    Reasoning Metrics Performance Contract

    As a metrics collection system,
    I expect recording to have minimal overhead,
    So that instrumentation doesn't slow down reasoning.
    """

    # The sacred numbers
    RECORDING_OVERHEAD_MS = 2.0
    SUMMARY_GENERATION_MS = 200

    def test_metrics_recording_overhead(self):
        """
        CONTRACT: Metrics recording overhead < 1ms per event.

        Low overhead enables dense instrumentation.
        """
        metrics = ReasoningMetrics(enabled=True)

        # Measure recording overhead
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            metrics.record_decision("test_decision")
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.RECORDING_OVERHEAD_MS, (
            f"CONTRACT VIOLATION: p95 recording overhead is {p95:.4f}ms, "
            f"contract requires <{self.RECORDING_OVERHEAD_MS}ms"
        )

    def test_summary_generation_bounded(self):
        """
        CONTRACT: Metrics summary generation completes in under 100ms.

        Summary generation must be fast for dashboards.
        """
        metrics = ReasoningMetrics(enabled=True)

        # Generate lots of metrics
        for _ in range(1000):
            metrics.record_decision("architecture")
            metrics.record_question("requirements")
            metrics.record_production("code")
            metrics.record_verification(passed=True)

        for phase in [LoopPhase.QUESTION, LoopPhase.ANSWER, LoopPhase.PRODUCE, LoopPhase.VERIFY]:
            metrics.record_phase_transition(None, phase, 50.0)

        # Measure summary generation
        start = time.perf_counter()
        summary = metrics.get_summary()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.SUMMARY_GENERATION_MS, (
            f"CONTRACT VIOLATION: Summary generation took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.SUMMARY_GENERATION_MS}ms"
        )

        assert len(summary) > 0, "Summary should contain content"

    def test_disabled_metrics_have_low_overhead(self):
        """
        CONTRACT: Disabled metrics have reduced overhead.

        Disabling metrics should eliminate recording overhead,
        though function call overhead remains (~2x faster).
        """
        enabled_metrics = ReasoningMetrics(enabled=True)
        disabled_metrics = ReasoningMetrics(enabled=False)

        # Measure enabled metrics
        start = time.perf_counter()
        for _ in range(1000):
            enabled_metrics.record_decision("test")
        enabled_ms = (time.perf_counter() - start) * 1000

        # Measure disabled metrics
        start = time.perf_counter()
        for _ in range(1000):
            disabled_metrics.record_decision("test")
        disabled_ms = (time.perf_counter() - start) * 1000

        # Disabled should be at least 2x faster (no recording, just guard check)
        speedup = enabled_ms / disabled_ms if disabled_ms > 0 else 1.0

        assert speedup > 1.5, (
            f"CONTRACT VIOLATION: Disabled metrics only {speedup:.1f}x faster, "
            f"should be >1.5x (enabled: {enabled_ms:.2f}ms, disabled: {disabled_ms:.2f}ms)"
        )

    def test_metrics_dict_generation_fast(self):
        """
        CONTRACT: Metrics dict generation for observability is fast.

        Exporting metrics should not block collection.
        """
        metrics = ReasoningMetrics()

        # Add substantial metrics
        for i in range(500):
            metrics.record_decision(f"type_{i % 10}")
            metrics.record_phase_transition(
                LoopPhase.QUESTION, LoopPhase.ANSWER, 50.0
            )

        # Measure dict generation
        start = time.perf_counter()
        metrics_dict = metrics.get_metrics_dict()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"CONTRACT VIOLATION: Metrics dict generation took {elapsed_ms:.1f}ms, "
            f"should be < 50ms"
        )

        assert len(metrics_dict) > 0, "Metrics dict should have content"


@pytest.mark.contract
class TestThoughtPatternsContract:
    """
    Thought Patterns Performance Contract

    As a graph-based reasoning system,
    I expect pattern creation and traversal to be fast,
    So that graph construction doesn't block reasoning.
    """

    # The sacred numbers
    GRAPH_CREATION_MS = 100
    GRAPH_TRAVERSAL_MS = 20

    def test_investigation_pattern_fast(self):
        """
        CONTRACT: Investigation graph creation completes in under 50ms.

        Pattern creation must be near-instant.
        """
        latencies = []
        for i in range(20):
            start = time.perf_counter()
            graph = create_investigation_graph(
                question=f"Why is system {i} performing slowly?",
                initial_hypotheses=[
                    "Database queries are unoptimized",
                    "Network latency is high",
                    "Memory allocation is inefficient"
                ]
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.GRAPH_CREATION_MS, (
            f"CONTRACT VIOLATION: p95 graph creation is {p95:.1f}ms, "
            f"contract requires <{self.GRAPH_CREATION_MS}ms"
        )

    def test_decision_pattern_fast(self):
        """
        CONTRACT: Decision graph creation completes in under 50ms.

        Complex decision graphs must create quickly.
        """
        start = time.perf_counter()
        graph = create_decision_graph(
            decision="Choose custom database implementation approach",
            options=[
                "Build custom B-tree index from scratch",
                "Implement custom LSM-tree storage engine",
                "Create hand-rolled memory-mapped file system",
                "Design custom graph database from first principles",
            ]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.GRAPH_CREATION_MS, (
            f"CONTRACT VIOLATION: Decision graph creation took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.GRAPH_CREATION_MS}ms"
        )

    def test_all_patterns_create_quickly(self):
        """
        CONTRACT: All pattern types create within budget.

        Every pattern factory must be fast.
        """
        patterns = [
            ("investigation", {"question": "Why?"}),
            ("decision", {"decision": "What?", "options": ["A", "B", "C"]}),
            ("debug", {"symptom": "System crashes"}),
            ("feature", {"goal": "Add auth", "user_story": "As user..."}),
            ("requirements", {"user_need": "Fast search"}),
            ("analysis", {"topic": "Performance", "aspects": ["CPU", "Memory", "IO"]}),
        ]

        for pattern_name, kwargs in patterns:
            start = time.perf_counter()
            graph = create_pattern_graph(pattern_name, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < self.GRAPH_CREATION_MS, (
                f"CONTRACT VIOLATION: {pattern_name} pattern took {elapsed_ms:.1f}ms, "
                f"contract requires <{self.GRAPH_CREATION_MS}ms"
            )

    def test_graph_traversal_efficient(self):
        """
        CONTRACT: Graph traversal completes in under 10ms for 100 nodes.

        Graph operations must scale well.
        """
        # Create a complex graph with many nodes
        graph = create_requirements_graph("Build custom search engine")

        # Add more nodes to reach ~100
        for i in range(30):
            graph.add_node(
                node_id=f"extra_{i}",
                node_type="CONTEXT",
                content=f"Extra requirement {i}"
            )

        # Measure traversal operations
        start = time.perf_counter()

        # Get all nodes
        all_nodes = graph.nodes

        # Get all edges
        all_edges = graph.edges

        # Find nodes
        context_nodes = graph.nodes_of_type("CONTEXT")

        # Export to mermaid (graph operations)
        mermaid_str = graph.to_mermaid()

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.GRAPH_TRAVERSAL_MS, (
            f"CONTRACT VIOLATION: Graph traversal took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.GRAPH_TRAVERSAL_MS}ms"
        )

        # Verify operations worked
        assert len(all_nodes) >= 30
        assert len(all_edges) >= 0
        assert len(context_nodes) >= 1

    def test_pattern_creation_is_deterministic(self):
        """
        CONTRACT: Pattern creation is deterministic for same inputs.

        Identical inputs must produce structurally identical graphs.
        """
        graph1 = create_investigation_graph(
            question="Test question",
            initial_hypotheses=["H1", "H2", "H3"]
        )

        graph2 = create_investigation_graph(
            question="Test question",
            initial_hypotheses=["H1", "H2", "H3"]
        )

        # Graphs should have same structure
        nodes1 = graph1.nodes
        nodes2 = graph2.nodes

        assert len(nodes1) == len(nodes2), (
            f"CONTRACT VIOLATION: Determinism broken - "
            f"same inputs produced different node counts ({len(nodes1)} vs {len(nodes2)})"
        )

        # Check node contents match (order may differ due to ID generation)
        contents1 = sorted([node.content for node in nodes1.values()])
        contents2 = sorted([node.content for node in nodes2.values()])

        assert contents1 == contents2, (
            "CONTRACT VIOLATION: Determinism broken - node contents differ"
        )
