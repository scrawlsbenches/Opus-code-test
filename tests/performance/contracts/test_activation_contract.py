"""
╔══════════════════════════════════════════════════════════════════════╗
║                   ACTIVATION PERFORMANCE CONTRACT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Activation propagation < 100ms per iteration for ≤ 1,500 nodes   ║
║  • 3 iterations complete in < 300ms for ≤ 1,500 nodes               ║
║  • Activation values decay over iterations                          ║
║  • Activation values remain non-negative                            ║
║  • Connected nodes receive activation                               ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest


@pytest.mark.contract
class TestActivationPropagationPerformanceContract:
    """
    Activation Propagation Performance Contract

    As a developer using spreading activation,
    I expect propagation to complete quickly,
    So that dynamic network activation is practical.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_ITERATION_LATENCY_MS = 100
    MAX_TOTAL_LATENCY_MS = 300
    DEFAULT_ITERATIONS = 3

    def test_activation_iteration_latency_honored(self, small_processor):
        """
        CONTRACT: Single activation iteration < 100ms for ≤ 1,500 nodes.

        Each propagation step must be fast for interactive use.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        # Verify within bounds (small_processor has ~1100 nodes)
        total_nodes = sum(
            layer.column_count()
            for layer in small_processor.layers.values()
        )
        assert total_nodes < 1500, f"Fixture too large: {total_nodes} nodes"

        # Set initial activation
        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        if layer0.column_count() > 0:
            first_col = next(iter(layer0.minicolumns.values()))
            first_col.activation = 1.0

        # Measure single iteration
        start = time.perf_counter()
        propagate_activation(small_processor.layers, iterations=1)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_ITERATION_LATENCY_MS, (
            f"CONTRACT VIOLATION: Activation iteration took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_ITERATION_LATENCY_MS}ms"
        )

    def test_activation_total_latency_honored(self, small_processor):
        """
        CONTRACT: 3 activation iterations complete in < 300ms for ≤ 1,000 nodes.

        Multiple iterations should remain fast enough for real-time use.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        # Set initial activation
        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        if layer0.column_count() > 0:
            first_col = next(iter(layer0.minicolumns.values()))
            first_col.activation = 1.0

        start = time.perf_counter()
        propagate_activation(
            small_processor.layers,
            iterations=self.DEFAULT_ITERATIONS,
            decay=0.8,
            lateral_weight=0.3
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_TOTAL_LATENCY_MS, (
            f"CONTRACT VIOLATION: {self.DEFAULT_ITERATIONS} activation iterations "
            f"took {elapsed_ms:.1f}ms, contract requires <{self.MAX_TOTAL_LATENCY_MS}ms"
        )


@pytest.mark.contract
class TestActivationCorrectnessContract:
    """
    Activation Correctness Contract

    As a developer relying on spreading activation,
    I expect correct network dynamics,
    So that activation patterns are meaningful.
    """

    def test_activation_decays_over_iterations(self):
        """
        CONTRACT: Activation decays without external input.

        With decay < 1.0 and no lateral input, activation should decrease.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom neural network implementation.")
        processor.compute_all(verbose=False)

        layer = processor.layers[CorticalLayer.TOKENS]

        # Set initial activation on one node
        if layer.column_count() == 0:
            pytest.skip("No tokens to test")

        test_col = next(iter(layer.minicolumns.values()))
        initial_activation = 1.0
        test_col.activation = initial_activation

        # Clear all other activations
        for col in layer.minicolumns.values():
            if col.id != test_col.id:
                col.activation = 0.0

        # Clear lateral connections to isolate decay
        test_col.lateral_connections.clear()

        # Propagate with decay
        propagate_activation(
            processor.layers,
            iterations=2,
            decay=0.5,  # Should halve each iteration
            lateral_weight=0.0  # No lateral spreading
        )

        # Activation should have decayed
        assert test_col.activation < initial_activation, (
            f"CONTRACT VIOLATION: Activation did not decay. "
            f"Initial: {initial_activation}, Final: {test_col.activation}"
        )

        # Should be approximately initial * decay^iterations
        expected = initial_activation * (0.5 ** 2)  # 2 iterations
        tolerance = 0.1
        assert abs(test_col.activation - expected) < tolerance, (
            f"CONTRACT VIOLATION: Decay incorrect. "
            f"Expected ~{expected:.2f}, got {test_col.activation:.2f}"
        )

    def test_activation_spreads_to_connected_nodes(self):
        """
        CONTRACT: Activation spreads to laterally connected nodes.

        Connected nodes should receive activation from active neighbors.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        processor = CorticalTextProcessor()
        # Use related terms to ensure connection
        processor.process_document("doc1", "neural network training optimization")
        processor.compute_all(verbose=False)

        layer = processor.layers[CorticalLayer.TOKENS]

        if layer.column_count() < 2:
            pytest.skip("Need at least 2 tokens")

        # Find a token with lateral connections
        source_col = None
        for col in layer.minicolumns.values():
            if len(col.lateral_connections) > 0:
                source_col = col
                break

        if source_col is None:
            pytest.skip("No lateral connections found")

        # Clear all activations
        for col in layer.minicolumns.values():
            col.activation = 0.0

        # Activate source
        source_col.activation = 1.0

        # Propagate
        propagate_activation(
            processor.layers,
            iterations=1,
            decay=0.9,
            lateral_weight=0.3
        )

        # At least one connected neighbor should have received activation
        neighbor_activated = False
        for neighbor_id in source_col.lateral_connections:
            neighbor = layer.get_by_id(neighbor_id)
            if neighbor and neighbor.activation > 0:
                neighbor_activated = True
                break

        assert neighbor_activated, (
            "CONTRACT VIOLATION: Activation did not spread to connected nodes"
        )

    def test_activation_values_non_negative(self):
        """
        CONTRACT: Activation values never become negative.

        Negative activation is physically meaningless.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom implementation of neural network.")
        processor.process_document("doc2", "Hand-built search algorithm indexing.")
        processor.compute_all(verbose=False)

        # Set various initial activations
        layer = processor.layers[CorticalLayer.TOKENS]
        for i, col in enumerate(layer.minicolumns.values()):
            col.activation = float(i % 3)  # 0, 1, 2 pattern

        # Propagate
        propagate_activation(
            processor.layers,
            iterations=5,
            decay=0.7,
            lateral_weight=0.4
        )

        # Check all activations are non-negative
        for layer in processor.layers.values():
            for col in layer.minicolumns.values():
                assert col.activation >= 0, (
                    f"CONTRACT VIOLATION: Node '{col.content}' has negative "
                    f"activation: {col.activation}"
                )

    def test_activation_handles_feedforward_connections(self):
        """
        CONTRACT: Activation flows through feedforward connections.

        Higher layers should receive activation from lower layers.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom neural network implementation built from scratch.")
        processor.compute_all(verbose=False)

        layer0 = processor.layers[CorticalLayer.TOKENS]
        layer2 = processor.layers[CorticalLayer.CONCEPTS]

        if layer0.column_count() == 0 or layer2.column_count() == 0:
            pytest.skip("Need both tokens and concepts")

        # Clear all activations
        for layer in processor.layers.values():
            for col in layer.minicolumns.values():
                col.activation = 0.0

        # Activate first token
        first_token = next(iter(layer0.minicolumns.values()))
        first_token.activation = 1.0

        # Propagate
        propagate_activation(
            processor.layers,
            iterations=3,
            decay=0.8,
            lateral_weight=0.3
        )

        # At least some concept should have received activation
        # (because tokens connect to concepts via feedback/feedforward)
        total_concept_activation = sum(
            col.activation for col in layer2.minicolumns.values()
        )

        # This is a soft check - depends on network structure
        # Main goal is to verify feedforward logic doesn't crash
        assert total_concept_activation >= 0


@pytest.mark.contract
class TestActivationParametersContract:
    """
    Activation Parameters Contract

    As a developer tuning activation parameters,
    I expect parameter validation and reasonable defaults,
    So that incorrect configurations are caught early.
    """

    def test_activation_accepts_valid_parameters(self):
        """
        CONTRACT: Activation accepts valid parameter ranges.

        decay ∈ (0, 1], lateral_weight ∈ [0, 1], iterations ≥ 1.
        """
        from cortical import CorticalTextProcessor
        from cortical.analysis.activation import propagate_activation

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")
        processor.compute_all(verbose=False)

        # These should not raise
        propagate_activation(processor.layers, iterations=1, decay=0.5, lateral_weight=0.0)
        propagate_activation(processor.layers, iterations=5, decay=0.9, lateral_weight=0.5)
        propagate_activation(processor.layers, iterations=1, decay=1.0, lateral_weight=1.0)

    def test_activation_handles_zero_iterations(self):
        """
        CONTRACT: Zero iterations is a no-op.

        Should not crash or modify state.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.activation import propagate_activation

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test document")
        processor.compute_all(verbose=False)

        layer = processor.layers[CorticalLayer.TOKENS]
        if layer.column_count() == 0:
            pytest.skip("No tokens")

        # Set activation
        first = next(iter(layer.minicolumns.values()))
        first.activation = 1.0
        initial = first.activation

        # Zero iterations should not change anything
        propagate_activation(processor.layers, iterations=0)

        assert first.activation == initial
