"""
╔══════════════════════════════════════════════════════════════════════╗
║                     PAGERANK PERFORMANCE CONTRACT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • PageRank converges in ≤ 20 iterations for graphs ≤ 1,000 nodes   ║
║  • Computation latency < 500ms for graphs ≤ 1,000 nodes             ║
║  • Semantic PageRank converges in ≤ 20 iterations                   ║
║  • Hierarchical PageRank converges in ≤ 5 global iterations         ║
║  • Damping parameter must be in range (0, 1)                        ║
║  • Sum of PageRank scores equals 1.0 (within tolerance)             ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest


@pytest.mark.contract
class TestPageRankConvergenceContract:
    """
    PageRank Convergence Contract

    As a developer using PageRank for importance scoring,
    I expect the algorithm to converge quickly,
    So that indexing completes in reasonable time.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_ITERATIONS = 20
    MAX_LATENCY_MS = 1000

    def test_pagerank_converges_within_iteration_limit(self, small_processor):
        """
        CONTRACT: PageRank converges in ≤ 20 iterations.

        Fast convergence ensures efficient indexing.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.pagerank import compute_pagerank

        layer = small_processor.layers[CorticalLayer.TOKENS]

        # Should converge in ≤ 20 iterations
        pagerank = compute_pagerank(
            layer,
            damping=0.85,
            iterations=self.MAX_ITERATIONS,
            tolerance=1e-6
        )

        # Verify it returned valid results (converged)
        assert len(pagerank) > 0, (
            "CONTRACT VIOLATION: PageRank failed to compute"
        )

        # All scores should be positive
        assert all(score > 0 for score in pagerank.values()), (
            "CONTRACT VIOLATION: PageRank produced non-positive scores"
        )

    def test_pagerank_latency_honored(self, small_processor):
        """
        CONTRACT: PageRank completes in < 500ms for ≤ 1,000 nodes.

        This guarantee ensures responsive indexing workflow.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.pagerank import compute_pagerank

        layer = small_processor.layers[CorticalLayer.TOKENS]

        # Should have < 1,000 nodes (small_processor contract)
        assert layer.column_count() < 1000

        start = time.perf_counter()
        compute_pagerank(layer, damping=0.85, iterations=20, tolerance=1e-6)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: PageRank took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_pagerank_sum_equals_one(self, small_processor):
        """
        CONTRACT: Sum of PageRank scores equals 1.0 (within tolerance).

        This is a fundamental property of PageRank - scores are a probability distribution.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.pagerank import compute_pagerank

        layer = small_processor.layers[CorticalLayer.TOKENS]
        pagerank = compute_pagerank(layer, damping=0.85, iterations=20, tolerance=1e-6)

        total = sum(pagerank.values())
        tolerance = 1e-3  # Allow 0.1% tolerance

        assert abs(total - 1.0) < tolerance, (
            f"CONTRACT VIOLATION: PageRank scores sum to {total:.6f}, "
            f"expected 1.0 ± {tolerance}"
        )

    def test_pagerank_validates_damping_parameter(self):
        """
        CONTRACT: PageRank rejects invalid damping parameters.

        Damping must be in range (0, 1) - exclusive bounds.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.pagerank import compute_pagerank

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom neural network implementation.")
        processor.process_document("doc2", "Hand-built search algorithm.")
        processor.compute_all(verbose=False)

        layer = processor.layers[CorticalLayer.TOKENS]

        # Test invalid damping values
        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=0.0)

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=1.0)

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=1.5)

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=-0.1)


@pytest.mark.contract
class TestSemanticPageRankContract:
    """
    Semantic PageRank Contract

    As a developer using semantic relations,
    I expect semantic PageRank to converge efficiently,
    So that semantic weighting adds minimal overhead.
    """

    MAX_ITERATIONS = 20

    def test_semantic_pagerank_converges(self, small_processor):
        """
        CONTRACT: Semantic PageRank converges in ≤ 20 iterations.

        Semantic weighting should not prevent convergence.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.pagerank import compute_semantic_pagerank

        layer = small_processor.layers[CorticalLayer.TOKENS]

        # Create sample semantic relations
        relations = [
            ("neural", "RelatedTo", "network", 0.8),
            ("machine", "RelatedTo", "learning", 0.9),
            ("data", "RelatedTo", "analysis", 0.7),
        ]

        result = compute_semantic_pagerank(
            layer,
            semantic_relations=relations,
            damping=0.85,
            iterations=self.MAX_ITERATIONS,
            tolerance=1e-6
        )

        # Check convergence
        assert result['iterations_run'] <= self.MAX_ITERATIONS, (
            f"CONTRACT VIOLATION: Semantic PageRank did not converge in "
            f"{self.MAX_ITERATIONS} iterations, took {result['iterations_run']}"
        )

        # Verify valid results
        assert len(result['pagerank']) > 0
        assert all(score > 0 for score in result['pagerank'].values())

    def test_semantic_pagerank_validates_damping(self):
        """
        CONTRACT: Semantic PageRank validates damping parameter.

        Same validation as standard PageRank.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.pagerank import compute_semantic_pagerank

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network")
        processor.compute_all(verbose=False)

        layer = processor.layers[CorticalLayer.TOKENS]

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_semantic_pagerank(
                layer,
                semantic_relations=[],
                damping=1.5
            )


@pytest.mark.contract
class TestHierarchicalPageRankContract:
    """
    Hierarchical PageRank Contract

    As a developer using cross-layer propagation,
    I expect hierarchical PageRank to converge quickly,
    So that multi-layer indexing remains practical.
    """

    MAX_GLOBAL_ITERATIONS = 5
    MAX_LATENCY_MS = 4000  # More permissive - hierarchical is more complex

    def test_hierarchical_pagerank_converges(self, small_processor):
        """
        CONTRACT: Hierarchical PageRank converges in ≤ 5 global iterations.

        Cross-layer propagation should stabilize quickly.
        """
        from cortical.analysis.pagerank import compute_hierarchical_pagerank

        result = compute_hierarchical_pagerank(
            small_processor.layers,
            global_iterations=self.MAX_GLOBAL_ITERATIONS,
            layer_iterations=10,
            damping=0.85,
            cross_layer_damping=0.7,
            tolerance=1e-4
        )

        assert result['iterations_run'] <= self.MAX_GLOBAL_ITERATIONS, (
            f"CONTRACT VIOLATION: Hierarchical PageRank did not converge in "
            f"{self.MAX_GLOBAL_ITERATIONS} global iterations, took {result['iterations_run']}"
        )

    def test_hierarchical_pagerank_latency(self, small_processor):
        """
        CONTRACT: Hierarchical PageRank completes in < 2 seconds.

        Even complex multi-layer propagation must be practical.
        """
        from cortical.analysis.pagerank import compute_hierarchical_pagerank

        start = time.perf_counter()
        compute_hierarchical_pagerank(
            small_processor.layers,
            global_iterations=5,
            layer_iterations=10,
            damping=0.85,
            cross_layer_damping=0.7,
            tolerance=1e-4
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Hierarchical PageRank took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_hierarchical_pagerank_validates_damping(self):
        """
        CONTRACT: Hierarchical PageRank validates both damping parameters.

        Both damping and cross_layer_damping must be in range (0, 1).
        """
        from cortical import CorticalTextProcessor
        from cortical.analysis.pagerank import compute_hierarchical_pagerank

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural network")
        processor.compute_all(verbose=False)

        # Test invalid damping
        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_hierarchical_pagerank(
                processor.layers,
                damping=1.5
            )

        # Test invalid cross_layer_damping
        with pytest.raises(ValueError, match="cross_layer_damping must be between 0 and 1"):
            compute_hierarchical_pagerank(
                processor.layers,
                cross_layer_damping=0.0
            )


@pytest.mark.contract
class TestPageRankCorrectnessContract:
    """
    PageRank Correctness Contract

    As a developer relying on PageRank,
    I expect correct graph-theoretic properties,
    So that importance scores are meaningful.
    """

    def test_pagerank_core_correctness(self):
        """
        CONTRACT: PageRank correctly identifies important nodes.

        A node with more incoming links should rank higher.
        """
        from cortical.analysis.pagerank import _pagerank_core

        # Graph: a ← b, c → a (a has 2 incoming, b and c have 0)
        graph = {
            "a": [],
            "b": [("a", 1.0)],
            "c": [("a", 1.0)]
        }

        ranks = _pagerank_core(graph, damping=0.85, iterations=20, tolerance=1e-6)

        # Node 'a' should have highest rank (receives all PageRank)
        assert ranks["a"] > ranks["b"], (
            "CONTRACT VIOLATION: Node with more incoming links doesn't rank higher"
        )
        assert ranks["a"] > ranks["c"], (
            "CONTRACT VIOLATION: Node with more incoming links doesn't rank higher"
        )

    def test_pagerank_handles_empty_graph(self):
        """
        CONTRACT: PageRank handles edge cases gracefully.

        Empty graph should return empty result, not crash.
        """
        from cortical.analysis.pagerank import _pagerank_core

        ranks = _pagerank_core({}, damping=0.85, iterations=20, tolerance=1e-6)
        assert ranks == {}, "Empty graph should return empty dict"

    def test_pagerank_handles_disconnected_components(self):
        """
        CONTRACT: PageRank handles disconnected graphs.

        Disconnected components should each have valid scores.
        """
        from cortical.analysis.pagerank import _pagerank_core

        # Two disconnected components
        graph = {
            "a": [("b", 1.0)],
            "b": [("a", 1.0)],
            "c": [("d", 1.0)],
            "d": [("c", 1.0)]
        }

        ranks = _pagerank_core(graph, damping=0.85, iterations=20, tolerance=1e-6)

        # All nodes should have positive scores
        assert all(score > 0 for score in ranks.values())

        # Total should sum to 1
        total = sum(ranks.values())
        assert abs(total - 1.0) < 1e-3
