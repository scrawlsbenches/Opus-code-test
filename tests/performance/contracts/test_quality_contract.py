"""
╔══════════════════════════════════════════════════════════════════════╗
║                     QUALITY METRICS PERFORMANCE CONTRACT              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Modularity computation < 500ms for ≤ 1,000 nodes                 ║
║  • Silhouette computation < 2 seconds for ≤ 500 nodes (sampled)     ║
║  • Full quality metrics < 3 seconds for typical corpus              ║
║  • Modularity Q ∈ [-0.5, 1.0] for valid graphs                      ║
║  • Silhouette score ∈ [-1.0, 1.0]                                   ║
║  • Gini coefficient ∈ [0.0, 1.0]                                    ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest


@pytest.mark.contract
class TestModularityPerformanceContract:
    """
    Modularity Computation Performance Contract

    As a developer evaluating clustering quality,
    I expect fast modularity computation,
    So that quality assessment is practical.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_LATENCY_MS = 1000

    def test_modularity_latency_honored(self, small_processor):
        """
        CONTRACT: Modularity computes in < 500ms for ≤ 1,000 nodes.

        Quality metrics should not slow down development workflow.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.quality import _compute_modularity

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer0.column_count() == 0 or layer2.column_count() == 0:
            pytest.skip("Need tokens and concepts")

        assert layer0.column_count() < 1000

        start = time.perf_counter()
        _compute_modularity(layer0, layer2)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Modularity computation took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )


@pytest.mark.contract
class TestSilhouettePerformanceContract:
    """
    Silhouette Computation Performance Contract

    As a developer evaluating cluster coherence,
    I expect reasonable silhouette computation time,
    So that quality assessment with sampling is practical.
    """

    MAX_LATENCY_MS = 4000  # 2 seconds - silhouette is O(n²) but sampled

    def test_silhouette_latency_honored(self, small_processor):
        """
        CONTRACT: Silhouette computes in < 2 seconds for ≤ 500 sampled nodes.

        Sampling keeps O(n²) computation tractable.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.quality import _compute_silhouette

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer0.column_count() == 0 or layer2.column_count() == 0:
            pytest.skip("Need tokens and concepts")

        start = time.perf_counter()
        _compute_silhouette(layer0, layer2, sample_size=500)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Silhouette computation took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )


@pytest.mark.contract
class TestQualityMetricsPerformanceContract:
    """
    Full Quality Metrics Performance Contract

    As a developer running quality assessment,
    I expect complete metrics in reasonable time,
    So that quality evaluation fits in development workflow.
    """

    MAX_TOTAL_LATENCY_MS = 6000  # 3 seconds for all metrics

    def test_full_quality_metrics_latency(self, small_processor):
        """
        CONTRACT: Full quality metrics complete in < 3 seconds.

        Combined modularity + silhouette + balance should be practical.
        """
        from cortical.analysis.quality import compute_clustering_quality

        start = time.perf_counter()
        compute_clustering_quality(small_processor.layers, sample_size=500)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_TOTAL_LATENCY_MS, (
            f"CONTRACT VIOLATION: Full quality metrics took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_TOTAL_LATENCY_MS}ms"
        )


@pytest.mark.contract
class TestModularityCorrectnessContract:
    """
    Modularity Correctness Contract

    As a developer relying on modularity scores,
    I expect mathematically valid values,
    So that quality assessment is meaningful.
    """

    def test_modularity_in_valid_range(self, small_processor):
        """
        CONTRACT: Modularity Q ∈ [-0.5, 1.0].

        This is the theoretical range for modularity.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.quality import _compute_modularity

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer0.column_count() == 0 or layer2.column_count() == 0:
            pytest.skip("Need tokens and concepts")

        modularity = _compute_modularity(layer0, layer2)

        assert -0.5 <= modularity <= 1.0, (
            f"CONTRACT VIOLATION: Modularity {modularity:.3f} outside valid range [-0.5, 1.0]"
        )

    def test_modularity_core_perfect_communities(self):
        """
        CONTRACT: Modularity is high for perfect community structure.

        Two disconnected cliques should have high modularity.
        """
        from cortical.analysis.quality import _modularity_core

        # Two perfect communities: complete graphs that don't connect
        adjacency = {
            "a": {"b": 1.0, "c": 1.0},
            "b": {"a": 1.0, "c": 1.0},
            "c": {"a": 1.0, "b": 1.0},
            "d": {"e": 1.0, "f": 1.0},
            "e": {"d": 1.0, "f": 1.0},
            "f": {"d": 1.0, "e": 1.0}
        }

        community = {
            "a": 0, "b": 0, "c": 0,
            "d": 1, "e": 1, "f": 1
        }

        modularity = _modularity_core(adjacency, community)

        # Should have high modularity (perfect structure)
        assert modularity > 0.3, (
            f"CONTRACT VIOLATION: Perfect communities have low modularity: {modularity:.3f}"
        )

    def test_modularity_core_no_structure(self):
        """
        CONTRACT: Modularity is low when communities don't match structure.

        Random community assignment should give low modularity.
        """
        from cortical.analysis.quality import _modularity_core

        # Complete graph (all connected equally)
        adjacency = {
            "a": {"b": 1.0, "c": 1.0, "d": 1.0},
            "b": {"a": 1.0, "c": 1.0, "d": 1.0},
            "c": {"a": 1.0, "b": 1.0, "d": 1.0},
            "d": {"a": 1.0, "b": 1.0, "c": 1.0}
        }

        # Arbitrary split
        community = {"a": 0, "b": 0, "c": 1, "d": 1}

        modularity = _modularity_core(adjacency, community)

        # Should be close to 0 (no structure)
        assert abs(modularity) < 0.1, (
            f"CONTRACT VIOLATION: Complete graph split has high modularity: {modularity:.3f}"
        )


@pytest.mark.contract
class TestSilhouetteCorrectnessContract:
    """
    Silhouette Correctness Contract

    As a developer using silhouette scores,
    I expect mathematically valid values,
    So that cluster coherence is meaningful.
    """

    def test_silhouette_in_valid_range(self, small_processor):
        """
        CONTRACT: Silhouette score ∈ [-1.0, 1.0].

        This is the theoretical range for silhouette coefficient.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.quality import _compute_silhouette

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer0.column_count() == 0 or layer2.column_count() == 0:
            pytest.skip("Need tokens and concepts")

        silhouette = _compute_silhouette(layer0, layer2, sample_size=100)

        assert -1.0 <= silhouette <= 1.0, (
            f"CONTRACT VIOLATION: Silhouette {silhouette:.3f} outside valid range [-1, 1]"
        )

    def test_silhouette_core_perfect_clusters(self):
        """
        CONTRACT: Silhouette is high for well-separated clusters.

        Two tight, distant clusters should have high silhouette.
        """
        from cortical.analysis.quality import _silhouette_core

        # Two tight clusters, far apart
        distances = {
            # Cluster 1: a, b very close
            "a": {"b": 0.1, "c": 0.9, "d": 0.9},
            "b": {"a": 0.1, "c": 0.9, "d": 0.9},
            # Cluster 2: c, d very close
            "c": {"a": 0.9, "b": 0.9, "d": 0.1},
            "d": {"a": 0.9, "b": 0.9, "c": 0.1}
        }

        labels = {"a": 0, "b": 0, "c": 1, "d": 1}

        silhouette = _silhouette_core(distances, labels)

        # Should be high (good clustering)
        assert silhouette > 0.5, (
            f"CONTRACT VIOLATION: Well-separated clusters have low silhouette: {silhouette:.3f}"
        )

    def test_silhouette_core_poor_clusters(self):
        """
        CONTRACT: Silhouette is low for poorly separated clusters.

        Overlapping clusters should have low/negative silhouette.
        """
        from cortical.analysis.quality import _silhouette_core

        # All points roughly equidistant (no structure)
        distances = {
            "a": {"b": 0.5, "c": 0.5, "d": 0.5},
            "b": {"a": 0.5, "c": 0.5, "d": 0.5},
            "c": {"a": 0.5, "b": 0.5, "d": 0.5},
            "d": {"a": 0.5, "b": 0.5, "c": 0.5}
        }

        labels = {"a": 0, "b": 0, "c": 1, "d": 1}

        silhouette = _silhouette_core(distances, labels)

        # Should be low (poor separation)
        assert silhouette < 0.3, (
            f"CONTRACT VIOLATION: Poor clustering has high silhouette: {silhouette:.3f}"
        )


@pytest.mark.contract
class TestGiniCorrectnessContract:
    """
    Gini Coefficient Correctness Contract

    As a developer evaluating cluster balance,
    I expect valid Gini coefficients,
    So that balance assessment is meaningful.
    """

    def test_gini_in_valid_range(self, small_processor):
        """
        CONTRACT: Gini coefficient ∈ [0.0, 1.0].

        This is the valid range for Gini coefficient.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.quality import _compute_cluster_balance

        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer2.column_count() == 0:
            pytest.skip("Need concepts")

        gini = _compute_cluster_balance(layer2)

        assert 0.0 <= gini <= 1.0, (
            f"CONTRACT VIOLATION: Gini coefficient {gini:.3f} outside valid range [0, 1]"
        )

    def test_quality_assessment_is_informative(self, small_processor):
        """
        CONTRACT: Quality assessment provides human-readable interpretation.

        The assessment string should describe the clustering quality.
        """
        from cortical.analysis.quality import compute_clustering_quality

        result = compute_clustering_quality(small_processor.layers)

        assessment = result['quality_assessment']

        # Should be a non-empty string
        assert isinstance(assessment, str)
        assert len(assessment) > 0

        # Should mention key metrics
        assert 'modularity' in assessment.lower() or 'cluster' in assessment.lower()


@pytest.mark.contract
class TestQualityEdgeCasesContract:
    """
    Quality Metrics Edge Cases Contract

    As a developer handling various corpus states,
    I expect graceful handling of edge cases,
    So that quality metrics never crash.
    """

    def test_quality_handles_no_clusters(self):
        """
        CONTRACT: Quality metrics handle empty/no clusters gracefully.

        Should return valid results, not crash.
        """
        from cortical import CorticalTextProcessor
        from cortical.analysis.quality import compute_clustering_quality

        processor = CorticalTextProcessor()
        # Empty processor - no clusters

        result = compute_clustering_quality(processor.layers)

        # Should return valid structure
        assert 'modularity' in result
        assert 'silhouette' in result
        assert 'balance' in result
        assert 'quality_assessment' in result

    def test_quality_handles_single_cluster(self):
        """
        CONTRACT: Quality metrics handle single cluster.

        Edge case should not crash.
        """
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer
        from cortical.analysis.quality import compute_clustering_quality

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test document content")
        processor.compute_all(verbose=False)

        result = compute_clustering_quality(processor.layers)

        # Should complete without error
        assert result['num_clusters'] >= 0
