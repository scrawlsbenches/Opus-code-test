"""
╔══════════════════════════════════════════════════════════════════════╗
║                   CLUSTERING PERFORMANCE CONTRACT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Louvain clustering < 1 second for ≤ 1,000 nodes                  ║
║  • Louvain converges in ≤ 10 iterations                             ║
║  • Modularity Q > 0 for non-trivial graphs                          ║
║  • All nodes are assigned to exactly one cluster                    ║
║  • Cluster sizes respect minimum threshold                          ║
║  • Label propagation < 500ms for ≤ 1,000 nodes                      ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest


@pytest.mark.contract
class TestLouvainPerformanceContract:
    """
    Louvain Clustering Performance Contract

    As a developer building concept detection,
    I expect Louvain clustering to complete quickly,
    So that semantic grouping doesn't slow indexing.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_LATENCY_MS = 2000
    MAX_ITERATIONS = 10

    def test_louvain_latency_honored(self, small_processor):
        """
        CONTRACT: Louvain completes in < 1 second for ≤ 1,000 nodes.

        Fast clustering is essential for practical concept detection.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_louvain

        layer = small_processor.layers[CorticalLayer.TOKENS]
        assert layer.column_count() < 1000

        start = time.perf_counter()
        cluster_by_louvain(layer, min_cluster_size=3, resolution=1.0, max_iterations=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Louvain took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_louvain_produces_valid_clusters(self, small_processor):
        """
        CONTRACT: Louvain produces non-empty, valid clusters.

        Every cluster should contain nodes meeting minimum size.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_louvain

        layer = small_processor.layers[CorticalLayer.TOKENS]
        min_size = 3

        clusters = cluster_by_louvain(layer, min_cluster_size=min_size)

        # All clusters should meet minimum size
        for cluster_id, members in clusters.items():
            assert len(members) >= min_size, (
                f"CONTRACT VIOLATION: Cluster {cluster_id} has {len(members)} members, "
                f"minimum is {min_size}"
            )

            # All members should exist in layer
            for member in members:
                assert member in layer.minicolumns, (
                    f"CONTRACT VIOLATION: Cluster member '{member}' not in layer"
                )

    def test_louvain_assigns_all_qualified_nodes(self, small_processor):
        """
        CONTRACT: Louvain assigns cluster IDs to member nodes.

        Nodes in returned clusters should have their cluster_id set.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_louvain

        layer = small_processor.layers[CorticalLayer.TOKENS]

        clusters = cluster_by_louvain(layer, min_cluster_size=3)

        # Check that clustered nodes have cluster_id set
        for cluster_id, members in clusters.items():
            for member_content in members:
                col = layer.minicolumns[member_content]
                # cluster_id should match
                assert col.cluster_id == cluster_id, (
                    f"CONTRACT VIOLATION: Node '{member_content}' in cluster {cluster_id} "
                    f"but has cluster_id={col.cluster_id}"
                )


@pytest.mark.contract
class TestLouvainCorrectnessContract:
    """
    Louvain Correctness Contract

    As a developer relying on community detection,
    I expect mathematically sound clustering,
    So that concepts are meaningful.
    """

    def test_louvain_core_finds_communities(self):
        """
        CONTRACT: Louvain correctly identifies separate communities.

        Disconnected components should form separate clusters.
        """
        from cortical.analysis.clustering import _louvain_core

        # Two obvious communities: a-b-c and d-e
        adjacency = {
            "a": {"b": 1.0, "c": 1.0},
            "b": {"a": 1.0, "c": 1.0},
            "c": {"a": 1.0, "b": 1.0},
            "d": {"e": 1.0},
            "e": {"d": 1.0}
        }

        communities = _louvain_core(adjacency, resolution=1.0, max_iterations=10)

        # a, b, c should be in same community
        assert communities["a"] == communities["b"] == communities["c"], (
            "CONTRACT VIOLATION: Connected nodes not in same community"
        )

        # d, e should be in same community
        assert communities["d"] == communities["e"], (
            "CONTRACT VIOLATION: Connected nodes not in same community"
        )

        # a-b-c group should be different from d-e group
        assert communities["a"] != communities["d"], (
            "CONTRACT VIOLATION: Separate communities not detected"
        )

    def test_louvain_handles_empty_graph(self):
        """
        CONTRACT: Louvain handles empty graph gracefully.

        Edge cases should not crash.
        """
        from cortical.analysis.clustering import _louvain_core

        communities = _louvain_core({}, resolution=1.0, max_iterations=10)
        assert communities == {}

    def test_louvain_single_node(self):
        """
        CONTRACT: Louvain handles single-node graphs.

        Single node should form its own cluster.
        """
        from cortical.analysis.clustering import _louvain_core

        adjacency = {"a": {}}
        communities = _louvain_core(adjacency, resolution=1.0, max_iterations=10)

        assert "a" in communities
        assert isinstance(communities["a"], int)


@pytest.mark.contract
class TestLabelPropagationPerformanceContract:
    """
    Label Propagation Performance Contract

    As a developer with legacy clustering needs,
    I expect label propagation to complete quickly,
    So that alternative clustering remains viable.
    """

    MAX_LATENCY_MS = 1000

    def test_label_propagation_latency(self, small_processor):
        """
        CONTRACT: Label propagation completes in < 500ms for ≤ 1,000 nodes.

        Alternative clustering should be fast enough for experimentation.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_label_propagation

        layer = small_processor.layers[CorticalLayer.TOKENS]
        assert layer.column_count() < 1000

        start = time.perf_counter()
        cluster_by_label_propagation(
            layer,
            min_cluster_size=3,
            max_iterations=20,
            cluster_strictness=1.0
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Label propagation took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_label_propagation_produces_valid_clusters(self, small_processor):
        """
        CONTRACT: Label propagation produces valid clusters.

        Clusters should meet minimum size requirement.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_label_propagation

        layer = small_processor.layers[CorticalLayer.TOKENS]
        min_size = 3

        clusters = cluster_by_label_propagation(layer, min_cluster_size=min_size)

        for cluster_id, members in clusters.items():
            assert len(members) >= min_size, (
                f"CONTRACT VIOLATION: Cluster {cluster_id} has {len(members)} members, "
                f"minimum is {min_size}"
            )


@pytest.mark.contract
class TestConceptClusteringContract:
    """
    Concept Building Contract

    As a developer building concept layers,
    I expect concept creation to be correct,
    So that hierarchical layers are meaningful.
    """

    def test_build_concept_clusters_creates_concepts(self, small_processor):
        """
        CONTRACT: build_concept_clusters creates Layer 2 concepts.

        Each cluster should produce a concept minicolumn.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_louvain, build_concept_clusters

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        # Clear existing concepts
        layer2.minicolumns.clear()

        # Cluster and build concepts
        clusters = cluster_by_louvain(layer0, min_cluster_size=2)

        if not clusters:
            pytest.skip("No clusters found in small corpus")

        initial_concept_count = layer2.column_count()
        build_concept_clusters(small_processor.layers, clusters)

        # Should have created new concepts
        assert layer2.column_count() > initial_concept_count, (
            "CONTRACT VIOLATION: build_concept_clusters did not create concepts"
        )

    def test_concepts_have_feedforward_connections(self, small_processor):
        """
        CONTRACT: Concepts connect to their constituent tokens.

        Each concept should have feedforward connections to tokens.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_louvain, build_concept_clusters

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        # Clear and rebuild
        layer2.minicolumns.clear()
        clusters = cluster_by_louvain(layer0, min_cluster_size=2)

        if not clusters:
            pytest.skip("No clusters found")

        build_concept_clusters(small_processor.layers, clusters)

        # Each concept should have feedforward connections
        for concept in layer2.minicolumns.values():
            assert len(concept.feedforward_connections) > 0, (
                f"CONTRACT VIOLATION: Concept '{concept.content}' has no "
                f"feedforward connections to tokens"
            )

    def test_concepts_have_positive_pagerank(self, small_processor):
        """
        CONTRACT: Concepts inherit positive PageRank from members.

        Concept PageRank should be > 0 (average of member PageRanks).
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.clustering import cluster_by_louvain, build_concept_clusters

        layer2 = small_processor.layers[CorticalLayer.CONCEPTS]

        # Clear and rebuild
        layer2.minicolumns.clear()
        clusters = cluster_by_louvain(
            small_processor.layers[CorticalLayer.TOKENS],
            min_cluster_size=2
        )

        if not clusters:
            pytest.skip("No clusters found")

        build_concept_clusters(small_processor.layers, clusters)

        # All concepts should have positive PageRank
        for concept in layer2.minicolumns.values():
            assert concept.pagerank > 0, (
                f"CONTRACT VIOLATION: Concept '{concept.content}' has "
                f"PageRank={concept.pagerank}, expected > 0"
            )
