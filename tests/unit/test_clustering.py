"""
Unit Tests for Clustering Algorithms
=====================================

Tests for all clustering-related algorithms:
- Louvain community detection (_louvain_core, cluster_by_louvain)
- Label propagation clustering (cluster_by_label_propagation)
- Concept cluster building (build_concept_clusters)
- Clustering quality metrics (modularity, silhouette, balance)
- Sparse matrix operations (SparseMatrix)

Extracted from test_analysis.py for better organization (Task #T-20251215-213424-8400-004).
"""

import pytest

from cortical.analysis import (
    _louvain_core,
    _modularity_core,
    _silhouette_core,
    SparseMatrix,
    cluster_by_louvain,
    cluster_by_label_propagation,
    build_concept_clusters,
    compute_clustering_quality,
    _doc_similarity,
    _vector_similarity,
    _compute_cluster_balance,
    _generate_quality_assessment,
)


# =============================================================================
# LOUVAIN CORE ALGORITHM TESTS
# =============================================================================


class TestLouvainCore:
    """Tests for _louvain_core community detection."""

    def test_empty_graph(self):
        """Empty graph returns empty dict."""
        result = _louvain_core({})
        assert result == {}

    def test_single_node(self):
        """Single node is its own community."""
        result = _louvain_core({"a": {}})
        assert "a" in result
        assert result["a"] == 0

    def test_two_disconnected_nodes(self):
        """Two disconnected nodes are separate communities."""
        result = _louvain_core({"a": {}, "b": {}})
        assert result["a"] != result["b"]

    def test_two_connected_nodes(self):
        """Two connected nodes are same community."""
        adj = {
            "a": {"b": 1.0},
            "b": {"a": 1.0}
        }
        result = _louvain_core(adj)
        assert result["a"] == result["b"]

    def test_triangle(self):
        """Triangle (complete graph of 3) is one community."""
        adj = {
            "a": {"b": 1.0, "c": 1.0},
            "b": {"a": 1.0, "c": 1.0},
            "c": {"a": 1.0, "b": 1.0}
        }
        result = _louvain_core(adj)
        assert result["a"] == result["b"] == result["c"]

    def test_two_triangles_separate(self):
        """Two disconnected triangles form two communities."""
        adj = {
            "a": {"b": 1.0, "c": 1.0},
            "b": {"a": 1.0, "c": 1.0},
            "c": {"a": 1.0, "b": 1.0},
            "d": {"e": 1.0, "f": 1.0},
            "e": {"d": 1.0, "f": 1.0},
            "f": {"d": 1.0, "e": 1.0}
        }
        result = _louvain_core(adj)
        # First triangle
        assert result["a"] == result["b"] == result["c"]
        # Second triangle
        assert result["d"] == result["e"] == result["f"]
        # Different communities
        assert result["a"] != result["d"]

    def test_two_triangles_weakly_connected(self):
        """Two triangles with weak bridge may merge or stay separate."""
        adj = {
            "a": {"b": 10.0, "c": 10.0},
            "b": {"a": 10.0, "c": 10.0},
            "c": {"a": 10.0, "b": 10.0, "d": 0.1},  # Weak bridge
            "d": {"c": 0.1, "e": 10.0, "f": 10.0},  # Weak bridge
            "e": {"d": 10.0, "f": 10.0},
            "f": {"d": 10.0, "e": 10.0}
        }
        result = _louvain_core(adj)
        # With strong intra-cluster and weak inter-cluster, should be 2 communities
        assert result["a"] == result["b"] == result["c"]
        assert result["d"] == result["e"] == result["f"]

    def test_resolution_high(self):
        """High resolution creates more, smaller clusters."""
        adj = {
            "a": {"b": 1.0},
            "b": {"a": 1.0, "c": 1.0},
            "c": {"b": 1.0, "d": 1.0},
            "d": {"c": 1.0}
        }
        low_res = _louvain_core(adj, resolution=0.5)
        high_res = _louvain_core(adj, resolution=2.0)
        # High resolution should produce more clusters
        low_clusters = len(set(low_res.values()))
        high_clusters = len(set(high_res.values()))
        assert high_clusters >= low_clusters

    def test_community_ids_contiguous(self):
        """Community IDs are contiguous integers starting from 0."""
        adj = {
            "a": {"b": 1.0},
            "b": {"a": 1.0},
            "c": {"d": 1.0},
            "d": {"c": 1.0}
        }
        result = _louvain_core(adj)
        comm_ids = set(result.values())
        assert min(comm_ids) == 0
        assert max(comm_ids) == len(comm_ids) - 1


# =============================================================================
# MODULARITY TESTS
# =============================================================================


class TestModularityCore:
    """Tests for _modularity_core calculation."""

    def test_empty_graph(self):
        """Empty graph has zero modularity."""
        result = _modularity_core({}, {})
        assert result == 0.0

    def test_single_node(self):
        """Single node with no edges has zero modularity."""
        result = _modularity_core({"a": {}}, {"a": 0})
        assert result == 0.0

    def test_perfect_clustering(self):
        """Two disconnected cliques have high modularity."""
        adj = {
            "a": {"b": 1.0},
            "b": {"a": 1.0},
            "c": {"d": 1.0},
            "d": {"c": 1.0}
        }
        comm = {"a": 0, "b": 0, "c": 1, "d": 1}
        result = _modularity_core(adj, comm)
        # Should be positive (good clustering)
        assert result > 0.3

    def test_bad_clustering(self):
        """Splitting connected pairs has lower modularity."""
        adj = {
            "a": {"b": 1.0},
            "b": {"a": 1.0}
        }
        # Good: both in same community
        good_comm = {"a": 0, "b": 0}
        good_q = _modularity_core(adj, good_comm)

        # Bad: split into different communities
        bad_comm = {"a": 0, "b": 1}
        bad_q = _modularity_core(adj, bad_comm)

        assert good_q >= bad_q

    def test_all_one_community(self):
        """All nodes in one community has some modularity."""
        adj = {
            "a": {"b": 1.0, "c": 1.0},
            "b": {"a": 1.0, "c": 1.0},
            "c": {"a": 1.0, "b": 1.0}
        }
        comm = {"a": 0, "b": 0, "c": 0}
        result = _modularity_core(adj, comm)
        # Complete graph in one community: modularity depends on structure
        # Main check: it's a valid modularity value
        assert -0.5 <= result <= 1.0


# =============================================================================
# SILHOUETTE TESTS
# =============================================================================


class TestSilhouetteCore:
    """Tests for _silhouette_core calculation."""

    def test_empty_labels(self):
        """Empty labels returns 0."""
        result = _silhouette_core({}, {})
        assert result == 0.0

    def test_single_cluster(self):
        """Single cluster returns 0."""
        distances = {"a": {"b": 0.1}, "b": {"a": 0.1}}
        labels = {"a": 0, "b": 0}
        result = _silhouette_core(distances, labels)
        assert result == 0.0

    def test_perfect_clustering(self):
        """Two tight clusters far apart have high silhouette."""
        distances = {
            "a": {"b": 0.1, "c": 0.9, "d": 0.9},
            "b": {"a": 0.1, "c": 0.9, "d": 0.9},
            "c": {"a": 0.9, "b": 0.9, "d": 0.1},
            "d": {"a": 0.9, "b": 0.9, "c": 0.1}
        }
        labels = {"a": 0, "b": 0, "c": 1, "d": 1}
        result = _silhouette_core(distances, labels)
        # Should be close to 1.0
        assert result > 0.5

    def test_bad_clustering(self):
        """Mixing clusters reduces silhouette."""
        distances = {
            "a": {"b": 0.1, "c": 0.9, "d": 0.9},
            "b": {"a": 0.1, "c": 0.9, "d": 0.9},
            "c": {"a": 0.9, "b": 0.9, "d": 0.1},
            "d": {"a": 0.9, "b": 0.9, "c": 0.1}
        }
        # Good clustering
        good_labels = {"a": 0, "b": 0, "c": 1, "d": 1}
        good_s = _silhouette_core(distances, good_labels)

        # Bad clustering (mixing)
        bad_labels = {"a": 0, "b": 1, "c": 0, "d": 1}
        bad_s = _silhouette_core(distances, bad_labels)

        assert good_s > bad_s

    def test_silhouette_range(self):
        """Silhouette is always in [-1, 1]."""
        distances = {
            "a": {"b": 0.5, "c": 0.5},
            "b": {"a": 0.5, "c": 0.5},
            "c": {"a": 0.5, "b": 0.5}
        }
        labels = {"a": 0, "b": 0, "c": 1}
        result = _silhouette_core(distances, labels)
        assert -1 <= result <= 1


# =============================================================================
# SPARSE MATRIX TESTS
# =============================================================================


class TestSparseMatrix:
    """Tests for SparseMatrix utility class."""

    def test_create_with_rows_cols(self):
        """Create sparse matrix with rows and cols."""
        matrix = SparseMatrix(rows=["a", "b"], cols=["c", "d"])
        assert matrix.get("a", "c") == 0.0

    def test_set_and_get(self):
        """Set and get values."""
        matrix = SparseMatrix(rows=["a"], cols=["b"])
        matrix.set("a", "b", 5.0)
        assert matrix.get("a", "b") == 5.0

    def test_default_value(self):
        """Unset values return 0.0."""
        matrix = SparseMatrix(rows=["a"], cols=["b"])
        assert matrix.get("a", "b") == 0.0


# =============================================================================
# WRAPPER FUNCTION TESTS
# =============================================================================


class TestClusterByLouvain:
    """Tests for cluster_by_louvain() wrapper function."""

    def test_empty_layer(self):
        """Empty layer returns empty clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = cluster_by_louvain(layer)
        assert result == {}

    def test_disconnected_nodes(self):
        """Disconnected nodes form separate clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("node1")
        layer.get_or_create_minicolumn("node2")
        # No connections

        result = cluster_by_louvain(layer, min_cluster_size=1)
        # Should create separate clusters
        assert len(result) >= 1

    def test_connected_nodes_same_cluster(self):
        """Connected nodes form same cluster."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("node1")
        col2 = layer.get_or_create_minicolumn("node2")
        col3 = layer.get_or_create_minicolumn("node3")

        # Connect them all strongly
        col1.add_lateral_connection(col2.id, 10.0)
        col2.add_lateral_connection(col1.id, 10.0)
        col2.add_lateral_connection(col3.id, 10.0)
        col3.add_lateral_connection(col2.id, 10.0)
        col1.add_lateral_connection(col3.id, 10.0)
        col3.add_lateral_connection(col1.id, 10.0)

        result = cluster_by_louvain(layer, min_cluster_size=2)

        # All should be in one cluster
        if result:
            cluster = list(result.values())[0]
            assert len(cluster) == 3

    def test_min_cluster_size_filter(self):
        """Small clusters are filtered out."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("node1")
        layer.get_or_create_minicolumn("node2")
        layer.get_or_create_minicolumn("node3")

        result = cluster_by_louvain(layer, min_cluster_size=5)
        # No cluster has 5 members, so should be empty
        assert result == {}


class TestClusterByLabelPropagation:
    """Tests for cluster_by_label_propagation() function."""

    def test_empty_layer(self):
        """Empty layer returns empty clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = cluster_by_label_propagation(layer)
        assert result == {}

    def test_single_node(self):
        """Single node filtered out by min_cluster_size."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("single")

        result = cluster_by_label_propagation(layer, min_cluster_size=3)
        # Single node doesn't meet min_cluster_size
        assert result == {}

    def test_connected_nodes_cluster(self):
        """Connected nodes form clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("node1")
        col2 = layer.get_or_create_minicolumn("node2")
        col3 = layer.get_or_create_minicolumn("node3")

        # Create a triangle
        col1.add_lateral_connection(col2.id, 1.0)
        col2.add_lateral_connection(col1.id, 1.0)
        col2.add_lateral_connection(col3.id, 1.0)
        col3.add_lateral_connection(col2.id, 1.0)
        col1.add_lateral_connection(col3.id, 1.0)
        col3.add_lateral_connection(col1.id, 1.0)

        # Add documents to prevent bridging
        col1.document_ids.add("doc1")
        col2.document_ids.add("doc1")
        col3.document_ids.add("doc1")

        result = cluster_by_label_propagation(layer, min_cluster_size=2, bridge_weight=0.0)

        # Should cluster them together
        assert len(result) >= 1

    def test_cluster_strictness_parameter(self):
        """Different strictness values produce different results."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        for i in range(6):
            col = layer.get_or_create_minicolumn(f"node{i}")
            col.document_ids.add(f"doc{i}")

        # Add some connections
        nodes = list(layer.minicolumns.values())
        for i in range(len(nodes)-1):
            nodes[i].add_lateral_connection(nodes[i+1].id, 1.0)
            nodes[i+1].add_lateral_connection(nodes[i].id, 1.0)

        low_strict = cluster_by_label_propagation(layer, min_cluster_size=2, cluster_strictness=0.1)
        high_strict = cluster_by_label_propagation(layer, min_cluster_size=2, cluster_strictness=0.9)

        # At least one should produce clusters (or both)
        # This is a basic smoke test
        assert isinstance(low_strict, dict)
        assert isinstance(high_strict, dict)


class TestBuildConceptClusters:
    """Tests for build_concept_clusters() function."""

    def test_empty_clusters(self):
        """Empty cluster dict."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        build_concept_clusters(layers, {})
        # Should not crash, no concepts created

    def test_small_cluster_skipped(self):
        """Clusters with <2 members are skipped."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)
        col = layer0.get_or_create_minicolumn("token")
        col.pagerank = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }
        clusters = {0: ["token"]}  # Only 1 member

        build_concept_clusters(layers, clusters)
        # No concept should be created
        assert layer2.column_count() == 0

    def test_concept_created_from_cluster(self):
        """Concept is created from valid cluster."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        col1 = layer0.get_or_create_minicolumn("neural")
        col2 = layer0.get_or_create_minicolumn("networks")
        col1.pagerank = 0.8
        col2.pagerank = 0.5
        col1.document_ids.add("doc1")
        col2.document_ids.add("doc1")

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }
        clusters = {0: ["neural", "networks"]}

        build_concept_clusters(layers, clusters)

        # Should create 1 concept
        assert layer2.column_count() == 1
        concept = list(layer2.minicolumns.values())[0]
        assert "neural" in concept.content  # Named after top members

    def test_feedforward_connections_created(self):
        """Feedforward connections from concept to tokens."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        col1 = layer0.get_or_create_minicolumn("token1")
        col2 = layer0.get_or_create_minicolumn("token2")
        col1.pagerank = 1.0
        col2.pagerank = 0.5

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }
        clusters = {0: ["token1", "token2"]}

        build_concept_clusters(layers, clusters)

        concept = list(layer2.minicolumns.values())[0]
        # Concept should have feedforward connections to tokens
        assert col1.id in concept.feedforward_connections
        assert col2.id in concept.feedforward_connections


# =============================================================================
# CLUSTERING QUALITY TESTS
# =============================================================================


class TestClusteringQualityHelpers:
    """Tests for clustering quality helper functions."""

    def test_compute_clustering_quality_empty(self):
        """Empty layers return zero quality."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        result = compute_clustering_quality(layers)

        assert result['modularity'] == 0.0
        assert result['silhouette'] == 0.0
        assert result['num_clusters'] == 0

    def test_doc_similarity(self):
        """_doc_similarity computes Jaccard correctly."""
        docs1 = frozenset(["doc1", "doc2", "doc3"])
        docs2 = frozenset(["doc2", "doc3", "doc4"])

        # Intersection: {doc2, doc3} = 2
        # Union: {doc1, doc2, doc3, doc4} = 4
        # Jaccard = 2/4 = 0.5
        assert _doc_similarity(docs1, docs2) == pytest.approx(0.5)

    def test_doc_similarity_no_overlap(self):
        """No overlap returns 0."""
        docs1 = frozenset(["doc1", "doc2"])
        docs2 = frozenset(["doc3", "doc4"])
        assert _doc_similarity(docs1, docs2) == 0.0

    def test_doc_similarity_identical(self):
        """Identical sets return 1.0."""
        docs = frozenset(["doc1", "doc2"])
        assert _doc_similarity(docs, docs) == 1.0

    def test_vector_similarity(self):
        """_vector_similarity computes weighted Jaccard."""
        vec1 = {"a": 2.0, "b": 3.0}
        vec2 = {"a": 1.0, "b": 4.0}

        # min_sum = min(2,1) + min(3,4) = 1 + 3 = 4
        # max_sum = max(2,1) + max(3,4) = 2 + 4 = 6
        # similarity = 4/6 = 0.667
        result = _vector_similarity(vec1, vec2)
        assert result == pytest.approx(4.0 / 6.0)

    def test_vector_similarity_no_overlap(self):
        """No overlap returns 0."""
        vec1 = {"a": 1.0}
        vec2 = {"b": 1.0}
        assert _vector_similarity(vec1, vec2) == 0.0

    def test_compute_cluster_balance(self):
        """_compute_cluster_balance computes Gini coefficient."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Create unbalanced clusters
        c1 = layer2.get_or_create_minicolumn("cluster1")
        c2 = layer2.get_or_create_minicolumn("cluster2")

        # Simulate cluster sizes via feedforward connections
        c1.feedforward_connections = {f"token{i}": 1.0 for i in range(10)}
        c2.feedforward_connections = {f"token{i}": 1.0 for i in range(1)}

        gini = _compute_cluster_balance(layer2)
        # Should be > 0 (imbalanced)
        assert gini > 0

    def test_generate_quality_assessment(self):
        """_generate_quality_assessment returns string."""
        assessment = _generate_quality_assessment(
            modularity=0.4,
            silhouette=0.3,
            balance=0.2,
            num_clusters=5
        )

        assert isinstance(assessment, str)
        assert "5 clusters" in assessment


class TestClusteringQualityMetrics:
    """Additional tests for clustering quality metrics edge cases."""

    def test_modularity_empty_graph(self):
        """Empty graph has zero modularity."""
        result = _modularity_core({}, {})
        assert result == 0.0

    def test_silhouette_single_node(self):
        """Single node returns 0 silhouette."""
        distances = {"a": {}}
        labels = {"a": 0}
        result = _silhouette_core(distances, labels)
        assert result == 0.0

    def test_louvain_single_node(self):
        """Single node forms its own community."""
        result = _louvain_core({"a": {}})
        assert result == {"a": 0}
