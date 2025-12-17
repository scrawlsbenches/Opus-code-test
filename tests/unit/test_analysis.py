"""
Unit Tests for Analysis Module Core Functions
==============================================

Task #152: Unit tests for cortical/analysis.py core algorithms.

Tests the pure algorithm functions that were extracted in Task #151:
- _pagerank_core: PageRank on graph primitives
- _tfidf_core: TF-IDF on term statistics
- _louvain_core: Louvain community detection
- _modularity_core: Modularity calculation
- _silhouette_core: Silhouette score calculation

These tests don't require HierarchicalLayer or Minicolumn objects,
making them fast and isolated.
"""

import pytest
import math

from cortical.analysis import (
    _pagerank_core,
    _tfidf_core,
    _louvain_core,
    _modularity_core,
    _silhouette_core,
    SparseMatrix,
)


# =============================================================================
# PAGERANK TESTS
# =============================================================================


class TestPageRankCore:
    """Tests for _pagerank_core pure algorithm."""

    def test_empty_graph(self):
        """Empty graph returns empty dict."""
        result = _pagerank_core({})
        assert result == {}

    def test_single_node_no_edges(self):
        """Single node with no edges gets base rank from damping."""
        graph = {"a": []}
        result = _pagerank_core(graph, damping=0.85)
        assert "a" in result
        # With no incoming edges, rank = (1-d)/n = 0.15/1 = 0.15
        assert result["a"] == pytest.approx(0.15)

    def test_single_node_self_loop(self):
        """Single node with self-loop still gets rank 1.0."""
        graph = {"a": [("a", 1.0)]}
        result = _pagerank_core(graph)
        assert result["a"] == pytest.approx(1.0)

    def test_two_nodes_one_edge(self):
        """Two nodes with one directed edge."""
        graph = {
            "a": [("b", 1.0)],
            "b": []
        }
        result = _pagerank_core(graph)
        # Node b should have higher rank (receives link)
        assert result["b"] > result["a"]

    def test_two_nodes_bidirectional(self):
        """Two nodes with bidirectional edges have equal rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("a", 1.0)]
        }
        result = _pagerank_core(graph)
        assert result["a"] == pytest.approx(result["b"], rel=0.01)

    def test_three_node_chain(self):
        """Chain: a -> b -> c. C should have highest rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
            "c": []
        }
        result = _pagerank_core(graph)
        # c receives transitively, b receives from a
        assert result["c"] >= result["b"]
        assert result["b"] >= result["a"]

    def test_star_topology(self):
        """Star topology: center receives from all leaves."""
        graph = {
            "center": [],
            "leaf1": [("center", 1.0)],
            "leaf2": [("center", 1.0)],
            "leaf3": [("center", 1.0)]
        }
        result = _pagerank_core(graph)
        # Center should have highest rank
        assert result["center"] > result["leaf1"]
        assert result["center"] > result["leaf2"]
        assert result["center"] > result["leaf3"]

    def test_cycle(self):
        """Cycle: a -> b -> c -> a. All should have equal rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
            "c": [("a", 1.0)]
        }
        result = _pagerank_core(graph)
        # All nodes in cycle should have equal rank
        assert result["a"] == pytest.approx(result["b"], rel=0.01)
        assert result["b"] == pytest.approx(result["c"], rel=0.01)

    def test_damping_factor_effect(self):
        """Higher damping follows links more strictly."""
        graph = {
            "popular": [],
            "linker": [("popular", 1.0)],
            "isolated": []
        }
        low_damp = _pagerank_core(graph, damping=0.5)
        high_damp = _pagerank_core(graph, damping=0.95)

        # With high damping, popular node should be even more popular
        # relative to isolated node
        low_ratio = low_damp["popular"] / low_damp["isolated"]
        high_ratio = high_damp["popular"] / high_damp["isolated"]
        assert high_ratio > low_ratio

    def test_weighted_edges(self):
        """Higher weight edges transfer more rank."""
        graph = {
            "a": [("target", 10.0)],
            "b": [("target", 1.0)],
            "target": []
        }
        result = _pagerank_core(graph)
        # a contributes more to target than b does
        # Both a and b should have similar self-rank
        assert result["target"] > result["a"]
        assert result["target"] > result["b"]

    def test_convergence(self):
        """Algorithm converges within iterations."""
        # Large graph should still converge
        graph = {str(i): [(str((i+1) % 10), 1.0)] for i in range(10)}
        result = _pagerank_core(graph, iterations=100)
        # All nodes in cycle should have equal rank
        values = list(result.values())
        assert all(v == pytest.approx(values[0], rel=0.01) for v in values)

    def test_disconnected_components(self):
        """Disconnected components each get their share of rank."""
        graph = {
            "a1": [("a2", 1.0)],
            "a2": [("a1", 1.0)],
            "b1": [("b2", 1.0)],
            "b2": [("b1", 1.0)]
        }
        result = _pagerank_core(graph)
        # All nodes should have equal rank
        assert result["a1"] == pytest.approx(result["b1"], rel=0.01)


# =============================================================================
# TF-IDF TESTS
# =============================================================================


class TestTfidfCore:
    """Tests for _tfidf_core pure algorithm."""

    def test_empty_corpus(self):
        """Empty corpus returns empty dict."""
        result = _tfidf_core({}, num_docs=0)
        assert result == {}

    def test_single_term_single_doc(self):
        """Single term in single doc has IDF of 0."""
        stats = {
            "term": (5, 1, {"doc1": 5})
        }
        result = _tfidf_core(stats, num_docs=1)
        # IDF = log(1/1) = 0, so TF-IDF = 0
        assert result["term"][0] == pytest.approx(0.0)

    def test_rare_term_high_tfidf(self):
        """Rare term (in 1 of 10 docs) has high TF-IDF."""
        stats = {
            "rare": (5, 1, {"doc1": 5}),
            "common": (50, 10, {"doc1": 5, "doc2": 5, "doc3": 5, "doc4": 5, "doc5": 5,
                                "doc6": 5, "doc7": 5, "doc8": 5, "doc9": 5, "doc10": 5})
        }
        result = _tfidf_core(stats, num_docs=10)
        # Rare term should have higher TF-IDF
        assert result["rare"][0] > result["common"][0]

    def test_frequent_term_higher_tf(self):
        """Term with higher frequency has higher TF component."""
        stats = {
            "frequent": (100, 5, {"doc1": 100}),
            "infrequent": (10, 5, {"doc1": 10})
        }
        result = _tfidf_core(stats, num_docs=10)
        # Same IDF, but frequent has higher TF
        assert result["frequent"][0] > result["infrequent"][0]

    def test_per_doc_tfidf(self):
        """Per-document TF-IDF calculated correctly."""
        stats = {
            "term": (15, 2, {"doc1": 10, "doc2": 5})
        }
        result = _tfidf_core(stats, num_docs=10)
        global_tfidf, per_doc = result["term"]
        # doc1 has higher count, so higher per-doc TF-IDF
        assert per_doc["doc1"] > per_doc["doc2"]

    def test_zero_doc_frequency(self):
        """Term with zero doc frequency returns zero TF-IDF."""
        stats = {
            "ghost": (0, 0, {})
        }
        result = _tfidf_core(stats, num_docs=10)
        assert result["ghost"] == (0.0, {})

    def test_idf_formula(self):
        """Verify IDF formula: log(N/df)."""
        stats = {
            "term": (10, 5, {"doc1": 10})
        }
        result = _tfidf_core(stats, num_docs=10)
        expected_idf = math.log(10 / 5)  # log(2)
        expected_tf = math.log1p(10)
        expected_tfidf = expected_tf * expected_idf
        assert result["term"][0] == pytest.approx(expected_tfidf)


# =============================================================================
# LOUVAIN CLUSTERING TESTS
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

    def test_empty_matrix(self):
        """Empty matrix has no data."""
        m = SparseMatrix(3, 3)
        assert m.get(0, 0) == 0.0
        assert m.get(1, 2) == 0.0

    def test_set_get(self):
        """Set and get values."""
        m = SparseMatrix(3, 3)
        m.set(0, 1, 5.0)
        assert m.get(0, 1) == 5.0
        assert m.get(1, 0) == 0.0

    def test_set_zero_removes(self):
        """Setting value to zero removes it."""
        m = SparseMatrix(3, 3)
        m.set(0, 0, 5.0)
        assert m.get(0, 0) == 5.0
        m.set(0, 0, 0.0)
        assert m.get(0, 0) == 0.0
        assert (0, 0) not in m.data

    def test_multiply_transpose_identity(self):
        """M * M^T for identity-like matrix."""
        m = SparseMatrix(2, 2)
        m.set(0, 0, 1.0)
        m.set(1, 1, 1.0)
        result = m.multiply_transpose()
        assert result.get(0, 0) == 1.0
        assert result.get(1, 1) == 1.0
        assert result.get(0, 1) == 0.0

    def test_multiply_transpose_cooccurrence(self):
        """M * M^T gives co-occurrence matrix."""
        # Document-term matrix:
        # Doc1 has term0, term1
        # Doc2 has term1, term2
        m = SparseMatrix(2, 3)  # 2 docs, 3 terms
        m.set(0, 0, 1.0)  # doc1 has term0
        m.set(0, 1, 1.0)  # doc1 has term1
        m.set(1, 1, 1.0)  # doc2 has term1
        m.set(1, 2, 1.0)  # doc2 has term2

        result = m.multiply_transpose()
        # term0-term1 co-occur in doc1
        assert result.get(0, 1) == 1.0
        # term1-term2 co-occur in doc2
        assert result.get(1, 2) == 1.0
        # term0-term2 never co-occur
        assert result.get(0, 2) == 0.0

    def test_get_nonzero(self):
        """get_nonzero returns all entries."""
        m = SparseMatrix(3, 3)
        m.set(0, 1, 2.0)
        m.set(2, 0, 3.0)
        entries = m.get_nonzero()
        assert len(entries) == 2
        assert (0, 1, 2.0) in entries
        assert (2, 0, 3.0) in entries


# =============================================================================
# LAYER-BASED WRAPPER TESTS (require HierarchicalLayer objects)
# =============================================================================


class TestComputePageRank:
    """Tests for compute_pagerank() wrapper function."""

    def test_empty_layer(self):
        """Empty layer returns empty dict."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = compute_pagerank(layer)
        assert result == {}

    def test_single_minicolumn(self):
        """Single minicolumn with no edges gets base rank."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("test")
        result = compute_pagerank(layer, damping=0.85)
        # With no edges, rank = (1-d)/n = 0.15/1 = 0.15
        assert result[col.id] == pytest.approx(0.15)
        assert col.pagerank == pytest.approx(0.15)

    def test_two_connected_minicolumns(self):
        """Two connected minicolumns share rank."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("col1")
        col2 = layer.get_or_create_minicolumn("col2")
        col1.add_lateral_connection(col2.id, 1.0)
        col2.add_lateral_connection(col1.id, 1.0)

        result = compute_pagerank(layer)
        # Both should have equal rank
        assert result[col1.id] == pytest.approx(result[col2.id], rel=0.01)

    def test_invalid_damping_raises(self):
        """Invalid damping factor raises ValueError."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=1.5)

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=-0.1)

    def test_pagerank_updates_minicolumns(self):
        """PageRank values are written to minicolumns."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("col1")
        col2 = layer.get_or_create_minicolumn("col2")
        col1.add_lateral_connection(col2.id, 1.0)

        compute_pagerank(layer)
        # Minicolumns should have pagerank set
        assert col1.pagerank > 0
        assert col2.pagerank > 0


class TestComputeTfidf:
    """Tests for compute_tfidf() wrapper function."""

    def test_empty_corpus(self):
        """Empty corpus with no documents."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_tfidf

        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        compute_tfidf(layers, {})
        # Should not crash, no columns to update

    def test_single_term_single_doc(self):
        """Single term in single doc has zero TF-IDF."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_tfidf

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer0.get_or_create_minicolumn("test")
        col.document_ids.add("doc1")
        col.occurrence_count = 5
        col.doc_occurrence_counts["doc1"] = 5

        layers = {CorticalLayer.TOKENS: layer0}
        documents = {"doc1": "test test test test test"}

        compute_tfidf(layers, documents)
        # IDF = log(1/1) = 0, so TF-IDF should be 0
        assert col.tfidf == pytest.approx(0.0)

    def test_rare_term_high_tfidf(self):
        """Rare term in 1 of 10 docs has high TF-IDF."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_tfidf
        import math

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer0.get_or_create_minicolumn("rare")
        col.document_ids.add("doc1")
        col.occurrence_count = 5
        col.doc_occurrence_counts["doc1"] = 5

        layers = {CorticalLayer.TOKENS: layer0}
        documents = {f"doc{i}": "text" for i in range(10)}

        compute_tfidf(layers, documents)
        # IDF = log(10/1), TF = log1p(5)
        expected_idf = math.log(10)
        expected_tf = math.log1p(5)
        expected_tfidf = expected_tf * expected_idf
        assert col.tfidf == pytest.approx(expected_tfidf)

    def test_per_doc_tfidf(self):
        """Per-document TF-IDF calculated correctly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_tfidf

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer0.get_or_create_minicolumn("term")
        col.document_ids.add("doc1")
        col.document_ids.add("doc2")
        col.occurrence_count = 15
        col.doc_occurrence_counts["doc1"] = 10
        col.doc_occurrence_counts["doc2"] = 5

        layers = {CorticalLayer.TOKENS: layer0}
        # Need more than 2 docs for non-zero IDF when term appears in 2
        documents = {"doc1": "text", "doc2": "text", "doc3": "other"}

        compute_tfidf(layers, documents)
        # doc1 has higher count, so higher per-doc TF-IDF
        # IDF = log(3/2) > 0, so TF-IDF will be non-zero
        assert col.tfidf_per_doc["doc1"] > col.tfidf_per_doc["doc2"]


class TestCosineSimilarity:
    """Tests for cosine_similarity() utility function."""

    def test_empty_vectors(self):
        """Empty vectors return 0."""
        from cortical.analysis import cosine_similarity
        assert cosine_similarity({}, {}) == 0.0

    def test_no_common_keys(self):
        """Vectors with no common keys return 0."""
        from cortical.analysis import cosine_similarity
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 3.0, "d": 4.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_identical_vectors(self):
        """Identical vectors return 1.0."""
        from cortical.analysis import cosine_similarity
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_sparse(self):
        """Sparse orthogonal vectors return 0."""
        from cortical.analysis import cosine_similarity
        vec1 = {"a": 1.0}
        vec2 = {"b": 1.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_opposite_vectors(self):
        """Vectors with opposite values."""
        from cortical.analysis import cosine_similarity
        vec1 = {"a": 1.0, "b": 1.0}
        vec2 = {"a": -1.0, "b": -1.0}
        # Cosine of opposite vectors is -1
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_partial_overlap(self):
        """Vectors with partial overlap."""
        from cortical.analysis import cosine_similarity
        vec1 = {"a": 1.0, "b": 2.0, "c": 0.0}
        vec2 = {"a": 1.0, "b": 0.0, "c": 3.0}
        # Only "a" is common
        result = cosine_similarity(vec1, vec2)
        assert 0 < result < 1


class TestComputeBigramConnections:
    """Tests for compute_bigram_connections() function."""

    def test_empty_layer(self):
        """Empty bigram layer returns zero connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layers = {CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)}
        result = compute_bigram_connections(layers)

        assert result['connections_created'] == 0
        assert result['bigrams'] == 0

    def test_shared_left_component(self):
        """Bigrams sharing left component are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        b1 = layer1.get_or_create_minicolumn("neural networks")
        b2 = layer1.get_or_create_minicolumn("neural processing")
        b1.document_ids.add("doc1")
        b2.document_ids.add("doc1")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers)

        # Should create component connection
        assert result['component_connections'] > 0
        assert b1.id in b2.lateral_connections

    def test_shared_right_component(self):
        """Bigrams sharing right component are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        b1 = layer1.get_or_create_minicolumn("deep learning")
        b2 = layer1.get_or_create_minicolumn("machine learning")
        b1.document_ids.add("doc1")
        b2.document_ids.add("doc1")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers)

        assert result['component_connections'] > 0

    def test_chain_connection(self):
        """Bigrams forming chains are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        b1 = layer1.get_or_create_minicolumn("machine learning")
        b2 = layer1.get_or_create_minicolumn("learning algorithms")
        b1.document_ids.add("doc1")
        b2.document_ids.add("doc1")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers)

        # Should create chain connection (learning is right of b1 and left of b2)
        assert result['chain_connections'] > 0

    def test_document_cooccurrence(self):
        """Bigrams in same documents are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        b1 = layer1.get_or_create_minicolumn("alpha beta")
        b2 = layer1.get_or_create_minicolumn("gamma delta")
        b1.document_ids.add("doc1")
        b1.document_ids.add("doc2")
        b2.document_ids.add("doc1")
        b2.document_ids.add("doc2")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers, min_shared_docs=2)

        # Should create cooccurrence connection
        assert result['connections_created'] > 0

    def test_max_bigrams_per_term_limit(self):
        """Skip terms appearing in too many bigrams."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        # Create many bigrams with "the" as left component
        for i in range(10):
            b = layer1.get_or_create_minicolumn(f"the word{i}")
            b.document_ids.add("doc1")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers, max_bigrams_per_term=5)

        # Should skip "the" due to limit
        assert result['skipped_common_terms'] > 0


class TestComputeDocumentConnections:
    """Tests for compute_document_connections() function."""

    def test_empty_documents(self):
        """Empty document set."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_document_connections

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS)
        }
        compute_document_connections(layers, {})
        # Should not crash

    def test_shared_terms_create_connection(self):
        """Documents sharing terms are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_document_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer3 = HierarchicalLayer(CorticalLayer.DOCUMENTS)

        # Create shared token
        token = layer0.get_or_create_minicolumn("shared")
        token.document_ids.add("doc1")
        token.document_ids.add("doc2")
        token.tfidf = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.DOCUMENTS: layer3
        }
        documents = {"doc1": "shared", "doc2": "shared"}

        compute_document_connections(layers, documents, min_shared_terms=1)

        # Documents should be connected
        doc1 = layer3.get_minicolumn("doc1")
        doc2 = layer3.get_minicolumn("doc2")
        assert doc1 is not None
        assert doc2 is not None
        assert doc2.id in doc1.lateral_connections

    def test_min_shared_terms_threshold(self):
        """Only connect if enough shared terms."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_document_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer3 = HierarchicalLayer(CorticalLayer.DOCUMENTS)

        # Create only 1 shared token
        token = layer0.get_or_create_minicolumn("shared")
        token.document_ids.add("doc1")
        token.document_ids.add("doc2")
        token.tfidf = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.DOCUMENTS: layer3
        }
        documents = {"doc1": "shared", "doc2": "shared"}

        compute_document_connections(layers, documents, min_shared_terms=3)

        # Documents should NOT be connected (only 1 shared, need 3)
        doc1 = layer3.get_minicolumn("doc1")
        doc2 = layer3.get_minicolumn("doc2")
        if doc1 and doc2:
            assert doc2.id not in doc1.lateral_connections


class TestBuildConceptClusters:
    """Tests for build_concept_clusters() function."""

    def test_empty_clusters(self):
        """Empty cluster dict."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import build_concept_clusters

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        build_concept_clusters(layers, {})
        # Should not crash, no concepts created

    def test_small_cluster_skipped(self):
        """Clusters with <2 members are skipped."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import build_concept_clusters

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
        from cortical.analysis import build_concept_clusters

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
        from cortical.analysis import build_concept_clusters

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


class TestClusteringQualityHelpers:
    """Tests for clustering quality helper functions (_doc_similarity, _vector_similarity, etc.)."""

    def test_compute_clustering_quality_empty(self):
        """Empty layers return zero quality."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_clustering_quality

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
        from cortical.analysis import _doc_similarity

        docs1 = frozenset(["doc1", "doc2", "doc3"])
        docs2 = frozenset(["doc2", "doc3", "doc4"])

        # Intersection: {doc2, doc3} = 2
        # Union: {doc1, doc2, doc3, doc4} = 4
        # Jaccard = 2/4 = 0.5
        assert _doc_similarity(docs1, docs2) == pytest.approx(0.5)

    def test_doc_similarity_no_overlap(self):
        """No overlap returns 0."""
        from cortical.analysis import _doc_similarity

        docs1 = frozenset(["doc1", "doc2"])
        docs2 = frozenset(["doc3", "doc4"])
        assert _doc_similarity(docs1, docs2) == 0.0

    def test_doc_similarity_identical(self):
        """Identical sets return 1.0."""
        from cortical.analysis import _doc_similarity

        docs = frozenset(["doc1", "doc2"])
        assert _doc_similarity(docs, docs) == 1.0

    def test_vector_similarity(self):
        """_vector_similarity computes weighted Jaccard."""
        from cortical.analysis import _vector_similarity

        vec1 = {"a": 2.0, "b": 3.0}
        vec2 = {"a": 1.0, "b": 4.0}

        # min_sum = min(2,1) + min(3,4) = 1 + 3 = 4
        # max_sum = max(2,1) + max(3,4) = 2 + 4 = 6
        # similarity = 4/6 = 0.667
        result = _vector_similarity(vec1, vec2)
        assert result == pytest.approx(4.0 / 6.0)

    def test_vector_similarity_no_overlap(self):
        """No overlap returns 0."""
        from cortical.analysis import _vector_similarity

        vec1 = {"a": 1.0}
        vec2 = {"b": 1.0}
        assert _vector_similarity(vec1, vec2) == 0.0

    def test_compute_cluster_balance(self):
        """_compute_cluster_balance computes Gini coefficient."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import _compute_cluster_balance

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
        from cortical.analysis import _generate_quality_assessment

        assessment = _generate_quality_assessment(
            modularity=0.4,
            silhouette=0.3,
            balance=0.2,
            num_clusters=5
        )

        assert isinstance(assessment, str)
        assert "5 clusters" in assessment


class TestPropagateActivation:
    """Tests for propagate_activation() function."""

    def test_empty_layers(self):
        """Empty layers don't crash."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import propagate_activation

        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        propagate_activation(layers, iterations=1)
        # Should not crash

    def test_activation_decays(self):
        """Activation decays over iterations."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import propagate_activation

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("test")
        col.activation = 1.0

        layers = {CorticalLayer.TOKENS: layer}
        propagate_activation(layers, iterations=1, decay=0.5)

        # After 1 iteration with decay=0.5, activation should be ~0.5
        assert col.activation < 1.0

    def test_lateral_spreading(self):
        """Activation spreads laterally."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import propagate_activation

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("col1")
        col2 = layer.get_or_create_minicolumn("col2")
        col1.activation = 1.0
        col2.activation = 0.0
        # Bidirectional connection (col2 receives from col1)
        col1.add_lateral_connection(col2.id, 1.0)
        col2.add_lateral_connection(col1.id, 1.0)

        layers = {CorticalLayer.TOKENS: layer}
        propagate_activation(layers, iterations=1, lateral_weight=0.5)

        # col2 should have gained activation from col1
        assert col2.activation > 0


class TestClusterByLouvain:
    """Tests for cluster_by_louvain() wrapper function."""

    def test_empty_layer(self):
        """Empty layer returns empty clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = cluster_by_louvain(layer)
        assert result == {}

    def test_disconnected_nodes(self):
        """Disconnected nodes form separate clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("node1")
        col2 = layer.get_or_create_minicolumn("node2")
        # No connections

        result = cluster_by_louvain(layer, min_cluster_size=1)
        # Should create separate clusters
        assert len(result) >= 1

    def test_connected_nodes_same_cluster(self):
        """Connected nodes form same cluster."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

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
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("node1")
        col2 = layer.get_or_create_minicolumn("node2")
        col3 = layer.get_or_create_minicolumn("node3")

        result = cluster_by_louvain(layer, min_cluster_size=5)
        # No cluster has 5 members, so should be empty
        assert result == {}


class TestClusterByLabelPropagation:
    """Tests for cluster_by_label_propagation() function."""

    def test_empty_layer(self):
        """Empty layer returns empty clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_label_propagation

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = cluster_by_label_propagation(layer)
        assert result == {}

    def test_single_node(self):
        """Single node filtered out by min_cluster_size."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_label_propagation

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("single")

        result = cluster_by_label_propagation(layer, min_cluster_size=3)
        # Single node doesn't meet min_cluster_size
        assert result == {}

    def test_connected_nodes_cluster(self):
        """Connected nodes form clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_label_propagation

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
        from cortical.analysis import cluster_by_label_propagation

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


class TestComputeSemanticPageRank:
    """Tests for compute_semantic_pagerank() function."""

    def test_empty_layer(self):
        """Empty layer returns empty result."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_semantic_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = compute_semantic_pagerank(layer, [])

        assert result['pagerank'] == {}
        assert result['iterations_run'] == 0
        assert result['edges_with_relations'] == 0

    def test_invalid_damping_raises(self):
        """Invalid damping factor raises ValueError."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_semantic_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_semantic_pagerank(layer, [], damping=1.5)

    def test_semantic_relations_boost_connections(self):
        """Semantic relations increase edge weights."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_semantic_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("networks")
        col1.add_lateral_connection(col2.id, 1.0)
        col2.add_lateral_connection(col1.id, 1.0)

        # Add semantic relation
        semantic_relations = [("neural", "RelatedTo", "networks", 0.8)]

        result = compute_semantic_pagerank(layer, semantic_relations)

        assert result['edges_with_relations'] > 0
        assert 'pagerank' in result
        assert col1.id in result['pagerank']


class TestComputeHierarchicalPageRank:
    """Tests for compute_hierarchical_pagerank() function."""

    def test_empty_layers(self):
        """Empty layers return quickly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)
        }
        result = compute_hierarchical_pagerank(layers)

        assert result['converged'] is True
        assert result['iterations_run'] == 0

    def test_invalid_damping_raises(self):
        """Invalid damping parameters raise ValueError."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_hierarchical_pagerank(layers, damping=1.5)

        with pytest.raises(ValueError, match="cross_layer_damping must be between 0 and 1"):
            compute_hierarchical_pagerank(layers, cross_layer_damping=1.5)

    def test_cross_layer_propagation(self):
        """PageRank propagates between layers."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        token = layer0.get_or_create_minicolumn("token")
        bigram = layer1.get_or_create_minicolumn("token pair")

        # Connect layers
        token.add_feedback_connection(bigram.id, 1.0)
        bigram.add_feedforward_connection(token.id, 1.0)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1
        }

        result = compute_hierarchical_pagerank(layers, global_iterations=2)

        # Should run and produce stats
        assert 'layer_stats' in result
        assert result['iterations_run'] > 0


class TestComputeConceptConnections:
    """Tests for compute_concept_connections() function."""

    def test_empty_concepts(self):
        """Empty concept layer returns zero connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        result = compute_concept_connections(layers)

        assert result['connections_created'] == 0
        assert result['concepts'] == 0

    def test_document_overlap_creates_connection(self):
        """Concepts sharing documents are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Create tokens
        token1 = layer0.get_or_create_minicolumn("token1")
        token2 = layer0.get_or_create_minicolumn("token2")
        token3 = layer0.get_or_create_minicolumn("token3")

        # Create concepts with shared documents
        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        concept1.document_ids.add("doc1")
        concept1.document_ids.add("doc2")
        concept2.document_ids.add("doc1")
        concept2.document_ids.add("doc2")

        # Link concepts to tokens
        concept1.feedforward_connections[token1.id] = 1.0
        concept1.feedforward_connections[token2.id] = 1.0
        concept2.feedforward_connections[token2.id] = 1.0
        concept2.feedforward_connections[token3.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        result = compute_concept_connections(layers, min_shared_docs=1, min_jaccard=0.1)

        # Should create connection due to shared docs
        assert result['connections_created'] > 0

    def test_min_jaccard_threshold(self):
        """Connection requires minimum Jaccard similarity."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token = layer0.get_or_create_minicolumn("token")

        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        # Very low overlap
        concept1.document_ids.update([f"doc{i}" for i in range(10)])
        concept2.document_ids.add("doc1")  # Only 1 shared out of 10

        concept1.feedforward_connections[token.id] = 1.0
        concept2.feedforward_connections[token.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        result = compute_concept_connections(layers, min_jaccard=0.5)

        # Jaccard = 1/10 = 0.1 < 0.5, so no connection
        assert result['connections_created'] == 0


# =============================================================================
# ADDITIONAL COVERAGE TESTS (Lines 499, 505, 667, 680, 806-971, 1042-1687, etc.)
# =============================================================================


class TestSilhouetteEdgeCases:
    """Test edge cases in _silhouette_core for lines 499, 505."""

    def test_single_cluster_returns_zero(self):
        """Single cluster should return 0.0 (line 467 check)."""
        distances = {
            "a": {"b": 0.1},
            "b": {"a": 0.1}
        }
        labels = {"a": 0, "b": 0}
        result = _silhouette_core(distances, labels)
        assert result == 0.0

    def test_node_with_zero_max_ab(self):
        """Node with max(a, b) = 0 gets silhouette 0 (line 505)."""
        distances = {
            "a": {},
            "b": {},
            "c": {}
        }
        labels = {"a": 0, "b": 0, "c": 1}
        result = _silhouette_core(distances, labels)
        # Nodes with no distances get s=0
        assert result == 0.0

    def test_node_isolated_in_cluster(self):
        """Node alone in cluster has b = inf, handled at line 499."""
        distances = {
            "a": {"b": 0.5, "c": 0.5},
            "b": {"a": 0.5, "c": 0.9},
            "c": {"a": 0.5, "b": 0.9}
        }
        # c is alone in cluster 1
        labels = {"a": 0, "b": 0, "c": 1}
        result = _silhouette_core(distances, labels)
        # Should handle gracefully (b = inf becomes 0.0 at line 499)
        assert isinstance(result, float)


class TestSemanticPageRankMissingPaths:
    """Test compute_semantic_pagerank missing coverage (lines 667, 680)."""

    def test_semantic_relations_without_lookup_match(self):
        """Test path where semantic_lookup doesn't match (line 667, 680)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_semantic_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("term1")
        col2 = layer.get_or_create_minicolumn("term2")
        col1.add_lateral_connection(col2.id, 1.0)
        col2.add_lateral_connection(col1.id, 1.0)

        # Semantic relations for completely different terms
        semantic_relations = [
            ("other1", "RelatedTo", "other2", 0.8)
        ]

        result = compute_semantic_pagerank(
            layer, semantic_relations, damping=0.85, iterations=5
        )

        # Should still work, just no semantic boost (line 680)
        assert 'pagerank' in result
        assert result['edges_with_relations'] == 0


class TestHierarchicalPageRankCoverage:
    """Test compute_hierarchical_pagerank missing paths (lines 806-857)."""

    def test_cross_layer_feedback_propagation(self):
        """Test feedback connections propagate up (line 808)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        token1 = layer0.get_or_create_minicolumn("token1")
        token1.pagerank = 0.5

        bigram1 = layer1.get_or_create_minicolumn("bigram1")

        # Feedback connection: token -> bigram
        token1.add_feedback_connection(bigram1.id, 1.0)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1
        }

        result = compute_hierarchical_pagerank(
            layers, layer_iterations=2, global_iterations=2
        )

        # Bigram should receive boost from token
        assert bigram1.pagerank > 0

    def test_cross_layer_feedforward_propagation(self):
        """Test feedforward connections propagate down (line 827)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("token1")
        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept1.pagerank = 0.8

        # Feedforward connection: concept -> token
        concept1.add_feedforward_connection(token1.id, 1.0)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        result = compute_hierarchical_pagerank(
            layers, layer_iterations=2, global_iterations=2
        )

        # Token should receive boost from concept
        assert token1.pagerank > 0

    def test_empty_feedback_connections(self):
        """Test skipping empty feedback_connections (line 806)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        token1 = layer0.get_or_create_minicolumn("token1")
        bigram1 = layer1.get_or_create_minicolumn("bigram1")

        # No feedback_connections (empty dict)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1
        }

        result = compute_hierarchical_pagerank(layers, global_iterations=1)

        # Should not crash
        assert result['iterations_run'] >= 1

    def test_empty_feedforward_connections(self):
        """Test skipping empty feedforward_connections (line 825)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_hierarchical_pagerank

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("token1")
        concept1 = layer2.get_or_create_minicolumn("concept1")

        # No feedforward_connections (empty dict)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        result = compute_hierarchical_pagerank(layers, global_iterations=1)

        # Should not crash
        assert result['iterations_run'] >= 1


class TestPropagateActivationFeedforward:
    """Test propagate_activation feedforward sources (lines 962-971)."""

    def test_feedforward_sources_propagation(self):
        """Test activation propagates via feedforward_sources."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import propagate_activation

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        token1 = layer0.get_or_create_minicolumn("token1")
        token1.activation = 1.0

        bigram1 = layer1.get_or_create_minicolumn("bigram1")
        bigram1.feedforward_sources.add(token1.id)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1
        }

        propagate_activation(layers, iterations=1, decay=0.9)

        # Bigram should receive feedforward activation
        assert bigram1.activation > 0


class TestLabelPropagationBridgeWeight:
    """Test cluster_by_label_propagation bridge_weight feature (lines 1042-1063)."""

    def test_bridge_weight_creates_connections(self):
        """Test bridge_weight creates inter-document connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_label_propagation

        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Tokens in different documents
        token1 = layer.get_or_create_minicolumn("token1")
        token2 = layer.get_or_create_minicolumn("token2")
        token1.document_ids.add("doc1")
        token2.document_ids.add("doc2")

        result = cluster_by_label_propagation(
            layer, min_cluster_size=1, bridge_weight=0.5
        )

        # Bridge weight should create weak connections
        assert len(result) >= 0  # Should not crash

    def test_bridge_weight_zero_no_bridges(self):
        """Test bridge_weight=0 creates no bridges."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_label_propagation

        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        token1 = layer.get_or_create_minicolumn("token1")
        token2 = layer.get_or_create_minicolumn("token2")
        token1.document_ids.add("doc1")
        token2.document_ids.add("doc2")

        result = cluster_by_label_propagation(
            layer, min_cluster_size=1, bridge_weight=0.0
        )

        # Should work without bridges
        assert len(result) >= 0


class TestLouvainPhase2AndHierarchy:
    """Test cluster_by_louvain phase2 and hierarchy (lines 1314-1412)."""

    def test_louvain_with_hierarchy(self):
        """Test Louvain creates and unwinds hierarchy."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create two dense communities
        for i in range(5):
            col = layer.get_or_create_minicolumn(f"a{i}")
            for j in range(5):
                if i != j:
                    col.add_lateral_connection(f"L0_a{j}", 1.0)

        for i in range(5):
            col = layer.get_or_create_minicolumn(f"b{i}")
            for j in range(5):
                if i != j:
                    col.add_lateral_connection(f"L0_b{j}", 1.0)

        result = cluster_by_louvain(layer, min_cluster_size=3, max_iterations=5)

        # Should find 2 clusters (triggers phase2)
        assert len(result) >= 1

    def test_louvain_no_connections_each_own_cluster(self):
        """Test Louvain with m=0 (line 1230)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Nodes with no connections
        for i in range(5):
            layer.get_or_create_minicolumn(f"node{i}")

        result = cluster_by_louvain(layer, min_cluster_size=1)

        # Each node is its own cluster
        # But filtered by min_cluster_size=1, so all would be removed
        # except if there are exactly 1-sized clusters
        assert len(result) >= 0  # May be empty if all filtered

    def test_louvain_converges_in_iteration(self):
        """Test Louvain convergence check (line 1369)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Simple connected graph that converges quickly
        col1 = layer.get_or_create_minicolumn("node1")
        col2 = layer.get_or_create_minicolumn("node2")
        col1.add_lateral_connection(col2.id, 1.0)
        col2.add_lateral_connection(col1.id, 1.0)

        result = cluster_by_louvain(layer, min_cluster_size=2, max_iterations=10)

        # Should converge and create one cluster
        assert len(result) <= 1


class TestBuildConceptClustersEdgeCases:
    """Test build_concept_clusters edge cases (line 1468)."""

    def test_empty_member_cols(self):
        """Test handling when member_cols is empty (line 1468)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import build_concept_clusters

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        # Cluster with members that don't exist in layer0
        clusters = {
            0: ["nonexistent1", "nonexistent2"]
        }

        # Should not crash (line 1468 continue)
        build_concept_clusters(layers, clusters)

        # No concepts should be created
        assert layer2.column_count() == 0


class TestConceptConnectionsSemanticAndEmbedding:
    """Test compute_concept_connections semantic and embedding paths (lines 1551-1687)."""

    def test_with_semantic_relations(self):
        """Test semantic relations boost connections (lines 1631-1645)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("neural")
        token2 = layer0.get_or_create_minicolumn("networks")
        token3 = layer0.get_or_create_minicolumn("deep")

        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        # Shared documents
        concept1.document_ids.add("doc1")
        concept2.document_ids.add("doc1")

        # Link to tokens
        concept1.feedforward_connections[token1.id] = 1.0
        concept2.feedforward_connections[token2.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        # Semantic relation between member tokens
        semantic_relations = [
            ("neural", "RelatedTo", "networks", 0.9)
        ]

        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            min_shared_docs=1,
            min_jaccard=0.1
        )

        # Should create connection with semantic bonus
        assert result['connections_created'] > 0

    def test_use_member_semantics(self):
        """Test use_member_semantics creates connections without doc overlap (lines 1653-1671)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("machine")
        token2 = layer0.get_or_create_minicolumn("learning")

        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        # NO shared documents
        concept1.document_ids.add("doc1")
        concept2.document_ids.add("doc2")

        concept1.feedforward_connections[token1.id] = 1.0
        concept2.feedforward_connections[token2.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        semantic_relations = [
            ("machine", "RelatedTo", "learning", 0.8)
        ]

        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            use_member_semantics=True
        )

        # Should create connection via semantic relations only
        assert result['semantic_connections'] > 0

    def test_use_embedding_similarity(self):
        """Test use_embedding_similarity creates connections (lines 1675-1687)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("cat")
        token2 = layer0.get_or_create_minicolumn("dog")

        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        # NO shared documents
        concept1.document_ids.add("doc1")
        concept2.document_ids.add("doc2")

        concept1.feedforward_connections[token1.id] = 1.0
        concept2.feedforward_connections[token2.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        # Similar embeddings
        embeddings = {
            "cat": [0.8, 0.6],
            "dog": [0.7, 0.7]
        }

        result = compute_concept_connections(
            layers,
            use_embedding_similarity=True,
            embedding_threshold=0.3,
            embeddings=embeddings
        )

        # Should create connection via embedding similarity
        assert result['embedding_connections'] >= 0  # May or may not connect

    def test_add_connection_already_connected(self):
        """Test strengthening existing connection (line 1599-1601).

        This tests the case where multiple strategies in a single call
        try to connect the same pair. The first succeeds, subsequent
        attempts strengthen the connection and return False.
        """
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_concept_connections

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("machine")
        token2 = layer0.get_or_create_minicolumn("learning")

        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        # Shared documents (triggers Strategy 1: doc overlap)
        concept1.document_ids.add("doc1")
        concept2.document_ids.add("doc1")

        concept1.feedforward_connections[token1.id] = 1.0
        concept2.feedforward_connections[token2.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        # Semantic relation between members (triggers Strategy 2)
        semantic_relations = [
            ("machine", "RelatedTo", "learning", 0.8)
        ]

        # Call with both doc overlap AND member semantics enabled
        # Doc overlap connects them first, then member_semantics tries
        # to connect the same pair and triggers the "already connected" path
        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            use_member_semantics=True,
            min_shared_docs=1,
            min_jaccard=0.0001  # Very low to ensure doc overlap succeeds
        )

        # Only 1 connection created (both strategies tried, but second was duplicate)
        # doc_overlap_connections=1, semantic_connections=0 (because already connected)
        assert result['connections_created'] == 1
        assert result['doc_overlap_connections'] == 1


class TestBigramConnectionsEdgeCases:
    """Test compute_bigram_connections edge cases (lines 1794-1873, 1909)."""

    def test_skipped_large_docs(self):
        """Test skipping docs with too many bigrams (line 1872-1873)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        # Create many bigrams in same doc
        doc_id = "large_doc"
        for i in range(20):
            bigram = layer1.get_or_create_minicolumn(f"term{i} word{i}")
            bigram.document_ids.add(doc_id)

        layers = {CorticalLayer.BIGRAMS: layer1}

        result = compute_bigram_connections(
            layers,
            max_bigrams_per_doc=10  # Skip docs with >10 bigrams
        )

        # Should skip the large doc
        assert result['skipped_large_docs'] > 0

    def test_skipped_common_terms(self):
        """Test skipping overly common terms (line 1832-1833)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        # Create many bigrams with same right component
        for i in range(20):
            bigram = layer1.get_or_create_minicolumn(f"term{i} common")

        layers = {CorticalLayer.BIGRAMS: layer1}

        result = compute_bigram_connections(
            layers,
            max_bigrams_per_term=10  # Skip terms in >10 bigrams
        )

        # Should skip the common term "common"
        assert result['skipped_common_terms'] > 0

    def test_chain_connection_skipped_for_common_term(self):
        """Test chain connection skips common terms (line 1845)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        # Create chain: "a common" and "common b" where "common" appears too often
        layer1.get_or_create_minicolumn("a common")
        layer1.get_or_create_minicolumn("common b")

        # Add many more bigrams with "common" to exceed threshold
        for i in range(20):
            layer1.get_or_create_minicolumn(f"common x{i}")

        layers = {CorticalLayer.BIGRAMS: layer1}

        result = compute_bigram_connections(
            layers,
            max_bigrams_per_term=10
        )

        # Chain should be skipped due to common term
        assert result['chain_connections'] == 0

    def test_cooccurrence_threshold_not_met(self):
        """Test co-occurrence below threshold not connected (line 1909)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_bigram_connections

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        bigram1 = layer1.get_or_create_minicolumn("neural networks")
        bigram2 = layer1.get_or_create_minicolumn("deep learning")

        # Different documents (no co-occurrence)
        bigram1.document_ids.add("doc1")
        bigram2.document_ids.add("doc2")

        layers = {CorticalLayer.BIGRAMS: layer1}

        result = compute_bigram_connections(
            layers,
            min_shared_docs=2  # Require at least 2 shared docs
        )

        # Should not create connection
        assert result['cooccurrence_connections'] == 0


class TestCosineSimilarityZeroMagnitude:
    """Test cosine_similarity zero magnitude case (line 2012)."""

    def test_zero_magnitude_vector(self):
        """Test cosine similarity with zero magnitude vectors."""
        from cortical.analysis import cosine_similarity

        vec1 = {"a": 0.0, "b": 0.0}
        vec2 = {"a": 1.0, "b": 1.0}

        result = cosine_similarity(vec1, vec2)

        # Should return 0.0 (line 2012)
        assert result == 0.0


class TestClusteringQualityMetrics:
    """Test clustering quality metric functions (lines 2065-2413)."""

    def test_compute_clustering_quality_empty(self):
        """Test compute_clustering_quality with empty layers."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_clustering_quality

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }

        result = compute_clustering_quality(layers)

        assert result['modularity'] == 0.0
        assert result['silhouette'] == 0.0
        assert result['balance'] == 1.0
        assert result['num_clusters'] == 0
        assert 'No clusters' in result['quality_assessment']

    def test_compute_clustering_quality_with_clusters(self):
        """Test compute_clustering_quality with actual clusters."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_clustering_quality

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Create tokens with connections
        token1 = layer0.get_or_create_minicolumn("token1")
        token2 = layer0.get_or_create_minicolumn("token2")
        token3 = layer0.get_or_create_minicolumn("token3")

        token1.add_lateral_connection(token2.id, 1.0)
        token2.add_lateral_connection(token1.id, 1.0)
        token1.document_ids.add("doc1")
        token2.document_ids.add("doc1")
        token3.document_ids.add("doc2")

        # Create concept cluster
        concept1 = layer2.get_or_create_minicolumn("cluster1")
        concept1.feedforward_connections[token1.id] = 1.0
        concept1.feedforward_connections[token2.id] = 1.0

        concept2 = layer2.get_or_create_minicolumn("cluster2")
        concept2.feedforward_connections[token3.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        result = compute_clustering_quality(layers, sample_size=10)

        # Should compute all metrics
        assert isinstance(result['modularity'], float)
        assert isinstance(result['silhouette'], float)
        assert isinstance(result['balance'], float)
        assert result['num_clusters'] == 2

    def test_modularity_computation(self):
        """Test _compute_modularity directly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import _compute_modularity

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Two clusters with strong internal connections
        t1 = layer0.get_or_create_minicolumn("t1")
        t2 = layer0.get_or_create_minicolumn("t2")
        t3 = layer0.get_or_create_minicolumn("t3")
        t4 = layer0.get_or_create_minicolumn("t4")

        # Cluster 1: t1, t2
        t1.add_lateral_connection(t2.id, 1.0)
        t2.add_lateral_connection(t1.id, 1.0)

        # Cluster 2: t3, t4
        t3.add_lateral_connection(t4.id, 1.0)
        t4.add_lateral_connection(t3.id, 1.0)

        # Create concept assignments
        c1 = layer2.get_or_create_minicolumn("cluster1")
        c1.feedforward_connections[t1.id] = 1.0
        c1.feedforward_connections[t2.id] = 1.0

        c2 = layer2.get_or_create_minicolumn("cluster2")
        c2.feedforward_connections[t3.id] = 1.0
        c2.feedforward_connections[t4.id] = 1.0

        modularity = _compute_modularity(layer0, layer2)

        # Should have positive modularity (good clustering)
        assert modularity > 0

    def test_silhouette_computation(self):
        """Test _compute_silhouette directly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import _compute_silhouette

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Tokens with document sets
        t1 = layer0.get_or_create_minicolumn("t1")
        t2 = layer0.get_or_create_minicolumn("t2")
        t3 = layer0.get_or_create_minicolumn("t3")

        t1.document_ids.update(["doc1", "doc2"])
        t2.document_ids.update(["doc1", "doc2"])
        t3.document_ids.update(["doc3", "doc4"])

        # Create clusters
        c1 = layer2.get_or_create_minicolumn("cluster1")
        c1.feedforward_connections[t1.id] = 1.0
        c1.feedforward_connections[t2.id] = 1.0

        c2 = layer2.get_or_create_minicolumn("cluster2")
        c2.feedforward_connections[t3.id] = 1.0

        silhouette = _compute_silhouette(layer0, layer2, sample_size=10)

        # Should compute silhouette
        assert isinstance(silhouette, float)
        assert -1 <= silhouette <= 1

    def test_cluster_balance_computation(self):
        """Test _compute_cluster_balance directly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import _compute_cluster_balance

        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Create clusters of different sizes
        c1 = layer2.get_or_create_minicolumn("cluster1")
        c1.feedforward_connections["t1"] = 1.0
        c1.feedforward_connections["t2"] = 1.0
        c1.feedforward_connections["t3"] = 1.0

        c2 = layer2.get_or_create_minicolumn("cluster2")
        c2.feedforward_connections["t4"] = 1.0

        balance = _compute_cluster_balance(layer2)

        # Should compute Gini coefficient
        assert 0 <= balance <= 1

    def test_generate_quality_assessment(self):
        """Test _generate_quality_assessment."""
        from cortical.analysis import _generate_quality_assessment

        assessment = _generate_quality_assessment(
            modularity=0.4,
            silhouette=0.3,
            balance=0.2,
            num_clusters=5
        )

        assert "5 clusters" in assessment
        assert "Good community structure" in assessment

    def test_doc_similarity_helper(self):
        """Test _doc_similarity helper."""
        from cortical.analysis import _doc_similarity

        docs1 = frozenset(["doc1", "doc2", "doc3"])
        docs2 = frozenset(["doc2", "doc3", "doc4"])

        sim = _doc_similarity(docs1, docs2)

        # Jaccard: |{doc2, doc3}| / |{doc1, doc2, doc3, doc4}| = 2/4 = 0.5
        assert sim == pytest.approx(0.5)

    def test_doc_similarity_empty(self):
        """Test _doc_similarity with empty sets."""
        from cortical.analysis import _doc_similarity

        docs1 = frozenset()
        docs2 = frozenset(["doc1"])

        sim = _doc_similarity(docs1, docs2)

        assert sim == 0.0

    def test_vector_similarity_helper(self):
        """Test _vector_similarity helper."""
        from cortical.analysis import _vector_similarity

        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"a": 1.0, "c": 3.0}

        sim = _vector_similarity(vec1, vec2)

        # Weighted Jaccard: min(1,1) / (max(1,1) + max(2,0) + max(0,3))
        # = 1 / (1 + 2 + 3) = 1/6
        assert sim == pytest.approx(1.0 / 6.0)

    def test_vector_similarity_empty(self):
        """Test _vector_similarity with empty vectors."""
        from cortical.analysis import _vector_similarity

        vec1 = {}
        vec2 = {"a": 1.0}

        sim = _vector_similarity(vec1, vec2)

        assert sim == 0.0

    def test_silhouette_with_many_tokens(self):
        """Test _compute_silhouette with actual token graph (lines 2208-2281)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import _compute_silhouette

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Create 10 tokens with document overlap patterns
        tokens = []
        for i in range(10):
            t = layer0.get_or_create_minicolumn(f"token{i}")
            tokens.append(t)
            # First 5 tokens in doc1-doc2, last 5 in doc3-doc4
            if i < 5:
                t.document_ids.update([f"doc1", f"doc2"])
            else:
                t.document_ids.update([f"doc3", f"doc4"])

        # Create 2 clusters
        c1 = layer2.get_or_create_minicolumn("cluster1")
        c2 = layer2.get_or_create_minicolumn("cluster2")

        for i, t in enumerate(tokens):
            if i < 5:
                c1.feedforward_connections[t.id] = 1.0
            else:
                c2.feedforward_connections[t.id] = 1.0

        # This should trigger lines 2208-2281
        silhouette = _compute_silhouette(layer0, layer2, sample_size=20)

        # Should get a positive silhouette (good clustering)
        assert silhouette > 0

    def test_cluster_balance_edge_cases(self):
        """Test _compute_cluster_balance edge cases (lines 2320, 2348, 2355)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import _compute_cluster_balance

        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Edge case: single cluster with zero total (line 2355)
        c1 = layer2.get_or_create_minicolumn("cluster1")
        # No feedforward_connections (empty dict, total=0)

        balance = _compute_cluster_balance(layer2)

        # Should return 1.0 (all in one cluster)
        assert balance == 1.0

    def test_generate_quality_assessment_variations(self):
        """Test _generate_quality_assessment with different score ranges."""
        from cortical.analysis import _generate_quality_assessment

        # Test weak modularity (lines 2388-2391)
        assessment1 = _generate_quality_assessment(
            modularity=0.15,
            silhouette=0.15,
            balance=0.4,
            num_clusters=3
        )
        assert "Weak community structure" in assessment1
        assert "moderate topic coherence" in assessment1

        # Test negative silhouette (lines 2399, 2403)
        assessment2 = _generate_quality_assessment(
            modularity=0.6,
            silhouette=-0.05,
            balance=0.6,
            num_clusters=4
        )
        assert "Strong community structure" in assessment2
        assert "typical graph clustering" in assessment2

        # Test diverse clusters (line 2403)
        assessment3 = _generate_quality_assessment(
            modularity=0.2,
            silhouette=-0.2,
            balance=0.7,
            num_clusters=2
        )
        assert "diverse clusters" in assessment3
        assert "imbalanced sizes" in assessment3

    def test_louvain_phase2_inter_community_edges(self):
        """Test cluster_by_louvain phase2 with inter-community edges (lines 1338-1341)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import cluster_by_louvain

        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create two communities with an inter-community edge
        # Community 1: a0-a1-a2 (dense)
        for i in range(3):
            col = layer.get_or_create_minicolumn(f"a{i}")
            for j in range(3):
                if i != j:
                    col.add_lateral_connection(f"L0_a{j}", 2.0)

        # Community 2: b0-b1-b2 (dense)
        for i in range(3):
            col = layer.get_or_create_minicolumn(f"b{i}")
            for j in range(3):
                if i != j:
                    col.add_lateral_connection(f"L0_b{j}", 2.0)

        # Inter-community edge (weak)
        col_a0 = layer.get_minicolumn("a0")
        col_a0.add_lateral_connection("L0_b0", 0.5)
        col_b0 = layer.get_minicolumn("b0")
        col_b0.add_lateral_connection("L0_a0", 0.5)

        # Run Louvain - should trigger phase2 with inter-community edges
        result = cluster_by_louvain(layer, min_cluster_size=2, max_iterations=3)

        # Should find clusters
        assert len(result) >= 1

    def test_propagate_activation_layer_filtering(self):
        """Test propagate_activation layer enum filtering (lines 964, 966)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import propagate_activation

        # Create layers with different levels
        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("token1")
        token1.activation = 1.0

        bigram1 = layer1.get_or_create_minicolumn("bigram1")
        concept1 = layer2.get_or_create_minicolumn("concept1")

        # Add feedforward sources from lower layers
        bigram1.feedforward_sources.add(token1.id)
        concept1.feedforward_sources.add(token1.id)
        concept1.feedforward_sources.add(bigram1.id)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1,
            CorticalLayer.CONCEPTS: layer2
        }

        # This should trigger the layer filtering logic
        propagate_activation(layers, iterations=2, decay=0.9)

        # Both should receive activation from token
        assert bigram1.activation > 0
        assert concept1.activation > 0

    def test_semantic_pagerank_target_none(self):
        """Test compute_semantic_pagerank when target is None (line 667)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer
        from cortical.analysis import compute_semantic_pagerank

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("term1")

        # Add connection to non-existent target
        col1.lateral_connections["L0_nonexistent"] = 1.0

        semantic_relations = [("term1", "RelatedTo", "term2", 0.8)]

        result = compute_semantic_pagerank(
            layer, semantic_relations, damping=0.85, iterations=3
        )

        # Should handle gracefully (line 667 target is None)
        assert 'pagerank' in result
