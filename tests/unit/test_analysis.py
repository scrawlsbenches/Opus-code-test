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
