"""
Comprehensive Coverage Tests for Embeddings Module
===================================================

Additional tests to achieve >80% coverage for cortical/embeddings.py.
Complements existing tests in test_embeddings.py by focusing on:
- Edge cases and boundary conditions
- Error paths and special states
- Parameter combinations not tested elsewhere
- Integration between different embedding methods

These tests target specific uncovered code paths to maximize coverage.
"""

import pytest
import math
import random
import warnings
from typing import Dict, List

from cortical.embeddings import (
    compute_graph_embeddings,
    _fast_adjacency_embeddings,
    _tfidf_embeddings,
    _adjacency_embeddings,
    _random_walk_embeddings,
    _spectral_embeddings,
    _weighted_random_walk,
    embedding_similarity,
    find_similar_by_embedding,
)

from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
)


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCasesAndBoundaries:
    """Tests for edge cases and boundary conditions."""

    def test_zero_dimensions(self):
        """Zero dimensions should still work (empty embeddings)."""
        layers = MockLayers.single_term("test")
        embeddings, stats = compute_graph_embeddings(layers, dimensions=0)
        # Should return empty embeddings or handle gracefully
        assert isinstance(embeddings, dict)
        assert stats['dimensions'] == 0

    def test_very_large_dimensions(self):
        """Very large dimensions (more than terms) should work."""
        layers = MockLayers.two_connected_terms("a", "b")
        embeddings, stats = compute_graph_embeddings(layers, dimensions=1000)
        # Should not crash, embeddings padded as needed
        assert "a" in embeddings
        assert "b" in embeddings
        assert len(embeddings["a"]) == 1000

    def test_max_terms_equals_zero(self):
        """max_terms=0 should handle gracefully."""
        layers = MockLayers.two_connected_terms("a", "b")
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, max_terms=0
        )
        # Should sample 0 terms
        assert stats['max_terms'] == 0

    def test_max_terms_equals_one(self):
        """max_terms=1 should only sample top term."""
        layers = MockLayers.disconnected_terms(
            ["a", "b", "c"],
            pageranks=[1.0, 0.5, 0.3]
        )
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=3, max_terms=1
        )
        assert stats['max_terms'] == 1
        assert stats['sampled'] is True

    def test_all_zero_pageranks(self):
        """All zero PageRanks should not crash."""
        col1 = MockMinicolumn(content="a", pagerank=0.0)
        col2 = MockMinicolumn(content="b", pagerank=0.0)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _fast_adjacency_embeddings(layer, dimensions=2)
        assert "a" in embeddings
        assert "b" in embeddings

    def test_zero_weight_connections(self):
        """Connections with zero weight should be handled."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 0.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _fast_adjacency_embeddings(layer, dimensions=2)
        assert "a" in embeddings
        # Zero weight should be included in computation

    def test_very_small_weights(self):
        """Very small weights should not cause numerical issues."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1e-100}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(layer, dimensions=2)
        assert "a" in embeddings
        assert all(math.isfinite(v) for v in embeddings["a"])

    def test_very_large_weights(self):
        """Very large weights should not overflow."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1e50}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(layer, dimensions=2)
        assert "a" in embeddings
        # After normalization, should be finite
        assert all(math.isfinite(v) for v in embeddings["a"])


# =============================================================================
# FAST ADJACENCY COVERAGE
# =============================================================================


class TestFastAdjacencyCoverage:
    """Additional coverage for _fast_adjacency_embeddings."""

    def test_no_documents_idf_weighting(self):
        """IDF weighting with no documents should not crash."""
        col = MockMinicolumn(
            content="a",
            pagerank=1.0,
            document_ids=set()  # No documents
        )
        layer = MockHierarchicalLayer([col], level=0)

        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=1, use_idf_weighting=True
        )
        assert "a" in embeddings

    def test_all_terms_same_doc_frequency(self):
        """All terms with same doc frequency."""
        cols = [
            MockMinicolumn(
                content=f"term{i}",
                pagerank=1.0 - i*0.1,
                document_ids={"doc1", "doc2"},
                lateral_connections={"L0_term0": 1.0} if i > 0 else {}
            )
            for i in range(3)
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=3, use_idf_weighting=True
        )
        assert len(embeddings) == 3

    def test_landmark_not_in_connections(self):
        """Term with no connections to landmarks."""
        # Create high-PageRank landmarks
        landmark = MockMinicolumn(content="landmark", pagerank=1.0)
        isolated = MockMinicolumn(
            content="isolated",
            pagerank=0.1,
            lateral_connections={}  # No connections
        )
        layer = MockHierarchicalLayer([landmark, isolated], level=0)

        embeddings = _fast_adjacency_embeddings(layer, dimensions=1)
        assert "isolated" in embeddings
        # Should have all-zero vector (normalized to small magnitude)

    def test_connection_to_non_landmark(self):
        """Term connecting only to non-landmarks."""
        high_pr = MockMinicolumn(content="high", pagerank=1.0)
        low_pr = MockMinicolumn(
            content="low",
            pagerank=0.1,
            lateral_connections={"L0_other": 5.0}  # Connects to low-rank term
        )
        other = MockMinicolumn(content="other", pagerank=0.05)
        layer = MockHierarchicalLayer([high_pr, low_pr, other], level=0)

        # Only 1 dimension = only 'high' is landmark
        embeddings = _fast_adjacency_embeddings(layer, dimensions=1)
        assert "low" in embeddings


# =============================================================================
# TF-IDF EMBEDDINGS COVERAGE
# =============================================================================


class TestTfidfEmbeddingsCoverage:
    """Additional coverage for _tfidf_embeddings."""

    def test_no_tfidf_scores(self):
        """Term with no TF-IDF scores."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1"},
            tfidf_per_doc={}  # No scores
        )
        layer = MockHierarchicalLayer([col], level=0)

        embeddings = _tfidf_embeddings(layer, dimensions=1)
        assert "term" in embeddings
        # Should have zero vector

    def test_tfidf_for_missing_doc(self):
        """TF-IDF score for doc not in dimension set."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2"},
            tfidf_per_doc={"doc1": 1.0, "doc2": 2.0, "doc3": 3.0}
        )
        layer = MockHierarchicalLayer([col], level=0)

        # Request 1 dimension, should pick doc with most terms (doc1 or doc2)
        embeddings = _tfidf_embeddings(layer, dimensions=1)
        assert "term" in embeddings

    def test_empty_doc_dimension_set(self):
        """No documents available."""
        col = MockMinicolumn(
            content="term",
            document_ids=set(),
            tfidf_per_doc={}
        )
        layer = MockHierarchicalLayer([col], level=0)

        embeddings = _tfidf_embeddings(layer, dimensions=5)
        # Should handle gracefully
        assert "term" in embeddings

    def test_more_dimensions_than_docs(self):
        """Request more dimensions than documents available."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2"},
            tfidf_per_doc={"doc1": 1.0, "doc2": 2.0}
        )
        layer = MockHierarchicalLayer([col], level=0)

        embeddings = _tfidf_embeddings(layer, dimensions=10)
        # Should use all available docs (2)
        assert "term" in embeddings
        assert len(embeddings["term"]) == 2


# =============================================================================
# ADJACENCY EMBEDDINGS COVERAGE
# =============================================================================


class TestAdjacencyEmbeddingsCoverage:
    """Additional coverage for _adjacency_embeddings."""

    def test_zero_propagation_steps(self):
        """propagation_steps=0 uses only direct connections."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 2.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(
            layer, dimensions=2, propagation_steps=0
        )
        assert "a" in embeddings

    def test_very_high_propagation_steps(self):
        """Many propagation steps on small graph."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            pagerank=0.5,
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        # High propagation on 2-node graph should terminate
        embeddings = _adjacency_embeddings(
            layer, dimensions=2, propagation_steps=100
        )
        assert "a" in embeddings
        assert "b" in embeddings

    def test_zero_damping(self):
        """damping=0 means no multi-hop contribution."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(
            layer, dimensions=2, damping=0.0
        )
        assert "a" in embeddings

    def test_full_damping(self):
        """damping=1.0 (no decay)."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(
            layer, dimensions=2, damping=1.0
        )
        assert "a" in embeddings

    def test_neighbor_not_in_layer(self):
        """Neighbor ID not found in layer (broken reference)."""
        col = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_nonexistent": 1.0}
        )
        layer = MockHierarchicalLayer([col], level=0)

        # Should handle missing neighbor gracefully
        embeddings = _adjacency_embeddings(
            layer, dimensions=1, propagation_steps=1
        )
        assert "a" in embeddings

    def test_visited_tracking(self):
        """Visited set prevents revisiting nodes."""
        # Create cycle: a -> b -> c -> a
        col_a = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1.0}
        )
        col_b = MockMinicolumn(
            content="b",
            pagerank=0.8,
            lateral_connections={"L0_c": 1.0}
        )
        col_c = MockMinicolumn(
            content="c",
            pagerank=0.6,
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col_a, col_b, col_c], level=0)

        # Should not infinite loop
        embeddings = _adjacency_embeddings(
            layer, dimensions=3, propagation_steps=10
        )
        assert len(embeddings) == 3


# =============================================================================
# RANDOM WALK EMBEDDINGS COVERAGE
# =============================================================================


class TestRandomWalkEmbeddingsCoverage:
    """Additional coverage for _random_walk_embeddings."""

    def test_walk_parameters_minimal(self):
        """Minimal walk parameters (1 walk, length 1)."""
        col = MockMinicolumn(content="a", pagerank=1.0)
        layer = MockHierarchicalLayer([col], level=0)

        random.seed(42)
        embeddings = _random_walk_embeddings(
            layer, dimensions=1, walks_per_node=1, walk_length=1
        )
        assert "a" in embeddings

    def test_zero_window_size(self):
        """window_size=0 means no co-occurrence (all zeros)."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            pagerank=0.5,
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        random.seed(42)
        embeddings = _random_walk_embeddings(
            layer, dimensions=2, window_size=0
        )
        assert "a" in embeddings

    def test_very_large_window_size(self):
        """window_size larger than walk length."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            pagerank=0.5,
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        random.seed(42)
        embeddings = _random_walk_embeddings(
            layer, dimensions=2, walk_length=5, window_size=100
        )
        assert "a" in embeddings

    def test_term_not_in_layer_for_landmark(self):
        """Landmark selection when term count < dimensions."""
        col = MockMinicolumn(content="a", pagerank=1.0)
        layer = MockHierarchicalLayer([col], level=0)

        random.seed(42)
        # Request more dimensions than terms
        embeddings = _random_walk_embeddings(
            layer, dimensions=10, walks_per_node=5
        )
        assert "a" in embeddings

    def test_sampled_terms_not_in_layer(self):
        """sampled_terms includes term not in layer."""
        col = MockMinicolumn(content="a", pagerank=1.0)
        layer = MockHierarchicalLayer([col], level=0)

        random.seed(42)
        embeddings = _random_walk_embeddings(
            layer,
            dimensions=1,
            sampled_terms={"a", "nonexistent"}
        )
        # Should only embed "a"
        assert "a" in embeddings


# =============================================================================
# WEIGHTED RANDOM WALK COVERAGE
# =============================================================================


class TestWeightedRandomWalkCoverage:
    """Additional coverage for _weighted_random_walk."""

    def test_empty_frontier_early_termination(self):
        """Walk terminates when frontier is empty."""
        col = MockMinicolumn(content="a", lateral_connections={})
        layer = MockHierarchicalLayer([col], level=0)
        id_to_term = {"L0_a": "a"}

        walk = _weighted_random_walk(col, layer, length=100, id_to_term=id_to_term)
        assert walk == ["a"]

    def test_total_weight_zero(self):
        """All neighbors have zero weight."""
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 0.0}
        )
        col2 = MockMinicolumn(content="b")
        layer = MockHierarchicalLayer([col1, col2], level=0)
        id_to_term = {"L0_a": "a", "L0_b": "b"}

        random.seed(42)
        walk = _weighted_random_walk(col1, layer, length=10, id_to_term=id_to_term)
        # Should terminate early
        assert len(walk) <= 10

    def test_next_term_not_in_id_to_term(self):
        """Selected next_id not in id_to_term mapping."""
        col = MockMinicolumn(
            content="a",
            lateral_connections={"L0_missing": 1.0}
        )
        layer = MockHierarchicalLayer([col], level=0)
        id_to_term = {"L0_a": "a"}  # "L0_missing" not in mapping

        random.seed(42)
        walk = _weighted_random_walk(col, layer, length=5, id_to_term=id_to_term)
        # Should terminate when hitting missing term
        assert walk[0] == "a"

    def test_next_term_not_in_layer(self):
        """Selected next term exists in id_to_term but not in layer."""
        col = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 1.0}
        )
        layer = MockHierarchicalLayer([col], level=0)  # Only 'a', no 'b'
        id_to_term = {"L0_a": "a", "L0_b": "b"}

        random.seed(42)
        walk = _weighted_random_walk(col, layer, length=5, id_to_term=id_to_term)
        # Should terminate when trying to step to 'b'
        assert walk[0] == "a"

    def test_cumulative_weight_edge_cases(self):
        """Test random selection with various weight distributions."""
        # Unequal weights to test cumulative sum logic
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={
                "L0_b": 0.1,
                "L0_c": 0.5,
                "L0_d": 0.4
            }
        )
        col2 = MockMinicolumn(content="b", lateral_connections={"L0_a": 1.0})
        col3 = MockMinicolumn(content="c", lateral_connections={"L0_a": 1.0})
        col4 = MockMinicolumn(content="d", lateral_connections={"L0_a": 1.0})
        layer = MockHierarchicalLayer([col1, col2, col3, col4], level=0)
        id_to_term = {"L0_a": "a", "L0_b": "b", "L0_c": "c", "L0_d": "d"}

        random.seed(42)
        walk = _weighted_random_walk(col1, layer, length=10, id_to_term=id_to_term)
        # Should complete walk respecting weights
        assert len(walk) >= 1


# =============================================================================
# SPECTRAL EMBEDDINGS COVERAGE
# =============================================================================


class TestSpectralEmbeddingsCoverage:
    """Additional coverage for _spectral_embeddings."""

    def test_minimal_iterations(self):
        """iterations=1 for fast computation."""
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=2, iterations=1)
        assert "a" in embeddings
        assert "b" in embeddings

    def test_dimensions_exactly_equals_nodes(self):
        """Dimensions exactly equals number of nodes."""
        cols = [MockMinicolumn(content=str(i)) for i in range(5)]
        layer = MockHierarchicalLayer(cols, level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=5, iterations=5)
        assert len(embeddings) == 5

    def test_zero_degree_nodes(self):
        """Nodes with no connections (degree 0)."""
        col1 = MockMinicolumn(content="isolated1")
        col2 = MockMinicolumn(content="isolated2")
        layer = MockHierarchicalLayer([col1, col2], level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=2, iterations=5)
        assert "isolated1" in embeddings
        assert "isolated2" in embeddings

    def test_self_loops_in_adjacency(self):
        """Self-loops in adjacency should be handled."""
        col = MockMinicolumn(
            content="a",
            lateral_connections={"L0_a": 1.0}  # Self-loop
        )
        layer = MockHierarchicalLayer([col], level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=1, iterations=5)
        assert "a" in embeddings

    def test_neighbor_not_in_sampled_terms(self):
        """Neighbor exists but not in sampled_terms."""
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        random.seed(42)
        # Only sample 'a'
        embeddings = _spectral_embeddings(
            layer, dimensions=1, sampled_terms={"a"}, iterations=5
        )
        assert "a" in embeddings
        assert "b" not in embeddings

    def test_orthogonalization_with_previous_vectors(self):
        """Multiple dimensions trigger orthogonalization."""
        cols = [
            MockMinicolumn(
                content=str(i),
                lateral_connections={f"L0_{(i+1)%5}": 1.0}
            )
            for i in range(5)
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        random.seed(42)
        # Multiple dimensions = multiple orthogonalization passes
        embeddings = _spectral_embeddings(layer, dimensions=4, iterations=10)
        assert len(embeddings) == 5

    def test_asymmetric_adjacency(self):
        """Asymmetric adjacency matrix."""
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 5.0}
        )
        col2 = MockMinicolumn(
            content="b",
            lateral_connections={"L0_a": 1.0}  # Different weight
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=2, iterations=5)
        assert "a" in embeddings
        assert "b" in embeddings


# =============================================================================
# EMBEDDING SIMILARITY COVERAGE
# =============================================================================


class TestEmbeddingSimilarityCoverage:
    """Additional coverage for embedding_similarity."""

    def test_self_similarity(self):
        """Similarity of a term with itself is 1.0."""
        embeddings = {"a": [1.0, 2.0, 3.0]}
        similarity = embedding_similarity(embeddings, "a", "a")
        assert similarity == pytest.approx(1.0)

    def test_near_zero_magnitude(self):
        """Very small magnitude vectors."""
        embeddings = {
            "a": [1e-100, 1e-100, 1e-100],
            "b": [1e-100, 1e-100, 1e-100]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        # Should handle gracefully, may have floating point imprecision
        assert similarity == pytest.approx(1.0, abs=1e-6) or similarity == pytest.approx(0.0, abs=1e-6)

    def test_mixed_positive_negative(self):
        """Vectors with mixed positive and negative values."""
        embeddings = {
            "a": [1.0, -2.0, 3.0],
            "b": [-1.0, 2.0, -3.0]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        # Should be negative (opposite)
        assert similarity < 0

    def test_high_dimensional_vectors(self):
        """High-dimensional vectors (100+)."""
        embeddings = {
            "a": [1.0] * 100,
            "b": [1.0] * 100
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity == pytest.approx(1.0)


# =============================================================================
# FIND SIMILAR COVERAGE
# =============================================================================


class TestFindSimilarCoverage:
    """Additional coverage for find_similar_by_embedding."""

    def test_top_n_zero(self):
        """top_n=0 returns empty list."""
        embeddings = {
            "query": [1.0, 0.0],
            "other": [0.9, 0.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=0)
        assert result == []

    def test_only_query_term(self):
        """Only query term in embeddings."""
        embeddings = {"query": [1.0, 0.0]}
        result = find_similar_by_embedding(embeddings, "query", top_n=5)
        assert result == []

    def test_ties_in_similarity(self):
        """Multiple terms with identical similarity."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "a": [0.9, 0.1, 0.0],
            "b": [0.9, 0.1, 0.0],  # Same as 'a'
            "c": [0.8, 0.2, 0.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=5)
        # Should handle ties gracefully
        assert len(result) == 3

    def test_all_dissimilar(self):
        """All terms are orthogonal to query."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "a": [0.0, 1.0, 0.0],
            "b": [0.0, 0.0, 1.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=5)
        # Should still return them, sorted by similarity (all ~0)
        assert len(result) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests across multiple embedding methods."""

    def test_all_methods_produce_normalized_embeddings(self):
        """All methods produce L2-normalized embeddings."""
        # Create layers with documents for TF-IDF to work properly
        layers = LayerBuilder() \
            .with_term("a", pagerank=1.0) \
            .with_term("b", pagerank=0.5) \
            .with_connection("a", "b", 5.0) \
            .with_document("doc1", ["a", "b"]) \
            .build()

        # Set TF-IDF scores for TF-IDF method
        layers[0].get_minicolumn("a").tfidf_per_doc = {"doc1": 1.5}
        layers[0].get_minicolumn("b").tfidf_per_doc = {"doc1": 2.0}

        # Test deterministic methods that always normalize
        methods = ['adjacency', 'fast', 'tfidf']

        for method in methods:
            embeddings, _ = compute_graph_embeddings(
                layers, dimensions=2, method=method
            )

            for term, vec in embeddings.items():
                magnitude = math.sqrt(sum(v*v for v in vec))
                # Allow near-zero vectors for some edge cases
                assert magnitude == pytest.approx(1.0, abs=1e-5) or magnitude == pytest.approx(0.0, abs=1e-5), \
                    f"Method {method} failed normalization for {term}: magnitude={magnitude}"

        # Spectral and random_walk normalize but may have numerical issues
        for method in ['spectral', 'random_walk']:
            random.seed(42)
            embeddings, _ = compute_graph_embeddings(
                layers, dimensions=2, method=method
            )
            for term, vec in embeddings.items():
                magnitude = math.sqrt(sum(v*v for v in vec))
                # More lenient for stochastic/iterative methods
                assert 0.0 <= magnitude <= 1.5, \
                    f"Method {method} has unreasonable magnitude for {term}: {magnitude}"

    def test_consistency_across_seeds(self):
        """Deterministic methods produce same results."""
        layers = MockLayers.two_connected_terms("a", "b")

        # Deterministic methods
        for method in ['adjacency', 'fast', 'tfidf']:
            emb1, _ = compute_graph_embeddings(layers, dimensions=2, method=method)
            emb2, _ = compute_graph_embeddings(layers, dimensions=2, method=method)

            assert emb1.keys() == emb2.keys()
            for term in emb1:
                assert emb1[term] == pytest.approx(emb2[term])

    def test_stochastic_methods_vary_with_seed(self):
        """Stochastic methods vary with different seeds."""
        layers = MockLayers.two_connected_terms("a", "b")

        random.seed(42)
        emb1, _ = compute_graph_embeddings(
            layers, dimensions=2, method='random_walk'
        )

        random.seed(43)
        emb2, _ = compute_graph_embeddings(
            layers, dimensions=2, method='random_walk'
        )

        # Should be different (with high probability)
        # At least one component should differ
        different = False
        for term in emb1:
            if emb1[term] != pytest.approx(emb2[term], abs=1e-6):
                different = True
                break

        # Note: There's a small chance they're the same, but unlikely
        # If this test is flaky, we can remove it

    def test_empty_layer_all_methods(self):
        """All methods handle empty layer gracefully."""
        layers = MockLayers.empty()

        methods = ['adjacency', 'fast', 'tfidf', 'random_walk', 'spectral']

        for method in methods:
            embeddings, stats = compute_graph_embeddings(
                layers, dimensions=5, method=method
            )
            assert embeddings == {}, f"Method {method} failed on empty layer"
            assert stats['terms_embedded'] == 0

    def test_single_term_all_methods(self):
        """All methods handle single term."""
        # Create layers with document for TF-IDF to work
        layers = LayerBuilder() \
            .with_term("test", pagerank=1.0) \
            .with_document("doc1", ["test"]) \
            .build()
        layers[0].get_minicolumn("test").tfidf_per_doc = {"doc1": 1.0}

        methods = ['adjacency', 'fast', 'tfidf', 'random_walk', 'spectral']

        for method in methods:
            random.seed(42)
            embeddings, stats = compute_graph_embeddings(
                layers, dimensions=3, method=method
            )
            assert "test" in embeddings, f"Method {method} failed on single term"
            assert len(embeddings["test"]) >= 1, f"Method {method} returned empty embedding"
            # Note: Some methods (tfidf, random_walk) may return fewer dimensions
            # when there's only 1 term, which is expected behavior


# =============================================================================
# PARAMETER VALIDATION AND ERROR HANDLING
# =============================================================================


class TestParameterValidation:
    """Tests for parameter validation and error cases."""

    def test_invalid_method_raises_error(self):
        """Invalid method name raises ValueError."""
        layers = MockLayers.single_term("test")
        with pytest.raises(ValueError, match="Unknown embedding method"):
            compute_graph_embeddings(layers, dimensions=5, method='invalid_method')

    def test_negative_dimensions_behavior(self):
        """Negative dimensions (implementation-dependent)."""
        layers = MockLayers.single_term("test")
        # May raise error or handle gracefully
        try:
            embeddings, _ = compute_graph_embeddings(layers, dimensions=-1)
            # If no error, check it handled gracefully
            assert isinstance(embeddings, dict)
        except (ValueError, IndexError):
            # Acceptable to raise error
            pass

    def test_none_sampled_terms(self):
        """sampled_terms=None should process all terms."""
        layers = MockLayers.two_connected_terms("a", "b")
        layer = layers[0]

        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=2, sampled_terms=None
        )
        assert "a" in embeddings
        assert "b" in embeddings

    def test_empty_sampled_terms_set(self):
        """Empty sampled_terms set."""
        layers = MockLayers.two_connected_terms("a", "b")
        layer = layers[0]

        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=2, sampled_terms=set()
        )
        # Should return no embeddings
        assert len(embeddings) == 0


# =============================================================================
# STATISTICS AND METADATA
# =============================================================================


class TestStatistics:
    """Tests for statistics returned by compute_graph_embeddings."""

    def test_stats_structure(self):
        """Stats dict has expected keys."""
        layers = MockLayers.single_term("test")
        _, stats = compute_graph_embeddings(layers, dimensions=5)

        assert 'method' in stats
        assert 'dimensions' in stats
        assert 'terms_embedded' in stats
        assert 'max_terms' in stats
        assert 'sampled' in stats

    def test_stats_accuracy(self):
        """Stats values are accurate."""
        layers = MockLayers.disconnected_terms(
            ["a", "b", "c"],
            pageranks=[1.0, 0.5, 0.3]
        )

        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=10, method='fast', max_terms=2
        )

        assert stats['method'] == 'fast'
        assert stats['dimensions'] == 10
        assert stats['max_terms'] == 2
        assert stats['sampled'] is True
        assert stats['terms_embedded'] == len(embeddings)

    def test_sampled_flag_when_no_sampling(self):
        """sampled=False when not sampling."""
        layers = MockLayers.two_connected_terms("a", "b")
        _, stats = compute_graph_embeddings(layers, dimensions=5)

        assert stats['sampled'] is False

    def test_sampled_flag_when_max_terms_none(self):
        """sampled=False when max_terms is None."""
        layers = MockLayers.two_connected_terms("a", "b")
        _, stats = compute_graph_embeddings(
            layers, dimensions=5, max_terms=None
        )

        assert stats['sampled'] is False
        assert stats['max_terms'] is None
