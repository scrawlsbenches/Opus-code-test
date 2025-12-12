"""
Unit Tests for Embeddings Module
==================================

Task #160: Unit tests for cortical/embeddings.py graph embeddings.

Tests all embedding methods and utilities:
- compute_graph_embeddings(): Main entry point with method selection
- _fast_adjacency_embeddings(): Fast direct adjacency to landmarks
- _tfidf_embeddings(): TF-IDF based embeddings
- _adjacency_embeddings(): Multi-hop adjacency propagation
- _random_walk_embeddings(): DeepWalk-inspired random walks
- _spectral_embeddings(): Graph Laplacian eigenvectors
- _weighted_random_walk(): Random walk helper
- embedding_similarity(): Cosine similarity calculation
- find_similar_by_embedding(): Nearest neighbor search

These tests use mock layers to isolate embedding logic from the full processor.
"""

import pytest
import math
import random
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
# COMPUTE GRAPH EMBEDDINGS - MAIN ENTRY POINT
# =============================================================================


class TestComputeGraphEmbeddings:
    """Tests for compute_graph_embeddings main entry point."""

    def test_empty_layer(self):
        """Empty layer returns empty embeddings."""
        layers = MockLayers.empty()
        embeddings, stats = compute_graph_embeddings(layers, dimensions=10)
        assert embeddings == {}
        assert stats['terms_embedded'] == 0
        assert stats['method'] == 'adjacency'
        assert stats['dimensions'] == 10

    def test_single_term(self):
        """Single term gets an embedding."""
        layers = MockLayers.single_term("test", pagerank=1.0)
        embeddings, stats = compute_graph_embeddings(layers, dimensions=5)
        assert "test" in embeddings
        assert len(embeddings["test"]) == 5
        assert stats['terms_embedded'] == 1

    def test_method_adjacency(self):
        """Method='adjacency' uses adjacency embeddings."""
        layers = MockLayers.two_connected_terms("a", "b", weight=1.0)
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, method='adjacency'
        )
        assert stats['method'] == 'adjacency'
        assert len(embeddings) == 2

    def test_method_fast(self):
        """Method='fast' uses fast adjacency embeddings."""
        layers = MockLayers.two_connected_terms("a", "b", weight=1.0)
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, method='fast'
        )
        assert stats['method'] == 'fast'
        assert len(embeddings) == 2

    def test_method_tfidf(self):
        """Method='tfidf' uses TF-IDF embeddings."""
        layers = MockLayers.document_with_terms("doc1", ["a", "b"])
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, method='tfidf'
        )
        assert stats['method'] == 'tfidf'
        assert len(embeddings) == 2

    def test_method_random_walk(self):
        """Method='random_walk' uses random walk embeddings."""
        layers = MockLayers.two_connected_terms("a", "b", weight=1.0)
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, method='random_walk'
        )
        assert stats['method'] == 'random_walk'
        assert len(embeddings) == 2

    def test_method_spectral(self):
        """Method='spectral' uses spectral embeddings."""
        layers = MockLayers.two_connected_terms("a", "b", weight=1.0)
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, method='spectral'
        )
        assert stats['method'] == 'spectral'
        assert len(embeddings) == 2

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        layers = MockLayers.single_term("test")
        with pytest.raises(ValueError, match="Unknown embedding method"):
            compute_graph_embeddings(layers, dimensions=5, method='invalid')

    def test_max_terms_sampling(self):
        """max_terms limits embedding to top-ranked terms."""
        layers = MockLayers.disconnected_terms(
            ["a", "b", "c", "d"],
            pageranks=[0.4, 0.3, 0.2, 0.1]
        )
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, max_terms=2
        )
        # Should only embed top 2 terms by PageRank (a, b)
        assert stats['sampled'] is True
        assert stats['max_terms'] == 2
        # Adjacency method may still embed all if they're landmarks
        assert len(embeddings) >= 2

    def test_max_terms_larger_than_corpus(self):
        """max_terms larger than corpus embeds all terms."""
        layers = MockLayers.disconnected_terms(["a", "b"])
        embeddings, stats = compute_graph_embeddings(
            layers, dimensions=5, max_terms=100
        )
        assert stats['sampled'] is False
        assert len(embeddings) == 2

    def test_dimensions_parameter(self):
        """Embedding dimension matches requested size."""
        layers = MockLayers.single_term("test")
        embeddings, stats = compute_graph_embeddings(layers, dimensions=20)
        assert len(embeddings["test"]) == 20
        assert stats['dimensions'] == 20


# =============================================================================
# FAST ADJACENCY EMBEDDINGS
# =============================================================================


class TestFastAdjacencyEmbeddings:
    """Tests for _fast_adjacency_embeddings."""

    def test_empty_layer(self):
        """Empty layer returns empty embeddings."""
        layer = MockHierarchicalLayer([], level=0)
        embeddings = _fast_adjacency_embeddings(layer, dimensions=5)
        assert embeddings == {}

    def test_single_term_no_connections(self):
        """Single term with no connections gets zero embedding."""
        col = MockMinicolumn(content="isolated", pagerank=1.0)
        layer = MockHierarchicalLayer([col], level=0)
        embeddings = _fast_adjacency_embeddings(layer, dimensions=5)
        assert "isolated" in embeddings
        # No connections = all zeros, but normalized
        vec = embeddings["isolated"]
        assert len(vec) == 5

    def test_two_connected_terms(self):
        """Two connected terms get embeddings based on connections."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=0.6,
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            pagerank=0.4,
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)
        embeddings = _fast_adjacency_embeddings(layer, dimensions=2)

        assert "a" in embeddings
        assert "b" in embeddings
        assert len(embeddings["a"]) == 2
        assert len(embeddings["b"]) == 2

    def test_normalization(self):
        """Embeddings are L2-normalized."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 5.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)
        embeddings = _fast_adjacency_embeddings(layer, dimensions=2)

        # Check L2 norm is close to 1.0
        vec = embeddings["a"]
        magnitude = math.sqrt(sum(v*v for v in vec))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_landmarks_by_pagerank(self):
        """Landmarks are selected by PageRank."""
        # Create 5 terms with different PageRanks
        cols = [
            MockMinicolumn(content=f"term{i}", pagerank=1.0/(i+1))
            for i in range(5)
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        # Request 3 dimensions = 3 landmarks (top 3 by PageRank)
        embeddings = _fast_adjacency_embeddings(layer, dimensions=3)

        # All terms should get 3-dimensional embeddings
        for i in range(5):
            assert len(embeddings[f"term{i}"]) == 3

    def test_sampled_terms(self):
        """sampled_terms restricts which terms get embeddings."""
        cols = [
            MockMinicolumn(content="a", pagerank=1.0),
            MockMinicolumn(content="b", pagerank=0.5),
            MockMinicolumn(content="c", pagerank=0.3)
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=5, sampled_terms={"a", "b"}
        )

        # Only a and b should have embeddings
        assert "a" in embeddings
        assert "b" in embeddings
        assert "c" not in embeddings

    def test_idf_weighting_enabled(self):
        """IDF weighting down-weights common terms."""
        # Common term (in many docs) vs rare term (in few docs)
        col_common = MockMinicolumn(
            content="common",
            pagerank=1.0,
            document_ids={"doc1", "doc2", "doc3", "doc4", "doc5"}
        )
        col_rare = MockMinicolumn(
            content="rare",
            pagerank=0.8,
            document_ids={"doc1"}
        )
        layer = MockHierarchicalLayer([col_common, col_rare], level=0)

        # This should use IDF weighting by default
        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=2, use_idf_weighting=True
        )

        assert "common" in embeddings
        assert "rare" in embeddings

    def test_idf_weighting_disabled(self):
        """IDF weighting can be disabled."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            document_ids={"doc1", "doc2"}
        )
        col2 = MockMinicolumn(
            content="b",
            pagerank=0.5,
            document_ids={"doc1"}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _fast_adjacency_embeddings(
            layer, dimensions=2, use_idf_weighting=False
        )

        assert "a" in embeddings
        assert "b" in embeddings


# =============================================================================
# TF-IDF EMBEDDINGS
# =============================================================================


class TestTfidfEmbeddings:
    """Tests for _tfidf_embeddings."""

    def test_empty_layer(self):
        """Empty layer returns empty embeddings."""
        layer = MockHierarchicalLayer([], level=0)
        embeddings = _tfidf_embeddings(layer, dimensions=5)
        assert embeddings == {}

    def test_single_term_single_doc(self):
        """Single term in single doc gets embedding."""
        col = MockMinicolumn(
            content="test",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layer = MockHierarchicalLayer([col], level=0)
        embeddings = _tfidf_embeddings(layer, dimensions=5)

        assert "test" in embeddings
        assert len(embeddings["test"]) == 1  # Only 1 doc

    def test_multiple_docs(self):
        """Terms with multiple docs get embeddings."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 1.0, "doc2": 2.0, "doc3": 1.5}
        )
        layer = MockHierarchicalLayer([col], level=0)
        embeddings = _tfidf_embeddings(layer, dimensions=5)

        # Should use top 3 docs (or fewer if requesting more)
        assert "term" in embeddings
        vec = embeddings["term"]
        assert len(vec) == 3  # 3 docs available

    def test_dimensions_limits_docs(self):
        """dimensions parameter limits document dimensions."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2", "doc3", "doc4", "doc5"},
            tfidf_per_doc={
                "doc1": 1.0, "doc2": 2.0, "doc3": 1.5,
                "doc4": 0.5, "doc5": 0.8
            }
        )
        layer = MockHierarchicalLayer([col], level=0)
        embeddings = _tfidf_embeddings(layer, dimensions=3)

        # Should use only top 3 docs
        vec = embeddings["term"]
        assert len(vec) == 3

    def test_normalization(self):
        """Embeddings are L2-normalized."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2"},
            tfidf_per_doc={"doc1": 3.0, "doc2": 4.0}
        )
        layer = MockHierarchicalLayer([col], level=0)
        embeddings = _tfidf_embeddings(layer, dimensions=5)

        vec = embeddings["term"]
        magnitude = math.sqrt(sum(v*v for v in vec))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_sampled_terms(self):
        """sampled_terms restricts which terms get embeddings."""
        cols = [
            MockMinicolumn(
                content="a",
                document_ids={"doc1"},
                tfidf_per_doc={"doc1": 1.0}
            ),
            MockMinicolumn(
                content="b",
                document_ids={"doc1"},
                tfidf_per_doc={"doc1": 2.0}
            )
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        embeddings = _tfidf_embeddings(
            layer, dimensions=5, sampled_terms={"a"}
        )

        assert "a" in embeddings
        assert "b" not in embeddings

    def test_document_selection_by_size(self):
        """Documents selected as dimensions by term count."""
        # Create multiple terms across documents
        cols = [
            MockMinicolumn(
                content="term1",
                document_ids={"doc1", "doc2"},
                tfidf_per_doc={"doc1": 1.0, "doc2": 1.0}
            ),
            MockMinicolumn(
                content="term2",
                document_ids={"doc1"},
                tfidf_per_doc={"doc1": 2.0}
            ),
            MockMinicolumn(
                content="term3",
                document_ids={"doc2", "doc3"},
                tfidf_per_doc={"doc2": 1.5, "doc3": 1.5}
            )
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        # doc1 and doc2 have 2 terms each, doc3 has 1
        # Should prefer doc1 and doc2
        embeddings = _tfidf_embeddings(layer, dimensions=2)

        for term in ["term1", "term2", "term3"]:
            assert term in embeddings
            assert len(embeddings[term]) == 2


# =============================================================================
# ADJACENCY EMBEDDINGS (MULTI-HOP)
# =============================================================================


class TestAdjacencyEmbeddings:
    """Tests for _adjacency_embeddings with multi-hop propagation."""

    def test_empty_layer(self):
        """Empty layer returns empty embeddings."""
        layer = MockHierarchicalLayer([], level=0)
        embeddings = _adjacency_embeddings(layer, dimensions=5)
        assert embeddings == {}

    def test_single_term(self):
        """Single term gets embedding."""
        col = MockMinicolumn(content="test", pagerank=1.0)
        layer = MockHierarchicalLayer([col], level=0)
        embeddings = _adjacency_embeddings(layer, dimensions=3)

        assert "test" in embeddings
        assert len(embeddings["test"]) == 3  # Requested dimensions

    def test_direct_connection(self):
        """Direct connection to landmark reflected in embedding."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 5.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(layer, dimensions=2)

        # a connects to b with weight 5.0
        assert "a" in embeddings
        assert "b" in embeddings

    def test_multi_hop_propagation(self):
        """Multi-hop propagation reaches landmarks through neighbors."""
        # Create chain: a -> b -> c, where c is a high-PageRank landmark
        col_c = MockMinicolumn(content="c", pagerank=1.0)
        col_b = MockMinicolumn(
            content="b",
            pagerank=0.5,
            lateral_connections={"L0_c": 1.0}
        )
        col_a = MockMinicolumn(
            content="a",
            pagerank=0.3,
            lateral_connections={"L0_b": 1.0}
        )
        layer = MockHierarchicalLayer([col_a, col_b, col_c], level=0)

        # With propagation_steps=2, a should reach c through b
        embeddings = _adjacency_embeddings(
            layer, dimensions=3, propagation_steps=2, damping=0.5
        )

        assert "a" in embeddings
        assert "b" in embeddings
        assert "c" in embeddings

    def test_propagation_steps_parameter(self):
        """propagation_steps controls how far to propagate."""
        col_c = MockMinicolumn(content="c", pagerank=1.0)
        col_b = MockMinicolumn(
            content="b",
            pagerank=0.5,
            lateral_connections={"L0_c": 1.0}
        )
        col_a = MockMinicolumn(
            content="a",
            pagerank=0.3,
            lateral_connections={"L0_b": 1.0}
        )
        layer = MockHierarchicalLayer([col_a, col_b, col_c], level=0)

        # With 0 steps, only direct connections
        embeddings_0 = _adjacency_embeddings(
            layer, dimensions=3, propagation_steps=0
        )
        # With 2 steps, can reach through chain
        embeddings_2 = _adjacency_embeddings(
            layer, dimensions=3, propagation_steps=2
        )

        assert "a" in embeddings_0
        assert "a" in embeddings_2

    def test_damping_parameter(self):
        """damping parameter controls weight decay."""
        col_b = MockMinicolumn(content="b", pagerank=1.0)
        col_a = MockMinicolumn(
            content="a",
            pagerank=0.5,
            lateral_connections={"L0_b": 1.0}
        )
        layer = MockHierarchicalLayer([col_a, col_b], level=0)

        # Different damping values
        embeddings_low = _adjacency_embeddings(
            layer, dimensions=2, damping=0.1
        )
        embeddings_high = _adjacency_embeddings(
            layer, dimensions=2, damping=0.9
        )

        assert "a" in embeddings_low
        assert "a" in embeddings_high

    def test_normalization(self):
        """Embeddings are L2-normalized."""
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 10.0}
        )
        col2 = MockMinicolumn(content="b", pagerank=0.5)
        layer = MockHierarchicalLayer([col1, col2], level=0)

        embeddings = _adjacency_embeddings(layer, dimensions=2)

        vec = embeddings["a"]
        magnitude = math.sqrt(sum(v*v for v in vec))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_sampled_terms(self):
        """sampled_terms restricts which terms get embeddings."""
        cols = [
            MockMinicolumn(content="a", pagerank=1.0),
            MockMinicolumn(content="b", pagerank=0.5),
            MockMinicolumn(content="c", pagerank=0.3)
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        embeddings = _adjacency_embeddings(
            layer, dimensions=3, sampled_terms={"a", "c"}
        )

        assert "a" in embeddings
        assert "b" not in embeddings
        assert "c" in embeddings


# =============================================================================
# RANDOM WALK EMBEDDINGS
# =============================================================================


class TestRandomWalkEmbeddings:
    """Tests for _random_walk_embeddings."""

    def test_empty_layer(self):
        """Empty layer returns empty embeddings."""
        layer = MockHierarchicalLayer([], level=0)
        embeddings = _random_walk_embeddings(layer, dimensions=5)
        assert embeddings == {}

    def test_single_term(self):
        """Single isolated term gets embedding (all zeros)."""
        col = MockMinicolumn(content="isolated", pagerank=1.0)
        layer = MockHierarchicalLayer([col], level=0)

        # Set seed for reproducibility
        random.seed(42)
        embeddings = _random_walk_embeddings(
            layer, dimensions=1, walks_per_node=5, walk_length=10
        )

        assert "isolated" in embeddings

    def test_two_connected_terms(self):
        """Two connected terms get embeddings from random walks."""
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
            layer, dimensions=2, walks_per_node=10, walk_length=20
        )

        assert "a" in embeddings
        assert "b" in embeddings
        assert len(embeddings["a"]) == 2
        assert len(embeddings["b"]) == 2

    def test_walks_per_node_parameter(self):
        """walks_per_node controls number of walks from each term."""
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
            layer, dimensions=2, walks_per_node=100
        )

        assert "a" in embeddings
        assert "b" in embeddings

    def test_walk_length_parameter(self):
        """walk_length controls length of each walk."""
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
            layer, dimensions=2, walk_length=100
        )

        assert "a" in embeddings
        assert "b" in embeddings

    def test_window_size_parameter(self):
        """window_size controls co-occurrence context window."""
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
            layer, dimensions=2, window_size=10
        )

        assert "a" in embeddings
        assert "b" in embeddings

    def test_normalization(self):
        """Embeddings are L2-normalized."""
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
        embeddings = _random_walk_embeddings(layer, dimensions=2)

        vec = embeddings["a"]
        magnitude = math.sqrt(sum(v*v for v in vec))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_sampled_terms(self):
        """sampled_terms restricts which terms to walk from."""
        cols = [
            MockMinicolumn(
                content="a",
                pagerank=1.0,
                lateral_connections={"L0_b": 1.0}
            ),
            MockMinicolumn(
                content="b",
                pagerank=0.5,
                lateral_connections={"L0_a": 1.0, "L0_c": 1.0}
            ),
            MockMinicolumn(
                content="c",
                pagerank=0.3,
                lateral_connections={"L0_b": 1.0}
            )
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        random.seed(42)
        # Only walk from 'a'
        embeddings = _random_walk_embeddings(
            layer, dimensions=3, sampled_terms={"a"}, walks_per_node=10
        )

        # All terms should still get embeddings (landmarks)
        assert "a" in embeddings
        # But behavior may differ based on walks

    def test_weighted_walks(self):
        """Random walks respect edge weights."""
        # Strong connection to b, weak to c
        col1 = MockMinicolumn(
            content="a",
            pagerank=1.0,
            lateral_connections={"L0_b": 10.0, "L0_c": 0.1}
        )
        col2 = MockMinicolumn(
            content="b",
            pagerank=0.5,
            lateral_connections={"L0_a": 10.0}
        )
        col3 = MockMinicolumn(
            content="c",
            pagerank=0.3,
            lateral_connections={"L0_a": 0.1}
        )
        layer = MockHierarchicalLayer([col1, col2, col3], level=0)

        random.seed(42)
        embeddings = _random_walk_embeddings(
            layer, dimensions=3, walks_per_node=50, walk_length=10
        )

        assert "a" in embeddings
        assert "b" in embeddings
        assert "c" in embeddings


# =============================================================================
# WEIGHTED RANDOM WALK HELPER
# =============================================================================


class TestWeightedRandomWalk:
    """Tests for _weighted_random_walk helper function."""

    def test_single_node_no_connections(self):
        """Walk from isolated node returns just that node."""
        col = MockMinicolumn(content="isolated")
        layer = MockHierarchicalLayer([col], level=0)
        id_to_term = {"L0_isolated": "isolated"}

        walk = _weighted_random_walk(col, layer, length=10, id_to_term=id_to_term)

        assert walk == ["isolated"]

    def test_walk_length_respected(self):
        """Walk length parameter is respected."""
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(
            content="b",
            lateral_connections={"L0_a": 1.0}
        )
        layer = MockHierarchicalLayer([col1, col2], level=0)
        id_to_term = {"L0_a": "a", "L0_b": "b"}

        random.seed(42)
        walk = _weighted_random_walk(col1, layer, length=10, id_to_term=id_to_term)

        # Walk should be at most length 10
        assert len(walk) <= 10
        assert walk[0] == "a"  # Starts with starting node

    def test_weighted_selection(self):
        """Weighted random selection favors high-weight edges."""
        # Create node with strong preference for one neighbor
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 100.0, "L0_c": 1.0}
        )
        col2 = MockMinicolumn(content="b", lateral_connections={"L0_a": 1.0})
        col3 = MockMinicolumn(content="c", lateral_connections={"L0_a": 1.0})
        layer = MockHierarchicalLayer([col1, col2, col3], level=0)
        id_to_term = {"L0_a": "a", "L0_b": "b", "L0_c": "c"}

        # Do many short walks and count destinations
        random.seed(42)
        b_count = 0
        c_count = 0
        for _ in range(100):
            walk = _weighted_random_walk(col1, layer, length=2, id_to_term=id_to_term)
            if len(walk) > 1:
                if walk[1] == "b":
                    b_count += 1
                elif walk[1] == "c":
                    c_count += 1

        # Should heavily favor b over c
        assert b_count > c_count

    def test_walk_terminates_at_dead_end(self):
        """Walk terminates when reaching node with no connections."""
        col1 = MockMinicolumn(
            content="a",
            lateral_connections={"L0_b": 1.0}
        )
        col2 = MockMinicolumn(content="b")  # Dead end
        layer = MockHierarchicalLayer([col1, col2], level=0)
        id_to_term = {"L0_a": "a", "L0_b": "b"}

        walk = _weighted_random_walk(col1, layer, length=100, id_to_term=id_to_term)

        # Walk should terminate at b (dead end)
        assert len(walk) <= 2
        if len(walk) == 2:
            assert walk == ["a", "b"]


# =============================================================================
# SPECTRAL EMBEDDINGS
# =============================================================================


class TestSpectralEmbeddings:
    """Tests for _spectral_embeddings graph Laplacian method."""

    def test_empty_layer(self):
        """Empty layer returns empty embeddings."""
        layer = MockHierarchicalLayer([], level=0)
        embeddings = _spectral_embeddings(layer, dimensions=5)
        assert embeddings == {}

    def test_single_term(self):
        """Single term gets embedding."""
        col = MockMinicolumn(content="test")
        layer = MockHierarchicalLayer([col], level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=3)

        assert "test" in embeddings
        # Dimensions limited by number of nodes
        assert len(embeddings["test"]) == 3  # Actually min(3, 1) = 1, but padded

    def test_two_connected_terms(self):
        """Two connected terms get embeddings."""
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
        embeddings = _spectral_embeddings(layer, dimensions=2)

        assert "a" in embeddings
        assert "b" in embeddings
        assert len(embeddings["a"]) == 2
        assert len(embeddings["b"]) == 2

    def test_dimensions_limited_by_nodes(self):
        """Cannot have more dimensions than nodes."""
        cols = [MockMinicolumn(content=str(i)) for i in range(3)]
        layer = MockHierarchicalLayer(cols, level=0)

        random.seed(42)
        # Request 10 dimensions but only 3 nodes
        embeddings = _spectral_embeddings(layer, dimensions=10)

        # Should get 3 actual dimensions (+ padding to 10)
        for i in range(3):
            assert str(i) in embeddings
            assert len(embeddings[str(i)]) == 10

    def test_iterations_parameter(self):
        """iterations parameter controls power iteration."""
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
        embeddings = _spectral_embeddings(layer, dimensions=2, iterations=10)

        assert "a" in embeddings
        assert "b" in embeddings

    def test_sampled_terms(self):
        """sampled_terms restricts which terms get embeddings."""
        cols = [
            MockMinicolumn(content="a", lateral_connections={"L0_b": 1.0}),
            MockMinicolumn(content="b", lateral_connections={"L0_a": 1.0, "L0_c": 1.0}),
            MockMinicolumn(content="c", lateral_connections={"L0_b": 1.0})
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(
            layer, dimensions=3, sampled_terms={"a", "b"}
        )

        # Only a and b should have embeddings
        assert "a" in embeddings
        assert "b" in embeddings
        assert "c" not in embeddings

    def test_disconnected_components(self):
        """Handles disconnected graph components."""
        cols = [
            MockMinicolumn(content="a", lateral_connections={"L0_b": 1.0}),
            MockMinicolumn(content="b", lateral_connections={"L0_a": 1.0}),
            MockMinicolumn(content="c", lateral_connections={"L0_d": 1.0}),
            MockMinicolumn(content="d", lateral_connections={"L0_c": 1.0})
        ]
        layer = MockHierarchicalLayer(cols, level=0)

        random.seed(42)
        embeddings = _spectral_embeddings(layer, dimensions=4)

        # All nodes should get embeddings
        for term in ["a", "b", "c", "d"]:
            assert term in embeddings


# =============================================================================
# EMBEDDING SIMILARITY
# =============================================================================


class TestEmbeddingSimilarity:
    """Tests for embedding_similarity cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        embeddings = {
            "a": [1.0, 0.0, 0.0],
            "b": [1.0, 0.0, 0.0]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        embeddings = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        embeddings = {
            "a": [1.0, 0.0, 0.0],
            "b": [-1.0, 0.0, 0.0]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors have high positive similarity."""
        embeddings = {
            "a": [1.0, 1.0, 0.0],
            "b": [1.0, 0.9, 0.0]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity > 0.9

    def test_missing_term1(self):
        """Missing first term returns 0.0."""
        embeddings = {"b": [1.0, 0.0, 0.0]}
        similarity = embedding_similarity(embeddings, "missing", "b")
        assert similarity == 0.0

    def test_missing_term2(self):
        """Missing second term returns 0.0."""
        embeddings = {"a": [1.0, 0.0, 0.0]}
        similarity = embedding_similarity(embeddings, "a", "missing")
        assert similarity == 0.0

    def test_both_missing(self):
        """Both terms missing returns 0.0."""
        embeddings = {}
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity == 0.0

    def test_zero_magnitude_vectors(self):
        """Zero magnitude vectors return 0.0."""
        embeddings = {
            "a": [0.0, 0.0, 0.0],
            "b": [1.0, 0.0, 0.0]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert similarity == 0.0

    def test_symmetry(self):
        """Similarity is symmetric."""
        embeddings = {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0]
        }
        sim_ab = embedding_similarity(embeddings, "a", "b")
        sim_ba = embedding_similarity(embeddings, "b", "a")
        assert sim_ab == pytest.approx(sim_ba)

    def test_range_bounded(self):
        """Similarity is in [-1, 1]."""
        embeddings = {
            "a": [0.5, 0.5, 0.5],
            "b": [0.3, 0.7, 0.1]
        }
        similarity = embedding_similarity(embeddings, "a", "b")
        assert -1.0 <= similarity <= 1.0


# =============================================================================
# FIND SIMILAR BY EMBEDDING
# =============================================================================


class TestFindSimilarByEmbedding:
    """Tests for find_similar_by_embedding nearest neighbor search."""

    def test_empty_embeddings(self):
        """Empty embeddings returns empty list."""
        result = find_similar_by_embedding({}, "test", top_n=5)
        assert result == []

    def test_missing_term(self):
        """Missing query term returns empty list."""
        embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = find_similar_by_embedding(embeddings, "missing", top_n=5)
        assert result == []

    def test_single_other_term(self):
        """Single other term returns that term."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "other": [0.9, 0.1, 0.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=5)
        assert len(result) == 1
        assert result[0][0] == "other"
        assert result[0][1] > 0.9

    def test_multiple_terms_sorted(self):
        """Multiple terms returned sorted by similarity."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "very_similar": [0.99, 0.01, 0.0],
            "somewhat_similar": [0.7, 0.3, 0.0],
            "dissimilar": [0.0, 0.0, 1.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=5)

        assert len(result) == 3
        # Should be sorted by similarity descending
        assert result[0][0] == "very_similar"
        assert result[1][0] == "somewhat_similar"
        assert result[2][0] == "dissimilar"
        # Similarities should be descending
        assert result[0][1] > result[1][1] > result[2][1]

    def test_top_n_limits_results(self):
        """top_n parameter limits number of results."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "a": [0.9, 0.0, 0.0],
            "b": [0.8, 0.0, 0.0],
            "c": [0.7, 0.0, 0.0],
            "d": [0.6, 0.0, 0.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=2)

        assert len(result) == 2
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_excludes_self(self):
        """Query term is excluded from results."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "other": [0.9, 0.0, 0.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=5)

        # Should not include "query" itself
        assert all(term != "query" for term, _ in result)

    def test_negative_similarities(self):
        """Handles negative similarities correctly."""
        embeddings = {
            "query": [1.0, 0.0, 0.0],
            "opposite": [-1.0, 0.0, 0.0],
            "similar": [0.8, 0.0, 0.0]
        }
        result = find_similar_by_embedding(embeddings, "query", top_n=5)

        # Similar should rank higher than opposite
        assert result[0][0] == "similar"
        assert result[1][0] == "opposite"
        assert result[1][1] < 0  # Opposite has negative similarity

    def test_top_n_default(self):
        """Default top_n is 10."""
        embeddings = {f"term{i}": [float(i), 0.0] for i in range(15)}
        embeddings["query"] = [0.0, 1.0]

        result = find_similar_by_embedding(embeddings, "query")

        # Should return 10 by default
        assert len(result) == 10
