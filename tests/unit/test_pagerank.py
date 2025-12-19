"""
Unit Tests for PageRank Algorithms
===================================

Tests for all PageRank-related algorithms:
- _pagerank_core: Core PageRank algorithm
- compute_pagerank: Standard PageRank on layers
- compute_semantic_pagerank: PageRank with semantic relation boosting
- compute_hierarchical_pagerank: Multi-layer PageRank propagation

Extracted from test_analysis.py for better organization (Task #T-20251215-213424-8400-004).
"""

import pytest

from cortical.analysis import (
    _pagerank_core,
    compute_pagerank,
    compute_semantic_pagerank,
    compute_hierarchical_pagerank,
)


# =============================================================================
# PAGERANK CORE ALGORITHM TESTS
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
# LAYER PAGERANK TESTS
# =============================================================================


class TestComputePageRank:
    """Tests for compute_pagerank() wrapper function."""

    def test_empty_layer(self):
        """Empty layer returns empty dict."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = compute_pagerank(layer)
        assert result == {}

    def test_single_minicolumn(self):
        """Single minicolumn with no edges gets base rank."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("test")
        result = compute_pagerank(layer, damping=0.85)
        # With no edges, rank = (1-d)/n = 0.15/1 = 0.15
        assert result[col.id] == pytest.approx(0.15)
        assert col.pagerank == pytest.approx(0.15)

    def test_two_connected_minicolumns(self):
        """Two connected minicolumns share rank."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=1.5)

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_pagerank(layer, damping=-0.1)

    def test_pagerank_updates_minicolumns(self):
        """PageRank values are written to minicolumns."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("col1")
        col2 = layer.get_or_create_minicolumn("col2")
        col1.add_lateral_connection(col2.id, 1.0)

        compute_pagerank(layer)
        # Minicolumns should have pagerank set
        assert col1.pagerank > 0
        assert col2.pagerank > 0


# =============================================================================
# SEMANTIC PAGERANK TESTS
# =============================================================================


class TestComputeSemanticPageRank:
    """Tests for compute_semantic_pagerank() function."""

    def test_empty_layer(self):
        """Empty layer returns empty result."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = compute_semantic_pagerank(layer, [])

        assert result['pagerank'] == {}
        assert result['iterations_run'] == 0
        assert result['edges_with_relations'] == 0

    def test_invalid_damping_raises(self):
        """Invalid damping factor raises ValueError."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_semantic_pagerank(layer, [], damping=1.5)

    def test_semantic_relations_boost_connections(self):
        """Semantic relations increase edge weights."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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


class TestSemanticPageRankMissingPaths:
    """Test compute_semantic_pagerank missing coverage (lines 667, 680)."""

    def test_semantic_relations_without_lookup_match(self):
        """Test path where semantic_lookup doesn't match (line 667, 680)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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


# =============================================================================
# HIERARCHICAL PAGERANK TESTS
# =============================================================================


class TestComputeHierarchicalPageRank:
    """Tests for compute_hierarchical_pagerank() function."""

    def test_empty_layers(self):
        """Empty layers return quickly."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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

        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}

        with pytest.raises(ValueError, match="damping must be between 0 and 1"):
            compute_hierarchical_pagerank(layers, damping=1.5)

        with pytest.raises(ValueError, match="cross_layer_damping must be between 0 and 1"):
            compute_hierarchical_pagerank(layers, cross_layer_damping=1.5)

    def test_cross_layer_propagation(self):
        """PageRank propagates between layers."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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


class TestHierarchicalPageRankCoverage:
    """Test compute_hierarchical_pagerank missing paths (lines 806-857)."""

    def test_cross_layer_feedback_propagation(self):
        """Test feedback connections propagate up (line 808)."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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
