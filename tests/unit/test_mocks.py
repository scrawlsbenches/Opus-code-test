"""
Unit Tests for Mock Objects
===========================

Tests that the mock objects work correctly and can be used for unit testing.

This file also serves as documentation for how to use the mocks.
"""

import pytest

from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
    MockEdge,
    layers_to_graph,
    layers_to_adjacency,
)


class TestMockMinicolumn:
    """Tests for MockMinicolumn test double."""

    def test_auto_generates_id(self):
        """ID is auto-generated from layer and content."""
        col = MockMinicolumn(content="test", layer=0)
        assert col.id == "L0_test"

        col2 = MockMinicolumn(content="bigram", layer=1)
        assert col2.id == "L1_bigram"

    def test_explicit_id(self):
        """Explicit ID overrides auto-generation."""
        col = MockMinicolumn(id="custom_id", content="test")
        assert col.id == "custom_id"

    def test_default_values(self):
        """Default values are sensible."""
        col = MockMinicolumn(content="test")
        assert col.pagerank == 1.0
        assert col.tfidf == 0.0
        assert col.activation == 0.0
        assert col.occurrence_count == 1
        assert col.document_ids == set()
        assert col.lateral_connections == {}

    def test_controllable_attributes(self):
        """All attributes can be controlled."""
        col = MockMinicolumn(
            content="neural",
            pagerank=0.8,
            tfidf=2.5,
            activation=1.0,
            document_ids={"doc1", "doc2"},
            lateral_connections={"L0_networks": 0.9},
        )
        assert col.pagerank == 0.8
        assert col.tfidf == 2.5
        assert col.activation == 1.0
        assert col.document_ids == {"doc1", "doc2"}
        assert col.lateral_connections == {"L0_networks": 0.9}

    def test_add_lateral_connection(self):
        """add_lateral_connection accumulates weights."""
        col = MockMinicolumn(content="test")
        col.add_lateral_connection("L0_other", 0.5)
        col.add_lateral_connection("L0_other", 0.3)
        assert col.lateral_connections["L0_other"] == 0.8

    def test_add_typed_connection(self):
        """add_typed_connection creates MockEdge."""
        col = MockMinicolumn(content="test")
        col.add_typed_connection(
            "L0_related",
            weight=0.8,
            relation_type="IsA",
            confidence=0.9,
        )
        edge = col.get_typed_connection("L0_related")
        assert edge is not None
        assert edge.weight == 0.8
        assert edge.relation_type == "IsA"
        assert edge.confidence == 0.9
        # Also updates lateral_connections
        assert col.lateral_connections["L0_related"] == 0.8

    def test_connection_count(self):
        """connection_count returns number of lateral connections."""
        col = MockMinicolumn(
            content="test",
            lateral_connections={"a": 1.0, "b": 0.5, "c": 0.3}
        )
        assert col.connection_count() == 3

    def test_top_connections(self):
        """top_connections returns strongest connections."""
        col = MockMinicolumn(
            content="test",
            lateral_connections={"a": 0.5, "b": 1.0, "c": 0.3}
        )
        top = col.top_connections(n=2)
        assert top == [("b", 1.0), ("a", 0.5)]


class TestMockHierarchicalLayer:
    """Tests for MockHierarchicalLayer test double."""

    def test_empty_layer(self):
        """Empty layer works correctly."""
        layer = MockHierarchicalLayer()
        assert layer.column_count() == 0
        assert layer.get_minicolumn("nonexistent") is None
        assert layer.get_by_id("L0_nonexistent") is None

    def test_with_minicolumns(self):
        """Layer initialized with minicolumns."""
        cols = [
            MockMinicolumn(content="a"),
            MockMinicolumn(content="b"),
        ]
        layer = MockHierarchicalLayer(cols)
        assert layer.column_count() == 2
        assert layer.get_minicolumn("a") is not None
        assert layer.get_minicolumn("b") is not None

    def test_get_by_id(self):
        """get_by_id returns minicolumn by ID."""
        col = MockMinicolumn(content="test")
        layer = MockHierarchicalLayer([col])
        assert layer.get_by_id("L0_test") == col
        assert layer.get_by_id("L0_nonexistent") is None

    def test_get_or_create(self):
        """get_or_create_minicolumn creates if not exists."""
        layer = MockHierarchicalLayer()
        col = layer.get_or_create_minicolumn("new_term")
        assert col.content == "new_term"
        assert layer.column_count() == 1
        # Second call returns same object
        col2 = layer.get_or_create_minicolumn("new_term")
        assert col2 is col

    def test_remove_minicolumn(self):
        """remove_minicolumn removes from layer."""
        col = MockMinicolumn(content="test")
        layer = MockHierarchicalLayer([col])
        assert layer.column_count() == 1
        result = layer.remove_minicolumn("test")
        assert result is True
        assert layer.column_count() == 0
        assert layer.get_minicolumn("test") is None

    def test_iteration(self):
        """Layer supports iteration."""
        cols = [MockMinicolumn(content="a"), MockMinicolumn(content="b")]
        layer = MockHierarchicalLayer(cols)
        contents = [col.content for col in layer]
        assert set(contents) == {"a", "b"}

    def test_contains(self):
        """Layer supports 'in' operator."""
        layer = MockHierarchicalLayer([MockMinicolumn(content="test")])
        assert "test" in layer
        assert "nonexistent" not in layer

    def test_top_by_pagerank(self):
        """top_by_pagerank returns highest ranked."""
        cols = [
            MockMinicolumn(content="low", pagerank=0.1),
            MockMinicolumn(content="high", pagerank=0.9),
            MockMinicolumn(content="mid", pagerank=0.5),
        ]
        layer = MockHierarchicalLayer(cols)
        top = layer.top_by_pagerank(n=2)
        assert top[0] == ("high", 0.9)
        assert top[1] == ("mid", 0.5)


class TestMockLayers:
    """Tests for MockLayers factory."""

    def test_empty(self):
        """empty() creates 4 empty layers."""
        layers = MockLayers.empty()
        assert len(layers) == 4
        assert all(layer.column_count() == 0 for layer in layers.values())

    def test_single_term(self):
        """single_term creates one token."""
        layers = MockLayers.single_term("test", pagerank=0.5, tfidf=2.0)
        token_layer = layers[MockLayers.TOKENS]
        assert token_layer.column_count() == 1
        col = token_layer.get_minicolumn("test")
        assert col.pagerank == 0.5
        assert col.tfidf == 2.0

    def test_two_connected_terms(self):
        """two_connected_terms creates bidirectional connection."""
        layers = MockLayers.two_connected_terms("a", "b", weight=0.7)
        token_layer = layers[MockLayers.TOKENS]
        assert token_layer.column_count() == 2

        col_a = token_layer.get_minicolumn("a")
        col_b = token_layer.get_minicolumn("b")

        assert col_a.lateral_connections["L0_b"] == 0.7
        assert col_b.lateral_connections["L0_a"] == 0.7

    def test_connected_chain(self):
        """connected_chain creates a -> b -> c chain."""
        layers = MockLayers.connected_chain(["a", "b", "c"])
        token_layer = layers[MockLayers.TOKENS]

        col_a = token_layer.get_minicolumn("a")
        col_b = token_layer.get_minicolumn("b")
        col_c = token_layer.get_minicolumn("c")

        # a connects to b only
        assert "L0_b" in col_a.lateral_connections
        assert "L0_c" not in col_a.lateral_connections

        # b connects to both a and c
        assert "L0_a" in col_b.lateral_connections
        assert "L0_c" in col_b.lateral_connections

        # c connects to b only
        assert "L0_b" in col_c.lateral_connections
        assert "L0_a" not in col_c.lateral_connections

    def test_complete_graph(self):
        """complete_graph connects all terms."""
        layers = MockLayers.complete_graph(["a", "b", "c"])
        token_layer = layers[MockLayers.TOKENS]

        for content in ["a", "b", "c"]:
            col = token_layer.get_minicolumn(content)
            # Should connect to 2 other nodes
            assert col.connection_count() == 2

    def test_disconnected_terms(self):
        """disconnected_terms has no connections."""
        layers = MockLayers.disconnected_terms(["a", "b", "c"])
        token_layer = layers[MockLayers.TOKENS]

        for content in ["a", "b", "c"]:
            col = token_layer.get_minicolumn(content)
            assert col.connection_count() == 0

    def test_document_with_terms(self):
        """document_with_terms creates token and document layers."""
        layers = MockLayers.document_with_terms("doc1", ["term1", "term2"])

        token_layer = layers[MockLayers.TOKENS]
        doc_layer = layers[MockLayers.DOCUMENTS]

        assert token_layer.column_count() == 2
        assert doc_layer.column_count() == 1

        term_col = token_layer.get_minicolumn("term1")
        assert "doc1" in term_col.document_ids

        doc_col = doc_layer.get_minicolumn("doc1")
        assert "L0_term1" in doc_col.feedforward_connections

    def test_multi_document_corpus(self):
        """multi_document_corpus handles multiple docs."""
        layers = MockLayers.multi_document_corpus({
            "doc1": ["shared", "unique1"],
            "doc2": ["shared", "unique2"],
        })

        token_layer = layers[MockLayers.TOKENS]
        doc_layer = layers[MockLayers.DOCUMENTS]

        # Shared term appears in both docs
        shared_col = token_layer.get_minicolumn("shared")
        assert shared_col.document_ids == {"doc1", "doc2"}

        # Unique terms in one doc each
        unique1_col = token_layer.get_minicolumn("unique1")
        assert unique1_col.document_ids == {"doc1"}

        assert doc_layer.column_count() == 2

    def test_clustered_terms(self):
        """clustered_terms assigns cluster IDs."""
        layers = MockLayers.clustered_terms({
            "cluster_a": ["term1", "term2"],
            "cluster_b": ["term3"],
        })

        token_layer = layers[MockLayers.TOKENS]

        # Same cluster - strong connection
        term1 = token_layer.get_minicolumn("term1")
        term2 = token_layer.get_minicolumn("term2")
        assert term1.cluster_id == term2.cluster_id

        # Different cluster - weak connection
        term3 = token_layer.get_minicolumn("term3")
        assert term3.cluster_id != term1.cluster_id

    def test_with_bigrams(self):
        """with_bigrams creates bigram layer."""
        layers = MockLayers.with_bigrams(
            terms=["neural", "networks"],
            bigrams=[("neural", "networks")]
        )

        bigram_layer = layers[MockLayers.BIGRAMS]
        assert bigram_layer.column_count() == 1

        bigram_col = bigram_layer.get_minicolumn("neural networks")
        assert bigram_col is not None
        assert "L0_neural" in bigram_col.feedforward_connections


class TestLayerBuilder:
    """Tests for LayerBuilder fluent API."""

    def test_empty_build(self):
        """Build with no terms creates empty layers."""
        layers = LayerBuilder().build()
        assert layers[MockLayers.TOKENS].column_count() == 0

    def test_with_term(self):
        """with_term adds a term with attributes."""
        layers = LayerBuilder() \
            .with_term("test", pagerank=0.5, tfidf=2.0) \
            .build()

        col = layers[MockLayers.TOKENS].get_minicolumn("test")
        assert col.pagerank == 0.5
        assert col.tfidf == 2.0

    def test_with_terms_batch(self):
        """with_terms adds multiple terms."""
        layers = LayerBuilder() \
            .with_terms(["a", "b", "c"], pagerank=0.5) \
            .build()

        token_layer = layers[MockLayers.TOKENS]
        assert token_layer.column_count() == 3
        for content in ["a", "b", "c"]:
            assert token_layer.get_minicolumn(content).pagerank == 0.5

    def test_with_connection(self):
        """with_connection creates bidirectional connection."""
        layers = LayerBuilder() \
            .with_connection("a", "b", weight=0.8) \
            .build()

        token_layer = layers[MockLayers.TOKENS]
        col_a = token_layer.get_minicolumn("a")
        col_b = token_layer.get_minicolumn("b")

        assert col_a.lateral_connections["L0_b"] == 0.8
        assert col_b.lateral_connections["L0_a"] == 0.8

    def test_with_connection_unidirectional(self):
        """with_connection can be unidirectional."""
        layers = LayerBuilder() \
            .with_connection("a", "b", weight=0.8, bidirectional=False) \
            .build()

        token_layer = layers[MockLayers.TOKENS]
        col_a = token_layer.get_minicolumn("a")
        col_b = token_layer.get_minicolumn("b")

        assert col_a.lateral_connections["L0_b"] == 0.8
        assert "L0_a" not in col_b.lateral_connections

    def test_with_document(self):
        """with_document creates document layer."""
        layers = LayerBuilder() \
            .with_document("doc1", ["term1", "term2"]) \
            .build()

        doc_layer = layers[MockLayers.DOCUMENTS]
        doc_col = doc_layer.get_minicolumn("doc1")
        assert doc_col is not None
        assert "L0_term1" in doc_col.feedforward_connections

    def test_with_bigram(self):
        """with_bigram creates bigram layer."""
        layers = LayerBuilder() \
            .with_bigram("neural", "networks") \
            .build()

        bigram_layer = layers[MockLayers.BIGRAMS]
        assert bigram_layer.column_count() == 1

    def test_with_cluster(self):
        """with_cluster assigns cluster ID."""
        layers = LayerBuilder() \
            .with_cluster("term1", 0) \
            .with_cluster("term2", 0) \
            .with_cluster("term3", 1) \
            .build()

        token_layer = layers[MockLayers.TOKENS]
        assert token_layer.get_minicolumn("term1").cluster_id == 0
        assert token_layer.get_minicolumn("term2").cluster_id == 0
        assert token_layer.get_minicolumn("term3").cluster_id == 1

    def test_chaining(self):
        """Builder methods are chainable."""
        layers = LayerBuilder() \
            .with_term("a", pagerank=0.8) \
            .with_term("b", pagerank=0.6) \
            .with_connection("a", "b", 0.9) \
            .with_document("doc1", ["a", "b"]) \
            .with_bigram("a", "b") \
            .build()

        assert layers[MockLayers.TOKENS].column_count() == 2
        assert layers[MockLayers.BIGRAMS].column_count() == 1
        assert layers[MockLayers.DOCUMENTS].column_count() == 1

    def test_build_token_layer(self):
        """build_token_layer returns just token layer."""
        layer = LayerBuilder() \
            .with_term("test") \
            .build_token_layer()

        assert isinstance(layer, MockHierarchicalLayer)
        assert layer.column_count() == 1


class TestGraphHelpers:
    """Tests for graph conversion helpers."""

    def test_layers_to_graph(self):
        """layers_to_graph extracts simple graph."""
        layers = MockLayers.two_connected_terms("a", "b", weight=0.5)
        graph = layers_to_graph(layers)

        assert "a" in graph
        assert "b" in graph
        assert ("b", 0.5) in graph["a"]
        assert ("a", 0.5) in graph["b"]

    def test_layers_to_adjacency(self):
        """layers_to_adjacency extracts adjacency dict."""
        layers = MockLayers.two_connected_terms("a", "b", weight=0.7)
        adj = layers_to_adjacency(layers)

        assert adj["a"]["b"] == 0.7
        assert adj["b"]["a"] == 0.7

    def test_empty_layers_to_graph(self):
        """Empty layers produce empty graph."""
        layers = MockLayers.empty()
        graph = layers_to_graph(layers)
        assert graph == {}
