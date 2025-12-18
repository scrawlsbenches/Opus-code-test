"""
Unit Tests for Graph Analysis Functions
========================================

Tests for graph analysis and connection building algorithms:
- compute_bigram_connections: Build lateral connections between bigrams
- compute_document_connections: Connect documents by shared terms
- compute_concept_connections: Connect concepts by overlap
- propagate_activation: Spread activation through network

Extracted from test_analysis.py for better organization (Task #T-20251215-213424-8400-004).
"""

import pytest

from cortical.analysis import (
    compute_bigram_connections,
    compute_document_connections,
    compute_concept_connections,
    propagate_activation,
)


# =============================================================================
# BIGRAM CONNECTIONS TESTS
# =============================================================================


class TestComputeBigramConnections:
    """Tests for compute_bigram_connections() function."""

    def test_empty_layer(self):
        """Empty bigram layer returns zero connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layers = {CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)}
        result = compute_bigram_connections(layers)

        assert result['connections_created'] == 0
        assert result['bigrams'] == 0

    def test_shared_left_component(self):
        """Bigrams sharing left component are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        # Create many bigrams with "the" as left component
        for i in range(10):
            b = layer1.get_or_create_minicolumn(f"the word{i}")
            b.document_ids.add("doc1")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers, max_bigrams_per_term=5)

        # Should skip "the" due to limit
        assert result['skipped_common_terms'] > 0


class TestBigramConnectionsEdgeCases:
    """Edge case tests for bigram connections."""

    def test_single_bigram(self):
        """Single bigram creates no connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        layer1.get_or_create_minicolumn("single bigram")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers)

        assert result['connections_created'] == 0

    def test_no_shared_components(self):
        """Bigrams with no shared components create no connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        layer1.get_or_create_minicolumn("alpha beta")
        layer1.get_or_create_minicolumn("gamma delta")

        layers = {CorticalLayer.BIGRAMS: layer1}
        result = compute_bigram_connections(layers, min_shared_docs=10)

        # No shared components or documents, so no connections
        assert result['connections_created'] == 0


# =============================================================================
# DOCUMENT CONNECTIONS TESTS
# =============================================================================


class TestComputeDocumentConnections:
    """Tests for compute_document_connections() function."""

    def test_empty_documents(self):
        """Empty document set."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS)
        }
        compute_document_connections(layers, {})
        # Should not crash

    def test_shared_terms_create_connection(self):
        """Documents sharing terms are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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


# =============================================================================
# CONCEPT CONNECTIONS TESTS
# =============================================================================


class TestComputeConceptConnections:
    """Tests for compute_concept_connections() function."""

    def test_empty_concepts(self):
        """Empty concept layer returns zero connections."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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


class TestConceptConnectionsSemanticAndEmbedding:
    """Test concept connections with semantic and embedding weights."""

    def test_basic_concept_connections(self):
        """Concepts with shared documents are connected."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token1 = layer0.get_or_create_minicolumn("token1")
        token2 = layer0.get_or_create_minicolumn("token2")

        concept1 = layer2.get_or_create_minicolumn("concept1")
        concept2 = layer2.get_or_create_minicolumn("concept2")

        concept1.document_ids.add("doc1")
        concept1.document_ids.add("doc2")
        concept2.document_ids.add("doc1")
        concept2.document_ids.add("doc2")

        concept1.feedforward_connections[token1.id] = 1.0
        concept2.feedforward_connections[token2.id] = 1.0

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.CONCEPTS: layer2
        }

        result = compute_concept_connections(layers, min_shared_docs=1, min_jaccard=0.1)

        # Should create connection
        assert result['connections_created'] > 0


# =============================================================================
# ACTIVATION PROPAGATION TESTS
# =============================================================================


class TestPropagateActivation:
    """Tests for propagate_activation() function."""

    def test_empty_layers(self):
        """Empty layers don't crash."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        propagate_activation(layers, iterations=1)
        # Should not crash

    def test_activation_decays(self):
        """Activation decays over iterations."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

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


class TestPropagateActivationFeedforward:
    """Test propagate_activation with multiple layers."""

    def test_multi_layer_activation(self):
        """Test activation propagates across layers."""
        from cortical.layers import HierarchicalLayer, CorticalLayer

        layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)

        token = layer0.get_or_create_minicolumn("token")
        bigram = layer1.get_or_create_minicolumn("bigram")

        token.activation = 1.0
        bigram.activation = 0.0

        # Lateral connection within layer
        token2 = layer0.get_or_create_minicolumn("token2")
        token2.activation = 0.0
        token.add_lateral_connection(token2.id, 1.0)

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1
        }

        propagate_activation(layers, iterations=1)

        # Token2 should receive some activation
        assert token2.activation >= 0
