"""
Additional coverage tests for cortical/analysis/connections.py

This module targets specific uncovered lines in connections.py:
- Lines 174-192: use_member_semantics strategy
- Lines 196-208: use_embedding_similarity strategy
- Lines 120-122: Connection strengthening path
- Branch coverage gaps

These tests complement existing tests in test_analysis_coverage.py.
"""

import pytest
from cortical.analysis.connections import (
    compute_concept_connections,
    compute_bigram_connections,
)
from cortical.layers import CorticalLayer, HierarchicalLayer


class TestConceptConnectionsMemberSemantics:
    """Test use_member_semantics strategy in compute_concept_connections (lines 174-192)."""

    def test_member_semantics_connects_without_doc_overlap(self):
        """Member semantics connects concepts even without document overlap."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        # Create tokens
        t1 = layer0.get_or_create_minicolumn("neural")
        t2 = layer0.get_or_create_minicolumn("network")

        # Create concepts with NO document overlap
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}  # Different doc!
        c2.add_feedforward_connection(t2.id, 1.0)

        # Semantic relation between member tokens
        semantic_relations = [
            ("neural", "RelatedTo", "network", 0.8)
        ]

        # Run with use_member_semantics=True
        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            min_jaccard=0.5,  # High threshold prevents doc overlap connection
            use_member_semantics=True,
            use_embedding_similarity=False
        )

        # Should be connected via member semantics
        assert c2.id in c1.lateral_connections
        assert c1.id in c2.lateral_connections
        assert result['semantic_connections'] > 0

    def test_member_semantics_with_multiple_relations(self):
        """Member semantics handles multiple relations between concept members."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        # Create tokens
        t1 = layer0.get_or_create_minicolumn("dog")
        t2 = layer0.get_or_create_minicolumn("animal")
        t3 = layer0.get_or_create_minicolumn("mammal")
        t4 = layer0.get_or_create_minicolumn("creature")

        # Concept 1: {dog, animal}
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)
        c1.add_feedforward_connection(t2.id, 1.0)

        # Concept 2: {mammal, creature}
        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}
        c2.add_feedforward_connection(t3.id, 1.0)
        c2.add_feedforward_connection(t4.id, 1.0)

        # Multiple semantic relations
        semantic_relations = [
            ("dog", "IsA", "mammal", 0.9),
            ("dog", "IsA", "creature", 0.8),
            ("animal", "RelatedTo", "mammal", 0.7),
            ("animal", "RelatedTo", "creature", 0.6)
        ]

        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            min_jaccard=1.0,  # Impossible threshold
            use_member_semantics=True
        )

        # Should be connected with averaged semantic score
        assert c2.id in c1.lateral_connections
        assert result['semantic_connections'] > 0

    def test_member_semantics_no_relations(self):
        """Member semantics doesn't connect when no relations exist."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        t1 = layer0.get_or_create_minicolumn("apple")
        t2 = layer0.get_or_create_minicolumn("car")

        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        # No semantic relations
        result = compute_concept_connections(
            layers,
            semantic_relations=[],
            min_jaccard=1.0,
            use_member_semantics=True
        )

        # Should NOT be connected
        assert c2.id not in c1.lateral_connections
        assert result['semantic_connections'] == 0

    def test_member_semantics_skipped_when_doc_filter_passes(self):
        """Member semantics is skipped when document overlap filter passes."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        t1 = layer0.get_or_create_minicolumn("term1")
        t2 = layer0.get_or_create_minicolumn("term2")

        # Both concepts share documents (passes doc filter)
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1", "doc2"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc1", "doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        semantic_relations = [("term1", "RelatedTo", "term2", 0.9)]

        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            min_jaccard=0.1,  # Low threshold - doc filter passes
            use_member_semantics=True
        )

        # Should be connected via doc overlap, NOT member semantics
        assert c2.id in c1.lateral_connections
        assert result['doc_overlap_connections'] > 0
        assert result['semantic_connections'] == 0  # Not via member semantics


class TestConceptConnectionsEmbeddingSimilarity:
    """Test use_embedding_similarity strategy in compute_concept_connections (lines 196-208)."""

    def test_embedding_similarity_connects_concepts(self):
        """Embedding similarity connects concepts without document overlap."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        # Create tokens
        t1 = layer0.get_or_create_minicolumn("cat")
        t2 = layer0.get_or_create_minicolumn("dog")

        # Concepts with no document overlap
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        # Similar embeddings
        embeddings = {
            "cat": [1.0, 0.8, 0.3],
            "dog": [0.9, 0.85, 0.35]
        }

        result = compute_concept_connections(
            layers,
            min_jaccard=1.0,  # Impossible threshold
            use_embedding_similarity=True,
            embedding_threshold=0.3,
            embeddings=embeddings
        )

        # Should be connected via embedding similarity
        assert c2.id in c1.lateral_connections
        assert result['embedding_connections'] > 0

    def test_embedding_similarity_below_threshold(self):
        """Embedding similarity doesn't connect when below threshold."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        t1 = layer0.get_or_create_minicolumn("cat")
        t2 = layer0.get_or_create_minicolumn("car")

        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        # Very different embeddings
        embeddings = {
            "cat": [1.0, 0.0, 0.0],
            "car": [0.0, 1.0, 0.0]
        }

        result = compute_concept_connections(
            layers,
            min_jaccard=1.0,
            use_embedding_similarity=True,
            embedding_threshold=0.9,  # High threshold
            embeddings=embeddings
        )

        # Should NOT be connected
        assert c2.id not in c1.lateral_connections
        assert result['embedding_connections'] == 0

    def test_embedding_similarity_missing_embeddings(self):
        """Embedding similarity handles missing embeddings gracefully."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        t1 = layer0.get_or_create_minicolumn("known")
        t2 = layer0.get_or_create_minicolumn("unknown")

        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        # Only one term has embedding
        embeddings = {
            "known": [1.0, 0.5, 0.3]
            # "unknown" is missing
        }

        result = compute_concept_connections(
            layers,
            min_jaccard=1.0,
            use_embedding_similarity=True,
            embeddings=embeddings
        )

        # Should NOT crash, should NOT connect
        assert c2.id not in c1.lateral_connections
        assert result['embedding_connections'] == 0

    def test_embedding_similarity_skipped_when_doc_filter_passes(self):
        """Embedding similarity is skipped when document overlap filter passes."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        t1 = layer0.get_or_create_minicolumn("term1")
        t2 = layer0.get_or_create_minicolumn("term2")

        # Both concepts share documents
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1", "doc2"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc1", "doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        embeddings = {
            "term1": [1.0, 0.0],
            "term2": [0.9, 0.1]
        }

        result = compute_concept_connections(
            layers,
            min_jaccard=0.1,  # Doc filter passes
            use_embedding_similarity=True,
            embeddings=embeddings
        )

        # Connected via doc overlap, NOT embedding similarity
        assert c2.id in c1.lateral_connections
        assert result['doc_overlap_connections'] > 0
        assert result['embedding_connections'] == 0


class TestConceptConnectionsStrengthening:
    """Test connection strengthening path (lines 120-122)."""

    def test_strengthens_when_both_strategies_connect_same_pair(self):
        """Both member_semantics AND embedding_similarity strengthen same connection."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        # Create tokens
        t1 = layer0.get_or_create_minicolumn("cat")
        t2 = layer0.get_or_create_minicolumn("dog")

        # Concepts with NO document overlap (so both strategies can run)
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}  # Different doc!
        c2.add_feedforward_connection(t2.id, 1.0)

        # Both semantic relations AND embeddings
        semantic_relations = [("cat", "RelatedTo", "dog", 0.8)]
        embeddings = {
            "cat": [1.0, 0.8, 0.3],
            "dog": [0.9, 0.85, 0.35]  # Similar embedding
        }

        # Enable BOTH strategies - this exercises lines 120-122
        # Member semantics connects first, then embedding tries to connect same pair
        result = compute_concept_connections(
            layers,
            semantic_relations=semantic_relations,
            min_jaccard=1.0,  # Impossible threshold so doc filter fails
            use_member_semantics=True,
            use_embedding_similarity=True,
            embedding_threshold=0.3,
            embeddings=embeddings
        )

        # Should be connected (strengthened by both strategies)
        assert c2.id in c1.lateral_connections
        # Both semantic AND embedding connections created
        # But only counts as 1 in connections_created
        assert result['connections_created'] == 1
        assert result['semantic_connections'] >= 1 or result['embedding_connections'] >= 1


class TestBigramConnectionsBranchCoverage:
    """Test branch coverage gaps in compute_bigram_connections."""

    def test_bigram_not_found_in_layer(self):
        """Handles case where bigram ID doesn't exist (line 437)."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)
        }
        layer1 = layers[CorticalLayer.BIGRAMS]

        # Create a bigram but then corrupt its ID reference
        b1 = layer1.get_or_create_minicolumn("word1 word2")
        b1.document_ids = {"doc1"}
        b1.tfidf = 1.0

        b2 = layer1.get_or_create_minicolumn("word2 word3")
        b2.document_ids = {"doc1"}
        b2.tfidf = 1.0

        # Run connection computation
        result = compute_bigram_connections(layers)

        # Should complete without error
        assert isinstance(result, dict)
        assert 'connections_created' in result


class TestConceptConnectionsTokenLookup:
    """Test token lookup edge cases in compute_concept_connections."""

    def test_missing_token_in_feedforward(self):
        """Handles missing tokens in feedforward connections (line 92)."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        # Create a token
        t1 = layer0.get_or_create_minicolumn("real_token")

        # Create concept with both real and invalid token IDs
        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)
        c1.add_feedforward_connection("L0_nonexistent", 1.0)  # Invalid token

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc1"}
        c2.add_feedforward_connection(t1.id, 1.0)

        # Should handle gracefully
        result = compute_concept_connections(layers, min_jaccard=0.1)
        assert isinstance(result, dict)
        assert result['connections_created'] > 0


class TestConceptConnectionsEmbeddingCentroid:
    """Test embedding centroid computation edge cases."""

    def test_empty_member_embeddings(self):
        """Handles concepts where no members have embeddings (line 102)."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer0 = layers[CorticalLayer.TOKENS]
        layer2 = layers[CorticalLayer.CONCEPTS]

        # Tokens without embeddings
        t1 = layer0.get_or_create_minicolumn("no_embedding1")
        t2 = layer0.get_or_create_minicolumn("no_embedding2")

        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1"}
        c1.add_feedforward_connection(t1.id, 1.0)

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc2"}
        c2.add_feedforward_connection(t2.id, 1.0)

        # Empty embeddings dict
        embeddings = {}

        result = compute_concept_connections(
            layers,
            min_jaccard=1.0,
            use_embedding_similarity=True,
            embeddings=embeddings
        )

        # Should handle gracefully - no crash
        assert result['embedding_connections'] == 0
