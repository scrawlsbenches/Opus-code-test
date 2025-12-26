"""
Unit tests for cortical.query.analogy module.

Tests all functions for analogy completion and semantic relation discovery.
"""

import pytest
from unittest.mock import MagicMock, Mock
from cortical.query.analogy import (
    find_relation_between,
    find_terms_with_relation,
    complete_analogy,
    complete_analogy_simple,
)
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn


class TestFindRelationBetween:
    """Tests for find_relation_between function."""

    def test_finds_direct_relation(self):
        """Should find direct relation from a to b."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("knowledge", "COMPOUND_WITH", "graphs", 0.7),
        ]
        result = find_relation_between("neural", "networks", relations)
        assert len(result) == 1
        assert result[0][0] == "COMPOUND_WITH"
        assert result[0][1] == 0.8

    def test_finds_reverse_relation(self):
        """Should find reverse relation with penalty."""
        relations = [("neural", "COMPOUND_WITH", "networks", 0.8)]
        result = find_relation_between("networks", "neural", relations)
        assert len(result) == 1
        assert result[0][0] == "COMPOUND_WITH"
        assert result[0][1] == 0.8 * 0.9  # Reverse penalty

    def test_returns_empty_for_no_relation(self):
        """Should return empty list when no relation exists."""
        relations = [("neural", "COMPOUND_WITH", "networks", 0.8)]
        result = find_relation_between("foo", "bar", relations)
        assert result == []

    def test_empty_relations_list(self):
        """Should handle empty relations list."""
        result = find_relation_between("neural", "networks", [])
        assert result == []

    def test_multiple_relations_same_pair(self):
        """Should find multiple relations between same pair."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("neural", "RELATES_TO", "networks", 0.6),
            ("neural", "SIMILAR_TO", "networks", 0.5),
        ]
        result = find_relation_between("neural", "networks", relations)
        assert len(result) == 3
        # Should be sorted by weight descending
        assert result[0][1] == 0.8
        assert result[1][1] == 0.6
        assert result[2][1] == 0.5

    def test_sorts_by_weight_descending(self):
        """Should sort results by weight in descending order."""
        relations = [
            ("neural", "RELATES_TO", "networks", 0.3),
            ("neural", "COMPOUND_WITH", "networks", 0.9),
            ("neural", "SIMILAR_TO", "networks", 0.5),
        ]
        result = find_relation_between("neural", "networks", relations)
        assert result[0][1] == 0.9
        assert result[1][1] == 0.5
        assert result[2][1] == 0.3


class TestFindTermsWithRelation:
    """Tests for find_terms_with_relation function."""

    def test_finds_forward_relations(self):
        """Should find terms in forward direction."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("neural", "COMPOUND_WITH", "computation", 0.7),
        ]
        result = find_terms_with_relation(
            "neural", "COMPOUND_WITH", relations, direction='forward'
        )
        assert len(result) == 2
        assert ("networks", 0.8) in result
        assert ("computation", 0.7) in result

    def test_finds_backward_relations(self):
        """Should find terms in backward direction."""
        relations = [
            ("networks", "COMPOUND_WITH", "neural", 0.8),
            ("computation", "COMPOUND_WITH", "neural", 0.7),
        ]
        result = find_terms_with_relation(
            "neural", "COMPOUND_WITH", relations, direction='backward'
        )
        assert len(result) == 2
        assert ("networks", 0.8) in result
        assert ("computation", 0.7) in result

    def test_filters_by_relation_type(self):
        """Should only return relations of specified type."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("neural", "RELATES_TO", "computation", 0.7),
        ]
        result = find_terms_with_relation(
            "neural", "COMPOUND_WITH", relations, direction='forward'
        )
        assert len(result) == 1
        assert result[0][0] == "networks"

    def test_returns_empty_for_no_matches(self):
        """Should return empty list when no matches found."""
        relations = [("neural", "COMPOUND_WITH", "networks", 0.8)]
        result = find_terms_with_relation(
            "foo", "COMPOUND_WITH", relations, direction='forward'
        )
        assert result == []

    def test_empty_relations_list(self):
        """Should handle empty relations list."""
        result = find_terms_with_relation(
            "neural", "COMPOUND_WITH", [], direction='forward'
        )
        assert result == []

    def test_sorts_by_weight_descending(self):
        """Should sort results by weight in descending order."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.5),
            ("neural", "COMPOUND_WITH", "computation", 0.9),
            ("neural", "COMPOUND_WITH", "processing", 0.3),
        ]
        result = find_terms_with_relation(
            "neural", "COMPOUND_WITH", relations, direction='forward'
        )
        assert result[0][1] == 0.9
        assert result[1][1] == 0.5
        assert result[2][1] == 0.3


class TestCompleteAnalogy:
    """Tests for complete_analogy function."""

    @pytest.fixture
    def mock_layers(self):
        """Create mock layers with minicolumns."""
        layer0 = Mock(spec=HierarchicalLayer)

        # Create mock minicolumns
        col_a = Mock(spec=Minicolumn)
        col_a.id = "L0_neural"
        col_a.content = "neural"
        col_a.lateral_connections = {"L0_networks": 0.8, "L0_computation": 0.5}

        col_b = Mock(spec=Minicolumn)
        col_b.id = "L0_networks"
        col_b.content = "networks"
        col_b.lateral_connections = {"L0_neural": 0.8}

        col_c = Mock(spec=Minicolumn)
        col_c.id = "L0_knowledge"
        col_c.content = "knowledge"
        col_c.lateral_connections = {"L0_graphs": 0.7, "L0_base": 0.6}

        col_d = Mock(spec=Minicolumn)
        col_d.id = "L0_graphs"
        col_d.content = "graphs"
        col_d.lateral_connections = {"L0_knowledge": 0.7}

        # Setup get_minicolumn
        def get_minicolumn(term):
            mapping = {
                "neural": col_a,
                "networks": col_b,
                "knowledge": col_c,
                "graphs": col_d,
            }
            return mapping.get(term)

        # Setup get_by_id
        def get_by_id(col_id):
            mapping = {
                "L0_neural": col_a,
                "L0_networks": col_b,
                "L0_knowledge": col_c,
                "L0_graphs": col_d,
            }
            return mapping.get(col_id)

        layer0.get_minicolumn = Mock(side_effect=get_minicolumn)
        layer0.get_by_id = Mock(side_effect=get_by_id)

        return {CorticalLayer.TOKENS: layer0}

    @pytest.fixture
    def sample_relations(self):
        """Sample semantic relations."""
        return [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("knowledge", "COMPOUND_WITH", "graphs", 0.7),
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for vector arithmetic."""
        return {
            "neural": [1.0, 0.0, 0.0],
            "networks": [1.0, 1.0, 0.0],
            "knowledge": [0.0, 0.0, 1.0],
            "graphs": [0.0, 1.0, 1.0],
        }

    def test_returns_empty_when_term_a_missing(self, mock_layers, sample_relations):
        """Should return empty when term_a doesn't exist."""
        result = complete_analogy(
            "nonexistent", "networks", "knowledge",
            mock_layers, sample_relations
        )
        assert result == []

    def test_returns_empty_when_term_b_missing(self, mock_layers, sample_relations):
        """Should return empty when term_b doesn't exist."""
        result = complete_analogy(
            "neural", "nonexistent", "knowledge",
            mock_layers, sample_relations
        )
        assert result == []

    def test_returns_empty_when_term_c_missing(self, mock_layers, sample_relations):
        """Should return empty when term_c doesn't exist."""
        result = complete_analogy(
            "neural", "networks", "nonexistent",
            mock_layers, sample_relations
        )
        assert result == []

    def test_relation_based_completion(self, mock_layers, sample_relations):
        """Should find analogy using relation matching."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, sample_relations,
            use_embeddings=False
        )
        assert len(result) > 0
        # Should find "graphs" via COMPOUND_WITH relation
        terms = [r[0] for r in result]
        assert "graphs" in terms

        # Check that method is relation-based
        for term, score, method in result:
            if term == "graphs":
                assert method.startswith("relation:")

    def test_embedding_based_completion(self, mock_layers, sample_relations, sample_embeddings):
        """Should find analogy using vector arithmetic."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, sample_relations,
            embeddings=sample_embeddings,
            use_relations=False
        )
        assert len(result) > 0
        # Check that method is embedding
        for term, score, method in result:
            assert method == "embedding"

    def test_pattern_based_completion(self, mock_layers):
        """Should find analogy using co-occurrence patterns."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, [],
            use_embeddings=False,
            use_relations=False
        )
        # Pattern matching should find something based on lateral connections
        assert len(result) > 0
        for term, score, method in result:
            assert method == "pattern"

    def test_excludes_input_terms(self, mock_layers, sample_relations):
        """Should not include input terms in results."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, sample_relations
        )
        terms = [r[0] for r in result]
        assert "neural" not in terms
        assert "networks" not in terms
        assert "knowledge" not in terms

    def test_respects_top_n(self, mock_layers, sample_relations):
        """Should return at most top_n results."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, sample_relations,
            top_n=3
        )
        assert len(result) <= 3

    def test_empty_semantic_relations(self, mock_layers):
        """Should handle empty semantic relations."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, [],
            use_embeddings=False
        )
        # Should still work with pattern matching
        assert isinstance(result, list)

    def test_combined_strategies(self, mock_layers, sample_relations, sample_embeddings):
        """Should combine multiple strategies."""
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, sample_relations,
            embeddings=sample_embeddings,
            use_embeddings=True,
            use_relations=True
        )
        # Should have results from different methods
        methods = {r[2] for r in result}
        assert len(methods) > 0

    def test_embedding_similarity_threshold(self, mock_layers, sample_embeddings):
        """Should filter embeddings by similarity threshold (0.5)."""
        # Create embeddings with actually dissimilar directions
        # vec_d will be approximately [0.0, 1.0, 1.0]
        # "orthogonal" is perpendicular to vec_d
        low_sim_embeddings = {
            "neural": [1.0, 0.0, 0.0],
            "networks": [1.0, 1.0, 0.0],
            "knowledge": [0.0, 0.0, 1.0],
            "orthogonal": [1.0, 0.0, 0.0],  # Perpendicular to vec_d
            "similar": [0.0, 1.0, 1.0],  # Should be included
        }
        result = complete_analogy(
            "neural", "networks", "knowledge",
            mock_layers, [],
            embeddings=low_sim_embeddings,
            use_relations=False
        )
        # "orthogonal" should be filtered out due to low similarity
        # vec_d = [0, 0, 1] + ([1, 1, 0] - [1, 0, 0]) = [0, 0, 1] + [0, 1, 0] = [0, 1, 1]
        # similarity with [1, 0, 0] = 0 / (... * ...) = 0 < 0.5
        terms = [r[0] for r in result]
        assert "orthogonal" not in terms
        # "similar" should be included (high similarity)
        assert "similar" in terms


class TestCompleteAnalogySimple:
    """Tests for complete_analogy_simple function."""

    @pytest.fixture
    def mock_layers_with_bigrams(self):
        """Create mock layers including bigrams."""
        layer0 = Mock(spec=HierarchicalLayer)
        layer1 = Mock(spec=HierarchicalLayer)

        # Create token minicolumns
        col_a = Mock(spec=Minicolumn)
        col_a.id = "L0_neural"
        col_a.content = "neural"
        col_a.lateral_connections = {"L0_networks": 0.8}

        col_b = Mock(spec=Minicolumn)
        col_b.id = "L0_networks"
        col_b.content = "networks"
        col_b.lateral_connections = {"L0_neural": 0.8}

        col_c = Mock(spec=Minicolumn)
        col_c.id = "L0_knowledge"
        col_c.content = "knowledge"
        col_c.lateral_connections = {"L0_graphs": 0.7}

        col_d = Mock(spec=Minicolumn)
        col_d.id = "L0_graphs"
        col_d.content = "graphs"
        col_d.pagerank = 0.5

        # Create bigram minicolumns
        bigram_ab = Mock(spec=Minicolumn)
        bigram_ab.content = "neural networks"
        bigram_ab.pagerank = 0.8

        bigram_cd = Mock(spec=Minicolumn)
        bigram_cd.content = "knowledge graphs"
        bigram_cd.pagerank = 0.7

        # Setup layer0
        def get_minicolumn_l0(term):
            mapping = {
                "neural": col_a,
                "networks": col_b,
                "knowledge": col_c,
                "graphs": col_d,
            }
            return mapping.get(term)

        def get_by_id_l0(col_id):
            mapping = {
                "L0_neural": col_a,
                "L0_networks": col_b,
                "L0_knowledge": col_c,
                "L0_graphs": col_d,
            }
            return mapping.get(col_id)

        layer0.get_minicolumn = Mock(side_effect=get_minicolumn_l0)
        layer0.get_by_id = Mock(side_effect=get_by_id_l0)

        # Setup layer1
        def get_minicolumn_l1(bigram):
            mapping = {
                "neural networks": bigram_ab,
                "knowledge graphs": bigram_cd,
            }
            return mapping.get(bigram)

        layer1.get_minicolumn = Mock(side_effect=get_minicolumn_l1)
        layer1.minicolumns = {
            "L1_neural networks": bigram_ab,
            "L1_knowledge graphs": bigram_cd,
        }

        return {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1,
        }

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return Mock()

    def test_returns_empty_when_term_missing(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should return empty when any term doesn't exist."""
        result = complete_analogy_simple(
            "nonexistent", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer
        )
        assert result == []

    def test_bigram_pattern_matching(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should find analogy using bigram patterns."""
        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer
        )
        assert len(result) > 0
        # Should find "graphs" via bigram pattern
        terms = [r[0] for r in result]
        assert "graphs" in terms

    def test_co_occurrence_similarity(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should use co-occurrence patterns."""
        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer
        )
        # Should have results from co-occurrence
        assert len(result) > 0

    def test_semantic_relations_integration(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should integrate semantic relations when available."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("knowledge", "COMPOUND_WITH", "graphs", 0.7),
        ]
        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer,
            semantic_relations=relations
        )
        assert len(result) > 0

    def test_excludes_input_terms(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should not include input terms in results."""
        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer
        )
        terms = [r[0] for r in result]
        assert "neural" not in terms
        assert "networks" not in terms
        assert "knowledge" not in terms

    def test_respects_top_n(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should return at most top_n results."""
        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer,
            top_n=2
        )
        assert len(result) <= 2

    def test_without_bigrams_layer(self, mock_tokenizer):
        """Should work without bigrams layer."""
        layer0 = Mock(spec=HierarchicalLayer)

        col_a = Mock(spec=Minicolumn)
        col_a.id = "L0_neural"
        col_a.content = "neural"
        col_a.lateral_connections = {"L0_networks": 0.8}

        col_b = Mock(spec=Minicolumn)
        col_b.id = "L0_networks"
        col_b.content = "networks"

        col_c = Mock(spec=Minicolumn)
        col_c.id = "L0_knowledge"
        col_c.content = "knowledge"
        col_c.lateral_connections = {"L0_graphs": 0.7}

        col_d = Mock(spec=Minicolumn)
        col_d.id = "L0_graphs"
        col_d.content = "graphs"

        def get_minicolumn(term):
            mapping = {
                "neural": col_a,
                "networks": col_b,
                "knowledge": col_c,
                "graphs": col_d,
            }
            return mapping.get(term)

        def get_by_id(col_id):
            mapping = {
                "L0_neural": col_a,
                "L0_networks": col_b,
                "L0_knowledge": col_c,
                "L0_graphs": col_d,
            }
            return mapping.get(col_id)

        layer0.get_minicolumn = Mock(side_effect=get_minicolumn)
        layer0.get_by_id = Mock(side_effect=get_by_id)

        layers = {CorticalLayer.TOKENS: layer0}

        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            layers, mock_tokenizer
        )
        # Should still work using co-occurrence
        assert isinstance(result, list)

    def test_scores_combination(self, mock_layers_with_bigrams, mock_tokenizer):
        """Should combine scores from different strategies."""
        relations = [
            ("neural", "COMPOUND_WITH", "networks", 0.8),
            ("knowledge", "COMPOUND_WITH", "graphs", 0.7),
        ]
        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            mock_layers_with_bigrams, mock_tokenizer,
            semantic_relations=relations
        )
        # Should have combined scores from multiple strategies
        if result:
            assert all(isinstance(score, float) for _, score in result)
            assert all(score > 0 for _, score in result)

    def test_reverse_bigram_direction(self, mock_tokenizer):
        """Should find bigrams in both directions."""
        layer0 = Mock(spec=HierarchicalLayer)
        layer1 = Mock(spec=HierarchicalLayer)

        # Setup basic minicolumns
        col_c = Mock(spec=Minicolumn)
        col_c.id = "L0_knowledge"
        col_c.content = "knowledge"
        col_c.lateral_connections = {}

        col_d = Mock(spec=Minicolumn)
        col_d.id = "L0_graphs"
        col_d.content = "graphs"

        # Reverse direction bigram
        bigram_dc = Mock(spec=Minicolumn)
        bigram_dc.content = "graphs knowledge"
        bigram_dc.pagerank = 0.6

        def get_minicolumn_l0(term):
            return {"knowledge": col_c, "graphs": col_d}.get(term)

        layer0.get_minicolumn = Mock(side_effect=get_minicolumn_l0)
        layer0.get_by_id = Mock(return_value=col_d)

        layer1.get_minicolumn = Mock(return_value=None)
        layer1.minicolumns = {"L1_graphs knowledge": bigram_dc}

        layers = {
            CorticalLayer.TOKENS: layer0,
            CorticalLayer.BIGRAMS: layer1,
        }

        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            layers, mock_tokenizer
        )
        # Should find "graphs" in reverse bigram
        if result:
            terms = [r[0] for r in result]
            assert "graphs" in terms
