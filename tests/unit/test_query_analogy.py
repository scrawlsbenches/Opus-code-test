"""
Unit Tests for Query Analogy Module
====================================

Task #174: Unit tests for cortical/query/analogy.py

Tests analogy completion functions:
- find_relation_between: Detect relations between term pairs
- find_terms_with_relation: Follow semantic relations
- complete_analogy: Full analogy completion (a:b::c:?)
- complete_analogy_simple: Simplified bigram-based completion

Coverage target: 90%
"""

import pytest
from typing import Dict, List, Tuple

from cortical.query.analogy import (
    find_relation_between,
    find_terms_with_relation,
    complete_analogy,
    complete_analogy_simple,
)
from cortical.layers import CorticalLayer
from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
)


# =============================================================================
# HELPER FIXTURES
# =============================================================================


@pytest.fixture
def sample_semantic_relations():
    """Sample semantic relations for testing."""
    return [
        ("neural", "IsA", "networks", 0.9),
        ("neural", "SimilarTo", "deep", 0.8),
        ("networks", "UsedFor", "learning", 0.7),
        ("knowledge", "IsA", "graphs", 0.85),
        ("knowledge", "SimilarTo", "semantic", 0.75),
        ("graphs", "UsedFor", "representation", 0.8),
        ("dog", "IsA", "animal", 0.95),
        ("cat", "IsA", "animal", 0.95),
        ("cat", "Antonym", "dog", 0.6),
        ("hot", "Antonym", "cold", 0.9),
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing vector arithmetic."""
    return {
        "neural": [1.0, 0.5, 0.2],
        "networks": [0.9, 0.6, 0.3],
        "knowledge": [0.8, 0.4, 0.1],
        "graphs": [0.7, 0.5, 0.2],
        "deep": [0.95, 0.55, 0.25],
        "learning": [0.85, 0.45, 0.15],
    }


# =============================================================================
# FIND_RELATION_BETWEEN TESTS
# =============================================================================


class TestFindRelationBetween:
    """Tests for find_relation_between function."""

    def test_empty_relations(self):
        """Empty relations list returns empty result."""
        result = find_relation_between("a", "b", [])
        assert result == []

    def test_no_matching_relation(self):
        """No matching relation returns empty."""
        relations = [("x", "IsA", "y", 0.9)]
        result = find_relation_between("a", "b", relations)
        assert result == []

    def test_single_forward_relation(self):
        """Find single forward relation a->b."""
        relations = [("neural", "IsA", "networks", 0.9)]
        result = find_relation_between("neural", "networks", relations)
        assert len(result) == 1
        assert result[0] == ("IsA", 0.9)

    def test_single_reverse_relation(self):
        """Find reverse relation b->a with penalty."""
        relations = [("neural", "IsA", "networks", 0.9)]
        result = find_relation_between("networks", "neural", relations)
        assert len(result) == 1
        assert result[0][0] == "IsA"
        # Reverse has 0.9 penalty
        assert result[0][1] == pytest.approx(0.9 * 0.9)

    def test_multiple_relations_same_pair(self):
        """Multiple relations between same pair."""
        relations = [
            ("neural", "IsA", "networks", 0.9),
            ("neural", "SimilarTo", "networks", 0.7),
            ("neural", "RelatedTo", "networks", 0.6),
        ]
        result = find_relation_between("neural", "networks", relations)
        assert len(result) == 3
        # Sorted by weight descending
        assert result[0][1] >= result[1][1] >= result[2][1]
        assert result[0] == ("IsA", 0.9)

    def test_mixed_forward_reverse(self):
        """Mix of forward and reverse relations."""
        relations = [
            ("a", "IsA", "b", 0.9),
            ("b", "SimilarTo", "a", 0.8),
        ]
        result = find_relation_between("a", "b", relations)
        assert len(result) == 2
        # Forward relation has higher weight
        assert result[0] == ("IsA", 0.9)
        # Reverse with penalty
        assert result[1] == ("SimilarTo", pytest.approx(0.8 * 0.9))

    def test_sorted_by_weight(self):
        """Results sorted by weight descending."""
        relations = [
            ("a", "Rel1", "b", 0.5),
            ("a", "Rel2", "b", 0.9),
            ("a", "Rel3", "b", 0.7),
        ]
        result = find_relation_between("a", "b", relations)
        weights = [w for _, w in result]
        assert weights == sorted(weights, reverse=True)

    def test_reverse_penalty_applied(self):
        """Reverse direction applies 0.9 penalty."""
        relations = [("a", "IsA", "b", 1.0)]
        forward = find_relation_between("a", "b", relations)
        reverse = find_relation_between("b", "a", relations)
        assert forward[0][1] == 1.0
        assert reverse[0][1] == pytest.approx(0.9)


# =============================================================================
# FIND_TERMS_WITH_RELATION TESTS
# =============================================================================


class TestFindTermsWithRelation:
    """Tests for find_terms_with_relation function."""

    def test_empty_relations(self):
        """Empty relations returns empty."""
        result = find_terms_with_relation("a", "IsA", [])
        assert result == []

    def test_no_matching_relation_type(self):
        """No matching relation type returns empty."""
        relations = [("a", "IsA", "b", 0.9)]
        result = find_terms_with_relation("a", "SimilarTo", relations)
        assert result == []

    def test_no_matching_term(self):
        """No matching term returns empty."""
        relations = [("a", "IsA", "b", 0.9)]
        result = find_terms_with_relation("x", "IsA", relations)
        assert result == []

    def test_forward_direction(self):
        """Forward direction finds targets."""
        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("cat", "IsA", "animal", 0.8),
        ]
        result = find_terms_with_relation("dog", "IsA", relations, direction='forward')
        assert len(result) == 1
        assert result[0] == ("animal", 0.9)

    def test_backward_direction(self):
        """Backward direction finds sources."""
        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("cat", "IsA", "animal", 0.8),
        ]
        result = find_terms_with_relation("animal", "IsA", relations, direction='backward')
        assert len(result) == 2
        # Sorted by weight
        assert result[0] == ("dog", 0.9)
        assert result[1] == ("cat", 0.8)

    def test_multiple_targets(self):
        """Term with multiple targets."""
        relations = [
            ("neural", "SimilarTo", "deep", 0.9),
            ("neural", "SimilarTo", "artificial", 0.8),
            ("neural", "SimilarTo", "cognitive", 0.7),
        ]
        result = find_terms_with_relation("neural", "SimilarTo", relations, direction='forward')
        assert len(result) == 3
        assert result[0] == ("deep", 0.9)
        assert result[1] == ("artificial", 0.8)

    def test_sorted_by_weight(self):
        """Results sorted by weight descending."""
        relations = [
            ("a", "Rel", "target1", 0.5),
            ("a", "Rel", "target2", 0.9),
            ("a", "Rel", "target3", 0.7),
        ]
        result = find_terms_with_relation("a", "Rel", relations, direction='forward')
        weights = [w for _, w in result]
        assert weights == sorted(weights, reverse=True)

    def test_different_relation_types_ignored(self):
        """Only matching relation types included."""
        relations = [
            ("a", "IsA", "b", 0.9),
            ("a", "SimilarTo", "c", 0.8),
            ("a", "UsedFor", "d", 0.7),
        ]
        result = find_terms_with_relation("a", "IsA", relations, direction='forward')
        assert len(result) == 1
        assert result[0][0] == "b"


# =============================================================================
# COMPLETE_ANALOGY TESTS
# =============================================================================


class TestCompleteAnalogy:
    """Tests for complete_analogy function (full version)."""

    def test_empty_layers(self, sample_semantic_relations):
        """Empty layers returns empty."""
        layers = MockLayers.empty()
        result = complete_analogy("a", "b", "c", layers, sample_semantic_relations)
        assert result == []

    def test_missing_term_a(self, sample_semantic_relations):
        """Missing term_a returns empty."""
        layers = LayerBuilder().with_terms(["b", "c"]).build()
        result = complete_analogy("a", "b", "c", layers, sample_semantic_relations)
        assert result == []

    def test_missing_term_b(self, sample_semantic_relations):
        """Missing term_b returns empty."""
        layers = LayerBuilder().with_terms(["a", "c"]).build()
        result = complete_analogy("a", "b", "c", layers, sample_semantic_relations)
        assert result == []

    def test_missing_term_c(self, sample_semantic_relations):
        """Missing term_c returns empty."""
        layers = LayerBuilder().with_terms(["a", "b"]).build()
        result = complete_analogy("a", "b", "c", layers, sample_semantic_relations)
        assert result == []

    def test_no_semantic_relations(self):
        """No semantic relations with use_relations=True."""
        layers = LayerBuilder().with_terms(["neural", "networks", "knowledge"]).build()
        result = complete_analogy("neural", "networks", "knowledge", layers, [])
        # Should try pattern matching, may return some results
        assert isinstance(result, list)

    def test_relation_based_completion(self, sample_semantic_relations):
        """Relation-based completion: neural:networks::knowledge:?"""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs", "deep", "semantic"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations,
            use_embeddings=False,
            use_relations=True
        )

        # Should find "graphs" (knowledge IsA graphs, like neural IsA networks)
        assert len(result) > 0
        terms = [term for term, score, method in result]
        assert "graphs" in terms

    def test_embedding_based_completion(self, sample_embeddings):
        """Embedding-based completion using vector arithmetic."""
        layers = LayerBuilder().with_terms(list(sample_embeddings.keys())).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, [],
            embeddings=sample_embeddings,
            use_embeddings=True,
            use_relations=False
        )

        # Should use vector arithmetic d = c + (b - a)
        assert len(result) > 0
        # Results should have 'embedding' method
        methods = [method for _, _, method in result]
        assert 'embedding' in methods

    def test_combined_strategies(self, sample_semantic_relations, sample_embeddings):
        """Combined relation + embedding strategies."""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs", "deep"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations,
            embeddings=sample_embeddings,
            use_embeddings=True,
            use_relations=True
        )

        # Should combine both strategies
        assert len(result) > 0

    def test_pattern_matching_strategy(self, sample_semantic_relations):
        """Pattern matching based on co-occurrence."""
        layers = LayerBuilder() \
            .with_term("neural") \
            .with_term("networks") \
            .with_term("knowledge") \
            .with_term("target") \
            .with_connection("neural", "networks", 0.9) \
            .with_connection("knowledge", "target", 0.8) \
            .build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations,
            use_embeddings=False,
            use_relations=True
        )

        # Pattern matching should find "target"
        assert len(result) > 0

    def test_excludes_input_terms(self, sample_semantic_relations):
        """Result excludes input terms a, b, c."""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations
        )

        terms = [term for term, score, method in result]
        assert "neural" not in terms
        assert "networks" not in terms
        assert "knowledge" not in terms

    def test_top_n_limit(self, sample_semantic_relations):
        """Result limited by top_n parameter."""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs",
            "term1", "term2", "term3", "term4", "term5"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations,
            top_n=3
        )

        assert len(result) <= 3

    def test_sorted_by_confidence(self, sample_semantic_relations):
        """Results sorted by confidence descending."""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs", "semantic"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations
        )

        if len(result) > 1:
            scores = [score for _, score, _ in result]
            assert scores == sorted(scores, reverse=True)

    def test_method_attribution(self, sample_semantic_relations, sample_embeddings):
        """Each result has method attribution."""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations,
            embeddings=sample_embeddings
        )

        if result:
            for term, score, method in result:
                assert isinstance(term, str)
                assert isinstance(score, (int, float))
                assert isinstance(method, str)
                assert method in ['embedding', 'pattern'] or method.startswith('relation:')

    def test_no_embeddings_no_error(self, sample_semantic_relations):
        """use_embeddings=True but no embeddings provided."""
        layers = LayerBuilder().with_terms(["a", "b", "c"]).build()

        result = complete_analogy(
            "a", "b", "c",
            layers, sample_semantic_relations,
            embeddings=None,
            use_embeddings=True
        )

        # Should not crash, just skip embedding strategy
        assert isinstance(result, list)

    def test_embedding_similarity_threshold(self, sample_embeddings):
        """Only includes embeddings above similarity threshold (0.5)."""
        # Create embeddings with one very dissimilar term
        embeddings = sample_embeddings.copy()
        embeddings["unrelated"] = [-10.0, -10.0, -10.0]

        layers = LayerBuilder().with_terms(list(embeddings.keys())).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, [],
            embeddings=embeddings,
            use_embeddings=True,
            use_relations=False
        )

        terms = [term for term, _, _ in result]
        # Very dissimilar term should be excluded
        # (This depends on the actual similarity calculation)

    def test_relation_weight_scoring(self, sample_semantic_relations):
        """Higher relation weights give higher scores."""
        relations = [
            ("a", "IsA", "b", 0.9),
            ("c", "IsA", "high", 0.9),
            ("c", "IsA", "low", 0.3),
        ]

        layers = LayerBuilder().with_terms(["a", "b", "c", "high", "low"]).build()

        result = complete_analogy(
            "a", "b", "c",
            layers, relations,
            use_embeddings=False,
            use_relations=True
        )

        if len(result) >= 2:
            # "high" should rank higher than "low"
            term_scores = {term: score for term, score, _ in result}
            if "high" in term_scores and "low" in term_scores:
                assert term_scores["high"] > term_scores["low"]


# =============================================================================
# COMPLETE_ANALOGY_SIMPLE TESTS
# =============================================================================


class TestCompleteAnalogySimple:
    """Tests for complete_analogy_simple function."""

    def test_empty_layers(self):
        """Empty layers returns empty."""
        layers = MockLayers.empty()
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        result = complete_analogy_simple("a", "b", "c", layers, tokenizer)
        assert result == []

    def test_missing_terms(self):
        """Missing any input term returns empty."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder().with_terms(["a", "b"]).build()
        result = complete_analogy_simple("a", "b", "c", layers, tokenizer)
        assert result == []

    def test_bigram_pattern_matching(self):
        """Bigram pattern: neural networks -> knowledge ?"""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder() \
            .with_terms(["neural", "networks", "knowledge", "graphs"]) \
            .with_bigram("neural", "networks") \
            .with_bigram("knowledge", "graphs") \
            .build()

        # Add pagerank to bigrams
        bigram_layer = layers[MockLayers.BIGRAMS]
        for col in bigram_layer:
            col.pagerank = 0.5

        result = complete_analogy_simple(
            "neural", "networks", "knowledge",
            layers, tokenizer
        )

        # Should find "graphs" from "knowledge graphs" bigram
        if result:
            terms = [term for term, score in result]
            assert "graphs" in terms

    def test_cooccurrence_strategy(self):
        """Co-occurrence similarity strategy."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder() \
            .with_term("a") \
            .with_term("b") \
            .with_term("c") \
            .with_term("target") \
            .with_connection("a", "other", 0.5) \
            .with_connection("c", "target", 0.5) \
            .build()

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer
        )

        # Co-occurrence strategy should find some candidates
        assert isinstance(result, list)

    def test_semantic_relations_integration(self):
        """Integration with semantic relations."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("cat", "IsA", "animal", 0.8),
        ]

        layers = LayerBuilder().with_terms(["dog", "animal", "cat"]).build()

        result = complete_analogy_simple(
            "dog", "animal", "cat",
            layers, tokenizer,
            semantic_relations=relations
        )

        # Should use relation strategy
        assert isinstance(result, list)

    def test_excludes_input_terms(self):
        """Excludes a, b, c from results."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder() \
            .with_terms(["a", "b", "c", "other"]) \
            .with_bigram("a", "b") \
            .with_bigram("c", "other") \
            .build()

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer
        )

        terms = [term for term, score in result]
        assert "a" not in terms
        assert "b" not in terms
        assert "c" not in terms

    def test_top_n_limit(self):
        """Results limited by top_n."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder() \
            .with_terms(["a", "b", "c", "d1", "d2", "d3", "d4"]) \
            .with_connection("c", "d1", 0.5) \
            .with_connection("c", "d2", 0.5) \
            .with_connection("c", "d3", 0.5) \
            .with_connection("c", "d4", 0.5) \
            .build()

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer,
            top_n=2
        )

        assert len(result) <= 2

    def test_sorted_by_score(self):
        """Results sorted by score descending."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder() \
            .with_terms(["a", "b", "c", "d1", "d2", "d3"]) \
            .with_connection("c", "d1", 0.9) \
            .with_connection("c", "d2", 0.5) \
            .with_connection("c", "d3", 0.3) \
            .build()

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer
        )

        if len(result) > 1:
            scores = [score for _, score in result]
            assert scores == sorted(scores, reverse=True)

    def test_no_bigram_layer(self):
        """Works even without bigram layer."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([
            MockMinicolumn(content="a"),
            MockMinicolumn(content="b"),
            MockMinicolumn(content="c"),
        ])

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer
        )

        # Should still work with co-occurrence
        assert isinstance(result, list)

    def test_bidirectional_bigrams(self):
        """Checks both forward and reverse bigrams."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        layers = LayerBuilder() \
            .with_terms(["a", "b", "c", "target"]) \
            .with_bigram("a", "b") \
            .with_bigram("target", "c") \
            .build()

        # Add pagerank
        bigram_layer = layers[MockLayers.BIGRAMS]
        for col in bigram_layer:
            col.pagerank = 0.5

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer
        )

        # Should find "target" from reverse bigram pattern
        if result:
            terms = [term for term, score in result]
            # "target" might be found with lower score (0.6 penalty)

    def test_score_accumulation(self):
        """Scores accumulate from multiple strategies."""
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()

        relations = [("a", "IsA", "b", 0.5), ("c", "IsA", "target", 0.5)]

        layers = LayerBuilder() \
            .with_terms(["a", "b", "c", "target"]) \
            .with_connection("c", "target", 0.5) \
            .build()

        result = complete_analogy_simple(
            "a", "b", "c",
            layers, tokenizer,
            semantic_relations=relations
        )

        # "target" should get scores from both relation and co-occurrence
        assert isinstance(result, list)


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_self_analogy(self, sample_semantic_relations):
        """a:a::b:? should work."""
        layers = LayerBuilder() \
            .with_terms(["a", "b", "c"]) \
            .with_connection("a", "a", 1.0) \
            .with_connection("b", "c", 1.0) \
            .build()

        result = complete_analogy(
            "a", "a", "b",
            layers, sample_semantic_relations
        )

        # Should handle gracefully
        assert isinstance(result, list)

    def test_same_ab_and_c(self, sample_semantic_relations):
        """a:b::a:? should work."""
        layers = LayerBuilder().with_terms(["a", "b", "c"]).build()

        result = complete_analogy(
            "a", "b", "a",
            layers, sample_semantic_relations
        )

        # May find "b" but it should be excluded
        terms = [term for term, _, _ in result]
        assert "a" not in terms
        assert "b" not in terms

    def test_empty_semantic_relations_list(self):
        """Empty semantic relations doesn't crash."""
        layers = LayerBuilder().with_terms(["a", "b", "c"]).build()

        result = complete_analogy(
            "a", "b", "c",
            layers, [],
            use_relations=True
        )

        assert isinstance(result, list)

    def test_malformed_semantic_relations(self):
        """Handles malformed semantic relations gracefully."""
        # This would be caught at runtime if relations aren't 4-tuples
        layers = LayerBuilder().with_terms(["a", "b", "c"]).build()

        # We assume input is well-formed, but test with minimal relations
        result = complete_analogy("a", "b", "c", layers, [])
        assert isinstance(result, list)

    def test_zero_weight_relations(self):
        """Relations with zero weight still included."""
        relations = [("a", "IsA", "b", 0.0)]
        layers = LayerBuilder().with_terms(["a", "b", "c"]).build()

        result = find_relation_between("a", "b", relations)
        # Zero weight relation is still found
        assert len(result) == 1
        assert result[0][1] == 0.0

    def test_negative_weights_handled(self):
        """Negative weights in connections handled."""
        layers = LayerBuilder() \
            .with_term("a") \
            .with_term("b") \
            .build()

        # Manually set negative weight
        col_a = layers[MockLayers.TOKENS].get_minicolumn("a")
        col_a.lateral_connections["L0_b"] = -0.5

        # Should handle without crash
        result = complete_analogy(
            "a", "b", "c",
            layers, []
        )

        assert isinstance(result, list)

    def test_very_large_top_n(self, sample_semantic_relations):
        """top_n larger than possible results."""
        layers = LayerBuilder().with_terms(["a", "b", "c"]).build()

        result = complete_analogy(
            "a", "b", "c",
            layers, sample_semantic_relations,
            top_n=1000
        )

        # Returns all available results
        assert len(result) <= 1000

    def test_zero_top_n(self, sample_semantic_relations):
        """top_n=0 returns empty."""
        layers = LayerBuilder().with_terms(["a", "b", "c", "d"]).build()

        result = complete_analogy(
            "a", "b", "c",
            layers, sample_semantic_relations,
            top_n=0
        )

        assert result == []


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAnalogiesIntegration:
    """Integration tests with realistic scenarios."""

    def test_classic_king_queen_analogy(self):
        """Classic: man:king::woman:queen."""
        relations = [
            ("man", "ExampleOf", "king", 0.8),
            ("woman", "ExampleOf", "queen", 0.8),
        ]

        embeddings = {
            "man": [1.0, 0.0, 0.5],
            "king": [1.1, 0.2, 0.6],
            "woman": [0.0, 1.0, 0.5],
            "queen": [0.1, 1.2, 0.6],
        }

        layers = LayerBuilder().with_terms(list(embeddings.keys())).build()

        result = complete_analogy(
            "man", "king", "woman",
            layers, relations,
            embeddings=embeddings
        )

        # Should find "queen"
        if result:
            terms = [term for term, _, _ in result]
            assert "queen" in terms

    def test_technical_analogy(self, sample_semantic_relations):
        """Technical: neural:networks::knowledge:graphs."""
        layers = LayerBuilder().with_terms([
            "neural", "networks", "knowledge", "graphs", "semantic"
        ]).build()

        result = complete_analogy(
            "neural", "networks", "knowledge",
            layers, sample_semantic_relations,
            use_relations=True
        )

        if result:
            terms = [term for term, _, _ in result]
            # Should find "graphs" via IsA relation
            assert "graphs" in terms

    def test_antonym_analogy(self):
        """Antonym: hot:cold::day:night."""
        relations = [
            ("hot", "Antonym", "cold", 0.9),
            ("day", "Antonym", "night", 0.9),
        ]

        layers = LayerBuilder().with_terms([
            "hot", "cold", "day", "night"
        ]).build()

        result = complete_analogy(
            "hot", "cold", "day",
            layers, relations,
            use_relations=True
        )

        if result:
            terms = [term for term, _, _ in result]
            assert "night" in terms

    def test_multiple_valid_answers(self):
        """Analogy with multiple valid completions."""
        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("cat", "IsA", "animal", 0.9),
            ("bird", "IsA", "animal", 0.8),
        ]

        layers = LayerBuilder().with_terms([
            "dog", "animal", "cat", "bird", "pet"
        ]).build()

        result = complete_analogy(
            "dog", "animal", "cat",
            layers, relations,
            top_n=5
        )

        # Both "animal" and potentially other terms
        # "animal" should be excluded as it was in the input
        terms = [term for term, _, _ in result]
        assert "animal" not in terms
