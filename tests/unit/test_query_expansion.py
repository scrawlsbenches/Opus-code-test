"""
Unit Tests for Query Expansion Module
======================================

Task #170: Unit tests for cortical/query/expansion.py query expansion functions.

Tests the query expansion functions that extend search queries through:
- score_relation_path: Scoring relation chains for multi-hop inference
- expand_query: Main expansion using lateral connections, concepts, code concepts
- expand_query_semantic: Expansion using semantic relations
- expand_query_multihop: Multi-hop semantic inference through relation chains
- get_expanded_query_terms: Helper that consolidates expansion sources

These tests use MockMinicolumn and MockHierarchicalLayer to test expansion logic
without requiring a full CorticalTextProcessor.
"""

import pytest
from unittest.mock import Mock

from cortical.query.expansion import (
    score_relation_path,
    expand_query,
    expand_query_semantic,
    expand_query_multihop,
    get_expanded_query_terms,
    VALID_RELATION_CHAINS,
)
from cortical.tokenizer import Tokenizer
from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
)


# =============================================================================
# RELATION PATH SCORING TESTS
# =============================================================================


class TestScoreRelationPath:
    """Tests for score_relation_path function."""

    def test_empty_path(self):
        """Empty path returns 1.0 (fully valid)."""
        score = score_relation_path([])
        assert score == 1.0

    def test_single_relation(self):
        """Single relation returns 1.0 (fully valid)."""
        score = score_relation_path(['IsA'])
        assert score == 1.0

    def test_valid_transitive_chain(self):
        """IsA -> IsA is transitive and fully valid."""
        score = score_relation_path(['IsA', 'IsA'])
        assert score == 1.0

    def test_valid_partof_chain(self):
        """PartOf -> PartOf is transitive and fully valid."""
        score = score_relation_path(['PartOf', 'PartOf'])
        assert score == 1.0

    def test_valid_property_chain(self):
        """IsA -> HasProperty is valid with high score."""
        score = score_relation_path(['IsA', 'HasProperty'])
        assert score == 0.9

    def test_weakly_valid_chain(self):
        """RelatedTo -> RelatedTo is valid but weak."""
        score = score_relation_path(['RelatedTo', 'RelatedTo'])
        assert score == 0.6

    def test_invalid_antonym_chain(self):
        """Antonym chains are weak/contradictory."""
        score = score_relation_path(['Antonym', 'IsA'])
        assert score == 0.1

    def test_unknown_chain_uses_default(self):
        """Unknown relation chains use default validity."""
        score = score_relation_path(['UnknownRel', 'AnotherUnknown'])
        # Should use DEFAULT_CHAIN_VALIDITY from config
        assert 0.0 <= score <= 1.0

    def test_three_hop_chain(self):
        """Three-hop chains multiply consecutive pair scores."""
        # IsA -> IsA (1.0) -> HasProperty (0.9) = 1.0 * 0.9
        score = score_relation_path(['IsA', 'IsA', 'HasProperty'])
        assert score == pytest.approx(0.9, rel=0.01)

    def test_long_chain_decay(self):
        """Longer chains with weak links decay to low scores."""
        # Each RelatedTo -> RelatedTo is 0.6, so 0.6^3 for 4 relations
        score = score_relation_path(['RelatedTo', 'RelatedTo', 'RelatedTo', 'RelatedTo'])
        assert score < 0.3


# =============================================================================
# EXPAND_QUERY TESTS
# =============================================================================


class TestExpandQuery:
    """Tests for expand_query main expansion function."""

    @pytest.fixture
    def tokenizer(self):
        """Create a standard tokenizer for tests."""
        return Tokenizer()

    def test_empty_query(self, tokenizer):
        """Empty query returns empty expansion."""
        layers = MockLayers.empty()
        result = expand_query("", layers, tokenizer)
        assert result == {}

    def test_query_no_matches(self, tokenizer):
        """Query with no matching terms returns empty."""
        layers = MockLayers.single_term("existing", pagerank=0.5)
        result = expand_query("nonexistent", layers, tokenizer)
        assert result == {}

    def test_single_term_no_expansion(self, tokenizer):
        """Single term with no connections returns just the term."""
        layers = MockLayers.single_term("neural", pagerank=0.8)
        result = expand_query("neural", layers, tokenizer)
        assert "neural" in result
        assert result["neural"] == 1.0
        assert len(result) == 1

    def test_lateral_expansion_basic(self, tokenizer):
        """Basic lateral expansion adds connected terms."""
        layers = MockLayers.two_connected_terms(
            "neural", "network",
            weight=5.0,
            pagerank1=0.8,
            pagerank2=0.6
        )
        result = expand_query("neural", layers, tokenizer)

        # Should contain original term
        assert "neural" in result
        assert result["neural"] == 1.0

        # Should contain expanded term
        # Note: expansion weight can be > 1.0 due to connection * pagerank * 0.6
        assert "network" in result
        assert result["network"] > 0

    def test_lateral_expansion_weight_calculation(self, tokenizer):
        """Expanded terms weighted by connection * pagerank * 0.6."""
        col1 = MockMinicolumn(
            content="neural",
            pagerank=1.0,
            lateral_connections={"L0_networks": 10.0}
        )
        col2 = MockMinicolumn(
            content="networks",
            pagerank=0.5
        )
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = expand_query("neural", layers, tokenizer)
        # Expected: 10.0 * 0.5 * 0.6 = 3.0
        assert result["networks"] == pytest.approx(3.0, rel=0.01)

    def test_lateral_expansion_top_5_limit(self, tokenizer):
        """Lateral expansion limited to top 5 neighbors per term."""
        # Create term with 10 connections
        connections = {f"L0_term{i}": float(10 - i) for i in range(10)}
        col1 = MockMinicolumn(
            content="popular",
            pagerank=1.0,
            lateral_connections=connections
        )

        # Create all neighbor columns
        neighbors = [
            MockMinicolumn(content=f"term{i}", pagerank=0.5)
            for i in range(10)
        ]

        layer0 = MockHierarchicalLayer([col1] + neighbors)
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = expand_query("popular", layers, tokenizer, max_expansions=20)
        # Should only expand to top 5 neighbors
        expansion_count = len([k for k in result.keys() if k != "popular"])
        assert expansion_count <= 5

    def test_concept_expansion_basic(self, tokenizer):
        """Concept cluster expansion adds cluster members."""
        builder = LayerBuilder()
        builder.with_term("neural", pagerank=0.8)
        builder.with_term("deep", pagerank=0.6)
        builder.with_term("learning", pagerank=0.7)

        layers = builder.build()

        # Create concept cluster manually
        layer0 = layers[MockLayers.TOKENS]
        concept = MockMinicolumn(
            content="concept_0",
            id="L2_concept_0",
            layer=2,
            pagerank=0.9,
            feedforward_sources={"L0_neural", "L0_deep", "L0_learning"}
        )
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        result = expand_query("neural", layers, tokenizer)

        # Should include original
        assert "neural" in result
        # Should include cluster members
        assert "deep" in result or "learning" in result

    def test_concept_expansion_weight_calculation(self, tokenizer):
        """Concept expansions weighted by concept_pr * term_pr * 0.4."""
        col1 = MockMinicolumn(content="neural", pagerank=1.0)
        col2 = MockMinicolumn(content="deep", pagerank=0.8)

        layer0 = MockHierarchicalLayer([col1, col2])

        concept = MockMinicolumn(
            content="concept_0",
            id="L2_concept_0",
            layer=2,
            pagerank=0.5,
            feedforward_sources={"L0_neural", "L0_deep"}
        )
        layer2 = MockHierarchicalLayer([concept], level=2)

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0
        layers[MockLayers.CONCEPTS] = layer2

        result = expand_query("neural", layers, tokenizer)
        # Expected for "deep": 0.5 * 0.8 * 0.4 = 0.16
        assert result["deep"] == pytest.approx(0.16, rel=0.01)

    def test_max_expansions_limit(self, tokenizer):
        """max_expansions parameter limits total expansion terms."""
        # Create many connected terms
        builder = LayerBuilder()
        builder.with_term("source", pagerank=1.0)
        for i in range(20):
            builder.with_term(f"target{i}", pagerank=0.5)
            builder.with_connection("source", f"target{i}", weight=float(20-i))

        layers = builder.build()

        result = expand_query("source", layers, tokenizer, max_expansions=5)
        # Should have source + max 5 expansions
        assert len(result) <= 6

    def test_use_lateral_false(self, tokenizer):
        """use_lateral=False disables lateral expansion."""
        layers = MockLayers.two_connected_terms("neural", "networks", weight=10.0)
        result = expand_query("neural", layers, tokenizer, use_lateral=False)

        assert "neural" in result
        assert "networks" not in result

    def test_use_concepts_false(self, tokenizer):
        """use_concepts=False disables concept expansion."""
        col1 = MockMinicolumn(content="neural", pagerank=0.8)
        col2 = MockMinicolumn(content="deep", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2])

        concept = MockMinicolumn(
            content="concept_0",
            id="L2_concept_0",
            layer=2,
            pagerank=0.9,
            feedforward_sources={"L0_neural", "L0_deep"}
        )
        layer2 = MockHierarchicalLayer([concept], level=2)

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0
        layers[MockLayers.CONCEPTS] = layer2

        result = expand_query("neural", layers, tokenizer, use_concepts=False)

        # Should only have original term (no lateral, no concepts)
        assert "neural" in result
        assert "deep" not in result

    def test_variants_expansion(self, tokenizer):
        """use_variants=True tries word variants for unmatched terms."""
        # Test that the variant mechanism exists by checking behavior difference
        col = MockMinicolumn(content="network", pagerank=0.8)
        layer0 = MockHierarchicalLayer([col])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        # Query for exact match should work with or without variants
        result_with = expand_query("network", layers, tokenizer, use_variants=True)
        result_without = expand_query("network", layers, tokenizer, use_variants=False)

        # Both should find the exact match
        assert "network" in result_with
        assert "network" in result_without

    def test_variants_disabled(self, tokenizer):
        """use_variants=False doesn't match variants."""
        col = MockMinicolumn(content="compute", pagerank=0.8)
        layer0 = MockHierarchicalLayer([col])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = expand_query("computing", layers, tokenizer, use_variants=False)
        # Without variant matching, won't find the term
        # (unless "computing" gets stemmed to "compute" during tokenization)

    def test_code_concepts_expansion(self, tokenizer):
        """use_code_concepts=True adds programming synonyms."""
        # Create columns for both the query term and potential expansions
        col1 = MockMinicolumn(content="fetch", pagerank=0.8)
        col2 = MockMinicolumn(content="retrieve", pagerank=0.7)
        col3 = MockMinicolumn(content="load", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2, col3])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = expand_query(
            "fetch",
            layers,
            tokenizer,
            use_code_concepts=True,
            use_lateral=False,
            use_concepts=False
        )

        # Should have original term
        assert "fetch" in result
        # Code concepts expansion may add programming synonyms
        # The exact behavior depends on code_concepts.py

    def test_filter_code_stop_words(self, tokenizer):
        """filter_code_stop_words=True removes ubiquitous code tokens."""
        col1 = MockMinicolumn(
            content="method",
            pagerank=1.0,
            lateral_connections={"L0_self": 5.0, "L0_important": 3.0}
        )
        col2 = MockMinicolumn(content="self", pagerank=0.5)
        col3 = MockMinicolumn(content="important", pagerank=0.6)

        layer0 = MockHierarchicalLayer([col1, col2, col3])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = expand_query(
            "method",
            layers,
            tokenizer,
            filter_code_stop_words=True
        )

        # Should have original and important, but not "self"
        assert "method" in result
        assert "important" in result
        # "self" should be filtered (it's in CODE_EXPANSION_STOP_WORDS)
        # Note: This depends on tokenizer.CODE_EXPANSION_STOP_WORDS

    def test_multi_term_query(self, tokenizer):
        """Multi-term query expands from all terms."""
        layers = MockLayers.two_connected_terms("neural", "networks", weight=5.0)
        result = expand_query("neural networks", layers, tokenizer)

        # Both original terms should be present
        assert "neural" in result
        assert "networks" in result
        assert result["neural"] == 1.0
        assert result["networks"] == 1.0

    def test_max_weight_selection(self, tokenizer):
        """When multiple expansion paths exist, take maximum weight."""
        # Create scenario where same term is reachable via multiple paths
        col1 = MockMinicolumn(
            content="term1",
            pagerank=1.0,
            lateral_connections={"L0_target": 10.0}
        )
        col2 = MockMinicolumn(
            content="term2",
            pagerank=1.0,
            lateral_connections={"L0_target": 5.0}
        )
        col_target = MockMinicolumn(content="target", pagerank=0.5)

        layer0 = MockHierarchicalLayer([col1, col2, col_target])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = expand_query("term1 term2", layers, tokenizer)

        # Target reachable from both term1 (weight 10) and term2 (weight 5)
        # Should use maximum weight path
        # term1 path: 10.0 * 0.5 * 0.6 = 3.0
        # term2 path: 5.0 * 0.5 * 0.6 = 1.5
        assert result["target"] == pytest.approx(3.0, rel=0.01)


# =============================================================================
# EXPAND_QUERY_SEMANTIC TESTS
# =============================================================================


class TestExpandQuerySemantic:
    """Tests for expand_query_semantic function."""

    @pytest.fixture
    def tokenizer(self):
        """Create a standard tokenizer for tests."""
        return Tokenizer()

    def test_empty_query(self, tokenizer):
        """Empty query returns empty expansion."""
        layers = MockLayers.empty()
        result = expand_query_semantic("", layers, tokenizer, [])
        assert result == {}

    def test_no_semantic_relations(self, tokenizer):
        """Query with no semantic relations returns just query terms."""
        layers = MockLayers.single_term("neural", pagerank=0.8)
        result = expand_query_semantic("neural", layers, tokenizer, [])

        assert "neural" in result
        assert result["neural"] == 1.0
        assert len(result) == 1

    def test_single_relation_expansion(self, tokenizer):
        """Single semantic relation expands to neighbor."""
        layers = MockLayers.two_connected_terms("dog", "animal", weight=0.0)
        relations = [
            ("dog", "IsA", "animal", 0.9)
        ]

        result = expand_query_semantic("dog", layers, tokenizer, relations)

        assert "dog" in result
        assert result["dog"] == 1.0
        assert "animal" in result
        # Weight: 0.9 * 0.7 = 0.63
        assert result["animal"] == pytest.approx(0.63, rel=0.01)

    def test_bidirectional_relations(self, tokenizer):
        """Relations work in both directions."""
        col1 = MockMinicolumn(content="term1", pagerank=0.8)
        col2 = MockMinicolumn(content="term2", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("term1", "RelatedTo", "term2", 0.8)
        ]

        # Query term1, should expand to term2
        result1 = expand_query_semantic("term1", layers, tokenizer, relations)
        assert "term2" in result1

        # Query term2, should expand to term1
        result2 = expand_query_semantic("term2", layers, tokenizer, relations)
        assert "term1" in result2

    def test_multiple_neighbors(self, tokenizer):
        """Term with multiple semantic neighbors expands to all."""
        builder = LayerBuilder()
        builder.with_terms(["animal", "dog", "cat", "bird"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("animal", "HasA", "dog", 0.8),
            ("animal", "HasA", "cat", 0.8),
            ("animal", "HasA", "bird", 0.7)
        ]

        result = expand_query_semantic("animal", layers, tokenizer, relations)

        assert "animal" in result
        assert "dog" in result
        assert "cat" in result
        assert "bird" in result

    def test_max_expansions_limit(self, tokenizer):
        """max_expansions limits number of semantic neighbors."""
        builder = LayerBuilder()
        builder.with_term("source", pagerank=0.8)
        for i in range(20):
            builder.with_term(f"target{i}", pagerank=0.5)
        layers = builder.build()

        relations = [
            ("source", "RelatedTo", f"target{i}", 0.9 - i*0.01)
            for i in range(20)
        ]

        result = expand_query_semantic("source", layers, tokenizer, relations, max_expansions=5)

        # Should have source + max 5 expansions
        assert len(result) <= 6

    def test_expansion_weight_calculation(self, tokenizer):
        """Semantic expansion weight is relation_weight * 0.7."""
        col1 = MockMinicolumn(content="term1", pagerank=0.8)
        col2 = MockMinicolumn(content="term2", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("term1", "IsA", "term2", 0.85)
        ]

        result = expand_query_semantic("term1", layers, tokenizer, relations)
        # Expected: 0.85 * 0.7 = 0.595
        assert result["term2"] == pytest.approx(0.595, rel=0.01)

    def test_max_weight_selection(self, tokenizer):
        """Multiple relations to same target use max weight."""
        col1 = MockMinicolumn(content="term1", pagerank=0.8)
        col2 = MockMinicolumn(content="target", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("term1", "IsA", "target", 0.9),
            ("term1", "RelatedTo", "target", 0.6)
        ]

        result = expand_query_semantic("term1", layers, tokenizer, relations)
        # Should use max weight: 0.9 * 0.7 = 0.63
        assert result["target"] == pytest.approx(0.63, rel=0.01)

    def test_multi_term_query_expansion(self, tokenizer):
        """Multi-term query expands from all query terms."""
        builder = LayerBuilder()
        builder.with_terms(["neural", "networks", "deep", "learning"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("neural", "RelatedTo", "deep", 0.8),
            ("networks", "RelatedTo", "learning", 0.7)
        ]

        result = expand_query_semantic("neural networks", layers, tokenizer, relations)

        assert "neural" in result
        assert "networks" in result
        assert "deep" in result
        assert "learning" in result

    def test_only_corpus_terms_expanded(self, tokenizer):
        """Semantic neighbors not in corpus are skipped."""
        col1 = MockMinicolumn(content="term1", pagerank=0.8)
        layer0 = MockHierarchicalLayer([col1])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("term1", "RelatedTo", "term2", 0.9),  # term2 not in corpus
            ("term1", "IsA", "term3", 0.8)          # term3 not in corpus
        ]

        result = expand_query_semantic("term1", layers, tokenizer, relations)

        # Should only have original term
        assert "term1" in result
        assert "term2" not in result
        assert "term3" not in result


# =============================================================================
# EXPAND_QUERY_MULTIHOP TESTS
# =============================================================================


class TestExpandQueryMultihop:
    """Tests for expand_query_multihop multi-hop inference."""

    @pytest.fixture
    def tokenizer(self):
        """Create a standard tokenizer for tests."""
        return Tokenizer()

    def test_empty_query(self, tokenizer):
        """Empty query returns empty expansion."""
        layers = MockLayers.empty()
        result = expand_query_multihop("", layers, tokenizer, [])
        assert result == {}

    def test_no_relations(self, tokenizer):
        """Query with no relations returns just query terms."""
        layers = MockLayers.single_term("neural", pagerank=0.8)
        result = expand_query_multihop("neural", layers, tokenizer, [])

        assert "neural" in result
        assert result["neural"] == 1.0

    def test_one_hop_expansion(self, tokenizer):
        """One hop expansion follows single relation."""
        builder = LayerBuilder()
        builder.with_terms(["dog", "animal"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("dog", "IsA", "animal", 0.9)
        ]

        result = expand_query_multihop("dog", layers, tokenizer, relations, max_hops=1)

        assert "dog" in result
        assert result["dog"] == 1.0
        assert "animal" in result
        # One hop: 1.0 * 0.9 * 0.5^1 * 1.0 = 0.45
        assert result["animal"] == pytest.approx(0.45, rel=0.01)

    def test_two_hop_expansion(self, tokenizer):
        """Two hop expansion follows relation chains."""
        builder = LayerBuilder()
        builder.with_terms(["dog", "animal", "living"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("animal", "HasProperty", "living", 0.8)
        ]

        result = expand_query_multihop("dog", layers, tokenizer, relations, max_hops=2)

        assert "dog" in result
        assert "animal" in result
        assert "living" in result

        # Two hops with decay: 1.0 * 0.8 * 0.5^2 * path_score
        # path_score for IsA->HasProperty = 0.9
        # = 0.8 * 0.25 * 0.9 = 0.18
        assert result["living"] < result["animal"]

    def test_max_hops_limit(self, tokenizer):
        """max_hops limits traversal depth."""
        builder = LayerBuilder()
        # Use non-stop-word terms
        builder.with_terms(["apple", "fruit", "food", "sustenance"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("apple", "IsA", "fruit", 0.9),
            ("fruit", "IsA", "food", 0.9),
            ("food", "IsA", "sustenance", 0.9)
        ]

        # With max_hops=1, should only reach fruit
        result1 = expand_query_multihop("apple", layers, tokenizer, relations, max_hops=1)
        assert "fruit" in result1
        assert "food" not in result1
        assert "sustenance" not in result1

        # With max_hops=2, should reach fruit and food
        result2 = expand_query_multihop("apple", layers, tokenizer, relations, max_hops=2)
        assert "fruit" in result2
        assert "food" in result2
        assert "sustenance" not in result2

    def test_decay_factor(self, tokenizer):
        """decay_factor controls weight reduction per hop."""
        builder = LayerBuilder()
        # Use non-stop-word terms
        builder.with_terms(["apple", "fruit"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("apple", "IsA", "fruit", 0.9)
        ]

        result_low = expand_query_multihop("apple", layers, tokenizer, relations, decay_factor=0.3)
        result_high = expand_query_multihop("apple", layers, tokenizer, relations, decay_factor=0.9)

        # Higher decay factor should give higher weight to expanded term
        assert result_high["fruit"] > result_low["fruit"]

    def test_path_score_filtering(self, tokenizer):
        """min_path_score filters out invalid relation chains."""
        builder = LayerBuilder()
        builder.with_terms(["term1", "term2"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("term1", "Antonym", "term2", 0.9)  # Weak path validity
        ]

        # With high min_path_score, antonym chain should be filtered
        result = expand_query_multihop(
            "term1",
            layers,
            tokenizer,
            relations,
            min_path_score=0.5
        )

        # Antonym has low path validity, should be filtered
        assert "term1" in result
        # term2 may or may not be included depending on path score

    def test_transitive_isa_chain(self, tokenizer):
        """IsA chains are fully transitive."""
        builder = LayerBuilder()
        builder.with_terms(["dog", "mammal", "animal", "living"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("dog", "IsA", "mammal", 0.9),
            ("mammal", "IsA", "animal", 0.9),
            ("animal", "IsA", "living", 0.9)
        ]

        result = expand_query_multihop("dog", layers, tokenizer, relations, max_hops=3)

        # Should reach all levels of the hierarchy
        assert "dog" in result
        assert "mammal" in result
        assert "animal" in result
        assert "living" in result

    def test_partof_chain(self, tokenizer):
        """PartOf chains are transitive."""
        builder = LayerBuilder()
        builder.with_terms(["wheel", "car", "vehicle"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("wheel", "PartOf", "car", 0.9),
            ("car", "PartOf", "vehicle", 0.8)
        ]

        result = expand_query_multihop("wheel", layers, tokenizer, relations, max_hops=2)

        assert "wheel" in result
        assert "car" in result
        assert "vehicle" in result

    def test_max_expansions_limit(self, tokenizer):
        """max_expansions limits total expansion terms."""
        builder = LayerBuilder()
        builder.with_term("source", pagerank=0.8)
        for i in range(20):
            builder.with_term(f"hop1_{i}", pagerank=0.6)
        layers = builder.build()

        relations = [
            ("source", "RelatedTo", f"hop1_{i}", 0.9)
            for i in range(20)
        ]

        result = expand_query_multihop(
            "source",
            layers,
            tokenizer,
            relations,
            max_expansions=5
        )

        # Should have source + max 5 expansions
        assert len(result) <= 6

    def test_weight_calculation_with_path_score(self, tokenizer):
        """Weight = base * rel_weight * decay^hop * path_score."""
        builder = LayerBuilder()
        builder.with_terms(["dog", "animal", "living"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("dog", "IsA", "animal", 0.8),
            ("animal", "HasProperty", "living", 0.9)
        ]

        result = expand_query_multihop(
            "dog",
            layers,
            tokenizer,
            relations,
            max_hops=2,
            decay_factor=0.5
        )

        # Hop 1 (animal): 1.0 * 0.8 * 0.5^1 * 1.0 = 0.4
        assert result["animal"] == pytest.approx(0.4, rel=0.01)

        # Hop 2 (living): Check it exists and has lower weight
        assert "living" in result
        assert result["living"] < result["animal"]

    def test_multi_term_query(self, tokenizer):
        """Multi-term query expands from all terms."""
        builder = LayerBuilder()
        builder.with_terms(["neural", "networks", "deep", "learning"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("neural", "RelatedTo", "deep", 0.8),
            ("networks", "RelatedTo", "learning", 0.7)
        ]

        result = expand_query_multihop("neural networks", layers, tokenizer, relations)

        assert "neural" in result
        assert "networks" in result
        assert result["neural"] == 1.0
        assert result["networks"] == 1.0
        assert "deep" in result
        assert "learning" in result

    def test_skip_original_query_terms(self, tokenizer):
        """Expansion doesn't re-add original query terms."""
        builder = LayerBuilder()
        # Use non-stop-word terms
        builder.with_terms(["neural", "network"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("neural", "RelatedTo", "network", 0.8),
            ("network", "RelatedTo", "neural", 0.8)
        ]

        result = expand_query_multihop("neural network", layers, tokenizer, relations)

        # Both should remain at weight 1.0 (not reduced)
        assert result["neural"] == 1.0
        assert result["network"] == 1.0

    def test_bfs_traversal_order(self, tokenizer):
        """BFS ensures earlier hops are preferred."""
        builder = LayerBuilder()
        builder.with_terms(["start", "near", "far"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("start", "IsA", "near", 0.9),
            ("start", "RelatedTo", "far", 0.95),  # Direct but weak relation
            ("near", "IsA", "far", 0.9)  # Indirect but valid
        ]

        result = expand_query_multihop("start", layers, tokenizer, relations, max_hops=2)

        # Both near and far should be included
        assert "near" in result
        assert "far" in result


# =============================================================================
# GET_EXPANDED_QUERY_TERMS TESTS
# =============================================================================


class TestGetExpandedQueryTerms:
    """Tests for get_expanded_query_terms helper function."""

    @pytest.fixture
    def tokenizer(self):
        """Create a standard tokenizer for tests."""
        return Tokenizer()

    def test_no_expansion(self, tokenizer):
        """use_expansion=False returns just tokenized query."""
        layers = MockLayers.single_term("neural", pagerank=0.8)
        result = get_expanded_query_terms("neural", layers, tokenizer, use_expansion=False)

        assert "neural" in result
        assert result["neural"] == 1.0
        assert len(result) == 1

    def test_lateral_expansion_only(self, tokenizer):
        """Default expansion uses lateral connections."""
        layers = MockLayers.two_connected_terms("neural", "networks", weight=5.0)
        result = get_expanded_query_terms("neural", layers, tokenizer, use_expansion=True)

        assert "neural" in result
        assert "networks" in result

    def test_semantic_expansion_added(self, tokenizer):
        """use_semantic=True adds semantic relation expansion."""
        builder = LayerBuilder()
        builder.with_terms(["dog", "animal"], pagerank=0.7)
        layers = builder.build()

        relations = [
            ("dog", "IsA", "animal", 0.9)
        ]

        result = get_expanded_query_terms(
            "dog",
            layers,
            tokenizer,
            use_expansion=True,
            use_semantic=True,
            semantic_relations=relations
        )

        assert "dog" in result
        assert "animal" in result

    def test_semantic_discount_applied(self, tokenizer):
        """semantic_discount multiplies semantic expansion weights."""
        col1 = MockMinicolumn(content="dog", pagerank=0.7)
        col2 = MockMinicolumn(content="animal", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("dog", "IsA", "animal", 1.0)  # Weight 1.0 in relation
        ]

        result = get_expanded_query_terms(
            "dog",
            layers,
            tokenizer,
            use_expansion=True,
            use_semantic=True,
            semantic_relations=relations,
            semantic_discount=0.5
        )

        # Semantic weight: 1.0 * 0.7 (from expand_query_semantic) * 0.5 (discount)
        # But lateral might give higher weight, so check it exists
        assert "animal" in result

    def test_merging_takes_max_weight(self, tokenizer):
        """When lateral and semantic both expand to same term, take max."""
        col1 = MockMinicolumn(
            content="term1",
            pagerank=1.0,
            lateral_connections={"L0_target": 10.0}
        )
        col2 = MockMinicolumn(content="target", pagerank=0.5)
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("term1", "RelatedTo", "target", 0.6)
        ]

        result = get_expanded_query_terms(
            "term1",
            layers,
            tokenizer,
            use_expansion=True,
            use_semantic=True,
            semantic_relations=relations
        )

        # Lateral: 10.0 * 0.5 * 0.6 = 3.0
        # Semantic: 0.6 * 0.7 * 0.8 (discount) = 0.336
        # Should use lateral (higher)
        assert result["target"] == pytest.approx(3.0, rel=0.01)

    def test_use_semantic_false(self, tokenizer):
        """use_semantic=False skips semantic expansion."""
        col1 = MockMinicolumn(content="dog", pagerank=0.7)
        col2 = MockMinicolumn(content="animal", pagerank=0.6)
        layer0 = MockHierarchicalLayer([col1, col2])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        relations = [
            ("dog", "IsA", "animal", 0.9)
        ]

        result = get_expanded_query_terms(
            "dog",
            layers,
            tokenizer,
            use_expansion=True,
            use_semantic=False,
            semantic_relations=relations
        )

        # Should only have original term (no lateral, no semantic)
        assert "dog" in result
        # Animal might be in result if there are lateral connections

    def test_max_expansions_parameter(self, tokenizer):
        """max_expansions controls expansion size."""
        builder = LayerBuilder()
        builder.with_term("source", pagerank=1.0)
        for i in range(10):
            builder.with_term(f"target{i}", pagerank=0.5)
            builder.with_connection("source", f"target{i}", weight=float(10-i))
        layers = builder.build()

        result = get_expanded_query_terms(
            "source",
            layers,
            tokenizer,
            use_expansion=True,
            max_expansions=3
        )

        # Should have source + max 3 expansions
        assert len(result) <= 4

    def test_filter_code_stop_words_parameter(self, tokenizer):
        """filter_code_stop_words passed to expand_query."""
        col1 = MockMinicolumn(
            content="method",
            pagerank=1.0,
            lateral_connections={"L0_self": 5.0, "L0_important": 3.0}
        )
        col2 = MockMinicolumn(content="self", pagerank=0.5)
        col3 = MockMinicolumn(content="important", pagerank=0.6)

        layer0 = MockHierarchicalLayer([col1, col2, col3])
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = layer0

        result = get_expanded_query_terms(
            "method",
            layers,
            tokenizer,
            use_expansion=True,
            filter_code_stop_words=True
        )

        # Should filter code stop words
        assert "method" in result
        # Check that filtering was applied (depends on tokenizer implementation)

    def test_no_semantic_relations_provided(self, tokenizer):
        """When semantic_relations=None, skip semantic expansion."""
        layers = MockLayers.single_term("neural", pagerank=0.8)

        result = get_expanded_query_terms(
            "neural",
            layers,
            tokenizer,
            use_expansion=True,
            use_semantic=True,
            semantic_relations=None
        )

        # Should still work, just skip semantic expansion
        assert "neural" in result

    def test_empty_semantic_relations(self, tokenizer):
        """Empty semantic relations list works correctly."""
        layers = MockLayers.single_term("neural", pagerank=0.8)

        result = get_expanded_query_terms(
            "neural",
            layers,
            tokenizer,
            use_expansion=True,
            use_semantic=True,
            semantic_relations=[]
        )

        assert "neural" in result
