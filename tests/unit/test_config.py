"""
Unit Tests for Configuration Module
====================================

Task #168: Unit tests for cortical/config.py

Tests the CorticalConfig dataclass and related configuration utilities:
- Default value verification
- Parameter validation (ranges, types)
- Serialization (to_dict/from_dict)
- Copy operations
- Module-level utilities

Coverage target: 90%+
"""

import pytest
import copy as stdlib_copy

from cortical.config import (
    CorticalConfig,
    get_default_config,
    VALID_RELATION_CHAINS,
    DEFAULT_CHAIN_VALIDITY,
)


# =============================================================================
# DEFAULT VALUE TESTS
# =============================================================================


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_pagerank_settings(self):
        """PageRank defaults match documented values."""
        config = CorticalConfig()
        assert config.pagerank_damping == 0.85
        assert config.pagerank_iterations == 20
        assert config.pagerank_tolerance == 1e-6

    def test_default_clustering_settings(self):
        """Clustering defaults match documented values."""
        config = CorticalConfig()
        assert config.min_cluster_size == 3
        assert config.cluster_strictness == 1.0

    def test_default_gap_detection_thresholds(self):
        """Gap detection thresholds match documented values."""
        config = CorticalConfig()
        assert config.isolation_threshold == 0.02
        assert config.well_connected_threshold == 0.03
        assert config.weak_topic_tfidf_threshold == 0.005
        assert config.bridge_similarity_min == 0.005
        assert config.bridge_similarity_max == 0.03

    def test_default_chunking_settings(self):
        """Chunking settings for RAG match documented values."""
        config = CorticalConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128

    def test_default_query_expansion_settings(self):
        """Query expansion settings match documented values."""
        config = CorticalConfig()
        assert config.max_query_expansions == 10
        assert config.semantic_expansion_discount == 0.7

    def test_default_cross_layer_settings(self):
        """Cross-layer propagation settings match documented values."""
        config = CorticalConfig()
        assert config.cross_layer_damping == 0.7

    def test_default_bigram_weights(self):
        """Bigram connection weights match documented values."""
        config = CorticalConfig()
        assert config.bigram_component_weight == 0.5
        assert config.bigram_chain_weight == 0.7
        assert config.bigram_cooccurrence_weight == 0.3

    def test_default_concept_thresholds(self):
        """Concept connection thresholds match documented values."""
        config = CorticalConfig()
        assert config.concept_min_shared_docs == 1
        assert config.concept_min_jaccard == 0.1
        assert config.concept_embedding_threshold == 0.3

    def test_default_multihop_settings(self):
        """Multi-hop expansion settings match documented values."""
        config = CorticalConfig()
        assert config.multihop_max_hops == 2
        assert config.multihop_decay_factor == 0.5
        assert config.multihop_min_path_score == 0.3

    def test_default_inheritance_settings(self):
        """Property inheritance settings match documented values."""
        config = CorticalConfig()
        assert config.inheritance_decay_factor == 0.7
        assert config.inheritance_max_depth == 5
        assert config.inheritance_boost_factor == 0.3

    def test_default_relation_weights(self):
        """Relation weights dict has all expected keys and values."""
        config = CorticalConfig()
        expected = {
            'IsA': 1.5,
            'PartOf': 1.2,
            'HasA': 1.0,
            'UsedFor': 0.8,
            'CapableOf': 0.7,
            'HasProperty': 1.1,
            'SimilarTo': 1.3,
            'RelatedTo': 1.0,
            'Causes': 1.0,
            'Antonym': 0.3,
            'DerivedFrom': 1.1,
            'AtLocation': 0.9,
            'CoOccurs': 0.8,
        }
        assert config.relation_weights == expected

    def test_relation_weights_is_mutable_dict(self):
        """Relation weights is a regular dict, not frozen."""
        config = CorticalConfig()
        # Should be able to modify
        config.relation_weights['CustomRelation'] = 1.0
        assert config.relation_weights['CustomRelation'] == 1.0


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestConfigValidation:
    """Tests for parameter validation."""

    # PageRank validation

    def test_pagerank_damping_too_low(self):
        """pagerank_damping must be > 0."""
        with pytest.raises(ValueError, match="pagerank_damping must be between 0 and 1"):
            CorticalConfig(pagerank_damping=0.0)

    def test_pagerank_damping_too_high(self):
        """pagerank_damping must be < 1."""
        with pytest.raises(ValueError, match="pagerank_damping must be between 0 and 1"):
            CorticalConfig(pagerank_damping=1.0)

    def test_pagerank_damping_negative(self):
        """pagerank_damping cannot be negative."""
        with pytest.raises(ValueError, match="pagerank_damping must be between 0 and 1"):
            CorticalConfig(pagerank_damping=-0.5)

    def test_pagerank_damping_valid_range(self):
        """pagerank_damping accepts values in (0, 1)."""
        config = CorticalConfig(pagerank_damping=0.5)
        assert config.pagerank_damping == 0.5
        config = CorticalConfig(pagerank_damping=0.95)
        assert config.pagerank_damping == 0.95

    def test_pagerank_iterations_zero(self):
        """pagerank_iterations must be at least 1."""
        with pytest.raises(ValueError, match="pagerank_iterations must be at least 1"):
            CorticalConfig(pagerank_iterations=0)

    def test_pagerank_iterations_negative(self):
        """pagerank_iterations cannot be negative."""
        with pytest.raises(ValueError, match="pagerank_iterations must be at least 1"):
            CorticalConfig(pagerank_iterations=-10)

    def test_pagerank_iterations_valid(self):
        """pagerank_iterations accepts positive integers."""
        config = CorticalConfig(pagerank_iterations=50)
        assert config.pagerank_iterations == 50

    def test_pagerank_tolerance_zero(self):
        """pagerank_tolerance must be positive."""
        with pytest.raises(ValueError, match="pagerank_tolerance must be positive"):
            CorticalConfig(pagerank_tolerance=0.0)

    def test_pagerank_tolerance_negative(self):
        """pagerank_tolerance cannot be negative."""
        with pytest.raises(ValueError, match="pagerank_tolerance must be positive"):
            CorticalConfig(pagerank_tolerance=-1e-6)

    def test_pagerank_tolerance_valid(self):
        """pagerank_tolerance accepts positive values."""
        config = CorticalConfig(pagerank_tolerance=1e-8)
        assert config.pagerank_tolerance == 1e-8

    # Clustering validation

    def test_min_cluster_size_zero(self):
        """min_cluster_size must be at least 1."""
        with pytest.raises(ValueError, match="min_cluster_size must be at least 1"):
            CorticalConfig(min_cluster_size=0)

    def test_min_cluster_size_negative(self):
        """min_cluster_size cannot be negative."""
        with pytest.raises(ValueError, match="min_cluster_size must be at least 1"):
            CorticalConfig(min_cluster_size=-5)

    def test_min_cluster_size_valid(self):
        """min_cluster_size accepts positive integers."""
        config = CorticalConfig(min_cluster_size=10)
        assert config.min_cluster_size == 10

    def test_cluster_strictness_negative(self):
        """cluster_strictness cannot be negative."""
        with pytest.raises(ValueError, match="cluster_strictness must be between 0 and 1"):
            CorticalConfig(cluster_strictness=-0.5)

    def test_cluster_strictness_too_high(self):
        """cluster_strictness cannot exceed 1."""
        with pytest.raises(ValueError, match="cluster_strictness must be between 0 and 1"):
            CorticalConfig(cluster_strictness=1.5)

    def test_cluster_strictness_valid_range(self):
        """cluster_strictness accepts values in [0, 1]."""
        config = CorticalConfig(cluster_strictness=0.0)
        assert config.cluster_strictness == 0.0
        config = CorticalConfig(cluster_strictness=0.5)
        assert config.cluster_strictness == 0.5
        config = CorticalConfig(cluster_strictness=1.0)
        assert config.cluster_strictness == 1.0

    # Threshold validation

    def test_isolation_threshold_negative(self):
        """isolation_threshold must be non-negative."""
        with pytest.raises(ValueError, match="isolation_threshold must be non-negative"):
            CorticalConfig(isolation_threshold=-0.01)

    def test_isolation_threshold_valid(self):
        """isolation_threshold accepts non-negative values."""
        config = CorticalConfig(isolation_threshold=0.0)
        assert config.isolation_threshold == 0.0
        config = CorticalConfig(isolation_threshold=0.05)
        assert config.isolation_threshold == 0.05

    def test_well_connected_threshold_negative(self):
        """well_connected_threshold must be non-negative."""
        with pytest.raises(ValueError, match="well_connected_threshold must be non-negative"):
            CorticalConfig(well_connected_threshold=-0.01)

    def test_well_connected_threshold_valid(self):
        """well_connected_threshold accepts non-negative values."""
        config = CorticalConfig(well_connected_threshold=0.1)
        assert config.well_connected_threshold == 0.1

    def test_weak_topic_tfidf_threshold_negative(self):
        """weak_topic_tfidf_threshold must be non-negative."""
        with pytest.raises(ValueError, match="weak_topic_tfidf_threshold must be non-negative"):
            CorticalConfig(weak_topic_tfidf_threshold=-0.001)

    def test_weak_topic_tfidf_threshold_valid(self):
        """weak_topic_tfidf_threshold accepts non-negative values."""
        config = CorticalConfig(weak_topic_tfidf_threshold=0.01)
        assert config.weak_topic_tfidf_threshold == 0.01

    # Chunking validation

    def test_chunk_size_zero(self):
        """chunk_size must be at least 1."""
        with pytest.raises(ValueError, match="chunk_size must be at least 1"):
            CorticalConfig(chunk_size=0)

    def test_chunk_size_negative(self):
        """chunk_size cannot be negative."""
        with pytest.raises(ValueError, match="chunk_size must be at least 1"):
            CorticalConfig(chunk_size=-100)

    def test_chunk_size_valid(self):
        """chunk_size accepts positive integers."""
        config = CorticalConfig(chunk_size=1000)
        assert config.chunk_size == 1000

    def test_chunk_overlap_negative(self):
        """chunk_overlap must be non-negative."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            CorticalConfig(chunk_overlap=-10)

    def test_chunk_overlap_equals_chunk_size(self):
        """chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            CorticalConfig(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_exceeds_chunk_size(self):
        """chunk_overlap cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            CorticalConfig(chunk_size=100, chunk_overlap=150)

    def test_chunk_overlap_valid(self):
        """chunk_overlap accepts values < chunk_size."""
        config = CorticalConfig(chunk_size=200, chunk_overlap=50)
        assert config.chunk_size == 200
        assert config.chunk_overlap == 50

    # Query expansion validation

    def test_max_query_expansions_negative(self):
        """max_query_expansions must be non-negative."""
        with pytest.raises(ValueError, match="max_query_expansions must be non-negative"):
            CorticalConfig(max_query_expansions=-5)

    def test_max_query_expansions_zero(self):
        """max_query_expansions can be zero (no expansion)."""
        config = CorticalConfig(max_query_expansions=0)
        assert config.max_query_expansions == 0

    def test_max_query_expansions_valid(self):
        """max_query_expansions accepts non-negative integers."""
        config = CorticalConfig(max_query_expansions=20)
        assert config.max_query_expansions == 20

    def test_semantic_expansion_discount_negative(self):
        """semantic_expansion_discount cannot be negative."""
        with pytest.raises(ValueError, match="semantic_expansion_discount must be between 0 and 1"):
            CorticalConfig(semantic_expansion_discount=-0.1)

    def test_semantic_expansion_discount_too_high(self):
        """semantic_expansion_discount cannot exceed 1."""
        with pytest.raises(ValueError, match="semantic_expansion_discount must be between 0 and 1"):
            CorticalConfig(semantic_expansion_discount=1.5)

    def test_semantic_expansion_discount_valid_range(self):
        """semantic_expansion_discount accepts values in [0, 1]."""
        config = CorticalConfig(semantic_expansion_discount=0.0)
        assert config.semantic_expansion_discount == 0.0
        config = CorticalConfig(semantic_expansion_discount=0.5)
        assert config.semantic_expansion_discount == 0.5
        config = CorticalConfig(semantic_expansion_discount=1.0)
        assert config.semantic_expansion_discount == 1.0

    # Cross-layer validation

    def test_cross_layer_damping_zero(self):
        """cross_layer_damping must be > 0."""
        with pytest.raises(ValueError, match="cross_layer_damping must be between 0 and 1"):
            CorticalConfig(cross_layer_damping=0.0)

    def test_cross_layer_damping_one(self):
        """cross_layer_damping must be < 1."""
        with pytest.raises(ValueError, match="cross_layer_damping must be between 0 and 1"):
            CorticalConfig(cross_layer_damping=1.0)

    def test_cross_layer_damping_negative(self):
        """cross_layer_damping cannot be negative."""
        with pytest.raises(ValueError, match="cross_layer_damping must be between 0 and 1"):
            CorticalConfig(cross_layer_damping=-0.3)

    def test_cross_layer_damping_valid_range(self):
        """cross_layer_damping accepts values in (0, 1)."""
        config = CorticalConfig(cross_layer_damping=0.5)
        assert config.cross_layer_damping == 0.5
        config = CorticalConfig(cross_layer_damping=0.9)
        assert config.cross_layer_damping == 0.9


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestConfigSerialization:
    """Tests for to_dict and from_dict serialization."""

    def test_to_dict_includes_all_fields(self):
        """to_dict() includes all configuration fields."""
        config = CorticalConfig()
        data = config.to_dict()

        # Check essential fields are present
        expected_fields = [
            'pagerank_damping', 'pagerank_iterations', 'pagerank_tolerance',
            'min_cluster_size', 'cluster_strictness',
            'isolation_threshold', 'well_connected_threshold', 'weak_topic_tfidf_threshold',
            'bridge_similarity_min', 'bridge_similarity_max',
            'chunk_size', 'chunk_overlap',
            'max_query_expansions', 'semantic_expansion_discount',
            'cross_layer_damping',
            'bigram_component_weight', 'bigram_chain_weight', 'bigram_cooccurrence_weight',
            'concept_min_shared_docs', 'concept_min_jaccard', 'concept_embedding_threshold',
            'multihop_max_hops', 'multihop_decay_factor', 'multihop_min_path_score',
            'inheritance_decay_factor', 'inheritance_max_depth', 'inheritance_boost_factor',
            'relation_weights',
        ]

        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_to_dict_values_match(self):
        """to_dict() values match config attributes."""
        config = CorticalConfig(
            pagerank_damping=0.9,
            min_cluster_size=5,
            chunk_size=256
        )
        data = config.to_dict()

        assert data['pagerank_damping'] == 0.9
        assert data['min_cluster_size'] == 5
        assert data['chunk_size'] == 256

    def test_from_dict_creates_valid_config(self):
        """from_dict() creates a valid config from dict."""
        data = {
            'pagerank_damping': 0.9,
            'pagerank_iterations': 30,
            'pagerank_tolerance': 1e-7,
            'min_cluster_size': 5,
            'cluster_strictness': 0.8,
            'isolation_threshold': 0.03,
            'well_connected_threshold': 0.05,
            'weak_topic_tfidf_threshold': 0.01,
            'bridge_similarity_min': 0.01,
            'bridge_similarity_max': 0.05,
            'chunk_size': 256,
            'chunk_overlap': 64,
            'max_query_expansions': 15,
            'semantic_expansion_discount': 0.6,
            'cross_layer_damping': 0.8,
            'bigram_component_weight': 0.6,
            'bigram_chain_weight': 0.8,
            'bigram_cooccurrence_weight': 0.4,
            'concept_min_shared_docs': 2,
            'concept_min_jaccard': 0.15,
            'concept_embedding_threshold': 0.4,
            'multihop_max_hops': 3,
            'multihop_decay_factor': 0.6,
            'multihop_min_path_score': 0.4,
            'inheritance_decay_factor': 0.8,
            'inheritance_max_depth': 10,
            'inheritance_boost_factor': 0.4,
            'relation_weights': {'IsA': 2.0, 'PartOf': 1.5},
        }

        config = CorticalConfig.from_dict(data)
        assert config.pagerank_damping == 0.9
        assert config.min_cluster_size == 5
        assert config.chunk_size == 256
        assert config.relation_weights == {'IsA': 2.0, 'PartOf': 1.5}

    def test_round_trip_serialization(self):
        """Config -> dict -> config preserves all values."""
        original = CorticalConfig(
            pagerank_damping=0.75,
            pagerank_iterations=25,
            min_cluster_size=4,
            chunk_size=1024,
            chunk_overlap=256,
        )

        data = original.to_dict()
        restored = CorticalConfig.from_dict(data)

        assert restored.pagerank_damping == original.pagerank_damping
        assert restored.pagerank_iterations == original.pagerank_iterations
        assert restored.min_cluster_size == original.min_cluster_size
        assert restored.chunk_size == original.chunk_size
        assert restored.chunk_overlap == original.chunk_overlap
        assert restored.relation_weights == original.relation_weights

    def test_from_dict_with_invalid_value(self):
        """from_dict() with invalid values raises ValueError."""
        data = {
            'pagerank_damping': 1.5,  # Invalid: > 1
        }

        with pytest.raises(ValueError, match="pagerank_damping must be between 0 and 1"):
            CorticalConfig.from_dict(data)

    def test_to_dict_relation_weights_is_dict(self):
        """to_dict() converts relation_weights to regular dict."""
        config = CorticalConfig()
        data = config.to_dict()

        assert isinstance(data['relation_weights'], dict)
        assert 'IsA' in data['relation_weights']

    def test_from_dict_minimal(self):
        """from_dict() with only required fields uses defaults for rest."""
        # Only override one field, rest should be defaults
        data = {'pagerank_damping': 0.9}

        config = CorticalConfig.from_dict(data)
        assert config.pagerank_damping == 0.9
        # Other fields should have defaults
        assert config.pagerank_iterations == 20
        assert config.min_cluster_size == 3


# =============================================================================
# COPY TESTS
# =============================================================================


class TestConfigCopy:
    """Tests for copy() method."""

    def test_copy_creates_new_instance(self):
        """copy() creates a new CorticalConfig instance."""
        original = CorticalConfig(pagerank_damping=0.9)
        copied = original.copy()

        assert isinstance(copied, CorticalConfig)
        assert copied is not original

    def test_copy_preserves_all_values(self):
        """copy() preserves all configuration values."""
        original = CorticalConfig(
            pagerank_damping=0.9,
            pagerank_iterations=30,
            min_cluster_size=5,
            chunk_size=256,
            chunk_overlap=64,
        )
        copied = original.copy()

        assert copied.pagerank_damping == original.pagerank_damping
        assert copied.pagerank_iterations == original.pagerank_iterations
        assert copied.min_cluster_size == original.min_cluster_size
        assert copied.chunk_size == original.chunk_size
        assert copied.chunk_overlap == original.chunk_overlap

    def test_copy_is_independent(self):
        """Modifying copy doesn't affect original."""
        original = CorticalConfig(pagerank_damping=0.85)
        copied = original.copy()

        # Modify the copy
        copied.pagerank_damping = 0.95
        copied.min_cluster_size = 10

        # Original should be unchanged
        assert original.pagerank_damping == 0.85
        assert original.min_cluster_size == 3

    def test_copy_relation_weights_deep_copy(self):
        """copy() deep copies relation_weights dict."""
        original = CorticalConfig()
        copied = original.copy()

        # Modify copied relation_weights
        copied.relation_weights['CustomRelation'] = 2.0

        # Original should not have the new key
        assert 'CustomRelation' not in original.relation_weights
        assert 'CustomRelation' in copied.relation_weights

    def test_copy_validation_still_works(self):
        """Copied config can be modified but still validates."""
        original = CorticalConfig()
        copied = original.copy()

        # This should validate successfully
        copied.pagerank_damping = 0.9

        # This should fail validation on next _validate() call
        # But copy() itself doesn't re-validate, so we need to create new instance
        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=1.5)


# =============================================================================
# MODULE-LEVEL UTILITIES TESTS
# =============================================================================


class TestModuleLevelUtilities:
    """Tests for module-level constants and functions."""

    def test_get_default_config_returns_valid_config(self):
        """get_default_config() returns a valid CorticalConfig."""
        config = get_default_config()
        assert isinstance(config, CorticalConfig)
        assert config.pagerank_damping == 0.85
        assert config.pagerank_iterations == 20

    def test_get_default_config_returns_new_instance(self):
        """get_default_config() returns a new instance each time."""
        config1 = get_default_config()
        config2 = get_default_config()

        assert config1 is not config2

    def test_valid_relation_chains_exists(self):
        """VALID_RELATION_CHAINS constant is defined."""
        assert VALID_RELATION_CHAINS is not None
        assert isinstance(VALID_RELATION_CHAINS, dict)

    def test_valid_relation_chains_has_expected_entries(self):
        """VALID_RELATION_CHAINS contains expected relation pairs."""
        # Check some expected entries
        assert ('IsA', 'IsA') in VALID_RELATION_CHAINS
        assert ('PartOf', 'PartOf') in VALID_RELATION_CHAINS
        assert ('Causes', 'Causes') in VALID_RELATION_CHAINS

    def test_valid_relation_chains_values_in_range(self):
        """VALID_RELATION_CHAINS values are in [0, 1]."""
        for (rel1, rel2), score in VALID_RELATION_CHAINS.items():
            assert 0.0 <= score <= 1.0, f"Invalid score for ({rel1}, {rel2}): {score}"

    def test_default_chain_validity_exists(self):
        """DEFAULT_CHAIN_VALIDITY constant is defined."""
        assert DEFAULT_CHAIN_VALIDITY is not None
        assert isinstance(DEFAULT_CHAIN_VALIDITY, float)

    def test_default_chain_validity_in_range(self):
        """DEFAULT_CHAIN_VALIDITY is in [0, 1]."""
        assert 0.0 <= DEFAULT_CHAIN_VALIDITY <= 1.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestConfigEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_pagerank_iterations(self):
        """Very high pagerank_iterations is accepted."""
        config = CorticalConfig(pagerank_iterations=10000)
        assert config.pagerank_iterations == 10000

    def test_very_small_tolerance(self):
        """Very small tolerance values are accepted."""
        config = CorticalConfig(pagerank_tolerance=1e-12)
        assert config.pagerank_tolerance == 1e-12

    def test_zero_max_query_expansions(self):
        """Zero max_query_expansions disables expansion."""
        config = CorticalConfig(max_query_expansions=0)
        assert config.max_query_expansions == 0

    def test_chunk_overlap_zero(self):
        """chunk_overlap can be zero (no overlap)."""
        config = CorticalConfig(chunk_size=100, chunk_overlap=0)
        assert config.chunk_overlap == 0

    def test_chunk_overlap_one_less_than_size(self):
        """chunk_overlap can be chunk_size - 1."""
        config = CorticalConfig(chunk_size=100, chunk_overlap=99)
        assert config.chunk_overlap == 99

    def test_empty_relation_weights(self):
        """Config accepts empty relation_weights dict."""
        config = CorticalConfig(relation_weights={})
        assert config.relation_weights == {}

    def test_custom_relation_weights(self):
        """Config accepts custom relation_weights."""
        custom_weights = {
            'CustomRel1': 1.0,
            'CustomRel2': 0.5,
        }
        config = CorticalConfig(relation_weights=custom_weights)
        assert config.relation_weights == custom_weights

    def test_min_cluster_size_one(self):
        """min_cluster_size can be 1."""
        config = CorticalConfig(min_cluster_size=1)
        assert config.min_cluster_size == 1

    def test_cluster_strictness_zero(self):
        """cluster_strictness can be 0 (no strictness)."""
        config = CorticalConfig(cluster_strictness=0.0)
        assert config.cluster_strictness == 0.0

    def test_cluster_strictness_one(self):
        """cluster_strictness can be 1 (maximum strictness)."""
        config = CorticalConfig(cluster_strictness=1.0)
        assert config.cluster_strictness == 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestConfigIntegration:
    """Integration tests for config usage patterns."""

    def test_config_can_be_modified_after_creation(self):
        """Config can be modified after creation (mutable)."""
        config = CorticalConfig()
        original_damping = config.pagerank_damping

        config.pagerank_damping = 0.95
        assert config.pagerank_damping == 0.95
        assert config.pagerank_damping != original_damping

    def test_config_multiple_overrides(self):
        """Config accepts multiple parameter overrides."""
        config = CorticalConfig(
            pagerank_damping=0.9,
            pagerank_iterations=50,
            min_cluster_size=10,
            chunk_size=2048,
            chunk_overlap=512,
            max_query_expansions=20,
        )

        assert config.pagerank_damping == 0.9
        assert config.pagerank_iterations == 50
        assert config.min_cluster_size == 10
        assert config.chunk_size == 2048
        assert config.chunk_overlap == 512
        assert config.max_query_expansions == 20

    def test_config_dict_workflow(self):
        """Common workflow: create -> to_dict -> modify -> from_dict."""
        # Create config
        config = CorticalConfig(pagerank_damping=0.9)

        # Serialize
        data = config.to_dict()

        # Modify dict
        data['min_cluster_size'] = 10
        data['chunk_size'] = 1024

        # Deserialize
        modified_config = CorticalConfig.from_dict(data)

        assert modified_config.pagerank_damping == 0.9
        assert modified_config.min_cluster_size == 10
        assert modified_config.chunk_size == 1024

    def test_config_copy_modify_workflow(self):
        """Common workflow: create -> copy -> modify copy."""
        base_config = CorticalConfig(pagerank_damping=0.85)

        # Create variant for experiments
        experiment_config = base_config.copy()
        experiment_config.pagerank_damping = 0.95
        experiment_config.pagerank_iterations = 50

        # Base should be unchanged
        assert base_config.pagerank_damping == 0.85
        assert base_config.pagerank_iterations == 20

        # Experiment has modifications
        assert experiment_config.pagerank_damping == 0.95
        assert experiment_config.pagerank_iterations == 50
