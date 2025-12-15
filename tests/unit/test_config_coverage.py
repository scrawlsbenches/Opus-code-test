"""
Additional Coverage Tests for Configuration Module
===================================================

Comprehensive unit tests targeting uncovered edge cases and validation logic
in cortical/config.py to improve coverage from ~67% to >80%.

Focus areas:
- Louvain resolution special cases (NaN, inf, warning thresholds)
- BM25 parameter boundary values
- Bridge similarity threshold validation
- Additional validation for new parameters (bigram, concept, multihop, inheritance)
- Complex validation interactions
- Error message verification
"""

import math
import warnings
import pytest

from cortical.config import CorticalConfig, get_default_config


# =============================================================================
# LOUVAIN RESOLUTION VALIDATION TESTS
# =============================================================================


class TestLouvainResolutionValidation:
    """Tests for louvain_resolution special cases and boundary conditions."""

    def test_louvain_resolution_default(self):
        """louvain_resolution has correct default value."""
        config = CorticalConfig()
        assert config.louvain_resolution == 2.0

    def test_louvain_resolution_valid_low(self):
        """louvain_resolution accepts small positive values."""
        config = CorticalConfig(louvain_resolution=0.1)
        assert config.louvain_resolution == 0.1

    def test_louvain_resolution_valid_high(self):
        """louvain_resolution accepts high values up to 20."""
        config = CorticalConfig(louvain_resolution=10.0)
        assert config.louvain_resolution == 10.0

    def test_louvain_resolution_zero(self):
        """louvain_resolution must be positive (> 0)."""
        with pytest.raises(ValueError, match="louvain_resolution must be positive"):
            CorticalConfig(louvain_resolution=0.0)

    def test_louvain_resolution_negative(self):
        """louvain_resolution cannot be negative."""
        with pytest.raises(ValueError, match="louvain_resolution must be positive"):
            CorticalConfig(louvain_resolution=-1.0)

    def test_louvain_resolution_nan(self):
        """louvain_resolution rejects NaN values."""
        with pytest.raises(ValueError, match="louvain_resolution must be a finite number"):
            CorticalConfig(louvain_resolution=float('nan'))

    def test_louvain_resolution_positive_inf(self):
        """louvain_resolution rejects positive infinity."""
        with pytest.raises(ValueError, match="louvain_resolution must be a finite number"):
            CorticalConfig(louvain_resolution=float('inf'))

    def test_louvain_resolution_negative_inf(self):
        """louvain_resolution rejects negative infinity."""
        with pytest.raises(ValueError, match="louvain_resolution must be a finite number"):
            CorticalConfig(louvain_resolution=float('-inf'))

    def test_louvain_resolution_very_high_warning(self):
        """louvain_resolution > 20 triggers warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = CorticalConfig(louvain_resolution=25.0)

            # Should have created a warning
            assert len(w) == 1
            assert "very high" in str(w[0].message).lower()
            assert "25" in str(w[0].message)

            # But config should still be created
            assert config.louvain_resolution == 25.0

    def test_louvain_resolution_at_warning_boundary(self):
        """louvain_resolution = 20.0 does not trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = CorticalConfig(louvain_resolution=20.0)

            # Should NOT warn at exactly 20.0
            assert len(w) == 0
            assert config.louvain_resolution == 20.0

    def test_louvain_resolution_just_above_warning_boundary(self):
        """louvain_resolution = 20.1 triggers warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = CorticalConfig(louvain_resolution=20.1)

            # Should warn just above 20.0
            assert len(w) == 1
            assert config.louvain_resolution == 20.1


# =============================================================================
# BM25 PARAMETER VALIDATION TESTS
# =============================================================================


class TestBM25ParameterValidation:
    """Tests for BM25 scoring algorithm parameters."""

    def test_scoring_algorithm_default(self):
        """scoring_algorithm defaults to 'bm25'."""
        config = CorticalConfig()
        assert config.scoring_algorithm == 'bm25'

    def test_scoring_algorithm_tfidf(self):
        """scoring_algorithm accepts 'tfidf'."""
        config = CorticalConfig(scoring_algorithm='tfidf')
        assert config.scoring_algorithm == 'tfidf'

    def test_scoring_algorithm_bm25(self):
        """scoring_algorithm accepts 'bm25'."""
        config = CorticalConfig(scoring_algorithm='bm25')
        assert config.scoring_algorithm == 'bm25'

    def test_scoring_algorithm_invalid(self):
        """scoring_algorithm rejects invalid values."""
        with pytest.raises(ValueError, match="scoring_algorithm must be 'tfidf' or 'bm25'"):
            CorticalConfig(scoring_algorithm='invalid')

    def test_scoring_algorithm_case_sensitive(self):
        """scoring_algorithm is case-sensitive."""
        with pytest.raises(ValueError, match="scoring_algorithm must be 'tfidf' or 'bm25'"):
            CorticalConfig(scoring_algorithm='BM25')

        with pytest.raises(ValueError, match="scoring_algorithm must be 'tfidf' or 'bm25'"):
            CorticalConfig(scoring_algorithm='TFIDF')

    def test_bm25_k1_default(self):
        """bm25_k1 has correct default value."""
        config = CorticalConfig()
        assert config.bm25_k1 == 1.2

    def test_bm25_k1_boundary_zero(self):
        """bm25_k1 accepts 0 (lower boundary)."""
        config = CorticalConfig(bm25_k1=0.0)
        assert config.bm25_k1 == 0.0

    def test_bm25_k1_boundary_three(self):
        """bm25_k1 accepts 3 (upper boundary)."""
        config = CorticalConfig(bm25_k1=3.0)
        assert config.bm25_k1 == 3.0

    def test_bm25_k1_negative(self):
        """bm25_k1 cannot be negative."""
        with pytest.raises(ValueError, match="bm25_k1 must be between 0 and 3"):
            CorticalConfig(bm25_k1=-0.5)

    def test_bm25_k1_too_high(self):
        """bm25_k1 cannot exceed 3."""
        with pytest.raises(ValueError, match="bm25_k1 must be between 0 and 3"):
            CorticalConfig(bm25_k1=3.5)

    def test_bm25_k1_valid_range(self):
        """bm25_k1 accepts typical values in range."""
        config = CorticalConfig(bm25_k1=1.5)
        assert config.bm25_k1 == 1.5
        config = CorticalConfig(bm25_k1=2.0)
        assert config.bm25_k1 == 2.0

    def test_bm25_b_default(self):
        """bm25_b has correct default value."""
        config = CorticalConfig()
        assert config.bm25_b == 0.75

    def test_bm25_b_boundary_zero(self):
        """bm25_b accepts 0 (no length normalization)."""
        config = CorticalConfig(bm25_b=0.0)
        assert config.bm25_b == 0.0

    def test_bm25_b_boundary_one(self):
        """bm25_b accepts 1 (full length normalization)."""
        config = CorticalConfig(bm25_b=1.0)
        assert config.bm25_b == 1.0

    def test_bm25_b_negative(self):
        """bm25_b cannot be negative."""
        with pytest.raises(ValueError, match="bm25_b must be between 0 and 1"):
            CorticalConfig(bm25_b=-0.1)

    def test_bm25_b_too_high(self):
        """bm25_b cannot exceed 1."""
        with pytest.raises(ValueError, match="bm25_b must be between 0 and 1"):
            CorticalConfig(bm25_b=1.5)

    def test_bm25_b_valid_range(self):
        """bm25_b accepts values in [0, 1]."""
        config = CorticalConfig(bm25_b=0.5)
        assert config.bm25_b == 0.5


# =============================================================================
# BRIDGE SIMILARITY THRESHOLD TESTS
# =============================================================================


class TestBridgeSimilarityThresholds:
    """Tests for bridge_similarity_min and bridge_similarity_max."""

    def test_bridge_similarity_min_default(self):
        """bridge_similarity_min has correct default."""
        config = CorticalConfig()
        assert config.bridge_similarity_min == 0.005

    def test_bridge_similarity_max_default(self):
        """bridge_similarity_max has correct default."""
        config = CorticalConfig()
        assert config.bridge_similarity_max == 0.03

    def test_bridge_similarity_min_valid(self):
        """bridge_similarity_min accepts valid values."""
        config = CorticalConfig(bridge_similarity_min=0.01)
        assert config.bridge_similarity_min == 0.01

    def test_bridge_similarity_max_valid(self):
        """bridge_similarity_max accepts valid values."""
        config = CorticalConfig(bridge_similarity_max=0.05)
        assert config.bridge_similarity_max == 0.05

    def test_bridge_similarity_custom_range(self):
        """bridge_similarity_min and max can be customized together."""
        config = CorticalConfig(
            bridge_similarity_min=0.01,
            bridge_similarity_max=0.1
        )
        assert config.bridge_similarity_min == 0.01
        assert config.bridge_similarity_max == 0.1


# =============================================================================
# BIGRAM WEIGHT VALIDATION TESTS
# =============================================================================


class TestBigramWeightValidation:
    """Tests for bigram connection weight parameters."""

    def test_bigram_component_weight_default(self):
        """bigram_component_weight has correct default."""
        config = CorticalConfig()
        assert config.bigram_component_weight == 0.5

    def test_bigram_chain_weight_default(self):
        """bigram_chain_weight has correct default."""
        config = CorticalConfig()
        assert config.bigram_chain_weight == 0.7

    def test_bigram_cooccurrence_weight_default(self):
        """bigram_cooccurrence_weight has correct default."""
        config = CorticalConfig()
        assert config.bigram_cooccurrence_weight == 0.3

    def test_bigram_weights_accept_zero(self):
        """Bigram weights accept zero values."""
        config = CorticalConfig(
            bigram_component_weight=0.0,
            bigram_chain_weight=0.0,
            bigram_cooccurrence_weight=0.0
        )
        assert config.bigram_component_weight == 0.0
        assert config.bigram_chain_weight == 0.0
        assert config.bigram_cooccurrence_weight == 0.0

    def test_bigram_weights_accept_large_values(self):
        """Bigram weights accept values > 1."""
        config = CorticalConfig(
            bigram_component_weight=2.0,
            bigram_chain_weight=3.0,
            bigram_cooccurrence_weight=1.5
        )
        assert config.bigram_component_weight == 2.0
        assert config.bigram_chain_weight == 3.0
        assert config.bigram_cooccurrence_weight == 1.5


# =============================================================================
# CONCEPT THRESHOLD VALIDATION TESTS
# =============================================================================


class TestConceptThresholdValidation:
    """Tests for concept connection threshold parameters."""

    def test_concept_min_shared_docs_default(self):
        """concept_min_shared_docs has correct default."""
        config = CorticalConfig()
        assert config.concept_min_shared_docs == 1

    def test_concept_min_jaccard_default(self):
        """concept_min_jaccard has correct default."""
        config = CorticalConfig()
        assert config.concept_min_jaccard == 0.1

    def test_concept_embedding_threshold_default(self):
        """concept_embedding_threshold has correct default."""
        config = CorticalConfig()
        assert config.concept_embedding_threshold == 0.3

    def test_concept_min_shared_docs_accepts_positive(self):
        """concept_min_shared_docs accepts positive integers."""
        config = CorticalConfig(concept_min_shared_docs=5)
        assert config.concept_min_shared_docs == 5

    def test_concept_min_jaccard_accepts_range(self):
        """concept_min_jaccard accepts values in [0, 1]."""
        config = CorticalConfig(concept_min_jaccard=0.5)
        assert config.concept_min_jaccard == 0.5

    def test_concept_embedding_threshold_accepts_range(self):
        """concept_embedding_threshold accepts values in [0, 1]."""
        config = CorticalConfig(concept_embedding_threshold=0.7)
        assert config.concept_embedding_threshold == 0.7


# =============================================================================
# MULTIHOP EXPANSION VALIDATION TESTS
# =============================================================================


class TestMultihopExpansionValidation:
    """Tests for multi-hop expansion parameters."""

    def test_multihop_max_hops_default(self):
        """multihop_max_hops has correct default."""
        config = CorticalConfig()
        assert config.multihop_max_hops == 2

    def test_multihop_decay_factor_default(self):
        """multihop_decay_factor has correct default."""
        config = CorticalConfig()
        assert config.multihop_decay_factor == 0.5

    def test_multihop_min_path_score_default(self):
        """multihop_min_path_score has correct default."""
        config = CorticalConfig()
        assert config.multihop_min_path_score == 0.3

    def test_multihop_max_hops_accepts_positive(self):
        """multihop_max_hops accepts positive integers."""
        config = CorticalConfig(multihop_max_hops=5)
        assert config.multihop_max_hops == 5

    def test_multihop_max_hops_one(self):
        """multihop_max_hops can be 1 (single hop)."""
        config = CorticalConfig(multihop_max_hops=1)
        assert config.multihop_max_hops == 1

    def test_multihop_decay_factor_accepts_range(self):
        """multihop_decay_factor accepts values in [0, 1]."""
        config = CorticalConfig(multihop_decay_factor=0.8)
        assert config.multihop_decay_factor == 0.8

    def test_multihop_min_path_score_accepts_range(self):
        """multihop_min_path_score accepts values in [0, 1]."""
        config = CorticalConfig(multihop_min_path_score=0.5)
        assert config.multihop_min_path_score == 0.5


# =============================================================================
# INHERITANCE VALIDATION TESTS
# =============================================================================


class TestInheritanceValidation:
    """Tests for property inheritance parameters."""

    def test_inheritance_decay_factor_default(self):
        """inheritance_decay_factor has correct default."""
        config = CorticalConfig()
        assert config.inheritance_decay_factor == 0.7

    def test_inheritance_max_depth_default(self):
        """inheritance_max_depth has correct default."""
        config = CorticalConfig()
        assert config.inheritance_max_depth == 5

    def test_inheritance_boost_factor_default(self):
        """inheritance_boost_factor has correct default."""
        config = CorticalConfig()
        assert config.inheritance_boost_factor == 0.3

    def test_inheritance_decay_factor_accepts_range(self):
        """inheritance_decay_factor accepts values in [0, 1]."""
        config = CorticalConfig(inheritance_decay_factor=0.9)
        assert config.inheritance_decay_factor == 0.9

    def test_inheritance_max_depth_accepts_positive(self):
        """inheritance_max_depth accepts positive integers."""
        config = CorticalConfig(inheritance_max_depth=10)
        assert config.inheritance_max_depth == 10

    def test_inheritance_boost_factor_accepts_range(self):
        """inheritance_boost_factor accepts values in [0, 1]."""
        config = CorticalConfig(inheritance_boost_factor=0.5)
        assert config.inheritance_boost_factor == 0.5


# =============================================================================
# COMPLEX VALIDATION INTERACTION TESTS
# =============================================================================


class TestComplexValidationInteractions:
    """Tests for complex interactions between multiple parameters."""

    def test_all_parameters_at_boundaries(self):
        """Config accepts all parameters at their boundary values."""
        config = CorticalConfig(
            pagerank_damping=0.01,  # Near 0
            pagerank_iterations=1,  # Minimum
            pagerank_tolerance=1e-12,  # Very small
            min_cluster_size=1,  # Minimum
            cluster_strictness=0.0,  # Minimum
            louvain_resolution=0.01,  # Near 0
            chunk_size=1,  # Minimum
            chunk_overlap=0,  # Minimum
            max_query_expansions=0,  # Disabled
            semantic_expansion_discount=0.0,  # No discount
            cross_layer_damping=0.01,  # Near 0
            bm25_k1=0.0,  # Minimum
            bm25_b=0.0,  # Minimum
        )
        assert config.pagerank_damping == 0.01
        assert config.min_cluster_size == 1
        assert config.chunk_overlap == 0

    def test_all_parameters_at_upper_boundaries(self):
        """Config accepts all parameters at their upper boundaries."""
        config = CorticalConfig(
            pagerank_damping=0.99,  # Near 1
            pagerank_iterations=10000,  # Very high
            cluster_strictness=1.0,  # Maximum
            chunk_size=10000,
            chunk_overlap=9999,  # One less than chunk_size
            max_query_expansions=1000,  # Very high
            semantic_expansion_discount=1.0,  # Full discount
            cross_layer_damping=0.99,  # Near 1
            bm25_k1=3.0,  # Maximum
            bm25_b=1.0,  # Maximum
        )
        assert config.pagerank_damping == 0.99
        assert config.cluster_strictness == 1.0
        assert config.chunk_overlap == 9999

    def test_mixed_custom_and_default_values(self):
        """Config handles mix of custom and default values."""
        config = CorticalConfig(
            pagerank_damping=0.9,
            chunk_size=256,
            # Other parameters use defaults
        )
        assert config.pagerank_damping == 0.9
        assert config.chunk_size == 256
        # Verify defaults are still applied
        assert config.pagerank_iterations == 20
        assert config.min_cluster_size == 3

    def test_serialization_preserves_boundary_values(self):
        """Boundary values survive serialization round trip."""
        original = CorticalConfig(
            pagerank_damping=0.01,
            cluster_strictness=0.0,
            bm25_k1=3.0,
            bm25_b=1.0,
        )

        # Round trip
        data = original.to_dict()
        restored = CorticalConfig.from_dict(data)

        assert restored.pagerank_damping == 0.01
        assert restored.cluster_strictness == 0.0
        assert restored.bm25_k1 == 3.0
        assert restored.bm25_b == 1.0

    def test_copy_preserves_boundary_values(self):
        """Boundary values survive copy operation."""
        original = CorticalConfig(
            pagerank_damping=0.99,
            louvain_resolution=0.01,
            bm25_k1=0.0,
        )

        copied = original.copy()

        assert copied.pagerank_damping == 0.99
        assert copied.louvain_resolution == 0.01
        assert copied.bm25_k1 == 0.0


# =============================================================================
# ERROR MESSAGE VERIFICATION TESTS
# =============================================================================


class TestErrorMessageVerification:
    """Tests that error messages are clear and informative."""

    def test_pagerank_damping_error_includes_value(self):
        """pagerank_damping error includes the invalid value."""
        with pytest.raises(ValueError) as exc_info:
            CorticalConfig(pagerank_damping=1.5)
        assert "1.5" in str(exc_info.value)

    def test_louvain_resolution_zero_error_message(self):
        """louvain_resolution zero error is clear."""
        with pytest.raises(ValueError) as exc_info:
            CorticalConfig(louvain_resolution=0.0)
        assert "positive" in str(exc_info.value).lower()
        assert "0.0" in str(exc_info.value)

    def test_louvain_resolution_nan_error_message(self):
        """louvain_resolution NaN error mentions 'finite'."""
        with pytest.raises(ValueError) as exc_info:
            CorticalConfig(louvain_resolution=float('nan'))
        assert "finite" in str(exc_info.value).lower()

    def test_chunk_overlap_error_includes_both_values(self):
        """chunk_overlap >= chunk_size error shows both values."""
        with pytest.raises(ValueError) as exc_info:
            CorticalConfig(chunk_size=100, chunk_overlap=100)
        error_msg = str(exc_info.value)
        assert "100" in error_msg
        assert "chunk_overlap" in error_msg.lower()
        assert "chunk_size" in error_msg.lower()

    def test_scoring_algorithm_error_shows_valid_options(self):
        """scoring_algorithm error lists valid options."""
        with pytest.raises(ValueError) as exc_info:
            CorticalConfig(scoring_algorithm='invalid')
        error_msg = str(exc_info.value)
        assert "tfidf" in error_msg.lower()
        assert "bm25" in error_msg.lower()


# =============================================================================
# SPECIAL NUMERIC VALUE TESTS
# =============================================================================


class TestSpecialNumericValues:
    """Tests for handling of special floating point values."""

    def test_pagerank_tolerance_accepts_very_small_values(self):
        """pagerank_tolerance accepts extremely small positive values."""
        config = CorticalConfig(pagerank_tolerance=1e-100)
        assert config.pagerank_tolerance == 1e-100

    def test_zero_values_where_allowed(self):
        """Zero is accepted for parameters that allow it."""
        config = CorticalConfig(
            cluster_strictness=0.0,
            chunk_overlap=0,
            max_query_expansions=0,
            semantic_expansion_discount=0.0,
            bm25_k1=0.0,
            bm25_b=0.0,
        )
        assert config.cluster_strictness == 0.0
        assert config.chunk_overlap == 0
        assert config.max_query_expansions == 0

    def test_very_large_integer_values(self):
        """Very large integers are accepted where valid."""
        config = CorticalConfig(
            pagerank_iterations=1000000,
            chunk_size=1000000,
            max_query_expansions=1000000,
        )
        assert config.pagerank_iterations == 1000000
        assert config.chunk_size == 1000000


# =============================================================================
# MODULE UTILITY FUNCTION TESTS
# =============================================================================


class TestModuleUtilityFunctions:
    """Additional tests for module-level utility functions."""

    def test_get_default_config_independent_instances(self):
        """Each call to get_default_config returns independent instance."""
        config1 = get_default_config()
        config2 = get_default_config()

        # Modify one
        config1.pagerank_damping = 0.95

        # Other should be unaffected
        assert config2.pagerank_damping == 0.85

    def test_get_default_config_matches_constructor_defaults(self):
        """get_default_config() matches CorticalConfig() defaults."""
        from_function = get_default_config()
        from_constructor = CorticalConfig()

        assert from_function.pagerank_damping == from_constructor.pagerank_damping
        assert from_function.pagerank_iterations == from_constructor.pagerank_iterations
        assert from_function.min_cluster_size == from_constructor.min_cluster_size
        assert from_function.chunk_size == from_constructor.chunk_size
