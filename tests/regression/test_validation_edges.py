"""
Regression Tests for Validation Edge Cases
==========================================

Tests for parameter validation at boundary values and invalid inputs.
These prevent silent failures and ensure clear error messages.

Regression test for T-018: Edge case coverage for production robustness.
"""

import pytest
from cortical import CorticalTextProcessor


class TestGraphBoostedSearchValidation:
    """Test graph_boosted_search parameter validation."""

    def test_weight_sum_exceeds_one(self, small_processor):
        """
        graph_boosted_search with pagerank_weight + proximity_weight > 1.0 is valid.

        Regression test for T-018: Weights can sum to > 1.0 (additive boosts).
        This is a design choice - weights are additive, not normalized.
        """
        # This should work (weights are additive boosts, not probabilities)
        results = small_processor.graph_boosted_search(
            "machine learning",
            top_n=5,
            pagerank_weight=0.8,
            proximity_weight=0.5  # Sum = 1.3 > 1.0
        )

        # Should return results without error
        assert isinstance(results, list)

    def test_negative_pagerank_weight(self, small_processor):
        """
        Negative pagerank_weight should raise ValueError.

        Regression test for T-018: Parameter validation for weights.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.graph_boosted_search(
                "machine learning",
                pagerank_weight=-0.1,
                proximity_weight=0.2
            )

        assert "pagerank_weight" in str(exc_info.value).lower()

    def test_negative_proximity_weight(self, small_processor):
        """
        Negative proximity_weight should raise ValueError.

        Regression test for T-018: Parameter validation for weights.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.graph_boosted_search(
                "machine learning",
                pagerank_weight=0.3,
                proximity_weight=-0.1
            )

        assert "proximity_weight" in str(exc_info.value).lower()

    def test_pagerank_weight_exceeds_one(self, small_processor):
        """
        pagerank_weight > 1.0 should raise ValueError.

        Regression test for T-018: Parameter validation for weights.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.graph_boosted_search(
                "machine learning",
                pagerank_weight=1.5,
                proximity_weight=0.2
            )

        assert "pagerank_weight" in str(exc_info.value).lower()

    def test_proximity_weight_exceeds_one(self, small_processor):
        """
        proximity_weight > 1.0 should raise ValueError.

        Regression test for T-018: Parameter validation for weights.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.graph_boosted_search(
                "machine learning",
                pagerank_weight=0.3,
                proximity_weight=1.2
            )

        assert "proximity_weight" in str(exc_info.value).lower()


class TestQueryValidation:
    """Test query parameter validation."""

    def test_empty_query_string(self, small_processor):
        """
        Empty query string should raise ValueError.

        Regression test for T-018: Explicit error for empty queries.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.find_documents_for_query("", top_n=5)

        assert "non-empty" in str(exc_info.value).lower()

    def test_whitespace_only_query(self, small_processor):
        """
        Whitespace-only query should raise ValueError.

        Regression test for T-018: Explicit error for meaningless queries.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.find_documents_for_query("   \t\n  ", top_n=5)

        assert "non-empty" in str(exc_info.value).lower()

    def test_graph_boosted_search_empty_query(self, small_processor):
        """
        graph_boosted_search with empty query should raise ValueError.

        Regression test for T-018: Consistent validation across search methods.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.graph_boosted_search("", top_n=5)

        assert "non-empty" in str(exc_info.value).lower()


class TestTopNValidation:
    """Test top_n parameter validation."""

    def test_top_n_zero(self, small_processor):
        """
        top_n=0 should raise ValueError.

        Regression test for T-018: Invalid result count.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.find_documents_for_query("machine learning", top_n=0)

        assert "top_n" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()

    def test_top_n_negative(self, small_processor):
        """
        Negative top_n should raise ValueError.

        Regression test for T-018: Invalid result count.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.find_documents_for_query("machine learning", top_n=-1)

        assert "top_n" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()

    def test_graph_boosted_search_top_n_zero(self, small_processor):
        """
        graph_boosted_search with top_n=0 should raise ValueError.

        Regression test for T-018: Consistent validation across search methods.
        """
        with pytest.raises(ValueError) as exc_info:
            small_processor.graph_boosted_search("machine learning", top_n=0)

        assert "top_n" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()


class TestUnicodeAndSpecialCharacters:
    """Test handling of unicode and special characters."""

    def test_unicode_query(self, small_processor):
        """
        Unicode characters in query should be handled gracefully.

        Regression test for T-018: UTF-8 support in queries.
        """
        # Should not crash with unicode
        results = small_processor.find_documents_for_query("数据处理", top_n=5)

        # May return empty results (no matches) but shouldn't crash
        assert isinstance(results, list)

    def test_emoji_in_query(self, small_processor):
        """
        Emoji in query should be handled gracefully.

        Regression test for T-018: Special unicode handling.
        """
        results = small_processor.find_documents_for_query("🔥 machine learning", top_n=5)

        # Should not crash
        assert isinstance(results, list)

    def test_special_characters_in_query(self, small_processor):
        """
        Special characters in query should be handled gracefully.

        Regression test for T-018: Punctuation and symbols.
        """
        queries = [
            "machine-learning",
            "C++",
            "data.frame",
            "key:value",
            "50% accuracy",
            "cost=$100"
        ]

        for query in queries:
            results = small_processor.find_documents_for_query(query, top_n=5)
            # Should not crash
            assert isinstance(results, list)

    def test_very_long_query(self, small_processor):
        """
        Very long query (1000+ chars) should be handled gracefully.

        Regression test for T-018: No arbitrary length limits.
        """
        # Generate a very long query
        long_query = " ".join(["machine learning"] * 100)  # ~1600 chars

        results = small_processor.find_documents_for_query(long_query, top_n=5)

        # Should not crash
        assert isinstance(results, list)


class TestConfigValidation:
    """Test configuration parameter validation."""

    def test_invalid_scoring_algorithm(self):
        """
        Invalid scoring algorithm should raise ValueError.

        Regression test for T-018: Config validation.
        """
        from cortical.config import CorticalConfig

        with pytest.raises(ValueError):
            CorticalConfig(scoring_algorithm='invalid_algorithm')

    def test_bm25_k1_negative(self):
        """
        Negative bm25_k1 should raise ValueError.

        Regression test for T-018: BM25 parameter validation.
        """
        from cortical.config import CorticalConfig

        with pytest.raises(ValueError):
            CorticalConfig(scoring_algorithm='bm25', bm25_k1=-0.1)

    def test_bm25_b_out_of_range(self):
        """
        bm25_b outside [0, 1] should raise ValueError.

        Regression test for T-018: BM25 parameter validation.
        """
        from cortical.config import CorticalConfig

        with pytest.raises(ValueError):
            CorticalConfig(scoring_algorithm='bm25', bm25_b=1.5)

        with pytest.raises(ValueError):
            CorticalConfig(scoring_algorithm='bm25', bm25_b=-0.1)

    def test_pagerank_damping_out_of_range(self):
        """
        PageRank damping outside [0, 1] should raise ValueError.

        Regression test for T-018: PageRank parameter validation.
        """
        from cortical.config import CorticalConfig

        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=1.5)

        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=-0.1)
