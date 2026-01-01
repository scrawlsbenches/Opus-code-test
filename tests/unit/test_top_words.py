"""
Tests for Top Words module.
"""

import pytest

from cortical.top_words import (
    TOP_WORDS,
    TOP_WORDS_COUNT,
    TOTAL_TOP_WORD_OCCURRENCES,
    get_top_words,
    get_word_frequency,
    is_common_word,
    get_words_by_frequency_range,
)


class TestTopWordsConstants:
    """Test top_words module constants."""

    def test_top_words_dict_not_empty(self):
        """Test that TOP_WORDS dictionary is populated."""
        assert len(TOP_WORDS) > 0
        assert len(TOP_WORDS) == 100

    def test_top_words_count_matches(self):
        """Test that TOP_WORDS_COUNT matches dictionary length."""
        assert TOP_WORDS_COUNT == len(TOP_WORDS)

    def test_total_occurrences_calculated(self):
        """Test that TOTAL_TOP_WORD_OCCURRENCES is sum of counts."""
        assert TOTAL_TOP_WORD_OCCURRENCES == sum(TOP_WORDS.values())
        assert TOTAL_TOP_WORD_OCCURRENCES > 0


class TestGetTopWords:
    """Test get_top_words function."""

    def test_get_top_5_words(self):
        """Test getting top 5 words."""
        result = get_top_words(5)

        assert len(result) == 5
        assert result[0][0] == 'models'
        assert result[0][1] == 866

    def test_get_all_top_words(self):
        """Test getting all top words."""
        result = get_top_words(100)

        assert len(result) == 100

    def test_get_more_than_available(self):
        """Test requesting more words than available."""
        result = get_top_words(200)

        # Should cap at available words
        assert len(result) == 100


class TestGetWordFrequency:
    """Test get_word_frequency function."""

    def test_known_word(self):
        """Test frequency of a known word."""
        freq = get_word_frequency('learning')
        assert freq == 802

    def test_unknown_word(self):
        """Test frequency of an unknown word."""
        freq = get_word_frequency('xyznonexistent')
        assert freq == 0

    def test_case_insensitive(self):
        """Test that lookup is case insensitive."""
        freq_lower = get_word_frequency('learning')
        freq_upper = get_word_frequency('LEARNING')
        freq_mixed = get_word_frequency('LeArNiNg')

        assert freq_lower == freq_upper == freq_mixed


class TestIsCommonWord:
    """Test is_common_word function."""

    def test_common_word_above_threshold(self):
        """Test that high frequency word is common."""
        # 'models' has 866 occurrences, threshold is 300
        assert is_common_word('models') is True

    def test_word_below_threshold(self):
        """Test that low frequency word is not common."""
        # 'approach' has 230 occurrences, threshold is 300
        assert is_common_word('approach') is False

    def test_custom_threshold(self):
        """Test with custom threshold."""
        # 'approach' has 230 occurrences
        assert is_common_word('approach', threshold=200) is True
        assert is_common_word('approach', threshold=250) is False


class TestGetWordsByFrequencyRange:
    """Test get_words_by_frequency_range function."""

    def test_high_frequency_range(self):
        """Test getting words in high frequency range."""
        words = get_words_by_frequency_range(800, 900)

        assert 'models' in words
        assert 'data' in words
        assert len(words) > 0

    def test_no_words_in_range(self):
        """Test range with no words."""
        words = get_words_by_frequency_range(10000, 20000)

        assert words == []

    def test_all_words(self):
        """Test range covering all words."""
        words = get_words_by_frequency_range(0, 1000)

        assert len(words) == 100
