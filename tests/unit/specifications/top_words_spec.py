"""
Top Words Dictionary Specifications
====================================

SPECIFICATION: These tests document LOAD-BEARING behavior for the top words dictionary.

The top words dictionary provides frequency data for the 100 most common words
in the samples corpus. This is used for NGram model initialization, query
expansion weighting, and vocabulary analysis.

DO NOT CHANGE these specifications without:
1. Re-analyzing the samples corpus
2. Documenting why the vocabulary changed
3. Updating dependent code that relies on specific words

Ratified: 2025-12-30
Guardian: CI Pipeline
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


class TestTopWordsDictionarySpecification:
    """
    Specifications for TOP_WORDS dictionary structure and content.

    Each specification documents a fact about the corpus that must remain true
    unless the samples corpus itself changes significantly.
    """

    def test_spec_dictionary_contains_exactly_100_words(self):
        """
        SPECIFICATION: The TOP_WORDS dictionary contains exactly 100 entries.

        This is by design - we track the top 100 most frequent words.
        Changing this count would affect downstream code that depends on it.
        """
        assert TOP_WORDS_COUNT == 100, (
            f"TOP_WORDS must contain exactly 100 words, got {TOP_WORDS_COUNT}"
        )
        assert len(TOP_WORDS) == 100, (
            f"TOP_WORDS dictionary must have 100 entries, got {len(TOP_WORDS)}"
        )

    def test_spec_all_values_are_positive_integers(self):
        """
        SPECIFICATION: All frequency values are positive integers.

        Frequencies represent actual occurrence counts and must be natural numbers.
        """
        for word, count in TOP_WORDS.items():
            assert isinstance(count, int), (
                f"Frequency for '{word}' must be int, got {type(count)}"
            )
            assert count > 0, (
                f"Frequency for '{word}' must be positive, got {count}"
            )

    def test_spec_all_keys_are_lowercase_strings(self):
        """
        SPECIFICATION: All dictionary keys are lowercase strings.

        This ensures consistent lookups regardless of input casing.
        """
        for word in TOP_WORDS.keys():
            assert isinstance(word, str), (
                f"Word key must be string, got {type(word)}"
            )
            assert word == word.lower(), (
                f"Word '{word}' must be lowercase"
            )
            assert len(word) >= 3, (
                f"Word '{word}' must be at least 3 characters"
            )

    def test_spec_most_frequent_word_is_models(self):
        """
        SPECIFICATION: 'models' is the most frequent word in the corpus.

        This reflects the technical/ML focus of the sample documents.
        """
        top_word, top_count = get_top_words(1)[0]
        assert top_word == 'models', (
            f"Most frequent word should be 'models', got '{top_word}'"
        )

    def test_spec_top_five_words_reflect_corpus_domain(self):
        """
        SPECIFICATION: Top 5 words reflect the corpus's focus on ML/data/systems.

        The sample corpus is heavily focused on machine learning, data analysis,
        and systems thinking. The top words must reflect this domain.
        """
        top_five = [word for word, _ in get_top_words(5)]
        expected_domain_words = {'models', 'data', 'systems', 'learning', 'time'}
        actual_top_five = set(top_five)

        assert actual_top_five == expected_domain_words, (
            f"Top 5 words should be {expected_domain_words}, got {actual_top_five}"
        )

    def test_spec_total_occurrences_is_consistent(self):
        """
        SPECIFICATION: TOTAL_TOP_WORD_OCCURRENCES equals sum of all frequencies.

        This derived constant must remain consistent with the dictionary.
        """
        calculated_total = sum(TOP_WORDS.values())
        assert TOTAL_TOP_WORD_OCCURRENCES == calculated_total, (
            f"Total should be {calculated_total}, got {TOTAL_TOP_WORD_OCCURRENCES}"
        )


class TestTopWordsAPISpecification:
    """
    Specifications for the top_words module API functions.
    """

    def test_spec_get_top_words_returns_sorted_by_frequency(self):
        """
        SPECIFICATION: get_top_words() returns words sorted by frequency descending.

        The first item always has the highest count.
        """
        words = get_top_words(10)
        for i in range(len(words) - 1):
            assert words[i][1] >= words[i + 1][1], (
                f"Words not sorted: {words[i]} should come before {words[i+1]}"
            )

    def test_spec_get_top_words_respects_limit(self):
        """
        SPECIFICATION: get_top_words(n) returns exactly n items (up to 100).
        """
        assert len(get_top_words(5)) == 5
        assert len(get_top_words(50)) == 50
        assert len(get_top_words(100)) == 100
        assert len(get_top_words(200)) == 100  # Capped at 100

    def test_spec_get_word_frequency_case_insensitive(self):
        """
        SPECIFICATION: get_word_frequency() is case-insensitive.

        Users can query with any casing and get the correct frequency.
        """
        lower_freq = get_word_frequency('models')
        upper_freq = get_word_frequency('MODELS')
        mixed_freq = get_word_frequency('Models')

        assert lower_freq == upper_freq == mixed_freq, (
            "Frequency lookup must be case-insensitive"
        )
        assert lower_freq > 0, "Known word must have positive frequency"

    def test_spec_get_word_frequency_returns_zero_for_unknown(self):
        """
        SPECIFICATION: get_word_frequency() returns 0 for unknown words.

        This is defensive behavior - unknown words have zero frequency.
        """
        assert get_word_frequency('xyznonexistent') == 0
        assert get_word_frequency('') == 0

    def test_spec_is_common_word_uses_threshold(self):
        """
        SPECIFICATION: is_common_word() returns True if frequency >= threshold.

        Default threshold is 300 occurrences.
        """
        # 'models' has 866 occurrences (common)
        assert is_common_word('models') is True
        # 'approach' has 230 occurrences (not common at default threshold)
        assert is_common_word('approach') is False
        # But 'approach' is common at lower threshold
        assert is_common_word('approach', threshold=200) is True

    def test_spec_get_words_by_frequency_range_inclusive(self):
        """
        SPECIFICATION: get_words_by_frequency_range() uses inclusive bounds.

        Both min_freq and max_freq boundaries are included in results.
        """
        # Get words with exactly 356 occurrences (there are several)
        words = get_words_by_frequency_range(356, 356)
        for word in words:
            assert get_word_frequency(word) == 356

    def test_spec_get_words_by_frequency_range_returns_list(self):
        """
        SPECIFICATION: get_words_by_frequency_range() returns a list of strings.
        """
        words = get_words_by_frequency_range(500, 600)
        assert isinstance(words, list)
        for word in words:
            assert isinstance(word, str)
