"""
Unit Tests for Tokenizer Module
================================

Task #159: Unit tests for cortical/tokenizer.py.

Tests the tokenization, stemming, and word variant support:
- split_identifier: Identifier splitting (camelCase, snake_case)
- Tokenizer.tokenize: Main tokenization with filtering
- Tokenizer.extract_ngrams: N-gram extraction
- Tokenizer.stem: Porter-lite stemming
- Tokenizer.get_word_variants: Word variant expansion
- Tokenizer.add_word_mapping: Custom word mappings

These tests cover basic text tokenization, code tokenization with
identifier splitting, stop word filtering, and n-gram extraction.
"""

import pytest

from cortical.tokenizer import (
    Tokenizer,
    split_identifier,
    CODE_EXPANSION_STOP_WORDS,
    CODE_NOISE_TOKENS,
    PROGRAMMING_KEYWORDS,
)


# =============================================================================
# IDENTIFIER SPLITTING TESTS
# =============================================================================


class TestSplitIdentifier:
    """Tests for split_identifier function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = split_identifier("")
        assert result == []

    def test_simple_lowercase(self):
        """Simple lowercase word returns as-is."""
        result = split_identifier("simple")
        assert result == ["simple"]

    def test_camelcase(self):
        """camelCase splits into components."""
        result = split_identifier("getUserCredentials")
        assert result == ["get", "user", "credentials"]

    def test_pascalcase(self):
        """PascalCase splits into components."""
        result = split_identifier("UserCredentials")
        assert result == ["user", "credentials"]

    def test_snake_case(self):
        """snake_case splits on underscores."""
        result = split_identifier("get_user_data")
        assert result == ["get", "user", "data"]

    def test_constant_style(self):
        """CONSTANT_STYLE splits on underscores and lowercases."""
        result = split_identifier("MAX_RETRY_COUNT")
        assert result == ["max", "retry", "count"]

    def test_acronym_at_start(self):
        """Acronym at start: XMLParser -> ['xml', 'parser']."""
        result = split_identifier("XMLParser")
        assert result == ["xml", "parser"]

    def test_acronym_in_middle(self):
        """Acronym in middle: parseHTTPResponse -> ['parse', 'http', 'response']."""
        result = split_identifier("parseHTTPResponse")
        assert result == ["parse", "http", "response"]

    def test_mixed_case_and_underscores(self):
        """Mixed camelCase_and_underscores splits both ways."""
        result = split_identifier("get_HTTPResponse")
        assert "get" in result
        assert "http" in result
        assert "response" in result

    def test_leading_underscore(self):
        """Leading underscore is handled: _privateMethod."""
        result = split_identifier("_privateMethod")
        assert "private" in result
        assert "method" in result

    def test_double_underscore(self):
        """Double underscore: __init__ -> ['init']."""
        result = split_identifier("__init__")
        assert "init" in result

    def test_single_letter(self):
        """Single letter remains as-is."""
        result = split_identifier("a")
        assert result == ["a"]

    def test_numbers_in_identifier(self):
        """Numbers are preserved: word2vec."""
        result = split_identifier("word2vec")
        assert result == ["word2vec"]

    def test_all_caps(self):
        """All caps identifier: API -> ['api']."""
        result = split_identifier("API")
        assert result == ["api"]

    def test_consecutive_caps(self):
        """Consecutive caps: parseXML -> ['parse', 'xml']."""
        result = split_identifier("parseXML")
        assert result == ["parse", "xml"]


# =============================================================================
# BASIC TOKENIZATION TESTS
# =============================================================================


class TestBasicTokenization:
    """Tests for basic text tokenization."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("")
        assert result == []

    def test_single_word(self):
        """Single word is tokenized."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("hello")
        assert result == ["hello"]

    def test_multiple_words(self):
        """Multiple words are tokenized."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("neural networks process information")
        assert "neural" in result
        assert "networks" in result
        assert "process" in result
        assert "information" in result

    def test_punctuation_removed(self):
        """Punctuation is removed."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("Hello, world! How are you?")
        assert "hello" in result
        assert "world" in result
        # Punctuation marks themselves should not be tokens
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_whitespace_normalized(self):
        """Multiple spaces/tabs/newlines normalized."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("hello    world\n\tfoo")
        assert "hello" in result
        assert "world" in result
        assert "foo" in result

    def test_lowercase_conversion(self):
        """All tokens converted to lowercase."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("Hello WORLD")
        assert "hello" in result
        assert "world" in result
        assert "Hello" not in result
        assert "WORLD" not in result

    def test_min_word_length_default(self):
        """Words shorter than min_word_length (default 3) filtered."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("a be cat dogs")
        # 'a' (1 char), 'be' (2 chars) should be filtered
        assert "a" not in result
        assert "be" not in result
        assert "cat" in result
        assert "dogs" in result

    def test_min_word_length_custom(self):
        """Custom min_word_length respected."""
        tokenizer = Tokenizer(min_word_length=2, stop_words=set())
        result = tokenizer.tokenize("a be cat")
        assert "a" not in result  # Still < 2
        assert "be" in result  # >= 2
        assert "cat" in result

    def test_unicode_text(self):
        """Unicode characters filtered (ASCII-only regex)."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("café résumé naïve")
        # Tokenizer uses ASCII-only regex, so accented chars filtered
        # This is expected behavior - just test it doesn't crash
        assert result == []  # No ASCII-only words in this text

    def test_numbers_filtered(self):
        """Pure numbers filtered out."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("hello 123 world 456")
        assert "hello" in result
        assert "world" in result
        assert "123" not in result
        assert "456" not in result

    def test_hyphenated_words(self):
        """Hyphenated words split into components."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("state-of-the-art")
        assert "state" in result
        assert "art" in result


# =============================================================================
# STOP WORD FILTERING TESTS
# =============================================================================


class TestStopWordFiltering:
    """Tests for stop word filtering."""

    def test_default_stop_words(self):
        """Default stop words are filtered."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("the quick brown fox")
        assert "the" not in result  # Stop word
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_custom_stop_words(self):
        """Custom stop words replace defaults."""
        tokenizer = Tokenizer(stop_words={'foo', 'bar'})
        result = tokenizer.tokenize("the foo bar baz")
        # Custom stop words only
        assert "foo" not in result
        assert "bar" not in result
        # Default stop word 'the' is NOT filtered (we replaced, not extended)
        assert "the" in result
        assert "baz" in result

    def test_empty_stop_words(self):
        """Empty stop words set allows all words."""
        tokenizer = Tokenizer(stop_words=set())
        result = tokenizer.tokenize("the and or but")
        # All words allowed (still need >= min_word_length)
        assert "the" in result
        assert "and" in result
        assert "but" in result

    def test_filter_code_noise(self):
        """filter_code_noise adds code tokens to stop words."""
        tokenizer = Tokenizer(filter_code_noise=True)
        result = tokenizer.tokenize("self def class return process data")
        # Code noise filtered (data is in CODE_NOISE_TOKENS)
        assert "self" not in result
        assert "def" not in result
        assert "class" not in result
        assert "return" not in result
        assert "data" not in result  # 'data' is filtered too
        # Non-code words preserved
        assert "process" in result

    def test_filter_code_noise_disabled(self):
        """Code noise tokens allowed when filter_code_noise=False."""
        tokenizer = Tokenizer(filter_code_noise=False)
        result = tokenizer.tokenize("self def process")
        # When not filtering code noise and 'self', 'def' not in default stop words
        # they may appear. But default stop words might filter them.
        # Let's test with a code word that's definitely not in default stop words
        result = tokenizer.tokenize("isinstance process")
        assert "isinstance" in result
        assert "process" in result


# =============================================================================
# CODE TOKENIZATION TESTS
# =============================================================================


class TestCodeTokenization:
    """Tests for code-specific tokenization features."""

    def test_split_identifiers_disabled(self):
        """With split_identifiers=False, identifiers kept whole."""
        tokenizer = Tokenizer(split_identifiers=False)
        result = tokenizer.tokenize("getUserCredentials")
        assert "getusercredentials" in result
        # Components should NOT be present
        assert result.count("get") == 0

    def test_split_identifiers_enabled(self):
        """With split_identifiers=True, identifiers split."""
        tokenizer = Tokenizer(split_identifiers=True)
        result = tokenizer.tokenize("getUserCredentials")
        # Original token included
        assert "getusercredentials" in result
        # Components also included
        assert "get" in result
        assert "user" in result
        assert "credentials" in result

    def test_split_identifiers_override(self):
        """tokenize() split_identifiers parameter overrides instance setting."""
        tokenizer = Tokenizer(split_identifiers=False)
        result = tokenizer.tokenize("getUserData", split_identifiers=True)
        # Should split despite instance setting
        assert "get" in result
        assert "user" in result
        assert "data" in result

    def test_snake_case_splitting(self):
        """snake_case identifiers split correctly."""
        tokenizer = Tokenizer(split_identifiers=True)
        result = tokenizer.tokenize("get_user_data")
        assert "get_user_data" in result  # Original
        assert "get" in result
        assert "user" in result
        assert "data" in result

    def test_dunder_methods(self):
        """Dunder methods (__init__, __slots__) split correctly."""
        tokenizer = Tokenizer(split_identifiers=True)
        result = tokenizer.tokenize("__init__ __slots__")
        assert "init" in result
        assert "slots" in result

    def test_programming_keywords_preserved(self):
        """Programming keywords preserved even if in stop words."""
        # Create tokenizer with 'get' in stop words
        tokenizer = Tokenizer(stop_words={'get'}, split_identifiers=True)
        result = tokenizer.tokenize("getUserData")
        # 'get' is a programming keyword (in PROGRAMMING_KEYWORDS)
        # When split from identifier, it should be preserved
        assert "get" in result

    def test_mixed_text_and_code(self):
        """Mixed text and code tokenized correctly."""
        tokenizer = Tokenizer(split_identifiers=True)
        result = tokenizer.tokenize("The getUserData function processes information")
        assert "getusercredentials" not in result  # Different identifier
        assert "function" in result
        assert "processes" in result
        assert "information" in result

    def test_operators_filtered(self):
        """Operators and special chars filtered."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("x = y + z * 2")
        # Only variable names extracted
        # '=' '+' '*' '2' should not be in result
        assert "=" not in result
        assert "+" not in result
        assert "*" not in result

    def test_underscores_in_text(self):
        """Underscores in text handled correctly."""
        tokenizer = Tokenizer(split_identifiers=True)
        result = tokenizer.tokenize("user_name is_valid")
        assert "user_name" in result
        assert "user" in result
        assert "name" in result
        assert "is_valid" in result

    def test_camelcase_no_underscores(self):
        """Pure camelCase (no underscores) splits correctly."""
        tokenizer = Tokenizer(split_identifiers=True)
        result = tokenizer.tokenize("myVariableName")
        assert "myvariablename" in result
        assert "variable" in result
        assert "name" in result


# =============================================================================
# STEMMING TESTS
# =============================================================================


class TestStemming:
    """Tests for Porter-lite stemming."""

    def test_simple_word(self):
        """Simple word unchanged."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("simple")
        assert result == "simple"

    def test_ing_suffix(self):
        """'-ing' suffix removed (Porter-lite may leave some chars)."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("running")
        # Porter-lite stems to 'runn' (removes 'ing' but keeps double 'n')
        assert result == "runn"

    def test_ed_suffix(self):
        """'-ed' suffix removed."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("played")
        assert result == "play"

    def test_ly_suffix(self):
        """'-ly' suffix removed."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("quickly")
        assert result == "quick"

    def test_ies_to_y(self):
        """'-ies' becomes '-y'."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("flies")
        assert result == "fly"

    def test_short_word_unchanged(self):
        """Words <= 4 chars unchanged."""
        tokenizer = Tokenizer()
        assert tokenizer.stem("cat") == "cat"
        assert tokenizer.stem("dogs") == "dogs"  # 4 chars, unchanged

    def test_ation_to_ate(self):
        """'-ation' becomes '-ate'."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("creation")
        assert result == "create"

    def test_ization_to_ize(self):
        """'-ization' becomes '-ize'."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("organization")
        assert result == "organize"

    def test_ness_removed(self):
        """'-ness' suffix removed."""
        tokenizer = Tokenizer()
        result = tokenizer.stem("happiness")
        assert result == "happi"  # May stem to 'happi'

    def test_minimum_stem_length(self):
        """Stemmed word must be >= 3 chars."""
        tokenizer = Tokenizer()
        # If stem would be too short, return original
        result = tokenizer.stem("being")
        # 'being' - 'ing' = 'be' (2 chars), should keep original
        assert len(result) >= 3


# =============================================================================
# N-GRAM EXTRACTION TESTS
# =============================================================================


class TestNgramExtraction:
    """Tests for n-gram extraction."""

    def test_empty_tokens(self):
        """Empty token list returns empty list."""
        tokenizer = Tokenizer()
        result = tokenizer.extract_ngrams([], n=2)
        assert result == []

    def test_insufficient_tokens(self):
        """Token list smaller than n returns empty list."""
        tokenizer = Tokenizer()
        result = tokenizer.extract_ngrams(["hello"], n=2)
        assert result == []

    def test_bigrams(self):
        """Bigrams extracted correctly."""
        tokenizer = Tokenizer()
        tokens = ["neural", "networks", "process", "data"]
        result = tokenizer.extract_ngrams(tokens, n=2)
        assert result == [
            "neural networks",
            "networks process",
            "process data"
        ]

    def test_trigrams(self):
        """Trigrams extracted correctly."""
        tokenizer = Tokenizer()
        tokens = ["a", "b", "c", "d"]
        result = tokenizer.extract_ngrams(tokens, n=3)
        assert result == ["a b c", "b c d"]

    def test_exact_n_tokens(self):
        """Exactly n tokens produces one n-gram."""
        tokenizer = Tokenizer()
        tokens = ["hello", "world"]
        result = tokenizer.extract_ngrams(tokens, n=2)
        assert result == ["hello world"]

    def test_unigrams(self):
        """Unigrams (n=1) returns individual tokens joined."""
        tokenizer = Tokenizer()
        tokens = ["a", "b", "c"]
        result = tokenizer.extract_ngrams(tokens, n=1)
        # Each token is its own "1-gram"
        assert result == ["a", "b", "c"]

    def test_fourgrams(self):
        """4-grams extracted correctly."""
        tokenizer = Tokenizer()
        tokens = ["a", "b", "c", "d", "e"]
        result = tokenizer.extract_ngrams(tokens, n=4)
        assert result == ["a b c d", "b c d e"]

    def test_ngrams_preserve_order(self):
        """N-grams preserve token order."""
        tokenizer = Tokenizer()
        tokens = ["one", "two", "three"]
        result = tokenizer.extract_ngrams(tokens, n=2)
        assert result[0] == "one two"
        assert result[1] == "two three"


# =============================================================================
# WORD VARIANTS TESTS
# =============================================================================


class TestWordVariants:
    """Tests for word variant expansion."""

    def test_simple_word(self):
        """Simple word returns itself and variations."""
        tokenizer = Tokenizer()
        result = tokenizer.get_word_variants("test")
        assert "test" in result
        # Should include plural
        assert "tests" in result

    def test_plural_word(self):
        """Plural word includes singular."""
        tokenizer = Tokenizer()
        result = tokenizer.get_word_variants("tests")
        assert "tests" in result
        # Should include singular
        assert "test" in result

    def test_mapped_word(self):
        """Mapped word includes predefined variants."""
        tokenizer = Tokenizer()
        result = tokenizer.get_word_variants("bread")
        assert "bread" in result
        # Predefined mappings
        assert "sourdough" in result
        assert "dough" in result
        assert "flour" in result

    def test_stemmed_variant(self):
        """Stemmed version included in variants."""
        tokenizer = Tokenizer()
        result = tokenizer.get_word_variants("running")
        assert "running" in result
        # Should include stem (Porter-lite stems to 'runn')
        assert "runn" in result

    def test_no_duplicates(self):
        """Variants list has no duplicates."""
        tokenizer = Tokenizer()
        result = tokenizer.get_word_variants("test")
        assert len(result) == len(set(result))

    def test_lowercase_conversion(self):
        """Input converted to lowercase."""
        tokenizer = Tokenizer()
        result = tokenizer.get_word_variants("BREAD")
        assert "bread" in result
        # Should use lowercase for mappings
        assert "sourdough" in result


# =============================================================================
# CUSTOM WORD MAPPING TESTS
# =============================================================================


class TestCustomWordMappings:
    """Tests for custom word mapping additions."""

    def test_add_new_mapping(self):
        """Add new word mapping."""
        tokenizer = Tokenizer()
        tokenizer.add_word_mapping("python", ["programming", "code", "language"])
        result = tokenizer.get_word_variants("python")
        assert "python" in result
        assert "programming" in result
        assert "code" in result
        assert "language" in result

    def test_extend_existing_mapping(self):
        """Extending existing mapping adds to variants."""
        tokenizer = Tokenizer()
        # 'bread' already has mappings
        original_variants = tokenizer.get_word_variants("bread")
        tokenizer.add_word_mapping("bread", ["yeast", "oven"])
        new_variants = tokenizer.get_word_variants("bread")
        assert "yeast" in new_variants
        assert "oven" in new_variants
        # Original variants still present
        assert "sourdough" in new_variants

    def test_no_duplicate_variants(self):
        """Adding duplicate variants doesn't create duplicates."""
        tokenizer = Tokenizer()
        tokenizer.add_word_mapping("test", ["testing", "tested"])
        tokenizer.add_word_mapping("test", ["testing", "tester"])
        result = tokenizer.get_word_variants("test")
        # Count 'testing' should appear only once
        assert result.count("testing") == 1

    def test_lowercase_mapping(self):
        """Mapping keys stored in lowercase."""
        tokenizer = Tokenizer()
        tokenizer.add_word_mapping("PYTHON", ["code"])
        result = tokenizer.get_word_variants("python")
        # Variants are stored as-is, only the key is lowercased
        assert "code" in result


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_very_long_text(self):
        """Very long text handled correctly."""
        tokenizer = Tokenizer()
        text = " ".join(["word"] * 10000)
        result = tokenizer.tokenize(text)
        # Should have many instances of 'word'
        assert len(result) > 100

    def test_special_characters_only(self):
        """Text with only special characters returns empty."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("!@#$%^&*()")
        assert result == []

    def test_mixed_languages(self):
        """Mixed language text (basic handling)."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("hello world bonjour monde")
        # Should extract words (ASCII)
        assert "hello" in result
        assert "world" in result
        assert "bonjour" in result
        assert "monde" in result

    def test_repeated_tokens(self):
        """Repeated tokens all included (for bigram extraction)."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("test test test")
        # All instances should be in result for proper bigram extraction
        assert result.count("test") == 3

    def test_tokenize_then_ngrams(self):
        """Full pipeline: tokenize then extract n-grams."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("neural networks process information")
        bigrams = tokenizer.extract_ngrams(tokens, n=2)
        assert "neural networks" in bigrams
        assert "networks process" in bigrams
        assert "process information" in bigrams

    def test_code_with_split_then_ngrams(self):
        """Code tokenization with splitting, then n-grams."""
        tokenizer = Tokenizer(split_identifiers=True)
        tokens = tokenizer.tokenize("getUserData processInfo")
        bigrams = tokenizer.extract_ngrams(tokens, n=2)
        # Should have bigrams of both original and split tokens
        assert len(bigrams) > 0

    def test_minimum_word_length_zero(self):
        """min_word_length=0 still filters stop words."""
        tokenizer = Tokenizer(min_word_length=0, stop_words=set())
        result = tokenizer.tokenize("a be cat")
        # With no stop words, all should be present
        assert "a" in result
        assert "be" in result
        assert "cat" in result

    def test_minimum_word_length_large(self):
        """Large min_word_length filters aggressively."""
        tokenizer = Tokenizer(min_word_length=10)
        result = tokenizer.tokenize("hello world supercalifragilisticexpialidocious")
        # Only very long words
        assert "hello" not in result
        assert "world" not in result
        assert "supercalifragilisticexpialidocious" in result
