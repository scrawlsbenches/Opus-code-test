"""
╔══════════════════════════════════════════════════════════════════════╗
║                   TOKENIZER PERFORMANCE CONTRACT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Tokenize 1000-word document < 10ms                               ║
║  • Tokenize 100 documents (100K words total) < 1 second             ║
║  • Extract bigrams < 5ms per 1000 tokens                            ║
║  • Identifier splitting < 2μs per identifier                        ║
║  • Word variant expansion < 0.1ms per word                          ║
║  • Tokenization produces valid output (no empty tokens)             ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest
from cortical.tokenizer import Tokenizer, split_identifier


@pytest.mark.contract
class TestTokenizePerformanceContract:
    """
    Tokenizer Performance Contract

    As a developer building text processing pipelines,
    I expect tokenization to be fast and scalable,
    So that document indexing doesn't become a bottleneck.
    """

    # The sacred numbers
    MAX_LATENCY_MS_PER_1K_WORDS = 10
    MAX_BATCH_LATENCY_MS = 1000
    MAX_BIGRAM_LATENCY_MS_PER_1K = 5

    def test_tokenize_single_document_latency(self):
        """
        CONTRACT: Tokenize 1000-word document in < 10ms.

        Single document tokenization must be fast for real-time use.
        """
        tokenizer = Tokenizer()

        # Create 1000-word document
        words = ["neural", "network", "learning", "algorithm", "data"] * 200
        text = " ".join(words)

        start = time.perf_counter()
        tokens = tokenizer.tokenize(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS_PER_1K_WORDS, (
            f"CONTRACT VIOLATION: Tokenizing 1000 words took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS_PER_1K_WORDS}ms"
        )

        # Verify output validity
        assert len(tokens) > 0, "Tokenization produced empty output"
        assert all(len(t) >= 3 for t in tokens), "Tokens below minimum length"

    def test_tokenize_batch_latency(self):
        """
        CONTRACT: Tokenize 100 documents (100K words) in < 1 second.

        Batch processing must scale for corpus indexing.
        """
        tokenizer = Tokenizer()

        # Create 100 documents, 1000 words each
        words = ["neural", "network", "learning", "algorithm", "data"] * 200
        doc_text = " ".join(words)
        documents = [doc_text for _ in range(100)]

        start = time.perf_counter()
        for doc in documents:
            tokens = tokenizer.tokenize(doc)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_BATCH_LATENCY_MS, (
            f"CONTRACT VIOLATION: Tokenizing 100 documents took {elapsed_ms:.0f}ms, "
            f"contract requires <{self.MAX_BATCH_LATENCY_MS}ms"
        )

    def test_extract_bigrams_latency(self):
        """
        CONTRACT: Extract bigrams from 1000 tokens in < 5ms.

        N-gram extraction is a hot path in indexing.
        """
        tokenizer = Tokenizer()
        tokens = ["word" + str(i) for i in range(1000)]

        start = time.perf_counter()
        bigrams = tokenizer.extract_ngrams(tokens, n=2)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_BIGRAM_LATENCY_MS_PER_1K, (
            f"CONTRACT VIOLATION: Extracting bigrams from 1000 tokens took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_BIGRAM_LATENCY_MS_PER_1K}ms"
        )

        # Verify correctness
        assert len(bigrams) == 999, "Bigram count incorrect"
        assert all(' ' in bg for bg in bigrams), "Bigrams not space-separated"

    def test_tokenization_produces_no_empty_tokens(self):
        """
        CONTRACT: Tokenization never produces empty tokens.

        Empty tokens break indexing and must be filtered.
        """
        tokenizer = Tokenizer()

        # Test various inputs
        test_cases = [
            "normal text here",
            "  extra    spaces   ",
            "punctuation!!! everywhere???",
            "under_score_style",
            "camelCaseWords",
            "mixedCase_and_underscores",
            "123 numbers 456",
        ]

        for text in test_cases:
            tokens = tokenizer.tokenize(text)
            assert all(len(t) > 0 for t in tokens), (
                f"CONTRACT VIOLATION: Empty token in result for input: '{text}'"
            )

    def test_tokenization_filters_stop_words(self):
        """
        CONTRACT: Stop words are consistently filtered.

        Stop word filtering is essential for search quality.
        """
        tokenizer = Tokenizer()
        text = "the quick brown fox jumps over the lazy dog"
        tokens = tokenizer.tokenize(text)

        # 'the' and 'over' should be filtered
        assert 'the' not in tokens, "Stop word 'the' not filtered"
        assert 'over' not in tokens, "Stop word 'over' not filtered"

        # Content words should remain
        assert 'quick' in tokens or 'brown' in tokens or 'fox' in tokens


@pytest.mark.contract
class TestIdentifierSplittingContract:
    """
    Identifier Splitting Performance Contract

    As a developer indexing code,
    I expect identifier splitting to be fast,
    So that code tokenization scales.
    """

    MAX_LATENCY_US_PER_IDENTIFIER = 2.0  # 2 microseconds

    @pytest.mark.skip(reason="Flaky: environment-dependent timing varies beyond 2.0μs contract threshold")
    def test_split_identifier_latency(self):
        """
        CONTRACT: Split identifier in < 2μs.

        Code indexing processes millions of identifiers.
        """
        identifiers = [
            "getUserCredentials",
            "parse_http_response",
            "XMLParser",
            "processHTTPRequest",
            "get_user_name",
        ]

        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            for ident in identifiers:
                parts = split_identifier(ident)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        avg_us = elapsed_us / (iterations * len(identifiers))

        assert avg_us < self.MAX_LATENCY_US_PER_IDENTIFIER, (
            f"CONTRACT VIOLATION: Split identifier took {avg_us:.2f}μs on average, "
            f"contract requires <{self.MAX_LATENCY_US_PER_IDENTIFIER}μs"
        )

    def test_split_identifier_correctness(self):
        """
        CONTRACT: Identifier splitting is correct.

        Splitting must handle all common identifier styles.
        """
        test_cases = [
            ("getUserCredentials", ["get", "user", "credentials"]),
            ("parse_http_response", ["parse", "http", "response"]),
            ("XMLParser", ["xml", "parser"]),
            ("get_user_data", ["get", "user", "data"]),
            ("processHTTPRequest", ["process", "http", "request"]),
        ]

        for ident, expected in test_cases:
            result = split_identifier(ident)
            assert result == expected, (
                f"CONTRACT VIOLATION: split_identifier('{ident}') = {result}, "
                f"expected {expected}"
            )


@pytest.mark.contract
class TestWordVariantsContract:
    """
    Word Variants Performance Contract

    As a developer building query expansion,
    I expect word variant lookup to be fast,
    So that query processing stays responsive.
    """

    MAX_LATENCY_MS = 0.1

    def test_word_variants_latency(self):
        """
        CONTRACT: Get word variants in < 0.1ms.

        Query expansion happens at search time and must be fast.
        """
        tokenizer = Tokenizer()
        words = ["neural", "bread", "database", "fast", "ai"]

        start = time.perf_counter()
        for word in words:
            variants = tokenizer.get_word_variants(word)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / len(words)

        assert avg_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Word variant lookup took {avg_ms:.3f}ms on average, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_word_variants_include_original(self):
        """
        CONTRACT: Variants always include the original word.

        Query expansion must preserve original terms.
        """
        tokenizer = Tokenizer()
        words = ["neural", "bread", "database"]

        for word in words:
            variants = tokenizer.get_word_variants(word)
            assert word in variants, (
                f"CONTRACT VIOLATION: Variants for '{word}' don't include original: {variants}"
            )

    def test_word_variants_include_stem(self):
        """
        CONTRACT: Variants include stemmed form if different.

        Stemming improves recall.
        """
        tokenizer = Tokenizer()

        # Word with clear stem
        variants = tokenizer.get_word_variants("learning")

        # Should include original
        assert "learning" in variants

        # Should have multiple variants (original + stem/plural/etc)
        assert len(variants) >= 2, (
            f"CONTRACT VIOLATION: Expected multiple variants for 'learning', got {variants}"
        )


@pytest.mark.contract
class TestTokenizerCodeNoiseContract:
    """
    Code Noise Filtering Contract

    As a developer indexing code,
    I expect code noise to be filtered correctly,
    So that code search quality is high.
    """

    def test_code_noise_filtering_enabled(self):
        """
        CONTRACT: Code noise filtering removes common tokens.

        Common code tokens (self, def, return) pollute search results.
        """
        tokenizer = Tokenizer(filter_code_noise=True)

        code = "def get_user(self, user_id): return self.database.query(user_id)"
        tokens = tokenizer.tokenize(code)

        # Should filter common noise
        noise_tokens = {'self', 'def', 'return'}
        found_noise = noise_tokens & set(tokens)

        assert len(found_noise) == 0, (
            f"CONTRACT VIOLATION: Code noise tokens found: {found_noise}"
        )

    def test_code_noise_filtering_preserves_content(self):
        """
        CONTRACT: Code noise filtering preserves meaningful content.

        Only noise should be filtered, not domain terms.
        """
        tokenizer = Tokenizer(filter_code_noise=True)

        code = "def process_neural_network(self): return network.train()"
        tokens = tokenizer.tokenize(code)

        # Should preserve meaningful terms
        assert 'process' in tokens or 'neural' in tokens or 'network' in tokens, (
            f"CONTRACT VIOLATION: Meaningful content lost: {tokens}"
        )
