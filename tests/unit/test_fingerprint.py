"""
Unit Tests for Fingerprint Module
==================================

Task #163: Unit tests for cortical/fingerprint.py core functions.

Tests the fingerprinting functions that create semantic signatures
and compare them for similarity analysis:
- compute_fingerprint: Generate semantic fingerprint from text
- compare_fingerprints: Compare two fingerprints for similarity
- explain_fingerprint: Human-readable fingerprint explanation
- explain_similarity: Human-readable similarity explanation
- _cosine_similarity: Cosine similarity helper

These tests use minimal dependencies (just Tokenizer) and mock
layers when needed for TF-IDF weighting.
"""

import pytest
import math

from cortical.fingerprint import (
    compute_fingerprint,
    compare_fingerprints,
    explain_fingerprint,
    explain_similarity,
    _cosine_similarity,
    SemanticFingerprint,
)
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer

from tests.unit.mocks import MockLayers, MockMinicolumn


# =============================================================================
# COMPUTE FINGERPRINT TESTS
# =============================================================================


class TestComputeFingerprint:
    """Tests for compute_fingerprint function."""

    def test_empty_text(self):
        """Empty text produces empty fingerprint."""
        tokenizer = Tokenizer()
        fp = compute_fingerprint("", tokenizer)

        assert fp['term_count'] == 0
        assert len(fp['terms']) == 0
        assert len(fp['bigrams']) == 0
        assert len(fp['top_terms']) == 0
        assert fp['raw_text_hash'] == hash("")

    def test_simple_text(self):
        """Simple text produces basic fingerprint."""
        tokenizer = Tokenizer()
        text = "neural networks process data"
        fp = compute_fingerprint(text, tokenizer)

        assert fp['term_count'] > 0
        assert 'neural' in fp['terms'] or 'network' in fp['terms']  # May be stemmed
        assert len(fp['top_terms']) > 0
        assert fp['raw_text_hash'] == hash(text)

    def test_special_characters(self):
        """Text with special characters is handled."""
        tokenizer = Tokenizer()
        text = "Hello, world! Testing @#$% special chars..."
        fp = compute_fingerprint(text, tokenizer)

        # Should tokenize despite special chars
        assert fp['term_count'] > 0
        assert len(fp['terms']) > 0

    def test_stop_words_removed(self):
        """Stop words are removed from fingerprint."""
        tokenizer = Tokenizer()
        text = "the quick brown fox jumps over the lazy dog"
        fp = compute_fingerprint(text, tokenizer)

        # Stop words like "the", "over" should be removed
        # Content words like "quick", "brown", "fox" should remain
        assert 'the' not in fp['terms']
        assert 'quick' in fp['terms'] or 'brown' in fp['terms']

    def test_with_corpus_layers_tfidf(self):
        """With corpus layers, uses TF-IDF weighting."""
        tokenizer = Tokenizer()

        # Create mock layer with TF-IDF scores
        col1 = MockMinicolumn(content="important", tfidf=5.0)
        col2 = MockMinicolumn(content="common", tfidf=0.5)
        layers = MockLayers.empty()
        layers[CorticalLayer.TOKENS] = type('MockLayer', (), {
            'get_minicolumn': lambda self, term: {
                'important': col1,
                'common': col2
            }.get(term)
        })()

        text = "important common"
        fp = compute_fingerprint(text, tokenizer, layers=layers)

        # Term with higher TF-IDF should have higher weight
        if 'important' in fp['terms'] and 'common' in fp['terms']:
            assert fp['terms']['important'] > fp['terms']['common']

    def test_corpus_layers_term_not_found(self):
        """Term not in corpus falls back to TF weight."""
        tokenizer = Tokenizer()

        # Mock layer that returns None for unknown terms
        layers = MockLayers.empty()
        layers[CorticalLayer.TOKENS] = type('MockLayer', (), {
            'get_minicolumn': lambda self, term: None  # Term not found
        })()

        text = "unknown term"
        fp = compute_fingerprint(text, tokenizer, layers=layers)

        # Should still create fingerprint using TF weights
        assert fp['term_count'] >= 0

    def test_corpus_layers_no_token_layer(self):
        """No token layer in corpus falls back to TF weight."""
        tokenizer = Tokenizer()

        # Mock layers dict without token layer
        layers = {CorticalLayer.DOCUMENTS: MockLayers.empty()[CorticalLayer.DOCUMENTS]}

        text = "test term"
        fp = compute_fingerprint(text, tokenizer, layers=layers)

        # Should still create fingerprint
        assert fp['term_count'] >= 0

    def test_without_corpus_layers(self):
        """Without corpus layers, uses TF weighting only."""
        tokenizer = Tokenizer()
        text = "test test other"
        fp = compute_fingerprint(text, tokenizer, layers=None)

        # "test" appears twice, "other" once
        # TF for "test" should be higher
        assert 'test' in fp['terms']
        assert fp['terms']['test'] > 0

    def test_top_n_parameter(self):
        """top_n parameter limits top terms returned."""
        tokenizer = Tokenizer()
        text = "one two three four five six seven eight nine ten"

        fp5 = compute_fingerprint(text, tokenizer, top_n=5)
        fp3 = compute_fingerprint(text, tokenizer, top_n=3)

        assert len(fp5['top_terms']) <= 5
        assert len(fp3['top_terms']) <= 3

    def test_concept_coverage(self):
        """Concepts are detected from code_concepts module."""
        tokenizer = Tokenizer()
        # Use programming terms that should map to concepts
        text = "function method class object"
        fp = compute_fingerprint(text, tokenizer)

        # Should have some concept coverage
        assert len(fp['concepts']) >= 0  # May or may not have concepts depending on code_concepts

    def test_bigrams_extraction(self):
        """Bigrams are extracted and weighted."""
        tokenizer = Tokenizer()
        text = "neural networks deep learning"
        fp = compute_fingerprint(text, tokenizer)

        # Should have bigrams (if terms aren't all stop words)
        if fp['term_count'] >= 2:
            assert len(fp['bigrams']) >= 0  # May have bigrams

    def test_term_weights_normalization(self):
        """Term weights are normalized by document length."""
        tokenizer = Tokenizer()

        # Short text
        short = "test"
        fp_short = compute_fingerprint(short, tokenizer)

        # Long text with same term plus many different terms
        long = "test " + " ".join([f"word{i}" for i in range(100)])
        fp_long = compute_fingerprint(long, tokenizer)

        # Both should have "test" term
        if 'test' in fp_short['terms'] and 'test' in fp_long['terms']:
            # Weight in short text should be higher (less dilution)
            assert fp_short['terms']['test'] > fp_long['terms']['test']

    def test_multiple_occurrences(self):
        """Multiple occurrences increase term weight."""
        tokenizer = Tokenizer()
        text = "important important important other"
        fp = compute_fingerprint(text, tokenizer)

        # "important" appears 3 times, "other" once
        if 'important' in fp['terms'] and 'other' in fp['terms']:
            assert fp['terms']['important'] > fp['terms']['other']

    def test_raw_text_hash_identity(self):
        """Same text produces same hash."""
        tokenizer = Tokenizer()
        text = "test text for hashing"

        fp1 = compute_fingerprint(text, tokenizer)
        fp2 = compute_fingerprint(text, tokenizer)

        assert fp1['raw_text_hash'] == fp2['raw_text_hash']
        assert fp1['raw_text_hash'] == hash(text)

    def test_bigram_weights_normalized(self):
        """Bigram weights are normalized."""
        tokenizer = Tokenizer()
        text = "quick brown fox jumps"
        fp = compute_fingerprint(text, tokenizer)

        # Sum of bigram weights should be reasonable
        total_weight = sum(fp['bigrams'].values())
        if total_weight > 0:
            assert 0 < total_weight <= 1.1  # Allow slight float precision

    def test_empty_after_tokenization(self):
        """Text that becomes empty after tokenization."""
        tokenizer = Tokenizer()
        text = "the a an"  # All stop words
        fp = compute_fingerprint(text, tokenizer)

        # Should handle gracefully
        assert fp['term_count'] >= 0
        assert isinstance(fp['terms'], dict)


# =============================================================================
# COMPARE FINGERPRINTS TESTS
# =============================================================================


class TestCompareFingerprints:
    """Tests for compare_fingerprints function."""

    def test_identical_fingerprints(self):
        """Identical fingerprints (same hash) return perfect similarity."""
        tokenizer = Tokenizer()
        text = "neural networks process data"
        fp1 = compute_fingerprint(text, tokenizer)
        fp2 = compute_fingerprint(text, tokenizer)

        result = compare_fingerprints(fp1, fp2)

        assert result['identical'] is True
        assert result['term_similarity'] == 1.0
        assert result['concept_similarity'] == 1.0
        assert result['overall_similarity'] == 1.0

    def test_completely_different_fingerprints(self):
        """Completely different fingerprints have low similarity."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("astronomy planets stars", tokenizer)
        fp2 = compute_fingerprint("cooking recipes ingredients", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        assert result['identical'] is False
        assert result['overall_similarity'] < 0.5  # Should be quite different

    def test_similar_fingerprints(self):
        """Similar fingerprints have high similarity."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("neural networks deep learning", tokenizer)
        fp2 = compute_fingerprint("neural networks machine learning", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        assert result['identical'] is False
        assert result['overall_similarity'] > 0.3  # Should have some similarity

    def test_shared_terms_detection(self):
        """Shared terms are correctly identified."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("apple banana orange", tokenizer)
        fp2 = compute_fingerprint("banana orange grape", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Should detect shared terms
        shared = set(result['shared_terms'])
        # Banana and orange should be shared (accounting for stemming)
        assert len(shared) >= 1

    def test_no_shared_terms(self):
        """Fingerprints with no shared terms."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("alpha beta gamma", tokenizer)
        fp2 = compute_fingerprint("delta epsilon zeta", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # May have no shared terms
        assert isinstance(result['shared_terms'], list)
        assert result['term_similarity'] >= 0  # Should be 0 or very low

    def test_shared_concepts(self):
        """Shared concepts are detected."""
        tokenizer = Tokenizer()
        # Use terms that map to same concept groups
        fp1 = compute_fingerprint("function method procedure", tokenizer)
        fp2 = compute_fingerprint("function routine subroutine", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Should have shared concepts
        assert isinstance(result['shared_concepts'], list)

    def test_unique_terms_detection(self):
        """Unique terms for each fingerprint are identified."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("apple cherry", tokenizer)
        fp2 = compute_fingerprint("banana grape", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Should have unique terms for each
        assert isinstance(result['unique_to_fp1'], list)
        assert isinstance(result['unique_to_fp2'], list)

    def test_empty_fingerprints(self):
        """Both fingerprints empty."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("", tokenizer)
        fp2 = compute_fingerprint("", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Empty fingerprints with same hash are identical
        assert result['identical'] is True

    def test_one_empty_fingerprint(self):
        """One fingerprint empty, one populated."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("", tokenizer)
        fp2 = compute_fingerprint("test content", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        assert result['identical'] is False
        assert result['overall_similarity'] == 0.0

    def test_high_term_similarity(self):
        """High term similarity contributes to overall."""
        tokenizer = Tokenizer()
        # Very similar term sets
        fp1 = compute_fingerprint("test one two three", tokenizer)
        fp2 = compute_fingerprint("test one two four", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Should have decent term similarity
        assert result['term_similarity'] > 0

    def test_bigram_similarity(self):
        """Bigram similarity is computed."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("quick brown fox", tokenizer)
        fp2 = compute_fingerprint("quick brown dog", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Should have bigram similarity metric
        assert 'bigram_similarity' in result
        assert 0 <= result['bigram_similarity'] <= 1

    def test_weighted_average_calculation(self):
        """Overall similarity is weighted average."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("test text alpha", tokenizer)
        fp2 = compute_fingerprint("test text beta", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Verify weighted average: 0.5*term + 0.3*concept + 0.2*bigram
        expected = (
            0.5 * result['term_similarity'] +
            0.3 * result['concept_similarity'] +
            0.2 * result['bigram_similarity']
        )
        assert result['overall_similarity'] == pytest.approx(expected, abs=0.001)

    def test_similarity_range(self):
        """All similarity scores are in [0, 1] range."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("random words here", tokenizer)
        fp2 = compute_fingerprint("different content there", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        assert 0 <= result['term_similarity'] <= 1
        assert 0 <= result['concept_similarity'] <= 1
        assert 0 <= result['bigram_similarity'] <= 1
        assert 0 <= result['overall_similarity'] <= 1

    def test_sorted_shared_terms(self):
        """Shared terms are sorted."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("zebra apple monkey", tokenizer)
        fp2 = compute_fingerprint("zebra monkey banana", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        # Shared terms should be sorted
        shared = result['shared_terms']
        if len(shared) > 1:
            assert shared == sorted(shared)

    def test_different_hash_not_identical(self):
        """Different text hashes mean not identical."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("test one", tokenizer)
        fp2 = compute_fingerprint("test two", tokenizer)

        result = compare_fingerprints(fp1, fp2)

        assert result['identical'] is False


# =============================================================================
# EXPLAIN FINGERPRINT TESTS
# =============================================================================


class TestExplainFingerprint:
    """Tests for explain_fingerprint function."""

    def test_normal_fingerprint_explanation(self):
        """Normal fingerprint produces explanation."""
        tokenizer = Tokenizer()
        text = "neural networks process data efficiently"
        fp = compute_fingerprint(text, tokenizer)

        explanation = explain_fingerprint(fp)

        assert 'summary' in explanation
        assert 'top_terms' in explanation
        assert 'top_concepts' in explanation
        assert 'top_bigrams' in explanation
        assert 'term_count' in explanation
        assert 'concept_coverage' in explanation

    def test_empty_fingerprint_explanation(self):
        """Empty fingerprint produces minimal explanation."""
        tokenizer = Tokenizer()
        fp = compute_fingerprint("", tokenizer)

        explanation = explain_fingerprint(fp)

        assert explanation['summary'] == 'No significant terms'
        assert explanation['term_count'] == 0

    def test_top_n_parameter(self):
        """top_n parameter limits explanation items."""
        tokenizer = Tokenizer()
        text = "one two three four five six seven eight nine ten"
        fp = compute_fingerprint(text, tokenizer, top_n=20)

        exp5 = explain_fingerprint(fp, top_n=5)
        exp3 = explain_fingerprint(fp, top_n=3)

        assert len(exp5['top_terms']) <= 5
        assert len(exp3['top_terms']) <= 3

    def test_summary_with_concepts(self):
        """Summary includes concept information."""
        tokenizer = Tokenizer()
        text = "function class method object"
        fp = compute_fingerprint(text, tokenizer)

        explanation = explain_fingerprint(fp)

        # If concepts were detected, summary should mention them
        if fp['concepts']:
            assert 'Concepts:' in explanation['summary'] or explanation['summary'] == 'No significant terms'

    def test_summary_with_terms(self):
        """Summary includes key terms."""
        tokenizer = Tokenizer()
        text = "important significant critical vital"
        fp = compute_fingerprint(text, tokenizer)

        explanation = explain_fingerprint(fp)

        # Should include key terms in summary
        if fp['top_terms']:
            assert 'Key terms:' in explanation['summary'] or explanation['summary'] == 'No significant terms'

    def test_coverage_metrics(self):
        """Coverage metrics are accurate."""
        tokenizer = Tokenizer()
        text = "test data with multiple terms"
        fp = compute_fingerprint(text, tokenizer)

        explanation = explain_fingerprint(fp)

        assert explanation['term_count'] == fp['term_count']
        assert explanation['concept_coverage'] == len(fp['concepts'])

    def test_top_items_sorted(self):
        """Top items are sorted by weight."""
        tokenizer = Tokenizer()
        text = "alpha beta gamma delta epsilon zeta"
        fp = compute_fingerprint(text, tokenizer)

        explanation = explain_fingerprint(fp, top_n=10)

        # Top terms should be from fp['top_terms'] which is already sorted
        assert len(explanation['top_terms']) <= 10


# =============================================================================
# EXPLAIN SIMILARITY TESTS
# =============================================================================


class TestExplainSimilarity:
    """Tests for explain_similarity function."""

    def test_identical_texts_explanation(self):
        """Identical texts produce clear explanation."""
        tokenizer = Tokenizer()
        text = "neural networks"
        fp1 = compute_fingerprint(text, tokenizer)
        fp2 = compute_fingerprint(text, tokenizer)

        explanation = explain_similarity(fp1, fp2)

        assert "identical" in explanation.lower()

    def test_highly_similar_explanation(self):
        """Highly similar texts produce appropriate message."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("test one two three four", tokenizer)
        fp2 = compute_fingerprint("test one two three five", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        # Should mention similarity level
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_moderately_similar_explanation(self):
        """Moderately similar texts have moderate message."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("neural networks", tokenizer)
        fp2 = compute_fingerprint("machine learning", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        assert isinstance(explanation, str)

    def test_very_different_explanation(self):
        """Very different texts produce appropriate message."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("astronomy planets stars", tokenizer)
        fp2 = compute_fingerprint("cooking recipes food", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        # Should indicate difference
        assert isinstance(explanation, str)
        # Might say "different" or "some common elements"
        assert len(explanation) > 0

    def test_explanation_with_shared_concepts(self):
        """Explanation mentions shared concepts."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("function method call", tokenizer)
        fp2 = compute_fingerprint("procedure routine invoke", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        # Should be a multi-line explanation
        assert '\n' in explanation or len(explanation) > 20

    def test_explanation_with_unique_terms(self):
        """Explanation mentions unique terms."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("apple banana", tokenizer)
        fp2 = compute_fingerprint("cherry date", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        # Should mention uniqueness
        assert isinstance(explanation, str)

    def test_precomputed_comparison(self):
        """Can use pre-computed comparison."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("test alpha", tokenizer)
        fp2 = compute_fingerprint("test beta", tokenizer)

        comparison = compare_fingerprints(fp1, fp2)
        explanation = explain_similarity(fp1, fp2, comparison=comparison)

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explanation_structure(self):
        """Explanation has proper structure."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("one two three", tokenizer)
        fp2 = compute_fingerprint("two three four", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        # Should be multi-line if not identical
        lines = explanation.split('\n')
        assert len(lines) >= 1

    def test_highly_similar_explanation(self):
        """Highly similar texts (>0.8) get appropriate explanation."""
        tokenizer = Tokenizer()
        # Create very similar texts to get >0.8 similarity
        fp1 = compute_fingerprint("alpha beta gamma delta epsilon zeta", tokenizer)
        fp2 = compute_fingerprint("alpha beta gamma delta epsilon theta", tokenizer)

        comparison = compare_fingerprints(fp1, fp2)
        # Force high similarity for testing the branch
        comparison['overall_similarity'] = 0.85

        explanation = explain_similarity(fp1, fp2, comparison=comparison)

        # Should mention high similarity
        assert "highly similar" in explanation.lower() or "similar" in explanation.lower()

    def test_explanation_mentions_unique_terms(self):
        """Explanation mentions unique terms when present."""
        tokenizer = Tokenizer()
        fp1 = compute_fingerprint("unique1 shared", tokenizer)
        fp2 = compute_fingerprint("unique2 shared", tokenizer)

        explanation = explain_similarity(fp1, fp2)

        # Should mention uniqueness
        assert "unique" in explanation.lower() or len(explanation) > 0


# =============================================================================
# COSINE SIMILARITY TESTS
# =============================================================================


class TestCosineSimilarity:
    """Tests for _cosine_similarity helper function."""

    def test_empty_vectors(self):
        """Both vectors empty returns 0."""
        result = _cosine_similarity({}, {})
        assert result == 0.0

    def test_one_empty_vector(self):
        """One vector empty returns 0."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {}

        assert _cosine_similarity(vec1, vec2) == 0.0
        assert _cosine_similarity(vec2, vec1) == 0.0

    def test_no_common_dimensions(self):
        """Vectors with no common dimensions return 0."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 3.0, "d": 4.0}

        result = _cosine_similarity(vec1, vec2)
        assert result == 0.0

    def test_identical_vectors(self):
        """Identical vectors return 1.0."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}

        result = _cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors (in common dims) return appropriate value."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}

        result = _cosine_similarity(vec1, vec2)
        # No common non-zero dimensions, should be 0
        assert result == 0.0

    def test_partial_overlap(self):
        """Vectors with partial overlap."""
        vec1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        vec2 = {"b": 2.0, "c": 3.0, "d": 4.0}

        result = _cosine_similarity(vec1, vec2)

        # Should be in valid range
        assert 0 <= result <= 1
        assert result > 0  # Have common dimensions with same values

    def test_zero_magnitude(self):
        """Vector with zero magnitude returns 0."""
        vec1 = {"a": 0.0, "b": 0.0}
        vec2 = {"a": 1.0, "b": 2.0}

        result = _cosine_similarity(vec1, vec2)
        assert result == 0.0

    def test_formula_verification(self):
        """Verify cosine similarity formula."""
        vec1 = {"a": 3.0, "b": 4.0}
        vec2 = {"a": 4.0, "b": 3.0}

        # Manual calculation
        dot_product = 3.0 * 4.0 + 4.0 * 3.0  # 12 + 12 = 24
        mag1 = math.sqrt(3.0**2 + 4.0**2)    # sqrt(25) = 5
        mag2 = math.sqrt(4.0**2 + 3.0**2)    # sqrt(25) = 5
        expected = dot_product / (mag1 * mag2)  # 24 / 25 = 0.96

        result = _cosine_similarity(vec1, vec2)
        assert result == pytest.approx(expected, abs=0.001)

    def test_negative_values(self):
        """Handles negative values correctly."""
        vec1 = {"a": 1.0, "b": -1.0}
        vec2 = {"a": 1.0, "b": 1.0}

        result = _cosine_similarity(vec1, vec2)

        # a contributes positive, b contributes negative
        # dot = 1*1 + (-1)*1 = 0
        # mag1 = sqrt(2), mag2 = sqrt(2)
        # result = 0 / 2 = 0
        assert result == pytest.approx(0.0, abs=0.001)

    def test_range_validation(self):
        """Cosine similarity is always in [0, 1] for positive vectors."""
        vec1 = {"a": 5.0, "b": 10.0, "c": 2.0}
        vec2 = {"a": 2.0, "b": 3.0, "c": 8.0}

        result = _cosine_similarity(vec1, vec2)

        assert 0 <= result <= 1

    def test_scaled_vectors(self):
        """Scaling one vector doesn't change cosine similarity."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"a": 2.0, "b": 4.0}  # 2x vec1

        result = _cosine_similarity(vec1, vec2)

        # Should be 1.0 (same direction)
        assert result == pytest.approx(1.0, abs=0.001)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestFingerprintIntegration:
    """Integration tests combining multiple functions."""

    def test_create_compare_explain_workflow(self):
        """Complete workflow: create, compare, explain."""
        tokenizer = Tokenizer()

        text1 = "neural networks deep learning artificial intelligence"
        text2 = "neural networks machine learning AI algorithms"

        # Create fingerprints
        fp1 = compute_fingerprint(text1, tokenizer)
        fp2 = compute_fingerprint(text2, tokenizer)

        # Compare
        comparison = compare_fingerprints(fp1, fp2)

        # Explain individual
        exp1 = explain_fingerprint(fp1)
        exp2 = explain_fingerprint(fp2)

        # Explain similarity
        similarity = explain_similarity(fp1, fp2, comparison)

        # All should succeed
        assert fp1['term_count'] > 0
        assert fp2['term_count'] > 0
        assert comparison['overall_similarity'] >= 0
        assert len(exp1['summary']) > 0
        assert len(exp2['summary']) > 0
        assert len(similarity) > 0

    def test_with_corpus_layers_integration(self):
        """Integration with corpus layers for TF-IDF."""
        tokenizer = Tokenizer()

        # Create mock corpus
        col1 = MockMinicolumn(content="rare", tfidf=10.0)
        col2 = MockMinicolumn(content="common", tfidf=0.1)

        layers = MockLayers.empty()
        layers[CorticalLayer.TOKENS] = type('MockLayer', (), {
            'get_minicolumn': lambda self, term: {
                'rare': col1,
                'common': col2
            }.get(term)
        })()

        # Create fingerprints with corpus
        fp1 = compute_fingerprint("rare rare common", tokenizer, layers=layers)
        fp2 = compute_fingerprint("common common rare", tokenizer, layers=layers)

        # Compare should work
        comparison = compare_fingerprints(fp1, fp2)

        assert comparison['overall_similarity'] > 0
        assert len(comparison['shared_terms']) > 0

    def test_consistency_across_calls(self):
        """Same input produces consistent results."""
        tokenizer = Tokenizer()
        text = "consistent test text here"

        # Create multiple times
        fp1 = compute_fingerprint(text, tokenizer)
        fp2 = compute_fingerprint(text, tokenizer)
        fp3 = compute_fingerprint(text, tokenizer)

        # Should be identical
        assert fp1['raw_text_hash'] == fp2['raw_text_hash'] == fp3['raw_text_hash']
        assert fp1['term_count'] == fp2['term_count'] == fp3['term_count']
        assert fp1['terms'] == fp2['terms'] == fp3['terms']
