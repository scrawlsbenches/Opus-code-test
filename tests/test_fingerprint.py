"""
Tests for fingerprint module.

Tests the semantic fingerprinting functionality for code comparison.
"""

import unittest
from cortical.fingerprint import (
    compute_fingerprint,
    compare_fingerprints,
    explain_fingerprint,
    explain_similarity,
    SemanticFingerprint,
)
from cortical.tokenizer import Tokenizer


class TestComputeFingerprint(unittest.TestCase):
    """Test the compute_fingerprint function."""

    def setUp(self):
        """Set up test tokenizer."""
        self.tokenizer = Tokenizer()

    def test_basic_fingerprint(self):
        """Test basic fingerprint computation."""
        text = "The function validates user input and handles errors."
        fp = compute_fingerprint(text, self.tokenizer)

        self.assertIn('terms', fp)
        self.assertIn('concepts', fp)
        self.assertIn('bigrams', fp)
        self.assertIn('top_terms', fp)
        self.assertIn('term_count', fp)
        self.assertIn('raw_text_hash', fp)

    def test_fingerprint_contains_terms(self):
        """Test that fingerprint contains expected terms."""
        text = "fetch user data from database"
        fp = compute_fingerprint(text, self.tokenizer)

        self.assertIn('fetch', fp['terms'])
        self.assertIn('user', fp['terms'])
        self.assertIn('data', fp['terms'])
        self.assertIn('database', fp['terms'])

    def test_fingerprint_concepts(self):
        """Test that fingerprint captures concept membership."""
        text = "fetch data and save results"
        fp = compute_fingerprint(text, self.tokenizer)

        # 'fetch' is in retrieval group, 'save' is in storage group
        # (if code_concepts recognizes them)
        self.assertIsInstance(fp['concepts'], dict)

    def test_fingerprint_bigrams(self):
        """Test that fingerprint captures bigrams."""
        text = "neural networks process data efficiently"
        fp = compute_fingerprint(text, self.tokenizer)

        self.assertIn('bigrams', fp)
        self.assertIsInstance(fp['bigrams'], dict)

    def test_fingerprint_top_terms_limit(self):
        """Test that top_n limits top terms."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        fp = compute_fingerprint(text, self.tokenizer, top_n=5)

        self.assertLessEqual(len(fp['top_terms']), 5)

    def test_empty_text_fingerprint(self):
        """Test fingerprint of empty text."""
        fp = compute_fingerprint("", self.tokenizer)

        self.assertEqual(fp['term_count'], 0)
        self.assertEqual(fp['terms'], {})

    def test_fingerprint_term_weights_positive(self):
        """Test that term weights are positive."""
        text = "validate user input data"
        fp = compute_fingerprint(text, self.tokenizer)

        for term, weight in fp['terms'].items():
            self.assertGreater(weight, 0)

    def test_fingerprint_with_layers(self):
        """Test fingerprint with corpus layers for TF-IDF."""
        from cortical import CorticalTextProcessor
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test document content")
        processor.compute_all()

        # Compute fingerprint with layers
        fp = compute_fingerprint(
            "test content",
            processor.tokenizer,
            processor.layers
        )

        self.assertIn('terms', fp)
        self.assertGreater(len(fp['terms']), 0)


class TestCompareFingerprints(unittest.TestCase):
    """Test the compare_fingerprints function."""

    def setUp(self):
        """Set up test tokenizer."""
        self.tokenizer = Tokenizer()

    def test_identical_texts(self):
        """Test comparing identical texts."""
        text = "validate user input data"
        fp1 = compute_fingerprint(text, self.tokenizer)
        fp2 = compute_fingerprint(text, self.tokenizer)

        result = compare_fingerprints(fp1, fp2)

        self.assertTrue(result['identical'])
        self.assertEqual(result['overall_similarity'], 1.0)

    def test_similar_texts(self):
        """Test comparing similar texts."""
        fp1 = compute_fingerprint("validate user input", self.tokenizer)
        fp2 = compute_fingerprint("check user data", self.tokenizer)

        result = compare_fingerprints(fp1, fp2)

        self.assertFalse(result['identical'])
        self.assertIn('user', result['shared_terms'])
        self.assertGreater(result['term_similarity'], 0)

    def test_different_texts(self):
        """Test comparing different texts."""
        fp1 = compute_fingerprint("neural network training", self.tokenizer)
        fp2 = compute_fingerprint("database query optimization", self.tokenizer)

        result = compare_fingerprints(fp1, fp2)

        self.assertFalse(result['identical'])
        # Should have lower similarity
        self.assertLess(result['overall_similarity'], 0.5)

    def test_comparison_contains_metrics(self):
        """Test that comparison contains all expected metrics."""
        fp1 = compute_fingerprint("text one", self.tokenizer)
        fp2 = compute_fingerprint("text two", self.tokenizer)

        result = compare_fingerprints(fp1, fp2)

        self.assertIn('identical', result)
        self.assertIn('term_similarity', result)
        self.assertIn('concept_similarity', result)
        self.assertIn('overall_similarity', result)
        self.assertIn('shared_terms', result)

    def test_similarity_in_valid_range(self):
        """Test that similarity scores are in [0, 1]."""
        fp1 = compute_fingerprint("fetch user data", self.tokenizer)
        fp2 = compute_fingerprint("save user results", self.tokenizer)

        result = compare_fingerprints(fp1, fp2)

        self.assertGreaterEqual(result['term_similarity'], 0)
        self.assertLessEqual(result['term_similarity'], 1)
        self.assertGreaterEqual(result['overall_similarity'], 0)
        self.assertLessEqual(result['overall_similarity'], 1)

    def test_shared_terms_correct(self):
        """Test that shared terms are correctly identified."""
        fp1 = compute_fingerprint("user data validation", self.tokenizer)
        fp2 = compute_fingerprint("user input checking", self.tokenizer)

        result = compare_fingerprints(fp1, fp2)

        self.assertIn('user', result['shared_terms'])


class TestExplainFingerprint(unittest.TestCase):
    """Test the explain_fingerprint function."""

    def setUp(self):
        """Set up test tokenizer."""
        self.tokenizer = Tokenizer()

    def test_explanation_structure(self):
        """Test that explanation has expected structure."""
        text = "validate user input and handle errors"
        fp = compute_fingerprint(text, self.tokenizer)
        explanation = explain_fingerprint(fp)

        self.assertIn('summary', explanation)
        self.assertIn('top_terms', explanation)
        self.assertIn('top_concepts', explanation)
        self.assertIn('term_count', explanation)

    def test_explanation_top_n_limit(self):
        """Test that top_n limits items in explanation."""
        text = "word1 word2 word3 word4 word5 word6"
        fp = compute_fingerprint(text, self.tokenizer)
        explanation = explain_fingerprint(fp, top_n=3)

        self.assertLessEqual(len(explanation['top_terms']), 3)

    def test_summary_is_string(self):
        """Test that summary is a string."""
        text = "process data"
        fp = compute_fingerprint(text, self.tokenizer)
        explanation = explain_fingerprint(fp)

        self.assertIsInstance(explanation['summary'], str)


class TestExplainSimilarity(unittest.TestCase):
    """Test the explain_similarity function."""

    def setUp(self):
        """Set up test tokenizer."""
        self.tokenizer = Tokenizer()

    def test_explanation_is_string(self):
        """Test that similarity explanation is a string."""
        fp1 = compute_fingerprint("fetch user data", self.tokenizer)
        fp2 = compute_fingerprint("load user info", self.tokenizer)

        explanation = explain_similarity(fp1, fp2)

        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)

    def test_identical_texts_explanation(self):
        """Test explanation for identical texts."""
        text = "validate input"
        fp1 = compute_fingerprint(text, self.tokenizer)
        fp2 = compute_fingerprint(text, self.tokenizer)

        explanation = explain_similarity(fp1, fp2)

        self.assertIn('identical', explanation.lower())


class TestProcessorIntegration(unittest.TestCase):
    """Test fingerprint integration with processor."""

    def setUp(self):
        """Set up test processor."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        self.processor.process_document("auth", """
            Authentication module handles user login and credentials.
            Validates tokens and manages sessions.
        """)
        self.processor.process_document("data", """
            Data processing module fetches and transforms data.
            Handles database queries and result formatting.
        """)
        self.processor.compute_all()

    def test_processor_get_fingerprint(self):
        """Test processor get_fingerprint method."""
        fp = self.processor.get_fingerprint("validate user credentials")

        self.assertIn('terms', fp)
        self.assertIn('concepts', fp)
        self.assertGreater(fp['term_count'], 0)

    def test_processor_compare_fingerprints(self):
        """Test processor compare_fingerprints method."""
        fp1 = self.processor.get_fingerprint("user authentication")
        fp2 = self.processor.get_fingerprint("user validation")

        result = self.processor.compare_fingerprints(fp1, fp2)

        self.assertIn('overall_similarity', result)
        self.assertIn('user', result['shared_terms'])

    def test_processor_explain_fingerprint(self):
        """Test processor explain_fingerprint method."""
        fp = self.processor.get_fingerprint("fetch data from database")
        explanation = self.processor.explain_fingerprint(fp)

        self.assertIn('summary', explanation)
        self.assertIn('top_terms', explanation)

    def test_processor_explain_similarity(self):
        """Test processor explain_similarity method."""
        fp1 = self.processor.get_fingerprint("fetch data")
        fp2 = self.processor.get_fingerprint("load data")

        explanation = self.processor.explain_similarity(fp1, fp2)

        self.assertIsInstance(explanation, str)

    def test_processor_find_similar_texts(self):
        """Test processor find_similar_texts method."""
        candidates = [
            ("auth_code", "validate user credentials and create session"),
            ("data_code", "fetch records from database and transform"),
            ("ui_code", "render button and handle click event"),
        ]

        results = self.processor.find_similar_texts(
            "authenticate user login",
            candidates,
            top_n=2
        )

        self.assertLessEqual(len(results), 2)
        # Results should be sorted by similarity
        if len(results) >= 2:
            self.assertGreaterEqual(results[0][1], results[1][1])


if __name__ == '__main__':
    unittest.main()
