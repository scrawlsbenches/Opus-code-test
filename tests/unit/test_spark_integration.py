"""
Unit tests for SparkSLM integration with CorticalTextProcessor.

Tests the SparkMixin methods that connect SparkSLM to the main processor.
"""

import os
import tempfile
import unittest

from cortical import CorticalTextProcessor


class TestSparkMixinBasic(unittest.TestCase):
    """Test basic SparkMixin functionality."""

    def test_spark_disabled_by_default(self):
        """Verify spark is disabled by default."""
        p = CorticalTextProcessor()
        self.assertFalse(p.spark_enabled)

    def test_spark_enabled_in_constructor(self):
        """Test enabling spark via constructor."""
        p = CorticalTextProcessor(spark=True)
        self.assertTrue(p.spark_enabled)

    def test_enable_spark_method(self):
        """Test enabling spark via method call."""
        p = CorticalTextProcessor()
        self.assertFalse(p.spark_enabled)
        p.enable_spark()
        self.assertTrue(p.spark_enabled)

    def test_disable_spark(self):
        """Test disabling spark."""
        p = CorticalTextProcessor(spark=True)
        self.assertTrue(p.spark_enabled)
        p.disable_spark()
        self.assertFalse(p.spark_enabled)

    def test_enable_spark_custom_order(self):
        """Test enabling spark with custom n-gram order."""
        p = CorticalTextProcessor()
        p.enable_spark(ngram_order=4)
        self.assertTrue(p.spark_enabled)
        stats = p.get_spark_stats()
        self.assertEqual(stats['ngram_order'], 4)


class TestSparkMixinTraining(unittest.TestCase):
    """Test SparkMixin training functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor(spark=True)
        self.processor.process_document(
            'doc1', 'Neural networks process information quickly.')
        self.processor.process_document(
            'doc2', 'Machine learning uses neural networks for training models.')
        self.processor.process_document(
            'doc3', 'Deep learning is a subset of machine learning.')
        self.processor.compute_all()

    def test_train_spark(self):
        """Test training spark on corpus."""
        stats = self.processor.train_spark()
        self.assertIn('documents', stats)
        self.assertIn('tokens', stats)
        self.assertEqual(stats['documents'], 3)
        self.assertGreater(stats['tokens'], 0)

    def test_train_spark_min_length_filter(self):
        """Test training with minimum document length filter."""
        # Add a short document
        self.processor.process_document('short', 'Hi')

        # Train with high minimum length - should exclude short doc
        stats = self.processor.train_spark(min_doc_length=20)
        self.assertEqual(stats['documents'], 3)

    def test_train_spark_requires_enabled(self):
        """Test that training requires spark to be enabled."""
        p = CorticalTextProcessor()
        with self.assertRaises(RuntimeError):
            p.train_spark()


class TestSparkMixinPriming(unittest.TestCase):
    """Test SparkMixin priming functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor(spark=True)
        self.processor.process_document(
            'doc1', 'Neural networks process information quickly.')
        self.processor.process_document(
            'doc2', 'Machine learning uses neural networks for training.')
        self.processor.compute_all()
        self.processor.train_spark()

    def test_prime_query(self):
        """Test query priming."""
        hints = self.processor.prime_query('neural')
        self.assertIsInstance(hints, dict)
        self.assertIn('completions', hints)
        self.assertIn('alignment', hints)
        self.assertIn('keywords', hints)

    def test_prime_query_requires_enabled(self):
        """Test that priming requires spark to be enabled."""
        p = CorticalTextProcessor()
        with self.assertRaises(RuntimeError):
            p.prime_query('test')

    def test_complete_query(self):
        """Test query completion."""
        completed = self.processor.complete_query('neural')
        self.assertIsInstance(completed, str)
        self.assertTrue(completed.startswith('neural'))


class TestSparkMixinAlignment(unittest.TestCase):
    """Test SparkMixin alignment functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor(spark=True)

    def test_load_alignment_from_directory(self):
        """Test loading alignment from samples/alignment directory."""
        if not os.path.exists('samples/alignment'):
            self.skipTest('samples/alignment directory not found')

        count = self.processor.load_alignment('samples/alignment')
        self.assertGreater(count, 0)

    def test_get_alignment_context(self):
        """Test getting alignment context for a term."""
        if not os.path.exists('samples/alignment'):
            self.skipTest('samples/alignment directory not found')

        self.processor.load_alignment('samples/alignment')
        ctx = self.processor.get_alignment_context('spark')
        self.assertIsInstance(ctx, list)
        self.assertGreater(len(ctx), 0)
        self.assertEqual(ctx[0]['key'], 'spark')
        self.assertEqual(ctx[0]['type'], 'definition')

    def test_get_alignment_context_empty(self):
        """Test getting alignment context for unknown term."""
        ctx = self.processor.get_alignment_context('nonexistent_term_xyz')
        self.assertEqual(ctx, [])

    def test_get_alignment_summary(self):
        """Test getting alignment summary."""
        if not os.path.exists('samples/alignment'):
            self.skipTest('samples/alignment directory not found')

        self.processor.load_alignment('samples/alignment')
        summary = self.processor.get_alignment_summary()
        self.assertIsInstance(summary, str)
        self.assertIn('Alignment', summary)


class TestSparkMixinExpansion(unittest.TestCase):
    """Test SparkMixin query expansion with spark."""

    def setUp(self):
        self.processor = CorticalTextProcessor(spark=True)
        self.processor.process_document(
            'doc1', 'Neural networks process information quickly.')
        self.processor.process_document(
            'doc2', 'Machine learning uses neural networks for training.')
        self.processor.compute_all()
        self.processor.train_spark()

    def test_expand_query_with_spark(self):
        """Test query expansion with spark priming."""
        expanded = self.processor.expand_query_with_spark('neural')
        self.assertIsInstance(expanded, dict)
        self.assertGreater(len(expanded), 0)

    def test_expand_query_with_spark_boost(self):
        """Test query expansion with custom spark boost."""
        expanded = self.processor.expand_query_with_spark(
            'neural', spark_boost=0.5)
        self.assertIsInstance(expanded, dict)


class TestSparkMixinPersistence(unittest.TestCase):
    """Test SparkMixin save/load functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor(spark=True)
        self.processor.process_document('doc1', 'Test document content.')
        self.processor.compute_all()
        self.processor.train_spark()

    def test_save_and_load_spark(self):
        """Test saving and loading spark state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spark_path = os.path.join(tmpdir, 'spark_state')

            # Save
            self.processor.save_spark(spark_path)
            self.assertTrue(os.path.exists(spark_path))

            # Load into new processor
            p2 = CorticalTextProcessor(spark=True)
            p2.load_spark(spark_path)
            self.assertTrue(p2.spark_enabled)

            stats = p2.get_spark_stats()
            self.assertGreater(stats['vocabulary_size'], 0)


class TestSparkMixinStats(unittest.TestCase):
    """Test SparkMixin statistics."""

    def setUp(self):
        self.processor = CorticalTextProcessor(spark=True)
        self.processor.process_document('doc1', 'Test document content.')
        self.processor.compute_all()
        self.processor.train_spark()

    def test_get_spark_stats(self):
        """Test getting spark statistics."""
        stats = self.processor.get_spark_stats()
        self.assertTrue(stats['enabled'])
        self.assertEqual(stats['ngram_order'], 3)
        self.assertIn('vocabulary_size', stats)
        self.assertIn('alignment_breakdown', stats)
        self.assertIn('context_count', stats)

    def test_get_spark_stats_requires_enabled(self):
        """Test that stats require spark to be enabled."""
        p = CorticalTextProcessor()
        with self.assertRaises(RuntimeError):
            p.get_spark_stats()


class TestSparkMixinBackwardsCompatibility(unittest.TestCase):
    """Test that processor works normally without spark."""

    def test_basic_operations_without_spark(self):
        """Test that all basic operations work without spark."""
        p = CorticalTextProcessor()
        p.process_document('doc1', 'Test document.')
        p.compute_all()
        results = p.find_documents_for_query('test')
        self.assertGreater(len(results), 0)

    def test_expand_query_without_spark(self):
        """Test that regular expand_query works without spark."""
        p = CorticalTextProcessor()
        p.process_document('doc1', 'Neural networks process data.')
        p.compute_all()
        expanded = p.expand_query('neural')
        self.assertIsInstance(expanded, dict)


if __name__ == '__main__':
    unittest.main()
