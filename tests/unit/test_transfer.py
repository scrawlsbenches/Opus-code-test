"""
Unit tests for cross-project transfer learning.

Tests vocabulary analysis, portable models, and transfer adaptation.
"""

import os
import tempfile
import unittest

from cortical.spark.ngram import NGramModel
from cortical.spark.transfer import (
    VocabularyAnalyzer,
    VocabularyAnalysis,
    PortableModel,
    TransferAdapter,
    TransferMetrics,
    PROGRAMMING_VOCABULARY,
    create_portable_model,
    transfer_knowledge,
)


class TestProgrammingVocabulary(unittest.TestCase):
    """Test the built-in programming vocabulary."""

    def test_vocabulary_not_empty(self):
        """Verify programming vocabulary is populated."""
        self.assertGreater(len(PROGRAMMING_VOCABULARY), 50)

    def test_common_keywords_present(self):
        """Verify common keywords are in vocabulary."""
        common = ['if', 'else', 'for', 'while', 'def', 'class', 'return']
        for keyword in common:
            self.assertIn(keyword, PROGRAMMING_VOCABULARY)

    def test_common_patterns_present(self):
        """Verify common patterns are in vocabulary."""
        patterns = ['get', 'set', 'add', 'remove', 'create', 'delete']
        for pattern in patterns:
            self.assertIn(pattern, PROGRAMMING_VOCABULARY)


class TestVocabularyAnalyzer(unittest.TestCase):
    """Test vocabulary analysis."""

    def setUp(self):
        self.analyzer = VocabularyAnalyzer(min_frequency=1)
        self.model = NGramModel(n=3)
        # Train on mixed vocabulary - pass as list of strings
        self.model.train(["def function return value for item in list"])
        self.model.train(["minicolumn pagerank tfidf neural network"])

    def test_init_default(self):
        """Test default initialization."""
        analyzer = VocabularyAnalyzer()
        self.assertEqual(analyzer.programming_vocab, PROGRAMMING_VOCABULARY)

    def test_init_custom_vocab(self):
        """Test custom vocabulary initialization."""
        custom = {'custom', 'terms'}
        analyzer = VocabularyAnalyzer(programming_vocab=custom)
        self.assertEqual(analyzer.programming_vocab, custom)

    def test_analyze_returns_analysis(self):
        """Test analyze returns VocabularyAnalysis."""
        analysis = self.analyzer.analyze(self.model)
        self.assertIsInstance(analysis, VocabularyAnalysis)

    def test_analyze_counts_terms(self):
        """Test analyze counts terms correctly."""
        analysis = self.analyzer.analyze(self.model)
        self.assertGreater(analysis.total_terms, 0)

    def test_analyze_separates_programming_terms(self):
        """Test programming terms are identified."""
        analysis = self.analyzer.analyze(self.model)
        # Should have some programming terms (def, for, in, return, etc.)
        self.assertGreater(analysis.programming_terms, 0)

    def test_analyze_separates_project_terms(self):
        """Test project-specific terms are identified."""
        analysis = self.analyzer.analyze(self.model)
        # Should have project terms (minicolumn, pagerank, etc.)
        self.assertGreater(analysis.project_specific_terms, 0)

    def test_analyze_programming_ratio(self):
        """Test programming ratio is calculated."""
        analysis = self.analyzer.analyze(self.model)
        self.assertGreaterEqual(analysis.programming_ratio, 0.0)
        self.assertLessEqual(analysis.programming_ratio, 1.0)

    def test_get_transferable_terms(self):
        """Test getting transferable terms."""
        model = NGramModel(n=3)
        # Train with many occurrences
        for _ in range(10):
            model.train(["def function return value"])

        transferable = self.analyzer.get_transferable_terms(model, min_frequency=3)
        self.assertIn('def', transferable)
        self.assertIn('return', transferable)

    def test_get_transferable_excludes_rare_terms(self):
        """Test rare terms are not transferable."""
        model = NGramModel(n=3)
        model.train(["def function return value"])

        transferable = self.analyzer.get_transferable_terms(model, min_frequency=10)
        # Terms only appeared once, should not be transferable
        self.assertEqual(len(transferable), 0)

    def test_calculate_overlap_identical(self):
        """Test overlap of identical models is 1.0."""
        model = NGramModel(n=3)
        model.train(["test words here"])

        overlap = self.analyzer.calculate_overlap(model, model)
        self.assertEqual(overlap, 1.0)

    def test_calculate_overlap_disjoint(self):
        """Test overlap of disjoint models is 0.0."""
        model_a = NGramModel(n=3)
        model_a.train(["alpha beta gamma"])

        model_b = NGramModel(n=3)
        model_b.train(["one two three"])

        overlap = self.analyzer.calculate_overlap(model_a, model_b)
        self.assertEqual(overlap, 0.0)

    def test_calculate_overlap_partial(self):
        """Test partial overlap is between 0 and 1."""
        model_a = NGramModel(n=3)
        model_a.train(["alpha beta shared"])

        model_b = NGramModel(n=3)
        model_b.train(["shared gamma delta"])

        overlap = self.analyzer.calculate_overlap(model_a, model_b)
        self.assertGreater(overlap, 0.0)
        self.assertLess(overlap, 1.0)


class TestVocabularyAnalysis(unittest.TestCase):
    """Test VocabularyAnalysis dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = VocabularyAnalysis(
            total_terms=100,
            programming_terms=40,
            project_specific_terms=60,
            programming_ratio=0.4,
            top_programming_terms=[('def', 10), ('return', 8)],
            top_project_terms=[('minicolumn', 5), ('pagerank', 3)],
        )

        d = analysis.to_dict()
        self.assertEqual(d['total_terms'], 100)
        self.assertEqual(d['programming_ratio'], 0.4)
        self.assertEqual(len(d['top_programming_terms']), 2)


class TestPortableModel(unittest.TestCase):
    """Test portable model creation and serialization."""

    def setUp(self):
        self.model = NGramModel(n=3)
        for _ in range(5):
            self.model.train(["def function return value for item in list"])

    def test_from_ngram_model(self):
        """Test creating portable model from n-gram model."""
        portable = PortableModel.from_ngram_model(self.model)
        self.assertIsInstance(portable, PortableModel)

    def test_from_ngram_model_with_source(self):
        """Test portable model includes source project name."""
        portable = PortableModel.from_ngram_model(
            self.model,
            source_project="test_project"
        )
        self.assertEqual(portable.source_project, "test_project")

    def test_portable_has_shared_counts(self):
        """Test portable model has shared counts."""
        portable = PortableModel.from_ngram_model(self.model)
        # Should have some shared programming term n-grams
        self.assertGreater(len(portable.shared_counts), 0)

    def test_portable_has_shared_vocab(self):
        """Test portable model has shared vocabulary."""
        portable = PortableModel.from_ngram_model(self.model)
        self.assertGreater(len(portable.shared_vocab), 0)

    def test_save_and_load(self):
        """Test saving and loading portable model."""
        portable = PortableModel.from_ngram_model(
            self.model,
            source_project="test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "portable")
            portable.save(path)

            loaded = PortableModel.load(path)

            self.assertEqual(loaded.ngram_order, portable.ngram_order)
            self.assertEqual(loaded.source_project, "test")
            self.assertEqual(len(loaded.shared_vocab), len(portable.shared_vocab))

    def test_get_stats(self):
        """Test getting portable model statistics."""
        portable = PortableModel.from_ngram_model(self.model, source_project="test")
        stats = portable.get_stats()

        self.assertIn('ngram_order', stats)
        self.assertIn('context_count', stats)
        self.assertIn('vocab_size', stats)
        self.assertEqual(stats['source_project'], "test")


class TestTransferAdapter(unittest.TestCase):
    """Test transfer adaptation."""

    def setUp(self):
        # Source model with programming patterns
        self.source = NGramModel(n=3)
        for _ in range(10):
            self.source.train(["def function return value for item in list"])

        # Target model with different content
        self.target = NGramModel(n=3)
        self.target.train(["custom project specific terms here"])

        self.portable = PortableModel.from_ngram_model(
            self.source,
            source_project="source"
        )

    def test_init(self):
        """Test adapter initialization."""
        adapter = TransferAdapter(self.portable)
        self.assertEqual(adapter.blend_weight, 0.3)

    def test_init_custom_weight(self):
        """Test adapter with custom blend weight."""
        adapter = TransferAdapter(self.portable, blend_weight=0.5)
        self.assertEqual(adapter.blend_weight, 0.5)

    def test_adapt_creates_new_model(self):
        """Test adapt creates new model by default."""
        adapter = TransferAdapter(self.portable)

        original_counts = len(self.target.counts)
        adapted = adapter.adapt(self.target, in_place=False)

        # Original should be unchanged
        self.assertEqual(len(self.target.counts), original_counts)
        # Adapted should be different object
        self.assertIsNot(adapted, self.target)

    def test_adapt_in_place(self):
        """Test in-place adaptation."""
        adapter = TransferAdapter(self.portable)

        adapted = adapter.adapt(self.target, in_place=True)

        # Should be same object
        self.assertIs(adapted, self.target)

    def test_adapt_adds_vocabulary(self):
        """Test adaptation adds transferred vocabulary."""
        adapter = TransferAdapter(self.portable)

        original_vocab_size = len(self.target.vocab)
        adapted = adapter.adapt(self.target, in_place=False)

        # Should have more vocabulary
        self.assertGreaterEqual(len(adapted.vocab), original_vocab_size)

    def test_measure_effectiveness(self):
        """Test measuring transfer effectiveness."""
        adapter = TransferAdapter(self.portable)

        metrics = adapter.measure_effectiveness(self.target)

        self.assertIsInstance(metrics, TransferMetrics)

    def test_metrics_vocabulary_overlap(self):
        """Test vocabulary overlap metric."""
        adapter = TransferAdapter(self.portable)
        metrics = adapter.measure_effectiveness(self.target)

        self.assertGreaterEqual(metrics.vocabulary_overlap, 0.0)
        self.assertLessEqual(metrics.vocabulary_overlap, 1.0)

    def test_metrics_ngram_coverage(self):
        """Test n-gram coverage metric."""
        adapter = TransferAdapter(self.portable)
        metrics = adapter.measure_effectiveness(self.target)

        self.assertGreaterEqual(metrics.ngram_coverage, 0.0)
        self.assertLessEqual(metrics.ngram_coverage, 1.0)

    def test_get_metrics_before_measure(self):
        """Test get_metrics returns None before measuring."""
        adapter = TransferAdapter(self.portable)
        self.assertIsNone(adapter.get_metrics())

    def test_get_metrics_after_measure(self):
        """Test get_metrics returns metrics after measuring."""
        adapter = TransferAdapter(self.portable)
        adapter.measure_effectiveness(self.target)

        self.assertIsNotNone(adapter.get_metrics())

    def test_get_transfer_summary_before_measure(self):
        """Test summary before measuring."""
        adapter = TransferAdapter(self.portable)
        summary = adapter.get_transfer_summary()

        self.assertIn("No transfer metrics", summary)

    def test_get_transfer_summary_after_measure(self):
        """Test summary after measuring."""
        adapter = TransferAdapter(self.portable)
        adapter.measure_effectiveness(self.target)

        summary = adapter.get_transfer_summary()

        self.assertIn("Transfer Summary", summary)
        self.assertIn("Vocabulary Overlap", summary)
        self.assertIn("N-gram Coverage", summary)


class TestTransferMetrics(unittest.TestCase):
    """Test TransferMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TransferMetrics(
            vocabulary_overlap=0.5,
            ngram_coverage=0.3,
            perplexity_improvement=0.1,
            adapted_terms=100,
        )

        d = metrics.to_dict()
        self.assertEqual(d['vocabulary_overlap'], 0.5)
        self.assertEqual(d['adapted_terms'], 100)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        self.source = NGramModel(n=3)
        for _ in range(5):
            self.source.train(["def function return value"])

        self.target = NGramModel(n=3)
        self.target.train(["custom terms here"])

    def test_create_portable_model(self):
        """Test create_portable_model function."""
        portable = create_portable_model(self.source, "test_project")

        self.assertIsInstance(portable, PortableModel)
        self.assertEqual(portable.source_project, "test_project")

    def test_transfer_knowledge(self):
        """Test transfer_knowledge function."""
        adapted, metrics = transfer_knowledge(
            self.source,
            self.target,
            blend_weight=0.3,
            source_project="source"
        )

        self.assertIsInstance(adapted, NGramModel)
        self.assertIsInstance(metrics, TransferMetrics)

    def test_transfer_knowledge_custom_weight(self):
        """Test transfer with custom blend weight."""
        adapted, metrics = transfer_knowledge(
            self.source,
            self.target,
            blend_weight=0.7,
        )

        # Higher weight should transfer more
        self.assertIsInstance(adapted, NGramModel)


class TestTransferScenarios(unittest.TestCase):
    """Test realistic transfer scenarios."""

    def test_python_to_python_transfer(self):
        """Test transfer between Python projects."""
        # Source: Data processing project
        source = NGramModel(n=3)
        for _ in range(10):
            source.train(["def process data return result for item in data"])
            source.train(["class DataProcessor import pandas numpy"])

        # Target: Web project
        target = NGramModel(n=3)
        target.train(["def handle request return response"])
        target.train(["class WebHandler import flask django"])

        adapted, metrics = transfer_knowledge(source, target)

        # Should have some overlap (Python constructs)
        self.assertGreater(metrics.vocabulary_overlap, 0.0)

    def test_empty_target_transfer(self):
        """Test transfer to empty target model."""
        source = NGramModel(n=3)
        for _ in range(5):
            source.train(["def function return value"])

        target = NGramModel(n=3)  # Empty

        adapted, metrics = transfer_knowledge(source, target)

        # Should add some vocabulary
        self.assertGreater(len(adapted.vocab), 0)

    def test_identical_projects_transfer(self):
        """Test transfer between identical models."""
        source = NGramModel(n=3)
        for _ in range(5):
            source.train(["def function return value for item in list"])

        # Target trained on same content
        target = NGramModel(n=3)
        for _ in range(5):
            target.train(["def function return value for item in list"])

        adapted, metrics = transfer_knowledge(source, target)

        # Should have high overlap
        self.assertGreater(metrics.vocabulary_overlap, 0.5)


if __name__ == '__main__':
    unittest.main()
