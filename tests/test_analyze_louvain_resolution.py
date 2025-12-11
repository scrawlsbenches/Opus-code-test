"""
Tests for scripts/analyze_louvain_resolution.py - Louvain resolution analysis utilities.
"""

import unittest
import sys
from pathlib import Path

# Add parent and scripts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer
from analyze_louvain_resolution import (
    compute_modularity,
    compute_cluster_balance,
    evaluate_semantic_coherence,
    load_corpus
)


class TestComputeModularity(unittest.TestCase):
    """Tests for modularity computation."""

    def test_modularity_empty_processor(self):
        """Test modularity returns 0 for empty processor."""
        processor = CorticalTextProcessor()
        result = compute_modularity(processor)
        self.assertEqual(result, 0.0)

    def test_modularity_no_clusters(self):
        """Test modularity with documents but no clusters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hello world")
        processor.compute_all(build_concepts=False, verbose=False)
        result = compute_modularity(processor)
        self.assertEqual(result, 0.0)

    def test_modularity_with_clusters(self):
        """Test modularity with actual clusters."""
        processor = CorticalTextProcessor()
        # Add documents from different topics
        processor.process_document("ml", "Neural networks deep learning training")
        processor.process_document("cooking", "Bread baking flour yeast oven")
        processor.compute_all(verbose=False)

        result = compute_modularity(processor)
        # Modularity should be between -1 and 1
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_modularity_returns_float(self):
        """Test that modularity returns a float."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document content")
        processor.compute_all(verbose=False)

        result = compute_modularity(processor)
        self.assertIsInstance(result, float)


class TestComputeClusterBalance(unittest.TestCase):
    """Tests for Gini coefficient calculation."""

    def test_balance_empty_list(self):
        """Test balance with empty list returns 1.0."""
        result = compute_cluster_balance([])
        self.assertEqual(result, 1.0)

    def test_balance_single_cluster(self):
        """Test balance with single cluster returns 1.0."""
        result = compute_cluster_balance([100])
        self.assertEqual(result, 1.0)

    def test_balance_perfectly_balanced(self):
        """Test balance with perfectly equal clusters."""
        # Four clusters of equal size should have low Gini
        result = compute_cluster_balance([25, 25, 25, 25])
        self.assertLess(result, 0.1)  # Should be close to 0

    def test_balance_highly_skewed(self):
        """Test balance with one dominant cluster."""
        # One big cluster and many small ones = high Gini
        result = compute_cluster_balance([1000, 1, 1, 1, 1])
        self.assertGreater(result, 0.7)  # Should be high

    def test_balance_moderate_skew(self):
        """Test balance with moderate distribution."""
        result = compute_cluster_balance([100, 50, 25, 15, 10])
        # Should be somewhere in the middle
        self.assertGreater(result, 0.2)
        self.assertLess(result, 0.8)

    def test_balance_range(self):
        """Test that balance is always between 0 and 1."""
        test_cases = [
            [1],
            [1, 1],
            [100, 1],
            [10, 20, 30, 40],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
        ]
        for sizes in test_cases:
            result = compute_cluster_balance(sizes)
            self.assertGreaterEqual(result, 0.0, f"Failed for {sizes}")
            self.assertLessEqual(result, 1.0, f"Failed for {sizes}")

    def test_balance_zero_total(self):
        """Test balance with all-zero sizes."""
        result = compute_cluster_balance([0, 0, 0])
        self.assertEqual(result, 1.0)


class TestEvaluateSemanticCoherence(unittest.TestCase):
    """Tests for semantic coherence evaluation."""

    def test_coherence_empty_processor(self):
        """Test coherence with empty processor."""
        processor = CorticalTextProcessor()
        result = evaluate_semantic_coherence(processor)
        self.assertEqual(result, [])

    def test_coherence_no_clusters(self):
        """Test coherence with no clusters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hello world")
        processor.compute_all(build_concepts=False, verbose=False)

        result = evaluate_semantic_coherence(processor)
        self.assertEqual(result, [])

    def test_coherence_returns_list(self):
        """Test that coherence returns a list."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks deep learning")
        processor.process_document("doc2", "Cooking baking bread flour")
        processor.compute_all(verbose=False)

        result = evaluate_semantic_coherence(processor, top_n=3)
        self.assertIsInstance(result, list)

    def test_coherence_structure(self):
        """Test that coherence results have correct structure."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks deep learning training models")
        processor.process_document("doc2", "Cooking baking bread flour yeast oven")
        processor.compute_all(verbose=False)

        results = evaluate_semantic_coherence(processor, top_n=2)

        for entry in results:
            self.assertIn('cluster_id', entry)
            self.assertIn('size', entry)
            self.assertIn('coherence', entry)
            self.assertIn('sample_terms', entry)
            self.assertIsInstance(entry['coherence'], float)
            self.assertIsInstance(entry['sample_terms'], list)

    def test_coherence_values_bounded(self):
        """Test that coherence values are between 0 and 1."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks deep learning training models")
        processor.process_document("doc2", "Cooking baking bread flour yeast oven")
        processor.compute_all(verbose=False)

        results = evaluate_semantic_coherence(processor, top_n=5)

        for entry in results:
            self.assertGreaterEqual(entry['coherence'], 0.0)
            self.assertLessEqual(entry['coherence'], 1.0)


class TestLoadCorpus(unittest.TestCase):
    """Tests for corpus loading."""

    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        processor = CorticalTextProcessor()
        result = load_corpus(processor, "nonexistent_dir_12345")
        self.assertEqual(result, 0)

    def test_load_samples_directory(self):
        """Test loading from samples directory."""
        processor = CorticalTextProcessor()
        result = load_corpus(processor, "samples")
        # Should load some documents if samples dir exists
        self.assertGreater(result, 0)

    def test_load_populates_processor(self):
        """Test that loading populates the processor."""
        processor = CorticalTextProcessor()
        num_loaded = load_corpus(processor, "samples")

        if num_loaded > 0:
            # Processor should have documents
            layer3 = processor.layers[CorticalLayer.DOCUMENTS]
            self.assertGreater(layer3.column_count(), 0)


class TestGiniCoefficientMathematics(unittest.TestCase):
    """Tests for mathematical correctness of Gini coefficient."""

    def test_gini_two_equal_values(self):
        """Two equal values should give Gini = 0."""
        result = compute_cluster_balance([50, 50])
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_gini_extreme_inequality(self):
        """Extreme inequality with many small clusters and one large."""
        # With many elements, extreme inequality approaches Gini=1
        # 100 clusters: one has 99%, others have 1% each
        sizes = [9900] + [1] * 99
        result = compute_cluster_balance(sizes)
        self.assertGreater(result, 0.9)

    def test_gini_two_element_inequality(self):
        """Two-element extreme inequality gives Gini ~0.5."""
        # Mathematical property: with only 2 elements, max Gini is ~0.5
        result = compute_cluster_balance([1000000, 1])
        self.assertGreater(result, 0.4)
        self.assertLess(result, 0.6)

    def test_gini_ascending_order(self):
        """Gini should work regardless of input order."""
        ascending = compute_cluster_balance([10, 20, 30, 40])
        descending = compute_cluster_balance([40, 30, 20, 10])
        random_order = compute_cluster_balance([30, 10, 40, 20])

        # All should give same result
        self.assertAlmostEqual(ascending, descending, places=5)
        self.assertAlmostEqual(ascending, random_order, places=5)


if __name__ == '__main__':
    unittest.main()
