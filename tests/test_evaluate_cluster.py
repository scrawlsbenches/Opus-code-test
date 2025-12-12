"""
Tests for scripts/evaluate_cluster.py - Cluster coverage evaluation utilities.
"""

import unittest
import sys
from pathlib import Path

# Add parent and scripts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer
from evaluate_cluster import (
    find_documents_by_keywords,
    compute_document_similarity,
    compute_cluster_metrics,
    find_expansion_suggestions,
    assess_coverage,
)


class TestFindDocumentsByKeywords(unittest.TestCase):
    """Tests for keyword-based document finding."""

    def setUp(self):
        """Create a processor with test documents."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("ml1", "Neural networks deep learning training models")
        self.processor.process_document("ml2", "Machine learning algorithms neural data")
        self.processor.process_document("cook1", "Bread baking flour yeast oven temperature")
        self.processor.process_document("cook2", "Italian pasta cooking tomato sauce")
        self.processor.compute_all(verbose=False)

    def test_single_keyword_match(self):
        """Test finding documents with a single keyword."""
        docs = find_documents_by_keywords(self.processor, ["neural"])
        self.assertIn("ml1", docs)
        self.assertIn("ml2", docs)
        self.assertNotIn("cook1", docs)

    def test_multiple_keywords_any(self):
        """Test finding documents with any of multiple keywords."""
        docs = find_documents_by_keywords(self.processor, ["neural", "pasta"], min_keywords=1)
        self.assertIn("ml1", docs)
        self.assertIn("ml2", docs)
        self.assertIn("cook2", docs)

    def test_multiple_keywords_all(self):
        """Test finding documents with all keywords."""
        docs = find_documents_by_keywords(self.processor, ["neural", "learning"], min_keywords=2)
        # ml1 has both "neural" and "learning"
        self.assertIn("ml1", docs)

    def test_no_matches(self):
        """Test with keywords that don't match any documents."""
        docs = find_documents_by_keywords(self.processor, ["quantum", "physics"])
        self.assertEqual(docs, [])


class TestComputeDocumentSimilarity(unittest.TestCase):
    """Tests for document similarity computation."""

    def setUp(self):
        """Create a processor with test documents."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Neural networks deep learning models")
        self.processor.process_document("doc2", "Neural networks machine learning algorithms")
        self.processor.process_document("doc3", "Bread baking flour yeast recipes")
        self.processor.compute_all(verbose=False)

    def test_similar_documents(self):
        """Test similarity between related documents."""
        sim = compute_document_similarity(self.processor, "doc1", "doc2")
        # Both are about neural networks/ML, should have positive similarity
        self.assertGreater(sim, 0.0)

    def test_dissimilar_documents(self):
        """Test similarity between unrelated documents."""
        sim_ml_cook = compute_document_similarity(self.processor, "doc1", "doc3")
        sim_ml_ml = compute_document_similarity(self.processor, "doc1", "doc2")
        # ML docs should be more similar to each other than to cooking
        self.assertGreater(sim_ml_ml, sim_ml_cook)

    def test_self_similarity(self):
        """Test similarity of a document with itself."""
        sim = compute_document_similarity(self.processor, "doc1", "doc1")
        # Self-similarity should be 1.0 (or close to it)
        self.assertGreaterEqual(sim, 0.9)

    def test_nonexistent_document(self):
        """Test similarity with non-existent document."""
        sim = compute_document_similarity(self.processor, "doc1", "nonexistent")
        self.assertEqual(sim, 0.0)


class TestComputeClusterMetrics(unittest.TestCase):
    """Tests for cluster metrics computation."""

    def setUp(self):
        """Create a processor with diverse documents."""
        self.processor = CorticalTextProcessor()

        # ML cluster
        self.processor.process_document("ml1", "Neural networks deep learning training models backpropagation")
        self.processor.process_document("ml2", "Machine learning algorithms neural data classification")
        self.processor.process_document("ml3", "Deep learning convolutional networks image recognition")

        # Cooking cluster
        self.processor.process_document("cook1", "Bread baking flour yeast oven temperature recipes")
        self.processor.process_document("cook2", "Italian pasta cooking tomato sauce ingredients")
        self.processor.process_document("cook3", "French cuisine cooking techniques sauces")

        self.processor.compute_all(verbose=False)

    def test_metrics_returns_dict(self):
        """Test that metrics returns expected dictionary structure."""
        metrics = compute_cluster_metrics(self.processor, ["ml1", "ml2", "ml3"])

        self.assertIn("cohesion", metrics)
        self.assertIn("separation", metrics)
        self.assertIn("concept_count", metrics)
        self.assertIn("term_count", metrics)
        self.assertIn("diversity", metrics)
        self.assertIn("hub_document", metrics)
        self.assertIn("key_terms", metrics)

    def test_cohesion_range(self):
        """Test that cohesion is in valid range."""
        metrics = compute_cluster_metrics(self.processor, ["ml1", "ml2", "ml3"])
        self.assertGreaterEqual(metrics["cohesion"], 0.0)
        self.assertLessEqual(metrics["cohesion"], 1.0)

    def test_separation_range(self):
        """Test that separation is in valid range."""
        metrics = compute_cluster_metrics(self.processor, ["ml1", "ml2", "ml3"])
        self.assertGreaterEqual(metrics["separation"], 0.0)
        self.assertLessEqual(metrics["separation"], 1.0)

    def test_diversity_range(self):
        """Test that diversity is in valid range."""
        metrics = compute_cluster_metrics(self.processor, ["ml1", "ml2", "ml3"])
        self.assertGreaterEqual(metrics["diversity"], 0.0)
        self.assertLessEqual(metrics["diversity"], 1.0)

    def test_hub_document_in_cluster(self):
        """Test that hub document is one of the cluster documents."""
        cluster = ["ml1", "ml2", "ml3"]
        metrics = compute_cluster_metrics(self.processor, cluster)
        self.assertIn(metrics["hub_document"], cluster)

    def test_term_count_positive(self):
        """Test that term count is positive for non-empty cluster."""
        metrics = compute_cluster_metrics(self.processor, ["ml1", "ml2"])
        self.assertGreater(metrics["term_count"], 0)

    def test_single_document_cluster(self):
        """Test metrics for single-document cluster."""
        metrics = compute_cluster_metrics(self.processor, ["ml1"])
        # Single doc has no internal pairs, cohesion should be 0
        self.assertEqual(metrics["cohesion"], 0.0)
        # Should still have valid hub
        self.assertEqual(metrics["hub_document"], "ml1")


class TestAssessCoverage(unittest.TestCase):
    """Tests for coverage assessment logic."""

    def test_strong_coverage(self):
        """Test that high metrics yield STRONG assessment."""
        metrics = {
            "cohesion": 0.4,
            "separation": 0.7,
            "concept_count": 10,
        }
        label, _ = assess_coverage(metrics, num_docs=6)
        self.assertEqual(label, "STRONG")

    def test_adequate_coverage(self):
        """Test that moderate metrics yield ADEQUATE assessment."""
        metrics = {
            "cohesion": 0.2,
            "separation": 0.5,
            "concept_count": 5,
        }
        label, _ = assess_coverage(metrics, num_docs=4)
        self.assertEqual(label, "ADEQUATE")

    def test_needs_expansion(self):
        """Test that low metrics yield NEEDS EXPANSION assessment."""
        metrics = {
            "cohesion": 0.05,
            "separation": 0.3,
            "concept_count": 1,
        }
        label, _ = assess_coverage(metrics, num_docs=2)
        self.assertEqual(label, "NEEDS EXPANSION")

    def test_explanation_provided(self):
        """Test that assessment includes explanation."""
        metrics = {
            "cohesion": 0.3,
            "separation": 0.6,
            "concept_count": 5,
        }
        label, explanation = assess_coverage(metrics, num_docs=5)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)


class TestFindExpansionSuggestions(unittest.TestCase):
    """Tests for expansion suggestion generation."""

    def setUp(self):
        """Create a processor with test documents."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("ml1", "Neural networks deep learning training")
        self.processor.process_document("ml2", "Machine learning algorithms data")
        self.processor.process_document("other1", "Cooking recipes baking bread")
        self.processor.process_document("other2", "Legal contract law agreements")
        self.processor.compute_all(verbose=False)

    def test_suggestions_returns_list(self):
        """Test that suggestions returns a list of tuples."""
        cluster_docs = ["ml1", "ml2"]
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        cluster_tokens = set()
        for doc_id in cluster_docs:
            for col in layer0.minicolumns.values():
                if doc_id in col.document_ids:
                    cluster_tokens.add(col.content)

        suggestions = find_expansion_suggestions(
            self.processor, cluster_tokens, cluster_docs, max_suggestions=3
        )
        self.assertIsInstance(suggestions, list)

    def test_suggestions_format(self):
        """Test that each suggestion is a (term, reason) tuple."""
        cluster_docs = ["ml1", "ml2"]
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        cluster_tokens = set()
        for doc_id in cluster_docs:
            for col in layer0.minicolumns.values():
                if doc_id in col.document_ids:
                    cluster_tokens.add(col.content)

        suggestions = find_expansion_suggestions(
            self.processor, cluster_tokens, cluster_docs, max_suggestions=3
        )

        for suggestion in suggestions:
            self.assertEqual(len(suggestion), 2)
            self.assertIsInstance(suggestion[0], str)  # term
            self.assertIsInstance(suggestion[1], str)  # reason

    def test_max_suggestions_respected(self):
        """Test that max_suggestions limit is respected."""
        cluster_docs = ["ml1"]
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        cluster_tokens = set()
        for doc_id in cluster_docs:
            for col in layer0.minicolumns.values():
                if doc_id in col.document_ids:
                    cluster_tokens.add(col.content)

        suggestions = find_expansion_suggestions(
            self.processor, cluster_tokens, cluster_docs, max_suggestions=2
        )
        self.assertLessEqual(len(suggestions), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests using the full workflow."""

    @classmethod
    def setUpClass(cls):
        """Load showcase corpus for integration tests."""
        cls.processor = CorticalTextProcessor()
        samples_dir = Path(__file__).parent.parent / 'samples'

        if not samples_dir.exists():
            cls.skip_tests = True
            return

        txt_files = list(samples_dir.glob('*.txt'))[:20]  # Use subset for speed
        if len(txt_files) < 5:
            cls.skip_tests = True
            return

        cls.skip_tests = False
        for f in txt_files:
            cls.processor.process_document(f.stem, f.read_text())

        cls.processor.compute_all(verbose=False)

    def setUp(self):
        if getattr(self.__class__, 'skip_tests', False):
            self.skipTest("Sample corpus not available")

    def test_keyword_search_finds_documents(self):
        """Test that keyword search finds relevant documents."""
        docs = find_documents_by_keywords(self.processor, ["neural", "network"])
        # Should find at least one ML-related document
        self.assertGreater(len(docs), 0)

    def test_full_metrics_workflow(self):
        """Test the full metrics computation workflow."""
        docs = find_documents_by_keywords(self.processor, ["learning"], min_keywords=1)
        if len(docs) < 2:
            self.skipTest("Not enough matching documents")

        metrics = compute_cluster_metrics(self.processor, docs[:5])

        # All metrics should be valid
        self.assertGreaterEqual(metrics["cohesion"], 0.0)
        self.assertGreaterEqual(metrics["separation"], 0.0)
        self.assertGreater(metrics["term_count"], 0)


if __name__ == "__main__":
    unittest.main()
