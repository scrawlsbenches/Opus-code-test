"""Tests for the gaps module."""

import unittest
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.gaps import analyze_knowledge_gaps, detect_anomalies


class TestGaps(unittest.TestCase):
    """Test the gaps module."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with sample data including an outlier."""
        cls.processor = CorticalTextProcessor()
        # Create cluster of related documents
        for i in range(3):
            cls.processor.process_document(f"tech_{i}", """
                Machine learning neural networks deep learning.
                Training models data processing algorithms.
                Pattern recognition artificial intelligence.
            """)
        # Add outlier document with different topic
        cls.processor.process_document("outlier", """
            Medieval falconry birds hunting prey.
            Falcons hawks eagles training techniques.
            Ancient hunting traditions wildlife.
        """)
        cls.processor.compute_all(verbose=False)

    def test_analyze_knowledge_gaps_structure(self):
        """Test that gap analysis returns expected structure."""
        gaps = analyze_knowledge_gaps(
            self.processor.layers,
            self.processor.documents
        )

        # Check all expected keys are present
        self.assertIn('isolated_documents', gaps)
        self.assertIn('weak_topics', gaps)
        self.assertIn('bridge_opportunities', gaps)
        self.assertIn('connector_terms', gaps)
        self.assertIn('coverage_score', gaps)
        self.assertIn('connectivity_score', gaps)
        self.assertIn('summary', gaps)

    def test_analyze_knowledge_gaps_summary(self):
        """Test that summary contains expected fields."""
        gaps = analyze_knowledge_gaps(
            self.processor.layers,
            self.processor.documents
        )

        summary = gaps['summary']
        self.assertIn('total_documents', summary)
        self.assertIn('isolated_count', summary)
        self.assertIn('well_connected_count', summary)
        self.assertIn('weak_topic_count', summary)

        self.assertEqual(summary['total_documents'], 4)

    def test_analyze_knowledge_gaps_isolated_documents(self):
        """Test isolated documents detection."""
        gaps = analyze_knowledge_gaps(
            self.processor.layers,
            self.processor.documents
        )

        isolated = gaps['isolated_documents']
        self.assertIsInstance(isolated, list)

        # Each isolated doc should have expected fields
        for doc in isolated:
            self.assertIn('doc_id', doc)
            self.assertIn('avg_similarity', doc)
            self.assertIn('max_similarity', doc)

    def test_analyze_knowledge_gaps_weak_topics(self):
        """Test weak topics detection."""
        gaps = analyze_knowledge_gaps(
            self.processor.layers,
            self.processor.documents
        )

        weak_topics = gaps['weak_topics']
        self.assertIsInstance(weak_topics, list)

        for topic in weak_topics:
            self.assertIn('term', topic)
            self.assertIn('tfidf', topic)
            self.assertIn('doc_count', topic)
            self.assertIn('documents', topic)

    def test_analyze_knowledge_gaps_coverage_score(self):
        """Test coverage score is valid."""
        gaps = analyze_knowledge_gaps(
            self.processor.layers,
            self.processor.documents
        )

        self.assertIsInstance(gaps['coverage_score'], float)
        self.assertGreaterEqual(gaps['coverage_score'], 0.0)
        self.assertLessEqual(gaps['coverage_score'], 1.0)

    def test_detect_anomalies_structure(self):
        """Test anomaly detection returns expected structure."""
        anomalies = detect_anomalies(
            self.processor.layers,
            self.processor.documents,
            threshold=0.3
        )

        self.assertIsInstance(anomalies, list)

        for anomaly in anomalies:
            self.assertIn('doc_id', anomaly)
            self.assertIn('avg_similarity', anomaly)
            self.assertIn('max_similarity', anomaly)
            self.assertIn('connections', anomaly)
            self.assertIn('reasons', anomaly)
            self.assertIn('distinctive_terms', anomaly)

    def test_detect_anomalies_reasons(self):
        """Test that anomalies have reasons."""
        anomalies = detect_anomalies(
            self.processor.layers,
            self.processor.documents,
            threshold=0.3
        )

        for anomaly in anomalies:
            self.assertIsInstance(anomaly['reasons'], list)
            # Each anomaly should have at least one reason
            self.assertGreater(len(anomaly['reasons']), 0)

    def test_detect_anomalies_sorted(self):
        """Test that anomalies are sorted by similarity (ascending)."""
        anomalies = detect_anomalies(
            self.processor.layers,
            self.processor.documents,
            threshold=0.5
        )

        if len(anomalies) > 1:
            similarities = [a['avg_similarity'] for a in anomalies]
            self.assertEqual(similarities, sorted(similarities))

    def test_detect_anomalies_threshold(self):
        """Test that threshold affects anomaly detection."""
        anomalies_low = detect_anomalies(
            self.processor.layers,
            self.processor.documents,
            threshold=0.1
        )

        anomalies_high = detect_anomalies(
            self.processor.layers,
            self.processor.documents,
            threshold=0.5
        )

        # Higher threshold should find more or equal anomalies
        self.assertGreaterEqual(len(anomalies_high), len(anomalies_low))


class TestGapsEmptyCorpus(unittest.TestCase):
    """Test gaps module with empty or minimal corpus."""

    def test_empty_corpus_gaps(self):
        """Test gap analysis on empty processor."""
        processor = CorticalTextProcessor()
        gaps = analyze_knowledge_gaps(
            processor.layers,
            processor.documents
        )

        self.assertEqual(gaps['summary']['total_documents'], 0)
        self.assertEqual(gaps['isolated_documents'], [])
        self.assertEqual(gaps['weak_topics'], [])

    def test_single_document_gaps(self):
        """Test gap analysis with single document."""
        processor = CorticalTextProcessor()
        processor.process_document("only_doc", "Single document content here.")
        processor.compute_all(verbose=False)

        gaps = analyze_knowledge_gaps(
            processor.layers,
            processor.documents
        )

        self.assertEqual(gaps['summary']['total_documents'], 1)

    def test_single_document_anomalies(self):
        """Test anomaly detection with single document."""
        processor = CorticalTextProcessor()
        processor.process_document("only_doc", "Single document content here.")
        processor.compute_all(verbose=False)

        anomalies = detect_anomalies(
            processor.layers,
            processor.documents,
            threshold=0.3
        )

        # Single doc can't have similarity to others
        self.assertIsInstance(anomalies, list)


class TestGapsBridgeOpportunities(unittest.TestCase):
    """Test bridge opportunity detection."""

    def test_bridge_opportunities_format(self):
        """Test bridge opportunities have correct format."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep")
        processor.process_document("doc2", "machine learning algorithms")
        processor.process_document("doc3", "database systems storage")
        processor.compute_all(verbose=False)

        gaps = analyze_knowledge_gaps(
            processor.layers,
            processor.documents
        )

        bridges = gaps['bridge_opportunities']
        self.assertIsInstance(bridges, list)

        for bridge in bridges:
            self.assertIn('doc1', bridge)
            self.assertIn('doc2', bridge)
            self.assertIn('similarity', bridge)
            self.assertIn('shared_terms', bridge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
