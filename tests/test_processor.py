"""Tests for the CorticalTextProcessor class."""

import unittest
import tempfile
import os
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer


class TestProcessorBasic(unittest.TestCase):
    """Test basic processor functionality."""
    
    def setUp(self):
        self.processor = CorticalTextProcessor()
    
    def test_process_document(self):
        """Test document processing."""
        stats = self.processor.process_document("doc1", "Neural networks process information.")
        self.assertGreater(stats['tokens'], 0)
        self.assertIn("doc1", self.processor.documents)
    
    def test_multiple_documents(self):
        """Test processing multiple documents."""
        self.processor.process_document("doc1", "Neural networks learn.")
        self.processor.process_document("doc2", "Deep learning models.")
        self.assertEqual(len(self.processor.documents), 2)
    
    def test_token_layer_populated(self):
        """Test that token layer is populated."""
        self.processor.process_document("doc1", "Neural networks process information.")
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        self.assertGreater(layer0.column_count(), 0)
    
    def test_lateral_connections(self):
        """Test that lateral connections are formed."""
        self.processor.process_document("doc1", "Neural networks process information.")
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        
        neural = layer0.get_minicolumn("neural")
        self.assertIsNotNone(neural)
        self.assertGreater(neural.connection_count(), 0)


class TestProcessorComputation(unittest.TestCase):
    """Test processor computation methods."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", """
            Neural networks process information through layers.
            Deep learning enables pattern recognition.
        """)
        cls.processor.process_document("doc2", """
            Machine learning models learn from data.
            Training neural networks requires optimization.
        """)
    
    def test_propagate_activation(self):
        """Test activation propagation."""
        self.processor.propagate_activation(iterations=3, verbose=False)
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        
        # Check some columns have activation
        activations = [col.activation for col in layer0]
        self.assertTrue(any(a > 0 for a in activations))
    
    def test_compute_importance(self):
        """Test PageRank computation."""
        self.processor.propagate_activation(iterations=3, verbose=False)
        self.processor.compute_importance(verbose=False)
        
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        pageranks = [col.pagerank for col in layer0]
        self.assertTrue(all(p > 0 for p in pageranks))
    
    def test_compute_tfidf(self):
        """Test TF-IDF computation."""
        # Create fresh processor for this test
        # Use 3 docs where 'neural' only appears in 2, so IDF > 0
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information.")
        processor.process_document("doc2", "Machine learning neural models.")
        processor.process_document("doc3", "Database systems store data efficiently.")
        processor.compute_tfidf(verbose=False)
        
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        neural = layer0.get_minicolumn("neural")
        self.assertIsNotNone(neural)
        # Now IDF = log(3/2) > 0
        self.assertGreater(neural.tfidf, 0)
    
    def test_compute_all(self):
        """Test compute_all runs without error."""
        processor = CorticalTextProcessor()
        processor.process_document("test", "Test document content.")
        processor.compute_all(verbose=False)


class TestProcessorQuery(unittest.TestCase):
    """Test processor query functionality."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("neural_doc", """
            Neural networks process information through multiple layers.
            Deep learning enables complex pattern recognition.
            Backpropagation trains neural network weights.
        """)
        cls.processor.process_document("ml_doc", """
            Machine learning algorithms learn from data.
            Supervised learning uses labeled examples.
            Model training optimizes parameters.
        """)
        cls.processor.compute_all(verbose=False)
    
    def test_expand_query(self):
        """Test query expansion."""
        expanded = self.processor.expand_query("neural", max_expansions=5)
        self.assertIn("neural", expanded)
        self.assertGreater(len(expanded), 1)
    
    def test_find_documents(self):
        """Test document finding."""
        results = self.processor.find_documents_for_query("neural networks", top_n=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "neural_doc")
    
    def test_query_expanded(self):
        """Test expanded query."""
        results = self.processor.query_expanded("learning", top_n=5)
        self.assertIsInstance(results, list)


class TestProcessorPersistence(unittest.TestCase):
    """Test processor save/load functionality."""
    
    def test_save_and_load(self):
        """Test saving and loading processor."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document content.")
        processor.compute_all(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            processor.save(filepath, verbose=False)
            
            loaded = CorticalTextProcessor.load(filepath, verbose=False)
            self.assertEqual(len(loaded.documents), 1)
            self.assertIn("doc1", loaded.documents)


class TestProcessorGaps(unittest.TestCase):
    """Test gap detection functionality."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        for i in range(3):
            cls.processor.process_document(f"tech_{i}", """
                Machine learning neural networks deep learning.
                Training models data processing algorithms.
            """)
        cls.processor.process_document("outlier", """
            Medieval falconry birds hunting prey.
            Falcons hawks eagles training.
        """)
        cls.processor.compute_all(verbose=False)
    
    def test_analyze_knowledge_gaps(self):
        """Test gap analysis returns expected structure."""
        gaps = self.processor.analyze_knowledge_gaps()
        self.assertIn('isolated_documents', gaps)
        self.assertIn('weak_topics', gaps)
        self.assertIn('coverage_score', gaps)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        anomalies = self.processor.detect_anomalies(threshold=0.1)
        self.assertIsInstance(anomalies, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
