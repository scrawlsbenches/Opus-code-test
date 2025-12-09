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


class TestProcessorMetadata(unittest.TestCase):
    """Test document metadata functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor()

    def test_process_document_with_metadata(self):
        """Test processing document with metadata."""
        metadata = {"source": "https://example.com", "author": "Test Author"}
        self.processor.process_document("doc1", "Test content.", metadata=metadata)
        retrieved = self.processor.get_document_metadata("doc1")
        self.assertEqual(retrieved["source"], "https://example.com")
        self.assertEqual(retrieved["author"], "Test Author")

    def test_set_document_metadata(self):
        """Test setting metadata after processing."""
        self.processor.process_document("doc1", "Test content.")
        self.processor.set_document_metadata("doc1", source="https://test.com", timestamp="2025-12-09")
        metadata = self.processor.get_document_metadata("doc1")
        self.assertEqual(metadata["source"], "https://test.com")
        self.assertEqual(metadata["timestamp"], "2025-12-09")

    def test_update_document_metadata(self):
        """Test updating existing metadata."""
        self.processor.process_document("doc1", "Test content.", metadata={"author": "Original"})
        self.processor.set_document_metadata("doc1", author="Updated", category="AI")
        metadata = self.processor.get_document_metadata("doc1")
        self.assertEqual(metadata["author"], "Updated")
        self.assertEqual(metadata["category"], "AI")

    def test_get_document_metadata_missing(self):
        """Test getting metadata for nonexistent document."""
        metadata = self.processor.get_document_metadata("nonexistent")
        self.assertEqual(metadata, {})

    def test_get_all_document_metadata(self):
        """Test getting all document metadata."""
        self.processor.process_document("doc1", "Content 1", metadata={"type": "article"})
        self.processor.process_document("doc2", "Content 2", metadata={"type": "paper"})
        all_metadata = self.processor.get_all_document_metadata()
        self.assertEqual(len(all_metadata), 2)
        self.assertEqual(all_metadata["doc1"]["type"], "article")
        self.assertEqual(all_metadata["doc2"]["type"], "paper")

    def test_metadata_not_modified_externally(self):
        """Test that get_all_document_metadata returns a copy."""
        self.processor.process_document("doc1", "Content", metadata={"key": "value"})
        all_metadata = self.processor.get_all_document_metadata()
        all_metadata["doc1"]["key"] = "modified"
        # Original should be unchanged
        original = self.processor.get_document_metadata("doc1")
        self.assertEqual(original["key"], "value")


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

    def test_save_and_load_with_metadata(self):
        """Test that document metadata is preserved through save/load."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "Test document content.",
            metadata={"source": "https://example.com", "author": "Test Author"}
        )
        processor.set_document_metadata("doc1", category="test")
        processor.compute_all(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            processor.save(filepath, verbose=False)

            loaded = CorticalTextProcessor.load(filepath, verbose=False)
            metadata = loaded.get_document_metadata("doc1")
            self.assertEqual(metadata["source"], "https://example.com")
            self.assertEqual(metadata["author"], "Test Author")
            self.assertEqual(metadata["category"], "test")


class TestProcessorPassageRetrieval(unittest.TestCase):
    """Test chunk-level passage retrieval for RAG systems."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        # Create documents with distinct content for testing passage retrieval
        cls.processor.process_document("neural_doc", """
            Neural networks are computational models inspired by biological neurons.
            They process information through interconnected layers of nodes.
            Deep learning uses many layers to learn hierarchical representations.
            Backpropagation is the key algorithm for training neural networks.
            Convolutional neural networks excel at image recognition tasks.
        """)
        cls.processor.process_document("ml_doc", """
            Machine learning algorithms learn patterns from data automatically.
            Supervised learning requires labeled training examples.
            Unsupervised learning discovers hidden structure in unlabeled data.
            Reinforcement learning trains agents through rewards and penalties.
            Model evaluation uses metrics like accuracy and precision.
        """)
        cls.processor.process_document("data_doc", """
            Data preprocessing is essential for machine learning pipelines.
            Feature engineering creates meaningful input representations.
            Data normalization scales features to similar ranges.
            Missing value imputation handles incomplete datasets.
            Cross-validation ensures robust model performance estimates.
        """)
        cls.processor.compute_all(verbose=False)

    def test_find_passages_returns_list(self):
        """Test that find_passages_for_query returns a list."""
        results = self.processor.find_passages_for_query("neural networks")
        self.assertIsInstance(results, list)

    def test_find_passages_returns_tuples(self):
        """Test that results are tuples with correct structure."""
        results = self.processor.find_passages_for_query("neural networks", top_n=1)
        self.assertGreater(len(results), 0)
        passage, doc_id, start, end, score = results[0]
        self.assertIsInstance(passage, str)
        self.assertIsInstance(doc_id, str)
        self.assertIsInstance(start, int)
        self.assertIsInstance(end, int)
        self.assertIsInstance(score, float)

    def test_find_passages_contains_text(self):
        """Test that passages contain actual text."""
        results = self.processor.find_passages_for_query("neural", top_n=3)
        self.assertGreater(len(results), 0)
        passage, _, _, _, _ = results[0]
        self.assertGreater(len(passage), 0)

    def test_find_passages_position_valid(self):
        """Test that start/end positions are valid."""
        results = self.processor.find_passages_for_query("learning", top_n=3)
        for passage, doc_id, start, end, score in results:
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertEqual(len(passage), end - start)

    def test_find_passages_top_n_limit(self):
        """Test that top_n parameter limits results."""
        results = self.processor.find_passages_for_query("learning", top_n=2)
        self.assertLessEqual(len(results), 2)

    def test_find_passages_chunk_size(self):
        """Test that chunk_size parameter is respected."""
        results = self.processor.find_passages_for_query(
            "neural", top_n=5, chunk_size=100, overlap=20
        )
        for passage, _, _, _, _ in results:
            self.assertLessEqual(len(passage), 100)

    def test_find_passages_doc_filter(self):
        """Test that doc_filter restricts search."""
        results = self.processor.find_passages_for_query(
            "learning", top_n=10, doc_filter=["neural_doc"]
        )
        for _, doc_id, _, _, _ in results:
            self.assertEqual(doc_id, "neural_doc")

    def test_find_passages_scores_descending(self):
        """Test that results are sorted by score descending."""
        results = self.processor.find_passages_for_query("neural networks", top_n=5)
        if len(results) > 1:
            scores = [score for _, _, _, _, score in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_find_passages_no_expansion(self):
        """Test passage retrieval without query expansion."""
        results = self.processor.find_passages_for_query(
            "neural", top_n=3, use_expansion=False
        )
        self.assertIsInstance(results, list)

    def test_find_passages_empty_query(self):
        """Test handling of queries with no matching terms."""
        results = self.processor.find_passages_for_query("xyznonexistent123")
        self.assertEqual(len(results), 0)


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
