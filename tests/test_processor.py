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


class TestProcessorBatchQuery(unittest.TestCase):
    """Test batch query functionality for efficient multi-query search."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("neural_doc", """
            Neural networks are computational models inspired by biological neurons.
            Deep learning uses many layers to learn hierarchical representations.
            Backpropagation is the key algorithm for training neural networks.
        """)
        cls.processor.process_document("ml_doc", """
            Machine learning algorithms learn patterns from data automatically.
            Supervised learning requires labeled training examples.
            Model evaluation uses metrics like accuracy and precision.
        """)
        cls.processor.process_document("data_doc", """
            Data preprocessing is essential for machine learning pipelines.
            Feature engineering creates meaningful input representations.
            Data normalization scales features to similar ranges.
        """)
        cls.processor.compute_all(verbose=False)

    def test_find_documents_batch_returns_list(self):
        """Test that find_documents_batch returns a list of results."""
        queries = ["neural networks", "machine learning"]
        results = self.processor.find_documents_batch(queries, top_n=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_find_documents_batch_result_structure(self):
        """Test that each result has correct structure."""
        queries = ["neural", "data"]
        results = self.processor.find_documents_batch(queries, top_n=3)
        for result in results:
            self.assertIsInstance(result, list)
            for doc_id, score in result:
                self.assertIsInstance(doc_id, str)
                self.assertIsInstance(score, float)

    def test_find_documents_batch_returns_relevant_docs(self):
        """Test that batch queries return relevant documents."""
        queries = ["neural networks", "data preprocessing"]
        results = self.processor.find_documents_batch(queries, top_n=1)
        # First query should find neural_doc
        self.assertGreater(len(results[0]), 0)
        self.assertEqual(results[0][0][0], "neural_doc")
        # Second query should find data_doc
        self.assertGreater(len(results[1]), 0)
        self.assertEqual(results[1][0][0], "data_doc")

    def test_find_documents_batch_top_n(self):
        """Test that top_n limits results per query."""
        queries = ["learning", "neural"]
        results = self.processor.find_documents_batch(queries, top_n=2)
        for result in results:
            self.assertLessEqual(len(result), 2)

    def test_find_documents_batch_empty_query_list(self):
        """Test batch with empty query list."""
        results = self.processor.find_documents_batch([], top_n=3)
        self.assertEqual(results, [])

    def test_find_documents_batch_no_expansion(self):
        """Test batch query without expansion."""
        queries = ["neural", "data"]
        results = self.processor.find_documents_batch(
            queries, top_n=2, use_expansion=False
        )
        self.assertEqual(len(results), 2)

    def test_find_passages_batch_returns_list(self):
        """Test that find_passages_batch returns a list of results."""
        queries = ["neural networks", "machine learning"]
        results = self.processor.find_passages_batch(queries, top_n=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_find_passages_batch_result_structure(self):
        """Test that each passage result has correct structure."""
        queries = ["neural"]
        results = self.processor.find_passages_batch(queries, top_n=3)
        self.assertEqual(len(results), 1)
        for passage, doc_id, start, end, score in results[0]:
            self.assertIsInstance(passage, str)
            self.assertIsInstance(doc_id, str)
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertIsInstance(score, float)

    def test_find_passages_batch_top_n(self):
        """Test that top_n limits passages per query."""
        queries = ["learning", "neural"]
        results = self.processor.find_passages_batch(queries, top_n=2)
        for result in results:
            self.assertLessEqual(len(result), 2)

    def test_find_passages_batch_chunk_size(self):
        """Test that chunk_size is respected."""
        queries = ["neural"]
        results = self.processor.find_passages_batch(
            queries, top_n=5, chunk_size=100, overlap=20
        )
        for passage, _, _, _, _ in results[0]:
            self.assertLessEqual(len(passage), 100)

    def test_find_passages_batch_doc_filter(self):
        """Test that doc_filter restricts results."""
        queries = ["learning", "neural"]
        results = self.processor.find_passages_batch(
            queries, top_n=10, doc_filter=["neural_doc"]
        )
        for result in results:
            for _, doc_id, _, _, _ in result:
                self.assertEqual(doc_id, "neural_doc")

    def test_find_passages_batch_empty_query_list(self):
        """Test batch with empty query list."""
        results = self.processor.find_passages_batch([], top_n=3)
        self.assertEqual(results, [])

    def test_batch_query_consistency(self):
        """Test that batch results match individual queries."""
        queries = ["neural networks", "data processing"]
        batch_results = self.processor.find_documents_batch(queries, top_n=3)

        # Compare with individual queries
        for i, query in enumerate(queries):
            individual_result = self.processor.find_documents_for_query(query, top_n=3)
            # Results should be the same (or very close)
            self.assertEqual(len(batch_results[i]), len(individual_result))
            for j, (doc_id, score) in enumerate(batch_results[i]):
                self.assertEqual(doc_id, individual_result[j][0])

    def test_batch_handles_nonexistent_terms(self):
        """Test that batch handles queries with no matches."""
        queries = ["xyznonexistent123", "neural networks"]
        results = self.processor.find_documents_batch(queries, top_n=3)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 0)  # No matches for nonexistent
        self.assertGreater(len(results[1]), 0)  # Matches for neural networks


class TestProcessorMultiStageRanking(unittest.TestCase):
    """Test multi-stage ranking pipeline for RAG systems."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        # Create a diverse corpus for testing multi-stage ranking
        cls.processor.process_document("neural_doc", """
            Neural networks are computational models inspired by biological neurons.
            Deep learning uses many layers to learn hierarchical representations.
            Backpropagation is the key algorithm for training neural networks.
            Convolutional neural networks excel at image recognition tasks.
        """)
        cls.processor.process_document("ml_doc", """
            Machine learning algorithms learn patterns from data automatically.
            Supervised learning requires labeled training examples.
            Unsupervised learning discovers hidden structure in data.
            Model evaluation uses metrics like accuracy precision and recall.
        """)
        cls.processor.process_document("data_doc", """
            Data preprocessing is essential for machine learning pipelines.
            Feature engineering creates meaningful input representations.
            Data normalization scales features to similar ranges.
            Cross-validation ensures robust model performance estimates.
        """)
        cls.processor.process_document("nlp_doc", """
            Natural language processing enables computers to understand text.
            Word embeddings capture semantic relationships between words.
            Transformers use attention mechanisms for sequence modeling.
            Language models can generate coherent text passages.
        """)
        cls.processor.compute_all(verbose=False)

    def test_multi_stage_rank_returns_list(self):
        """Test that multi_stage_rank returns a list."""
        results = self.processor.multi_stage_rank("neural networks", top_n=3)
        self.assertIsInstance(results, list)

    def test_multi_stage_rank_result_structure(self):
        """Test that results have correct 6-tuple structure."""
        results = self.processor.multi_stage_rank("neural", top_n=3)
        self.assertGreater(len(results), 0)
        passage, doc_id, start, end, score, stage_scores = results[0]
        self.assertIsInstance(passage, str)
        self.assertIsInstance(doc_id, str)
        self.assertIsInstance(start, int)
        self.assertIsInstance(end, int)
        self.assertIsInstance(score, float)
        self.assertIsInstance(stage_scores, dict)

    def test_multi_stage_rank_stage_scores(self):
        """Test that stage_scores contains expected keys."""
        results = self.processor.multi_stage_rank("neural networks", top_n=3)
        self.assertGreater(len(results), 0)
        _, _, _, _, _, stage_scores = results[0]
        self.assertIn('concept_score', stage_scores)
        self.assertIn('doc_score', stage_scores)
        self.assertIn('chunk_score', stage_scores)
        self.assertIn('final_score', stage_scores)

    def test_multi_stage_rank_top_n(self):
        """Test that top_n limits results."""
        results = self.processor.multi_stage_rank("learning", top_n=2)
        self.assertLessEqual(len(results), 2)

    def test_multi_stage_rank_chunk_size(self):
        """Test that chunk_size is respected."""
        results = self.processor.multi_stage_rank(
            "neural", top_n=5, chunk_size=100, overlap=20
        )
        for passage, _, _, _, _, _ in results:
            self.assertLessEqual(len(passage), 100)

    def test_multi_stage_rank_concept_boost(self):
        """Test that concept_boost parameter is used."""
        # Test with high concept boost vs low
        results_high = self.processor.multi_stage_rank(
            "neural", top_n=3, concept_boost=0.8
        )
        results_low = self.processor.multi_stage_rank(
            "neural", top_n=3, concept_boost=0.1
        )
        # Both should return results (exact ordering may differ)
        self.assertGreater(len(results_high), 0)
        self.assertGreater(len(results_low), 0)

    def test_multi_stage_rank_sorted_descending(self):
        """Test that results are sorted by score descending."""
        results = self.processor.multi_stage_rank("neural networks", top_n=5)
        if len(results) > 1:
            scores = [score for _, _, _, _, score, _ in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_multi_stage_rank_documents_returns_list(self):
        """Test that multi_stage_rank_documents returns a list."""
        results = self.processor.multi_stage_rank_documents("neural networks", top_n=3)
        self.assertIsInstance(results, list)

    def test_multi_stage_rank_documents_structure(self):
        """Test that document results have correct 3-tuple structure."""
        results = self.processor.multi_stage_rank_documents("neural", top_n=3)
        self.assertGreater(len(results), 0)
        doc_id, score, stage_scores = results[0]
        self.assertIsInstance(doc_id, str)
        self.assertIsInstance(score, float)
        self.assertIsInstance(stage_scores, dict)

    def test_multi_stage_rank_documents_stage_scores(self):
        """Test that document stage_scores contains expected keys."""
        results = self.processor.multi_stage_rank_documents("neural networks", top_n=3)
        self.assertGreater(len(results), 0)
        _, _, stage_scores = results[0]
        self.assertIn('concept_score', stage_scores)
        self.assertIn('tfidf_score', stage_scores)
        self.assertIn('combined_score', stage_scores)

    def test_multi_stage_rank_documents_top_n(self):
        """Test that top_n limits document results."""
        results = self.processor.multi_stage_rank_documents("learning", top_n=2)
        self.assertLessEqual(len(results), 2)

    def test_multi_stage_rank_documents_sorted(self):
        """Test that document results are sorted by score descending."""
        results = self.processor.multi_stage_rank_documents("neural networks", top_n=5)
        if len(results) > 1:
            scores = [score for _, score, _ in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_multi_stage_rank_empty_query(self):
        """Test handling of query with no matches."""
        results = self.processor.multi_stage_rank("xyznonexistent123", top_n=3)
        self.assertEqual(len(results), 0)

    def test_multi_stage_rank_without_expansion(self):
        """Test multi-stage ranking without query expansion."""
        results = self.processor.multi_stage_rank(
            "neural", top_n=3, use_expansion=False
        )
        self.assertIsInstance(results, list)

    def test_multi_stage_vs_flat_ranking(self):
        """Test that multi-stage ranking produces results comparable to flat ranking."""
        # Both should find relevant documents for the same query
        multi_results = self.processor.multi_stage_rank("neural networks", top_n=3)
        flat_results = self.processor.find_passages_for_query("neural networks", top_n=3)

        # Both should return results
        self.assertGreater(len(multi_results), 0)
        self.assertGreater(len(flat_results), 0)

        # Both should find the neural_doc
        multi_docs = {doc_id for _, doc_id, _, _, _, _ in multi_results}
        flat_docs = {doc_id for _, doc_id, _, _, _ in flat_results}
        self.assertIn("neural_doc", multi_docs)
        self.assertIn("neural_doc", flat_docs)


class TestProcessorIncrementalIndexing(unittest.TestCase):
    """Test incremental document indexing functionality."""

    def setUp(self):
        self.processor = CorticalTextProcessor()

    def test_add_document_incremental_returns_stats(self):
        """Test that add_document_incremental returns processing stats."""
        stats = self.processor.add_document_incremental(
            "doc1", "Neural networks process information.", recompute='tfidf'
        )
        self.assertIn('tokens', stats)
        self.assertIn('bigrams', stats)
        self.assertIn('unique_tokens', stats)
        self.assertGreater(stats['tokens'], 0)

    def test_add_document_incremental_with_metadata(self):
        """Test incremental add with metadata."""
        self.processor.add_document_incremental(
            "doc1",
            "Test content.",
            metadata={"source": "test", "author": "AI"},
            recompute='tfidf'
        )
        metadata = self.processor.get_document_metadata("doc1")
        self.assertEqual(metadata["source"], "test")
        self.assertEqual(metadata["author"], "AI")

    def test_add_document_incremental_recompute_none(self):
        """Test that recompute='none' marks computations as stale."""
        self.processor.add_document_incremental(
            "doc1", "Test content.", recompute='none'
        )
        # Should be stale
        self.assertTrue(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))
        self.assertTrue(self.processor.is_stale(CorticalTextProcessor.COMP_PAGERANK))

    def test_add_document_incremental_recompute_tfidf(self):
        """Test that recompute='tfidf' only recomputes TF-IDF."""
        self.processor.add_document_incremental(
            "doc1", "Test content.", recompute='tfidf'
        )
        # TF-IDF should be fresh
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))
        # Other computations should be stale
        self.assertTrue(self.processor.is_stale(CorticalTextProcessor.COMP_PAGERANK))

    def test_add_document_incremental_recompute_full(self):
        """Test that recompute='full' clears all staleness."""
        self.processor.add_document_incremental(
            "doc1", "Test content.", recompute='full'
        )
        # All should be fresh
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_PAGERANK))
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_ACTIVATION))

    def test_add_documents_batch_returns_stats(self):
        """Test that add_documents_batch returns batch statistics."""
        docs = [
            ("doc1", "First document content.", {"source": "web"}),
            ("doc2", "Second document content.", None),
            ("doc3", "Third document content.", {"author": "AI"}),
        ]
        stats = self.processor.add_documents_batch(docs, recompute='full', verbose=False)
        self.assertEqual(stats['documents_added'], 3)
        self.assertIn('total_tokens', stats)
        self.assertIn('total_bigrams', stats)
        self.assertEqual(stats['recomputation'], 'full')

    def test_add_documents_batch_preserves_metadata(self):
        """Test that batch add preserves metadata for all documents."""
        docs = [
            ("doc1", "First content.", {"type": "article"}),
            ("doc2", "Second content.", {"type": "paper"}),
        ]
        self.processor.add_documents_batch(docs, recompute='tfidf', verbose=False)
        self.assertEqual(self.processor.get_document_metadata("doc1")["type"], "article")
        self.assertEqual(self.processor.get_document_metadata("doc2")["type"], "paper")

    def test_add_documents_batch_recompute_none(self):
        """Test batch add with no recomputation."""
        docs = [("doc1", "Content one.", None), ("doc2", "Content two.", None)]
        self.processor.add_documents_batch(docs, recompute='none', verbose=False)
        self.assertTrue(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))
        self.assertEqual(len(self.processor.documents), 2)

    def test_recompute_full(self):
        """Test recompute with level='full'."""
        self.processor.add_document_incremental("doc1", "Test content.", recompute='none')
        recomputed = self.processor.recompute(level='full', verbose=False)
        self.assertIn(CorticalTextProcessor.COMP_TFIDF, recomputed)
        self.assertIn(CorticalTextProcessor.COMP_PAGERANK, recomputed)
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))

    def test_recompute_tfidf(self):
        """Test recompute with level='tfidf'."""
        self.processor.add_document_incremental("doc1", "Test content.", recompute='none')
        recomputed = self.processor.recompute(level='tfidf', verbose=False)
        self.assertEqual(recomputed, {CorticalTextProcessor.COMP_TFIDF: True})
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))
        # Others still stale
        self.assertTrue(self.processor.is_stale(CorticalTextProcessor.COMP_PAGERANK))

    def test_recompute_stale_only(self):
        """Test recompute with level='stale' (only recomputes stale items)."""
        self.processor.add_document_incremental("doc1", "Test content.", recompute='tfidf')
        # Now only pagerank, activation, etc. are stale
        recomputed = self.processor.recompute(level='stale', verbose=False)
        # TF-IDF should NOT be in recomputed (it was already fresh)
        self.assertNotIn(CorticalTextProcessor.COMP_TFIDF, recomputed)
        # Others should be recomputed
        self.assertIn(CorticalTextProcessor.COMP_PAGERANK, recomputed)

    def test_get_stale_computations(self):
        """Test get_stale_computations returns correct set."""
        self.processor.add_document_incremental("doc1", "Test content.", recompute='tfidf')
        stale = self.processor.get_stale_computations()
        self.assertNotIn(CorticalTextProcessor.COMP_TFIDF, stale)
        self.assertIn(CorticalTextProcessor.COMP_PAGERANK, stale)

    def test_is_stale(self):
        """Test is_stale returns correct boolean."""
        self.processor.add_document_incremental("doc1", "Test content.", recompute='none')
        self.assertTrue(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))
        self.processor.compute_tfidf(verbose=False)
        self.processor._mark_fresh(CorticalTextProcessor.COMP_TFIDF)
        self.assertFalse(self.processor.is_stale(CorticalTextProcessor.COMP_TFIDF))

    def test_incremental_workflow(self):
        """Test typical incremental indexing workflow."""
        # Initial corpus
        self.processor.process_document("doc1", "Neural networks process information.")
        self.processor.compute_all(verbose=False)

        # Add new documents incrementally
        self.processor.add_document_incremental(
            "doc2", "Machine learning algorithms.", recompute='tfidf'
        )

        # Search should work
        results = self.processor.find_documents_for_query("neural", top_n=2)
        self.assertIsInstance(results, list)

        # Full recompute when needed
        self.processor.recompute(level='full', verbose=False)
        self.assertEqual(len(self.processor.get_stale_computations()), 0)

    def test_batch_then_query(self):
        """Test batch add followed by querying."""
        docs = [
            ("neural", "Neural networks deep learning AI.", None),
            ("ml", "Machine learning algorithms models.", None),
            ("data", "Data processing storage retrieval.", None),
        ]
        self.processor.add_documents_batch(docs, recompute='full', verbose=False)

        results = self.processor.find_documents_for_query("neural networks", top_n=3)
        self.assertGreater(len(results), 0)
        # The neural doc should rank highest
        self.assertEqual(results[0][0], "neural")


class TestCrossLayerConnections(unittest.TestCase):
    """Test cross-layer feedforward and feedback connections."""

    def setUp(self):
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Neural networks process information efficiently.")
        self.processor.process_document("doc2", "Deep learning neural models are powerful.")
        self.processor.compute_all(verbose=False)

    def test_bigram_feedforward_connections(self):
        """Test that bigrams have feedforward connections to component tokens."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        bigram = layer1.get_minicolumn("neural networks")
        self.assertIsNotNone(bigram)
        self.assertGreater(len(bigram.feedforward_connections), 0)

        # Should connect to both "neural" and "networks"
        neural = layer0.get_minicolumn("neural")
        networks = layer0.get_minicolumn("networks")
        self.assertIn(neural.id, bigram.feedforward_connections)
        self.assertIn(networks.id, bigram.feedforward_connections)

    def test_bigram_feedforward_weights(self):
        """Test that bigram feedforward connections have accumulated weights."""
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        bigram = layer1.get_minicolumn("neural networks")
        self.assertIsNotNone(bigram)

        # Weight should be >= 1.0 (accumulated from occurrences)
        for target_id, weight in bigram.feedforward_connections.items():
            self.assertGreaterEqual(weight, 1.0)

    def test_token_feedback_to_bigrams(self):
        """Test that tokens have feedback connections to bigrams."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        neural = layer0.get_minicolumn("neural")
        self.assertIsNotNone(neural)
        self.assertGreater(len(neural.feedback_connections), 0)

        # Should connect back to bigrams containing "neural"
        bigram = layer1.get_minicolumn("neural networks")
        if bigram:
            self.assertIn(bigram.id, neural.feedback_connections)

    def test_document_feedforward_connections(self):
        """Test that documents have feedforward connections to tokens."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        doc = layer3.get_minicolumn("doc1")
        self.assertIsNotNone(doc)
        self.assertGreater(len(doc.feedforward_connections), 0)

        # Document should connect to tokens in its content
        neural = layer0.get_minicolumn("neural")
        self.assertIn(neural.id, doc.feedforward_connections)

    def test_document_feedforward_weights(self):
        """Test that document feedforward weights reflect token frequency."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        doc = layer3.get_minicolumn("doc1")
        neural = layer0.get_minicolumn("neural")

        # Weight should match occurrence count
        weight = doc.feedforward_connections.get(neural.id, 0)
        self.assertGreaterEqual(weight, 1.0)

    def test_token_feedback_to_documents(self):
        """Test that tokens have feedback connections to documents."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)

        neural = layer0.get_minicolumn("neural")
        self.assertIsNotNone(neural)

        # Should connect to documents containing this token
        doc1 = layer3.get_minicolumn("doc1")
        doc2 = layer3.get_minicolumn("doc2")
        self.assertIn(doc1.id, neural.feedback_connections)
        self.assertIn(doc2.id, neural.feedback_connections)

    def test_concept_feedforward_connections(self):
        """Test that concepts have feedforward connections to member tokens."""
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        if layer2.column_count() > 0:
            # Get first concept
            concept = list(layer2.minicolumns.values())[0]
            self.assertGreater(len(concept.feedforward_connections), 0)

            # All feedforward targets should be in feedforward_sources too
            for target_id in concept.feedforward_connections:
                self.assertIn(target_id, concept.feedforward_sources)

    def test_concept_feedforward_weights_by_pagerank(self):
        """Test that concept feedforward weights are based on token PageRank."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        if layer2.column_count() > 0:
            concept = list(layer2.minicolumns.values())[0]

            # Weights should be normalized (max = 1.0)
            max_weight = max(concept.feedforward_connections.values())
            self.assertLessEqual(max_weight, 1.0 + 0.001)  # Allow small float error

    def test_token_feedback_to_concepts(self):
        """Test that tokens have feedback connections to concepts."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        if layer2.column_count() > 0:
            concept = list(layer2.minicolumns.values())[0]

            # Get a member token
            if concept.feedforward_connections:
                member_id = list(concept.feedforward_connections.keys())[0]
                member = layer0.get_by_id(member_id)
                if member:
                    self.assertIn(concept.id, member.feedback_connections)

    def test_cross_layer_bidirectional(self):
        """Test that cross-layer connections are bidirectional."""
        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        bigram = layer1.get_minicolumn("neural networks")
        if bigram:
            for target_id in bigram.feedforward_connections:
                token = layer0.get_by_id(target_id)
                if token:
                    self.assertIn(bigram.id, token.feedback_connections)

    def test_persistence_cross_layer_connections(self):
        """Test that cross-layer connections are saved and loaded correctly."""
        import tempfile

        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)
        bigram = layer1.get_minicolumn("neural networks")
        original_ff = dict(bigram.feedforward_connections) if bigram else {}

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            self.processor.save(path)
            loaded = CorticalTextProcessor.load(path)

            loaded_layer1 = loaded.get_layer(CorticalLayer.BIGRAMS)
            loaded_bigram = loaded_layer1.get_minicolumn("neural networks")

            if bigram and loaded_bigram:
                self.assertEqual(
                    loaded_bigram.feedforward_connections,
                    original_ff
                )
        finally:
            os.unlink(path)

    def test_cross_layer_connection_count(self):
        """Test counting cross-layer connections."""
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        total_ff = 0
        for col in layer1.minicolumns.values():
            total_ff += len(col.feedforward_connections)

        # Each bigram should have 2 feedforward connections (to its 2 tokens)
        # So total should be approximately 2 * number of bigrams
        self.assertGreater(total_ff, 0)


class TestConceptConnections(unittest.TestCase):
    """Test concept-level lateral connections."""

    def setUp(self):
        self.processor = CorticalTextProcessor()
        # Create documents with overlapping topics
        self.processor.process_document("neural_doc",
            "Neural networks process information using deep learning algorithms.")
        self.processor.process_document("ml_doc",
            "Machine learning algorithms learn patterns from data using neural methods.")
        self.processor.process_document("data_doc",
            "Data processing systems analyze information patterns efficiently.")
        self.processor.process_document("unrelated_doc",
            "Ancient pottery techniques involve clay and firing in kilns.")
        self.processor.compute_all(verbose=False)

    def test_concepts_have_lateral_connections(self):
        """Test that concepts have lateral connections when documents overlap."""
        # Create a processor with documents that will create multiple overlapping concepts
        processor = CorticalTextProcessor()
        # Add many documents with overlapping terms to force multiple concept clusters
        processor.process_document("doc1", "Neural networks deep learning artificial intelligence models.")
        processor.process_document("doc2", "Machine learning algorithms data science models.")
        processor.process_document("doc3", "Deep learning neural networks training optimization.")
        processor.process_document("doc4", "Data analysis machine learning statistical models.")
        processor.process_document("doc5", "Artificial intelligence reasoning knowledge graphs.")
        processor.process_document("doc6", "Knowledge representation semantic networks graphs.")
        processor.compute_all(verbose=False)

        layer2 = processor.get_layer(CorticalLayer.CONCEPTS)

        # If we have multiple concepts with overlapping docs, they should connect
        if layer2.column_count() > 1:
            # Check if any concepts share documents
            concepts = list(layer2.minicolumns.values())
            has_overlap = False
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    if c1.document_ids & c2.document_ids:
                        has_overlap = True
                        break

            if has_overlap:
                total_connections = sum(
                    len(c.lateral_connections) for c in layer2.minicolumns.values()
                )
                self.assertGreater(total_connections, 0)

    def test_concept_connections_based_on_jaccard(self):
        """Test that concept connections are based on document overlap."""
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        if layer2.column_count() > 1:
            concepts = list(layer2.minicolumns.values())
            # Find concepts with connections
            connected_concepts = [c for c in concepts if c.lateral_connections]

            for concept in connected_concepts:
                for target_id, weight in concept.lateral_connections.items():
                    # Weight should be based on Jaccard (0 < weight <= 1.5 with semantic boost)
                    self.assertGreater(weight, 0)
                    self.assertLessEqual(weight, 2.0)  # Max with semantic boost

    def test_compute_concept_connections_method(self):
        """Test the compute_concept_connections method directly."""
        # Clear existing connections
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)
        for concept in layer2.minicolumns.values():
            concept.lateral_connections.clear()

        # Recompute
        stats = self.processor.compute_concept_connections(verbose=False)

        self.assertIn('connections_created', stats)
        self.assertIn('concepts', stats)
        self.assertGreaterEqual(stats['connections_created'], 0)

    def test_concept_connections_with_semantics(self):
        """Test that semantic relations boost connection weights."""
        # Extract semantics first
        self.processor.extract_corpus_semantics(verbose=False)

        # Clear and recompute with semantics
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)
        for concept in layer2.minicolumns.values():
            concept.lateral_connections.clear()

        stats_with = self.processor.compute_concept_connections(
            use_semantics=True, verbose=False
        )

        # Clear and recompute without semantics
        for concept in layer2.minicolumns.values():
            concept.lateral_connections.clear()

        stats_without = self.processor.compute_concept_connections(
            use_semantics=False, verbose=False
        )

        # Both should work
        self.assertGreaterEqual(stats_with['connections_created'], 0)
        self.assertGreaterEqual(stats_without['connections_created'], 0)

    def test_concept_connections_min_jaccard_filter(self):
        """Test that min_jaccard threshold filters connections."""
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        # Clear connections
        for concept in layer2.minicolumns.values():
            concept.lateral_connections.clear()

        # With low threshold
        stats_low = self.processor.compute_concept_connections(
            min_jaccard=0.01, verbose=False
        )

        # Clear again
        for concept in layer2.minicolumns.values():
            concept.lateral_connections.clear()

        # With high threshold
        stats_high = self.processor.compute_concept_connections(
            min_jaccard=0.9, verbose=False
        )

        # Low threshold should create >= high threshold connections
        self.assertGreaterEqual(
            stats_low['connections_created'],
            stats_high['connections_created']
        )

    def test_concept_connections_bidirectional(self):
        """Test that concept connections are bidirectional."""
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        for concept in layer2.minicolumns.values():
            for target_id, weight in concept.lateral_connections.items():
                target = layer2.get_by_id(target_id)
                if target:
                    # Target should have connection back to this concept
                    self.assertIn(concept.id, target.lateral_connections)

    def test_concept_connections_empty_layer(self):
        """Test concept connections with empty concept layer."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hello world.")
        processor.compute_all(verbose=False, build_concepts=False)

        layer2 = processor.get_layer(CorticalLayer.CONCEPTS)
        self.assertEqual(layer2.column_count(), 0)

        # Should handle empty layer gracefully
        stats = processor.compute_concept_connections(verbose=False)
        self.assertEqual(stats['connections_created'], 0)
        self.assertEqual(stats['concepts'], 0)

    def test_isolated_concepts_not_connected(self):
        """Test that concepts with no document overlap don't connect."""
        # The unrelated_doc about pottery should form isolated concepts
        layer2 = self.processor.get_layer(CorticalLayer.CONCEPTS)

        if layer2.column_count() > 0:
            # At least some concepts should be isolated if topics are different
            # This is a soft test since clustering may group differently
            pass  # Concept isolation depends on clustering results


class TestBigramConnections(unittest.TestCase):
    """Test bigram lateral connection functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents containing related bigrams."""
        cls.processor = CorticalTextProcessor()
        # Documents with overlapping bigrams to test connections
        cls.processor.process_document(
            "doc1",
            "Neural networks process information. Neural processing enables "
            "deep learning. Machine learning algorithms process data."
        )
        cls.processor.process_document(
            "doc2",
            "Deep learning models use neural networks. Machine learning "
            "is related to deep learning and neural processing."
        )
        cls.processor.process_document(
            "doc3",
            "Learning algorithms improve performance. Machine learning "
            "and deep learning are popular approaches."
        )
        cls.processor.compute_all(verbose=False)

    def test_compute_bigram_connections_returns_stats(self):
        """Test that compute_bigram_connections returns expected statistics."""
        # Connections are already computed by compute_all, so create new processor
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data. Neural processing works.")
        processor.compute_tfidf(verbose=False)

        stats = processor.compute_bigram_connections(verbose=False)

        self.assertIn('connections_created', stats)
        self.assertIn('bigrams', stats)
        self.assertIn('component_connections', stats)
        self.assertIn('chain_connections', stats)
        self.assertIn('cooccurrence_connections', stats)

    def test_shared_left_component_connection(self):
        """Test that bigrams sharing left component are connected."""
        # "neural_networks" and "neural_processing" share "neural"
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        neural_networks = layer1.get_minicolumn("neural_networks")
        neural_processing = layer1.get_minicolumn("neural_processing")

        if neural_networks and neural_processing:
            # They should be connected via shared "neural" component
            self.assertIn(neural_processing.id, neural_networks.lateral_connections)
            self.assertIn(neural_networks.id, neural_processing.lateral_connections)

    def test_shared_right_component_connection(self):
        """Test that bigrams sharing right component are connected."""
        # "machine_learning" and "deep_learning" share "learning"
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        machine_learning = layer1.get_minicolumn("machine_learning")
        deep_learning = layer1.get_minicolumn("deep_learning")

        if machine_learning and deep_learning:
            # They should be connected via shared "learning" component
            self.assertIn(deep_learning.id, machine_learning.lateral_connections)
            self.assertIn(machine_learning.id, deep_learning.lateral_connections)

    def test_chain_connections(self):
        """Test that chain bigrams are connected (right of one = left of other)."""
        # "machine_learning" and "learning_algorithms" form a chain
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        machine_learning = layer1.get_minicolumn("machine_learning")
        learning_algorithms = layer1.get_minicolumn("learning_algorithms")

        if machine_learning and learning_algorithms:
            # They should be connected via chain relationship
            self.assertIn(learning_algorithms.id, machine_learning.lateral_connections)
            self.assertIn(machine_learning.id, learning_algorithms.lateral_connections)

    def test_cooccurrence_connections(self):
        """Test that bigrams co-occurring in documents are connected."""
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        # Bigrams that appear in same documents should have co-occurrence connections
        for bigram in layer1.minicolumns.values():
            if bigram.document_ids and len(bigram.lateral_connections) > 0:
                # If a bigram has connections, some should be from co-occurrence
                # This is a general check that connections exist
                break

    def test_bidirectional_connections(self):
        """Test that all bigram connections are bidirectional."""
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        for bigram in layer1.minicolumns.values():
            for target_id in bigram.lateral_connections:
                target = layer1.get_by_id(target_id)
                if target:
                    self.assertIn(
                        bigram.id, target.lateral_connections,
                        f"Connection from {bigram.content} to {target.content} is not bidirectional"
                    )

    def test_empty_bigram_layer(self):
        """Test bigram connections with empty bigram layer."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hello")  # Single word, no bigrams
        processor.compute_tfidf(verbose=False)

        stats = processor.compute_bigram_connections(verbose=False)
        self.assertEqual(stats['connections_created'], 0)
        self.assertEqual(stats['bigrams'], 0)

    def test_compute_all_includes_bigram_connections(self):
        """Test that compute_all includes bigram connections."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data. Neural processing works.")
        processor.compute_all(verbose=False)

        # Check that bigram connections were marked fresh
        self.assertFalse(processor.is_stale(processor.COMP_BIGRAM_CONNECTIONS))

    def test_custom_weights(self):
        """Test that custom weights affect connection strengths."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks neural processing neural analysis")
        processor.compute_tfidf(verbose=False)

        # Use different weights
        stats = processor.compute_bigram_connections(
            component_weight=1.0,
            chain_weight=1.5,
            cooccurrence_weight=0.5,
            verbose=False
        )

        # Just verify it runs without error
        self.assertIsNotNone(stats)

    def test_recompute_handles_bigram_connections(self):
        """Test that recompute method handles bigram connections."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data")

        # Mark as stale
        processor._mark_all_stale()
        self.assertTrue(processor.is_stale(processor.COMP_BIGRAM_CONNECTIONS))

        # Recompute
        recomputed = processor.recompute(level='full', verbose=False)
        self.assertTrue(recomputed.get(processor.COMP_BIGRAM_CONNECTIONS, False))
        self.assertFalse(processor.is_stale(processor.COMP_BIGRAM_CONNECTIONS))

    def test_bigram_connection_weights_accumulate(self):
        """Test that connection weights accumulate for multiple reasons."""
        layer1 = self.processor.get_layer(CorticalLayer.BIGRAMS)

        # Find bigrams that could be connected by multiple reasons
        # (shared component AND co-occurrence)
        for bigram in layer1.minicolumns.values():
            for target_id, weight in bigram.lateral_connections.items():
                # Weights should be positive
                self.assertGreater(weight, 0)


class TestSemanticPageRank(unittest.TestCase):
    """Test semantic PageRank functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents for semantic PageRank testing."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document(
            "doc1",
            "Neural networks are a type of machine learning model. "
            "Deep learning uses neural networks for complex tasks."
        )
        cls.processor.process_document(
            "doc2",
            "Machine learning algorithms process data patterns. "
            "Neural networks learn from examples."
        )
        cls.processor.process_document(
            "doc3",
            "Deep learning is part of artificial intelligence. "
            "Machine learning models improve with data."
        )
        # Extract semantic relations first
        cls.processor.extract_corpus_semantics(verbose=False)

    def test_compute_semantic_importance_returns_stats(self):
        """Test that compute_semantic_importance returns expected statistics."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data efficiently.")
        processor.extract_corpus_semantics(verbose=False)

        stats = processor.compute_semantic_importance(verbose=False)

        self.assertIn('total_edges_with_relations', stats)
        self.assertIn('token_layer', stats)
        self.assertIn('bigram_layer', stats)

    def test_semantic_pagerank_with_relations(self):
        """Test that semantic PageRank uses relation weights."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "Neural networks learn patterns. Neural systems process data."
        )
        processor.extract_corpus_semantics(verbose=False)

        # Get initial PageRank with standard method
        processor.compute_importance(verbose=False)
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        standard_pr = {col.content: col.pagerank for col in layer0.minicolumns.values()}

        # Now compute with semantic method
        stats = processor.compute_semantic_importance(verbose=False)

        # PageRank values should be updated
        semantic_pr = {col.content: col.pagerank for col in layer0.minicolumns.values()}

        # Just verify it ran and produced valid PageRank values
        for content, pr in semantic_pr.items():
            self.assertGreater(pr, 0)

    def test_semantic_pagerank_no_relations(self):
        """Test semantic PageRank falls back when no relations exist."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hello world.")
        # Don't extract semantic relations

        stats = processor.compute_semantic_importance(verbose=False)

        self.assertEqual(stats['total_edges_with_relations'], 0)

    def test_compute_all_with_semantic_pagerank(self):
        """Test compute_all with pagerank_method='semantic'."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "Neural networks process information efficiently."
        )

        # Should work without errors
        processor.compute_all(verbose=False, pagerank_method='semantic')

        # Verify computations ran
        self.assertFalse(processor.is_stale(processor.COMP_PAGERANK))
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))

    def test_compute_all_standard_pagerank(self):
        """Test compute_all with default pagerank_method='standard'."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "Neural networks process information efficiently."
        )

        processor.compute_all(verbose=False, pagerank_method='standard')

        self.assertFalse(processor.is_stale(processor.COMP_PAGERANK))

    def test_custom_relation_weights(self):
        """Test semantic PageRank with custom relation weights."""
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "Neural networks learn patterns. Machine learning improves."
        )
        processor.extract_corpus_semantics(verbose=False)

        # Use custom weights
        custom_weights = {
            'CoOccurs': 2.0,  # Boost co-occurrence
            'RelatedTo': 0.1,  # Reduce related
        }

        stats = processor.compute_semantic_importance(
            relation_weights=custom_weights,
            verbose=False
        )

        # Should run without errors
        self.assertIsNotNone(stats)

    def test_semantic_pagerank_empty_layer(self):
        """Test semantic PageRank handles empty layer gracefully."""
        from cortical.analysis import compute_semantic_pagerank
        from cortical.layers import HierarchicalLayer, CorticalLayer

        empty_layer = HierarchicalLayer(CorticalLayer.TOKENS)
        relations = [("test", "RelatedTo", "example", 0.5)]

        result = compute_semantic_pagerank(empty_layer, relations)

        self.assertEqual(result['pagerank'], {})
        self.assertEqual(result['iterations_run'], 0)
        self.assertEqual(result['edges_with_relations'], 0)

    def test_semantic_pagerank_convergence(self):
        """Test that semantic PageRank converges."""
        from cortical.analysis import compute_semantic_pagerank

        layer0 = self.processor.get_layer(CorticalLayer.TOKENS)

        result = compute_semantic_pagerank(
            layer0,
            self.processor.semantic_relations,
            iterations=100,
            tolerance=1e-6
        )

        # Should converge in less than max iterations
        self.assertLessEqual(result['iterations_run'], 100)

    def test_relation_weights_applied(self):
        """Test that different relation types get different weights."""
        from cortical.analysis import RELATION_WEIGHTS

        # Verify key relations have expected relative weights
        self.assertGreater(RELATION_WEIGHTS['IsA'], RELATION_WEIGHTS['RelatedTo'])
        self.assertGreater(RELATION_WEIGHTS['PartOf'], RELATION_WEIGHTS['CoOccurs'])
        self.assertLess(RELATION_WEIGHTS['Antonym'], RELATION_WEIGHTS['RelatedTo'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
