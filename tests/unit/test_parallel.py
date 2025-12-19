"""
Tests for parallel processing functionality.

This test suite verifies:
1. Chunking logic works correctly
2. Parallel execution produces same results as sequential
3. Fallback behavior for small corpora
4. Integration with processor methods
"""

import unittest
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.layers import CorticalLayer
from cortical.analysis import (
    ParallelConfig,
    chunk_dict,
    extract_term_stats,
    parallel_tfidf,
    parallel_bm25,
    _tfidf_core,
    _bm25_core
)


class TestChunking(unittest.TestCase):
    """Test the chunk_dict helper function."""

    def test_chunk_dict_basic(self):
        """Test basic chunking."""
        data = {f"term_{i}": i for i in range(10)}
        chunks = chunk_dict(data, chunk_size=3)

        # Should create 4 chunks (3+3+3+1)
        self.assertEqual(len(chunks), 4)

        # All items should be preserved
        merged = {}
        for chunk in chunks:
            merged.update(chunk)
        self.assertEqual(merged, data)

    def test_chunk_dict_exact_multiple(self):
        """Test chunking when size is exact multiple."""
        data = {f"term_{i}": i for i in range(9)}
        chunks = chunk_dict(data, chunk_size=3)

        # Should create exactly 3 chunks
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(len(chunk), 3)

    def test_chunk_dict_single_chunk(self):
        """Test chunking when data fits in one chunk."""
        data = {f"term_{i}": i for i in range(5)}
        chunks = chunk_dict(data, chunk_size=10)

        # Should create 1 chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], data)

    def test_chunk_dict_empty(self):
        """Test chunking empty dictionary."""
        chunks = chunk_dict({}, chunk_size=10)
        self.assertEqual(len(chunks), 0)


class TestTermStatsExtraction(unittest.TestCase):
    """Test extraction of term statistics from layers."""

    def setUp(self):
        """Create a simple processor for testing."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "neural networks process data efficiently")
        self.processor.process_document("doc2", "neural networks are powerful")
        self.processor.process_document("doc3", "data processing with neural nets")

    def test_extract_term_stats(self):
        """Test extracting term stats from layer."""
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # Should have stats for all terms
        self.assertGreater(len(stats), 0)

        # Each stat should be a tuple of (occurrence_count, doc_frequency, doc_counts)
        for term, (occ_count, doc_freq, doc_counts) in stats.items():
            self.assertIsInstance(term, str)
            self.assertIsInstance(occ_count, int)
            self.assertIsInstance(doc_freq, int)
            self.assertIsInstance(doc_counts, dict)

            # Verify consistency
            self.assertGreater(occ_count, 0)
            self.assertGreater(doc_freq, 0)
            self.assertEqual(len(doc_counts), doc_freq)

            # Verify doc_counts sums to occurrence_count
            self.assertEqual(sum(doc_counts.values()), occ_count)

    def test_extract_term_stats_picklable(self):
        """Test that extracted stats are picklable (required for multiprocessing)."""
        import pickle

        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # Should be picklable
        pickled = pickle.dumps(stats)
        unpickled = pickle.loads(pickled)

        # Should match original
        self.assertEqual(unpickled, stats)


class TestParallelTFIDF(unittest.TestCase):
    """Test parallel TF-IDF computation."""

    def setUp(self):
        """Create processor with enough terms to trigger parallel processing."""
        self.processor = CorticalTextProcessor()

        # Create corpus with 3000+ terms to trigger parallel processing
        docs = []
        for i in range(100):
            # Generate unique terms for each document
            terms = [f"term_{j}" for j in range(i*30, (i+1)*30)]
            # Add some common terms
            terms.extend(["common", "neural", "network", "data"])
            docs.append(" ".join(terms))

        for i, doc in enumerate(docs):
            self.processor.process_document(f"doc_{i}", doc)

    def test_parallel_tfidf_matches_sequential(self):
        """Test that parallel TF-IDF produces same results as sequential."""
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)
        num_docs = len(self.processor.documents)

        # Compute with sequential
        sequential_results = _tfidf_core(stats, num_docs)

        # Compute with parallel (force parallel even for small corpus)
        config = ParallelConfig(
            num_workers=2,
            chunk_size=500,
            min_items_for_parallel=100  # Lower threshold for testing
        )
        parallel_results = parallel_tfidf(stats, num_docs, config=config)

        # Results should match exactly
        self.assertEqual(set(sequential_results.keys()), set(parallel_results.keys()))

        for term in sequential_results:
            seq_global, seq_per_doc = sequential_results[term]
            par_global, par_per_doc = parallel_results[term]

            # Global TF-IDF should match
            self.assertAlmostEqual(seq_global, par_global, places=10)

            # Per-document TF-IDF should match
            self.assertEqual(set(seq_per_doc.keys()), set(par_per_doc.keys()))
            for doc_id in seq_per_doc:
                self.assertAlmostEqual(
                    seq_per_doc[doc_id],
                    par_per_doc[doc_id],
                    places=10,
                    msg=f"Mismatch for term={term}, doc={doc_id}"
                )

    def test_parallel_tfidf_fallback(self):
        """Test that parallel falls back to sequential for small corpora."""
        # Create small corpus (below threshold)
        small_processor = CorticalTextProcessor()
        small_processor.process_document("doc1", "neural networks")
        small_processor.process_document("doc2", "data processing")

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # With default config, should fall back to sequential
        config = ParallelConfig()  # min_items_for_parallel=2000 by default
        results = parallel_tfidf(stats, len(small_processor.documents), config=config)

        # Should still produce valid results
        self.assertGreater(len(results), 0)
        for term, (global_tfidf, per_doc_tfidf) in results.items():
            self.assertIsInstance(global_tfidf, float)
            self.assertIsInstance(per_doc_tfidf, dict)


class TestParallelBM25(unittest.TestCase):
    """Test parallel BM25 computation."""

    def setUp(self):
        """Create processor with enough terms to trigger parallel processing."""
        self.processor = CorticalTextProcessor()

        # Create corpus with 3000+ terms to trigger parallel processing
        docs = []
        for i in range(100):
            # Generate unique terms for each document
            terms = [f"term_{j}" for j in range(i*30, (i+1)*30)]
            # Add some common terms
            terms.extend(["common", "neural", "network", "data"])
            docs.append(" ".join(terms))

        for i, doc in enumerate(docs):
            self.processor.process_document(f"doc_{i}", doc)

    def test_parallel_bm25_matches_sequential(self):
        """Test that parallel BM25 produces same results as sequential."""
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)
        num_docs = len(self.processor.documents)

        # Compute with sequential
        sequential_results = _bm25_core(
            stats,
            num_docs,
            self.processor.doc_lengths,
            self.processor.avg_doc_length,
            k1=1.2,
            b=0.75
        )

        # Compute with parallel (force parallel even for small corpus)
        config = ParallelConfig(
            num_workers=2,
            chunk_size=500,
            min_items_for_parallel=100  # Lower threshold for testing
        )
        parallel_results = parallel_bm25(
            stats,
            num_docs,
            self.processor.doc_lengths,
            self.processor.avg_doc_length,
            k1=1.2,
            b=0.75,
            config=config
        )

        # Results should match exactly
        self.assertEqual(set(sequential_results.keys()), set(parallel_results.keys()))

        for term in sequential_results:
            seq_global, seq_per_doc = sequential_results[term]
            par_global, par_per_doc = parallel_results[term]

            # Global BM25 should match
            self.assertAlmostEqual(seq_global, par_global, places=10)

            # Per-document BM25 should match
            self.assertEqual(set(seq_per_doc.keys()), set(par_per_doc.keys()))
            for doc_id in seq_per_doc:
                self.assertAlmostEqual(
                    seq_per_doc[doc_id],
                    par_per_doc[doc_id],
                    places=10,
                    msg=f"Mismatch for term={term}, doc={doc_id}"
                )

    def test_parallel_bm25_fallback(self):
        """Test that parallel falls back to sequential for small corpora."""
        # Create small corpus (below threshold)
        small_processor = CorticalTextProcessor()
        small_processor.process_document("doc1", "neural networks")
        small_processor.process_document("doc2", "data processing")

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # With default config, should fall back to sequential
        config = ParallelConfig()  # min_items_for_parallel=2000 by default
        results = parallel_bm25(
            stats,
            len(small_processor.documents),
            small_processor.doc_lengths,
            small_processor.avg_doc_length,
            config=config
        )

        # Should still produce valid results
        self.assertGreater(len(results), 0)
        for term, (global_bm25, per_doc_bm25) in results.items():
            self.assertIsInstance(global_bm25, float)
            self.assertIsInstance(per_doc_bm25, dict)


class TestProcessorIntegration(unittest.TestCase):
    """Test integration with CorticalTextProcessor methods."""

    def setUp(self):
        """Create processor for testing."""
        self.processor = CorticalTextProcessor()

        # Create corpus with enough variety
        docs = [
            "neural networks process data efficiently",
            "machine learning algorithms analyze patterns",
            "deep learning models require large datasets",
            "artificial intelligence transforms industries",
            "neural networks learn from experience",
        ]

        for i, doc in enumerate(docs):
            self.processor.process_document(f"doc_{i}", doc)

    def test_compute_tfidf_parallel(self):
        """Test compute_tfidf_parallel method."""
        # Get initial state
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        initial_tfidf = {col.content: col.tfidf for col in layer0.minicolumns.values()}

        # Compute with parallel (should fall back to sequential for small corpus)
        stats = self.processor.compute_tfidf_parallel(verbose=False)

        # Should return stats
        self.assertIn('terms_processed', stats)
        self.assertIn('method', stats)

        # Should have updated TF-IDF values
        for col in layer0.minicolumns.values():
            self.assertGreater(col.tfidf, 0.0)
            self.assertIsInstance(col.tfidf_per_doc, dict)

    def test_compute_bm25_parallel(self):
        """Test compute_bm25_parallel method."""
        # Get initial state
        layer0 = self.processor.layers[CorticalLayer.TOKENS]

        # Compute with parallel (should fall back to sequential for small corpus)
        stats = self.processor.compute_bm25_parallel(verbose=False)

        # Should return stats
        self.assertIn('terms_processed', stats)
        self.assertIn('method', stats)
        self.assertIn('k1', stats)
        self.assertIn('b', stats)

        # Should have updated BM25 values
        for col in layer0.minicolumns.values():
            self.assertGreater(col.tfidf, 0.0)
            self.assertIsInstance(col.tfidf_per_doc, dict)

    def test_compute_all_with_parallel(self):
        """Test compute_all with parallel=True."""
        # Compute with parallel flag
        stats = self.processor.compute_all(
            parallel=True,
            parallel_num_workers=2,
            parallel_chunk_size=100,
            verbose=False
        )

        # Should have computed everything
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        for col in layer0.minicolumns.values():
            self.assertGreater(col.tfidf, 0.0)
            self.assertGreater(col.pagerank, 0.0)

    def test_compute_all_default_sequential(self):
        """Test that compute_all defaults to sequential (backward compatibility)."""
        # Default should use sequential
        stats = self.processor.compute_all(verbose=False)

        # Should have computed everything
        layer0 = self.processor.layers[CorticalLayer.TOKENS]
        for col in layer0.minicolumns.values():
            self.assertGreater(col.tfidf, 0.0)
            self.assertGreater(col.pagerank, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_corpus_parallel_tfidf(self):
        """Test parallel TF-IDF with empty corpus."""
        processor = CorticalTextProcessor()
        layer0 = processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # Should handle empty gracefully
        results = parallel_tfidf(stats, 0)
        self.assertEqual(results, {})

    def test_empty_corpus_parallel_bm25(self):
        """Test parallel BM25 with empty corpus."""
        processor = CorticalTextProcessor()
        layer0 = processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # Should handle empty gracefully
        results = parallel_bm25(stats, 0, {}, 0.0)
        self.assertEqual(results, {})

    def test_single_term_parallel(self):
        """Test parallel processing with single term."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        layer0 = processor.layers[CorticalLayer.TOKENS]
        stats = extract_term_stats(layer0)

        # Should handle single term gracefully (will use sequential)
        results = parallel_tfidf(stats, 1)
        self.assertEqual(len(results), 1)

    def test_parallel_config_validation(self):
        """Test ParallelConfig dataclass."""
        # Default config
        config = ParallelConfig()
        self.assertIsNone(config.num_workers)
        self.assertEqual(config.chunk_size, 1000)
        self.assertEqual(config.min_items_for_parallel, 2000)

        # Custom config
        config = ParallelConfig(num_workers=4, chunk_size=500, min_items_for_parallel=100)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.chunk_size, 500)
        self.assertEqual(config.min_items_for_parallel, 100)


if __name__ == '__main__':
    unittest.main()
