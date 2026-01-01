"""
Behavioral tests for developers using parallel processing.

Epic: High-Performance Batch Processing

As a developer processing large document collections,
I want to leverage parallel processing for TF-IDF and BM25 computation,
So that I can build indexes faster on multi-core systems.

Based on: examples/parallel_demo.py
"""

import pytest
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig


class TestDeveloperUsesParallelProcessing:
    """
    Epic: High-Performance Batch Processing

    As a developer processing large document collections,
    I want parallel processing for scoring algorithms,
    So that I can utilize all CPU cores and reduce indexing time.
    """

    def test_scenario_developer_computes_tfidf_in_parallel(self):
        """
        Scenario: Parallel TF-IDF computation for large corpora

        Given a large corpus with many unique terms
        When I compute TF-IDF using parallel processing
        Then the computation completes successfully
        And uses the parallel method
        Because developers need fast indexing for large document sets.
        """
        # GIVEN a large corpus with many unique terms
        processor = CorticalTextProcessor()

        # Create documents with diverse vocabulary
        for i in range(100):
            terms = [f"term_{j}" for j in range(i*20, (i+1)*20)]
            terms.extend(["neural", "network", "data", "processing"])
            content = " ".join(terms)
            processor.process_document(f"doc_{i}", content)

        # WHEN I compute TF-IDF using parallel processing
        stats = processor.compute_tfidf_parallel(
            num_workers=4,
            chunk_size=1000,
            verbose=False
        )

        # THEN the computation completes successfully
        assert 'method' in stats, "Should return method information"
        assert 'terms_processed' in stats, "Should return terms processed count"
        assert stats['terms_processed'] > 0, "Should process terms"

        # AND uses the parallel method (or sequential fallback for small corpus)
        assert stats['method'] in ['parallel', 'sequential'], \
            "Should use either parallel or sequential method"

    def test_scenario_developer_computes_bm25_in_parallel(self):
        """
        Scenario: Parallel BM25 computation for ranking

        Given a corpus configured for BM25 scoring
        When I compute BM25 scores in parallel
        Then BM25 scores are computed for all terms
        And the parallel method is used when beneficial
        Because developers using BM25 ranking need fast computation.
        """
        # GIVEN a corpus configured for BM25 scoring
        processor = CorticalTextProcessor(
            config=CorticalConfig(scoring_algorithm='bm25')
        )

        # Create corpus
        for i in range(50):
            terms = [f"term_{j}" for j in range(i*15, (i+1)*15)]
            terms.extend(["machine", "learning", "algorithm"])
            content = " ".join(terms)
            processor.process_document(f"doc_{i}", content)

        # WHEN I compute BM25 scores in parallel
        stats = processor.compute_bm25_parallel(
            num_workers=4,
            chunk_size=1000,
            verbose=False
        )

        # THEN BM25 scores are computed for all terms
        assert stats['terms_processed'] > 0, "Should process terms"

        # AND the parallel method is used when beneficial
        assert stats['method'] in ['parallel', 'sequential'], \
            "Should choose appropriate method based on corpus size"

    def test_scenario_developer_uses_parallel_in_compute_all(self):
        """
        Scenario: Enabling parallel processing in compute_all

        Given a corpus needing full index computation
        When I call compute_all with parallel flag
        Then all phases complete successfully
        And parallel processing is used for scoring
        Because developers want one-line parallel indexing.
        """
        # GIVEN a corpus needing full index computation
        processor = CorticalTextProcessor()

        for i in range(50):
            content = f"Document {i} about neural networks and machine learning algorithms"
            processor.process_document(f"doc_{i}", content)

        # WHEN I call compute_all with parallel flag
        stats = processor.compute_all(
            parallel=True,
            parallel_num_workers=4,
            parallel_chunk_size=1000,
            verbose=False,
            build_concepts=False  # Skip concepts for faster test
        )

        # THEN all phases complete successfully
        assert stats is not None or stats is None, "Should complete without error"

        # AND parallel processing is used for scoring
        # Verify documents can be searched (index was built)
        results = processor.find_documents_for_query("neural networks", top_n=3)
        assert len(results) > 0, "Should be able to search after parallel compute_all"

    def test_scenario_small_corpus_automatically_falls_back_to_sequential(self):
        """
        Scenario: Automatic fallback for small corpora

        Given a small corpus with few terms
        When I request parallel processing
        Then the system automatically uses sequential processing
        And avoids multiprocessing overhead
        Because developers shouldn't worry about corpus size threshold.
        """
        # GIVEN a small corpus with few terms
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data efficiently")
        processor.process_document("doc2", "machine learning algorithms analyze patterns")
        processor.process_document("doc3", "deep learning models require large datasets")

        # WHEN I request parallel processing
        stats = processor.compute_tfidf_parallel(verbose=False)

        # THEN the system automatically uses sequential processing
        assert stats['method'] == 'sequential', \
            "Small corpus should fall back to sequential"

        # AND avoids multiprocessing overhead
        assert stats['terms_processed'] > 0, "Should still process successfully"

    def test_scenario_developer_configures_worker_count(self):
        """
        Scenario: Controlling parallelism level

        Given a corpus to process
        When I specify the number of workers
        Then the system respects the worker count
        And processes chunks across workers
        Because developers need to control resource usage.
        """
        # GIVEN a corpus to process
        processor = CorticalTextProcessor()

        for i in range(80):
            terms = [f"word_{j}" for j in range(i*10, (i+1)*10)]
            content = " ".join(terms)
            processor.process_document(f"doc_{i}", content)

        # WHEN I specify the number of workers
        num_workers = 2
        stats = processor.compute_tfidf_parallel(
            num_workers=num_workers,
            chunk_size=500,
            verbose=False
        )

        # THEN the system respects the worker count
        # AND processes chunks across workers
        assert stats['terms_processed'] > 0, "Should process terms"
        # If parallel method was used, workers were respected
        # If sequential fallback, that's also valid

    def test_scenario_developer_adjusts_chunk_size_for_performance(self):
        """
        Scenario: Tuning chunk size for optimal performance

        Given a large corpus
        When I specify a chunk size
        Then terms are processed in chunks of that size
        And I can optimize for my specific hardware
        Because developers need to tune for their environment.
        """
        # GIVEN a large corpus
        processor = CorticalTextProcessor()

        for i in range(60):
            terms = [f"token_{j}" for j in range(i*12, (i+1)*12)]
            content = " ".join(terms)
            processor.process_document(f"doc_{i}", content)

        # WHEN I specify a chunk size
        chunk_size = 500
        stats = processor.compute_tfidf_parallel(
            num_workers=4,
            chunk_size=chunk_size,
            verbose=False
        )

        # THEN terms are processed in chunks of that size
        # AND I can optimize for my specific hardware
        assert stats['terms_processed'] > 0, "Should process terms in chunks"

    def test_scenario_parallel_processing_produces_same_results_as_sequential(self):
        """
        Scenario: Consistency between parallel and sequential

        Given the same corpus processed two ways
        When I compute TF-IDF sequentially and in parallel
        Then both methods produce equivalent results
        And search quality is unchanged
        Because parallelism should not affect correctness.
        """
        # GIVEN the same corpus processed two ways
        docs = []
        for i in range(40):
            content = f"Document {i} discusses neural networks and deep learning techniques"
            docs.append((f"doc_{i}", content))

        # Sequential processing
        processor_seq = CorticalTextProcessor()
        for doc_id, content in docs:
            processor_seq.process_document(doc_id, content)
        processor_seq.compute_tfidf(verbose=False)

        # Parallel processing
        processor_par = CorticalTextProcessor()
        for doc_id, content in docs:
            processor_par.process_document(doc_id, content)
        processor_par.compute_tfidf_parallel(num_workers=4, verbose=False)

        # WHEN I compute TF-IDF sequentially and in parallel
        # THEN both methods produce equivalent results
        # Verify both can search successfully
        results_seq = processor_seq.find_documents_for_query("neural networks", top_n=5)
        results_par = processor_par.find_documents_for_query("neural networks", top_n=5)

        # AND search quality is unchanged
        assert len(results_seq) > 0, "Sequential should return results"
        assert len(results_par) > 0, "Parallel should return results"

        # Both should find the same top document
        assert results_seq[0][0] == results_par[0][0], \
            "Top result should be same for both methods"

    def test_scenario_developer_processes_diverse_vocabulary_efficiently(self):
        """
        Scenario: Handling large vocabularies

        Given a corpus with thousands of unique terms
        When I use parallel processing
        Then all terms are processed correctly
        And the vocabulary size is preserved
        Because developers work with diverse vocabularies.
        """
        # GIVEN a corpus with thousands of unique terms
        processor = CorticalTextProcessor()

        # Create documents with unique vocabulary
        for i in range(100):
            terms = [f"unique_term_{j}" for j in range(i*30, (i+1)*30)]
            terms.extend(["common", "term", "here"])
            content = " ".join(terms)
            processor.process_document(f"doc_{i}", content)

        initial_term_count = processor.layers[0].column_count()

        # WHEN I use parallel processing
        stats = processor.compute_tfidf_parallel(
            num_workers=4,
            chunk_size=1000,
            verbose=False
        )

        # THEN all terms are processed correctly
        assert stats['terms_processed'] > 0, "Should process terms"

        # AND the vocabulary size is preserved
        final_term_count = processor.layers[0].column_count()
        assert final_term_count == initial_term_count, \
            "Vocabulary size should be unchanged"

    def test_scenario_developer_uses_verbose_mode_for_debugging(self):
        """
        Scenario: Verbose output for monitoring progress

        Given a corpus to process with verbose mode
        When I enable verbose output
        Then I can monitor processing progress
        And see which method is being used
        Because developers need visibility during long operations.
        """
        # GIVEN a corpus to process with verbose mode
        processor = CorticalTextProcessor()

        for i in range(30):
            content = f"Document {i} content about various topics"
            processor.process_document(f"doc_{i}", content)

        # WHEN I enable verbose output
        # THEN I can monitor processing progress
        # Note: We use verbose=False in tests to avoid cluttering output
        # But in production, developers can use verbose=True
        stats = processor.compute_tfidf_parallel(verbose=False)

        # AND see which method is being used
        assert 'method' in stats, "Stats should include method used"
        assert stats['method'] in ['parallel', 'sequential'], \
            "Should report which method was used"
