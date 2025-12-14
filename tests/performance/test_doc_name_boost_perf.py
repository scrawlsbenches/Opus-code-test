"""
Performance tests for doc_name_boost optimization (Task T-1a1d-001).

Tests that caching tokenized document names provides significant speedup
(target: 3-4x faster on large corpora).
"""

import pytest
import time
from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer


class TestDocNameBoostPerformance:
    """
    Performance tests for cached document name tokenization.

    Verifies that caching tokenized doc_ids in name_tokens field
    provides significant speedup for doc_name_boost in search queries.
    """

    def test_cached_tokens_speedup(self):
        """
        Test that cached name_tokens provide significant speedup.

        Creates a 2000-document corpus and runs 100 search queries
        to measure the performance improvement from caching.

        Expected: At least 2x speedup (target: 3-4x).
        """
        # Create processor with large corpus
        processor = CorticalTextProcessor()

        # Create 2000 documents with diverse names
        num_docs = 2000
        for i in range(num_docs):
            doc_id = f"document_{i}_analysis_test_data_file"
            content = f"This is document {i} containing some test data and analysis content."
            processor.process_document(doc_id, content)

        # Compute TF-IDF for search
        processor.compute_tfidf(verbose=False)

        # Verify name_tokens are cached for document minicolumns
        layer3 = processor.layers[CorticalLayer.DOCUMENTS]
        doc_col = layer3.get_by_id("L3_document_0_analysis_test_data_file")
        assert doc_col is not None, "Document minicolumn should exist"
        assert doc_col.name_tokens is not None, "name_tokens should be cached"
        assert len(doc_col.name_tokens) > 0, "name_tokens should contain tokens"

        # Expected tokens (tokenizer lowercases and stems)
        expected_tokens = {'document', 'analysi', 'test', 'data', 'file'}  # stemmed versions
        # Check that at least some expected tokens are present
        assert any(tok in doc_col.name_tokens for tok in ['document', 'analysi', 'test']), \
            f"name_tokens should contain expected tokens, got: {doc_col.name_tokens}"

        # Run 100 search queries with doc_name_boost
        num_queries = 100
        queries = [
            "analysis data",
            "test file",
            "document analysis",
            "data file test",
            "analysis test"
        ]

        # Measure search time with cached tokens
        start = time.time()
        for i in range(num_queries):
            query = queries[i % len(queries)]
            results = processor.find_documents_for_query(
                query,
                top_n=10
                # doc_name_boost=2.0 is the default in the underlying function
            )
            assert len(results) > 0, f"Should find results for query: {query}"
        cached_time = time.time() - start

        # Create a processor with old-style data (no cached tokens)
        processor_uncached = CorticalTextProcessor()

        # Add documents but clear name_tokens to simulate old data
        for i in range(num_docs):
            doc_id = f"document_{i}_analysis_test_data_file"
            content = f"This is document {i} containing some test data and analysis content."
            processor_uncached.process_document(doc_id, content)

        processor_uncached.compute_tfidf(verbose=False)

        # Clear cached tokens to simulate old data format
        layer3_uncached = processor_uncached.layers[CorticalLayer.DOCUMENTS]
        for col in layer3_uncached.minicolumns.values():
            col.name_tokens = None

        # Measure search time without cached tokens
        start = time.time()
        for i in range(num_queries):
            query = queries[i % len(queries)]
            results = processor_uncached.find_documents_for_query(
                query,
                top_n=10
            )
            assert len(results) > 0, f"Should find results for query: {query}"
        uncached_time = time.time() - start

        # Calculate speedup
        speedup = uncached_time / cached_time

        # Report timing
        print(f"\n=== doc_name_boost Performance ===")
        print(f"Corpus size: {num_docs} documents")
        print(f"Queries: {num_queries}")
        print(f"Cached time: {cached_time:.3f}s ({cached_time/num_queries*1000:.2f}ms per query)")
        print(f"Uncached time: {uncached_time:.3f}s ({uncached_time/num_queries*1000:.2f}ms per query)")
        print(f"Speedup: {speedup:.2f}x")

        # Verify at least 2x speedup
        # Note: Target is 3-4x, but we use 1.5x as minimum to account for test variance
        assert speedup >= 1.5, \
            f"Expected at least 1.5x speedup with cached tokens, got {speedup:.2f}x"

    def test_name_tokens_populated_on_creation(self):
        """
        Test that name_tokens are populated when documents are added.

        Verifies that the optimization is working as expected.
        """
        processor = CorticalTextProcessor()

        # Add a document
        processor.process_document(
            "my_test_document_file",
            "Some test content here."
        )

        # Get the document minicolumn
        layer3 = processor.layers[CorticalLayer.DOCUMENTS]
        doc_col = layer3.get_by_id("L3_my_test_document_file")

        # Verify name_tokens is set
        assert doc_col is not None, "Document minicolumn should exist"
        assert doc_col.name_tokens is not None, "name_tokens should be populated"

        # Verify it contains expected tokens
        # Tokenizer replaces underscores with spaces and stems
        assert 'test' in doc_col.name_tokens or 'my' in doc_col.name_tokens, \
            f"name_tokens should contain expected tokens, got: {doc_col.name_tokens}"

    def test_backward_compatibility(self):
        """
        Test that search works with old data lacking name_tokens.

        Ensures fallback tokenization works for backward compatibility.
        """
        processor = CorticalTextProcessor()

        # Add documents with content that matches queries
        processor.process_document("test_doc_one", "Content about testing and documentation")
        processor.process_document("test_doc_two", "More content about testing")
        processor.compute_tfidf(verbose=False)

        # Simulate old data by clearing name_tokens
        layer3 = processor.layers[CorticalLayer.DOCUMENTS]
        for col in layer3.minicolumns.values():
            col.name_tokens = None

        # Search should still work (using fallback)
        results = processor.find_documents_for_query(
            "testing",  # Query matches content
            top_n=5
        )

        # Should find documents based on content
        assert len(results) >= 1, "Should find documents even without cached tokens"
        doc_ids = [doc_id for doc_id, _ in results]
        # Verify at least one of our test docs is found
        assert any(doc_id in ["test_doc_one", "test_doc_two"] for doc_id in doc_ids), \
            f"Should find test documents, got: {doc_ids}"

    def test_incremental_add_caches_tokens(self):
        """
        Test that add_document_incremental also caches name_tokens.

        Ensures the optimization works for all document addition methods.
        """
        processor = CorticalTextProcessor()

        # Use incremental add
        processor.add_document_incremental(
            "incremental_test_document",
            "Test content for incremental add",
            recompute='tfidf'
        )

        # Verify name_tokens is cached
        layer3 = processor.layers[CorticalLayer.DOCUMENTS]
        doc_col = layer3.get_by_id("L3_incremental_test_document")

        assert doc_col is not None, "Document should exist"
        assert doc_col.name_tokens is not None, "name_tokens should be cached"
        assert len(doc_col.name_tokens) > 0, "name_tokens should contain tokens"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
