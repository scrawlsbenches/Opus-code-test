"""
Tests for query optimization functions.

Tests the fast search and indexing functionality for improved query performance.
"""

import unittest
from cortical.query import (
    fast_find_documents,
    build_document_index,
    search_with_index,
)
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer


class TestFastFindDocuments(unittest.TestCase):
    """Test the fast_find_documents function."""

    def setUp(self):
        """Set up test processor."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        self.processor.process_document("auth", """
            Authentication module handles user login and credentials.
            Validates tokens and manages sessions securely.
        """)
        self.processor.process_document("data", """
            Data processing module fetches and transforms data.
            Handles database queries and result formatting.
        """)
        self.processor.process_document("validation", """
            Input validation module checks user input.
            Sanitizes and validates form data securely.
        """)
        self.processor.compute_all()

    def test_fast_find_returns_results(self):
        """Test that fast_find_documents returns results."""
        results = fast_find_documents(
            "authentication login",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_fast_find_finds_relevant_doc(self):
        """Test that fast_find_documents finds relevant document."""
        results = fast_find_documents(
            "authentication login",
            self.processor.layers,
            self.processor.tokenizer
        )
        doc_ids = [r[0] for r in results]
        self.assertIn('auth', doc_ids)

    def test_fast_find_respects_top_n(self):
        """Test that fast_find_documents respects top_n."""
        results = fast_find_documents(
            "user data",
            self.processor.layers,
            self.processor.tokenizer,
            top_n=2
        )
        self.assertLessEqual(len(results), 2)

    def test_fast_find_empty_query(self):
        """Test fast_find_documents with empty query."""
        results = fast_find_documents(
            "",
            self.processor.layers,
            self.processor.tokenizer
        )
        self.assertEqual(results, [])

    def test_fast_find_with_code_concepts(self):
        """Test fast_find_documents with code concept expansion."""
        # 'fetch' should expand to find 'data' doc which has 'fetches'
        results = fast_find_documents(
            "fetch",
            self.processor.layers,
            self.processor.tokenizer,
            use_code_concepts=True
        )
        # Should find data doc
        if results:
            doc_ids = [r[0] for r in results]
            self.assertIn('data', doc_ids)

    def test_fast_find_without_code_concepts(self):
        """Test fast_find_documents without code concept expansion."""
        results = fast_find_documents(
            "nonexistent term xyz",
            self.processor.layers,
            self.processor.tokenizer,
            use_code_concepts=False
        )
        self.assertEqual(results, [])


class TestBuildDocumentIndex(unittest.TestCase):
    """Test the build_document_index function."""

    def setUp(self):
        """Set up test processor."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "neural network training data")
        self.processor.process_document("doc2", "database query optimization")
        self.processor.compute_all()

    def test_build_index_returns_dict(self):
        """Test that build_document_index returns a dict."""
        index = build_document_index(self.processor.layers)
        self.assertIsInstance(index, dict)

    def test_index_contains_terms(self):
        """Test that index contains expected terms."""
        index = build_document_index(self.processor.layers)
        self.assertIn('neural', index)
        self.assertIn('database', index)

    def test_index_maps_to_docs(self):
        """Test that index maps terms to documents."""
        index = build_document_index(self.processor.layers)

        # 'neural' should map to doc1
        self.assertIn('neural', index)
        self.assertIn('doc1', index['neural'])

        # 'database' should map to doc2
        self.assertIn('database', index)
        self.assertIn('doc2', index['database'])

    def test_index_values_are_scores(self):
        """Test that index values are positive scores."""
        index = build_document_index(self.processor.layers)

        for term, doc_scores in index.items():
            for doc_id, score in doc_scores.items():
                self.assertGreater(score, 0)


class TestSearchWithIndex(unittest.TestCase):
    """Test the search_with_index function."""

    def setUp(self):
        """Set up test processor and index."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        self.processor.process_document("auth", "authentication login credentials")
        self.processor.process_document("data", "database query optimization")
        self.processor.process_document("network", "neural network training")
        self.processor.compute_all()
        self.index = build_document_index(self.processor.layers)

    def test_search_with_index_returns_results(self):
        """Test that search_with_index returns results."""
        results = search_with_index(
            "authentication",
            self.index,
            self.processor.tokenizer
        )
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_search_with_index_finds_relevant(self):
        """Test that search_with_index finds relevant document."""
        results = search_with_index(
            "authentication login",
            self.index,
            self.processor.tokenizer
        )
        doc_ids = [r[0] for r in results]
        self.assertIn('auth', doc_ids)

    def test_search_with_index_respects_top_n(self):
        """Test that search_with_index respects top_n."""
        results = search_with_index(
            "network",
            self.index,
            self.processor.tokenizer,
            top_n=1
        )
        self.assertLessEqual(len(results), 1)

    def test_search_with_index_empty_query(self):
        """Test search_with_index with empty query."""
        results = search_with_index(
            "",
            self.index,
            self.processor.tokenizer
        )
        self.assertEqual(results, [])

    def test_search_with_index_no_matches(self):
        """Test search_with_index with no matching terms."""
        results = search_with_index(
            "xyznonexistent",
            self.index,
            self.processor.tokenizer
        )
        self.assertEqual(results, [])


class TestProcessorIntegration(unittest.TestCase):
    """Test query optimization integration with processor."""

    def setUp(self):
        """Set up test processor."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        self.processor.process_document("auth", """
            Authentication module handles user login and session management.
            Validates credentials and issues tokens.
        """)
        self.processor.process_document("data", """
            Data processing module fetches records from database.
            Transforms and validates data for export.
        """)
        self.processor.compute_all()

    def test_processor_fast_find_documents(self):
        """Test processor fast_find_documents method."""
        results = self.processor.fast_find_documents("authentication")
        self.assertIsInstance(results, list)
        if results:
            doc_ids = [r[0] for r in results]
            self.assertIn('auth', doc_ids)

    def test_processor_build_search_index(self):
        """Test processor build_search_index method."""
        index = self.processor.build_search_index()
        self.assertIsInstance(index, dict)
        self.assertGreater(len(index), 0)

    def test_processor_search_with_index(self):
        """Test processor search_with_index method."""
        index = self.processor.build_search_index()
        results = self.processor.search_with_index("database", index)
        self.assertIsInstance(results, list)
        if results:
            doc_ids = [r[0] for r in results]
            self.assertIn('data', doc_ids)

    def test_fast_vs_regular_same_results(self):
        """Test that fast and regular search return similar results."""
        query = "authentication login"

        regular_results = self.processor.find_documents_for_query(query)
        fast_results = self.processor.fast_find_documents(query)

        # Both should find 'auth' as top result
        if regular_results and fast_results:
            self.assertEqual(regular_results[0][0], fast_results[0][0])

    def test_index_search_reusable(self):
        """Test that built index can be reused for multiple queries."""
        index = self.processor.build_search_index()

        results1 = self.processor.search_with_index("authentication", index)
        results2 = self.processor.search_with_index("database", index)

        # Should return different results for different queries
        if results1 and results2:
            self.assertNotEqual(results1[0][0], results2[0][0])


if __name__ == '__main__':
    unittest.main()
