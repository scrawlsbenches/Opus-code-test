"""
Unit Tests for processor.py - Search Facade Methods
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


class TestQuickSearch(unittest.TestCase):
    """Tests for quick_search() facade method."""

    def test_quick_search_returns_list_of_doc_ids(self):
        """quick_search returns list of document IDs."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming language")
        processor.process_document("doc2", "java programming syntax")
        processor.compute_all()

        results = processor.quick_search("programming")

        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, str)
            self.assertIn(item, ["doc1", "doc2"])

    def test_quick_search_default_top_n(self):
        """quick_search returns up to 5 results by default."""
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(f"doc{i}", f"test content document {i}")
        processor.compute_all()

        results = processor.quick_search("test")

        self.assertLessEqual(len(results), 5)

    def test_quick_search_custom_top_n(self):
        """quick_search respects custom top_n parameter."""
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(f"doc{i}", f"common content {i}")
        processor.compute_all()

        results = processor.quick_search("common", top_n=3)

        self.assertLessEqual(len(results), 3)

    def test_quick_search_empty_corpus(self):
        """quick_search on empty corpus returns empty list."""
        processor = CorticalTextProcessor()

        results = processor.quick_search("anything")

        self.assertEqual(results, [])

    def test_quick_search_no_match(self):
        """quick_search with no matching terms returns empty list."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming")
        processor.compute_all()

        results = processor.quick_search("xyznonexistent123")

        self.assertEqual(results, [])


class TestRagRetrieve(unittest.TestCase):
    """Tests for rag_retrieve() facade method."""

    def test_rag_retrieve_returns_list_of_dicts(self):
        """rag_retrieve returns list of passage dictionaries."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Python is a programming language.")
        processor.compute_all()

        results = processor.rag_retrieve("python")

        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, dict)

    def test_rag_retrieve_dict_structure(self):
        """rag_retrieve returns dicts with correct keys."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Python is a programming language used for many tasks.")
        processor.compute_all()

        results = processor.rag_retrieve("python")

        if results:  # May be empty if no match
            item = results[0]
            self.assertIn('text', item)
            self.assertIn('doc_id', item)
            self.assertIn('start', item)
            self.assertIn('end', item)
            self.assertIn('score', item)

    def test_rag_retrieve_text_type(self):
        """rag_retrieve returns string text."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Python is a programming language.")
        processor.compute_all()

        results = processor.rag_retrieve("python")

        if results:
            self.assertIsInstance(results[0]['text'], str)

    def test_rag_retrieve_position_types(self):
        """rag_retrieve returns integer positions."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Python is a programming language.")
        processor.compute_all()

        results = processor.rag_retrieve("python")

        if results:
            self.assertIsInstance(results[0]['start'], int)
            self.assertIsInstance(results[0]['end'], int)

    def test_rag_retrieve_default_top_n(self):
        """rag_retrieve returns up to 3 results by default."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content. " * 50)
        processor.compute_all()

        results = processor.rag_retrieve("test")

        self.assertLessEqual(len(results), 3)

    def test_rag_retrieve_custom_top_n(self):
        """rag_retrieve respects custom top_n parameter."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content. " * 50)
        processor.compute_all()

        results = processor.rag_retrieve("test", top_n=1)

        self.assertLessEqual(len(results), 1)

    def test_rag_retrieve_empty_corpus(self):
        """rag_retrieve on empty corpus returns empty list."""
        processor = CorticalTextProcessor()

        results = processor.rag_retrieve("anything")

        self.assertEqual(results, [])

    def test_rag_retrieve_max_chars_parameter(self):
        """rag_retrieve respects max_chars_per_passage parameter."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content. " * 100)
        processor.compute_all()

        results = processor.rag_retrieve("test", max_chars_per_passage=200)

        if results:
            # Passage might be slightly longer due to chunk boundaries
            self.assertLess(len(results[0]['text']), 300)


class TestExplore(unittest.TestCase):
    """Tests for explore() facade method."""

    def test_explore_returns_dict(self):
        """explore returns a dictionary."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming language")
        processor.compute_all()

        result = processor.explore("python")

        self.assertIsInstance(result, dict)

    def test_explore_has_results_key(self):
        """explore result contains 'results' key."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming")
        processor.compute_all()

        result = processor.explore("python")

        self.assertIn('results', result)
        self.assertIsInstance(result['results'], list)

    def test_explore_has_expansion_key(self):
        """explore result contains 'expansion' key."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming")
        processor.compute_all()

        result = processor.explore("python")

        self.assertIn('expansion', result)
        self.assertIsInstance(result['expansion'], dict)

    def test_explore_has_original_terms_key(self):
        """explore result contains 'original_terms' key."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming")
        processor.compute_all()

        result = processor.explore("python")

        self.assertIn('original_terms', result)
        self.assertIsInstance(result['original_terms'], list)

    def test_explore_results_format(self):
        """explore results are (doc_id, score) tuples."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming")
        processor.compute_all()

        result = processor.explore("python")

        if result['results']:
            doc_id, score = result['results'][0]
            self.assertIsInstance(doc_id, str)
            self.assertIsInstance(score, (int, float))

    def test_explore_expansion_contains_query_terms(self):
        """explore expansion includes original query terms."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming language")
        processor.compute_all()

        result = processor.explore("python programming")

        # Original terms should be in expansion with weight 1.0
        self.assertIn('python', result['expansion'])

    def test_explore_original_terms_from_query(self):
        """explore original_terms contains tokenized query."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "python programming")
        processor.compute_all()

        result = processor.explore("python programming")

        self.assertIn('python', result['original_terms'])
        self.assertIn('programming', result['original_terms'])

    def test_explore_default_top_n(self):
        """explore returns up to 5 results by default."""
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(f"doc{i}", f"common term {i}")
        processor.compute_all()

        result = processor.explore("common")

        self.assertLessEqual(len(result['results']), 5)

    def test_explore_custom_top_n(self):
        """explore respects custom top_n parameter."""
        processor = CorticalTextProcessor()
        for i in range(10):
            processor.process_document(f"doc{i}", f"common term {i}")
        processor.compute_all()

        result = processor.explore("common", top_n=2)

        self.assertLessEqual(len(result['results']), 2)

    def test_explore_empty_corpus(self):
        """explore on empty corpus returns valid structure."""
        processor = CorticalTextProcessor()

        result = processor.explore("anything")

        self.assertIn('results', result)
        self.assertIn('expansion', result)
        self.assertIn('original_terms', result)
        self.assertEqual(result['results'], [])




if __name__ == '__main__':
    unittest.main()
