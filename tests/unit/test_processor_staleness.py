"""
Unit Tests for processor.py - Staleness Tracking
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


class TestStalenessTracking(unittest.TestCase):
    """Test staleness tracking system."""

    def test_is_stale_initially_false(self):
        """New processor has no stale computations initially."""
        processor = CorticalTextProcessor()

        # Initially nothing is stale until documents are added
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        self.assertFalse(processor.is_stale(processor.COMP_PAGERANK))

    def test_mark_all_stale_sets_all(self):
        """_mark_all_stale marks all computation types."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()

        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))
        self.assertTrue(processor.is_stale(processor.COMP_ACTIVATION))
        self.assertTrue(processor.is_stale(processor.COMP_DOC_CONNECTIONS))
        self.assertTrue(processor.is_stale(processor.COMP_BIGRAM_CONNECTIONS))
        self.assertTrue(processor.is_stale(processor.COMP_CONCEPTS))
        self.assertTrue(processor.is_stale(processor.COMP_EMBEDDINGS))
        self.assertTrue(processor.is_stale(processor.COMP_SEMANTICS))

    def test_mark_fresh_single(self):
        """Mark a single computation as fresh."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()

        processor._mark_fresh(processor.COMP_TFIDF)

        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))  # Others still stale

    def test_mark_fresh_multiple(self):
        """Mark multiple computations as fresh."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()

        processor._mark_fresh(processor.COMP_TFIDF, processor.COMP_PAGERANK)

        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        self.assertFalse(processor.is_stale(processor.COMP_PAGERANK))
        self.assertTrue(processor.is_stale(processor.COMP_ACTIVATION))  # Others still stale

    def test_mark_fresh_nonexistent_safe(self):
        """Marking non-existent computation as fresh doesn't error."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()

        # Should not raise
        processor._mark_fresh("nonexistent_computation")

    def test_get_stale_computations_empty(self):
        """Get stale computations when none are stale."""
        processor = CorticalTextProcessor()

        stale = processor.get_stale_computations()

        self.assertEqual(len(stale), 0)
        self.assertIsInstance(stale, set)

    def test_get_stale_computations_all(self):
        """Get all stale computations."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()

        stale = processor.get_stale_computations()

        self.assertEqual(len(stale), 8)  # All 8 computation types
        self.assertIn(processor.COMP_TFIDF, stale)
        self.assertIn(processor.COMP_PAGERANK, stale)

    def test_get_stale_computations_partial(self):
        """Get stale computations when some are fresh."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()
        processor._mark_fresh(processor.COMP_TFIDF, processor.COMP_PAGERANK)

        stale = processor.get_stale_computations()

        self.assertNotIn(processor.COMP_TFIDF, stale)
        self.assertNotIn(processor.COMP_PAGERANK, stale)
        self.assertIn(processor.COMP_ACTIVATION, stale)

    def test_get_stale_computations_returns_copy(self):
        """get_stale_computations returns a copy."""
        processor = CorticalTextProcessor()
        processor._mark_all_stale()

        stale1 = processor.get_stale_computations()
        stale1.clear()

        # Original should be unchanged
        stale2 = processor.get_stale_computations()
        self.assertGreater(len(stale2), 0)

    def test_process_document_marks_all_stale(self):
        """Processing document marks all computations stale."""
        processor = CorticalTextProcessor()
        # Start fresh
        processor._stale_computations.clear()

        processor.process_document("doc1", "test")

        stale = processor.get_stale_computations()
        self.assertEqual(len(stale), 8)

    def test_remove_document_marks_all_stale(self):
        """Removing document marks all computations stale."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        # Clear stale state
        processor._stale_computations.clear()

        processor.remove_document("doc1")

        stale = processor.get_stale_computations()
        self.assertEqual(len(stale), 8)

    def test_staleness_constants_defined(self):
        """All staleness constants are defined."""
        processor = CorticalTextProcessor()

        # Check all constants exist
        self.assertEqual(processor.COMP_TFIDF, 'tfidf')
        self.assertEqual(processor.COMP_PAGERANK, 'pagerank')
        self.assertEqual(processor.COMP_ACTIVATION, 'activation')
        self.assertEqual(processor.COMP_DOC_CONNECTIONS, 'doc_connections')
        self.assertEqual(processor.COMP_BIGRAM_CONNECTIONS, 'bigram_connections')
        self.assertEqual(processor.COMP_CONCEPTS, 'concepts')
        self.assertEqual(processor.COMP_EMBEDDINGS, 'embeddings')
        self.assertEqual(processor.COMP_SEMANTICS, 'semantics')

    def test_is_stale_unknown_type(self):
        """is_stale with unknown type returns False."""
        processor = CorticalTextProcessor()

        # Unknown computation type
        is_stale = processor.is_stale("unknown_computation")

        self.assertFalse(is_stale)

    def test_stale_after_incremental_none(self):
        """Incremental with recompute='none' leaves all stale."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))

    def test_fresh_after_incremental_tfidf(self):
        """Incremental with recompute='tfidf' marks TFIDF fresh."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='tfidf')

        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        # Others still stale
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))


# =============================================================================
# LAYER ACCESS TESTS (10+ tests)
# =============================================================================


class TestRecompute(unittest.TestCase):
    """Test the recompute() method for batch operations."""

    @patch.object(CorticalTextProcessor, 'compute_all')
    def test_recompute_full(self, mock_compute_all):
        """Recompute full calls compute_all."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        result = processor.recompute(level='full', verbose=False)

        mock_compute_all.assert_called_once_with(verbose=False)
        self.assertTrue(result[processor.COMP_PAGERANK])
        self.assertTrue(result[processor.COMP_TFIDF])

    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    def test_recompute_tfidf(self, mock_compute_tfidf):
        """Recompute tfidf calls compute_tfidf."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        result = processor.recompute(level='tfidf', verbose=False)

        mock_compute_tfidf.assert_called_once_with(verbose=False)
        self.assertTrue(result[processor.COMP_TFIDF])

    def test_recompute_full_clears_stale(self):
        """Recompute full clears all stale computations."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        processor.recompute(level='full', verbose=False)

        stale = processor.get_stale_computations()
        self.assertEqual(len(stale), 0)

    def test_recompute_tfidf_marks_fresh(self):
        """Recompute tfidf marks TFIDF fresh."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        processor.recompute(level='tfidf', verbose=False)

        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))

    @patch.object(CorticalTextProcessor, 'propagate_activation')
    @patch.object(CorticalTextProcessor, 'compute_importance')
    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    def test_recompute_stale_selective(self, mock_tfidf, mock_importance, mock_activation):
        """Recompute stale only recomputes what's needed."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        # Mark only some as stale
        processor._stale_computations = {
            processor.COMP_ACTIVATION,
            processor.COMP_PAGERANK,
            processor.COMP_TFIDF
        }

        result = processor.recompute(level='stale', verbose=False)

        mock_activation.assert_called_once()
        mock_importance.assert_called_once()
        mock_tfidf.assert_called_once()

    def test_recompute_returns_dict(self):
        """Recompute returns dict of what was recomputed."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        result = processor.recompute(level='full', verbose=False)

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_recompute_stale_empty_does_nothing(self):
        """Recompute stale with nothing stale does nothing."""
        processor = CorticalTextProcessor()
        processor._stale_computations.clear()

        result = processor.recompute(level='stale', verbose=False)

        self.assertEqual(len(result), 0)

    def test_recompute_use_case(self):
        """Test typical recompute use case."""
        processor = CorticalTextProcessor()

        # Add multiple documents without recomputing
        processor.add_document_incremental("doc1", "test one", recompute='none')
        processor.add_document_incremental("doc2", "test two", recompute='none')

        # Verify stale
        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))

        # Batch recompute
        processor.recompute(level='tfidf', verbose=False)

        # Verify fresh
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))


# =============================================================================
# ADDITIONAL BATCH TESTS (5+ tests)
# =============================================================================


class TestErrorHandling(unittest.TestCase):
    """Test error handling paths."""

    def test_find_documents_query_validation(self):
        """find_documents_for_query validates input types."""
        processor = CorticalTextProcessor()

        # Empty string
        with self.assertRaises(ValueError):
            processor.find_documents_for_query("")

        # Non-string
        with self.assertRaises(ValueError):
            processor.find_documents_for_query(123)

        # Invalid top_n
        with self.assertRaises(ValueError):
            processor.find_documents_for_query("test", top_n=0)

        # Non-int top_n
        with self.assertRaises(ValueError):
            processor.find_documents_for_query("test", top_n="5")

    def test_set_query_cache_size_validation(self):
        """set_query_cache_size validates positive integer."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError):
            processor.set_query_cache_size(0)

        with self.assertRaises(ValueError):
            processor.set_query_cache_size(-10)

    def test_expand_query_cached_cache_management(self):
        """expand_query_cached manages cache size."""
        processor = CorticalTextProcessor()
        processor.set_query_cache_size(2)  # Small cache

        # Fill cache
        processor.expand_query_cached("query1")
        processor.expand_query_cached("query2")
        processor.expand_query_cached("query3")  # Should evict oldest

        # Cache has a max size
        self.assertLessEqual(len(processor._query_expansion_cache), 2)




if __name__ == '__main__':
    unittest.main()
