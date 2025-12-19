"""
Unit Tests for processor.py - Document Management
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


class TestDocumentManagement(unittest.TestCase):
    """Test document addition, removal, and metadata."""

    def test_process_document_basic(self):
        """Process a simple document."""
        processor = CorticalTextProcessor()
        stats = processor.process_document("doc1", "Hello world test")

        self.assertIn("doc1", processor.documents)
        self.assertEqual(processor.documents["doc1"], "Hello world test")
        self.assertIsInstance(stats, dict)
        self.assertIn('tokens', stats)
        self.assertIn('bigrams', stats)
        self.assertIn('unique_tokens', stats)

    def test_process_document_stats(self):
        """Process document returns correct statistics."""
        processor = CorticalTextProcessor()
        stats = processor.process_document("doc1", "neural networks process data")

        self.assertGreater(stats['tokens'], 0)
        self.assertGreater(stats['bigrams'], 0)
        self.assertGreater(stats['unique_tokens'], 0)
        self.assertLessEqual(stats['unique_tokens'], stats['tokens'])

    def test_process_document_with_metadata(self):
        """Process document with metadata."""
        processor = CorticalTextProcessor()
        metadata = {"source": "web", "author": "AI", "timestamp": "2025-12-12"}
        stats = processor.process_document("doc1", "Test content", metadata)

        self.assertIn("doc1", processor.document_metadata)
        self.assertEqual(processor.document_metadata["doc1"]["source"], "web")
        self.assertEqual(processor.document_metadata["doc1"]["author"], "AI")

    def test_process_document_metadata_copied(self):
        """Metadata is copied, not referenced."""
        processor = CorticalTextProcessor()
        metadata = {"key": "value"}
        processor.process_document("doc1", "Test", metadata)

        # Modify original
        metadata["key"] = "modified"

        # Stored metadata should be unchanged
        self.assertEqual(processor.document_metadata["doc1"]["key"], "value")

    def test_process_document_updates_layers(self):
        """Processing document updates layers."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")

        layer0 = processor.layers[CorticalLayer.TOKENS]
        layer1 = processor.layers[CorticalLayer.BIGRAMS]
        layer3 = processor.layers[CorticalLayer.DOCUMENTS]

        self.assertGreater(layer0.column_count(), 0)  # Tokens created
        self.assertGreater(layer1.column_count(), 0)  # Bigrams created
        self.assertEqual(layer3.column_count(), 1)    # Document created

    def test_process_document_marks_stale(self):
        """Processing document marks all computations stale."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        # All computations should be marked stale
        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))
        self.assertTrue(processor.is_stale(processor.COMP_ACTIVATION))

    def test_process_document_empty_doc_id_raises(self):
        """Empty doc_id raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.process_document("", "content")
        self.assertIn("doc_id", str(ctx.exception))

    def test_process_document_non_string_doc_id_raises(self):
        """Non-string doc_id raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.process_document(123, "content")
        self.assertIn("doc_id", str(ctx.exception))

    def test_process_document_empty_content_raises(self):
        """Empty content raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.process_document("doc1", "")
        self.assertIn("content", str(ctx.exception))

    def test_process_document_whitespace_only_raises(self):
        """Whitespace-only content raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.process_document("doc1", "   \n\t  ")
        self.assertIn("content", str(ctx.exception))

    def test_process_document_non_string_content_raises(self):
        """Non-string content raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.process_document("doc1", 123)
        self.assertIn("content", str(ctx.exception))

    def test_remove_document_basic(self):
        """Remove an existing document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.remove_document("doc1")

        self.assertTrue(result['found'])
        self.assertNotIn("doc1", processor.documents)
        self.assertGreater(result['tokens_affected'], 0)

    def test_remove_document_not_found(self):
        """Removing non-existent document returns not found."""
        processor = CorticalTextProcessor()

        result = processor.remove_document("nonexistent")

        self.assertFalse(result['found'])
        self.assertEqual(result['tokens_affected'], 0)
        self.assertEqual(result['bigrams_affected'], 0)

    def test_remove_document_clears_metadata(self):
        """Removing document clears its metadata."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test", {"key": "value"})

        processor.remove_document("doc1")

        self.assertNotIn("doc1", processor.document_metadata)

    def test_remove_document_marks_stale(self):
        """Removing document marks all computations stale."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        # Mark everything fresh
        processor._stale_computations.clear()

        processor.remove_document("doc1")

        # Should be stale again
        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))


# =============================================================================
# INCREMENTAL DOCUMENT ADDITION TESTS (10+ tests)
# =============================================================================

class TestIncrementalDocumentAddition(unittest.TestCase):
    """Test add_document_incremental with various recompute modes."""

    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    @patch.object(CorticalTextProcessor, 'compute_all')
    def test_incremental_none_mode(self, mock_compute_all, mock_compute_tfidf):
        """Incremental with recompute='none' doesn't recompute."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        mock_compute_tfidf.assert_not_called()
        mock_compute_all.assert_not_called()

    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    @patch.object(CorticalTextProcessor, 'compute_all')
    def test_incremental_tfidf_mode(self, mock_compute_all, mock_compute_tfidf):
        """Incremental with recompute='tfidf' recomputes TF-IDF only."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='tfidf')

        mock_compute_tfidf.assert_called_once_with(verbose=False)
        mock_compute_all.assert_not_called()

    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    @patch.object(CorticalTextProcessor, 'compute_all')
    def test_incremental_full_mode(self, mock_compute_all, mock_compute_tfidf):
        """Incremental with recompute='full' calls compute_all."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='full')

        mock_compute_all.assert_called_once_with(verbose=False)
        mock_compute_tfidf.assert_not_called()

    def test_incremental_tfidf_marks_fresh(self):
        """Incremental TF-IDF mode marks COMP_TFIDF fresh."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='tfidf')

        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))

    def test_incremental_full_clears_all_stale(self):
        """Incremental full mode clears all stale computations."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='full')

        stale = processor.get_stale_computations()
        self.assertEqual(len(stale), 0)

    def test_incremental_with_metadata(self):
        """Incremental addition with metadata."""
        processor = CorticalTextProcessor()
        metadata = {"source": "test"}
        processor.add_document_incremental("doc1", "test", metadata, recompute='none')

        self.assertEqual(processor.document_metadata["doc1"]["source"], "test")

    def test_incremental_returns_stats(self):
        """Incremental addition returns processing stats."""
        processor = CorticalTextProcessor()
        stats = processor.add_document_incremental("doc1", "test content", recompute='none')

        self.assertIn('tokens', stats)
        self.assertIn('bigrams', stats)
        self.assertIn('unique_tokens', stats)

    def test_incremental_adds_to_corpus(self):
        """Incremental addition adds document to corpus."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test content", recompute='none')

        self.assertIn("doc1", processor.documents)
        self.assertEqual(processor.documents["doc1"], "test content")

    def test_incremental_default_recompute_tfidf(self):
        """Default recompute level is 'tfidf'."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test")

        # TF-IDF should not be stale
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))

    def test_incremental_none_leaves_stale(self):
        """Recompute='none' leaves all computations stale."""
        processor = CorticalTextProcessor()
        processor.add_document_incremental("doc1", "test", recompute='none')

        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))
        self.assertTrue(processor.is_stale(processor.COMP_PAGERANK))


# =============================================================================
# BATCH DOCUMENT OPERATIONS TESTS (10+ tests)
# =============================================================================

class TestBatchDocumentOperations(unittest.TestCase):
    """Test batch add and remove operations."""

    def test_add_documents_batch_basic(self):
        """Add multiple documents in batch."""
        processor = CorticalTextProcessor()
        docs = [
            ("doc1", "First document", None),
            ("doc2", "Second document", None),
            ("doc3", "Third document", None),
        ]

        result = processor.add_documents_batch(docs, recompute='none', verbose=False)

        self.assertEqual(result['documents_added'], 3)
        self.assertIn("doc1", processor.documents)
        self.assertIn("doc2", processor.documents)
        self.assertIn("doc3", processor.documents)

    def test_add_documents_batch_with_metadata(self):
        """Batch add with metadata."""
        processor = CorticalTextProcessor()
        docs = [
            ("doc1", "Content", {"source": "web"}),
            ("doc2", "Content", {"source": "file"}),
        ]

        processor.add_documents_batch(docs, recompute='none', verbose=False)

        self.assertEqual(processor.document_metadata["doc1"]["source"], "web")
        self.assertEqual(processor.document_metadata["doc2"]["source"], "file")

    def test_add_documents_batch_returns_stats(self):
        """Batch add returns comprehensive stats."""
        processor = CorticalTextProcessor()
        docs = [("doc1", "test one", None), ("doc2", "test two", None)]

        result = processor.add_documents_batch(docs, recompute='none', verbose=False)

        self.assertIn('documents_added', result)
        self.assertIn('total_tokens', result)
        self.assertIn('total_bigrams', result)
        self.assertIn('recomputation', result)
        self.assertGreater(result['total_tokens'], 0)

    def test_add_documents_batch_empty_list_raises(self):
        """Empty documents list raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch([], verbose=False)
        self.assertIn("must not be empty", str(ctx.exception))

    def test_add_documents_batch_not_list_raises(self):
        """Non-list documents raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch("not a list", verbose=False)
        self.assertIn("must be a list", str(ctx.exception))

    def test_add_documents_batch_invalid_tuple_raises(self):
        """Invalid document tuple raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch([("doc1",)], verbose=False)  # Missing content
        self.assertIn("must be a tuple", str(ctx.exception))

    def test_add_documents_batch_invalid_doc_id_raises(self):
        """Invalid doc_id in batch raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch([("", "content", None)], verbose=False)
        self.assertIn("doc_id", str(ctx.exception))

    def test_add_documents_batch_invalid_recompute_raises(self):
        """Invalid recompute level raises ValueError."""
        processor = CorticalTextProcessor()
        docs = [("doc1", "content", None)]

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch(docs, recompute='invalid', verbose=False)
        self.assertIn("recompute must be one of", str(ctx.exception))

    @patch.object(CorticalTextProcessor, 'compute_all')
    def test_add_documents_batch_full_recompute(self, mock_compute_all):
        """Batch add with full recompute calls compute_all once."""
        processor = CorticalTextProcessor()
        docs = [("doc1", "test", None), ("doc2", "test", None)]

        processor.add_documents_batch(docs, recompute='full', verbose=False)

        mock_compute_all.assert_called_once_with(verbose=False)

    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    def test_add_documents_batch_tfidf_recompute(self, mock_compute_tfidf):
        """Batch add with TF-IDF recompute calls compute_tfidf once."""
        processor = CorticalTextProcessor()
        docs = [("doc1", "test", None), ("doc2", "test", None)]

        processor.add_documents_batch(docs, recompute='tfidf', verbose=False)

        mock_compute_tfidf.assert_called_once_with(verbose=False)

    def test_remove_documents_batch_basic(self):
        """Remove multiple documents in batch."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")
        processor.process_document("doc2", "test")
        processor.process_document("doc3", "test")

        result = processor.remove_documents_batch(["doc1", "doc2"], verbose=False)

        self.assertEqual(result['documents_removed'], 2)
        self.assertNotIn("doc1", processor.documents)
        self.assertNotIn("doc2", processor.documents)
        self.assertIn("doc3", processor.documents)

    def test_remove_documents_batch_not_found(self):
        """Batch remove tracks not found documents."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.remove_documents_batch(
            ["doc1", "nonexistent1", "nonexistent2"],
            verbose=False
        )

        self.assertEqual(result['documents_removed'], 1)
        self.assertEqual(result['documents_not_found'], 2)


# =============================================================================
# METADATA MANAGEMENT TESTS (10+ tests)
# =============================================================================

class TestMetadataManagement(unittest.TestCase):
    """Test document metadata operations."""

    def test_get_document_metadata_exists(self):
        """Get metadata for existing document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test", {"key": "value"})

        metadata = processor.get_document_metadata("doc1")

        self.assertEqual(metadata["key"], "value")

    def test_get_document_metadata_not_exists(self):
        """Get metadata for non-existent document returns empty dict."""
        processor = CorticalTextProcessor()

        metadata = processor.get_document_metadata("nonexistent")

        self.assertEqual(metadata, {})
        self.assertIsInstance(metadata, dict)

    def test_get_document_metadata_no_metadata_set(self):
        """Get metadata when none was set returns empty dict."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")  # No metadata

        metadata = processor.get_document_metadata("doc1")

        self.assertEqual(metadata, {})

    def test_set_document_metadata_new(self):
        """Set metadata for document that has none."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        processor.set_document_metadata("doc1", source="web", author="AI")

        metadata = processor.get_document_metadata("doc1")
        self.assertEqual(metadata["source"], "web")
        self.assertEqual(metadata["author"], "AI")

    def test_set_document_metadata_update(self):
        """Update existing metadata."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test", {"key1": "value1"})

        processor.set_document_metadata("doc1", key2="value2")

        metadata = processor.get_document_metadata("doc1")
        self.assertEqual(metadata["key1"], "value1")  # Still there
        self.assertEqual(metadata["key2"], "value2")  # Added

    def test_set_document_metadata_overwrite(self):
        """Setting same key overwrites value."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test", {"key": "old"})

        processor.set_document_metadata("doc1", key="new")

        metadata = processor.get_document_metadata("doc1")
        self.assertEqual(metadata["key"], "new")

    def test_set_document_metadata_nonexistent_doc(self):
        """Set metadata for document not in corpus creates entry."""
        processor = CorticalTextProcessor()

        processor.set_document_metadata("doc1", key="value")

        metadata = processor.get_document_metadata("doc1")
        self.assertEqual(metadata["key"], "value")

    def test_get_all_document_metadata_empty(self):
        """Get all metadata when none exists."""
        processor = CorticalTextProcessor()

        all_metadata = processor.get_all_document_metadata()

        self.assertEqual(all_metadata, {})

    def test_get_all_document_metadata_multiple(self):
        """Get all metadata for multiple documents."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test", {"key1": "val1"})
        processor.process_document("doc2", "test", {"key2": "val2"})

        all_metadata = processor.get_all_document_metadata()

        self.assertIn("doc1", all_metadata)
        self.assertIn("doc2", all_metadata)
        self.assertEqual(all_metadata["doc1"]["key1"], "val1")
        self.assertEqual(all_metadata["doc2"]["key2"], "val2")

    def test_get_all_document_metadata_deep_copy(self):
        """get_all_document_metadata returns deep copy."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test", {"key": "value"})

        all_metadata = processor.get_all_document_metadata()
        all_metadata["doc1"]["key"] = "modified"

        # Original should be unchanged
        original = processor.get_document_metadata("doc1")
        self.assertEqual(original["key"], "value")


# =============================================================================
# STALENESS TRACKING TESTS (15+ tests)
# =============================================================================

class TestAdditionalBatchOperations(unittest.TestCase):
    """Additional tests for batch operations."""

    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    def test_remove_batch_with_tfidf_recompute(self, mock_tfidf):
        """Batch remove with TF-IDF recompute."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")
        processor.process_document("doc2", "test")

        processor.remove_documents_batch(["doc1"], recompute='tfidf', verbose=False)

        mock_tfidf.assert_called_once()

    @patch.object(CorticalTextProcessor, 'compute_all')
    def test_remove_batch_with_full_recompute(self, mock_compute_all):
        """Batch remove with full recompute."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        processor.remove_documents_batch(["doc1"], recompute='full', verbose=False)

        mock_compute_all.assert_called_once()

    def test_remove_batch_returns_stats(self):
        """Batch remove returns comprehensive stats."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")
        processor.process_document("doc2", "test")

        result = processor.remove_documents_batch(["doc1", "doc2"], verbose=False)

        self.assertIn('documents_removed', result)
        self.assertIn('documents_not_found', result)
        self.assertIn('total_tokens_affected', result)
        self.assertIn('total_bigrams_affected', result)

    def test_batch_operations_integration(self):
        """Test add and remove batch together."""
        processor = CorticalTextProcessor()

        # Add batch
        add_docs = [("doc1", "test", None), ("doc2", "test", None)]
        processor.add_documents_batch(add_docs, recompute='none', verbose=False)
        self.assertEqual(len(processor.documents), 2)

        # Remove batch
        processor.remove_documents_batch(["doc1"], recompute='none', verbose=False)
        self.assertEqual(len(processor.documents), 1)


# =============================================================================
# EDGE CASES AND ERROR HANDLING (5+ tests)
# =============================================================================

class TestEdgeCasesAndErrors(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_processor_operations(self):
        """Operations on empty processor don't crash."""
        processor = CorticalTextProcessor()

        # Should not raise
        processor._mark_all_stale()
        stale = processor.get_stale_computations()
        self.assertIsInstance(stale, set)

    def test_multiple_metadata_updates(self):
        """Multiple metadata updates work correctly."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        processor.set_document_metadata("doc1", key1="val1")
        processor.set_document_metadata("doc1", key2="val2")
        processor.set_document_metadata("doc1", key1="modified")

        metadata = processor.get_document_metadata("doc1")
        self.assertEqual(metadata["key1"], "modified")
        self.assertEqual(metadata["key2"], "val2")

    def test_process_document_with_special_chars(self):
        """Process document with special characters."""
        processor = CorticalTextProcessor()

        # Should not raise
        stats = processor.process_document("doc1", "Test @#$% content with 123 numbers!")

        self.assertGreater(stats['tokens'], 0)

    def test_process_document_very_long_id(self):
        """Process document with very long ID."""
        processor = CorticalTextProcessor()
        long_id = "x" * 1000

        stats = processor.process_document(long_id, "test content")

        self.assertIn(long_id, processor.documents)

    def test_staleness_persistence_across_operations(self):
        """Staleness state persists correctly."""
        processor = CorticalTextProcessor()

        # Add document
        processor.process_document("doc1", "test")
        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))

        # Mark fresh
        processor._mark_fresh(processor.COMP_TFIDF)
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))

        # Add another document - should be stale again
        processor.process_document("doc2", "test")
        self.assertTrue(processor.is_stale(processor.COMP_TFIDF))




if __name__ == '__main__':
    unittest.main()
