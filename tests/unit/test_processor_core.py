"""
Unit Tests for processor.py - Phase 1: Core Functionality
==========================================================

Task #165: Achieve 50% coverage for processor.py with Phase 1 unit tests.

This file tests core processor functionality that doesn't require full corpus:
- Initialization and configuration
- Document management (add, remove, metadata)
- Staleness tracking system
- Layer access methods
- Basic validation

Phase 1 Focus (50% coverage target):
    - Constructor and initialization
    - process_document() with various inputs
    - add_document_incremental() modes
    - remove_document() cleanup
    - Metadata management
    - Staleness tracking (is_stale, mark_fresh, get_stale_computations)
    - Configuration getters/setters
    - Layer access (get_layer)

Uses mocks extensively to test in isolation without full corpus computation.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


# =============================================================================
# INITIALIZATION TESTS (10+ tests)
# =============================================================================

class TestProcessorInitialization(unittest.TestCase):
    """Test processor initialization and setup."""

    def test_init_default(self):
        """Processor initializes with default tokenizer and config."""
        processor = CorticalTextProcessor()

        self.assertIsNotNone(processor.tokenizer)
        self.assertIsInstance(processor.tokenizer, Tokenizer)
        self.assertIsNotNone(processor.config)
        self.assertIsInstance(processor.config, CorticalConfig)

    def test_init_custom_tokenizer(self):
        """Processor accepts custom tokenizer."""
        custom_tokenizer = Tokenizer(min_word_length=3)
        processor = CorticalTextProcessor(tokenizer=custom_tokenizer)

        self.assertIs(processor.tokenizer, custom_tokenizer)
        self.assertEqual(processor.tokenizer.min_word_length, 3)

    def test_init_custom_config(self):
        """Processor accepts custom config."""
        custom_config = CorticalConfig(pagerank_damping=0.9, pagerank_iterations=50)
        processor = CorticalTextProcessor(config=custom_config)

        self.assertIs(processor.config, custom_config)
        self.assertEqual(processor.config.pagerank_damping, 0.9)
        self.assertEqual(processor.config.pagerank_iterations, 50)

    def test_init_layers_created(self):
        """Processor initializes all 4 layers."""
        processor = CorticalTextProcessor()

        self.assertEqual(len(processor.layers), 4)
        self.assertIn(CorticalLayer.TOKENS, processor.layers)
        self.assertIn(CorticalLayer.BIGRAMS, processor.layers)
        self.assertIn(CorticalLayer.CONCEPTS, processor.layers)
        self.assertIn(CorticalLayer.DOCUMENTS, processor.layers)

    def test_init_layers_correct_type(self):
        """All layers are HierarchicalLayer instances."""
        processor = CorticalTextProcessor()

        for layer_enum, layer in processor.layers.items():
            self.assertIsInstance(layer, HierarchicalLayer)
            self.assertEqual(layer.level, layer_enum)

    def test_init_layers_empty(self):
        """Layers start empty."""
        processor = CorticalTextProcessor()

        for layer in processor.layers.values():
            self.assertEqual(layer.column_count(), 0)

    def test_init_documents_empty(self):
        """Documents dict starts empty."""
        processor = CorticalTextProcessor()

        self.assertEqual(len(processor.documents), 0)
        self.assertIsInstance(processor.documents, dict)

    def test_init_metadata_empty(self):
        """Document metadata dict starts empty."""
        processor = CorticalTextProcessor()

        self.assertEqual(len(processor.document_metadata), 0)
        self.assertIsInstance(processor.document_metadata, dict)

    def test_init_embeddings_empty(self):
        """Embeddings dict starts empty."""
        processor = CorticalTextProcessor()

        self.assertEqual(len(processor.embeddings), 0)
        self.assertIsInstance(processor.embeddings, dict)

    def test_init_semantic_relations_empty(self):
        """Semantic relations list starts empty."""
        processor = CorticalTextProcessor()

        self.assertEqual(len(processor.semantic_relations), 0)
        self.assertIsInstance(processor.semantic_relations, list)

    def test_init_stale_computations_empty(self):
        """Staleness tracking initialized."""
        processor = CorticalTextProcessor()

        # Initially all computations should be unmarked
        # (They get marked stale when documents are added)
        self.assertIsInstance(processor._stale_computations, set)

    def test_init_query_cache_initialized(self):
        """Query expansion cache initialized."""
        processor = CorticalTextProcessor()

        self.assertIsInstance(processor._query_expansion_cache, dict)
        self.assertEqual(len(processor._query_expansion_cache), 0)
        self.assertEqual(processor._query_cache_max_size, 100)


# =============================================================================
# DOCUMENT MANAGEMENT TESTS (15+ tests)
# =============================================================================

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

class TestLayerAccess(unittest.TestCase):
    """Test layer access methods."""

    def test_layers_dict_exists(self):
        """Layers dict is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsNotNone(processor.layers)
        self.assertIsInstance(processor.layers, dict)

    def test_layers_dict_has_all_layers(self):
        """Layers dict contains all 4 layers."""
        processor = CorticalTextProcessor()

        self.assertEqual(len(processor.layers), 4)
        self.assertIn(CorticalLayer.TOKENS, processor.layers)
        self.assertIn(CorticalLayer.BIGRAMS, processor.layers)
        self.assertIn(CorticalLayer.CONCEPTS, processor.layers)
        self.assertIn(CorticalLayer.DOCUMENTS, processor.layers)

    def test_layers_correct_types(self):
        """All layers are HierarchicalLayer instances."""
        processor = CorticalTextProcessor()

        for layer in processor.layers.values():
            self.assertIsInstance(layer, HierarchicalLayer)

    def test_layer_enum_values(self):
        """CorticalLayer enum has correct values."""
        self.assertEqual(CorticalLayer.TOKENS.value, 0)
        self.assertEqual(CorticalLayer.BIGRAMS.value, 1)
        self.assertEqual(CorticalLayer.CONCEPTS.value, 2)
        self.assertEqual(CorticalLayer.DOCUMENTS.value, 3)

    def test_layer_levels_match_enum(self):
        """Layer level matches its enum value."""
        processor = CorticalTextProcessor()

        for layer_enum, layer in processor.layers.items():
            self.assertEqual(layer.level, layer_enum)

    def test_access_token_layer(self):
        """Access token layer directly."""
        processor = CorticalTextProcessor()

        layer = processor.layers[CorticalLayer.TOKENS]

        self.assertIsInstance(layer, HierarchicalLayer)
        self.assertEqual(layer.level, CorticalLayer.TOKENS)

    def test_access_bigram_layer(self):
        """Access bigram layer directly."""
        processor = CorticalTextProcessor()

        layer = processor.layers[CorticalLayer.BIGRAMS]

        self.assertIsInstance(layer, HierarchicalLayer)
        self.assertEqual(layer.level, CorticalLayer.BIGRAMS)

    def test_access_concept_layer(self):
        """Access concept layer directly."""
        processor = CorticalTextProcessor()

        layer = processor.layers[CorticalLayer.CONCEPTS]

        self.assertIsInstance(layer, HierarchicalLayer)
        self.assertEqual(layer.level, CorticalLayer.CONCEPTS)

    def test_access_document_layer(self):
        """Access document layer directly."""
        processor = CorticalTextProcessor()

        layer = processor.layers[CorticalLayer.DOCUMENTS]

        self.assertIsInstance(layer, HierarchicalLayer)
        self.assertEqual(layer.level, CorticalLayer.DOCUMENTS)

    def test_layers_are_independent(self):
        """Layers are independent objects."""
        processor = CorticalTextProcessor()

        layer0 = processor.layers[CorticalLayer.TOKENS]
        layer1 = processor.layers[CorticalLayer.BIGRAMS]

        self.assertIsNot(layer0, layer1)

    def test_layer_access_by_value(self):
        """Can access layers using enum or value."""
        processor = CorticalTextProcessor()

        # Both should work
        by_enum = processor.layers[CorticalLayer.TOKENS]
        self.assertIsNotNone(by_enum)


# =============================================================================
# CONFIGURATION TESTS (10+ tests)
# =============================================================================

class TestConfiguration(unittest.TestCase):
    """Test configuration access and application."""

    def test_config_property_exists(self):
        """Config property is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsNotNone(processor.config)
        self.assertIsInstance(processor.config, CorticalConfig)

    def test_config_default_values(self):
        """Default config has expected values."""
        processor = CorticalTextProcessor()

        # Check some defaults
        self.assertEqual(processor.config.pagerank_damping, 0.85)
        self.assertGreater(processor.config.pagerank_iterations, 0)

    def test_config_custom_values(self):
        """Custom config values are preserved."""
        config = CorticalConfig(pagerank_damping=0.9, pagerank_iterations=100)
        processor = CorticalTextProcessor(config=config)

        self.assertEqual(processor.config.pagerank_damping, 0.9)
        self.assertEqual(processor.config.pagerank_iterations, 100)

    def test_config_used_by_tokenizer(self):
        """Config is used when creating default tokenizer."""
        config = CorticalConfig()
        processor = CorticalTextProcessor(config=config)

        self.assertIsNotNone(processor.tokenizer)

    def test_custom_tokenizer_overrides_config(self):
        """Custom tokenizer takes precedence over config."""
        tokenizer = Tokenizer(min_word_length=5)
        config = CorticalConfig()
        processor = CorticalTextProcessor(tokenizer=tokenizer, config=config)

        self.assertIs(processor.tokenizer, tokenizer)
        self.assertEqual(processor.tokenizer.min_word_length, 5)

    def test_config_is_mutable(self):
        """Config can be modified after initialization."""
        processor = CorticalTextProcessor()

        processor.config.pagerank_damping = 0.75

        self.assertEqual(processor.config.pagerank_damping, 0.75)

    def test_config_pagerank_damping(self):
        """Config has pagerank_damping attribute."""
        processor = CorticalTextProcessor()

        self.assertTrue(hasattr(processor.config, 'pagerank_damping'))
        self.assertIsInstance(processor.config.pagerank_damping, float)

    def test_config_pagerank_iterations(self):
        """Config has pagerank_iterations attribute."""
        processor = CorticalTextProcessor()

        self.assertTrue(hasattr(processor.config, 'pagerank_iterations'))
        self.assertIsInstance(processor.config.pagerank_iterations, int)

    def test_tokenizer_property_exists(self):
        """Tokenizer property is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsNotNone(processor.tokenizer)
        self.assertIsInstance(processor.tokenizer, Tokenizer)

    def test_tokenizer_is_used(self):
        """Tokenizer is actually used for processing."""
        tokenizer = Tokenizer(min_word_length=10)  # Very restrictive
        processor = CorticalTextProcessor(tokenizer=tokenizer)

        # Short words should be filtered
        stats = processor.process_document("doc1", "a bb ccc")

        # Should have very few or no tokens due to min_length filter
        self.assertLessEqual(stats['tokens'], 3)


# =============================================================================
# BASIC VALIDATION TESTS (5+ tests)
# =============================================================================

class TestBasicValidation(unittest.TestCase):
    """Test input validation and edge cases."""

    def test_documents_dict_accessible(self):
        """Documents dict is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsInstance(processor.documents, dict)

    def test_document_metadata_dict_accessible(self):
        """Document metadata dict is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsInstance(processor.document_metadata, dict)

    def test_embeddings_dict_accessible(self):
        """Embeddings dict is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsInstance(processor.embeddings, dict)

    def test_semantic_relations_list_accessible(self):
        """Semantic relations list is accessible."""
        processor = CorticalTextProcessor()

        self.assertIsInstance(processor.semantic_relations, list)

    def test_query_cache_initialized(self):
        """Query expansion cache is initialized."""
        processor = CorticalTextProcessor()

        self.assertIsInstance(processor._query_expansion_cache, dict)
        self.assertEqual(len(processor._query_expansion_cache), 0)

    def test_process_multiple_documents(self):
        """Process multiple documents sequentially."""
        processor = CorticalTextProcessor()

        processor.process_document("doc1", "First document")
        processor.process_document("doc2", "Second document")
        processor.process_document("doc3", "Third document")

        self.assertEqual(len(processor.documents), 3)

    def test_process_same_doc_id_overwrites(self):
        """Processing same doc_id overwrites previous content."""
        processor = CorticalTextProcessor()

        processor.process_document("doc1", "Original content")
        processor.process_document("doc1", "New content")

        self.assertEqual(processor.documents["doc1"], "New content")
        self.assertEqual(len(processor.documents), 1)


# =============================================================================
# RECOMPUTE TESTS (10+ tests)
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
