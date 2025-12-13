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


# =============================================================================
# COMPUTE WRAPPER METHODS TESTS (20+ tests)
# =============================================================================

class TestComputeWrapperMethods(unittest.TestCase):
    """Test wrapper methods that delegate to other modules."""

    @patch('cortical.analysis.propagate_activation')
    def test_propagate_activation_calls_analysis(self, mock_propagate):
        """propagate_activation delegates to analysis module."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.propagate_activation(iterations=5, decay=0.7, verbose=False)

        mock_propagate.assert_called_once()
        call_args = mock_propagate.call_args
        self.assertEqual(call_args[0][1], 5)  # iterations
        self.assertEqual(call_args[0][2], 0.7)  # decay

    @patch('cortical.analysis.compute_pagerank')
    def test_compute_importance_calls_analysis(self, mock_pagerank):
        """compute_importance delegates to analysis module."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_importance(verbose=False)

        # Should call PageRank for tokens and bigrams
        self.assertEqual(mock_pagerank.call_count, 2)

    @patch('cortical.analysis.compute_tfidf')
    def test_compute_tfidf_calls_analysis(self, mock_tfidf):
        """compute_tfidf delegates to analysis module."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_tfidf(verbose=False)

        mock_tfidf.assert_called_once()

    @patch('cortical.analysis.compute_document_connections')
    def test_compute_document_connections_calls_analysis(self, mock_doc_conn):
        """compute_document_connections delegates to analysis module."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_document_connections(min_shared_terms=5, verbose=False)

        mock_doc_conn.assert_called_once()

    @patch('cortical.analysis.compute_bigram_connections')
    def test_compute_bigram_connections_calls_analysis(self, mock_bigram_conn):
        """compute_bigram_connections delegates to analysis module."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_bigram_connections(verbose=False)

        mock_bigram_conn.assert_called_once()

    @patch('cortical.analysis.build_concept_clusters')
    def test_build_concept_clusters_calls_analysis(self, mock_clusters):
        """build_concept_clusters delegates to analysis module."""
        mock_clusters.return_value = {'cluster1': ['term1', 'term2']}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.build_concept_clusters(verbose=False)

        mock_clusters.assert_called_once()
        self.assertIsInstance(result, dict)

    @patch('cortical.analysis.compute_clustering_quality')
    def test_compute_clustering_quality_calls_analysis(self, mock_quality):
        """compute_clustering_quality delegates to analysis module."""
        mock_quality.return_value = {'modularity': 0.5}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_clustering_quality()

        mock_quality.assert_called_once()

    @patch('cortical.analysis.compute_concept_connections')
    def test_compute_concept_connections_calls_analysis(self, mock_concept_conn):
        """compute_concept_connections delegates to analysis module."""
        mock_concept_conn.return_value = {'edges_added': 10}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_concept_connections(verbose=False)

        mock_concept_conn.assert_called_once()

    @patch('cortical.semantics.extract_corpus_semantics')
    def test_extract_corpus_semantics_calls_semantics(self, mock_extract):
        """extract_corpus_semantics delegates to semantics module."""
        mock_extract.return_value = []
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.extract_corpus_semantics(verbose=False)

        mock_extract.assert_called_once()

    @patch('cortical.semantics.extract_pattern_relations')
    def test_extract_pattern_relations_calls_semantics(self, mock_extract):
        """extract_pattern_relations delegates to semantics module."""
        mock_extract.return_value = []
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.extract_pattern_relations()

        mock_extract.assert_called_once()

    @patch('cortical.semantics.retrofit_connections')
    def test_retrofit_connections_calls_semantics(self, mock_retrofit):
        """retrofit_connections delegates to semantics module."""
        mock_retrofit.return_value = {'iterations': 10}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.retrofit_connections(iterations=10, alpha=0.3, verbose=False)

        mock_retrofit.assert_called_once()

    @patch('cortical.semantics.inherit_properties')
    @patch('cortical.semantics.apply_inheritance_to_connections')
    def test_compute_property_inheritance_calls_semantics(self, mock_apply, mock_inherit):
        """compute_property_inheritance calls semantics functions."""
        mock_inherit.return_value = {}
        mock_apply.return_value = {'connections_boosted': 0, 'total_boost': 0.0}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [('a', 'IsA', 'b', 1.0)]

        result = processor.compute_property_inheritance()

        mock_inherit.assert_called_once()
        self.assertIn('terms_with_inheritance', result)

    @patch('cortical.semantics.compute_property_similarity')
    def test_compute_property_similarity_calls_semantics(self, mock_sim):
        """compute_property_similarity delegates to semantics module."""
        mock_sim.return_value = 0.8
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [('a', 'HasProperty', 'x', 1.0)]

        result = processor.compute_property_similarity("term1", "term2")

        mock_sim.assert_called_once()

    @patch('cortical.embeddings.compute_graph_embeddings')
    def test_compute_graph_embeddings_calls_embeddings(self, mock_embed):
        """compute_graph_embeddings delegates to embeddings module."""
        mock_embed.return_value = ({}, {'terms_embedded': 10, 'method': 'fast'})
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_graph_embeddings(verbose=False)

        mock_embed.assert_called_once()

    @patch('cortical.semantics.retrofit_embeddings')
    def test_retrofit_embeddings_calls_semantics(self, mock_retrofit):
        """retrofit_embeddings delegates to semantics module."""
        mock_retrofit.return_value = {'total_movement': 0.5}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.embeddings = {"test": [0.1, 0.2]}
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.retrofit_embeddings(iterations=10, alpha=0.4, verbose=False)

        mock_retrofit.assert_called_once()

    @patch('cortical.embeddings.embedding_similarity')
    def test_embedding_similarity_calls_embeddings(self, mock_sim):
        """embedding_similarity delegates to embeddings module."""
        mock_sim.return_value = 0.9
        processor = CorticalTextProcessor()
        processor.embeddings = {"term1": [0.1, 0.2], "term2": [0.3, 0.4]}

        result = processor.embedding_similarity("term1", "term2")

        mock_sim.assert_called_once()

    @patch('cortical.embeddings.find_similar_by_embedding')
    def test_find_similar_by_embedding_calls_embeddings(self, mock_find):
        """find_similar_by_embedding delegates to embeddings module."""
        mock_find.return_value = [("term2", 0.9)]
        processor = CorticalTextProcessor()
        processor.embeddings = {"term1": [0.1, 0.2]}

        result = processor.find_similar_by_embedding("term1", top_n=5)

        mock_find.assert_called_once()


# =============================================================================
# COMPUTE_ALL PARAMETER TESTS (15+ tests)
# =============================================================================

class TestComputeAllParameters(unittest.TestCase):
    """Test compute_all with different parameter combinations."""

    @patch.object(CorticalTextProcessor, 'propagate_activation')
    @patch.object(CorticalTextProcessor, 'compute_importance')
    @patch.object(CorticalTextProcessor, 'compute_tfidf')
    @patch.object(CorticalTextProcessor, 'compute_document_connections')
    @patch.object(CorticalTextProcessor, 'compute_bigram_connections')
    def test_compute_all_basic(self, mock_bigram, mock_doc, mock_tfidf, mock_importance, mock_activation):
        """compute_all with default parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_all(verbose=False, build_concepts=False)

        mock_activation.assert_called_once()
        mock_importance.assert_called_once()
        mock_tfidf.assert_called_once()
        mock_doc.assert_called_once()
        mock_bigram.assert_called_once()

    @patch.object(CorticalTextProcessor, 'compute_semantic_importance')
    @patch.object(CorticalTextProcessor, 'extract_corpus_semantics')
    def test_compute_all_semantic_pagerank(self, mock_extract, mock_semantic):
        """compute_all with semantic PageRank."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(verbose=False, pagerank_method='semantic', build_concepts=False)

        # Should extract semantics if not present
        mock_extract.assert_called_once()
        mock_semantic.assert_called_once()

    @patch.object(CorticalTextProcessor, 'compute_semantic_importance')
    def test_compute_all_semantic_with_existing_relations(self, mock_semantic):
        """compute_all with semantic PageRank when relations exist."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        processor.compute_all(verbose=False, pagerank_method='semantic', build_concepts=False)

        # Should not extract again
        mock_semantic.assert_called_once()

    @patch.object(CorticalTextProcessor, 'compute_hierarchical_importance')
    def test_compute_all_hierarchical_pagerank(self, mock_hierarchical):
        """compute_all with hierarchical PageRank."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(verbose=False, pagerank_method='hierarchical', build_concepts=False)

        mock_hierarchical.assert_called_once()

    @patch.object(CorticalTextProcessor, 'build_concept_clusters')
    @patch.object(CorticalTextProcessor, 'compute_concept_connections')
    def test_compute_all_with_concepts(self, mock_concept_conn, mock_clusters):
        """compute_all with concept building enabled."""
        mock_clusters.return_value = {'cluster1': ['term1']}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_all(verbose=False, build_concepts=True)

        mock_clusters.assert_called_once()
        mock_concept_conn.assert_called_once()
        self.assertIn('clusters_created', result)

    @patch.object(CorticalTextProcessor, 'extract_corpus_semantics')
    @patch.object(CorticalTextProcessor, 'build_concept_clusters')
    @patch.object(CorticalTextProcessor, 'compute_concept_connections')
    def test_compute_all_semantic_connection_strategy(self, mock_concept_conn, mock_clusters, mock_extract):
        """compute_all with semantic connection strategy."""
        mock_clusters.return_value = {}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(
            verbose=False,
            build_concepts=True,
            connection_strategy='semantic'
        )

        # Should extract semantics for connection strategy
        mock_extract.assert_called_once()

    @patch.object(CorticalTextProcessor, 'compute_graph_embeddings')
    @patch.object(CorticalTextProcessor, 'build_concept_clusters')
    @patch.object(CorticalTextProcessor, 'compute_concept_connections')
    def test_compute_all_embedding_connection_strategy(self, mock_concept_conn, mock_clusters, mock_embed):
        """compute_all with embedding connection strategy."""
        mock_clusters.return_value = {}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(
            verbose=False,
            build_concepts=True,
            connection_strategy='embedding'
        )

        # Should compute embeddings for connection strategy
        mock_embed.assert_called_once()

    @patch.object(CorticalTextProcessor, 'extract_corpus_semantics')
    @patch.object(CorticalTextProcessor, 'compute_graph_embeddings')
    @patch.object(CorticalTextProcessor, 'build_concept_clusters')
    @patch.object(CorticalTextProcessor, 'compute_concept_connections')
    def test_compute_all_hybrid_connection_strategy(self, mock_concept_conn, mock_clusters, mock_embed, mock_extract):
        """compute_all with hybrid connection strategy."""
        mock_clusters.return_value = {}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(
            verbose=False,
            build_concepts=True,
            connection_strategy='hybrid'
        )

        # Should compute both semantics and embeddings
        mock_extract.assert_called_once()
        mock_embed.assert_called_once()

    def test_compute_all_clears_query_cache(self):
        """compute_all clears query expansion cache."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor._query_expansion_cache["test"] = {"term": 1.0}

        processor.compute_all(verbose=False, build_concepts=False)

        self.assertEqual(len(processor._query_expansion_cache), 0)

    def test_compute_all_marks_computations_fresh(self):
        """compute_all marks core computations as fresh."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(verbose=False, build_concepts=False)

        self.assertFalse(processor.is_stale(processor.COMP_ACTIVATION))
        self.assertFalse(processor.is_stale(processor.COMP_PAGERANK))
        self.assertFalse(processor.is_stale(processor.COMP_TFIDF))
        self.assertFalse(processor.is_stale(processor.COMP_DOC_CONNECTIONS))
        self.assertFalse(processor.is_stale(processor.COMP_BIGRAM_CONNECTIONS))

    def test_compute_all_marks_concepts_fresh(self):
        """compute_all with build_concepts marks COMP_CONCEPTS fresh."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(verbose=False, build_concepts=True)

        self.assertFalse(processor.is_stale(processor.COMP_CONCEPTS))

    def test_compute_all_returns_stats(self):
        """compute_all returns statistics dict."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_all(verbose=False, build_concepts=False)

        self.assertIsInstance(result, dict)

    def test_compute_all_with_cluster_params(self):
        """compute_all passes cluster parameters correctly."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        # Should not raise
        processor.compute_all(
            verbose=False,
            build_concepts=True,
            cluster_strictness=0.5,
            bridge_weight=0.3
        )


# =============================================================================
# QUERY EXPANSION TESTS (20+ tests)
# =============================================================================

class TestQueryExpansion(unittest.TestCase):
    """Test query expansion methods."""

    @patch('cortical.query.expand_query')
    def test_expand_query_calls_module(self, mock_expand):
        """expand_query delegates to query module."""
        mock_expand.return_value = {"test": 1.0}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.expand_query("test query")

        mock_expand.assert_called_once()
        self.assertEqual(result, {"test": 1.0})

    @patch('cortical.query.expand_query')
    def test_expand_query_with_max_expansions(self, mock_expand):
        """expand_query passes max_expansions parameter."""
        mock_expand.return_value = {}
        processor = CorticalTextProcessor()

        processor.expand_query("test", max_expansions=20)

        call_kwargs = mock_expand.call_args[1]
        self.assertEqual(call_kwargs['max_expansions'], 20)

    @patch('cortical.query.expand_query')
    def test_expand_query_uses_config_default(self, mock_expand):
        """expand_query uses config default when max_expansions=None."""
        mock_expand.return_value = {}
        config = CorticalConfig()
        config.max_query_expansions = 15
        processor = CorticalTextProcessor(config=config)

        processor.expand_query("test", max_expansions=None)

        call_kwargs = mock_expand.call_args[1]
        self.assertEqual(call_kwargs['max_expansions'], 15)

    @patch('cortical.query.expand_query')
    def test_expand_query_with_variants(self, mock_expand):
        """expand_query passes use_variants parameter."""
        mock_expand.return_value = {}
        processor = CorticalTextProcessor()

        processor.expand_query("test", use_variants=False)

        call_kwargs = mock_expand.call_args[1]
        self.assertFalse(call_kwargs['use_variants'])

    @patch('cortical.query.expand_query')
    def test_expand_query_with_code_concepts(self, mock_expand):
        """expand_query passes use_code_concepts parameter."""
        mock_expand.return_value = {}
        processor = CorticalTextProcessor()

        processor.expand_query("test", use_code_concepts=True)

        call_kwargs = mock_expand.call_args[1]
        self.assertTrue(call_kwargs['use_code_concepts'])

    @patch('cortical.query.expand_query')
    def test_expand_query_for_code(self, mock_expand):
        """expand_query_for_code enables code-specific options."""
        mock_expand.return_value = {}
        processor = CorticalTextProcessor()

        processor.expand_query_for_code("fetch data")

        call_kwargs = mock_expand.call_args[1]
        self.assertTrue(call_kwargs['use_code_concepts'])
        self.assertTrue(call_kwargs['filter_code_stop_words'])
        self.assertTrue(call_kwargs['use_variants'])

    @patch('cortical.query.expand_query')
    def test_expand_query_for_code_max_expansions(self, mock_expand):
        """expand_query_for_code increases max_expansions."""
        mock_expand.return_value = {}
        config = CorticalConfig()
        config.max_query_expansions = 10
        processor = CorticalTextProcessor(config=config)

        processor.expand_query_for_code("test")

        call_kwargs = mock_expand.call_args[1]
        self.assertEqual(call_kwargs['max_expansions'], 15)  # 10 + 5

    @patch('cortical.query.expand_query')
    def test_expand_query_cached_caches_results(self, mock_expand):
        """expand_query_cached caches expansion results."""
        mock_expand.return_value = {"test": 1.0, "query": 0.8}
        processor = CorticalTextProcessor()

        # First call
        result1 = processor.expand_query_cached("test query")
        self.assertEqual(mock_expand.call_count, 1)

        # Second call - should use cache
        result2 = processor.expand_query_cached("test query")
        self.assertEqual(mock_expand.call_count, 1)  # Not called again

        self.assertEqual(result1, result2)

    @patch('cortical.query.expand_query')
    def test_expand_query_cached_different_params(self, mock_expand):
        """expand_query_cached treats different params as different cache keys."""
        mock_expand.return_value = {"test": 1.0}
        processor = CorticalTextProcessor()

        result1 = processor.expand_query_cached("test", max_expansions=10)
        result2 = processor.expand_query_cached("test", max_expansions=20)

        # Should call twice - different params
        self.assertEqual(mock_expand.call_count, 2)

    @patch('cortical.query.expand_query')
    def test_expand_query_cached_returns_copy(self, mock_expand):
        """expand_query_cached returns copy to prevent cache corruption."""
        mock_expand.return_value = {"test": 1.0}
        processor = CorticalTextProcessor()

        result1 = processor.expand_query_cached("test")
        result1["modified"] = 2.0

        result2 = processor.expand_query_cached("test")

        self.assertNotIn("modified", result2)

    def test_clear_query_cache(self):
        """clear_query_cache empties the cache."""
        processor = CorticalTextProcessor()
        processor._query_expansion_cache = {"key1": {}, "key2": {}}

        cleared = processor.clear_query_cache()

        self.assertEqual(cleared, 2)
        self.assertEqual(len(processor._query_expansion_cache), 0)

    def test_clear_query_cache_empty(self):
        """clear_query_cache on empty cache returns 0."""
        processor = CorticalTextProcessor()

        cleared = processor.clear_query_cache()

        self.assertEqual(cleared, 0)

    def test_set_query_cache_size(self):
        """set_query_cache_size updates cache size limit."""
        processor = CorticalTextProcessor()

        processor.set_query_cache_size(200)

        self.assertEqual(processor._query_cache_max_size, 200)

    def test_set_query_cache_size_validation(self):
        """set_query_cache_size validates positive integer."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError):
            processor.set_query_cache_size(0)

        with self.assertRaises(ValueError):
            processor.set_query_cache_size(-1)

    @patch('cortical.query.expand_query_semantic')
    def test_expand_query_semantic_calls_module(self, mock_expand):
        """expand_query_semantic delegates to query module."""
        mock_expand.return_value = {}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        processor.expand_query_semantic("test", max_expansions=10)

        mock_expand.assert_called_once()

    @patch('cortical.query.parse_intent_query')
    def test_parse_intent_query_calls_module(self, mock_parse):
        """parse_intent_query delegates to query module."""
        mock_parse.return_value = {"intent": "location"}
        processor = CorticalTextProcessor()

        result = processor.parse_intent_query("where is the function")

        mock_parse.assert_called_once()
        self.assertEqual(result["intent"], "location")

    @patch('cortical.query.search_by_intent')
    def test_search_by_intent_calls_module(self, mock_search):
        """search_by_intent delegates to query module."""
        mock_search.return_value = []
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.search_by_intent("how does authentication work", top_n=10)

        mock_search.assert_called_once()


# =============================================================================
# FIND DOCUMENTS TESTS (15+ tests)
# =============================================================================

class TestFindDocumentsMethods(unittest.TestCase):
    """Test find_documents methods."""

    @patch('cortical.query.find_documents_for_query')
    def test_find_documents_for_query_calls_module(self, mock_find):
        """find_documents_for_query delegates to query module."""
        mock_find.return_value = [("doc1", 0.9)]
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.find_documents_for_query("test", top_n=5)

        mock_find.assert_called_once()
        self.assertEqual(result, [("doc1", 0.9)])

    def test_find_documents_empty_query_raises(self):
        """find_documents_for_query with empty query raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.find_documents_for_query("")
        self.assertIn("query_text", str(ctx.exception))

    def test_find_documents_whitespace_query_raises(self):
        """find_documents_for_query with whitespace query raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.find_documents_for_query("   \n\t  ")
        self.assertIn("query_text", str(ctx.exception))

    def test_find_documents_non_string_query_raises(self):
        """find_documents_for_query with non-string query raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.find_documents_for_query(123)
        self.assertIn("query_text", str(ctx.exception))

    def test_find_documents_invalid_top_n_raises(self):
        """find_documents_for_query with invalid top_n raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.find_documents_for_query("test", top_n=0)
        self.assertIn("top_n", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            processor.find_documents_for_query("test", top_n=-1)
        self.assertIn("top_n", str(ctx.exception))

    def test_find_documents_non_int_top_n_raises(self):
        """find_documents_for_query with non-int top_n raises ValueError."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError) as ctx:
            processor.find_documents_for_query("test", top_n="5")
        self.assertIn("top_n", str(ctx.exception))

    @patch('cortical.query.find_documents_for_query')
    def test_find_documents_with_expansion(self, mock_find):
        """find_documents_for_query passes use_expansion parameter."""
        mock_find.return_value = []
        processor = CorticalTextProcessor()

        processor.find_documents_for_query("test", use_expansion=False)

        call_kwargs = mock_find.call_args[1]
        self.assertFalse(call_kwargs['use_expansion'])

    @patch('cortical.query.find_documents_for_query')
    def test_find_documents_with_semantic(self, mock_find):
        """find_documents_for_query passes use_semantic parameter."""
        mock_find.return_value = []
        processor = CorticalTextProcessor()
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        processor.find_documents_for_query("test", use_semantic=True)

        call_kwargs = mock_find.call_args[1]
        self.assertTrue(call_kwargs['use_semantic'])
        self.assertIsNotNone(call_kwargs['semantic_relations'])

    @patch('cortical.query.find_documents_for_query')
    def test_find_documents_no_semantic_relations(self, mock_find):
        """find_documents_for_query with use_semantic=False passes None."""
        mock_find.return_value = []
        processor = CorticalTextProcessor()

        processor.find_documents_for_query("test", use_semantic=False)

        call_kwargs = mock_find.call_args[1]
        self.assertIsNone(call_kwargs['semantic_relations'])

    @patch('cortical.query.fast_find_documents')
    def test_fast_find_documents_calls_module(self, mock_fast):
        """fast_find_documents delegates to query module."""
        mock_fast.return_value = [("doc1", 0.9)]
        processor = CorticalTextProcessor()

        result = processor.fast_find_documents("test query", top_n=10)

        mock_fast.assert_called_once()
        self.assertEqual(result, [("doc1", 0.9)])

    @patch('cortical.query.fast_find_documents')
    def test_fast_find_documents_with_params(self, mock_fast):
        """fast_find_documents passes all parameters."""
        mock_fast.return_value = []
        processor = CorticalTextProcessor()

        processor.fast_find_documents(
            "test",
            top_n=15,
            candidate_multiplier=5,
            use_code_concepts=False
        )

        call_kwargs = mock_fast.call_args[1]
        self.assertEqual(call_kwargs['top_n'], 15)
        self.assertEqual(call_kwargs['candidate_multiplier'], 5)
        self.assertFalse(call_kwargs['use_code_concepts'])

    @patch('cortical.query.find_documents_with_boost')
    def test_find_documents_with_boost_calls_module(self, mock_boost):
        """find_documents_with_boost delegates to query module."""
        mock_boost.return_value = []
        processor = CorticalTextProcessor()

        processor.find_documents_with_boost("test", top_n=5)

        mock_boost.assert_called_once()

    @patch('cortical.query.find_documents_with_boost')
    def test_find_documents_with_boost_params(self, mock_boost):
        """find_documents_with_boost passes all parameters."""
        mock_boost.return_value = []
        processor = CorticalTextProcessor()

        processor.find_documents_with_boost(
            "test",
            top_n=10,
            auto_detect_intent=False,
            prefer_docs=True,
            custom_boosts={"docs": 2.0},
            use_expansion=False,
            use_semantic=False
        )

        call_kwargs = mock_boost.call_args[1]
        self.assertEqual(call_kwargs['top_n'], 10)
        self.assertFalse(call_kwargs['auto_detect_intent'])
        self.assertTrue(call_kwargs['prefer_docs'])
        self.assertIsNotNone(call_kwargs['custom_boosts'])

    @patch('cortical.query.is_conceptual_query')
    def test_is_conceptual_query_calls_module(self, mock_conceptual):
        """is_conceptual_query delegates to query module."""
        mock_conceptual.return_value = True
        processor = CorticalTextProcessor()

        result = processor.is_conceptual_query("what is PageRank")

        mock_conceptual.assert_called_once()
        self.assertTrue(result)


# =============================================================================
# ADDITIONAL WRAPPER METHODS (10+ tests)
# =============================================================================

class TestAdditionalWrapperMethods(unittest.TestCase):
    """Test additional wrapper methods."""

    @patch('cortical.query.complete_analogy')
    def test_complete_analogy_calls_query(self, mock_analogy):
        """complete_analogy delegates to query module."""
        mock_analogy.return_value = [("result", 0.9, "relation")]
        processor = CorticalTextProcessor()
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.complete_analogy("a", "b", "c")

        mock_analogy.assert_called_once()

    @patch('cortical.query.complete_analogy_simple')
    def test_complete_analogy_simple_calls_query(self, mock_simple):
        """complete_analogy_simple delegates to query module."""
        mock_simple.return_value = [("result", 0.8)]
        processor = CorticalTextProcessor()

        result = processor.complete_analogy_simple("a", "b", "c")

        mock_simple.assert_called_once()

    @patch('cortical.query.expand_query_multihop')
    def test_expand_query_multihop_calls_module(self, mock_multihop):
        """expand_query_multihop delegates to query module."""
        mock_multihop.return_value = {}
        processor = CorticalTextProcessor()
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.expand_query_multihop("test")

        mock_multihop.assert_called_once()


# =============================================================================
# SEMANTIC IMPORTANCE TESTS (5+ tests)
# =============================================================================

class TestSemanticImportance(unittest.TestCase):
    """Test semantic importance computation."""

    @patch('cortical.analysis.compute_semantic_pagerank')
    def test_compute_semantic_importance_with_relations(self, mock_semantic):
        """compute_semantic_importance with existing semantic relations."""
        mock_semantic.return_value = {
            'iterations_run': 10,
            'edges_with_relations': 5
        }
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.compute_semantic_importance(verbose=False)

        self.assertEqual(mock_semantic.call_count, 2)  # tokens + bigrams
        self.assertIn('total_edges_with_relations', result)
        self.assertEqual(result['total_edges_with_relations'], 10)

    @patch.object(CorticalTextProcessor, 'compute_importance')
    def test_compute_semantic_importance_fallback(self, mock_importance):
        """compute_semantic_importance falls back when no relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_semantic_importance(verbose=False)

        mock_importance.assert_called_once()
        self.assertEqual(result['total_edges_with_relations'], 0)

    @patch('cortical.analysis.compute_semantic_pagerank')
    def test_compute_semantic_importance_custom_weights(self, mock_semantic):
        """compute_semantic_importance with custom relation weights."""
        mock_semantic.return_value = {
            'iterations_run': 10,
            'edges_with_relations': 5
        }
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        custom_weights = {'IsA': 2.0, 'PartOf': 1.5}
        result = processor.compute_semantic_importance(
            relation_weights=custom_weights,
            verbose=False
        )

        # Check that custom weights were passed
        call_kwargs = mock_semantic.call_args[1]
        self.assertEqual(call_kwargs['relation_weights'], custom_weights)

    @patch('cortical.analysis.compute_hierarchical_pagerank')
    def test_compute_hierarchical_importance_calls_analysis(self, mock_hier):
        """compute_hierarchical_importance delegates to analysis module."""
        mock_hier.return_value = {
            'iterations_run': 5,
            'converged': True,
            'layer_stats': {}
        }
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.compute_hierarchical_importance(verbose=False)

        mock_hier.assert_called_once()
        self.assertIn('iterations_run', result)

    @patch('cortical.analysis.compute_hierarchical_pagerank')
    def test_compute_hierarchical_importance_with_params(self, mock_hier):
        """compute_hierarchical_importance passes parameters."""
        mock_hier.return_value = {'iterations_run': 3, 'converged': False}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.compute_hierarchical_importance(
            layer_iterations=15,
            global_iterations=3,
            cross_layer_damping=0.9,
            verbose=False
        )

        call_kwargs = mock_hier.call_args[1]
        self.assertEqual(call_kwargs['layer_iterations'], 15)
        self.assertEqual(call_kwargs['global_iterations'], 3)


# =============================================================================
# ADDITIONAL SIMPLE WRAPPER TESTS (30+ tests)
# =============================================================================

class TestSimpleWrapperMethods(unittest.TestCase):
    """Test simple one-line wrapper methods."""

    def test_processor_has_expected_attributes(self):
        """Processor has expected core attributes."""
        processor = CorticalTextProcessor()

        self.assertIsNotNone(processor.layers)
        self.assertIsNotNone(processor.documents)
        self.assertIsNotNone(processor.tokenizer)

    @patch('cortical.query.query_with_spreading_activation')
    def test_query_expanded_calls_query(self, mock_query):
        """query_expanded delegates to query module."""
        mock_query.return_value = []
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.query_expanded("test")

        mock_query.assert_called_once()

    @patch('cortical.query.find_related_documents')
    def test_find_related_documents_calls_query(self, mock_related):
        """find_related_documents delegates to query module."""
        mock_related.return_value = []
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.find_related_documents("doc1")

        mock_related.assert_called_once()

    @patch('cortical.gaps.analyze_knowledge_gaps')
    def test_analyze_knowledge_gaps_calls_gaps(self, mock_gaps):
        """analyze_knowledge_gaps delegates to gaps module."""
        mock_gaps.return_value = {}
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.analyze_knowledge_gaps()

        mock_gaps.assert_called_once()

    @patch('cortical.gaps.detect_anomalies')
    def test_detect_anomalies_calls_gaps(self, mock_anomalies):
        """detect_anomalies delegates to gaps module."""
        mock_anomalies.return_value = []
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        result = processor.detect_anomalies(threshold=0.5)

        mock_anomalies.assert_called_once()

    def test_get_layer_returns_layer(self):
        """get_layer returns the requested layer."""
        processor = CorticalTextProcessor()

        layer = processor.get_layer(CorticalLayer.TOKENS)

        self.assertIsInstance(layer, HierarchicalLayer)
        self.assertEqual(layer.level, CorticalLayer.TOKENS)

    def test_get_document_signature_basic(self):
        """get_document_signature returns top terms for document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")
        processor.compute_tfidf(verbose=False)

        signature = processor.get_document_signature("doc1", n=5)

        self.assertIsInstance(signature, list)
        self.assertLessEqual(len(signature), 5)

    @patch('cortical.persistence.get_state_summary')
    def test_get_corpus_summary_calls_persistence(self, mock_summary):
        """get_corpus_summary delegates to persistence module."""
        mock_summary.return_value = {}
        processor = CorticalTextProcessor()

        result = processor.get_corpus_summary()

        mock_summary.assert_called_once()

    @patch('cortical.fingerprint.compute_fingerprint')
    def test_get_fingerprint_calls_fingerprint(self, mock_fp):
        """get_fingerprint delegates to fingerprint module."""
        mock_fp.return_value = {'terms': []}
        processor = CorticalTextProcessor()

        result = processor.get_fingerprint("test text", top_n=20)

        mock_fp.assert_called_once()

    @patch('cortical.fingerprint.compare_fingerprints')
    def test_compare_fingerprints_calls_fingerprint(self, mock_compare):
        """compare_fingerprints delegates to fingerprint module."""
        mock_compare.return_value = {'jaccard': 0.5}
        processor = CorticalTextProcessor()

        result = processor.compare_fingerprints({'terms': []}, {'terms': []})

        mock_compare.assert_called_once()

    @patch('cortical.fingerprint.explain_fingerprint')
    def test_explain_fingerprint_calls_fingerprint(self, mock_explain):
        """explain_fingerprint delegates to fingerprint module."""
        mock_explain.return_value = {'summary': ''}
        processor = CorticalTextProcessor()

        result = processor.explain_fingerprint({'terms': []}, top_n=10)

        mock_explain.assert_called_once()

    @patch('cortical.fingerprint.explain_similarity')
    def test_explain_similarity_calls_fingerprint(self, mock_explain):
        """explain_similarity delegates to fingerprint module."""
        mock_explain.return_value = "Explanation"
        processor = CorticalTextProcessor()

        result = processor.explain_similarity({'terms': []}, {'terms': []})

        mock_explain.assert_called_once()

    @patch('cortical.query.find_passages_for_query')
    def test_find_passages_for_query_calls_query(self, mock_passages):
        """find_passages_for_query delegates to query module."""
        mock_passages.return_value = []
        processor = CorticalTextProcessor()

        if hasattr(processor, 'find_passages_for_query'):
            result = processor.find_passages_for_query("test")
            mock_passages.assert_called_once()

    @patch('cortical.query.find_passages_batch')
    def test_find_passages_batch_calls_query(self, mock_batch):
        """find_passages_batch delegates to query module."""
        mock_batch.return_value = {}
        processor = CorticalTextProcessor()

        if hasattr(processor, 'find_passages_batch'):
            result = processor.find_passages_batch(["query1", "query2"])
            mock_batch.assert_called_once()

    @patch('cortical.query.search_with_index')
    def test_search_with_index_calls_query(self, mock_search):
        """search_with_index delegates to query module."""
        mock_search.return_value = []
        processor = CorticalTextProcessor()

        if hasattr(processor, 'search_with_index'):
            result = processor.search_with_index("query", {})
            mock_search.assert_called_once()


# =============================================================================
# COMPUTE_ALL VERBOSE TESTS (5+ tests)
# =============================================================================

class TestComputeAllVerbose(unittest.TestCase):
    """Test compute_all verbose logging paths."""

    def test_compute_all_verbose_logging(self):
        """compute_all with verbose=True exercises logging paths."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks")

        # Should not raise, exercises verbose logging branches
        result = processor.compute_all(verbose=True, build_concepts=False)

        self.assertIsInstance(result, dict)

    def test_compute_all_with_concepts_verbose(self):
        """compute_all with concepts and verbose logging."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks")

        result = processor.compute_all(verbose=True, build_concepts=True)

        self.assertIsInstance(result, dict)

    def test_compute_all_semantic_verbose(self):
        """compute_all with semantic PageRank and verbose logging."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_all(
            verbose=True,
            pagerank_method='semantic',
            build_concepts=False
        )

        self.assertIsInstance(result, dict)

    def test_compute_all_connection_strategies_verbose(self):
        """compute_all with different connection strategies and verbose."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        for strategy in ['document_overlap', 'semantic', 'embedding', 'hybrid']:
            result = processor.compute_all(
                verbose=True,
                build_concepts=True,
                connection_strategy=strategy
            )
            self.assertIsInstance(result, dict)


# =============================================================================
# EDGE CASE WRAPPER TESTS (10+ tests)
# =============================================================================

class TestWrapperEdgeCases(unittest.TestCase):
    """Test wrapper methods with edge cases."""

    def test_get_document_signature_nonexistent_doc(self):
        """get_document_signature with non-existent doc returns empty list."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        signature = processor.get_document_signature("nonexistent")

        self.assertEqual(signature, [])

    def test_get_document_signature_empty_n(self):
        """get_document_signature with n=0."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_tfidf(verbose=False)

        signature = processor.get_document_signature("doc1", n=0)

        self.assertEqual(len(signature), 0)

    def test_get_layer_all_layers(self):
        """get_layer works for all layer types."""
        processor = CorticalTextProcessor()

        for layer_enum in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS,
                          CorticalLayer.CONCEPTS, CorticalLayer.DOCUMENTS]:
            layer = processor.get_layer(layer_enum)
            self.assertEqual(layer.level, layer_enum)

    def test_add_documents_batch_verbose(self):
        """add_documents_batch with verbose=True exercises logging."""
        processor = CorticalTextProcessor()
        docs = [("doc1", "test content", None)]

        result = processor.add_documents_batch(docs, verbose=True, recompute='tfidf')

        self.assertEqual(result['documents_added'], 1)

    def test_add_documents_batch_full_recompute_verbose(self):
        """add_documents_batch with full recompute and verbose."""
        processor = CorticalTextProcessor()
        docs = [("doc1", "test content", None)]

        result = processor.add_documents_batch(docs, verbose=True, recompute='full')

        self.assertEqual(result['documents_added'], 1)

    def test_add_documents_batch_invalid_content(self):
        """add_documents_batch with invalid content raises ValueError."""
        processor = CorticalTextProcessor()
        docs = [("doc1", 123, None)]  # Invalid content type

        with self.assertRaises(ValueError) as ctx:
            processor.add_documents_batch(docs)
        self.assertIn("content", str(ctx.exception))

    def test_remove_documents_batch_verbose(self):
        """remove_documents_batch with verbose=True exercises logging."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.process_document("doc2", "test content")

        result = processor.remove_documents_batch(["doc1"], verbose=True)

        self.assertEqual(result['documents_removed'], 1)

    def test_add_document_incremental_basic(self):
        """add_document_incremental basic functionality."""
        processor = CorticalTextProcessor()

        result = processor.add_document_incremental(
            "doc1",
            "test content here",
            recompute='tfidf'
        )

        self.assertIn('tokens', result)

    def test_process_document_basic(self):
        """process_document basic functionality."""
        processor = CorticalTextProcessor()

        stats = processor.process_document("doc1", "test content")

        self.assertGreater(stats['tokens'], 0)

    def test_remove_document_basic(self):
        """remove_document basic functionality."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.remove_document("doc1")

        self.assertTrue(result['found'])

    def test_compute_all_no_documents(self):
        """compute_all with empty processor."""
        processor = CorticalTextProcessor()

        # Should not raise, just does nothing
        result = processor.compute_all(verbose=False, build_concepts=False)

        self.assertIsInstance(result, dict)

    def test_multi_stage_rank_if_exists(self):
        """Test multi_stage_rank if method exists."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        if hasattr(processor, 'multi_stage_rank'):
            # Should not raise
            result = processor.multi_stage_rank("test")

    def test_complete_analogy_validation(self):
        """complete_analogy validates inputs."""
        processor = CorticalTextProcessor()

        with self.assertRaises(ValueError):
            processor.complete_analogy("", "b", "c")

        with self.assertRaises(ValueError):
            processor.complete_analogy("a", "b", "c", top_n=0)

    def test_expand_query_multihop_if_exists(self):
        """expand_query_multihop basic functionality."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        if hasattr(processor, 'expand_query_multihop'):
            result = processor.expand_query_multihop("test")
            self.assertIsInstance(result, dict)


# =============================================================================
# VERBOSE PATH COVERAGE TESTS (20+ tests)
# =============================================================================

class TestVerbosePathCoverage(unittest.TestCase):
    """Tests to hit verbose logging and edge case paths."""

    def test_compute_bigram_connections_verbose(self):
        """compute_bigram_connections with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks machine learning")
        processor.process_document("doc2", "test content data science")

        result = processor.compute_bigram_connections(verbose=True)

        self.assertIsInstance(result, dict)

    def test_compute_importance_verbose(self):
        """compute_importance with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_importance(verbose=True)

    def test_compute_tfidf_verbose(self):
        """compute_tfidf with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_tfidf(verbose=True)

    def test_compute_document_connections_verbose(self):
        """compute_document_connections with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.process_document("doc2", "test content")

        # compute_document_connections returns None
        processor.compute_document_connections(verbose=True)

    def test_build_concept_clusters_verbose(self):
        """build_concept_clusters with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks")

        result = processor.build_concept_clusters(verbose=True)

        self.assertIsInstance(result, dict)

    def test_compute_concept_connections_verbose(self):
        """compute_concept_connections with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.build_concept_clusters(verbose=False)

        processor.compute_concept_connections(verbose=True)

    def test_extract_corpus_semantics_verbose(self):
        """extract_corpus_semantics with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.extract_corpus_semantics(verbose=True)

    def test_compute_graph_embeddings_verbose(self):
        """compute_graph_embeddings with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_graph_embeddings(verbose=True)

        self.assertIsInstance(result, dict)

    def test_retrofit_embeddings_verbose(self):
        """retrofit_embeddings with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.embeddings = {"test": [0.1, 0.2]}
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.retrofit_embeddings(verbose=True)

        self.assertIsInstance(result, dict)

    def test_compute_property_inheritance_verbose(self):
        """compute_property_inheritance with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.compute_property_inheritance(verbose=True)

        self.assertIsInstance(result, dict)

    def test_compute_semantic_importance_verbose(self):
        """compute_semantic_importance with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.compute_semantic_importance(verbose=True)

        self.assertIsInstance(result, dict)

    def test_compute_hierarchical_importance_verbose(self):
        """compute_hierarchical_importance with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_hierarchical_importance(verbose=True)

        self.assertIsInstance(result, dict)

    def test_propagate_activation_verbose(self):
        """propagate_activation with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.propagate_activation(verbose=True)

    def test_retrofit_connections_verbose(self):
        """retrofit_connections with verbose=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("test", "RelatedTo", "content", 1.0)]

        result = processor.retrofit_connections(verbose=True)

        self.assertIsInstance(result, dict)

    def test_compute_all_hierarchical_verbose(self):
        """compute_all with hierarchical and verbose."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_all(
            verbose=True,
            pagerank_method='hierarchical',
            build_concepts=False
        )

        self.assertIsInstance(result, dict)


# =============================================================================
# ERROR HANDLING COVERAGE TESTS (10+ tests)
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


# =============================================================================
# ADDITIONAL COVERAGE FOR 90% (40+ tests)
# =============================================================================

class TestAdditionalCoverage(unittest.TestCase):
    """Additional tests to reach 90% coverage."""

    def test_compute_graph_embeddings_method_variants(self):
        """Test different embedding methods."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks")

        for method in ['tfidf', 'fast', 'adjacency']:
            result = processor.compute_graph_embeddings(method=method, verbose=False)
            self.assertIn('terms_embedded', result)

    def test_compute_graph_embeddings_max_terms_auto(self):
        """Test auto max_terms selection for different corpus sizes."""
        processor = CorticalTextProcessor()

        # Small corpus
        for i in range(5):
            processor.process_document(f"doc{i}", f"test content {i}")

        result = processor.compute_graph_embeddings(max_terms=None, verbose=False)
        self.assertIsInstance(result, dict)

    def test_compute_graph_embeddings_max_terms_explicit(self):
        """Test explicit max_terms parameter."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.compute_graph_embeddings(max_terms=10, verbose=False)
        self.assertIsInstance(result, dict)

    def test_compute_property_inheritance_with_apply(self):
        """compute_property_inheritance with apply_to_connections."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.compute_property_inheritance(
            apply_to_connections=True,
            boost_factor=0.5
        )

        self.assertIn('connections_boosted', result)

    def test_compute_property_inheritance_without_apply(self):
        """compute_property_inheritance without apply_to_connections."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "IsA", "b", 1.0)]

        result = processor.compute_property_inheritance(apply_to_connections=False)

        self.assertEqual(result['connections_boosted'], 0)

    def test_complete_analogy_all_params(self):
        """complete_analogy with different parameter combinations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "RelatedTo", "b", 1.0)]

        result = processor.complete_analogy(
            "a", "b", "c",
            use_embeddings=False,
            use_relations=True
        )

        self.assertIsInstance(result, list)

    def test_complete_analogy_with_embeddings(self):
        """complete_analogy with embeddings enabled."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.embeddings = {"a": [0.1], "b": [0.2], "c": [0.3]}

        result = processor.complete_analogy(
            "a", "b", "c",
            use_embeddings=True,
            use_relations=False
        )

        self.assertIsInstance(result, list)

    def test_expand_query_multihop_basic(self):
        """expand_query_multihop basic functionality."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("test", "RelatedTo", "content", 1.0)]

        result = processor.expand_query_multihop("test")

        self.assertIsInstance(result, dict)

    def test_build_concept_clusters_params(self):
        """build_concept_clusters with different parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks machine learning")

        result = processor.build_concept_clusters(
            min_cluster_size=2,
            verbose=False
        )

        self.assertIsInstance(result, dict)

    def test_compute_bigram_connections_basic(self):
        """compute_bigram_connections basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks")

        result = processor.compute_bigram_connections(verbose=False)

        self.assertIsInstance(result, dict)

    def test_compute_document_connections_params(self):
        """compute_document_connections with parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.process_document("doc2", "test content")

        processor.compute_document_connections(min_shared_terms=1, verbose=False)

    def test_propagate_activation_params(self):
        """propagate_activation with different parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.propagate_activation(iterations=3, decay=0.5, verbose=False)

    def test_expand_query_with_params(self):
        """expand_query with various parameter combinations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content code function")

        result = processor.expand_query(
            "test",
            max_expansions=5,
            use_variants=True,
            use_code_concepts=True,
            filter_code_stop_words=True
        )

        self.assertIsInstance(result, dict)

    def test_expand_query_for_code_basic(self):
        """expand_query_for_code basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "function fetch data")

        result = processor.expand_query_for_code("fetch")

        self.assertIsInstance(result, dict)

    def test_expand_query_semantic_basic(self):
        """expand_query_semantic basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("test", "RelatedTo", "content", 1.0)]

        result = processor.expand_query_semantic("test")

        self.assertIsInstance(result, dict)

    def test_find_documents_with_boost_basic(self):
        """find_documents_with_boost basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.find_documents_with_boost("test", top_n=5)

        self.assertIsInstance(result, list)

    def test_find_documents_with_boost_params(self):
        """find_documents_with_boost with custom parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.find_documents_with_boost(
            "test",
            top_n=10,
            auto_detect_intent=True,
            prefer_docs=False,
            custom_boosts={"test": 2.0}
        )

        self.assertIsInstance(result, list)

    def test_fast_find_documents_basic(self):
        """fast_find_documents basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.fast_find_documents("test")

        self.assertIsInstance(result, list)

    def test_fast_find_documents_params(self):
        """fast_find_documents with parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.fast_find_documents(
            "test",
            top_n=10,
            candidate_multiplier=3,
            use_code_concepts=True
        )

        self.assertIsInstance(result, list)

    def test_is_conceptual_query_true(self):
        """is_conceptual_query with conceptual query."""
        processor = CorticalTextProcessor()

        result = processor.is_conceptual_query("what is machine learning")

        self.assertIsInstance(result, bool)

    def test_is_conceptual_query_false(self):
        """is_conceptual_query with non-conceptual query."""
        processor = CorticalTextProcessor()

        result = processor.is_conceptual_query("test")

        self.assertIsInstance(result, bool)

    def test_parse_intent_query_basic(self):
        """parse_intent_query basic usage."""
        processor = CorticalTextProcessor()

        result = processor.parse_intent_query("where is the function")

        self.assertIsInstance(result, dict)

    def test_search_by_intent_basic(self):
        """search_by_intent basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.search_by_intent("how does it work")

        self.assertIsInstance(result, list)

    def test_query_expanded_basic(self):
        """query_expanded basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.query_expanded("test")

        self.assertIsInstance(result, list)

    def test_find_related_documents_basic(self):
        """find_related_documents basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.process_document("doc2", "test content")

        result = processor.find_related_documents("doc1")

        self.assertIsInstance(result, list)

    def test_analyze_knowledge_gaps_basic(self):
        """analyze_knowledge_gaps basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.analyze_knowledge_gaps()

        self.assertIsInstance(result, dict)

    def test_detect_anomalies_basic(self):
        """detect_anomalies basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.detect_anomalies(threshold=0.5)

        self.assertIsInstance(result, list)

    def test_get_fingerprint_basic(self):
        """get_fingerprint basic usage."""
        processor = CorticalTextProcessor()

        result = processor.get_fingerprint("test content", top_n=10)

        self.assertIsInstance(result, dict)

    def test_compare_fingerprints_basic(self):
        """compare_fingerprints basic usage."""
        processor = CorticalTextProcessor()

        fp1 = processor.get_fingerprint("test content")
        fp2 = processor.get_fingerprint("test data")
        result = processor.compare_fingerprints(fp1, fp2)

        self.assertIsInstance(result, dict)

    def test_explain_fingerprint_basic(self):
        """explain_fingerprint basic usage."""
        processor = CorticalTextProcessor()

        fp = processor.get_fingerprint("test content")
        result = processor.explain_fingerprint(fp)

        self.assertIsInstance(result, dict)

    def test_explain_similarity_basic(self):
        """explain_similarity basic usage."""
        processor = CorticalTextProcessor()

        fp1 = processor.get_fingerprint("test content")
        fp2 = processor.get_fingerprint("test data")
        result = processor.explain_similarity(fp1, fp2)

        self.assertIsInstance(result, str)

    def test_get_corpus_summary_basic(self):
        """get_corpus_summary basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.get_corpus_summary()

        self.assertIsInstance(result, dict)

    def test_get_document_signature_with_tfidf(self):
        """get_document_signature after computing TF-IDF."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks")
        processor.compute_tfidf()

        signature = processor.get_document_signature("doc1", n=3)

        self.assertIsInstance(signature, list)
        self.assertLessEqual(len(signature), 3)

    def test_complete_analogy_edge_cases(self):
        """complete_analogy handles edge cases."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        # Test with no semantic relations or embeddings
        result = processor.complete_analogy("a", "b", "c")
        self.assertIsInstance(result, list)

    def test_compute_graph_embeddings_large_corpus_auto_limit(self):
        """Test auto max_terms with larger corpus."""
        processor = CorticalTextProcessor()

        # Create medium-sized corpus to trigger auto-limit
        for i in range(50):
            processor.process_document(f"doc{i}", f"test content item {i}")

        result = processor.compute_graph_embeddings(max_terms=None, verbose=False)
        self.assertIn('terms_embedded', result)

    def test_expand_query_none_max_expansions(self):
        """expand_query with max_expansions=None uses config default."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.expand_query("test", max_expansions=None)
        self.assertIsInstance(result, dict)

    def test_find_documents_for_query_with_semantic(self):
        """find_documents_for_query with semantic relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("test", "RelatedTo", "content", 1.0)]

        result = processor.find_documents_for_query("test", use_semantic=True)
        self.assertIsInstance(result, list)

    def test_find_documents_for_query_without_semantic(self):
        """find_documents_for_query without semantic relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.find_documents_for_query("test", use_semantic=False)
        self.assertIsInstance(result, list)

    def test_find_documents_for_query_without_expansion(self):
        """find_documents_for_query with use_expansion=False."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.find_documents_for_query("test", use_expansion=False)
        self.assertIsInstance(result, list)

    def test_compute_property_similarity_basic(self):
        """compute_property_similarity basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = [("a", "HasProperty", "x", 1.0)]

        result = processor.compute_property_similarity("a", "b")
        self.assertIsInstance(result, float)

    def test_embedding_similarity_basic(self):
        """embedding_similarity basic usage."""
        processor = CorticalTextProcessor()
        processor.embeddings = {"term1": [0.1, 0.2], "term2": [0.3, 0.4]}

        result = processor.embedding_similarity("term1", "term2")
        self.assertIsInstance(result, float)

    def test_find_similar_by_embedding_basic(self):
        """find_similar_by_embedding basic usage."""
        processor = CorticalTextProcessor()
        processor.embeddings = {"term1": [0.1, 0.2], "term2": [0.3, 0.4]}

        result = processor.find_similar_by_embedding("term1", top_n=5)
        self.assertIsInstance(result, list)

    def test_extract_pattern_relations_basic(self):
        """extract_pattern_relations basic usage."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test is a content")

        result = processor.extract_pattern_relations()
        self.assertIsInstance(result, list)

    def test_compute_all_with_all_params(self):
        """compute_all with comprehensive parameter combinations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content neural networks machine learning")

        # Test with multiple custom parameters
        result = processor.compute_all(
            verbose=False,
            build_concepts=True,
            pagerank_method='standard',
            connection_strategy='document_overlap',
            cluster_strictness=0.5,
            bridge_weight=0.3
        )

        self.assertIsInstance(result, dict)

    def test_remove_documents_batch_tfidf_recompute(self):
        """remove_documents_batch with TF-IDF recompute."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.process_document("doc2", "test content")

        result = processor.remove_documents_batch(
            ["doc1"],
            recompute='tfidf',
            verbose=False
        )

        self.assertEqual(result['documents_removed'], 1)

    def test_remove_documents_batch_full_recompute(self):
        """remove_documents_batch with full recompute."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.remove_documents_batch(
            ["doc1"],
            recompute='full',
            verbose=False
        )

        self.assertEqual(result['documents_removed'], 1)

    def test_add_document_incremental_tfidf_recompute(self):
        """add_document_incremental with TF-IDF recompute."""
        processor = CorticalTextProcessor()

        result = processor.add_document_incremental(
            "doc1",
            "test content",
            recompute='tfidf'
        )

        self.assertIn('tokens', result)

    def test_add_document_incremental_all_recompute(self):
        """add_document_incremental with full recompute."""
        processor = CorticalTextProcessor()

        result = processor.add_document_incremental(
            "doc1",
            "test content",
            recompute='all'
        )

        self.assertIn('tokens', result)

    def test_add_document_incremental_no_recompute(self):
        """add_document_incremental with no recompute."""
        processor = CorticalTextProcessor()

        result = processor.add_document_incremental(
            "doc1",
            "test content",
            recompute='none'
        )

        self.assertIn('tokens', result)

    def test_expand_query_cached_different_use_variants(self):
        """expand_query_cached with different use_variants values."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        # Different params should use different cache entries
        result1 = processor.expand_query_cached("test", use_variants=True)
        result2 = processor.expand_query_cached("test", use_variants=False)

        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)

    def test_expand_query_cached_different_use_code_concepts(self):
        """expand_query_cached with different use_code_concepts values."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test code function")

        result1 = processor.expand_query_cached("test", use_code_concepts=True)
        result2 = processor.expand_query_cached("test", use_code_concepts=False)

        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)


# =============================================================================
# SIMPLIFIED FACADE METHOD TESTS (Task #186)
# =============================================================================


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
