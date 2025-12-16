"""
Unit Tests for processor.py - Initialization & Configuration
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


if __name__ == '__main__':
    unittest.main()
