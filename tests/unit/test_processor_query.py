"""
Unit Tests for processor.py - Query Expansion & Search
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


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


if __name__ == '__main__':
    unittest.main()
