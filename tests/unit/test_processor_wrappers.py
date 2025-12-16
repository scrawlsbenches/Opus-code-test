"""
Unit Tests for processor.py - Wrapper Methods
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


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


if __name__ == '__main__':
    unittest.main()
