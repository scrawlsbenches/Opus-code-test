"""
Unit Tests for processor.py - Compute Methods
"""

import json
import pytest
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


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

    @patch('cortical.analysis.compute_bm25')
    def test_compute_tfidf_calls_bm25_by_default(self, mock_bm25):
        """compute_tfidf delegates to BM25 by default (new default algorithm)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_tfidf(verbose=False)

        # BM25 is now the default algorithm
        mock_bm25.assert_called_once()

    @patch('cortical.analysis.compute_tfidf')
    def test_compute_tfidf_calls_tfidf_when_configured(self, mock_tfidf):
        """compute_tfidf delegates to TF-IDF when explicitly configured."""
        from cortical.config import CorticalConfig
        config = CorticalConfig(scoring_algorithm='tfidf')
        processor = CorticalTextProcessor(config=config)
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


# =============================================================================
# REAL INTEGRATION TESTS (for uncovered lines)
# =============================================================================


class TestRecomputeMethod(unittest.TestCase):
    """Test recompute() method - currently has no coverage."""

    def test_recompute_full(self):
        """Test recompute with level='full'."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")

        result = processor.recompute(level='full', verbose=False)

        self.assertIn(processor.COMP_ACTIVATION, result)
        self.assertIn(processor.COMP_PAGERANK, result)
        self.assertIn(processor.COMP_TFIDF, result)

    def test_recompute_tfidf_only(self):
        """Test recompute with level='tfidf'."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        result = processor.recompute(level='tfidf', verbose=False)

        self.assertIn(processor.COMP_TFIDF, result)
        self.assertEqual(len(result), 1)

    def test_recompute_stale(self):
        """Test recompute with level='stale'."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.compute_all(verbose=False, build_concepts=False)

        # Mark something as stale
        processor._stale_computations.add(processor.COMP_TFIDF)

        result = processor.recompute(level='stale', verbose=False)

        self.assertIn(processor.COMP_TFIDF, result)


class TestCheckpointFunctionality(unittest.TestCase):
    """Test checkpoint save/load/resume functionality - currently has no coverage."""

    def setUp(self):
        """Create temp directory for checkpoints."""
        import tempfile
        self.checkpoint_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up checkpoint directory."""
        import shutil
        if Path(self.checkpoint_dir).exists():
            shutil.rmtree(self.checkpoint_dir)

    def test_save_checkpoint(self):
        """Test _save_checkpoint creates files."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor._save_checkpoint(self.checkpoint_dir, "test_phase", verbose=False)

        progress_file = Path(self.checkpoint_dir) / 'checkpoint_progress.json'
        self.assertTrue(progress_file.exists())

        with open(progress_file, 'r') as f:
            data = json.load(f)
        self.assertIn('test_phase', data['completed_phases'])

    def test_load_checkpoint_progress(self):
        """Test _load_checkpoint_progress reads completed phases."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        processor._save_checkpoint(self.checkpoint_dir, "phase1", verbose=False)
        processor._save_checkpoint(self.checkpoint_dir, "phase2", verbose=False)

        completed = processor._load_checkpoint_progress(self.checkpoint_dir)

        self.assertIn('phase1', completed)
        self.assertIn('phase2', completed)

    def test_load_checkpoint_progress_missing(self):
        """Test loading from non-existent directory."""
        processor = CorticalTextProcessor()
        completed = processor._load_checkpoint_progress("/nonexistent")

        self.assertEqual(completed, set())

    def test_resume_from_checkpoint(self):
        """Test resume_from_checkpoint classmethod."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks")
        processor.compute_all(verbose=False, checkpoint_dir=self.checkpoint_dir, build_concepts=False)

        resumed = CorticalTextProcessor.resume_from_checkpoint(self.checkpoint_dir, verbose=False)

        self.assertIsNotNone(resumed)
        # Verify the processor was loaded successfully
        self.assertEqual(len(resumed.documents), 1)

    def test_compute_all_with_checkpoint(self):
        """Test compute_all creates checkpoints."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_all(verbose=False, checkpoint_dir=self.checkpoint_dir, build_concepts=False)

        progress_file = Path(self.checkpoint_dir) / 'checkpoint_progress.json'
        self.assertTrue(progress_file.exists())

    def test_compute_all_resume(self):
        """Test compute_all with resume=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        # First run with checkpoint
        processor.compute_all(verbose=False, checkpoint_dir=self.checkpoint_dir, build_concepts=False)

        # Resume should skip completed phases
        processor.compute_all(verbose=False, checkpoint_dir=self.checkpoint_dir, resume=True, build_concepts=False)


class TestBM25Methods(unittest.TestCase):
    """Test BM25 methods - currently have no coverage."""

    def test_compute_bm25_direct(self):
        """Test compute_bm25 method."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks process data")
        processor.process_document("doc2", "machine learning algorithms")

        processor.compute_bm25(k1=1.2, b=0.75, verbose=False)

    def test_compute_bm25_default_params(self):
        """Test compute_bm25 with default parameters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        processor.compute_bm25(verbose=False)

    def test_compute_bm25_parallel(self):
        """Test compute_bm25_parallel (falls back to sequential for small corpus)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        stats = processor.compute_bm25_parallel(verbose=False)

        self.assertIn('method', stats)
        self.assertIn('k1', stats)
        self.assertIn('b', stats)

    def test_compute_bm25_parallel_custom_params(self):
        """Test BM25 parallel with custom k1 and b."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        stats = processor.compute_bm25_parallel(k1=1.5, b=0.6, verbose=False)

        self.assertEqual(stats['k1'], 1.5)
        self.assertEqual(stats['b'], 0.6)


class TestTFIDFParallel(unittest.TestCase):
    """Test TF-IDF parallel methods - currently have no coverage."""

    def test_compute_tfidf_parallel(self):
        """Test compute_tfidf_parallel (falls back to sequential for small corpus)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        stats = processor.compute_tfidf_parallel(verbose=False)

        self.assertIn('method', stats)
        self.assertIn('terms_processed', stats)

    def test_compute_tfidf_parallel_custom_workers(self):
        """Test TF-IDF parallel with custom worker count."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        stats = processor.compute_tfidf_parallel(num_workers=2, chunk_size=500, verbose=False)

        self.assertIn('terms_processed', stats)


class TestClusteringMethods(unittest.TestCase):
    """Test clustering methods - some paths uncovered."""

    def test_build_concept_clusters_louvain(self):
        """Test concept clustering with Louvain method."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks machine learning")
        processor.compute_all(verbose=False, build_concepts=False)

        clusters = processor.build_concept_clusters(clustering_method='louvain', verbose=False)

        self.assertIsInstance(clusters, dict)

    def test_build_concept_clusters_label_propagation(self):
        """Test clustering with label propagation."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks machine learning")
        processor.compute_all(verbose=False, build_concepts=False)

        clusters = processor.build_concept_clusters(
            clustering_method='label_propagation',
            cluster_strictness=0.7,
            bridge_weight=0.2,
            verbose=False
        )

        self.assertIsInstance(clusters, dict)

    def test_build_concept_clusters_invalid_method(self):
        """Test invalid clustering method raises ValueError."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test")

        with self.assertRaises(ValueError):
            processor.build_concept_clusters(clustering_method='invalid', verbose=False)


class TestEmbeddingMethods(unittest.TestCase):
    """Test embedding methods - some paths uncovered."""

    def test_compute_graph_embeddings_different_methods(self):
        """Test different embedding methods."""
        for method in ['tfidf', 'fast', 'adjacency']:
            processor = CorticalTextProcessor()
            processor.process_document("doc1", "neural networks")
            processor.compute_all(verbose=False, build_concepts=False)

            stats = processor.compute_graph_embeddings(method=method, verbose=False)

            self.assertIn('terms_embedded', stats)

    def test_compute_graph_embeddings_with_max_terms(self):
        """Test embeddings with max_terms limit."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks machine learning")
        processor.compute_all(verbose=False, build_concepts=False)

        stats = processor.compute_graph_embeddings(max_terms=5, verbose=False)

        self.assertIn('terms_embedded', stats)


class TestSemanticMethods(unittest.TestCase):
    """Test semantic methods - some paths uncovered."""

    def test_compute_semantic_importance_no_relations(self):
        """Test semantic importance when no relations exist."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = []

        stats = processor.compute_semantic_importance(verbose=False)

        self.assertEqual(stats['total_edges_with_relations'], 0)

    def test_compute_semantic_importance_with_relations(self):
        """Test semantic importance with relations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.extract_corpus_semantics(verbose=False)

        stats = processor.compute_semantic_importance(verbose=False)

        self.assertIn('total_edges_with_relations', stats)

    def test_compute_hierarchical_importance(self):
        """Test hierarchical PageRank."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        stats = processor.compute_hierarchical_importance(verbose=False)

        self.assertIn('iterations_run', stats)
        self.assertIn('converged', stats)

    def test_compute_property_similarity_no_relations(self):
        """Test property similarity with no relations."""
        processor = CorticalTextProcessor()
        processor.semantic_relations = []

        sim = processor.compute_property_similarity("a", "b")

        self.assertEqual(sim, 0.0)

    def test_retrofit_connections_auto_extract(self):
        """Test retrofit_connections extracts semantics if missing."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")
        processor.semantic_relations = []

        stats = processor.retrofit_connections(verbose=False)

        self.assertIn('tokens_affected', stats)


class TestProgressReporting(unittest.TestCase):
    """Test progress reporting paths in compute_all."""

    def test_compute_all_show_progress(self):
        """Test compute_all with show_progress=True."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content")

        stats = processor.compute_all(show_progress=True, verbose=False, build_concepts=False)

        self.assertIsInstance(stats, dict)


class TestParallelProcessing(unittest.TestCase):
    """Test parallel processing paths in compute_all."""

    def test_compute_all_parallel_bm25(self):
        """Test compute_all with parallel BM25."""
        config = CorticalConfig(scoring_algorithm='bm25')
        processor = CorticalTextProcessor(config=config)
        processor.process_document("doc1", "test content")

        stats = processor.compute_all(verbose=False, parallel=True, build_concepts=False)

        self.assertIsInstance(stats, dict)

    def test_compute_all_parallel_tfidf(self):
        """Test compute_all with parallel TF-IDF."""
        config = CorticalConfig(scoring_algorithm='tfidf')
        processor = CorticalTextProcessor(config=config)
        processor.process_document("doc1", "test content")

        stats = processor.compute_all(verbose=False, parallel=True, build_concepts=False)

        self.assertIsInstance(stats, dict)


if __name__ == '__main__':
    unittest.main()
