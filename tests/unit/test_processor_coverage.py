"""
Unit Tests for processor.py - Coverage Gap Tests
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from cortical.processor import CorticalTextProcessor
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer
from cortical.layers import CorticalLayer, HierarchicalLayer


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



if __name__ == '__main__':
    unittest.main()
