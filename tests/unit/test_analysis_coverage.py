"""
Additional coverage tests for cortical/analysis.py

This module fills coverage gaps in analysis.py, focusing on:
- BM25 scoring functions
- Edge cases in connection limits
- Boundary conditions
- Error handling paths

These tests complement the existing tests in test_analysis.py.
"""

import unittest
import math
from cortical.analysis import (
    _bm25_core,
    compute_bm25,
    compute_bigram_connections,
    compute_concept_connections,
    _doc_similarity,
    _vector_similarity,
    _compute_cluster_balance,
    _generate_quality_assessment,
    cosine_similarity,
)
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn


class TestBM25Core(unittest.TestCase):
    """Test the _bm25_core function for BM25 scoring."""

    def test_empty_corpus(self):
        """BM25 with zero documents returns empty dict."""
        term_stats = {"term1": (10, 1, {"doc1": 10})}
        doc_lengths = {"doc1": 100}
        result = _bm25_core(term_stats, num_docs=0, doc_lengths=doc_lengths, avg_doc_length=100)
        self.assertEqual(result, {})

    def test_zero_avg_doc_length(self):
        """BM25 with zero average document length returns empty dict."""
        term_stats = {"term1": (10, 1, {"doc1": 10})}
        doc_lengths = {"doc1": 100}
        result = _bm25_core(term_stats, num_docs=10, doc_lengths=doc_lengths, avg_doc_length=0)
        self.assertEqual(result, {})

    def test_basic_bm25_calculation(self):
        """BM25 calculates scores for terms."""
        term_stats = {
            "rare": (5, 1, {"doc1": 5}),
            "common": (100, 10, {"doc1": 10, "doc2": 10, "doc3": 10, "doc4": 10, "doc5": 10,
                                  "doc6": 10, "doc7": 10, "doc8": 10, "doc9": 10, "doc10": 10})
        }
        doc_lengths = {f"doc{i}": 100 for i in range(1, 11)}
        result = _bm25_core(term_stats, num_docs=10, doc_lengths=doc_lengths, avg_doc_length=100)

        # Rare term should have higher global BM25 than common term
        self.assertIn("rare", result)
        self.assertIn("common", result)
        self.assertGreater(result["rare"][0], result["common"][0])

    def test_bm25_per_doc_scores(self):
        """BM25 calculates per-document scores."""
        term_stats = {"term1": (15, 3, {"doc1": 5, "doc2": 5, "doc3": 5})}
        doc_lengths = {"doc1": 100, "doc2": 150, "doc3": 50}
        result = _bm25_core(term_stats, num_docs=10, doc_lengths=doc_lengths, avg_doc_length=100)

        global_score, per_doc = result["term1"]
        self.assertIsInstance(global_score, float)
        self.assertEqual(len(per_doc), 3)
        self.assertIn("doc1", per_doc)
        self.assertIn("doc2", per_doc)
        self.assertIn("doc3", per_doc)

    def test_bm25_length_normalization(self):
        """BM25 normalizes by document length."""
        # Same term frequency but different document lengths
        term_stats = {"term1": (10, 2, {"short": 5, "long": 5})}
        doc_lengths = {"short": 50, "long": 200}
        result = _bm25_core(term_stats, num_docs=5, doc_lengths=doc_lengths, avg_doc_length=100, b=0.75)

        _, per_doc = result["term1"]
        # Shorter doc should score higher (same TF in smaller context is more important)
        self.assertGreater(per_doc["short"], per_doc["long"])

    def test_bm25_no_length_normalization(self):
        """BM25 with b=0 disables length normalization."""
        term_stats = {"term1": (10, 2, {"short": 5, "long": 5})}
        doc_lengths = {"short": 50, "long": 200}
        result = _bm25_core(term_stats, num_docs=5, doc_lengths=doc_lengths, avg_doc_length=100, b=0.0)

        _, per_doc = result["term1"]
        # With b=0, both should have same score (no length normalization)
        self.assertAlmostEqual(per_doc["short"], per_doc["long"], places=5)

    def test_bm25_k1_saturation(self):
        """BM25 k1 parameter controls term frequency saturation."""
        # High TF should saturate with low k1
        term_stats = {"term1": (100, 1, {"doc1": 100})}
        doc_lengths = {"doc1": 200}

        result_low_k1 = _bm25_core(term_stats, num_docs=5, doc_lengths=doc_lengths, avg_doc_length=200, k1=0.5)
        result_high_k1 = _bm25_core(term_stats, num_docs=5, doc_lengths=doc_lengths, avg_doc_length=200, k1=2.0)

        # Higher k1 should give more weight to term frequency
        self.assertGreater(result_high_k1["term1"][0], result_low_k1["term1"][0])

    def test_bm25_zero_doc_frequency(self):
        """BM25 handles zero document frequency edge case."""
        term_stats = {"term1": (10, 0, {})}
        doc_lengths = {"doc1": 100}
        result = _bm25_core(term_stats, num_docs=5, doc_lengths=doc_lengths, avg_doc_length=100)

        # Zero doc frequency should result in zero scores
        self.assertEqual(result["term1"], (0.0, {}))


class TestComputeBM25(unittest.TestCase):
    """Test compute_bm25 function."""

    def test_bm25_basic(self):
        """compute_bm25 calculates BM25 scores."""
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Add minicolumns
        col1 = layer0.get_or_create_minicolumn("term1")
        col1.document_ids = {"doc1", "doc2"}
        col1.occurrence_count = 15
        col1.doc_occurrence_counts = {"doc1": 10, "doc2": 5}

        col2 = layer0.get_or_create_minicolumn("term2")
        col2.document_ids = {"doc1"}
        col2.occurrence_count = 5
        col2.doc_occurrence_counts = {"doc1": 5}

        documents = {"doc1": "text" * 50, "doc2": "text" * 25}  # doc_id: content
        doc_lengths = {"doc1": 100, "doc2": 50}

        # Compute BM25
        compute_bm25(layers, documents, doc_lengths, avg_doc_length=75)

        # Check that tfidf values were computed
        self.assertGreater(col1.tfidf, 0)
        self.assertGreater(col2.tfidf, 0)
        self.assertIn("doc1", col1.tfidf_per_doc)
        self.assertIn("doc2", col1.tfidf_per_doc)

    def test_bm25_empty_corpus(self):
        """compute_bm25 with empty corpus doesn't crash."""
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        documents = {}
        doc_lengths = {}

        # Should handle gracefully
        compute_bm25(layers, documents, doc_lengths, avg_doc_length=0)
        # No assertion - just shouldn't crash

    def test_bm25_with_parameters(self):
        """compute_bm25 respects k1 and b parameters."""
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        col = layer0.get_or_create_minicolumn("term")
        col.document_ids = {"doc1"}
        col.occurrence_count = 10
        col.doc_occurrence_counts = {"doc1": 10}

        documents = {"doc1": "text" * 50}
        doc_lengths = {"doc1": 100}

        # Test with different k1 values
        compute_bm25(layers, documents, doc_lengths, avg_doc_length=100, k1=1.5, b=0.5)
        score1 = col.tfidf

        compute_bm25(layers, documents, doc_lengths, avg_doc_length=100, k1=0.5, b=0.5)
        score2 = col.tfidf

        # Different k1 should produce different scores
        self.assertNotEqual(score1, score2)


class TestBigramConnectionsEdgeCases(unittest.TestCase):
    """Test edge cases in compute_bigram_connections."""

    def test_max_connections_per_bigram_limit_reached(self):
        """Bigrams stop connecting after reaching max_connections_per_bigram."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)
        }
        layer1 = layers[CorticalLayer.BIGRAMS]

        # Create many bigrams that would want to connect
        bigrams = []
        for i in range(10):
            b = layer1.get_or_create_minicolumn(f"bigram{i}")
            b.document_ids = {"doc1"}  # All in same doc
            b.tfidf = 1.0
            bigrams.append(b)

        # Very low max_connections_per_bigram
        compute_bigram_connections(
            layers,
            max_connections_per_bigram=2,  # Very restrictive
            min_shared_docs=1
        )

        # Each bigram should have at most 2 connections
        for b in bigrams:
            self.assertLessEqual(len(b.lateral_connections), 2)

    def test_continue_on_max_connections(self):
        """compute_bigram_connections continues when a bigram hits connection limit."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)
        }
        layer1 = layers[CorticalLayer.BIGRAMS]

        # Create bigrams with shared documents
        b1 = layer1.get_or_create_minicolumn("b1")
        b1.document_ids = {"doc1", "doc2"}
        b1.tfidf = 1.0

        b2 = layer1.get_or_create_minicolumn("b2")
        b2.document_ids = {"doc1", "doc2"}
        b2.tfidf = 1.0

        b3 = layer1.get_or_create_minicolumn("b3")
        b3.document_ids = {"doc1", "doc2"}
        b3.tfidf = 1.0

        # Run with max_connections_per_bigram=1
        compute_bigram_connections(
            layers,
            max_connections_per_bigram=1,
            min_shared_docs=1
        )

        # At least some connections should have been made
        total_connections = sum(len(b.lateral_connections) for b in [b1, b2, b3])
        self.assertGreater(total_connections, 0)


class TestConceptConnectionsEdgeCases(unittest.TestCase):
    """Test edge cases in compute_concept_connections."""

    def test_already_connected_concepts_updated(self):
        """compute_concept_connections updates already connected concept pairs."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS)
        }
        layer2 = layers[CorticalLayer.CONCEPTS]

        c1 = layer2.get_or_create_minicolumn("concept1")
        c1.document_ids = {"doc1", "doc2"}

        c2 = layer2.get_or_create_minicolumn("concept2")
        c2.document_ids = {"doc1", "doc2"}

        # Pre-connect them with initial weight
        initial_weight = 0.5
        c1.add_lateral_connection(c2.id, initial_weight)
        c2.add_lateral_connection(c1.id, initial_weight)

        # Run connection computation
        compute_concept_connections(
            layers,
            min_jaccard=0.1,
            use_member_semantics=False,
            use_embedding_similarity=False
        )

        # Connection weight should be updated (increased)
        self.assertGreater(c1.lateral_connections[c2.id], initial_weight)


class TestHelperFunctionsEdgeCases(unittest.TestCase):
    """Test edge cases in helper functions."""

    def test_doc_similarity_empty_sets(self):
        """_doc_similarity with empty sets returns 0."""
        result = _doc_similarity(frozenset(), frozenset())
        self.assertEqual(result, 0.0)

    def test_doc_similarity_one_empty(self):
        """_doc_similarity with one empty set returns 0."""
        result = _doc_similarity(frozenset({"doc1"}), frozenset())
        self.assertEqual(result, 0.0)

    def test_vector_similarity_empty_vectors(self):
        """_vector_similarity with empty vectors returns 0."""
        result = _vector_similarity({}, {})
        self.assertEqual(result, 0.0)

    def test_vector_similarity_one_empty(self):
        """_vector_similarity with one empty vector returns 0."""
        result = _vector_similarity({"a": 1.0}, {})
        self.assertEqual(result, 0.0)

    def test_vector_similarity_no_overlap(self):
        """_vector_similarity with no overlapping keys returns 0."""
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"c": 1.0, "d": 2.0}
        result = _vector_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)

    def test_compute_cluster_balance_empty_layer(self):
        """_compute_cluster_balance with empty layer returns 1.0."""
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)
        result = _compute_cluster_balance(layer2)
        self.assertEqual(result, 1.0)

    def test_compute_cluster_balance_single_cluster(self):
        """_compute_cluster_balance with one cluster returns 1.0."""
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)
        c1 = layer2.get_or_create_minicolumn("c1")
        result = _compute_cluster_balance(layer2)
        self.assertEqual(result, 1.0)

    def test_compute_cluster_balance_perfect_balance(self):
        """_compute_cluster_balance with equal-sized clusters returns 1.0."""
        layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)
        for i in range(4):
            layer2.get_or_create_minicolumn(f"c{i}")
        result = _compute_cluster_balance(layer2)
        self.assertEqual(result, 1.0)

    def test_generate_quality_assessment_edge_cases(self):
        """_generate_quality_assessment handles edge case values."""
        # Test with extreme values
        assessment = _generate_quality_assessment(
            modularity=0.0,
            silhouette=-1.0,
            balance=1.0,
            num_clusters=1
        )
        # Should generate a valid assessment string
        self.assertIsInstance(assessment, str)
        self.assertGreater(len(assessment), 0)

        # Test with perfect values
        assessment = _generate_quality_assessment(
            modularity=1.0,
            silhouette=1.0,
            balance=1.0,
            num_clusters=10
        )
        # Should generate a valid assessment string
        self.assertIsInstance(assessment, str)
        self.assertGreater(len(assessment), 0)


class TestCosineSimilarityEdgeCases(unittest.TestCase):
    """Test edge cases in cosine_similarity function."""

    def test_cosine_both_zero_magnitude(self):
        """cosine_similarity with both zero vectors returns 0."""
        vec1 = {"a": 0.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 0.0}
        result = cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)

    def test_cosine_first_zero_magnitude(self):
        """cosine_similarity with first vector zero returns 0."""
        vec1 = {"a": 0.0, "b": 0.0}
        vec2 = {"a": 1.0, "b": 1.0}
        result = cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)

    def test_cosine_second_zero_magnitude(self):
        """cosine_similarity with second vector zero returns 0."""
        vec1 = {"a": 1.0, "b": 1.0}
        vec2 = {"a": 0.0, "b": 0.0}
        result = cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)

    def test_cosine_negative_values(self):
        """cosine_similarity handles negative values correctly."""
        vec1 = {"a": 1.0, "b": -1.0}
        vec2 = {"a": -1.0, "b": 1.0}
        result = cosine_similarity(vec1, vec2)
        # Vectors point in opposite directions
        self.assertLess(result, 0.0)


class TestSparseMatrixEdgeCases(unittest.TestCase):
    """Test edge cases in SparseMatrix operations."""

    def test_multiply_transpose_empty_matrix(self):
        """SparseMatrix.multiply_transpose on empty matrix."""
        from cortical.analysis import SparseMatrix

        matrix = SparseMatrix(10, 10)
        result = matrix.multiply_transpose()

        self.assertEqual(result.rows, 10)
        self.assertEqual(result.cols, 10)
        self.assertEqual(len(result.data), 0)

    def test_multiply_transpose_single_entry(self):
        """SparseMatrix.multiply_transpose with single entry."""
        from cortical.analysis import SparseMatrix

        matrix = SparseMatrix(5, 5)
        matrix.set(0, 2, 3.0)
        result = matrix.multiply_transpose()

        # Should produce diagonal entry: 3.0 * 3.0 = 9.0
        self.assertEqual(result.get(2, 2), 9.0)

    def test_set_zero_removes_entry(self):
        """SparseMatrix.set with zero value removes entry."""
        from cortical.analysis import SparseMatrix

        matrix = SparseMatrix(5, 5)
        matrix.set(1, 2, 5.0)
        self.assertEqual(matrix.get(1, 2), 5.0)

        matrix.set(1, 2, 0.0)
        self.assertEqual(matrix.get(1, 2), 0.0)
        self.assertNotIn((1, 2), matrix.data)


class TestBigramConnectionsMinSharedDocs(unittest.TestCase):
    """Test min_shared_docs threshold in compute_bigram_connections."""

    def test_min_shared_docs_blocks_connection(self):
        """Bigrams with too few shared docs don't connect."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS)
        }
        layer1 = layers[CorticalLayer.BIGRAMS]

        # Create bigrams with only 1 shared doc
        b1 = layer1.get_or_create_minicolumn("b1")
        b1.document_ids = {"doc1", "doc2"}
        b1.tfidf = 1.0

        b2 = layer1.get_or_create_minicolumn("b2")
        b2.document_ids = {"doc1", "doc3"}  # Only doc1 in common
        b2.tfidf = 1.0

        # Require min_shared_docs=2
        compute_bigram_connections(
            layers,
            min_shared_docs=2
        )

        # Should not be connected
        self.assertNotIn(b2.id, b1.lateral_connections)
        self.assertNotIn(b1.id, b2.lateral_connections)


if __name__ == '__main__':
    unittest.main()
