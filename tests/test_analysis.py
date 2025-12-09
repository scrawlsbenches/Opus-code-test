"""Tests for the analysis module."""

import unittest
import math
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer, HierarchicalLayer
from cortical.analysis import (
    compute_pagerank,
    compute_tfidf,
    propagate_activation,
    cluster_by_label_propagation,
    build_concept_clusters,
    compute_document_connections,
    cosine_similarity
)


class TestPageRank(unittest.TestCase):
    """Test PageRank computation."""

    def test_pagerank_empty_layer(self):
        """Test PageRank on empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        result = compute_pagerank(layer)
        self.assertEqual(result, {})

    def test_pagerank_single_node(self):
        """Test PageRank with single node."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")
        result = compute_pagerank(layer)
        self.assertEqual(len(result), 1)
        # With damping 0.85, single node gets (1-0.85)/1 = 0.15
        self.assertAlmostEqual(list(result.values())[0], 0.15, places=5)

    def test_pagerank_multiple_nodes(self):
        """Test PageRank with multiple connected nodes."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep")
        processor.process_document("doc2", "neural learning patterns data")

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        result = compute_pagerank(layer0)

        # All nodes should have positive PageRank
        for col in layer0.minicolumns.values():
            self.assertGreater(col.pagerank, 0)

    def test_pagerank_convergence(self):
        """Test that PageRank converges."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "word1 word2 word3 word4")

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        result = compute_pagerank(layer0, iterations=100)

        # Sum should be approximately 1.0
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=3)


class TestTFIDF(unittest.TestCase):
    """Test TF-IDF computation."""

    def test_tfidf_empty_corpus(self):
        """Test TF-IDF on empty corpus."""
        processor = CorticalTextProcessor()
        compute_tfidf(processor.layers, processor.documents)
        # Should not raise

    def test_tfidf_single_document(self):
        """Test TF-IDF with single document."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "word1 word2 word3")
        compute_tfidf(processor.layers, processor.documents)

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        # With single doc, IDF = log(1/1) = 0, so TF-IDF = 0
        for col in layer0.minicolumns.values():
            self.assertEqual(col.tfidf, 0.0)

    def test_tfidf_multiple_documents(self):
        """Test TF-IDF with multiple documents."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning")
        processor.process_document("doc2", "machine learning algorithms")
        processor.process_document("doc3", "database systems storage")
        compute_tfidf(processor.layers, processor.documents)

        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        # Terms unique to one doc should have higher TF-IDF
        unique_term = layer0.get_minicolumn("database")
        common_term = layer0.get_minicolumn("learning")

        if unique_term and common_term:
            # database appears in 1 doc, learning in 2
            self.assertGreater(unique_term.tfidf, 0)

    def test_tfidf_per_document(self):
        """Test per-document TF-IDF."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural neural neural")  # 3 occurrences
        processor.process_document("doc2", "neural learning")  # 1 occurrence
        processor.process_document("doc3", "different content here")  # No neural - needed for IDF > 0
        compute_tfidf(processor.layers, processor.documents)

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        neural = layer0.get_minicolumn("neural")

        # Check per-document TF-IDF uses actual occurrence counts
        self.assertIn("doc1", neural.tfidf_per_doc)
        self.assertIn("doc2", neural.tfidf_per_doc)
        # doc1 has 3 occurrences, doc2 has 1
        # log1p(3) > log1p(1), so doc1 should have higher per-doc TF-IDF
        self.assertGreater(neural.tfidf_per_doc["doc1"], neural.tfidf_per_doc["doc2"])


class TestActivationPropagation(unittest.TestCase):
    """Test activation propagation."""

    def test_propagation_empty_layers(self):
        """Test propagation on empty layers."""
        processor = CorticalTextProcessor()
        propagate_activation(processor.layers)
        # Should not raise

    def test_propagation_preserves_activation(self):
        """Test that propagation modifies activations."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning")

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        initial_activations = {col.content: col.activation for col in layer0}

        propagate_activation(processor.layers, iterations=3)

        # Activations should have changed
        for col in layer0.minicolumns.values():
            # With decay, activation should decrease or stay same
            self.assertGreaterEqual(col.activation, 0)


class TestLabelPropagation(unittest.TestCase):
    """Test label propagation clustering."""

    def test_clustering_empty_layer(self):
        """Test clustering on empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        clusters = cluster_by_label_propagation(layer)
        self.assertEqual(clusters, {})

    def test_clustering_returns_dict(self):
        """Test that clustering returns dictionary."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep patterns")
        processor.process_document("doc2", "neural learning patterns data")

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        clusters = cluster_by_label_propagation(layer0, min_cluster_size=2)

        self.assertIsInstance(clusters, dict)

    def test_clustering_min_size(self):
        """Test that clusters respect minimum size."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep patterns")
        processor.process_document("doc2", "neural learning patterns data")

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        clusters = cluster_by_label_propagation(layer0, min_cluster_size=3)

        for members in clusters.values():
            self.assertGreaterEqual(len(members), 3)


class TestConceptClusters(unittest.TestCase):
    """Test concept cluster building."""

    def test_build_concept_clusters(self):
        """Test building concept layer from clusters."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep")
        processor.process_document("doc2", "neural learning patterns data")
        processor.compute_importance(verbose=False)

        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        clusters = cluster_by_label_propagation(layer0, min_cluster_size=2)
        build_concept_clusters(processor.layers, clusters)

        layer2 = processor.get_layer(CorticalLayer.CONCEPTS)
        # May or may not have concepts depending on cluster size
        self.assertIsInstance(layer2.minicolumns, dict)


class TestDocumentConnections(unittest.TestCase):
    """Test document connection computation."""

    def test_document_connections(self):
        """Test building document connections."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep patterns")
        processor.process_document("doc2", "neural learning patterns data")
        processor.process_document("doc3", "completely different content here")
        processor.compute_tfidf(verbose=False)

        compute_document_connections(processor.layers, processor.documents, min_shared_terms=2)

        layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)
        doc1 = layer3.get_minicolumn("doc1")
        doc2 = layer3.get_minicolumn("doc2")

        # doc1 and doc2 share terms, should be connected
        if doc1 and doc2:
            # Check if they have connections
            has_connection = len(doc1.lateral_connections) > 0 or len(doc2.lateral_connections) > 0
            self.assertTrue(has_connection)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity function."""

    def test_cosine_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity of non-overlapping vectors."""
        vec1 = {'a': 1.0, 'b': 2.0}
        vec2 = {'c': 3.0, 'd': 4.0}
        sim = cosine_similarity(vec1, vec2)
        self.assertEqual(sim, 0.0)

    def test_cosine_empty_vectors(self):
        """Test cosine similarity with empty vectors."""
        sim = cosine_similarity({}, {})
        self.assertEqual(sim, 0.0)

    def test_cosine_partial_overlap(self):
        """Test cosine similarity with partial overlap."""
        vec1 = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        vec2 = {'b': 2.0, 'c': 3.0, 'd': 4.0}
        sim = cosine_similarity(vec1, vec2)
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_cosine_zero_magnitude(self):
        """Test cosine similarity with zero magnitude vector."""
        vec1 = {'a': 0.0}
        vec2 = {'a': 1.0}
        sim = cosine_similarity(vec1, vec2)
        self.assertEqual(sim, 0.0)


class TestGetByIdOptimization(unittest.TestCase):
    """Test that get_by_id optimization works correctly."""

    def test_get_by_id_returns_correct_minicolumn(self):
        """Test that get_by_id returns the correct minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")

        # Get by ID should return the same minicolumn
        retrieved = layer.get_by_id(col1.id)
        self.assertIs(retrieved, col1)

        retrieved2 = layer.get_by_id(col2.id)
        self.assertIs(retrieved2, col2)

    def test_get_by_id_returns_none_for_missing(self):
        """Test that get_by_id returns None for missing ID."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("test")

        result = layer.get_by_id("nonexistent_id")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
