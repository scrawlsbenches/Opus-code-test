"""Tests for the embeddings module."""

import unittest
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.embeddings import (
    compute_graph_embeddings,
    embedding_similarity,
    find_similar_by_embedding,
    _adjacency_embeddings,
    _random_walk_embeddings,
    _spectral_embeddings
)


class TestEmbeddings(unittest.TestCase):
    """Test the embeddings module."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with sample data."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", """
            Neural networks process information through layers.
            Deep learning enables pattern recognition.
        """)
        cls.processor.process_document("doc2", """
            Machine learning algorithms learn from data.
            Training neural networks requires optimization.
        """)
        cls.processor.process_document("doc3", """
            Graph algorithms traverse nodes and edges.
            Network analysis reveals structure.
        """)
        cls.processor.compute_all(verbose=False)

    def test_compute_graph_embeddings_adjacency(self):
        """Test adjacency-based embeddings."""
        embeddings, stats = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )
        self.assertIsInstance(embeddings, dict)
        self.assertGreater(len(embeddings), 0)
        self.assertEqual(stats['method'], 'adjacency')
        self.assertEqual(stats['dimensions'], 16)
        self.assertEqual(stats['terms_embedded'], len(embeddings))

    def test_compute_graph_embeddings_random_walk(self):
        """Test random walk embeddings."""
        embeddings, stats = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='random_walk'
        )
        self.assertIsInstance(embeddings, dict)
        self.assertGreater(len(embeddings), 0)
        self.assertEqual(stats['method'], 'random_walk')

    def test_compute_graph_embeddings_spectral(self):
        """Test spectral embeddings."""
        embeddings, stats = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='spectral'
        )
        self.assertIsInstance(embeddings, dict)
        self.assertGreater(len(embeddings), 0)
        self.assertEqual(stats['method'], 'spectral')

    def test_compute_graph_embeddings_invalid_method(self):
        """Test that invalid method raises error."""
        with self.assertRaises(ValueError):
            compute_graph_embeddings(
                self.processor.layers,
                dimensions=16,
                method='invalid'
            )

    def test_embedding_similarity(self):
        """Test cosine similarity between embeddings."""
        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        # Find two terms that exist in embeddings
        terms = list(embeddings.keys())
        if len(terms) >= 2:
            sim = embedding_similarity(embeddings, terms[0], terms[1])
            self.assertIsInstance(sim, float)
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)

    def test_embedding_similarity_self(self):
        """Test that a term has similarity 1.0 with itself."""
        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        terms = list(embeddings.keys())
        if terms:
            sim = embedding_similarity(embeddings, terms[0], terms[0])
            self.assertAlmostEqual(sim, 1.0, places=5)

    def test_embedding_similarity_missing_term(self):
        """Test similarity with missing term returns 0."""
        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        sim = embedding_similarity(embeddings, "nonexistent_term", "another_missing")
        self.assertEqual(sim, 0.0)

    def test_find_similar_by_embedding(self):
        """Test finding similar terms by embedding."""
        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        terms = list(embeddings.keys())
        if terms:
            similar = find_similar_by_embedding(embeddings, terms[0], top_n=5)
            self.assertIsInstance(similar, list)
            self.assertLessEqual(len(similar), 5)

            # Check format of results
            for term, score in similar:
                self.assertIsInstance(term, str)
                self.assertIsInstance(score, float)

    def test_find_similar_by_embedding_missing_term(self):
        """Test finding similar for missing term returns empty list."""
        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        similar = find_similar_by_embedding(embeddings, "nonexistent_term", top_n=5)
        self.assertEqual(similar, [])

    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        dimensions = 16
        embeddings, stats = compute_graph_embeddings(
            self.processor.layers,
            dimensions=dimensions,
            method='adjacency'
        )

        # Dimensions are min(requested, num_terms)
        expected_dims = min(dimensions, stats['terms_embedded'])
        for term, vec in embeddings.items():
            self.assertEqual(len(vec), expected_dims)

    def test_embedding_normalization(self):
        """Test that adjacency embeddings are normalized."""
        import math

        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        for term, vec in embeddings.items():
            magnitude = math.sqrt(sum(v * v for v in vec))
            # Should be approximately 1.0 (normalized)
            self.assertAlmostEqual(magnitude, 1.0, places=5)


class TestEmbeddingsEmptyLayer(unittest.TestCase):
    """Test embeddings with empty layer."""

    def test_empty_layer_embeddings(self):
        """Test embeddings on empty processor."""
        processor = CorticalTextProcessor()
        embeddings, stats = compute_graph_embeddings(
            processor.layers,
            dimensions=16,
            method='adjacency'
        )
        self.assertEqual(len(embeddings), 0)
        self.assertEqual(stats['terms_embedded'], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
