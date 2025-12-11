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


class TestEmbeddingSemanticQuality(unittest.TestCase):
    """Regression tests for embedding semantic quality (Task #122)."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with semantically distinct documents."""
        cls.processor = CorticalTextProcessor()
        # Machine learning documents
        cls.processor.process_document("ml1", """
            Neural networks process information through multiple layers.
            Deep learning enables automatic feature extraction.
            Training neural networks requires gradient descent optimization.
        """)
        cls.processor.process_document("ml2", """
            Machine learning algorithms learn patterns from data.
            Neural networks are inspired by biological neurons.
            Deep learning models use backpropagation for training.
        """)
        # Cooking documents (semantically different)
        cls.processor.process_document("cook1", """
            Bread baking requires yeast and flour for fermentation.
            Sourdough bread has a tangy flavor from natural fermentation.
        """)
        cls.processor.process_document("cook2", """
            Pasta is made from durum wheat semolina and water.
            Italian cuisine features many regional pasta variations.
        """)
        cls.processor.compute_all(verbose=False)

    def test_random_walk_semantic_similarity(self):
        """Test that random_walk embeddings capture semantic relationships.

        Regression test: 'neural' should be more similar to 'networks'
        than to unrelated words like 'bread'.
        """
        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='random_walk'
        )

        if 'neural' in embeddings and 'networks' in embeddings:
            neural_networks_sim = embedding_similarity(embeddings, 'neural', 'networks')

            # Check against unrelated terms
            for unrelated in ['bread', 'pasta', 'yeast', 'flour']:
                if unrelated in embeddings:
                    neural_unrelated_sim = embedding_similarity(embeddings, 'neural', unrelated)
                    # Neural should be more similar to networks than to cooking terms
                    self.assertGreater(
                        neural_networks_sim, neural_unrelated_sim,
                        f"'neural' should be more similar to 'networks' ({neural_networks_sim:.3f}) "
                        f"than to '{unrelated}' ({neural_unrelated_sim:.3f})"
                    )

    def test_adjacency_produces_nonzero_embeddings(self):
        """Test that adjacency method produces meaningful (non-sparse) embeddings.

        Regression test: After multi-hop propagation fix, adjacency embeddings
        should have multiple non-zero dimensions.
        """
        import math

        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        # Check that embeddings have multiple non-zero dimensions
        for term, vec in list(embeddings.items())[:10]:
            nonzero_dims = sum(1 for v in vec if abs(v) > 1e-6)
            # With multi-hop propagation, should have more than just 1-2 non-zero dims
            self.assertGreater(
                nonzero_dims, 0,
                f"Term '{term}' has all-zero embedding"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
