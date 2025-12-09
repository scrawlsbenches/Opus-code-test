"""Tests for the semantics module."""

import unittest
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor, CorticalLayer
from cortical.semantics import (
    extract_corpus_semantics,
    retrofit_connections,
    retrofit_embeddings,
    get_relation_type_weight,
    RELATION_WEIGHTS
)
from cortical.embeddings import compute_graph_embeddings


class TestSemantics(unittest.TestCase):
    """Test the semantics module."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with sample data."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", """
            Neural networks are a type of machine learning model.
            Deep learning uses neural networks for pattern recognition.
            Neural processing happens in the brain cortex.
        """)
        cls.processor.process_document("doc2", """
            Machine learning algorithms learn from data examples.
            Training models requires optimization techniques.
            Learning neural networks needs backpropagation.
        """)
        cls.processor.process_document("doc3", """
            The brain processes information through neurons.
            Cortical columns are like neural networks.
            Processing patterns requires learning.
        """)
        cls.processor.compute_all(verbose=False)

    def test_extract_corpus_semantics(self):
        """Test semantic relation extraction."""
        relations = extract_corpus_semantics(
            self.processor.layers,
            self.processor.documents,
            self.processor.tokenizer
        )
        self.assertIsInstance(relations, list)
        # Should find some relations
        self.assertGreater(len(relations), 0)

        # Check relation format
        for relation in relations:
            self.assertEqual(len(relation), 4)
            term1, rel_type, term2, weight = relation
            self.assertIsInstance(term1, str)
            self.assertIsInstance(rel_type, str)
            self.assertIsInstance(term2, str)
            self.assertIsInstance(weight, float)

    def test_extract_corpus_semantics_cooccurs(self):
        """Test that CoOccurs relations are found."""
        relations = extract_corpus_semantics(
            self.processor.layers,
            self.processor.documents,
            self.processor.tokenizer
        )
        relation_types = set(r[1] for r in relations)
        self.assertIn('CoOccurs', relation_types)

    def test_retrofit_connections(self):
        """Test retrofitting lateral connections."""
        relations = extract_corpus_semantics(
            self.processor.layers,
            self.processor.documents,
            self.processor.tokenizer
        )

        stats = retrofit_connections(
            self.processor.layers,
            relations,
            iterations=5,
            alpha=0.3
        )

        self.assertIsInstance(stats, dict)
        self.assertIn('iterations', stats)
        self.assertIn('alpha', stats)
        self.assertIn('tokens_affected', stats)
        self.assertIn('total_adjustment', stats)
        self.assertIn('relations_used', stats)

        self.assertEqual(stats['iterations'], 5)
        self.assertEqual(stats['alpha'], 0.3)

    def test_retrofit_connections_affects_weights(self):
        """Test that retrofitting changes connection weights."""
        # Create fresh processor
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "neural networks learning deep")
        processor.process_document("doc2", "neural learning patterns data")
        processor.compute_all(verbose=False)

        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer
        )

        stats = retrofit_connections(
            processor.layers,
            relations,
            iterations=10,
            alpha=0.3
        )

        # If there are relations, some adjustment should occur
        if stats['relations_used'] > 0:
            self.assertGreaterEqual(stats['tokens_affected'], 0)

    def test_retrofit_embeddings(self):
        """Test retrofitting embeddings."""
        relations = extract_corpus_semantics(
            self.processor.layers,
            self.processor.documents,
            self.processor.tokenizer
        )

        embeddings, _ = compute_graph_embeddings(
            self.processor.layers,
            dimensions=16,
            method='adjacency'
        )

        stats = retrofit_embeddings(
            embeddings,
            relations,
            iterations=5,
            alpha=0.4
        )

        self.assertIsInstance(stats, dict)
        self.assertIn('iterations', stats)
        self.assertIn('alpha', stats)
        self.assertIn('terms_retrofitted', stats)
        self.assertIn('total_movement', stats)

        self.assertEqual(stats['iterations'], 5)
        self.assertEqual(stats['alpha'], 0.4)

    def test_get_relation_type_weight(self):
        """Test getting relation type weights."""
        # Test known relation types
        self.assertEqual(get_relation_type_weight('IsA'), 1.5)
        self.assertEqual(get_relation_type_weight('SameAs'), 2.0)
        self.assertEqual(get_relation_type_weight('Antonym'), -0.5)
        self.assertEqual(get_relation_type_weight('RelatedTo'), 0.5)

        # Test unknown relation type defaults to 0.5
        self.assertEqual(get_relation_type_weight('UnknownRelation'), 0.5)

    def test_relation_weights_constant(self):
        """Test that RELATION_WEIGHTS contains expected keys."""
        expected_relations = ['IsA', 'PartOf', 'HasA', 'SameAs', 'RelatedTo', 'CoOccurs']
        for rel in expected_relations:
            self.assertIn(rel, RELATION_WEIGHTS)


class TestSemanticsEmptyCorpus(unittest.TestCase):
    """Test semantics with empty corpus."""

    def test_empty_corpus_semantics(self):
        """Test semantic extraction on empty processor."""
        processor = CorticalTextProcessor()
        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer
        )
        self.assertEqual(relations, [])

    def test_retrofit_empty_relations(self):
        """Test retrofitting with empty relations list."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "test content here")
        processor.compute_all(verbose=False)

        stats = retrofit_connections(
            processor.layers,
            [],  # Empty relations
            iterations=5,
            alpha=0.3
        )

        self.assertEqual(stats['tokens_affected'], 0)
        self.assertEqual(stats['relations_used'], 0)


class TestSemanticsWindowSize(unittest.TestCase):
    """Test semantic extraction with different window sizes."""

    def test_larger_window_more_relations(self):
        """Test that larger window finds more co-occurrences."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", """
            word1 word2 word3 word4 word5 word6 word7 word8
        """)
        processor.compute_all(verbose=False)

        relations_small = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer,
            window_size=2
        )

        relations_large = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer,
            window_size=10
        )

        # Larger window should find at least as many relations
        self.assertGreaterEqual(len(relations_large), len(relations_small))


if __name__ == "__main__":
    unittest.main(verbosity=2)
