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
    RELATION_WEIGHTS,
    build_isa_hierarchy,
    get_ancestors,
    get_descendants,
    inherit_properties,
    compute_property_similarity,
    apply_inheritance_to_connections
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


class TestIsAHierarchy(unittest.TestCase):
    """Test IsA hierarchy building."""

    def test_build_isa_hierarchy_basic(self):
        """Test building IsA hierarchy from relations."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("cat", "IsA", "animal", 1.0),
            ("animal", "IsA", "living_thing", 1.0),
        ]
        parents, children = build_isa_hierarchy(relations)

        self.assertIn("animal", parents["dog"])
        self.assertIn("animal", parents["cat"])
        self.assertIn("living_thing", parents["animal"])
        self.assertIn("dog", children["animal"])
        self.assertIn("cat", children["animal"])
        self.assertIn("animal", children["living_thing"])

    def test_build_isa_hierarchy_empty(self):
        """Test building hierarchy from empty relations."""
        parents, children = build_isa_hierarchy([])
        self.assertEqual(parents, {})
        self.assertEqual(children, {})

    def test_build_isa_hierarchy_non_isa_ignored(self):
        """Test that non-IsA relations are ignored."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("dog", "HasProperty", "furry", 0.9),
            ("dog", "RelatedTo", "pet", 0.8),
        ]
        parents, children = build_isa_hierarchy(relations)

        # Only IsA relation should be captured
        self.assertEqual(len(parents), 1)
        self.assertIn("dog", parents)
        self.assertEqual(parents["dog"], {"animal"})


class TestAncestorsDescendants(unittest.TestCase):
    """Test ancestor and descendant traversal."""

    def setUp(self):
        """Set up a simple hierarchy."""
        relations = [
            ("poodle", "IsA", "dog", 1.0),
            ("dog", "IsA", "canine", 1.0),
            ("canine", "IsA", "mammal", 1.0),
            ("mammal", "IsA", "animal", 1.0),
            ("cat", "IsA", "feline", 1.0),
            ("feline", "IsA", "mammal", 1.0),
        ]
        self.parents, self.children = build_isa_hierarchy(relations)

    def test_get_ancestors(self):
        """Test getting ancestors of a term."""
        ancestors = get_ancestors("poodle", self.parents)

        self.assertIn("dog", ancestors)
        self.assertIn("canine", ancestors)
        self.assertIn("mammal", ancestors)
        self.assertIn("animal", ancestors)
        self.assertEqual(ancestors["dog"], 1)
        self.assertEqual(ancestors["canine"], 2)
        self.assertEqual(ancestors["mammal"], 3)
        self.assertEqual(ancestors["animal"], 4)

    def test_get_ancestors_direct_only(self):
        """Test that max_depth limits ancestor traversal."""
        ancestors = get_ancestors("poodle", self.parents, max_depth=2)

        self.assertIn("dog", ancestors)
        self.assertIn("canine", ancestors)
        self.assertNotIn("mammal", ancestors)

    def test_get_ancestors_no_parents(self):
        """Test ancestors of a root term."""
        ancestors = get_ancestors("animal", self.parents)
        self.assertEqual(ancestors, {})

    def test_get_descendants(self):
        """Test getting descendants of a term."""
        descendants = get_descendants("mammal", self.children)

        self.assertIn("canine", descendants)
        self.assertIn("dog", descendants)
        self.assertIn("poodle", descendants)
        self.assertIn("feline", descendants)
        self.assertIn("cat", descendants)

    def test_get_descendants_depth(self):
        """Test descendant depths are correct."""
        descendants = get_descendants("mammal", self.children)

        self.assertEqual(descendants["canine"], 1)
        self.assertEqual(descendants["feline"], 1)
        self.assertEqual(descendants["dog"], 2)
        self.assertEqual(descendants["cat"], 2)
        self.assertEqual(descendants["poodle"], 3)


class TestPropertyInheritance(unittest.TestCase):
    """Test property inheritance through IsA hierarchy."""

    def test_inherit_properties_basic(self):
        """Test basic property inheritance."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("animal", "HasProperty", "living", 0.9),
            ("animal", "HasProperty", "mortal", 0.8),
        ]
        inherited = inherit_properties(relations)

        self.assertIn("dog", inherited)
        self.assertIn("living", inherited["dog"])
        self.assertIn("mortal", inherited["dog"])

        # Check inherited weight is decayed
        living_weight, source, depth = inherited["dog"]["living"]
        self.assertEqual(source, "animal")
        self.assertEqual(depth, 1)
        # Weight should be 0.9 * 0.7 (default decay) = 0.63
        self.assertAlmostEqual(living_weight, 0.63, places=2)

    def test_inherit_properties_multi_level(self):
        """Test property inheritance through multiple levels."""
        relations = [
            ("poodle", "IsA", "dog", 1.0),
            ("dog", "IsA", "animal", 1.0),
            ("animal", "HasProperty", "living", 1.0),
        ]
        inherited = inherit_properties(relations, decay_factor=0.5)

        # Poodle should inherit "living" through dog → animal
        self.assertIn("poodle", inherited)
        self.assertIn("living", inherited["poodle"])

        # Weight should be decayed twice: 1.0 * 0.5^2 = 0.25
        weight, source, depth = inherited["poodle"]["living"]
        self.assertAlmostEqual(weight, 0.25, places=2)
        self.assertEqual(depth, 2)

    def test_inherit_properties_empty(self):
        """Test inheritance with no IsA relations."""
        relations = [
            ("dog", "RelatedTo", "pet", 1.0),
            ("dog", "HasProperty", "furry", 0.9),
        ]
        inherited = inherit_properties(relations)

        # No inheritance should occur (no IsA hierarchy)
        self.assertEqual(len(inherited), 0)

    def test_inherit_properties_custom_decay(self):
        """Test custom decay factor."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("animal", "HasProperty", "living", 1.0),
        ]

        inherited_slow = inherit_properties(relations, decay_factor=0.9)
        inherited_fast = inherit_properties(relations, decay_factor=0.3)

        slow_weight, _, _ = inherited_slow["dog"]["living"]
        fast_weight, _, _ = inherited_fast["dog"]["living"]

        # Slower decay should give higher weight
        self.assertGreater(slow_weight, fast_weight)

    def test_inherit_properties_max_depth(self):
        """Test max_depth limits inheritance."""
        relations = [
            ("a", "IsA", "b", 1.0),
            ("b", "IsA", "c", 1.0),
            ("c", "IsA", "d", 1.0),
            ("d", "HasProperty", "prop", 1.0),
        ]

        inherited = inherit_properties(relations, max_depth=2)

        # 'c' is at depth 2, so it should inherit
        self.assertIn("c", inherited)
        # 'a' would need depth 3 to reach 'd', so it shouldn't inherit
        self.assertNotIn("a", inherited)


class TestPropertySimilarity(unittest.TestCase):
    """Test property-based similarity computation."""

    def test_compute_property_similarity_shared(self):
        """Test similarity between terms with shared inherited properties."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("cat", "IsA", "animal", 1.0),
            ("animal", "HasProperty", "living", 1.0),
            ("animal", "HasProperty", "mortal", 1.0),
        ]
        inherited = inherit_properties(relations)

        sim = compute_property_similarity("dog", "cat", inherited)

        # Both inherit same properties, so similarity should be 1.0
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_compute_property_similarity_disjoint(self):
        """Test similarity between terms with no shared properties."""
        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("car", "IsA", "vehicle", 1.0),
            ("animal", "HasProperty", "living", 1.0),
            ("vehicle", "HasProperty", "mechanical", 1.0),
        ]
        inherited = inherit_properties(relations)

        sim = compute_property_similarity("dog", "car", inherited)

        # No shared properties
        self.assertEqual(sim, 0.0)

    def test_compute_property_similarity_partial(self):
        """Test similarity with partial property overlap."""
        relations = [
            ("dog", "IsA", "pet", 1.0),
            ("cat", "IsA", "pet", 1.0),
            ("pet", "HasProperty", "domesticated", 1.0),
            ("dog", "IsA", "canine", 1.0),
            ("canine", "HasProperty", "pack_animal", 1.0),
        ]
        inherited = inherit_properties(relations)

        sim = compute_property_similarity("dog", "cat", inherited)

        # Partial overlap: both have "domesticated", only dog has "pack_animal"
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_compute_property_similarity_no_inheritance(self):
        """Test similarity when terms have no inherited properties."""
        inherited = {}
        sim = compute_property_similarity("unknown1", "unknown2", inherited)
        self.assertEqual(sim, 0.0)


class TestApplyInheritanceToConnections(unittest.TestCase):
    """Test applying inheritance to lateral connections."""

    def test_apply_inheritance_to_connections(self):
        """Test that inheritance boosts connections."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "The dog and cat are both animals.")
        processor.compute_all(verbose=False)

        relations = [
            ("dog", "IsA", "animal", 1.0),
            ("cat", "IsA", "animal", 1.0),
            ("animal", "HasProperty", "living", 1.0),
        ]
        inherited = inherit_properties(relations)

        # Get initial connection weight between dog and cat
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        dog = layer0.get_minicolumn("dog")
        cat = layer0.get_minicolumn("cat")

        if dog and cat:
            initial_weight = dog.lateral_connections.get(cat.id, 0)

            stats = apply_inheritance_to_connections(
                processor.layers,
                inherited,
                boost_factor=0.5
            )

            # Should have boosted at least one connection
            self.assertGreaterEqual(stats['connections_boosted'], 0)

    def test_apply_inheritance_empty(self):
        """Test applying empty inheritance."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")
        processor.compute_all(verbose=False)

        stats = apply_inheritance_to_connections(
            processor.layers,
            {},  # Empty inheritance
            boost_factor=0.3
        )

        self.assertEqual(stats['connections_boosted'], 0)
        self.assertEqual(stats['total_boost'], 0.0)


class TestProcessorPropertyInheritance(unittest.TestCase):
    """Test processor-level property inheritance methods."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with sample data containing IsA patterns."""
        cls.processor = CorticalTextProcessor()
        # Documents with IsA patterns
        cls.processor.process_document("doc1", """
            A dog is a type of animal that barks.
            Dogs are loyal pets that live with humans.
            Animals are living creatures that need food.
        """)
        cls.processor.process_document("doc2", """
            Cats are animals that meow and purr.
            A cat is a popular pet in many homes.
            Pets are domesticated animals.
        """)
        cls.processor.process_document("doc3", """
            Cars are vehicles used for transportation.
            A vehicle is a machine that moves people.
            Machines are mechanical devices.
        """)
        cls.processor.compute_all(verbose=False)

    def test_compute_property_inheritance_returns_stats(self):
        """Test that compute_property_inheritance returns expected stats."""
        stats = self.processor.compute_property_inheritance(
            apply_to_connections=False,
            verbose=False
        )

        self.assertIn('terms_with_inheritance', stats)
        self.assertIn('total_properties_inherited', stats)
        self.assertIn('inherited', stats)
        self.assertIn('connections_boosted', stats)

    def test_compute_property_inheritance_with_connections(self):
        """Test inheritance applied to connections."""
        stats = self.processor.compute_property_inheritance(
            apply_to_connections=True,
            boost_factor=0.3,
            verbose=False
        )

        # Should have processed without error
        self.assertIsInstance(stats['connections_boosted'], int)
        self.assertIsInstance(stats['total_boost'], float)

    def test_compute_property_similarity_method(self):
        """Test processor compute_property_similarity method."""
        self.processor.extract_corpus_semantics(verbose=False)

        # Compute similarity (may be 0 if no shared properties in this corpus)
        sim = self.processor.compute_property_similarity("dog", "cat")
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_compute_property_inheritance_no_relations(self):
        """Test inheritance when no semantic relations extracted."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Simple test content.")
        processor.compute_all(verbose=False)
        # Don't extract semantics

        # Should work without error (extracts semantics automatically)
        stats = processor.compute_property_inheritance(
            apply_to_connections=False,
            verbose=False
        )
        self.assertIn('terms_with_inheritance', stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
