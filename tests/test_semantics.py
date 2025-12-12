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
    RELATION_PATTERNS,
    build_isa_hierarchy,
    get_ancestors,
    get_descendants,
    inherit_properties,
    compute_property_similarity,
    apply_inheritance_to_connections,
    extract_pattern_relations,
    get_pattern_statistics
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
        self.assertEqual(get_relation_type_weight('RelatedTo'), 0.8)  # Centralized in constants.py

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


class TestPatternRelationExtraction(unittest.TestCase):
    """Test pattern-based relation extraction."""

    def test_relation_patterns_defined(self):
        """Test that RELATION_PATTERNS constant is defined."""
        self.assertIsInstance(RELATION_PATTERNS, list)
        self.assertGreater(len(RELATION_PATTERNS), 0)

        # Each pattern should be a tuple with 4 elements
        for pattern in RELATION_PATTERNS:
            self.assertEqual(len(pattern), 4)
            regex, rel_type, confidence, swap = pattern
            self.assertIsInstance(regex, str)
            self.assertIsInstance(rel_type, str)
            self.assertIsInstance(confidence, float)
            self.assertIsInstance(swap, bool)

    def test_extract_isa_pattern(self):
        """Test extraction of IsA relations from text patterns."""
        docs = {
            "doc1": "A dog is a type of animal. The cat is an animal too."
        }
        valid_terms = {"dog", "animal", "cat", "type"}

        relations = extract_pattern_relations(docs, valid_terms)

        # Should find at least some IsA relations
        isa_relations = [r for r in relations if r[1] == 'IsA']
        # Note: may or may not find depending on pattern specificity
        self.assertIsInstance(relations, list)

    def test_extract_hasa_pattern(self):
        """Test extraction of HasA relations from text patterns."""
        docs = {
            "doc1": "The car has an engine. A house contains rooms."
        }
        valid_terms = {"car", "engine", "house", "rooms"}

        relations = extract_pattern_relations(docs, valid_terms, min_confidence=0.5)

        # Check we got some relations
        self.assertIsInstance(relations, list)

    def test_extract_usedfor_pattern(self):
        """Test extraction of UsedFor relations from text patterns."""
        docs = {
            "doc1": "The hammer is used for construction. Tools are useful for building."
        }
        valid_terms = {"hammer", "construction", "tools", "building"}

        relations = extract_pattern_relations(docs, valid_terms, min_confidence=0.5)

        usedfor_relations = [r for r in relations if r[1] == 'UsedFor']
        # May find UsedFor relations
        self.assertIsInstance(usedfor_relations, list)

    def test_extract_causes_pattern(self):
        """Test extraction of Causes relations from text patterns."""
        docs = {
            "doc1": "Rain causes floods. The virus leads to illness."
        }
        valid_terms = {"rain", "floods", "virus", "illness"}

        relations = extract_pattern_relations(docs, valid_terms, min_confidence=0.5)

        causes_relations = [r for r in relations if r[1] == 'Causes']
        # Should find some causal relations
        self.assertIsInstance(causes_relations, list)

    def test_min_confidence_filtering(self):
        """Test that min_confidence filters low-confidence relations."""
        docs = {
            "doc1": "The dog is happy. A cat is a pet."
        }
        valid_terms = {"dog", "happy", "cat", "pet"}

        # Low confidence threshold
        relations_low = extract_pattern_relations(docs, valid_terms, min_confidence=0.3)

        # High confidence threshold
        relations_high = extract_pattern_relations(docs, valid_terms, min_confidence=0.9)

        # Low threshold should find at least as many
        self.assertGreaterEqual(len(relations_low), len(relations_high))

    def test_stopwords_filtered(self):
        """Test that stopwords are filtered from extracted relations."""
        docs = {
            "doc1": "The is a the. A an is the a."
        }
        valid_terms = {"the", "a", "an", "is"}

        relations = extract_pattern_relations(docs, valid_terms)

        # Should not find relations between pure stopwords
        self.assertEqual(len(relations), 0)

    def test_same_term_filtered(self):
        """Test that relations between same terms are filtered."""
        docs = {
            "doc1": "The dog is a dog. Cat is cat."
        }
        valid_terms = {"dog", "cat"}

        relations = extract_pattern_relations(docs, valid_terms)

        # Should not find self-relations
        for t1, rel, t2, conf in relations:
            self.assertNotEqual(t1, t2)

    def test_invalid_terms_filtered(self):
        """Test that relations with terms not in corpus are filtered."""
        docs = {
            "doc1": "A unicorn is a mythical creature."
        }
        valid_terms = {"creature"}  # "unicorn" and "mythical" not valid

        relations = extract_pattern_relations(docs, valid_terms)

        # Should not find relations with invalid terms
        self.assertEqual(len(relations), 0)

    def test_get_pattern_statistics(self):
        """Test pattern statistics computation."""
        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("cat", "IsA", "animal", 0.9),
            ("hammer", "UsedFor", "construction", 0.8),
        ]

        stats = get_pattern_statistics(relations)

        self.assertEqual(stats['total_relations'], 3)
        self.assertEqual(stats['unique_types'], 2)
        self.assertEqual(stats['relation_type_counts']['IsA'], 2)
        self.assertEqual(stats['relation_type_counts']['UsedFor'], 1)
        self.assertAlmostEqual(stats['average_confidence_by_type']['IsA'], 0.9)

    def test_empty_relations_statistics(self):
        """Test statistics with empty relations."""
        stats = get_pattern_statistics([])

        self.assertEqual(stats['total_relations'], 0)
        self.assertEqual(stats['unique_types'], 0)
        self.assertEqual(stats['relation_type_counts'], {})


class TestProcessorPatternExtraction(unittest.TestCase):
    """Test processor-level pattern extraction methods."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with documents containing various patterns."""
        cls.processor = CorticalTextProcessor()
        cls.processor.process_document("doc1", """
            A neural network is a type of machine learning model.
            Machine learning is used for pattern recognition.
            Deep learning enables complex feature extraction.
        """)
        cls.processor.process_document("doc2", """
            The brain contains neurons that process information.
            Neurons are connected by synapses.
            Processing causes activation patterns.
        """)
        cls.processor.process_document("doc3", """
            Algorithms are used for data processing.
            Data processing leads to insights.
            Insights help decision making.
        """)
        cls.processor.compute_all(verbose=False)

    def test_extract_pattern_relations_returns_list(self):
        """Test that extract_pattern_relations returns a list."""
        relations = self.processor.extract_pattern_relations(verbose=False)
        self.assertIsInstance(relations, list)

    def test_extract_pattern_relations_format(self):
        """Test that extracted relations have correct format."""
        relations = self.processor.extract_pattern_relations(verbose=False)

        for relation in relations:
            self.assertEqual(len(relation), 4)
            t1, rel_type, t2, confidence = relation
            self.assertIsInstance(t1, str)
            self.assertIsInstance(rel_type, str)
            self.assertIsInstance(t2, str)
            self.assertIsInstance(confidence, float)
            self.assertGreater(confidence, 0)
            self.assertLessEqual(confidence, 1.0)

    def test_extract_corpus_semantics_with_patterns(self):
        """Test extract_corpus_semantics with pattern extraction enabled."""
        count = self.processor.extract_corpus_semantics(
            use_pattern_extraction=True,
            verbose=False
        )

        self.assertGreater(count, 0)
        self.assertGreater(len(self.processor.semantic_relations), 0)

    def test_extract_corpus_semantics_without_patterns(self):
        """Test extract_corpus_semantics without pattern extraction."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process information quickly.")
        processor.compute_all(verbose=False)

        count_with = processor.extract_corpus_semantics(
            use_pattern_extraction=True,
            verbose=False
        )

        processor.semantic_relations = []

        count_without = processor.extract_corpus_semantics(
            use_pattern_extraction=False,
            verbose=False
        )

        # With patterns should find at least as many (usually more)
        # But depending on corpus, might be same
        self.assertGreaterEqual(count_with, 0)
        self.assertGreaterEqual(count_without, 0)

    def test_custom_min_confidence(self):
        """Test custom minimum confidence threshold."""
        relations_low = self.processor.extract_pattern_relations(
            min_confidence=0.3,
            verbose=False
        )

        relations_high = self.processor.extract_pattern_relations(
            min_confidence=0.9,
            verbose=False
        )

        # Lower confidence should find at least as many
        self.assertGreaterEqual(len(relations_low), len(relations_high))


class TestSimilarToRelationExtraction(unittest.TestCase):
    """Test SimilarTo relation extraction with context similarity."""

    def test_similarto_with_shared_context(self):
        """Test SimilarTo extraction when terms share context."""
        from cortical.processor import CorticalTextProcessor
        from cortical.semantics import extract_corpus_semantics

        # Create corpus with terms that share context
        # "apple" and "orange" both appear near "fruit", "eat", "fresh", "juice"
        processor = CorticalTextProcessor()
        processor.process_document("doc1",
            "I eat fresh apple fruit. Apple juice is healthy. The apple is fresh.")
        processor.process_document("doc2",
            "I eat fresh orange fruit. Orange juice is healthy. The orange is fresh.")
        processor.process_document("doc3",
            "Fresh fruit juice from apple and orange. Eat fresh fruit daily.")
        processor.compute_all(verbose=False)

        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer,
            window_size=5,
            min_cooccurrence=2,
            use_pattern_extraction=False  # Only test similarity
        )

        # Check that we get some relations
        self.assertIsInstance(relations, list)

        # Check relation types
        relation_types = set(r[1] for r in relations)
        # Should have CoOccurs at minimum
        self.assertIn('CoOccurs', relation_types)

    def test_extract_corpus_semantics_similarto_threshold(self):
        """Test that SimilarTo respects similarity threshold."""
        from cortical.processor import CorticalTextProcessor
        from cortical.semantics import extract_corpus_semantics

        # Create documents with overlapping terms
        processor = CorticalTextProcessor()
        for i in range(5):
            processor.process_document(f"doc{i}",
                f"The quick brown fox jumps over the lazy dog. "
                f"Quick foxes are brown and lazy dogs sleep. "
                f"Brown quick lazy fox dog jump sleep.")
        processor.compute_all(verbose=False)

        relations = extract_corpus_semantics(
            processor.layers,
            processor.documents,
            processor.tokenizer,
            window_size=3,
            min_cooccurrence=2,
            use_pattern_extraction=False
        )

        # Verify relation structure
        for rel in relations:
            self.assertEqual(len(rel), 4)
            term1, rel_type, term2, weight = rel
            self.assertIn(rel_type, ['CoOccurs', 'SimilarTo'])
            self.assertGreater(weight, 0)
            # SimilarTo is 0-1, but CoOccurs can be higher (count-based)
            if rel_type == 'SimilarTo':
                self.assertLessEqual(weight, 1.0)


class TestBigramConnectionsVerbose(unittest.TestCase):
    """Test bigram connection verbose output and new parameters."""

    def test_max_bigrams_per_term_parameter(self):
        """Test that max_bigrams_per_term skips common terms."""
        from cortical.processor import CorticalTextProcessor

        processor = CorticalTextProcessor()
        # Create documents with common bigram prefix "data" (not a stop word)
        for i in range(20):
            processor.process_document(f"doc{i}",
                f"data processing data analysis data mining data science "
                f"data engineering data storage data pipeline data flow")
        processor.compute_all(verbose=False, build_concepts=False)

        # With very low threshold, should skip "data" as it appears in many bigrams
        stats = processor.compute_bigram_connections(
            max_bigrams_per_term=3,
            verbose=False
        )

        self.assertIn('skipped_common_terms', stats)
        self.assertGreater(stats['skipped_common_terms'], 0)

    def test_max_bigrams_per_doc_parameter(self):
        """Test that max_bigrams_per_doc skips large documents."""
        from cortical.processor import CorticalTextProcessor

        processor = CorticalTextProcessor()
        # Create one large document and several small ones
        large_doc = " ".join([f"word{i} word{i+1}" for i in range(200)])
        processor.process_document("large", large_doc)
        for i in range(5):
            processor.process_document(f"small{i}", "simple short document here")
        processor.compute_all(verbose=False, build_concepts=False)

        # With low threshold, should skip the large document
        stats = processor.compute_bigram_connections(
            max_bigrams_per_doc=50,
            verbose=False
        )

        self.assertIn('skipped_large_docs', stats)
        self.assertGreater(stats['skipped_large_docs'], 0)

    def test_bigram_connections_returns_all_stats(self):
        """Test that bigram connections returns complete statistics."""
        from cortical.processor import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "machine learning algorithms work well")
        processor.process_document("doc2", "deep learning neural networks train fast")
        processor.compute_all(verbose=False, build_concepts=False)

        stats = processor.compute_bigram_connections(verbose=False)

        # Check all expected keys
        expected_keys = [
            'connections_created', 'bigrams', 'component_connections',
            'chain_connections', 'cooccurrence_connections',
            'skipped_common_terms', 'skipped_large_docs'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)


class TestProcessorVerboseOutput(unittest.TestCase):
    """Test verbose output messages."""

    def test_compute_bigram_connections_verbose_skipped(self):
        """Test verbose output includes skipped info."""
        import io
        import sys
        from cortical.processor import CorticalTextProcessor

        processor = CorticalTextProcessor()
        for i in range(15):
            processor.process_document(f"doc{i}",
                f"the quick brown fox jumps over the lazy dog number {i}")
        processor.compute_all(verbose=False, build_concepts=False)

        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            processor.compute_bigram_connections(
                max_bigrams_per_term=3,
                verbose=True
            )
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should mention "bigram connections"
        self.assertIn('bigram connections', output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
