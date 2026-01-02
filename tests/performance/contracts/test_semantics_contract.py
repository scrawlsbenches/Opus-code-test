"""
╔══════════════════════════════════════════════════════════════════════╗
║                   SEMANTICS PERFORMANCE CONTRACT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Extract corpus semantics < 5 seconds for ≤ 100 documents         ║
║  • Pattern extraction < 2 seconds for ≤ 100 documents               ║
║  • Retrofit 1000 tokens < 1 second                                  ║
║  • Build IsA hierarchy < 500ms for 1000 relations                   ║
║  • Inherit properties < 1 second for 500 terms                      ║
║  • Semantic relations include confidence scores                     ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest
from cortical.layers import HierarchicalLayer, CorticalLayer
from cortical.semantics import (
    extract_corpus_semantics,
    extract_pattern_relations,
    retrofit_connections,
    build_isa_hierarchy,
    inherit_properties,
    get_ancestors,
    compute_property_similarity,
)
from cortical.tokenizer import Tokenizer


@pytest.mark.contract
class TestSemanticExtractionPerformanceContract:
    """
    Semantic Extraction Performance Contract

    As a developer building semantic networks,
    I expect semantic extraction to complete in reasonable time,
    So that corpus analysis is practical.
    """

    MAX_EXTRACTION_SECONDS = 5.0
    MAX_PATTERN_EXTRACTION_SECONDS = 2.0

    def test_corpus_semantics_extraction_latency(self, small_processor):
        """
        CONTRACT: Extract corpus semantics in < 5 seconds for ≤ 100 documents.

        Semantic extraction is expensive but must be bounded.
        """
        # Verify we're within contract bounds
        assert len(small_processor.documents) <= 100

        tokenizer = Tokenizer()

        start = time.perf_counter()
        relations = extract_corpus_semantics(
            small_processor.layers,
            small_processor.documents,
            tokenizer,
            window_size=5,
            min_cooccurrence=2,
            use_pattern_extraction=True,
            max_similarity_pairs=10000  # Limit for performance
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_EXTRACTION_SECONDS, (
            f"CONTRACT VIOLATION: Semantic extraction took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_EXTRACTION_SECONDS}s"
        )

        # Verify output validity
        assert isinstance(relations, list)
        # Each relation should be (term1, relation_type, term2, weight)
        if relations:
            assert len(relations[0]) == 4

    def test_pattern_extraction_latency(self, small_processor):
        """
        CONTRACT: Pattern extraction in < 2 seconds for ≤ 100 documents.

        Pattern matching is I/O heavy but must be bounded.
        """
        assert len(small_processor.documents) <= 100

        layer0 = small_processor.layers[CorticalLayer.TOKENS]
        valid_terms = set(layer0.minicolumns.keys())

        start = time.perf_counter()
        relations = extract_pattern_relations(
            small_processor.documents,
            valid_terms,
            min_confidence=0.6
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_PATTERN_EXTRACTION_SECONDS, (
            f"CONTRACT VIOLATION: Pattern extraction took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_PATTERN_EXTRACTION_SECONDS}s"
        )

    def test_semantic_relations_have_confidence(self, small_processor):
        """
        CONTRACT: Semantic relations include confidence scores.

        Confidence allows filtering low-quality relations.
        """
        tokenizer = Tokenizer()

        relations = extract_corpus_semantics(
            small_processor.layers,
            small_processor.documents,
            tokenizer,
            use_pattern_extraction=True,
            max_similarity_pairs=5000
        )

        # Relations from pattern extraction should have confidence
        if relations:
            # Each relation: (term1, relation_type, term2, weight/confidence)
            for rel in relations[:10]:  # Check first 10
                assert len(rel) == 4, f"Relation malformed: {rel}"
                term1, rel_type, term2, weight = rel
                assert isinstance(weight, (int, float)), (
                    f"CONTRACT VIOLATION: Weight/confidence not numeric: {weight}"
                )
                assert weight > 0, (
                    f"CONTRACT VIOLATION: Non-positive weight: {weight}"
                )


@pytest.mark.contract
class TestRetrofittingPerformanceContract:
    """
    Retrofitting Performance Contract

    As a developer improving semantic quality,
    I expect retrofitting to be fast,
    So that semantic enhancement is practical.
    """

    MAX_RETROFIT_1K_SECONDS = 1.0

    def test_retrofit_connections_latency(self):
        """
        CONTRACT: Retrofit 1000 tokens in < 1 second.

        Retrofitting adjusts weights based on semantic relations.
        """
        # Create layer with 1000 tokens
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        layer0 = layers[CorticalLayer.TOKENS]

        # Create tokens with connections
        for i in range(1000):
            col = layer0.get_or_create_minicolumn(f"word_{i}")
            # Add some connections
            for j in range(5):
                target = f"L0_word_{(i + j + 1) % 1000}"
                col.add_lateral_connection(target, 0.5)

        # Create semantic relations
        relations = []
        for i in range(0, 1000, 2):
            relations.append((
                f"word_{i}",
                'SimilarTo',
                f"word_{i + 1}",
                0.8
            ))

        start = time.perf_counter()
        result = retrofit_connections(
            layers,
            relations,
            iterations=10,
            alpha=0.5
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_RETROFIT_1K_SECONDS, (
            f"CONTRACT VIOLATION: Retrofitting 1000 tokens took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_RETROFIT_1K_SECONDS}s"
        )

        # Verify it did something
        assert result['tokens_affected'] > 0

    def test_retrofit_alpha_validation(self):
        """
        CONTRACT: Alpha parameter must be in [0, 1].

        Invalid alpha should raise ValueError.
        """
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        relations = []

        # Test invalid alpha values
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            retrofit_connections(layers, relations, alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            retrofit_connections(layers, relations, alpha=1.5)


@pytest.mark.contract
class TestIsAHierarchyContract:
    """
    IsA Hierarchy Performance Contract

    As a developer building taxonomies,
    I expect IsA hierarchy construction to be fast,
    So that taxonomy analysis is practical.
    """

    MAX_BUILD_HIERARCHY_MS = 500

    def test_build_isa_hierarchy_latency(self):
        """
        CONTRACT: Build IsA hierarchy from 1000 relations in < 500ms.

        Taxonomy construction must be fast.
        """
        # Create 1000 IsA relations
        relations = []
        for i in range(1000):
            parent = f"category_{i // 10}"
            child = f"item_{i}"
            relations.append((child, 'IsA', parent, 1.0))

        start = time.perf_counter()
        parents, children = build_isa_hierarchy(relations)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_BUILD_HIERARCHY_MS, (
            f"CONTRACT VIOLATION: Building hierarchy took {elapsed_ms:.0f}ms, "
            f"contract requires <{self.MAX_BUILD_HIERARCHY_MS}ms"
        )

        # Verify structure
        assert len(children) > 0
        assert len(parents) > 0

    def test_get_ancestors_correctness(self):
        """
        CONTRACT: get_ancestors returns all ancestors with correct depths.

        Ancestry traversal must be correct.
        """
        # Build hierarchy: dog -> canine -> mammal -> animal
        relations = [
            ('dog', 'IsA', 'canine', 1.0),
            ('canine', 'IsA', 'mammal', 1.0),
            ('mammal', 'IsA', 'animal', 1.0),
        ]

        parents, children = build_isa_hierarchy(relations)
        ancestors = get_ancestors('dog', parents, max_depth=10)

        # Should find all ancestors
        assert 'canine' in ancestors
        assert 'mammal' in ancestors
        assert 'animal' in ancestors

        # Check depths
        assert ancestors['canine'] == 1
        assert ancestors['mammal'] == 2
        assert ancestors['animal'] == 3


@pytest.mark.contract
class TestPropertyInheritanceContract:
    """
    Property Inheritance Performance Contract

    As a developer building semantic reasoning,
    I expect property inheritance to be fast,
    So that semantic inference is practical.
    """

    MAX_INHERIT_PROPERTIES_SECONDS = 1.0

    def test_inherit_properties_latency(self):
        """
        CONTRACT: Inherit properties for 500 terms in < 1 second.

        Property inheritance propagates attributes down taxonomy.
        """
        # Create taxonomy with properties
        relations = []

        # Create IsA hierarchy
        for i in range(500):
            parent = f"category_{i // 20}"
            child = f"item_{i}"
            relations.append((child, 'IsA', parent, 1.0))

        # Add properties to categories
        for i in range(25):  # 500 / 20 = 25 categories
            category = f"category_{i}"
            relations.append((category, 'HasProperty', f'property_{i % 5}', 0.9))

        start = time.perf_counter()
        inherited = inherit_properties(relations, decay_factor=0.7, max_depth=5)
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_INHERIT_PROPERTIES_SECONDS, (
            f"CONTRACT VIOLATION: Inheriting properties took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_INHERIT_PROPERTIES_SECONDS}s"
        )

        # Verify inheritance happened
        assert len(inherited) > 0

    def test_property_inheritance_decay(self):
        """
        CONTRACT: Property weights decay with inheritance depth.

        Inheritance should apply decay_factor per level.
        """
        # Create simple hierarchy: child -> parent, parent has property
        relations = [
            ('child', 'IsA', 'parent', 1.0),
            ('parent', 'HasProperty', 'living', 1.0),
        ]

        inherited = inherit_properties(relations, decay_factor=0.7, max_depth=5)

        # Child should inherit 'living' with decayed weight
        assert 'child' in inherited
        assert 'living' in inherited['child']

        weight, source, depth = inherited['child']['living']
        assert weight == pytest.approx(1.0 * 0.7, rel=0.01), (
            f"CONTRACT VIOLATION: Inherited weight {weight} incorrect (expected 0.7)"
        )
        assert depth == 1

    def test_compute_property_similarity_correctness(self):
        """
        CONTRACT: Property similarity is symmetric and bounded [0, 1].

        Similarity must be mathematically valid.
        """
        relations = [
            ('dog', 'IsA', 'animal', 1.0),
            ('cat', 'IsA', 'animal', 1.0),
            ('animal', 'HasProperty', 'living', 1.0),
            ('animal', 'HasProperty', 'breathing', 1.0),
        ]

        inherited = inherit_properties(relations)

        # Both dog and cat should inherit properties from animal
        sim = compute_property_similarity('dog', 'cat', inherited)

        # Similarity should be in [0, 1]
        assert 0.0 <= sim <= 1.0, (
            f"CONTRACT VIOLATION: Similarity {sim} not in [0, 1]"
        )

        # Should be high since they share all inherited properties
        assert sim > 0.5, (
            f"CONTRACT VIOLATION: Dog-cat similarity {sim} too low (both inherit from animal)"
        )


@pytest.mark.contract
class TestSemanticCorrectnessContract:
    """
    Semantic Correctness Contract

    As a developer relying on semantic relations,
    I expect relations to be valid and meaningful,
    So that semantic reasoning is trustworthy.
    """

    def test_no_self_relations(self, small_processor):
        """
        CONTRACT: Most relations don't relate terms to themselves.

        Self-relations should be rare or nonexistent.
        """
        tokenizer = Tokenizer()

        relations = extract_corpus_semantics(
            small_processor.layers,
            small_processor.documents,
            tokenizer,
            max_similarity_pairs=5000
        )

        # Count self-relations
        self_relations = sum(1 for t1, _, t2, _ in relations if t1 == t2)
        total_relations = len(relations)

        if total_relations > 0:
            self_relation_rate = self_relations / total_relations

            # Allow up to 1% self-relations (implementation may have edge cases)
            assert self_relation_rate < 0.01, (
                f"CONTRACT VIOLATION: {self_relations}/{total_relations} "
                f"({self_relation_rate:.1%}) self-relations found"
            )

    def test_relation_types_are_valid(self, small_processor):
        """
        CONTRACT: Relation types are from known set.

        Unknown relation types indicate bugs.
        """
        tokenizer = Tokenizer()

        relations = extract_corpus_semantics(
            small_processor.layers,
            small_processor.documents,
            tokenizer,
            use_pattern_extraction=True,
            max_similarity_pairs=5000
        )

        valid_types = {
            'CoOccurs', 'SimilarTo', 'IsA', 'HasA', 'PartOf', 'UsedFor',
            'Causes', 'CapableOf', 'AtLocation', 'HasProperty', 'Antonym',
            'DerivedFrom', 'DefinedBy', 'RelatedTo'
        }

        for term1, rel_type, term2, weight in relations:
            assert rel_type in valid_types, (
                f"CONTRACT VIOLATION: Unknown relation type '{rel_type}' "
                f"in relation: {term1} {rel_type} {term2}"
            )
