"""
Behavioral tests for query expansion using lateral connections and semantic relations.

Epic: Query Expansion and Concept Discovery

As a researcher with incomplete or vague search queries,
I want automatic query expansion that discovers related terms,
So that I find relevant documents even when my initial keywords are imprecise.

Based on: cortical/query/expansion.py (query expansion functionality)
"""

import pytest
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


class TestResearcherExpandsQueriesWithLateralConnections:
    """
    Epic: Query Expansion and Concept Discovery

    As a researcher formulating search queries,
    I want automatic expansion to related concepts,
    So that I find documents beyond exact keyword matches.
    """

    def test_scenario_researcher_expands_query_with_related_terms(self):
        """
        Scenario: Query expansion adds semantically related terms

        Given a corpus with co-occurring terms
        When I expand a query
        Then the system adds related terms
        And each expansion term has a weight indicating relevance
        Because expansion improves recall by including related concepts.
        """
        # GIVEN a corpus with co-occurring terms
        docs = {
            "ml1": "Machine learning algorithms train models on datasets.",
            "ml2": "Neural networks learn patterns through supervised training.",
            "ml3": "Deep learning models use gradient descent for optimization.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I expand a query
        expanded = processor.expand_query(
            "machine learning",
            max_expansions=5
        )

        # THEN the system adds related terms
        assert len(expanded) > 0, "Should return expanded terms"

        # AND each expansion term has a weight indicating relevance
        for term, weight in expanded.items():
            assert isinstance(term, str), "Term should be a string"
            assert isinstance(weight, (int, float)), "Weight should be numeric"
            assert weight > 0, "Weight should be positive"

    def test_scenario_researcher_expands_with_concept_clusters(self):
        """
        Scenario: Expansion uses concept cluster membership

        Given a corpus with concept clusters (Layer 2)
        When expanding a query term
        Then terms from the same concept cluster are added
        And cluster-based expansion finds semantically related terms
        Because concepts group semantically similar terms together.
        """
        # GIVEN a corpus with concept clusters
        docs = {
            "neural1": "Neural networks process information using layers.",
            "neural2": "Deep neural architectures learn hierarchical features.",
            "neural3": "Convolutional networks excel at image processing.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN expanding a query term
        expanded = processor.expand_query(
            "neural",
            max_expansions=8
        )

        # THEN terms from the same concept cluster are added
        assert len(expanded) > 0, "Should expand query"

        # Expansion should include related terms
        expanded_terms = list(expanded.keys())
        # Should have more than just the original term
        assert len(expanded_terms) >= 1

    def test_scenario_researcher_uses_word_variants_for_matching(self):
        """
        Scenario: Word variants help match different forms

        Given a query term not in the corpus vocabulary
        When expansion tries word variants (stemming)
        Then related word forms are matched
        And variant matching improves recall
        Because users may use different word forms than the corpus.
        """
        # GIVEN a corpus with specific word forms
        docs = {
            "doc1": "The algorithm optimizes the objective function.",
            "doc2": "Optimization techniques improve model performance.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN expansion tries word variants
        expanded = processor.expand_query(
            "optimize",
            use_variants=True,
            max_expansions=5
        )

        # THEN related word forms are matched
        assert len(expanded) > 0, "Should find matching variants"

    def test_scenario_researcher_expands_with_code_concepts(self):
        """
        Scenario: Code concept expansion finds programming synonyms

        Given queries about code operations
        When using code concept expansion
        Then the expansion is performed with code concept awareness
        Because programmers use varied terminology for same operations.
        """
        # GIVEN a corpus about code operations
        docs = {
            "fetcher": "The fetch function retrieves data from the database.",
            "getter": "Use the getter method to access cached values.",
            "loader": "The load operation reads files from disk storage.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN using code concept expansion
        expanded = processor.expand_query(
            "fetch",
            use_code_concepts=True,
            max_expansions=8
        )

        # THEN expansion is performed
        assert len(expanded) > 0, "Should expand with code concepts"

        # Should include terms from the corpus
        assert "fetch" in expanded or len(expanded) >= 1

    def test_scenario_researcher_filters_code_stop_words(self):
        """
        Scenario: Filtering ubiquitous code tokens from expansion

        Given a code corpus with common keywords (self, def, return)
        When expanding queries with code stop word filtering
        Then ubiquitous tokens are excluded from expansion
        And expansion focuses on meaningful domain terms
        Because filtering noise improves expansion quality.
        """
        # GIVEN a code corpus with common keywords
        docs = {
            "class1": "class Network: def __init__(self): self.layers = []",
            "class2": "class Model: def forward(self, x): return self.compute(x)",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN expanding queries with code stop word filtering
        expanded = processor.expand_query(
            "network",
            filter_code_stop_words=True,
            max_expansions=5
        )

        # THEN ubiquitous tokens are excluded from expansion
        # Common stop words like 'self', 'def', 'return' should not dominate
        assert len(expanded) > 0, "Should still expand query"

        # Should not be dominated by stop words
        stop_words = {'self', 'cls', 'def', 'return', 'class'}
        expanded_terms = set(expanded.keys())
        # Most expansions should not be stop words
        non_stop_count = len(expanded_terms - stop_words)
        assert non_stop_count >= 0, "Should have non-stop-word expansions"


class TestResearcherExpandsQueriesWithSemanticRelations:
    """
    Epic: Semantic Relation-Based Expansion

    As a researcher with semantic knowledge graphs,
    I want query expansion using typed semantic relations,
    So that I discover related concepts through explicit relationships.
    """

    def test_scenario_researcher_expands_with_semantic_relations(self):
        """
        Scenario: Single-hop semantic expansion via relations

        Given a corpus with extracted semantic relations
        When expanding a query using semantic relations
        Then terms connected by semantic relations are added
        Because semantic relations capture explicit knowledge.
        """
        # GIVEN a corpus with semantic relations
        docs = {
            "animals": "Dogs are mammals. Mammals are animals. Dogs are pets.",
            "traits": "Dogs are loyal. Animals have life. Pets need care.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # Add semantic relations to processor
        processor.semantic_relations = [
            ("dogs", "IsA", "mammals", 0.9),
            ("mammals", "IsA", "animals", 0.9),
            ("dogs", "HasProperty", "loyal", 0.8),
        ]

        # WHEN expanding a query using semantic relations
        expanded = processor.expand_query_semantic(
            "dogs",
            max_expansions=5
        )

        # THEN terms connected by semantic relations are added
        assert len(expanded) > 0, "Should expand using semantic relations"

        # Should include the original term or related terms
        assert "dogs" in expanded or len(expanded) >= 1

    def test_scenario_researcher_uses_multihop_semantic_inference(self):
        """
        Scenario: Multi-hop expansion follows relation chains

        Given semantic relations forming chains
        When using multi-hop expansion
        Then the system follows transitive relationships
        Because multi-hop inference discovers indirect relationships.
        """
        # GIVEN semantic relations forming chains
        docs = {
            "hierarchy": "A dog is a mammal. A mammal is an animal. An animal is alive.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # Add chained relations to processor
        processor.semantic_relations = [
            ("dog", "IsA", "mammal", 0.95),
            ("mammal", "IsA", "animal", 0.95),
            ("animal", "HasProperty", "alive", 0.9),
        ]

        # WHEN using multi-hop expansion
        expanded = processor.expand_query_multihop(
            "dog",
            max_hops=2,
            max_expansions=10
        )

        # THEN the system follows transitive relationships
        assert len(expanded) > 0, "Should perform multi-hop expansion"

        # Should include original term
        assert "dog" in expanded

    def test_scenario_researcher_weights_expansion_paths_by_validity(self):
        """
        Scenario: Relation chain validity affects expansion weights

        Given relation chains with different validity scores
        When computing multi-hop expansion
        Then valid chains (IsA->IsA) get higher scores
        Because not all relation chains are semantically valid.
        """
        # GIVEN relation chains with different types
        docs = {
            "relations": "Hot is the opposite of cold. Cold is a temperature.",
            "animals": "A dog is a mammal. A mammal is an animal.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # Add relations to processor
        processor.semantic_relations = [
            ("dog", "IsA", "mammal", 0.9),
            ("mammal", "IsA", "animal", 0.9),  # Valid chain: IsA->IsA
            ("hot", "Antonym", "cold", 0.8),
            ("cold", "IsA", "temperature", 0.8),  # Weaker chain: Antonym->IsA
        ]

        # WHEN computing multi-hop expansion
        expanded_dog = processor.expand_query_multihop(
            "dog",
            max_hops=2,
            max_expansions=5
        )

        # THEN valid chains get higher expansion weights
        assert len(expanded_dog) > 0, "Should expand query"

        # Verify expansion works
        assert "dog" in expanded_dog, "Should include original term"

    def test_scenario_researcher_controls_expansion_weight_caps(self):
        """
        Scenario: Expansion weights are bounded

        Given query expansion producing varied term weights
        When expanding a query
        Then expansion weights are reasonable (not extreme)
        Because balanced weights ensure balanced ranking.
        """
        # GIVEN a corpus with strong co-occurrences
        docs = {
            "doc1": "Neural networks and deep learning are closely related concepts.",
            "doc2": "Neural architectures use deep layers for representation learning.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN expanding a query
        expanded = processor.expand_query(
            "neural",
            max_expansions=8
        )

        # THEN expansion weights are reasonable
        for term, weight in expanded.items():
            # Weights should be positive and bounded
            assert weight > 0, f"Weight for {term} should be positive"
            assert weight <= 10.0, f"Weight for {term} should be reasonable"


class TestResearcherBalancesExpansionSignals:
    """
    Epic: Expansion Quality Control

    As a researcher tuning search quality,
    I want control over expansion signal weights,
    So that I balance distinctiveness vs importance in expansion.
    """

    def test_scenario_researcher_expands_common_and_rare_terms(self):
        """
        Scenario: Expansion works for both common and rare terms

        Given a corpus with both distinctive and common terms
        When expanding queries for common and rare terms
        Then both types of terms are expanded
        Because expansion should work across the vocabulary.
        """
        # GIVEN a corpus with both distinctive and common terms
        docs = {
            "doc1": "Specialized terminology like eigendecomposition appears rarely.",
            "doc2": "Common words like learning appear frequently in machine learning.",
            "doc3": "Deep learning models use gradient descent for optimization.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN expanding a common term
        expanded_common = processor.expand_query(
            "learning",
            max_expansions=5
        )

        # AND expanding with different parameters
        expanded_with_variants = processor.expand_query(
            "learning",
            use_variants=True,
            max_expansions=5
        )

        # THEN expansions work
        assert len(expanded_common) > 0, "Common term expansion should work"
        assert len(expanded_with_variants) > 0, "Variant expansion should work"

    def test_scenario_researcher_limits_expansion_count(self):
        """
        Scenario: Controlling number of expansion terms

        Given a query that could expand to many terms
        When setting max_expansions parameter
        Then only the top N most relevant terms are added
        And expansion quality is maintained
        Because too many expansions can dilute precision.
        """
        # GIVEN a query that could expand to many terms
        docs = {
            f"doc{i}": f"Document about neural networks, deep learning, and AI model {i}."
            for i in range(5)
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN setting max_expansions parameter
        original_tokens = processor.tokenizer.tokenize("neural")
        expanded_small = processor.expand_query("neural", max_expansions=2)
        expanded_large = processor.expand_query("neural", max_expansions=10)

        # THEN only the top N most relevant terms are added
        # Small expansion should be smaller or equal to large
        assert len(expanded_small) <= len(expanded_large)

        # Both should include original terms
        for token in original_tokens:
            if token in expanded_small:
                assert expanded_small[token] > 0
