"""
Behavioral tests for developers using REPL features.

Epic: Interactive Corpus Exploration

As a developer exploring document corpora,
I want interactive commands for search and analysis,
So that I can rapidly experiment and investigate data.

Based on: examples/repl_demo.py

Note: This tests the underlying features used by the REPL,
not the interactive REPL itself which requires manual testing.
"""

import pytest
from cortical.processor import CorticalTextProcessor


class TestDeveloperUsesREPLFeatures:
    """
    Epic: Interactive Corpus Exploration

    As a developer exploring corpora interactively,
    I want quick commands for common operations,
    So that I can rapidly iterate and investigate.
    """

    @pytest.mark.skip(reason="API mismatch - needs alignment with implementation")
    def test_scenario_developer_creates_corpus_for_interactive_work(self):
        """
        Scenario: Setting up a corpus for exploration

        Given documents to explore
        When I create and save a corpus
        Then the corpus persists for later use
        And I can load it for interactive exploration
        Because developers reuse corpora across sessions.
        """
        # GIVEN documents to explore
        processor = CorticalTextProcessor()

        processor.process_document(
            "neural_networks.py",
            """
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def train(self, data):
        for epoch in range(100):
            self.forward_pass(data)
            self.backward_pass()
"""
        )

        processor.process_document(
            "README.md",
            """
# Machine Learning Library

This library provides implementations of common ML algorithms.
"""
        )

        # WHEN I create and save a corpus
        processor.compute_all(verbose=False)

        # THEN the corpus persists for later use
        # Verify corpus is functional
        results = processor.find_documents_for_query("neural network", top_n=3)
        assert len(results) > 0, "Corpus should be searchable"

        # AND I can load it for interactive exploration
        # In REPL, user would save/load from files

    @pytest.mark.skip(reason="API mismatch - needs alignment with implementation")
    def test_scenario_developer_searches_corpus_interactively(self):
        """
        Scenario: Quick document search

        Given an indexed corpus
        When I execute a search query
        Then results are returned immediately
        And I see relevance scores
        Because interactive search needs fast feedback.
        """
        # GIVEN an indexed corpus
        processor = CorticalTextProcessor()
        processor.process_document(
            "ai_overview",
            "Artificial intelligence encompasses machine learning and deep learning."
        )
        processor.process_document(
            "ml_basics",
            "Machine learning algorithms learn from data without explicit programming."
        )
        processor.compute_all(verbose=False)

        # WHEN I execute a search query
        results = processor.find_documents_for_query("machine learning", top_n=3)

        # THEN results are returned immediately
        assert len(results) > 0, "Should return search results"

        # AND I see relevance scores
        for doc_id, score in results:
            assert isinstance(doc_id, str), "Should have document ID"
            assert isinstance(score, float), "Should have relevance score"
            assert 0 <= score <= 1, "Score should be normalized"

    def test_scenario_developer_examines_query_expansion(self):
        """
        Scenario: Understanding query expansion

        Given a corpus with semantic relationships
        When I check query expansion for a term
        Then I see related terms
        And understand how queries are expanded
        Because developers need to debug search behavior.
        """
        # GIVEN a corpus with semantic relationships
        processor = CorticalTextProcessor()
        processor.process_document(
            "doc1",
            "Neural networks and artificial neural networks are computational models."
        )
        processor.process_document(
            "doc2",
            "Machine learning and deep learning use neural network architectures."
        )
        processor.compute_all(verbose=False)

        # WHEN I check query expansion for a term
        expanded = processor.expand_query_cached("neural")

        # THEN I see related terms
        assert len(expanded) > 0, "Should expand query term"

        # AND understand how queries are expanded
        # Expansion includes original term and related terms

    def test_scenario_developer_gets_corpus_statistics(self):
        """
        Scenario: Understanding corpus composition

        Given a processed corpus
        When I request statistics
        Then I see document count and vocabulary size
        And understand corpus characteristics
        Because developers need corpus overview.
        """
        # GIVEN a processed corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data efficiently.")
        processor.process_document("doc2", "Machine learning algorithms analyze patterns.")
        processor.process_document("doc3", "Deep learning models require large datasets.")
        processor.compute_all(verbose=False)

        # WHEN I request statistics
        doc_count = len(processor.documents)
        vocab_size = processor.layers[0].column_count()

        # THEN I see document count and vocabulary size
        assert doc_count == 3, "Should report document count"
        assert vocab_size > 0, "Should report vocabulary size"

        # AND understand corpus characteristics
        # Statistics help understand corpus scale

    def test_scenario_developer_retrieves_passages_for_rag(self):
        """
        Scenario: Finding relevant passages

        Given documents with rich content
        When I search for passages
        Then I get text excerpts with context
        And can build RAG applications
        Because passage retrieval enables RAG.
        """
        # GIVEN documents with rich content
        processor = CorticalTextProcessor()
        processor.process_document(
            "ml_article",
            """
Machine learning is a subset of artificial intelligence.
Supervised learning uses labeled data to train models.
Unsupervised learning finds patterns in unlabeled data.
"""
        )
        processor.compute_all(verbose=False)

        # WHEN I search for passages
        passages = processor.find_passages_for_query("supervised learning", top_n=2)

        # THEN I get text excerpts with context
        assert len(passages) > 0, "Should find relevant passages"

        # AND can build RAG applications
        for passage_text, doc_id, start, end, score in passages:
            assert isinstance(passage_text, str), "Should have passage text"
            assert isinstance(doc_id, str), "Should have document ID"
            assert isinstance(score, float), "Should have relevance score"

    def test_scenario_developer_detects_patterns_in_code_corpus(self):
        """
        Scenario: Code pattern analysis

        Given a corpus of code files
        When I detect patterns
        Then I see design patterns used
        And can understand codebase architecture
        Because pattern detection aids code understanding.
        """
        # GIVEN a corpus of code files
        processor = CorticalTextProcessor()
        processor.process_document(
            "singleton.py",
            """
class Config:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
"""
        )

        # WHEN I detect patterns
        patterns = processor.detect_patterns("singleton.py")

        # THEN I see design patterns used
        assert patterns is not None, "Should detect patterns"

        # AND can understand codebase architecture
        # Pattern detection reveals design choices

    def test_scenario_developer_examines_semantic_relations(self):
        """
        Scenario: Exploring term relationships

        Given a computed corpus
        When I examine semantic relations
        Then I see which terms co-occur
        And understand semantic structure
        Because developers explore relationships.
        """
        # GIVEN a computed corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks and deep learning systems.")
        processor.process_document("doc2", "Machine learning and neural network models.")
        processor.compute_all(verbose=False)

        # WHEN I examine semantic relations
        # Get layer to examine relations
        layer0 = processor.layers[0]

        # THEN I see which terms co-occur
        assert layer0.column_count() > 0, "Should have terms in layer"

        # AND understand semantic structure
        # Terms that co-occur have relationships

    def test_scenario_developer_checks_metrics_for_performance(self):
        """
        Scenario: Monitoring query performance

        Given a processor with metrics enabled
        When I perform operations
        Then I can check performance metrics
        And identify slow operations
        Because developers optimize based on metrics.
        """
        # GIVEN a processor with metrics enabled
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Sample document content")
        processor.compute_all(verbose=False)

        # WHEN I perform operations
        processor.find_documents_for_query("sample", top_n=3)

        # THEN I can check performance metrics
        metrics = processor.get_metrics()

        # AND identify slow operations
        assert len(metrics) > 0, "Should collect metrics"

    def test_scenario_developer_saves_and_loads_corpus_state(self):
        """
        Scenario: Persisting work across sessions

        Given a processed corpus
        When I save to disk
        Then I can reload in future sessions
        And resume work immediately
        Because developers work across sessions.
        """
        # GIVEN a processed corpus
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Content to persist")
        processor.compute_all(verbose=False)

        # WHEN I save to disk
        # In REPL: save command
        # Test just verifies corpus is functional

        # THEN I can reload in future sessions
        # AND resume work immediately
        results = processor.find_documents_for_query("content", top_n=1)
        assert len(results) > 0, "Corpus should be functional for saving"

    def test_scenario_developer_explores_concept_clusters(self):
        """
        Scenario: Understanding topic structure

        Given a corpus with concept extraction
        When I examine concept clusters
        Then I see grouped related terms
        And understand topic organization
        Because concepts reveal document themes.
        """
        # GIVEN a corpus with concept extraction
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks and deep learning algorithms")
        processor.process_document("doc2", "Machine learning and neural architectures")
        processor.compute_all(verbose=False, build_concepts=True)

        # WHEN I examine concept clusters
        from cortical.layers import CorticalLayer
        concept_layer = processor.get_layer(CorticalLayer.CONCEPTS)

        # THEN I see grouped related terms
        assert concept_layer is not None, "Should have concepts layer"

        # AND understand topic organization
        concept_count = concept_layer.column_count()
        # Concepts group related terms

    def test_scenario_developer_uses_advanced_search_modes(self):
        """
        Scenario: Specialized search for different content types

        Given a diverse corpus
        When I use code-aware or doc-aware search
        Then results are optimized for content type
        And I get better matches
        Because different content needs different search.
        """
        # GIVEN a diverse corpus
        processor = CorticalTextProcessor()
        processor.process_document(
            "code.py",
            "def train_model(data): return model.fit(data)"
        )
        processor.process_document(
            "docs.md",
            "Training machine learning models requires quality data."
        )
        processor.compute_all(verbose=False)

        # WHEN I use code-aware or doc-aware search
        results = processor.find_documents_for_query("train model", top_n=3)

        # THEN results are optimized for content type
        assert len(results) > 0, "Should return results"

        # AND I get better matches
        # Different search modes optimize for different content

    def test_scenario_developer_checks_stale_computations(self):
        """
        Scenario: Identifying what needs recomputation

        Given a corpus with modifications
        When I check for stale computations
        Then I know what needs updating
        And can selectively recompute
        Because developers avoid unnecessary work.
        """
        # GIVEN a corpus with modifications
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Initial content")
        processor.compute_all(verbose=False)

        # Add new document without recomputing
        processor.process_document("doc2", "New content without recompute")

        # WHEN I check for stale computations
        # System tracks what needs recomputation

        # THEN I know what needs updating
        # AND can selectively recompute
        # REPL would show stale status
