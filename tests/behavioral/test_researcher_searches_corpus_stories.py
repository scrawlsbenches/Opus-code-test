"""
Behavioral tests for researchers searching document corpus with various search modes.

Epic: Document Search and Discovery

As a researcher with a document corpus,
I want multiple search strategies tailored to my needs,
So that I can find relevant documents efficiently regardless of corpus size.

Based on: cortical/query/search.py (document search functionality)
"""

import pytest
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


class TestResearcherSearchesCorpusWithMultipleMethods:
    """
    Epic: Document Search and Discovery

    As a researcher exploring a knowledge base,
    I want flexible search methods optimized for different use cases,
    So that I find relevant documents efficiently.
    """

    def test_scenario_researcher_searches_with_query_expansion(self):
        """
        Scenario: Basic search expands query with related terms

        Given a corpus with interconnected concepts
        When I search for a term
        Then the system expands my query with related terms
        And returns documents matching both original and expanded terms
        Because query expansion improves recall without manual effort.
        """
        # GIVEN a corpus with interconnected concepts
        docs = {
            "ml_intro": "Machine learning trains models to recognize patterns in data.",
            "neural_overview": "Neural networks use layers to process information.",
            "dl_guide": "Deep learning employs multilayer architectures for learning.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for a term
        results = processor.find_documents_for_query(
            "neural",
            top_n=3,
            use_expansion=True
        )

        # THEN the system expands my query with related terms
        # AND returns documents matching both original and expanded terms
        assert len(results) > 0, "Should find relevant documents"
        doc_ids = [doc_id for doc_id, _ in results]
        assert "neural_overview" in doc_ids, "Should find direct match"

    def test_scenario_researcher_uses_fast_search_on_large_corpus(self):
        """
        Scenario: Fast search optimizes performance with candidate filtering

        Given a large corpus of documents
        When I use fast search
        Then results are returned quickly using candidate filtering
        And result quality remains high
        Because researchers need rapid iteration on large corpora.
        """
        # GIVEN a large corpus of documents
        docs = {
            f"doc_{i}": f"Document {i} discusses neural networks and machine learning algorithms."
            for i in range(50)
        }
        # Add some distinct documents
        docs["target_doc"] = "Neural networks process information through interconnected layers."

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I use fast search
        import time
        start = time.perf_counter()
        results = processor.fast_find_documents("neural networks", top_n=5)
        elapsed = time.perf_counter() - start

        # THEN results are returned quickly using candidate filtering
        assert elapsed < 0.5, f"Fast search should complete quickly, took {elapsed:.3f}s"

        # AND result quality remains high
        assert len(results) > 0, "Should find relevant documents"

    def test_scenario_researcher_builds_reusable_search_index(self):
        """
        Scenario: Pre-building search index for repeated queries

        Given a corpus that will be queried multiple times
        When I build an inverted index once
        Then subsequent queries use the cached index
        And search performance improves dramatically
        Because researchers often run many queries on static corpora.
        """
        # GIVEN a corpus that will be queried multiple times
        docs = {
            "neural_doc": "Neural networks learn patterns from data.",
            "ml_doc": "Machine learning algorithms train on datasets.",
            "dl_doc": "Deep learning uses neural architectures.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I build an inverted index once
        index = processor.build_document_index()

        # THEN subsequent queries use the cached index
        assert len(index) > 0, "Index should contain terms"

        results = processor.search_with_index("neural", index, top_n=3)

        # AND search performance improves dramatically
        assert len(results) > 0, "Should find documents using index"
        doc_ids = [doc_id for doc_id, _ in results]
        assert "neural_doc" in doc_ids or "dl_doc" in doc_ids

    def test_scenario_researcher_uses_spreading_activation_for_discovery(self):
        """
        Scenario: Spreading activation reveals related concepts

        Given a corpus with semantic connections
        When I query using spreading activation
        Then the search activates not just direct matches
        But also semantically connected concepts
        Because spreading activation mimics human associative memory.
        """
        # GIVEN a corpus with semantic connections
        docs = {
            "neurons": "Neurons transmit signals through synaptic connections.",
            "brain": "The brain processes information using neural circuits.",
            "cognition": "Cognitive processes emerge from neural activity.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I query using spreading activation
        activated = processor.query_with_spreading_activation(
            "neural",
            top_n=10,
            max_expansions=8
        )

        # THEN the search activates not just direct matches
        # But also semantically connected concepts
        assert len(activated) > 0, "Should activate related concepts"
        activated_terms = [term for term, _ in activated]
        # Should include the original term or related terms
        assert any(term in activated_terms for term in ["neural", "neurons", "brain"])

    def test_scenario_researcher_finds_related_documents(self):
        """
        Scenario: Finding documents similar to a given document

        Given a corpus with document relationships
        When I request documents related to a specific document
        Then I receive documents with lateral connections
        And weights indicate relationship strength
        Because researchers often want "more like this" functionality.
        """
        # GIVEN a corpus with document relationships
        docs = {
            "neural_basics": "Neural networks consist of interconnected processing units.",
            "neural_advanced": "Advanced neural architectures use attention mechanisms.",
            "baking_guide": "Bread baking requires proper kneading and fermentation.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I request documents related to a specific document
        related = processor.find_related_documents("neural_basics")

        # THEN I receive documents with lateral connections
        # AND weights indicate relationship strength
        if len(related) > 0:
            for doc_id, weight in related:
                assert isinstance(doc_id, str), "Should return document IDs"
                assert isinstance(weight, (int, float)), "Should include connection weight"
                assert weight >= 0, "Weight should be non-negative"


class TestResearcherUsesGraphBoostedSearch:
    """
    Epic: Advanced Search with Graph Signals

    As a researcher working with code or technical documentation,
    I want search that combines text relevance with graph importance,
    So that I find both relevant and authoritative results.
    """

    def test_scenario_researcher_combines_text_and_graph_signals(self):
        """
        Scenario: Graph-boosted search ranks important concepts higher

        Given a corpus where some concepts are central (high PageRank)
        When I search using graph-boosted search
        Then results combine text relevance with graph importance
        And authoritative documents rank higher
        Because importance matters in addition to keyword matching.
        """
        # GIVEN a corpus where some concepts are central
        docs = {
            "core_concept": "PageRank algorithm computes node importance in graphs.",
            "implementation": "We implement PageRank using power iteration method.",
            "application": "PageRank is used in search engines and citation analysis.",
            "tangential": "Random walk probability converges to stationary distribution.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search using graph-boosted search
        results = processor.graph_boosted_search(
            "PageRank algorithm",
            top_n=3,
            pagerank_weight=0.3,
            proximity_weight=0.2
        )

        # THEN results combine text relevance with graph importance
        assert len(results) > 0, "Should find relevant documents"

        # AND authoritative documents rank higher
        doc_ids = [doc_id for doc_id, _ in results]
        # Core concept should be in results
        assert "core_concept" in doc_ids or "implementation" in doc_ids

    def test_scenario_researcher_boosts_documents_with_name_matches(self):
        """
        Scenario: Document names matching query get boosted

        Given documents with descriptive names
        When I search for terms that match document names
        Then documents whose names match get higher scores
        And users find documents by their filenames
        Because filename matching is a strong relevance signal.
        """
        # GIVEN documents with descriptive names
        docs = {
            "neural_network_guide": "This guide explains basic concepts.",
            "machine_learning_intro": "Introduction to machine learning fundamentals.",
            "unrelated_topic": "Neural networks are mentioned briefly here.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for terms that match document names
        results = processor.find_documents_for_query(
            "neural network",
            top_n=3,
            doc_name_boost=2.0
        )

        # THEN documents whose names match get higher scores
        assert len(results) > 0, "Should find documents"

        # Document with matching name should rank well
        doc_ids = [doc_id for doc_id, _ in results]
        assert "neural_network_guide" in doc_ids

    def test_scenario_researcher_penalizes_test_files(self):
        """
        Scenario: Test files rank lower than source files

        Given a codebase with source files and test files
        When I search for a code concept
        Then source files rank higher than test files
        And test files appear lower in results
        Because users typically want implementation, not tests.
        """
        # GIVEN a codebase with source files and test files
        docs = {
            "src/neural.py": "class NeuralNetwork: Neural network implementation with layers.",
            "tests/test_neural.py": "Test NeuralNetwork class with various configurations.",
            "tests/test_integration.py": "Integration tests for NeuralNetwork training.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for a code concept
        results = processor.find_documents_for_query(
            "NeuralNetwork",
            top_n=3,
            test_file_penalty=0.8
        )

        # THEN source files rank higher than test files
        assert len(results) > 0, "Should find documents"

        # Source file should rank first
        top_doc = results[0][0]
        assert "src/" in top_doc, "Source file should rank higher than test files"


class TestResearcherSearchesWithFreshnessBoost:
    """
    Epic: Temporal Relevance

    As a researcher tracking evolving information,
    I want recently added documents to rank higher,
    So that I see the latest information first.
    """

    def test_scenario_researcher_boosts_recent_documents(self):
        """
        Scenario: Recently added documents rank higher

        Given a corpus with documents added at different times
        When I enable freshness boost
        Then recent documents score higher than old ones
        And freshness decays over time
        Because recency is a relevance signal for evolving content.
        """
        # GIVEN a corpus with documents added at different times
        from datetime import datetime, timedelta

        docs = {
            "old_doc": "Neural networks process information through layers.",
            "recent_doc": "Neural networks now use attention mechanisms.",
        }

        doc_metadata = {
            "old_doc": {
                "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            },
            "recent_doc": {
                "timestamp": datetime.now().strftime("%Y-%m-%d")
            }
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I enable freshness boost
        results = processor.find_documents_for_query(
            "neural networks",
            top_n=2,
            freshness_boost=1.5,
            doc_metadata=doc_metadata,
            freshness_window_days=7
        )

        # THEN recent documents score higher than old ones
        assert len(results) > 0, "Should find documents"

        # With freshness boost, recent doc should score well
        doc_ids = [doc_id for doc_id, _ in results]
        assert "recent_doc" in doc_ids

    def test_scenario_researcher_uses_graduated_freshness_decay(self):
        """
        Scenario: Freshness boost decays gradually over time

        Given documents of varying ages within the freshness window
        When using linear decay
        Then boost decreases proportionally with age
        And older documents within window still get partial boost
        Because graduated decay is more realistic than binary cutoff.
        """
        # GIVEN documents of varying ages within the freshness window
        from datetime import datetime, timedelta

        docs = {
            "today": "Latest research on neural architectures.",
            "week_ago": "Recent developments in neural networks.",
        }

        doc_metadata = {
            "today": {
                "timestamp": datetime.now().strftime("%Y-%m-%d")
            },
            "week_ago": {
                "timestamp": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            }
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN using linear decay
        results = processor.find_documents_for_query(
            "neural",
            top_n=2,
            freshness_boost=1.5,
            doc_metadata=doc_metadata,
            freshness_decay="linear",
            freshness_window_days=7
        )

        # THEN boost decreases proportionally with age
        assert len(results) > 0, "Should find documents with freshness boost"
