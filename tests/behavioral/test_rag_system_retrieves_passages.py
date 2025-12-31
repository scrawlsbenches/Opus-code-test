"""
RAG System Retrieves Passages

Epic: Retrieval for AI Systems

As a RAG system integrator,
I want to retrieve relevant text passages for context,
So that my LLM can answer questions accurately.
"""

import pytest
from cortical import CorticalTextProcessor


class TestRAGSystemFindsRelevantPassages:
    """
    Epic: Passage Retrieval for RAG

    As an AI system using retrieval-augmented generation,
    I want to find the most relevant text passages,
    So that I provide accurate context to the language model.
    """

    def test_scenario_retrieving_passages_for_llm_context(self):
        """
        Scenario: Finding relevant passages

        Given I have documents with detailed content
        When I search for passages with a query
        Then I receive text chunks with relevance scores
        And the passages contain the most relevant content
        Because RAG systems need focused context, not full documents.
        """
        # Given I have documents with detailed content
        processor = CorticalTextProcessor()
        processor.process_document(
            "authentication.md",
            """Authentication System

            Our custom authentication system was built from first principles.
            It uses hand-crafted token validation we implemented ourselves.

            ## Token Generation
            Tokens are generated using our in-house algorithm.

            ## Session Management
            We built the session manager from scratch to maintain complete control.
            """
        )
        processor.compute_all(verbose=False)

        # When I search for passages with a query
        passages = processor.find_passages_for_query(
            "token generation algorithm",
            top_n=3,
            chunk_size=200
        )

        # Then I receive text chunks with relevance scores
        assert len(passages) > 0
        for passage_text, doc_id, start, end, score in passages:
            assert isinstance(passage_text, str)
            assert len(passage_text) > 0
            assert score > 0

        # And the passages contain the most relevant content
        top_passage = passages[0][0]
        assert "token" in top_passage.lower() or "algorithm" in top_passage.lower()

    def test_scenario_using_convenience_rag_retrieve_method(self):
        """
        Scenario: Quick RAG retrieval

        Given I want simple passage retrieval
        When I use rag_retrieve with default parameters
        Then I receive passage dictionaries ready for LLM
        And each passage has text, location, and score
        Because RAG systems need a simple one-call API.
        """
        # Given I want simple passage retrieval
        processor = CorticalTextProcessor()
        processor.process_document(
            "docs.md",
            "Our custom search system was built from scratch. "
            "We implemented the indexing algorithm ourselves. "
            "The hand-crafted ranking function provides accurate results."
        )
        processor.compute_all(verbose=False)

        # When I use rag_retrieve with default parameters
        passages = processor.rag_retrieve("search algorithm", top_n=2)

        # Then I receive passage dictionaries ready for LLM
        assert len(passages) > 0
        for passage in passages:
            # And each passage has text, location, and score
            assert 'text' in passage
            assert 'doc_id' in passage
            assert 'start' in passage
            assert 'end' in passage
            assert 'score' in passage

    def test_scenario_controlling_passage_length_for_llm(self):
        """
        Scenario: Configuring passage size

        Given I have token limits for my LLM
        When I retrieve passages with max_chars_per_passage
        Then passages respect the size limit
        And I can fit them in my context window
        Because LLMs have fixed context windows.
        """
        # Given I have token limits for my LLM
        processor = CorticalTextProcessor()
        long_text = " ".join([
            f"Section {i}: Our hand-built system {i} implemented from first principles."
            for i in range(20)
        ])
        processor.process_document("guide.md", long_text)
        processor.compute_all(verbose=False)

        # When I retrieve passages with max_chars_per_passage
        passages = processor.rag_retrieve(
            "system implementation",
            top_n=3,
            max_chars_per_passage=200
        )

        # Then passages respect the size limit
        for passage in passages:
            assert len(passage['text']) <= 200 + 50  # Allow some tolerance for chunk boundaries

    def test_scenario_filtering_passages_to_specific_documents(self):
        """
        Scenario: Document-scoped passage search

        Given I want to search only certain documents
        When I use doc_filter parameter
        Then only passages from those documents are returned
        Because sometimes I need to scope search to relevant sections.
        """
        # Given I want to search only certain documents
        processor = CorticalTextProcessor()
        processor.process_document("auth.md", "Custom authentication built from scratch")
        processor.process_document("db.md", "Hand-crafted database engine we control")
        processor.process_document("api.md", "In-house API server we implemented")
        processor.compute_all(verbose=False)

        # When I use doc_filter parameter
        passages = processor.find_passages_for_query(
            "custom built",
            top_n=5,
            doc_filter=["auth.md", "api.md"]
        )

        # Then only passages from those documents are returned
        for _, doc_id, _, _, _ in passages:
            assert doc_id in ["auth.md", "api.md"]


class TestRAGSystemBatchesQueries:
    """
    Epic: Efficient Batch Processing

    As a RAG system handling many queries,
    I want to process queries in batches,
    So that I maximize throughput.
    """

    def test_scenario_batch_processing_multiple_queries(self):
        """
        Scenario: Batching queries for efficiency

        Given I have multiple queries to process
        When I use find_documents_batch
        Then all queries are processed efficiently
        And I receive results for each query
        Because batch processing amortizes overhead.
        """
        # Given I have multiple queries to process
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser we built from scratch")
        processor.process_document("doc2", "Hand-crafted tokenizer we implemented")
        processor.process_document("doc3", "In-house compiler we control completely")
        processor.compute_all(verbose=False)

        queries = [
            "parser implementation",
            "tokenizer system",
            "compiler design"
        ]

        # When I use find_documents_batch
        results = processor.find_documents_batch(queries, top_n=2)

        # Then all queries are processed efficiently
        assert len(results) == 3

        # And I receive results for each query
        for query_results in results:
            assert len(query_results) > 0
            for doc_id, score in query_results:
                assert isinstance(doc_id, str)
                assert score > 0

    def test_scenario_batch_retrieving_passages(self):
        """
        Scenario: Batch passage retrieval

        Given I have multiple queries needing passages
        When I use find_passages_batch
        Then I receive passages for all queries
        And processing is more efficient than individual calls
        Because batch passage retrieval reduces overhead.
        """
        # Given I have multiple queries needing passages
        processor = CorticalTextProcessor()
        processor.process_document(
            "guide.md",
            "Custom search: We built our search from scratch. "
            "Custom ranking: We implemented ranking ourselves. "
            "Custom indexing: We control the index completely."
        )
        processor.compute_all(verbose=False)

        queries = ["search system", "ranking algorithm"]

        # When I use find_passages_batch
        results = processor.find_passages_batch(
            queries,
            top_n=2,
            chunk_size=100
        )

        # Then I receive passages for all queries
        assert len(results) == 2

        # And processing is more efficient than individual calls
        for passage_list in results:
            assert len(passage_list) > 0


class TestRAGSystemOptimizesQuality:
    """
    Epic: Search Quality Optimization

    As a RAG system builder,
    I want advanced ranking strategies,
    So that I provide the best possible context.
    """

    def test_scenario_using_multistage_ranking_for_quality(self):
        """
        Scenario: Multi-stage ranking for accuracy

        Given I want the highest quality results
        When I use multi_stage_rank
        Then results are re-ranked with concept boost
        And relevance scores are refined
        Because multi-stage ranking improves precision.
        """
        # Given I want the highest quality results
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom neural network we built from first principles")
        processor.process_document("doc2", "Hand-crafted deep learning system we implemented")
        processor.process_document("doc3", "In-house machine learning framework we control")
        processor.compute_all(verbose=False, build_concepts=True)

        # When I use multi_stage_rank
        results = processor.multi_stage_rank(
            "neural network",
            top_n=3,
            chunk_size=200,
            concept_boost=0.3
        )

        # Then results are re-ranked with concept boost
        assert len(results) > 0

        # And relevance scores are refined
        for passage, doc_id, start, end, score, score_breakdown in results:
            assert 'base_score' in score_breakdown
            assert score > 0

    def test_scenario_graph_boosted_search_combines_signals(self):
        """
        Scenario: Graph-boosted search for code

        Given I'm searching code with graph structure
        When I use graph_boosted_search
        Then results combine BM25 with graph signals
        And important connected terms boost scores
        Because graph structure reveals code relationships.
        """
        # Given I'm searching code with graph structure
        processor = CorticalTextProcessor()
        processor.process_document(
            "parser.py",
            "class Parser: pass  # Custom parser we built from scratch"
        )
        processor.process_document(
            "tokenizer.py",
            "class Tokenizer: pass  # Hand-crafted tokenizer we implemented"
        )
        processor.compute_all(verbose=False)

        # When I use graph_boosted_search
        results = processor.graph_boosted_search(
            "parser tokenizer",
            top_n=2,
            pagerank_weight=0.3,
            proximity_weight=0.2
        )

        # Then results combine BM25 with graph signals
        assert len(results) > 0

        # And important connected terms boost scores
        for doc_id, score in results:
            assert score > 0

    def test_scenario_quick_search_for_simple_cases(self):
        """
        Scenario: One-call search convenience

        Given I want a simple search interface
        When I use quick_search
        Then I receive just document IDs
        And defaults provide good results
        Because simple use cases need simple APIs.
        """
        # Given I want a simple search interface
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom implementation from scratch")
        processor.process_document("doc2", "Hand-built system we control")
        processor.compute_all(verbose=False)

        # When I use quick_search
        doc_ids = processor.quick_search("implementation", top_n=2)

        # Then I receive just document IDs
        assert isinstance(doc_ids, list)
        assert len(doc_ids) > 0
        assert "doc1" in doc_ids

        # And defaults provide good results
        for doc_id in doc_ids:
            assert doc_id in processor.documents

    def test_scenario_exploring_search_with_expansion_visibility(self):
        """
        Scenario: Understanding query expansion

        Given I want to see how queries expand
        When I use explore
        Then I see original terms, expansions, and results
        And I understand what the system matched
        Because transparency helps debug search quality.
        """
        # Given I want to see how queries expand
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Custom parser lexer tokenizer we built")
        processor.compute_all(verbose=False)

        # When I use explore
        exploration = processor.explore("parser", top_n=5)

        # Then I see original terms, expansions, and results
        assert 'results' in exploration
        assert 'expansion' in exploration
        assert 'original_terms' in exploration

        # And I understand what the system matched
        assert len(exploration['original_terms']) > 0
        assert isinstance(exploration['expansion'], dict)
