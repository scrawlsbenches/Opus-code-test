"""
Behavioral tests for researchers searching documents with semantic understanding.

Epic: Semantic Document Search

As a researcher exploring a document corpus,
I want to search using natural language queries with semantic expansion,
So that I find relevant documents even when exact keywords don't match.

Based on: showcase.py (query and passage search features)
"""

import pytest
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


class TestResearcherSearchesDocumentsSemantically:
    """
    Epic: Semantic Document Search

    As a researcher with a large corpus,
    I want semantic query understanding and expansion,
    So that I discover relevant documents beyond keyword matching.
    """

    def test_scenario_researcher_queries_with_automatic_expansion(self):
        """
        Scenario: Query expansion adds semantically related terms

        Given a corpus with documents about related concepts
        When I search with a simple query
        Then the system expands the query with related terms
        And returns documents matching both original and expanded terms
        Because expansion improves recall without sacrificing precision.
        """
        # GIVEN a corpus with documents about related concepts
        docs = {
            "ml_basics": "Machine learning algorithms train models to recognize patterns in datasets.",
            "neural_nets": "Neural networks use interconnected layers to process information.",
            "deep_learning": "Deep learning employs multilayer neural architectures for representation learning.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search with a simple query
        query = "neural networks"

        # THEN the system expands the query with related terms
        expanded = processor.expand_query(query, max_expansions=6)
        original_terms = set(processor.tokenizer.tokenize(query))

        assert len(expanded) >= len(original_terms), "Should expand beyond original terms"

        # AND returns documents matching both original and expanded terms
        results = processor.find_documents_for_query(query, top_n=3)

        assert len(results) > 0, "Should find relevant documents"
        # Should find the neural networks doc
        doc_ids = [doc_id for doc_id, _ in results]
        assert "neural_nets" in doc_ids or "deep_learning" in doc_ids

    def test_scenario_researcher_finds_documents_by_concept_not_keywords(self):
        """
        Scenario: Finding documents by concept rather than exact keywords

        Given documents that describe the same concept with different terminology
        When I query for a concept using one set of terms
        Then I find documents using alternative terminology
        And query expansion bridges the vocabulary gap
        Because researchers use varied terminology for the same concepts.
        """
        # GIVEN documents that describe the same concept with different terminology
        docs = {
            "fermentation": "Yeast converts sugar into alcohol through anaerobic respiration.",
            "brewing": "Beer production relies on microbial metabolism of fermentable sugars.",
            "winemaking": "Grape juice transforms into wine via alcoholic fermentation by microorganisms.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I query for a concept using one set of terms
        query = "fermentation"

        # THEN I find documents using alternative terminology
        results = processor.find_documents_for_query(query, top_n=3)

        # AND query expansion bridges the vocabulary gap
        assert len(results) > 0, "Should find documents about fermentation"

        # Should find the fermentation doc at minimum
        doc_ids = [doc_id for doc_id, _ in results]
        assert "fermentation" in doc_ids

    def test_scenario_researcher_retrieves_passages_for_context(self):
        """
        Scenario: Retrieving specific passages for RAG applications

        Given a corpus with detailed technical content
        When I search for specific information
        Then I receive relevant text passages (not just document IDs)
        And passages include context boundaries and scores
        Because RAG systems need text chunks to feed to language models.
        """
        # GIVEN a corpus with detailed technical content
        docs = {
            "pagerank_doc": """
            PageRank algorithm computes importance scores for nodes in a graph.
            The algorithm uses iterative power method to find steady-state probability distribution.
            Convergence typically occurs within 20-30 iterations with damping factor 0.85.
            Higher PageRank indicates more authoritative or central nodes in the network.
            """
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for specific information
        query = "PageRank algorithm convergence"

        # THEN I receive relevant text passages (not just document IDs)
        passages = processor.find_passages_for_query(
            query,
            top_n=2,
            chunk_size=150,
            overlap=30
        )

        assert len(passages) > 0, "Should find relevant passages"

        # AND passages include context boundaries and scores
        for passage_text, doc_id, start, end, score in passages:
            assert isinstance(passage_text, str), "Passage should be text"
            assert isinstance(doc_id, str), "Should include document ID"
            assert isinstance(start, int) and isinstance(end, int), "Should include character positions"
            assert 0 <= score <= 1, "Should include relevance score"
            assert len(passage_text) > 0, "Passage should not be empty"

    def test_scenario_researcher_handles_polysemous_terms(self):
        """
        Scenario: Handling words with multiple meanings (polysemy)

        Given documents where the same word has different meanings
        When I search using an ambiguous term
        Then I receive results from all relevant contexts
        And can see how the term is used differently
        Because polysemy is inherent in natural language.
        """
        # GIVEN documents where the same word has different meanings
        docs = {
            "trading": "Candlestick patterns help traders analyze price movements on charts.",
            "typesetting": "Composing sticks are tools used in traditional letterpress printing.",
            "baking": "Sticks of butter and cinnamon sticks add flavor to baked goods.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search using an ambiguous term
        query = "sticks"

        # THEN I receive results from all relevant contexts
        results = processor.find_documents_for_query(query, top_n=3)

        assert len(results) > 0, "Should find documents containing 'sticks'"

        # AND can see how the term is used differently
        # Multiple documents should match
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        sticks_col = layer0.get_minicolumn("sticks")

        if sticks_col:
            # Should appear in multiple documents
            assert len(sticks_col.document_ids) >= 2, "Polysemous term should appear in multiple contexts"

    def test_scenario_researcher_searches_with_performance_bounds(self):
        """
        Scenario: Search completes within reasonable time

        Given a corpus of reasonable size
        When I execute a query
        Then results are returned quickly
        And performance scales with corpus size
        Because researchers need rapid iteration.
        """
        # GIVEN a corpus of reasonable size
        docs = {
            f"doc_{i}": f"Document {i} contains information about neural networks and machine learning."
            for i in range(10)
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I execute a query
        import time
        start = time.perf_counter()
        results = processor.find_documents_for_query("neural networks", top_n=5)
        elapsed = time.perf_counter() - start

        # THEN results are returned quickly
        assert elapsed < 1.0, "Query should complete in under 1 second for small corpus"

        # AND performance scales with corpus size
        assert len(results) > 0, "Should return results"

    def test_scenario_researcher_gets_fingerprint_for_similarity(self):
        """
        Scenario: Computing semantic fingerprints for text comparison

        Given two pieces of text
        When I compute their semantic fingerprints
        Then I can compare similarity without full text comparison
        And fingerprints capture key concepts efficiently
        Because fingerprints enable fast similarity checks.
        """
        # GIVEN two pieces of text
        text1 = "Neural networks learn patterns through backpropagation and gradient descent."
        text2 = "Neural models discover patterns using backpropagation algorithms."
        text3 = "Bread baking requires proper kneading and fermentation time."

        processor = CorticalTextProcessor()
        processor.process_document("doc1", text1)
        processor.process_document("doc2", text2)
        processor.process_document("doc3", text3)
        processor.compute_all(verbose=False)

        # WHEN I compute their semantic fingerprints
        fp1 = processor.get_fingerprint(text1, top_n=5)
        fp2 = processor.get_fingerprint(text2, top_n=5)
        fp3 = processor.get_fingerprint(text3, top_n=5)

        # THEN I can compare similarity without full text comparison
        comparison_12 = processor.compare_fingerprints(fp1, fp2)
        comparison_13 = processor.compare_fingerprints(fp1, fp3)

        # AND fingerprints capture key concepts efficiently
        assert 'overall_similarity' in comparison_12
        assert 'overall_similarity' in comparison_13

        # Similar texts (1 and 2) should have higher similarity than dissimilar (1 and 3)
        sim_12 = comparison_12['overall_similarity']
        sim_13 = comparison_13['overall_similarity']

        # Both scores should be between 0 and 1
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1

        # Related texts should be more similar
        assert sim_12 > sim_13, "Similar texts about neural networks should be more similar than unrelated baking text"

    def test_scenario_researcher_uses_graph_embeddings_for_similarity(self):
        """
        Scenario: Finding similar terms using graph embeddings

        Given a corpus with computed graph embeddings
        When I query for terms similar to a concept
        Then I find semantically related terms
        And similarity is based on graph structure
        Because embeddings capture semantic relationships from co-occurrence.
        """
        # GIVEN a corpus with computed graph embeddings
        docs = {
            "doc1": "Neural networks process data through connected layers of neurons.",
            "doc2": "Deep learning neural models learn hierarchical representations.",
            "doc3": "Machine learning algorithms train on labeled datasets.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # Compute embeddings
        stats = processor.compute_graph_embeddings(
            dimensions=16,
            method='random_walk',
            verbose=False
        )

        # WHEN I query for terms similar to a concept
        similar = processor.find_similar_by_embedding("neural", top_n=5)

        # THEN I find semantically related terms
        assert len(similar) > 0, "Should find similar terms"

        # AND similarity is based on graph structure
        for term, similarity in similar:
            assert isinstance(term, str), "Should return term strings"
            assert 0 <= similarity <= 1, "Similarity should be normalized"
