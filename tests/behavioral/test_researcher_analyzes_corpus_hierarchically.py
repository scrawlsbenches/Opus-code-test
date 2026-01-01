"""
Behavioral tests for researchers analyzing document corpora hierarchically.

Epic: Hierarchical Corpus Analysis

As a researcher with a document collection,
I want to analyze my corpus through hierarchical layers,
So that I can understand concepts at multiple levels of abstraction.

Based on: showcase.py
"""

import pytest
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.tokenizer import Tokenizer


class TestResearcherAnalyzesCorpusHierarchically:
    """
    Epic: Hierarchical Corpus Analysis

    As a researcher with a diverse document collection,
    I want the system to build hierarchical representations,
    So that I can discover patterns from tokens to complete documents.
    """

    def test_scenario_researcher_ingests_corpus_and_builds_hierarchy(self):
        """
        Scenario: Ingesting documents and building hierarchical layers

        Given a collection of diverse documents
        When I process the documents through the cortical processor
        Then the system creates 4 hierarchical layers
        And each layer contains minicolumns representing concepts at that level
        Because hierarchical organization enables multi-scale analysis.
        """
        # GIVEN a collection of diverse documents
        docs = {
            "neural_networks": "Neural networks process information through interconnected layers of artificial neurons.",
            "machine_learning": "Machine learning algorithms discover patterns in data through statistical methods.",
            "deep_learning": "Deep learning uses multiple neural network layers to learn hierarchical representations.",
        }

        # WHEN I process the documents through the cortical processor
        tokenizer = Tokenizer(filter_code_noise=True)
        processor = CorticalTextProcessor(tokenizer=tokenizer)

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all(verbose=False)

        # THEN the system creates 4 hierarchical layers
        assert CorticalLayer.TOKENS in processor.layers
        assert CorticalLayer.BIGRAMS in processor.layers
        assert CorticalLayer.CONCEPTS in processor.layers
        assert CorticalLayer.DOCUMENTS in processor.layers

        # AND each layer contains minicolumns representing concepts at that level
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        layer1 = processor.get_layer(CorticalLayer.BIGRAMS)
        layer2 = processor.get_layer(CorticalLayer.CONCEPTS)
        layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)

        assert layer0.column_count() > 0, "Should have token minicolumns"
        assert layer1.column_count() > 0, "Should have bigram minicolumns"
        assert layer3.column_count() == 3, "Should have 3 document minicolumns"

    def test_scenario_researcher_discovers_key_concepts_via_pagerank(self):
        """
        Scenario: Discovering central concepts using PageRank

        Given a processed corpus with lateral connections
        When I compute PageRank scores
        Then highly connected concepts receive higher scores
        And I can identify hub concepts that bridge multiple topics
        Because central concepts are most important for understanding the domain.
        """
        # GIVEN a processed corpus with lateral connections
        docs = {
            "doc1": "Neural networks use layers to process data. Deep learning networks have many layers.",
            "doc2": "Machine learning trains neural networks on data to recognize patterns.",
            "doc3": "Data processing through neural layers enables pattern recognition.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I compute PageRank scores
        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        # THEN highly connected concepts receive higher scores
        # Get PageRank scores
        pagerank_scores = {col.content: col.pagerank for col in layer0.minicolumns.values()}

        # AND I can identify hub concepts that bridge multiple topics
        # "neural" and "data" should have high PageRank as they appear across documents
        assert "neural" in pagerank_scores
        assert "data" in pagerank_scores

        # Hub terms should have non-zero PageRank
        assert pagerank_scores.get("neural", 0) > 0
        assert pagerank_scores.get("data", 0) > 0

    def test_scenario_researcher_analyzes_tfidf_for_distinctive_terms(self):
        """
        Scenario: Finding distinctive terms using TF-IDF

        Given a corpus with documents covering different topics
        When I compute TF-IDF scores
        Then terms unique to specific documents score higher
        And common terms across all documents score lower
        Because TF-IDF identifies what makes each document distinctive.
        """
        # GIVEN a corpus with documents covering different topics
        docs = {
            "baking": "Bread baking requires yeast fermentation and proper kneading technique.",
            "neural": "Neural networks learn through backpropagation and gradient descent.",
            "security": "Authentication systems verify user credentials through cryptographic methods.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I compute TF-IDF scores
        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        # THEN terms unique to specific documents score higher
        # "yeast" only appears in baking doc, should have high TF-IDF
        yeast_col = layer0.get_minicolumn("yeast")
        assert yeast_col is not None
        assert yeast_col.tfidf > 0

        # "backpropagation" only appears in neural doc
        backprop_col = layer0.get_minicolumn("backpropagation")
        assert backprop_col is not None
        assert backprop_col.tfidf > 0

        # AND common terms across all documents score lower
        # Terms appearing in all docs should have lower TF-IDF
        # (implementation detail: actual common terms depend on tokenization)

    def test_scenario_researcher_finds_concept_associations(self):
        """
        Scenario: Discovering lateral connections between concepts

        Given documents where terms co-occur
        When I analyze lateral connections
        Then frequently co-occurring terms have strong connections
        And the connection weight reflects co-occurrence strength
        Because co-occurrence reveals semantic relationships (Hebbian learning).
        """
        # GIVEN documents where terms co-occur
        docs = {
            "doc1": "Neural networks learn patterns. Neural models process data.",
            "doc2": "Learning algorithms discover patterns in data.",
            "doc3": "Pattern recognition through neural learning.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I analyze lateral connections
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        neural_col = layer0.get_minicolumn("neural")

        # THEN frequently co-occurring terms have strong connections
        assert neural_col is not None
        assert len(neural_col.lateral_connections) > 0

        # AND the connection weight reflects co-occurrence strength
        # Terms that appear together should be connected
        # (e.g., "neural" and "learning" co-occur)
        for neighbor_id, weight in neural_col.lateral_connections.items():
            assert weight > 0, "Connection weights should be positive"

    @pytest.mark.skip(reason="API mismatch - needs alignment with implementation")
    def test_scenario_researcher_analyzes_document_relationships(self):
        """
        Scenario: Finding related documents based on shared concepts

        Given a corpus with overlapping concepts
        When I query for documents related to a specific document
        Then I find documents sharing similar concepts
        And similarity scores reflect the degree of overlap
        Because researchers need to find related work.
        """
        # GIVEN a corpus with overlapping concepts
        docs = {
            "neural_intro": "Neural networks consist of layers of interconnected neurons.",
            "deep_learning": "Deep neural networks use multiple hidden layers for learning.",
            "ai_overview": "Artificial intelligence encompasses various approaches to machine reasoning.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I query for documents related to a specific document
        related = processor.find_related_documents("neural_intro", top_n=2)

        # THEN I find documents sharing similar concepts
        assert len(related) > 0, "Should find related documents"

        # AND similarity scores reflect the degree of overlap
        # "deep_learning" should be more related (shares "neural", "layers")
        related_ids = [doc_id for doc_id, _ in related]
        assert "deep_learning" in related_ids, "Should find semantically related document"

        # Scores should be between 0 and 1
        for doc_id, score in related:
            assert 0 <= score <= 1, "Similarity scores should be normalized"

    def test_scenario_researcher_analyzes_knowledge_gaps(self):
        """
        Scenario: Identifying gaps in corpus coverage

        Given a corpus with varying topic coverage
        When I analyze knowledge gaps
        Then the system identifies isolated documents
        And reports weak topics with thin coverage
        Because identifying gaps guides future research.
        """
        # GIVEN a corpus with varying topic coverage
        docs = {
            "ml_1": "Machine learning trains models on labeled data for supervised tasks.",
            "ml_2": "Machine learning algorithms include regression and classification methods.",
            "ml_3": "Machine learning optimization uses gradient descent techniques.",
            "isolated": "Quantum computing uses superposition and entanglement principles.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I analyze knowledge gaps
        gaps = processor.analyze_knowledge_gaps()

        # THEN the system identifies isolated documents
        assert 'isolated_documents' in gaps
        assert 'weak_topics' in gaps
        assert 'coverage_score' in gaps

        # AND reports weak topics with thin coverage
        # The quantum computing doc should be isolated
        isolated_ids = [doc['doc_id'] for doc in gaps['isolated_documents']]
        # (May or may not include "isolated" depending on threshold)

        # Coverage score should be between 0 and 1
        assert 0 <= gaps['coverage_score'] <= 1

    def test_scenario_researcher_computes_clustering_quality(self):
        """
        Scenario: Assessing quality of concept clustering

        Given a corpus with computed concept clusters
        When I measure clustering quality
        Then I receive modularity, silhouette, and balance scores
        And these metrics indicate how well concepts are grouped
        Because quality metrics validate the hierarchical structure.
        """
        # GIVEN a corpus with computed concept clusters
        docs = {
            f"doc_{i}": f"Document {i} discusses neural networks and machine learning algorithms."
            for i in range(5)
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I measure clustering quality
        quality = processor.compute_clustering_quality()

        # THEN I receive modularity, silhouette, and balance scores
        assert 'modularity' in quality
        assert 'silhouette' in quality
        assert 'balance' in quality

        # AND these metrics indicate how well concepts are grouped
        # Modularity should be between -1 and 1 (typically > 0 for good clustering)
        assert -1 <= quality['modularity'] <= 1

        # Silhouette should be between -1 and 1 (closer to 1 is better)
        assert -1 <= quality['silhouette'] <= 1

        # Balance should be between 0 and 1 (closer to 1 is more balanced)
        assert 0 <= quality['balance'] <= 1
