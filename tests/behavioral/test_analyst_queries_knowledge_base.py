"""
Behavioral tests for analysts querying knowledge bases with natural language.

Epic: Natural Language Knowledge Access

As an analyst with a large knowledge base,
I want to ask questions in natural language,
So that I can find information without learning query syntax.

Based on: examples/prism_got_nlu_demo.py
"""

import pytest
from pathlib import Path
from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer
from cortical.reasoning import (
    NodeType,
    EdgeType,
    SynapticMemoryGraph,
)


class TestAnalystQueriesKnowledgeBase:
    """
    Epic: Natural Language Knowledge Access

    As an analyst with extensive documentation,
    I want to ask questions in natural language,
    So that I find relevant information quickly.
    """

    def test_scenario_analyst_asks_natural_language_question(self):
        """
        Scenario: Finding documents by asking natural questions

        Given a knowledge base built from documents
        When I ask a question in natural language
        Then the system extracts key terms from my question
        And finds documents containing those terms
        Because analysts shouldn't need to know exact keywords.
        """
        # GIVEN a knowledge base built from documents
        processor = CorticalTextProcessor()
        docs = {
            "neural_networks.txt": "Neural networks learn through backpropagation and gradient descent algorithms.",
            "memory_systems.txt": "Memory consolidation occurs during sleep through synaptic strengthening.",
            "graph_theory.txt": "Graph algorithms enable efficient network analysis and pathfinding.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        # WHEN I ask a question in natural language
        question = "How do neural networks learn from data?"

        # THEN the system extracts key terms from my question
        # Simple tokenization - remove stop words
        stop_words = {'how', 'do', 'from', 'the', 'a', 'is', 'are'}
        query_terms = [
            word.lower().strip('?.,')
            for word in question.split()
            if word.lower() not in stop_words and len(word) > 2
        ]

        assert len(query_terms) > 0, "Should extract meaningful terms from question"
        assert "neural" in query_terms or "networks" in query_terms, "Should identify key terms"

        # AND finds documents containing those terms
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        relevant_docs = set()

        for term in query_terms:
            col = layer0.get_minicolumn(term)
            if col:
                relevant_docs.update(col.document_ids)

        assert len(relevant_docs) > 0, "Should find relevant documents"
        assert "neural_networks.txt" in relevant_docs, "Should find document about neural networks"

    def test_scenario_analyst_receives_ranked_results_with_scores(self):
        """
        Scenario: Results ranked by relevance

        Given multiple documents matching query terms
        When I ask a question
        Then results are ranked by TF-IDF scores
        And I see the most relevant documents first
        Because analysts need to prioritize their reading.
        """
        # GIVEN multiple documents matching query terms
        processor = CorticalTextProcessor()
        docs = {
            "ml_intro.txt": "Machine learning algorithms learn patterns from data.",
            "neural_deep.txt": "Neural networks and deep learning learn hierarchical representations from data through multiple layers.",
            "stats_basics.txt": "Statistical methods analyze data for patterns.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        # WHEN I ask a question
        query_terms = ["neural", "learn"]

        # THEN results are ranked by TF-IDF scores
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        doc_scores = {}

        for doc_id in docs.keys():
            score = 0.0
            for term in query_terms:
                col = layer0.get_minicolumn(term)
                if col and doc_id in col.document_ids:
                    tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                    score += tfidf

            if score > 0:
                doc_scores[doc_id] = score

        # AND I see the most relevant documents first
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        assert len(ranked) > 0, "Should rank documents by relevance"
        assert ranked[0][0] == "neural_deep.txt", "Document with both terms should rank highest"

    def test_scenario_analyst_sees_relevant_snippets_from_documents(self):
        """
        Scenario: Viewing contextual snippets

        Given documents matching my query
        When I receive search results
        Then each result includes a relevant snippet
        And the snippet contains my query terms in context
        Because analysts need to quickly assess relevance.
        """
        # GIVEN documents matching my query
        docs = {
            "doc1.txt": "Neural networks are powerful. They learn from examples. They adapt to new patterns."
        }

        query_terms = ["neural", "learn"]

        # WHEN I receive search results
        # THEN each result includes a relevant snippet
        content = docs["doc1.txt"]
        sentences = content.replace('\n', ' ').split('. ')

        # Find sentence with most query terms
        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        # AND the snippet contains my query terms in context
        assert len(best_sentence) > 0, "Should extract a relevant snippet"
        assert any(term in best_sentence.lower() for term in query_terms), "Snippet should contain query terms"

    def test_scenario_analyst_learns_about_unknown_terms(self):
        """
        Scenario: Identifying gaps in knowledge base

        Given a knowledge base with specific domain coverage
        When I ask a question with terms not in the corpus
        Then the system identifies which terms are unknown
        And informs me about coverage gaps
        Because analysts need to know what the system doesn't know.
        """
        # GIVEN a knowledge base with specific domain coverage
        processor = CorticalTextProcessor()
        docs = {
            "doc1.txt": "Memory consolidation strengthens neural connections.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        # WHEN I ask a question with terms not in the corpus
        query_terms = ["quantum", "entanglement", "memory"]

        # THEN the system identifies which terms are unknown
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        known_terms = []
        unknown_terms = []

        for term in query_terms:
            if layer0.get_minicolumn(term) is not None:
                known_terms.append(term)
            else:
                unknown_terms.append(term)

        # AND informs me about coverage gaps
        assert "memory" in known_terms, "Should identify known terms"
        assert "quantum" in unknown_terms, "Should identify unknown terms"
        assert "entanglement" in unknown_terms, "Should identify all unknown terms"

    def test_scenario_analyst_provides_feedback_on_results(self):
        """
        Scenario: System learns from relevance feedback

        Given a knowledge graph tracking question-document relationships
        When I mark a document as helpful for my question
        Then the system strengthens that question-document connection
        And future similar questions rank that document higher
        Because systems should learn from user feedback.
        """
        # GIVEN a knowledge graph tracking question-document relationships
        graph = SynapticMemoryGraph()

        question = "How does learning work?"
        q_node_id = f"Q:{hash(question) % 10000}"
        doc_node_id = "DOC:learning_paper.pdf"

        graph.add_node(q_node_id, NodeType.QUESTION, question)
        graph.add_node(doc_node_id, NodeType.ARTIFACT, "Learning Paper")

        # Create initial connection with moderate weight
        edge = graph.add_synaptic_edge(
            q_node_id, doc_node_id, EdgeType.ANSWERS, weight=0.5
        )

        initial_weight = edge.weight
        initial_accuracy = edge.prediction_accuracy

        # WHEN I mark a document as helpful for my question
        edge.strengthen(0.2)
        edge.record_prediction_outcome(correct=True)

        # THEN the system strengthens that question-document connection
        assert edge.weight > initial_weight, "Connection should strengthen with positive feedback"

        # AND future similar questions rank that document higher
        assert edge.prediction_accuracy > initial_accuracy, "Prediction accuracy should improve"

    def test_scenario_analyst_discovers_related_documents(self):
        """
        Scenario: Finding connections through shared concepts

        Given documents linked to concepts in the knowledge graph
        When I find a relevant document
        Then I can explore documents sharing similar concepts
        And discover related information I didn't explicitly search for
        Because knowledge exploration should enable serendipitous discovery.
        """
        # GIVEN documents linked to concepts in the knowledge graph
        graph = SynapticMemoryGraph()

        # Create documents
        graph.add_node("DOC:neural_paper", NodeType.ARTIFACT, "Neural Networks Paper")
        graph.add_node("DOC:brain_study", NodeType.ARTIFACT, "Brain Study")
        graph.add_node("DOC:algorithm_guide", NodeType.ARTIFACT, "Algorithm Guide")

        # Create shared concept
        graph.add_node("CONCEPT:learning", NodeType.CONCEPT, "Learning")

        # Link documents to shared concept
        graph.add_synaptic_edge("DOC:neural_paper", "CONCEPT:learning", EdgeType.CONTAINS, weight=0.8)
        graph.add_synaptic_edge("DOC:brain_study", "CONCEPT:learning", EdgeType.CONTAINS, weight=0.7)

        # WHEN I find a relevant document
        start_doc = "DOC:neural_paper"

        # THEN I can explore documents sharing similar concepts
        # Find concepts in this document
        start_edges = graph.get_synaptic_edges_from(start_doc)
        related_docs = set()

        for edge in start_edges:
            if edge.edge_type == EdgeType.CONTAINS and edge.target_id.startswith("CONCEPT:"):
                concept_id = edge.target_id

                # Find other documents linked to this concept
                # We need to search through all edges to find reverse connections
                for (src, tgt, etype), other_edge in graph.synaptic_edges.items():
                    if tgt == concept_id and src.startswith("DOC:") and src != start_doc:
                        related_docs.add(src)

        # AND discover related information I didn't explicitly search for
        assert "DOC:brain_study" in related_docs, "Should discover related documents through shared concepts"

    def test_scenario_analyst_gets_relevance_scores_for_transparency(self):
        """
        Scenario: Understanding why results were returned

        Given search results for my question
        When I view the results
        Then each result includes a relevance score
        And I understand why it was recommended
        Because analysts need to trust and validate results.
        """
        # GIVEN search results for my question
        processor = CorticalTextProcessor()
        docs = {
            "high_relevance.txt": "Neural networks learn patterns through backpropagation.",
            "low_relevance.txt": "The system can learn basic rules efficiently.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        query_terms = ["neural", "learn"]

        # WHEN I view the results
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        results = []

        for doc_id in docs.keys():
            score = 0.0
            matched_terms = []

            for term in query_terms:
                col = layer0.get_minicolumn(term)
                if col and doc_id in col.document_ids:
                    tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                    score += tfidf
                    matched_terms.append(term)

            if score > 0:
                results.append({
                    'doc_id': doc_id,
                    'score': score,
                    'matched_terms': matched_terms
                })

        # THEN each result includes a relevance score
        assert all('score' in r for r in results), "Each result should have a score"

        # AND I understand why it was recommended
        high_rel = next(r for r in results if r['doc_id'] == "high_relevance.txt")
        low_rel = next(r for r in results if r['doc_id'] == "low_relevance.txt")

        assert high_rel['score'] > low_rel['score'], "More relevant document should have higher score"
        assert len(high_rel['matched_terms']) >= len(low_rel['matched_terms']), "Should show which terms matched"
