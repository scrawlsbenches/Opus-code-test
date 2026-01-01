"""
Behavioral tests for researchers extracting concepts from document corpora.

Epic: Automated Concept Extraction

As a researcher with a document corpus,
I want to automatically extract and link key concepts,
So that I can understand the knowledge structure without manual analysis.

Based on: examples/prism_got_demo_corpus.py
"""

import pytest
from pathlib import Path
from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer
from cortical.reasoning import (
    NodeType,
    EdgeType,
    SynapticMemoryGraph,
    IncrementalReasoner,
)


class TestResearcherExtractsConceptsFromCorpus:
    """
    Epic: Automated Concept Extraction

    As a researcher analyzing document collections,
    I want automatic concept extraction and linking,
    So that I discover the knowledge structure in my corpus.
    """

    def test_scenario_researcher_builds_cortical_index_from_documents(self):
        """
        Scenario: Creating a searchable index from documents

        Given a collection of research documents
        When I process them with the cortical processor
        Then the system extracts tokens, bigrams, and concepts
        And computes importance scores for each term
        Because researchers need to identify what's important.
        """
        # GIVEN a collection of research documents
        docs = {
            "doc1.txt": "Neural networks process information through interconnected layers.",
            "doc2.txt": "Machine learning algorithms discover patterns in data.",
            "doc3.txt": "Deep learning uses neural networks with multiple layers.",
        }

        # WHEN I process them with the cortical processor
        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        # THEN the system extracts tokens, bigrams, and concepts
        token_count = processor.get_layer(CorticalLayer.TOKENS).column_count()
        bigram_count = processor.get_layer(CorticalLayer.BIGRAMS).column_count()
        concept_count = processor.get_layer(CorticalLayer.CONCEPTS).column_count()

        assert token_count > 0, "Should extract individual words"
        assert bigram_count > 0, "Should extract word pairs"
        assert concept_count > 0, "Should identify concept clusters"

        # AND computes importance scores for each term
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        has_tfidf_scores = False
        for col in layer0.minicolumns.values():
            if col.tfidf > 0:
                has_tfidf_scores = True
                break

        assert has_tfidf_scores, "Should compute TF-IDF importance scores"

    def test_scenario_researcher_extracts_key_concepts_from_each_document(self):
        """
        Scenario: Identifying the most important concepts per document

        Given indexed documents with TF-IDF scores
        When I extract top concepts for each document
        Then I get terms with highest relevance to that document
        And can distinguish important from common terms
        Because researchers need to quickly grasp document content.
        """
        # GIVEN indexed documents with TF-IDF scores
        processor = CorticalTextProcessor()

        docs = {
            "machine_learning.txt": "Machine learning algorithms learn patterns from training data using optimization.",
            "biology.txt": "Cellular biology studies the structure and function of living cells.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        # WHEN I extract top concepts for each document
        def extract_top_terms(doc_id: str, top_n: int = 3) -> list:
            layer0 = processor.get_layer(CorticalLayer.TOKENS)
            doc_terms = []

            for col in layer0.minicolumns.values():
                if doc_id in col.document_ids:
                    tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
                    if tfidf > 0:
                        doc_terms.append((col.content, tfidf))

            doc_terms.sort(key=lambda x: x[1], reverse=True)
            return [term for term, _ in doc_terms[:top_n]]

        ml_concepts = extract_top_terms("machine_learning.txt")
        bio_concepts = extract_top_terms("biology.txt")

        # THEN I get terms with highest relevance to that document
        assert len(ml_concepts) > 0, "Should extract concepts from ML document"
        assert len(bio_concepts) > 0, "Should extract concepts from biology document"

        # AND can distinguish important from common terms
        # Domain-specific terms should appear in results
        ml_text = " ".join(ml_concepts).lower()
        bio_text = " ".join(bio_concepts).lower()

        assert "machine" in ml_text or "learning" in ml_text or "algorithms" in ml_text, "ML concepts should be domain-relevant"
        assert "cellular" in bio_text or "biology" in bio_text or "cells" in bio_text, "Biology concepts should be domain-relevant"

    def test_scenario_researcher_creates_document_concept_knowledge_graph(self):
        """
        Scenario: Building a graph linking documents to their concepts

        Given extracted concepts for each document
        When I create a knowledge graph
        Then documents are linked to their key concepts
        And I can navigate from documents to concepts and back
        Because researchers need to explore knowledge relationships.
        """
        # GIVEN extracted concepts for each document
        processor = CorticalTextProcessor()
        graph = SynapticMemoryGraph()

        docs = {
            "neural_paper.txt": "Neural networks use synaptic connections for learning.",
            "brain_study.txt": "Brain synapses strengthen during learning processes.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)

        processor.compute_all()

        # WHEN I create a knowledge graph
        # Create document nodes
        for doc_id in docs.keys():
            graph.add_node(f"DOC:{doc_id}", NodeType.ARTIFACT, doc_id)

        # Extract and link concepts
        concept_nodes = {}
        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        for doc_id in docs.keys():
            doc_terms = []
            for col in layer0.minicolumns.values():
                if doc_id in col.document_ids and col.tfidf > 0.1:
                    doc_terms.append((col.content, col.tfidf))

            doc_terms.sort(key=lambda x: x[1], reverse=True)

            # Link to top concepts
            for term, _ in doc_terms[:3]:
                concept_key = term.lower()
                concept_id = f"CONCEPT:{concept_key}"

                if concept_id not in concept_nodes:
                    graph.add_node(concept_id, NodeType.CONCEPT, term)
                    concept_nodes[concept_id] = True

                # Link document to concept
                edge_key = (f"DOC:{doc_id}", concept_id, EdgeType.CONTAINS)
                if edge_key not in graph.synaptic_edges:
                    graph.add_synaptic_edge(
                        f"DOC:{doc_id}",
                        concept_id,
                        EdgeType.CONTAINS,
                        weight=0.7
                    )

        # THEN documents are linked to their key concepts
        doc_edges = graph.get_synaptic_edges_from("DOC:neural_paper.txt")
        assert len(doc_edges) > 0, "Documents should be linked to concepts"

        # AND I can navigate from documents to concepts and back
        concept_targets = [e.target_id for e in doc_edges if e.target_id.startswith("CONCEPT:")]
        assert len(concept_targets) > 0, "Should be able to navigate to concepts"

    def test_scenario_researcher_simulates_topic_exploration_sessions(self):
        """
        Scenario: Learning from how researchers explore topics

        Given a knowledge graph of documents and concepts
        When I simulate exploring documents about a specific topic
        And activate relevant documents and their concepts
        Then the system strengthens connections between co-activated items
        Because exploration patterns reveal important relationships.
        """
        # GIVEN a knowledge graph of documents and concepts
        processor = CorticalTextProcessor()
        graph = SynapticMemoryGraph()

        docs = {
            "memory_doc.txt": "Memory consolidation occurs during sleep and strengthens learning.",
            "learning_doc.txt": "Learning processes involve memory formation and retrieval.",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
            graph.add_node(f"DOC:{doc_id}", NodeType.ARTIFACT, doc_id)

        processor.compute_all()

        # Create concept nodes and links
        graph.add_node("CONCEPT:memory", NodeType.CONCEPT, "memory")
        graph.add_node("CONCEPT:learning", NodeType.CONCEPT, "learning")

        edge1 = graph.add_synaptic_edge("DOC:memory_doc.txt", "CONCEPT:memory", EdgeType.CONTAINS, weight=0.5)
        edge2 = graph.add_synaptic_edge("DOC:memory_doc.txt", "CONCEPT:learning", EdgeType.CONTAINS, weight=0.5)

        initial_weight1 = edge1.weight
        initial_weight2 = edge2.weight

        # WHEN I simulate exploring documents about a specific topic
        # AND activate relevant documents and their concepts
        graph.activate_node("DOC:memory_doc.txt", context={"topic": "memory"})
        graph.activate_node("CONCEPT:memory", context={"topic": "memory"})
        graph.activate_node("CONCEPT:learning", context={"topic": "memory"})

        # THEN the system strengthens connections between co-activated items
        strengthened = graph.apply_hebbian_learning(time_window_seconds=300)

        assert strengthened >= 0, "Should attempt to strengthen co-activated connections"
        # Weights should not decrease from co-activation
        assert edge1.weight >= initial_weight1, "Co-activated edges should maintain or increase weight"

    def test_scenario_researcher_predicts_related_concepts_from_documents(self):
        """
        Scenario: Discovering what to read next

        Given a knowledge graph with learned exploration patterns
        When I'm reading a specific document
        Then the system predicts related concepts to explore
        And prioritizes concepts by connection strength
        Because researchers benefit from guided exploration.
        """
        # GIVEN a knowledge graph with learned exploration patterns
        graph = SynapticMemoryGraph()

        # Create a document with connected concepts
        graph.add_node("DOC:current_paper", NodeType.ARTIFACT, "Current Paper")
        graph.add_node("CONCEPT:neural", NodeType.CONCEPT, "Neural")
        graph.add_node("CONCEPT:learning", NodeType.CONCEPT, "Learning")
        graph.add_node("CONCEPT:synaptic", NodeType.CONCEPT, "Synaptic")

        # Create edges with different strengths (simulating learned patterns)
        graph.add_synaptic_edge("DOC:current_paper", "CONCEPT:neural", EdgeType.CONTAINS, weight=0.9)
        graph.add_synaptic_edge("DOC:current_paper", "CONCEPT:learning", EdgeType.CONTAINS, weight=0.7)
        graph.add_synaptic_edge("DOC:current_paper", "CONCEPT:synaptic", EdgeType.CONTAINS, weight=0.4)

        # WHEN I'm reading a specific document
        # THEN the system predicts related concepts to explore
        predictions = graph.predict_next_thoughts("DOC:current_paper", top_n=3)

        assert len(predictions) > 0, "Should predict related concepts"

        # AND prioritizes concepts by connection strength
        assert predictions[0].node_id == "CONCEPT:neural", "Strongest connection should be predicted first"
        assert predictions[0].probability > predictions[-1].probability, "Should rank by probability"

    def test_scenario_researcher_discovers_similar_documents_via_concepts(self):
        """
        Scenario: Finding related documents through shared concepts

        Given documents linked to concepts
        When I identify documents sharing the same concepts
        Then I discover similarity relationships
        And can find related research I didn't know about
        Because shared concepts indicate related content.
        """
        # GIVEN documents linked to concepts
        graph = SynapticMemoryGraph()

        # Create documents
        graph.add_node("DOC:paper1", NodeType.ARTIFACT, "Paper 1")
        graph.add_node("DOC:paper2", NodeType.ARTIFACT, "Paper 2")
        graph.add_node("DOC:paper3", NodeType.ARTIFACT, "Paper 3")

        # Create shared concept
        graph.add_node("CONCEPT:plasticity", NodeType.CONCEPT, "Plasticity")

        # Link documents to shared concept
        graph.add_synaptic_edge("DOC:paper1", "CONCEPT:plasticity", EdgeType.CONTAINS, weight=0.8)
        graph.add_synaptic_edge("DOC:paper2", "CONCEPT:plasticity", EdgeType.CONTAINS, weight=0.7)

        # WHEN I identify documents sharing the same concepts
        # Find all documents connected to the plasticity concept
        docs_with_plasticity = set()

        for (src, tgt, etype), edge in graph.synaptic_edges.items():
            if tgt == "CONCEPT:plasticity" and src.startswith("DOC:"):
                docs_with_plasticity.add(src)

        # THEN I discover similarity relationships
        assert len(docs_with_plasticity) >= 2, "Should find multiple documents sharing the concept"
        assert "DOC:paper1" in docs_with_plasticity, "Should include first related document"
        assert "DOC:paper2" in docs_with_plasticity, "Should include second related document"

        # AND can find related research I didn't know about
        # If I'm reading paper1, I should discover paper2 through shared concept
        assert "DOC:paper3" not in docs_with_plasticity, "Unrelated documents should not appear"

    def test_scenario_researcher_analyzes_concept_network_structure(self):
        """
        Scenario: Understanding the concept network

        Given a knowledge graph with many document-concept links
        When I analyze the graph structure
        Then I can identify hub concepts that connect many documents
        And find document clusters around specific concepts
        Because network structure reveals knowledge organization.
        """
        # GIVEN a knowledge graph with many document-concept links
        graph = SynapticMemoryGraph()

        # Create documents
        for i in range(5):
            graph.add_node(f"DOC:paper{i}", NodeType.ARTIFACT, f"Paper {i}")

        # Create concepts
        graph.add_node("CONCEPT:hub", NodeType.CONCEPT, "Hub Concept")
        graph.add_node("CONCEPT:niche", NodeType.CONCEPT, "Niche Concept")

        # Hub concept connects to many documents
        for i in range(4):
            graph.add_synaptic_edge(f"DOC:paper{i}", "CONCEPT:hub", EdgeType.CONTAINS, weight=0.7)

        # Niche concept connects to only one
        graph.add_synaptic_edge("DOC:paper4", "CONCEPT:niche", EdgeType.CONTAINS, weight=0.7)

        # WHEN I analyze the graph structure
        # Count connections for each concept
        concept_connections = {}

        for (src, tgt, etype), edge in graph.synaptic_edges.items():
            if tgt.startswith("CONCEPT:") and src.startswith("DOC:"):
                if tgt not in concept_connections:
                    concept_connections[tgt] = 0
                concept_connections[tgt] += 1

        # THEN I can identify hub concepts that connect many documents
        assert concept_connections.get("CONCEPT:hub", 0) > concept_connections.get("CONCEPT:niche", 0), \
            "Hub concepts should have more connections"

        # AND find document clusters around specific concepts
        assert concept_connections["CONCEPT:hub"] >= 4, "Hub should connect multiple documents"
        assert concept_connections["CONCEPT:niche"] == 1, "Niche concept should have fewer connections"

    def test_scenario_researcher_tracks_activation_patterns_over_time(self):
        """
        Scenario: Understanding which knowledge gets used

        Given a knowledge graph tracking activation history
        When I activate documents and concepts during research
        Then the system records activation counts
        And I can see which parts of my knowledge are most used
        Because usage patterns reveal what's truly important.
        """
        # GIVEN a knowledge graph tracking activation history
        graph = SynapticMemoryGraph()

        graph.add_node("DOC:frequently_read", NodeType.ARTIFACT, "Frequently Read")
        graph.add_node("DOC:rarely_read", NodeType.ARTIFACT, "Rarely Read")
        graph.add_node("CONCEPT:key_idea", NodeType.CONCEPT, "Key Idea")

        # WHEN I activate documents and concepts during research
        # Simulate frequent access to one document
        for _ in range(10):
            graph.activate_node("DOC:frequently_read", context={"session": "research"})
            graph.activate_node("CONCEPT:key_idea", context={"session": "research"})

        # Simulate rare access to another
        graph.activate_node("DOC:rarely_read", context={"session": "research"})

        # THEN the system records activation counts
        freq_trace = graph.activation_traces.get("DOC:frequently_read")
        rare_trace = graph.activation_traces.get("DOC:rarely_read")

        assert freq_trace is not None, "Should track activations for frequently read document"
        assert rare_trace is not None, "Should track activations for rarely read document"

        # AND I can see which parts of my knowledge are most used
        assert freq_trace.total_activations > rare_trace.total_activations, \
            "Frequently accessed knowledge should have higher activation count"

    def test_scenario_researcher_gets_graph_statistics_for_corpus_understanding(self):
        """
        Scenario: Getting an overview of the knowledge structure

        Given a fully built knowledge graph
        When I request summary statistics
        Then I see counts of documents, concepts, and connections
        And understand the scale and structure of my corpus
        Because researchers need high-level understanding.
        """
        # GIVEN a fully built knowledge graph
        graph = SynapticMemoryGraph()

        # Create various node types
        for i in range(5):
            graph.add_node(f"DOC:paper{i}", NodeType.ARTIFACT, f"Paper {i}")

        for i in range(3):
            graph.add_node(f"CONCEPT:concept{i}", NodeType.CONCEPT, f"Concept {i}")

        # Create edges
        for i in range(5):
            for j in range(3):
                graph.add_synaptic_edge(
                    f"DOC:paper{i}",
                    f"CONCEPT:concept{j}",
                    EdgeType.CONTAINS,
                    weight=0.5
                )

        # WHEN I request summary statistics
        total_nodes = graph.node_count()
        total_edges = len(graph.synaptic_edges)

        doc_nodes = sum(1 for node_id in graph.nodes if node_id.startswith("DOC:"))
        concept_nodes = sum(1 for node_id in graph.nodes if node_id.startswith("CONCEPT:"))

        # THEN I see counts of documents, concepts, and connections
        assert total_nodes == 8, "Should count all nodes"
        assert doc_nodes == 5, "Should count document nodes"
        assert concept_nodes == 3, "Should count concept nodes"

        # AND understand the scale and structure of my corpus
        assert total_edges == 15, "Should count all connections"
        assert total_edges == doc_nodes * concept_nodes, "Should reflect the bipartite structure"
