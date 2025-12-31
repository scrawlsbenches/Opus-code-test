"""
Behavioral tests for researchers building knowledge graphs from document collections.

Epic: Knowledge Graph Discovery

As a researcher with a large document collection,
I want to build interconnected knowledge graphs,
So that I can discover relationships and patterns across my corpus.

Based on: examples/prism_got_comprehensive_demo.py
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
    PlasticityRules,
)


class TestResearcherBuildsKnowledgeGraphs:
    """
    Epic: Knowledge Graph Discovery

    As a researcher with a vast document collection,
    I want to automatically build knowledge graphs,
    So that I discover relationships I wouldn't find manually.
    """

    def test_scenario_researcher_loads_corpus_and_builds_index(self, tmp_path):
        """
        Scenario: Loading documents and building searchable index

        Given a directory containing research documents
        When I index the documents with the cortical processor
        Then I can extract tokens, bigrams, and concepts
        And the system computes TF-IDF scores for importance ranking
        Because researchers need to identify key terms quickly.
        """
        # GIVEN a directory containing research documents
        docs = {
            "cognitive_science.txt": "Memory consolidation during sleep strengthens neural connections.",
            "ai_systems.txt": "Neural networks learn through backpropagation and gradient descent.",
            "knowledge_graphs.txt": "Graph structures enable semantic search and knowledge discovery.",
        }

        # WHEN I index the documents with the cortical processor
        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all()

        # THEN I can extract tokens, bigrams, and concepts
        token_count = processor.get_layer(CorticalLayer.TOKENS).column_count()
        bigram_count = processor.get_layer(CorticalLayer.BIGRAMS).column_count()
        concept_count = processor.get_layer(CorticalLayer.CONCEPTS).column_count()

        assert token_count > 0, "Should extract tokens from documents"
        assert bigram_count > 0, "Should extract bigrams from documents"
        assert concept_count > 0, "Should identify concept clusters"

        # AND the system computes TF-IDF scores for importance ranking
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        for col in layer0.minicolumns.values():
            if len(col.document_ids) > 0:
                assert col.tfidf >= 0, "Should compute TF-IDF scores"
                break

    def test_scenario_researcher_creates_multi_type_knowledge_graph(self):
        """
        Scenario: Building graphs with multiple node types

        Given a synaptic memory graph
        When I create nodes representing domains, documents, and concepts
        Then I can link them with typed edges
        And navigate the knowledge structure semantically
        Because different types of knowledge require different representations.
        """
        # GIVEN a synaptic memory graph
        graph = SynapticMemoryGraph()

        # WHEN I create nodes representing domains, documents, and concepts
        domain_node = graph.add_node("DOMAIN:cognitive_science", NodeType.CONTEXT, "Cognitive Science")
        doc_node = graph.add_node("DOC:memory_paper.pdf", NodeType.ARTIFACT, "Memory Consolidation Paper")
        concept_node = graph.add_node("CONCEPT:hebbian_learning", NodeType.CONCEPT, "Hebbian Learning")

        # THEN I can link them with typed edges
        graph.add_synaptic_edge("DOMAIN:cognitive_science", "DOC:memory_paper.pdf", EdgeType.CONTAINS, weight=0.8)
        graph.add_synaptic_edge("DOC:memory_paper.pdf", "CONCEPT:hebbian_learning", EdgeType.CONTAINS, weight=0.7)

        # AND navigate the knowledge structure semantically
        domain_edges = graph.get_synaptic_edges_from("DOMAIN:cognitive_science")
        assert len(domain_edges) == 1
        assert domain_edges[0].target_id == "DOC:memory_paper.pdf"
        assert domain_edges[0].edge_type == EdgeType.CONTAINS

        doc_edges = graph.get_synaptic_edges_from("DOC:memory_paper.pdf")
        assert len(doc_edges) == 1
        assert doc_edges[0].target_id == "CONCEPT:hebbian_learning"

    def test_scenario_researcher_discovers_cross_document_connections(self):
        """
        Scenario: Discovering documents connected by shared concepts

        Given multiple documents about related topics
        When I extract key concepts from each document
        And create bridges between documents sharing concepts
        Then I discover connections I might have missed
        Because related ideas often use different terminology.
        """
        # GIVEN multiple documents about related topics
        graph = SynapticMemoryGraph()
        processor = CorticalTextProcessor()

        docs = {
            "neuroscience_paper.txt": "Synaptic plasticity enables learning through connection strengthening",
            "ml_paper.txt": "Neural networks adjust weights during training to improve performance",
            "memory_study.txt": "Hebbian learning strengthens connections between co-activated neurons",
        }

        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
            graph.add_node(f"DOC:{doc_id}", NodeType.ARTIFACT, doc_id)

        processor.compute_all()

        # WHEN I extract key concepts from each document
        shared_concepts = set()
        for doc_id in docs.keys():
            layer0 = processor.get_layer(CorticalLayer.TOKENS)
            for col in layer0.minicolumns.values():
                if doc_id in col.document_ids and col.tfidf > 0.1:
                    concept_key = col.content.lower()

                    # Create concept node if it doesn't exist
                    concept_node_id = f"CONCEPT:{concept_key}"
                    if concept_node_id not in graph.nodes:
                        graph.add_node(concept_node_id, NodeType.CONCEPT, col.content)

                    # Link document to concept
                    edge_key = (f"DOC:{doc_id}", concept_node_id, EdgeType.CONTAINS)
                    if edge_key not in graph.synaptic_edges:
                        graph.add_synaptic_edge(
                            f"DOC:{doc_id}",
                            concept_node_id,
                            EdgeType.CONTAINS,
                            weight=0.6
                        )

        # AND create bridges between documents sharing concepts
        # Find documents that share the concept "learning"
        learning_docs = []
        for doc_id in docs.keys():
            doc_edges = graph.get_synaptic_edges_from(f"DOC:{doc_id}")
            for edge in doc_edges:
                if "learning" in edge.target_id.lower():
                    learning_docs.append(doc_id)
                    break

        # THEN I discover connections I might have missed
        assert len(learning_docs) >= 2, "Should find multiple documents about learning"

    def test_scenario_researcher_learns_from_exploration_patterns(self):
        """
        Scenario: Strengthening connections through repeated exploration

        Given a knowledge graph with documents and concepts
        When I explore documents about a specific topic
        And I activate related nodes during exploration
        Then the system strengthens connections between co-activated items
        Because frequently used paths should become more prominent.
        """
        # GIVEN a knowledge graph with documents and concepts
        rules = PlasticityRules(
            hebbian_rate=0.15,
            anti_hebbian_rate=0.03,
            reward_rate=0.20,
        )
        graph = SynapticMemoryGraph(plasticity_rules=rules)

        # Create a simple knowledge structure
        graph.add_node("DOC:memory_paper", NodeType.ARTIFACT, "Memory Paper")
        graph.add_node("CONCEPT:consolidation", NodeType.CONCEPT, "Consolidation")
        graph.add_node("CONCEPT:sleep", NodeType.CONCEPT, "Sleep")

        edge1 = graph.add_synaptic_edge("DOC:memory_paper", "CONCEPT:consolidation", EdgeType.CONTAINS, weight=0.5)
        edge2 = graph.add_synaptic_edge("DOC:memory_paper", "CONCEPT:sleep", EdgeType.CONTAINS, weight=0.5)

        initial_weight_1 = edge1.weight
        initial_weight_2 = edge2.weight

        # WHEN I explore documents about a specific topic
        # AND I activate related nodes during exploration
        graph.activate_node("DOC:memory_paper", context={"topic": "memory"})
        graph.activate_node("CONCEPT:consolidation", context={"topic": "memory"})
        graph.activate_node("CONCEPT:sleep", context={"topic": "memory"})

        # THEN the system strengthens connections through co-activation
        strengthened = graph.apply_hebbian_learning(time_window_seconds=60)

        assert strengthened > 0, "Should strengthen some connections"
        assert edge1.weight >= initial_weight_1, "Co-activated edges should maintain or increase weight"
        assert edge2.weight >= initial_weight_2, "Co-activated edges should maintain or increase weight"

    def test_scenario_researcher_receives_predictions_for_related_content(self):
        """
        Scenario: Predicting relevant content based on current reading

        Given a knowledge graph with learned patterns
        When I'm reading a specific document
        Then the system predicts related concepts I should explore
        And prioritizes suggestions based on past exploration patterns
        Because researchers benefit from serendipitous discovery.
        """
        # GIVEN a knowledge graph with learned patterns
        graph = SynapticMemoryGraph()

        # Create a document with multiple related concepts
        graph.add_node("DOC:current", NodeType.ARTIFACT, "Current Document")
        graph.add_node("CONCEPT:neural", NodeType.CONCEPT, "Neural")
        graph.add_node("CONCEPT:synaptic", NodeType.CONCEPT, "Synaptic")
        graph.add_node("CONCEPT:plasticity", NodeType.CONCEPT, "Plasticity")

        # Create edges with different weights (simulating learned preferences)
        graph.add_synaptic_edge("DOC:current", "CONCEPT:neural", EdgeType.CONTAINS, weight=0.9)
        graph.add_synaptic_edge("DOC:current", "CONCEPT:synaptic", EdgeType.CONTAINS, weight=0.7)
        graph.add_synaptic_edge("DOC:current", "CONCEPT:plasticity", EdgeType.CONTAINS, weight=0.4)

        # WHEN I'm reading a specific document
        # THEN the system predicts related concepts I should explore
        predictions = graph.predict_next_thoughts("DOC:current", top_n=3)

        assert len(predictions) > 0, "Should provide predictions"

        # AND prioritizes suggestions based on past exploration patterns
        # Higher weight edges should be predicted first
        assert predictions[0].node_id == "CONCEPT:neural", "Highest weight concept should be predicted first"
        assert predictions[0].probability > predictions[-1].probability, "Higher weights should have higher probability"

    def test_scenario_researcher_reinforces_valuable_knowledge_paths(self):
        """
        Scenario: Learning which paths are most valuable through feedback

        Given a reasoning path through the knowledge graph
        When I mark this path as valuable with positive feedback
        Then the system strengthens all edges in that path
        And future predictions favor this proven pathway
        Because successful exploration patterns should be reinforced.
        """
        # GIVEN a reasoning path through the knowledge graph
        graph = SynapticMemoryGraph()

        path_nodes = [
            ("DOMAIN:neuroscience", NodeType.CONTEXT),
            ("DOC:hebbian_paper", NodeType.ARTIFACT),
            ("CONCEPT:learning", NodeType.CONCEPT),
        ]

        for node_id, node_type in path_nodes:
            graph.add_node(node_id, node_type, node_id.split(":")[-1])

        # Create path edges
        edge1 = graph.add_synaptic_edge(
            "DOMAIN:neuroscience", "DOC:hebbian_paper", EdgeType.CONTAINS, weight=0.6
        )
        edge2 = graph.add_synaptic_edge(
            "DOC:hebbian_paper", "CONCEPT:learning", EdgeType.CONTAINS, weight=0.6
        )

        initial_weight_1 = edge1.weight
        initial_weight_2 = edge2.weight

        # WHEN I mark this path as valuable with positive feedback
        path = ["DOMAIN:neuroscience", "DOC:hebbian_paper", "CONCEPT:learning"]
        graph.apply_reward(path, reward=0.5)

        # THEN the system strengthens all edges in that path
        assert edge1.weight > initial_weight_1, "First edge should be strengthened"
        assert edge2.weight > initial_weight_2, "Second edge should be strengthened"

    def test_scenario_researcher_observes_knowledge_decay_over_time(self):
        """
        Scenario: Unused knowledge gradually becomes less prominent

        Given a knowledge graph with various connections
        When time passes without activating certain paths
        Then unused connections gradually weaken
        But frequently accessed paths resist decay
        Because the graph should reflect current research focus.
        """
        # GIVEN a knowledge graph with various connections
        graph = SynapticMemoryGraph()

        graph.add_node("A", NodeType.CONCEPT, "Concept A")
        graph.add_node("B", NodeType.CONCEPT, "Concept B")

        # Create edges with different decay rates
        fast_decay = graph.add_synaptic_edge("A", "B", EdgeType.SIMILAR, weight=1.0, decay_factor=0.9)
        slow_decay = graph.add_synaptic_edge("B", "A", EdgeType.SIMILAR, weight=1.0, decay_factor=0.99)

        initial_fast = fast_decay.weight
        initial_slow = slow_decay.weight

        # WHEN time passes without activating certain paths
        for _ in range(10):
            graph.apply_global_decay()

        # THEN unused connections gradually weaken
        assert fast_decay.weight < initial_fast, "Fast-decaying edge should weaken"
        assert slow_decay.weight < initial_slow, "Slow-decaying edge should weaken"

        # BUT frequently accessed paths resist decay
        assert slow_decay.weight > fast_decay.weight, "Slow-decay edge should retain more weight"
