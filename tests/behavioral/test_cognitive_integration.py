"""
Behavioral Scenarios for Cognitive Architecture Integration.

These scenarios describe how the SemanticKnowledgeGraph and HubrisMoE
integrate with CEL, GoT, WovenMind, PRISM, and SparkSLM.

Epic: Unified Cognitive Architecture

As a cognitive system,
I want all components to work together seamlessly,
So that knowledge, reasoning, and learning are coherent.
"""

import pytest
from typing import Any, Dict, List


class CognitiveSystemIntegration:
    """
    Epic: Unified Cognitive System

    As an AI system processing information,
    I want my components to integrate seamlessly,
    So that knowledge flows naturally between subsystems.
    """

    def scenario_graph_mutations_are_event_sourced(self):
        """
        Scenario: All graph changes are recorded as CEL events

        Given a SemanticKnowledgeGraph with CEL integration enabled
        When I add documents and build the graph
        Then CEL records Observation events for each mutation
        And the graph can be reconstructed from the event stream.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given CEL integration enabled
        skg = SemanticKnowledgeGraph(enable_cel=True)

        # When I add documents and build
        skg.add_document("doc1", "Machine learning enables AI.")
        skg.add_document("doc2", "Neural networks are powerful.")
        skg.build()

        # Then CEL records events
        events = skg.get_cel_events()
        event_types = [e.event_type for e in events]

        assert "document_added" in event_types
        assert "graph_built" in event_types
        assert len([e for e in events if e.event_type == "document_added"]) == 2

    def scenario_tasks_reference_graph_nodes(self):
        """
        Scenario: GoT tasks can reference knowledge graph nodes

        Given a SemanticKnowledgeGraph with documents about a topic
        And a GoT task related to that topic
        When the task is created with graph context
        Then the task references relevant graph nodes
        And decisions can cite evidence from the graph.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given a graph with ML documents
        skg = SemanticKnowledgeGraph(enable_got=True)
        skg.add_document("ml_intro", "Machine learning uses algorithms to learn.")
        skg.add_document("dl_intro", "Deep learning uses neural networks.")
        skg.build()

        # When creating a task with graph context
        task = skg.create_linked_task(
            title="Implement ML classifier",
            related_query="machine learning algorithms"
        )

        # Then task references relevant nodes
        assert task is not None
        assert len(task.related_nodes) > 0
        assert any("machine" in node.label or "learning" in node.label
                   for node in task.related_nodes)

    def scenario_surprise_triggers_graph_exploration(self):
        """
        Scenario: WovenMind surprise triggers deeper graph analysis

        Given a SemanticKnowledgeGraph integrated with WovenMind
        When processing input that triggers surprise (prediction mismatch)
        Then WovenMind switches to SLOW mode
        And explores the graph for related concepts
        And builds abstractions from graph structure.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given graph with WovenMind integration
        skg = SemanticKnowledgeGraph(enable_woven_mind=True)
        skg.add_document("patterns", "Standard patterns include singleton and factory.")
        skg.add_document("unusual", "The monad pattern transforms computation flows.")
        skg.build()

        # Train on common patterns
        skg.train_woven_mind("singleton factory observer strategy")

        # When processing unexpected input
        result = skg.process_with_woven_mind("monad functor applicative")

        # Then surprise triggers exploration
        assert result.mode == "SLOW"  # Switched due to surprise
        assert result.explored_concepts is not None
        assert len(result.explored_concepts) > 0

    def scenario_attention_modulates_search_ranking(self):
        """
        Scenario: PRISM attention modulates graph search results

        Given a SemanticKnowledgeGraph with PRISM attention
        When searching with attention focused on specific aspects
        Then results are reranked based on attention weights
        And focused aspects appear higher in results.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given graph with PRISM integration
        skg = SemanticKnowledgeGraph(enable_prism=True)
        skg.add_document("code1", "Function implementation with error handling.")
        skg.add_document("code2", "API design patterns and best practices.")
        skg.add_document("code3", "Error handling strategies for robust code.")
        skg.build()

        # When searching with attention focus
        results_default = skg.search("code patterns")
        results_focused = skg.search(
            "code patterns",
            attention_focus="error_handling"
        )

        # Then focused results prioritize error handling docs
        focused_ids = [r.doc_id for r in results_focused[:2]]
        assert "code3" in focused_ids or "code1" in focused_ids

    def scenario_predictions_prime_search(self):
        """
        Scenario: SparkSLM predictions prime graph search

        Given a SemanticKnowledgeGraph with SparkSLM integration
        When SparkSLM predicts likely concepts for a query
        Then search is primed with predicted expansions
        And related concepts are surfaced proactively.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given graph with SparkSLM integration
        skg = SemanticKnowledgeGraph(enable_spark=True)
        skg.add_document("ml", "Machine learning algorithms process data.")
        skg.add_document("dl", "Deep learning uses neural networks.")
        skg.add_document("rl", "Reinforcement learning rewards actions.")
        skg.build()

        # Train SparkSLM on patterns
        skg.train_spark("machine learning deep learning reinforcement learning")

        # When searching with prediction priming
        results = skg.search_with_priming("machine")

        # Then predictions expand search
        assert results.primed_terms is not None
        assert len(results.primed_terms) > 0
        # "learning" should be predicted after "machine"


class ExpertKnowledgeGrounding:
    """
    Epic: Expert Knowledge Grounding

    As a mixture of experts system,
    I want to ground my responses in the knowledge graph,
    So that my answers are factually grounded.
    """

    def scenario_experts_consult_knowledge_graph(self):
        """
        Scenario: Experts use the graph for knowledge retrieval

        Given HubrisMoE integrated with SemanticKnowledgeGraph
        When an expert handles a domain-specific question
        Then the expert searches the graph for relevant knowledge
        And grounds the response in retrieved information.
        """
        from cortical.graph import SemanticKnowledgeGraph
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        # Given integrated system
        skg = SemanticKnowledgeGraph()
        skg.add_document("auth", "Authentication verifies user identity using tokens.")
        skg.add_document("authz", "Authorization determines access permissions.")
        skg.build()

        moe = HubrisMoE(knowledge_graph=skg)
        security_expert = MicroExpert(
            "security", "security",
            ["authentication", "authorization"]
        )
        moe.register_expert(security_expert)

        # When querying about security
        result = moe.query("How does authentication work?")

        # Then response is grounded in graph
        assert result.grounding_docs is not None
        assert len(result.grounding_docs) > 0
        assert any("auth" in doc for doc in result.grounding_docs)

    def scenario_expert_predictions_logged_to_cel(self):
        """
        Scenario: Expert predictions are logged to CEL

        Given HubrisMoE with CEL integration
        When experts make predictions
        Then predictions are logged as CEL Observation events
        And calibration can be computed from event history.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert, CreditLedger

        # Given MoE with CEL
        moe = HubrisMoE(enable_cel=True)
        expert = MicroExpert("test", "test", ["skill"])
        moe.register_expert(expert)

        # When making predictions
        for i in range(5):
            result = moe.query(f"question {i}")
            moe.record_outcome(result.prediction_id, correct=(i % 2 == 0))

        # Then predictions are in CEL
        events = moe.get_cel_events()
        prediction_events = [e for e in events if e.event_type == "expert_prediction"]
        assert len(prediction_events) >= 5

    def scenario_decisions_record_expert_confidence(self):
        """
        Scenario: GoT decisions record expert confidence

        Given HubrisMoE integrated with GoT
        When a decision is made based on expert consultation
        Then the decision records contributing experts and confidence
        And the rationale includes calibration information.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        # Given MoE with GoT
        moe = HubrisMoE(enable_got=True)
        moe.register_expert(MicroExpert("arch", "architecture", ["design"]))
        moe.register_expert(MicroExpert("impl", "implementation", ["coding"]))

        # When making a decision
        result = moe.query("Should we use microservices or monolith?")
        decision = moe.create_decision(
            question="Architecture choice",
            chosen=result.answer,
            consultation_result=result
        )

        # Then decision includes expert info
        assert decision is not None
        assert decision.contributing_experts is not None
        assert len(decision.contributing_experts) > 0
        assert decision.confidence is not None


class CognitiveLoopIntegration:
    """
    Epic: Cognitive Loop Integration

    As a system with continuous cognition,
    I want the cognitive loop to orchestrate all components,
    So that reasoning proceeds coherently across cycles.
    """

    def scenario_cognitive_cycle_flows_through_all_systems(self):
        """
        Scenario: Full cognitive cycle integrates all systems

        Given a CognitiveOrchestrator integrating all components
        When processing a complex query requiring multiple systems
        Then the query flows through:
            1. SparkSLM (prediction/priming)
            2. SemanticKnowledgeGraph (knowledge retrieval)
            3. WovenMind (processing mode selection)
            4. HubrisMoE (expert consultation)
            5. PRISM (attention/plasticity updates)
            6. GoT (decision/task tracking)
            7. CEL (event logging)
        And the result combines insights from all systems.
        """
        from cortical.graph import SemanticKnowledgeGraph
        from cortical.reasoning.hubris import HubrisMoE

        # Given fully integrated system
        skg = SemanticKnowledgeGraph(
            enable_cel=True,
            enable_got=True,
            enable_woven_mind=True,
            enable_prism=True,
            enable_spark=True
        )
        skg.add_document("doc1", "Complex systems require careful design.")
        skg.build()

        # When processing through cognitive orchestrator
        result = skg.cognitive_process(
            query="How should we design a complex system?",
            mode="full_integration"
        )

        # Then all systems contributed
        assert result.spark_priming is not None  # SparkSLM primed
        assert result.graph_results is not None  # SKG searched
        assert result.woven_mind_mode is not None  # Mode selected
        assert result.events_logged > 0  # CEL recorded

    def scenario_consolidation_transfers_to_graph(self):
        """
        Scenario: WovenMind consolidation transfers patterns to graph

        Given a SemanticKnowledgeGraph with WovenMind
        When WovenMind consolidates patterns during "sleep"
        Then high-frequency Hive patterns become graph concepts
        And Cortex abstractions become semantic relations.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given integrated system
        skg = SemanticKnowledgeGraph(enable_woven_mind=True)
        skg.add_document("base", "Foundation concepts for the system.")
        skg.build()

        initial_nodes = skg.node_count()

        # Train WovenMind on patterns
        for _ in range(10):
            skg.train_woven_mind("pattern A leads to pattern B")
            skg.train_woven_mind("pattern B results in pattern C")

        # When consolidating
        consolidation_result = skg.consolidate_woven_mind()

        # Then patterns transfer to graph
        assert consolidation_result.patterns_transferred > 0
        assert skg.node_count() >= initial_nodes  # May add concept nodes

    def scenario_plasticity_updates_edge_weights(self):
        """
        Scenario: PRISM plasticity updates graph edge weights

        Given a SemanticKnowledgeGraph with PRISM plasticity
        When certain paths are frequently activated
        Then those edge weights increase (Hebbian learning)
        And unused edges decay over time.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given integrated system
        skg = SemanticKnowledgeGraph(enable_prism=True)
        skg.add_document("a", "Concept A relates to concept B.")
        skg.add_document("b", "Concept B connects to concept C.")
        skg.build()

        # Get initial edge weights
        initial_weight = skg.get_edge_weight("concept:a", "concept:b")

        # When activating path repeatedly
        for _ in range(10):
            skg.activate_path(["concept:a", "concept:b"])

        # Then weights increase
        final_weight = skg.get_edge_weight("concept:a", "concept:b")
        assert final_weight > initial_weight

        # And apply decay
        skg.apply_plasticity_decay()
        # Unused edges would decay (not tested here)


class IntegratedSearchAndReasoning:
    """
    Epic: Integrated Search and Reasoning

    As a knowledge retrieval system,
    I want search to leverage all cognitive capabilities,
    So that results are maximally relevant and insightful.
    """

    def scenario_multi_hop_reasoning_through_graph(self):
        """
        Scenario: Search performs multi-hop reasoning

        Given a SemanticKnowledgeGraph with chained concepts
        When searching for a concept connected through intermediates
        Then multi-hop paths are discovered
        And indirect connections surface related documents.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given graph with chained concepts
        skg = SemanticKnowledgeGraph()
        skg.add_document("a", "Python is a programming language.")
        skg.add_document("b", "Machine learning uses Python extensively.")
        skg.add_document("c", "Neural networks are a machine learning technique.")
        skg.build()

        # When searching with multi-hop
        results = skg.search_multihop(
            "programming language neural networks",
            max_hops=2
        )

        # Then all connected docs found
        doc_ids = [r.doc_id for r in results]
        assert len(doc_ids) >= 2  # Should find chain

    def scenario_expert_ensemble_resolves_ambiguity(self):
        """
        Scenario: Expert ensemble resolves ambiguous queries

        Given HubrisMoE with multiple domain experts
        When a query spans multiple domains ambiguously
        Then multiple experts contribute perspectives
        And the combined response addresses the ambiguity.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        # Given multi-domain MoE
        moe = HubrisMoE()
        moe.register_expert(MicroExpert("lang", "programming", ["python", "java"]))
        moe.register_expert(MicroExpert("bio", "biology", ["python", "snake"]))

        # When querying ambiguous term
        result = moe.query("Tell me about python")

        # Then multiple experts contribute
        assert len(result.contributing_experts) >= 1
        # Both programming and biology experts may respond

    def scenario_anomaly_detection_flags_unusual_patterns(self):
        """
        Scenario: SparkSLM anomaly detection flags unusual graph patterns

        Given a SemanticKnowledgeGraph with trained patterns
        When a document with unusual patterns is added
        Then SparkSLM detects the anomaly
        And the graph marks the document for review.
        """
        from cortical.graph import SemanticKnowledgeGraph

        # Given trained graph
        skg = SemanticKnowledgeGraph(enable_spark=True)
        for i in range(10):
            skg.add_document(f"normal_{i}", "Standard patterns and typical structures.")
        skg.build()
        skg.train_spark_on_corpus()

        # When adding anomalous document
        skg.add_document("anomaly", "Xyzzy plugh completely unusual gibberish content.")
        anomalies = skg.detect_anomalies("anomaly")

        # Then anomaly is flagged
        assert len(anomalies) > 0 or anomalies.is_anomalous


# Pytest test class
class TestCognitiveIntegration:
    """Pytest wrapper for integration scenarios."""

    def test_cel_event_sourcing(self):
        """Test CEL integration for event sourcing."""
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph(enable_cel=True)
        skg.add_document("test", "Test content")
        skg.build()

        events = skg.get_cel_events()
        assert len(events) >= 2  # At least add and build

    def test_graph_search_works(self):
        """Test basic graph search still works with integrations."""
        from cortical.graph import SemanticKnowledgeGraph

        skg = SemanticKnowledgeGraph()
        skg.add_document("ml", "Machine learning algorithms")
        skg.build()

        results = skg.search("machine learning")
        assert len(results) >= 1

    def test_hubris_with_knowledge_graph(self):
        """Test HubrisMoE can use knowledge graph."""
        from cortical.graph import SemanticKnowledgeGraph
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        skg = SemanticKnowledgeGraph()
        skg.add_document("info", "Information for grounding")
        skg.build()

        moe = HubrisMoE(knowledge_graph=skg)
        moe.register_expert(MicroExpert("test", "test", ["skill"]))

        result = moe.query("test query")
        assert result is not None
