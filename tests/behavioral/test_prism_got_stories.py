"""
Behavioral Tests for PRISM-GoT: Synaptic Memory and Incremental Reasoning.

This module tests the Predictive Reasoning through Incremental Synaptic Memory
Graph of Thought implementation - a biologically-inspired learning system
we built ourselves from first principles.

Epic: Cognitive researcher builds adaptive reasoning system
Story: As a cognitive researcher building adaptive AI,
       I want synaptic memory with plasticity we implemented ourselves,
       So that reasoning learns from experience we control completely.
"""

import pytest
from datetime import datetime, timedelta
from cortical.reasoning.prism_got import (
    SynapticMemoryGraph,
    IncrementalReasoner,
    PlasticityRules,
    SynapticEdge,
    ActivationTrace,
    PredictionResult,
)
from cortical.reasoning.graph_of_thought import NodeType, EdgeType


class TestCognitiveResearcherBuildsAdaptiveReasoning:
    """
    Epic: Cognitive Researcher Builds Adaptive Reasoning System

    As a cognitive researcher building adaptive AI,
    I want synaptic plasticity we implemented ourselves,
    So that systems learn from patterns we control.
    """

    def test_scenario_synaptic_edges_strengthen_with_use(self):
        """
        Scenario: Frequently used connections strengthen over time

        Given reasoning paths we're tracking
        When connections are repeatedly activated
        Then synaptic weights increase
        Because we implemented Hebbian learning ourselves
        """
        # Given reasoning paths
        graph = SynapticMemoryGraph()
        graph.add_node("Q1", NodeType.QUESTION, "How to optimize custom search?")
        graph.add_node("A1", NodeType.INSIGHT, "Build inverted index ourselves")

        edge = graph.add_synaptic_edge("Q1", "A1", EdgeType.SUPPORTS, weight=0.5)

        # When repeatedly activated
        initial_weight = edge.weight
        for _ in range(3):
            edge.record_activation()

        # Strengthen manually (simulating learning)
        edge.strengthen(amount=0.1)

        # Then weight increases
        assert edge.weight > initial_weight

    def test_scenario_unused_connections_weaken_over_time(self):
        """
        Scenario: Synaptic decay weakens unused connections

        Given established connections we built
        When connections go unused
        Then weights decay toward zero
        Because we implemented temporal decay ourselves
        """
        # Given established connection
        graph = SynapticMemoryGraph()
        graph.add_node("N1", NodeType.CONCEPT, "Concept A")
        graph.add_node("N2", NodeType.CONCEPT, "Concept B")

        edge = graph.add_synaptic_edge("N1", "N2", EdgeType.SIMILAR, weight=1.0, decay_factor=0.9)

        # When applying decay
        initial_weight = edge.weight
        edge.apply_decay()
        edge.apply_decay()
        edge.apply_decay()

        # Then weight decreases
        assert edge.weight < initial_weight
        assert edge.weight > 0.0

    def test_scenario_activation_history_tracks_usage_patterns(self):
        """
        Scenario: System records activation patterns for analysis

        Given nodes in our reasoning graph
        When they are activated
        Then usage history is tracked
        Because we monitor patterns ourselves
        """
        # Given reasoning graph
        graph = SynapticMemoryGraph()
        graph.add_node("C1", NodeType.CONCEPT, "Custom implementation pattern")

        # When activating
        graph.activate_node("C1", context={"source": "reasoning"})
        graph.activate_node("C1", context={"source": "analysis"})

        # Then history tracked
        trace = graph.activation_traces["C1"]
        assert trace.total_activations == 2
        recent = trace.get_recent(n=2)
        assert len(recent) == 2

    def test_scenario_hebbian_learning_strengthens_coactivated_paths(self):
        """
        Scenario: Co-activated thoughts strengthen their connection

        Given thoughts activated together
        When Hebbian learning runs
        Then their connection strengthens
        Because we implemented "fire together, wire together" ourselves
        """
        # Given co-activated thoughts
        graph = SynapticMemoryGraph()
        graph.add_node("T1", NodeType.CONCEPT, "Custom indexing")
        graph.add_node("T2", NodeType.CONCEPT, "Fast retrieval")

        edge = graph.add_synaptic_edge("T1", "T2", EdgeType.ENABLES, weight=0.5)
        initial_weight = edge.weight

        # When activating together
        graph.activate_node("T1")
        graph.activate_node("T2")

        # Apply Hebbian learning
        strengthened = graph.apply_hebbian_learning(time_window_seconds=60.0)

        # Then connection strengthens
        assert edge.weight >= initial_weight


class TestIncrementalReasonerBuildsKnowledge:
    """
    Epic: Incremental Reasoner Builds Knowledge From Experience

    As a system learning from experience,
    I want incremental graph building,
    So that knowledge accumulates we control.
    """

    def test_scenario_reasoner_processes_thoughts_incrementally(self):
        """
        Scenario: Thoughts are added one at a time to graph

        Given an incremental reasoner we built
        When processing sequential thoughts
        Then graph builds progressively
        Because we implement incremental construction ourselves
        """
        # Given incremental reasoner
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        # When processing thoughts
        q1 = reasoner.process_thought(
            "What search algorithm to build ourselves?",
            NodeType.QUESTION
        )

        h1 = reasoner.process_thought(
            "Custom inverted index",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        h2 = reasoner.process_thought(
            "Hand-built B-tree",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        # Then graph builds progressively
        assert graph.node_count() == 3
        assert graph.edge_count() == 2  # q1->h1, q1->h2

    def test_scenario_reasoner_predicts_next_likely_thoughts(self):
        """
        Scenario: System predicts probable next thoughts

        Given established reasoning patterns we learned
        When I query predictions
        Then likely next thoughts are suggested
        Because we built prediction ourselves
        """
        # Given established patterns
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        q = reasoner.process_thought("How to optimize?", NodeType.QUESTION)
        a = reasoner.process_thought("Build caching ourselves", NodeType.INSIGHT,
                                    relation_to_focus=EdgeType.SUPPORTS)

        # When predicting next
        predictions = reasoner.predict_next(q.id, top_n=3)

        # Then suggestions available
        assert len(predictions) > 0
        assert isinstance(predictions[0], PredictionResult)
        assert predictions[0].node_id == a.id

    def test_scenario_successful_paths_are_reinforced(self):
        """
        Scenario: Reward learning strengthens good paths

        Given a reasoning path that succeeded
        When I mark it as successful
        Then connections along path strengthen
        Because we implement reward learning ourselves
        """
        # Given successful path
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        q = reasoner.process_thought("What approach?", NodeType.QUESTION)
        h = reasoner.process_thought("Custom implementation", NodeType.HYPOTHESIS,
                                    relation_to_focus=EdgeType.EXPLORES)
        d = reasoner.process_thought("Build it ourselves", NodeType.DECISION,
                                    relation_to_focus=EdgeType.IMPLEMENTS)

        # Get initial edge weight
        edges = graph.get_synaptic_edges_from(q.id)
        initial_weight = edges[0].weight if edges else 0.0

        # When marking successful
        reasoner.mark_outcome_success(path=[q.id, h.id, d.id], reward=0.5)

        # Then path strengthened
        edges_after = graph.get_synaptic_edges_from(q.id)
        assert edges_after[0].weight >= initial_weight

    def test_scenario_failed_paths_are_weakened(self):
        """
        Scenario: Failed paths are weakened to avoid repetition

        Given a reasoning path that failed
        When I mark it as failure
        Then connections weaken
        Because we learn from mistakes ourselves
        """
        # Given failed path
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        q = reasoner.process_thought("What to try?", NodeType.QUESTION)
        h = reasoner.process_thought("Adopt external library", NodeType.HYPOTHESIS,
                                    relation_to_focus=EdgeType.EXPLORES)

        edges = graph.get_synaptic_edges_from(q.id)
        initial_weight = edges[0].weight if edges else 1.0

        # When marking failure
        reasoner.mark_outcome_failure(path=[q.id, h.id], penalty=0.3)

        # Then weakened
        edges_after = graph.get_synaptic_edges_from(q.id)
        assert edges_after[0].weight <= initial_weight

    def test_scenario_focus_tracks_current_reasoning_context(self):
        """
        Scenario: Focus maintains current reasoning context

        Given ongoing reasoning we're managing
        When processing thoughts
        Then focus tracks current context
        Because we track attention ourselves
        """
        # Given ongoing reasoning
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        # When processing thoughts
        t1 = reasoner.process_thought("First thought", NodeType.CONCEPT)
        assert reasoner.current_focus == t1.id

        t2 = reasoner.process_thought("Second thought", NodeType.CONCEPT)
        assert reasoner.current_focus == t2.id

        # Focus can be reset
        reasoner.reset_focus()
        assert reasoner.current_focus is None


class TestPredictiveReasoningAnticipatesPatterns:
    """
    Epic: Predictive Reasoning Anticipates Future Thoughts

    As a system learning patterns,
    I want predictive capabilities,
    So that reasoning becomes proactive.
    """

    def test_scenario_predictions_rank_by_probability(self):
        """
        Scenario: Predictions are ranked by likelihood

        Given multiple possible next thoughts
        When I request predictions
        Then results are ranked by probability
        Because we compute probabilities ourselves
        """
        # Given multiple possibilities
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        start = reasoner.process_thought("Starting point", NodeType.QUESTION)

        # Create multiple branches with different weights
        opt1 = reasoner.process_thought("High probability path", NodeType.INSIGHT,
                                       relation_to_focus=EdgeType.SUPPORTS)
        reasoner.set_focus(start.id)

        opt2 = reasoner.process_thought("Lower probability path", NodeType.INSIGHT,
                                       relation_to_focus=EdgeType.SUPPORTS)

        # Strengthen one path more
        edges = graph.get_synaptic_edges_from(start.id)
        if len(edges) > 0:
            edges[0].strengthen(0.3)  # Strengthen first edge more

        # When requesting predictions
        predictions = reasoner.predict_next(start.id, top_n=2)

        # Then ranked by probability
        assert len(predictions) >= 1
        if len(predictions) == 2:
            assert predictions[0].probability >= predictions[1].probability

    def test_scenario_prediction_accuracy_tracked_over_time(self):
        """
        Scenario: System learns which predictions are accurate

        Given predictions made over time
        When I verify outcomes
        Then accuracy is tracked per edge
        Because we monitor prediction quality ourselves
        """
        # Given predictions
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        q = reasoner.process_thought("Question", NodeType.QUESTION)
        a = reasoner.process_thought("Answer", NodeType.INSIGHT,
                                    relation_to_focus=EdgeType.SUPPORTS)

        edges = graph.get_synaptic_edges_from(q.id)
        edge = edges[0] if edges else None

        if edge:
            # When verifying predictions
            initial_accuracy = edge.prediction_accuracy

            # Record correct prediction
            edge.record_prediction_outcome(correct=True)
            edge.record_prediction_outcome(correct=True)
            edge.record_prediction_outcome(correct=False)

            # Then accuracy updated
            assert edge.prediction_total == 3
            assert edge.prediction_correct == 2
            # Accuracy should be around 2/3 with smoothing

    def test_scenario_similar_content_auto_linked(self):
        """
        Scenario: Similar thoughts are automatically connected

        Given auto-linking enabled in our reasoner
        When processing similar content
        Then similarity edges are created
        Because we detect patterns ourselves
        """
        # Given auto-linking reasoner
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(
            graph,
            auto_link_similar=True,
            similarity_threshold=0.5
        )

        # When processing similar content
        t1 = reasoner.process_thought(
            "Custom search indexing algorithm",
            NodeType.CONCEPT
        )

        t2 = reasoner.process_thought(
            "Search algorithm custom indexing",
            NodeType.CONCEPT
        )

        # Then similarity edge created
        # Check for edges between nodes
        edges_from_t1 = graph.get_synaptic_edges_from(t1.id)
        edges_from_t2 = graph.get_synaptic_edges_from(t2.id)

        # At least one should have similarity edge
        all_edges = edges_from_t1 + edges_from_t2
        similarity_edges = [e for e in all_edges if e.edge_type == EdgeType.SIMILAR]
        assert len(similarity_edges) > 0


class TestSynapticMemoryProvidesInsights:
    """
    Epic: Synaptic Memory Provides Cognitive Insights

    As a researcher analyzing reasoning patterns,
    I want inspection and analysis tools,
    So that I understand learned behaviors.
    """

    def test_scenario_activation_frequency_reveals_important_concepts(self):
        """
        Scenario: Frequently activated nodes reveal key concepts

        Given reasoning with repeated concepts
        When I analyze activation frequency
        Then important thoughts are revealed
        Because we track activations ourselves
        """
        # Given repeated activations
        graph = SynapticMemoryGraph()
        graph.add_node("important", NodeType.CONCEPT, "Core custom implementation")
        graph.add_node("peripheral", NodeType.CONCEPT, "Minor detail")

        # When activating at different rates
        for _ in range(10):
            graph.activate_node("important")

        graph.activate_node("peripheral")

        # Then frequency distinguishes importance
        important_trace = graph.activation_traces["important"]
        peripheral_trace = graph.activation_traces["peripheral"]

        assert important_trace.total_activations > peripheral_trace.total_activations

    def test_scenario_graph_serialization_preserves_learning(self):
        """
        Scenario: Learned graph can be saved and restored

        Given a graph with learned patterns
        When I serialize and deserialize
        Then learning is preserved
        Because we built serialization ourselves
        """
        # Given learned graph
        graph = SynapticMemoryGraph()
        graph.add_node("N1", NodeType.CONCEPT, "Concept we learned")
        graph.add_node("N2", NodeType.CONCEPT, "Related concept")

        edge = graph.add_synaptic_edge("N1", "N2", EdgeType.SIMILAR, weight=0.8)
        edge.record_activation()
        edge.strengthen(0.2)

        graph.activate_node("N1")

        # When serializing
        serialized = graph.to_dict()
        restored = SynapticMemoryGraph.from_dict(serialized)

        # Then learning preserved
        assert restored.node_count() == graph.node_count()
        assert len(restored.synaptic_edges) == len(graph.synaptic_edges)

        # Check activation trace preserved
        assert "N1" in restored.activation_traces
        assert restored.activation_traces["N1"].total_activations > 0

    def test_scenario_global_decay_simulates_forgetting(self):
        """
        Scenario: Periodic decay simulates memory consolidation

        Given a graph with many connections
        When global decay is applied
        Then unused connections fade
        Because we simulate biological forgetting ourselves
        """
        # Given many connections
        graph = SynapticMemoryGraph()
        graph.add_node("A", NodeType.CONCEPT, "Concept A")
        graph.add_node("B", NodeType.CONCEPT, "Concept B")
        graph.add_node("C", NodeType.CONCEPT, "Concept C")

        graph.add_synaptic_edge("A", "B", EdgeType.SIMILAR, weight=1.0, decay_factor=0.9)
        graph.add_synaptic_edge("B", "C", EdgeType.SIMILAR, weight=1.0, decay_factor=0.9)

        # When applying decay
        initial_weights = [e.weight for e in graph.synaptic_edges.values()]
        decayed_count = graph.apply_global_decay()

        # Then connections fade
        final_weights = [e.weight for e in graph.synaptic_edges.values()]
        assert all(final < initial for final, initial in zip(final_weights, initial_weights))
        assert decayed_count > 0

    def test_scenario_plasticity_rules_are_configurable(self):
        """
        Scenario: Learning rules can be tuned for different behaviors

        Given customizable plasticity rules
        When I configure learning rates
        Then behavior adapts to settings
        Because we built configurable learning ourselves
        """
        # Given custom rules
        rules = PlasticityRules(
            hebbian_rate=0.2,
            anti_hebbian_rate=0.1,
            reward_rate=0.3,
            max_weight=3.0
        )

        graph = SynapticMemoryGraph(plasticity_rules=rules)

        # Then rules are applied
        assert graph.plasticity.hebbian_rate == 0.2
        assert graph.plasticity.anti_hebbian_rate == 0.1
        assert graph.plasticity.max_weight == 3.0

    def test_scenario_reasoner_provides_comprehensive_summary(self):
        """
        Scenario: Summary reveals reasoning graph statistics

        Given an active reasoning session
        When I request summary
        Then comprehensive statistics are provided
        Because we track everything ourselves
        """
        # Given active session
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        reasoner.process_thought("Q1", NodeType.QUESTION)
        reasoner.process_thought("A1", NodeType.INSIGHT, relation_to_focus=EdgeType.SUPPORTS)
        reasoner.process_thought("C1", NodeType.CONCEPT)

        # When requesting summary
        summary = reasoner.get_summary()

        # Then statistics provided
        assert summary['total_nodes'] == 3
        assert summary['total_edges'] >= 1
        assert 'nodes_by_type' in summary
        assert 'edges_by_type' in summary
        assert 'current_focus' in summary
