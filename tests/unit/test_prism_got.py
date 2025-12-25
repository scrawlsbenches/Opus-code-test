"""
Tests for Predictive Reasoning through Incremental Synaptic Memory Graph of Thought (PRISM-GoT).

This module tests the synaptic memory and predictive reasoning capabilities
that extend the Graph of Thought framework with biologically-inspired learning.

Key concepts tested:
1. SynapticEdge - Edges with activation history, decay, and prediction accuracy
2. SynapticMemoryGraph - ThoughtGraph with plasticity rules
3. PlasticityRules - Hebbian, Anti-Hebbian, and Reward-based learning
4. IncrementalReasoner - Prediction and incremental graph building
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# These imports will fail until we implement the modules
from cortical.reasoning.prism_got import (
    SynapticEdge,
    SynapticMemoryGraph,
    PlasticityRules,
    IncrementalReasoner,
    ActivationTrace,
    PredictionResult,
)
from cortical.reasoning.graph_of_thought import NodeType, EdgeType


class TestSynapticEdge:
    """Tests for SynapticEdge - edges with synaptic plasticity."""

    def test_synaptic_edge_creation(self):
        """Test basic SynapticEdge creation with default values."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
        )

        assert edge.source_id == "Q1"
        assert edge.target_id == "H1"
        assert edge.edge_type == EdgeType.EXPLORES
        assert edge.weight == 1.0
        assert edge.confidence == 1.0
        assert edge.last_activation_time is None
        assert edge.activation_count == 0
        assert edge.prediction_accuracy == 0.5  # Prior: uncertain
        assert edge.decay_factor == 0.99  # Default slow decay

    def test_synaptic_edge_with_activation_history(self):
        """Test SynapticEdge tracking activation over time."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=0.8,
        )

        # First activation
        edge.record_activation(timestamp=datetime(2025, 12, 25, 10, 0, 0))
        assert edge.activation_count == 1
        assert edge.last_activation_time == datetime(2025, 12, 25, 10, 0, 0)

        # Second activation
        edge.record_activation(timestamp=datetime(2025, 12, 25, 10, 5, 0))
        assert edge.activation_count == 2
        assert edge.last_activation_time == datetime(2025, 12, 25, 10, 5, 0)

    def test_synaptic_edge_weight_decay(self):
        """Test that edge weight decays over time without activation."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=1.0,
            decay_factor=0.9,  # Fast decay for testing
        )

        # Record initial activation
        edge.record_activation(timestamp=datetime(2025, 12, 25, 10, 0, 0))

        # Apply decay for 5 time steps
        for i in range(5):
            edge.apply_decay()

        # Weight should have decayed: 1.0 * 0.9^5 ≈ 0.59
        assert 0.58 < edge.weight < 0.60

    def test_synaptic_edge_prediction_tracking(self):
        """Test tracking prediction accuracy."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
        )

        # Record some predictions and outcomes
        edge.record_prediction_outcome(correct=True)
        edge.record_prediction_outcome(correct=True)
        edge.record_prediction_outcome(correct=False)

        # 2/3 correct with Beta prior smoothing: (2+1)/(3+2) = 3/5 = 0.6
        assert 0.55 <= edge.prediction_accuracy <= 0.75

    def test_synaptic_edge_strengthening(self):
        """Test Hebbian strengthening when activated."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=0.5,
        )

        # Strengthen the connection
        edge.strengthen(amount=0.1)
        assert edge.weight == 0.6

        # Weight should not exceed 1.0
        edge.strengthen(amount=0.5)
        assert edge.weight == 1.0

    def test_synaptic_edge_weakening(self):
        """Test Anti-Hebbian weakening when not co-activated."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=0.5,
        )

        # Weaken the connection
        edge.weaken(amount=0.1)
        assert edge.weight == 0.4

        # Weight should not go below 0.0
        edge.weaken(amount=0.5)
        assert edge.weight == 0.0

    def test_synaptic_edge_serialization(self):
        """Test serialization and deserialization of SynapticEdge."""
        edge = SynapticEdge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=0.8,
            decay_factor=0.95,
        )
        edge.record_activation(timestamp=datetime(2025, 12, 25, 10, 0, 0))
        edge.record_prediction_outcome(correct=True)

        # Serialize
        data = edge.to_dict()

        # Deserialize
        restored = SynapticEdge.from_dict(data)

        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.edge_type == edge.edge_type
        assert restored.weight == edge.weight
        assert restored.activation_count == edge.activation_count
        assert restored.decay_factor == edge.decay_factor


class TestActivationTrace:
    """Tests for ActivationTrace - history of node activations."""

    def test_activation_trace_creation(self):
        """Test creating an activation trace."""
        trace = ActivationTrace(node_id="Q1", max_history=100)

        assert trace.node_id == "Q1"
        assert len(trace.history) == 0
        assert trace.total_activations == 0

    def test_activation_trace_recording(self):
        """Test recording activations with context."""
        trace = ActivationTrace(node_id="Q1")

        trace.record(
            timestamp=datetime(2025, 12, 25, 10, 0, 0),
            context={"trigger": "user_question", "session_id": "abc123"}
        )

        assert trace.total_activations == 1
        assert len(trace.history) == 1
        assert trace.history[0]["context"]["trigger"] == "user_question"

    def test_activation_trace_frequency(self):
        """Test calculating activation frequency."""
        trace = ActivationTrace(node_id="Q1")

        # Record 10 activations over 10 minutes (relative to now)
        now = datetime.now()
        for i in range(10):
            # Record going backwards from now to stay within the window
            trace.record(timestamp=now - timedelta(minutes=9-i))

        # Frequency should be ~1 per minute (10 activations in 10 minute window)
        freq = trace.get_frequency(window_minutes=10)
        assert 0.9 < freq < 1.1

    def test_activation_trace_recent_patterns(self):
        """Test getting recent activation patterns."""
        trace = ActivationTrace(node_id="Q1")

        base_time = datetime(2025, 12, 25, 10, 0, 0)
        for i in range(5):
            trace.record(
                timestamp=base_time + timedelta(minutes=i),
                context={"phase": "question" if i < 2 else "answer"}
            )

        recent = trace.get_recent(n=3)
        assert len(recent) == 3
        assert all(r["context"]["phase"] == "answer" for r in recent[:2])


class TestSynapticMemoryGraph:
    """Tests for SynapticMemoryGraph - ThoughtGraph with plasticity."""

    def test_synaptic_graph_creation(self):
        """Test creating a synaptic memory graph."""
        graph = SynapticMemoryGraph()

        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_add_node_with_activation_trace(self):
        """Test that nodes get activation traces."""
        graph = SynapticMemoryGraph()

        node = graph.add_node("Q1", NodeType.QUESTION, "What auth method?")

        assert "Q1" in graph.activation_traces
        assert graph.activation_traces["Q1"].node_id == "Q1"

    def test_add_synaptic_edge(self):
        """Test adding synaptic edges."""
        graph = SynapticMemoryGraph()
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")

        edge = graph.add_synaptic_edge(
            from_id="Q1",
            to_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=0.8,
        )

        assert isinstance(edge, SynapticEdge)
        assert edge.weight == 0.8

    def test_activate_node(self):
        """Test node activation updates traces and edges."""
        graph = SynapticMemoryGraph()
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES)

        # Activate Q1
        graph.activate_node("Q1", context={"trigger": "user_input"})

        # Check trace was updated
        trace = graph.activation_traces["Q1"]
        assert trace.total_activations == 1

        # Check outgoing edges were activated
        edges = graph.get_synaptic_edges_from("Q1")
        assert edges[0].activation_count == 1

    def test_co_activation_strengthening(self):
        """Test that co-activated nodes strengthen their connection."""
        graph = SynapticMemoryGraph()
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        edge = graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.5)

        initial_weight = edge.weight

        # Activate both nodes (co-activation)
        graph.activate_node("Q1")
        graph.activate_node("H1")

        # Apply Hebbian learning
        graph.apply_hebbian_learning(time_window_seconds=60)

        # Edge should be strengthened
        assert edge.weight > initial_weight

    def test_decay_unused_connections(self):
        """Test that unused connections decay over time."""
        graph = SynapticMemoryGraph()
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        edge = graph.add_synaptic_edge(
            "Q1", "H1", EdgeType.EXPLORES,
            weight=1.0, decay_factor=0.9
        )

        initial_weight = edge.weight

        # Apply decay without activation
        graph.apply_global_decay()

        assert edge.weight < initial_weight

    def test_predict_next_thoughts(self):
        """Test predicting likely next thoughts from current state."""
        graph = SynapticMemoryGraph()

        # Build a simple reasoning graph
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        graph.add_node("H2", NodeType.HYPOTHESIS, "Use OAuth")
        graph.add_node("E1", NodeType.EVIDENCE, "Team knows JWT")

        # Create edges with different weights
        graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)
        graph.add_synaptic_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.3)
        graph.add_synaptic_edge("H1", "E1", EdgeType.SUPPORTS, weight=0.8)

        # Activate Q1 and predict next thoughts
        graph.activate_node("Q1")
        predictions = graph.predict_next_thoughts("Q1", top_n=3)

        # H1 should be most likely (highest weight)
        assert len(predictions) > 0
        assert predictions[0].node_id == "H1"
        assert predictions[0].probability > predictions[1].probability

    def test_reward_learning(self):
        """Test reward-based learning strengthens successful paths."""
        graph = SynapticMemoryGraph()

        # Build a reasoning path
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        graph.add_node("D1", NodeType.DECISION, "Implement JWT")

        edge1 = graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.5)
        edge2 = graph.add_synaptic_edge("H1", "D1", EdgeType.SUGGESTS, weight=0.5)

        initial_weights = [edge1.weight, edge2.weight]

        # Apply reward to the successful path
        graph.apply_reward(path=["Q1", "H1", "D1"], reward=0.5)

        # Both edges in the path should be strengthened
        assert edge1.weight > initial_weights[0]
        assert edge2.weight > initial_weights[1]


class TestPlasticityRules:
    """Tests for PlasticityRules - learning algorithms."""

    def test_hebbian_rule(self):
        """Test Hebbian learning: 'neurons that fire together wire together'."""
        rules = PlasticityRules(
            hebbian_rate=0.1,
            anti_hebbian_rate=0.05,
            reward_rate=0.2,
        )

        # Create test edge
        edge = SynapticEdge(
            source_id="A",
            target_id="B",
            edge_type=EdgeType.SUPPORTS,
            weight=0.5,
        )

        # Both nodes active within time window
        source_active = True
        target_active = True

        new_weight = rules.apply_hebbian(edge, source_active, target_active)
        assert new_weight > edge.weight

    def test_anti_hebbian_rule(self):
        """Test Anti-Hebbian: weaken unused connections."""
        rules = PlasticityRules(
            hebbian_rate=0.1,
            anti_hebbian_rate=0.05,
            min_weight=0.01,
        )

        edge = SynapticEdge(
            source_id="A",
            target_id="B",
            edge_type=EdgeType.SUPPORTS,
            weight=0.5,
        )

        # Source active but target not active (no co-activation)
        new_weight = rules.apply_anti_hebbian(edge, source_active=True, target_active=False)
        assert new_weight < edge.weight

    def test_reward_modulated_learning(self):
        """Test reward-modulated learning for successful outcomes."""
        rules = PlasticityRules(reward_rate=0.2)

        edge = SynapticEdge(
            source_id="A",
            target_id="B",
            edge_type=EdgeType.SUPPORTS,
            weight=0.5,
        )

        # Positive reward strengthens
        new_weight = rules.apply_reward(edge, reward=1.0)
        assert new_weight > edge.weight

        # Negative reward weakens
        edge.weight = 0.5  # Reset
        new_weight = rules.apply_reward(edge, reward=-0.5)
        assert new_weight < edge.weight

    def test_weight_bounds(self):
        """Test that weights stay within bounds."""
        rules = PlasticityRules(
            hebbian_rate=0.5,
            min_weight=0.0,
            max_weight=1.0,
        )

        # Edge near maximum
        edge = SynapticEdge(
            source_id="A",
            target_id="B",
            edge_type=EdgeType.SUPPORTS,
            weight=0.99,
        )

        new_weight = rules.apply_hebbian(edge, True, True)
        assert new_weight <= 1.0

        # Edge near minimum
        edge.weight = 0.01
        new_weight = rules.apply_anti_hebbian(edge, True, False)
        assert new_weight >= 0.0


class TestIncrementalReasoner:
    """Tests for IncrementalReasoner - orchestrates incremental graph building."""

    def test_reasoner_creation(self):
        """Test creating an incremental reasoner."""
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        assert reasoner.graph is graph
        assert reasoner.current_focus is None

    def test_process_thought(self):
        """Test processing a single thought incrementally."""
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        # Process a question
        node = reasoner.process_thought(
            content="What authentication method should we use?",
            node_type=NodeType.QUESTION,
        )

        assert node.id is not None
        assert node.node_type == NodeType.QUESTION
        assert reasoner.current_focus == node.id

    def test_process_related_thought(self):
        """Test processing a thought related to current focus."""
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        # Process question
        q1 = reasoner.process_thought(
            content="What auth method?",
            node_type=NodeType.QUESTION,
        )

        # Process related hypothesis
        h1 = reasoner.process_thought(
            content="Use JWT tokens",
            node_type=NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES,
        )

        # Should create edge from Q1 to H1
        edges = graph.get_synaptic_edges_from(q1.id)
        assert len(edges) == 1
        assert edges[0].target_id == h1.id
        assert edges[0].edge_type == EdgeType.EXPLORES

    def test_predict_and_verify(self):
        """Test prediction and verification cycle."""
        graph = SynapticMemoryGraph()

        # Pre-populate with some learned patterns
        graph.add_node("Q1", NodeType.QUESTION, "What auth method?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        graph.add_node("H2", NodeType.HYPOTHESIS, "Use OAuth")

        edge1 = graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)
        edge2 = graph.add_synaptic_edge("Q1", "H2", EdgeType.EXPLORES, weight=0.3)

        reasoner = IncrementalReasoner(graph)

        # Get predictions from Q1
        predictions = reasoner.predict_next("Q1")

        # Verify the prediction
        reasoner.verify_prediction(
            predicted_node_id="H1",
            actual_node_id="H1",  # Correct prediction
        )

        # Edge accuracy should improve
        assert edge1.prediction_accuracy > 0.5

    def test_automatic_edge_creation_from_similarity(self):
        """Test that similar content creates SIMILAR edges."""
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(
            graph,
            auto_link_similar=True,
            similarity_threshold=0.5,  # Lower threshold for word-based Jaccard
        )

        # Process similar thoughts (high word overlap)
        h1 = reasoner.process_thought(
            content="Use JWT tokens for API authentication security",
            node_type=NodeType.HYPOTHESIS,
        )

        h2 = reasoner.process_thought(
            content="JWT tokens for API authentication and security",
            node_type=NodeType.HYPOTHESIS,
        )

        # Should detect similarity and create edge from h2 to h1
        # (similarity is checked when adding h2, linking back to h1)
        edges = graph.get_synaptic_edges_from(h2.id)
        similar_edges = [e for e in edges if e.edge_type == EdgeType.SIMILAR]

        assert len(similar_edges) >= 1
        assert similar_edges[0].target_id == h1.id

    def test_incremental_learning_session(self):
        """Test a full incremental learning session."""
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        # Session: Question → Hypothesis → Evidence → Decision
        q1 = reasoner.process_thought(
            "How should we handle user sessions?",
            NodeType.QUESTION,
        )

        h1 = reasoner.process_thought(
            "Use Redis for session storage",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES,
        )

        e1 = reasoner.process_thought(
            "Redis has 99.99% uptime in our infrastructure",
            NodeType.EVIDENCE,
            relation_to_focus=EdgeType.SUPPORTS,
        )

        d1 = reasoner.process_thought(
            "Implement Redis sessions with 1-hour TTL",
            NodeType.DECISION,
            relation_to_focus=EdgeType.JUSTIFIES,
        )

        # Mark the decision as successful
        reasoner.mark_outcome_success(path=[q1.id, h1.id, e1.id, d1.id])

        # All edges in path should be strengthened
        edges = graph.get_synaptic_edges_from(q1.id)
        assert edges[0].weight > 0.5  # Strengthened from initial

    def test_get_reasoning_summary(self):
        """Test getting a summary of the reasoning graph."""
        graph = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(graph)

        # Build some reasoning
        reasoner.process_thought("Auth question?", NodeType.QUESTION)
        reasoner.process_thought("JWT hypothesis", NodeType.HYPOTHESIS, EdgeType.EXPLORES)
        reasoner.process_thought("OAuth hypothesis", NodeType.HYPOTHESIS, EdgeType.EXPLORES)

        summary = reasoner.get_summary()

        assert summary["total_nodes"] == 3
        assert summary["total_edges"] == 2
        assert summary["nodes_by_type"]["question"] == 1
        assert summary["nodes_by_type"]["hypothesis"] == 2


class TestIntegration:
    """Integration tests for PRISM-GoT with existing GoT infrastructure."""

    def test_integration_with_thought_graph(self):
        """Test that SynapticMemoryGraph is compatible with ThoughtGraph."""
        from cortical.reasoning.thought_graph import ThoughtGraph

        # SynapticMemoryGraph should inherit from ThoughtGraph
        graph = SynapticMemoryGraph()
        assert isinstance(graph, ThoughtGraph) or hasattr(graph, 'nodes')

    def test_persistence_roundtrip(self):
        """Test saving and loading synaptic graph state."""
        graph = SynapticMemoryGraph()

        # Build a graph with synaptic state
        graph.add_node("Q1", NodeType.QUESTION, "What auth?")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT")
        edge = graph.add_synaptic_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)

        # Activate to create history
        graph.activate_node("Q1")
        graph.activate_node("H1")
        edge.record_prediction_outcome(correct=True)

        # Serialize
        state = graph.to_dict()

        # Restore
        restored = SynapticMemoryGraph.from_dict(state)

        assert restored.node_count() == 2
        assert restored.edge_count() == 1

        restored_edge = restored.get_synaptic_edges_from("Q1")[0]
        assert restored_edge.weight == 0.8
        assert restored_edge.activation_count == 1

    def test_concurrent_activation_handling(self):
        """Test handling multiple concurrent node activations."""
        graph = SynapticMemoryGraph()

        # Create multiple nodes
        for i in range(5):
            graph.add_node(f"N{i}", NodeType.CONCEPT, f"Concept {i}")

        # Create edges
        for i in range(4):
            graph.add_synaptic_edge(f"N{i}", f"N{i+1}", EdgeType.REQUIRES)

        # Activate multiple nodes "simultaneously"
        timestamp = datetime.now()
        for i in range(5):
            graph.activate_node(f"N{i}", context={}, timestamp=timestamp)

        # Apply learning
        graph.apply_hebbian_learning(time_window_seconds=1)

        # All edges should be strengthened (all co-activated)
        for i in range(4):
            edges = graph.get_synaptic_edges_from(f"N{i}")
            assert edges[0].weight > 1.0  # Strengthened from initial 1.0
