"""
Behavioral tests for PRISM Attention Mechanisms.

These tests define the target behavior for attention-based reasoning,
inspired by the Mountain of Attention chapter in the Wonderland roadmap.

Attention enables selective focus on relevant parts of the thought graph,
combining:
- Query-Key-Value attention from transformers
- Synaptic gating from neuroscience
- Relevance weighting from information retrieval

"The Caterpillar sat on a mushroom, paying attention to only what mattered."
"""

import pytest
from typing import List, Tuple, Optional


class TestPRISMAttentionMechanisms:
    """Tests for attention-based selective activation."""

    def test_query_attention_focuses_on_relevant_nodes(self):
        """
        Given a query, attention should weight relevant nodes higher.

        The Caterpillar's focused gaze - only the hookah and Alice matter.
        """
        from cortical.reasoning.prism_attention import AttentionLayer
        from cortical.reasoning import PRISMGraph, NodeType

        # Build a graph with various nodes - content must match query terms
        graph = PRISMGraph()
        graph.add_node("alice", NodeType.CONCEPT, "Alice attended the tea party")
        graph.add_node("queen", NodeType.CONCEPT, "Queen of Hearts plays croquet")
        graph.add_node("cards", NodeType.CONCEPT, "Playing cards as soldiers")
        graph.add_node("tea", NodeType.CONCEPT, "Tea party with the Hatter")
        graph.add_node("mushroom", NodeType.CONCEPT, "Magic mushroom in forest")

        # Create attention layer
        attention = AttentionLayer(graph)

        # Query about the tea party
        query = "Who was at the tea party?"
        weights = attention.attend(query)

        # Tea-related nodes should have higher attention
        assert weights["tea"] > weights["cards"]
        assert weights["alice"] > weights["queen"]  # Alice attended tea party

    def test_attention_heads_capture_different_relations(self):
        """
        Multi-head attention should capture different relationship types.

        One head for "who", another for "where", another for "what".
        """
        from cortical.reasoning.prism_attention import MultiHeadAttention
        from cortical.reasoning import PRISMGraph, NodeType, EdgeType

        graph = PRISMGraph()

        # Build a scene with content matching expected queries
        graph.add_node("alice", NodeType.ENTITY, "Alice is playing")
        graph.add_node("garden", NodeType.LOCATION, "The garden where croquet happens")
        graph.add_node("croquet", NodeType.ACTION, "Playing croquet game")
        graph.add_node("flamingo", NodeType.OBJECT, "Flamingo used as mallet tool")

        # Connect with typed edges
        graph.add_edge("alice", "garden", EdgeType.LOCATED_IN)
        graph.add_edge("alice", "croquet", EdgeType.PERFORMS)
        graph.add_edge("croquet", "flamingo", EdgeType.USES)

        # Multi-head attention with 3 heads
        mha = MultiHeadAttention(graph, num_heads=3)

        # Different queries activate different heads
        who_weights = mha.attend("Who is playing?")
        where_weights = mha.attend("Where is this happening?")
        what_weights = mha.attend("What tool is being used?")

        # Each query should peak on different nodes
        assert who_weights["alice"] > who_weights["garden"]
        assert where_weights["garden"] > where_weights["flamingo"]
        assert what_weights["flamingo"] > what_weights["alice"]

    def test_attention_with_synaptic_gating(self):
        """
        Attention should respect synaptic strength - strong paths get more focus.

        Well-worn paths through Wonderland are easier to traverse.
        """
        from cortical.reasoning.prism_attention import SynapticAttention
        from cortical.reasoning import PRISMGraph, NodeType, EdgeType

        graph = PRISMGraph()
        graph.add_node("rabbit", NodeType.CONCEPT, "White Rabbit")
        graph.add_node("hole", NodeType.CONCEPT, "Rabbit hole")
        graph.add_node("watch", NodeType.CONCEPT, "Pocket watch")
        graph.add_node("burrow", NodeType.CONCEPT, "Underground burrow")

        # Create synaptic edges with different strengths
        graph.add_synaptic_edge("rabbit", "hole", EdgeType.SUPPORTS, weight=5.0)
        graph.add_synaptic_edge("rabbit", "watch", EdgeType.SUPPORTS, weight=3.0)
        graph.add_synaptic_edge("hole", "burrow", EdgeType.SUPPORTS, weight=1.0)

        attention = SynapticAttention(graph)

        # Attend from "rabbit" - stronger edges get more attention
        weights = attention.attend_from("rabbit")

        assert weights["hole"] > weights["watch"]
        assert weights["watch"] > weights["burrow"]

    def test_attention_learns_from_feedback(self):
        """
        Attention patterns should learn from reinforcement.

        Alice learns to pay attention to the right clues.
        """
        from cortical.reasoning.prism_attention import LearnableAttention
        from cortical.reasoning import PRISMGraph, NodeType

        graph = PRISMGraph()
        # All nodes should have "wonderland" in content for baseline attention
        graph.add_node("key", NodeType.OBJECT, "Golden key in wonderland unlocks door")
        graph.add_node("door", NodeType.OBJECT, "Tiny door in wonderland garden")
        graph.add_node("cake", NodeType.OBJECT, "Eat me cake in wonderland grows big")
        graph.add_node("bottle", NodeType.OBJECT, "Drink me bottle in wonderland shrinks")

        attention = LearnableAttention(graph)

        # Initial attention with common term
        initial = attention.attend("What items are in wonderland?")

        # Provide feedback - key and bottle were relevant
        attention.reinforce(["key", "bottle"], reward=1.0)
        attention.reinforce(["cake"], reward=-0.5)  # Cake made her too big

        # After learning, attention should shift
        learned = attention.attend("What items are in wonderland?")

        # Key and bottle should increase
        assert learned["key"] > initial["key"]
        assert learned["bottle"] > initial["bottle"]
        # Cake should decrease (or at least be less than key/bottle)
        assert learned["cake"] < learned["key"]
        assert learned["cake"] < learned["bottle"]

    def test_temporal_attention_over_thought_sequence(self):
        """
        Attention over a sequence of thoughts, like reading a story.

        "Begin at the beginning and go on till you come to the end."
        """
        from cortical.reasoning.prism_attention import TemporalAttention

        # A sequence of thoughts (like the SLM generates)
        thought_sequence = [
            "Alice fell down the rabbit hole",
            "She found a tiny door",
            "The door led to a beautiful garden",
            "But Alice was too big to fit",
            "She drank from a bottle labeled DRINK ME",
        ]

        attention = TemporalAttention()
        attention.process_sequence(thought_sequence)

        # Query about the current situation
        weights = attention.attend("What is Alice's problem?")

        # Recent context and problem statement should have high attention
        assert weights[3] > weights[0]  # "too big" is the problem
        assert weights[4] > weights[1]  # solution attempt is relevant

    def test_cross_system_attention_integration(self):
        """
        Attention should integrate across GoT, SLM, and PLN.

        The Grand Unified Theory of PRISM attention.
        """
        from cortical.reasoning.prism_attention import UnifiedAttention
        from cortical.reasoning import PRISMGraph, NodeType
        from cortical.reasoning.prism_slm import PRISMLanguageModel
        from cortical.reasoning.prism_pln import PLNReasoner

        # All three systems
        graph = PRISMGraph()
        slm = PRISMLanguageModel()
        pln = PLNReasoner()

        # Train on Wonderland content
        slm.train("The Cheshire Cat grinned and slowly disappeared.")
        pln.assert_fact("cheshire", strength=0.9)
        graph.add_node("cheshire", NodeType.ENTITY, content="Cheshire Cat can disappear grin")

        # Unified attention combines all three
        unified = UnifiedAttention(graph, slm, pln)

        result = unified.attend("What can disappear while grinning?")

        # Should identify Cheshire Cat
        assert "cheshire" in result.top_nodes
        assert result.slm_fluency > 0.0  # SLM confirms language patterns


class TestAttentionWithPLNReasoning:
    """Tests for attention guiding probabilistic inference."""

    def test_attention_focuses_pln_inference(self):
        """
        Attention should guide which inference paths to explore.

        Don't search the whole garden - follow the Cheshire Cat's gaze.
        """
        from cortical.reasoning.prism_attention import AttentionGuidedReasoner
        from cortical.reasoning.prism_pln import PLNReasoner

        pln = PLNReasoner()

        # Assert facts that will be queried
        pln.assert_fact("cat", strength=0.95)
        pln.assert_fact("cheshire", strength=0.99)
        pln.assert_fact("can_disappear", strength=0.99)

        reasoner = AttentionGuidedReasoner(pln)

        # Query about cats - attention should focus on cat-related rules
        result = reasoner.query_with_attention("What is special about the Cheshire Cat?")

        # Should find can_disappear through focused search
        assert "can_disappear" in result.top_nodes or "cheshire" in result.top_nodes
        # Efficient search - limited exploration
        assert result.rules_explored <= 10


class TestAttentionVisualization:
    """Tests for attention visualization and interpretability."""

    def test_attention_heatmap_generation(self):
        """
        Generate interpretable attention heatmaps.

        "Curiouser and curiouser!" - see where the model looks.
        """
        from cortical.reasoning.prism_attention import AttentionVisualizer
        from cortical.reasoning import PRISMGraph, NodeType

        graph = PRISMGraph()
        nodes = ["alice", "rabbit", "queen", "hatter", "cat"]
        # Add nodes with content matching query terms
        graph.add_node("alice", NodeType.ENTITY, "Alice in Wonderland")
        graph.add_node("rabbit", NodeType.ENTITY, "White Rabbit")
        graph.add_node("queen", NodeType.ENTITY, "Queen of Hearts")
        graph.add_node("hatter", NodeType.ENTITY, "Mad Hatter hosted the tea party")
        graph.add_node("cat", NodeType.ENTITY, "Cheshire Cat")

        visualizer = AttentionVisualizer(graph)

        heatmap = visualizer.generate_heatmap(
            query="Who hosted the tea party?",
            nodes=nodes
        )

        # Heatmap should be a matrix of attention weights
        assert heatmap.shape == (len(nodes), len(nodes))
        assert heatmap.sum() > 0  # Non-trivial attention

        # Hatter should have highest self-attention for this query
        hatter_idx = nodes.index("hatter")
        assert heatmap[hatter_idx, hatter_idx] == heatmap.max()
