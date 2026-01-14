"""
Unit tests for AttentionGraph: self-attention based graph neural network.

Tests the core components:
- AttentionNode with learnable embeddings
- AttentionEdge for attention masking
- Forward pass (self-attention)
- Backward pass (gradient computation through attention)
- TrainableGraphProtocol compliance
- Save/load state functionality
- Convenience functions for causal graphs
- Visualization utilities
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from cortical.graph.attention import (
    # Core types
    AttentionGraph,
    AttentionNode,
    AttentionEdge,
    AttentionLayer,
    Parameter,
    # Protocol
    TrainableGraphProtocol,
    # Utility functions
    scaled_dot_product_attention,
    attention_backward,
    create_causal_attention_graph,
)


# =============================================================================
# Graph Construction Tests
# =============================================================================


class TestAttentionGraphConstruction:
    """Tests for AttentionGraph construction."""

    def test_create_graph(self):
        """Test basic graph creation."""
        graph = AttentionGraph(embedding_dim=8)

        assert graph.embedding_dim == 8
        assert graph.num_heads == 1
        assert graph.use_bias is False
        assert graph.dropout == 0.0

    def test_create_graph_with_config(self):
        """Test graph creation with custom config."""
        graph = AttentionGraph(
            embedding_dim=16,
            num_heads=4,
            use_bias=True,
            dropout=0.1,
            seed=42,
        )

        assert graph.embedding_dim == 16
        assert graph.num_heads == 4
        assert graph.use_bias is True
        assert graph.dropout == 0.1

    def test_add_node_creates_attention_node(self):
        """Test that add_node creates AttentionNode with embedding."""
        graph = AttentionGraph(embedding_dim=8, seed=42)
        node = graph.add_node("node1")

        assert isinstance(node, AttentionNode)
        assert node.id == "node1"
        assert node.embedding is not None
        assert isinstance(node.embedding, Parameter)
        assert len(node.embedding.data) == 8
        assert node.embedding.requires_grad is True

    def test_add_node_with_custom_embedding(self):
        """Test adding node with custom embedding."""
        graph = AttentionGraph(embedding_dim=4)
        custom_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        node = graph.add_node("node1", embedding=custom_embedding)

        assert_array_almost_equal(node.embedding.data, custom_embedding)

    def test_add_node_embedding_dimension(self):
        """Test that node embedding respects graph embedding_dim."""
        graph = AttentionGraph(embedding_dim=16)
        node = graph.add_node("node1")

        assert len(node.embedding.data) == 16

    def test_add_edge_creates_attention_edge(self):
        """Test that add_edge creates AttentionEdge."""
        graph = AttentionGraph(embedding_dim=8)
        graph.add_node("A")
        graph.add_node("B")
        edge = graph.add_edge("A", "B")

        assert isinstance(edge, AttentionEdge)
        assert edge.source_id == "A"
        assert edge.target_id == "B"

    def test_node_and_edge_counts(self):
        """Test that node/edge counts are correct."""
        graph = AttentionGraph(embedding_dim=8)

        # Add nodes
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")

        assert graph.node_count == 3

        # Add edges
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "C")

        assert graph.edge_count == 3

    def test_layer_initialization_cleared_between_nodes(self):
        """Test that layer_inputs and layer_outputs start empty."""
        graph = AttentionGraph(embedding_dim=8)
        node = graph.add_node("node1")

        assert node.layer_inputs == []
        assert node.layer_outputs == []


# =============================================================================
# TrainableGraphProtocol Compliance Tests
# =============================================================================


class TestTrainableGraphProtocol:
    """Tests for TrainableGraphProtocol compliance."""

    @pytest.fixture
    def graph(self):
        """Create a simple attention graph."""
        g = AttentionGraph(embedding_dim=8, seed=42)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        return g

    def test_satisfies_protocol(self, graph):
        """Test that AttentionGraph satisfies TrainableGraphProtocol."""
        assert isinstance(graph, TrainableGraphProtocol)

    def test_has_embedding_dim_property(self, graph):
        """Test graph has embedding_dim property."""
        assert hasattr(graph, "embedding_dim")
        assert isinstance(graph.embedding_dim, int)
        assert graph.embedding_dim == 8

    def test_parameters_returns_list(self, graph):
        """Test parameters() returns list of Parameters."""
        params = graph.parameters()

        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, Parameter) for p in params)

    def test_parameters_includes_embeddings(self, graph):
        """Test parameters() includes node embeddings."""
        params = graph.parameters()

        # Should have at least 2 node embeddings
        node_params = [p for p in params if "embedding" in p.name]
        assert len(node_params) >= 2

    def test_parameters_includes_attention_layers_after_forward(self, graph):
        """Test parameters() includes attention layer params after forward."""
        graph.forward(num_layers=1)
        params = graph.parameters()

        # Should have attention layer parameters (W_q, W_k, W_v, W_o)
        # Parameter names use "layer_N_W_q" format after naming fix
        attention_params = [p for p in params if "W_q" in p.name or "W_k" in p.name]
        assert len(attention_params) >= 2  # At least W_q and W_k for one layer

    def test_forward_signature(self, graph):
        """Test forward() has correct signature."""
        # Should accept num_layers and input_nodes
        outputs = graph.forward(num_layers=1, input_nodes=None)

        assert isinstance(outputs, dict)

    def test_forward_return_type(self, graph):
        """Test forward() returns Dict[str, Array]."""
        outputs = graph.forward(num_layers=1)

        assert isinstance(outputs, dict)
        assert "A" in outputs
        assert "B" in outputs
        assert isinstance(outputs["A"], np.ndarray)
        assert isinstance(outputs["B"], np.ndarray)

    def test_backward_signature(self, graph):
        """Test backward() has correct signature."""
        graph.forward(num_layers=1)

        # Should accept output_gradients and num_layers
        output_grads = {"B": np.ones(8)}
        graph.backward(output_grads, num_layers=1)

        # Should not raise any errors

    def test_zero_grad_clears_all_gradients(self, graph):
        """Test zero_grad() clears all gradients."""
        # Forward and backward to create gradients
        graph.forward(num_layers=1)
        graph.backward({"B": np.ones(8)}, num_layers=1)

        # Verify gradients exist
        has_grads = any(p.grad is not None for p in graph.parameters())
        assert has_grads

        # Zero gradients
        graph.zero_grad()

        # All gradients should be None
        for param in graph.parameters():
            assert param.grad is None

    def test_save_state_returns_dict(self, graph):
        """Test save_state() returns dict with expected keys."""
        graph.forward(num_layers=1)  # Initialize layers
        state = graph.save_state()

        assert isinstance(state, dict)
        assert "embeddings" in state
        assert "layers" in state

    def test_save_state_embeddings(self, graph):
        """Test save_state() includes all node embeddings."""
        state = graph.save_state()

        assert "A" in state["embeddings"]
        assert "B" in state["embeddings"]
        assert isinstance(state["embeddings"]["A"], np.ndarray)
        assert len(state["embeddings"]["A"]) == 8

    def test_save_state_layers(self, graph):
        """Test save_state() includes attention layer parameters."""
        graph.forward(num_layers=2)  # Initialize 2 layers
        state = graph.save_state()

        assert len(state["layers"]) == 2
        for layer_state in state["layers"]:
            assert "W_q" in layer_state
            assert "W_k" in layer_state
            assert "W_v" in layer_state
            assert "W_o" in layer_state

    def test_load_state_restores_embeddings(self, graph):
        """Test load_state() restores node embeddings."""
        # Save initial state
        state = graph.save_state()
        original_embedding = graph.get_node("A").embedding.data.copy()

        # Modify embedding
        graph.get_node("A").embedding.data = np.zeros(8)

        # Load state
        graph.load_state(state)

        # Check restoration
        assert_array_almost_equal(
            graph.get_node("A").embedding.data,
            original_embedding
        )

    def test_load_state_restores_attention_parameters(self, graph):
        """Test load_state() restores attention layer parameters."""
        # Initialize and save
        graph.forward(num_layers=1)
        state = graph.save_state()
        original_W_q = state["layers"][0]["W_q"].copy()

        # Modify parameters
        graph._attention_layers[0].W_q.data = np.zeros_like(
            graph._attention_layers[0].W_q.data
        )

        # Load state
        graph.load_state(state)

        # Check restoration
        assert_array_almost_equal(
            graph._attention_layers[0].W_q.data,
            original_W_q
        )

    def test_save_load_round_trip(self, graph):
        """Test save/load round trip preserves values."""
        # Initialize layers
        graph.forward(num_layers=2)

        # Save state
        state1 = graph.save_state()

        # Modify graph
        for node in graph.nodes:
            if node.embedding:
                node.embedding.data = np.random.randn(8)
        for layer in graph._attention_layers:
            layer.W_q.data = np.random.randn(8, 8)

        # Load and save again
        graph.load_state(state1)
        state2 = graph.save_state()

        # States should be identical
        for node_id in state1["embeddings"]:
            assert_array_almost_equal(
                state1["embeddings"][node_id],
                state2["embeddings"][node_id]
            )


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestAttentionGraphForward:
    """Tests for forward pass through attention layers."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple 2-node graph."""
        g = AttentionGraph(embedding_dim=4, seed=42)
        g.add_node("A", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        g.add_node("B", embedding=np.array([0.0, 1.0, 0.0, 0.0]))
        g.add_edge("A", "B")
        return g

    def test_forward_single_layer(self, simple_graph):
        """Test forward pass with 1 layer."""
        outputs = simple_graph.forward(num_layers=1)

        assert "A" in outputs
        assert "B" in outputs
        assert len(outputs["A"]) == 4
        assert len(outputs["B"]) == 4

    def test_forward_two_layers(self, simple_graph):
        """Test forward pass with 2 layers."""
        outputs = simple_graph.forward(num_layers=2)

        assert "A" in outputs
        assert "B" in outputs
        assert len(outputs["A"]) == 4
        assert len(outputs["B"]) == 4

    def test_forward_three_layers(self, simple_graph):
        """Test forward pass with 3 layers."""
        outputs = simple_graph.forward(num_layers=3)

        assert "A" in outputs
        assert "B" in outputs
        assert len(outputs["A"]) == 4
        assert len(outputs["B"]) == 4

    def test_forward_creates_attention_layers(self, simple_graph):
        """Test forward creates necessary attention layers."""
        assert len(simple_graph._attention_layers) == 0

        simple_graph.forward(num_layers=3)

        assert len(simple_graph._attention_layers) == 3
        assert all(isinstance(layer, AttentionLayer) for layer in simple_graph._attention_layers)

    def test_forward_with_input_nodes_override(self, simple_graph):
        """Test forward with input_nodes override."""
        custom_inputs = {
            "A": np.array([5.0, 5.0, 5.0, 5.0]),
            "B": np.array([10.0, 10.0, 10.0, 10.0]),
        }

        outputs = simple_graph.forward(num_layers=1, input_nodes=custom_inputs)

        # Outputs should be based on custom inputs, not embeddings
        assert "A" in outputs
        assert "B" in outputs

    def test_forward_outputs_all_node_ids(self):
        """Test that outputs dict has all node IDs."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        outputs = graph.forward(num_layers=1)

        assert "A" in outputs
        assert "B" in outputs
        assert "C" in outputs

    def test_forward_clears_layer_memory(self):
        """Test that layer_inputs and layer_outputs are cleared between calls."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        node = graph.add_node("A")

        # First forward pass
        graph.forward(num_layers=2)
        first_call_inputs = len(node.layer_inputs)
        first_call_outputs = len(node.layer_outputs)

        # Second forward pass
        graph.forward(num_layers=2)
        second_call_inputs = len(node.layer_inputs)
        second_call_outputs = len(node.layer_outputs)

        # Should have same number of layers, not accumulating
        assert first_call_inputs == second_call_inputs
        assert first_call_outputs == second_call_outputs
        # For 2 layers: layer_inputs = initial + layer0_output = 2
        # layer_outputs = layer0_output + layer1_output = 2
        assert first_call_inputs == 2
        assert first_call_outputs == 2

    def test_forward_memory_leak_fix(self):
        """Test that memory leak is fixed (layer state cleared)."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        node = graph.add_node("A")

        # Multiple forward passes
        for _ in range(5):
            graph.forward(num_layers=2)

        # Should only have 2 entries each (initial + layer0_output for inputs,
        # layer0_output + layer1_output for outputs), not accumulating across calls
        assert len(node.layer_inputs) == 2
        assert len(node.layer_outputs) == 2

    def test_forward_stores_attention_weights(self, simple_graph):
        """Test that forward stores attention weights in nodes."""
        simple_graph.forward(num_layers=1)

        node_b = simple_graph.get_node("B")
        # B should have attention weights for A
        assert "A" in node_b.attention_weights
        assert isinstance(node_b.attention_weights["A"], float)

    def test_forward_causal_structure(self):
        """Test forward with causal structure (3 positions)."""
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=4, seed=42)

        outputs = graph.forward(num_layers=1)

        # All positions should have outputs
        assert "pos_0" in outputs
        assert "pos_1" in outputs
        assert "pos_2" in outputs

    def test_forward_node_without_incoming_edges(self):
        """Test forward for node without incoming edges."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B")

        outputs = graph.forward(num_layers=1)

        # Node A has no incoming edges, should still produce output
        assert "A" in outputs
        assert len(outputs["A"]) == 4


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestAttentionGraphBackward:
    """Tests for backward pass through attention layers."""

    @pytest.fixture
    def graph_with_forward(self):
        """Create graph and run forward pass."""
        g = AttentionGraph(embedding_dim=4, seed=42)
        g.add_node("A", embedding=np.ones(4))
        g.add_node("B", embedding=np.ones(4))
        g.add_edge("A", "B")
        g.forward(num_layers=1)
        return g

    def test_backward_computes_gradients(self, graph_with_forward):
        """Test that backward computes gradients."""
        output_grads = {"B": np.ones(4) * 0.1}
        graph_with_forward.backward(output_grads, num_layers=1)

        # Check that some parameters have gradients
        has_grads = any(p.grad is not None for p in graph_with_forward.parameters())
        assert has_grads

    def test_backward_gradients_reach_embeddings(self):
        """Test that gradients reach node embeddings."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        node_a = graph.add_node("A", embedding=np.ones(4))
        node_b = graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B")

        # Forward and backward
        graph.forward(num_layers=1)
        graph.backward({"B": np.ones(4)}, num_layers=1)

        # Both embeddings should have gradients
        assert node_a.embedding.grad is not None
        assert node_b.embedding.grad is not None

    def test_backward_gradients_reach_attention_parameters(self, graph_with_forward):
        """Test that gradients reach attention layer parameters."""
        graph_with_forward.backward({"B": np.ones(4)}, num_layers=1)

        # Attention layer parameters should have gradients
        layer = graph_with_forward._attention_layers[0]
        assert layer.W_q.grad is not None
        assert layer.W_k.grad is not None
        assert layer.W_v.grad is not None
        assert layer.W_o.grad is not None

    def test_backward_with_partial_output_gradients(self):
        """Test backward with partial output_gradients (only some nodes)."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_node("C", embedding=np.ones(4))
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        graph.forward(num_layers=1)

        # Only provide gradient for C
        graph.backward({"C": np.ones(4)}, num_layers=1)

        # Should not raise errors
        # C has incoming edges from B, so B should get gradient
        node_b = graph.get_node("B")
        node_c = graph.get_node("C")
        assert node_c.embedding.grad is not None
        # B should also receive gradient since C attends to it
        assert node_b.embedding.grad is not None

    def test_backward_multi_layer(self):
        """Test backward through multiple layers."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B")

        graph.forward(num_layers=2)
        graph.backward({"B": np.ones(4)}, num_layers=2)

        # Both layers should have gradients
        assert graph._attention_layers[0].W_q.grad is not None
        assert graph._attention_layers[1].W_q.grad is not None

    def test_backward_accumulates_gradients(self):
        """Test that backward accumulates gradients correctly."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        node_a = graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B")

        # First forward/backward
        graph.forward(num_layers=1)
        graph.backward({"B": np.ones(4) * 0.5}, num_layers=1)
        first_grad = node_a.embedding.grad.copy()

        # Second forward/backward without zero_grad
        graph.forward(num_layers=1)
        graph.backward({"B": np.ones(4) * 0.5}, num_layers=1)
        second_grad = node_a.embedding.grad

        # Gradients should have accumulated
        assert not np.allclose(first_grad, second_grad)
        assert np.all(np.abs(second_grad) >= np.abs(first_grad) - 1e-6)

    def test_gradient_flow_through_attention(self):
        """Test that gradients flow correctly through attention mechanism."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        node_a = graph.add_node("A", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        node_b = graph.add_node("B", embedding=np.array([0.0, 1.0, 0.0, 0.0]))
        graph.add_edge("A", "B")

        graph.forward(num_layers=1)
        outputs = graph.forward(num_layers=1)

        # Large gradient on B
        graph.zero_grad()
        graph.backward({"B": np.ones(4) * 10.0}, num_layers=1)

        # A should receive gradient through attention
        assert node_a.embedding.grad is not None
        assert np.any(node_a.embedding.grad != 0)


# =============================================================================
# Scaled Dot-Product Attention Tests
# =============================================================================


class TestScaledDotProductAttention:
    """Tests for scaled_dot_product_attention function."""

    def test_attention_output_shape(self):
        """Test that attention output has correct shape."""
        query = np.array([1.0, 0.0, 0.0, 0.0])
        keys = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        values = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        assert output.shape == (4,)
        assert weights.shape == (2,)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        query = np.array([1.0, 0.0, 0.0, 0.0])
        keys = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        values = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        assert abs(np.sum(weights) - 1.0) < 1e-6

    def test_attention_high_similarity(self):
        """Test attention with high query-key similarity."""
        query = np.array([1.0, 0.0, 0.0, 0.0])
        # First key matches query exactly
        keys = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        values = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Should attend more to first key
        assert weights[0] > weights[1]
        assert weights[0] > 0.5

    def test_attention_with_mask(self):
        """Test attention with masking."""
        query = np.array([1.0, 0.0, 0.0, 0.0])
        keys = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        values = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        mask = np.array([0.0, -np.inf])  # Mask out second position

        output, weights = scaled_dot_product_attention(query, keys, values, mask)

        # Second weight should be ~0
        assert weights[1] < 0.01
        assert weights[0] > 0.99


class TestAttentionBackward:
    """Tests for attention_backward function."""

    def test_backward_returns_three_gradients(self):
        """Test that backward returns gradients for query, keys, values."""
        query = np.array([1.0, 0.0, 0.0, 0.0])
        keys = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        values = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        grad_output = np.ones(4)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        assert grad_query.shape == query.shape
        assert grad_keys.shape == keys.shape
        assert grad_values.shape == values.shape

    def test_backward_gradient_shapes(self):
        """Test that backward gradients have correct shapes."""
        d_k = 8
        n_sources = 3

        query = np.random.randn(d_k)
        keys = np.random.randn(n_sources, d_k)
        values = np.random.randn(n_sources, d_k)

        output, weights = scaled_dot_product_attention(query, keys, values)
        grad_output = np.random.randn(d_k)

        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        assert grad_query.shape == (d_k,)
        assert grad_keys.shape == (n_sources, d_k)
        assert grad_values.shape == (n_sources, d_k)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestCreateCausalAttentionGraph:
    """Tests for create_causal_attention_graph function."""

    def test_creates_correct_number_of_nodes(self):
        """Test that causal graph has correct number of nodes."""
        graph = create_causal_attention_graph(seq_len=5, embedding_dim=8)

        assert graph.node_count == 5

    def test_creates_causal_edges(self):
        """Test that causal graph has correct edge structure."""
        graph = create_causal_attention_graph(seq_len=4, embedding_dim=8)

        # pos_1 should have edge from pos_0
        edges_to_1 = graph.edges_to("pos_1")
        assert len(edges_to_1) == 1
        assert edges_to_1[0].source_id == "pos_0"

        # pos_2 should have edges from pos_0 and pos_1
        edges_to_2 = graph.edges_to("pos_2")
        assert len(edges_to_2) == 2

        # pos_3 should have edges from pos_0, pos_1, pos_2
        edges_to_3 = graph.edges_to("pos_3")
        assert len(edges_to_3) == 3

    def test_causal_edge_count(self):
        """Test correct number of edges for causal mask (n*(n-1)/2)."""
        n = 5
        graph = create_causal_attention_graph(seq_len=n, embedding_dim=8)

        expected_edges = n * (n - 1) // 2
        assert graph.edge_count == expected_edges

    def test_no_forward_edges(self):
        """Test that causal graph has no forward-looking edges."""
        graph = create_causal_attention_graph(seq_len=5, embedding_dim=8)

        # pos_0 should have no incoming edges
        edges_to_0 = graph.edges_to("pos_0")
        assert len(edges_to_0) == 0

        # No position should have edges from later positions
        for i in range(5):
            edges_to_i = graph.edges_to(f"pos_{i}")
            for edge in edges_to_i:
                source_idx = int(edge.source_id.split("_")[1])
                target_idx = i
                assert source_idx < target_idx

    def test_custom_embedding_dim(self):
        """Test causal graph with custom embedding dimension."""
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=16)

        assert graph.embedding_dim == 16
        for node in graph.nodes:
            assert len(node.embedding.data) == 16

    def test_passes_kwargs_to_graph(self):
        """Test that kwargs are passed to AttentionGraph constructor."""
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=8,
            use_bias=True,
            dropout=0.1,
            seed=42,
        )

        assert graph.use_bias is True
        assert graph.dropout == 0.1


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Tests for attention visualization utilities."""

    @pytest.fixture
    def graph_with_attention(self):
        """Create graph and run forward to get attention weights."""
        g = AttentionGraph(embedding_dim=4, seed=42)
        g.add_node("A", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        g.add_node("B", embedding=np.array([0.0, 1.0, 0.0, 0.0]))
        g.add_node("C", embedding=np.array([0.0, 0.0, 1.0, 0.0]))
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.forward(num_layers=1)
        return g

    def test_get_attention_weights_returns_dict(self, graph_with_attention):
        """Test get_attention_weights() returns dict of dicts."""
        weights = graph_with_attention.get_attention_weights()

        assert isinstance(weights, dict)
        assert "A" in weights
        assert "B" in weights
        assert "C" in weights

    def test_get_attention_weights_structure(self, graph_with_attention):
        """Test attention weights have correct structure."""
        weights = graph_with_attention.get_attention_weights()

        # Node C attends to A and B
        assert isinstance(weights["C"], dict)
        assert "A" in weights["C"]
        assert "B" in weights["C"]
        assert isinstance(weights["C"]["A"], float)
        assert isinstance(weights["C"]["B"], float)

    def test_get_attention_weights_empty_for_no_sources(self, graph_with_attention):
        """Test attention weights empty for nodes with no sources."""
        weights = graph_with_attention.get_attention_weights()

        # Node A has no incoming edges
        assert weights["A"] == {}

    def test_visualize_attention_returns_string(self, graph_with_attention):
        """Test visualize_attention() returns string."""
        viz = graph_with_attention.visualize_attention("C")

        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_visualize_attention_contains_source_ids(self, graph_with_attention):
        """Test visualization contains source node IDs.

        Node C has incoming edges from A and B, so the visualization
        should clearly identify both source nodes with their attention weights.
        """
        viz = graph_with_attention.visualize_attention("C")

        # Get actual attention weights to verify visualization accuracy
        node_c = graph_with_attention.get_node("C")
        assert node_c is not None, "Node C should exist"

        # Visualization MUST contain both source node identifiers
        # These are the actual node IDs that C attends to
        assert "A" in viz, (
            f"Visualization should contain source 'A'. Got: {viz}"
        )
        assert "B" in viz, (
            f"Visualization should contain source 'B'. Got: {viz}"
        )

        # Additionally verify the visualization includes attention weight info
        # It should contain some numeric value (the weight)
        import re
        has_numeric = bool(re.search(r'\d+\.?\d*', viz))
        assert has_numeric, (
            f"Visualization should include attention weights (numbers). Got: {viz}"
        )

    def test_visualize_attention_nonexistent_node(self, graph_with_attention):
        """Test visualize_attention with nonexistent node."""
        viz = graph_with_attention.visualize_attention("NONEXISTENT")

        assert isinstance(viz, str)
        assert "not found" in viz.lower()

    def test_visualize_attention_no_sources(self, graph_with_attention):
        """Test visualize_attention for node with no sources."""
        viz = graph_with_attention.visualize_attention("A")

        assert isinstance(viz, str)
        # Should indicate no attention weights


# =============================================================================
# Parameter Tests
# =============================================================================


class TestParameter:
    """Tests for Parameter class."""

    def test_create_parameter(self):
        """Test basic parameter creation."""
        data = np.array([1.0, 2.0, 3.0])
        param = Parameter(data=data, name="test")

        assert param.name == "test"
        assert param.requires_grad is True
        assert param.grad is None
        assert_array_almost_equal(param.data, data)

    def test_parameter_shape(self):
        """Test parameter shape property."""
        param = Parameter(data=np.zeros((3, 4)))
        assert param.shape == (3, 4)

    def test_add_grad(self):
        """Test gradient accumulation."""
        param = Parameter(data=np.array([1.0, 2.0]))

        param.add_grad(np.array([0.1, 0.2]))
        assert_array_almost_equal(param.grad, np.array([0.1, 0.2]))

        param.add_grad(np.array([0.3, 0.4]))
        assert_array_almost_equal(param.grad, np.array([0.4, 0.6]))

    def test_zero_grad(self):
        """Test gradient reset."""
        param = Parameter(data=np.array([1.0, 2.0]))
        param.add_grad(np.array([0.1, 0.2]))

        param.zero_grad()
        assert param.grad is None


# =============================================================================
# AttentionNode Tests
# =============================================================================


class TestAttentionNode:
    """Tests for AttentionNode class."""

    def test_create_attention_node(self):
        """Test creating attention node."""
        embedding = Parameter(data=np.array([1.0, 2.0, 3.0]))
        node = AttentionNode(id="N1", embedding=embedding)

        assert node.id == "N1"
        assert node.embedding is not None
        assert len(node.embedding.data) == 3

    def test_node_hash(self):
        """Test that attention nodes hash by ID."""
        node1 = AttentionNode(id="N1")
        node2 = AttentionNode(id="N1")

        assert hash(node1) == hash(node2)

    def test_attention_weights_storage(self):
        """Test that node can store attention weights."""
        node = AttentionNode(id="N1")
        node.attention_weights = {"src1": 0.7, "src2": 0.3}

        assert "src1" in node.attention_weights
        assert node.attention_weights["src1"] == 0.7


# =============================================================================
# Training Mode Tests
# =============================================================================


class TestTrainingMode:
    """Tests for train/eval mode switching."""

    def test_default_training_mode(self):
        """Test that graph starts in training mode."""
        graph = AttentionGraph(embedding_dim=8)
        assert graph._training is True

    def test_train_mode_switch(self):
        """Test switching to train mode."""
        graph = AttentionGraph(embedding_dim=8)
        graph.eval()
        assert graph._training is False

        graph.train()
        assert graph._training is True

    def test_eval_mode_switch(self):
        """Test switching to eval mode."""
        graph = AttentionGraph(embedding_dim=8)
        graph.eval()
        assert graph._training is False

    def test_mode_propagates_to_layers(self):
        """Test that mode change propagates to attention layers."""
        graph = AttentionGraph(embedding_dim=8)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        # Initialize layers
        graph.forward(num_layers=1)

        # Switch to eval
        graph.eval()
        assert graph._attention_layers[0]._training is False

        # Switch back to train
        graph.train()
        assert graph._attention_layers[0]._training is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestAttentionGraphIntegration:
    """Integration tests for AttentionGraph."""

    def test_simple_training_loop(self):
        """Test a simple training loop reduces loss."""
        np.random.seed(42)

        graph = AttentionGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        graph.add_node("B", embedding=np.array([0.0, 1.0, 0.0, 0.0]))
        graph.add_edge("A", "B")

        target = np.array([1.0, 1.0, 1.0, 1.0])

        # Initial loss
        initial_output = graph.forward(num_layers=1)["B"]
        initial_loss = np.mean((initial_output - target) ** 2)

        # Train for several steps
        lr = 0.01
        for _ in range(20):
            outputs = graph.forward(num_layers=1)
            output = outputs["B"]

            # Compute loss and gradient
            loss = np.mean((output - target) ** 2)
            grad = 2 * (output - target) / len(output)

            # Backward
            graph.backward({"B": grad}, num_layers=1)

            # Update parameters (simple SGD)
            for param in graph.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad

            # Zero gradients
            graph.zero_grad()

        # Final loss
        final_output = graph.forward(num_layers=1)["B"]
        final_loss = np.mean((final_output - target) ** 2)

        # Loss should decrease
        assert final_loss < initial_loss

    def test_multi_layer_information_flow(self):
        """Test that information flows through multiple layers."""
        graph = AttentionGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        graph.add_node("B", embedding=np.array([0.0, 0.0, 0.0, 0.0]))
        graph.add_node("C", embedding=np.array([0.0, 0.0, 0.0, 0.0]))
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # With 1 layer, C doesn't see A
        outputs_1 = graph.forward(num_layers=1)

        # With 2 layers, C should receive information from A via B
        outputs_2 = graph.forward(num_layers=2)

        # Outputs should be different
        # (This is a weak test, but verifies layers do something)
        assert not np.allclose(outputs_1["C"], outputs_2["C"])

    def test_causal_sequence_processing(self):
        """Test processing a causal sequence."""
        graph = create_causal_attention_graph(seq_len=4, embedding_dim=8, seed=42)

        # Set distinct embeddings
        for i, node in enumerate(graph.nodes):
            embedding = np.zeros(8)
            embedding[i % 8] = 1.0
            node.embedding.data = embedding

        # Forward pass
        outputs = graph.forward(num_layers=1)

        # Each position should produce an output
        assert len(outputs) == 4

        # Later positions should have attended to earlier ones
        node_3 = graph.get_node("pos_3")
        assert len(node_3.attention_weights) == 3  # Attends to pos_0, pos_1, pos_2

    def test_gradient_descent_convergence(self):
        """Test that gradient descent can converge to a target."""
        np.random.seed(42)

        graph = AttentionGraph(embedding_dim=4, seed=42)
        # Use small embeddings to avoid numerical issues
        graph.add_node("input", embedding=np.array([0.1, 0.2, 0.3, 0.4]))
        graph.add_node("output", embedding=np.array([0.1, 0.1, 0.1, 0.1]))
        graph.add_edge("input", "output")

        target = np.array([0.5, 0.5, 0.5, 0.5])
        lr = 0.001  # Small learning rate for stability

        losses = []
        for _ in range(30):
            output = graph.forward(num_layers=1)["output"]
            loss = np.mean((output - target) ** 2)
            losses.append(loss)

            # Check for numerical issues
            if not np.isfinite(loss):
                break

            grad = 2 * (output - target) / len(output)
            graph.backward({"output": grad}, num_layers=1)

            # Gradient clipping for stability
            for param in graph.parameters():
                if param.grad is not None:
                    param.grad = np.clip(param.grad, -1.0, 1.0)
                    param.data -= lr * param.grad

            graph.zero_grad()

        # Loss should decrease (even slightly)
        assert losses[-1] < losses[0]

    def test_protocol_implementation_complete(self):
        """Test that all protocol methods are implemented."""
        graph = AttentionGraph(embedding_dim=8)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        # Test all protocol methods
        assert hasattr(graph, "embedding_dim")
        assert callable(getattr(graph, "parameters"))
        assert callable(getattr(graph, "forward"))
        assert callable(getattr(graph, "backward"))
        assert callable(getattr(graph, "zero_grad"))
        assert callable(getattr(graph, "save_state"))
        assert callable(getattr(graph, "load_state"))

        # Test they all work
        params = graph.parameters()
        outputs = graph.forward(num_layers=1)
        graph.backward({"B": np.ones(8)}, num_layers=1)
        graph.zero_grad()
        state = graph.save_state()
        graph.load_state(state)

        # Should complete without errors
        assert True
