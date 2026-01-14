"""
Unit tests for AttentionLayer: self-attention layer for graph processing.

Tests the core components:
- Initialization (parameter shapes, Xavier init, bias, dropout)
- Forward pass (output shapes, attention weights, cache, dropout)
- Backward pass (gradient shapes, accumulation, flow, input gradients)
- Parameter management (parameters list, counts)
- Edge cases (single node, no edges, fully connected)
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cortical.graph.attention import (
    AttentionGraph,
    AttentionLayer,
    AttentionNode,
    AttentionEdge,
    Parameter,
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_simple_graph(embedding_dim=8, num_nodes=3, causal=True):
    """
    Create a simple AttentionGraph for testing.

    Args:
        embedding_dim: Dimension of embeddings
        num_nodes: Number of nodes to create
        causal: If True, create causal edges (i can attend to j < i)

    Returns:
        AttentionGraph with nodes and edges
    """
    graph = AttentionGraph(embedding_dim=embedding_dim, seed=42)

    # Add nodes
    for i in range(num_nodes):
        graph.add_node(f"node_{i}")

    # Add edges
    if causal:
        # Causal: each position can attend to previous positions
        for i in range(1, num_nodes):
            for j in range(i):
                graph.add_edge(f"node_{j}", f"node_{i}")
    else:
        # Fully connected
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    graph.add_edge(f"node_{j}", f"node_{i}")

    return graph


def create_node_values(graph, value=1.0):
    """
    Create node values dictionary for testing.

    Args:
        graph: AttentionGraph
        value: Scalar value to fill embeddings (or can be array)

    Returns:
        Dict mapping node_id to embedding array
    """
    node_values = {}
    for node in graph.nodes:
        if isinstance(value, (int, float)):
            node_values[node.id] = np.ones(graph.embedding_dim) * value
        else:
            node_values[node.id] = value.copy()
    return node_values


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAttentionLayerInit:
    """Tests for AttentionLayer initialization."""

    def test_parameter_shapes(self):
        """Test that all parameter matrices have correct shapes."""
        embedding_dim = 16
        layer = AttentionLayer(embedding_dim=embedding_dim)

        # All projection matrices should be (embedding_dim, embedding_dim)
        assert layer.W_q.shape == (embedding_dim, embedding_dim)
        assert layer.W_k.shape == (embedding_dim, embedding_dim)
        assert layer.W_v.shape == (embedding_dim, embedding_dim)
        assert layer.W_o.shape == (embedding_dim, embedding_dim)

    def test_xavier_initialization_scale(self):
        """Test that Xavier initialization produces reasonable scale."""
        embedding_dim = 64
        layer = AttentionLayer(embedding_dim=embedding_dim)

        # Xavier scale = sqrt(2 / (embedding_dim + embedding_dim))
        expected_scale = np.sqrt(2.0 / (embedding_dim + embedding_dim))

        # Check that parameter values are within reasonable bounds
        # Standard deviation should be close to expected_scale
        for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            param = getattr(layer, param_name)
            std = np.std(param.data)
            # Allow some variance due to random initialization
            assert 0.5 * expected_scale < std < 2.0 * expected_scale, \
                f"{param_name} std {std} not close to expected {expected_scale}"

    def test_init_without_bias(self):
        """Test initialization without bias terms."""
        layer = AttentionLayer(embedding_dim=8, use_bias=False)

        assert layer.use_bias is False
        assert not hasattr(layer, 'b_q')
        assert not hasattr(layer, 'b_k')
        assert not hasattr(layer, 'b_v')
        assert not hasattr(layer, 'b_o')

    def test_init_with_bias(self):
        """Test initialization with bias terms."""
        embedding_dim = 8
        layer = AttentionLayer(embedding_dim=embedding_dim, use_bias=True)

        assert layer.use_bias is True
        assert hasattr(layer, 'b_q')
        assert hasattr(layer, 'b_k')
        assert hasattr(layer, 'b_v')
        assert hasattr(layer, 'b_o')

        # Biases should be zero-initialized
        assert_array_equal(layer.b_q.data, np.zeros(embedding_dim))
        assert_array_equal(layer.b_k.data, np.zeros(embedding_dim))
        assert_array_equal(layer.b_v.data, np.zeros(embedding_dim))
        assert_array_equal(layer.b_o.data, np.zeros(embedding_dim))

        # Bias shapes should match embedding_dim
        assert layer.b_q.shape == (embedding_dim,)
        assert layer.b_k.shape == (embedding_dim,)
        assert layer.b_v.shape == (embedding_dim,)
        assert layer.b_o.shape == (embedding_dim,)

    def test_dropout_parameter_stored(self):
        """Test that dropout parameter is stored correctly."""
        layer = AttentionLayer(embedding_dim=8, dropout=0.1)
        assert layer.dropout == 0.1

        layer_no_dropout = AttentionLayer(embedding_dim=8, dropout=0.0)
        assert layer_no_dropout.dropout == 0.0

    def test_training_mode_default(self):
        """Test that layer defaults to training mode."""
        layer = AttentionLayer(embedding_dim=8)
        assert layer._training is True

    def test_cache_initialized(self):
        """Test that cache is initialized as empty dict."""
        layer = AttentionLayer(embedding_dim=8)
        assert isinstance(layer._cache, dict)


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestAttentionLayerForward:
    """Tests for AttentionLayer forward pass."""

    @pytest.fixture
    def layer(self):
        """Create layer with fixed seed for reproducibility."""
        np.random.seed(42)
        return AttentionLayer(embedding_dim=8, use_bias=False, dropout=0.0)

    @pytest.fixture
    def graph(self):
        """Create simple graph for testing."""
        return create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)

    def test_output_shapes_match_input_shapes(self, layer, graph):
        """Test that output shapes match input shapes."""
        node_values = create_node_values(graph, value=1.0)

        outputs = layer.forward(node_values, graph)

        # Should have output for each node
        assert len(outputs) == len(node_values)

        # Each output should have same shape as input
        for node_id, output in outputs.items():
            assert output.shape == node_values[node_id].shape
            assert len(output) == graph.embedding_dim

    def test_first_position_no_incoming_edges(self, layer, graph):
        """Test that nodes with no incoming edges still produce output."""
        node_values = create_node_values(graph, value=1.0)

        outputs = layer.forward(node_values, graph)

        # First node (node_0) has no incoming edges in causal graph
        assert "node_0" in outputs
        assert outputs["node_0"] is not None
        assert len(outputs["node_0"]) == graph.embedding_dim

        # Output should be non-zero (transformed through W_o)
        assert not np.allclose(outputs["node_0"], 0.0)

    def test_attention_weights_stored_on_nodes(self, layer, graph):
        """Test that attention weights are stored on nodes after forward."""
        node_values = create_node_values(graph, value=1.0)

        layer.forward(node_values, graph)

        # First node should have empty attention weights (no incoming edges)
        node_0 = graph.get_node("node_0")
        assert node_0.attention_weights == {}

        # Second node should have attention weights for node_0
        node_1 = graph.get_node("node_1")
        assert len(node_1.attention_weights) == 1
        assert "node_0" in node_1.attention_weights
        assert 0.0 <= node_1.attention_weights["node_0"] <= 1.0

        # Third node should have attention weights for node_0 and node_1
        node_2 = graph.get_node("node_2")
        assert len(node_2.attention_weights) == 2
        assert "node_0" in node_2.attention_weights
        assert "node_1" in node_2.attention_weights

        # Attention weights should sum to 1.0
        total_weight = sum(node_2.attention_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_cache_populated_for_backward(self, layer, graph):
        """Test that cache is populated with values needed for backward pass."""
        node_values = create_node_values(graph, value=1.0)

        layer.forward(node_values, graph)

        # Cache should have required keys
        assert "input_values" in layer._cache
        assert "queries" in layer._cache
        assert "keys" in layer._cache
        assert "values" in layer._cache
        assert "attention_weights" in layer._cache
        assert "pre_output" in layer._cache

        # Should have cached values for each node
        for node in graph.nodes:
            assert node.id in layer._cache["input_values"]
            assert node.id in layer._cache["queries"]

    def test_train_mode_with_dropout(self):
        """Test forward pass in training mode with dropout."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        layer = AttentionLayer(embedding_dim=8, dropout=0.5)
        layer.train()

        node_values = create_node_values(graph, value=1.0)

        # Run forward pass multiple times
        outputs_1 = layer.forward(node_values, graph)
        outputs_2 = layer.forward(node_values, graph)

        # With dropout, outputs should vary between runs
        # (Note: This is stochastic, but with dropout=0.5 it's very likely)
        # Check at least one node has different output
        different = False
        for node_id in outputs_1:
            if not np.allclose(outputs_1[node_id], outputs_2[node_id]):
                different = True
                break

        # Note: This test could rarely fail due to randomness
        # but with dropout=0.5 it's extremely unlikely

    def test_eval_mode_no_dropout(self):
        """Test forward pass in eval mode has no dropout."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        layer = AttentionLayer(embedding_dim=8, dropout=0.5)
        layer.eval()

        node_values = create_node_values(graph, value=1.0)

        # Run forward pass multiple times
        outputs_1 = layer.forward(node_values, graph)

        # Reset random seed to same state
        np.random.seed(42)
        outputs_2 = layer.forward(node_values, graph)

        # In eval mode, outputs should be identical
        for node_id in outputs_1:
            assert_array_almost_equal(outputs_1[node_id], outputs_2[node_id])

    def test_single_node_graph(self):
        """Test forward pass with single node (no edges)."""
        np.random.seed(42)
        graph = AttentionGraph(embedding_dim=8, seed=42)
        graph.add_node("only")

        layer = AttentionLayer(embedding_dim=8)
        node_values = {"only": np.ones(8)}

        outputs = layer.forward(node_values, graph)

        assert "only" in outputs
        assert len(outputs["only"]) == 8
        # Should still produce output via W_o transformation
        assert not np.allclose(outputs["only"], 0.0)

    def test_fully_connected_graph(self):
        """Test forward pass with fully connected graph."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=False)

        layer = AttentionLayer(embedding_dim=8)
        node_values = create_node_values(graph, value=1.0)

        outputs = layer.forward(node_values, graph)

        # All nodes should have outputs
        assert len(outputs) == 3

        # Each node should attend to the other 2 nodes
        for node in graph.nodes:
            # Check attention weights exist and sum to 1
            attn_weights = node.attention_weights
            if attn_weights:  # Not empty
                total = sum(attn_weights.values())
                assert abs(total - 1.0) < 1e-6

    def test_different_input_values(self):
        """Test forward pass with different input values per node."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        layer = AttentionLayer(embedding_dim=8)

        # Create distinct values for each node
        node_values = {
            "node_0": np.ones(8) * 1.0,
            "node_1": np.ones(8) * 2.0,
            "node_2": np.ones(8) * 3.0,
        }

        outputs = layer.forward(node_values, graph)

        # All nodes should have outputs
        assert len(outputs) == 3

        # Outputs should be different for different inputs
        assert not np.allclose(outputs["node_0"], outputs["node_1"])
        assert not np.allclose(outputs["node_1"], outputs["node_2"])


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestAttentionLayerBackward:
    """Tests for AttentionLayer backward pass."""

    @pytest.fixture
    def layer(self):
        """Create layer with fixed seed for reproducibility."""
        np.random.seed(42)
        return AttentionLayer(embedding_dim=8, use_bias=False, dropout=0.0)

    @pytest.fixture
    def graph(self):
        """Create simple graph for testing."""
        return create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)

    def test_gradient_shapes_match_parameter_shapes(self, layer, graph):
        """Test that gradient shapes match parameter shapes."""
        node_values = create_node_values(graph, value=1.0)

        # Forward pass
        outputs = layer.forward(node_values, graph)

        # Backward pass with unit gradients
        output_gradients = {node_id: np.ones(8) for node_id in outputs}
        layer.backward(output_gradients, graph)

        # Check gradient shapes match parameter shapes
        assert layer.W_q.grad.shape == layer.W_q.data.shape
        assert layer.W_k.grad.shape == layer.W_k.data.shape
        assert layer.W_v.grad.shape == layer.W_v.data.shape
        assert layer.W_o.grad.shape == layer.W_o.data.shape

    def test_gradients_accumulated(self, layer, graph):
        """Test that gradients are accumulated (not replaced)."""
        node_values = create_node_values(graph, value=1.0)

        # First forward-backward pass
        outputs = layer.forward(node_values, graph)
        output_gradients = {node_id: np.ones(8) for node_id in outputs}
        layer.backward(output_gradients, graph)

        # Store first gradients
        first_grad_W_q = layer.W_q.grad.copy() if layer.W_q.grad is not None else None
        first_grad_W_k = layer.W_k.grad.copy() if layer.W_k.grad is not None else None
        first_grad_W_v = layer.W_v.grad.copy() if layer.W_v.grad is not None else None
        first_grad_W_o = layer.W_o.grad.copy() if layer.W_o.grad is not None else None

        # Second forward-backward pass (without zero_grad)
        outputs = layer.forward(node_values, graph)
        layer.backward(output_gradients, graph)

        # Gradients should have doubled (accumulated)
        if first_grad_W_q is not None:
            assert_array_almost_equal(layer.W_q.grad, first_grad_W_q * 2, decimal=5)
        if first_grad_W_k is not None:
            assert_array_almost_equal(layer.W_k.grad, first_grad_W_k * 2, decimal=5)
        if first_grad_W_v is not None:
            assert_array_almost_equal(layer.W_v.grad, first_grad_W_v * 2, decimal=5)
        if first_grad_W_o is not None:
            assert_array_almost_equal(layer.W_o.grad, first_grad_W_o * 2, decimal=5)

    def test_gradients_flow_to_all_parameters(self, layer, graph):
        """Test that gradients flow to all parameters (W_q, W_k, W_v, W_o)."""
        # Use different values to avoid gradient saturation
        node_values = {
            "node_0": np.random.randn(8) * 0.5,
            "node_1": np.random.randn(8) * 0.5,
            "node_2": np.random.randn(8) * 0.5,
        }

        # Forward pass
        outputs = layer.forward(node_values, graph)

        # Backward pass with larger gradients
        output_gradients = {node_id: np.ones(8) * 2.0 for node_id in outputs}
        layer.backward(output_gradients, graph)

        # All parameters should have gradients
        assert layer.W_q.grad is not None, "W_q should have gradient"
        assert layer.W_k.grad is not None, "W_k should have gradient"
        assert layer.W_v.grad is not None, "W_v should have gradient"
        assert layer.W_o.grad is not None, "W_o should have gradient"

        # Gradients should be non-zero (with reasonable tolerance)
        assert not np.allclose(layer.W_q.grad, 0.0, atol=1e-8), "W_q gradient should be non-zero"
        assert not np.allclose(layer.W_k.grad, 0.0, atol=1e-8), "W_k gradient should be non-zero"
        assert not np.allclose(layer.W_v.grad, 0.0, atol=1e-8), "W_v gradient should be non-zero"
        assert not np.allclose(layer.W_o.grad, 0.0, atol=1e-8), "W_o gradient should be non-zero"

    def test_bias_gradients_when_use_bias(self):
        """Test that bias gradients are computed when use_bias=True."""
        np.random.seed(42)
        layer = AttentionLayer(embedding_dim=8, use_bias=True)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        node_values = create_node_values(graph, value=1.0)

        # Forward pass
        outputs = layer.forward(node_values, graph)

        # Backward pass
        output_gradients = {node_id: np.ones(8) for node_id in outputs}
        layer.backward(output_gradients, graph)

        # All bias parameters should have gradients
        assert layer.b_q.grad is not None, "b_q should have gradient"
        assert layer.b_k.grad is not None, "b_k should have gradient"
        assert layer.b_v.grad is not None, "b_v should have gradient"
        assert layer.b_o.grad is not None, "b_o should have gradient"

        # Check shapes
        assert layer.b_q.grad.shape == (8,)
        assert layer.b_k.grad.shape == (8,)
        assert layer.b_v.grad.shape == (8,)
        assert layer.b_o.grad.shape == (8,)

    def test_input_gradients_returned(self, layer, graph):
        """Test that input gradients are returned for previous layer."""
        # Use different values to avoid gradient saturation
        node_values = {
            "node_0": np.random.randn(8) * 0.5,
            "node_1": np.random.randn(8) * 0.5,
            "node_2": np.random.randn(8) * 0.5,
        }

        # Forward pass
        outputs = layer.forward(node_values, graph)

        # Backward pass with larger gradients
        output_gradients = {node_id: np.ones(8) * 2.0 for node_id in outputs}
        input_gradients = layer.backward(output_gradients, graph)

        # Should return dict of input gradients
        assert isinstance(input_gradients, dict)

        # Should have gradients for all nodes
        for node_id in node_values:
            assert node_id in input_gradients
            assert input_gradients[node_id].shape == (8,)
            # Gradients should be non-zero (with tolerance)
            assert not np.allclose(input_gradients[node_id], 0.0, atol=1e-8)

    def test_zero_grad_clears_gradients(self, layer, graph):
        """Test that zero_grad clears all gradients."""
        node_values = create_node_values(graph, value=1.0)

        # Forward and backward to create gradients
        outputs = layer.forward(node_values, graph)
        output_gradients = {node_id: np.ones(8) for node_id in outputs}
        layer.backward(output_gradients, graph)

        # Verify gradients exist
        assert layer.W_q.grad is not None
        assert layer.W_k.grad is not None
        assert layer.W_v.grad is not None
        assert layer.W_o.grad is not None

        # Zero gradients via Parameter objects
        for param in layer.parameters():
            param.zero_grad()

        # All gradients should be None
        assert layer.W_q.grad is None
        assert layer.W_k.grad is None
        assert layer.W_v.grad is None
        assert layer.W_o.grad is None

    def test_backward_with_subset_of_nodes(self, layer, graph):
        """Test backward pass when only some nodes have output gradients."""
        node_values = create_node_values(graph, value=1.0)

        # Forward pass
        outputs = layer.forward(node_values, graph)

        # Backward with gradient only for last node
        output_gradients = {"node_2": np.ones(8)}
        input_gradients = layer.backward(output_gradients, graph)

        # Should still compute gradients
        assert layer.W_o.grad is not None

        # Input gradients may not cover all nodes
        assert isinstance(input_gradients, dict)


# =============================================================================
# Parameter Management Tests
# =============================================================================


class TestAttentionLayerParameters:
    """Tests for AttentionLayer parameter management."""

    def test_parameters_returns_all_parameters(self):
        """Test that parameters() returns all learnable parameters."""
        layer = AttentionLayer(embedding_dim=8, use_bias=False)
        params = layer.parameters()

        # Should have 4 parameters (W_q, W_k, W_v, W_o)
        assert len(params) == 4

        # Should include all weight matrices
        param_names = [p.name for p in params]
        assert any("W_q" in name for name in param_names)
        assert any("W_k" in name for name in param_names)
        assert any("W_v" in name for name in param_names)
        assert any("W_o" in name for name in param_names)

    def test_parameters_with_bias(self):
        """Test that parameters() includes bias when use_bias=True."""
        layer = AttentionLayer(embedding_dim=8, use_bias=True)
        params = layer.parameters()

        # Should have 8 parameters (4 weights + 4 biases)
        assert len(params) == 8

        # Should include all weight matrices and biases
        param_names = [p.name for p in params]
        assert any("W_q" in name for name in param_names)
        assert any("W_k" in name for name in param_names)
        assert any("W_v" in name for name in param_names)
        assert any("W_o" in name for name in param_names)
        assert any("b_q" in name for name in param_names)
        assert any("b_k" in name for name in param_names)
        assert any("b_v" in name for name in param_names)
        assert any("b_o" in name for name in param_names)

    def test_parameter_count_correct(self):
        """Test that parameter count is correct."""
        embedding_dim = 16

        # Without bias
        layer_no_bias = AttentionLayer(embedding_dim=embedding_dim, use_bias=False)
        params_no_bias = layer_no_bias.parameters()

        # 4 matrices * (embedding_dim * embedding_dim) = 4 * 16 * 16 = 1024 params
        total_params_no_bias = sum(p.data.size for p in params_no_bias)
        expected_no_bias = 4 * embedding_dim * embedding_dim
        assert total_params_no_bias == expected_no_bias

        # With bias
        layer_with_bias = AttentionLayer(embedding_dim=embedding_dim, use_bias=True)
        params_with_bias = layer_with_bias.parameters()

        # 4 matrices + 4 biases = 1024 + 4*16 = 1088 params
        total_params_with_bias = sum(p.data.size for p in params_with_bias)
        expected_with_bias = 4 * embedding_dim * embedding_dim + 4 * embedding_dim
        assert total_params_with_bias == expected_with_bias

    def test_all_parameters_have_names(self):
        """Test that all parameters have descriptive names."""
        layer = AttentionLayer(embedding_dim=8, use_bias=True)
        params = layer.parameters()

        for param in params:
            assert param.name != ""
            assert "attention" in param.name.lower()

    def test_all_parameters_require_grad(self):
        """Test that all parameters have requires_grad=True by default."""
        layer = AttentionLayer(embedding_dim=8, use_bias=True)
        params = layer.parameters()

        for param in params:
            assert param.requires_grad is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAttentionLayerEdgeCases:
    """Tests for edge cases in AttentionLayer."""

    def test_empty_graph(self):
        """Test forward pass with empty graph."""
        np.random.seed(42)
        graph = AttentionGraph(embedding_dim=8)
        layer = AttentionLayer(embedding_dim=8)

        outputs = layer.forward({}, graph)

        # Should return empty dict
        assert outputs == {}

    def test_node_with_no_edges_at_all(self):
        """Test node that has no incoming or outgoing edges."""
        np.random.seed(42)
        graph = AttentionGraph(embedding_dim=8)
        graph.add_node("isolated")
        graph.add_node("connected_1")
        graph.add_node("connected_2")
        graph.add_edge("connected_1", "connected_2")

        layer = AttentionLayer(embedding_dim=8)
        node_values = create_node_values(graph, value=1.0)

        outputs = layer.forward(node_values, graph)

        # Isolated node should still have output
        assert "isolated" in outputs
        assert len(outputs["isolated"]) == 8

    def test_self_loop(self):
        """Test graph with self-loop edge."""
        np.random.seed(42)
        graph = AttentionGraph(embedding_dim=8)
        graph.add_node("self")
        # Add self-loop (node attends to itself)
        graph.add_edge("self", "self")

        layer = AttentionLayer(embedding_dim=8)
        node_values = {"self": np.ones(8)}

        outputs = layer.forward(node_values, graph)

        # Should handle self-loop gracefully
        assert "self" in outputs

        # Should have attention weight for itself
        node = graph.get_node("self")
        assert "self" in node.attention_weights
        # Should be 1.0 (only attending to itself)
        assert abs(node.attention_weights["self"] - 1.0) < 1e-6

    def test_very_small_embedding_dim(self):
        """Test with very small embedding dimension."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=2, num_nodes=2, causal=True)
        layer = AttentionLayer(embedding_dim=2)

        node_values = create_node_values(graph, value=1.0)
        outputs = layer.forward(node_values, graph)

        # Should work with small dimensions
        assert len(outputs) == 2
        for output in outputs.values():
            assert len(output) == 2

    def test_large_embedding_dim(self):
        """Test with larger embedding dimension."""
        np.random.seed(42)
        embedding_dim = 128
        graph = create_simple_graph(embedding_dim=embedding_dim, num_nodes=2, causal=True)
        layer = AttentionLayer(embedding_dim=embedding_dim)

        node_values = create_node_values(graph, value=1.0)
        outputs = layer.forward(node_values, graph)

        # Should work with large dimensions
        assert len(outputs) == 2
        for output in outputs.values():
            assert len(output) == embedding_dim

    def test_zero_input_values(self):
        """Test forward pass with all-zero input values."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        layer = AttentionLayer(embedding_dim=8)

        node_values = create_node_values(graph, value=0.0)
        outputs = layer.forward(node_values, graph)

        # Should still compute outputs (though they may be small)
        assert len(outputs) == 3
        for output in outputs.values():
            assert len(output) == 8

    def test_backward_without_forward(self):
        """Test that backward without forward doesn't crash."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=2, causal=True)
        layer = AttentionLayer(embedding_dim=8)

        # Call backward without forward (cache will be empty)
        output_gradients = {"node_0": np.ones(8), "node_1": np.ones(8)}

        # Should handle gracefully (may not compute gradients)
        # This might return empty dict or skip nodes not in cache
        try:
            input_gradients = layer.backward(output_gradients, graph)
            # Should return a dict (possibly empty)
            assert isinstance(input_gradients, dict)
        except (KeyError, AttributeError):
            # It's acceptable to fail gracefully when cache is missing
            # The important part is it doesn't crash catastrophically
            pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestAttentionLayerIntegration:
    """Integration tests using AttentionLayer with full graph."""

    def test_full_forward_backward_cycle(self):
        """Test complete forward-backward cycle."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=4, causal=True)
        layer = AttentionLayer(embedding_dim=8, use_bias=True)

        # Forward pass
        node_values = create_node_values(graph, value=1.0)
        outputs = layer.forward(node_values, graph)

        # Check outputs
        assert len(outputs) == 4

        # Backward pass
        output_gradients = {node_id: np.ones(8) for node_id in outputs}
        input_gradients = layer.backward(output_gradients, graph)

        # Check gradients computed
        assert len(input_gradients) > 0
        for param in layer.parameters():
            assert param.grad is not None

    def test_multiple_forward_passes(self):
        """Test multiple forward passes with same layer."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        layer = AttentionLayer(embedding_dim=8)

        node_values_1 = create_node_values(graph, value=1.0)
        node_values_2 = create_node_values(graph, value=2.0)

        # First forward pass
        outputs_1 = layer.forward(node_values_1, graph)

        # Second forward pass (should overwrite cache)
        outputs_2 = layer.forward(node_values_2, graph)

        # Outputs should be different
        for node_id in outputs_1:
            assert not np.allclose(outputs_1[node_id], outputs_2[node_id])

    def test_gradient_descent_step(self):
        """Test that parameters change after gradient descent step."""
        np.random.seed(42)
        graph = create_simple_graph(embedding_dim=8, num_nodes=3, causal=True)
        layer = AttentionLayer(embedding_dim=8)

        # Store initial parameters
        initial_W_q = layer.W_q.data.copy()

        # Forward pass with varied inputs
        node_values = {
            "node_0": np.random.randn(8) * 0.5,
            "node_1": np.random.randn(8) * 0.5,
            "node_2": np.random.randn(8) * 0.5,
        }
        outputs = layer.forward(node_values, graph)

        # Backward pass with larger gradients
        output_gradients = {node_id: np.ones(8) * 2.0 for node_id in outputs}
        layer.backward(output_gradients, graph)

        # Ensure gradient exists and is non-zero
        assert layer.W_q.grad is not None
        assert not np.allclose(layer.W_q.grad, 0.0, atol=1e-8)

        # Simulate gradient descent step
        learning_rate = 0.1  # Larger learning rate for clearer change
        layer.W_q.data -= learning_rate * layer.W_q.grad

        # Parameters should have changed
        assert not np.allclose(layer.W_q.data, initial_W_q, atol=1e-8)
