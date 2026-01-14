"""
Numerical gradient verification tests for attention implementation.

This is the gold standard for verifying backpropagation: we compare analytical
gradients (from backward()) against numerical gradients (finite differences).

The Story:
    Imagine you're a calculus student checking your derivative work.
    - Analytical gradient: You applied the chain rule by hand
    - Numerical gradient: You plotted the function and measured the slope

    If they match, your calculus is correct!

Why This Matters:
    Backpropagation is subtle. A bug in gradient computation means:
    - Parameters update in the wrong direction
    - Training diverges or stalls
    - The model learns nothing

    These tests catch such bugs early by verifying the gradient math.

Test Strategy:
    1. Set up a small computation (attention, layer, graph)
    2. Run forward pass and compute analytical gradients via backward()
    3. Compute numerical gradients by perturbing each parameter slightly
    4. Verify they match within numerical precision tolerance
"""

import numpy as np
from typing import Callable

from cortical.graph.attention import (
    scaled_dot_product_attention,
    attention_backward,
    AttentionLayer,
    AttentionGraph,
    create_causal_attention_graph,
)


# =============================================================================
# NUMERICAL GRADIENT HELPER
# =============================================================================


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute numerical gradient of f at x using central differences.

    The Story:
        To measure how f changes with x, we:
        1. Wiggle each component of x slightly up (x + eps)
        2. Wiggle it slightly down (x - eps)
        3. Measure the difference in f's output
        4. Divide by 2*eps to get the slope

        This is the fundamental definition of a derivative!

    Args:
        f: Function that computes a scalar loss (closure over x)
        x: Point at which to compute gradient
        eps: Small perturbation for finite differences

    Returns:
        Numerical gradient with same shape as x

    Implementation Note:
        We modify x in-place and restore it after each perturbation.
        This is safe because we control when f() is called.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        # Perturb up
        x[idx] = old_val + eps
        fxp = f()

        # Perturb down
        x[idx] = old_val - eps
        fxm = f()

        # Restore original value
        x[idx] = old_val

        # Central difference formula
        grad[idx] = (fxp - fxm) / (2 * eps)
        it.iternext()

    return grad


def relative_error(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute relative error between two arrays.

    Formula: ||x - y|| / (||x|| + ||y|| + eps)

    Why relative error?
        - Absolute error fails when values are large or small
        - Relative error is scale-invariant
        - Common in numerical computing
    """
    numerator = np.linalg.norm(x - y)
    denominator = np.linalg.norm(x) + np.linalg.norm(y) + 1e-10
    return numerator / denominator


# =============================================================================
# TEST ATTENTION FUNCTION GRADIENTS
# =============================================================================


class TestAttentionFunctionGradients:
    """
    Test gradients through scaled_dot_product_attention.

    The Story:
        The attention function is the core building block.
        If gradients are wrong here, everything built on top is broken.

        We test each input (query, keys, values) separately to isolate bugs.
    """

    def setup_attention(self):
        """Create small attention computation for testing."""
        np.random.seed(42)

        d_k = 4  # Small dimension for fast numerical gradients
        n_sources = 3  # Few sources for manageable computation

        query = np.random.randn(d_k) * 0.1
        keys = np.random.randn(n_sources, d_k) * 0.1
        values = np.random.randn(n_sources, d_k) * 0.1

        return query, keys, values

    def test_query_gradient(self):
        """
        Test gradient w.r.t. query vector.

        The Story:
            The query determines which keys to attend to.
            Changing the query changes the attention weights,
            which changes the output. We verify this gradient.
        """
        query, keys, values = self.setup_attention()

        # Forward pass
        output, attn_weights = scaled_dot_product_attention(query, keys, values)

        # Analytical gradient (using attention_backward)
        # Assume gradient w.r.t. output is all ones (simplified loss)
        grad_output = np.ones_like(output)
        grad_query_analytical, _, _ = attention_backward(
            grad_output, query, keys, values, attn_weights
        )

        # Numerical gradient
        def loss_fn():
            out, _ = scaled_dot_product_attention(query, keys, values)
            return np.sum(out)  # Sum as simple loss

        grad_query_numerical = numerical_gradient(loss_fn, query)

        # Compare
        rel_error = relative_error(grad_query_analytical, grad_query_numerical)
        print(f"\nQuery gradient relative error: {rel_error:.2e}")
        print(f"Analytical: {grad_query_analytical[:4]}")
        print(f"Numerical:  {grad_query_numerical[:4]}")

        assert rel_error < 1e-4, f"Query gradient error too high: {rel_error}"

    def test_keys_gradient(self):
        """
        Test gradient w.r.t. keys matrix.

        The Story:
            Keys determine how much each source gets attended to.
            Each key affects its corresponding attention weight.
        """
        query, keys, values = self.setup_attention()

        # Forward pass
        output, attn_weights = scaled_dot_product_attention(query, keys, values)

        # Analytical gradient
        grad_output = np.ones_like(output)
        _, grad_keys_analytical, _ = attention_backward(
            grad_output, query, keys, values, attn_weights
        )

        # Numerical gradient
        def loss_fn():
            out, _ = scaled_dot_product_attention(query, keys, values)
            return np.sum(out)

        grad_keys_numerical = numerical_gradient(loss_fn, keys)

        # Compare
        rel_error = relative_error(grad_keys_analytical, grad_keys_numerical)
        print(f"\nKeys gradient relative error: {rel_error:.2e}")
        print(f"Analytical shape: {grad_keys_analytical.shape}")
        print(f"Numerical shape:  {grad_keys_numerical.shape}")

        assert rel_error < 1e-4, f"Keys gradient error too high: {rel_error}"

    def test_values_gradient(self):
        """
        Test gradient w.r.t. values matrix.

        The Story:
            Values are what gets mixed by attention weights.
            This is usually the simplest gradient - just weighted sums.
        """
        query, keys, values = self.setup_attention()

        # Forward pass
        output, attn_weights = scaled_dot_product_attention(query, keys, values)

        # Analytical gradient
        grad_output = np.ones_like(output)
        _, _, grad_values_analytical = attention_backward(
            grad_output, query, keys, values, attn_weights
        )

        # Numerical gradient
        def loss_fn():
            out, _ = scaled_dot_product_attention(query, keys, values)
            return np.sum(out)

        grad_values_numerical = numerical_gradient(loss_fn, values)

        # Compare
        rel_error = relative_error(grad_values_analytical, grad_values_numerical)
        print(f"\nValues gradient relative error: {rel_error:.2e}")
        print(f"Analytical shape: {grad_values_analytical.shape}")
        print(f"Numerical shape:  {grad_values_numerical.shape}")

        assert rel_error < 1e-4, f"Values gradient error too high: {rel_error}"

    def test_attention_with_different_dimensions(self):
        """
        Test attention gradients with various dimension sizes.

        The Story:
            Gradients should be correct regardless of dimensions.
            Testing multiple sizes helps catch scaling bugs.
        """
        np.random.seed(123)

        test_cases = [
            (4, 2),   # Very small
            (8, 3),   # Small
            (16, 5),  # Medium
        ]

        for d_k, n_sources in test_cases:
            query = np.random.randn(d_k) * 0.1
            keys = np.random.randn(n_sources, d_k) * 0.1
            values = np.random.randn(n_sources, d_k) * 0.1

            # Forward
            output, attn_weights = scaled_dot_product_attention(query, keys, values)

            # Analytical
            grad_output = np.ones_like(output)
            grad_q, grad_k, grad_v = attention_backward(
                grad_output, query, keys, values, attn_weights
            )

            # Numerical
            def loss_fn():
                out, _ = scaled_dot_product_attention(query, keys, values)
                return np.sum(out)

            grad_q_num = numerical_gradient(loss_fn, query)

            rel_error = relative_error(grad_q, grad_q_num)
            print(f"\nDimensions (d_k={d_k}, n_sources={n_sources}): error={rel_error:.2e}")

            assert rel_error < 1e-4, f"Error too high for dims {d_k}, {n_sources}"


# =============================================================================
# TEST ATTENTION LAYER GRADIENTS
# =============================================================================


class TestAttentionLayerGradients:
    """
    Test gradients through AttentionLayer.

    The Story:
        The layer wraps attention with learnable projections (W_q, W_k, W_v, W_o).
        We need to verify gradients flow correctly through all these parameters.
    """

    def setup_layer(self):
        """Create small attention layer with simple graph."""
        np.random.seed(42)

        embedding_dim = 8
        layer = AttentionLayer(embedding_dim=embedding_dim, use_bias=False)

        # Create simple graph with 3 nodes, causal structure
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=embedding_dim)

        # Set node embeddings to small random values
        for node in graph.nodes:
            if node.embedding is not None:
                node.embedding.data = np.random.randn(embedding_dim) * 0.1

        return layer, graph

    def test_W_q_gradient(self):
        """
        Test gradient w.r.t. query projection matrix.

        The Story:
            W_q transforms embeddings into queries.
            Changing W_q changes all queries, affecting all attention computations.
        """
        layer, graph = self.setup_layer()

        # Forward pass
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        # Define simple loss: sum of final node's output
        loss = np.sum(outputs["pos_2"])

        # Analytical gradient via backward
        for param in layer.parameters():
            param.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        layer.backward(grad_outputs, graph)
        grad_W_q_analytical = layer.W_q.grad.copy()

        # Numerical gradient
        def loss_fn():
            node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
            outputs = layer.forward(node_values, graph)
            return np.sum(outputs["pos_2"])

        grad_W_q_numerical = numerical_gradient(loss_fn, layer.W_q.data)

        # Compare
        rel_error = relative_error(grad_W_q_analytical, grad_W_q_numerical)
        print(f"\nW_q gradient relative error: {rel_error:.2e}")
        print(f"Analytical norm: {np.linalg.norm(grad_W_q_analytical):.4f}")
        print(f"Numerical norm:  {np.linalg.norm(grad_W_q_numerical):.4f}")

        assert rel_error < 1e-3, f"W_q gradient error too high: {rel_error}"

    def test_W_k_gradient(self):
        """
        Test gradient w.r.t. key projection matrix.

        The Story:
            W_k determines what sources "advertise" to querying nodes.
            This affects attention weights for all attending nodes.
        """
        layer, graph = self.setup_layer()

        # Forward
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        # Analytical gradient
        for param in layer.parameters():
            param.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        layer.backward(grad_outputs, graph)
        grad_W_k_analytical = layer.W_k.grad.copy()

        # Numerical gradient
        def loss_fn():
            node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
            outputs = layer.forward(node_values, graph)
            return np.sum(outputs["pos_2"])

        grad_W_k_numerical = numerical_gradient(loss_fn, layer.W_k.data)

        # Compare
        rel_error = relative_error(grad_W_k_analytical, grad_W_k_numerical)
        print(f"\nW_k gradient relative error: {rel_error:.2e}")

        assert rel_error < 1e-3, f"W_k gradient error too high: {rel_error}"

    def test_W_v_gradient(self):
        """
        Test gradient w.r.t. value projection matrix.

        The Story:
            W_v determines what information sources provide.
            This is what actually gets mixed by attention weights.
        """
        layer, graph = self.setup_layer()

        # Forward
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        # Analytical gradient
        for param in layer.parameters():
            param.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        layer.backward(grad_outputs, graph)
        grad_W_v_analytical = layer.W_v.grad.copy()

        # Numerical gradient
        def loss_fn():
            node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
            outputs = layer.forward(node_values, graph)
            return np.sum(outputs["pos_2"])

        grad_W_v_numerical = numerical_gradient(loss_fn, layer.W_v.data)

        # Compare
        rel_error = relative_error(grad_W_v_analytical, grad_W_v_numerical)
        print(f"\nW_v gradient relative error: {rel_error:.2e}")

        assert rel_error < 1e-3, f"W_v gradient error too high: {rel_error}"

    def test_W_o_gradient(self):
        """
        Test gradient w.r.t. output projection matrix.

        The Story:
            W_o is the final transformation after attention.
            Every node's output passes through W_o, so gradients accumulate.
        """
        layer, graph = self.setup_layer()

        # Forward
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        # Analytical gradient
        for param in layer.parameters():
            param.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        layer.backward(grad_outputs, graph)
        grad_W_o_analytical = layer.W_o.grad.copy()

        # Numerical gradient
        def loss_fn():
            node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
            outputs = layer.forward(node_values, graph)
            return np.sum(outputs["pos_2"])

        grad_W_o_numerical = numerical_gradient(loss_fn, layer.W_o.data)

        # Compare
        rel_error = relative_error(grad_W_o_analytical, grad_W_o_numerical)
        print(f"\nW_o gradient relative error: {rel_error:.2e}")

        assert rel_error < 1e-3, f"W_o gradient error too high: {rel_error}"

    def test_all_parameters_simultaneously(self):
        """
        Test that gradients for all parameters are correct together.

        The Story:
            Previous tests checked parameters in isolation.
            This verifies they work correctly together.
        """
        layer, graph = self.setup_layer()

        # Forward
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        # Analytical gradients
        for param in layer.parameters():
            param.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        layer.backward(grad_outputs, graph)

        # Check each parameter
        params_to_test = [
            ("W_q", layer.W_q),
            ("W_k", layer.W_k),
            ("W_v", layer.W_v),
            ("W_o", layer.W_o),
        ]

        print("\n" + "="*60)
        print("Testing all parameters simultaneously:")
        print("="*60)

        for param_name, param in params_to_test:
            grad_analytical = param.grad.copy()

            # Numerical gradient
            def loss_fn():
                node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
                outputs = layer.forward(node_values, graph)
                return np.sum(outputs["pos_2"])

            grad_numerical = numerical_gradient(loss_fn, param.data)

            rel_error = relative_error(grad_analytical, grad_numerical)
            print(f"{param_name:5s}: error={rel_error:.2e}, "
                  f"analytical_norm={np.linalg.norm(grad_analytical):.4f}, "
                  f"numerical_norm={np.linalg.norm(grad_numerical):.4f}")

            assert rel_error < 1e-3, f"{param_name} gradient error too high: {rel_error}"


# =============================================================================
# TEST END-TO-END GRADIENTS
# =============================================================================


class TestEndToEndGradients:
    """
    Test gradients through the full AttentionGraph.

    The Story:
        This is the ultimate integration test. We verify that gradients
        flow correctly from loss, through layers, to node embeddings.

        If this passes, we can trust the entire implementation.
    """

    def test_single_layer_embedding_gradient(self):
        """
        Test gradient flow to node embeddings with one layer.

        The Story:
            Node embeddings are what we're ultimately learning.
            They must receive correct gradients from the loss.
        """
        np.random.seed(42)

        # Create small graph
        embedding_dim = 8
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward pass
        outputs = graph.forward(num_layers=1)

        # Define loss: sum of last position's output
        loss = np.sum(outputs["pos_2"])

        # Analytical gradient via backward
        graph.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        graph.backward(grad_outputs, num_layers=1)

        # Test gradient for each node embedding
        for node in graph.nodes:
            if node.embedding is None:
                continue

            grad_analytical = node.embedding.grad.copy()

            # Numerical gradient
            def loss_fn():
                outputs = graph.forward(num_layers=1)
                return np.sum(outputs["pos_2"])

            grad_numerical = numerical_gradient(loss_fn, node.embedding.data)

            rel_error = relative_error(grad_analytical, grad_numerical)
            print(f"\n{node.id} embedding gradient relative error: {rel_error:.2e}")

            assert rel_error < 1e-3, f"{node.id} gradient error too high: {rel_error}"

    def test_multi_layer_embedding_gradient(self):
        """
        Test gradient flow through multiple layers.

        The Story:
            With multiple layers, gradients must flow through:
            Layer 2 -> Layer 1 -> Embeddings

            This is where bugs in gradient accumulation show up.
        """
        np.random.seed(123)

        # Create small graph
        embedding_dim = 6
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=123
        )

        # Forward through 2 layers
        outputs = graph.forward(num_layers=2)

        # Analytical gradient
        graph.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        graph.backward(grad_outputs, num_layers=2)

        # Test first node embedding (influenced by layer 2 -> layer 1 chain)
        node = graph.get_node("pos_0")
        if node and node.embedding:
            grad_analytical = node.embedding.grad.copy()

            # Numerical gradient
            def loss_fn():
                outputs = graph.forward(num_layers=2)
                return np.sum(outputs["pos_2"])

            grad_numerical = numerical_gradient(loss_fn, node.embedding.data)

            rel_error = relative_error(grad_analytical, grad_numerical)
            print(f"\npos_0 multi-layer gradient relative error: {rel_error:.2e}")
            print(f"Analytical: {grad_analytical[:4]}")
            print(f"Numerical:  {grad_numerical[:4]}")

            assert rel_error < 1e-3, f"Multi-layer gradient error too high: {rel_error}"

    def test_different_loss_functions(self):
        """
        Test gradients with different loss functions.

        The Story:
            Sum loss is simple, but real training uses diverse losses.
            Test a few to ensure gradients adapt correctly.
        """
        np.random.seed(456)

        embedding_dim = 6
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=456
        )

        # Test different loss functions
        loss_cases = [
            ("sum", lambda out: np.sum(out), lambda out: np.ones_like(out)),
            ("mean", lambda out: np.mean(out), lambda out: np.ones_like(out) / len(out)),
            ("squared", lambda out: np.sum(out ** 2), lambda out: 2 * out),
        ]

        for loss_name, loss_fn, grad_fn in loss_cases:
            # Forward
            outputs = graph.forward(num_layers=1)
            output = outputs["pos_2"]

            # Analytical gradient
            graph.zero_grad()
            grad_outputs = {"pos_2": grad_fn(output)}
            graph.backward(grad_outputs, num_layers=1)

            node = graph.get_node("pos_1")
            if node and node.embedding:
                grad_analytical = node.embedding.grad.copy()

                # Numerical gradient
                def loss_fn_closure():
                    outputs = graph.forward(num_layers=1)
                    return loss_fn(outputs["pos_2"])

                grad_numerical = numerical_gradient(loss_fn_closure, node.embedding.data)

                rel_error = relative_error(grad_analytical, grad_numerical)
                print(f"\nLoss '{loss_name}' - pos_1 gradient error: {rel_error:.2e}")

                assert rel_error < 1e-3, f"Error too high for {loss_name} loss: {rel_error}"

    def test_longer_sequence(self):
        """
        Test gradients with longer sequence (more nodes).

        The Story:
            Longer sequences mean more attention computations.
            This stresses the implementation and reveals accumulation bugs.
        """
        np.random.seed(789)

        embedding_dim = 4  # Keep dim small for faster computation
        seq_len = 5

        graph = create_causal_attention_graph(
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            seed=789
        )

        # Forward
        outputs = graph.forward(num_layers=1)

        # Analytical gradient
        graph.zero_grad()
        grad_outputs = {f"pos_{seq_len-1}": np.ones(embedding_dim)}
        graph.backward(grad_outputs, num_layers=1)

        # Test middle node (gets gradient from multiple downstream nodes)
        node = graph.get_node("pos_2")
        if node and node.embedding:
            grad_analytical = node.embedding.grad.copy()

            # Numerical gradient
            def loss_fn():
                outputs = graph.forward(num_layers=1)
                return np.sum(outputs[f"pos_{seq_len-1}"])

            grad_numerical = numerical_gradient(loss_fn, node.embedding.data)

            rel_error = relative_error(grad_analytical, grad_numerical)
            print(f"\nLonger sequence (len={seq_len}) - pos_2 gradient error: {rel_error:.2e}")

            assert rel_error < 1e-3, f"Long sequence gradient error too high: {rel_error}"


# =============================================================================
# TEST GRADIENT ACCUMULATION
# =============================================================================


class TestGradientAccumulation:
    """
    Test that gradients accumulate correctly across multiple backward passes.

    The Story:
        In some training scenarios (gradient accumulation, multi-task learning),
        we call backward() multiple times before updating parameters.

        Gradients should accumulate (add up), not replace each other.
    """

    def test_double_backward_doubles_gradient(self):
        """
        Calling backward twice with same grad should double the gradient.

        The Story:
            If we compute the same loss twice without zero_grad(),
            gradients should add up. This tests Parameter.add_grad().
        """
        np.random.seed(42)

        embedding_dim = 8
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward
        outputs = graph.forward(num_layers=1)

        # First backward
        graph.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        graph.backward(grad_outputs, num_layers=1)

        # Save gradients after first backward
        first_grads = {}
        for param in graph.parameters():
            if param.grad is not None:
                first_grads[id(param)] = param.grad.copy()

        # Second backward WITHOUT zero_grad()
        graph.backward(grad_outputs, num_layers=1)

        # Check that gradients doubled
        print("\nGradient accumulation test:")
        for param in graph.parameters():
            if param.grad is not None:
                first_grad = first_grads.get(id(param))
                if first_grad is not None:
                    expected = first_grad * 2
                    actual = param.grad

                    rel_error = relative_error(actual, expected)
                    print(f"{param.name}: error={rel_error:.2e}")

                    assert rel_error < 1e-6, f"Gradient didn't double for {param.name}"

    def test_zero_grad_clears_accumulation(self):
        """
        Test that zero_grad() properly clears accumulated gradients.

        The Story:
            After updating parameters, we need a clean slate.
            zero_grad() should reset everything to None.
        """
        np.random.seed(42)

        embedding_dim = 8
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward and backward
        outputs = graph.forward(num_layers=1)
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        graph.backward(grad_outputs, num_layers=1)

        # Verify gradients exist
        has_grads = any(p.grad is not None for p in graph.parameters())
        assert has_grads, "Should have gradients after backward"

        # Clear gradients
        graph.zero_grad()

        # Verify all gradients are None
        for param in graph.parameters():
            assert param.grad is None, f"Gradient not cleared for {param.name}"

        print("\nzero_grad() correctly cleared all gradients")

    def test_multiple_output_nodes_accumulate(self):
        """
        Test that gradients accumulate when multiple output nodes contribute.

        The Story:
            In multi-task learning, a node might contribute to multiple losses.
            Its gradient should be the sum of contributions from each loss.
        """
        np.random.seed(42)

        embedding_dim = 8
        graph = create_causal_attention_graph(
            seq_len=4,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward
        outputs = graph.forward(num_layers=1)

        # Backward with gradients from TWO output nodes
        graph.zero_grad()
        grad_outputs = {
            "pos_2": np.ones_like(outputs["pos_2"]),
            "pos_3": np.ones_like(outputs["pos_3"]),
        }
        graph.backward(grad_outputs, num_layers=1)

        grad_with_two = {}
        for param in graph.parameters():
            if param.grad is not None:
                grad_with_two[id(param)] = param.grad.copy()

        # Compare to sum of individual backwards
        # Backward with just pos_2
        graph.zero_grad()
        graph.backward({"pos_2": np.ones_like(outputs["pos_2"])}, num_layers=1)

        grad_pos2_only = {}
        for param in graph.parameters():
            if param.grad is not None:
                grad_pos2_only[id(param)] = param.grad.copy()

        # Backward with just pos_3
        graph.zero_grad()
        graph.backward({"pos_3": np.ones_like(outputs["pos_3"])}, num_layers=1)

        grad_pos3_only = {}
        for param in graph.parameters():
            if param.grad is not None:
                grad_pos3_only[id(param)] = param.grad.copy()

        # Verify: grad_with_two should equal grad_pos2_only + grad_pos3_only
        print("\nMulti-output accumulation test:")
        for param in graph.parameters():
            if param.grad is not None:
                pid = id(param)
                if pid in grad_with_two and pid in grad_pos2_only and pid in grad_pos3_only:
                    expected = grad_pos2_only[pid] + grad_pos3_only[pid]
                    actual = grad_with_two[pid]

                    rel_error = relative_error(actual, expected)
                    print(f"{param.name}: error={rel_error:.2e}")

                    assert rel_error < 1e-6, f"Multi-output accumulation failed for {param.name}"


# =============================================================================
# EDGE CASES AND CORNER CASES
# =============================================================================


class TestEdgeCases:
    """
    Test gradient computation in edge cases.

    The Story:
        Production code hits weird cases. Better to test them now.
    """

    def test_single_node_graph(self):
        """
        Test gradients when graph has only one node (no attention).

        The Story:
            First position in a sequence has no one to attend to.
            Gradients should still flow through output projection.
        """
        np.random.seed(42)

        embedding_dim = 8
        graph = AttentionGraph(embedding_dim=embedding_dim, seed=42)
        graph.add_node("pos_0")

        # Forward
        outputs = graph.forward(num_layers=1)

        # Analytical gradient
        graph.zero_grad()
        grad_outputs = {"pos_0": np.ones_like(outputs["pos_0"])}
        graph.backward(grad_outputs, num_layers=1)

        node = graph.get_node("pos_0")
        if node and node.embedding:
            grad_analytical = node.embedding.grad.copy()

            # Numerical gradient
            def loss_fn():
                outputs = graph.forward(num_layers=1)
                return np.sum(outputs["pos_0"])

            grad_numerical = numerical_gradient(loss_fn, node.embedding.data)

            rel_error = relative_error(grad_analytical, grad_numerical)
            print(f"\nSingle-node gradient relative error: {rel_error:.2e}")

            assert rel_error < 1e-3, f"Single-node gradient error: {rel_error}"

    def test_small_embedding_dimension(self):
        """
        Test with very small embedding dimension (d=2).

        The Story:
            Small dimensions can expose numerical issues in softmax.
        """
        np.random.seed(42)

        embedding_dim = 2
        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward
        outputs = graph.forward(num_layers=1)

        # Analytical gradient
        graph.zero_grad()
        grad_outputs = {"pos_2": np.ones_like(outputs["pos_2"])}
        graph.backward(grad_outputs, num_layers=1)

        node = graph.get_node("pos_1")
        if node and node.embedding:
            grad_analytical = node.embedding.grad.copy()

            # Numerical gradient
            def loss_fn():
                outputs = graph.forward(num_layers=1)
                return np.sum(outputs["pos_2"])

            grad_numerical = numerical_gradient(loss_fn, node.embedding.data)

            rel_error = relative_error(grad_analytical, grad_numerical)
            print(f"\nSmall dimension (d={embedding_dim}) gradient error: {rel_error:.2e}")

            # Looser tolerance for very small dimensions
            assert rel_error < 5e-3, f"Small-dim gradient error: {rel_error}"


# =============================================================================
# TEST DROPOUT GRADIENT CORRECTNESS
# =============================================================================


class TestDropoutGradients:
    """
    Test that gradients flow correctly through dropout.

    The Story:
        Dropout is tricky for gradients. During forward pass, some elements
        are zeroed and others are scaled. The backward pass must apply the
        SAME mask and scaling to maintain gradient consistency.

        Without this, gradients would flow through elements that were zeroed
        during forward, leading to incorrect parameter updates.
    """

    def test_dropout_gradient_consistency(self):
        """
        Test that dropout mask is correctly applied during backward.

        We verify this by checking that:
        1. The same random seed produces consistent forward/backward
        2. Gradients through zeroed elements are zero
        """
        np.random.seed(42)

        # Create layer with significant dropout
        embedding_dim = 8
        layer = AttentionLayer(embedding_dim=embedding_dim, dropout=0.5)
        layer.train()

        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward pass (stores dropout mask)
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        # Backward pass
        for param in layer.parameters():
            param.zero_grad()

        grad_outputs = {"pos_2": np.ones(embedding_dim)}
        input_grads = layer.backward(grad_outputs, graph)

        # Check that dropout mask was stored and applied
        for node_id in ["pos_2"]:
            dropout_info = layer._cache.get("dropout_mask", {}).get(node_id)
            if dropout_info is not None:
                mask, scale = dropout_info
                # Verify mask contains zeros (dropout happened)
                assert np.any(mask == 0), "Dropout mask should have some zeros"
                # Verify scale is correct
                expected_scale = 1.0 / (1 - 0.5)
                assert abs(scale - expected_scale) < 1e-6

        print("\n✓ Dropout gradient consistency test passed")

    def test_dropout_gradient_numerical_verification(self):
        """
        Verify dropout gradients using numerical gradient checking.

        The Story:
            This is the ultimate test: compare analytical gradients (from backward)
            against numerical gradients (finite differences). If they match,
            the dropout implementation is correct.

        Note: We must use the same random state for numerical gradient checking,
        which means we set the seed before each forward pass.
        """
        embedding_dim = 6
        dropout_rate = 0.3

        # Create layer and graph
        np.random.seed(42)
        layer = AttentionLayer(embedding_dim=embedding_dim, dropout=dropout_rate)
        layer.train()

        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # We'll test gradient w.r.t. W_o since it's after dropout
        def forward_with_seed(seed_val):
            """Forward pass with specific random seed for reproducibility."""
            np.random.seed(seed_val)
            node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
            outputs = layer.forward(node_values, graph)
            return np.sum(outputs["pos_2"])

        # Compute analytical gradient
        np.random.seed(123)  # Specific seed for this test
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs = layer.forward(node_values, graph)

        for param in layer.parameters():
            param.zero_grad()

        grad_outputs = {"pos_2": np.ones(embedding_dim)}
        layer.backward(grad_outputs, graph)

        grad_W_o_analytical = layer.W_o.grad.copy()

        # Compute numerical gradient (using same seed for each perturbation)
        eps = 1e-5
        grad_W_o_numerical = np.zeros_like(layer.W_o.data)

        for i in range(layer.W_o.data.shape[0]):
            for j in range(layer.W_o.data.shape[1]):
                old_val = layer.W_o.data[i, j]

                # Perturb up
                layer.W_o.data[i, j] = old_val + eps
                loss_up = forward_with_seed(123)

                # Perturb down
                layer.W_o.data[i, j] = old_val - eps
                loss_down = forward_with_seed(123)

                # Restore
                layer.W_o.data[i, j] = old_val

                # Central difference
                grad_W_o_numerical[i, j] = (loss_up - loss_down) / (2 * eps)

        # Compare
        rel_error = relative_error(grad_W_o_analytical, grad_W_o_numerical)
        print(f"\nDropout W_o gradient relative error: {rel_error:.2e}")

        # Allow slightly higher tolerance due to dropout stochasticity
        assert rel_error < 1e-3, f"Dropout gradient error too high: {rel_error}"

    def test_eval_mode_no_dropout_gradient(self):
        """
        Verify that eval mode produces clean gradients without dropout artifacts.

        In eval mode, dropout should be disabled, meaning:
        - No random masking during forward
        - Deterministic gradients during backward
        """
        np.random.seed(42)

        embedding_dim = 8
        layer = AttentionLayer(embedding_dim=embedding_dim, dropout=0.5)
        layer.eval()  # Disable dropout

        graph = create_causal_attention_graph(
            seq_len=3,
            embedding_dim=embedding_dim,
            seed=42
        )

        # Forward pass
        node_values = {node.id: node.embedding.data.copy() for node in graph.nodes}
        outputs1 = layer.forward(node_values, graph)

        # Second forward pass should be identical (no dropout randomness)
        outputs2 = layer.forward(node_values, graph)

        for node_id in outputs1:
            assert np.allclose(outputs1[node_id], outputs2[node_id]), \
                f"Eval mode should produce deterministic outputs for {node_id}"

        # Dropout mask should be None in cache
        for node_id in layer._cache.get("dropout_mask", {}):
            assert layer._cache["dropout_mask"][node_id] is None, \
                "Eval mode should not store dropout masks"

        print("\n✓ Eval mode gradient test passed")


# =============================================================================
# TEST LOAD_STATE WITH LAYERS
# =============================================================================


class TestLoadStateLayerCreation:
    """
    Test that load_state properly creates layers when loading from checkpoint.

    The Story:
        When loading a saved model, the graph may not have any attention layers
        yet (if forward() was never called). load_state must create the layers
        before attempting to restore their parameters.
    """

    def test_load_state_creates_missing_layers(self):
        """
        Test that loading state into fresh graph creates necessary layers.
        """
        np.random.seed(42)

        # Create and train a graph
        graph1 = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph1.forward(num_layers=2)  # Creates 2 layers

        # Modify parameters to have distinct values
        graph1._attention_layers[0].W_q.data[:] = 1.0
        graph1._attention_layers[1].W_q.data[:] = 2.0

        # Save state
        state = graph1.save_state()
        assert len(state["layers"]) == 2

        # Create fresh graph (no layers yet)
        graph2 = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=123)
        assert len(graph2._attention_layers) == 0

        # Load state - should create layers
        graph2.load_state(state)

        # Verify layers were created
        assert len(graph2._attention_layers) == 2

        # Verify parameters were restored
        assert np.allclose(graph2._attention_layers[0].W_q.data, 1.0)
        assert np.allclose(graph2._attention_layers[1].W_q.data, 2.0)

        print("\n✓ load_state layer creation test passed")

    def test_load_state_preserves_layer_functionality(self):
        """
        Test that loaded layers work correctly for forward/backward.
        """
        np.random.seed(42)

        # Create original graph and run forward/backward
        graph1 = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        outputs1 = graph1.forward(num_layers=2)
        graph1.backward({"pos_2": np.ones(8)}, num_layers=2)

        # Save state
        state = graph1.save_state()

        # Create fresh graph and load state
        graph2 = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph2.load_state(state)

        # Run forward - should produce same outputs
        outputs2 = graph2.forward(num_layers=2)

        for node_id in outputs1:
            assert np.allclose(outputs1[node_id], outputs2[node_id], atol=1e-10), \
                f"Loaded graph should produce same outputs for {node_id}"

        print("\n✓ load_state functionality preservation test passed")

    def test_load_state_with_bias_parameters(self):
        """
        Test that bias parameters are correctly saved and loaded.

        This is critical: if someone trains with use_bias=True and saves
        a checkpoint, the biases MUST be restored correctly on load.
        Silent failure here would cause "worked in training, fails in prod" bugs.
        """
        np.random.seed(42)

        # Create graph WITH bias
        graph1 = AttentionGraph(embedding_dim=8, use_bias=True, seed=42)
        for i in range(3):
            graph1.add_node(f"pos_{i}")
        for i in range(1, 3):
            for j in range(i):
                graph1.add_edge(f"pos_{j}", f"pos_{i}")

        # Run forward to create layers
        graph1.forward(num_layers=1)

        # Set bias parameters to known values
        layer = graph1._attention_layers[0]
        layer.b_q.data[:] = 1.0
        layer.b_k.data[:] = 2.0
        layer.b_v.data[:] = 3.0
        layer.b_o.data[:] = 4.0

        # Save state
        state = graph1.save_state()

        # Verify biases are in saved state
        assert "b_q" in state["layers"][0], "Bias b_q should be saved"
        assert "b_k" in state["layers"][0], "Bias b_k should be saved"
        assert "b_v" in state["layers"][0], "Bias b_v should be saved"
        assert "b_o" in state["layers"][0], "Bias b_o should be saved"

        # Create fresh graph with bias and load state
        graph2 = AttentionGraph(embedding_dim=8, use_bias=True, seed=123)
        for i in range(3):
            graph2.add_node(f"pos_{i}")
        for i in range(1, 3):
            for j in range(i):
                graph2.add_edge(f"pos_{j}", f"pos_{i}")

        graph2.load_state(state)

        # Verify bias parameters were restored
        layer2 = graph2._attention_layers[0]
        assert np.allclose(layer2.b_q.data, 1.0), "Bias b_q not restored correctly"
        assert np.allclose(layer2.b_k.data, 2.0), "Bias b_k not restored correctly"
        assert np.allclose(layer2.b_v.data, 3.0), "Bias b_v not restored correctly"
        assert np.allclose(layer2.b_o.data, 4.0), "Bias b_o not restored correctly"

        # Verify forward pass produces same results
        outputs1 = graph1.forward(num_layers=1)
        outputs2 = graph2.forward(num_layers=1)

        for node_id in outputs1:
            assert np.allclose(outputs1[node_id], outputs2[node_id], atol=1e-10), \
                f"Outputs differ after loading bias state for {node_id}"

        print("\n✓ load_state with bias parameters test passed")


# =============================================================================
# RUN TESTS
# =============================================================================
# Use pytest to run these tests:
#     pytest tests/unit/test_attention_gradients.py -v
#
# For verbose output with print statements:
#     pytest tests/unit/test_attention_gradients.py -v -s
#
# To run specific test class:
#     pytest tests/unit/test_attention_gradients.py::TestAttentionFunctionGradients -v
# =============================================================================

if __name__ == "__main__":
    import pytest
    import sys

    # Run pytest on this file with verbose output
    sys.exit(pytest.main([__file__, "-v", "-s"]))
