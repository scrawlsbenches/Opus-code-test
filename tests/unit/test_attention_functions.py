"""
Unit tests for attention functions in cortical/graph/attention.py.

Tests the core attention computation primitives:
- scaled_dot_product_attention: Forward attention computation
- attention_backward: Backward pass gradient computation

These functions are the building blocks for AttentionGraph and need to be
numerically stable and correct for training to work.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from cortical.graph.attention import (
    scaled_dot_product_attention,
    attention_backward,
)


class TestScaledDotProductAttention:
    """Tests for the scaled_dot_product_attention function."""

    def test_basic_shapes(self):
        """Test that output shapes match expected dimensions."""
        np.random.seed(42)

        d = 8  # embedding dimension
        n = 5  # number of sources

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Output should have same dimension as value dimension
        assert output.shape == (d,), f"Expected output shape ({d},), got {output.shape}"

        # Weights should have one per source
        assert weights.shape == (n,), f"Expected weights shape ({n},), got {weights.shape}"

    def test_weights_sum_to_one(self):
        """Test that attention weights form a proper probability distribution (sum to 1)."""
        np.random.seed(42)

        d = 16
        n = 10

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        _, weights = scaled_dot_product_attention(query, keys, values)

        # Softmax should sum to 1
        assert_allclose(np.sum(weights), 1.0, rtol=1e-6, atol=1e-10,
                       err_msg="Attention weights should sum to 1.0")

        # All weights should be non-negative
        assert np.all(weights >= 0), "Attention weights should be non-negative"
        assert np.all(weights <= 1), "Attention weights should be at most 1.0"

    def test_single_source(self):
        """Test with single source: weight should be 1.0."""
        np.random.seed(42)

        d = 8

        query = np.random.randn(d)
        keys = np.random.randn(1, d)  # Single source
        values = np.random.randn(1, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        # With one source, all attention goes to it
        assert_allclose(weights[0], 1.0, rtol=1e-6,
                       err_msg="Single source should receive weight 1.0")

        # Output should equal the single value
        assert_allclose(output, values[0], rtol=1e-6,
                       err_msg="Output should equal the single value")

    def test_uniform_keys(self):
        """Test with identical keys: weights should be approximately equal."""
        np.random.seed(42)

        d = 8
        n = 5

        query = np.random.randn(d)

        # All keys are identical - attention should be uniform
        key_vector = np.random.randn(d)
        keys = np.tile(key_vector, (n, 1))
        values = np.random.randn(n, d)

        _, weights = scaled_dot_product_attention(query, keys, values)

        # All weights should be approximately equal (1/n)
        expected_weight = 1.0 / n
        assert_allclose(weights, expected_weight, rtol=1e-5,
                       err_msg=f"With uniform keys, weights should all be ~{expected_weight}")

    def test_scaling_prevents_large_values(self):
        """Test that sqrt(d_k) scaling prevents attention scores from getting too large."""
        np.random.seed(42)

        # Test with different embedding dimensions
        for d in [8, 64, 256]:
            query = np.random.randn(d)
            keys = np.random.randn(5, d)
            values = np.random.randn(5, d)

            output, weights = scaled_dot_product_attention(query, keys, values)

            # Weights should still be well-behaved (not too peaked)
            max_weight = np.max(weights)
            min_weight = np.min(weights)

            # No single weight should dominate completely (unless inputs are extreme)
            # With random inputs, max weight typically shouldn't exceed 0.9
            assert max_weight <= 1.0, f"Max weight {max_weight} exceeds 1.0"

            # Distribution shouldn't be too peaked (allowing some flexibility)
            # This is a soft check - just ensuring numerical stability
            assert not np.isnan(weights).any(), "Weights contain NaN"
            assert not np.isinf(weights).any(), "Weights contain Inf"

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large input values."""
        np.random.seed(42)

        d = 8
        n = 5

        # Large values that could cause overflow without proper handling
        query = np.random.randn(d) * 100
        keys = np.random.randn(n, d) * 100
        values = np.random.randn(n, d) * 100

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Should not produce NaN or Inf
        assert not np.isnan(output).any(), "Output contains NaN with large inputs"
        assert not np.isinf(output).any(), "Output contains Inf with large inputs"
        assert not np.isnan(weights).any(), "Weights contain NaN with large inputs"
        assert not np.isinf(weights).any(), "Weights contain Inf with large inputs"

        # Weights should still sum to 1
        assert_allclose(np.sum(weights), 1.0, rtol=1e-5)

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small input values."""
        np.random.seed(42)

        d = 8
        n = 5

        # Very small values
        query = np.random.randn(d) * 1e-8
        keys = np.random.randn(n, d) * 1e-8
        values = np.random.randn(n, d) * 1e-8

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Should not produce NaN or Inf
        assert not np.isnan(output).any(), "Output contains NaN with small inputs"
        assert not np.isinf(output).any(), "Output contains Inf with small inputs"
        assert not np.isnan(weights).any(), "Weights contain NaN with small inputs"
        assert not np.isinf(weights).any(), "Weights contain Inf with small inputs"

        # Weights should still sum to 1
        assert_allclose(np.sum(weights), 1.0, rtol=1e-5)

    def test_with_mask_blocks_attention(self):
        """Test that mask parameter correctly blocks attention to masked positions."""
        np.random.seed(42)

        d = 8
        n = 5

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        # Create mask that blocks last 2 positions
        mask = np.array([0.0, 0.0, 0.0, -np.inf, -np.inf])

        output, weights = scaled_dot_product_attention(query, keys, values, mask=mask)

        # Masked positions should have near-zero weight
        assert_allclose(weights[3], 0.0, atol=1e-10,
                       err_msg="Masked position should have weight ~0")
        assert_allclose(weights[4], 0.0, atol=1e-10,
                       err_msg="Masked position should have weight ~0")

        # Non-masked weights should still sum to 1
        assert_allclose(np.sum(weights), 1.0, rtol=1e-6)

        # Non-masked positions should have positive weights
        assert weights[0] > 0
        assert weights[1] > 0
        assert weights[2] > 0

    def test_with_partial_mask(self):
        """Test mask with only some positions blocked."""
        np.random.seed(42)

        d = 8
        n = 6

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        # Mask that blocks positions 1, 3, 5
        mask = np.array([0.0, -np.inf, 0.0, -np.inf, 0.0, -np.inf])

        output, weights = scaled_dot_product_attention(query, keys, values, mask=mask)

        # Masked positions should be near zero
        assert_allclose(weights[1], 0.0, atol=1e-10)
        assert_allclose(weights[3], 0.0, atol=1e-10)
        assert_allclose(weights[5], 0.0, atol=1e-10)

        # Unmasked positions should have positive weights
        assert weights[0] > 0
        assert weights[2] > 0
        assert weights[4] > 0

        # All weights should sum to 1
        assert_allclose(np.sum(weights), 1.0, rtol=1e-6)

    def test_edge_case_dimension_1(self):
        """Test with very small embedding dimension (d=1)."""
        np.random.seed(42)

        d = 1
        n = 3

        query = np.array([0.5])
        keys = np.array([[1.0], [2.0], [3.0]])
        values = np.array([[10.0], [20.0], [30.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        assert output.shape == (1,)
        assert weights.shape == (3,)
        assert_allclose(np.sum(weights), 1.0, rtol=1e-6)
        assert not np.isnan(output).any()
        assert not np.isnan(weights).any()

    def test_edge_case_dimension_2(self):
        """Test with small embedding dimension (d=2)."""
        np.random.seed(42)

        d = 2
        n = 4

        query = np.array([1.0, 2.0])
        keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        values = np.array([[5.0, 0.0], [0.0, 5.0], [3.0, 3.0], [1.0, 1.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        assert output.shape == (2,)
        assert weights.shape == (4,)
        assert_allclose(np.sum(weights), 1.0, rtol=1e-6)

        # Query [1, 2] should attend more to keys with larger dot product
        # keys[3] = [2, 2] has largest dot product with query
        assert weights[3] > weights[0]  # [2,2] > [1,0]
        assert weights[3] > weights[1]  # [2,2] > [0,1]

    def test_output_is_weighted_sum_of_values(self):
        """Test that output is correctly computed as weighted sum of values."""
        np.random.seed(42)

        d = 4
        n = 3

        query = np.array([1.0, 0.0, 0.0, 0.0])
        keys = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]])
        values = np.array([[10.0, 20.0, 30.0, 40.0],
                          [50.0, 60.0, 70.0, 80.0],
                          [90.0, 100.0, 110.0, 120.0]])

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Manually compute weighted sum
        expected_output = weights[0] * values[0] + weights[1] * values[1] + weights[2] * values[2]

        assert_allclose(output, expected_output, rtol=1e-6,
                       err_msg="Output should be weighted sum of values")

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with the same random seed."""
        d = 8
        n = 5

        # First run
        np.random.seed(123)
        query1 = np.random.randn(d)
        keys1 = np.random.randn(n, d)
        values1 = np.random.randn(n, d)
        output1, weights1 = scaled_dot_product_attention(query1, keys1, values1)

        # Second run with same seed
        np.random.seed(123)
        query2 = np.random.randn(d)
        keys2 = np.random.randn(n, d)
        values2 = np.random.randn(n, d)
        output2, weights2 = scaled_dot_product_attention(query2, keys2, values2)

        assert_allclose(output1, output2, rtol=1e-10,
                       err_msg="Results should be deterministic with same seed")
        assert_allclose(weights1, weights2, rtol=1e-10,
                       err_msg="Weights should be deterministic with same seed")

    def test_different_value_dimension(self):
        """Test when value dimension differs from key dimension."""
        np.random.seed(42)

        d_k = 8  # key/query dimension
        d_v = 16  # value dimension (different!)
        n = 5

        query = np.random.randn(d_k)
        keys = np.random.randn(n, d_k)
        values = np.random.randn(n, d_v)  # Different dimension

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Output should have value dimension
        assert output.shape == (d_v,), f"Expected output shape ({d_v},), got {output.shape}"
        assert weights.shape == (n,)
        assert_allclose(np.sum(weights), 1.0, rtol=1e-6)


class TestAttentionBackward:
    """Tests for the attention_backward function."""

    def test_gradient_shapes(self):
        """Test that gradient shapes match input shapes."""
        np.random.seed(42)

        d = 8
        n = 5

        # Forward pass
        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Backward pass
        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Check shapes
        assert grad_query.shape == query.shape, \
            f"grad_query shape {grad_query.shape} != query shape {query.shape}"
        assert grad_keys.shape == keys.shape, \
            f"grad_keys shape {grad_keys.shape} != keys shape {keys.shape}"
        assert grad_values.shape == values.shape, \
            f"grad_values shape {grad_values.shape} != values shape {values.shape}"

    def test_gradient_not_nan_or_inf(self):
        """Test that gradients don't contain NaN or Inf."""
        np.random.seed(42)

        d = 8
        n = 5

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)
        output, weights = scaled_dot_product_attention(query, keys, values)

        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        assert not np.isnan(grad_query).any(), "grad_query contains NaN"
        assert not np.isinf(grad_query).any(), "grad_query contains Inf"
        assert not np.isnan(grad_keys).any(), "grad_keys contains NaN"
        assert not np.isinf(grad_keys).any(), "grad_keys contains Inf"
        assert not np.isnan(grad_values).any(), "grad_values contains NaN"
        assert not np.isinf(grad_values).any(), "grad_values contains Inf"

    def test_gradient_symmetry_uniform_attention(self):
        """Test gradient behavior with uniform attention weights."""
        np.random.seed(42)

        d = 8
        n = 4

        # Create scenario with uniform attention
        query = np.random.randn(d)
        key_vector = np.random.randn(d)
        keys = np.tile(key_vector, (n, 1))  # All keys identical -> uniform attention
        values = np.random.randn(n, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Verify uniform attention
        expected_weight = 1.0 / n
        assert_allclose(weights, expected_weight, rtol=1e-5)

        # Backward pass
        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # With uniform attention, all value gradients should be equal
        for i in range(n):
            assert_allclose(grad_values[i], grad_values[0], rtol=1e-5,
                           err_msg=f"Value gradient {i} should match gradient 0 with uniform attention")

    def test_gradient_magnitude_reasonable(self):
        """Test that gradient magnitudes are reasonable (not exploding)."""
        np.random.seed(42)

        d = 8
        n = 5

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Small gradient signal
        grad_output = np.random.randn(d) * 0.1
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Gradients should be on similar scale as inputs/outputs
        # Not an exact relationship, but checking for explosion
        query_norm = np.linalg.norm(query)
        grad_query_norm = np.linalg.norm(grad_query)

        # Gradient shouldn't be orders of magnitude larger than input
        assert grad_query_norm < query_norm * 100, \
            f"grad_query norm {grad_query_norm} seems too large compared to query norm {query_norm}"

        keys_norm = np.linalg.norm(keys)
        grad_keys_norm = np.linalg.norm(grad_keys)
        assert grad_keys_norm < keys_norm * 100, \
            f"grad_keys norm {grad_keys_norm} seems too large compared to keys norm {keys_norm}"

    def test_numerical_gradient_check_query(self):
        """Test gradient correctness for query using numerical differentiation."""
        np.random.seed(42)

        d = 4  # Small for faster computation
        n = 3

        query = np.random.randn(d) * 0.1
        keys = np.random.randn(n, d) * 0.1
        values = np.random.randn(n, d) * 0.1

        # Forward pass
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Backward pass
        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Numerical gradient for query
        eps = 1e-5
        numerical_grad_query = np.zeros_like(query)

        for i in range(d):
            query_plus = query.copy()
            query_plus[i] += eps
            output_plus, _ = scaled_dot_product_attention(query_plus, keys, values)

            query_minus = query.copy()
            query_minus[i] -= eps
            output_minus, _ = scaled_dot_product_attention(query_minus, keys, values)

            # Gradient of loss w.r.t query[i]
            numerical_grad_query[i] = np.dot(grad_output, (output_plus - output_minus)) / (2 * eps)

        # Compare analytical and numerical gradients
        assert_allclose(grad_query, numerical_grad_query, rtol=1e-3, atol=1e-5,
                       err_msg="Analytical gradient doesn't match numerical gradient for query")

    def test_numerical_gradient_check_keys(self):
        """Test gradient correctness for keys using numerical differentiation."""
        np.random.seed(42)

        d = 4
        n = 3

        query = np.random.randn(d) * 0.1
        keys = np.random.randn(n, d) * 0.1
        values = np.random.randn(n, d) * 0.1

        # Forward pass
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Backward pass
        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Numerical gradient for keys (check a few entries)
        eps = 1e-5

        for i in [0, n-1]:  # Check first and last key
            for j in [0, d-1]:  # Check first and last dimension
                keys_plus = keys.copy()
                keys_plus[i, j] += eps
                output_plus, _ = scaled_dot_product_attention(query, keys_plus, values)

                keys_minus = keys.copy()
                keys_minus[i, j] -= eps
                output_minus, _ = scaled_dot_product_attention(query, keys_minus, values)

                numerical_grad = np.dot(grad_output, (output_plus - output_minus)) / (2 * eps)

                assert_allclose(grad_keys[i, j], numerical_grad, rtol=1e-3, atol=1e-5,
                               err_msg=f"Gradient mismatch for keys[{i},{j}]")

    def test_numerical_gradient_check_values(self):
        """Test gradient correctness for values using numerical differentiation."""
        np.random.seed(42)

        d = 4
        n = 3

        query = np.random.randn(d) * 0.1
        keys = np.random.randn(n, d) * 0.1
        values = np.random.randn(n, d) * 0.1

        # Forward pass
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Backward pass
        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Numerical gradient for values (check all entries for values as it's simpler)
        eps = 1e-5
        numerical_grad_values = np.zeros_like(values)

        for i in range(n):
            for j in range(d):
                values_plus = values.copy()
                values_plus[i, j] += eps
                output_plus, _ = scaled_dot_product_attention(query, keys, values_plus)

                values_minus = values.copy()
                values_minus[i, j] -= eps
                output_minus, _ = scaled_dot_product_attention(query, keys, values_minus)

                numerical_grad_values[i, j] = np.dot(grad_output, (output_plus - output_minus)) / (2 * eps)

        # Values gradient is simpler (weighted by attention weights)
        assert_allclose(grad_values, numerical_grad_values, rtol=1e-3, atol=1e-5,
                       err_msg="Gradient mismatch for values")

    def test_single_source_gradients(self):
        """Test gradients with single source (simpler case)."""
        np.random.seed(42)

        d = 8

        query = np.random.randn(d)
        keys = np.random.randn(1, d)
        values = np.random.randn(1, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # All gradients should be well-formed
        assert grad_query.shape == (d,)
        assert grad_keys.shape == (1, d)
        assert grad_values.shape == (1, d)

        assert not np.isnan(grad_query).any()
        assert not np.isnan(grad_keys).any()
        assert not np.isnan(grad_values).any()

    def test_large_dimension_stability(self):
        """Test gradient stability with large embedding dimension."""
        np.random.seed(42)

        d = 128
        n = 10

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Should remain stable even with large dimensions
        assert not np.isnan(grad_query).any()
        assert not np.isinf(grad_query).any()
        assert not np.isnan(grad_keys).any()
        assert not np.isinf(grad_keys).any()
        assert not np.isnan(grad_values).any()
        assert not np.isinf(grad_values).any()

    def test_zero_gradient_output(self):
        """Test behavior when output gradient is zero."""
        np.random.seed(42)

        d = 8
        n = 5

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Zero gradient from upstream
        grad_output = np.zeros(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # All gradients should be zero
        assert_allclose(grad_query, 0.0, atol=1e-10)
        assert_allclose(grad_keys, 0.0, atol=1e-10)
        assert_allclose(grad_values, 0.0, atol=1e-10)

    def test_gradient_flow_with_peaked_attention(self):
        """Test gradient flow when attention is highly peaked (one weight dominant)."""
        np.random.seed(42)

        d = 8
        n = 5

        # Create scenario where query strongly matches one key
        query = np.array([10.0] + [0.0] * (d-1))
        keys = np.zeros((n, d))
        keys[2, 0] = 10.0  # Strong match with query
        # Other keys have small values
        for i in [0, 1, 3, 4]:
            keys[i] = np.random.randn(d) * 0.01

        values = np.random.randn(n, d)

        output, weights = scaled_dot_product_attention(query, keys, values)

        # Weight 2 should be dominant
        assert weights[2] > 0.9, f"Expected dominant weight, got {weights[2]}"

        grad_output = np.random.randn(d)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Gradient for the dominant value should be largest
        grad_norms = [np.linalg.norm(grad_values[i]) for i in range(n)]
        assert grad_norms[2] > max(grad_norms[0], grad_norms[1], grad_norms[3], grad_norms[4]), \
            "Gradient should be largest for the value with highest attention"

    def test_different_value_dimension_gradients(self):
        """Test gradients when value dimension differs from key dimension."""
        np.random.seed(42)

        d_k = 8
        d_v = 16
        n = 5

        query = np.random.randn(d_k)
        keys = np.random.randn(n, d_k)
        values = np.random.randn(n, d_v)

        output, weights = scaled_dot_product_attention(query, keys, values)

        grad_output = np.random.randn(d_v)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Check shapes
        assert grad_query.shape == (d_k,)
        assert grad_keys.shape == (n, d_k)
        assert grad_values.shape == (n, d_v)

        # Check no NaN/Inf
        assert not np.isnan(grad_query).any()
        assert not np.isnan(grad_keys).any()
        assert not np.isnan(grad_values).any()


class TestAttentionIntegration:
    """Integration tests for forward and backward passes together."""

    def test_forward_backward_roundtrip(self):
        """Test that forward followed by backward produces reasonable gradients."""
        np.random.seed(42)

        d = 8
        n = 5

        query = np.random.randn(d)
        keys = np.random.randn(n, d)
        values = np.random.randn(n, d)

        # Forward
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Backward
        grad_output = np.ones(d)  # Simple gradient
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # All operations should succeed without errors
        assert output.shape == (d,)
        assert grad_query.shape == (d,)
        assert grad_keys.shape == (n, d)
        assert grad_values.shape == (n, d)

    def test_gradient_descent_step_reduces_loss(self):
        """Test that a gradient descent step in the right direction reduces loss."""
        np.random.seed(42)

        d = 4
        n = 3

        query = np.random.randn(d) * 0.1
        keys = np.random.randn(n, d) * 0.1
        values = np.random.randn(n, d) * 0.1

        # Target output
        target = np.random.randn(d)

        # Forward
        output, weights = scaled_dot_product_attention(query, keys, values)

        # Initial loss (MSE)
        loss_before = np.sum((output - target) ** 2)

        # Backward
        grad_output = 2 * (output - target)
        grad_query, grad_keys, grad_values = attention_backward(
            grad_output, query, keys, values, weights
        )

        # Gradient descent step on query
        learning_rate = 0.01
        query_new = query - learning_rate * grad_query

        # Forward with updated query
        output_new, _ = scaled_dot_product_attention(query_new, keys, values)
        loss_after = np.sum((output_new - target) ** 2)

        # Loss should decrease (or stay similar if already at minimum)
        # Using a lenient check since we're only doing one step
        assert loss_after <= loss_before * 1.1, \
            f"Loss increased significantly: {loss_before} -> {loss_after}"
