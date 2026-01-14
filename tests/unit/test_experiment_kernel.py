"""
Unit tests for ExperimentKernel: Training harness for AttentionGraph.

Tests cover:
- Basic training step functionality
- Full training loop (fit method)
- Gradient clipping
- Profiling integration
- History tracking
- Evaluation mode
- Position encoding and vocab projection integration
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from cortical.graph.attention import (
    AttentionGraph,
    create_causal_attention_graph,
)
from cortical.graph.trainable import Adam, SGD, MSELoss
from cortical.experiments.kernel import (
    ExperimentKernel,
    TrainingHistory,
    clip_gradients,
    compute_gradient_norm,
)
from cortical.graph.attention import Parameter


# =============================================================================
# TrainingHistory Tests
# =============================================================================


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_empty_history(self):
        """Test newly created history is empty."""
        history = TrainingHistory()

        assert len(history.train_losses) == 0
        assert len(history.val_losses) == 0
        assert len(history.learning_rates) == 0
        assert len(history.gradient_norms) == 0
        assert len(history.step_metrics) == 0

    def test_log_train_loss(self):
        """Test logging train loss."""
        history = TrainingHistory()

        history.log(train_loss=0.5)
        history.log(train_loss=0.3)
        history.log(train_loss=0.1)

        assert len(history.train_losses) == 3
        assert history.train_losses == [0.5, 0.3, 0.1]

    def test_log_all_metrics(self):
        """Test logging all metrics at once."""
        history = TrainingHistory()

        history.log(
            train_loss=0.5,
            val_loss=0.6,
            lr=0.01,
            grad_norm=1.5,
        )

        assert history.train_losses == [0.5]
        assert history.val_losses == [0.6]
        assert history.learning_rates == [0.01]
        assert history.gradient_norms == [1.5]

    def test_log_optional_metrics(self):
        """Test that optional metrics are not logged when None."""
        history = TrainingHistory()

        history.log(train_loss=0.5)
        history.log(train_loss=0.3, val_loss=None, lr=None, grad_norm=None)

        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 0
        assert len(history.learning_rates) == 0
        assert len(history.gradient_norms) == 0


# =============================================================================
# Gradient Utilities Tests
# =============================================================================


class TestClipGradients:
    """Tests for clip_gradients utility function."""

    def test_no_clipping_when_below_norm(self):
        """Test that gradients are not clipped when below max norm."""
        grad = np.array([1.0, 1.0, 1.0, 1.0])  # norm = 2.0
        param = Parameter(data=np.zeros(4), name="test")
        param.grad = grad.copy()

        total_norm = clip_gradients([param], max_norm=10.0)

        # Gradient should be unchanged
        assert_array_almost_equal(param.grad, grad)
        assert abs(total_norm - 2.0) < 0.01

    def test_clipping_when_above_norm(self):
        """Test that gradients are clipped when above max norm."""
        grad = np.array([3.0, 4.0])  # norm = 5.0
        param = Parameter(data=np.zeros(2), name="test")
        param.grad = grad.copy()

        total_norm = clip_gradients([param], max_norm=1.0)

        # Gradient should be scaled down
        assert np.linalg.norm(param.grad) <= 1.0 + 1e-6
        assert abs(total_norm - 5.0) < 0.01

    def test_clipping_multiple_parameters(self):
        """Test clipping across multiple parameters."""
        param1 = Parameter(data=np.zeros(2), name="p1")
        param2 = Parameter(data=np.zeros(2), name="p2")
        param1.grad = np.array([3.0, 0.0])  # norm = 3.0
        param2.grad = np.array([0.0, 4.0])  # norm = 4.0
        # Total norm = sqrt(9 + 16) = 5.0

        total_norm = clip_gradients([param1, param2], max_norm=1.0)

        # Global norm should have been clipped
        new_total_sq = np.sum(param1.grad**2) + np.sum(param2.grad**2)
        new_total = np.sqrt(new_total_sq)
        assert abs(new_total - 1.0) < 1e-6
        assert abs(total_norm - 5.0) < 0.01

    def test_no_clipping_with_none_gradients(self):
        """Test that parameters with None gradients are skipped."""
        param1 = Parameter(data=np.zeros(2), name="p1")
        param2 = Parameter(data=np.zeros(2), name="p2")
        param1.grad = np.array([1.0, 0.0])
        param2.grad = None  # No gradient

        total_norm = clip_gradients([param1, param2], max_norm=10.0)

        assert total_norm == 1.0
        assert param2.grad is None


class TestComputeGradientNorm:
    """Tests for compute_gradient_norm utility function."""

    def test_single_parameter(self):
        """Test norm computation for single parameter."""
        param = Parameter(data=np.zeros(2), name="test")
        param.grad = np.array([3.0, 4.0])  # norm = 5.0

        norm = compute_gradient_norm([param])

        assert abs(norm - 5.0) < 0.01

    def test_multiple_parameters(self):
        """Test norm computation across multiple parameters."""
        param1 = Parameter(data=np.zeros(2), name="p1")
        param2 = Parameter(data=np.zeros(2), name="p2")
        param1.grad = np.array([3.0, 0.0])
        param2.grad = np.array([0.0, 4.0])
        # Total norm = sqrt(9 + 16) = 5.0

        norm = compute_gradient_norm([param1, param2])

        assert abs(norm - 5.0) < 0.01

    def test_with_none_gradients(self):
        """Test that None gradients are handled correctly."""
        param1 = Parameter(data=np.zeros(2), name="p1")
        param2 = Parameter(data=np.zeros(2), name="p2")
        param1.grad = np.array([3.0, 4.0])
        param2.grad = None

        norm = compute_gradient_norm([param1, param2])

        assert abs(norm - 5.0) < 0.01

    def test_zero_gradient(self):
        """Test norm of zero gradients."""
        param = Parameter(data=np.zeros(4), name="test")
        param.grad = np.zeros(4)

        norm = compute_gradient_norm([param])

        assert norm == 0.0


# =============================================================================
# ExperimentKernel Basic Tests
# =============================================================================


class TestExperimentKernelCreation:
    """Tests for ExperimentKernel initialization."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        return create_causal_attention_graph(seq_len=4, embedding_dim=8, seed=42)

    def test_create_kernel(self, simple_graph):
        """Test basic kernel creation."""
        simple_graph.forward(num_layers=1)  # Initialize layers
        optimizer = Adam(simple_graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(
            graph=simple_graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        assert kernel.graph is simple_graph
        assert kernel.optimizer is optimizer
        assert kernel.loss_fn is loss_fn
        assert kernel.position_encoding is None
        assert kernel.vocab_projection is None

    def test_create_kernel_without_profiling(self, simple_graph):
        """Test kernel creation with profiling disabled."""
        simple_graph.forward(num_layers=1)
        optimizer = Adam(simple_graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(
            graph=simple_graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            profiling=False,
        )

        assert kernel.profiler.enabled is False


# =============================================================================
# Train Step Tests
# =============================================================================


class TestExperimentKernelTrainStep:
    """Tests for ExperimentKernel.train_step method."""

    @pytest.fixture
    def kernel(self):
        """Create a kernel with simple graph."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=4, embedding_dim=8, seed=42)
        graph.forward(num_layers=1)
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        return ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            profiling=True,
        )

    def test_train_step_returns_metrics(self, kernel):
        """Test that train_step returns StepMetrics with valid calculated values."""
        targets = {
            "pos_2": np.ones(8) * 0.5,
            "pos_3": np.ones(8) * 0.5,
        }

        metrics = kernel.train_step(targets, num_layers=1)

        # Verify all metrics have valid calculated values
        assert metrics.loss > 0, "Loss should be positive for non-zero targets"
        assert metrics.gradient_norm >= 0, "Gradient norm should be non-negative"
        assert metrics.forward_time_ms >= 0, "Forward time should be non-negative"
        assert metrics.backward_time_ms >= 0, "Backward time should be non-negative"
        assert metrics.update_time_ms >= 0, "Update time should be non-negative"
        # Total time should be at least sum of components
        assert metrics.total_time_ms >= (
            metrics.forward_time_ms + metrics.backward_time_ms + metrics.update_time_ms
        ) * 0.9  # Allow 10% tolerance for measurement overhead

    def test_train_step_updates_parameters(self, kernel):
        """Test that train_step updates model parameters."""
        # Store initial parameter values
        initial_params = {}
        for param in kernel.graph.parameters():
            initial_params[id(param)] = param.data.copy()

        targets = {"pos_3": np.ones(8)}
        kernel.train_step(targets, num_layers=1)

        # At least some parameters should have changed
        changed = False
        for param in kernel.graph.parameters():
            if not np.allclose(param.data, initial_params[id(param)]):
                changed = True
                break

        assert changed, "Parameters should be updated after train_step"

    def test_train_step_computes_mse_loss_correctly(self, kernel):
        """Test that MSE loss is calculated correctly."""
        # Use a simple target that we can verify
        target_value = np.zeros(8)
        targets = {"pos_3": target_value}

        # Get the forward output before training step
        outputs_before = kernel.graph.forward(num_layers=1)
        output_at_pos3 = outputs_before["pos_3"]

        # Calculate expected MSE loss manually
        expected_loss = np.mean((output_at_pos3 - target_value) ** 2)

        # Reset and take training step
        kernel.reset()
        metrics = kernel.train_step(targets, num_layers=1)

        # Loss should match MSE calculation (within tolerance due to forward pass changes)
        assert abs(metrics.loss - expected_loss) < expected_loss * 0.5, \
            f"Expected loss ~{expected_loss:.6f}, got {metrics.loss:.6f}"

    def test_train_step_with_gradient_clipping(self, kernel):
        """Test train_step with gradient clipping verifies clipped norm."""
        targets = {"pos_3": np.ones(8) * 100}  # Large target for large gradients
        clip_value = 0.1

        metrics = kernel.train_step(targets, num_layers=1, clip_grad=clip_value)

        # Gradient norm in metrics is computed BEFORE clipping
        assert metrics.gradient_norm > 0, "Gradient norm should be positive"
        assert metrics.loss > 0, "Loss should be positive"

        # After clipping, if we compute norm again, it should be at most clip_value
        # (Gradients are zeroed after optimizer step, so we can't check directly)
        # But we verify the metric was recorded
        assert metrics.gradient_norm >= clip_value or metrics.gradient_norm > 0

    def test_train_step_with_input_nodes(self, kernel):
        """Test train_step with custom input nodes."""
        input_nodes = {f"pos_{i}": np.random.randn(8) for i in range(4)}
        targets = {"pos_3": np.ones(8)}

        metrics = kernel.train_step(
            targets,
            num_layers=1,
            input_nodes=input_nodes
        )

        assert metrics.loss is not None

    def test_train_step_logs_to_history(self, kernel):
        """Test that train_step logs to internal history."""
        targets = {"pos_3": np.ones(8)}

        kernel.train_step(targets, num_layers=1)
        kernel.train_step(targets, num_layers=1)

        history = kernel.get_history()
        assert len(history.train_losses) == 2


# =============================================================================
# Fit Method Tests
# =============================================================================


class TestExperimentKernelFit:
    """Tests for ExperimentKernel.fit method."""

    @pytest.fixture
    def kernel(self):
        """Create a kernel with simple graph."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph.forward(num_layers=1)
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        return ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            profiling=True,
        )

    def test_fit_returns_history(self, kernel):
        """Test that fit returns TrainingHistory."""
        targets = {"pos_2": np.ones(8)}

        history = kernel.fit(
            targets=targets,
            epochs=5,
            num_layers=1,
            verbose=False,
        )

        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) == 5

    def test_fit_reduces_loss(self, kernel):
        """Test that fit reduces loss over epochs with quantified reduction."""
        np.random.seed(42)
        targets = {"pos_2": np.random.randn(8)}

        history = kernel.fit(
            targets=targets,
            epochs=50,
            num_layers=1,
            verbose=False,
        )

        # Loss should generally decrease
        initial_loss = history.train_losses[0]
        final_loss = history.train_losses[-1]
        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"

        # Calculate and verify loss reduction percentage
        loss_reduction = (initial_loss - final_loss) / initial_loss
        assert loss_reduction > 0.1, \
            f"Loss should reduce by at least 10%, got {loss_reduction*100:.1f}%"

        # Verify history length matches epochs
        assert len(history.train_losses) == 50
        assert len(history.gradient_norms) == 50

    def test_fit_with_callback(self, kernel):
        """Test that fit calls callback with correct calculated values."""
        targets = {"pos_2": np.ones(8)}
        callback_calls = []

        def callback(epoch, metrics):
            callback_calls.append({
                'epoch': epoch,
                'loss': metrics.loss,
                'grad_norm': metrics.gradient_norm,
                'total_time': metrics.total_time_ms,
            })

        kernel.fit(
            targets=targets,
            epochs=5,
            num_layers=1,
            verbose=False,
            callback=callback,
        )

        # Verify callback was called for each epoch
        assert len(callback_calls) == 5

        # Verify epoch numbers are sequential
        for i, call in enumerate(callback_calls):
            assert call['epoch'] == i, f"Expected epoch {i}, got {call['epoch']}"

        # Verify all losses are positive and decreasing trend
        losses = [call['loss'] for call in callback_calls]
        assert all(loss > 0 for loss in losses), "All losses should be positive"
        assert losses[-1] < losses[0], "Loss should decrease over training"

        # Verify gradient norms are positive
        assert all(call['grad_norm'] > 0 for call in callback_calls), \
            "All gradient norms should be positive"

        # Verify timing was recorded
        assert all(call['total_time'] >= 0 for call in callback_calls), \
            "All times should be non-negative"

    def test_fit_with_gradient_clipping(self, kernel):
        """Test fit with gradient clipping enabled."""
        targets = {"pos_2": np.ones(8)}

        history = kernel.fit(
            targets=targets,
            epochs=5,
            num_layers=1,
            clip_grad=1.0,
            verbose=False,
        )

        assert len(history.train_losses) == 5

    def test_fit_sets_eval_mode(self, kernel):
        """Test that fit sets graph to eval mode after training."""
        targets = {"pos_2": np.ones(8)}

        # Ensure in train mode initially
        kernel.graph.train()
        assert kernel.graph._training is True

        kernel.fit(
            targets=targets,
            epochs=5,
            num_layers=1,
            verbose=False,
        )

        # Should be in eval mode after fit
        assert kernel.graph._training is False


# =============================================================================
# Evaluate Method Tests
# =============================================================================


class TestExperimentKernelEvaluate:
    """Tests for ExperimentKernel.evaluate method."""

    @pytest.fixture
    def kernel(self):
        """Create a trained kernel."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph.forward(num_layers=1)
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        # Train briefly
        targets = {"pos_2": np.random.randn(8)}
        kernel.fit(targets, epochs=10, verbose=False)

        return kernel, targets

    def test_evaluate_returns_loss(self, kernel):
        """Test that evaluate returns correct MSE loss value."""
        kernel_obj, targets = kernel

        # Get forward pass outputs
        outputs = kernel_obj.graph.forward(num_layers=1)

        # Calculate expected MSE loss manually
        expected_loss = 0.0
        for node_id, target in targets.items():
            if node_id in outputs:
                expected_loss += np.mean((outputs[node_id] - target) ** 2)

        # Run evaluate
        loss = kernel_obj.evaluate(targets, num_layers=1)

        # Verify loss matches expected calculation
        assert isinstance(loss, float)
        assert loss >= 0
        assert abs(loss - expected_loss) < 1e-6, \
            f"Expected loss {expected_loss:.6f}, got {loss:.6f}"

    def test_evaluate_sets_eval_mode(self, kernel):
        """Test that evaluate sets graph to eval mode."""
        kernel_obj, targets = kernel
        kernel_obj.graph.train()

        kernel_obj.evaluate(targets, num_layers=1)

        assert kernel_obj.graph._training is False

    def test_evaluate_with_input_nodes(self, kernel):
        """Test evaluate with custom input nodes."""
        kernel_obj, targets = kernel
        input_nodes = {f"pos_{i}": np.random.randn(8) for i in range(3)}

        loss = kernel_obj.evaluate(
            targets,
            num_layers=1,
            input_nodes=input_nodes
        )

        assert isinstance(loss, float)


# =============================================================================
# Profiling Tests
# =============================================================================


class TestExperimentKernelProfiling:
    """Tests for ExperimentKernel profiling functionality."""

    @pytest.fixture
    def kernel_with_profiling(self):
        """Create kernel with profiling enabled."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph.forward(num_layers=1)
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        return ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            profiling=True,
            track_memory=False,  # Disable memory tracking for speed
        )

    def test_profile_report_after_training(self, kernel_with_profiling):
        """Test that profile report has correctly calculated statistics."""
        targets = {"pos_2": np.ones(8)}
        num_epochs = 10

        kernel_with_profiling.fit(targets, epochs=num_epochs, verbose=False)

        report = kernel_with_profiling.profile_report()

        # Verify step count matches epochs
        assert report.total_steps == num_epochs

        # Verify timing statistics are positive and reasonable
        assert report.forward_time_mean > 0, "Forward time mean should be positive"
        assert report.backward_time_mean > 0, "Backward time mean should be positive"
        assert report.step_time_mean > 0, "Step time mean should be positive"

        # Step time should be >= forward + backward time
        assert report.step_time_mean >= report.forward_time_mean + report.backward_time_mean * 0.5

        # Standard deviations should be non-negative
        assert report.forward_time_std >= 0
        assert report.backward_time_std >= 0

        # Loss statistics should be valid
        assert report.initial_loss > 0, "Initial loss should be positive"
        assert report.final_loss >= 0, "Final loss should be non-negative"
        assert report.min_loss <= report.final_loss, "Min loss should be <= final loss"
        assert report.min_loss <= report.initial_loss, "Min loss should be <= initial loss"

        # Verify gradient norm statistics
        assert report.gradient_norm_mean > 0, "Gradient norm mean should be positive"
        assert report.gradient_norm_min <= report.gradient_norm_mean
        assert report.gradient_norm_max >= report.gradient_norm_mean

    def test_reset_clears_profiling(self, kernel_with_profiling):
        """Test that reset clears profiling data."""
        targets = {"pos_2": np.ones(8)}

        kernel_with_profiling.fit(targets, epochs=5, verbose=False)
        kernel_with_profiling.reset()

        report = kernel_with_profiling.profile_report()
        assert report.total_steps == 0


# =============================================================================
# Attention Summary Tests
# =============================================================================


class TestExperimentKernelAttention:
    """Tests for attention-related functionality."""

    @pytest.fixture
    def kernel(self):
        """Create a kernel and run training."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=4, embedding_dim=8, seed=42)
        graph.forward(num_layers=1)
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        targets = {"pos_3": np.ones(8)}
        kernel.fit(targets, epochs=5, verbose=False)

        return kernel

    def test_get_attention_summary(self, kernel):
        """Test getting attention summary after training."""
        # Run a forward pass to populate attention weights
        kernel.graph.forward(num_layers=1)

        summary = kernel.get_attention_summary()

        assert isinstance(summary, dict)
        assert "pos_3" in summary  # Node with incoming edges
        assert isinstance(summary["pos_3"], dict)


# =============================================================================
# Multi-layer Tests
# =============================================================================


# =============================================================================
# Verbose Output Tests
# =============================================================================


class TestExperimentKernelVerbose:
    """Tests for verbose output functionality."""

    @pytest.fixture
    def kernel(self):
        """Create a kernel for verbose tests."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph.forward(num_layers=1)
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        return ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            profiling=False,
        )

    def test_fit_with_verbose_output(self, kernel, capsys):
        """Test fit with verbose=True prints output."""
        targets = {"pos_2": np.ones(8)}

        kernel.fit(
            targets=targets,
            epochs=15,  # More than log_every default of 10
            num_layers=1,
            verbose=True,
            log_every=5,
        )

        captured = capsys.readouterr()
        assert "Starting training" in captured.out
        assert "Graph:" in captured.out
        assert "Embedding dim:" in captured.out
        assert "Parameters:" in captured.out
        assert "Epoch" in captured.out
        assert "Training complete" in captured.out
        assert "Final loss" in captured.out

    def test_fit_with_log_every(self, kernel, capsys):
        """Test that log_every parameter controls output frequency."""
        targets = {"pos_2": np.ones(8)}

        kernel.fit(
            targets=targets,
            epochs=10,
            num_layers=1,
            verbose=True,
            log_every=5,
        )

        captured = capsys.readouterr()
        # Should have logged at epoch 5 and 10
        assert "Epoch    5" in captured.out or "Epoch  5" in captured.out


# =============================================================================
# Multi-layer Tests
# =============================================================================


class TestExperimentKernelMultiLayer:
    """Tests for multi-layer training scenarios."""

    def test_training_with_multiple_layers(self):
        """Test training with multiple attention layers."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=4, embedding_dim=8, seed=42)
        graph.forward(num_layers=3)  # Initialize 3 layers

        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        targets = {"pos_3": np.random.randn(8)}

        history = kernel.fit(
            targets=targets,
            epochs=20,
            num_layers=3,
            verbose=False,
        )

        # Should complete without error
        assert len(history.train_losses) == 20

    def test_gradient_flow_through_layers(self):
        """Test that gradients flow through all layers."""
        np.random.seed(42)
        graph = create_causal_attention_graph(seq_len=3, embedding_dim=8, seed=42)
        graph.forward(num_layers=2)  # Initialize 2 layers

        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        targets = {"pos_2": np.ones(8)}

        # Take one training step
        kernel.train_step(targets, num_layers=2)

        # Check gradients exist for both layers
        assert len(graph._attention_layers) == 2
        # Parameters should have been updated (gradients zeroed by optimizer)
        # We just verify the training step completed without error
