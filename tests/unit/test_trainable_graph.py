"""
Unit tests for TrainableGraph: gradient descent optimizable graph neural network.

Tests the core components:
- TrainableNode with learnable embeddings
- TrainableEdge with learnable weights
- Forward pass (message passing)
- Backward pass (gradient computation)
- Optimizers (SGD, Adam, AdaGrad, RMSprop)
- Loss functions (MSE, MAE, CrossEntropy, etc.)
- Learning rate schedulers
- Training utilities
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from cortical.graph import (
    # Core types
    TrainableNode,
    TrainableEdge,
    TrainableGraph,
    Parameter,
    # Activation and aggregation
    Activation,
    Aggregation,
    apply_activation,
    activation_derivative,
    aggregate_messages,
    # Loss functions
    MSELoss,
    MAELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    HuberLoss,
    ContrastiveLoss,
    # Optimizers
    SGD,
    Adam,
    AdaGrad,
    RMSprop,
    # Learning rate schedulers
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    # Training utilities
    EarlyStopping,
    TrainingHistory,
    train_step,
    fit,
)


# =============================================================================
# Activation Function Tests
# =============================================================================


class TestActivationFunctions:
    """Tests for activation functions."""

    def test_relu(self):
        """Test ReLU activation."""
        x = np.array([-2, -1, 0, 1, 2])
        result = apply_activation(x, Activation.RELU)
        expected = np.array([0, 0, 0, 1, 2])
        assert_array_almost_equal(result, expected)

    def test_leaky_relu(self):
        """Test Leaky ReLU activation."""
        x = np.array([-2, -1, 0, 1, 2])
        result = apply_activation(x, Activation.LEAKY_RELU, alpha=0.1)
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        assert_array_almost_equal(result, expected)

    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = np.array([0])
        result = apply_activation(x, Activation.SIGMOID)
        assert_array_almost_equal(result, np.array([0.5]))

        # Large positive -> 1
        x_pos = np.array([100])
        result_pos = apply_activation(x_pos, Activation.SIGMOID)
        assert result_pos[0] > 0.99

        # Large negative -> 0
        x_neg = np.array([-100])
        result_neg = apply_activation(x_neg, Activation.SIGMOID)
        assert result_neg[0] < 0.01

    def test_tanh(self):
        """Test tanh activation."""
        x = np.array([0])
        result = apply_activation(x, Activation.TANH)
        assert_array_almost_equal(result, np.array([0]))

    def test_softmax(self):
        """Test softmax activation."""
        x = np.array([1, 2, 3])
        result = apply_activation(x, Activation.SOFTMAX)
        # Should sum to 1
        assert abs(np.sum(result) - 1.0) < 1e-6
        # Larger values should have larger probabilities
        assert result[2] > result[1] > result[0]

    def test_elu(self):
        """Test ELU activation."""
        x = np.array([-2, -1, 0, 1, 2])
        result = apply_activation(x, Activation.ELU, alpha=1.0)
        # Positive values unchanged
        assert result[3] == 1
        assert result[4] == 2
        # Negative values transformed
        assert result[0] < 0
        assert result[2] == 0

    def test_none_activation(self):
        """Test identity (no activation)."""
        x = np.array([-2, -1, 0, 1, 2])
        result = apply_activation(x, Activation.NONE)
        assert_array_almost_equal(result, x)


class TestActivationDerivatives:
    """Tests for activation function derivatives."""

    def test_relu_derivative(self):
        """Test ReLU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        result = activation_derivative(x, Activation.RELU)
        expected = np.array([0, 0, 0, 1, 1])
        assert_array_almost_equal(result, expected)

    def test_sigmoid_derivative(self):
        """Test sigmoid derivative."""
        x = np.array([0])
        result = activation_derivative(x, Activation.SIGMOID)
        # sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        assert_array_almost_equal(result, np.array([0.25]))

    def test_tanh_derivative(self):
        """Test tanh derivative."""
        x = np.array([0])
        result = activation_derivative(x, Activation.TANH)
        # tanh(0) = 0, derivative = 1 - 0^2 = 1
        assert_array_almost_equal(result, np.array([1.0]))

    def test_none_derivative(self):
        """Test identity derivative (all ones)."""
        x = np.array([-2, -1, 0, 1, 2])
        result = activation_derivative(x, Activation.NONE)
        expected = np.ones(5)
        assert_array_almost_equal(result, expected)


# =============================================================================
# Aggregation Function Tests
# =============================================================================


class TestAggregationFunctions:
    """Tests for message aggregation functions."""

    def test_sum_aggregation(self):
        """Test sum aggregation."""
        messages = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        result = aggregate_messages(messages, Aggregation.SUM)
        expected = np.array([9, 12])
        assert_array_almost_equal(result, expected)

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        messages = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        result = aggregate_messages(messages, Aggregation.MEAN)
        expected = np.array([3, 4])
        assert_array_almost_equal(result, expected)

    def test_max_aggregation(self):
        """Test max aggregation."""
        messages = [np.array([1, 6]), np.array([3, 2]), np.array([5, 4])]
        result = aggregate_messages(messages, Aggregation.MAX)
        expected = np.array([5, 6])
        assert_array_almost_equal(result, expected)

    def test_min_aggregation(self):
        """Test min aggregation."""
        messages = [np.array([1, 6]), np.array([3, 2]), np.array([5, 4])]
        result = aggregate_messages(messages, Aggregation.MIN)
        expected = np.array([1, 2])
        assert_array_almost_equal(result, expected)

    def test_empty_messages(self):
        """Test aggregation with empty list."""
        result = aggregate_messages([], Aggregation.SUM)
        assert len(result) == 0


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
# TrainableNode Tests
# =============================================================================


class TestTrainableNode:
    """Tests for TrainableNode class."""

    def test_create_trainable_node(self):
        """Test creating trainable node."""
        embedding = Parameter(data=np.array([1.0, 2.0, 3.0]))
        node = TrainableNode(id="N1", embedding=embedding)

        assert node.id == "N1"
        assert node.embedding is not None
        assert len(node.embedding.data) == 3

    def test_node_hash(self):
        """Test that trainable nodes hash by ID."""
        node1 = TrainableNode(id="N1")
        node2 = TrainableNode(id="N1")

        assert hash(node1) == hash(node2)


# =============================================================================
# TrainableEdge Tests
# =============================================================================


class TestTrainableEdge:
    """Tests for TrainableEdge class."""

    def test_create_trainable_edge(self):
        """Test creating trainable edge."""
        weight_param = Parameter(data=np.array([0.5]))
        edge = TrainableEdge(
            source_id="A",
            target_id="B",
            weight_param=weight_param,
        )

        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.weight_param is not None

    def test_effective_weight(self):
        """Test effective weight property."""
        weight_param = Parameter(data=np.array([0.7]))
        edge = TrainableEdge(
            source_id="A",
            target_id="B",
            weight_param=weight_param,
        )

        assert abs(edge.effective_weight - 0.7) < 0.01

    def test_effective_weight_clipping(self):
        """Test that effective weight is clipped to [0, 1]."""
        # Weight > 1
        weight_param = Parameter(data=np.array([1.5]))
        edge = TrainableEdge(source_id="A", target_id="B", weight_param=weight_param)
        assert edge.effective_weight == 1.0

        # Weight < 0
        weight_param = Parameter(data=np.array([-0.5]))
        edge = TrainableEdge(source_id="A", target_id="B", weight_param=weight_param)
        assert edge.effective_weight == 0.0


# =============================================================================
# TrainableGraph Tests
# =============================================================================


class TestTrainableGraph:
    """Tests for TrainableGraph class."""

    @pytest.fixture
    def graph(self):
        """Create fresh trainable graph for each test."""
        return TrainableGraph(embedding_dim=4, seed=42)

    def test_create_graph(self, graph):
        """Test basic graph creation."""
        assert graph.embedding_dim == 4
        assert graph.activation == Activation.RELU
        assert graph.aggregation == Aggregation.SUM

    def test_add_node_with_embedding(self, graph):
        """Test adding node with custom embedding."""
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        node = graph.add_node("A", embedding=embedding)

        assert node.id == "A"
        assert node.embedding is not None
        assert_array_almost_equal(node.embedding.data, embedding)

    def test_add_node_auto_embedding(self, graph):
        """Test that node gets auto-initialized embedding."""
        node = graph.add_node("A")

        assert node.embedding is not None
        assert len(node.embedding.data) == 4

    def test_add_edge_with_weight(self, graph):
        """Test adding edge with learnable weight."""
        graph.add_node("A")
        graph.add_node("B")
        edge = graph.add_edge("A", "B", weight=0.8)

        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.weight_param is not None
        assert abs(edge.weight_param.data[0] - 0.8) < 0.01

    def test_parameters(self, graph):
        """Test getting all learnable parameters."""
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        params = graph.parameters()
        # 2 node embeddings + 1 edge weight
        assert len(params) >= 3

    def test_train_eval_mode(self, graph):
        """Test switching between train and eval modes."""
        graph.train()
        assert graph._training is True

        graph.eval()
        assert graph._training is False

    def test_forward_basic(self, graph):
        """Test basic forward pass."""
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B", weight=1.0)

        outputs = graph.forward(num_layers=1)

        assert "A" in outputs
        assert "B" in outputs
        assert len(outputs["A"]) == 4
        assert len(outputs["B"]) == 4

    def test_forward_multi_layer(self, graph):
        """Test multi-layer message passing."""
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_node("C", embedding=np.ones(4))
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        outputs = graph.forward(num_layers=2)

        assert "A" in outputs
        assert "B" in outputs
        assert "C" in outputs

    def test_forward_with_input_override(self, graph):
        """Test forward pass with input override."""
        graph.add_node("A", embedding=np.zeros(4))
        graph.add_node("B", embedding=np.zeros(4))
        graph.add_edge("A", "B")

        custom_input = {"A": np.ones(4)}
        outputs = graph.forward(num_layers=1, input_nodes=custom_input)

        # B should receive message from overridden A input
        assert "B" in outputs

    def test_backward_basic(self, graph):
        """Test basic backward pass."""
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B", weight=1.0)

        # Forward
        outputs = graph.forward(num_layers=1)

        # Backward with gradient for B
        output_grads = {"B": np.ones(4) * 0.1}
        graph.backward(output_grads, num_layers=1)

        # Check that gradients were computed
        params = graph.parameters()
        has_gradient = any(p.grad is not None for p in params)
        assert has_gradient

    def test_zero_grad(self, graph):
        """Test zeroing gradients."""
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B")

        # Forward and backward to create gradients
        outputs = graph.forward(num_layers=1)
        graph.backward({"B": np.ones(4)}, num_layers=1)

        # Zero gradients
        graph.zero_grad()

        # All gradients should be None
        for param in graph.parameters():
            assert param.grad is None

    def test_get_embeddings(self, graph):
        """Test getting all embeddings."""
        graph.add_node("A", embedding=np.array([1.0, 2.0, 3.0, 4.0]))
        graph.add_node("B", embedding=np.array([5.0, 6.0, 7.0, 8.0]))

        embeddings = graph.get_embeddings()

        assert "A" in embeddings
        assert "B" in embeddings
        assert_array_almost_equal(embeddings["A"], np.array([1.0, 2.0, 3.0, 4.0]))

    def test_set_embedding(self, graph):
        """Test setting node embedding."""
        graph.add_node("A", embedding=np.zeros(4))

        new_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        graph.set_embedding("A", new_embedding)

        node = graph.get_node("A")
        assert_array_almost_equal(node.embedding.data, new_embedding)

    def test_clip_gradients(self, graph):
        """Test gradient clipping."""
        graph.add_node("A", embedding=np.ones(4))
        graph.add_node("B", embedding=np.ones(4))
        graph.add_edge("A", "B")

        outputs = graph.forward(num_layers=1)
        graph.backward({"B": np.ones(4) * 100}, num_layers=1)

        # Clip gradients
        norm_before = graph.clip_gradients(max_norm=1.0)

        # Check that norm was larger before clipping
        assert norm_before > 1.0

        # Check that gradients are now bounded
        total_norm = 0.0
        for param in graph.parameters():
            if param.grad is not None:
                total_norm += np.sum(param.grad**2)
        total_norm = np.sqrt(total_norm)
        assert total_norm <= 1.0 + 1e-6

    def test_save_load_state(self, graph):
        """Test saving and loading graph state."""
        graph.add_node("A", embedding=np.array([1.0, 2.0, 3.0, 4.0]))
        graph.add_node("B", embedding=np.array([5.0, 6.0, 7.0, 8.0]))
        graph.add_edge("A", "B", weight=0.7)

        # Save state
        state = graph.save_state()

        # Modify graph
        graph.set_embedding("A", np.zeros(4))

        # Load state
        graph.load_state(state)

        # Check restoration
        assert_array_almost_equal(
            graph.get_node("A").embedding.data,
            np.array([1.0, 2.0, 3.0, 4.0])
        )


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestLossFunctions:
    """Tests for loss functions."""

    def test_mse_loss(self):
        """Test MSE loss."""
        loss_fn = MSELoss()
        predicted = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])

        loss = loss_fn(predicted, target)
        assert loss == 0.0

        # Test with difference
        predicted = np.array([2.0, 2.0, 2.0])
        target = np.array([1.0, 2.0, 3.0])
        loss = loss_fn(predicted, target)
        assert loss > 0

    def test_mse_gradient(self):
        """Test MSE loss gradient."""
        loss_fn = MSELoss()
        predicted = np.array([2.0, 2.0, 2.0])
        target = np.array([1.0, 2.0, 3.0])

        grad = loss_fn.gradient(predicted, target)
        # Gradient should point away from target
        assert grad[0] > 0  # predicted > target
        assert grad[1] == 0  # predicted == target
        assert grad[2] < 0  # predicted < target

    def test_mae_loss(self):
        """Test MAE loss."""
        loss_fn = MAELoss()
        predicted = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])

        loss = loss_fn(predicted, target)
        assert loss == 0.0

    def test_cross_entropy_loss(self):
        """Test cross-entropy loss."""
        loss_fn = CrossEntropyLoss()
        predicted = np.array([0.9, 0.05, 0.05])
        target = np.array([1.0, 0.0, 0.0])

        loss = loss_fn(predicted, target)
        assert loss > 0

    def test_binary_cross_entropy_loss(self):
        """Test binary cross-entropy loss."""
        loss_fn = BinaryCrossEntropyLoss()
        predicted = np.array([0.9])
        target = np.array([1.0])

        loss = loss_fn(predicted, target)
        assert loss > 0
        assert loss < 0.2  # Should be small when prediction is close

    def test_huber_loss(self):
        """Test Huber loss."""
        loss_fn = HuberLoss(delta=1.0)
        predicted = np.array([0.0])
        target = np.array([0.5])

        loss = loss_fn(predicted, target)
        assert loss > 0

    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss(margin=1.0)

        # Similar pair (target=1) - should minimize distance
        predicted = np.array([0.5])
        target = np.array([1.0])
        loss = loss_fn(predicted, target)
        assert loss > 0


# =============================================================================
# Optimizer Tests
# =============================================================================


class TestOptimizers:
    """Tests for optimizers."""

    @pytest.fixture
    def params(self):
        """Create test parameters."""
        return [
            Parameter(data=np.array([1.0, 2.0]), name="p1"),
            Parameter(data=np.array([3.0, 4.0]), name="p2"),
        ]

    def test_sgd_basic(self, params):
        """Test basic SGD update."""
        optimizer = SGD(params, lr=0.1)

        # Set gradients
        params[0].add_grad(np.array([1.0, 1.0]))
        params[1].add_grad(np.array([1.0, 1.0]))

        optimizer.step()

        # Values should decrease by lr * grad
        assert_array_almost_equal(params[0].data, np.array([0.9, 1.9]))
        assert_array_almost_equal(params[1].data, np.array([2.9, 3.9]))

    def test_sgd_momentum(self, params):
        """Test SGD with momentum."""
        optimizer = SGD(params, lr=0.1, momentum=0.9)

        # First step
        params[0].add_grad(np.array([1.0, 1.0]))
        optimizer.step()
        first_update = params[0].data.copy()

        # Zero grad and second step
        optimizer.zero_grad()
        params[0].add_grad(np.array([1.0, 1.0]))
        optimizer.step()

        # With momentum, second update should be larger
        second_diff = params[0].data - first_update
        assert np.abs(second_diff[0]) > 0.1

    def test_sgd_weight_decay(self, params):
        """Test SGD with weight decay."""
        optimizer = SGD(params, lr=0.1, weight_decay=0.01)

        initial_norm = np.sum(params[0].data**2)

        params[0].add_grad(np.array([0.0, 0.0]))
        optimizer.step()

        # Weight decay should reduce parameter magnitude
        new_norm = np.sum(params[0].data**2)
        assert new_norm < initial_norm

    def test_adam_basic(self, params):
        """Test basic Adam update."""
        optimizer = Adam(params, lr=0.1)

        params[0].add_grad(np.array([1.0, 1.0]))
        optimizer.step()

        # Parameters should be updated
        assert not np.allclose(params[0].data, np.array([1.0, 2.0]))

    def test_adam_adapts_learning_rate(self, params):
        """Test that Adam adapts learning rate per parameter."""
        optimizer = Adam(params, lr=0.1)

        # Give different gradients to different parameters
        params[0].add_grad(np.array([10.0, 10.0]))  # Large gradient
        params[1].add_grad(np.array([0.01, 0.01]))  # Small gradient

        optimizer.step()

        # Both should be updated, but proportionally different
        update1 = np.abs(params[0].data - np.array([1.0, 2.0]))
        update2 = np.abs(params[1].data - np.array([3.0, 4.0]))

        # Large gradient should not cause proportionally larger update due to adaptation
        assert update1[0] > 0
        assert update2[0] > 0

    def test_adagrad(self, params):
        """Test AdaGrad optimizer."""
        optimizer = AdaGrad(params, lr=0.1)

        params[0].add_grad(np.array([1.0, 1.0]))
        optimizer.step()

        assert not np.allclose(params[0].data, np.array([1.0, 2.0]))

    def test_rmsprop(self, params):
        """Test RMSprop optimizer."""
        optimizer = RMSprop(params, lr=0.1)

        params[0].add_grad(np.array([1.0, 1.0]))
        optimizer.step()

        assert not np.allclose(params[0].data, np.array([1.0, 2.0]))

    def test_zero_grad(self, params):
        """Test optimizer zero_grad."""
        optimizer = SGD(params, lr=0.1)

        params[0].add_grad(np.array([1.0, 1.0]))
        params[1].add_grad(np.array([1.0, 1.0]))

        optimizer.zero_grad()

        assert params[0].grad is None
        assert params[1].grad is None


# =============================================================================
# Learning Rate Scheduler Tests
# =============================================================================


class TestLRSchedulers:
    """Tests for learning rate schedulers."""

    @pytest.fixture
    def optimizer(self):
        """Create test optimizer."""
        params = [Parameter(data=np.array([1.0]))]
        return SGD(params, lr=0.1)

    def test_step_lr(self, optimizer):
        """Test StepLR scheduler."""
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        initial_lr = optimizer.lr
        scheduler.step()  # Epoch 1
        assert optimizer.lr == initial_lr

        scheduler.step()  # Epoch 2 - should reduce
        assert optimizer.lr == initial_lr * 0.5

    def test_exponential_lr(self, optimizer):
        """Test ExponentialLR scheduler."""
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        initial_lr = optimizer.lr
        scheduler.step()
        assert abs(optimizer.lr - initial_lr * 0.9) < 1e-6

    def test_cosine_annealing_lr(self, optimizer):
        """Test CosineAnnealingLR scheduler."""
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(optimizer.lr)

        # Should decrease then increase (cosine pattern)
        assert lrs[0] > lrs[4]  # First half decreasing
        assert lrs[-1] >= 0.01  # Ends at eta_min

    def test_reduce_lr_on_plateau(self, optimizer):
        """Test ReduceLROnPlateau scheduler."""
        scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        initial_lr = optimizer.lr

        # Improving loss - no reduction
        scheduler.step(loss=1.0)
        scheduler.step(loss=0.9)
        assert optimizer.lr == initial_lr

        # Stagnant loss - should reduce after patience
        scheduler.step(loss=0.9)
        scheduler.step(loss=0.9)
        scheduler.step(loss=0.9)  # Patience exceeded
        assert optimizer.lr < initial_lr


# =============================================================================
# Training Utility Tests
# =============================================================================


class TestTrainingUtilities:
    """Tests for training utilities."""

    def test_early_stopping(self):
        """Test early stopping callback."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        # Improving loss
        assert early_stopping(0.5) is False
        assert early_stopping(0.4) is False
        assert early_stopping(0.3) is False

        # Stagnant loss
        assert early_stopping(0.3) is False
        assert early_stopping(0.3) is False
        assert early_stopping(0.3) is True  # Should stop

    def test_early_stopping_restore(self):
        """Test early stopping with state restoration."""
        graph = TrainableGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.array([1.0, 2.0, 3.0, 4.0]))

        early_stopping = EarlyStopping(patience=2, restore_best=True)

        # Record best state
        early_stopping(0.5, graph)

        # Modify graph
        graph.set_embedding("A", np.zeros(4))

        # Trigger early stopping
        early_stopping(0.6, graph)
        early_stopping(0.7, graph)
        early_stopping(0.8, graph)

        # Restore
        early_stopping.restore(graph)

        # Check restoration
        assert_array_almost_equal(
            graph.get_node("A").embedding.data,
            np.array([1.0, 2.0, 3.0, 4.0])
        )

    def test_training_history(self):
        """Test training history logging."""
        history = TrainingHistory()

        history.log(train_loss=0.5, val_loss=0.6, lr=0.01, accuracy=0.8)
        history.log(train_loss=0.4, val_loss=0.5, lr=0.01, accuracy=0.85)
        history.log(train_loss=0.3, val_loss=0.4, lr=0.01, accuracy=0.9)

        assert len(history.train_losses) == 3
        assert len(history.val_losses) == 3
        assert history.metrics["accuracy"] == [0.8, 0.85, 0.9]

        best_epoch = history.get_best_epoch("val_loss")
        assert best_epoch == 2


class TestTrainStep:
    """Tests for train_step function."""

    def test_train_step_reduces_loss(self):
        """Test that train_step reduces loss."""
        graph = TrainableGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.random.randn(4))
        graph.add_node("B", embedding=np.random.randn(4))
        graph.add_edge("A", "B")

        optimizer = Adam(graph.parameters(), lr=0.1)
        loss_fn = MSELoss()
        target = np.zeros(4)
        targets = {"B": target}

        # Compute initial loss
        initial_outputs = graph.forward(num_layers=1)
        initial_loss = loss_fn(initial_outputs["B"], target)

        # Train for several steps
        for _ in range(10):
            train_step(graph, optimizer, loss_fn, targets, num_layers=1)

        # Compute final loss
        final_outputs = graph.forward(num_layers=1)
        final_loss = loss_fn(final_outputs["B"], target)

        # Loss should decrease
        assert final_loss < initial_loss


class TestFit:
    """Tests for fit function."""

    def test_fit_basic(self):
        """Test basic fit function."""
        graph = TrainableGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.random.randn(4))
        graph.add_node("B", embedding=np.random.randn(4))
        graph.add_edge("A", "B")

        optimizer = Adam(graph.parameters(), lr=0.1)
        loss_fn = MSELoss()
        targets = {"B": np.zeros(4)}

        history = fit(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_targets=targets,
            epochs=10,
            num_layers=1,
            verbose=False,
        )

        assert len(history.train_losses) == 10
        # Loss should generally decrease
        assert history.train_losses[-1] < history.train_losses[0]

    def test_fit_with_validation(self):
        """Test fit with validation data."""
        graph = TrainableGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.random.randn(4))
        graph.add_node("B", embedding=np.random.randn(4))
        graph.add_edge("A", "B")

        optimizer = Adam(graph.parameters(), lr=0.1)
        loss_fn = MSELoss()
        train_targets = {"B": np.zeros(4)}
        val_targets = {"B": np.zeros(4)}

        history = fit(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_targets=train_targets,
            val_targets=val_targets,
            epochs=10,
            num_layers=1,
            verbose=False,
        )

        assert len(history.val_losses) == 10

    def test_fit_with_early_stopping(self):
        """Test fit with early stopping."""
        graph = TrainableGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.zeros(4))  # Start at target
        graph.add_node("B", embedding=np.zeros(4))
        graph.add_edge("A", "B")

        optimizer = Adam(graph.parameters(), lr=0.001)  # Very small lr
        loss_fn = MSELoss()
        targets = {"B": np.zeros(4)}

        early_stopping = EarlyStopping(patience=3)

        history = fit(
            graph=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_targets=targets,
            epochs=100,
            num_layers=1,
            early_stopping=early_stopping,
            verbose=False,
        )

        # Should stop early due to no improvement
        assert len(history.train_losses) < 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainableGraphIntegration:
    """Integration tests for TrainableGraph."""

    def test_learn_identity(self):
        """Test learning identity function through graph."""
        np.random.seed(42)

        graph = TrainableGraph(
            embedding_dim=4,
            activation=Activation.NONE,  # Linear for easier learning
            seed=42,
        )

        # Create simple two-node graph
        graph.add_node("input", embedding=np.array([1.0, 2.0, 3.0, 4.0]))
        graph.add_node("output", embedding=np.zeros(4))
        graph.add_edge("input", "output", weight=1.0)

        # Target: output should match input
        target = np.array([1.0, 2.0, 3.0, 4.0])

        optimizer = Adam(graph.parameters(), lr=0.05)
        loss_fn = MSELoss()

        # Train
        for _ in range(100):
            outputs = graph.forward(num_layers=1)
            loss = loss_fn(outputs["output"], target)
            graph.backward({"output": loss_fn.gradient(outputs["output"], target)})
            optimizer.step()
            optimizer.zero_grad()

        # Final output should be close to target
        final_output = graph.forward(num_layers=1)["output"]
        final_loss = loss_fn(final_output, target)
        assert final_loss < 0.5  # Should have learned something

    def test_graph_with_multiple_paths(self):
        """Test graph with multiple paths between nodes."""
        graph = TrainableGraph(embedding_dim=4, seed=42)

        # Diamond shape: A -> B, A -> C, B -> D, C -> D
        graph.add_node("A", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        graph.add_node("B", embedding=np.zeros(4))
        graph.add_node("C", embedding=np.zeros(4))
        graph.add_node("D", embedding=np.zeros(4))

        graph.add_edge("A", "B", weight=0.5)
        graph.add_edge("A", "C", weight=0.5)
        graph.add_edge("B", "D", weight=0.5)
        graph.add_edge("C", "D", weight=0.5)

        # Forward should aggregate messages from both paths
        outputs = graph.forward(num_layers=2)

        assert "D" in outputs
        # D should have received information from A via both B and C

    def test_save_load_round_trip(self):
        """Test that save/load preserves training progress."""
        graph = TrainableGraph(embedding_dim=4, seed=42)
        graph.add_node("A", embedding=np.array([1.0, 2.0, 3.0, 4.0]))
        graph.add_node("B", embedding=np.array([5.0, 6.0, 7.0, 8.0]))
        graph.add_edge("A", "B", weight=0.7)

        # Initialize transform parameters
        graph.forward(num_layers=1)

        # Save state
        state = graph.save_state()

        # Create new graph and load
        graph2 = TrainableGraph(embedding_dim=4, seed=42)
        graph2.add_node("A", embedding=np.zeros(4))
        graph2.add_node("B", embedding=np.zeros(4))
        graph2.add_edge("A", "B", weight=0.5)

        # Initialize transform parameters
        graph2.forward(num_layers=1)

        # Load state
        graph2.load_state(state)

        # Check embeddings match
        assert_array_almost_equal(
            graph.get_embeddings()["A"],
            graph2.get_embeddings()["A"]
        )
