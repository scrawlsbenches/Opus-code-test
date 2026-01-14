"""
Trainable Graph: Graph neural network with gradient descent optimization.

This module provides a trainable graph implementation that supports:
- Learnable node embeddings
- Learnable edge weights
- Forward pass (message passing)
- Backward pass (backpropagation)
- Various optimizers (SGD, Adam)
- Loss functions (MSE, CrossEntropy, etc.)

The TrainableGraph follows the message-passing neural network (MPNN) paradigm:
1. Message: Compute messages from neighboring nodes
2. Aggregate: Combine messages (sum, mean, max)
3. Update: Update node embeddings based on aggregated messages

Example:
    from cortical.graph.trainable import TrainableGraph, Adam, MSELoss

    # Create graph with 4-dimensional node embeddings
    graph = TrainableGraph(embedding_dim=4)

    # Add nodes with initial embeddings
    graph.add_node("A", embedding=[0.1, 0.2, 0.3, 0.4])
    graph.add_node("B", embedding=[0.5, 0.6, 0.7, 0.8])

    # Add trainable edge
    graph.add_edge("A", "B", weight=0.5)

    # Training loop
    optimizer = Adam(graph.parameters(), lr=0.01)
    loss_fn = MSELoss()

    for epoch in range(100):
        # Forward pass
        outputs = graph.forward()

        # Compute loss
        loss = loss_fn(outputs["B"], target)

        # Backward pass
        graph.backward(loss)

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

See docs/trainable-graph-design.md for architecture details.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from .base import BaseGraph
from .protocols import NodeBase, EdgeBase
from .storage import InMemoryGraphStorage


# Type aliases
Array = NDArray[np.float64]


# =============================================================================
# Activation Functions
# =============================================================================


class Activation(Enum):
    """Supported activation functions."""
    NONE = "none"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    ELU = "elu"
    GELU = "gelu"


def apply_activation(x: Array, activation: Activation, alpha: float = 0.01) -> Array:
    """
    Apply activation function to array.

    Args:
        x: Input array
        activation: Activation function to apply
        alpha: Parameter for leaky_relu and elu

    Returns:
        Activated array
    """
    if activation == Activation.NONE:
        return x
    elif activation == Activation.RELU:
        return np.maximum(0, x)
    elif activation == Activation.LEAKY_RELU:
        return np.where(x > 0, x, alpha * x)
    elif activation == Activation.SIGMOID:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation == Activation.TANH:
        return np.tanh(x)
    elif activation == Activation.SOFTMAX:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    elif activation == Activation.ELU:
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    elif activation == Activation.GELU:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    else:
        return x


def activation_derivative(
    x: Array,
    activation: Activation,
    alpha: float = 0.01
) -> Array:
    """
    Compute derivative of activation function.

    Args:
        x: Input array (pre-activation values)
        activation: Activation function
        alpha: Parameter for leaky_relu and elu

    Returns:
        Derivative array
    """
    if activation == Activation.NONE:
        return np.ones_like(x)
    elif activation == Activation.RELU:
        return np.where(x > 0, 1.0, 0.0)
    elif activation == Activation.LEAKY_RELU:
        return np.where(x > 0, 1.0, alpha)
    elif activation == Activation.SIGMOID:
        s = apply_activation(x, Activation.SIGMOID)
        return s * (1 - s)
    elif activation == Activation.TANH:
        return 1 - np.tanh(x)**2
    elif activation == Activation.ELU:
        return np.where(x > 0, 1.0, apply_activation(x, Activation.ELU, alpha) + alpha)
    elif activation == Activation.GELU:
        # Approximate derivative
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf
    else:
        return np.ones_like(x)


# =============================================================================
# Aggregation Functions
# =============================================================================


class Aggregation(Enum):
    """Message aggregation methods."""
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


def aggregate_messages(
    messages: List[Array],
    aggregation: Aggregation
) -> Array:
    """
    Aggregate multiple messages into a single representation.

    Args:
        messages: List of message arrays
        aggregation: Aggregation method

    Returns:
        Aggregated message array
    """
    if not messages:
        return np.array([])

    stacked = np.stack(messages)

    if aggregation == Aggregation.SUM:
        return np.sum(stacked, axis=0)
    elif aggregation == Aggregation.MEAN:
        return np.mean(stacked, axis=0)
    elif aggregation == Aggregation.MAX:
        return np.max(stacked, axis=0)
    elif aggregation == Aggregation.MIN:
        return np.min(stacked, axis=0)
    else:
        return np.sum(stacked, axis=0)


# =============================================================================
# Parameter Container
# =============================================================================


@dataclass
class Parameter:
    """
    A learnable parameter with gradient tracking.

    Attributes:
        data: The parameter values
        grad: Accumulated gradients (None until backward called)
        requires_grad: Whether this parameter should be trained
        name: Optional name for debugging
    """
    data: Array
    grad: Optional[Array] = None
    requires_grad: bool = True
    name: str = ""

    def zero_grad(self) -> None:
        """Reset gradients to zero."""
        self.grad = None

    def add_grad(self, grad: Array) -> None:
        """Add gradient (for accumulation from multiple paths)."""
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the parameter."""
        return self.data.shape


# =============================================================================
# Trainable Node
# =============================================================================


@dataclass
class TrainableNode(NodeBase):
    """
    Node with learnable embedding vector.

    The embedding is a dense vector representation that can be
    trained via gradient descent. During forward pass, embeddings
    are transformed and aggregated with neighbor information.

    Attributes:
        embedding: Learnable embedding parameter
        output: Computed output after forward pass
        pre_activation: Value before activation (for backprop)
        incoming_messages: Messages from neighbors (for backprop)
        layer_inputs: Input values at each layer (for multi-layer backprop)
    """
    embedding: Optional[Parameter] = None
    output: Optional[Array] = None
    pre_activation: Optional[Array] = None
    incoming_messages: List[Tuple[str, Array, float]] = field(default_factory=list)
    layer_inputs: List[Array] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class TrainableEdge(EdgeBase):
    """
    Edge with learnable weight parameter.

    The edge weight modulates message passing between nodes.
    During training, edge weights are updated to learn the
    optimal connectivity structure.

    Attributes:
        weight_param: Learnable weight parameter (scalar or vector)
        transform: Optional learnable transformation matrix
    """
    weight_param: Optional[Parameter] = None
    transform: Optional[Parameter] = None

    def __post_init__(self) -> None:
        """Initialize with valid weight."""
        # Don't validate weight in [0,1] for trainable edges
        # as they need to be unconstrained for gradient descent
        pass

    @property
    def effective_weight(self) -> float:
        """Get effective weight (from parameter if available)."""
        if self.weight_param is not None:
            return float(np.clip(self.weight_param.data[0], 0, 1))
        return self.weight


# =============================================================================
# Loss Functions
# =============================================================================


class LossFunction(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def __call__(self, predicted: Array, target: Array) -> float:
        """Compute loss value."""
        ...

    @abstractmethod
    def gradient(self, predicted: Array, target: Array) -> Array:
        """Compute gradient of loss with respect to predicted."""
        ...


class MSELoss(LossFunction):
    """Mean Squared Error loss."""

    def __call__(self, predicted: Array, target: Array) -> float:
        return float(np.mean((predicted - target) ** 2))

    def gradient(self, predicted: Array, target: Array) -> Array:
        return 2 * (predicted - target) / predicted.size


class MAELoss(LossFunction):
    """Mean Absolute Error loss."""

    def __call__(self, predicted: Array, target: Array) -> float:
        return float(np.mean(np.abs(predicted - target)))

    def gradient(self, predicted: Array, target: Array) -> Array:
        return np.sign(predicted - target) / predicted.size


class CrossEntropyLoss(LossFunction):
    """Cross-entropy loss for classification."""

    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon

    def __call__(self, predicted: Array, target: Array) -> float:
        # Clip to avoid log(0)
        p = np.clip(predicted, self.epsilon, 1 - self.epsilon)
        return float(-np.sum(target * np.log(p)))

    def gradient(self, predicted: Array, target: Array) -> Array:
        p = np.clip(predicted, self.epsilon, 1 - self.epsilon)
        return -target / p


class BinaryCrossEntropyLoss(LossFunction):
    """Binary cross-entropy loss."""

    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon

    def __call__(self, predicted: Array, target: Array) -> float:
        p = np.clip(predicted, self.epsilon, 1 - self.epsilon)
        return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))

    def gradient(self, predicted: Array, target: Array) -> Array:
        p = np.clip(predicted, self.epsilon, 1 - self.epsilon)
        return (p - target) / (p * (1 - p)) / predicted.size


class HuberLoss(LossFunction):
    """Huber loss (smooth L1)."""

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, predicted: Array, target: Array) -> float:
        error = predicted - target
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        return float(np.mean(0.5 * quadratic**2 + self.delta * linear))

    def gradient(self, predicted: Array, target: Array) -> Array:
        error = predicted - target
        return np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error)
        ) / predicted.size


class ContrastiveLoss(LossFunction):
    """Contrastive loss for similarity learning."""

    def __init__(self, margin: float = 1.0):
        self.margin = margin

    def __call__(self, predicted: Array, target: Array) -> float:
        # predicted: pairwise distances, target: 1 for similar, 0 for dissimilar
        pos_loss = target * predicted**2
        neg_loss = (1 - target) * np.maximum(0, self.margin - predicted)**2
        return float(0.5 * np.mean(pos_loss + neg_loss))

    def gradient(self, predicted: Array, target: Array) -> Array:
        pos_grad = target * predicted
        neg_grad = (1 - target) * np.where(
            predicted < self.margin,
            -(self.margin - predicted),
            0
        )
        return (pos_grad + neg_grad) / predicted.size


# =============================================================================
# Optimizers
# =============================================================================


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(self, parameters: List[Parameter], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        """Update parameters based on gradients."""
        ...

    def zero_grad(self) -> None:
        """Reset all parameter gradients."""
        for param in self.parameters:
            param.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """
        Get optimizer state for checkpointing.

        Returns:
            Dict containing optimizer hyperparameters and internal state
        """
        return {"lr": self.lr}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load optimizer state from checkpoint.

        Args:
            state: State dict from state_dict()
        """
        self.lr = state.get("lr", self.lr)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (0 = no momentum)
        weight_decay: L2 regularization factor
        nesterov: Whether to use Nesterov momentum
    """

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities: Dict[int, Array] = {}

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad

            # L2 regularization
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Momentum
            if self.momentum > 0:
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(param.data)

                v = self.velocities[i]
                v[:] = self.momentum * v + grad

                if self.nesterov:
                    grad = grad + self.momentum * v
                else:
                    grad = v

            param.data -= self.lr * grad

    def state_dict(self) -> Dict[str, Any]:
        """Get SGD state for checkpointing."""
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
            "velocities": {k: v.copy() for k, v in self.velocities.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load SGD state from checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.nesterov = state.get("nesterov", self.nesterov)
        self.velocities = {k: v.copy() for k, v in state.get("velocities", {}).items()}


class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        betas: Coefficients for running averages (beta1, beta2)
        eps: Term for numerical stability
        weight_decay: L2 regularization factor
        amsgrad: Whether to use AMSGrad variant
    """

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.m: Dict[int, Array] = {}  # First moment
        self.v: Dict[int, Array] = {}  # Second moment
        self.v_max: Dict[int, Array] = {}  # Max second moment (amsgrad)
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for i, param in enumerate(self.parameters):
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad

            # L2 regularization
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Initialize moments
            if i not in self.m:
                self.m[i] = np.zeros_like(param.data)
                self.v[i] = np.zeros_like(param.data)
                if self.amsgrad:
                    self.v_max[i] = np.zeros_like(param.data)

            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            if self.amsgrad:
                self.v_max[i] = np.maximum(self.v_max[i], v_hat)
                v_hat = self.v_max[i]

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> Dict[str, Any]:
        """Get Adam state for checkpointing."""
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
            "t": self.t,
            "m": {k: v.copy() for k, v in self.m.items()},
            "v": {k: v.copy() for k, v in self.v.items()},
            "v_max": {k: v.copy() for k, v in self.v_max.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load Adam state from checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.amsgrad = state.get("amsgrad", self.amsgrad)
        self.t = state.get("t", self.t)
        self.m = {k: v.copy() for k, v in state.get("m", {}).items()}
        self.v = {k: v.copy() for k, v in state.get("v", {}).items()}
        self.v_max = {k: v.copy() for k, v in state.get("v_max", {}).items()}


class AdaGrad(Optimizer):
    """AdaGrad optimizer with per-parameter learning rates."""

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.01,
        eps: float = 1e-10,
    ):
        super().__init__(parameters, lr)
        self.eps = eps
        self.sum_sq_grads: Dict[int, Array] = {}

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad

            if i not in self.sum_sq_grads:
                self.sum_sq_grads[i] = np.zeros_like(param.data)

            self.sum_sq_grads[i] += grad**2
            param.data -= self.lr * grad / (np.sqrt(self.sum_sq_grads[i]) + self.eps)

    def state_dict(self) -> Dict[str, Any]:
        """Get AdaGrad state for checkpointing."""
        return {
            "lr": self.lr,
            "eps": self.eps,
            "sum_sq_grads": {k: v.copy() for k, v in self.sum_sq_grads.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load AdaGrad state from checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.eps = state.get("eps", self.eps)
        self.sum_sq_grads = {k: v.copy() for k, v in state.get("sum_sq_grads", {}).items()}


class RMSprop(Optimizer):
    """RMSprop optimizer with adaptive learning rates."""

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.avg_sq_grads: Dict[int, Array] = {}
        self.velocities: Dict[int, Array] = {}

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad

            if i not in self.avg_sq_grads:
                self.avg_sq_grads[i] = np.zeros_like(param.data)
                if self.momentum > 0:
                    self.velocities[i] = np.zeros_like(param.data)

            self.avg_sq_grads[i] = (
                self.alpha * self.avg_sq_grads[i] + (1 - self.alpha) * grad**2
            )

            update = self.lr * grad / (np.sqrt(self.avg_sq_grads[i]) + self.eps)

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + update
                update = self.velocities[i]

            param.data -= update

    def state_dict(self) -> Dict[str, Any]:
        """Get RMSprop state for checkpointing."""
        return {
            "lr": self.lr,
            "alpha": self.alpha,
            "eps": self.eps,
            "momentum": self.momentum,
            "avg_sq_grads": {k: v.copy() for k, v in self.avg_sq_grads.items()},
            "velocities": {k: v.copy() for k, v in self.velocities.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load RMSprop state from checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.alpha = state.get("alpha", self.alpha)
        self.eps = state.get("eps", self.eps)
        self.momentum = state.get("momentum", self.momentum)
        self.avg_sq_grads = {k: v.copy() for k, v in state.get("avg_sq_grads", {}).items()}
        self.velocities = {k: v.copy() for k, v in state.get("velocities", {}).items()}


# =============================================================================
# Learning Rate Schedulers
# =============================================================================


class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    @abstractmethod
    def step(self) -> None:
        """Update learning rate."""
        ...

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


class StepLR(LRScheduler):
    """Step decay learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 10,
        gamma: float = 0.1,
    ):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    """Exponential decay learning rate scheduler."""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self.step_count)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.step_count / self.T_max)
        ) / 2


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when metric stops improving."""

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
    ):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.wait = 0

    def step(self, loss: Optional[float] = None) -> None:  # type: ignore
        if loss is None:
            return

        self.step_count += 1

        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.wait = 0


# =============================================================================
# Trainable Graph
# =============================================================================


class TrainableGraph(BaseGraph[TrainableNode, TrainableEdge]):
    """
    A graph neural network that can be trained via gradient descent.

    Implements message-passing neural network (MPNN) operations:
    1. Message: Compute messages along edges
    2. Aggregate: Combine incoming messages
    3. Update: Update node embeddings

    Features:
    - Learnable node embeddings
    - Learnable edge weights
    - Configurable activation functions
    - Configurable aggregation methods
    - Full gradient computation for backpropagation
    - Multiple message passing layers

    Example:
        graph = TrainableGraph(embedding_dim=32)

        # Add nodes
        graph.add_node("A", embedding=np.random.randn(32))
        graph.add_node("B", embedding=np.random.randn(32))

        # Add edges
        graph.add_edge("A", "B", weight=0.5)

        # Forward pass
        outputs = graph.forward(num_layers=2)

        # Training
        optimizer = Adam(graph.parameters(), lr=0.01)
        loss_fn = MSELoss()

        loss = loss_fn(outputs["B"], target)
        graph.backward({"B": loss_fn.gradient(outputs["B"], target)})
        optimizer.step()
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        activation: Activation = Activation.RELU,
        aggregation: Aggregation = Aggregation.SUM,
        use_bias: bool = True,
        use_edge_weights: bool = True,
        dropout: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize trainable graph.

        Args:
            embedding_dim: Dimension of node embeddings
            activation: Activation function for updates
            aggregation: Method for aggregating messages
            use_bias: Whether to use bias in transformations
            use_edge_weights: Whether edges have learnable weights
            dropout: Dropout rate (0 = no dropout)
            seed: Random seed for reproducibility
        """
        super().__init__(InMemoryGraphStorage())

        self.embedding_dim = embedding_dim
        self.activation = activation
        self.aggregation = aggregation
        self.use_bias = use_bias
        self.use_edge_weights = use_edge_weights
        self.dropout = dropout

        if seed is not None:
            np.random.seed(seed)

        # Layer-wise transformation matrices (for multi-layer message passing)
        self._layer_transforms: List[Parameter] = []
        self._layer_biases: List[Parameter] = []

        # Training state
        self._training = True

        # Adjacency cache for optimized forward pass
        self._adjacency_cache: Optional[Dict[str, List[Tuple[str, int]]]] = None
        self._node_index: Optional[Dict[str, int]] = None
        self._index_to_node: Optional[Dict[int, str]] = None
        self._cache_valid = False

    def _create_node(self, id: str, **kwargs: Any) -> TrainableNode:
        """Create a trainable node with embedding."""
        embedding_data = kwargs.get("embedding")

        if embedding_data is None:
            # Initialize with Xavier/Glorot initialization
            scale = np.sqrt(2.0 / self.embedding_dim)
            embedding_data = np.random.randn(self.embedding_dim) * scale
        elif isinstance(embedding_data, (list, tuple)):
            embedding_data = np.array(embedding_data, dtype=np.float64)

        embedding = Parameter(
            data=embedding_data,
            requires_grad=kwargs.get("requires_grad", True),
            name=f"embedding_{id}",
        )

        return TrainableNode(
            id=id,
            node_type=kwargs.get("node_type", ""),
            content=kwargs.get("content", ""),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
            embedding=embedding,
        )

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        **kwargs: Any,
    ) -> TrainableEdge:
        """Create a trainable edge with learnable weight."""
        weight = kwargs.get("weight", 1.0)

        weight_param = None
        if self.use_edge_weights:
            # Initialize weight as learnable parameter
            weight_param = Parameter(
                data=np.array([weight], dtype=np.float64),
                requires_grad=kwargs.get("requires_grad", True),
                name=f"weight_{source_id}_{target_id}",
            )

        return TrainableEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            bidirectional=kwargs.get("bidirectional", False),
            properties=kwargs.get("properties", {}),
            weight_param=weight_param,
        )

    def add_node(
        self,
        node_id: str,
        node_type: str = "",
        content: str = "",
        **kwargs: Any,
    ) -> TrainableNode:
        """Add node and invalidate adjacency cache."""
        self._invalidate_cache()
        return super().add_node(node_id, node_type, content, **kwargs)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> Optional[TrainableEdge]:
        """Add edge and invalidate adjacency cache."""
        self._invalidate_cache()
        return super().add_edge(source_id, target_id, edge_type, weight, **kwargs)

    def _invalidate_cache(self) -> None:
        """Invalidate the adjacency cache."""
        self._cache_valid = False
        self._adjacency_cache = None
        self._node_index = None
        self._index_to_node = None

    def _build_adjacency_cache(self) -> None:
        """
        Build adjacency cache for optimized forward pass.

        Pre-computes:
        - Node index mappings for array-based operations
        - Incoming edges for each node (source_id, edge_index)
        - Edge weight array for vectorized operations
        """
        if self._cache_valid:
            return

        # Build node index mappings
        nodes_list = list(self.nodes)
        self._node_index = {node.id: idx for idx, node in enumerate(nodes_list)}
        self._index_to_node = {idx: node.id for idx, node in enumerate(nodes_list)}

        # Build adjacency cache: for each node, list of (source_id, edge_idx)
        self._adjacency_cache = {node.id: [] for node in nodes_list}

        edges_list = list(self.edges)
        for edge_idx, edge in enumerate(edges_list):
            self._adjacency_cache[edge.target_id].append((edge.source_id, edge_idx))

        self._cache_valid = True

    def parameters(self) -> List[Parameter]:
        """
        Get all learnable parameters.

        Returns:
            List of all trainable parameters (embeddings, edge weights, transforms)
        """
        params: List[Parameter] = []

        # Node embeddings
        for node in self.nodes:
            if node.embedding is not None and node.embedding.requires_grad:
                params.append(node.embedding)

        # Edge weights
        for edge in self.edges:
            if edge.weight_param is not None and edge.weight_param.requires_grad:
                params.append(edge.weight_param)

        # Layer transformations
        params.extend(self._layer_transforms)
        params.extend(self._layer_biases)

        return params

    def train(self, mode: bool = True) -> "TrainableGraph":
        """
        Set training mode.

        Args:
            mode: True for training, False for evaluation

        Returns:
            Self for chaining
        """
        self._training = mode
        return self

    def eval(self) -> "TrainableGraph":
        """Set evaluation mode (disables dropout)."""
        return self.train(False)

    def _ensure_layer_params(self, num_layers: int) -> None:
        """Ensure we have transformation parameters for the specified layers."""
        while len(self._layer_transforms) < num_layers:
            layer_idx = len(self._layer_transforms)

            # Xavier initialization for transformation matrix
            scale = np.sqrt(2.0 / (self.embedding_dim + self.embedding_dim))
            transform = Parameter(
                data=np.random.randn(self.embedding_dim, self.embedding_dim) * scale,
                requires_grad=True,
                name=f"transform_layer_{layer_idx}",
            )
            self._layer_transforms.append(transform)

            if self.use_bias:
                bias = Parameter(
                    data=np.zeros(self.embedding_dim),
                    requires_grad=True,
                    name=f"bias_layer_{layer_idx}",
                )
                self._layer_biases.append(bias)

    def _apply_dropout(self, x: Array) -> Array:
        """Apply dropout during training."""
        if not self._training or self.dropout <= 0:
            return x

        mask = np.random.binomial(1, 1 - self.dropout, size=x.shape)
        return x * mask / (1 - self.dropout)

    def forward(
        self,
        num_layers: int = 1,
        input_nodes: Optional[Dict[str, Array]] = None,
    ) -> Dict[str, Array]:
        """
        Forward pass through the graph.

        Performs message passing for the specified number of layers.
        Each layer:
        1. Computes messages from neighbors
        2. Aggregates messages
        3. Updates node embeddings via learned transformation

        Args:
            num_layers: Number of message passing layers
            input_nodes: Optional dict of node_id -> input features
                        (overrides node embeddings for those nodes)

        Returns:
            Dict mapping node_id to output embeddings
        """
        self._ensure_layer_params(num_layers)
        self._build_adjacency_cache()

        # Cache edge list for fast index access
        edges_list = list(self.edges)

        # Initialize node values and clear layer_inputs for fresh forward pass
        values: Dict[str, Array] = {}
        for node in self.nodes:
            node.layer_inputs = []  # Clear previous forward pass state
            if input_nodes and node.id in input_nodes:
                values[node.id] = input_nodes[node.id].copy()
            elif node.embedding is not None:
                values[node.id] = node.embedding.data.copy()
            else:
                values[node.id] = np.zeros(self.embedding_dim)

        # Pre-allocate aggregation buffer to avoid repeated np.stack
        agg_buffer = np.zeros((max(1, len(edges_list)), self.embedding_dim))

        # Message passing layers
        for layer in range(num_layers):
            new_values: Dict[str, Array] = {}
            transform = self._layer_transforms[layer].data

            for node in self.nodes:
                node_id = node.id
                incoming_edges = self._adjacency_cache[node_id]

                # Collect messages using cached adjacency
                incoming_info: List[Tuple[str, Array, float]] = []
                num_messages = len(incoming_edges)

                if num_messages > 0:
                    # Use pre-allocated buffer for aggregation
                    for msg_idx, (source_id, edge_idx) in enumerate(incoming_edges):
                        edge = edges_list[edge_idx]
                        source_value = values[source_id]

                        # Get edge weight
                        if edge.weight_param is not None:
                            weight = float(edge.weight_param.data[0])
                        else:
                            weight = edge.weight

                        # Compute message directly into buffer
                        agg_buffer[msg_idx] = source_value * weight
                        incoming_info.append((source_id, source_value.copy(), weight))

                    # Aggregate without np.stack
                    msg_slice = agg_buffer[:num_messages]
                    if self.aggregation == Aggregation.SUM:
                        aggregated = np.sum(msg_slice, axis=0)
                    elif self.aggregation == Aggregation.MEAN:
                        aggregated = np.mean(msg_slice, axis=0)
                    elif self.aggregation == Aggregation.MAX:
                        aggregated = np.max(msg_slice, axis=0)
                    elif self.aggregation == Aggregation.MIN:
                        aggregated = np.min(msg_slice, axis=0)
                    else:
                        aggregated = np.sum(msg_slice, axis=0)
                else:
                    aggregated = np.zeros(self.embedding_dim)

                # Store for backward pass
                node.incoming_messages = incoming_info

                # Combine with self-loop (original embedding)
                combined = values[node_id] + aggregated

                # Store layer input for backward pass (the value before transform)
                node.layer_inputs.append(combined.copy())

                # Apply transformation
                pre_activation = combined @ transform

                if self.use_bias and layer < len(self._layer_biases):
                    pre_activation += self._layer_biases[layer].data

                # Store pre-activation for backward pass
                node.pre_activation = pre_activation.copy()

                # Apply activation
                output = apply_activation(pre_activation, self.activation)

                # Apply dropout
                output = self._apply_dropout(output)

                new_values[node_id] = output
                node.output = output.copy()

            values = new_values

        return values

    def backward(
        self,
        output_gradients: Dict[str, Array],
        num_layers: int = 1,
    ) -> None:
        """
        Backward pass to compute gradients.

        Computes gradients for all parameters via backpropagation
        through the message passing layers.

        Args:
            output_gradients: Dict mapping node_id to gradient of loss
                            with respect to that node's output
            num_layers: Number of layers (must match forward pass)
        """
        # Node gradients (d_loss / d_node_value after each layer)
        node_grads: Dict[str, Array] = {}

        # Initialize with output gradients
        for node_id, grad in output_gradients.items():
            node_grads[node_id] = grad.copy()

        # Backpropagate through layers (reverse order)
        for layer in range(num_layers - 1, -1, -1):
            new_node_grads: Dict[str, Array] = {}

            for node in self.nodes:
                node_id = node.id

                if node_id not in node_grads:
                    continue

                grad = node_grads[node_id]

                # Gradient through activation
                if node.pre_activation is not None:
                    grad = grad * activation_derivative(
                        node.pre_activation,
                        self.activation
                    )

                # Gradient through transformation
                transform = self._layer_transforms[layer]

                # Gradient w.r.t. transformation matrix
                # Use stored layer_inputs from forward pass for correct gradient
                if layer < len(node.layer_inputs):
                    layer_input = node.layer_inputs[layer]
                elif node.embedding is not None:
                    # Fallback for layer 0 or if forward wasn't called
                    layer_input = node.embedding.data
                else:
                    layer_input = np.zeros(self.embedding_dim)

                # Outer product for transform gradient
                transform_grad = np.outer(layer_input, grad)
                transform.add_grad(transform_grad)

                # Gradient w.r.t. bias
                if self.use_bias and layer < len(self._layer_biases):
                    self._layer_biases[layer].add_grad(grad)

                # Gradient w.r.t. input (propagate to previous layer)
                input_grad = grad @ transform.data.T

                # Gradient w.r.t. self (embedding)
                if node_id not in new_node_grads:
                    new_node_grads[node_id] = np.zeros(self.embedding_dim)
                new_node_grads[node_id] += input_grad

                # Gradient through message aggregation
                for source_id, source_value, weight in node.incoming_messages:
                    # Gradient w.r.t. source node
                    source_grad = input_grad * weight
                    if source_id not in new_node_grads:
                        new_node_grads[source_id] = np.zeros(self.embedding_dim)
                    new_node_grads[source_id] += source_grad

                    # Gradient w.r.t. edge weight
                    edge = self.get_edge(source_id, node_id, "")
                    if edge and edge.weight_param is not None:
                        weight_grad = np.sum(input_grad * source_value)
                        edge.weight_param.add_grad(np.array([weight_grad]))

            node_grads = new_node_grads

        # Final gradient to embeddings
        for node in self.nodes:
            if node.id in node_grads and node.embedding is not None:
                node.embedding.add_grad(node_grads[node.id])

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for param in self.parameters():
            param.zero_grad()

    def get_embeddings(self) -> Dict[str, Array]:
        """
        Get current node embeddings.

        Returns:
            Dict mapping node_id to embedding array
        """
        return {
            node.id: node.embedding.data.copy()
            for node in self.nodes
            if node.embedding is not None
        }

    def set_embedding(self, node_id: str, embedding: Array) -> None:
        """
        Set embedding for a specific node.

        Args:
            node_id: Node ID
            embedding: New embedding values
        """
        node = self.get_node(node_id)
        if node is not None and node.embedding is not None:
            node.embedding.data = embedding.copy()

    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by global norm.

        Args:
            max_norm: Maximum gradient norm

        Returns:
            Total gradient norm before clipping
        """
        total_norm = 0.0

        for param in self.parameters():
            if param.grad is not None:
                total_norm += np.sum(param.grad**2)

        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            scale = max_norm / total_norm
            for param in self.parameters():
                if param.grad is not None:
                    param.grad *= scale

        return float(total_norm)

    def save_state(self) -> Dict[str, Any]:
        """
        Save graph state for checkpointing.

        Returns:
            Dict containing all learnable parameters
        """
        state = {
            "embeddings": {},
            "edge_weights": {},
            "transforms": [],
            "biases": [],
        }

        for node in self.nodes:
            if node.embedding is not None:
                state["embeddings"][node.id] = node.embedding.data.copy()

        for edge in self.edges:
            if edge.weight_param is not None:
                # Use "::" separator to safely handle node IDs containing underscores
                # (e.g., "pos_0::pos_1" instead of "pos_0_pos_1" which breaks on split)
                key = f"{edge.source_id}::{edge.target_id}"
                state["edge_weights"][key] = edge.weight_param.data.copy()

        for transform in self._layer_transforms:
            state["transforms"].append(transform.data.copy())

        for bias in self._layer_biases:
            state["biases"].append(bias.data.copy())

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load graph state from checkpoint.

        Args:
            state: State dict from save_state()
        """
        for node_id, embedding in state.get("embeddings", {}).items():
            node = self.get_node(node_id)
            if node is not None and node.embedding is not None:
                node.embedding.data = embedding.copy()

        for key, weight in state.get("edge_weights", {}).items():
            # Support both new "::" separator and legacy "_" for backward compatibility
            # New format: "pos_0::pos_1", legacy: "a_b" (only works if IDs have no underscores)
            if "::" in key:
                source_id, target_id = key.split("::", 1)
            else:
                # Legacy format - only works correctly for IDs without underscores
                parts = key.split("_")
                if len(parts) < 2:
                    continue
                source_id, target_id = parts[0], parts[1]
            edge = self.get_edge(source_id, target_id, "")
            if edge and edge.weight_param is not None:
                edge.weight_param.data = weight.copy()

        for i, transform in enumerate(state.get("transforms", [])):
            if i < len(self._layer_transforms):
                self._layer_transforms[i].data = transform.copy()

        for i, bias in enumerate(state.get("biases", [])):
            if i < len(self._layer_biases):
                self._layer_biases[i].data = bias.copy()

    def save_checkpoint(
        self,
        optimizer: Optional["Optimizer"] = None,
        epoch: int = 0,
        loss: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save full training checkpoint for resumption.

        Args:
            optimizer: Optimizer to save state from
            epoch: Current epoch number
            loss: Current loss value
            extra: Any additional data to save

        Returns:
            Checkpoint dict that can be used with load_checkpoint()
        """
        checkpoint = {
            "model_state": self.save_state(),
            "epoch": epoch,
            "loss": loss,
            "embedding_dim": self.embedding_dim,
            "activation": self.activation.value,
            "aggregation": self.aggregation.value,
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
            checkpoint["optimizer_class"] = type(optimizer).__name__

        if extra is not None:
            checkpoint["extra"] = extra

        return checkpoint

    def load_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        optimizer: Optional["Optimizer"] = None,
    ) -> Dict[str, Any]:
        """
        Load training checkpoint and resume training.

        Args:
            checkpoint: Checkpoint dict from save_checkpoint()
            optimizer: Optimizer to restore state to

        Returns:
            Dict with epoch, loss, and any extra data from checkpoint
        """
        # Load model state
        self.load_state(checkpoint["model_state"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", 0.0),
            "extra": checkpoint.get("extra", {}),
        }


# =============================================================================
# Training Utilities
# =============================================================================


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best weights when stopped
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = float("inf")
        self.wait = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self.stopped = False

    def __call__(
        self,
        loss: float,
        graph: Optional[TrainableGraph] = None
    ) -> bool:
        """
        Check if training should stop.

        Args:
            loss: Current validation loss
            graph: Graph to save state from (if restore_best)

        Returns:
            True if training should stop
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            if self.restore_best and graph is not None:
                self.best_state = graph.save_state()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                return True

        return False

    def restore(self, graph: TrainableGraph) -> None:
        """Restore best weights to graph."""
        if self.best_state is not None:
            graph.load_state(self.best_state)


class TrainingHistory:
    """Track training metrics over epochs."""

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.metrics: Dict[str, List[float]] = {}

    def log(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        **metrics: float,
    ) -> None:
        """Log metrics for an epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)

        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_best_epoch(self, metric: str = "val_loss") -> int:
        """Get epoch with best metric value."""
        if metric == "val_loss" and self.val_losses:
            return int(np.argmin(self.val_losses))
        elif metric == "train_loss":
            return int(np.argmin(self.train_losses))
        elif metric in self.metrics:
            return int(np.argmin(self.metrics[metric]))
        return 0


def train_step(
    graph: TrainableGraph,
    optimizer: Optimizer,
    loss_fn: LossFunction,
    targets: Dict[str, Array],
    num_layers: int = 1,
    clip_grad: Optional[float] = None,
) -> float:
    """
    Perform a single training step.

    Args:
        graph: Trainable graph
        optimizer: Optimizer instance
        loss_fn: Loss function
        targets: Dict mapping node_id to target values
        num_layers: Number of message passing layers
        clip_grad: Optional gradient clipping norm

    Returns:
        Loss value
    """
    # Forward pass
    outputs = graph.forward(num_layers=num_layers)

    # Compute loss and gradients for each target node
    total_loss = 0.0
    output_grads: Dict[str, Array] = {}

    for node_id, target in targets.items():
        if node_id in outputs:
            loss = loss_fn(outputs[node_id], target)
            total_loss += loss
            output_grads[node_id] = loss_fn.gradient(outputs[node_id], target)

    # Backward pass
    graph.backward(output_grads, num_layers=num_layers)

    # Gradient clipping
    if clip_grad is not None:
        graph.clip_gradients(clip_grad)

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    return total_loss


def fit(
    graph: TrainableGraph,
    optimizer: Optimizer,
    loss_fn: LossFunction,
    train_targets: Dict[str, Array],
    val_targets: Optional[Dict[str, Array]] = None,
    epochs: int = 100,
    num_layers: int = 1,
    clip_grad: Optional[float] = None,
    early_stopping: Optional[EarlyStopping] = None,
    scheduler: Optional[LRScheduler] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """
    Complete training loop.

    Args:
        graph: Trainable graph
        optimizer: Optimizer instance
        loss_fn: Loss function
        train_targets: Training targets (node_id -> target)
        val_targets: Validation targets (optional)
        epochs: Number of training epochs
        num_layers: Number of message passing layers
        clip_grad: Optional gradient clipping norm
        early_stopping: Optional early stopping callback
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress

    Returns:
        Training history with metrics
    """
    history = TrainingHistory()

    for epoch in range(epochs):
        # Training
        graph.train()
        train_loss = train_step(
            graph, optimizer, loss_fn, train_targets,
            num_layers, clip_grad
        )

        # Validation
        val_loss = None
        if val_targets:
            graph.eval()
            outputs = graph.forward(num_layers=num_layers)
            val_loss = sum(
                loss_fn(outputs[nid], target)
                for nid, target in val_targets.items()
                if nid in outputs
            )

        # Log history
        history.log(train_loss, val_loss, optimizer.lr)

        # Callbacks
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss or train_loss)
            else:
                scheduler.step()

        if early_stopping is not None:
            if early_stopping(val_loss or train_loss, graph):
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Progress
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            msg = f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f" - val_loss: {val_loss:.4f}"
            print(msg)

    # Restore best weights
    if early_stopping is not None and early_stopping.restore_best:
        early_stopping.restore(graph)

    return history
