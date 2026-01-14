"""
ExperimentKernel: Training harness for TrainableGraphProtocol implementations.

This module provides a flexible training loop that works with any graph
implementing the TrainableGraphProtocol interface (AttentionGraph, TrainableGraph, etc.)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np

from .profiler import Profiler, StepMetrics, ProfilingReport

if TYPE_CHECKING:
    from cortical.graph.attention import TrainableGraphProtocol, Parameter

# Type alias for numpy arrays
Array = np.ndarray


def clip_gradients(parameters: List["Parameter"], max_norm: float) -> float:
    """
    Clip gradients by global norm.

    This is a standalone utility that works with any list of Parameter objects,
    making it compatible with TrainableGraphProtocol (which doesn't require
    a clip_gradients method on the graph itself).

    Args:
        parameters: List of Parameter objects with .grad attributes
        max_norm: Maximum allowed gradient norm

    Returns:
        The global gradient norm before clipping
    """
    # Compute global norm
    total_norm_sq = 0.0

    for param in parameters:
        if param.grad is not None:
            total_norm_sq += np.sum(param.grad ** 2)

    total_norm = np.sqrt(total_norm_sq)

    # Clip if necessary
    if total_norm > max_norm and total_norm > 0:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad * clip_coef

    return float(total_norm)


def compute_gradient_norm(parameters: List["Parameter"]) -> float:
    """
    Compute the global gradient norm across all parameters.

    Args:
        parameters: List of Parameter objects

    Returns:
        The L2 norm of all gradients concatenated
    """
    total_norm_sq = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm_sq += np.sum(param.grad ** 2)
    return float(np.sqrt(total_norm_sq))


@dataclass
class TrainingHistory:
    """Records training metrics over time."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    step_metrics: List[StepMetrics] = field(default_factory=list)

    def log(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        metrics: Optional[StepMetrics] = None,
    ) -> None:
        """Log metrics for one step/epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
        if metrics is not None:
            self.step_metrics.append(metrics)


class ExperimentKernel:
    """
    Training harness for TrainableGraphProtocol implementations.

    Provides a flexible training loop with:
    - Profiling support (timing, memory, gradients)
    - Gradient clipping
    - Verbose logging
    - Clean interface for any protocol-compliant graph
    - Position encoding support with proper gradient flow

    Usage:
        from cortical.graph.attention import AttentionGraph, create_causal_attention_graph
        from cortical.graph.trainable import Adam, MSELoss

        graph = create_causal_attention_graph(seq_len=100, embedding_dim=64)
        optimizer = Adam(graph.parameters(), lr=0.001)
        loss_fn = MSELoss()

        kernel = ExperimentKernel(graph, optimizer, loss_fn)
        history = kernel.fit(targets, epochs=500)
        print(kernel.profile_report())
    """

    def __init__(
        self,
        graph: "TrainableGraphProtocol",
        optimizer: Any,  # Optimizer from trainable.py
        loss_fn: Any,  # LossFunction from trainable.py
        profiling: bool = True,
        track_memory: bool = True,
        position_encoding: Any = None,  # Optional position encoding module
        vocab_projection: Any = None,  # Optional vocab projection for cross-entropy
    ):
        """
        Initialize the experiment kernel.

        Args:
            graph: Graph implementing TrainableGraphProtocol
            optimizer: Optimizer instance (SGD, Adam, etc.)
            loss_fn: Loss function instance (MSELoss, CrossEntropyLoss, etc.)
            profiling: Whether to collect profiling data
            track_memory: Whether to track memory allocation
            position_encoding: Optional position encoding module (e.g., LearnedPositionEncoding)
                             Must have a backward(input_gradients) method for gradient propagation
            vocab_projection: Optional vocabulary projection for cross-entropy loss
                            Must have forward() and backward() methods
        """
        self.graph = graph
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.position_encoding = position_encoding
        self.vocab_projection = vocab_projection
        self.profiler = Profiler(enabled=profiling, track_memory=track_memory)
        self._history = TrainingHistory()

    def train_step(
        self,
        targets: Dict[str, Array],
        num_layers: int = 1,
        clip_grad: Optional[float] = None,
        input_nodes: Optional[Dict[str, Array]] = None,
    ) -> StepMetrics:
        """
        Perform a single training step.

        Args:
            targets: Dict mapping node_id to target values
            num_layers: Number of processing layers to apply
            clip_grad: Optional gradient clipping max norm
            input_nodes: Optional input override for graph.forward()

        Returns:
            StepMetrics with timing and loss information
        """
        step_num = len(self._history.train_losses)

        with self.profiler.step(step_num) as metrics:
            # Forward pass
            with self.profiler.forward():
                outputs = self.graph.forward(
                    num_layers=num_layers,
                    input_nodes=input_nodes,
                )

                # Apply vocab projection if present (for cross-entropy)
                if self.vocab_projection is not None:
                    projected = self.vocab_projection.forward(outputs, apply_softmax=False)
                else:
                    projected = outputs

            # Compute loss and gradients
            total_loss = 0.0
            output_grads: Dict[str, Array] = {}

            for node_id, target in targets.items():
                if node_id in projected:
                    # For cross-entropy with logits, pass node_id for caching
                    if hasattr(self.loss_fn, '_probs_cache'):
                        loss = self.loss_fn(projected[node_id], target, node_id=node_id)
                        output_grads[node_id] = self.loss_fn.gradient(
                            projected[node_id], target, node_id=node_id
                        )
                    else:
                        loss = self.loss_fn(projected[node_id], target)
                        output_grads[node_id] = self.loss_fn.gradient(
                            projected[node_id], target
                        )
                    total_loss += loss

            metrics.loss = total_loss

            # Backward pass
            with self.profiler.backward():
                # If vocab projection is present, backprop through it first
                if self.vocab_projection is not None:
                    graph_grads = self.vocab_projection.backward(output_grads, from_softmax=False)
                else:
                    graph_grads = output_grads

                input_gradients = self.graph.backward(graph_grads, num_layers=num_layers)

                # Propagate gradients to position encoding if present
                # This is critical: position encoding is added BEFORE forward,
                # so we need to manually propagate gradients from input nodes
                if self.position_encoding is not None and input_gradients is not None:
                    self.position_encoding.backward(input_gradients)

            # Compute gradient norm before clipping (include all trainable params)
            all_params = self.graph.parameters()
            if self.position_encoding is not None:
                all_params = all_params + self.position_encoding.parameters()
            if self.vocab_projection is not None:
                all_params = all_params + self.vocab_projection.parameters()
            grad_norm = compute_gradient_norm(all_params)
            metrics.gradient_norm = grad_norm

            # Gradient clipping
            if clip_grad is not None:
                clip_gradients(all_params, clip_grad)

            # Parameter update
            with self.profiler.update():
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Log to history
        self._history.log(
            train_loss=total_loss,
            lr=getattr(self.optimizer, 'lr', None),
            grad_norm=grad_norm,
            metrics=metrics,
        )

        return metrics

    def fit(
        self,
        targets: Dict[str, Array],
        epochs: int = 100,
        num_layers: int = 1,
        clip_grad: Optional[float] = None,
        input_nodes: Optional[Dict[str, Array]] = None,
        verbose: bool = True,
        log_every: int = 10,
        callback: Optional[Callable[[int, StepMetrics], None]] = None,
    ) -> TrainingHistory:
        """
        Complete training loop.

        Args:
            targets: Dict mapping node_id to target values
            epochs: Number of training iterations
            num_layers: Number of processing layers
            clip_grad: Optional gradient clipping max norm
            input_nodes: Optional input override
            verbose: Whether to print progress
            log_every: Print progress every N epochs
            callback: Optional callback(epoch, metrics) called each epoch

        Returns:
            TrainingHistory with all recorded metrics
        """
        self.graph.train()

        if verbose:
            print(f"Starting training for {epochs} epochs...")
            print(f"  Graph: {type(self.graph).__name__}")
            print(f"  Embedding dim: {self.graph.embedding_dim}")
            print(f"  Parameters: {sum(p.data.size for p in self.graph.parameters())}")
            print(f"  Targets: {len(targets)} nodes")
            print(f"  Layers: {num_layers}")
            print()

        for epoch in range(epochs):
            metrics = self.train_step(
                targets=targets,
                num_layers=num_layers,
                clip_grad=clip_grad,
                input_nodes=input_nodes,
            )

            if callback is not None:
                callback(epoch, metrics)

            if verbose and (epoch + 1) % log_every == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{epochs} | "
                    f"Loss: {metrics.loss:.6f} | "
                    f"Grad: {metrics.gradient_norm:.4f} | "
                    f"Time: {metrics.total_time_ms:.1f}ms"
                )

        # Switch to eval mode after training
        self.graph.eval()

        if verbose:
            print()
            print("Training complete.")
            print(f"  Final loss: {metrics.loss:.6f}")
            print(f"  Loss reduction: {(1 - metrics.loss / max(self._history.train_losses[0], 1e-10)) * 100:.1f}%")

        return self._history

    def evaluate(
        self,
        targets: Dict[str, Array],
        num_layers: int = 1,
        input_nodes: Optional[Dict[str, Array]] = None,
    ) -> float:
        """
        Evaluate the graph on targets without updating parameters.

        Args:
            targets: Dict mapping node_id to target values
            num_layers: Number of processing layers
            input_nodes: Optional input override

        Returns:
            Total loss
        """
        self.graph.eval()

        outputs = self.graph.forward(
            num_layers=num_layers,
            input_nodes=input_nodes,
        )

        # Apply vocab projection if present (consistent with train_step)
        if self.vocab_projection is not None:
            projected = self.vocab_projection.forward(outputs, apply_softmax=False)
        else:
            projected = outputs

        total_loss = 0.0
        for node_id, target in targets.items():
            if node_id in projected:
                total_loss += self.loss_fn(projected[node_id], target)

        return total_loss

    def profile_report(self) -> ProfilingReport:
        """Get aggregated profiling report."""
        return self.profiler.report()

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return self._history

    def get_attention_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of attention weights after training.

        Returns:
            Dict mapping node_id to dict of source_id -> attention_weight
        """
        if hasattr(self.graph, 'get_attention_weights'):
            return self.graph.get_attention_weights()
        return {}

    def reset(self) -> None:
        """Reset profiler and history for a new training run."""
        self.profiler.reset()
        self._history = TrainingHistory()
