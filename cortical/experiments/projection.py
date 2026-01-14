"""
Vocabulary Projection
=====================

Projects embedding outputs to vocabulary logits for language modeling.

This module provides the missing piece needed for cross-entropy loss:
a trainable projection layer that maps from embedding space to vocabulary space.

Usage:
    vocab_proj = VocabProjection(embedding_dim=64, vocab_size=1000)

    # During forward pass
    outputs = graph.forward(...)
    logits = vocab_proj.forward(outputs)  # Dict[str, Array] with vocab_size dims

    # For loss
    loss = cross_entropy(logits, target_indices)

    # Backward
    grad_logits = cross_entropy.gradient(logits, target_indices)
    input_grads = vocab_proj.backward(grad_logits)
    graph.backward(input_grads, ...)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from cortical.graph.trainable import Parameter


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class VocabProjection:
    """
    Projects embeddings to vocabulary logits.

    Implements: logits = embedding @ W + b
    Where W has shape (embedding_dim, vocab_size) and b has shape (vocab_size,)

    The layer also computes softmax to produce probabilities for cross-entropy.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        use_bias: bool = True,
        scale: float = 0.02,
    ):
        """
        Initialize vocabulary projection.

        Args:
            embedding_dim: Dimension of input embeddings
            vocab_size: Size of vocabulary (output dimension)
            use_bias: Whether to use bias term
            scale: Initialization scale for weights
        """
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.use_bias = use_bias

        # Initialize weights with small random values
        self.W = Parameter(
            data=np.random.randn(embedding_dim, vocab_size) * scale,
            name="vocab_projection_W",
        )

        if use_bias:
            self.b = Parameter(
                data=np.zeros(vocab_size),
                name="vocab_projection_b",
            )
        else:
            self.b = None

        # Cache for backward pass
        self._input_cache: Dict[str, np.ndarray] = {}
        self._logits_cache: Dict[str, np.ndarray] = {}
        self._probs_cache: Dict[str, np.ndarray] = {}

    def forward(
        self,
        embeddings: Dict[str, np.ndarray],
        apply_softmax: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Project embeddings to vocabulary logits/probabilities.

        Args:
            embeddings: Dict mapping node_id to embedding vectors
            apply_softmax: Whether to apply softmax (True for cross-entropy)

        Returns:
            Dict mapping node_id to logits/probabilities of shape (vocab_size,)
        """
        self._input_cache = {}
        self._logits_cache = {}
        self._probs_cache = {}

        results = {}

        for node_id, emb in embeddings.items():
            # Cache input for backward
            self._input_cache[node_id] = emb.copy()

            # Linear projection: logits = emb @ W + b
            logits = emb @ self.W.data
            if self.b is not None:
                logits = logits + self.b.data

            self._logits_cache[node_id] = logits

            if apply_softmax:
                probs = softmax(logits)
                self._probs_cache[node_id] = probs
                results[node_id] = probs
            else:
                results[node_id] = logits

        return results

    def backward(
        self,
        grad_output: Dict[str, np.ndarray],
        from_softmax: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Backward pass through projection layer.

        Args:
            grad_output: Gradient w.r.t. output (logits or probs)
            from_softmax: Whether grad_output is w.r.t. softmax output

        Returns:
            Dict of gradients w.r.t. input embeddings
        """
        input_grads = {}

        for node_id, grad in grad_output.items():
            if node_id not in self._input_cache:
                continue

            emb = self._input_cache[node_id]

            if from_softmax and node_id in self._probs_cache:
                # For softmax + cross-entropy, the combined gradient is:
                # d(loss)/d(logits) = probs - one_hot(target)
                # But if we're receiving grad w.r.t. probs from cross-entropy.gradient(),
                # we need to apply the softmax jacobian.
                #
                # Actually, for numerical stability, cross-entropy + softmax gradient
                # simplifies to: grad_logits = probs - target
                # So if caller passes (probs - target), we can use it directly as grad_logits
                grad_logits = grad
            else:
                grad_logits = grad

            # Gradient w.r.t. input: grad_emb = grad_logits @ W.T
            input_grads[node_id] = grad_logits @ self.W.data.T

            # Gradient w.r.t. W: grad_W = emb.T @ grad_logits (outer product for 1D)
            # For single vector: grad_W = outer(emb, grad_logits)
            grad_W = np.outer(emb, grad_logits)

            if self.W.grad is None:
                self.W.grad = np.zeros_like(self.W.data)
            self.W.grad += grad_W

            # Gradient w.r.t. bias
            if self.b is not None:
                if self.b.grad is None:
                    self.b.grad = np.zeros_like(self.b.data)
                self.b.grad += grad_logits

        return input_grads

    def parameters(self) -> List[Parameter]:
        """Return trainable parameters."""
        if self.b is not None:
            return [self.W, self.b]
        return [self.W]

    def zero_grad(self) -> None:
        """Reset gradients and clear forward pass caches."""
        self.W.zero_grad()
        if self.b is not None:
            self.b.zero_grad()
        # Clear caches to prevent memory leak and stale values
        self._input_cache.clear()
        self._logits_cache.clear()
        self._probs_cache.clear()


class CrossEntropyWithLogits:
    """
    Cross-entropy loss that works directly with logits.

    Combines softmax + cross-entropy for numerical stability.
    This is more stable than separate softmax and cross-entropy.

    For language modeling:
        - Input (logits): shape (vocab_size,)
        - Target: either one-hot (vocab_size,) or token index (int)
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self._probs_cache: Dict[str, np.ndarray] = {}

    def __call__(
        self,
        logits: np.ndarray,
        target: np.ndarray,
        node_id: str = "",
    ) -> float:
        """
        Compute cross-entropy loss from logits.

        Args:
            logits: Raw logits of shape (vocab_size,)
            target: One-hot target of shape (vocab_size,) or index
            node_id: Optional node ID for caching

        Returns:
            Scalar loss value
        """
        # Convert index to one-hot if needed
        if target.ndim == 0 or (target.ndim == 1 and target.shape[0] != logits.shape[0]):
            idx = int(target.item() if hasattr(target, 'item') else target)
            one_hot = np.zeros_like(logits)
            one_hot[idx] = 1.0
            target = one_hot

        # Compute softmax (numerically stable)
        probs = softmax(logits)
        self._probs_cache[node_id] = probs

        # Cross-entropy: -sum(target * log(probs))
        loss = -np.sum(target * np.log(probs + self.epsilon))
        return float(loss)

    def gradient(
        self,
        logits: np.ndarray,
        target: np.ndarray,
        node_id: str = "",
    ) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. logits.

        The gradient of cross-entropy + softmax is simply: probs - target

        Args:
            logits: Raw logits
            target: One-hot target or index
            node_id: Optional node ID for cached probs

        Returns:
            Gradient w.r.t. logits
        """
        # Convert index to one-hot if needed
        if target.ndim == 0 or (target.ndim == 1 and target.shape[0] != logits.shape[0]):
            idx = int(target.item() if hasattr(target, 'item') else target)
            one_hot = np.zeros_like(logits)
            one_hot[idx] = 1.0
            target = one_hot

        # Use cached probs if available
        if node_id in self._probs_cache:
            probs = self._probs_cache[node_id]
        else:
            probs = softmax(logits)

        # Gradient: probs - target
        return probs - target
