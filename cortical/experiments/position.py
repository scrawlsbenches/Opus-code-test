"""
Position Encodings
==================

Provides position information to the attention mechanism.

Without position encodings, the model only knows token identity but not
where tokens appear in the sequence. Position encodings add this information.

Supported types:
- learned: Trainable embedding per position (default)
- sinusoidal: Fixed sin/cos patterns (TODO)

TODO(agent): Position encoding shows mixed results in experiments.
SESSION_HANDOFF: Initial tests show learned position encoding doesn't consistently
improve accuracy over baseline. Possible causes:
1. Causal masking already provides implicit positional information
2. Learned embeddings add parameters that make optimization harder
3. May need different learning rate for position vs attention params
4. Sinusoidal (fixed) encoding might work better than learned
CONTEXT: Gradient flow is verified working. See comparison results at
30/50/75/100 tokens showing position encoding sometimes hurts accuracy.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from cortical.graph.trainable import Parameter


class LearnedPositionEncoding:
    """
    Learned position embeddings.

    Creates a trainable embedding matrix where each row corresponds to
    a position in the sequence. These embeddings are added to token
    embeddings before feeding into attention.

    Usage:
        pos_enc = LearnedPositionEncoding(max_len=100, embedding_dim=32)

        # Add positions to token embeddings
        for i, token_emb in enumerate(token_embeddings):
            input_with_pos = token_emb + pos_enc.encode(i)

        # Get parameters for optimizer
        optimizer = Adam(graph.parameters() + pos_enc.parameters(), lr=0.03)
    """

    def __init__(
        self,
        max_len: int,
        embedding_dim: int,
        scale: float = 0.02,
    ):
        """
        Initialize learned position encoding.

        Args:
            max_len: Maximum sequence length supported
            embedding_dim: Dimension of position embeddings (must match token embeddings)
            scale: Initialization scale (default 0.02, similar to BERT)
        """
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Initialize position embeddings with small random values
        # Using truncated normal-like initialization
        self.embeddings = Parameter(
            data=np.random.randn(max_len, embedding_dim) * scale,
            name="position_embeddings",
        )

    def encode(self, position: int) -> np.ndarray:
        """
        Get position encoding for a single position.

        Args:
            position: Position index (0-based)

        Returns:
            Position embedding vector of shape (embedding_dim,)
        """
        if position >= self.max_len:
            raise ValueError(
                f"Position {position} exceeds max_len {self.max_len}"
            )
        return self.embeddings.data[position]

    def encode_sequence(self, length: int) -> np.ndarray:
        """
        Get position encodings for a sequence.

        Args:
            length: Sequence length

        Returns:
            Position embeddings of shape (length, embedding_dim)
        """
        if length > self.max_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_len {self.max_len}"
            )
        return self.embeddings.data[:length]

    def add_to_inputs(
        self,
        input_nodes: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Add position encodings to input node embeddings.

        Assumes node IDs follow the pattern "pos_0", "pos_1", etc.

        Args:
            input_nodes: Dict mapping node IDs to embeddings

        Returns:
            New dict with position encodings added
        """
        result = {}
        for node_id, embedding in input_nodes.items():
            if node_id.startswith("pos_"):
                try:
                    position = int(node_id.split("_")[1])
                    result[node_id] = embedding + self.encode(position)
                except (IndexError, ValueError):
                    # Not a position node, keep as-is
                    result[node_id] = embedding
            else:
                result[node_id] = embedding
        return result

    def parameters(self):
        """Return trainable parameters."""
        return [self.embeddings]

    def zero_grad(self):
        """Reset gradients."""
        self.embeddings.zero_grad()

    def backward(self, input_gradients: Dict[str, np.ndarray]) -> None:
        """
        Accumulate gradients from input node gradients.

        Since position encoding is added element-wise to inputs:
            input_with_pos = token_emb + pos_enc

        The gradient of loss w.r.t. pos_enc equals the gradient w.r.t. input_with_pos.

        Args:
            input_gradients: Dict mapping node_id to gradient w.r.t. input
        """
        for node_id, grad in input_gradients.items():
            if node_id.startswith("pos_"):
                try:
                    position = int(node_id.split("_")[1])
                    if position < self.max_len:
                        # Accumulate gradient for this position
                        if self.embeddings.grad is None:
                            self.embeddings.grad = np.zeros_like(self.embeddings.data)
                        self.embeddings.grad[position] += grad
                except (IndexError, ValueError):
                    pass  # Not a position node


class SinusoidalPositionEncoding:
    """
    Fixed sinusoidal position encoding from "Attention Is All You Need".

    Uses sine and cosine functions of different frequencies to encode
    positions. Not trainable but generalizes to unseen sequence lengths.

    TODO(agent): Implement sinusoidal encoding
    SESSION_HANDOFF: Formula is:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    CONTEXT: Useful when we need to generalize to longer sequences
    """

    def __init__(self, max_len: int, embedding_dim: int):
        raise NotImplementedError(
            "Sinusoidal position encoding not yet implemented. "
            "Use 'learned' position encoding for now."
        )


def create_position_encoding(
    encoding_type: str,
    max_len: int,
    embedding_dim: int,
) -> Optional[LearnedPositionEncoding]:
    """
    Factory function to create position encoding.

    Args:
        encoding_type: Type of encoding ("none", "learned", "sinusoidal")
        max_len: Maximum sequence length
        embedding_dim: Embedding dimension

    Returns:
        Position encoding instance, or None if encoding_type is "none"
    """
    if encoding_type == "none":
        return None
    elif encoding_type == "learned":
        return LearnedPositionEncoding(max_len, embedding_dim)
    elif encoding_type == "sinusoidal":
        return SinusoidalPositionEncoding(max_len, embedding_dim)
    else:
        raise ValueError(f"Unknown position encoding type: {encoding_type}")
