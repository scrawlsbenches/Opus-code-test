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
from typing import Dict, Optional, Union

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

    Design Decision:
        Sinusoidal encoding uses the formula from the original transformer paper:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Key properties:
        1. Each dimension oscillates at a different frequency
        2. Lower dimensions = higher frequency (local position info)
        3. Higher dimensions = lower frequency (global position info)
        4. Relative positions can be computed as linear transformations

        Compared to learned encodings:
        - Pro: No extra parameters to train
        - Pro: Generalizes to longer sequences than seen during training
        - Pro: Deterministic (no initialization variance)
        - Con: Less flexible than learned encodings
        - Con: Fixed pattern may not be optimal for all tasks

    Usage:
        pos_enc = SinusoidalPositionEncoding(max_len=100, embedding_dim=32)

        # Add positions to token embeddings
        input_with_pos = pos_enc.add_to_inputs(input_nodes)

        # No parameters to add to optimizer (fixed encoding)
    """

    def __init__(self, max_len: int, embedding_dim: int):
        """
        Initialize sinusoidal position encoding.

        Pre-computes the encoding matrix for efficiency.

        Args:
            max_len: Maximum sequence length supported
            embedding_dim: Dimension of position encodings (must match token embeddings)
        """
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Pre-compute the sinusoidal encoding matrix
        # Shape: (max_len, embedding_dim)
        self._encodings = self._compute_encodings(max_len, embedding_dim)

    def _compute_encodings(self, max_len: int, embedding_dim: int) -> np.ndarray:
        """
        Compute sinusoidal position encodings.

        Formula from "Attention Is All You Need":
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Returns:
            Encoding matrix of shape (max_len, embedding_dim)
        """
        encodings = np.zeros((max_len, embedding_dim))

        # Position indices: 0, 1, 2, ..., max_len-1
        positions = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)

        # Dimension indices for computing frequencies
        # For embedding_dim=64: [0, 2, 4, ..., 62]
        dim_indices = np.arange(0, embedding_dim, 2)  # Even indices

        # Compute the division term: 10000^(2i/d_model)
        # Using log for numerical stability: exp(2i/d_model * log(10000))
        div_term = np.exp(dim_indices * (-math.log(10000.0) / embedding_dim))

        # Compute sin for even indices, cos for odd indices
        encodings[:, 0::2] = np.sin(positions * div_term)
        encodings[:, 1::2] = np.cos(positions * div_term)

        return encodings

    def encode(self, position: int) -> np.ndarray:
        """
        Get position encoding for a single position.

        Args:
            position: Position index (0-based)

        Returns:
            Position encoding vector of shape (embedding_dim,)
        """
        if position >= self.max_len:
            # Unlike learned encoding, we could compute on-the-fly for any position
            # but we keep the same interface for consistency
            raise ValueError(
                f"Position {position} exceeds max_len {self.max_len}. "
                "Note: Sinusoidal encoding could handle longer sequences - "
                "increase max_len if needed."
            )
        return self._encodings[position]

    def encode_sequence(self, length: int) -> np.ndarray:
        """
        Get position encodings for a sequence.

        Args:
            length: Sequence length

        Returns:
            Position encodings of shape (length, embedding_dim)
        """
        if length > self.max_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_len {self.max_len}"
            )
        return self._encodings[:length]

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
        """
        Return trainable parameters.

        Sinusoidal encoding has no trainable parameters.
        """
        return []

    def zero_grad(self):
        """
        Reset gradients.

        No-op for sinusoidal encoding (nothing to train).
        """
        pass

    def backward(self, input_gradients: Dict[str, np.ndarray]) -> None:
        """
        Accumulate gradients from input node gradients.

        No-op for sinusoidal encoding since it's not trainable.
        Gradients pass through (added to input) but don't update encoding.

        Args:
            input_gradients: Dict mapping node_id to gradient w.r.t. input
        """
        # No gradients to accumulate - encoding is fixed
        pass


# Type alias for position encoding (either learned or sinusoidal)
PositionEncoding = Union[LearnedPositionEncoding, SinusoidalPositionEncoding]


def create_position_encoding(
    encoding_type: str,
    max_len: int,
    embedding_dim: int,
) -> Optional[PositionEncoding]:
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
