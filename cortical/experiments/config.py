"""
Experiment Configuration
========================

Dataclass for experiment hyperparameters with serialization support.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for an AttentionGraph training experiment.

    All hyperparameters are stored here for reproducibility.
    Can be created from CLI args, dict, or JSON file.
    """

    # Required
    name: str
    input_path: str

    # Model architecture
    embedding_dim: int = 16
    num_heads: int = 2  # Default to 2 - tests show much better than 1
    num_layers: int = 2

    # Training
    epochs: int = 500
    lr: float = 0.03
    clip_grad: float = 1.0
    max_tokens: int = 50
    seed: int = 42

    # Optional features
    dropout: float = 0.0
    use_bias: bool = False
    residual: bool = False  # Residual connections: output = attention + input
    weight_decay: float = 0.0  # L2 regularization factor
    val_split: float = 0.0  # Fraction of tokens for validation (0.0 = no validation)

    # Loss function: "mse" for embedding matching, "cross_entropy" for language modeling
    # - mse: Targets are next-token embeddings, outputs match embedding space
    # - cross_entropy: Targets are token indices, outputs are logits over vocabulary
    loss_fn: str = "mse"

    # Position encoding type: "none", "learned", or "sinusoidal"
    # - none: No position information (default for backward compatibility)
    # - learned: Trainable position embeddings (recommended)
    # - sinusoidal: Fixed sin/cos patterns from "Attention Is All You Need"
    position_encoding: str = "none"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.loss_fn not in ("mse", "cross_entropy"):
            raise ValueError(
                f"loss_fn '{self.loss_fn}' not supported. Use 'mse' or 'cross_entropy'."
            )

        if self.position_encoding not in ("none", "learned", "sinusoidal"):
            raise ValueError(
                f"position_encoding '{self.position_encoding}' not supported. "
                "Use 'none', 'learned', or 'sinusoidal'."
            )

        if not 0.0 <= self.val_split <= 0.5:
            raise ValueError(
                f"val_split must be between 0.0 and 0.5, got {self.val_split}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text())

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from argparse namespace."""
        return cls(
            name=args.name,
            input_path=args.input,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            epochs=args.epochs,
            lr=args.lr,
            clip_grad=args.clip_grad,
            max_tokens=args.max_tokens,
            seed=args.seed,
            dropout=getattr(args, "dropout", 0.0),
            use_bias=getattr(args, "use_bias", False),
            residual=getattr(args, "residual", False),
            weight_decay=getattr(args, "weight_decay", 0.0),
            val_split=getattr(args, "val_split", 0.0),
            loss_fn=getattr(args, "loss_fn", "mse"),
            position_encoding=getattr(args, "position_encoding", "none"),
        )

    def summary(self) -> str:
        """Return human-readable summary of config."""
        lines = [
            f"Experiment: {self.name}",
            f"Input: {self.input_path}",
            "",
            "Architecture:",
            f"  embedding_dim: {self.embedding_dim}",
            f"  num_heads: {self.num_heads}",
            f"  num_layers: {self.num_layers}",
            f"  position_encoding: {self.position_encoding}",
            f"  dropout: {self.dropout}",
            f"  use_bias: {self.use_bias}",
            f"  residual: {self.residual}",
            "",
            "Training:",
            f"  epochs: {self.epochs}",
            f"  lr: {self.lr}",
            f"  clip_grad: {self.clip_grad}",
            f"  weight_decay: {self.weight_decay}",
            f"  val_split: {self.val_split}",
            f"  max_tokens: {self.max_tokens}",
            f"  seed: {self.seed}",
            f"  loss_fn: {self.loss_fn}",
        ]
        return "\n".join(lines)
