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

    # TODO(agent): Implement cross_entropy loss function
    # SESSION_HANDOFF: Requires softmax output layer and vocab projection
    # CONTEXT: Currently only "mse" is supported
    loss_fn: str = "mse"

    # TODO(agent): Implement position encodings
    # SESSION_HANDOFF: Options are "learned" (trainable embeddings) or
    # "sinusoidal" (fixed, from Attention Is All You Need paper)
    # BLOCKED_BY: Need to decide how to add positions to input embeddings
    position_encoding: str = "none"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.loss_fn not in ("mse",):
            # TODO(agent): Remove this check when cross_entropy is implemented
            raise ValueError(
                f"loss_fn '{self.loss_fn}' not supported. Currently only 'mse' is available."
            )

        if self.position_encoding not in ("none",):
            # TODO(agent): Remove this check when position encodings are implemented
            raise ValueError(
                f"position_encoding '{self.position_encoding}' not supported. "
                "Currently only 'none' is available."
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
            f"  dropout: {self.dropout}",
            f"  use_bias: {self.use_bias}",
            "",
            "Training:",
            f"  epochs: {self.epochs}",
            f"  lr: {self.lr}",
            f"  clip_grad: {self.clip_grad}",
            f"  max_tokens: {self.max_tokens}",
            f"  seed: {self.seed}",
            f"  loss_fn: {self.loss_fn}",
        ]
        return "\n".join(lines)
