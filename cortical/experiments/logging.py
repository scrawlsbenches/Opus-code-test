"""
Experiment Logging
==================

JSON-based logging for experiment results and metrics.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig


def get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


@dataclass
class ExperimentMetrics:
    """
    Metrics collected during an experiment run.
    """

    # Training curves
    train_losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)

    # Final metrics
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    min_loss: Optional[float] = None
    max_accuracy: Optional[float] = None

    # Timing
    training_time_seconds: Optional[float] = None

    # Metadata
    git_commit: Optional[str] = None
    timestamp: Optional[str] = None

    # TODO(agent): Add validation metrics when train/val split is implemented
    # SESSION_HANDOFF: Will need val_losses, val_accuracies lists
    # CONTEXT: Currently only tracking training metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentMetrics":
        """Create metrics from dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentMetrics":
        """Create metrics from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ExperimentLog:
    """
    Handles saving and loading experiment results.

    Directory structure:
        experiments/runs/{date}_{name}/
            config.json
            metrics.json
            summary.txt
    """

    def __init__(
        self,
        config: ExperimentConfig,
        base_dir: Path = Path("experiments/runs"),
    ):
        """
        Initialize experiment log.

        Args:
            config: Experiment configuration
            base_dir: Base directory for experiment runs
        """
        self.config = config
        self.base_dir = Path(base_dir)
        self.metrics = ExperimentMetrics()

        # Create unique directory name
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.run_dir = self.base_dir / f"{date_str}_{config.name}"

        # Track if we've saved
        self._saved = False

    @property
    def config_path(self) -> Path:
        """Path to config.json."""
        return self.run_dir / "config.json"

    @property
    def metrics_path(self) -> Path:
        """Path to metrics.json."""
        return self.run_dir / "metrics.json"

    @property
    def summary_path(self) -> Path:
        """Path to summary.txt."""
        return self.run_dir / "summary.txt"

    def log_epoch(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
    ) -> None:
        """
        Log metrics for a single epoch.

        Args:
            loss: Training loss for this epoch
            accuracy: Optional accuracy metric
            gradient_norm: Optional gradient norm
        """
        self.metrics.train_losses.append(loss)
        if accuracy is not None:
            self.metrics.accuracies.append(accuracy)
        if gradient_norm is not None:
            self.metrics.gradient_norms.append(gradient_norm)

    def finalize(
        self,
        final_loss: float,
        final_accuracy: float,
        training_time: float,
    ) -> None:
        """
        Finalize metrics after training completes.

        Args:
            final_loss: Final training loss
            final_accuracy: Final accuracy on training data
            training_time: Total training time in seconds
        """
        self.metrics.final_loss = final_loss
        self.metrics.final_accuracy = final_accuracy
        self.metrics.training_time_seconds = training_time

        if self.metrics.train_losses:
            self.metrics.min_loss = min(self.metrics.train_losses)
        if self.metrics.accuracies:
            self.metrics.max_accuracy = max(self.metrics.accuracies)

        self.metrics.git_commit = get_git_commit()
        self.metrics.timestamp = datetime.now().isoformat()

    def save(self) -> Path:
        """
        Save experiment results to disk.

        Returns:
            Path to the run directory
        """
        # Create directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(self.config_path)

        # Save metrics
        self.metrics_path.write_text(self.metrics.to_json())

        # Save human-readable summary
        self._save_summary()

        self._saved = True
        return self.run_dir

    def _save_summary(self) -> None:
        """Save human-readable summary."""
        lines = [
            "=" * 60,
            "EXPERIMENT SUMMARY",
            "=" * 60,
            "",
            self.config.summary(),
            "",
            "Results:",
            f"  final_loss: {self.metrics.final_loss:.4f}" if self.metrics.final_loss else "  final_loss: N/A",
            f"  final_accuracy: {self.metrics.final_accuracy:.1%}" if self.metrics.final_accuracy else "  final_accuracy: N/A",
            f"  min_loss: {self.metrics.min_loss:.4f}" if self.metrics.min_loss else "  min_loss: N/A",
            f"  training_time: {self.metrics.training_time_seconds:.1f}s" if self.metrics.training_time_seconds else "  training_time: N/A",
            "",
            f"Git commit: {self.metrics.git_commit or 'N/A'}",
            f"Timestamp: {self.metrics.timestamp or 'N/A'}",
            "",
            "=" * 60,
        ]
        self.summary_path.write_text("\n".join(lines))

    @classmethod
    def load(cls, run_dir: Path) -> "ExperimentLog":
        """
        Load experiment results from disk.

        Args:
            run_dir: Path to experiment run directory

        Returns:
            ExperimentLog with loaded config and metrics
        """
        run_dir = Path(run_dir)

        config = ExperimentConfig.load(run_dir / "config.json")
        log = cls(config, base_dir=run_dir.parent)
        log.run_dir = run_dir  # Override computed run_dir

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            log.metrics = ExperimentMetrics.from_json(metrics_path.read_text())

        log._saved = True
        return log


def list_experiments(base_dir: Path = Path("experiments/runs")) -> List[Path]:
    """
    List all experiment run directories.

    Args:
        base_dir: Base directory containing experiment runs

    Returns:
        List of paths to experiment directories, sorted by name
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    runs = [p for p in base_dir.iterdir() if p.is_dir() and (p / "config.json").exists()]
    return sorted(runs)


# TODO(agent): Implement checkpoint saving/loading
# SESSION_HANDOFF: Should save model parameters to checkpoint.npz
# BLOCKED_BY: Need to decide checkpoint format (numpy .npz vs pickle vs safetensors)
# CONTEXT: Currently only config and metrics are saved
