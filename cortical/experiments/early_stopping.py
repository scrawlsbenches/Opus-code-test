"""
Early Stopping
==============

Early stopping helper for training loops.

Monitors a metric (typically validation loss) and stops training
when no improvement is seen for a specified number of epochs.

Usage:
    from cortical.experiments.early_stopping import EarlyStopper

    stopper = EarlyStopper(patience=10, min_delta=1e-4)

    for epoch in range(max_epochs):
        train_step()
        val_loss = evaluate()

        result = stopper.step(val_loss)

        if result.is_best:
            best_params = save_snapshot(params)

        if result.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    restore_snapshot(params, best_params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopResult:
    """
    Result from early stopping check.

    Attributes:
        should_stop: True if training should stop (patience exceeded)
        is_best: True if this is a new best metric value
        patience_remaining: Number of epochs left before stopping
    """
    should_stop: bool
    is_best: bool
    patience_remaining: int


class EarlyStopper:
    """
    Early stopping helper for training loops.

    Tracks a metric (e.g., validation loss) and determines when to stop
    training based on patience and improvement threshold.

    Example:
        stopper = EarlyStopper(patience=10, min_delta=1e-4)

        for epoch in range(epochs):
            val_loss = evaluate()
            result = stopper.step(val_loss)

            if result.is_best:
                save_best_checkpoint()

            if result.should_stop:
                print("Early stopping!")
                break
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Initialize early stopper.

        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (lower is better, e.g., loss) or 'max' (higher is better, e.g., accuracy)
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # State
        self.best: Optional[float] = None
        self.counter = 0
        self.stopped_epoch: Optional[int] = None

    def step(self, metric: float, epoch: Optional[int] = None) -> EarlyStopResult:
        """
        Check if training should stop.

        Call this method after each epoch with the validation metric.

        Args:
            metric: Current metric value (e.g., validation loss)
            epoch: Optional current epoch number (for logging stopped_epoch)

        Returns:
            EarlyStopResult indicating whether to stop and if this is best
        """
        # Initialize best on first call
        if self.best is None:
            self.best = metric
            return EarlyStopResult(
                should_stop=False,
                is_best=True,
                patience_remaining=self.patience,
            )

        # Check if metric improved
        if self._is_improvement(metric):
            self.best = metric
            self.counter = 0
            return EarlyStopResult(
                should_stop=False,
                is_best=True,
                patience_remaining=self.patience,
            )
        else:
            self.counter += 1
            should_stop = self.counter >= self.patience
            if should_stop:
                self.stopped_epoch = epoch
            return EarlyStopResult(
                should_stop=should_stop,
                is_best=False,
                patience_remaining=max(0, self.patience - self.counter),
            )

    def _is_improvement(self, metric: float) -> bool:
        """
        Check if metric improved by at least min_delta.

        Args:
            metric: Current metric value

        Returns:
            True if metric improved by more than min_delta
        """
        if self.mode == "min":
            # Lower is better: improved if metric < best - min_delta
            return metric < self.best - self.min_delta
        else:
            # Higher is better: improved if metric > best + min_delta
            return metric > self.best + self.min_delta

    def state_dict(self) -> dict:
        """
        Return state for checkpointing.

        Can be used to save early stopper state when checkpointing.

        Returns:
            Dictionary containing early stopper state
        """
        return {
            "best": self.best,
            "counter": self.counter,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Load state from checkpoint.

        Args:
            state: Dictionary from state_dict()
        """
        self.best = state["best"]
        self.counter = state["counter"]
        self.stopped_epoch = state["stopped_epoch"]
