"""
Learning Rate Schedulers
========================

Stub implementations for learning rate scheduling.

TODO(agent): Implement these schedulers for the experiment CLI
SESSION_HANDOFF: These classes are stubbed out with NotImplementedError
CONTEXT: Optimizer.lr can be modified dynamically - schedulers should do this

Usage (once implemented):
    optimizer = Adam(params, lr=0.01)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(epochs):
        train_step(...)
        scheduler.step()  # Update LR based on schedule
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cortical.graph.trainable import Optimizer


class LRScheduler(ABC):
    """
    Base class for learning rate schedulers.

    All schedulers modify the optimizer's learning rate based on
    training progress (epoch count or metrics).

    TODO(agent): Implement the step() method in subclasses
    """

    def __init__(self, optimizer: "Optimizer", last_epoch: int = -1):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer instance with modifiable 'lr' attribute
            last_epoch: The index of last epoch (-1 means start fresh)
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self._step_count = 0

    @abstractmethod
    def get_lr(self) -> float:
        """
        Compute the learning rate for the current epoch.

        Returns:
            The new learning rate
        """
        pass

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Update the learning rate.

        Args:
            epoch: Optional epoch number (uses internal counter if None)
        """
        if epoch is None:
            self._step_count += 1
            epoch = self._step_count

        self.last_epoch = epoch
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "base_lr": self.base_lr,
            "last_epoch": self.last_epoch,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.base_lr = state["base_lr"]
        self.last_epoch = state["last_epoch"]
        self._step_count = state["_step_count"]


class StepLR(LRScheduler):
    """
    Decay learning rate by gamma every step_size epochs.

    lr = base_lr * gamma^(epoch // step_size)

    Example:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        # LR: 0.01 -> 0.001 (at epoch 100) -> 0.0001 (at epoch 200)

    TODO(agent): Implement get_lr() method
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        step_size: int = 100,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer instance
            step_size: Period of LR decay (epochs)
            gamma: Multiplicative factor of LR decay
            last_epoch: The index of last epoch
        """
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        """
        Compute stepped learning rate.

        TODO(agent): Implement this method
        Formula: base_lr * gamma^(epoch // step_size)
        """
        raise NotImplementedError(
            "StepLR.get_lr() is not yet implemented.\n"
            "Formula: base_lr * gamma^(epoch // step_size)\n"
            "Example: With step_size=100, gamma=0.1:\n"
            "  epoch 0-99: base_lr\n"
            "  epoch 100-199: base_lr * 0.1\n"
            "  epoch 200-299: base_lr * 0.01"
        )


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.

    Smoothly decays LR from base_lr to lr_min following a cosine curve.

    lr = lr_min + (base_lr - lr_min) * (1 + cos(pi * epoch / T_max)) / 2

    Example:
        scheduler = CosineAnnealingLR(optimizer, T_max=500, lr_min=1e-6)
        # Smooth decay from 0.01 to 1e-6 over 500 epochs

    TODO(agent): Implement get_lr() method
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        T_max: int,
        lr_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer instance
            T_max: Maximum number of epochs
            lr_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        super().__init__(optimizer, last_epoch)
        self.T_max = T_max
        self.lr_min = lr_min

    def get_lr(self) -> float:
        """
        Compute cosine-annealed learning rate.

        TODO(agent): Implement this method
        Formula: lr_min + (base_lr - lr_min) * (1 + cos(pi * epoch / T_max)) / 2

        Use: import math; math.cos(math.pi * epoch / T_max)
        """
        raise NotImplementedError(
            "CosineAnnealingLR.get_lr() is not yet implemented.\n"
            "Formula: lr_min + (base_lr - lr_min) * (1 + cos(pi * epoch / T_max)) / 2\n"
            "This creates a smooth cosine curve from base_lr to lr_min."
        )


class ReduceLROnPlateau(LRScheduler):
    """
    Reduce learning rate when a metric has stopped improving.

    Unlike other schedulers, this one requires a metric value to be
    passed to step(). Commonly used with validation loss.

    Example:
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        for epoch in range(epochs):
            train_loss = train_step(...)
            val_loss = evaluate(...)
            scheduler.step(val_loss)  # Pass metric, not epoch

    TODO(agent): Implement step() and get_lr() methods
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer instance
            mode: 'min' or 'max' - direction of improvement
            factor: Factor to reduce LR by (new_lr = old_lr * factor)
            patience: Number of epochs to wait before reducing
            min_lr: Minimum learning rate
            threshold: Minimum change to qualify as improvement
            last_epoch: The index of last epoch
        """
        super().__init__(optimizer, last_epoch)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold

        # Tracking state
        self.best: Optional[float] = None
        self.num_bad_epochs = 0
        self.current_lr = self.base_lr

    def get_lr(self) -> float:
        """Return current learning rate (set by step())."""
        return self.current_lr

    def step(self, metric: float) -> None:  # type: ignore[override]
        """
        Update LR based on metric value.

        TODO(agent): Implement this method
        Logic:
            1. Check if metric improved (considering mode and threshold)
            2. If improved: reset num_bad_epochs, update best
            3. If not improved: increment num_bad_epochs
            4. If num_bad_epochs >= patience: reduce LR by factor

        Args:
            metric: Current metric value (e.g., validation loss)
        """
        raise NotImplementedError(
            "ReduceLROnPlateau.step() is not yet implemented.\n"
            "Logic:\n"
            "  1. Check if metric improved (mode='min' means lower is better)\n"
            "  2. If improved by threshold: reset patience counter, update best\n"
            "  3. If not improved: increment bad epoch counter\n"
            "  4. If counter >= patience: new_lr = max(current_lr * factor, min_lr)"
        )

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        state = super().state_dict()
        state.update({
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "current_lr": self.current_lr,
        })
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        super().load_state_dict(state)
        self.best = state["best"]
        self.num_bad_epochs = state["num_bad_epochs"]
        self.current_lr = state["current_lr"]


def create_scheduler(
    optimizer: "Optimizer",
    schedule_type: str,
    epochs: int,
    **kwargs,
) -> LRScheduler:
    """
    Factory function to create a learning rate scheduler.

    TODO(agent): This function works once scheduler implementations are complete

    Args:
        optimizer: Optimizer instance
        schedule_type: "step", "cosine", or "plateau"
        epochs: Total training epochs (used for T_max in cosine)
        **kwargs: Additional scheduler-specific arguments
            - step: step_size, gamma
            - cosine: lr_min
            - plateau: factor, patience, min_lr, threshold

    Returns:
        LRScheduler instance

    Raises:
        ValueError: If schedule_type is not recognized
    """
    if schedule_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 100),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif schedule_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            lr_min=kwargs.get("lr_min", 1e-6),
        )
    elif schedule_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            factor=kwargs.get("gamma", 0.1),  # Use gamma for consistency
            patience=kwargs.get("patience", 10),
            min_lr=kwargs.get("lr_min", 1e-6),
            threshold=kwargs.get("threshold", 1e-4),
        )
    else:
        raise ValueError(
            f"Unknown schedule_type '{schedule_type}'. "
            "Use 'step', 'cosine', or 'plateau'."
        )
