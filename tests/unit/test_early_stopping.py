"""
Unit tests for Early Stopping functionality.

Tests cover:
- EarlyStopper class behavior
- Patience counter increments on no improvement
- Patience counter resets on improvement
- Min delta threshold for determining improvement
- Best parameter snapshot saving and restoration
- Integration with training loop
"""

import pytest
import numpy as np
from typing import List, Dict

from cortical.experiments.early_stopping import EarlyStopper, EarlyStopResult


# =============================================================================
# Basic EarlyStopper Tests
# =============================================================================


class TestEarlyStopperBasic:
    """Tests for basic EarlyStopper functionality."""

    def test_first_call_sets_best(self):
        """First metric value should become best."""
        stopper = EarlyStopper(patience=5, min_delta=0.01)

        result = stopper.step(1.0)

        assert stopper.best == 1.0
        assert result.is_best is True
        assert result.should_stop is False

    def test_improvement_resets_counter(self):
        """Improvement should reset patience counter."""
        stopper = EarlyStopper(patience=5, min_delta=0.01)

        stopper.step(1.0)  # best = 1.0
        stopper.step(1.0)  # no improvement, counter = 1
        stopper.step(1.0)  # no improvement, counter = 2

        assert stopper.counter == 2

        result = stopper.step(0.5)  # improvement!

        assert stopper.counter == 0
        assert stopper.best == 0.5
        assert result.is_best is True

    def test_no_improvement_increments_counter(self):
        """No improvement should increment patience counter."""
        stopper = EarlyStopper(patience=5, min_delta=0.01)

        stopper.step(1.0)  # best = 1.0
        result1 = stopper.step(1.0)  # no improvement
        result2 = stopper.step(1.1)  # worse

        assert stopper.counter == 2
        assert result1.is_best is False
        assert result2.is_best is False

    def test_stops_when_patience_exceeded(self):
        """Should stop when patience is exceeded."""
        stopper = EarlyStopper(patience=3, min_delta=0.01)

        stopper.step(1.0)  # best = 1.0
        r1 = stopper.step(1.0)  # counter = 1
        r2 = stopper.step(1.0)  # counter = 2
        r3 = stopper.step(1.0)  # counter = 3 -> stop!

        assert r1.should_stop is False
        assert r2.should_stop is False
        assert r3.should_stop is True
        assert r3.patience_remaining == 0

    def test_patience_remaining_decreases(self):
        """patience_remaining should decrease with each bad epoch."""
        stopper = EarlyStopper(patience=5, min_delta=0.01)

        stopper.step(1.0)  # best
        r1 = stopper.step(1.0)  # counter = 1
        r2 = stopper.step(1.0)  # counter = 2
        r3 = stopper.step(1.0)  # counter = 3

        assert r1.patience_remaining == 4
        assert r2.patience_remaining == 3
        assert r3.patience_remaining == 2


class TestEarlyStopperMinDelta:
    """Tests for min_delta threshold behavior."""

    def test_small_improvement_below_min_delta_is_not_improvement(self):
        """Improvement smaller than min_delta should not count."""
        stopper = EarlyStopper(patience=5, min_delta=0.1)

        stopper.step(1.0)  # best = 1.0
        result = stopper.step(0.95)  # only 0.05 improvement < 0.1 min_delta

        assert result.is_best is False
        assert stopper.counter == 1
        assert stopper.best == 1.0  # unchanged

    def test_improvement_at_exactly_min_delta_is_not_improvement(self):
        """Improvement exactly at min_delta should not count (need to exceed)."""
        stopper = EarlyStopper(patience=5, min_delta=0.1)

        stopper.step(1.0)  # best = 1.0
        result = stopper.step(0.9)  # exactly 0.1 improvement = min_delta

        # Strictly less than, so 0.9 is NOT < 1.0 - 0.1 = 0.9
        assert result.is_best is False

    def test_improvement_exceeding_min_delta_counts(self):
        """Improvement exceeding min_delta should count."""
        stopper = EarlyStopper(patience=5, min_delta=0.1)

        stopper.step(1.0)  # best = 1.0
        result = stopper.step(0.89)  # 0.11 improvement > 0.1 min_delta

        assert result.is_best is True
        assert stopper.best == 0.89

    def test_zero_min_delta_requires_any_improvement(self):
        """With min_delta=0, any improvement should count."""
        stopper = EarlyStopper(patience=5, min_delta=0.0)

        stopper.step(1.0)
        result = stopper.step(0.9999)  # tiny improvement

        assert result.is_best is True


class TestEarlyStopperModeMax:
    """Tests for mode='max' (higher is better)."""

    def test_mode_max_higher_is_better(self):
        """In mode='max', higher values should be better."""
        stopper = EarlyStopper(patience=5, min_delta=0.01, mode="max")

        stopper.step(0.5)  # best = 0.5
        r1 = stopper.step(0.4)  # worse (lower)
        r2 = stopper.step(0.6)  # better (higher)

        assert r1.is_best is False
        assert r2.is_best is True
        assert stopper.best == 0.6

    def test_mode_max_min_delta_threshold(self):
        """In mode='max', min_delta should work in opposite direction."""
        stopper = EarlyStopper(patience=5, min_delta=0.1, mode="max")

        stopper.step(0.5)  # best = 0.5
        r1 = stopper.step(0.55)  # only 0.05 improvement < 0.1
        r2 = stopper.step(0.65)  # 0.15 improvement > 0.1

        assert r1.is_best is False
        assert r2.is_best is True


class TestEarlyStopperStateDict:
    """Tests for state persistence."""

    def test_state_dict_roundtrip(self):
        """State should survive save/load cycle."""
        stopper1 = EarlyStopper(patience=5, min_delta=0.01)

        stopper1.step(1.0)  # best = 1.0
        stopper1.step(1.0)  # counter = 1
        stopper1.step(1.0)  # counter = 2

        state = stopper1.state_dict()

        stopper2 = EarlyStopper(patience=5, min_delta=0.01)
        stopper2.load_state_dict(state)

        assert stopper2.best == stopper1.best
        assert stopper2.counter == stopper1.counter


# =============================================================================
# Parameter Snapshot Tests
# =============================================================================


def save_param_snapshot(params: List) -> Dict[str, np.ndarray]:
    """Save a snapshot of parameter values."""
    return {p.name: p.data.copy() for p in params}


def restore_param_snapshot(params: List, snapshot: Dict[str, np.ndarray]) -> int:
    """Restore parameters from snapshot. Returns count restored."""
    restored = 0
    for p in params:
        if p.name in snapshot:
            p.data[:] = snapshot[p.name]
            restored += 1
    return restored


class TestParameterSnapshot:
    """Tests for parameter snapshot save/restore."""

    def test_snapshot_saves_copies(self):
        """Snapshot should save copies, not references."""
        from cortical.graph.trainable import Parameter

        params = [Parameter(data=np.array([1.0, 2.0, 3.0]), name="test")]
        snapshot = save_param_snapshot(params)

        # Modify original
        params[0].data[:] = [10.0, 20.0, 30.0]

        # Snapshot should be unchanged
        np.testing.assert_array_equal(snapshot["test"], [1.0, 2.0, 3.0])

    def test_restore_overwrites_params(self):
        """Restore should overwrite current parameter values."""
        from cortical.graph.trainable import Parameter

        params = [Parameter(data=np.array([10.0, 20.0, 30.0]), name="test")]
        snapshot = {"test": np.array([1.0, 2.0, 3.0])}

        restore_param_snapshot(params, snapshot)

        np.testing.assert_array_equal(params[0].data, [1.0, 2.0, 3.0])

    def test_restore_returns_count(self):
        """Restore should return number of parameters restored."""
        from cortical.graph.trainable import Parameter

        params = [
            Parameter(data=np.array([1.0]), name="a"),
            Parameter(data=np.array([2.0]), name="b"),
            Parameter(data=np.array([3.0]), name="c"),
        ]
        snapshot = {"a": np.array([10.0]), "b": np.array([20.0])}

        count = restore_param_snapshot(params, snapshot)

        assert count == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestEarlyStoppingIntegration:
    """Integration tests for early stopping in training loop."""

    def test_training_stops_early(self):
        """Training should stop before max epochs when loss plateaus."""
        stopper = EarlyStopper(patience=3, min_delta=0.01)

        # Simulate training with plateauing loss
        losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        stopped_at = None

        for epoch, loss in enumerate(losses):
            result = stopper.step(loss, epoch=epoch)
            if result.should_stop:
                stopped_at = epoch
                break

        # Should stop at epoch 7 (patience=3 after best at epoch 3)
        # epoch 4: counter=1, epoch 5: counter=2, epoch 6: counter=3 -> stop
        assert stopped_at == 6

    def test_best_model_saved_at_lowest_loss(self):
        """Best model should be saved when loss is lowest."""
        from cortical.graph.trainable import Parameter

        stopper = EarlyStopper(patience=5, min_delta=0.01)
        params = [Parameter(data=np.array([0.0]), name="w")]
        best_snapshot = None

        # Simulate training: loss decreases then increases
        losses = [1.0, 0.8, 0.6, 0.4, 0.5, 0.6, 0.7]  # best at 0.4

        for epoch, loss in enumerate(losses):
            # Update param to track which epoch we're at
            params[0].data[:] = epoch

            result = stopper.step(loss, epoch=epoch)
            if result.is_best:
                best_snapshot = save_param_snapshot(params)

            if result.should_stop:
                break

        # Best should be saved at epoch 3 (loss = 0.4)
        assert best_snapshot is not None
        assert best_snapshot["w"][0] == 3  # epoch 3

    def test_best_params_restored_after_early_stop(self):
        """Best params should be restored after early stopping."""
        from cortical.graph.trainable import Parameter

        stopper = EarlyStopper(patience=3, min_delta=0.01)
        params = [Parameter(data=np.array([0.0]), name="w")]
        best_snapshot = None

        losses = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9]  # best at 0.5 (epoch 1)

        for epoch, loss in enumerate(losses):
            params[0].data[:] = epoch  # Track epoch in param

            result = stopper.step(loss, epoch=epoch)
            if result.is_best:
                best_snapshot = save_param_snapshot(params)

            if result.should_stop:
                break

        # Current param is at epoch 4 (where we stopped)
        assert params[0].data[0] == 4

        # Restore best
        restore_param_snapshot(params, best_snapshot)

        # Should be back to epoch 1 (best)
        assert params[0].data[0] == 1

    def test_full_training_simulation(self):
        """Full simulation of training with early stopping."""
        from cortical.graph.trainable import Parameter, Adam

        np.random.seed(42)

        # Setup
        params = [
            Parameter(data=np.random.randn(10), name="weights"),
            Parameter(data=np.zeros(5), name="bias"),
        ]
        optimizer = Adam(params, lr=0.01)
        stopper = EarlyStopper(patience=5, min_delta=0.001)

        max_epochs = 100
        best_snapshot = None
        stopped_at = None

        # Simulate training with loss that decreases then plateaus
        for epoch in range(max_epochs):
            # Simulate training step
            for p in params:
                p.grad = np.random.randn(*p.data.shape) * 0.1
            optimizer.step()

            # Compute fake validation loss (decreases then plateaus)
            if epoch < 20:
                val_loss = 1.0 - epoch * 0.04  # 1.0 -> 0.2
            else:
                val_loss = 0.2 + np.random.randn() * 0.001  # plateaus with noise

            result = stopper.step(val_loss, epoch=epoch)

            if result.is_best:
                best_snapshot = save_param_snapshot(params)

            if result.should_stop:
                stopped_at = epoch
                break

        # Should have stopped early (before max_epochs)
        assert stopped_at is not None
        assert stopped_at < max_epochs

        # Best snapshot should exist
        assert best_snapshot is not None

        # Restore and verify
        restore_param_snapshot(params, best_snapshot)
        # (Just verify it doesn't crash - values are random)


class TestEarlyStoppingRequiresValidation:
    """Tests for validation split requirement."""

    def test_early_stop_without_val_split_should_warn(self):
        """Early stopping without validation should be handled gracefully."""
        # This test documents the expected behavior:
        # If val_split=0, early stopping has no val_loss to monitor
        # The CLI should either:
        # 1. Raise an error if --early-stop is used without --val-split
        # 2. Fall back to using train_loss (less ideal)

        # For now, we'll test that EarlyStopper works with any metric
        stopper = EarlyStopper(patience=3, min_delta=0.01)

        # Using train_loss instead of val_loss
        train_losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
        stopped_at = None

        for epoch, loss in enumerate(train_losses):
            result = stopper.step(loss, epoch=epoch)
            if result.should_stop:
                stopped_at = epoch
                break

        # Should still work
        assert stopped_at == 5  # patience=3 after best at epoch 2
