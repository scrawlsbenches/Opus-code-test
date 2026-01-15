"""
Unit tests for LR Schedulers.

Tests cover:
- StepLR: step-wise decay
- CosineAnnealingLR: smooth cosine decay
- ReduceLROnPlateau: metric-based reduction
- State persistence (state_dict/load_state_dict)
- Factory function create_scheduler()

TDD: Tests written before implementation.
"""

import math
import pytest

from cortical.experiments.scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    create_scheduler,
)


# =============================================================================
# Mock Optimizer for Testing
# =============================================================================


class MockOptimizer:
    """Simple mock optimizer with modifiable lr attribute."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr


# =============================================================================
# StepLR Tests
# =============================================================================


class TestStepLR:
    """Tests for StepLR scheduler."""

    def test_initial_lr_unchanged(self):
        """LR should not change at epoch 0."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        scheduler.step(epoch=0)

        assert optimizer.lr == pytest.approx(0.01)

    def test_lr_unchanged_before_step_size(self):
        """LR should remain at base_lr for epochs < step_size."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        for epoch in range(99):
            scheduler.step(epoch=epoch)

        assert optimizer.lr == pytest.approx(0.01)

    def test_lr_decays_at_step_size(self):
        """LR should decay by gamma at epoch = step_size."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        scheduler.step(epoch=100)

        assert optimizer.lr == pytest.approx(0.001)

    def test_lr_decays_multiple_steps(self):
        """LR should decay multiple times at multiples of step_size."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        # At epoch 200: gamma^2 = 0.01
        scheduler.step(epoch=200)
        assert optimizer.lr == pytest.approx(0.0001)

        # At epoch 300: gamma^3 = 0.001
        scheduler.step(epoch=300)
        assert optimizer.lr == pytest.approx(0.00001)

    def test_lr_with_different_gamma(self):
        """Test with gamma = 0.5 (halving every step_size epochs)."""
        optimizer = MockOptimizer(lr=0.1)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        scheduler.step(epoch=0)
        assert optimizer.lr == pytest.approx(0.1)

        scheduler.step(epoch=50)
        assert optimizer.lr == pytest.approx(0.05)

        scheduler.step(epoch=100)
        assert optimizer.lr == pytest.approx(0.025)

    def test_step_without_explicit_epoch(self):
        """Test using internal epoch counter."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

        # Steps: 1, 2, 3, 4
        scheduler.step()  # epoch 1 -> lr = 0.01
        assert optimizer.lr == pytest.approx(0.01)

        scheduler.step()  # epoch 2 -> lr = 0.01
        assert optimizer.lr == pytest.approx(0.01)

        scheduler.step()  # epoch 3 -> lr = 0.005
        assert optimizer.lr == pytest.approx(0.005)

    def test_state_dict_save_restore(self):
        """Test state persistence."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # Advance to epoch 25
        for _ in range(25):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()

        # Create new scheduler and restore
        optimizer2 = MockOptimizer(lr=0.01)
        scheduler2 = StepLR(optimizer2, step_size=10, gamma=0.1)
        scheduler2.load_state_dict(state)

        assert scheduler2.last_epoch == scheduler.last_epoch
        assert scheduler2._step_count == scheduler._step_count
        assert scheduler2.base_lr == scheduler.base_lr


# =============================================================================
# CosineAnnealingLR Tests
# =============================================================================


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_initial_lr_at_base(self):
        """LR should be at base_lr at epoch 0."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, lr_min=1e-6)

        scheduler.step(epoch=0)

        assert optimizer.lr == pytest.approx(0.01)

    def test_lr_at_midpoint(self):
        """LR should be midway between base and min at T_max/2."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, lr_min=0.0)

        scheduler.step(epoch=50)

        # cos(pi * 0.5) = 0, so (1 + 0) / 2 = 0.5
        # lr = 0 + (0.01 - 0) * 0.5 = 0.005
        expected = 0.005
        assert optimizer.lr == pytest.approx(expected)

    def test_lr_at_end(self):
        """LR should be at lr_min at epoch T_max."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, lr_min=1e-6)

        scheduler.step(epoch=100)

        # cos(pi) = -1, so (1 - 1) / 2 = 0
        # lr = lr_min + (base - lr_min) * 0 = lr_min
        assert optimizer.lr == pytest.approx(1e-6)

    def test_lr_decay_is_smooth(self):
        """LR should decrease smoothly (monotonically) from start to end."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, lr_min=1e-6)

        prev_lr = 0.01
        for epoch in range(1, 101):
            scheduler.step(epoch=epoch)
            assert optimizer.lr <= prev_lr, f"LR increased at epoch {epoch}"
            prev_lr = optimizer.lr

    def test_different_lr_min(self):
        """Test with different minimum LR."""
        optimizer = MockOptimizer(lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, lr_min=0.01)

        scheduler.step(epoch=200)

        assert optimizer.lr == pytest.approx(0.01)

    def test_state_dict_save_restore(self):
        """Test state persistence."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, lr_min=1e-6)

        # Advance to epoch 50
        for epoch in range(50):
            scheduler.step(epoch=epoch)

        # Save state
        state = scheduler.state_dict()

        # Create new scheduler and restore
        optimizer2 = MockOptimizer(lr=0.01)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=100, lr_min=1e-6)
        scheduler2.load_state_dict(state)

        assert scheduler2.last_epoch == scheduler.last_epoch


# =============================================================================
# ReduceLROnPlateau Tests
# =============================================================================


class TestReduceLROnPlateau:
    """Tests for ReduceLROnPlateau scheduler."""

    def test_no_reduction_when_improving(self):
        """LR should not change when metric improves."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=3, factor=0.1, min_lr=1e-6
        )

        # Improving metrics: 1.0, 0.9, 0.8, 0.7
        scheduler.step(1.0)
        assert optimizer.lr == pytest.approx(0.01)

        scheduler.step(0.9)
        assert optimizer.lr == pytest.approx(0.01)

        scheduler.step(0.8)
        assert optimizer.lr == pytest.approx(0.01)

    def test_reduction_after_patience_exceeded(self):
        """LR should reduce after patience epochs without improvement."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=3, factor=0.1, min_lr=1e-6
        )

        # First value sets best
        scheduler.step(1.0)
        assert optimizer.lr == pytest.approx(0.01)

        # No improvement for 3 epochs (patience)
        scheduler.step(1.0)  # bad 1
        scheduler.step(1.0)  # bad 2
        scheduler.step(1.0)  # bad 3 -> reduce!

        assert optimizer.lr == pytest.approx(0.001)

    def test_improvement_resets_patience(self):
        """Improvement should reset the patience counter."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=3, factor=0.1, min_lr=1e-6
        )

        scheduler.step(1.0)  # best = 1.0
        scheduler.step(1.0)  # bad 1
        scheduler.step(1.0)  # bad 2
        scheduler.step(0.9)  # improvement! resets counter
        scheduler.step(0.9)  # bad 1
        scheduler.step(0.9)  # bad 2

        # Should NOT have reduced (only 2 bad epochs after last improvement)
        assert optimizer.lr == pytest.approx(0.01)

    def test_respects_min_lr(self):
        """LR should not go below min_lr."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=1, factor=0.1, min_lr=0.001
        )

        scheduler.step(1.0)  # best = 1.0
        scheduler.step(1.0)  # bad 1 -> reduce to 0.001

        assert optimizer.lr == pytest.approx(0.001)

        scheduler.step(1.0)  # bad 1 -> would reduce to 0.0001, but min_lr=0.001

        assert optimizer.lr == pytest.approx(0.001)  # Clamped at min_lr

    def test_threshold_for_improvement(self):
        """Small improvements below threshold should not count."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=2, factor=0.1, min_lr=1e-6, threshold=0.01
        )

        scheduler.step(1.0)   # best = 1.0
        scheduler.step(0.999)  # improvement is 0.001 < threshold -> bad
        scheduler.step(0.998)  # improvement is 0.002 < threshold -> bad -> reduce!

        assert optimizer.lr == pytest.approx(0.001)

    def test_mode_max(self):
        """Test mode='max' where higher is better."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5, min_lr=1e-6
        )

        scheduler.step(0.5)  # best = 0.5
        scheduler.step(0.4)  # worse (lower)
        scheduler.step(0.3)  # worse (lower) -> reduce!

        assert optimizer.lr == pytest.approx(0.005)

    def test_state_dict_save_restore(self):
        """Test state persistence including best value."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=3, factor=0.1, min_lr=1e-6
        )

        scheduler.step(1.0)
        scheduler.step(0.8)
        scheduler.step(0.9)  # 1 bad epoch

        state = scheduler.state_dict()

        # Create new scheduler and restore
        optimizer2 = MockOptimizer(lr=0.01)
        scheduler2 = ReduceLROnPlateau(
            optimizer2, patience=3, factor=0.1, min_lr=1e-6
        )
        scheduler2.load_state_dict(state)

        assert scheduler2.best == pytest.approx(0.8)
        assert scheduler2.num_bad_epochs == 1
        assert scheduler2.current_lr == scheduler.current_lr


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateScheduler:
    """Tests for create_scheduler factory function."""

    def test_create_step_scheduler(self):
        """Test creating StepLR scheduler."""
        optimizer = MockOptimizer(lr=0.01)

        scheduler = create_scheduler(
            optimizer,
            schedule_type="step",
            epochs=500,
            step_size=100,
            gamma=0.5,
        )

        assert isinstance(scheduler, StepLR)
        assert scheduler.step_size == 100
        assert scheduler.gamma == 0.5

    def test_create_cosine_scheduler(self):
        """Test creating CosineAnnealingLR scheduler."""
        optimizer = MockOptimizer(lr=0.01)

        scheduler = create_scheduler(
            optimizer,
            schedule_type="cosine",
            epochs=500,
            lr_min=1e-5,
        )

        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 500
        assert scheduler.lr_min == 1e-5

    def test_create_plateau_scheduler(self):
        """Test creating ReduceLROnPlateau scheduler."""
        optimizer = MockOptimizer(lr=0.01)

        scheduler = create_scheduler(
            optimizer,
            schedule_type="plateau",
            epochs=500,
            gamma=0.5,
            patience=10,
            lr_min=1e-5,
        )

        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.factor == 0.5
        assert scheduler.patience == 10

    def test_invalid_schedule_type_raises(self):
        """Test that invalid schedule type raises ValueError."""
        optimizer = MockOptimizer(lr=0.01)

        with pytest.raises(ValueError, match="Unknown schedule_type"):
            create_scheduler(optimizer, schedule_type="invalid", epochs=500)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSchedulerIntegration:
    """Integration tests simulating real training scenarios."""

    def test_step_lr_training_simulation(self):
        """Simulate 300 epoch training with StepLR."""
        optimizer = MockOptimizer(lr=0.1)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        lrs = []
        for epoch in range(300):
            scheduler.step(epoch=epoch)
            lrs.append(optimizer.lr)

        # Check LR at key points
        assert lrs[0] == pytest.approx(0.1)    # epoch 0
        assert lrs[99] == pytest.approx(0.1)   # epoch 99
        assert lrs[100] == pytest.approx(0.01)  # epoch 100
        assert lrs[199] == pytest.approx(0.01)  # epoch 199
        assert lrs[200] == pytest.approx(0.001) # epoch 200

    def test_cosine_lr_training_simulation(self):
        """Simulate 100 epoch training with CosineAnnealing."""
        optimizer = MockOptimizer(lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, lr_min=0.001)

        lrs = []
        for epoch in range(101):
            scheduler.step(epoch=epoch)
            lrs.append(optimizer.lr)

        # Should start high and end low
        assert lrs[0] == pytest.approx(0.1)
        assert lrs[-1] == pytest.approx(0.001)

        # Should be monotonically decreasing
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], f"LR increased at epoch {i}"

    def test_plateau_lr_with_noisy_loss(self):
        """Simulate training with noisy validation loss."""
        optimizer = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6, threshold=0.01
        )

        # Simulate: loss decreases overall but with noise
        losses = [1.0, 0.95, 0.97, 0.90, 0.92, 0.88, 0.89, 0.87, 0.88, 0.87]

        for loss in losses:
            scheduler.step(loss)

        # LR should remain at initial since loss generally improved
        assert optimizer.lr == pytest.approx(0.01)

        # Now plateau for 6 epochs
        for _ in range(6):
            scheduler.step(0.87)

        # Should have reduced after patience (5) exceeded
        assert optimizer.lr < 0.01
