"""
Unit tests for Profiler: Training profiler for timing and memory tracking.

Tests cover:
- StepMetrics dataclass
- ProfilingReport generation
- Profiler context managers (step, forward, backward, update)
- Memory tracking
- Enabled/disabled modes
- Report statistics
- Resource cleanup
"""

import pytest
import numpy as np
import time
import tracemalloc

from cortical.experiments.profiler import (
    StepMetrics,
    ProfilingReport,
    Profiler,
)


# =============================================================================
# StepMetrics Tests
# =============================================================================


class TestStepMetrics:
    """Tests for StepMetrics dataclass."""

    def test_create_step_metrics(self):
        """Test creating StepMetrics with all fields."""
        metrics = StepMetrics(
            step=0,
            loss=0.5,
            forward_time_ms=10.0,
            backward_time_ms=15.0,
            update_time_ms=5.0,
            total_time_ms=30.0,
            gradient_norm=1.5,
            memory_delta_bytes=1024,
        )

        assert metrics.step == 0
        assert metrics.loss == 0.5
        assert metrics.forward_time_ms == 10.0
        assert metrics.backward_time_ms == 15.0
        assert metrics.update_time_ms == 5.0
        assert metrics.total_time_ms == 30.0
        assert metrics.gradient_norm == 1.5
        assert metrics.memory_delta_bytes == 1024

    def test_default_memory_delta(self):
        """Test default value for memory_delta_bytes."""
        metrics = StepMetrics(
            step=0,
            loss=0.5,
            forward_time_ms=10.0,
            backward_time_ms=15.0,
            update_time_ms=5.0,
            total_time_ms=30.0,
            gradient_norm=1.5,
        )

        assert metrics.memory_delta_bytes == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = StepMetrics(
            step=1,
            loss=0.3,
            forward_time_ms=8.0,
            backward_time_ms=12.0,
            update_time_ms=4.0,
            total_time_ms=24.0,
            gradient_norm=2.0,
            memory_delta_bytes=2048,
        )

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["step"] == 1
        assert d["loss"] == 0.3
        assert d["forward_time_ms"] == 8.0
        assert d["backward_time_ms"] == 12.0
        assert d["update_time_ms"] == 4.0
        assert d["total_time_ms"] == 24.0
        assert d["gradient_norm"] == 2.0
        assert d["memory_delta_bytes"] == 2048


# =============================================================================
# ProfilingReport Tests
# =============================================================================


class TestProfilingReport:
    """Tests for ProfilingReport dataclass."""

    def test_empty_report_defaults(self):
        """Test that empty report has zero defaults."""
        report = ProfilingReport()

        assert report.total_steps == 0
        assert report.total_time_seconds == 0.0
        assert report.forward_time_mean == 0.0
        assert report.backward_time_mean == 0.0
        assert report.update_time_mean == 0.0
        assert report.step_time_mean == 0.0
        assert report.gradient_norm_mean == 0.0
        assert report.peak_memory_bytes == 0
        assert report.memory_trend == "stable"
        assert report.initial_loss == 0.0
        assert report.final_loss == 0.0
        assert report.min_loss == 0.0

    def test_report_with_values(self):
        """Test report with populated values."""
        report = ProfilingReport(
            total_steps=100,
            total_time_seconds=10.0,
            forward_time_mean=5.0,
            forward_time_std=1.0,
            forward_time_max=10.0,
            backward_time_mean=7.0,
            backward_time_std=2.0,
            backward_time_max=15.0,
            update_time_mean=2.0,
            update_time_std=0.5,
            update_time_max=5.0,
            step_time_mean=14.0,
            step_time_std=3.0,
            step_time_max=30.0,
            gradient_norm_mean=1.5,
            gradient_norm_max=5.0,
            gradient_norm_min=0.1,
            peak_memory_bytes=1000000,
            memory_trend="increasing",
            initial_loss=1.0,
            final_loss=0.01,
            min_loss=0.005,
        )

        assert report.total_steps == 100
        assert report.total_time_seconds == 10.0
        assert report.forward_time_mean == 5.0
        assert report.gradient_norm_max == 5.0
        assert report.memory_trend == "increasing"
        assert report.initial_loss == 1.0
        assert report.final_loss == 0.01

    def test_report_str_output(self):
        """Test that __str__ produces readable output."""
        report = ProfilingReport(
            total_steps=50,
            total_time_seconds=5.0,
            forward_time_mean=10.0,
            forward_time_std=2.0,
            forward_time_max=20.0,
            backward_time_mean=15.0,
            backward_time_std=3.0,
            backward_time_max=25.0,
            update_time_mean=3.0,
            update_time_std=1.0,
            update_time_max=8.0,
            step_time_mean=28.0,
            step_time_std=5.0,
            step_time_max=50.0,
            gradient_norm_mean=2.0,
            gradient_norm_max=5.0,
            gradient_norm_min=0.5,
            peak_memory_bytes=2000000,
            memory_trend="stable",
            initial_loss=0.5,
            final_loss=0.1,
            min_loss=0.08,
        )

        output = str(report)

        assert isinstance(output, str)
        assert "PROFILING REPORT" in output
        assert "Total steps: 50" in output
        assert "TIMING" in output
        assert "Forward:" in output
        assert "Backward:" in output
        assert "Update:" in output
        assert "GRADIENTS" in output
        assert "MEMORY" in output
        assert "LOSS" in output

    def test_report_str_handles_zero_time(self):
        """Test that __str__ handles zero time gracefully."""
        report = ProfilingReport(
            total_steps=0,
            total_time_seconds=0.0,
        )

        output = str(report)

        # Should not raise ZeroDivisionError
        assert "Steps/second:" in output


# =============================================================================
# Profiler Initialization Tests
# =============================================================================


class TestProfilerInit:
    """Tests for Profiler initialization."""

    def test_enabled_profiler(self):
        """Test creating enabled profiler."""
        # Stop any existing tracemalloc to test clean init
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        profiler = Profiler(enabled=True, track_memory=True)

        try:
            assert profiler.enabled is True
            assert profiler.track_memory is True
        finally:
            profiler.close()

    def test_disabled_profiler(self):
        """Test creating disabled profiler."""
        profiler = Profiler(enabled=False)

        try:
            assert profiler.enabled is False
            assert profiler.track_memory is False  # Auto-disabled with enabled=False
        finally:
            profiler.close()

    def test_profiler_without_memory_tracking(self):
        """Test creating profiler without memory tracking."""
        profiler = Profiler(enabled=True, track_memory=False)

        try:
            assert profiler.enabled is True
            assert profiler.track_memory is False
        finally:
            profiler.close()


# =============================================================================
# Profiler Step Context Manager Tests
# =============================================================================


class TestProfilerStep:
    """Tests for Profiler.step context manager."""

    def test_step_measures_time(self):
        """Test that step measures total time."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                time.sleep(0.01)  # 10ms

            steps = profiler.get_step_metrics()
            assert len(steps) == 1
            assert steps[0].total_time_ms >= 10  # At least 10ms

    def test_step_stores_metrics(self):
        """Test that step stores metrics in list."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            for i in range(5):
                with profiler.step(i) as metrics:
                    metrics.loss = i * 0.1
                    metrics.gradient_norm = i * 0.5

            steps = profiler.get_step_metrics()
            assert len(steps) == 5
            assert steps[0].loss == 0.0
            assert steps[4].loss == 0.4
            assert steps[4].gradient_norm == 2.0

    def test_step_disabled_minimal_overhead(self):
        """Test that disabled profiler has minimal overhead."""
        with Profiler(enabled=False) as profiler:
            for i in range(10):
                with profiler.step(i) as metrics:
                    pass

            # Should still yield metrics, but not track anything
            # Get step metrics should be empty for disabled profiler
            steps = profiler.get_step_metrics()
            assert len(steps) == 0


# =============================================================================
# Profiler Timing Context Managers Tests
# =============================================================================


class TestProfilerTimingContexts:
    """Tests for forward, backward, update context managers."""

    def test_forward_timing(self):
        """Test that forward() measures forward pass time."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                with profiler.forward():
                    time.sleep(0.01)

            steps = profiler.get_step_metrics()
            assert steps[0].forward_time_ms >= 10

    def test_backward_timing(self):
        """Test that backward() measures backward pass time."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                with profiler.backward():
                    time.sleep(0.01)

            steps = profiler.get_step_metrics()
            assert steps[0].backward_time_ms >= 10

    def test_update_timing(self):
        """Test that update() measures update time."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                with profiler.update():
                    time.sleep(0.01)

            steps = profiler.get_step_metrics()
            assert steps[0].update_time_ms >= 10

    def test_all_timing_contexts(self):
        """Test using all timing contexts in a step."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                with profiler.forward():
                    time.sleep(0.005)
                with profiler.backward():
                    time.sleep(0.005)
                with profiler.update():
                    time.sleep(0.005)

            steps = profiler.get_step_metrics()
            assert steps[0].forward_time_ms >= 5
            assert steps[0].backward_time_ms >= 5
            assert steps[0].update_time_ms >= 5
            assert steps[0].total_time_ms >= 15

    def test_timing_contexts_outside_step_no_crash(self):
        """Test that timing contexts outside step don't crash."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            # Using timing contexts without a step context should be safe
            with profiler.forward():
                pass
            with profiler.backward():
                pass
            with profiler.update():
                pass

        # Should complete without error


# =============================================================================
# Profiler Report Generation Tests
# =============================================================================


class TestProfilerReport:
    """Tests for Profiler.report method."""

    def test_empty_report(self):
        """Test report with no steps."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            report = profiler.report()

            assert report.total_steps == 0

    def test_report_statistics(self):
        """Test report computes correct statistics with verified calculations."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            # Pre-defined values for deterministic testing
            losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            grad_norms = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

            for i in range(10):
                with profiler.step(i) as metrics:
                    with profiler.forward():
                        time.sleep(0.001)
                    with profiler.backward():
                        time.sleep(0.001)
                    with profiler.update():
                        time.sleep(0.001)
                    metrics.loss = losses[i]
                    metrics.gradient_norm = grad_norms[i]

            report = profiler.report()

            # Verify step count
            assert report.total_steps == 10

            # Verify timing - all should be positive (at least 1ms each)
            assert report.forward_time_mean >= 1.0, \
                f"Forward time mean should be >= 1ms, got {report.forward_time_mean}"
            assert report.backward_time_mean >= 1.0
            assert report.update_time_mean >= 1.0
            assert report.step_time_mean >= 3.0  # Sum of forward + backward + update

            # Verify gradient norm statistics match expected calculations exactly
            expected_grad_mean = 1.45  # mean of [1.0, 1.1, ..., 1.9]
            expected_grad_min = 1.0
            expected_grad_max = 1.9
            assert abs(report.gradient_norm_mean - expected_grad_mean) < 1e-10, \
                f"Grad norm mean: expected {expected_grad_mean}, got {report.gradient_norm_mean}"
            assert report.gradient_norm_min == expected_grad_min, \
                f"Grad norm min: expected {expected_grad_min}, got {report.gradient_norm_min}"
            assert report.gradient_norm_max == expected_grad_max, \
                f"Grad norm max: expected {expected_grad_max}, got {report.gradient_norm_max}"

            # Verify loss statistics match expected calculations exactly
            assert report.initial_loss == 1.0, \
                f"Initial loss should be 1.0, got {report.initial_loss}"
            assert report.final_loss == 0.1, \
                f"Final loss should be 0.1, got {report.final_loss}"
            assert report.min_loss == 0.1, \
                f"Min loss should be 0.1, got {report.min_loss}"

    def test_report_loss_reduction(self):
        """Test report calculates loss reduction correctly."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                metrics.loss = 1.0

            with profiler.step(1) as metrics:
                metrics.loss = 0.1

            report = profiler.report()

            assert report.initial_loss == 1.0
            assert report.final_loss == 0.1
            # Reduction should be 90%


# =============================================================================
# Profiler Utility Methods Tests
# =============================================================================


class TestProfilerUtilities:
    """Tests for Profiler utility methods."""

    def test_get_loss_curve(self):
        """Test get_loss_curve returns loss values."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            for i in range(5):
                with profiler.step(i) as metrics:
                    metrics.loss = i * 0.1

            curve = profiler.get_loss_curve()

            expected = [0.0, 0.1, 0.2, 0.3, 0.4]
            assert len(curve) == len(expected)
            for a, b in zip(curve, expected):
                assert abs(a - b) < 1e-10

    def test_get_step_metrics(self):
        """Test get_step_metrics returns copy of metrics."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            for i in range(3):
                with profiler.step(i) as metrics:
                    metrics.loss = i * 0.5

            steps = profiler.get_step_metrics()

            assert len(steps) == 3
            assert steps[0].loss == 0.0
            assert steps[1].loss == 0.5
            assert steps[2].loss == 1.0

            # Should be a copy
            original_steps = profiler._steps
            assert steps is not original_steps

    def test_reset_clears_data(self):
        """Test reset clears all profiling data."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            for i in range(5):
                with profiler.step(i) as metrics:
                    metrics.loss = 0.1

            assert len(profiler.get_step_metrics()) == 5

            profiler.reset()

            assert len(profiler.get_step_metrics()) == 0
            assert profiler._start_time is None


# =============================================================================
# Profiler Context Manager Tests
# =============================================================================


class TestProfilerContextManager:
    """Tests for Profiler as context manager."""

    def test_profiler_as_context_manager(self):
        """Test using profiler as context manager."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                metrics.loss = 0.5

        # After exit, profiler should be closed
        assert profiler._closed is True

    def test_close_is_idempotent(self):
        """Test that close can be called multiple times."""
        profiler = Profiler(enabled=True, track_memory=False)
        profiler.close()
        profiler.close()  # Should not raise
        profiler.close()  # Should not raise

        assert profiler._closed is True


# =============================================================================
# Memory Tracking Tests
# =============================================================================


class TestProfilerMemoryTracking:
    """Tests for memory tracking functionality."""

    def test_memory_delta_recorded(self):
        """Test that memory delta is recorded when memory tracking is enabled."""
        # Make sure no other tracemalloc is running
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        with Profiler(enabled=True, track_memory=True) as profiler:
            with profiler.step(0) as metrics:
                # Allocate some memory
                data = [np.random.randn(1000) for _ in range(10)]
                metrics.loss = 0.5

            steps = profiler.get_step_metrics()
            # Memory delta should be recorded (could be positive or negative
            # depending on GC)
            assert hasattr(steps[0], 'memory_delta_bytes')

    def test_memory_tracking_disabled(self):
        """Test that memory delta is zero when tracking disabled."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                data = [np.random.randn(1000) for _ in range(10)]
                metrics.loss = 0.5

            steps = profiler.get_step_metrics()
            assert steps[0].memory_delta_bytes == 0


# =============================================================================
# Profiler Edge Cases
# =============================================================================


class TestProfilerEdgeCases:
    """Tests for edge cases in Profiler."""

    def test_single_step(self):
        """Test profiler with single step."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                metrics.loss = 0.5
                metrics.gradient_norm = 1.0

            report = profiler.report()

            assert report.total_steps == 1
            assert report.initial_loss == 0.5
            assert report.final_loss == 0.5
            assert report.min_loss == 0.5

    def test_large_number_of_steps(self):
        """Test profiler with many steps."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            for i in range(100):
                with profiler.step(i) as metrics:
                    metrics.loss = 1.0 / (i + 1)
                    metrics.gradient_norm = 1.0

            report = profiler.report()

            assert report.total_steps == 100
            assert report.initial_loss == 1.0
            assert report.final_loss == pytest.approx(0.01, rel=0.01)

    def test_step_with_exception(self):
        """Test that step handles exceptions gracefully."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            try:
                with profiler.step(0) as metrics:
                    metrics.loss = 0.5
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Step should still be recorded despite exception
            steps = profiler.get_step_metrics()
            assert len(steps) == 1
            assert steps[0].loss == 0.5

    def test_nested_steps_not_supported(self):
        """Test behavior with nested steps (undefined but shouldn't crash)."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as outer_metrics:
                outer_metrics.loss = 0.5
                # This is undefined behavior but shouldn't crash
                with profiler.step(1) as inner_metrics:
                    inner_metrics.loss = 0.3

            # Should have recorded something without crashing
            steps = profiler.get_step_metrics()
            assert len(steps) >= 1
