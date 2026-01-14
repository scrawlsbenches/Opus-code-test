"""
Profiling utilities for training diagnostics.

Provides timing, memory tracking, and gradient statistics collection
during training runs.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

import numpy as np


@dataclass
class StepMetrics:
    """Metrics captured for a single training step."""

    step: int
    loss: float
    forward_time_ms: float
    backward_time_ms: float
    update_time_ms: float
    total_time_ms: float
    gradient_norm: float
    memory_delta_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "loss": self.loss,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "update_time_ms": self.update_time_ms,
            "total_time_ms": self.total_time_ms,
            "gradient_norm": self.gradient_norm,
            "memory_delta_bytes": self.memory_delta_bytes,
        }


@dataclass
class ProfilingReport:
    """Aggregated profiling statistics from a training run."""

    total_steps: int = 0
    total_time_seconds: float = 0.0

    # Timing statistics (in milliseconds)
    forward_time_mean: float = 0.0
    forward_time_std: float = 0.0
    forward_time_max: float = 0.0

    backward_time_mean: float = 0.0
    backward_time_std: float = 0.0
    backward_time_max: float = 0.0

    update_time_mean: float = 0.0
    update_time_std: float = 0.0
    update_time_max: float = 0.0

    step_time_mean: float = 0.0
    step_time_std: float = 0.0
    step_time_max: float = 0.0

    # Gradient statistics
    gradient_norm_mean: float = 0.0
    gradient_norm_max: float = 0.0
    gradient_norm_min: float = 0.0

    # Memory statistics
    peak_memory_bytes: int = 0
    memory_trend: str = "stable"  # "increasing", "decreasing", "stable"

    # Loss statistics
    initial_loss: float = 0.0
    final_loss: float = 0.0
    min_loss: float = 0.0

    def __str__(self) -> str:
        """Human-readable profiling summary."""
        lines = [
            "=" * 60,
            "PROFILING REPORT",
            "=" * 60,
            f"Total steps: {self.total_steps}",
            f"Total time: {self.total_time_seconds:.2f}s",
            f"Steps/second: {self.total_steps / max(self.total_time_seconds, 0.001):.1f}",
            "",
            "TIMING (ms per step):",
            f"  Forward:  {self.forward_time_mean:.2f} +/- {self.forward_time_std:.2f} (max: {self.forward_time_max:.2f})",
            f"  Backward: {self.backward_time_mean:.2f} +/- {self.backward_time_std:.2f} (max: {self.backward_time_max:.2f})",
            f"  Update:   {self.update_time_mean:.2f} +/- {self.update_time_std:.2f} (max: {self.update_time_max:.2f})",
            f"  Total:    {self.step_time_mean:.2f} +/- {self.step_time_std:.2f} (max: {self.step_time_max:.2f})",
            "",
            "GRADIENTS:",
            f"  Norm mean: {self.gradient_norm_mean:.4f}",
            f"  Norm max:  {self.gradient_norm_max:.4f}",
            f"  Norm min:  {self.gradient_norm_min:.4f}",
            "",
            "MEMORY:",
            f"  Peak: {self.peak_memory_bytes / 1024 / 1024:.2f} MB",
            f"  Trend: {self.memory_trend}",
            "",
            "LOSS:",
            f"  Initial: {self.initial_loss:.6f}",
            f"  Final:   {self.final_loss:.6f}",
            f"  Min:     {self.min_loss:.6f}",
            f"  Reduction: {(1 - self.final_loss / max(self.initial_loss, 1e-10)) * 100:.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class Profiler:
    """
    Training profiler for timing and memory tracking.

    Usage:
        profiler = Profiler(enabled=True)

        with profiler.step(step_num) as metrics:
            with profiler.forward():
                outputs = model.forward()

            with profiler.backward():
                model.backward(grads)

            with profiler.update():
                optimizer.step()

            metrics.loss = loss_value
            metrics.gradient_norm = grad_norm

        report = profiler.report()
        print(report)
        profiler.close()  # Explicit cleanup

    Or use as context manager:
        with Profiler(enabled=True) as profiler:
            # ... training loop ...
        # Automatically cleaned up
    """

    def __init__(self, enabled: bool = True, track_memory: bool = True):
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is active (if False, minimal overhead)
            track_memory: Whether to track memory allocation
        """
        self.enabled = enabled
        self.track_memory = track_memory and enabled
        self._owns_tracemalloc = False  # Track if we started tracemalloc
        self._closed = False

        self._steps: List[StepMetrics] = []
        self._current_step: Optional[_StepContext] = None
        self._start_time: Optional[float] = None
        self._peak_memory: int = 0

        if self.track_memory:
            # Guard: only start tracemalloc if not already tracing
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._owns_tracemalloc = True

    @contextmanager
    def step(self, step_num: int):
        """
        Context manager for a training step.

        Yields a StepMetrics object that should be populated with loss
        and gradient_norm by the caller.
        """
        if not self.enabled:
            # Minimal overhead path
            metrics = StepMetrics(
                step=step_num, loss=0.0, forward_time_ms=0.0,
                backward_time_ms=0.0, update_time_ms=0.0,
                total_time_ms=0.0, gradient_norm=0.0
            )
            yield metrics
            return

        if self._start_time is None:
            self._start_time = time.perf_counter()

        ctx = _StepContext(step_num, self.track_memory)
        self._current_step = ctx

        step_start = time.perf_counter()
        memory_before = self._get_memory() if self.track_memory else 0

        try:
            yield ctx.metrics
        finally:
            step_end = time.perf_counter()
            ctx.metrics.total_time_ms = (step_end - step_start) * 1000

            if self.track_memory:
                memory_after = self._get_memory()
                ctx.metrics.memory_delta_bytes = memory_after - memory_before
                self._peak_memory = max(self._peak_memory, memory_after)

            self._steps.append(ctx.metrics)
            self._current_step = None

    @contextmanager
    def forward(self):
        """Time the forward pass."""
        if not self.enabled or self._current_step is None:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._current_step.metrics.forward_time_ms = (end - start) * 1000

    @contextmanager
    def backward(self):
        """Time the backward pass."""
        if not self.enabled or self._current_step is None:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._current_step.metrics.backward_time_ms = (end - start) * 1000

    @contextmanager
    def update(self):
        """Time the parameter update."""
        if not self.enabled or self._current_step is None:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._current_step.metrics.update_time_ms = (end - start) * 1000

    def _get_memory(self) -> int:
        """Get current memory usage in bytes."""
        if not self.track_memory:
            return 0
        current, _ = tracemalloc.get_traced_memory()
        return current

    def report(self) -> ProfilingReport:
        """Generate aggregated profiling report."""
        if not self._steps:
            return ProfilingReport()

        report = ProfilingReport()
        report.total_steps = len(self._steps)

        if self._start_time is not None:
            report.total_time_seconds = time.perf_counter() - self._start_time

        # Extract arrays for statistics
        forward_times = np.array([s.forward_time_ms for s in self._steps])
        backward_times = np.array([s.backward_time_ms for s in self._steps])
        update_times = np.array([s.update_time_ms for s in self._steps])
        total_times = np.array([s.total_time_ms for s in self._steps])
        grad_norms = np.array([s.gradient_norm for s in self._steps])
        losses = np.array([s.loss for s in self._steps])
        memory_deltas = np.array([s.memory_delta_bytes for s in self._steps])

        # Timing statistics
        report.forward_time_mean = float(np.mean(forward_times))
        report.forward_time_std = float(np.std(forward_times))
        report.forward_time_max = float(np.max(forward_times))

        report.backward_time_mean = float(np.mean(backward_times))
        report.backward_time_std = float(np.std(backward_times))
        report.backward_time_max = float(np.max(backward_times))

        report.update_time_mean = float(np.mean(update_times))
        report.update_time_std = float(np.std(update_times))
        report.update_time_max = float(np.max(update_times))

        report.step_time_mean = float(np.mean(total_times))
        report.step_time_std = float(np.std(total_times))
        report.step_time_max = float(np.max(total_times))

        # Gradient statistics
        report.gradient_norm_mean = float(np.mean(grad_norms))
        report.gradient_norm_max = float(np.max(grad_norms))
        report.gradient_norm_min = float(np.min(grad_norms))

        # Memory statistics
        report.peak_memory_bytes = self._peak_memory

        # Determine memory trend (linear regression on cumulative memory)
        if self.track_memory and len(memory_deltas) > 10:
            cumulative = np.cumsum(memory_deltas)
            slope = np.polyfit(np.arange(len(cumulative)), cumulative, 1)[0]
            if slope > 1000:  # Growing by >1KB per step
                report.memory_trend = "increasing"
            elif slope < -1000:
                report.memory_trend = "decreasing"
            else:
                report.memory_trend = "stable"

        # Loss statistics
        report.initial_loss = float(losses[0])
        report.final_loss = float(losses[-1])
        report.min_loss = float(np.min(losses))

        return report

    def get_loss_curve(self) -> List[float]:
        """Get loss values for plotting."""
        return [s.loss for s in self._steps]

    def get_step_metrics(self) -> List[StepMetrics]:
        """Get all step metrics."""
        return self._steps.copy()

    def reset(self) -> None:
        """Reset profiler state."""
        self._steps = []
        self._current_step = None
        self._start_time = None
        self._peak_memory = 0

    def close(self) -> None:
        """
        Explicitly release resources.

        Call this when done with the profiler, or use as context manager.
        Safe to call multiple times.
        """
        if self._closed:
            return

        if self._owns_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._owns_tracemalloc = False

        self._closed = True

    def __enter__(self) -> "Profiler":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()


class _StepContext:
    """Internal context for tracking a single step."""

    def __init__(self, step_num: int, track_memory: bool):
        self.metrics = StepMetrics(
            step=step_num,
            loss=0.0,
            forward_time_ms=0.0,
            backward_time_ms=0.0,
            update_time_ms=0.0,
            total_time_ms=0.0,
            gradient_norm=0.0,
            memory_delta_bytes=0,
        )
