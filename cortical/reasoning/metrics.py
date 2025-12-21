"""
Reasoning Loop Metrics and Observability.

This module provides metrics collection and observability features for the
reasoning framework (cortical/reasoning/). It tracks:

- Loop phase transitions and durations
- Decision counts by type
- Question and production metrics
- Crisis events and recovery rates
- Verification pass/fail statistics

The metrics format is compatible with cortical.observability.MetricsCollector,
enabling unified observability across the entire Cortical Text Processor.

Example:
    >>> from cortical.reasoning import CognitiveLoop
    >>> from cortical.reasoning.metrics import ReasoningMetrics
    >>>
    >>> metrics = ReasoningMetrics()
    >>> loop = CognitiveLoop(goal="Test task")
    >>> loop.start()
    >>>
    >>> with metrics.phase_timer(loop.current_phase):
    ...     # Do work in this phase
    ...     pass
    >>>
    >>> metrics.record_decision("architecture")
    >>> print(metrics.get_summary())

Integration with CognitiveLoop:
    The MetricsContextManager provides automatic phase timing and can be
    integrated with CognitiveLoop via the _on_transition callback.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import time

from .cognitive_loop import LoopPhase, LoopTransition, CognitiveLoop


@dataclass
class PhaseMetrics:
    """
    Metrics for a single loop phase.

    Tracks how many times a phase was entered and timing statistics.

    Attributes:
        phase_name: Name of the phase (e.g., "question", "answer")
        entry_count: Number of times this phase was entered
        total_duration_ms: Total time spent in this phase (milliseconds)
        min_duration_ms: Shortest phase duration
        max_duration_ms: Longest phase duration
    """
    phase_name: str
    entry_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0

    def record_entry(self, duration_ms: float) -> None:
        """
        Record a phase entry with its duration.

        Args:
            duration_ms: Duration in milliseconds
        """
        self.entry_count += 1
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)

    def get_average_ms(self) -> float:
        """Get average phase duration in milliseconds."""
        if self.entry_count == 0:
            return 0.0
        return self.total_duration_ms / self.entry_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with observability.py."""
        return {
            'count': self.entry_count,
            'total_ms': self.total_duration_ms,
            'avg_ms': self.get_average_ms(),
            'min_ms': self.min_duration_ms if self.min_duration_ms != float('inf') else 0.0,
            'max_ms': self.max_duration_ms,
        }


@dataclass
class ReasoningMetrics:
    """
    Metrics collector for reasoning loops.

    Provides comprehensive tracking of reasoning loop operations including:
    - Phase transitions and durations
    - Decisions, questions, and production counts
    - Crisis events and recovery statistics
    - Verification pass/fail rates
    - Loop lifecycle metrics

    The metrics format is compatible with cortical.observability.MetricsCollector,
    enabling unified reporting across the processor and reasoning framework.

    Attributes:
        enabled: Whether metrics collection is active
        phases: Per-phase timing and entry statistics
        decisions_made: Total decisions recorded
        decisions_by_type: Decisions grouped by type
        questions_asked: Total questions recorded
        productions_created: Total artifacts produced
        verifications_passed: Successful verification count
        verifications_failed: Failed verification count
        crises_detected: Total crisis events
        crises_recovered: Successfully recovered crises
        loops_started: Total loops started
        loops_completed: Successfully completed loops
        loops_aborted: Loops that were abandoned

    Example:
        >>> metrics = ReasoningMetrics()
        >>> with metrics.phase_timer(LoopPhase.QUESTION):
        ...     # Do work
        ...     pass
        >>> metrics.record_decision("architecture")
        >>> metrics.record_question("What auth method?")
        >>> summary = metrics.get_summary()
    """

    enabled: bool = True

    # Phase tracking
    phases: Dict[str, PhaseMetrics] = field(default_factory=dict)

    # Production tracking
    decisions_made: int = 0
    decisions_by_type: Dict[str, int] = field(default_factory=dict)
    questions_asked: int = 0
    productions_created: int = 0

    # Verification tracking
    verifications_passed: int = 0
    verifications_failed: int = 0

    # Crisis tracking
    crises_detected: int = 0
    crises_recovered: int = 0

    # Loop tracking
    loops_started: int = 0
    loops_completed: int = 0
    loops_aborted: int = 0

    def record_phase_transition(
        self,
        from_phase: Optional[LoopPhase],
        to_phase: LoopPhase,
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Record a phase transition.

        Args:
            from_phase: Phase transitioning from (None if starting)
            to_phase: Phase transitioning to
            duration_ms: Duration spent in from_phase (if applicable)
        """
        if not self.enabled:
            return

        # Record duration for the phase we're leaving
        if from_phase is not None and duration_ms is not None:
            phase_name = from_phase.value
            if phase_name not in self.phases:
                self.phases[phase_name] = PhaseMetrics(phase_name=phase_name)
            self.phases[phase_name].record_entry(duration_ms)

        # Record entry to new phase (with 0 duration for now)
        to_phase_name = to_phase.value
        if to_phase_name not in self.phases:
            self.phases[to_phase_name] = PhaseMetrics(phase_name=to_phase_name)

    def record_decision(self, decision_type: str = "general") -> None:
        """
        Record a decision made during reasoning.

        Args:
            decision_type: Category of decision (e.g., "architecture", "design", "implementation")
        """
        if not self.enabled:
            return

        self.decisions_made += 1
        if decision_type not in self.decisions_by_type:
            self.decisions_by_type[decision_type] = 0
        self.decisions_by_type[decision_type] += 1

    def record_question(self, question_category: str = "general") -> None:
        """
        Record a question raised during reasoning.

        Args:
            question_category: Category of question (e.g., "clarification", "technical", "requirements")
        """
        if not self.enabled:
            return

        self.questions_asked += 1

    def record_production(self, artifact_type: str = "general") -> None:
        """
        Record an artifact produced during reasoning.

        Args:
            artifact_type: Type of artifact (e.g., "code", "test", "documentation")
        """
        if not self.enabled:
            return

        self.productions_created += 1

    def record_verification(self, passed: bool, level: Optional[str] = None) -> None:
        """
        Record a verification result.

        Args:
            passed: Whether verification passed
            level: Optional verification level (e.g., "unit", "integration", "e2e")
        """
        if not self.enabled:
            return

        if passed:
            self.verifications_passed += 1
        else:
            self.verifications_failed += 1

    def record_crisis(self, recovered: bool = False, level: Optional[str] = None) -> None:
        """
        Record a crisis event.

        Args:
            recovered: Whether the crisis was successfully recovered
            level: Optional crisis level (e.g., "hiccup", "obstacle", "wall", "crisis")
        """
        if not self.enabled:
            return

        self.crises_detected += 1
        if recovered:
            self.crises_recovered += 1

    def record_loop_start(self) -> None:
        """Record that a loop was started."""
        if not self.enabled:
            return
        self.loops_started += 1

    def record_loop_complete(self, success: bool = True) -> None:
        """
        Record that a loop completed.

        Args:
            success: Whether loop completed successfully (vs aborted)
        """
        if not self.enabled:
            return

        if success:
            self.loops_completed += 1
        else:
            self.loops_aborted += 1

    def get_verification_pass_rate(self) -> float:
        """
        Calculate verification pass rate.

        Returns:
            Pass rate as a percentage (0.0-100.0), or 0.0 if no verifications
        """
        total = self.verifications_passed + self.verifications_failed
        if total == 0:
            return 0.0
        return (self.verifications_passed / total) * 100.0

    def get_crisis_recovery_rate(self) -> float:
        """
        Calculate crisis recovery rate.

        Returns:
            Recovery rate as a percentage (0.0-100.0), or 0.0 if no crises
        """
        if self.crises_detected == 0:
            return 0.0
        return (self.crises_recovered / self.crises_detected) * 100.0

    def get_loop_completion_rate(self) -> float:
        """
        Calculate loop completion rate.

        Returns:
            Completion rate as a percentage (0.0-100.0), or 0.0 if no loops
        """
        total = self.loops_completed + self.loops_aborted
        if total == 0:
            return 0.0
        return (self.loops_completed / total) * 100.0

    def get_summary(self) -> str:
        """
        Get a human-readable summary of all metrics.

        Returns:
            Formatted string with metrics table
        """
        if not any([
            self.phases,
            self.decisions_made,
            self.questions_asked,
            self.productions_created,
            self.verifications_passed + self.verifications_failed,
            self.crises_detected,
            self.loops_started
        ]):
            return "No metrics collected."

        lines = ["Reasoning Metrics Summary", "=" * 80]

        # Phase metrics
        if self.phases:
            lines.append("\nPhase Transitions:")
            lines.append(f"{'Phase':<15} {'Count':>8} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Total(ms)':>12}")
            lines.append("-" * 80)

            for phase_name in sorted(self.phases.keys()):
                metrics = self.phases[phase_name]
                lines.append(
                    f"{phase_name:<15} {metrics.entry_count:>8} "
                    f"{metrics.get_average_ms():>10.2f} "
                    f"{metrics.min_duration_ms if metrics.min_duration_ms != float('inf') else 0.0:>10.2f} "
                    f"{metrics.max_duration_ms:>10.2f} {metrics.total_duration_ms:>12.2f}"
                )

        # Production metrics
        if any([self.decisions_made, self.questions_asked, self.productions_created]):
            lines.append("\nProduction Metrics:")
            lines.append(f"  Decisions made: {self.decisions_made}")
            if self.decisions_by_type:
                for dec_type, count in sorted(self.decisions_by_type.items()):
                    lines.append(f"    - {dec_type}: {count}")
            lines.append(f"  Questions asked: {self.questions_asked}")
            lines.append(f"  Artifacts produced: {self.productions_created}")

        # Verification metrics
        if self.verifications_passed + self.verifications_failed > 0:
            total_verifications = self.verifications_passed + self.verifications_failed
            pass_rate = self.get_verification_pass_rate()
            lines.append("\nVerification Metrics:")
            lines.append(f"  Passed: {self.verifications_passed}")
            lines.append(f"  Failed: {self.verifications_failed}")
            lines.append(f"  Total: {total_verifications}")
            lines.append(f"  Pass rate: {pass_rate:.1f}%")

        # Crisis metrics
        if self.crises_detected > 0:
            recovery_rate = self.get_crisis_recovery_rate()
            lines.append("\nCrisis Metrics:")
            lines.append(f"  Detected: {self.crises_detected}")
            lines.append(f"  Recovered: {self.crises_recovered}")
            lines.append(f"  Recovery rate: {recovery_rate:.1f}%")

        # Loop metrics
        if self.loops_started > 0:
            completion_rate = self.get_loop_completion_rate()
            lines.append("\nLoop Lifecycle:")
            lines.append(f"  Started: {self.loops_started}")
            lines.append(f"  Completed: {self.loops_completed}")
            lines.append(f"  Aborted: {self.loops_aborted}")
            lines.append(f"  Completion rate: {completion_rate:.1f}%")

        return "\n".join(lines)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get metrics in dictionary format compatible with observability.py.

        Returns:
            Dictionary with all metrics in MetricsCollector-compatible format
        """
        metrics = {}

        # Add phase metrics
        for phase_name, phase_metrics in self.phases.items():
            metrics[f"phase_{phase_name}"] = phase_metrics.to_dict()

        # Add count metrics
        metrics['decisions_made'] = {'count': self.decisions_made}
        metrics['questions_asked'] = {'count': self.questions_asked}
        metrics['productions_created'] = {'count': self.productions_created}
        metrics['verifications_passed'] = {'count': self.verifications_passed}
        metrics['verifications_failed'] = {'count': self.verifications_failed}
        metrics['crises_detected'] = {'count': self.crises_detected}
        metrics['crises_recovered'] = {'count': self.crises_recovered}
        metrics['loops_started'] = {'count': self.loops_started}
        metrics['loops_completed'] = {'count': self.loops_completed}
        metrics['loops_aborted'] = {'count': self.loops_aborted}

        # Add computed metrics
        metrics['verification_pass_rate'] = {'value': self.get_verification_pass_rate()}
        metrics['crisis_recovery_rate'] = {'value': self.get_crisis_recovery_rate()}
        metrics['loop_completion_rate'] = {'value': self.get_loop_completion_rate()}

        return metrics

    def reset(self) -> None:
        """Clear all collected metrics."""
        self.phases.clear()
        self.decisions_made = 0
        self.decisions_by_type.clear()
        self.questions_asked = 0
        self.productions_created = 0
        self.verifications_passed = 0
        self.verifications_failed = 0
        self.crises_detected = 0
        self.crises_recovered = 0
        self.loops_started = 0
        self.loops_completed = 0
        self.loops_aborted = 0

    def enable(self) -> None:
        """Enable metrics collection."""
        self.enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self.enabled = False

    @contextmanager
    def phase_timer(self, phase: LoopPhase):
        """
        Context manager for timing a phase.

        Args:
            phase: The phase being timed

        Example:
            >>> metrics = ReasoningMetrics()
            >>> with metrics.phase_timer(LoopPhase.QUESTION):
            ...     # Do work in QUESTION phase
            ...     pass
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            phase_name = phase.value
            if phase_name not in self.phases:
                self.phases[phase_name] = PhaseMetrics(phase_name=phase_name)
            self.phases[phase_name].record_entry(duration_ms)


class MetricsContextManager:
    """
    Context manager for timing operations with automatic metrics recording.

    This provides a convenient way to time blocks of code and automatically
    record the results to a ReasoningMetrics instance.

    Attributes:
        metrics: ReasoningMetrics instance to record to
        phase: Phase being timed

    Example:
        >>> metrics = ReasoningMetrics()
        >>> ctx = MetricsContextManager(metrics, LoopPhase.ANSWER)
        >>> with ctx:
        ...     # Do work
        ...     pass
    """

    def __init__(self, metrics: ReasoningMetrics, phase: LoopPhase):
        """
        Initialize the context manager.

        Args:
            metrics: ReasoningMetrics instance to record to
            phase: The phase being timed
        """
        self.metrics = metrics
        self.phase = phase
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000.0
            phase_name = self.phase.value
            if phase_name not in self.metrics.phases:
                self.metrics.phases[phase_name] = PhaseMetrics(phase_name=phase_name)
            self.metrics.phases[phase_name].record_entry(duration_ms)


def create_loop_metrics_handler(metrics: ReasoningMetrics):
    """
    Create a handler for CognitiveLoop transitions that records metrics.

    This can be registered with CognitiveLoopManager to automatically collect
    metrics for all loop transitions.

    Args:
        metrics: ReasoningMetrics instance to record to

    Returns:
        Handler function suitable for CognitiveLoopManager.register_transition_handler

    Example:
        >>> from cortical.reasoning import CognitiveLoopManager
        >>> from cortical.reasoning.metrics import ReasoningMetrics, create_loop_metrics_handler
        >>>
        >>> metrics = ReasoningMetrics()
        >>> manager = CognitiveLoopManager()
        >>> manager.register_transition_handler(create_loop_metrics_handler(metrics))
        >>>
        >>> # Now all loop transitions are automatically tracked
        >>> loop = manager.create_loop("Test goal")
        >>> loop.start()
    """
    def handler(loop: CognitiveLoop, transition: LoopTransition) -> None:
        """Handle loop transition by recording metrics."""
        # Calculate duration if transitioning from a phase
        duration_ms = None
        if transition.from_phase is not None and loop.phase_contexts:
            # Find the context for the phase we're leaving
            for ctx in reversed(loop.phase_contexts):
                if ctx.phase == transition.from_phase and ctx.ended_at is None:
                    # Calculate duration
                    duration_s = (transition.timestamp - ctx.started_at).total_seconds()
                    duration_ms = duration_s * 1000.0
                    break

        # Record the transition
        metrics.record_phase_transition(
            from_phase=transition.from_phase,
            to_phase=transition.to_phase,
            duration_ms=duration_ms
        )

    return handler
