"""
Progress reporting infrastructure for long-running operations.

This module provides a flexible progress reporting system that supports:
- Console output with nice formatting
- Custom callbacks for integration with UIs
- Optional ETA estimation
- Phase-based progress tracking
"""

import sys
import time
from typing import Protocol, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod


class ProgressReporter(Protocol):
    """Protocol for progress reporters.

    Implementations must provide update() and complete() methods.
    """

    def update(self, phase: str, percent: float, message: Optional[str] = None) -> None:
        """
        Update progress for a specific phase.

        Args:
            phase: Name of the current phase (e.g., "Computing TF-IDF")
            percent: Progress percentage (0.0 to 100.0)
            message: Optional additional message to display
        """
        ...

    def complete(self, phase: str, message: Optional[str] = None) -> None:
        """
        Mark a phase as complete.

        Args:
            phase: Name of the completed phase
            message: Optional completion message
        """
        ...


class ConsoleProgressReporter:
    """
    Console-based progress reporter with nice formatting.

    Displays progress with in-place updates using carriage returns.

    Example output:
        Computing TF-IDF... [████████████████----] 75% (ETA: 5s)
    """

    def __init__(
        self,
        file=None,
        width: int = 40,
        show_eta: bool = True,
        use_unicode: bool = True
    ):
        """
        Initialize console progress reporter.

        Args:
            file: Output file (default: sys.stderr)
            width: Width of progress bar in characters
            show_eta: Whether to show estimated time remaining
            use_unicode: Use Unicode block characters for progress bar
        """
        self.file = file or sys.stderr
        self.width = width
        self.show_eta = show_eta
        self.use_unicode = use_unicode

        # Tracking for ETA calculation
        self._phase_start_times: Dict[str, float] = {}
        self._last_phase: Optional[str] = None

        # Characters for progress bar
        if use_unicode:
            self.fill_char = '█'
            self.empty_char = '░'
        else:
            self.fill_char = '#'
            self.empty_char = '-'

    def update(self, phase: str, percent: float, message: Optional[str] = None) -> None:
        """
        Update progress display.

        Args:
            phase: Name of the current phase
            percent: Progress percentage (0.0 to 100.0)
            message: Optional additional message
        """
        # Track phase start time for ETA
        if phase != self._last_phase:
            self._phase_start_times[phase] = time.time()
            self._last_phase = phase

        # Clamp percentage
        percent = max(0.0, min(100.0, percent))

        # Build progress bar
        filled = int(self.width * percent / 100.0)
        bar = self.fill_char * filled + self.empty_char * (self.width - filled)

        # Build status line
        status = f"\r{phase}... [{bar}] {percent:.0f}%"

        # Add ETA if enabled
        if self.show_eta and percent > 0 and percent < 100:
            eta = self._estimate_eta(phase, percent)
            if eta is not None:
                status += f" (ETA: {eta:.0f}s)"

        # Add custom message if provided
        if message:
            status += f" - {message}"

        # Write with carriage return for in-place update
        self.file.write(status)
        self.file.flush()

    def complete(self, phase: str, message: Optional[str] = None) -> None:
        """
        Mark phase as complete and move to new line.

        Args:
            phase: Name of the completed phase
            message: Optional completion message
        """
        # Show 100% complete
        bar = self.fill_char * self.width
        status = f"\r{phase}... [{bar}] 100%"

        # Add elapsed time
        if phase in self._phase_start_times:
            elapsed = time.time() - self._phase_start_times[phase]
            status += f" ({elapsed:.1f}s)"

        # Add custom message if provided
        if message:
            status += f" - {message}"

        # Write final status and newline
        self.file.write(status + "\n")
        self.file.flush()

        # Clean up tracking
        self._phase_start_times.pop(phase, None)

    def _estimate_eta(self, phase: str, percent: float) -> Optional[float]:
        """
        Estimate time remaining for current phase.

        Args:
            phase: Current phase name
            percent: Current progress percentage

        Returns:
            Estimated seconds remaining, or None if not calculable
        """
        if phase not in self._phase_start_times or percent <= 0:
            return None

        elapsed = time.time() - self._phase_start_times[phase]
        if elapsed < 1.0:  # Wait at least 1 second for reasonable estimate
            return None

        # Linear extrapolation
        total_estimated = elapsed / (percent / 100.0)
        remaining = total_estimated - elapsed

        return max(0.0, remaining)


class CallbackProgressReporter:
    """
    Progress reporter that calls a custom callback function.

    Useful for integrating with UI frameworks, logging systems, etc.

    Example:
        >>> def my_callback(phase, percent, message):
        ...     print(f"{phase}: {percent}% - {message}")
        >>> reporter = CallbackProgressReporter(my_callback)
        >>> reporter.update("Processing", 50.0, "halfway done")
        Processing: 50.0% - halfway done
    """

    def __init__(self, callback: Callable[[str, float, Optional[str]], None]):
        """
        Initialize callback-based progress reporter.

        Args:
            callback: Function to call with (phase, percent, message) arguments
        """
        self.callback = callback

    def update(self, phase: str, percent: float, message: Optional[str] = None) -> None:
        """
        Call callback with progress update.

        Args:
            phase: Name of the current phase
            percent: Progress percentage (0.0 to 100.0)
            message: Optional additional message
        """
        self.callback(phase, percent, message)

    def complete(self, phase: str, message: Optional[str] = None) -> None:
        """
        Call callback with completion notification.

        Args:
            phase: Name of the completed phase
            message: Optional completion message
        """
        self.callback(phase, 100.0, message or "Complete")


class SilentProgressReporter:
    """
    No-op progress reporter for silent operation.

    Used as default when no progress reporting is needed.
    """

    def update(self, phase: str, percent: float, message: Optional[str] = None) -> None:
        """Do nothing."""
        pass

    def complete(self, phase: str, message: Optional[str] = None) -> None:
        """Do nothing."""
        pass


class MultiPhaseProgress:
    """
    Helper for tracking progress across multiple sequential phases.

    Automatically calculates overall percentage based on phase weights.

    Example:
        >>> phases = {
        ...     "Phase 1": 30,  # 30% of total time
        ...     "Phase 2": 50,  # 50% of total time
        ...     "Phase 3": 20   # 20% of total time
        ... }
        >>> progress = MultiPhaseProgress(reporter, phases)
        >>> progress.start_phase("Phase 1")
        >>> progress.update(50)  # 50% of Phase 1 = 15% overall
        >>> progress.complete_phase()
        >>> progress.start_phase("Phase 2")
        >>> progress.update(100)  # 100% of Phase 2 = 80% overall
    """

    def __init__(
        self,
        reporter: ProgressReporter,
        phases: Dict[str, float],
        normalize: bool = True
    ):
        """
        Initialize multi-phase progress tracker.

        Args:
            reporter: Progress reporter to use
            phases: Dict mapping phase names to relative weights
            normalize: Whether to normalize weights to sum to 100
        """
        self.reporter = reporter
        self.phases = phases.copy()

        # Normalize weights if requested
        if normalize:
            total = sum(phases.values())
            if total > 0:
                self.phases = {k: v / total * 100 for k, v in phases.items()}

        # Calculate cumulative offsets for each phase
        self._phase_offsets: Dict[str, float] = {}
        cumulative = 0.0
        for phase, weight in self.phases.items():
            self._phase_offsets[phase] = cumulative
            cumulative += weight

        self._current_phase: Optional[str] = None
        self._overall_progress: float = 0.0

    def start_phase(self, phase: str) -> None:
        """
        Start a new phase.

        Args:
            phase: Name of the phase to start

        Raises:
            ValueError: If phase name is not in the configured phases
        """
        if phase not in self.phases:
            raise ValueError(f"Unknown phase: {phase}")

        self._current_phase = phase
        self._overall_progress = self._phase_offsets[phase]
        self.reporter.update(phase, 0.0)

    def update(self, percent: float, message: Optional[str] = None) -> None:
        """
        Update progress within current phase.

        Args:
            percent: Progress percentage within current phase (0-100)
            message: Optional status message
        """
        if self._current_phase is None:
            return

        phase_weight = self.phases[self._current_phase]
        phase_offset = self._phase_offsets[self._current_phase]

        # Calculate overall progress
        self._overall_progress = phase_offset + (percent / 100.0 * phase_weight)

        self.reporter.update(
            self._current_phase,
            percent,
            message
        )

    def complete_phase(self, message: Optional[str] = None) -> None:
        """
        Mark current phase as complete.

        Args:
            message: Optional completion message
        """
        if self._current_phase is None:
            return

        self.reporter.complete(self._current_phase, message)
        self._current_phase = None

    @property
    def overall_progress(self) -> float:
        """Get overall progress across all phases (0-100)."""
        return self._overall_progress
