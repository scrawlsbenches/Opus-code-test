"""
QAPV Cycle Behavioral Verification.

This module provides behavioral verification for QAPV (Question→Answer→Produce→Verify)
cognitive loops, detecting anomalies and tracking cycle health metrics.

The verifier ensures that QAPV cycles follow correct patterns and identifies issues like:
- Skipped phases (e.g., QUESTION → PRODUCE without ANSWER)
- Infinite loops (cycling without progress)
- Stuck phases (spending too long in a single phase)
- Invalid transitions (transitions not allowed by the state machine)
- Premature exits (completing without verification)
- Missing production (verification without production)

Integration:
    >>> from cortical.reasoning import CognitiveLoop, LoopPhase
    >>> from cortical.reasoning.qapv_verification import QAPVVerifier
    >>>
    >>> verifier = QAPVVerifier()
    >>> loop = CognitiveLoop(goal="Test")
    >>> loop.start(LoopPhase.QUESTION)
    >>>
    >>> # Record transitions
    >>> for transition in loop.transitions:
    >>>     verifier.record_transition(
    >>>         transition.from_phase.value if transition.from_phase else None,
    >>>         transition.to_phase.value
    >>>     )
    >>>
    >>> # Check health
    >>> anomalies = verifier.check_health()
    >>> if anomalies:
    >>>     for anomaly in anomalies:
    >>>         print(f"{anomaly.severity.upper()}: {anomaly.description}")

See Also:
    - cognitive_loop.py: Core QAPV loop implementation
    - loop_validator.py: Generic loop validation
    - crisis_manager.py: Crisis detection and recovery
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import time


class QAPVAnomaly(Enum):
    """Types of behavioral anomalies in QAPV cycles."""
    PHASE_SKIP = "phase_skip"
    INFINITE_LOOP = "infinite_loop"
    STUCK_PHASE = "stuck_phase"
    INVALID_TRANSITION = "invalid_transition"
    PREMATURE_EXIT = "premature_exit"
    MISSING_PRODUCTION = "missing_production"


@dataclass
class TransitionEvent:
    """
    Record of a single phase transition.

    Attributes:
        from_phase: Phase transitioning from (None for initial)
        to_phase: Phase transitioning to
        timestamp: When the transition occurred (seconds since epoch)
    """
    from_phase: Optional[str]
    to_phase: str
    timestamp: float


@dataclass
class AnomalyReport:
    """
    Report of a detected behavioral anomaly.

    Attributes:
        anomaly_type: Type of anomaly detected
        description: Human-readable description of the issue
        severity: Severity level (low, medium, high, critical)
        transition_history: Relevant transitions that led to this anomaly
        suggestions: List of actionable suggestions to fix the issue
    """
    anomaly_type: QAPVAnomaly
    description: str
    severity: str  # low, medium, high, critical
    transition_history: List[TransitionEvent] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class QAPVVerifier:
    """
    Behavioral verification for QAPV cycles.

    This class tracks phase transitions and detects behavioral anomalies
    in QAPV cognitive loops. It implements a state machine for valid
    transitions and provides diagnostic reports.

    Valid QAPV Transitions:
        QUESTION → ANSWER         (begin answering)
        ANSWER → PRODUCE          (implement solution)
        ANSWER → QUESTION         (need clarification)
        PRODUCE → VERIFY          (test implementation)
        VERIFY → QUESTION         (found issues, new cycle)
        VERIFY → COMPLETE         (all tests passed)

    Attributes:
        stuck_threshold_seconds: Time in one phase before warning
        max_cycles_before_warning: Complete cycles before warning

    Example:
        >>> verifier = QAPVVerifier(
        ...     stuck_threshold_seconds=120.0,
        ...     max_cycles_before_warning=10
        ... )
        >>> verifier.record_transition(None, "question")
        >>> verifier.record_transition("question", "answer")
        >>> verifier.record_transition("answer", "produce")
        >>>
        >>> # Check for anomalies
        >>> anomalies = verifier.check_health()
        >>> if anomalies:
        ...     report = verifier.get_diagnostic_report()
        ...     print(f"Total anomalies: {report['total_anomalies']}")
    """

    # Valid phase transitions (lowercase for compatibility)
    VALID_TRANSITIONS: Dict[str, Set[str]] = {
        'question': {'answer'},
        'answer': {'produce', 'question'},  # Can loop back for clarification
        'produce': {'verify'},
        'verify': {'question', 'complete'},  # Can start new cycle or complete
    }

    def __init__(self,
                 stuck_threshold_seconds: float = 60.0,
                 max_cycles_before_warning: int = 10):
        """
        Initialize the QAPV verifier.

        Args:
            stuck_threshold_seconds: Seconds in one phase before stuck warning
            max_cycles_before_warning: Complete QAPV cycles before infinite loop warning
        """
        self.stuck_threshold_seconds = stuck_threshold_seconds
        self.max_cycles_before_warning = max_cycles_before_warning

        self._transitions: List[TransitionEvent] = []
        self._phase_enter_times: Dict[str, float] = {}
        self._current_phase: Optional[str] = None
        self._anomalies_cache: Optional[List[AnomalyReport]] = None

    def record_transition(self, from_phase: Optional[str], to_phase: str) -> None:
        """
        Record a phase transition for analysis.

        Args:
            from_phase: Phase transitioning from (None for initial transition)
            to_phase: Phase transitioning to

        Example:
            >>> verifier.record_transition(None, "question")  # Start
            >>> verifier.record_transition("question", "answer")
            >>> verifier.record_transition("answer", "produce")
        """
        timestamp = time.time()

        # Normalize to lowercase
        from_phase_norm = from_phase.lower() if from_phase else None
        to_phase_norm = to_phase.lower()

        event = TransitionEvent(
            from_phase=from_phase_norm,
            to_phase=to_phase_norm,
            timestamp=timestamp
        )
        self._transitions.append(event)

        # Update current phase tracking
        self._current_phase = to_phase_norm
        self._phase_enter_times[to_phase_norm] = timestamp

        # Clear anomalies cache on new transition
        self._anomalies_cache = None

    def is_transition_valid(self, from_phase: Optional[str], to_phase: str) -> bool:
        """
        Check if a transition is valid according to the QAPV state machine.

        Args:
            from_phase: Phase transitioning from (None for initial)
            to_phase: Phase transitioning to

        Returns:
            True if transition is valid, False otherwise

        Example:
            >>> verifier.is_transition_valid("question", "answer")
            True
            >>> verifier.is_transition_valid("question", "verify")
            False
        """
        # Normalize to lowercase
        from_phase_norm = from_phase.lower() if from_phase else None
        to_phase_norm = to_phase.lower()

        # Initial transition is always valid
        if from_phase_norm is None:
            return True

        # Check if from_phase is a known phase
        if from_phase_norm not in self.VALID_TRANSITIONS:
            return False

        # Check if transition is in the valid set
        valid_targets = self.VALID_TRANSITIONS[from_phase_norm]
        return to_phase_norm in valid_targets

    def check_health(self) -> List[AnomalyReport]:
        """
        Check for behavioral anomalies in the transition history.

        Detects:
        - Invalid transitions (not allowed by state machine)
        - Phase skips (e.g., QUESTION → PRODUCE)
        - Stuck phases (too long in one phase)
        - Infinite loops (too many complete cycles)
        - Premature exits (completed without verification)
        - Missing production (verified without producing)

        Returns:
            List of detected anomalies, empty if healthy

        Example:
            >>> verifier.record_transition(None, "question")
            >>> verifier.record_transition("question", "verify")  # Invalid!
            >>> anomalies = verifier.check_health()
            >>> len(anomalies) > 0
            True
            >>> anomalies[0].anomaly_type
            <QAPVAnomaly.INVALID_TRANSITION: 'invalid_transition'>
        """
        # Return cached results if available
        if self._anomalies_cache is not None:
            return self._anomalies_cache

        anomalies: List[AnomalyReport] = []

        # Check for invalid transitions
        anomalies.extend(self._check_invalid_transitions())

        # Check for stuck phases
        anomalies.extend(self._check_stuck_phases())

        # Check for infinite loops
        anomalies.extend(self._check_infinite_loops())

        # Check for premature exits
        anomalies.extend(self._check_premature_exits())

        # Check for missing production
        anomalies.extend(self._check_missing_production())

        # Cache results
        self._anomalies_cache = anomalies

        return anomalies

    def get_cycle_count(self) -> int:
        """
        Count complete QAPV cycles (QUESTION → ANSWER → PRODUCE → VERIFY).

        A complete cycle is defined as visiting all four phases in order.
        Loops back to QUESTION after VERIFY start a new cycle.

        Returns:
            Number of complete QAPV cycles

        Example:
            >>> verifier.record_transition(None, "question")
            >>> verifier.record_transition("question", "answer")
            >>> verifier.record_transition("answer", "produce")
            >>> verifier.record_transition("produce", "verify")
            >>> verifier.get_cycle_count()
            1
        """
        if not self._transitions:
            return 0

        cycle_count = 0
        phases_in_cycle = set()
        required_phases = {'question', 'answer', 'produce', 'verify'}

        for event in self._transitions:
            phases_in_cycle.add(event.to_phase)

            # Check if we completed a cycle (all phases visited)
            if phases_in_cycle >= required_phases:
                cycle_count += 1
                phases_in_cycle = {event.to_phase}  # Start new cycle

        return cycle_count

    def get_diagnostic_report(self) -> Dict:
        """
        Generate comprehensive diagnostic report.

        Returns:
            Dictionary with:
            - total_transitions: Number of phase transitions
            - current_phase: Current phase (or None)
            - cycle_count: Number of complete QAPV cycles
            - total_anomalies: Number of detected anomalies
            - anomalies_by_type: Count of each anomaly type
            - anomalies: List of anomaly reports
            - health_status: Overall health (healthy, warning, critical)

        Example:
            >>> report = verifier.get_diagnostic_report()
            >>> print(f"Health: {report['health_status']}")
            >>> print(f"Cycles: {report['cycle_count']}")
            >>> for anomaly in report['anomalies']:
            ...     print(f"  - {anomaly.description}")
        """
        anomalies = self.check_health()

        # Count anomalies by type
        anomalies_by_type = {}
        for anomaly in anomalies:
            type_name = anomaly.anomaly_type.value
            anomalies_by_type[type_name] = anomalies_by_type.get(type_name, 0) + 1

        # Determine overall health status
        if not anomalies:
            health_status = "healthy"
        elif any(a.severity == "critical" for a in anomalies):
            health_status = "critical"
        elif any(a.severity == "high" for a in anomalies):
            health_status = "warning"
        else:
            health_status = "minor_issues"

        return {
            'total_transitions': len(self._transitions),
            'current_phase': self._current_phase,
            'cycle_count': self.get_cycle_count(),
            'total_anomalies': len(anomalies),
            'anomalies_by_type': anomalies_by_type,
            'anomalies': anomalies,
            'health_status': health_status,
        }

    def reset(self) -> None:
        """
        Reset verification state.

        Clears all transition history and anomaly tracking.
        Use this when starting verification of a new loop.

        Example:
            >>> verifier.record_transition(None, "question")
            >>> len(verifier._transitions)
            1
            >>> verifier.reset()
            >>> len(verifier._transitions)
            0
        """
        self._transitions = []
        self._phase_enter_times = {}
        self._current_phase = None
        self._anomalies_cache = None

    # =========================================================================
    # PRIVATE ANOMALY DETECTION METHODS
    # =========================================================================

    def _check_invalid_transitions(self) -> List[AnomalyReport]:
        """Check for transitions not allowed by the state machine."""
        anomalies = []

        for event in self._transitions:
            if not self.is_transition_valid(event.from_phase, event.to_phase):
                from_label = event.from_phase or "(start)"
                anomalies.append(AnomalyReport(
                    anomaly_type=QAPVAnomaly.INVALID_TRANSITION,
                    description=f"Invalid transition: {from_label} → {event.to_phase}",
                    severity="high",
                    transition_history=[event],
                    suggestions=[
                        f"From {from_label}, valid transitions are: "
                        f"{', '.join(self.VALID_TRANSITIONS.get(event.from_phase or '', ['any']))}"
                    ]
                ))

        return anomalies

    def _check_stuck_phases(self) -> List[AnomalyReport]:
        """Check for phases that have exceeded the stuck threshold."""
        anomalies = []
        current_time = time.time()

        if self._current_phase and self._current_phase in self._phase_enter_times:
            enter_time = self._phase_enter_times[self._current_phase]
            elapsed = current_time - enter_time

            if elapsed > self.stuck_threshold_seconds:
                # Find all transitions to/from current phase
                relevant_transitions = [
                    t for t in self._transitions
                    if t.to_phase == self._current_phase or t.from_phase == self._current_phase
                ]

                anomalies.append(AnomalyReport(
                    anomaly_type=QAPVAnomaly.STUCK_PHASE,
                    description=f"Stuck in {self._current_phase} phase for {elapsed:.1f} seconds "
                                f"(threshold: {self.stuck_threshold_seconds:.1f}s)",
                    severity="medium",
                    transition_history=relevant_transitions[-3:],  # Last 3 relevant
                    suggestions=[
                        f"Consider if {self._current_phase} phase is blocked on external dependency",
                        f"Review if {self._current_phase} work can be broken into smaller chunks",
                        "Check if transition criteria are too strict"
                    ]
                ))

        return anomalies

    def _check_infinite_loops(self) -> List[AnomalyReport]:
        """Check for too many complete cycles without completion."""
        anomalies = []
        cycle_count = self.get_cycle_count()

        if cycle_count >= self.max_cycles_before_warning:
            # Check if we're still active (not completed)
            last_phase = self._transitions[-1].to_phase if self._transitions else None
            if last_phase != 'complete':
                anomalies.append(AnomalyReport(
                    anomaly_type=QAPVAnomaly.INFINITE_LOOP,
                    description=f"Completed {cycle_count} QAPV cycles without finishing "
                                f"(threshold: {self.max_cycles_before_warning})",
                    severity="critical",
                    transition_history=self._transitions[-10:],  # Last 10 transitions
                    suggestions=[
                        "Review if problem scope is too large and should be decomposed",
                        "Check if acceptance criteria are achievable",
                        "Consider if requirements keep changing (scope creep)",
                        "May need to escalate to human for decision"
                    ]
                ))

        return anomalies

    def _check_premature_exits(self) -> List[AnomalyReport]:
        """Check for completing without verification."""
        anomalies = []

        if not self._transitions:
            return anomalies

        # Check if completed from a phase other than VERIFY
        if self._transitions[-1].to_phase == 'complete':
            previous_phase = self._transitions[-1].from_phase
            if previous_phase != 'verify':
                anomalies.append(AnomalyReport(
                    anomaly_type=QAPVAnomaly.PREMATURE_EXIT,
                    description=f"Completed from {previous_phase} phase without verification",
                    severity="high",
                    transition_history=self._transitions[-3:],
                    suggestions=[
                        "Always verify implementation before completing",
                        "Add verification phase to confirm correctness",
                        "Run tests before marking work as done"
                    ]
                ))

        return anomalies

    def _check_missing_production(self) -> List[AnomalyReport]:
        """Check for verification without production."""
        anomalies = []

        # Track if we've seen PRODUCE before VERIFY in current cycle
        seen_produce = False

        for event in self._transitions:
            if event.to_phase == 'produce':
                seen_produce = True
            elif event.to_phase == 'verify' and not seen_produce:
                anomalies.append(AnomalyReport(
                    anomaly_type=QAPVAnomaly.MISSING_PRODUCTION,
                    description="Entered VERIFY phase without producing artifacts",
                    severity="high",
                    transition_history=[event],
                    suggestions=[
                        "Verify phase requires artifacts from produce phase",
                        "Ensure produce phase creates testable artifacts",
                        "Check if transition to verify was premature"
                    ]
                ))
            elif event.to_phase == 'question':
                # New cycle starts, reset produce tracking
                seen_produce = False

        return anomalies
