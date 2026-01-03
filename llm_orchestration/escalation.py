"""
Escalation protocol for Director-Worker coordination.

Provides formal escalation mechanisms for handling worker confusion:
- EscalationLevel: Severity levels for escalation
- EscalationProtocol: Formal protocol with recommended actions
- EscalationManager: Manages evaluation and execution of escalation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from .recovery import ConfusionSignal


# Import ConfusionRecord from agents to avoid circular import
# We'll define it here to make the module standalone
@dataclass
class ConfusionRecord:
    """Record of a confusion detection event during execution."""

    signal_type: str
    severity: str
    recovery_action: str
    recovered: bool
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class EscalationLevel(Enum):
    """Escalation severity levels for worker confusion."""
    NONE = 0        # No escalation needed
    MONITOR = 1     # Increased monitoring
    INTERVENE = 2   # Director intervention needed
    REASSIGN = 3    # Reassign task to different worker
    ESCALATE = 4    # Escalate to higher authority
    ABORT = 5       # Abort task entirely


@dataclass
class EscalationProtocol:
    """
    Protocol for handling worker confusion escalation.

    Defines the escalation level, reason, and recommended action
    based on worker confusion history and severity.
    """
    level: EscalationLevel
    reason: str
    worker_id: str
    task_id: str
    confusion_history: List[ConfusionRecord]
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.name,
            "reason": self.reason,
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "confusion_count": len(self.confusion_history),
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
        }


class EscalationManager:
    """
    Manages escalation of worker confusion for Directors.

    The EscalationManager implements a formal escalation protocol for handling
    worker confusion signals. It uses a three-strikes policy combined with
    severity-based escalation to determine appropriate intervention levels.

    Features:
        - **Confusion Tracking**: Per-worker confusion history
          - Tracks all confusion signals
          - Records recovery attempts
          - Maintains strike count

        - **Severity-Based Escalation**: Graduated response
          - Severity inferred from confusion confidence
          - Escalation level increases with count and severity
          - Three-strikes policy for worker reassignment

        - **Protocol Execution**: Automated intervention
          - Generates recommended actions
          - Executes escalation protocols
          - Records outcomes for learning

        - **History Recording**: Learning from patterns
          - Maintains escalation history
          - Tracks worker-specific patterns
          - Enables pattern analysis

    Escalation Levels (in order of severity):
        0. **NONE**: No escalation needed
        1. **MONITOR**: Increased monitoring, no intervention
        2. **INTERVENE**: Director provides guidance/hints
        3. **REASSIGN**: Task reassigned to different worker
        4. **ESCALATE**: Escalate to higher authority
        5. **ABORT**: Abort task entirely

    Escalation Rules Matrix:
        ```
        Confusion Count | LOW         | MEDIUM      | HIGH        | CRITICAL
        ─────────────────────────────────────────────────────────────────────
        1st confusion   | MONITOR     | MONITOR     | INTERVENE   | INTERVENE
        2nd confusion   | INTERVENE   | REASSIGN    | ESCALATE    | ESCALATE
        3rd+ confusion  | ABORT       | ABORT       | ABORT       | ABORT
        ```

    Three-Strikes Policy:
        - Strike 1: Monitor or intervene based on severity
        - Strike 2: Escalate response (intervene → reassign)
        - Strike 3: Abort task (worker unable to complete)

    Example:
        >>> from llm_orchestration.escalation import (
        ...     EscalationManager,
        ...     EscalationLevel,
        ... )
        >>> from llm_orchestration.recovery import ConfusionSignal
        >>>
        >>> # Create escalation manager
        >>> manager = EscalationManager()
        >>>
        >>> # Worker reports confusion
        >>> confusion = ConfusionSignal(
        ...     signal_type="repetition_loop",
        ...     description="Repeating same failed approach",
        ...     evidence=["read_file", "read_file", "read_file"],
        ...     confidence=0.85,  # HIGH severity
        ...     source="worker-1",
        ... )
        >>>
        >>> # Evaluate escalation
        >>> protocol = manager.evaluate(
        ...     worker_id="worker-1",
        ...     confusion=confusion,
        ...     task_id="T-123",
        ... )
        >>>
        >>> print(f"Level: {protocol.level.name}")  # INTERVENE
        >>> print(f"Action: {protocol.recommended_action}")
        >>>
        >>> # Execute protocol
        >>> if protocol.level.value >= EscalationLevel.INTERVENE.value:
        ...     result = manager.execute(protocol)
        ...     print(f"Executed: {result}")

    Attributes:
        ESCALATION_RULES (dict): Matrix mapping (count, severity) → level

    Private Attributes:
        _escalation_history: History of all escalation protocols
        _worker_strikes: Strike count per worker
        _worker_confusion_history: Confusion records per worker

    See Also:
        EscalationLevel: Severity levels for escalation
        EscalationProtocol: Escalation response protocol
        ConfusionSignal: Worker confusion indicators
        Director: Uses EscalationManager to handle worker confusion
    """

    # Escalation rules: (confusion_count, severity) -> level
    ESCALATION_RULES = {
        # First confusion - monitor or intervene based on severity
        (1, "LOW"): EscalationLevel.MONITOR,
        (1, "MEDIUM"): EscalationLevel.MONITOR,
        (1, "HIGH"): EscalationLevel.INTERVENE,
        (1, "CRITICAL"): EscalationLevel.INTERVENE,

        # Second confusion - escalate based on severity
        (2, "LOW"): EscalationLevel.INTERVENE,
        (2, "MEDIUM"): EscalationLevel.REASSIGN,
        (2, "HIGH"): EscalationLevel.ESCALATE,
        (2, "CRITICAL"): EscalationLevel.ESCALATE,

        # Third or more confusions - abort or escalate
        (3, "any"): EscalationLevel.ABORT,
    }

    def __init__(self):
        """Initialize the escalation manager."""
        self._escalation_history: List[EscalationProtocol] = []
        self._worker_strikes: Dict[str, int] = {}  # worker_id -> strike count
        self._worker_confusion_history: Dict[str, List[ConfusionRecord]] = {}

    def evaluate(
        self,
        worker_id: str,
        confusion: ConfusionSignal,
        task_id: str = "unknown"
    ) -> EscalationProtocol:
        """
        Determine appropriate escalation level for worker confusion.

        Args:
            worker_id: ID of the confused worker
            confusion: Confusion signal from worker
            task_id: ID of the task being worked on

        Returns:
            EscalationProtocol with recommended action
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get or create confusion history for this worker
        if worker_id not in self._worker_confusion_history:
            self._worker_confusion_history[worker_id] = []

        # Create confusion record
        confusion_record = ConfusionRecord(
            signal_type=confusion.signal_type,
            severity=self._infer_severity_from_confidence(confusion.confidence),
            recovery_action="pending",
            recovered=False,
            details={
                "description": confusion.description,
                "evidence": confusion.evidence,
                "confidence": confusion.confidence,
                "source": confusion.source,
            }
        )

        # Add to history
        self._worker_confusion_history[worker_id].append(confusion_record)

        # Increment strike count
        self._worker_strikes[worker_id] = self._worker_strikes.get(worker_id, 0) + 1

        # Get confusion history for this worker
        history = self._worker_confusion_history[worker_id]
        confusion_count = len(history)

        # Determine severity level
        severity = confusion_record.severity

        # Look up escalation level based on rules
        level = self._determine_escalation_level(confusion_count, severity)

        # Generate recommended action
        action = self._generate_action(level, worker_id, confusion)

        # Create protocol
        protocol = EscalationProtocol(
            level=level,
            reason=f"Worker {worker_id} confusion: {confusion.description}",
            worker_id=worker_id,
            task_id=task_id,
            confusion_history=history.copy(),
            recommended_action=action,
        )

        logger.info(
            f"Escalation evaluated: {level.name} for worker {worker_id} "
            f"(confusion count: {confusion_count}, severity: {severity})"
        )

        return protocol

    def _infer_severity_from_confidence(self, confidence: float) -> str:
        """Infer severity level from confusion confidence score."""
        if confidence >= 0.9:
            return "CRITICAL"
        elif confidence >= 0.7:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _determine_escalation_level(
        self,
        confusion_count: int,
        severity: str
    ) -> EscalationLevel:
        """
        Determine escalation level based on confusion count and severity.

        Args:
            confusion_count: Number of confusion signals from worker
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)

        Returns:
            Appropriate EscalationLevel
        """
        # Three or more confusions - always abort
        if confusion_count >= 3:
            return EscalationLevel.ABORT

        # Look up in rules table
        rule_key = (confusion_count, severity)
        if rule_key in self.ESCALATION_RULES:
            return self.ESCALATION_RULES[rule_key]

        # Default fallback
        if confusion_count == 1:
            return EscalationLevel.MONITOR
        elif confusion_count == 2:
            return EscalationLevel.INTERVENE
        else:
            return EscalationLevel.ABORT

    def _generate_action(
        self,
        level: EscalationLevel,
        worker_id: str,
        confusion: ConfusionSignal
    ) -> str:
        """
        Generate recommended action for escalation level.

        Args:
            level: Escalation level
            worker_id: ID of confused worker
            confusion: Confusion signal

        Returns:
            Recommended action string
        """
        actions = {
            EscalationLevel.NONE: "Continue normal operation",
            EscalationLevel.MONITOR: (
                f"Increase monitoring for worker {worker_id}. "
                "Reduce batch size and log all actions."
            ),
            EscalationLevel.INTERVENE: (
                f"Pause worker {worker_id}. "
                f"Analyze state: {confusion.description}. "
                "Provide guidance or restore from checkpoint."
            ),
            EscalationLevel.REASSIGN: (
                f"Reassign task from worker {worker_id} to different worker. "
                f"Blacklist {worker_id} for this task type. "
                f"Reason: {confusion.description}"
            ),
            EscalationLevel.ESCALATE: (
                f"Escalate to higher authority. "
                f"Worker {worker_id} cannot complete task. "
                f"Signal: {confusion.signal_type}. "
                "Request human review or fallback strategy."
            ),
            EscalationLevel.ABORT: (
                f"Abort task for worker {worker_id}. "
                f"Too many confusion signals ({self._worker_strikes.get(worker_id, 0)}). "
                "Create failure record and trigger learning capture."
            ),
        }

        return actions.get(level, "No action defined")

    def execute(self, protocol: EscalationProtocol) -> bool:
        """
        Execute the escalation protocol.

        Args:
            protocol: The escalation protocol to execute

        Returns:
            True if execution succeeded, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(
            f"Executing escalation protocol: {protocol.level.name} "
            f"for worker {protocol.worker_id}"
        )

        try:
            # Record the protocol
            self.record(protocol)

            # Execute based on level
            if protocol.level == EscalationLevel.MONITOR:
                logger.info(f"Monitoring enabled for worker {protocol.worker_id}")
                return True

            elif protocol.level == EscalationLevel.INTERVENE:
                logger.warning(
                    f"Intervention required for worker {protocol.worker_id}: "
                    f"{protocol.reason}"
                )
                # Director should pause worker and analyze state
                return True

            elif protocol.level == EscalationLevel.REASSIGN:
                logger.warning(
                    f"Task reassignment required for worker {protocol.worker_id}"
                )
                # Director should move task to different worker
                return True

            elif protocol.level == EscalationLevel.ESCALATE:
                logger.error(
                    f"Escalating to higher authority: worker {protocol.worker_id} "
                    f"cannot complete task {protocol.task_id}"
                )
                # Director should escalate to orchestrator
                return True

            elif protocol.level == EscalationLevel.ABORT:
                logger.error(
                    f"Aborting task {protocol.task_id} for worker {protocol.worker_id}"
                )
                # Director should abort task and create failure record
                return True

            else:
                logger.debug("No escalation action needed")
                return True

        except Exception as e:
            logger.error(f"Failed to execute escalation protocol: {e}")
            return False

    def record(self, protocol: EscalationProtocol) -> None:
        """
        Record an escalation protocol in history.

        Args:
            protocol: The protocol to record
        """
        self._escalation_history.append(protocol)

    def get_worker_strikes(self, worker_id: str) -> int:
        """Get strike count for a worker."""
        return self._worker_strikes.get(worker_id, 0)

    def reset_worker_strikes(self, worker_id: str) -> None:
        """Reset strike count for a worker (e.g., after successful completion)."""
        if worker_id in self._worker_strikes:
            self._worker_strikes[worker_id] = 0
        if worker_id in self._worker_confusion_history:
            self._worker_confusion_history[worker_id].clear()

    def get_escalation_history(self) -> List[EscalationProtocol]:
        """Get full escalation history."""
        return self._escalation_history.copy()

    def get_worker_confusion_history(self, worker_id: str) -> List[ConfusionRecord]:
        """Get confusion history for a specific worker."""
        return self._worker_confusion_history.get(worker_id, []).copy()
