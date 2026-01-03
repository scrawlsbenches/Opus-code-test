"""
Unit tests for escalation protocol.

Tests for EscalationLevel, EscalationProtocol, and EscalationManager.
"""

import pytest
from datetime import datetime

from llm_orchestration.escalation import (
    EscalationLevel,
    EscalationProtocol,
    EscalationManager,
    ConfusionRecord,
)
from llm_orchestration.recovery import ConfusionSignal


# =============================================================================
# ESCALATION LEVEL TESTS
# =============================================================================


class TestEscalationLevel:
    """Test EscalationLevel enum."""

    def test_escalation_levels_exist(self):
        """Test all escalation levels are defined."""
        assert EscalationLevel.NONE
        assert EscalationLevel.MONITOR
        assert EscalationLevel.INTERVENE
        assert EscalationLevel.REASSIGN
        assert EscalationLevel.ESCALATE
        assert EscalationLevel.ABORT

    def test_escalation_level_ordering(self):
        """Test escalation levels have correct severity ordering."""
        assert EscalationLevel.NONE.value < EscalationLevel.MONITOR.value
        assert EscalationLevel.MONITOR.value < EscalationLevel.INTERVENE.value
        assert EscalationLevel.INTERVENE.value < EscalationLevel.REASSIGN.value
        assert EscalationLevel.REASSIGN.value < EscalationLevel.ESCALATE.value
        assert EscalationLevel.ESCALATE.value < EscalationLevel.ABORT.value


# =============================================================================
# ESCALATION PROTOCOL TESTS
# =============================================================================


class TestEscalationProtocol:
    """Test EscalationProtocol dataclass."""

    def test_protocol_creation(self):
        """Test creating an escalation protocol."""
        protocol = EscalationProtocol(
            level=EscalationLevel.MONITOR,
            reason="Worker showing confusion",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[],
            recommended_action="Increase monitoring"
        )

        assert protocol.level == EscalationLevel.MONITOR
        assert protocol.reason == "Worker showing confusion"
        assert protocol.worker_id == "worker-1"
        assert protocol.task_id == "task-1"
        assert len(protocol.confusion_history) == 0

    def test_protocol_to_dict(self):
        """Test converting protocol to dictionary."""
        confusion_record = ConfusionRecord(
            signal_type="test",
            severity="LOW",
            recovery_action="monitor",
            recovered=True
        )

        protocol = EscalationProtocol(
            level=EscalationLevel.INTERVENE,
            reason="Test reason",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[confusion_record],
            recommended_action="Test action"
        )

        result = protocol.to_dict()

        assert result["level"] == "INTERVENE"
        assert result["reason"] == "Test reason"
        assert result["worker_id"] == "worker-1"
        assert result["task_id"] == "task-1"
        assert result["confusion_count"] == 1
        assert result["recommended_action"] == "Test action"
        assert "timestamp" in result


# =============================================================================
# ESCALATION MANAGER TESTS
# =============================================================================


class TestEscalationManager:
    """Test EscalationManager functionality."""

    def test_manager_initialization(self):
        """Test creating an escalation manager."""
        manager = EscalationManager()

        assert len(manager._escalation_history) == 0
        assert len(manager._worker_strikes) == 0
        assert len(manager._worker_confusion_history) == 0

    def test_evaluate_first_low_severity(self):
        """Test evaluation of first low-severity confusion."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="repetition",
            description="Worker repeating same action",
            evidence=["action1", "action1"],
            confidence=0.3,  # Low confidence -> LOW severity
            source="test"
        )

        protocol = manager.evaluate(
            worker_id="worker-1",
            confusion=signal,
            task_id="task-1"
        )

        assert protocol.level == EscalationLevel.MONITOR
        assert protocol.worker_id == "worker-1"
        assert protocol.task_id == "task-1"
        assert len(protocol.confusion_history) == 1

    def test_evaluate_first_high_severity(self):
        """Test evaluation of first high-severity confusion."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="context_loss",
            description="Worker lost context",
            evidence=["missing_context"],
            confidence=0.85,  # High confidence -> HIGH severity
            source="test"
        )

        protocol = manager.evaluate(
            worker_id="worker-1",
            confusion=signal,
            task_id="task-1"
        )

        assert protocol.level == EscalationLevel.INTERVENE
        assert "pause" in protocol.recommended_action.lower() or "intervene" in protocol.recommended_action.lower()

    def test_evaluate_second_medium_severity(self):
        """Test evaluation of second medium-severity confusion."""
        manager = EscalationManager()

        # First confusion
        signal1 = ConfusionSignal(
            signal_type="test1",
            description="First confusion",
            evidence=["e1"],
            confidence=0.6,  # MEDIUM
            source="test"
        )
        manager.evaluate("worker-1", signal1, "task-1")

        # Second confusion
        signal2 = ConfusionSignal(
            signal_type="test2",
            description="Second confusion",
            evidence=["e2"],
            confidence=0.6,  # MEDIUM
            source="test"
        )
        protocol = manager.evaluate("worker-1", signal2, "task-1")

        assert protocol.level == EscalationLevel.REASSIGN
        assert len(protocol.confusion_history) == 2

    def test_evaluate_third_confusion_always_aborts(self):
        """Test that three confusions always leads to ABORT."""
        manager = EscalationManager()

        # Add three confusions
        for i in range(3):
            signal = ConfusionSignal(
                signal_type=f"test{i}",
                description=f"Confusion {i}",
                evidence=[f"e{i}"],
                confidence=0.3,  # Even LOW severity
                source="test"
            )
            protocol = manager.evaluate("worker-1", signal, "task-1")

        assert protocol.level == EscalationLevel.ABORT
        assert manager.get_worker_strikes("worker-1") == 3

    def test_worker_strikes_tracking(self):
        """Test worker strike counting."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="test",
            description="Test",
            evidence=["e1"],
            confidence=0.5,
            source="test"
        )

        # Add strikes
        manager.evaluate("worker-1", signal, "task-1")
        assert manager.get_worker_strikes("worker-1") == 1

        manager.evaluate("worker-1", signal, "task-1")
        assert manager.get_worker_strikes("worker-1") == 2

    def test_reset_worker_strikes(self):
        """Test resetting worker strikes."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="test",
            description="Test",
            evidence=["e1"],
            confidence=0.5,
            source="test"
        )

        manager.evaluate("worker-1", signal, "task-1")
        assert manager.get_worker_strikes("worker-1") == 1

        manager.reset_worker_strikes("worker-1")
        assert manager.get_worker_strikes("worker-1") == 0

    def test_execute_protocol_monitor(self):
        """Test executing MONITOR escalation."""
        manager = EscalationManager()

        protocol = EscalationProtocol(
            level=EscalationLevel.MONITOR,
            reason="Test",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[],
            recommended_action="Monitor"
        )

        success = manager.execute(protocol)

        assert success is True
        assert len(manager.get_escalation_history()) == 1

    def test_execute_protocol_intervene(self):
        """Test executing INTERVENE escalation."""
        manager = EscalationManager()

        protocol = EscalationProtocol(
            level=EscalationLevel.INTERVENE,
            reason="Test intervention",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[],
            recommended_action="Intervene"
        )

        success = manager.execute(protocol)

        assert success is True

    def test_execute_protocol_escalate(self):
        """Test executing ESCALATE level."""
        manager = EscalationManager()

        protocol = EscalationProtocol(
            level=EscalationLevel.ESCALATE,
            reason="Critical confusion",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[],
            recommended_action="Escalate to higher authority"
        )

        success = manager.execute(protocol)

        assert success is True

    def test_execute_protocol_abort(self):
        """Test executing ABORT level."""
        manager = EscalationManager()

        protocol = EscalationProtocol(
            level=EscalationLevel.ABORT,
            reason="Too many failures",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[],
            recommended_action="Abort task"
        )

        success = manager.execute(protocol)

        assert success is True

    def test_escalation_history_tracking(self):
        """Test that escalation history is tracked."""
        manager = EscalationManager()

        protocol1 = EscalationProtocol(
            level=EscalationLevel.MONITOR,
            reason="Test 1",
            worker_id="worker-1",
            task_id="task-1",
            confusion_history=[],
            recommended_action="Action 1"
        )

        protocol2 = EscalationProtocol(
            level=EscalationLevel.INTERVENE,
            reason="Test 2",
            worker_id="worker-2",
            task_id="task-2",
            confusion_history=[],
            recommended_action="Action 2"
        )

        manager.execute(protocol1)
        manager.execute(protocol2)

        history = manager.get_escalation_history()

        assert len(history) == 2
        assert history[0].level == EscalationLevel.MONITOR
        assert history[1].level == EscalationLevel.INTERVENE

    def test_worker_confusion_history_retrieval(self):
        """Test retrieving confusion history for a worker."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="test",
            description="Test confusion",
            evidence=["e1"],
            confidence=0.5,
            source="test"
        )

        manager.evaluate("worker-1", signal, "task-1")
        manager.evaluate("worker-1", signal, "task-1")

        history = manager.get_worker_confusion_history("worker-1")

        assert len(history) == 2
        assert all(isinstance(record, ConfusionRecord) for record in history)

    def test_severity_inference_from_confidence(self):
        """Test that severity is correctly inferred from confidence."""
        manager = EscalationManager()

        # Test critical (>= 0.9)
        assert manager._infer_severity_from_confidence(0.95) == "CRITICAL"

        # Test high (0.7-0.9)
        assert manager._infer_severity_from_confidence(0.8) == "HIGH"

        # Test medium (0.5-0.7)
        assert manager._infer_severity_from_confidence(0.6) == "MEDIUM"

        # Test low (< 0.5)
        assert manager._infer_severity_from_confidence(0.3) == "LOW"

    def test_different_workers_tracked_separately(self):
        """Test that different workers are tracked independently."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="test",
            description="Test",
            evidence=["e1"],
            confidence=0.5,
            source="test"
        )

        manager.evaluate("worker-1", signal, "task-1")
        manager.evaluate("worker-2", signal, "task-2")

        assert manager.get_worker_strikes("worker-1") == 1
        assert manager.get_worker_strikes("worker-2") == 1
        assert len(manager.get_worker_confusion_history("worker-1")) == 1
        assert len(manager.get_worker_confusion_history("worker-2")) == 1

    def test_generate_action_descriptions(self):
        """Test that action descriptions are generated for each level."""
        manager = EscalationManager()

        signal = ConfusionSignal(
            signal_type="test",
            description="Test confusion",
            evidence=["e1"],
            confidence=0.5,
            source="test"
        )

        # Test each level has a specific action
        for level in [EscalationLevel.NONE, EscalationLevel.MONITOR,
                      EscalationLevel.INTERVENE, EscalationLevel.REASSIGN,
                      EscalationLevel.ESCALATE, EscalationLevel.ABORT]:
            action = manager._generate_action(level, "worker-1", signal)
            assert len(action) > 0
            assert isinstance(action, str)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEscalationIntegration:
    """Test integration scenarios."""

    def test_escalation_progression(self):
        """Test normal escalation progression from MONITOR to ABORT."""
        manager = EscalationManager()

        # First: MONITOR
        signal1 = ConfusionSignal(
            signal_type="test",
            description="Minor confusion",
            evidence=["e1"],
            confidence=0.4,  # LOW
            source="test"
        )
        protocol1 = manager.evaluate("worker-1", signal1, "task-1")
        assert protocol1.level == EscalationLevel.MONITOR

        # Second: INTERVENE
        signal2 = ConfusionSignal(
            signal_type="test",
            description="Persistent confusion",
            evidence=["e2"],
            confidence=0.4,  # Still LOW
            source="test"
        )
        protocol2 = manager.evaluate("worker-1", signal2, "task-1")
        assert protocol2.level == EscalationLevel.INTERVENE

        # Third: ABORT
        signal3 = ConfusionSignal(
            signal_type="test",
            description="Continued confusion",
            evidence=["e3"],
            confidence=0.4,  # Still LOW, but 3rd strike
            source="test"
        )
        protocol3 = manager.evaluate("worker-1", signal3, "task-1")
        assert protocol3.level == EscalationLevel.ABORT

    def test_high_severity_fast_escalation(self):
        """Test that high severity leads to faster escalation."""
        manager = EscalationManager()

        # First high severity -> INTERVENE (skips MONITOR)
        signal1 = ConfusionSignal(
            signal_type="critical",
            description="Critical confusion",
            evidence=["e1"],
            confidence=0.85,  # HIGH
            source="test"
        )
        protocol1 = manager.evaluate("worker-1", signal1, "task-1")
        assert protocol1.level == EscalationLevel.INTERVENE

        # Second high severity -> ESCALATE
        signal2 = ConfusionSignal(
            signal_type="critical",
            description="Still critical",
            evidence=["e2"],
            confidence=0.85,  # HIGH
            source="test"
        )
        protocol2 = manager.evaluate("worker-1", signal2, "task-1")
        assert protocol2.level == EscalationLevel.ESCALATE
