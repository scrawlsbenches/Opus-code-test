"""
Unit tests for reasoning metrics module.

Tests for cortical.reasoning.metrics, including:
- PhaseMetrics tracking
- ReasoningMetrics collection
- MetricsContextManager
- Integration with CognitiveLoop
"""

import unittest
import time
from datetime import datetime, timedelta

from cortical.reasoning import (
    LoopPhase,
    LoopStatus,
    TerminationReason,
    CognitiveLoop,
    CognitiveLoopManager,
)
from cortical.reasoning.metrics import (
    PhaseMetrics,
    ReasoningMetrics,
    MetricsContextManager,
    create_loop_metrics_handler,
)


class TestPhaseMetrics(unittest.TestCase):
    """Test PhaseMetrics class."""

    def test_initialization(self):
        """Test PhaseMetrics initialization."""
        metrics = PhaseMetrics(phase_name="question")

        self.assertEqual(metrics.phase_name, "question")
        self.assertEqual(metrics.entry_count, 0)
        self.assertEqual(metrics.total_duration_ms, 0.0)
        self.assertEqual(metrics.min_duration_ms, float('inf'))
        self.assertEqual(metrics.max_duration_ms, 0.0)

    def test_record_entry(self):
        """Test recording phase entries."""
        metrics = PhaseMetrics(phase_name="answer")

        metrics.record_entry(100.0)
        self.assertEqual(metrics.entry_count, 1)
        self.assertEqual(metrics.total_duration_ms, 100.0)
        self.assertEqual(metrics.min_duration_ms, 100.0)
        self.assertEqual(metrics.max_duration_ms, 100.0)

        metrics.record_entry(200.0)
        self.assertEqual(metrics.entry_count, 2)
        self.assertEqual(metrics.total_duration_ms, 300.0)
        self.assertEqual(metrics.min_duration_ms, 100.0)
        self.assertEqual(metrics.max_duration_ms, 200.0)

    def test_get_average_ms(self):
        """Test average calculation."""
        metrics = PhaseMetrics(phase_name="produce")

        # No entries
        self.assertEqual(metrics.get_average_ms(), 0.0)

        # With entries
        metrics.record_entry(100.0)
        metrics.record_entry(200.0)
        metrics.record_entry(300.0)
        self.assertAlmostEqual(metrics.get_average_ms(), 200.0, places=2)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PhaseMetrics(phase_name="verify")
        metrics.record_entry(150.0)
        metrics.record_entry(250.0)

        result = metrics.to_dict()

        self.assertEqual(result['count'], 2)
        self.assertEqual(result['total_ms'], 400.0)
        self.assertAlmostEqual(result['avg_ms'], 200.0, places=2)
        self.assertEqual(result['min_ms'], 150.0)
        self.assertEqual(result['max_ms'], 250.0)


class TestReasoningMetrics(unittest.TestCase):
    """Test ReasoningMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ReasoningMetrics()

    def test_initialization(self):
        """Test ReasoningMetrics initialization."""
        self.assertTrue(self.metrics.enabled)
        self.assertEqual(len(self.metrics.phases), 0)
        self.assertEqual(self.metrics.decisions_made, 0)
        self.assertEqual(self.metrics.questions_asked, 0)
        self.assertEqual(self.metrics.productions_created, 0)
        self.assertEqual(self.metrics.verifications_passed, 0)
        self.assertEqual(self.metrics.verifications_failed, 0)
        self.assertEqual(self.metrics.crises_detected, 0)
        self.assertEqual(self.metrics.crises_recovered, 0)
        self.assertEqual(self.metrics.loops_started, 0)
        self.assertEqual(self.metrics.loops_completed, 0)
        self.assertEqual(self.metrics.loops_aborted, 0)

    def test_record_phase_transition(self):
        """Test recording phase transitions."""
        # First transition (starting)
        self.metrics.record_phase_transition(
            from_phase=None,
            to_phase=LoopPhase.QUESTION,
            duration_ms=None
        )

        self.assertIn("question", self.metrics.phases)

        # Transition with duration
        self.metrics.record_phase_transition(
            from_phase=LoopPhase.QUESTION,
            to_phase=LoopPhase.ANSWER,
            duration_ms=150.0
        )

        self.assertEqual(self.metrics.phases["question"].entry_count, 1)
        self.assertEqual(self.metrics.phases["question"].total_duration_ms, 150.0)

    def test_record_decision(self):
        """Test recording decisions."""
        self.metrics.record_decision("architecture")
        self.assertEqual(self.metrics.decisions_made, 1)
        self.assertEqual(self.metrics.decisions_by_type["architecture"], 1)

        self.metrics.record_decision("architecture")
        self.metrics.record_decision("implementation")
        self.assertEqual(self.metrics.decisions_made, 3)
        self.assertEqual(self.metrics.decisions_by_type["architecture"], 2)
        self.assertEqual(self.metrics.decisions_by_type["implementation"], 1)

    def test_record_question(self):
        """Test recording questions."""
        self.metrics.record_question("clarification")
        self.assertEqual(self.metrics.questions_asked, 1)

        self.metrics.record_question("technical")
        self.assertEqual(self.metrics.questions_asked, 2)

    def test_record_production(self):
        """Test recording productions."""
        self.metrics.record_production("code")
        self.assertEqual(self.metrics.productions_created, 1)

        self.metrics.record_production("test")
        self.assertEqual(self.metrics.productions_created, 2)

    def test_record_verification(self):
        """Test recording verifications."""
        self.metrics.record_verification(passed=True)
        self.assertEqual(self.metrics.verifications_passed, 1)
        self.assertEqual(self.metrics.verifications_failed, 0)

        self.metrics.record_verification(passed=False)
        self.assertEqual(self.metrics.verifications_passed, 1)
        self.assertEqual(self.metrics.verifications_failed, 1)

    def test_record_crisis(self):
        """Test recording crises."""
        self.metrics.record_crisis(recovered=True)
        self.assertEqual(self.metrics.crises_detected, 1)
        self.assertEqual(self.metrics.crises_recovered, 1)

        self.metrics.record_crisis(recovered=False)
        self.assertEqual(self.metrics.crises_detected, 2)
        self.assertEqual(self.metrics.crises_recovered, 1)

    def test_record_loop_lifecycle(self):
        """Test recording loop lifecycle."""
        self.metrics.record_loop_start()
        self.assertEqual(self.metrics.loops_started, 1)

        self.metrics.record_loop_complete(success=True)
        self.assertEqual(self.metrics.loops_completed, 1)
        self.assertEqual(self.metrics.loops_aborted, 0)

        self.metrics.record_loop_start()
        self.metrics.record_loop_complete(success=False)
        self.assertEqual(self.metrics.loops_started, 2)
        self.assertEqual(self.metrics.loops_completed, 1)
        self.assertEqual(self.metrics.loops_aborted, 1)

    def test_get_verification_pass_rate(self):
        """Test verification pass rate calculation."""
        # No verifications
        self.assertEqual(self.metrics.get_verification_pass_rate(), 0.0)

        # Some verifications
        self.metrics.record_verification(passed=True)
        self.metrics.record_verification(passed=True)
        self.metrics.record_verification(passed=False)

        pass_rate = self.metrics.get_verification_pass_rate()
        self.assertAlmostEqual(pass_rate, 66.67, places=1)

    def test_get_crisis_recovery_rate(self):
        """Test crisis recovery rate calculation."""
        # No crises
        self.assertEqual(self.metrics.get_crisis_recovery_rate(), 0.0)

        # Some crises
        self.metrics.record_crisis(recovered=True)
        self.metrics.record_crisis(recovered=True)
        self.metrics.record_crisis(recovered=False)

        recovery_rate = self.metrics.get_crisis_recovery_rate()
        self.assertAlmostEqual(recovery_rate, 66.67, places=1)

    def test_get_loop_completion_rate(self):
        """Test loop completion rate calculation."""
        # No loops
        self.assertEqual(self.metrics.get_loop_completion_rate(), 0.0)

        # Some loops
        self.metrics.record_loop_start()
        self.metrics.record_loop_complete(success=True)
        self.metrics.record_loop_start()
        self.metrics.record_loop_complete(success=True)
        self.metrics.record_loop_start()
        self.metrics.record_loop_complete(success=False)

        completion_rate = self.metrics.get_loop_completion_rate()
        self.assertAlmostEqual(completion_rate, 66.67, places=1)

    def test_get_summary(self):
        """Test summary generation."""
        # Empty metrics
        summary = self.metrics.get_summary()
        self.assertIn("No metrics collected", summary)

        # With metrics
        self.metrics.record_phase_transition(None, LoopPhase.QUESTION, None)
        self.metrics.record_phase_transition(LoopPhase.QUESTION, LoopPhase.ANSWER, 100.0)
        self.metrics.record_decision("architecture")
        self.metrics.record_question("clarification")
        self.metrics.record_verification(passed=True)
        self.metrics.record_crisis(recovered=True)
        self.metrics.record_loop_start()

        summary = self.metrics.get_summary()
        self.assertIn("Reasoning Metrics Summary", summary)
        self.assertIn("Phase Transitions", summary)
        self.assertIn("Production Metrics", summary)
        self.assertIn("Verification Metrics", summary)
        self.assertIn("Crisis Metrics", summary)
        self.assertIn("Loop Lifecycle", summary)

    def test_get_metrics_dict(self):
        """Test dictionary export."""
        self.metrics.record_phase_transition(None, LoopPhase.QUESTION, None)
        self.metrics.record_phase_transition(LoopPhase.QUESTION, LoopPhase.ANSWER, 100.0)
        self.metrics.record_decision("architecture")
        self.metrics.record_verification(passed=True)

        result = self.metrics.get_metrics_dict()

        self.assertIn("phase_question", result)
        self.assertIn("decisions_made", result)
        self.assertIn("verifications_passed", result)
        self.assertIn("verification_pass_rate", result)

        # Check format compatibility with observability.py
        self.assertIn("count", result["decisions_made"])
        self.assertIn("count", result["verifications_passed"])

    def test_reset(self):
        """Test metrics reset."""
        # Add some metrics
        self.metrics.record_decision("test")
        self.metrics.record_question("test")
        self.metrics.record_verification(passed=True)

        # Reset
        self.metrics.reset()

        # Verify all cleared
        self.assertEqual(len(self.metrics.phases), 0)
        self.assertEqual(self.metrics.decisions_made, 0)
        self.assertEqual(self.metrics.questions_asked, 0)
        self.assertEqual(self.metrics.verifications_passed, 0)

    def test_enable_disable(self):
        """Test enabling/disabling metrics."""
        self.assertTrue(self.metrics.enabled)

        self.metrics.disable()
        self.assertFalse(self.metrics.enabled)

        # Recording should be no-op when disabled
        self.metrics.record_decision("test")
        self.assertEqual(self.metrics.decisions_made, 0)

        self.metrics.enable()
        self.assertTrue(self.metrics.enabled)

        # Recording should work when enabled
        self.metrics.record_decision("test")
        self.assertEqual(self.metrics.decisions_made, 1)

    def test_phase_timer_context_manager(self):
        """Test phase_timer context manager."""
        with self.metrics.phase_timer(LoopPhase.PRODUCE):
            time.sleep(0.01)  # Sleep for 10ms

        self.assertIn("produce", self.metrics.phases)
        self.assertEqual(self.metrics.phases["produce"].entry_count, 1)
        self.assertGreater(self.metrics.phases["produce"].total_duration_ms, 5.0)

    def test_phase_timer_when_disabled(self):
        """Test phase_timer when metrics are disabled."""
        self.metrics.disable()

        with self.metrics.phase_timer(LoopPhase.QUESTION):
            time.sleep(0.01)

        # Should not record anything
        self.assertEqual(len(self.metrics.phases), 0)


class TestMetricsContextManager(unittest.TestCase):
    """Test MetricsContextManager class."""

    def test_context_manager(self):
        """Test basic context manager usage."""
        metrics = ReasoningMetrics()
        ctx = MetricsContextManager(metrics, LoopPhase.ANSWER)

        with ctx:
            time.sleep(0.01)  # Sleep for 10ms

        self.assertIn("answer", metrics.phases)
        self.assertEqual(metrics.phases["answer"].entry_count, 1)
        self.assertGreater(metrics.phases["answer"].total_duration_ms, 5.0)


class TestLoopMetricsIntegration(unittest.TestCase):
    """Test integration with CognitiveLoop."""

    def test_create_loop_metrics_handler(self):
        """Test handler creation and registration."""
        metrics = ReasoningMetrics()
        manager = CognitiveLoopManager()

        # Register handler
        handler = create_loop_metrics_handler(metrics)
        manager.register_transition_handler(handler)

        # Create and run loop
        loop = manager.create_loop("Test goal")
        loop.start(LoopPhase.QUESTION)

        # Transition should be recorded
        self.assertIn("question", metrics.phases)

        # Transition with duration
        time.sleep(0.01)
        loop.transition(LoopPhase.ANSWER, reason="Test transition")

        # Should record question phase duration
        self.assertEqual(metrics.phases["question"].entry_count, 1)
        self.assertGreater(metrics.phases["question"].total_duration_ms, 5.0)

    def test_full_loop_metrics_tracking(self):
        """Test full loop lifecycle with metrics."""
        metrics = ReasoningMetrics()
        manager = CognitiveLoopManager()

        # Register handler
        handler = create_loop_metrics_handler(metrics)
        manager.register_transition_handler(handler)

        # Create loop
        loop = manager.create_loop("Full test")
        metrics.record_loop_start()

        # Go through phases
        loop.start(LoopPhase.QUESTION)
        time.sleep(0.01)

        loop.transition(LoopPhase.ANSWER, reason="Questions answered")
        time.sleep(0.01)

        loop.transition(LoopPhase.PRODUCE, reason="Solution designed")
        time.sleep(0.01)

        loop.transition(LoopPhase.VERIFY, reason="Artifacts produced")
        time.sleep(0.01)

        loop.complete(TerminationReason.SUCCESS)
        metrics.record_loop_complete(success=True)

        # Verify all phases were tracked
        self.assertIn("question", metrics.phases)
        self.assertIn("answer", metrics.phases)
        self.assertIn("produce", metrics.phases)
        self.assertIn("verify", metrics.phases)

        # Verify all have entries
        for phase_name in ["question", "answer", "produce"]:
            self.assertEqual(metrics.phases[phase_name].entry_count, 1)
            self.assertGreater(metrics.phases[phase_name].total_duration_ms, 5.0)

        # Verify loop lifecycle
        self.assertEqual(metrics.loops_started, 1)
        self.assertEqual(metrics.loops_completed, 1)
        self.assertEqual(metrics.loops_aborted, 0)

    def test_multiple_loops_tracking(self):
        """Test tracking multiple concurrent loops."""
        metrics = ReasoningMetrics()
        manager = CognitiveLoopManager()

        handler = create_loop_metrics_handler(metrics)
        manager.register_transition_handler(handler)

        # Create multiple loops
        loop1 = manager.create_loop("Loop 1")
        loop2 = manager.create_loop("Loop 2")

        metrics.record_loop_start()
        loop1.start(LoopPhase.QUESTION)
        time.sleep(0.01)
        loop1.transition(LoopPhase.ANSWER, reason="Test")

        metrics.record_loop_start()
        loop2.start(LoopPhase.PRODUCE)
        time.sleep(0.01)
        loop2.transition(LoopPhase.VERIFY, reason="Test")

        # Both loops should contribute to metrics
        self.assertEqual(metrics.loops_started, 2)

        # Phase entries should be accumulated
        self.assertEqual(metrics.phases["question"].entry_count, 1)
        self.assertEqual(metrics.phases["produce"].entry_count, 1)


if __name__ == '__main__':
    unittest.main()
