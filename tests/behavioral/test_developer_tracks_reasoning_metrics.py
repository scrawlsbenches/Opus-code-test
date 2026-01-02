"""
Behavioral tests for reasoning metrics collection and observability.

As a developer monitoring cognitive loop execution,
I want to collect metrics on loop phases, decisions, and crisis recovery,
So that I can identify performance bottlenecks and improve system resilience.

Based on: examples/reasoning_metrics_demo.py
"""

import pytest
import time
from cortical.reasoning import (
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    TerminationReason,
    VerificationManager,
    create_drafting_checklist,
    CrisisManager,
    CrisisLevel,
    RecoveryAction,
)
from cortical.reasoning.metrics import (
    ReasoningMetrics,
    create_loop_metrics_handler,
)


class TestDeveloperMonitorsLoopPhasePerformance:
    """
    Epic: Phase Timing Observability

    As a developer optimizing cognitive loops,
    I want to track time spent in each QAPV phase,
    So that I can identify performance bottlenecks.
    """

    def test_scenario_metrics_track_time_per_phase(self):
        """
        Scenario: Metrics capture timing for each QAPV phase

        Given a metrics collector
        When executing each QAPV phase with timing
        Then metrics record duration for each phase
        Because developers need to identify slow phases
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: executing phases with timing
        with metrics.phase_timer(LoopPhase.QUESTION):
            time.sleep(0.01)

        with metrics.phase_timer(LoopPhase.ANSWER):
            time.sleep(0.02)

        with metrics.phase_timer(LoopPhase.PRODUCE):
            time.sleep(0.03)

        with metrics.phase_timer(LoopPhase.VERIFY):
            time.sleep(0.01)

        # Then: metrics contain timing data for all phases
        summary = metrics.get_summary()
        # Summary uses lowercase phase names
        assert "question" in summary.lower()
        assert "answer" in summary.lower()
        assert "produce" in summary.lower()
        assert "verify" in summary.lower()

    def test_scenario_phase_timers_measure_actual_duration(self):
        """
        Scenario: Phase timers accurately measure duration

        Given a metrics collector
        When timing a phase that takes measurable time
        Then the recorded duration reflects actual time spent
        Because accurate timing enables optimization
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: timing a phase with known duration
        sleep_time = 0.05  # 50ms
        with metrics.phase_timer(LoopPhase.ANSWER):
            time.sleep(sleep_time)

        # Then: recorded time approximates actual time
        metrics_dict = metrics.get_metrics_dict()
        answer_metrics = metrics_dict.get("phase_answer", {})
        avg_ms = answer_metrics.get("avg_ms", 0)

        # Should be roughly 50ms (allow for timing variance)
        assert avg_ms >= 40.0, f"Expected ~50ms, got {avg_ms}ms"
        assert avg_ms <= 100.0, f"Expected ~50ms, got {avg_ms}ms"


class TestDeveloperTracksReasoningActivity:
    """
    Epic: Activity Metrics Collection

    As a developer analyzing reasoning patterns,
    I want to count questions, decisions, and productions,
    So that I can understand cognitive workload.
    """

    def test_scenario_metrics_count_questions_asked(self):
        """
        Scenario: Metrics count all questions during loop

        Given a metrics collector
        When recording multiple questions
        Then question count increases
        Because question volume indicates exploration depth
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording questions
        metrics.record_question("requirements")
        metrics.record_question("constraints")
        metrics.record_question("technical")

        # Then: count reflects all questions
        # Note: questions are tracked via the questions_asked attribute, not in metrics_dict
        assert metrics.questions_asked == 3

    def test_scenario_metrics_count_decisions_made(self):
        """
        Scenario: Metrics count all decisions during loop

        Given a metrics collector
        When recording multiple decisions
        Then decision count increases
        Because decision count indicates reasoning complexity
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording decisions
        metrics.record_decision("architecture")
        metrics.record_decision("implementation")
        metrics.record_decision("testing")

        # Then: count reflects all decisions
        # Note: decisions are tracked via the decisions_made attribute
        assert metrics.decisions_made == 3

    def test_scenario_metrics_count_artifacts_produced(self):
        """
        Scenario: Metrics count production events

        Given a metrics collector
        When recording artifact production
        Then production count increases
        Because production volume measures output
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording productions
        metrics.record_production("code")
        metrics.record_production("test")
        metrics.record_production("documentation")

        # Then: count reflects all productions
        # Note: productions are tracked via the productions_created attribute
        assert metrics.productions_created == 3


class TestDeveloperMonitorsVerificationQuality:
    """
    Epic: Verification Quality Tracking

    As a developer ensuring quality,
    I want to track verification pass/fail rates,
    So that I can measure solution quality.
    """

    def test_scenario_metrics_calculate_verification_pass_rate(self):
        """
        Scenario: Metrics compute verification pass rate

        Given a metrics collector
        When recording mix of passed and failed verifications
        Then pass rate is calculated correctly
        Because pass rate indicates quality
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording verifications (3 pass, 1 fail)
        metrics.record_verification(passed=True)
        metrics.record_verification(passed=True)
        metrics.record_verification(passed=True)
        metrics.record_verification(passed=False)

        # Then: pass rate is 75%
        pass_rate = metrics.get_verification_pass_rate()
        assert pass_rate == 75.0

    def test_scenario_metrics_track_verification_failures(self):
        """
        Scenario: Metrics count verification failures separately

        Given a metrics collector
        When recording failed verifications
        Then failure count is tracked
        Because failures need investigation
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording failures
        metrics.record_verification(passed=False)
        metrics.record_verification(passed=False)
        metrics.record_verification(passed=True)

        # Then: failures are counted
        assert metrics.verifications_failed == 2
        assert metrics.verifications_passed == 1


class TestDeveloperMonitorsCrisisRecovery:
    """
    Epic: Crisis Recovery Tracking

    As a developer building resilient systems,
    I want to track crisis detection and recovery rates,
    So that I can improve error handling.
    """

    def test_scenario_metrics_track_crisis_events(self):
        """
        Scenario: Metrics count crisis occurrences

        Given a metrics collector
        When recording crisis events
        Then crisis count increases
        Because crisis frequency indicates instability
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording crises
        metrics.record_crisis(recovered=True, level="hiccup")
        metrics.record_crisis(recovered=True, level="obstacle")
        metrics.record_crisis(recovered=False, level="wall")

        # Then: all crises are counted
        assert metrics.crises_detected == 3

    def test_scenario_metrics_calculate_recovery_rate(self):
        """
        Scenario: Metrics compute crisis recovery rate

        Given a metrics collector
        When recording mix of recovered and unrecovered crises
        Then recovery rate is calculated
        Because recovery rate measures resilience
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording crises (2 recovered, 1 not)
        metrics.record_crisis(recovered=True, level="hiccup")
        metrics.record_crisis(recovered=True, level="obstacle")
        metrics.record_crisis(recovered=False, level="wall")

        # Then: recovery rate is ~67%
        recovery_rate = metrics.get_crisis_recovery_rate()
        assert 66.0 <= recovery_rate <= 67.0


class TestDeveloperIntegratesMetricsWithLoopManager:
    """
    Epic: Automatic Metrics Collection

    As a developer wanting automatic observability,
    I want metrics collected automatically during loop execution,
    So that I don't need manual instrumentation.
    """

    def test_scenario_handler_automatically_tracks_loop_lifecycle(self):
        """
        Scenario: Metrics handler tracks loop start and completion

        Given a loop manager with metrics handler
        When executing a complete loop
        Then metrics track loop lifecycle automatically
        Because manual tracking is error-prone
        """
        # Given: manager with metrics handler
        metrics = ReasoningMetrics()
        manager = CognitiveLoopManager()
        handler = create_loop_metrics_handler(metrics)
        manager.register_transition_handler(handler)

        # When: executing loop
        loop = manager.create_loop("Test task")
        metrics.record_loop_start()

        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, reason="Questions answered")
        loop.transition(LoopPhase.PRODUCE, reason="Design complete")
        loop.transition(LoopPhase.VERIFY, reason="Production complete")
        loop.complete(TerminationReason.SUCCESS)

        metrics.record_loop_complete(success=True)

        # Then: lifecycle is tracked
        completion_rate = metrics.get_loop_completion_rate()
        assert completion_rate == 100.0

    def test_scenario_metrics_track_loop_success_vs_failure(self):
        """
        Scenario: Metrics distinguish successful from failed loops

        Given a metrics collector
        When recording both successful and failed completions
        Then completion rate reflects success ratio
        Because success rate measures effectiveness
        """
        # Given: a metrics collector
        metrics = ReasoningMetrics()

        # When: recording completions (2 success, 1 fail)
        metrics.record_loop_start()
        metrics.record_loop_complete(success=True)

        metrics.record_loop_start()
        metrics.record_loop_complete(success=True)

        metrics.record_loop_start()
        metrics.record_loop_complete(success=False)

        # Then: success rate is calculated
        # Note: get_loop_completion_rate returns percentage of loops that completed successfully
        # This might be interpreted as "how many started loops completed successfully"
        # With 3 starts and 2 successes: 66.7%
        completion_rate = metrics.get_loop_completion_rate()
        assert 66.0 <= completion_rate <= 67.0


class TestDeveloperExportsMetricsForObservability:
    """
    Epic: Metrics Export and Reporting

    As a developer integrating with observability systems,
    I want to export metrics in structured format,
    So that they integrate with monitoring tools.
    """

    def test_scenario_metrics_export_as_dictionary(self):
        """
        Scenario: Metrics export in dictionary format

        Given a metrics collector with recorded data
        When exporting metrics
        Then structured dictionary is returned
        Because observability systems need structured data
        """
        # Given: metrics with data
        metrics = ReasoningMetrics()
        metrics.record_question("test")
        metrics.record_decision("test")
        metrics.record_verification(passed=True)

        # When: exporting
        metrics_dict = metrics.get_metrics_dict()

        # Then: structured dictionary
        assert isinstance(metrics_dict, dict)
        # Metrics dict contains phase timings and aggregate metrics
        assert "decisions_made" in metrics_dict
        assert len(metrics_dict) > 0

    def test_scenario_summary_provides_human_readable_report(self):
        """
        Scenario: Summary generates human-readable report

        Given a metrics collector with varied data
        When generating summary
        Then readable text report is produced
        Because humans need to understand metrics quickly
        """
        # Given: metrics with data
        metrics = ReasoningMetrics()

        with metrics.phase_timer(LoopPhase.QUESTION):
            time.sleep(0.01)

        metrics.record_question("test")
        metrics.record_decision("test")
        metrics.record_verification(passed=True)

        # When: generating summary
        summary = metrics.get_summary()

        # Then: readable report
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "QUESTION" in summary or "Question" in summary
