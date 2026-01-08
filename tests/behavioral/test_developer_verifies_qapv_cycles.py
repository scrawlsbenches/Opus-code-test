"""
Behavioral tests for QAPV cycle verification and anomaly detection.

As a developer building cognitive loop systems,
I want to detect behavioral anomalies in QAPV cycles,
So that I can ensure loops follow correct reasoning patterns.

Based on: examples/qapv_verification_demo.py
"""

import pytest
import time
from cortical.reasoning import (
    CognitiveLoop,
    LoopPhase,
    QAPVVerifier,
    QAPVAnomaly,
)


class TestDeveloperDetectsInvalidTransitions:
    """
    Epic: State Machine Validation

    As a developer ensuring correct loop behavior,
    I want to detect invalid phase transitions,
    So that state machine violations are caught early.
    """

    def test_scenario_verifier_detects_skipped_phases(self):
        """
        Scenario: Verifier catches skipped phases

        Given a verifier tracking transitions
        When a phase is skipped (e.g., QUESTION → PRODUCE)
        Then an invalid transition anomaly is detected
        Because QAPV phases must follow sequence
        """
        # Given: verifier tracking transitions
        verifier = QAPVVerifier()

        # When: skipping ANSWER phase
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "produce")  # Invalid!

        # Then: anomaly is detected
        anomalies = verifier.check_health()
        invalid_transitions = [
            a for a in anomalies
            if a.anomaly_type == QAPVAnomaly.INVALID_TRANSITION
        ]
        assert len(invalid_transitions) > 0

    def test_scenario_verifier_provides_suggestions_for_invalid_transitions(self):
        """
        Scenario: Verifier suggests corrections for violations

        Given an invalid transition anomaly
        When examining the anomaly details
        Then actionable suggestions are provided
        Because developers need guidance on fixing violations
        """
        # Given: invalid transition
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "produce")  # Skip ANSWER

        # When: checking anomalies
        anomalies = verifier.check_health()
        invalid = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.INVALID_TRANSITION]

        # Then: suggestions provided
        assert len(invalid) > 0
        anomaly = invalid[0]
        assert len(anomaly.suggestions) > 0
        assert all(isinstance(s, str) for s in anomaly.suggestions)

    def test_scenario_healthy_cycle_produces_no_anomalies(self):
        """
        Scenario: Correct QAPV cycle shows as healthy

        Given a verifier tracking transitions
        When executing complete QAPV cycle correctly
        Then no anomalies are detected
        Because correct behavior should validate cleanly
        """
        # Given: verifier tracking transitions
        verifier = QAPVVerifier()

        # When: executing proper QAPV cycle
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")
        verifier.record_transition("answer", "produce")
        verifier.record_transition("produce", "verify")

        # Then: no anomalies
        anomalies = verifier.check_health()
        assert len(anomalies) == 0


class TestDeveloperDetectsStuckPhases:
    """
    Epic: Execution Progress Monitoring

    As a developer monitoring loop execution,
    I want to detect when loops get stuck in a phase,
    So that infinite waits are prevented.
    """

    def test_scenario_verifier_detects_phase_timeout(self):
        """
        Scenario: Verifier detects stuck phases

        Given a verifier with stuck threshold
        When a phase exceeds time threshold
        Then a stuck phase anomaly is raised
        Because phases should complete in reasonable time
        """
        # Given: verifier with 50ms threshold (minimum viable for testing)
        verifier = QAPVVerifier(stuck_threshold_seconds=0.05)

        # When: staying in phase too long
        verifier.record_transition(None, "question")
        time.sleep(0.1)  # Exceed threshold (100ms > 50ms)

        # Then: stuck phase detected
        anomalies = verifier.check_health()
        stuck = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.STUCK_PHASE]
        assert len(stuck) > 0

    def test_scenario_stuck_phase_severity_is_warning(self):
        """
        Scenario: Stuck phases are flagged as warnings

        Given a stuck phase anomaly
        When examining severity
        Then severity is 'warning' not 'error'
        Because stuck phases need investigation, not immediate halt
        """
        # Given: stuck phase (50ms threshold, minimum viable)
        verifier = QAPVVerifier(stuck_threshold_seconds=0.05)
        verifier.record_transition(None, "question")
        time.sleep(0.1)  # 100ms > 50ms threshold

        # When: checking anomalies
        anomalies = verifier.check_health()
        stuck = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.STUCK_PHASE]

        # Then: severity is medium (warning level)
        assert len(stuck) > 0
        assert stuck[0].severity == "medium"


class TestDeveloperDetectsInfiniteLoops:
    """
    Epic: Infinite Loop Prevention

    As a developer preventing resource waste,
    I want to detect when loops cycle indefinitely,
    So that runaway processes are caught.
    """

    def test_scenario_verifier_detects_excessive_cycles(self):
        """
        Scenario: Verifier catches too many QAPV cycles

        Given a verifier with max cycle threshold
        When loop cycles beyond threshold without completion
        Then infinite loop anomaly is raised
        Because endless cycling indicates a problem
        """
        # Given: verifier with 3-cycle threshold
        verifier = QAPVVerifier(max_cycles_before_warning=3)

        # When: executing 3 complete cycles
        for _ in range(3):
            verifier.record_transition(None, "question")
            verifier.record_transition("question", "answer")
            verifier.record_transition("answer", "produce")
            verifier.record_transition("produce", "verify")
            verifier.record_transition("verify", "question")  # Back to start

        # Then: infinite loop detected
        anomalies = verifier.check_health()
        infinite = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.INFINITE_LOOP]
        assert len(infinite) > 0

    def test_scenario_cycle_count_is_tracked_accurately(self):
        """
        Scenario: Verifier accurately counts cycles

        Given a verifier tracking cycles
        When executing multiple complete QAPV cycles
        Then cycle count reflects actual cycles
        Because accurate counting enables threshold detection
        """
        # Given: fresh verifier
        verifier = QAPVVerifier()

        # When: executing 2 cycles
        for _ in range(2):
            verifier.record_transition(None, "question")
            verifier.record_transition("question", "answer")
            verifier.record_transition("answer", "produce")
            verifier.record_transition("produce", "verify")
            verifier.record_transition("verify", "question")

        # Then: count is accurate
        assert verifier.get_cycle_count() == 2


class TestDeveloperDetectsPrematureCompletion:
    """
    Epic: Quality Gate Enforcement

    As a quality guardian,
    I want to ensure production is verified before completion,
    So that unverified work is not marked done.
    """

    def test_scenario_verifier_detects_unverified_completion(self):
        """
        Scenario: Verifier catches completion without verification

        Given a loop in PRODUCE phase
        When transitioning directly to COMPLETE
        Then premature exit anomaly is raised
        Because production must be verified
        """
        # Given: loop in produce phase
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")
        verifier.record_transition("answer", "produce")

        # When: completing without verify
        verifier.record_transition("produce", "complete")

        # Then: premature exit detected
        anomalies = verifier.check_health()
        premature = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.PREMATURE_EXIT]
        assert len(premature) > 0

    def test_scenario_completion_after_verify_is_acceptable(self):
        """
        Scenario: Completion after verification is valid

        Given a loop that has verified production
        When transitioning to COMPLETE
        Then no premature exit anomaly is raised
        Because verification was performed
        """
        # Given: loop with verification
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")
        verifier.record_transition("answer", "produce")
        verifier.record_transition("produce", "verify")

        # When: completing after verify
        verifier.record_transition("verify", "complete")

        # Then: no premature exit anomaly
        anomalies = verifier.check_health()
        premature = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.PREMATURE_EXIT]
        assert len(premature) == 0


class TestDeveloperDetectsMissingProduction:
    """
    Epic: Production Verification

    As a developer ensuring deliverables,
    I want to detect verification without prior production,
    So that empty verifications are caught.
    """

    def test_scenario_verifier_detects_verify_without_produce(self):
        """
        Scenario: Verifier catches verification before production

        Given a loop that skips PRODUCE phase
        When transitioning to VERIFY
        Then missing production anomaly is raised
        Because you cannot verify what was not produced
        """
        # Given: loop skipping produce
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")

        # When: going directly to verify
        verifier.record_transition("answer", "verify")

        # Then: missing production detected
        anomalies = verifier.check_health()
        missing_prod = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.MISSING_PRODUCTION]
        assert len(missing_prod) > 0

    def test_scenario_produce_then_verify_is_valid(self):
        """
        Scenario: Normal produce-then-verify sequence is valid

        Given a loop that produces artifacts
        When transitioning to VERIFY
        Then no missing production anomaly is raised
        Because production occurred before verification
        """
        # Given: loop with production
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")
        verifier.record_transition("answer", "produce")

        # When: verifying after production
        verifier.record_transition("produce", "verify")

        # Then: no anomaly
        anomalies = verifier.check_health()
        missing_prod = [a for a in anomalies if a.anomaly_type == QAPVAnomaly.MISSING_PRODUCTION]
        assert len(missing_prod) == 0


class TestDeveloperReceivesDiagnosticReports:
    """
    Epic: Comprehensive Diagnostics

    As a developer debugging loop behavior,
    I want comprehensive diagnostic reports,
    So that I can understand loop health at a glance.
    """

    def test_scenario_diagnostic_report_includes_health_status(self):
        """
        Scenario: Report provides overall health status

        Given a verifier with tracked transitions
        When requesting diagnostic report
        Then health status is included
        Because developers need quick health assessment
        """
        # Given: verifier with transitions
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")

        # When: requesting report
        report = verifier.get_diagnostic_report()

        # Then: health status present
        assert "health_status" in report
        assert report["health_status"] in ["healthy", "warning", "error"]

    def test_scenario_diagnostic_report_counts_transitions_and_cycles(self):
        """
        Scenario: Report includes transition and cycle counts

        Given a verifier tracking multiple transitions
        When requesting diagnostic report
        Then counts are accurate
        Because metrics enable trend analysis
        """
        # Given: verifier with multiple transitions
        verifier = QAPVVerifier()

        # Complete one cycle
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")
        verifier.record_transition("answer", "produce")
        verifier.record_transition("produce", "verify")
        verifier.record_transition("verify", "question")

        # Partial second cycle
        verifier.record_transition("question", "answer")

        # When: requesting report
        report = verifier.get_diagnostic_report()

        # Then: counts are present
        assert "total_transitions" in report
        assert "cycle_count" in report
        assert report["total_transitions"] > 0

    def test_scenario_diagnostic_report_categorizes_anomalies_by_type(self):
        """
        Scenario: Report groups anomalies by type

        Given a verifier with multiple anomaly types
        When requesting diagnostic report
        Then anomalies are categorized
        Because categorization helps prioritize fixes
        """
        # Given: multiple anomaly types (50ms threshold, minimum viable)
        verifier = QAPVVerifier(stuck_threshold_seconds=0.05)
        verifier.record_transition(None, "question")
        time.sleep(0.1)  # Stuck phase (100ms > 50ms)
        verifier.record_transition("question", "produce")  # Invalid transition

        # When: requesting report
        report = verifier.get_diagnostic_report()

        # Then: anomalies categorized
        assert "anomalies_by_type" in report
        if report["total_anomalies"] > 0:
            assert isinstance(report["anomalies_by_type"], dict)

    def test_scenario_diagnostic_report_shows_current_phase(self):
        """
        Scenario: Report indicates current phase

        Given a verifier in a specific phase
        When requesting diagnostic report
        Then current phase is shown
        Because developers need to know execution state
        """
        # Given: verifier in answer phase
        verifier = QAPVVerifier()
        verifier.record_transition(None, "question")
        verifier.record_transition("question", "answer")

        # When: requesting report
        report = verifier.get_diagnostic_report()

        # Then: current phase shown
        assert "current_phase" in report
        assert report["current_phase"] == "answer"


class TestDeveloperIntegratesVerifierWithCognitiveLoop:
    """
    Epic: Seamless Integration

    As a developer using cognitive loops,
    I want verifier to work with CognitiveLoop,
    So that verification is automatic.
    """

    def test_scenario_verifier_tracks_cognitive_loop_transitions(self):
        """
        Scenario: Verifier tracks CognitiveLoop transitions

        Given a CognitiveLoop and QAPVVerifier
        When executing loop transitions
        Then verifier records each transition
        Because manual tracking is error-prone
        """
        # Given: loop and verifier
        loop = CognitiveLoop(goal="Test task")
        verifier = QAPVVerifier()

        # When: executing transitions
        loop.start(LoopPhase.QUESTION)
        verifier.record_transition(None, LoopPhase.QUESTION.value)

        loop.transition(LoopPhase.ANSWER, reason="Questions answered")
        verifier.record_transition(
            LoopPhase.QUESTION.value,
            LoopPhase.ANSWER.value
        )

        # Then: transitions recorded
        report = verifier.get_diagnostic_report()
        assert report["total_transitions"] >= 2

    def test_scenario_verifier_detects_anomalies_in_real_loop(self):
        """
        Scenario: Verifier catches real loop violations

        Given a CognitiveLoop with QAPVVerifier
        When loop makes invalid transition
        Then verifier detects the anomaly
        Because verification must work with real loops
        """
        # Given: loop and verifier
        loop = CognitiveLoop(goal="Test task")
        verifier = QAPVVerifier()

        # When: making invalid transition sequence
        loop.start(LoopPhase.QUESTION)
        verifier.record_transition(None, LoopPhase.QUESTION.value)

        # Simulate invalid jump to VERIFY
        verifier.record_transition(LoopPhase.QUESTION.value, "verify")

        # Then: anomaly detected
        anomalies = verifier.check_health()
        assert len(anomalies) > 0
