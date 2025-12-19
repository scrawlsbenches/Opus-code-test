"""
Extended unit tests for Verification module.
"""

import pytest
from cortical.reasoning import (
    VerificationLevel,
    VerificationPhase,
    VerificationStatus,
    VerificationCheck,
    VerificationFailure,
    VerificationSuite,
    VerificationManager,
    create_drafting_checklist,
    create_refining_checklist,
    create_finalizing_checklist,
)
from cortical.reasoning.verification import (
    VerificationRunner,
    FailureAnalyzer,
    RegressionDetector,
)


class TestVerificationLevel:
    """Tests for VerificationLevel enum."""

    def test_levels_ordered(self):
        """Test verification levels are properly ordered."""
        assert VerificationLevel.UNIT.value < VerificationLevel.INTEGRATION.value
        assert VerificationLevel.INTEGRATION.value < VerificationLevel.E2E.value
        assert VerificationLevel.E2E.value < VerificationLevel.ACCEPTANCE.value


class TestVerificationPhase:
    """Tests for VerificationPhase enum."""

    def test_all_phases_exist(self):
        """Test all verification phases exist."""
        assert VerificationPhase.DRAFTING is not None
        assert VerificationPhase.REFINING is not None
        assert VerificationPhase.FINALIZING is not None


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_all_statuses_exist(self):
        """Test all verification statuses exist."""
        assert VerificationStatus.PENDING is not None
        assert VerificationStatus.RUNNING is not None
        assert VerificationStatus.PASSED is not None
        assert VerificationStatus.FAILED is not None
        assert VerificationStatus.SKIPPED is not None


class TestVerificationCheck:
    """Tests for VerificationCheck dataclass."""

    def test_create_check(self):
        """Test creating a verification check."""
        check = VerificationCheck(
            name="Test passes",
            description="All unit tests should pass",
            level=VerificationLevel.UNIT,
        )

        assert check.name == "Test passes"
        assert check.level == VerificationLevel.UNIT
        assert check.status == VerificationStatus.PENDING

    def test_check_with_phase(self):
        """Test check with phase."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING
        )

        assert check.phase == VerificationPhase.DRAFTING

    def test_mark_passed(self):
        """Test marking a check as passed."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_passed("All tests passed", 100)

        assert check.status == VerificationStatus.PASSED
        assert check.result == "All tests passed"
        assert check.duration_ms == 100

    def test_mark_failed(self):
        """Test marking a check as failed."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_failed("Test assertion error", 50)

        assert check.status == VerificationStatus.FAILED
        assert check.result == "Test assertion error"


class TestVerificationFailure:
    """Tests for VerificationFailure dataclass."""

    def test_create_failure(self):
        """Test creating a verification failure."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Got 3 instead of 5",
            expected_vs_actual="Expected 5, got 3"
        )

        assert failure.check is check
        assert failure.observed == "Got 3 instead of 5"

    def test_add_hypothesis(self):
        """Test adding hypotheses to failure."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Test failed",
            expected_vs_actual="Expected pass, got fail"
        )
        failure.add_hypothesis("Off-by-one error", "high")

        assert len(failure.hypotheses) == 1
        assert "[high]" in failure.hypotheses[0]

    def test_record_fix(self):
        """Test recording fix for failure."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Test failed",
            expected_vs_actual="Expected pass, got fail"
        )
        failure.record_fix("Fixed the bug", True)

        assert failure.fix_applied == "Fixed the bug"
        assert failure.fix_successful is True


class TestVerificationSuite:
    """Tests for VerificationSuite class."""

    def test_create_suite(self):
        """Test creating a verification suite."""
        suite = VerificationSuite(
            name="Test Suite",
            description="A test suite"
        )

        assert suite.name == "Test Suite"
        assert len(suite.checks) == 0

    def test_add_check(self):
        """Test adding check to suite."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT
        )

        suite.add_check(check)

        assert len(suite.checks) == 1

    def test_get_checks_for_phase(self):
        """Test getting checks by phase."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(
            name="Draft check",
            description="Drafting check",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING
        )
        check2 = VerificationCheck(
            name="Refine check",
            description="Refining check",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.REFINING
        )

        suite.add_check(check1)
        suite.add_check(check2)

        draft_checks = suite.get_checks_for_phase(VerificationPhase.DRAFTING)
        assert len(draft_checks) == 1
        assert draft_checks[0].name == "Draft check"

    def test_suite_status_pending(self):
        """Test suite all_passed method when all checks pending."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(name="Test", description="Test", level=VerificationLevel.UNIT)
        suite.add_check(check)

        assert suite.all_passed() is False

    def test_suite_status_passed(self):
        """Test suite all_passed when all checks passed."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(name="Test", description="Test", level=VerificationLevel.UNIT)
        check.status = VerificationStatus.PASSED
        suite.add_check(check)

        assert suite.all_passed() is True

    def test_suite_status_failed(self):
        """Test suite all_passed when any check failed."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)
        check1.status = VerificationStatus.PASSED
        check2.status = VerificationStatus.FAILED
        suite.add_check(check1)
        suite.add_check(check2)

        assert suite.all_passed() is False

    def test_get_summary(self):
        """Test getting suite summary."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.INTEGRATION)
        check1.status = VerificationStatus.PASSED
        suite.add_check(check1)
        suite.add_check(check2)

        summary = suite.get_summary()
        assert summary['total_checks'] == 2
        assert summary['by_status']['PASSED'] == 1


class TestVerificationManager:
    """Tests for VerificationManager class."""

    def test_create_manager(self):
        """Test creating verification manager."""
        manager = VerificationManager()

        assert manager is not None

    def test_create_standard_suite(self):
        """Test creating standard verification suite."""
        manager = VerificationManager()
        suite = manager.create_standard_suite("test", "Test Suite")

        assert suite is not None
        assert suite.name == "test"

    def test_get_suite(self):
        """Test getting suite by name."""
        manager = VerificationManager()
        manager.create_standard_suite("test", "Test Suite")

        suite = manager.get_suite("test")
        assert suite is not None

    def test_get_nonexistent_suite(self):
        """Test getting nonexistent suite returns None."""
        manager = VerificationManager()

        suite = manager.get_suite("nonexistent")
        assert suite is None

    def test_register_failure_handler(self):
        """Test registering failure handler."""
        manager = VerificationManager()
        failures = []

        def handler(check, failure):
            failures.append((check.name, failure.observed))

        manager.register_failure_handler(handler)

        # The handler should be registered
        assert len(manager._on_failure) >= 1

    def test_get_summary(self):
        """Test getting manager summary."""
        manager = VerificationManager()
        manager.create_standard_suite("s1", "Suite 1")
        manager.create_standard_suite("s2", "Suite 2")

        summary = manager.get_summary()

        assert "suites" in summary


class TestChecklists:
    """Tests for checklist factory functions."""

    def test_drafting_checklist(self):
        """Test creating drafting checklist."""
        checks = create_drafting_checklist()

        assert checks is not None
        assert len(checks) > 0
        assert all(c.phase == VerificationPhase.DRAFTING for c in checks)

    def test_refining_checklist(self):
        """Test creating refining checklist."""
        checks = create_refining_checklist()

        assert checks is not None
        assert len(checks) > 0
        assert all(c.phase == VerificationPhase.REFINING for c in checks)

    def test_finalizing_checklist(self):
        """Test creating finalizing checklist."""
        checks = create_finalizing_checklist()

        assert checks is not None
        assert len(checks) > 0
        assert all(c.phase == VerificationPhase.FINALIZING for c in checks)


class TestVerificationCheckAdditional:
    """Additional tests for VerificationCheck."""

    def test_mark_skipped(self):
        """Test marking a check as skipped."""
        check = VerificationCheck(
            name="Optional test",
            description="Test optional feature",
            level=VerificationLevel.UNIT,
        )
        check.mark_skipped("Feature not enabled")

        assert check.status == VerificationStatus.SKIPPED
        assert check.result == "Feature not enabled"

    def test_reset(self):
        """Test resetting a check."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_passed("Passed", 100)
        check.reset()

        assert check.status == VerificationStatus.PENDING
        assert check.result is None

    def test_check_with_command(self):
        """Test check with command."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
            command="pytest tests/"
        )

        assert check.command == "pytest tests/"


class TestVerificationSuiteAdditional:
    """Additional tests for VerificationSuite."""

    def test_get_failed_checks(self):
        """Test getting failed checks."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)
        check1.status = VerificationStatus.PASSED
        check2.status = VerificationStatus.FAILED
        suite.add_check(check1)
        suite.add_check(check2)

        failed = suite.get_failed_checks()
        assert len(failed) == 1
        assert failed[0].name == "Test2"

    def test_get_checks_for_level(self):
        """Test getting checks by level."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Unit Test", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Integration Test", description="Test2", level=VerificationLevel.INTEGRATION)
        suite.add_check(check1)
        suite.add_check(check2)

        unit_checks = suite.get_checks_for_level(VerificationLevel.UNIT)
        assert len(unit_checks) == 1

    def test_run_all(self):
        """Test running all checks (stub behavior)."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(name="Test", description="Test", level=VerificationLevel.UNIT)
        suite.add_check(check)

        results = suite.run_all()
        # Stub returns results
        assert results is not None
        assert results['passed'] == 1

    def test_add_unit_check(self):
        """Test adding a unit check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_unit_check("syntax", "Check syntax", command="python -m py_compile")

        assert check.name == "syntax"
        assert check.level == VerificationLevel.UNIT
        assert check.phase == VerificationPhase.DRAFTING
        assert len(suite.checks) == 1

    def test_add_integration_check(self):
        """Test adding an integration check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_integration_check("integration", "Test integration")

        assert check.name == "integration"
        assert check.level == VerificationLevel.INTEGRATION
        assert check.phase == VerificationPhase.REFINING

    def test_add_e2e_check(self):
        """Test adding an E2E check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_e2e_check("e2e", "End-to-end test")

        assert check.name == "e2e"
        assert check.level == VerificationLevel.E2E
        assert check.phase == VerificationPhase.FINALIZING

    def test_add_acceptance_check(self):
        """Test adding an acceptance check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_acceptance_check("acceptance", "User accepts")

        assert check.name == "acceptance"
        assert check.level == VerificationLevel.ACCEPTANCE
        assert check.phase == VerificationPhase.FINALIZING

    def test_get_pending_checks(self):
        """Test getting pending checks."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)
        check1.mark_passed("passed")
        suite.add_check(check1)
        suite.add_check(check2)

        pending = suite.get_pending_checks()
        assert len(pending) == 1
        assert pending[0].name == "Test2"

    def test_record_failure_in_suite(self):
        """Test recording failure in suite."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(name="Test", description="Test", level=VerificationLevel.UNIT)
        suite.add_check(check)

        failure = suite.record_failure(check, "Got error", "Expected pass, got fail")
        assert failure.observed == "Got error"
        assert len(suite.failures) == 1


class TestVerificationCheckError:
    """Test VerificationCheck mark_error method."""

    def test_mark_error(self):
        """Test marking a check as errored."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_error("Command not found")

        assert check.status == VerificationStatus.ERROR
        assert "ERROR: Command not found" in check.result


class TestVerificationFailureAdditional:
    """Additional tests for VerificationFailure."""

    def test_add_investigation_note(self):
        """Test adding investigation note."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Got error",
            expected_vs_actual="Expected pass, got fail"
        )
        failure.add_investigation_note("Checked the logs")

        assert len(failure.investigation_notes) == 1
        assert "Checked the logs" in failure.investigation_notes[0]


class TestVerificationRunner:
    """Tests for VerificationRunner stub class."""

    def test_create_runner(self):
        """Test creating verification runner."""
        runner = VerificationRunner()
        assert runner is not None

    def test_run_check(self):
        """Test running a single check."""
        runner = VerificationRunner()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )

        status = runner.run_check(check)
        assert status == VerificationStatus.PASSED
        assert check.run_at is not None

    def test_run_suite(self):
        """Test running a suite."""
        runner = VerificationRunner()
        suite = VerificationSuite(name="Test Suite")
        suite.add_unit_check("test1", "First test")
        suite.add_unit_check("test2", "Second test")

        results = runner.run_suite(suite)
        assert results['passed'] == 2

    def test_run_suite_for_phase(self):
        """Test running suite filtered by phase."""
        runner = VerificationRunner()
        suite = VerificationSuite(name="Test Suite")
        suite.add_unit_check("draft", "Draft check")  # DRAFTING phase
        suite.add_integration_check("refine", "Refine check")  # REFINING phase

        results = runner.run_suite(suite, phase=VerificationPhase.DRAFTING)
        assert results['passed'] == 1


class TestFailureAnalyzer:
    """Tests for FailureAnalyzer stub class."""

    def test_create_analyzer(self):
        """Test creating failure analyzer."""
        analyzer = FailureAnalyzer()
        assert analyzer is not None

    def test_analyze_failure(self):
        """Test analyzing a failure."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Got error",
            expected_vs_actual="Expected pass, got fail"
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'likely_cause' in analysis
        assert 'hypotheses' in analysis
        assert 'suggested_fix' in analysis

    def test_find_similar_failures(self):
        """Test finding similar failures."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Got error",
            expected_vs_actual="Expected pass, got fail"
        )

        similar = analyzer.find_similar_failures(failure)
        assert isinstance(similar, list)


class TestRegressionDetector:
    """Tests for RegressionDetector stub class."""

    def test_create_detector(self):
        """Test creating regression detector."""
        detector = RegressionDetector()
        assert detector is not None

    def test_detect_regression(self):
        """Test detecting regressions."""
        detector = RegressionDetector()
        baseline = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,
        }
        current = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.FAILED,
        }

        regressions = detector.detect_regression(current, baseline)
        assert len(regressions) == 1
        # New implementation returns detailed dicts
        if isinstance(regressions[0], dict):
            assert regressions[0]['test_name'] == 'test2'
        else:
            assert "test2" in regressions

    def test_no_regression(self):
        """Test when no regressions exist."""
        detector = RegressionDetector()
        baseline = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.FAILED,
        }
        current = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,  # Improved
        }

        regressions = detector.detect_regression(current, baseline)
        assert len(regressions) == 0


class TestVerificationManagerExtended:
    """Extended tests for VerificationManager."""

    def test_report_failure(self):
        """Test reporting a failure through manager."""
        manager = VerificationManager()
        suite = manager.create_suite("test", "Test Suite")
        check = suite.add_unit_check("test", "Test check")

        failure = manager.report_failure(suite, check, "Got 5", "Expected 3, got 5")
        assert failure.observed == "Got 5"

    def test_get_all_failures(self):
        """Test getting all failures across suites."""
        manager = VerificationManager()
        suite1 = manager.create_suite("s1", "Suite 1")
        suite2 = manager.create_suite("s2", "Suite 2")

        check1 = suite1.add_unit_check("test1", "Test 1")
        check2 = suite2.add_unit_check("test2", "Test 2")

        manager.report_failure(suite1, check1, "Error 1", "Mismatch")
        manager.report_failure(suite2, check2, "Error 2", "Mismatch")

        failures = manager.get_all_failures()
        assert len(failures) == 2
