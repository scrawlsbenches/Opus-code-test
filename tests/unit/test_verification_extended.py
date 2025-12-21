"""
Extended unit tests for Verification module.

This test file provides comprehensive coverage for cortical/reasoning/verification.py,
targeting 70%+ coverage by testing:
- All VerificationCheck status transitions
- VerificationSuite operations and edge cases
- VerificationManager with failure handlers
- Factory checklist functions
- VerificationRunner stub behavior
- FailureAnalyzer pattern matching and analysis
- RegressionDetector baseline tracking and trend analysis
"""

import unittest
from datetime import datetime
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


class TestVerificationLevel(unittest.TestCase):
    """Tests for VerificationLevel enum."""

    def test_levels_ordered(self):
        """Verify verification levels are properly ordered in pyramid structure."""
        self.assertLess(VerificationLevel.UNIT.value, VerificationLevel.INTEGRATION.value)
        self.assertLess(VerificationLevel.INTEGRATION.value, VerificationLevel.E2E.value)
        self.assertLess(VerificationLevel.E2E.value, VerificationLevel.ACCEPTANCE.value)

    def test_all_levels_exist(self):
        """Verify all pyramid levels are defined."""
        self.assertIsNotNone(VerificationLevel.UNIT)
        self.assertIsNotNone(VerificationLevel.INTEGRATION)
        self.assertIsNotNone(VerificationLevel.E2E)
        self.assertIsNotNone(VerificationLevel.ACCEPTANCE)


class TestVerificationPhase(unittest.TestCase):
    """Tests for VerificationPhase enum."""

    def test_all_phases_exist(self):
        """Verify all verification phases exist."""
        self.assertIsNotNone(VerificationPhase.DRAFTING)
        self.assertIsNotNone(VerificationPhase.REFINING)
        self.assertIsNotNone(VerificationPhase.FINALIZING)

    def test_phase_ordering(self):
        """Verify phases are ordered from quick to thorough."""
        self.assertLess(VerificationPhase.DRAFTING.value, VerificationPhase.REFINING.value)
        self.assertLess(VerificationPhase.REFINING.value, VerificationPhase.FINALIZING.value)


class TestVerificationStatus(unittest.TestCase):
    """Tests for VerificationStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all verification statuses exist."""
        self.assertIsNotNone(VerificationStatus.PENDING)
        self.assertIsNotNone(VerificationStatus.RUNNING)
        self.assertIsNotNone(VerificationStatus.PASSED)
        self.assertIsNotNone(VerificationStatus.FAILED)
        self.assertIsNotNone(VerificationStatus.SKIPPED)
        self.assertIsNotNone(VerificationStatus.ERROR)


class TestVerificationCheck(unittest.TestCase):
    """Tests for VerificationCheck dataclass."""

    def test_create_check_minimal(self):
        """Test creating a check with minimal required fields."""
        check = VerificationCheck(
            name="Test passes",
            description="All unit tests should pass",
            level=VerificationLevel.UNIT,
        )

        self.assertEqual(check.name, "Test passes")
        self.assertEqual(check.level, VerificationLevel.UNIT)
        self.assertEqual(check.status, VerificationStatus.PENDING)
        self.assertIsNone(check.result)
        self.assertEqual(check.duration_ms, 0)

    def test_create_check_with_all_fields(self):
        """Test creating a check with all optional fields."""
        check = VerificationCheck(
            name="Syntax check",
            description="Code compiles without errors",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING,
            command="python -m py_compile file.py",
            expected="no errors",
        )

        self.assertEqual(check.phase, VerificationPhase.DRAFTING)
        self.assertEqual(check.command, "python -m py_compile file.py")
        self.assertEqual(check.expected, "no errors")

    def test_mark_passed(self):
        """Test marking a check as passed with result and duration."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_passed("All tests passed", 100)

        self.assertEqual(check.status, VerificationStatus.PASSED)
        self.assertEqual(check.result, "All tests passed")
        self.assertEqual(check.duration_ms, 100)
        self.assertIsNotNone(check.run_at)

    def test_mark_passed_defaults(self):
        """Test marking as passed with default parameters."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_passed()

        self.assertEqual(check.status, VerificationStatus.PASSED)
        self.assertEqual(check.result, "passed")
        self.assertEqual(check.duration_ms, 0)

    def test_mark_failed(self):
        """Test marking a check as failed with error details."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_failed("Test assertion error", 50)

        self.assertEqual(check.status, VerificationStatus.FAILED)
        self.assertEqual(check.result, "Test assertion error")
        self.assertEqual(check.duration_ms, 50)
        self.assertIsNotNone(check.run_at)

    def test_mark_error(self):
        """Test marking a check as errored when the check itself fails."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_error("Command not found")

        self.assertEqual(check.status, VerificationStatus.ERROR)
        self.assertIn("ERROR: Command not found", check.result)
        self.assertIsNotNone(check.run_at)

    def test_mark_skipped(self):
        """Test marking a check as skipped with reason."""
        check = VerificationCheck(
            name="Optional test",
            description="Test optional feature",
            level=VerificationLevel.UNIT,
        )
        check.mark_skipped("Feature not enabled")

        self.assertEqual(check.status, VerificationStatus.SKIPPED)
        self.assertEqual(check.result, "Feature not enabled")
        self.assertIsNotNone(check.run_at)

    def test_mark_skipped_default_reason(self):
        """Test skipping with default reason."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_skipped()

        self.assertEqual(check.status, VerificationStatus.SKIPPED)
        self.assertEqual(check.result, "skipped")

    def test_reset(self):
        """Test resetting a check back to pending state."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        check.mark_passed("Passed", 100)
        check.reset()

        self.assertEqual(check.status, VerificationStatus.PENDING)
        self.assertIsNone(check.result)
        self.assertEqual(check.duration_ms, 0)
        self.assertIsNone(check.run_at)

    def test_status_transitions(self):
        """Test various status transitions to ensure they work correctly."""
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )

        # PENDING -> PASSED -> PENDING
        self.assertEqual(check.status, VerificationStatus.PENDING)
        check.mark_passed()
        self.assertEqual(check.status, VerificationStatus.PASSED)
        check.reset()
        self.assertEqual(check.status, VerificationStatus.PENDING)

        # PENDING -> FAILED -> PENDING
        check.mark_failed("error")
        self.assertEqual(check.status, VerificationStatus.FAILED)
        check.reset()
        self.assertEqual(check.status, VerificationStatus.PENDING)

        # PENDING -> SKIPPED -> PENDING
        check.mark_skipped()
        self.assertEqual(check.status, VerificationStatus.SKIPPED)
        check.reset()
        self.assertEqual(check.status, VerificationStatus.PENDING)

        # PENDING -> ERROR -> PENDING
        check.mark_error("check error")
        self.assertEqual(check.status, VerificationStatus.ERROR)
        check.reset()
        self.assertEqual(check.status, VerificationStatus.PENDING)


class TestVerificationFailure(unittest.TestCase):
    """Tests for VerificationFailure dataclass."""

    def test_create_failure(self):
        """Test creating a verification failure record."""
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

        self.assertIs(failure.check, check)
        self.assertEqual(failure.observed, "Got 3 instead of 5")
        self.assertEqual(failure.expected_vs_actual, "Expected 5, got 3")
        self.assertEqual(len(failure.hypotheses), 0)
        self.assertEqual(len(failure.investigation_notes), 0)
        self.assertIsNone(failure.fix_applied)
        self.assertFalse(failure.fix_successful)

    def test_add_hypothesis_with_likelihood(self):
        """Test adding hypotheses with different likelihood levels."""
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
        failure.add_hypothesis("Race condition", "low")

        self.assertEqual(len(failure.hypotheses), 2)
        self.assertIn("[high]", failure.hypotheses[0])
        self.assertIn("Off-by-one error", failure.hypotheses[0])
        self.assertIn("[low]", failure.hypotheses[1])

    def test_add_hypothesis_default_likelihood(self):
        """Test adding hypothesis with default likelihood."""
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

        failure.add_hypothesis("Logic error")

        self.assertEqual(len(failure.hypotheses), 1)
        self.assertIn("[medium]", failure.hypotheses[0])

    def test_add_investigation_note(self):
        """Test adding investigation notes with timestamps."""
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
        failure.add_investigation_note("Found null pointer")

        self.assertEqual(len(failure.investigation_notes), 2)
        self.assertIn("Checked the logs", failure.investigation_notes[0])
        self.assertIn("Found null pointer", failure.investigation_notes[1])

    def test_record_fix_successful(self):
        """Test recording a successful fix."""
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

        self.assertEqual(failure.fix_applied, "Fixed the bug")
        self.assertTrue(failure.fix_successful)

    def test_record_fix_unsuccessful(self):
        """Test recording an unsuccessful fix attempt."""
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

        failure.record_fix("Tried fixing the bug", False)

        self.assertEqual(failure.fix_applied, "Tried fixing the bug")
        self.assertFalse(failure.fix_successful)


class TestVerificationSuite(unittest.TestCase):
    """Tests for VerificationSuite class."""

    def test_create_empty_suite(self):
        """Test creating an empty verification suite."""
        suite = VerificationSuite(
            name="Test Suite",
            description="A test suite"
        )

        self.assertEqual(suite.name, "Test Suite")
        self.assertEqual(suite.description, "A test suite")
        self.assertEqual(len(suite.checks), 0)
        self.assertEqual(len(suite.failures), 0)

    def test_add_check(self):
        """Test adding a check to the suite."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT
        )

        suite.add_check(check)

        self.assertEqual(len(suite.checks), 1)
        self.assertIn(check, suite.checks)

    def test_add_multiple_checks(self):
        """Test adding multiple checks to a suite."""
        suite = VerificationSuite(name="Test Suite")

        for i in range(5):
            check = VerificationCheck(
                name=f"Test {i}",
                description=f"Test check {i}",
                level=VerificationLevel.UNIT
            )
            suite.add_check(check)

        self.assertEqual(len(suite.checks), 5)

    def test_add_unit_check(self):
        """Test adding a unit check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_unit_check("syntax", "Check syntax", command="python -m py_compile")

        self.assertEqual(check.name, "syntax")
        self.assertEqual(check.level, VerificationLevel.UNIT)
        self.assertEqual(check.phase, VerificationPhase.DRAFTING)
        self.assertEqual(check.command, "python -m py_compile")
        self.assertEqual(len(suite.checks), 1)

    def test_add_integration_check(self):
        """Test adding an integration check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_integration_check("integration", "Test integration")

        self.assertEqual(check.name, "integration")
        self.assertEqual(check.level, VerificationLevel.INTEGRATION)
        self.assertEqual(check.phase, VerificationPhase.REFINING)
        self.assertIn(check, suite.checks)

    def test_add_e2e_check(self):
        """Test adding an E2E check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_e2e_check("e2e", "End-to-end test")

        self.assertEqual(check.name, "e2e")
        self.assertEqual(check.level, VerificationLevel.E2E)
        self.assertEqual(check.phase, VerificationPhase.FINALIZING)

    def test_add_acceptance_check(self):
        """Test adding an acceptance check via helper method."""
        suite = VerificationSuite(name="Test Suite")
        check = suite.add_acceptance_check("acceptance", "User accepts")

        self.assertEqual(check.name, "acceptance")
        self.assertEqual(check.level, VerificationLevel.ACCEPTANCE)
        self.assertEqual(check.phase, VerificationPhase.FINALIZING)
        self.assertIsNone(check.command)  # Acceptance checks don't have commands

    def test_get_checks_for_phase(self):
        """Test filtering checks by phase."""
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
        check3 = VerificationCheck(
            name="Final check",
            description="Finalizing check",
            level=VerificationLevel.INTEGRATION,
            phase=VerificationPhase.FINALIZING
        )

        suite.add_check(check1)
        suite.add_check(check2)
        suite.add_check(check3)

        draft_checks = suite.get_checks_for_phase(VerificationPhase.DRAFTING)
        refine_checks = suite.get_checks_for_phase(VerificationPhase.REFINING)
        final_checks = suite.get_checks_for_phase(VerificationPhase.FINALIZING)

        self.assertEqual(len(draft_checks), 1)
        self.assertEqual(draft_checks[0].name, "Draft check")
        self.assertEqual(len(refine_checks), 1)
        self.assertEqual(len(final_checks), 1)

    def test_get_checks_for_level(self):
        """Test filtering checks by pyramid level."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Unit Test", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Integration Test", description="Test2", level=VerificationLevel.INTEGRATION)
        check3 = VerificationCheck(name="Another Unit Test", description="Test3", level=VerificationLevel.UNIT)

        suite.add_check(check1)
        suite.add_check(check2)
        suite.add_check(check3)

        unit_checks = suite.get_checks_for_level(VerificationLevel.UNIT)
        integration_checks = suite.get_checks_for_level(VerificationLevel.INTEGRATION)

        self.assertEqual(len(unit_checks), 2)
        self.assertEqual(len(integration_checks), 1)

    def test_get_pending_checks(self):
        """Test getting all pending checks."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)
        check3 = VerificationCheck(name="Test3", description="Test3", level=VerificationLevel.UNIT)

        check1.mark_passed("passed")
        check2.mark_failed("failed")
        # check3 remains pending

        suite.add_check(check1)
        suite.add_check(check2)
        suite.add_check(check3)

        pending = suite.get_pending_checks()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].name, "Test3")

    def test_get_failed_checks(self):
        """Test getting all failed checks."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)
        check3 = VerificationCheck(name="Test3", description="Test3", level=VerificationLevel.UNIT)

        check1.mark_passed("passed")
        check2.mark_failed("failed")
        check3.mark_failed("also failed")

        suite.add_check(check1)
        suite.add_check(check2)
        suite.add_check(check3)

        failed = suite.get_failed_checks()
        self.assertEqual(len(failed), 2)
        self.assertIn("Test2", [c.name for c in failed])
        self.assertIn("Test3", [c.name for c in failed])

    def test_record_failure(self):
        """Test recording a failure in the suite."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(name="Test", description="Test", level=VerificationLevel.UNIT)
        suite.add_check(check)

        failure = suite.record_failure(check, "Got error", "Expected pass, got fail")

        self.assertEqual(failure.observed, "Got error")
        self.assertEqual(failure.expected_vs_actual, "Expected pass, got fail")
        self.assertEqual(len(suite.failures), 1)
        self.assertIn(failure, suite.failures)

    def test_all_passed_empty_suite(self):
        """Test all_passed on empty suite returns True."""
        suite = VerificationSuite(name="Empty Suite")
        self.assertTrue(suite.all_passed())

    def test_all_passed_when_all_pending(self):
        """Test all_passed returns False when checks are pending."""
        suite = VerificationSuite(name="Test Suite")
        check = VerificationCheck(name="Test", description="Test", level=VerificationLevel.UNIT)
        suite.add_check(check)

        self.assertFalse(suite.all_passed())

    def test_all_passed_when_all_checks_passed(self):
        """Test all_passed returns True when all checks passed."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)

        check1.mark_passed()
        check2.mark_passed()

        suite.add_check(check1)
        suite.add_check(check2)

        self.assertTrue(suite.all_passed())

    def test_all_passed_with_one_failure(self):
        """Test all_passed returns False when any check failed."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.UNIT)

        check1.mark_passed()
        check2.mark_failed("error")

        suite.add_check(check1)
        suite.add_check(check2)

        self.assertFalse(suite.all_passed())

    def test_run_all(self):
        """Test running all checks (stub implementation)."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test", level=VerificationLevel.UNIT)

        suite.add_check(check1)
        suite.add_check(check2)

        results = suite.run_all()

        # Stub marks all as passed
        self.assertIsNotNone(results)
        self.assertEqual(results['passed'], 2)
        self.assertEqual(check1.status, VerificationStatus.PASSED)
        self.assertEqual(check2.status, VerificationStatus.PASSED)

    def test_get_summary(self):
        """Test getting comprehensive suite summary."""
        suite = VerificationSuite(name="Test Suite")
        check1 = VerificationCheck(name="Test1", description="Test1", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test2", description="Test2", level=VerificationLevel.INTEGRATION)
        check3 = VerificationCheck(name="Test3", description="Test3", level=VerificationLevel.UNIT)

        check1.mark_passed()
        check2.mark_failed("error")
        # check3 pending

        suite.add_check(check1)
        suite.add_check(check2)
        suite.add_check(check3)

        summary = suite.get_summary()

        self.assertEqual(summary['total_checks'], 3)
        self.assertEqual(summary['by_status']['PASSED'], 1)
        self.assertEqual(summary['by_status']['FAILED'], 1)
        self.assertEqual(summary['by_status']['PENDING'], 1)
        self.assertFalse(summary['all_passed'])
        self.assertEqual(summary['failure_count'], 0)


class TestFactoryChecklists(unittest.TestCase):
    """Tests for checklist factory functions."""

    def test_create_drafting_checklist(self):
        """Test creating standard drafting checklist."""
        checks = create_drafting_checklist()

        self.assertIsNotNone(checks)
        self.assertGreater(len(checks), 0)
        self.assertTrue(all(c.phase == VerificationPhase.DRAFTING for c in checks))
        self.assertTrue(all(c.level == VerificationLevel.UNIT for c in checks))

        # Check for expected standard checks
        check_names = [c.name for c in checks]
        self.assertIn("syntax_check", check_names)
        self.assertIn("smoke_test", check_names)
        self.assertIn("import_check", check_names)

    def test_create_refining_checklist(self):
        """Test creating standard refining checklist."""
        checks = create_refining_checklist()

        self.assertIsNotNone(checks)
        self.assertGreater(len(checks), 0)
        self.assertTrue(all(c.phase == VerificationPhase.REFINING for c in checks))

        # Check for expected standard checks
        check_names = [c.name for c in checks]
        self.assertIn("unit_tests", check_names)
        self.assertIn("coverage_check", check_names)
        self.assertIn("type_check", check_names)

    def test_create_finalizing_checklist(self):
        """Test creating standard finalizing checklist."""
        checks = create_finalizing_checklist()

        self.assertIsNotNone(checks)
        self.assertGreater(len(checks), 0)
        self.assertTrue(all(c.phase == VerificationPhase.FINALIZING for c in checks))

        # Check for expected standard checks
        check_names = [c.name for c in checks]
        self.assertIn("full_test_suite", check_names)
        self.assertIn("integration_tests", check_names)
        self.assertIn("performance_tests", check_names)
        self.assertIn("documentation_check", check_names)

    def test_checklist_commands_present(self):
        """Test that checklists include executable commands."""
        drafting = create_drafting_checklist()
        refining = create_refining_checklist()

        # Drafting and refining should have commands
        self.assertTrue(any(c.command is not None for c in drafting))
        self.assertTrue(any(c.command is not None for c in refining))


class TestVerificationManager(unittest.TestCase):
    """Tests for VerificationManager class."""

    def test_create_manager(self):
        """Test creating an empty verification manager."""
        manager = VerificationManager()

        self.assertIsNotNone(manager)
        self.assertEqual(len(manager._suites), 0)

    def test_create_suite(self):
        """Test creating a suite via manager."""
        manager = VerificationManager()
        suite = manager.create_suite("test", "Test Suite")

        self.assertIsNotNone(suite)
        self.assertEqual(suite.name, "test")
        self.assertEqual(suite.description, "Test Suite")
        self.assertIn("test", manager._suites)

    def test_create_multiple_suites(self):
        """Test creating multiple suites."""
        manager = VerificationManager()
        suite1 = manager.create_suite("s1", "Suite 1")
        suite2 = manager.create_suite("s2", "Suite 2")

        self.assertEqual(len(manager._suites), 2)
        self.assertIs(manager._suites["s1"], suite1)
        self.assertIs(manager._suites["s2"], suite2)

    def test_create_standard_suite(self):
        """Test creating standard verification suite with all phases."""
        manager = VerificationManager()
        suite = manager.create_standard_suite("test", "Test Suite")

        self.assertIsNotNone(suite)
        self.assertEqual(suite.name, "test")
        self.assertGreater(len(suite.checks), 0)

        # Should have checks from all phases
        drafting = suite.get_checks_for_phase(VerificationPhase.DRAFTING)
        refining = suite.get_checks_for_phase(VerificationPhase.REFINING)
        finalizing = suite.get_checks_for_phase(VerificationPhase.FINALIZING)

        self.assertGreater(len(drafting), 0)
        self.assertGreater(len(refining), 0)
        self.assertGreater(len(finalizing), 0)

    def test_get_suite(self):
        """Test retrieving suite by name."""
        manager = VerificationManager()
        suite = manager.create_suite("test", "Test Suite")

        retrieved = manager.get_suite("test")

        self.assertIs(retrieved, suite)

    def test_get_nonexistent_suite(self):
        """Test getting nonexistent suite returns None."""
        manager = VerificationManager()

        suite = manager.get_suite("nonexistent")

        self.assertIsNone(suite)

    def test_register_failure_handler(self):
        """Test registering a failure handler callback."""
        manager = VerificationManager()
        called = []

        def handler(check, failure):
            called.append((check.name, failure.observed))

        manager.register_failure_handler(handler)

        self.assertIn(handler, manager._on_failure)

    def test_report_failure_calls_handlers(self):
        """Test that reporting a failure triggers registered handlers."""
        manager = VerificationManager()
        suite = manager.create_suite("test", "Test Suite")
        check = suite.add_unit_check("test", "Test check")

        called = []

        def handler(c, f):
            called.append((c.name, f.observed))

        manager.register_failure_handler(handler)

        failure = manager.report_failure(suite, check, "Got 5", "Expected 3, got 5")

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][0], "test")
        self.assertEqual(called[0][1], "Got 5")

    def test_report_failure_handler_exception_handling(self):
        """Test that handler exceptions don't break failure reporting."""
        manager = VerificationManager()
        suite = manager.create_suite("test", "Test Suite")
        check = suite.add_unit_check("test", "Test check")

        def bad_handler(c, f):
            raise ValueError("Handler error")

        manager.register_failure_handler(bad_handler)

        # Should not raise
        failure = manager.report_failure(suite, check, "Got error", "Mismatch")

        self.assertIsNotNone(failure)

    def test_get_all_failures(self):
        """Test getting all failures across all suites."""
        manager = VerificationManager()
        suite1 = manager.create_suite("s1", "Suite 1")
        suite2 = manager.create_suite("s2", "Suite 2")

        check1 = suite1.add_unit_check("test1", "Test 1")
        check2 = suite2.add_unit_check("test2", "Test 2")

        manager.report_failure(suite1, check1, "Error 1", "Mismatch")
        manager.report_failure(suite2, check2, "Error 2", "Mismatch")

        failures = manager.get_all_failures()

        self.assertEqual(len(failures), 2)

    def test_get_summary_empty(self):
        """Test getting summary from empty manager."""
        manager = VerificationManager()

        summary = manager.get_summary()

        self.assertEqual(summary['suites'], 0)
        self.assertEqual(summary['total_checks'], 0)
        self.assertEqual(summary['passed'], 0)
        self.assertEqual(summary['failed'], 0)

    def test_get_summary_with_suites(self):
        """Test getting aggregate summary across suites."""
        manager = VerificationManager()
        suite1 = manager.create_suite("s1", "Suite 1")
        suite2 = manager.create_suite("s2", "Suite 2")

        check1 = suite1.add_unit_check("test1", "Test 1")
        check2 = suite2.add_unit_check("test2", "Test 2")
        check3 = suite2.add_unit_check("test3", "Test 3")

        check1.mark_passed()
        check2.mark_passed()
        check3.mark_failed("error")

        summary = manager.get_summary()

        self.assertEqual(summary['suites'], 2)
        self.assertEqual(summary['total_checks'], 3)
        self.assertEqual(summary['passed'], 2)
        self.assertEqual(summary['failed'], 1)
        self.assertAlmostEqual(summary['pass_rate'], 2/3, places=2)


class TestVerificationRunner(unittest.TestCase):
    """Tests for VerificationRunner stub class."""

    def test_create_runner(self):
        """Test creating verification runner."""
        runner = VerificationRunner()
        self.assertIsNotNone(runner)

    def test_run_check(self):
        """Test running a single check (stub implementation)."""
        runner = VerificationRunner()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )

        status = runner.run_check(check)

        self.assertEqual(status, VerificationStatus.PASSED)
        self.assertEqual(check.status, VerificationStatus.PASSED)
        self.assertIsNotNone(check.run_at)

    def test_run_suite(self):
        """Test running entire suite (stub implementation)."""
        runner = VerificationRunner()
        suite = VerificationSuite(name="Test Suite")
        suite.add_unit_check("test1", "First test")
        suite.add_unit_check("test2", "Second test")

        results = runner.run_suite(suite)

        self.assertIsNotNone(results)
        self.assertEqual(results['passed'], 2)
        self.assertEqual(results['failed'], 0)
        self.assertEqual(results['error'], 0)

    def test_run_suite_for_specific_phase(self):
        """Test running suite filtered by phase."""
        runner = VerificationRunner()
        suite = VerificationSuite(name="Test Suite")
        suite.add_unit_check("draft", "Draft check")  # DRAFTING phase
        suite.add_integration_check("refine", "Refine check")  # REFINING phase

        results = runner.run_suite(suite, phase=VerificationPhase.DRAFTING)

        self.assertEqual(results['passed'], 1)

    def test_run_suite_all_phases(self):
        """Test running suite with no phase filter."""
        runner = VerificationRunner()
        suite = VerificationSuite(name="Test Suite")
        suite.add_unit_check("draft", "Draft check")
        suite.add_integration_check("refine", "Refine check")
        suite.add_e2e_check("final", "Final check")

        results = runner.run_suite(suite, phase=None)

        self.assertEqual(results['passed'], 3)


class TestFailureAnalyzer(unittest.TestCase):
    """Tests for FailureAnalyzer class."""

    def test_create_analyzer(self):
        """Test creating failure analyzer with pattern library."""
        analyzer = FailureAnalyzer()

        self.assertIsNotNone(analyzer)
        self.assertGreater(len(analyzer._patterns), 0)

    def test_analyze_import_error(self):
        """Test analyzing an import error failure."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work, got ImportError"
        )

        analysis = analyzer.analyze_failure(failure)

        self.assertIn('import_error', analysis['matched_patterns'])
        self.assertIn('likely_cause', analysis)
        self.assertIn('dependency', analysis['likely_cause'].lower())
        self.assertIn('hypotheses', analysis)
        self.assertGreater(len(analysis['hypotheses']), 0)

    def test_analyze_assertion_error(self):
        """Test analyzing an assertion error failure."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: assert 5 == 3",
            expected_vs_actual="Expected 3, got 5"
        )

        analysis = analyzer.analyze_failure(failure)

        self.assertIn('assertion_error', analysis['matched_patterns'])
        self.assertIn('expectation', analysis['likely_cause'].lower())

    def test_analyze_timeout_error(self):
        """Test analyzing a timeout failure."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="TimeoutError: operation timed out after 30s",
            expected_vs_actual="Expected completion, got timeout"
        )

        analysis = analyzer.analyze_failure(failure)

        self.assertIn('timeout', analysis['matched_patterns'])
        self.assertIn('took too long', analysis['likely_cause'].lower())

    def test_analyze_unknown_pattern(self):
        """Test analyzing a failure with no known pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Some weird error that doesn't match patterns",
            expected_vs_actual="Expected pass, got weird error"
        )

        analysis = analyzer.analyze_failure(failure)

        self.assertEqual(len(analysis['matched_patterns']), 0)
        self.assertEqual(analysis['likely_cause'], 'Unknown error pattern')
        self.assertGreater(len(analysis['hypotheses']), 0)
        self.assertGreater(len(analysis['investigation_steps']), 0)

    def test_record_failure(self):
        """Test recording a failure in analyzer history."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Got error",
            expected_vs_actual="Expected pass, got error"
        )

        analyzer.record_failure(failure)

        self.assertEqual(len(analyzer._history), 1)
        self.assertIn(failure, analyzer._history)

    def test_record_fix_success(self):
        """Test recording successful fix updates statistics."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work"
        )

        analyzer.record_failure(failure)
        analyzer.record_fix_success(failure, True)

        stats = analyzer.get_pattern_statistics()
        self.assertEqual(stats['import_error']['successful_fixes'], 1)

    def test_find_similar_failures_empty_history(self):
        """Test finding similar failures with no history."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Got error",
            expected_vs_actual="Expected pass"
        )

        similar = analyzer.find_similar_failures(failure)

        self.assertEqual(len(similar), 0)

    def test_find_similar_failures_with_history(self):
        """Test finding similar failures in history."""
        analyzer = FailureAnalyzer()
        check1 = VerificationCheck(name="Test", description="Test check", level=VerificationLevel.UNIT)
        check2 = VerificationCheck(name="Test", description="Test check", level=VerificationLevel.UNIT)

        failure1 = VerificationFailure(
            check=check1,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import"
        )
        failure2 = VerificationFailure(
            check=check2,
            observed="ImportError: No module named 'bar'",
            expected_vs_actual="Expected import"
        )

        analyzer.record_failure(failure1)

        similar = analyzer.find_similar_failures(failure2, top_n=5)

        self.assertGreater(len(similar), 0)
        self.assertIsInstance(similar[0], tuple)
        self.assertEqual(len(similar[0]), 2)  # (failure, similarity_score)

    def test_generate_investigation_plan(self):
        """Test generating investigation plan from failure."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="Test",
            description="Test check",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work"
        )

        plan = analyzer.generate_investigation_plan(failure)

        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)

    def test_get_pattern_statistics(self):
        """Test getting pattern occurrence statistics."""
        analyzer = FailureAnalyzer()

        stats = analyzer.get_pattern_statistics()

        self.assertIn('import_error', stats)
        self.assertIn('assertion_error', stats)
        self.assertIn('timeout', stats)

        for pattern_name, pattern_stats in stats.items():
            self.assertIn('total_occurrences', pattern_stats)
            self.assertIn('successful_fixes', pattern_stats)
            self.assertIn('success_rate', pattern_stats)


class TestRegressionDetector(unittest.TestCase):
    """Tests for RegressionDetector class."""

    def test_create_detector(self):
        """Test creating regression detector."""
        detector = RegressionDetector()

        self.assertIsNotNone(detector)
        self.assertEqual(len(detector._baselines), 0)
        self.assertEqual(len(detector._history), 0)

    def test_save_baseline(self):
        """Test saving a named baseline."""
        detector = RegressionDetector()
        results = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,
        }

        detector.save_baseline("main", results)

        self.assertIn("main", detector._baselines)
        self.assertEqual(detector._baselines["main"], results)

    def test_save_multiple_baselines(self):
        """Test saving multiple named baselines."""
        detector = RegressionDetector()

        detector.save_baseline("main", {"test1": VerificationStatus.PASSED})
        detector.save_baseline("dev", {"test1": VerificationStatus.FAILED})

        self.assertEqual(len(detector._baselines), 2)
        self.assertIn("main", detector._baselines)
        self.assertIn("dev", detector._baselines)

    def test_detect_regression_with_baseline(self):
        """Test detecting regressions against a baseline."""
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

        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]['test_name'], 'test2')
        self.assertEqual(regressions[0]['baseline_status'], VerificationStatus.PASSED)
        self.assertEqual(regressions[0]['current_status'], VerificationStatus.FAILED)

    def test_detect_no_regression(self):
        """Test when no regressions exist."""
        detector = RegressionDetector()
        baseline = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.FAILED,
        }
        current = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,
        }

        regressions = detector.detect_regression(current, baseline)

        self.assertEqual(len(regressions), 0)

    def test_detect_regression_with_named_baseline(self):
        """Test detecting regressions using named baseline."""
        detector = RegressionDetector()
        baseline = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,
        }
        current = {
            "test1": VerificationStatus.FAILED,
            "test2": VerificationStatus.PASSED,
        }

        detector.save_baseline("release", baseline)
        regressions = detector.detect_regression(current, baseline_name="release")

        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]['test_name'], 'test1')

    def test_detect_improvements(self):
        """Test detecting improvements (tests that went from FAILED to PASSED)."""
        detector = RegressionDetector()
        baseline = {
            "test1": VerificationStatus.FAILED,
            "test2": VerificationStatus.PASSED,
        }
        current = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,
        }

        improvements = detector.detect_improvements(current, baseline)

        self.assertEqual(len(improvements), 1)
        self.assertIn("test1", improvements)

    def test_detect_improvements_with_named_baseline(self):
        """Test detecting improvements using named baseline."""
        detector = RegressionDetector()
        baseline = {
            "test1": VerificationStatus.FAILED,
            "test2": VerificationStatus.FAILED,
        }
        current = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.PASSED,
        }

        detector.save_baseline("v1.0", baseline)
        improvements = detector.detect_improvements(current, baseline_name="v1.0")

        self.assertEqual(len(improvements), 2)

    def test_record_results(self):
        """Test recording results adds to history."""
        detector = RegressionDetector()
        results = {
            "test1": VerificationStatus.PASSED,
            "test2": VerificationStatus.FAILED,
        }

        detector.record_results(results)

        self.assertEqual(len(detector._history), 1)
        self.assertEqual(detector._history[0]['results'], results)

    def test_detect_flaky_tests(self):
        """Test detecting flaky tests that flip between pass/fail."""
        detector = RegressionDetector()

        # Record 10 results with test1 flipping
        for i in range(10):
            results = {
                "test1": VerificationStatus.PASSED if i % 2 == 0 else VerificationStatus.FAILED,
                "test2": VerificationStatus.PASSED,
            }
            detector.record_results(results)

        flaky = detector.detect_flaky_tests(window=10, threshold=0.3)

        self.assertIn("test1", flaky)
        self.assertNotIn("test2", flaky)

    def test_detect_flaky_tests_insufficient_history(self):
        """Test flaky detection with insufficient history."""
        detector = RegressionDetector()

        # Record only 3 results
        for i in range(3):
            detector.record_results({"test1": VerificationStatus.PASSED})

        flaky = detector.detect_flaky_tests(window=10, threshold=0.3)

        # Not enough history, so test1 shouldn't be flagged
        self.assertEqual(len(flaky), 0)

    def test_get_trend(self):
        """Test getting trend for a specific test."""
        detector = RegressionDetector()

        # Record some results
        for i in range(5):
            results = {
                "test1": VerificationStatus.PASSED if i < 3 else VerificationStatus.FAILED,
            }
            detector.record_results(results)

        trend = detector.get_trend("test1")

        self.assertEqual(len(trend['history']), 5)
        self.assertEqual(trend['total_runs'], 5)
        self.assertAlmostEqual(trend['pass_rate'], 0.6, places=2)

    def test_get_trend_with_limit(self):
        """Test getting limited trend history."""
        detector = RegressionDetector()

        # Record 10 results
        for i in range(10):
            detector.record_results({"test1": VerificationStatus.PASSED})

        trend = detector.get_trend("test1", limit=3)

        self.assertEqual(len(trend['history']), 3)

    def test_get_trend_nonexistent_test(self):
        """Test getting trend for test not in history."""
        detector = RegressionDetector()

        trend = detector.get_trend("nonexistent")

        self.assertEqual(len(trend['history']), 0)
        self.assertEqual(trend['pass_rate'], 0.0)
        self.assertEqual(trend['total_runs'], 0)

    def test_get_summary(self):
        """Test getting comprehensive summary statistics."""
        detector = RegressionDetector()

        # Add baseline
        detector.save_baseline("main", {"test1": VerificationStatus.PASSED})

        # Record some results
        detector.record_results({"test1": VerificationStatus.PASSED})
        detector.record_results({"test1": VerificationStatus.FAILED})

        summary = detector.get_summary()

        self.assertIn('total_tests_tracked', summary)
        self.assertIn('baselines_stored', summary)
        self.assertIn('total_snapshots', summary)
        self.assertEqual(summary['baselines_stored'], 1)
        self.assertEqual(summary['total_snapshots'], 2)


if __name__ == '__main__':
    unittest.main()
