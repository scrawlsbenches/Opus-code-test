"""
Behavioral tests for verification and failure analysis.

These tests verify realistic scenarios for verification workflows,
inspired by the complex-reasoning-workflow.md document Part 6
(Verify: The Quality Loop).

Test scenarios:
- Verification suite execution
- Failure pattern analysis
- Regression detection over time
- Flaky test identification
- Investigation plan generation
"""

import pytest
from datetime import datetime, timedelta

from cortical.reasoning.verification import (
    VerificationCheck,
    VerificationLevel,
    VerificationStatus,
    VerificationSuite,
    VerificationFailure,
    FailureAnalyzer,
    RegressionDetector,
)

# Map abstract levels to actual enum values
MUST_PASS = VerificationLevel.UNIT
SHOULD_PASS = VerificationLevel.INTEGRATION
NICE_TO_HAVE = VerificationLevel.E2E


class TestVerificationWorkflow:
    """Test realistic verification workflows."""

    def test_comprehensive_pre_commit_checks(self):
        """
        Scenario: Run comprehensive checks before committing.
        Expected: All critical checks pass, warnings noted.
        """
        suite = VerificationSuite(name="pre-commit")

        # Add various check levels
        suite.add_check(VerificationCheck(
            name="syntax",
            level=MUST_PASS,
            description="No syntax errors"
        ))
        suite.add_check(VerificationCheck(
            name="types",
            level=MUST_PASS,
            description="Type annotations valid"
        ))
        suite.add_check(VerificationCheck(
            name="unit_tests",
            level=MUST_PASS,
            description="All unit tests pass"
        ))
        suite.add_check(VerificationCheck(
            name="coverage",
            level=SHOULD_PASS,
            description="Coverage above 85%"
        ))
        suite.add_check(VerificationCheck(
            name="lint",
            level=NICE_TO_HAVE,
            description="No lint warnings"
        ))

        # Run all checks
        results = suite.run_all()

        # Verify results structure
        assert 'passed' in results
        assert 'failed' in results
        assert results['passed'] == 5  # All marked passed in stub

    def test_check_status_lifecycle(self):
        """
        Scenario: Track check through its lifecycle.
        Expected: Status transitions correctly.
        """
        check = VerificationCheck(
            name="integration_test",
            level=MUST_PASS,
            description="Integration tests pass"
        )

        # Initial state
        assert check.status == VerificationStatus.PENDING

        # Mark as running (simulated)
        check.mark_passed("All 42 tests passed")
        assert check.status == VerificationStatus.PASSED
        assert "42 tests" in check.result

        # Reset and try failure
        check.reset()
        assert check.status == VerificationStatus.PENDING

        check.mark_failed("3 tests failed: test_auth, test_api, test_db")
        assert check.status == VerificationStatus.FAILED

    def test_record_failure_details(self):
        """
        Scenario: Record detailed failure information.
        Expected: All failure context captured.
        """
        suite = VerificationSuite(name="post-merge")

        check = VerificationCheck(
            name="smoke_test",
            level=MUST_PASS,
            description="Basic functionality works"
        )
        suite.add_check(check)

        # Record a failure
        failure = suite.record_failure(
            check=check,
            observed="ConnectionError: Database connection refused",
            expected_vs_actual="Expected: Application starts successfully, Got: Connection refused"
        )

        assert failure.check == check
        assert "ConnectionError" in failure.observed


class TestFailureAnalysisWorkflow:
    """Test failure analysis and pattern matching."""

    def test_analyze_import_error(self):
        """
        Scenario: Analyze ImportError failure.
        Expected: Matched to import_error pattern with investigation steps.
        """
        analyzer = FailureAnalyzer()

        check = VerificationCheck(
            name="module_load",
            level=MUST_PASS,
            description="All modules load"
        )

        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'missing_package'",
            expected_vs_actual="Expected: Module loads without error"
        )

        analysis = analyzer.analyze_failure(failure)

        assert 'import_error' in analysis['matched_patterns']
        assert 'likely_cause' in analysis
        assert len(analysis['investigation_steps']) > 0
        assert any("requirements" in step.lower() for step in analysis['investigation_steps'])

    def test_analyze_assertion_error(self):
        """
        Scenario: Analyze AssertionError in test.
        Expected: Matched to assertion_error pattern.
        """
        analyzer = FailureAnalyzer()

        check = VerificationCheck(
            name="unit_test",
            level=MUST_PASS,
            description="Test expectations met"
        )

        failure = VerificationFailure(
            check=check,
            observed="AssertionError: assert 41 == 42",
            expected_vs_actual="Expected: result == 42"
        )

        analysis = analyzer.analyze_failure(failure)

        assert 'assertion_error' in analysis['matched_patterns']
        assert any("expected" in step.lower() or "actual" in step.lower()
                   for step in analysis['investigation_steps'])

    def test_analyze_timeout_error(self):
        """
        Scenario: Analyze timeout in performance test.
        Expected: Matched to timeout pattern with performance suggestions.
        """
        analyzer = FailureAnalyzer()

        check = VerificationCheck(
            name="performance_test",
            level=SHOULD_PASS,
            description="Response time < 200ms"
        )

        failure = VerificationFailure(
            check=check,
            observed="TimeoutError: Request timed out after 30000ms",
            expected_vs_actual="Expected: Response in < 200ms"
        )

        analysis = analyzer.analyze_failure(failure)

        assert 'timeout' in analysis['matched_patterns']
        assert any("profile" in step.lower() or "slow" in step.lower()
                   for step in analysis['investigation_steps'])

    def test_analyze_unknown_error_pattern(self):
        """
        Scenario: Analyze error with no known pattern.
        Expected: Generic investigation steps provided.
        """
        analyzer = FailureAnalyzer()

        check = VerificationCheck(
            name="mystery_check",
            level=MUST_PASS,
            description="Something works"
        )

        failure = VerificationFailure(
            check=check,
            observed="UnexpectedWeirdError: Something strange happened in the quantum flux",
            expected_vs_actual="Expected: Success"
        )

        analysis = analyzer.analyze_failure(failure)

        # Should still provide useful output
        assert 'likely_cause' in analysis
        assert len(analysis['investigation_steps']) > 0
        assert analysis['matched_patterns'] == []

    def test_find_similar_historical_failures(self):
        """
        Scenario: Find similar past failures for context.
        Expected: Similar failures ranked by relevance.
        """
        analyzer = FailureAnalyzer()

        check1 = VerificationCheck("auth_test", MUST_PASS, "Auth works")
        check2 = VerificationCheck("auth_integration", MUST_PASS, "Auth integrates")
        check3 = VerificationCheck("unrelated_test", MUST_PASS, "Other thing")

        # Record historical failures
        past_failure1 = VerificationFailure(
            check=check1,
            observed="ConnectionError: Auth service unreachable",
            expected_vs_actual="Expected: Login succeeds"
        )
        past_failure2 = VerificationFailure(
            check=check2,
            observed="ConnectionError: Auth service timeout",
            expected_vs_actual="Expected: Token validates"
        )
        past_failure3 = VerificationFailure(
            check=check3,
            observed="ValueError: Invalid input",
            expected_vs_actual="Expected: Data saves"
        )

        analyzer.record_failure(past_failure1)
        analyzer.record_failure(past_failure2)
        analyzer.record_failure(past_failure3)

        # New similar failure
        new_failure = VerificationFailure(
            check=check1,
            observed="ConnectionError: Auth service down",
            expected_vs_actual="Expected: Login succeeds"
        )

        similar = analyzer.find_similar_failures(new_failure, top_n=5)

        # Should find past auth-related failures as most similar
        assert len(similar) > 0
        # First match should be most similar (same check name)
        most_similar, score = similar[0]
        assert most_similar.check.name in ["auth_test", "auth_integration"]

    def test_generate_investigation_plan(self):
        """
        Scenario: Generate ordered investigation plan.
        Expected: Steps in logical order for investigation.
        """
        analyzer = FailureAnalyzer()

        check = VerificationCheck(
            name="api_test",
            level=MUST_PASS,
            description="API responds correctly"
        )

        failure = VerificationFailure(
            check=check,
            observed="TypeError: 'NoneType' object is not subscriptable",
            expected_vs_actual="Expected: Status 200"
        )

        plan = analyzer.generate_investigation_plan(failure)

        assert isinstance(plan, list)
        assert len(plan) > 0
        # Should have actionable steps
        assert all(isinstance(step, str) for step in plan)


class TestRegressionDetection:
    """Test regression detection across test runs."""

    def test_detect_simple_regression(self):
        """
        Scenario: Test that passed now fails.
        Expected: Regression detected.
        """
        detector = RegressionDetector()

        baseline = {
            "test_login": VerificationStatus.PASSED,
            "test_logout": VerificationStatus.PASSED,
            "test_profile": VerificationStatus.PASSED,
        }

        current = {
            "test_login": VerificationStatus.PASSED,
            "test_logout": VerificationStatus.FAILED,  # Regression!
            "test_profile": VerificationStatus.PASSED,
        }

        detector.save_baseline("main", baseline)
        regressions = detector.detect_regression(current, baseline_name="main")

        assert len(regressions) == 1
        assert regressions[0]['test_name'] == "test_logout"
        assert regressions[0]['baseline_status'] == VerificationStatus.PASSED
        assert regressions[0]['current_status'] == VerificationStatus.FAILED

    def test_detect_improvements(self):
        """
        Scenario: Test that failed now passes.
        Expected: Improvement detected.
        """
        detector = RegressionDetector()

        baseline = {
            "test_broken": VerificationStatus.FAILED,
            "test_ok": VerificationStatus.PASSED,
        }

        current = {
            "test_broken": VerificationStatus.PASSED,  # Fixed!
            "test_ok": VerificationStatus.PASSED,
        }

        detector.save_baseline("before_fix", baseline)
        improvements = detector.detect_improvements(current, baseline_name="before_fix")

        assert len(improvements) == 1
        assert "test_broken" in improvements

    def test_detect_flaky_tests(self):
        """
        Scenario: Test results flip-flop across runs.
        Expected: Test identified as flaky.
        """
        detector = RegressionDetector()

        # Simulate 10 runs with flip-flopping results
        for i in range(10):
            results = {
                "test_stable": VerificationStatus.PASSED,
                "test_flaky": VerificationStatus.PASSED if i % 2 == 0 else VerificationStatus.FAILED,
            }
            detector.record_results(results)

        flaky = detector.detect_flaky_tests(window=10, threshold=0.3)

        assert "test_flaky" in flaky
        assert "test_stable" not in flaky

    def test_get_test_trend(self):
        """
        Scenario: Analyze trend of specific test over time.
        Expected: Pass rate and flakiness calculated.
        """
        detector = RegressionDetector()

        # Simulate 5 runs: PASS, PASS, FAIL, PASS, PASS
        results_sequence = [
            {"test_target": VerificationStatus.PASSED},
            {"test_target": VerificationStatus.PASSED},
            {"test_target": VerificationStatus.FAILED},
            {"test_target": VerificationStatus.PASSED},
            {"test_target": VerificationStatus.PASSED},
        ]

        for results in results_sequence:
            detector.record_results(results)

        trend = detector.get_trend("test_target")

        assert trend['total_runs'] == 5
        assert trend['pass_rate'] == 0.8  # 4/5
        assert trend['flakiness_score'] > 0  # Has some transitions

    def test_regression_summary_statistics(self):
        """
        Scenario: Get overall regression summary.
        Expected: Comprehensive statistics.
        """
        detector = RegressionDetector()

        # Create baseline
        baseline = {
            "test_a": VerificationStatus.PASSED,
            "test_b": VerificationStatus.PASSED,
            "test_c": VerificationStatus.FAILED,
        }
        detector.save_baseline("release-1.0", baseline)

        # Record some history
        detector.record_results({
            "test_a": VerificationStatus.PASSED,
            "test_b": VerificationStatus.FAILED,  # Regression
            "test_c": VerificationStatus.PASSED,  # Improvement
        })

        summary = detector.get_summary()

        assert 'total_tests_tracked' in summary
        assert 'baselines_stored' in summary
        assert summary['baselines_stored'] == 1
        assert summary['total_tests_tracked'] == 3


class TestVerificationIntegration:
    """Test integration scenarios across verification components."""

    def test_full_verification_cycle(self):
        """
        Scenario: Complete verification cycle from checks to analysis.
        Expected: All components work together.
        """
        # Create suite
        suite = VerificationSuite(name="integration")

        check1 = VerificationCheck("api_health", MUST_PASS, "API healthy")
        check2 = VerificationCheck("db_connection", MUST_PASS, "DB connected")
        check3 = VerificationCheck("cache_warm", SHOULD_PASS, "Cache warmed")

        suite.add_check(check1)
        suite.add_check(check2)
        suite.add_check(check3)

        # Simulate running checks
        check1.mark_passed("API responding at 50ms")
        check2.mark_failed("ConnectionError: DB timeout")
        check3.mark_passed("Cache hit rate 95%")

        # Record failure
        failure = suite.record_failure(
            check=check2,
            observed="ConnectionError: Connection refused to db:5432",
            expected_vs_actual="Expected: Connection established"
        )

        # Analyze failure
        analyzer = FailureAnalyzer()
        analysis = analyzer.analyze_failure(failure)

        assert 'connection_error' in analysis['matched_patterns']
        assert len(analysis['investigation_steps']) > 0

        # Track for regression
        detector = RegressionDetector()
        detector.save_baseline("healthy", {
            "api_health": VerificationStatus.PASSED,
            "db_connection": VerificationStatus.PASSED,
            "cache_warm": VerificationStatus.PASSED,
        })

        current = {
            "api_health": check1.status,
            "db_connection": check2.status,
            "cache_warm": check3.status,
        }

        regressions = detector.detect_regression(current, baseline_name="healthy")

        assert len(regressions) == 1
        assert regressions[0]['test_name'] == "db_connection"

    def test_failure_pattern_statistics_accumulate(self):
        """
        Scenario: Track pattern statistics over many failures.
        Expected: Statistics reflect actual patterns.
        """
        analyzer = FailureAnalyzer()

        # Generate various failures
        failures_data = [
            ("test_import_1", "ImportError: No module 'foo'"),
            ("test_import_2", "ModuleNotFoundError: No module 'bar'"),
            ("test_import_3", "ImportError: circular import"),
            ("test_type_1", "TypeError: expected int, got str"),
            ("test_assert_1", "AssertionError: 1 != 2"),
        ]

        for name, observed in failures_data:
            check = VerificationCheck(name, MUST_PASS, "Test")
            failure = VerificationFailure(check, observed, "Expected: Success")
            analyzer.record_failure(failure)

        stats = analyzer.get_pattern_statistics()

        # Import errors should be most common
        assert stats['import_error']['total_occurrences'] == 3
        assert stats['type_error']['total_occurrences'] == 1
        assert stats['assertion_error']['total_occurrences'] == 1
