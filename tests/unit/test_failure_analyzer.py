"""
Unit tests for FailureAnalyzer class.

Tests cover:
- Pattern matching for each known error type
- Failure analysis structure and content
- Similar failure detection and ranking
- Investigation plan generation
- Pattern statistics tracking
- Historical failure recording
"""

import pytest
from cortical.reasoning import (
    VerificationCheck,
    VerificationFailure,
    VerificationLevel,
)
from cortical.reasoning.verification import FailureAnalyzer


class TestFailureAnalyzerInit:
    """Tests for FailureAnalyzer initialization."""

    def test_create_analyzer(self):
        """Test creating a failure analyzer."""
        analyzer = FailureAnalyzer()
        assert analyzer is not None
        assert analyzer._patterns is not None
        assert analyzer._history is not None
        assert analyzer._pattern_stats is not None

    def test_patterns_initialized(self):
        """Test that patterns are initialized."""
        analyzer = FailureAnalyzer()
        assert len(analyzer._patterns) > 0
        # Check for key patterns
        assert 'import_error' in analyzer._patterns
        assert 'assertion_error' in analyzer._patterns
        assert 'timeout' in analyzer._patterns
        assert 'connection_error' in analyzer._patterns
        assert 'type_error' in analyzer._patterns

    def test_pattern_structure(self):
        """Test that patterns have required fields."""
        analyzer = FailureAnalyzer()
        for pattern_name, pattern_data in analyzer._patterns.items():
            assert 'pattern' in pattern_data, f"{pattern_name} missing 'pattern'"
            assert 'likely_cause' in pattern_data, f"{pattern_name} missing 'likely_cause'"
            assert 'fix_template' in pattern_data, f"{pattern_name} missing 'fix_template'"
            assert 'investigation_steps' in pattern_data, f"{pattern_name} missing 'investigation_steps'"
            assert isinstance(pattern_data['investigation_steps'], list)

    def test_statistics_initialized(self):
        """Test that pattern statistics are initialized."""
        analyzer = FailureAnalyzer()
        assert len(analyzer._pattern_stats) > 0
        for pattern_name in analyzer._patterns:
            assert pattern_name in analyzer._pattern_stats
            assert 'total_occurrences' in analyzer._pattern_stats[pattern_name]
            assert 'successful_fixes' in analyzer._pattern_stats[pattern_name]


class TestPatternMatching:
    """Tests for pattern matching against known error types."""

    def test_import_error_match(self):
        """Test matching ImportError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="import_test",
            description="Test import",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'missing_package'",
            expected_vs_actual="Expected import to succeed",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'import_error' in analysis['matched_patterns']
        assert 'Missing dependency' in analysis['likely_cause']

    def test_module_not_found_match(self):
        """Test matching ModuleNotFoundError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="import_test",
            description="Test import",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ModuleNotFoundError: No module named 'foo'",
            expected_vs_actual="Expected import to succeed",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'import_error' in analysis['matched_patterns']

    def test_assertion_error_match(self):
        """Test matching AssertionError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="unit_test",
            description="Test assertion",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: expected 5 but got 3",
            expected_vs_actual="Expected 5, got 3",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'assertion_error' in analysis['matched_patterns']
        assert 'expectation not met' in analysis['likely_cause']

    def test_timeout_error_match(self):
        """Test matching TimeoutError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="performance_test",
            description="Test timeout",
            level=VerificationLevel.E2E,
        )
        failure = VerificationFailure(
            check=check,
            observed="TimeoutError: operation timed out after 30s",
            expected_vs_actual="Expected completion within 30s",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'timeout' in analysis['matched_patterns']
        assert 'too long' in analysis['likely_cause']

    def test_connection_error_match(self):
        """Test matching ConnectionError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="integration_test",
            description="Test connection",
            level=VerificationLevel.INTEGRATION,
        )
        failure = VerificationFailure(
            check=check,
            observed="ConnectionError: Connection refused",
            expected_vs_actual="Expected successful connection",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'connection_error' in analysis['matched_patterns']
        assert 'unavailable' in analysis['likely_cause']

    def test_type_error_match(self):
        """Test matching TypeError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="unit_test",
            description="Test types",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="TypeError: foo() takes 2 positional arguments but 3 were given",
            expected_vs_actual="Expected correct arguments",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'type_error' in analysis['matched_patterns']
        assert 'argument' in analysis['likely_cause']

    def test_attribute_error_match(self):
        """Test matching AttributeError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="unit_test",
            description="Test attribute",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AttributeError: 'NoneType' object has no attribute 'foo'",
            expected_vs_actual="Expected object to have attribute",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'attribute_error' in analysis['matched_patterns']
        assert 'missing expected attribute' in analysis['likely_cause']

    def test_key_error_match(self):
        """Test matching KeyError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="unit_test",
            description="Test key",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="KeyError: 'missing_key'",
            expected_vs_actual="Expected key to exist",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'key_error' in analysis['matched_patterns']
        assert 'key not found' in analysis['likely_cause']

    def test_index_error_match(self):
        """Test matching IndexError pattern."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="unit_test",
            description="Test index",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="IndexError: list index out of range",
            expected_vs_actual="Expected valid index",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'index_error' in analysis['matched_patterns']
        assert 'out of bounds' in analysis['likely_cause']

    def test_no_pattern_match(self):
        """Test behavior when no pattern matches."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="unknown_test",
            description="Test unknown",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="SomeWeirdError: unexpected behavior",
            expected_vs_actual="Expected normal behavior",
        )

        analysis = analyzer.analyze_failure(failure)
        assert analysis['matched_patterns'] == []
        assert 'Unknown error pattern' in analysis['likely_cause']
        assert len(analysis['hypotheses']) > 0
        assert len(analysis['investigation_steps']) > 0

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case-insensitive."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="importerror: cannot import module",
            expected_vs_actual="Expected import to work",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'import_error' in analysis['matched_patterns']


class TestAnalysisStructure:
    """Tests for analyze_failure return structure."""

    def test_analysis_has_required_keys(self):
        """Test that analysis result has all required keys."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: test failed",
            expected_vs_actual="Expected pass",
        )

        analysis = analyzer.analyze_failure(failure)
        assert 'likely_cause' in analysis
        assert 'matched_patterns' in analysis
        assert 'hypotheses' in analysis
        assert 'suggested_fix' in analysis
        assert 'investigation_steps' in analysis
        assert 'similar_failures' in analysis

    def test_hypotheses_have_likelihood(self):
        """Test that hypotheses include likelihood markers."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: test failed",
            expected_vs_actual="Expected pass",
        )

        analysis = analyzer.analyze_failure(failure)
        assert len(analysis['hypotheses']) > 0
        # At least one hypothesis should have a likelihood marker
        assert any('[high]' in h or '[medium]' in h or '[low]' in h for h in analysis['hypotheses'])

    def test_investigation_steps_are_list(self):
        """Test that investigation steps are a list of strings."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="TypeError: wrong type",
            expected_vs_actual="Expected correct type",
        )

        analysis = analyzer.analyze_failure(failure)
        assert isinstance(analysis['investigation_steps'], list)
        assert len(analysis['investigation_steps']) > 0
        assert all(isinstance(step, str) for step in analysis['investigation_steps'])

    def test_multiple_pattern_match(self):
        """Test behavior when multiple patterns match."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        # Error message with multiple pattern indicators
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: TimeoutError occurred",
            expected_vs_actual="Expected success",
        )

        analysis = analyzer.analyze_failure(failure)
        # Should match both assertion_error and timeout
        assert len(analysis['matched_patterns']) >= 2
        # Should have hypothesis about cascading failure
        assert any('cascading' in h.lower() for h in analysis['hypotheses'])


class TestSimilarFailures:
    """Tests for find_similar_failures functionality."""

    def test_empty_history(self):
        """Test finding similar failures with empty history."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Error",
            expected_vs_actual="Expected pass",
        )

        similar = analyzer.find_similar_failures(failure)
        assert similar == []

    def test_identical_check_name(self):
        """Test similarity scoring for identical check names."""
        analyzer = FailureAnalyzer()
        check1 = VerificationCheck(
            name="test_foo",
            description="Test foo",
            level=VerificationLevel.UNIT,
        )
        failure1 = VerificationFailure(
            check=check1,
            observed="Error A",
            expected_vs_actual="Expected pass",
        )
        analyzer.record_failure(failure1)

        check2 = VerificationCheck(
            name="test_foo",  # Same name
            description="Different description",
            level=VerificationLevel.UNIT,
        )
        failure2 = VerificationFailure(
            check=check2,
            observed="Error B",
            expected_vs_actual="Expected pass",
        )

        similar = analyzer.find_similar_failures(failure2)
        assert len(similar) > 0
        # Should have high similarity due to name match
        assert similar[0][1] >= 0.4  # At least 40% from name match

    def test_identical_error_message(self):
        """Test similarity scoring for identical error messages."""
        analyzer = FailureAnalyzer()
        check1 = VerificationCheck(
            name="test_a",
            description="Test A",
            level=VerificationLevel.UNIT,
        )
        failure1 = VerificationFailure(
            check=check1,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected pass",
        )
        analyzer.record_failure(failure1)

        check2 = VerificationCheck(
            name="test_b",
            description="Test B",
            level=VerificationLevel.UNIT,
        )
        failure2 = VerificationFailure(
            check=check2,
            observed="ImportError: No module named 'foo'",  # Same error
            expected_vs_actual="Expected pass",
        )

        similar = analyzer.find_similar_failures(failure2)
        assert len(similar) > 0
        # Should have similarity from error match
        assert similar[0][1] >= 0.3  # At least 30% from error match

    def test_sorted_by_similarity(self):
        """Test that similar failures are sorted by similarity score."""
        analyzer = FailureAnalyzer()

        # Record multiple failures
        for i in range(3):
            check = VerificationCheck(
                name=f"test_{i}",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed=f"Error {i}",
                expected_vs_actual="Expected pass",
            )
            analyzer.record_failure(failure)

        # Query with a failure similar to the first one
        check = VerificationCheck(
            name="test_0",  # Matches first one
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Error X",
            expected_vs_actual="Expected pass",
        )

        similar = analyzer.find_similar_failures(failure)
        # Should be sorted by similarity (descending)
        if len(similar) > 1:
            for i in range(len(similar) - 1):
                assert similar[i][1] >= similar[i + 1][1]

    def test_top_n_limit(self):
        """Test that top_n limits the number of results."""
        analyzer = FailureAnalyzer()

        # Record 10 failures
        for i in range(10):
            check = VerificationCheck(
                name="test",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed=f"Error {i}",
                expected_vs_actual="Expected pass",
            )
            analyzer.record_failure(failure)

        # Query with top_n=3
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="Error X",
            expected_vs_actual="Expected pass",
        )

        similar = analyzer.find_similar_failures(failure, top_n=3)
        assert len(similar) <= 3


class TestFailureRecording:
    """Tests for failure recording and history management."""

    def test_record_failure(self):
        """Test recording a failure."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: test failed",
            expected_vs_actual="Expected pass",
        )

        analyzer.record_failure(failure)
        assert len(analyzer._history) == 1
        assert analyzer._history[0] is failure

    def test_multiple_recordings(self):
        """Test recording multiple failures."""
        analyzer = FailureAnalyzer()
        failures = []
        for i in range(5):
            check = VerificationCheck(
                name=f"test_{i}",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed=f"Error {i}",
                expected_vs_actual="Expected pass",
            )
            failures.append(failure)
            analyzer.record_failure(failure)

        assert len(analyzer._history) == 5

    def test_pattern_stats_updated(self):
        """Test that pattern statistics are updated on recording."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work",
        )

        # Record the failure
        analyzer.record_failure(failure)

        # Check that import_error stats were updated
        stats = analyzer.get_pattern_statistics()
        assert stats['import_error']['total_occurrences'] == 1


class TestFixSuccessRecording:
    """Tests for recording fix success."""

    def test_record_successful_fix(self):
        """Test recording a successful fix."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work",
        )

        analyzer.record_failure(failure)
        analyzer.record_fix_success(failure, successful=True)

        stats = analyzer.get_pattern_statistics()
        assert stats['import_error']['successful_fixes'] == 1

    def test_record_unsuccessful_fix(self):
        """Test recording an unsuccessful fix."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work",
        )

        analyzer.record_failure(failure)
        analyzer.record_fix_success(failure, successful=False)

        stats = analyzer.get_pattern_statistics()
        assert stats['import_error']['successful_fixes'] == 0  # Not incremented


class TestInvestigationPlan:
    """Tests for investigation plan generation."""

    def test_generate_plan_for_import_error(self):
        """Test generating investigation plan for import error."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'foo'",
            expected_vs_actual="Expected import to work",
        )

        plan = analyzer.generate_investigation_plan(failure)
        assert isinstance(plan, list)
        assert len(plan) > 0
        # Should include steps specific to import errors
        assert any('package' in step.lower() or 'import' in step.lower() for step in plan)

    def test_generate_plan_for_assertion_error(self):
        """Test generating investigation plan for assertion error."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="AssertionError: expected 5 but got 3",
            expected_vs_actual="Expected 5, got 3",
        )

        plan = analyzer.generate_investigation_plan(failure)
        assert isinstance(plan, list)
        assert len(plan) > 0
        # Should include steps about checking assertions
        assert any('assertion' in step.lower() or 'expected' in step.lower() for step in plan)

    def test_generate_plan_for_unknown_error(self):
        """Test generating investigation plan for unknown error."""
        analyzer = FailureAnalyzer()
        check = VerificationCheck(
            name="test",
            description="Test",
            level=VerificationLevel.UNIT,
        )
        failure = VerificationFailure(
            check=check,
            observed="SomeWeirdError: unexpected",
            expected_vs_actual="Expected normal behavior",
        )

        plan = analyzer.generate_investigation_plan(failure)
        assert isinstance(plan, list)
        assert len(plan) > 0
        # Should include generic investigation steps
        assert any('error message' in step.lower() or 'debug' in step.lower() for step in plan)


class TestPatternStatistics:
    """Tests for pattern statistics tracking."""

    def test_get_statistics_structure(self):
        """Test that statistics have correct structure."""
        analyzer = FailureAnalyzer()
        stats = analyzer.get_pattern_statistics()

        assert isinstance(stats, dict)
        for pattern_name in analyzer._patterns:
            assert pattern_name in stats
            assert 'total_occurrences' in stats[pattern_name]
            assert 'successful_fixes' in stats[pattern_name]
            assert 'success_rate' in stats[pattern_name]

    def test_initial_statistics_zero(self):
        """Test that initial statistics are zero."""
        analyzer = FailureAnalyzer()
        stats = analyzer.get_pattern_statistics()

        for pattern_name, pattern_stats in stats.items():
            assert pattern_stats['total_occurrences'] == 0
            assert pattern_stats['successful_fixes'] == 0
            assert pattern_stats['success_rate'] == 0.0

    def test_statistics_updated_on_recording(self):
        """Test that statistics are updated when failures are recorded."""
        analyzer = FailureAnalyzer()

        # Record 3 import errors
        for i in range(3):
            check = VerificationCheck(
                name=f"test_{i}",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed="ImportError: No module named 'foo'",
                expected_vs_actual="Expected import to work",
            )
            analyzer.record_failure(failure)

        stats = analyzer.get_pattern_statistics()
        assert stats['import_error']['total_occurrences'] == 3

    def test_success_rate_calculation(self):
        """Test that success rate is calculated correctly."""
        analyzer = FailureAnalyzer()

        # Record 5 failures and 3 successful fixes
        failures = []
        for i in range(5):
            check = VerificationCheck(
                name=f"test_{i}",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed="AssertionError: test failed",
                expected_vs_actual="Expected pass",
            )
            failures.append(failure)
            analyzer.record_failure(failure)

        # Record 3 successful fixes
        for i in range(3):
            analyzer.record_fix_success(failures[i], successful=True)

        stats = analyzer.get_pattern_statistics()
        # 3 successful out of 5 = 60%
        assert stats['assertion_error']['success_rate'] == 60.0


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow: record, analyze, find similar, plan."""
        analyzer = FailureAnalyzer()

        # Record a historical failure
        check1 = VerificationCheck(
            name="test_import",
            description="Test import functionality",
            level=VerificationLevel.UNIT,
        )
        failure1 = VerificationFailure(
            check=check1,
            observed="ImportError: No module named 'missing_pkg'",
            expected_vs_actual="Expected import to work",
        )
        analyzer.record_failure(failure1)
        analyzer.record_fix_success(failure1, successful=True)

        # Analyze a new similar failure
        check2 = VerificationCheck(
            name="test_import",
            description="Test import functionality",
            level=VerificationLevel.UNIT,
        )
        failure2 = VerificationFailure(
            check=check2,
            observed="ImportError: No module named 'another_pkg'",
            expected_vs_actual="Expected import to work",
        )

        # Analyze
        analysis = analyzer.analyze_failure(failure2)
        assert 'import_error' in analysis['matched_patterns']

        # Find similar
        similar = analyzer.find_similar_failures(failure2)
        assert len(similar) > 0
        assert similar[0][0] is failure1

        # Generate plan
        plan = analyzer.generate_investigation_plan(failure2)
        assert len(plan) > 0

        # Check stats
        stats = analyzer.get_pattern_statistics()
        assert stats['import_error']['total_occurrences'] == 1
        assert stats['import_error']['successful_fixes'] == 1
        assert stats['import_error']['success_rate'] == 100.0

    def test_multiple_error_types(self):
        """Test handling multiple different error types."""
        analyzer = FailureAnalyzer()

        # Record different error types
        error_types = [
            ("ImportError: missing", 'import_error'),
            ("AssertionError: failed", 'assertion_error'),
            ("TimeoutError: too slow", 'timeout'),
            ("ConnectionError: refused", 'connection_error'),
            ("TypeError: wrong args", 'type_error'),
        ]

        for error_msg, expected_pattern in error_types:
            check = VerificationCheck(
                name="test",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed=error_msg,
                expected_vs_actual="Expected success",
            )
            analyzer.record_failure(failure)

            analysis = analyzer.analyze_failure(failure)
            assert expected_pattern in analysis['matched_patterns']

        # Check overall stats
        stats = analyzer.get_pattern_statistics()
        assert stats['import_error']['total_occurrences'] == 1
        assert stats['assertion_error']['total_occurrences'] == 1
        assert stats['timeout']['total_occurrences'] == 1
        assert stats['connection_error']['total_occurrences'] == 1
        assert stats['type_error']['total_occurrences'] == 1


class TestBehavioral:
    """Behavioral tests ensuring investigation plans are actionable."""

    def test_plans_are_actionable(self):
        """Test that investigation plans contain actionable steps."""
        analyzer = FailureAnalyzer()

        # Test for each pattern type
        test_cases = [
            ("ImportError: missing", "import"),
            ("AssertionError: failed", "assertion"),
            ("TimeoutError: slow", "profile"),
            ("TypeError: wrong", "signature"),
        ]

        for error_msg, expected_keyword in test_cases:
            check = VerificationCheck(
                name="test",
                description="Test",
                level=VerificationLevel.UNIT,
            )
            failure = VerificationFailure(
                check=check,
                observed=error_msg,
                expected_vs_actual="Expected success",
            )

            plan = analyzer.generate_investigation_plan(failure)
            # Plans should be non-empty and contain relevant keywords
            assert len(plan) > 0
            plan_text = ' '.join(plan).lower()
            # At least one step should mention the error type
            assert len(plan) >= 2  # Should have multiple steps

    def test_similar_failures_relevance(self):
        """Test that similar failures are actually relevant."""
        analyzer = FailureAnalyzer()

        # Record failures with clear similarity
        check1 = VerificationCheck(
            name="test_database_connection",
            description="Test DB connection",
            level=VerificationLevel.INTEGRATION,
        )
        failure1 = VerificationFailure(
            check=check1,
            observed="ConnectionError: Connection refused to localhost:5432",
            expected_vs_actual="Expected connection",
        )
        analyzer.record_failure(failure1)

        # Query with similar failure
        check2 = VerificationCheck(
            name="test_database_connection",
            description="Test DB connection",
            level=VerificationLevel.INTEGRATION,
        )
        failure2 = VerificationFailure(
            check=check2,
            observed="ConnectionError: Connection refused to localhost:5433",
            expected_vs_actual="Expected connection",
        )

        similar = analyzer.find_similar_failures(failure2)
        assert len(similar) > 0
        # Should have high similarity
        assert similar[0][1] > 0.5  # More than 50% similar
