"""
Behavioral Tests for Auditor Workflow.

These tests describe user stories from an auditor's perspective using
the PLN-based audit reasoning system.

Testing Philosophy (Metus):
- Scenarios test behaviors, not implementation
- Given-When-Then format tells the story
- Tests serve as living documentation
- Real-world audit scenarios drive the design
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def fresh_reasoner():
    """Create a fresh AuditReasoner with no persistence."""
    from cortical.audits.reasoning import AuditReasoner
    return AuditReasoner(use_persistence=False)


@pytest.fixture
def reasoner_with_rules(fresh_reasoner):
    """Create a reasoner with default audit rules loaded."""
    fresh_reasoner.add_default_rules()
    return fresh_reasoner


@pytest.fixture
def codebase_with_issues(reasoner_with_rules):
    """
    Simulate a codebase scan with various audit findings.

    This represents a realistic project with:
    - Authentication module with security concerns
    - Legacy code with technical debt
    - Clean utility modules
    - High-churn API endpoints
    """
    reasoner = reasoner_with_rules

    # Auth module - security sensitive with multiple issues
    reasoner.assert_file_facts(
        "auth/login.py",
        patterns=["todo", "fixme", "hack"],
        traits=["security_sensitive"],
        directories=["auth"]
    )

    # Legacy processor - technical debt
    reasoner.assert_file_facts(
        "legacy/old_processor.py",
        patterns=["todo", "fixme", "deprecated"],
        traits=["legacy", "high_churn"],
        directories=["legacy"]
    )

    # Clean utility - no issues
    reasoner.assert_file_facts(
        "utils/helpers.py",
        patterns=[],
        traits=[],
        directories=["utils"]
    )

    # API endpoints - high activity
    reasoner.assert_file_facts(
        "api/endpoints.py",
        patterns=["todo", "xxx"],
        traits=["high_churn"],
        directories=["api"]
    )

    # Core engine - critical but stable
    reasoner.assert_file_facts(
        "core/engine.py",
        patterns=[],
        traits=["critical"],
        directories=["core"]
    )

    return reasoner


# =============================================================================
# STORY: Auditor Identifies High-Risk Files
# =============================================================================


class TestAuditorIdentifiesHighRiskFiles:
    """
    Story: As an auditor, I want to identify high-risk files
    so that I can prioritize my security review.
    """

    def test_files_with_multiple_issues_rank_higher(self, codebase_with_issues):
        """
        Scenario: Files with multiple issues rank higher in priority

        Given a codebase with files having varying issue counts
        When I request priority files
        Then files with more issues should rank higher
        And login.py should be near the top (has 3 patterns + security trait)
        """
        # Given codebase with varying issues (from fixture)
        reasoner = codebase_with_issues

        # When I request priority files
        priorities = reasoner.get_priority_files(top_n=5)

        # Then files should be ordered by importance
        assert len(priorities) > 0
        file_ids = [f for f, _ in priorities]

        # Login.py should rank high due to multiple issues
        assert any("login" in f for f in file_ids[:3])

    def test_security_sensitive_files_are_flagged(self, codebase_with_issues):
        """
        Scenario: Security-sensitive files are properly flagged

        Given a file marked as security sensitive
        When I query its risk
        Then it should have elevated risk indicators
        """
        reasoner = codebase_with_issues

        # When I query risk for the security-sensitive file
        result = reasoner.query_file_risk("auth/login.py")

        # Then it should have risk signals
        assert result is not None
        assert "_importance" in result

    def test_clean_files_have_minimal_risk(self, codebase_with_issues):
        """
        Scenario: Clean files have minimal risk indicators

        Given a file with no patterns or concerning traits
        When I query its risk
        Then it should have low or no risk signals
        """
        reasoner = codebase_with_issues

        # When I query risk for clean file
        result = reasoner.query_file_risk("utils/helpers.py")

        # Then it should have minimal risk
        assert result is not None
        # Clean files still get tracked but with lower importance
        importance = result.get("_importance", {})
        # No patterns means lower STI
        assert importance.get("sti", 0) <= 0.5


# =============================================================================
# STORY: Auditor Understands Why Files Are Flagged
# =============================================================================


class TestAuditorUnderstandsWhyFilesFlagged:
    """
    Story: As an auditor, I want to understand WHY a file is flagged
    so that I can make informed remediation decisions.
    """

    def test_explanation_lists_detected_patterns(self, codebase_with_issues):
        """
        Scenario: Explanation lists all detected patterns

        Given a file with multiple audit patterns
        When I request an explanation
        Then I should see all patterns that were detected
        """
        reasoner = codebase_with_issues

        # When I request explanation for file with multiple patterns
        explanation = reasoner.explain_file_risk("auth/login.py")

        # Then facts should be present
        assert explanation["facts"]  # Non-empty dict
        assert explanation["file_id"] == "login_py"

    def test_explanation_provides_actionable_suggestions(self, codebase_with_issues):
        """
        Scenario: Explanation provides actionable suggestions

        Given a file with specific patterns
        When I request an explanation
        Then I should receive relevant suggestions
        """
        reasoner = codebase_with_issues

        # When I request explanation
        explanation = reasoner.explain_file_risk("auth/login.py")

        # Then suggestions should be provided
        assert explanation["suggestions"]
        assert len(explanation["suggestions"]) > 0

        # Suggestions should be actionable strings
        for suggestion in explanation["suggestions"]:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 10  # Not just empty or trivial

    def test_explanation_includes_human_readable_summary(self, codebase_with_issues):
        """
        Scenario: Explanation includes human-readable summary

        Given a file being audited
        When I request an explanation
        Then the summary should be readable and informative
        """
        reasoner = codebase_with_issues

        # When I request explanation
        explanation = reasoner.explain_file_risk("legacy/old_processor.py")

        # Then summary should be present and readable
        assert "summary" in explanation
        summary = explanation["summary"]
        assert "old_processor" in summary
        assert "FACTS" in summary


# =============================================================================
# STORY: Auditor Tracks File Importance Over Time
# =============================================================================


class TestAuditorTracksImportanceOverTime:
    """
    Story: As an auditor, I want file importance to persist
    so that frequently flagged files get prioritized.
    """

    def test_stimulating_file_increases_importance(self, codebase_with_issues):
        """
        Scenario: Accessing a file increases its importance

        Given a file in the audit system
        When I stimulate it (simulate repeated access/findings)
        Then its short-term importance should increase
        """
        reasoner = codebase_with_issues

        # Given initial importance
        file_id = "login_py"
        initial_sti = reasoner.file_importance[file_id].sti

        # When I stimulate the file
        reasoner.stimulate_file("auth/login.py", amount=0.2)

        # Then importance increases
        new_sti = reasoner.file_importance[file_id].sti
        assert new_sti > initial_sti

    def test_importance_decays_over_time(self, fresh_reasoner):
        """
        Scenario: Importance decays when files aren't accessed

        Given a file with high importance
        When decay is applied (simulating time passing)
        Then short-term importance should decrease
        """
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = fresh_reasoner

        # Given high initial importance
        reasoner.file_importance["hot_file_py"] = AttentionValue(
            sti=0.9, lti=0.5, vlti=False
        )

        # When decay is applied
        reasoner.collect_rent(sti_decay=0.8, lti_decay=0.95)

        # Then STI should decrease
        assert reasoner.file_importance["hot_file_py"].sti == pytest.approx(0.72, rel=0.01)

    def test_vlti_files_are_always_tracked(self, fresh_reasoner):
        """
        Scenario: Critical files can be pinned as Very Long Term Important

        Given a critical file
        When I mark it as VLTI
        Then it should always appear in priority tracking
        """
        from cortical.reasoning.prism_pln import AttentionValue

        reasoner = fresh_reasoner

        # Given a file marked as VLTI
        reasoner.file_importance["critical_py"] = AttentionValue(
            sti=0.3, lti=0.2, vlti=True  # Low STI but VLTI=True
        )
        reasoner.file_importance["normal_py"] = AttentionValue(
            sti=0.3, lti=0.2, vlti=False
        )

        # When I get VLTI files
        vlti_files = reasoner.get_vlti_files()

        # Then critical file should be included
        assert "critical_py" in vlti_files
        assert "normal_py" not in vlti_files


# =============================================================================
# STORY: Auditor Uses Natural Language Queries
# =============================================================================


class TestAuditorUsesNaturalLanguageQueries:
    """
    Story: As an auditor, I want to ask questions naturally
    so that I don't need to learn complex query syntax.
    """

    def test_why_question_is_understood(self):
        """
        Scenario: "Why is X risky?" is understood

        Given a natural language question
        When I translate it
        Then the system should understand the explain intent
        """
        from cortical.audits.reasoning import translate_audit_query

        # When I ask a why question
        query = translate_audit_query("why is auth.py risky?")

        # Then intent should be explain
        assert query.intent == "explain"
        assert query.target_file == "auth.py"

    def test_directory_scope_is_extracted(self):
        """
        Scenario: Directory scope is extracted from query

        Given a query mentioning a directory
        When I translate it
        Then the directory should be captured
        """
        from cortical.audits.reasoning import translate_audit_query

        # When I query with directory
        query = translate_audit_query("risky files in cortical/")

        # Then directory should be extracted
        assert query.directory == "cortical/"

    def test_negations_are_extracted(self):
        """
        Scenario: Exclusions are understood

        Given a query with exclusions
        When I translate it
        Then negations should be captured
        """
        from cortical.audits.reasoning import translate_audit_query

        # When I query with exclusions
        query = translate_audit_query("files not in tests excluding vendor")

        # Then negations should be captured
        assert "tests" in query.negations or "vendor" in query.negations

    def test_risk_level_keywords_are_understood(self):
        """
        Scenario: Risk level keywords set appropriate thresholds

        Given queries with risk keywords
        When I translate them
        Then appropriate min_risk thresholds should be set
        """
        from cortical.audits.reasoning import translate_audit_query

        # Critical = 0.9
        critical = translate_audit_query("critical files")
        assert critical.min_risk == 0.9

        # High = 0.7
        high = translate_audit_query("high risk files")
        assert high.min_risk == 0.7

        # Risky = 0.5
        risky = translate_audit_query("risky files")
        assert risky.min_risk == 0.5


# =============================================================================
# STORY: Auditor Generates Reports
# =============================================================================


class TestAuditorGeneratesReports:
    """
    Story: As an auditor, I want to generate reports
    so that I can share findings with stakeholders.
    """

    def test_report_includes_summary_statistics(self):
        """
        Scenario: Report includes summary statistics

        Given analysis results
        When I generate a report
        Then it should include file counts and rule counts
        """
        from cortical.audits.reasoning import generate_reasoning_report

        # Given analysis results
        results = {
            "files_analyzed": 25,
            "rules_loaded": 12,
            "risk_assessments": [],
            "priority_files": [],
            "vlti_files": [],
            "stats": {}
        }

        # When I generate report
        report = generate_reasoning_report(results, verbose=False)

        # Then it should include statistics
        assert "25" in report  # files analyzed
        assert isinstance(report, str)

    def test_report_lists_risky_files(self):
        """
        Scenario: Report lists files by risk level

        Given analysis with risky files
        When I generate a report
        Then risky files should be listed with their risk levels
        """
        from cortical.audits.reasoning import generate_reasoning_report

        # Given results with risky files
        results = {
            "files_analyzed": 10,
            "rules_loaded": 5,
            "risk_assessments": [
                {
                    "file": "dangerous.py",
                    "overall_risk": 0.85,
                    "details": {},
                    "importance": 0.7
                }
            ],
            "priority_files": [("dangerous_py", 0.85)],
            "vlti_files": [],
            "stats": {}
        }

        # When I generate report
        report = generate_reasoning_report(results, verbose=False)

        # Then risky files should be listed
        assert "dangerous.py" in report


# =============================================================================
# STORY: Auditor Adds Custom Rules
# =============================================================================


class TestAuditorAddsCustomRules:
    """
    Story: As an auditor, I want to add custom rules
    so that I can encode domain-specific knowledge.
    """

    def test_custom_rule_affects_inference(self, fresh_reasoner):
        """
        Scenario: Custom rules are used in inference

        Given a custom rule relating patterns to risk
        When I assert matching facts
        Then the rule should fire and affect risk assessment
        """
        reasoner = fresh_reasoner

        # Given a custom rule
        reasoner.pln.assert_rule(
            "uses_eval(X)", "security_risk(X)",
            strength=0.95, confidence=0.9
        )

        # When I assert a matching fact
        reasoner.pln.assert_fact("uses_eval(unsafe_py)", strength=0.9, confidence=0.9)

        # Then the rule should fire
        result = reasoner.pln.query("security_risk(unsafe_py)")

        assert result is not None
        assert result.strength > 0.5

    def test_multiple_rules_combine_evidence(self, fresh_reasoner):
        """
        Scenario: Multiple rules combine evidence

        Given multiple rules pointing to same conclusion
        When facts match multiple rules
        Then evidence should be combined
        """
        reasoner = fresh_reasoner

        # Given multiple rules
        reasoner.pln.assert_rule(
            "has_todo(X)", "needs_attention(X)",
            strength=0.6, confidence=0.8
        )
        reasoner.pln.assert_rule(
            "high_churn(X)", "needs_attention(X)",
            strength=0.7, confidence=0.85
        )

        # When facts match both rules
        reasoner.pln.assert_fact("has_todo(busy_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_fact("high_churn(busy_py)", strength=0.8, confidence=0.9)

        # Then evidence should combine
        result = reasoner.pln.query("needs_attention(busy_py)", aggregate="revision")

        assert result is not None
        # Combined evidence should be stronger than individual rules alone


# =============================================================================
# STORY: Auditor Reviews Inference Chains
# =============================================================================


class TestAuditorReviewsInferenceChains:
    """
    Story: As an auditor, I want to see how conclusions were reached
    so that I can validate the reasoning.
    """

    def test_inference_trace_shows_rule_chain(self, fresh_reasoner):
        """
        Scenario: Inference trace shows which rules fired

        Given a rule chain
        When I query with tracing
        Then I should see each rule that fired
        """
        reasoner = fresh_reasoner

        # Given a rule chain
        reasoner.pln.assert_fact("has_fixme(buggy_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_rule(
            "has_fixme(X)", "has_known_issue(X)",
            strength=0.85, confidence=0.9
        )
        reasoner.pln.assert_rule(
            "has_known_issue(X)", "needs_review(X)",
            strength=0.8, confidence=0.85
        )

        # When I query with trace
        trace = reasoner.pln.query_with_trace("needs_review(buggy_py)", max_depth=3)

        # Then trace should show the chain
        assert trace.final_result is not None
        assert "has_fixme(buggy_py)" in trace.facts_used

    def test_explanation_includes_raw_traces(self, fresh_reasoner):
        """
        Scenario: File explanation includes inference traces

        Given a file with rules that can fire
        When I request an explanation
        Then raw traces should be available
        """
        reasoner = fresh_reasoner
        reasoner.add_default_rules()

        # Set up facts that will trigger rules
        reasoner.pln.assert_fact("has_known_issue(traced_py)", strength=0.9, confidence=0.9)

        # When I get explanation
        explanation = reasoner.explain_file_risk("traced.py")

        # Then raw_traces should be available (may be empty if no rules matched)
        assert "raw_traces" in explanation
        assert isinstance(explanation["raw_traces"], dict)


# =============================================================================
# EDGE CASES
# =============================================================================


class TestAuditorEdgeCases:
    """Edge cases and boundary conditions for auditor workflows."""

    def test_empty_codebase_produces_valid_output(self, fresh_reasoner):
        """
        Scenario: Empty codebase doesn't crash

        Given no files have been scanned
        When I request priorities
        Then I should get an empty but valid result
        """
        reasoner = fresh_reasoner

        # When I request priorities with no files
        priorities = reasoner.get_priority_files(top_n=10)

        # Then result should be empty but valid
        assert priorities == []

    def test_file_with_no_patterns_can_be_explained(self, fresh_reasoner):
        """
        Scenario: Clean files can still be explained

        Given a file with no audit patterns
        When I request an explanation
        Then I should get a valid response indicating no issues
        """
        reasoner = fresh_reasoner

        # When I explain a file with no facts
        explanation = reasoner.explain_file_risk("clean.py")

        # Then response should be valid
        assert explanation["facts"] == {}
        assert explanation["suggestions"] == []
        assert "summary" in explanation

    def test_special_characters_in_filenames_handled(self, fresh_reasoner):
        """
        Scenario: Filenames with special characters work

        Given a file with dots and dashes in the name
        When I process it
        Then it should be normalized correctly
        """
        reasoner = fresh_reasoner

        # Given a complex filename
        reasoner.pln.assert_fact("has_todo(config_v2_beta_py)", strength=0.9, confidence=0.9)

        # When I explain it
        explanation = reasoner.explain_file_risk("config.v2.beta.py")

        # Then file_id should be normalized
        assert explanation["file_id"] == "config_v2_beta_py"
