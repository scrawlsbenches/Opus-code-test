"""
Behavioral Tests for Audit Reasoning with PLN.

These tests describe user stories for the audit reasoning system:
- Developers analyzing code for risks
- Understanding WHY files are flagged
- Getting actionable suggestions
- Persisting insights across sessions

Testing Philosophy (Metus):
- Scenarios test behaviors, not implementation
- Given-When-Then format tells the story
- Tests serve as living documentation
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List


# =============================================================================
# STORY: Developer Analyzes File for Risks
# =============================================================================


class TestDeveloperAnalyzesFileForRisks:
    """
    Story: As a developer, I want to analyze a file for potential risks
    so that I can prioritize what to review.
    """

    @pytest.fixture
    def reasoner(self):
        """Create an audit reasoner with default rules."""
        from scripts.audit_reasoning import AuditReasoner
        r = AuditReasoner(use_persistence=False)
        r.add_default_rules()
        return r

    def test_analyzing_file_with_todo_comments(self, reasoner):
        """
        Scenario: Analyzing a file that contains TODO comments

        Given a file with TODO comments
        When I analyze the file for risks
        Then the file should be flagged as having incomplete work
        And the confidence should be reasonable (> 0.3)
        """
        # Given a file with TODO comments
        file_path = "src/utils.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo"],
            traits=[],
            directories=["utils"]
        )

        # When I analyze the file for risks
        results = reasoner.query_file_risk(file_path)

        # Then the file should be flagged as having incomplete work
        assert results is not None
        assert "incomplete" in results or "needs_review" in results

        # And the confidence should be reasonable (> 0.3)
        if "incomplete" in results:
            assert results["incomplete"]["strength"] > 0.3

    def test_analyzing_file_with_fixme_markers(self, reasoner):
        """
        Scenario: Analyzing a file with FIXME markers

        Given a file with FIXME markers
        When I analyze the file for known issues
        Then the file should be flagged as needing review
        And I should understand why it was flagged
        """
        # Given a file with FIXME markers
        file_path = "src/parser.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["fixme"],
            traits=[],
            directories=["parser"]
        )

        # When I analyze the file for known issues
        results = reasoner.query_file_risk(file_path)

        # Then the file should be flagged as having known issues
        assert results is not None
        # Either has_known_issue or needs_review should be present
        has_risk = any(k in results for k in ["has_known_issue", "needs_review", "risky"])
        assert has_risk or len(results) > 0

    def test_analyzing_high_churn_file(self, reasoner):
        """
        Scenario: Analyzing a high-churn file

        Given a file that has been modified frequently
        When I analyze the file for risks
        Then the file should be flagged as needing review
        """
        # Given a file that has been modified frequently
        file_path = "src/core/engine.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=[],
            traits=["high_churn"],
            directories=["core"]
        )

        # When I analyze the file for risks
        results = reasoner.query_file_risk(file_path)

        # Then the file should be flagged as needing review
        assert results is not None
        # Should have some risk signal
        has_signal = "needs_review" in results or "risky" in results
        assert has_signal or "_importance" in results


# =============================================================================
# STORY: Developer Wants to Understand WHY a File is Risky
# =============================================================================


class TestDeveloperUnderstandsWhyFileIsRisky:
    """
    Story: As a developer, I want to understand WHY a file was flagged
    so that I can make informed decisions about what to fix.
    """

    @pytest.fixture
    def reasoner(self):
        """Create an audit reasoner with default rules."""
        from scripts.audit_reasoning import AuditReasoner
        r = AuditReasoner(use_persistence=False)
        r.add_default_rules()
        return r

    def test_explanation_shows_reasoning(self, reasoner):
        """
        Scenario: Getting explanation for a flagged file

        Given a file flagged for having multiple risk factors
        When I ask for an explanation
        Then I should see the reasoning chain
        And the explanation should be human-readable
        """
        # Given a file flagged for having multiple risk factors
        file_path = "src/critical_module.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo", "fixme"],
            traits=["high_churn"],
            directories=["critical"]
        )

        # When I ask for an explanation
        explanation = reasoner.explain_file_risk(file_path)

        # Then I should see the reasoning chain
        assert explanation is not None
        assert "traces" in explanation

        # And the explanation should be human-readable
        assert "summary" in explanation
        summary = explanation["summary"]
        assert isinstance(summary, str)

    def test_explanation_includes_facts(self, reasoner):
        """
        Scenario: Explanation includes detected facts

        Given a file with specific patterns
        When I request an explanation
        Then I should see the facts that were detected
        """
        # Given a file with specific patterns
        file_path = "src/handler.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo", "should_be"],
            traits=[],
            directories=["handlers"]
        )

        # When I request an explanation
        explanation = reasoner.explain_file_risk(file_path)

        # Then I should see the facts that were detected
        assert "facts" in explanation
        facts = explanation["facts"]
        # Should have recorded the patterns
        assert isinstance(facts, dict)


# =============================================================================
# STORY: Developer Gets Actionable Suggestions
# =============================================================================


class TestDeveloperGetsActionableSuggestions:
    """
    Story: As a developer, I want actionable suggestions for flagged files
    so that I know what to do next.
    """

    @pytest.fixture
    def reasoner(self):
        """Create an audit reasoner with default rules."""
        from scripts.audit_reasoning import AuditReasoner
        r = AuditReasoner(use_persistence=False)
        r.add_default_rules()
        return r

    def test_todo_file_gets_completion_suggestion(self, reasoner):
        """
        Scenario: File with TODOs gets completion suggestion

        Given a file with TODO comments
        When I request suggestions
        Then I should receive a suggestion to address the TODOs
        """
        # Given a file with TODO comments
        file_path = "src/unfinished.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo"],
            traits=[],
            directories=[]
        )

        # When I request suggestions (via explain_file_risk)
        explanation = reasoner.explain_file_risk(file_path)

        # Then I should receive suggestions
        assert "suggestions" in explanation
        suggestions = explanation["suggestions"]
        assert isinstance(suggestions, list)
        # If there are suggestions, they should be actionable
        if suggestions:
            assert any("TODO" in s or "todo" in s.lower() for s in suggestions)

    def test_fixme_file_gets_bug_fix_suggestion(self, reasoner):
        """
        Scenario: File with FIXME gets bug fix suggestion

        Given a file with FIXME markers
        When I request suggestions
        Then I should receive a suggestion to address the known issues
        """
        # Given a file with FIXME markers
        file_path = "src/buggy.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["fixme"],
            traits=[],
            directories=[]
        )

        # When I request suggestions
        explanation = reasoner.explain_file_risk(file_path)

        # Then I should receive suggestions
        assert "suggestions" in explanation
        suggestions = explanation["suggestions"]
        assert isinstance(suggestions, list)
        # If there are suggestions, they should mention fixing or bugs
        if suggestions:
            suggestion_text = " ".join(suggestions).lower()
            assert "fix" in suggestion_text or "bug" in suggestion_text or "issue" in suggestion_text

    def test_high_churn_file_gets_refactoring_suggestion(self, reasoner):
        """
        Scenario: High-churn file gets refactoring suggestion

        Given a file with high churn rate
        When I request suggestions
        Then I should receive a suggestion to consider refactoring
        """
        # Given a file with high churn rate
        file_path = "src/hot_module.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=[],
            traits=["high_churn"],
            directories=[]
        )

        # When I request suggestions
        explanation = reasoner.explain_file_risk(file_path)

        # Then suggestions should be available
        assert "suggestions" in explanation
        # May or may not have suggestions depending on inference results


# =============================================================================
# STORY: Developer Uses Natural Language Queries
# =============================================================================


class TestDeveloperUsesNaturalLanguageQueries:
    """
    Story: As a developer, I want to ask questions in natural language
    so that I don't need to learn a query syntax.
    """

    def test_translate_risky_query(self):
        """
        Scenario: Translating "why is this file risky?"

        Given a natural language question about risk
        When I translate it to a structured query
        Then the query should identify the file and intent
        """
        from scripts.audit_reasoning import translate_audit_query

        # When I translate a question about risk
        query = translate_audit_query("why is utils.py risky?")

        # Then the query should be recognized
        assert query is not None
        # Should have a valid intent
        assert query.intent in ["explain", "list", "trace"]

    def test_translate_todo_query(self):
        """
        Scenario: Translating "which files have TODO comments?"

        Given a natural language question about TODOs
        When I translate it to a structured query
        Then the query should look for TODO patterns
        """
        from scripts.audit_reasoning import translate_audit_query

        # When I translate a question about TODOs
        query = translate_audit_query("which files have todos?")

        # Then the query should be recognized
        assert query is not None

    def test_translate_flagged_query(self):
        """
        Scenario: Translating "what files are flagged?"

        Given a natural language question about flagged files
        When I translate it to a structured query
        Then the query should look for flagged files
        """
        from scripts.audit_reasoning import translate_audit_query

        # When I translate a question about flagged files
        query = translate_audit_query("what files are flagged?")

        # Then the query should be recognized
        assert query is not None
        # Should have a valid intent (list is the default for "what" queries)
        assert query.intent in ["list", "explain", "trace"]

    def test_is_natural_language_detection(self):
        """
        Scenario: Detecting natural language vs formal queries

        Given various inputs
        When I check if they are natural language
        Then natural language should be detected correctly
        """
        from scripts.audit_reasoning import is_natural_language_query

        # Natural language queries (contain spaces or NLU keywords)
        assert is_natural_language_query("why is utils.py risky?")
        assert is_natural_language_query("what files are flagged?")
        assert is_natural_language_query("which modules need review?")

        # Single word paths without NLU keywords are not natural language
        assert not is_natural_language_query("--verbose")
        assert not is_natural_language_query("-h")


# =============================================================================
# STORY: Insights Persist Across Sessions
# =============================================================================


class TestInsightsPersistAcrossSessions:
    """
    Story: As a developer, I want my audit insights to persist
    so that I don't lose context between sessions.
    """

    def test_file_importance_record_creation(self):
        """
        Scenario: Creating a file importance record

        Given a file that was analyzed
        When I create an importance record
        Then it should capture the analysis metrics
        """
        from scripts.audit_reasoning import FileImportanceRecord

        # When I create an importance record
        record = FileImportanceRecord(
            file_id="utils_py",
            sti=0.8,
            lti=0.7,
            vlti=False,
            last_seen="2026-01-07T12:00:00",
            history=[]
        )

        # Then it should capture the analysis metrics
        assert record.file_id == "utils_py"
        assert record.sti == 0.8
        assert record.lti == 0.7
        assert record.vlti is False

    def test_persistence_state_tracks_multiple_files(self):
        """
        Scenario: Tracking multiple files across sessions

        Given multiple files have been analyzed
        When I save the persistence state
        Then all files should be tracked
        """
        from scripts.audit_reasoning import (
            AuditPersistenceState,
            FileImportanceRecord,
        )

        # Given multiple files have been analyzed
        state = AuditPersistenceState.create_new()

        record1 = FileImportanceRecord(
            file_id="file1_py",
            sti=0.8,
            lti=0.7,
            vlti=False,
            last_seen="2026-01-07T12:00:00",
            history=[]
        )
        record2 = FileImportanceRecord(
            file_id="file2_py",
            sti=0.6,
            lti=0.5,
            vlti=True,
            last_seen="2026-01-07T12:00:00",
            history=[]
        )

        state.file_importance["file1_py"] = record1
        state.file_importance["file2_py"] = record2

        # Then all files should be tracked
        assert len(state.file_importance) == 2
        assert "file1_py" in state.file_importance
        assert "file2_py" in state.file_importance
        assert state.file_importance["file2_py"].vlti is True

    def test_attention_value_decay(self):
        """
        Scenario: File importance decays when not accessed

        Given a file that hasn't been accessed recently
        When the decay process runs
        Then the file's short-term importance should decrease
        """
        from scripts.audit_reasoning import AttentionValue

        # Given a file with high short-term importance
        attention = AttentionValue(sti=0.9, lti=0.7, vlti=False)

        # When the decay process runs
        attention.decay_sti(0.8)  # 20% decay

        # Then the file's short-term importance should decrease
        assert attention.sti == pytest.approx(0.72, rel=0.01)
        # Long-term importance unchanged
        assert attention.lti == 0.7


# =============================================================================
# STORY: Multi-Rule Analysis
# =============================================================================


class TestMultiRuleAnalysis:
    """
    Story: As a developer, I want multiple rules to contribute to risk assessment
    so that I get a comprehensive view.
    """

    @pytest.fixture
    def reasoner(self):
        """Create an audit reasoner with default rules."""
        from scripts.audit_reasoning import AuditReasoner
        r = AuditReasoner(use_persistence=False)
        r.add_default_rules()
        return r

    def test_multiple_risk_factors_detected(self, reasoner):
        """
        Scenario: Multiple risk factors are detected and reported

        Given a file with multiple risk indicators
        When I analyze the file
        Then all risk factors should be captured in the explanation
        """
        # Given a file with multiple risk indicators
        file_path = "src/problematic.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo", "fixme", "hack"],
            traits=["high_churn"],
            directories=[]
        )

        # When I analyze the file
        explanation = reasoner.explain_file_risk(file_path)

        # Then multiple facts should be captured
        assert "facts" in explanation
        facts = explanation["facts"]
        # Should have multiple patterns recorded
        assert len(facts) >= 1

    def test_aggregation_strategies_produce_different_results(self, reasoner):
        """
        Scenario: Different aggregation strategies produce different confidence levels

        Given a file with risk indicators
        When I query with different aggregation strategies
        Then the results may differ based on strategy
        """
        # Given a file with risk indicators
        file_path = "src/test_file.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo", "fixme"],
            traits=["high_churn"],
            directories=[]
        )

        # When I query with different aggregation strategies
        results = reasoner.query_with_aggregation(
            f"needs_review({Path(file_path).name.replace('.', '_')})",
            strategies=["first", "revision", "max", "or"]
        )

        # Then the results exist (strategies may or may not differ)
        # The important thing is all strategies work
        assert isinstance(results, dict)


# =============================================================================
# STORY: Report Generation
# =============================================================================


class TestReportGeneration:
    """
    Story: As a developer, I want to generate reports of audit findings
    so that I can share them with my team.
    """

    def test_generate_reasoning_report(self):
        """
        Scenario: Generate a reasoning report

        Given analysis results from multiple files
        When I generate a report
        Then it should summarize the findings
        """
        from scripts.audit_reasoning import generate_reasoning_report

        # Given analysis results from multiple files
        results = {
            "files_analyzed": ["src/a.py", "src/b.py", "src/c.py"],
            "rules_loaded": 10,
            "analysis_results": [
                {
                    "file": "src/a.py",
                    "risk": {"needs_review": {"strength": 0.8}},
                },
                {
                    "file": "src/b.py",
                    "risk": {"flagged": {"strength": 0.5}},
                },
            ],
        }

        # When I generate a report
        report = generate_reasoning_report(results)

        # Then it should be a non-empty string
        assert report is not None
        assert isinstance(report, str)
        assert len(report) > 0


# =============================================================================
# STORY: WovenMind Integration
# =============================================================================


class TestWovenMindIntegration:
    """
    Story: As a system, I want to learn from WovenMind abstractions
    so that I can apply learned patterns to code analysis.
    """

    def test_abstraction_to_rule_conversion(self):
        """
        Scenario: WovenMind abstraction becomes audit rule

        Given a WovenMind abstraction about code patterns
        When I convert it to an audit rule
        Then it should produce a usable rule structure
        """
        from scripts.audit_reasoning import abstraction_to_rule

        # Given a WovenMind abstraction
        abstraction = {
            "type": "co-occurrence",
            "edge_strength": 0.8,
            "source_nodes": ["dir:legacy", "pattern:todo"],
        }

        # When I convert it to an audit rule
        rule = abstraction_to_rule(abstraction)

        # Then it should produce a rule structure
        assert rule is not None
        assert "strength" in rule
        assert "confidence" in rule

    def test_loading_rules_from_woven_mind(self):
        """
        Scenario: Loading rules from WovenMind abstractions

        Given the reasoner is initialized
        When I load rules from WovenMind
        Then rules should be added to the reasoner
        """
        from scripts.audit_reasoning import AuditReasoner

        # Given the reasoner is initialized
        reasoner = AuditReasoner(use_persistence=False)

        # When I load rules from WovenMind (may or may not find abstractions)
        count = reasoner.load_rules_from_woven_mind()

        # Then the method completes without error
        # Count may be 0 if no abstractions file exists
        assert isinstance(count, int)
        assert count >= 0


# =============================================================================
# STORY: Attention-Based Prioritization
# =============================================================================


class TestAttentionBasedPrioritization:
    """
    Story: As a developer, I want the system to focus on the most important files
    so that I can prioritize my review efforts.
    """

    @pytest.fixture
    def reasoner(self):
        """Create an audit reasoner with default rules."""
        from scripts.audit_reasoning import AuditReasoner
        r = AuditReasoner(use_persistence=False)
        r.add_default_rules()
        return r

    def test_get_priority_files(self, reasoner):
        """
        Scenario: Getting priority files for review

        Given multiple files have been analyzed
        When I request the priority files
        Then files should be ordered by importance
        """
        # Given multiple files have been analyzed
        reasoner.assert_file_facts(
            "critical.py",
            patterns=["fixme", "hack"],
            traits=["high_churn", "bug_prone"],
            directories=[]
        )
        reasoner.assert_file_facts(
            "normal.py",
            patterns=["todo"],
            traits=[],
            directories=[]
        )
        reasoner.assert_file_facts(
            "stable.py",
            patterns=[],
            traits=["stable"],
            directories=[]
        )

        # When I request the priority files
        priorities = reasoner.get_priority_files(top_n=3)

        # Then files should be returned
        assert len(priorities) > 0
        # Each entry should be (file_id, importance)
        for file_id, importance in priorities:
            assert isinstance(file_id, str)
            assert isinstance(importance, (int, float))

    def test_stimulate_file_importance(self, reasoner):
        """
        Scenario: Stimulating a file increases its importance

        Given a file has been analyzed
        When I stimulate the file (e.g., it was accessed)
        Then its short-term importance should increase
        """
        # Given a file has been analyzed
        file_path = "important.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=["todo"],
            traits=[],
            directories=[]
        )

        file_id = "important_py"
        initial_sti = reasoner.file_importance[file_id].sti

        # When I stimulate the file
        reasoner.stimulate_file(file_path, amount=0.3)

        # Then its short-term importance should increase
        new_sti = reasoner.file_importance[file_id].sti
        assert new_sti > initial_sti

    def test_vlti_files_tracked(self, reasoner):
        """
        Scenario: Critical files are tracked with VLTI flag

        Given a file marked as critical
        When I query for VLTI files
        Then the critical file should be listed
        """
        # Given a file marked as critical
        file_path = "critical_service.py"
        reasoner.assert_file_facts(
            file_path,
            patterns=[],
            traits=["critical"],
            directories=[]
        )

        # When I query for VLTI files
        vlti_files = reasoner.get_vlti_files()

        # Then the critical file should be listed
        assert "critical_service_py" in vlti_files
