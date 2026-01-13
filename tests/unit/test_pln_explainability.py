"""
Behavioral tests for PLN Explainability (Phase 2).

These tests define the target behavior for real PLN explainability:
1. InferenceTrace - captures complete reasoning chains
2. InferenceStep - individual rule applications
3. query_with_trace() - inference with full traceability
4. explain_file_risk() - file-level risk explanation
5. Suggested actions - actionable recommendations

This is Phase 2 "Real Explainability" - actual rules that fired,
not templated responses.
"""

import pytest
from typing import Dict, Any


# =============================================================================
# INFERENCE TRACE BASICS
# =============================================================================


class TestInferenceStepBasics:
    """Tests for InferenceStep - individual rule applications."""

    def test_inference_step_creation(self):
        """InferenceStep captures a single rule application."""
        from cortical.reasoning.prism_pln import InferenceStep, TruthValue

        step = InferenceStep(
            rule_antecedent="has_todo(X)",
            rule_consequent="needs_review(X)",
            rule_truth_value=TruthValue(0.8, 0.9),
            antecedent_truth_value=TruthValue(0.95, 0.9),
            result_truth_value=TruthValue(0.7, 0.81),
            substitutions={"X": "file_py"},
            depth=0
        )

        assert step.rule_antecedent == "has_todo(X)"
        assert step.rule_consequent == "needs_review(X)"
        assert step.substitutions == {"X": "file_py"}
        assert step.depth == 0

    def test_inference_step_to_dict(self):
        """InferenceStep can be serialized to dict for storage/transmission."""
        from cortical.reasoning.prism_pln import InferenceStep, TruthValue

        step = InferenceStep(
            rule_antecedent="has_fixme(X)",
            rule_consequent="has_known_issue(X)",
            rule_truth_value=TruthValue(0.9, 0.9),
            antecedent_truth_value=TruthValue(0.85, 0.9),
            result_truth_value=TruthValue(0.75, 0.81),
            substitutions={"X": "buggy_file"},
            depth=1
        )

        d = step.to_dict()

        assert "rule" in d
        assert "has_fixme(X)" in d["rule"]
        assert "has_known_issue(X)" in d["rule"]
        assert d["substitutions"] == {"X": "buggy_file"}
        assert d["depth"] == 1
        assert "rule_tv" in d
        assert "antecedent_tv" in d
        assert "result_tv" in d

    def test_inference_step_string_representation(self):
        """InferenceStep has human-readable string output."""
        from cortical.reasoning.prism_pln import InferenceStep, TruthValue

        step = InferenceStep(
            rule_antecedent="high_churn(X)",
            rule_consequent="risky(X)",
            rule_truth_value=TruthValue(0.75, 0.85),
            antecedent_truth_value=TruthValue(0.8, 0.9),
            result_truth_value=TruthValue(0.6, 0.765),
            substitutions={"X": "hot_file"},
            depth=0
        )

        s = str(step)

        assert "high_churn(X)" in s
        assert "risky(X)" in s
        assert "X=hot_file" in s
        assert "75" in s  # Rule strength percentage


class TestInferenceTraceBasics:
    """Tests for InferenceTrace - complete reasoning chains."""

    def test_inference_trace_creation(self):
        """InferenceTrace captures a complete inference chain."""
        from cortical.reasoning.prism_pln import InferenceTrace, TruthValue

        trace = InferenceTrace(
            query="needs_review(my_file)",
            aggregation_strategy="revision"
        )

        assert trace.query == "needs_review(my_file)"
        assert trace.aggregation_strategy == "revision"
        assert trace.steps == []
        assert trace.facts_used == {}
        assert trace.final_result is None

    def test_inference_trace_add_step(self):
        """Steps can be added to trace."""
        from cortical.reasoning.prism_pln import InferenceTrace, InferenceStep, TruthValue

        trace = InferenceTrace(query="risky(file_a)")

        step = InferenceStep(
            rule_antecedent="external_dependency(X)",
            rule_consequent="risky(X)",
            rule_truth_value=TruthValue(0.7, 0.8),
            antecedent_truth_value=TruthValue(0.9, 0.95),
            result_truth_value=TruthValue(0.63, 0.76),
            substitutions={"X": "file_a"}
        )

        trace.add_step(step)

        assert len(trace.steps) == 1
        assert trace.steps[0].rule_antecedent == "external_dependency(X)"

    def test_inference_trace_add_fact(self):
        """Facts used in inference are recorded."""
        from cortical.reasoning.prism_pln import InferenceTrace, TruthValue

        trace = InferenceTrace(query="needs_review(core_module)")

        trace.add_fact("has_todo(core_module)", TruthValue(0.9, 0.9))
        trace.add_fact("in_legacy(core_module)", TruthValue(0.8, 0.85))

        assert len(trace.facts_used) == 2
        assert "has_todo(core_module)" in trace.facts_used
        assert trace.facts_used["has_todo(core_module)"].strength == 0.9

    def test_inference_trace_to_dict(self):
        """InferenceTrace can be serialized to dict."""
        from cortical.reasoning.prism_pln import InferenceTrace, InferenceStep, TruthValue

        trace = InferenceTrace(
            query="flagged(auth_py)",
            aggregation_strategy="max"
        )
        trace.add_fact("has_hack(auth_py)", TruthValue(0.85, 0.9))
        trace.final_result = TruthValue(0.7, 0.8)

        d = trace.to_dict()

        assert d["query"] == "flagged(auth_py)"
        assert d["aggregation_strategy"] == "max"
        assert "has_hack(auth_py)" in d["facts_used"]
        assert d["final_result"]["strength"] == 0.7

    def test_inference_trace_explain_generates_human_readable_output(self):
        """InferenceTrace.explain() produces human-readable explanation."""
        from cortical.reasoning.prism_pln import InferenceTrace, InferenceStep, TruthValue

        trace = InferenceTrace(
            query="needs_review(test_file)",
            aggregation_strategy="revision"
        )
        trace.add_fact("has_todo(test_file)", TruthValue(0.9, 0.9))

        step = InferenceStep(
            rule_antecedent="has_todo(X)",
            rule_consequent="needs_review(X)",
            rule_truth_value=TruthValue(0.8, 0.9),
            antecedent_truth_value=TruthValue(0.9, 0.9),
            result_truth_value=TruthValue(0.72, 0.81),
            substitutions={"X": "test_file"},
            depth=0
        )
        trace.add_step(step)
        trace.final_result = TruthValue(0.72, 0.81)

        explanation = trace.explain()

        # Should contain key elements
        assert "Query: needs_review(test_file)" in explanation
        assert "has_todo(test_file)" in explanation
        assert "has_todo(X)" in explanation
        assert "needs_review(X)" in explanation
        assert "Final result:" in explanation


# =============================================================================
# QUERY WITH TRACE
# =============================================================================


class TestQueryWithTrace:
    """Tests for query_with_trace() - inference with full traceability."""

    def test_query_with_trace_returns_inference_trace(self):
        """query_with_trace returns an InferenceTrace object."""
        from cortical.reasoning.prism_pln import PLNReasoner, InferenceTrace

        reasoner = PLNReasoner()
        reasoner.assert_fact("has_pattern(file_a, todo)", strength=0.9, confidence=0.9)

        trace = reasoner.query_with_trace("has_pattern(file_a, todo)")

        assert isinstance(trace, InferenceTrace)
        assert trace.query == "has_pattern(file_a, todo)"

    def test_query_with_trace_direct_fact_lookup(self):
        """Direct fact lookup is recorded in trace."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        reasoner.assert_fact("risky(module_x)", strength=0.85, confidence=0.9)

        trace = reasoner.query_with_trace("risky(module_x)")

        # Should find the fact directly
        assert trace.final_result is not None
        assert trace.final_result.strength == pytest.approx(0.85, rel=0.01)
        assert "risky(module_x)" in trace.facts_used

    def test_query_with_trace_single_rule_inference(self):
        """Single rule inference is traced with step recorded."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        reasoner.assert_fact("has_todo(api_py)", strength=0.9, confidence=0.9)
        reasoner.assert_rule("has_todo(X)", "needs_review(X)", strength=0.8, confidence=0.9)

        trace = reasoner.query_with_trace("needs_review(api_py)")

        # Should have inference step recorded
        assert len(trace.steps) >= 1

        # Find the step that used our rule
        has_todo_step = None
        for step in trace.steps:
            if "has_todo" in step.rule_antecedent:
                has_todo_step = step
                break

        assert has_todo_step is not None
        assert has_todo_step.substitutions.get("X") == "api_py"

    def test_query_with_trace_chained_inference(self):
        """Multi-step inference chain is fully traced."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Chain: has_pattern → has_known_issue → needs_review
        reasoner.assert_fact("has_pattern(file_z, fixme)", strength=0.9, confidence=0.9)
        reasoner.assert_rule(
            "has_pattern(X, fixme)", "has_known_issue(X)",
            strength=0.85, confidence=0.9
        )
        reasoner.assert_rule(
            "has_known_issue(X)", "needs_review(X)",
            strength=0.7, confidence=0.85
        )

        trace = reasoner.query_with_trace("needs_review(file_z)", max_depth=5)

        # Should have the fact used
        assert "has_pattern(file_z, fixme)" in trace.facts_used

        # Should have final result
        assert trace.final_result is not None
        assert trace.final_result.strength > 0

    def test_query_with_trace_no_match_returns_empty_trace(self):
        """Query with no matching facts/rules returns trace with None result."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        # No facts or rules added

        trace = reasoner.query_with_trace("nonexistent(query)")

        assert trace.final_result is None
        assert trace.facts_used == {}
        assert trace.steps == []

    def test_query_with_trace_aggregation_recorded(self):
        """Multi-rule aggregation is recorded in trace."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Two different paths to same conclusion
        reasoner.assert_fact("has_todo(file_b)", strength=0.8, confidence=0.9)
        reasoner.assert_fact("high_churn(file_b)", strength=0.7, confidence=0.85)
        reasoner.assert_rule("has_todo(X)", "risky(X)", strength=0.6, confidence=0.8)
        reasoner.assert_rule("high_churn(X)", "risky(X)", strength=0.7, confidence=0.85)

        trace = reasoner.query_with_trace("risky(file_b)", aggregate="revision")

        assert trace.aggregation_strategy == "revision"
        # Should have aggregation inputs if multiple rules matched
        # (depending on implementation, may have 0, 1, or 2 inputs)

    def test_query_with_trace_max_depth_respected(self):
        """Inference depth is limited by max_depth parameter."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Deep chain that would require depth > 1
        reasoner.assert_fact("level0(file)", strength=0.9, confidence=0.9)
        reasoner.assert_rule("level0(X)", "level1(X)", strength=0.8, confidence=0.9)
        reasoner.assert_rule("level1(X)", "level2(X)", strength=0.8, confidence=0.9)
        reasoner.assert_rule("level2(X)", "level3(X)", strength=0.8, confidence=0.9)

        # Depth 1 should stop before reaching level3
        trace = reasoner.query_with_trace("level3(file)", max_depth=1)

        # Should not reach the conclusion with depth=1
        # (needs 3 hops: level0 -> level1 -> level2 -> level3)
        assert trace.final_result is None or len(trace.steps) <= 1


# =============================================================================
# AUDIT REASONER EXPLAINABILITY
# =============================================================================


class TestAuditReasonerExplainFileRisk:
    """Tests for explain_file_risk() in AuditReasoner."""

    def test_explain_file_risk_returns_dict_with_required_fields(self):
        """explain_file_risk returns dict with all required fields."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_todo(test_py)", strength=0.9, confidence=0.9)

        result = reasoner.explain_file_risk("test.py")

        # Required fields
        assert "file" in result
        assert "file_id" in result
        assert "facts" in result
        assert "traces" in result
        assert "summary" in result
        assert "raw_traces" in result
        assert "suggestions" in result

    def test_explain_file_risk_finds_facts_for_file(self):
        """explain_file_risk collects facts asserted about the file."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_fixme(module_py)", strength=0.85, confidence=0.9)
        reasoner.pln.assert_fact("has_hack(module_py)", strength=0.7, confidence=0.8)
        reasoner.pln.assert_fact("unrelated(other_py)", strength=0.9, confidence=0.9)

        result = reasoner.explain_file_risk("module.py")

        # Should find facts for module_py
        assert len(result["facts"]) == 2
        assert "has_fixme(module_py)" in result["facts"]
        assert "has_hack(module_py)" in result["facts"]
        # Should not include unrelated facts
        assert "unrelated(other_py)" not in result["facts"]

    def test_explain_file_risk_runs_traced_inference(self):
        """explain_file_risk runs inference with traces for risk queries."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_known_issue(critical_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_rule(
            "has_known_issue(X)", "needs_review(X)",
            strength=0.8, confidence=0.9
        )

        result = reasoner.explain_file_risk("critical.py")

        # Should have run inference for needs_review
        assert "needs_review" in result["traces"] or "needs_review" in result["raw_traces"]

    def test_explain_file_risk_generates_summary(self):
        """explain_file_risk generates human-readable summary."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_pattern(legacy_py, should_be)", strength=0.8, confidence=0.9)

        result = reasoner.explain_file_risk("legacy.py")

        summary = result["summary"]

        # Summary should contain key elements
        assert "legacy.py" in summary
        assert "FACTS" in summary or "Facts" in summary


class TestAuditReasonerSuggestedActions:
    """Tests for suggested actions generation."""

    def test_suggestions_for_todo_pattern(self):
        """TODO pattern triggers appropriate suggestion."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_todo(task_py)", strength=0.9, confidence=0.9)

        result = reasoner.explain_file_risk("task.py")

        assert any("TODO" in s for s in result["suggestions"])

    def test_suggestions_for_fixme_pattern(self):
        """FIXME pattern triggers appropriate suggestion."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_fixme(buggy_py)", strength=0.85, confidence=0.9)

        result = reasoner.explain_file_risk("buggy.py")

        assert any("FIXME" in s for s in result["suggestions"])

    def test_suggestions_for_hack_pattern(self):
        """HACK pattern triggers refactoring suggestion."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_hack(workaround_py)", strength=0.8, confidence=0.9)

        result = reasoner.explain_file_risk("workaround.py")

        assert any("HACK" in s or "refactor" in s.lower() for s in result["suggestions"])

    def test_suggestions_for_should_be_pattern(self):
        """'should be' pattern triggers investigation suggestion."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("has_should_be(spec_py)", strength=0.75, confidence=0.85)

        result = reasoner.explain_file_risk("spec.py")

        assert any("should be" in s.lower() or "spec" in s.lower() for s in result["suggestions"])

    def test_suggestions_for_high_churn(self):
        """High churn triggers splitting suggestion."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("high_churn(monolith_py)", strength=0.85, confidence=0.9)

        result = reasoner.explain_file_risk("monolith.py")

        assert any("split" in s.lower() or "module" in s.lower() for s in result["suggestions"])

    def test_suggestions_for_risky_inference(self):
        """Inferred 'risky' triggers test suggestion."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("risky(volatile_py)", strength=0.8, confidence=0.9)

        result = reasoner.explain_file_risk("volatile.py")

        assert any("test" in s.lower() for s in result["suggestions"])

    def test_suggestions_limited_to_five(self):
        """Suggestions are capped at 5 to avoid overwhelming output."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()

        # Add many different patterns
        reasoner.pln.assert_fact("has_todo(mega_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_fact("has_fixme(mega_py)", strength=0.85, confidence=0.9)
        reasoner.pln.assert_fact("has_hack(mega_py)", strength=0.8, confidence=0.9)
        reasoner.pln.assert_fact("has_future(mega_py)", strength=0.75, confidence=0.85)
        reasoner.pln.assert_fact("has_should_be(mega_py)", strength=0.7, confidence=0.85)
        reasoner.pln.assert_fact("high_churn(mega_py)", strength=0.8, confidence=0.9)
        reasoner.pln.assert_fact("incomplete(mega_py)", strength=0.7, confidence=0.8)
        reasoner.pln.assert_fact("risky(mega_py)", strength=0.85, confidence=0.9)

        result = reasoner.explain_file_risk("mega.py")

        assert len(result["suggestions"]) <= 5

    def test_no_suggestions_for_clean_file(self):
        """Files with no patterns have no suggestions."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        # No facts asserted for clean_file

        result = reasoner.explain_file_risk("clean.py")

        assert result["suggestions"] == []

    def test_suggestions_deduplicated(self):
        """Same suggestion is not repeated."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()

        # Multiple facts that would trigger same suggestion
        reasoner.pln.assert_fact("has_todo(dup_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_fact("file_pattern(dup_py, todo)", strength=0.85, confidence=0.9)

        result = reasoner.explain_file_risk("dup.py")

        # Count TODO-related suggestions
        todo_suggestions = [s for s in result["suggestions"] if "TODO" in s]
        assert len(todo_suggestions) <= 1  # Should not duplicate


# =============================================================================
# NLU EXPLAIN QUERIES
# =============================================================================


class TestNLUExplainQueries:
    """Tests for NLU translation of explain queries."""

    def test_translate_why_is_flagged_query(self):
        """'why is X flagged' is translated to explain intent."""
        from cortical.audits.reasoning import translate_audit_query

        query = translate_audit_query("why is auth.py flagged")

        assert query.intent == "explain"
        assert query.target_file == "auth.py"

    def test_translate_why_is_flagged_with_directory(self):
        """'dir/ why is X flagged' includes directory scope."""
        from cortical.audits.reasoning import translate_audit_query

        query = translate_audit_query("cortical/ why is storage.py flagged")

        assert query.intent == "explain"
        assert query.target_file == "storage.py"
        assert query.directory == "cortical/"

    def test_translate_explain_query(self):
        """'explain X' is translated to explain intent."""
        from cortical.audits.reasoning import translate_audit_query

        query = translate_audit_query("explain module.py")

        assert query.intent == "explain"
        assert query.target_file == "module.py"

    def test_translate_why_is_risky_query(self):
        """'why is X risky' is translated to explain intent."""
        from cortical.audits.reasoning import translate_audit_query

        query = translate_audit_query("why is test_utils.py risky")

        assert query.intent == "explain"
        assert query.target_file == "test_utils.py"

    def test_explain_query_preserves_file_extension(self):
        """File extensions are preserved in explain queries."""
        from cortical.audits.reasoning import translate_audit_query

        query = translate_audit_query("why is config.yaml flagged")

        assert query.target_file == "config.yaml"

    def test_explain_query_with_complex_filename(self):
        """Complex filenames with underscores/dashes are handled."""
        from cortical.audits.reasoning import translate_audit_query

        query = translate_audit_query("why is my_module_v2.py flagged")

        assert query.target_file == "my_module_v2.py"


# =============================================================================
# INTEGRATION SCENARIOS
# =============================================================================


class TestExplainabilityIntegration:
    """Integration tests for complete explainability scenarios."""

    def test_full_audit_explanation_workflow(self):
        """
        Complete workflow: assert facts → run inference → explain.

        Scenario: A file has TODO and high churn, should get multiple suggestions.
        """
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()

        # Setup facts
        reasoner.pln.assert_fact("has_todo(legacy_auth_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_fact("high_churn(legacy_auth_py)", strength=0.8, confidence=0.85)

        # Setup rules
        reasoner.pln.assert_rule(
            "has_todo(X)", "needs_review(X)",
            strength=0.8, confidence=0.9
        )
        reasoner.pln.assert_rule(
            "high_churn(X)", "risky(X)",
            strength=0.75, confidence=0.85
        )

        # Get explanation
        result = reasoner.explain_file_risk("legacy_auth.py")

        # Should have facts
        assert len(result["facts"]) >= 2

        # Should have suggestions
        assert len(result["suggestions"]) >= 2

        # Summary should be informative
        assert "legacy_auth" in result["summary"]

    def test_inference_trace_matches_explanation(self):
        """
        Raw traces should be consistent with summary explanation.

        The summary is derived from traces, so they should match.
        """
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()

        reasoner.pln.assert_fact("has_known_issue(problem_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_rule(
            "has_known_issue(X)", "needs_review(X)",
            strength=0.7, confidence=0.85
        )

        result = reasoner.explain_file_risk("problem.py")

        # If needs_review appears in raw_traces, it should appear in summary
        if "needs_review" in result["raw_traces"]:
            assert "needs_review" in result["summary"]

    def test_verbose_explanation_has_more_detail(self):
        """
        Verbose mode should provide more detailed traces.

        Note: This tests that verbose=True is handled (implementation may vary).
        """
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("incomplete(wip_py)", strength=0.7, confidence=0.8)

        # Non-verbose
        result_brief = reasoner.explain_file_risk("wip.py", verbose=False)

        # Verbose
        result_verbose = reasoner.explain_file_risk("wip.py", verbose=True)

        # Both should succeed
        assert "summary" in result_brief
        assert "summary" in result_verbose

    def test_explanation_works_with_multiple_facts(self):
        """
        Explanation handles files with multiple facts and patterns.
        """
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()

        # Setup scenario with multiple facts
        reasoner.pln.assert_fact("has_todo(risky_file_py)", strength=0.9, confidence=0.9)
        reasoner.pln.assert_fact("has_hack(risky_file_py)", strength=0.85, confidence=0.9)

        # Multiple rules that could fire
        reasoner.pln.assert_rule(
            "has_todo(X)", "needs_review(X)",
            strength=0.8, confidence=0.9
        )
        reasoner.pln.assert_rule(
            "has_hack(X)", "needs_review(X)",
            strength=0.85, confidence=0.9
        )

        result = reasoner.explain_file_risk("risky_file.py")

        # Should have facts for both patterns
        assert "has_todo(risky_file_py)" in result["facts"]
        assert "has_hack(risky_file_py)" in result["facts"]
        # Should have suggestions for both patterns
        assert len(result["suggestions"]) >= 2


# =============================================================================
# EDGE CASES
# =============================================================================


class TestExplainabilityEdgeCases:
    """Edge cases and boundary conditions."""

    def test_explain_file_with_special_characters_in_name(self):
        """Files with special characters are handled correctly."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()

        # Dots are replaced with underscores in file_id
        reasoner.pln.assert_fact("has_todo(config_v2_py)", strength=0.9, confidence=0.9)

        result = reasoner.explain_file_risk("config.v2.py")

        # Should still work (file_id normalizes the name)
        assert result["file_id"] == "config_v2_py"

    def test_explain_file_with_no_facts_no_inference(self):
        """File with no facts produces empty but valid response."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        # No facts added

        result = reasoner.explain_file_risk("new_file.py")

        assert result["facts"] == {}
        assert result["suggestions"] == []
        assert "summary" in result  # Summary still exists

    def test_explain_handles_zero_confidence_facts(self):
        """Facts with zero confidence are handled gracefully."""
        from cortical.audits.reasoning import AuditReasoner

        reasoner = AuditReasoner()
        reasoner.pln.assert_fact("uncertain(maybe_py)", strength=0.5, confidence=0.0)

        result = reasoner.explain_file_risk("maybe.py")

        # Should still work, though low confidence
        assert result is not None

    def test_trace_explain_with_empty_substitutions(self):
        """InferenceStep with empty substitutions renders correctly."""
        from cortical.reasoning.prism_pln import InferenceStep, TruthValue

        step = InferenceStep(
            rule_antecedent="always_true",
            rule_consequent="definitely_true",
            rule_truth_value=TruthValue(1.0, 1.0),
            antecedent_truth_value=TruthValue(1.0, 1.0),
            result_truth_value=TruthValue(1.0, 1.0),
            substitutions={},  # Empty
            depth=0
        )

        s = str(step)

        # Should not have empty brackets
        assert "[]" not in s or "[" not in s
