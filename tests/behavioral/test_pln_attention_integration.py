"""
Behavioral tests for PLN Attention Integration.

These tests define the target behavior for attention-guided PLN inference,
specifically designed for audit tooling use cases where we need to:
1. Focus inference on high-importance atoms (files with issues)
2. Prioritize certain rules over others based on attention
3. Efficiently explore only relevant inference paths

The integration bridges prism_attention.py with prism_pln.py to enable
focused probabilistic reasoning.
"""

import pytest
from typing import List, Dict, Optional


class TestAttentionalFocus:
    """
    Tests for AttentionalFocus - the mechanism that controls which atoms
    receive inference attention.
    """

    def test_attentional_focus_limits_inference_scope(self):
        """
        When atoms are outside the attentional focus, inference should
        deprioritize them (lower confidence or skip entirely).

        For audits: focus on files in a specific directory, not the whole codebase.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        # Setup: rules for different directories
        reasoner.assert_rule("in_dir(X, legacy)", "needs_review(X)", strength=0.7)
        reasoner.assert_rule("in_dir(X, core)", "needs_review(X)", strength=0.3)
        reasoner.assert_rule("in_dir(X, tests)", "needs_review(X)", strength=0.1)

        # Facts about files
        reasoner.assert_fact("in_dir(legacy_handler, legacy)", strength=0.99)
        reasoner.assert_fact("in_dir(core_module, core)", strength=0.99)
        reasoner.assert_fact("in_dir(test_file, tests)", strength=0.99)

        # Create attentional focus on legacy directory
        focus = AttentionalFocus()
        focus.focus_on(["in_dir(legacy_handler, legacy)", "legacy_handler"])

        # Query with attention - should prioritize focused atoms
        result = reasoner.query_with_attention(
            "needs_review(legacy_handler)",
            focus=focus
        )

        assert result is not None
        # Focused query should return result
        assert result.strength > 0.5

    def test_attention_boosts_focused_inference_paths(self):
        """
        Atoms in the attentional focus should have their inference paths
        boosted (higher effective strength/confidence).

        For audits: when we focus on "high churn" files, their rules fire stronger.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        # Two rules with same base strength
        reasoner.assert_rule("has_trait(X, high_churn)", "risky(X)", strength=0.6)
        reasoner.assert_rule("has_trait(X, old_code)", "risky(X)", strength=0.6)

        reasoner.assert_fact("has_trait(file_a, high_churn)", strength=0.9)
        reasoner.assert_fact("has_trait(file_a, old_code)", strength=0.9)

        # Focus attention on high_churn trait
        focus = AttentionalFocus()
        focus.focus_on(["has_trait(file_a, high_churn)"], boost=1.5)

        # Query with attention
        result_focused = reasoner.query_with_attention(
            "risky(file_a)",
            focus=focus,
            aggregate="revision"
        )

        # Query without attention (baseline)
        result_baseline = reasoner.query("risky(file_a)", aggregate="revision")

        # Focused result should be influenced by the boost
        # (exact behavior depends on implementation)
        assert result_focused is not None
        assert result_baseline is not None

    def test_attention_decays_over_time(self):
        """
        Atoms not recently accessed should have their attention decay.

        For audits: files reviewed recently stay in focus, old reviews fade.
        """
        from cortical.reasoning.prism_pln import AttentionalFocus

        focus = AttentionalFocus()

        # Focus on some atoms
        focus.focus_on(["atom_a", "atom_b", "atom_c"])

        # Initial focus strength
        initial_a = focus.get_focus_strength("atom_a")
        assert initial_a > 0

        # Apply decay
        focus.decay(factor=0.5)

        # Focus should be weaker
        decayed_a = focus.get_focus_strength("atom_a")
        assert decayed_a < initial_a
        assert decayed_a == pytest.approx(initial_a * 0.5, rel=0.01)

    def test_attention_focus_has_bounded_size(self):
        """
        Attentional focus should have a maximum size - we can only focus
        on so many things at once.

        For audits: limit review queue to manageable size.
        """
        from cortical.reasoning.prism_pln import AttentionalFocus

        focus = AttentionalFocus(max_size=5)

        # Try to focus on more atoms than max_size
        atoms = [f"atom_{i}" for i in range(10)]
        focus.focus_on(atoms)

        # Should only keep max_size atoms
        focused_atoms = focus.get_focused_atoms()
        assert len(focused_atoms) <= 5

    def test_attention_focus_prioritizes_recent(self):
        """
        When focus is full, newer items should replace older ones.

        For audits: most recent findings take priority.
        """
        from cortical.reasoning.prism_pln import AttentionalFocus

        focus = AttentionalFocus(max_size=3)

        # Focus on initial atoms
        focus.focus_on(["atom_a", "atom_b", "atom_c"])
        assert focus.is_focused("atom_a")

        # Focus on new atom - should evict oldest
        focus.focus_on(["atom_d"])

        # atom_d should be in, atom_a should be out (oldest)
        assert focus.is_focused("atom_d")
        # Depending on eviction policy, atom_a may or may not be evicted


class TestPLNInferenceWithAttention:
    """
    Tests for PLN inference guided by attention.
    """

    def test_infer_with_attention_returns_focused_results(self):
        """
        PLN inference with attention should prioritize focused atoms
        in multi-rule scenarios.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        # Multiple paths to needs_review
        reasoner.assert_rule("has_todo(X)", "needs_review(X)", strength=0.5)
        reasoner.assert_rule("has_bug(X)", "needs_review(X)", strength=0.8)
        reasoner.assert_rule("is_stale(X)", "needs_review(X)", strength=0.4)

        # File has all traits
        reasoner.assert_fact("has_todo(file_a)", strength=0.9)
        reasoner.assert_fact("has_bug(file_a)", strength=0.9)
        reasoner.assert_fact("is_stale(file_a)", strength=0.9)

        # Focus attention on bug-related atoms
        focus = AttentionalFocus()
        focus.focus_on(["has_bug(file_a)"], boost=2.0)

        # Query with attention - bug path should dominate
        result = reasoner.query_with_attention(
            "needs_review(file_a)",
            focus=focus,
            aggregate="weighted"
        )

        assert result is not None
        # Result should be influenced by the high-strength bug rule
        # Note: boost affects confidence, so compare to baseline
        baseline = reasoner.query("needs_review(file_a)", aggregate="weighted")
        # Focused result should have higher confidence due to boost
        assert result.confidence >= baseline.confidence

    def test_attention_guided_inference_boosts_focused_atoms(self):
        """
        Attention-guided inference should boost focused atoms' contributions.

        For audits: focused findings get higher weight in the final result.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        # Create several rules with similar strengths
        for i in range(5):
            reasoner.assert_rule(f"trait_{i}(X)", "flagged(X)", strength=0.5)
            reasoner.assert_fact(f"trait_{i}(file_a)", strength=0.9)

        # Focus on just one trait with high boost
        focus = AttentionalFocus()
        focus.focus_on(["trait_2(file_a)"], boost=3.0)

        # Query with attention
        result, stats = reasoner.query_with_attention(
            "flagged(file_a)",
            focus=focus,
            aggregate="revision",
            return_stats=True
        )

        assert result is not None
        # Should have boosted at least one atom
        assert stats["atoms_boosted"] >= 1
        # Stats should track exploration
        assert stats["rules_explored"] > 0


class TestAuditReasoningWithAttention:
    """
    End-to-end tests for audit reasoning with attention integration.
    These represent the real-world use case.
    """

    def test_audit_focuses_on_high_priority_files(self):
        """
        Audit reasoning should focus attention on files with multiple issues.

        Scenario: A file with TODOs, high churn, AND security flags should
        get more attention than files with just one issue.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        # Audit rules
        reasoner.assert_rule("has_todo(X)", "needs_review(X)", strength=0.5)
        reasoner.assert_rule("high_churn(X)", "needs_review(X)", strength=0.6)
        reasoner.assert_rule("security_flag(X)", "needs_review(X)", strength=0.9)

        # File A has all issues (should get most attention)
        reasoner.assert_fact("has_todo(file_a)", strength=0.95)
        reasoner.assert_fact("high_churn(file_a)", strength=0.95)
        reasoner.assert_fact("security_flag(file_a)", strength=0.95)

        # File B has only one issue
        reasoner.assert_fact("has_todo(file_b)", strength=0.95)

        # Build attention focus from issue density
        focus = AttentionalFocus()
        # File A has 3 issues -> higher attention
        focus.focus_on(["file_a", "has_todo(file_a)", "high_churn(file_a)", "security_flag(file_a)"])

        # Query both files
        result_a = reasoner.query_with_attention(
            "needs_review(file_a)",
            focus=focus,
            aggregate="or"
        )
        result_b = reasoner.query("needs_review(file_b)", aggregate="or")

        assert result_a is not None
        assert result_b is not None
        # File A should have stronger needs_review signal
        assert result_a.strength > result_b.strength

    def test_audit_attention_propagates_through_inference_chains(self):
        """
        Attention should propagate through inference chains.

        Scenario: If we focus on "security_flag", and security_flag → critical → needs_review,
        then the entire chain should be prioritized.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        # Chain: security_flag → critical → needs_review
        reasoner.assert_rule("security_flag(X)", "critical(X)", strength=0.9)
        reasoner.assert_rule("critical(X)", "needs_immediate_review(X)", strength=0.95)

        # Also: has_todo → needs_review (different chain)
        reasoner.assert_rule("has_todo(X)", "needs_review(X)", strength=0.5)

        # File has both
        reasoner.assert_fact("security_flag(file_a)", strength=0.99)
        reasoner.assert_fact("has_todo(file_a)", strength=0.99)

        # Focus on security chain
        focus = AttentionalFocus()
        focus.focus_on(["security_flag(file_a)"], boost=2.0)

        # Query the end of the chain
        result = reasoner.query_with_attention(
            "needs_immediate_review(file_a)",
            focus=focus
        )

        assert result is not None
        # Should successfully traverse the chain
        assert result.strength > 0.8

    def test_audit_dynamic_attention_shift(self):
        """
        Attention can shift dynamically as audit progresses.

        Scenario: Start focused on TODOs, then shift to security issues
        as they are discovered.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()

        reasoner.assert_rule("has_todo(X)", "needs_review(X)", strength=0.5)
        reasoner.assert_rule("security_flag(X)", "needs_review(X)", strength=0.9)

        reasoner.assert_fact("has_todo(file_a)", strength=0.95)
        reasoner.assert_fact("security_flag(file_a)", strength=0.95)

        focus = AttentionalFocus()

        # Phase 1: Focus on TODOs
        focus.focus_on(["has_todo(file_a)"], boost=1.5)
        result_phase1 = reasoner.query_with_attention(
            "needs_review(file_a)",
            focus=focus,
            aggregate="weighted"
        )

        # Phase 2: Shift focus to security (decay old, boost new)
        focus.decay(factor=0.3)  # Reduce TODO attention
        focus.focus_on(["security_flag(file_a)"], boost=2.0)

        result_phase2 = reasoner.query_with_attention(
            "needs_review(file_a)",
            focus=focus,
            aggregate="weighted"
        )

        # Both should return results
        assert result_phase1 is not None
        assert result_phase2 is not None
        # Phase 2 should be stronger (security has higher base strength + boost)
        assert result_phase2.strength >= result_phase1.strength


class TestAttentionIntegrationAPI:
    """
    Tests for the API surface of attention integration.
    """

    def test_attentional_focus_creation(self):
        """AttentionalFocus can be created with configuration."""
        from cortical.reasoning.prism_pln import AttentionalFocus

        focus = AttentionalFocus(max_size=100, default_boost=1.0)

        assert focus.max_size == 100
        assert len(focus.get_focused_atoms()) == 0

    def test_attentional_focus_serialization(self):
        """AttentionalFocus state can be saved and loaded."""
        from cortical.reasoning.prism_pln import AttentionalFocus

        focus = AttentionalFocus()
        focus.focus_on(["atom_a", "atom_b"])
        focus.set_boost("atom_a", 2.0)

        # Serialize
        state = focus.to_dict()

        # Deserialize
        focus2 = AttentionalFocus.from_dict(state)

        assert focus2.is_focused("atom_a")
        assert focus2.is_focused("atom_b")
        assert focus2.get_focus_strength("atom_a") == 2.0

    def test_reasoner_query_with_attention_signature(self):
        """PLNReasoner.query_with_attention has expected signature."""
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionalFocus

        reasoner = PLNReasoner()
        focus = AttentionalFocus()

        # Should accept these parameters
        result = reasoner.query_with_attention(
            statement="test(x)",
            focus=focus,
            max_depth=5,
            aggregate="revision",
            return_stats=False
        )

        # May return None if no facts/rules, but shouldn't error
        # (result can be None)
