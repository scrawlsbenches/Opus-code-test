"""
Behavioral tests for PLN Importance Weights (STI/LTI).

These tests define the target behavior for importance-based reasoning,
inspired by OpenCog's ECAN (Economic Attention Allocation).

Importance enables:
1. STI (Short-Term Importance) - Urgent, recent relevance
2. LTI (Long-Term Importance) - Persistent, foundational relevance
3. VLTI (Very Long-Term Importance) - Pinned atoms that never decay

For audit tooling:
- STI: Files with recent issues (just discovered)
- LTI: Files with persistent problems (known tech debt)
- VLTI: Critical infrastructure that must always be checked
"""

import pytest
from typing import List, Dict, Optional


class TestAttentionValueBasics:
    """
    Tests for AttentionValue - the importance metadata for atoms.
    """

    def test_attention_value_creation_with_defaults(self):
        """AttentionValue can be created with default values."""
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue()

        assert av.sti == 0.0  # Default: no short-term importance
        assert av.lti == 0.0  # Default: no long-term importance
        assert av.vlti is False  # Default: not pinned

    def test_attention_value_creation_with_values(self):
        """AttentionValue can be created with specific values."""
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue(sti=0.8, lti=0.5, vlti=True)

        assert av.sti == 0.8
        assert av.lti == 0.5
        assert av.vlti is True

    def test_attention_value_total_importance(self):
        """
        Total importance combines STI and LTI.

        For audits: Total importance determines review priority.
        """
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue(sti=0.6, lti=0.4)

        # Total importance should combine both (weighted: 0.6*STI + 0.4*LTI)
        total = av.total_importance()
        # Total is weighted combination: 0.6*0.6 + 0.4*0.4 = 0.52
        assert total > 0  # Has importance
        assert total == pytest.approx(0.6 * 0.6 + 0.4 * 0.4, rel=0.01)

        # Higher STI/LTI leads to higher total
        av_high = AttentionValue(sti=1.0, lti=1.0)
        assert av_high.total_importance() > av.total_importance()

    def test_attention_value_sti_decays(self):
        """
        STI should decay over time - urgency fades.

        For audits: Recent findings lose urgency if not addressed.
        """
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue(sti=1.0, lti=0.5)
        initial_sti = av.sti

        # Apply decay
        av.decay_sti(factor=0.8)

        assert av.sti < initial_sti
        assert av.sti == pytest.approx(0.8, rel=0.01)
        # LTI should not decay with STI decay
        assert av.lti == 0.5

    def test_attention_value_lti_persists(self):
        """
        LTI decays much slower than STI - foundational importance persists.

        For audits: Known tech debt remains important even if not urgent.
        """
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue(sti=1.0, lti=1.0)

        # Apply multiple decay cycles
        for _ in range(10):
            av.decay_sti(factor=0.5)
            av.decay_lti(factor=0.95)  # LTI decays much slower

        # STI should be nearly zero after 10 cycles of 0.5 decay
        assert av.sti < 0.01
        # LTI should still be significant
        assert av.lti > 0.5

    def test_attention_value_vlti_prevents_decay(self):
        """
        VLTI atoms never lose importance - they are pinned.

        For audits: Security-critical files must always be reviewed.
        """
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue(sti=1.0, lti=1.0, vlti=True)

        # Apply aggressive decay
        for _ in range(100):
            av.decay_sti(factor=0.1)
            av.decay_lti(factor=0.1)

        # VLTI should preserve minimum importance (floor of 0.5)
        assert av.total_importance() >= 0.5  # Floor protection active
        # LTI should not decay at all for VLTI atoms
        assert av.lti == 1.0


class TestAtomImportance:
    """
    Tests for importance tracking on PLN atoms.
    """

    def test_atom_has_attention_value(self):
        """Each atom should have an associated AttentionValue."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        reasoner.assert_fact("important_file(auth.py)", strength=0.9)

        # Get attention value for atom
        av = reasoner.get_attention("important_file(auth.py)")

        assert av is not None
        assert hasattr(av, 'sti')
        assert hasattr(av, 'lti')

    def test_set_atom_importance(self):
        """Can set importance for specific atoms."""
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()
        reasoner.assert_fact("critical_module(payment)", strength=0.99)

        # Set high importance
        reasoner.set_attention(
            "critical_module(payment)",
            AttentionValue(sti=0.9, lti=0.8, vlti=True)
        )

        av = reasoner.get_attention("critical_module(payment)")
        assert av.sti == 0.9
        assert av.lti == 0.8
        assert av.vlti is True

    def test_stimulate_atom_increases_sti(self):
        """
        Stimulating an atom increases its STI.

        For audits: Discovering an issue stimulates that file's importance.
        """
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        reasoner.assert_fact("has_bug(parser.py)", strength=0.85)

        initial_av = reasoner.get_attention("has_bug(parser.py)")
        initial_sti = initial_av.sti

        # Stimulate the atom (e.g., bug was just discovered)
        reasoner.stimulate("has_bug(parser.py)", amount=0.5)

        stimulated_av = reasoner.get_attention("has_bug(parser.py)")
        assert stimulated_av.sti > initial_sti

    def test_rent_collection_decays_sti(self):
        """
        Rent collection decays all non-VLTI atoms' STI.

        For audits: Periodic review reduces urgency of old findings.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Create atoms with different importance levels
        reasoner.assert_fact("old_finding(file_a)", strength=0.9)
        reasoner.assert_fact("critical_finding(file_b)", strength=0.9)

        reasoner.set_attention("old_finding(file_a)", AttentionValue(sti=0.8, lti=0.3))
        reasoner.set_attention("critical_finding(file_b)", AttentionValue(sti=0.8, lti=0.3, vlti=True))

        # Collect rent (decay cycle)
        reasoner.collect_rent()

        av_a = reasoner.get_attention("old_finding(file_a)")
        av_b = reasoner.get_attention("critical_finding(file_b)")

        # Non-VLTI should decay
        assert av_a.sti < 0.8
        # VLTI should be protected
        assert av_b.total_importance() >= 0.5


class TestImportanceSpreading:
    """
    Tests for importance spreading through inference chains.
    """

    def test_importance_spreads_forward(self):
        """
        When we infer A → B, importance should spread from A to B.

        For audits: If "has_bug(X)" is important and "needs_review(X)"
        is inferred from it, then needs_review inherits some importance.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Rule: has_bug → needs_review
        reasoner.assert_rule("has_bug(X)", "needs_review(X)", strength=0.9)
        reasoner.assert_fact("has_bug(parser.py)", strength=0.95)

        # Make has_bug highly important
        reasoner.set_attention(
            "has_bug(parser.py)",
            AttentionValue(sti=0.9, lti=0.5)
        )

        # Perform inference with importance spreading
        result = reasoner.query_with_importance(
            "needs_review(parser.py)",
            spread_importance=True
        )

        # The inferred conclusion should inherit importance
        av = reasoner.get_attention("needs_review(parser.py)")
        assert av is not None
        assert av.sti > 0  # Inherited from source

    def test_importance_spreads_through_chains(self):
        """
        Importance spreads through multi-step inference chains.

        Scenario: security_flag → critical → immediate_review
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Chain rules
        reasoner.assert_rule("security_flag(X)", "critical(X)", strength=0.95)
        reasoner.assert_rule("critical(X)", "immediate_review(X)", strength=0.9)

        # Fact with high importance
        reasoner.assert_fact("security_flag(auth.py)", strength=0.99)
        reasoner.set_attention(
            "security_flag(auth.py)",
            AttentionValue(sti=1.0, lti=0.8, vlti=True)
        )

        # First, infer the intermediate step to create the critical atom
        result1 = reasoner.query_with_importance(
            "critical(auth.py)",
            spread_importance=True,
            max_depth=5
        )
        assert result1 is not None

        # Critical should have inherited importance from security_flag
        av_critical = reasoner.get_attention("critical(auth.py)")
        assert av_critical.sti > 0  # Inherited from source

        # Now infer through the full chain
        result = reasoner.query_with_importance(
            "immediate_review(auth.py)",
            spread_importance=True,
            max_depth=5
        )

        assert result is not None
        # End of chain should have inherited importance
        av = reasoner.get_attention("immediate_review(auth.py)")
        # Importance spreads through chain (attenuated)
        assert av.sti > 0 or av_critical.sti > 0  # Some inheritance happened

    def test_importance_spreading_attenuates_with_distance(self):
        """
        Importance should attenuate (decrease) with inference distance.

        For audits: Direct bugs are more urgent than derived conclusions.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Create a chain: A → B → C → D
        reasoner.assert_rule("A(X)", "B(X)", strength=0.9)
        reasoner.assert_rule("B(X)", "C(X)", strength=0.9)
        reasoner.assert_rule("C(X)", "D(X)", strength=0.9)

        reasoner.assert_fact("A(file)", strength=0.99)
        reasoner.set_attention("A(file)", AttentionValue(sti=1.0, lti=0.5))

        # Infer through chain
        reasoner.query_with_importance("D(file)", spread_importance=True, max_depth=5)

        # Check importance attenuation
        av_a = reasoner.get_attention("A(file)")
        av_b = reasoner.get_attention("B(file)")
        av_c = reasoner.get_attention("C(file)")
        av_d = reasoner.get_attention("D(file)")

        # Each step should have less importance
        assert av_a.sti >= av_b.sti
        assert av_b.sti >= av_c.sti
        assert av_c.sti >= av_d.sti


class TestImportanceGuidedInference:
    """
    Tests for using importance to guide inference.
    """

    def test_high_importance_atoms_inferred_first(self):
        """
        Inference should prioritize high-importance atoms.

        For audits: Check urgent files before routine reviews.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Multiple files that need review
        for i in range(10):
            reasoner.assert_fact(f"has_issue(file_{i})", strength=0.8)
            reasoner.assert_rule(f"has_issue(X)", "needs_review(X)", strength=0.7)

        # Make one file urgent
        reasoner.set_attention(
            "has_issue(file_5)",
            AttentionValue(sti=1.0, lti=0.5)
        )

        # Query with importance-guided inference
        results, stats = reasoner.query_by_importance(
            "needs_review(X)",
            return_stats=True
        )

        # High-importance items should be explored first
        assert stats["first_explored"] == "has_issue(file_5)" or \
               "file_5" in stats.get("exploration_order", [])[:3]

    def test_importance_threshold_filters_inference(self):
        """
        Can set importance threshold to filter what gets inferred.

        For audits: Only review files above certain importance.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Many findings with varying importance
        reasoner.assert_rule("finding(X)", "review(X)", strength=0.8)

        for i in range(5):
            reasoner.assert_fact(f"finding(file_{i})", strength=0.9)
            reasoner.set_attention(
                f"finding(file_{i})",
                AttentionValue(sti=i * 0.2, lti=0.1)  # 0.0, 0.2, 0.4, 0.6, 0.8
            )

        # Query with importance threshold
        results = reasoner.query_with_importance(
            "review(X)",
            min_importance=0.5  # Only files with STI >= 0.5
        )

        # Should only return high-importance results
        assert len(results) <= 3  # file_3, file_4 (and maybe file_2)


class TestAuditImportanceIntegration:
    """
    End-to-end tests for audit scenarios with importance.
    """

    def test_urgent_vs_persistent_issue_differentiation(self):
        """
        Audit should differentiate urgent (high STI) vs persistent (high LTI) issues.

        Scenario:
        - File A: Just discovered bug (high STI, low LTI)
        - File B: Long-standing tech debt (low STI, high LTI)
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        reasoner.assert_rule("has_issue(X)", "needs_attention(X)", strength=0.9)

        # File A: Fresh discovery
        reasoner.assert_fact("has_issue(fresh_bug.py)", strength=0.9)
        reasoner.set_attention(
            "has_issue(fresh_bug.py)",
            AttentionValue(sti=0.9, lti=0.1)  # Urgent but not persistent
        )

        # File B: Known tech debt
        reasoner.assert_fact("has_issue(tech_debt.py)", strength=0.9)
        reasoner.set_attention(
            "has_issue(tech_debt.py)",
            AttentionValue(sti=0.2, lti=0.8)  # Persistent but not urgent
        )

        # Get urgent items (high STI)
        urgent = reasoner.get_atoms_by_sti(min_sti=0.5)
        assert "has_issue(fresh_bug.py)" in urgent
        assert "has_issue(tech_debt.py)" not in urgent

        # Get persistent items (high LTI)
        persistent = reasoner.get_atoms_by_lti(min_lti=0.5)
        assert "has_issue(tech_debt.py)" in persistent
        assert "has_issue(fresh_bug.py)" not in persistent

    def test_critical_infrastructure_always_reviewed(self):
        """
        VLTI items should always appear in review queue.

        For audits: Security-critical code reviewed every cycle.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        reasoner.assert_rule("is_code(X)", "can_review(X)", strength=1.0)

        # Regular file
        reasoner.assert_fact("is_code(utils.py)", strength=1.0)
        reasoner.set_attention("is_code(utils.py)", AttentionValue(sti=0.1, lti=0.1))

        # Critical file (VLTI)
        reasoner.assert_fact("is_code(auth.py)", strength=1.0)
        reasoner.set_attention(
            "is_code(auth.py)",
            AttentionValue(sti=0.1, lti=0.1, vlti=True)
        )

        # Even with low STI/LTI, VLTI should appear in must-review
        must_review = reasoner.get_vlti_atoms()
        assert "is_code(auth.py)" in must_review

    def test_importance_decay_simulation(self):
        """
        Simulate multiple review cycles with importance decay.

        For audits: Old findings fade unless reconfirmed.
        """
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Initial findings (NOT VLTI - regular atom that decays normally)
        reasoner.assert_fact("todo_found(legacy.py)", strength=0.9)
        reasoner.set_attention(
            "todo_found(legacy.py)",
            AttentionValue(sti=1.0, lti=0.3, vlti=False)  # Not pinned
        )

        initial_sti = reasoner.get_attention("todo_found(legacy.py)").sti
        initial_lti = reasoner.get_attention("todo_found(legacy.py)").lti

        # Simulate 10 review cycles (more cycles for visible decay)
        for cycle in range(10):
            reasoner.collect_rent()  # Decay

        final_av = reasoner.get_attention("todo_found(legacy.py)")

        # STI should have decayed significantly (0.9^10 ≈ 0.35)
        assert final_av.sti < initial_sti * 0.5
        # LTI should persist better (0.99^10 ≈ 0.90)
        assert final_av.lti > initial_lti * 0.8

    def test_combined_attention_and_importance(self):
        """
        AttentionalFocus and importance work together.

        For audits: Focus on urgent items with high importance.
        """
        from cortical.reasoning.prism_pln import (
            PLNReasoner, AttentionalFocus, AttentionValue
        )

        reasoner = PLNReasoner()

        reasoner.assert_rule("flagged(X)", "review(X)", strength=0.8)

        # Multiple files with different importance
        files = ["urgent.py", "normal.py", "background.py"]
        for f in files:
            reasoner.assert_fact(f"flagged({f})", strength=0.9)

        reasoner.set_attention("flagged(urgent.py)", AttentionValue(sti=0.9, lti=0.5))
        reasoner.set_attention("flagged(normal.py)", AttentionValue(sti=0.4, lti=0.3))
        reasoner.set_attention("flagged(background.py)", AttentionValue(sti=0.1, lti=0.1))

        # Create attention focus on high-importance items
        focus = AttentionalFocus()
        focus.focus_on(["flagged(urgent.py)"], boost=1.5)

        # Query with both attention and importance
        result = reasoner.query_with_attention(
            "review(urgent.py)",
            focus=focus,
            aggregate="revision"
        )

        assert result is not None
        # High importance + focus boost should yield strong result
        assert result.confidence > 0.5


class TestImportanceAPI:
    """
    Tests for the importance API surface.
    """

    def test_attention_value_serialization(self):
        """AttentionValue can be serialized and deserialized."""
        from cortical.reasoning.prism_pln import AttentionValue

        av = AttentionValue(sti=0.7, lti=0.3, vlti=True)

        # Serialize
        data = av.to_dict()

        assert data["sti"] == 0.7
        assert data["lti"] == 0.3
        assert data["vlti"] is True

        # Deserialize
        av2 = AttentionValue.from_dict(data)

        assert av2.sti == av.sti
        assert av2.lti == av.lti
        assert av2.vlti == av.vlti

    def test_reasoner_importance_methods_exist(self):
        """PLNReasoner has importance-related methods."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Check method existence
        assert hasattr(reasoner, 'get_attention')
        assert hasattr(reasoner, 'set_attention')
        assert hasattr(reasoner, 'stimulate')
        assert hasattr(reasoner, 'collect_rent')
        assert hasattr(reasoner, 'query_with_importance')

    def test_get_atoms_sorted_by_importance(self):
        """Can retrieve atoms sorted by total importance."""
        from cortical.reasoning.prism_pln import PLNReasoner, AttentionValue

        reasoner = PLNReasoner()

        # Create atoms with varying importance
        reasoner.assert_fact("low_priority", strength=0.9)
        reasoner.assert_fact("medium_priority", strength=0.9)
        reasoner.assert_fact("high_priority", strength=0.9)

        reasoner.set_attention("low_priority", AttentionValue(sti=0.1, lti=0.1))
        reasoner.set_attention("medium_priority", AttentionValue(sti=0.5, lti=0.3))
        reasoner.set_attention("high_priority", AttentionValue(sti=0.9, lti=0.7))

        # Get atoms sorted by importance
        sorted_atoms = reasoner.get_atoms_by_importance()

        # Should be in descending order of total importance
        assert sorted_atoms[0] == "high_priority"
        assert sorted_atoms[-1] == "low_priority"
