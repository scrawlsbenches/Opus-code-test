"""
Tests for PRISM-PLN: Probabilistic Logic Networks with Synaptic Learning.

TDD: These tests define expected behavior before implementation.
"""

import pytest
import math


class TestTruthValue:
    """Test probabilistic truth values."""

    def test_truth_value_creation(self):
        """Test creating a truth value with strength and confidence."""
        from cortical.reasoning.prism_pln import TruthValue

        tv = TruthValue(strength=0.8, confidence=0.9)

        assert tv.strength == 0.8
        assert tv.confidence == 0.9

    def test_truth_value_bounds(self):
        """Truth values are clamped to [0, 1]."""
        from cortical.reasoning.prism_pln import TruthValue

        tv = TruthValue(strength=1.5, confidence=-0.1)

        assert tv.strength == 1.0
        assert tv.confidence == 0.0

    def test_truth_value_mean(self):
        """Mean combines strength with prior assumption."""
        from cortical.reasoning.prism_pln import TruthValue

        tv = TruthValue(strength=0.8, confidence=0.9)
        # High confidence → mean close to strength
        assert 0.75 < tv.mean() < 0.85

    def test_truth_value_revision(self):
        """Two truth values can be revised together."""
        from cortical.reasoning.prism_pln import TruthValue

        tv1 = TruthValue(strength=0.8, confidence=0.5)
        tv2 = TruthValue(strength=0.6, confidence=0.7)

        revised = tv1.revise(tv2)

        # Revised confidence should be higher than either alone
        assert revised.confidence >= max(tv1.confidence, tv2.confidence)

    def test_truth_value_to_probability(self):
        """Convert truth value to probability estimate."""
        from cortical.reasoning.prism_pln import TruthValue

        tv = TruthValue(strength=0.9, confidence=0.95)
        prob = tv.to_probability()

        assert 0.0 <= prob <= 1.0
        assert prob > 0.8  # High strength + high confidence


class TestLogicalOperations:
    """Test logical operations on truth values."""

    def test_negation(self):
        """NOT operation inverts strength."""
        from cortical.reasoning.prism_pln import TruthValue, pln_not

        tv = TruthValue(strength=0.8, confidence=0.9)
        neg = pln_not(tv)

        assert neg.strength == pytest.approx(0.2, rel=0.01)
        assert neg.confidence == tv.confidence  # Confidence preserved

    def test_conjunction(self):
        """AND operation combines truth values."""
        from cortical.reasoning.prism_pln import TruthValue, pln_and

        tv1 = TruthValue(strength=0.8, confidence=0.9)
        tv2 = TruthValue(strength=0.7, confidence=0.8)

        conj = pln_and(tv1, tv2)

        # Conjunction strength <= min of operands
        assert conj.strength <= min(tv1.strength, tv2.strength)

    def test_disjunction(self):
        """OR operation combines truth values."""
        from cortical.reasoning.prism_pln import TruthValue, pln_or

        tv1 = TruthValue(strength=0.8, confidence=0.9)
        tv2 = TruthValue(strength=0.7, confidence=0.8)

        disj = pln_or(tv1, tv2)

        # Disjunction strength >= max of operands
        assert disj.strength >= max(tv1.strength, tv2.strength)

    def test_implication(self):
        """Implication A → B."""
        from cortical.reasoning.prism_pln import TruthValue, pln_implication

        # If A is true and A→B is strong, B should be likely true
        tv_a = TruthValue(strength=0.9, confidence=0.9)
        tv_impl = TruthValue(strength=0.8, confidence=0.85)

        tv_b = pln_implication(tv_a, tv_impl)

        assert tv_b.strength > 0.5


class TestInferenceRules:
    """Test PLN inference rules."""

    def test_deduction(self):
        """Deduction: A→B, B→C ⊢ A→C."""
        from cortical.reasoning.prism_pln import TruthValue, deduce

        # A implies B (strong)
        tv_ab = TruthValue(strength=0.9, confidence=0.9)
        # B implies C (strong)
        tv_bc = TruthValue(strength=0.85, confidence=0.85)

        # Therefore A implies C
        tv_ac = deduce(tv_ab, tv_bc)

        assert tv_ac.strength > 0.7
        assert tv_ac.confidence > 0.5

    def test_induction(self):
        """Induction: A→B, A→C ⊢ B→C (with lower confidence)."""
        from cortical.reasoning.prism_pln import TruthValue, induce

        tv_ab = TruthValue(strength=0.9, confidence=0.9)
        tv_ac = TruthValue(strength=0.8, confidence=0.85)

        tv_bc = induce(tv_ab, tv_ac)

        # Induction has lower confidence than deduction
        assert tv_bc.confidence < min(tv_ab.confidence, tv_ac.confidence)

    def test_abduction(self):
        """Abduction: A→B, B ⊢ A (with lower confidence)."""
        from cortical.reasoning.prism_pln import TruthValue, abduce

        tv_ab = TruthValue(strength=0.9, confidence=0.9)
        tv_b = TruthValue(strength=0.95, confidence=0.9)

        tv_a = abduce(tv_ab, tv_b)

        # Abduction has lower confidence (reasoning backwards)
        assert tv_a.confidence < tv_ab.confidence


class TestProbabilisticAtom:
    """Test atoms (statements) with probabilistic truth values."""

    def test_atom_creation(self):
        """Create a probabilistic atom."""
        from cortical.reasoning.prism_pln import Atom, TruthValue

        atom = Atom(
            name="likes(alice, bob)",
            truth_value=TruthValue(0.8, 0.9)
        )

        assert atom.name == "likes(alice, bob)"
        assert atom.truth_value.strength == 0.8

    def test_atom_with_arguments(self):
        """Atoms can have structured arguments."""
        from cortical.reasoning.prism_pln import Atom, TruthValue

        atom = Atom(
            predicate="likes",
            arguments=["alice", "bob"],
            truth_value=TruthValue(0.8, 0.9)
        )

        assert atom.predicate == "likes"
        assert atom.arguments == ["alice", "bob"]


class TestPLNGraph:
    """Test the PLN knowledge graph."""

    def test_graph_creation(self):
        """Create a PLN graph."""
        from cortical.reasoning.prism_pln import PLNGraph

        graph = PLNGraph()

        assert graph.atom_count == 0
        assert graph.link_count == 0

    def test_add_atom(self):
        """Add atoms to graph."""
        from cortical.reasoning.prism_pln import PLNGraph, TruthValue

        graph = PLNGraph()
        graph.add_atom("bird(tweety)", TruthValue(0.95, 0.9))
        graph.add_atom("canfly(tweety)", TruthValue(0.8, 0.7))

        assert graph.atom_count == 2

    def test_add_implication_link(self):
        """Add implication links between atoms."""
        from cortical.reasoning.prism_pln import PLNGraph, TruthValue

        graph = PLNGraph()
        graph.add_atom("bird(X)", TruthValue(1.0, 1.0))
        graph.add_atom("canfly(X)", TruthValue(1.0, 1.0))

        graph.add_implication("bird(X)", "canfly(X)", TruthValue(0.9, 0.85))

        assert graph.link_count == 1

    def test_query_truth_value(self):
        """Query the truth value of an atom."""
        from cortical.reasoning.prism_pln import PLNGraph, TruthValue

        graph = PLNGraph()
        graph.add_atom("smart(alice)", TruthValue(0.85, 0.9))

        tv = graph.get_truth_value("smart(alice)")

        assert tv.strength == 0.85
        assert tv.confidence == 0.9

    def test_inference_chain(self):
        """Perform inference through chain of implications."""
        from cortical.reasoning.prism_pln import PLNGraph, TruthValue

        graph = PLNGraph()

        # Tweety is a bird (high confidence)
        graph.add_atom("bird(tweety)", TruthValue(0.99, 0.95))

        # Birds typically fly
        graph.add_atom("bird(X)", TruthValue(1.0, 1.0))
        graph.add_atom("canfly(X)", TruthValue(1.0, 1.0))
        graph.add_implication("bird(X)", "canfly(X)", TruthValue(0.85, 0.9))

        # Infer: can Tweety fly?
        result = graph.infer("canfly(tweety)")

        assert result is not None
        assert result.strength > 0.7


class TestPRISMIntegration:
    """Test integration with PRISM synaptic learning."""

    def test_synaptic_truth_value(self):
        """Truth values can have synaptic learning."""
        from cortical.reasoning.prism_pln import SynapticTruthValue

        stv = SynapticTruthValue(strength=0.7, confidence=0.1)

        # Observe positive evidence
        stv.observe(positive=True)
        stv.observe(positive=True)
        stv.observe(positive=True)

        # Strength and confidence should increase
        assert stv.strength > 0.7
        assert stv.confidence > 0.5

    def test_truth_value_decay(self):
        """Truth values decay without reinforcement."""
        from cortical.reasoning.prism_pln import SynapticTruthValue

        stv = SynapticTruthValue(strength=0.9, confidence=0.9)

        stv.apply_decay(factor=0.9)

        assert stv.confidence < 0.9  # Confidence decays

    def test_evidence_accumulation(self):
        """Evidence accumulates over multiple observations."""
        from cortical.reasoning.prism_pln import SynapticTruthValue

        stv = SynapticTruthValue(strength=0.5, confidence=0.1)

        # Many positive observations
        for _ in range(20):
            stv.observe(positive=True)

        assert stv.strength > 0.8
        assert stv.confidence > 0.8

    def test_conflicting_evidence(self):
        """Conflicting evidence increases uncertainty."""
        from cortical.reasoning.prism_pln import SynapticTruthValue

        stv = SynapticTruthValue(strength=0.5, confidence=0.5)

        # Mixed evidence
        for _ in range(10):
            stv.observe(positive=True)
            stv.observe(positive=False)

        # Strength should stay near 0.5, confidence may decrease
        assert 0.4 < stv.strength < 0.6


class TestPLNReasoner:
    """Test the PLN reasoning engine."""

    def test_reasoner_creation(self):
        """Create a PLN reasoner."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        assert reasoner is not None

    def test_assert_fact(self):
        """Assert facts to the reasoner."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        reasoner.assert_fact("mammal(dog)", strength=0.99)
        reasoner.assert_fact("mammal(cat)", strength=0.99)

        assert reasoner.fact_count >= 2

    def test_assert_rule(self):
        """Assert rules (implications) to the reasoner."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()
        reasoner.assert_rule(
            antecedent="mammal(X)",
            consequent="warm_blooded(X)",
            strength=0.95
        )

        assert reasoner.rule_count >= 1

    def test_query(self):
        """Query the reasoner."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        reasoner.assert_fact("bird(tweety)", strength=0.99)
        reasoner.assert_rule("bird(X)", "has_feathers(X)", strength=0.98)

        result = reasoner.query("has_feathers(tweety)")

        assert result is not None
        assert result.strength > 0.9

    def test_backward_chaining(self):
        """Backward chaining inference."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        # Knowledge base
        reasoner.assert_fact("bird(tweety)", strength=0.99)
        reasoner.assert_rule("bird(X)", "canfly(X)", strength=0.8)
        reasoner.assert_rule("canfly(X)", "has_wings(X)", strength=0.95)

        # Query: does tweety have wings?
        result = reasoner.query("has_wings(tweety)")

        assert result is not None
        assert result.strength > 0.7

    def test_learning_from_evidence(self):
        """Reasoner learns from observed evidence."""
        from cortical.reasoning.prism_pln import PLNReasoner

        reasoner = PLNReasoner()

        reasoner.assert_rule("bird(X)", "canfly(X)", strength=0.8)

        # First establish that penguin is a bird
        reasoner.assert_fact("bird(penguin)", strength=0.99)

        # Observe evidence: penguins are birds but don't fly
        reasoner.observe("canfly(penguin)", False)

        # Rule strength should decrease (one negative observation)
        rule_tv = reasoner.get_rule_truth("bird(X)", "canfly(X)")
        # After 1 negative observation on a SynapticTruthValue that started at 0.8
        # with 0 positive and 1 negative: strength = (0+1)/(1+2) = 0.33
        assert rule_tv.strength < 0.8


class TestSerialization:
    """Test saving and loading PLN state."""

    def test_save_load_roundtrip(self):
        """Save and load PLN graph."""
        import tempfile
        import os
        from cortical.reasoning.prism_pln import PLNGraph, TruthValue

        graph = PLNGraph()
        graph.add_atom("test(a)", TruthValue(0.8, 0.9))
        graph.add_atom("test(b)", TruthValue(0.7, 0.85))
        graph.add_implication("test(a)", "test(b)", TruthValue(0.75, 0.8))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pln.json")
            graph.save(path)

            loaded = PLNGraph.load(path)

            assert loaded.atom_count == graph.atom_count
            assert loaded.link_count == graph.link_count
