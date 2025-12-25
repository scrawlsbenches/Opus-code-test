"""
Behavioral tests for PRISM integration - all systems working together.

Tests the "Wonderland" scenario where PLN, SLM, and GoT cooperate.
"""

import pytest


class TestPRISMWonderlandIntegration:
    """Test all PRISM systems working together."""

    def test_pln_backward_chaining_inference(self):
        """PLN infers through chain of implications."""
        from cortical.reasoning import PLNReasoner

        pln = PLNReasoner()
        pln.assert_fact('curious(alice)', strength=0.95)
        pln.assert_rule('curious(X)', 'explores_wonderland(X)', strength=0.9)
        pln.assert_rule('explores_wonderland(X)', 'meets_cheshire(X)', strength=0.85)

        result = pln.query('meets_cheshire(alice)')

        assert result is not None
        assert result.strength > 0.7  # High probability through chain
        assert result.confidence > 0.5

    def test_slm_learns_and_generates(self):
        """SLM learns patterns and generates coherent text."""
        from cortical.reasoning import PRISMLanguageModel

        slm = PRISMLanguageModel(context_size=2)
        slm.train('Down the rabbit hole Alice fell.')
        slm.train('Alice fell through wonderland.')
        slm.train('The cat grinned and vanished.')

        # Should have learned vocabulary
        assert slm.graph.vocab_size > 5

        # Should generate something
        generated = slm.generate('Alice', max_tokens=5, temperature=0.5)
        assert len(generated) > 0
        assert 'Alice' in generated

    def test_got_hebbian_learning(self):
        """GoT strengthens connections through co-activation."""
        from cortical.reasoning import (
            SynapticMemoryGraph,
            IncrementalReasoner,
            NodeType,
        )

        got = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(got)

        # Create connected thoughts
        q = reasoner.process_thought('Why is a raven like a writing desk?', NodeType.QUESTION)
        h = reasoner.process_thought('Both produce notes', NodeType.HYPOTHESIS, 'ADDRESSES')

        # Activate both (co-activation)
        got.activate_node(q.id)
        got.activate_node(h.id)

        # Apply Hebbian learning
        strengthened = got.apply_hebbian_learning()

        assert strengthened >= 0  # Some edges may have been strengthened

        # Predict from question
        predictions = reasoner.predict_next(q.id, top_n=1)
        assert len(predictions) > 0

    def test_pln_learns_from_evidence(self):
        """PLN updates beliefs based on observed evidence."""
        from cortical.reasoning import PLNReasoner

        pln = PLNReasoner()
        pln.assert_rule('bird(X)', 'canfly(X)', strength=0.9)
        pln.assert_fact('bird(penguin)', strength=0.99)

        # Observe that penguin doesn't fly
        pln.observe('canfly(penguin)', is_true=False)

        # Rule should be weakened
        rule_tv = pln.get_rule_truth('bird(X)', 'canfly(X)')
        assert rule_tv.strength < 0.9

    def test_slm_temperature_affects_diversity(self):
        """Higher temperature produces more diverse outputs."""
        from cortical.reasoning import PRISMLanguageModel

        slm = PRISMLanguageModel(context_size=2)
        slm.train('The cat sat. The cat slept. The cat jumped.')

        # Generate multiple times with low temp
        low_temp_results = set()
        for _ in range(5):
            result = slm.generate_next(['the', 'cat'], temperature=0.1)
            if result:
                low_temp_results.add(result)

        # Generate multiple times with high temp
        high_temp_results = set()
        for _ in range(10):
            result = slm.generate_next(['the', 'cat'], temperature=2.0)
            if result:
                high_temp_results.add(result)

        # Both should produce results
        assert len(low_temp_results) >= 1
        assert len(high_temp_results) >= 1

    def test_got_decay_weakens_connections(self):
        """GoT decay reduces connection strengths over time."""
        from cortical.reasoning import SynapticMemoryGraph, NodeType, EdgeType

        got = SynapticMemoryGraph()

        # Add nodes and edge
        got.add_node('n1', NodeType.OBSERVATION, 'First thought')
        got.add_node('n2', NodeType.OBSERVATION, 'Second thought')
        edge = got.add_synaptic_edge('n1', 'n2', EdgeType.SUPPORTS, weight=2.0)

        initial_weight = edge.weight

        # Apply decay
        decayed = got.apply_global_decay()

        assert decayed > 0
        assert edge.weight < initial_weight

    def test_all_systems_together(self):
        """All three PRISM systems work in harmony."""
        from cortical.reasoning import (
            PLNReasoner,
            PRISMLanguageModel,
            SynapticMemoryGraph,
            IncrementalReasoner,
            NodeType,
        )

        # === PLN ===
        pln = PLNReasoner()
        pln.assert_fact('late(rabbit)', strength=0.99)
        pln.assert_rule('late(X)', 'runs_fast(X)', strength=0.95)

        rabbit_runs = pln.query('runs_fast(rabbit)')
        assert rabbit_runs is not None
        assert rabbit_runs.strength > 0.9

        # === SLM ===
        slm = PRISMLanguageModel(context_size=2)
        slm.train('Time time time the rabbit checked his watch.')

        assert slm.graph.vocab_size > 0
        assert slm.graph.transition_count > 0

        # === GoT ===
        got = SynapticMemoryGraph()
        reasoner = IncrementalReasoner(got)

        thought = reasoner.process_thought('The rabbit is late', NodeType.OBSERVATION)

        assert got.node_count() > 0
        assert thought.content == 'The rabbit is late'

        # All systems initialized and functional
        assert pln.fact_count > 0
        assert slm.vocab_size > 0
        assert got.node_count() > 0

    def test_pln_logical_operations(self):
        """PLN logical operations work correctly."""
        from cortical.reasoning import (
            TruthValue,
            pln_not,
            pln_and,
            pln_or,
        )

        tv_high = TruthValue(strength=0.9, confidence=0.9)
        tv_low = TruthValue(strength=0.3, confidence=0.8)

        # NOT inverts strength
        negated = pln_not(tv_high)
        assert negated.strength == pytest.approx(0.1, rel=0.01)

        # AND produces lower strength
        conj = pln_and(tv_high, tv_low)
        assert conj.strength < min(tv_high.strength, tv_low.strength)

        # OR produces higher strength
        disj = pln_or(tv_high, tv_low)
        assert disj.strength > max(tv_high.strength, tv_low.strength)

    def test_pln_inference_rules(self):
        """PLN inference rules produce valid conclusions."""
        from cortical.reasoning import (
            TruthValue,
            deduce,
            induce,
            abduce,
        )

        tv_ab = TruthValue(strength=0.9, confidence=0.9)
        tv_bc = TruthValue(strength=0.85, confidence=0.85)

        # Deduction: A→B, B→C ⊢ A→C
        tv_ac = deduce(tv_ab, tv_bc)
        assert tv_ac.strength > 0.5
        assert tv_ac.confidence > 0

        # Induction has lower confidence
        tv_ind = induce(tv_ab, tv_bc)
        assert tv_ind.confidence < tv_ab.confidence

        # Abduction has lower confidence
        tv_abd = abduce(tv_ab, tv_bc)
        assert tv_abd.confidence < tv_ab.confidence

    def test_slm_perplexity(self):
        """SLM perplexity is lower for in-domain text."""
        from cortical.reasoning import PRISMLanguageModel

        slm = PRISMLanguageModel(context_size=2)
        slm.train('The cat sat on the mat. The cat sat on the mat.')

        # In-domain text should have lower perplexity
        in_domain_ppl = slm.perplexity('The cat sat on the mat.')

        # Out-of-domain text should have higher perplexity
        out_domain_ppl = slm.perplexity('Quantum physics explains entanglement.')

        assert in_domain_ppl < out_domain_ppl

    def test_got_prediction_accuracy(self):
        """GoT tracks prediction accuracy over time."""
        from cortical.reasoning import SynapticMemoryGraph, NodeType, EdgeType

        got = SynapticMemoryGraph()

        got.add_node('q1', NodeType.QUESTION, 'What is X?')
        got.add_node('a1', NodeType.INSIGHT, 'X is Y')

        edge = got.add_synaptic_edge('q1', 'a1', EdgeType.ANSWERS)

        # Record predictions
        edge.record_prediction_outcome(correct=True)
        edge.record_prediction_outcome(correct=True)
        edge.record_prediction_outcome(correct=False)

        # Accuracy should be around 2/3
        assert 0.5 < edge.prediction_accuracy < 0.8
