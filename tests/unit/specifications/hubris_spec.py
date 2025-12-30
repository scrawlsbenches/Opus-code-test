"""
Unit Specifications for Hubris MoE Components.

These specifications document the precise behavior of the Hubris MoE system.
Each specification is a fact about the system that must remain true.
"""

import pytest
import math


class MicroExpertSpecification:
    """
    Specifications for MicroExpert behavior.
    """

    def spec_expert_has_unique_id(self):
        """
        SPECIFICATION: Each expert gets a unique ID on creation.

        This is load-bearing behavior for expert tracking.
        """
        from cortical.reasoning.hubris import MicroExpert

        expert1 = MicroExpert("test1", "domain")
        expert2 = MicroExpert("test2", "domain")

        assert expert1.id != expert2.id

    def spec_competence_estimate_is_bounded(self):
        """
        SPECIFICATION: Competence estimates are always in [0, 1].

        Prevents invalid confidence values from propagating.
        """
        from cortical.reasoning.hubris import MicroExpert

        expert = MicroExpert("test", "nlp", ["parsing", "generation"])

        # Even extreme queries should give bounded estimates
        assert 0.0 <= expert.estimate_competence("") <= 1.0
        assert 0.0 <= expert.estimate_competence("x" * 10000) <= 1.0
        assert 0.0 <= expert.estimate_competence("parsing") <= 1.0

    def spec_abstention_respects_threshold(self):
        """
        SPECIFICATION: Experts abstain when competence < threshold.

        Prevents low-quality responses from overconfident experts.
        """
        from cortical.reasoning.hubris import MicroExpert

        expert = MicroExpert(
            "nlp_expert",
            "nlp",
            ["parsing"],
            abstention_threshold=0.5
        )

        # Query outside domain should trigger abstention
        response = expert.respond("quantum physics equations")
        assert response.abstained or response.confidence < 0.5

    def spec_competencies_can_be_added_dynamically(self):
        """
        SPECIFICATION: Competencies can be added after creation.

        Enables expert specialization through training.
        """
        from cortical.reasoning.hubris import MicroExpert

        expert = MicroExpert("test", "general", [])
        assert len(expert.competencies) == 0

        expert.add_competency("new_skill")
        assert "new_skill" in expert.competencies

    def spec_response_includes_competencies_used(self):
        """
        SPECIFICATION: Responses track which competencies were used.

        Enables analysis of expert reasoning.
        """
        from cortical.reasoning.hubris import MicroExpert

        expert = MicroExpert("test", "nlp", ["parsing", "generation"])
        response = expert.respond("parsing task")

        # Should have competencies_used attribute
        assert hasattr(response, 'competencies_used')


class CreditLedgerSpecification:
    """
    Specifications for CreditLedger behavior.
    """

    def spec_initial_credit_is_configurable(self):
        """
        SPECIFICATION: Initial credit can be configured.

        Different systems may need different starting credits.
        """
        from cortical.reasoning.hubris import CreditLedger

        ledger1 = CreditLedger(initial_credit=100.0)
        ledger2 = CreditLedger(initial_credit=500.0)

        assert ledger1.get_credit("new_expert") == 100.0
        assert ledger2.get_credit("new_expert") == 500.0

    def spec_correct_predictions_increase_credit(self):
        """
        SPECIFICATION: Correct predictions increase credit.

        Accuracy should be rewarded.
        """
        from cortical.reasoning.hubris import CreditLedger

        ledger = CreditLedger(initial_credit=100.0)

        initial = ledger.get_credit("expert")
        ledger.record_prediction("expert", 0.8, correct=True)

        assert ledger.get_credit("expert") > initial

    def spec_wrong_predictions_decrease_credit(self):
        """
        SPECIFICATION: Wrong predictions decrease credit.

        Mistakes should be penalized.
        """
        from cortical.reasoning.hubris import CreditLedger

        ledger = CreditLedger(initial_credit=100.0)

        ledger.record_prediction("expert", 0.5, correct=True)  # Build some credit
        current = ledger.get_credit("expert")
        ledger.record_prediction("expert", 0.9, correct=False)  # High conf wrong

        assert ledger.get_credit("expert") < current

    def spec_ece_is_zero_for_perfectly_calibrated(self):
        """
        SPECIFICATION: ECE approaches zero for perfectly calibrated experts.

        Perfect calibration means confidence = accuracy.
        """
        from cortical.reasoning.hubris import CreditLedger

        ledger = CreditLedger()

        # 80% confident, 80% correct (8 out of 10)
        for i in range(10):
            ledger.record_prediction("calibrated", 0.8, correct=(i < 8))

        ece = ledger.compute_ece("calibrated")
        # Should be close to zero (within binning noise)
        assert ece < 0.15

    def spec_ece_is_high_for_overconfident(self):
        """
        SPECIFICATION: ECE is high for overconfident experts.

        Overconfidence should be detected.
        """
        from cortical.reasoning.hubris import CreditLedger

        ledger = CreditLedger()

        # 90% confident, only 50% correct (overconfident)
        for i in range(10):
            ledger.record_prediction("overconfident", 0.9, correct=(i < 5))

        ece = ledger.compute_ece("overconfident")
        # Should be substantial (around 0.4)
        assert ece > 0.2

    def spec_credit_never_goes_negative(self):
        """
        SPECIFICATION: Credit is floored at zero.

        Negative credit would break many calculations.
        """
        from cortical.reasoning.hubris import CreditLedger

        ledger = CreditLedger(initial_credit=10.0)

        # Many wrong predictions
        for _ in range(100):
            ledger.record_prediction("expert", 0.99, correct=False)

        assert ledger.get_credit("expert") >= 0.0


class ValueSignalSpecification:
    """
    Specifications for ValueSignal behavior.
    """

    def spec_initial_value_is_configurable(self):
        """
        SPECIFICATION: Initial value can be configured.
        """
        from cortical.reasoning.hubris import ValueSignal

        signal = ValueSignal(initial_value=0.5)
        assert signal.get_value("unknown_action") == 0.5

    def spec_positive_reward_increases_value(self):
        """
        SPECIFICATION: Positive rewards increase action value.
        """
        from cortical.reasoning.hubris import ValueSignal

        signal = ValueSignal()
        initial = signal.get_value("action")
        signal.reward("action", 1.0)

        assert signal.get_value("action") > initial

    def spec_negative_reward_decreases_value(self):
        """
        SPECIFICATION: Negative rewards decrease action value.
        """
        from cortical.reasoning.hubris import ValueSignal

        signal = ValueSignal()
        # Build some value first
        signal.reward("action", 1.0)
        current = signal.get_value("action")

        signal.reward("action", -1.0)
        assert signal.get_value("action") < current

    def spec_learning_rate_controls_update_speed(self):
        """
        SPECIFICATION: Higher learning rate means faster updates.
        """
        from cortical.reasoning.hubris import ValueSignal

        slow = ValueSignal(learning_rate=0.01)
        fast = ValueSignal(learning_rate=0.5)

        slow.reward("action", 1.0)
        fast.reward("action", 1.0)

        # Fast learner should have larger change
        assert abs(fast.get_value("action")) > abs(slow.get_value("action"))

    def spec_select_action_chooses_from_candidates(self):
        """
        SPECIFICATION: select_action returns one of the candidates.
        """
        from cortical.reasoning.hubris import ValueSignal

        signal = ValueSignal()
        candidates = ["a", "b", "c"]

        for _ in range(10):
            selected = signal.select_action(candidates)
            assert selected in candidates


class StakingManagerSpecification:
    """
    Specifications for StakingManager behavior.
    """

    def spec_initial_stake_is_configurable(self):
        """
        SPECIFICATION: Initial stake can be configured.
        """
        from cortical.reasoning.hubris import StakingManager

        manager = StakingManager(initial_stake=200.0)
        assert manager.get_balance("new_expert") == 200.0

    def spec_stake_reduces_balance(self):
        """
        SPECIFICATION: Placing a stake reduces available balance.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        manager = StakingManager(initial_stake=100.0)
        expert = MicroExpert("test", "domain")

        initial = manager.get_balance(expert)
        manager.place_stake(expert, confidence=0.8, amount=30.0)

        assert manager.get_balance(expert) < initial

    def spec_correct_prediction_returns_stake_plus_reward(self):
        """
        SPECIFICATION: Correct predictions return stake with bonus.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        manager = StakingManager(initial_stake=100.0, reward_multiplier=0.5)
        expert = MicroExpert("test", "domain")

        stake_id = manager.place_stake(expert, confidence=0.8, amount=30.0)
        after_stake = manager.get_balance(expert)

        manager.resolve_stake(expert, stake_id, correct=True)
        after_resolve = manager.get_balance(expert)

        # Should get stake back plus reward
        assert after_resolve > after_stake + 30.0

    def spec_wrong_prediction_slashes_stake(self):
        """
        SPECIFICATION: Wrong predictions result in stake loss.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        manager = StakingManager(initial_stake=100.0)
        expert = MicroExpert("test", "domain")

        stake_id = manager.place_stake(expert, confidence=0.9, amount=40.0)
        after_stake = manager.get_balance(expert)

        manager.resolve_stake(expert, stake_id, correct=False)
        after_resolve = manager.get_balance(expert)

        # Should not get full stake back
        assert after_resolve < after_stake + 40.0

    def spec_cannot_stake_more_than_balance(self):
        """
        SPECIFICATION: Cannot stake more than available balance.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        manager = StakingManager(initial_stake=50.0, max_stake_fraction=1.0)
        expert = MicroExpert("test", "domain")

        with pytest.raises(ValueError):
            manager.place_stake(expert, confidence=0.9, amount=100.0)


class HubrisMoESpecification:
    """
    Specifications for HubrisMoE orchestrator behavior.
    """

    def spec_experts_can_be_registered_and_retrieved(self):
        """
        SPECIFICATION: Experts can be registered and retrieved by name.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        moe = HubrisMoE()
        expert = MicroExpert("test_expert", "test", ["skill"])

        moe.register_expert(expert)
        retrieved = moe.get_expert("test_expert")

        assert retrieved is not None
        assert retrieved.name == "test_expert"

    def spec_select_experts_returns_competent_ones(self):
        """
        SPECIFICATION: select_experts returns experts above threshold.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        moe = HubrisMoE(selection_threshold=0.3)

        # NLP expert
        nlp = MicroExpert("nlp", "nlp", ["parsing", "generation"])
        moe.register_expert(nlp)

        # CV expert (won't match NLP query)
        cv = MicroExpert("cv", "cv", ["detection", "classification"])
        moe.register_expert(cv)

        selected = moe.select_experts("parsing text")
        selected_names = [e.name for e in selected]

        # NLP expert should be selected
        assert "nlp" in selected_names or len(selected) > 0

    def spec_query_returns_result_with_metadata(self):
        """
        SPECIFICATION: Queries return results with metadata.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        moe = HubrisMoE()
        moe.register_expert(MicroExpert("test", "test", ["skill"]))

        result = moe.query("test query")

        assert hasattr(result, 'confidence')
        assert hasattr(result, 'contributing_experts')
        assert hasattr(result, 'processing_time_ms')

    def spec_empty_expert_pool_returns_empty_result(self):
        """
        SPECIFICATION: Empty expert pool returns empty result gracefully.
        """
        from cortical.reasoning.hubris import HubrisMoE

        moe = HubrisMoE()  # No experts
        result = moe.query("any query")

        assert result.confidence == 0.0
        assert result.contributing_experts == []


# Pytest test class
class TestHubrisSpecifications:
    """Pytest wrapper for Hubris specifications."""

    def test_expert_unique_id(self):
        spec = MicroExpertSpecification()
        spec.spec_expert_has_unique_id()

    def test_competence_bounded(self):
        spec = MicroExpertSpecification()
        spec.spec_competence_estimate_is_bounded()

    def test_abstention_threshold(self):
        spec = MicroExpertSpecification()
        spec.spec_abstention_respects_threshold()

    def test_credit_increases_on_correct(self):
        spec = CreditLedgerSpecification()
        spec.spec_correct_predictions_increase_credit()

    def test_credit_decreases_on_wrong(self):
        spec = CreditLedgerSpecification()
        spec.spec_wrong_predictions_decrease_credit()

    def test_ece_calibration(self):
        spec = CreditLedgerSpecification()
        spec.spec_ece_is_zero_for_perfectly_calibrated()

    def test_value_signal_learning(self):
        spec = ValueSignalSpecification()
        spec.spec_positive_reward_increases_value()
        spec.spec_negative_reward_decreases_value()

    def test_staking_balance(self):
        spec = StakingManagerSpecification()
        spec.spec_stake_reduces_balance()

    def test_moe_query(self):
        spec = HubrisMoESpecification()
        spec.spec_query_returns_result_with_metadata()
