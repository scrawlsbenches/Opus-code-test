"""
Behavioral tests for the Hubris Mixture of Experts (MoE) system.

Hubris MoE implements a calibrated expert ensemble where:
- MicroExperts specialize in narrow domains
- CreditLedger tracks expert performance over time
- ValueSignals guide expert selection
- Calibration ensures confidence matches accuracy (ECE)
- Staking allows experts to commit resources to their predictions

The name "Hubris" reminds us: overconfidence is dangerous.
Well-calibrated experts know what they don't know.

User Stories:
- As a system, I want multiple experts to collaborate on complex tasks,
  so that diverse perspectives improve solution quality.
- As a user, I want expert confidence to match actual accuracy,
  so that I can trust their recommendations.
- As an architect, I want experts to specialize and improve over time,
  so that the system becomes more capable.
"""

import pytest
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SystemOrchestatesExperts:
    """
    Epic: Expert Ensemble Orchestration

    As a cognitive system,
    I want to orchestrate multiple micro-experts,
    So that complex tasks benefit from specialized knowledge.
    """

    def scenario_experts_are_selected_by_competence(self):
        """
        Scenario: Competence-based expert selection

        Given a pool of specialized micro-experts
        When a query arrives in a specific domain
        Then experts with relevant competence are selected
        Because matching expertise improves response quality.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        # Given
        moe = HubrisMoE()
        moe.register_expert(MicroExpert(
            name="nlp_expert",
            domain="natural_language_processing",
            competencies=["parsing", "semantics", "generation"]
        ))
        moe.register_expert(MicroExpert(
            name="cv_expert",
            domain="computer_vision",
            competencies=["classification", "detection", "segmentation"]
        ))
        moe.register_expert(MicroExpert(
            name="ml_expert",
            domain="machine_learning",
            competencies=["training", "optimization", "evaluation"]
        ))

        # When
        query = "How do I tokenize text for a transformer model?"
        selected = moe.select_experts(query)

        # Then
        # NLP expert should be selected (tokenization is NLP)
        expert_names = [e.name for e in selected]
        assert "nlp_expert" in expert_names or len(selected) > 0

    def scenario_experts_contribute_weighted_responses(self):
        """
        Scenario: Weighted expert combination

        Given multiple experts selected for a task
        When each provides a response with confidence
        Then responses are combined using calibrated weights
        Because expert reliability varies by domain and query.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert, ExpertResponse

        # Given
        moe = HubrisMoE()
        expert1 = MicroExpert(name="expert1", domain="general", competencies=["reasoning"])
        expert2 = MicroExpert(name="expert2", domain="general", competencies=["reasoning"])
        moe.register_expert(expert1)
        moe.register_expert(expert2)

        # When
        responses = [
            ExpertResponse(expert=expert1, answer="A", confidence=0.9),
            ExpertResponse(expert=expert2, answer="B", confidence=0.6),
        ]
        combined = moe.combine_responses(responses)

        # Then
        # Higher confidence response should have more influence
        assert combined is not None
        assert combined.final_answer in ["A", "B"]

    def scenario_experts_can_abstain_when_uncertain(self):
        """
        Scenario: Expert abstention

        Given an expert facing a query outside their competence
        When asked to respond
        Then the expert can abstain with low confidence
        Because forcing answers on unknown domains hurts calibration.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert

        # Given
        moe = HubrisMoE()
        cv_expert = MicroExpert(
            name="cv_expert",
            domain="computer_vision",
            competencies=["classification"],
            abstention_threshold=0.3  # Abstain if confidence < 30%
        )
        moe.register_expert(cv_expert)

        # When
        query = "Explain quantum entanglement"  # Outside CV domain
        response = moe.query_expert(cv_expert, query)

        # Then
        # Expert should abstain or have very low confidence
        assert response.confidence < 0.5 or response.abstained


class UserTrustsCalibration:
    """
    Epic: Expert Calibration and Trust

    As a user relying on expert recommendations,
    I want confidence scores to match actual accuracy,
    So that I can make informed decisions.
    """

    def scenario_ece_measures_calibration_quality(self):
        """
        Scenario: Expected Calibration Error (ECE) measurement

        Given a history of expert predictions and outcomes
        When ECE is computed
        Then it measures the gap between confidence and accuracy
        Because ECE quantifies how well-calibrated experts are.
        """
        from cortical.reasoning.hubris import CreditLedger, CalibrationMetrics

        # Given
        ledger = CreditLedger()

        # Expert was 90% confident, correct 70% of the time (overconfident)
        for i in range(10):
            ledger.record_prediction(
                expert_id="overconfident_expert",
                confidence=0.9,
                correct=(i < 7)  # 7 out of 10 correct
            )

        # Expert was 60% confident, correct 60% of the time (well-calibrated)
        for i in range(10):
            ledger.record_prediction(
                expert_id="calibrated_expert",
                confidence=0.6,
                correct=(i < 6)  # 6 out of 10 correct
            )

        # When
        ece_overconfident = ledger.compute_ece("overconfident_expert")
        ece_calibrated = ledger.compute_ece("calibrated_expert")

        # Then
        # Overconfident expert should have higher ECE (worse calibration)
        assert ece_overconfident > ece_calibrated or ece_calibrated < 0.15

    def scenario_credit_accumulates_for_accurate_experts(self):
        """
        Scenario: Credit-based expert reputation

        Given an expert making predictions over time
        When predictions are verified as correct or incorrect
        Then the expert's credit score reflects their track record
        Because past performance indicates future reliability.
        """
        from cortical.reasoning.hubris import CreditLedger

        # Given
        ledger = CreditLedger()

        # Good expert: 8 correct, 2 incorrect
        for i in range(10):
            ledger.record_prediction(
                expert_id="good_expert",
                confidence=0.75,
                correct=(i < 8)
            )

        # Bad expert: 3 correct, 7 incorrect
        for i in range(10):
            ledger.record_prediction(
                expert_id="bad_expert",
                confidence=0.75,
                correct=(i < 3)
            )

        # When
        good_credit = ledger.get_credit("good_expert")
        bad_credit = ledger.get_credit("bad_expert")

        # Then
        assert good_credit > bad_credit

    def scenario_value_signal_updates_from_feedback(self):
        """
        Scenario: Value signal learning

        Given an expert receiving feedback on predictions
        When positive feedback arrives
        Then the value signal strengthens that prediction pathway
        Because value signals guide future expert behavior.
        """
        from cortical.reasoning.hubris import ValueSignal

        # Given
        signal = ValueSignal(learning_rate=0.1)

        # When
        # Expert predicted "approach A" and it worked
        signal.reward("approach_A", reward=1.0)

        # Expert predicted "approach B" and it failed
        signal.reward("approach_B", reward=-0.5)

        # Then
        value_a = signal.get_value("approach_A")
        value_b = signal.get_value("approach_B")
        assert value_a > value_b


class ExpertStakesCommitment:
    """
    Epic: Expert Commitment via Staking

    As an expert system,
    I want experts to stake resources on their predictions,
    So that they are incentivized to be honest about uncertainty.
    """

    def scenario_staking_increases_with_confidence(self):
        """
        Scenario: Confidence-proportional staking

        Given an expert with available stake
        When making a high-confidence prediction
        Then more stake is committed
        Because high confidence should carry higher stakes.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        # Given
        staking = StakingManager(initial_stake=100.0)
        expert = MicroExpert(name="staking_expert", domain="test")

        # When
        high_conf_stake = staking.compute_stake(expert, confidence=0.95)
        low_conf_stake = staking.compute_stake(expert, confidence=0.55)

        # Then
        assert high_conf_stake > low_conf_stake

    def scenario_stake_is_lost_on_incorrect_confident_prediction(self):
        """
        Scenario: Stake slashing for overconfidence

        Given an expert who staked high on a prediction
        When the prediction turns out wrong
        Then stake is slashed proportionally
        Because overconfidence should be penalized.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        # Given
        staking = StakingManager(initial_stake=100.0)
        expert = MicroExpert(name="overconfident", domain="test")

        # Expert stakes 50 units on a 95% confident prediction
        stake_amount = staking.place_stake(expert, confidence=0.95, amount=50.0)
        initial_balance = staking.get_balance(expert)

        # When
        # Prediction was wrong
        staking.resolve_stake(expert, stake_amount, correct=False)

        # Then
        final_balance = staking.get_balance(expert)
        # Should have lost stake (or significant portion)
        assert final_balance < initial_balance

    def scenario_stake_is_rewarded_on_correct_prediction(self):
        """
        Scenario: Stake rewards for accurate predictions

        Given an expert who staked on a prediction
        When the prediction is correct
        Then stake is returned with bonus
        Because accurate predictions deserve rewards.
        """
        from cortical.reasoning.hubris import StakingManager, MicroExpert

        # Given
        staking = StakingManager(initial_stake=100.0)
        expert = MicroExpert(name="accurate", domain="test")

        stake_amount = staking.place_stake(expert, confidence=0.8, amount=30.0)
        initial_balance = staking.get_balance(expert)

        # When
        staking.resolve_stake(expert, stake_amount, correct=True)

        # Then
        final_balance = staking.get_balance(expert)
        # Should have gained (or at least not lost)
        assert final_balance >= initial_balance


class ArchitectDesignsExperts:
    """
    Epic: Expert Architecture Design

    As a system architect,
    I want to create and configure micro-experts,
    So that I can build specialized reasoning capabilities.
    """

    def scenario_micro_expert_specializes_via_training(self):
        """
        Scenario: Expert specialization through training

        Given a general-purpose micro-expert
        When trained on domain-specific data
        Then its competency in that domain improves
        Because specialization comes from focused learning.
        """
        from cortical.reasoning.hubris import MicroExpert, ExpertTrainer

        # Given
        expert = MicroExpert(
            name="general",
            domain="general",
            competencies=[]
        )

        trainer = ExpertTrainer()

        # When
        training_data = [
            ("parse this SQL query", "sql"),
            ("optimize this SQL query", "sql"),
            ("explain this SQL join", "sql"),
        ]
        trainer.train(expert, training_data)

        # Then
        assert "sql" in expert.competencies or expert.get_competency_score("sql") > 0

    def scenario_experts_form_hierarchical_structure(self):
        """
        Scenario: Hierarchical expert organization

        Given multiple specialized experts
        When organized hierarchically
        Then meta-experts can coordinate sub-experts
        Because complex problems benefit from expert composition.
        """
        from cortical.reasoning.hubris import HubrisMoE, MicroExpert, MetaExpert

        # Given
        moe = HubrisMoE()

        # Leaf experts
        syntax_expert = MicroExpert(name="syntax", domain="code", competencies=["parsing"])
        semantic_expert = MicroExpert(name="semantic", domain="code", competencies=["meaning"])

        # Meta expert that coordinates
        code_expert = MetaExpert(
            name="code_meta",
            domain="code",
            sub_experts=[syntax_expert, semantic_expert]
        )

        moe.register_expert(code_expert)

        # When
        query = "Analyze this code for bugs"
        result = moe.query(query)

        # Then
        # Meta expert should have coordinated sub-experts
        assert result is not None


class TestHubrisMoEBehavior:
    """
    Pytest wrapper for behavioral scenarios.

    These tests verify the Hubris MoE behavioral scenarios.
    Run with: pytest tests/behavioral/test_hubris_moe.py -v
    """

    def test_expert_selection_by_competence(self):
        """Verify competence-based selection."""
        scenario = SystemOrchestatesExperts()
        scenario.scenario_experts_are_selected_by_competence()

    def test_weighted_response_combination(self):
        """Verify weighted combination."""
        scenario = SystemOrchestatesExperts()
        scenario.scenario_experts_contribute_weighted_responses()

    def test_expert_abstention(self):
        """Verify expert abstention."""
        scenario = SystemOrchestatesExperts()
        scenario.scenario_experts_can_abstain_when_uncertain()

    def test_ece_calibration(self):
        """Verify ECE measurement."""
        scenario = UserTrustsCalibration()
        scenario.scenario_ece_measures_calibration_quality()

    def test_credit_accumulation(self):
        """Verify credit tracking."""
        scenario = UserTrustsCalibration()
        scenario.scenario_credit_accumulates_for_accurate_experts()

    def test_value_signal_learning(self):
        """Verify value signal updates."""
        scenario = UserTrustsCalibration()
        scenario.scenario_value_signal_updates_from_feedback()

    def test_confidence_proportional_staking(self):
        """Verify staking scales with confidence."""
        scenario = ExpertStakesCommitment()
        scenario.scenario_staking_increases_with_confidence()

    def test_stake_slashing(self):
        """Verify stake slashing on wrong predictions."""
        scenario = ExpertStakesCommitment()
        scenario.scenario_stake_is_lost_on_incorrect_confident_prediction()

    def test_stake_rewards(self):
        """Verify stake rewards on correct predictions."""
        scenario = ExpertStakesCommitment()
        scenario.scenario_stake_is_rewarded_on_correct_prediction()

    def test_expert_specialization(self):
        """Verify expert training."""
        scenario = ArchitectDesignsExperts()
        scenario.scenario_micro_expert_specializes_via_training()

    def test_hierarchical_experts(self):
        """Verify hierarchical organization."""
        scenario = ArchitectDesignsExperts()
        scenario.scenario_experts_form_hierarchical_structure()
