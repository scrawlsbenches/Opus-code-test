#!/usr/bin/env python3
"""
Credit-Weighted Expert Router

Routes predictions to experts and weights their outputs based on credit balance.
Uses softmax with temperature control and enforces minimum weight floors.

Classes:
    ExpertWeight: Weight information for a single expert
    CreditRouter: Routes queries and aggregates expert predictions by credit
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from credit_account import CreditLedger
from micro_expert import ExpertPrediction, AggregatedPrediction


@dataclass
class ExpertWeight:
    """
    Weight information for a single expert.

    Attributes:
        expert_id: ID of the expert
        raw_weight: Raw weight based on credit balance (before normalization)
        normalized_weight: Weight normalized to sum to 1.0 across all experts
        confidence_boost: Additional factor for high-credit experts (1.0+)
    """
    expert_id: str
    raw_weight: float
    normalized_weight: float
    confidence_boost: float


class CreditRouter:
    """
    Routes predictions to experts and weights outputs based on credit.

    Uses softmax with temperature to convert credit balances to weights,
    applies a minimum weight floor to ensure all experts contribute,
    and provides confidence boosts to high-performing experts.

    Attributes:
        ledger: Credit ledger tracking expert value
        min_weight: Minimum weight floor (default 0.1)
        temperature: Softmax temperature for weight sharpness (default 1.0)
        routing_history: History of routing decisions for stats
    """

    def __init__(
        self,
        ledger: CreditLedger,
        min_weight: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize credit router.

        Args:
            ledger: Credit ledger with expert balances
            min_weight: Minimum weight floor (0.0-1.0, default 0.1)
            temperature: Temperature for softmax (>0, default 1.0)

        Raises:
            ValueError: If min_weight not in [0, 1] or temperature <= 0
        """
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.ledger = ledger
        self.min_weight = min_weight
        self.temperature = temperature
        self.routing_history: List[Dict[str, Any]] = []

    def compute_weights(
        self,
        expert_ids: List[str]
    ) -> Dict[str, ExpertWeight]:
        """
        Compute weights for given experts based on credit balance.

        Uses softmax with temperature:
        1. Get credit balances for each expert
        2. Apply softmax: exp((balance - max_balance) / temperature)
        3. Normalize to sum to 1.0
        4. Apply min_weight floor
        5. Renormalize after floor

        Args:
            expert_ids: List of expert IDs to compute weights for

        Returns:
            Dictionary mapping expert_id to ExpertWeight
        """
        if not expert_ids:
            return {}

        # Get balances (use initial balance for unknown experts)
        balances = {}
        for expert_id in expert_ids:
            account = self.ledger.get_or_create_account(expert_id)
            balances[expert_id] = account.balance

        # Softmax with temperature (numerically stable)
        max_balance = max(balances.values())
        exp_scores = {
            exp_id: math.exp((balance - max_balance) / self.temperature)
            for exp_id, balance in balances.items()
        }
        total_exp = sum(exp_scores.values())

        # Normalize
        raw_weights = {
            exp_id: score / total_exp
            for exp_id, score in exp_scores.items()
        }

        # Apply min_weight floor
        floored_weights = {
            exp_id: max(self.min_weight, weight)
            for exp_id, weight in raw_weights.items()
        }

        # Renormalize after applying floor
        total_floored = sum(floored_weights.values())
        normalized_weights = {
            exp_id: weight / total_floored
            for exp_id, weight in floored_weights.items()
        }

        # Calculate confidence boost for high-credit experts
        weights = {}
        for expert_id in expert_ids:
            balance = balances[expert_id]

            # Experts with balance > 150 get confidence boost
            # boost = 1.0 + 0.1 * min((balance - 100) / 100, 1.0)
            if balance > 150:
                boost_factor = min((balance - 100) / 100, 1.0)
                boost = 1.0 + 0.1 * boost_factor
            else:
                boost = 1.0

            weights[expert_id] = ExpertWeight(
                expert_id=expert_id,
                raw_weight=raw_weights[expert_id],
                normalized_weight=normalized_weights[expert_id],
                confidence_boost=boost
            )

        return weights

    def aggregate_predictions(
        self,
        predictions: Dict[str, ExpertPrediction]
    ) -> AggregatedPrediction:
        """
        Aggregate predictions weighted by expert credit.

        Combines predictions from multiple experts:
        1. Compute weights based on credit
        2. Weight each expert's item scores
        3. Aggregate scores across experts
        4. Apply confidence boost to high-credit experts
        5. Sort by final weighted score

        Args:
            predictions: Dict mapping expert_id to ExpertPrediction

        Returns:
            AggregatedPrediction with weighted consensus
        """
        if not predictions:
            return AggregatedPrediction(
                items=[],
                contributing_experts=[],
                disagreement_score=0.0,
                confidence=0.0,
                metadata={'method': 'credit_weighted'}
            )

        # Compute weights
        expert_ids = list(predictions.keys())
        weights = self.compute_weights(expert_ids)

        # Aggregate item scores
        item_scores: Dict[str, float] = {}
        item_expert_votes: Dict[str, List[str]] = {}

        for expert_id, prediction in predictions.items():
            weight_info = weights[expert_id]
            expert_weight = weight_info.normalized_weight
            confidence_boost = weight_info.confidence_boost

            for item, confidence in prediction.items:
                # Apply expert weight and confidence boost
                weighted_score = confidence * expert_weight * confidence_boost

                if item not in item_scores:
                    item_scores[item] = 0.0
                    item_expert_votes[item] = []

                item_scores[item] += weighted_score
                item_expert_votes[item].append(expert_id)

        # Sort by weighted score
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: -x[1]
        )

        # Calculate disagreement (variance in expert votes per item)
        if len(predictions) > 1:
            vote_counts = [len(votes) for votes in item_expert_votes.values()]
            if vote_counts:
                avg_votes = sum(vote_counts) / len(vote_counts)
                max_votes = len(predictions)
                # Disagreement is 1 - (avg_votes / max_votes)
                disagreement = 1.0 - (avg_votes / max_votes)
            else:
                disagreement = 0.0
        else:
            disagreement = 0.0

        # Overall confidence is weighted average of top items
        if sorted_items:
            # Take top 3 items or all if fewer
            top_items = sorted_items[:min(3, len(sorted_items))]
            confidence = sum(score for _, score in top_items) / len(top_items)
        else:
            confidence = 0.0

        return AggregatedPrediction(
            items=sorted_items,
            contributing_experts=expert_ids,
            disagreement_score=disagreement,
            confidence=confidence,
            metadata={
                'method': 'credit_weighted',
                'weights': {
                    exp_id: weight.normalized_weight
                    for exp_id, weight in weights.items()
                },
                'boosts': {
                    exp_id: weight.confidence_boost
                    for exp_id, weight in weights.items()
                }
            }
        )

    def select_expert(
        self,
        context: Dict[str, Any],
        available: List[str]
    ) -> str:
        """
        Select best expert for given context.

        Considers both weight (credit balance) and context relevance.
        For now, uses pure credit-based selection. Future versions
        could incorporate context matching.

        Args:
            context: Context dictionary (may include query, error_message, etc.)
            available: List of available expert IDs

        Returns:
            Expert ID of selected expert

        Raises:
            ValueError: If no experts available
        """
        if not available:
            raise ValueError("No experts available for selection")

        # Compute weights
        weights = self.compute_weights(available)

        # Select expert with highest normalized weight
        selected = max(
            weights.items(),
            key=lambda x: x[1].normalized_weight
        )

        # Record routing decision
        self.routing_history.append({
            'timestamp': time.time(),
            'context': context,
            'available': available,
            'selected': selected[0],
            'weights': {
                exp_id: w.normalized_weight
                for exp_id, w in weights.items()
            }
        })

        return selected[0]

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.

        Tracks which experts are being used most frequently
        and their average weights over time.

        Returns:
            Dictionary with routing statistics:
            - total_routings: Total number of routing decisions
            - expert_usage: Count of times each expert was selected
            - average_weights: Average weight per expert
            - recent_decisions: Last 10 routing decisions
        """
        if not self.routing_history:
            return {
                'total_routings': 0,
                'expert_usage': {},
                'average_weights': {},
                'recent_decisions': []
            }

        # Count expert selections
        expert_usage: Dict[str, int] = {}
        expert_weights_sum: Dict[str, float] = {}
        expert_weights_count: Dict[str, int] = {}

        for decision in self.routing_history:
            selected = decision['selected']
            expert_usage[selected] = expert_usage.get(selected, 0) + 1

            # Track weights
            for exp_id, weight in decision['weights'].items():
                if exp_id not in expert_weights_sum:
                    expert_weights_sum[exp_id] = 0.0
                    expert_weights_count[exp_id] = 0
                expert_weights_sum[exp_id] += weight
                expert_weights_count[exp_id] += 1

        # Calculate averages
        average_weights = {
            exp_id: expert_weights_sum[exp_id] / expert_weights_count[exp_id]
            for exp_id in expert_weights_sum
        }

        return {
            'total_routings': len(self.routing_history),
            'expert_usage': expert_usage,
            'average_weights': average_weights,
            'recent_decisions': self.routing_history[-10:]
        }
