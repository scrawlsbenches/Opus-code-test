#!/usr/bin/env python3
"""
Voting Aggregator

Combines predictions from multiple micro-experts using confidence-weighted voting.

Similar to how cortical columns in the brain vote to reach consensus,
this aggregator merges predictions from multiple experts, weighting by
both prediction confidence and expert historical accuracy.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from .micro_expert import ExpertPrediction, AggregatedPrediction
except ImportError:
    from micro_expert import ExpertPrediction, AggregatedPrediction


@dataclass
class AggregationConfig:
    """
    Configuration for prediction aggregation.

    Attributes:
        top_n: Maximum number of items to return
        min_confidence: Minimum confidence threshold for items
        expert_weights: Optional weights per expert (expert_id -> weight)
        use_calibrated: Whether to use calibrated confidences
        disagreement_penalty: Reduce confidence when experts disagree (0-1)
    """
    top_n: int = 10
    min_confidence: float = 0.0
    expert_weights: Optional[Dict[str, float]] = None
    use_calibrated: bool = True
    disagreement_penalty: float = 0.0


class VotingAggregator:
    """
    Aggregates predictions from multiple experts using weighted voting.

    The aggregator:
    1. Collects predictions from multiple experts
    2. Weights each prediction by confidence × expert_weight
    3. Combines votes for the same item
    4. Calculates disagreement metrics
    5. Returns ranked aggregated predictions
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Initialize aggregator.

        Args:
            config: Aggregation configuration (uses defaults if None)
        """
        self.config = config or AggregationConfig()

    def aggregate(
        self,
        predictions: List[ExpertPrediction],
        config: Optional[AggregationConfig] = None
    ) -> AggregatedPrediction:
        """
        Aggregate predictions from multiple experts.

        Uses confidence-weighted voting where each expert's vote is
        scaled by their confidence and their historical accuracy weight.

        Args:
            predictions: List of predictions from different experts
            config: Optional config override (uses self.config if None)

        Returns:
            AggregatedPrediction with ranked items and consensus metrics
        """
        if not predictions:
            return AggregatedPrediction(
                items=[],
                contributing_experts=[],
                disagreement_score=0.0,
                confidence=0.0,
                metadata={'num_experts': 0}
            )

        cfg = config or self.config

        # Initialize scoring structures
        item_scores: Dict[str, float] = defaultdict(float)
        item_voters: Dict[str, List[str]] = defaultdict(list)
        item_confidences: Dict[str, List[float]] = defaultdict(list)

        # Collect votes from each expert
        for pred in predictions:
            expert_weight = 1.0
            if cfg.expert_weights:
                expert_weight = cfg.expert_weights.get(pred.expert_id, 1.0)

            for item, confidence in pred.items:
                # Weight vote by confidence × expert weight
                vote = confidence * expert_weight
                item_scores[item] += vote
                item_voters[item].append(pred.expert_id)
                item_confidences[item].append(confidence)

        # Calculate disagreement score
        disagreement = self._calculate_disagreement(predictions)

        # Sort items by aggregated score
        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])

        # Apply disagreement penalty if configured
        if cfg.disagreement_penalty > 0:
            sorted_items = [
                (item, score * (1.0 - disagreement * cfg.disagreement_penalty))
                for item, score in sorted_items
            ]

        # Filter by minimum confidence
        if cfg.min_confidence > 0:
            sorted_items = [
                (item, score) for item, score in sorted_items
                if score >= cfg.min_confidence
            ]

        # Limit to top_n
        final_items = sorted_items[:cfg.top_n]

        # Calculate overall confidence
        # Higher when top item has strong consensus
        max_score = final_items[0][1] if final_items else 0.0
        num_experts = len(predictions)
        overall_confidence = max_score / num_experts if num_experts > 0 else 0.0

        # Reduce confidence if high disagreement
        if disagreement > 0.5:
            overall_confidence *= (1.0 - (disagreement - 0.5))

        # Clamp to [0, 1]
        overall_confidence = max(0.0, min(1.0, overall_confidence))

        return AggregatedPrediction(
            items=final_items,
            contributing_experts=list(set(p.expert_id for p in predictions)),
            disagreement_score=disagreement,
            confidence=overall_confidence,
            metadata={
                'num_experts': num_experts,
                'num_unique_items': len(item_scores),
                'top_item_voters': item_voters[final_items[0][0]] if final_items else [],
                'config': {
                    'top_n': cfg.top_n,
                    'min_confidence': cfg.min_confidence,
                    'disagreement_penalty': cfg.disagreement_penalty
                }
            }
        )

    def _calculate_disagreement(self, predictions: List[ExpertPrediction]) -> float:
        """
        Calculate disagreement score across expert predictions.

        Uses normalized entropy of item rankings. High entropy (disagreement)
        means experts predicted very different items.

        Args:
            predictions: List of expert predictions

        Returns:
            Disagreement score (0 = perfect consensus, 1 = maximum disagreement)
        """
        if len(predictions) <= 1:
            return 0.0

        # Collect all items predicted by any expert
        all_items: Dict[str, int] = defaultdict(int)
        for pred in predictions:
            for item, _ in pred.items:
                all_items[item] += 1

        if not all_items:
            return 0.0

        # Calculate entropy of item distribution
        num_experts = len(predictions)
        entropy = 0.0

        for item, count in all_items.items():
            # Probability this item was predicted
            p = count / num_experts

            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        # Max entropy when each expert predicts completely different items
        max_entropy = math.log2(num_experts) if num_experts > 1 else 1.0

        disagreement = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, disagreement)

    def _calculate_overlap(self, predictions: List[ExpertPrediction]) -> float:
        """
        Calculate overlap in top predictions across experts.

        Args:
            predictions: List of expert predictions

        Returns:
            Overlap score (0 = no overlap, 1 = perfect overlap)
        """
        if len(predictions) <= 1:
            return 1.0

        # Get top item from each expert
        top_items = set()
        for pred in predictions:
            if pred.items:
                top_items.add(pred.items[0][0])

        # Overlap is inverse of unique top items
        max_unique = len(predictions)
        actual_unique = len(top_items)

        overlap = 1.0 - (actual_unique - 1) / max(1, max_unique - 1)

        return overlap

    def merge_by_rank(
        self,
        predictions: List[ExpertPrediction],
        top_n: int = 10
    ) -> List[str]:
        """
        Merge predictions using Borda count (rank-based voting).

        Instead of using confidence scores, this uses rank positions.
        An item at rank 1 gets N points, rank 2 gets N-1 points, etc.

        Useful when expert confidences aren't calibrated or comparable.

        Args:
            predictions: List of expert predictions
            top_n: Number of items to return

        Returns:
            List of items ranked by Borda count
        """
        item_points: Dict[str, float] = defaultdict(float)

        for pred in predictions:
            # Assign points based on rank (1-indexed)
            max_rank = len(pred.items)

            for rank, (item, _) in enumerate(pred.items, start=1):
                # Points = max_rank - rank + 1
                # Rank 1 gets max_rank points, rank 2 gets max_rank-1, etc.
                points = max_rank - rank + 1
                item_points[item] += points

        # Sort by points
        sorted_items = sorted(item_points.items(), key=lambda x: -x[1])

        return [item for item, _ in sorted_items[:top_n]]

    def weighted_average_confidence(
        self,
        predictions: List[ExpertPrediction],
        item: str
    ) -> float:
        """
        Calculate weighted average confidence for a specific item.

        Args:
            predictions: List of expert predictions
            item: Item to calculate confidence for

        Returns:
            Weighted average confidence (0-1)
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for pred in predictions:
            expert_weight = 1.0
            if self.config.expert_weights:
                expert_weight = self.config.expert_weights.get(pred.expert_id, 1.0)

            # Find item in this expert's predictions
            for pred_item, confidence in pred.items:
                if pred_item == item:
                    weighted_sum += confidence * expert_weight
                    total_weight += expert_weight
                    break

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight
