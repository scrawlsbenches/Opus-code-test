"""
CreditLedger: Expert Performance Tracking with Calibration Metrics.

The CreditLedger tracks expert predictions over time, computing:
- Credit scores based on prediction accuracy
- Expected Calibration Error (ECE) for confidence assessment
- Performance trends and reliability metrics

Design Philosophy:
    Trust is earned through consistent performance. An expert's credit
    reflects their track record—accurate predictions build credit,
    while overconfident failures destroy it.

Key Metrics:
    - ECE (Expected Calibration Error): Measures how well confidence
      matches actual accuracy. Perfect calibration = ECE of 0.
    - Credit Score: Accumulated reputation based on prediction history.
    - Brier Score: Measures probabilistic prediction accuracy.

Example:
    >>> ledger = CreditLedger()
    >>> ledger.record_prediction("expert_a", confidence=0.9, correct=True)
    >>> ledger.record_prediction("expert_a", confidence=0.9, correct=False)
    >>> print(f"ECE: {ledger.compute_ece('expert_a'):.3f}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class PredictionRecord:
    """
    A single prediction record.

    Captures the confidence level and whether the prediction was correct,
    enabling calibration analysis.
    """
    expert_id: str
    confidence: float  # Predicted probability of being correct
    correct: bool  # Was the prediction actually correct?
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clamp confidence to valid range
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class CalibrationBin:
    """
    A bin for calibration analysis.

    Groups predictions by confidence level to compute calibration metrics.
    """
    lower_bound: float
    upper_bound: float
    predictions: List[PredictionRecord] = field(default_factory=list)

    def count(self) -> int:
        """Number of predictions in this bin."""
        return len(self.predictions)

    def mean_confidence(self) -> float:
        """Average confidence in this bin."""
        if not self.predictions:
            return (self.lower_bound + self.upper_bound) / 2
        return sum(p.confidence for p in self.predictions) / len(self.predictions)

    def accuracy(self) -> float:
        """Fraction of correct predictions in this bin."""
        if not self.predictions:
            return 0.0
        return sum(1 for p in self.predictions if p.correct) / len(self.predictions)

    def gap(self) -> float:
        """Gap between confidence and accuracy (calibration error)."""
        return abs(self.mean_confidence() - self.accuracy())


class CalibrationMetrics:
    """
    Calibration metrics for expert evaluation.

    Provides methods to compute various calibration metrics including
    ECE, MCE (Maximum Calibration Error), and reliability diagrams.
    """

    @staticmethod
    def compute_ece(
        predictions: List[PredictionRecord],
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE is the weighted average of the gap between confidence and
        accuracy across all bins. Lower is better; 0 is perfect calibration.

        Args:
            predictions: List of prediction records
            n_bins: Number of bins for binning predictions by confidence

        Returns:
            ECE value (0.0 = perfect calibration)
        """
        if not predictions:
            return 0.0

        # Create bins
        bins = [
            CalibrationBin(
                lower_bound=i / n_bins,
                upper_bound=(i + 1) / n_bins
            )
            for i in range(n_bins)
        ]

        # Assign predictions to bins
        for pred in predictions:
            bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
            bins[bin_idx].predictions.append(pred)

        # Compute ECE
        total = len(predictions)
        ece = 0.0
        for bin in bins:
            if bin.count() > 0:
                weight = bin.count() / total
                ece += weight * bin.gap()

        return ece

    @staticmethod
    def compute_mce(
        predictions: List[PredictionRecord],
        n_bins: int = 10
    ) -> float:
        """
        Compute Maximum Calibration Error.

        MCE is the maximum gap between confidence and accuracy
        across all bins. Useful for identifying worst-case calibration.

        Args:
            predictions: List of prediction records
            n_bins: Number of bins

        Returns:
            MCE value
        """
        if not predictions:
            return 0.0

        # Create bins
        bins = [
            CalibrationBin(
                lower_bound=i / n_bins,
                upper_bound=(i + 1) / n_bins
            )
            for i in range(n_bins)
        ]

        for pred in predictions:
            bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
            bins[bin_idx].predictions.append(pred)

        # Find maximum gap
        max_gap = 0.0
        for bin in bins:
            if bin.count() > 0:
                max_gap = max(max_gap, bin.gap())

        return max_gap

    @staticmethod
    def compute_brier_score(predictions: List[PredictionRecord]) -> float:
        """
        Compute Brier Score for probabilistic predictions.

        Brier Score = (1/N) * sum((confidence - outcome)^2)
        Lower is better; 0 is perfect.

        Args:
            predictions: List of prediction records

        Returns:
            Brier score
        """
        if not predictions:
            return 0.0

        total = 0.0
        for pred in predictions:
            outcome = 1.0 if pred.correct else 0.0
            total += (pred.confidence - outcome) ** 2

        return total / len(predictions)

    @staticmethod
    def reliability_diagram(
        predictions: List[PredictionRecord],
        n_bins: int = 10
    ) -> List[Tuple[float, float, int]]:
        """
        Generate data for a reliability diagram.

        Returns (mean_confidence, accuracy, count) for each bin.

        Args:
            predictions: List of prediction records
            n_bins: Number of bins

        Returns:
            List of (confidence, accuracy, count) tuples
        """
        if not predictions:
            return []

        bins = [
            CalibrationBin(
                lower_bound=i / n_bins,
                upper_bound=(i + 1) / n_bins
            )
            for i in range(n_bins)
        ]

        for pred in predictions:
            bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
            bins[bin_idx].predictions.append(pred)

        return [
            (bin.mean_confidence(), bin.accuracy(), bin.count())
            for bin in bins
            if bin.count() > 0
        ]


class CreditLedger:
    """
    Ledger for tracking expert predictions and computing credit scores.

    The CreditLedger maintains a history of all expert predictions,
    enabling long-term performance tracking and calibration analysis.

    Credit Scoring:
        - Correct predictions add credit
        - Incorrect predictions subtract credit
        - Overconfident wrong predictions penalize more
        - Well-calibrated experts maintain stable credit

    Example:
        >>> ledger = CreditLedger()
        >>> ledger.record_prediction("expert_a", 0.9, True)  # High conf, correct
        >>> ledger.record_prediction("expert_a", 0.9, False)  # High conf, wrong
        >>> print(f"Credit: {ledger.get_credit('expert_a'):.2f}")
        >>> print(f"ECE: {ledger.compute_ece('expert_a'):.3f}")
    """

    def __init__(
        self,
        initial_credit: float = 100.0,
        correct_reward_base: float = 10.0,
        incorrect_penalty_base: float = 5.0,
        overconfidence_multiplier: float = 2.0,
    ):
        """
        Initialize the credit ledger.

        Args:
            initial_credit: Starting credit for new experts
            correct_reward_base: Base reward for correct predictions
            incorrect_penalty_base: Base penalty for wrong predictions
            overconfidence_multiplier: Extra penalty for overconfident mistakes
        """
        self.initial_credit = initial_credit
        self.correct_reward_base = correct_reward_base
        self.incorrect_penalty_base = incorrect_penalty_base
        self.overconfidence_multiplier = overconfidence_multiplier

        self._predictions: Dict[str, List[PredictionRecord]] = {}
        self._credits: Dict[str, float] = {}
        self._metrics = CalibrationMetrics()

    def record_prediction(
        self,
        expert_id: str,
        confidence: float,
        correct: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Record a prediction and update credit.

        Args:
            expert_id: Unique identifier for the expert
            confidence: Expert's stated confidence (0.0 to 1.0)
            correct: Whether the prediction was correct
            context: Optional additional context

        Returns:
            Updated credit score
        """
        # Initialize if needed
        if expert_id not in self._predictions:
            self._predictions[expert_id] = []
            self._credits[expert_id] = self.initial_credit

        # Create record
        record = PredictionRecord(
            expert_id=expert_id,
            confidence=confidence,
            correct=correct,
            context=context or {},
        )
        self._predictions[expert_id].append(record)

        # Update credit
        credit_change = self._compute_credit_change(confidence, correct)
        self._credits[expert_id] += credit_change

        # Ensure credit doesn't go negative
        self._credits[expert_id] = max(0.0, self._credits[expert_id])

        return self._credits[expert_id]

    def _compute_credit_change(self, confidence: float, correct: bool) -> float:
        """
        Compute credit change based on confidence and correctness.

        Reward structure:
        - Correct + high confidence = good reward
        - Correct + low confidence = small reward (under-confident)
        - Incorrect + low confidence = small penalty (appropriate caution)
        - Incorrect + high confidence = large penalty (overconfidence)
        """
        if correct:
            # Reward scales with confidence (confident and correct is good)
            return self.correct_reward_base * confidence
        else:
            # Penalty scales with confidence^2 (overconfidence is bad)
            base_penalty = self.incorrect_penalty_base
            # Extra penalty for high confidence mistakes
            overconfidence_penalty = base_penalty * confidence * self.overconfidence_multiplier
            return -overconfidence_penalty

    def get_credit(self, expert_id: str) -> float:
        """Get current credit for an expert."""
        return self._credits.get(expert_id, self.initial_credit)

    def get_predictions(self, expert_id: str) -> List[PredictionRecord]:
        """Get all predictions for an expert."""
        return self._predictions.get(expert_id, [])

    def compute_ece(self, expert_id: str, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error for an expert.

        Args:
            expert_id: Expert to evaluate
            n_bins: Number of calibration bins

        Returns:
            ECE value (0.0 = perfect calibration)
        """
        predictions = self._predictions.get(expert_id, [])
        return self._metrics.compute_ece(predictions, n_bins)

    def compute_mce(self, expert_id: str, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error for an expert."""
        predictions = self._predictions.get(expert_id, [])
        return self._metrics.compute_mce(predictions, n_bins)

    def compute_brier_score(self, expert_id: str) -> float:
        """Compute Brier Score for an expert."""
        predictions = self._predictions.get(expert_id, [])
        return self._metrics.compute_brier_score(predictions)

    def get_reliability_diagram(
        self,
        expert_id: str,
        n_bins: int = 10
    ) -> List[Tuple[float, float, int]]:
        """Get reliability diagram data for an expert."""
        predictions = self._predictions.get(expert_id, [])
        return self._metrics.reliability_diagram(predictions, n_bins)

    def get_accuracy(self, expert_id: str) -> float:
        """Get overall accuracy for an expert."""
        predictions = self._predictions.get(expert_id, [])
        if not predictions:
            return 0.5
        return sum(1 for p in predictions if p.correct) / len(predictions)

    def get_stats(self, expert_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an expert."""
        predictions = self._predictions.get(expert_id, [])
        if not predictions:
            return {
                'expert_id': expert_id,
                'predictions': 0,
                'accuracy': 0.5,
                'credit': self.initial_credit,
                'ece': 0.0,
                'brier_score': 0.0,
            }

        return {
            'expert_id': expert_id,
            'predictions': len(predictions),
            'accuracy': self.get_accuracy(expert_id),
            'credit': self.get_credit(expert_id),
            'ece': self.compute_ece(expert_id),
            'mce': self.compute_mce(expert_id),
            'brier_score': self.compute_brier_score(expert_id),
            'mean_confidence': sum(p.confidence for p in predictions) / len(predictions),
        }

    def get_all_experts(self) -> List[str]:
        """Get list of all tracked experts."""
        return list(self._predictions.keys())

    def rank_experts_by_credit(self) -> List[Tuple[str, float]]:
        """Rank experts by credit score (highest first)."""
        return sorted(
            self._credits.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def rank_experts_by_calibration(self) -> List[Tuple[str, float]]:
        """Rank experts by calibration (lowest ECE first)."""
        rankings = [
            (expert_id, self.compute_ece(expert_id))
            for expert_id in self._predictions
        ]
        return sorted(rankings, key=lambda x: x[1])

    def get_summary(self) -> Dict[str, Any]:
        """Get ledger summary."""
        return {
            'total_experts': len(self._predictions),
            'total_predictions': sum(len(p) for p in self._predictions.values()),
            'average_credit': (
                sum(self._credits.values()) / len(self._credits)
                if self._credits else self.initial_credit
            ),
            'top_experts': self.rank_experts_by_credit()[:5],
            'best_calibrated': self.rank_experts_by_calibration()[:5],
        }
