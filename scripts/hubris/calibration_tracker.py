#!/usr/bin/env python3
"""
Calibration Tracker for Expert Confidence Analysis

Tracks predicted confidence vs actual accuracy over time to detect
and correct systematic over/under-confidence in expert predictions.

Metrics computed:
- ECE (Expected Calibration Error): Average |confidence - accuracy|
- MCE (Max Calibration Error): Worst calibration gap
- Brier Score: Mean squared error of probability forecasts

Usage:
    from calibration_tracker import CalibrationTracker

    tracker = CalibrationTracker()
    tracker.load_from_resolved()  # Load from resolved.jsonl

    metrics = tracker.get_metrics()
    print(f"ECE: {metrics['ece']:.3f}")
    print(f"Trend: {metrics['trend']}")

    curve = tracker.get_calibration_curve()
    # [(0.1, 0.08), (0.2, 0.15), ...] - (predicted, actual) pairs
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class CalibrationRecord:
    """Single prediction-outcome pair for calibration analysis."""
    prediction_id: str
    expert_id: str
    confidence: float  # Predicted confidence (0-1)
    accuracy: float    # Actual accuracy (0-1)
    timestamp: float   # When prediction was made
    outcome_timestamp: float  # When outcome was evaluated


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics for an expert or system."""
    ece: float  # Expected Calibration Error
    mce: float  # Max Calibration Error
    brier_score: float  # Brier score (lower = better)
    confidence_mean: float  # Average predicted confidence
    accuracy_mean: float  # Average actual accuracy
    trend: str  # 'overconfident', 'underconfident', or 'well_calibrated'
    sample_count: int  # Number of predictions analyzed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CalibrationTracker:
    """
    Tracks and analyzes expert calibration quality.

    Loads resolved predictions and computes calibration metrics to
    detect systematic over/under-confidence.

    Attributes:
        records: List of calibration records
        predictions_dir: Directory containing prediction files
    """

    def __init__(self, predictions_dir: Optional[Path] = None):
        """
        Initialize calibration tracker.

        Args:
            predictions_dir: Directory for prediction files
                           (default: .git-ml/predictions)
        """
        if predictions_dir is None:
            git_ml_dir = Path(__file__).parent.parent.parent / '.git-ml'
            predictions_dir = git_ml_dir / 'predictions'

        self.predictions_dir = Path(predictions_dir)
        self.resolved_path = self.predictions_dir / 'resolved.jsonl'
        self.records: List[CalibrationRecord] = []
        self._by_expert: Dict[str, List[CalibrationRecord]] = {}

    def load_from_resolved(self) -> int:
        """
        Load calibration records from resolved.jsonl.

        Returns:
            Number of records loaded (only those with outcome data)
        """
        self.records = []
        self._by_expert = {}

        if not self.resolved_path.exists():
            return 0

        loaded = 0
        with open(self.resolved_path) as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                # Skip records without outcome data (old format)
                if 'accuracy' not in data or data.get('accuracy', 0) == 0:
                    # Check if it's truly missing or just zero accuracy
                    if 'actual_files' not in data:
                        continue

                record = CalibrationRecord(
                    prediction_id=data.get('prediction_id', ''),
                    expert_id=data.get('expert_id', 'unknown'),
                    confidence=data.get('confidence', 0.0),
                    accuracy=data.get('accuracy', 0.0),
                    timestamp=data.get('timestamp', 0.0),
                    outcome_timestamp=data.get('outcome_timestamp', 0.0)
                )

                self.records.append(record)

                # Index by expert
                if record.expert_id not in self._by_expert:
                    self._by_expert[record.expert_id] = []
                self._by_expert[record.expert_id].append(record)

                loaded += 1

        return loaded

    def add_record(
        self,
        prediction_id: str,
        expert_id: str,
        confidence: float,
        accuracy: float,
        timestamp: float = 0.0,
        outcome_timestamp: float = 0.0
    ) -> None:
        """
        Add a calibration record manually.

        Args:
            prediction_id: Unique prediction identifier
            expert_id: Expert that made the prediction
            confidence: Predicted confidence (0-1)
            accuracy: Actual accuracy (0-1)
            timestamp: When prediction was made
            outcome_timestamp: When outcome was evaluated
        """
        record = CalibrationRecord(
            prediction_id=prediction_id,
            expert_id=expert_id,
            confidence=confidence,
            accuracy=accuracy,
            timestamp=timestamp,
            outcome_timestamp=outcome_timestamp
        )

        self.records.append(record)

        if expert_id not in self._by_expert:
            self._by_expert[expert_id] = []
        self._by_expert[expert_id].append(record)

    def get_metrics(self, expert_id: Optional[str] = None) -> Optional[CalibrationMetrics]:
        """
        Compute calibration metrics.

        Args:
            expert_id: Specific expert to analyze (None = all experts)

        Returns:
            CalibrationMetrics or None if insufficient data
        """
        if expert_id:
            records = self._by_expert.get(expert_id, [])
        else:
            records = self.records

        if not records:
            return None

        # Compute metrics
        calibration_errors = []
        squared_errors = []
        confidences = []
        accuracies = []

        for r in records:
            calibration_errors.append(abs(r.confidence - r.accuracy))
            squared_errors.append((r.confidence - r.accuracy) ** 2)
            confidences.append(r.confidence)
            accuracies.append(r.accuracy)

        n = len(records)
        ece = sum(calibration_errors) / n
        mce = max(calibration_errors)
        brier = sum(squared_errors) / n
        conf_mean = sum(confidences) / n
        acc_mean = sum(accuracies) / n

        # Determine trend
        gap = conf_mean - acc_mean
        if gap > 0.05:
            trend = 'overconfident'
        elif gap < -0.05:
            trend = 'underconfident'
        else:
            trend = 'well_calibrated'

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            confidence_mean=conf_mean,
            accuracy_mean=acc_mean,
            trend=trend,
            sample_count=n
        )

    def get_calibration_curve(
        self,
        expert_id: Optional[str] = None,
        num_bins: int = 10
    ) -> List[Tuple[float, float, int]]:
        """
        Generate calibration curve via binning.

        Bins predictions by confidence, calculates actual accuracy in each bin.

        Args:
            expert_id: Specific expert (None = all)
            num_bins: Number of bins (default: 10 for 0.0-0.1, 0.1-0.2, etc.)

        Returns:
            List of (bin_center, actual_accuracy, count) tuples
        """
        if expert_id:
            records = self._by_expert.get(expert_id, [])
        else:
            records = self.records

        if not records:
            return []

        # Initialize bins
        bins: Dict[int, List[float]] = {i: [] for i in range(num_bins)}

        # Assign records to bins
        for r in records:
            bin_idx = min(int(r.confidence * num_bins), num_bins - 1)
            bins[bin_idx].append(r.accuracy)

        # Calculate curve
        curve = []
        for i in range(num_bins):
            if bins[i]:
                bin_center = (i + 0.5) / num_bins
                avg_accuracy = sum(bins[i]) / len(bins[i])
                count = len(bins[i])
                curve.append((bin_center, avg_accuracy, count))

        return curve

    def detect_overconfidence(self, threshold_ece: float = 0.10) -> Dict[str, bool]:
        """
        Detect which experts are overconfident.

        Args:
            threshold_ece: ECE threshold above which we flag overconfidence

        Returns:
            Dict mapping expert_id -> is_overconfident
        """
        results = {}

        for expert_id in self._by_expert:
            metrics = self.get_metrics(expert_id)
            if metrics and metrics.sample_count >= 5:
                is_overconfident = (
                    metrics.ece > threshold_ece and
                    metrics.trend == 'overconfident'
                )
                results[expert_id] = is_overconfident

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive calibration summary.

        Returns:
            Dict with overall and per-expert metrics
        """
        summary = {
            'total_records': len(self.records),
            'experts': {},
            'overall': None,
            'worst_calibrated': None,
            'best_calibrated': None
        }

        # Overall metrics
        overall = self.get_metrics()
        if overall:
            summary['overall'] = overall.to_dict()

        # Per-expert metrics
        worst_ece = 0.0
        best_ece = float('inf')

        for expert_id in self._by_expert:
            metrics = self.get_metrics(expert_id)
            if metrics:
                summary['experts'][expert_id] = {
                    'metrics': metrics.to_dict(),
                    'sample_count': metrics.sample_count,
                    'status': self._get_status(metrics.ece)
                }

                if metrics.ece > worst_ece and metrics.sample_count >= 5:
                    worst_ece = metrics.ece
                    summary['worst_calibrated'] = expert_id

                if metrics.ece < best_ece and metrics.sample_count >= 5:
                    best_ece = metrics.ece
                    summary['best_calibrated'] = expert_id

        return summary

    def _get_status(self, ece: float) -> str:
        """Classify ECE into status category."""
        if ece < 0.05:
            return 'excellent'
        elif ece < 0.10:
            return 'good'
        elif ece < 0.15:
            return 'acceptable'
        elif ece < 0.20:
            return 'needs_attention'
        else:
            return 'poor'

    def format_report(self) -> str:
        """
        Generate human-readable calibration report.

        Returns:
            Formatted string report
        """
        summary = self.get_summary()

        lines = [
            "=" * 60,
            "CALIBRATION ANALYSIS REPORT",
            "=" * 60,
            ""
        ]

        if summary['total_records'] == 0:
            lines.append("No calibration data available yet.")
            lines.append("Predictions with outcome data will appear after commits.")
            return "\n".join(lines)

        lines.append(f"Total predictions analyzed: {summary['total_records']}")
        lines.append("")

        # Overall metrics
        if summary['overall']:
            o = summary['overall']
            lines.append("OVERALL METRICS:")
            lines.append(f"  ECE (Expected Calibration Error): {o['ece']:.3f}")
            lines.append(f"  MCE (Max Calibration Error):      {o['mce']:.3f}")
            lines.append(f"  Brier Score:                      {o['brier_score']:.3f}")
            lines.append(f"  Average Confidence:               {o['confidence_mean']:.3f}")
            lines.append(f"  Average Accuracy:                 {o['accuracy_mean']:.3f}")
            lines.append(f"  Trend:                            {o['trend']}")
            lines.append("")

        # Per-expert breakdown
        if summary['experts']:
            lines.append("PER-EXPERT BREAKDOWN:")
            lines.append("-" * 60)

            for expert_id, data in sorted(summary['experts'].items()):
                m = data['metrics']
                status = data['status']
                lines.append(f"  {expert_id}:")
                lines.append(f"    Samples: {data['sample_count']}")
                lines.append(f"    ECE: {m['ece']:.3f} ({status})")
                lines.append(f"    Trend: {m['trend']}")
                lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        if summary['worst_calibrated']:
            lines.append(f"  ⚠️  {summary['worst_calibrated']} needs calibration attention")
        if summary['best_calibrated']:
            lines.append(f"  ✓  {summary['best_calibrated']} is best calibrated")

        if summary['overall'] and summary['overall']['ece'] < 0.10:
            lines.append("  ✓  Overall calibration is good (ECE < 0.10)")
        elif summary['overall'] and summary['overall']['ece'] >= 0.15:
            lines.append("  ⚠️  Overall calibration needs improvement (ECE >= 0.15)")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """CLI entry point for calibration analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze expert calibration')
    parser.add_argument('--expert', '-e', help='Specific expert to analyze')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--curve', action='store_true', help='Show calibration curve')

    args = parser.parse_args()

    tracker = CalibrationTracker()
    loaded = tracker.load_from_resolved()

    if args.json:
        import json
        print(json.dumps(tracker.get_summary(), indent=2))
    elif args.curve:
        curve = tracker.get_calibration_curve(args.expert)
        print("Calibration Curve (predicted_conf, actual_acc, count):")
        for conf, acc, count in curve:
            bar = "█" * int(acc * 20)
            print(f"  {conf:.1f}: {acc:.3f} [{count:3d}] {bar}")
    else:
        print(tracker.format_report())


if __name__ == '__main__':
    main()
