#!/usr/bin/env python3
"""
Test Selection Calibration Tracker

Tracks test selection accuracy for TestExpert predictions. Measures how well
the system predicts which tests should be run given a set of changed files.

Metrics:
- Precision@K: Of top K suggested tests, how many were relevant (ran/failed)?
- Recall: Of tests that failed, what fraction did we suggest?
- Hit Rate: Did we suggest at least one failing test?
- MRR: Mean reciprocal rank of first failing test in suggestions

Data Flow:
1. TestExpert suggests tests → TestPrediction recorded
2. Tests run → TestOutcome recorded with results
3. Metrics computed by comparing predictions to outcomes

Storage:
- .git-ml/predictions/test_predictions.jsonl (predictions)
- .git-ml/predictions/test_outcomes.jsonl (outcomes)
- .git-ml/predictions/test_calibration.jsonl (matched pairs for analysis)
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set


@dataclass
class TestPrediction:
    """
    Test selection prediction awaiting evaluation.

    Attributes:
        prediction_id: Unique identifier for this prediction
        suggested_tests: List of test names suggested to run
        confidence: Overall confidence in this selection (0-1)
        changed_files: Files that changed (context for prediction)
        timestamp: When prediction was made
        metadata: Additional context (commit message, branch, etc.)
    """
    prediction_id: str
    suggested_tests: List[str]
    confidence: float
    changed_files: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestPrediction':
        """Load from dict."""
        # Handle missing metadata field (backward compatibility)
        if 'metadata' not in data:
            data['metadata'] = {}
        return cls(**data)


@dataclass
class TestOutcome:
    """
    Actual test execution results for a prediction.

    Attributes:
        prediction_id: Links to TestPrediction
        tests_run: All tests that were executed
        tests_failed: Tests that failed
        tests_passed: Tests that passed
        timestamp: When tests were executed
        metadata: Additional context (CI run, duration, etc.)
    """
    prediction_id: str
    tests_run: List[str]
    tests_failed: List[str]
    tests_passed: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestOutcome':
        """Load from dict."""
        if 'metadata' not in data:
            data['metadata'] = {}
        return cls(**data)


@dataclass
class TestCalibrationRecord:
    """
    Matched prediction-outcome pair for calibration analysis.

    Combines TestPrediction + TestOutcome to enable metric calculation.

    Attributes:
        prediction_id: Unique identifier
        suggested_tests: Tests suggested by expert
        tests_failed: Tests that actually failed
        tests_run: All tests that ran
        confidence: Predicted confidence
        precision_at_5: Of top 5 suggestions, how many were relevant?
        recall: Of failing tests, what fraction did we suggest?
        hit_rate: 1.0 if we suggested any failing test, else 0.0
        mrr: Reciprocal rank of first failing test (0 if none)
        timestamp: When prediction was made
    """
    prediction_id: str
    suggested_tests: List[str]
    tests_failed: List[str]
    tests_run: List[str]
    confidence: float
    precision_at_5: float
    recall: float
    hit_rate: float
    mrr: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCalibrationRecord':
        """Load from dict."""
        return cls(**data)


@dataclass
class TestCalibrationMetrics:
    """
    Aggregate test selection calibration metrics.

    Attributes:
        precision_at_5_mean: Average precision of top 5 suggestions
        recall_mean: Average recall of failing tests
        hit_rate: Fraction of predictions that caught at least one failure
        mrr: Mean reciprocal rank across all predictions
        sample_count: Number of predictions analyzed
        false_alarm_rate: Fraction of suggested tests that didn't fail
        coverage: Fraction of all test failures caught by suggestions
    """
    precision_at_5_mean: float
    recall_mean: float
    hit_rate: float
    mrr: float
    sample_count: int
    false_alarm_rate: float
    coverage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_status(self) -> str:
        """Classify overall test selection quality."""
        if self.hit_rate >= 0.95 and self.precision_at_5_mean >= 0.8:
            return 'excellent'
        elif self.hit_rate >= 0.85 and self.precision_at_5_mean >= 0.6:
            return 'good'
        elif self.hit_rate >= 0.70 and self.precision_at_5_mean >= 0.4:
            return 'acceptable'
        elif self.hit_rate >= 0.50:
            return 'needs_attention'
        else:
            return 'poor'


class TestCalibrationTracker:
    """
    Tracks test selection accuracy and calibration quality.

    Records test predictions, matches them with outcomes, and computes
    metrics to measure how well TestExpert selects which tests to run.

    Attributes:
        predictions_dir: Directory containing prediction files
        predictions_path: Path to test predictions file
        outcomes_path: Path to test outcomes file
        calibration_path: Path to calibration records file
    """

    def __init__(self, predictions_dir: Optional[Path] = None):
        """
        Initialize test calibration tracker.

        Args:
            predictions_dir: Directory for prediction files
                           (default: .git-ml/predictions)
        """
        if predictions_dir is None:
            git_ml_dir = Path(__file__).parent.parent.parent / '.git-ml'
            predictions_dir = git_ml_dir / 'predictions'

        self.predictions_dir = Path(predictions_dir)
        self.predictions_path = self.predictions_dir / 'test_predictions.jsonl'
        self.outcomes_path = self.predictions_dir / 'test_outcomes.jsonl'
        self.calibration_path = self.predictions_dir / 'test_calibration.jsonl'

        # Create directory if needed
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.predictions: Dict[str, TestPrediction] = {}
        self.outcomes: Dict[str, TestOutcome] = {}
        self.calibration_records: List[TestCalibrationRecord] = []

    def record_prediction(
        self,
        prediction_id: str,
        suggested_tests: List[str],
        confidence: float,
        changed_files: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TestPrediction:
        """
        Record a test selection prediction.

        Args:
            prediction_id: Unique identifier for this prediction
            suggested_tests: Test names suggested to run
            confidence: Overall confidence (0-1)
            changed_files: Files that changed (prediction context)
            metadata: Optional additional context

        Returns:
            TestPrediction that was recorded

        Raises:
            ValueError: If confidence not in [0, 1]
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

        prediction = TestPrediction(
            prediction_id=prediction_id,
            suggested_tests=suggested_tests,
            confidence=confidence,
            changed_files=changed_files,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        # Store in memory
        self.predictions[prediction_id] = prediction

        # Append to file
        self._append_jsonl(self.predictions_path, prediction.to_dict())

        return prediction

    def record_outcome(
        self,
        prediction_id: str,
        tests_run: List[str],
        tests_failed: List[str],
        tests_passed: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TestOutcome:
        """
        Record test execution outcome.

        Args:
            prediction_id: Links to TestPrediction
            tests_run: All tests executed
            tests_failed: Tests that failed
            tests_passed: Tests that passed
            metadata: Optional additional context (CI info, etc.)

        Returns:
            TestOutcome that was recorded
        """
        outcome = TestOutcome(
            prediction_id=prediction_id,
            tests_run=tests_run,
            tests_failed=tests_failed,
            tests_passed=tests_passed,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        # Store in memory
        self.outcomes[prediction_id] = outcome

        # Append to file
        self._append_jsonl(self.outcomes_path, outcome.to_dict())

        # Try to create calibration record if prediction exists
        if prediction_id in self.predictions:
            self._create_calibration_record(prediction_id)

        return outcome

    def _create_calibration_record(self, prediction_id: str) -> None:
        """
        Create calibration record by matching prediction to outcome.

        Args:
            prediction_id: ID to match
        """
        if prediction_id not in self.predictions or prediction_id not in self.outcomes:
            return

        pred = self.predictions[prediction_id]
        outcome = self.outcomes[prediction_id]

        # Calculate metrics
        suggested_set = set(pred.suggested_tests)
        failed_set = set(outcome.tests_failed)
        run_set = set(outcome.tests_run)

        # Precision@5: of top 5 suggestions, how many were relevant?
        # Relevant = test ran or test failed
        top_5 = set(pred.suggested_tests[:5])
        relevant_top_5 = top_5 & (run_set | failed_set)
        precision_at_5 = len(relevant_top_5) / min(5, len(pred.suggested_tests)) if pred.suggested_tests else 0.0

        # Recall: of tests that failed, what fraction did we suggest?
        if failed_set:
            recall = len(suggested_set & failed_set) / len(failed_set)
        else:
            recall = 1.0  # No failures → perfect recall

        # Hit rate: did we suggest at least one failing test?
        hit_rate = 1.0 if (suggested_set & failed_set) else 0.0

        # MRR: reciprocal rank of first failing test
        mrr = 0.0
        for rank, test_name in enumerate(pred.suggested_tests, 1):
            if test_name in failed_set:
                mrr = 1.0 / rank
                break

        record = TestCalibrationRecord(
            prediction_id=prediction_id,
            suggested_tests=pred.suggested_tests,
            tests_failed=outcome.tests_failed,
            tests_run=outcome.tests_run,
            confidence=pred.confidence,
            precision_at_5=precision_at_5,
            recall=recall,
            hit_rate=hit_rate,
            mrr=mrr,
            timestamp=pred.timestamp
        )

        self.calibration_records.append(record)
        self._append_jsonl(self.calibration_path, record.to_dict())

    def load_all(self) -> int:
        """
        Load all predictions, outcomes, and calibration records.

        Returns:
            Number of calibration records loaded
        """
        # Load predictions
        if self.predictions_path.exists():
            with open(self.predictions_path) as f:
                for line in f:
                    if line.strip():
                        pred = TestPrediction.from_dict(json.loads(line))
                        self.predictions[pred.prediction_id] = pred

        # Load outcomes
        if self.outcomes_path.exists():
            with open(self.outcomes_path) as f:
                for line in f:
                    if line.strip():
                        outcome = TestOutcome.from_dict(json.loads(line))
                        self.outcomes[outcome.prediction_id] = outcome

        # Load calibration records
        self.calibration_records = []
        if self.calibration_path.exists():
            with open(self.calibration_path) as f:
                for line in f:
                    if line.strip():
                        record = TestCalibrationRecord.from_dict(json.loads(line))
                        self.calibration_records.append(record)

        return len(self.calibration_records)

    def get_metrics(self) -> Optional[TestCalibrationMetrics]:
        """
        Compute aggregate test selection metrics.

        Returns:
            TestCalibrationMetrics or None if insufficient data
        """
        if not self.calibration_records:
            return None

        precision_at_5_values = []
        recall_values = []
        hit_rates = []
        mrr_values = []
        false_alarms = []

        total_failures = 0
        caught_failures = 0

        for record in self.calibration_records:
            precision_at_5_values.append(record.precision_at_5)
            recall_values.append(record.recall)
            hit_rates.append(record.hit_rate)
            mrr_values.append(record.mrr)

            # False alarm rate: suggested tests that didn't fail
            suggested_set = set(record.suggested_tests[:5])
            failed_set = set(record.tests_failed)
            false_alarm = len(suggested_set - failed_set) / len(suggested_set) if suggested_set else 0.0
            false_alarms.append(false_alarm)

            # Coverage: total failures caught
            total_failures += len(record.tests_failed)
            caught_failures += len(set(record.suggested_tests) & failed_set)

        n = len(self.calibration_records)

        return TestCalibrationMetrics(
            precision_at_5_mean=sum(precision_at_5_values) / n,
            recall_mean=sum(recall_values) / n,
            hit_rate=sum(hit_rates) / n,
            mrr=sum(mrr_values) / n,
            sample_count=n,
            false_alarm_rate=sum(false_alarms) / n,
            coverage=caught_failures / total_failures if total_failures > 0 else 1.0
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of test calibration.

        Returns:
            Dict with metrics, status, and recommendations
        """
        metrics = self.get_metrics()

        summary = {
            'predictions_recorded': len(self.predictions),
            'outcomes_recorded': len(self.outcomes),
            'calibration_records': len(self.calibration_records),
            'metrics': None,
            'status': 'no_data',
            'recommendations': []
        }

        if metrics:
            summary['metrics'] = metrics.to_dict()
            summary['status'] = metrics.get_status()

            # Generate recommendations
            recommendations = []

            if metrics.hit_rate < 0.70:
                recommendations.append(
                    "⚠️  Low hit rate - many test failures not predicted. "
                    "Consider expanding test selection coverage."
                )

            if metrics.precision_at_5_mean < 0.5:
                recommendations.append(
                    "⚠️  Low precision - many suggested tests are not relevant. "
                    "Consider improving test selection model or filtering."
                )

            if metrics.false_alarm_rate > 0.6:
                recommendations.append(
                    "ℹ️  High false alarm rate - suggested tests mostly pass. "
                    "This wastes CI time. Improve relevance filtering."
                )

            if metrics.mrr < 0.3:
                recommendations.append(
                    "ℹ️  Low MRR - failing tests ranked low in suggestions. "
                    "Improve ranking to surface likely failures earlier."
                )

            if metrics.get_status() in ['excellent', 'good']:
                recommendations.append(
                    "✓ Test selection quality is good. Continue monitoring."
                )

            summary['recommendations'] = recommendations

        return summary

    def format_report(self) -> str:
        """
        Generate human-readable test calibration report.

        Returns:
            Formatted string report
        """
        summary = self.get_summary()

        lines = [
            "=" * 70,
            "TEST SELECTION CALIBRATION REPORT",
            "=" * 70,
            ""
        ]

        if summary['calibration_records'] == 0:
            lines.append("No test calibration data available yet.")
            lines.append("")
            lines.append("To generate test calibration data:")
            lines.append("  1. Record test predictions via TestCalibrationTracker")
            lines.append("  2. Run tests and record outcomes")
            lines.append("  3. Re-run this command")
            return "\n".join(lines)

        lines.append(f"Predictions recorded:  {summary['predictions_recorded']}")
        lines.append(f"Outcomes recorded:     {summary['outcomes_recorded']}")
        lines.append(f"Calibration records:   {summary['calibration_records']}")
        lines.append("")

        if summary['metrics']:
            m = summary['metrics']
            status = summary['status']

            lines.append("TEST SELECTION METRICS:")
            lines.append(f"  Precision@5:      {m['precision_at_5_mean']:.3f}  (of top 5, how many relevant?)")
            lines.append(f"  Recall:           {m['recall_mean']:.3f}  (of failures, what % caught?)")
            lines.append(f"  Hit Rate:         {m['hit_rate']:.3f}  (% predictions catching failures)")
            lines.append(f"  MRR:              {m['mrr']:.3f}  (rank of first failure)")
            lines.append(f"  False Alarm Rate: {m['false_alarm_rate']:.3f}  (suggested but didn't fail)")
            lines.append(f"  Coverage:         {m['coverage']:.3f}  (% all failures caught)")
            lines.append("")
            lines.append(f"Status: {status.upper()}")
            lines.append("")

        if summary['recommendations']:
            lines.append("RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                lines.append(f"  {rec}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _append_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically append JSON line to file.

        Args:
            path: File to append to
            data: Dictionary to serialize
        """
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix='.tmp_',
            suffix='.jsonl'
        )

        try:
            # Copy existing content + new line
            if path.exists():
                with open(path) as src:
                    with os.fdopen(fd, 'w') as dst:
                        dst.write(src.read())
                        dst.write(json.dumps(data) + '\n')
            else:
                with os.fdopen(fd, 'w') as dst:
                    dst.write(json.dumps(data) + '\n')

            # Atomic rename
            os.replace(temp_path, path)
        except:
            # Clean up on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise


def main():
    """CLI entry point for test calibration analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze test selection calibration'
    )
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')

    args = parser.parse_args()

    tracker = TestCalibrationTracker()
    loaded = tracker.load_all()

    if args.json:
        import json
        print(json.dumps(tracker.get_summary(), indent=2))
    else:
        print(tracker.format_report())


if __name__ == '__main__':
    main()
