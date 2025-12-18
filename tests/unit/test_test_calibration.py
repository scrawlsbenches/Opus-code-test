#!/usr/bin/env python3
"""
Unit tests for TestCalibrationTracker
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add scripts/hubris to path
hubris_dir = Path(__file__).parent.parent.parent / 'scripts' / 'hubris'
sys.path.insert(0, str(hubris_dir))

# Import with aliases to avoid pytest collection (pytest collects classes starting with 'Test')
from test_calibration_tracker import (
    TestCalibrationTracker as CalibrationTracker,
    TestPrediction as Prediction,
    TestOutcome as Outcome,
    TestCalibrationRecord as CalibrationRecord,
    TestCalibrationMetrics as CalibrationMetrics
)


class TestTestCalibrationTracker(unittest.TestCase):
    """Test TestCalibrationTracker functionality."""

    def setUp(self):
        """Create temp directory for test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = CalibrationTracker(predictions_dir=Path(self.temp_dir))

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_prediction(self):
        """Test recording a test prediction."""
        prediction = self.tracker.record_prediction(
            prediction_id="test_123",
            suggested_tests=["test_a", "test_b", "test_c"],
            confidence=0.8,
            changed_files=["file1.py", "file2.py"],
            metadata={"commit": "abc123"}
        )

        self.assertEqual(prediction.prediction_id, "test_123")
        self.assertEqual(len(prediction.suggested_tests), 3)
        self.assertEqual(prediction.confidence, 0.8)
        self.assertEqual(len(prediction.changed_files), 2)

        # Verify stored in memory
        self.assertIn("test_123", self.tracker.predictions)

    def test_record_outcome(self):
        """Test recording test outcome."""
        # Record prediction first
        self.tracker.record_prediction(
            prediction_id="test_456",
            suggested_tests=["test_a", "test_b"],
            confidence=0.9,
            changed_files=["file1.py"]
        )

        # Record outcome
        outcome = self.tracker.record_outcome(
            prediction_id="test_456",
            tests_run=["test_a", "test_b", "test_c"],
            tests_failed=["test_a"],
            tests_passed=["test_b", "test_c"]
        )

        self.assertEqual(outcome.prediction_id, "test_456")
        self.assertEqual(len(outcome.tests_run), 3)
        self.assertEqual(len(outcome.tests_failed), 1)

        # Verify calibration record created
        self.assertEqual(len(self.tracker.calibration_records), 1)

    def test_precision_at_5_calculation(self):
        """Test precision@5 metric calculation."""
        # Perfect prediction: all 5 suggestions run
        self.tracker.record_prediction(
            prediction_id="p1",
            suggested_tests=["t1", "t2", "t3", "t4", "t5"],
            confidence=0.9,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p1",
            tests_run=["t1", "t2", "t3", "t4", "t5"],
            tests_failed=["t1"],
            tests_passed=["t2", "t3", "t4", "t5"]
        )

        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.precision_at_5_mean, 1.0)

    def test_recall_calculation(self):
        """Test recall metric calculation."""
        # Caught 2 out of 3 failures
        self.tracker.record_prediction(
            prediction_id="p2",
            suggested_tests=["t1", "t2", "t3"],
            confidence=0.7,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p2",
            tests_run=["t1", "t2", "t3", "t4"],
            tests_failed=["t1", "t2", "t4"],  # t4 not suggested
            tests_passed=["t3"]
        )

        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)
        # Recall = 2/3 = 0.667
        self.assertAlmostEqual(metrics.recall_mean, 2/3, places=2)

    def test_hit_rate_calculation(self):
        """Test hit rate metric calculation."""
        # Prediction 1: caught failure (hit_rate = 1.0)
        self.tracker.record_prediction(
            prediction_id="p3",
            suggested_tests=["t1"],
            confidence=0.8,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p3",
            tests_run=["t1", "t2"],
            tests_failed=["t1"],
            tests_passed=["t2"]
        )

        # Prediction 2: missed failure (hit_rate = 0.0)
        self.tracker.record_prediction(
            prediction_id="p4",
            suggested_tests=["t3"],
            confidence=0.6,
            changed_files=["f2.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p4",
            tests_run=["t3", "t4"],
            tests_failed=["t4"],  # Not suggested!
            tests_passed=["t3"]
        )

        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)
        # Hit rate = 1/2 = 0.5
        self.assertEqual(metrics.hit_rate, 0.5)

    def test_mrr_calculation(self):
        """Test MRR (Mean Reciprocal Rank) calculation."""
        # Failure at rank 1
        self.tracker.record_prediction(
            prediction_id="p5",
            suggested_tests=["t1", "t2", "t3"],
            confidence=0.9,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p5",
            tests_run=["t1", "t2", "t3"],
            tests_failed=["t1"],
            tests_passed=["t2", "t3"]
        )

        # Failure at rank 2
        self.tracker.record_prediction(
            prediction_id="p6",
            suggested_tests=["t4", "t5", "t6"],
            confidence=0.8,
            changed_files=["f2.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p6",
            tests_run=["t4", "t5", "t6"],
            tests_failed=["t5"],
            tests_passed=["t4", "t6"]
        )

        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)
        # MRR = (1/1 + 1/2) / 2 = 0.75
        self.assertAlmostEqual(metrics.mrr, 0.75, places=2)

    def test_no_failures_scenario(self):
        """Test scenario where no tests fail (perfect recall)."""
        self.tracker.record_prediction(
            prediction_id="p7",
            suggested_tests=["t1", "t2"],
            confidence=0.7,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p7",
            tests_run=["t1", "t2"],
            tests_failed=[],  # No failures
            tests_passed=["t1", "t2"]
        )

        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)
        # No failures → recall = 1.0
        self.assertEqual(metrics.recall_mean, 1.0)

    def test_status_classification(self):
        """Test metric status classification."""
        # Create excellent performance scenario
        self.tracker.record_prediction(
            prediction_id="p8",
            suggested_tests=["t1", "t2", "t3", "t4", "t5"],
            confidence=0.95,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p8",
            tests_run=["t1", "t2", "t3", "t4", "t5"],
            tests_failed=["t1"],
            tests_passed=["t2", "t3", "t4", "t5"]
        )

        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)

        # Perfect precision and hit rate
        status = metrics.get_status()
        # Status depends on hit_rate >= 0.95 and precision >= 0.8
        # We have 100% hit rate and 100% precision for this single case
        # But the thresholds need hit_rate >= 0.95
        self.assertIn(status, ['excellent', 'good'])

    def test_persistence(self):
        """Test data persistence to files."""
        # Record and save
        self.tracker.record_prediction(
            prediction_id="p9",
            suggested_tests=["t1", "t2"],
            confidence=0.8,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="p9",
            tests_run=["t1", "t2"],
            tests_failed=["t1"],
            tests_passed=["t2"]
        )

        # Create new tracker and load
        tracker2 = CalibrationTracker(predictions_dir=Path(self.temp_dir))
        loaded = tracker2.load_all()

        self.assertEqual(loaded, 1)
        self.assertEqual(len(tracker2.predictions), 1)
        self.assertEqual(len(tracker2.outcomes), 1)
        self.assertEqual(len(tracker2.calibration_records), 1)

    def test_invalid_confidence(self):
        """Test validation of confidence values."""
        with self.assertRaises(ValueError):
            self.tracker.record_prediction(
                prediction_id="bad",
                suggested_tests=["t1"],
                confidence=1.5,  # Invalid: > 1.0
                changed_files=["f1.py"]
            )

        with self.assertRaises(ValueError):
            self.tracker.record_prediction(
                prediction_id="bad2",
                suggested_tests=["t1"],
                confidence=-0.1,  # Invalid: < 0.0
                changed_files=["f1.py"]
            )


if __name__ == '__main__':
    unittest.main()
