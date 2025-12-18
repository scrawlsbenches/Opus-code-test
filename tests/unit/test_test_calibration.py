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

    def test_empty_predictions_list(self):
        """Test edge case where suggested_tests is empty."""
        prediction = self.tracker.record_prediction(
            prediction_id="empty_pred",
            suggested_tests=[],  # Empty list
            confidence=0.5,
            changed_files=["file1.py"]
        )

        self.assertEqual(len(prediction.suggested_tests), 0)

        # Record outcome
        outcome = self.tracker.record_outcome(
            prediction_id="empty_pred",
            tests_run=["t1", "t2"],
            tests_failed=["t1"],
            tests_passed=["t2"]
        )

        # Should create calibration record with zero precision and recall
        self.assertEqual(len(self.tracker.calibration_records), 1)
        record = self.tracker.calibration_records[0]
        self.assertEqual(record.precision_at_5, 0.0)
        self.assertEqual(record.recall, 0.0)
        self.assertEqual(record.hit_rate, 0.0)
        self.assertEqual(record.mrr, 0.0)

    def test_duplicate_prediction_ids(self):
        """Test recording same prediction_id twice (should overwrite)."""
        # First prediction
        self.tracker.record_prediction(
            prediction_id="dup_id",
            suggested_tests=["t1", "t2"],
            confidence=0.6,
            changed_files=["f1.py"]
        )

        # Second prediction with same ID
        self.tracker.record_prediction(
            prediction_id="dup_id",
            suggested_tests=["t3", "t4", "t5"],
            confidence=0.9,
            changed_files=["f2.py"]
        )

        # In-memory dict should contain the second one
        self.assertIn("dup_id", self.tracker.predictions)
        pred = self.tracker.predictions["dup_id"]
        self.assertEqual(len(pred.suggested_tests), 3)
        self.assertEqual(pred.confidence, 0.9)

    def test_outcome_without_prediction(self):
        """Test recording outcome for non-existent prediction_id."""
        # Record outcome without prior prediction
        outcome = self.tracker.record_outcome(
            prediction_id="missing_pred",
            tests_run=["t1", "t2"],
            tests_failed=["t1"],
            tests_passed=["t2"]
        )

        # Outcome should be recorded
        self.assertIn("missing_pred", self.tracker.outcomes)

        # But no calibration record should be created (no matching prediction)
        self.assertEqual(len(self.tracker.calibration_records), 0)

    def test_large_number_of_tests(self):
        """Test prediction with 100+ suggested tests."""
        # Create 150 test names
        large_test_list = [f"test_{i:03d}" for i in range(150)]

        prediction = self.tracker.record_prediction(
            prediction_id="large_pred",
            suggested_tests=large_test_list,
            confidence=0.7,
            changed_files=["bigfile.py"]
        )

        self.assertEqual(len(prediction.suggested_tests), 150)

        # Record outcome with some failures
        outcome = self.tracker.record_outcome(
            prediction_id="large_pred",
            tests_run=large_test_list[:100],  # Only first 100 ran
            tests_failed=["test_002", "test_050"],  # 2 failures
            tests_passed=large_test_list[:100][:98]  # Rest passed
        )

        # Verify calibration record created
        self.assertEqual(len(self.tracker.calibration_records), 1)
        record = self.tracker.calibration_records[0]

        # Precision@5 should only look at first 5
        # All 5 should be in tests_run, so precision = 1.0
        self.assertEqual(record.precision_at_5, 1.0)

        # Recall: caught 2 out of 2 failures
        self.assertEqual(record.recall, 1.0)

        # MRR: first failure is at rank 3 (test_002)
        self.assertAlmostEqual(record.mrr, 1.0/3, places=2)

    def test_unicode_in_test_names(self):
        """Test handling of unicode characters in test paths."""
        unicode_tests = [
            "tests/test_données.py::test_français",
            "tests/test_日本語.py::test_漢字",
            "tests/test_emoji_😀.py::test_🎉"
        ]

        prediction = self.tracker.record_prediction(
            prediction_id="unicode_pred",
            suggested_tests=unicode_tests,
            confidence=0.8,
            changed_files=["src/unicode_文件.py"]
        )

        self.assertEqual(len(prediction.suggested_tests), 3)

        # Record outcome
        outcome = self.tracker.record_outcome(
            prediction_id="unicode_pred",
            tests_run=unicode_tests,
            tests_failed=[unicode_tests[0]],
            tests_passed=unicode_tests[1:]
        )

        # Should handle unicode correctly
        self.assertEqual(len(self.tracker.calibration_records), 1)
        record = self.tracker.calibration_records[0]
        self.assertIn(unicode_tests[0], record.tests_failed)

    def test_metrics_with_single_record(self):
        """Test metrics calculation with just 1 calibration record."""
        # Single prediction and outcome
        self.tracker.record_prediction(
            prediction_id="single",
            suggested_tests=["t1", "t2"],
            confidence=0.85,
            changed_files=["f1.py"]
        )
        self.tracker.record_outcome(
            prediction_id="single",
            tests_run=["t1", "t2"],
            tests_failed=["t1"],
            tests_passed=["t2"]
        )

        # Metrics should work with n=1
        metrics = self.tracker.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.sample_count, 1)

        # Should have valid values (not NaN or None)
        self.assertIsInstance(metrics.precision_at_5_mean, float)
        self.assertIsInstance(metrics.recall_mean, float)
        self.assertIsInstance(metrics.hit_rate, float)
        self.assertIsInstance(metrics.mrr, float)

        # Hit rate should be 1.0 (caught the failure)
        self.assertEqual(metrics.hit_rate, 1.0)

    def test_load_from_empty_directory(self):
        """Test loading when no data files exist."""
        # Create completely fresh tracker
        empty_dir = tempfile.mkdtemp()
        try:
            empty_tracker = CalibrationTracker(predictions_dir=Path(empty_dir))

            # Load should succeed but return 0 records
            loaded = empty_tracker.load_all()
            self.assertEqual(loaded, 0)
            self.assertEqual(len(empty_tracker.predictions), 0)
            self.assertEqual(len(empty_tracker.outcomes), 0)
            self.assertEqual(len(empty_tracker.calibration_records), 0)

            # Metrics should return None
            metrics = empty_tracker.get_metrics()
            self.assertIsNone(metrics)

            # Summary should indicate no data
            summary = empty_tracker.get_summary()
            self.assertEqual(summary['status'], 'no_data')
            self.assertEqual(summary['predictions_recorded'], 0)

        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
