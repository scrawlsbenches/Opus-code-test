#!/usr/bin/env python3
"""
Demo script for TestCalibrationTracker

Shows how to record test predictions and outcomes to track test selection accuracy.
"""

import time
import uuid
from pathlib import Path
from test_calibration_tracker import TestCalibrationTracker


def generate_prediction_id() -> str:
    """Generate unique prediction ID."""
    return f"test_pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def demo_basic_usage():
    """Demonstrate basic test calibration workflow."""
    print("=" * 70)
    print("TEST CALIBRATION TRACKER DEMO")
    print("=" * 70)
    print()

    tracker = TestCalibrationTracker()

    # Scenario 1: Good prediction - caught most failures
    print("Scenario 1: Good prediction (catches most failures)")
    print("-" * 70)

    pred_id_1 = generate_prediction_id()
    prediction_1 = tracker.record_prediction(
        prediction_id=pred_id_1,
        suggested_tests=[
            "tests/test_auth.py::test_login",
            "tests/test_auth.py::test_logout",
            "tests/test_session.py::test_create",
            "tests/test_database.py::test_connect",
            "tests/test_api.py::test_endpoint"
        ],
        confidence=0.85,
        changed_files=["cortical/auth.py", "cortical/session.py"],
        metadata={"commit_message": "Add authentication feature"}
    )
    print(f"  Recorded prediction: {pred_id_1}")
    print(f"  Suggested {len(prediction_1.suggested_tests)} tests")
    print()

    # Simulate test run - 2 tests fail, 3 pass
    outcome_1 = tracker.record_outcome(
        prediction_id=pred_id_1,
        tests_run=[
            "tests/test_auth.py::test_login",
            "tests/test_auth.py::test_logout",
            "tests/test_session.py::test_create",
            "tests/test_database.py::test_connect",
            "tests/test_api.py::test_endpoint"
        ],
        tests_failed=[
            "tests/test_auth.py::test_login",
            "tests/test_session.py::test_create"
        ],
        tests_passed=[
            "tests/test_auth.py::test_logout",
            "tests/test_database.py::test_connect",
            "tests/test_api.py::test_endpoint"
        ],
        metadata={"ci_run": "12345", "duration_seconds": 45}
    )
    print(f"  Tests run: {len(outcome_1.tests_run)}")
    print(f"  Tests failed: {len(outcome_1.tests_failed)}")
    print(f"  ✓ Caught both failing tests!")
    print()

    # Scenario 2: Poor prediction - missed failures
    print("Scenario 2: Poor prediction (misses failures)")
    print("-" * 70)

    pred_id_2 = generate_prediction_id()
    prediction_2 = tracker.record_prediction(
        prediction_id=pred_id_2,
        suggested_tests=[
            "tests/test_utils.py::test_helper",
            "tests/test_config.py::test_load",
            "tests/test_logging.py::test_format"
        ],
        confidence=0.60,
        changed_files=["cortical/processor.py"],
        metadata={"commit_message": "Refactor processor"}
    )
    print(f"  Recorded prediction: {pred_id_2}")
    print(f"  Suggested {len(prediction_2.suggested_tests)} tests")
    print()

    # Simulate test run - failures in different tests
    outcome_2 = tracker.record_outcome(
        prediction_id=pred_id_2,
        tests_run=[
            "tests/test_utils.py::test_helper",
            "tests/test_config.py::test_load",
            "tests/test_logging.py::test_format",
            "tests/test_processor.py::test_compute",
            "tests/test_analysis.py::test_pagerank"
        ],
        tests_failed=[
            "tests/test_processor.py::test_compute",
            "tests/test_analysis.py::test_pagerank"
        ],
        tests_passed=[
            "tests/test_utils.py::test_helper",
            "tests/test_config.py::test_load",
            "tests/test_logging.py::test_format"
        ],
        metadata={"ci_run": "12346", "duration_seconds": 52}
    )
    print(f"  Tests run: {len(outcome_2.tests_run)}")
    print(f"  Tests failed: {len(outcome_2.tests_failed)}")
    print(f"  ❌ Missed both failing tests!")
    print()

    # Scenario 3: Partial success
    print("Scenario 3: Partial success (caught some failures)")
    print("-" * 70)

    pred_id_3 = generate_prediction_id()
    prediction_3 = tracker.record_prediction(
        prediction_id=pred_id_3,
        suggested_tests=[
            "tests/test_query.py::test_search",
            "tests/test_query.py::test_expand",
            "tests/test_semantics.py::test_extract",
            "tests/test_tokenizer.py::test_stem"
        ],
        confidence=0.72,
        changed_files=["cortical/query.py"],
        metadata={"commit_message": "Improve search ranking"}
    )
    print(f"  Recorded prediction: {pred_id_3}")
    print(f"  Suggested {len(prediction_3.suggested_tests)} tests")
    print()

    outcome_3 = tracker.record_outcome(
        prediction_id=pred_id_3,
        tests_run=[
            "tests/test_query.py::test_search",
            "tests/test_query.py::test_expand",
            "tests/test_semantics.py::test_extract",
            "tests/test_tokenizer.py::test_stem",
            "tests/test_layers.py::test_lookup"
        ],
        tests_failed=[
            "tests/test_query.py::test_search",
            "tests/test_layers.py::test_lookup"
        ],
        tests_passed=[
            "tests/test_query.py::test_expand",
            "tests/test_semantics.py::test_extract",
            "tests/test_tokenizer.py::test_stem"
        ],
        metadata={"ci_run": "12347", "duration_seconds": 38}
    )
    print(f"  Tests run: {len(outcome_3.tests_run)}")
    print(f"  Tests failed: {len(outcome_3.tests_failed)}")
    print(f"  ⚠️  Caught 1 out of 2 failing tests")
    print()

    # Show metrics
    print("=" * 70)
    print("CALIBRATION METRICS")
    print("=" * 70)
    print()

    metrics = tracker.get_metrics()

    if metrics:
        print(f"Precision@5:      {metrics.precision_at_5_mean:.3f}")
        print(f"  (Of top 5 suggestions, how many were relevant?)")
        print()

        print(f"Recall:           {metrics.recall_mean:.3f}")
        print(f"  (Of failures, what fraction did we catch?)")
        print()

        print(f"Hit Rate:         {metrics.hit_rate:.3f}")
        print(f"  (% predictions catching at least one failure)")
        print()

        print(f"MRR:              {metrics.mrr:.3f}")
        print(f"  (Rank of first failure in suggestions)")
        print()

        print(f"False Alarm Rate: {metrics.false_alarm_rate:.3f}")
        print(f"  (Suggested tests that didn't fail)")
        print()

        print(f"Coverage:         {metrics.coverage:.3f}")
        print(f"  (% of all failures caught)")
        print()

        print(f"Status: {metrics.get_status().upper()}")
        print()

    # Show full report
    print("=" * 70)
    print("FULL REPORT")
    print("=" * 70)
    print()
    print(tracker.format_report())


if __name__ == '__main__':
    demo_basic_usage()
