# Test Selection Calibration Metrics

## Overview

The **Test Calibration Tracker** measures how accurately TestExpert predicts which tests should be run given a set of changed files. This enables continuous improvement of test selection algorithms by tracking real-world prediction accuracy.

## Core Concepts

### What Gets Tracked

1. **Predictions**: TestExpert suggests which tests to run
2. **Outcomes**: Tests actually run, with pass/fail results
3. **Calibration**: Comparing predictions vs outcomes to compute accuracy metrics

### Storage

All data is stored in `.git-ml/predictions/` (gitignored):

```
.git-ml/predictions/
├── test_predictions.jsonl   # Test selection predictions
├── test_outcomes.jsonl       # Test execution results
└── test_calibration.jsonl    # Matched prediction-outcome pairs
```

## Metrics Explained

### Precision@5
**Definition**: Of the top 5 suggested tests, how many were relevant (ran or failed)?

**Formula**: `relevant_in_top_5 / min(5, total_suggested)`

**Good threshold**: ≥ 0.7 (70% of suggestions are useful)

**Example**:
- Suggested top 5: `[test_a, test_b, test_c, test_d, test_e]`
- Tests run: `[test_a, test_b, test_c, test_f]`
- Relevant in top 5: 3 (a, b, c)
- Precision@5 = 3/5 = 0.6

### Recall
**Definition**: Of all tests that failed, what fraction did we suggest?

**Formula**: `suggested_failures / total_failures`

**Good threshold**: ≥ 0.8 (catch 80% of failures)

**Example**:
- Tests failed: `[test_a, test_b, test_c]`
- Suggested tests: `[test_a, test_b, test_d]`
- Caught failures: 2 (a, b)
- Recall = 2/3 = 0.667

### Hit Rate
**Definition**: What fraction of predictions caught at least one failing test?

**Formula**: `predictions_with_failures / total_predictions`

**Good threshold**: ≥ 0.85 (rarely miss failures entirely)

**Example**:
- Prediction 1: suggested `[test_a]`, `test_a` failed → **hit**
- Prediction 2: suggested `[test_b]`, `test_c` failed → **miss**
- Hit rate = 1/2 = 0.5

### MRR (Mean Reciprocal Rank)
**Definition**: Average reciprocal rank of the first failing test in suggestions.

**Formula**: `mean(1 / rank_of_first_failure)`

**Good threshold**: ≥ 0.5 (failures appear in top 2 on average)

**Example**:
- Prediction 1: first failure at rank 1 → RR = 1.0
- Prediction 2: first failure at rank 3 → RR = 0.333
- MRR = (1.0 + 0.333) / 2 = 0.667

### False Alarm Rate
**Definition**: Fraction of suggested tests that didn't fail (wasted CI time).

**Formula**: `suggested_but_passed / total_suggested`

**Good threshold**: ≤ 0.3 (minimize wasted CI resources)

### Coverage
**Definition**: Fraction of all test failures caught by suggestions.

**Formula**: `total_failures_caught / total_failures_across_all_predictions`

**Good threshold**: ≥ 0.9 (catch 90%+ of all failures)

## Usage

### Recording Predictions

When TestExpert makes a prediction:

```python
from test_calibration_tracker import TestCalibrationTracker

tracker = TestCalibrationTracker()

prediction = tracker.record_prediction(
    prediction_id="pred_12345",
    suggested_tests=[
        "tests/test_auth.py::test_login",
        "tests/test_auth.py::test_logout",
        "tests/test_session.py::test_create"
    ],
    confidence=0.85,
    changed_files=["cortical/auth.py", "cortical/session.py"],
    metadata={
        "commit_message": "Add authentication feature",
        "timestamp": "2025-12-18T10:30:00Z"
    }
)
```

### Recording Outcomes

After tests run:

```python
outcome = tracker.record_outcome(
    prediction_id="pred_12345",
    tests_run=[
        "tests/test_auth.py::test_login",
        "tests/test_auth.py::test_logout",
        "tests/test_session.py::test_create",
        "tests/test_database.py::test_connect"
    ],
    tests_failed=[
        "tests/test_auth.py::test_login"
    ],
    tests_passed=[
        "tests/test_auth.py::test_logout",
        "tests/test_session.py::test_create",
        "tests/test_database.py::test_connect"
    ],
    metadata={
        "ci_run": "12345",
        "duration_seconds": 45
    }
)
```

### Viewing Metrics

#### Command Line

```bash
# View test selection calibration
python scripts/hubris_cli.py calibration --tests

# JSON output
python scripts/hubris_cli.py calibration --tests --json
```

#### Programmatic Access

```python
# Load all data
tracker = TestCalibrationTracker()
tracker.load_all()

# Get metrics
metrics = tracker.get_metrics()
print(f"Precision@5: {metrics.precision_at_5_mean:.3f}")
print(f"Recall: {metrics.recall_mean:.3f}")
print(f"Hit Rate: {metrics.hit_rate:.3f}")
print(f"Status: {metrics.get_status()}")

# Get full report
print(tracker.format_report())
```

## Interpreting Results

### Status Levels

| Status | Hit Rate | Precision@5 | Meaning |
|--------|----------|-------------|---------|
| **Excellent** | ≥ 0.95 | ≥ 0.8 | Nearly perfect test selection |
| **Good** | ≥ 0.85 | ≥ 0.6 | Solid performance, minor improvements possible |
| **Acceptable** | ≥ 0.70 | ≥ 0.4 | Usable but needs improvement |
| **Needs Attention** | ≥ 0.50 | - | Missing many failures or too many false alarms |
| **Poor** | < 0.50 | - | Not production-ready |

### Common Issues & Fixes

#### Low Hit Rate (< 0.7)
**Problem**: Missing many test failures

**Solutions**:
- Expand test selection coverage (suggest more tests)
- Improve file-to-test mapping
- Add fallback rules (e.g., always suggest core tests)

#### Low Precision (< 0.5)
**Problem**: Too many irrelevant suggestions

**Solutions**:
- Tighten relevance filtering
- Improve ranking algorithm
- Use historical failure data to prioritize

#### High False Alarm Rate (> 0.6)
**Problem**: Wasting CI time on unnecessary tests

**Solutions**:
- Add confidence thresholds (only suggest high-confidence tests)
- Use historical pass rates to filter
- Implement smarter dependency analysis

#### Low MRR (< 0.3)
**Problem**: Failures buried deep in suggestions

**Solutions**:
- Improve ranking algorithm
- Boost recently-failing tests
- Use failure frequency in ranking

## Example Output

```
======================================================================
TEST SELECTION CALIBRATION ANALYSIS
======================================================================

Loaded 42 calibration records.

Predictions recorded:  42
Outcomes recorded:     42
Calibration records:   42

TEST SELECTION METRICS:

  Precision@5:      0.820  (of top 5 suggestions, how many relevant?)
  Recall:           0.756  (of failures, what % did we catch?)
  Hit Rate:         0.881  (% predictions catching at least one failure)
  MRR:              0.623  (rank of first failure in suggestions)
  False Alarm Rate: 0.342  (suggested tests that didn't fail)
  Coverage:         0.901  (% of all failures caught)

Status: GOOD

RECOMMENDATIONS:
  ✓ Test selection quality is good. Continue monitoring.

METRIC INTERPRETATION:
  Good thresholds:
    • Precision@5 ≥ 0.7  (most suggestions are useful)
    • Recall ≥ 0.8       (catch most failures)
    • Hit Rate ≥ 0.85    (rarely miss failures entirely)
    • MRR ≥ 0.5          (failures ranked in top 2 on average)
    • Coverage ≥ 0.9     (catch 90%+ of all failures)
```

## Integration Points

### With Git Hooks

Record predictions in `prepare-commit-msg`:

```bash
# .git/hooks/prepare-commit-msg
python scripts/record_test_prediction.py \
    --changed-files "$(git diff --cached --name-only)" \
    --commit-msg "$1"
```

Record outcomes in `post-commit` or CI:

```bash
# After pytest run
python scripts/record_test_outcome.py \
    --prediction-id "$PREDICTION_ID" \
    --junit-xml test-results.xml
```

### With CI/CD

In GitHub Actions:

```yaml
- name: Record Test Prediction
  run: |
    python scripts/record_test_prediction.py \
      --changed-files "${{ steps.changed.outputs.files }}"

- name: Run Tests
  run: pytest --junit-xml=results.xml

- name: Record Test Outcome
  if: always()
  run: |
    python scripts/record_test_outcome.py \
      --prediction-id "${{ env.PREDICTION_ID }}" \
      --junit-xml results.xml
```

## Cold Start Behavior

When no calibration data exists:

```
No test calibration data available yet.
Test calibration tracks how well TestExpert predicts which tests to run.

To generate test calibration data:
  1. Record test predictions via TestCalibrationTracker.record_prediction()
  2. Run tests and record outcomes via TestCalibrationTracker.record_outcome()
  3. Re-run this command
```

This is expected and normal for new systems. Calibration quality improves over time as more predictions are recorded.

## API Reference

### TestCalibrationTracker

**Methods**:
- `record_prediction()`: Record a test selection prediction
- `record_outcome()`: Record test execution results
- `load_all()`: Load all predictions, outcomes, and calibration records
- `get_metrics()`: Compute aggregate metrics
- `get_summary()`: Get comprehensive summary with recommendations
- `format_report()`: Generate human-readable report

### Data Classes

- `TestPrediction`: Recorded prediction awaiting evaluation
- `TestOutcome`: Actual test execution results
- `TestCalibrationRecord`: Matched prediction-outcome pair
- `TestCalibrationMetrics`: Aggregate accuracy metrics

## Best Practices

1. **Record every prediction**: Even if confidence is low
2. **Always record outcomes**: Needed for calibration
3. **Include metadata**: Helps with debugging and analysis
4. **Monitor regularly**: Check calibration weekly
5. **Act on recommendations**: Address issues flagged by the system
6. **Iterate**: Use metrics to improve test selection algorithms

## See Also

- [File Prediction Calibration](calibration-tracker.md) - Similar metrics for file predictions
- [TestExpert Architecture](testexpert-architecture.md) - How TestExpert works
- [Sprint 6 Goals](../tasks/CURRENT_SPRINT.md) - TestExpert activation sprint
