# Test Selection Calibration Metrics - Implementation Summary

**Sprint 6, Goal 4: Create test selection accuracy metrics**

## What Was Implemented

### 1. Core TestCalibrationTracker Module
**File**: `/home/user/Opus-code-test/scripts/hubris/test_calibration_tracker.py`

A complete test selection calibration system that tracks how accurately TestExpert predicts which tests should be run.

**Key Components**:
- `TestPrediction` - Records test selection predictions
- `TestOutcome` - Records actual test execution results
- `TestCalibrationRecord` - Matched prediction-outcome pairs for analysis
- `TestCalibrationMetrics` - Aggregate accuracy metrics
- `TestCalibrationTracker` - Main tracker class with recording and analysis

**Metrics Implemented**:
1. **Precision@5**: Of top 5 suggestions, how many were relevant?
2. **Recall**: Of tests that failed, what fraction did we suggest?
3. **Hit Rate**: Did we catch at least one failing test?
4. **MRR (Mean Reciprocal Rank)**: Average rank of first failing test
5. **False Alarm Rate**: Fraction of suggested tests that didn't fail
6. **Coverage**: Fraction of all failures caught across predictions

**Status Classification**:
- Excellent: Hit rate ≥ 0.95, Precision@5 ≥ 0.8
- Good: Hit rate ≥ 0.85, Precision@5 ≥ 0.6
- Acceptable: Hit rate ≥ 0.70, Precision@5 ≥ 0.4
- Needs Attention: Hit rate ≥ 0.50
- Poor: Hit rate < 0.50

### 2. CLI Integration
**File**: `/home/user/Opus-code-test/scripts/hubris_cli.py`

Added `--tests` flag to the `calibration` command:

```bash
# View test selection calibration
python scripts/hubris_cli.py calibration --tests

# JSON output
python scripts/hubris_cli.py calibration --tests --json
```

**Routing Logic**:
- Default (`calibration`): File prediction calibration (existing)
- With `--tests` flag: Test selection calibration (new)

### 3. Demo Script
**File**: `/home/user/Opus-code-test/scripts/hubris/test_calibration_demo.py`

Interactive demo showing 3 scenarios:
1. Good prediction - catches most failures
2. Poor prediction - misses failures
3. Partial success - catches some failures

Run with: `python scripts/hubris/test_calibration_demo.py`

### 4. Unit Tests
**File**: `/home/user/Opus-code-test/tests/unit/test_test_calibration.py`

Comprehensive test suite with 10 tests covering:
- Recording predictions and outcomes
- Metric calculations (Precision@5, Recall, Hit Rate, MRR)
- Edge cases (no failures, invalid confidence)
- Data persistence
- Status classification

**All tests pass**: ✓ 10/10

### 5. Documentation
**File**: `/home/user/Opus-code-test/docs/test-calibration-metrics.md`

Complete documentation including:
- Metric definitions with formulas
- Usage examples (recording predictions/outcomes)
- Interpretation guidelines
- Status thresholds
- Common issues and solutions
- Integration with Git hooks and CI/CD
- API reference

## Data Storage

All data stored in `.git-ml/predictions/` (gitignored):

```
.git-ml/predictions/
├── test_predictions.jsonl   # Test selection predictions
├── test_outcomes.jsonl       # Test execution results
└── test_calibration.jsonl    # Matched pairs for analysis
```

## API Usage

### Recording a Prediction

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
    metadata={"commit_message": "Add authentication"}
)
```

### Recording an Outcome

```python
outcome = tracker.record_outcome(
    prediction_id="pred_12345",
    tests_run=["tests/test_auth.py::test_login", ...],
    tests_failed=["tests/test_auth.py::test_login"],
    tests_passed=["tests/test_auth.py::test_logout", ...],
    metadata={"ci_run": "12345", "duration_seconds": 45}
)
```

### Viewing Metrics

```python
# Load all data
tracker.load_all()

# Get metrics
metrics = tracker.get_metrics()
print(f"Hit Rate: {metrics.hit_rate:.3f}")
print(f"Status: {metrics.get_status()}")

# Full report
print(tracker.format_report())
```

## CLI Output Example

```
======================================================================
TEST SELECTION CALIBRATION ANALYSIS
======================================================================

Loaded 3 calibration records.

Predictions recorded:  3
Outcomes recorded:     3
Calibration records:   3

TEST SELECTION METRICS:

  Precision@5:      1.000  (of top 5 suggestions, how many relevant?)
  Recall:           0.500  (of failures, what % did we catch?)
  Hit Rate:         0.667  (% predictions catching at least one failure)
  MRR:              0.667  (rank of first failure in suggestions)
  False Alarm Rate: 0.783  (suggested tests that didn't fail)
  Coverage:         0.500  (% of all failures caught)

Status: NEEDS_ATTENTION

RECOMMENDATIONS:
  ⚠️  Low hit rate - many test failures not predicted. Consider expanding test selection coverage.
  ℹ️  High false alarm rate - suggested tests mostly pass. This wastes CI time. Improve relevance filtering.

METRIC INTERPRETATION:
  Good thresholds:
    • Precision@5 ≥ 0.7  (most suggestions are useful)
    • Recall ≥ 0.8       (catch most failures)
    • Hit Rate ≥ 0.85    (rarely miss failures entirely)
    • MRR ≥ 0.5          (failures ranked in top 2 on average)
    • Coverage ≥ 0.9     (catch 90%+ of all failures)
```

## Cold Start Behavior

When no data exists, shows helpful guidance:

```
No test calibration data available yet.
Test calibration tracks how well TestExpert predicts which tests to run.

To generate test calibration data:
  1. Record test predictions via TestCalibrationTracker.record_prediction()
  2. Run tests and record outcomes via TestCalibrationTracker.record_outcome()
  3. Re-run this command
```

## Key Design Decisions

### 1. Separate from File Prediction Calibration
- File predictions and test predictions have different semantics
- File prediction: accuracy of which files will change
- Test prediction: accuracy of which tests will fail
- Keeping them separate allows focused optimization

### 2. Precision@5 Instead of Precision@K
- Most CI systems run a limited set of "quick tests" first
- Top 5 is a realistic constraint for fast feedback
- Can be extended with `@K` variants if needed

### 3. Hit Rate as Primary Metric
- Critical metric: "Did we catch ANY failure?"
- High hit rate prevents "completely missed" scenarios
- Complements recall (which measures how many we caught)

### 4. False Alarm Rate for CI Efficiency
- Tracks wasted CI resources (tests that didn't need to run)
- Helps balance coverage vs efficiency
- Important for large test suites where CI time matters

### 5. Atomic File Writes
- All JSONL files use atomic writes (temp + rename)
- Prevents corruption from crashes or interrupts
- Safe for concurrent access from parallel processes

## Testing Coverage

- ✓ Prediction recording
- ✓ Outcome recording
- ✓ Precision@5 calculation
- ✓ Recall calculation
- ✓ Hit rate calculation
- ✓ MRR calculation
- ✓ No failures scenario (perfect recall)
- ✓ Status classification
- ✓ Data persistence
- ✓ Validation (invalid confidence)

## Next Steps (Future Work)

### 1. Integration with TestExpert
Connect to actual TestExpert predictions when they're implemented:

```python
# In TestExpert
predictions = test_expert.predict(changed_files)
tracker.record_prediction(
    prediction_id=pred_id,
    suggested_tests=predictions,
    confidence=test_expert.confidence,
    changed_files=changed_files
)
```

### 2. CI Integration
Add hooks to record outcomes automatically:

```bash
# .github/workflows/test.yml
- name: Record test outcome
  if: always()
  run: python scripts/record_test_outcome.py --junit-xml results.xml
```

### 3. Historical Trend Analysis
Track calibration quality over time:
- Weekly/monthly trends
- Degradation detection
- Automatic alerts when metrics drop

### 4. Per-Expert Calibration
Track calibration for different test selection strategies:
- File-based expert
- History-based expert
- Dependency-based expert

### 5. Confidence Calibration
Similar to file predictions, track whether test selection confidence matches actual accuracy.

## Files Created/Modified

### New Files
1. `scripts/hubris/test_calibration_tracker.py` (560 lines)
2. `scripts/hubris/test_calibration_demo.py` (200 lines)
3. `tests/unit/test_test_calibration.py` (290 lines)
4. `docs/test-calibration-metrics.md` (450 lines)

### Modified Files
1. `scripts/hubris_cli.py`
   - Added import for `TestCalibrationTracker`
   - Added routing logic in `cmd_calibration()`
   - Added new `cmd_calibration_tests()` function
   - Added `--tests` flag to calibration parser

**Total**: ~1,500 lines of implementation + tests + documentation

## Verification

### Run Demo
```bash
python scripts/hubris/test_calibration_demo.py
```

### Run Tests
```bash
python -m unittest tests.unit.test_test_calibration -v
# Result: 10 tests, all passing
```

### Check CLI
```bash
# Cold start (no data)
python scripts/hubris_cli.py calibration --tests

# With data (after running demo)
python scripts/hubris_cli.py calibration --tests
python scripts/hubris_cli.py calibration --tests --json
```

### Verify File Calibration Still Works
```bash
# Should show existing file prediction calibration
python scripts/hubris_cli.py calibration
```

## Success Criteria Met

✓ **Defined "accurate" for test selection**
  - Precision@5: relevance of suggestions
  - Recall: fraction of failures caught
  - Hit Rate: at least one failure caught

✓ **Created test-specific metrics in calibration system**
  - Extends existing calibration patterns
  - Separate storage in `.git-ml/predictions/test_*.jsonl`

✓ **Added `--tests` flag to calibration command**
  - Routes to test calibration when present
  - Maintains backward compatibility with file calibration

✓ **Data structures for predictions and outcomes**
  - `TestPrediction` and `TestOutcome` dataclasses
  - Validation, serialization, persistence

✓ **Metric calculation with good/bad thresholds**
  - Status classification (excellent → poor)
  - Actionable recommendations
  - Color-coded CLI output

✓ **Cold start handling**
  - Graceful behavior with no data
  - Clear instructions for getting started

✓ **Comprehensive testing**
  - 10 unit tests, all passing
  - Edge cases covered
  - Demo script for visualization

✓ **Documentation**
  - Complete metric definitions
  - Usage examples
  - Integration patterns
  - Best practices

## Impact

This implementation provides the **foundation for TestExpert quality monitoring**. Once TestExpert is actively making predictions, this calibration system will:

1. **Detect prediction quality issues** early
2. **Guide algorithm improvements** with specific metrics
3. **Build confidence** through transparent measurement
4. **Enable A/B testing** of different test selection strategies
5. **Track improvement** over time as the system learns

The metrics are interpretable, actionable, and follow the same patterns as the existing file prediction calibration, making them easy to adopt.
