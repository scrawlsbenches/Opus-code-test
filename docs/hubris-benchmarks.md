# Hubris MoE Benchmarks

## Overview

Hubris benchmarks measure **prediction accuracy**, not execution performance. The Mixture of Experts (MoE) system includes multiple specialized experts that predict development outcomes:

- **FileExpert**: Predicts which files to modify for a task
- **TestExpert**: Predicts which tests to run given code changes
- **ErrorExpert**: Predicts potential errors and failure patterns
- **EpisodeExpert**: Predicts task sequences and development workflows

These benchmarks track how well each expert's predictions match real outcomes, enabling continuous improvement through calibration feedback loops.

**Key distinction**: Unlike traditional benchmarks measuring speed or throughput, Hubris benchmarks measure prediction quality using metrics like Precision@K, Recall, Hit Rate, and MRR (Mean Reciprocal Rank).

---

## TestExpert Metrics

### Current Training Stats

As of 2025-12-18, TestExpert is trained on:

| Metric | Value | Description |
|--------|-------|-------------|
| **Commits analyzed** | 868 | Git commits used for training |
| **Source files** | 4,946 | Unique source files in training data |
| **Test files** | 174 | Unique test files identified |
| **Source→Test mappings** | 138,196 | Total source-to-test relationships learned |
| **Model version** | 1.0.0 | Current TestExpert version |

**Training approach**: TestExpert learns from git history by analyzing which source files co-change with which test files in commits. It builds a co-occurrence matrix weighted by recency and change frequency.

### Metrics Definitions

TestExpert predictions are evaluated using these metrics:

#### Precision@5
**Definition**: Of the top 5 suggested tests, what fraction were relevant (ran or failed)?

**Formula**: `relevant_in_top_5 / min(5, total_suggested)`

**Use case**: Measures suggestion quality - how much CI time would be wasted running irrelevant tests.

#### Recall
**Definition**: Of all tests that failed, what fraction did TestExpert suggest running?

**Formula**: `suggested_failures / total_failures`

**Use case**: Measures failure detection - are we catching the bugs that exist?

#### Hit Rate
**Definition**: What fraction of predictions caught at least one failing test?

**Formula**: `predictions_with_failures / total_predictions`

**Use case**: Measures reliability - how often do we completely miss failures?

#### MRR (Mean Reciprocal Rank)
**Definition**: Average reciprocal rank of the first failing test in suggestions.

**Formula**: `mean(1 / rank_of_first_failure)`

**Use case**: Measures ranking quality - are failures ranked high in the suggestion list?

#### Coverage
**Definition**: Fraction of the full test suite suggested by TestExpert.

**Formula**: `tests_suggested / total_tests`

**Use case**: Measures efficiency - can we safely skip tests to save CI time?

### Target Thresholds

Target thresholds will be refined as calibration data accumulates. Initial estimates:

| Quality Level | Hit Rate | Precision@5 | Recall | Coverage |
|---------------|----------|-------------|--------|----------|
| **Excellent** | ≥ 0.95 | ≥ 0.80 | ≥ 0.90 | < 0.30 |
| **Good** | ≥ 0.80 | ≥ 0.60 | ≥ 0.70 | < 0.50 |
| **Needs Improvement** | < 0.80 | < 0.60 | < 0.70 | ≥ 0.50 |

**Rationale:**
- **Hit Rate**: Should rarely miss failures entirely (< 5% miss rate for excellent)
- **Precision@5**: Top suggestions should be highly relevant (80%+ for excellent)
- **Recall**: Should catch most failures (90%+ for excellent)
- **Coverage**: Should reduce test burden significantly (< 30% of tests for excellent)

---

## Calibration Data Collection

### How Calibration Works

TestExpert calibration uses a prediction→outcome feedback loop:

1. **Prediction**: TestExpert suggests tests to run for a set of changed files
2. **Execution**: Tests are actually run (manually or via CI)
3. **Outcome**: Results are recorded (which tests ran, which passed/failed)
4. **Calibration**: Prediction is compared to outcome to compute accuracy metrics

### Storage

All calibration data is stored in `.git-ml/predictions/` (gitignored):

```
.git-ml/predictions/
├── test_predictions.jsonl   # Test selection predictions
├── test_outcomes.jsonl      # Test execution results
└── test_calibration.jsonl   # Matched prediction-outcome pairs
```

### Recording Data

**Manual recording** (for local development):

```python
from scripts.hubris.test_calibration_tracker import TestCalibrationTracker

tracker = TestCalibrationTracker()

# Record a prediction
tracker.record_prediction(
    prediction_id="pred-123",
    changed_files=["cortical/processor.py"],
    suggested_tests=["tests/test_processor.py::test_foo"],
    confidence_scores={"tests/test_processor.py::test_foo": 0.95}
)

# After running tests, record the outcome
tracker.record_outcome(
    prediction_id="pred-123",
    tests_run=["tests/test_processor.py::test_foo"],
    tests_passed=["tests/test_processor.py::test_foo"],
    tests_failed=[]
)
```

**Automatic recording** (via CI integration - future):
- Pre-test hook records prediction
- Post-test hook records outcome
- Calibration computed automatically

### Viewing Calibration Data

```bash
# View current calibration metrics
python scripts/hubris_cli.py calibration --tests

# View raw prediction/outcome data
cat .git-ml/predictions/test_predictions.jsonl | jq .
cat .git-ml/predictions/test_outcomes.jsonl | jq .
cat .git-ml/predictions/test_calibration.jsonl | jq .
```

---

## Baseline Metrics

Baseline metrics will be populated as calibration data accumulates. This table tracks TestExpert performance over time.

| Date | Commits | Mappings | Hit Rate | Precision@5 | Recall | MRR | Coverage |
|------|---------|----------|----------|-------------|--------|-----|----------|
| 2025-12-18 | 868 | 138,196 | TBD | TBD | TBD | TBD | TBD |

**Notes:**
- First baseline requires ≥10 calibration samples for statistical validity
- Metrics will be computed monthly as calibration data grows
- Significant model changes (version bumps) will start new baseline rows

---

## Running Benchmarks

### Quick Status Check

```bash
# View all expert statistics
python scripts/hubris_cli.py stats

# View TestExpert-specific stats
python scripts/hubris_cli.py stats --expert test
```

### Evaluation on Historical Data

```bash
# Evaluate on last 50 commits
python scripts/hubris_cli.py evaluate --commits 50

# Evaluate on specific date range
python scripts/hubris_cli.py evaluate --since 2025-12-01 --until 2025-12-18
```

### Calibration Analysis

```bash
# View test selection calibration metrics
python scripts/hubris_cli.py calibration --tests

# View calibration over time (requires ≥10 samples)
python scripts/hubris_cli.py calibration --tests --time-series

# Export calibration data for analysis
python scripts/hubris_cli.py calibration --tests --export calibration.csv
```

---

## Model Versioning

### Storage Location

All Hubris expert models are stored in:

```
.git-ml/models/hubris/
├── file_expert.json       # FileExpert model
├── test_expert.json       # TestExpert model (21 MB)
├── error_expert.json      # ErrorExpert model
├── episode_expert.json    # EpisodeExpert model
└── credit_ledger.json     # Credit system state
```

**Note**: Models (`.git-ml/models/`) can be regenerated from commit history. However, **chat transcripts and action logs are NOT regeneratable** - see `docs/ml-ephemeral-architecture.md` for the migration plan to preserve this data.

### Version Tracking

Each expert model includes version metadata:

```json
{
  "expert_id": "test_expert",
  "version": "1.0.0",
  "created_at": "2025-12-18T20:59:57.975520",
  "trained_on_commits": 868,
  "git_hash": "abc123..."
}
```

**Versioning scheme**: `major.minor.patch`
- **Major**: Breaking changes to model format or prediction API
- **Minor**: New features, significant algorithm changes
- **Patch**: Bug fixes, minor improvements

### Rebuilding Models

```bash
# Rebuild all expert models from git history
python scripts/hubris_cli.py train --all

# Rebuild specific expert
python scripts/hubris_cli.py train --expert test

# Rebuild with specific commit range
python scripts/hubris_cli.py train --expert test --commits 500
```

---

## See Also

- **Calibration Details**: [docs/test-calibration-metrics.md](test-calibration-metrics.md)
- **Test Feedback Integration**: [docs/test-feedback-integration.md](test-feedback-integration.md)
- **ML Data Collection**: [docs/ml-data-collection-knowledge-transfer.md](ml-data-collection-knowledge-transfer.md)
- **Ephemeral Architecture**: [docs/ml-ephemeral-architecture.md](ml-ephemeral-architecture.md)
