# Sprint 6: TestExpert Activation - Knowledge Transfer

**Sprint ID:** sprint-006-test-expert
**Session:** 0c9WR
**Date:** 2025-12-18
**Status:** ✅ Complete

---

## Executive Summary

Sprint 6 activated TestExpert - the Hubris MoE expert specialized in predicting which tests to run for code changes. Prior to this sprint, TestExpert existed but was dormant (no real training data flowing through). Now it's fully operational with:

- **397 source-to-test mappings** learned from 500 commits
- **37 module-to-test mappings** for coarse-grained predictions
- **CLI command** `suggest-tests` for interactive use
- **Post-test feedback hook** for continuous learning
- **Test selection metrics** for calibration tracking

---

## What Was Delivered

### 1. CLI Command: `suggest-tests`

```bash
# Suggest tests for specific files
python scripts/hubris_cli.py suggest-tests --files cortical/query/search.py

# Suggest tests for staged changes
python scripts/hubris_cli.py suggest-tests --staged

# Suggest tests for modified files
python scripts/hubris_cli.py suggest-tests --modified
```

**Output Example:**
```
🧪 Suggested Tests for Changed Files:

Changed files (2):
  • cortical/query/search.py
  • cortical/analysis.py

Suggested tests (by confidence):
   1. tests/test_processor.py                    1.000
   2. tests/test_layers.py                       0.524
   3. tests/test_semantics.py                    0.509
   ...

Coverage estimate: 100% of changed files have test mappings
Status: ✓ Good coverage
```

### 2. Training Data Integration

**Problem Solved:** TestExpert's `train()` method expected `files` field but commit data used `files_changed`.

**Fix:** Updated `test_expert.py` line 303:
```python
files = commit.get('files', []) or commit.get('files_changed', [])
```

**Also fixed:** `hubris_cli.py` `load_commit_data()` now reads from:
1. `.git-ml/tracked/commits.jsonl` (JSONL format - 863 commits)
2. Falls back to `.git-ml/commits/*.json` (legacy format)

### 3. CI Results Integration

Added `--include-ci` flag to training:
```bash
python scripts/hubris_cli.py train --commits 500 --include-ci
```

**Transform function** `transform_commit_for_test_expert()`:
- When `ci_result: 'fail'` + test files changed → maps to `test_results.failed`
- When `ci_result: 'pass'` + test files changed → maps to `test_results.passed`

### 4. Post-Test Feedback Hook

**Files created:**
- `scripts/ml-post-test-hook.sh` - Bash entry point
- `scripts/hubris/test_feedback.py` - Python parser and updater

**Workflow:**
```
pytest output (.txt or .xml)
    ↓
ml-post-test-hook.sh
    ↓
test_feedback.py --parse-output .pytest-output.txt
    ↓
FeedbackProcessor.process_test_outcome()
    ↓
TestExpert model update (test_failure_patterns)
```

### 5. Test Selection Calibration Metrics

**New file:** `scripts/hubris/test_calibration_tracker.py`

**Metrics tracked:**
| Metric | Description | Good Threshold |
|--------|-------------|----------------|
| Precision@5 | Of top 5 suggestions, how many relevant? | ≥0.7 |
| Recall | Of failures, what % did we catch? | ≥0.8 |
| Hit Rate | % predictions catching at least one failure | ≥0.85 |
| MRR | Mean reciprocal rank of first failure | ≥0.5 |
| Coverage | % of all failures caught | ≥0.9 |

**CLI:**
```bash
python scripts/hubris_cli.py calibration --tests
```

---

## Architecture Changes

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/hubris/test_feedback.py` | Parse pytest output, update TestExpert | ~400 |
| `scripts/hubris/test_calibration_tracker.py` | Track test prediction accuracy | ~560 |
| `scripts/hubris/test_calibration_demo.py` | Interactive demo | ~200 |
| `scripts/ml-post-test-hook.sh` | Bash hook for post-test | ~150 |
| `tests/unit/test_test_feedback.py` | Unit tests for parser | ~280 |
| `tests/unit/test_test_calibration.py` | Unit tests for calibration | ~290 |
| `docs/test-feedback-integration.md` | Integration guide | - |
| `docs/test-feedback-quick-ref.md` | Quick reference | - |
| `docs/test-calibration-metrics.md` | Metrics documentation | - |

### Files Modified

| File | Change |
|------|--------|
| `scripts/hubris_cli.py` | Added `suggest-tests`, `--tests` flag, CI integration |
| `scripts/hubris/experts/test_expert.py` | Fixed field name compatibility |
| `scripts/hubris/README.md` | Added Sprint 6 CLI commands |
| `tasks/CURRENT_SPRINT.md` | Marked Sprint 6 complete |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     TESTEXPERT DATA FLOW                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ .git-ml/tracked │     │ CI Results      │     │ Pytest Output   │
│ commits.jsonl   │     │ (ci_result)     │     │ .pytest-output  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │ Training              │ --include-ci          │ Post-test hook
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TestExpert.train()                        │
│                                                                 │
│  Model Data:                                                    │
│  • source_to_tests: 397 mappings                               │
│  • module_to_tests: 37 mappings                                │
│  • test_failure_patterns: (populated by feedback)              │
│  • test_cochange: co-occurrence patterns                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ suggest-tests command   │
                    │ or predict() API        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Ranked test suggestions │
                    │ with confidence scores  │
                    └─────────────────────────┘
```

---

## Key Technical Decisions

### 1. JSONL vs Individual JSON Files

**Decision:** Support both formats with JSONL preferred.

**Rationale:** The tracked commits data (863 entries) is in JSONL format which is git-friendly and efficient. Legacy individual JSON files are still supported for backwards compatibility.

### 2. Field Name Compatibility

**Decision:** Check both `files` and `files_changed` in training.

**Rationale:** Different data sources use different field names. Rather than require transformation everywhere, the expert handles both.

### 3. CI Heuristic for Failure Patterns

**Decision:** When CI fails and test files are changed, assume those tests failed.

**Rationale:** We don't have granular per-test pass/fail from CI (only aggregate). This heuristic provides signal where perfect data isn't available.

### 4. Separate Calibration Tracker for Tests

**Decision:** Created `TestCalibrationTracker` separate from file prediction calibration.

**Rationale:** Test selection has different metrics (precision/recall/hit rate) than file prediction (accuracy/MRR). Separate trackers allow specialized analysis.

---

## Known Limitations

1. **Granular CI Test Results:** Current CI only captures `test_passed: bool`. Specific test failures require parsing pytest output post-hoc.

2. **Cold Start:** With no historical test failures, `test_failure_patterns` starts empty. Predictions rely on naming conventions and co-change patterns initially.

3. **pytest Dependency:** Post-test hook assumes pytest output format. Other test frameworks need additional parsers.

---

## Future Work (Sprint 7+)

1. **CI Job Enhancement:** Add step to capture per-test results in GitHub Actions
2. **Historical Backfill:** Train on historical pytest output from CI logs
3. **Test Prioritization:** Use confidence to order test execution
4. **Selective Test Runs:** `pytest $(suggest-tests --staged --format pytest)`

---

## Verification Checklist

- [x] `suggest-tests` command works with `--files`
- [x] `suggest-tests` command works with `--staged`
- [x] `suggest-tests` command works with `--modified`
- [x] Training loads from JSONL format
- [x] TestExpert has 397+ source-to-test mappings
- [x] `calibration --tests` shows test metrics
- [x] Unit tests pass (19 tests)
- [x] README updated with new commands
- [x] Sprint status marked complete

---

## Commands Quick Reference

```bash
# Train TestExpert
python scripts/hubris_cli.py train --commits 500

# Suggest tests for files
python scripts/hubris_cli.py suggest-tests --files FILE1 FILE2

# Suggest tests for staged changes
python scripts/hubris_cli.py suggest-tests --staged

# View test calibration
python scripts/hubris_cli.py calibration --tests

# Process pytest output (after running tests)
pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
python scripts/hubris/test_feedback.py --parse-output .pytest-output.txt
```

---

## Contact

Sprint completed by Claude (session 0c9WR) on 2025-12-18.
Sub-agents delegated for parallel implementation of CLI, feedback hook, CI integration, and metrics.
