# Sprint 6 Goal 5 Complete: TestExpert Feedback Loop

**Date:** 2025-12-18
**Sprint:** Sprint 6 - TestExpert Activation
**Goal:** Add TestExpert to feedback loop
**Status:** ✅ Complete

## Summary

Implemented a complete post-test feedback hook system that captures pytest results and feeds them into TestExpert's failure pattern learning. This closes the feedback loop: TestExpert predicts tests → tests run → results update TestExpert's knowledge.

## Deliverables

### 1. Post-Test Hook (`scripts/ml-post-test-hook.sh`)

Bash hook that:
- Auto-detects pytest output files (`.pytest-output.txt`, `.pytest-results.xml`)
- Validates output freshness (< 5 minutes to avoid reprocessing)
- Calls Python parser with appropriate flags
- Provides formatted output with success/failure reporting
- Respects `ML_COLLECTION_ENABLED` environment variable

**Key Features:**
- Graceful handling of missing files
- Staleness prevention (won't reprocess old results)
- Environment variable configuration
- Compatible with both verbose and JUnit XML output

### 2. Python Parser (`scripts/hubris/test_feedback.py`)

Comprehensive Python script (15KB, 400+ lines) that:
- Parses pytest verbose output with regex
- Parses JUnit XML with ElementTree
- Auto-detects available output format
- Gets changed files from git
- Filters source files from test files
- Updates FeedbackProcessor with test outcomes
- Updates TestExpert's failure patterns
- Saves updated model to `.git-ml/models/test_expert.json`

**API:**
```python
# Parse output
results = parse_pytest_output('output.txt')
results = parse_junit_xml('results.xml')

# Process feedback
summary = process_test_feedback(
    test_results=results,
    changed_files=['cortical/query.py'],
    dry_run=False,
    verbose=True
)
```

**Command Line:**
```bash
python test_feedback.py --parse-output .pytest-output.txt
python test_feedback.py --parse-xml .pytest-results.xml
python test_feedback.py --auto
python test_feedback.py --auto --dry-run --verbose
```

### 3. Unit Tests (`tests/unit/test_test_feedback.py`)

Comprehensive test suite with 9 tests covering:
- ✅ Pytest verbose output parsing
- ✅ JUnit XML parsing
- ✅ ERROR and SKIPPED status handling
- ✅ Source file filtering
- ✅ TestExpert failure pattern updates
- ✅ Incremental failure counting
- ✅ No failures (all passing) scenario
- ✅ End-to-end dry run
- ✅ End-to-end with save

**Test Results:**
```
Ran 9 tests in 0.007s
OK
```

### 4. Documentation

**Comprehensive Guide:** [`docs/test-feedback-integration.md`](../../docs/test-feedback-integration.md)
- Overview and architecture
- Usage examples (manual and automatic)
- Configuration options
- Data flow diagram
- TestExpert model structure
- Output format specifications
- Testing instructions
- Troubleshooting guide
- Sprint 6 completion checklist
- Next steps

**Quick Reference:** [`docs/test-feedback-quick-ref.md`](../../docs/test-feedback-quick-ref.md)
- One-line usage commands
- File locations table
- Command reference
- Configuration variables
- Output example
- Troubleshooting table

## Technical Details

### Data Flow

```
pytest output (.txt or .xml)
    ↓
ml-post-test-hook.sh (auto-detects format)
    ↓
test_feedback.py (parses results)
    ├─ Parse test pass/fail status
    ├─ Get changed files from git
    └─ Filter source files
    ↓
FeedbackProcessor.process_test_outcome()
    ├─ Update credit ledger
    └─ Resolve pending predictions
    ↓
TestExpert model update
    ├─ test_failure_patterns[source_file][failed_test] += 1
    └─ Save to .git-ml/models/test_expert.json
```

### TestExpert Model Structure

```json
{
  "model_data": {
    "test_failure_patterns": {
      "cortical/query/search.py": {
        "tests/test_query.py::test_search_basic": 3,
        "tests/test_query.py::test_search_advanced": 1
      },
      "cortical/analysis.py": {
        "tests/test_analysis.py::test_pagerank": 2
      }
    }
  }
}
```

**Interpretation:** When `cortical/query/search.py` changes, `test_search_basic` has historically failed 3 times. TestExpert uses this signal to predict which tests are most likely to fail when that file is modified.

### Integration Points

1. **FeedbackProcessor** - Credit tracking for expert predictions
2. **TestExpert** - Failure pattern learning
3. **SessionStart hook** - Already runs tests (could auto-process)
4. **Stop hook** - Could add automatic feedback processing
5. **Git hooks** - File change detection

## Verification

### Manual Test

```bash
# Create sample output
cat > .pytest-output.txt << 'EOF'
tests/test_foo.py::test_basic PASSED
tests/test_foo.py::test_advanced FAILED
tests/test_bar.py::test_simple PASSED
EOF

# Run hook
./scripts/ml-post-test-hook.sh

# Output:
# 🧪 Processing Test Results
# Parsed 3 tests from .pytest-output.txt
# Processing 3 test results...
#   Passed: 2
#   Failed: 1
#   Changed files: 48
#   Source files: 1
#     - scripts/analyze_corpus_balance.py
# Created new TestExpert
#   Added 1 failure pattern entries
#   Saved updated expert to .git-ml/models/test_expert.json
# ✅ Test feedback processed successfully
```

### Unit Test

```bash
python3 -m unittest tests.unit.test_test_feedback -v

# Output:
# test_process_test_feedback_dry_run ... ok
# test_process_test_feedback_with_save ... ok
# test_incremental_failure_patterns ... ok
# test_no_failures ... ok
# test_update_failure_patterns ... ok
# test_parse_junit_xml ... ok
# test_parse_verbose_output ... ok
# test_parse_with_error_status ... ok
# test_get_source_files ... ok
# Ran 9 tests in 0.007s
# OK
```

## What Was Learned

### 1. Parser Flexibility Matters

Supporting both verbose output and JUnit XML was important because:
- Verbose output is easier to read and debug
- JUnit XML is more structured and reliable
- Different CI systems prefer different formats
- Users have different workflow preferences

### 2. Freshness Validation Prevents Issues

The 5-minute staleness check prevents:
- Reprocessing old test results multiple times
- Accumulating stale failure patterns
- Confusion from outdated signals

This was crucial for integration with hooks that run automatically.

### 3. Graceful Degradation

Making everything optional and fail-safe:
- Missing output files → skip silently
- No changed files → auto-detect from git
- Parsing errors → log warning, continue
- All tests passed → success with 0 patterns

This makes the hook robust in CI environments and during development.

### 4. Dry Run Mode Is Essential

The `--dry-run` flag was invaluable for:
- Testing parsers without side effects
- Debugging format issues
- Verifying git file detection
- Understanding what would change

### 5. Incremental Learning Works

The failure pattern accumulation (incrementing counts) enables:
- Learning from repeated failures
- Weighting frequent failures higher
- Building confidence over time
- Statistical significance

## Next Steps (Sprint 7?)

### 1. Automatic Integration

Add to `scripts/ml-session-capture-hook.sh` Stop hook:
```bash
# Process test results for TestExpert feedback
if [[ -f .pytest-output.txt ]]; then
    ./scripts/ml-post-test-hook.sh 2>/dev/null || true
fi
```

### 2. Historical Backfill

Train TestExpert on past failures:
```bash
# Extract historical test failures from git history
python scripts/backfill_test_failures.py --commits 500
```

### 3. Test Selection

Use predictions to run only relevant tests:
```bash
# Predict tests for changed files
python scripts/suggest_tests.py --changed-files cortical/query.py
# → tests/test_query.py (confidence: 0.92)
# → tests/integration/test_search.py (confidence: 0.78)

# Run only predicted tests
pytest $(python scripts/suggest_tests.py --changed-files cortical/query.py --top 5)
```

### 4. Metrics Dashboard

Visualize failure patterns and prediction accuracy:
```bash
python scripts/test_expert_metrics.py
# → Prediction accuracy: 82%
# → Most unstable files: [...]
# → Most fragile tests: [...]
```

### 5. CI Integration

Add to `.github/workflows/ci.yml`:
```yaml
- name: Process test feedback
  if: always()  # Run even if tests fail
  run: |
    python scripts/hubris/test_feedback.py --auto
```

## Sprint 6 Goal 5 Completion Checklist

✅ **Created post-test feedback hook**
- [x] Bash hook: `scripts/ml-post-test-hook.sh`
- [x] Python parser: `scripts/hubris/test_feedback.py`
- [x] Auto-detection of output formats
- [x] Freshness validation (< 5 minutes)
- [x] Environment variable configuration

✅ **Integrated with FeedbackProcessor**
- [x] Calls `process_test_outcome(test_results)`
- [x] Updates credit ledger
- [x] Resolves pending predictions
- [x] Handles missing predictions gracefully

✅ **Updated TestExpert model**
- [x] Failure patterns accumulate: `source_file → {test: count}`
- [x] Model persists to `.git-ml/models/test_expert.json`
- [x] Incremental learning (counts increment)
- [x] Load existing model and update

✅ **Comprehensive testing**
- [x] Unit tests for parsing (verbose + XML)
- [x] Unit tests for source file filtering
- [x] Unit tests for TestExpert updates
- [x] Unit tests for incremental counting
- [x] End-to-end integration tests
- [x] 9 tests, all passing

✅ **Documentation**
- [x] Integration guide: `docs/test-feedback-integration.md`
- [x] Quick reference: `docs/test-feedback-quick-ref.md`
- [x] Usage examples and workflows
- [x] Troubleshooting guide
- [x] Architecture diagrams

✅ **Verification**
- [x] Manual testing with sample data
- [x] Verified model saves correctly
- [x] Verified incremental updates work
- [x] Verified staleness prevention
- [x] Verified both output formats

## Conclusion

Sprint 6 Goal 5 is **complete**. The test feedback loop is fully implemented, tested, and documented. TestExpert can now learn from real test outcomes, building knowledge about which tests fail when specific files are modified.

The system is production-ready and can be:
1. Used manually after test runs
2. Integrated into SessionStart/Stop hooks
3. Added to CI workflows
4. Extended with historical backfill

**Impact:** This closes the critical feedback loop needed for TestExpert to become a valuable prediction tool. As the model accumulates data, it will become increasingly accurate at predicting which tests to run for a given code change.

---

**Tags:** `sprint6`, `testexpert`, `feedback-loop`, `ml-training`, `test-prediction`
**Related:** [[test-expert.py]], [[feedback-collector.py]], [[ml-data-collection.md]]
