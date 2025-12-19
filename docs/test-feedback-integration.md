# Test Feedback Integration Guide

## Overview

The **Test Feedback Hook** captures pytest results and feeds them into TestExpert's failure pattern learning system. This closes the feedback loop: TestExpert predicts tests → tests run → results update TestExpert's knowledge.

## Components

### 1. `scripts/hubris/test_feedback.py`

Python script that:
- Parses pytest output (verbose or JUnit XML)
- Extracts test pass/fail status
- Gets changed files from git
- Updates TestExpert's `test_failure_patterns` with failures
- Feeds results to FeedbackProcessor for credit tracking

### 2. `scripts/ml-post-test-hook.sh`

Bash hook that:
- Auto-detects pytest output files
- Validates output freshness (< 5 minutes old)
- Calls `test_feedback.py` to process results
- Integrates with SessionStart/Stop hooks

## Usage

### Manual Integration

Run tests and capture output, then process:

```bash
# Option 1: Verbose output
pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
./scripts/ml-post-test-hook.sh

# Option 2: JUnit XML
pytest tests/ --junitxml=.pytest-results.xml
./scripts/ml-post-test-hook.sh

# Option 3: Direct Python call
pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
python scripts/hubris/test_feedback.py --parse-output .pytest-output.txt
```

### Automatic Integration (SessionStart/Stop Hooks)

The SessionStart and Stop hooks already run tests. To add feedback processing:

#### Option A: Modify Stop Hook

Add to `scripts/ml-session-capture-hook.sh` after the test suite runs:

```bash
# After the test suite block (around line 104)
# Process test results for TestExpert feedback
if [[ -f .pytest-output.txt ]]; then
    python3 scripts/ml-post-test-hook.sh 2>/dev/null || true
fi
```

#### Option B: Standalone Hook

Keep as standalone script and call it explicitly when needed:

```bash
# After making changes and running tests
pytest tests/ -v 2>&1 | tee .pytest-output.txt
./scripts/ml-post-test-hook.sh
```

### Configuration

Environment variables:

- `ML_COLLECTION_ENABLED=0` - Disable all ML hooks (default: 1)
- `PYTEST_OUTPUT_FILE=/path/to/output.txt` - Override output file location
- `ML_FEEDBACK_MAX_AGE=300` - Max age in seconds for output (default: 300 = 5 minutes)

## How It Works

### Data Flow

```
1. Tests run → pytest output saved
   ├─ Verbose: .pytest-output.txt
   └─ JUnit XML: .pytest-results.xml

2. Hook detects output file
   └─ Validates freshness (< 5 min)

3. Python parser extracts results
   ├─ Test name → pass/fail mapping
   └─ Changed files from git

4. FeedbackProcessor updates
   ├─ Credit ledger (expert rewards)
   └─ Prediction resolution

5. TestExpert learns
   ├─ Failure patterns updated
   │   source_file → {failed_test: count}
   └─ Model saved to .git-ml/models/test_expert.json
```

### TestExpert Model Updates

The `test_failure_patterns` field accumulates failure history:

```json
{
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
```

**Meaning**: When `cortical/query/search.py` changes, `test_search_basic` has failed 3 times historically. TestExpert uses this to predict which tests are most likely to fail.

## Output Formats

### Pytest Verbose Output

```
tests/test_foo.py::test_basic PASSED                    [ 10%]
tests/test_foo.py::test_advanced FAILED                 [ 20%]
```

Parsed with regex: `r'^([\w/\.]+\.py::\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)'`

### JUnit XML

```xml
<testcase classname="tests.test_foo" name="test_basic" time="0.001"/>
<testcase classname="tests.test_foo" name="test_advanced" time="0.002">
  <failure message="assertion failed">AssertionError</failure>
</testcase>
```

Parsed with `xml.etree.ElementTree`, checking for `<failure>`, `<error>`, `<skipped>` elements.

## Testing

Run the test suite:

```bash
python3 -m unittest tests.unit.test_test_feedback -v
```

Test coverage includes:
- ✅ Pytest verbose output parsing
- ✅ JUnit XML parsing
- ✅ Source file filtering (removes test files)
- ✅ TestExpert failure pattern updates
- ✅ Incremental failure counting
- ✅ End-to-end integration with dry run
- ✅ Model save/load

## Example Session

```bash
# 1. Make code changes
vim cortical/query/search.py

# 2. Run tests with output capture
pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
# Output shows 2 failures

# 3. Process feedback
./scripts/ml-post-test-hook.sh

# Output:
# 🧪 Processing Test Results
#    Output file: .pytest-output.txt
# Parsed 47 tests from .pytest-output.txt
# Processing 47 test results...
#   Passed: 45
#   Failed: 2
#   Changed files: 3
#   Source files: 1
#     - cortical/query/search.py
# Loaded TestExpert from .git-ml/models/test_expert.json
#   Added 2 failure pattern entries
#   Saved updated expert to .git-ml/models/test_expert.json
# ✅ Test feedback processed successfully

# 4. View updated patterns
python3 -c "
import json
data = json.load(open('.git-ml/models/test_expert.json'))
patterns = data['model_data']['test_failure_patterns']
print(json.dumps(patterns, indent=2))
"
```

## Troubleshooting

### Hook doesn't run

**Check:**
- Is `ML_COLLECTION_ENABLED=1`?
- Does `.pytest-output.txt` or `.pytest-results.xml` exist?
- Is the output file recent (< 5 minutes)?
- Is the hook executable? (`chmod +x scripts/ml-post-test-hook.sh`)

### No failure patterns added

**Possible reasons:**
- All tests passed (nothing to learn)
- No source files changed (only test files)
- Parsing failed (check file format)

### Parsing errors

**Verify format:**
```bash
# For verbose output
python3 scripts/hubris/test_feedback.py --parse-output .pytest-output.txt --dry-run --verbose

# For JUnit XML
python3 scripts/hubris/test_feedback.py --parse-xml .pytest-results.xml --dry-run --verbose
```

## Sprint 6 Goal Completion

This implementation completes **Goal 5** of Sprint 6 (TestExpert Activation):

✅ **Created post-test feedback hook**
- Bash hook: `scripts/ml-post-test-hook.sh`
- Python parser: `scripts/hubris/test_feedback.py`
- Test suite: `tests/unit/test_test_feedback.py`

✅ **Integrated with FeedbackProcessor**
- Calls `process_test_outcome(test_results)`
- Updates credit ledger automatically
- Resolves pending predictions

✅ **Updated TestExpert model**
- Failure patterns accumulate: `source_file → {test: count}`
- Model persists to `.git-ml/models/test_expert.json`
- Incremental learning (counts increment)

✅ **Documented and tested**
- Comprehensive unit tests (9 tests, all passing)
- Integration guide with examples
- End-to-end workflow documented

## Next Steps

1. **Integrate into SessionStart/Stop hooks** - Add automatic processing
2. **Add to .claude/settings.local.json** - Hook configuration
3. **Train TestExpert on historical data** - Backfill failure patterns from git history
4. **Add metrics dashboard** - Visualize failure patterns and predictions
5. **Implement test selection** - Use predictions to run only relevant tests

## Related Files

- `scripts/hubris/feedback_collector.py` - FeedbackProcessor API
- `scripts/hubris/experts/test_expert.py` - TestExpert implementation
- `scripts/ml-session-start-hook.sh` - SessionStart hook (runs tests)
- `scripts/ml-session-capture-hook.sh` - Stop hook (could add feedback processing)
- `.claude/settings.local.json` - Hook configuration
