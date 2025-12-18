# Sprint 6 Goal 3: CI Results Integration for TestExpert Training

## Implementation Summary

Successfully implemented CI result integration for TestExpert training in the Hubris MoE system. This allows the TestExpert to learn from historical test failure patterns captured by CI.

## Changes Made

### 1. Core Transformation Logic (`scripts/hubris_cli.py`)

#### Helper Function: `_is_test_file(file_path: str) -> bool`
- Detects test files by checking for:
  - `'test'` in filename
  - `'tests/'` prefix
  - `'_test.py'` suffix

#### Main Transformation: `transform_commit_for_test_expert(commit) -> Dict`
- **Input**: Commit dict with optional `ci_result` field
- **Output**: Commit dict with `test_results` field added

**Heuristic Logic:**
```python
if ci_result == 'fail' and test_files_changed:
    test_results = {
        'failed': [changed_test_files],
        'passed': [],
        'source': 'ci_heuristic'
    }
elif ci_result == 'pass' and test_files_changed:
    test_results = {
        'failed': [],
        'passed': [changed_test_files],
        'source': 'ci_heuristic'
    }
```

**Key Features:**
- Non-destructive: Uses `commit.copy()` to avoid mutating original data
- Respects existing `test_results` if present
- Only adds data when meaningful (test files + CI status)
- Handles both `'files'` and `'files_changed'` keys

#### Modified: `load_commit_data(limit, include_ci=False)`
- Added `include_ci` parameter
- Transforms each commit when `include_ci=True`
- Backward compatible (defaults to `False`)

#### Modified: `cmd_train(args)`
- Uses `include_ci` flag from args
- Reports CI data statistics when enabled:
  ```
  Commits: 500
  CI data: 324 commits
  Test results: 187 commits (for TestExpert)
  ```

### 2. CLI Interface

#### New Flag: `--include-ci`
```bash
python scripts/hubris_cli.py train --commits 500 --include-ci
```

**Usage:**
- Optional flag for `train` command
- Enables CI-to-test-results transformation
- Recommended for TestExpert training when CI data is available

### 3. Test Coverage

Created `tests/unit/test_hubris_cli_ci_integration.py` with 8 tests:

| Test | Coverage |
|------|----------|
| `test_is_test_file` | Test file detection patterns |
| `test_transform_with_no_ci_data` | Pass-through without CI |
| `test_transform_with_ci_fail_and_test_files` | Map failures to test files |
| `test_transform_with_ci_pass_and_test_files` | Map passes to test files |
| `test_transform_with_ci_fail_no_test_files` | Skip when no tests changed |
| `test_transform_preserves_existing_test_results` | Don't overwrite existing data |
| `test_transform_with_files_changed_key` | Handle both file key variants |
| `test_transform_immutability` | Verify non-destructive transform |

**Result:** All 8 tests pass ✅

## Integration with Existing Systems

### CI Workflow (`.github/workflows/ci.yml`)
- **Already captures** CI results via `ml-ci-capture` job
- Sets `ci_result` field on commit data
- Runs after `coverage-report` job completes
- Uses environment variables:
  - `CI_RESULT`: pass/fail/error
  - `CI_COVERAGE`: coverage percentage
  - `GITHUB_SHA`: commit hash

### ML Data Collector (`scripts/ml_data_collector.py`)
- **Already stores** CI data in `.git-ml/commits/*.json`
- `ci_autocapture()` function reads GitHub Actions environment
- `update_commit_ci_result()` writes CI status to commit files

### TestExpert (`scripts/hubris/experts/test_expert.py`)
- **Already consumes** `test_results` format:
  ```python
  test_results = commit.get('test_results', {})
  if test_results:
      failed_tests = test_results.get('failed', [])
      for source in source_files:
          for failed_test in failed_tests:
              test_failure_patterns[source][failed_test] += 1
  ```

## Heuristic Limitations

### What the Heuristic Assumes:
1. **When CI fails + test files changed** → Those tests likely failed
2. **When CI passes + test files changed** → Those tests passed

### Limitations:
- **Cannot identify specific failing tests** if no test files changed
- **May be inaccurate** if:
  - Tests failed due to environment issues (flaky tests)
  - CI failure was in a different stage (linting, type checking)
  - Test file was refactored but didn't actually break

### Future Enhancements:
1. **Parse pytest output** from CI logs to get exact test names:
   ```
   FAILED tests/test_processor.py::test_compute_all
   FAILED tests/test_query.py::test_search
   ```
2. **Store test output** in `ci_details` field
3. **Distinguish failure types**:
   - Test failures vs. lint errors vs. import errors
4. **Track flaky tests** via pass/fail history

## Usage Guide

### For Training with CI Data:

```bash
# Train with last 500 commits, including CI results
python scripts/hubris_cli.py train --commits 500 --include-ci

# Train with all available data
python scripts/hubris_cli.py train --include-ci
```

### For Training without CI Data (default):

```bash
# Standard training (no CI transformation)
python scripts/hubris_cli.py train --commits 500
```

### Checking Results:

```bash
# View TestExpert stats
python scripts/hubris_cli.py stats --expert test

# Evaluate TestExpert accuracy
python scripts/hubris_cli.py evaluate --commits 20

# Suggest tests for changes
python scripts/hubris_cli.py suggest-tests --staged
```

## Verification

### Manual Testing:
```bash
# 1. Run training with CI flag
python scripts/hubris_cli.py train --commits 100 --include-ci

# 2. Check stats show CI data was used
# Expected output:
#   Commits: 100
#   CI data: XX commits
#   Test results: XX commits (for TestExpert)

# 3. Verify TestExpert learned failure patterns
python scripts/hubris_cli.py stats --expert test
```

### Unit Tests:
```bash
# Run CI integration tests
python -m unittest tests/unit/test_hubris_cli_ci_integration.py -v

# Expected: 8 tests pass
```

## Data Flow Diagram

```
┌─────────────────────┐
│  GitHub Actions CI  │
│  (ml-ci-capture)    │
└──────────┬──────────┘
           │ ci_result: pass/fail
           ▼
┌─────────────────────┐
│  .git-ml/commits/   │
│  {hash}_commit.json │
│  - files: [...]     │
│  - ci_result: fail  │
└──────────┬──────────┘
           │
           │ hubris train --include-ci
           ▼
┌─────────────────────┐
│ transform_commit_   │
│ for_test_expert()   │
│                     │
│ Heuristic:          │
│ CI fail + tests     │
│ changed = failed    │
└──────────┬──────────┘
           │ test_results: {failed: [...]}
           ▼
┌─────────────────────┐
│   TestExpert.train()│
│                     │
│ Learns patterns:    │
│ source_file ->      │
│   failed_tests      │
└─────────────────────┘
```

## Success Criteria ✅

- [x] Transform commit CI data into `test_results` format
- [x] Add `--include-ci` flag to `train` command
- [x] Maintain backward compatibility (flag optional)
- [x] Handle edge cases (no CI data, no test files, existing test_results)
- [x] Add comprehensive unit tests (8 tests, all passing)
- [x] Document heuristic limitations
- [x] Verify integration with existing CI workflow
- [x] Non-destructive transformation (doesn't mutate original data)

## Next Steps

### Immediate:
1. **Collect more CI data** - Run CI on several commits to build dataset
2. **Train TestExpert** - Use `--include-ci` flag with ≥100 commits
3. **Evaluate accuracy** - Check if TestExpert predictions improve

### Future:
1. **Parse pytest output** - Get exact test names from CI logs
2. **Enhance heuristic** - Distinguish test failures from other CI errors
3. **Track flaky tests** - Identify tests that fail intermittently
4. **Add test coverage signals** - Use coverage data to predict affected tests

## References

- TestExpert implementation: `scripts/hubris/experts/test_expert.py`
- CI workflow: `.github/workflows/ci.yml`
- ML data collector: `scripts/ml_data_collector.py`
- Hubris CLI: `scripts/hubris_cli.py`
