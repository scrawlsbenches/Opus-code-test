# Test Feedback Hook - Quick Reference

## One-Line Usage

```bash
pytest tests/ -v 2>&1 | tee .pytest-output.txt && ./scripts/ml-post-test-hook.sh
```

## What It Does

1. Parses pytest output (pass/fail per test)
2. Gets changed files from git
3. Updates TestExpert's failure pattern knowledge
4. Updates credit ledger for expert predictions

## File Locations

| File | Purpose |
|------|---------|
| `scripts/ml-post-test-hook.sh` | Bash hook (entry point) |
| `scripts/hubris/test_feedback.py` | Python parser and updater |
| `tests/unit/test_test_feedback.py` | Unit tests |
| `.pytest-output.txt` | Pytest verbose output (auto-detected) |
| `.pytest-results.xml` | JUnit XML output (auto-detected) |
| `.git-ml/models/test_expert.json` | TestExpert model (updated) |

## Commands

```bash
# Manual run after tests
pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
./scripts/ml-post-test-hook.sh

# With JUnit XML
pytest tests/ --junitxml=.pytest-results.xml
./scripts/ml-post-test-hook.sh

# Direct Python call
python scripts/hubris/test_feedback.py --parse-output .pytest-output.txt

# Auto-detect output
python scripts/hubris/test_feedback.py --auto

# Dry run (no changes)
python scripts/hubris/test_feedback.py --auto --dry-run --verbose

# Run tests
python3 -m unittest tests.unit.test_test_feedback -v
```

## Configuration

```bash
# Disable feedback processing
export ML_COLLECTION_ENABLED=0

# Override output file
export PYTEST_OUTPUT_FILE=/path/to/output.txt

# Change max age (default: 300 seconds)
export ML_FEEDBACK_MAX_AGE=600
```

## Output Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧪 Processing Test Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Output file: .pytest-output.txt
Parsed 47 tests from .pytest-output.txt
Processing 47 test results...
  Passed: 45
  Failed: 2
  Changed files: 3
  Source files: 1
    - cortical/query/search.py

Loaded TestExpert from .git-ml/models/test_expert.json
  Added 2 failure pattern entries
  Saved updated expert to .git-ml/models/test_expert.json

============================================================
Summary:
  Total tests: 47
  Passed: 45
  Failed: 2
  Failure patterns added: 2
  Expert updated: True
============================================================
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Test feedback processed successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## View Failure Patterns

```bash
python3 -c "
import json
data = json.load(open('.git-ml/models/test_expert.json'))
patterns = data['model_data']['test_failure_patterns']
print(json.dumps(patterns, indent=2))
"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Hook doesn't run | Check `ML_COLLECTION_ENABLED=1` |
| No output file found | Run pytest with `-v` or `--junitxml` |
| Output too old | Re-run tests (< 5 minutes) |
| No patterns added | All tests passed (nothing to learn) |
| Parsing errors | Check format: `--dry-run --verbose` |

## Integration Points

- **SessionStart hook**: Runs tests at session start
- **Stop hook**: Could add automatic processing
- **FeedbackProcessor**: Credit tracking integration
- **TestExpert**: Failure pattern learning

## Learn More

See [`docs/test-feedback-integration.md`](test-feedback-integration.md) for complete documentation.
