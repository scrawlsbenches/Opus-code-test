# ML Pre-Commit File Suggestions

## Overview

The ML pre-commit hook uses file prediction to suggest potentially missing files based on your commit message. It runs automatically before each commit and warns you if high-confidence predictions aren't staged.

**Key Features:**
- Non-blocking by default (warns but allows commit)
- Configurable via environment variables
- Uses trained ML model from commit history
- Skips automatically if model isn't trained
- Integrates seamlessly with existing workflow

## Installation

The hook is automatically installed when you run:

```bash
python scripts/ml_data_collector.py install-hooks
```

This creates `.git/hooks/prepare-commit-msg` which calls `scripts/ml-precommit-suggest.sh`.

### Manual Installation

If you need to install manually:

```bash
# Copy the hook script
cp scripts/ml-precommit-suggest.sh .git/hooks/prepare-commit-msg

# Make it executable
chmod +x .git/hooks/prepare-commit-msg
```

## Usage

Once installed, the hook runs automatically during `git commit`:

```bash
# 1. Stage some files
git add cortical/analysis.py

# 2. Commit with a message
git commit -m "feat: Add PageRank optimization"

# 3. Hook analyzes message and suggests missing files
```

## Example Output

### Warning (Non-Blocking)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 ML File Prediction Suggestion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on your commit message, these files might need changes:

  • tests/test_analysis.py                       (confidence: 0.823)
  • cortical/processor/compute.py                (confidence: 0.654)
  • docs/architecture.md                         (confidence: 0.512)

Staged files:
  ✓ cortical/analysis.py

ℹ️  Tip: Review the suggestions above. To block commits with missing files:
   export ML_SUGGEST_BLOCKING=1

To disable suggestions: export ML_SUGGEST_ENABLED=0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### No Suggestions

If all predicted files are staged or predictions are below threshold, the hook exits silently without output.

## Configuration

Control behavior via environment variables:

### ML_SUGGEST_ENABLED

Enable/disable suggestions (default: `1`).

```bash
# Disable for current session
export ML_SUGGEST_ENABLED=0
git commit -m "message"

# Disable for single commit
ML_SUGGEST_ENABLED=0 git commit -m "message"

# Disable permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export ML_SUGGEST_ENABLED=0' >> ~/.bashrc
```

### ML_SUGGEST_THRESHOLD

Confidence threshold for warnings (default: `0.5`).

```bash
# Only warn about very confident predictions (>0.8)
export ML_SUGGEST_THRESHOLD=0.8
git commit -m "message"

# Show more suggestions (lower threshold)
export ML_SUGGEST_THRESHOLD=0.3
git commit -m "message"
```

### ML_SUGGEST_BLOCKING

Block commit if missing files (default: `0`).

```bash
# Make hook blocking (abort commit if missing files)
export ML_SUGGEST_BLOCKING=1
git commit -m "message"
```

**Note:** Blocking mode is useful for enforcing completeness but can be disruptive during rapid prototyping.

### ML_SUGGEST_TOP_N

Number of predictions to check (default: `5`).

```bash
# Check top 10 predictions
export ML_SUGGEST_TOP_N=10
git commit -m "message"

# Only check top 3 predictions
export ML_SUGGEST_TOP_N=3
git commit -m "message"
```

## When Hook Runs

The hook runs in these scenarios:

✅ **Regular commits:** `git commit -m "message"`
✅ **Interactive commits:** `git commit` (opens editor)
✅ **With staged files:** Normal workflow

❌ **Merge commits:** Skipped (too noisy)
❌ **Amend commits:** Skipped (message already used)
❌ **Rebase commits:** Skipped (interactive context)
❌ **Empty commits:** Skipped (no staged files)
❌ **ML data commits:** Skipped (`data: ML tracking data`)

## Requirements

- **Trained model:** Model must exist at `.git-ml/models/file_prediction.json`
- **Git repository:** Must be in a git repository
- **Staged files:** At least one file must be staged
- **Project detection:** `scripts/ml_file_prediction.py` must exist

If any requirement isn't met, the hook silently exits without output.

## Training the Model

Before suggestions work, you need to train the model:

```bash
# Train on commit history
python scripts/ml_file_prediction.py train

# Check model stats
python scripts/ml_file_prediction.py stats
```

The model learns from your commit history:
- Commit type patterns (feat, fix, docs, etc.)
- File co-occurrence patterns
- Keyword-to-file associations
- Task reference patterns

**Recommendation:** Retrain periodically as your codebase evolves.

## Testing

Test the prediction manually without committing:

```bash
# Predict for a hypothetical commit message
python scripts/ml_file_prediction.py predict "feat: Add authentication"

# Check with seed files (co-occurrence boost)
python scripts/ml_file_prediction.py predict "fix: Update validation" \
  --seed cortical/tokenizer.py cortical/config.py

# Test with different top-N
python scripts/ml_file_prediction.py predict "test: Add coverage" --top 10
```

## Troubleshooting

### Hook doesn't run

1. Check if hook is installed:
   ```bash
   ls -l .git/hooks/prepare-commit-msg
   ```

2. Reinstall hooks:
   ```bash
   python scripts/ml_data_collector.py install-hooks
   ```

3. Check if suggestions are disabled:
   ```bash
   echo $ML_SUGGEST_ENABLED
   ```

### Model not found

Train the model:
```bash
python scripts/ml_file_prediction.py train
```

### Wrong predictions

The model learns from your commit history:
- More commits = better predictions
- Consistent patterns = more accurate
- Retrain after major codebase changes

```bash
# Check model stats
python scripts/ml_file_prediction.py stats

# Evaluate accuracy
python scripts/ml_file_prediction.py evaluate --split 0.2
```

### Hook is too slow

The hook typically takes <1s. If it's slow:

1. Check model size:
   ```bash
   ls -lh .git-ml/models/file_prediction.json
   ```

2. Reduce predictions checked:
   ```bash
   export ML_SUGGEST_TOP_N=3
   ```

3. Disable AI metadata (faster but less accurate):
   ```bash
   python scripts/ml_file_prediction.py predict "message" --no-ai-meta
   ```

## Integration with Workflow

### Recommended Setup (Developer Friendly)

```bash
# Add to ~/.bashrc or ~/.zshrc
export ML_SUGGEST_ENABLED=1        # Enable suggestions
export ML_SUGGEST_THRESHOLD=0.5    # Balanced threshold
export ML_SUGGEST_BLOCKING=0       # Non-blocking (warn only)
export ML_SUGGEST_TOP_N=5          # Check top 5 predictions
```

### Strict Mode (CI/CD Enforcement)

```bash
# Add to CI environment or strict development branches
export ML_SUGGEST_ENABLED=1
export ML_SUGGEST_THRESHOLD=0.7    # High confidence only
export ML_SUGGEST_BLOCKING=1       # Block incomplete commits
export ML_SUGGEST_TOP_N=10         # Check more predictions
```

### Minimal Mode (Rapid Prototyping)

```bash
# Disable during rapid iteration
export ML_SUGGEST_ENABLED=0
```

## Technical Details

### Hook Flow

1. User runs `git commit -m "message"`
2. Git prepares commit message file
3. Git calls `.git/hooks/prepare-commit-msg` with message file path
4. Hook script calls `scripts/ml-precommit-suggest.sh`
5. Script reads commit message from file
6. Script gets list of staged files
7. Script runs `ml_file_prediction.py predict`
8. Script parses predictions and compares with staged files
9. Script displays warnings for missing high-confidence files
10. Script exits 0 (allow commit) or 1 (block commit)

### Performance

- **Typical runtime:** 0.5-1.5 seconds
- **Model loading:** ~0.3s (cached in memory for Python)
- **Prediction:** ~0.2s
- **AI metadata:** ~0.5s (optional, can be disabled)

### Accuracy Metrics

Based on evaluation with 80/20 train/test split (403 commits):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MRR | 0.43 | First correct prediction at position ~2-3 |
| Recall@10 | 0.48 | Half of actual files in top 10 predictions |
| Precision@1 | 0.31 | 31% of top predictions are correct |

**Interpretation:** The model is good at suggesting relevant files, but not perfect. This is why it's non-blocking by default—it's a helpful hint, not a strict rule.

## See Also

- [ML File Prediction](../scripts/ml_file_prediction.py) - Prediction model implementation
- [ML Data Collection](ml-data-collection.md) - Training data collection
- [Git Hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) - Git hooks documentation
