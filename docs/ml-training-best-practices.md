# ML Training Best Practices

## Overview

This guide covers best practices for training and maintaining ML models in the Cortical Text Processor project. The current ML system includes a file prediction model that learns from commit history to suggest which files to modify for a given task.

**Target Audience:** Developers training ML models from project data

**Prerequisites:**
- Understanding of ML basics (training/test splits, overfitting, metrics)
- Familiarity with `scripts/ml_file_prediction.py` and `scripts/ml_data_collector.py`
- At least 100 commits in `.git-ml/tracked/commits.jsonl`

**Quick Links:**
- Training command: `python scripts/ml_file_prediction.py train --evaluate`
- Data collection: See [ML Data Collection Knowledge Transfer](ml-data-collection-knowledge-transfer.md)
- Pre-commit integration: See [ML Pre-Commit Suggestions](ml-precommit-suggestions.md)

---

## Table of Contents

1. [Data Quality Guidelines](#data-quality-guidelines)
2. [Training Workflow](#training-workflow)
3. [Performance Optimization](#performance-optimization)
4. [Common Pitfalls](#common-pitfalls)
5. [Evaluation Guidelines](#evaluation-guidelines)
6. [Integration Best Practices](#integration-best-practices)
7. [Model Versioning](#model-versioning)
8. [Troubleshooting](#troubleshooting)

---

## Data Quality Guidelines

### What Makes Good Training Data

**High-quality commit data has these characteristics:**

1. **Diverse commit types** - Mix of features, fixes, refactors, docs
2. **Balanced file distribution** - Not dominated by a few frequently-changed files
3. **Recent and relevant** - Reflects current codebase structure
4. **Meaningful messages** - Descriptive commit messages, not "wip" or "fix"
5. **Non-merge commits** - Focus on actual development work
6. **Appropriate granularity** - Not too large (100+ files) or too small (typo fixes)

**Check data quality:**

```bash
# Show commit statistics
python scripts/ml_file_prediction.py stats

# View distribution of commit types
python scripts/ml_data_collector.py stats
```

### Data Filtering Best Practices

The ML data collector automatically filters:
- ✅ Merge commits (marked with `is_merge: true`)
- ✅ ML tracking commits (messages starting with "data: ML")
- ✅ Deleted files (migrated paths during training)
- ✅ Sensitive data (see `REDACTION_PATTERNS` in `ml_data_collector.py`)

**Manual filtering may be needed for:**

```python
# Example: Filter commits by file count
examples = load_commit_data(filter_deleted=True)
filtered = [ex for ex in examples if 1 <= len(ex.files_changed) <= 20]
print(f"Filtered {len(examples)} → {len(filtered)} commits")
```

**When to filter:**
- Single-file commits with no context (e.g., "fix typo")
- Mass refactors touching 50+ files (not representative of typical work)
- Auto-generated commits (e.g., dependency updates)
- Commits with only config changes (e.g., `.gitignore`, `pyproject.toml`)

**When NOT to filter:**
- Test-only commits (tests are valuable training data!)
- Documentation commits (docs often accompany code changes)
- Small bug fixes (these are common real-world patterns)

### Handling Edge Cases

**File renames and moves:**

The system uses `FILE_PATH_MIGRATIONS` to map old paths to new:

```python
# Example from ml_file_prediction.py
FILE_PATH_MIGRATIONS = {
    'cortical/processor.py': [
        'cortical/processor/__init__.py',
        'cortical/processor/core.py',
        'cortical/processor/compute.py',
    ],
}
```

**Add mappings when refactoring:**

1. Before major refactoring, note current file paths
2. After refactoring, update `FILE_PATH_MIGRATIONS` in `ml_file_prediction.py`
3. Retrain model to incorporate migration

**Deleted files:**

Training automatically filters deleted files by default:

```python
examples = load_commit_data(filter_deleted=True)  # Default
```

To see impact of deleted files:

```bash
# Train without filtering (not recommended)
python scripts/ml_file_prediction.py train  # Includes deleted files in stats
```

### Data Quality Checklist

Before training, verify:

- [ ] At least 100 commits collected (50 minimum for file prediction)
- [ ] Commit type distribution is reasonable (not 90% "feat" commits)
- [ ] Top 10 most-changed files account for <30% of total changes
- [ ] Commits span at least 2 weeks of development time
- [ ] No sensitive data in commit messages (run `validate` command)
- [ ] File path migrations updated for recent refactorings

```bash
# Validate collected data
python scripts/ml_data_collector.py validate

# Check for sensitive patterns
python scripts/ml_data_collector.py redact-test --text "$(git log -1 --format=%B)"
```

---

## Training Workflow

### When to Train

**Initial training:**
```bash
# First time setup with evaluation
python scripts/ml_file_prediction.py train --evaluate --save-version
```

**When to retrain:**

| Trigger | Frequency | Command |
|---------|-----------|---------|
| After major refactoring | Once per refactor | `train --evaluate` |
| New files/modules added | Every 50+ commits | `train` |
| Model staleness warning | When prompted | `train --save-version` |
| Performance degradation | As needed | `train --evaluate` |
| Weekly maintenance | Optional | `train` |

**Check model staleness:**

```bash
# Shows commits since last training
python scripts/ml_file_prediction.py stats

# Or during prediction
python scripts/ml_file_prediction.py predict "Add authentication" --verbose
# Shows: "⚠️  Model is 25 commits behind HEAD. Consider retraining."
```

**Automatic staleness detection:**

Pre-commit hook warns when model is >10 commits behind:

```bash
export ML_SUGGEST_THRESHOLD=0.5  # Default
git commit -m "feat: Add feature"
# Output: "⚠️ Model is 15 commits behind. Run: python scripts/ml_file_prediction.py train"
```

### Training Procedure

**Step 1: Verify data integrity**

```bash
# Check for issues before training
python scripts/ml_data_collector.py validate

# View recent commits
python scripts/ml_data_collector.py stats
```

**Step 2: Train with evaluation**

```bash
# 80/20 train/test split
python scripts/ml_file_prediction.py train --evaluate --save-version

# Custom split ratio
python scripts/ml_file_prediction.py train --evaluate --split 0.1  # 90/10 split
```

**Step 3: Review metrics**

Expected metrics for good performance:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| MRR (Mean Reciprocal Rank) | >0.4 | 0.2-0.4 | <0.2 |
| Recall@10 | >0.5 | 0.3-0.5 | <0.3 |
| Precision@1 | >0.3 | 0.15-0.3 | <0.15 |

**Example output:**

```
Model trained and saved to .git-ml/models/file_prediction.json
  Git commit: a1b2c3d4e5f6
  Total commits: 403
  Unique files: 127
  Commit types: 8
  Keywords: 234

  Evaluation metrics:
    MRR: 0.4285
    Recall@5: 0.3819
    Recall@10: 0.4801
    Precision@1: 0.3125

  Saved to history: .git-ml/models/history/model_20251216_143052_a1b2c3.json
```

**Step 4: Test predictions**

```bash
# Sanity check predictions
python scripts/ml_file_prediction.py predict "Fix authentication bug"
python scripts/ml_file_prediction.py predict "Add tests for query module"
python scripts/ml_file_prediction.py predict "Update documentation"
```

### Validation Strategies

**1. Hold-out validation (default):**

```bash
# 80/20 split - standard for model evaluation
python scripts/ml_file_prediction.py evaluate --split 0.2
```

**2. Manual validation:**

Test on known commit messages:

```bash
# Get recent commit messages
git log --oneline -10

# Test prediction for each
for msg in "$(git log --format=%s -10)"; do
    echo "Message: $msg"
    python scripts/ml_file_prediction.py predict "$msg" --top 5
    echo "---"
done
```

**3. A/B comparison:**

Compare old vs new model:

```bash
# List model versions
python scripts/ml_file_prediction.py history

# Compare predictions
python scripts/ml_file_prediction.py compare \
    "Add user authentication" \
    --version1 .git-ml/models/history/model_20251210_*.json \
    --top 10
```

**4. Real-world validation:**

Most reliable - use the model during development:

```bash
# Enable pre-commit suggestions
export ML_SUGGEST_ENABLED=1
export ML_SUGGEST_THRESHOLD=0.5

# Develop for a week, note:
# - False positives (suggested files not needed)
# - False negatives (needed files not suggested)
# - Irrelevant suggestions
```

### Training Scripts and Automation

**Automated weekly retraining (cron job):**

```bash
# Add to crontab
0 2 * * 0 cd /path/to/project && python scripts/ml_file_prediction.py train --evaluate --save-version 2>&1 | tee logs/ml_train_$(date +\%Y\%m\%d).log
```

**Training with CI/CD:**

```yaml
# .github/workflows/ml-training.yml
name: ML Model Training

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for commit data

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Train model
        run: |
          python scripts/ml_file_prediction.py train --evaluate --save-version
          python scripts/ml_file_prediction.py stats

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: ml-model
          path: .git-ml/models/file_prediction.json
```

---

## Performance Optimization

### Feature Selection

The file prediction model uses these feature types:

| Feature Type | Weight | Use Case |
|--------------|--------|----------|
| Commit type patterns | 2.0× | "feat:", "fix:", "docs:" prefixes |
| Keyword matching | 1.5× | Words in commit message |
| File co-occurrence | 3.0× | Files changed together |
| Semantic similarity | 0.5× | Message-to-commit similarity |
| AI metadata | 1.0-2.0× | Function/section matching |

**Tuning feature weights:**

Edit `scripts/ml_file_prediction.py` function `predict_files()`:

```python
# Line ~1062 - Commit type scoring
file_scores[f] += tf * idf * 2.0  # Increase to 3.0 for stronger type signal

# Line ~1072 - Keyword scoring
file_scores[f] += tf * idf * 1.5  # Decrease to 1.0 to reduce keyword dominance

# Line ~1084 - Co-occurrence scoring
file_scores[f] += similarity * 3.0  # This is usually the strongest signal

# Line ~1093 - Semantic similarity
file_scores[filepath] += sim * 0.5  # Increase to 1.0 if messages are descriptive
```

**Feature importance analysis:**

```python
# Add debug output to predict_files()
print(f"Commit type score: {type_score}")
print(f"Keyword score: {keyword_score}")
print(f"Co-occurrence score: {cooc_score}")
print(f"Semantic score: {semantic_score}")
```

### Hyperparameter Tuning

**Key hyperparameters:**

1. **TF-IDF weights** (line ~1060-1072):
   - Controls term frequency vs inverse document frequency balance
   - Default: `tf * idf` - balanced
   - High-frequency files: Use `tf * idf^2` to penalize common files
   - Rare files: Use `tf^2 * idf` to boost distinctive patterns

2. **Frequency penalty** (line ~1154):
   - Prevents always suggesting frequently-changed files
   - Default: `1.0 - (freq / max_freq) * 0.3` (30% penalty)
   - Increase penalty: `* 0.5` (50% penalty for high-frequency files)
   - Decrease penalty: `* 0.1` (10% penalty)

3. **Test/train split** (line ~1386):
   - Default: 0.2 (20% test, 80% train)
   - More data: 0.1 (10% test) - better for small datasets (<200 commits)
   - More rigorous: 0.3 (30% test) - better for large datasets (>1000 commits)

4. **Confidence threshold** (line ~52):
   - Default: `DEFAULT_MIN_CONFIDENCE = 0.1`
   - Fewer false positives: Increase to 0.3
   - More recall: Decrease to 0.05

**Example: Tuning for your project**

If predictions are too conservative (missing relevant files):

```python
# In ml_file_prediction.py

# Reduce frequency penalty - allow common files
freq_penalty = 1.0 - (model.file_frequency.get(f, 0) / max_freq) * 0.1  # Was 0.3

# Increase keyword weight - trust commit messages more
file_scores[f] += tf * idf * 2.0  # Was 1.5

# Lower confidence threshold
DEFAULT_MIN_CONFIDENCE = 0.05  # Was 0.1
```

If predictions have too many false positives:

```python
# Increase frequency penalty - penalize common files
freq_penalty = 1.0 - (model.file_frequency.get(f, 0) / max_freq) * 0.5  # Was 0.3

# Require stronger co-occurrence signal
if similarity > 0.3:  # Was no threshold
    file_scores[f] += similarity * 3.0

# Raise confidence threshold
DEFAULT_MIN_CONFIDENCE = 0.3  # Was 0.1
```

### Memory and Compute Considerations

**Memory usage:**

| Component | Typical Size | Max Observed |
|-----------|--------------|--------------|
| Model file | 100-500 KB | 2 MB |
| Commit data (full) | 10-50 MB | 200 MB |
| Commit data (lite) | 1-5 MB | 20 MB |
| AI metadata cache | 50-200 KB | 500 KB |

**Reducing memory footprint:**

1. **Limit commit message storage:**

```python
# In train_model() - line ~890
if len(model.file_to_commits[f]) > 10:
    model.file_to_commits[f] = model.file_to_commits[f][-10:]  # Keep 10 → change to 5
```

2. **Filter low-frequency files:**

```python
# After training, prune rare files
min_frequency = 2
model.file_frequency = {
    f: count for f, count in model.file_frequency.items()
    if count >= min_frequency
}
```

3. **Use lite commits only:**

```bash
# Don't collect full diffs (saves ~80% storage)
export ML_COLLECTION_ENABLED=0  # Disable full collection
python scripts/ml_data_collector.py backfill-lite -n 1000  # Lite only
```

**Speeding up training:**

Training is fast (~1-2 seconds for 500 commits), but can be optimized:

1. **Skip evaluation during development:**

```bash
# Without evaluation (~50% faster)
python scripts/ml_file_prediction.py train
```

2. **Cache AI metadata:**

```bash
# Pre-cache to avoid YAML parsing
python scripts/ml_file_prediction.py ai-meta --rebuild
```

3. **Filter commits before training:**

```python
# Load and filter before training (saves 10-20%)
examples = load_commit_data(filter_deleted=True)
examples = [ex for ex in examples if 1 <= len(ex.files_changed) <= 20]
model = train_model(examples)
```

---

## Common Pitfalls

### 1. Overfitting to Project-Specific Patterns

**Symptom:** Model performs well on old commits but poorly on new features.

**Cause:** Training on narrow time window or repetitive commit patterns.

**Example:**

```bash
# Check if model is too specialized
python scripts/ml_file_prediction.py predict "Add new module for X"
# Returns: Only suggests existing module files, not new locations
```

**Solution:**

- Ensure training data spans diverse commit types
- Use commits from multiple contributors
- Retrain after major architectural changes

```bash
# Check commit diversity
python scripts/ml_data_collector.py stats | grep "Commit types"
# Should see: feat (30%), fix (25%), refactor (15%), docs (10%), test (10%), etc.
```

**Prevention:**

```python
# Add diversity check to training script
from collections import Counter
commit_types = Counter(ex.commit_type for ex in examples if ex.commit_type)
dominant_type_ratio = max(commit_types.values()) / sum(commit_types.values())
if dominant_type_ratio > 0.5:
    print(f"⚠️  Warning: {dominant_type_ratio:.0%} of commits are same type")
```

### 2. Data Leakage (Test Set Contamination)

**Symptom:** Evaluation metrics are unrealistically high (MRR >0.8, Recall@10 >0.9).

**Cause:** Test commits used during training via co-occurrence relationships.

**Example:**

```python
# WRONG: Using seed files from test set
test_commit = test_examples[0]
predictions = predict_files(
    test_commit.message,
    model,
    seed_files=test_commit.files_changed  # LEAKAGE!
)
```

**Solution:**

- Never use seed files during evaluation
- Ensure train/test split happens before any feature extraction
- Use temporal split (older commits = train, newer = test)

```python
# CORRECT: No seed files during evaluation
def evaluate_model(model, test_examples, top_k=[1, 5, 10]):
    for example in test_examples:
        predictions = predict_files(
            example.message,
            model,
            top_n=max(top_k),
            seed_files=None  # Don't leak test data
        )
        # Calculate metrics...
```

**Prevention:**

```bash
# Temporal split - train on old, test on new
python scripts/ml_file_prediction.py evaluate --split 0.2  # Last 20% are test
```

### 3. Stale Model Symptoms

**Symptom:** Predictions suggest deleted/renamed files or miss new modules.

**Example:**

```bash
python scripts/ml_file_prediction.py predict "Update processor"
# Suggests: cortical/processor.py (deleted 50 commits ago)
# Misses: cortical/processor/core.py (current file)
```

**Cause:** Model trained before refactoring, not updated.

**Detection:**

```bash
# Check staleness
python scripts/ml_file_prediction.py stats
# Shows: "Git commit: a1b2c3 (25 commits behind)"

# Or during prediction
python scripts/ml_file_prediction.py predict "query" --verbose
# Shows: "⚠️  Model is 25 commits behind HEAD"
```

**Solution:**

1. **Immediate fix:** Update file path migrations

```python
# In ml_file_prediction.py - FILE_PATH_MIGRATIONS
FILE_PATH_MIGRATIONS = {
    'cortical/processor.py': [
        'cortical/processor/__init__.py',
        'cortical/processor/core.py',
        # ... add new files
    ],
}
```

2. **Retrain model:**

```bash
python scripts/ml_file_prediction.py train --evaluate --save-version
```

**Prevention:**

- Set up automatic retraining (weekly cron job)
- Add pre-commit check for model staleness
- Use `--save-version` to track model history

### 4. Imbalanced File Frequency

**Symptom:** Model always suggests the same 5 files regardless of task.

**Cause:** A few files (e.g., `test_processor.py`, `CLAUDE.md`) changed in 60% of commits.

**Detection:**

```bash
python scripts/ml_file_prediction.py stats
# Check "Most frequently changed files" section
# Bad: test_processor.py: 250 commits (62% of total)
```

**Solution:**

Increase frequency penalty in `predict_files()`:

```python
# Line ~1154 - Increase from 0.3 to 0.5
freq_penalty = 1.0 - (model.file_frequency.get(f, 0) / max_freq) * 0.5
```

Or cap file frequency during training:

```python
# After training, cap maximum frequency
max_freq_cap = model.total_commits * 0.3  # Max 30% of commits
for f in model.file_frequency:
    if model.file_frequency[f] > max_freq_cap:
        model.file_frequency[f] = int(max_freq_cap)
```

### 5. Poor Keyword Extraction

**Symptom:** Generic commit messages lead to irrelevant suggestions.

**Example:**

```bash
python scripts/ml_file_prediction.py predict "update code"
# Suggests: random files (no signal from generic message)
```

**Cause:** Stop words not comprehensive enough, or messages too vague.

**Solution:**

1. **Improve stop word filtering:**

Add project-specific stop words to `DEVELOPMENT_STOP_WORDS` (line ~76):

```python
DEVELOPMENT_STOP_WORDS = {
    # ... existing ...
    'code', 'file', 'module', 'system',  # Generic terms
    'cortical', 'processor', 'query',    # Project-specific but too common
}
```

2. **Boost semantic similarity:**

Enable semantic matching for vague messages:

```bash
python scripts/ml_file_prediction.py predict "update code" --use-semantic
```

3. **Educate developers on commit messages:**

Add to `.git/hooks/prepare-commit-msg`:

```bash
# Warn about generic messages
if [[ "$COMMIT_MSG" =~ ^(update|fix|change)\ (code|file) ]]; then
    echo "⚠️  Generic commit message. Be more specific for better ML predictions."
fi
```

---

## Evaluation Guidelines

### Understanding Metrics

**1. MRR (Mean Reciprocal Rank)**

Measures: How quickly do we find the first relevant file?

```
MRR = average(1 / rank_of_first_correct)
```

**Example:**

```
Commit: "Fix authentication bug"
Predictions: [auth.py, test_auth.py, config.py, login.py]
Actual: [auth.py, login.py]
First correct: auth.py at rank 1 → 1/1 = 1.0
```

**Interpretation:**

| MRR | Meaning |
|-----|---------|
| 0.8-1.0 | Excellent - first prediction almost always correct |
| 0.4-0.8 | Good - correct file typically in top 2-3 |
| 0.2-0.4 | Acceptable - correct file in top 5 |
| <0.2 | Poor - rarely predicts correct file early |

**Current project (403 commits, 20% test):**
- MRR: 0.428 → Correct file typically at rank 2-3

**2. Recall@K**

Measures: What fraction of actual files appear in top K predictions?

```
Recall@10 = (correct files in top 10) / (total actual files)
```

**Example:**

```
Commit: "Add authentication feature"
Predictions (top 10): [auth.py, test_auth.py, config.py, ...]
Actual: [auth.py, login.py, user.py, test_auth.py]
Correct in top 10: [auth.py, test_auth.py] = 2/4 = 0.5
```

**Interpretation:**

| Recall@10 | Meaning |
|-----------|---------|
| >0.7 | Excellent - captures most relevant files |
| 0.5-0.7 | Good - captures majority of files |
| 0.3-0.5 | Acceptable - captures some key files |
| <0.3 | Poor - misses most relevant files |

**Current project:**
- Recall@10: 0.48 → About half of relevant files in top 10

**3. Precision@K**

Measures: What fraction of top K predictions are correct?

```
Precision@1 = (correct files in top 1) / 1
```

**Example:**

```
Commit: "Fix authentication bug"
Prediction (top 1): [auth.py]
Actual: [auth.py, test_auth.py]
Correct: 1/1 = 1.0
```

**Interpretation:**

| Precision@1 | Meaning |
|-------------|---------|
| >0.5 | Excellent - top prediction usually correct |
| 0.3-0.5 | Good - top prediction correct 1/3 of time |
| 0.15-0.3 | Acceptable - top prediction occasionally correct |
| <0.15 | Poor - top prediction rarely correct |

**Current project:**
- Precision@1: 0.31 → Top prediction correct 31% of time

### When Metrics Are Misleading

**1. Multi-file commits skew Recall@K:**

```
Commit: "Refactor entire authentication module"
Actual: 15 files changed
Recall@10: 0.67 (10/15 in top 10)
```

This looks good, but 15-file commits are rare. Model may be poor at typical 2-3 file commits.

**Solution:** Segment evaluation by commit size:

```python
# Evaluate separately for small/medium/large commits
small = [ex for ex in test_examples if len(ex.files_changed) <= 3]
medium = [ex for ex in test_examples if 3 < len(ex.files_changed) <= 10]
large = [ex for ex in test_examples if len(ex.files_changed) > 10]

for subset, name in [(small, "small"), (medium, "medium"), (large, "large")]:
    metrics = evaluate_model(model, subset)
    print(f"{name} commits: MRR={metrics['mrr']:.3f}")
```

**2. Test file bias:**

```
Many commits include "test_X.py" alongside "X.py"
Model learns: "Always suggest test file"
Precision@1 looks high, but model isn't learning semantic patterns
```

**Solution:** Exclude test files from evaluation:

```python
# During evaluation, filter out test file predictions
predictions = predict_files(message, model, top_n=20)
non_test_predictions = [
    (f, score) for f, score in predictions
    if not f.startswith('tests/')
][:10]
```

**3. Temporal bias:**

If test set is from last 20% of commits chronologically:

```
Recent commits touch new features/files not in training set
Metrics appear worse than true model quality
```

**Solution:** Use random split instead of temporal:

```python
# In train_test_split()
train_test_split(examples, test_ratio=0.2, shuffle=True)  # shuffle=True
```

But note: Temporal split better represents real-world usage (model trained on past, predicts future).

### A/B Testing Best Practices

**Setup:**

```bash
# Save baseline model
python scripts/ml_file_prediction.py train --save-version
cp .git-ml/models/file_prediction.json .git-ml/models/baseline.json

# Make changes to ml_file_prediction.py
# (e.g., tune hyperparameters)

# Train new model
python scripts/ml_file_prediction.py train --save-version

# Compare on same test queries
python scripts/ml_file_prediction.py compare \
    "Add authentication feature" \
    --version1 .git-ml/models/baseline.json \
    --top 10
```

**Evaluation queries:**

Use representative queries from your project:

```bash
# queries.txt
feat: Add user authentication
fix: Fix memory leak in processor
refactor: Split processor into multiple modules
docs: Update API documentation
test: Add tests for query expansion
```

Then:

```bash
while read query; do
    echo "=== $query ==="
    python scripts/ml_file_prediction.py compare "$query" \
        --version1 .git-ml/models/baseline.json \
        --top 5
done < queries.txt
```

**Statistical significance:**

For meaningful comparison, test on at least 30 queries:

```python
# Calculate win rate
wins = sum(1 for q in queries if new_score(q) > baseline_score(q))
ties = sum(1 for q in queries if new_score(q) == baseline_score(q))
losses = len(queries) - wins - ties

print(f"Win rate: {wins/len(queries):.1%} ({wins}W / {ties}T / {losses}L)")
# Good: >60% win rate
# Acceptable: 50-60% win rate
# Poor: <50% win rate (new model is worse)
```

---

## Integration Best Practices

### Git Hook Integration

**1. Pre-commit suggestions (recommended):**

Install via:

```bash
python scripts/ml_data_collector.py install-hooks
```

This creates `.git/hooks/prepare-commit-msg` which:
- Runs before commit message is finalized
- Suggests missing files based on message
- Non-blocking by default (warns only)

**Configuration:**

```bash
# ~/.bashrc or ~/.zshrc
export ML_SUGGEST_ENABLED=1           # Enable (default)
export ML_SUGGEST_THRESHOLD=0.5       # Confidence threshold (0.0-1.0)
export ML_SUGGEST_BLOCKING=0          # Blocking mode (0=warn, 1=abort)
export ML_SUGGEST_TOP_N=5             # Check top N predictions
```

**Per-project overrides:**

```bash
# .envrc (if using direnv)
export ML_SUGGEST_THRESHOLD=0.7  # Higher threshold for this project
```

**2. Post-commit data collection:**

Automatically captures commit metadata:

```bash
# .git/hooks/post-commit
#!/bin/bash
python scripts/ml_data_collector.py commit 2>&1 | logger -t ml-data
```

**Best practices:**

- Run in background to avoid slowing commits
- Log errors to syslog for debugging
- Skip for ML-only commits (already handled in template)

**3. Pre-push hooks (optional):**

Remind to retrain if model is stale:

```bash
# .git/hooks/pre-push
#!/bin/bash
staleness=$(python scripts/ml_file_prediction.py stats 2>/dev/null | grep "commits behind" | grep -oP '\d+')
if [[ $staleness -gt 20 ]]; then
    echo "⚠️  ML model is $staleness commits behind. Consider retraining:"
    echo "    python scripts/ml_file_prediction.py train --evaluate"
fi
```

### CI/CD Integration

**1. Automated model training:**

```yaml
# .github/workflows/ml-training.yml
name: ML Model Training

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday 2 AM
  workflow_dispatch:      # Manual trigger

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Need full history

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Train model
        id: train
        run: |
          python scripts/ml_file_prediction.py train --evaluate --save-version 2>&1 | tee train.log
          echo "mrr=$(grep 'MRR:' train.log | grep -oP '\d+\.\d+')" >> $GITHUB_OUTPUT

      - name: Check performance
        run: |
          mrr=${{ steps.train.outputs.mrr }}
          if (( $(echo "$mrr < 0.3" | bc -l) )); then
            echo "::warning::Model MRR ($mrr) is below threshold (0.3)"
          fi

      - name: Upload model
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: |
            .git-ml/models/file_prediction.json
            .git-ml/models/history/*.json

      - name: Comment on commit (if manual trigger)
        if: github.event_name == 'workflow_dispatch'
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.repos.createCommitComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              commit_sha: context.sha,
              body: `🤖 ML model retrained. MRR: ${{ steps.train.outputs.mrr }}`
            })
```

**2. Model validation in PR checks:**

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on: pull_request

jobs:
  validate-ml-impact:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Check if model should be retrained
        run: |
          # If files in cortical/ changed significantly
          changed_files=$(git diff --name-only origin/main...HEAD | grep '^cortical/' | wc -l)
          if [[ $changed_files -gt 10 ]]; then
            echo "::notice::Consider retraining ML model (10+ core files changed)"
          fi

      - name: Check file path migrations
        run: |
          # If files were moved/deleted, check FILE_PATH_MIGRATIONS
          moved=$(git diff --name-status origin/main...HEAD | grep '^R' | wc -l)
          if [[ $moved -gt 0 ]]; then
            if ! git diff origin/main...HEAD scripts/ml_file_prediction.py | grep 'FILE_PATH_MIGRATIONS'; then
              echo "::warning::Files were renamed but FILE_PATH_MIGRATIONS not updated"
            fi
          fi
```

**3. Performance monitoring:**

Track model metrics over time:

```yaml
# .github/workflows/ml-metrics.yml
name: ML Metrics Tracking

on:
  schedule:
    - cron: '0 3 * * 1'  # Weekly on Monday 3 AM

jobs:
  track-metrics:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Evaluate current model
        run: |
          python scripts/ml_file_prediction.py evaluate --split 0.2 > metrics.txt

      - name: Upload to metrics service
        run: |
          # Example: Upload to Datadog, Prometheus, etc.
          mrr=$(grep 'mrr:' metrics.txt | grep -oP '\d+\.\d+')
          curl -X POST https://metrics.example.com/api/v1/metrics \
            -d "ml.file_prediction.mrr=$mrr"
```

### Team Workflow Integration

**1. Onboarding new developers:**

Add to onboarding checklist:

```markdown
## ML Setup (5 min)

1. Install git hooks:
   ```bash
   python scripts/ml_data_collector.py install-hooks
   ```

2. Configure preferences:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export ML_SUGGEST_ENABLED=1
   export ML_SUGGEST_THRESHOLD=0.5
   ```

3. Test the system:
   ```bash
   git add .
   git commit -m "test: ML hook test"
   # You should see ML suggestions (if trained)
   git reset HEAD~1  # Undo test commit
   ```

4. Optional: Enable semantic search
   ```bash
   pip install pyyaml  # For AI metadata
   python scripts/ml_file_prediction.py ai-meta --rebuild
   ```
```

**2. Shared model distribution:**

For teams, share trained models:

```bash
# Option 1: Commit model to repo (if <1MB)
git add .git-ml/models/file_prediction.json
git commit -m "chore: Update ML file prediction model"

# Option 2: Use git LFS for larger models
git lfs track "*.json"
git add .gitattributes .git-ml/models/file_prediction.json
git commit -m "chore: Update ML model (via LFS)"

# Option 3: Artifact storage (recommended for large teams)
# Upload to S3/GCS and download during setup
```

**3. Model quality reviews:**

Include in code review process:

```markdown
## ML Model Review Checklist

When merging significant refactoring or architecture changes:

- [ ] Model retrained after changes
- [ ] FILE_PATH_MIGRATIONS updated for renamed/moved files
- [ ] Evaluation metrics acceptable (MRR >0.3, Recall@10 >0.3)
- [ ] Tested on 5+ representative commit messages
- [ ] Model version saved to history
```

---

## Model Versioning

### Version History

Every model saves metadata:

```json
{
  "version": "1.1.0",
  "trained_at": "2025-12-16T14:30:52",
  "git_commit_hash": "a1b2c3d4e5f6",
  "total_commits": 403,
  "metrics": {
    "mrr": 0.428,
    "recall@10": 0.480,
    "precision@1": 0.312
  }
}
```

**Save versions during training:**

```bash
python scripts/ml_file_prediction.py train --evaluate --save-version
# Saves to: .git-ml/models/history/model_YYYYMMDD_HHMMSS_<githash>.json
```

**List version history:**

```bash
python scripts/ml_file_prediction.py history --limit 10

# Output:
#   [1] model_20251216_143052_a1b2c3.json
#       Trained: 2025-12-16T14:30:52
#       Git: a1b2c3d4e5f6
#       Commits: 403
#       MRR: 0.4285, Recall@5: 0.3819
```

### Rollback Procedure

If new model performs poorly:

```bash
# 1. List versions
python scripts/ml_file_prediction.py history

# 2. Load previous version
cp .git-ml/models/history/model_20251210_*.json .git-ml/models/file_prediction.json

# 3. Verify
python scripts/ml_file_prediction.py stats
python scripts/ml_file_prediction.py predict "test query"
```

**Automated rollback:**

```bash
# If evaluation fails, restore last good version
if ! python scripts/ml_file_prediction.py evaluate --split 0.2 | grep -q "mrr: 0\.[3-9]"; then
    echo "⚠️  Model quality degraded. Rolling back..."
    latest=$(ls -t .git-ml/models/history/*.json | head -2 | tail -1)
    cp "$latest" .git-ml/models/file_prediction.json
    echo "✓ Restored: $latest"
fi
```

### Version Comparison

Compare predictions between versions:

```bash
python scripts/ml_file_prediction.py compare \
    "Add authentication feature" \
    --version1 .git-ml/models/history/model_20251210_*.json \
    --version2 .git-ml/models/history/model_20251216_*.json \
    --top 10

# Output:
# Model 1: 2025-12-10 (a1b2c3)
#   ✓ 1. cortical/auth.py              (0.823)
#   ✓ 2. tests/test_auth.py            (0.654)
#     3. cortical/processor/core.py    (0.512)
#
# Model 2: 2025-12-16 (d4e5f6)
#   ✓ 1. cortical/auth.py              (0.891)
#   ✓ 2. tests/test_auth.py            (0.723)
#     3. cortical/security.py          (0.587)  # NEW
#
# Common: 2, Only in Model 1: 1, Only in Model 2: 1
# Jaccard similarity: 67%
```

### Branching Strategy

For teams working on multiple features:

```
main branch:
  └─ .git-ml/models/file_prediction.json  (production model)

feature-auth branch:
  └─ .git-ml/models/file_prediction_auth.json  (specialized model)
```

Train branch-specific models:

```bash
# On feature branch
python scripts/ml_file_prediction.py train \
    --output .git-ml/models/file_prediction_auth.json

# Use branch-specific model
python scripts/ml_file_prediction.py predict "auth query" \
    --model .git-ml/models/file_prediction_auth.json  # TODO: Add flag
```

**Note:** Current implementation doesn't support `--model` flag. To implement:

```python
# In main() function, add to predict_parser:
predict_parser.add_argument('--model', type=str,
                            help='Path to model file (default: .git-ml/models/file_prediction.json)')

# Then:
model_path = Path(args.model) if args.model else FILE_PREDICTION_MODEL
model = load_model(model_path)
```

---

## Troubleshooting

### Model Not Training

**Symptom:**

```bash
python scripts/ml_file_prediction.py train
# No commits file found at .git-ml/tracked/commits.jsonl
```

**Causes:**

1. ML data collection not initialized
2. No commits captured yet
3. File permission issues

**Solutions:**

```bash
# 1. Check if data collection is enabled
echo $ML_COLLECTION_ENABLED  # Should be empty or "1"

# 2. Initialize data collection
python scripts/ml_data_collector.py install-hooks

# 3. Backfill historical commits
python scripts/ml_data_collector.py backfill-lite -n 100

# 4. Verify commit data
ls -lh .git-ml/tracked/commits.jsonl
head -5 .git-ml/tracked/commits.jsonl  # Should be valid JSON lines

# 5. Check permissions
chmod +x scripts/ml_data_collector.py
chmod +x scripts/ml_file_prediction.py
```

### Poor Prediction Quality

**Symptom:** Model suggests irrelevant files or misses obvious ones.

**Diagnostic steps:**

```bash
# 1. Check model stats
python scripts/ml_file_prediction.py stats
# Look for:
# - Low commit count (<100)
# - Unbalanced file frequency (top file >50% of commits)
# - Few commit types (<4)

# 2. Evaluate with metrics
python scripts/ml_file_prediction.py evaluate --split 0.2
# Look for:
# - MRR <0.2 (very poor)
# - Recall@10 <0.2 (missing most files)
# - Precision@1 <0.1 (rarely correct)

# 3. Test on known queries
python scripts/ml_file_prediction.py predict "fix bug in processor" --verbose
# Check warnings:
# - LOW_TRAINING_DATA: Need more commits
# - NO_KEYWORD_MATCH: Message too generic
# - LOW_CONFIDENCE: Predictions unreliable

# 4. Check data quality
python scripts/ml_data_collector.py validate
# Looks for:
# - Schema violations
# - Missing required fields
# - Suspicious patterns
```

**Solutions by symptom:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Always suggests same files | Frequency imbalance | Increase `freq_penalty` (line ~1154) |
| Misses new modules | Stale model | Retrain, update migrations |
| Generic predictions | Poor keywords | Add stop words, enable semantic matching |
| Wrong file types | Commit type mismatch | Review `COMMIT_TYPE_PATTERNS` |
| Low confidence warnings | Small dataset | Collect more commits (target: 200+) |

### Evaluation Metrics Don't Match Reality

**Symptom:** Evaluation shows MRR=0.6 but predictions seem worse.

**Causes:**

1. Test set not representative
2. Metric interpretation issue
3. Overfitting to test patterns

**Investigation:**

```bash
# 1. Check test set composition
python scripts/ml_file_prediction.py evaluate --split 0.2 --verbose
# Manually review 10 test examples

# 2. Manual evaluation on real commits
git log --oneline -20 | while read hash msg; do
    echo "=== $msg ==="
    files=$(git show --name-only --format="" $hash)
    pred=$(python scripts/ml_file_prediction.py predict "$msg" --top 3)
    echo "Actual: $files"
    echo "Predicted: $pred"
    read -p "Correct? (y/n) " answer
done
# Calculate manual accuracy

# 3. A/B test with developers
# Have team use model for 1 week, track:
# - True positives (suggested file was needed)
# - False positives (suggested file not needed)
# - False negatives (needed file not suggested)
```

### Hooks Not Running

**Symptom:** Commits succeed but no ML suggestions appear.

**Diagnostic:**

```bash
# 1. Check if hooks are installed
ls -l .git/hooks/prepare-commit-msg
ls -l .git/hooks/post-commit

# 2. Check hook is executable
[ -x .git/hooks/prepare-commit-msg ] && echo "Executable" || echo "Not executable"

# 3. Test hook manually
export ML_SUGGEST_ENABLED=1
bash .git/hooks/prepare-commit-msg .git/COMMIT_EDITMSG commit

# 4. Check for errors
git commit -m "test" 2>&1 | grep -i error

# 5. Verify ML_COLLECTION_ENABLED
echo $ML_COLLECTION_ENABLED  # Should NOT be "0"
```

**Solutions:**

```bash
# Reinstall hooks
python scripts/ml_data_collector.py install-hooks

# Make executable
chmod +x .git/hooks/prepare-commit-msg
chmod +x .git/hooks/post-commit
chmod +x scripts/ml-precommit-suggest.sh

# Enable collection
unset ML_COLLECTION_ENABLED  # Or set to "1"

# Test
git add .
git commit -m "feat: Test ML hooks" --dry-run
```

### Model File Corruption

**Symptom:** `json.decoder.JSONDecodeError` when loading model.

**Recovery:**

```bash
# 1. Check model file
cat .git-ml/models/file_prediction.json | jq . > /dev/null
# If error, file is corrupted

# 2. Restore from version history
ls -lt .git-ml/models/history/*.json | head -1
cp .git-ml/models/history/model_20251210_*.json .git-ml/models/file_prediction.json

# 3. If no history, retrain
python scripts/ml_file_prediction.py train --evaluate --save-version

# 4. Verify
python scripts/ml_file_prediction.py stats
```

**Prevention:**

```bash
# Backup before training
cp .git-ml/models/file_prediction.json .git-ml/models/file_prediction.json.bak

# Or use version history (automatic with --save-version)
python scripts/ml_file_prediction.py train --save-version
```

---

## Summary

**Key Takeaways:**

1. **Data Quality First**: Ensure diverse, balanced commit data before training
2. **Evaluate Regularly**: Use `--evaluate` flag to track model performance
3. **Retrain Often**: Weekly or after 50+ commits to avoid staleness
4. **Version Everything**: Use `--save-version` to enable rollback
5. **Monitor in Production**: Track false positives/negatives from pre-commit hook
6. **Tune Thoughtfully**: Adjust hyperparameters based on evaluation metrics, not intuition
7. **Integrate Smoothly**: Non-blocking hooks by default, educate team on usage

**Quick Commands:**

```bash
# Initial setup
python scripts/ml_data_collector.py install-hooks
python scripts/ml_data_collector.py backfill-lite -n 200

# Training workflow
python scripts/ml_file_prediction.py train --evaluate --save-version

# Evaluation
python scripts/ml_file_prediction.py evaluate --split 0.2
python scripts/ml_file_prediction.py predict "your task description" --verbose

# Maintenance
python scripts/ml_file_prediction.py stats  # Check staleness
python scripts/ml_file_prediction.py history  # View versions
python scripts/ml_file_prediction.py dashboard  # Complete overview
```

**Next Steps:**

- [ ] Review [ML Data Collection Knowledge Transfer](ml-data-collection-knowledge-transfer.md)
- [ ] Configure [Pre-Commit Suggestions](ml-precommit-suggestions.md)
- [ ] Set up CI/CD training (see [Integration Best Practices](#integration-best-practices))
- [ ] Add to team onboarding documentation

---

**Document Version:** 1.0
**Last Updated:** 2025-12-16
**Maintained By:** ML Training Team
