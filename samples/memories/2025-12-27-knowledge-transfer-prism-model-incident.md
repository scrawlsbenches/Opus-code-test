# Knowledge Transfer: PRISM Model Overwrite Incident

**Date:** 2025-12-27
**Session:** claude/accept-handoff-ctrSI
**Tags:** `incident`, `prism-slm`, `training`, `data-loss`, `prevention`

---

## Executive Summary

The 13MB PRISM-SLM model (692K lines, 15,814 vocab, 37,318 documents) was accidentally overwritten with a tiny 2.7K line model (329 vocab) during a benchmark evaluation session. The model was restored from git history, and safeguards were added to prevent recurrence.

---

## What Happened

### Timeline

1. **Original State:** PRISM model trained on 37,318 documents with 15,814 vocabulary terms
2. **Incident (commit 079d0c00):** Agent ran `train_augmented.py` during benchmark analysis
   - Script overwrote `models/prism_augmented.json` without warning
   - New model: 329 vocab, 736 contexts, 43,960 tokens
   - Commit message: "Update PRISM model after training evaluation"
3. **Discovery:** User asked "why did you delete the prism model?"
4. **Recovery (commit cc84e3d2):** Restored from `git show 5bbd8714:benchmarks/codebase_slm/models/prism_augmented.json`

### The Misleading Metrics

The agent saw these metrics and thought it was an improvement:

```
Benchmark results:
- concept: 67% (+67% from baseline)     ← Looked good!
- file_location: 33% (-54% from baseline) ← Ignored this regression
- hierarchical: 100%
- Overall: 61%
```

But didn't realize:
- Vocabulary shrunk from **15,814 → 329** (98% loss!)
- Documents shrunk from **37,318 → ~2,000** (95% loss!)
- The "baseline" comparison was against a worse model, not the original

---

## Root Causes

### 1. No Safeguards in Training Script
- `train_augmented.py` had no `--dry-run` option
- Saved directly to default path without confirmation
- No backup mechanism
- No provenance tracking

### 2. Default Behavior Was Destructive
```python
# OLD (dangerous)
def main():
    corpus = load_corpus()
    model = train(corpus)
    save(model, "models/prism_augmented.json")  # Just overwrites!
```

### 3. No Model Comparison Before Overwrite
- Script didn't compare new model size vs existing
- Didn't warn about vocab reduction
- Didn't track what data produced the original model

### 4. Misleading Commit Message
- "Update PRISM model" sounds intentional
- Metrics in commit message hid the severity (vocab/doc counts)
- Easy to miss the `-54% file_location` regression

---

## Fixes Applied

### 1. Training Script Safeguards (commit 945cd340)

**`train_augmented.py` now has:**

```bash
--dry-run          # Evaluate only, don't save
--output PATH      # Explicit output path required
--force            # Skip backup (must be explicit)
--no-existing      # Train only on new data
```

**Automatic backup:**
```python
def backup_existing_model(model_path):
    # Creates timestamped backup before overwrite
    # Keeps last 5 backups in models/backups/
```

**Provenance tracking:**
```json
{
  "_provenance": {
    "trained_at": "2025-12-27T...",
    "corpus_hash": "abc123...",
    "corpus_size": 19392,
    "sources": ["augmented_corpus.txt", "training_patterns.jsonl"],
    "script": "train_augmented.py"
  }
}
```

### 2. `.gitignore` Updated (commit 3432f364)

```gitignore
# Backups are local recovery, not tracked
benchmarks/codebase_slm/models/backups/
```

### 3. Similar Fixes to `train_slm.py`
- Same safeguards applied
- Only saves when `--output` is specified

---

## How to Prevent Recurrence

### For Agents Running Training Scripts

1. **Always use `--dry-run` first:**
   ```bash
   python -m benchmarks.codebase_slm.train_augmented --dry-run
   ```

2. **Compare model sizes before training:**
   ```bash
   wc -l benchmarks/codebase_slm/models/prism_augmented.json
   # Current: 692,026 lines
   ```

3. **Check provenance if present:**
   ```bash
   python -c "import json; print(json.load(open('...'))['_provenance'])"
   ```

4. **Use explicit output for new models:**
   ```bash
   python -m benchmarks.codebase_slm.train_augmented --output models/my_new_model.json
   ```

### For Training Script Authors

1. **Never default to overwriting existing models**
2. **Always require explicit output path for destructive operations**
3. **Add backup mechanism with timestamped files**
4. **Track provenance (what data, when, what script)**
5. **Warn loudly when new model is smaller than existing**

---

## Forensic Analysis

### Timeline Reconstruction

| Date/Time | Commit | Event |
|-----------|--------|-------|
| Dec 26, 22:40 | 137e5585 | Generators added, 27,998 patterns generated |
| Dec 26, 23:17 | 982c89b6 | Knowledge-base added, 35,504 patterns, 626K transitions |
| Dec 26, 22:44 | 02dc3a58 | train_slm.py added |
| **Dec 27, 00:04** | **2b458285** | **13MB model committed (37,318 docs)** |
| Dec 27, 01:45 | 079d0c00 | MODEL OVERWRITTEN (329 vocab) |
| Dec 27, 01:56 | cc84e3d2 | Model restored from git |

### Root Cause: Training Data Never Tracked

**Critical finding:** `benchmarks/codebase_slm/corpus/` was added to `.gitignore` from day 1.

```
Commit 137e5585 added to .gitignore:
  benchmarks/codebase_slm/corpus/
```

This means:
1. `training_patterns.jsonl` was **never committed**
2. Only the MODEL was tracked, not its source data
3. The original 37,318 document corpus is **unrecoverable**

### Data Accounting

| What | Lines | Status |
|------|-------|--------|
| Model (prism_augmented.json) | 692,026 | ✅ Tracked, restored |
| augmented_corpus.txt | 1,814 | ✅ Tracked |
| knowledge-base/*.md | 3,144 | ✅ Tracked |
| corpus/training_patterns.jsonl | ~35,500 | ❌ **Never tracked** |

**Gap:** Model had 37,318 docs but only ~5,000 lines of training data were committed.

### Why 37K Documents?

The commit messages tell the story:
- 137e5585: "27,998 training patterns generated"
- 982c89b6: "35,504 total patterns"

The model was trained on **locally generated** data that was:
1. Created by running `generate_corpus.py`
2. Stored in `corpus/training_patterns.jsonl`
3. **Never committed** (gitignored)
4. Used to train the model that WAS committed

### Prevention: Track Training Data

**Recommended change:** Remove `benchmarks/codebase_slm/corpus/` from `.gitignore`

This ensures:
- Training data is versioned alongside models
- Provenance is maintained
- Models can be reproduced exactly

Alternatively, add provenance hash to model:
```json
{
  "_provenance": {
    "corpus_hash": "sha256:abc123...",
    "corpus_size": 35504,
    "generated_at": "2025-12-27T00:00:00Z"
  }
}
```

---

## Key Learnings

1. **Size matters for models** - Vocab shrinking from 15K→329 is catastrophic, not an optimization
2. **Don't trust overall scores** - 61% overall hid a 54% regression in the primary use case
3. **Safeguards prevent accidents** - A simple `--dry-run` flag would have prevented this
4. **Git history is your friend** - Model was fully recoverable from git
5. **Provenance is essential** - Without knowing what trained the original, we can't improve it

---

## Commands Reference

```bash
# Check current model
wc -l benchmarks/codebase_slm/models/prism_augmented.json
python -c "import json; m=json.load(open('benchmarks/codebase_slm/models/prism_augmented.json')); print(f'Vocab: {len(m[\"vocab\"])}, Docs: {m[\"total_documents\"]}')"

# Dry run training (safe)
python -m benchmarks.codebase_slm.train_augmented --dry-run

# Generate corpus
python -m benchmarks.codebase_slm.generate_corpus --full

# Train to new file (safe)
python -m benchmarks.codebase_slm.train_augmented --output models/experimental.json

# Restore from git if needed
git show COMMIT_HASH:benchmarks/codebase_slm/models/prism_augmented.json > restored_model.json
```

---

## Related Documents

- `docs/ml-training-best-practices.md` - Training guidelines
- `samples/memories/2025-12-22-session-knowledge-transfer-got-migration.md` - Previous knowledge transfer
- Commits: 079d0c00 (incident), cc84e3d2 (recovery), 945cd340 (safeguards)
