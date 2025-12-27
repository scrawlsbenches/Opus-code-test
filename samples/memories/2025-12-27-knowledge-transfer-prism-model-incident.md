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

## Open Questions

### Where Did the Original 37K Documents Come From?

The current corpus generation only produces ~19K patterns:
```
training_patterns.jsonl: 35,582 lines
augmented_corpus.txt: 2,094 lines
Total: ~19,392 training patterns
```

But the original model had **37,318 documents**. Possible explanations:
1. Additional training data was used but not committed
2. Multiple training runs accumulated data
3. Different/larger corpus was used historically

**Action needed:** Investigate how to regenerate the full training corpus to match or exceed the original.

### Why Was file_location Score 88% in Baseline?

The original model achieved 88% on file_location queries. Current training achieves only 50%. Need to understand:
1. What training patterns produced the 88% score?
2. How to replicate that training data?

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
