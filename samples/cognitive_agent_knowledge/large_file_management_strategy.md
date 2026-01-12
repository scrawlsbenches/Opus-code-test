# Large File Management Strategy: Git History as Rebuild Source

*Session: 2026-01-12 | Topic: Solving the Training Artifact Problem*

---

## The Problem

We have multiple large artifacts that keep getting gitignored:

| Artifact | Size | Why Removed | Rebuild Time |
|----------|------|-------------|--------------|
| `models/cognitive_agent/` | 154MB | Too large for git | ~45s |
| `.got/` index | Variable | Merge conflicts | ~10s |
| `corpus_dev/` | Variable | Regeneratable | ~30s |
| `.spark_intelligence_model.json` | ~5MB | Regeneratable | ~5s |

**The cycle:**
1. Artifact is useful → commit it
2. Repo bloats or conflicts occur → gitignore it
3. Cold-start breaks → fight about recommitting
4. Repeat

---

## Why Git Doesn't Like These Files

1. **Size**: Git stores full copies of binary/large JSON files
2. **Churn**: Training produces different output each time (timestamps, ordering)
3. **Merge conflicts**: JSON files with same keys conflict on merge
4. **History bloat**: Old versions persist in .git forever

---

## The Insight: Separate Essential from Derived

Analysis of `models/cognitive_agent/`:

```
154MB total breakdown:
├── tokenizer/           1.1MB  ← ESSENTIAL (vocabulary, learned merges)
├── bridge/
│   ├── atoms_word.json  4.7MB  ← DERIVED but stable (word atoms)
│   ├── atoms_follows_*  76MB   ← DERIVED from source docs
│   ├── atoms_similarity_* 72MB ← DERIVED from word co-occurrence
│   └── code entities    ~5MB   ← DERIVED from code indexing
└── training_manifest.json 150KB ← ESSENTIAL (what was trained)
```

**Key realization**: Given the vocabulary + source files + algorithm, we can DETERMINISTICALLY rebuild the links.

---

## Proposed Strategy: Tiered Storage

### Tier 1: Commit (Small, Essential, Merge-Safe)

```
models/cognitive_agent/
├── tokenizer/              # 1.1MB - vocabulary is essential
│   ├── vocab_*.json        # Sharded by first letter (merge-safe)
│   ├── merges.json         # BPE learned merges
│   ├── doc_frequency.json  # IDF weights
│   └── meta.json           # Metadata
└── training_manifest.json  # 150KB - what was trained
```

**Total committed: ~1.3MB**

### Tier 2: Gitignore + Lazy Rebuild (Large, Derived)

```
models/cognitive_agent/bridge/  # 152MB - rebuild on demand
├── atoms_word.json             # Rebuild from vocabulary
├── atoms_follows_*.json        # Rebuild from source docs
├── atoms_similarity_*.json     # Rebuild from co-occurrence
└── code entities               # Rebuild from code indexing
```

### Tier 3: Cache Branch (Optional, Pre-built)

```
git branch: model-cache
└── models/cognitive_agent/bridge/  # Full pre-built model
```

---

## Implementation: Rebuild from Git History

### The Manifest is the Source of Truth

```json
// training_manifest.json (committed)
{
  "model_version": "1.0",
  "vocabulary_size": 22594,
  "documents": {
    "cortical/cognitive/graph.py": {
      "content_hash": "abc123...",
      "trained_at": "2026-01-12T..."
    }
  }
}
```

### Bootstrap Script

```bash
#!/bin/bash
# scripts/bootstrap_cognitive.sh

MODEL_DIR="models/cognitive_agent"

# Check if full model exists
if [ -f "$MODEL_DIR/bridge/meta.json" ]; then
    echo "Model already built."
    exit 0
fi

# Check if vocabulary exists (Tier 1)
if [ ! -f "$MODEL_DIR/tokenizer/meta.json" ]; then
    echo "ERROR: Vocabulary not committed. Run full training."
    python -m cortical.cognitive train cortical/ samples/ --pattern "*.py" "*.md"
    exit 0
fi

# Rebuild links from vocabulary + sources (fast path)
echo "Rebuilding links from committed vocabulary..."
python -m cortical.cognitive rebuild-links
```

### New CLI Command: rebuild-links

```python
# In training.py
def rebuild_links_from_vocabulary():
    """
    Fast rebuild using committed vocabulary.

    Instead of full training (~45s):
    1. Load vocabulary from committed tokenizer/ (~1s)
    2. Load manifest to know which files (~0.1s)
    3. Rebuild only FOLLOWS/SIMILARITY links (~20s)

    Total: ~21s vs ~45s
    """
    trainer = IncrementalTrainer.load()

    # Vocabulary already loaded from committed files
    # Just need to rebuild the link graph

    for doc_path in trainer.manifest.documents:
        if Path(doc_path).exists():
            content = Path(doc_path).read_text()
            trainer.bridge.rebuild_links_for_document(content)

    trainer.save()
```

---

## Merge Conflict Prevention

### Problem: JSON files conflict on merge

Two branches train on different files → both modify manifest → conflict.

### Solution: Append-Only Manifest with Hash-Based Merging

```python
# Instead of:
{
  "documents": {"file1": {...}, "file2": {...}}
}

# Use content-addressed storage:
{
  "documents": {
    "sha256:abc123": {"path": "file1", ...},
    "sha256:def456": {"path": "file2", ...}
  }
}
```

Content hashes are deterministic → same file = same key → no conflict.

### .gitattributes for Merge Strategy

```gitattributes
# Merge strategy for training artifacts
models/cognitive_agent/training_manifest.json merge=union
models/cognitive_agent/tokenizer/vocab_*.json merge=union
```

The `union` strategy keeps all unique lines from both sides.

---

## Using Git History for Rebuild

### Concept: Training Commands as Source of Truth

Instead of storing artifacts, store the INTENT:

```bash
# .cognitive-training-history (committed)
2026-01-10 train cortical/ --pattern "*.py"
2026-01-11 train samples/ --pattern "*.md"
2026-01-12 train samples/cognitive_agent_knowledge/
```

On cold-start:
1. Read training history
2. Replay commands in order
3. Result is deterministic (same inputs → same outputs)

### Git Hooks for Automatic Rebuild

```bash
# .git/hooks/post-checkout
#!/bin/bash
# Rebuild model if vocabulary exists but links don't

if [ -f "models/cognitive_agent/tokenizer/meta.json" ] && \
   [ ! -f "models/cognitive_agent/bridge/meta.json" ]; then
    echo "Rebuilding cognitive model links..."
    python -m cortical.cognitive rebuild-links &
fi
```

---

## Handling Other Gitignored Artifacts

### .got/ Index

**Current problem**: `_version.json` causes merge conflicts.

**Solution already attempted**: `merge=ours` strategy (commit 17705577)

**Better solution**: Content-addressed entity storage
- Entity ID = hash of content
- Same entity = same ID = no conflict

### corpus_dev/

**Current**: Full corpus gitignored, rebuilt on demand.

**Better**: Commit only corpus manifest (list of source files), rebuild corpus on demand.

### .spark_intelligence_model.json

**Current**: Gitignored, 5MB.

**Better**: Commit vocabulary only (~500KB), rebuild predictions on demand.

---

## Recommended Approach

### Phase 1: Implement Tiered Storage (Low Effort)

1. Remove `models/cognitive_agent/` from .gitignore
2. Add `models/cognitive_agent/bridge/` to .gitignore
3. Commit tokenizer + manifest (~1.3MB)
4. Add bootstrap script

```bash
# Update .gitignore
- models/cognitive_agent/
+ models/cognitive_agent/bridge/
```

### Phase 2: Add rebuild-links Command (Medium Effort)

```python
# New command that skips vocabulary building
python -m cortical.cognitive rebuild-links
# Rebuilds FOLLOWS/SIMILARITY from committed vocabulary
# ~20s instead of ~45s
```

### Phase 3: Cache Branch for CI (Optional)

```bash
# CI workflow
git fetch origin model-cache
git checkout origin/model-cache -- models/cognitive_agent/bridge/
# Now have pre-built model without rebuild
```

---

## Size Impact Analysis

| Approach | Repo Size | Cold-Start Time | Merge Risk |
|----------|-----------|-----------------|------------|
| Current (gitignore all) | +0MB | 45s rebuild | None |
| Commit everything | +154MB | 0s | High (JSON conflicts) |
| **Tier 1 only** | **+1.3MB** | **~20s rebuild** | **Low** |
| Tier 1 + cache branch | +1.3MB (main) | 0s (if cache pulled) | Low |

**Recommendation**: Tier 1 approach gives 90% of the benefit with 1% of the size impact.

---

## Questions Answered

**Q: Can we use git history to rebuild what we need?**

Yes. The training manifest IS the history. Given:
- Committed vocabulary (what words exist)
- Committed manifest (what files were trained)
- Source files (already in git)

We can deterministically rebuild the link graph.

**Q: Do we need to fight about what to commit?**

No. Commit the MINIMUM needed for rebuild:
- Vocabulary (can't derive this without full retraining)
- Manifest (records what was trained)

Let everything else be derived.

**Q: What about merge conflicts?**

Use content-addressed storage (hash-based keys) and union merge strategy. Same content = same hash = no conflict.

---

## Action Items

1. **Immediate**: Update .gitignore to commit tokenizer + manifest
2. **Next**: Implement `rebuild-links` command for fast cold-start
3. **Future**: Apply same pattern to .got/, corpus_dev/, other artifacts

---

*The principle: Store intent, derive artifacts. Git tracks what you WANT, not what you BUILT.*
