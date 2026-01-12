# Cognitive Agent Roadmap

**Date**: 2026-01-12
**Branch**: `claude/recover-idf-spec-code-6yA7K`
**Status**: Active Development

---

## Current State

### What's Working
- **IDF-weighted similarity links** - Full BDD spec implemented (18 tests passing)
- **Incremental training** - Can train on files incrementally
- **Staleness tracking** - Warns when IDF weights need refresh (>20% growth)
- **CLI commands** - train, status, reindex, query, ask, generate, index-code
- **Performance optimizations**:
  - `predict_next()`: O(1) via `_outgoing` index (was O(n) 55-70ms)
  - Incremental saves: ~11s (was 54s full rewrite)
  - No-change saves: 0.05s

### Model Statistics
- Documents: 660 trained (531 samples + 129 code files)
- Atoms: ~553k (21k WORD, 218k SIMILARITY, 267k FOLLOWS)
- Vocabulary: 21,358 terms
- Untrained Python files: ~217 remaining

---

## Known Issues (Low Priority)

### 1. `_idf_epoch` Not Restored on Load
**Location**: `text_bridge.py:763-766`
**Impact**: Epoch resets to 1 on each session instead of incrementing
**Fix**: Restore `_idf_epoch` from manifest during `IncrementalTrainer.load()`

```python
# In IncrementalTrainer.load() after loading bridge:
trainer.bridge._idf_epoch = trainer.manifest.idf_epoch
```

### 2. Save Still ~11s for Small Changes
**Cause**: Even 1 dirty FOLLOWS atom requires rewriting all 267k FOLLOWS atoms
**Potential fix**: Per-subdivision dirty tracking (only rewrite affected subdivisions)
**ROI**: Medium - would reduce 11s to ~3s for typical incremental saves

---

## Proposed Focus Areas

### Tier 1: High Value, Near-Term

#### A. CLI Performance Metrics
**Why**: Catch performance regressions automatically during normal usage
**Scope**:
```
python -m cortical.cognitive train --metrics
# Outputs:
# {
#   "documents_trained": 50,
#   "training_time_ms": 4500,
#   "save_time_ms": 11000,
#   "atoms_created": 35000,
#   "links_created": 42000,
#   "memory_mb": 602,
#   "staleness_pct": 9.4
# }
```
**Effort**: 1-2 hours

#### B. Complete Repository Training
**Why**: Full corpus enables better queries and associations
**Scope**: Train remaining ~217 Python files
**Effort**: ~30 minutes (automated)

#### C. Natural Language Query Improvements
**Why**: The `ask` command is the primary user interface
**Scope**:
- Better concept extraction from questions
- Rank results by relevance
- Include code snippets in answers
**Effort**: 2-4 hours

### Tier 2: Medium Value, Medium-Term

#### D. Semantic Code Search
**Why**: Bridge between NL queries and code entities
**Scope**:
- Index function/class docstrings
- Link NL terms to code patterns
- "Find functions that handle authentication" -> actual code
**Effort**: 4-8 hours

#### E. Incremental IDF Reindex
**Why**: Currently reindex processes ALL links even if only a few changed
**Scope**: Track which links have stale IDF and only update those
**Effort**: 2-4 hours

### Tier 3: Future Considerations

#### F. PRISM Integration
Connect cognitive graph to PRISM's Hebbian learning for:
- Reinforcement of frequently-traversed paths
- Decay of unused associations
- Emergent pattern recognition

#### G. Woven Mind Integration
Use cognitive associations in dual-process reasoning:
- Fast path: Direct word associations (System 1)
- Slow path: Multi-hop inference (System 2)

---

## Recommended Next Steps

1. **Quick win**: Fix `_idf_epoch` restoration (5 min)
2. **Foundation**: Add CLI metrics output (1-2 hours)
3. **Coverage**: Complete repository training (30 min)
4. **Value**: Improve NL query quality (2-4 hours)

---

## Session Handoff Notes

If continuing this work:
```bash
# Check current state
python -m cortical.cognitive status
python -m pytest tests/behavioral/test_idf_weighted_links_spec.py -v

# Continue training
python -m cortical.cognitive train cortical/ --incremental

# Test queries
python -m cortical.cognitive ask "How does the tokenizer work?"
```

The incremental save optimization is working but could be further improved with per-subdivision dirty tracking if save times become a bottleneck again.
