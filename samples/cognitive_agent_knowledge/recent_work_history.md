# Recent Work History

## What Was Done and Why

This document captures recent development work to help future agents understand the current state.

## Session: January 12, 2026

### IDF Implementation Recovery

**Context**: Code was being developed on branch `claude/review-idf-spec-cVA8B` but the session was interrupted. Work was recovered and merged.

**What was implemented**:
1. IDF (Inverse Document Frequency) weighting for similarity links
2. Dual value storage: `raw_strength` and `idf_strength`
3. Staleness tracking via `last_reindex_doc_count`
4. `--reindex` CLI command to refresh IDF weights
5. 18 BDD tests in `test_idf_weighted_links_spec.py`

**Key files changed**:
- `cortical/cognitive/text_bridge.py` - BPETokenizer IDF tracking
- `cortical/cognitive/training.py` - Staleness tracking, reindex method
- `cortical/cognitive/__main__.py` - CLI reindex command

### Performance Optimization

**Problem discovered**: `predict_next()` was O(n), taking 55-70ms per call.

**Root cause**: Scanning all 248k FOLLOWS links to find those starting from a specific word.

**Fix implemented**: Added `_outgoing` index to InMemoryStorage for O(1) lookup.

**Files changed**:
- `cortical/cognitive/graph.py`:
  - Added `_outgoing: Dict[str, Set[str]]` to InMemoryStorage
  - Added `get_outgoing()` method to InMemoryStorage and CognitiveGraph
  - Updated `predict_next()` to use indexed lookup

**Performance improvement**:
- Before: 55-70ms per call
- After: <1ms for rare words, ~20ms for common words

### Incremental Save Optimization

**Problem discovered**: Every save rewrote ALL 522k atoms, taking 54 seconds.

**Root cause**: No dirty tracking - full rewrite every time.

**Fix implemented**: Added dirty atom tracking and incremental sharded saves.

**Files changed**:
- `cortical/cognitive/graph.py`:
  - Added `_dirty_atoms: Set[str]` to InMemoryStorage
  - Added `_all_dirty: bool` flag
  - Added `clear_dirty()`, `mark_all_clean_after_load()` methods

- `cortical/cognitive/graph_storage.py`:
  - Modified `save()` to check dirty state
  - Only rewrites shards containing dirty atoms
  - Clears dirty state after successful save

- `cortical/cognitive/training.py`:
  - Added `mark_all_clean_after_load()` call after loading

**Performance improvement**:
- No changes: 0.05s (was 54s) - 1000x faster
- Incremental: ~10s (was 54s) - 5x faster

### Integration Tests Added

**Purpose**: Validate cognitive agent works correctly for real-world queries.

**File created**: `tests/integration/test_cognitive_agent_queries.py`

**Test coverage**:
- Natural language queries (associations, predictions)
- Code entity queries (atoms, links)
- Context recovery scenarios
- Performance contracts
- Incremental save integrity

**19 tests**, all passing.

### Knowledge Base Created

**Purpose**: Self-teaching samples to help future agents understand the system.

**Files created in `samples/cognitive_agent_knowledge/`**:
- `what_is_cognitive_agent.md` - Identity and purpose
- `design_decisions.md` - Rationale for choices
- `questions_and_answers.md` - Q&A for context recovery
- `context_recovery_scenarios.md` - How to recover when lost
- `architecture_overview.md` - System structure
- `recent_work_history.md` - This file

## Current Model State

As of January 12, 2026:
- **Documents trained**: 660 (531 samples + 129 code files)
- **Vocabulary size**: 21,358 terms
- **Total atoms**: ~553,000
- **SIMILARITY links**: ~218,000
- **FOLLOWS links**: ~267,000
- **Staleness**: 0% (after reindex)

## Known Issues (Low Priority)

1. **`_idf_epoch` not restored on load**: Bridge shows epoch 0 instead of manifest epoch. Minor - doesn't affect functionality.

2. **Save still ~10s for small changes**: When touching FOLLOWS or SIMILARITY, entire shard must be rewritten. Could improve with per-subdivision dirty tracking.

## Branch State

Working branch: `claude/recover-idf-spec-code-6yA7K`

Recent commits:
```
05bbe060 test(integration): Add cognitive agent NL query tests
47ed22b9 docs: Add cognitive agent roadmap and session plan
53df71ce perf(cognitive): Add incremental save with dirty tracking
84b5a3a3 perf(cognitive): Add O(1) _outgoing index for predict_next()
```

## Next Steps (from Roadmap)

1. **Quick win**: Fix `_idf_epoch` restoration (5 min)
2. **Foundation**: Add CLI metrics output (1-2 hours)
3. **Coverage**: Complete repository training (~200 more Python files)
4. **Value**: Improve NL query quality (2-4 hours)

## How to Continue This Work

```bash
# 1. Check current state
python -m cortical.cognitive status

# 2. Run tests to verify nothing broke
python -m pytest tests/smoke/ tests/behavioral/test_idf_weighted_links_spec.py tests/integration/test_cognitive_agent_queries.py -v

# 3. Train on knowledge samples
python -m cortical.cognitive train samples/cognitive_agent_knowledge/ --incremental

# 4. Test knowledge queries
python -m cortical.cognitive ask "What is the cognitive agent?"
python -m cortical.cognitive ask "Why does IDF weighting matter?"
```
