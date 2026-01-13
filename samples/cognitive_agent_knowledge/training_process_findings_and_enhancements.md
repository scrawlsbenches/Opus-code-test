# Training Process Findings & Enhancement Proposals

*Date: 2026-01-12 | Session: claude/recover-idf-cognitive-agent-gnH6g*

---

## Executive Summary

During routine validation, I discovered the cognitive agent had **not been trained on its own source code**. The model knew about samples but couldn't answer questions about `cortical/`, `tests/`, or `docs/`. This document analyzes how this happened and proposes enhancements to prevent it.

---

## Findings

### Finding 1: Critical Training Gap

**Problem**: The model was trained on ~669 documents, almost exclusively from `samples/`. Key directories had 0% coverage:

| Directory | Coverage Before | Files |
|-----------|-----------------|-------|
| cortical/ | 0% | 271 |
| tests/ | 0% | 478 |
| docs/ | 0% | 306 |
| scripts/ | 0% | 115 |

**Impact**: When asked "What is TextToAtomsBridge?", the model couldn't answer meaningfully because it had never seen `cortical/cognitive/text_bridge.py`.

**Root Cause**: No automated process ensures the model is trained on its own codebase. Training was done ad-hoc on `samples/` only.

---

### Finding 2: Path Format Inconsistency

**Problem**: Some files were trained with a `code:` prefix, others without:
```
code:cortical/analysis/__init__.py  (from index-code)
cortical/analysis/__init__.py       (from train command)
```

**Impact**: 136 files were effectively trained twice, wasting time and potentially skewing IDF weights.

**Root Cause**: The `index-code` command uses a `code:` prefix to namespace code entities, but the `train` command uses raw paths. No normalization occurs.

---

### Finding 3: No Coverage Visibility

**Problem**: There's no way to see training coverage without manual investigation:
```bash
# Currently requires custom Python scripts to check coverage
python3 -c "import json; ..."  # 20+ lines of analysis code
```

**Impact**: Training gaps go unnoticed until someone manually audits.

**Root Cause**: The `status` command only shows totals, not coverage by directory.

---

### Finding 4: No Self-Awareness Validation

**Problem**: No test verifies the model can answer questions about itself.

**Impact**: A model that can't explain its own components is significantly less useful for context recovery.

**Root Cause**: Tests focus on functional correctness, not knowledge completeness.

---

## Enhancement Proposals

### Proposal 1: Add `coverage` CLI Command

**Priority**: HIGH
**Effort**: Medium (1-2 hours)

Add a command to show training coverage:

```bash
$ python -m cortical.cognitive coverage

Training Coverage Report
========================
Directory            Covered    Total      Coverage
--------------------------------------------------
cortical/            267        271        98.5% ✓
tests/               477        478        99.8% ✓
docs/                302        306        98.7% ✓
samples/             543        682        79.6% ⚠
scripts/             112        115        97.4% ✓
--------------------------------------------------
OVERALL              1701       1852       91.8%

Untrained files in key directories:
  samples/: 139 files (run: python -m cortical.cognitive train samples/)
```

**Implementation Location**: `cortical/cognitive/training.py` - add `_run_coverage()` function

---

### Proposal 2: Bootstrap Self-Training

**Priority**: HIGH
**Effort**: Low (30 min)

Modify `bootstrap_cognitive.sh` to check and train on `cortical/` if coverage is low:

```bash
# In bootstrap_cognitive.sh
check_self_training() {
    coverage=$(python -m cortical.cognitive coverage --json | jq '.cortical.coverage')
    if (( $(echo "$coverage < 80" | bc -l) )); then
        echo "Self-training: cortical/ coverage is ${coverage}%"
        python -m cortical.cognitive train cortical/ --pattern "*.py"
    fi
}
```

**Rationale**: The model should always know about itself. This is table stakes for context recovery.

---

### Proposal 3: Path Normalization

**Priority**: MEDIUM
**Effort**: Medium (1 hour)

Normalize paths before storing in manifest:

```python
# In TrainingManifest.add_document()
def add_document(self, path: str, ...):
    # Normalize: remove code: prefix, convert to relative path
    path = path.removeprefix('code:')
    path = str(Path(path))  # Normalize slashes

    # Skip if already trained (check both with and without prefix)
    if path in self.documents:
        return
    ...
```

**Benefit**: Eliminates duplicate training, consistent manifest format.

---

### Proposal 4: Self-Awareness Test

**Priority**: MEDIUM
**Effort**: Low (30 min)

Add a behavioral test that validates the model knows about itself:

```python
# tests/behavioral/test_cognitive_self_awareness.py

class TestCognitiveSelfAwareness:
    """Verify the model can answer questions about itself."""

    def test_knows_about_text_bridge(self, trained_agent):
        """Model should know about TextToAtomsBridge."""
        response = trained_agent.ask("What is TextToAtomsBridge?")
        # Should mention: bridge, text, atoms, graph
        assert any(word in response.lower() for word in ['bridge', 'atom', 'graph'])

    def test_knows_about_cognitive_graph(self, trained_agent):
        """Model should know about CognitiveGraph."""
        response = trained_agent.ask("What is CognitiveGraph?")
        assert any(word in response.lower() for word in ['graph', 'atom', 'storage'])

    def test_knows_about_incremental_trainer(self, trained_agent):
        """Model should know about IncrementalTrainer."""
        associations = trained_agent.get_associations("incrementaltrainer")
        assert len(associations) > 0
```

**Integration**: Run as part of `test-precommit` to catch training regressions.

---

### Proposal 5: Training Onboarding Command

**Priority**: LOW
**Effort**: Medium (1 hour)

Add a single command that fully trains the model:

```bash
$ python -m cortical.cognitive onboard

Cognitive Agent Onboarding
==========================
Step 1/4: Training on cortical/ (267 files)...
Step 2/4: Training on tests/ (478 files)...
Step 3/4: Training on docs/ (306 files)...
Step 4/4: Training on samples/ (682 files)...

Reindexing IDF weights...

Onboarding complete:
  Documents: 1733
  Vocabulary: 29590 words
  Coverage: 91.8%
```

**Benefit**: New contributors can fully train the model with one command.

---

### Proposal 6: IDF Staleness Auto-Reindex

**Priority**: LOW
**Effort**: Low (15 min)

Auto-reindex when staleness exceeds threshold:

```python
# In IncrementalTrainer.train_directory()
def train_directory(self, ...):
    # ... training code ...

    # Auto-reindex if very stale
    staleness = self.manifest.get_staleness()
    if staleness > 0.5:  # 50% threshold
        print(f"Auto-reindexing (staleness: {staleness:.0%})...")
        self.reindex(show_progress=False)
```

**Benefit**: IDF weights stay accurate without manual intervention.

---

## Implementation Priority

| Proposal | Priority | Effort | Impact |
|----------|----------|--------|--------|
| 1. Coverage command | HIGH | Medium | Visibility |
| 2. Bootstrap self-training | HIGH | Low | Self-awareness |
| 3. Path normalization | MEDIUM | Medium | Data quality |
| 4. Self-awareness test | MEDIUM | Low | Regression prevention |
| 5. Onboarding command | LOW | Medium | Developer experience |
| 6. Auto-reindex | LOW | Low | Maintenance reduction |

---

## How This Session Found the Issue

### Investigation Steps

1. **Started with validation** - Ran `bootstrap_cognitive.sh --check`
2. **Noticed generic responses** - "What is cognitive agent?" gave vague answers
3. **Checked manifest** - Found only 669 documents, mostly samples/
4. **Compared to repository** - Found 2000+ trainable files
5. **Identified gaps** - cortical/, tests/, docs/ all at 0%
6. **Trained iteratively** - One directory at a time, validating each

### Key Insight

The `python -m cortical.cognitive status` command shows totals but not coverage. Without explicitly comparing manifest entries to repository files, the gap was invisible.

### Commands Used for Investigation

```bash
# Check what's in manifest
python3 -c "
import json
m = json.load(open('models/cognitive_agent/training_manifest.json'))
print(f'Total: {len(m[\"documents\"])}')
# Group by directory...
"

# Compare to repository
find cortical -name "*.py" | wc -l  # 271 files
# vs manifest entries starting with cortical: 0
```

---

## Conclusion

The cognitive agent's value comes from knowing the codebase. A model that can't answer "What is GoTManager?" is not useful for context recovery. The enhancements proposed here ensure:

1. **Visibility** - Know what's trained and what's not
2. **Automation** - Self-train on first run
3. **Validation** - Tests catch training regressions
4. **Quality** - Consistent data, no duplicates

The highest impact, lowest effort improvement is adding the `coverage` command. This single addition would have immediately revealed the training gap.

---

*This document is training data for the cognitive agent. Future sessions can ask "What training issues were found?" and get this context.*
