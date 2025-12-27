# Knowledge Transfer: PRISM-SLM Data Generation Pipeline

**Date:** 2025-12-27
**Branch:** `claude/prism-slm-demo-exploration-I8VcE`
**Session Focus:** Building data generation infrastructure for Repository-Native SLM

---

## Executive Summary

This session established a comprehensive data generation pipeline for training a repository-native Statistical Language Model (SLM). The work addressed a critical issue: the benchmark showed 0% accuracy on concept explanations while file locations dominated at 87.5%. We built multiple data generation techniques and tuned oversampling weights to rebalance training data.

**Key Outcomes:**
1. Built hybrid data generation pipeline combining 5 techniques
2. Fixed ML collector to save chat data to git-tracked location
3. Tuned oversampling weights (concept: 0% → 58.7%, location: 87.5% → 0.7%)
4. Created backfill prompt for collecting chat history from old threads

---

## Technical Architecture

### Data Generation Techniques

| Technique | File | Purpose | Patterns Generated |
|-----------|------|---------|-------------------|
| PLN (Probabilistic Logic Networks) | `pln_generator.py` | Logical inference with confidence propagation | ~114 patterns |
| Dialogue Generator | `dialogue_generator.py` | Natural Q&A with 4 agent personas | ~23 patterns |
| Data Augmentation | `data_augmentation.py` | Curated definitions + chat history | ~100+ patterns |
| SparkSLM (NGram) | Via `NGramModel` | Sequence completion | Variable |
| Woven Mind | Via `WovenMind` | Confidence scoring (classifier, not generator) | N/A (scores only) |

### Hybrid Pipeline (`hybrid_pipeline.py`)

Orchestrates all techniques in sequence:
```
1. PLN patterns (logical inference)
2. Dialogue patterns (natural Q&A)
3. Augmentation patterns (definitions, chat, hierarchical)
4. Spark completions (trained on above)
5. Woven Mind scoring (confidence adjustment)
6. Deduplication
```

**Output:** 208 unique patterns → 7,285 training lines (with oversampling)

---

## Critical Bug Fixes

### 1. Chat Data Not Tracked in Git

**Problem:** `save_chat_entry()` only saved to `.git-ml/chats/` which is gitignored. Backfill data couldn't be pushed.

**Solution:** Modified `scripts/ml_collector/persistence.py` to save to THREE locations:
```python
def save_chat_entry(entry: ChatEntry, validate: bool = True):
    # 1. .git-ml/chats/ - Full data (gitignored - local only)
    # 2. .git-ml/tracked/chunked/ - Compressed (git-tracked - shareable)
    # 3. CALI storage - O(1) lookups (local cache)
```

**Commit:** `045a8dbd` - "fix(ml-collector): Save chat data to git-tracked location"

### 2. Gitignore Configuration

**Correct behavior (verified):**
- `.git-ml/chats/` → IGNORED (full transcripts are large/sensitive)
- `.git-ml/tracked/chunked/` → NOT IGNORED (shareable via git)

The gitignore was correct; the code was the issue.

---

## Oversampling Weights

### Data Augmentation Pipeline (`data_augmentation.py`)

```python
weights = {
    'definition': 20,      # Concept explanations - critical for 0% concept
    'hierarchical': 15,    # Type relationships - "X is a type of Y"
    'chat_qa': 10,         # Real Q&A - high quality
    'completion': 3,       # Generated sequences - less reliable
    'pln_inference': 12,   # Logical inferences - good for relationships
    'dialogue': 8,         # Agent dialogues - natural Q&A style
}
```

**Result Distribution:**
- chat_qa: 36.4%
- hierarchical: 35.4%
- definition: 27.6%
- completion: 0.6%

### Hybrid Pipeline (`hybrid_pipeline.py`)

```python
weights = {
    # Source weights
    'source:pln': 15,
    'source:dialogue': 10,
    'source:curated_definitions': 20,
    'source:chat': 12,
    'source:spark': 4,
    'source:augmentation': 8,
    # Category multipliers
    'category:concept': 3.0,
    'category:definition': 2.5,
    'category:type': 2.0,
    'category:how_to': 1.5,
    'category:location': 0.5,  # Reduced - was dominant
    'category:completion': 0.8,
}
```

**Result Distribution:**
- concept: 58.7% (was 0%)
- definition: 14.8%
- chat: 12.1%
- type: 10.4%
- location: 0.7% (was 87.5%)

---

## Key Learnings

### 1. Woven Mind is a Classifier, Not a Generator

Woven Mind routes between FAST (Hive) and SLOW (Cortex) modes based on surprise detection. It's excellent for:
- Confidence scoring
- Familiarity detection
- Mode routing

But it does NOT generate text sequences.

### 2. SparkSLM (NGramModel) is the Generator

```python
from cortical.spark import NGramModel
ngram = NGramModel(n=3)
ngram.train(training_texts)
sequence = ngram.predict_sequence(prompt, length=5)
```

Works well when trained on domain text.

### 3. PLN Provides Logical Consistency

Forward chaining with confidence decay:
```python
@dataclass
class Rule:
    name: str
    antecedent: List[Tuple[str, str, str]]
    consequent: Tuple[str, str, str]
    confidence_decay: float = 0.9
```

45 base facts + 4 rules → 114 derived patterns

### 4. Chat History is High-Quality Training Data

Real Q&A conversations about the codebase are excellent for concept explanations. The data augmentation pipeline looks in:
```python
possible_paths = [
    PROJECT_ROOT / ".git-ml" / "tracked" / "chunked",  # Git-tracked
    PROJECT_ROOT / ".git-ml" / "chats",                # Local only
]
```

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `benchmarks/codebase_slm/pln_generator.py` | PLN with forward chaining |
| `benchmarks/codebase_slm/dialogue_generator.py` | Agent persona dialogues |
| `benchmarks/codebase_slm/hybrid_pipeline.py` | Orchestrates all techniques |
| `scripts/backfill_chat_history.py` | Git worktree-based extraction |
| `docs/backfill-chat-prompt.md` | Instructions for old thread backfill |

### Modified Files

| File | Changes |
|------|---------|
| `benchmarks/codebase_slm/data_augmentation.py` | Added tuned oversampling weights |
| `scripts/ml_collector/persistence.py` | Added chunked storage for git tracking |

---

## Backfill Process for Old Threads

### Prompt to Use

```markdown
## Backfill Chat History for ML Training

### Step 1: Pull Latest from Main
git fetch origin main && git merge origin/main --no-edit

### Step 2: Verify .gitignore
grep -n "git-ml/tracked" .gitignore

### Step 3: Start ML Session
python scripts/ml_data_collector.py session start

### Step 4: Have Conversation About Codebase
(Ask about PageRank, Woven Mind, GoTManager, etc.)

### Step 5: End Session and Push
python scripts/ml_data_collector.py session end --summary "Backfill"
git add .git-ml/tracked/
git commit -m "ml: Backfill chat history from $(git branch --show-current)"
git push origin $(git branch --show-current)
```

### After Backfilling

```bash
git fetch --all
# Merge from backfilled branches
python -m benchmarks.codebase_slm.data_augmentation
python -m benchmarks.codebase_slm.train_augmented
python -m benchmarks.codebase_slm.benchmark_suite
```

---

## Benchmark Status

**Before tuning:**
- file_location: 87.5%
- concept: 0%
- Overall: ~40%

**After tuning (with oversampling):**
- Training corpus rebalanced
- Concept patterns now 58.7% of corpus
- Model generates "pagerank is a type of component" (verified)

**Next steps to improve:**
1. Run backfill on old threads to collect more chat history
2. Retrain after collecting more data
3. Run benchmarks to measure improvement

---

## API Quick Reference

### PLN Generator
```python
from benchmarks.codebase_slm.pln_generator import ProbabilisticLogicNetwork
pln = ProbabilisticLogicNetwork()
pln.load_knowledge_base()
pln.load_inference_rules()
pln.forward_chain()
patterns = pln.generate_training_patterns()
```

### Dialogue Generator
```python
from benchmarks.codebase_slm.dialogue_generator import DialogueGenerator
gen = DialogueGenerator()
patterns = gen.generate_dialogues(num_exchanges=50)
```

### Hybrid Pipeline
```python
from benchmarks.codebase_slm.hybrid_pipeline import HybridPipeline
pipeline = HybridPipeline()
patterns = pipeline.run_full_pipeline()
pipeline.export(output_path, oversample=True)
```

### Data Augmentation
```python
from benchmarks.codebase_slm.data_augmentation import DataAugmentationPipeline
pipeline = DataAugmentationPipeline()
patterns = pipeline.run_full_pipeline(base_corpus)
pipeline.export_training_corpus(output_path, weights=custom_weights)
```

---

## Commits This Session

| Hash | Message |
|------|---------|
| `045a8dbd` | fix(ml-collector): Save chat data to git-tracked location |
| `3277b6ef` | feat(codebase-slm): Tune oversampling weights and improve backfill docs |
| `58a51287` | docs(ml): Add chat history backfill instructions for old threads |

---

## Open Questions / Future Work

1. **Optimal weight tuning:** Current weights are heuristic; could use validation set
2. **PLN rule expansion:** Only 4 rules currently; could add more inference patterns
3. **Dialogue diversity:** 4 personas; could add domain-specific personas
4. **Cross-session learning:** How to aggregate learnings across backfill sessions

---

**Handoff Ready:** Yes
**Branch:** `claude/prism-slm-demo-exploration-I8VcE`
**All changes committed and pushed:** Yes
