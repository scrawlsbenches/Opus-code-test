# Knowledge Transfer: PRISM Synaptic Learning Framework

**Date:** 2025-12-25
**Session:** claude/synaptic-memory-graph-ogYHo
**Author:** Claude (Opus 4.5)

---

## Executive Summary

This session implemented two biologically-inspired learning systems based on the PRISM (Predictive Reasoning through Incremental Synaptic Memory) architecture:

1. **PRISM-GoT** - Graph of Thought with synaptic learning for reasoning
2. **PRISM-SLM** - Statistical Language Model for text generation

Both systems use Hebbian plasticity ("neurons that fire together wire together") to strengthen connections through use and decay unused connections over time.

---

## What Was Built

### 1. PRISM-GoT (`cortical/reasoning/prism_got.py`)

A reasoning framework that extends ThoughtGraph with synaptic learning.

**Core Classes:**

| Class | Purpose |
|-------|---------|
| `SynapticEdge` | Edge that learns from activation (weight, decay, prediction accuracy) |
| `ActivationTrace` | Tracks when nodes fire for temporal correlation |
| `ActivationRecord` | Single activation event with timestamp and context |
| `SynapticMemoryGraph` | ThoughtGraph extended with Hebbian learning |
| `PlasticityRules` | Hebbian, anti-Hebbian, and reward-based learning rules |
| `IncrementalReasoner` | Builds graphs incrementally through experience |
| `PredictionResult` | Predicted next thought with confidence |

**Key Features:**
- Edges strengthen when source and target activate together (Hebbian)
- Unused connections decay over time (temporal decay)
- Prediction accuracy tracked with Beta prior smoothing
- Reward-based reinforcement of successful reasoning paths
- Auto-linking of similar content via Jaccard similarity

**Usage:**
```python
from cortical.reasoning import SynapticMemoryGraph, IncrementalReasoner, NodeType

graph = SynapticMemoryGraph()
reasoner = IncrementalReasoner(graph)

# Process thoughts incrementally
node1 = reasoner.process_thought("What is PageRank?", NodeType.QUESTION)
node2 = reasoner.process_thought("PageRank is a graph algorithm", NodeType.ANSWER,
                                  relation_to_focus="ANSWERS")

# Predict next thoughts
predictions = reasoner.predict_next(node1.id, top_n=3)

# Reinforce successful paths
reasoner.mark_outcome_success([node1.id, node2.id], reward=1.0)
```

### 2. PRISM-SLM (`cortical/reasoning/prism_slm.py`)

A statistical language model that treats word transitions as synaptic connections.

**Core Classes:**

| Class | Purpose |
|-------|---------|
| `SynapticTransition` | Connection between tokens that strengthens with use |
| `ContextWindow` | Sliding window for n-gram context tracking |
| `TransitionGraph` | Network of all token transitions |
| `PRISMLanguageModel` | Main model for training and generation |

**Key Features:**
- Variable context size (n-gram like)
- Temperature-controlled sampling for generation
- Hebbian strengthening of repeated patterns
- Decay for unused transitions
- Reward-based path reinforcement
- Perplexity computation for evaluation
- Model serialization (save/load)

**Usage:**
```python
from cortical.reasoning import PRISMLanguageModel

model = PRISMLanguageModel(context_size=3)

# Train on text
model.train("The quick brown fox jumps over the lazy dog.")

# Generate text
text = model.generate("The quick", max_tokens=10, temperature=1.0)

# Apply decay to unused transitions
model.apply_decay(factor=0.9)

# Reward good generations
result = model.generate("The", max_tokens=5, return_path=True)
model.reward_path(result["path"], reward=1.0)

# Evaluate perplexity
ppl = model.perplexity("Test sentence.")

# Save/load
model.save("model.json")
loaded = PRISMLanguageModel.load("model.json")
```

### 3. NLU Demo Enhancement (`examples/prism_got_nlu_demo.py`)

Enhanced the NLU demo to report unknown terms:
- `ask()` now returns `(results, unknown_terms)` tuple
- Clear messaging when query words aren't in corpus
- Warning symbol (⚠) for unknown terms

---

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| `tests/unit/test_prism_got.py` | 33 | ✅ All passing |
| `tests/unit/test_prism_slm.py` | 25 | ✅ All passing |
| **Total** | **58** | ✅ |

---

## Performance

### PRISM-SLM Training

| Metric | Value |
|--------|-------|
| Training time (190 files) | ~2 seconds |
| Vocabulary size | 9,486 tokens |
| Transition count | 240,451 |
| Total tokens processed | 91,777 |

### Generation Quality

The model produces coherent corpus-style text:
- "The neural subsystem handles perception and pattern matching while symbolic reasoning ensures logical consistency"
- "Graph algorithms path finding, centrality, clustering reveal structure not visible in raw"

---

## Files Created/Modified

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `cortical/reasoning/prism_got.py` | ~700 | PRISM-GoT implementation |
| `cortical/reasoning/prism_slm.py` | ~400 | PRISM-SLM implementation |
| `tests/unit/test_prism_got.py` | ~400 | PRISM-GoT tests |
| `tests/unit/test_prism_slm.py` | ~350 | PRISM-SLM tests |
| `examples/prism_got_demo.py` | ~150 | Basic PRISM-GoT demo |
| `examples/prism_got_nlu_demo.py` | ~300 | NLU demo with corpus |
| `examples/prism_got_comprehensive_demo.py` | ~250 | Full 8-phase demo |
| `examples/prism_slm_demo.py` | ~180 | PRISM-SLM demo |
| `samples/prism_got_overview.txt` | ~100 | Static documentation |

### Modified Files

| File | Change |
|------|--------|
| `cortical/reasoning/__init__.py` | Added exports for PRISM-GoT and PRISM-SLM |

---

## Key Algorithms

### Hebbian Learning
```
If source and target activate within time_window:
    edge.weight += learning_rate
```

### Temporal Decay
```
edge.weight *= decay_factor  # e.g., 0.99 per epoch
```

### Prediction Accuracy (Beta Prior Smoothing)
```
accuracy = (successes + 1) / (total + 2)  # Laplace smoothing
```

### Temperature Sampling
```
prob[i] = (weight[i] / max_weight) ^ (1 / temperature)
```

---

## Commits

1. `feat(reasoning): Add PRISM-GoT synaptic memory graph implementation`
2. `feat(examples): Add PRISM-GoT demo with synthetic data`
3. `feat(examples): Add PRISM-GoT corpus demo with real documents`
4. `feat(examples): Add comprehensive PRISM-GoT demo with full corpus`
5. `feat(examples): Add PRISM-GoT natural language understanding demo`
6. `feat(nlu): Report unknown terms when query words not in corpus`
7. `feat(reasoning): Add PRISM-SLM statistical language model with synaptic learning`

---

## Running the Demos

```bash
# PRISM-GoT NLU (answer questions from corpus)
python examples/prism_got_nlu_demo.py

# PRISM-SLM (text generation)
python examples/prism_slm_demo.py

# PRISM-GoT comprehensive (8-phase demo)
python examples/prism_got_comprehensive_demo.py

# Run all PRISM tests
python -m pytest tests/unit/test_prism_*.py -v
```

---

## Architecture Decisions

### Why Extend ThoughtGraph?
- Reuses existing node/edge infrastructure
- Adds synaptic learning without breaking existing code
- Natural fit for reasoning chains

### Why Variable Context Size in SLM?
- Captures both short-range and long-range dependencies
- Falls back to shorter contexts when longer ones have no data
- More robust than fixed n-gram models

### Why Beta Prior for Prediction Accuracy?
- Handles cold start (few observations)
- Smooth degradation with uncertainty
- Standard Bayesian approach for binary outcomes

---

## Future Enhancements

1. **Attention mechanism** - Weight recent activations more heavily
2. **Forgetting curves** - Non-linear decay based on activation history
3. **Transfer learning** - Share learned weights between graphs
4. **Hierarchical contexts** - Multi-scale temporal windows
5. **Online learning** - Continuous updates during generation

---

## Branch Information

- **Branch:** `claude/synaptic-memory-graph-ogYHo`
- **Status:** Pushed and up to date
- **Base:** Main development branch

---

*Generated by Claude Code session on 2025-12-25*
