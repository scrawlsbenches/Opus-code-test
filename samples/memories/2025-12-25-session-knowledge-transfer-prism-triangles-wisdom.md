# Knowledge Transfer: PRISM Triangles of Wisdom

**Date:** 2025-12-25
**Session:** Synaptic Memory Graph (ogYHo)
**Branch:** `claude/synaptic-memory-graph-ogYHo`

## Session Summary

This session completed the PRISM trilogy and charted the path forward with aspirational tests for attention mechanisms and causal reasoning. We also created interconnected corpus documents forming "triangles of wisdom."

## What Was Accomplished

### 1. Completed PRISM Trilogy (Previous Session)
Three interconnected reasoning systems, all with synaptic learning:

| System | Purpose | Key Classes |
|--------|---------|-------------|
| **PRISM-GoT** | Graph of Thought with Hebbian learning | `PRISMGraph`, `SynapticEdge` |
| **PRISM-SLM** | Statistical Language Model | `PRISMLanguageModel`, `SynapticTransition` |
| **PRISM-PLN** | Probabilistic Logic Networks | `PLNReasoner`, `TruthValue` |

### 2. Triangles of Wisdom (This Session)
Created three corpus documents that cross-reference each other:

```
        synaptic_memory_hebbian_learning.txt
                     /\
                    /  \
                   /    \
                  /      \
statistical_language_    probabilistic_logic_
models_synaptic.txt      networks.txt
```

Each document's "See Also" references the other two, creating a triangle of wisdom for semantic search and context recovery.

### 3. Aspirational Tests (This Session)
Defined future capabilities with skipped tests:

**PRISM-Attention** (`tests/behavioral/test_prism_attention.py`):
- Query-Key-Value attention over thought graphs
- Multi-head attention for relation types
- Synaptic gating respecting edge strengths
- Learnable attention from reinforcement
- Temporal attention over sequences
- Cross-system integration

**PRISM-Causal** (`tests/behavioral/test_prism_causal.py`):
- Intervention vs observation (do-calculus)
- Counterfactual reasoning ("What if...?")
- Necessary vs sufficient cause analysis
- Causal discovery from data
- Hidden confounder detection
- Causal explanation generation

## Key Technical Details

### File Locations

```
cortical/reasoning/
├── prism_slm.py          # Statistical Language Model (~400 lines)
├── prism_pln.py          # Probabilistic Logic Networks (~700 lines)
├── thought_graph.py      # Extended with synaptic edges
└── [future] prism_attention.py
└── [future] prism_causal.py

tests/
├── unit/
│   ├── test_prism_slm.py   # 25 tests
│   └── test_prism_pln.py   # 30 tests
├── behavioral/
│   ├── test_prism_integration.py  # 11 tests (Wonderland theme)
│   ├── test_prism_attention.py    # 8 aspirational tests
│   └── test_prism_causal.py       # 11 aspirational tests

samples/
├── synaptic_memory_hebbian_learning.txt
├── statistical_language_models_synaptic.txt
└── probabilistic_logic_networks.txt
```

### Test Counts

| Category | Passing | Skipped |
|----------|---------|---------|
| PRISM-GoT | 33 | 0 |
| PRISM-SLM | 25 | 0 |
| PRISM-PLN | 30 | 0 |
| Integration | 11 | 0 |
| Attention (aspirational) | 0 | 8 |
| Causal (aspirational) | 0 | 11 |
| **Total** | **99** | **19** |

### Key Formulas

**Synaptic Truth Value (PLN):**
```python
strength = (positive + 1) / (total + 2)  # Beta prior smoothing
confidence = total / (total + 2)          # Evidence accumulation
```

**Hebbian Learning:**
```python
if source.activated AND target.activated within time_window:
    edge.weight += learning_rate
edge.weight *= decay_factor  # Anti-Hebbian decay
```

**Temperature Sampling (SLM):**
```python
adjusted = weight ** (1.0 / temperature)
# Low temp → deterministic, High temp → random
```

## The Wonderland Metaphors

Throughout the code and tests, we use Alice in Wonderland as the running metaphor:

| Concept | Wonderland Metaphor |
|---------|---------------------|
| Hebbian strengthening | "We're all mad here" - chaos becomes order |
| PLN uncertainty | Cheshire Cat's abductive reasoning |
| SLM generation | Caterpillar's "Who are YOU?" |
| Attention focus | Caterpillar on mushroom, seeing only what matters |
| Counterfactuals | "What if Alice had NOT drunk from the bottle?" |
| Causal chains | Drink → Shrink → Fit door → Enter garden |

## Roadmap (From Wonderland Roadmap Doc)

```
Chapter 1: Garden of Synapses     ✅ DONE (GoT)
Chapter 2: Forest of Patterns     ✅ DONE (SLM)
Chapter 3: Ocean of Uncertainty   ✅ DONE (PLN)
Chapter 4: Mountain of Attention  📝 DEFINED (aspirational tests)
Chapter 5: Sky of Causality       📝 DEFINED (aspirational tests)
Chapter 6: Stars of Intuition     🔮 FUTURE (meta-learning)
```

## Critical Patterns to Preserve

### 1. EdgeType Required in add_synaptic_edge
```python
# WRONG - will fail
graph.add_synaptic_edge("a", "b")

# CORRECT
graph.add_synaptic_edge("a", "b", EdgeType.SUPPORTS)
```

### 2. NodeType.ANSWER Doesn't Exist
```python
# WRONG
NodeType.ANSWER

# CORRECT
NodeType.INSIGHT  # or HYPOTHESIS, QUESTION, etc.
```

### 3. PredictionResult Access
```python
# WRONG
prediction.predicted_content

# CORRECT
prediction.node.content
```

### 4. PLN Variable Pattern Matching
The `infer()` method handles patterns like `bird(X)` matching `bird(tweety)`:
```python
# Pattern: bird(X) → can_fly(X)
# Fact: bird(tweety)
# Infers: can_fly(tweety) with substitution {X: tweety}
```

## Commits This Session

1. `docs(samples): Add PRISM triangles of wisdom corpus documents` (3 files, 320 lines)
2. `test(behavioral): Add aspirational PRISM attention and causal tests` (2 files, 640 lines)

## How to Continue

### To implement PRISM-Attention:
1. Read `tests/behavioral/test_prism_attention.py` for the interface
2. Create `cortical/reasoning/prism_attention.py`
3. Remove `@pytest.mark.skip` as you implement each test
4. Follow the Wonderland theme (Caterpillar's focused gaze)

### To implement PRISM-Causal:
1. Read `tests/behavioral/test_prism_causal.py` for the interface
2. Create `cortical/reasoning/prism_causal.py`
3. Start with intervention/observation distinction
4. Build up to counterfactuals and causal discovery

### Quick Verification
```bash
# All PRISM tests (should be 99 pass, 19 skip)
python -m pytest tests/unit/test_prism_*.py tests/behavioral/test_prism_*.py -v

# Just the implemented systems
python -m pytest tests/unit/test_prism_*.py tests/behavioral/test_prism_integration.py -v
```

## See Also

- `docs/prism-wonderland-roadmap.md` - Full visionary roadmap
- `samples/memories/2025-12-25-session-knowledge-transfer-prism-synaptic-learning.md` - Previous session
- Triangle of wisdom documents in `samples/`

---

*"Begin at the beginning and go on till you come to the end: then stop." - The King of Hearts*
