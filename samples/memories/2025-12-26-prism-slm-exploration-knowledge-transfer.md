# Knowledge Transfer: PRISM-SLM Exploration

**Date:** 2025-12-26
**Session:** PRISM-SLM Demo Exploration
**Branch:** `claude/prism-slm-demo-exploration-I8VcE`

---

## Executive Summary

Explored PRISM-SLM (Statistical Language Model with Synaptic Learning) and its integration with Woven Mind's dual-process cognitive architecture. Key findings: PRISM-SLM serves as the "Hebbian Hive" (System 1) providing fast pattern matching, while the Loom orchestrates mode switching based on surprise detection.

---

## What is PRISM-SLM?

A biologically-inspired language model that treats word transitions as synaptic connections:

| Component | Purpose | Biological Analogy |
|-----------|---------|-------------------|
| `SynapticTransition` | Word-to-word connections | Synapses between neurons |
| `TransitionGraph` | Network of all transitions | Neural network topology |
| `ContextWindow` | Recent token history | Working memory |
| `PRISMLanguageModel` | Main facade | Cognitive processor |

### Key Features

1. **Hebbian Learning** - "Neurons that fire together wire together"
   - Transitions strengthen with repeated use (`observe()` method)
   - Weight increases proportional to co-occurrence frequency

2. **Synaptic Decay** - Unused connections weaken
   - `apply_decay(factor=0.99)` reduces all weights
   - Simulates forgetting of unused patterns

3. **Temperature-Controlled Generation**
   - Low temp (0.3) = deterministic, follows strongest paths
   - High temp (2.0) = creative, more randomness

4. **Reward Learning** - Reinforcement on generation paths
   - `reward_path(path, reward=2.0)` strengthens successful sequences

---

## Integration with Woven Mind

### Architecture

```
WOVEN MIND
├── Cultured Cortex (System 2) - Slow, deliberate
│   └── Uses: PRISM-GoT + PRISM-PLN
├── The Loom (Mode Switching)
│   └── Surprise detection → routes to Hive or Cortex
├── Hebbian Hive (System 1) - Fast, automatic
│   └── Uses: PRISM-SLM  ← THIS IS WHAT WE EXPLORED
└── Consolidation Engine
    └── Transfers patterns Hive → Cortex (like sleep)
```

### Mode Switching Observations

| Input | Mode | Surprise | Explanation |
|-------|------|----------|-------------|
| `["neural", "networks"]` | FAST | — | Strong learned transition |
| `["deep", "learning"]` | SLOW | 0.800 | Weaker prediction |
| `["quantum", "computing"]` | SLOW | 0.833 | Never seen |

**Key Insight:** Surprise threshold is 0.3. Above that triggers SLOW mode (Cortex engagement).

### PRISM-SLM Methods Used by Woven Mind

| Method | Role in Woven Mind |
|--------|-------------------|
| `train(text)` | Build Hive's pattern knowledge |
| `generate_next(context)` | Generate predictions for surprise detection |
| `spreading_activation()` | Associative retrieval |
| `lateral_inhibition()` | Sparse coding (5-10% active) |
| `k_winners_take_all()` | Competition for representation |

---

## Experimental Results

### Training Statistics (269-file corpus)

| Metric | Value |
|--------|-------|
| Vocabulary | 10,746 tokens |
| Transitions | 367,502 |
| Total tokens | 154,355 |

### Generation Examples

Prompt: "What is a neural..."
- T=0.5: "What is a neural networks to focus selectively on relevant portions of input data."
- T=1.0: Similar (dominant path)
- T=1.5: Similar (dominant path)

**Finding:** Temperature had minimal effect due to strong Hebbian dominance of certain paths.

### Perplexity Analysis

| Sentence | Perplexity | Interpretation |
|----------|------------|----------------|
| "The neural network learns patterns" | 273 | Moderately familiar |
| "Memory consolidation during sleep" | 905 | Less common phrasing |
| "Xyzzy foobar gibberish" | 388M+ | Complete nonsense |

**Finding:** Low perplexity = "sounds like training data", not "makes sense generally".

### Spreading Activation (from "neural")

```
neural          ████████████████████ 1.000 (seed)
networks        █████               0.292
learn           █████               0.273
connections     █████               0.262
patterns        █████               0.262
```

### Lateral Inhibition Effect

| Token | Before | After |
|-------|--------|-------|
| neural | 0.90 | 0.58 |
| network | 0.80 | 0.30 |
| learning | 0.70 | 0.12 |

**Effect:** Strong activations suppress neighbors → sparse representations.

---

## Key Learned Associations

From small training corpus:

```
"neural" → "networks" (weight=6.1, count=51)
"neural" → "connections" (weight=1.1)
"learning" → "algorithms" (weight=1.1)
"graph" → "of" (weight=4.8)
"of thought" → "." (weight=1.9)
```

---

## Code Locations

| What | Where |
|------|-------|
| PRISM-SLM core | `cortical/reasoning/prism_slm.py` |
| Woven Mind facade | `cortical/reasoning/woven_mind.py` |
| Loom-Hive connector | `cortical/reasoning/loom_hive.py` |
| Demo script | `examples/prism_slm_demo.py` |
| Existing benchmarks | `benchmarks/woven_mind/` |

---

## Benchmark Opportunities Identified

1. **Generation Quality**
   - Coherence scoring
   - Repetition detection
   - Diversity metrics

2. **Hebbian Learning Dynamics**
   - Weight growth curves
   - Decay stability
   - Saturation detection

3. **Perplexity Calibration**
   - Correlation with human judgment
   - Domain adaptation speed

4. **Integration Performance**
   - Mode switching accuracy
   - Surprise calibration
   - Hive-Cortex handoff latency

5. **Sparse Coding Efficiency**
   - Sparsity levels achieved
   - Information preservation
   - k-winners stability

---

## Next Steps

1. Create `benchmarks/prism_slm/` module
2. Implement core benchmarks following existing pattern
3. Establish baseline metrics
4. Compare with Woven Mind benchmarks

---

## Related Documents

- `docs/roadmap-woven-prism-marriage.md` - Full integration roadmap
- `docs/research-prism-woven-mind-comparison.md` - Comparative analysis
- `docs/woven-mind-user-guide.md` - User documentation
- `benchmarks/woven_mind/` - Existing benchmark infrastructure

---

*This knowledge transfer captures exploration session findings for future reference.*
