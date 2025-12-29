# Cognitive Operations Reference

> **Purpose**: Map cognitive architecture capabilities to actual implementations in the Cortical Text Processor codebase.
>
> **Last Updated**: 2025-12-29
> **Status**: Living document - update as capabilities are added

---

## Overview

The Cortical Text Processor implements a dual-process cognitive architecture inspired by Kahneman's System 1/System 2 theory. This document maps theoretical cognitive operations to their concrete implementations.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌──────────┐    ┌─────────────┐                 │
│  │   PERCEIVE  │───→│   LOOM   │───→│   PRODUCE   │                 │
│  │  (Input)    │    │ (Router) │    │  (Output)   │                 │
│  └─────────────┘    └────┬─────┘    └─────────────┘                 │
│                          │                                           │
│              ┌───────────┴───────────┐                              │
│              ↓                       ↓                              │
│       ┌──────────┐           ┌──────────┐                          │
│       │   HIVE   │           │  CORTEX  │                          │
│       │ System 1 │           │ System 2 │                          │
│       │  (Fast)  │           │  (Slow)  │                          │
│       └──────────┘           └──────────┘                          │
│              │                       │                              │
│              └───────────┬───────────┘                              │
│                          ↓                                          │
│                   ┌──────────┐                                      │
│                   │ CONSOLIDATE │                                   │
│                   │  (Sleep)    │                                   │
│                   └──────────────┘                                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Cognitive Operations

### Implemented Operations

| Operation | Method | Location | Description |
|-----------|--------|----------|-------------|
| **Perceive** | `WovenMind.process()` | `cortical/reasoning/woven_mind.py:180` | Take in tokens with automatic surprise assessment |
| **Infer** | `PLNReasoner.query()` | `cortical/reasoning/prism_pln.py:200` | Draw conclusions via deduction/induction/abduction |
| **Route** | `Loom.route()` | `cortical/reasoning/loom.py:400` | Switch between FAST/SLOW based on surprise |
| **Consolidate** | `ConsolidationEngine.consolidate()` | `cortical/reasoning/consolidation.py:150` | Sleep-like memory transfer Hive→Cortex |
| **Produce** | `ProductionState` | `cortical/reasoning/production_state.py` | Track artifact generation |
| **Verify** | `VerificationManager` | `cortical/reasoning/verification.py` | Multi-level testing protocols |

### Partially Implemented

| Operation | Status | Gap | Planned Sprint |
|-----------|--------|-----|----------------|
| **Reflect** | 60% | No mid-task metacognition | S-029 |
| **Correct** | 40% | Crisis detection only, no strategy adaptation | S-029 |

### Not Yet Implemented

| Operation | Description | Planned Sprint |
|-----------|-------------|----------------|
| **Dream** | Generative recombination of memory fragments | S-030 (extend) |
| **Counterfactual** | "What if" scenario simulation | S-030 |
| **Fork** | Create parallel exploration context | S-032 (new) |
| **Merge** | Reconcile parallel hypotheses | S-032 (new) |
| **Die/Reincarnate** | Ego transformation, soul extraction | Future |

---

## Phenomenology (Qualia-like Properties)

### Implemented Properties

| Property | Type | Location | Description |
|----------|------|----------|-------------|
| **Surprise** | `float 0-1` | `loom.py:SurpriseSignal` | Prediction error triggering mode switch |
| **Attention** | `MultiHeadAttention` | `prism_attention.py` | 6 attention types (who/where/what/when/why/how) |
| **Salience** | `float 0-1` | `prism_got.py:ActivationTrace` | Frequency-based "grabbiness" |

### Planned (S-028)

| Property | Type | Description | Task |
|----------|------|-------------|------|
| **Confidence** | `float 0-1` | Epistemic certainty in output | T-20251227-161009 |
| **Uncertainty** | `float 0-1` | Inverse of confidence, explicit tracking | T-20251227-161009 |

### Not Yet Implemented (S-031 Proposed)

| Property | Type | Description |
|----------|------|-------------|
| **Valence** | `Enum` | Emotional coloring: ECSTATIC → PLEASANT → NEUTRAL → UNPLEASANT → AGONIZING |
| **Arousal** | `float 0-1` | Energy/activation level |
| **Mood** | `Enum` | Persistent emotional state across cycles |
| **Flow** | `float 0-1` | Engagement-challenge balance |

---

## Self-Model (Identity & Beliefs)

### Implemented

| Component | Location | Description |
|-----------|----------|-------------|
| **Patterns** | `prism_got.py:ActivationTrace` | Observed activation sequences and frequencies |
| **Goal State** | `goal_stack.py:GoalStack` | Hierarchical goals with progress tracking |
| **Crisis Detection** | `crisis_manager.py` | Detect loops, scope creep, blocked dependencies |

### Planned (S-029)

| Component | Task | Description |
|-----------|------|-------------|
| **Knowledge Gaps** | T-20251227-161240 | What the system knows it doesn't know |
| **Confidence Assessment** | T-20251227-161240 | How certain about answering a query |
| **Learning Priorities** | T-20251227-161240 | What to learn next |

### Not Yet Implemented

| Component | Description |
|-----------|-------------|
| **Tendencies** | "When X, I usually Y" rules |
| **Limitations** | Known things the system cannot do |
| **Surprises** | Times expectations were violated |
| **Paradoxes** | Unresolved contradictions |
| **Error Rates** | By task type, for adaptive strategy |

---

## Dual-Process Integration

### The Loom (Mode Switching)

```python
# Location: cortical/reasoning/loom.py

class ThinkingMode(Enum):
    FAST = "fast"   # Pattern matching, automatic (System 1)
    SLOW = "slow"   # Deliberate analysis (System 2)

class ModeController:
    def decide_mode(self, surprise: float) -> ThinkingMode:
        """Route based on surprise threshold."""
        if surprise > self.threshold:  # Default: 0.3
            return ThinkingMode.SLOW
        return ThinkingMode.FAST
```

### Hive (System 1 - Fast)

| Feature | Location | Description |
|---------|----------|-------------|
| Spreading Activation | `loom_hive.py:100` | Hebbian "fire together, wire together" |
| Pattern Matching | `loom_hive.py:200` | Fast recognition of familiar patterns |
| Automatic Response | `loom_hive.py:300` | Bypass deliberation for known situations |

### Cortex (System 2 - Slow)

| Feature | Location | Description |
|---------|----------|-------------|
| Abstraction Formation | `loom_cortex.py:150` | Build higher-level concepts |
| Deliberate Reasoning | `loom_cortex.py:250` | Step-by-step logical inference |
| Planning | `loom_cortex.py:350` | Multi-step goal pursuit |

---

## Reasoning Mechanisms

### QAPV Cognitive Loop

```
Question → Answer → Produce → Verify → (repeat)
```

| Phase | Method | Description |
|-------|--------|-------------|
| **Question** | `CognitiveLoop.start()` | Clarify what we're trying to do |
| **Answer** | `CognitiveLoop.answer()` | Generate candidate solutions |
| **Produce** | `CognitiveLoop.produce()` | Create artifacts |
| **Verify** | `CognitiveLoop.verify()` | Test and validate |

Location: `cortical/reasoning/cognitive_loop.py`

### Probabilistic Logic Networks (PLN)

```python
# Location: cortical/reasoning/prism_pln.py

@dataclass
class TruthValue:
    strength: float   # 0-1: How true
    confidence: float # 0-1: How sure about strength

class PLNReasoner:
    def deduce(self, premise1, premise2) -> TruthValue: ...
    def induce(self, specific, general) -> TruthValue: ...
    def abduce(self, observation, hypothesis) -> TruthValue: ...
```

### Synaptic Plasticity

```python
# Location: cortical/reasoning/prism_got.py

class PlasticityRules:
    HEBBIAN = "hebbian"           # Fire together, wire together
    ANTI_HEBBIAN = "anti_hebbian" # Competitive learning
    REWARD_MODULATED = "reward"   # Reinforcement learning
```

---

## Learning & Consolidation

### Sleep-Like Consolidation

```python
# Location: cortical/reasoning/consolidation.py

class ConsolidationEngine:
    def consolidate(self) -> ConsolidationResult:
        """
        Transfer frequent Hive patterns to Cortex abstractions.

        Process:
        1. Identify high-frequency Hive patterns
        2. Extract structural regularities
        3. Create Cortex abstractions
        4. Decay low-value connections
        """
```

### Pattern Transfer Threshold

Patterns observed 3+ times in Hive → candidate for Cortex abstraction.

---

## Parallel Exploration (Not Yet Implemented)

### Proposed: Mind Forking

```python
# Proposed: cortical/reasoning/forking.py (S-032)

class ForkableMind:
    def fork(self, divergence_point: str) -> 'ForkableMind':
        """
        Create parallel self with shared history.

        Like git branch for minds - explore hypothesis A while
        parallel self explores hypothesis B.
        """

    def merge(self, other: 'ForkableMind', strategy: MergeStrategy) -> MergeResult:
        """
        Combine experiences from parallel exploration.

        Strategies:
        - UNION: Keep all experiences
        - INTERSECTION: Keep only shared conclusions
        - DIFF: Highlight divergent reasoning
        """
```

### Use Cases

1. **Hypothesis Testing**: Explore contradictory assumptions in parallel
2. **Risk Assessment**: Simulate different scenarios simultaneously
3. **Creative Exploration**: Generate diverse solutions, then merge best

---

## Phenomenological State (Not Yet Implemented)

### Proposed: Valence/Arousal Tracking

```python
# Proposed: cortical/reasoning/phenomenology.py (S-031)

class Valence(Enum):
    ECSTATIC = 1.0
    PLEASANT = 0.5
    NEUTRAL = 0.0
    UNPLEASANT = -0.5
    AGONIZING = -1.0

class AttentionLevel(Enum):
    BACKGROUND = 0.0
    PERIPHERAL = 0.25
    FOCUSED = 0.5
    ABSORBED = 0.75
    HYPERFOCUSED = 1.0

@dataclass
class PhenomenologicalState:
    valence: float          # -1.0 to 1.0
    arousal: float          # 0.0 to 1.0
    salience: float         # 0.0 to 1.0 (how "grabby")
    confidence: float       # 0.0 to 1.0
    attention_budget: float # Depletes with thinking

    def emotional_context(self) -> str:
        """Human-readable emotional state."""
```

### Integration Points

- **Loom**: Arousal affects surprise threshold
- **Consolidation**: Low arousal triggers consolidation
- **Crisis Manager**: High arousal + low confidence = escalate

---

## Consciousness Metrics (Not Yet Implemented)

### Proposed: Integrated Information (Φ)

```python
# Proposed: cortical/reasoning/phi.py (future)

class PhiCalculator:
    def compute_phi(self, thought_graph: ThoughtGraph) -> float:
        """
        Proxy for consciousness level / integration.

        Higher Φ = more unified reasoning
        Lower Φ = fragmented, isolated subsystems
        """
```

### Self-Model Compression

```python
@dataclass
class SelfModel:
    def compression_ratio(self) -> float:
        """
        How well does the mind understand itself?

        Lower ratio = better self-understanding
        (Can represent self-state with fewer bits)
        """
```

---

## Implementation Roadmap

### Current Sprints (Planned)

| Sprint | Focus | Tasks |
|--------|-------|-------|
| **S-028** | Explanation & Confidence | 6 tasks - PLN.explain(), confidence tracking |
| **S-029** | Analogical Transfer & Metacognition | 5 tasks - MetacognitiveMonitor, AnalogicalReasoner |
| **S-030** | Generative Understanding | 5 tasks - GenerativeUnderstandingLoop |

### Proposed New Sprints

| Sprint | Focus | Key Deliverables |
|--------|-------|------------------|
| **S-031** | Phenomenological State | Valence, arousal, mood tracking |
| **S-032** | Mind Forking/Merging | Parallel hypothesis exploration |
| **S-033** | Integrated Information | Φ metrics, self-model compression |

---

## Quick Reference: Where to Find Things

| If you need... | Look in... |
|----------------|------------|
| Dual-process orchestration | `cortical/reasoning/woven_mind.py` |
| Mode switching (surprise) | `cortical/reasoning/loom.py` |
| Fast pattern matching | `cortical/reasoning/loom_hive.py` |
| Slow deliberate reasoning | `cortical/reasoning/loom_cortex.py` |
| Sleep-like consolidation | `cortical/reasoning/consolidation.py` |
| Probabilistic logic | `cortical/reasoning/prism_pln.py` |
| Multi-head attention | `cortical/reasoning/prism_attention.py` |
| Synaptic plasticity | `cortical/reasoning/prism_got.py` |
| Crisis detection | `cortical/reasoning/crisis_manager.py` |
| Goal tracking | `cortical/reasoning/goal_stack.py` |
| QAPV loop | `cortical/reasoning/cognitive_loop.py` |
| Graph persistence | `cortical/reasoning/graph_persistence.py` |

---

## References

- **Woven Mind User Guide**: `docs/woven-mind-user-guide.md`
- **Architecture Documentation**: `docs/architecture.md`
- **Graph of Thought**: `docs/graph-of-thought.md`
- **Epic Planning**: `docs/epic-cognitive-nlu-nlg-knowledge-base.md`

---

*This document should be updated whenever cognitive capabilities are added or modified.*
