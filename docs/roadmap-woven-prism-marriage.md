# The Marriage of Woven Mind and PRISM

## A Roadmap for Unified Cognitive Architecture

---

> *"Two roads diverged in a wood, and I—*
> *I took the one less traveled by,*
> *And that has made all the difference."*
> — Robert Frost

> *"But what if the roads converge again?"*
> — This project

---

## Executive Summary

This roadmap charts the course from two separate cognitive architectures—**PRISM** (implemented) and **Woven Mind** (theoretical)—to a unified system that is greater than the sum of its parts.

**Current State:**
- PRISM: 2,974 lines of working Python code
- Woven Mind: Architectural design document

**Target State:**
- A dual-process cognitive architecture with:
  - Fast pattern matching (Hebbian Hive backed by PRISM-SLM)
  - Slow deliberate reasoning (Cultured Cortex backed by PRISM-GoT + PLN)
  - Surprise-based mode switching (The Loom via PRISM-Attention)
  - Consolidation cycles for learning transfer

**Timeline:** 6 sprints over ~12 weeks (assuming part-time development)

---

## Part I: Where We Are Now

### 1.1 PRISM Inventory

| Module | Lines | Purpose | Completeness |
|--------|-------|---------|--------------|
| `prism_got.py` | 1,148 | Synaptic memory, prediction | ██████████ 100% |
| `prism_pln.py` | 719 | Probabilistic logic | ██████████ 100% |
| `prism_attention.py` | 615 | Selective focus | ██████████ 100% |
| `prism_slm.py` | 492 | Language model | ██████████ 100% |

**PRISM Strengths:**
- Synaptic plasticity with decay and strengthening
- Prediction tracking with accuracy metrics
- Probabilistic reasoning with truth values
- Multi-head attention mechanisms
- Full test coverage

**PRISM Gaps (that Woven Mind addresses):**
- No explicit System 1/System 2 distinction
- No surprise-based mode switching
- No lateral inhibition (sparse coding)
- No homeostatic regulation
- No explicit abstraction formation
- No consolidation cycles

### 1.2 Woven Mind Inventory

| Component | Status | Purpose |
|-----------|--------|---------|
| HebbianHive | Design only | Fast pattern matching |
| CulturedCortex | Design only | Slow deliberate reasoning |
| The Loom | Design only | Mode switching, integration |
| Consolidation | Design only | Learning transfer |

**Woven Mind Strengths:**
- Clear dual-process architecture
- Cognitive plausibility
- Explicit mode switching via surprise
- Homeostatic regulation
- Abstraction as first-class concept

**Woven Mind Gaps (that PRISM addresses):**
- Not implemented
- No probabilistic reasoning
- No language modeling
- No sophisticated attention

### 1.3 The Marriage Vision

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED COGNITIVE ARCHITECTURE                       │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │              CULTURED CORTEX (System 2)                        │    │
│   │         ┌─────────────────────────────────────┐                │    │
│   │         │  PRISM-GoT + PRISM-PLN              │                │    │
│   │         │  + Explicit Abstraction Layer       │                │    │
│   │         │  + Goal-Directed Planning           │                │    │
│   │         └─────────────────────────────────────┘                │    │
│   └───────────────────────────┬────────────────────────────────────┘    │
│                               │                                          │
│   ┌───────────────────────────┼────────────────────────────────────┐    │
│   │                    THE LOOM                                     │    │
│   │  ┌─────────────────┬─────┴─────┬─────────────────┐             │    │
│   │  │ PRISM-Attention │ Surprise  │ Mode Controller │             │    │
│   │  │ (unified)       │ Detector  │ (new)           │             │    │
│   │  └─────────────────┴───────────┴─────────────────┘             │    │
│   └───────────────────────────┬────────────────────────────────────┘    │
│                               │                                          │
│   ┌───────────────────────────┴────────────────────────────────────┐    │
│   │              HEBBIAN HIVE (System 1)                            │    │
│   │         ┌─────────────────────────────────────┐                │    │
│   │         │  PRISM-SLM                          │                │    │
│   │         │  + Lateral Inhibition               │                │    │
│   │         │  + Homeostatic Regulation           │                │    │
│   │         │  + Spreading Activation             │                │    │
│   │         └─────────────────────────────────────┘                │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │              CONSOLIDATION ENGINE                               │    │
│   │         Pattern Transfer • Abstraction Mining • Decay Cycles   │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part II: The Epics

### EPIC-LOOM: The Integration Layer

**Goal:** Build the orchestration layer that decides when to think fast vs. slow.

**Success Criteria:**
- Surprise detection working with configurable thresholds
- Mode switching between Hive and Cortex
- Attention routing based on mode
- Observable mode transitions for debugging

**Dependencies:** None (can start immediately)

**Risk Level:** Medium (new code, but clear requirements)

---

### EPIC-HIVE: Enhanced Hebbian Processing

**Goal:** Extend PRISM-SLM with biological realism from Woven Mind.

**Success Criteria:**
- Lateral inhibition producing sparse activations
- Homeostatic regulation maintaining stable activation levels
- Spreading activation for associative retrieval
- k-winners-take-all competition

**Dependencies:** None (can start immediately, parallel with EPIC-LOOM)

**Risk Level:** Low (extending existing code)

---

### EPIC-CORTEX: Deliberate Reasoning Enhancement

**Goal:** Extend PRISM-GoT with explicit abstraction and goal tracking.

**Success Criteria:**
- Abstraction detection from repeated patterns
- Hierarchical concept formation
- Goal stack with progress tracking
- Integration with PRISM-PLN for uncertainty

**Dependencies:** EPIC-LOOM (needs mode switching to know when to engage)

**Risk Level:** Medium (abstraction is tricky)

---

### EPIC-CONSOLIDATION: Learning Transfer

**Goal:** Implement "sleep" cycles that transfer learning between systems.

**Success Criteria:**
- Scheduled consolidation cycles
- Pattern transfer from Hive to Cortex
- Abstraction mining from frequent patterns
- Decay application during consolidation

**Dependencies:** EPIC-HIVE, EPIC-CORTEX (needs both systems working)

**Risk Level:** High (novel algorithm design)

---

### EPIC-INTEGRATION: The Marriage Complete

**Goal:** Full integration testing and demonstration.

**Success Criteria:**
- End-to-end demo showing dual-process in action
- Performance benchmarks
- Documentation complete
- Test coverage ≥ 95%

**Dependencies:** All other epics

**Risk Level:** Low (integration, not invention)

---

## Part III: The Sprints

### Sprint 1: The Loom Foundation (Weeks 1-2)

**Epic:** EPIC-LOOM

**Theme:** *"Before we can weave, we need a loom."*

**Tasks:**

| ID | Task | Priority | Estimate | Dependencies |
|----|------|----------|----------|--------------|
| T1.1 | Design Loom interface contract | High | 2h | None |
| T1.2 | Implement SurpriseDetector class | High | 4h | T1.1 |
| T1.3 | Implement ModeController class | High | 4h | T1.1 |
| T1.4 | Wire Loom to PRISM-Attention | Medium | 3h | T1.2, T1.3 |
| T1.5 | Add observable mode transitions | Medium | 2h | T1.4 |
| T1.6 | Write unit tests for Loom | High | 3h | T1.4 |
| T1.7 | Write integration test with PRISM-GoT | Medium | 2h | T1.6 |

**Deliverables:**
- `cortical/reasoning/loom.py` - The Loom implementation
- `tests/unit/test_loom.py` - Unit tests
- Working surprise detection and mode switching

**Definition of Done:**
- [ ] All tests pass
- [ ] Coverage ≥ 95% for new code
- [ ] Mode switching observable in logs
- [ ] Demo script shows mode transitions

---

### Sprint 2: Hebbian Hive Enhancement (Weeks 3-4)

**Epic:** EPIC-HIVE

**Theme:** *"Teaching the hive to whisper, not shout."*

**Tasks:**

| ID | Task | Priority | Estimate | Dependencies |
|----|------|----------|----------|--------------|
| T2.1 | Add lateral_inhibition() to PRISM-SLM | High | 4h | None |
| T2.2 | Implement k_winners_take_all() | High | 3h | T2.1 |
| T2.3 | Add HomeostasisRegulator class | Medium | 4h | None |
| T2.4 | Implement excitability adjustment | Medium | 2h | T2.3 |
| T2.5 | Add spreading_activation() method | High | 4h | T2.1 |
| T2.6 | Create HiveNode wrapper with traces | Medium | 3h | T2.5 |
| T2.7 | Write tests for enhanced Hive | High | 4h | All above |
| T2.8 | Benchmark sparsity levels | Low | 2h | T2.7 |

**Deliverables:**
- Enhanced `prism_slm.py` with new methods
- `cortical/reasoning/homeostasis.py` - Regulation logic
- Sparse activation patterns (target: 5-10% active)

**Definition of Done:**
- [ ] Lateral inhibition produces sparse patterns
- [ ] Homeostasis maintains stable mean activation
- [ ] Spreading activation retrieves associated concepts
- [ ] All tests pass with ≥ 95% coverage

---

### Sprint 3: Cortex Abstraction (Weeks 5-6)

**Epic:** EPIC-CORTEX

**Theme:** *"From many patterns, one understanding."*

**Tasks:**

| ID | Task | Priority | Estimate | Dependencies |
|----|------|----------|----------|--------------|
| T3.1 | Design Abstraction class | High | 2h | None |
| T3.2 | Implement PatternDetector | High | 5h | T3.1 |
| T3.3 | Add abstraction_candidates() | High | 3h | T3.2 |
| T3.4 | Implement form_abstraction() | High | 4h | T3.3 |
| T3.5 | Wire abstractions to PRISM-PLN | Medium | 4h | T3.4 |
| T3.6 | Add GoalStack class | Medium | 3h | None |
| T3.7 | Implement goal progress tracking | Medium | 3h | T3.6 |
| T3.8 | Write tests for abstraction | High | 4h | T3.4 |
| T3.9 | Write tests for goal tracking | Medium | 2h | T3.7 |

**Deliverables:**
- `cortical/reasoning/abstraction.py` - Abstraction system
- `cortical/reasoning/goal_stack.py` - Goal management
- Hierarchical concepts emerging from patterns

**Definition of Done:**
- [ ] Abstractions form from repeated patterns
- [ ] Goals track progress toward completion
- [ ] PLN assigns truth values to abstractions
- [ ] All tests pass with ≥ 95% coverage

---

### Sprint 4: The Loom Weaves (Weeks 7-8)

**Epic:** EPIC-LOOM + Integration

**Theme:** *"Now the left hand knows what the right hand is doing."*

**Tasks:**

| ID | Task | Priority | Estimate | Dependencies |
|----|------|----------|----------|--------------|
| T4.1 | Connect Loom to enhanced Hive | High | 4h | Sprint 2 |
| T4.2 | Connect Loom to enhanced Cortex | High | 4h | Sprint 3 |
| T4.3 | Implement surprise_from_predictions() | High | 3h | T4.2 |
| T4.4 | Add attention_routing() based on mode | Medium | 3h | T4.1, T4.2 |
| T4.5 | Create WovenMind facade class | High | 4h | T4.4 |
| T4.6 | Implement process() unified method | High | 3h | T4.5 |
| T4.7 | Write integration tests | High | 4h | T4.6 |
| T4.8 | Create demo showing dual-process | Medium | 3h | T4.7 |

**Deliverables:**
- `cortical/reasoning/woven_mind.py` - Unified facade
- Integration tests showing Hive ↔ Cortex handoff
- Demo with observable mode switching

**Definition of Done:**
- [ ] WovenMind.process() routes to appropriate system
- [ ] High surprise engages Cortex
- [ ] Low surprise stays in Hive
- [ ] Transitions are logged and observable

---

### Sprint 5: Consolidation (Weeks 9-10)

**Epic:** EPIC-CONSOLIDATION

**Theme:** *"Sleep on it."*

**Tasks:**

| ID | Task | Priority | Estimate | Dependencies |
|----|------|----------|----------|--------------|
| T5.1 | Design ConsolidationEngine interface | High | 2h | None |
| T5.2 | Implement pattern_transfer() | High | 5h | T5.1 |
| T5.3 | Implement abstraction_mining() | High | 5h | T5.1, Sprint 3 |
| T5.4 | Add decay_cycle() for forgetting | Medium | 3h | T5.1 |
| T5.5 | Create consolidation scheduler | Medium | 3h | T5.2, T5.3 |
| T5.6 | Wire to WovenMind.consolidate() | High | 2h | T5.5, Sprint 4 |
| T5.7 | Write tests for consolidation | High | 4h | T5.6 |
| T5.8 | Benchmark learning retention | Low | 3h | T5.7 |

**Deliverables:**
- `cortical/reasoning/consolidation.py` - Consolidation engine
- Automated "sleep" cycles
- Measurable learning transfer

**Definition of Done:**
- [ ] Patterns from Hive transfer to Cortex abstractions
- [ ] Unused connections decay during consolidation
- [ ] Frequent patterns form new abstractions
- [ ] Learning persists across consolidation cycles

---

### Sprint 6: Integration & Polish (Weeks 11-12)

**Epic:** EPIC-INTEGRATION

**Theme:** *"The marriage is complete."*

**Tasks:**

| ID | Task | Priority | Estimate | Dependencies |
|----|------|----------|----------|--------------|
| T6.1 | End-to-end integration test suite | High | 6h | All sprints |
| T6.2 | Performance benchmarking | Medium | 4h | T6.1 |
| T6.3 | Write user documentation | High | 4h | T6.1 |
| T6.4 | Create tutorial notebook | Medium | 4h | T6.3 |
| T6.5 | Update CLAUDE.md with new APIs | Medium | 2h | T6.3 |
| T6.6 | Create comprehensive demo | High | 4h | T6.1 |
| T6.7 | Verify test coverage ≥ 95% | High | 2h | T6.1 |
| T6.8 | Code review and cleanup | Medium | 4h | T6.7 |
| T6.9 | Release announcement draft | Low | 2h | T6.8 |

**Deliverables:**
- Complete test suite
- Performance benchmarks
- User documentation
- Tutorial notebook
- Polished demo

**Definition of Done:**
- [ ] All integration tests pass
- [ ] Coverage ≥ 95% across all new code
- [ ] Documentation complete
- [ ] Demo shows full dual-process cycle
- [ ] No critical bugs

---

## Part IV: Risk Management

### Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Abstraction algorithm complexity | Medium | High | Start simple, iterate |
| Performance degradation | Low | Medium | Benchmark early and often |
| Integration complexity | Medium | Medium | Clear interfaces, incremental integration |
| Scope creep | High | High | Strict sprint boundaries, defer new ideas |
| Test coverage debt | Medium | High | TDD strictly enforced |

### Contingency Plans

**If Sprint 3 (Abstraction) runs long:**
- Simplify to frequency-based abstraction only
- Defer PLN integration to Sprint 6

**If Sprint 5 (Consolidation) runs long:**
- Implement basic decay only
- Defer pattern transfer to future sprint

**If integration proves difficult:**
- Focus on Loom → Hive path first
- Add Cortex integration incrementally

---

## Part V: Success Metrics

### Functional Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Mode switching accuracy | ≥ 90% | Manual review of 100 test cases |
| Abstraction quality | ≥ 80% relevant | Expert review of formed abstractions |
| Sparsity level | 5-10% active | Mean activation across tests |
| Prediction improvement | ≥ 10% | Compare before/after accuracy |

### Technical Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Test coverage | ≥ 95% | pytest-cov |
| Performance overhead | ≤ 20% | Benchmark suite |
| Code quality | No critical issues | pylint, mypy |
| Documentation | 100% public APIs | docstring coverage |

### Process Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Sprint completion | ≥ 80% of tasks | GoT tracking |
| Defect rate | ≤ 2 per sprint | Bug count |
| Technical debt | Declining | Tracked issues |

---

## Part VI: Getting Started

### Immediate Next Steps

1. **Create the Epic in GoT:**
   ```bash
   python scripts/got_utils.py epic create "EPIC-WOVEN: Woven Mind + PRISM Marriage" \
     --description "Unify PRISM and Woven Mind into dual-process architecture"
   ```

2. **Create Sprint 1:**
   ```bash
   python scripts/got_utils.py sprint create "The Loom Foundation" --number 18 \
     --epic EPIC-WOVEN
   ```

3. **Create initial tasks:**
   ```bash
   python scripts/got_utils.py task create "Design Loom interface contract" \
     --priority high --sprint S-sprint-018-loom-foundation
   ```

4. **Start coding:**
   - Create `cortical/reasoning/loom.py`
   - Write tests first (TDD)
   - Implement SurpriseDetector

### The First Line of Code

```python
# cortical/reasoning/loom.py
"""
The Loom: Where Hebbian Hive and Cultured Cortex weave together.

"Begin at the beginning," the King said gravely, "and go on till you
come to the end: then stop."

The Loom decides when to think fast (Hive) and when to think slow (Cortex).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Set, Optional, Any

from .prism_attention import UnifiedAttention
from .prism_got import SynapticMemoryGraph
from .prism_slm import PRISMLanguageModel


class ThinkingMode(Enum):
    """The two modes of thought."""
    FAST = auto()   # System 1: Hebbian Hive
    SLOW = auto()   # System 2: Cultured Cortex


@dataclass
class SurpriseSignal:
    """A signal indicating something unexpected happened."""
    magnitude: float          # How surprising (0.0 to 1.0+)
    source: str               # What caused the surprise
    context: Dict[str, Any] = field(default_factory=dict)


class SurpriseDetector:
    """
    Detects when the world doesn't match expectations.

    High surprise → engage System 2
    Low surprise → stay in System 1
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.baseline: Dict[str, float] = {}

    def compute_surprise(
        self,
        predicted: Dict[str, float],
        actual: Set[str]
    ) -> SurpriseSignal:
        """
        Compute surprise from prediction error.

        Args:
            predicted: What we expected (node_id → probability)
            actual: What actually happened (active node IDs)

        Returns:
            SurpriseSignal with magnitude and context
        """
        # ... implementation follows
```

---

## Epilogue: The Road Ahead

We stand at a crossroads where two paths—PRISM's practical implementation and Woven Mind's theoretical clarity—can become one.

The marriage will not be quick or easy. Six sprints. Dozens of tasks. Thousands of lines of code yet to be written.

But the destination is worth the journey: a cognitive architecture that knows when to think fast and when to think slow. That builds abstractions from experience. That dreams during consolidation and wakes with new understanding.

The loom is ready. The threads are prepared. Let us begin to weave.

---

> *"Would you tell me, please, which way I ought to go from here?"*
> *"That depends a good deal on where you want to get to," said the Cat.*
>
> We know where we want to get to.
> The roadmap shows the way.
> All that remains is to walk it.

---

## Appendix: Task Dependency Graph

```
Sprint 1 (Loom Foundation)
├── T1.1 Design Loom interface
│   ├── T1.2 SurpriseDetector
│   └── T1.3 ModeController
│       └── T1.4 Wire to PRISM-Attention
│           ├── T1.5 Observable transitions
│           └── T1.6 Unit tests
│               └── T1.7 Integration test

Sprint 2 (Hebbian Hive)          Sprint 3 (Cortex Abstraction)
├── T2.1 Lateral inhibition      ├── T3.1 Design Abstraction
│   └── T2.2 k-winners           │   └── T3.2 PatternDetector
│       └── T2.5 Spreading act   │       └── T3.3 abstraction_candidates
├── T2.3 Homeostasis             │           └── T3.4 form_abstraction
│   └── T2.4 Excitability        │               └── T3.5 Wire to PLN
└── T2.6 HiveNode wrapper        ├── T3.6 GoalStack
    └── T2.7 Tests               │   └── T3.7 Goal tracking
        └── T2.8 Benchmark       └── T3.8, T3.9 Tests
                │                         │
                └─────────┬───────────────┘
                          ▼
              Sprint 4 (Loom Weaves)
              ├── T4.1 Connect to Hive
              ├── T4.2 Connect to Cortex
              │   └── T4.3 surprise_from_predictions
              └── T4.4 attention_routing
                  └── T4.5 WovenMind facade
                      └── T4.6 process() method
                          └── T4.7 Integration tests
                              └── T4.8 Demo
                                      │
                                      ▼
                          Sprint 5 (Consolidation)
                          ├── T5.1 Design interface
                          │   ├── T5.2 pattern_transfer
                          │   ├── T5.3 abstraction_mining
                          │   └── T5.4 decay_cycle
                          └── T5.5 Scheduler
                              └── T5.6 Wire to WovenMind
                                  └── T5.7 Tests
                                      └── T5.8 Benchmark
                                              │
                                              ▼
                                  Sprint 6 (Integration)
                                  └── T6.1-T6.9 Polish & Release
```

---

*Document version: 1.0*
*Created: 2025-12-25*
*Epic: EPIC-WOVEN*
