# Knowledge Transfer: Code Review, PRISM, and CEL Architecture

**Session Date:** 2025-12-29
**Session ID:** xqzFK
**Branch:** claude/code-review-guidance-xqzFK

---

## Executive Summary

This session performed a comprehensive code review of the Cortical Text Processor codebase and explored two key cognitive architecture components: PRISM (reasoning engine) and CEL (knowledge substrate). Eight critical/high-priority tasks were created from review findings.

---

## 1. Code Review Findings

### Overall Grade: A- (88/100)

**Exceptional Strengths:**
- Zero runtime dependencies (pure stdlib Python)
- 94% type hint coverage
- 289 test files with 15,245 assertions
- Clean mixin-based architecture (no god classes)
- 2,800+ lines of documentation in CLAUDE.md

### Critical Issues Identified

| Issue | Location | Impact |
|-------|----------|--------|
| Transaction-unsafe deletes | `got/api.py:661-862` | Crash during delete → orphaned edges |
| Deadlock potential | `versioned_store.py:51-84` | Two locks, no acquisition order |
| 260 bare except clauses | 64 files | Silent failures mask bugs |
| 35 silent `pass` handlers | `graph_persistence.py` (35!) | Errors invisible |

### Tasks Created

| Task ID | Priority | Description |
|---------|----------|-------------|
| T-20251229-123957-0affebee | Critical | Fix transaction-unsafe delete operations |
| T-20251229-124004-042ccc7d | Critical | Fix deadlock in lock acquisition |
| T-20251229-124011-545a545d | High | Audit 260 bare except clauses |
| T-20251229-124017-1bf32ba0 | High | Add logging to 35 silent handlers |
| T-20251229-124024-7c901875 | Medium | Add docstrings to query/analysis |
| T-20251229-124031-d137dcb0 | Medium | Split got/api.py (2918 lines) |
| T-20251229-124037-1ad50d52 | Medium | Address 41 TODO comments |
| T-20251229-124044-e9e99166 | Medium | Fix test anti-patterns |

### Artifacts Created

- `CODE_REVIEW_2025-12-29.md` - Full review with remediation guidance

---

## 2. PRISM: Predictive Reasoning through Incremental Synaptic Memory

### What It Is

PRISM is a **biologically-inspired reasoning engine** (3,410 lines) that treats knowledge graphs like neural networks where connections learn and forget.

### The Four Subsystems

```
┌──────────────┬──────────────┬─────────────┬─────────────────┐
│  PRISM-GoT   │  PRISM-SLM   │  PRISM-PLN  │ PRISM-Attention │
│  1,147 lines │   929 lines  │  719 lines  │    615 lines    │
├──────────────┼──────────────┼─────────────┼─────────────────┤
│  Synaptic    │  Language    │ Probabilistic│  Query-based   │
│  Memory      │  Model with  │   Logic     │   Focus        │
│  Graph       │  Learning    │  Networks   │  Mechanisms    │
└──────────────┴──────────────┴─────────────┴─────────────────┘
```

### Core Concepts

**SynapticEdge** - Edges that remember and learn:
```python
@dataclass
class SynapticEdge:
    weight: float = 1.0           # Synaptic strength
    decay_factor: float = 0.99    # Unused connections fade
    prediction_accuracy: float    # Track prediction success
    activation_count: int = 0     # How often used
```

**Key Principles:**
1. **Hebbian Learning**: "Neurons that fire together wire together"
2. **Decay**: Unused connections weaken (0.99 per cycle)
3. **Prediction Tracking**: Successful paths strengthen
4. **Uncertainty First-Class**: TruthValue = strength × confidence

### File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `cortical/reasoning/prism_got.py` | 1,147 | Synaptic memory graph |
| `cortical/reasoning/prism_slm.py` | 929 | Language model |
| `cortical/reasoning/prism_pln.py` | 719 | Probabilistic logic |
| `cortical/reasoning/prism_attention.py` | 615 | Attention mechanisms |

---

## 3. CEL: Cognitive Event Lattice

### What It Is

CEL is the **knowledge substrate** that PRISM builds upon (~4,300 lines). It's the "memory layer" while PRISM is the "thinking layer."

### Core Philosophy

**"Events are primary. Entities are derived."**

Instead of storing current state, CEL stores immutable events. Entity state is computed by replaying events - like Redux, Git, or double-entry bookkeeping.

### The Double Helix Architecture

```
WISDOM (knowing)          SANITY (healing)
├─ MerkleDAG             ├─ Health Monitor
├─ Materializer          ├─ Migration Engine
└─ SemanticIndex         └─ Compaction
```

**Wisdom without sanity → corruption. Sanity without wisdom → empty process.**

### Temporal References (Key Innovation)

```python
@dataclass
class TemporalReference:
    entity_id: str        # What entity
    horizon: EventHorizon # As of which event
```

- "The system" is ambiguous (changes constantly)
- "The system at event E" is concrete and stable
- **Solves self-reference paradox without contradiction**

### File Locations

```
cortical/cel/                     # ~4,300 lines
├── core/                         # Abstractions (~700 lines)
│   ├── events.py                 # Event types
│   └── references.py             # Temporal references
├── wisdom/                       # Knowledge (~1,400 lines)
│   ├── dag.py                    # Merkle DAG
│   └── materializer.py           # Events → Entities
├── sanity/                       # Health (~1,740 lines)
│   ├── health.py                 # Self-monitoring
│   └── migration.py              # Schema evolution
└── adapters/got.py               # GoT bridge
```

---

## 4. The Architecture Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Woven Mind    │ Dual-process orchestration (System 1/2)   │
├─────────────────────────────────────────────────────────────┤
│  PRISM         │ Reasoning engine (synaptic, probabilistic)│
├─────────────────────────────────────────────────────────────┤
│  CEL           │ Knowledge substrate (events, temporal)    │
├─────────────────────────────────────────────────────────────┤
│  GoT           │ Task/decision tracking (being replaced)   │
└─────────────────────────────────────────────────────────────┘
```

### How They Connect

```
CEL provides:                    PRISM consumes:
├─ Event history            →    Synaptic activation traces
├─ Temporal references      →    Stable context for reasoning
├─ Materialized entities    →    Knowledge nodes in graph
├─ Semantic index           →    Fast concept lookup
└─ Causal DAG               →    Dependency-aware prediction
```

### GoT → CEL Migration

```
Phase 1: BRIDGE (Current) - GoT + CEL coexist
Phase 2: PARALLEL - CEL primary, GoT backup  
Phase 3: CEL ONLY - Retire GoT
```

---

## 5. Woven Mind + PRISM Marriage

The unified cognitive architecture being built:

```
┌─────────────────────────────────────────────────────────────┐
│  CULTURED CORTEX (System 2 - Slow Deliberate)               │
│  └─ PRISM-GoT + PRISM-PLN                                  │
├─────────────────────────────────────────────────────────────┤
│  THE LOOM (Mode Switching)                                  │
│  └─ PRISM-Attention + Surprise Detector                    │
├─────────────────────────────────────────────────────────────┤
│  HEBBIAN HIVE (System 1 - Fast Automatic)                   │
│  └─ PRISM-SLM + lateral inhibition                         │
├─────────────────────────────────────────────────────────────┤
│  CEL SUBSTRATE                                              │
│  └─ Events, temporal refs, health, migration               │
└─────────────────────────────────────────────────────────────┘
```

**The Loom decides:** Low surprise → stay fast. High surprise → engage slow.

---

## 6. Key Insights for Future Sessions

### Code Quality

1. **Transaction safety is incomplete** - Delete operations bypass WAL
2. **Lock ordering undefined** - Deadlock risk in concurrent commits
3. **Silent failures pervasive** - 260+ bare except clauses
4. **Test fixtures underused** - 479 direct instantiations vs fixtures

### Architecture Understanding

1. **CEL is the foundation** - Events primary, entities derived
2. **PRISM reasons over CEL** - Synaptic learning + probabilistic logic
3. **Woven Mind orchestrates** - System 1/2 switching via surprise
4. **GoT is being replaced** - Bridge pattern for gradual migration

### Critical Files to Know

| System | Key File | Purpose |
|--------|----------|---------|
| GoT | `got/api.py` (2,918 lines) | Task/decision CRUD |
| CEL | `cel/wisdom/dag.py` | Event storage |
| PRISM | `reasoning/prism_got.py` | Synaptic memory |
| Woven | `reasoning/woven_mind.py` | Dual-process facade |

---

## 7. Recommended Next Steps

### Immediate (Phase 1 - Safety)
1. Fix transaction-unsafe deletes (T-20251229-123957)
2. Define lock acquisition order (T-20251229-124004)
3. Audit bare except clauses (T-20251229-124011)

### Short-term (Phase 2 - Quality)
1. Add docstrings to public functions
2. Split large files (got/api.py)
3. Migrate tests to fixture pattern

### Medium-term (Architecture)
1. Complete CEL ↔ GoT bridge testing
2. Continue Woven Mind + PRISM marriage (6 sprints)
3. Implement The Loom for mode switching

---

## 8. Session Artifacts

| Artifact | Location |
|----------|----------|
| Code Review | `CODE_REVIEW_2025-12-29.md` |
| This Knowledge Transfer | `samples/memories/2025-12-29-session-knowledge-transfer-code-review-prism-cel.md` |
| Tasks Created | 8 tasks (see Task IDs above) |
| Branch | `claude/code-review-guidance-xqzFK` |

---

## 9. Quick Reference Commands

```bash
# View created tasks
python scripts/got_utils.py task list --status pending

# Show specific task
python scripts/got_utils.py task show T-20251229-123957-0affebee

# Start working on critical task
python scripts/got_utils.py task start T-20251229-123957-0affebee

# Read this knowledge transfer
cat samples/memories/2025-12-29-session-knowledge-transfer-code-review-prism-cel.md
```

---

**End of Knowledge Transfer**

*Generated: 2025-12-29 | Session: xqzFK | Agent: Claude Opus 4.5*
