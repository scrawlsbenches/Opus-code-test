# How To: Working with Cognitive Science and World Model Concepts

A practical guide learned from analyzing the cognitive corpus with the Cortical Text Processor.

## Overview

This guide documents effective patterns for working with cognitive science, world models, and cross-domain knowledge discovered through hierarchical text analysis.

**Key Insight:** The cognitive corpus (87 documents, 7 domains) forms a densely connected network with 2.2M+ connections, enabling rich query expansion and concept discovery.

---

## 1. Understanding the Domain Structure

### The Seven Cognitive Domains

| Domain | Docs | Focus | Key Bridge Terms |
|--------|------|-------|------------------|
| `cognitive_science/` | 18 | Memory, attention, decision-making, metacognition | learning, cognitive, memory |
| `world_models/` | 10 | Internal representations, prediction, simulation | models, prediction, simulation |
| `future_thinking/` | 10 | Forecasting, planning, uncertainty | forecasting, uncertainty, planning |
| `social_influence/` | 15 | Persuasion, influence, group dynamics | influence, social, behavior |
| `ai_market_prediction/` | 12 | ML for markets, regime detection, risk | market, prediction, risk |
| `workflow_practices/` | 10 | Development practices, collaboration | feedback, iteration, knowledge |
| `cross_domain/` | 12 | Explicit bridges between domains | (multi-domain) |

### Most Central Concepts (by PageRank)

The top concepts that act as "hubs" connecting the corpus:

1. **models** (0.0078) - Appears in all 7 domains
2. **learning** (0.0066) - Core to cognitive science and AI
3. **information** (0.0051) - Bridges prediction and decision-making
4. **social** (0.0051) - Links influence to cognition
5. **cognitive** (0.0049) - Foundation for all mental processes

**Practical Use:** When searching, queries containing these hub terms will naturally expand to reach more documents.

---

## 2. The Prediction-Decision Chain

A key conceptual structure discovered in the corpus:

```
perception → attention → prediction → uncertainty → decision → learning
    │            │            │             │            │          │
    └────────────┴────────────┴─────────────┴────────────┴──────────┘
                         (all strongly connected)
```

### Connection Strengths

| From | To | Weight | Interpretation |
|------|-----|--------|----------------|
| perception | attention | 10.0 | Tight coupling |
| attention | prediction | 10.0 | What we attend to, we predict |
| prediction | uncertainty | 10.0 | Predictions carry confidence |
| uncertainty | decision | 10.0 | Decisions require uncertainty estimates |
| decision | learning | 10.0 | Outcomes update future decisions |

**Practical Use:** When querying about any stage in this chain, expect related stages to appear in results. Use this for exploration.

---

## 3. Effective Query Patterns

### Query Expansion Works Well

The system automatically expands queries with semantically related terms:

| Original Query | Expanded With |
|---------------|---------------|
| "world models update prediction errors" | +model, +social, +learning, +beliefs |
| "decision-making under uncertainty" | +cognitive, +models, +quantification, +planning |
| "social proof group behavior" | +influence, +effects, +world |
| "market regime changes" | +forecasting, +ensemble, +learning, +model |

### Best Practices for Queries

1. **Include domain-specific terms** - "metacognition forecasting" not just "thinking ahead"
2. **Use the chain concepts** - prediction, uncertainty, decision work as anchors
3. **Leverage cross-domain docs** - Prefix with domain for precision: "cross_domain/metacognition"
4. **Query time: ~110-120ms** - Fast enough for interactive use

### Example: Multi-Domain Discovery

```python
# Query that spans cognitive science + markets + future thinking
query = "How does metacognition improve forecasting?"

# Results bridge domains:
# 1. cross_domain/metacognition_in_ai (37.5)
# 2. cognitive_science/metacognition_introspection (34.6)
# 3. future_thinking/forecasting_methods (20.2)
```

---

## 4. Cross-Domain Bridge Concepts

Terms that appear in 3+ domains are powerful for discovery:

### High-Value Bridge Terms

| Term | Domains | Use Case |
|------|---------|----------|
| models | 7 | Universal connector |
| learning | 7 | Connects training with cognition |
| prediction | 6 | Links forecasting to perception |
| uncertainty | 5 | Bridges decisions with risk |
| information | 6 | Connects processing with markets |
| cognitive | 5 | Anchors mental process discussions |

### Suggested New Bridges

The analysis identified pairs that share many neighbors but aren't directly connected:

| Pair | Shared Neighbors | Potential Document |
|------|------------------|-------------------|
| models ↔ memory | 309 | "Memory as Internal World Model" |
| learning ↔ influence | 279 | "How Influence Shapes Learning" |
| information ↔ planning | 272 | "Information Flow in Strategic Planning" |
| cognitive ↔ risk | 235 | "Cognitive Aspects of Risk Assessment" |

**Action:** Consider creating documents that explicitly bridge these concepts.

---

## 5. Concept Category Coverage

All eight key concept categories have 100% coverage:

| Category | Sample Terms | Coverage |
|----------|--------------|----------|
| prediction | prediction, forecast, anticipate, expect, future | 100% |
| learning | learning, adaptation, update, acquire, experience | 100% |
| models | model, representation, simulation, schema, mental | 100% |
| decision | decision, choice, select, judgment, evaluate | 100% |
| influence | influence, persuade, social, conform, nudge | 100% |
| uncertainty | uncertainty, probability, risk, confidence, error | 100% |
| attention | attention, focus, salience, awareness, cognitive | 100% |
| memory | memory, recall, encoding, consolidation, retrieval | 100% |

### Weak Topics (Need More Coverage)

Single-document topics that could benefit from expansion:

- `likability` - Only in social influence
- `propaganda` - Only in information operations
- `nudge` - Limited behavioral economics coverage
- `obedience` - Milgram experiments only

---

## 6. World Model Network

The `world_models/` domain forms a coherent sub-network:

### Core World Model Concepts

```
world_model_fundamentals
    ├── predictive_world_models (anticipating futures)
    ├── generative_world_models (imagination, simulation)
    ├── hierarchical_world_models (multi-scale representations)
    ├── learned_world_models (experience-based acquisition)
    ├── model_based_reasoning (planning, inference)
    ├── world_model_updating (belief revision)
    ├── embodied_world_models (sensorimotor grounding)
    ├── social_world_models (theory of mind)
    └── ai_world_models (ML implementations)
```

### Key Connections Within World Models

| Term | Top Connections |
|------|-----------------|
| model | prediction, world, learning, internal, based |
| prediction | model, errors, future, uncertainty, updating |
| simulation | internal, mental, model, forward, imagined |
| representation | internal, hierarchical, model, abstract, states |

---

## 7. Running the Analysis

### Quick Overview (~10s)

```bash
python scripts/world_model_analysis.py --quick
```

Shows: domain structure, core concepts, cross-domain bridges, world model network.

### Full Analysis (~30s)

```bash
python scripts/world_model_analysis.py
```

Adds: prediction-decision chain, cognitive queries, knowledge gaps, connection suggestions.

### Custom Samples Directory

```bash
python scripts/world_model_analysis.py --samples /path/to/corpus
```

---

## 8. Practical Workflows

### Workflow A: Explore a New Concept

1. Run quick analysis to see if concept exists in corpus
2. Use `expand_query()` to find related terms
3. Check which domains contain the concept
4. Follow cross-domain bridges for broader context

### Workflow B: Find Gaps for New Content

1. Run full analysis
2. Check "Weak topics needing more coverage"
3. Check "Suggested connections" for bridge opportunities
4. Create new documents in appropriate domain directories

### Workflow C: Cross-Domain Research

1. Start with `cross_domain/` documents
2. Use bridge concepts (models, learning, prediction) in queries
3. Follow results across domains
4. Note which domain combinations appear together

---

## 9. Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Documents | 87 | Comprehensive cognitive coverage |
| Unique concepts | 7,196 | Rich vocabulary |
| Concept clusters | 79 | Well-organized semantic groups |
| Total connections | 2,241,270 | Densely connected graph |
| Coverage score | 100% | All key categories present |
| Connectivity score | 0.0665 | Moderate inter-document links |
| Query time | ~115ms | Interactive performance |
| Most central concept | "models" | Hub for all domains |
| Most connected domain | cognitive_science | 1,548 connections |

---

## 10. Summary

The cognitive corpus is designed for:

1. **Rich query expansion** - Hub concepts connect to many terms
2. **Cross-domain discovery** - Bridge documents enable exploration
3. **Prediction-decision reasoning** - The chain provides conceptual structure
4. **Gap identification** - Analysis reveals where to add content
5. **World model understanding** - Dedicated sub-network for internal representations

Use the `world_model_analysis.py` script regularly to:
- Track corpus health as content is added
- Identify new bridging opportunities
- Validate concept coverage
- Discover unexpected connections

---

*Generated from world_model_analysis.py results on 2025-12-19*
