# MANIFEST.md — Repository Guide

> *The map is not the territory, but without a map, you're lost.*

---

## What Is This?

**Cortical Text Processor** is a zero-dependency Python library for semantic text analysis, knowledge graph construction, and intelligent document retrieval. It combines graph algorithms (PageRank, TF-IDF, clustering) with cognitive architectures for reasoning, task tracking, and dual-process cognition.

**In one sentence:** Build a searchable knowledge graph from your documents, then query it with natural language.

---

## Quick Start — By Intent

| I want to... | Start here |
|--------------|------------|
| **Use the library** | `python showcase.py` then read `cortical/processor/` |
| **Search documents** | `cortical.search(query)` — see `cortical/query/search.py` |
| **Understand the architecture** | Read [Architecture](#architecture-at-a-glance) below |
| **Run all tests** | `python -m pytest tests/ -v` |
| **Run smoke tests only** | `python -m pytest tests/smoke/ -v` |
| **Track tasks/decisions** | `python scripts/got_utils.py --help` |
| **Train on this codebase** | `benchmarks/codebase_slm/train_slm.py` |
| **Contribute code** | Read `CLAUDE.md` (Metus philosophy) |
| **Find a specific feature** | See [Where To Find Things](#where-to-find-things) |

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER / APPLICATION                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CorticalTextProcessor                                 │
│                         (cortical/processor/)                                │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐    │
│  │  core    │ │ documents │ │ compute  │ │ query_api │ │ persistence  │    │
│  │  .py     │ │   .py     │ │   .py    │ │    .py    │ │   _api.py    │    │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └─────┬─────┘ └──────┬───────┘    │
└───────┼─────────────┼────────────┼─────────────┼──────────────┼────────────┘
        │             │            │             │              │
        ▼             ▼            ▼             ▼              ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           CORE MODULES                                     │
├───────────────┬───────────────┬───────────────┬───────────────────────────┤
│   analysis/   │    query/     │   reasoning/  │         spark/            │
│               │               │               │                           │
│  • PageRank   │  • search     │  • WovenMind  │  • NGramModel             │
│  • TF-IDF     │  • expansion  │  • Loom       │  • CodeIntelligence       │
│  • clustering │  • passages   │  • GoT nodes  │  • AnomalyDetector        │
│  • activation │  • ranking    │  • QAPV loop  │  • GitHistoryTrainer      │
└───────────────┴───────────────┴───────────────┴───────────────────────────┘
        │                               │
        ▼                               ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         PERSISTENCE LAYER                                  │
├───────────────────────────────┬───────────────────────────────────────────┤
│            got/               │              cel/                          │
│   (Graph of Thought)          │    (Cognitive Event Lattice)              │
│                               │                                           │
│  • Task/Decision tracking     │  • Event sourcing substrate               │
│  • ACID transactions          │  • Wisdom strand (knowledge)              │
│  • Sprint management          │  • Sanity strand (health)                 │
│  • Write-Ahead Log            │  • Merkle DAG persistence                 │
└───────────────────────────────┴───────────────────────────────────────────┘
```

### Data Flow

```
Documents → Tokenizer → Minicolumns → Graph → Analysis → Query Results
    │                       │            │
    │                       ▼            ▼
    │                   Layers      PageRank + TF-IDF
    │                       │            │
    └───────────────────────┴────────────┴──→ Searchable Knowledge Graph
```

---

## Component Guide

### Core Library (`cortical/`)

| Component | Purpose | Status | Entry Point |
|-----------|---------|--------|-------------|
| **processor/** | Main API — document management, search, analysis | Stable | `__init__.py` |
| **analysis/** | Graph algorithms (PageRank, TF-IDF, clustering) | Stable | `pagerank.py`, `tfidf.py` |
| **query/** | Search, expansion, passage retrieval | Stable | `search.py` |
| **tokenizer.py** | Text tokenization, stemming, stop words | Stable | Direct import |
| **layers.py** | Hierarchical document structure | Stable | Direct import |
| **minicolumn.py** | Core data structures (nodes, edges) | Stable | Direct import |

### Cognitive Systems

| Component | Purpose | Status | When to Use |
|-----------|---------|--------|-------------|
| **reasoning/** | Dual-process cognition, QAPV loops | Beta | Complex multi-step reasoning |
| **reasoning/woven_mind.py** | System 1/2 routing facade | Beta | Fast vs slow thinking |
| **reasoning/loom.py** | Mode switching based on surprise | Beta | Automatic fast/slow selection |
| **spark/** | Statistical language model | Beta | Code completion, predictions |

### Persistence & Tracking

| Component | Purpose | Status | When to Use |
|-----------|---------|--------|-------------|
| **got/** | Task/decision tracking with transactions | Stable | Sprint planning, decisions |
| **cel/** | Event sourcing, temporal queries | Alpha | Audit trails, event replay |
| **got/wal.py** | Write-Ahead Log for durability | Stable | Automatic (internal) |

### Supporting Systems

| Component | Purpose | Status | Entry Point |
|-----------|---------|--------|-------------|
| **llm_orchestration/** | Multi-agent coordination | Beta | `orchestration.py` |
| **ml_experiments/** | ML training framework | Beta | `experiment.py` |
| **utils/** | Shared utilities (IDs, checksums, locking) | Stable | Various |

---

## Glossary

> *Plain English for our metaphorical terms*

| Term | Plain English | Location |
|------|---------------|----------|
| **Cortical** | Brain-inspired text processing (like cortical columns) | Project name |
| **Minicolumn** | A node in the knowledge graph representing a concept | `cortical/minicolumn.py` |
| **Woven Mind** | Dual-process system combining fast + slow thinking | `reasoning/woven_mind.py` |
| **The Loom** | Decision point that routes to fast or slow processing | `reasoning/loom.py` |
| **Hive** | Fast pattern matching cache (System 1 / intuition) | `reasoning/loom_hive.py` |
| **Cortex** | Slow deliberate reasoning (System 2 / analysis) | `reasoning/loom_cortex.py` |
| **CEL** | Cognitive Event Lattice — event sourcing for cognition | `cortical/cel/` |
| **Wisdom Strand** | Knowledge storage in CEL (what the system knows) | `cel/wisdom/` |
| **Sanity Strand** | Health/validation in CEL (keeping it coherent) | `cel/sanity/` |
| **GoT** | Graph of Thought — task and decision tracking | `cortical/got/` |
| **QAPV** | Question → Answer → Produce → Verify cycle | `reasoning/cognitive_loop.py` |
| **Spark** | Statistical language model for fast predictions | `cortical/spark/` |
| **SparkSLM** | Spark Statistical Language Model | `spark/predictor.py` |
| **PRISM** | Probabilistic Reasoning In Semantic Models | `reasoning/prism_*.py` |
| **Metus** | Our BDD philosophy: Mindful Execution Through Unwavering Specification | `CLAUDE.md` |

---

## Where To Find Things

### By Task

| I need to... | Look in... |
|--------------|------------|
| Add a document to the corpus | `processor/documents.py` → `add_document()` |
| Search for documents | `processor/query_api.py` → `search()` |
| Get similar documents | `processor/query_api.py` → `find_similar()` |
| Expand a query with synonyms | `query/expansion.py` |
| Compute PageRank scores | `analysis/pagerank.py` |
| Compute TF-IDF scores | `analysis/tfidf.py` |
| Cluster documents | `analysis/clustering.py` |
| Create a task | `got/api.py` → `create_task()` |
| Track a decision | `got/api.py` → `create_decision()` |
| Save/load processor state | `processor/persistence_api.py` |
| Detect anomalies in input | `spark/anomaly.py` |
| Get code completions | `spark/intelligence.py` |

### By Concept

| Concept | Primary File | Tests |
|---------|--------------|-------|
| BM25 scoring | `analysis/tfidf.py` | `tests/unit/test_tfidf.py` |
| Query expansion | `query/expansion.py` | `tests/unit/test_query_expansion.py` |
| Passage retrieval | `query/passages.py` | `tests/unit/test_query_passages.py` |
| Community detection | `analysis/clustering.py` | `tests/unit/test_clustering.py` |
| Transaction support | `got/transaction.py` | `tests/unit/got/test_transaction.py` |
| Event sourcing | `cel/core/events.py` | `tests/behavioral/test_cel_*.py` |
| Dual-process routing | `reasoning/loom.py` | `tests/unit/test_loom.py` |

### By File Type

| Looking for... | Location |
|----------------|----------|
| Main library code | `cortical/` |
| CLI tools | `scripts/` |
| Interactive demos | `examples/` |
| Benchmarks | `benchmarks/` |
| Unit tests | `tests/unit/` |
| Behavioral scenarios | `tests/behavioral/` |
| Performance contracts | `tests/performance/` |
| Sample documents | `samples/` |

---

## Test Structure (Metus-Aligned)

We follow **Metus** — Behavior-Driven Development with performance contracts.

See `CLAUDE.md` for the complete philosophy.

### The Confidence Ladder

```
tests/
├── smoke/              ← Gate 1: Does it breathe? (<1 second)
├── unit/               ← Gate 2: Atomic correctness (95%+ coverage)
│   └── specifications/ ← Facts that must remain true
├── behavioral/         ← Gate 3: User stories work end-to-end
├── performance/
│   └── contracts/      ← Gate 4: Performance guarantees defended
├── integration/        ← Gate 5: Components work together
└── security/           ← Gate 6: No vulnerabilities
```

### Running Tests

```bash
# Quick check (run constantly)
python -m pytest tests/smoke/ -v

# Full validation (before merge)
python -m pytest tests/ -v --cov=cortical --cov-fail-under=95

# Specific gates
python -m pytest tests/behavioral/ -v           # User stories
python -m pytest tests/performance/ -m contract  # Performance contracts
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Python files | ~693 |
| Main library (`cortical/`) | ~120 files |
| Test files | ~240 files |
| Test coverage target | 95%+ |
| External dependencies | Zero (core library) |

---

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Development philosophy (Metus BDD) |
| `MANIFEST.md` | This file — repository navigation |
| `showcase.py` | Interactive demo of main features |
| `pyproject.toml` | Project configuration |
| `cortical/__init__.py` | Public API exports |

---

## Getting Help

| Question | Resource |
|----------|----------|
| "How do I use X?" | Check `examples/` for demos |
| "What does X mean?" | See [Glossary](#glossary) above |
| "Where is X?" | See [Where To Find Things](#where-to-find-things) |
| "How do I test?" | See [Test Structure](#test-structure-metus-aligned) |
| "What's the philosophy?" | Read `CLAUDE.md` (Metus) |

---

## Contributing

1. Read `CLAUDE.md` — understand Metus philosophy
2. Write the **user story** first
3. Write **behavioral scenarios** that prove it works
4. Implement to make scenarios pass
5. Ensure **performance contracts** are honored
6. All CI gates must be green

---

*"The map helps you navigate. The territory is yours to explore."*
