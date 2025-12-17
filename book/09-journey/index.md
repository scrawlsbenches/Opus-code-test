---
title: "Your Learning Journey"
generated: "2025-12-17T00:01:54.221173Z"
generator: "journey"
source_files:
  - "git log"
  - "CLAUDE.md"
tags:
  - journey
  - learning-path
  - index
---

# Your Learning Journey

*A progressive path through the Cortical Text Processor, designed for learners at all levels.*

---

## Overview

This learning journey is organized into three progressive stages, each building on the previous one. The concepts are ordered based on:

- **Dependencies:** What you need to know first
- **Complexity:** From simple to sophisticated
- **Historical emergence:** When concepts first appeared in development

## Learning Paths

### [Beginner Path](journey-beginner.md)

**Concepts:** 3  
**Time:** ~45 minutes  
**Preview:** Tokenization, Stop Words, Tf-Idf

### [Intermediate Path](journey-intermediate.md)

**Concepts:** 6  
**Time:** ~150 minutes  
**Preview:** Pagerank, Lateral Connections, Query Expansion, and 3 more

### [Advanced Path](journey-advanced.md)

**Concepts:** 3  
**Time:** ~105 minutes  
**Preview:** Concept Clustering, Semantic Relations, Louvain

---

# Suggested Study Schedule

*A practical 4-week plan to master the Cortical Text Processor.*

---

## Week 1: Foundations

**Goal:** Understand the core building blocks

**Concepts:**
- Tokenization (~15 min)
- Stop Words (~15 min)
- Tf-Idf (~15 min)

**Total Time:** ~45 minutes

**Activities:**
- Read foundation chapters
- Run `showcase.py` to see concepts in action
- Experiment with basic tokenization and TF-IDF

## Week 2: Building Complexity

**Goal:** Add graph-based features to your mental model

**Concepts:**
- Pagerank (~25 min)
- Lateral Connections (~25 min)
- Query Expansion (~25 min)

**Total Time:** ~75 minutes

**Activities:**
- Study PageRank and BM25 implementations
- Index a sample corpus with `scripts/index_codebase.py`
- Explore query expansion behavior

## Week 3: Advanced Structures

**Goal:** Understand hierarchical organization and persistence

**Concepts:**
- Incremental Indexing (~25 min)
- Bm25 (~25 min)
- Persistence (~25 min)

**Total Time:** ~75 minutes

**Activities:**
- Review minicolumn and layer architecture
- Practice save/load operations
- Test incremental indexing

## Week 4: Mastery

**Goal:** Master sophisticated algorithms and optimization

**Concepts:**
- Concept Clustering (~35 min)
- Semantic Relations (~35 min)
- Louvain (~35 min)

**Total Time:** ~105 minutes

**Activities:**
- Study Louvain clustering implementation
- Experiment with semantic relations extraction
- Profile performance with `scripts/profile_full_analysis.py`
- Implement a custom feature using the library

## Tips for Success

1. **Follow the order** - Prerequisites build on each other
2. **Code along** - Run examples from `showcase.py` and scripts
3. **Read tests** - `tests/` directory shows real usage patterns
4. **Ask questions** - Use the semantic search: `python scripts/search_codebase.py "your question"`
5. **Build something** - Best way to learn is to apply the concepts

