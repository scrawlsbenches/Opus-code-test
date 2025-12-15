# Mixture of Expert Indexes: Knowledge Transfer Document

**Author:** Claude (AI Assistant)
**Date:** 2025-12-15
**Status:** Design Proposal
**Related:** [architecture.md](architecture.md), [algorithms.md](algorithms.md)

---

## Executive Summary

This document provides comprehensive background knowledge for implementing a **Mixture of Experts (MoE) architecture** for the Cortical Text Processor's indexing system. The core idea: instead of one general-purpose index, maintain multiple specialized "expert" indexes, each optimized for different query types, with a learned routing mechanism that selects which experts to activate for each query.

**Key insight:** Different query types have fundamentally different optimal representations. A code navigation query needs call graph awareness; a semantic similarity query needs distributional embeddings; an exact lookup needs inverted indexes. One index cannot optimize for all.

---

## Table of Contents

1. [Background: Mixture of Experts](#1-background-mixture-of-experts)
2. [Information Retrieval Foundations](#2-information-retrieval-foundations)
3. [Current System Analysis](#3-current-system-analysis)
4. [MoE Applied to Indexing](#4-moe-applied-to-indexing)
5. [Cognitive Science Connections](#5-cognitive-science-connections)
6. [Key Concepts and Terminology](#6-key-concepts-and-terminology)
7. [Prior Art and Research](#7-prior-art-and-research)
8. [Challenges and Considerations](#8-challenges-and-considerations)
9. [Success Metrics](#9-success-metrics)

---

## 1. Background: Mixture of Experts

### 1.1 Origins in Machine Learning

Mixture of Experts (MoE) was introduced by Jacobs et al. (1991) as a method for combining multiple specialized neural networks. The key insight: rather than training one large model to handle all cases, train multiple smaller "expert" models, each specializing in different input regions.

**Core components:**
1. **Expert networks** - Specialized models for different input patterns
2. **Gating network** - Learns which expert(s) to activate for each input
3. **Combination function** - Merges expert outputs into final prediction

### 1.2 Mathematical Foundation

The standard MoE output is:

```
y = Σᵢ gᵢ(x) · fᵢ(x)
```

Where:
- `x` is the input
- `fᵢ(x)` is the output of expert `i`
- `gᵢ(x)` is the gating weight for expert `i`
- `Σᵢ gᵢ(x) = 1` (weights sum to 1)

### 1.3 Sparse MoE

Modern MoE systems use **sparse activation**—only a subset of experts fire for each input:

```
y = Σᵢ∈TopK gᵢ(x) · fᵢ(x)
```

**Benefits:**
- Compute scales with K, not total experts
- Can add experts without slowing inference
- Each expert can specialize more deeply

### 1.4 Key Properties

| Property | Description | Relevance to Indexing |
|----------|-------------|----------------------|
| **Conditional computation** | Different inputs activate different experts | Different queries use different indexes |
| **Specialization** | Experts develop distinct competencies | Indexes optimize for specific query types |
| **Scalability** | Add capacity without proportional compute | Add new index types without slowing all queries |
| **Load balancing** | Distribute work across experts | Prevent any single index from bottlenecking |
| **Routing** | Learn optimal expert selection | Learn which index works best for which queries |

---

## 2. Information Retrieval Foundations

### 2.1 The Index Diversity Problem

Different IR tasks require different data structures:

| Query Type | Optimal Structure | Why |
|------------|-------------------|-----|
| Exact match | Inverted index, hash maps | O(1) lookup |
| Phrase match | Positional index | Word order matters |
| Semantic similarity | Dense embeddings | Distributional semantics |
| Faceted search | Bitmap indexes | Fast set operations |
| Code navigation | AST, call graphs | Structural relationships |
| Temporal queries | Time-series indexes | Ordered access patterns |

**The problem:** A single unified index cannot optimize for all these simultaneously.

### 2.2 Traditional Approaches

**Single unified index:**
- Pro: Simple to maintain
- Con: Suboptimal for specialized queries

**Multiple independent indexes:**
- Pro: Each optimized for its use case
- Con: No coordination, duplicate storage, complex query planning

**Federated search:**
- Pro: Combines multiple sources
- Con: Late fusion loses cross-index insights

**MoE indexing is different:** It adds intelligent routing and early fusion, allowing indexes to inform each other.

### 2.3 Index Types Relevant to This System

Given the Cortical Text Processor's domain (code search, documentation, semantic analysis), these index types are most relevant:

#### Lexical Index
- Inverted index with term positions
- BM25/TF-IDF scoring
- Best for: "Find files containing `authenticate`"

#### Semantic Index
- Distributional similarity
- Concept clusters
- Query expansion
- Best for: "Files about user verification" (finds auth, login, session)

#### Structural Index
- Call graphs
- Import relationships
- AST-aware
- Best for: "What functions call `processDocument`?"

#### Temporal Index
- Change history
- Co-change patterns
- Version relationships
- Best for: "What changed in the last week?"

#### Episodic Index
- Session context
- Recent queries
- Navigation patterns
- Best for: "Similar to what I searched earlier"

---

## 3. Current System Analysis

### 3.1 Existing Architecture

The Cortical Text Processor uses a 4-layer hierarchical architecture:

```
Layer 3 (DOCUMENTS)  ← Full documents
    ↑↓
Layer 2 (CONCEPTS)   ← Semantic clusters (Louvain)
    ↑↓
Layer 1 (BIGRAMS)    ← Word pairs
    ↑↓
Layer 0 (TOKENS)     ← Individual words
```

**Current strengths:**
- Rich semantic relationships (PageRank, TF-IDF, clustering)
- Query expansion through lateral connections
- Code-aware tokenization
- Intent parsing

**Current limitations:**
- Single monolithic index serves all query types
- Simple queries pay semantic expansion overhead
- No structural awareness for code queries
- No temporal dimension

### 3.2 Query Patterns in Current System

Analyzing how the current system handles different queries:

| Query Type | Current Handling | Limitation |
|------------|------------------|------------|
| Exact lookup | Full expansion anyway | Overhead |
| Semantic search | Query expansion + TF-IDF | Good, but slow |
| Code navigation | Treats code as text | Misses structure |
| Definition lookup | Heuristic boosting | Not systematic |
| Temporal queries | Not supported | Can't ask "what changed?" |

### 3.3 Data Flow Analysis

```
Query → Tokenize → Expand → Score Documents → Rank → Return
                     ↓
              (always same path)
```

**The problem:** Every query follows the same path regardless of type. A simple exact-match query goes through full semantic expansion.

### 3.4 Performance Characteristics

From profiling (`scripts/profile_full_analysis.py`):

| Phase | Time | Notes |
|-------|------|-------|
| Tokenization | ~10ms | Fast |
| Query expansion | ~50-200ms | Variable, depends on connections |
| Document scoring | ~100-500ms | Scales with corpus size |
| Ranking | ~20ms | Fast |

For simple queries, expansion and scoring dominate—often unnecessarily.

---

## 4. MoE Applied to Indexing

### 4.1 Core Architecture

```
                    Query
                      │
                      ▼
            ┌─────────────────┐
            │  Query Router   │
            │  (Gating)       │
            └────────┬────────┘
                     │
     ┌───────────────┼───────────────┐
     │ w₁           │ w₂           │ w₃
     ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Expert  │   │ Expert  │   │ Expert  │
│ Index 1 │   │ Index 2 │   │ Index 3 │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
                   ▼
            ┌─────────────┐
            │ Result      │
            │ Fusion      │
            └─────────────┘
```

### 4.2 Expert Index Specializations

#### Expert 1: Lexical Index
**Responsibility:** Fast exact and near-exact matching

**Data structures:**
- Inverted index (term → document list with positions)
- Prefix/suffix tries for autocomplete
- N-gram index for fuzzy matching

**Scoring:** BM25 (lighter than full TF-IDF graph)

**When activated:**
- Short queries (< 4 words)
- Quoted phrases
- Code identifiers
- File path patterns

#### Expert 2: Semantic Index
**Responsibility:** Meaning-based retrieval

**Data structures:**
- Current 4-layer hierarchy
- Concept clusters
- Query expansion graph
- Synonym/hypernym chains

**Scoring:** Expanded TF-IDF with PageRank weighting

**When activated:**
- Question queries ("how do I...", "what is...")
- Conceptual queries ("files about authentication")
- Multi-word descriptive queries

#### Expert 3: Structural Index
**Responsibility:** Code-aware navigation

**Data structures:**
- Function call graphs
- Import/dependency DAGs
- Class hierarchies
- Symbol tables with scope

**Scoring:** Graph distance + reference frequency

**When activated:**
- Queries with identifiers (camelCase, snake_case)
- "What calls X" / "What does X call"
- Import/dependency questions
- Type hierarchy questions

#### Expert 4: Temporal Index
**Responsibility:** Change-aware retrieval

**Data structures:**
- Document version history
- Change timestamp index
- Co-change correlation matrix
- Commit/edit frequency

**Scoring:** Recency-weighted relevance

**When activated:**
- Time expressions ("recently", "last week")
- Change queries ("what changed", "history of")
- Trend queries ("frequently modified")

#### Expert 5: Episodic Index
**Responsibility:** Session-aware personalization

**Data structures:**
- Recent query history (ring buffer)
- Click-through data
- Navigation graph
- User context vector

**Scoring:** Contextual relevance boost

**When activated:**
- Continuation queries ("more like that")
- Implicit context (recent topics boost)
- Pattern detection (user frequently searches X after Y)

### 4.3 Gating Mechanisms

Without ML dependencies (per project philosophy), we implement gating through:

#### Feature-Based Gating
Extract query features and compute expert weights:

```
features:
  - query_length (words)
  - has_quotes (bool)
  - has_identifiers (bool)
  - has_question_word (bool)
  - has_time_expression (bool)
  - code_pattern_score (0-1)

weights[expert] = Σ(feature_weight × feature_value)
```

#### Intent-Based Gating
Use existing intent parser to route:

```
intent_to_experts = {
    'definition': ['lexical', 'semantic'],
    'location': ['structural', 'lexical'],
    'explanation': ['semantic'],
    'relationship': ['structural', 'semantic'],
    'history': ['temporal', 'lexical'],
}
```

#### Feedback-Adaptive Gating
Learn from query success/failure patterns:

```
success_rate[query_pattern][expert] = successes / total

weight[expert] = success_rate[pattern][expert] / Σ(success_rates)
```

### 4.4 Result Fusion Strategies

#### Weighted Score Combination
```
final_score[doc] = Σ(expert_weight × expert_score[doc])
```

#### Rank Fusion (Reciprocal Rank Fusion)
```
RRF_score[doc] = Σ(1 / (k + rank[expert][doc]))
```
Where `k` is typically 60.

#### Learned Combination
Track which expert provided best results for similar queries and weight accordingly.

### 4.5 Cross-Index Communication

The real power of MoE over independent indexes: experts can inform each other.

**Expansion sharing:**
Semantic expert's expanded terms can refine lexical expert's query.

**Structural boosting:**
Structural expert's "callers of X" can boost documents in semantic results.

**Temporal re-ranking:**
Temporal expert's recency scores can re-rank semantic results.

**Episodic contextualization:**
Episodic expert's session context influences all other experts.

---

## 5. Cognitive Science Connections

### 5.1 Memory Systems Analogy

The MoE index architecture maps to cognitive memory systems:

| Expert Index | Memory System | Function |
|--------------|---------------|----------|
| Lexical | Lexical memory | Word forms, spelling |
| Semantic | Semantic memory | Concepts, meanings, relations |
| Structural | Procedural memory | How things work, sequences |
| Temporal | Episodic memory | Events, when things happened |
| Episodic | Working memory | Recent context, attention |

### 5.2 Attention as Routing

The gating network functions like attention:
- **Selective attention**: Not all memory systems are queried for every recall
- **Executive function**: Deciding which systems to engage
- **Inhibition**: Suppressing irrelevant systems

### 5.3 Spreading Activation

The current system already uses spreading activation. In MoE:
- Activation can spread **within** experts (current behavior)
- Activation can spread **across** experts (new capability)

### 5.4 Learning and Adaptation

Feedback-adaptive gating mirrors:
- **Reinforcement learning**: Success strengthens pathways
- **Habituation**: Repeated patterns become automatic
- **Contextual learning**: Different contexts → different strategies

---

## 6. Key Concepts and Terminology

### Core MoE Terms

| Term | Definition |
|------|------------|
| **Expert** | A specialized index optimized for specific query types |
| **Gating network** | The component that decides which experts to activate |
| **Routing** | The process of directing queries to appropriate experts |
| **Sparse activation** | Activating only a subset of experts per query |
| **Load balancing** | Distributing queries evenly across experts |
| **Expert capacity** | How many queries an expert can handle |
| **Auxiliary loss** | Training signal encouraging expert specialization |

### Index-Specific Terms

| Term | Definition |
|------|------------|
| **Inverted index** | Mapping from terms to documents containing them |
| **Positional index** | Inverted index with word positions for phrase queries |
| **Call graph** | Directed graph of function/method calls |
| **AST** | Abstract Syntax Tree representing code structure |
| **Co-change** | Files that are frequently modified together |
| **Session context** | State accumulated during a user's search session |

### Fusion Terms

| Term | Definition |
|------|------------|
| **Early fusion** | Combining signals before scoring |
| **Late fusion** | Combining ranked result lists |
| **Score fusion** | Combining expert scores via weighted sum |
| **Rank fusion** | Combining expert rankings (RRF, Borda count) |
| **Cross-pollination** | Using one expert's output to improve another's |

### Gating Terms

| Term | Definition |
|------|------------|
| **Hard routing** | Select exactly K experts, others get weight 0 |
| **Soft routing** | All experts get some weight, even if small |
| **Top-K routing** | Activate only the K highest-weighted experts |
| **Load-balanced routing** | Route to maintain even expert utilization |
| **Hierarchical routing** | Two-level: first coarse, then fine |

---

## 7. Prior Art and Research

### 7.1 MoE in Large Language Models

Modern LLMs use MoE for efficiency:

**Switch Transformer (Google, 2021)**
- Simplified routing: route to single expert
- 4x efficiency gains

**GShard (Google, 2020)**
- Expert parallelism across devices
- Load balancing via auxiliary loss

**Mixtral (Mistral, 2024)**
- 8 experts, 2 active per token
- Matches GPT-4 at fraction of compute

**Relevance:** Demonstrates sparse MoE scales efficiently.

### 7.2 Federated Search and Metasearch

**CORI (Collection Retrieval Inference)**
- Resource selection for distributed IR
- Estimates collection relevance per query

**ReDDE (Relevant Document Distribution Estimation)**
- Samples collections to estimate result sizes
- Routes queries to likely-relevant collections

**Relevance:** Establishes query-dependent source selection.

### 7.3 Multi-Index Systems

**Elasticsearch + Vector Search**
- Hybrid BM25 + dense retrieval
- Score fusion via reciprocal rank

**Vespa Multi-Phase Ranking**
- Fast first-phase (inverted index)
- Expensive second-phase (ML ranking)

**Relevance:** Production systems combining index types.

### 7.4 Learning-to-Route

**Adaptive Query Processing (AQP)**
- Learn cost models for query operators
- Route to minimize latency

**Learned Index Structures (Kraska et al., 2018)**
- ML models as indexes
- Replace B-trees with neural networks

**Relevance:** Using learning to improve index selection.

### 7.5 Relevant Papers

1. Jacobs et al. (1991) - "Adaptive Mixtures of Local Experts"
2. Shazeer et al. (2017) - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
3. Fedus et al. (2021) - "Switch Transformers: Scaling to Trillion Parameter Models"
4. Craswell et al. (2020) - "Overview of TREC 2020 Deep Learning Track"
5. Tonellotto et al. (2018) - "Efficient Query Processing for Scalable Web Search"

---

## 8. Challenges and Considerations

### 8.1 Synchronization Complexity

**Challenge:** When documents change, all relevant indexes must update consistently.

**Considerations:**
- Atomic updates across indexes
- Eventual consistency vs. strong consistency
- Version tracking per index

**Mitigation:**
- Transaction wrapper for multi-index updates
- Staleness tracking per index (extend existing system)
- Batch updates to amortize overhead

### 8.2 Storage Overhead

**Challenge:** Multiple indexes store overlapping information.

**Estimation:**
- Lexical index: ~0.3x document size
- Semantic index: ~1.0x document size (current)
- Structural index: ~0.2x document size (code only)
- Temporal index: ~0.1x document size
- Episodic index: ~0.05x document size

**Total:** ~1.6-2x current storage

**Mitigation:**
- Share common data structures where possible
- Lazy building of expensive indexes
- Compression strategies per index type

### 8.3 Cold Start Problem

**Challenge:** Without query history, adaptive routing has no signal.

**Mitigation:**
- Feature-based routing as fallback
- Prior distributions from intent analysis
- Rapid adaptation with small learning rate

### 8.4 Expert Collapse

**Challenge:** All experts may converge to similar behavior.

**Detection:**
- Monitor result set diversity across experts
- Track routing weight distributions

**Mitigation:**
- Diversity regularization
- Negative correlation encouragement
- Explicit specialization via data partitioning

### 8.5 Router Complexity

**Challenge:** Router becomes a bottleneck if too complex.

**Target:** Router should be <10% of total query latency.

**Mitigation:**
- Simple feature extraction
- Cached routing for repeated patterns
- Hierarchical routing to reduce expert comparisons

### 8.6 Evaluation Difficulty

**Challenge:** How to evaluate individual experts vs. ensemble?

**Approach:**
- Ablation studies (remove one expert)
- Expert contribution tracking
- User satisfaction metrics

---

## 9. Success Metrics

### 9.1 Latency Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Router overhead | <10ms | Time from query to expert selection |
| Simple query latency | <50ms | Lexical-only queries |
| Complex query latency | <300ms | Multi-expert semantic queries |
| P95 latency | <500ms | 95th percentile across all queries |

### 9.2 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MRR@10 | >0.6 | Mean reciprocal rank at 10 |
| Precision@5 | >0.7 | Relevant in top 5 |
| Expert utilization | 20-80% each | No expert over/underused |
| Result diversity | >0.5 Jaccard | Experts return different results |

### 9.3 Efficiency Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sparse activation rate | 2-3 experts/query | Average experts activated |
| Storage overhead | <2x baseline | Total index size |
| Index update time | <2x baseline | Time to add document |
| Memory footprint | <1.5x baseline | Runtime memory |

### 9.4 Operational Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Routing accuracy | >80% | Expert selected matches query type |
| Fallback rate | <10% | Queries requiring all experts |
| Error isolation | 100% | Single expert failure doesn't crash system |

---

## Appendix A: Glossary Cross-Reference

See [glossary.md](glossary.md) for definitions of core terms used throughout this document.

## Appendix B: Related Documentation

- [architecture.md](architecture.md) - Current system architecture
- [algorithms.md](algorithms.md) - Core algorithm details
- [moe-index-design.md](moe-index-design.md) - Technical design specification
- [moe-index-implementation-plan.md](moe-index-implementation-plan.md) - Implementation phases

---

*This knowledge transfer document provides the conceptual foundation for the MoE index architecture. Refer to the design document for technical specifications and the implementation plan for development phases.*
