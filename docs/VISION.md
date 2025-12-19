# Cortical Text Processor: Product Vision

> *"A zero-dependency semantic search engine that thinks like a developer"*

## Executive Summary

The Cortical Text Processor is a **hierarchical semantic information retrieval library** designed for AI agents, developers, and teams who need intelligent code and document search without the complexity of vector databases or external dependencies.

This document articulates the **product vision** for the 9 pending "legacy" feature tasks, placing them in strategic context for users, developers, and the AI agents that serve them.

---

## The Core Problem We Solve

**Developers and AI agents need to understand codebases quickly.**

Current solutions fall short:
- **Grep/ripgrep**: Fast but literal—misses semantic meaning
- **Vector databases**: Powerful but require ML pipelines, embeddings, infrastructure
- **IDE search**: Good for known symbols, poor for conceptual queries
- **LLM context windows**: Limited to ~100K tokens, can't hold entire codebases

**Our approach**: Build a graph of semantic relationships using classical IR algorithms (PageRank, TF-IDF, Louvain clustering) that runs anywhere Python runs, with zero external dependencies.

---

## User Personas

### 1. The AI Agent Developer
*"I build AI assistants that help developers navigate codebases"*

**Needs:**
- MCP server integration for Claude Desktop
- Semantic search that understands code concepts
- Fast enough for interactive use (<100ms queries)
- Explainable results (not black-box embeddings)

**Current Support**: ⭐⭐⭐⭐⭐ Exceptional
- 6 Claude skills provided
- AI metadata system for rapid codebase understanding
- Dog-fooding: we search our own code with our own system

**Future Needs** (Legacy Tasks):
- Streaming results for large result sets (LEGACY-188)
- Async API for non-blocking integrations (LEGACY-187)

---

### 2. The Platform Engineer
*"I need to deploy semantic search as a service for my team"*

**Needs:**
- REST API (not just MCP)
- Health checks and monitoring
- Horizontal scaling for multiple users
- Docker/Kubernetes deployment patterns

**Current Support**: ⭐⭐⭐ Moderate
- MCP server works for single-user scenarios
- Basic observability exists (metrics, timing)
- Docker examples provided

**Future Needs** (Legacy Tasks):
- REST API wrapper with FastAPI (LEGACY-190)
- WAL + snapshot persistence for fault tolerance (LEGACY-133)
- Chunked parallel processing for large corpora (LEGACY-135)

---

### 3. The Internal Contributor
*"I want to extend the system with custom algorithms"*

**Needs:**
- Clear extension points
- Plugin architecture for custom analyzers
- Good test coverage to catch regressions
- Learning resources to understand internals

**Current Support**: ⭐⭐⭐⭐ Good
- Mixin-based architecture is extensible
- 3,800+ tests with >90% coverage
- CLAUDE.md provides expert guidance

**Future Needs** (Legacy Tasks):
- Plugin/extension registry (LEGACY-100)
- Learning Mode for contributors (LEGACY-080)
- Code pattern detection for understanding codebases (LEGACY-078)

---

### 4. The Interactive Developer
*"I want to explore a codebase interactively"*

**Needs:**
- REPL for ad-hoc queries
- Visual feedback on what's happening
- Quick iteration on search strategies

**Current Support**: ⭐⭐ Limited
- CLI scripts exist but aren't interactive
- No visualization of graph structure
- Must write Python to experiment

**Future Needs** (Legacy Tasks):
- Interactive REPL mode (LEGACY-191)

---

## Strategic Roadmap

The 9 legacy tasks map to **three strategic phases**:

### Phase 1: Production Readiness (High Priority)
*Enable deployment beyond single-developer use*

| Task | Value | Effort | Priority |
|------|-------|--------|----------|
| **LEGACY-190**: REST API (FastAPI) | Opens to non-MCP clients | 2 weeks | HIGH |
| **LEGACY-187**: Async API | Non-blocking for production | 3 weeks | HIGH |
| **LEGACY-133**: WAL + snapshot persistence | Fault tolerance | 3 weeks | MEDIUM |

**Why this matters**: Without these, the system is limited to local development and MCP-only integrations. REST API alone opens doors to:
- Web frontends
- CI/CD integrations
- Language-agnostic clients
- Load balancer deployments

---

### Phase 2: Scale & Performance (Medium Priority)
*Handle larger corpora and more users*

| Task | Value | Effort | Priority |
|------|-------|--------|----------|
| **LEGACY-135**: Chunked parallel processing | 10x corpus size | 4 weeks | MEDIUM |
| **LEGACY-188**: Streaming query results | Large result sets | 2 weeks | MEDIUM |

**Why this matters**: Current architecture handles ~500 documents well. Enterprise codebases have 10,000+ files. These features bridge that gap without requiring distributed infrastructure.

---

### Phase 3: Ecosystem & Experience (Lower Priority)
*Build community and improve usability*

| Task | Value | Effort | Priority |
|------|-------|--------|----------|
| **LEGACY-100**: Plugin registry | Community contributions | 3 weeks | LOW |
| **LEGACY-080**: Learning Mode | Onboarding | 2 weeks | LOW |
| **LEGACY-078**: Code pattern detection | Intelligence | 2 weeks | LOW |
| **LEGACY-191**: Interactive REPL | Developer experience | 2 weeks | LOW |

**Why this matters**: These features accelerate adoption and build community, but the core product must work in production first.

---

## Architectural Principles

### Zero Dependencies, Maximum Portability
The library has **no runtime dependencies**. This is intentional:
- Runs on any Python 3.8+ environment
- No NumPy, no ML frameworks, no databases
- Single `pip install` deploys everywhere

**Implication for legacy tasks**: New features must maintain this principle. REST API uses only stdlib or optional dependencies.

### Graph-Based Intelligence
All semantic understanding flows from **graph algorithms**:
- PageRank for importance
- TF-IDF for distinctiveness
- Louvain for clustering
- Co-occurrence for relationships

**Implication for legacy tasks**: New intelligence features (LEGACY-078 pattern detection) should leverage existing graph infrastructure, not add ML dependencies.

### Explainability Over Black Boxes
Users can trace why a document matched:
- See query expansion terms
- Inspect connection weights
- Understand PageRank contribution

**Implication for legacy tasks**: Learning Mode (LEGACY-080) and REPL (LEGACY-191) should expose these internals, not hide them.

---

## Deep Algorithm Analysis

The Cortical Text Processor achieves semantic understanding through a carefully orchestrated ensemble of classical IR algorithms. Each algorithm plays a specific role, and their combination creates emergent intelligence that exceeds any single technique.

### Algorithm 1: PageRank — Importance Discovery

**Purpose:** Identify which terms matter most in the corpus, independent of raw frequency.

**Implementation:** `cortical/analysis/pagerank.py`

**How It Works:**
```
importance[term] = (1 - damping) / N + damping × Σ (neighbor_importance × edge_weight / neighbor_outgoing_sum)
```

The algorithm iteratively propagates importance through the term co-occurrence graph. Terms that are referenced by many important terms become important themselves—a recursive definition that converges to stable values.

**Key Parameters:**
- `damping = 0.85`: The probability of following a link vs. jumping to a random node
- `tolerance = 1e-6`: Convergence threshold (stops when no term changes by more than this)
- `max_iterations = 20`: Upper bound on iterations

**Three Variants:**
1. **Standard PageRank**: Applied to Layer 0 (tokens) and Layer 1 (bigrams)
2. **Semantic PageRank**: Adjusts edge weights by relation type (IsA connections count 1.5× more than CoOccurs)
3. **Hierarchical PageRank**: Propagates importance across all 4 layers with separate cross-layer damping

**Why This Matters for Code Search:**
- Common utility functions referenced everywhere get high PageRank
- Core abstractions that everything depends on surface naturally
- Prevents over-emphasis on boilerplate code that appears frequently but isn't semantically central

**Performance:** O(iterations × edges), typically 100-500ms for 10K tokens with early convergence usually at 5-10 iterations.

---

### Algorithm 2: BM25/TF-IDF — Distinctiveness Scoring

**Purpose:** Score how well a term distinguishes a specific document from the rest of the corpus.

**Implementation:** `cortical/analysis/tfidf.py`

**BM25 Formula (Default):**
```
BM25(t, d) = IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

Where:
- `IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)` — Inverse document frequency with smoothing
- `tf(t,d)` — Term frequency in document d
- `k1 = 1.2` — Term frequency saturation (diminishing returns after ~12 occurrences)
- `b = 0.75` — Length normalization factor

**Why BM25 Over TF-IDF:**
- Non-negative IDF even for terms appearing in most documents
- Length normalization prevents long files from unfairly dominating
- Term frequency saturation models realistic relevance (saying "API" 100 times doesn't make a doc 100× more relevant than saying it once)

**Dual Storage Strategy:**
- **Global TF-IDF** (`col.tfidf`): Term importance to entire corpus
- **Per-Document TF-IDF** (`col.tfidf_per_doc[doc_id]`): Term importance within specific document

This dual approach allows:
- Fast corpus-wide importance filtering
- Accurate per-document relevance scoring for search

---

### Algorithm 3: Louvain Community Detection — Concept Discovery

**Purpose:** Discover semantic clusters (concepts) from the term co-occurrence graph.

**Implementation:** `cortical/analysis/clustering.py`

**Two-Phase Algorithm:**

**Phase 1 — Local Optimization:**
```
for each node:
    find neighboring communities
    calculate modularity gain for moving to each
    move to best community if gain > 0
repeat until no nodes move
```

**Phase 2 — Network Aggregation:**
```
collapse each community into a single super-node
edges between communities become edges between super-nodes
repeat Phase 1 on the aggregated network
```

**Modularity Formula:**
```
Q = (1/2m) × Σ [A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)
```

The algorithm optimizes Q, which measures how much edge weight falls within communities versus what would be expected by random chance.

**Resolution Parameter:**
- `resolution = 1.0` (default): Balanced clusters, ~32 concepts
- `resolution = 0.5`: Coarse clusters, ~38 concepts (max cluster 64% of tokens)
- `resolution = 2.0`: Fine-grained clusters, ~79 concepts (max cluster 4.2% of tokens)

**Concept Naming:**
```python
top_members = sorted(cluster_members, key=lambda m: m.pagerank, reverse=True)[:3]
concept_name = '/'.join(top_members)  # e.g., "neural/learning/networks"
```

**Why This Matters:**
- Enables concept-level search ("find documents about authentication")
- Reduces dimensionality while preserving semantic structure
- Creates Layer 2 (Concepts) that bridges raw terms and documents

---

### Algorithm 4: Query Expansion — Semantic Bridging

**Purpose:** Transform literal query terms into semantically enriched term sets.

**Implementation:** `cortical/query/expansion.py`

**Three Expansion Methods:**

1. **Lateral Connection Expansion:**
   - Follow co-occurrence edges from query terms
   - Score: `edge_weight × neighbor_score × 0.6`
   - Takes top 5 neighbors per query term

2. **Concept Cluster Membership:**
   - Find concepts containing query terms
   - Add other cluster members as expansions
   - Score: `concept.pagerank × member.pagerank × 0.4`

3. **Code Concept Synonyms:**
   - Programming-specific synonym groups (get/fetch/load, create/make/build)
   - Limited to 3 synonyms per term to prevent drift

**Multi-Hop Inference:**
```
Query: "neural"
  Hop 0: neural (1.0)
  Hop 1: networks (0.4), learning (0.35)
  Hop 2: deep (0.098) — via learning with decay
```

Chain validity is scored by relation type pairs:
- `(IsA, IsA)`: 1.0 — fully transitive (dog→animal→living_thing)
- `(RelatedTo, RelatedTo)`: 0.6 — weaker transitivity
- `(Antonym, Antonym)`: 0.3 — double negation, avoid

---

### Algorithm 5: Graph-Boosted Search (GB-BM25) — Hybrid Ranking

**Purpose:** Combine BM25 relevance with graph structure signals.

**Implementation:** `cortical/query/search.py:425-564`

**Scoring Formula:**
```
final_score = (0.5 × normalized_bm25) + (0.3 × normalized_pagerank) + (0.2 × normalized_proximity)
            × coverage_multiplier (0.5 to 1.5)
```

**Three Signal Sources:**

1. **BM25 Base Score (50%):**
   - Standard term frequency × inverse document frequency
   - Per-document scoring using `col.tfidf_per_doc`

2. **PageRank Boost (30%):**
   - Sum of matched term PageRanks
   - Rewards documents containing important terms

3. **Proximity Boost (20%):**
   - For each pair of original query terms:
     - Check if they're connected in the co-occurrence graph
     - If connected, boost documents containing both
   - Rewards documents where query terms appear together

**Coverage Multiplier:**
- Documents matching 1/5 query terms: 0.7× multiplier
- Documents matching all 5 query terms: 1.5× multiplier
- Prevents documents matching one rare term from outranking documents matching many terms

---

### Algorithm 6: Semantic Relation Extraction — Knowledge Graph Construction

**Purpose:** Extract typed relationships (IsA, PartOf, Causes) from document text.

**Implementation:** `cortical/semantics.py`

**Pattern-Based Extraction:**
24 regex patterns detect 10+ relation types:
```python
r'(\w+)\s+(?:is|are)\s+(?:a|an)\s+(?:type\s+of\s+)?(\w+)' → IsA (0.9 confidence)
r'(\w+)\s+(?:is|are)\s+(?:a\s+)?part\s+of' → PartOf (0.95 confidence)
r'(\w+)\s+(?:causes|leads?\s+to)' → Causes (0.9 confidence)
```

**Semantic Retrofitting:**
Blends co-occurrence weights with semantic relation knowledge:
```
new_weight = α × original_weight + (1-α) × semantic_target_weight
```
With α = 0.3, semantic signals dominate (70%) while preserving some corpus statistics (30%).

**Relation Weight Multipliers:**
| Relation | Weight | Semantics |
|----------|--------|-----------|
| SameAs | 2.0 | Strongest synonymy |
| IsA | 1.5 | Hypernymy |
| PartOf | 1.3 | Meronymy |
| RelatedTo | 0.8 | Generic |
| Antonym | -0.5 | Opposition |

---

### Algorithm Synergy: How They Work Together

The real power emerges from how these algorithms interact:

```
Document Ingestion:
  ├─→ Tokenization → Layer 0 (tokens)
  ├─→ Bigram extraction → Layer 1 (bigrams)
  ├─→ Co-occurrence counting → lateral connections
  └─→ Document indexing → Layer 3 (documents)

Compute Phase:
  ├─→ TF-IDF/BM25 → distinctiveness scores
  ├─→ PageRank → importance scores (uses lateral connections)
  ├─→ Louvain clustering → concept clusters (uses importance + connections)
  └─→ Semantic extraction → typed relations → retrofitted connections

Query Phase:
  ├─→ Query expansion (uses lateral connections + concepts + PageRank)
  ├─→ GB-BM25 search (uses TF-IDF + PageRank + proximity)
  └─→ Multi-stage ranking (uses concepts → documents → passages)
```

**Key Insight:** Each algorithm feeds into the next. PageRank uses the connection graph. Louvain uses PageRank scores for naming clusters. Query expansion uses both connections and concepts. Search combines all signals. This layered approach creates compound intelligence.

---

## Author's Reflections

### On the Power of Classical Algorithms

The Cortical Text Processor demonstrates that **semantic understanding doesn't require neural networks**. PageRank, TF-IDF, and Louvain are algorithms from the 1990s-2000s, yet their combination produces remarkably intelligent behavior:

- Finding relevant code by understanding concepts, not just keywords
- Discovering what's *important* vs. what's merely *frequent*
- Bridging terminology gaps through multi-hop inference

This isn't to dismiss modern ML approaches—they excel at many tasks. But there's profound value in systems that are:
1. **Explainable**: Every result can be traced through expansion weights, PageRank contributions, and connection paths
2. **Portable**: Zero dependencies means running anywhere Python runs
3. **Debuggable**: When results are wrong, you can inspect the graph and fix the model

### On the Graph Metaphor

The "cortical" metaphor is apt not because this system mimics actual neurons, but because it captures a key insight: **understanding emerges from connections**. Individual terms are meaningless; their meaning arises from relationships to other terms.

This is why co-occurrence graphs work so well for semantic search. Words that appear together share context. Context is meaning.

### On Zero Dependencies

The constraint of zero external dependencies forced creative solutions:
- Pure Python PageRank instead of NumPy matrix operations
- Regex-based relation extraction instead of NLP libraries
- Iterative Louvain instead of graph library implementations

These constraints produced a system that's simultaneously simpler (no dependency hell) and more educational (implementations are readable). Every algorithm is visible in the source.

### On What's Missing

The 9 legacy tasks represent genuine gaps:
- **REST API**: The current system requires Python. HTTP would democratize access.
- **Async API**: Blocking calls limit web application integration.
- **Streaming**: Large result sets shouldn't require loading everything into memory.
- **Interactive REPL**: Exploration should be frictionless.

These aren't nice-to-haves—they're the difference between a library and a platform.

### On Future Directions

The most exciting future isn't in the legacy tasks, but in what becomes possible after them:

1. **Hybrid Search**: Combining graph-based retrieval with LLM reranking
2. **Active Learning**: Using search feedback to improve the graph
3. **Cross-Language**: Applying the same algorithms to different programming languages
4. **Real-Time Updates**: Incremental graph updates as code changes

The foundation is solid. The algorithms are proven. What remains is packaging them for production use.

---

## Success Metrics

### For Production Readiness
- REST API handles 100 requests/second
- Async operations don't block event loop
- System recovers from crashes without data loss

### For Scale
- Process 10,000 documents in <5 minutes
- Query latency <100ms at P95
- Memory usage scales linearly with corpus size

### For Ecosystem
- 5+ community plugins in registry within 6 months
- Contributor onboarding time <2 hours
- REPL sessions average 15+ minutes (engagement)

---

## What We're NOT Building

To maintain focus, we explicitly **won't** pursue:

1. **Distributed architecture** - Single machine is our sweet spot
2. **Vector embeddings** - Our graph approach is different by design
3. **Real-time streaming ingestion** - Batch processing is fine
4. **Multi-tenant SaaS** - Deploy your own instance
5. **GUI/Web UI** - CLI and API are primary interfaces

---

## The Path Forward

### Immediate (Next Sprint)
1. Archive legacy tasks as formal backlog items
2. Create GitHub issues with detailed specs
3. Prioritize LEGACY-190 (REST API) as first major feature

### Near-Term (1-3 Months)
1. Ship REST API with auth and rate limiting
2. Implement async compute for non-blocking operations
3. Add streaming results for large queries

### Medium-Term (3-6 Months)
1. Chunked parallel processing for scale
2. Plugin registry for community contributions
3. Interactive REPL for exploration

### Long-Term (6-12 Months)
1. Learning Mode for contributor onboarding
2. Advanced pattern detection
3. Production deployment guides and case studies

---

## Conclusion

The 9 legacy tasks represent a **coherent product roadmap** that evolves the Cortical Text Processor from a powerful library into a **production-ready semantic search platform**.

The priorities are clear:
1. **Production readiness first** (REST API, async, persistence)
2. **Scale second** (parallel processing, streaming)
3. **Ecosystem third** (plugins, REPL, learning mode)

By following this roadmap, we serve all four user personas while maintaining our core principles of zero dependencies, graph-based intelligence, and explainability.

---

*"Understanding code shouldn't require a GPU cluster. It should require understanding."*

---

## Appendix: Legacy Task Details

| ID | Title | Category | Status |
|----|-------|----------|--------|
| LEGACY-078 | Add code pattern detection | Intelligence | Backlog |
| LEGACY-080 | Add "Learning Mode" for contributors | Experience | Backlog |
| LEGACY-100 | Implement plugin/extension registry | Ecosystem | Backlog |
| LEGACY-133 | Implement WAL + snapshot persistence | Production | Backlog |
| LEGACY-135 | Implement chunked parallel processing | Scale | Backlog |
| LEGACY-187 | Add async API support | Production | Backlog |
| LEGACY-188 | Add streaming query results | Scale | Backlog |
| LEGACY-190 | Create REST API wrapper (FastAPI) | Production | Backlog |
| LEGACY-191 | Add Interactive REPL mode | Experience | Backlog |
