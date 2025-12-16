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
