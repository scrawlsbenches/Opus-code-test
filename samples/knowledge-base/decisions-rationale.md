# Design Decisions and Rationale

This document explains WHY key design decisions were made in the Cortical Text Processor project. For WHAT and HOW, see the code and documentation. This focuses on practical benefits and reasoning.

---

## Why Hebbian Learning?

We use Hebbian learning ("neurons that fire together wire together") to build lateral connections between co-occurring terms because it mirrors how human memory naturally associates related concepts. When "neural" and "networks" appear together frequently, the system automatically strengthens their connection without requiring labeled training data. This enables semantic search and query expansion to work from corpus structure alone, avoiding the complexity and computational cost of pre-trained word embeddings or neural models.

**Key benefit:** Zero-dependency semantic understanding derived purely from co-occurrence patterns.

---

## Why Dual-Process Architecture (Woven Mind)?

The Woven Mind uses fast (Hive/System 1) and slow (Cortex/System 2) processing because real intelligence requires both automatic pattern matching and deliberate reasoning. The Loom routes inputs based on surprise detection—familiar patterns use fast Hive processing for efficiency, while novel or unexpected inputs engage the slower Cortex for careful analysis. This matches Kahneman's dual-process theory and provides adaptive performance: common queries execute quickly, complex reasoning engages when needed.

**Key benefit:** Intelligent mode switching—fast when possible, thorough when necessary.

---

## Why Zero Runtime Dependencies?

The cortical library has zero runtime dependencies because every dependency is a potential security vulnerability, maintenance burden, and compatibility risk. By implementing PageRank, TF-IDF, BM25, and clustering ourselves in ~13,000 lines of pure Python, we control the entire stack. When security issues emerge in third-party packages (like the arbitrary code execution risk in pickle), we're not blocked waiting for upstream fixes. This also ensures the library works anywhere Python runs without complex installation procedures.

**Key benefit:** Complete control over security, maintenance, and compatibility.

---

## Why Graph of Thought (GoT)?

We use GoT for task tracking instead of simple linear task lists because real software development involves complex dependencies, multiple competing hypotheses, and non-linear reasoning paths. GoT represents tasks, decisions, and evidence as a graph with typed edges (DEPENDS_ON, BLOCKS, SUPPORTS, REFUTES), making it possible to query "what blocks task X?" or "what evidence supports hypothesis Y?" This graph structure prevents work on blocked tasks, surfaces dependency conflicts early, and preserves reasoning context across sessions.

**Key benefit:** Complex dependency tracking and reasoning preservation that simple task lists can't provide.

---

## Why BM25 Over TF-IDF?

BM25 is the default scoring algorithm because it handles term frequency saturation and document length normalization better than classic TF-IDF. In code search, TF-IDF over-weights terms that appear many times in a single file, while BM25 recognizes that the 10th occurrence of "authentication" provides diminishing relevance signal. The length normalization prevents longer documents from dominating results simply because they contain more terms. Empirically, BM25 produces more relevant code search results, especially for repositories with mixed file sizes.

**Key benefit:** Superior relevance ranking for code search compared to TF-IDF, especially with varied document lengths.

---

## Why Batched Processing?

Data generators use batched processing (default batch_size=50) because processing large corpora in a single pass risks timeouts and memory exhaustion. The synthetic data generator processes 149 Python files, 264 Markdown files, and 500 commits—attempting this all at once would exceed execution time limits. Batching also enables progress monitoring, resumable execution via checkpoints, and graceful degradation (partial results on timeout). Each batch commits results incrementally, so interruptions only lose the current batch, not all work.

**Key benefit:** Timeout prevention, memory management, and resumable execution for large-scale processing.

---

## Why JSON Over Pickle?

We migrated from pickle to JSON persistence because pickle has a critical security vulnerability: `pickle.load()` can execute arbitrary code during deserialization, enabling Remote Code Execution (RCE) attacks. If a user loads a malicious `.pkl` file, attackers gain full system access. JSON is immune to this because it only contains data, never code. As a bonus, JSON is git-friendly (human-readable diffs, no binary conflicts), cross-platform compatible, and debuggable without loading into Python. The migration eliminated our primary security vulnerability.

**Key benefit:** Security (no RCE risk), git-friendliness, and cross-platform compatibility.

---

## Why Staleness Tracking?

The processor tracks computation staleness (which analyses are up-to-date vs. needing recomputation) because full recomputation after every document addition would waste time. When you add one document, only TF-IDF values change—PageRank, clustering, and embeddings remain valid. Staleness tracking marks specific computations as stale after changes, then `compute_all()` only recomputes what's needed. For incremental updates, this reduces compute time from seconds to milliseconds. The `_stale_computations` set ensures correctness while maximizing efficiency.

**Key benefit:** Dramatic performance improvement for incremental updates without sacrificing correctness.

---

## Why Test-Driven Development (TDD)?

We require TDD (write failing tests first, then implement) because tests written after implementation tend to test what the code does, not what it should do. Writing tests first forces you to understand the problem and define the contract before diving into implementation. This catches misunderstandings early, creates executable documentation of expected behavior, and prevents the "it works on my machine" problem. The project maintains 98%+ coverage on core logic precisely because tests define the implementation, not the reverse.

**Key benefit:** Better understanding of requirements, executable documentation, and consistent high coverage.

---

## Why Mixin-Based Composition?

The `CorticalTextProcessor` class uses mixin-based composition (CoreMixin, DocumentsMixin, ComputeMixin, etc.) instead of monolithic implementation because a single 3,000-line class is unmaintainable and untestable. Each mixin focuses on one concern: CoreMixin handles initialization, DocumentsMixin manages documents, ComputeMixin runs algorithms. This enables testing mixins in isolation, parallel development on different concerns, and clear ownership of functionality. When bugs occur, mixin boundaries make it obvious where to look.

**Key benefit:** Maintainability, testability, and clear separation of concerns in a large class.

---

## Why Layer Hierarchy (0-3)?

The four-layer architecture (TOKENS → BIGRAMS → CONCEPTS → DOCUMENTS) mirrors visual cortex organization because hierarchical abstraction is how biological systems handle complexity. Layer 0 (tokens) provides raw features, Layer 1 (bigrams) captures local patterns, Layer 2 (concepts) forms semantic clusters, Layer 3 (documents) represents complete units. This hierarchy enables both bottom-up (document → tokens) and top-down (query expansion → related concepts → documents) information flow, matching how human understanding operates at multiple levels of abstraction simultaneously.

**Key benefit:** Multi-level abstraction that supports both granular and holistic text understanding.

---

## Why Graph-Based Embeddings?

We use graph embeddings (adjacency-based, random walk, spectral) derived from the term co-occurrence network rather than neural word embeddings (Word2Vec, GloVe) because graph methods work with zero external training data and reflect the actual corpus structure. A term's embedding is computed from its position in the co-occurrence graph, so embeddings automatically adapt to domain-specific vocabulary without pre-training on billions of tokens. For code search, this means "processor" and "algorithm" are similar because they co-occur in our codebase, not because they're similar in Wikipedia.

**Key benefit:** Domain-adapted embeddings with zero pre-training or external dependencies.

---

## Why Merge-Friendly Task System?

The task system uses unique timestamp-based IDs (T-YYYYMMDD-HHMMSS-{hex}) and separate JSON files instead of a central database because centralized task files cause merge conflicts in parallel development. When multiple agents or branches work simultaneously, each creates tasks with collision-free IDs and writes to separate files. Git merges succeed automatically because files don't overlap. This enables true parallel agent workflows where 5 agents can create 10 tasks each without coordination or conflict resolution.

**Key benefit:** Zero-conflict parallel task creation across agents and branches.

---

## Why Write-Ahead Logging (WAL)?

Graph persistence uses WAL (append-only operation log) before modifying graph state because crash recovery requires knowing what operations were attempted. If a process crashes mid-operation, the WAL contains all successfully completed operations plus the incomplete one. Recovery replays the WAL to restore state up to the last complete operation. This is the same pattern databases use (PostgreSQL WAL, SQLite journal) because it's the only way to guarantee consistency after crashes. Snapshots provide fast restore points, WAL handles incremental changes.

**Key benefit:** Guaranteed crash recovery and state consistency without sacrificing performance.

---

## Why Auto-Commit to Git?

GoT mutations auto-commit to git (when on `claude/*` branches) because cognitive state stored only in memory is fragile in cloud environments where sessions can terminate unexpectedly. Each task creation, edge addition, or decision log is immediately persisted to git, providing a durable audit trail. Protected branches (main, master, prod) are never auto-pushed, preventing accidental contamination. Network failures retry with exponential backoff. This treats git as the persistent store for reasoning state, not just code.

**Key benefit:** Session termination resilience and complete reasoning audit trail.

---

## Why Behavioral vs. Unit Test Distinction?

We distinguish behavioral tests (temporary, may create real data) from unit tests (permanent, mocked) because mixing them leads to test pollution and cleanup failures. Behavioral tests are written during exploration to understand how a feature should work—they often create real GoT tasks or processor state to verify behavior. Once the feature works, behavioral tests are converted to unit tests with mocks, then deleted. This prevents test databases from accumulating thousands of orphaned test entities. The lifecycle difference prevents conflating "how does this work?" tests with "does this still work?" tests.

**Key benefit:** Clean test state, no data pollution, clear test purpose distinction.

---

## Why Corpus-Specific Code Concepts?

The `code_concepts.py` module provides programming concept synonyms (get/fetch/retrieve, database/db/storage) because general-purpose synonym sets don't understand code. When searching for "authentication", you want to find "auth" and "login" even though they're not linguistic synonyms. The concept groups are learned from the codebase itself (ML concepts, frontend concepts, security concepts), making query expansion domain-aware. This dramatically improves code search recall without requiring users to know all possible naming variations.

**Key benefit:** Code-aware query expansion that understands domain-specific terminology.

---

## Why Session-Based Knowledge Transfer?

The ML data collector captures complete session transcripts with all tool uses because context is critical for training a repository-native model. A commit diff shows WHAT changed, but not WHY or what alternatives were considered. Session transcripts preserve the reasoning process: questions asked, files read, decisions made, and verification steps. This contextual data trains models to understand not just "change X to Y" but "when you see problem P, consider solutions A/B/C, then verify with test T".

**Key benefit:** Context-rich training data that captures reasoning, not just outcomes.

---

## Why Consolidation Engine?

Woven Mind's consolidation engine implements "sleep-like" cycles that transfer frequent Hive patterns to Cortex abstractions because working memory (Hive) has limited capacity while long-term memory (Cortex) does not. Without consolidation, the Hive accumulates thousands of specific patterns without extracting general principles. Consolidation mines frequent co-occurrence patterns from Hive, abstracts them into concepts, and transfers to Cortex. This matches biological memory consolidation during sleep and prevents the system from being overwhelmed by specifics.

**Key benefit:** Scalable long-term learning that extracts generalities from specifics.

---

## Why Surprise-Based Mode Switching?

The Loom switches between FAST (Hive) and SLOW (Cortex) modes based on prediction surprise because surprise is the universal signal for "this needs more thought." When Hive predictions closely match input (low surprise), the situation is familiar—use fast pattern matching. When predictions diverge from input (high surprise), something unexpected occurred—engage slow deliberate reasoning. This adaptive strategy prevents wasting Cortex cycles on routine tasks while ensuring novel situations get appropriate attention. The surprise threshold is tunable for different risk tolerances.

**Key benefit:** Adaptive intelligence that's fast by default, thorough when uncertainty demands it.

---

## Why Repository-Native SLM?

We're training a small language model specifically on this repository's code, docs, and task history because general-purpose models (GPT, Claude) don't deeply understand project-specific patterns, conventions, and evolution. A 50M parameter model trained on 28K patterns from this codebase will outperform a 1B parameter general model for repository-specific queries like "Where is PageRank implemented?" or "Why did we migrate from pickle to JSON?" The model learns actual project conventions, not generic programming knowledge.

**Key benefit:** Expert-level repository knowledge in a lightweight, fast, private model.

---

## Summary Table

| Decision | Primary Benefit |
|----------|----------------|
| Hebbian Learning | Zero-dependency semantic understanding |
| Dual-Process (Woven Mind) | Adaptive performance (fast/thorough) |
| Zero Dependencies | Complete security control |
| Graph of Thought | Complex dependency tracking |
| BM25 Scoring | Superior code search relevance |
| Batched Processing | Timeout prevention, resumability |
| JSON over Pickle | Security (no RCE), git-friendly |
| Staleness Tracking | Incremental update efficiency |
| TDD First | Requirements clarity, high coverage |
| Mixin Composition | Maintainability, testability |
| Layer Hierarchy | Multi-level abstraction |
| Graph Embeddings | Domain-adapted, zero pre-training |
| Merge-Friendly Tasks | Zero-conflict parallel work |
| WAL Persistence | Crash recovery, consistency |
| Git Auto-Commit | Session resilience |
| Behavioral/Unit Split | Clean test state |
| Code Concepts | Domain-aware query expansion |
| Session Transcripts | Context-rich training data |
| Consolidation Engine | Scalable long-term learning |
| Surprise Switching | Adaptive intelligence |
| Repository-Native SLM | Expert repository knowledge |

---

**Created:** 2025-12-26
**Purpose:** Training data for Repository-Native SLM to understand design rationale
**Audience:** SLM during training, developers understanding the "why"
