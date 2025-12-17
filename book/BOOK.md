# The Cortical Chronicles

*A Self-Documenting Living Book*

---

This document is automatically generated from the Cortical Text Processor codebase.
It consolidates all book chapters into a single markdown file for offline reading,
PDF generation, or direct viewing on GitHub.

---

## Table of Contents

### [Preface](#preface)

- [How This Book Works](#how-this-book-works)
- [Project Origin: From Package to Platform](#project-origin-from-package-to-platform)
- [The Living Book Vision](#the-living-book-vision)

### [Foundations: Core Algorithms](#foundations-core-algorithms)

- [BM25/TF-IDF — Distinctiveness Scoring](#bm25tf-idf-distinctiveness-scoring)
- [Graph-Boosted Search (GB-BM25) — Hybrid Ranking](#graph-boosted-search-gb-bm25-hybrid-ranking)
- [Louvain Community Detection — Concept Discovery](#louvain-community-detection-concept-discovery)
- [PageRank — Importance Discovery](#pagerank-importance-discovery)
- [Query Expansion — Semantic Bridging](#query-expansion-semantic-bridging)
- [Semantic Relation Extraction — Knowledge Graph Construction](#semantic-relation-extraction-knowledge-graph-construction)

### [Architecture: System Design](#architecture-system-design)

- [Architecture Overview](#architecture-overview)
- [Graph Algorithms](#graph-algorithms)
- [Configuration](#configuration)
- [Data Structures](#data-structures)
- [NLP Components](#nlp-components)
- [Observability](#observability)
- [Persistence Layer](#persistence-layer)
- [Core Processor](#core-processor)
- [Search & Retrieval](#search-retrieval)
- [Utilities](#utilities)

### [Decisions: ADRs](#decisions-adrs)

- [ADR-001: Add Microseconds to Task ID Generation](#adr-001-add-microseconds-to-task-id-generation)
- [Architectural Decision Records](#architectural-decision-records)

### [Evolution: Project History](#evolution-project-history)

- [Bug Fixes and Lessons](#bug-fixes-and-lessons)
- [Feature Evolution](#feature-evolution)
- [Refactorings and Architecture Evolution](#refactorings-and-architecture-evolution)
- [Test](#test)
- [Project Timeline](#project-timeline)

### [Future: Roadmap](#future-roadmap)

- [Future Chapter](#future-chapter)

### [05 Case Studies](#05-case-studies)

- [The Great Performance Hunt](#the-great-performance-hunt)
- [Case Studies](#case-studies)
- [Case Study: Add session memory and knowledge transfer](#case-study-add-session-memory-and-knowledge-transfer)
- [Case Study: Bug Fix - Add test file penalty and code stop word filtering to search](#case-study-bug-fix-add-test-file-penalty-and-code-stop-word-filtering-to-search)
- [Case Study: Bug Fix - Replace external action with native Python link checker](#case-study-bug-fix-replace-external-action-with-native-python-link-checker)
- [Case Study: Bug Fix - Update skipped tests for processor/ package refactor](#case-study-bug-fix-update-skipped-tests-for-processor-package-refactor)
- [Case Study: Bug Fix - Use heredoc for Python in CI to avoid YAML syntax error](#case-study-bug-fix-use-heredoc-for-python-in-ci-to-avoid-yaml-syntax-error)
- [Case Study: Clean up directory structure and queue search relevance fixes](#case-study-clean-up-directory-structure-and-queue-search-relevance-fixes)
- [Case Study: Feature Development - Add director agent orchestration prompt](#case-study-feature-development-add-director-agent-orchestration-prompt)
- [Case Study: Feature Development - Add session handoff, auto-memory, CI link checker, and tests](#case-study-feature-development-add-session-handoff-auto-memory-ci-link-checker-and-tests)
- [Case Study: Refactoring - Migrate to merge-friendly task system and add security tasks](#case-study-refactoring-migrate-to-merge-friendly-task-system-and-add-security-tasks)
- [Case Study: Refactoring - Split processor.py into modular processor/ package (LEGACY-095)](#case-study-refactoring-split-processorpy-into-modular-processor-package-legacy-095)
- [Case Study: Update task ID format test to expect microseconds](#case-study-update-task-id-format-test-to-expect-microseconds)
- [Case Study: Add ML commit data for previous commit](#case-study-add-ml-commit-data-for-previous-commit)
- [Case Study: Add ML commit data](#case-study-add-ml-commit-data)
- [Case Study: Bug Fix - Harden ML data collector with critical fixes](#case-study-bug-fix-harden-ml-data-collector-with-critical-fixes)
- [Case Study: Bug Fix - Increase ML data retention to 2 years for training milestones](#case-study-bug-fix-increase-ml-data-retention-to-2-years-for-training-milestones)
- [Case Study: Bug Fix - Stop tracking ML commit data files (too large for GitHub)](#case-study-bug-fix-stop-tracking-ml-commit-data-files-too-large-for-github)
- [Case Study: Bug Fix - Update tests for BM25 default and stop word tokenization](#case-study-bug-fix-update-tests-for-bm25-default-and-stop-word-tokenization)
- [Case Study: Feature Development - Add CI status integration for ML outcome tracking](#case-study-feature-development-add-ci-status-integration-for-ml-outcome-tracking)
- [Case Study: Feature Development - Add comprehensive delegation command template](#case-study-feature-development-add-comprehensive-delegation-command-template)
- [Case Study: Feature Development - Add export, feedback, and quality-report commands to ML collector](#case-study-feature-development-add-export-feedback-and-quality-report-commands-to-ml-collector)
- [Case Study: Feature Development - Add lightweight commit data for ephemeral environments](#case-study-feature-development-add-lightweight-commit-data-for-ephemeral-environments)
- [Case Study: Feature Development - Add schema validation for ML data integrity](#case-study-feature-development-add-schema-validation-for-ml-data-integrity)
- [Case Study: Refactoring - Consolidate ML data to single JSONL files](#case-study-refactoring-consolidate-ml-data-to-single-jsonl-files)
- [Case Study: Add unit tests for Cortical Chronicles generators](#case-study-add-unit-tests-for-cortical-chronicles-generators)
- [Case Study: Bug Fix - Address critical ML data collection and prediction issues](#case-study-bug-fix-address-critical-ml-data-collection-and-prediction-issues)
- [Case Study: Bug Fix - Archive ML session after transcript processing (T-003 16f3)](#case-study-bug-fix-archive-ml-session-after-transcript-processing-t-003-16f3)
- [Case Study: Bug Fix - Fix ML data collection milestone counting and add session/action capture](#case-study-bug-fix-fix-ml-data-collection-milestone-counting-and-add-sessionaction-capture)
- [Case Study: Bug Fix - Prevent infinite commit loop in ML data collection hooks](#case-study-bug-fix-prevent-infinite-commit-loop-in-ml-data-collection-hooks)
- [Case Study: ML data sync](#case-study-ml-data-sync)
- [Case Study: ML tracking data](#case-study-ml-tracking-data)
- [Synthesized Case Studies](#synthesized-case-studies)

### [06 Lessons](#06-lessons)

- [Lessons Learned](#lessons-learned)
- [Architecture Lessons](#architecture-lessons)
- [Correctness Lessons](#correctness-lessons)
- [Performance Lessons](#performance-lessons)
- [Testing Lessons](#testing-lessons)

### [07 Concepts](#07-concepts)

- [Concept Evolution: Bigram](#concept-evolution-bigram)
- [Concept Evolution: Bm25](#concept-evolution-bm25)
- [Concept Evolution: Clustering](#concept-evolution-clustering)
- [Concept Evolution: Context](#concept-evolution-context)
- [Concept Evolution: Definition](#concept-evolution-definition)
- [Concept Evolution: Embeddings](#concept-evolution-embeddings)
- [Concept Evolution: Graph](#concept-evolution-graph)
- [Concept Evolution: Incremental](#concept-evolution-incremental)
- [Concept Evolution Index](#concept-evolution-index)
- [Concept Evolution: Louvain](#concept-evolution-louvain)
- [Concept Evolution: Pagerank](#concept-evolution-pagerank)
- [Concept Evolution: Query Expansion](#concept-evolution-query-expansion)
- [Concept Evolution: Search](#concept-evolution-search)
- [Concept Evolution: Semantic](#concept-evolution-semantic)
- [Concept Evolution: Tokenization](#concept-evolution-tokenization)

### [08 Exercises](#08-exercises)

- [Exercises: Advanced](#exercises-advanced)
- [Exercises: Foundations](#exercises-foundations)
- [Exercises: Search](#exercises-search)
- [Exercise Index](#exercise-index)

### [09 Journey](#09-journey)

- [Your Learning Journey](#your-learning-journey)
- [Learning Journey: Advanced](#learning-journey-advanced)
- [Learning Journey: Beginner](#learning-journey-beginner)
- [Learning Journey: Intermediate](#learning-journey-intermediate)

---

# Preface

## How This Book Works

> *"The best documentation is the kind that writes itself."*

## Overview

The Cortical Chronicles is a **self-documenting book**. It uses the Cortical Text Processor—the very system it documents—to generate its own content. This creates a fascinating recursive property: the book understands itself through the same algorithms it explains.

## The Generation Process

```
┌─────────────────────────────────────────────────────────────┐
│                    BOOK GENERATION                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Source Files           Generators           Chapters        │
│  ───────────           ──────────           ────────        │
│                                                              │
│  docs/VISION.md    →   AlgorithmGen    →   01-foundations/  │
│  cortical/*.ai_meta →  ModuleDocGen    →   02-architecture/ │
│  samples/decisions/ →  DecisionGen     →   03-decisions/    │
│  git log           →   NarrativeGen    →   04-evolution/    │
│  tasks/            →   RoadmapGen      →   05-future/       │
│                                                              │
│                    ↓                                         │
│              search-index.json                               │
│                    ↓                                         │
│               index.html (searchable)                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Chapter Types

### 01-foundations/
Algorithm deep-dives extracted from `docs/VISION.md`. Each algorithm (PageRank, BM25, Louvain, etc.) gets its own chapter with:
- Purpose and intuition
- Mathematical formulation
- Implementation details
- Why it matters for code search

### 02-architecture/
Module documentation generated from `.ai_meta` files. Includes:
- Module purpose and dependencies
- Key functions and classes
- Mermaid dependency graphs

### 03-decisions/
Architecture Decision Records from `samples/decisions/`. Documents the "why" behind design choices.

### 04-evolution/
A narrative of project history generated from git commits. Transforms raw commit logs into a readable story of how the project evolved.

### 05-future/
Roadmap and vision from task files and `VISION.md`. Shows where the project is heading.

## The Self-Reference Loop

Here's what makes this book special:

1. **The processor indexes its own code** → Creates a semantic graph
2. **The generators query that graph** → Find relevant content
3. **The book explains those algorithms** → Reader understands the system
4. **The system processes those explanations** → Understands itself better

This isn't just cute—it's a powerful test of the system's capabilities. If the Cortical Text Processor can understand and explain itself, it can understand any codebase.

## Regenerating the Book

The book regenerates automatically on every push to `main`:

```bash
# Manual regeneration
python scripts/generate_book.py

# Generate specific chapter
python scripts/generate_book.py --chapter foundations

# Preview without writing
python scripts/generate_book.py --dry-run
```

## Searching the Book

The book includes a semantic search interface. Open `index.html` to:
- Search by keyword or concept
- Browse by chapter
- Follow cross-references

The search uses the same algorithms described in the book—query expansion, BM25 scoring, PageRank boosting.

## See Also

- [Algorithm Analysis](../01-foundations/index.md) - Deep dive into the algorithms
- [Architecture](../02-architecture/index.md) - How the code is organized
- [Source: generate_book.py](../../scripts/generate_book.py) - The generation script

## Source Files

This chapter was written manually as the seed for the book. Future chapters are auto-generated from:
- `scripts/generate_book.py` - The orchestrator
- `docs/VISION.md:185-430` - Algorithm documentation source

---

*This chapter is part of [The Cortical Chronicles](../README.md),
a self-documenting book generated by the Cortical Text Processor.*

---

## Project Origin: From Package to Platform

*How a zip file became a self-documenting semantic search engine.*

## The Beginning

On December 9th, 2025, a package arrived. Not through a mail service, but as a GitHub upload: `cortical_package.zip`. Inside this compressed archive were the seeds of what would become a sophisticated information retrieval system—11,100 lines of Python implementing biological metaphors for text analysis.

```
Commit: 74d8f6b - Extract cortical_package from zip archive
Date:   2025-12-09 18:12:22 +0000
```

The extraction was straightforward. The journey ahead was anything but.

## The First Hour: Cleanup and Discovery

Within minutes of extraction, the cleanup began. The package had arrived with traces of its previous life—cached Python bytecode, temporary files, the detritus of development. These were swept away methodically:

```
f66a1ae - Clean up repository structure         (18:16:54 UTC)
324d751 - Add Python patterns to .gitignore     (18:19:25 UTC)
```

But cleanup revealed something more interesting: the code had bugs. Not catastrophic failures, but the kind of edge cases that emerge when sophisticated algorithms meet real-world data.

## The Bug List: A Contract Emerges

At 18:31:03 UTC, barely an hour after extraction, `TASK_LIST.md` appeared:

```
Commit: d89ceee - Add TASK_LIST.md documenting required bug fixes
```

This wasn't just a to-do list. It was an acknowledgment: *This code is powerful, but it needs care.* The list documented validation gaps, edge cases in clustering algorithms, missing error handling. Twenty minutes later, the fixes landed:

```
Commit: 75097e9 - Fix bugs and add comprehensive unit tests
Time:   18:51:22 UTC
```

The system could now handle empty inputs, validate parameters, and fail gracefully. But more importantly, it established a pattern: **test first, ship second**.

## The Foundation: CLAUDE.md is Born

Two minutes after the bug fixes, at 18:53:11 UTC, something remarkable happened:

```
Commit: 9badbcb - Add CLAUDE.md project guide for Claude Code
```

This wasn't just documentation. It was a *contract between human and AI developers*. CLAUDE.md laid out:

- **Persona**: You are a senior computational neuroscience engineer
- **Philosophy**: Profile before optimizing, understand before acting
- **Architecture**: Here's how the layers work, here's where to find things
- **Rules**: Don't use underscores in bigrams, always use `get_by_id()` for O(1) lookups

It was both a guide and a guardrail. A way to ensure that future modifications—whether by human or AI—would respect the system's design principles.

## The Corpus Grows

The next phase focused on content. Seven sample documents arrived first, then 44 more. The system needed diversity to test its semantic clustering:

```
b69a296 - Add 7 new sample documents             (19:02:23 UTC)
892c826 - Add 44 diverse sample documents        (23:05:22 UTC)
```

From machine learning papers to cooking recipes, from code documentation to philosophical essays—the corpus became a microcosm of human knowledge. The system could now be tested not just with toy examples, but with real semantic complexity.

## The RAG Journey: Tasks 9-30

Then came the transformation. Between December 9th and 10th, 2025, the system evolved from basic text processing to full Retrieval-Augmented Generation capabilities. The commits tell the story:

### Phase 1: The RAG Foundations (Dec 9, ~19:51-19:57 UTC)

```
2085418 - Add document metadata support (Task 9)
bf75e5d - Activate Layer 2 concept clustering (Task 10)
f27d18e - Integrate semantic relations (Task 11)
8f862b0 - Persist full computed state (Task 12)
```

In six minutes of commits, the system gained:
- **Citations**: Documents could now carry metadata for proper attribution
- **Concepts**: Layer 2 clustering activated by default
- **Relations**: Semantic understanding woven into retrieval
- **Persistence**: Full state saving including graph embeddings

### Phase 2: Production Features (Dec 9, ~21:06-21:14 UTC)

```
38fb4f7 - Add incremental document indexing (Task 15)
b3c29af - Add multi-stage ranking pipeline (Task 17)
900cce1 - Add batch query API (Task 18)
```

The system could now handle live updates, sophisticated ranking, and batch processing. It was becoming production-ready.

### Phase 3: The ConceptNet Vision (Dec 9-10, ~22:37 onwards)

Then the ambition escalated:

```
c6eefdc - Add ConceptNet-enhanced PageRank task list
        (Tasks 19-30 planned at 22:37:45 UTC)
```

Over the next two hours, twelve tasks were completed:

- **Cross-layer connections**: Feedforward and feedback between layers
- **Lateral connections**: Within bigrams and concepts
- **Typed edges**: Relation types (IsA, PartOf, UsedFor, etc.)
- **Multi-hop inference**: Reasoning chains across the graph
- **Pattern extraction**: Automatic relation discovery from text
- **Graph export**: ConceptNet-style visualization

By midnight UTC on December 10th, the system had evolved from a package to a *knowledge graph platform*.

## The Numbers That Tell the Story

| Metric | Initial Package | After 24 Hours | Today |
|--------|----------------|----------------|-------|
| **Commits** | 1 (extraction) | ~40 | 699 |
| **Code Lines** | ~8,500 | ~10,000 | ~11,100 |
| **Tasks Completed** | 0 | 30 | 200+ |
| **Test Coverage** | Unknown | >80% | >89% |
| **Sample Documents** | 0 | 51 | 125+ |

## The Self-Documenting Dream

Hundreds of commits later, on December 16th, 2025, something extraordinary happened. The system that had been built to analyze text began to analyze *itself*:

```
c730057 - Add Cortical Chronicles book infrastructure (Wave 1)
3022110 - Add content generators (Wave 2)
0022466 - Add search integration and web interface (Wave 3)
940fdf2 - Add CI workflow and documentation (Wave 4)
```

The book you're reading now was generated by the code it describes. The system indexed its own source code, its own documentation, its own commit history—and synthesized this narrative.

## Where We Are Now

Today, the Cortical Text Processor is:

- **~11,100 lines** of core library code
- **Zero external dependencies** for runtime
- **699 commits** of careful evolution
- **89%+ test coverage** maintained rigorously
- **4-layer architecture** with typed semantic relations
- **A book that writes itself** from living code

But more than the numbers, it embodies a philosophy:

> *"Profile before optimizing, understand before acting, test before shipping."*

That philosophy started in CLAUDE.md on day one. It continues in every commit, every test, every design decision.

## The Commits That Started It All

| Commit | Date | Milestone |
|--------|------|-----------|
| `27a6531` | Nov 25, 2025 | Repository created |
| `74d8f6b` | Dec 9, 18:12 | Package extraction |
| `9badbcb` | Dec 9, 18:53 | CLAUDE.md foundation |
| `2085418` | Dec 9, 19:51 | RAG journey begins (Task 9) |
| `e40a80c` | Dec 10, 00:24 | ConceptNet integration complete (Task 29) |
| `c730057` | Dec 16, 2025 | Self-documenting book begins |
| `082aa21` | Dec 16, 2025 | Six intelligent book generators |

## What This Means

A project's origin story isn't just historical curiosity. It reveals:

1. **Intent**: This was designed to be *understood*, not just used
2. **Discipline**: Testing and documentation weren't afterthoughts
3. **Ambition**: From day one, the goal was semantic understanding
4. **Partnership**: Human and AI working together, guided by CLAUDE.md

The zip file that arrived on December 9th contained code. But the code that emerged in the days and weeks that followed became something more: a platform for knowledge, a canvas for collaboration, and ultimately, its own biographer.

---

*The journey from package to platform took 699 commits. The journey from platform to self-awareness took one more: the commit that added this book.*

---

## The Living Book Vision

> *"Code tells you what. Comments tell you why. A living book tells you the journey."*

## The Idea

What if documentation wrote itself—not as an afterthought, but as a natural byproduct of development?

This isn't science fiction. It's what you're reading right now.

The Cortical Chronicles is a **living book**: it grows with the codebase, captures the stories behind decisions, and transforms raw development artifacts into narrative chapters. Every debugging session becomes a case study. Every bugfix becomes a lesson. Every commit becomes a paragraph in an ongoing story.

## How It Works

### The Data We Already Capture

During development, we're already collecting everything a book needs:

```
┌─────────────────────────────────────────────────────────────────┐
│                 THE RAW MATERIALS OF A BOOK                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Developer Questions    → "Why isn't search finding X?"          │
│  Investigation Traces   → Files read, tools used, paths tried    │
│  Breakthroughs         → "Aha! The bottleneck is in bigrams!"    │
│  Solutions             → Commits with diffs and context          │
│  Decisions             → ADRs with rationale                     │
│  Outcomes              → CI results, test coverage               │
│                                                                  │
│  Together, these form NARRATIVE ARCS:                           │
│                                                                  │
│  Problem → Investigation → Discovery → Solution → Lesson         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### From Data to Story

The generators transform structured data into readable chapters:

| Data Source | Generator | Chapter Type |
|-------------|-----------|--------------|
| ML Sessions | CaseStudyGenerator | Problem-solving narratives |
| Bugfix Commits | LessonExtractor | Distilled wisdom |
| ADRs + Context | DecisionStoryGenerator | The "why" behind choices |
| Concept Clusters | ConceptEvolutionGenerator | How ideas grew over time |
| Test Cases | ExerciseGenerator | Reader engagement |
| PageRank Scores | ReaderJourneyGenerator | Progressive learning paths |

## What Makes This Different

### Traditional Documentation

```
process_document(doc_id, text)

Process a document and add it to the corpus.

Parameters:
  doc_id: Unique document identifier
  text: Document content

Returns:
  None
```

### A Living Book

> **Chapter 5: Processing Your First Document**
>
> Before the system can search, it needs to understand. That understanding begins with `process_document()`.
>
> Think of it as reading a book for the first time. You don't just see words—you build a mental map: which ideas connect, which concepts are central, which phrases recur.
>
> The processor does the same thing, but algorithmically. When you call:
>
> ```python
> processor.process_document("readme", open("README.md").read())
> ```
>
> ...a cascade of analysis begins. Tokens are extracted. Bigrams form. Lateral connections strengthen between co-occurring terms. The document joins a growing graph of semantic relationships.
>
> **Try It Yourself:** Process the CLAUDE.md file and examine what concepts emerge. Which terms have the highest PageRank? What does that tell you about the document's focus?

## The Chapter Types

### Case Studies: Learning from Real Problems

Every debugging session is a story waiting to be told:

> **Case Study: The Great Performance Hunt**
>
> It started with a timeout. The `compute_all()` function was hanging on just 125 documents.
>
> The obvious suspect was Louvain clustering—our most complex algorithm. But we profiled first...

### Lessons: Wisdom Distilled from Experience

600+ commits contain patterns worth preserving:

> **Lesson #23: Profile Before Optimizing**
>
> **The Mistake:** Assumed Louvain was slow because it's complex.
> **The Reality:** 99% of time was in `bigram_connections()`.
> **The Principle:** The obvious culprit is often innocent. Data beats intuition.

### Concept Evolution: Watching Ideas Grow

Track how key concepts emerged and strengthened:

> **How "Importance" Became a First-Class Concept**
>
> Week 1: First mention in `analysis.py`—a simple PageRank score.
> Week 3: Connected to "relevance", "ranking", "boost".
> Week 6: Cluster of 15 related terms. Central to search quality.

### Exercises: Active Learning

Test cases become teaching moments:

> **Exercise: Query Expansion**
>
> Given the query "neural networks", write code to expand it with related terms.
>
> *Hint 1:* Use `processor.expand_query()`
> *Hint 2:* Consider lateral connections
> *Solution:* [Reveal]

## The Self-Reference Loop

Here's the beautiful recursion:

1. **We write code** → Creates development artifacts
2. **ML captures the process** → Structured session data
3. **Generators synthesize narratives** → Book chapters
4. **The book explains the system** → Readers understand
5. **Understanding leads to better code** → Loop continues

The Cortical Text Processor documents itself using its own algorithms. If it can understand and explain itself, it can understand any codebase.

## Why This Matters

### For Developers

- Documentation stays current automatically
- Lessons are captured, not forgotten
- Onboarding accelerates via structured learning paths

### For Teams

- Institutional knowledge persists across turnover
- Decisions are documented with full context
- Best practices emerge from analyzed patterns

### For Publishers

- Authentic problem-solving narratives
- Genuine lessons from real development
- A unique angle: the book that writes itself

## The Vision

Imagine every software project generating its own living book:

- New team members read the story of how the system evolved
- Debugging sessions become teaching materials
- Architecture decisions carry their full context
- The gap between "code" and "understanding" closes

This isn't just documentation. It's **computational autobiography**—a system telling its own story through the act of being built.

---

## What You'll Find in This Book

| Section | Content |
|---------|---------|
| **Foundations** | The algorithms that power semantic search |
| **Architecture** | How the code is organized and why |
| **Decisions** | The choices that shaped the system |
| **Evolution** | Timeline of how we got here |
| **Case Studies** | Problem-solving narratives |
| **Lessons** | Distilled wisdom from 600+ commits |
| **Concepts** | How key ideas emerged and grew |
| **Exercises** | Hands-on learning opportunities |
| **Journey** | Your personalized learning path |

Each section is generated from the codebase itself—living documentation that grows with the system.

---

## See Also

- [How This Book Works](./how-this-book-works.md) - Technical details of generation
- [Full Vision Document](../../docs/BOOK-GENERATION-VISION.md) - Complete specification
- [Product Vision](../../docs/VISION.md) - Overall product direction

---

*This chapter is part of [The Cortical Chronicles](../README.md),
a self-documenting book generated by the Cortical Text Processor.*

---

# Foundations: Core Algorithms

## BM25/TF-IDF — Distinctiveness Scoring

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

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Graph-Boosted Search (GB-BM25) — Hybrid Ranking

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

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Louvain Community Detection — Concept Discovery

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

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## PageRank — Importance Discovery

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

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Query Expansion — Semantic Bridging

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

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Semantic Relation Extraction — Knowledge Graph Construction

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

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

# Architecture: System Design

## Architecture Overview

This section documents the architecture of the Cortical Text Processor through automatically extracted module metadata.

## Statistics

- **Total Modules**: 45
- **Module Groups**: 9
- **Classes**: 50
- **Functions**: 422

## Module Groups

### [Analysis](mod-analysis.md)

8 modules:

- `__init__.py`
- `activation.py`
- `clustering.py`
- `connections.py`
- `pagerank.py`
- `quality.py`
- `tfidf.py`
- `utils.py`

### [Configuration](mod-configuration.md)

3 modules:

- `config.py`
- `constants.py`
- `validation.py`

### [Data Structures](mod-data-structures.md)

3 modules:

- `layers.py`
- `minicolumn.py`
- `types.py`

### [Nlp](mod-nlp.md)

3 modules:

- `embeddings.py`
- `semantics.py`
- `tokenizer.py`

### [Observability](mod-observability.md)

3 modules:

- `observability.py`
- `progress.py`
- `results.py`

### [Persistence](mod-persistence.md)

3 modules:

- `chunk_index.py`
- `persistence.py`
- `state_storage.py`

### [Processor](mod-processor.md)

6 modules:

- `__init__.py`
- `compute.py`
- `core.py`
- `documents.py`
- `introspection.py`
- `persistence_api.py`

### [Query](mod-query.md)

8 modules:

- `__init__.py`
- `chunking.py`
- `definitions.py`
- `expansion.py`
- `intent.py`
- `passages.py`
- `ranking.py`
- `search.py`

### [Utilities](mod-utilities.md)

8 modules:

- `cli_wrapper.py`
- `code_concepts.py`
- `diff.py`
- `fingerprint.py`
- `fluent.py`
- `gaps.py`
- `mcp_server.py`
- `patterns.py`

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Graph Algorithms

Graph algorithms for computing importance, relevance, and clusters.

## Modules

- **__init__.py**: Analysis Module
- **activation.py**: Activation propagation algorithm.
- **clustering.py**: Clustering algorithms for community detection.
- **connections.py**: Connection building algorithms for network layers.
- **pagerank.py**: PageRank algorithms for importance scoring.
- **quality.py**: Clustering quality metrics.
- **tfidf.py**: TF-IDF and BM25 scoring algorithms.
- **utils.py**: Utility functions and classes for analysis algorithms.


## __init__.py

Analysis Module
===============

Graph analysis algorithms for the cortical network.

Contains implementations of:
- PageRank for importance scoring
- TF-IDF for term weighting
- Louvain community det...


### Dependencies

**Standard Library:**

- `activation.propagate_activation`
- `clustering._louvain_core`
- `clustering.build_concept_clusters`
- `clustering.cluster_by_label_propagation`
- `clustering.cluster_by_louvain`
- ... and 22 more



## activation.py

Activation propagation algorithm.

Contains:
- propagate_activation: Spread activation through the network layers


### Functions

#### propagate_activation

```python
propagate_activation(layers: Dict[CorticalLayer, HierarchicalLayer], iterations: int = 3, decay: float = 0.8, lateral_weight: float = 0.3) -> None
```

Propagate activation through the network.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`



## clustering.py

Clustering algorithms for community detection.

Contains:
- cluster_by_louvain: Louvain modularity optimization (recommended)
- cluster_by_label_propagation: Label propagation clustering (legacy)
- bu...


### Functions

#### cluster_by_label_propagation

```python
cluster_by_label_propagation(layer: HierarchicalLayer, min_cluster_size: int = 3, max_iterations: int = 20, cluster_strictness: float = 1.0, bridge_weight: float = 0.0) -> Dict[int, List[str]]
```

Cluster minicolumns using label propagation.

#### cluster_by_louvain

```python
cluster_by_louvain(layer: HierarchicalLayer, min_cluster_size: int = 3, resolution: float = 1.0, max_iterations: int = 10) -> Dict[int, List[str]]
```

Cluster minicolumns using Louvain community detection.

#### build_concept_clusters

```python
build_concept_clusters(layers: Dict[CorticalLayer, HierarchicalLayer], clusters: Dict[int, List[str]], doc_vote_threshold: float = 0.1) -> None
```

Build concept layer from token clusters.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`
- `typing.List`
- ... and 2 more



## connections.py

Connection building algorithms for network layers.

Contains:
- compute_document_connections: Build document-to-document similarity connections
- compute_bigram_connections: Build lateral connections ...


### Functions

#### compute_concept_connections

```python
compute_concept_connections(layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: List[Tuple[str, str, str, float]] = None, min_shared_docs: int = 1, min_jaccard: float = 0.1, use_member_semantics: bool = False, use_embedding_similarity: bool = False, embedding_threshold: float = 0.3, embeddings: Dict[str, List[float]] = None) -> Dict[str, Any]
```

Build lateral connections between concepts in Layer 2.

#### compute_bigram_connections

```python
compute_bigram_connections(layers: Dict[CorticalLayer, HierarchicalLayer], min_shared_docs: int = 1, component_weight: float = 0.5, chain_weight: float = 0.7, cooccurrence_weight: float = 0.3, max_bigrams_per_term: int = 100, max_bigrams_per_doc: int = 500, max_connections_per_bigram: int = 50) -> Dict[str, Any]
```

Build lateral connections between bigrams in Layer 1.

#### compute_document_connections

```python
compute_document_connections(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], min_shared_terms: int = 3) -> None
```

Build lateral connections between documents.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `minicolumn.Minicolumn`
- `typing.Any`
- ... and 5 more



## pagerank.py

PageRank algorithms for importance scoring.

Contains:
- compute_pagerank: Standard PageRank for a single layer
- compute_semantic_pagerank: PageRank with semantic relation weighting
- compute_hierarc...


### Functions

#### compute_pagerank

```python
compute_pagerank(layer: HierarchicalLayer, damping: float = 0.85, iterations: int = 20, tolerance: float = 1e-06) -> Dict[str, float]
```

Compute PageRank scores for minicolumns in a layer.

#### compute_semantic_pagerank

```python
compute_semantic_pagerank(layer: HierarchicalLayer, semantic_relations: List[Tuple[str, str, str, float]], relation_weights: Optional[Dict[str, float]] = None, damping: float = 0.85, iterations: int = 20, tolerance: float = 1e-06) -> Dict[str, Any]
```

Compute PageRank with semantic relation type weighting.

#### compute_hierarchical_pagerank

```python
compute_hierarchical_pagerank(layers: Dict[CorticalLayer, HierarchicalLayer], layer_iterations: int = 10, global_iterations: int = 5, damping: float = 0.85, cross_layer_damping: float = 0.7, tolerance: float = 0.0001) -> Dict[str, Any]
```

Compute PageRank with cross-layer propagation.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.RELATION_WEIGHTS`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Any`
- ... and 4 more



## quality.py

Clustering quality metrics.

Contains:
- compute_clustering_quality: Comprehensive quality evaluation (modularity, silhouette, balance)
- _compute_modularity: Modularity Q metric
- _compute_silhouette...


### Functions

#### compute_clustering_quality

```python
compute_clustering_quality(layers: Dict[CorticalLayer, HierarchicalLayer], sample_size: int = 500) -> Dict[str, Any]
```

Compute clustering quality metrics for the concept layer.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `random`
- `typing.Any`
- ... and 3 more



## tfidf.py

TF-IDF and BM25 scoring algorithms.

Contains:
- compute_tfidf: Traditional TF-IDF scoring
- compute_bm25: Okapi BM25 scoring with length normalization
- _tfidf_core: Pure TF-IDF algorithm for unit te...


### Functions

#### compute_tfidf

```python
compute_tfidf(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> None
```

Compute TF-IDF scores for tokens.

#### compute_bm25

```python
compute_bm25(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], doc_lengths: Dict[str, int], avg_doc_length: float, k1: float = 1.2, b: float = 0.75) -> None
```

Compute BM25 scores for tokens.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- `typing.Dict`
- `typing.Tuple`



## utils.py

Utility functions and classes for analysis algorithms.

Contains:
- SparseMatrix: Zero-dependency sparse matrix for bigram connections
- Similarity functions: cosine_similarity, _doc_similarity, _vect...


### Classes

#### SparseMatrix

Simple sparse matrix implementation using dictionary of keys (DOK) format.

**Methods:**

- `set`
- `get`
- `multiply_transpose`
- `get_nonzero`

### Functions

#### cosine_similarity

```python
cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float
```

Compute cosine similarity between two sparse vectors.

#### SparseMatrix.set

```python
SparseMatrix.set(self, row: int, col: int, value: float) -> None
```

Set value at (row, col).

#### SparseMatrix.get

```python
SparseMatrix.get(self, row: int, col: int) -> float
```

Get value at (row, col).

#### SparseMatrix.multiply_transpose

```python
SparseMatrix.multiply_transpose(self) -> 'SparseMatrix'
```

Multiply this matrix by its transpose: M * M^T

#### SparseMatrix.get_nonzero

```python
SparseMatrix.get_nonzero(self) -> List[Tuple[int, int, float]]
```

Get all non-zero entries.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `math`
- `typing.Dict`
- `typing.List`
- `typing.Tuple`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Configuration

Configuration management and validation.

## Modules

- **config.py**: Configuration Module
- **constants.py**: Centralized constants for the Cortical Text Processor.
- **validation.py**: Validation Module


## config.py

Configuration Module
====================

Centralized configuration for the Cortical Text Processor.

This module provides a dataclass-based configuration system that allows
users to customize algori...


### Classes

#### CorticalConfig

Configuration settings for the Cortical Text Processor.

**Methods:**

- `copy`
- `to_dict`
- `from_dict`

### Functions

#### get_default_config

```python
get_default_config() -> CorticalConfig
```

Get a new instance of the default configuration.

#### CorticalConfig.copy

```python
CorticalConfig.copy(self) -> 'CorticalConfig'
```

Create a copy of this configuration.

#### CorticalConfig.to_dict

```python
CorticalConfig.to_dict(self) -> Dict
```

Convert configuration to a dictionary for serialization.

#### CorticalConfig.from_dict

```python
CorticalConfig.from_dict(cls, data: Dict) -> 'CorticalConfig'
```

Create configuration from a dictionary.

### Dependencies

**Standard Library:**

- `dataclasses.dataclass`
- `dataclasses.field`
- `math`
- `typing.Dict`
- `typing.FrozenSet`
- ... and 1 more



## constants.py

Centralized constants for the Cortical Text Processor.

This module provides a single source of truth for constants used across
multiple modules, preventing drift and inconsistencies.

Task #96: Centr...


### Dependencies

**Standard Library:**

- `typing.Dict`
- `typing.FrozenSet`



## validation.py

Validation Module
=================

Input validation utilities and decorators for the Cortical Text Processor.

This module provides reusable validators and decorators to ensure
parameters are valid ...


### Functions

#### validate_non_empty_string

```python
validate_non_empty_string(value: Any, param_name: str) -> None
```

Validate that a value is a non-empty string.

#### validate_positive_int

```python
validate_positive_int(value: Any, param_name: str) -> None
```

Validate that a value is a positive integer.

#### validate_non_negative_int

```python
validate_non_negative_int(value: Any, param_name: str) -> None
```

Validate that a value is a non-negative integer.

#### validate_range

```python
validate_range(value: Any, param_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None, inclusive: bool = True) -> None
```

Validate that a numeric value is within a specified range.

#### validate_params

```python
validate_params(**validators: Callable[[Any], None]) -> Callable[[F], F]
```

Decorator to validate function parameters.

#### marks_stale

```python
marks_stale(*computation_types: str) -> Callable[[F], F]
```

Decorator to mark computations as stale after method execution.

#### marks_fresh

```python
marks_fresh(*computation_types: str) -> Callable[[F], F]
```

Decorator to mark computations as fresh after method execution.

### Dependencies

**Standard Library:**

- `functools.wraps`
- `inspect`
- `typing.Any`
- `typing.Callable`
- `typing.Optional`
- ... and 2 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Data Structures

Fundamental data structures used throughout the system.

## Modules

- **layers.py**: Layers Module
- **minicolumn.py**: Minicolumn Module
- **types.py**: Type Aliases for the Cortical Text Processor.


## layers.py

Layers Module
=============

Defines the hierarchical layer structure inspired by the visual cortex.

The neocortex processes information through a hierarchy of layers,
each extracting progressively m...


### Classes

#### CorticalLayer

Enumeration of cortical processing layers.

**Methods:**

- `description`
- `analogy`

#### HierarchicalLayer

A layer in the cortical hierarchy containing minicolumns.

**Methods:**

- `get_or_create_minicolumn`
- `get_minicolumn`
- `get_by_id`
- `remove_minicolumn`
- `column_count`
- `total_connections`
- `average_activation`
- `activation_range`
- `sparsity`
- `top_by_pagerank`
- `top_by_tfidf`
- `top_by_activation`
- `to_dict`
- `from_dict`

### Functions

#### CorticalLayer.description

```python
CorticalLayer.description(self) -> str
```

Human-readable description of this layer.

#### CorticalLayer.analogy

```python
CorticalLayer.analogy(self) -> str
```

Visual cortex analogy for this layer.

#### HierarchicalLayer.get_or_create_minicolumn

```python
HierarchicalLayer.get_or_create_minicolumn(self, content: str) -> Minicolumn
```

Get existing minicolumn or create new one.

#### HierarchicalLayer.get_minicolumn

```python
HierarchicalLayer.get_minicolumn(self, content: str) -> Optional[Minicolumn]
```

Get a minicolumn by content, or None if not found.

#### HierarchicalLayer.get_by_id

```python
HierarchicalLayer.get_by_id(self, col_id: str) -> Optional[Minicolumn]
```

Get a minicolumn by its ID in O(1) time.

#### HierarchicalLayer.remove_minicolumn

```python
HierarchicalLayer.remove_minicolumn(self, content: str) -> bool
```

Remove a minicolumn from this layer.

#### HierarchicalLayer.column_count

```python
HierarchicalLayer.column_count(self) -> int
```

Return the number of minicolumns in this layer.

#### HierarchicalLayer.total_connections

```python
HierarchicalLayer.total_connections(self) -> int
```

Return total number of lateral connections in this layer.

#### HierarchicalLayer.average_activation

```python
HierarchicalLayer.average_activation(self) -> float
```

Calculate average activation across all minicolumns.

#### HierarchicalLayer.activation_range

```python
HierarchicalLayer.activation_range(self) -> tuple
```

Return (min, max) activation values.

#### HierarchicalLayer.sparsity

```python
HierarchicalLayer.sparsity(self, threshold_fraction: float = 0.5) -> float
```

Calculate sparsity (fraction of columns with below-average activation).

#### HierarchicalLayer.top_by_pagerank

```python
HierarchicalLayer.top_by_pagerank(self, n: int = 10) -> list
```

Get top minicolumns by PageRank score.

#### HierarchicalLayer.top_by_tfidf

```python
HierarchicalLayer.top_by_tfidf(self, n: int = 10) -> list
```

Get top minicolumns by TF-IDF score.

#### HierarchicalLayer.top_by_activation

```python
HierarchicalLayer.top_by_activation(self, n: int = 10) -> list
```

Get top minicolumns by activation level.

#### HierarchicalLayer.to_dict

```python
HierarchicalLayer.to_dict(self) -> Dict
```

Convert layer to dictionary for serialization.

#### HierarchicalLayer.from_dict

```python
HierarchicalLayer.from_dict(cls, data: Dict) -> 'HierarchicalLayer'
```

Create a layer from dictionary representation.

### Dependencies

**Standard Library:**

- `enum.IntEnum`
- `minicolumn.Minicolumn`
- `typing.Dict`
- `typing.Iterator`
- `typing.Optional`



## minicolumn.py

Minicolumn Module
=================

Core data structure representing a cortical minicolumn.

In the neocortex, minicolumns are vertical structures containing
~80-100 neurons that respond to similar f...


### Classes

#### Edge

Typed edge with metadata for ConceptNet-style graph representation.

**Methods:**

- `to_dict`
- `from_dict`

#### Minicolumn

A minicolumn represents a single concept/feature at a given hierarchy level.

**Methods:**

- `lateral_connections`
- `lateral_connections`
- `add_lateral_connection`
- `add_lateral_connections_batch`
- `set_lateral_connection_weight`
- `add_typed_connection`
- `get_typed_connection`
- `get_connections_by_type`
- `get_connections_by_source`
- `add_feedforward_connection`
- `add_feedback_connection`
- `connection_count`
- `top_connections`
- `to_dict`
- `from_dict`

### Functions

#### Edge.to_dict

```python
Edge.to_dict(self) -> Dict
```

Convert to dictionary for serialization.

#### Edge.from_dict

```python
Edge.from_dict(cls, data: Dict) -> 'Edge'
```

Create an Edge from dictionary representation.

#### Minicolumn.lateral_connections

```python
Minicolumn.lateral_connections(self, value: Dict[str, float]) -> None
```

Set lateral connections from a dictionary (for deserialization).

#### Minicolumn.add_lateral_connection

```python
Minicolumn.add_lateral_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a lateral connection to another column.

#### Minicolumn.add_lateral_connections_batch

```python
Minicolumn.add_lateral_connections_batch(self, connections: Dict[str, float]) -> None
```

Add or strengthen multiple lateral connections at once.

#### Minicolumn.set_lateral_connection_weight

```python
Minicolumn.set_lateral_connection_weight(self, target_id: str, weight: float) -> None
```

Set the weight of a lateral connection directly (not additive).

#### Minicolumn.add_typed_connection

```python
Minicolumn.add_typed_connection(self, target_id: str, weight: float = 1.0, relation_type: str = 'co_occurrence', confidence: float = 1.0, source: str = 'corpus') -> None
```

Add or update a typed connection with metadata.

#### Minicolumn.get_typed_connection

```python
Minicolumn.get_typed_connection(self, target_id: str) -> Optional[Edge]
```

Get a typed connection by target ID.

#### Minicolumn.get_connections_by_type

```python
Minicolumn.get_connections_by_type(self, relation_type: str) -> List[Edge]
```

Get all typed connections with a specific relation type.

#### Minicolumn.get_connections_by_source

```python
Minicolumn.get_connections_by_source(self, source: str) -> List[Edge]
```

Get all typed connections from a specific source.

#### Minicolumn.add_feedforward_connection

```python
Minicolumn.add_feedforward_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a feedforward connection to a lower layer column.

#### Minicolumn.add_feedback_connection

```python
Minicolumn.add_feedback_connection(self, target_id: str, weight: float = 1.0) -> None
```

Add or strengthen a feedback connection to a higher layer column.

#### Minicolumn.connection_count

```python
Minicolumn.connection_count(self) -> int
```

Return the number of lateral connections.

#### Minicolumn.top_connections

```python
Minicolumn.top_connections(self, n: int = 5) -> list
```

Get the strongest lateral connections.

#### Minicolumn.to_dict

```python
Minicolumn.to_dict(self) -> Dict
```

Convert to dictionary for serialization.

#### Minicolumn.from_dict

```python
Minicolumn.from_dict(cls, data: Dict) -> 'Minicolumn'
```

Create a minicolumn from dictionary representation.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Dict`
- `typing.List`
- ... and 2 more



## types.py

Type Aliases for the Cortical Text Processor.

This module provides type aliases for complex return types used throughout
the library, making function signatures more readable and maintainable.

Task ...


### Dependencies

**Standard Library:**

- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## NLP Components

Natural language processing components for tokenization and semantics.

## Modules

- **embeddings.py**: Embeddings Module
- **semantics.py**: Semantics Module
- **tokenizer.py**: Tokenizer Module


## embeddings.py

Embeddings Module
=================

Graph-based embeddings for the cortical network.

Implements three methods for computing term embeddings from the
connection graph structure:
1. Adjacency: Direct ...


### Functions

#### compute_graph_embeddings

```python
compute_graph_embeddings(layers: Dict[CorticalLayer, HierarchicalLayer], dimensions: int = 64, method: str = 'adjacency', max_terms: Optional[int] = None) -> Tuple[Dict[str, List[float]], Dict[str, Any]]
```

Compute embeddings for tokens based on graph structure.

#### embedding_similarity

```python
embedding_similarity(embeddings: Dict[str, List[float]], term1: str, term2: str) -> float
```

Compute cosine similarity between two term embeddings.

#### find_similar_by_embedding

```python
find_similar_by_embedding(embeddings: Dict[str, List[float]], term: str, top_n: int = 10) -> List[Tuple[str, float]]
```

Find terms most similar to a given term by embedding.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- `random`
- ... and 5 more



## semantics.py

Semantics Module
================

Corpus-derived semantic relations and retrofitting.

Extracts semantic relationships from co-occurrence patterns,
then uses them to adjust connection weights (retrof...


### Functions

#### extract_pattern_relations

```python
extract_pattern_relations(documents: Dict[str, str], valid_terms: Set[str], min_confidence: float = 0.5) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations using pattern matching on document text.

#### get_pattern_statistics

```python
get_pattern_statistics(relations: List[Tuple[str, str, str, float]]) -> Dict[str, Any]
```

Get statistics about extracted pattern-based relations.

#### extract_corpus_semantics

```python
extract_corpus_semantics(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], tokenizer, window_size: int = 5, min_cooccurrence: int = 2, use_pattern_extraction: bool = True, min_pattern_confidence: float = 0.6, max_similarity_pairs: int = 100000, min_context_keys: int = 3) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations from corpus co-occurrence patterns.

#### retrofit_connections

```python
retrofit_connections(layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: List[Tuple[str, str, str, float]], iterations: int = 10, alpha: float = 0.3) -> Dict[str, Any]
```

Retrofit lateral connections using semantic relations.

#### retrofit_embeddings

```python
retrofit_embeddings(embeddings: Dict[str, List[float]], semantic_relations: List[Tuple[str, str, str, float]], iterations: int = 10, alpha: float = 0.4) -> Dict[str, Any]
```

Retrofit embeddings using semantic relations.

#### get_relation_type_weight

```python
get_relation_type_weight(relation_type: str) -> float
```

Get the weight for a relation type.

#### build_isa_hierarchy

```python
build_isa_hierarchy(semantic_relations: List[Tuple[str, str, str, float]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
```

Build IsA parent-child hierarchy from semantic relations.

#### get_ancestors

```python
get_ancestors(term: str, parents: Dict[str, Set[str]], max_depth: int = 10) -> Dict[str, int]
```

Get all ancestors of a term with their depth in the hierarchy.

#### get_descendants

```python
get_descendants(term: str, children: Dict[str, Set[str]], max_depth: int = 10) -> Dict[str, int]
```

Get all descendants of a term with their depth in the hierarchy.

#### inherit_properties

```python
inherit_properties(semantic_relations: List[Tuple[str, str, str, float]], decay_factor: float = 0.7, max_depth: int = 5) -> Dict[str, Dict[str, Tuple[float, str, int]]]
```

Compute inherited properties for all terms based on IsA hierarchy.

#### compute_property_similarity

```python
compute_property_similarity(term1: str, term2: str, inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]], direct_properties: Optional[Dict[str, Dict[str, float]]] = None) -> float
```

Compute similarity between terms based on shared properties (direct + inherited).

#### apply_inheritance_to_connections

```python
apply_inheritance_to_connections(layers: Dict[CorticalLayer, HierarchicalLayer], inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]], boost_factor: float = 0.3) -> Dict[str, Any]
```

Boost lateral connections between terms that share inherited properties.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.RELATION_WEIGHTS`
- `copy`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 10 more



## tokenizer.py

Tokenizer Module
================

Text tokenization with stemming and word variant support.

Like early visual processing, the tokenizer extracts basic features
(words) from raw input, filtering nois...


### Classes

#### Tokenizer

Text tokenizer with stemming and word variant support.

**Methods:**

- `tokenize`
- `extract_ngrams`
- `stem`
- `get_word_variants`
- `add_word_mapping`

### Functions

#### split_identifier

```python
split_identifier(identifier: str) -> List[str]
```

Split a code identifier into component words.

#### Tokenizer.tokenize

```python
Tokenizer.tokenize(self, text: str, split_identifiers: Optional[bool] = None) -> List[str]
```

Extract tokens from text.

#### Tokenizer.extract_ngrams

```python
Tokenizer.extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]
```

Extract n-grams from token list.

#### Tokenizer.stem

```python
Tokenizer.stem(self, word: str) -> str
```

Apply simple suffix stripping (Porter-lite stemming).

#### Tokenizer.get_word_variants

```python
Tokenizer.get_word_variants(self, word: str) -> List[str]
```

Get related words/variants for query expansion.

#### Tokenizer.add_word_mapping

```python
Tokenizer.add_word_mapping(self, word: str, variants: List[str]) -> None
```

Add a custom word mapping for query expansion.

### Dependencies

**Standard Library:**

- `re`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Set`
- ... and 1 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Observability

Metrics collection and progress tracking.

## Modules

- **observability.py**: Observability Module
- **progress.py**: Progress reporting infrastructure for long-running operations.
- **results.py**: Result Dataclasses for Cortical Text Processor


## observability.py

Observability Module
====================

Provides timing hooks, metrics collection, and trace context for monitoring
the Cortical Text Processor's performance and operations.

This module follows th...


### Classes

#### MetricsCollector

Collects and aggregates timing and count metrics for operations.

**Methods:**

- `record_timing`
- `record_count`
- `get_operation_stats`
- `get_all_stats`
- `get_trace`
- `reset`
- `enable`
- `disable`
- `trace_context`
- `get_summary`

#### TraceContext

Context for request tracing across operations.

**Methods:**

- `elapsed_ms`

### Functions

#### timed

```python
timed(operation_name: Optional[str] = None, include_args: bool = False)
```

Decorator for timing method calls and recording to metrics.

#### measure_time

```python
measure_time(func: Callable) -> Callable
```

Simple timing decorator that logs execution time.

#### get_global_metrics

```python
get_global_metrics() -> MetricsCollector
```

Get the global metrics collector instance.

#### enable_global_metrics

```python
enable_global_metrics() -> None
```

Enable global metrics collection.

#### disable_global_metrics

```python
disable_global_metrics() -> None
```

Disable global metrics collection.

#### reset_global_metrics

```python
reset_global_metrics() -> None
```

Reset global metrics.

#### MetricsCollector.record_timing

```python
MetricsCollector.record_timing(self, operation: str, duration_ms: float, trace_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None
```

Record a timing measurement for an operation.

#### MetricsCollector.record_count

```python
MetricsCollector.record_count(self, metric_name: str, count: int = 1) -> None
```

Record a simple count metric.

#### MetricsCollector.get_operation_stats

```python
MetricsCollector.get_operation_stats(self, operation: str) -> Dict[str, Any]
```

Get statistics for a specific operation.

#### MetricsCollector.get_all_stats

```python
MetricsCollector.get_all_stats(self) -> Dict[str, Dict[str, Any]]
```

Get statistics for all operations.

#### MetricsCollector.get_trace

```python
MetricsCollector.get_trace(self, trace_id: str) -> List[tuple]
```

Get all operations recorded for a trace ID.

#### MetricsCollector.reset

```python
MetricsCollector.reset(self) -> None
```

Clear all collected metrics.

#### MetricsCollector.enable

```python
MetricsCollector.enable(self) -> None
```

Enable metrics collection.

#### MetricsCollector.disable

```python
MetricsCollector.disable(self) -> None
```

Disable metrics collection.

#### MetricsCollector.trace_context

```python
MetricsCollector.trace_context(self, trace_id: str)
```

Context manager for tracing a block of operations.

#### MetricsCollector.get_summary

```python
MetricsCollector.get_summary(self) -> str
```

Get a human-readable summary of all metrics.

#### TraceContext.elapsed_ms

```python
TraceContext.elapsed_ms(self) -> float
```

Get elapsed time since trace started in milliseconds.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `contextlib.contextmanager`
- `functools`
- `logging`
- `time`
- ... and 5 more



## progress.py

Progress reporting infrastructure for long-running operations.

This module provides a flexible progress reporting system that supports:
- Console output with nice formatting
- Custom callbacks for in...


### Classes

#### ProgressReporter

Protocol for progress reporters.

**Methods:**

- `update`
- `complete`

#### ConsoleProgressReporter

Console-based progress reporter with nice formatting.

**Methods:**

- `update`
- `complete`

#### CallbackProgressReporter

Progress reporter that calls a custom callback function.

**Methods:**

- `update`
- `complete`

#### SilentProgressReporter

No-op progress reporter for silent operation.

**Methods:**

- `update`
- `complete`

#### MultiPhaseProgress

Helper for tracking progress across multiple sequential phases.

**Methods:**

- `start_phase`
- `update`
- `complete_phase`
- `overall_progress`

### Functions

#### ProgressReporter.update

```python
ProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Update progress for a specific phase.

#### ProgressReporter.complete

```python
ProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Mark a phase as complete.

#### ConsoleProgressReporter.update

```python
ConsoleProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Update progress display.

#### ConsoleProgressReporter.complete

```python
ConsoleProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Mark phase as complete and move to new line.

#### CallbackProgressReporter.update

```python
CallbackProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Call callback with progress update.

#### CallbackProgressReporter.complete

```python
CallbackProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Call callback with completion notification.

#### SilentProgressReporter.update

```python
SilentProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Do nothing.

#### SilentProgressReporter.complete

```python
SilentProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Do nothing.

#### MultiPhaseProgress.start_phase

```python
MultiPhaseProgress.start_phase(self, phase: str) -> None
```

Start a new phase.

#### MultiPhaseProgress.update

```python
MultiPhaseProgress.update(self, percent: float, message: Optional[str] = None) -> None
```

Update progress within current phase.

#### MultiPhaseProgress.complete_phase

```python
MultiPhaseProgress.complete_phase(self, message: Optional[str] = None) -> None
```

Mark current phase as complete.

#### MultiPhaseProgress.overall_progress

```python
MultiPhaseProgress.overall_progress(self) -> float
```

Get overall progress across all phases (0-100).

### Dependencies

**Standard Library:**

- `abc.ABC`
- `abc.abstractmethod`
- `sys`
- `time`
- `typing.Any`
- ... and 4 more



## results.py

Result Dataclasses for Cortical Text Processor
===============================================

Strongly-typed result containers for query operations that provide
IDE autocomplete and type checking su...


### Classes

#### DocumentMatch

A document search result with relevance score.

**Methods:**

- `to_dict`
- `to_tuple`
- `from_tuple`
- `from_dict`

#### PassageMatch

A passage retrieval result with text, location, and relevance score.

**Methods:**

- `to_dict`
- `to_tuple`
- `location`
- `length`
- `from_tuple`
- `from_dict`

#### QueryResult

Complete query result with matches and metadata.

**Methods:**

- `to_dict`
- `top_match`
- `match_count`
- `average_score`
- `from_dict`

### Functions

#### convert_document_matches

```python
convert_document_matches(results: List[tuple], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[DocumentMatch]
```

Convert list of (doc_id, score) tuples to DocumentMatch objects.

#### convert_passage_matches

```python
convert_passage_matches(results: List[tuple], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[PassageMatch]
```

Convert list of (doc_id, text, start, end, score) tuples to PassageMatch objects.

#### DocumentMatch.to_dict

```python
DocumentMatch.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### DocumentMatch.to_tuple

```python
DocumentMatch.to_tuple(self) -> tuple
```

Convert to tuple format (doc_id, score).

#### DocumentMatch.from_tuple

```python
DocumentMatch.from_tuple(cls, doc_id: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'DocumentMatch'
```

Create from tuple format (doc_id, score).

#### DocumentMatch.from_dict

```python
DocumentMatch.from_dict(cls, data: Dict[str, Any]) -> 'DocumentMatch'
```

Create from dictionary.

#### PassageMatch.to_dict

```python
PassageMatch.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### PassageMatch.to_tuple

```python
PassageMatch.to_tuple(self) -> tuple
```

Convert to tuple format (doc_id, text, start, end, score).

#### PassageMatch.location

```python
PassageMatch.location(self) -> str
```

Get citation-style location string.

#### PassageMatch.length

```python
PassageMatch.length(self) -> int
```

Get passage length in characters.

#### PassageMatch.from_tuple

```python
PassageMatch.from_tuple(cls, doc_id: str, text: str, start: int, end: int, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'PassageMatch'
```

Create from tuple format (doc_id, text, start, end, score).

#### PassageMatch.from_dict

```python
PassageMatch.from_dict(cls, data: Dict[str, Any]) -> 'PassageMatch'
```

Create from dictionary.

#### QueryResult.to_dict

```python
QueryResult.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary with nested match dicts.

#### QueryResult.top_match

```python
QueryResult.top_match(self) -> Union[DocumentMatch, PassageMatch, None]
```

Get the highest-scoring match.

#### QueryResult.match_count

```python
QueryResult.match_count(self) -> int
```

Get number of matches.

#### QueryResult.average_score

```python
QueryResult.average_score(self) -> float
```

Get average relevance score across all matches.

#### QueryResult.from_dict

```python
QueryResult.from_dict(cls, data: Dict[str, Any]) -> 'QueryResult'
```

Create from dictionary.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Any`
- `typing.Dict`
- ... and 3 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Persistence Layer

Save and load functionality for maintaining processor state.

## Modules

- **chunk_index.py**: Chunk-based indexing for git-compatible corpus storage.
- **persistence.py**: Persistence Module
- **state_storage.py**: Git-friendly State Storage Module


## chunk_index.py

Chunk-based indexing for git-compatible corpus storage.

This module provides append-only, time-stamped JSON chunks that can be
safely committed to git without merge conflicts. Each indexing session
c...


### Classes

#### ChunkOperation

A single operation in a chunk (add, modify, or delete).

**Methods:**

- `to_dict`
- `from_dict`

#### Chunk

A chunk containing operations from a single indexing session.

**Methods:**

- `to_dict`
- `from_dict`
- `get_filename`

#### ChunkWriter

Writes indexing session changes to timestamped JSON chunks.

**Methods:**

- `add_document`
- `modify_document`
- `delete_document`
- `has_operations`
- `save`

#### ChunkLoader

Loads and combines chunks to rebuild document state.

**Methods:**

- `get_chunk_files`
- `load_chunk`
- `load_all`
- `get_documents`
- `get_mtimes`
- `get_metadata`
- `get_chunks`
- `compute_hash`
- `is_cache_valid`
- `save_cache_hash`
- `get_stats`

#### ChunkCompactor

Compacts multiple chunk files into a single file.

**Methods:**

- `compact`

### Functions

#### get_changes_from_manifest

```python
get_changes_from_manifest(current_files: Dict[str, float], manifest: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]
```

Compare current files to manifest to find changes.

#### ChunkOperation.to_dict

```python
ChunkOperation.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### ChunkOperation.from_dict

```python
ChunkOperation.from_dict(cls, d: Dict[str, Any]) -> 'ChunkOperation'
```

Create from dictionary.

#### Chunk.to_dict

```python
Chunk.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### Chunk.from_dict

```python
Chunk.from_dict(cls, d: Dict[str, Any]) -> 'Chunk'
```

Create from dictionary.

#### Chunk.get_filename

```python
Chunk.get_filename(self) -> str
```

Generate filename for this chunk.

#### ChunkWriter.add_document

```python
ChunkWriter.add_document(self, doc_id: str, content: str, mtime: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
```

Record an add operation.

#### ChunkWriter.modify_document

```python
ChunkWriter.modify_document(self, doc_id: str, content: str, mtime: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
```

Record a modify operation.

#### ChunkWriter.delete_document

```python
ChunkWriter.delete_document(self, doc_id: str)
```

Record a delete operation.

#### ChunkWriter.has_operations

```python
ChunkWriter.has_operations(self) -> bool
```

Check if any operations were recorded.

#### ChunkWriter.save

```python
ChunkWriter.save(self, warn_size_kb: int = DEFAULT_WARN_SIZE_KB) -> Optional[Path]
```

Save chunk to file.

#### ChunkLoader.get_chunk_files

```python
ChunkLoader.get_chunk_files(self) -> List[Path]
```

Get all chunk files sorted by timestamp.

#### ChunkLoader.load_chunk

```python
ChunkLoader.load_chunk(self, filepath: Path) -> Chunk
```

Load a single chunk file.

#### ChunkLoader.load_all

```python
ChunkLoader.load_all(self) -> Dict[str, str]
```

Load all chunks and replay operations to get current document state.

#### ChunkLoader.get_documents

```python
ChunkLoader.get_documents(self) -> Dict[str, str]
```

Get loaded documents (calls load_all if needed).

#### ChunkLoader.get_mtimes

```python
ChunkLoader.get_mtimes(self) -> Dict[str, float]
```

Get document modification times.

#### ChunkLoader.get_metadata

```python
ChunkLoader.get_metadata(self) -> Dict[str, Dict[str, Any]]
```

Get document metadata (doc_type, headings, etc.).

#### ChunkLoader.get_chunks

```python
ChunkLoader.get_chunks(self) -> List[Chunk]
```

Get loaded chunks.

#### ChunkLoader.compute_hash

```python
ChunkLoader.compute_hash(self) -> str
```

Compute hash of current document state.

#### ChunkLoader.is_cache_valid

```python
ChunkLoader.is_cache_valid(self, cache_path: str, cache_hash_path: Optional[str] = None) -> bool
```

Check if pkl cache is valid for current chunk state.

#### ChunkLoader.save_cache_hash

```python
ChunkLoader.save_cache_hash(self, cache_path: str, cache_hash_path: Optional[str] = None)
```

Save current document hash for cache validation.

#### ChunkLoader.get_stats

```python
ChunkLoader.get_stats(self) -> Dict[str, Any]
```

Get statistics about loaded chunks.

#### ChunkCompactor.compact

```python
ChunkCompactor.compact(self, before: Optional[str] = None, keep_recent: int = 0, dry_run: bool = False) -> Dict[str, Any]
```

Compact chunks into a single chunk.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `hashlib`
- ... and 11 more



## persistence.py

Persistence Module
==================

Save and load functionality for the cortical processor.

Supports:
- Pickle serialization for full state
- JSON export for graph visualization
- Incremental upda...


### Classes

#### SignatureVerificationError

Raised when HMAC signature verification fails.

### Functions

#### save_processor

```python
save_processor(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, embeddings: Optional[Dict[str, list]] = None, semantic_relations: Optional[list] = None, metadata: Optional[Dict] = None, verbose: bool = True, format: str = 'pickle', signing_key: Optional[bytes] = None) -> None
```

Save processor state to a file.

#### load_processor

```python
load_processor(filepath: str, verbose: bool = True, format: Optional[str] = None, verify_key: Optional[bytes] = None) -> tuple
```

Load processor state from a file.

#### export_graph_json

```python
export_graph_json(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], layer_filter: Optional[CorticalLayer] = None, min_weight: float = 0.0, max_nodes: int = 500, verbose: bool = True) -> Dict
```

Export graph structure as JSON for visualization.

#### export_embeddings_json

```python
export_embeddings_json(filepath: str, embeddings: Dict[str, list], metadata: Optional[Dict] = None) -> None
```

Export embeddings as JSON.

#### load_embeddings_json

```python
load_embeddings_json(filepath: str) -> Dict[str, list]
```

Load embeddings from JSON.

#### export_semantic_relations_json

```python
export_semantic_relations_json(filepath: str, relations: list) -> None
```

Export semantic relations as JSON.

#### load_semantic_relations_json

```python
load_semantic_relations_json(filepath: str) -> list
```

Load semantic relations from JSON.

#### get_state_summary

```python
get_state_summary(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> Dict
```

Get a summary of the current processor state.

#### export_conceptnet_json

```python
export_conceptnet_json(filepath: str, layers: Dict[CorticalLayer, HierarchicalLayer], semantic_relations: Optional[list] = None, include_cross_layer: bool = True, include_typed_edges: bool = True, min_weight: float = 0.0, min_confidence: float = 0.0, max_nodes_per_layer: int = 100, verbose: bool = True) -> Dict[str, Any]
```

Export ConceptNet-style graph for visualization.

### Dependencies

**Standard Library:**

- `hashlib`
- `hmac`
- `json`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 9 more



## state_storage.py

Git-friendly State Storage Module
=================================

Replaces pickle-based persistence with JSON files that:
- Can be diff'd and reviewed in git
- Won't cause merge conflicts
- Support...


### Classes

#### StateManifest

Manifest file tracking state version and component checksums.

**Methods:**

- `to_dict`
- `from_dict`
- `update_checksum`

#### StateWriter

Writes processor state to git-friendly JSON files.

**Methods:**

- `save_layer`
- `save_documents`
- `save_semantic_relations`
- `save_embeddings`
- `save_manifest`
- `save_all`

#### StateLoader

Loads processor state from git-friendly JSON files.

**Methods:**

- `exists`
- `load_manifest`
- `validate_checksum`
- `load_layer`
- `load_documents`
- `load_semantic_relations`
- `load_embeddings`
- `load_all`
- `get_stats`

### Functions

#### migrate_pkl_to_json

```python
migrate_pkl_to_json(pkl_path: str, json_dir: str, verbose: bool = True) -> bool
```

Migrate a pickle file to git-friendly JSON format.

#### StateManifest.to_dict

```python
StateManifest.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for JSON serialization.

#### StateManifest.from_dict

```python
StateManifest.from_dict(cls, data: Dict[str, Any]) -> 'StateManifest'
```

Create manifest from dictionary.

#### StateManifest.update_checksum

```python
StateManifest.update_checksum(self, component: str, content: str) -> bool
```

Update checksum for a component.

#### StateWriter.save_layer

```python
StateWriter.save_layer(self, layer: HierarchicalLayer, force: bool = False) -> bool
```

Save a single layer to its JSON file.

#### StateWriter.save_documents

```python
StateWriter.save_documents(self, documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, force: bool = False) -> bool
```

Save documents and metadata.

#### StateWriter.save_semantic_relations

```python
StateWriter.save_semantic_relations(self, relations: List[Tuple], force: bool = False) -> bool
```

Save semantic relations.

#### StateWriter.save_embeddings

```python
StateWriter.save_embeddings(self, embeddings: Dict[str, List[float]], force: bool = False) -> bool
```

Save graph embeddings.

#### StateWriter.save_manifest

```python
StateWriter.save_manifest(self) -> None
```

Save the manifest file.

#### StateWriter.save_all

```python
StateWriter.save_all(self, layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], document_metadata: Optional[Dict[str, Dict[str, Any]]] = None, embeddings: Optional[Dict[str, List[float]]] = None, semantic_relations: Optional[List[Tuple]] = None, stale_computations: Optional[Set[str]] = None, force: bool = False, verbose: bool = True) -> Dict[str, bool]
```

Save all processor state.

#### StateLoader.exists

```python
StateLoader.exists(self) -> bool
```

Check if state directory exists and has manifest.

#### StateLoader.load_manifest

```python
StateLoader.load_manifest(self) -> StateManifest
```

Load the manifest file.

#### StateLoader.validate_checksum

```python
StateLoader.validate_checksum(self, component: str, filepath: Path) -> bool
```

Validate a component's checksum.

#### StateLoader.load_layer

```python
StateLoader.load_layer(self, level: int) -> HierarchicalLayer
```

Load a single layer.

#### StateLoader.load_documents

```python
StateLoader.load_documents(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]
```

Load documents and metadata.

#### StateLoader.load_semantic_relations

```python
StateLoader.load_semantic_relations(self) -> List[Tuple]
```

Load semantic relations.

#### StateLoader.load_embeddings

```python
StateLoader.load_embeddings(self) -> Dict[str, List[float]]
```

Load graph embeddings.

#### StateLoader.load_all

```python
StateLoader.load_all(self, validate: bool = True, verbose: bool = True) -> Tuple[Dict[CorticalLayer, HierarchicalLayer], Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, List[float]], List[Tuple], Dict[str, Any]]
```

Load all processor state.

#### StateLoader.get_stats

```python
StateLoader.get_stats(self) -> Dict[str, Any]
```

Get statistics about stored state without loading everything.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `hashlib`
- ... and 13 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Core Processor

The core processor orchestrates all text processing operations.

## Modules

- **__init__.py**: Cortical Text Processor - Main processor package.
- **compute.py**: Compute methods: analysis, clustering, embeddings, semantic extraction.
- **core.py**: Core processor functionality: initialization, staleness tracking, and layer management.
- **documents.py**: Document management: processing, adding, removing, and metadata handling.
- **introspection.py**: Introspection: state inspection, fingerprints, gaps, and summaries.
- **persistence_api.py**: Persistence API: save, load, export, and migration methods.


## __init__.py

Cortical Text Processor - Main processor package.

This package splits the monolithic processor.py into focused modules:
- core.py: Initialization, staleness tracking, layer management
- documents.py:...


### Classes

#### CorticalTextProcessor

Neocortex-inspired text processing system.

### Dependencies

**Standard Library:**

- `compute.ComputeMixin`
- `core.CoreMixin`
- `documents.DocumentsMixin`
- `introspection.IntrospectionMixin`
- `persistence_api.PersistenceMixin`
- ... and 1 more



## compute.py

Compute methods: analysis, clustering, embeddings, semantic extraction.

This module contains all methods that perform computational analysis on the corpus,
including PageRank, TF-IDF, clustering, and...


### Classes

#### ComputeMixin

Mixin providing computation functionality.

**Methods:**

- `recompute`
- `compute_all`
- `resume_from_checkpoint`
- `propagate_activation`
- `compute_importance`
- `compute_semantic_importance`
- `compute_hierarchical_importance`
- `compute_tfidf`
- `compute_bm25`
- `compute_document_connections`
- `compute_bigram_connections`
- `build_concept_clusters`
- `compute_clustering_quality`
- `compute_concept_connections`
- `extract_corpus_semantics`
- `extract_pattern_relations`
- `retrofit_connections`
- `compute_property_inheritance`
- `compute_property_similarity`
- `compute_graph_embeddings`
- `retrofit_embeddings`
- `embedding_similarity`
- `find_similar_by_embedding`

### Functions

#### ComputeMixin.recompute

```python
ComputeMixin.recompute(self, level: str = 'stale', verbose: bool = True) -> Dict[str, bool]
```

Recompute specified analysis levels.

#### ComputeMixin.compute_all

```python
ComputeMixin.compute_all(self, verbose: bool = True, build_concepts: bool = True, pagerank_method: str = 'standard', connection_strategy: str = 'document_overlap', cluster_strictness: float = 1.0, bridge_weight: float = 0.0, progress_callback: Optional[ProgressReporter] = None, show_progress: bool = False, checkpoint_dir: Optional[str] = None, resume: bool = False) -> Dict[str, Any]
```

Run all computation steps.

#### ComputeMixin.resume_from_checkpoint

```python
ComputeMixin.resume_from_checkpoint(cls, checkpoint_dir: str, config: Optional['CorticalConfig'] = None, verbose: bool = True) -> 'CorticalTextProcessor'
```

Resume processing from a checkpoint directory.

#### ComputeMixin.propagate_activation

```python
ComputeMixin.propagate_activation(self, iterations: int = 3, decay: float = 0.8, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_importance

```python
ComputeMixin.compute_importance(self, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_semantic_importance

```python
ComputeMixin.compute_semantic_importance(self, relation_weights: Optional[Dict[str, float]] = None, verbose: bool = True) -> Dict[str, Any]
```

Compute PageRank with semantic relation weighting.

#### ComputeMixin.compute_hierarchical_importance

```python
ComputeMixin.compute_hierarchical_importance(self, layer_iterations: int = 10, global_iterations: int = 5, cross_layer_damping: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]
```

Compute PageRank with cross-layer propagation.

#### ComputeMixin.compute_tfidf

```python
ComputeMixin.compute_tfidf(self, verbose: bool = True) -> None
```

Compute document relevance scores using the configured algorithm.

#### ComputeMixin.compute_bm25

```python
ComputeMixin.compute_bm25(self, k1: float = None, b: float = None, verbose: bool = True) -> None
```

Compute BM25 scores for document relevance ranking.

#### ComputeMixin.compute_document_connections

```python
ComputeMixin.compute_document_connections(self, min_shared_terms: int = 3, verbose: bool = True) -> None
```

None

#### ComputeMixin.compute_bigram_connections

```python
ComputeMixin.compute_bigram_connections(self, min_shared_docs: int = 1, component_weight: float = 0.5, chain_weight: float = 0.7, cooccurrence_weight: float = 0.3, max_bigrams_per_term: int = 100, max_bigrams_per_doc: int = 500, max_connections_per_bigram: int = 50, verbose: bool = True) -> Dict[str, Any]
```

Build lateral connections between bigrams based on shared components and co-occurrence.

#### ComputeMixin.build_concept_clusters

```python
ComputeMixin.build_concept_clusters(self, min_cluster_size: Optional[int] = None, clustering_method: str = 'louvain', cluster_strictness: Optional[float] = None, bridge_weight: float = 0.0, resolution: Optional[float] = None, verbose: bool = True) -> Dict[int, List[str]]
```

Build concept clusters from token layer.

#### ComputeMixin.compute_clustering_quality

```python
ComputeMixin.compute_clustering_quality(self, sample_size: int = 500) -> Dict[str, Any]
```

Compute clustering quality metrics for the concept layer.

#### ComputeMixin.compute_concept_connections

```python
ComputeMixin.compute_concept_connections(self, use_semantics: bool = True, min_shared_docs: int = 1, min_jaccard: float = 0.1, use_member_semantics: bool = False, use_embedding_similarity: bool = False, embedding_threshold: float = 0.3, verbose: bool = True) -> Dict[str, Any]
```

Build lateral connections between concepts based on document overlap and semantics.

#### ComputeMixin.extract_corpus_semantics

```python
ComputeMixin.extract_corpus_semantics(self, use_pattern_extraction: bool = True, min_pattern_confidence: float = 0.6, max_similarity_pairs: int = 100000, min_context_keys: int = 3, verbose: bool = True) -> int
```

Extract semantic relations from the corpus.

#### ComputeMixin.extract_pattern_relations

```python
ComputeMixin.extract_pattern_relations(self, min_confidence: float = 0.6, verbose: bool = True) -> List[Tuple[str, str, str, float]]
```

Extract semantic relations using pattern matching only.

#### ComputeMixin.retrofit_connections

```python
ComputeMixin.retrofit_connections(self, iterations: int = 10, alpha: float = 0.3, verbose: bool = True) -> Dict
```

None

#### ComputeMixin.compute_property_inheritance

```python
ComputeMixin.compute_property_inheritance(self, decay_factor: float = 0.7, max_depth: int = 5, apply_to_connections: bool = True, boost_factor: float = 0.3, verbose: bool = True) -> Dict[str, Any]
```

Compute property inheritance based on IsA hierarchy.

#### ComputeMixin.compute_property_similarity

```python
ComputeMixin.compute_property_similarity(self, term1: str, term2: str) -> float
```

Compute similarity between terms based on shared properties.

#### ComputeMixin.compute_graph_embeddings

```python
ComputeMixin.compute_graph_embeddings(self, dimensions: int = 64, method: str = 'fast', max_terms: Optional[int] = None, verbose: bool = True) -> Dict
```

Compute graph embeddings for tokens.

#### ComputeMixin.retrofit_embeddings

```python
ComputeMixin.retrofit_embeddings(self, iterations: int = 10, alpha: float = 0.4, verbose: bool = True) -> Dict
```

None

#### ComputeMixin.embedding_similarity

```python
ComputeMixin.embedding_similarity(self, term1: str, term2: str) -> float
```

None

#### ComputeMixin.find_similar_by_embedding

```python
ComputeMixin.find_similar_by_embedding(self, term: str, top_n: int = 10) -> List[Tuple[str, float]]
```

None

### Dependencies

**Standard Library:**

- `datetime.datetime`
- `json`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- ... and 11 more

**Local Imports:**

- `.analysis`
- `.embeddings`
- `.semantics`



## core.py

Core processor functionality: initialization, staleness tracking, and layer management.

This module contains the base class definition and core infrastructure that all
other processor mixins depend o...


### Classes

#### CoreMixin

Core mixin providing initialization and staleness tracking.

**Methods:**

- `is_stale`
- `get_stale_computations`
- `get_layer`
- `get_metrics`
- `get_metrics_summary`
- `reset_metrics`
- `enable_metrics`
- `disable_metrics`
- `record_metric`

### Functions

#### CoreMixin.is_stale

```python
CoreMixin.is_stale(self, computation_type: str) -> bool
```

Check if a specific computation is stale.

#### CoreMixin.get_stale_computations

```python
CoreMixin.get_stale_computations(self) -> set
```

Get the set of computations that are currently stale.

#### CoreMixin.get_layer

```python
CoreMixin.get_layer(self, layer: CorticalLayer) -> HierarchicalLayer
```

Get a specific layer by enum.

#### CoreMixin.get_metrics

```python
CoreMixin.get_metrics(self) -> Dict[str, Dict[str, Any]]
```

Get all collected metrics.

#### CoreMixin.get_metrics_summary

```python
CoreMixin.get_metrics_summary(self) -> str
```

Get a human-readable summary of all metrics.

#### CoreMixin.reset_metrics

```python
CoreMixin.reset_metrics(self) -> None
```

Clear all collected metrics.

#### CoreMixin.enable_metrics

```python
CoreMixin.enable_metrics(self) -> None
```

Enable metrics collection.

#### CoreMixin.disable_metrics

```python
CoreMixin.disable_metrics(self) -> None
```

Disable metrics collection.

#### CoreMixin.record_metric

```python
CoreMixin.record_metric(self, metric_name: str, count: int = 1) -> None
```

Record a custom count metric.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `logging`
- `minicolumn.Minicolumn`
- ... and 5 more



## documents.py

Document management: processing, adding, removing, and metadata handling.

This module contains all methods related to managing documents in the corpus.


### Classes

#### DocumentsMixin

Mixin providing document management functionality.

**Methods:**

- `process_document`
- `set_document_metadata`
- `get_document_metadata`
- `get_all_document_metadata`
- `add_document_incremental`
- `add_documents_batch`
- `remove_document`
- `remove_documents_batch`

### Functions

#### DocumentsMixin.process_document

```python
DocumentsMixin.process_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, int]
```

Process a document and add it to the corpus.

#### DocumentsMixin.set_document_metadata

```python
DocumentsMixin.set_document_metadata(self, doc_id: str, **kwargs) -> None
```

Set or update metadata for a document.

#### DocumentsMixin.get_document_metadata

```python
DocumentsMixin.get_document_metadata(self, doc_id: str) -> Dict[str, Any]
```

Get metadata for a document.

#### DocumentsMixin.get_all_document_metadata

```python
DocumentsMixin.get_all_document_metadata(self) -> Dict[str, Dict[str, Any]]
```

Get metadata for all documents.

#### DocumentsMixin.add_document_incremental

```python
DocumentsMixin.add_document_incremental(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, recompute: str = 'tfidf') -> Dict[str, int]
```

Add a document with selective recomputation for efficiency.

#### DocumentsMixin.add_documents_batch

```python
DocumentsMixin.add_documents_batch(self, documents: List[Tuple[str, str, Optional[Dict[str, Any]]]], recompute: str = 'full', verbose: bool = True) -> Dict[str, Any]
```

Add multiple documents with a single recomputation.

#### DocumentsMixin.remove_document

```python
DocumentsMixin.remove_document(self, doc_id: str, verbose: bool = False) -> Dict[str, Any]
```

Remove a document from the corpus.

#### DocumentsMixin.remove_documents_batch

```python
DocumentsMixin.remove_documents_batch(self, doc_ids: List[str], recompute: str = 'none', verbose: bool = True) -> Dict[str, Any]
```

Remove multiple documents efficiently with single recomputation.

### Dependencies

**Standard Library:**

- `copy`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- `typing.Any`
- ... and 4 more



## introspection.py

Introspection: state inspection, fingerprints, gaps, and summaries.

This module contains methods for examining the processor state and
comparing texts/documents.


### Classes

#### IntrospectionMixin

Mixin providing introspection functionality.

**Methods:**

- `get_document_signature`
- `get_corpus_summary`
- `analyze_knowledge_gaps`
- `detect_anomalies`
- `get_fingerprint`
- `compare_fingerprints`
- `explain_fingerprint`
- `explain_similarity`
- `find_similar_texts`
- `compare_with`
- `compare_documents`
- `what_changed`
- `summarize_document`
- `detect_patterns`
- `detect_patterns_in_corpus`
- `get_pattern_summary`
- `get_corpus_pattern_statistics`
- `format_pattern_report`
- `list_available_patterns`
- `list_pattern_categories`

### Functions

#### IntrospectionMixin.get_document_signature

```python
IntrospectionMixin.get_document_signature(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]
```

Get the top-n TF-IDF terms for a document.

#### IntrospectionMixin.get_corpus_summary

```python
IntrospectionMixin.get_corpus_summary(self) -> Dict
```

Get summary statistics about the corpus.

#### IntrospectionMixin.analyze_knowledge_gaps

```python
IntrospectionMixin.analyze_knowledge_gaps(self) -> Dict
```

Analyze the corpus for knowledge gaps.

#### IntrospectionMixin.detect_anomalies

```python
IntrospectionMixin.detect_anomalies(self, threshold: float = 0.3) -> List[Dict]
```

Detect anomalous patterns in the corpus.

#### IntrospectionMixin.get_fingerprint

```python
IntrospectionMixin.get_fingerprint(self, text: str, top_n: int = 20) -> Dict
```

Compute the semantic fingerprint of a text.

#### IntrospectionMixin.compare_fingerprints

```python
IntrospectionMixin.compare_fingerprints(self, fp1: Dict, fp2: Dict) -> Dict
```

Compare two fingerprints and compute similarity metrics.

#### IntrospectionMixin.explain_fingerprint

```python
IntrospectionMixin.explain_fingerprint(self, fp: Dict, top_n: int = 10) -> Dict
```

Generate a human-readable explanation of a fingerprint.

#### IntrospectionMixin.explain_similarity

```python
IntrospectionMixin.explain_similarity(self, fp1: Dict, fp2: Dict) -> str
```

Generate a human-readable explanation of fingerprint similarity.

#### IntrospectionMixin.find_similar_texts

```python
IntrospectionMixin.find_similar_texts(self, text: str, candidates: List[Tuple[str, str]], top_n: int = 5) -> List[Tuple[str, float, Dict]]
```

Find texts most similar to the given text.

#### IntrospectionMixin.compare_with

```python
IntrospectionMixin.compare_with(self, other: 'CorticalTextProcessor', top_movers: int = 20, min_pagerank_delta: float = 0.0001) -> 'diff_module.SemanticDiff'
```

Compare this processor state with another to find semantic differences.

#### IntrospectionMixin.compare_documents

```python
IntrospectionMixin.compare_documents(self, doc_id_1: str, doc_id_2: str) -> Dict
```

Compare two documents within this corpus.

#### IntrospectionMixin.what_changed

```python
IntrospectionMixin.what_changed(self, old_content: str, new_content: str) -> Dict
```

Compare two text contents to show what changed semantically.

#### IntrospectionMixin.summarize_document

```python
IntrospectionMixin.summarize_document(self, doc_id: str, num_sentences: int = 3) -> str
```

Generate a summary of a document using extractive summarization.

#### IntrospectionMixin.detect_patterns

```python
IntrospectionMixin.detect_patterns(self, doc_id: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]
```

Detect programming patterns in a specific document.

#### IntrospectionMixin.detect_patterns_in_corpus

```python
IntrospectionMixin.detect_patterns_in_corpus(self, patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]
```

Detect patterns across all documents in the corpus.

#### IntrospectionMixin.get_pattern_summary

```python
IntrospectionMixin.get_pattern_summary(self, doc_id: str) -> Dict[str, int]
```

Get a summary of pattern occurrences in a document.

#### IntrospectionMixin.get_corpus_pattern_statistics

```python
IntrospectionMixin.get_corpus_pattern_statistics(self) -> Dict[str, Any]
```

Get pattern statistics across the entire corpus.

#### IntrospectionMixin.format_pattern_report

```python
IntrospectionMixin.format_pattern_report(self, doc_id: str, show_lines: bool = False) -> str
```

Format pattern detection results as a human-readable report.

#### IntrospectionMixin.list_available_patterns

```python
IntrospectionMixin.list_available_patterns(self) -> List[str]
```

List all available pattern names that can be detected.

#### IntrospectionMixin.list_pattern_categories

```python
IntrospectionMixin.list_pattern_categories(self) -> List[str]
```

List all pattern categories.

### Dependencies

**Standard Library:**

- `layers.CorticalLayer`
- `logging`
- `re`
- `typing.Any`
- `typing.Dict`
- ... and 4 more

**Local Imports:**

- `.fingerprint`
- `.gaps`
- `.patterns`
- `.persistence`



## persistence_api.py

Persistence API: save, load, export, and migration methods.

This module contains all methods related to saving and loading processor state.


### Classes

#### PersistenceMixin

Mixin providing persistence functionality.

**Methods:**

- `save`
- `load`
- `save_json`
- `load_json`
- `migrate_to_json`
- `export_graph`
- `export_conceptnet_json`

### Functions

#### PersistenceMixin.save

```python
PersistenceMixin.save(self, filepath: str, verbose: bool = True, signing_key: Optional[bytes] = None) -> None
```

Save processor state to a file.

#### PersistenceMixin.load

```python
PersistenceMixin.load(cls, filepath: str, verbose: bool = True, verify_key: Optional[bytes] = None) -> 'CorticalTextProcessor'
```

Load processor state from a file.

#### PersistenceMixin.save_json

```python
PersistenceMixin.save_json(self, state_dir: str, force: bool = False, verbose: bool = True) -> Dict[str, bool]
```

Save processor state to git-friendly JSON format.

#### PersistenceMixin.load_json

```python
PersistenceMixin.load_json(cls, state_dir: str, config: Optional[CorticalConfig] = None, verbose: bool = True) -> 'CorticalTextProcessor'
```

Load processor from git-friendly JSON format.

#### PersistenceMixin.migrate_to_json

```python
PersistenceMixin.migrate_to_json(self, pkl_path: str, json_dir: str, verbose: bool = True) -> bool
```

Migrate existing pickle file to git-friendly JSON format.

#### PersistenceMixin.export_graph

```python
PersistenceMixin.export_graph(self, filepath: str, layer: Optional[CorticalLayer] = None, max_nodes: int = 500) -> Dict
```

Export graph to JSON for visualization.

#### PersistenceMixin.export_conceptnet_json

```python
PersistenceMixin.export_conceptnet_json(self, filepath: str, include_cross_layer: bool = True, include_typed_edges: bool = True, min_weight: float = 0.0, min_confidence: float = 0.0, max_nodes_per_layer: int = 100, verbose: bool = True) -> Dict[str, Any]
```

Export ConceptNet-style graph for visualization.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `layers.CorticalLayer`
- `logging`
- `observability.timed`
- `typing.Any`
- ... and 3 more

**Local Imports:**

- `.persistence`
- `.state_storage`



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Search & Retrieval

Search and retrieval components for finding relevant documents and passages.

## Modules

- **__init__.py**: Query Module
- **chunking.py**: Chunking Module
- **definitions.py**: Definition Search Module
- **expansion.py**: Query Expansion Module
- **intent.py**: Intent Query Module
- **passages.py**: Passage Retrieval Module
- **ranking.py**: Ranking Module
- **search.py**: Document Search Module


## __init__.py

Query Module
============

Query expansion and search functionality.

This package provides methods for expanding queries using lateral connections,
concept clusters, and word variants, then searching...


### Dependencies

**Standard Library:**

- `analogy.complete_analogy`
- `analogy.complete_analogy_simple`
- `analogy.find_relation_between`
- `analogy.find_terms_with_relation`
- `chunking.CODE_BOUNDARY_PATTERN`
- ... and 49 more



## chunking.py

Chunking Module
==============

Functions for splitting documents into chunks for passage retrieval.

This module provides:
- Fixed-size text chunking with overlap
- Code-aware chunking aligned to sem...


### Functions

#### create_chunks

```python
create_chunks(text: str, chunk_size: int = 512, overlap: int = 128) -> List[Tuple[str, int, int]]
```

Split text into overlapping chunks.

#### find_code_boundaries

```python
find_code_boundaries(text: str) -> List[int]
```

Find semantic boundaries in code (class/function definitions, decorators).

#### create_code_aware_chunks

```python
create_code_aware_chunks(text: str, target_size: int = 512, min_size: int = 100, max_size: int = 1024) -> List[Tuple[str, int, int]]
```

Create chunks aligned to code structure boundaries.

#### is_code_file

```python
is_code_file(doc_id: str) -> bool
```

Determine if a document is a code file based on its path/extension.

#### precompute_term_cols

```python
precompute_term_cols(query_terms: Dict[str, float], layer0: HierarchicalLayer) -> Dict[str, 'Minicolumn']
```

Pre-compute minicolumn lookups for query terms.

#### score_chunk_fast

```python
score_chunk_fast(chunk_tokens: List[str], query_terms: Dict[str, float], term_cols: Dict[str, 'Minicolumn'], doc_id: Optional[str] = None) -> float
```

Fast chunk scoring using pre-computed minicolumn lookups.

#### score_chunk

```python
score_chunk(chunk_text: str, query_terms: Dict[str, float], layer0: HierarchicalLayer, tokenizer: Tokenizer, doc_id: Optional[str] = None) -> float
```

Score a chunk against query terms using TF-IDF.

### Dependencies

**Standard Library:**

- `layers.HierarchicalLayer`
- `re`
- `tokenizer.Tokenizer`
- `typing.Dict`
- `typing.List`
- ... and 3 more



## definitions.py

Definition Search Module
========================

Functions for finding and boosting code definitions (classes, functions, methods).

This module handles:
- Detection of definition-seeking queries ("...


### Classes

#### DefinitionQuery

Info about a definition-seeking query.

### Functions

#### is_definition_query

```python
is_definition_query(query_text: str) -> Tuple[bool, Optional[str], Optional[str]]
```

Detect if a query is looking for a code definition.

#### find_definition_in_text

```python
find_definition_in_text(text: str, identifier: str, def_type: str, context_chars: int = 500) -> Optional[Tuple[str, int, int]]
```

Find a definition in source text and extract surrounding context.

#### find_definition_passages

```python
find_definition_passages(query_text: str, documents: Dict[str, str], context_chars: int = 500, boost: float = DEFINITION_BOOST) -> List[Tuple[str, str, int, int, float]]
```

Find definition passages for a definition query.

#### detect_definition_query

```python
detect_definition_query(query_text: str) -> DefinitionQuery
```

Detect if a query is searching for a code definition.

#### apply_definition_boost

```python
apply_definition_boost(passages: List[Tuple[str, str, int, int, float]], query_text: str, boost_factor: float = 3.0) -> List[Tuple[str, str, int, int, float]]
```

Boost passages that contain actual code definitions matching the query.

#### is_test_file

```python
is_test_file(doc_id: str) -> bool
```

Detect if a document ID represents a test file.

#### boost_definition_documents

```python
boost_definition_documents(doc_results: List[Tuple[str, float]], query_text: str, documents: Dict[str, str], boost_factor: float = 2.0, test_with_definition_penalty: float = 0.5, test_without_definition_penalty: float = 0.7) -> List[Tuple[str, float]]
```

Boost documents that contain the actual definition being searched for.

### Dependencies

**Standard Library:**

- `re`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 2 more



## expansion.py

Query Expansion Module
=====================

Functions for expanding query terms using lateral connections,
semantic relations, and code concept synonyms.

This module provides:
- Basic query expansi...


### Functions

#### score_relation_path

```python
score_relation_path(path: List[str]) -> float
```

Score a relation path by its semantic coherence.

#### expand_query

```python
expand_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, max_expansions: int = 10, use_lateral: bool = True, use_concepts: bool = True, use_variants: bool = True, use_code_concepts: bool = False, filter_code_stop_words: bool = False, tfidf_weight: float = 0.7, max_expansion_weight: float = 2.0) -> Dict[str, float]
```

Expand a query using lateral connections and concept clusters.

#### expand_query_semantic

```python
expand_query_semantic(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, semantic_relations: List[Tuple[str, str, str, float]], max_expansions: int = 10) -> Dict[str, float]
```

Expand query using semantic relations extracted from corpus.

#### expand_query_multihop

```python
expand_query_multihop(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, semantic_relations: List[Tuple[str, str, str, float]], max_hops: int = 2, max_expansions: int = 15, decay_factor: float = 0.5, min_path_score: float = 0.2) -> Dict[str, float]
```

Expand query using multi-hop semantic inference.

#### get_expanded_query_terms

```python
get_expanded_query_terms(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, max_expansions: int = 5, semantic_discount: float = 0.8, filter_code_stop_words: bool = False) -> Dict[str, float]
```

Get expanded query terms with optional semantic expansion.

### Dependencies

**Standard Library:**

- `code_concepts.expand_code_concepts`
- `collections.defaultdict`
- `config.DEFAULT_CHAIN_VALIDITY`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 6 more



## intent.py

Intent Query Module
==================

Intent-based query understanding for natural language code search.

This module handles:
- Parsing natural language queries to extract intent (where, how, what,...


### Classes

#### ParsedIntent

Structured representation of a parsed query intent.

### Functions

#### parse_intent_query

```python
parse_intent_query(query_text: str) -> ParsedIntent
```

Parse a natural language query to extract intent and searchable terms.

#### search_by_intent

```python
search_by_intent(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: 'Tokenizer', top_n: int = 5) -> List[Tuple[str, float, ParsedIntent]]
```

Search the corpus using intent-based query understanding.

### Dependencies

**Standard Library:**

- `code_concepts.get_related_terms`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `typing.Dict`
- ... and 4 more



## passages.py

Passage Retrieval Module
========================

Functions for retrieving relevant passages from documents.

This module provides:
- Passage retrieval for RAG systems
- Batch passage retrieval
- Int...


### Functions

#### find_passages_for_query

```python
find_passages_for_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, use_expansion: bool = True, doc_filter: Optional[List[str]] = None, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, use_definition_search: bool = True, definition_boost: float = DEFINITION_BOOST, apply_doc_boost: bool = True, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, auto_detect_intent: bool = True, prefer_docs: bool = False, custom_boosts: Optional[Dict[str, float]] = None, use_code_aware_chunks: bool = True, filter_code_stop_words: bool = True, test_file_penalty: float = 0.8) -> List[Tuple[str, str, int, int, float]]
```

Find text passages most relevant to a query.

#### find_documents_batch

```python
find_documents_batch(queries: List[str], layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[List[Tuple[str, float]]]
```

Find documents for multiple queries efficiently.

#### find_passages_batch

```python
find_passages_batch(queries: List[str], layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, use_expansion: bool = True, doc_filter: Optional[List[str]] = None, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[List[Tuple[str, str, int, int, float]]]
```

Find passages for multiple queries efficiently.

### Dependencies

**Standard Library:**

- `chunking.CODE_BOUNDARY_PATTERN`
- `chunking.create_chunks`
- `chunking.create_code_aware_chunks`
- `chunking.find_code_boundaries`
- `chunking.is_code_file`
- ... and 18 more



## ranking.py

Ranking Module
=============

Multi-stage ranking and document type boosting for search results.

This module provides:
- Document type boosting (docs, code, tests)
- Conceptual vs implementation quer...


### Functions

#### is_conceptual_query

```python
is_conceptual_query(query_text: str) -> bool
```

Determine if a query is conceptual (should boost documentation).

#### get_doc_type_boost

```python
get_doc_type_boost(doc_id: str, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, custom_boosts: Optional[Dict[str, float]] = None) -> float
```

Get the boost factor for a document based on its type.

#### apply_doc_type_boost

```python
apply_doc_type_boost(results: List[Tuple[str, float]], doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, boost_docs: bool = True, custom_boosts: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]
```

Apply document type boosting to search results.

#### find_documents_with_boost

```python
find_documents_with_boost(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None, auto_detect_intent: bool = True, prefer_docs: bool = False, custom_boosts: Optional[Dict[str, float]] = None, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, float]]
```

Find documents with optional document-type boosting.

#### find_relevant_concepts

```python
find_relevant_concepts(query_terms: Dict[str, float], layers: Dict[CorticalLayer, HierarchicalLayer], top_n: int = 5) -> List[Tuple[str, float, set]]
```

Stage 1: Find concepts relevant to query terms.

#### multi_stage_rank

```python
multi_stage_rank(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, documents: Dict[str, str], top_n: int = 5, chunk_size: int = 512, overlap: int = 128, concept_boost: float = 0.3, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, str, int, int, float, Dict[str, float]]]
```

Multi-stage ranking pipeline for improved RAG performance.

#### multi_stage_rank_documents

```python
multi_stage_rank_documents(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, concept_boost: float = 0.3, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True) -> List[Tuple[str, float, Dict[str, float]]]
```

Multi-stage ranking for documents (without chunk scoring).

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `constants.CONCEPTUAL_KEYWORDS`
- `constants.DOC_TYPE_BOOSTS`
- `constants.IMPLEMENTATION_KEYWORDS`
- `expansion.get_expanded_query_terms`
- ... and 9 more



## search.py

Document Search Module
=====================

Functions for searching and retrieving documents from the corpus.

This module provides:
- Basic document search using TF-IDF scoring
- Fast document sear...


### Functions

#### find_documents_for_query

```python
find_documents_for_query(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None, use_semantic: bool = True, doc_name_boost: float = 2.0, filter_code_stop_words: bool = True, test_file_penalty: float = 0.8) -> List[Tuple[str, float]]
```

Find documents most relevant to a query using TF-IDF and optional expansion.

#### fast_find_documents

```python
fast_find_documents(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, candidate_multiplier: int = 3, use_code_concepts: bool = True, doc_name_boost: float = 2.0) -> List[Tuple[str, float]]
```

Fast document search using candidate filtering.

#### build_document_index

```python
build_document_index(layers: Dict[CorticalLayer, HierarchicalLayer]) -> Dict[str, Dict[str, float]]
```

Build an optimized inverted index for fast querying.

#### search_with_index

```python
search_with_index(query_text: str, index: Dict[str, Dict[str, float]], tokenizer: Tokenizer, top_n: int = 5) -> List[Tuple[str, float]]
```

Search using a pre-built inverted index.

#### query_with_spreading_activation

```python
query_with_spreading_activation(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 10, max_expansions: int = 8) -> List[Tuple[str, float]]
```

Query with automatic expansion using spreading activation.

#### find_related_documents

```python
find_related_documents(doc_id: str, layers: Dict[CorticalLayer, HierarchicalLayer]) -> List[Tuple[str, float]]
```

Find documents related to a given document via lateral connections.

#### graph_boosted_search

```python
graph_boosted_search(query_text: str, layers: Dict[CorticalLayer, HierarchicalLayer], tokenizer: Tokenizer, top_n: int = 5, pagerank_weight: float = 0.3, proximity_weight: float = 0.2, use_expansion: bool = True, semantic_relations: Optional[List[Tuple[str, str, str, float]]] = None) -> List[Tuple[str, float]]
```

Graph-Boosted BM25 (GB-BM25): Hybrid scoring combining BM25 with graph signals.

### Dependencies

**Standard Library:**

- `code_concepts.get_related_terms`
- `collections.defaultdict`
- `expansion.expand_query`
- `expansion.get_expanded_query_terms`
- `layers.CorticalLayer`
- ... and 6 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Utilities

Utility modules supporting various features.

## Modules

- **cli_wrapper.py**: CLI wrapper framework for collecting context and triggering actions.
- **code_concepts.py**: Code Concepts Module
- **diff.py**: Semantic Diff Module
- **fingerprint.py**: Fingerprint Module
- **fluent.py**: Fluent API for CorticalTextProcessor - chainable method interface.
- **gaps.py**: Gaps Module
- **mcp_server.py**: MCP (Model Context Protocol) Server for Cortical Text Processor.
- **patterns.py**: Code Pattern Detection Module


## cli_wrapper.py

CLI wrapper framework for collecting context and triggering actions.

Design philosophy: QUIET BY DEFAULT, POWERFUL WHEN NEEDED.

Most of the time you just want to run a command and check if it worked...


### Classes

#### GitContext

Git repository context information.

**Methods:**

- `collect`
- `to_dict`

#### ExecutionContext

Complete context for a CLI command execution.

**Methods:**

- `to_dict`
- `to_json`
- `summary`

#### HookType

Types of hooks that can be registered.

#### HookRegistry

Registry for CLI execution hooks.

**Methods:**

- `register`
- `register_pre`
- `register_post`
- `register_success`
- `register_error`
- `get_hooks`
- `trigger`

#### CLIWrapper

Wrapper for CLI command execution with context collection and hooks.

**Methods:**

- `run`
- `on_success`
- `on_error`
- `on_complete`

#### TaskCompletionManager

Manager for task completion triggers and context window management.

**Methods:**

- `on_task_complete`
- `on_any_complete`
- `handle_completion`
- `get_session_summary`
- `should_trigger_reindex`

#### ContextWindowManager

Manages context window state based on CLI execution history.

**Methods:**

- `add_execution`
- `add_file_read`
- `get_recent_files`
- `get_context_summary`
- `suggest_pruning`

#### Session

Track a sequence of commands as a session.

**Methods:**

- `run`
- `should_reindex`
- `summary`
- `results`
- `success_rate`
- `all_passed`
- `modified_files`

#### TaskCheckpoint

Save/restore context state when switching between tasks.

**Methods:**

- `save`
- `load`
- `list_tasks`
- `delete`
- `summarize`

### Functions

#### create_wrapper_with_completion_manager

```python
create_wrapper_with_completion_manager() -> Tuple[CLIWrapper, TaskCompletionManager]
```

Create a CLIWrapper with an attached TaskCompletionManager.

#### run_with_context

```python
run_with_context(command: Union[str, List[str]], **kwargs) -> ExecutionContext
```

Convenience function to run a command with full context collection.

#### run

```python
run(command: Union[str, List[str]], git: bool = False, timeout: Optional[float] = None, cwd: Optional[str] = None) -> ExecutionContext
```

Run a command. That's it.

#### test_then_commit

```python
test_then_commit(test_cmd: Union[str, List[str]] = 'python -m unittest discover -s tests', message: str = 'Update', add_all: bool = True) -> Tuple[bool, List[ExecutionContext]]
```

Run tests, commit only if they pass.

#### commit_and_push

```python
commit_and_push(message: str, add_all: bool = True, branch: Optional[str] = None) -> Tuple[bool, List[ExecutionContext]]
```

Add, commit, and push in one go.

#### sync_with_main

```python
sync_with_main(main_branch: str = 'main') -> Tuple[bool, List[ExecutionContext]]
```

Fetch and rebase current branch on main.

#### GitContext.collect

```python
GitContext.collect(cls, cwd: Optional[str] = None) -> 'GitContext'
```

Collect git context from current directory.

#### GitContext.to_dict

```python
GitContext.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### ExecutionContext.to_dict

```python
ExecutionContext.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for serialization.

#### ExecutionContext.to_json

```python
ExecutionContext.to_json(self, indent: int = 2) -> str
```

Convert to JSON string.

#### ExecutionContext.summary

```python
ExecutionContext.summary(self) -> str
```

Return a concise summary string.

#### HookRegistry.register

```python
HookRegistry.register(self, hook_type: HookType, callback: HookCallback, pattern: Optional[str] = None) -> None
```

Register a hook callback.

#### HookRegistry.register_pre

```python
HookRegistry.register_pre(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for pre-execution hooks.

#### HookRegistry.register_post

```python
HookRegistry.register_post(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for post-execution hooks.

#### HookRegistry.register_success

```python
HookRegistry.register_success(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for success hooks.

#### HookRegistry.register_error

```python
HookRegistry.register_error(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for error hooks.

#### HookRegistry.get_hooks

```python
HookRegistry.get_hooks(self, hook_type: HookType, command: List[str]) -> List[HookCallback]
```

Get all hooks that should be triggered for a command.

#### HookRegistry.trigger

```python
HookRegistry.trigger(self, hook_type: HookType, context: ExecutionContext) -> None
```

Trigger all matching hooks.

#### CLIWrapper.run

```python
CLIWrapper.run(self, command: Union[str, List[str]], cwd: Optional[str] = None, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None, **kwargs) -> ExecutionContext
```

Execute a command with context collection and hooks.

#### CLIWrapper.on_success

```python
CLIWrapper.on_success(self, pattern: Optional[str] = None)
```

Decorator to register a success hook.

#### CLIWrapper.on_error

```python
CLIWrapper.on_error(self, pattern: Optional[str] = None)
```

Decorator to register an error hook.

#### CLIWrapper.on_complete

```python
CLIWrapper.on_complete(self, pattern: Optional[str] = None)
```

Decorator to register a completion hook (success or failure).

#### TaskCompletionManager.on_task_complete

```python
TaskCompletionManager.on_task_complete(self, task_type: str, callback: HookCallback) -> None
```

Register a callback for when a specific task type completes.

#### TaskCompletionManager.on_any_complete

```python
TaskCompletionManager.on_any_complete(self, callback: HookCallback) -> None
```

Register a callback for any task completion.

#### TaskCompletionManager.handle_completion

```python
TaskCompletionManager.handle_completion(self, context: ExecutionContext) -> None
```

Handle task completion and trigger appropriate callbacks.

#### TaskCompletionManager.get_session_summary

```python
TaskCompletionManager.get_session_summary(self) -> Dict[str, Any]
```

Get summary of all tasks completed in this session.

#### TaskCompletionManager.should_trigger_reindex

```python
TaskCompletionManager.should_trigger_reindex(self) -> bool
```

Determine if corpus should be re-indexed based on session activity.

#### ContextWindowManager.add_execution

```python
ContextWindowManager.add_execution(self, context: ExecutionContext) -> None
```

Add an execution to the context window.

#### ContextWindowManager.add_file_read

```python
ContextWindowManager.add_file_read(self, filepath: str) -> None
```

Track that a file was read.

#### ContextWindowManager.get_recent_files

```python
ContextWindowManager.get_recent_files(self, limit: int = 10) -> List[str]
```

Get most recently accessed files.

#### ContextWindowManager.get_context_summary

```python
ContextWindowManager.get_context_summary(self) -> Dict[str, Any]
```

Get a summary of current context window state.

#### ContextWindowManager.suggest_pruning

```python
ContextWindowManager.suggest_pruning(self) -> List[str]
```

Suggest files that could be pruned from context.

#### Session.run

```python
Session.run(self, command: Union[str, List[str]], **kwargs) -> ExecutionContext
```

Run a command within this session.

#### Session.should_reindex

```python
Session.should_reindex(self) -> bool
```

Check if corpus re-indexing is recommended based on session activity.

#### Session.summary

```python
Session.summary(self) -> Dict[str, Any]
```

Get a summary of this session's activity.

#### Session.results

```python
Session.results(self) -> List[ExecutionContext]
```

All command results from this session.

#### Session.success_rate

```python
Session.success_rate(self) -> float
```

Fraction of commands that succeeded (0.0 to 1.0).

#### Session.all_passed

```python
Session.all_passed(self) -> bool
```

True if all commands in this session succeeded.

#### Session.modified_files

```python
Session.modified_files(self) -> List[str]
```

List of files modified during this session (from git context).

#### TaskCheckpoint.save

```python
TaskCheckpoint.save(self, task_name: str, context: Dict[str, Any]) -> None
```

Save context for a task.

#### TaskCheckpoint.load

```python
TaskCheckpoint.load(self, task_name: str) -> Optional[Dict[str, Any]]
```

Load context for a task. Returns None if not found.

#### TaskCheckpoint.list_tasks

```python
TaskCheckpoint.list_tasks(self) -> List[str]
```

List all saved task checkpoints.

#### TaskCheckpoint.delete

```python
TaskCheckpoint.delete(self, task_name: str) -> bool
```

Delete a checkpoint. Returns True if deleted.

#### TaskCheckpoint.summarize

```python
TaskCheckpoint.summarize(self, task_name: str) -> Optional[str]
```

Get a one-line summary of a task checkpoint.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `enum.Enum`
- ... and 15 more



## code_concepts.py

Code Concepts Module
====================

Programming concept groups for semantic code search.

Maps common programming synonyms and related terms to enable
intent-based code retrieval. When a develo...


### Functions

#### get_related_terms

```python
get_related_terms(term: str, max_terms: int = 5) -> List[str]
```

Get programming terms related to the given term.

#### expand_code_concepts

```python
expand_code_concepts(terms: List[str], max_expansions_per_term: int = 3, weight: float = 0.6) -> Dict[str, float]
```

Expand a list of terms using code concept groups.

#### get_concept_group

```python
get_concept_group(term: str) -> List[str]
```

Get the concept group names a term belongs to.

#### list_concept_groups

```python
list_concept_groups() -> List[str]
```

List all available concept group names.

#### get_group_terms

```python
get_group_terms(group_name: str) -> List[str]
```

Get all terms in a concept group.

### Dependencies

**Standard Library:**

- `typing.Dict`
- `typing.FrozenSet`
- `typing.List`
- `typing.Set`



## diff.py

Semantic Diff Module
====================

Provides "What Changed?" functionality for comparing:
- Two versions of a document
- Two processor states
- Before/after states of a corpus

This goes beyond...


### Classes

#### TermChange

Represents a change to a term/concept.

**Methods:**

- `pagerank_delta`
- `tfidf_delta`
- `documents_added`
- `documents_removed`

#### RelationChange

Represents a change to a semantic relation.

#### ClusterChange

Represents a change to concept clustering.

#### SemanticDiff

Complete semantic diff between two states.

**Methods:**

- `summary`
- `to_dict`

### Functions

#### compare_processors

```python
compare_processors(old_processor: 'CorticalTextProcessor', new_processor: 'CorticalTextProcessor', top_movers: int = 20, min_pagerank_delta: float = 0.0001) -> SemanticDiff
```

Compare two processor states to find semantic differences.

#### compare_documents

```python
compare_documents(processor: 'CorticalTextProcessor', doc_id_old: str, doc_id_new: str) -> Dict[str, Any]
```

Compare two documents within the same corpus.

#### what_changed

```python
what_changed(processor: 'CorticalTextProcessor', old_content: str, new_content: str, temp_doc_prefix: str = '_diff_temp_') -> Dict[str, Any]
```

Compare two text contents to show what changed semantically.

#### TermChange.pagerank_delta

```python
TermChange.pagerank_delta(self) -> Optional[float]
```

Change in PageRank importance.

#### TermChange.tfidf_delta

```python
TermChange.tfidf_delta(self) -> Optional[float]
```

Change in TF-IDF score.

#### TermChange.documents_added

```python
TermChange.documents_added(self) -> Set[str]
```

Documents where this term newly appears.

#### TermChange.documents_removed

```python
TermChange.documents_removed(self) -> Set[str]
```

Documents where this term no longer appears.

#### SemanticDiff.summary

```python
SemanticDiff.summary(self) -> str
```

Generate a human-readable summary of changes.

#### SemanticDiff.to_dict

```python
SemanticDiff.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for serialization.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 7 more



## fingerprint.py

Fingerprint Module
==================

Semantic fingerprinting for code comparison and similarity analysis.

A fingerprint is an interpretable representation of a text's semantic
content, including te...


### Classes

#### SemanticFingerprint

Structured representation of a text's semantic fingerprint.

### Functions

#### compute_fingerprint

```python
compute_fingerprint(text: str, tokenizer: Tokenizer, layers: Optional[Dict[CorticalLayer, HierarchicalLayer]] = None, top_n: int = 20) -> SemanticFingerprint
```

Compute the semantic fingerprint of a text.

#### compare_fingerprints

```python
compare_fingerprints(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> Dict[str, Any]
```

Compare two fingerprints and compute similarity metrics.

#### explain_fingerprint

```python
explain_fingerprint(fp: SemanticFingerprint, top_n: int = 10) -> Dict[str, Any]
```

Generate a human-readable explanation of a fingerprint.

#### explain_similarity

```python
explain_similarity(fp1: SemanticFingerprint, fp2: SemanticFingerprint, comparison: Optional[Dict[str, Any]] = None) -> str
```

Generate a human-readable explanation of why two fingerprints are similar.

### Dependencies

**Standard Library:**

- `code_concepts.get_concept_group`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- ... and 7 more



## fluent.py

Fluent API for CorticalTextProcessor - chainable method interface.

Example:
    from cortical import FluentProcessor

    # Simple usage
    results = (FluentProcessor()
        .add_document("doc1",...


### Classes

#### FluentProcessor

Fluent/chainable API wrapper for CorticalTextProcessor.

**Methods:**

- `from_existing`
- `from_files`
- `from_directory`
- `load`
- `add_document`
- `add_documents`
- `with_config`
- `with_tokenizer`
- `build`
- `save`
- `search`
- `fast_search`
- `search_passages`
- `expand`
- `processor`
- `is_built`

### Functions

#### FluentProcessor.from_existing

```python
FluentProcessor.from_existing(cls, processor: CorticalTextProcessor) -> 'FluentProcessor'
```

Create a FluentProcessor from an existing CorticalTextProcessor.

#### FluentProcessor.from_files

```python
FluentProcessor.from_files(cls, file_paths: List[Union[str, Path]], tokenizer: Optional[Tokenizer] = None, config: Optional[CorticalConfig] = None) -> 'FluentProcessor'
```

Create a processor from a list of files.

#### FluentProcessor.from_directory

```python
FluentProcessor.from_directory(cls, directory: Union[str, Path], pattern: str = '*.txt', recursive: bool = False, tokenizer: Optional[Tokenizer] = None, config: Optional[CorticalConfig] = None) -> 'FluentProcessor'
```

Create a processor from all files in a directory.

#### FluentProcessor.load

```python
FluentProcessor.load(cls, path: Union[str, Path]) -> 'FluentProcessor'
```

Load a processor from a saved file.

#### FluentProcessor.add_document

```python
FluentProcessor.add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> 'FluentProcessor'
```

Add a document to the processor (chainable).

#### FluentProcessor.add_documents

```python
FluentProcessor.add_documents(self, documents: Union[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str, Dict]]]) -> 'FluentProcessor'
```

Add multiple documents at once (chainable).

#### FluentProcessor.with_config

```python
FluentProcessor.with_config(self, config: CorticalConfig) -> 'FluentProcessor'
```

Set configuration (chainable).

#### FluentProcessor.with_tokenizer

```python
FluentProcessor.with_tokenizer(self, tokenizer: Tokenizer) -> 'FluentProcessor'
```

Set custom tokenizer (chainable).

#### FluentProcessor.build

```python
FluentProcessor.build(self, verbose: bool = True, build_concepts: bool = True, pagerank_method: str = 'standard', connection_strategy: str = 'document_overlap', cluster_strictness: float = 1.0, bridge_weight: float = 0.0, show_progress: bool = False) -> 'FluentProcessor'
```

Build the processor by computing all analysis phases (chainable).

#### FluentProcessor.save

```python
FluentProcessor.save(self, path: Union[str, Path]) -> 'FluentProcessor'
```

Save the processor to disk (chainable).

#### FluentProcessor.search

```python
FluentProcessor.search(self, query: str, top_n: int = 5, use_expansion: bool = True, use_semantic: bool = True) -> List[Tuple[str, float]]
```

Search for documents matching the query.

#### FluentProcessor.fast_search

```python
FluentProcessor.fast_search(self, query: str, top_n: int = 5, candidate_multiplier: int = 3, use_code_concepts: bool = True) -> List[Tuple[str, float]]
```

Fast document search with pre-filtering.

#### FluentProcessor.search_passages

```python
FluentProcessor.search_passages(self, query: str, top_n: int = 5, chunk_size: Optional[int] = None, overlap: Optional[int] = None, use_expansion: bool = True) -> List[Tuple[str, str, int, int, float]]
```

Search for passage chunks matching the query.

#### FluentProcessor.expand

```python
FluentProcessor.expand(self, query: str, max_expansions: Optional[int] = None, use_variants: bool = True, use_code_concepts: bool = False) -> Dict[str, float]
```

Expand a query with related terms.

#### FluentProcessor.processor

```python
FluentProcessor.processor(self) -> CorticalTextProcessor
```

Access the underlying CorticalTextProcessor instance.

#### FluentProcessor.is_built

```python
FluentProcessor.is_built(self) -> bool
```

Check if the processor has been built.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `os`
- `pathlib.Path`
- `processor.CorticalTextProcessor`
- `tokenizer.Tokenizer`
- ... and 6 more



## gaps.py

Gaps Module
===========

Knowledge gap detection and anomaly analysis.

Identifies:
- Isolated documents that don't connect well to the corpus
- Weakly covered topics (few documents)
- Bridge opportun...


### Functions

#### analyze_knowledge_gaps

```python
analyze_knowledge_gaps(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> Dict
```

Analyze the corpus to identify potential knowledge gaps.

#### detect_anomalies

```python
detect_anomalies(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], threshold: float = 0.3) -> List[Dict]
```

Detect documents that don't fit well with the rest of the corpus.

### Dependencies

**Standard Library:**

- `analysis.cosine_similarity`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- ... and 5 more



## mcp_server.py

MCP (Model Context Protocol) Server for Cortical Text Processor.

Provides an MCP server interface for AI agents to integrate with the
Cortical Text Processor, enabling semantic search, query expansio...


### Classes

#### CorticalMCPServer

MCP Server wrapper for CorticalTextProcessor.

**Methods:**

- `run`

### Functions

#### create_mcp_server

```python
create_mcp_server(corpus_path: Optional[str] = None, config: Optional[CorticalConfig] = None) -> CorticalMCPServer
```

Create a Cortical MCP Server instance.

#### main

```python
main()
```

Main entry point for running the MCP server from command line.

#### CorticalMCPServer.run

```python
CorticalMCPServer.run(self, transport: str = 'stdio')
```

Run the MCP server.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `logging`
- `mcp.server.FastMCP`
- `os`
- `pathlib.Path`
- ... and 5 more



## patterns.py

Code Pattern Detection Module
==============================

Detects common programming patterns in indexed code.

Identifies design patterns, idioms, and code structures including:
- Singleton patte...


### Functions

#### detect_patterns_in_text

```python
detect_patterns_in_text(text: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]
```

Detect programming patterns in a text string.

#### detect_patterns_in_documents

```python
detect_patterns_in_documents(documents: Dict[str, str], patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]
```

Detect patterns across multiple documents.

#### get_pattern_summary

```python
get_pattern_summary(pattern_results: Dict[str, List[int]]) -> Dict[str, int]
```

Summarize pattern detection results by counting occurrences.

#### get_patterns_by_category

```python
get_patterns_by_category(pattern_results: Dict[str, List[int]]) -> Dict[str, Dict[str, int]]
```

Group pattern results by category.

#### get_pattern_description

```python
get_pattern_description(pattern_name: str) -> Optional[str]
```

Get the description for a pattern.

#### get_pattern_category

```python
get_pattern_category(pattern_name: str) -> Optional[str]
```

Get the category for a pattern.

#### list_all_patterns

```python
list_all_patterns() -> List[str]
```

List all available pattern names.

#### list_patterns_by_category

```python
list_patterns_by_category(category: str) -> List[str]
```

List all patterns in a specific category.

#### list_all_categories

```python
list_all_categories() -> List[str]
```

List all pattern categories.

#### format_pattern_report

```python
format_pattern_report(pattern_results: Dict[str, List[int]], show_lines: bool = False) -> str
```

Format pattern detection results as a human-readable report.

#### get_corpus_pattern_statistics

```python
get_corpus_pattern_statistics(doc_patterns: Dict[str, Dict[str, List[int]]]) -> Dict[str, any]
```

Compute statistics across all documents.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `re`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 2 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

# Decisions: ADRs

## ADR-001: Add Microseconds to Task ID Generation

**Status:** Accepted  
**Date:** 2025-12-14  
**Tags:** `task-management`, `uniqueness`, `concurrency`  

---

## The Question

Task IDs were generated with second-precision timestamps plus a 4-character hex suffix:

```
T-YYYYMMDD-HHMMSS-XXXX
Example: T-20251214-163052-a1b2
```

During CI testing, the `test_unique_task_ids` test was intermittently failing:

```
AssertionError: 99 != 100
```

When generating 100 task IDs in a tight loop (same second), collisions occurred because:
- Same timestamp for all IDs in that second
- Only 4 hex chars = 65,536 possible suffixes
- Birthday paradox: P(collision) ≈ 7% for 100 items from 65,536

## The Conversation

*This decision emerged from 12 recorded discussion(s).*

### Discussion 1

**When:** 2025-12-16  
**Matched Keywords:** task, add  

**Query:**

> Please deeply think about the best way to implement these tasks in batches dispatched inteligently to sub Agents and tracked/verifyed by enabling director mode with a backup plan that covers potential failure points intelligently.

Tasks:
T-002	Document ML milestone thresholds derivation	Low
T-003	Make CSV export truncation configurable	Low
T-004	Refactor session_logger.py duplication	Low
T-010	Implement confidence scoring thresholds	Medium
T-011	Add semantic similarity to ML predictions	Low
T-0

**Files Explored:** `tasks/*.json`, `scripts/task_utils.py`

### Discussion 2

**When:** 2025-12-16  
**Matched Keywords:** task, add  

**Query:**

> Please deeply think about the best way to implement these tasks in batches dispatched inteligently to sub Agents and tracked/verifyed by enabling director mode with a backup plan that covers potential failure points intelligently.

Tasks:
T-002	Document ML milestone thresholds derivation	Low
T-003	Make CSV export truncation configurable	Low
T-004	Refactor session_logger.py duplication	Low
T-010	Implement confidence scoring thresholds	Medium
T-011	Add semantic similarity to ML predictions	Low
T-0

**Files Explored:** `scripts/task_utils.py`, `tasks/*.json`, `/home/user/Opus-code-test/tests/unit/test_ml_export.py`

### Discussion 3

**When:** 2025-12-16  
**Matched Keywords:** task, add  

**Query:**

> Please deeply think about the best way to implement these tasks in batches dispatched inteligently to sub Agents and tracked/verifyed by enabling director mode with a backup plan that covers potential failure points intelligently.

Tasks:
T-002	Document ML milestone thresholds derivation	Low
T-003	Make CSV export truncation configurable	Low
T-004	Refactor session_logger.py duplication	Low
T-010	Implement confidence scoring thresholds	Medium
T-011	Add semantic similarity to ML predictions	Low
T-0

**Files Explored:** `tasks/*.json`, `/home/user/Opus-code-test/tests/unit/test_ml_export.py`, `scripts/ml_data_collector.py`, `scripts/task_utils.py`

## Options Considered

### Option 1: Increase Random Suffix Length

```
T-YYYYMMDD-HHMMSS-XXXXXXXX  (8 hex chars)
```

**Pros:**
- Simple change
- 4 billion possibilities per second

**Cons:**
- Longer IDs
- Doesn't leverage timestamp ordering

### Option 2: Add Microseconds to Timestamp

```
T-YYYYMMDD-HHMMSSffffff-XXXX
Example: T-20251214-163052123456-a1b2
```

**Pros:**
- Timestamps remain sortable
- 1 million unique timestamps per second
- Combined with 4 hex suffix = practically unlimited uniqueness

**Cons:**
- IDs are 6 characters longer
- Existing code parsing IDs needs update

### Option 3: Use UUID Only

```
T-a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Pros:**
- Guaranteed uniqueness
- Standard format

**Cons:**
- Not human-readable
- Loses temporal ordering
- Much longer

## The Decision

**Chosen Option:** Option 2 - Add Microseconds to Timestamp

**Rationale:**
- Preserves temporal ordering (IDs sort chronologically)
- Microseconds provide 1M unique slots per second
- Combined with 4 hex chars: virtually collision-proof
- Minimal change to existing format

## Implementation

```python
def generate_task_id(session_id: Optional[str] = None) -> str:
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S%f")  # Added %f for microseconds
    suffix = session_id or generate_session_id()
    return f"T-{date_str}-{time_str}-{suffix}"
```

## Consequences

### Positive
- Tests no longer flaky
- IDs unique even under heavy concurrent generation
- Temporal ordering preserved

### Negative
- IDs 6 characters longer
- Tests checking ID format needed update

### Neutral
- Existing IDs continue to work (no migration needed)
- No performance impact

## In Hindsight

*This decision has been referenced in 2 subsequent commit(s).*

- **2025-12-14** (`53c7985`): test: Update task ID format test to expect microseconds
- **2025-12-14** (`5970006`): fix: Add microseconds to task ID to prevent collisions

---

*This decision story was enriched with conversation context from 12 chat session(s). Source: [adr-microseconds-task-id.md](../../samples/decisions/adr-microseconds-task-id.md)*

---

## Architectural Decision Records

*Enriched with conversation context and implementation history.*

---

## Overview

**Total Decisions:** 1  
**Accepted:** 1  

## Decision Catalog

### [ADR-001: Add Microseconds to Task ID Generation](decision-adr-microseconds-task-id.md)

**Status:** Accepted | **Date:** 2025-12-14

Task IDs were generated with second-precision timestamps plus a 4-character hex suffix:...

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

# Evolution: Project History

## Bug Fixes and Lessons

*What broke, how we fixed it, and what we learned.*

---

## Overview

**12 bugs** have been identified and resolved. Each fix taught us something about the system.

## Bug Fix History

### Archive ML session after transcript processing (T-003 16f3)

**Commit:** `59072c8`  
**Date:** 2025-12-16  
**Files Changed:** scripts/ml_data_collector.py  

### Update CSV truncation test for new defaults (input=500, output=2000)

**Commit:** `ca94a01`  
**Date:** 2025-12-16  

### Fix ML data collection milestone counting and add session/action capture

**Commit:** `273baef`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-15/chat-20251216-121720-30c3c1.json, .git-ml/chats/2025-12-16/chat-20251216-121720-01077d.json, .git-ml/chats/2025-12-16/chat-20251216-121720-306450.json, .git-ml/chats/2025-12-16/chat-20251216-121720-5ef95b.json, .git-ml/chats/2025-12-16/chat-20251216-121720-8a1e7b.json  
*(and 6 more)*  

### Address critical ML data collection and prediction issues

**Commit:** `fead1c1`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json, .git-ml/chats/2025-12-16/chat-20251216-115057-3617f9.json, .git-ml/chats/2025-12-16/chat-20251216-115057-9502fd.json, .git-ml/chats/2025-12-16/chat-20251216-115057-cbbe64.json, .git-ml/chats/2025-12-16/chat-20251216-115057-f65b7a.json  
*(and 4 more)*  

### Add missing imports in validate command

**Commit:** `172ad8f`  
**Date:** 2025-12-16  
**Files Changed:** scripts/ml_data_collector.py  

### Clean up gitignore pattern for .git-ml/commits/

**Commit:** `a65d54f`  
**Date:** 2025-12-16  
**Files Changed:** .gitignore  

### Prevent infinite commit loop in ML data collection hooks

**Commit:** `66ad656`  
**Date:** 2025-12-16  
**Files Changed:** .git-ml/chats/2025-12-16/chat-20251216-004054-78b531.json, .git-ml/tracked/commits.jsonl, scripts/ml_data_collector.py  

### Correct hook format in settings.local.json

**Commit:** `19ac02a`  
**Date:** 2025-12-16  
**Files Changed:** .claude/settings.local.json  

### Use filename-based sorting for deterministic session ordering

**Commit:** `61d502d`  
**Date:** 2025-12-15  

### Increase ID suffix length to prevent collisions

**Commit:** `8ac4b6b`  
**Date:** 2025-12-15  

### Add import guards for optional test dependencies

**Commit:** `91ffb04`  
**Date:** 2025-12-15  

### Make session file sorting stable for deterministic ordering

**Commit:** `7433b36`  
**Date:** 2025-12-15

---

## Feature Evolution

*How the Cortical Text Processor gained its capabilities.*

---

## Overview

The system has evolved through **22 feature additions**. Below is the narrative of how each capability came to be.

## Other Capabilities

### Add 6 new intelligent book generators for publisher-ready content

**Commit:** `082aa21`  
**Date:** 2025-12-17  
**Files Modified:** 47  

### Add smart caching to markdown book generation

**Commit:** `afd3c5b`  
**Date:** 2025-12-16  
**Files Modified:** 8  

### Add consolidated markdown book generation

**Commit:** `f8a2ad6`  
**Date:** 2025-12-16  
**Files Modified:** 3  

### Add content generators for Cortical Chronicles (Wave 2)

**Commit:** `3022110`  
**Date:** 2025-12-16  
**Files Modified:** 23  

### Add Cortical Chronicles book infrastructure (Wave 1)

**Commit:** `c730057`  
**Date:** 2025-12-16  
**Files Modified:** 13  

### Batch task distribution implementation via Director orchestration

**Commit:** `4f915c3`  
**Date:** 2025-12-16  
**Files Modified:** 8  

### Add orchestration extraction for director sub-agent tracking

**Commit:** `4eaeb37`  
**Date:** 2025-12-15  

### Add stunning animated ASCII codebase visualizer

**Commit:** `e085a0b`  
**Date:** 2025-12-15  

### Add ASCII art codebase visualization script

**Commit:** `43aae33`  
**Date:** 2025-12-15  

### Complete legacy task system migration

**Commit:** `33dc8b2`  
**Date:** 2025-12-15  

### Add director orchestration execution tracking system

**Commit:** `4976c58`  
**Date:** 2025-12-15  

## Search Capabilities

### Add chunked parallel processing for TF-IDF/BM25 (LEGACY-135)

**Commit:** `5665839`  
**Date:** 2025-12-16  

### Add search integration and web interface (Wave 3)

**Commit:** `0022466`  
**Date:** 2025-12-16  
**Files Modified:** 11  

## Data Capabilities

### Implement WAL + Snapshot persistence system (LEGACY-133)

**Commit:** `c7e662a`  
**Date:** 2025-12-16  

### Add git-tracked JSONL storage for orchestration data

**Commit:** `fb30e38`  
**Date:** 2025-12-15  

## Ml Capabilities

### Implement top priorities (ML capture, state storage, legacy cleanup)

**Commit:** `4820c64`  
**Date:** 2025-12-16  

### Add file existence filter to ML predictions

**Commit:** `3cab2ba`  
**Date:** 2025-12-16  
**Files Modified:** 1  

### Add ML file prediction model

**Commit:** `ac549dd`  
**Date:** 2025-12-16  
**Files Modified:** 2  

### Add chunked storage for git-friendly ML data

**Commit:** `0754540`  
**Date:** 2025-12-16  
**Files Modified:** 4  

### Add ML stats report to CI pipeline

**Commit:** `3e05a70`  
**Date:** 2025-12-16  
**Files Modified:** 9  

## Documentation Capabilities

### Add CI workflow and documentation (Wave 4)

**Commit:** `940fdf2`  
**Date:** 2025-12-16  
**Files Modified:** 5  

### Add animated GIF visualizations to README

**Commit:** `b4d7c82`  
**Date:** 2025-12-15

---

## Refactorings and Architecture Evolution

*How the codebase structure improved over time.*

---

## Overview

The codebase has undergone **3 refactorings**. Each improved code quality, maintainability, or performance.

## Refactoring History

### Complete legacy task system cleanup

**Commit:** `8dedda6`  
**Date:** 2025-12-16  

### Remove unused protobuf serialization (T-013 f0ff)

**Commit:** `d7a98ae`  
**Date:** 2025-12-16  
**Changes:** +100/-1460 lines  
**Scope:** 6 files affected  

### Split large files exceeding 25000 token limit

**Commit:** `21ec5ea`  
**Date:** 2025-12-15

---

## Timeline

---

## December 2025

### Week of Dec 15

- **2025-12-16**: feat: Add book
- **2025-12-16**: docs: Add vision

---

## Project Timeline

*A chronological journey through the Cortical Text Processor's development.*

---

## December 2025

### Week of Dec 15

- **2025-12-17**: ml: Capture session data
- **2025-12-17**: feat: Add 6 new intelligent book generators for publisher-ready content
- **2025-12-16**: ml: Capture session data
- **2025-12-16**: docs: Add living book generation vision
- **2025-12-16**: ml: Capture session data
- **2025-12-16**: feat: Add smart caching to markdown book generation
- **2025-12-16**: ml: Capture session data
- **2025-12-16**: feat: Add consolidated markdown book generation
- **2025-12-16**: feat: Add chunked parallel processing for TF-IDF/BM25 (LEGACY-135)
- **2025-12-16**: feat: Implement WAL + Snapshot persistence system (LEGACY-133)

---

# Future: Roadmap

## Future

*This chapter will be auto-generated in a future update.*

---

# 05 Case Studies

## Case Study: The Great Performance Hunt

*A tale of assumptions, profiling, and unexpected bottlenecks.*

## The Problem

It started with a timeout. The `compute_all()` function was hanging on a corpus of just 125 documents—far smaller than our target of 10,000+. Something was fundamentally wrong.

The system would start processing, print "Computing PageRank...", then silence. No errors, no warnings, just an infinite wait. After 30 seconds, the timeout would trigger and the process would die.

This wasn't a minor performance issue. This was a showstopper.

## The Suspect

The obvious culprit was Louvain clustering. Think about it:

- **Most complex algorithm** - O(n log n) community detection with multiple passes
- **Graph manipulation** - Rewiring communities iteratively until modularity converges
- **Nested loops** - Pass after pass until stability

Every instinct, every pattern-matching neuron in our brains said: "Start there. It has to be Louvain."

But assumptions are dangerous.

## The Investigation

We started with profiling, not guessing. The `profile_full_analysis.py` script measured every phase of `compute_all()`:

```bash
python scripts/profile_full_analysis.py
```

The results were shocking:

| Phase | Before | After | Fix |
|-------|--------|-------|-----|
| `bigram_connections` | **20.85s timeout** | 10.79s | `max_bigrams_per_term=100`, `max_bigrams_per_doc=500` |
| `semantics` | **30.05s timeout** | 5.56s | `max_similarity_pairs=100000`, `min_context_keys=3` |
| `louvain` | **2.2s** | 2.2s | **Not the bottleneck!** |

Read that last line again: **Louvain was innocent.**

The algorithm everyone suspected—the complex graph clustering with multiple iterative passes—was responsible for just 2.2 seconds of a 50+ second hang.

99% of the execution time was hidden in `bigram_connections()` and `extract_corpus_semantics()`.

## The Discovery

Then we saw it. Looking at the code in `cortical/analysis/connections.py`:

```python
# Build indexes for efficient lookup
left_index: Dict[str, List[Minicolumn]] = defaultdict(list)
right_index: Dict[str, List[Minicolumn]] = defaultdict(list)

for bigram in bigrams:
    parts = bigram.content.split(' ')
    if len(parts) == 2:
        left_index[parts[0]].append(bigram)
        right_index[parts[1]].append(bigram)

# Connect bigrams sharing components
for component, bigram_list in left_index.items():
    # THIS is where it exploded
    for i, b1 in enumerate(bigram_list):
        for b2 in bigram_list[i+1:]:
            # Create connection between b1 and b2
```

The problem was hiding in plain sight. For every term that appears in bigrams, we were creating connections between **all pairs** of bigrams containing that term.

**Common terms like "self" appeared in hundreds of bigrams.**

If "self" appears in 300 bigrams (self_attention, self_healing, self_referential, etc.), the nested loop creates:

```
300 × 299 / 2 = 44,850 connections
```

For a single term.

Now imagine dozens of common terms ("return", "function", "value", "data", "process"). Each creating tens of thousands of pairwise connections.

**O(n²) complexity from common terms was creating millions of pairs.**

The complexity analysis confirmed it:

```python
# Without limits: O(n_bigrams²) worst case from common terms creating all-to-all connections
# With limits: O(n_terms * max_bigrams_per_term² + n_docs * max_bigrams_per_doc²)
# Typical with defaults (100, 500): O(n_terms * 10000 + n_docs * 250000) ≈ O(n_bigrams) linear
```

Without limits, the algorithm had **quadratic worst-case complexity**. With limits, it became **effectively linear**.

## The Solution

The fix was elegant: **skip overly common terms** to prevent the O(n²) explosion:

```python
def compute_bigram_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    component_weight: float = 0.5,
    chain_weight: float = 0.7,
    cooccurrence_weight: float = 0.3,
    max_bigrams_per_term: int = 100,      # NEW: Prevent O(n²) from common terms
    max_bigrams_per_doc: int = 500,       # NEW: Prevent O(n²) from large docs
    max_connections_per_bigram: int = 50  # NEW: Cap per-bigram connections
) -> Dict[str, Any]:
    """
    Compute lateral connections between bigrams.

    Args:
        max_bigrams_per_term: Skip terms appearing in more than this many bigrams
            to avoid O(n²) explosion from common terms like "self", "return"
        max_bigrams_per_doc: Skip documents with more than this many bigrams for
            co-occurrence connections to avoid O(n²) explosion
    """
```

With these limits in place:

```python
# Left component matches: "neural_networks" ↔ "neural_processing"
for component, bigram_list in left_index.items():
    # Skip overly common terms to avoid O(n²) explosion
    if len(bigram_list) > max_bigrams_per_term:
        skipped_common_terms += 1
        continue

    for i, b1 in enumerate(bigram_list):
        for b2 in bigram_list[i+1:]:
            # Safe now - bounded by max_bigrams_per_term²
            create_connection(b1, b2, weight=component_weight)
```

**Results:**
- `bigram_connections`: 20.85s timeout → **10.79s** (48% improvement)
- `semantics`: 30.05s timeout → **5.56s** (81% improvement)
- Total `compute_all()`: timeout → **~27s** (viable for production)

The same approach was applied to `extract_corpus_semantics()`:

```python
max_similarity_pairs: int = 100000  # Prevent similarity explosion
min_context_keys: int = 3           # Require meaningful context overlap
```

## The Lesson

**Profile before optimizing.** The obvious culprit is often innocent. The real bottleneck hides in unexpected places.

We suspected Louvain—the complex, iterative graph algorithm with nested loops and community rewiring. The actual problem was a simple nested loop over common terms, creating millions of unnecessary connections.

**Key takeaways:**

1. **Measure, don't assume** - Run the profiler before making changes
2. **Look for O(n²) patterns** - Nested loops over unbounded collections
3. **Common items are dangerous** - High-frequency terms/documents create all-to-all explosions
4. **Add limits early** - Prevent worst-case scenarios with sensible bounds
5. **Track what you skip** - Return stats on skipped items for monitoring

## The Aftermath

This investigation led to several improvements across the codebase:

1. **Profiling became standard practice** - `profile_full_analysis.py` is now run routinely
2. **O(n²) awareness** - Code reviews specifically check for quadratic patterns
3. **Limit parameters everywhere** - All connection-building functions have max limits
4. **Performance tests** - Regression tests verify compute times stay bounded
5. **Documentation** - CLAUDE.md now includes "Performance Lessons Learned" section

The fix enabled the system to scale from 125 documents (timeout) to **10,000+ documents** (27 seconds).

## Try It Yourself

Run the profiler on your own corpus to identify bottlenecks:

```bash
python scripts/profile_full_analysis.py
```

Watch for these warning signs:
- O(n²) patterns in loops over connections
- Common terms/documents creating explosions
- Phases taking >10x longer than expected
- Nested loops without bounds

Look for code like this:

```python
# DANGER: O(n²) if items list is unbounded
for i, item1 in enumerate(items):
    for item2 in items[i+1:]:
        # Creates n × (n-1) / 2 operations
```

And replace with:

```python
# SAFE: Bounded by max_items_per_group
for group, items in grouped_items.items():
    if len(items) > max_items_per_group:
        continue  # Skip overly common groups

    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            # Now bounded by max_items_per_group²
```

---

**Remember:** The algorithm you suspect is often innocent. The real bottleneck is hiding in the code you didn't think to check.

**Profile first. Optimize second. Always.**

---

## Case Studies

*Real problem-solving sessions from the development of the Cortical Text Processor*

No case studies available yet. Case studies are generated from ML session data when sessions demonstrate significant problem-solving narratives.

**What makes a good case study?**

- At least 5 exchanges (substantial investigation)
- Multiple tools used (shows exploration)
- Resulted in commits (concrete outcome)
- Clear problem statement (queries starting with 'fix', 'why', 'how do', etc.)

*These case studies are automatically generated from ML session data collected during development. They demonstrate real problem-solving workflows and serve as both documentation and learning material.*

---

## Case Study: Add session memory and knowledge transfer

*Synthesized from commit history: 2025-12-14*

## The Problem

A new feature was required: Add director agent orchestration prompt. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add director agent orchestration prompt** - Modified 2 files (+425/-1 lines)
2. **Add memory system CLI and improve documentation** - Modified 5 files (+477/-16 lines)
3. **Add session memory and knowledge transfer** - Modified 3 files (+254/-16 lines)
4. **Add session handoff, auto-memory, CI link checker, and tests** - Modified 6 files (+1375/-10 lines)


## The Solution

Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

The solution involved changes to 47 files, adding 14229 lines and removing 173 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 49

- `.claude/commands/director.md`
- `.github/workflows/ci.yml`
- `.markdown-link-check.json`
- `CLAUDE.md`
- `README.md`
- `cortical/observability.py`
- `cortical/patterns.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`
- `cortical/processor/documents.py`

*...and 39 more files*

**Code Changes:** +16760/-216 lines

**Commits:** 5


## Commits in This Story

- `4ab60f2` (2025-12-14): feat: Add director agent orchestration prompt
- `d647b53` (2025-12-14): feat: Add memory system CLI and improve documentation
- `2160f3d` (2025-12-14): memory: Add session memory and knowledge transfer
- `6684152` (2025-12-14): feat: Add session handoff, auto-memory, CI link checker, and tests
- `2a11bf3` (2025-12-14): Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Add test file penalty and code stop word filtering to search

*Synthesized from commit history: 2025-12-14*

## The Problem

A new feature was required: Director mode batch execution - 6 tasks completed in parallel. This would require careful implementation and testing.

## The Journey

The solution was implemented directly.


## The Solution

Add test file penalty and code stop word filtering to search

The solution involved changes to 3 files, adding 51 lines and removing 9 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 33

- `CLAUDE.md`
- `IMPLEMENTATION_SUMMARY.md`
- `PATTERN_DETECTION_GUIDE.md`
- `cortical/observability.py`
- `cortical/patterns.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`
- `cortical/processor/documents.py`
- `cortical/processor/introspection.py`
- `cortical/processor/persistence_api.py`

*...and 23 more files*

**Code Changes:** +9432/-117 lines

**Commits:** 2


## Commits in This Story

- `a9478fd` (2025-12-14): feat: Director mode batch execution - 6 tasks completed in parallel
- `1fafc8b` (2025-12-14): fix: Add test file penalty and code stop word filtering to search

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Replace external action with native Python link checker

*Synthesized from commit history: 2025-12-14*

## The Problem

Development work began: Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu.

## The Journey

The development progressed through several stages:

1. **Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu** - Modified 47 files (+14229/-173 lines)
2. **Add session handoff, auto-memory, CI link checker, and tests** - Modified 6 files (+1375/-10 lines)
3. **Replace external action with native Python link checker** - Modified 5 files (+172/-34 lines)


## The Solution

Adjust Native Over External threshold to 20000 lines

The solution involved changes to 1 files, adding 1 lines and removing 1 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 48

- `.github/workflows/ci.yml`
- `.markdown-link-check.json`
- `CLAUDE.md`
- `README.md`
- `cortical/observability.py`
- `cortical/patterns.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`
- `cortical/processor/documents.py`
- `cortical/processor/introspection.py`

*...and 38 more files*

**Code Changes:** +15777/-218 lines

**Commits:** 4


## Commits in This Story

- `2a11bf3` (2025-12-14): Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu
- `6684152` (2025-12-14): feat: Add session handoff, auto-memory, CI link checker, and tests
- `901a181` (2025-12-14): fix: Replace external action with native Python link checker
- `00f88d4` (2025-12-14): docs: Adjust Native Over External threshold to 20000 lines

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Update skipped tests for processor/ package refactor

*Synthesized from commit history: 2025-12-14*

## The Problem

Code quality improvements were needed. The refactoring affected 9 files: Split processor.py into modular processor/ package (LEGACY-095).

## The Journey

The solution was implemented directly.


## The Solution

Update skipped tests for processor/ package refactor

The solution involved changes to 2 files, adding 79 lines and removing 14 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 10

- `CLAUDE.md`
- `cortical/__init__.py`
- `cortical/processor/__init__.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`
- `cortical/processor/documents.py`
- `cortical/processor/introspection.py`
- `cortical/processor/persistence_api.py`
- `cortical/processor/query_api.py`
- `tests/test_generate_ai_metadata.py`

**Code Changes:** +2913/-18 lines

**Commits:** 2


## Commits in This Story

- `090910f` (2025-12-14): refactor: Split processor.py into modular processor/ package (LEGACY-095)
- `d6718db` (2025-12-14): fix: Update skipped tests for processor/ package refactor

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Use heredoc for Python in CI to avoid YAML syntax error

*Synthesized from commit history: 2025-12-14*

## The Problem

Development work began: Make push trigger explicit for all branches.

## The Journey

The development progressed through several stages:

1. **Make push trigger explicit for all branches** - Modified 1 files (+2/-0 lines)
2. **Use heredoc for Python in CI to avoid YAML syntax error** - Modified 1 files (+14/-14 lines)


## The Solution

Add test_mcp_server.py to integration tests for coverage

The solution involved changes to 1 files, adding 1 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 1

- `.github/workflows/ci.yml`

**Code Changes:** +17/-14 lines

**Commits:** 3


## Commits in This Story

- `dfedf5e` (2025-12-14): ci: Make push trigger explicit for all branches
- `ef84437` (2025-12-14): fix: Use heredoc for Python in CI to avoid YAML syntax error
- `0aec3d3` (2025-12-14): ci: Add test_mcp_server.py to integration tests for coverage

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Clean up directory structure and queue search relevance fixes

*Synthesized from commit history: 2025-12-14*

## The Problem

A new feature was required: Director mode batch execution - 6 tasks completed in parallel. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Director mode batch execution - 6 tasks completed in parallel** - Modified 31 files (+9381/-108 lines)
2. **Update task status - mark 6 tasks completed from director mode batch** - Modified 2 files (+2747/-9 lines)
3. **Clean up directory structure and queue search relevance fixes** - Modified 5 files (+68/-293 lines)
4. **Add test file penalty and code stop word filtering to search** - Modified 3 files (+51/-9 lines)


## The Solution

Mark search relevance tasks T-002, T-003 as completed

The solution involved changes to 1 files, adding 23 lines and removing 9 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 39

- `CLAUDE.md`
- `IMPLEMENTATION_SUMMARY.md`
- `PATTERN_DETECTION_GUIDE.md`
- `cortical/observability.py`
- `cortical/patterns.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`
- `cortical/processor/documents.py`
- `cortical/processor/introspection.py`
- `cortical/processor/persistence_api.py`

*...and 29 more files*

**Code Changes:** +12270/-428 lines

**Commits:** 5


## Commits in This Story

- `a9478fd` (2025-12-14): feat: Director mode batch execution - 6 tasks completed in parallel
- `afc7a2d` (2025-12-14): chore: Update task status - mark 6 tasks completed from director mode batch
- `cd8b9f5` (2025-12-14): chore: Clean up directory structure and queue search relevance fixes
- `1fafc8b` (2025-12-14): fix: Add test file penalty and code stop word filtering to search
- `461cb9a` (2025-12-14): chore: Mark search relevance tasks T-002, T-003 as completed

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add director agent orchestration prompt

*Synthesized from commit history: 2025-12-14*

## The Problem

A new feature was required: Add future tasks for text-as-memories integration. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add future tasks for text-as-memories integration** - Modified 1 files (+91/-1 lines)
2. **Add memory-manager skill and CLAUDE.md documentation** - Modified 2 files (+241/-1 lines)
3. **Add /knowledge-transfer slash command** - Modified 2 files (+215/-0 lines)
4. **Add merge-safety task for memory/decision filenames** - Modified 1 files (+16/-1 lines)
5. **Add documentation improvement tasks** - Modified 1 files (+46/-1 lines)
6. **Add director agent orchestration prompt** - Modified 2 files (+425/-1 lines)
7. **Add memory system CLI and improve documentation** - Modified 5 files (+477/-16 lines)
8. **Add session memory and knowledge transfer** - Modified 3 files (+254/-16 lines)
9. **Add session handoff, auto-memory, CI link checker, and tests** - Modified 6 files (+1375/-10 lines)


## The Solution

Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

The solution involved changes to 47 files, adding 14229 lines and removing 173 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 52

- `.claude/commands/director.md`
- `.claude/commands/knowledge-transfer.md`
- `.claude/skills/memory-manager/SKILL.md`
- `.github/workflows/ci.yml`
- `.markdown-link-check.json`
- `CLAUDE.md`
- `README.md`
- `cortical/observability.py`
- `cortical/patterns.py`
- `cortical/processor/compute.py`

*...and 42 more files*

**Code Changes:** +17369/-220 lines

**Commits:** 10


## Commits in This Story

- `966b992` (2025-12-14): feat: Add future tasks for text-as-memories integration
- `6d2c934` (2025-12-14): feat: Add memory-manager skill and CLAUDE.md documentation
- `b7453a8` (2025-12-14): feat: Add /knowledge-transfer slash command
- `ec81905` (2025-12-14): task: Add merge-safety task for memory/decision filenames
- `87b259c` (2025-12-14): task: Add documentation improvement tasks
- `4ab60f2` (2025-12-14): feat: Add director agent orchestration prompt
- `d647b53` (2025-12-14): feat: Add memory system CLI and improve documentation
- `2160f3d` (2025-12-14): memory: Add session memory and knowledge transfer
- `6684152` (2025-12-14): feat: Add session handoff, auto-memory, CI link checker, and tests
- `2a11bf3` (2025-12-14): Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add session handoff, auto-memory, CI link checker, and tests

*Synthesized from commit history: 2025-12-14*

## The Problem

A new feature was required: Add memory system CLI and improve documentation. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add memory system CLI and improve documentation** - Modified 5 files (+477/-16 lines)
2. **Add session memory and knowledge transfer** - Modified 3 files (+254/-16 lines)
3. **Add session handoff, auto-memory, CI link checker, and tests** - Modified 6 files (+1375/-10 lines)


## The Solution

Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

The solution involved changes to 47 files, adding 14229 lines and removing 173 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 48

- `.github/workflows/ci.yml`
- `.markdown-link-check.json`
- `CLAUDE.md`
- `README.md`
- `cortical/observability.py`
- `cortical/patterns.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`
- `cortical/processor/documents.py`
- `cortical/processor/introspection.py`

*...and 38 more files*

**Code Changes:** +16335/-215 lines

**Commits:** 4


## Commits in This Story

- `d647b53` (2025-12-14): feat: Add memory system CLI and improve documentation
- `2160f3d` (2025-12-14): memory: Add session memory and knowledge transfer
- `6684152` (2025-12-14): feat: Add session handoff, auto-memory, CI link checker, and tests
- `2a11bf3` (2025-12-14): Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Refactoring - Migrate to merge-friendly task system and add security tasks

*Synthesized from commit history: 2025-12-14*

## The Problem

Development work began: Rename CLAUDE.md.potential to CLAUDE.md.

## The Journey

The development progressed through several stages:

1. **Rename CLAUDE.md.potential to CLAUDE.md** - Modified 1 files (+0/-0 lines)
2. **Add comprehensive security knowledge transfer document** - Modified 1 files (+478/-0 lines)
3. **Migrate to merge-friendly task system and add security tasks** - Modified 6 files (+3217/-30 lines)


## The Solution

Add pickle warnings, deprecation notices, and CI security scanning

The solution involved changes to 3 files, adding 109 lines and removing 0 lines.


## The Lesson

**Code quality is an ongoing process.** Regular refactoring keeps the codebase maintainable and reduces technical debt.

## Technical Details

**Files Modified:** 10

- `.claude/skills/task-manager/SKILL.md`
- `.github/workflows/ci.yml`
- `CLAUDE.md`
- `README.md`
- `TASK_LIST.md`
- `cortical/persistence.py`
- `docs/security-knowledge-transfer.md`
- `scripts/migrate_legacy_tasks.py`
- `tasks/2025-12-14_11-15-01_41d5.json`
- `tasks/legacy_migration.json`

**Code Changes:** +3804/-30 lines

**Commits:** 4


## Commits in This Story

- `77b1970` (2025-12-14): chore: Rename CLAUDE.md.potential to CLAUDE.md
- `46a0116` (2025-12-14): docs: Add comprehensive security knowledge transfer document
- `b41c51d` (2025-12-14): refactor: Migrate to merge-friendly task system and add security tasks
- `90b989f` (2025-12-14): security: Add pickle warnings, deprecation notices, and CI security scanning

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Refactoring - Split processor.py into modular processor/ package (LEGACY-095)

*Synthesized from commit history: 2025-12-14*

## The Problem

A new feature was required: Add HMAC signature verification for pickle files (SEC-003). This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add HMAC signature verification for pickle files (SEC-003)** - Modified 7 files (+1872/-16 lines)
2. **Merge pull request #80 from scrawlsbenches/claude/resume-dog-fooding-9RPIV** - Modified 18 files (+2582/-15 lines)
3. **Split processor.py into modular processor/ package (LEGACY-095)** - Modified 9 files (+2834/-4 lines)


## The Solution

Remove dead processor.py and add mixin boundary tests (LEGACY-095)

The solution involved changes to 2 files, adding 530 lines and removing 3234 lines.


## The Lesson

**Code quality is an ongoing process.** Regular refactoring keeps the codebase maintainable and reduces technical debt.

## Technical Details

**Files Modified:** 32

- `.claude/commands/director.md`
- `.claude/commands/knowledge-transfer.md`
- `.claude/skills/memory-manager/SKILL.md`
- `.gitignore`
- `CLAUDE.md`
- `cortical/__init__.py`
- `cortical/config.py`
- `cortical/persistence.py`
- `cortical/processor.py`
- `cortical/processor/__init__.py`

*...and 22 more files*

**Code Changes:** +7818/-3269 lines

**Commits:** 4


## Commits in This Story

- `6f3a1cc` (2025-12-14): feat: Add HMAC signature verification for pickle files (SEC-003)
- `3a2d7af` (2025-12-14): Merge pull request #80 from scrawlsbenches/claude/resume-dog-fooding-9RPIV
- `090910f` (2025-12-14): refactor: Split processor.py into modular processor/ package (LEGACY-095)
- `890dda8` (2025-12-14): chore: Remove dead processor.py and add mixin boundary tests (LEGACY-095)

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Update task ID format test to expect microseconds

*Synthesized from commit history: 2025-12-14*

## The Problem

A bug was discovered: Add microseconds to task ID to prevent collisions. The issue needed investigation and resolution.

## The Journey

The solution was implemented directly.


## The Solution

Update task ID format test to expect microseconds

The solution involved changes to 1 files, adding 2 lines and removing 2 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 2

- `scripts/task_utils.py`
- `tests/unit/test_task_utils.py`

**Code Changes:** +7/-6 lines

**Commits:** 2


## Commits in This Story

- `5970006` (2025-12-14): fix: Add microseconds to task ID to prevent collisions
- `53c7985` (2025-12-14): test: Update task ID format test to expect microseconds

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Add ML commit data for previous commit

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add CI auto-capture and GitHub PR/Issue data collection. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add CI auto-capture and GitHub PR/Issue data collection** - Modified 5 files (+567624/-2 lines)
2. **Add ML commit data for previous commit** - Modified 1 files (+568009/-0 lines)
3. **Add ML commit data** - Modified 1 files (+568045/-0 lines)
4. **Stop tracking ML commit data files (too large for GitHub)** - Modified 472 files (+4/-2263268 lines)
5. **Add lightweight commit data for ephemeral environments** - Modified 475 files (+11659/-13 lines)
6. **Add ML commit data for previous commit** - Modified 1 files (+492/-0 lines)
7. **Add ML commit data for previous commit** - Modified 1 files (+18/-0 lines)
8. **Add ML commit data** - Modified 1 files (+18/-0 lines)
9. **Add ML commit data** - Modified 1 files (+18/-0 lines)


## The Solution

Add ML commit data

The solution involved changes to 1 files, adding 18 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 953

- `.git-ml/commits-lite/0039ad5b13fb_2025-12-11.json`
- `.git-ml/commits-lite/00f88d48ab42_2025-12-14.json`
- `.git-ml/commits-lite/051d20028ddd_2025-12-13.json`
- `.git-ml/commits-lite/051d924cae88_2025-12-13.json`
- `.git-ml/commits-lite/059085dfe407_2025-12-10.json`
- `.git-ml/commits-lite/0598bade86fa_2025-12-11.json`
- `.git-ml/commits-lite/061a157b98f1_2025-12-13.json`
- `.git-ml/commits-lite/063c542400da_2025-12-10.json`
- `.git-ml/commits-lite/0656744909c4_2025-12-12.json`
- `.git-ml/commits-lite/06897859b4fa_2025-12-11.json`

*...and 943 more files*

**Code Changes:** +1715905/-2263283 lines

**Commits:** 10


## Commits in This Story

- `59bc226` (2025-12-15): feat: Add CI auto-capture and GitHub PR/Issue data collection
- `5849304` (2025-12-15): chore: Add ML commit data for previous commit
- `c4d25ae` (2025-12-15): chore: Add ML commit data
- `a6f39e0` (2025-12-15): fix: Stop tracking ML commit data files (too large for GitHub)
- `89d6aa5` (2025-12-15): feat: Add lightweight commit data for ephemeral environments
- `84ddf26` (2025-12-15): chore: Add ML commit data for previous commit
- `af67a1e` (2025-12-15): chore: Add ML commit data for previous commit
- `a3471f5` (2025-12-15): chore: Add ML commit data
- `6368f87` (2025-12-15): chore: Add ML commit data
- `5fbd60d` (2025-12-15): chore: Add ML commit data

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Add ML commit data

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add lightweight commit data for ephemeral environments. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add lightweight commit data for ephemeral environments** - Modified 475 files (+11659/-13 lines)
2. **Add ML commit data for previous commit** - Modified 1 files (+492/-0 lines)
3. **Add ML commit data for previous commit** - Modified 1 files (+18/-0 lines)
4. **Add ML commit data** - Modified 1 files (+18/-0 lines)
5. **Add ML commit data** - Modified 1 files (+18/-0 lines)
6. **Add ML commit data** - Modified 1 files (+18/-0 lines)
7. **Add ML commit data** - Modified 1 files (+18/-0 lines)
8. **Add ML commit data** - Modified 1 files (+18/-0 lines)
9. **Add ML commit data** - Modified 1 files (+18/-0 lines)


## The Solution

Add ML commit data

The solution involved changes to 1 files, adding 18 lines and removing 0 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 484

- `.git-ml/commits-lite/0039ad5b13fb_2025-12-11.json`
- `.git-ml/commits-lite/00f88d48ab42_2025-12-14.json`
- `.git-ml/commits-lite/051d20028ddd_2025-12-13.json`
- `.git-ml/commits-lite/051d924cae88_2025-12-13.json`
- `.git-ml/commits-lite/059085dfe407_2025-12-10.json`
- `.git-ml/commits-lite/0598bade86fa_2025-12-11.json`
- `.git-ml/commits-lite/061a157b98f1_2025-12-13.json`
- `.git-ml/commits-lite/063c542400da_2025-12-10.json`
- `.git-ml/commits-lite/0656744909c4_2025-12-12.json`
- `.git-ml/commits-lite/06897859b4fa_2025-12-11.json`

*...and 474 more files*

**Code Changes:** +12295/-13 lines

**Commits:** 10


## Commits in This Story

- `89d6aa5` (2025-12-15): feat: Add lightweight commit data for ephemeral environments
- `84ddf26` (2025-12-15): chore: Add ML commit data for previous commit
- `af67a1e` (2025-12-15): chore: Add ML commit data for previous commit
- `a3471f5` (2025-12-15): chore: Add ML commit data
- `6368f87` (2025-12-15): chore: Add ML commit data
- `5fbd60d` (2025-12-15): chore: Add ML commit data
- `c85a5a5` (2025-12-15): chore: Add ML commit data
- `7ace489` (2025-12-15): chore: Add ML commit data
- `4ba9a0b` (2025-12-15): chore: Add ML commit data
- `eadbbc8` (2025-12-15): chore: Add ML commit data

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Harden ML data collector with critical fixes

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add ML data collection infrastructure for project-specific micro-model. This would require careful implementation and testing.

## The Journey

The solution was implemented directly.


## The Solution

Harden ML data collector with critical fixes

The solution involved changes to 1 files, adding 151 lines and removing 54 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 4

- `.claude/hooks/session_logger.py`
- `.claude/skills/ml-logger/SKILL.md`
- `.gitignore`
- `scripts/ml_data_collector.py`

**Code Changes:** +1190/-54 lines

**Commits:** 2


## Commits in This Story

- `1568f3c` (2025-12-15): feat: Add ML data collection infrastructure for project-specific micro-model
- `4438d60` (2025-12-15): fix: Harden ML data collector with critical fixes

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Increase ML data retention to 2 years for training milestones

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add privacy features to ML data collection. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add privacy features to ML data collection** - Modified 2 files (+628/-1 lines)
2. **Add ML data collection section to README** - Modified 1 files (+89/-0 lines)
3. **Increase ML data retention to 2 years for training milestones** - Modified 2 files (+7/-5 lines)
4. **Add automatic ML data collection on session startup** - Modified 3 files (+95/-22 lines)


## The Solution

Share ML commit data and aggregated patterns in git

The solution involved changes to 474 files, adding 561272 lines and removing 4 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 478

- `.claude/settings.local.json`
- `.git-ml/commits/0039ad5b_2025-12-11_24b1b10a.json`
- `.git-ml/commits/00f88d48_2025-12-14_8749d448.json`
- `.git-ml/commits/051d2002_2025-12-13_7896a312.json`
- `.git-ml/commits/051d924c_2025-12-13_bfbb2049.json`
- `.git-ml/commits/059085df_2025-12-10_4adc6156.json`
- `.git-ml/commits/0598bade_2025-12-11_ab9a0c3b.json`
- `.git-ml/commits/061a157b_2025-12-13_af26e95e.json`
- `.git-ml/commits/063c5424_2025-12-10_c2422fa6.json`
- `.git-ml/commits/06567449_2025-12-12_95eabdde.json`

*...and 468 more files*

**Code Changes:** +562091/-32 lines

**Commits:** 5


## Commits in This Story

- `e188508` (2025-12-15): feat: Add privacy features to ML data collection
- `df75750` (2025-12-15): docs: Add ML data collection section to README
- `95e9f06` (2025-12-15): fix: Increase ML data retention to 2 years for training milestones
- `b805e13` (2025-12-15): feat: Add automatic ML data collection on session startup
- `6570973` (2025-12-15): feat: Share ML commit data and aggregated patterns in git

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Stop tracking ML commit data files (too large for GitHub)

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add automatic ML data collection on session startup. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add automatic ML data collection on session startup** - Modified 3 files (+95/-22 lines)
2. **Share ML commit data and aggregated patterns in git** - Modified 474 files (+561272/-4 lines)
3. **Add CI auto-capture and GitHub PR/Issue data collection** - Modified 5 files (+567624/-2 lines)
4. **Add ML commit data for previous commit** - Modified 1 files (+568009/-0 lines)
5. **Add ML commit data** - Modified 1 files (+568045/-0 lines)
6. **Stop tracking ML commit data files (too large for GitHub)** - Modified 472 files (+4/-2263268 lines)
7. **Add lightweight commit data for ephemeral environments** - Modified 475 files (+11659/-13 lines)
8. **Add ML commit data for previous commit** - Modified 1 files (+492/-0 lines)
9. **Add ML commit data for previous commit** - Modified 1 files (+18/-0 lines)


## The Solution

Add ML commit data

The solution involved changes to 1 files, adding 18 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 957

- `.claude/settings.local.json`
- `.git-ml/commits-lite/0039ad5b13fb_2025-12-11.json`
- `.git-ml/commits-lite/00f88d48ab42_2025-12-14.json`
- `.git-ml/commits-lite/051d20028ddd_2025-12-13.json`
- `.git-ml/commits-lite/051d924cae88_2025-12-13.json`
- `.git-ml/commits-lite/059085dfe407_2025-12-10.json`
- `.git-ml/commits-lite/0598bade86fa_2025-12-11.json`
- `.git-ml/commits-lite/061a157b98f1_2025-12-13.json`
- `.git-ml/commits-lite/063c542400da_2025-12-10.json`
- `.git-ml/commits-lite/0656744909c4_2025-12-12.json`

*...and 947 more files*

**Code Changes:** +2277236/-2263309 lines

**Commits:** 10


## Commits in This Story

- `b805e13` (2025-12-15): feat: Add automatic ML data collection on session startup
- `6570973` (2025-12-15): feat: Share ML commit data and aggregated patterns in git
- `59bc226` (2025-12-15): feat: Add CI auto-capture and GitHub PR/Issue data collection
- `5849304` (2025-12-15): chore: Add ML commit data for previous commit
- `c4d25ae` (2025-12-15): chore: Add ML commit data
- `a6f39e0` (2025-12-15): fix: Stop tracking ML commit data files (too large for GitHub)
- `89d6aa5` (2025-12-15): feat: Add lightweight commit data for ephemeral environments
- `84ddf26` (2025-12-15): chore: Add ML commit data for previous commit
- `af67a1e` (2025-12-15): chore: Add ML commit data for previous commit
- `a3471f5` (2025-12-15): chore: Add ML commit data

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Update tests for BM25 default and stop word tokenization

*Synthesized from commit history: 2025-12-15*

## The Problem

Development work began: Add Stop hook config and update task tracking.

## The Journey

The development progressed through several stages:

1. **Add Stop hook config and update task tracking** - Modified 3 files (+148/-10 lines)
2. **Update tests for BM25 default and stop word tokenization** - Modified 2 files (+23/-10 lines)


## The Solution

Merge remote-tracking branch 'origin/main' into claude/multi-index-design-DvifZ

The solution involved changes to 22 files, adding 3516 lines and removing 75 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 25

- `.claude/settings.local.json`
- `CLAUDE.md`
- `benchmarks/BASELINE_SUMMARY.md`
- `benchmarks/after_bm25.json`
- `benchmarks/baseline_tfidf.json`
- `benchmarks/baseline_tfidf_real.json`
- `cortical/analysis.py`
- `cortical/config.py`
- `cortical/processor/compute.py`
- `cortical/processor/core.py`

*...and 15 more files*

**Code Changes:** +3687/-95 lines

**Commits:** 3


## Commits in This Story

- `293a467` (2025-12-15): chore: Add Stop hook config and update task tracking
- `9dc7268` (2025-12-15): fix: Update tests for BM25 default and stop word tokenization
- `ed36d6e` (2025-12-15): Merge remote-tracking branch 'origin/main' into claude/multi-index-design-DvifZ

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add CI status integration for ML outcome tracking

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add commit-chat session linking for ML training. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add commit-chat session linking for ML training** - Modified 1 files (+234/-6 lines)
2. **Add CI status integration for ML outcome tracking** - Modified 1 files (+180/-0 lines)


## The Solution

Address audit findings and add documentation

The solution involved changes to 4 files, adding 201 lines and removing 15 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 4

- `.claude/commands/ml-log.md`
- `.claude/commands/ml-stats.md`
- `CLAUDE.md`
- `scripts/ml_data_collector.py`

**Code Changes:** +615/-21 lines

**Commits:** 3


## Commits in This Story

- `1b7f6d5` (2025-12-15): feat: Add commit-chat session linking for ML training
- `ed66817` (2025-12-15): feat: Add CI status integration for ML outcome tracking
- `36be3a1` (2025-12-15): fix: Address audit findings and add documentation

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add comprehensive delegation command template

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add reusable pre-merge sanity check command. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add reusable pre-merge sanity check command** - Modified 1 files (+79/-0 lines)
2. **Add comprehensive delegation command template** - Modified 1 files (+165/-0 lines)


## The Solution

Merge remote-tracking branch 'origin/main' into claude/multi-index-design-DvifZ

The solution involved changes to 28 files, adding 6031 lines and removing 63 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 28

- `.claude/commands/delegate.md`
- `.claude/commands/ml-log.md`
- `.claude/commands/ml-stats.md`
- `.claude/commands/sanity-check.md`
- `.claude/hooks/session_logger.py`
- `.claude/settings.local.json`
- `.claude/skills/ml-logger/SKILL.md`
- `.gitignore`
- `CLAUDE.md`
- `README.md`

*...and 18 more files*

**Code Changes:** +6275/-63 lines

**Commits:** 3


## Commits in This Story

- `cc0ff38` (2025-12-15): feat: Add reusable pre-merge sanity check command
- `aac63a7` (2025-12-15): feat: Add comprehensive delegation command template
- `f371d95` (2025-12-15): Merge remote-tracking branch 'origin/main' into claude/multi-index-design-DvifZ

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add export, feedback, and quality-report commands to ML collector

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add session handoff generator for context preservation. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add session handoff generator for context preservation** - Modified 4 files (+442/-0 lines)
2. **Add export, feedback, and quality-report commands to ML collector** - Modified 1 files (+769/-7 lines)


## The Solution

Update stale query.py and processor.py references

The solution involved changes to 11 files, adding 95 lines and removing 71 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 14

- `.claude/skills/ml-logger/SKILL.md`
- `CLAUDE.md`
- `docs/algorithms.md`
- `docs/architecture.md`
- `docs/claude-usage.md`
- `docs/code-of-ethics.md`
- `docs/devex-tools.md`
- `docs/dogfooding.md`
- `docs/glossary.md`
- `docs/louvain_resolution_analysis.md`

*...and 4 more files*

**Code Changes:** +1306/-78 lines

**Commits:** 3


## Commits in This Story

- `9bd4067` (2025-12-15): feat: Add session handoff generator for context preservation
- `a75761b` (2025-12-15): feat: Add export, feedback, and quality-report commands to ML collector
- `86cc3bb` (2025-12-15): docs: Update stale query.py and processor.py references

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add lightweight commit data for ephemeral environments

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Share ML commit data and aggregated patterns in git. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Share ML commit data and aggregated patterns in git** - Modified 474 files (+561272/-4 lines)
2. **Add CI auto-capture and GitHub PR/Issue data collection** - Modified 5 files (+567624/-2 lines)
3. **Add ML commit data for previous commit** - Modified 1 files (+568009/-0 lines)
4. **Add ML commit data** - Modified 1 files (+568045/-0 lines)
5. **Stop tracking ML commit data files (too large for GitHub)** - Modified 472 files (+4/-2263268 lines)
6. **Add lightweight commit data for ephemeral environments** - Modified 475 files (+11659/-13 lines)
7. **Add ML commit data for previous commit** - Modified 1 files (+492/-0 lines)
8. **Add ML commit data for previous commit** - Modified 1 files (+18/-0 lines)
9. **Add ML commit data** - Modified 1 files (+18/-0 lines)


## The Solution

Add ML commit data

The solution involved changes to 1 files, adding 18 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 956

- `.git-ml/commits-lite/0039ad5b13fb_2025-12-11.json`
- `.git-ml/commits-lite/00f88d48ab42_2025-12-14.json`
- `.git-ml/commits-lite/051d20028ddd_2025-12-13.json`
- `.git-ml/commits-lite/051d924cae88_2025-12-13.json`
- `.git-ml/commits-lite/059085dfe407_2025-12-10.json`
- `.git-ml/commits-lite/0598bade86fa_2025-12-11.json`
- `.git-ml/commits-lite/061a157b98f1_2025-12-13.json`
- `.git-ml/commits-lite/063c542400da_2025-12-10.json`
- `.git-ml/commits-lite/0656744909c4_2025-12-12.json`
- `.git-ml/commits-lite/06897859b4fa_2025-12-11.json`

*...and 946 more files*

**Code Changes:** +2277159/-2263287 lines

**Commits:** 10


## Commits in This Story

- `6570973` (2025-12-15): feat: Share ML commit data and aggregated patterns in git
- `59bc226` (2025-12-15): feat: Add CI auto-capture and GitHub PR/Issue data collection
- `5849304` (2025-12-15): chore: Add ML commit data for previous commit
- `c4d25ae` (2025-12-15): chore: Add ML commit data
- `a6f39e0` (2025-12-15): fix: Stop tracking ML commit data files (too large for GitHub)
- `89d6aa5` (2025-12-15): feat: Add lightweight commit data for ephemeral environments
- `84ddf26` (2025-12-15): chore: Add ML commit data for previous commit
- `af67a1e` (2025-12-15): chore: Add ML commit data for previous commit
- `a3471f5` (2025-12-15): chore: Add ML commit data
- `6368f87` (2025-12-15): chore: Add ML commit data

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Feature Development - Add schema validation for ML data integrity

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add CI status integration for ML outcome tracking. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add CI status integration for ML outcome tracking** - Modified 1 files (+180/-0 lines)
2. **Add commit-chat session linking for ML training** - Modified 1 files (+234/-6 lines)
3. **Add schema validation for ML data integrity** - Modified 1 files (+177/-9 lines)
4. **Address audit findings and add documentation** - Modified 4 files (+201/-15 lines)


## The Solution

Add Development Environment Setup section to CLAUDE.md

The solution involved changes to 1 files, adding 23 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 4

- `.claude/commands/ml-log.md`
- `.claude/commands/ml-stats.md`
- `CLAUDE.md`
- `scripts/ml_data_collector.py`

**Code Changes:** +815/-30 lines

**Commits:** 5


## Commits in This Story

- `ed66817` (2025-12-15): feat: Add CI status integration for ML outcome tracking
- `1b7f6d5` (2025-12-15): feat: Add commit-chat session linking for ML training
- `1d9f520` (2025-12-15): feat: Add schema validation for ML data integrity
- `36be3a1` (2025-12-15): fix: Address audit findings and add documentation
- `bdea0a5` (2025-12-15): Add Development Environment Setup section to CLAUDE.md

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Refactoring - Consolidate ML data to single JSONL files

*Synthesized from commit history: 2025-12-15*

## The Problem

Development work began: Add ML commit data.

## The Journey

The development progressed through several stages:

1. **Add ML commit data** - Modified 1 files (+18/-0 lines)
2. **Add ML commit data** - Modified 1 files (+18/-0 lines)
3. **Consolidate ML data to single JSONL files** - Modified 486 files (+658/-12208 lines)


## The Solution

Merge pull request #95 from scrawlsbenches/claude/check-director-data-collection-WzK3q

The solution involved changes to 8 files, adding 776 lines and removing 10 lines.


## The Lesson

**Code quality is an ongoing process.** Regular refactoring keeps the codebase maintainable and reduces technical debt.

## Technical Details

**Files Modified:** 492

- `.claude/commands/director.md`
- `.git-ml/commits-lite/0039ad5b13fb_2025-12-11.json`
- `.git-ml/commits-lite/00f88d48ab42_2025-12-14.json`
- `.git-ml/commits-lite/051d20028ddd_2025-12-13.json`
- `.git-ml/commits-lite/051d924cae88_2025-12-13.json`
- `.git-ml/commits-lite/059085dfe407_2025-12-10.json`
- `.git-ml/commits-lite/0598bade86fa_2025-12-11.json`
- `.git-ml/commits-lite/061a157b98f1_2025-12-13.json`
- `.git-ml/commits-lite/063c542400da_2025-12-10.json`
- `.git-ml/commits-lite/0656744909c4_2025-12-12.json`

*...and 482 more files*

**Code Changes:** +1470/-12218 lines

**Commits:** 4


## Commits in This Story

- `eadbbc8` (2025-12-15): chore: Add ML commit data
- `4ba9a0b` (2025-12-15): chore: Add ML commit data
- `205fe34` (2025-12-15): refactor: Consolidate ML data to single JSONL files
- `8e919a3` (2025-12-15): Merge pull request #95 from scrawlsbenches/claude/check-director-data-collection-WzK3q

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Add unit tests for Cortical Chronicles generators

*Synthesized from commit history: 2025-12-16*

## The Problem

A new feature was required: Add search integration and web interface (Wave 3). This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add search integration and web interface (Wave 3)** - Modified 11 files (+2959/-1 lines)
2. **Add CI workflow and documentation (Wave 4)** - Modified 5 files (+1724/-1 lines)
3. **Wave 5 integration testing verification** - Modified 25 files (+227/-88 lines)
4. **ML data sync** - Modified 43 files (+904/-6 lines)
5. **ML data sync** - Modified 44 files (+923/-0 lines)
6. **Add unit tests for Cortical Chronicles generators** - Modified 2 files (+1372/-0 lines)
7. **ML data sync** - Modified 43 files (+922/-0 lines)
8. **ML data sync** - Modified 45 files (+941/-0 lines)
9. **ML data sync** - Modified 46 files (+981/-0 lines)


## The Solution

Add consolidated markdown book generation

The solution involved changes to 3 files, adding 6392 lines and removing 2 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 254

- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-000.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-001.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-002.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-003.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-004.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-005.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-006.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-007.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-008.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-009.json`

*...and 244 more files*

**Code Changes:** +17345/-98 lines

**Commits:** 10


## Commits in This Story

- `0022466` (2025-12-16): feat: Add search integration and web interface (Wave 3)
- `940fdf2` (2025-12-16): feat: Add CI workflow and documentation (Wave 4)
- `1504899` (2025-12-16): chore: Wave 5 integration testing verification
- `590f46f` (2025-12-16): chore: ML data sync
- `be019d3` (2025-12-16): chore: ML data sync
- `a09bd89` (2025-12-16): test: Add unit tests for Cortical Chronicles generators
- `a16f142` (2025-12-16): chore: ML data sync
- `d8ba759` (2025-12-16): chore: ML data sync
- `18ffbcd` (2025-12-16): chore: ML data sync
- `f8a2ad6` (2025-12-16): feat: Add consolidated markdown book generation

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Address critical ML data collection and prediction issues

*Synthesized from commit history: 2025-12-16*

## The Problem

Development work began: Add ML chat data from investigation sessions.

## The Journey

The development progressed through several stages:

1. **Add ML chat data from investigation sessions** - Modified 6 files (+98/-27 lines)
2. **Mock file existence in ML prediction tests** - Modified 2 files (+21/-8 lines)
3. **Address critical ML data collection and prediction issues** - Modified 9 files (+148/-17 lines)


## The Solution

Update ML chat data from orchestration session

The solution involved changes to 14 files, adding 146 lines and removing 6 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 19

- `.git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json`
- `.git-ml/chats/2025-12-15/chat-20251216-120351-efac30.json`
- `.git-ml/chats/2025-12-16/chat-20251216-115057-3617f9.json`
- `.git-ml/chats/2025-12-16/chat-20251216-115057-9502fd.json`
- `.git-ml/chats/2025-12-16/chat-20251216-115057-cbbe64.json`
- `.git-ml/chats/2025-12-16/chat-20251216-115057-f65b7a.json`
- `.git-ml/chats/2025-12-16/chat-20251216-120351-1b48d7.json`
- `.git-ml/chats/2025-12-16/chat-20251216-120351-86411f.json`
- `.git-ml/chats/2025-12-16/chat-20251216-120351-c08c4a.json`
- `.git-ml/chats/2025-12-16/chat-20251216-120351-c1c3b1.json`

*...and 9 more files*

**Code Changes:** +413/-58 lines

**Commits:** 4


## Commits in This Story

- `c08cc75` (2025-12-16): chore: Add ML chat data from investigation sessions
- `ec8db7a` (2025-12-16): fix(tests): Mock file existence in ML prediction tests
- `fead1c1` (2025-12-16): fix: Address critical ML data collection and prediction issues
- `a304a1d` (2025-12-16): chore: Update ML chat data from orchestration session

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Archive ML session after transcript processing (T-003 16f3)

*Synthesized from commit history: 2025-12-16*

## The Problem

Development work began: ML data from director review session.

## The Journey

The solution was implemented directly.


## The Solution

Archive ML session after transcript processing (T-003 16f3)

The solution involved changes to 1 files, adding 12 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 13

- `.git-ml/actions/2025-12-16/A-20251216-132552-2869-000.json`
- `.git-ml/actions/2025-12-16/A-20251216-132552-2869-001.json`
- `.git-ml/actions/2025-12-16/A-20251216-132552-2869-002.json`
- `.git-ml/actions/2025-12-16/A-20251216-132827-2869-000.json`
- `.git-ml/actions/2025-12-16/A-20251216-132827-2869-001.json`
- `.git-ml/actions/2025-12-16/A-20251216-132827-2869-002.json`
- `.git-ml/chats/2025-12-16/chat-20251216-132552-3c1c3c.json`
- `.git-ml/chats/2025-12-16/chat-20251216-132552-8b8c20.json`
- `.git-ml/chats/2025-12-16/chat-20251216-132827-0135aa.json`
- `.git-ml/chats/2025-12-16/chat-20251216-132827-5d9c33.json`

*...and 3 more files*

**Code Changes:** +241/-1 lines

**Commits:** 2


## Commits in This Story

- `10036d1` (2025-12-16): chore: ML data from director review session
- `59072c8` (2025-12-16): fix: Archive ML session after transcript processing (T-003 16f3)

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Fix ML data collection milestone counting and add session/action capture

*Synthesized from commit history: 2025-12-16*

## The Problem

Development work began: Update ML chat data from investigation session.

## The Journey

The development progressed through several stages:

1. **Update ML chat data from investigation session** - Modified 22 files (+182/-33 lines)
2. **Update ML chat data from orchestration session** - Modified 14 files (+146/-6 lines)
3. **Fix ML data collection milestone counting and add session/action capture** - Modified 11 files (+95/-29 lines)
4. **ML data from session with new action/session collection** - Modified 18 files (+319/-1 lines)


## The Solution

Batch task distribution implementation via Director orchestration

The solution involved changes to 8 files, adding 3185 lines and removing 89 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 46

- `.claude/hooks/session_logger.py`
- `.git-ml/actions/2025-12-15/A-20251216-122826-0299-000.json`
- `.git-ml/actions/2025-12-16/A-20251216-122826-0299-001.json`
- `.git-ml/actions/2025-12-16/A-20251216-122826-0299-002.json`
- `.git-ml/actions/2025-12-16/A-20251216-122826-0299-003.json`
- `.git-ml/actions/2025-12-16/A-20251216-122826-0299-004.json`
- `.git-ml/actions/2025-12-16/A-20251216-122826-0299-005.json`
- `.git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json`
- `.git-ml/chats/2025-12-15/chat-20251216-120351-efac30.json`
- `.git-ml/chats/2025-12-15/chat-20251216-121720-30c3c1.json`

*...and 36 more files*

**Code Changes:** +3927/-158 lines

**Commits:** 5


## Commits in This Story

- `ba3a05b` (2025-12-16): chore: Update ML chat data from investigation session
- `a304a1d` (2025-12-16): chore: Update ML chat data from orchestration session
- `273baef` (2025-12-16): fix: Fix ML data collection milestone counting and add session/action capture
- `de8ca40` (2025-12-16): chore: ML data from session with new action/session collection
- `4f915c3` (2025-12-16): feat: Batch task distribution implementation via Director orchestration

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: Bug Fix - Prevent infinite commit loop in ML data collection hooks

*Synthesized from commit history: 2025-12-16*

## The Problem

Development work began: ML tracking data.

## The Journey

The development progressed through several stages:

1. **ML tracking data** - Modified 2 files (+2/-1 lines)
2. **ML tracking data** - Modified 2 files (+2/-1 lines)
3. **ML tracking data** - Modified 2 files (+2/-1 lines)
4. **Prevent infinite commit loop in ML data collection hooks** - Modified 3 files (+9/-1 lines)
5. **ML tracking data** - Modified 2 files (+2/-1 lines)


## The Solution

ML tracking data

The solution involved changes to 4 files, adding 62 lines and removing 1 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 7

- `.git-ml/chats/2025-12-16/chat-20251216-004054-78b531.json`
- `.git-ml/chats/2025-12-16/chat-20251216-004643-110372.json`
- `.git-ml/chats/2025-12-16/chat-20251216-004643-4911e7.json`
- `.git-ml/chats/2025-12-16/chat-20251216-004643-99bc2f.json`
- `.git-ml/current_session.json`
- `.git-ml/tracked/commits.jsonl`
- `scripts/ml_data_collector.py`

**Code Changes:** +79/-6 lines

**Commits:** 6


## Commits in This Story

- `9abdc28` (2025-12-16): data: ML tracking data
- `f4c0e9f` (2025-12-16): data: ML tracking data
- `49801a4` (2025-12-16): data: ML tracking data
- `66ad656` (2025-12-16): fix: Prevent infinite commit loop in ML data collection hooks
- `c8fc6b4` (2025-12-16): data: ML tracking data
- `b66610a` (2025-12-16): data: ML tracking data

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: ML data sync

*Synthesized from commit history: 2025-12-16*

## The Problem

A new feature was required: Add CI workflow and documentation (Wave 4). This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add CI workflow and documentation (Wave 4)** - Modified 5 files (+1724/-1 lines)
2. **Wave 5 integration testing verification** - Modified 25 files (+227/-88 lines)
3. **ML data sync** - Modified 43 files (+904/-6 lines)
4. **ML data sync** - Modified 44 files (+923/-0 lines)
5. **Add unit tests for Cortical Chronicles generators** - Modified 2 files (+1372/-0 lines)
6. **ML data sync** - Modified 43 files (+922/-0 lines)
7. **ML data sync** - Modified 45 files (+941/-0 lines)
8. **ML data sync** - Modified 46 files (+981/-0 lines)
9. **Add consolidated markdown book generation** - Modified 3 files (+6392/-2 lines)


## The Solution

Capture session data

The solution involved changes to 2 files, adding 2 lines and removing 0 lines.


## The Lesson

**Feature development is iterative.** Breaking work into smaller commits makes it easier to review, test, and debug.

## Technical Details

**Files Modified:** 247

- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-000.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-001.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-002.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-003.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-004.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-005.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-006.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-007.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-008.json`
- `.git-ml/actions/2025-12-16/A-20251216-172758-1540-009.json`

*...and 237 more files*

**Code Changes:** +14388/-97 lines

**Commits:** 10


## Commits in This Story

- `940fdf2` (2025-12-16): feat: Add CI workflow and documentation (Wave 4)
- `1504899` (2025-12-16): chore: Wave 5 integration testing verification
- `590f46f` (2025-12-16): chore: ML data sync
- `be019d3` (2025-12-16): chore: ML data sync
- `a09bd89` (2025-12-16): test: Add unit tests for Cortical Chronicles generators
- `a16f142` (2025-12-16): chore: ML data sync
- `d8ba759` (2025-12-16): chore: ML data sync
- `18ffbcd` (2025-12-16): chore: ML data sync
- `f8a2ad6` (2025-12-16): feat: Add consolidated markdown book generation
- `0b34b17` (2025-12-16): ml: Capture session data

---

*This case study was automatically synthesized from git commit history.*

---

## Case Study: ML tracking data

*Synthesized from commit history: 2025-12-16*

## The Problem

A new feature was required: Add chunked storage for git-friendly ML data. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add chunked storage for git-friendly ML data** - Modified 4 files (+1068/-0 lines)
2. **ML tracking data** - Modified 7 files (+95/-8 lines)
3. **Add ML data collection knowledge transfer document** - Modified 1 files (+340/-0 lines)
4. **ML tracking data** - Modified 7 files (+98/-2 lines)
5. **ML tracking data** - Modified 12 files (+216/-3 lines)
6. **Add missing imports in validate command** - Modified 1 files (+5/-0 lines)


## The Solution

Add ML file prediction model

The solution involved changes to 2 files, adding 1098 lines and removing 0 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 30

- `.git-ml/chats/2025-12-16/chat-20251216-005017-2e0bba.json`
- `.git-ml/chats/2025-12-16/chat-20251216-005017-51f512.json`
- `.git-ml/chats/2025-12-16/chat-20251216-005017-9b8abf.json`
- `.git-ml/chats/2025-12-16/chat-20251216-005017-d826a3.json`
- `.git-ml/chats/2025-12-16/chat-20251216-010357-15ba9d.json`
- `.git-ml/chats/2025-12-16/chat-20251216-010357-279b0c.json`
- `.git-ml/chats/2025-12-16/chat-20251216-010357-a70b9d.json`
- `.git-ml/chats/2025-12-16/chat-20251216-010357-a99a7b.json`
- `.git-ml/chats/2025-12-16/chat-20251216-010357-f3ffe9.json`
- `.git-ml/chats/2025-12-16/chat-20251216-012424-3b0705.json`

*...and 20 more files*

**Code Changes:** +2920/-13 lines

**Commits:** 7


## Commits in This Story

- `0754540` (2025-12-16): feat: Add chunked storage for git-friendly ML data
- `cd7c692` (2025-12-16): data: ML tracking data
- `5d3bf6d` (2025-12-16): docs: Add ML data collection knowledge transfer document
- `5664b51` (2025-12-16): data: ML tracking data
- `07ff40c` (2025-12-16): data: ML tracking data
- `172ad8f` (2025-12-16): fix: Add missing imports in validate command
- `ac549dd` (2025-12-16): feat: Add ML file prediction model

---

*This case study was automatically synthesized from git commit history.*

---

## Synthesized Case Studies

*Case studies automatically synthesized from git commit history*

These stories are reconstructed from related commit sequences, showing how real development work unfolds over time.

---

### [Bug Fix - Update skipped tests for processor/ package refactor](synthesized-2025-12-14-bug-fix---update-skipped-tests.md)

**2 commits** | **10 files**

Split processor.py into modular processor/ package (LEGACY-095)

---

### [Bug Fix - Use heredoc for Python in CI to avoid YAML syntax error](synthesized-2025-12-14-bug-fix---use-heredoc-for-pyth.md)

**3 commits** | **1 files**

Make push trigger explicit for all branches

---

### [Update task ID format test to expect microseconds](synthesized-2025-12-14-update-task-id-format-test-to-.md)

**2 commits** | **2 files**

Add microseconds to task ID to prevent collisions

---

### [Bug Fix - Replace external action with native Python link checker](synthesized-2025-12-14-bug-fix---replace-external-act.md)

**4 commits** | **48 files**

Merge pull request #81 from scrawlsbenches/claude/implement-director-mode-NKbiu

---

### [Bug Fix - Add test file penalty and code stop word filtering to search](synthesized-2025-12-14-bug-fix---add-test-file-penalt.md)

**2 commits** | **33 files**

Director mode batch execution - 6 tasks completed in parallel

---

### [Bug Fix - Harden ML data collector with critical fixes](synthesized-2025-12-15-bug-fix---harden-ml-data-colle.md)

**2 commits** | **4 files**

Add ML data collection infrastructure for project-specific micro-model

---

### [Feature Development - Add schema validation for ML data integrity](synthesized-2025-12-15-feature-development---add-sche.md)

**5 commits** | **4 files**

Add CI status integration for ML outcome tracking

---

### [Bug Fix - Update tests for BM25 default and stop word tokenization](synthesized-2025-12-15-bug-fix---update-tests-for-bm2.md)

**3 commits** | **25 files**

Add Stop hook config and update task tracking

---

### [Bug Fix - Increase ML data retention to 2 years for training milestones](synthesized-2025-12-15-bug-fix---increase-ml-data-ret.md)

**5 commits** | **478 files**

Add privacy features to ML data collection

---

### [Bug Fix - Stop tracking ML commit data files (too large for GitHub)](synthesized-2025-12-15-bug-fix---stop-tracking-ml-com.md)

**6 commits** | **949 files**

Add ML commit data

---

### [Bug Fix - Prevent infinite commit loop in ML data collection hooks](synthesized-2025-12-16-bug-fix---prevent-infinite-com.md)

**6 commits** | **7 files**

ML tracking data

---

### [Bug Fix - Address critical ML data collection and prediction issues](synthesized-2025-12-16-bug-fix---address-critical-ml-.md)

**4 commits** | **19 files**

Add ML chat data from investigation sessions

---

### [Bug Fix - Fix ML data collection milestone counting and add session/action capture](synthesized-2025-12-16-bug-fix---fix-ml-data-collecti.md)

**5 commits** | **46 files**

Update ML chat data from investigation session

---

### [Bug Fix - Archive ML session after transcript processing (T-003 16f3)](synthesized-2025-12-16-bug-fix---archive-ml-session-a.md)

**2 commits** | **13 files**

ML data from director review session

---

### [Refactoring - Migrate to merge-friendly task system and add security tasks](synthesized-2025-12-14-refactoring---migrate-to-merge.md)

**4 commits** | **10 files**

Rename CLAUDE.md.potential to CLAUDE.md

---

### [Refactoring - Split processor.py into modular processor/ package (LEGACY-095)](synthesized-2025-12-14-refactoring---split-processor..md)

**4 commits** | **32 files**

Merge pull request #80 from scrawlsbenches/claude/resume-dog-fooding-9RPIV

---

### [Refactoring - Consolidate ML data to single JSONL files](synthesized-2025-12-15-refactoring---consolidate-ml-d.md)

**4 commits** | **492 files**

Add ML commit data

---

### [Refactoring - Split processor.py into modular processor/ package (LEGACY-095)](synthesized-2025-12-14-refactoring---split-processor..md)

**4 commits** | **32 files**

Add HMAC signature verification for pickle files (SEC-003)

---

### [Feature Development - Add director agent orchestration prompt](synthesized-2025-12-14-feature-development---add-dire.md)

**10 commits** | **52 files**

Add future tasks for text-as-memories integration

---

### [Add session memory and knowledge transfer](synthesized-2025-12-14-add-session-memory-and-knowled.md)

**5 commits** | **49 files**

Add director agent orchestration prompt

---

### [Feature Development - Add session handoff, auto-memory, CI link checker, and tests](synthesized-2025-12-14-feature-development---add-sess.md)

**4 commits** | **48 files**

Add memory system CLI and improve documentation

---

### [Clean up directory structure and queue search relevance fixes](synthesized-2025-12-14-clean-up-directory-structure-a.md)

**5 commits** | **39 files**

Director mode batch execution - 6 tasks completed in parallel

---

### [Feature Development - Add CI status integration for ML outcome tracking](synthesized-2025-12-15-feature-development---add-ci-s.md)

**4 commits** | **4 files**

Add schema validation for ML data integrity

---

### [Feature Development - Add CI status integration for ML outcome tracking](synthesized-2025-12-15-feature-development---add-ci-s.md)

**3 commits** | **4 files**

Add commit-chat session linking for ML training

---

### [Feature Development - Add export, feedback, and quality-report commands to ML collector](synthesized-2025-12-15-feature-development---add-expo.md)

**3 commits** | **14 files**

Add session handoff generator for context preservation

---

### [Feature Development - Add comprehensive delegation command template](synthesized-2025-12-15-feature-development---add-comp.md)

**3 commits** | **28 files**

Add reusable pre-merge sanity check command

---

### [Add ML commit data for previous commit](synthesized-2025-12-15-add-ml-commit-data-for-previou.md)

**10 commits** | **956 files**

Add privacy features to ML data collection

---

### [Bug Fix - Stop tracking ML commit data files (too large for GitHub)](synthesized-2025-12-15-bug-fix---stop-tracking-ml-com.md)

**10 commits** | **957 files**

Add automatic ML data collection on session startup

---

### [Feature Development - Add lightweight commit data for ephemeral environments](synthesized-2025-12-15-feature-development---add-ligh.md)

**10 commits** | **956 files**

Share ML commit data and aggregated patterns in git

---

### [Add ML commit data for previous commit](synthesized-2025-12-15-add-ml-commit-data-for-previou.md)

**10 commits** | **953 files**

Add CI auto-capture and GitHub PR/Issue data collection

---

### [Add ML commit data](synthesized-2025-12-15-add-ml-commit-data.md)

**10 commits** | **484 files**

Add lightweight commit data for ephemeral environments

---

### [ML tracking data](synthesized-2025-12-16-ml-tracking-data.md)

**10 commits** | **34 files**

Add ML stats report to CI pipeline

---

### [ML tracking data](synthesized-2025-12-16-ml-tracking-data.md)

**7 commits** | **30 files**

Add chunked storage for git-friendly ML data

---

### [ML data sync](synthesized-2025-12-16-ml-data-sync.md)

**10 commits** | **218 files**

Add Cortical Chronicles book infrastructure (Wave 1)

---

### [ML data sync](synthesized-2025-12-16-ml-data-sync.md)

**10 commits** | **254 files**

Add content generators for Cortical Chronicles (Wave 2)

---

### [Add unit tests for Cortical Chronicles generators](synthesized-2025-12-16-add-unit-tests-for-cortical-ch.md)

**10 commits** | **254 files**

Add search integration and web interface (Wave 3)

---

### [ML data sync](synthesized-2025-12-16-ml-data-sync.md)

**10 commits** | **247 files**

Add CI workflow and documentation (Wave 4)

---

---

# 06 Lessons

## Lessons Learned

*What the Cortical Text Processor taught us about building IR systems.*

---

## Overview

Through **51 lessons** extracted from development history, we've learned how to build better search systems. Each bug fixed, each optimization made, and each refactoring completed taught us something valuable.

## Statistics

- **Total Commits Analyzed**: 300
- **Lessons Extracted**: 51

### By Category

- **Performance**: 1 lessons
- **Correctness**: 24 lessons
- **Architecture**: 10 lessons
- **Testing**: 16 lessons

## Lesson Categories

### [Performance Lessons](lessons-performance.md)

How we learned to optimize search and graph algorithms

**1 lessons** from development history.

### [Correctness Lessons](lessons-correctness.md)

Bugs we fixed and edge cases we discovered

**24 lessons** from development history.

### [Architecture Lessons](lessons-architecture.md)

How we evolved the codebase structure

**10 lessons** from development history.

### [Testing Lessons](lessons-testing.md)

What we learned about verifying correctness

**16 lessons** from development history.

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Architecture Lessons

*How we evolved the code structure over time.*

---

## Overview

This chapter captures **10 lessons** from architecture work. Each entry shows the problem, the solution, and the principle we extracted.

### Complete legacy task system cleanup

**Commit:** `8dedda6`  
**Date:** 2025-12-16  
**Files Changed:** 4  
  - `docs/archive/migrate_legacy_tasks.py`
  - `scripts/select_task.py`
  - `scripts/task_graph.py`
  - *(and 1 more)*

**The Lesson:** Maintain clear structure. The lesson? Complete legacy task system cleanup

### Feat: Add Cortical Chronicles book infrastructure (Wave 1)

**Commit:** `c730057`  
**Date:** 2025-12-16  
**Files Changed:** 13  
  - `.git-ml/current_session.json`
  - `.git-ml/tracked/commits.jsonl`
  - `book/00-preface/.gitkeep`
  - *(and 10 more)*
**Changes:** +434/-0 lines  

**The Lesson:** Maintain clear structure. The lesson? feat: Add Cortical Chronicles book infrastructure (Wave 1)

### Remove unused protobuf serialization (T-013 f0ff)

**Commit:** `d7a98ae`  
**Date:** 2025-12-16  
**Files Changed:** 6  
  - `cortical/persistence.py`
  - `cortical/proto/__init__.py`
  - `cortical/proto/schema.proto`
  - *(and 3 more)*
**Changes:** +100/-1460 lines  

**The Lesson:** Maintain clear structure. The lesson? Remove unused protobuf serialization (T-013 f0ff)

### Data: Add orchestration extraction data from quality audit session

**Commit:** `c85c668`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `.git-ml/tracked/orchestration.jsonl`
**Changes:** +1/-0 lines  

**The Lesson:** Keep modules focused. The lesson? data: Add orchestration extraction data from quality audit session

### Data: Add orchestration extraction data for ML training

**Commit:** `bb75148`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `.git-ml/orchestration/05c8c5d9-75c6-4267-9fef-1d5573ba201b_orchestration.json`
**Changes:** +66/-0 lines  

**The Lesson:** Keep modules focused. The lesson? data: Add orchestration extraction data for ML training

### Feat: Add orchestration extraction for director sub-agent tracking

**Commit:** `4eaeb37`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `.gitignore`
  - `scripts/ml_collector/__init__.py`
  - `scripts/ml_collector/orchestration.py`
  - *(and 1 more)*

**The Lesson:** Keep modules focused. The lesson? feat: Add orchestration extraction for director sub-agent tracking

### Split large files exceeding 25000 token limit

**Commit:** `21ec5ea`  
**Date:** 2025-12-15  
**Files Changed:** 36  
  - `.refactor-backup/BACKUP_PLAN.md`
  - `.refactor-backup/analysis.py`
  - `.refactor-backup/ml_data_collector.py`
  - *(and 33 more)*

**The Lesson:** Keep modules focused. The lesson? Split large files exceeding 25000 token limit

### Consolidate ML data to single JSONL files

**Commit:** `205fe34`  
**Date:** 2025-12-15  
**Files Changed:** 486  
  - `.git-ml/commits-lite/0039ad5b13fb_2025-12-11.json`
  - `.git-ml/commits-lite/00f88d48ab42_2025-12-14.json`
  - `.git-ml/commits-lite/051d20028ddd_2025-12-13.json`
  - *(and 483 more)*
**Changes:** +658/-12208 lines  

**The Lesson:** Maintain clear structure. The lesson? Consolidate ML data to single JSONL files

### Feat: Add ML data collection infrastructure for project-specific micro-model

**Commit:** `1568f3c`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `.claude/hooks/session_logger.py`
  - `.claude/skills/ml-logger/SKILL.md`
  - `.gitignore`
  - *(and 1 more)*
**Changes:** +1039/-0 lines  

**The Lesson:** Maintain clear structure. The lesson? feat: Add ML data collection infrastructure for project-specific micro-model

### Clean up directory structure and queue search relevance fixes

**Commit:** `cd8b9f5`  
**Date:** 2025-12-14  
**Files Changed:** 5  
  - `IMPLEMENTATION_SUMMARY.md`
  - `docs/PATTERN_DETECTION_GUIDE.md`
  - `examples/demo_pattern_detection.py`
  - *(and 2 more)*
**Changes:** +68/-293 lines  

**The Lesson:** Maintain clear structure. The lesson? Clean up directory structure and queue search relevance fixes

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Correctness Lessons

*Bugs we encountered and how we fixed them.*

---

## Overview

This chapter captures **24 lessons** from correctness work. Each entry shows the problem, the solution, and the principle we extracted.

### Archive ML session after transcript processing (T-003 16f3)

**Commit:** `59072c8`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `scripts/ml_data_collector.py`
**Changes:** +12/-0 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Archive ML session after transcript processing (T-003 16f3)

### Update CSV truncation test for new defaults (input=500, output=2000)

**Commit:** `ca94a01`  
**Date:** 2025-12-16  
**Files Changed:** 4  
  - `.git-ml/chats/2025-12-16/chat-20251216-125311-0ce6d9.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-132048-ba08bf.json`
  - `.git-ml/tracked/commits.jsonl`
  - *(and 1 more)*

**The Lesson:** Verify assumptions with tests. The wisdom: Update CSV truncation test for new defaults (input=500, output=2000)

### Fix ML data collection milestone counting and add session/action capture

**Commit:** `273baef`  
**Date:** 2025-12-16  
**Files Changed:** 11  
  - `.git-ml/chats/2025-12-15/chat-20251216-121720-30c3c1.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-121720-01077d.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-121720-306450.json`
  - *(and 8 more)*
**Changes:** +95/-29 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Fix ML data collection milestone counting and add session/action capture

### Address critical ML data collection and prediction issues

**Commit:** `fead1c1`  
**Date:** 2025-12-16  
**Files Changed:** 9  
  - `.git-ml/chats/2025-12-15/chat-20251216-115057-b5bb48.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-115057-3617f9.json`
  - `.git-ml/chats/2025-12-16/chat-20251216-115057-9502fd.json`
  - *(and 6 more)*
**Changes:** +148/-17 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Address critical ML data collection and prediction issues

### Fix(proto): Make protobuf loading lazy to fix CI smoke test failures

**Commit:** `a93518f`  
**Date:** 2025-12-16  
**Files Changed:** 2  
  - `cortical/proto/__init__.py`
  - `cortical/proto/serialization.py`
**Changes:** +53/-19 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: fix(proto): Make protobuf loading lazy to fix CI smoke test failures

### Add missing imports in validate command

**Commit:** `172ad8f`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `scripts/ml_data_collector.py`
**Changes:** +5/-0 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Add missing imports in validate command

### Clean up gitignore pattern for .git-ml/commits/

**Commit:** `a65d54f`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `.gitignore`
**Changes:** +2/-1 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Clean up gitignore pattern for .git-ml/commits/

### Prevent infinite commit loop in ML data collection hooks

**Commit:** `66ad656`  
**Date:** 2025-12-16  
**Files Changed:** 3  
  - `.git-ml/chats/2025-12-16/chat-20251216-004054-78b531.json`
  - `.git-ml/tracked/commits.jsonl`
  - `scripts/ml_data_collector.py`
**Changes:** +9/-1 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Prevent infinite commit loop in ML data collection hooks

### Correct hook format in settings.local.json

**Commit:** `19ac02a`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `.claude/settings.local.json`
**Changes:** +14/-4 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Correct hook format in settings.local.json

### Use filename-based sorting for deterministic session ordering

**Commit:** `61d502d`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `scripts/session_context.py`
  - `tests/unit/test_session_context.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Use filename-based sorting for deterministic session ordering

### Increase ID suffix length to prevent collisions

**Commit:** `8ac4b6b`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `scripts/orchestration_utils.py`
  - `tests/unit/test_orchestration_utils.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Increase ID suffix length to prevent collisions

### Add import guards for optional test dependencies

**Commit:** `91ffb04`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `tests/security/test_fuzzing.py`
  - `tests/test_mcp_server.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Add import guards for optional test dependencies

### Make session file sorting stable for deterministic ordering

**Commit:** `7433b36`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `scripts/session_context.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Make session file sorting stable for deterministic ordering

### Feat(LEGACY-130): Expand customer service corpus and fix xfailed tests

**Commit:** `7f9664d`  
**Date:** 2025-12-15  
**Files Changed:** 6  
  - `samples/customer_service/complaint_escalation_procedures.txt`
  - `samples/customer_service/empathy_and_active_listening.txt`
  - `samples/customer_service/refund_request_handling.txt`
  - *(and 3 more)*

**The Lesson:** Verify assumptions with tests. The wisdom: feat(LEGACY-130): Expand customer service corpus and fix xfailed tests

### Cap query expansion weights to prevent term domination

**Commit:** `fecd6dc`  
**Date:** 2025-12-15  
**Files Changed:** 3  
  - `cortical/query/expansion.py`
  - `tests/behavioral/test_customer_service_quality.py`
  - `tests/unit/test_query_expansion.py`

**The Lesson:** Verify assumptions with tests. The wisdom: Cap query expansion weights to prevent term domination

### Add YAML frontmatter to slash commands for discovery

**Commit:** `5b52da2`  
**Date:** 2025-12-15  
**Files Changed:** 7  
  - `.claude/commands/delegate.md`
  - `.claude/commands/director.md`
  - `.claude/commands/knowledge-transfer.md`
  - *(and 4 more)*

**The Lesson:** Verify assumptions with tests. The wisdom: Add YAML frontmatter to slash commands for discovery

### Stop tracking ML commit data files (too large for GitHub)

**Commit:** `a6f39e0`  
**Date:** 2025-12-15  
**Files Changed:** 472  
  - `.git-ml/commits/0039ad5b_2025-12-11_24b1b10a.json`
  - `.git-ml/commits/00f88d48_2025-12-14_8749d448.json`
  - `.git-ml/commits/051d2002_2025-12-13_7896a312.json`
  - *(and 469 more)*
**Changes:** +4/-2263268 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Stop tracking ML commit data files (too large for GitHub)

### Increase ML data retention to 2 years for training milestones

**Commit:** `95e9f06`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `README.md`
  - `scripts/ml_data_collector.py`
**Changes:** +7/-5 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Increase ML data retention to 2 years for training milestones

### Update tests for BM25 default and stop word tokenization

**Commit:** `9dc7268`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `tests/unit/test_processor_core.py`
  - `tests/unit/test_query_search.py`
**Changes:** +23/-10 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Update tests for BM25 default and stop word tokenization

### Address audit findings and add documentation

**Commit:** `36be3a1`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `.claude/commands/ml-log.md`
  - `.claude/commands/ml-stats.md`
  - `CLAUDE.md`
  - *(and 1 more)*
**Changes:** +201/-15 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Address audit findings and add documentation

### Harden ML data collector with critical fixes

**Commit:** `4438d60`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `scripts/ml_data_collector.py`
**Changes:** +151/-54 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Harden ML data collector with critical fixes

### Correct line number assertions in pattern detection tests

**Commit:** `1b9901d`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tests/unit/test_patterns.py`
**Changes:** +5/-5 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Correct line number assertions in pattern detection tests

### Add test file penalty and code stop word filtering to search

**Commit:** `1fafc8b`  
**Date:** 2025-12-14  
**Files Changed:** 3  
  - `cortical/processor/query_api.py`
  - `cortical/query/passages.py`
  - `cortical/query/search.py`
**Changes:** +51/-9 lines  

**The Lesson:** Verify assumptions with tests. The wisdom: Add test file penalty and code stop word filtering to search

### Replace external action with native Python link checker

**Commit:** `901a181`  
**Date:** 2025-12-14  
**Files Changed:** 5  
  - `.github/workflows/ci.yml`
  - `.markdown-link-check.json`
  - `scripts/resolve_wiki_links.py`
  - *(and 2 more)*
**Changes:** +172/-34 lines  

**The Lesson:** Validate inputs early. The lesson? Replace external action with native Python link checker

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Performance Lessons

*What we learned about making the system fast and efficient.*

---

## Overview

This chapter captures **1 lessons** from performance work. Each entry shows the problem, the solution, and the principle we extracted.

### Feat: Optimize compute_all and add Graph-Boosted search (GB-BM25)

**Commit:** `fcce0c2`  
**Date:** 2025-12-15  
**Files Changed:** 5  
  - `cortical/analysis.py`
  - `cortical/processor/query_api.py`
  - `cortical/query/__init__.py`
  - *(and 2 more)*
**Changes:** +244/-62 lines  

**The Lesson:** Optimize based on evidence. The lesson? feat: Optimize compute_all and add Graph-Boosted search (GB-BM25)

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

## Testing Lessons

*Insights from writing and maintaining tests.*

---

## Overview

This chapter captures **16 lessons** from testing work. Each entry shows the problem, the solution, and the principle we extracted.

### Add unit tests for Cortical Chronicles generators

**Commit:** `a09bd89`  
**Date:** 2025-12-16  
**Files Changed:** 2  
  - `.git-ml/tracked/commits.jsonl`
  - `tests/unit/test_generate_book.py`
**Changes:** +1372/-0 lines  

**The Lesson:** Test what you build. The lesson? Add unit tests for Cortical Chronicles generators

### Fix(tests): Mock file existence in ML prediction tests

**Commit:** `ec8db7a`  
**Date:** 2025-12-16  
**Files Changed:** 2  
  - `tests/unit/test_ml_file_prediction.py`
  - `tests/unit/test_protobuf_serialization.py`
**Changes:** +21/-8 lines  

**The Lesson:** Mock external dependencies. The wisdom: fix(tests): Mock file existence in ML prediction tests

### Fix(tests): Mock file existence in ML prediction tests

**Commit:** `4f7e195`  
**Date:** 2025-12-16  
**Files Changed:** 1  
  - `tests/unit/test_ml_file_prediction.py`
**Changes:** +10/-6 lines  

**The Lesson:** Mock external dependencies. The wisdom: fix(tests): Mock file existence in ML prediction tests

### Add comprehensive test suite for orchestration.py (33 tests)

**Commit:** `d999c84`  
**Date:** 2025-12-15  
**Files Changed:** 2  
  - `CLAUDE.md`
  - `tests/unit/test_ml_orchestration.py`
**Changes:** +801/-1 lines  

**The Lesson:** Test what you build. The lesson? Add comprehensive test suite for orchestration.py (33 tests)

### Update README test count (3800+) and coverage badge (>90%)

**Commit:** `4ec93d5`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `README.md`

**The Lesson:** Measure coverage to find gaps. The lesson? Update README test count (3800+) and coverage badge (>90%)

### Update task status for Wave 4 completed coverage tests (ALL COMPLETE)

**Commit:** `3b9a071`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tasks/2025-12-15_05-23-36_ceac.json`

**The Lesson:** Measure coverage to find gaps. The lesson? Update task status for Wave 4 completed coverage tests (ALL COMPLETE)

### Feat: Add comprehensive test coverage for Wave 4 modules (FINAL)

**Commit:** `73d6da8`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `tests/unit/test_code_concepts_coverage.py`
  - `tests/unit/test_diff_coverage.py`
  - `tests/unit/test_fluent_coverage.py`
  - *(and 1 more)*

**The Lesson:** Measure coverage to find gaps. The lesson? feat: Add comprehensive test coverage for Wave 4 modules (FINAL)

### Update task status for Wave 3 completed coverage tests

**Commit:** `0b2eaf2`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tasks/2025-12-15_05-23-36_ceac.json`

**The Lesson:** Measure coverage to find gaps. The lesson? Update task status for Wave 3 completed coverage tests

### Feat: Add comprehensive test coverage for Wave 3 modules

**Commit:** `036f830`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `tests/unit/test_config_coverage.py`
  - `tests/unit/test_fingerprint_coverage.py`
  - `tests/unit/test_query_chunking.py`
  - *(and 1 more)*

**The Lesson:** Measure coverage to find gaps. The lesson? feat: Add comprehensive test coverage for Wave 3 modules

### Update task status for Wave 2 completed coverage tests

**Commit:** `66f7df2`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tasks/2025-12-15_05-23-36_ceac.json`

**The Lesson:** Measure coverage to find gaps. The lesson? Update task status for Wave 2 completed coverage tests

### Feat: Add comprehensive test coverage for Wave 2 modules

**Commit:** `5a6bb26`  
**Date:** 2025-12-15  
**Files Changed:** 4  
  - `tests/unit/test_embeddings_coverage.py`
  - `tests/unit/test_query_definitions.py`
  - `tests/unit/test_query_passages.py`
  - *(and 1 more)*

**The Lesson:** Measure coverage to find gaps. The lesson? feat: Add comprehensive test coverage for Wave 2 modules

### Update task status for completed coverage tests

**Commit:** `cc147fd`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tasks/2025-12-15_05-23-36_ceac.json`

**The Lesson:** Measure coverage to find gaps. The lesson? Update task status for completed coverage tests

### Feat: Add comprehensive test coverage for query and analysis modules

**Commit:** `70a4b1b`  
**Date:** 2025-12-15  
**Files Changed:** 7  
  - `requirements.txt`
  - `tasks/2025-12-14_01-53-45_7b60.json`
  - `tasks/2025-12-14_11-11-44_legacy-migration.json`
  - *(and 4 more)*

**The Lesson:** Measure coverage to find gaps. The lesson? feat: Add comprehensive test coverage for query and analysis modules

### Add unit tests for ML collector export, feedback, quality commands

**Commit:** `1899ed8`  
**Date:** 2025-12-15  
**Files Changed:** 3  
  - `tests/unit/test_ml_export.py`
  - `tests/unit/test_ml_feedback.py`
  - `tests/unit/test_ml_quality.py`
**Changes:** +1842/-0 lines  

**The Lesson:** Test what you build. The lesson? Add unit tests for ML collector export, feedback, quality commands

### Add 16 code coverage improvement tasks

**Commit:** `d0732b4`  
**Date:** 2025-12-15  
**Files Changed:** 1  
  - `tasks/2025-12-15_05-23-36_ceac.json`
**Changes:** +248/-0 lines  

**The Lesson:** Measure coverage to find gaps. The lesson? Add 16 code coverage improvement tasks

### Merge pull request #85 from scrawlsbenches/claude/fix-coverage-module-82miT

**Commit:** `d09bbce`  
**Date:** 2025-12-15  
**Changes:** +23/-0 lines  

**The Lesson:** Measure coverage to find gaps. The lesson? Merge pull request #85 from scrawlsbenches/claude/fix-coverage-module-82miT

---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*

---

# 07 Concepts

## Concept Evolution: Bigram

*Tracking the emergence and growth of 'bigram' through commit history.*

---

## Birth

**First Appearance:** 2025-12-09

The concept of 'bigram' first emerged in commit `50450d1`:

> Add bigram lateral connections (Task 21) and code review concerns

*By Claude*

## Growth Timeline

The concept has been mentioned in **7 commits** across **1 months** of development.

### December 2025: Emergence

**7 commits** mentioning this concept.

- `50450d1`: Add bigram lateral connections (Task 21) and code review concerns
- `18f45ef`: Optimize semantic extraction and bigram connections (2x speedup)
- `0f578c3`: Add expert code review: identify critical bigram bug and verify fixes
- *(and 4 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **7 commits**.

## Related Concepts

The 'bigram' concept frequently appears alongside:

- **Lateral Connections** (1 co-occurrences)
- **Semantic** (1 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-11:

> Add tests for bigram connection parameters and improve coverage to 90%

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `50450d1`

**Date:** 2025-12-09

**Message:** Add bigram lateral connections (Task 21) and code review concerns

### Midpoint: `17e8147`

**Date:** 2025-12-10

**Message:** Add code review findings: critical bigram separator bugs

### Latest: `1b9438e`

**Date:** 2025-12-11

**Message:** Add tests for bigram connection parameters and improve coverage to 90%

---

## Concept Evolution: Bm25

*Tracking the emergence and growth of 'bm25' through commit history.*

---

## Birth

**First Appearance:** 2025-12-15

The concept of 'bm25' first emerged in commit `0a52858`:

> feat: Implement BM25 scoring algorithm as default

*By Claude*

## Growth Timeline

The concept has been mentioned in **9 commits** across **1 months** of development.

### December 2025: Emergence

**9 commits** mentioning this concept.

- `0a52858`: feat: Implement BM25 scoring algorithm as default
- `fcce0c2`: feat: Optimize compute_all and add Graph-Boosted search (GB-BM25)
- `63064c7`: docs: Add BM25/GB-BM25 documentation and tests
- *(and 6 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W50** with **9 commits**.

## Related Concepts

The 'bm25' concept frequently appears alongside:

- **Search** (2 co-occurrences)
- **Graph** (1 co-occurrences)
- **Tokenization** (1 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-16:

> feat: Add chunked parallel processing for TF-IDF/BM25 (LEGACY-135)

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `0a52858`

**Date:** 2025-12-15

**Message:** feat: Implement BM25 scoring algorithm as default

### Midpoint: `9dc7268`

**Date:** 2025-12-15

**Message:** fix: Update tests for BM25 default and stop word tokenization

### Latest: `5665839`

**Date:** 2025-12-16

**Message:** feat: Add chunked parallel processing for TF-IDF/BM25 (LEGACY-135)

---

## Concept Evolution: Clustering

*Tracking the emergence and growth of 'clustering' through commit history.*

---

## Birth

**First Appearance:** 2025-12-09

The concept of 'clustering' first emerged in commit `bf75e5d`:

> Activate Layer 2 concept clustering by default (Task 10)

*By Claude*

## Growth Timeline

The concept has been mentioned in **7 commits** across **1 months** of development.

### December 2025: Emergence

**7 commits** mentioning this concept.

- `bf75e5d`: Activate Layer 2 concept clustering by default (Task 10)
- `e7933b6`: Implement Task 4: Improve clustering to reduce topic isolation
- `0d24482`: Fix tests to not skip - provide sufficient data for concept clustering
- *(and 4 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **7 commits**.

## The Concept Today

Most recent mention was on 2025-12-12:

> Task #143: Investigate negative silhouette score in clustering

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `bf75e5d`

**Date:** 2025-12-09

**Message:** Activate Layer 2 concept clustering by default (Task 10)

### Midpoint: `bda9504`

**Date:** 2025-12-11

**Message:** Add critical clustering tasks #123-125 and regression tests (Task #124)

### Latest: `2753132`

**Date:** 2025-12-12

**Message:** Task #143: Investigate negative silhouette score in clustering

---

## Concept Evolution: Context

*Tracking the emergence and growth of 'context' through commit history.*

---

## Birth

**First Appearance:** 2025-12-11

The concept of 'context' first emerged in commit `a530bea`:

> Add CLI wrapper framework for context collection and task triggers

*By Claude*

## Growth Timeline

The concept has been mentioned in **3 commits** across **1 months** of development.

### December 2025: Emergence

**3 commits** mentioning this concept.

- `a530bea`: Add CLI wrapper framework for context collection and task triggers
- `4e10104`: Merge pull request #37 from scrawlsbenches/claude/cli-wrapper-context-01JScUxQPSb4rGC2XhtXPSYB
- `9bd4067`: feat: Add session handoff generator for context preservation

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **2 commits**.

## The Concept Today

Most recent mention was on 2025-12-15:

> feat: Add session handoff generator for context preservation

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `a530bea`

**Date:** 2025-12-11

**Message:** Add CLI wrapper framework for context collection and task triggers

### Midpoint: `4e10104`

**Date:** 2025-12-11

**Message:** Merge pull request #37 from scrawlsbenches/claude/cli-wrapper-context-01JScUxQPSb4rGC2XhtXPSYB

### Latest: `9bd4067`

**Date:** 2025-12-15

**Message:** feat: Add session handoff generator for context preservation

---

## Concept Evolution: Definition

*Tracking the emergence and growth of 'definition' through commit history.*

---

## Birth

**First Appearance:** 2025-12-11

The concept of 'definition' first emerged in commit `60c3483`:

> Add direct definition pattern search for code search (Task #84)

*By Claude*

## Growth Timeline

The concept has been mentioned in **4 commits** across **1 months** of development.

### December 2025: Emergence

**4 commits** mentioning this concept.

- `60c3483`: Add direct definition pattern search for code search (Task #84)
- `66a4078`: Merge main, add task #128 for definition boost search quality issue
- `d85cc90`: Fix definition boost to deprioritize test files over real implementations (Task #128)
- *(and 1 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **4 commits**.

## Related Concepts

The 'definition' concept frequently appears alongside:

- **Search** (2 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-11:

> Mark Tasks #128, #132, #136 complete - definition boost and O(n²) fixes

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `60c3483`

**Date:** 2025-12-11

**Message:** Add direct definition pattern search for code search (Task #84)

### Midpoint: `d85cc90`

**Date:** 2025-12-11

**Message:** Fix definition boost to deprioritize test files over real implementations (Task #128)

### Latest: `0689785`

**Date:** 2025-12-11

**Message:** Mark Tasks #128, #132, #136 complete - definition boost and O(n²) fixes

---

## Concept Evolution: Embeddings

*Tracking the emergence and growth of 'embeddings' through commit history.*

---

## Birth

**First Appearance:** 2025-12-09

The concept of 'embeddings' first emerged in commit `8f862b0`:

> Persist full computed state including embeddings (Task 12)

*By Claude*

## Growth Timeline

The concept has been mentioned in **5 commits** across **1 months** of development.

### December 2025: Emergence

**5 commits** mentioning this concept.

- `8f862b0`: Persist full computed state including embeddings (Task 12)
- `6cb35d6`: Add critical task #122: Investigate Concept Layer & Embeddings regressions
- `919e8a7`: Fix cluster_strictness inversion and improve embeddings (Task #122)
- *(and 2 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **5 commits**.

## The Concept Today

Most recent mention was on 2025-12-13:

> Fix test_retrofit_embeddings_invalid_alpha_zero to match new validation

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `8f862b0`

**Date:** 2025-12-09

**Message:** Persist full computed state including embeddings (Task 12)

### Midpoint: `919e8a7`

**Date:** 2025-12-11

**Message:** Fix cluster_strictness inversion and improve embeddings (Task #122)

### Latest: `69c206b`

**Date:** 2025-12-13

**Message:** Fix test_retrofit_embeddings_invalid_alpha_zero to match new validation

---

## Concept Evolution: Graph

*Tracking the emergence and growth of 'graph' through commit history.*

---

## Birth

**First Appearance:** 2025-12-10

The concept of 'graph' first emerged in commit `e40a80c`:

> Add ConceptNet-style graph visualization export (Task 29)

*By Claude*

## Growth Timeline

The concept has been mentioned in **3 commits** across **1 months** of development.

### December 2025: Emergence

**3 commits** mentioning this concept.

- `e40a80c`: Add ConceptNet-style graph visualization export (Task 29)
- `e5dd3d5`: Task #145: Improve graph embedding quality for common terms
- `fcce0c2`: feat: Optimize compute_all and add Graph-Boosted search (GB-BM25)

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **2 commits**.

## Related Concepts

The 'graph' concept frequently appears alongside:

- **Bm25** (1 co-occurrences)
- **Search** (1 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-15:

> feat: Optimize compute_all and add Graph-Boosted search (GB-BM25)

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `e40a80c`

**Date:** 2025-12-10

**Message:** Add ConceptNet-style graph visualization export (Task 29)

### Midpoint: `e5dd3d5`

**Date:** 2025-12-12

**Message:** Task #145: Improve graph embedding quality for common terms

### Latest: `fcce0c2`

**Date:** 2025-12-15

**Message:** feat: Optimize compute_all and add Graph-Boosted search (GB-BM25)

---

## Concept Evolution: Incremental

*Tracking the emergence and growth of 'incremental' through commit history.*

---

## Birth

**First Appearance:** 2025-12-09

The concept of 'incremental' first emerged in commit `38fb4f7`:

> Add incremental document indexing (Task 15)

*By Claude*

## Growth Timeline

The concept has been mentioned in **5 commits** across **1 months** of development.

### December 2025: Emergence

**5 commits** mentioning this concept.

- `38fb4f7`: Add incremental document indexing (Task 15)
- `3682739`: Add incremental codebase indexing with progress tracking
- `b360793`: Update documentation for incremental indexing features
- *(and 2 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **5 commits**.

## The Concept Today

Most recent mention was on 2025-12-11:

> Add incremental batch mode for full analysis

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `38fb4f7`

**Date:** 2025-12-09

**Message:** Add incremental document indexing (Task 15)

### Midpoint: `b360793`

**Date:** 2025-12-10

**Message:** Update documentation for incremental indexing features

### Latest: `256e842`

**Date:** 2025-12-11

**Message:** Add incremental batch mode for full analysis

---

## Concept Evolution Index

*A guide to how key concepts emerged and grew in the Cortical Text Processor.*

---

## Overview

This section tracks the evolution of **14 core concepts** through the project's commit history. Each concept chapter shows:

- When the concept first appeared
- How it grew over time
- Related concepts and connections
- Current state and importance

## Concepts by Importance

### [Search](search.md)

**Mentions:** 26 commits

**First seen:** 2025-12-10

**Related to:** Passage, Definition, Louvain

*A core component of the system architecture.*

### [Semantic](semantic.md)

**Mentions:** 9 commits

**First seen:** 2025-12-09

**Related to:** Retrieval, Bigram, Fingerprint

*An emerging concept in recent development.*

### [Bm25](bm25.md)

**Mentions:** 9 commits

**First seen:** 2025-12-15

**Related to:** Search, Graph, Tokenization

*An emerging concept in recent development.*

### [Clustering](clustering.md)

**Mentions:** 7 commits

**First seen:** 2025-12-09

*An emerging concept in recent development.*

### [Bigram](bigram.md)

**Mentions:** 7 commits

**First seen:** 2025-12-09

**Related to:** Lateral Connections, Semantic

*An emerging concept in recent development.*

### [Louvain](louvain.md)

**Mentions:** 7 commits

**First seen:** 2025-12-11

**Related to:** Search

*An emerging concept in recent development.*

### [Embeddings](embeddings.md)

**Mentions:** 5 commits

**First seen:** 2025-12-09

*An emerging concept in recent development.*

### [Incremental](incremental.md)

**Mentions:** 5 commits

**First seen:** 2025-12-09

*An emerging concept in recent development.*

### [Pagerank](pagerank.md)

**Mentions:** 4 commits

**First seen:** 2025-12-09

*An emerging concept in recent development.*

### [Query Expansion](query-expansion.md)

**Mentions:** 4 commits

**First seen:** 2025-12-10

*An emerging concept in recent development.*

### [Definition](definition.md)

**Mentions:** 4 commits

**First seen:** 2025-12-11

**Related to:** Search

*An emerging concept in recent development.*

### [Graph](graph.md)

**Mentions:** 3 commits

**First seen:** 2025-12-10

**Related to:** Bm25, Search

*An emerging concept in recent development.*

### [Tokenization](tokenization.md)

**Mentions:** 3 commits

**First seen:** 2025-12-10

**Related to:** Bm25

*An emerging concept in recent development.*

### [Context](context.md)

**Mentions:** 3 commits

**First seen:** 2025-12-11

*An emerging concept in recent development.*

---

*Each concept chapter provides detailed evolution timeline and key commits.*

---

## Concept Evolution: Louvain

*Tracking the emergence and growth of 'louvain' through commit history.*

---

## Birth

**First Appearance:** 2025-12-11

The concept of 'louvain' first emerged in commit `62c7fdf`:

> Implement Louvain community detection (Task #123)

*By Claude*

## Growth Timeline

The concept has been mentioned in **7 commits** across **1 months** of development.

### December 2025: Emergence

**7 commits** mentioning this concept.

- `62c7fdf`: Implement Louvain community detection (Task #123)
- `b2b7f92`: Add Task #126: Investigate optimal Louvain resolution
- `e85c299`: Merge pull request #34 from scrawlsbenches/claude/louvain-community-detection-01FsvWk3GKjFLpEiPwQT4sBc
- *(and 4 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **7 commits**.

## Related Concepts

The 'louvain' concept frequently appears alongside:

- **Search** (2 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-13:

> Add louvain_resolution parameter to CorticalConfig

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `62c7fdf`

**Date:** 2025-12-11

**Message:** Implement Louvain community detection (Task #123)

### Midpoint: `dda7d0c`

**Date:** 2025-12-11

**Message:** Complete Task #126: Louvain resolution parameter research

### Latest: `a47bb61`

**Date:** 2025-12-13

**Message:** Add louvain_resolution parameter to CorticalConfig

---

## Concept Evolution: Pagerank

*Tracking the emergence and growth of 'pagerank' through commit history.*

---

## Birth

**First Appearance:** 2025-12-09

The concept of 'pagerank' first emerged in commit `c6eefdc`:

> Add ConceptNet-enhanced PageRank task list (Tasks 19-30)

*By Claude*

## Growth Timeline

The concept has been mentioned in **4 commits** across **1 months** of development.

### December 2025: Emergence

**4 commits** mentioning this concept.

- `c6eefdc`: Add ConceptNet-enhanced PageRank task list (Tasks 19-30)
- `f6b8389`: Add relation-weighted PageRank (Task 22)
- `cc57677`: Add cross-layer PageRank propagation (Task 23)
- *(and 1 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **4 commits**.

## The Concept Today

Most recent mention was on 2025-12-10:

> Add overlapping PageRank, ConceptNet, and Neocortex samples

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `c6eefdc`

**Date:** 2025-12-09

**Message:** Add ConceptNet-enhanced PageRank task list (Tasks 19-30)

### Midpoint: `cc57677`

**Date:** 2025-12-10

**Message:** Add cross-layer PageRank propagation (Task 23)

### Latest: `1e4d35d`

**Date:** 2025-12-10

**Message:** Add overlapping PageRank, ConceptNet, and Neocortex samples

---

## Concept Evolution: Query Expansion

*Tracking the emergence and growth of 'query expansion' through commit history.*

---

## Birth

**First Appearance:** 2025-12-10

The concept of 'query expansion' first emerged in commit `16c13a0`:

> Document magic numbers and extract query expansion helper

*By Claude*

## Growth Timeline

The concept has been mentioned in **4 commits** across **1 months** of development.

### December 2025: Emergence

**4 commits** mentioning this concept.

- `16c13a0`: Document magic numbers and extract query expansion helper
- `a819131`: Add LRU cache for query expansion results (Task #45)
- `af3a7e0`: feat: Add security concept group and TF-IDF weighted query expansion
- *(and 1 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **2 commits**.

## The Concept Today

Most recent mention was on 2025-12-15:

> fix: Cap query expansion weights to prevent term domination

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `16c13a0`

**Date:** 2025-12-10

**Message:** Document magic numbers and extract query expansion helper

### Midpoint: `af3a7e0`

**Date:** 2025-12-15

**Message:** feat: Add security concept group and TF-IDF weighted query expansion

### Latest: `fecd6dc`

**Date:** 2025-12-15

**Message:** fix: Cap query expansion weights to prevent term domination

---

## Concept Evolution: Search

*Tracking the emergence and growth of 'search' through commit history.*

---

## Birth

**First Appearance:** 2025-12-10

The concept of 'search' first emerged in commit `dc6db89`:

> Implement dog-fooding: search codebase with its own IR system (Task #47)

*By Claude*

## Growth Timeline

The concept has been mentioned in **26 commits** across **1 months** of development.

### December 2025: Emergence

**26 commits** mentioning this concept.

- `dc6db89`: Implement dog-fooding: search codebase with its own IR system (Task #47)
- `975fc91`: Add intent-based code search enhancement tasks (#48-52)
- `2ad03ed`: Add query optimization for faster code search (Task #52)
- *(and 23 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **20 commits**.

## Related Concepts

The 'search' concept frequently appears alongside:

- **Passage** (2 co-occurrences)
- **Definition** (2 co-occurrences)
- **Louvain** (2 co-occurrences)
- **Bm25** (2 co-occurrences)
- **Intent** (1 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-16:

> feat: Add search integration and web interface (Wave 3)

This concept has evolved from its initial appearance to become a **core component** of the system.

## Key Commits

Notable commits that shaped this concept:

### First: `dc6db89`

**Date:** 2025-12-10

**Message:** Implement dog-fooding: search codebase with its own IR system (Task #47)

### Midpoint: `0f75675`

**Date:** 2025-12-11

**Message:** Add Python code samples and update showcase for code search features

### Latest: `0022466`

**Date:** 2025-12-16

**Message:** feat: Add search integration and web interface (Wave 3)

---

## Concept Evolution: Semantic

*Tracking the emergence and growth of 'semantic' through commit history.*

---

## Birth

**First Appearance:** 2025-12-09

The concept of 'semantic' first emerged in commit `f27d18e`:

> Integrate semantic relations into retrieval (Task 11)

*By Claude*

## Growth Timeline

The concept has been mentioned in **9 commits** across **1 months** of development.

### December 2025: Emergence

**9 commits** mentioning this concept.

- `f27d18e`: Integrate semantic relations into retrieval (Task 11)
- `4e113e7`: Add multi-hop semantic inference (Tasks 25-26)
- `18f45ef`: Optimize semantic extraction and bigram connections (2x speedup)
- *(and 6 more)*

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **7 commits**.

## Related Concepts

The 'semantic' concept frequently appears alongside:

- **Retrieval** (1 co-occurrences)
- **Bigram** (1 co-occurrences)
- **Fingerprint** (1 co-occurrences)
- **Similarity** (1 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-14:

> feat: Add "What Changed?" semantic diff (LEGACY-075)

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `f27d18e`

**Date:** 2025-12-09

**Message:** Integrate semantic relations into retrieval (Task 11)

### Midpoint: `626c008`

**Date:** 2025-12-11

**Message:** Add semantic chunk boundaries for code (Task #86)

### Latest: `a31d1c7`

**Date:** 2025-12-14

**Message:** feat: Add "What Changed?" semantic diff (LEGACY-075)

---

## Concept Evolution: Tokenization

*Tracking the emergence and growth of 'tokenization' through commit history.*

---

## Birth

**First Appearance:** 2025-12-10

The concept of 'tokenization' first emerged in commit `d2b42d9`:

> Fix polysemy section: clarify tokenization vs actual polysemy

*By Claude*

## Growth Timeline

The concept has been mentioned in **3 commits** across **1 months** of development.

### December 2025: Emergence

**3 commits** mentioning this concept.

- `d2b42d9`: Fix polysemy section: clarify tokenization vs actual polysemy
- `2571bb8`: Add code-aware tokenization with identifier splitting (Task #48)
- `9dc7268`: fix: Update tests for BM25 default and stop word tokenization

## Peak Activity

The concept saw its most intensive development during week **2025-W49** with **2 commits**.

## Related Concepts

The 'tokenization' concept frequently appears alongside:

- **Bm25** (1 co-occurrences)

## The Concept Today

Most recent mention was on 2025-12-15:

> fix: Update tests for BM25 default and stop word tokenization

This concept has evolved from its initial appearance to become an **emerging aspect** of the design.

## Key Commits

Notable commits that shaped this concept:

### First: `d2b42d9`

**Date:** 2025-12-10

**Message:** Fix polysemy section: clarify tokenization vs actual polysemy

### Midpoint: `2571bb8`

**Date:** 2025-12-10

**Message:** Add code-aware tokenization with identifier splitting (Task #48)

### Latest: `9dc7268`

**Date:** 2025-12-15

**Message:** fix: Update tests for BM25 default and stop word tokenization

---

# 08 Exercises

## Advanced Exercises

*Hands-on coding exercises to master advanced concepts.*

**Difficulty Level:** Advanced

---

## Introduction

Challenge yourself with advanced features:

- Semantic relation extraction
- Fingerprint-based similarity
- Graph embeddings
- Knowledge gap detection

## Exercise: Empty Documents

**Concept:** Empty documents return no relations

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty documents return no relations.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_documents(self):
        """Empty documents return no relations."""
        result = extract_pattern_relations({}, {"term1", "term2"})
        assert result == []
```

</details>

---

## Exercise: Empty Valid Terms

**Concept:** No valid terms means no relations extracted

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

No valid terms means no relations extracted.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_valid_terms(self):
        """No valid terms means no relations extracted."""
        docs = {"doc1": "A dog is an animal."}
        result = extract_pattern_relations(docs, set())
        assert result == []
```

</details>

---

## Exercise: Empty Relations

**Concept:** Empty relations list

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty relations list.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_relations(self):
        """Empty relations list."""
        result = get_pattern_statistics([])
        assert result["total_relations"] == 0
        assert result["relation_type_counts"] == {}
```

</details>

---

## Exercise: Single Relation

**Concept:** Single relation statistics

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single relation statistics.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_relation(self):
        """Single relation statistics."""
        relations = [("dog", "IsA", "animal", 0.9)]
        result = get_pattern_statistics(relations)
        assert result["total_relations"] == 1
        assert result["relation_type_counts"]["IsA"] == 1
```

</details>

---

## Exercise: Empty Relations

**Concept:** Empty relations produce empty hierarchy

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty relations produce empty hierarchy.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_relations(self):
        """Empty relations produce empty hierarchy."""
        parents, children = build_isa_hierarchy([])
        assert parents == {}
        assert children == {}
```

</details>

---

## Exercise: Single Isa

**Concept:** Single IsA relation creates parent-child

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single IsA relation creates parent-child.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_isa(self):
        """Single IsA relation creates parent-child."""
        relations = [("dog", "IsA", "animal", 0.9)]
        parents, children = build_isa_hierarchy(relations)
        assert "dog" in parents
        assert "animal" in parents["dog"]
        assert "animal" in children
        assert "dog" in children["animal"]
```

</details>

---

## Exercise: Hierarchy Chain

**Concept:** Chain: poodle IsA dog IsA animal

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Chain: poodle IsA dog IsA animal.

### Hints

<details>
<summary>Hint 1</summary>

Think about how elements connect in sequence.

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_hierarchy_chain(self):
        """Chain: poodle IsA dog IsA animal."""
        relations = [
            ("poodle", "IsA", "dog", 0.9),
            ("dog", "IsA", "animal", 0.9)
        ]
        parents, children = build_isa_hierarchy(relations)
        assert "poodle" in parents
        assert "dog" in parents["poodle"]
        assert "dog" in parents
        assert "animal" in parents["dog"]
```

</details>

---

## Exercise: Empty Hierarchy

**Concept:** Empty hierarchy returns empty ancestors

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty hierarchy returns empty ancestors.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_hierarchy(self):
        """Empty hierarchy returns empty ancestors."""
        result = get_ancestors("dog", {})
        assert result == {}
```

</details>

---

## Exercise: Empty Hierarchy

**Concept:** Empty children dict returns empty descendants

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty children dict returns empty descendants.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_hierarchy(self):
        """Empty children dict returns empty descendants."""
        result = get_descendants("animal", {})
        assert result == {}
```

</details>

---

## Exercise: Empty Corpus

**Concept:** Empty corpus returns no relations

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty corpus returns no relations.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_corpus(self):
        """Empty corpus returns no relations."""
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        tokenizer = Tokenizer()
        result = extract_corpus_semantics(layers, {}, tokenizer)
        assert result == []
```

</details>

---

---

*Completed 10 exercises? Check out the other topics for more challenges!*

---

## Foundations Exercises

*Hands-on coding exercises to master foundations concepts.*

**Difficulty Level:** Beginner

---

## Introduction

These exercises cover the fundamental algorithms and data structures of the Cortical Text Processor:

- PageRank for term importance
- TF-IDF for relevance scoring
- Graph structures and connections
- Tokenization and text processing

## Exercise: Empty Graph

**Concept:** Empty graph returns empty dict

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty graph returns empty dict.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_graph(self):
        """Empty graph returns empty dict."""
        result = _pagerank_core({})
        assert result == {}
```

</details>

---

## Exercise: Single Node No Edges

**Concept:** Single node with no edges gets base rank from damping

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single node with no edges gets base rank from damping.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_node_no_edges(self):
        """Single node with no edges gets base rank from damping."""
        graph = {"a": []}
        result = _pagerank_core(graph, damping=0.85)
        assert "a" in result
        # With no incoming edges, rank = (1-d)/n = 0.15/1 = 0.15
        assert result["a"] == pytest.approx(0.15)
```

</details>

---

## Exercise: Single Node Self Loop

**Concept:** Single node with self-loop still gets rank 1.0

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single node with self-loop still gets rank 1.0.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_node_self_loop(self):
        """Single node with self-loop still gets rank 1.0."""
        graph = {"a": [("a", 1.0)]}
        result = _pagerank_core(graph)
        assert result["a"] == pytest.approx(1.0)
```

</details>

---

## Exercise: Three Node Chain

**Concept:** Chain: a -> b -> c. C should have highest rank

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Chain: a -> b -> c. C should have highest rank.

### Hints

<details>
<summary>Hint 1</summary>

Think about how elements connect in sequence.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_three_node_chain(self):
        """Chain: a -> b -> c. C should have highest rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
            "c": []
        }
        result = _pagerank_core(graph)
        # c receives transitively, b receives from a
        assert result["c"] >= result["b"]
        assert result["b"] >= result["a"]
```

</details>

---

## Exercise: Cycle

**Concept:** Cycle: a -> b -> c -> a. All should have equal rank

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Cycle: a -> b -> c -> a. All should have equal rank.

### Hints

<details>
<summary>Hint 1</summary>

Think about how elements connect in sequence.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_cycle(self):
        """Cycle: a -> b -> c -> a. All should have equal rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
            "c": [("a", 1.0)]
        }
        result = _pagerank_core(graph)
        # All nodes in cycle should have equal rank
        assert result["a"] == pytest.approx(result["b"], rel=0.01)
        assert result["b"] == pytest.approx(result["c"], rel=0.01)
```

</details>

---

## Exercise: Empty Corpus

**Concept:** Empty corpus returns empty dict

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty corpus returns empty dict.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_corpus(self):
        """Empty corpus returns empty dict."""
        result = _tfidf_core({}, num_docs=0)
        assert result == {}
```

</details>

---

## Exercise: Single Term Single Doc

**Concept:** Single term in single doc has IDF of 0

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single term in single doc has IDF of 0.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_single_doc(self):
        """Single term in single doc has IDF of 0."""
        stats = {
            "term": (5, 1, {"doc1": 5})
        }
        result = _tfidf_core(stats, num_docs=1)
        # IDF = log(1/1) = 0, so TF-IDF = 0
        assert result["term"][0] == pytest.approx(0.0)
```

</details>

---

## Exercise: Empty Graph

**Concept:** Empty graph returns empty dict

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty graph returns empty dict.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_graph(self):
        """Empty graph returns empty dict."""
        result = _louvain_core({})
        assert result == {}
```

</details>

---

## Exercise: Single Node

**Concept:** Single node is its own community

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single node is its own community.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_node(self):
        """Single node is its own community."""
        result = _louvain_core({"a": {}})
        assert "a" in result
        assert result["a"] == 0
```

</details>

---

## Exercise: Empty Graph

**Concept:** Empty graph has zero modularity

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty graph has zero modularity.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_graph(self):
        """Empty graph has zero modularity."""
        result = _modularity_core({}, {})
        assert result == 0.0
```

</details>

---

---

*Completed 10 exercises? Check out the other topics for more challenges!*

---

## Search Exercises

*Hands-on coding exercises to master search concepts.*

**Difficulty Level:** Intermediate

---

## Introduction

Master the search and retrieval capabilities:

- Query expansion techniques
- Document ranking algorithms
- Passage retrieval
- Definition extraction

## Exercise: Empty Query

**Concept:** Empty query returns empty results

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty query returns empty results.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", tfidf=1.0, doc_ids=["doc1"])
        tokenizer = Tokenizer()

        # Tokenizer will return empty list for empty string
        result = find_documents_for_query("", layers, tokenizer)
        assert result == []
```

</details>

---

## Exercise: Single Term Single Doc

**Concept:** Single term matching single document

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Single term matching single document.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_single_doc(self):
        """Single term matching single document."""
        # Create layer with term in doc1
        col = MockMinicolumn(
            content="neural",
            tfidf=2.5,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] > 0
```

</details>

---

## Exercise: Single Term Multiple Docs

**Concept:** Single term in multiple documents ranked by TF-IDF

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Single term in multiple documents ranked by TF-IDF.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_multiple_docs(self):
        """Single term in multiple documents ranked by TF-IDF."""
        col = MockMinicolumn(
            content="algorithm",
            tfidf=3.0,
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 5.0, "doc2": 3.0, "doc3": 1.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "algorithm", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 3
        # Should be sorted by TF-IDF score
        assert result[0][0] == "doc1"  # Highest score
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"  # Lowest score
        assert result[0][1] > result[1][1] > result[2][1]
```

</details>

---

## Exercise: Query Expansion Disabled

**Concept:** use_expansion=False uses only query terms

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

use_expansion=False uses only query terms.

### Hints

<details>
<summary>Hint 1</summary>

Break down the problem into smaller steps.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_query_expansion_disabled(self):
        """use_expansion=False uses only query terms."""
        # Create connected terms
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0, pagerank=0.8)
            .with_term("network", tfidf=2.0, pagerank=0.6)
            .with_connection("neural", "network", weight=5.0)
            .with_document("doc1", ["neural"])
            .with_document("doc2", ["network"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {"doc1": 2.0}
        layer0.get_minicolumn("network").tfidf_per_doc = {"doc2": 2.0}

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer,
            use_expansion=False
        )

        # Should only find doc1 (contains "neural")
        assert len(result) == 1
        assert result[0][0] == "doc1"
```

</details>

---

## Exercise: Empty Corpus

**Concept:** Empty corpus returns empty results

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty corpus returns empty results.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_corpus(self):
        """Empty corpus returns empty results."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()

        result = find_documents_for_query("query", layers, tokenizer)

        assert result == []
```

</details>

---

## Exercise: Single Term Match

**Concept:** Fast search finds document with matching term

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Fast search finds document with matching term.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_match(self):
        """Fast search finds document with matching term."""
        col = MockMinicolumn(
            content="algorithm",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 3.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = fast_find_documents("algorithm", layers, tokenizer)

        assert len(result) == 1
        assert result[0][0] == "doc1"
```

</details>

---

## Exercise: Empty Query

**Concept:** Empty query returns empty results

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty query returns empty results.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = fast_find_documents("", layers, tokenizer)

        assert result == []
```

</details>

---

## Exercise: No Candidates Returns Empty

**Concept:** No matching candidates returns empty

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

No matching candidates returns empty.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_no_candidates_returns_empty(self):
        """No matching candidates returns empty."""
        layers = MockLayers.single_term("existing", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = fast_find_documents("nonexistent", layers, tokenizer)

        assert result == []
```

</details>

---

## Exercise: Empty Layer

**Concept:** Empty layer returns empty index

**Difficulty:** Intermediate

**Time:** ~10 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty layer returns empty index.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_layer(self):
        """Empty layer returns empty index."""
        layers = MockLayers.empty()
        result = build_document_index(layers)
        assert result == {}
```

</details>

---

## Exercise: Single Term Single Doc

**Concept:** Single term in single document

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single term in single document.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_single_doc(self):
        """Single term in single document."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        result = build_document_index(layers)

        assert "term" in result
        assert result["term"] == {"doc1": 2.5}
```

</details>

---

---

*Completed 10 exercises? Check out the other topics for more challenges!*

---

## Exercises

*Hands-on coding exercises derived from the test suite.*

---

## Overview

Learn by doing! These exercises are extracted from the Cortical Text Processor's test suite and transformed into learning challenges.

Each exercise includes:

- **Clear task description** - What you need to implement
- **Progressive hints** - Guidance without spoilers
- **Complete solution** - Reference implementation from tests
- **Verification** - How to check your answer

## Exercise Topics

### [Foundations](ex-foundations.md)

**Difficulty:** Beginner

**Exercises:** 10

Core algorithms and data structures. Start here if you're new!

### [Search](ex-search.md)

**Difficulty:** Intermediate

**Exercises:** 10

Search, ranking, and retrieval techniques. Build on foundations.

### [Advanced](ex-advanced.md)

**Difficulty:** Advanced

**Exercises:** 10

Advanced features and complex algorithms. For experienced users.

## Learning Path

**Recommended progression:**

1. Start with **Foundations** exercises
2. Move to **Search** once comfortable
3. Challenge yourself with **Advanced** topics

## Tips for Success

- **Read the test carefully** - The docstring explains what's being tested
- **Use hints progressively** - Try solving first, then reveal hints as needed
- **Run the solution** - Verify your understanding by executing the code
- **Experiment** - Modify parameters and see how behavior changes

---

*Total exercises: 30*

---

# 09 Journey

## Your Learning Journey

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

---

## Mastery Path (Advanced)

*Deep dives into sophisticated algorithms. For those ready to master the full system.*

---

**Concepts:** 3

**Estimated Time:** ~105 minutes

---

## 1. Concept Clustering

**First Introduced:** 2025-12-09

**Mentions in History:** 2 commits

**Reading Time:** ~35 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/concept-clustering.md`

**Key Takeaway:**
Understand the role of concept clustering in the information retrieval pipeline.
---

## 2. Semantic Relations

**First Introduced:** 2025-12-09

**Mentions in History:** 1 commits

**Reading Time:** ~35 min

**Prerequisites:**
- Query Expansion

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/semantic-relations.md`

**Key Takeaway:**
Dive into pattern-based extraction of typed relationships.
---

## 3. Louvain

**First Introduced:** 2025-12-11

**Mentions in History:** 7 commits

**Reading Time:** ~35 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/louvain.md`

**Key Takeaway:**
Explore community detection for automatic concept clustering.
---

---

## Start Here (Foundational)

*Core concepts that unlock everything else. Start here if you're new to information retrieval.*

---

**Concepts:** 3

**Estimated Time:** ~45 minutes

---

## 1. Tokenization

**First Introduced:** 2025-12-10

**Mentions in History:** 3 commits

**Reading Time:** ~15 min

**Prerequisites:** None (foundational concept)

**Related Concepts:**
- Bm25

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/tokenization.md`

**Key Takeaway:**
Learn how text is broken into processable units - the foundation of all text analysis.
---

## 2. Stop Words

**First Introduced:** 2025-12-13

**Mentions in History:** 1 commits

**Reading Time:** ~15 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/stop-words.md`

**Key Takeaway:**
See why common words are filtered to focus on meaningful content.
---

## 3. Tf-Idf

**First Introduced:** 2025-12-15

**Mentions in History:** 2 commits

**Reading Time:** ~15 min

**Prerequisites:** None (foundational concept)

**Related Concepts:**
- Bm25
- Query Expansion

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/tf-idf.md`

**Key Takeaway:**
Master the classic algorithm for term importance - weighing frequency against distinctiveness.
---

---

## Going Deeper (Intermediate)

*Build on the foundations. These concepts add power and flexibility to your understanding.*

---

**Concepts:** 6

**Estimated Time:** ~150 minutes

---

## 1. Pagerank

**First Introduced:** 2025-12-09

**Mentions in History:** 4 commits

**Reading Time:** ~25 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/pagerank.md`

**Key Takeaway:**
Learn how graph algorithms measure importance through connections.
---

## 2. Lateral Connections

**First Introduced:** 2025-12-09

**Mentions in History:** 2 commits

**Reading Time:** ~25 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/lateral-connections.md`

**Key Takeaway:**
Grasp the Hebbian-inspired network of term relationships.
---

## 3. Query Expansion

**First Introduced:** 2025-12-10

**Mentions in History:** 4 commits

**Reading Time:** ~25 min

**Prerequisites:**
- Tf-Idf

**Related Concepts:**
- Tf-Idf

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/query-expansion.md`

**Key Takeaway:**
See how searches become smarter by exploring related terms.
---

## 4. Incremental Indexing

**First Introduced:** 2025-12-10

**Mentions in History:** 1 commits

**Reading Time:** ~25 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/incremental-indexing.md`

**Key Takeaway:**
Understand the role of incremental indexing in the information retrieval pipeline.
---

## 5. Bm25

**First Introduced:** 2025-12-15

**Mentions in History:** 9 commits

**Reading Time:** ~25 min

**Prerequisites:**
- Tokenization
- Tf-Idf

**Related Concepts:**
- Tf-Idf
- Tokenization

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/bm25.md`

**Key Takeaway:**
Understand the modern scoring function that improves on TF-IDF with saturation.
---

## 6. Persistence

**First Introduced:** 2025-12-16

**Mentions in History:** 1 commits

**Reading Time:** ~25 min

**Prerequisites:** None (foundational concept)

**Where to Learn:**
- Foundations: `book/01-foundations/`
- Modules: `book/02-modules/`
- Evolution: `book/07-concepts/persistence.md`

**Key Takeaway:**
Understand the role of persistence in the information retrieval pipeline.
---

---

---

## About This Book

**The Cortical Chronicles** is a self-documenting book generated by the Cortical Text Processor.
It documents its own architecture, algorithms, and evolution through automated extraction
of code metadata, git history, and architectural decision records.

### How to Regenerate

```bash
# Generate individual chapters
python scripts/generate_book.py

# Generate consolidated markdown
python scripts/generate_book.py --markdown

# Force regeneration (ignore cache)
python scripts/generate_book.py --markdown --force
```

### Source Code

The source code and generation scripts are available at the project repository.
