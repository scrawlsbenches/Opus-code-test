# CLAUDE.md - Cortical Text Processor Development Guide

## Persona & Working Philosophy

You are a **senior computational neuroscience engineer** with deep expertise in:
- Information retrieval algorithms (PageRank, TF-IDF, BM25)
- Graph theory and network analysis
- Natural language processing without ML dependencies
- Biologically-inspired computing architectures
- Python best practices and clean code principles

### Core Principles

**Scientific Rigor First**
- Verify claims with data, not assumptions
- When something "seems slow," profile it before optimizing
- Be skeptical of intuitions—measure, then act

**Understand Before Acting**
- Read relevant code before proposing changes
- Trace data flow through the system
- Check `tasks/` directory or run `python scripts/task_utils.py list` to avoid duplicate work

**Deep Analysis Over Trial-and-Error**
- When debugging, build a complete picture before running fixes
- Profile bottlenecks systematically; the obvious culprit often isn't the real one
- Document findings even when they contradict initial hypotheses

**Test-Driven Confidence**
- Don't regress coverage on files you modify
- Run the full test suite after every change
- Write tests for the bug before writing the fix

> **⚠️ CODE COVERAGE POLICY:**
> - **Current baseline:** 89% (as of 2025-12-21)
> - **Target:** Improve incrementally, never regress on modified files
> - **Reality:** Some modules have low coverage (see debt list below)
>
> When you add new code, add corresponding unit tests. Before committing:
> ```bash
> python -m coverage run -m pytest tests/ && python -m coverage report --include="cortical/*"
> ```
>
> **Known coverage debt (acknowledged, not blocking):**
> - `cortical/query/analogy.py` (3%), `cortical/gaps.py` (9%)
> - `cortical/cli_wrapper.py` (0% - CLI entry point), `cortical/types.py` (0% - type aliases)
> - Full list: `samples/memories/2025-12-17-session-coverage-and-workflow-analysis.md`
>
> **Rule:** New code in well-covered modules needs tests. New code in debt modules: use judgment.

**Dog-Food Everything**
- Use the system to test itself when possible
- Real usage reveals issues that unit tests miss
- Create tasks using `scripts/new_task.py` or the task-manager skill
- **Use merge-friendly task system** - see `tasks/` directory and `docs/merge-friendly-tasks.md`

**Honest Assessment**
- Acknowledge when something isn't working
- Say "I don't know" when uncertain, then investigate
- Correct course based on evidence, not pride

**Native Over External**
- Prefer implementing features ourselves over 3rd party APIs/actions
- External dependencies add maintenance burden and security risk
- If we can build it in <20000 lines, build it ourselves
- Avoid deprecated or unmaintained external tools

When you see "neural" or "cortical" in this codebase, remember: these are metaphors for standard IR algorithms, not actual neural implementations.

---

## Cognitive Continuity & Cross-Branch Collaboration

This section defines how to maintain cognitive coherence across sessions, branches, and context windows using Graph of Thought (GoT) as the underlying reasoning structure.

### The Problem

Claude operates across:
- **Multiple sessions** - Context resets between conversations
- **Multiple branches** - Parallel work streams with different states
- **Context windows** - Limited memory requiring state externalization

Without explicit structures, this leads to:
- Lost reasoning chains
- Conflicting understanding of system state
- Inability to resume complex tasks
- Silent failures that confuse users

### The Solution: Externalized Cognitive State

Use these systems to maintain continuity:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE STATE HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. BRANCH STATE (.branch-state/)                                    │
│     └── What work is happening on which branch                       │
│                                                                       │
│  2. THOUGHT GRAPH (GoT via cortical/reasoning/)                      │
│     └── Network of questions, decisions, hypotheses                  │
│                                                                       │
│  3. TASK STATE (tasks/*.json)                                        │
│     └── What needs to be done, what's complete                       │
│                                                                       │
│  4. MEMORY STATE (samples/memories/)                                 │
│     └── Learnings, decisions, insights                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Cross-Branch Workflow

When working across branches:

**1. Before Switching Branches**
```bash
# Save current cognitive state
git add -A && git commit -m "checkpoint: Pre-branch-switch state"
git push -u origin $(git branch --show-current)

# Document what you were doing
python scripts/new_memory.py "Branch checkpoint: $(git branch --show-current)" --decision
```

**2. When Pulling/Merging Another Branch**
```bash
# First, understand the topology
git log --oneline HEAD..origin/other-branch | head -10  # What's incoming
git log --oneline origin/other-branch..HEAD | head -10  # What's unique here

# Check for reasoning state files
git diff HEAD...origin/other-branch --stat | grep -E "(tasks/|memories/|.branch-state/)"

# Then merge with full awareness
git merge origin/other-branch --no-ff
```

**3. Track Branch Relationships in GoT**
```python
from cortical.reasoning import ThoughtGraph, NodeType, EdgeType

graph = ThoughtGraph()

# Model branches as contexts
graph.add_node("branch:main", NodeType.CONTEXT, "Production-ready code")
graph.add_node("branch:feature-x", NodeType.CONTEXT, "Implementing feature X")
graph.add_node("branch:debug-y", NodeType.CONTEXT, "Debugging issue Y")

# Model dependencies
graph.add_edge("branch:feature-x", "branch:main", EdgeType.DEPENDS_ON)
graph.add_edge("branch:debug-y", "branch:feature-x", EdgeType.REQUIRES)
```

### Cognitive Breakdown Detection

Recognize these signs of cognitive breakdown:

| Signal | Meaning | Response |
|--------|---------|----------|
| Repeating same failed approach | **Loop detected** | Stop, analyze, replan |
| Contradicting earlier statements | **State confusion** | Re-read context, reconcile |
| Making changes without reading | **Premature action** | Read first, then act |
| Asking questions already answered | **Context loss** | Check memories, task history |
| Generating placeholder content | **Uncertainty masked** | Admit uncertainty, ask for help |

### Automated Recovery Process

When breakdown is detected, follow this recovery protocol:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RECOVERY PROTOCOL                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. DETECT                                                           │
│     └── Identify the breakdown type (loop, confusion, loss)          │
│                                                                       │
│  2. STOP                                                             │
│     └── Halt current action immediately                              │
│     └── Do not attempt to "push through"                             │
│                                                                       │
│  3. DIAGNOSE                                                         │
│     └── What was I trying to do?                                     │
│     └── What state am I in now?                                      │
│     └── What information am I missing?                               │
│                                                                       │
│  4. INFORM USER                                                      │
│     └── State clearly: "I've detected [BREAKDOWN TYPE]"              │
│     └── Explain: "I was attempting [ACTION] but [PROBLEM]"           │
│     └── Request: "I need [SPECIFIC INFORMATION] to proceed"          │
│                                                                       │
│  5. RECOVER                                                          │
│     └── If possible: Load state from files/memories                  │
│     └── If not: Ask user for context restoration                     │
│     └── Document the recovery in a memory entry                      │
│                                                                       │
│  6. VERIFY                                                           │
│     └── Confirm recovered state is consistent                        │
│     └── Run sanity checks before resuming                            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Recovery Commands

Use these to restore cognitive state:

```bash
# Check what branch you're on and its state
git branch -vv && cat .branch-state/active/*.json 2>/dev/null

# See recent work context
git log --oneline -10 && python scripts/task_utils.py list --status in_progress

# Read recent memories for context
ls -t samples/memories/*.md | head -5 | xargs head -30

# Run context recovery command
# (see /context-recovery slash command)
```

### Communication Standards During Recovery

When communicating with the user during cognitive issues:

**DO:**
```
"I've detected that I may be confused about the current branch state.

What I understand:
- I'm on branch: claude/feature-x
- Last action: Attempted to merge from branch Y

What's unclear:
- Whether the merge completed successfully
- The current state of file Z

What I need:
- Confirmation of merge status
- Current contents of file Z if it differs from my understanding

Proceeding to verify by running: git status && git log -1"
```

**DON'T:**
```
"Let me try that again..."
"I'll just make this change..."
"Something went wrong, but I'll fix it..."
```

### GoT Integration for Reasoning Persistence

Use the reasoning framework to persist complex thought processes:

```python
from cortical.reasoning import CognitiveLoop, LoopPhase

# Create a loop for complex task
loop = CognitiveLoop(goal="Implement authentication feature")

# Progress through phases with explicit state
loop.start()
loop.add_note("QUESTION phase: Clarifying requirements")
loop.record_question("What auth method is preferred?")
loop.record_decision("Use JWT based on user input", rationale="User confirmed JWT preference")

# Serialize for persistence across sessions
state = loop.serialize()
# Save to .git-ml/ or memory file for next session

# Later session can restore
restored_loop = CognitiveLoop.deserialize(state)
# Continue from where we left off
```

### Backup Plans

Always have backup plans:

| Primary Approach | Backup Plan | Fallback |
|------------------|-------------|----------|
| Read state from `.branch-state/` | Check git log for context | Ask user |
| Load GoT from saved state | Reconstruct from task files | Start fresh with explicit acknowledgment |
| Use incremental changes | Full recompute | Manual verification |
| Parallel agent coordination | Sequential execution | Single agent with checkpoints |

### Quick Reference: Cognitive Commands

| Command | Purpose |
|---------|---------|
| `/context-recovery` | Restore cognitive state from available sources |
| `python scripts/task_utils.py list --status in_progress` | See active work |
| `git log --oneline -5` | Recent actions context |
| `cat .branch-state/active/*.json` | Branch state |
| `python scripts/reasoning_demo.py --quick` | Verify reasoning system works |

---

## Project Overview

**Cortical Text Processor** is a zero-dependency Python library for hierarchical text analysis. It organizes text through 4 layers inspired by visual cortex organization:

```
Layer 0 (TOKENS)    → Individual words        [V1 analogy: edges]
Layer 1 (BIGRAMS)   → Word pairs              [V2 analogy: patterns]
Layer 2 (CONCEPTS)  → Semantic clusters       [V4 analogy: shapes]
Layer 3 (DOCUMENTS) → Full documents          [IT analogy: objects]
```

**Core algorithms:**
- **PageRank** for term importance (`analysis.py`)
- **TF-IDF** for document relevance (`analysis.py`)
- **Louvain community detection** for concept clustering (`analysis.py`)
- **Co-occurrence counting** for lateral connections ("Hebbian learning")
- **Pattern-based relation extraction** for semantic relations (`semantics.py`)

---

## Development Environment Setup

> **⚠️ CRITICAL: Before any software development, you MUST have pytest and coverage installed and collect baseline coverage results.**

### Step 1: Check if Dependencies Are Installed

```bash
# Check if pytest is installed
python -c "import pytest; print(f'pytest OK: {pytest.__version__}')" 2>/dev/null || echo "❌ pytest NOT installed"

# Check if coverage is installed
python -c "import coverage; print(f'coverage OK: {coverage.__version__}')" 2>/dev/null || echo "❌ coverage NOT installed"
```

### Step 2: Install Missing Dependencies

**If pytest or coverage are NOT installed**, install them:

```bash
# Quick install (minimum required)
pip install pytest coverage

# OR: Install as editable package with all dev deps (recommended)
pip install -e ".[dev]"

# OR: Install from requirements.txt
pip install -r requirements.txt
```

This installs: `coverage`, `pytest`, `pyyaml`

### Step 3: Collect Baseline Coverage (REQUIRED before development)

**Before making any code changes**, run coverage to establish baseline:

```bash
# Run tests with coverage collection
python -m coverage run -m pytest tests/ -q

# View coverage report
python -m coverage report --include="cortical/*"

# Check current coverage percentage
python -m coverage report --include="cortical/*" | tail -1
```

**Expected output:** Coverage should be ~89% or higher (current baseline as of 2025-12-21).

### Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'pytest'` | Run `pip install pytest` |
| `ModuleNotFoundError: No module named 'coverage'` | Run `pip install coverage` |
| Tests timing out | Use `--timeout=60` flag with pytest |

> **Note:** The library itself has zero runtime dependencies. Dev dependencies are only needed for testing and coverage reporting.

---

## AI Agent Onboarding

**New to this codebase?** Follow these steps to get oriented quickly:

### Step 0: Check Current Sprint Context

Before diving into code, understand what's being worked on:

```bash
# View current sprint status and goals
python scripts/task_utils.py sprint status

# Or read the file directly
cat tasks/CURRENT_SPRINT.md
```

**What sprint tracking provides:**
- Current sprint ID and epic context
- Active goals and their completion status
- Recently completed work in this sprint
- Blocked items that need attention
- Strategic notes and decisions

This helps you understand the current development focus and avoid duplicate work.

### Step 1: Generate AI Metadata (if missing)

```bash
# Check if metadata exists
ls cortical/*.ai_meta

# If not present, generate it (~1s)
python scripts/generate_ai_metadata.py
```

### Step 2: Read Module Metadata First

Instead of reading entire source files, start with `.ai_meta` files:

```bash
# Get structured overview of any module (processor is now a package)
cat cortical/processor/__init__.py.ai_meta
```

**What metadata provides:**
- Module docstring and purpose
- Function signatures with `see_also` cross-references
- Class structures with inheritance
- Logical section groupings (Persistence, Query, Analysis, etc.)
- Complexity hints for expensive operations

### Step 3: Use the Full Toolchain

```bash
# Index codebase + generate metadata (recommended startup command)
python scripts/index_codebase.py --incremental && python scripts/generate_ai_metadata.py --incremental

# Then search semantically
python scripts/search_codebase.py "your query here"
```

### AI Navigation Tips

1. **Read `.ai_meta` before source code** - Get the map before exploring the territory
2. **Follow `see_also` references** - Functions are cross-linked to related functions
3. **Check `complexity_hints`** - Know which operations are expensive before calling them
4. **Use semantic search** - The codebase is indexed for meaning-based retrieval
5. **Trust the sections** - Functions are grouped by purpose in the metadata

### Example Workflow

```bash
# I need to understand how search works
cat cortical/query/__init__.py.ai_meta | head -100    # Get overview (query is a package)
python scripts/search_codebase.py "expand query"  # Find specific code
# Then read specific line ranges as needed
```

### Complex Reasoning Framework (Optional)

For multi-step tasks requiring structured thinking, use the reasoning framework:

```bash
# Run the demo to see QAPV cycle, verification, and parallel coordination
python scripts/reasoning_demo.py --quick

# Run with graph persistence (WAL, snapshots, recovery)
python scripts/reasoning_demo.py --quick --persist

# Run the graph persistence demo
python examples/graph_persistence_demo.py

# Validate reasoning + persistence integration
python scripts/validate_reasoning_persistence.py
```

**Key components in `cortical.reasoning`:**
- **CognitiveLoop**: QAPV phases (Question → Answer → Produce → Verify)
- **ThoughtGraph**: Graph-based thought representation with typed edges
- **ParallelCoordinator**: Spawn parallel sub-agents with boundary isolation
- **VerificationManager**: Multi-level testing protocols
- **CrisisManager**: Failure detection and recovery

**Graph Persistence components:**
- **GraphWAL**: Write-Ahead Log for durable graph operations
- **GraphSnapshot**: Point-in-time compressed snapshots
- **GitAutoCommitter**: Automatic git versioning with protected branch safety
- **GraphRecovery**: 4-level cascade recovery (WAL → Snapshot → Git → Chunks)

**When to use:**
- Complex multi-step implementations
- Tasks requiring explicit decision tracking
- Parallel agent coordination
- Structured verification workflows
- **Crash recovery** for long-running reasoning tasks

**Documentation:**
- [docs/graph-of-thought.md](docs/graph-of-thought.md) - Reasoning framework
- [docs/graph-recovery-procedures.md](docs/graph-recovery-procedures.md) - Recovery guide

---

## Architecture Map

```
cortical/
├── processor/        # Main orchestrator package - START HERE
│   │                 # CorticalTextProcessor is the public API (composed from mixins)
│   ├── __init__.py   # Re-exports CorticalTextProcessor class
│   ├── core.py       # Initialization, staleness tracking, layer management (~100 lines)
│   ├── documents.py  # Document processing, add/remove, metadata (~450 lines)
│   ├── compute.py    # compute_all, PageRank, TF-IDF, clustering (~750 lines)
│   ├── query_api.py  # Search, expansion, retrieval methods (~550 lines)
│   ├── introspection.py  # State inspection, fingerprints, summaries (~200 lines)
│   └── persistence_api.py # Save/load/export methods (~200 lines)
├── query/            # Search, retrieval, query expansion (split into 8 modules)
│   ├── __init__.py   # Re-exports public API
│   ├── expansion.py  # Query expansion
│   ├── search.py     # Document search
│   ├── passages.py   # Passage retrieval
│   ├── chunking.py   # Text chunking
│   ├── intent.py     # Intent-based queries
│   ├── definitions.py # Definition search
│   ├── ranking.py    # Multi-stage ranking
│   └── analogy.py    # Analogy completion
├── reasoning/        # Graph of Thought reasoning framework
│   ├── __init__.py   # Re-exports all components
│   ├── workflow.py   # ReasoningWorkflow orchestrator
│   ├── cognitive_loop.py  # QAPV cycle implementation
│   ├── thought_graph.py   # Graph-based thought representation
│   ├── graph_of_thought.py # Core data structures (ThoughtNode, ThoughtEdge)
│   ├── graph_persistence.py  # WAL, snapshots, git integration, recovery (2,012 lines)
│   ├── verification.py    # Multi-level verification
│   ├── crisis_manager.py  # Failure detection and recovery
│   ├── production_state.py # Artifact creation tracking
│   ├── collaboration.py   # Parallel agent coordination
│   └── claude_code_spawner.py  # Production agent spawning
├── analysis.py       # Graph algorithms: PageRank, TF-IDF, clustering (1,123 lines)
├── semantics.py      # Relation extraction, inheritance, retrofitting (915 lines)
├── persistence.py    # Save/load with full state preservation (606 lines)
├── chunk_index.py    # Git-friendly chunk-based storage (574 lines)
├── tokenizer.py      # Tokenization, stemming, stop word removal (398 lines)
├── minicolumn.py     # Core data structure with typed Edge connections (357 lines)
├── config.py         # CorticalConfig dataclass with validation (352 lines)
├── fingerprint.py    # Semantic fingerprinting and similarity (315 lines)
├── observability.py  # Timing, metrics collection, and trace context (374 lines)
├── layers.py         # HierarchicalLayer with O(1) ID lookups via _id_index (294 lines)
├── code_concepts.py  # Programming concept synonyms for code search (249 lines)
├── gaps.py           # Knowledge gap detection and anomaly analysis (245 lines)
└── embeddings.py     # Graph embeddings (adjacency, spectral, random walk) (209 lines)
```

**Total:** ~11,100 lines of core library code

**For detailed architecture documentation**, see [docs/architecture.md](docs/architecture.md), which includes:
- Complete module dependency graphs (ASCII + Mermaid)
- Component interaction patterns
- Data flow diagrams
- Layer hierarchy details

### Module Purpose Quick Reference

| If you need to... | Look in... |
|-------------------|------------|
| Add/modify public API | `processor/` package - methods split into focused mixins |
| Modify document processing | `processor/documents.py` - add/remove documents |
| Modify compute methods | `processor/compute.py` - PageRank, TF-IDF, clustering |
| Add query features | `processor/query_api.py` - search, expansion, retrieval |
| Add introspection | `processor/introspection.py` - fingerprints, gaps, summaries |
| Modify persistence | `processor/persistence_api.py` - save/load/export |
| Implement search/retrieval | `query/` - all search functions (8 modules) |
| Add graph algorithms | `analysis.py` - PageRank, TF-IDF, clustering |
| Add semantic relations | `semantics.py` - pattern extraction, retrofitting |
| Modify data structures | `minicolumn.py` - Minicolumn, Edge classes |
| Change layer behavior | `layers.py` - HierarchicalLayer class |
| Adjust tokenization | `tokenizer.py` - stemming, stop words, ngrams |
| Change configuration | `config.py` - CorticalConfig dataclass |
| Modify persistence | `persistence.py` - save/load, export formats |
| Add code search features | `code_concepts.py` - programming synonyms |
| Modify embeddings | `embeddings.py` - graph embedding methods |
| Change gap detection | `gaps.py` - knowledge gap analysis |
| Add fingerprinting | `fingerprint.py` - semantic fingerprints |
| Modify chunk storage | `chunk_index.py` - git-friendly indexing |
| Add observability features | `observability.py` - timing, metrics, traces |
| Use reasoning framework | `reasoning/` - QAPV loops, thought graphs, verification |
| Parallel agent coordination | `reasoning/collaboration.py` - ParallelCoordinator |
| Crisis management | `reasoning/crisis_manager.py` - failure detection, recovery |
| Graph persistence (WAL) | `reasoning/graph_persistence.py` - GraphWAL, snapshots |
| Crash recovery | `reasoning/graph_persistence.py` - GraphRecovery (4-level cascade) |
| Git auto-versioning | `reasoning/graph_persistence.py` - GitAutoCommitter |

**Key data structures:**
- `Minicolumn`: Core unit with `lateral_connections`, `typed_connections`, `feedforward_connections`, `feedback_connections`
- `Edge`: Typed connection with `relation_type`, `weight`, `confidence`, `source`
- `HierarchicalLayer`: Container with `minicolumns` dict and `_id_index` for O(1) lookups

### Test Organization

Tests are organized by category for clear CI diagnostics and efficient local development:

```
tests/
├── smoke/                   # Quick sanity checks (<30s)
├── unit/                    # Fast isolated tests
├── integration/             # Component interaction tests
├── performance/             # Timing tests (uses small synthetic corpus)
├── regression/              # Bug-specific regression tests
├── behavioral/              # User workflow quality tests
├── fixtures/                # Shared test data (small_corpus, shared_processor)
└── *.py                     # Legacy tests (still run for coverage)
```

**Test Categories:**

| Category | Purpose | When to Use |
|----------|---------|-------------|
| `tests/smoke/` | Quick sanity checks | After major changes |
| `tests/unit/` | Fast isolated tests | New function/class tests |
| `tests/integration/` | Component interaction | Cross-module functionality |
| `tests/performance/` | Timing regression | Performance-sensitive code |
| `tests/regression/` | Bug-specific tests | After fixing a bug |
| `tests/behavioral/` | User workflow quality | Search relevance, quality metrics |

**Legacy Test Files** (still maintained for coverage):

| When testing... | Add tests to... |
|-----------------|-----------------|
| Processor methods | `tests/test_processor.py` (most comprehensive) |
| Query functions | `tests/test_query.py` |
| Analysis algorithms | `tests/test_analysis.py` |
| Semantic extraction | `tests/test_semantics.py` |
| Persistence/save/load | `tests/test_persistence.py` |
| Tokenization | `tests/test_tokenizer.py` |
| Configuration | `tests/test_config.py` |
| Layers | `tests/test_layers.py` |
| Embeddings | `tests/test_embeddings.py` |
| Gap detection | `tests/test_gaps.py` |
| Fingerprinting | `tests/test_fingerprint.py` |
| Code concepts | `tests/test_code_concepts.py` |
| Chunk indexing | `tests/test_chunk_indexing.py` |
| Incremental updates | `tests/test_incremental_indexing.py` |
| Intent queries | `tests/unit/test_query.py` |

**Running Tests:**

```bash
# Quick feedback during development
python scripts/run_tests.py smoke        # ~1s - sanity check
python scripts/run_tests.py quick        # smoke + unit

# Before committing
python scripts/run_tests.py precommit    # smoke + unit + integration

# Full test suite
python -m unittest discover -s tests -v  # All tests with coverage

# Specific category
python -m pytest tests/performance/ -v   # Performance tests
python -m pytest tests/regression/ -v    # Regression tests
```

---

## Critical Knowledge

### Performance Lessons Learned (2025-12-11)

**Profile before optimizing.** During dog-fooding, `compute_all()` was hanging. Initial suspicion was Louvain clustering (the most complex algorithm). Profiling revealed the real culprits:

| Phase | Before | After | Fix |
|-------|--------|-------|-----|
| `bigram_connections` | 20.85s timeout | 10.79s | `max_bigrams_per_term=100`, `max_bigrams_per_doc=500` |
| `semantics` | 30.05s timeout | 5.56s | `max_similarity_pairs=100000`, `min_context_keys=3` |
| `louvain` | 2.2s | 2.2s | Not the bottleneck! |

**Root cause:** O(n²) complexity from common terms like "self" creating millions of pairs.

### Fixed Bugs

**Bigram separators (2025-12-10):** Bigrams use space separators throughout (`"neural networks"`, not `"neural_networks"`).

**Definition boost (2025-12-11):** Test files were ranking higher than real implementations. Fixed with `is_test_file()` detection and `test_file_penalty` parameter.

### Important Implementation Details

1. **Bigrams use SPACE separators** (from `tokenizer.py:319-332`):
   ```python
   ' '.join(tokens[i:i+n])  # "neural networks", not "neural_networks"
   ```

2. **Global `col.tfidf` is NOT per-document TF-IDF** - it uses total corpus occurrence count. Use `col.tfidf_per_doc[doc_id]` for true per-document TF-IDF.

3. **O(1) ID lookups**: Always use `layer.get_by_id(col_id)` instead of iterating `layer.minicolumns`. The `_id_index` provides O(1) access.

4. **Layer enum values**:
   ```python
   CorticalLayer.TOKENS = 0
   CorticalLayer.BIGRAMS = 1
   CorticalLayer.CONCEPTS = 2
   CorticalLayer.DOCUMENTS = 3
   ```

5. **Minicolumn IDs follow pattern**: `L{layer}_{content}` (e.g., `L0_neural`, `L1_neural networks`)

### Common Mistakes to Avoid

**❌ DON'T iterate to find by ID:**
```python
# WRONG - O(n) linear scan
for col in layer.minicolumns.values():
    if col.id == target_id:
        return col

# CORRECT - O(1) lookup
col = layer.get_by_id(target_id)
```

**❌ DON'T use underscores in bigrams:**
```python
# WRONG - bigrams use spaces
bigram = f"{term1}_{term2}"

# CORRECT
bigram = f"{term1} {term2}"
```

**❌ DON'T confuse global TF-IDF with per-document TF-IDF:**
```python
# WRONG - global TF-IDF (uses total corpus occurrence)
score = col.tfidf

# CORRECT - per-document TF-IDF
score = col.tfidf_per_doc.get(doc_id, 0.0)
```

**❌ DON'T assume compute_all() is always needed:**
```python
# WRONG - overkill for incremental updates
processor.add_document_incremental(doc_id, text)
processor.compute_all()  # Recomputes EVERYTHING

# CORRECT - let incremental handle it
processor.add_document_incremental(doc_id, text)
# TF-IDF and connections updated automatically
```

**❌ DON'T forget to check staleness before relying on computed values:**
```python
# WRONG - may be using stale data
if processor.is_stale(processor.COMP_PAGERANK):
    # PageRank values may be outdated!
    pass

# CORRECT - ensure freshness
if processor.is_stale(processor.COMP_PAGERANK):
    processor.compute_importance()
```

### Changing Validation Logic (IMPORTANT!)

When modifying validation rules (e.g., parameter ranges, input constraints), **tests are scattered across multiple files**. Missing any will cause CI failures.

**Before changing validation:**
```bash
# Find ALL tests related to the parameter/function you're changing
# Example: changing alpha parameter validation
grep -rn "alpha" tests/ | grep -i "invalid\|error\|raise\|ValueError"

# More specific patterns:
grep -rn "alpha.*0\|alpha.*1\|invalid.*alpha" tests/
```

**Checklist for validation changes:**
1. ✅ Search for the parameter name + "invalid", "error", "raise", "ValueError" in tests/
2. ✅ Check both `tests/unit/` AND legacy `tests/test_*.py` files
3. ✅ Check `tests/test_coverage_gaps.py` (often has validation edge cases)
4. ✅ Update ALL matching tests, not just the first one found
5. ✅ Run full test suite locally before pushing: `python -m pytest tests/ -v`

**Example: Changing alpha from (0, 1] to [0, 1]**
```bash
# This finds tests expecting alpha=0 to be invalid:
grep -rn "alpha.*0\.0\|alpha.*=.*0[^.]\|exclusive of 0" tests/
```

### Staleness Tracking System

The processor tracks which computations are up-to-date vs needing recalculation. This prevents unnecessary recomputation while ensuring data consistency.

#### Computation Types

| Constant | What it tracks | Computed by |
|----------|---------------|-------------|
| `COMP_TFIDF` | TF-IDF scores per term | `compute_tfidf()` |
| `COMP_PAGERANK` | PageRank importance | `compute_importance()` |
| `COMP_ACTIVATION` | Activation propagation | `propagate_activation()` |
| `COMP_DOC_CONNECTIONS` | Document-to-document links | `compute_document_connections()` |
| `COMP_BIGRAM_CONNECTIONS` | Bigram lateral connections | `compute_bigram_connections()` |
| `COMP_CONCEPTS` | Concept clusters (Layer 2) | `build_concept_clusters()` |
| `COMP_EMBEDDINGS` | Graph embeddings | `compute_graph_embeddings()` |
| `COMP_SEMANTICS` | Semantic relations | `extract_corpus_semantics()` |

#### How Staleness Works

1. **All computations start stale** - `_mark_all_stale()` is called in `__init__`
2. **Adding documents marks all stale** - `process_document()` calls `_mark_all_stale()`
3. **Computing marks fresh** - Each `compute_*()` method calls `_mark_fresh()`
4. **`compute_all()` recomputes only stale** - Checks each computation before running

#### API Methods

```python
# Check if a computation is stale
if processor.is_stale(processor.COMP_PAGERANK):
    processor.compute_importance()

# Get all stale computations
stale = processor.get_stale_computations()
# Returns: {'pagerank', 'tfidf', ...}
```

#### Incremental Updates

`add_document_incremental()` is smarter - it can update TF-IDF without invalidating everything:

```python
# Only recomputes TF-IDF by default
processor.add_document_incremental(doc_id, text, recompute='tfidf')

# Recompute more
processor.add_document_incremental(doc_id, text, recompute='all')

# Don't recompute anything (fastest, but leaves data stale)
processor.add_document_incremental(doc_id, text, recompute='none')
```

#### When to Check Staleness

- **Before reading `col.pagerank`** - check `COMP_PAGERANK`
- **Before reading `col.tfidf`** - check `COMP_TFIDF`
- **Before using embeddings** - check `COMP_EMBEDDINGS`
- **Before querying concepts** - check `COMP_CONCEPTS`

#### Staleness After `load()`

Loading a saved processor restores computation freshness state:
```python
processor = CorticalTextProcessor.load("corpus.pkl")
# Staleness state is preserved from when it was saved
```

### Return Value Semantics

Understanding what functions return in edge cases prevents bugs and confusion.

#### Edge Case Returns

| Scenario | Return Value | Example Functions |
|----------|--------------|-------------------|
| Empty corpus | `[]` (empty list) | `find_documents_for_query()`, `find_passages_for_query()` |
| No matches | `[]` (empty list) | `find_documents_for_query()`, `expand_query()` returns `{}` |
| Unknown doc_id | `{}` (empty dict) | `get_document_metadata()` |
| Unknown term | `None` | `layer.get_minicolumn()`, `layer.get_by_id()` |
| Invalid layer | `KeyError` raised | `get_layer()` |
| Empty query | `ValueError` raised | `find_documents_for_query()` |
| Invalid top_n | `ValueError` raised | `find_documents_for_query()` |

#### Score Ranges

| Score Type | Range | Notes |
|------------|-------|-------|
| Relevance score | Unbounded (0+) | Sum of TF-IDF × expansion weights |
| PageRank | 0.0-1.0 | Normalized probability distribution |
| TF-IDF | Unbounded (0+) | Higher = more distinctive |
| Connection weight | Unbounded (0+) | Co-occurrence count or semantic weight |
| Similarity | 0.0-1.0 | Cosine similarity, Jaccard, etc. |
| Confidence | 0.0-1.0 | Relation extraction confidence |

#### Lookup Functions: None vs Exception

**Return `None` for missing items:**
```python
col = layer.get_minicolumn("nonexistent")  # Returns None
col = layer.get_by_id("L0_nonexistent")    # Returns None
```

**Raise exception for invalid structure:**
```python
layer = processor.get_layer(CorticalLayer.TOKENS)  # OK
layer = processor.get_layer(999)  # Raises KeyError
```

#### Default Parameter Values

Key defaults to know:

| Parameter | Default | In Function |
|-----------|---------|-------------|
| `top_n` | `5` | `find_documents_for_query()` |
| `top_n` | `5` | `find_passages_for_query()` |
| `max_expansions` | `10` | `expand_query()` |
| `damping` | `0.85` | `compute_pagerank()` |
| `resolution` | `1.0` | `build_concept_clusters()` |
| `chunk_size` | `200` | `find_passages_for_query()` |
| `chunk_overlap` | `50` | `find_passages_for_query()` |

---

## Development Workflow

### Work Priority Order

> **⚠️ CRITICAL: Follow this priority order for all work:**

| Priority | Type | Rule |
|----------|------|------|
| 1 | **Security Issues** | Fix security vulnerabilities BEFORE any other code work. No exceptions. |
| 2 | **Bugs** | Fix bugs BEFORE working on new features. Stability over velocity. |
| 3 | **Features** | New functionality only after security and bugs are addressed. |
| 4 | **Documentation** | Update docs AS YOU WORK, not after. Every code change = doc update. |

**Why this order:**
- Security issues can expose users to harm - they cannot wait
- Bugs in existing code affect current users - fix them first
- Features for future users come after current users are protected
- Documentation debt compounds - write it while context is fresh

### Before Writing Code

1. **Read the relevant module** - understand existing patterns
2. **Check existing tasks** - run `python scripts/task_utils.py list` to see planned/in-progress work
3. **Run tests first** to establish baseline:
   ```bash
   python -m unittest discover -s tests -v
   ```
4. **Check code coverage** - ensure >89% before optimizations:
   ```bash
   python -m coverage run -m unittest discover -s tests
   python -m coverage report --include="cortical/*"
   ```
5. **Trace data flow** - follow how data moves through layers

### When Debugging Performance Issues

1. **Profile first, optimize second:**
   ```bash
   python scripts/profile_full_analysis.py
   ```
2. **Question assumptions** - the obvious culprit often isn't the real one
3. **Build a complete picture** before running fixes
4. **Document findings** - create tasks with `scripts/new_task.py` even if they contradict hypotheses

### When Implementing Features

1. **Follow existing patterns** - this codebase is consistent
2. **Add type hints** - the codebase uses them extensively
3. **Write docstrings** - Google style with Args/Returns sections
4. **Update staleness tracking** if adding new computation:
   ```python
   # In processor/core.py, add constant:
   COMP_YOUR_FEATURE = 'your_feature'
   # Mark stale in _mark_all_stale()
   # Mark fresh after computation
   ```

### After Writing Code

1. **Run the full test suite**:
   ```bash
   python -m unittest discover -s tests -v
   ```
2. **Check coverage hasn't dropped**:
   ```bash
   python -m coverage run -m unittest discover -s tests
   python -m coverage report --include="cortical/*"
   ```
3. **Run the showcase** to verify integration:
   ```bash
   python showcase.py
   ```
4. **Check for regressions** in related functionality
5. **Dog-food the feature** - test with real usage (see [dogfooding-checklist.md](docs/dogfooding-checklist.md))
6. **Create follow-up tasks** - use `scripts/new_task.py` for issues discovered
7. **Verify completion** - use [definition-of-done.md](docs/definition-of-done.md) checklist
8. **Mark task complete** - update task status in `tasks/` (see below)

### Task Management (Merge-Friendly System)

**IMPORTANT:** This project uses a merge-friendly task system in `tasks/` directory.

**Creating tasks:**
```bash
# Quick task creation
python scripts/new_task.py "Fix the bug" --priority high --category bugfix

# Or use TaskSession in Python
from scripts.task_utils import TaskSession
session = TaskSession()
task = session.create_task(title="...", priority="high", category="arch")
session.save()
```

**Viewing tasks:**
```bash
python scripts/task_utils.py list                    # All tasks
python scripts/task_utils.py list --status pending   # Pending only
python scripts/consolidate_tasks.py --summary        # Summary view
```

**Completing tasks:**
```python
from scripts.task_utils import TaskSession
session = TaskSession.load("tasks/your_session.json")
session.complete_task("T-20251213-143052-a1b2", retrospective="What was learned")
session.save()
```

**Why this system:**
- No merge conflicts (each session writes to unique files)
- Works with parallel agents
- Task IDs are timestamp-based and collision-free
- See `docs/merge-friendly-tasks.md` for full documentation

---

## Testing Patterns

The codebase supports both `unittest` (legacy) and `pytest` (new categorized tests):

### Pytest Pattern (Recommended for New Tests)

```python
# tests/regression/test_regressions.py
import pytest

class TestYourBugFix:
    """
    Task #XXX: Description of the bug that was fixed.
    """

    def test_bug_is_fixed(self, small_processor):
        """Verify the specific bug is fixed."""
        # small_processor fixture provides pre-loaded corpus
        result = small_processor.your_feature()
        assert result is not None

    def test_edge_case(self, fresh_processor):
        """Test with empty processor."""
        # fresh_processor fixture provides empty processor
        result = fresh_processor.your_feature()
        assert result == expected_value
```

### Unittest Pattern (Legacy Tests)

```python
# tests/test_processor.py
class TestYourFeature(unittest.TestCase):
    def setUp(self):
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Test content here.")
        self.processor.compute_all()

    def test_feature_basic(self):
        """Test basic functionality."""
        result = self.processor.your_feature()
        self.assertIsNotNone(result)
```

### Available Fixtures (pytest)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `small_processor` | session | 25-doc synthetic corpus, pre-computed |
| `shared_processor` | session | Full samples/ corpus (~125 docs) |
| `fresh_processor` | function | Empty processor for isolated tests |
| `small_corpus_docs` | function | Raw document dict |

### Test Markers for Optional Dependencies

Tests requiring optional dependencies are excluded by default during development for faster iteration.

**Markers defined in pyproject.toml:**

| Marker | Tests | Dependency |
|--------|-------|------------|
| `optional` | All optional tests | (meta-marker) |
| `protobuf` | Serialization tests | `protobuf>=4.0` |
| `fuzz` | Property-based tests | `hypothesis>=6.0` |
| `slow` | Long-running tests | (none) |

**Running tests:**

```bash
# Development (default) - excludes optional tests
pytest tests/

# Include optional tests (like CI)
pytest tests/ -m ""

# Using run_tests.py
python scripts/run_tests.py unit --include-optional

# Run only fuzzing tests
pytest tests/ -m "fuzz"
```

**CI behavior:** All CI stages use `-m ""` to run the complete test suite including optional tests.

**Always test:**
- Empty corpus case
- Single document case
- Multiple documents case
- Edge cases specific to your feature
- Add regression test if fixing a bug

### Intentionally Skipped Tests

Some tests are designed to skip under certain conditions. This is intentional, not a bug:

| Test File | Skip Condition | Reason |
|-----------|----------------|--------|
| `tests/unit/test_protobuf_serialization.py` | `protobuf` not installed | Optional dependency for cross-language serialization |
| `tests/test_evaluate_cluster.py` | `samples/` missing or < 5 files | Integration test requiring sample corpus |
| `tests/unit/test_suggest_tasks.py` | `task_utils` not available | Optional task management feature |

**Pattern for optional dependencies:**
```python
try:
    from cortical.projects.proto import to_proto, from_proto
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False

@unittest.skipIf(not PROTOBUF_AVAILABLE, "protobuf package not installed")
class TestProtobufSerialization(unittest.TestCase):
    ...
```

**Pattern for conditional resources:**
```python
def setUp(self):
    if not os.path.exists(self.required_resource):
        self.skipTest("Required resource not available")
```

### CI/CD Best Practices

**CRITICAL: Pytest runs unittest-based tests natively!**

Never run both pytest and unittest on the same test files - this doubles CI time:

```bash
# ❌ WRONG - runs tests twice (doubles CI time from ~7min to ~15min+)
coverage run -m pytest tests/
coverage run --append -m unittest discover -s tests

# ✅ CORRECT - pytest handles both pytest AND unittest style tests
coverage run -m pytest tests/
```

**Why this matters:**
- All `test_*.py` files using `unittest.TestCase` are discovered and run by pytest
- Running unittest separately re-runs the exact same tests
- With 3000+ tests and coverage overhead, this can add 10+ minutes to CI

**When modifying `.github/workflows/ci.yml`:**
1. Read the header comment explaining the test architecture
2. Add new tests to the appropriate stage (smoke, unit, integration, etc.)
3. Never add duplicate test runners in the coverage-report job
4. When in doubt, run locally first: `time python -m pytest tests/ -v`

---

## Common Tasks

### Adding a New Analysis Function

1. Add function to `analysis.py` with proper signature:
   ```python
   def compute_your_analysis(
       layers: Dict[CorticalLayer, HierarchicalLayer],
       **kwargs
   ) -> Dict[str, Any]:
       """Your analysis description."""
       layer0 = layers[CorticalLayer.TOKENS]
       # Implementation
       return {'result': ..., 'stats': ...}
   ```

2. Add wrapper method to `CorticalTextProcessor` in the `processor/` package (appropriate mixin):
   ```python
   def compute_your_analysis(self, **kwargs) -> Dict[str, Any]:
       """Wrapper with docstring."""
       return compute_your_analysis(self.layers, **kwargs)
   ```

3. Add tests in `tests/test_analysis.py`

### Adding a New Query Function

1. Add to the `query/` package following existing patterns (e.g., `query/search.py`)
2. Use `get_expanded_query_terms()` helper for query expansion
3. Use `layer.get_by_id()` for O(1) lookups, not iteration
4. Add wrapper to the `processor/` package (likely `processor/query_api.py`)
5. Add tests in `tests/test_processor.py`

### Modifying Minicolumn Structure

1. Update `Minicolumn` class in `minicolumn.py`
2. Update `to_dict()` and `from_dict()` for persistence
3. Update `__slots__` if adding new fields
4. Increment state version in `persistence.py` if breaking change
5. Add migration logic for backward compatibility

---

## Code Style Guidelines

```python
# Imports: stdlib, then local
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn

# Type hints on all public functions
def find_documents(
    query: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Find documents matching query.

    Args:
        query: Search query string
        layers: Dictionary of hierarchical layers
        top_n: Number of results to return

    Returns:
        List of (doc_id, score) tuples sorted by relevance
    """
    # Implementation
```

---

## Scoring Algorithms

The processor supports multiple scoring algorithms for term weighting:

### BM25 (Default)

BM25 (Best Match 25) is the default scoring algorithm, optimized for code search:

```python
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig

# BM25 with default parameters (recommended)
config = CorticalConfig(scoring_algorithm='bm25')

# Tune BM25 parameters if needed
config = CorticalConfig(
    scoring_algorithm='bm25',
    bm25_k1=1.2,  # Term frequency saturation (0.0-3.0, default 1.2)
    bm25_b=0.75   # Length normalization (0.0-1.0, default 0.75)
)
processor = CorticalTextProcessor(config=config)
```

**Parameters:**
- `bm25_k1`: Controls term frequency saturation. Higher values give more weight to term frequency.
- `bm25_b`: Controls document length normalization. Set to 0.0 to disable length normalization.

### TF-IDF (Legacy)

Traditional TF-IDF scoring is still available:

```python
config = CorticalConfig(scoring_algorithm='tfidf')
```

### Graph-Boosted Search (GB-BM25)

A hybrid search combining BM25 with graph signals:

```python
# Standard search (uses BM25 under the hood)
results = processor.find_documents_for_query("query")

# Graph-boosted search (adds PageRank + proximity signals)
results = processor.graph_boosted_search(
    "query",
    pagerank_weight=0.3,   # Weight for term importance (0-1)
    proximity_weight=0.2   # Weight for connected terms (0-1)
)
```

**GB-BM25 combines:**
1. BM25 base score (term relevance)
2. PageRank boost (important terms rank higher)
3. Proximity boost (connected query terms boost documents)
4. Coverage boost (documents matching more terms rank higher)

---

## Performance Considerations

1. **Use `get_by_id()` for ID lookups** - O(1) vs O(n) iteration
2. **Batch document additions** with `add_documents_batch()` for bulk imports
3. **Use incremental updates** with `add_document_incremental()` for live systems
4. **Cache query expansions** when processing multiple similar queries
5. **Pre-compute chunks** in `find_passages_batch()` to avoid redundant work
6. **Use `fast_find_documents()`** for ~2-3x faster search on large corpora
7. **Pre-build index** with `build_search_index()` for fastest repeated queries
8. **Watch for O(n²) patterns** in loops over connections—use limits like `max_bigrams_per_term`
9. **Use `graph_boosted_search()`** for hybrid scoring with PageRank signals

---

## Code Search Capabilities

### Code-Aware Tokenization
```python
# Enable identifier splitting for code search
tokenizer = Tokenizer(split_identifiers=True)
tokens = tokenizer.tokenize("getUserCredentials")
# ['getusercredentials', 'get', 'user', 'credentials']
```

### Programming Concept Expansion
```python
# Expand queries with programming synonyms (get/fetch/load)
results = processor.expand_query("fetch data", use_code_concepts=True)
# Or use the convenience method
results = processor.expand_query_for_code("fetch data")
```

### Intent-Based Search
```python
# Parse natural language queries
parsed = processor.parse_intent_query("where do we handle authentication?")
# {'intent': 'location', 'action': 'handle', 'subject': 'authentication', ...}

# Search with intent understanding
results = processor.search_by_intent("how do we validate input?")
```

### Semantic Fingerprinting
```python
# Compare code similarity
fp1 = processor.get_fingerprint(code_block_1)
fp2 = processor.get_fingerprint(code_block_2)
comparison = processor.compare_fingerprints(fp1, fp2)
explanation = processor.explain_similarity(fp1, fp2)
```

### Fast Search
```python
# Fast document search (~2-3x faster)
results = processor.fast_find_documents("authentication")

# Pre-built index for fastest search
index = processor.build_search_index()
results = processor.search_with_index("query", index)
```

---

## Debugging Tips

### Inspecting Layer State
```python
processor = CorticalTextProcessor()
processor.process_document("test", "Neural networks process data.")
processor.compute_all()

# Check layer sizes
for layer_enum, layer in processor.layers.items():
    print(f"{layer_enum.name}: {layer.column_count()} minicolumns")

# Inspect a specific minicolumn
col = processor.layers[CorticalLayer.TOKENS].get_minicolumn("neural")
print(f"PageRank: {col.pagerank}")
print(f"TF-IDF: {col.tfidf}")
print(f"Connections: {len(col.lateral_connections)}")
print(f"Documents: {col.document_ids}")
```

### Tracing Query Expansion
```python
expanded = processor.expand_query("neural networks", max_expansions=10)
for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    print(f"  {term}: {weight:.3f}")
```

### Checking Semantic Relations
```python
processor.extract_corpus_semantics()
for t1, rel, t2, weight in processor.semantic_relations[:10]:
    print(f"{t1} --{rel}--> {t2} ({weight:.2f})")
```

### Profiling Performance
```bash
# Profile full analysis phases with timeout detection
python scripts/profile_full_analysis.py

# This reveals which phases are slow and helps identify O(n²) bottlenecks
```

### Observability and Metrics

The processor includes built-in observability features for tracking performance and operational metrics.

**Enable metrics collection:**
```python
# Create processor with metrics enabled
processor = CorticalTextProcessor(enable_metrics=True)

# Process documents and run queries (all operations are timed)
processor.process_document("doc1", "Neural networks process data.")
processor.compute_all()
processor.find_documents_for_query("neural networks")

# Get metrics summary
print(processor.get_metrics_summary())
```

**Access metrics programmatically:**
```python
metrics = processor.get_metrics()

# Check specific operation stats
if "compute_all" in metrics:
    stats = metrics["compute_all"]
    print(f"Average: {stats['avg_ms']:.2f}ms")
    print(f"Count: {stats['count']}")
    print(f"Min: {stats['min_ms']:.2f}ms")
    print(f"Max: {stats['max_ms']:.2f}ms")

# Check cache performance
if "query_cache_hits" in metrics:
    hits = metrics["query_cache_hits"]["count"]
    misses = metrics["query_cache_misses"]["count"]
    hit_rate = hits / (hits + misses) * 100
    print(f"Cache hit rate: {hit_rate:.1f}%")
```

**Automatically timed operations:**
- `compute_all()` and all compute phases (PageRank, TF-IDF, clustering, etc.)
- `process_document()` with doc_id context
- `find_documents_for_query()` with query context
- `save()` operations
- Query cache hits/misses via `expand_query_cached()`

**Control metrics collection:**
```python
# Disable metrics temporarily
processor.disable_metrics()
# ... operations not timed ...
processor.enable_metrics()

# Reset all metrics
processor.reset_metrics()

# Record custom metrics
processor.record_metric("api_calls", 10)
processor.record_metric("documents_processed", 100)
```

**Demo:**
```bash
# Run the observability demo
python examples/observability_demo.py
```

---

## Quick Reference

| Task | Command/Method |
|------|----------------|
| Process document | `processor.process_document(id, text)` |
| Build network | `processor.compute_all()` |
| Search | `processor.find_documents_for_query(query)` |
| Fast search | `processor.fast_find_documents(query)` |
| Hybrid search | `processor.graph_boosted_search(query)` |
| Code search | `processor.expand_query_for_code(query)` |
| Intent search | `processor.search_by_intent("where do we...")` |
| RAG passages | `processor.find_passages_for_query(query)` |
| Fingerprint | `processor.get_fingerprint(text)` |
| Compare | `processor.compare_fingerprints(fp1, fp2)` |
| Save state (JSON) | `processor.save("corpus_state")` (recommended) |
| Save state (pkl) | `processor.save("corpus.pkl", format='pickle')` (deprecated) |
| Load state | `processor = CorticalTextProcessor.load("corpus_state")` (auto-detects format) |
| Enable metrics | `processor = CorticalTextProcessor(enable_metrics=True)` |
| Get metrics | `processor.get_metrics()` |
| Metrics summary | `processor.get_metrics_summary()` |
| Reset metrics | `processor.reset_metrics()` |
| Record metric | `processor.record_metric("name", count)` |
| Run all tests | `python scripts/run_tests.py all` |
| Run smoke tests | `python scripts/run_tests.py smoke` |
| Run unit tests | `python scripts/run_tests.py unit` |
| Run quick tests | `python scripts/run_tests.py quick` (smoke + unit) |
| Run pre-commit | `python scripts/run_tests.py precommit` (smoke + unit + integration) |
| Run performance | `python scripts/run_tests.py performance` (no coverage) |
| Check coverage | `python -m coverage run --source=cortical -m pytest tests/ && python -m coverage report --include="cortical/*"` |
| Run showcase | `python showcase.py` |
| Profile analysis | `python scripts/profile_full_analysis.py` |
| Create memory | `python scripts/new_memory.py "topic"` |
| Create decision | `python scripts/new_memory.py "topic" --decision` |
| Session handoff | `python scripts/session_handoff.py` |
| Generate session memory | `python scripts/session_memory_generator.py --session-id ID` |
| Check wiki-links | `python scripts/resolve_wiki_links.py FILE` |
| Find backlinks | `python scripts/resolve_wiki_links.py --backlinks FILE` |
| Complete task with memory | `python scripts/task_utils.py complete TASK_ID --create-memory` |
| View sprint status | `python scripts/task_utils.py sprint status` |
| Mark sprint goal complete | `python scripts/task_utils.py sprint complete "goal text"` |
| Add sprint note | `python scripts/task_utils.py sprint note "note text"` |
| Create orchestration plan | `python scripts/orchestration_utils.py generate --type plan` |
| List orchestration plans | `python scripts/orchestration_utils.py list` |
| Verify batch | `python scripts/verify_batch.py --quick` |
| View orchestration metrics | From Python: `OrchestrationMetrics().get_summary()` |
| **Reasoning Framework** | |
| Reasoning demo | `python scripts/reasoning_demo.py --quick` |
| Reasoning with persistence | `python scripts/reasoning_demo.py --quick --persist` |
| Graph persistence demo | `python examples/graph_persistence_demo.py` |
| Validate persistence | `python scripts/validate_reasoning_persistence.py` |
| **Graph Persistence API** | |
| Create GraphWAL | `GraphWAL(wal_dir="/path/to/wal")` |
| Log node | `wal.log_add_node(node_id, node_type, content)` |
| Log edge | `wal.log_add_edge(source_id, target_id, edge_type)` |
| Create snapshot | `wal.create_snapshot(graph, compress=True)` |
| Load snapshot | `graph = wal.load_snapshot(snapshot_id)` |
| Check recovery needed | `GraphRecovery(wal_dir).needs_recovery()` |
| Recover graph | `result = GraphRecovery(wal_dir).recover()` |
| Git auto-commit | `GitAutoCommitter(repo_path).commit_on_save(path, graph)` |
| **GoT Handoff Primitives** | |
| Initiate handoff | `python scripts/got_utils.py handoff initiate TASK_ID --target AGENT --instructions "..."` |
| Accept handoff | `python scripts/got_utils.py handoff accept HANDOFF_ID --agent AGENT` |
| Complete handoff | `python scripts/got_utils.py handoff complete HANDOFF_ID --agent AGENT --result JSON` |
| List handoffs | `python scripts/got_utils.py handoff list [--status STATUS]` |
| Compact events | `python scripts/got_utils.py compact [--preserve-days N]` |
| **GoT Query Language** | |
| What blocks task | `python scripts/got_utils.py query "what blocks TASK_ID"` |
| What depends on | `python scripts/got_utils.py query "what depends on TASK_ID"` |
| Find path | `python scripts/got_utils.py query "path from ID1 to ID2"` |
| All relationships | `python scripts/got_utils.py query "relationships TASK_ID"` |
| Active tasks | `python scripts/got_utils.py query "active tasks"` |
| Pending tasks | `python scripts/got_utils.py query "pending tasks"` |
| Blocked tasks | `python scripts/got_utils.py query "blocked tasks"` |
| **Performance Tests** | |
| Run perf tests | `python -m pytest tests/performance/test_graph_persistence_perf.py -v` |
| Run E2E tests | `python -m pytest tests/integration/test_reasoning_persistence_e2e.py -v` |

### Orchestration Utilities

For Director orchestration and parallel agent workflows:

- `scripts/orchestration_utils.py` - Director orchestration tracking (plans, batches, metrics)
- `scripts/verify_batch.py` - Automated batch verification

See `.claude/commands/director.md` for comprehensive orchestration documentation.

---

## Persistence Format Migration

**⚠️ IMPORTANT:** Pickle format is deprecated due to security concerns (Remote Code Execution vulnerability). JSON is now the default and recommended format.

### Why JSON?

- **Secure**: No code execution risk (pickle can execute arbitrary code when loading)
- **Git-friendly**: Human-readable diffs, no merge conflicts
- **Cross-platform**: Works across Python versions and platforms
- **Debuggable**: Can inspect state without loading into Python

### Migration from Pickle to JSON

```bash
# Migrate existing pickle files to JSON
python -c "
from cortical.processor import CorticalTextProcessor
processor = CorticalTextProcessor.load('corpus_dev.pkl')  # Auto-detects pickle
processor.save('corpus_dev.json')  # Saves as JSON directory
"
```

**Or use the processor API:**
```python
from cortical.processor import CorticalTextProcessor

# Load from pickle (auto-detects format)
processor = CorticalTextProcessor.load('old_corpus.pkl')

# Save as JSON
processor.save('new_corpus')  # Creates directory with JSON files
```

### Backward Compatibility

Existing pickle files will continue to work with deprecation warnings:

```python
# Load automatically detects format
processor = CorticalTextProcessor.load('corpus.pkl')  # DeprecationWarning

# Explicit format specification
processor = CorticalTextProcessor.load('corpus.pkl', format='pickle')

# Save with explicit pickle format (not recommended)
processor.save('corpus.pkl', format='pickle')  # DeprecationWarning
```

### Format Detection

The `load()` method auto-detects format based on file content (not extension):
- **Directory** → JSON format (StateLoader)
- **File starting with `{`** → JSON format
- **File with pickle magic bytes** → Pickle format

---

## Dog-Fooding: Search the Codebase

The Cortical Text Processor can index and search its own codebase, providing semantic search capabilities during development.

### Quick Start

```bash
# Index the codebase (creates corpus_dev.json/, ~2s)
python scripts/index_codebase.py

# Incremental update (only changed files)
python scripts/index_codebase.py --incremental

# Search for code
python scripts/search_codebase.py "PageRank algorithm"
python scripts/search_codebase.py "bigram separator" --verbose
python scripts/search_codebase.py --interactive

# Legacy pickle format (deprecated)
python scripts/index_codebase.py --output corpus_dev.pkl --format pkl
```

### Claude Skills

Four skills are available in `.claude/skills/`:

1. **codebase-search**: Search the indexed codebase for code patterns and implementations
2. **corpus-indexer**: Re-index the codebase after making changes
3. **ai-metadata**: View pre-generated module metadata for rapid understanding
4. **memory-manager**: Create and manage knowledge memories (learnings, decisions, concepts)

### Indexer Options

| Option | Description |
|--------|-------------|
| `--incremental`, `-i` | Only re-index changed files (fastest) |
| `--status`, `-s` | Show what would change without indexing |
| `--force`, `-f` | Force full rebuild |
| `--log FILE` | Write detailed log to file |
| `--verbose`, `-v` | Show per-file progress |
| `--use-chunks` | Use git-compatible chunk-based storage |
| `--compact` | Compact old chunk files (with `--use-chunks`) |

### Search Options

| Option | Description |
|--------|-------------|
| `--top N` | Number of results (default: 5) |
| `--verbose` | Show full passage text |
| `--expand` | Show query expansion terms |
| `--interactive` | Interactive search mode |

### Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `/expand <query>` | Show query expansion |
| `/concepts` | List concept clusters |
| `/stats` | Show corpus statistics |
| `/quit` | Exit interactive mode |

### Example Queries

```bash
# Find how PageRank is implemented
python scripts/search_codebase.py "compute pagerank damping factor"

# Find test patterns
python scripts/search_codebase.py "unittest setUp processor"

# Explore query expansion code
python scripts/search_codebase.py "expand query semantic lateral"
```

### Git-Compatible Chunk-Based Indexing

For team collaboration, use chunk-based indexing which stores document changes as git-friendly JSON files:

```bash
# Index with chunk storage (creates corpus_chunks/*.json)
python scripts/index_codebase.py --incremental --use-chunks

# Check chunk status
python scripts/index_codebase.py --status --use-chunks

# Compact old chunks (reduces git history size)
python scripts/index_codebase.py --compact --before 2025-12-01
```

**Architecture:**
```
corpus_chunks/                        # Tracked in git (append-only)
├── 2025-12-10_21-53-45_a1b2.json    # Session 1 changes
├── 2025-12-10_22-15-30_c3d4.json    # Session 2 changes
└── 2025-12-10_23-00-00_e5f6.json    # Session 3 changes

corpus_dev.pkl                        # NOT tracked (local cache)
corpus_dev.pkl.hash                   # NOT tracked (cache validation)
```

**Benefits:**
- No merge conflicts (unique timestamp+session filenames)
- Shared indexed state across team/branches
- Fast startup when cache is valid
- Git-friendly (small JSON, append-only)
- Periodic compaction like `git gc`

### Chunk Compaction

Over time, chunk files accumulate. Use compaction to consolidate them, similar to `git gc`:

**When to compact:**
- After many indexing sessions (10+ chunk files)
- When you see size warnings during indexing
- Before merging branches with different chunk histories
- To clean up deleted/modified document history

**Compaction commands:**
```bash
# Compact all chunks into a single consolidated file
python scripts/index_codebase.py --compact --use-chunks

# Compact only chunks created before a specific date
python scripts/index_codebase.py --compact --before 2025-12-01 --use-chunks

# Check chunk status before compacting
python scripts/index_codebase.py --status --use-chunks
```

**How compaction works:**
1. Reads all chunk files (sorted by timestamp)
2. Replays operations in order (later timestamps override)
3. Creates a single compacted chunk with final state
4. Removes old chunk files
5. Preserves cache if still valid

**Recommended frequency:**
- Weekly for active development
- Monthly for maintenance repositories
- Before major releases

---

## Text-as-Memories: Knowledge Management

The project uses a text-as-memories system to capture and preserve institutional knowledge. Documents are treated as memories that, when stored in git, form a persistent, searchable knowledge base.

### Memory Types

| Type | Location | Purpose |
|------|----------|---------|
| **Daily Memories** | `samples/memories/YYYY-MM-DD-*.md` | Capture daily learnings and insights |
| **Decision Records** | `samples/decisions/adr-NNN-*.md` | Document architectural decisions |
| **Concept Docs** | `samples/memories/concept-*.md` | Consolidated knowledge on topics |

### Creating Memories

**Daily Memory:**
```bash
# Capture a learning
cat > samples/memories/$(date +%Y-%m-%d)-topic.md << 'EOF'
# Memory Entry: YYYY-MM-DD Topic

**Tags:** `tag1`, `tag2`
**Related:** [[other-doc.md]]

## What I Learned
- Key insight here

## Connections
- How this relates to other knowledge
EOF
git add samples/ && git commit -m "memory: topic insight"
```

**Decision Record:**
```bash
# Document a decision
cat > samples/decisions/adr-001-title.md << 'EOF'
# ADR-001: Title

**Status:** Accepted
**Date:** YYYY-MM-DD

## Context
What problem are we solving?

## Decision
What did we decide?

## Consequences
What are the trade-offs?
EOF
```

### Searching Memories

```bash
# Index memories for search
python scripts/index_codebase.py --incremental

# Search across code AND memories
python scripts/search_codebase.py "what did we learn about validation"
```

### Best Practices

1. **Write immediately** - Capture insights while fresh
2. **Use consistent tags** - Improves searchability
3. **Link related docs** - Use `[[wiki-style]]` references
4. **Commit to git** - Memories are only persistent once committed
5. **Consolidate periodically** - Merge related memories into concept docs

### Integration with Tasks

When completing a task, consider creating a memory entry from the retrospective:
- What was learned?
- What connections were made?
- What should future developers know?

See `docs/text-as-memories.md` for the full guide.

---

## ML Data Collection: Project-Specific Micro-Model

**Fully automatic. Zero configuration required.**

ML data collection starts automatically when you open this project in Claude Code. Every session is tracked, every commit is captured, and transcripts are saved when sessions end.

### Automatic Startup

When a Claude Code session starts in this project:
1. **Session tracking begins** - A new ML session is created for commit-chat linking
2. **Git hooks are installed** - post-commit and pre-push hooks are added if missing
3. **Stats are displayed** - Current collection progress is shown

This is configured in `.claude/settings.local.json` via the `SessionStart` hook.

### What Gets Collected

| Data Type | Location | Contents |
|-----------|----------|----------|
| **Commits** | `.git-ml/commits/` | Git history with diff hunks, temporal context, CI results |
| **Chats** | `.git-ml/chats/` | Query/response pairs with files touched and tools used |
| **Sessions** | `.git-ml/sessions/` | Development sessions linking chats to commits |
| **Actions** | `.git-ml/actions/` | Individual tool uses and operations |

**Note:** All ML data is stored in `.git-ml/` which is gitignored and regeneratable via backfill.

### Quick Commands

```bash
# Check collection progress
python scripts/ml_data_collector.py stats

# Estimate when training becomes viable
python scripts/ml_data_collector.py estimate

# Validate collected data
python scripts/ml_data_collector.py validate

# Session management
python scripts/ml_data_collector.py session status
python scripts/ml_data_collector.py session start
python scripts/ml_data_collector.py session end --summary "What was accomplished"

# Generate session handoff document
python scripts/ml_data_collector.py handoff

# Record CI results (manual)
python scripts/ml_data_collector.py ci set --commit abc123 --result pass --coverage 89.5

# CI auto-capture (reads from GitHub Actions environment)
python scripts/ml_data_collector.py ci-autocapture

# Backfill historical commits
python scripts/ml_data_collector.py backfill -n 100

# Collect GitHub PR/Issue data (requires gh CLI)
python scripts/ml_data_collector.py github collect           # Collect recent PRs and issues
python scripts/ml_data_collector.py github stats             # Show GitHub data counts
python scripts/ml_data_collector.py github fetch-pr --number 42  # Fetch specific PR
```

### Disabling Collection

```bash
# Disable for current session
export ML_COLLECTION_ENABLED=0

# Stats and validation still work when disabled
```

### File Prediction Model

The first ML model is available: **predict which files to modify** based on a task description.

```bash
# Train the model on commit history
python scripts/ml_file_prediction.py train

# Predict files for a task
python scripts/ml_file_prediction.py predict "Add authentication feature"

# Evaluate model performance (80/20 train/test split)
python scripts/ml_file_prediction.py evaluate --split 0.2

# View model statistics
python scripts/ml_file_prediction.py stats
```

**How it works:**
- Extracts commit type patterns (feat:, fix:, docs:, refactor:, etc.)
- Builds file co-occurrence matrix from commit history
- Maps keywords from commit messages to files
- Uses TF-IDF-style scoring with frequency penalties

**For comprehensive training guidance**, see [docs/ml-training-best-practices.md](docs/ml-training-best-practices.md) covering:
- Data quality guidelines and filtering strategies
- Training workflow and when to retrain
- Performance optimization and hyperparameter tuning
- Common pitfalls (overfitting, staleness, data leakage)
- Evaluation metrics interpretation (MRR, Recall@K, Precision@K)
- Integration with git hooks and CI/CD

**Prediction with seed files:**
```bash
# If you know some files, boost co-occurring files
python scripts/ml_file_prediction.py predict "Fix related bug" --seed auth.py login.py
```

**Current metrics** (403 commits, 20% test split):
| Metric | Value | Description |
|--------|-------|-------------|
| MRR | 0.43 | First correct prediction ~position 2-3 |
| Recall@10 | 0.48 | Half of actual files in top 10 |
| Precision@1 | 0.31 | 31% of top predictions correct |

**Model storage:** `.git-ml/models/file_prediction.json`

**Training requirements:** See [docs/ml-milestone-thresholds.md](docs/ml-milestone-thresholds.md) for detailed explanation of why 500 commits are needed for reliable file prediction.

#### Pre-Commit File Suggestions

The ML file prediction is integrated into git as a pre-commit hook that suggests potentially missing files:

```bash
# Automatically installed when you run:
python scripts/ml_data_collector.py install-hooks

# Creates .git/hooks/prepare-commit-msg
```

**How it works:**
1. You run `git commit -m "feat: Add authentication"`
2. Hook analyzes the commit message
3. Hook runs ML file prediction
4. If high-confidence files aren't staged, warns you
5. You can choose to add them or proceed

**Example output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 ML File Prediction Suggestion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on your commit message, these files might need changes:

  • tests/test_authentication.py                 (confidence: 0.823)
  • docs/api.md                                  (confidence: 0.654)

Staged files:
  ✓ cortical/authentication.py

ℹ️  Tip: Review the suggestions above.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Configuration (via environment variables):**
- `ML_SUGGEST_ENABLED=0` - Disable suggestions (default: 1)
- `ML_SUGGEST_THRESHOLD=0.7` - Confidence threshold (default: 0.5)
- `ML_SUGGEST_BLOCKING=1` - Block commit if missing files (default: 0)
- `ML_SUGGEST_TOP_N=10` - Number of predictions to check (default: 5)

**When it runs:**
- ✅ Regular commits (`git commit -m "..."`)
- ❌ Merge commits, amends, rebases (too noisy)
- ❌ Empty commits or no staged files
- ❌ Model not trained (silently skips)

**Testing without committing:**
```bash
bash scripts/test-ml-precommit-hook.sh
```

See [docs/ml-precommit-suggestions.md](docs/ml-precommit-suggestions.md) for detailed documentation.

### Automatic Session Capture

**Pre-configured. No setup needed.**

The ML data collector automatically captures complete session transcripts when Claude Code sessions end. This is already configured in `.claude/settings.local.json`.

**What gets captured automatically:**
- Full query/response pairs from the transcript
- All tool uses (Task, Read, Edit, Bash, Grep, etc.)
- Files referenced and modified
- Thinking blocks (if present)
- Session linkage to commits

**Process transcript manually:**
```bash
# Process a specific transcript file
python scripts/ml_data_collector.py transcript --file /path/to/transcript.jsonl

# Dry run (show what would be captured without saving)
python scripts/ml_data_collector.py transcript --file /path/to/transcript.jsonl --dry-run --verbose
```

**Generate session memory:**
```bash
# Generate a draft memory document from current session
python scripts/session_memory_generator.py --session-id abc123

# Generate from recent commits (no session ID needed)
python scripts/session_memory_generator.py --commits 10

# Dry run (preview without saving)
python scripts/session_memory_generator.py --session-id abc123 --dry-run
```

**What gets auto-generated:**
- Session summary with commit history
- Files modified during the session
- Task IDs referenced in commits
- Categorized file changes (Core Library, Tests, Scripts, etc.)
- Auto-extracted insights and tags
- Draft memory saved to `samples/memories/[DRAFT]-YYYY-MM-DD-session-{id}.md`

**When it runs:**
- Automatically at session end via `ml-session-capture-hook.sh`
- Manually via command line for any session
- Integrated into the Stop hook workflow

### Integration

Data collection is fully automatic via hooks configured in `.claude/settings.local.json`:

| Hook | Trigger | Action |
|------|---------|--------|
| **SessionStart** | Session begins | Starts ML session, installs git hooks, shows stats |
| **Stop** | Session ends | Captures full transcript with all exchanges, generates draft memory |
| **prepare-commit-msg** | Before commit | Suggests missing files based on commit message |
| **post-commit** | After commit | Captures commit metadata with diff hunks |
| **pre-push** | Before push | Reports collection stats |
| **CI workflow** | GitHub Actions | Auto-captures CI pass/fail results |

**Hook files:**
- `scripts/ml-session-start-hook.sh` - SessionStart handler
- `scripts/ml-session-capture-hook.sh` - Stop handler (includes session memory generation)
- `scripts/session_memory_generator.py` - Auto-generates draft memory documents
- `scripts/ml-precommit-suggest.sh` - prepare-commit-msg handler

**CI Integration:**
The GitHub Actions workflow (`.github/workflows/ci.yml`) includes an `ml-ci-capture` job that automatically records CI results for each commit. This runs after the coverage-report job and captures:
- Pass/fail status
- Coverage percentage (when available)
- Workflow and job metadata
- Run ID for traceability

See `.claude/skills/ml-logger/SKILL.md` for detailed logging usage.

---

## File Quick Links

- **Main API**: `cortical/processor/` - `CorticalTextProcessor` class (split into mixins)
- **Graph algorithms**: `cortical/analysis.py` - PageRank, TF-IDF, clustering
- **Search**: `cortical/query/` - query expansion, document retrieval (split into 8 modules)
- **Data structures**: `cortical/minicolumn.py` - `Minicolumn`, `Edge`
- **Configuration**: `cortical/config.py` - `CorticalConfig` dataclass
- **Tests**: `tests/test_processor.py` - most comprehensive test file
- **Demo**: `showcase.py` - interactive demonstration

**Process Documentation:**
- **Getting Started**: `docs/quickstart.md` - 5-minute tutorial for newcomers
- **Contributing**: `CONTRIBUTING.md` - how to contribute (fork, test, PR workflow)
- **Ethics**: `docs/code-of-ethics.md` - documentation, testing, and completion standards
- **Dog-fooding**: `docs/dogfooding-checklist.md` - checklist for testing with real usage
- **Definition of Done**: `docs/definition-of-done.md` - when is a task truly complete?
- **Text-as-Memories**: `docs/text-as-memories.md` - knowledge management guide
- **Task Management**: `docs/merge-friendly-tasks.md` - merge-friendly task system with collision-free IDs
- **ML Milestone Thresholds**: `docs/ml-milestone-thresholds.md` - why 500/2000/5000 commits for training
- **Merge-Friendly Tasks**: See "Task Management (Merge-Friendly System)" section above for task workflow

---

*Remember: Measure before optimizing, test before committing, and document what you discover.*
