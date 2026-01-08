# Cortical Text Processor

> *"The measure of sophisticated software is not whether it can solve problems, but whether it can understand why it solved them, remember how it solved them, and explain itself to whatever comes next."*

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-10254%2B%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen.svg)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-orange.svg)

---

## The Vision

This is not just a text processing library.

It is an experiment in building software that *knows itself*—software that indexes its own source code, learns from its own evolution, externalizes its cognitive state, and guides the agents that will modify it.

Three principles animate this work:

**1. The Codebase as Prompt**

A codebase exists across three dimensions: *time* (git history, commits, branches), *compute* (what fits in context, what must be retrieved), and *space* (module topology, hot and cold regions). This project treats the codebase itself as a prompt—one that teaches machines how to understand and extend it.

**2. Work as Graph**

Tasks are not items in a list. They are nodes in a network of relationships—dependencies, decisions, justifications, handoffs. The Graph of Thought (GoT) system makes these relationships explicit and queryable, turning work management into graph traversal.

**3. Software That Reflects**

The system generates metadata about itself, indexes its own documentation, collects data about its own evolution, and uses that data to train models that predict how it will change. It is recursive: the eye that sees is also the thing being seen.

---

## What It Does

**Cortical Text Processor** is a zero-dependency Python library for hierarchical text analysis and semantic search. Despite the neuroscience-inspired naming, it uses proven information retrieval algorithms—**PageRank**, **TF-IDF**, **BM25**, and **Louvain clustering**—not neural networks.

```python
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()
processor.process_document("doc1", "Neural networks process information hierarchically.")
processor.process_document("doc2", "The brain uses layers of neurons for processing.")
processor.compute_all()

results = processor.find_documents_for_query("neural processing")
# [('doc1', 0.877), ('doc2', 0.832)]

processor.save("my_corpus")  # JSON format, git-friendly
```

**Key capabilities:**
- **Semantic search** with automatic query expansion
- **Code search** with identifier splitting (`getUserName` → `get`, `user`, `name`)
- **RAG support** with chunk-level passage retrieval
- **Knowledge analysis** detecting gaps, outliers, and missing connections
- **Graph of Thought** for task/decision tracking across sessions
- **Woven Mind** dual-process cognitive architecture (System 1/System 2)

**Zero dependencies.** Copy the `cortical/` folder into your project and go.

---

## The Architecture

### Four Layers, Like the Visual Cortex

Your visual cortex doesn't grep through pixels looking for cats. It builds hierarchies—edges become patterns, patterns become shapes, shapes become objects. This library applies the same principle to text:

| Layer | Name | Analogy | What It Captures |
|-------|------|---------|------------------|
| 0 | **Tokens** | V1 (edges) | Individual words, the atomic units |
| 1 | **Bigrams** | V2 (patterns) | Word pairs, local structure |
| 2 | **Concepts** | V4 (shapes) | Semantic clusters, emergent meaning |
| 3 | **Documents** | IT (objects) | Full documents, holistic understanding |

Each layer contains **minicolumns**—units that maintain four types of connections:

- **Lateral**: "What am I related to at my level?" (Hebbian learning: neurons that fire together wire together)
- **Feedforward**: "What am I made of?" (decomposition)
- **Feedback**: "What am I part of?" (composition)
- **Typed edges**: "How exactly do we relate?" (IsA, PartOf, Causes, etc.)

### The Algorithms Behind the Metaphors

| Algorithm | What It Does | Implementation |
|-----------|--------------|----------------|
| **PageRank** | Finds important terms by connectivity | `cortical/analysis/pagerank.py` |
| **TF-IDF / BM25** | Scores term distinctiveness | `cortical/analysis/tfidf.py` |
| **Louvain** | Clusters terms into concepts | `cortical/analysis/clustering.py` |
| **Query Expansion** | Finds related terms for search | `cortical/query/expansion.py` |

The neuroscience metaphors help intuition. The implementations are standard information retrieval.

---

## The Cognitive Layer

### Graph of Thought (GoT)

Traditional task management asks: *"What needs to be done?"*

GoT asks: *"How is everything connected?"*

```python
from cortical.got import GoTManager

manager = GoTManager(".got")

# Create tasks with explicit relationships
task = manager.create_task("Implement authentication", priority="high")
decision = manager.log_decision("Use JWT", rationale="Stateless, scales horizontally")
manager.add_edge(decision.id, task.id, "JUSTIFIES")

# Query the work graph
blockers = manager.get_blockers(task.id)
dependents = manager.get_dependents(task.id)
```

**Sixteen edge types** encode semantic relationships:

| Edge Type | Question It Answers |
|-----------|---------------------|
| `DEPENDS_ON` | What must be done first? |
| `BLOCKS` | What prevents this from starting? |
| `JUSTIFIES` | Why was this choice made? |
| `IMPLEMENTS` | How does this realize a decision? |
| `PRODUCES` | What does this create? |
| `TRANSFERS` | Where is this work going? |

**Four query mechanisms** for different modes of reasoning:

```python
from cortical.got.query_builder import Query
from cortical.got.graph_walker import GraphWalker
from cortical.got.path_finder import PathFinder
from cortical.got.pattern_matcher import Pattern, PatternMatcher

# SQL-like fluent queries
pending_high = (Query(manager)
    .tasks()
    .where(status="pending", priority="high")
    .order_by("created_at", desc=True)
    .execute())

# Graph traversal with visitor pattern
def count_by_status(node, acc):
    acc[node.status] = acc.get(node.status, 0) + 1
    return acc

counts = (GraphWalker(manager)
    .starting_from(task_id)
    .follow("DEPENDS_ON")
    .bfs()
    .visit(count_by_status, initial={})
    .run())

# Path finding
path = PathFinder(manager).shortest_path(task_a, task_b)
reachable = PathFinder(manager).reachable_from(start_task)

# Pattern matching
chain_pattern = (Pattern()
    .node("a", type="task")
    .outgoing("DEPENDS_ON")
    .node("b", type="task")
    .outgoing("DEPENDS_ON")
    .node("c", type="task"))

matches = PatternMatcher(manager).find(chain_pattern)
```

### Woven Mind: Dual-Process Cognition

Inspired by Kahneman's *Thinking, Fast and Slow*, the Woven Mind implements dual-process cognition:

```python
from cortical.reasoning.woven_mind import WovenMind

mind = WovenMind()
mind.train("neural networks process information")
mind.train("deep learning uses neural networks")

result = mind.process(["neural", "networks"])
print(f"Mode: {result.mode.name}")   # FAST or SLOW
print(f"Source: {result.source}")     # 'hive' or 'cortex'
```

| System | Name | Characteristics |
|--------|------|-----------------|
| **System 1** | The Hive | Fast, automatic, pattern-matching |
| **System 2** | The Cortex | Slow, deliberate, analytical |

**The Loom** sits between them, routing based on *surprise*—the gap between prediction and reality. When patterns match, stay fast. When they don't, slow down and think.

### Cognitive Loops (QAPV)

Complex tasks flow through structured phases:

```python
from cortical.reasoning import ReasoningWorkflow

workflow = ReasoningWorkflow()
ctx = workflow.start_session("Implement authentication")

workflow.begin_question_phase(ctx)
workflow.record_question(ctx, "What auth method? OAuth, JWT, or session?")

workflow.begin_answer_phase(ctx)
workflow.record_decision(ctx, "Use JWT", rationale="Stateless, scales well")

workflow.begin_produce_phase(ctx)
# Implementation happens here...

workflow.begin_verify_phase(ctx)
# Verification happens here...
```

**Question → Answer → Produce → Verify**. Each phase can spawn child loops. Each loop is serializable, resumable, and auditable.

---

## The Meta Layer

### A Codebase That Knows Itself

This repository is designed to be understood by AI agents. Every layer includes explicit affordances:

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Metadata** | `.ai_meta` files | Structured navigation for rapid understanding |
| **Self-indexing** | Dog-fooding | The system indexes and searches its own code |
| **ML Collection** | `.git-ml/` | Training data for project-specific models |
| **Memories** | `samples/memories/` | Persistent knowledge across sessions |
| **External State** | GoT + reasoning | Serializable cognition that survives context loss |

### AI Metadata Generation

```bash
python scripts/generate_ai_metadata.py

cat cortical/processor/__init__.py.ai_meta
```

Metadata files provide:
- Function signatures with `see_also` cross-references
- Complexity hints for expensive operations (`O(n²) where n = minicolumns`)
- Logical section groupings
- Test coverage mapping

### Self-Indexing (Dog-Fooding)

The system indexes its own source code:

```bash
# Quick incremental update (only changed files)
python scripts/index_codebase.py --incremental

# Search the indexed codebase
python scripts/search_codebase.py "PageRank algorithm"
python scripts/search_codebase.py "how to expand queries" --verbose
```

**Indexer options:**

```bash
# Full rebuild with semantic analysis (slower, more thorough)
python scripts/index_codebase.py --full-analysis --foreground

# Resumable batch mode (for environments with timeouts)
python scripts/index_codebase.py --full-analysis --batch --batch-size 20
python scripts/index_codebase.py --full-analysis --batch  # Run again to continue

# Check what would change without indexing
python scripts/index_codebase.py --status

# Git-friendly chunk-based storage (for team collaboration)
python scripts/index_codebase.py --incremental --use-chunks

# Compact old chunks (like git gc)
python scripts/index_codebase.py --compact --use-chunks
```

| Option | Purpose |
|--------|---------|
| `--incremental` | Only re-index changed files |
| `--full-analysis` | Run semantic PageRank and hybrid connections |
| `--batch` | Process in resumable batches |
| `--use-chunks` | Store as git-friendly JSON chunks |
| `--status` | Show changes without indexing |
| `--compact` | Consolidate old chunk files |

This creates a feedback loop: implement search → test on real code → find relevance issues → fix → test again.

### ML Data Collection

Every commit, conversation, and session is captured for training project-specific models:

```
.git-ml/
├── commits/    # Commit metadata + diff hunks
├── chats/      # Query/response pairs
├── sessions/   # Development sessions
└── models/     # Trained models (file prediction, etc.)
```

**File prediction model** (already trained):

```bash
python scripts/ml_file_prediction.py predict "Add authentication feature"
# Predicts which files are likely to change
```

The system learns its own structure. It discovers that changes to `auth.py` usually require changes to `auth_test.py`. It learns how it evolves.

### Text-as-Memories

Knowledge persists in markdown:

```bash
python scripts/new_memory.py "What I learned about validation"
python scripts/new_memory.py "Use JSON over pickle" --decision
```

Memories are indexed by the same system that indexes code. Search finds both implementations and the reasoning behind them.

---

## The Learning Layer

### A System That Learns From Its Own Work

While GoT tracks what work is done, the Learning Layer captures *how* it was done and *what was learned*. Every completed task, failed attempt, and decision becomes training data for AI agents that will work on this codebase.

The system implements a **dual-purpose philosophy**: GoT serves both as work management infrastructure and as a continuous data collection pipeline for machine learning.

### GoT-LearningCycle Integration

**Location:** `cortical/got/learning_integration.py`

The `GoTLearningBridge` converts task completions into structured learning experiences:

```python
from cortical.got.learning_integration import GoTLearningBridge

bridge = GoTLearningBridge(".got")

# When task completes, capture as learning experience
bridge.capture_task_completion(
    task_id="T-20260103-123456",
    retrospective="Used TDD. Tests passed first try. Key: write test before implementation.",
    files_changed=["api.py", "test_api.py"],
    approach="test-first",
    task_category="feature"
)

# When planning new task, retrieve relevant lessons
guidance = bridge.get_guidance_for_task(
    task_title="Implement user authentication",
    task_category="feature"
)
# Returns past successes, common pitfalls, recommended approaches
```

**Features:**
- Automatically tags experiences based on task properties
- Extracts patterns from multiple similar tasks
- Provides context-aware guidance for new work
- Stores experiences in `.got/learning/` subdirectory

### Failure Tracking

**CLI:** `python -m cortical.got failure`

Failed approaches are as valuable as successes—sometimes more so. The failure tracking system captures what *didn't* work:

```bash
# Log a failed attempt
python -m cortical.got failure log T-20260103-123456 \
    --attempt "Tried using library X for authentication" \
    --error "Library X conflicts with our zero-dependency policy" \
    --lesson "Build auth ourselves or use stdlib only"

# List recent failures
python -m cortical.got failure list --limit 10

# Show failures for specific task
python -m cortical.got failure show T-20260103-123456
```

**How it works:**
- Creates `FAILED_ATTEMPT` edges in the GoT graph
- Links failures to the tasks they blocked
- Prevents repeating the same failed approaches
- Exported as negative examples for training

### Commit-Task Linking

**Location:** `scripts/commit_task_linker.py`

Connects git commits to GoT tasks, creating a complete narrative from planning → decision → implementation → commit:

```bash
# Link commits to tasks (run after committing)
python scripts/commit_task_linker.py link

# Show commits for a task
python scripts/commit_task_linker.py show T-20260103-123456

# Export commit-task mappings
python scripts/commit_task_linker.py export --output commit_links.json
```

**Linking strategies:**
1. **Explicit references:** Searches commit messages for `T-XXXXX` patterns
2. **Semantic similarity:** Compares commit message to task title/description
3. **File overlap:** Matches files changed in commit to files mentioned in task

**Value:**
- Traces every line of code back to its motivating task
- Enables "why was this changed?" queries at the git level
- Provides code context for retrospective analysis
- Training data shows how tasks translate to code changes

### Training Data Export

**Location:** `scripts/training_data_exporter.py`

Exports GoT data as ML-ready datasets:

```bash
# Show what's available for export
python scripts/training_data_exporter.py stats

# Export all high-quality data
python scripts/training_data_exporter.py export --output ./training_data/

# Export specific types
python scripts/training_data_exporter.py export-decisions
python scripts/training_data_exporter.py export-retrospectives
python scripts/training_data_exporter.py export-handoffs
python scripts/training_data_exporter.py export-edges
```

**Output format (JSONL):**

```json
{"context": "Decision: Use JWT for auth", "decision": "Use JWT", "rationale": "Stateless, scales horizontally", "quality_score": 0.95}
{"task": "Fix authentication bug", "approach": "bugfix", "retrospective": "Root cause was...", "success": true, "quality_score": 0.87}
```

**Quality scoring:**
- Completeness (has rationale/retrospective?)
- Length (too short is low-signal)
- Specificity (concrete details vs vague statements)
- Outcome clarity (success/failure explicitly stated?)

**Exported types:**
- **Decisions:** What was chosen and why
- **Retrospectives:** What worked, what didn't, what was learned
- **Handoffs:** Context transfers between agents
- **Knowledge Transfers:** Session summaries
- **Edges:** Relationships (BLOCKS, DEPENDS_ON, JUSTIFIES, etc.)

### The Dual Purpose

Every action in GoT serves two masters:

| Action | Work Management Purpose | ML Training Purpose |
|--------|------------------------|---------------------|
| Create task | Track what needs doing | Example of task decomposition |
| Log decision | Justify architecture choice | Decision-making training data |
| Complete task | Mark progress | Success pattern + retrospective |
| Log failure | Avoid repeating mistakes | Negative example for learning |
| Create edge | Model dependencies | Relationship inference training |
| Write retrospective | Knowledge transfer | Reflection and learning data |

**The vision:** An AI that has read every task, decision, and retrospective in this repository will understand not just *what* the code does, but *why* it was written that way, *what alternatives were considered*, and *what approaches failed*.

**The feedback loop:**

```
1. Developer/AI completes task
   └─> GoT captures metadata, retrospective, files changed

2. Exporter creates training examples
   └─> JSONL files with quality scores

3. ML pipeline trains project-specific model
   └─> Model learns codebase patterns and reasoning

4. Next AI agent loads trained model
   └─> Benefits from accumulated knowledge

5. Agent completes tasks more effectively
   └─> Cycle continues, system gets smarter
```

This is **institutional memory as machine learning**. The codebase doesn't just preserve code—it preserves the reasoning behind the code.

---

## The Persistence Philosophy

### Transactions and Durability

All GoT operations are transactional:

```python
with manager.transaction() as tx:
    task = tx.create_task("Implement auth")
    tx.add_edge(decision_id, task.id, "JUSTIFIES")
    # Commits together or rolls back
```

Behind this:
- **Write-Ahead Log (WAL)**: Every operation logged before execution
- **Checksums**: Corruption detected on read
- **Snapshot isolation**: Reads see consistent point-in-time view

### Four-Level Recovery Cascade

When things go wrong:

```
Level 1: WAL Replay (fastest)
    └─ Replay operations since last snapshot

Level 2: Snapshot Rollback
    └─ Load previous consistent snapshot

Level 3: Git History Recovery
    └─ Extract state from git commits

Level 4: Event Reconstruction
    └─ Rebuild from raw operation log
```

Each level trades speed for thoroughness. The system automatically escalates until recovery succeeds.

### Git as Truth

All state lives in version control. The merge strategy is append-only events:

```
Branch A: creates task T-1
Branch B: creates task T-2
Merge: both tasks exist (no conflict)
```

Timestamp-based IDs prevent merge conflicts across branches.

---

## Quick Start

### Installation

```bash
git clone <repository-url>
cd cortical-text-processor
pip install -e .
```

Or simply copy `cortical/` into your project—zero dependencies.

### Run the Showcase

```bash
python showcase.py
```

Watch the system analyze 176 documents covering quantum computing to medieval falconry, discovering central concepts, expanding queries, and detecting knowledge gaps.

### Programmatic Usage

```python
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor()

# Add documents
processor.process_document("doc1", "Neural networks process hierarchically.")
processor.process_document("doc2", "The brain uses layers of neurons.")

# Build the network
processor.compute_all()

# Query
results = processor.find_documents_for_query("neural processing")

# RAG passages
passages = processor.find_passages_for_query("how neurons work", top_n=3)

# Save
processor.save("my_corpus")  # JSON format
```

---

## Core API Reference

### Document Processing

```python
processor.process_document(doc_id, content, metadata=None)
processor.add_document_incremental(doc_id, content)
processor.add_documents_batch([(doc_id, content, metadata), ...])
```

### Network Building

```python
processor.compute_all(
    verbose=False,
    connection_strategy='hybrid',  # 'document_overlap', 'semantic', 'embedding', 'hybrid'
    cluster_strictness=0.5,
    bridge_weight=0.3
)
```

### Query & Retrieval

```python
processor.find_documents_for_query(query, top_n=5)
processor.find_passages_for_query(query, top_n=5)
processor.expand_query(text, max_expansions=10)
processor.search_by_intent("where do we handle authentication?")
processor.expand_query_for_code("fetch data")  # Programming synonyms
```

### Semantics & Analysis

```python
processor.extract_corpus_semantics()
processor.analyze_knowledge_gaps()
processor.detect_anomalies(threshold=0.1)
processor.complete_analogy(a, b, c)  # a:b :: c:?
```

### Persistence

```python
processor.save("corpus")      # JSON directory (recommended)
processor.save("corpus.pkl")  # Pickle (deprecated, security risk)
processor = CorticalTextProcessor.load("corpus")
```

---

## GoT Command Line

```bash
# Task management
python -m cortical.got task create "Fix bug" --priority high
python -m cortical.got task start T-20251228-...
python -m cortical.got task complete T-20251228-... --notes "Fixed in 2 hours"

# Sprint management
python -m cortical.got sprint create "Sprint 1" --number 1
python -m cortical.got sprint status

# Decision logging
python -m cortical.got decision log "Use JWT" --rationale "Stateless, scales"

# Queries
python -m cortical.got query "what blocks T-..."
python -m cortical.got blocked
python -m cortical.got dashboard

# Handoffs (agent-to-agent)
python -m cortical.got handoff initiate T-... --target sub-agent --instructions "..."
python -m cortical.got handoff accept H-... --agent sub-agent
python -m cortical.got handoff complete H-... --agent sub-agent --result '{"status":"done"}'
```

---

## Package Structure

```
cortical/
├── processor/           # Main orchestrator (mixin-based composition)
│   ├── core.py          # Initialization, staleness tracking
│   ├── documents.py     # Document add/remove/batch
│   ├── compute.py       # PageRank, TF-IDF, clustering
│   ├── query_api.py     # Search, expansion, retrieval
│   └── persistence_api.py
├── query/               # Search & retrieval (8 modules)
│   ├── expansion.py     # Query expansion
│   ├── search.py        # Document search
│   ├── passages.py      # RAG passage retrieval
│   └── ...
├── analysis/            # Graph algorithms
│   ├── pagerank.py
│   ├── tfidf.py
│   └── clustering.py
├── reasoning/           # Cognitive architecture
│   ├── woven_mind.py    # Dual-process orchestration
│   ├── loom.py          # Mode switching, surprise detection
│   ├── cognitive_loop.py # QAPV cycles
│   └── graph_persistence.py  # WAL, snapshots, recovery
├── got/                 # Graph of Thought
│   ├── api.py           # GoTManager
│   ├── query_builder.py # Fluent Query API
│   ├── graph_walker.py  # Visitor pattern traversal
│   ├── path_finder.py   # BFS/DFS algorithms
│   ├── pattern_matcher.py
│   └── learning_integration.py  # Learning cycle bridge
└── utils/               # Shared utilities

tests/                   # 10,254+ tests
├── smoke/               # Quick sanity checks (~1s)
├── unit/                # Fast isolated tests
├── integration/         # Component interaction
└── performance/         # Timing regression

scripts/                 # Development and learning tools
├── got_utils.py         # GoT CLI (tasks, decisions, failures)
├── training_data_exporter.py  # Export ML training data
├── commit_task_linker.py      # Link commits to tasks
├── index_codebase.py    # Self-indexing dog-fooding
└── search_codebase.py   # Semantic code search

docs/concepts/           # Philosophical explorations
├── codebase-as-prompt.md
├── software-that-knows-itself.md
└── work-as-graph.md
```

---

## Use Cases

### When to Use

| Use Case | Why It Fits |
|----------|-------------|
| **Documentation search** | Learns domain terminology from corpus |
| **Code repository search** | Built-in identifier splitting and programming synonyms |
| **Knowledge base Q&A** | Query expansion finds related documents |
| **RAG/LLM context** | Chunk-level passage retrieval with scoring |
| **Offline environments** | Zero dependencies, no API calls |
| **Privacy-sensitive** | All processing is local |

### When Not to Use

| Scenario | Better Alternative |
|----------|-------------------|
| State-of-the-art semantic similarity | Sentence transformers, OpenAI embeddings |
| Millions of documents | Elasticsearch, vector databases |
| Cross-lingual search | Multilingual embedding models |

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Index 100 documents | ~1.3ms | BM25 scoring |
| Search query | ~0.15ms | Pre-computed TF-IDF |
| Add document (incremental) | ~50ms | Without full recompute |
| Full compute_all() | ~500ms | 100 docs, all algorithms |

**Corpus size recommendations:**
- < 1,000 docs: Perfect fit
- 1,000 - 10,000: Good fit, consider tuning
- 10,000+: Works, but consider dedicated infrastructure

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | AI agent onboarding and development guide |
| [docs/architecture.md](docs/architecture.md) | Technical architecture deep dive |
| [docs/graph-of-thought.md](docs/graph-of-thought.md) | GoT system documentation |
| [docs/woven-mind-user-guide.md](docs/woven-mind-user-guide.md) | Dual-process cognition guide |
| [docs/concepts/](docs/concepts/) | Philosophical explorations |

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and workflow
- Code style and testing requirements
- Pull request guidelines

Quality resources:
- [Definition of Done](docs/definition-of-done.md)
- [Code of Ethics](docs/code-of-ethics.md)

---

## The Recursive Loop

This README is itself part of the system it describes.

It will be indexed by the search system. It will be processed by the hierarchical layers. It will become nodes in the knowledge graph. Future agents will query it, modify it, and extend it.

The codebase is a prompt. The prompt teaches machines to understand the codebase. The understanding enables modification. The modification changes the prompt.

The loop closes. The system evolves.

---

## License

MIT License

---

> *"A codebase is not a static artifact. It is a trajectory through solution-space, a record of decisions accumulated over time. When an AI encounters a codebase, it's seeing a single frame of a movie that's been playing for months or years. The question is: how do we make that movie legible?"*
>
> — From [docs/concepts/codebase-as-prompt.md](docs/concepts/codebase-as-prompt.md)
