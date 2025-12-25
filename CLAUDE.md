# CLAUDE.md - Cortical Cognitive Architecture Guide

---

## Overview

**Cortical Text Processor** is a cognitive architecture for intelligent agents. It provides:

1. **Hierarchical Knowledge Representation** - Text organized through 4 layers (tokens → bigrams → concepts → documents)
2. **Graph-Based Reasoning** - Chains, trees, and networks of thought with typed relationships
3. **Transactional State Management** - ACID-compliant task/decision tracking with full audit trails
4. **Agent Coordination** - Pub/sub messaging, work boundaries, and approval workflows
5. **Adaptive Learning** - User profiles, habits, and self-organizing knowledge structures

> **Platform Support:** Linux and macOS only. Windows is not supported (uses POSIX `fcntl.flock()`).

---

## Part I: Quick Session Start

**New session? Start here.**

### 1. Validate System State (30 seconds)
```bash
python scripts/got_utils.py validate      # Health check
python scripts/got_utils.py task list --status in_progress  # Active work
```

### 2. Restore Context (2 minutes)
```bash
# Read most recent knowledge transfer
ls -t samples/memories/*knowledge-transfer*.md | head -1 | xargs cat

# Check sprint status
python scripts/got_utils.py sprint status
```

### 3. What is GoT?

GoT (Graph of Thought) is the transactional task, sprint, and decision tracking system:

| Entity | Purpose | Example ID |
|--------|---------|------------|
| **Tasks** | Work items | `T-20251221-014654-d4b7` |
| **Sprints** | Time-boxed work periods | `S-sprint-017-spark-slm` |
| **Epics** | Large initiatives | `EPIC-nlu` |
| **Decisions** | Logged choices with rationale | `D-20251222-093045-a1b2` |
| **Handoffs** | Agent-to-agent work transfers | `H-20251222-093045-a1b2c3d4` |
| **Edges** | Relationships | `DEPENDS_ON`, `BLOCKS`, `CONTAINS` |

### 4. Essential Commands

```bash
# Task management
python scripts/got_utils.py task create "Title" --priority high
python scripts/got_utils.py task start T-XXX
python scripts/got_utils.py task complete T-XXX

# Decision logging
python scripts/got_utils.py decision log "Decision" --rationale "Why"

# Sprint status
python scripts/got_utils.py sprint status
python scripts/got_utils.py sprint list

# System health
python scripts/got_utils.py validate
python scripts/got_utils.py dashboard
```

> **Never delete GoT files directly!** Use `got task delete` which handles cleanup properly.

### 5. Work Priority Order

| Priority | Type | Rule |
|----------|------|------|
| 0 | **Tests First** | Write failing tests before implementation |
| 1 | **Security** | Fix vulnerabilities immediately |
| 2 | **Bugs** | Reproduce with test, then fix |
| 3 | **Features** | Define with tests, then implement |
| 4 | **Documentation** | Update as you work |

---

## Part II: Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CORTICAL COGNITIVE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     KNOWLEDGE LAYER                                    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │   TOKENS    │ │   BIGRAMS   │ │  CONCEPTS   │ │  DOCUMENTS  │     │   │
│  │  │   Layer 0   │→│   Layer 1   │→│   Layer 2   │→│   Layer 3   │     │   │
│  │  │  (words)    │ │  (phrases)  │ │ (clusters)  │ │   (full)    │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     REASONING LAYER                                    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │  Cognitive  │ │   Thought   │ │  Production │ │   Crisis    │     │   │
│  │  │    Loops    │ │   Graphs    │ │    State    │ │  Manager    │     │   │
│  │  │   (QAPV)    │ │ (networks)  │ │ (artifacts) │ │ (recovery)  │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     COORDINATION LAYER                                 │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │   Pub/Sub   │ │   Approval  │ │    Work     │ │   Handoff   │     │   │
│  │  │   Broker    │ │   Workflows │ │ Boundaries  │ │  Protocol   │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     PERSISTENCE LAYER                                  │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │    WAL      │ │  Snapshots  │ │  Versioned  │ │     Git     │     │   │
│  │  │ (durability)│ │  (recovery) │ │   Store     │ │ (history)   │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
cortical/
├── processor/        # Public API - mixin-based composition
│   ├── core.py       # Initialization, staleness tracking
│   ├── documents.py  # Document add/remove operations
│   ├── compute.py    # PageRank, TF-IDF, clustering
│   ├── query_api.py  # Search, expansion, retrieval
│   └── persistence_api.py  # Save/load/export
│
├── reasoning/        # Cognitive processing framework
│   ├── cognitive_loop.py    # QAPV cycle implementation
│   ├── thought_graph.py     # Network-based thought representation
│   ├── workflow.py          # Orchestration coordinator
│   ├── verification.py      # Multi-level verification pyramid
│   ├── crisis_manager.py    # Failure detection and recovery
│   ├── collaboration.py     # Multi-agent coordination
│   ├── pubsub.py           # Topic-based messaging
│   └── graph_persistence.py # WAL, snapshots, git integration
│
├── got/              # Graph of Thought - transactional state
│   ├── api.py        # GoTManager - high-level operations
│   ├── types.py      # Entity, Task, Decision, Sprint, Handoff
│   ├── tx_manager.py # Transaction management
│   ├── wal.py        # Write-Ahead Log
│   ├── query_builder.py  # Fluent query API
│   └── versioned_store.py # Storage with history
│
├── query/            # Search and retrieval (8 modules)
├── analysis/         # Graph algorithms
├── spark/            # Statistical language model
└── utils/            # Shared utilities
```

---

## Part III: Knowledge Management

### Knowledge Generators

Knowledge generators create structured knowledge from raw input:

```python
from cortical import CorticalTextProcessor

# Process raw documents into hierarchical knowledge
processor = CorticalTextProcessor()
processor.process_document("doc1", text)
processor.compute_all()

# Extract semantic relations
processor.extract_corpus_semantics()
relations = processor.semantic_relations  # [(term1, relation, term2, weight), ...]

# Generate concept clusters (self-organizing)
concepts = processor.get_layer(CorticalLayer.CONCEPTS)
for cluster in concepts.minicolumns.values():
    print(f"Theme: {cluster.id}, Terms: {cluster.document_ids}")
```

**Types of Knowledge Generated:**

| Generator | Output | Use Case |
|-----------|--------|----------|
| `process_document()` | Token/bigram/concept layers | Raw text → structured |
| `extract_corpus_semantics()` | Typed relations | IS_A, PART_OF, CAUSES |
| `build_concept_clusters()` | Semantic clusters | Topic discovery |
| `detect_knowledge_gaps()` | Missing connections | Knowledge audit |

### Knowledge Fetchers

Retrieve specific knowledge by query or ID:

```python
# By direct ID lookup (O(1))
col = processor.layers[CorticalLayer.TOKENS].get_by_id("L0_neural")

# By query expansion
expanded = processor.expand_query("neural networks", max_expansions=10)
# Returns: {"neural": 1.0, "network": 0.9, "deep": 0.7, ...}

# By semantic search
results = processor.find_documents_for_query("authentication patterns")
```

### Knowledge Searchers

Advanced search patterns:

```python
# Graph-boosted search (BM25 + PageRank + proximity)
results = processor.graph_boosted_search(
    "authentication",
    pagerank_weight=0.3,   # Term importance
    proximity_weight=0.2   # Connected term boost
)

# Passage retrieval for RAG
passages = processor.find_passages_for_query(
    "OAuth token refresh",
    top_n=5,
    chunk_size=200,
    chunk_overlap=50
)

# Intent-based search
results = processor.search_by_intent("where do we handle authentication?")
```

### Self-Organizing Data

The system organizes knowledge without explicit structure:

```python
# Automatic concept clustering via Louvain algorithm
processor.compute_all()  # Includes build_concept_clusters()

# The algorithm:
# 1. Builds co-occurrence graph from term connections
# 2. Detects communities using modularity optimization
# 3. Creates Layer 2 (CONCEPTS) from discovered clusters
```

**Self-Organization Triggers:**

| Event | Reorganization |
|-------|----------------|
| New documents added | TF-IDF recalculated, clusters updated |
| Threshold reached | Concepts re-clustered |
| Gap detected | Bridge concepts suggested |
| Pattern repeated | Habit/learning recorded |

---

## Part IV: Reasoning Topologies

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     REASONING TOPOLOGY SPECTRUM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CHAIN OF THOUGHT          TREE OF THOUGHT          GRAPH OF THOUGHT    │
│  ═══════════════           ═══════════════          ═══════════════     │
│                                                                          │
│  Linear sequence           Hierarchical branching   Network with cycles │
│                                                                          │
│  A → B → C → D             A                        A ←→ B              │
│                           /│\                        ↕   ↕              │
│                          B C D                      C ←→ D              │
│                         /|   |\                      ↕                  │
│                        E F   G H                     E                  │
│                                                                          │
│  Best for:              Best for:                  Best for:            │
│  • Simple tasks         • Complex decisions        • Knowledge bases    │
│  • Step-by-step         • Multiple approaches      • Investigations     │
│  • Clear sequence       • Parallel exploration     • Interconnected     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Chain of Thought (QAPV Loop)

The foundational reasoning pattern:

```python
from cortical.reasoning import CognitiveLoop, LoopPhase

loop = CognitiveLoop(goal="Implement authentication")
loop.start()

# Question Phase: What needs to be understood?
loop.transition_to(LoopPhase.QUESTION)
loop.record_question("What auth method is preferred?")

# Answer Phase: Gather information
loop.transition_to(LoopPhase.ANSWER)
loop.record_decision("Use OAuth 2.0", rationale="Industry standard")

# Produce Phase: Create artifacts
loop.transition_to(LoopPhase.PRODUCE)
loop.add_artifact("src/auth.py", "OAuth service implementation")

# Verify Phase: Check correctness
loop.transition_to(LoopPhase.VERIFY)
loop.add_verification("pytest tests/test_auth.py", status="passed")

loop.complete()
```

### Tree of Thought (Nested Loops)

For complex tasks requiring parallel exploration:

```python
from cortical.reasoning import NestedLoopExecutor

executor = NestedLoopExecutor()

# Root loop
root = executor.create_loop("Implement full authentication system")

# Child loops for parallel work streams
oauth_loop = executor.spawn_child(root, "Implement OAuth service")
storage_loop = executor.spawn_child(root, "Implement token storage")
integration_loop = executor.spawn_child(root, "Integrate with frontend")

# Execute tree (respects dependencies)
results = executor.execute_tree(root)
```

### Graph of Thought (ThoughtGraph)

For knowledge networks with complex relationships:

```python
from cortical.reasoning import ThoughtGraph, NodeType, EdgeType

graph = ThoughtGraph()

# Add nodes of different types
graph.add_node("C1", NodeType.CONCEPT, "OAuth 2.0",
    properties={'definition': '...', 'examples': [...]})
graph.add_node("Q1", NodeType.QUESTION, "How to handle token expiry?")
graph.add_node("H1", NodeType.HYPOTHESIS, "Use sliding window refresh")
graph.add_node("D1", NodeType.DECISION, "Implement OAuth 2.0")
graph.add_node("E1", NodeType.EVIDENCE, "OWASP recommends OAuth")

# Connect with typed edges
graph.add_edge("D1", "C1", EdgeType.IMPLEMENTS)
graph.add_edge("D1", "E1", EdgeType.SUPPORTED_BY)
graph.add_edge("Q1", "H1", EdgeType.SUGGESTS)

# Analyze
reachable = graph.get_reachable_nodes("D1")
cycles = graph.find_cycles()  # Detect circular reasoning
clusters = graph.find_clusters()  # Find topic groups
```

### Auto-Generating Reasoning Topologies

The system suggests appropriate structures based on task type:

```python
from cortical.reasoning import TopologySelector

selector = TopologySelector()

# Analyze task and suggest topology
task = "Debug why authentication fails intermittently"
recommendation = selector.analyze(task)

# Returns:
# {
#   'suggested_topology': 'graph',
#   'rationale': 'Investigation tasks benefit from connecting symptoms,
#                 hypotheses, and evidence non-linearly',
#   'template': 'investigation_graph'
# }

# Auto-generate the graph structure
graph = selector.create_from_template(recommendation['template'])
```

**Topology Selection Heuristics:**

| Task Pattern | Suggested Topology | Reason |
|--------------|-------------------|--------|
| "Implement X step by step" | Chain | Clear sequential steps |
| "Compare options for X" | Tree | Parallel evaluation branches |
| "Investigate why X" | Graph | Non-linear evidence gathering |
| "Refactor X" | Tree | Hierarchical decomposition |
| "Design system for X" | Graph | Interconnected components |

### Processes on Graphs

Operations that analyze and transform thought graphs:

```python
from cortical.got import GraphWalker, PathFinder, PatternMatcher

# 1. Traversal - Visit nodes in order
walker = GraphWalker(got_manager)
results = walker.starting_from(start_id) \
    .follow("DEPENDS_ON") \
    .max_depth(3) \
    .bfs() \
    .visit(lambda node, acc: acc + [node.id], initial=[]) \
    .run()

# 2. Path Finding - Find connections
finder = PathFinder(got_manager)
path = finder.shortest_path(task_a, task_b)
all_reachable = finder.reachable_from(start_node)

# 3. Pattern Matching - Find structures
pattern = Pattern() \
    .node("a", type="task", status="blocked") \
    .edge("DEPENDS_ON") \
    .node("b", type="task", status="blocked")

blocked_chains = PatternMatcher(got_manager).find(pattern)
```

### Metrics on High-Pressure Areas

Identify critical knowledge points requiring review:

```python
from cortical.reasoning import GraphMetrics

metrics = GraphMetrics(thought_graph)
report = metrics.analyze()

# Returns:
# {
#   'hotspots': [
#     {'node': 'D-auth-decision', 'score': 0.95,
#      'reason': 'High in-degree (many depend on this)'},
#   ],
#   'bottlenecks': [
#     {'edge': ('D1', 'T1'), 'score': 0.72,
#      'reason': 'Single path to critical task'}
#   ],
#   'review_priority': ['D-auth-decision', 'Q-token-refresh']
# }
```

**Metrics Collected:**

| Metric | Description | Review Trigger |
|--------|-------------|----------------|
| In-degree | Nodes depending on this | > 5 dependents |
| Betweenness | Bridge between clusters | > 0.7 centrality |
| Staleness | Time since last update | > 7 days |
| Uncertainty | Hypothesis vs. Fact ratio | > 0.5 uncertain |
| Contradiction | Conflicting evidence | Any conflicts |

---

## Part V: User Profiles, Learnings, and Habits

### User Profile Management

Track preferences and context across sessions:

```python
from cortical.profiles import UserProfile

# Create or load user profile
profile = UserProfile.load("user_123")

# Record preferences
profile.set_preference("code_style", "functional")
profile.set_preference("verbosity", "concise")
profile.set_preference("approval_threshold", "medium")  # low/medium/high/paranoid

# Record context
profile.add_context("current_project", "cortical-text-processor")
profile.add_context("expertise_level", "senior")

profile.save()
```

### Learning Records

Capture and apply learnings from interactions:

```python
from cortical.profiles import LearningRecord, LearningType

# Record a learning
learning = LearningRecord(
    type=LearningType.CORRECTION,
    context="User corrected my understanding of OAuth flow",
    before="OAuth uses session tokens",
    after="OAuth uses access tokens with refresh mechanism",
    confidence=0.95,
    applies_to=["authentication", "oauth", "tokens"]
)
profile.add_learning(learning)

# Query learnings
relevant = profile.get_learnings_for_context("oauth implementation")
```

**Learning Types:**

| Type | Description | Application |
|------|-------------|-------------|
| `CORRECTION` | User corrected agent output | Avoid same mistake |
| `PREFERENCE` | User expressed preference | Apply to future choices |
| `INSIGHT` | New understanding gained | Enrich knowledge base |
| `HABIT` | Repeated user behavior | Anticipate needs |
| `BOUNDARY` | What NOT to do | Hard constraints |

### Habit Detection

Automatically detect patterns in user behavior:

```python
from cortical.profiles import HabitDetector

detector = HabitDetector(profile)
habits = detector.detect()

# [
#   Habit(pattern="user always runs tests before commit",
#         confidence=0.92, occurrences=47),
#   Habit(pattern="user prefers functional style over OOP",
#         confidence=0.88, occurrences=23)
# ]

# Auto-apply habits
if detector.should_suggest("run tests"):
    suggest("Would you like me to run tests before committing?")
```

---

## Part VI: Integration Points

### Background Workers and Queues

Asynchronous processing for long-running tasks:

```python
from cortical.workers import TaskQueue, Worker, Priority

# Create task queue
queue = TaskQueue(persistence_dir=".queue")

# Enqueue work
job_id = queue.enqueue(
    task="reindex_corpus",
    payload={"corpus_id": "main", "full_rebuild": True},
    priority=Priority.LOW,
    ttl_seconds=3600  # Expire after 1 hour
)

# Process asynchronously
worker = Worker(queue)

@worker.handler("reindex_corpus")
def handle_reindex(payload):
    processor.reindex(payload["corpus_id"], full=payload["full_rebuild"])
    return {"status": "complete"}

# Start worker
worker.start()

# Check job status
status = queue.get_status(job_id)
```

**Queue Features:**

| Feature | Description |
|---------|-------------|
| Priority levels | `CRITICAL`, `HIGH`, `NORMAL`, `LOW` |
| TTL expiration | Jobs expire if not processed |
| Retry with backoff | Failed jobs retry with exponential backoff |
| Dead letter queue | Failed jobs preserved for analysis |
| Persistence | Queue survives restarts |

### Pub/Sub System

Topic-based messaging for agent coordination:

```python
from cortical.reasoning import PubSubBroker

broker = PubSubBroker()

# Subscribe to topics (supports wildcards)
broker.subscribe("task.*.completed", "analytics")
broker.subscribe("security.alert.*", "security_monitor")
broker.subscribe("approval.required", "human_reviewer")

# Publish events
broker.publish(
    topic="task.auth.completed",
    payload={"task_id": "T-123", "result": "success"},
    sender="agent_1"
)

# Poll for messages
messages = broker.poll("analytics")
for msg in messages:
    process_analytics(msg.payload)
    broker.acknowledge(msg.id, "analytics")

# Dead letter queue for failed deliveries
failed = broker.get_dead_letters("analytics")
```

### Custom Integration Points

Define custom adapters for external systems:

```python
from cortical.integrations import IntegrationAdapter, IntegrationRegistry

class GitHubAdapter(IntegrationAdapter):
    """Custom integration with GitHub."""

    def on_task_complete(self, task):
        """Create PR when task completes."""
        if task.metadata.get("create_pr"):
            self.create_pull_request(task)

    def on_decision_made(self, decision):
        """Log decisions to GitHub discussions."""
        self.create_discussion(decision)

# Register adapter
registry = IntegrationRegistry()
registry.register("github", GitHubAdapter(token))
```

---

## Part VII: Data Security and Access Control

### Data Classification

All data is classified for appropriate access control:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA CLASSIFICATION LEVELS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  EPHEMERAL          WORKING           CURATED          SECURED          │
│  ═════════          ═══════           ═══════          ═══════          │
│                                                                          │
│  Temporary          Active work       Validated        Critical          │
│  scratch data       in progress       knowledge        data              │
│                                                                          │
│  • Cache files      • Draft docs      • Decisions      • Credentials    │
│  • Temp results     • WIP code        • Knowledge      • User data      │
│  • Session state    • Task state      • Memories       • Audit logs     │
│                                                                          │
│  Agent: FULL        Agent: FULL       Agent: ADD+LOG   Agent: REQUEST   │
│  History: NO        History: 7 days   History: FULL    History: FULL    │
│  Approval: NO       Approval: NO      Delete: NOTIFY   All: APPROVE     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Approval Workflows

Agents request approval for protected operations:

```python
from cortical.security import ApprovalWorkflow, ApprovalRequest

workflow = ApprovalWorkflow()

# Agent requests permission to delete
request = ApprovalRequest(
    action="delete",
    target_id="decision-123",
    target_class=DataClass.SECURED,
    requestor="agent_1",
    reason="Consolidating duplicate decisions",
    context={
        "current_content": "...",
        "affected_entities": ["task-1", "task-2"],
        "reversibility": "can restore from history"
    }
)

# Submit request (notifies user)
request_id = workflow.submit(request)

# User reviews and decides
workflow.approve(request_id, reviewer="user_1", comment="Approved")
# or
workflow.reject(request_id, reviewer="user_1", reason="Keep separate")

# Agent checks status
status = workflow.get_status(request_id)
if status.approved:
    got_manager.delete_decision("decision-123")
```

### Temporary Access Grants

Grant time-limited elevated access:

```python
from cortical.security import TemporaryGrant, GrantCondition

# Grant temporary delete access for refactoring
grant = TemporaryGrant(
    grantee="agent_1",
    permissions=[Permission.DELETE],
    targets=["tasks/*", "decisions/*"],  # Glob patterns
    conditions=[
        GrantCondition.TIME_LIMIT(hours=4),
        GrantCondition.OPERATION_LIMIT(max_operations=100),
        GrantCondition.REQUIRES_BACKUP(True),
        GrantCondition.NOTIFY_ON_USE(True)
    ],
    reason="Sprint cleanup refactoring",
    granted_by="user_1"
)

# Apply grant
grant_id = access_manager.grant(grant)

# Agent operations within grant
with access_manager.use_grant(grant_id):
    # Operations here have elevated permissions
    # Still logged, still backed up, user notified
    for task in old_tasks:
        got_manager.delete_task(task.id)

# Grant auto-expires after time limit or operation limit
```

**Grant Conditions:**

| Condition | Description |
|-----------|-------------|
| `TIME_LIMIT` | Expires after duration |
| `OPERATION_LIMIT` | Max number of operations |
| `REQUIRES_BACKUP` | Must backup before modify |
| `NOTIFY_ON_USE` | Notify user each time used |
| `REQUIRES_REVIEW` | Changes staged, not applied until reviewed |
| `SCOPE_LIMIT` | Only specific files/entities |

### Preventing Unauthorized Deletion

**Critical safeguard: Agents cannot delete secured data without approval.**

```python
from cortical.security import SafetyGuard

guard = SafetyGuard()

@guard.protect_delete
def delete_entity(entity_id):
    entity = got_manager.get_entity(entity_id)

    # Check data classification
    if entity.data_class == DataClass.SECURED:
        raise ApprovalRequiredError(
            f"Cannot delete secured data '{entity_id}' without approval. "
            f"Use ApprovalWorkflow.submit() to request permission."
        )

    if entity.data_class == DataClass.CURATED:
        notify_user(f"Agent deleting curated data: {entity_id}")

    return _do_delete(entity_id)
```

---

## Part VIII: Versioning and History

### Design Philosophy

**Version everything at the storage layer, expose selectively.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VERSIONING STRATEGY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Storage Layer: ALWAYS version (cheap, essential for recovery)          │
│  API Layer: Expose based on data classification                         │
│  Retention: Based on data class and age                                 │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  EPHEMERAL    │  No history exposed  │  Delete after session   │    │
│  │  WORKING      │  7-day history       │  Compact after 30 days  │    │
│  │  CURATED      │  Full history        │  Compact after 1 year   │    │
│  │  SECURED      │  Full history        │  Never delete           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Entity Versioning

Every entity tracks its own version:

```python
entity = got_manager.get_task("T-123")

print(entity.version)      # 5 (modified 5 times)
print(entity.created_at)   # 2025-12-20T10:00:00Z
print(entity.modified_at)  # 2025-12-25T14:30:00Z

# Each modification bumps version
got_manager.update_task("T-123", status="completed")
# entity.version is now 6
```

### Historical Access

Read entities as they existed at any point:

```python
from cortical.got import VersionedStore

store = VersionedStore(".got")

# Read at specific global version
historical = store.read_at_version("T-123", version=42)

# List all versions of an entity
history = store.get_history("T-123")
for entry in history:
    print(f"Version {entry.global_version}: {entry.modified_at}")
```

### Snapshot Isolation

Transactions see consistent point-in-time views:

```python
# Transaction 1 starts
with got_manager.transaction() as tx1:
    task = tx1.get_task("T-123")  # Sees version 5

    # Transaction 2 commits while tx1 is active
    # T-123 is now version 6

    # tx1 still sees version 5 (snapshot isolation)
    task = tx1.get_task("T-123")  # Still version 5
```

### Retention Policies

Configure how long history is kept:

```python
from cortical.got import RetentionPolicy

policies = {
    DataClass.EPHEMERAL: RetentionPolicy(
        retain_history=False,
        delete_after=timedelta(hours=24)
    ),
    DataClass.WORKING: RetentionPolicy(
        retain_history=True,
        compact_after=timedelta(days=7)
    ),
    DataClass.SECURED: RetentionPolicy(
        retain_history=True,
        compact_after=None,  # Never compact
        delete_after=None    # Never delete
    )
}

store.set_retention_policies(policies)
```

---

## Part IX: Workflows and Cognitive Processes

### Standard Workflows

Pre-defined workflows for common tasks:

```python
from cortical.workflows import (
    InvestigationWorkflow,
    ImplementationWorkflow,
    RefactoringWorkflow
)

# Investigation: Debug why something fails
investigation = InvestigationWorkflow(goal="Debug auth failures")
investigation.add_symptom("Users report intermittent 401 errors")
investigation.add_hypothesis("Token expiry race condition")
investigation.add_evidence("Logs show requests 2ms after expiry")
investigation.conclude("Implement grace period for token validation")

# Implementation: Build a feature
implementation = ImplementationWorkflow(goal="Add OAuth support")
implementation.define_requirements(["Support Google", "Support GitHub"])
implementation.design(approach="Adapter pattern for providers")
implementation.implement(files=["src/auth/oauth.py"])
implementation.verify(tests=["tests/test_oauth.py"])
implementation.complete()
```

### Cognitive Process Best Practices

| Practice | Description | Why |
|----------|-------------|-----|
| **Write tests first** | TDD is mandatory | Tests define success criteria |
| **Read before modify** | Understand existing code | Prevents regressions |
| **Checkpoint before risky ops** | Save state before major changes | Enables safe rollback |
| **Escalate after 3 failures** | Don't loop indefinitely | Human may see what agent misses |
| **Log decisions with rationale** | Record why, not just what | Enables learning and audit |
| **Verify before claiming done** | Run tests, check coverage | Prevents false completion |

### Crisis Management

Failure detection and escalation:

```
CrisisLevel.HICCUP      # Self-recoverable (1)
    └→ Continue (fix & document)

CrisisLevel.OBSTACLE    # Needs adaptation (2)
    └→ Adapt (pause, analyze, adjust)

CrisisLevel.WALL        # Needs human intervention (3)
    └→ Escalate (stop, document, alert)

CrisisLevel.CRISIS      # Immediate stop required (4)
    └→ Stop (preserve state, alert immediately)
```

---

## Part X: Development Setup

### Prerequisites

```bash
# Check if dependencies are installed
python -c "import pytest; print(f'pytest OK: {pytest.__version__}')"
python -c "import coverage; print(f'coverage OK: {coverage.__version__}')"

# Install if missing
pip install pytest coverage
# OR: pip install -e ".[dev]"
```

### Baseline Coverage

```bash
# Run tests with coverage
python -m coverage run -m pytest tests/ -q

# View report
python -m coverage report --include="cortical/*"

# Current baseline: ~98%
```

### Test Categories

```bash
# Quick sanity check (~1s)
make test-smoke

# Fast tests (~5s)
make test-fast

# Before commit (~30s)
make test-quick

# Full suite with parallel execution
make test-parallel
```

---

## Part XI: Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Process document | `processor.process_document(id, text)` |
| Search documents | `processor.find_documents_for_query(query)` |
| RAG passages | `processor.find_passages_for_query(query)` |
| Save state | `processor.save("corpus")` |
| Load state | `CorticalTextProcessor.load("corpus")` |
| Run tests | `make test-smoke` or `make test-quick` |
| Create task | `python scripts/got_utils.py task create "Title"` |
| Log decision | `python scripts/got_utils.py decision log "Decision"` |
| View sprint | `python scripts/got_utils.py sprint status` |

### Key Patterns

```python
# CORRECT: O(1) lookup
col = layer.get_by_id("L0_neural")

# WRONG: O(n) iteration
for col in layer.minicolumns.values():
    if col.id == target: ...

# CORRECT: Bigrams use spaces
bigram = "neural networks"

# WRONG: Underscores
bigram = "neural_networks"

# CORRECT: Use canonical ID generation
from cortical.utils.id_generation import generate_task_id
task_id = generate_task_id()
```

### Critical Bugs (Don't Reintroduce)

- **Edge rebuild**: Use `from_id`/`to_id`, NOT `source_id`/`target_id`
- **EdgeType lookup**: Use `EdgeType[name]` with try/except, NOT `hasattr()`
- **Priority executor**: Skip query echo line when parsing blockers

---

## Part XII: Sub-Agent Delegation

### When to Delegate

| Situation | Agent Type | Example |
|-----------|-----------|---------|
| "Where is X?" | `Explore` (quick) | Find implementation patterns |
| "How does X work?" | `Explore` (thorough) | Understand full system flow |
| "Implement feature X" | `general-purpose` | Write tests, implement features |
| "How should we build X?" | `Plan` | Architecture decisions |

### Parallel Execution Pattern

```
Main Agent (keeps context):
├── Task 1: Complex work requiring full context (do this yourself)
├── Task 2: Complex work requiring decisions (do this yourself)
└── Spawn parallel sub-agents for mechanical tasks:
    ├── Sub-agent A: "Consolidate checksums.py"
    ├── Sub-agent B: "Consolidate query/utils.py"
    └── Sub-agent C: "Consolidate persistence.py"
```

### Verification After Sub-Agent Completion

> **Always verify sub-agent changes persisted:**

```bash
git status                    # Check if files actually changed
git diff path/to/file.py     # Verify the actual changes
```

---

## Conclusion

This architecture supports:

1. **Hierarchical Knowledge** - From tokens to documents
2. **Flexible Reasoning** - Chains, trees, and graphs of thought
3. **Safe Agent Operations** - Approval workflows for protected data
4. **Temporal Access Control** - Permissions that expire
5. **Full Auditability** - Version history and decision logs
6. **Extensibility** - Custom entities, workflows, and integrations

**Remember:**
- Measure before optimizing
- Test before committing
- Document what you discover
- Request approval for protected data
- Preserve history for recovery
