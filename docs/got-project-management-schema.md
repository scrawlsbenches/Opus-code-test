# Graph of Thought: Project Management Schema

This document defines how to use the Graph of Thought (GoT) framework for project management, replacing the file-based task/sprint/epic system with a graph-based approach.

## Table of Contents

1. [Overview](#overview)
2. [Node Type Mappings](#node-type-mappings)
3. [Edge Type Usage](#edge-type-usage)
4. [Property Conventions](#property-conventions)
5. [Query Patterns](#query-patterns)
6. [Merge Conflict Prevention](#merge-conflict-prevention)
7. [Migration Path](#migration-path)
8. [Example Usage](#example-usage)

---

## Overview

### Why Graph of Thought for Project Management?

The GoT framework provides:
- **Unified representation**: Tasks, sprints, epics, decisions, and insights in one graph
- **Rich relationships**: Multiple edge types capture dependencies, blockers, refinements
- **Persistence**: GraphWAL and snapshots provide durability and recovery
- **Cross-branch coordination**: Merge-friendly through timestamp-based IDs
- **Analysis capabilities**: Find critical paths, bottlenecks, orphaned tasks
- **Visualization**: Mermaid, DOT, ASCII output for stakeholder communication

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT GRAPH STRUCTURE                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Epic (GOAL)                                                 │
│    │                                                          │
│    ├─[CONTAINS]─> Sprint 1 (GOAL)                           │
│    │                │                                        │
│    │                ├─[CONTAINS]─> Task A (TASK)            │
│    │                │                 ├─[IMPLEMENTS]─> Feature (CONCEPT) │
│    │                │                 └─[DEPENDS_ON]─> Task B (TASK) │
│    │                │                                        │
│    │                └─[CONTAINS]─> Task C (TASK)            │
│    │                                 └─[BLOCKS]─> Task D     │
│    │                                                          │
│    └─[CONTAINS]─> Sprint 2 (GOAL)                           │
│                     └─[CONTAINS]─> Task E (TASK)            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Node Type Mappings

### Tasks: `NodeType.TASK`

**Purpose:** Represent work items to be done.

**Node ID Format:**
```
task:T-YYYYMMDD-HHMMSS-XXXX
```
Example: `task:T-20251213-143052-a1b2`

**Required Properties:**
```python
{
    "title": str,              # Human-readable task title
    "status": str,             # One of: pending, in_progress, completed, blocked, deferred
    "priority": str,           # One of: critical, high, medium, low
    "category": str,           # One of: arch, feature, bugfix, test, docs, refactor, debt
    "description": str,        # Detailed description (optional)
    "effort": str,             # One of: trivial, small, medium, large, xl (optional)
}
```

**Optional Properties:**
```python
{
    "assignee": str,           # Who's working on this
    "tags": List[str],         # Tags for categorization
    "created_at": str,         # ISO 8601 timestamp
    "updated_at": str,         # ISO 8601 timestamp
    "completed_at": str,       # ISO 8601 timestamp
    "estimated_hours": float,  # Time estimate
    "actual_hours": float,     # Actual time spent
}
```

**Metadata:**
```python
{
    "session_id": str,         # Session that created this task
    "branch": str,             # Git branch where task was created
    "files": List[str],        # Files affected by this task
    "commits": List[str],      # Commit hashes related to this task
}
```

**Example:**
```python
from cortical.reasoning import ThoughtGraph, NodeType

graph = ThoughtGraph()

task = graph.add_node(
    node_id="task:T-20251213-143052-a1b2",
    node_type=NodeType.TASK,
    content="Implement BM25 scoring algorithm",
    properties={
        "title": "Implement BM25 scoring algorithm",
        "status": "in_progress",
        "priority": "high",
        "category": "feature",
        "effort": "medium",
        "description": "Replace TF-IDF with BM25 for better relevance ranking"
    },
    metadata={
        "session_id": "a1b2",
        "branch": "feature/bm25-scoring",
        "files": ["cortical/analysis.py", "tests/test_analysis.py"]
    }
)
```

### Sprints: `NodeType.GOAL`

**Purpose:** Represent time-boxed work periods.

**Node ID Format:**
```
sprint:S-NNN          # Sequential sprint number
sprint:YYYY-MM        # Year-month for monthly sprints
```
Examples: `sprint:S-042`, `sprint:2025-12`

**Required Properties:**
```python
{
    "sprint_id": str,          # Unique sprint identifier
    "name": str,               # Sprint name or theme
    "status": str,             # One of: planned, active, completed, cancelled
    "start_date": str,         # ISO 8601 date
    "end_date": str,           # ISO 8601 date
}
```

**Optional Properties:**
```python
{
    "goals": List[str],        # Text descriptions of sprint goals
    "capacity": float,         # Team capacity in person-hours
    "velocity": float,         # Completed effort from previous sprint
    "notes": str,              # Strategic notes and decisions
}
```

**Example:**
```python
sprint = graph.add_node(
    node_id="sprint:S-042",
    node_type=NodeType.GOAL,
    content="Sprint 42: Search Relevance Improvements",
    properties={
        "sprint_id": "S-042",
        "name": "Search Relevance Improvements",
        "status": "active",
        "start_date": "2025-12-13",
        "end_date": "2025-12-27",
        "goals": [
            "Implement BM25 scoring",
            "Add semantic search prototype",
            "Improve test coverage to 65%"
        ],
        "capacity": 80.0,  # 2 weeks × 40 hours
        "velocity": 12.0   # From last sprint
    }
)
```

### Epics: `NodeType.GOAL`

**Purpose:** Represent large initiatives spanning multiple sprints.

**Node ID Format:**
```
epic:E-XXXX           # 4-char unique identifier
epic:NAME             # Descriptive name slug
```
Examples: `epic:E-a1b2`, `epic:search-overhaul`

**Required Properties:**
```python
{
    "epic_id": str,            # Unique epic identifier
    "name": str,               # Epic name
    "phase": str,              # One of: proposed, active, completed, deferred, cancelled
    "description": str,        # Detailed description
}
```

**Optional Properties:**
```python
{
    "owner": str,              # Epic owner/champion
    "start_date": str,         # ISO 8601 date
    "target_date": str,        # ISO 8601 date
    "priority": str,           # One of: critical, high, medium, low
    "business_value": str,     # Business justification
    "success_metrics": List[str],  # How to measure success
}
```

**Example:**
```python
epic = graph.add_node(
    node_id="epic:search-overhaul",
    node_type=NodeType.GOAL,
    content="Search System Overhaul",
    properties={
        "epic_id": "search-overhaul",
        "name": "Search System Overhaul",
        "phase": "active",
        "description": "Replace TF-IDF with modern ranking, add semantic search, improve relevance",
        "owner": "Engineering Team",
        "start_date": "2025-12-01",
        "target_date": "2026-03-31",
        "priority": "high",
        "success_metrics": [
            "Search relevance improved by 40% (user study)",
            "Query latency < 100ms p95",
            "Code search accuracy > 80%"
        ]
    }
)
```

### Decisions: `NodeType.DECISION`

**Purpose:** Represent architectural or strategic decisions.

**Node ID Format:**
```
decision:ADR-NNN      # Architecture Decision Record number
decision:TOPIC        # Descriptive topic
```

**Required Properties:**
```python
{
    "title": str,              # Decision title
    "status": str,             # One of: proposed, accepted, rejected, deprecated, superseded
    "date": str,               # ISO 8601 date
    "context": str,            # What problem are we solving?
    "decision": str,           # What did we decide?
    "consequences": str,       # What are the trade-offs?
}
```

**Example:**
```python
decision = graph.add_node(
    node_id="decision:ADR-001",
    node_type=NodeType.DECISION,
    content="Use BM25 instead of TF-IDF for search ranking",
    properties={
        "title": "Use BM25 instead of TF-IDF",
        "status": "accepted",
        "date": "2025-12-10",
        "context": "TF-IDF doesn't handle document length well, causing short docs to rank higher",
        "decision": "Implement BM25 with k1=1.2, b=0.75 for length normalization",
        "consequences": "Better relevance, but requires reindexing and parameter tuning"
    }
)
```

### Insights: `NodeType.INSIGHT`

**Purpose:** Represent learnings discovered during development.

**Node ID Format:**
```
insight:YYYY-MM-DD-TOPIC
```

**Required Properties:**
```python
{
    "title": str,              # Insight title
    "date": str,               # ISO 8601 date
    "discovery": str,          # What was learned
    "implications": List[str], # What does this mean for the project?
}
```

---

## Edge Type Usage

### Hierarchical Relationships

#### `EdgeType.CONTAINS`
**Direction:** Container → Contained
**Use cases:**
- Epic → Sprint: "Epic contains this sprint"
- Sprint → Task: "Sprint contains this task"
- Task → Subtask: "Task contains this subtask"

**Example:**
```python
# Epic contains sprint
graph.add_edge(
    "epic:search-overhaul",
    "sprint:S-042",
    EdgeType.CONTAINS,
    weight=1.0
)

# Sprint contains task
graph.add_edge(
    "sprint:S-042",
    "task:T-20251213-143052-a1b2",
    EdgeType.CONTAINS,
    weight=1.0
)
```

### Dependency Relationships

#### `EdgeType.DEPENDS_ON`
**Direction:** Dependent → Prerequisite
**Use cases:**
- Task A depends on Task B completing first
- Sprint depends on prior sprint outcomes
- Epic depends on technical foundation

**Example:**
```python
# Task A depends on Task B
graph.add_edge(
    "task:T-20251213-143052-a1b2",  # BM25 implementation
    "task:T-20251210-120000-c3d4",  # Refactor scoring module
    EdgeType.DEPENDS_ON,
    weight=1.0,
    confidence=1.0
)
```

#### `EdgeType.BLOCKS`
**Direction:** Blocker → Blocked
**Use cases:**
- Task X blocks Task Y until resolved
- Bug blocks feature implementation
- Decision blocks sprint progress

**Example:**
```python
# Bug blocks feature
graph.add_edge(
    "task:T-20251212-090000-bug1",  # Fix tokenization bug
    "task:T-20251213-143052-a1b2",  # BM25 implementation
    EdgeType.BLOCKS,
    weight=1.0,
    confidence=1.0
)
```

#### `EdgeType.PRECEDES`
**Direction:** Earlier → Later
**Use cases:**
- Task A should happen before Task B (not strict dependency)
- Sprint ordering
- Phase ordering

**Example:**
```python
# Sprint 41 precedes Sprint 42
graph.add_edge(
    "sprint:S-041",
    "sprint:S-042",
    EdgeType.PRECEDES,
    weight=1.0
)
```

### Refinement Relationships

#### `EdgeType.REFINES`
**Direction:** Specific → General
**Use cases:**
- Task refines a sprint goal
- Sprint refines an epic
- Decision refines a concept

**Example:**
```python
# Task refines sprint goal
graph.add_edge(
    "task:T-20251213-143052-a1b2",
    "sprint:S-042",
    EdgeType.REFINES,
    weight=0.8  # Partially refines the sprint goal
)
```

#### `EdgeType.IMPLEMENTS`
**Direction:** Implementation → Concept
**Use cases:**
- Task implements a feature
- Task implements an architecture decision
- Sprint implements strategic direction

**Example:**
```python
# Task implements decision
graph.add_edge(
    "task:T-20251213-143052-a1b2",
    "decision:ADR-001",
    EdgeType.IMPLEMENTS,
    weight=1.0
)
```

### Knowledge Relationships

#### `EdgeType.MOTIVATES`
**Direction:** Motivation → Action
**Use cases:**
- Insight motivates task creation
- Decision motivates sprint planning
- User feedback motivates epic

**Example:**
```python
# Insight motivates task
graph.add_edge(
    "insight:2025-12-10-tfidf-weakness",
    "task:T-20251213-143052-a1b2",
    EdgeType.MOTIVATES,
    weight=1.0
)
```

#### `EdgeType.SUPPORTS`
**Direction:** Evidence → Claim
**Use cases:**
- Test results support decision
- Metrics support epic success
- Retrospective supports sprint completion

**Example:**
```python
# Evidence supports decision
graph.add_edge(
    "evidence:benchmark-results",
    "decision:ADR-001",
    EdgeType.SUPPORTS,
    weight=0.9,
    confidence=0.95
)
```

---

## Property Conventions

### Node ID Format Standards

#### Tasks
```
Format:   task:T-YYYYMMDD-HHMMSS-XXXX
Example:  task:T-20251213-143052-a1b2

Where:
  - YYYYMMDD: Date created
  - HHMMSS:   Time created (24-hour)
  - XXXX:     4-char session ID
```

#### Sprints
```
Format 1: sprint:S-NNN        (sequential)
Example:  sprint:S-042

Format 2: sprint:YYYY-MM      (monthly)
Example:  sprint:2025-12
```

#### Epics
```
Format 1: epic:E-XXXX         (4-char ID)
Example:  epic:E-a1b2

Format 2: epic:NAME           (slug)
Example:  epic:search-overhaul
```

#### Decisions
```
Format:   decision:ADR-NNN
Example:  decision:ADR-001
```

#### Insights
```
Format:   insight:YYYY-MM-DD-TOPIC
Example:  insight:2025-12-10-bm25-discovery
```

### Standard Property Fields

#### All Nodes (Metadata)

```python
metadata = {
    "created_at": str,         # ISO 8601 timestamp
    "updated_at": str,         # ISO 8601 timestamp
    "session_id": str,         # Session that created/modified
    "branch": str,             # Git branch
    "author": str,             # Who created this (optional)
}
```

#### Status Values

**Task Status:**
- `pending`: Not yet started
- `in_progress`: Currently being worked on
- `completed`: Finished successfully
- `blocked`: Cannot proceed due to blocker
- `deferred`: Postponed to future sprint

**Sprint/Epic Status:**
- `planned`: Future work
- `active`: Currently in progress
- `completed`: Successfully finished
- `cancelled`: No longer pursuing

**Decision Status:**
- `proposed`: Under consideration
- `accepted`: Approved and active
- `rejected`: Not pursuing
- `deprecated`: No longer relevant
- `superseded`: Replaced by newer decision

#### Priority Values

```python
priority = "critical" | "high" | "medium" | "low"
```

#### Category Values (Tasks)

```python
category = "arch" | "feature" | "bugfix" | "test" | "docs" | "refactor" | "debt"
```

#### Effort Values (Tasks)

```python
effort = "trivial" | "small" | "medium" | "large" | "xl"
```

### Cross-Branch Coordination

When the same logical task exists on multiple branches:

**Strategy 1: Shared ID, Branch Metadata**
```python
# Same task on multiple branches
task_main = graph.add_node(
    "task:T-20251213-143052-a1b2",
    NodeType.TASK,
    "Implement BM25",
    properties={"status": "completed"},
    metadata={"branch": "main"}
)

task_dev = graph.add_node(
    "task:T-20251213-143052-a1b2",
    NodeType.TASK,
    "Implement BM25",
    properties={"status": "in_progress"},
    metadata={"branch": "dev"}
)
```

**Strategy 2: Branch-Scoped IDs**
```python
# Different IDs per branch
task_main = graph.add_node(
    "task:T-20251213-143052-a1b2:main",
    NodeType.TASK,
    "Implement BM25",
    properties={"status": "completed"},
    metadata={"branch": "main", "canonical_id": "T-20251213-143052-a1b2"}
)

task_dev = graph.add_node(
    "task:T-20251213-143052-a1b2:dev",
    NodeType.TASK,
    "Implement BM25",
    properties={"status": "in_progress"},
    metadata={"branch": "dev", "canonical_id": "T-20251213-143052-a1b2"}
)
```

**Recommended:** Use Strategy 2 for parallel development to avoid state conflicts.

---

## Query Patterns

### Get All Tasks in a Sprint

```python
def get_tasks_in_sprint(graph: ThoughtGraph, sprint_id: str) -> List[ThoughtNode]:
    """Get all tasks contained in a sprint."""
    tasks = []

    # Get all edges from sprint
    edges = graph.get_edges_from(sprint_id)

    # Filter CONTAINS edges pointing to tasks
    for edge in edges:
        if edge.edge_type == EdgeType.CONTAINS:
            node = graph.get_node(edge.target_id)
            if node and node.node_type == NodeType.TASK:
                tasks.append(node)

    return tasks
```

### Get Blocked Tasks

```python
def get_blocked_tasks(graph: ThoughtGraph) -> List[Tuple[ThoughtNode, List[ThoughtNode]]]:
    """Get all blocked tasks and their blockers."""
    blocked = []

    # Get all tasks
    tasks = graph.nodes_of_type(NodeType.TASK)

    for task in tasks:
        # Get all edges pointing TO this task
        incoming = graph.get_edges_to(task.id)

        # Find BLOCKS edges
        blockers = []
        for edge in incoming:
            if edge.edge_type == EdgeType.BLOCKS:
                blocker = graph.get_node(edge.source_id)
                if blocker:
                    blockers.append(blocker)

        if blockers:
            blocked.append((task, blockers))

    return blocked
```

### Get Task Dependency Chain

```python
def get_dependency_chain(graph: ThoughtGraph, task_id: str) -> List[str]:
    """Get all tasks that must complete before this task."""
    visited = set()
    chain = []

    def dfs(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)

        # Get all DEPENDS_ON edges from this task
        edges = graph.get_edges_from(node_id)
        for edge in edges:
            if edge.edge_type == EdgeType.DEPENDS_ON:
                dfs(edge.target_id)
                chain.append(edge.target_id)

    dfs(task_id)
    return chain
```

### Get Sprint Progress

```python
def get_sprint_progress(graph: ThoughtGraph, sprint_id: str) -> Dict[str, Any]:
    """Calculate sprint completion metrics."""
    tasks = get_tasks_in_sprint(graph, sprint_id)

    total = len(tasks)
    completed = sum(1 for t in tasks if t.properties.get("status") == "completed")
    in_progress = sum(1 for t in tasks if t.properties.get("status") == "in_progress")
    blocked = sum(1 for t in tasks if t.properties.get("status") == "blocked")

    return {
        "total_tasks": total,
        "completed": completed,
        "in_progress": in_progress,
        "blocked": blocked,
        "completion_rate": completed / total if total > 0 else 0.0,
        "remaining": total - completed
    }
```

### Find Critical Path

```python
def find_critical_path(graph: ThoughtGraph, sprint_id: str) -> List[str]:
    """Find longest dependency chain in sprint (critical path)."""
    tasks = get_tasks_in_sprint(graph, sprint_id)

    # Build dependency graph
    dependencies = {}
    for task in tasks:
        deps = []
        edges = graph.get_edges_from(task.id)
        for edge in edges:
            if edge.edge_type == EdgeType.DEPENDS_ON:
                deps.append(edge.target_id)
        dependencies[task.id] = deps

    # Find longest path using DFS
    max_path = []

    def dfs_path(node_id: str, path: List[str]):
        nonlocal max_path
        path = path + [node_id]

        if len(path) > len(max_path):
            max_path = path

        for dep in dependencies.get(node_id, []):
            dfs_path(dep, path)

    # Start from each task
    for task in tasks:
        dfs_path(task.id, [])

    return max_path
```

### Get Tasks by Priority

```python
def get_tasks_by_priority(
    graph: ThoughtGraph,
    priority: str,
    status: Optional[str] = None
) -> List[ThoughtNode]:
    """Get all tasks with given priority and optional status filter."""
    tasks = graph.nodes_of_type(NodeType.TASK)

    filtered = [
        t for t in tasks
        if t.properties.get("priority") == priority
        and (status is None or t.properties.get("status") == status)
    ]

    return filtered
```

---

## Merge Conflict Prevention

### Strategy 1: Timestamp-Based IDs

All IDs include timestamps, making collisions virtually impossible:

```python
# Agent A creates at 14:30:52
task_a = graph.add_node("task:T-20251213-143052-a1b2", ...)

# Agent B creates at 14:30:53 (even 1 second later)
task_b = graph.add_node("task:T-20251213-143053-c3d4", ...)

# No collision!
```

### Strategy 2: GraphWAL Persistence

Use Write-Ahead Log for atomic operations:

```python
from cortical.reasoning.graph_persistence import GraphWAL

# Initialize WAL
wal = GraphWAL(wal_dir=".got-wal/project")

# All operations are logged atomically
wal.log_add_node("task:T-20251213-143052-a1b2", NodeType.TASK, "Implement BM25")
wal.log_add_edge("sprint:S-042", "task:T-20251213-143052-a1b2", EdgeType.CONTAINS)

# Create snapshot for fast loading
wal.create_snapshot(graph, compress=True)
```

### Strategy 3: Branch-Scoped Graphs

Each branch maintains its own graph state:

```
.got-wal/
├── project-main/          # Main branch graph
│   ├── wal/
│   └── snapshots/
├── project-dev/           # Dev branch graph
│   ├── wal/
│   └── snapshots/
└── project-feature-x/     # Feature branch graph
    ├── wal/
    └── snapshots/
```

**Merge strategy:**
1. Load graphs from both branches
2. Identify common nodes (same canonical_id)
3. Merge properties (take newer timestamp)
4. Union all edges
5. Resolve conflicts based on rules

### Conflict Detection

```python
def detect_conflicts(graph_a: ThoughtGraph, graph_b: ThoughtGraph) -> List[Dict]:
    """Detect conflicting nodes between two graphs."""
    conflicts = []

    # Find nodes with same ID but different content
    common_ids = set(graph_a.nodes.keys()) & set(graph_b.nodes.keys())

    for node_id in common_ids:
        node_a = graph_a.get_node(node_id)
        node_b = graph_b.get_node(node_id)

        # Check if properties differ
        if node_a.properties != node_b.properties:
            conflicts.append({
                "node_id": node_id,
                "type": "property_conflict",
                "branch_a": node_a.properties,
                "branch_b": node_b.properties,
                "resolution": "merge" if can_auto_merge(node_a, node_b) else "manual"
            })

    return conflicts

def can_auto_merge(node_a: ThoughtNode, node_b: ThoughtNode) -> bool:
    """Check if nodes can be auto-merged."""
    # Auto-merge if one is strictly newer
    updated_a = node_a.metadata.get("updated_at", node_a.metadata.get("created_at"))
    updated_b = node_b.metadata.get("updated_at", node_b.metadata.get("created_at"))

    if updated_a and updated_b:
        return updated_a != updated_b

    # Cannot auto-merge
    return False
```

### Conflict Resolution

```python
def resolve_conflict(node_a: ThoughtNode, node_b: ThoughtNode) -> ThoughtNode:
    """Resolve conflict by taking newer version."""
    updated_a = node_a.metadata.get("updated_at", node_a.metadata.get("created_at"))
    updated_b = node_b.metadata.get("updated_at", node_b.metadata.get("created_at"))

    # Take newer version
    if updated_b > updated_a:
        return node_b
    else:
        return node_a
```

---

## Migration Path

### Phase 1: Parallel Systems

Run both file-based and GoT systems in parallel:

```python
from cortical.reasoning import ThoughtGraph, NodeType, EdgeType
from scripts.task_utils import TaskSession

# Old system (still works)
session = TaskSession()
task = session.create_task(title="Implement BM25", ...)
session.save()

# New system (running in parallel)
graph = ThoughtGraph()
task_node = graph.add_node(
    f"task:{task.id}",
    NodeType.TASK,
    task.title,
    properties={
        "title": task.title,
        "status": task.status,
        "priority": task.priority,
        "category": task.category,
    }
)
```

### Phase 2: Import Existing Tasks

```python
def import_tasks_to_graph(session_file: str, graph: ThoughtGraph) -> None:
    """Import tasks from JSON session file to GoT graph."""
    import json

    with open(session_file) as f:
        data = json.load(f)

    for task in data["tasks"]:
        graph.add_node(
            f"task:{task['id']}",
            NodeType.TASK,
            task["title"],
            properties={
                "title": task["title"],
                "status": task.get("status", "pending"),
                "priority": task.get("priority", "medium"),
                "category": task.get("category", "feature"),
                "description": task.get("description", ""),
                "effort": task.get("effort", "medium"),
            },
            metadata={
                "session_id": data["session_id"],
                "created_at": task["created_at"],
            }
        )

        # Add dependencies
        for dep_id in task.get("depends_on", []):
            graph.add_edge(
                f"task:{task['id']}",
                f"task:{dep_id}",
                EdgeType.DEPENDS_ON
            )
```

### Phase 3: Import Sprints

```python
def import_sprint_to_graph(sprint_file: str, graph: ThoughtGraph) -> None:
    """Import sprint from CURRENT_SPRINT.md to GoT graph."""
    # Parse sprint file (simplified)
    sprint_id = extract_sprint_id(sprint_file)
    name = extract_sprint_name(sprint_file)
    goals = extract_goals(sprint_file)

    # Create sprint node
    sprint_node = graph.add_node(
        f"sprint:{sprint_id}",
        NodeType.GOAL,
        name,
        properties={
            "sprint_id": sprint_id,
            "name": name,
            "status": "active",
            "start_date": extract_start_date(sprint_file),
            "end_date": extract_end_date(sprint_file),
            "goals": goals,
        }
    )

    # Link tasks to sprint
    task_ids = extract_task_ids(sprint_file)
    for task_id in task_ids:
        graph.add_edge(
            f"sprint:{sprint_id}",
            f"task:{task_id}",
            EdgeType.CONTAINS
        )
```

### Phase 4: Deprecate File-Based System

Once GoT is stable:
1. Stop creating new files in `tasks/`
2. Archive existing task files
3. Use GoT as single source of truth
4. Update all scripts to query GoT

---

## Example Usage

### Complete Project Setup

```python
from cortical.reasoning import ThoughtGraph, NodeType, EdgeType
from cortical.reasoning.graph_persistence import GraphWAL
from datetime import datetime

# Initialize graph and persistence
graph = ThoughtGraph()
wal = GraphWAL(wal_dir=".got-wal/project")

# Create epic
epic = graph.add_node(
    "epic:search-overhaul",
    NodeType.GOAL,
    "Search System Overhaul",
    properties={
        "epic_id": "search-overhaul",
        "name": "Search System Overhaul",
        "phase": "active",
        "description": "Replace TF-IDF with modern ranking algorithms",
        "start_date": "2025-12-01",
        "target_date": "2026-03-31",
        "priority": "high",
    }
)

# Create sprint
sprint = graph.add_node(
    "sprint:S-042",
    NodeType.GOAL,
    "Sprint 42: Search Relevance",
    properties={
        "sprint_id": "S-042",
        "name": "Search Relevance Improvements",
        "status": "active",
        "start_date": "2025-12-13",
        "end_date": "2025-12-27",
        "goals": ["Implement BM25", "Add semantic search prototype"],
    }
)

# Link epic → sprint
graph.add_edge("epic:search-overhaul", "sprint:S-042", EdgeType.CONTAINS)

# Create architecture decision
decision = graph.add_node(
    "decision:ADR-001",
    NodeType.DECISION,
    "Use BM25 for search ranking",
    properties={
        "title": "Use BM25 instead of TF-IDF",
        "status": "accepted",
        "date": "2025-12-10",
        "context": "TF-IDF has document length bias issues",
        "decision": "Implement BM25 with k1=1.2, b=0.75",
        "consequences": "Better relevance, requires reindexing",
    }
)

# Create task implementing the decision
task1 = graph.add_node(
    "task:T-20251213-143052-a1b2",
    NodeType.TASK,
    "Implement BM25 scoring algorithm",
    properties={
        "title": "Implement BM25 scoring algorithm",
        "status": "in_progress",
        "priority": "high",
        "category": "feature",
        "effort": "medium",
        "description": "Replace TF-IDF with BM25 in analysis.py",
    },
    metadata={
        "session_id": "a1b2",
        "branch": "feature/bm25-scoring",
        "files": ["cortical/analysis.py"],
    }
)

# Create dependent test task
task2 = graph.add_node(
    "task:T-20251213-150000-a1b2",
    NodeType.TASK,
    "Add tests for BM25 scoring",
    properties={
        "title": "Add tests for BM25 scoring",
        "status": "pending",
        "priority": "high",
        "category": "test",
        "effort": "small",
    },
    metadata={
        "session_id": "a1b2",
        "branch": "feature/bm25-scoring",
        "files": ["tests/test_analysis.py"],
    }
)

# Build relationships
graph.add_edge("sprint:S-042", "task:T-20251213-143052-a1b2", EdgeType.CONTAINS)
graph.add_edge("sprint:S-042", "task:T-20251213-150000-a1b2", EdgeType.CONTAINS)
graph.add_edge("task:T-20251213-143052-a1b2", "decision:ADR-001", EdgeType.IMPLEMENTS)
graph.add_edge("task:T-20251213-150000-a1b2", "task:T-20251213-143052-a1b2", EdgeType.DEPENDS_ON)
graph.add_edge("task:T-20251213-143052-a1b2", "sprint:S-042", EdgeType.REFINES, weight=0.5)

# Persist to WAL
wal.log_add_node(epic.id, epic.node_type, epic.content)
wal.log_add_node(sprint.id, sprint.node_type, sprint.content)
wal.log_add_node(decision.id, decision.node_type, decision.content)
wal.log_add_node(task1.id, task1.node_type, task1.content)
wal.log_add_node(task2.id, task2.node_type, task2.content)

for edge in graph.edges:
    wal.log_add_edge(edge.source_id, edge.target_id, edge.edge_type, edge.weight)

# Create snapshot for fast loading
snapshot_id = wal.create_snapshot(graph, compress=True)
print(f"Snapshot created: {snapshot_id}")
```

### Query Project State

```python
# Get sprint progress
progress = get_sprint_progress(graph, "sprint:S-042")
print(f"Sprint S-042: {progress['completed']}/{progress['total_tasks']} complete")

# Get blocked tasks
blocked = get_blocked_tasks(graph)
for task, blockers in blocked:
    print(f"Task {task.id} blocked by:")
    for blocker in blockers:
        print(f"  - {blocker.id}: {blocker.content}")

# Get critical path
critical_path = find_critical_path(graph, "sprint:S-042")
print(f"Critical path: {' → '.join(critical_path)}")

# Get high-priority pending tasks
high_pri = get_tasks_by_priority(graph, "high", status="pending")
print(f"High-priority pending tasks: {len(high_pri)}")
```

### Visualize Project

```python
# ASCII tree view
print(graph.to_ascii(root_id="epic:search-overhaul"))

# Mermaid diagram (for markdown docs)
with open("docs/project-graph.md", "w") as f:
    f.write("# Project Graph\n\n")
    f.write("```mermaid\n")
    f.write(graph.to_mermaid())
    f.write("\n```\n")

# DOT format (for Graphviz)
with open("project.dot", "w") as f:
    f.write(graph.to_dot())
```

---

## Best Practices

### 1. Use Descriptive Node IDs

**Good:**
```python
"task:T-20251213-143052-a1b2"  # Clear timestamp and session
"sprint:S-042"                 # Sequential sprint number
"epic:search-overhaul"         # Descriptive name
```

**Bad:**
```python
"t1"                           # Not descriptive
"sprint"                       # Not unique
"epic123"                      # No context
```

### 2. Capture Rich Metadata

Always include:
- `created_at` timestamp
- `session_id` for traceability
- `branch` for cross-branch coordination
- `updated_at` when modifying

### 3. Use Appropriate Edge Types

Don't overuse generic edges like `RELATED_TO`. Be specific:
- Use `DEPENDS_ON` for hard dependencies
- Use `PRECEDES` for soft ordering
- Use `BLOCKS` for active blockers
- Use `REFINES` for goal decomposition

### 4. Persist Frequently

Use GraphWAL to persist after each logical change:

```python
# After adding nodes/edges
wal.log_add_node(...)
wal.log_add_edge(...)

# Periodic snapshots (e.g., end of session)
wal.create_snapshot(graph, compress=True)
```

### 5. Query Efficiently

Use indices and avoid full graph scans:

```python
# Good: Use graph methods
tasks = graph.nodes_of_type(NodeType.TASK)

# Bad: Manual iteration
tasks = [n for n in graph.nodes.values() if n.node_type == NodeType.TASK]
```

### 6. Validate Graph Integrity

Periodically check for:
- Orphaned nodes (no edges)
- Circular dependencies
- Broken references

```python
# Find orphans
orphans = graph.find_orphans()

# Find cycles
cycles = graph.find_cycles()

# Validate all edge endpoints exist
for edge in graph.edges:
    assert edge.source_id in graph.nodes
    assert edge.target_id in graph.nodes
```

---

## Summary

This schema enables:
- **Unified project management** through graph representation
- **Merge-friendly collaboration** via timestamp-based IDs and WAL persistence
- **Rich relationships** between tasks, sprints, epics, decisions, and insights
- **Powerful queries** for project analytics and visualization
- **Incremental migration** from file-based to graph-based system

The Graph of Thought framework provides the foundation for sophisticated project management while maintaining simplicity and git-friendliness.
