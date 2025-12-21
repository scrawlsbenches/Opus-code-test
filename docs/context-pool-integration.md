# Context Pool Integration with Graph of Thought

## Overview

The **ContextPool** enables multi-agent coordination by providing a shared, conflict-aware repository for agent discoveries. This document explains how to integrate ContextPool with the Graph of Thought (GoT) reasoning framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐         ┌───────────────┐         ┌──────────┐       │
│  │ Agent A  │────────>│ Context Pool  │<────────│ Agent B  │       │
│  └──────────┘         └───────────────┘         └──────────┘       │
│       │                      │                        │              │
│       │                      │                        │              │
│       v                      v                        v              │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              Graph of Thought (ThoughtGraph)              │      │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐            │      │
│  │  │ Task 001 │──>│ Task 002 │──>│ Task 003 │            │      │
│  │  └──────────┘   └──────────┘   └──────────┘            │      │
│  │       │              │              │                    │      │
│  │       └──────────────┴──────────────┘                    │      │
│  │                 Context Findings                         │      │
│  │         (via metadata linkage from pool)                 │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### ContextFinding

A **ContextFinding** represents a single discovery made by an agent:

```python
finding = ContextFinding(
    topic="bug_analysis",              # Category of finding
    content="Found NPE in login.py",   # The discovery
    source_agent="agent_a",             # Which agent found it
    timestamp=1234567890.0,             # When discovered
    confidence=0.95,                     # How certain (0.0-1.0)
    finding_id="abc123",                # Unique identifier
    metadata={                          # Links to GoT
        "task_id": "T-001",
        "node_id": "N-123",
        "priority": "high"
    }
)
```

### ContextPool

A **ContextPool** manages findings with conflict detection:

```python
pool = ContextPool(
    ttl_seconds=3600,  # Findings expire after 1 hour
    conflict_strategy=ConflictResolutionStrategy.MANUAL,
    storage_dir=Path(".got/context/")  # Persist to disk
)
```

## Integration Patterns

### Pattern 1: Task Progress Tracking

Agents publish progress updates linked to GoT task nodes:

```python
from cortical.reasoning import ContextPool, ThoughtGraph, NodeType

# Setup
pool = ContextPool()
graph = ThoughtGraph()

# Agent A starts work on a task
task_node = graph.add_node("T-001", NodeType.TASK, "Implement auth")

# Agent A publishes progress
pool.publish(
    topic="task_progress",
    content="Authentication logic 60% complete",
    source_agent="agent_a",
    metadata={
        "task_id": "T-001",
        "node_id": task_node,
        "progress": 0.6,
        "blockers": []
    }
)

# Agent B (Director) queries progress
for finding in pool.query("task_progress"):
    task_id = finding.metadata["task_id"]
    progress = finding.metadata["progress"]
    print(f"Task {task_id}: {progress * 100}% complete")
```

### Pattern 2: Dependency Discovery

Agents discover dependencies and publish them for GoT integration:

```python
# Agent discovers a dependency
pool.publish(
    topic="dependencies",
    content="Task T-002 requires T-001 completion",
    source_agent="analyzer_agent",
    metadata={
        "source_task": "T-002",
        "depends_on": "T-001",
        "edge_type": "DEPENDS_ON"
    }
)

# Director queries dependencies and builds GoT edges
for finding in pool.query("dependencies"):
    source = finding.metadata["source_task"]
    target = finding.metadata["depends_on"]
    edge_type = finding.metadata["edge_type"]

    # Add edge to thought graph
    graph.add_edge(source, target, edge_type)
```

### Pattern 3: Conflict Resolution by Director

When agents disagree, the director resolves conflicts:

```python
# Use MANUAL conflict strategy
pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

# Agent A's finding
pool.publish(
    topic="code_location",
    content="Auth code in validators.py",
    source_agent="agent_a",
    confidence=0.8
)

# Agent B's conflicting finding
pool.publish(
    topic="code_location",
    content="Auth code in utils/validation.py",
    source_agent="agent_b",
    confidence=0.9
)

# Director detects conflict
conflicts = pool.get_conflicts()
for f1, f2 in conflicts:
    # Create a QUESTION node in GoT
    question_id = graph.add_node(
        f"Q-conflict-{f1.finding_id}",
        NodeType.QUESTION,
        f"Which is correct: {f1.content} or {f2.content}?"
    )

    # Assign to human or spawn investigation task
    print(f"Conflict detected: {f1.source_agent} vs {f2.source_agent}")
    print(f"Created question node: {question_id}")
```

### Pattern 4: Bug Discovery Pipeline

Agents discover bugs and link them through the context pool:

```python
# QA Agent discovers bug
pool.publish(
    topic="bug_analysis",
    content="Auth fails with special chars in username",
    source_agent="qa_agent",
    confidence=0.95,
    metadata={
        "severity": "high",
        "reproducible": True
    }
)

# Code analyzer locates the bug
pool.publish(
    topic="code_location",
    content="Bug is in validators.py:validate_username() line 42",
    source_agent="analyzer_agent",
    metadata={
        "file": "validators.py",
        "function": "validate_username",
        "line": 42
    }
)

# Implementer retrieves context
bugs = pool.query("bug_analysis")
locations = pool.query("code_location")

print(f"Found {len(bugs)} bugs")
print(f"Located {len(locations)} code sites")

# Implementer fixes and publishes status
pool.publish(
    topic="fix_status",
    content="Added regex validation for special chars",
    source_agent="implementer_agent",
    metadata={
        "commit": "abc123",
        "test_status": "passing"
    }
)
```

### Pattern 5: Batch Coordination

Director uses context pool to coordinate parallel batches:

```python
from cortical.reasoning import ParallelCoordinator, ContextPool

# Create pool scoped to batch
batch_pool = ContextPool(
    ttl_seconds=7200,  # 2-hour batch window
    storage_dir=Path(".got/context/batch-001/")
)

# Subscribe director to batch events
def on_agent_update(finding: ContextFinding):
    agent_id = finding.source_agent
    progress = finding.metadata.get("progress", 0)
    print(f"[Director] {agent_id} progress: {progress}%")

batch_pool.subscribe("agent_status", on_agent_update)

# Spawn parallel agents
coordinator = ParallelCoordinator()
coordinator.spawn_agent("agent_1", task="Implement feature A")
coordinator.spawn_agent("agent_2", task="Implement feature B")

# Agents publish status via pool
batch_pool.publish(
    topic="agent_status",
    content="Feature A implementation started",
    source_agent="agent_1",
    metadata={"progress": 10, "task_id": "T-001"}
)

# Director queries all agent statuses
all_findings = batch_pool.query_all()
print(f"Total findings from {len(set(f.source_agent for f in all_findings))} agents")
```

## Conflict Resolution Strategies

### MANUAL (Default)

Director must resolve conflicts:

```python
pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

# Both findings are kept
pool.publish("topic", "Finding A", "agent_a", confidence=0.8)
pool.publish("topic", "Finding B", "agent_b", confidence=0.9)

# Director handles conflict
for f1, f2 in pool.get_conflicts():
    # Spawn investigation or ask human
    decision = ask_human(f"Which is correct: {f1.content} or {f2.content}?")
```

### LAST_WRITE_WINS

Most recent finding wins:

```python
pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.LAST_WRITE_WINS)

pool.publish("perf", "Query takes 500ms", "agent_a")
pool.publish("perf", "Query takes 450ms", "agent_b")  # This wins

findings = pool.query("perf")
# Only "Query takes 450ms" is kept
```

### HIGHEST_CONFIDENCE

Higher confidence finding wins:

```python
pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)

pool.publish("location", "File in dir_a", "agent_a", confidence=0.7)
pool.publish("location", "File in dir_b", "agent_b", confidence=0.9)  # This wins

findings = pool.query("location")
# Only "File in dir_b" is kept (confidence 0.9)
```

## Persistence and Recovery

### Save/Load Pool State

```python
# Create pool with storage
pool = ContextPool(storage_dir=Path(".got/context/"))

# Agents publish findings
pool.publish("topic", "content", "agent_a")

# Save to disk
pool.save()  # Creates .got/context/context_pool.json

# Later: restore state
new_pool = ContextPool(storage_dir=Path(".got/context/"))
new_pool.load()

print(f"Restored {new_pool.count()} findings")
```

### Integration with GraphWAL

Combine with GraphWAL for crash recovery:

```python
from cortical.reasoning import GraphWAL, ContextPool

# Setup WAL and context pool
wal = GraphWAL(wal_dir=".got/wal/")
pool = ContextPool(storage_dir=Path(".got/context/"))

# Agent publishes finding
finding = pool.publish("discovery", "Found pattern X", "agent_a")

# Log to WAL for recovery
wal.log_custom("context_finding", {
    "finding_id": finding.finding_id,
    "topic": finding.topic,
    "content": finding.content
})

# Save pool state
pool.save()
```

## Best Practices

### 1. Use Metadata for GoT Linkage

Always include task/node references in metadata:

```python
pool.publish(
    topic="code_change",
    content="Modified auth.py",
    source_agent="agent_a",
    metadata={
        "task_id": "T-001",       # Link to task
        "node_id": "N-456",       # Link to GoT node
        "file_path": "auth.py",   # Context
        "commit": "abc123"        # Traceability
    }
)
```

### 2. Use Subscriptions for Real-Time Coordination

Director subscribes to critical topics:

```python
def on_blocker(finding: ContextFinding):
    task_id = finding.metadata["task_id"]
    blocker = finding.content
    print(f"[ALERT] Task {task_id} blocked: {blocker}")
    # Spawn investigation or escalate

pool.subscribe("blockers", on_blocker)
```

### 3. Scope Pools to Work Units

Create separate pools for batches/plans:

```python
# Batch-scoped pool
batch_pool = ContextPool(
    ttl_seconds=3600,
    storage_dir=Path(f".got/context/batch-{batch_id}/")
)

# Plan-scoped pool
plan_pool = ContextPool(
    ttl_seconds=86400,  # 24 hours
    storage_dir=Path(f".got/context/plan-{plan_id}/")
)
```

### 4. Use TTL for Batch/Plan Scoping

Set TTL to match batch/plan duration:

```python
# 2-hour batch
batch_pool = ContextPool(ttl_seconds=7200)

# 1-day plan
plan_pool = ContextPool(ttl_seconds=86400)

# Session-scoped (no expiration)
session_pool = ContextPool(ttl_seconds=None)
```

## Example: Complete Director Workflow

```python
from cortical.reasoning import (
    ContextPool,
    ThoughtGraph,
    ParallelCoordinator,
    NodeType,
    EdgeType
)

# Initialize
pool = ContextPool(
    ttl_seconds=7200,
    conflict_strategy=ConflictResolutionStrategy.MANUAL,
    storage_dir=Path(".got/context/batch-001/")
)
graph = ThoughtGraph()
coordinator = ParallelCoordinator()

# Subscribe to critical events
pool.subscribe("blockers", lambda f: handle_blocker(f))
pool.subscribe("conflicts", lambda f: handle_conflict(f))

# Spawn agents
coordinator.spawn_agent("explorer", "Analyze codebase")
coordinator.spawn_agent("implementer", "Fix bugs")

# Agents publish findings
pool.publish(
    topic="bugs",
    content="NPE in login.py:42",
    source_agent="explorer",
    metadata={"severity": "high", "file": "login.py"}
)

pool.publish(
    topic="fix_status",
    content="NPE fixed with null check",
    source_agent="implementer",
    metadata={"commit": "def456"}
)

# Director queries and builds GoT
bugs = pool.query("bugs")
fixes = pool.query("fix_status")

for bug in bugs:
    bug_node = graph.add_node(
        f"bug-{bug.finding_id}",
        NodeType.OBSERVATION,
        bug.content
    )

for fix in fixes:
    fix_node = graph.add_node(
        f"fix-{fix.finding_id}",
        NodeType.ACTION,
        fix.content
    )
    # Link fix to bug
    graph.add_edge(fix_node, bug_node, EdgeType.RESOLVES)

# Save state
pool.save()
graph.save(".got/graph/batch-001.json")

print(f"Processed {pool.count()} findings")
print(f"Built graph with {graph.node_count()} nodes")
```

## See Also

- `cortical/reasoning/context_pool.py` - Implementation
- `examples/context_pool_demo.py` - Usage examples
- `docs/graph-of-thought.md` - GoT framework
- `docs/director-orchestration.md` - Director patterns
