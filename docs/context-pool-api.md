# ContextPool API Reference

## Overview

The ContextPool API enables multi-agent coordination through a shared, conflict-aware repository for agent discoveries.

## Quick Start

```python
from cortical.reasoning import ContextPool, ConflictResolutionStrategy

# Create pool
pool = ContextPool(
    ttl_seconds=3600,  # 1-hour expiration
    conflict_strategy=ConflictResolutionStrategy.MANUAL
)

# Agent publishes finding
finding = pool.publish(
    topic="bug_analysis",
    content="Found NPE in login.py:42",
    source_agent="agent_a",
    confidence=0.95,
    metadata={"task_id": "T-001"}
)

# Other agents query findings
bugs = pool.query("bug_analysis")
for bug in bugs:
    print(f"{bug.source_agent}: {bug.content}")
```

## Core Classes

### ContextFinding

Immutable dataclass representing a single finding.

**Attributes:**
- `topic: str` - Category/subject of the finding
- `content: str` - The actual discovery
- `source_agent: str` - Agent ID that published this
- `timestamp: float` - Unix timestamp when published
- `confidence: float` - Confidence score 0.0-1.0
- `finding_id: str` - Unique identifier
- `metadata: Dict[str, Any]` - Optional metadata (task_id, node_id, etc.)

**Methods:**
- `to_dict() -> dict` - Serialize to dictionary
- `from_dict(data: dict) -> ContextFinding` - Deserialize from dictionary
- `conflicts_with(other: ContextFinding) -> bool` - Check if conflicts with another finding

**Example:**
```python
finding = ContextFinding(
    topic="code_location",
    content="Auth code in validators.py",
    source_agent="agent_a",
    timestamp=time.time(),
    confidence=0.9,
    finding_id="abc123",
    metadata={"task_id": "T-001"}
)

# Check for conflicts
other = ContextFinding(...)
if finding.conflicts_with(other):
    print("Conflict detected!")
```

### ContextPool

Shared context pool for multi-agent coordination.

**Constructor:**
```python
pool = ContextPool(
    ttl_seconds=None,  # Optional: time-to-live in seconds
    conflict_strategy=ConflictResolutionStrategy.MANUAL,  # Conflict handling
    storage_dir=None   # Optional: directory for persistence
)
```

**Parameters:**
- `ttl_seconds` - Findings expire after this duration (None = no expiration)
- `conflict_strategy` - How to handle conflicts (MANUAL, LAST_WRITE_WINS, HIGHEST_CONFIDENCE, MERGE)
- `storage_dir` - Directory for persistence (None = memory-only)

**Methods:**

#### publish()
Publish a new finding to the pool.

```python
finding = pool.publish(
    topic: str,
    content: str,
    source_agent: str,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextFinding
```

**Returns:** The created ContextFinding

**Raises:** `ValueError` if confidence not in [0.0, 1.0]

#### query()
Query findings by topic.

```python
findings = pool.query(topic: str) -> List[ContextFinding]
```

**Returns:** List of findings for this topic (may be empty)

#### query_all()
Query all findings across all topics.

```python
all_findings = pool.query_all() -> List[ContextFinding]
```

**Returns:** List of all findings

#### subscribe()
Subscribe to findings on a topic.

```python
pool.subscribe(
    topic: str,
    callback: Callable[[ContextFinding], None]
) -> None
```

**Example:**
```python
def on_bug(finding: ContextFinding):
    print(f"New bug: {finding.content}")

pool.subscribe("bugs", on_bug)
```

#### get_conflicts()
Get all detected conflicts.

```python
conflicts = pool.get_conflicts() -> List[Tuple[ContextFinding, ContextFinding]]
```

**Returns:** List of (finding1, finding2) conflict pairs

#### get_topics()
Get all topics with findings.

```python
topics = pool.get_topics() -> List[str]
```

#### count()
Count findings.

```python
total = pool.count()                # All findings
count = pool.count(topic="bugs")   # Specific topic
```

**Returns:** Number of findings

#### clear()
Clear all findings and conflicts.

```python
pool.clear()
```

#### save()
Save pool state to JSON.

```python
pool.save(filepath: Optional[Path] = None)
```

If `filepath` is None, uses `storage_dir/context_pool.json`

#### load()
Load pool state from JSON.

```python
pool.load(filepath: Optional[Path] = None)
```

If `filepath` is None, uses `storage_dir/context_pool.json`

### ConflictResolutionStrategy

Enum defining conflict resolution strategies.

**Values:**
- `MANUAL` - Director must resolve (both findings kept)
- `LAST_WRITE_WINS` - Most recent finding wins
- `HIGHEST_CONFIDENCE` - Highest confidence finding wins
- `MERGE` - Keep all conflicting findings

**Example:**
```python
from cortical.reasoning import ConflictResolutionStrategy

pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
```

## Conflict Detection

Two findings conflict if they:
1. Share the same topic
2. Have different content (case-insensitive comparison)
3. Are from different agents

**Example:**
```python
pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

# These conflict (same topic, different content, different agents)
pool.publish("location", "File in dir_a", "agent_a")
pool.publish("location", "File in dir_b", "agent_b")

conflicts = pool.get_conflicts()
print(f"Detected {len(conflicts)} conflicts")
```

## TTL Expiration

Findings can automatically expire after a duration:

```python
# 2-hour TTL
pool = ContextPool(ttl_seconds=7200)

pool.publish("temp", "content", "agent")
print(pool.count())  # 1

# Wait 2 hours...
time.sleep(7200)

print(pool.count())  # 0 (expired)
```

Expired findings are removed when:
- `query()` or `query_all()` is called
- `count()` is called
- `get_topics()` is called

## Subscriptions

Real-time notifications when findings are published:

```python
def on_blocker(finding: ContextFinding):
    task_id = finding.metadata["task_id"]
    print(f"Task {task_id} blocked: {finding.content}")

pool.subscribe("blockers", on_blocker)

# When published, callback is invoked
pool.publish(
    topic="blockers",
    content="Waiting for API approval",
    source_agent="agent_a",
    metadata={"task_id": "T-001"}
)
# Output: Task T-001 blocked: Waiting for API approval
```

**Notes:**
- Multiple subscribers can subscribe to the same topic
- Subscribers are only notified for their specific topic
- If a finding is rejected due to conflict resolution, subscribers are NOT notified

## Persistence

Save and load pool state for crash recovery:

```python
from pathlib import Path

# Create pool with storage
pool = ContextPool(storage_dir=Path(".got/context/"))

# Publish findings
pool.publish("topic1", "content1", "agent_a")
pool.publish("topic2", "content2", "agent_b")

# Save to disk
pool.save()  # Creates .got/context/context_pool.json

# Later: restore state
new_pool = ContextPool(storage_dir=Path(".got/context/"))
new_pool.load()

print(f"Restored {new_pool.count()} findings")
```

**Saved data includes:**
- All findings with full metadata
- Conflict records
- TTL and conflict strategy settings

## Integration with GoT

Link findings to GoT tasks/nodes via metadata:

```python
from cortical.reasoning import ContextPool, ThoughtGraph, NodeType

pool = ContextPool()
graph = ThoughtGraph()

# Create task in GoT
task_node = graph.add_node("T-001", NodeType.TASK, "Fix authentication")

# Publish finding linked to task
pool.publish(
    topic="task_progress",
    content="Auth fix 60% complete",
    source_agent="agent_a",
    metadata={
        "task_id": "T-001",
        "node_id": task_node,
        "progress": 0.6
    }
)

# Query and update GoT
for finding in pool.query("task_progress"):
    node_id = finding.metadata["node_id"]
    progress = finding.metadata["progress"]
    # Update node or create edges
```

## Common Patterns

### Pattern: Bug Discovery Pipeline

```python
# QA discovers bug
pool.publish(
    topic="bugs",
    content="Auth fails with special chars",
    source_agent="qa_agent",
    metadata={"severity": "high"}
)

# Analyzer locates code
pool.publish(
    topic="code_location",
    content="Bug in validators.py:42",
    source_agent="analyzer",
    metadata={"file": "validators.py", "line": 42}
)

# Implementer retrieves context
bugs = pool.query("bugs")
locations = pool.query("code_location")
# Fix the bug...

# Publish fix status
pool.publish(
    topic="fixes",
    content="Added null check",
    source_agent="implementer",
    metadata={"commit": "abc123"}
)
```

### Pattern: Director Monitoring

```python
# Director subscribes to all critical topics
def on_blocker(finding):
    escalate_to_human(finding)

def on_conflict(finding):
    spawn_investigation_task(finding)

pool.subscribe("blockers", on_blocker)
pool.subscribe("conflicts", on_conflict)

# Agents publish, director receives notifications
```

### Pattern: Batch Coordination

```python
# Create batch-scoped pool
batch_pool = ContextPool(
    ttl_seconds=7200,  # 2-hour batch
    storage_dir=Path(f".got/context/batch-{batch_id}/")
)

# Spawn parallel agents
coordinator.spawn_agent("agent_1", task="Feature A")
coordinator.spawn_agent("agent_2", task="Feature B")

# Agents publish progress
batch_pool.publish("progress", "Feature A 50%", "agent_1")
batch_pool.publish("progress", "Feature B 30%", "agent_2")

# Director monitors
all_progress = batch_pool.query("progress")
print(f"Agents working: {len(set(f.source_agent for f in all_progress))}")
```

## Performance Characteristics

- **Publish:** O(n) where n = existing findings on same topic (conflict checking)
- **Query:** O(n + k) where n = all findings (TTL pruning), k = topic findings
- **Subscribe:** O(1) to add subscriber
- **Notification:** O(m) where m = number of subscribers for topic
- **Save/Load:** O(n) where n = total findings

## Thread Safety

⚠️ **Warning:** ContextPool is NOT thread-safe. For multi-threaded usage, wrap operations in locks:

```python
import threading

pool = ContextPool()
lock = threading.Lock()

def thread_safe_publish(topic, content, agent):
    with lock:
        return pool.publish(topic, content, agent)
```

## See Also

- **Implementation:** `cortical/reasoning/context_pool.py`
- **Examples:** `examples/context_pool_demo.py`
- **Tests:** `tests/unit/test_context_pool.py`
- **Integration Guide:** `docs/context-pool-integration.md`
- **GoT Framework:** `docs/graph-of-thought.md`
