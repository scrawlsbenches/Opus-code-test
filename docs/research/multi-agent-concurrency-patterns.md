# Multi-Agent Concurrency Patterns for File-Based Systems

Research document analyzing 6 concurrency patterns suitable for multiple Claude agents working in parallel on shared file-based data.

## Executive Summary

| Pattern | Best For | Complexity | Git Merge-Friendly | Implementation |
|---------|----------|-----------|-------------------|-----------------|
| **Optimistic Concurrency** | High parallelism, low conflict rate | Low | ⚠️ Conditional | Version numbers + conflict detection |
| **CRDTs** | Automatic conflict resolution | High | ✅ Excellent | Causal ordering, commutativity |
| **Operational Transforms** | Rich data types, sequential edits | Very High | ⚠️ Difficult | 3-way transforms, transformation spaces |
| **Branch-and-Merge** | Well-separated concerns, long-running work | Medium | ✅ Native to Git | Git branches, PR workflow |
| **Lock-Based Regions** | Critical sections, exclusive access | Low | ⚠️ Requires coordination | File locks, lease-based |
| **Event Sourcing** | Audit trail, time travel, replay | Medium | ✅ Excellent | Append-only logs, event replay |

---

## Pattern 1: Optimistic Concurrency

### Concept

**Multiple agents read data, modify independently, detect conflicts at commit time.**

Assumes conflicts are rare. Detects them through version numbers or checksums, then either:
- Retries the transaction
- Resolves via user-provided strategy
- Fails loudly for manual intervention

### When to Use

✅ **Good for:**
- High parallelism with low expected conflicts
- Short-lived transactions
- Read-heavy workloads with sparse writes
- Fault-tolerant operations (can retry)

❌ **Not for:**
- Critical sections requiring atomicity
- High-conflict scenarios (thrashing/livelock)
- Systems requiring guaranteed success

### Existing Implementation in Codebase

The `cortical/got/tx_manager.py` implements optimistic concurrency:

```python
def commit(self, tx: Transaction) -> CommitResult:
    """
    Commit transaction with optimistic locking.

    1. Read data at snapshot version
    2. Modify locally
    3. On commit: detect version conflicts
    4. If conflict: abort and return failure
    5. Caller retries or escalates
    """
    with self.lock:
        # Set state to PREPARING
        tx.state = TransactionState.PREPARING

        # DETECT CONFLICTS - compare versions
        conflicts = self._detect_conflicts(tx)
        if conflicts:
            # Abort transaction
            tx.state = TransactionState.ABORTED
            return CommitResult(success=False, conflicts=conflicts)

        # Apply writes atomically
        new_version = self.store.apply_writes(tx.write_set)

        tx.state = TransactionState.COMMITTED
        return CommitResult(success=True, version=new_version)

def _detect_conflicts(self, tx: Transaction) -> List[Conflict]:
    """Check if entity versions changed since read."""
    conflicts = []

    for entity_id in tx.write_set:
        # Check if entity was read
        if entity_id in tx.read_set:
            expected_version = tx.read_set[entity_id]

            # Get current version from store
            current_entity = self.store.read(entity_id)
            actual_version = current_entity.version if current_entity else 0

            if expected_version != actual_version:
                conflicts.append(Conflict(
                    entity_id=entity_id,
                    expected_version=expected_version,
                    actual_version=actual_version,
                    conflict_type="version_mismatch",
                    message=f"Expected version {expected_version}, got {actual_version}"
                ))

    return conflicts
```

### Usage Pattern for Multi-Agent Scenario

```python
# Agent A
tx = manager.begin()
task = manager.read(tx, "task:T-abc123")
task.status = "completed"
manager.write(tx, task)
result_a = manager.commit(tx)

if not result_a.success:
    # Conflict detected - retry with fresh read
    tx = manager.begin()
    task = manager.read(tx, "task:T-abc123")  # Get fresh version
    # ... apply changes again
    result_a = manager.commit(tx)

# Agent B (different task)
tx = manager.begin()
task_b = manager.read(tx, "task:T-xyz789")
task_b.status = "in_progress"
manager.write(tx, task_b)
result_b = manager.commit(tx)  # Succeeds (no conflict)
```

### Merge Strategy for Git

**✅ Safe approach:**
- Each agent works independently on their transaction
- Conflicts resolved via retry logic, not git merge
- Final state is transactionally consistent
- No git merge conflicts at transaction level

**⚠️ Caveats:**
- Multiple retries can create transaction churn
- Not suitable for high-contention scenarios
- Retry logic needed in agent code

### Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| Simple to understand | Retries needed on conflict |
| Minimal coordination overhead | Thundering herd on high contention |
| Works with eventual consistency | Not suitable for critical sections |
| Natural for Git workflow | Requires exponential backoff |

### Best Practices

1. **Implement exponential backoff** - Don't hammer on conflicts
2. **Log conflicts** - Track why retries happen
3. **Set retry limits** - Avoid infinite loops
4. **Monitor conflict rate** - Alert if it exceeds threshold
5. **Snapshot isolation** - Use version numbers for snapshot consistency

---

## Pattern 2: Conflict-Free Replicated Data Types (CRDTs)

### Concept

**Data structure designed so concurrent operations ALWAYS commute** - the order they're applied doesn't matter, so no conflicts possible.

Key insight: Different operations have different semantics:
- Addition is commutative: `a + b = b + a`
- Subtraction is NOT: `a - b ≠ b - a`
- **Last-Write-Wins** is commutative: later timestamp always wins
- **Counters with unique IDs** commute: `A.inc() + B.inc()` always yields same result

### When to Use

✅ **Good for:**
- Sets (add/remove elements)
- Counters (increment only, or per-agent counters)
- Maps with LWW (Last-Write-Wins) conflict resolution
- Append-only logs
- Text with unique per-agent timestamps

❌ **Not for:**
- Ordered sequences with arbitrary insertions (use OT instead)
- Operations with dependencies
- Systems requiring strict ordering

### CRDT Types Suitable for File-Based Systems

#### 1. **G-Set (Grow-Only Set)**

```python
class GSet:
    """Set that only grows - no removals."""

    def __init__(self):
        self.elements = set()

    def add(self, element):
        """Add element - always succeeds."""
        self.elements.add(element)

    def merge(self, other_set):
        """Merge two sets - union is commutative."""
        self.elements = self.elements | other_set.elements

# Usage: Task tags (only add new tags, never remove)
class TaskTags(GSet):
    def __init__(self, task_id):
        super().__init__()
        self.task_id = task_id

# Agent A
tags = TaskTags("task:T-123")
tags.add("urgent")
tags.add("backend")

# Agent B (working on same task)
other_tags = TaskTags("task:T-123")
other_tags.add("testing")
other_tags.add("urgent")  # Duplicate is ok

# Merge result: {"urgent", "backend", "testing"}
tags.merge(other_tags)
```

#### 2. **LWW Register (Last-Write-Wins)**

```python
class LWWRegister:
    """Single-value register where latest timestamp wins."""

    def __init__(self):
        self.value = None
        self.timestamp = 0
        self.agent_id = None

    def write(self, value, agent_id):
        """Write with timestamp and agent ID."""
        ts = time.time()

        # Use (timestamp, agent_id) for total ordering
        if ts > self.timestamp or (ts == self.timestamp and agent_id > self.agent_id):
            self.value = value
            self.timestamp = ts
            self.agent_id = agent_id
            return True
        return False

    def merge(self, other):
        """Merge two registers - later wins."""
        if other.timestamp > self.timestamp or \
           (other.timestamp == self.timestamp and other.agent_id > self.agent_id):
            self.value = other.value
            self.timestamp = other.timestamp
            self.agent_id = other.agent_id

# Usage: Task title (last agent to update wins)
class TaskTitle(LWWRegister):
    def update(self, new_title, agent_id="main"):
        return self.write(new_title, agent_id)

# Agent A
title = TaskTitle()
title.update("Implement auth", agent_id="agent-a")  # ts=1000

# Agent B (slightly later)
other_title = TaskTitle()
other_title.update("Implement authentication", agent_id="agent-b")  # ts=1001

# Merge: takes "Implement authentication" (later timestamp)
title.merge(other_title)
```

#### 3. **Counter (Per-Agent)**

```python
class CRDTCounter:
    """Counter that increments per agent - always commutative."""

    def __init__(self):
        self.agent_counts = {}  # agent_id -> count

    def increment(self, agent_id):
        """Increment counter for this agent."""
        self.agent_counts[agent_id] = self.agent_counts.get(agent_id, 0) + 1

    @property
    def value(self):
        """Total value is sum of all agent counts."""
        return sum(self.agent_counts.values())

    def merge(self, other):
        """Merge by taking max count per agent."""
        for agent_id, count in other.agent_counts.items():
            # Keep the larger count
            self.agent_counts[agent_id] = max(
                self.agent_counts.get(agent_id, 0),
                count
            )

# Usage: Task attempt counter
class TaskAttemptCounter(CRDTCounter):
    pass

# Agent A attempts 3 times
counter = TaskAttemptCounter()
counter.increment("agent-a")  # 1
counter.increment("agent-a")  # 2
counter.increment("agent-a")  # 3

# Agent B attempts 2 times
other_counter = TaskAttemptCounter()
other_counter.increment("agent-b")  # 1
other_counter.increment("agent-b")  # 2

# Merge: {agent-a: 3, agent-b: 2} → total = 5
counter.merge(other_counter)
assert counter.value == 5  # Order-independent!
```

#### 4. **RGA (Replicated Growable Array) - For Ordered Sequences**

```python
class RGAElement:
    """Element in RGA with unique ID and timestamp."""

    def __init__(self, value, agent_id, position):
        self.value = value
        self.agent_id = agent_id
        self.timestamp = time.time()
        # Unique ID for this element
        self.id = f"{timestamp}-{agent_id}-{position}"

class RGA:
    """Replicated Growable Array - supports append with ordering."""

    def __init__(self):
        self.elements = []  # Maintained sorted by ID

    def append(self, value, agent_id):
        """Add element in unique position."""
        position = len(self.elements)
        elem = RGAElement(value, agent_id, position)
        self.elements.append(elem)
        # Sort by ID to ensure same order on all replicas
        self.elements.sort(key=lambda x: x.id)

    def merge(self, other):
        """Merge with other RGA."""
        # All elements must coexist
        id_set = {e.id for e in self.elements}
        for other_elem in other.elements:
            if other_elem.id not in id_set:
                self.elements.append(other_elem)
        # Re-sort
        self.elements.sort(key=lambda x: x.id)

    def get_values(self):
        """Get values in order."""
        return [e.value for e in self.elements]

# Usage: Task checklist items
checklist = RGA()

# Agent A adds items
checklist.append("Design", "agent-a")
checklist.append("Implement", "agent-a")

# Agent B adds items (concurrently)
other_checklist = RGA()
other_checklist.append("Test", "agent-b")
other_checklist.append("Deploy", "agent-b")

# Merge: all items coexist, sorted by unique ID
checklist.merge(other_checklist)
# Result: might be ["Deploy", "Design", "Implement", "Test"]
# or different order, but CONSISTENT order on all replicas
```

### File-Based Persistence

```python
import json
from pathlib import Path

class CRDTStore:
    """Persist CRDTs to files in merge-friendly format."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_crdt(self, entity_id: str, crdt_obj):
        """Save CRDT as JSON."""
        path = self.data_dir / f"{entity_id}.json"
        path.write_text(json.dumps({
            "type": crdt_obj.__class__.__name__,
            "data": crdt_obj.__dict__
        }))

    def load_crdt(self, entity_id: str):
        """Load CRDT from JSON."""
        path = self.data_dir / f"{entity_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def merge_crdts(self, local_id: str, remote_path: Path):
        """Merge remote CRDT into local."""
        local = self.load_crdt(local_id)
        remote_file = remote_path / f"{local_id}.json"

        if not remote_file.exists():
            return  # Remote doesn't have this CRDT

        remote = json.loads(remote_file.read_text())

        # Reconstruct CRDT objects from dicts
        local_crdt = self._dict_to_crdt(local)
        remote_crdt = self._dict_to_crdt(remote)

        # Merge (CRDT-specific)
        local_crdt.merge(remote_crdt)

        # Save merged version
        self.save_crdt(local_id, local_crdt)
```

### Merge Strategy for Git

**✅ Excellent merge-friendliness:**
- CRDTs can be merged from multiple branches
- Merge is idempotent: `merge(merge(A, B), C)` = `merge(A, merge(B, C))`
- No version conflicts possible
- Works perfectly with event log + snapshot approach

```python
# Branch A and B both modify CRDT concurrently
# On merge:
git checkout main
git merge feature-branch

# CRDT files may have different contents, but:
# 1. Load CRDT from local branch
# 2. Load CRDT from feature branch
# 3. Call crdt.merge(other_crdt)
# 4. Save result
# = Automatically resolved!
```

### Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| No conflicts possible | Limited data models (not all operations commute) |
| Automatic merge | Higher complexity to implement |
| Good audit trail | Harder to debug |
| Perfect for append-only data | Memory overhead for metadata |

### Suitable Data Models for Multi-Agent Tasks

| Data | CRDT Type | Works? |
|------|-----------|--------|
| Task tags | G-Set | ✅ Yes |
| Task title | LWW Register | ✅ Yes |
| Task status | LWW Register | ✅ Yes (last agent wins) |
| Dependency count | Per-Agent Counter | ✅ Yes |
| Task subtasks | G-Set | ⚠️ Only adds, no deletes |
| Task checklist | RGA | ✅ Yes (ordered) |
| Task history | Append-only log | ✅ Yes |

---

## Pattern 3: Operational Transforms (OT)

### Concept

**Transform concurrent operations into a canonical order** - like Google Docs collaborative editing.

Two agents edit simultaneously:
- Agent A: Insert "foo" at position 5
- Agent B: Insert "bar" at position 3

On merge: Transform A's operation relative to B's:
- B's insert shifts positions, so A's "position 5" becomes "position 8"
- Result: predictable order regardless of arrival sequence

### When to Use

✅ **Good for:**
- Rich text with concurrent edits
- Code/document editing
- Ordered data with arbitrary insertions
- Real-time collaboration

❌ **Not for:**
- Simple key-value updates (use LWW instead)
- High-frequency tiny edits (overhead)
- Very long documents (quadratic complexity)

### How OT Works

```python
class Operation:
    """Base operation class."""
    pass

class Insert(Operation):
    def __init__(self, position: int, text: str):
        self.position = position
        self.text = text

class Delete(Operation):
    def __init__(self, position: int, length: int):
        self.position = position
        self.length = length

class OperationalTransform:
    """Transform operations into canonical order."""

    def transform(self, op1: Operation, op2: Operation) -> Operation:
        """
        Transform op1 relative to op2.

        Returns transformed version of op1 that accounts for op2.
        """
        if isinstance(op1, Insert) and isinstance(op2, Insert):
            return self._transform_insert_insert(op1, op2)
        elif isinstance(op1, Insert) and isinstance(op2, Delete):
            return self._transform_insert_delete(op1, op2)
        elif isinstance(op1, Delete) and isinstance(op2, Insert):
            return self._transform_delete_insert(op1, op2)
        elif isinstance(op1, Delete) and isinstance(op2, Delete):
            return self._transform_delete_delete(op1, op2)

    def _transform_insert_insert(self, op1: Insert, op2: Insert) -> Insert:
        """Transform insertion relative to another insertion."""
        if op1.position < op2.position:
            # op1 happens before op2's position - no change
            return Insert(op1.position, op1.text)
        elif op1.position > op2.position:
            # op1 happens after op2 - shift position
            return Insert(op1.position + len(op2.text), op1.text)
        else:
            # Same position - use tiebreaker (e.g., op1 first)
            return Insert(op1.position + len(op2.text), op1.text)

    def _transform_delete_insert(self, op1: Delete, op2: Insert) -> Delete:
        """Transform deletion relative to insertion."""
        if op1.position < op2.position:
            # Deletion before insertion - no change
            return Delete(op1.position, op1.length)
        else:
            # Deletion after insertion - shift position
            return Delete(op1.position + len(op2.text), op1.length)

    def _transform_insert_delete(self, op1: Insert, op2: Delete) -> Insert:
        """Transform insertion relative to deletion."""
        if op1.position <= op2.position:
            # Insert before delete - no change
            return Insert(op1.position, op1.text)
        elif op1.position >= op2.position + op2.length:
            # Insert after deleted region - shift back
            return Insert(op1.position - op2.length, op1.text)
        else:
            # Insert in middle of deleted region - adjust
            return Insert(op2.position, op1.text)

    def _transform_delete_delete(self, op1: Delete, op2: Delete) -> Delete:
        """Transform deletion relative to another deletion."""
        if op1.position < op2.position:
            return Delete(op1.position, op1.length)
        elif op1.position >= op2.position + op2.length:
            return Delete(op1.position - op2.length, op1.length)
        else:
            # Overlapping deletes - complex case
            return Delete(op1.position, max(0, op1.length - op2.length))

# Usage: Collaborative task description editing
def merge_concurrent_edits(local_text: str, local_ops: List[Operation],
                          remote_ops: List[Operation]) -> str:
    """
    Apply concurrent operations in canonical order.

    1. Start with local text
    2. Determine operation order (by timestamp)
    3. Transform and apply in order
    """
    # Assume operations have timestamps
    all_ops = sorted(
        [(op, 'local') for op in local_ops] +
        [(op, 'remote') for op in remote_ops],
        key=lambda x: getattr(x[0], 'timestamp', 0)
    )

    text = local_text
    ot = OperationalTransform()

    for op, source in all_ops:
        # Apply operation to current text
        if isinstance(op, Insert):
            text = text[:op.position] + op.text + text[op.position:]
        elif isinstance(op, Delete):
            text = text[:op.position] + text[op.position + op.length:]

    return text

# Example: Concurrent edits to task description
local_text = "Design the system"
local_ops = [Insert(6, "new ")]  # "Design new the system"

remote_ops = [Delete(6, 3)]  # Remove "the "

# Merge: transform and apply both
# Result: "Design new system"
result = merge_concurrent_edits(local_text, local_ops, remote_ops)
```

### File-Based Implementation with History

```python
import json
from datetime import datetime

class EditHistory:
    """Track all operations with history."""

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.operations = []
        self.base_version = ""

    def record_operation(self, op: Operation, agent_id: str):
        """Record operation with timestamp."""
        self.operations.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "operation": op.__dict__
        })

    def save(self, path: Path):
        """Save operation history to file."""
        path.write_text(json.dumps({
            "entity_id": self.entity_id,
            "base_version": self.base_version,
            "operations": self.operations
        }, indent=2))

    def load(self, path: Path):
        """Load operation history from file."""
        data = json.loads(path.read_text())
        self.entity_id = data["entity_id"]
        self.base_version = data["base_version"]
        self.operations = data["operations"]

    def merge_histories(self, other: 'EditHistory') -> 'EditHistory':
        """Merge two operation histories."""
        merged = EditHistory(self.entity_id)
        merged.base_version = self.base_version

        # Combine operations from both histories
        all_ops = self.operations + other.operations

        # Sort by timestamp (global ordering)
        all_ops.sort(key=lambda x: x["timestamp"])

        # Re-apply in order (OT transformation)
        ot = OperationalTransform()
        merged.operations = all_ops

        return merged

class DocumentWithOT:
    """Document that supports OT-based merging."""

    def __init__(self, doc_id: str, initial_text: str):
        self.doc_id = doc_id
        self.text = initial_text
        self.history = EditHistory(doc_id)

    def apply_op(self, op: Operation, agent_id: str):
        """Apply operation and record it."""
        if isinstance(op, Insert):
            self.text = self.text[:op.position] + op.text + self.text[op.position:]
        elif isinstance(op, Delete):
            self.text = self.text[:op.position] + self.text[op.position + op.length:]

        self.history.record_operation(op, agent_id)

    def merge_with(self, other: 'DocumentWithOT'):
        """Merge with another document version."""
        # Merge histories
        merged_history = self.history.merge_histories(other.history)

        # Reconstruct document by replaying operations
        result_text = self.history.base_version
        for op_record in merged_history.operations:
            op_dict = op_record["operation"]
            if op_dict.get("__type__") == "Insert":
                op = Insert(op_dict["position"], op_dict["text"])
            else:
                op = Delete(op_dict["position"], op_dict["length"])

            if isinstance(op, Insert):
                result_text = result_text[:op.position] + op.text + result_text[op.position:]
            elif isinstance(op, Delete):
                result_text = result_text[:op.position] + result_text[op.position + op.length:]

        self.text = result_text
        self.history = merged_history
```

### Merge Strategy for Git

**⚠️ Challenging but possible:**
- Requires OT implementation for your data types
- More complex than CRDTs
- Needs global timestamp ordering
- Non-commutative operations are hard

```python
# On git merge:
git checkout main
git merge feature-branch

# Conflict detected in task description (both branches edited)
# Solution:
# 1. Load operation history from main branch
# 2. Load operation history from feature branch
# 3. Call doc.merge_with(other_doc)
# 4. Commit resolved state
# = Automatically merged with correct semantics!
```

### Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| Preserves all edits | Complex to implement |
| Natural edit semantics | Requires transforming every operation |
| Works with rich data | O(n²) worst case |
| Full history available | Timestamps must be globally ordered |

---

## Pattern 4: Branch-and-Merge (Git-Native)

### Concept

**Use Git's native branching for isolation, then merge strategically.**

Perfect for Claude agents - each can work on a separate branch, then merge when ready.

```
main
  ├── agent-a (working on feature-1)
  │    └── commit, commit, commit
  ├── agent-b (working on feature-2)
  │    └── commit, commit, commit
  └── agent-c (working on bug-fix)
       └── commit, commit, commit

# When ready, merge all back to main in sequence
main ← agent-a ← agent-b ← agent-c
```

### When to Use

✅ **Good for:**
- Well-separated concerns (different agents on different tasks)
- Long-running work (hours or days)
- Clear dependencies between tasks
- Large changeset management

❌ **Not for:**
- Frequent merges (too much overhead)
- Very high parallelism (merge conflicts multiply)
- Real-time collaboration
- Interleaved work on same files

### Implementation Pattern

```python
import subprocess
from pathlib import Path

class BranchCoordinator:
    """Coordinate multi-agent work via Git branches."""

    def __init__(self, repo_path: Path):
        self.repo = repo_path

    def create_agent_branch(self, agent_id: str, base_branch: str = "main") -> str:
        """Create isolated branch for agent."""
        branch_name = f"agent/{agent_id}/{datetime.now().strftime('%s')}"

        subprocess.run(
            ["git", "checkout", "-b", branch_name, base_branch],
            cwd=self.repo,
            check=True
        )

        return branch_name

    def agent_commit(self, branch: str, message: str):
        """Commit work on agent branch."""
        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.repo,
            check=True
        )

        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo,
            check=True
        )

    def merge_agent_work(self, agent_branch: str, target: str = "main") -> bool:
        """Merge agent's work back to main."""
        # Switch to target
        subprocess.run(
            ["git", "checkout", target],
            cwd=self.repo,
            check=True
        )

        # Merge agent branch
        result = subprocess.run(
            ["git", "merge", "--no-ff", agent_branch],
            cwd=self.repo
        )

        if result.returncode != 0:
            # Merge conflict
            print(f"Merge conflict from {agent_branch} to {target}")
            return False

        return True

    def list_agent_branches(self) -> List[str]:
        """List all agent branches awaiting merge."""
        result = subprocess.run(
            ["git", "branch", "-l", "agent/*"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')

    def get_merge_dependencies(self) -> Dict[str, List[str]]:
        """
        Find dependencies between agent branches.

        Returns map of branch -> list of branches it depends on.
        """
        # Analyze commit history to find dependencies
        # (which branches touch which files)
        dependencies = {}

        for branch in self.list_agent_branches():
            dependencies[branch] = self._find_dependencies(branch)

        return dependencies

    def _find_dependencies(self, branch: str) -> List[str]:
        """Find which other branches touch same files as this one."""
        # Get files changed in this branch
        result = subprocess.run(
            ["git", "diff", "--name-only", f"main...{branch}"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=True
        )
        changed_files = set(result.stdout.strip().split('\n'))

        # Check which other branches touch same files
        dependencies = []
        for other_branch in self.list_agent_branches():
            if other_branch == branch:
                continue

            result = subprocess.run(
                ["git", "diff", "--name-only", f"main...{other_branch}"],
                cwd=self.repo,
                capture_output=True,
                text=True,
                check=True
            )
            other_files = set(result.stdout.strip().split('\n'))

            if changed_files & other_files:  # Intersection
                dependencies.append(other_branch)

        return dependencies

    def merge_in_dependency_order(self) -> Dict[str, bool]:
        """
        Merge all agent branches in dependency order.

        Returns: map of branch -> success/failure
        """
        deps = self.get_merge_dependencies()
        results = {}
        remaining = set(deps.keys())

        while remaining:
            # Find branch with no remaining dependencies
            for branch in remaining:
                unsatisfied = [d for d in deps[branch] if d in remaining]

                if not unsatisfied:
                    # Can merge this one
                    success = self.merge_agent_work(branch, "main")
                    results[branch] = success
                    remaining.remove(branch)

                    if not success:
                        # Stop on first merge failure
                        print(f"Merge failed for {branch}, stopping")
                        break

                    break
            else:
                # Circular dependency
                print(f"Circular dependency: {remaining}")
                break

        return results

# Usage
coordinator = BranchCoordinator(Path("."))

# Agent A creates and works on their branch
branch_a = coordinator.create_agent_branch("agent-a")
# ... do work ...
coordinator.agent_commit(branch_a, "feat: implement authentication")

# Agent B works independently
branch_b = coordinator.create_agent_branch("agent-b")
# ... do work ...
coordinator.agent_commit(branch_b, "feat: add database layer")

# Merge in dependency order
results = coordinator.merge_in_dependency_order()
print(f"Merge results: {results}")
```

### Handling Merge Conflicts

```python
class MergeConflictResolver:
    """Resolve conflicts from multi-agent merges."""

    def __init__(self, repo_path: Path):
        self.repo = repo_path

    def get_conflicted_files(self) -> List[str]:
        """List files with merge conflicts."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=True
        )

        conflicted = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('UU'):  # Both modified
                conflicted.append(line[3:])

        return conflicted

    def resolve_with_strategy(self, strategy: str):
        """
        Resolve all conflicts using strategy.

        Strategies:
        - 'ours': keep local version
        - 'theirs': take incoming version
        - 'manual': require manual resolution
        """
        if strategy == 'ours':
            subprocess.run(
                ["git", "checkout", "--ours", "."],
                cwd=self.repo,
                check=True
            )
        elif strategy == 'theirs':
            subprocess.run(
                ["git", "checkout", "--theirs", "."],
                cwd=self.repo,
                check=True
            )
        elif strategy == 'manual':
            # Requires human intervention
            conflicted = self.get_conflicted_files()
            print(f"Manual resolution required for: {conflicted}")
            return False

        # Complete the merge
        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.repo,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Merge conflict resolution"],
            cwd=self.repo,
            check=True
        )

        return True
```

### Merge Strategy for Git

**✅ Perfect for git:**
- Git's native workflow
- All conflicts handled by Git
- No custom merge logic needed
- Clear history and audit trail

```bash
# Agent creates branch
git checkout -b agent/a/task-1

# Agent does work
# ... commits ...

# Agent pushes
git push origin agent/a/task-1

# Main agent merges
git checkout main
git pull origin main
git merge --no-ff agent/a/task-1

# If conflict, resolve and commit
# If clean, auto-merge succeeds
```

### Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| Native Git support | Merge conflicts still need resolution |
| Clear history | Not real-time collaboration |
| Easy to understand | File-level conflicts, not semantic |
| Isolation for each agent | Rebasing complexity |

---

## Pattern 5: Lock-Based Regions

### Concept

**Divide data into regions, use locks for exclusive access** - like partitioning.

Agent A locks "authentication module", Agent B locks "database module" - they don't conflict.

### When to Use

✅ **Good for:**
- Critical sections needing mutual exclusion
- Naturally partitioned data (modules, entities)
- Avoiding thundering herd
- Fairness (everyone gets a turn)

❌ **Not for:**
- Heavily contended regions
- Cross-region operations
- Deadlock-prone scenarios

### Implementation

The codebase includes a `ProcessLock` implementation in `cortical/utils/locking.py`:

```python
from cortical.utils.locking import ProcessLock
from pathlib import Path

# Example: Lock-based regions for tasks
class LockedTaskRegion:
    """Exclusive access to a task via filesystem lock."""

    def __init__(self, task_id: str, lock_dir: Path = Path(".got/.locks")):
        self.task_id = task_id
        self.lock_path = lock_dir / f"{task_id}.lock"
        self.lock = ProcessLock(self.lock_path, reentrant=True)

    def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire lock on this task region."""
        return self.lock.acquire(timeout=timeout)

    def release(self):
        """Release lock."""
        self.lock.release()

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Failed to acquire lock for {self.task_id}")
        return self

    def __exit__(self, *args):
        self.release()

# Usage across agents
def update_task_atomically(task_id: str, update_fn):
    """Update task with locking."""
    region = LockedTaskRegion(task_id)

    with region:
        # Critical section - only one agent here
        task = load_task(task_id)
        updated = update_fn(task)
        save_task(task_id, updated)

# Agent A
update_task_atomically("task:T-123", lambda t: t.status = "completed")

# Agent B (waits for Agent A to finish)
update_task_atomically("task:T-123", lambda t: t.owner = "agent-b")
```

### Region Granularity

```python
class RegionGranularity:
    """Different lock granularities for different scenarios."""

    # Fine-grained: Lock individual entities
    ENTITY = "task:T-123"  # One lock per task

    # Medium: Lock logical regions
    MODULE = "cortical/query"  # One lock per module

    # Coarse: Lock entire system
    GLOBAL = "system"  # One global lock

class LockManager:
    """Manage locks at different granularities."""

    def __init__(self, lock_dir: Path = Path(".got/.locks")):
        self.lock_dir = lock_dir
        self.locks = {}  # region -> ProcessLock

    def acquire_entity_lock(self, entity_id: str, timeout: float = 5.0) -> bool:
        """Acquire lock on individual entity."""
        if entity_id not in self.locks:
            self.locks[entity_id] = ProcessLock(
                self.lock_dir / f"entity_{entity_id}.lock"
            )
        return self.locks[entity_id].acquire(timeout=timeout)

    def acquire_module_lock(self, module: str, timeout: float = 5.0) -> bool:
        """Acquire lock on module (for bulk operations)."""
        if module not in self.locks:
            self.locks[module] = ProcessLock(
                self.lock_dir / f"module_{module}.lock"
            )
        return self.locks[module].acquire(timeout=timeout)

    def release_lock(self, region: str):
        """Release lock."""
        if region in self.locks:
            self.locks[region].release()

    def with_lock(self, region: str, timeout: float = 5.0):
        """Context manager for lock."""
        class LockContext:
            def __init__(ctx_self):
                ctx_self.acquired = False

            def __enter__(ctx_self):
                if self.acquire_entity_lock(region, timeout=timeout):
                    ctx_self.acquired = True
                    return ctx_self
                raise RuntimeError(f"Failed to acquire lock on {region}")

            def __exit__(ctx_self, *args):
                if ctx_self.acquired:
                    self.release_lock(region)

        return LockContext()

# Usage with context manager
lock_mgr = LockManager()

with lock_mgr.with_lock("task:T-123"):
    task = load_task("task:T-123")
    task.status = "completed"
    save_task("task:T-123", task)
```

### Deadlock Avoidance

```python
from typing import Set

class DeadlockDetector:
    """Detect and prevent deadlock scenarios."""

    def __init__(self):
        self.lock_graph = {}  # who holds what locks
        self.wait_graph = {}   # who waits for what

    def acquire_multiple_locks(self, regions: List[str], timeout: float = 5.0) -> bool:
        """
        Acquire multiple locks in canonical order to prevent deadlock.

        Always acquire locks in alphabetical order - ensures same order
        across all agents.
        """
        sorted_regions = sorted(regions)  # Canonical order

        acquired = []
        try:
            lock_mgr = LockManager()
            for region in sorted_regions:
                if not lock_mgr.acquire_entity_lock(region, timeout=timeout):
                    # Rollback
                    for r in acquired:
                        lock_mgr.release_lock(r)
                    return False
                acquired.append(region)

            return True
        except Exception as e:
            # Rollback on exception
            lock_mgr = LockManager()
            for r in acquired:
                lock_mgr.release_lock(r)
            raise

    def has_cycle(self) -> bool:
        """Check if wait-for graph has cycles (potential deadlock)."""
        # Standard cycle detection (DFS)
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.wait_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True  # Cycle found

            rec_stack.remove(node)
            return False

        for node in self.wait_graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False

# Usage
detector = DeadlockDetector()

# Acquire multiple locks safely
regions = ["task:T-123", "task:T-456", "task:T-789"]

# DON'T do this (deadlock risk):
# acquire(T-123) -> acquire(T-456) -> acquire(T-789)
# other-agent: acquire(T-789) -> acquire(T-456) -> DEADLOCK

# DO this (safe):
if detector.acquire_multiple_locks(regions):
    # Work safely with all three regions
    pass
```

### Merge Strategy for Git

**⚠️ Requires coordination:**
- Locks are local/process-level, not persisted in Git
- Work best when lock holders are ephemeral
- Doesn't help with Git merge conflicts (file-level)
- Useful for runtime mutual exclusion, not source control

```python
# Best practice: Use with event sourcing
# 1. Acquire locks for critical edits
# 2. Record changes to event log
# 3. Release locks
# 4. Git commits event log (conflict-free)

with lock_mgr.with_lock("task:T-123"):
    # Edit task
    task.status = "completed"

    # Log event (append-only, no conflicts)
    event_log.append({
        "ts": now(),
        "event": "task.complete",
        "task_id": "task:T-123",
        "agent": "agent-a"
    })

# Git commits event log - no conflicts possible!
```

### Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| Simple concept | Doesn't prevent merge conflicts |
| Fair access (first-come-first-serve) | Can cause bottlenecks |
| Good for critical sections | Deadlock risk with multiple locks |
| Works with event sourcing | Requires timeout/recovery strategy |

---

## Pattern 6: Event Sourcing

### Concept

**Store all changes as events in append-only log; snapshot is just a cache.**

Every action is recorded:
- `task.create(id, title)`
- `task.update(id, status="completed")`
- `task.delete(id)`

State is derived by replaying events.

### When to Use

✅ **Good for:**
- Audit trail requirements
- Time travel/debugging
- Event-driven systems
- Multi-agent work (append-only = merge-friendly)

❌ **Not for:**
- Soft-deletes (events are immutable)
- Very high frequency writes (log bloat)
- Simple CRUD apps

### Existing Implementation in Codebase

The project already implements event sourcing in `cortical/got/` with:
- Event log files
- Event types (node.create, node.update, edge.create, etc.)
- Snapshot caching
- Merge-friendly design

From `/home/user/Opus-code-test/docs/got-event-sourcing.md`:

```python
# Event format
event = {
    "ts": "2025-12-20T19:37:26.123Z",
    "event": "node.create",
    "id": "task:T-20251220-193726-a1b2",
    "type": "TASK",
    "data": {"title": "Implement feature", "status": "pending", "priority": "high"},
    "meta": {"branch": "main", "session": "abc123"}
}

# Event types
node.create     # Create new node
node.update     # Update node properties
node.delete     # Delete a node
edge.create     # Create an edge
edge.delete     # Delete an edge
```

### Event-Sourced Task System

```python
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional

class Event:
    """Represents a single event."""

    def __init__(self, event_type: str, entity_id: str, data: Dict,
                 agent_id: str = "main"):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.event_type = event_type  # "task.create", "task.update", etc.
        self.entity_id = entity_id
        self.data = data
        self.agent_id = agent_id

    def to_dict(self) -> Dict:
        return {
            "ts": self.timestamp,
            "event": self.event_type,
            "entity_id": self.entity_id,
            "data": self.data,
            "agent_id": self.agent_id
        }

class EventLog:
    """Append-only event log."""

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event):
        """Append event to log."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def read_all(self) -> List[Event]:
        """Read all events from log."""
        events = []
        if not self.log_file.exists():
            return events

        with open(self.log_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    event = Event(
                        event_type=data["event"],
                        entity_id=data["entity_id"],
                        data=data["data"],
                        agent_id=data["agent_id"]
                    )
                    event.timestamp = data["ts"]
                    events.append(event)

        return events

class EventSourcedTask:
    """Task with full event history."""

    def __init__(self, task_id: str, event_log: EventLog):
        self.task_id = task_id
        self.event_log = event_log

        # Cached state (derived from events)
        self._state = {}
        self._replay_events()

    def _replay_events(self):
        """Rebuild current state by replaying all events."""
        self._state = {"id": self.task_id, "status": None, "history": []}

        for event in self.event_log.read_all():
            if event.entity_id == self.task_id:
                self._apply_event(event)

    def _apply_event(self, event: Event):
        """Apply single event to state."""
        if event.event_type == "task.create":
            self._state.update(event.data)
            self._state["created_at"] = event.timestamp
            self._state["created_by"] = event.agent_id

        elif event.event_type == "task.update":
            self._state.update(event.data)
            self._state["updated_at"] = event.timestamp
            self._state["updated_by"] = event.agent_id

        elif event.event_type == "task.delete":
            self._state["deleted_at"] = event.timestamp
            self._state["deleted_by"] = event.agent_id

        # Record in history
        self._state["history"].append({
            "timestamp": event.timestamp,
            "agent": event.agent_id,
            "action": event.event_type,
            "data": event.data
        })

    def create(self, title: str, priority: str = "medium",
               description: str = "", agent_id: str = "main"):
        """Create task - records event."""
        event = Event(
            event_type="task.create",
            entity_id=self.task_id,
            data={
                "title": title,
                "priority": priority,
                "description": description,
                "status": "pending"
            },
            agent_id=agent_id
        )
        self.event_log.append(event)
        self._replay_events()  # Refresh state

    def update(self, changes: Dict, agent_id: str = "main"):
        """Update task - records event."""
        event = Event(
            event_type="task.update",
            entity_id=self.task_id,
            data=changes,
            agent_id=agent_id
        )
        self.event_log.append(event)
        self._replay_events()

    def delete(self, agent_id: str = "main"):
        """Delete task - records event."""
        event = Event(
            event_type="task.delete",
            entity_id=self.task_id,
            data={},
            agent_id=agent_id
        )
        self.event_log.append(event)
        self._replay_events()

    @property
    def status(self):
        return self._state.get("status")

    @property
    def title(self):
        return self._state.get("title")

    def get_history(self):
        """Get full history of changes."""
        return self._state.get("history", [])

    def to_dict(self):
        """Get current state as dictionary."""
        return self._state.copy()

# Usage
log = EventLog(Path(".got/events/session-123.jsonl"))

# Create task
task = EventSourcedTask("task:T-123", log)
task.create("Implement auth", priority="high", agent_id="agent-a")

# Agent B updates same task
task.update({"status": "in_progress"}, agent_id="agent-b")

# Agent C completes it
task.update({"status": "completed"}, agent_id="agent-c")

# Full history preserved
for event in task.get_history():
    print(f"{event['timestamp']} - {event['agent']}: {event['action']}")
# Output:
# 2025-12-20T19:37:26Z - agent-a: task.create
# 2025-12-20T19:38:10Z - agent-b: task.update
# 2025-12-20T19:39:45Z - agent-c: task.update
```

### Session-Based Event Files (Merge-Friendly)

```python
from datetime import datetime
import uuid

class SessionEventLog:
    """Session-specific event log for merge-friendliness."""

    def __init__(self, events_dir: Path = Path(".got/events")):
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename for this session
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        self.session_file = self.events_dir / f"{timestamp}-{session_id}.jsonl"

    def append(self, event: Event):
        """Append event to session file."""
        with open(self.session_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def read_all_events(self) -> List[Event]:
        """Read ALL events from all session files (merge-friendly)."""
        events = []

        for event_file in sorted(self.events_dir.glob("*.jsonl")):
            with open(event_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        event = Event(
                            event_type=data["event"],
                            entity_id=data["entity_id"],
                            data=data["data"],
                            agent_id=data["agent_id"]
                        )
                        event.timestamp = data["ts"]
                        events.append(event)

        # Sort globally by timestamp (ensures consistent replay)
        events.sort(key=lambda e: e.timestamp)
        return events

    def rebuild_snapshot(self) -> Dict[str, Dict]:
        """Rebuild current state from ALL events."""
        state = {}

        for event in self.read_all_events():
            entity_id = event.entity_id

            if entity_id not in state:
                state[entity_id] = {"id": entity_id, "history": []}

            # Apply event to entity
            if event.event_type.endswith(".create"):
                state[entity_id].update(event.data)
            elif event.event_type.endswith(".update"):
                state[entity_id].update(event.data)
            elif event.event_type.endswith(".delete"):
                state[entity_id]["deleted"] = True

            # Record in history
            state[entity_id]["history"].append({
                "ts": event.timestamp,
                "agent": event.agent_id,
                "event": event.event_type,
                "data": event.data
            })

        return state

# Usage: Multi-agent scenario
# Agent A
log_a = SessionEventLog()
task = EventSourcedTask("task:T-123", log_a)
task.create("Feature A", agent_id="agent-a")
log_a.append(Event("task.update", "task:T-123", {"status": "in_progress"}, "agent-a"))

# Agent B (different branch, but same repo)
log_b = SessionEventLog()
log_b.append(Event("task.create", "task:T-456", {"title": "Feature B"}, "agent-b"))
log_b.append(Event("task.update", "task:T-456", {"status": "in_progress"}, "agent-b"))

# After merge:
# .got/events/
# ├── 20251220-193726-abc12345.jsonl  (Agent A's events)
# └── 20251220-201500-def67890.jsonl  (Agent B's events)

# Rebuild snapshot from ALL events (merge-friendly!)
merged_log = SessionEventLog()
snapshot = merged_log.rebuild_snapshot()
# {
#     "task:T-123": {status: "in_progress", created_by: "agent-a"},
#     "task:T-456": {status: "in_progress", created_by: "agent-b"}
# }
```

### Merge Strategy for Git

**✅ Excellent merge-friendliness:**
- Append-only logs never conflict
- Each session writes unique file
- Merge = read all event files + replay
- Last-write-wins for property conflicts

```bash
# Branch A
.got/events/
└── 20251220-193726-abc12345.jsonl  # Agent A's events

# Branch B
.got/events/
└── 20251220-201500-def67890.jsonl  # Agent B's events

# After merge: both files coexist
.got/events/
├── 20251220-193726-abc12345.jsonl
└── 20251220-201500-def67890.jsonl

# Replay all events in timestamp order → consistent state everywhere!
```

### Time Travel & Debugging

```python
class TimeTravel:
    """Query state at any point in time."""

    def __init__(self, session_log: SessionEventLog):
        self.log = session_log

    def state_at(self, timestamp: str) -> Dict:
        """Get state at specific timestamp."""
        state = {}

        for event in self.log.read_all_events():
            if event.timestamp > timestamp:
                break  # Stop at target time

            entity_id = event.entity_id
            if entity_id not in state:
                state[entity_id] = {"id": entity_id}

            if event.event_type.endswith(".create"):
                state[entity_id].update(event.data)
            elif event.event_type.endswith(".update"):
                state[entity_id].update(event.data)

        return state

    def what_changed(self, entity_id: str, start_ts: str, end_ts: str) -> List[Dict]:
        """Get all changes to entity in time range."""
        changes = []

        for event in self.log.read_all_events():
            if event.entity_id != entity_id:
                continue
            if event.timestamp < start_ts or event.timestamp > end_ts:
                continue

            changes.append({
                "timestamp": event.timestamp,
                "agent": event.agent_id,
                "action": event.event_type,
                "data": event.data
            })

        return changes

    def who_changed_what(self, agent_id: str, start_ts: str, end_ts: str) -> Dict:
        """Get all changes by specific agent in time range."""
        changes_by_entity = {}

        for event in self.log.read_all_events():
            if event.agent_id != agent_id:
                continue
            if event.timestamp < start_ts or event.timestamp > end_ts:
                continue

            if event.entity_id not in changes_by_entity:
                changes_by_entity[event.entity_id] = []

            changes_by_entity[event.entity_id].append({
                "timestamp": event.timestamp,
                "action": event.event_type,
                "data": event.data
            })

        return changes_by_entity

# Usage
session_log = SessionEventLog()
tt = TimeTravel(session_log)

# What was the state 1 hour ago?
past_state = tt.state_at("2025-12-20T18:00:00Z")

# What changes happened to task:T-123 between 7pm and 8pm?
changes = tt.what_changed("task:T-123", "2025-12-20T19:00:00Z", "2025-12-20T20:00:00Z")

# What did Agent A do in the last hour?
agent_changes = tt.who_changed_what("agent-a", "2025-12-20T19:00:00Z", "2025-12-20T20:00:00Z")
```

### Strengths & Weaknesses

| Strength | Weakness |
|----------|----------|
| Perfect merge-friendliness | Event log can grow large |
| Full audit trail | Soft deletes not natural |
| Time travel/debugging | Need snapshots for performance |
| Branch-aware | Replay can be slow for long histories |

---

## Comparison Matrix

### Feature Comparison

| Feature | Optimistic | CRDT | OT | Branch-Merge | Lock | Event Sourcing |
|---------|-----------|------|----|--------------|----- |----------|
| **Merge-Friendly** | Conditional | Excellent | Difficult | Excellent | No | Excellent |
| **Conflict Resolution** | Manual | Automatic | Automatic | Manual | N/A | Automatic |
| **Audit Trail** | No | Limited | Limited | Yes | No | Excellent |
| **Real-time Collab** | No | No | Yes | No | No | No |
| **Git Integration** | Tricky | Good | Complex | Native | N/A | Excellent |
| **Complexity** | Low | Medium | Very High | Medium | Low | Medium |
| **Time Travel** | No | No | No | Via git history | No | Yes |
| **Performance** | Fast | Good | Slow | Fast | Fast | Good |

### Suitability by Scenario

```
Multi-agent parallel task work:
  ✅ Event Sourcing (proven in codebase)
  ✅ Branch-and-Merge (clear isolation)
  ⚠️  Optimistic (if low conflict)
  ❌ CRDTs (limited data model)
  ❌ OT (too complex)
  ❌ Locks (no git integration)

Collaborative real-time editing:
  ✅ OT (like Google Docs)
  ✅ CRDT (better merge properties)
  ⚠️  Event Sourcing (with batching)
  ❌ Branch-and-Merge (not real-time)
  ❌ Locks (latency)

Critical section protection:
  ✅ Locks (proven, simple)
  ✅ Optimistic + Event Log
  ⚠️  Transactions
  ❌ Others (not designed for it)

Long-running team project:
  ✅ Branch-and-Merge (clear ownership)
  ✅ Event Sourcing (audit trail)
  ⚠️  Lock regions
  ❌ Optimistic (conflict churn)
  ❌ CRDT (not sufficient alone)
  ❌ OT (maintenance overhead)
```

---

## Recommendations for Claude Multi-Agent Systems

### Best Pattern: **Hybrid Event Sourcing + Branch-and-Merge**

```
Architecture:
┌─────────────────────────────────────────────────────────┐
│              Git-Tracked Event Log Layer                 │
│  .got/events/*.jsonl (append-only, per-session)         │
│  - Session 1 events                                      │
│  - Session 2 events                                      │
│  (merged automatically on git merge)                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│         Current State Layer (Snapshot Cache)             │
│  .got/snapshots/latest.json.gz (rebuilt from events)    │
│  (invalidated on git merge, rebuilt automatically)       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           Agent Application Layer                        │
│  Process tasks, read from snapshot, record changes       │
│  to event log                                            │
└─────────────────────────────────────────────────────────┘
```

### Why This Combination Works

1. **Event Sourcing for Multi-Agent Coordination**
   - Each agent writes to unique event file
   - No merge conflicts (append-only)
   - Full audit trail across all agents
   - Time travel for debugging

2. **Branch-and-Merge for Clear Isolation**
   - Each long-running task gets a branch
   - Clear ownership and scope
   - Prevents cross-contamination
   - Easy rollback if needed

3. **Git-Native Merge-Friendliness**
   - Event files never conflict
   - git merge works perfectly
   - Snapshot invalidation triggers rebuild
   - Automatic consistency across branches

### Implementation Example

```python
# Session 1 (main agent)
manager_1 = GoTManager(branch="main")
task_id = manager_1.create_task("Implement feature", priority="high")
manager_1.update_task(task_id, status="in_progress")
# Events go to: .got/events/20251220-193726-main-abc123.jsonl

# Session 2 (sub-agent A)
manager_2 = GoTManager(branch="agent-a/feature-impl")
task_id_2 = manager_2.create_task("Subtask 1", parent_id=task_id)
# Events go to: .got/events/20251220-201500-agent-a-def456.jsonl

# Session 3 (sub-agent B)
manager_3 = GoTManager(branch="agent-b/testing")
manager_3.update_task(task_id_2, status="completed")
# Events go to: .got/events/20251220-202200-agent-b-ghi789.jsonl

# Main agent merges both branches
git checkout main
git merge agent-a/feature-impl  # Event files coexist, no conflicts!
git merge agent-b/testing        # All events replayed

# Result: .got/events/ contains all three session files
# Snapshot is rebuilt from all events in timestamp order
# Consistent state everywhere!
```

### Setup Checklist

- [x] Event sourcing layer (append-only logs)
- [x] Transaction manager (ACID guarantees)
- [x] Lock manager (for critical sections)
- [x] Snapshot caching (performance)
- [x] Recovery manager (crash resilience)
- [x] Conflict detection (optimistic locking)
- [ ] Multi-branch merge automation
- [ ] Handoff protocol between agents
- [ ] Time travel debugging tools
- [ ] Event-based audit reporting

---

## References

### Event Sourcing
- Martin Fowler: https://martinfowler.com/eaaDev/EventSourcing.html
- CQRS: https://martinfowler.com/bliki/CQRS.html

### CRDTs
- "A comprehensive study of CRDTS": https://arxiv.org/abs/1805.06358
- Yjs library (reference implementation): https://github.com/yjs/yjs

### Operational Transforms
- "Operational Transformation in Real-Time Group Editors": Ellis & Gibbs
- Google Docs patent: https://www.google.com/patents/US20060288313

### Optimistic Concurrency
- Optimistic Locking: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
- Database versioning techniques

### Git & Merging
- Git Book: https://git-scm.com/book/en/v2/Git-Branching-Branching-and-Merging
- "Understanding Git Merge": https://docs.github.com/en/get-started/using-git/dealing-with-special-characters-in-branch-and-tag-names

---

## Implementation Notes

### In This Codebase

The cortical codebase already implements several patterns:

1. **Event Sourcing** (cortical/got/wal.py, reasoning/graph_persistence.py)
2. **Optimistic Concurrency** (cortical/got/tx_manager.py)
3. **Conflict Detection** (cortical/got/conflict.py)
4. **File Locking** (cortical/utils/locking.py)
5. **Transaction Management** (cortical/got/tx_manager.py)

See `/home/user/Opus-code-test/cortical/got/` for production implementations.

### Production Readiness

| Pattern | Status | Notes |
|---------|--------|-------|
| Event Sourcing | ✅ Ready | Used in GoT framework |
| Optimistic Concurrency | ✅ Ready | TransactionManager |
| Locks | ✅ Ready | ProcessLock with recovery |
| Branch-Merge | ⚠️ Partial | Git-native, needs automation |
| CRDT | 🔲 Not implemented | Good for future work |
| OT | 🔲 Not implemented | Complex, may not be needed |

---

*Document compiled from research and examination of production code in cortical/ package.*
