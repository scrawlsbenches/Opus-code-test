# Multi-Agent Concurrency Patterns: Quick Reference

**Implemented patterns in this codebase + recommendations for parallel Claude agents.**

## At a Glance

| Pattern | Implemented | File | Best For |
|---------|-----------|------|----------|
| **Event Sourcing** | ✅ Full | `cortical/got/wal.py`, `reasoning/graph_persistence.py` | Audit trail, time travel, merge-friendly |
| **Optimistic Concurrency** | ✅ Full | `cortical/got/tx_manager.py` | High parallelism, conflict detection |
| **File Locking** | ✅ Full | `cortical/utils/locking.py` | Critical sections, mutual exclusion |
| **Transaction Management** | ✅ Full | `cortical/got/tx_manager.py` | ACID guarantees |
| **Conflict Resolution** | ✅ Full | `cortical/got/conflict.py` | Sync conflicts between branches |
| **Branch-and-Merge** | ⚠️ Partial | Git-native | Clear isolation, long-running work |
| **CRDTs** | 🔲 Not Done | — | Automatic conflict resolution |
| **Operational Transforms** | 🔲 Not Done | — | Real-time collaborative editing |

---

## 1. Event Sourcing ✅

### Where It's Implemented

**`cortical/reasoning/graph_persistence.py`** (2000+ lines)
- `GraphWAL`: Write-ahead log for reasoning operations
- `GitAutoCommitter`: Automatic git commits with safety
- Event replay for crash recovery

**`cortical/got/wal.py`** (200+ lines)
- `WALManager`: Transaction write-ahead log
- `TransactionWALEntry`: Event representation
- Durability modes (PARANOID, BALANCED, LAZY)

**`cortical/got/transaction.py`**
- Transaction state machine (ACTIVE → PREPARING → COMMITTED/ABORTED)

### How It Works

```python
# 1. Record event BEFORE applying
wal.log_add_node("Q1", NodeType.QUESTION, "What's the solution?")

# 2. Apply changes
graph.add_node("Q1", NodeType.QUESTION, "What's the solution?")

# 3. On crash, replay WAL to recover
graph = recover_from_wal()
```

### Usage Pattern for Multi-Agent

```python
# Session 1 (Agent A)
wal_a = GraphWAL("reasoning_wal")
wal_a.log_add_node("task:T-123", NodeType.TASK, {"title": "Feature"})
wal_a.log_add_edge("task:T-123", "epic:E-1", EdgeType.CONTAINS)

# Session 2 (Agent B, different branch)
wal_b = GraphWAL("reasoning_wal")  # Same directory!
wal_b.log_add_node("task:T-456", NodeType.TASK, {"title": "Bug fix"})

# After git merge, both WAL entries coexist
# Replay all events → consistent graph on both branches
```

### Key Strength

**Perfect for git merge** - append-only logs never conflict.

---

## 2. Optimistic Concurrency ✅

### Where It's Implemented

**`cortical/got/tx_manager.py`** (328 lines)

```python
class TransactionManager:
    def commit(self, tx: Transaction) -> CommitResult:
        # 1. Detect conflicts
        conflicts = self._detect_conflicts(tx)
        if conflicts:
            return CommitResult(success=False, conflicts=conflicts)

        # 2. Apply atomically
        new_version = self.store.apply_writes(tx.write_set)

        # 3. Return success
        return CommitResult(success=True, version=new_version)

    def _detect_conflicts(self, tx: Transaction) -> List[Conflict]:
        # Check version numbers - if changed since read, conflict!
        for entity_id in tx.write_set:
            if entity_id in tx.read_set:
                expected_version = tx.read_set[entity_id]
                actual_version = self.store.read(entity_id).version
                if expected_version != actual_version:
                    return Conflict(...)
        return []
```

### Usage Pattern

```python
# Agent A
tx = manager.begin()
task = manager.read(tx, "task:T-123")  # snapshot_version = 5
task.status = "completed"
manager.write(tx, task)
result = manager.commit(tx)  # Version changed to 6? No conflict!

# Agent B (same task, but conflicting change)
tx = manager.begin()
task = manager.read(tx, "task:T-123")  # snapshot_version = 5
task.status = "deferred"  # Different change!
manager.write(tx, task)
result = manager.commit(tx)  # Version is now 6 - CONFLICT!
if not result.success:
    # Retry with fresh read
```

### Key Strength

**Low overhead, good for low-conflict scenarios** - detects conflicts instead of preventing them.

---

## 3. File Locking ✅

### Where It's Implemented

**`cortical/utils/locking.py`** (285 lines)

```python
class ProcessLock:
    """Cross-process lock via fcntl.flock()"""

    def __init__(self, lock_path: Path, reentrant: bool = True,
                 stale_timeout: float = 3600.0):
        # Stale lock detection - recovers from dead processes
        # Reentrant - same process can acquire multiple times
        # Timeout support - exponential backoff
```

### Features

- ✅ Cross-process synchronization (via fcntl on POSIX)
- ✅ Stale lock detection (dead process recovery)
- ✅ Reentrant locking (same process multiple times)
- ✅ Timeout with exponential backoff
- ✅ Context manager interface

### Usage Pattern

```python
from cortical.utils.locking import ProcessLock
from pathlib import Path

# Lock-based regions for tasks
class LockedRegion:
    def __init__(self, region_id: str):
        self.lock = ProcessLock(Path(f".got/.locks/{region_id}.lock"))

    def __enter__(self):
        if not self.lock.acquire(timeout=5.0):
            raise RuntimeError("Could not acquire lock")
        return self

    def __exit__(self, *args):
        self.lock.release()

# Usage
with LockedRegion("task:T-123"):
    task = load_task("task:T-123")
    task.status = "completed"
    save_task("task:T-123", task)
```

### Key Strength

**Simple, reliable mutual exclusion** - good for critical sections.

### Limitation

**Doesn't prevent Git merge conflicts** - only process-level mutual exclusion. Use with event sourcing for true git-merge-friendliness.

---

## 4. Transaction Management ✅

### Where It's Implemented

**`cortical/got/tx_manager.py`**
**`cortical/got/transaction.py`**
**`cortical/got/versioned_store.py`**

```python
class TransactionManager:
    def begin(self) -> Transaction:
        """Start transaction with snapshot isolation"""
        tx_id = generate_transaction_id()
        snapshot_version = self.store.current_version()
        return Transaction(id=tx_id, snapshot_version=snapshot_version)

    def read(self, tx: Transaction, entity_id: str):
        """Read within snapshot - only sees committed data"""
        entity = self.store.read_at_version(entity_id, tx.snapshot_version)
        tx.add_read(entity_id, entity.version)
        return entity

    def write(self, tx: Transaction, entity: Entity):
        """Buffer write - doesn't apply until commit"""
        tx.add_write(entity)

    def commit(self, tx: Transaction) -> CommitResult:
        """All-or-nothing commit with conflict detection"""
        with self.lock:
            # Detect conflicts
            conflicts = self._detect_conflicts(tx)
            if conflicts:
                tx.state = TransactionState.ABORTED
                return CommitResult(success=False, conflicts=conflicts)

            # Apply atomically
            new_version = self.store.apply_writes(tx.write_set)
            tx.state = TransactionState.COMMITTED
            return CommitResult(success=True, version=new_version)
```

### ACID Properties

- ✅ **Atomicity**: All writes succeed or all fail
- ✅ **Consistency**: Checksums verify data integrity
- ✅ **Isolation**: Snapshot isolation via versioning
- ✅ **Durability**: WAL + fsync (configurable)

### Usage Pattern

```python
# Multi-agent scenario: safe concurrent updates
manager = TransactionManager(Path(".got"))

# Agent A modifies task
tx_a = manager.begin()
task_a = manager.read(tx_a, "task:T-123")
task_a.status = "completed"
manager.write(tx_a, task_a)
result_a = manager.commit(tx_a)  # Success!

# Agent B modifies same task (different field)
tx_b = manager.begin()
task_b = manager.read(tx_b, "task:T-123")
task_b.owner = "agent-b"
manager.write(tx_b, task_b)
result_b = manager.commit(tx_b)  # Conflict? Depends on version
if not result_b.success:
    # Retry
    tx_b = manager.begin()
    task_b = manager.read(tx_b, "task:T-123")
    # ... retry change ...
```

### Key Strength

**ACID guarantees** - ensures data consistency under concurrent access.

---

## 5. Conflict Resolution ✅

### Where It's Implemented

**`cortical/got/conflict.py`** (248 lines)

```python
class ConflictResolver:
    """Resolve conflicts between local and remote entity versions"""

    def detect_conflicts(self, local_entities, remote_entities):
        """Find entities with version mismatches"""
        # Returns list of SyncConflict objects

    def resolve(self, conflict, local, remote, strategy):
        """Resolve using strategy"""
        if strategy == ConflictStrategy.OURS:
            return local  # Keep local
        elif strategy == ConflictStrategy.THEIRS:
            return remote  # Take remote
        elif strategy == ConflictStrategy.MERGE:
            return self._merge_entities(conflict, local, remote)

    def _merge_entities(self, conflict, local, remote):
        """Attempt field-level merge"""
        # For MERGE: can't resolve if same content field differs
        if conflict.conflict_fields:
            raise ConflictError(f"Cannot auto-merge: {conflict.conflict_fields}")

        # Otherwise: take higher version number
        return local if local.version >= remote.version else remote
```

### Conflict Strategies

| Strategy | Behavior | Use When |
|----------|----------|----------|
| `OURS` | Keep local version | Local is authoritative |
| `THEIRS` | Take remote version | Remote is newer/correct |
| `MERGE` | Combine non-conflicting changes | Safe field-level merge possible |

### Usage Pattern

```python
# After sync with remote branch
resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)

conflicts = resolver.detect_conflicts(local_entities, remote_entities)
if conflicts:
    try:
        resolved = resolver.resolve_all(conflicts, local_entities, remote_entities)
    except ConflictError as e:
        # Manual intervention needed
        print(f"Cannot auto-resolve: {e}")
```

### Key Strength

**Intelligent conflict detection** - field-level awareness, not just full-file conflicts.

---

## 6. Branch-and-Merge ⚠️

### Status

**Partially implemented** - uses Git native branching, but no automatic orchestration yet.

### Recommendation

```python
# Recommended practice for parallel agents

# Main agent creates branches for sub-agents
coordinator = BranchCoordinator(Path("."))

# Sub-agents work independently
branch_a = coordinator.create_agent_branch("agent-a")
branch_b = coordinator.create_agent_branch("agent-b")

# When ready, merge in dependency order
results = coordinator.merge_in_dependency_order()

# On conflict, use conflict resolver
if not results.get("agent-a"):
    # Merge failed - resolve manually or escalate
```

### Key Strengths

- Native Git support
- Clear isolation per agent
- Easy history tracking
- Rollback capability

### Limitation

**Still need to resolve file-level conflicts** - not suitable for high parallelism on same files.

---

## 7 & 8. CRDTs & Operational Transforms 🔲

### Status

**Not yet implemented** - these are for more advanced scenarios.

### When to Consider

| Pattern | When Needed |
|---------|------------|
| **CRDTs** | Automatic conflict resolution for all data types (sets, registers, counters) |
| **OT** | Real-time collaborative editing with rich data types |

### Why Not Yet

- Event sourcing + optimistic concurrency already handles 95% of use cases
- CRDTs add complexity for marginal benefit in single-agent-at-a-time workflow
- OT is overkill unless doing real-time editing (Claude isn't doing that)

### If You Need Them Later

See `/home/user/Opus-code-test/docs/research/multi-agent-concurrency-patterns.md` for full implementations of both patterns with examples.

---

## Recommended Architecture for Parallel Claude Agents

### Layering

```
Layer 1: Event Sourcing (Git-tracked)
└─ .got/events/*.jsonl (append-only per session)
   └─ Always merge-friendly

Layer 2: Transaction Manager (Process-level)
└─ In-memory transactions with ACID
   └─ Handles short-term conflicts

Layer 3: Lock Manager (Critical Sections)
└─ ProcessLock for mutual exclusion
   └─ Used sparingly for true critical sections

Layer 4: Branch-and-Merge (Long-running Isolation)
└─ Git branches for major features
   └─ Merge when feature complete
```

### Typical Workflow

```python
from cortical.got.tx_manager import TransactionManager
from cortical.utils.locking import ProcessLock
from pathlib import Path

# Main agent
manager = TransactionManager(Path(".got"))

# Agent A: Update task status (low conflict)
tx_a = manager.begin()
task = manager.read(tx_a, "task:T-123")
task.status = "in_progress"
manager.write(tx_a, task)
result_a = manager.commit(tx_a)  # Optimistic concurrency

# Agent B: Update task owner (critical section)
with ProcessLock(Path(".got/.locks/task-T-456.lock"), timeout=5.0):
    tx_b = manager.begin()
    task = manager.read(tx_b, "task:T-456")
    task.owner = "agent-b"
    manager.write(tx_b, task)
    result_b = manager.commit(tx_b)  # Protected by lock

# Agent C: Long-running feature work
git checkout -b agent-c/new-feature
# ... do lots of work ...
git commit -m "feat: new feature"
git push origin agent-c/new-feature
# Later: merge to main

# All work recorded in event log → merge-friendly!
```

---

## Decision Tree: Which Pattern to Use?

```
"I need to coordinate parallel agents on the same file"
│
├─ "Can agents work on DIFFERENT parts?"
│  └─ YES → Branch-and-Merge (Agent A: module auth, Agent B: module db)
│
├─ "Do agents work on SAME entity but DIFFERENT fields?"
│  └─ YES → Optimistic Concurrency (version conflict on commit)
│           + Event Sourcing (for audit trail)
│
├─ "Do I need EXACT mutual exclusion?"
│  └─ YES → File Locking (ProcessLock)
│           Use sparingly!
│
├─ "Do I need FULL AUDIT TRAIL and TIME TRAVEL?"
│  └─ YES → Event Sourcing (WAL + snapshots)
│           Already implemented!
│
├─ "Do I need AUTOMATIC CONFLICT RESOLUTION?"
│  └─ YES → Consider CRDTs (not implemented, medium effort)
│           OR use fixed LWW rules with event sourcing
│
└─ "Do I need REAL-TIME COLLABORATIVE EDITING?"
   └─ YES → Operational Transforms (very complex, don't do this)
            OR use simple polling + event log
```

---

## Quick Checklist: Multi-Agent Setup

- [x] Event log system (cortical/got/wal.py)
- [x] Transaction manager (cortical/got/tx_manager.py)
- [x] Conflict detection (cortical/got/conflict.py)
- [x] Lock manager (cortical/utils/locking.py)
- [x] Snapshot caching (cortical/reasoning/graph_persistence.py)
- [x] Recovery procedures (cortical/got/recovery.py)
- [x] Git integration (cortical/reasoning/graph_persistence.py)
- ⚠️ Agent orchestration (partial - needs coordination)
- ⚠️ Handoff protocol (partial - needs formalization)
- 🔲 CRDTs (defer unless needed)
- 🔲 Operational Transforms (defer - out of scope)

---

## File Organization

### Core Implementation Files

```
cortical/
├── got/                           # GoT framework
│   ├── transaction.py             # Transaction state machine
│   ├── tx_manager.py              # TransactionManager (ACID)
│   ├── versioned_store.py         # Versioned entity storage
│   ├── conflict.py                # Conflict resolution
│   ├── wal.py                     # Write-ahead log
│   ├── sync.py                    # Git synchronization
│   └── recovery.py                # Crash recovery
│
├── utils/
│   └── locking.py                 # ProcessLock (file-based locks)
│
└── reasoning/
    ├── graph_persistence.py       # GraphWAL, GitAutoCommitter
    └── thought_graph.py           # Graph-of-thought data structures
```

### Usage Examples

```
examples/
├── got_demo.py                    # GoT framework examples
└── graph_persistence_demo.py      # Event sourcing examples

tests/
├── unit/got/                      # Transaction tests
├── integration/test_got_transaction.py
└── behavioral/test_got_workflow.py
```

---

## Testing Your Setup

```bash
# 1. Test transaction manager
python -c "
from cortical.got.tx_manager import TransactionManager
from pathlib import Path

manager = TransactionManager(Path('/tmp/test_got'))
tx = manager.begin()
print(f'Transaction {tx.id} started')
"

# 2. Test event log
python -c "
from cortical.reasoning.graph_persistence import GraphWAL
from cortical.reasoning.graph_of_thought import NodeType

wal = GraphWAL('test_wal')
wal.log_add_node('test-1', NodeType.TASK, 'Test task')
print('Event logged successfully')
"

# 3. Test locking
python -c "
from cortical.utils.locking import ProcessLock
from pathlib import Path

lock = ProcessLock(Path('/tmp/test.lock'))
if lock.acquire(timeout=1.0):
    print('Lock acquired!')
    lock.release()
"

# 4. Run full test suite
python -m pytest tests/unit/got/ -v
python -m pytest tests/integration/test_got_transaction.py -v
```

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Using locks for everything | Only for critical sections; prefer optimistic concurrency |
| Not handling transaction retry | Implement exponential backoff on conflict |
| Ignoring stale locks | ProcessLock has recovery; use it |
| Not using event log | Essential for audit trail and recovery |
| Merging branches without conflict detection | Use ConflictResolver explicitly |
| Assuming git merge will work on binary snapshots | Use append-only event logs instead |

---

## Further Reading

1. **Full Research**: `docs/research/multi-agent-concurrency-patterns.md`
2. **Event Sourcing Design**: `docs/got-event-sourcing.md`
3. **Transaction Details**: Source code in `cortical/got/`
4. **Graph Persistence**: `cortical/reasoning/graph_persistence.py`

---

*Quick reference guide. For comprehensive analysis, see the full research document.*
