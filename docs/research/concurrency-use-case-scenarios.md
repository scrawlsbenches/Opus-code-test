# Multi-Agent Concurrency: Use Case Scenarios

**Practical patterns for common Claude agent coordination scenarios.**

---

## Scenario 1: Parallel Independent Tasks

**Multiple agents working on completely different features simultaneously.**

```
Agent A: Implement authentication
Agent B: Add database layer
Agent C: Write documentation
Agent D: Optimize performance

All working in parallel, no file overlap.
```

### Architecture

```
Branch: main
  ├── Branch: agent-a/auth
  │    └── commit, commit
  ├── Branch: agent-b/database
  │    └── commit, commit
  ├── Branch: agent-c/docs
  │    └── commit, commit
  └── Branch: agent-d/perf
       └── commit, commit
```

### Implementation

```python
from pathlib import Path
import subprocess

class ParallelAgentOrchestrator:
    """Coordinate independent parallel agents."""

    def __init__(self, repo_path: Path):
        self.repo = repo_path

    def spawn_agents(self, tasks: Dict[str, str]):
        """
        Spawn multiple agents working independently.

        Args:
            tasks: dict of {agent_id: task_description}
        """
        branches = {}

        for agent_id, task_desc in tasks.items():
            # Create isolated branch
            branch = f"agent/{agent_id}/{int(time.time())}"
            subprocess.run(
                ["git", "checkout", "-b", branch, "main"],
                cwd=self.repo,
                check=True
            )

            # Record branch for later merge
            branches[agent_id] = branch

            # Spawn agent (pseudo-code - actual implementation uses Task tool)
            # agent_process = spawn_task(agent_id, task_desc, branch)

        return branches

    def merge_all(self, branches: Dict[str, str]) -> Dict[str, bool]:
        """Merge all agent branches back to main."""
        results = {}

        # Switch to main
        subprocess.run(["git", "checkout", "main"], cwd=self.repo, check=True)
        subprocess.run(["git", "pull"], cwd=self.repo, check=True)

        # Merge each branch
        for agent_id, branch in branches.items():
            result = subprocess.run(
                ["git", "merge", "--no-ff", branch],
                cwd=self.repo
            )
            results[agent_id] = (result.returncode == 0)

            if result.returncode == 0:
                # Push successful merge
                subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=self.repo,
                    check=True
                )

        return results

# Usage
orchestrator = ParallelAgentOrchestrator(Path("."))

tasks = {
    "agent-a": "Implement user authentication module",
    "agent-b": "Create database layer with migrations",
    "agent-c": "Write API documentation",
    "agent-d": "Optimize query performance"
}

branches = orchestrator.spawn_agents(tasks)
# ... agents work ...
results = orchestrator.merge_all(branches)

print(f"Merge results: {results}")
# Output: {'agent-a': True, 'agent-b': True, 'agent-c': True, 'agent-d': True}
```

### Pattern Characteristics

✅ **Strengths:**
- Minimal conflict (working on different files)
- Clear ownership (each agent → branch)
- Easy merge (git merge --no-ff)
- Simple coordination

⚠️ **Considerations:**
- No coordination between agents
- Dependencies must be pre-satisfied
- Merge order doesn't matter (no conflicts)

### Best For

- Feature development (different modules)
- Documentation + code + tests simultaneously
- Large refactors split across modules
- Test coverage addition across codebase

---

## Scenario 2: Shared Entity with Non-Conflicting Fields

**Multiple agents updating different fields of same entity.**

```
Entity: Task T-123 with fields:
{
  "title": "Implement auth",
  "status": "pending",
  "owner": "unassigned",
  "priority": "high",
  "tags": []
}

Agent A: Updating status (pending → in_progress)
Agent B: Assigning owner (unassigned → agent-b)
Agent C: Adding tags ([auth, security])

All concurrent on same task!
```

### Architecture

```
TransactionManager with Optimistic Locking:

Agent A:
  tx_a = begin()
  task = read(tx_a, "task:T-123")  [status: pending, owner: unassigned, tags: []]
  task.status = "in_progress"
  write(tx_a, task)
  commit(tx_a)  → SUCCESS (version 1 → 2)

Agent B (concurrent):
  tx_b = begin()
  task = read(tx_b, "task:T-123")  [status: pending, owner: unassigned, tags: []]
  task.owner = "agent-b"
  write(tx_b, task)
  commit(tx_b)  → CONFLICT! Version changed (2 != 1)
             → Retry with fresh read

Agent B (retry):
  tx_b = begin()
  task = read(tx_b, "task:T-123")  [status: in_progress, owner: unassigned, tags: []]
  task.owner = "agent-b"
  write(tx_b, task)
  commit(tx_b)  → SUCCESS (version 2 → 3)
```

### Implementation

```python
from cortical.got.tx_manager import TransactionManager
from pathlib import Path
import time

class SharedEntityCoordinator:
    """Coordinate concurrent updates to same entity."""

    def __init__(self, got_dir: Path = Path(".got")):
        self.manager = TransactionManager(got_dir)

    def update_field(self, entity_id: str, field: str, value,
                     agent_id: str = "main", max_retries: int = 3):
        """
        Update single field on entity.

        Handles conflicts with exponential backoff.
        """
        retry_delay = 0.1  # 100ms initial

        for attempt in range(max_retries):
            tx = self.manager.begin()

            try:
                # Read current state
                entity = self.manager.read(tx, entity_id)
                if entity is None:
                    return False

                # Update field
                setattr(entity, field, value)
                entity.updated_by = agent_id
                entity.updated_at = datetime.utcnow().isoformat()

                # Try to commit
                self.manager.write(tx, entity)
                result = self.manager.commit(tx)

                if result.success:
                    print(f"✓ Agent {agent_id}: Updated {entity_id}.{field} = {value}")
                    return True

                # Conflict detected - retry
                print(f"⚠ Agent {agent_id}: Conflict on {entity_id}, "
                      f"retrying (attempt {attempt + 1}/{max_retries})")

                # Exponential backoff
                time.sleep(retry_delay)
                retry_delay *= 2

            except Exception as e:
                self.manager.rollback(tx)
                print(f"✗ Agent {agent_id}: Error - {e}")
                return False

        # Max retries exceeded
        print(f"✗ Agent {agent_id}: Could not update {entity_id}.{field} "
              f"after {max_retries} retries")
        return False

# Usage: Three agents updating same task
coordinator = SharedEntityCoordinator()

# Agent A updates status
coordinator.update_field("task:T-123", "status", "in_progress", agent_id="agent-a")

# Agent B updates owner (concurrently)
coordinator.update_field("task:T-123", "owner", "agent-b", agent_id="agent-b")

# Agent C updates tags (concurrently)
coordinator.update_field("task:T-123", "tags", ["urgent", "backend"], agent_id="agent-c")

# Result: All updates eventually succeed with retries
```

### Pattern Characteristics

✅ **Strengths:**
- Handles concurrent field updates
- Automatic conflict detection
- Fairness (no blocked waiting)
- Good for low-conflict scenarios

⚠️ **Considerations:**
- Requires retry logic in caller
- Exponential backoff needed to prevent thundering herd
- Not suitable for high-contention fields

### Monitoring & Tuning

```python
class OptimisticConcurrencyMetrics:
    """Track conflict rates."""

    def __init__(self):
        self.attempts = {}  # entity_id -> count
        self.conflicts = {}  # entity_id -> count
        self.retries = {}    # entity_id -> list of delays

    def record_attempt(self, entity_id: str):
        self.attempts[entity_id] = self.attempts.get(entity_id, 0) + 1

    def record_conflict(self, entity_id: str):
        self.conflicts[entity_id] = self.conflicts.get(entity_id, 0) + 1

    def get_conflict_rate(self, entity_id: str) -> float:
        """Percentage of attempts that conflicted."""
        attempts = self.attempts.get(entity_id, 0)
        conflicts = self.conflicts.get(entity_id, 0)
        return (conflicts / attempts * 100) if attempts > 0 else 0

    def alert_if_high_contention(self, threshold: float = 30.0):
        """Alert if entity has >30% conflict rate."""
        for entity_id, conflict_rate in [
            (id, self.get_conflict_rate(id))
            for id in self.attempts
        ]:
            if conflict_rate > threshold:
                print(f"⚠ HIGH CONTENTION: {entity_id} has {conflict_rate:.1f}% conflict rate")
                print(f"  Consider: Lock-based access or field-level partitioning")

# Use metrics
metrics = OptimisticConcurrencyMetrics()

# On each update...
metrics.record_attempt("task:T-123")
if conflict_detected:
    metrics.record_conflict("task:T-123")

# Periodically check
metrics.alert_if_high_contention(threshold=20.0)
```

### Best For

- Task metadata updates (status, owner, priority)
- Non-critical concurrent updates
- Fields that change independently
- Applications with bursty, not sustained, conflict

---

## Scenario 3: Ordered Operations on Shared Structure

**Multiple agents appending to shared list/queue/log.**

```
Shared Structure: Task checklist
[
  {"id": "item-1", "text": "Design", "done": true},
  {"id": "item-2", "text": "Implement", "done": false},
  {"id": "item-3", "text": "Test", "done": false}
]

Agent A: Add "Deploy" to checklist
Agent B: Mark "Implement" as done
Agent C: Add "Monitoring" to checklist
```

### Architecture

```
Append-Only Log Pattern:

Event Log (.got/events/session-123.jsonl):
{"ts": "...", "event": "checklist.item.add", "task": "T-123", "item": "item-4", "text": "Deploy"}
{"ts": "...", "event": "checklist.item.update", "task": "T-123", "item": "item-2", "done": true}
{"ts": "...", "event": "checklist.item.add", "task": "T-123", "item": "item-5", "text": "Monitoring"}

Replay order determined by timestamp (globally ordered, no conflicts!)
```

### Implementation

```python
from datetime import datetime
from pathlib import Path
import json
from typing import List

class ChecklistItem:
    """Single checklist item."""

    def __init__(self, item_id: str, text: str, done: bool = False):
        self.item_id = item_id
        self.text = text
        self.done = done

class EventSourcedChecklist:
    """Append-only checklist with event sourcing."""

    def __init__(self, task_id: str, event_log_dir: Path = Path(".got/events")):
        self.task_id = task_id
        self.event_log_dir = event_log_dir
        self.event_log_dir.mkdir(parents=True, exist_ok=True)

        # Create session-specific event file
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        self.session_file = self.event_log_dir / f"{timestamp}-{session_id}.jsonl"

        # Cached state (rebuilt from events)
        self.items: Dict[str, ChecklistItem] = {}
        self._rebuild_state()

    def _rebuild_state(self):
        """Rebuild checklist from all events."""
        self.items.clear()

        # Read all event files
        all_events = []
        for event_file in sorted(self.event_log_dir.glob("*.jsonl")):
            with open(event_file, "r") as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get("task") == self.task_id:
                            all_events.append(event)

        # Sort by timestamp (globally ordered)
        all_events.sort(key=lambda e: e["ts"])

        # Apply events
        for event in all_events:
            if event["event"] == "checklist.item.add":
                item_id = event["item"]
                self.items[item_id] = ChecklistItem(
                    item_id=item_id,
                    text=event["text"],
                    done=False
                )
            elif event["event"] == "checklist.item.update":
                item_id = event["item"]
                if item_id in self.items:
                    self.items[item_id].done = event.get("done", False)

    def add_item(self, text: str, agent_id: str = "main") -> str:
        """Add item to checklist - records event."""
        item_id = f"item-{int(time.time() * 1000)}"

        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "checklist.item.add",
            "task": self.task_id,
            "item": item_id,
            "text": text,
            "agent": agent_id
        }

        # Append to session file
        with open(self.session_file, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Refresh state
        self._rebuild_state()

        return item_id

    def mark_done(self, item_id: str, agent_id: str = "main"):
        """Mark item done - records event."""
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "checklist.item.update",
            "task": self.task_id,
            "item": item_id,
            "done": True,
            "agent": agent_id
        }

        with open(self.session_file, "a") as f:
            f.write(json.dumps(event) + "\n")

        self._rebuild_state()

    def get_items(self) -> List[ChecklistItem]:
        """Get all items in current state."""
        # Sort by ID (deterministic order)
        return sorted(self.items.values(), key=lambda x: x.item_id)

    def get_items_by_status(self) -> tuple[List[ChecklistItem], List[ChecklistItem]]:
        """Get (done, not_done) items."""
        done = [item for item in self.items.values() if item.done]
        not_done = [item for item in self.items.values() if not item.done]
        return sorted(done, key=lambda x: x.item_id), sorted(not_done, key=lambda x: x.item_id)

# Usage: Concurrent checklist updates
checklist = EventSourcedChecklist("task:T-123")

# Agent A adds items
checklist.add_item("Design architecture", agent_id="agent-a")
checklist.add_item("Implement core", agent_id="agent-a")

# Agent B marks done (concurrently)
checklist.mark_done("item-1", agent_id="agent-b")

# Agent C adds more
checklist.add_item("Write tests", agent_id="agent-c")

# Agent D marks another done
checklist.mark_done("item-2", agent_id="agent-d")

# Result: All operations recorded in timestamp order
# Same final state everywhere (after replay)!

done, pending = checklist.get_items_by_status()
print(f"Done: {[item.text for item in done]}")
print(f"Pending: {[item.text for item in pending]}")

# .got/events/ contains multiple session files:
# ├── 20251220-193726-abc123.jsonl  (Agent A's adds)
# ├── 20251220-193726-def456.jsonl  (Agent B's update + Agent C's adds)
# └── 20251220-193727-ghi789.jsonl  (Agent D's update)
#
# On git merge: all files coexist
# Replay all events in global timestamp order
# Result: consistent checklist everywhere!
```

### Pattern Characteristics

✅ **Strengths:**
- No conflicts possible (append-only)
- Causal ordering via timestamps
- Perfect for git merge
- Full audit trail

⚠️ **Considerations:**
- Event log grows over time
- Rebuild can be slow for long histories (mitigate with snapshots)
- Total ordering requires global clock (use ISO timestamp)

### Best For

- Task histories and logs
- Checklists and kanban boards
- Comment threads
- Audit trails
- Activity feeds

---

## Scenario 4: Critical Section (Mutual Exclusion)

**Only one agent can modify at a time.**

```
Critical Section: Task closure
- Update task status
- Record completion time
- Assign reward points
- Notify stakeholders

All must happen atomically, no concurrent modification.
```

### Implementation

```python
from cortical.utils.locking import ProcessLock
from pathlib import Path
import json

class CriticalSection:
    """Protect modifications with mutex lock."""

    def __init__(self, resource_id: str, lock_dir: Path = Path(".got/.locks")):
        self.resource_id = resource_id
        self.lock = ProcessLock(lock_dir / f"{resource_id}.lock", reentrant=True)

    def __enter__(self):
        if not self.lock.acquire(timeout=5.0):
            raise RuntimeError(f"Failed to acquire lock for {self.resource_id}")
        return self

    def __exit__(self, *args):
        self.lock.release()

class TaskClosureManager:
    """Atomic task closure with mutual exclusion."""

    def __init__(self, task_dir: Path = Path(".got/tasks")):
        self.task_dir = task_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)

    def close_task(self, task_id: str, completion_data: dict, agent_id: str = "main"):
        """
        Close task atomically.

        Multiple agents might try simultaneously - only one succeeds.
        """
        critical = CriticalSection(task_id)

        with critical:
            # Load task
            task_file = self.task_dir / f"{task_id}.json"
            if task_file.exists():
                task = json.loads(task_file.read_text())
            else:
                task = {"id": task_id, "status": "pending"}

            # Check if already closed
            if task.get("status") == "closed":
                print(f"Task {task_id} already closed by {task.get('closed_by')}")
                return False

            # Atomic updates
            task.update({
                "status": "closed",
                "closed_at": datetime.utcnow().isoformat(),
                "closed_by": agent_id,
                "completion_data": completion_data
            })

            # Write back atomically
            task_file.write_text(json.dumps(task, indent=2))

            print(f"✓ Task {task_id} closed by {agent_id}")
            return True

# Usage: Multiple agents racing to close task
manager = TaskClosureManager()

# Agent A
success_a = manager.close_task(
    "task:T-123",
    {"points": 100, "status": "completed"},
    agent_id="agent-a"
)

# Agent B (concurrent, but waits for A)
success_b = manager.close_task(
    "task:T-123",
    {"points": 100, "status": "completed"},
    agent_id="agent-b"
)

# Output:
# ✓ Task task:T-123 closed by agent-a
# Task task:T-123 already closed by agent-a
```

### Pattern Characteristics

✅ **Strengths:**
- Simple to understand
- Guaranteed mutual exclusion
- Built-in stale lock detection

⚠️ **Considerations:**
- Blocks other agents (serialization)
- Not suitable for frequent contention
- Requires timeout to avoid deadlock

### When to Use

- Task closure (final state change)
- Reward point allocation
- Triggering notifications
- State transitions that must be atomic

### When NOT to Use

- Read operations (no lock needed!)
- Frequent updates (causes bottleneck)
- Cascading locks (deadlock risk)

### Best Practices

```python
class LockBestPractices:
    """Guidelines for lock usage."""

    @staticmethod
    def use_locks_sparingly():
        """Lock only critical sections, not entire operations."""
        # ❌ BAD: Lock for entire task update
        with CriticalSection("task:T-123"):
            network_call()  # Long operation!
            database_write()
            file_io()

        # ✅ GOOD: Lock only state transition
        state = fetch_state("task:T-123")
        # ... modify state without lock ...
        with CriticalSection("task:T-123"):
            save_atomic(state)  # Quick!

    @staticmethod
    def acquire_locks_in_order():
        """Prevent deadlock by consistent lock ordering."""
        # If you need multiple locks, always acquire in same order

        locks = sorted(["task:T-1", "task:T-2", "task:T-3"])
        # Always: T-1 → T-2 → T-3, never T-3 → T-1
        with CriticalSection(locks[0]):
            with CriticalSection(locks[1]):
                with CriticalSection(locks[2]):
                    # Safe!
                    pass

    @staticmethod
    def set_reasonable_timeout():
        """Avoid infinite waits."""
        # 5 seconds for short operations
        # 30 seconds for medium operations
        # Longer = need to rethink design

        timeout = 5.0  # Reasonable default
        # If timeout exceeded: log error, alert, investigate

    @staticmethod
    def log_lock_contention():
        """Monitor lock performance."""
        start = time.time()
        acquired = lock.acquire(timeout=5.0)
        elapsed = time.time() - start

        if not acquired:
            logger.error(f"Lock timeout after {elapsed:.2f}s")
            # Alert: lock is hot!

        if elapsed > 1.0:
            logger.warning(f"Lock waited {elapsed:.2f}s (consider optimization)")
```

---

## Scenario 5: Hierarchical Locking (Multiple Resources)

**Acquire multiple locks safely without deadlock.**

```
Agent A: Create edge from task:T-1 to task:T-2
  - Must lock both tasks
  - Must lock edge registry

Agent B: Create edge from task:T-2 to task:T-1
  - Same locks, but different order
  - DEADLOCK RISK!
```

### Solution: Canonical Lock Ordering

```python
class HierarchicalLockManager:
    """Acquire multiple locks safely."""

    def acquire_locks(self, resource_ids: List[str], timeout: float = 5.0) -> bool:
        """
        Acquire multiple locks in canonical order.

        This ensures all agents acquire in same order → prevents deadlock.
        """
        # Sort to ensure canonical order
        sorted_ids = sorted(resource_ids)

        locks = []
        try:
            for resource_id in sorted_ids:
                lock = ProcessLock(Path(f".got/.locks/{resource_id}.lock"))
                if not lock.acquire(timeout=timeout):
                    # Rollback acquired locks
                    for acquired_lock in locks:
                        acquired_lock.release()
                    return False
                locks.append(lock)

            # All acquired successfully
            return True

        except Exception as e:
            # Rollback on any exception
            for lock in locks:
                lock.release()
            raise

    def release_locks(self, locks: List[ProcessLock]):
        """Release all locks."""
        for lock in locks:
            lock.release()

# Usage: Safe multi-resource operations
lock_mgr = HierarchicalLockManager()

# Agent A: Add edge T-1 → T-2
resources = sorted(["task:T-1", "task:T-2", "edge:registry"])
if lock_mgr.acquire_locks(resources, timeout=5.0):
    try:
        # Critical section: modify both tasks and edge registry
        add_edge_atomically("task:T-1", "task:T-2", EdgeType.DEPENDS_ON)
    finally:
        lock_mgr.release_locks(...)

# Agent B: Add edge T-2 → T-1 (concurrently)
# Will wait for Agent A, but NO DEADLOCK because same lock order!
```

### Pattern Characteristics

✅ **Strengths:**
- Deadlock-free with canonical ordering
- Works for arbitrary number of locks
- Clear ownership

⚠️ **Considerations:**
- Must sort consistently (alphanumeric)
- Rollback on partial failure
- Serializes when needed

### Best For

- Multi-entity operations (edges, relationships)
- Cascading updates
- Complex transactions requiring mutual exclusion

---

## Scenario 6: Hybrid Approach (Large Project)

**Real-world project using all patterns together.**

```
Architecture:

┌─────────────────────────────────────────────────┐
│           Main Agent (Orchestrator)              │
│  - Spawns sub-agents                            │
│  - Coordinates merges                           │
│  - Records decisions in event log               │
└─────────────────────────────────────────────────┘
           ↓ (branch-and-merge)      ↓ (event log)
    ┌─────────────────┐          ┌──────────────────┐
    │  Sub-agents     │          │  Event Log       │
    │ ─────────────   │          │  .got/events/    │
    │ Agent A: auth   │          │  - Session 1     │
    │ Agent B: db     │          │  - Session 2     │
    │ Agent C: api    │          │  - Session 3     │
    │ Agent D: ui     │          │  - ...           │
    └─────────────────┘          └──────────────────┘
           ↓
    ┌─────────────────┐
    │   GoT Manager   │
    │  ─────────────  │
    │ Transactions    │
    │ (optimistic)    │
    │                 │
    │ Locks           │
    │ (critical       │
    │  sections)      │
    │                 │
    │ Event sourcing  │
    │ (audit trail)   │
    └─────────────────┘
```

### Coordinating Large Initiative

```python
from cortical.got.tx_manager import TransactionManager
from cortical.utils.locking import ProcessLock
from pathlib import Path
import subprocess

class LargeInitiativeOrchestrator:
    """Coordinate large multi-agent project."""

    def __init__(self, repo_path: Path):
        self.repo = repo_path
        self.tx_manager = TransactionManager(self.repo / ".got")

    def begin_initiative(self, initiative_id: str, tasks: Dict[str, str]):
        """
        Start large initiative with multiple agents.

        Example: "Implement authentication"
        - Agent A: User model + database
        - Agent B: Auth middleware + JWT
        - Agent C: Tests + integration
        - Agent D: Documentation
        """
        # Record initiative in event log
        self.log_event({
            "event": "initiative.start",
            "id": initiative_id,
            "tasks": len(tasks),
            "timestamp": datetime.utcnow().isoformat()
        })

        # Create branches for each agent
        branches = {}
        for agent_id, task_desc in tasks.items():
            branch = f"initiative/{initiative_id}/{agent_id}"
            subprocess.run(
                ["git", "checkout", "-b", branch, "main"],
                cwd=self.repo,
                check=True
            )
            branches[agent_id] = branch

            print(f"Created branch: {branch}")

        return branches

    def coordinate_shared_resource(self, initiative_id: str, resource_id: str):
        """
        Coordinate shared resource access within initiative.

        Example: Shared database schema
        - Agent A: Creates user table
        - Agent B: Adds auth fields
        - Agent C: Creates indexes
        """
        # Use transaction manager for optimistic concurrency
        # or locks for critical sections

        if self.is_critical_resource(resource_id):
            # Critical: Use mutual exclusion
            with ProcessLock(Path(f".got/.locks/{resource_id}.lock"), timeout=10.0):
                # Only one agent modifying at a time
                self.modify_resource(resource_id)
        else:
            # Non-critical: Use optimistic concurrency
            self.update_with_optimistic_locking(resource_id)

    def merge_initiative_results(self, initiative_id: str, branches: Dict[str, str]) -> bool:
        """Merge all agent branches back to main."""
        # Record start
        self.log_event({
            "event": "initiative.merge.start",
            "id": initiative_id,
            "branches": list(branches.keys())
        })

        subprocess.run(["git", "checkout", "main"], cwd=self.repo, check=True)
        subprocess.run(["git", "pull"], cwd=self.repo, check=True)

        # Merge in dependency order (if any)
        failed = []
        for agent_id, branch in sorted(branches.items()):
            result = subprocess.run(
                ["git", "merge", "--no-ff", "-m", f"Merge {agent_id} work", branch],
                cwd=self.repo
            )

            if result.returncode != 0:
                failed.append((agent_id, branch))

                # Try to resolve conflicts
                if not self.resolve_merge_conflicts(agent_id, branch):
                    # Manual intervention needed
                    self.log_event({
                        "event": "initiative.merge.conflict",
                        "id": initiative_id,
                        "agent": agent_id,
                        "branch": branch
                    })
                    continue

            # Successful merge
            subprocess.run(["git", "push"], cwd=self.repo, check=True)
            self.log_event({
                "event": "initiative.merge.complete",
                "id": initiative_id,
                "agent": agent_id
            })

        if failed:
            self.log_event({
                "event": "initiative.merge.failed",
                "id": initiative_id,
                "failed_agents": [f[0] for f in failed]
            })
            return False

        self.log_event({
            "event": "initiative.complete",
            "id": initiative_id
        })
        return True

    def log_event(self, event: dict):
        """Record event in initiative log."""
        # Append to event log (append-only)
        event_file = self.repo / ".got" / "events" / "initiative.jsonl"
        event_file.parent.mkdir(parents=True, exist_ok=True)

        with open(event_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def resolve_merge_conflicts(self, agent_id: str, branch: str) -> bool:
        """Attempt to resolve merge conflicts."""
        conflicted_files = self.get_conflicted_files()

        for file_path in conflicted_files:
            if file_path.endswith(".jsonl"):
                # Event log conflicts → append both
                self.resolve_event_log_conflict(file_path)
            elif file_path.endswith(".json.gz"):
                # Snapshot conflicts → rebuild from events
                self.resolve_snapshot_conflict(file_path)
            else:
                # Code conflicts → require manual resolution
                print(f"Manual resolution needed: {file_path}")
                return False

        # Complete merge
        subprocess.run(["git", "add", "-A"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Resolve conflicts from {agent_id}"],
            cwd=self.repo,
            check=True
        )
        return True

    def get_conflicted_files(self) -> List[str]:
        """Get list of files with merge conflicts."""
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')

    def is_critical_resource(self, resource_id: str) -> bool:
        """Determine if resource needs mutual exclusion."""
        critical_patterns = ["config", "schema", "migration", "keys"]
        return any(pattern in resource_id for pattern in critical_patterns)

    def modify_resource(self, resource_id: str):
        """Modify resource (locked)."""
        # Implementation...
        pass

    def update_with_optimistic_locking(self, resource_id: str):
        """Update resource with optimistic concurrency."""
        # Implementation...
        pass

# Usage: Large authentication initiative
orchestrator = LargeInitiativeOrchestrator(Path("."))

tasks = {
    "agent-a": "Implement user model and database schema",
    "agent-b": "Implement JWT authentication middleware",
    "agent-c": "Write comprehensive tests and integration tests",
    "agent-d": "Write API documentation"
}

# Start initiative
branches = orchestrator.begin_initiative("AUTH-001", tasks)

# Agents work...
# Main agent coordinates shared resources

# When done, merge results
success = orchestrator.merge_initiative_results("AUTH-001", branches)
if success:
    print("✓ Initiative complete and merged!")
else:
    print("✗ Initiative has merge conflicts - manual review needed")
```

### Pattern Characteristics

✅ **Strengths:**
- Handles complex multi-agent scenarios
- Clear separation of concerns
- Audit trail for all decisions
- Graceful conflict resolution

⚠️ **Considerations:**
- Requires coordination logic
- Some manual resolution needed
- Complex error handling

---

## Decision Matrix by Scenario

```
┌────────────────────────────────────────────────────────────┐
│                  CHOOSE PATTERN BY SCENARIO                 │
├────────────────────────────────────────────────────────────┤
│                                                              │
│ "Agents working on DIFFERENT modules?"                      │
│  → Branch-and-Merge (clear isolation)                      │
│                                                              │
│ "Agents updating DIFFERENT FIELDS of same entity?"          │
│  → Optimistic Concurrency (transaction manager)            │
│                                                              │
│ "Agents APPENDING to shared list/log?"                      │
│  → Event Sourcing (append-only, no conflicts)              │
│                                                              │
│ "Only ONE agent allowed to modify at a time?"               │
│  → Locking (ProcessLock)                                   │
│                                                              │
│ "Need AUDIT TRAIL and DEBUGGING capability?"                │
│  → Event Sourcing (full history)                           │
│                                                              │
│ "Multiple resources with DEPENDENCY?"                       │
│  → Hierarchical Locking (canonical ordering)               │
│                                                              │
│ "Large project with MULTIPLE CONCERNS?"                     │
│  → Hybrid (all patterns together)                          │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

---

## Testing Your Scenario

```python
# Test Scenario 1: Parallel independent tasks
def test_parallel_tasks():
    orchestrator = ParallelAgentOrchestrator(Path("."))
    results = orchestrator.merge_all({
        "a": "agent/a/branch",
        "b": "agent/b/branch",
        "c": "agent/c/branch"
    })
    assert all(results.values())

# Test Scenario 2: Shared entity
def test_concurrent_field_updates():
    coordinator = SharedEntityCoordinator()

    # Three agents updating simultaneously
    assert coordinator.update_field("task:T-123", "status", "in_progress", "a")
    assert coordinator.update_field("task:T-123", "owner", "b", "b")
    assert coordinator.update_field("task:T-123", "priority", "high", "c")

# Test Scenario 3: Append-only list
def test_concurrent_checklist():
    checklist = EventSourcedChecklist("task:T-123")

    checklist.add_item("Design", "agent-a")
    checklist.mark_done("item-1", "agent-b")
    checklist.add_item("Test", "agent-c")

    done, pending = checklist.get_items_by_status()
    assert len(done) == 1
    assert len(pending) == 1

# Test Scenario 4: Mutual exclusion
def test_critical_section():
    manager = TaskClosureManager()

    success_a = manager.close_task("task:T-123", {}, "agent-a")
    success_b = manager.close_task("task:T-123", {}, "agent-b")

    assert success_a == True
    assert success_b == False  # Already closed!

# Test Scenario 5: Hierarchical locks
def test_multi_resource_locks():
    lock_mgr = HierarchicalLockManager()

    resources = ["task:T-1", "task:T-2"]
    assert lock_mgr.acquire_locks(resources, timeout=5.0)

# Test Scenario 6: Large initiative
def test_large_initiative():
    orchestrator = LargeInitiativeOrchestrator(Path("."))

    branches = orchestrator.begin_initiative("INIT-001", {
        "a": "Task A",
        "b": "Task B",
        "c": "Task C"
    })
    assert len(branches) == 3

    # ... simulate work ...

    success = orchestrator.merge_initiative_results("INIT-001", branches)
    assert success
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Lock timeout | Increase timeout or reduce critical section |
| Merge conflicts on json files | Use event sourcing (append-only logs) |
| High transaction conflict rate | Reduce lock scope or use branch-and-merge |
| Deadlock | Use hierarchical locking (canonical order) |
| Inconsistent state after merge | Rebuild from event log |
| Event log growing too large | Create snapshots periodically |

---

*Scenarios based on production patterns from cortical/ codebase.*
