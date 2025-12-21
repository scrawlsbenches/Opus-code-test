# GoT Transactional System Survey

**Purpose:** Gather feedback from sub-agents on the newly implemented Graph of Thought (GoT) transactional architecture to understand usability, identify improvements, and validate design decisions.

---

## System Overview

The GoT transactional system provides ACID-compliant task and decision tracking for AI agents working on complex projects. It enables:

- **Task Management**: Create, update, complete tasks with priorities and statuses
- **Decision Logging**: Record decisions with rationale for audit trails
- **Relationship Tracking**: Connect tasks/decisions with typed edges (DEPENDS_ON, BLOCKS, etc.)
- **Concurrent Access**: Multiple agents can work simultaneously with conflict detection
- **Crash Recovery**: Automatic recovery from interrupted operations
- **Git Synchronization**: Push/pull for team collaboration

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL API                                │
│  GoTManager, TransactionContext (auto-commit/rollback)          │
├─────────────────────────────────────────────────────────────────┤
│                    TRANSACTION LAYER                             │
│  TransactionManager (begin/commit/rollback, conflict detection) │
│  Transaction (state machine, write buffering, read tracking)    │
├─────────────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                                 │
│  VersionedStore (checksums, atomic writes, history)             │
│  WALManager (write-ahead log, crash recovery)                   │
├─────────────────────────────────────────────────────────────────┤
│                    SYNC LAYER (SEPARATE)                         │
│  SyncManager (git push/pull, blocks during active TX)           │
│  ConflictResolver (OURS/THEIRS/MERGE strategies)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Simple Task Creation
```python
from cortical.got import GoTManager

manager = GoTManager("/path/to/.got-tx")
task = manager.create_task("Implement feature X", priority="high", status="pending")
```

### Multi-Operation Transaction
```python
with manager.transaction() as tx:
    task = tx.create_task("Main task", priority="high")
    subtask = tx.create_task("Subtask", priority="medium")
    tx.add_edge(task.id, subtask.id, "CONTAINS")
# Auto-commits on success, rolls back on exception
```

### Read-Only Query
```python
with manager.transaction(read_only=True) as tx:
    task = tx.get_task("T-20251221-120000-a1b2")
    if task:
        print(f"Status: {task.status}")
```

---

## Survey Questions

### Section 1: Usability

**Q1.1** How intuitive do you find the `with manager.transaction()` context manager pattern?
- [ ] Very intuitive
- [ ] Somewhat intuitive
- [ ] Neutral
- [ ] Somewhat confusing
- [ ] Very confusing

**Q1.2** Would you prefer different naming for any of these components?
- `GoTManager` → (your suggestion)
- `TransactionContext` → (your suggestion)
- `create_task()` → (your suggestion)
- `add_edge()` → (your suggestion)

**Q1.3** What operations would you use most frequently?
- [ ] Creating tasks
- [ ] Updating task status
- [ ] Querying tasks
- [ ] Adding relationships between tasks
- [ ] Logging decisions
- [ ] Crash recovery
- [ ] Git sync

### Section 2: API Design

**Q2.1** The system uses explicit transactions. Do you prefer:
- [ ] Explicit transactions (current): `with manager.transaction() as tx:`
- [ ] Implicit auto-commit: `manager.create_task(...)` auto-commits
- [ ] Both options available

**Q2.2** Conflict resolution strategies are OURS, THEIRS, MERGE. Would you need:
- [ ] These are sufficient
- [ ] Additional strategies (describe)
- [ ] Custom conflict handler callback

**Q2.3** Task IDs are auto-generated like `T-20251221-120000-a1b2`. Do you prefer:
- [ ] Current format (timestamp + random)
- [ ] Shorter IDs (just random)
- [ ] Custom ID support
- [ ] Sequential numbers

### Section 3: Features

**Q3.1** Which features are most valuable to you?
Rate 1-5 (1=not valuable, 5=essential):
- [ ] Atomic multi-operation transactions ___
- [ ] Crash recovery ___
- [ ] Conflict detection ___
- [ ] Git synchronization ___
- [ ] Version history ___
- [ ] Checksum verification ___

**Q3.2** What features are missing that you would need?
- (free text)

**Q3.3** The system is text-only (JSON). Would you prefer:
- [ ] Text-only (current) - human readable, git-friendly
- [ ] Binary format - faster, smaller
- [ ] Both options

### Section 4: Performance & Reliability

**Q4.1** How important is crash recovery to you?
- [ ] Critical - I expect crashes
- [ ] Important - Nice to have
- [ ] Not important - Crashes are rare

**Q4.2** For multi-agent scenarios, how many concurrent agents do you expect?
- [ ] 1 (just me)
- [ ] 2-5
- [ ] 6-10
- [ ] 10+

**Q4.3** How would you handle a conflict?
- [ ] Always keep my changes (OURS)
- [ ] Always take remote changes (THEIRS)
- [ ] Review and merge manually
- [ ] Depends on the situation

### Section 5: Integration

**Q5.1** How would you integrate this with your workflow?
- [ ] Direct API calls
- [ ] CLI commands (got_utils.py)
- [ ] Custom wrapper/abstraction
- [ ] Through a skill/command

**Q5.2** What would make adoption easier?
- (free text)

### Section 6: Overall Assessment

**Q6.1** Would you use this system for your tasks?
- [ ] Definitely yes
- [ ] Probably yes
- [ ] Maybe
- [ ] Probably not
- [ ] Definitely not

**Q6.2** What is your main concern about the system?
- (free text)

**Q6.3** What is the best aspect of the system?
- (free text)

---

## Additional Comments

Please provide any other feedback, suggestions, or questions about the GoT transactional system:

(free text)

---

## Files to Review (Optional)

If you want to examine the implementation:
- `cortical/got/__init__.py` - Public API exports
- `cortical/got/api.py` - High-level API (GoTManager, TransactionContext)
- `cortical/got/types.py` - Entity types (Task, Decision, Edge)
- `examples/got_demo.py` - Comprehensive demo with 8 scenarios
- `docs/got-transactional-architecture.md` - Design document

---

*Thank you for your feedback!*
