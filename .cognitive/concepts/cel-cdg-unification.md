# Concept Document: CEL-CDG Storage Unification

**Status:** Draft - Awaiting Review
**Author:** Claude (AI Agent)
**Date:** 2026-01-16
**Stakeholders:** User, Future AI Agents

---

## Executive Summary

This document explores unifying CEL (Cognitive Event Lattice) storage with CDG (Cognitive Data Graph) to solve the auto-commit problem and reduce architectural complexity. The core question: **Should cognitive events be stored as CDG entities instead of in a separate file-based system?**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture](#2-current-architecture)
3. [Proposed Architecture](#3-proposed-architecture)
4. [What Changes](#4-what-changes)
5. [What Stays the Same](#5-what-stays-the-same)
6. [Risks and Concerns](#6-risks-and-concerns)
7. [Migration Path](#7-migration-path)
8. [Alternatives Considered](#8-alternatives-considered)
9. [Decision Criteria](#9-decision-criteria)
10. [Open Questions](#10-open-questions)

---

## 1. Problem Statement

### The Immediate Problem: Auto-Commit

When CognitiveMemory creates events, they are written to disk immediately but not committed to git. This causes friction:

```
Session Start
    └─► session_start() called
        └─► acknowledge_handoff() creates MetaCognition event
            └─► FileSystemEventStore.append()
                └─► Writes .cognitive/events/events/32/abc123.json
                └─► Writes .cognitive/events/heads.json

*** Files exist on disk but are uncommitted ***
*** Stop hook blocks until manual git commit ***
```

### The Deeper Problem: Two Storage Systems

We currently maintain two separate storage systems with similar patterns:

| System | Location | Purpose | Transactions |
|--------|----------|---------|--------------|
| CDG | `.got/entities/` | Tasks, Decisions, Edges | Yes (CDGTransactionManager) |
| CEL | `.cognitive/events/` | Cognitive Events | No (immediate write) |

Both systems:
- Write JSON files to disk
- Need git tracking for persistence
- Want transactional semantics
- Have similar indexing needs

**This duplication creates maintenance burden and inconsistent behavior.**

---

## 2. Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                              │
│                                                                             │
│  ┌─────────────────────────┐          ┌─────────────────────────┐          │
│  │     CognitiveMemory     │          │        GoT API          │          │
│  │                         │          │                         │          │
│  │  observe(), learn()     │          │  create_task()          │          │
│  │  intend(), recover()    │          │  complete_task()        │          │
│  └───────────┬─────────────┘          └───────────┬─────────────┘          │
│              │                                    │                         │
└──────────────│────────────────────────────────────│─────────────────────────┘
               │                                    │
               ▼                                    ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│           CEL                │    │           CDG                │
│                              │    │                              │
│  EventStore Protocol         │    │  CDGStore + TransactionMgr   │
│  └─► FileSystemEventStore    │    │  └─► Entity CRUD             │
│                              │    │  └─► ACID Transactions       │
│  Storage:                    │    │                              │
│  .cognitive/events/          │    │  Storage:                    │
│  ├── heads.json              │    │  .got/entities/              │
│  └── events/**/*.json        │    │  ├── T-*.json (Tasks)        │
│                              │    │  ├── D-*.json (Decisions)    │
│  Features:                   │    │  └── _history/               │
│  ✗ No transactions           │    │                              │
│  ✗ No git sync               │    │  Features:                   │
│  ✓ Merkle DAG                │    │  ✓ Transactions              │
│  ✓ Causal ordering           │    │  ✗ No git sync (yet)         │
│  ✓ Immutable events          │    │  ✓ Schema validation         │
│                              │    │  ✓ Indexes                   │
└──────────────────────────────┘    └──────────────────────────────┘
               │                                    │
               ▼                                    ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│      .cognitive/events/      │    │      .got/entities/          │
│      (Filesystem)            │    │      (Filesystem)            │
└──────────────────────────────┘    └──────────────────────────────┘
               │                                    │
               └────────────────┬───────────────────┘
                                │
                         *** SEPARATE ***
                         *** GIT COMMITS ***
```

### Problems with Current Architecture

1. **No Transaction Boundaries in CEL**
   - Each `append()` writes immediately
   - No way to batch related events
   - Can't rollback on failure

2. **Separate Git Workflows**
   - CEL events need separate git add/commit
   - CDG entities need separate git add/commit
   - Stop hooks catch uncommitted changes from both

3. **Duplicated Infrastructure**
   - Two file storage implementations
   - Two places to add indexing
   - Two places to add git sync

4. **Inconsistent Semantics**
   - CDG: Mutable entities with versions
   - CEL: Immutable append-only events
   - Different ID schemes, different query patterns

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                              │
│                                                                             │
│  ┌─────────────────────────┐          ┌─────────────────────────┐          │
│  │     CognitiveMemory     │          │        GoT API          │          │
│  │                         │          │                         │          │
│  │  observe(), learn()     │          │  create_task()          │          │
│  │  intend(), recover()    │          │  complete_task()        │          │
│  └───────────┬─────────────┘          └───────────┬─────────────┘          │
│              │                                    │                         │
└──────────────│────────────────────────────────────│─────────────────────────┘
               │                                    │
               ▼                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CEL LAYER                                       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      EventStore Protocol                                │ │
│  │                                                                         │ │
│  │  append(event) / get(id) / iterate() / heads() / ancestors()           │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    │                               │                        │
│                    ▼                               ▼                        │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐         │
│  │   FileSystemEventStore     │   │     CDGEventStore (NEW)     │         │
│  │   (existing, for testing)   │   │                             │         │
│  │                             │   │  Implements EventStore      │         │
│  │  Direct file writes         │   │  Uses CDGStore internally   │         │
│  │  No transactions            │   │  Inherits CDG transactions  │         │
│  └─────────────────────────────┘   └──────────────┬──────────────┘         │
│                                                   │                         │
└───────────────────────────────────────────────────│─────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CDG LAYER (Unified Storage)                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      CDGTransactionManager                              │ │
│  │                                                                         │ │
│  │  begin() ──► Transaction ──► commit() ──► Git Sync (NEW)               │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────────┐ │
│  │                           CDGStore                                       │ │
│  │                                                                          │ │
│  │  .got/entities/                                                          │ │
│  │  ├── T-001.json              (Task - mutable)                           │ │
│  │  ├── D-001.json              (Decision - mutable)                       │ │
│  │  ├── CEL-abc123.json         (Observation - IMMUTABLE)                  │ │
│  │  ├── CEL-def456.json         (Intention - IMMUTABLE)                    │ │
│  │  └── CEL-789xyz.json         (MetaCognition - IMMUTABLE)                │ │
│  │                                                                          │ │
│  │  Schema Registry: cognitive_event schema (immutable=true)               │ │
│  │  Index Manager: event_type, concepts, timestamp indexes                 │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GIT LAYER (Unified)                             │
│                                                                              │
│  On CDGTransactionManager.commit():                                          │
│      git add .got/entities/                                                  │
│      git commit -m "cdg: {entity_types affected}"                           │
│                                                                              │
│  ALL data (Tasks, Decisions, Cognitive Events) committed together           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Changes in Proposed Architecture

1. **CDGEventStore** - New adapter implementing EventStore protocol using CDG
2. **Immutable Schema** - CDG learns to handle immutable entities (no update/delete)
3. **Unified Storage** - All entities in `.got/entities/`
4. **Unified Git Sync** - Single commit point after CDG transactions

---

## 4. What Changes

### 4.1 New Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `CDGEventStore` | EventStore impl using CDG | `cortical/cel/adapters/cdg_store.py` |
| `cognitive_event` schema | Schema for CEL events | `cortical/cdg/schemas/` |
| Git sync hook | Auto-commit after tx | `cortical/cdg/transaction_manager.py` |

### 4.2 Modified Components

| Component | Change |
|-----------|--------|
| `CDGStore` | Respect `immutable=true` schema flag |
| `SchemaRegistry` | Handle immutability constraints |
| `CognitiveMemory` | Option to use CDGEventStore |

### 4.3 Storage Location

**Before:**
```
.cognitive/
└── events/
    ├── heads.json
    └── events/
        └── ab/
            └── cdef1234.json

.got/
└── entities/
    ├── T-001.json
    └── D-001.json
```

**After:**
```
.got/
└── entities/
    ├── T-001.json           (Task)
    ├── D-001.json           (Decision)
    ├── CEL-abc123.json      (Observation)
    ├── CEL-def456.json      (Intention)
    └── CEL-789xyz.json      (MetaCognition)

.cognitive/
└── (empty or deprecated)
```

### 4.4 Data Format

**Before (CEL FileSystemEventStore):**
```json
{
  "id": "abc123def456...",
  "timestamp": "2026-01-16T10:30:00Z",
  "event_type": "OBSERVATION",
  "content": {"what": "session_started"},
  "concepts": ["session", "handoff"],
  "causal_parents": ["parent123..."],
  "metadata": {"session": "xyz"}
}
```

**After (CDG Entity):**
```json
{
  "id": "CEL-abc123def456",
  "entity_type": "cognitive_event",
  "version": 1,
  "created_at": "2026-01-16T10:30:00Z",
  "modified_at": "2026-01-16T10:30:00Z",
  "properties": {
    "event_type": "OBSERVATION",
    "timestamp": "2026-01-16T10:30:00Z",
    "content": {"what": "session_started"},
    "concepts": ["session", "handoff"],
    "causal_parents": ["CEL-parent123"],
    "metadata": {"session": "xyz"}
  }
}
```

---

## 5. What Stays the Same

### 5.1 CognitiveMemory API

```python
# These all work exactly the same
memory = CognitiveMemory.open()
memory.observe("something happened")
memory.learn("problem", "solution")
memory.intend("do something")
memory.session_start()
memory.handoff("summary")
memory.recover()
```

### 5.2 CEL EventStore Protocol

```python
# All protocol methods remain unchanged
store.append(event)      # Still works
store.get(event_id)      # Still works
store.iterate()          # Still works
store.heads()            # Still works
store.ancestors(id)      # Still works
```

### 5.3 Event Semantics

- Events remain **immutable** (never modified)
- Events remain **append-only** (never deleted)
- Events maintain **causal ordering** (parent references)
- Event IDs remain **content-addressed** (Merkle hashes)

### 5.4 GoT API

```python
# GoT continues to work unchanged
api = GoTAPI.open()
api.create_task("title")
api.complete_task(task_id)
```

---

## 6. Risks and Concerns

### 6.1 High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Data migration failure** | Loss of cognitive history | Parallel run period, backup before migration |
| **Immutability violation** | CDG accidentally updates CEL event | Schema enforcement, runtime checks |
| **Performance regression** | CDG overhead for simple appends | Benchmark before/after, optimize if needed |

### 6.2 Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Merkle verification breaks** | Can't verify event integrity | Re-implement verification in CDGEventStore |
| **DAG traversal slower** | Ancestor/descendant queries slower | Build in-memory DAG on load (like current impl) |
| **ID collision** | CEL-* prefix collides with something | Reserve prefix in CDG schema |

### 6.3 Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Schema evolution** | Future CEL changes need CDG schema changes | Plan schema versioning from start |
| **Testing complexity** | More integration tests needed | Comprehensive test suite for adapter |

### 6.4 Concerns That Are NOT Risks

| Concern | Why It's Not a Risk |
|---------|---------------------|
| "Storage doubles" | No - we remove `.cognitive/events/`, not duplicate |
| "CEL loses identity" | No - CEL is still the API, just different storage backend |
| "GoT gets cluttered" | No - CEL entities have distinct prefix, can filter |

---

## 7. Migration Path

### Phase 1: Preparation (No User Impact)

1. Implement `CDGEventStore` adapter
2. Add `immutable` flag to CDG schema system
3. Create `cognitive_event` schema
4. Add git sync to CDGTransactionManager
5. Comprehensive testing

### Phase 2: Parallel Operation

1. CognitiveMemory writes to BOTH stores
   - FileSystemEventStore (existing)
   - CDGEventStore (new)
2. Verify data consistency
3. Monitor for issues

### Phase 3: Switchover

1. CognitiveMemory reads from CDGEventStore
2. FileSystemEventStore becomes write-only backup
3. Verify all queries work

### Phase 4: Cleanup

1. Remove FileSystemEventStore from CognitiveMemory
2. Archive `.cognitive/events/` (don't delete immediately)
3. Update documentation

### Rollback Plan

At any phase, can revert to FileSystemEventStore by:
1. Changing CognitiveMemory configuration
2. `.cognitive/events/` data is preserved until Phase 4

---

## 8. Alternatives Considered

### Alternative A: Keep Separate Systems, Add Transaction Layer to CEL

**Approach:** Add transactions to CEL's FileSystemEventStore without CDG integration.

**Pros:**
- Less architectural change
- CEL remains independent

**Cons:**
- Duplicates CDG's transaction infrastructure
- Still two git sync points
- More code to maintain

**Why Not Chosen:** Reinventing what CDG already does.

### Alternative B: Shared Git Layer Only

**Approach:** Keep separate storage, unify only the git commit logic.

**Pros:**
- Minimal changes
- Low risk

**Cons:**
- Still two storage systems
- Transaction coordination complex
- Doesn't solve the deeper problem

**Why Not Chosen:** Band-aid, not a solution.

### Alternative C: CEL Replaces CDG

**Approach:** Store GoT entities as CEL events.

**Pros:**
- Single storage system
- Event sourcing for everything

**Cons:**
- Massive migration effort for GoT
- GoT's mutable model doesn't fit events
- Higher risk than Option B

**Why Not Chosen:** Too disruptive, wrong direction.

### Alternative D: Do Nothing

**Approach:** Manual git commits, accept the friction.

**Pros:**
- No development effort

**Cons:**
- Ongoing friction for users/agents
- Stop hooks keep blocking
- Technical debt accumulates

**Why Not Chosen:** Problem doesn't go away.

---

## 9. Decision Criteria

### Must Have

- [ ] CognitiveMemory API unchanged
- [ ] Event immutability preserved
- [ ] Causal ordering preserved
- [ ] No data loss during migration
- [ ] Rollback possible

### Should Have

- [ ] Unified git sync (single commit point)
- [ ] Transaction support for batched events
- [ ] Performance equal or better than current

### Nice to Have

- [ ] Reduced code maintenance burden
- [ ] Unified querying across entity types
- [ ] Simplified architecture diagram

---

## 10. Open Questions

### Technical Questions

1. **How do we handle `heads.json`?**
   - Current: Separate file tracking DAG heads
   - Options: Store as special entity? Compute from data?

2. **How do we maintain the in-memory DAG?**
   - Current: Built on load from files
   - Proposed: Build on load from CDG entities with `causal_parents`

3. **Should CEL events be in a separate CDG partition?**
   - Could improve query performance
   - Adds complexity

4. **How do we handle the Merkle root verification?**
   - Current: ID is computed hash of content
   - Need: Verify stored ID matches computed hash on read

### Process Questions

5. **What's the timeline for this work?**
   - Depends on priority relative to other work

6. **Should we prototype first?**
   - Small proof-of-concept to validate assumptions

7. **Who tests the migration?**
   - Need comprehensive test plan

---

## Appendix A: Schema Definition

```python
COGNITIVE_EVENT_SCHEMA = {
    "entity_type": "cognitive_event",
    "id_prefix": "CEL-",
    "immutable": True,

    "fields": {
        "event_type": {
            "type": "string",
            "required": True,
            "indexed": True,
            "enum": [
                "OBSERVATION",
                "INTENTION",
                "FULFILLMENT",
                "INVALIDATION",
                "COMPACTION",
                "METACOGNITION",
                "MIGRATION",
                "REPAIR",
                "HEALTH_CHECK",
                "MAINTENANCE"
            ]
        },
        "timestamp": {
            "type": "string",
            "format": "datetime",
            "required": True,
            "indexed": True
        },
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "indexed": True
        },
        "content": {
            "type": "object",
            "required": True
        },
        "causal_parents": {
            "type": "array",
            "items": {"type": "string"},
            "indexed": True
        },
        "metadata": {
            "type": "object",
            "default": {}
        }
    },

    "constraints": [
        {
            "name": "immutable_version",
            "rule": "version == 1",
            "message": "Cognitive events are immutable, version must always be 1"
        }
    ]
}
```

---

## Appendix B: CDGEventStore Interface

```python
class CDGEventStore:
    """
    EventStore implementation backed by CDG.

    Maps CEL's append-only event model to CDG entities while
    preserving all CEL semantics (immutability, causal ordering,
    Merkle verification).
    """

    def __init__(
        self,
        cdg_store: CDGStore,
        tx_manager: CDGTransactionManager,
        auto_commit: bool = True,
    ):
        """
        Initialize CDG-backed event store.

        Args:
            cdg_store: CDG storage instance
            tx_manager: CDG transaction manager
            auto_commit: If True, each append commits immediately
                        If False, caller manages transactions
        """
        pass

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """
        Append an event to the store.

        Creates a CDG entity with entity_type="cognitive_event".
        Entity is immutable (version always 1, no updates allowed).

        If auto_commit=True, commits transaction and triggers git sync.
        """
        pass

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get event by ID (CEL-prefixed)."""
        pass

    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
    ) -> Iterator[CognitiveEvent]:
        """Iterate events in causal order."""
        pass

    def heads(self) -> List[MerkleRoot]:
        """Get current DAG heads (events with no children)."""
        pass

    def latest(self) -> Optional[MerkleRoot]:
        """Get most recent event."""
        pass

    def ancestors(self, event_id: str, depth: int = -1) -> Iterator[CognitiveEvent]:
        """Traverse causal ancestors."""
        pass

    def descendants(self, event_id: str) -> Iterator[CognitiveEvent]:
        """Traverse causal descendants."""
        pass

    @property
    def count(self) -> int:
        """Total event count."""
        pass

    # Transaction support (when auto_commit=False)

    def begin_transaction(self) -> CDGTransaction:
        """Begin explicit transaction."""
        pass

    def commit_transaction(self, tx: CDGTransaction, message: str = None):
        """Commit transaction and trigger git sync."""
        pass
```

---

## Appendix C: Estimated Effort

| Task | Effort | Dependencies |
|------|--------|--------------|
| Add `immutable` flag to CDG schema | 2 hours | None |
| Create `cognitive_event` schema | 1 hour | Immutable flag |
| Implement `CDGEventStore` | 8 hours | Schema |
| Add git sync to TransactionManager | 4 hours | None |
| Wire CognitiveMemory option | 2 hours | CDGEventStore |
| Unit tests | 4 hours | All above |
| Integration tests | 4 hours | All above |
| Migration script | 4 hours | CDGEventStore |
| Documentation | 2 hours | All above |
| **Total** | **~31 hours** | |

---

*End of Concept Document*
