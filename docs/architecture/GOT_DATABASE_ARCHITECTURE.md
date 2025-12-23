# GoT Database Architecture: Purpose-Built Graph Database for Multi-Agent Task Management

**Document Version:** 1.0
**Date:** 2025-12-23
**Status:** Design Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Tiers](#2-data-tiers)
3. [Client API Design](#3-client-api-design)
4. [Storage Architecture](#4-storage-architecture)
5. [Locking & Transactions](#5-locking--transactions)
6. [Foreign Key Management](#6-foreign-key-management)
7. [Indexing Strategy](#7-indexing-strategy)
8. [Query & Information Retrieval](#8-query--information-retrieval)
9. [Multi-Agent Concurrency](#9-multi-agent-concurrency)
10. [Self-Healing & Diagnostics](#10-self-healing--diagnostics)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Purpose

GoT (Graph of Thought) is a **purpose-built graph database** designed for multi-agent task management, decision tracking, and knowledge organization. Unlike general-purpose databases, GoT is optimized for:

- **Multi-agent concurrent work** with minimal conflicts
- **Event-sourced reasoning** with complete audit trails
- **Git-native storage** for version control and collaboration
- **Self-healing operations** with automatic recovery
- **Zero external dependencies** (pure Python + Git)

### 1.2 Core Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DESIGN PRINCIPLES                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. GIT-NATIVE FIRST                                                 │
│     - Storage format optimized for git merge                         │
│     - Append-only event logs prevent conflicts                       │
│     - Git history IS the database history                            │
│                                                                       │
│  2. MULTI-AGENT OPTIMIZED                                            │
│     - Session-based event files (no collisions)                      │
│     - Optimistic concurrency with conflict detection                 │
│     - Branch-per-agent isolation                                     │
│                                                                       │
│  3. SELF-CONTAINED                                                   │
│     - No PostgreSQL, Redis, or external services                     │
│     - File-based with write-ahead logging                            │
│     - Works offline, syncs via git push/pull                         │
│                                                                       │
│  4. OBSERVABLE & DEBUGGABLE                                          │
│     - Complete event log for time travel                             │
│     - Self-diagnosing health checks                                  │
│     - Human-readable JSON storage                                    │
│                                                                       │
│  5. PERFORMANCE-AWARE                                                │
│     - Tiered storage (hot/warm/cold)                                 │
│     - Write-behind caching with WAL durability                       │
│     - Progressive indexing (query-driven)                            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Main Agent   │  │ Sub-Agent A  │  │ Sub-Agent B  │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                  │                  │                      │
│         └──────────────────┴──────────────────┘                      │
│                            │                                          │
├────────────────────────────┼──────────────────────────────────────────┤
│                   CLIENT API (Fluent + Progressive)                  │
│                            │                                          │
│  ┌─────────────────────────┴──────────────────────────┐             │
│  │ GoTManager (Primary Interface)                      │             │
│  │ - create_task(), update_task(), query()            │             │
│  │ - Transactions, sessions, context managers         │             │
│  └─────────────────────────┬──────────────────────────┘             │
├────────────────────────────┼──────────────────────────────────────────┤
│                   TRANSACTION LAYER                                  │
│  ┌─────────────────────────┴──────────────────────────┐             │
│  │ TransactionManager (ACID guarantees)               │             │
│  │ - Optimistic concurrency (version checking)        │             │
│  │ - Conflict detection & retry logic                 │             │
│  └─────────────────────────┬──────────────────────────┘             │
├────────────────────────────┼──────────────────────────────────────────┤
│                   STORAGE TIER SYSTEM                                │
│  ┌─────────────────────────┴──────────────────────────┐             │
│  │  TIER 0: Identity (ID generation, reservation)     │             │
│  │  TIER 1: Critical (task.id, status, edges)         │             │
│  │  TIER 2: Important (priority, metadata, indexes)   │             │
│  │  TIER 3: Observability (metrics, diagnostics)      │             │
│  └─────────────────────────┬──────────────────────────┘             │
├────────────────────────────┼──────────────────────────────────────────┤
│                   PERSISTENCE LAYER                                  │
│  ┌─────────────────────────┴──────────────────────────┐             │
│  │ HOT:  In-memory cache (LRU, 100MB limit)           │             │
│  │ WARM: Local JSON files (.got/entities/*.json)      │             │
│  │ COLD: Git repository (.git/objects/...)            │             │
│  │ WAL:  Transaction log (.got/wal/*.jsonl)           │             │
│  └─────────────────────────┬──────────────────────────┘             │
├────────────────────────────┼──────────────────────────────────────────┤
│                   EVENT SOURCING LAYER                               │
│  ┌─────────────────────────┴──────────────────────────┐             │
│  │ Event Log (append-only, session-based)             │             │
│  │ .got/events/{timestamp}-{session}.jsonl            │             │
│  │ - Merge-friendly (no conflicts)                    │             │
│  │ - Complete audit trail                             │             │
│  │ - Time travel & replay                             │             │
│  └─────────────────────────┬──────────────────────────┘             │
├────────────────────────────┼──────────────────────────────────────────┤
│                   FILE SYSTEM + GIT                                  │
│  ┌─────────────────────────┴──────────────────────────┐             │
│  │ .got/                                               │             │
│  │ ├── entities/       (Tier 1+2 data)                │             │
│  │ ├── events/         (Event sourcing logs)          │             │
│  │ ├── wal/            (Write-ahead logs)             │             │
│  │ ├── indexes/        (Primary, secondary, bloom)    │             │
│  │ ├── snapshots/      (Compressed snapshots)         │             │
│  │ └── metrics/        (Tier 3 observability)         │             │
│  └─────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Tiers

GoT uses **four data tiers** with different consistency, durability, and performance characteristics.

### 2.1 Tier Classification

```
┌──────────────────────────────────────────────────────────────────────┐
│  TIER 0: IDENTITY (ID Generation & Reservation)                      │
├──────────────────────────────────────────────────────────────────────┤
│  Purpose:     Unique ID generation before entity materialization     │
│  Data:        Reserved IDs, sequences, UUIDs                         │
│  Consistency: Strong (must be unique across all agents)              │
│  Locking:     Pessimistic (atomic increment)                         │
│  Storage:     Warm (counter files + reservation table)               │
│  Examples:    task_id_counter.json, reservations.json               │
│                                                                        │
│  Critical Property: IDs never collide, even across agents            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  TIER 1: CRITICAL (Immutable Identity & Core State)                  │
├──────────────────────────────────────────────────────────────────────┤
│  Purpose:     Data that must NEVER be lost or corrupted              │
│  Data:        task.id, task.status, edge.type, edge.endpoints       │
│  Consistency: ACID (Serializable isolation)                          │
│  Locking:     Pessimistic (row-level locks)                          │
│  Storage:     Hot + Warm + WAL + Git                                 │
│  Durability:  WAL fsync + snapshot + git commit                      │
│  Examples:    T-123.json (id, status, created_at)                   │
│                                                                        │
│  Critical Property: Data loss = system failure                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  TIER 2: IMPORTANT (Mutable Properties & Relationships)              │
├──────────────────────────────────────────────────────────────────────┤
│  Purpose:     Data that should be preserved but can be reconstructed │
│  Data:        task.priority, task.description, edge.weight          │
│  Consistency: Snapshot isolation (optimistic concurrency)            │
│  Locking:     Optimistic (version numbers + retry)                   │
│  Storage:     Hot + Warm (WAL optional)                              │
│  Durability:  Async fsync (batched every 5s)                         │
│  Examples:    T-123.json (priority, description, tags)              │
│                                                                        │
│  Critical Property: Conflicts detected, but retries acceptable       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  TIER 3: OBSERVABILITY (Metrics, Logs, Diagnostics)                 │
├──────────────────────────────────────────────────────────────────────┤
│  Purpose:     Debugging, monitoring, analytics (lossy acceptable)    │
│  Data:        metrics, performance counters, health checks           │
│  Consistency: Eventual (best-effort)                                 │
│  Locking:     None (lock-free, append-only)                          │
│  Storage:     Session-based logs (not in git)                        │
│  Durability:  None (memory → overflow to disk)                       │
│  Examples:    session_metrics.jsonl, health_checks.log              │
│                                                                        │
│  Critical Property: High throughput (95K ops/sec), loss acceptable   │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Field Classification Example

```python
# Task entity field → tier mapping
FIELD_TIERS = {
    # TIER 0 (Identity - generated once)
    'id': 0,

    # TIER 1 (Critical - never lose)
    'status': 1,              # pending/in_progress/completed
    'entity_type': 1,         # task/decision/epic/sprint
    'created_at': 1,          # ISO timestamp
    'created_by': 1,          # Agent ID

    # TIER 2 (Important - can retry on conflict)
    'title': 2,
    'description': 2,
    'priority': 2,            # high/medium/low
    'tags': 2,                # List of tags
    'assigned_to': 2,         # Agent ID
    'sprint_id': 2,           # Foreign key

    # TIER 3 (Observability - lossy acceptable)
    'view_count': 3,
    'last_accessed': 3,
    'access_log': 3,
}
```

### 2.3 Tier-Specific Operations

| Operation | Tier 0 | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|--------|
| **Read Latency** | <1ms | <5ms | <10ms | <1ms |
| **Write Latency** | <10ms (lock) | <50ms (WAL fsync) | <10ms (no sync) | <0.1ms (buffer) |
| **Throughput** | ~1K/sec | ~100/sec | ~500/sec | ~95K/sec |
| **Conflict Rate** | <0.1% | <1% | <5% | N/A |
| **Retry Strategy** | Error (must succeed) | 3 retries + backoff | 5 retries | N/A |
| **Durability** | Immediate fsync | WAL + snapshot + git | Async flush | Best-effort |

---

## 3. Client API Design

The GoT API follows **fluent design** with **progressive disclosure** and **context managers** for transactions.

### 3.1 Fluent API Pattern

```python
# Level 1: Simple operations (most common, minimal code)
got = GoTManager()
task_id = got.create_task("Implement auth", priority="high")
got.update_task(task_id, status="in_progress")

# Level 2: Chained operations (intermediate)
task_id = (got
    .create_task("Implement auth")
    .with_priority("high")
    .with_tags(["backend", "security"])
    .assign_to("agent-a")
    .get_id())

# Level 3: Advanced (full control)
with got.transaction() as tx:
    task_id = tx.reserve_id(entity_type="task")
    tx.create_task(task_id, title="Implement auth", priority="high")
    tx.create_edge(task_id, dependency_id, edge_type="DEPENDS_ON")
    tx.commit()  # Atomic
```

### 3.2 Context Managers for Transactions

```python
class GoTManager:
    """Fluent API for Graph of Thought database."""

    def transaction(self) -> Transaction:
        """Begin a transaction with automatic rollback."""
        return Transaction(self)

    # Level 1 methods (simple)
    def create_task(self, title: str, **kwargs) -> str:
        """Create task (auto-transaction)."""
        with self.transaction() as tx:
            task_id = tx.create_task(title, **kwargs)
            return task_id

    # Level 2 methods (chainable builder)
    def create_task_builder(self, title: str) -> TaskBuilder:
        """Create task with builder pattern."""
        return TaskBuilder(self, title)

    # Level 3 methods (advanced)
    def bulk_import(self, entities: List[Dict]) -> BulkResult:
        """Bulk import with 2PC across multiple entities."""
        with self.transaction() as tx:
            for entity in entities:
                tx.create_entity(**entity)
            return tx.commit()

# Context manager implementation
class Transaction:
    def __enter__(self) -> 'Transaction':
        """Begin transaction."""
        self.tx_id = self.manager._begin_tx()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-commit or rollback."""
        if exc_type is None:
            self.manager._commit(self.tx_id)
        else:
            self.manager._rollback(self.tx_id)
        return False  # Don't suppress exceptions
```

### 3.3 Progressive Disclosure

```python
# Novice: Single method, sensible defaults
task_id = got.create_task("Fix bug")

# Intermediate: Common options exposed
task_id = got.create_task(
    "Fix bug",
    priority="high",
    assigned_to="agent-a"
)

# Advanced: Full configuration object
task_id = got.create_task(
    "Fix bug",
    priority="high",
    assigned_to="agent-a",
    config=TaskConfig(
        auto_assign=True,
        notify_on_complete=["agent-b"],
        retry_on_conflict=5,
        conflict_strategy="last_write_wins"
    )
)

# Expert: Manual transaction control
with got.transaction(isolation="serializable") as tx:
    task_id = tx.reserve_id("task")
    tx.write("task", task_id, {"title": "Fix bug", ...})
    tx.add_edge(task_id, sprint_id, "PART_OF")

    if not tx.validate():
        tx.rollback()
        raise ConflictError("Sprint was deleted")

    tx.commit()
```

---

## 4. Storage Architecture

GoT uses **three-tier storage** (hot/warm/cold) with **git as the cold tier**.

### 4.1 Storage Tier Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STORAGE TIER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  HOT TIER (Memory - LRU Cache)                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ OrderedDict (max 100MB, ~10K entities)                     │     │
│  │ - Recently accessed entities                                │     │
│  │ - Computed indexes (primary, secondary)                     │     │
│  │ - Active transaction buffers                                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓ (evict LRU)                           │
│                                                                       │
│  WARM TIER (Local Disk - JSON Files)                                │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ .got/entities/{entity_type}/{id}.json                      │     │
│  │ .got/events/{timestamp}-{session}.jsonl (event sourcing)   │     │
│  │ .got/wal/{sequence}.jsonl (write-ahead log)                │     │
│  │ .got/indexes/{index_name}.json (secondary indexes)         │     │
│  │ .got/snapshots/{timestamp}.json.gz (compressed snapshots)  │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓ (git add + commit)                    │
│                                                                       │
│  COLD TIER (Git Repository)                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ .git/objects/ (all history, compressed)                    │     │
│  │ - Full version history                                      │     │
│  │ - Shared across branches via git merge                      │     │
│  │ - Remote backup via git push                                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Read-Through + Write-Behind Pattern

```python
class TieredStorage:
    """
    Three-tier storage with automatic promotion/demotion.
    """

    def read(self, entity_id: str) -> Optional[Entity]:
        """Read-through cache pattern."""
        # Try hot (memory)
        if entity_id in self.hot_cache:
            self.hot_cache.move_to_end(entity_id)  # LRU update
            return self.hot_cache[entity_id]

        # Try warm (disk)
        path = self.warm_path / f"{entity_id}.json"
        if path.exists():
            entity = json.loads(path.read_text())
            self._promote_to_hot(entity_id, entity)
            return entity

        # Try cold (git)
        entity = self._load_from_git(entity_id)
        if entity:
            self._promote_to_hot(entity_id, entity)
        return entity

    def write(self, entity_id: str, entity: Entity):
        """Write-behind cache pattern."""
        # STEP 1: Write to WAL immediately (durability)
        self.wal.append(WALEntry(
            operation='write',
            entity_id=entity_id,
            data=entity
        ))

        # STEP 2: Update hot cache (fast)
        self._promote_to_hot(entity_id, entity)

        # STEP 3: Schedule async flush to warm
        self.write_buffer[entity_id] = entity
        self._schedule_flush()
```

### 4.3 Git as Cold Storage

```
Directory Structure:
.got/
├── entities/              # Warm tier (working set)
│   ├── tasks/
│   │   ├── T-001.json
│   │   └── T-002.json
│   ├── decisions/
│   └── sprints/
│
├── events/                # Event sourcing (append-only)
│   ├── 20251220-193726-abc123.jsonl
│   └── 20251220-201500-def456.jsonl
│
├── wal/                   # Write-ahead log (durability)
│   └── 00001.jsonl
│
├── indexes/               # Secondary indexes
│   ├── primary_task.json
│   ├── secondary_status.json
│   └── bloom_dedup.bloom
│
└── snapshots/             # Compressed snapshots
    └── 20251220-200000.json.gz

.git/                      # Cold tier (history + backup)
└── objects/               # All historical data, compressed
```

### 4.4 Sync Points and Durability

| Sync Level | Use Case | Guarantee | Performance |
|-----------|----------|-----------|-------------|
| **None** | Development, testing | Data in memory only | Fastest (1x) |
| **Loose** | Tier 2 data | Data on disk (no fsync) | Fast (1.1x) |
| **Sync** | Tier 1 data | Data flushed to disk | Medium (5-10x slower) |
| **Committed** | Critical data | Git commit + fsync | Slow (20-50x slower) |
| **Remote** | Backup required | Git push to remote | Very slow (100x+) |

---

## 5. Locking & Transactions

GoT uses **tiered locking** - different strategies for different data criticality levels.

### 5.1 Tiered Locking Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LOCK TIER MAPPING                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  TIER 0 (Identity): Pessimistic row-level locks                      │
│  ├─ ProcessLock on counter file                                      │
│  ├─ Atomic increment for sequence                                    │
│  └─ TTL-based reservation cleanup                                    │
│                                                                        │
│  TIER 1 (Critical): Pessimistic hierarchical locks                   │
│  ├─ Lock hierarchy: DATABASE → ENTITY_TYPE → RECORD_ID               │
│  ├─ Deadlock prevention via canonical ordering                       │
│  └─ Timeout: 5 seconds                                               │
│                                                                        │
│  TIER 2 (Important): Optimistic concurrency (version numbers)        │
│  ├─ Read: Capture version number                                     │
│  ├─ Write: Check version unchanged                                   │
│  ├─ Conflict: Retry with exponential backoff                         │
│  └─ Max retries: 5                                                   │
│                                                                        │
│  TIER 3 (Observability): Lock-free (content-addressable)             │
│  ├─ Session-based unique filenames                                   │
│  ├─ No coordination needed                                           │
│  └─ Append-only logs                                                 │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Hierarchical Lock Manager

```python
class HierarchicalLockManager:
    """
    Lock hierarchy prevents deadlocks via canonical ordering.

    Order: DATABASE → ENTITY_TYPE → RECORD_ID
    """

    def acquire(self, lock_path: List[str], timeout: float = 5.0) -> bool:
        """
        Acquire locks in hierarchical order.

        Args:
            lock_path: ['record_id', 'task_123'] or ['entity_type', 'tasks']
            timeout: Max wait time

        Returns:
            True if all locks acquired
        """
        # Validate canonical ordering
        valid_orders = [
            ['database'],
            ['entity_type', 'tasks'],
            ['record_id', 'task_123']
        ]

        acquired = []
        try:
            for level in lock_path:
                lock_file = self.lock_dir / f"{level}.lock"
                lock = ProcessLock(lock_file, reentrant=True)

                if not lock.acquire(timeout=timeout):
                    # Rollback
                    for prev_lock in acquired:
                        prev_lock.release()
                    return False

                acquired.append(lock)

            return True
        except Exception:
            # Cleanup on error
            for lock in acquired:
                lock.release()
            raise
```

### 5.3 Optimistic Concurrency Pattern

```python
class OptimisticTransaction:
    """
    Optimistic concurrency for Tier 2 data.
    """

    def __init__(self):
        self.read_set = {}   # entity_id → version
        self.write_set = {}  # entity_id → entity

    def read(self, entity_id: str) -> Entity:
        """Read with version tracking."""
        entity = storage.get(entity_id)
        self.read_set[entity_id] = entity.version
        return entity

    def write(self, entity_id: str, entity: Entity):
        """Buffer write for commit."""
        self.write_set[entity_id] = entity

    def commit(self) -> CommitResult:
        """Commit with conflict detection."""
        # Detect conflicts
        for entity_id in self.write_set:
            if entity_id in self.read_set:
                expected_version = self.read_set[entity_id]
                current = storage.get(entity_id)

                if current.version != expected_version:
                    return CommitResult(
                        success=False,
                        conflict=Conflict(
                            entity_id=entity_id,
                            expected=expected_version,
                            actual=current.version
                        )
                    )

        # Apply writes atomically
        new_version = storage.atomic_write_batch(self.write_set)
        return CommitResult(success=True, version=new_version)
```

---

## 6. Foreign Key Management

GoT supports both **hard references** (CASCADE) and **soft references** (orphan-tolerant).

### 6.1 ID Reservation System

```python
class IDReservation:
    """
    Reserve IDs before entity materialization.
    Prevents race conditions in multi-agent scenarios.
    """

    def reserve_id(self, entity_type: str, agent_id: str) -> str:
        """
        Reserve globally unique ID.

        Format: {PREFIX}-YYYYMMDD-HHMMSS-{8-hex}
        Example: T-20251223-093045-a1b2c3d4
        """
        with ProcessLock(self.counter_file):
            counter = self._load_counter()
            next_seq = counter.get(entity_type, 0) + 1
            counter[entity_type] = next_seq
            self._save_counter(counter)

        # Create reservation
        reserved_id = self._generate_id(entity_type, next_seq)
        reservation = {
            'id': reserved_id,
            'reserved_by': agent_id,
            'reserved_at': utc_now(),
            'status': 'reserved',
            'ttl': 3600  # 1 hour expiry
        }

        self._save_reservation(reservation)
        return reserved_id

    def materialize(self, reserved_id: str, entity_data: Dict) -> bool:
        """Convert reservation to actual entity."""
        reservation = self._load_reservation(reserved_id)

        if not reservation:
            raise ValueError(f"Reservation {reserved_id} not found")

        # Check TTL
        if time.time() > reservation['reserved_at'] + reservation['ttl']:
            return False  # Expired

        # Mark materialized
        reservation['status'] = 'materialized'
        self._save_reservation(reservation)
        return True
```

### 6.2 Hard vs Soft References

```python
# Reference strength classification
class ReferenceStrength(Enum):
    HARD = "hard"    # CASCADE DELETE
    SOFT = "soft"    # Orphan-tolerant
    WEAK = "weak"    # Denormalized cache

# Edge type → strength mapping
EDGE_STRENGTH = {
    # Hard references (cannot exist without target)
    'DEPENDS_ON': ReferenceStrength.HARD,
    'REQUIRES': ReferenceStrength.HARD,
    'PART_OF': ReferenceStrength.HARD,
    'CONTAINS': ReferenceStrength.HARD,

    # Soft references (can survive target deletion)
    'MENTIONS': ReferenceStrength.SOFT,
    'SUGGESTS': ReferenceStrength.SOFT,
    'INFLUENCES': ReferenceStrength.SOFT,
    'REFERENCES': ReferenceStrength.SOFT,

    # Weak references (denormalized snapshot)
    'RECENT_VIEW': ReferenceStrength.WEAK,
}

# CASCADE DELETE implementation
def delete_entity_cascade(entity_id: str) -> Dict[str, int]:
    """Delete entity and all hard dependents."""
    deleted_entities = 0
    deleted_edges = 0

    # Find hard dependents
    for edge in find_edges(target_id=entity_id):
        if EDGE_STRENGTH.get(edge.edge_type) == ReferenceStrength.HARD:
            # Recursively delete dependent
            result = delete_entity_cascade(edge.source_id)
            deleted_entities += result['deleted_entities']
            deleted_edges += result['deleted_edges']

    # Delete all edges
    edges = get_edges_for_entity(entity_id)
    deleted_edges += len(edges)
    for edge in edges:
        delete_edge(edge.id)

    # Delete entity
    delete_entity(entity_id)
    deleted_entities += 1

    return {
        'deleted_entities': deleted_entities,
        'deleted_edges': deleted_edges
    }

# SOFT reference cleanup (async)
def cleanup_orphaned_soft_refs(older_than_days: int = 7) -> int:
    """Clean up soft references to deleted entities."""
    cutoff = time.time() - (older_than_days * 86400)
    cleaned = 0

    for edge in find_all_edges():
        if EDGE_STRENGTH.get(edge.edge_type) != ReferenceStrength.SOFT:
            continue

        # Check if target exists
        if not entity_exists(edge.target_id):
            # Edge is old enough to clean
            if edge.created_at < cutoff:
                delete_edge(edge.id)
                cleaned += 1

    return cleaned
```

---

## 7. Indexing Strategy

GoT uses **progressive indexing** - indexes are created when query patterns justify them.

### 7.1 Index Types

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INDEX HIERARCHY                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  PRIMARY INDEX (Always present)                                       │
│  ├─ Purpose: O(1) lookup by entity ID                                │
│  ├─ Structure: {entity_id → file_path} in memory                     │
│  ├─ Cost: ~10-20% of data size in memory                             │
│  └─ Update: Incremental on add/delete                                │
│                                                                        │
│  SECONDARY INDEX (Created on demand)                                 │
│  ├─ Purpose: Filter by field (status, priority, sprint_id)           │
│  ├─ Structure: {field_value → [entity_ids]} on disk                  │
│  ├─ Cost: ~20-50% of data size                                       │
│  ├─ Update: Incremental on field change                              │
│  └─ Candidates: Fields queried 10+ times                             │
│                                                                        │
│  BLOOM FILTER (Deduplication)                                        │
│  ├─ Purpose: Quick duplicate check before lookup                     │
│  ├─ Structure: Bit array with k hash functions                       │
│  ├─ Cost: ~1-3% of data size (~2 bytes/entity)                       │
│  ├─ False positive rate: 0.1% (configurable)                         │
│  └─ Update: Append-only (rebuild weekly)                             │
│                                                                        │
│  PROVISIONAL INDEX (Query-driven)                                    │
│  ├─ Purpose: Adaptive optimization                                   │
│  ├─ Structure: Same as secondary, but temporary                      │
│  ├─ Lifecycle: Created after 10 queries, promoted if ROI positive    │
│  ├─ Eviction: Delete if unused for 7 days                            │
│  └─ Decision: Track queries_per_day and latency_improvement          │
│                                                                        │
│  INVERTED INDEX (Full-text search)                                   │
│  ├─ Purpose: Text search across entity content                       │
│  ├─ Structure: {term → [(entity_id, tf-idf), ...]}                   │
│  ├─ Cost: ~30-40% of text content                                    │
│  ├─ Update: Incremental on content change                            │
│  └─ Enabled: For fields with avg length > 50 chars                   │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Adaptive Index Manager

```python
class AdaptiveIndexManager:
    """
    Monitor query patterns and create indexes when justified.
    """

    def __init__(self):
        self.query_frequency = {}  # field → query count
        self.indexes = {}           # field → SecondaryIndex
        self.threshold = 10         # Create after 10 queries

    def query(self, field: str, value: Any) -> Set[str]:
        """Execute query, creating index if warranted."""
        self.query_frequency[field] = self.query_frequency.get(field, 0) + 1

        # Use index if available
        if field in self.indexes:
            return self.indexes[field].find(field, value)

        # Fall back to full scan
        result = self._full_scan(field, value)

        # Check if index should be created
        if self.query_frequency[field] >= self.threshold:
            if self._should_index(field):
                self._create_index(field)

        return result

    def _should_index(self, field: str) -> bool:
        """Decide if index is worth creating."""
        cardinality = self._estimate_cardinality(field)
        entity_count = self._get_entity_count()

        # Don't index high-cardinality fields (near-unique)
        if cardinality > entity_count * 0.5:
            return False

        # Don't index low-cardinality fields (<10 values)
        if cardinality < 10:
            return False

        # Estimate cost savings
        full_scan_cost = entity_count * 0.001  # ms per entity
        index_lookup_cost = cardinality * 0.01  # ms per unique value

        return index_lookup_cost < full_scan_cost * 0.5
```

---

## 8. Query & Information Retrieval

GoT supports **graph queries** (traversal) and **text queries** (search) with intelligent caching.

### 8.1 Query Types

```
Query Classification:

┌─────────────────────────────────────────────────────────────────┐
│  GRAPH TRAVERSAL QUERIES                                         │
├─────────────────────────────────────────────────────────────────┤
│  - "What depends on task X?"        → BFS forward traversal     │
│  - "What does task X depend on?"    → BFS backward traversal    │
│  - "Path from A to B?"              → Bidirectional BFS          │
│  - "All blocked tasks?"             → Filter + edge lookup       │
│  - "Critical path?"                 → Longest path (DP)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TEXT SEARCH QUERIES                                             │
├─────────────────────────────────────────────────────────────────┤
│  - "Find tasks about auth"          → TF-IDF ranking            │
│  - "Tasks mentioning OAuth tokens"  → Inverted index + phrases  │
│  - "Similar to task X"              → Semantic fingerprint       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  FACETED QUERIES (Filters)                                       │
├─────────────────────────────────────────────────────────────────┤
│  - "High priority pending tasks"    → Secondary index           │
│  - "Tasks in sprint S-17"           → Foreign key index          │
│  - "Tasks by agent A"               → Facet filter               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  HYBRID QUERIES (Combined)                                       │
├─────────────────────────────────────────────────────────────────┤
│  - "High priority auth tasks"       → Text + facet filter       │
│  - "Tasks depending on X about Y"   → Graph + text              │
│  - "Related to X in sprint S"       → Graph + foreign key       │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Query Optimizer

```python
class QueryOptimizer:
    """
    Choose optimal execution plan based on query characteristics.
    """

    def optimize(self, query_spec: Dict) -> QueryPlan:
        """
        Analyze query and select best strategy.

        Strategies:
        1. INDEX_SCAN: Scan all entities (baseline)
        2. FACET_FILTER: Filter by facets first
        3. TEXT_SEARCH: TF-IDF inverted index
        4. GRAPH_TRAVERSE: BFS from starting entity
        5. HYBRID: Combine multiple strategies
        """
        strategies = []

        # Strategy 1: Full scan (always possible)
        strategies.append(self._plan_index_scan(query_spec))

        # Strategy 2: Facet filter (if filters provided)
        if query_spec.get('filters'):
            strategies.append(self._plan_facet_filter(query_spec))

        # Strategy 3: Text search (if query text provided)
        if query_spec.get('query_text'):
            strategies.append(self._plan_text_search(query_spec))

        # Strategy 4: Graph traversal (if start entity provided)
        if query_spec.get('start_entity'):
            strategies.append(self._plan_graph_traverse(query_spec))

        # Choose lowest-cost strategy
        best = min(strategies, key=lambda p: p.estimated_cost)
        return best

    def _plan_facet_filter(self, query_spec: Dict) -> QueryPlan:
        """Estimate cost of facet filtering."""
        filters = query_spec['filters']
        total = self.stats['total_entities']

        # Estimate selectivity
        estimated_size = total
        for facet, values in filters.items():
            cardinality = self.stats.get(f'cardinality_{facet}', 10)
            selectivity = len(values) / cardinality
            estimated_size *= selectivity

        cost = sum(len(v) for v in filters.values()) + estimated_size

        return QueryPlan(
            strategy='FACET_FILTER',
            estimated_cost=cost,
            estimated_results=int(estimated_size)
        )
```

### 8.3 Query Result Caching

```python
class QueryResultCache:
    """
    LRU cache with dependency-based invalidation.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}  # key → CachedQuery
        self.entity_to_queries = {}  # entity_id → set of cache keys

    def get(self, query_spec: Dict) -> Optional[List]:
        """Get cached result if valid."""
        key = self._make_key(query_spec)

        if key not in self.cache:
            return None

        cached = self.cache[key]

        # Check TTL
        if datetime.now() > cached.expires_at:
            del self.cache[key]
            return None

        # Check dependencies (has any entity changed?)
        if not self._check_dependencies(cached):
            del self.cache[key]
            return None

        cached.hit_count += 1
        return cached.results

    def invalidate_entity(self, entity_id: str) -> int:
        """Invalidate all queries depending on entity."""
        if entity_id not in self.entity_to_queries:
            return 0

        invalidated = 0
        for cache_key in self.entity_to_queries[entity_id]:
            if cache_key in self.cache:
                del self.cache[cache_key]
                invalidated += 1

        del self.entity_to_queries[entity_id]
        return invalidated
```

---

## 9. Multi-Agent Concurrency

GoT uses **event sourcing** + **optimistic concurrency** for multi-agent coordination.

### 9.1 Event Sourcing Architecture

```
Multi-Agent Event Flow:

Agent A (main)                Agent B (feature-1)         Agent C (bug-fix)
    │                              │                            │
    ├─ create_task(T-1)            ├─ create_task(T-2)         ├─ update_task(T-1)
    │  → events/abc123.jsonl       │  → events/def456.jsonl    │  → events/ghi789.jsonl
    │                              │                            │
    ├─ update(T-1, status=...)     ├─ add_edge(T-2→T-1)        ├─ complete(T-1)
    │  → events/abc123.jsonl       │  → events/def456.jsonl    │  → events/ghi789.jsonl
    │                              │                            │
    └─ Git commit                  └─ Git commit               └─ Git commit
         ↓                              ↓                            ↓

┌────────────────────────────────────────────────────────────────────┐
│                         GIT MERGE                                   │
│  - All event files coexist (no conflicts!)                         │
│  - Replay events in timestamp order                                │
│  - Last-write-wins for property conflicts                          │
└────────────────────────────────────────────────────────────────────┘
         ↓

Merged State (deterministic):
.got/events/
├── 20251220-193726-abc123.jsonl  (Agent A's events)
├── 20251220-201500-def456.jsonl  (Agent B's events)
└── 20251220-202200-ghi789.jsonl  (Agent C's events)

Replay all → Consistent state everywhere!
```

### 9.2 Session-Based Event Files

```python
class SessionEventLog:
    """
    Session-specific event log for merge-friendliness.
    """

    def __init__(self, events_dir: Path = Path(".got/events")):
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename per session (no collisions!)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        self.session_file = self.events_dir / f"{timestamp}-{session_id}.jsonl"

    def append(self, event: Event):
        """Append event to session file (lock-free!)."""
        with open(self.session_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def rebuild_snapshot(self) -> Dict[str, Dict]:
        """Rebuild state from ALL event files."""
        state = {}

        # Read ALL session files
        for event_file in sorted(self.events_dir.glob("*.jsonl")):
            for line in event_file.read_text().splitlines():
                event = json.loads(line)
                entity_id = event['entity_id']

                if entity_id not in state:
                    state[entity_id] = {}

                # Apply event
                if event['event'].endswith('.create'):
                    state[entity_id].update(event['data'])
                elif event['event'].endswith('.update'):
                    state[entity_id].update(event['data'])
                elif event['event'].endswith('.delete'):
                    state[entity_id]['deleted'] = True

        return state
```

### 9.3 Conflict Resolution Strategies

| Conflict Type | Strategy | Example |
|---------------|----------|---------|
| **Property update** | Last-write-wins (by timestamp) | Both agents update task.priority → later timestamp wins |
| **Status transition** | State machine validation | Ensure valid transitions (pending→in_progress→completed) |
| **Edge creation** | Union (both edges coexist) | Agent A adds DEPENDS_ON, Agent B adds BLOCKS → both kept |
| **Edge deletion** | Tombstone (mark deleted, don't remove) | Agent A deletes edge, Agent B updates edge → tombstone wins |
| **Dependency cycle** | Graph validation + rejection | Detect cycles via DFS before committing transaction |
| **Concurrent task completion** | First-to-complete wins | Both agents complete task → first completion timestamp wins |
| **Sprint assignment** | CRDT Set (union) | Agent A assigns task to sprint 1, Agent B to sprint 2 → both kept |

**Merge strategy:**
```python
def merge_entity_state(base, ours, theirs):
    """Three-way merge with conflict detection."""
    merged = base.copy()
    conflicts = []

    # Properties
    for key in set(ours.keys()) | set(theirs.keys()):
        if key not in base:
            # New property
            if key in ours and key in theirs and ours[key] != theirs[key]:
                # Conflict: both added same property with different values
                if isinstance(ours[key], dict) and isinstance(theirs[key], dict):
                    merged[key], sub_conflicts = merge_entity_state({}, ours[key], theirs[key])
                    conflicts.extend(sub_conflicts)
                else:
                    # Last-write-wins
                    merged[key] = theirs[key] if theirs.get('updated_at') > ours.get('updated_at') else ours[key]
            else:
                merged[key] = theirs.get(key, ours.get(key))
        elif ours.get(key) != base[key] and theirs.get(key) != base[key]:
            # Conflict: both modified
            if ours[key] == theirs[key]:
                merged[key] = ours[key]  # Same change
            else:
                # Different changes - use timestamp
                merged[key] = theirs[key] if theirs.get('updated_at') > ours.get('updated_at') else ours[key]
                conflicts.append({
                    'key': key,
                    'base': base[key],
                    'ours': ours[key],
                    'theirs': theirs[key],
                    'resolution': merged[key]
                })
        elif ours.get(key) != base[key]:
            merged[key] = ours[key]
        elif theirs.get(key) != base[key]:
            merged[key] = theirs[key]

    return merged, conflicts
```

---

## 10. Self-Healing & Diagnostics

GoT implements continuous self-monitoring and automatic recovery to ensure data integrity in multi-agent environments.

### 10.1 Health Check Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    HEALTH CHECK LAYERS                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  L0: STRUCTURAL INTEGRITY                                │
│      ├─ File system consistency                          │
│      ├─ JSON syntax validation                           │
│      └─ Directory structure                              │
│                                                           │
│  L1: REFERENTIAL INTEGRITY                               │
│      ├─ Foreign key validation                           │
│      ├─ Edge target existence                            │
│      └─ Orphan detection                                 │
│                                                           │
│  L2: BUSINESS LOGIC INTEGRITY                            │
│      ├─ State machine validation                         │
│      ├─ Cycle detection (DEPENDS_ON, BLOCKS)             │
│      └─ Sprint capacity limits                           │
│                                                           │
│  L3: PERFORMANCE & OBSERVABILITY                         │
│      ├─ Index staleness                                  │
│      ├─ Cache hit rates                                  │
│      └─ Query performance metrics                        │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class GoTHealthCheck:
    """Multi-level health diagnostics."""

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        # L0: Structural
        results['checks']['structural'] = self._check_structural()
        if not results['checks']['structural']['passed']:
            return results  # Fatal - can't proceed

        # L1: Referential
        results['checks']['referential'] = self._check_referential()

        # L2: Business logic
        results['checks']['business_logic'] = self._check_business_logic()

        # L3: Performance
        results['checks']['performance'] = self._check_performance()

        results['overall_health'] = self._compute_health_score(results['checks'])
        return results

    def _check_structural(self) -> Dict[str, Any]:
        """Verify file system and JSON integrity."""
        issues = []

        # Check directory structure
        required_dirs = ['.got/entities', '.got/events', '.got/indexes']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"Missing directory: {dir_path}")

        # Check JSON files
        for entity_file in glob.glob('.got/entities/*.json'):
            try:
                with open(entity_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON in {entity_file}: {e}")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'severity': 'CRITICAL' if issues else 'OK'
        }

    def _check_referential(self) -> Dict[str, Any]:
        """Verify foreign key and edge integrity."""
        issues = []

        # Load all entities
        entities = self._load_all_entities()
        entity_ids = set(entities.keys())

        # Check edges
        for entity_id, entity in entities.items():
            for edge in entity.get('edges', []):
                if edge['to_id'] not in entity_ids:
                    issues.append({
                        'type': 'dangling_edge',
                        'from': entity_id,
                        'to': edge['to_id'],
                        'edge_type': edge['type']
                    })

        # Check orphans (entities with no incoming edges, except roots)
        incoming = defaultdict(int)
        for entity in entities.values():
            for edge in entity.get('edges', []):
                incoming[edge['to_id']] += 1

        root_types = {'epic', 'sprint'}
        for entity_id, entity in entities.items():
            if entity.get('type') not in root_types and incoming[entity_id] == 0:
                # Orphan detected
                issues.append({
                    'type': 'orphan',
                    'entity_id': entity_id,
                    'entity_type': entity.get('type')
                })

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'severity': 'HIGH' if issues else 'OK'
        }

    def _check_business_logic(self) -> Dict[str, Any]:
        """Verify domain constraints."""
        issues = []

        entities = self._load_all_entities()

        # Check for dependency cycles
        graph = self._build_dependency_graph(entities)
        cycles = self._detect_cycles(graph)
        for cycle in cycles:
            issues.append({
                'type': 'dependency_cycle',
                'cycle': cycle
            })

        # Check state transitions
        for entity_id, entity in entities.items():
            if entity.get('type') == 'task':
                status = entity.get('status')
                if status not in ['pending', 'in_progress', 'completed', 'blocked']:
                    issues.append({
                        'type': 'invalid_status',
                        'entity_id': entity_id,
                        'status': status
                    })

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'severity': 'MEDIUM' if issues else 'OK'
        }

    def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics."""
        metrics = {
            'index_staleness': self._check_index_staleness(),
            'cache_hit_rate': self._get_cache_hit_rate(),
            'avg_query_time_ms': self._get_avg_query_time()
        }

        warnings = []
        if metrics['index_staleness'] > 100:  # More than 100 updates since rebuild
            warnings.append(f"Index is stale ({metrics['index_staleness']} updates)")
        if metrics['cache_hit_rate'] < 0.5:
            warnings.append(f"Low cache hit rate ({metrics['cache_hit_rate']:.1%})")

        return {
            'passed': len(warnings) == 0,
            'metrics': metrics,
            'warnings': warnings,
            'severity': 'LOW' if warnings else 'OK'
        }
```

**See:** `docs/database-self-diagnostic-patterns.md` for detailed diagnostic patterns.

### 10.2 Recovery Cascade

```
┌──────────────────────────────────────────────────────────┐
│                    RECOVERY CASCADE                       │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  Level 1: REPAIR IN PLACE                                │
│           └─ Fix referential issues without data loss     │
│                                                            │
│  Level 2: REBUILD FROM EVENTS                            │
│           └─ Replay event log to reconstruct state        │
│                                                            │
│  Level 3: RESTORE FROM GIT                               │
│           └─ Checkout last known-good state              │
│                                                            │
│  Level 4: REBUILD INDEXES                                │
│           └─ Reconstruct derived data from entities       │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class GoTRecovery:
    """Automated recovery with escalation."""

    def recover(self, health_check_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recovery based on health check results."""
        recovery_log = []

        # Level 1: Repair in place
        if self._can_repair_in_place(health_check_results):
            recovery_log.append(self._repair_referential_issues())

            # Re-check health
            new_health = GoTHealthCheck().check_all()
            if new_health['overall_health'] > 0.9:
                return {'success': True, 'level': 1, 'log': recovery_log}

        # Level 2: Rebuild from events
        recovery_log.append("Attempting event log replay")
        if self._rebuild_from_events():
            new_health = GoTHealthCheck().check_all()
            if new_health['overall_health'] > 0.9:
                return {'success': True, 'level': 2, 'log': recovery_log}

        # Level 3: Restore from git
        recovery_log.append("Attempting git restore")
        if self._restore_from_git():
            new_health = GoTHealthCheck().check_all()
            if new_health['overall_health'] > 0.9:
                return {'success': True, 'level': 3, 'log': recovery_log}

        # Level 4: Rebuild indexes (last resort)
        recovery_log.append("Rebuilding all indexes")
        self._rebuild_indexes()

        return {
            'success': False,
            'level': 4,
            'log': recovery_log,
            'message': 'Manual intervention required'
        }

    def _repair_referential_issues(self) -> Dict[str, Any]:
        """Fix dangling edges and orphans."""
        repairs = []

        entities = self._load_all_entities()
        entity_ids = set(entities.keys())

        # Remove dangling edges
        for entity_id, entity in entities.items():
            original_edges = entity.get('edges', [])
            valid_edges = [e for e in original_edges if e['to_id'] in entity_ids]

            if len(valid_edges) < len(original_edges):
                entity['edges'] = valid_edges
                self._save_entity(entity_id, entity)
                repairs.append(f"Removed {len(original_edges) - len(valid_edges)} dangling edges from {entity_id}")

        return {
            'action': 'repair_referential',
            'repairs': repairs
        }

    def _rebuild_from_events(self) -> bool:
        """Replay event log to reconstruct state."""
        try:
            # Load all event files
            events = []
            for event_file in sorted(glob.glob('.got/events/*.json')):
                with open(event_file, 'r') as f:
                    events.extend(json.load(f))

            # Sort by timestamp
            events.sort(key=lambda e: e['timestamp'])

            # Replay events
            state = {}
            for event in events:
                entity_id = event['entity_id']
                if entity_id not in state:
                    state[entity_id] = {'id': entity_id}

                if event['event'].endswith('.create'):
                    state[entity_id].update(event['data'])
                elif event['event'].endswith('.update'):
                    state[entity_id].update(event['data'])
                elif event['event'].endswith('.delete'):
                    state[entity_id]['deleted'] = True

            # Save reconstructed state
            for entity_id, entity_data in state.items():
                if not entity_data.get('deleted'):
                    self._save_entity(entity_id, entity_data)

            return True
        except Exception as e:
            print(f"Event replay failed: {e}")
            return False

    def _restore_from_git(self) -> bool:
        """Restore from last known-good git commit."""
        try:
            import subprocess

            # Find last commit that modified .got/
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%H', '--', '.got/'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                # Restore .got/ from that commit
                subprocess.run(['git', 'checkout', commit_hash, '--', '.got/'])
                return True

            return False
        except Exception as e:
            print(f"Git restore failed: {e}")
            return False

    def _rebuild_indexes(self):
        """Rebuild all indexes from entity files."""
        # Clear existing indexes
        for index_file in glob.glob('.got/indexes/*.json'):
            os.remove(index_file)

        # Rebuild from entities
        entities = self._load_all_entities()

        # Primary index (id -> file)
        primary_index = {eid: f".got/entities/{eid}.json" for eid in entities.keys()}
        self._save_index('primary', primary_index)

        # Secondary indexes
        by_type = defaultdict(list)
        by_status = defaultdict(list)

        for entity_id, entity in entities.items():
            by_type[entity.get('type', 'unknown')].append(entity_id)
            by_status[entity.get('status', 'unknown')].append(entity_id)

        self._save_index('by_type', dict(by_type))
        self._save_index('by_status', dict(by_status))
```

### 10.3 Anomaly Detection

**Pattern detection:**
- **High orphan rate:** More than 10% of entities have no incoming edges
- **Cycle formation:** Dependency cycles detected in DEPENDS_ON/BLOCKS edges
- **Status regression:** Task transitions backward (completed → in_progress)
- **Rapid churn:** Entity updated more than 10 times in 1 minute
- **Index staleness:** More than 1000 updates without index rebuild

**Automatic responses:**
```python
class AnomalyDetector:
    """Detect and respond to anomalous patterns."""

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Scan for known anomaly patterns."""
        anomalies = []

        entities = self._load_all_entities()
        events = self._load_recent_events(minutes=60)

        # High orphan rate
        orphan_rate = self._compute_orphan_rate(entities)
        if orphan_rate > 0.1:
            anomalies.append({
                'type': 'high_orphan_rate',
                'severity': 'MEDIUM',
                'value': orphan_rate,
                'recommended_action': 'review_entity_creation'
            })

        # Dependency cycles
        cycles = self._detect_cycles(entities)
        if cycles:
            anomalies.append({
                'type': 'dependency_cycles',
                'severity': 'HIGH',
                'cycles': cycles,
                'recommended_action': 'break_cycles'
            })

        # Rapid churn
        churn = self._detect_rapid_churn(events)
        if churn:
            anomalies.append({
                'type': 'rapid_churn',
                'severity': 'MEDIUM',
                'entities': churn,
                'recommended_action': 'investigate_updates'
            })

        return anomalies
```

**See:** `docs/database-self-diagnostic-patterns.md` for anomaly detection patterns.

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Establish core data model and storage layer.

```
┌────────────────────────────────────────┐
│         PHASE 1: FOUNDATION            │
├────────────────────────────────────────┤
│  Week 1:                               │
│  ✓ Data tier definitions (0-3)        │
│  ✓ Entity file structure               │
│  ✓ Event log format                    │
│  ✓ Basic CRUD operations               │
│                                         │
│  Week 2:                               │
│  ✓ Hot storage (LRU cache)             │
│  ✓ Warm storage (local JSON)           │
│  ✓ Cold storage (git integration)      │
│  ✓ Write-behind caching                │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/model.py` - Entity definitions with tier annotations
- [ ] `cortical/got/storage.py` - Three-tier storage implementation
- [ ] `cortical/got/wal.py` - Event log with session-based files
- [ ] Unit tests: 90%+ coverage for storage layer
- [ ] Performance target: <10ms for hot cache reads, <50ms for warm reads

**Validation:**
```bash
pytest tests/got/test_storage.py -v
python scripts/got_utils.py validate --check-tiers
```

### Phase 2: Transactions & Locking (Weeks 3-4)

**Goal:** Enable safe concurrent access with tiered locking.

```
┌────────────────────────────────────────┐
│    PHASE 2: TRANSACTIONS & LOCKING     │
├────────────────────────────────────────┤
│  Week 3:                               │
│  ✓ Pessimistic locks (Tier 0-1)       │
│  ✓ Optimistic locks (Tier 2-3)        │
│  ✓ Transaction context manager         │
│  ✓ Deadlock detection                  │
│                                         │
│  Week 4:                               │
│  ✓ Hierarchical lock acquisition       │
│  ✓ Lock timeout and retry              │
│  ✓ Transaction abort and rollback      │
│  ✓ Concurrent test scenarios           │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/locking.py` - Tiered lock manager
- [ ] `cortical/got/transaction.py` - Transaction context with ACID guarantees
- [ ] Integration tests: Multi-agent concurrent scenarios
- [ ] Performance target: <100ms for lock acquisition, <1s for transaction commit

**Validation:**
```bash
pytest tests/got/test_transactions.py -v --concurrent
python scripts/got_utils.py stress-test --agents 5 --duration 60
```

### Phase 3: Foreign Keys & Referential Integrity (Week 5)

**Goal:** Ensure cross-entity relationships are consistent.

```
┌────────────────────────────────────────┐
│   PHASE 3: FOREIGN KEYS & INTEGRITY    │
├────────────────────────────────────────┤
│  ✓ ID reservation system               │
│  ✓ Hard reference validation           │
│  ✓ Soft reference tracking             │
│  ✓ CASCADE delete behavior             │
│  ✓ Orphan detection and cleanup        │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/foreign_keys.py` - ID reservation and validation
- [ ] `cortical/got/cascade.py` - CASCADE/RESTRICT/SET_NULL behaviors
- [ ] Migration tool: Add foreign key constraints to existing data
- [ ] Performance target: <5ms for FK validation (cached), <100ms for CASCADE delete

**Validation:**
```bash
pytest tests/got/test_foreign_keys.py -v
python scripts/got_utils.py validate --check-refs --fix-orphans
```

**See:** `docs/research-foreign-key-patterns.md` for implementation details.

### Phase 4: Indexing & Query (Weeks 6-7)

**Goal:** Fast retrieval with primary, secondary, and full-text indexes.

```
┌────────────────────────────────────────┐
│      PHASE 4: INDEXING & QUERY         │
├────────────────────────────────────────┤
│  Week 6:                               │
│  ✓ Primary index (id → file)           │
│  ✓ Secondary indexes (type, status)    │
│  ✓ Bloom filters for existence checks  │
│  ✓ Provisional indexes (query-driven)  │
│                                         │
│  Week 7:                               │
│  ✓ Inverted index (full-text)          │
│  ✓ Graph traversal (BFS/DFS)           │
│  ✓ TF-IDF scoring for text search      │
│  ✓ Result caching (LRU)                │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/indexes.py` - Primary, secondary, Bloom, provisional indexes
- [ ] `cortical/got/query.py` - Query DSL with graph traversal and text search
- [ ] `cortical/got/cache.py` - Multi-level caching (hot/warm)
- [ ] Performance target: <10ms for indexed lookups, <100ms for graph queries, <200ms for text search

**Validation:**
```bash
pytest tests/got/test_indexing.py -v --benchmark
python scripts/got_utils.py query "what blocks T-20251222-001122-a1b2" --explain
```

**See:** `docs/file-based-index-architecture.md`, `docs/graph-ir-patterns.md` for index design.

### Phase 5: Multi-Agent Concurrency (Week 8)

**Goal:** Support parallel agents with conflict resolution.

```
┌────────────────────────────────────────┐
│   PHASE 5: MULTI-AGENT CONCURRENCY     │
├────────────────────────────────────────┤
│  ✓ Session-based event files           │
│  ✓ Optimistic concurrency control      │
│  ✓ Three-way merge for entities        │
│  ✓ CRDT-inspired conflict resolution   │
│  ✓ Branch-per-agent strategy           │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/merge.py` - Three-way merge with conflict detection
- [ ] `cortical/got/crdt.py` - CRDT data types (LWW-Register, OR-Set)
- [ ] Multi-agent test scenarios (2-10 parallel agents)
- [ ] Performance target: <500ms for event merge, <2s for conflict resolution

**Validation:**
```bash
pytest tests/got/test_multi_agent.py -v --agents 10
python scripts/got_utils.py simulate-agents --count 5 --tasks 20
```

**See:** `docs/research/multi-agent-concurrency-patterns.md` for concurrency patterns.

### Phase 6: Self-Healing & Diagnostics (Week 9)

**Goal:** Continuous health monitoring with automatic recovery.

```
┌────────────────────────────────────────┐
│   PHASE 6: SELF-HEALING & DIAGNOSTICS  │
├────────────────────────────────────────┤
│  ✓ Multi-level health checks (L0-L3)  │
│  ✓ Recovery cascade (4 levels)         │
│  ✓ Anomaly detection                   │
│  ✓ Automated index rebuild             │
│  ✓ Git-based rollback                  │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/health.py` - Multi-level health checks
- [ ] `cortical/got/recovery.py` - Recovery cascade implementation
- [ ] `cortical/got/anomaly.py` - Anomaly detection and alerting
- [ ] Performance target: <1s for full health check, <10s for recovery cascade

**Validation:**
```bash
pytest tests/got/test_health.py -v
python scripts/got_utils.py health-check --fix --report
```

**See:** `docs/database-self-diagnostic-patterns.md` for diagnostic patterns.

### Phase 7: Client API & Developer Experience (Week 10)

**Goal:** Fluent, intuitive API with progressive disclosure.

```
┌────────────────────────────────────────┐
│      PHASE 7: CLIENT API & DX          │
├────────────────────────────────────────┤
│  ✓ Fluent query builder                │
│  ✓ Progressive disclosure (simple→adv) │
│  ✓ Transaction context managers        │
│  ✓ Type hints and IDE autocomplete     │
│  ✓ Comprehensive documentation         │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] `cortical/got/api.py` - Public API with fluent query builder
- [ ] Type stubs (`.pyi`) for all public interfaces
- [ ] Comprehensive examples: `examples/got_usage.py`
- [ ] API documentation: `docs/got-api-reference.md`
- [ ] Migration guide: `docs/got-migration.md`

**Validation:**
```bash
pytest tests/got/test_api.py -v --doctest
python examples/got_usage.py --all
mypy cortical/got/ --strict
```

**See:** `docs/client-api-design-patterns.md` for API design principles.

### Phase 8: Performance Optimization & Production Readiness (Weeks 11-12)

**Goal:** Optimize for production workloads and deployment.

```
┌────────────────────────────────────────┐
│      PHASE 8: PRODUCTION READINESS     │
├────────────────────────────────────────┤
│  Week 11:                              │
│  ✓ Profiling and bottleneck analysis   │
│  ✓ Cache tuning (hit rates >80%)       │
│  ✓ Index optimization                  │
│  ✓ Memory profiling                    │
│                                         │
│  Week 12:                              │
│  ✓ Load testing (1000+ entities)       │
│  ✓ Stress testing (10+ agents)         │
│  ✓ Production configuration            │
│  ✓ Monitoring and observability        │
└────────────────────────────────────────┘
```

**Deliverables:**
- [ ] Performance benchmarks: `tests/got/benchmarks/`
- [ ] Profiling reports: `docs/got-performance-analysis.md`
- [ ] Production config: `got.production.yaml`
- [ ] Monitoring dashboard: `scripts/got_dashboard.py`
- [ ] Deployment guide: `docs/got-deployment.md`

**Performance Targets:**
| Operation | Target | Measurement |
|-----------|--------|-------------|
| Hot read | <10ms | 95th percentile |
| Warm read | <50ms | 95th percentile |
| Cold read (git) | <500ms | 95th percentile |
| Transaction commit | <100ms | 95th percentile |
| Index rebuild | <5s | For 1000 entities |
| Query (indexed) | <50ms | 95th percentile |
| Query (full-text) | <200ms | 95th percentile |
| Graph traversal | <100ms | BFS/DFS, depth 5 |
| Health check | <1s | Full L0-L3 scan |
| Recovery (L1-L2) | <10s | Repair + rebuild |

**Validation:**
```bash
pytest tests/got/benchmarks/ -v --benchmark-only
python scripts/got_stress_test.py --entities 5000 --agents 20 --duration 300
python scripts/got_utils.py profile --report
```

---

## Conclusion

This architecture transforms GoT from a task tracker into a **purpose-built database** designed for:
- **Multi-agent collaboration** with event sourcing and CRDTs
- **Hierarchical data tiers** matching criticality to storage and locking strategies
- **Self-healing capabilities** with automated diagnostics and recovery
- **Developer experience** via fluent APIs and progressive disclosure
- **Production readiness** with tiered storage, indexing, and observability

The implementation roadmap provides a clear path from foundation (storage, transactions) through advanced features (multi-agent, self-healing) to production deployment.

**Next Steps:**
1. Review and approve architecture
2. Begin Phase 1 implementation
3. Iterate based on feedback from dog-fooding
4. Scale testing to validate performance targets

**Key References:**
- `docs/research-foreign-key-patterns.md` - ID reservation, referential integrity
- `docs/tiered-locking-summary.md` - Hierarchical locking patterns
- `docs/file-based-index-architecture.md` - Index design (primary, secondary, Bloom, provisional)
- `docs/client-api-design-patterns.md` - Fluent API, progressive disclosure
- `docs/database-self-diagnostic-patterns.md` - Health checks, anomaly detection
- `docs/research/multi-agent-concurrency-patterns.md` - Event sourcing, CRDTs, OT
- `docs/graph-ir-patterns.md` - Graph traversal, text search, caching
- `STORAGE_TIER_PATTERNS_RESEARCH.md` - Hot/warm/cold storage architecture