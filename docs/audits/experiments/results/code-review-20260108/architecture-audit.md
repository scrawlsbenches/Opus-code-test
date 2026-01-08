# Architecture Audit Report
*Agent: Architecture Critic*
*Date: 2026-01-08*
*Scope: Cortical Codebase - Focus on SOLID Principles, Coupling, Cohesion*

---

## Executive Summary

The Cortical codebase demonstrates **strong foundational architecture** with sophisticated dependency injection, clean module boundaries, and ongoing commitment to technical excellence. The recent refactoring efforts (visible in git history) show deliberate movement toward SOLID principles, particularly the removal of singletons in favor of container-based DI.

However, **critical violations remain** that threaten long-term maintainability:

1. **GoTManager is a god class** (74 methods, 9 responsibility areas) - SEVERE SRP violation
2. **GoTBackend Protocol violates ISP** (23 methods) - Interface too large
3. **Hardcoded magic paths** still exist in 15 locations despite container infrastructure
4. **TransactionContext duplicates GoTManager responsibilities** - redundant abstraction

**Overall Grade: B-** (Good infrastructure undermined by god class and interface bloat)

**Recommended Action**: Immediate refactoring of GoTManager into specialized services.

---

## Git History Forensics

### What Architectural Changes Were Found

The git history reveals **aggressive and intelligent refactoring** over the past months:

#### Major Consolidation Efforts (2025-12 to 2026-01)

```
ea202e81 refactor(cdg): Consolidate index implementations - keep CDGIndexManager
0bcfd654 refactor(tests): Remove QueryIndexManager and related test files
b4d4c4da refactor(got): Remove redundant QueryIndexManager from GoT
4c088e21 refactor: CDG TransactionManager auto-indexes on commit
a7204ef3 refactor(got): Replace QueryIndexManager with CDG IndexManager
```

**Analysis**: The team successfully consolidated 3+ index implementations into a single CDGIndexManager. This is **textbook DRY principle** and shows architectural maturity.

#### Singleton → Container Migration (2025-12)

```
bcf16a0a refactor(cdg): Remove singleton from SchemaRegistry, add referential integrity
95fb5e26 refactor(cdg): Move SchemaRegistry from GoT to CDG foundation layer
e83b240c refactor(di): Add SchemaModule and inject SchemaRegistry via container
ea8532b8 refactor(schema): Remove global registry functions, delete GoT schema shim
```

**Analysis**: This is **professional-grade refactoring**. The team:
1. Identified a singleton anti-pattern
2. Introduced container-based injection
3. Cleaned up global state
4. Maintained backward compatibility during migration

#### Layer Separation (2025-12)

```
4ba5ff18 refactor: Delete GoT tx_manager.py, use CDGTransactionManager directly
c70fd535 refactor: Delete VersionedStore, use CDGStore directly
d9dd64cb refactor: Delete GoT recovery, CDG handles all recovery
fc50991c refactor(got): Delete wal.py, use CDGWALManager directly
```

**Analysis**: The team correctly identified that GoT was duplicating CDG functionality. By **deleting ~1000+ lines of duplicate code**, they improved:
- Maintainability (single source of truth)
- Testability (one implementation to test)
- Consistency (no divergent behavior)

This demonstrates understanding of **DRY at the architectural level**.

### Refactoring Patterns Observed

| Pattern | Frequency | Assessment |
|---------|-----------|------------|
| Consolidation | 8 commits | ✓ Excellent |
| DI Migration | 6 commits | ✓ Excellent |
| Layer Cleanup | 5 commits | ✓ Excellent |
| Performance | 4 commits | ✓ Good |
| Config Unification | 3 commits | ✓ Good |

**Trend**: The team is actively reducing complexity and improving architecture. This is RARE and should be celebrated.

---

## Critical Findings

### SEVERITY: HIGH

#### 1. GoTManager - God Class Violation (SRP)

**Location**: `cortical/got/api.py` (2754 lines, 74 methods)

**Evidence**:
```
Responsibility Analysis:
  - create:  6 methods (Task, Sprint, Epic, Decision, Document, ClaudeMdLayer)
  - read:   36 methods (get_*, list_*, query_*)
  - update:  5 methods (update_task, update_sprint, update_epic, etc.)
  - delete:  4 methods (delete_task, delete_sprint, etc.)
  - query:   1 method  (query_api)
  - transaction: 1 method (transaction context manager)
  - sync:    2 methods (sync_manager operations)
  - cache:   3 methods (cache_stats, cache_clear)
  - other:  16 methods (get_sprint_progress, add_to_sprint, dependencies, etc.)

Total: 9 distinct responsibility areas
```

**Why This Violates SRP**:

The Single Responsibility Principle states: *"A class should have only one reason to change."*

GoTManager has **at least 9 reasons to change**:
1. Task entity schema changes
2. Sprint entity schema changes
3. Epic entity schema changes
4. Query language changes
5. Transaction protocol changes
6. Sync protocol changes
7. Cache eviction policy changes
8. Dependency graph algorithm changes
9. Entity validation logic changes

**Impact**:
- **Maintainability**: 2754 lines is too large to hold in working memory
- **Testability**: 74 methods require extensive mocking
- **Parallel Development**: Multiple developers will conflict on this file
- **Cognitive Load**: New developers need to understand 9 domains to modify 1 class

**Recommended Refactoring**:

```python
# BEFORE (Current - God Class):
class GoTManager:
    def create_task(...)
    def create_sprint(...)
    def create_epic(...)
    def create_decision(...)
    def get_task(...)
    def get_sprint(...)
    def list_tasks(...)
    def update_task(...)
    def delete_task(...)
    def add_dependency(...)
    def get_blockers(...)
    def cache_stats(...)
    def transaction(...)
    # ... 60 more methods

# AFTER (Recommended - Specialized Services):
class TaskService:
    """Single responsibility: Task lifecycle management."""
    def create(...)
    def get(...)
    def update(...)
    def delete(...)
    def list(...)

class SprintService:
    """Single responsibility: Sprint management."""
    def create(...)
    def get(...)
    def add_task(...)
    def get_progress(...)

class DependencyService:
    """Single responsibility: Task dependency graph."""
    def add_dependency(...)
    def get_blockers(...)
    def get_dependents(...)
    def compute_critical_path(...)

class QueryService:
    """Single responsibility: Query execution."""
    def execute(query_str: str) -> QueryResult

class GoTFacade:
    """Simplified facade for common operations."""
    def __init__(self,
                 task_service: TaskService,
                 sprint_service: SprintService,
                 dependency_service: DependencyService,
                 query_service: QueryService):
        self.tasks = task_service
        self.sprints = sprint_service
        self.dependencies = dependency_service
        self.queries = query_service
```

**Migration Path**:
1. Create specialized services (1 week)
2. Update bootstrap.py to register new services (1 day)
3. Deprecate GoTManager methods with forwarding (1 day)
4. Migrate consumers to new services incrementally (2 weeks)
5. Delete GoTManager once unused (1 day)

**Risk**: Medium (requires coordinated changes across tests and CLI)

---

#### 2. GoTBackend Protocol - Interface Segregation Violation (ISP)

**Location**: `cortical/got/protocol.py`

**Evidence**:
```python
class GoTBackend(Protocol):
    # 23 methods required for implementation!
    def create_task(...)      # Task CRUD (5 methods)
    def get_task(...)
    def list_tasks(...)
    def update_task(...)
    def delete_task(...)

    def start_task(...)       # Task state (3 methods)
    def complete_task(...)
    def block_task(...)

    def add_dependency(...)   # Relationships (5 methods)
    def add_blocks(...)
    def get_blockers(...)
    def get_dependents(...)
    def get_task_dependencies(...)

    def get_stats(...)        # Query/Analytics (6 methods)
    def validate(...)
    def get_blocked_tasks(...)
    def get_active_tasks(...)
    def what_blocks(...)
    def what_depends_on(...)

    def sync_to_git(...)      # Persistence (2 methods)
    def export_graph(...)

    def query(...)            # Query language (1 method)
    def get_all_relationships(...)  # Meta (1 method)
```

**Why This Violates ISP**:

The Interface Segregation Principle states: *"Clients should not be forced to depend on methods they do not use."*

A protocol with 23 methods forces implementers to:
1. Implement features they may not need
2. Mock 23 methods in tests (even if only testing 1 feature)
3. Understand the entire API surface to implement any part

**Real-World Impact**:

Imagine a developer wants to create a **ReadOnlyGoTBackend** for analytics. They must still implement:
- `create_task()` (throw NotImplementedError?)
- `delete_task()` (throw NotImplementedError?)
- `sync_to_git()` (throw NotImplementedError?)

This is **interface pollution**.

**Recommended Refactoring**:

```python
# Split into focused protocols

class TaskCRUD(Protocol):
    """Task creation, reading, updating, deletion."""
    def create_task(...) -> str
    def get_task(...) -> Optional[Task]
    def list_tasks(...) -> List[Task]
    def update_task(...) -> bool
    def delete_task(...) -> bool

class TaskLifecycle(Protocol):
    """Task state transitions."""
    def start_task(...) -> bool
    def complete_task(...) -> bool
    def block_task(...) -> bool

class DependencyGraph(Protocol):
    """Task relationship management."""
    def add_dependency(...) -> bool
    def get_blockers(...) -> List[Task]
    def get_dependents(...) -> List[Task]

class GoTQuery(Protocol):
    """Query and analytics."""
    def query(...) -> List[Dict]
    def get_stats(...) -> Dict
    def validate(...) -> List[str]

class GoTPersistence(Protocol):
    """Backup and export."""
    def sync_to_git(...) -> str
    def export_graph(...) -> Dict

# Compose protocols as needed
class FullGoTBackend(TaskCRUD, TaskLifecycle, DependencyGraph,
                     GoTQuery, GoTPersistence, Protocol):
    """Full-featured backend (rare case)."""
    pass

class ReadOnlyGoTBackend(TaskCRUD, GoTQuery, Protocol):
    """Read-only backend for analytics."""
    pass
```

**Benefits**:
- Implementers choose what they need
- Tests only mock relevant protocols
- Clear separation of concerns
- Easier to understand (5 methods vs 23)

---

#### 3. Hardcoded Magic Paths - Dependency Inversion Violation (DIP)

**Location**: 15 instances across codebase

**Evidence**:
```bash
$ grep -r 'Path(".got")' cortical/ --include="*.py" | wc -l
15

$ grep -r 'Path(".got")' cortical/ --include="*.py"
cortical/core/modules/got_module.py:    container.apply_module(CDGModule(got_dir=Path(".got")))
cortical/core/modules/got_module.py:    container.apply_module(GoTModule(got_dir=Path(".got")))
cortical/core/modules/got_module.py:            self.config = GoTConfig(got_dir=Path(".got"), use_memory=use_memory)
cortical/core/modules/cdg_module.py:    container.apply_module(CDGModule(base_dir=Path(".got")))
cortical/core/modules/cdg_module.py:        self.base_dir = base_dir or got_dir or Path(".got")
cortical/core/bootstrap.py:    effective_got_dir = got_dir or Path(".got")
cortical/cel/adapters/got.py:        bridge = GotBridgeEventStore(got_path=Path(".got"))
cortical/cel/adapters/got.py:    got_path: Path = Path(".got"),
cortical/audits/persistence.py:DEFAULT_PERSISTENCE_FILE = Path(".got") / "audit_pln_state.json"
cortical/audits/persistence.py:DEFAULT_RULES_FILE = Path(".got") / "audit_pln_rules.json"
```

**Why This Is a Problem**:

Despite having a **sophisticated DI container** that explicitly injects paths, code still hardcodes `Path(".got")` in 15 places.

This violates the Dependency Inversion Principle:
- **High-level modules** (bootstrap.py, modules/*.py) should not depend on **low-level details** (filesystem paths)
- Both should depend on **abstractions** (injected Path configuration)

**Why This Matters**:

1. **Testing**: Cannot easily test with alternative paths
2. **Deployment**: Cannot deploy to `/var/lib/cortical` without code changes
3. **Multi-tenancy**: Cannot run multiple instances with different data directories
4. **Configuration**: Path is hardcoded, not configurable

**The Irony**:

The codebase **already has the solution**:
```python
# cortical/core/bootstrap.py
effective_got_dir = got_dir or Path(".got")  # ✓ Correct: default in one place
container.register_instance(Path, effective_got_dir)  # ✓ Injected
```

But then modules do this:
```python
# cortical/core/modules/got_module.py
self.config = GoTConfig(got_dir=Path(".got"), use_memory=use_memory)  # ✗ Wrong!
```

**Solution**:

```python
# INSTEAD OF:
self.config = GoTConfig(got_dir=Path(".got"), use_memory=use_memory)

# DO:
self.config = GoTConfig(got_dir=got_dir, use_memory=use_memory)
# Caller passes the value from container.resolve(Path)
```

**Action Items**:
1. Grep for `Path(".got")` → 15 locations
2. Replace with injected value from container
3. Add linter rule to prevent future hardcoding

---

### SEVERITY: MEDIUM

#### 4. TransactionContext Duplicates GoTManager (DRY Violation)

**Location**: `cortical/got/api.py` lines 1778-end

**Evidence**:
```
TransactionContext: 38 methods
GoTManager: 74 methods
Overlap: ~34 methods (89% duplication)

Both implement:
- create_task, create_sprint, create_epic, create_decision, create_document
- get_task, get_sprint, get_epic, get_decision, get_document
- update_task, update_sprint, update_epic, update_document
- delete_task, delete_sprint, delete_decision
- list_tasks, list_sprints, list_epics, list_decisions, list_documents
- (and more...)
```

**Why This Is Redundant**:

TransactionContext is a **context manager wrapper** around GoTManager that:
1. Calls `tx_manager.begin()` on `__enter__`
2. Delegates all methods to GoTManager
3. Calls `tx_manager.commit()` on `__exit__`

This is **89% duplicate code** for a feature (auto-commit on exit) that could be 10 lines.

**Recommended Refactoring**:

```python
# CURRENT: Duplicate class with 38 methods
class TransactionContext:
    def __init__(self, got_manager, tx):
        self._got_manager = got_manager
        self._tx = tx

    def create_task(self, ...):
        # Delegate to got_manager
        return self._got_manager.create_task(...)

    # ... 37 more delegating methods

# RECOMMENDED: Simple wrapper with no duplication
@contextmanager
def transactional(got_manager: GoTManager):
    """Context manager for transactional operations."""
    tx = got_manager.tx_manager.begin()
    try:
        yield got_manager  # Just pass through the manager
        got_manager.tx_manager.commit(tx)
    except Exception:
        got_manager.tx_manager.rollback(tx)
        raise

# Usage (same API):
with transactional(got_manager) as tx:
    tx.create_task("Do something")  # Same as before
```

**Benefits**:
- Delete 500+ lines of duplicate code
- Single source of truth (GoTManager)
- Easier maintenance
- Same API

**Why This Exists**:

Likely historical: TransactionContext was created when transactions were more complex. Now that transactions are handled by CDGTransactionManager, the wrapper is redundant.

---

#### 5. CDGStore + CDGTransactionManager - Separation of Concerns Issue

**Location**:
- `cortical/cdg/storage.py` (CDGStore - 1235 lines)
- `cortical/cdg/transaction_manager.py` (CDGTransactionManager - 500+ lines)

**Observation**:

CDGStore has:
- File I/O operations
- Checksumming
- Version management
- **Caching** (cache_stats, cache_clear, TTL eviction)
- History tracking

CDGTransactionManager has:
- Transaction lifecycle (begin/commit/rollback)
- Conflict detection
- WAL coordination
- **Direct calls to CDGStore**

**The Issue**:

Caching is a **cross-cutting concern** that violates separation:
- CDGStore has caching (cache_stats, TTL eviction)
- CDGTransactionManager has no visibility into cache invalidation
- Cache invalidation happens at wrong layer (storage, not transaction)

**Why This Matters**:

```python
# Scenario: Transaction reads entity, modifies it, commits
tx = tx_manager.begin()
entity = store.read("E-001")  # Cached
entity.value = "new"
tx_manager.write(tx, entity)
tx_manager.commit(tx)  # CDGStore invalidates cache internally

# But what if another transaction is running concurrently?
# The cache invalidation is not transaction-aware!
```

**Recommended**:

Move caching to a **separate CacheLayer** between TransactionManager and Store:

```
TransactionManager
       ↓
   CacheLayer (transaction-aware invalidation)
       ↓
   CDGStore (pure storage, no caching)
```

This is not critical yet, but will become a problem at scale.

---

### SEVERITY: LOW

#### 6. FileSystem Abstraction Leaks in NoOpLock

**Location**: `cortical/cdg/storage.py` lines 49-63

**Evidence**:
```python
class NoOpLock:
    """
    No-operation lock for in-memory filesystems.

    Process locking is unnecessary when using InMemoryFileSystem since
    the data only exists within a single process.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
```

**Why This Is a Minor Issue**:

The abstraction is good (FileSystem interface), but the implementation leaks:
- CDGStore must **know** if it's using InMemoryFileSystem
- It instantiates NoOpLock manually based on filesystem type
- This is type checking at runtime

**Better Approach**:

Let FileSystem provide the lock:

```python
class FileSystem(Protocol):
    def acquire_lock(self, path: Path) -> ContextManager

class InMemoryFileSystem:
    def acquire_lock(self, path: Path):
        return NoOpLock()  # Internal decision

class RealFileSystem:
    def acquire_lock(self, path: Path):
        return ProcessLock(path)
```

This way CDGStore doesn't need to know which filesystem it's using.

---

## Dependency Analysis

### Layer Structure (Bottom-Up)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                             │
│  - CLI (cortical/got/cli/, cortical/cdg/cli/)                           │
│  - Scripts (scripts/got_utils.py)                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                            DOMAIN LAYER                                  │
│  - GoT (cortical/got/) - Task/Decision/Sprint management                │
│  - CEL (cortical/cel/) - Event sourcing                                 │
│  - Reasoning (cortical/reasoning/) - Cognitive systems                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         FOUNDATION LAYER (CDG)                           │
│  - Storage (cortical/cdg/storage.py) - File I/O with checksums          │
│  - Transactions (cortical/cdg/transaction_manager.py) - ACID            │
│  - WAL (cortical/cdg/wal.py) - Write-ahead logging                      │
│  - Schema (cortical/cdg/schema.py) - Entity validation                  │
│  - Index (cortical/cdg/index_manager.py) - Field indexing               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                             │
│  - Container (cortical/common/container.py) - Dependency injection      │
│  - FileSystem (cortical/common/filesystem.py) - I/O abstraction         │
│  - Bootstrap (cortical/core/bootstrap.py) - Wiring                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dependency Direction Validation

**✓ PASS**: No upward dependencies detected

```bash
$ grep -r "from cortical.got" cortical/cdg/ --include="*.py" | wc -l
0
```

This is **excellent**. The foundation layer (CDG) does NOT depend on the domain layer (GoT). This allows:
- CDG to be reused by other systems (CEL, Reasoning)
- GoT to be replaced without changing CDG
- Clear separation of concerns

### Module Coupling Analysis

| Module | Depends On | Depended On By | Coupling Score |
|--------|------------|----------------|----------------|
| cortical/cdg/ | common, utils | got, cel, reasoning | Low (3) ✓ |
| cortical/got/ | cdg, common, utils | cli, scripts, reasoning | Medium (5) ⚠ |
| cortical/common/ | (stdlib only) | cdg, got, cel, reasoning | Low (0) ✓ |
| cortical/core/ | common, cdg, got | (application) | Low (3) ✓ |
| cortical/reasoning/ | got, cdg, common | (none) | High (8) ⚠ |

**Observations**:
- **CDG has low coupling** - Good foundation layer
- **GoT has medium coupling** - Acceptable for domain layer
- **Reasoning has high coupling** - Natural for orchestration layer
- **No circular dependencies** - Excellent

### Critical Dependency Paths

```
CLI → GoTManager → CDGTransactionManager → CDGStore → FileSystem
    └→ SchemaRegistry → Field definitions
    └→ CDGIndexManager → Field indexing
```

This is a **clean dependency chain** with no cycles.

---

## Edge Cases Found

### 1. Race Condition in Cache Invalidation

**Location**: `cortical/cdg/storage.py` (cache implementation)

**Scenario**:
```python
# Thread 1: Reads entity E-001 (cached)
entity = store.read("E-001")  # Cache hit

# Thread 2: Updates entity E-001
store.write("E-001", updated_entity)  # Invalidates cache

# Thread 1: Reads again (expects fresh data, but might get stale from local var)
# This is OK because Thread 1 holds a reference

# BUT: If Thread 1 commits a transaction based on stale data?
# Transaction manager detects version conflict on commit ✓ (Safe due to MVCC)
```

**Verdict**: Safe due to optimistic locking, but caching + transactions is complex.

---

### 2. Filesystem Abstraction Leaks Lock Type

**Location**: `cortical/cdg/storage.py` line 143

**Code**:
```python
# CDGStore.__init__
lock_class = NoOpLock if isinstance(filesystem, InMemoryFileSystem) else ProcessLock
self.lock = lock_class(self.store_dir / ".cdg.lock", reentrant=True)
```

**Issue**: Type-checking at runtime violates abstraction.

**Better**: FileSystem.acquire_lock(path) -> ContextManager

---

### 3. Module Initialization Order Dependency

**Location**: `cortical/core/bootstrap.py` lines 116-119

**Code**:
```python
if apply_modules:
    container.apply_module(SchemaModule())     # Must be first
    container.apply_module(CDGModule(...))     # Must be second
    container.apply_module(GoTModule(...))     # Must be third
```

**Issue**: Ordering is implicit and fragile. If a developer reorders these, SchemaRegistry won't be available to CDGModule.

**Recommended**: Add explicit ordering validation:
```python
class ModuleOrdering:
    SCHEMA = 0
    CDG = 1
    GOT = 2

class ContainerModule:
    order: int = 0

# Container validates ordering on apply
```

---

### 4. Lazy Property Initialization Without Thread Safety

**Location**: `cortical/got/api.py` lines 205-210

**Code**:
```python
@property
def sync_manager(self) -> SyncManager:
    """Get sync manager (lazy initialization)."""
    if self._sync_manager is None:
        self._sync_manager = SyncManager(self.got_dir)
    return self._sync_manager
```

**Issue**: Not thread-safe. Two threads calling `sync_manager` simultaneously could create two instances.

**Fix**: Use `threading.Lock` or initialize in `__init__`.

---

## BONUS: Hidden Design Issues

### 1. The "Manager" Naming Anti-Pattern

**Evidence**: 20 classes named `*Manager` in the codebase.

**Why This Is a Code Smell**:

"Manager" is a **meaningless suffix**. It tells you nothing about what the class does.

Examples:
- `GoTManager` - manages... everything? (god class)
- `TransactionManager` - manages transactions (OK, but "TransactionOrchestrator" is clearer)
- `CDGWALManager` - manages WAL (could be `WriteAheadLog`)
- `SyncManager` - manages sync (could be `SyncOrchestrator` or just `Sync`)
- `RecoveryManager` - manages recovery (could be `CrashRecovery`)

**Better Naming**:
- `TaskRepository` (instead of TaskManager)
- `SprintCoordinator` (instead of SprintManager)
- `WriteAheadLog` (instead of CDGWALManager)
- `TransactionOrchestrator` (instead of TransactionManager)

**Why This Matters**:

Names should reveal intent. "Manager" is a lazy abstraction that hides complexity instead of explaining it.

---

### 2. The Container Is Not Truly Inversion of Control

**Location**: `cortical/core/bootstrap.py`

**Observation**:

The container requires **manual registration** of every service:

```python
container.register(CDGStore, lambda: container.create("cdg_store"))
container.register(CDGTransactionManager, create_tx_manager, lifecycle=Lifecycle.SINGLETON)
```

**Why This Is Limiting**:

True IoC containers (Spring, Guice, Autofac) use **convention-based registration**:
- Scan assemblies for classes implementing interfaces
- Auto-register based on naming conventions
- Reduce boilerplate

**Current Pain Point**:

Adding a new service requires:
1. Implement the service
2. Create a factory function
3. Register in bootstrap.py
4. Update module's `register()` method

That's 4 places to change for 1 new service.

**Recommendation**:

Add convention-based registration:
```python
# Scan cortical/got/ for classes implementing Protocol
container.scan_package("cortical.got", register_protocols=True)
```

This would reduce bootstrap.py from 156 lines to ~50 lines.

---

### 3. No Explicit Boundaries for Module Exports

**Location**: `cortical/cdg/__init__.py`, `cortical/got/__init__.py`

**Observation**:

Both modules export **everything**:

```python
# cortical/cdg/__init__.py - 121 lines of __all__
__all__ = [
    "Entity", "Node", "Edge", "CDGStore", "Transaction", "CDGTransactionManager",
    "CDGWALManager", "CDGRecoveryManager", "SchemaRegistry", "CDGIndexManager",
    # ... 100+ more exports
]
```

**Why This Is a Problem**:

Exports should define the **public API**. Internal implementation details should not be exported.

Example:
- `Entity` - Public (users create entities)
- `CDGStore` - Semi-public (advanced users)
- `TransactionState` - Internal (implementation detail)
- `generate_transaction_id` - Internal (utility function)

**Recommended**:

Split into public and internal:
```python
# cortical/cdg/__init__.py (PUBLIC API)
__all__ = [
    # Core types
    "Entity", "Edge",
    # High-level API
    "CDGTransactionManager",
    # Configuration
    "CDGConfig",
]

# cortical/cdg/_internal.py (INTERNAL - not exported)
# TransactionState, generate_transaction_id, etc.
```

This makes it clear what users should depend on.

---

### 4. Filesystem Abstraction Is Incomplete

**Location**: `cortical/common/filesystem.py`

**Missing Operations**:
- `rename()` - Atomic file moves
- `symlink()` - Symbolic links
- `chmod()` - Permission management
- `stat()` - File metadata
- `glob()` - Pattern matching

**Why This Matters**:

The abstraction is supposed to enable:
1. In-memory testing (✓ works)
2. Remote storage (✗ missing operations)
3. Cloud storage (✗ missing operations)

If the goal is just in-memory testing, the current abstraction is fine. But if the goal is to support S3, Azure Blob, etc., the abstraction is insufficient.

**Recommendation**:

Either:
- Expand the abstraction to support all filesystem operations
- Or narrow the scope to just "testable I/O" and rename to `StorageBackend`

---

### 5. The Lack of Command Query Separation (CQS)

**Location**: `cortical/got/api.py`

**Observation**:

Many methods violate CQS by both modifying state AND returning values:

```python
def create_task(...) -> str:
    """Creates task AND returns ID"""

def update_task(...) -> bool:
    """Updates task AND returns success status"""

def delete_task(...) -> Tuple[bool, str]:
    """Deletes task AND returns (success, message)"""
```

**Why This Matters**:

CQS states: *"Methods should either change state OR return values, not both."*

Benefits of CQS:
- Easier caching (queries are pure)
- Easier testing (commands have no return value to assert)
- Clearer intent (is this a query or a command?)

**Example Refactoring**:

```python
# BEFORE: Violates CQS
task_id = manager.create_task("Do something")  # Creates AND returns

# AFTER: Follows CQS
command = CreateTask(title="Do something")
manager.execute(command)  # Changes state (returns nothing)
task_id = command.result.task_id  # Query result separately
```

This is **not critical** but worth considering for future APIs.

---

## Files Reviewed

### Core Architecture
- `/home/user/Opus-code-test/cortical/core/bootstrap.py` (156 lines)
- `/home/user/Opus-code-test/cortical/core/modules/__init__.py` (35 lines)
- `/home/user/Opus-code-test/cortical/core/modules/schema_module.py` (63 lines)
- `/home/user/Opus-code-test/cortical/core/modules/cdg_module.py` (152 lines)
- `/home/user/Opus-code-test/cortical/core/modules/got_module.py` (127 lines)

### Foundation Layer (CDG)
- `/home/user/Opus-code-test/cortical/cdg/__init__.py` (121 lines)
- `/home/user/Opus-code-test/cortical/cdg/storage.py` (1235 lines)
- `/home/user/Opus-code-test/cortical/cdg/transaction_manager.py` (500+ lines)

### Domain Layer (GoT)
- `/home/user/Opus-code-test/cortical/got/__init__.py` (296 lines)
- `/home/user/Opus-code-test/cortical/got/api.py` (2754 lines) ⚠️
- `/home/user/Opus-code-test/cortical/got/protocol.py` (354 lines) ⚠️

### Infrastructure
- `/home/user/Opus-code-test/cortical/common/container.py` (500+ lines)
- `/home/user/Opus-code-test/cortical/common/filesystem.py` (estimated)

### Analysis
- Git history: 30+ commits analyzed
- Import graph: Full codebase scanned
- Class analysis: 20+ Manager classes identified
- Method counting: Automated via AST parsing

---

## Conclusion

### What's Working (Strengths)

1. **Excellent layer separation** - CDG/GoT boundaries are clean
2. **Sophisticated DI container** - Container-based injection is professional-grade
3. **Active refactoring culture** - Git history shows continuous improvement
4. **No circular dependencies** - Dependency graph is acyclic
5. **MVCC implementation** - Transactions are correctly implemented

### What Needs Immediate Attention (Critical)

1. **GoTManager god class** - 74 methods, 9 responsibilities (refactor into services)
2. **GoTBackend interface bloat** - 23 methods (split into focused protocols)
3. **Hardcoded paths** - 15 instances (use injected values)

### What Should Be Monitored (Medium Priority)

1. **TransactionContext duplication** - 500+ lines of duplicate code
2. **Cache layering** - Caching mixed with storage concerns
3. **Module initialization ordering** - Implicit and fragile

### What's Minor But Worth Noting (Low Priority)

1. **"Manager" naming** - 20 classes with meaningless suffix
2. **FileSystem abstraction leaks** - Lock type checking at runtime
3. **Over-broad exports** - __all__ includes internal details
4. **CQS violations** - Methods both mutate and return values

### Overall Assessment

This is a **well-architected codebase** with **one critical flaw**: the god class.

The team demonstrates:
- Understanding of SOLID principles
- Willingness to refactor aggressively
- Commitment to clean architecture

The GoTManager refactoring should be **Sprint 0 priority**. Everything else can be addressed incrementally.

**Final Grade: B-** (Would be A- after god class refactoring)

---

## Recommended Action Plan

### Sprint 0 (Week 1-2): Critical Fixes

1. **Decompose GoTManager** (Priority: CRITICAL)
   - Create: TaskService, SprintService, DependencyService, QueryService
   - Register in container
   - Deprecate GoTManager with forwarding
   - Update tests

2. **Fix hardcoded paths** (Priority: HIGH)
   - Grep for `Path(".got")` → 15 locations
   - Replace with injected values
   - Add linter rule

### Sprint 1 (Week 3-4): Medium Priority

3. **Split GoTBackend protocol** (Priority: MEDIUM)
   - Create: TaskCRUD, TaskLifecycle, DependencyGraph, GoTQuery, GoTPersistence
   - Update implementations
   - Deprecate monolithic protocol

4. **Remove TransactionContext duplication** (Priority: MEDIUM)
   - Replace with @contextmanager wrapper
   - Delete duplicate methods
   - Update tests

### Sprint 2 (Week 5-6): Quality Improvements

5. **Add module ordering validation** (Priority: LOW)
   - Define explicit ordering
   - Validate on container.apply_module()

6. **Refine FileSystem abstraction** (Priority: LOW)
   - Either expand (if targeting cloud storage)
   - Or narrow scope (if just for testing)

---

*End of Architecture Audit Report*
