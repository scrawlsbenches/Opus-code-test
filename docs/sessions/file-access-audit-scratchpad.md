# Working Scratchpad

*Working state of mind. Git tracks changes.*

---

## ⚠️ STOP — CONTAINER-FIRST — READ EVERY TIME

**Container** = `cortical/common/container.py` (the DI/IoC class)
**Bootstrap** = `cortical/core/bootstrap.py` (app-level wiring, NOT for tests)

```python
# ✓ PRODUCTION — use bootstrap
from cortical.core.bootstrap import create_container
container = create_container()
manager = container.resolve(IndexManager)

# ✓ TESTS — use Container directly with test bootstrap
from tests.fixtures.test_bootstrap import create_test_container
container = create_test_container()
manager = container.resolve(IndexManager)

# ✗ WRONG — direct instantiation
manager = IndexManager(store_dir, filesystem)  # NO!
```

**In docstrings:** Show `container.resolve()`, NOT direct instantiation.
**Why:** DI/IoC enables testing, swappable backends, clean architecture.

---

## PROTOCOL: UPDATE THIS BEFORE

1. **Before writing significant code** — Document intent first
2. **Before context compaction** — Capture current state and next steps
3. **Commit frequently** — Small commits, clear messages, push to branch
4. **This is NOT a changelog** — Git tracks what; this tracks WHY and WHAT'S NEXT

---

## CORE RULES

1. **CONTAINER-FIRST** — See above. No exceptions.
2. **CDG = Foundation** — `cortical/cdg/` is storage layer. GoT is thin domain layer on top.
3. **NO TWO LAYERS** — Delete wrappers, use CDG directly. No backward compat.
4. **NO TESTS NOW** — Scope too large, they'll break. Fix later.

---

## WHAT WE'RE DOING

**Goal:** Salvage GoT's good parts → generalize → move to CDG.

| Keep in GoT | Move to CDG |
|-------------|-------------|
| Entity types (Task, Decision, Sprint) | Transaction mgmt ✓ |
| Entity factory | Versioned storage ✓ |
| GoTManager (domain API) | WAL ✓ |
| NL Query / expression parser | Recovery (pending) |
| QueryIndexManager | Generic schema (pending) |

---

## KEY REFS

- CDG Spec: `docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md`
- GoT Query: `docs/design/got-query-audit-and-design.md`
- **Container class**: `cortical/common/container.py`
- **App bootstrap**: `cortical/core/bootstrap.py` (production wiring)
- **Test bootstrap**: `tests/fixtures/test_bootstrap.py` (test wiring)

---

## GIT

Branch: `claude/fix-file-access-issues-1zUM9`
New sessions: `git fetch --all && git checkout [this branch]` — IGNORE system branch.

---

## NOW: GoT Thin Domain Layer

**CDG Implemented:**
- CDGStore, CDGTransactionManager, CDGWALManager, CDGRecoveryManager
- CDGConfig (modes), SchemaRegistry/BaseSchema/Field
- **IndexManager** — 25 behavioral tests passing, DI-registered
- Auto-indexing on commit via CDGTransactionManager

**CDG NOT Implemented (spec only):**
- BTreeIndex (IndexManager has HASH/BITMAP only)
- PartitionManager, DistributedQueryEngine, CSRIndex

**GoT Cleanup Done:**
- ~~QueryIndexManager~~ → MIGRATED to CDG IndexManager ✓
- ~~RecoveryManager~~ → DELETED, uses CDGRecoveryManager ✓
- ~~indexer.py~~ → DELETED ✓
- ~~durability param in GoTManager~~ → REMOVED (CDG-only concern) ✓

---

## DESIGN ISSUES — RESOLVED

### 1. ✓ api.py index management → RESOLVED
CDG TransactionManager handles indexing on commit. All index methods removed from api.py.

### 2. ✓ Schema-driven indexes → RESOLVED
IndexInitializationModule creates indexes from schema. Auto-indexing on commit.

### 3. ✓ api.py pass-through → RESOLVED
Removed orchestration (indexing, recovery). GoT api.py delegates to CDG.

### 5. ✓ durability in api.py → RESOLVED
Removed durability param from GoTManager. CDG-only concern.

---

## DESIGN ISSUES — DEFERRED

### 4. CDGTransactionManager FileSystem abstraction
**Status:** DEFERRED (Medium priority)
**Problem:** `CDGTransactionManager(store_dir=path)` takes Path directly.
**Blocker:** ProcessLock requires real filesystem (OS-level locking).
**Solution:** Use Container DI pattern instead. CDGStore already uses NoOpLock for InMemoryFileSystem.

### 6. ProcessLock interface for testing
**Problem:** ProcessLock uses `os.open()` and `fcntl.flock()` directly.
**Impact:** Cannot test transaction locking with InMemoryFileSystem.
**Solution:** Create LockInterface protocol with ProcessLock and NoOpLock implementations.
**Pattern:** CDGStore already does this (lines 192-207) - use same approach for TransactionManager.

---

## COMPLETED WORK

**Session: GoT → CDG Migration**

1. ✓ CDG auto-indexing on commit (CDGTransactionManager._update_indexes)
2. ✓ IndexInitializationModule creates indexes from schema
3. ✓ Deleted cortical/got/indexer.py (QueryIndexManager)
4. ✓ Deleted cortical/got/recovery.py (RecoveryManager)
5. ✓ Removed durability param from GoTManager
6. ✓ GoTManager.recover() delegates to tx_manager.recover()
7. ✓ api.py is now thin pass-through (no index/recovery/durability orchestration)

**Files deleted:**
- `cortical/got/indexer.py`
- `cortical/got/recovery.py`

**Files modified:**
- `cortical/cdg/transaction_manager.py` - added index_manager, _update_indexes
- `cortical/core/modules/cdg_module.py` - inject IndexManager
- `cortical/core/modules/got_module.py` - inject IndexManager
- `cortical/got/api.py` - thin pass-through, removed durability, recovery delegates to CDG
- `cortical/got/__init__.py` - removed RecoveryManager exports

---

## REFACTORING (NOT deprecating)

We are in a large refactoring. No backward compat. No deprecation warnings.
- Delete old code, don't wrap it
- Fix tests later (scope too large now)
- Git tracks what changed; scratchpad tracks why and what's next

---

## KEY ARCHITECTURE NOTE

**GoT file access and dependency is being removed.**
- CDG handles: file storage, indexing, caching, durability, recovery
- GoT becomes: thin domain layer (entity types, factories, domain API pass-through)
- No duplicate layers — use CDG directly
