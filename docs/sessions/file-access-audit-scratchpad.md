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

## NOW: Recovery Cleanup + GoT Migration

**CDG Implemented:**
- CDGStore, CDGTransactionManager, CDGWALManager, CDGRecoveryManager
- CDGConfig (modes), SchemaRegistry/BaseSchema/Field
- **IndexManager** — 25 behavioral tests passing, DI-registered

**CDG NOT Implemented (spec only):**
- BTreeIndex (IndexManager has HASH/BITMAP only)
- PartitionManager, DistributedQueryEngine, CSRIndex

**GoT still has:**
- ~~QueryIndexManager~~ → MIGRATED to CDG IndexManager ✓
- RecoveryManager — VIOLATES Container-first, duplicates CDG → DELETE
- indexer.py file still exists (unused, can delete)

---

## DESIGN ISSUES TO RESOLVE

### 1. api.py should NOT manage indexes
**Problem:** `cortical/got/api.py` has `_update_index_for_task()` and calls `index_manager.index_entity()`.
**Correct:** CDG handles indexing automatically. GoT api.py is pass-through only.
**Fix:** Remove all index management from api.py. CDG TransactionManager should handle indexing on commit.

### 2. Indexes configured in schema, not api.py
**Status:** Schema has `indexes` attr ✓. IndexInitializationModule creates them ✓.
**Problem:** api.py still calls index methods manually.
**Fix:** CDG should auto-index on entity write based on schema config.

### 3. api.py should be pass-through
**Problem:** api.py does too much orchestration (indexing, recovery, caching).
**Correct:** GoT api.py → thin domain layer → delegates to CDG for all storage concerns.
**Fix:** Remove orchestration. Pass through to CDG services.

### 4. CDGTransactionManager shouldn't need Path
**Problem:** `CDGTransactionManager(store_dir=path)` takes Path directly.
**Correct:** Should use FileSystem interface from container for consistency.
**Question:** Is this a design issue? Should CDGTransactionManager receive FileSystem via DI?

### 5. api.py should not know about durability
**Problem:** `GoTManager.__init__` takes `durability: DurabilityMode`.
**Correct:** Durability is infrastructure concern → CDG config, not GoT.
**Question:** Is this a design issue? Should durability be CDG-only?

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
