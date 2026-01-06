# Working Scratchpad

*Working state of mind. Git tracks changes.*

---

## ⚠️ STOP — CONTAINER-FIRST — READ EVERY TIME

```python
# ✓ CORRECT — resolve from container
from cortical.core.bootstrap import create_container
container = create_container()
index_manager = container.resolve(IndexManager)

# ✗ WRONG — direct instantiation
index_manager = IndexManager(store_dir, filesystem)  # NO!
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
- Container: `cortical/core/bootstrap.py` + `cortical/common/container.py`

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
- QueryIndexManager — should migrate to use CDG IndexManager
- RecoveryManager — VIOLATES Container-first, duplicates CDG → DELETE

**Next steps:**
1. Delete GoT recovery.py, use CDGRecoveryManager
2. Migrate GoT QueryIndexManager → CDG IndexManager
3. Container-first for all instantiation
