# File Access Audit - Working Scratchpad

*Session: 2026-01-06*
*Branch: claude/fix-file-access-issues-1zUM9*

---

## Current State

We've uncovered major redundancy between GoT and CDG:

**Both layers have:**
- TransactionManager (GoT wraps CDG)
- Recovery (CDG has full implementation, GoT adds index rebuilding)
- VersionedStore (DELETED - CDGStore is the real one)

**What GoT uniquely provides:**
1. Entity factory (`create_entity_from_dict` in types.py) - dispatches to Task/Decision/Sprint
2. QueryIndexManager - GoT-specific indexing
3. Domain types (Task, Decision, Edge, Sprint, etc.)
4. Higher-level API (GoTManager)

---

## Architectural Question: How thin should GoT be?

**Option A: Keep GoT's TransactionManager as thin wrapper**
- Pro: Stable API for existing code
- Con: Extra layer of indirection

**Option B: Delete GoT's tx_manager.py, use CDGTransactionManager directly**
- Pro: Less code, clearer architecture
- Con: Need to update all callers, entity_factory passed differently

**Option C: Hybrid - GoTModule creates CDGTransactionManager with entity factory**
- GoTModule already does this!
- Just need to delete the wrapper

---

## Broken Imports Found

`cortical/got/tx_manager.py:32` imports from deleted file:
```python
from .versioned_store import _got_entity_factory
```

This needs immediate fix (use `types.create_entity_from_dict`).

---

## Architectural Insights (from user)

### Durability Mode
- GoT has NO business with durability mode
- Must be configured centrally (in Container/CDG)

### got/expression/*
- Contains half-baked but FANTASTIC ideas
- Will need to utilize schema more but in a GENERAL way
- This is harder than it sounds - be careful!

### got/cli/* Pattern
- Should have a member variable for Container
- DO NOT import cortical.core.bootstrap in functions

---

## Design Decisions

1. **No backward compatibility** - fix problems directly
2. **Container-first** - dependencies injected via Container
3. **CDG is foundation** - all storage goes through CDG
4. **Required parameters** - no optional with fallback to globals
5. **Centralized configuration** - durability, paths at Container level

---

## Notes

- Don't run tests until user says to
- Go slow, check in before making changes
- CLAUDE.md discussion deferred
