# File Access Audit - Working Scratchpad

*Session: 2026-01-06*
*Branch: claude/fix-file-access-issues-1zUM9*

---

## WORKFLOW NOTES (for context preservation)

1. **DON'T track "done" here** - commit often with clear messages instead
2. **Use subagents** to check git messages regularly (`git log --oneline -10`)
3. **Preserve context window** - scratchpad is for active thinking, not history
4. **We're refactoring while figuring out how to refactor** - iterative process

---

## Current State

**Active:** Fixing broken imports from deleted versioned_store.py

**Next actions:**
- Fix query_builder.py import (just did)
- Scan for more file access issues
- Commit changes with clear message

---

## CRITICAL NOTES

### DO NOT UPDATE TESTS
- Wait until fully done
- User will say when ready for testing
- Tests will catch issues - let them break for now

### CDG Architecture Philosophy
- CDG = clean, generic solution built on first principles
- Being built over time as needed
- CDGTransactionManager IS generic enough for GoT
- **NO TWO LAYERS** - don't maintain wrappers

### When Dealing with GoT
- **ALWAYS consider moving functionality DOWN to CDG**
- Prefer refactoring GoT into CDG over maintaining two versions
- GoT should be thin domain layer on top of CDG

---

## What GoT Uniquely Provides (keep in GoT)

1. **Entity types** - Task, Decision, Edge, Sprint, etc. (domain models)
2. **Entity factory** - `create_entity_from_dict()` dispatches to correct type
3. **QueryIndexManager** - GoT-specific indexing (consider: generic version for CDG?)
4. **GoTManager** - high-level domain API
5. **Domain logic** - orphan detection, etc.

---

## What Should Move to CDG / Already in CDG

- TransactionManager → CDGTransactionManager (DONE - GoT wrapper deleted)
- VersionedStore → CDGStore (DONE - GoT version deleted)
- Recovery → CDGRecoveryManager (INVESTIGATED - see notes below)
- Schema → CDG schema (DONE)

---

## Completed Tasks

### tx_manager.py Deletion
- Was a wrapper around CDGTransactionManager
- Had broken import from deleted versioned_store.py
- Per user: no two layers needed
- **Deleted:** Updated all imports to use CDGTransactionManager directly
- Aliased as `TransactionManager` in `__init__.py` for backward compatibility

### Recovery Analysis (got/recovery.py vs cdg/recovery.py)

**CDG Recovery is MORE comprehensive:**
- Has `reconstruct_entities_from_wal()` (GoT lacks this)
- Has `MIN_ENTITY_FILE_SIZE` check for truncated files
- Configurable via CDGConfig (recovery_mode, orphan_strategy)
- Uses proper WAL logging via `self.wal.log()`

**GoT Recovery has GoT-specific logic:**
- `needs_index_recovery()` with GoT QueryIndexManager
- `rebuild_indexes()` creates Task objects directly
- Uses GoT's WALManager not CDGWALManager

**Decision: NOT deleting GoT recovery yet**
- GoT recovery.py has domain-specific index logic
- Would need to refactor to use callback pattern like CDG
- Lower priority than other file access issues
- Could delegate core recovery to CDG and add GoT index layer

---

## Architectural Insights (from user)

### Durability Mode
- GoT has NO business with durability
- Configured centrally in CDG

### got/expression/*
- Half-baked but FANTASTIC ideas
- Will need schema in a GENERAL way
- Be careful - harder than it sounds

### got/cli/* Pattern
- Container as member variable
- DO NOT import bootstrap in functions

---

## Design Decisions

1. No backward compatibility
2. Container-first
3. CDG is foundation - GoT is thin domain layer
4. Required parameters - no fallbacks
5. Centralized configuration
6. **Move GoT functionality to CDG when possible**
