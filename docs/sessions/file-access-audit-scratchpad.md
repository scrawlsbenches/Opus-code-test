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

## GIT HANDLING FOR SESSION CONTINUATIONS

**Problem:** Each new session gets a NEW branch name from the system (based on session ID).
The system only allows pushing to the session-assigned branch.

**Solution:** Work on the system-assigned branch, merge previous work into it.

### FOR HUMANS: Creating a Continuation Prompt

```markdown
Continue refactoring work on the Cortical codebase.

**Previous session branch:** `claude/refactor-cortical-codebase-omD09`

**First Step:** Merge previous work into your session branch:

git fetch origin claude/refactor-cortical-codebase-omD09
git merge origin/claude/refactor-cortical-codebase-omD09

Then read the scratchpad:
cat docs/sessions/file-access-audit-scratchpad.md
```

**Key elements:**
1. Specify the PREVIOUS session's branch (where work was pushed)
2. Tell agent to merge that into their new session branch
3. Reference the scratchpad for context

### FOR AI AGENTS: What to Do in a New Session

```bash
# 1. You're already on your session-assigned branch - STAY ON IT

# 2. Fetch and merge the previous session's work
git fetch origin claude/refactor-cortical-codebase-omD09  # branch from prompt
git merge origin/claude/refactor-cortical-codebase-omD09

# 3. Verify you have the previous work
git log --oneline -5

# 4. Read the scratchpad
cat docs/sessions/file-access-audit-scratchpad.md

# 5. Work normally, push to YOUR session branch (the one system assigned)
git push -u origin HEAD
```

**Key insight:** Each session uses its own branch. Merge previous work in.
Push normally to your session branch - no special syntax needed.

### Complete Continuation Prompt Template (COPY-PASTE THIS)

```markdown
Continue refactoring work on the Cortical codebase.

**Previous session branch:** `claude/refactor-cortical-codebase-omD09`

## First Steps (DO THIS BEFORE ANYTHING ELSE)

1. Merge previous work into your session branch:

git fetch origin claude/refactor-cortical-codebase-omD09
git merge origin/claude/refactor-cortical-codebase-omD09
git log --oneline -5  # verify you have the history

2. Read the scratchpad:

cat docs/sessions/file-access-audit-scratchpad.md

## Current State

**File Access Audit:** COMPLETE ✓

Completed refactoring:
- Deleted `got/wal.py` → using CDGWALManager directly
- Deleted `got/tx_manager.py` → using CDGTransactionManager directly
- Deleted `got/versioned_store.py` → using CDGStore directly
- Removed backward-compatibility fallbacks in api.py and query_api.py

Remaining file access (all acceptable):
- `recovery.py` - Must bypass CDG to detect/fix corruption
- `indexer.py` - Index persistence separate from entity store
- `claudemd.py` - Context gathering, output files
- `cli/*` - User-facing display/import

## Key Principles (from scratchpad)

1. **Container-first** - DI for all dependencies
2. **No backward compatibility** - fix directly, no fallbacks
3. **CDG is foundation** - GoT is thin domain layer
4. **Move functionality to CDG** when possible

## When Done

Push to YOUR session branch (the one the system assigned you):

git push -u origin HEAD

Update the scratchpad's "Previous session branch" in this template before ending.
```

**Remember:** Update the branch name in the template above before ending your session!

---

## Current State

**Status:** AUDIT COMPLETE ✓

**Summary of changes (see git log):**
- Deleted `got/wal.py` → use CDGWALManager directly
- Removed fallback patterns in `api.py` and `query_api.py`
- Simplified `_iter_entities_by_prefix` and `list_claudemd_layers`

**Remaining file access (ALL ACCEPTABLE):**
- `recovery.py` - MUST bypass CDG to detect/fix corruption
- `indexer.py` - Index persistence is separate from entity store
- `claudemd.py` - Context gathering, generates output files
- `cli/*` - User-facing display/import operations

**No critical issues remain.** Architecture is solid.

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

### wal.py Deletion (2026-01-06)
- GoT's WALManager was 99% identical to CDGWALManager
- Only recovery.py was still using it (GoTManager used CDGTransactionManager which uses CDGWALManager)
- CDGWALManager is superior (deferred sequence commit, entity_data support)
- **Deleted:** Updated recovery.py to use CDGWALManager directly
- Updated `__init__.py` exports

### Fallback Pattern Removal (2026-01-06)
- Removed "backward compatibility" fallbacks in api.py and query_api.py
- `_iter_entities_by_prefix` now simply calls `tx_manager.store.iter_entities()`
- `list_claudemd_layers` simplified to use store directly
- Per user: "NO backward compatibility - fix directly"

### Recovery Analysis (got/recovery.py vs cdg/recovery.py)

**CDG Recovery is MORE comprehensive:**
- Has `reconstruct_entities_from_wal()` (GoT lacks this)
- Has `MIN_ENTITY_FILE_SIZE` check for truncated files
- Configurable via CDGConfig (recovery_mode, orphan_strategy)
- Uses proper WAL logging via `self.wal.log()`

**GoT Recovery has GoT-specific logic:**
- `needs_index_recovery()` with GoT QueryIndexManager
- `rebuild_indexes()` creates Task objects directly
- Now uses CDGWALManager (updated 2026-01-06)

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
