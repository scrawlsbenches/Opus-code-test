# File Access Audit - Working Scratchpad

*Session: 2026-01-06*
*Branch: claude/fix-file-access-issues-1zUM9*

---

## ⛔ SESSION OVERRIDES (supersedes CLAUDE.md Entry Gate)

**READ THIS FIRST. These override default workflow steps.**

- **DO NOT RUN TESTS** - tests are broken due to refactoring, user will say when ready
- **DO NOT RUN SMOKE TESTS** - same reason
- **DO RUN `got_utils.py validate`** - this is safe and helpful

**Why:** We're in the middle of a multi-session refactoring. Tests reference
deleted modules. Running them wastes time and produces noise.

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

**Previous session branch:** `claude/refactor-cortical-codebase-8ij55`

**First Step:** Merge previous work into your session branch:

git fetch origin claude/refactor-cortical-codebase-8ij55
git merge origin/claude/refactor-cortical-codebase-8ij55

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
git fetch origin claude/refactor-cortical-codebase-8ij55  # branch from prompt
git merge origin/claude/refactor-cortical-codebase-8ij55

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

**Previous session branch:** `claude/refactor-cortical-codebase-8ij55`

## First Steps (DO THIS BEFORE ANYTHING ELSE)

1. Merge previous work into your session branch:

git fetch origin claude/refactor-cortical-codebase-8ij55
git merge origin/claude/refactor-cortical-codebase-8ij55
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

**Status:** REFACTORING PHASE 2 - Recovery & Duplication

**Completed (Phase 1):**
- Deleted `got/wal.py` → use CDGWALManager directly
- Deleted `got/tx_manager.py` → use CDGTransactionManager directly
- Deleted `got/versioned_store.py` → use CDGStore directly
- Removed fallback patterns in `api.py` and `query_api.py`
- Schema moved to `cdg/schema/`

---

## ⚠️ PRIORITY TASK LIST (in order)

### 1. RECOVERY MODULE CONSOLIDATION [CRITICAL - ACTIVE BUGS]

**GoT recovery.py has BUGS that can cause data loss:**

| Bug | Location | Impact |
|-----|----------|--------|
| Direct WAL file append | Lines 612-624 | No locking, no fsync - crash corrupts WAL |
| Missing `reconstruct_entities_from_wal()` | N/A | Loses data after crash-after-commit |
| ADOPTED entry format mismatch | Lines 507-509 | Doesn't handle CDG's new format |
| String vs Enum strategy | Line 527 | Type confusion, called with wrong value |

**Duplication:** 350 lines (44% of combined 1425 lines)

**Action:** Consolidate NOW. GoT delegates to CDG for core recovery, keeps only index logic.

### 2. FIX TEST IMPORTS [MECHANICAL]

13 files, ~37 broken imports. Estimated 2 hours.
Blocked until recovery consolidation complete (may change more imports).

### 3. VALIDATION RULE EXTRACTION [TECHNICAL DEBT]

**Current state:** 9 if/elif blocks in `_validate_entity_specific()` (lines 406-508)
- Each entity type repeats same pattern
- Adding new type requires modifying function
- Rules embedded in code, not visible at a glance

**Target:** Extract to `ENTITY_SCHEMAS` data structure
- Declarative rules
- Adding new entity = adding dict entry
- ~90 lines → ~60 lines

### 4. GRAPH TRAVERSAL CONSOLIDATION [DUPLICATION]

3 overlapping utilities:
- `GraphWalker` (642 lines)
- `PathFinder` (646 lines)
- `PatternMatcher` (739 lines)

Could share base class for traversal logic.

---

## Recovery Consolidation Plan

**Strategy:** GoT recovery becomes thin wrapper around CDG recovery

1. Move `RecoveryResult` and `RepairResult` to `cortical/common/`
2. GoT recovery takes CDGRecoveryManager as dependency
3. GoT only adds: `needs_index_recovery()`, `rebuild_indexes()`
4. Delete duplicated methods, delegate to CDG
5. Fix the bugs (use WAL.log(), handle ADOPTED format)

**CDG has that GoT lacks:**
- `reconstruct_entities_from_wal()` - CRITICAL
- `MIN_ENTITY_FILE_SIZE` check
- RecoveryMode enum (configurable)
- OrphanStrategy enum (type-safe)
- Proper WAL logging in repair_orphans()

**GoT has that's unique:**
- `needs_index_recovery()` - real implementation (CDG is placeholder)
- `rebuild_indexes()` - GoT domain-specific

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
