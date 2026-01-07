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

**Status:** REFACTORING PHASE 3 - CDG Index Manager

**Completed (Phase 1 - File Access):**
- Deleted `got/wal.py` → use CDGWALManager directly
- Deleted `got/tx_manager.py` → use CDGTransactionManager directly
- Deleted `got/versioned_store.py` → use CDGStore directly
- Removed fallback patterns in `api.py` and `query_api.py`
- Schema moved to `cdg/schema/`

**Completed (Phase 2 - Recovery):**
- Deleted `cortical/got/recovery.py` entirely (641 lines → 0)
- Created `cortical/common/recovery_types.py` (shared RecoveryResult, RepairResult)
- CDG owns ALL recovery logic - no GoT recovery code

---

## ⚠️ PRIORITY TASK LIST (in order)

### 1. CDG INDEX MANAGER [IN PROGRESS]

**Architecture decision (from user):**
- Indexes should be maintained by CDG, NOT GoT
- Indexes configured in schema via `Field(indexed=True)` (like SQL Server column indexes)
- GoT is a query/facade layer - NO file I/O for index maintenance

**Implementation plan:**
1. ✅ Add `indexed` and `index_type` to Field in `cdg/schema/__init__.py`
2. 🔄 Create `cortical/cdg/index_manager.py` with CDGIndexManager
3. ⬜ Integrate IndexManager with CDGStore (update indexes on write)
4. ⬜ Integrate IndexManager with CDGRecoveryManager
5. ⬜ Remove `index_stale_callback` and `index_rebuild_callback` from CDGConfig
6. ⬜ Remove `_is_index_stale()`, `_rebuild_indexes_callback()` from GoTManager
7. ⬜ Update GoT entity schemas with `indexed=True` on queryable fields

**CDGIndexManager API:**
```python
class CDGIndexManager:
    def update_index(self, entity_type, entity_id, old_data, new_data)
    def lookup(self, entity_type, field_name, value) -> Set[str]
    def rebuild_all(self)
    def needs_rebuild(self) -> bool
```

### 2. FIX TEST IMPORTS [MECHANICAL]

13 files, ~37 broken imports.
Blocked until index manager complete (may change more imports).

### 3. VALIDATION RULE EXTRACTION [TECHNICAL DEBT]

**Current state:** 9 if/elif blocks in `_validate_entity_specific()` (lines 406-508)
- Each entity type repeats same pattern
- Adding new type requires modifying function

**Target:** Extract to `ENTITY_SCHEMAS` data structure

### 4. GRAPH TRAVERSAL CONSOLIDATION [DEFERRED]

3 overlapping utilities. Lower priority.

---

## ⚠️ CRITICAL ARCHITECTURE RULES

**CDG owns infrastructure:**
- Storage, transactions, WAL
- Recovery (ALL of it)
- Indexes (schema-based, like SQL Server)

**GoT is thin domain layer:**
- Query/facade over CDG
- Entity types and business logic
- NO file I/O for infrastructure
- Callbacks for NOTIFICATION only, not maintenance

**Wrong pattern (remove these from GoT):**
```python
# WRONG - GoT doing file I/O for indexes
def _is_index_stale(self):
    index_path = self.store_dir / "indexes.json"  # NO!

# WRONG - GoT doing file maintenance
def _rebuild_indexes_callback(self):
    self._indexer.rebuild()  # CDG should call this
```

**Right pattern:**
```python
# RIGHT - Index configured in schema
Field("status", FieldType.STRING, indexed=True)

# RIGHT - CDG maintains indexes internally
class CDGIndexManager:
    def update_index(...)  # Called by CDGStore on write
```

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
