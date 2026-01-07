# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## WORKFLOW NOTES (for context preservation)

1. **DON'T track "done" here** - commit often with clear messages instead
2. **Use subagents** to check git messages regularly (`git log --oneline -10`)
3. **Preserve context window** - scratchpad is for active thinking, not history
4. **We're refactoring while figuring out how to refactor** - iterative process

---

## HANDOFF FOR NEW THREAD

### What was done this session:
1. Fixed `use_memory=True` test isolation bug
2. Consolidated DurabilityMode: GoT now re-exports from CDG
3. Traced all CDGConfig fields - found only 6 of 23+ are actually used
4. **Found critical bugs in DurabilityMode implementation**

### Critical bugs found (NOT YET FIXED):

**BALANCED mode double-fsyncs entities** - This is wrong and wasteful:
- storage.py:838 fsyncs each entity write
- THEN transaction_manager:374 calls fsync_all() again

**The root cause:** storage.py checks `not in (FAST, RELAXED)` which means both PARANOID and BALANCED fsync per-write. BALANCED should only fsync at commit time.

### Changes needed (approved plan):

1. **storage.py** (4 locations: 838, 915, 1013, 1045)
   - Change: `if self.durability not in (FAST, RELAXED)`
   - To: `if self.durability == DurabilityMode.PARANOID`

2. **transaction_manager.py** (line 339)
   - Change: `if self.wal: self.wal.fsync_now()`
   - To: `if self.wal and self.config.durability != DurabilityMode.RELAXED: self.wal.fsync_now()`

3. **Remove FAST enum** - Keep only RELAXED (more descriptive)
   - Update CDGConfig.for_high_performance() to use RELAXED
   - Update any tests using FAST

4. **Behavioral tests** - Verify each mode works correctly

### Correct behavior after fix:

| Component | PARANOID | BALANCED | RELAXED |
|-----------|----------|----------|---------|
| WAL entry | fsync | flush | flush |
| WAL on commit | fsync | fsync | NO |
| Entity write | fsync | NO | NO |
| fsync_all post-commit | NO | YES | NO |

### Key files:
- `cortical/cdg/storage.py` - Main storage, has the bugs
- `cortical/cdg/transaction_manager.py` - Commit logic
- `cortical/cdg/config.py` - DurabilityMode enum
- `cortical/cdg/wal.py` - WAL fsync logic
- `tests/unit/cdg/test_cdg_durability.py` - Durability tests

### User guidance:
- Go slow, one step at a time
- Explain thinking before making changes
- This is a sensitive area - need behavioral tests
- Use scratchpad to track work

---

## CURRENT FOCUS: DurabilityMode Analysis

We need to understand how DurabilityMode SHOULD work vs how it DOES work.

---

## DurabilityMode: HOW IT SHOULD WORK

For a transactional system, durability controls when data survives a crash:

| Mode | Meaning | When to use |
|------|---------|-------------|
| PARANOID | fsync every write immediately | Maximum safety, slowest |
| BALANCED | fsync on commit only | Good safety, recommended |
| FAST | no fsync | Maximum speed, data loss on crash |

---

## DurabilityMode: HOW IT ACTUALLY WORKS (BUGS FOUND)

### Current behavior matrix:

| Component | PARANOID | BALANCED | FAST/RELAXED |
|-----------|----------|----------|--------------|
| WAL entry (wal.py:196) | fsync | flush only | flush only |
| WAL on commit (wal.fsync_now) | called | called | called |
| Entity write (storage.py:838) | fsync | **fsync** | no fsync |
| fsync_all post-commit (tx_mgr:374) | NO | YES | NO |

### BUGS:

1. **BALANCED double-fsyncs entities:**
   - storage.py:838 fsyncs each entity write (because not FAST/RELAXED)
   - THEN transaction_manager:374 calls fsync_all() again
   - This is wasteful and wrong!

2. **BALANCED should NOT fsync per-write:**
   - storage.py:838 checks `not in (FAST, RELAXED)`
   - This means PARANOID and BALANCED both fsync per-write
   - BALANCED should only fsync at commit time!

3. **storage.py check is wrong:**
   ```python
   # CURRENT (wrong):
   if self.durability not in (DurabilityMode.FAST, DurabilityMode.RELAXED):
       self._fs.fsync(path)

   # SHOULD BE:
   if self.durability == DurabilityMode.PARANOID:
       self._fs.fsync(path)
   ```

4. **FAST and RELAXED are duplicates:**
   - Two enum values: FAST="fast", RELAXED="relaxed"
   - Identical behavior everywhere
   - Confusing, should be one or the other

---

## WHAT NEEDS TO CHANGE:

### 1. storage.py (4 locations)
Change fsync checks to only fsync for PARANOID:
```python
# Lines 838, 1013, 1045: Change from:
if self.durability not in (DurabilityMode.FAST, DurabilityMode.RELAXED):
# To:
if self.durability == DurabilityMode.PARANOID:

# Line 915: Change from:
if self.durability in (DurabilityMode.FAST, DurabilityMode.RELAXED):
    return
# To:
if self.durability != DurabilityMode.PARANOID:
    return
```

### 2. transaction_manager.py (line 339)
WAL fsync on commit should respect durability mode:
```python
# Currently (unconditional):
if self.wal:
    self.wal.fsync_now()

# Should be:
if self.wal and self.config.durability != DurabilityMode.RELAXED:
    self.wal.fsync_now()
```

### 3. Remove FAST enum value
- Keep RELAXED (more descriptive of what it means)
- Remove FAST from DurabilityMode
- Update any code using FAST to use RELAXED
- Update CDGConfig.for_high_performance() to use RELAXED

### 4. Update behavioral tests
- Verify PARANOID fsyncs every write
- Verify BALANCED only fsyncs at commit
- Verify RELAXED never fsyncs

---

## CORRECT BEHAVIOR AFTER FIX:

| Component | PARANOID | BALANCED | RELAXED |
|-----------|----------|----------|---------|
| WAL entry | fsync | flush | flush |
| WAL on commit | fsync | fsync | NO |
| Entity write | fsync | NO | NO |
| fsync_all post-commit | NO | YES | NO |

---

## CDGStore CONFIG ANALYSIS (storage.py)

### Config fields CDGStore ACTUALLY USES:

| Field | How Used | Location |
|-------|----------|----------|
| `durability` | `self.durability = self.config.durability` | Line 166 |
| `validate_on_write` | `self.validate_on_save = self.config.validate_on_write` | Line 167 |

That's it. Only 2 fields from CDGConfig are used by CDGStore.

### Config fields CDGStore IGNORES:

| Field | What Happens Instead |
|-------|---------------------|
| `read_cache_enabled` | Constructor has separate `cache_enabled=True` parameter |
| `read_cache_max_items` | `_cache_max_size` initialized to `None` (unlimited) |
| `partition_count` | Not referenced |
| `partition_strategy` | Not referenced |
| `isolation_level` | Not referenced in storage.py |
| `transaction_timeout_seconds` | Not referenced |
| `enable_wal` | Not referenced in storage.py (used elsewhere?) |
| `wal_archive_*` | Not referenced |
| `recovery_mode` | Not referenced in storage.py |
| `orphan_strategy` | Not referenced in storage.py |
| `auto_recover_on_startup` | Not referenced in storage.py |
| `enable_history` | Not referenced in storage.py |
| `history_retention_days` | Not referenced |
| `compression_enabled` | Not referenced |
| `encryption_enabled` | Not referenced |
| `super_node_*` | Not referenced |
| `write_buffer_size` | Not referenced |
| `strict_edge_types` | Not referenced in storage.py |
| `transactions_enabled` | Not referenced in storage.py |

### THE CONFUSION:

1. **Cache config disconnect:**
   - CDGConfig has: `read_cache_enabled=True`, `read_cache_max_items=10000`
   - CDGStore constructor has: `cache_enabled=True` parameter (separate!)
   - CDGStore initializes: `_cache_max_size=None` (ignores config!)
   - These don't connect!

2. **Many config fields are placeholders:**
   - CDGConfig defines 23+ fields
   - CDGStore only uses 2 of them
   - This creates false expectations

3. **Validation schema registry:**
   - Injected via constructor, not config
   - Validation runs if schema is registered for entity type
   - This is correct, not "optional"

---

## QUESTIONS TO ANSWER:

1. Where are the other config fields used? (WAL, transactions, recovery)
2. Should unused config fields be removed from CDGConfig?
3. Should cache config be wired to CDGStore?
4. Are there bugs from this disconnect?

---

## TRACE RESULTS: Where Config Fields Are Used Across CDG

### USED (in other CDG files, not storage.py):

| Field | Where Used | File:Line |
|-------|------------|-----------|
| `enable_wal` | WAL creation | transaction_manager.py:128, recovery.py:107 |
| `recovery_mode` | Recovery behavior | recovery.py:124, 179, 199, 203 |
| `orphan_strategy` | Orphan handling | recovery.py:251, 663, 675 |
| `auto_recover_on_startup` | Startup recovery | transaction_manager.py:139 |

### NOT USED ANYWHERE IN CDG (pure placeholders):

| Field | Status |
|-------|--------|
| `enable_history` | NOT USED |
| `strict_edge_types` | NOT USED |
| `isolation_level` | NOT USED (only SNAPSHOT implemented) |
| `partition_count` | NOT USED |
| `partition_strategy` | NOT USED |
| `super_node_*` | NOT USED |
| `write_buffer_size` | NOT USED |
| `read_cache_enabled` | NOT USED (separate constructor param) |
| `read_cache_max_items` | NOT USED |
| `wal_archive_enabled` | NOT USED |
| `wal_archive_threshold` | NOT USED |
| `history_retention_days` | NOT USED |
| `compression_enabled` | NOT USED |
| `encryption_enabled` | NOT USED |
| `transaction_timeout_seconds` | NOT USED |
| `transactions_enabled` | Only validated in config.py:201, not used |

### SUMMARY:

- **storage.py uses:** durability, validate_on_write (2 fields)
- **transaction_manager.py uses:** enable_wal, auto_recover_on_startup (2 fields)
- **recovery.py uses:** enable_wal, recovery_mode, orphan_strategy (3 fields)
- **Total USED:** 6 fields
- **Total PLACEHOLDER:** 16+ fields

---

## NEXT STEPS:

- [ ] Decide: remove placeholder config fields OR document as "future"
- [ ] Decide: wire cache config OR remove from CDGConfig

---

## KEY CLARIFICATIONS FROM USER

- **GoT should NOT know about storage details** - CDG handles all storage
- **CDG's DurabilityMode is correct** - GoT's version should be removed/aliased
- **One question at a time** - iterate through decisions
- **Schema validation is configurable at schema level** - not "optional"
- **Reduce confusion** - code should do what it says

---

## DECISIONS MADE

- [x] CDG durability is the correct enum
- [x] GoT should not know storage details
- [x] DurabilityMode consolidated: GoT re-exports CDG's version
- [x] Added RELAXED alias to CDG's DurabilityMode for backward compatibility
- [x] CDG storage updated to treat FAST and RELAXED as equivalent (no fsync)

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O

---

## CRITICAL PRINCIPLES (DO NOT REMOVE)

### CDG Architecture Philosophy
- CDG = clean, generic solution built on first principles
- **NO TWO LAYERS** - don't maintain wrappers
- When dealing with GoT, **ALWAYS consider moving functionality DOWN to CDG**
- Prefer refactoring GoT into CDG over maintaining two versions

### What GoT Uniquely Provides (keep in GoT)
1. **Entity types** - Task, Decision, Edge, Sprint, etc. (domain models)
2. **Entity factory** - `create_entity_from_dict()` dispatches to correct type
3. **QueryIndexManager** - GoT-specific indexing
4. **GoTManager** - high-level domain API
5. **Domain logic** - orphan detection, etc.

### Design Decisions
1. No backward compatibility - fix directly, no fallbacks
2. Container-first - DI for all dependencies
3. CDG is foundation - GoT is thin domain layer
4. Required parameters - no fallbacks
5. Centralized configuration

### Architectural Insights (from user)
- **Durability Mode**: GoT has NO business with durability - configured centrally in CDG
- **got/expression/***: Half-baked but FANTASTIC ideas, will need schema in GENERAL way
- **got/cli/* Pattern**: Container as member variable, DO NOT import bootstrap in functions

---

## SESSION CONTINUATION PROCEDURE

### For Humans: Creating a Continuation Prompt

```markdown
Continue refactoring work on the Cortical codebase.

**Previous session branch:** `claude/refactor-cortical-codebase-OZ8em`

**First Step:** Merge previous work into your session branch:

git fetch origin claude/refactor-cortical-codebase-OZ8em
git merge origin/claude/refactor-cortical-codebase-OZ8em

Then read the scratchpad:
cat docs/sessions/file-access-audit-scratchpad.md
```

### For AI Agents: What to Do in a New Session

```bash
# 1. You're already on your session-assigned branch - STAY ON IT

# 2. Fetch and merge the previous session's work
git fetch origin claude/refactor-cortical-codebase-OZ8em
git merge origin/claude/refactor-cortical-codebase-OZ8em

# 3. Verify you have the previous work
git log --oneline -5

# 4. Read the scratchpad
cat docs/sessions/file-access-audit-scratchpad.md

# 5. Work normally, push to YOUR session branch
git push -u origin HEAD
```

### When Done

Push to YOUR session branch and update this scratchpad's branch name above.

---
