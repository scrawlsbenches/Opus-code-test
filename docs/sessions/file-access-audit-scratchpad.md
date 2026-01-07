# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## CURRENT FOCUS: CDGStore Configuration Confusion

We are confused about what CDGStore actually uses from CDGConfig.
We need to understand exactly what is implemented before making changes.

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
