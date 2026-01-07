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

## NEXT STEPS:

- [ ] Trace where enable_wal, transactions_enabled, recovery_mode are used
- [ ] Decide: remove placeholder config fields OR implement them
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
