# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## CURRENT FOCUS: Configuration Consolidation

Investigating implementation status of CDGConfig fields before making decisions.

---

## KEY CLARIFICATIONS FROM USER

- **GoT should NOT know about storage details** - CDG handles all storage
- **CDG's DurabilityMode is correct** - GoT's version should be removed/aliased
- **One question at a time** - iterate through decisions, potentially implement missing features

---

## INVESTIGATION RESULTS (completed)

| Feature | Status | Notes |
|---------|--------|-------|
| partition_count/strategy | NOT IMPLEMENTED | Placeholder only |
| isolation_level | PARTIAL | Only SNAPSHOT works, READ_COMMITTED is placeholder |
| transaction_timeout_seconds | NOT IMPLEMENTED | Placeholder only |
| wal_archive_enabled/threshold | NOT IMPLEMENTED | Manual truncate_before() exists |
| history_retention_days | NOT IMPLEMENTED | No cleanup mechanism |
| super_node_* thresholds | NOT IMPLEMENTED | Placeholder only |
| encryption_enabled | NOT IMPLEMENTED | Correct - should not be |
| read_cache_max_items | IMPLEMENTED BUT NOT WIRED | Cache works, config ignored |
| write_buffer_size | NOT IMPLEMENTED | Placeholder only |

**Summary:** Most CDGConfig "tuning" fields are placeholders. Core functionality works:
- durability, validate_on_write, strict_edge_types ✓
- transactions_enabled, enable_wal, recovery_mode ✓
- orphan_strategy, auto_recover_on_startup, enable_history ✓

---

## DECISIONS MADE

- [x] CDG durability is the correct enum
- [x] GoT should not know storage details
- [x] DurabilityMode consolidated: GoT re-exports CDG's version
- [x] Added RELAXED alias to CDG's DurabilityMode for backward compatibility
- [x] CDG storage updated to treat FAST and RELAXED as equivalent (no fsync)

---

## DECISIONS PENDING (one at a time, after investigation)

- [ ] Which CDGConfig fields to keep in unified config
- [ ] How to structure paths
- [ ] Test container setup

---

## COMPLETED THIS SESSION

1. Fixed `use_memory=True` test isolation bug
2. Extracted CDG durability tests from GoT test file
3. Analyzed CDGConfig (23 fields) and GoTConfig (2 versions)
4. Mapped all path configurations

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O
