# Knowledge Transfer: GoT Recovery Orphan Repair Fix

**Date:** 2025-12-22
**Session ID:** 9iHi6
**Branch:** claude/investigate-open-tasks-9iHi6
**Tags:** `got`, `recovery`, `orphan-repair`, `critical-fix`, `wal`, `git-tracked`

## Summary

Fixed a critical bug where the GoT recovery system was deleting all git-tracked entity files on every startup. Changed the orphan repair strategy from 'delete' to 'adopt' to preserve valid data.

## The Problem

### Architecture Context

```
.got/entities/  → GIT-TRACKED (entity storage for TX backend)
.got/wal/       → LOCAL ONLY (gitignored - transaction WAL)
```

### Root Cause

When files are checked out from git:
1. Entity files exist in `.got/entities/` (from git)
2. WAL entries don't exist (WAL is gitignored, not in git)
3. `RecoveryManager.detect_orphaned_entities()` sees files without WAL entries
4. `repair_orphans(strategy='delete')` deleted ALL of them

**Impact:** Every GoT command (`dashboard`, `task list`, etc.) wiped all tracked entities.

### Location

```python
# cortical/got/recovery.py:162 (before fix)
repair_result = self.repair_orphans(strategy='delete')
```

## The Fix

Changed strategy from 'delete' to 'adopt':

```python
# cortical/got/recovery.py:164 (after fix)
# Use 'adopt' strategy to preserve git-tracked files that lack WAL entries
# (files committed to git don't have WAL records, so 'delete' would wipe them)
repair_result = self.repair_orphans(strategy='adopt')
```

### How 'adopt' Works

1. Detects entities without WAL entries (orphans)
2. Validates checksums via `_read_and_verify()`
3. **Corrupted files** → DELETED (correct behavior)
4. **Valid files** → Creates synthetic ADOPTED entry in WAL
5. Future runs: entities have WAL entries, not detected as orphans

## Edge Cases Verified

| Scenario | 'delete' | 'adopt' |
|----------|----------|---------|
| Fresh git clone | ❌ Deletes all | ✅ Adopts all |
| WAL truncated | ❌ Deletes all | ✅ Re-adopts |
| Crash with invalid checksum | ✅ Deletes | ✅ Deletes |
| Crash with valid checksum | ❌ Loses valid data | ✅ Preserves |
| Multiple recovery runs | N/A | ✅ No WAL bloat |
| WAL integrity check | N/A | ✅ ADOPTED entries pass |

## Technical Details

### ADOPTED Entry Format

```json
{
  "op": "ADOPTED",
  "entity_id": "T-20251222-...",
  "reason": "orphan_recovery",
  "timestamp": 1766425561.134,
  "checksum": "4308b5f73cb7ea1d"
}
```

### WAL Entry Types

The WAL contains two entry formats:

1. **TransactionWALEntry** (for TX operations):
   - Format: `{seq, ts, tx, op, data, checksum}`
   - Used by: TX_BEGIN, TX_COMMIT, WRITE, etc.
   - Verified by: `WALManager.replay()`

2. **Simple entries** (for markers):
   - Format: `{op, entity_id, reason, timestamp, checksum}`
   - Used by: ADOPTED entries
   - Verified by: `verify_wal_integrity()` (generic checksum)
   - Recognized by: `detect_orphaned_entities()`

This is intentional - ADOPTED is not a transaction, just a tracking marker.

## Files Modified

| File | Changes |
|------|---------|
| `cortical/got/recovery.py` | Changed strategy 'delete' → 'adopt', updated log messages |
| `tests/unit/got/test_recovery.py` | Updated test to expect adopt behavior, added 4 edge case tests |

## Edge Case Tests Added

New `TestOrphanRepairEdgeCases` class with 4 tests:

1. **`test_fresh_clone_no_wal_file`** - Simulates fresh git clone (entities exist, no WAL)
2. **`test_multiple_recovery_runs_idempotent`** - Verifies no WAL bloat on repeated runs
3. **`test_adopted_entries_pass_wal_integrity`** - Validates ADOPTED checksums pass verification
4. **`test_wal_truncated_entities_readopted`** - Tests re-adoption after WAL truncation

## Commits

1. `8d474f80` - fix(got): Change orphan repair strategy from 'delete' to 'adopt'
2. `9fb42c77` - test(got): Update orphan repair test for 'adopt' strategy
3. `babd473f` - test(got): Add edge case tests for orphan repair adopt strategy

## Test Results

- 329 GoT tests pass
- Dashboard now shows: 23 nodes, 5 edges (previously 0)

## Related Tasks

- **T-20251222-180002-04659482**: Fix recovery orphan repair (COMPLETED)
- **T-20251222-145440-7fb36a5a**: Fix EdgeType enum (still needed for full edge loading)
- **T-20251222-145502-8edcf3a0**: Fix velocity metrics (still needed)

## Commands for Verification

```bash
# Verify entities persist across runs
ls .got/entities/T-*.json | wc -l  # Should show tasks
python scripts/got_utils.py task list  # Should show tasks
python scripts/got_utils.py dashboard  # Should show nodes > 0

# Check WAL for ADOPTED entries
cat .got/wal/current.wal | grep ADOPTED | wc -l

# Verify no orphans detected
python -c "
from cortical.got.recovery import RecoveryManager
from pathlib import Path
rm = RecoveryManager(Path('.got'))
print(f'Orphans: {len(rm.detect_orphaned_entities())}')
"
```

## Lessons Learned

1. **Git-tracked vs local state**: When storage is git-tracked but metadata (WAL) is local-only, recovery logic must account for legitimate files without metadata.

2. **Checksum validation is key**: The 'adopt' strategy still validates checksums, so corrupted crash orphans are deleted. Valid data is preserved.

3. **Test assumptions**: The original test assumed 'delete' was correct. Tests should verify the intended behavior, not just any behavior.

## Related Documentation

- `docs/graph-of-thought.md` - GoT framework overview
- `docs/graph-recovery-procedures.md` - Recovery procedures
- `samples/memories/2025-12-20-got-event-sourcing-knowledge-transfer.md` - Event-sourcing history
