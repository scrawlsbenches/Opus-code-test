# Knowledge Transfer: WAL Entity Rollback Implementation

**KT ID:** KT-20260103-013737
**Date:** 2026-01-03
**Task:** T-20260103-001507-4d3ad2bb
**Branch:** claude/review-engineering-handoff-2rJbF

---

## Summary

Implemented the ability to reconstruct entities from WAL after a crash occurs between TX_COMMIT and entity file writes. This closes a critical gap in the ACID guarantees where committed transaction data could be lost.

---

## Problem Statement

**Before this change:**
- WAL WRITE entries only stored: `{entity_id, old_version, new_version}`
- If crash occurred after TX_COMMIT but before entity files written:
  - WAL showed transaction as committed
  - But entity data was **lost** - no way to reconstruct

**After this change:**
- WAL WRITE entries store: `{entity_id, old_version, new_version, entity_data}`
- Recovery scans for committed transactions with missing entities
- Entities reconstructed from WAL's `entity_data` field

---

## Files Changed

| File | Change |
|------|--------|
| `cortical/cdg/wal.py:218-250` | Extended `log_write()` with optional `entity_data` parameter |
| `cortical/cdg/transaction_manager.py:195-234` | Pass `entity.to_dict()` to `log_write()` |
| `cortical/cdg/recovery.py:61,73,266-276,456-576` | Added `reconstruct_entities_from_wal()` method |
| `tests/behavioral/test_cdg_crash_recovery.py:472-810` | 5 new behavioral tests |
| `tests/unit/cdg/test_cdg_wal.py:126-168` | Fixed outdated sequence test |

---

## Technical Design

### WAL Entry Format

```json
{
  "seq": 2,
  "ts": "2026-01-03T00:00:00Z",
  "tx": "TX-001",
  "op": "WRITE",
  "data": {
    "entity_id": "E-001",
    "old_version": 0,
    "new_version": 1,
    "entity_data": {           // NEW: Full entity state
      "id": "E-001",
      "entity_type": "task",
      "version": 1,
      "properties": {...}
    }
  },
  "checksum": "..."
}
```

### Recovery Flow

```
1. Scan WAL entries
2. Track WRITE entries with entity_data per transaction
3. Track TX_COMMIT entries (committed transactions)
4. Discard writes for TX_ABORT/TX_ROLLBACK transactions
5. For each committed transaction:
   a. Check if entity file exists and is valid
   b. If missing/corrupted: reconstruct from entity_data
   c. Write entity with _reconstructed_from_wal flag
```

### Reconstruction Marker

Reconstructed entities include a marker in their wrapper:
```json
{
  "_checksum": "...",
  "_written_at": "...",
  "_reconstructed_from_wal": true,  // Audit trail
  "data": {...}
}
```

---

## Test Coverage

### New Behavioral Tests (TestEntityReconstructionFromWAL)

1. **test_scenario_missing_entity_reconstructed_from_wal** - Entity file missing after commit
2. **test_scenario_corrupted_entity_reconstructed_from_wal** - Entity file corrupted (truncated)
3. **test_scenario_multiple_entities_reconstructed_from_wal** - Batch reconstruction
4. **test_scenario_uncommitted_tx_entity_not_reconstructed** - Uncommitted tx rejected
5. **test_scenario_entity_reconstruction_via_transaction_manager** - End-to-end via TM

### Fixed Unit Test

- `test_next_seq_increments` → `test_next_seq_peeks_without_incrementing`
- Added `test_commit_seq_increments_and_persists`
- Added `test_log_properly_increments_sequence`

The old test assumed `_next_seq()` incremented the counter. The ACID fix changed it to peek-only (sequence committed after successful write).

---

## Related Tasks Updated

| Task ID | Title | Action |
|---------|-------|--------|
| T-20260102-170750-936c478c | CDG history saved before write | Marked **completed** (already fixed) |
| T-20260102-170719-b7d5a506 | Transaction rollback failure | Downgraded to **medium** (main risk mitigated) |

---

## Edge Cases Handled

1. **Deletions**: `new_version == -1` entries are skipped (no entity to reconstruct)
2. **Aborted transactions**: Writes discarded when TX_ABORT/TX_ROLLBACK seen
3. **Valid entities**: Existing valid entities are not overwritten
4. **Corruption detection**: Both JSON parse errors and checksum mismatches trigger reconstruction

---

## Performance Considerations

- **WAL size increase**: Entity data adds ~100-500 bytes per WRITE entry
- **Reconstruction cost**: O(n) scan of WAL on recovery only
- **Normal operation**: No additional overhead beyond WAL write

For large entities, consider future optimization: store entity hash in WAL, full data in separate recovery store.

---

## Future Work

1. **Exception handling improvement** (T-20260102-170719-b7d5a506): Add specific exception types for better diagnostics
2. **WAL compaction**: When WAL is truncated, entity_data is no longer needed for old transactions
3. **Large entity optimization**: For entities >10KB, consider separate recovery store

---

## Commits

```
cf26ba0e chore(got): Update CDG task statuses after WAL entity rollback
f351c88b fix(test): Update WAL sequence test for crash-safe behavior
5d67ddf2 feat(cdg): Implement WAL entity rollback for crash recovery
```

---

## Verification Commands

```bash
# Run crash recovery tests
python -m pytest tests/behavioral/test_cdg_crash_recovery.py -v

# Run all CDG tests
python -m pytest tests/behavioral/cdg_*.py tests/behavioral/test_cdg_*.py -v

# Run WAL unit tests
python -m pytest tests/unit/cdg/test_cdg_wal.py -v
```

---

## Key Insight

The WAL-first durability model now provides complete data protection:

```
TX_BEGIN → WRITE (with entity_data) → TX_COMMIT → [crash point] → entity write

If crash occurs at [crash point]:
- Before: Data lost (WAL has commit, but no entity data)
- After: Data recovered from WAL entity_data field
```

This completes the ACID guarantees for CDG transactions.
