# GoT Merge Conflict Root Cause Analysis

**Task:** T-20251226-114231-675e2be0
**Date:** 2025-12-26
**Status:** Complete

## Summary

GoT entity files contain **operation timestamps** instead of **semantic timestamps**, causing merge conflicts when the same entity is updated on different branches. The conflict sources are:

1. `_written_at` - File write timestamp (always differs)
2. `modified_at` - Entity modification timestamp (always differs)
3. `metadata.updated_at` - Task update timestamp (always differs)
4. `version` - Integer counter (conflicts when same entity updated)
5. `_checksum` - Derived from above (conflicts if any differ)

## Root Cause Details

### Current Entity Structure

```json
{
  "_checksum": "57a85a2d2b9d4d36",    // ← Conflicts: derived from content
  "_written_at": "2025-12-24T09:14:50", // ← Conflicts: write time
  "data": {
    "modified_at": "2025-12-24T09:14:50", // ← Conflicts: mod time
    "metadata": {
      "updated_at": "2025-12-24T09:14:50" // ← Conflicts: update time
    },
    "version": 4                         // ← Conflicts: integer counter
    // ... semantic content ...
  }
}
```

### Conflict Scenario

1. Branch A updates task → `_written_at: T1`, `version: 5`
2. Branch B updates same task → `_written_at: T2`, `version: 5`
3. **Git conflict:** Both have `version: 5` but different timestamps

### Why This Matters

- Every entity update on any branch changes 4+ timestamp fields
- Parallel agent work (different branches) guarantees conflicts
- Even non-conflicting semantic changes conflict due to timestamps
- The version counter makes Last-Writer-Wins impossible

## Solution Options

### Option 1: Vector Clocks (Recommended)

Replace integer version with agent-specific counters:

```json
{
  "version": {"agent_A": 3, "agent_B": 2, "agent_C": 1}
}
```

**Pros:**
- Merge = union of clocks (no conflicts)
- Tracks causal ordering
- Detects true concurrent updates

**Cons:**
- Slightly more storage
- Requires agent identification

### Option 2: Content-Addressable Storage

Move to CAS-based storage where ID = hash(semantic_content):

```
.got/content/abc123.json  # Content by hash
.got/refs/T-20251226-xxx  # Points to abc123
```

**Pros:**
- Identical content = identical ID (natural dedup)
- No timestamp conflicts in content files
- Git-native (matches git object model)

**Cons:**
- Major architectural change
- References need management

### Option 3: Separate Timestamps from Content

Move operational metadata to WAL-only:

```json
// Entity file (merge-friendly)
{
  "id": "T-20251226-xxx",
  "title": "Task",
  "status": "pending",
  "priority": "high"
  // No timestamps, no version, no checksum
}

// WAL (append-only, never conflicts)
{"op": "WRITE", "entity_id": "T-...", "timestamp": "...", "agent": "A"}
```

**Pros:**
- Entity files become merge-trivial
- Timestamps preserved in WAL
- Minimal code change

**Cons:**
- Can't verify entity integrity without WAL
- Loses checksum protection

### Option 4: Lamport Timestamps

Use logical time instead of wall clock:

```json
{
  "lamport_time": 42,  // Monotonic logical clock
  "agent_id": "A"
}
```

**Pros:**
- Monotonic (never decreases)
- Agent-scoped (no conflicts between agents)
- Establishes happens-before ordering

**Cons:**
- Requires clock sync protocol
- All agents must participate

## Recommendation

**Short-term (Sprint 22):** Option 3 - Remove timestamps from entity files
- Lowest risk, fastest to implement
- Entity files become merge-friendly
- WAL already captures timestamps

**Long-term (Future Sprint):** Option 1 - Vector Clocks
- Proper distributed system semantics
- Enables conflict detection and resolution
- Required for true multi-agent parallel work

## Implementation Notes

### For Option 3 (Short-term)

1. Modify `VersionedStore.write_entity()`:
   - Remove `_checksum` and `_written_at` from file
   - Store only semantic data

2. Modify `VersionedStore._read_and_verify()`:
   - Don't require checksum
   - Rely on git for integrity

3. Keep WAL entries unchanged:
   - Still record timestamps, versions, checksums
   - Used for recovery and audit trail

4. Migration:
   - Run cleanup to strip operational fields
   - Recompute entity files on load

### Affected Files

- `cortical/got/versioned_store.py` - Primary changes
- `cortical/got/tx_manager.py` - Version tracking
- `cortical/got/recovery.py` - Integrity checks
- `cortical/got/types.py` - Entity definitions (version field)

## Related Decisions

- D-20251223-010608-a8485ba7: "Use Option 3: Remove _version.json for merge-conflict-free GoT"
  - Already addressed the `_version.json` file
  - This analysis extends to per-entity timestamps

## Next Steps

1. Create implementation task for Option 3
2. Write tests for merge scenarios
3. Implement timestamp removal
4. Update recovery to not require checksums
5. Investigate vector clocks for future sprint
