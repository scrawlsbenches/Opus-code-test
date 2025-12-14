# ADR-001: Add Microseconds to Task ID Generation

**Status:** Accepted
**Date:** 2025-12-14
**Deciders:** Development team
**Tags:** `task-management`, `uniqueness`, `concurrency`

---

## Context and Problem Statement

Task IDs were generated with second-precision timestamps plus a 4-character hex suffix:

```
T-YYYYMMDD-HHMMSS-XXXX
Example: T-20251214-163052-a1b2
```

During CI testing, the `test_unique_task_ids` test was intermittently failing:

```
AssertionError: 99 != 100
```

When generating 100 task IDs in a tight loop (same second), collisions occurred because:
- Same timestamp for all IDs in that second
- Only 4 hex chars = 65,536 possible suffixes
- Birthday paradox: P(collision) ≈ 7% for 100 items from 65,536

## Decision Drivers

1. **Reliability**: Tests must not be flaky
2. **Uniqueness**: Task IDs must be unique even under concurrent generation
3. **Backwards Compatibility**: Existing task IDs shouldn't break
4. **Readability**: IDs should remain human-readable

## Considered Options

### Option 1: Increase Random Suffix Length

```
T-YYYYMMDD-HHMMSS-XXXXXXXX  (8 hex chars)
```

**Pros:**
- Simple change
- 4 billion possibilities per second

**Cons:**
- Longer IDs
- Doesn't leverage timestamp ordering

### Option 2: Add Microseconds to Timestamp

```
T-YYYYMMDD-HHMMSSffffff-XXXX
Example: T-20251214-163052123456-a1b2
```

**Pros:**
- Timestamps remain sortable
- 1 million unique timestamps per second
- Combined with 4 hex suffix = practically unlimited uniqueness

**Cons:**
- IDs are 6 characters longer
- Existing code parsing IDs needs update

### Option 3: Use UUID Only

```
T-a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Pros:**
- Guaranteed uniqueness
- Standard format

**Cons:**
- Not human-readable
- Loses temporal ordering
- Much longer

## Decision Outcome

**Chosen Option:** Option 2 - Add Microseconds to Timestamp

**Rationale:**
- Preserves temporal ordering (IDs sort chronologically)
- Microseconds provide 1M unique slots per second
- Combined with 4 hex chars: virtually collision-proof
- Minimal change to existing format

## Implementation

```python
def generate_task_id(session_id: Optional[str] = None) -> str:
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S%f")  # Added %f for microseconds
    suffix = session_id or generate_session_id()
    return f"T-{date_str}-{time_str}-{suffix}"
```

## Consequences

### Positive
- Tests no longer flaky
- IDs unique even under heavy concurrent generation
- Temporal ordering preserved

### Negative
- IDs 6 characters longer
- Tests checking ID format needed update

### Neutral
- Existing IDs continue to work (no migration needed)
- No performance impact

## Validation

```python
# Verify uniqueness with 1000 IDs
ids = {generate_task_id() for _ in range(1000)}
assert len(ids) == 1000  # All unique
```

## Related Decisions

- Task management system design (LEGACY-047)
- Merge-friendly task format (docs/merge-friendly-tasks.md)

---

*This decision was made after a flaky test exposed the birthday paradox collision probability.*
