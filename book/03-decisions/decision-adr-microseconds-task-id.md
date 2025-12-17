---
title: "ADR-001: Add Microseconds to Task ID Generation"
generated: "2025-12-17T00:26:16.589349Z"
generator: "decisions"
source_files:
  - "samples/decisions/adr-microseconds-task-id.md"
tags:
  - decisions
  - adr
  - adr-microseconds-task-id
---

# ADR-001: Add Microseconds to Task ID Generation

**Status:** Accepted  
**Date:** 2025-12-14  
**Tags:** `task-management`, `uniqueness`, `concurrency`  

---

## The Question

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

## The Conversation

*This decision emerged from 12 recorded discussion(s).*

### Discussion 1

**When:** 2025-12-16  
**Matched Keywords:** task, add  

**Query:**

> Please deeply think about the best way to implement these tasks in batches dispatched inteligently to sub Agents and tracked/verifyed by enabling director mode with a backup plan that covers potential failure points intelligently.

Tasks:
T-002	Document ML milestone thresholds derivation	Low
T-003	Make CSV export truncation configurable	Low
T-004	Refactor session_logger.py duplication	Low
T-010	Implement confidence scoring thresholds	Medium
T-011	Add semantic similarity to ML predictions	Low
T-0

**Files Explored:** `tasks/*.json`, `scripts/task_utils.py`

### Discussion 2

**When:** 2025-12-16  
**Matched Keywords:** task, add  

**Query:**

> Please deeply think about the best way to implement these tasks in batches dispatched inteligently to sub Agents and tracked/verifyed by enabling director mode with a backup plan that covers potential failure points intelligently.

Tasks:
T-002	Document ML milestone thresholds derivation	Low
T-003	Make CSV export truncation configurable	Low
T-004	Refactor session_logger.py duplication	Low
T-010	Implement confidence scoring thresholds	Medium
T-011	Add semantic similarity to ML predictions	Low
T-0

**Files Explored:** `scripts/task_utils.py`, `tasks/*.json`, `/home/user/Opus-code-test/tests/unit/test_ml_export.py`

### Discussion 3

**When:** 2025-12-16  
**Matched Keywords:** task, add  

**Query:**

> Please deeply think about the best way to implement these tasks in batches dispatched inteligently to sub Agents and tracked/verifyed by enabling director mode with a backup plan that covers potential failure points intelligently.

Tasks:
T-002	Document ML milestone thresholds derivation	Low
T-003	Make CSV export truncation configurable	Low
T-004	Refactor session_logger.py duplication	Low
T-010	Implement confidence scoring thresholds	Medium
T-011	Add semantic similarity to ML predictions	Low
T-0

**Files Explored:** `tasks/*.json`, `/home/user/Opus-code-test/tests/unit/test_ml_export.py`, `scripts/ml_data_collector.py`, `scripts/task_utils.py`

## Options Considered

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

## The Decision

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

## In Hindsight

*This decision has been referenced in 2 subsequent commit(s).*

- **2025-12-14** (`53c7985`): test: Update task ID format test to expect microseconds
- **2025-12-14** (`5970006`): fix: Add microseconds to task ID to prevent collisions

---

*This decision story was enriched with conversation context from 12 chat session(s). Source: [adr-microseconds-task-id.md](../../samples/decisions/adr-microseconds-task-id.md)*
