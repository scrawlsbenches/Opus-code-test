# Outstanding Issues Tracker

*Last Updated: 2026-01-05*
*Sprint: S-20260105-114800-bd32e373 (Expression System Phase 2)*

---

## Issue Summary

| ID | Priority | Category | Status | Description |
|----|----------|----------|--------|-------------|
| OI-001 | HIGH | Hardcoded Values | Open | Status strings duplicated across 8+ files |
| OI-002 | HIGH | Hardcoded Values | Open | Edge types scattered across multiple files |
| OI-003 | MEDIUM | Hardcoded Values | Open | Magic numbers in filter functions |
| OI-004 | MEDIUM | Hardcoded Values | Open | Learning thresholds not configurable |
| OI-005 | MEDIUM | Test Coverage | Open | Some graph functions lack unit tests |
| OI-006 | LOW | Schema Gap | Open | Task categories not in schema |
| OI-007 | LOW | Inconsistency | Open | Edge type validation lists differ between CLI commands |

---

## Detailed Issues

### OI-001: Status Strings Duplicated Across Files

**Priority:** HIGH
**Category:** Hardcoded Values
**Status:** Open
**Task:** T-20260105-114909-68226531

**Description:**
Task status strings (`"pending"`, `"in_progress"`, `"completed"`, `"blocked"`) are hardcoded in 8+ locations instead of being pulled from schema.

**Affected Files:**
| File | Line(s) | Current Value |
|------|---------|---------------|
| `cortical/got/validation.py` | 402-404 | `{'pending', 'in_progress', 'completed', 'blocked'}` |
| `cortical/got/validation.py` | 442-444 | Sprint statuses hardcoded |
| `cortical/got/validation.py` | 454-456 | Epic statuses hardcoded |
| `cortical/got/validation.py` | 466-468 | Handoff statuses hardcoded |
| `cortical/got/recovery.py` | 212 | `["pending", "in_progress", "completed", "blocked"]` |
| `cortical/got/cli/backlog.py` | 280 | CLI choices hardcoded |
| `cortical/got/cli/orphan.py` | 305 | CLI choices hardcoded |
| `cortical/got/query_api.py` | 384-387 | Status names in aggregation |

**Recommended Fix:**
```python
# Create cortical/got/constants.py or extend schema.py
from cortical.got.entity_schemas import get_schema_for_entity_type

def get_entity_statuses(entity_type: str) -> set:
    schema = get_schema_for_entity_type(entity_type)
    return set(schema.fields['status'].choices)
```

**Design Principle Violated:**
> "NO HARDCODED ENTITIES - All graph functions use registry pattern"

---

### OI-002: Edge Types Scattered Across Files

**Priority:** HIGH
**Category:** Hardcoded Values
**Status:** Open
**Task:** T-20260105-114916-15adf67e

**Description:**
Edge type strings (`"DEPENDS_ON"`, `"BLOCKS"`, `"CONTAINS"`) are hardcoded in multiple files instead of using centralized constants.

**Affected Files:**
| File | Line(s) | Values |
|------|---------|--------|
| `cortical/got/validation.py` | 140-280 | All edge types duplicated |
| `cortical/got/indexer.py` | 406 | `"CONTAINS"` |
| `cortical/got/query_api.py` | 147, 184, 568, 571 | `"BLOCKS"`, `"DEPENDS_ON"`, `"CONTAINS"` |
| `cortical/got/expression/functions/filters.py` | 262 | `"BLOCKS"` |
| `cortical/got/expression/functions/graph.py` | Multiple | `"DEPENDS_ON"` throughout |

**Recommended Fix:**
```python
# Use existing VALID_EDGE_TYPES from cortical/got/types.py
from cortical.got.types import VALID_EDGE_TYPES

class EdgeTypes:
    DEPENDS_ON = "DEPENDS_ON"
    BLOCKS = "BLOCKS"
    CONTAINS = "CONTAINS"
    # ... etc
```

---

### OI-003: Magic Numbers in Filter Functions

**Priority:** MEDIUM
**Category:** Hardcoded Values
**Status:** Open

**Description:**
Default values for time-based filters are hardcoded instead of being configurable.

**Affected Files:**
| File | Line | Value | Purpose |
|------|------|-------|---------|
| `cortical/got/expression/functions/filters.py` | 43 | `days=7` | Recent tasks default |
| `cortical/got/expression/functions/filters.py` | 99 | `days=30` | Stale tasks default |

**Recommended Fix:**
```python
# Add to GoTConfig or create QueryConfig
@dataclass
class QueryConfig:
    default_recent_days: int = 7
    default_stale_days: int = 30
```

**Note:** These defaults are documented in docstrings, so impact is LOW. However, configurability would be preferred.

---

### OI-004: Learning Thresholds Not Configurable

**Priority:** MEDIUM
**Category:** Hardcoded Values
**Status:** Open

**Description:**
Similarity thresholds and limits in learning integration are hardcoded.

**Affected Files:**
| File | Line | Value | Purpose |
|------|------|-------|---------|
| `cortical/got/learning_integration.py` | 542 | `min_similarity=0.15` | Similar task search |
| `cortical/got/learning_integration.py` | 543 | `limit=5` | Max related tasks |
| `cortical/got/learning_integration.py` | 697-698 | `min_similarity=0.3, limit=10` | Blocking analysis |
| `cortical/got/learning_integration.py` | 67-69 | `MAX_TASK_ID_LENGTH=100` | Validation |
| `cortical/got/learning_integration.py` | 270, 388, 511 | `1000` | Max list sizes |

**Recommended Fix:**
```python
@dataclass
class LearningConfig:
    min_similarity_related: float = 0.15
    min_similarity_blocking: float = 0.3
    max_related_tasks: int = 5
    max_file_list_size: int = 1000
```

---

### OI-005: Graph Functions Lack Unit Tests

**Priority:** MEDIUM
**Category:** Test Coverage
**Status:** Partially Resolved

**Description:**
While behavioral tests now exist (20 tests added 2026-01-05), some functions still lack dedicated unit tests.

**Coverage Status:**
| Function | Behavioral Test | Unit Test |
|----------|-----------------|-----------|
| `ancestors()` | ✅ | ❌ |
| `descendants()` | ✅ | ❌ |
| `children()` | ✅ | ❌ |
| `parents()` | ✅ | ❌ |
| `all_dependencies()` | ✅ | ❌ |
| `cycle_detect()` | ✅ | ❌ |
| `dependents()` | ✅ | ❌ |
| `exists()` | ✅ | ❌ |
| `type_of()` | ✅ | ❌ |
| `entity_type()` | ❌ | ❌ |

**Note:** Behavioral tests validate user-facing behavior. Unit tests would validate edge cases and error handling more thoroughly.

---

### OI-006: Task Categories Not in Schema

**Priority:** LOW
**Category:** Schema Gap
**Status:** Open

**Description:**
Task categories (`feature`, `bugfix`, `test`, `refactor`, `docs`) are hardcoded in CLI but not defined in TaskSchema.

**Affected Files:**
| File | Line(s) | Issue |
|------|---------|-------|
| `cortical/got/cli/shared.py` | 46-60 | Categories hardcoded in choices |
| `cortical/got/entity_schemas.py` | N/A | No `category` field in TaskSchema |

**Recommended Fix:**
Add `category` field to TaskSchema with choices list, then reference in CLI.

---

### OI-007: Edge Type Validation Inconsistency

**Priority:** LOW
**Category:** Inconsistency
**Status:** Open

**Description:**
Different CLI commands show different valid edge type lists, indicating multiple hardcoded lists exist.

**Evidence:**
```
# From one command:
Valid types: BLOCKS, CAUSED_BY, CHILD_OF, CONTAINS, CONTINUES, CONTRADICTS,
             DEPENDS_ON, DERIVED_FROM, DOCUMENTED_BY, DOCUMENTS, FAILED_ATTEMPT,
             IMPLEMENTS, JUSTIFIES, MOTIVATES, PARENT_OF, PART_OF, PRODUCES,
             REFERENCES, RELATES_TO, REQUIRES, SUPERSEDES, TRANSFERS

# From another command:
Valid types: ANSWERS, BLOCKS, CAUSED_BY, CONFLICTS, CONTAINS, CONTRADICTS,
             DEPENDS_ON, ENABLES, EXPLORES, FAILED_ATTEMPT, HAS_ASPECT,
             HAS_OPTION, IMPLEMENTS, JUSTIFIES, LOCATED_IN, MOTIVATES,
             OBSERVES, PART_OF, PRECEDES, RAISES, REFINES, REFUTES,
             REQUIRES, SIMILAR, SUGGESTS, SUPPORTS, TESTS, TRIGGERS
```

**Root Cause:** Multiple edge type definitions exist in different files.

**Recommended Fix:** Consolidate to single source of truth in `cortical/got/types.py`.

---

## Recently Resolved Issues

### RESOLVED: Inverted Edge Semantics in Graph Functions

**Resolved:** 2026-01-05
**Commit:** cf7ba717

**Description:**
Graph traversal functions (`children`, `parents`, `ancestors`, `descendants`, `all_dependencies`) had inverted edge direction logic.

**Root Cause:**
Code comments and implementation conflicted. Comment said "follow edges backwards" but implementation did the opposite of correct behavior.

**Fix Applied:**
- `children()`: Now looks for edges where entity is TARGET, returns SOURCE
- `parents()`: Now looks for edges where entity is SOURCE, returns TARGET
- `ancestors()`: Now uses forward adjacency (source→target)
- `descendants()`: Now uses reverse adjacency (target→source)
- `all_dependencies()`: Now uses forward adjacency

**Verification:** 20 behavioral tests pass, 27 natural query tests pass.

---

### RESOLVED: Missing exists() and type_of() Functions

**Resolved:** 2026-01-05
**Commit:** cf7ba717

**Description:**
Design document specified `exists(entity_id)` and `type_of(entity_id)` but they were not implemented in the expression system.

**Fix Applied:**
- Implemented `exists()` - checks all entity stores
- Implemented `type_of()` - uses ID prefix convention with fallback

---

## Issue Workflow

### Creating New Issues

1. Assign ID: `OI-XXX` (next sequential number)
2. Set Priority: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`
3. Set Category: `Hardcoded Values`, `Test Coverage`, `Bug`, `Schema Gap`, `Inconsistency`, `Performance`
4. Link to GoT Task if applicable
5. Add to summary table

### Resolving Issues

1. Move to "Recently Resolved" section
2. Add resolution date and commit hash
3. Describe fix applied
4. Update summary table status

### Priority Definitions

| Priority | Definition | Response Time |
|----------|------------|---------------|
| CRITICAL | System broken, tests failing | Immediate |
| HIGH | Design principle violated, significant tech debt | Next sprint |
| MEDIUM | Improvement opportunity, minor inconsistency | Backlog |
| LOW | Nice to have, cosmetic | When convenient |

---

## Related Documents

- Design Document: `docs/design/got-query-audit-and-design.md`
- Future Enhancements: `docs/design/got-query-future-enhancements.md`
- Sprint: `S-20260105-114800-bd32e373`

---

*This document is maintained as part of the Expression System Phase 2 sprint.*
