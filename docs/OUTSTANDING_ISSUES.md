# Outstanding Issues Tracker

*Last Updated: 2026-01-05*
*Sprint: S-20260105-114800-bd32e373 (Expression System Phase 2)*

---

## Issue Summary

| ID | Priority | Category | Status | Description |
|----|----------|----------|--------|-------------|
| OI-001 | HIGH | Hardcoded Values | **✅ Resolved** | Status strings: all files now use get_valid_statuses() |
| OI-002 | HIGH | Hardcoded Values | **✅ Resolved** | EdgeTypes class created, all files now use EdgeTypes constants |
| OI-003 | MEDIUM | Hardcoded Values | Open | Magic numbers in filter functions |
| OI-004 | MEDIUM | Hardcoded Values | Open | Learning thresholds not configurable |
| OI-005 | MEDIUM | Test Coverage | Open | Some graph functions lack unit tests |
| OI-006 | LOW | Schema Gap | Open | Task categories not in schema |
| OI-007 | LOW | Inconsistency | **Resolved** | CLI edge.py now uses authoritative VALID_EDGE_TYPES |

---

## Detailed Issues

### OI-001: Status Strings Duplicated Across Files

**Priority:** HIGH
**Category:** Hardcoded Values
**Status:** ✅ Resolved
**Task:** T-20260105-114909-68226531

**Description:**
Task status strings (`"pending"`, `"in_progress"`, `"completed"`, `"blocked"`) are hardcoded in 8+ locations instead of being pulled from schema.

**RESOLVED (2026-01-05):**
- ✅ Created `get_valid_statuses()` helper in `cortical/got/entity_schemas.py`
- ✅ Updated `cortical/got/validation.py` to use helper (4 locations)
- ✅ Fixed schema/validation discrepancies:
  - Sprint: 'on_hold' → 'blocked' (aligned with schema)
  - Epic: Removed 'archived' (not in schema)
  - Handoff CLI: Removed 'in_progress' (not in schema)
- ✅ Updated `cortical/got/recovery.py` to use `get_valid_statuses('task')`
- ✅ Updated `cortical/got/cli/backlog.py` to use `get_valid_statuses('task')`
- ✅ Updated `cortical/got/cli/orphan.py` to use `get_valid_statuses('task')`

**INTENTIONALLY NOT CHANGED:**
- `cortical/got/query_api.py` (lines 376-387): Status names are part of the documented API contract
- `cortical/got/types.py` (lines 343-524): Dataclass `__post_init__` validation is foundational and should not depend on higher-level schema module

**Design Principle:** Now satisfied - all dynamic lookups use schema helper

---

### OI-002: Edge Types Scattered Across Files

**Priority:** HIGH
**Category:** Hardcoded Values
**Status:** ✅ Resolved
**Task:** T-20260105-114916-15adf67e

**Description:**
Edge type strings (`"DEPENDS_ON"`, `"BLOCKS"`, `"CONTAINS"`) are hardcoded in multiple files instead of using centralized constants.

**RESOLVED (2026-01-05):**
- ✅ Created `EdgeTypes` constants class in `cortical/got/types.py`
- ✅ Exported `EdgeTypes` and `VALID_EDGE_TYPES` from `cortical/got/__init__.py`
- ✅ Fixed `cortical/got/cli/edge.py`: Removed duplicate 35-item list, imports from types.py
- ✅ Eliminated 17 phantom edge types that were in CLI but not authoritative
- ✅ Updated `cortical/got/expression/functions/graph.py` (8 occurrences)
- ✅ Updated `cortical/got/query_api.py` (4 occurrences)
- ✅ Updated `cortical/got/cli/decision.py` (4 occurrences)
- ✅ Updated `cortical/got/api.py` (6 occurrences)
- ✅ Updated `cortical/got/indexer.py` (1 occurrence)
- ✅ Updated `cortical/got/expression/functions/filters.py` (1 occurrence)
- ✅ Updated `cortical/got/orphan.py` (1 occurrence)

**CONTINUATION SESSION (2026-01-05):**
Validation discovered additional occurrences missed in initial resolution:
- ✅ Updated `cortical/got/orphan.py` (3 additional occurrences - lines 451, 456, 459)
- ✅ Updated `cortical/got/cli/analyze.py` (8 occurrences - lines 130, 187, 222, 255, 292, 294, 317, 319)

**All files now use `EdgeTypes` constants instead of string literals.**

**Usage Example:**
```python
from cortical.got.types import EdgeTypes

if edge.edge_type == EdgeTypes.DEPENDS_ON:
    # ...
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
**Status:** ✅ Resolved

**Description:**
Different CLI commands show different valid edge type lists, indicating multiple hardcoded lists exist.

**Resolution (2026-01-05):**
The duplicate `VALID_EDGE_TYPES` list in `cortical/got/cli/edge.py` (35 items with phantom types) was removed and replaced with an import from the authoritative source:

```python
from cortical.got.types import VALID_EDGE_TYPES
```

Now all CLI commands use the same 22 authoritative edge types from `cortical/got/types.py`.

**Phantom Types Removed:**
ENABLES, CONFLICTS, SUPPORTS, REFUTES, SIMILAR, TESTS, PRECEDES, TRIGGERS, ANSWERS, RAISES, EXPLORES, OBSERVES, SUGGESTS, REFINES, HAS_OPTION, HAS_ASPECT, LOCATED_IN

---

## Recently Resolved Issues

### RESOLVED: Schema/Validation Discrepancies & Status Helper

**Resolved:** 2026-01-05
**Session:** Senior Engineering Consultation

**Issues Fixed:**
1. **Epic validation** accepted 'archived' status not in schema → Removed
2. **Sprint validation** accepted 'on_hold' but schema has 'blocked' → Fixed
3. **Handoff CLI** accepted 'in_progress' not in schema → Removed

**Infrastructure Created:**
- `get_valid_statuses(entity_type)` helper in `cortical/got/entity_schemas.py`
- Updated `cortical/got/validation.py` to use helper (4 locations)

**Verification:** `python3 scripts/got_utils.py validate` → HEALTHY

---

### RESOLVED: EdgeTypes Constants Class & CLI Consolidation

**Resolved:** 2026-01-05
**Session:** Senior Engineering Consultation

**Issues Fixed:**
1. **Created `EdgeTypes` class** in `cortical/got/types.py` with 22 constants
2. **Removed duplicate list** in `cortical/got/cli/edge.py` (had 35 phantom types)
3. **Added exports** to `cortical/got/__init__.py`

**Phantom Types Eliminated:** 17 types (ENABLES, CONFLICTS, SUPPORTS, etc.)

**Usage Example:**
```python
from cortical.got import EdgeTypes
if edge.edge_type == EdgeTypes.DEPENDS_ON:
```

**Verification:** All imports work, `len(VALID_EDGE_TYPES) == 22`

---

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
