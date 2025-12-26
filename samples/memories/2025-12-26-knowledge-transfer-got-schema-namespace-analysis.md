# Knowledge Transfer: GoT Schema & Namespace Deep Analysis

**Date:** 2025-12-26
**Session:** claude/analyze-got-schemas-namespaces-eDjPT
**Author:** Claude (Opus 4.5)
**Tags:** `got`, `schema`, `namespace`, `validation`, `architecture`, `technical-debt`

---

## Executive Summary

A comprehensive analysis of the Graph of Thought (GoT) system's schema and namespace utilization revealed a **well-architected but incompletely enforced** validation system. The analysis identified **critical validation gaps** that could allow invalid data to persist silently, and **namespace inconsistencies** that complicate maintenance.

**Key Outcomes:**
- Created Sprint S-026 "Schema Validation Hardening" under EPIC-got-db
- Created 7 new tasks addressing identified gaps
- Logged Decision D-20251226-141841-c387695a documenting the initiative
- No duplicate tasks created (cross-referenced with 16 existing pending tasks)

---

## 1. Schema Architecture Analysis

### 1.1 Three-Layer Schema System

The GoT system implements a sophisticated three-layer schema architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Schema Registry (schema.py)                            │
│   - Singleton pattern for schema management                     │
│   - Validation orchestration                                    │
│   - Migration support                                           │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Declarative Schemas (entity_schemas.py)                │
│   - BaseSchema subclasses                                       │
│   - Field definitions with validators                           │
│   - EDGE_TYPES list defined here (lines 230-234)                │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: Entity Dataclasses (types.py)                          │
│   - Task, Decision, Sprint, Epic, Edge, Handoff, etc.           │
│   - __post_init__ validation for some fields                    │
│   - to_dict()/from_dict() serialization                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Entity Types Defined

| Entity Type | Location | Lines | Key Fields |
|-------------|----------|-------|------------|
| Entity (base) | types.py | 18-81 | id, entity_type, version, created_at, modified_at |
| Task | types.py | 83-142 | title, status, priority, description, properties, metadata |
| Decision | types.py | 145-186 | title, rationale, affects, properties |
| Sprint | types.py | 189-255 | title, status, epic_id, number, goals, notes |
| Epic | types.py | 258-312 | title, status, phase, phases |
| Edge | types.py | 315-378 | source_id, target_id, edge_type, weight, confidence |
| Handoff | types.py | 381-465 | 16 fields including lifecycle timestamps |
| ClaudeMdLayer | types.py | 468-600 | layer_number, content, section_id |
| ClaudeMdVersion | types.py | 603-700 | layer_id, version_number, changes |
| Team | types.py | 703-850 | name, members, capabilities |
| PersonaProfile | types.py | 853-1000 | name, expertise, traits |
| Document | types.py | 1003-1142 | path, content, doc_type |

### 1.3 Validation Status by Entity

| Entity | Constructor Validation | Schema Validation | Status |
|--------|----------------------|-------------------|--------|
| Task | ✅ status, priority enums | ✅ Available | Working |
| Decision | ❌ No enum validation | ✅ Available | Partial |
| Sprint | ✅ status enum | ✅ Available | Working |
| Epic | ✅ status enum | ✅ Available | Working |
| Edge | ⚠️ weight/confidence only | ❌ edge_type NOT validated | **CRITICAL GAP** |
| Handoff | ✅ status enum | ✅ Available | Working |

---

## 2. Critical Gaps Identified

### 2.1 Gap #1: Edge Type Validation Missing (CRITICAL)

**Location:** `cortical/got/types.py:315-378`

**Problem:** The Edge class validates weight and confidence ranges but does NOT validate edge_type against the allowed values.

**Current Code (lines 338-350):**
```python
def __post_init__(self):
    # Weight validation
    if self.weight < 0.0 or self.weight > 1.0:
        raise ValidationError(f"weight must be in [0.0, 1.0], got {self.weight}")
    # Confidence validation
    if self.confidence < 0.0 or self.confidence > 1.0:
        raise ValidationError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
    # NO edge_type validation!
```

**Valid Edge Types (defined in entity_schemas.py:230-234):**
```python
EDGE_TYPES = [
    'DEPENDS_ON', 'BLOCKS', 'CONTAINS', 'RELATES_TO',
    'REQUIRES', 'IMPLEMENTS', 'SUPERSEDES', 'DERIVED_FROM',
    'PARENT_OF', 'CHILD_OF', 'REFERENCES', 'CONTRADICTS',
]
```

**Impact:** Any arbitrary string can be used as edge_type, leading to:
- Inconsistent edge type naming
- Query failures when filtering by edge_type
- Silent data corruption

**Fix Required:**
```python
VALID_EDGE_TYPES = {
    'DEPENDS_ON', 'BLOCKS', 'CONTAINS', 'RELATES_TO',
    'REQUIRES', 'IMPLEMENTS', 'SUPERSEDES', 'DERIVED_FROM',
    'PARENT_OF', 'CHILD_OF', 'REFERENCES', 'CONTRADICTS',
}

def __post_init__(self):
    if self.edge_type not in VALID_EDGE_TYPES:
        raise ValidationError(f"Invalid edge_type: {self.edge_type}")
    # ... existing validation
```

### 2.2 Gap #2: Validation Disabled by Default (CRITICAL)

**Location:** `cortical/got/versioned_store.py:49`

**Problem:** The VersionedStore has `validate_on_save=False` by default, meaning schema validation is opt-in rather than enforced.

**Current Code:**
```python
def __init__(self, ..., validate_on_save: bool = False):  # Default is False!
```

**Impact:**
- Invalid entities persist without detection
- Schema definitions exist but aren't enforced
- Data quality degrades over time

**Fix Required:**
```python
def __init__(self, ..., validate_on_save: bool = True):  # Change default to True
```

### 2.3 Gap #3: Nested Structures Unvalidated (MEDIUM)

**Problem:** Several entity fields accept `Dict[str, Any]` without structure validation:

| Field | Entities | Current Type | Should Be |
|-------|----------|--------------|-----------|
| properties | All | Dict[str, Any] | TypedDict with known keys |
| metadata | Task, Sprint, Epic | Dict[str, Any] | TypedDict with timestamps |
| context | Handoff | Dict[str, Any] | Structured context schema |
| result | Handoff | Dict[str, Any] | Structured result schema |
| goals | Sprint | List[Dict] | List[SprintGoal] |
| phases | Epic | List[Dict] | List[EpicPhase] |

**Example from Production:**
```json
{
  "properties": {
    "category": "feature",
    "retrospective": "Added ProcessLock...",
    "arbitrary_field": "anything goes"  // No validation!
  }
}
```

### 2.4 Gap #4: Entity Type Validation on Deserialize (MEDIUM)

**Location:** `cortical/got/versioned_store.py:514-549`

**Problem:** Deserialization dispatches on `entity_type` but doesn't validate it's a known type.

**Current Code:**
```python
def _entity_from_dict(self, data: dict) -> Entity:
    entity_type = data.get('entity_type')
    if entity_type == 'task':
        return Task.from_dict(data)
    # ... more elif branches
    # No else clause with error handling!
```

**Fix Required:**
```python
VALID_ENTITY_TYPES = {'task', 'decision', 'sprint', 'epic', 'edge', 'handoff', ...}

def _entity_from_dict(self, data: dict) -> Entity:
    entity_type = data.get('entity_type')
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    # ... dispatch logic
```

### 2.5 Gap #5: No Foreign Key Validation (MEDIUM)

**Problem:** Edge entities reference source_id and target_id, but there's no validation these IDs exist.

**Impact:**
- Edges can reference non-existent entities
- Orphan edges accumulate after entity deletion
- Query results may include broken references

---

## 3. Namespace Analysis

### 3.1 ID Prefix Mapping

| Prefix | Entity Type | Format | Example |
|--------|-------------|--------|---------|
| **T-** | Task | T-YYYYMMDD-HHMMSS-XXXX | T-20251226-141000-a1b2c3d4 |
| **D-** | Decision | D-YYYYMMDD-HHMMSS-XXXX | D-20251226-141000-e5f6g7h8 |
| **S-** | Sprint | S-NNN or S-sprint-NNN-name | S-026 or S-sprint-017-spark-slm |
| **EPIC-** | Epic | EPIC-{name} | EPIC-got-db |
| **H-** | Handoff | H-YYYYMMDD-HHMMSS-XXXX | H-20251226-141000-q1r2s3t4 |
| **E-** | Edge | E-{source}-{target}-{type} | E-S-026-T-xxx-CONTAINS |

### 3.2 Extended Entity Prefixes

| Prefix | Entity Type | Format |
|--------|-------------|--------|
| CML- | ClaudeMdLayer | CML{layer}-{section}-YYYYMMDD-HHMMSS-XXXX |
| CMV- | ClaudeMdVersion | CMV-{layer_id}-v{version} |
| TEAM- | Team | TEAM-YYYYMMDD-HHMMSS-XXXX |
| PP- | PersonaProfile | PP-YYYYMMDD-HHMMSS-XXXX |
| DOC- | Document | DOC-{normalized-path} |
| G- | Goal | G-YYYYMMDD-XXXX |
| OP- | Orchestration Plan | OP-YYYYMMDD-HHMMSS-XXXX |
| EX- | Execution | EX-YYYYMMDD-HHMMSS-XXXX |

### 3.3 Namespace Issues Found

**Issue #1: E- Prefix Ambiguity**
- Edge entities use E- prefix in their auto-generated IDs
- `generate_epic_id()` in id_generation.py also generates E- prefix
- Production uses EPIC- for epics (correct), but code allows E- (confusing)

**Issue #2: Sprint ID Format Inconsistency**
```
S-018, S-019, S-020, S-021, S-022, S-023, S-024, S-025, S-026  # Sequential (preferred)
S-sprint-001, S-sprint-017-spark-slm, S-sprint-020-forensic   # Legacy format
```

**Issue #3: No Single Source of Truth**
- ID generation in `cortical/utils/id_generation.py`
- Entity classes in `cortical/got/types.py`
- Usage scattered throughout codebase
- No authoritative documentation

---

## 4. Production Data Statistics

```
.got/entities/: 386 total entities (at analysis time)
├── Tasks (T-):      141  (88.7% completed)
├── Edges (E-):      183  (density: 1.11 edges/node)
├── Sprints (S-):     28  (10 completed, 18 available)
├── Decisions (D-):   16
├── Handoffs (H-):    14  (90.9% success rate)
└── Epics (EPIC-):     4  (core, efba, got-db, nlu)
```

---

## 5. Tasks Created

### Sprint S-026: Schema Validation Hardening

**Epic:** EPIC-got-db
**Status:** Available
**Tasks:** 7

| ID | Title | Priority | Dependencies |
|----|-------|----------|--------------|
| T-20251226-141240-a584b507 | Enable edge type validation in Edge.__post_init__ | HIGH | None |
| T-20251226-141300-96f4577c | Enable validation by default in VersionedStore | HIGH | Depends on T-...a584b507 |
| T-20251226-141320-b9e1c3e9 | Define schemas for nested Dict structures | MEDIUM | None |
| T-20251226-141344-74b7dbc3 | Document namespace conventions comprehensively | MEDIUM | None |
| T-20251226-141401-ca0ec210 | Add entity_type validation on deserialize | MEDIUM | None |
| T-20251226-141420-b8c7aa8e | Add foreign key validation for edge references | MEDIUM | Depends on T-...ca0ec210 |
| T-20251226-141441-08b2860f | Consolidate Sprint ID format for consistency | LOW | None |

### Task Dependency Graph

```
T-...a584b507 (Edge type validation) [HIGH]
    └── blocks → T-...96f4577c (Default validation) [HIGH]

T-...ca0ec210 (entity_type validation) [MEDIUM]
    └── blocks → T-...b8c7aa8e (FK validation) [MEDIUM]

T-...b9e1c3e9 (Nested schemas) [MEDIUM] - Independent
T-...74b7dbc3 (Namespace docs) [MEDIUM] - Independent
T-...08b2860f (Sprint ID format) [LOW] - Independent
```

### Recommended Execution Order

1. **T-...a584b507** - Edge type validation (foundational, unblocks #2)
2. **T-...96f4577c** - Enable validation by default (depends on #1)
3. **T-...ca0ec210** - Entity type validation (unblocks #4)
4. **T-...b8c7aa8e** - FK validation (depends on #3)
5. **T-...b9e1c3e9** - Nested schema definitions
6. **T-...74b7dbc3** - Namespace documentation
7. **T-...08b2860f** - Sprint ID consolidation (lowest priority)

---

## 6. Cross-Reference: Existing vs New Tasks

### Tasks NOT Duplicated (Already Existed)

| Existing Task | Coverage | Our Action |
|---------------|----------|------------|
| T-20251226-114231-a640305e | Semantic versioning for GoT entities | No duplicate created |
| T-20251226-114231-2efe8c4a | Automated tests for GoT merge conflicts | No duplicate created |
| T-20251226-132353-68a469de | Fix VersionedStore race condition | No duplicate created |
| T-20251226-112830-8c315485 | Schema validation tests for QueryIndexManager | Different scope (QIM vs VersionedStore) |

### Why New Tasks Were Needed

| Gap | Existing Coverage | Justification for New Task |
|-----|------------------|---------------------------|
| Edge type validation | None | Completely unaddressed |
| Default validation | T-...8c315485 covers QueryIndexManager only | VersionedStore is different component |
| Nested schemas | None | Completely unaddressed |
| Namespace docs | None | Completely unaddressed |
| Entity type validation | None | Completely unaddressed |
| FK validation | None | Completely unaddressed |
| Sprint ID format | Partially in merge-aware task | Needs dedicated fix |

---

## 7. Files of Interest

### Core Schema Files
- `cortical/got/types.py` - Entity dataclasses (1,142 lines)
- `cortical/got/schema.py` - Schema registry (574 lines)
- `cortical/got/entity_schemas.py` - Declarative schemas (623 lines)
- `cortical/got/api.py` - GoTManager API (2,837 lines)
- `cortical/got/versioned_store.py` - Persistence layer

### ID Generation
- `cortical/utils/id_generation.py` - Canonical ID generation functions

### Storage
- `.got/entities/` - Production entity storage (386 entities)
- `.got/entities/_history/` - Historical snapshots
- `.got/indexes/` - Query indexes

### Documentation (to be created)
- `docs/got-namespace-specification.md` - Authoritative namespace docs (Task T-...74b7dbc3)

---

## 8. Recommendations for Future Sessions

### Immediate Actions (Sprint S-026)

1. **Start with T-...a584b507** - Edge type validation is the most critical gap
2. **Run existing tests first** - `python -m pytest tests/unit/test_got*.py -v`
3. **Check production data** - Verify no existing edges have invalid types before adding validation

### Before Modifying Validation Logic

```bash
# Find all tests related to the validation you're changing
grep -rn "edge_type\|EdgeType\|EDGE_TYPES" tests/ | grep -i "invalid\|error\|raise"

# Run coverage to ensure you're not missing edge cases
python -m coverage run -m pytest tests/unit/test_got*.py
python -m coverage report --include="cortical/got/*"
```

### Verification Commands

```bash
# Health check
python scripts/got_utils.py validate

# Sprint status
python scripts/got_utils.py sprint status S-026

# Task details
python scripts/got_utils.py task show T-20251226-141240-a584b507
```

---

## 9. Decision Record

**ID:** D-20251226-141841-c387695a
**Decision:** Schema validation hardening initiative based on deep analysis
**Rationale:** Deep analysis revealed critical gaps in schema validation enforcement. Created Sprint S-026 to address systematically.

---

## 10. Session Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Sprint | S-026 | Container for schema hardening tasks |
| Tasks | T-20251226-14* (7 tasks) | Individual work items |
| Decision | D-20251226-141841-c387695a | Documented rationale |
| Knowledge Transfer | This document | Future session context |

---

## Appendix A: Valid Edge Types Reference

```python
VALID_EDGE_TYPES = {
    'DEPENDS_ON',    # Task A depends on Task B
    'BLOCKS',        # Task A blocks Task B
    'CONTAINS',      # Sprint contains Task, Epic contains Sprint
    'RELATES_TO',    # General relationship
    'REQUIRES',      # Hard requirement
    'IMPLEMENTS',    # Task implements Decision
    'SUPERSEDES',    # Entity replaces another
    'DERIVED_FROM',  # Entity derived from another
    'PARENT_OF',     # Hierarchical parent
    'CHILD_OF',      # Hierarchical child
    'REFERENCES',    # Soft reference
    'CONTRADICTS',   # Conflicting entities
}
```

---

## Appendix B: Entity Status Enums

| Entity | Valid Statuses |
|--------|---------------|
| Task | pending, in_progress, completed, blocked |
| Sprint | available, in_progress, completed, blocked |
| Epic | active, completed, on_hold |
| Handoff | initiated, accepted, completed, rejected |

---

*Generated by Claude (Opus 4.5) on 2025-12-26*
*Session: claude/analyze-got-schemas-namespaces-eDjPT*
