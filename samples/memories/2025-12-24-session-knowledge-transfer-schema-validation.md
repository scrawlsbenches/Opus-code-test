# Knowledge Transfer: Schema Validation System Implementation

**Date:** 2025-12-24
**Session ID:** 9c2aN
**Branch:** claude/continue-handoff-tasks-9c2aN
**Sprint:** S-018 (Schema Evolution Foundation)

---

## Executive Summary

This session implemented a complete declarative schema validation system for GoT entities. The system provides field-level validation, schema versioning, automatic migration, and optional validation-on-save integration with the persistence layer.

**Key Outcome:** Sprint 018 advanced from 50% to 83.3% complete (9 → 15 tasks).

---

## What Was Built

### 1. Core Schema Infrastructure (`cortical/got/schema.py`)

**Purpose:** Declarative schema definitions with validation and migration support.

**Components:**
- `FieldType` enum - Supported types: STRING, INTEGER, FLOAT, BOOLEAN, LIST, DICT, ENUM, DATETIME, ANY
- `Field` dataclass - Field definition with type, required flag, default, choices, custom validator
- `ValidationResult` - Validation outcome with errors, warnings, migration info
- `BaseSchema` class - Base for entity schemas with:
  - `schema_version` - Separate from entity version
  - `fields` dict - Field definitions
  - `validate()` - Validate data against schema
  - `migrate()` - Upgrade data between versions
  - `apply_defaults()` - Fill missing optional fields
  - `prepare_for_save()` - Apply defaults + set schema version
- `SchemaRegistry` singleton - Central schema registration and lookup

**Key Design Decisions:**
- Schema version stored as `_schema_version` in entity data (not entity `version`)
- Migrations discovered via naming convention: `migrate_v1_to_v2(cls, data)`
- Validation is opt-in to maintain backward compatibility
- Registry is singleton for global access

### 2. Entity Schemas (`cortical/got/entity_schemas.py`)

**Purpose:** Declarative schemas for all 11 GoT entity types.

**Schemas Defined:**
| Schema | Entity Type | Key Fields |
|--------|-------------|------------|
| TaskSchema | task | title, status, priority, description |
| DecisionSchema | decision | title, rationale, affects |
| SprintSchema | sprint | title, status, epic_id, goals |
| EpicSchema | epic | title, status, phase, phases |
| EdgeSchema | edge | source_id, target_id, edge_type, weight |
| HandoffSchema | handoff | source_agent, target_agent, status |
| ClaudeMdLayerSchema | claudemd_layer | layer_type, content, freshness_status |
| ClaudeMdVersionSchema | claudemd_version | content, content_hash, layer_ids |
| TeamSchema | team | name, members, capabilities |
| PersonaProfileSchema | persona_profile | name, role, expertise |
| DocumentSchema | document | title, content, doc_type |

**Shared Base Fields:**
```python
BASE_ENTITY_FIELDS = {
    'id': Field('id', FieldType.STRING, required=True),
    'entity_type': Field('entity_type', FieldType.STRING, required=True),
    'version': Field('version', FieldType.INTEGER, default=1),
    'created_at': Field('created_at', FieldType.DATETIME),
    'modified_at': Field('modified_at', FieldType.DATETIME),
}
```

**Auto-Registration:** Schemas register on module import via `ensure_schemas_registered()`.

### 3. Validation on Save (`cortical/got/versioned_store.py`)

**Changes:**
- Added `validate_on_save: bool = False` parameter to `__init__()`
- Added `_validate_entity()` method
- Integrated validation into `write()` and `apply_writes()`

**Behavior:**
- When `validate_on_save=True`, entities validated before any disk writes
- `apply_writes()` validates ALL entities before writing ANY (atomic validation)
- Invalid entities raise `ValidationError` with detailed error messages
- Unknown entity types silently pass (no schema = no validation)

### 4. Documentation (`docs/got-entity-schema-validation.md`)

Comprehensive documentation covering:
- Architecture overview
- Quick start guide
- Field types reference
- Custom schema definition
- Migration guide
- API reference
- Best practices

### 5. Tests (`tests/unit/test_schema.py`)

37 unit tests covering:
- Field validation for all types
- Required/optional field handling
- Default value application
- Schema validation
- Migration chains
- Registry operations
- Global convenience functions

---

## Tasks Completed

| Task ID | Title | Priority |
|---------|-------|----------|
| T-20251223-151723-24aed5d3 | Design schema registry architecture | high |
| T-20251223-151730-a1e23e9e | Implement BaseSchema class with version tracking | high |
| T-20251223-151736-b174b5d7 | Implement schema migration support | high |
| T-20251223-151749-cb683757 | Define schemas for existing entities | medium |
| T-20251223-151742-d62f2abf | Add schema validation on save | medium |
| T-20251223-151755-da981003 | Update architecture docs | low |
| T-20251223-184503-0550720b | Fix HandoffManager AttributeError | medium |

---

## How To Use

### Enable Validation on Save

```python
from cortical.got.versioned_store import VersionedStore
from cortical.got.entity_schemas import ensure_schemas_registered

ensure_schemas_registered()  # Auto-called on import

store = VersionedStore(
    store_dir=Path(".got/entities"),
    validate_on_save=True  # Enable validation
)
```

### Manual Validation

```python
from cortical.got.schema import validate_entity, migrate_entity

result = validate_entity('task', data)
if not result.valid:
    print(f"Errors: {result.errors}")

# Migrate old data
migrated, result = migrate_entity('task', old_data)
```

### Define Custom Schema

```python
from cortical.got.schema import BaseSchema, Field, FieldType, register_schema

class MySchema(BaseSchema):
    schema_version = 1
    entity_type = 'my_type'
    fields = {
        'id': Field('id', FieldType.STRING, required=True),
        'name': Field('name', FieldType.STRING, required=True),
    }

register_schema('my_type', MySchema)
```

### Add Migration

```python
class MySchema(BaseSchema):
    schema_version = 2

    @classmethod
    def migrate_v1_to_v2(cls, data):
        data['new_field'] = 'default'
        return data
```

---

## Files Changed

| File | Change Type | Lines |
|------|-------------|-------|
| `cortical/got/schema.py` | NEW | 520 |
| `cortical/got/entity_schemas.py` | NEW | 620 |
| `cortical/got/versioned_store.py` | MODIFIED | +40 |
| `cortical/got/__init__.py` | MODIFIED | +35 |
| `tests/unit/test_schema.py` | NEW | 440 |
| `docs/got-entity-schema-validation.md` | NEW | 450 |

**Total:** ~2,100 lines added

---

## Commits

```
0a7648cf feat(got): Add declarative schema validation for GoT entities
```

---

## What NOT to Do

1. **Don't bypass validation for invalid data** - Fix the data, don't disable validation
2. **Don't use schema_version=0** - Start at 1, increment for breaking changes
3. **Don't forget migrations** - When changing required fields or types, add migration
4. **Don't modify BASE_ENTITY_FIELDS** - These are shared across all schemas

---

## Known Limitations

1. **No nested object validation** - Dict fields validated as type only, not structure
2. **No cross-field validation** - Each field validated independently
3. **No async validation** - All validation is synchronous
4. **Schema version in data** - Uses `_schema_version` key, could conflict with user data

---

## Remaining Sprint 018 Tasks

| Task ID | Title | Priority |
|---------|-------|----------|
| T-20251223-153551-5705de6e | Design orphan detection and auto-linking | high |
| (2 others) | Various non-schema tasks | various |

---

## Handoff Created

- **ID:** H-20251224-092509-1f9493d8
- **Target:** next-session
- **Task:** T-20251223-153551-5705de6e
- **Status:** initiated

---

## Quick Recovery Commands

```bash
# Check sprint status
python scripts/got_utils.py sprint status

# Validate GoT state
python scripts/got_utils.py validate

# Run schema tests
python -m pytest tests/unit/test_schema.py -v

# Accept handoff
python scripts/got_utils.py handoff accept H-20251224-092509-1f9493d8 --agent YOUR_AGENT
```

---

**Tags:** `schema`, `validation`, `got`, `sprint-018`, `migration`, `entity-types`
