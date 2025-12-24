# GoT Entity Schema Validation

This document describes the declarative schema system for validating GoT entities (Task, Decision, Sprint, Epic, Edge, Handoff, etc.) before persistence.

## Overview

The schema system provides:

- **Declarative field definitions** - Define required fields, types, defaults, and constraints
- **Schema versioning** - Track schema versions separately from entity versions
- **Migration support** - Automatically upgrade data between schema versions
- **Validation on save** - Optionally validate entities before persistence
- **Backward compatibility** - Works with existing data; validation is opt-in

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SCHEMA ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  cortical/got/schema.py                                                │
│  ├── FieldType (enum)          # STRING, INTEGER, ENUM, LIST, etc.    │
│  ├── Field (dataclass)         # Field definition with validation     │
│  ├── ValidationResult          # Validation result with errors        │
│  ├── BaseSchema (class)        # Base class for entity schemas        │
│  └── SchemaRegistry (singleton)# Central schema registration          │
│                                                                         │
│  cortical/got/entity_schemas.py                                        │
│  ├── TaskSchema                # Schema for Task entities             │
│  ├── DecisionSchema            # Schema for Decision entities         │
│  ├── SprintSchema              # Schema for Sprint entities           │
│  ├── EpicSchema                # Schema for Epic entities             │
│  ├── EdgeSchema                # Schema for Edge entities             │
│  ├── HandoffSchema             # Schema for Handoff entities          │
│  ├── ClaudeMdLayerSchema       # Schema for CLAUDE.md layers          │
│  ├── ClaudeMdVersionSchema     # Schema for CLAUDE.md versions        │
│  ├── TeamSchema                # Schema for Team entities             │
│  ├── PersonaProfileSchema      # Schema for PersonaProfile entities   │
│  └── DocumentSchema            # Schema for Document entities         │
│                                                                         │
│  cortical/got/versioned_store.py                                       │
│  └── VersionedStore            # validate_on_save flag integration    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Enable Validation on Save

```python
from cortical.got.versioned_store import VersionedStore
from cortical.got.entity_schemas import ensure_schemas_registered

# Ensure schemas are registered (auto-called on import)
ensure_schemas_registered()

# Create store with validation enabled
store = VersionedStore(
    store_dir=Path(".got/entities"),
    validate_on_save=True  # Enable schema validation
)

# Now all writes are validated
from cortical.got.types import Task

task = Task(
    id="T-123",
    title="Valid task",
    status="pending",  # Must be: pending, in_progress, completed, blocked
    priority="high"    # Must be: low, medium, high, critical
)
store.write(task)  # Validated before write

# Invalid task would raise ValidationError
```

### Manual Validation

```python
from cortical.got.schema import validate_entity, migrate_entity
from cortical.got.entity_schemas import ensure_schemas_registered

ensure_schemas_registered()

# Validate data
data = {
    'id': 'T-123',
    'entity_type': 'task',
    'title': 'My task',
    'status': 'pending',
    'priority': 'high',
}

result = validate_entity('task', data)
if result.valid:
    print("Data is valid!")
else:
    print(f"Errors: {result.errors}")

# Migrate data from older schema version
migrated, result = migrate_entity('task', data)
if result.migrated:
    print(f"Migrated from v{result.from_version} to v{result.to_version}")
```

## Field Types

The schema system supports these field types:

| FieldType | Python Type | Description |
|-----------|-------------|-------------|
| `STRING` | `str` | Text values |
| `INTEGER` | `int` | Whole numbers |
| `FLOAT` | `int`, `float` | Numbers (int accepted for float) |
| `BOOLEAN` | `bool` | True/False |
| `LIST` | `list` | Array values |
| `DICT` | `dict` | Object/mapping values |
| `ENUM` | `str` | String from allowed choices |
| `DATETIME` | `str` | ISO 8601 timestamp strings |
| `ANY` | any | No type validation |

## Defining Custom Schemas

### Basic Schema

```python
from cortical.got.schema import BaseSchema, Field, FieldType, register_schema

class MyEntitySchema(BaseSchema):
    schema_version = 1
    entity_type = 'my_entity'

    fields = {
        'id': Field('id', FieldType.STRING, required=True),
        'name': Field('name', FieldType.STRING, required=True),
        'count': Field('count', FieldType.INTEGER, required=False, default=0),
        'status': Field('status', FieldType.ENUM,
                       choices=['draft', 'published'],
                       default='draft'),
        'tags': Field('tags', FieldType.LIST,
                     item_type=FieldType.STRING,
                     default=[]),
    }

# Register the schema
register_schema('my_entity', MyEntitySchema)
```

### With Custom Validation

```python
def validate_positive(value: int) -> bool:
    return value > 0

class OrderSchema(BaseSchema):
    schema_version = 1
    entity_type = 'order'

    fields = {
        'id': Field('id', FieldType.STRING, required=True),
        'quantity': Field('quantity', FieldType.INTEGER,
                         required=True,
                         validator=validate_positive),
        'unit_price': Field('unit_price', FieldType.FLOAT,
                           required=True,
                           validator=validate_positive),
    }
```

### With Migrations

```python
class TaskSchema(BaseSchema):
    schema_version = 2  # Current version
    entity_type = 'task'

    fields = {
        'id': Field('id', FieldType.STRING, required=True),
        'title': Field('title', FieldType.STRING, required=True),
        'metadata': Field('metadata', FieldType.DICT, default={}),  # Added in v2
    }

    @classmethod
    def migrate_v1_to_v2(cls, data: dict) -> dict:
        """Migration from v1 to v2: Add metadata field."""
        if 'metadata' not in data:
            data['metadata'] = {}
        return data
```

## Schema Version Tracking

Schema versions are stored in the entity data under `_schema_version`:

```json
{
  "id": "T-123",
  "entity_type": "task",
  "title": "My task",
  "status": "pending",
  "_schema_version": 1
}
```

When loading data:
1. Check `_schema_version` in data
2. If older than current schema, run migrations
3. Update `_schema_version` to current

## Entity Schemas Reference

### TaskSchema (v1)

| Field | Type | Required | Default | Choices |
|-------|------|----------|---------|---------|
| id | STRING | Yes | - | - |
| entity_type | STRING | Yes | - | - |
| title | STRING | Yes | - | - |
| status | ENUM | Yes | pending | pending, in_progress, completed, blocked |
| priority | ENUM | No | medium | low, medium, high, critical |
| description | STRING | No | "" | - |
| properties | DICT | No | {} | - |
| metadata | DICT | No | {} | - |

### DecisionSchema (v1)

| Field | Type | Required | Default | Choices |
|-------|------|----------|---------|---------|
| id | STRING | Yes | - | - |
| entity_type | STRING | Yes | - | - |
| title | STRING | Yes | - | - |
| rationale | STRING | No | "" | - |
| affects | LIST[STRING] | No | [] | - |
| properties | DICT | No | {} | - |

### SprintSchema (v1)

| Field | Type | Required | Default | Choices |
|-------|------|----------|---------|---------|
| id | STRING | Yes | - | - |
| entity_type | STRING | Yes | - | - |
| title | STRING | Yes | - | - |
| status | ENUM | Yes | available | available, in_progress, completed, blocked |
| epic_id | STRING | No | "" | - |
| number | INTEGER | No | 0 | - |
| goals | LIST | No | [] | - |
| notes | LIST[STRING] | No | [] | - |

### EpicSchema (v1)

| Field | Type | Required | Default | Choices |
|-------|------|----------|---------|---------|
| id | STRING | Yes | - | - |
| entity_type | STRING | Yes | - | - |
| title | STRING | Yes | - | - |
| status | ENUM | Yes | active | active, completed, on_hold |
| phase | INTEGER | No | 1 | - |
| phases | LIST | No | [] | - |

### EdgeSchema (v1)

| Field | Type | Required | Default | Validator |
|-------|------|----------|---------|-----------|
| id | STRING | Yes | - | - |
| entity_type | STRING | Yes | - | - |
| source_id | STRING | Yes | - | - |
| target_id | STRING | Yes | - | - |
| edge_type | STRING | Yes | - | - |
| weight | FLOAT | No | 1.0 | Must be in [0.0, 1.0] |
| confidence | FLOAT | No | 1.0 | Must be in [0.0, 1.0] |

### HandoffSchema (v1)

| Field | Type | Required | Default | Choices |
|-------|------|----------|---------|---------|
| id | STRING | Yes | - | - |
| entity_type | STRING | Yes | - | - |
| source_agent | STRING | Yes | - | - |
| target_agent | STRING | Yes | - | - |
| status | ENUM | Yes | initiated | initiated, accepted, completed, rejected |
| instructions | STRING | No | "" | - |
| context | DICT | No | {} | - |
| result | DICT | No | {} | - |
| artifacts | LIST[STRING] | No | [] | - |

## API Reference

### Schema Module (`cortical.got.schema`)

```python
# Registry functions
register_schema(entity_type: str, schema: Type[BaseSchema]) -> None
validate_entity(entity_type: str, data: dict, strict: bool = False) -> ValidationResult
migrate_entity(entity_type: str, data: dict) -> Tuple[dict, ValidationResult]
get_registry() -> SchemaRegistry

# Classes
class Field:
    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None
    item_type: Optional[FieldType] = None
    validator: Optional[Callable[[Any], bool]] = None

class BaseSchema:
    schema_version: int
    entity_type: str
    fields: Dict[str, Field]

    @classmethod
    def validate(cls, data: dict, strict: bool = False) -> ValidationResult

    @classmethod
    def migrate(cls, data: dict, from_version: int = None) -> Tuple[dict, ValidationResult]

    @classmethod
    def apply_defaults(cls, data: dict) -> dict

    @classmethod
    def prepare_for_save(cls, data: dict) -> dict

class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    migrated: bool
    from_version: Optional[int]
    to_version: Optional[int]
```

### Entity Schemas Module (`cortical.got.entity_schemas`)

```python
# Ensure all schemas are registered
ensure_schemas_registered() -> None

# Get schema class for entity type
get_schema_for_entity_type(entity_type: str) -> Type[BaseSchema]

# List all entity types with schemas
list_entity_types() -> List[str]

# Available schema classes
TaskSchema, DecisionSchema, SprintSchema, EpicSchema, EdgeSchema
HandoffSchema, ClaudeMdLayerSchema, ClaudeMdVersionSchema
TeamSchema, PersonaProfileSchema, DocumentSchema
```

### VersionedStore Integration

```python
from cortical.got.versioned_store import VersionedStore

store = VersionedStore(
    store_dir=Path(".got/entities"),
    validate_on_save=True  # Enable validation
)

# ValidationError raised if entity fails schema validation
store.write(entity)
store.apply_writes(write_set)
```

## Best Practices

### 1. Always Use Schema Versions

```python
class MySchema(BaseSchema):
    schema_version = 1  # Start at 1, increment for breaking changes
```

### 2. Provide Defaults for Optional Fields

```python
fields = {
    'metadata': Field('metadata', FieldType.DICT, required=False, default={}),
}
```

### 3. Write Migrations for Breaking Changes

```python
class MySchema(BaseSchema):
    schema_version = 2

    @classmethod
    def migrate_v1_to_v2(cls, data):
        # Add new required field
        data['new_field'] = 'default_value'
        return data
```

### 4. Use Strict Mode for Development

```python
# In tests, use strict mode to catch unexpected fields
result = validate_entity('task', data, strict=True)
for warning in result.warnings:
    print(f"Warning: {warning}")
```

### 5. Enable Validation in Production

```python
# Enable validation to catch data corruption early
store = VersionedStore(store_dir, validate_on_save=True)
```

## Migration Guide

### Adding a New Field

1. Add field to schema with `required=False` and a default
2. Keep schema_version the same (backward compatible)

```python
fields = {
    ...
    'new_field': Field('new_field', FieldType.STRING, required=False, default=''),
}
```

### Making a Field Required

1. Increment schema_version
2. Add migration to set default for existing data
3. Change field to `required=True`

```python
class MySchema(BaseSchema):
    schema_version = 2  # Increment!

    fields = {
        'required_field': Field('required_field', FieldType.STRING, required=True),
    }

    @classmethod
    def migrate_v1_to_v2(cls, data):
        if 'required_field' not in data:
            data['required_field'] = 'migrated_default'
        return data
```

### Removing a Field

1. Increment schema_version
2. Remove field from schema
3. Add migration that removes field from data (optional)

```python
@classmethod
def migrate_v1_to_v2(cls, data):
    data.pop('old_field', None)  # Remove deprecated field
    return data
```

### Changing Field Type

1. Increment schema_version
2. Add migration to convert data

```python
@classmethod
def migrate_v1_to_v2(cls, data):
    # Convert count from string to integer
    if 'count' in data and isinstance(data['count'], str):
        data['count'] = int(data['count'])
    return data
```

## Error Handling

```python
from cortical.got.errors import ValidationError

try:
    store.write(entity)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Entity type: {e.context.get('entity_type')}")
    print(f"Errors: {e.context.get('errors')}")
```

## Testing

```python
import pytest
from cortical.got.schema import validate_entity
from cortical.got.entity_schemas import ensure_schemas_registered

@pytest.fixture(autouse=True)
def setup_schemas():
    ensure_schemas_registered()

def test_valid_task():
    result = validate_entity('task', {
        'id': 'T-123',
        'entity_type': 'task',
        'title': 'Test task',
        'status': 'pending',
    })
    assert result.valid

def test_invalid_status():
    result = validate_entity('task', {
        'id': 'T-123',
        'entity_type': 'task',
        'title': 'Test task',
        'status': 'invalid',  # Bad status
    })
    assert not result.valid
    assert any('status' in e for e in result.errors)
```
