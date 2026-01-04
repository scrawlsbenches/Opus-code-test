# Task T-001-A Validation Report: Generic Entity Accessor for Query Builder

**Date:** 2026-01-04
**Task:** T-001-A - Add generic entity accessor to Query builder for extensibility
**Validator:** Senior Engineer (Deep Inspection)
**Status:** VALIDATED WITH CRITICAL FINDINGS

---

## Executive Summary

The proposed `.entities('type')` method **CAN be implemented** as designed, but requires addressing **critical infrastructure gaps** discovered during validation. The design is sound but incomplete.

**Key Finding:** The EntityType enum is a bottleneck. It only covers 5 of 12 registered entity types, making a purely enum-based approach insufficient.

**Verdict:** **APPROVED WITH CONDITIONS**

### Conditions for Approval

1. **EntityType enum must be bypassed** - Use string-based entity type with schema validation
2. **Missing list methods must be implemented** - 4 entity types lack GoTManager list methods
3. **Field validation implementation required** - Schema introspection infrastructure is ready
4. **ID prefix mapping needs completion** - 5 entity types have no confirmed prefix pattern

---

## Part 1: Validated Assumptions

### ✅ Assumption 1: SchemaRegistry Provides Field Introspection

**Status:** VALIDATED - Works perfectly

```python
from cortical.got.schema import get_registry

registry = get_registry()
task_schema = registry.get_schema('task')

# Fields are fully introspectable
for field_name, field in task_schema.fields.items():
    print(f'{field_name}: {field.field_type.name}')
    if field.choices:  # ENUM fields have valid values
        print(f'  choices: {field.choices}')
```

**Discovered Capabilities:**

| Feature | Available | Quality |
|---------|-----------|---------|
| Field names | ✅ Yes | Complete |
| Field types | ✅ Yes | Typed (FieldType enum) |
| ENUM choices | ✅ Yes | List of valid values |
| Required/optional | ✅ Yes | Boolean flag |
| Field descriptions | ✅ Yes | Human-readable |
| Default values | ✅ Yes | Defined per field |

**Example Schema Introspection:**

```
Task Schema Fields (11):
  id: STRING (required)
  entity_type: STRING (required)
  version: INTEGER (optional)
  created_at: DATETIME (optional)
  modified_at: DATETIME (optional)
  title: STRING (required)
  status: ENUM (required)
    choices: ['pending', 'in_progress', 'completed', 'blocked']
  priority: ENUM (optional)
    choices: ['low', 'medium', 'high', 'critical']
  description: STRING (optional)
  properties: DICT (optional)
  metadata: DICT (optional)
```

**Field Validation with Suggestions:**

```python
import difflib

unknown_field = 'statsu'
valid_fields = list(task_schema.fields.keys())
matches = difflib.get_close_matches(unknown_field, valid_fields, n=3, cutoff=0.6)
# Result: ['status']
```

**Recommendation:** Use `difflib.get_close_matches()` for "Did you mean?" suggestions.

---

### ❌ Assumption 2: EntityType Enum is Sufficient

**Status:** INVALIDATED - Critical gap discovered

**Problem:** The EntityType enum only covers 5 of 12 registered entity types.

```python
# Current EntityType enum (cortical/got/query_builder.py)
class EntityType(Enum):
    TASK = auto()
    DECISION = auto()
    SPRINT = auto()
    EDGE = auto()
    HANDOFF = auto()  # 5 total

# Registered schemas (cortical/got/entity_schemas.py)
ALL_SCHEMAS = {
    'task': TaskSchema,
    'decision': DecisionSchema,
    'sprint': SprintSchema,
    'epic': EpicSchema,  # ❌ NOT in EntityType
    'edge': EdgeSchema,
    'handoff': HandoffSchema,
    'knowledge_transfer': KnowledgeTransferSchema,  # ❌ NOT in EntityType
    'claudemd_layer': ClaudeMdLayerSchema,  # ❌ NOT in EntityType
    'claudemd_version': ClaudeMdVersionSchema,  # ❌ NOT in EntityType
    'team': TeamSchema,  # ❌ NOT in EntityType
    'persona_profile': PersonaProfileSchema,  # ❌ NOT in EntityType
    'document': DocumentSchema,  # ❌ NOT in EntityType
}
```

**Missing from EntityType enum:**
- claudemd_version
- claudemd_layer
- epic
- document
- team
- persona_profile
- knowledge_transfer

**Impact:** Current design using `self._entity_type = EntityType.TASK` won't work for 7 entity types.

**Recommended Solution:**

```python
# Instead of EntityType enum, use string-based with schema validation
class Query:
    def entities(self, entity_type: str) -> "Query[T]":
        """Query entities by type name.

        Args:
            entity_type: Entity type name (case-insensitive)
                        e.g., 'task', 'sprint', 'decision', 'epic'

        Returns:
            Self for chaining

        Raises:
            QueryValidationError: If entity_type is not registered
        """
        from cortical.got.schema import get_registry

        # Normalize to lowercase
        entity_type = entity_type.lower()

        # Validate against schema registry
        registry = get_registry()
        if not registry.has_schema(entity_type):
            available = sorted(registry._schemas.keys())
            raise QueryValidationError(
                f"Unknown entity type '{entity_type}'. "
                f"Available types: {', '.join(available)}"
            )

        # Store as string, not enum
        self._entity_type_str = entity_type
        self._entity_type = None  # Legacy enum field
        return self
```

---

### ⚠️ Assumption 3: GoTManager Has List Methods for All Entity Types

**Status:** PARTIALLY VALIDATED - 4 entity types missing

**Verified Methods:**

```python
# GoTManager methods that exist:
manager.list_all_tasks() -> List[Task]
manager.list_sprints() -> List[Sprint]
manager.list_decisions() -> List[Decision]
manager.list_edges() -> List[Edge]
manager.list_epics() -> List[Epic]
manager.list_handoffs() -> List[Handoff]
manager.list_documents() -> List[Document]
manager.list_claudemd_layers() -> List[ClaudeMdLayer]
```

**Missing Methods:**

| Entity Type | Expected Method | Status |
|-------------|-----------------|--------|
| knowledge_transfer | `list_knowledge_transfers()` | ❌ NOT FOUND |
| claudemd_version | `list_claudemd_versions()` | ❌ NOT FOUND |
| team | `list_teams()` | ❌ NOT FOUND |
| persona_profile | `list_persona_profiles()` | ❌ NOT FOUND |

**Impact:** Cannot query these entity types until list methods are implemented.

**Workaround for Phase 1:**

```python
# Generic fallback using file globbing
def _get_base_entities_generic(self, entity_type: str) -> List[Entity]:
    """Fallback for entity types without list methods."""
    from pathlib import Path
    import json

    entities_dir = self._manager.got_dir / "entities"
    prefix = ENTITY_PREFIX_MAP.get(entity_type)

    if not prefix:
        # No known prefix - scan all files (slow!)
        pattern = "*.json"
    else:
        pattern = f"{prefix}*.json"

    results = []
    for f in entities_dir.glob(pattern):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if data.get('data', {}).get('entity_type') == entity_type:
                    # Deserialize to proper entity class
                    entity_class = self._get_entity_class(entity_type)
                    results.append(entity_class.from_dict(data['data']))
        except:
            continue
    return results
```

**Recommendation:** Implement missing list methods in GoTManager as separate tasks.

---

### ✅ Assumption 4: ID Prefix Convention Exists

**Status:** VALIDATED for 7 types, UNKNOWN for 5 types

**Confirmed ID Prefixes (from file analysis):**

```python
ENTITY_PREFIX_MAP = {
    'task': 'T',           # T-20251227-171054-46499f2f
    'decision': 'D',       # D-20251226-195140-84a14df0
    'sprint': 'S',         # S-20260102-231925-80abadeb
    'edge': 'E',           # E-S-...-T-...-CONTAINS
    'epic': 'EPIC',        # EPIC-20251226-223250-d5dc7122
    'handoff': 'H',        # H-20251226-124311-1ddfc7f7
    'knowledge_transfer': 'KT',  # KT-20260101-093241
}
```

**Unknown Prefixes:**

| Entity Type | Expected Pattern | Files Found | Status |
|-------------|------------------|-------------|--------|
| claudemd_layer | `CML*` ? | Not verified | ❓ |
| claudemd_version | `CMV*` ? | 0 files | ❌ No data |
| document | `DOC*` ? | Not checked | ❓ |
| team | `TEAM*` ? | 0 files | ❌ No data |
| persona_profile | `PP*` ? | 0 files | ❌ No data |

**Special Cases:**

```
Epic IDs have TWO patterns:
  - EPIC-20251226-223250-d5dc7122  (timestamp-based)
  - EPIC-test-123                   (custom)
  - EPIC-phenomenology              (custom)

Edge IDs encode relationship:
  - E-S-20251227-211213-ae934eab-T-20251227-211228-f64af3d9-CONTAINS
    (source-target-edgetype)
```

**Recommendation:** Document prefix patterns in a central registry.

---

### ✅ Assumption 5: Where Clause Field Validation

**Status:** VALIDATED - Schema supports comprehensive validation

**Current Where Implementation:**

```python
# From query_builder.py:859
def where(self, **conditions) -> "Query[T]":
    """Add WHERE conditions (AND)."""
    self._validate_not_executed("where")
    for field, value in conditions.items():
        self._where_clauses.append(WhereClause(field=field, value=value))
    return self
```

**Problem:** No field validation happens at query construction time!

**Recommended Enhanced Implementation:**

```python
def where(self, **conditions) -> "Query[T]":
    """Add WHERE conditions with schema validation."""
    self._validate_not_executed("where")

    # Get schema for current entity type
    if self._entity_type_str:
        from cortical.got.schema import get_registry
        registry = get_registry()
        schema = registry.get_schema(self._entity_type_str)

        # Validate each field
        for field, value in conditions.items():
            if field not in schema.fields:
                # Unknown field - suggest alternatives
                import difflib
                valid_fields = list(schema.fields.keys())
                suggestions = difflib.get_close_matches(field, valid_fields, n=3, cutoff=0.6)

                error_msg = f"Unknown field '{field}' for entity type '{self._entity_type_str}'"
                if suggestions:
                    error_msg += f"\n  Did you mean: {suggestions[0]}"
                error_msg += f"\n  Valid fields: {', '.join(sorted(valid_fields))}"

                raise QueryValidationError(error_msg)

            # TODO: Validate value type matches field type

        self._where_clauses.append(WhereClause(field=field, value=value))

    return self
```

**Field Validation Scenarios (from T-001-A spec):**

| Scenario | Input | Expected Behavior |
|----------|-------|-------------------|
| Unknown field with typo | `.where(statsu='pending')` | Error: "Did you mean: status" |
| Unknown field | `.where(nonexistent='x')` | Error with valid field list |
| Wrong entity type field | `.entities('sprint').where(priority='high')` | Error: "sprint has no field 'priority'" |
| Valid field | `.where(status='pending')` | Success |

---

## Part 2: Implementation Risks

### RISK 1: EntityType Enum Bottleneck (CRITICAL)

**Severity:** 🔴 Critical
**Impact:** Cannot query 7 of 12 entity types with current design

**Root Cause:**

```python
# Current _get_base_entities hardcodes EntityType enum
def _get_base_entities(self) -> List[Any]:
    if self._entity_type == EntityType.TASK:
        return self._manager.list_all_tasks()
    elif self._entity_type == EntityType.SPRINT:
        return self._manager.list_sprints()
    # ... only 5 types supported
    else:
        return []  # ❌ Silently returns empty for unknown types!
```

**Mitigation:**

Replace enum-based dispatch with string-based dispatch:

```python
def _get_base_entities(self) -> List[Any]:
    """Get base entities based on entity type string."""
    if not self._entity_type_str:
        return []

    # Map entity type to GoTManager method
    method_map = {
        'task': 'list_all_tasks',
        'sprint': 'list_sprints',
        'decision': 'list_decisions',
        'edge': 'list_edges',
        'epic': 'list_epics',
        'handoff': 'list_handoffs',
        'document': 'list_documents',
        'claudemd_layer': 'list_claudemd_layers',
        'knowledge_transfer': 'list_knowledge_transfers',  # TODO: implement
        # ... add others as methods are implemented
    }

    method_name = method_map.get(self._entity_type_str)
    if not method_name:
        # Fallback to generic method
        return self._get_base_entities_generic(self._entity_type_str)

    method = getattr(self._manager, method_name, None)
    if method is None:
        raise QueryExecutionError(
            f"GoTManager.{method_name}() not implemented for entity type '{self._entity_type_str}'"
        )

    return method()
```

---

### RISK 2: Missing GoTManager List Methods (HIGH)

**Severity:** 🟡 High
**Impact:** 4 entity types not queryable until methods added

**Missing Methods:**

1. `GoTManager.list_knowledge_transfers()`
2. `GoTManager.list_claudemd_versions()`
3. `GoTManager.list_teams()`
4. `GoTManager.list_persona_profiles()`

**Workaround:**

Implement generic file-based fallback (shown in Assumption 3).

**Long-term Fix:**

Create separate tasks to implement these methods following the pattern of existing list methods.

---

### RISK 3: Performance with Generic Fallback (MEDIUM)

**Severity:** 🟠 Medium
**Impact:** Slow queries for entity types without list methods

**Problem:**

Generic fallback scans all entity files:

```python
# Worst case: scan all 800+ entity files
for f in entities_dir.glob("*.json"):
    # Read and deserialize each file
    # Check entity_type field
    # Filter matches
```

**Mitigation:**

1. Use ID prefix if known (limits file scan)
2. Add caching layer
3. Implement missing list methods (best fix)

---

### RISK 4: Schema-Entity Type Mismatch (LOW)

**Severity:** 🟢 Low
**Impact:** Edge case handling needed

**Scenario:**

What if entity file has `entity_type='task'` but doesn't match TaskSchema?

**Current State:**

```python
# Entity files have checksum validation
is_valid, error = validate_entity_file(wrapper)
if not is_valid:
    logger.warning(f"Invalid file: {error}")
    return None
```

**Recommendation:** Trust existing validation, add unit tests for edge cases.

---

## Part 3: Recommended Guardrails

### Guardrail 1: Unknown Entity Type Error

```python
# In Query.entities()
if not registry.has_schema(entity_type):
    available = sorted(registry._schemas.keys())
    raise QueryValidationError(
        f"Unknown entity type '{entity_type}'.\n"
        f"Available types:\n" +
        "\n".join(f"  - {t}" for t in available)
    )
```

**Error Message Example:**

```
QueryValidationError: Unknown entity type 'tak'.

Available types:
  - claudemd_layer
  - claudemd_version
  - decision
  - document
  - edge
  - epic
  - handoff
  - knowledge_transfer
  - persona_profile
  - sprint
  - task
  - team

Did you mean: task
```

---

### Guardrail 2: Unknown Field Error with Suggestions

```python
# In Query.where()
if field not in schema.fields:
    import difflib
    valid_fields = sorted(schema.fields.keys())
    suggestions = difflib.get_close_matches(field, valid_fields, n=1, cutoff=0.6)

    error_msg = f"Unknown field '{field}' for entity type '{self._entity_type_str}'"
    if suggestions:
        error_msg += f"\n  Did you mean: {suggestions[0]}"
    error_msg += f"\n\n  Valid fields for '{self._entity_type_str}':"
    for vf in valid_fields:
        field_info = schema.fields[vf]
        error_msg += f"\n    {vf}: {field_info.field_type.name}"
        if field_info.choices:
            error_msg += f" (choices: {field_info.choices})"

    raise QueryValidationError(error_msg)
```

**Error Message Example:**

```
QueryValidationError: Unknown field 'statsu' for entity type 'task'
  Did you mean: status

  Valid fields for 'task':
    created_at: DATETIME
    description: STRING
    entity_type: STRING
    id: STRING
    metadata: DICT
    modified_at: DATETIME
    priority: ENUM (choices: ['low', 'medium', 'high', 'critical'])
    properties: DICT
    status: ENUM (choices: ['pending', 'in_progress', 'completed', 'blocked'])
    title: STRING
    version: INTEGER
```

---

### Guardrail 3: Case-Insensitive Entity Type

```python
def entities(self, entity_type: str) -> "Query[T]":
    # Normalize to lowercase for consistency
    entity_type = entity_type.lower()

    # This allows Query(m).entities('TASK') == Query(m).entities('task')
```

---

### Guardrail 4: Entity Type Must Be Set Before Where

```python
def where(self, **conditions) -> "Query[T]":
    if not self._entity_type_str:
        raise QueryValidationError(
            "Must call .entities() or .tasks()/.sprints() before .where(). "
            "Example: Query(manager).entities('task').where(status='pending')"
        )
```

---

## Part 4: Implementation Code Snippets

### Snippet 1: entities() Method

```python
# In cortical/got/query_builder.py (class Query)

def entities(self, entity_type: str) -> "Query[T]":
    """
    Query entities by type name.

    This is a generic accessor that works for any registered entity type.
    Entity type is validated against the SchemaRegistry.

    Args:
        entity_type: Entity type name (case-insensitive).
                    Examples: 'task', 'sprint', 'decision', 'epic',
                             'knowledge_transfer', 'document'

    Returns:
        Self for method chaining

    Raises:
        QueryValidationError: If entity_type is not registered in SchemaRegistry

    Example:
        >>> # Generic accessor
        >>> Query(manager).entities('task').where(status='pending').execute()
        >>> Query(manager).entities('epic').limit(5).execute()
        >>>
        >>> # Equivalent to specific methods
        >>> Query(manager).tasks()  # same as .entities('task')
        >>> Query(manager).sprints()  # same as .entities('sprint')
    """
    from cortical.got.schema import get_registry
    from cortical.got.errors import QueryValidationError

    # Normalize to lowercase
    entity_type = entity_type.lower()

    # Validate against schema registry
    registry = get_registry()
    if not registry.has_schema(entity_type):
        available = sorted(registry._schemas.keys())

        # Try to suggest similar type
        import difflib
        suggestions = difflib.get_close_matches(entity_type, available, n=1, cutoff=0.6)

        error_msg = f"Unknown entity type '{entity_type}'."
        if suggestions:
            error_msg += f"\n  Did you mean: {suggestions[0]}"
        error_msg += "\n\n  Available entity types:\n"
        error_msg += "\n".join(f"    - {t}" for t in available)

        raise QueryValidationError(error_msg)

    # Store entity type as string (not enum)
    self._entity_type_str = entity_type
    self._entity_type = None  # Legacy enum field - keep for backward compat

    return self
```

---

### Snippet 2: Enhanced _get_base_entities()

```python
# In cortical/got/query_builder.py (class Query)

def _get_base_entities(self) -> List[Any]:
    """Get base entities based on entity type."""

    # Legacy enum-based dispatch (for backward compatibility)
    if self._entity_type == EntityType.TASK:
        return self._manager.list_all_tasks()
    elif self._entity_type == EntityType.SPRINT:
        return self._manager.list_sprints()
    elif self._entity_type == EntityType.DECISION:
        return self._manager.list_decisions()
    elif self._entity_type == EntityType.EDGE:
        return self._manager.list_edges()
    elif self._entity_type == EntityType.HANDOFF:
        return self._manager.list_handoffs()

    # String-based dispatch (for .entities() method)
    if self._entity_type_str:
        return self._get_base_entities_by_string(self._entity_type_str)

    return []

def _get_base_entities_by_string(self, entity_type: str) -> List[Any]:
    """Get base entities by entity type string."""

    # Map entity type to GoTManager method
    METHOD_MAP = {
        'task': 'list_all_tasks',
        'sprint': 'list_sprints',
        'decision': 'list_decisions',
        'edge': 'list_edges',
        'epic': 'list_epics',
        'handoff': 'list_handoffs',
        'document': 'list_documents',
        'claudemd_layer': 'list_claudemd_layers',
        # Add new entity types here as list methods are implemented
    }

    method_name = METHOD_MAP.get(entity_type)

    if method_name is None:
        # Entity type not in method map - use generic fallback
        logger.warning(
            f"No list method for entity type '{entity_type}', "
            f"using generic file-based fallback (slower)"
        )
        return self._get_base_entities_generic(entity_type)

    # Call the GoTManager method
    method = getattr(self._manager, method_name, None)
    if method is None:
        raise QueryExecutionError(
            f"GoTManager.{method_name}() not found. "
            f"This is a bug - method is in METHOD_MAP but not implemented."
        )

    return method()

def _get_base_entities_generic(self, entity_type: str) -> List[Any]:
    """
    Generic fallback for entity types without dedicated list methods.

    WARNING: This scans entity files directly and is slower than dedicated
    list methods. Implement a proper list_X() method in GoTManager for
    better performance.
    """
    from pathlib import Path
    import json
    from cortical.got.validation import validate_entity_file

    entities_dir = self._manager.got_dir / "entities"

    # Try to use ID prefix for faster scanning
    prefix = self._get_id_prefix(entity_type)
    pattern = f"{prefix}*.json" if prefix else "*.json"

    results = []
    entity_class = self._get_entity_class(entity_type)

    for file_path in entities_dir.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                wrapper = json.load(f)

            # Validate file structure
            is_valid, error = validate_entity_file(wrapper)
            if not is_valid:
                logger.warning(f"Invalid entity file {file_path}: {error}")
                continue

            data = wrapper.get('data', {})
            if data.get('entity_type') != entity_type:
                continue

            # Deserialize to proper entity class
            entity = entity_class.from_dict(data)
            results.append(entity)

        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            continue

    return results

def _get_id_prefix(self, entity_type: str) -> Optional[str]:
    """Get ID prefix for entity type (for faster file scanning)."""
    PREFIX_MAP = {
        'task': 'T',
        'decision': 'D',
        'sprint': 'S',
        'edge': 'E',
        'epic': 'EPIC',
        'handoff': 'H',
        'knowledge_transfer': 'KT',
        'document': 'DOC',
        'claudemd_layer': 'CML',
        'claudemd_version': 'CMV',
        # Add others as discovered
    }
    return PREFIX_MAP.get(entity_type)

def _get_entity_class(self, entity_type: str) -> Type[Entity]:
    """Get entity class for deserializing from dict."""
    from cortical.got.types import (
        Task, Decision, Sprint, Epic, Edge, Handoff,
        ClaudeMdLayer, ClaudeMdVersion, Document,
        # Import others as needed
    )

    CLASS_MAP = {
        'task': Task,
        'decision': Decision,
        'sprint': Sprint,
        'epic': Epic,
        'edge': Edge,
        'handoff': Handoff,
        'claudemd_layer': ClaudeMdLayer,
        'claudemd_version': ClaudeMdVersion,
        'document': Document,
        # Add others
    }

    entity_class = CLASS_MAP.get(entity_type)
    if entity_class is None:
        raise QueryExecutionError(
            f"No entity class mapped for type '{entity_type}'. "
            f"Add to CLASS_MAP in _get_entity_class()."
        )

    return entity_class
```

---

### Snippet 3: Enhanced where() with Field Validation

```python
# In cortical/got/query_builder.py (class Query)

def where(self, **conditions) -> "Query[T]":
    """
    Add WHERE conditions (AND) with field validation.

    Args:
        **conditions: Field=value pairs to filter by

    Returns:
        Self for chaining

    Raises:
        QueryValidationError: If called after .execute() or field is unknown

    Example:
        >>> Query(manager).entities('task').where(status='pending', priority='high')
        >>> Query(manager).tasks().where(statsu='pending')  # Raises error with suggestion
    """
    from cortical.got.schema import get_registry
    from cortical.got.errors import QueryValidationError

    self._validate_not_executed("where")

    # Validate fields if entity type is known
    if self._entity_type_str:
        registry = get_registry()
        schema = registry.get_schema(self._entity_type_str)

        if schema:
            self._validate_where_fields(schema, conditions)

    # Add conditions
    for field, value in conditions.items():
        self._where_clauses.append(WhereClause(field=field, value=value))

    return self

def _validate_where_fields(self, schema: BaseSchema, conditions: Dict[str, Any]) -> None:
    """Validate WHERE clause fields against schema."""
    import difflib

    valid_fields = list(schema.fields.keys())

    for field, value in conditions.items():
        if field not in schema.fields:
            # Unknown field - provide helpful error
            suggestions = difflib.get_close_matches(field, valid_fields, n=1, cutoff=0.6)

            error_lines = [
                f"Unknown field '{field}' for entity type '{self._entity_type_str}'"
            ]

            if suggestions:
                error_lines.append(f"  Did you mean: {suggestions[0]}")

            error_lines.append(f"\n  Valid fields for '{self._entity_type_str}':")
            for vf in sorted(valid_fields):
                field_info = schema.fields[vf]
                field_desc = f"    {vf}: {field_info.field_type.name}"
                if field_info.choices:
                    field_desc += f" (choices: {field_info.choices})"
                error_lines.append(field_desc)

            raise QueryValidationError("\n".join(error_lines))

        # TODO: Validate value type matches field type
        # For now, just validate field name exists
```

---

## Part 5: Test Data for Validation

### Real GoT Data Available

```
Entity counts (from .got/entities):
  Tasks: 336
  Decisions: 56
  Handoffs: 41
  Sprints: 46
  Epics: 14
  Edges: 434
  Knowledge Transfers: 32

Sample IDs:
  Task: T-20251231-230615-b1e918b6
  Decision: D-20251226-195140-84a14df0
  Sprint: S-021, S-20260102-231925-80abadeb
  Epic: EPIC-20251226-223250-d5dc7122
  Handoff: H-20251226-124311-1ddfc7f7
  KT: KT-20260101-093241
```

### Test Scenarios Using Real Data

```python
# tests/behavioral/test_generic_entity_accessor.py

def test_query_tasks_using_generic_accessor(manager):
    """Verify .entities('task') works same as .tasks()"""

    # Using specific method
    tasks_specific = Query(manager).tasks().limit(5).execute()

    # Using generic accessor
    tasks_generic = Query(manager).entities('task').limit(5).execute()

    assert len(tasks_specific) == len(tasks_generic)
    assert all(isinstance(t, Task) for t in tasks_generic)

def test_unknown_entity_type_error(manager):
    """Verify helpful error for unknown entity type"""

    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('tak').execute()

    error_msg = str(exc_info.value)
    assert "Unknown entity type 'tak'" in error_msg
    assert "Did you mean: task" in error_msg
    assert "Available entity types:" in error_msg

def test_unknown_field_error_with_suggestion(manager):
    """Verify helpful error for typo in field name"""

    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('task').where(statsu='pending').execute()

    error_msg = str(exc_info.value)
    assert "Unknown field 'statsu'" in error_msg
    assert "Did you mean: status" in error_msg
    assert "Valid fields for 'task'" in error_msg

def test_field_validation_uses_correct_schema(manager):
    """Verify field validation checks the right schema"""

    # 'priority' exists in Task but not in Sprint
    Query(manager).entities('task').where(priority='high').execute()  # OK

    with pytest.raises(QueryValidationError) as exc_info:
        Query(manager).entities('sprint').where(priority='high').execute()  # ERROR

    assert "Unknown field 'priority' for entity type 'sprint'" in str(exc_info.value)
```

---

## Part 6: Final Recommendations

### Implementation Order

1. **Phase 1: Core entities() method** (T-001-A)
   - Implement `.entities(entity_type)` with schema validation
   - Implement enhanced `_get_base_entities()` with string dispatch
   - Implement field validation in `.where()`
   - Support entity types with existing list methods (8 types)
   - Use generic fallback for others (4 types)

2. **Phase 2: Missing list methods** (separate tasks)
   - T-001-A-1: `GoTManager.list_knowledge_transfers()`
   - T-001-A-2: `GoTManager.list_claudemd_versions()`
   - T-001-A-3: `GoTManager.list_teams()`
   - T-001-A-4: `GoTManager.list_persona_profiles()`

3. **Phase 3: Performance optimization** (if needed)
   - Add caching for generic fallback
   - Consider indexing by entity_type
   - Monitor query performance

### Breaking Changes

**None.** The implementation is backward-compatible:

- Existing `.tasks()`, `.sprints()`, etc. continue to work
- EntityType enum still used internally for backward compat
- New `.entities()` method is additive, not replacing

### Documentation Needs

1. Update Query builder docstring with `.entities()` examples
2. Document entity type to list method mapping
3. Document ID prefix patterns in entity_schemas.py
4. Add migration guide for eventual enum deprecation

---

## Appendix: Validation Commands Run

```bash
# Schema introspection
python3 -c "from cortical.got.entity_schemas import ensure_schemas_registered; ..."

# ID prefix analysis
python3 -c "from pathlib import Path; import json; ..."

# Query builder inspection
python3 -c "from cortical.got.query_builder import Query; ..."

# GoTManager method discovery
python3 -c "from cortical.got.api import GoTManager; ..."

# Field validation testing
python3 -c "from cortical.got.schema import get_registry; import difflib; ..."
```

All commands executed successfully with results documented above.

---

## Verdict

**APPROVED WITH CONDITIONS**

The `.entities('type')` method **CAN be implemented** as designed with the following conditions:

1. ✅ Use string-based entity type (bypass EntityType enum)
2. ✅ Implement schema-based field validation
3. ⚠️ Accept generic fallback for 4 entity types without list methods
4. ✅ Provide helpful error messages with suggestions
5. ⚠️ Document known ID prefix patterns
6. ✅ Maintain backward compatibility with existing methods

**The design is sound. The infrastructure exists. Implementation can proceed.**

---

**Report Completed:** 2026-01-04
**Next Step:** Share findings with design document author and proceed with implementation
