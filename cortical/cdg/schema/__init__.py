"""
Schema definition and validation for CDG entities.

Provides declarative schema definitions with:
- Version tracking (separate from entity versions)
- Field type validation
- Required/optional field handling
- Default values
- Migration support between schema versions

This module is the foundation for all entity validation in CDG.
Domain-specific schemas (e.g., GoT's TaskSchema) register themselves
with the SchemaRegistry to enable validation and migration.

Schema Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SchemaRegistry                            │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │  entity_type → Schema mapping                           ││
    │  │  "task" → TaskSchema(version=2)                         ││
    │  │  "decision" → DecisionSchema(version=1)                 ││
    │  │  "node" → NodeSchema(version=1)  (generic)              ││
    │  └─────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     BaseSchema                               │
    │  - schema_version: int                                      │
    │  - fields: Dict[str, Field]                                 │
    │  - validate(data) → ValidationResult                        │
    │  - migrate(data, from_version) → data                       │
    └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                       Field                                  │
    │  - name: str                                                │
    │  - field_type: FieldType                                    │
    │  - required: bool                                           │
    │  - default: Any                                             │
    │  - validator: Callable (optional)                           │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from cortical.cdg.schema import (
        BaseSchema, Field, FieldType, SchemaRegistry,
        get_registry, register_schema, validate_entity
    )

    # Define a schema
    class TaskSchema(BaseSchema):
        schema_version = 2
        entity_type = 'task'
        id_prefix = 'T-'
        fields = {
            'id': Field('id', FieldType.STRING, required=True),
            'title': Field('title', FieldType.STRING, required=True),
            'status': Field('status', FieldType.ENUM,
                           required=True,
                           choices=['pending', 'in_progress', 'completed']),
        }

        @classmethod
        def migrate_v1_to_v2(cls, data: Dict) -> Dict:
            if 'metadata' not in data:
                data['metadata'] = {}
            return data

    # Register schema
    register_schema('task', TaskSchema)

    # Validate data
    result = validate_entity('task', data)
    if not result.valid:
        print(result.errors)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Type, Tuple
)


class FieldType(Enum):
    """Supported field types for schema validation."""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    ENUM = auto()
    DATETIME = auto()  # ISO 8601 string
    ANY = auto()       # No type validation


@dataclass
class Field:
    """
    Schema field definition.

    Defines a single field in an entity schema with type, requirements,
    defaults, and optional custom validation.

    Index support:
        Fields can be indexed for fast lookups. Set indexed=True to create
        an index on this field. The index_type determines the index structure:
        - "hash": Fast equality lookups (default)
        - "btree": Range queries, ordering
        - "fulltext": Text search (for STRING fields)

    Example:
        Field('status', FieldType.ENUM, indexed=True, choices=['pending', 'done'])
    """

    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None  # For ENUM type
    item_type: Optional[FieldType] = None  # For LIST type
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    indexed: bool = False  # Whether to maintain an index on this field
    index_type: str = "hash"  # Index type: "hash", "btree", "fulltext"

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this field definition.

        Args:
            value: The value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required
        if value is None:
            if self.required:
                return False, f"Field '{self.name}' is required"
            return True, None

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Enum choices validation
        if self.field_type == FieldType.ENUM and self.choices:
            if value not in self.choices:
                return False, (
                    f"Field '{self.name}' must be one of {self.choices}, "
                    f"got '{value}'"
                )

        # Custom validator
        if self.validator:
            try:
                if not self.validator(value):
                    return False, f"Field '{self.name}' failed custom validation"
            except Exception as e:
                return False, f"Field '{self.name}' validation error: {e}"

        return True, None

    def _validate_type(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate value type."""
        expected_types = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),
            FieldType.BOOLEAN: bool,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.ENUM: str,  # Enums are stored as strings
            FieldType.DATETIME: str,  # ISO format strings
            FieldType.ANY: object,
        }

        expected = expected_types.get(self.field_type, object)

        if not isinstance(value, expected):
            return False, (
                f"Field '{self.name}' expected {self.field_type.name}, "
                f"got {type(value).__name__}"
            )

        # For LIST type, validate item types if specified
        if self.field_type == FieldType.LIST and self.item_type and value:
            item_expected = expected_types.get(self.item_type, object)
            for i, item in enumerate(value):
                if not isinstance(item, item_expected):
                    return False, (
                        f"Field '{self.name}[{i}]' expected {self.item_type.name}, "
                        f"got {type(item).__name__}"
                    )

        return True, None

    def apply_default(self, data: Dict[str, Any]) -> None:
        """Apply default value to data if field is missing."""
        if self.name not in data and self.default is not None:
            # Deep copy for mutable defaults
            if isinstance(self.default, (list, dict)):
                data[self.name] = copy.deepcopy(self.default)
            else:
                data[self.name] = self.default


@dataclass
class ValidationResult:
    """Result of schema validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    migrated: bool = False
    from_version: Optional[int] = None
    to_version: Optional[int] = None

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


class BaseSchema:
    """
    Base class for entity schemas.

    Subclasses define:
    - schema_version: int - Current schema version
    - entity_type: str - The entity type this schema validates
    - id_prefix: str - ID prefix for this entity type (e.g., 'T-' for tasks)
    - fields: Dict[str, Field] - Field definitions
    - migrations: Dict[int, Callable] - Version migration functions

    Example:
        class TaskSchema(BaseSchema):
            schema_version = 2
            entity_type = 'task'
            id_prefix = 'T-'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
                'title': Field('title', FieldType.STRING, required=True),
                'status': Field('status', FieldType.ENUM, choices=['pending', 'completed']),
            }
            indexes = ['status', ('priority', 'created_at')]  # Single and composite

            @classmethod
            def migrate_v1_to_v2(cls, data: Dict) -> Dict:
                data['new_field'] = 'default'
                return data
    """

    schema_version: int = 1
    entity_type: str = ""
    id_prefix: str = ""
    fields: Dict[str, Field] = {}
    indexes: List[Any] = []  # List of field names or tuples for composite indexes

    # Schema version field name in data
    SCHEMA_VERSION_KEY = "_schema_version"

    @classmethod
    def get_migrations(cls) -> Dict[int, Callable[[Dict], Dict]]:
        """
        Get migration functions for this schema.

        Discovers methods named migrate_vN_to_vM where N < M.

        Returns:
            Dict mapping source version to migration function
        """
        migrations = {}
        for name in dir(cls):
            if name.startswith('migrate_v'):
                # Parse migrate_v1_to_v2 format
                try:
                    parts = name.replace('migrate_v', '').split('_to_v')
                    if len(parts) == 2:
                        from_v = int(parts[0])
                        migrations[from_v] = getattr(cls, name)
                except (ValueError, IndexError):
                    continue
        return migrations

    @classmethod
    def get_indexes(cls) -> List[Tuple[str, List[str]]]:
        """
        Get index definitions for this schema.

        Returns normalized list of (index_name, field_list) tuples.
        Single field indexes like 'status' become ('status_idx', ['status']).
        Composite indexes like ('priority', 'created_at') become
        ('priority_created_at_idx', ['priority', 'created_at']).

        Returns:
            List of (index_name, fields) tuples
        """
        result = []
        for idx_def in cls.indexes:
            if isinstance(idx_def, str):
                # Single field: 'status' -> ('status_idx', ['status'])
                result.append((f"{idx_def}_idx", [idx_def]))
            elif isinstance(idx_def, (tuple, list)):
                # Composite: ('a', 'b') -> ('a_b_idx', ['a', 'b'])
                fields = list(idx_def)
                name = "_".join(fields) + "_idx"
                result.append((name, fields))
        return result

    @classmethod
    def validate(cls, data: Dict[str, Any], strict: bool = False) -> ValidationResult:
        """
        Validate data against this schema.

        Args:
            data: Dictionary to validate
            strict: If True, reject unknown fields

        Returns:
            ValidationResult with valid flag and any errors
        """
        result = ValidationResult(valid=True)

        # Check each defined field
        for field_name, field_def in cls.fields.items():
            value = data.get(field_name)
            valid, error = field_def.validate(value)
            if not valid:
                result.add_error(error)

        # Check for unknown fields in strict mode
        if strict:
            known_fields = set(cls.fields.keys())
            # Also allow standard entity fields
            known_fields.update({
                'id', 'entity_type', 'version', 'created_at', 'modified_at',
                'partition_key', 'checksum', 'namespace', 'node_type',
                cls.SCHEMA_VERSION_KEY
            })
            unknown = set(data.keys()) - known_fields
            for field_name in unknown:
                result.add_warning(f"Unknown field '{field_name}'")

        return result

    @classmethod
    def apply_defaults(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values for missing optional fields.

        Args:
            data: Dictionary to modify

        Returns:
            Modified dictionary with defaults applied
        """
        for field_def in cls.fields.values():
            field_def.apply_default(data)
        return data

    @classmethod
    def migrate(
        cls,
        data: Dict[str, Any],
        from_version: Optional[int] = None
    ) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Migrate data from an older schema version.

        Args:
            data: Dictionary to migrate
            from_version: Source version (auto-detected if None)

        Returns:
            Tuple of (migrated_data, result)
        """
        result = ValidationResult(valid=True)

        # Detect current version
        if from_version is None:
            from_version = data.get(cls.SCHEMA_VERSION_KEY, 1)

        result.from_version = from_version
        result.to_version = cls.schema_version

        # No migration needed
        if from_version >= cls.schema_version:
            return data, result

        # Get migration chain
        migrations = cls.get_migrations()
        current_data = data.copy()
        current_version = from_version

        # Apply migrations in sequence
        while current_version < cls.schema_version:
            if current_version in migrations:
                try:
                    current_data = migrations[current_version](current_data)
                    result.migrated = True
                except Exception as e:
                    result.add_error(
                        f"Migration from v{current_version} failed: {e}"
                    )
                    return current_data, result
            current_version += 1

        # Update schema version
        current_data[cls.SCHEMA_VERSION_KEY] = cls.schema_version

        return current_data, result

    @classmethod
    def prepare_for_save(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for saving by applying defaults and schema version.

        Args:
            data: Dictionary to prepare

        Returns:
            Prepared dictionary
        """
        data = cls.apply_defaults(data)
        data[cls.SCHEMA_VERSION_KEY] = cls.schema_version
        return data


class OnDeleteAction(Enum):
    """Action to take when referenced entity is deleted."""
    CASCADE = auto()     # Delete referencing entity too
    SET_NULL = auto()    # Set reference field to None
    RESTRICT = auto()    # Prevent deletion if references exist
    NO_ACTION = auto()   # Allow deletion, leave dangling reference


@dataclass
class ReferenceRule:
    """
    Generic referential integrity rule.

    Defines how entity references should be validated and what happens
    when referenced entities are deleted.

    Examples:
        # Edge must reference existing entities
        ReferenceRule(
            field="from_id",
            must_exist=True,
            on_delete=OnDeleteAction.CASCADE
        )

        # Sprint can reference tasks (optional)
        ReferenceRule(
            field="task_ids",
            is_collection=True,
            must_exist=False,  # Tasks can be deleted
            on_delete=OnDeleteAction.SET_NULL,
            target_types=["task"]
        )
    """

    field: str
    """Field name containing the reference (e.g., 'from_id', 'sprint_id')."""

    must_exist: bool = True
    """If True, referenced entity must exist before write."""

    on_delete: OnDeleteAction = OnDeleteAction.RESTRICT
    """Action when referenced entity is deleted."""

    target_types: Optional[List[str]] = None
    """Allowed entity types for reference. None = any type."""

    is_collection: bool = False
    """If True, field is a list of references (e.g., task_ids)."""


class SchemaRegistry:
    """
    Registry for entity schemas.

    Provides centralized schema management with:
    - Schema registration by entity type
    - Validation dispatch
    - Migration orchestration
    - Referential integrity rules

    This is the central point for all schema operations in CDG.
    Domain-specific modules (GoT, CEL, etc.) register their schemas here.

    Lifecycle Management:
        SchemaRegistry is managed by the Container via SchemaModule.
        Use constructor injection to receive SchemaRegistry:

            class MyService:
                def __init__(self, schema_registry: SchemaRegistry):
                    self._registry = schema_registry

        For backward compatibility, get_registry() still works but
        Container injection is preferred for explicit dependencies.

    Usage:
        # Via Container (preferred)
        registry = container.resolve(SchemaRegistry)
        registry.register('task', TaskSchema)

        # Via global helper (backward compatible)
        registry = get_registry()
        result = registry.validate('task', data)
    """

    def __init__(self):
        """Initialize a new schema registry."""
        self._schemas: Dict[str, Type[BaseSchema]] = {}
        self._reference_rules: Dict[str, List[ReferenceRule]] = {}

    def register(self, entity_type: str, schema: Type[BaseSchema]) -> None:
        """
        Register a schema for an entity type.

        Args:
            entity_type: Entity type name (e.g., 'task')
            schema: Schema class
        """
        self._schemas[entity_type] = schema

    def get_schema(self, entity_type: str) -> Optional[Type[BaseSchema]]:
        """
        Get schema for an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Schema class or None if not registered
        """
        return self._schemas.get(entity_type)

    def has_schema(self, entity_type: str) -> bool:
        """Check if a schema is registered for an entity type."""
        return entity_type in self._schemas

    def get_all_indexes(self) -> List[Tuple[str, str, List[str]]]:
        """
        Get all index definitions from all registered schemas.

        Returns:
            List of (entity_type, index_name, fields) tuples.
            Used by IndexManager to create schema-defined indexes.

        Example:
            [('task', 'status_idx', ['status']),
             ('task', 'priority_created_at_idx', ['priority', 'created_at'])]
        """
        result = []
        for entity_type, schema in self._schemas.items():
            for index_name, fields in schema.get_indexes():
                result.append((entity_type, index_name, fields))
        return result

    def validate(
        self,
        entity_type: str,
        data: Dict[str, Any],
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate data against registered schema.

        Args:
            entity_type: Entity type to validate as
            data: Dictionary to validate
            strict: If True, reject unknown fields

        Returns:
            ValidationResult
        """
        schema = self.get_schema(entity_type)
        if schema is None:
            result = ValidationResult(valid=True)
            result.add_warning(f"No schema registered for '{entity_type}'")
            return result

        return schema.validate(data, strict=strict)

    def migrate(
        self,
        entity_type: str,
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Migrate data to current schema version.

        Args:
            entity_type: Entity type
            data: Dictionary to migrate

        Returns:
            Tuple of (migrated_data, result)
        """
        schema = self.get_schema(entity_type)
        if schema is None:
            return data, ValidationResult(valid=True)

        return schema.migrate(data)

    def prepare_for_save(
        self,
        entity_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for saving with defaults and version.

        Args:
            entity_type: Entity type
            data: Dictionary to prepare

        Returns:
            Prepared dictionary
        """
        schema = self.get_schema(entity_type)
        if schema is None:
            return data

        return schema.prepare_for_save(data)

    def list_schemas(self) -> Dict[str, int]:
        """
        List all registered schemas with versions.

        Returns:
            Dict mapping entity_type to schema_version
        """
        return {
            entity_type: schema.schema_version
            for entity_type, schema in self._schemas.items()
        }

    def get_schema_by_prefix(self, prefix: str) -> Optional[Type[BaseSchema]]:
        """
        Get schema for an entity ID prefix.

        Args:
            prefix: Entity ID prefix (e.g., 'T-', 'E-', 'KT-')

        Returns:
            Schema class or None if no schema matches the prefix
        """
        for schema in self._schemas.values():
            if hasattr(schema, 'id_prefix') and schema.id_prefix == prefix:
                return schema
        return None

    def get_entity_type_by_prefix(self, prefix: str) -> Optional[str]:
        """
        Get entity type for an entity ID prefix.

        Args:
            prefix: Entity ID prefix (e.g., 'T-', 'E-', 'KT-')

        Returns:
            Entity type string or None if no schema matches the prefix
        """
        schema = self.get_schema_by_prefix(prefix)
        if schema is not None:
            return schema.entity_type
        return None

    def get_prefix_for_entity_type(self, entity_type: str) -> Optional[str]:
        """
        Get ID prefix for an entity type.

        Args:
            entity_type: Entity type name (e.g., 'task', 'edge')

        Returns:
            ID prefix string or None if entity type not registered
        """
        schema = self.get_schema(entity_type)
        if schema is not None and hasattr(schema, 'id_prefix'):
            return schema.id_prefix
        return None

    def list_prefixes(self) -> Dict[str, str]:
        """
        List all registered entity ID prefixes.

        Returns:
            Dict mapping id_prefix to entity_type
        """
        return {
            schema.id_prefix: schema.entity_type
            for schema in self._schemas.values()
            if hasattr(schema, 'id_prefix') and schema.id_prefix
        }

    def clear(self) -> None:
        """Clear all registered schemas and rules (for testing)."""
        self._schemas.clear()
        self._reference_rules.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Reference Rule Management
    # ─────────────────────────────────────────────────────────────────────

    def add_reference_rule(
        self,
        entity_type: str,
        rule: ReferenceRule
    ) -> None:
        """
        Add a referential integrity rule for an entity type.

        Args:
            entity_type: Entity type the rule applies to
            rule: The reference rule to add

        Example:
            registry.add_reference_rule('edge', ReferenceRule(
                field='from_id',
                must_exist=True,
                on_delete=OnDeleteAction.CASCADE
            ))
        """
        if entity_type not in self._reference_rules:
            self._reference_rules[entity_type] = []
        self._reference_rules[entity_type].append(rule)

    def get_reference_rules(self, entity_type: str) -> List[ReferenceRule]:
        """
        Get all reference rules for an entity type.

        Args:
            entity_type: Entity type to get rules for

        Returns:
            List of ReferenceRule objects (empty if none defined)
        """
        return self._reference_rules.get(entity_type, [])

    def get_referencing_rules(
        self,
        target_type: str
    ) -> List[Tuple[str, ReferenceRule]]:
        """
        Get all rules that reference a given entity type.

        Useful for determining what happens when an entity is deleted.

        Args:
            target_type: Entity type being referenced

        Returns:
            List of (entity_type, rule) tuples for rules that target this type
        """
        result = []
        for entity_type, rules in self._reference_rules.items():
            for rule in rules:
                if rule.target_types is None or target_type in rule.target_types:
                    result.append((entity_type, rule))
        return result

    def get_on_delete_actions(
        self,
        target_type: str
    ) -> Dict[str, List[Tuple[str, OnDeleteAction]]]:
        """
        Get on-delete actions grouped by referencing entity type.

        Args:
            target_type: Entity type being deleted

        Returns:
            Dict mapping referencing entity_type to list of (field, action) tuples
        """
        result: Dict[str, List[Tuple[str, OnDeleteAction]]] = {}
        for entity_type, rule in self.get_referencing_rules(target_type):
            if entity_type not in result:
                result[entity_type] = []
            result[entity_type].append((rule.field, rule.on_delete))
        return result


__all__ = [
    # Types
    'FieldType',
    'Field',
    'ValidationResult',
    'BaseSchema',
    'SchemaRegistry',
    # Referential Integrity
    'OnDeleteAction',
    'ReferenceRule',
]
