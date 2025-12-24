"""
Schema definition and validation for GoT entities.

Provides declarative schema definitions with:
- Version tracking (separate from entity versions)
- Field type validation
- Required/optional field handling
- Default values
- Migration support between schema versions

Schema Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    SchemaRegistry                        │
    │  ┌─────────────────────────────────────────────────────┐│
    │  │  entity_type → Schema mapping                       ││
    │  │  "task" → TaskSchema(version=2)                     ││
    │  │  "decision" → DecisionSchema(version=1)             ││
    │  └─────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     BaseSchema                           │
    │  - schema_version: int                                  │
    │  - fields: Dict[str, Field]                             │
    │  - validate(data) → ValidationResult                    │
    │  - migrate(data, from_version) → data                   │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                       Field                              │
    │  - name: str                                            │
    │  - field_type: FieldType                                │
    │  - required: bool                                       │
    │  - default: Any                                         │
    │  - validator: Callable (optional)                       │
    └─────────────────────────────────────────────────────────┘

Usage:
    # Define a schema
    class TaskSchema(BaseSchema):
        schema_version = 2
        fields = {
            'id': Field('id', FieldType.STRING, required=True),
            'title': Field('title', FieldType.STRING, required=True),
            'status': Field('status', FieldType.ENUM,
                           required=True,
                           choices=['pending', 'in_progress', 'completed', 'blocked']),
            'priority': Field('priority', FieldType.ENUM,
                             required=False,
                             default='medium',
                             choices=['low', 'medium', 'high', 'critical']),
        }

        @classmethod
        def migrate_v1_to_v2(cls, data: Dict) -> Dict:
            # Add new field with default
            if 'metadata' not in data:
                data['metadata'] = {}
            return data

    # Register schema
    registry = SchemaRegistry()
    registry.register('task', TaskSchema)

    # Validate data
    result = registry.validate('task', data)
    if not result.valid:
        print(result.errors)

    # Migrate data
    migrated = registry.migrate('task', data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Set, Type, Union, Tuple
)

from .errors import ValidationError


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
    """

    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None  # For ENUM type
    item_type: Optional[FieldType] = None  # For LIST type
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""

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
                import copy
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
    - fields: Dict[str, Field] - Field definitions
    - migrations: Dict[int, Callable] - Version migration functions

    Example:
        class TaskSchema(BaseSchema):
            schema_version = 2
            entity_type = 'task'
            fields = {
                'id': Field('id', FieldType.STRING, required=True),
                'title': Field('title', FieldType.STRING, required=True),
            }

            @classmethod
            def migrate_v1_to_v2(cls, data: Dict) -> Dict:
                data['new_field'] = 'default'
                return data
    """

    schema_version: int = 1
    entity_type: str = ""
    fields: Dict[str, Field] = {}

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


class SchemaRegistry:
    """
    Registry for entity schemas.

    Provides centralized schema management with:
    - Schema registration by entity type
    - Validation dispatch
    - Migration orchestration

    Usage:
        registry = SchemaRegistry()
        registry.register('task', TaskSchema)

        result = registry.validate('task', data)
        migrated = registry.migrate('task', data)
    """

    _instance: Optional[SchemaRegistry] = None

    def __new__(cls) -> SchemaRegistry:
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._schemas = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only on first creation)."""
        if not self._initialized:
            self._schemas: Dict[str, Type[BaseSchema]] = {}
            self._initialized = True

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

    def clear(self) -> None:
        """Clear all registered schemas (for testing)."""
        self._schemas.clear()


# Global registry instance
_registry = SchemaRegistry()


def get_registry() -> SchemaRegistry:
    """Get the global schema registry."""
    return _registry


def register_schema(entity_type: str, schema: Type[BaseSchema]) -> None:
    """Register a schema in the global registry."""
    _registry.register(entity_type, schema)


def validate_entity(
    entity_type: str,
    data: Dict[str, Any],
    strict: bool = False
) -> ValidationResult:
    """Validate data against global registry."""
    return _registry.validate(entity_type, data, strict=strict)


def migrate_entity(
    entity_type: str,
    data: Dict[str, Any]
) -> Tuple[Dict[str, Any], ValidationResult]:
    """Migrate data using global registry."""
    return _registry.migrate(entity_type, data)
