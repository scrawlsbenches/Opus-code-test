"""
Schema definition and validation for GoT entities.

MIGRATION NOTE (2026-01-06):
    Schema infrastructure has moved to CDG (cortical.cdg.schema).
    This module re-exports from CDG for backward compatibility.

    New code should import from cortical.cdg.schema directly:
        from cortical.cdg.schema import SchemaRegistry, BaseSchema, Field

    This re-export layer will be maintained for compatibility but
    CDG is now the canonical source for schema infrastructure.

See cortical.cdg.schema for full documentation.
"""

# Re-export everything from CDG schema for backward compatibility
from cortical.cdg.schema import (
    # Types
    FieldType,
    Field,
    ValidationResult,
    BaseSchema,
    SchemaRegistry,
    # Referential Integrity
    OnDeleteAction,
    ReferenceRule,
    # Functions
    get_registry,
    set_registry,
    register_schema,
    validate_entity,
    migrate_entity,
)

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
    # Functions
    'get_registry',
    'set_registry',
    'register_schema',
    'validate_entity',
    'migrate_entity',
]
