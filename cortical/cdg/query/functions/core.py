"""
Core CDG query functions.

These functions provide basic operations for CDG queries:
- count(): Count entities
- exists(entity_id): Check if entity exists
- type_of(entity_id): Get entity type from ID
- fields(entity_type): List fields for entity type
- entity_types(): List all registered entity types

See: docs/design/cdg-query-language.md
"""

from typing import Any, Dict, List, Optional

from ..registry import FunctionRegistry, FunctionSignature, QueryFunction, QueryContext


@FunctionRegistry.register('count')
class CountFunction(QueryFunction):
    """
    Count entities in result set.

    Usage:
        count()  # Returns count of all entities (requires FROM clause context)

    Note: This function is typically used as a modifier on query results,
    not as a standalone function. The executor handles this specially.
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='count',
            description='Count entities in result set',
            required_args=[],
            optional_args={},
            returns='Integer count',
            category='core'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> int:
        from ..errors import QueryNotImplementedError
        raise QueryNotImplementedError(
            "count() as standalone function not yet implemented. "
            "Use with query context: FROM task WHERE status = 'pending' then count results",
            doc_reference="docs/design/cdg-query-language.md#core-functions"
        )


@FunctionRegistry.register('exists')
class ExistsFunction(QueryFunction):
    """
    Check if an entity exists.

    Usage:
        exists('T-123')  # Returns True/False
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='exists',
            description='Check if an entity exists',
            required_args=['entity_id'],
            optional_args={},
            returns='Boolean: True if entity exists',
            category='core'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> bool:
        if not args:
            return False

        entity_id = args[0]
        if context.store is None:
            return False

        entity = context.store.read(entity_id)
        return entity is not None


@FunctionRegistry.register('type_of')
class TypeOfFunction(QueryFunction):
    """
    Get the entity type from an entity ID.

    Uses SchemaRegistry to map ID prefix to entity type.

    Usage:
        type_of('T-123')  # Returns 'task'
        type_of('D-456')  # Returns 'decision'
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='type_of',
            description='Get entity type from ID prefix',
            required_args=['entity_id'],
            optional_args={},
            returns='String entity type name',
            category='core'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Optional[str]:
        from ..errors import QueryNotImplementedError

        if not args:
            return None

        entity_id = str(args[0])

        if context.schema_registry is None:
            raise QueryNotImplementedError(
                "type_of() requires SchemaRegistry to resolve entity type from ID prefix",
                doc_reference="docs/design/cdg-query-language.md"
            )

        # Extract prefix from ID and look up in schema registry
        prefix = self._extract_prefix(entity_id)
        if prefix:
            return context.schema_registry.get_entity_type_by_prefix(prefix)
        return None

    def _extract_prefix(self, entity_id: str) -> Optional[str]:
        """Extract the prefix portion of an entity ID."""
        # Handle multi-character prefixes like EPIC-, KT-, etc.
        for prefix in ['EPIC-', 'TEAM-', 'CML-', 'CMV-', 'KT-', 'PP-', 'DOC-']:
            if entity_id.startswith(prefix):
                return prefix

        # Single character prefixes
        if '-' in entity_id:
            parts = entity_id.split('-', 1)
            return parts[0] + '-'

        return None


@FunctionRegistry.register('fields')
class FieldsFunction(QueryFunction):
    """
    List fields for an entity type.

    Usage:
        fields('task')  # Returns list of field definitions
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='fields',
            description='List fields for an entity type',
            required_args=['entity_type'],
            optional_args={},
            returns='List of field names',
            category='core'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        from ..errors import QueryNotImplementedError

        if not args:
            return []

        entity_type = str(args[0]).lower()

        if context.schema_registry is None:
            raise QueryNotImplementedError(
                "fields() requires SchemaRegistry to list entity type fields",
                doc_reference="docs/design/cdg-query-language.md"
            )

        schema = context.schema_registry.get_schema(entity_type)
        if schema is None:
            return []

        return list(schema.fields.keys())


@FunctionRegistry.register('entity_types')
class EntityTypesFunction(QueryFunction):
    """
    List all registered entity types.

    Usage:
        entity_types()  # Returns ['task', 'decision', 'sprint', ...]
    """

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='entity_types',
            description='List all registered entity types',
            required_args=[],
            optional_args={},
            returns='List of entity type names',
            category='core'
        )

    def execute(
        self,
        context: QueryContext,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        from ..errors import QueryNotImplementedError

        if context.schema_registry is None:
            raise QueryNotImplementedError(
                "entity_types() requires SchemaRegistry to list registered entity types",
                doc_reference="docs/design/cdg-query-language.md"
            )

        return list(context.schema_registry.list_schemas().keys())
