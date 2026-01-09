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
        # TODO(cdg-query): count() needs query context to count results
        # For now, return 0 as a placeholder
        # See: docs/design/cdg-query-language.md#core-functions
        return 0


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
        if not args:
            return None

        entity_id = str(args[0])

        # Try schema registry first
        if context.schema_registry is not None:
            # Extract prefix from ID
            prefix = self._extract_prefix(entity_id)
            if prefix:
                return context.schema_registry.get_entity_type_by_prefix(prefix)

        # Fallback to hardcoded mapping
        # TODO(cdg-query): Remove fallback once schema registry is always available
        return self._fallback_type_lookup(entity_id)

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

    def _fallback_type_lookup(self, entity_id: str) -> Optional[str]:
        """Fallback type lookup using hardcoded prefixes."""
        prefix_map = {
            'T-': 'task',
            'D-': 'decision',
            'S-': 'sprint',
            'EPIC-': 'epic',
            'E-': 'edge',
            'H-': 'handoff',
            'KT-': 'knowledge_transfer',
            'CML-': 'claudemd_layer',
            'CMV-': 'claudemd_version',
            'TEAM-': 'team',
            'PP-': 'persona_profile',
            'DOC-': 'document',
        }

        for prefix, entity_type in prefix_map.items():
            if entity_id.startswith(prefix):
                return entity_type

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
        if not args:
            return []

        entity_type = str(args[0]).lower()

        if context.schema_registry is None:
            # TODO(cdg-query): Return empty list without schema registry
            return []

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
        if context.schema_registry is None:
            # Fallback to known types
            return [
                'task', 'decision', 'sprint', 'epic', 'edge',
                'handoff', 'knowledge_transfer', 'document'
            ]

        return list(context.schema_registry.list_schemas().keys())
