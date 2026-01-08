"""
Index Initialization Module - Schema-Driven Index Creation.

Creates CDG indexes from schema definitions at startup. This module
bridges SchemaRegistry (which declares indexes) and IndexManager
(which implements them).

Services Consumed:
    - SchemaRegistry: Provides get_all_indexes() for schema-defined indexes
    - IndexManager: Receives create_index() calls

Why This Module Exists:
    Indexes are defined declaratively at the schema level (DDL-like):

        class TaskSchema(BaseSchema):
            indexes = ['status', 'priority']

    This module reads those declarations and creates the actual indexes
    in IndexManager. This separation keeps schemas pure (just definitions)
    while indexes get created automatically on startup.

Module Order:
    This module MUST be applied AFTER SchemaModule and CDGModule:
    1. SchemaModule - registers SchemaRegistry
    2. CDGModule - registers IndexManager
    3. IndexInitializationModule - wires them together

Usage:
    from cortical.core.modules import IndexInitializationModule

    container = Container()
    container.apply_module(SchemaModule())
    container.apply_module(CDGModule(got_dir=Path(".got")))
    container.apply_module(IndexInitializationModule())

    # At this point, indexes defined in schemas exist in IndexManager
    index_manager = container.resolve(IndexManager)
    assert index_manager.has_index("status_idx")  # Created from TaskSchema
"""

from cortical.common import Container, ContainerModule


class IndexInitializationModule(ContainerModule):
    """
    Container module that creates indexes from schema definitions.

    On registration, this module:
    1. Resolves SchemaRegistry from container
    2. Resolves IndexManager from container
    3. Iterates schema-defined indexes via get_all_indexes()
    4. Creates each index in IndexManager if it doesn't exist

    Idempotent: Safe to apply multiple times. Existing indexes are skipped.
    """

    def __init__(self, namespace: str | None = None):
        """
        Initialize index initialization module.

        Args:
            namespace: Optional namespace for index isolation (e.g., "got")
        """
        self._namespace = namespace

    def register(self, container: Container) -> None:
        """
        Create schema-defined indexes in IndexManager.

        Resolves SchemaRegistry and IndexManager from container,
        then creates indexes for each schema that defines them.
        """
        from cortical.cdg.schema import SchemaRegistry
        from cortical.cdg.index import IndexManager

        # Resolve dependencies
        schema_registry = container.resolve(SchemaRegistry)
        index_manager = container.resolve(IndexManager)

        # Get all schema-defined indexes
        all_indexes = schema_registry.get_all_indexes()

        # Create each index if it doesn't exist
        for entity_type, index_name, fields in all_indexes:
            # Prefix index name with entity type for uniqueness
            full_name = f"{entity_type}_{index_name}"

            if not index_manager.has_index(full_name):
                index_manager.create_index(
                    name=full_name,
                    fields=fields,
                )
