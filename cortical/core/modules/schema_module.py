"""
Schema Module - Entity Schema Registry.

Registers the SchemaRegistry in the container, making it injectable
rather than accessed via global singleton.

Services Provided:
    - SchemaRegistry: Central registry for entity schemas

Usage:
    from cortical.core.modules import SchemaModule

    container = Container()
    container.apply_module(SchemaModule())

    registry = container.resolve(SchemaRegistry)

Why This Module Exists:
    The SchemaRegistry was originally a global singleton accessed via
    get_registry(). This works but hides the dependency, making code
    harder to test and reason about.

    By registering the registry in the container:
    1. Dependencies become explicit (constructor injection)
    2. Tests can provide mock registries via child containers
    3. Code that needs schema access declares that need
    4. We can track which components depend on schemas

Migration Path:
    1. Apply SchemaModule in bootstrap.py
    2. Update classes to receive SchemaRegistry via constructor
    3. Find and remove direct get_registry() calls
    4. Classes that bypass this reveal themselves as violations
"""

from cortical.common import Container, ContainerModule


class SchemaModule(ContainerModule):
    """
    Container module for Schema services.

    Registers the SchemaRegistry singleton in the container.
    The registry is populated with all entity schemas on registration.

    Note: This should be applied early in bootstrap, before modules
    that depend on schema validation (GoT, CEL, etc.).
    """

    def register(self, container: Container) -> None:
        """Register Schema services with the container."""
        # Import schema infrastructure from CDG (the foundation)
        from cortical.cdg.schema import SchemaRegistry, set_registry

        # Create a fresh registry instance (no singleton)
        registry = SchemaRegistry()

        # Set as the global registry for backward compatibility
        # This ensures code using get_registry() gets the same instance
        set_registry(registry)

        # Reset the schema registration flag since we have a new registry
        # This ensures schemas get registered to the new registry
        from cortical.got.entity_schemas import (
            ensure_schemas_registered,
            reset_schema_registration
        )
        reset_schema_registration()

        # Import GoT entity schemas to register domain-specific schemas
        # This populates: TaskSchema, DecisionSchema, SprintSchema, etc.
        # Must happen AFTER set_registry so schemas register to our instance
        ensure_schemas_registered()

        # Register in container as an instance (not a factory)
        # This makes SchemaRegistry injectable through constructor injection
        container.register_instance(SchemaRegistry, registry)
