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
        from cortical.cdg.schema import SchemaRegistry
        from cortical.got.entity_schemas import register_all_schemas

        # Create registry instance (Container manages lifecycle)
        registry = SchemaRegistry()

        # Register all GoT entity schemas
        register_all_schemas(registry)

        # Register in container as singleton
        container.register_instance(SchemaRegistry, registry)
