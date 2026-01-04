"""
Common utilities shared across all cortical modules.

This package contains foundational components that are used by multiple
subsystems (CDG, CEL, GoT, etc.) but don't belong to any specific one.

Contents:
    - container: Dependency Injection Container for IoC patterns

Container is a FIRST-CLASS CITIZEN:
    All cortical components should receive dependencies through the container.
    Direct instantiation with hardcoded dependencies is prohibited.
    See cortical/core/bootstrap.py for the application container setup.
"""

from .container import (
    Container,
    ContainerModule,
    Lifecycle,
    ServiceDescriptor,
    ScopeContext,
)

__all__ = [
    'Container',
    'ContainerModule',
    'Lifecycle',
    'ServiceDescriptor',
    'ScopeContext',
]
