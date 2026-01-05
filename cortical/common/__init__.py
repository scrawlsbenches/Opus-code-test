"""
Common utilities shared across all cortical modules.

This package contains foundational components that are used by multiple
subsystems (CDG, CEL, GoT, etc.) but don't belong to any specific one.

Contents:
    - container: Dependency Injection Container for IoC patterns
    - filesystem: FileSystem abstraction for testable I/O

Container is a FIRST-CLASS CITIZEN:
    All cortical components should receive dependencies through the container.
    Direct instantiation with hardcoded dependencies is prohibited.
    See cortical/core/bootstrap.py for the application container setup.

FileSystem Abstraction:
    Components that need disk I/O should depend on the FileSystem protocol,
    not on Path/open() directly. This enables fast in-memory testing.
"""

from .container import (
    Container,
    ContainerModule,
    Lifecycle,
    ServiceDescriptor,
    ScopeContext,
)

from .filesystem import (
    FileSystem,
    RealFileSystem,
    InMemoryFileSystem,
)

__all__ = [
    # Container
    'Container',
    'ContainerModule',
    'Lifecycle',
    'ServiceDescriptor',
    'ScopeContext',
    # FileSystem
    'FileSystem',
    'RealFileSystem',
    'InMemoryFileSystem',
]
