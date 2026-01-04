"""
Common utilities shared across all cortical modules.

This package contains foundational components that are used by multiple
subsystems (CDG, CEL, GoT, etc.) but don't belong to any specific one.

Contents:
    - container: Dependency Injection Container for IoC patterns
"""

from .container import (
    Container,
    Lifecycle,
    ServiceDescriptor,
    ScopeContext,
)

__all__ = [
    'Container',
    'Lifecycle',
    'ServiceDescriptor',
    'ScopeContext',
]
