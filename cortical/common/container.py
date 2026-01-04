"""
Dependency Injection Container for Cortical modules.

This container implements Inversion of Control (IoC) to make the
system adaptable to future needs without requiring data migration.

Key Insight:
    "Don't call us, we'll call you."

    Components don't instantiate their dependencies. Instead,
    dependencies are injected by the container. This allows
    swapping implementations without changing consumers.

Design Pattern:
    Service Locator + Dependency Injection hybrid.
    - Services are registered with the container
    - Components receive dependencies through constructors
    - Late binding allows runtime configuration

Benefits:
    1. Testability: Mock any component
    2. Flexibility: Swap implementations
    3. Extensibility: Add new implementations
    4. Migration: Change storage without changing logic

Usage:
    from cortical.common import Container, Lifecycle

    container = Container()

    # Register implementations
    container.register(StorageBackend, FileSystemStorage)
    container.register(Cache, LRUCache, lifecycle=Lifecycle.TRANSIENT)

    # Resolve dependencies
    storage = container.resolve(StorageBackend)
    cache = container.resolve(Cache)

This module provides the foundational IoC infrastructure used by
CEL, GoT, CDG, and other cortical subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
)


T = TypeVar('T')


class Lifecycle(Enum):
    """Lifecycle scope for registered services."""

    SINGLETON = auto()   # One instance, shared
    TRANSIENT = auto()   # New instance each time
    SCOPED = auto()      # One per scope (e.g., request)


@dataclass
class ServiceDescriptor(Generic[T]):
    """
    Describes a registered service.

    Attributes:
        service_type: The protocol/interface type
        implementation: Factory or type to create instances
        lifecycle: When to create new instances
        instance: Cached instance (for singleton)
    """

    service_type: Type[T]
    implementation: Union[Type[T], Callable[..., T]]
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    instance: Optional[T] = None
    dependencies: Dict[str, Type] = field(default_factory=dict)

    def get_instance(self, container: 'Container') -> T:
        """Get or create an instance."""
        if self.lifecycle == Lifecycle.SINGLETON:
            if self.instance is None:
                self.instance = self._create(container)
            return self.instance

        return self._create(container)

    def _create(self, container: 'Container') -> T:
        """Create a new instance with dependencies resolved."""
        # Resolve dependencies
        resolved_deps = {}
        for name, dep_type in self.dependencies.items():
            resolved_deps[name] = container.resolve(dep_type)

        # Create instance
        if callable(self.implementation):
            return self.implementation(**resolved_deps)
        else:
            return self.implementation(**resolved_deps)


class Container:
    """
    Dependency Injection Container.

    Central registry for all services. Manages lifecycles,
    resolves dependencies, and provides instances.

    Usage:
        container = Container()

        # Register implementations
        container.register(EventStore, FileSystemEventStore)
        container.register(Materializer, CachingMaterializer)

        # Resolve dependencies
        store = container.resolve(EventStore)
        materializer = container.resolve(Materializer)
    """

    def __init__(self):
        """Initialize empty container."""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None

    def register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T]],
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
        **dependencies: Type,
    ) -> 'Container':
        """
        Register a service implementation.

        Args:
            service_type: The protocol/interface to register
            implementation: Class or factory function
            lifecycle: Instance lifecycle
            dependencies: Named dependencies to inject

        Returns:
            Self for chaining
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifecycle=lifecycle,
            dependencies=dependencies,
        )
        return self

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
    ) -> 'Container':
        """
        Register a pre-created instance.

        Useful for externally created objects or test mocks.

        Args:
            service_type: The protocol/interface
            instance: The instance to use

        Returns:
            Self for chaining
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=type(instance),
            lifecycle=Lifecycle.SINGLETON,
            instance=instance,
        )
        return self

    def register_factory(
        self,
        name: str,
        factory: Callable[..., Any],
    ) -> 'Container':
        """
        Register a named factory function.

        Args:
            name: Factory identifier
            factory: Function to create instances

        Returns:
            Self for chaining
        """
        self._factories[name] = factory
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service to an instance.

        Args:
            service_type: The protocol/interface to resolve

        Returns:
            Instance of the requested type

        Raises:
            KeyError: If service not registered
        """
        if service_type not in self._services:
            raise KeyError(
                f"Service not registered: {service_type.__name__}"
            )

        descriptor = self._services[service_type]

        # Handle scoped lifecycle
        if (
            descriptor.lifecycle == Lifecycle.SCOPED
            and self._current_scope is not None
        ):
            scope_cache = self._scoped_instances.get(self._current_scope, {})
            if service_type in scope_cache:
                return scope_cache[service_type]

            instance = descriptor.get_instance(self)
            scope_cache[service_type] = instance
            self._scoped_instances[self._current_scope] = scope_cache
            return instance

        return descriptor.get_instance(self)

    def resolve_optional(self, service_type: Type[T]) -> Optional[T]:
        """Resolve a service, returning None if not registered."""
        try:
            return self.resolve(service_type)
        except KeyError:
            return None

    def create(self, name: str, **kwargs: Any) -> Any:
        """
        Create an instance using a named factory.

        Args:
            name: Factory name
            kwargs: Arguments to pass to factory

        Returns:
            Created instance

        Raises:
            KeyError: If factory not registered
        """
        if name not in self._factories:
            raise KeyError(f"Factory not registered: {name}")
        return self._factories[name](**kwargs)

    def begin_scope(self, scope_id: str) -> 'ScopeContext':
        """
        Begin a new dependency scope.

        Args:
            scope_id: Unique scope identifier

        Returns:
            Context manager for the scope
        """
        return ScopeContext(self, scope_id)

    def _enter_scope(self, scope_id: str) -> None:
        """Enter a scope (internal)."""
        self._current_scope = scope_id
        self._scoped_instances[scope_id] = {}

    def _exit_scope(self, scope_id: str) -> None:
        """Exit a scope (internal)."""
        self._scoped_instances.pop(scope_id, None)
        self._current_scope = None

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services

    def get_all_registered(self) -> Dict[str, str]:
        """Get all registered services for debugging."""
        return {
            svc.__name__: desc.implementation.__name__
            for svc, desc in self._services.items()
        }


class ScopeContext:
    """Context manager for dependency scopes."""

    def __init__(self, container: Container, scope_id: str):
        self._container = container
        self._scope_id = scope_id

    def __enter__(self) -> Container:
        self._container._enter_scope(self._scope_id)
        return self._container

    def __exit__(self, *args) -> None:
        self._container._exit_scope(self._scope_id)
