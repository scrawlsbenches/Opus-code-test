"""
Dependency Injection Container for the Cognitive Event Lattice.

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

This module implements the foundational IoC infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
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

from .core.protocols import (
    CompactionStrategy,
    CognitiveLattice,
    EventReducer,
    EventStore,
    HealthMonitor,
    Materializer,
    MigrationEngine,
    SemanticIndex,
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


class LatticeBuilder:
    """
    Builder for constructing a complete CognitiveLattice.

    Provides fluent API for configuring all components
    and wiring them together.

    Usage:
        lattice = (
            LatticeBuilder()
            .with_storage(FileSystemEventStore, base_path=Path("./data"))
            .with_materializer(CachingMaterializer)
            .with_semantic_index(BloomSemanticIndex)
            .with_health_monitor(EventStoreHealthMonitor)
            .build()
        )
    """

    def __init__(self):
        """Initialize builder."""
        self._container = Container()
        self._storage_config: Dict[str, Any] = {}
        self._materializer_config: Dict[str, Any] = {}
        self._semantic_config: Dict[str, Any] = {}
        self._health_config: Dict[str, Any] = {}

    def with_storage(
        self,
        implementation: Type[EventStore],
        **config: Any,
    ) -> 'LatticeBuilder':
        """Configure event storage."""
        self._storage_config = {'impl': implementation, **config}
        return self

    def with_materializer(
        self,
        implementation: Type[Materializer],
        **config: Any,
    ) -> 'LatticeBuilder':
        """Configure entity materializer."""
        self._materializer_config = {'impl': implementation, **config}
        return self

    def with_semantic_index(
        self,
        implementation: Type[SemanticIndex],
        **config: Any,
    ) -> 'LatticeBuilder':
        """Configure semantic index."""
        self._semantic_config = {'impl': implementation, **config}
        return self

    def with_health_monitor(
        self,
        implementation: Type[HealthMonitor],
        **config: Any,
    ) -> 'LatticeBuilder':
        """Configure health monitoring."""
        self._health_config = {'impl': implementation, **config}
        return self

    def with_migration_engine(
        self,
        implementation: Type[MigrationEngine],
        **config: Any,
    ) -> 'LatticeBuilder':
        """Configure migration engine."""
        self._container.register(MigrationEngine, implementation)
        return self

    def with_compaction_strategy(
        self,
        implementation: Type[CompactionStrategy],
        **config: Any,
    ) -> 'LatticeBuilder':
        """Configure compaction strategy."""
        self._container.register(CompactionStrategy, implementation)
        return self

    def build(self) -> 'CognitiveLatticeImpl':
        """
        Build the configured lattice.

        Returns:
            Fully configured CognitiveLattice implementation
        """
        # Register and instantiate event store
        if 'impl' in self._storage_config:
            impl = self._storage_config.pop('impl')
            store = impl(**self._storage_config)
            self._container.register_instance(EventStore, store)
        else:
            raise ValueError("Event store not configured")

        # Register other components with their configs
        if 'impl' in self._materializer_config:
            impl = self._materializer_config.pop('impl')
            self._container.register(
                Materializer, impl,
                event_store=EventStore,
            )

        if 'impl' in self._semantic_config:
            impl = self._semantic_config.pop('impl')
            self._container.register(SemanticIndex, impl)

        if 'impl' in self._health_config:
            impl = self._health_config.pop('impl')
            self._container.register(
                HealthMonitor, impl,
                event_store=EventStore,
            )

        return CognitiveLatticeImpl(self._container)


class CognitiveLatticeImpl:
    """
    Implementation of the CognitiveLattice protocol.

    Orchestrates all components and provides unified access.

    Implements: CognitiveLattice protocol
    """

    def __init__(self, container: Container):
        """Initialize with configured container."""
        self._container = container

    @property
    def event_store(self) -> EventStore:
        """Get the event store."""
        return self._container.resolve(EventStore)

    @property
    def materializer(self) -> Materializer:
        """Get the materializer."""
        return self._container.resolve(Materializer)

    @property
    def semantic_index(self) -> Optional[SemanticIndex]:
        """Get the semantic index if configured."""
        return self._container.resolve_optional(SemanticIndex)

    @property
    def health_monitor(self) -> Optional[HealthMonitor]:
        """Get the health monitor if configured."""
        return self._container.resolve_optional(HealthMonitor)

    @property
    def migration_engine(self) -> Optional[MigrationEngine]:
        """Get the migration engine if configured."""
        return self._container.resolve_optional(MigrationEngine)

    @property
    def compaction_strategy(self) -> Optional[CompactionStrategy]:
        """Get the compaction strategy if configured."""
        return self._container.resolve_optional(CompactionStrategy)

    @property
    def current_horizon(self):
        """Get the current event horizon."""
        from .core.references import EventHorizon

        latest = self.event_store.latest()
        if latest is None:
            return EventHorizon(event_id="GENESIS", is_head=True)
        return EventHorizon(event_id=latest.value, is_head=True)

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve any service from the container."""
        return self._container.resolve(service_type)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_lattice(
    base_path: Path,
    expected_events: int = 10000,
) -> CognitiveLatticeImpl:
    """
    Create a lattice with sensible defaults.

    Args:
        base_path: Root directory for storage
        expected_events: Expected number of events (for tuning)

    Returns:
        Configured CognitiveLattice
    """
    from .wisdom.dag import FileSystemEventStore
    from .wisdom.materializer import (
        CachingMaterializer,
        default_reducer_registry,
    )
    from .wisdom.semantic import HybridSemanticIndex
    from .sanity.health import EventStoreHealthMonitor

    # Create event store
    store = FileSystemEventStore(base_path / "events")

    # Create reducer registry
    reducers = default_reducer_registry()

    # Create materializer
    materializer = CachingMaterializer(
        event_store=store,
        reducer_registry=reducers,
        cache_size=min(expected_events // 10, 1000),
    )

    # Create semantic index
    semantic = HybridSemanticIndex(
        base_path=base_path / "semantic",
        expected_concepts=expected_events,
    )

    # Create health monitor
    health = EventStoreHealthMonitor(store)

    # Build container
    container = Container()
    container.register_instance(EventStore, store)
    container.register_instance(Materializer, materializer)
    container.register_instance(SemanticIndex, semantic)
    container.register_instance(HealthMonitor, health)

    return CognitiveLatticeImpl(container)


def create_minimal_lattice(base_path: Path) -> CognitiveLatticeImpl:
    """
    Create a minimal lattice for testing.

    Only includes required components (storage, materializer).
    """
    from .wisdom.dag import FileSystemEventStore
    from .wisdom.materializer import (
        CachingMaterializer,
        default_reducer_registry,
    )

    store = FileSystemEventStore(base_path / "events")
    reducers = default_reducer_registry()
    materializer = CachingMaterializer(
        event_store=store,
        reducer_registry=reducers,
    )

    container = Container()
    container.register_instance(EventStore, store)
    container.register_instance(Materializer, materializer)

    return CognitiveLatticeImpl(container)


def create_lattice(
    container: Container,
    path: Union[str, Path] = ".got",
    **options: Any,
) -> CognitiveLatticeImpl:
    """
    Create a lattice with a pre-configured container.

    This is the primary factory function for creating a lattice
    with custom configuration via a container.

    Args:
        container: Pre-configured dependency container
        path: Base path for storage (default: ".got")
        **options: Additional configuration options

    Returns:
        Configured CognitiveLattice

    Example:
        >>> container = Container()
        >>> # Register custom implementations...
        >>> lattice = create_lattice(container, path=".got")
    """
    from .wisdom.dag import FileSystemEventStore
    from .wisdom.materializer import (
        CachingMaterializer,
        default_reducer_registry,
    )

    base_path = Path(path)

    # Only register defaults if not already registered
    if not container.is_registered(EventStore):
        store = FileSystemEventStore(base_path / "events")
        container.register_instance(EventStore, store)

    if not container.is_registered(Materializer):
        store = container.resolve(EventStore)
        reducers = default_reducer_registry()
        materializer = CachingMaterializer(
            event_store=store,
            reducer_registry=reducers,
            cache_size=options.get('cache_size', 1000),
        )
        container.register_instance(Materializer, materializer)

    return CognitiveLatticeImpl(container)


def create_high_performance_lattice(
    base_path: Path,
    expected_events: int = 100000,
    cache_size: int = 10000,
    batch_size: int = 100,
    snapshot_interval: int = 1000,
) -> CognitiveLatticeImpl:
    """
    Create a high-performance lattice optimized for large event volumes.

    Uses StreamingEventStore with:
    - Lazy loading (O(cache_size) memory vs O(all_events))
    - Write batching (amortized O(1) writes)
    - LRU caching (hot events stay in memory)
    - EntityIndex for O(1) entity lookups
    - OptimizedDAG for O(n log n) causal ordering

    Args:
        base_path: Root directory for storage
        expected_events: Expected number of events (for tuning indexes)
        cache_size: Event cache size (default 10000)
        batch_size: Write batch size (default 100)
        snapshot_interval: Events between snapshots (default 1000)

    Returns:
        High-performance CognitiveLattice

    Example:
        >>> # For 1M+ events
        >>> lattice = create_high_performance_lattice(
        ...     Path(".cel"),
        ...     expected_events=1_000_000,
        ...     cache_size=50000,
        ... )
    """
    from .performance import StreamingEventStore
    from .performance.streaming_store import StoreConfig
    from .wisdom.materializer import (
        CachingMaterializer,
        default_reducer_registry,
    )
    from .wisdom.semantic import HybridSemanticIndex
    from .sanity.health import EventStoreHealthMonitor

    # Configure streaming store
    store_config = StoreConfig(
        event_cache_size=cache_size,
        batch_size=batch_size,
        snapshot_interval=snapshot_interval,
        enable_entity_index=True,
        enable_concept_index=True,
        enable_temporal_index=True,
    )

    # Create high-performance event store
    store = StreamingEventStore(
        base_path=base_path / "events",
        config=store_config,
    )

    # Create reducer registry
    reducers = default_reducer_registry()

    # Create materializer with EntityIndex for O(1) entity lookups
    # The StreamingEventStore maintains an EntityIndex internally
    materializer = CachingMaterializer(
        event_store=store,
        reducer_registry=reducers,
        cache_size=min(expected_events // 10, 5000),
        entity_index=store._entity_index,  # Wire EntityIndex for O(1) lookups
    )

    # Create semantic index
    semantic = HybridSemanticIndex(
        base_path=base_path / "semantic",
        expected_concepts=expected_events,
    )

    # Create health monitor
    health = EventStoreHealthMonitor(store)

    # Build container
    container = Container()
    container.register_instance(EventStore, store)
    container.register_instance(Materializer, materializer)
    container.register_instance(SemanticIndex, semantic)
    container.register_instance(HealthMonitor, health)

    return CognitiveLatticeImpl(container)
