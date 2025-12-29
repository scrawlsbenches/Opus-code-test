"""
Unit tests for cortical/cel/container.py

Tests for the Dependency Injection Container implementation:
- Lifecycle enum
- ServiceDescriptor class
- Container class (registration, resolution, factories, scopes)
- ScopeContext context manager
- LatticeBuilder fluent API
- CognitiveLatticeImpl facade
- Factory functions
"""

import pytest
from pathlib import Path
from typing import Protocol
from unittest.mock import MagicMock, patch


# =============================================================================
# LIFECYCLE ENUM TESTS
# =============================================================================


class TestLifecycleEnum:
    """Tests for Lifecycle enum."""

    def test_lifecycle_singleton(self):
        """SINGLETON lifecycle is defined."""
        from cortical.cel.container import Lifecycle

        assert hasattr(Lifecycle, 'SINGLETON')
        assert Lifecycle.SINGLETON is not None

    def test_lifecycle_transient(self):
        """TRANSIENT lifecycle is defined."""
        from cortical.cel.container import Lifecycle

        assert hasattr(Lifecycle, 'TRANSIENT')
        assert Lifecycle.TRANSIENT is not None

    def test_lifecycle_scoped(self):
        """SCOPED lifecycle is defined."""
        from cortical.cel.container import Lifecycle

        assert hasattr(Lifecycle, 'SCOPED')
        assert Lifecycle.SCOPED is not None

    def test_all_lifecycles_unique(self):
        """All lifecycle values are unique."""
        from cortical.cel.container import Lifecycle

        values = [Lifecycle.SINGLETON, Lifecycle.TRANSIENT, Lifecycle.SCOPED]
        assert len(set(values)) == 3


# =============================================================================
# SERVICE DESCRIPTOR TESTS
# =============================================================================


class TestServiceDescriptor:
    """Tests for ServiceDescriptor dataclass."""

    def test_service_descriptor_creation(self):
        """ServiceDescriptor can be created with required fields."""
        from cortical.cel.container import ServiceDescriptor, Lifecycle

        class DummyService:
            pass

        descriptor = ServiceDescriptor(
            service_type=DummyService,
            implementation=DummyService,
        )

        assert descriptor.service_type is DummyService
        assert descriptor.implementation is DummyService
        assert descriptor.lifecycle == Lifecycle.SINGLETON
        assert descriptor.instance is None

    def test_service_descriptor_with_lifecycle(self):
        """ServiceDescriptor respects lifecycle parameter."""
        from cortical.cel.container import ServiceDescriptor, Lifecycle

        class DummyService:
            pass

        descriptor = ServiceDescriptor(
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.TRANSIENT,
        )

        assert descriptor.lifecycle == Lifecycle.TRANSIENT

    def test_service_descriptor_get_instance_singleton(self):
        """get_instance returns same instance for singleton."""
        from cortical.cel.container import ServiceDescriptor, Lifecycle, Container

        class DummyService:
            pass

        container = Container()
        descriptor = ServiceDescriptor(
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.SINGLETON,
        )

        instance1 = descriptor.get_instance(container)
        instance2 = descriptor.get_instance(container)

        assert instance1 is instance2

    def test_service_descriptor_get_instance_transient(self):
        """get_instance returns new instance for transient."""
        from cortical.cel.container import ServiceDescriptor, Lifecycle, Container

        class DummyService:
            pass

        container = Container()
        descriptor = ServiceDescriptor(
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.TRANSIENT,
        )

        instance1 = descriptor.get_instance(container)
        instance2 = descriptor.get_instance(container)

        assert instance1 is not instance2

    def test_service_descriptor_with_dependencies(self):
        """ServiceDescriptor can specify dependencies."""
        from cortical.cel.container import ServiceDescriptor, Lifecycle, Container

        class DepService:
            pass

        class MainService:
            def __init__(self, dep=None):
                self.dep = dep

        container = Container()
        container.register(DepService, DepService)

        descriptor = ServiceDescriptor(
            service_type=MainService,
            implementation=MainService,
            lifecycle=Lifecycle.TRANSIENT,
            dependencies={'dep': DepService},
        )

        instance = descriptor.get_instance(container)
        assert instance.dep is not None
        assert isinstance(instance.dep, DepService)

    def test_service_descriptor_factory_function(self):
        """ServiceDescriptor works with factory functions."""
        from cortical.cel.container import ServiceDescriptor, Lifecycle, Container

        class DummyService:
            def __init__(self, value):
                self.value = value

        def factory():
            return DummyService(42)

        container = Container()
        descriptor = ServiceDescriptor(
            service_type=DummyService,
            implementation=factory,
            lifecycle=Lifecycle.TRANSIENT,
        )

        instance = descriptor.get_instance(container)
        assert isinstance(instance, DummyService)
        assert instance.value == 42


# =============================================================================
# CONTAINER TESTS
# =============================================================================


class TestContainer:
    """Tests for Container class."""

    def test_container_initialization(self):
        """Container initializes with empty services."""
        from cortical.cel.container import Container

        container = Container()

        assert hasattr(container, '_services')
        assert len(container._services) == 0

    def test_container_register(self):
        """register adds service to container."""
        from cortical.cel.container import Container, Lifecycle

        class DummyService:
            pass

        container = Container()
        result = container.register(DummyService, DummyService)

        # Returns self for chaining
        assert result is container
        assert container.is_registered(DummyService)

    def test_container_register_with_lifecycle(self):
        """register respects lifecycle parameter."""
        from cortical.cel.container import Container, Lifecycle

        class DummyService:
            pass

        container = Container()
        container.register(DummyService, DummyService, lifecycle=Lifecycle.TRANSIENT)

        assert container._services[DummyService].lifecycle == Lifecycle.TRANSIENT

    def test_container_register_with_dependencies(self):
        """register accepts dependency mappings."""
        from cortical.cel.container import Container

        class DepService:
            pass

        class MainService:
            def __init__(self, dep=None):
                self.dep = dep

        container = Container()
        container.register(DepService, DepService)
        container.register(MainService, MainService, dep=DepService)

        instance = container.resolve(MainService)
        assert isinstance(instance.dep, DepService)

    def test_container_register_instance(self):
        """register_instance adds pre-created instance."""
        from cortical.cel.container import Container

        class DummyService:
            pass

        instance = DummyService()
        container = Container()
        result = container.register_instance(DummyService, instance)

        # Returns self for chaining
        assert result is container

        resolved = container.resolve(DummyService)
        assert resolved is instance

    def test_container_register_factory(self):
        """register_factory adds named factory."""
        from cortical.cel.container import Container

        class DummyService:
            def __init__(self, value):
                self.value = value

        def create_service(value):
            return DummyService(value)

        container = Container()
        result = container.register_factory('create_dummy', create_service)

        # Returns self for chaining
        assert result is container

        instance = container.create('create_dummy', value=42)
        assert isinstance(instance, DummyService)
        assert instance.value == 42

    def test_container_resolve(self):
        """resolve returns service instance."""
        from cortical.cel.container import Container

        class DummyService:
            pass

        container = Container()
        container.register(DummyService, DummyService)

        instance = container.resolve(DummyService)

        assert isinstance(instance, DummyService)

    def test_container_resolve_unregistered_raises(self):
        """resolve raises KeyError for unregistered service."""
        from cortical.cel.container import Container

        class DummyService:
            pass

        container = Container()

        with pytest.raises(KeyError) as exc_info:
            container.resolve(DummyService)

        assert 'DummyService' in str(exc_info.value)

    def test_container_resolve_optional(self):
        """resolve_optional returns None for unregistered service."""
        from cortical.cel.container import Container

        class DummyService:
            pass

        container = Container()

        result = container.resolve_optional(DummyService)
        assert result is None

    def test_container_resolve_optional_registered(self):
        """resolve_optional returns instance for registered service."""
        from cortical.cel.container import Container

        class DummyService:
            pass

        container = Container()
        container.register(DummyService, DummyService)

        result = container.resolve_optional(DummyService)
        assert isinstance(result, DummyService)

    def test_container_create(self):
        """create uses named factory to create instance."""
        from cortical.cel.container import Container

        def factory(label, count):
            return {'label': label, 'count': count}

        container = Container()
        container.register_factory('create_dict', factory)

        result = container.create('create_dict', label='test', count=5)

        assert result == {'label': 'test', 'count': 5}

    def test_container_create_unregistered_raises(self):
        """create raises KeyError for unregistered factory."""
        from cortical.cel.container import Container

        container = Container()

        with pytest.raises(KeyError) as exc_info:
            container.create('nonexistent')

        assert 'nonexistent' in str(exc_info.value)

    def test_container_is_registered(self):
        """is_registered checks service registration."""
        from cortical.cel.container import Container

        class DummyService:
            pass

        class OtherService:
            pass

        container = Container()
        container.register(DummyService, DummyService)

        assert container.is_registered(DummyService) is True
        assert container.is_registered(OtherService) is False

    def test_container_get_all_registered(self):
        """get_all_registered returns all service mappings."""
        from cortical.cel.container import Container

        class ServiceA:
            pass

        class ServiceB:
            pass

        container = Container()
        container.register(ServiceA, ServiceA)
        container.register(ServiceB, ServiceB)

        registered = container.get_all_registered()

        assert 'ServiceA' in registered
        assert 'ServiceB' in registered
        assert registered['ServiceA'] == 'ServiceA'
        assert registered['ServiceB'] == 'ServiceB'

    def test_container_chaining(self):
        """Container methods support method chaining."""
        from cortical.cel.container import Container

        class ServiceA:
            pass

        class ServiceB:
            pass

        container = (
            Container()
            .register(ServiceA, ServiceA)
            .register(ServiceB, ServiceB)
        )

        assert container.is_registered(ServiceA)
        assert container.is_registered(ServiceB)


# =============================================================================
# SCOPE CONTEXT TESTS
# =============================================================================


class TestScopeContext:
    """Tests for ScopeContext context manager."""

    def test_scope_context_enter_exit(self):
        """ScopeContext properly enters and exits scope."""
        from cortical.cel.container import Container, Lifecycle

        class ScopedService:
            pass

        container = Container()
        container.register(ScopedService, ScopedService, lifecycle=Lifecycle.SCOPED)

        with container.begin_scope('test-scope') as ctx:
            # Should be in scope
            assert container._current_scope == 'test-scope'
            instance1 = ctx.resolve(ScopedService)
            instance2 = ctx.resolve(ScopedService)
            # Same instance within scope
            assert instance1 is instance2

        # Scope should be exited
        assert container._current_scope is None
        assert 'test-scope' not in container._scoped_instances

    def test_scope_context_different_scopes(self):
        """Different scopes get different instances."""
        from cortical.cel.container import Container, Lifecycle

        class ScopedService:
            pass

        container = Container()
        container.register(ScopedService, ScopedService, lifecycle=Lifecycle.SCOPED)

        with container.begin_scope('scope-1') as ctx1:
            instance1 = ctx1.resolve(ScopedService)

        with container.begin_scope('scope-2') as ctx2:
            instance2 = ctx2.resolve(ScopedService)

        # Different scopes, different instances
        assert instance1 is not instance2

    def test_scope_context_returns_container(self):
        """ScopeContext __enter__ returns the container."""
        from cortical.cel.container import Container

        container = Container()

        with container.begin_scope('test') as ctx:
            assert ctx is container


# =============================================================================
# LATTICE BUILDER TESTS
# =============================================================================


class TestLatticeBuilder:
    """Tests for LatticeBuilder fluent API."""

    def test_lattice_builder_initialization(self):
        """LatticeBuilder initializes with empty config."""
        from cortical.cel.container import LatticeBuilder

        builder = LatticeBuilder()

        assert hasattr(builder, '_container')
        assert hasattr(builder, '_storage_config')

    def test_lattice_builder_with_storage(self):
        """with_storage configures storage implementation."""
        from cortical.cel.container import LatticeBuilder
        from cortical.cel.core.protocols import EventStore

        class MockEventStore:
            pass

        builder = LatticeBuilder()
        result = builder.with_storage(MockEventStore, base_path=Path('/tmp'))

        # Returns self for chaining
        assert result is builder
        assert builder._storage_config['impl'] is MockEventStore

    def test_lattice_builder_with_materializer(self):
        """with_materializer configures materializer."""
        from cortical.cel.container import LatticeBuilder

        class MockMaterializer:
            pass

        builder = LatticeBuilder()
        result = builder.with_materializer(MockMaterializer)

        assert result is builder
        assert builder._materializer_config['impl'] is MockMaterializer

    def test_lattice_builder_with_semantic_index(self):
        """with_semantic_index configures semantic index."""
        from cortical.cel.container import LatticeBuilder

        class MockSemanticIndex:
            pass

        builder = LatticeBuilder()
        result = builder.with_semantic_index(MockSemanticIndex)

        assert result is builder
        assert builder._semantic_config['impl'] is MockSemanticIndex

    def test_lattice_builder_with_health_monitor(self):
        """with_health_monitor configures health monitor."""
        from cortical.cel.container import LatticeBuilder

        class MockHealthMonitor:
            pass

        builder = LatticeBuilder()
        result = builder.with_health_monitor(MockHealthMonitor)

        assert result is builder
        assert builder._health_config['impl'] is MockHealthMonitor

    def test_lattice_builder_with_migration_engine(self):
        """with_migration_engine configures migration engine."""
        from cortical.cel.container import LatticeBuilder
        from cortical.cel.core.protocols import MigrationEngine

        class MockMigrationEngine:
            pass

        builder = LatticeBuilder()
        result = builder.with_migration_engine(MockMigrationEngine)

        assert result is builder
        assert builder._container.is_registered(MigrationEngine)

    def test_lattice_builder_with_compaction_strategy(self):
        """with_compaction_strategy configures compaction."""
        from cortical.cel.container import LatticeBuilder
        from cortical.cel.core.protocols import CompactionStrategy

        class MockCompactionStrategy:
            pass

        builder = LatticeBuilder()
        result = builder.with_compaction_strategy(MockCompactionStrategy)

        assert result is builder
        assert builder._container.is_registered(CompactionStrategy)

    def test_lattice_builder_build_without_storage_raises(self):
        """build raises if storage not configured."""
        from cortical.cel.container import LatticeBuilder

        builder = LatticeBuilder()

        with pytest.raises(ValueError) as exc_info:
            builder.build()

        assert "Event store not configured" in str(exc_info.value)

    def test_lattice_builder_build_with_minimal_config(self, tmp_path):
        """build succeeds with minimal storage config."""
        from cortical.cel.container import LatticeBuilder
        from cortical.cel.wisdom.dag import FileSystemEventStore

        builder = LatticeBuilder()
        builder.with_storage(FileSystemEventStore, base_path=tmp_path / "events")

        lattice = builder.build()

        assert lattice is not None
        assert hasattr(lattice, 'event_store')

    def test_lattice_builder_chaining(self, tmp_path):
        """LatticeBuilder supports method chaining."""
        from cortical.cel.container import LatticeBuilder
        from cortical.cel.wisdom.dag import FileSystemEventStore
        from cortical.cel.wisdom.materializer import CachingMaterializer

        lattice = (
            LatticeBuilder()
            .with_storage(FileSystemEventStore, base_path=tmp_path / "events")
            .with_materializer(CachingMaterializer)
            .build()
        )

        assert lattice is not None


# =============================================================================
# COGNITIVE LATTICE IMPL TESTS
# =============================================================================


class TestCognitiveLatticeImpl:
    """Tests for CognitiveLatticeImpl facade."""

    def test_cognitive_lattice_impl_initialization(self):
        """CognitiveLatticeImpl initializes with container."""
        from cortical.cel.container import CognitiveLatticeImpl, Container

        container = Container()
        lattice = CognitiveLatticeImpl(container)

        assert lattice._container is container

    def test_cognitive_lattice_impl_event_store_property(self, tmp_path):
        """event_store property resolves EventStore."""
        from cortical.cel.container import CognitiveLatticeImpl, Container
        from cortical.cel.core.protocols import EventStore
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        container = Container()
        container.register_instance(EventStore, store)

        lattice = CognitiveLatticeImpl(container)

        assert lattice.event_store is store

    def test_cognitive_lattice_impl_materializer_property(self, tmp_path):
        """materializer property resolves Materializer."""
        from cortical.cel.container import CognitiveLatticeImpl, Container
        from cortical.cel.core.protocols import Materializer, EventStore
        from cortical.cel.wisdom.dag import FileSystemEventStore
        from cortical.cel.wisdom.materializer import CachingMaterializer, default_reducer_registry

        store = FileSystemEventStore(tmp_path / "events")
        reducers = default_reducer_registry()
        mat = CachingMaterializer(event_store=store, reducer_registry=reducers)

        container = Container()
        container.register_instance(EventStore, store)
        container.register_instance(Materializer, mat)

        lattice = CognitiveLatticeImpl(container)

        assert lattice.materializer is mat

    def test_cognitive_lattice_impl_optional_properties(self):
        """Optional properties return None if not configured."""
        from cortical.cel.container import CognitiveLatticeImpl, Container

        container = Container()
        lattice = CognitiveLatticeImpl(container)

        assert lattice.semantic_index is None
        assert lattice.health_monitor is None
        assert lattice.migration_engine is None
        assert lattice.compaction_strategy is None

    def test_cognitive_lattice_impl_resolve_method(self, tmp_path):
        """resolve method delegates to container."""
        from cortical.cel.container import CognitiveLatticeImpl, Container
        from cortical.cel.core.protocols import EventStore
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        container = Container()
        container.register_instance(EventStore, store)

        lattice = CognitiveLatticeImpl(container)

        resolved = lattice.resolve(EventStore)
        assert resolved is store

    def test_cognitive_lattice_impl_current_horizon(self, tmp_path):
        """current_horizon returns EventHorizon."""
        from cortical.cel.container import CognitiveLatticeImpl, Container
        from cortical.cel.core.protocols import EventStore
        from cortical.cel.wisdom.dag import FileSystemEventStore

        store = FileSystemEventStore(tmp_path / "events")

        container = Container()
        container.register_instance(EventStore, store)

        lattice = CognitiveLatticeImpl(container)
        horizon = lattice.current_horizon

        # Genesis horizon for empty store
        assert horizon.event_id == "GENESIS"
        assert horizon.is_head is True


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_create_default_lattice(self, tmp_path):
        """create_default_lattice creates fully configured lattice."""
        from cortical.cel.container import create_default_lattice

        lattice = create_default_lattice(tmp_path, expected_events=100)

        assert lattice is not None
        assert lattice.event_store is not None
        assert lattice.materializer is not None
        assert lattice.semantic_index is not None
        assert lattice.health_monitor is not None

    def test_create_minimal_lattice(self, tmp_path):
        """create_minimal_lattice creates lattice with minimal components."""
        from cortical.cel.container import create_minimal_lattice

        lattice = create_minimal_lattice(tmp_path)

        assert lattice is not None
        assert lattice.event_store is not None
        assert lattice.materializer is not None
        # Optional components not configured
        assert lattice.semantic_index is None
        assert lattice.health_monitor is None

    def test_create_lattice_with_container(self, tmp_path):
        """create_lattice uses pre-configured container."""
        from cortical.cel.container import create_lattice, Container
        from cortical.cel.core.protocols import EventStore
        from cortical.cel.wisdom.dag import FileSystemEventStore

        # Pre-configure container
        store = FileSystemEventStore(tmp_path / "events")
        container = Container()
        container.register_instance(EventStore, store)

        lattice = create_lattice(container, path=tmp_path)

        assert lattice is not None
        assert lattice.event_store is store

    def test_create_lattice_default_path(self, tmp_path):
        """create_lattice uses default path if not specified."""
        from cortical.cel.container import create_lattice, Container
        import os

        # Change to tmp_path to avoid creating files in project
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            container = Container()
            lattice = create_lattice(container)

            assert lattice is not None
            # Default path is ".got"
            assert (tmp_path / ".got" / "events").exists()
        finally:
            os.chdir(original_cwd)

    def test_create_lattice_custom_cache_size(self, tmp_path):
        """create_lattice respects cache_size option."""
        from cortical.cel.container import create_lattice, Container

        container = Container()
        lattice = create_lattice(container, path=tmp_path, cache_size=500)

        assert lattice is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestContainerIntegration:
    """Integration tests for container with real components."""

    def test_full_lattice_workflow(self, tmp_path):
        """Complete workflow using Container directly for full component setup."""
        from cortical.cel.container import Container, CognitiveLatticeImpl
        from cortical.cel.core.protocols import EventStore, Materializer, SemanticIndex, HealthMonitor
        from cortical.cel.wisdom.dag import FileSystemEventStore
        from cortical.cel.wisdom.materializer import CachingMaterializer, default_reducer_registry
        from cortical.cel.wisdom.semantic import HybridSemanticIndex
        from cortical.cel.sanity.health import EventStoreHealthMonitor

        # Create components manually to ensure proper wiring
        store = FileSystemEventStore(tmp_path / "events")
        reducers = default_reducer_registry()
        materializer = CachingMaterializer(event_store=store, reducer_registry=reducers)
        semantic = HybridSemanticIndex(base_path=tmp_path / "semantic", expected_concepts=100)
        health = EventStoreHealthMonitor(store)

        # Register all components
        container = Container()
        container.register_instance(EventStore, store)
        container.register_instance(Materializer, materializer)
        container.register_instance(SemanticIndex, semantic)
        container.register_instance(HealthMonitor, health)

        lattice = CognitiveLatticeImpl(container)

        # Verify all components are accessible
        assert lattice.event_store is not None
        assert lattice.materializer is not None
        assert lattice.semantic_index is not None
        assert lattice.health_monitor is not None

    def test_singleton_lifecycle_across_resolves(self):
        """Singleton lifecycle returns same instance across resolves."""
        from cortical.cel.container import Container, Lifecycle

        class SingletonService:
            instance_count = 0

            def __init__(self):
                SingletonService.instance_count += 1

        container = Container()
        container.register(SingletonService, SingletonService, lifecycle=Lifecycle.SINGLETON)

        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)
        instance3 = container.resolve(SingletonService)

        assert instance1 is instance2 is instance3
        assert SingletonService.instance_count == 1

    def test_transient_lifecycle_creates_new_instances(self):
        """Transient lifecycle creates new instance each time."""
        from cortical.cel.container import Container, Lifecycle

        class TransientService:
            instance_count = 0

            def __init__(self):
                TransientService.instance_count += 1

        container = Container()
        container.register(TransientService, TransientService, lifecycle=Lifecycle.TRANSIENT)

        instance1 = container.resolve(TransientService)
        instance2 = container.resolve(TransientService)
        instance3 = container.resolve(TransientService)

        assert instance1 is not instance2
        assert instance2 is not instance3
        assert TransientService.instance_count == 3

    def test_scoped_lifecycle_within_scope(self):
        """Scoped lifecycle reuses instance within scope."""
        from cortical.cel.container import Container, Lifecycle

        class ScopedService:
            pass

        container = Container()
        container.register(ScopedService, ScopedService, lifecycle=Lifecycle.SCOPED)

        with container.begin_scope('request-1'):
            instance1 = container.resolve(ScopedService)
            instance2 = container.resolve(ScopedService)
            assert instance1 is instance2  # Same within scope

        with container.begin_scope('request-2'):
            instance3 = container.resolve(ScopedService)
            instance4 = container.resolve(ScopedService)
            assert instance3 is instance4  # Same within new scope
            assert instance1 is not instance3  # Different across scopes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
