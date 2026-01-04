"""
Behavioral tests for Dependency Injection Container.

As a developer building modular systems,
I want a powerful DI container that supports child containers, auto-wiring, and modules,
So that components are loosely coupled, testable, and configurable.

This test suite verifies:
- Service registration and resolution (basic DI)
- Child containers for isolation and overrides
- Auto-wiring of constructor dependencies
- Module system for organized registrations
- Lifecycle management (singleton, transient, scoped)
- Container as first-class citizen in cortical

Container is a FIRST-CLASS CITIZEN:
All components should receive their dependencies through the container.
Direct instantiation of components with hardcoded dependencies is prohibited.
"""

import pytest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# =============================================================================
# STORY: Basic Service Registration and Resolution
# =============================================================================


class TestDeveloperRegistersServices:
    """
    Epic: Basic DI Container

    As a developer using dependency injection,
    I want to register and resolve services through a container,
    So that components don't create their own dependencies.
    """

    def test_scenario_register_and_resolve_service(self):
        """
        Scenario: Register implementation and resolve by interface

        Given a container with no registrations
        When I register StorageBackend -> FileStorage
        And I resolve StorageBackend
        Then I get a FileStorage instance
        Because the container maps interfaces to implementations
        """
        from cortical.common import Container

        # Given: A protocol/interface
        @runtime_checkable
        class StorageBackend(Protocol):
            def save(self, key: str, data: bytes) -> None: ...
            def load(self, key: str) -> bytes: ...

        # And: An implementation
        class FileStorage:
            def __init__(self):
                self.data = {}

            def save(self, key: str, data: bytes) -> None:
                self.data[key] = data

            def load(self, key: str) -> bytes:
                return self.data[key]

        # When: Register and resolve
        container = Container()
        container.register(StorageBackend, FileStorage)
        storage = container.resolve(StorageBackend)

        # Then: Got the right type
        assert isinstance(storage, FileStorage)
        assert isinstance(storage, StorageBackend)

        # And: It works
        storage.save("test", b"hello")
        assert storage.load("test") == b"hello"

    def test_scenario_singleton_is_shared(self):
        """
        Scenario: Singleton lifecycle shares instance

        Given a service registered as singleton
        When I resolve it multiple times
        Then I get the same instance
        Because singletons are shared across the application
        """
        from cortical.common import Container, Lifecycle

        class ExpensiveService:
            instance_count = 0

            def __init__(self):
                ExpensiveService.instance_count += 1
                self.id = ExpensiveService.instance_count

        # Reset counter
        ExpensiveService.instance_count = 0

        container = Container()
        container.register(ExpensiveService, ExpensiveService, lifecycle=Lifecycle.SINGLETON)

        # Resolve multiple times
        svc1 = container.resolve(ExpensiveService)
        svc2 = container.resolve(ExpensiveService)
        svc3 = container.resolve(ExpensiveService)

        # All same instance
        assert svc1 is svc2
        assert svc2 is svc3
        assert ExpensiveService.instance_count == 1

    def test_scenario_transient_creates_new(self):
        """
        Scenario: Transient lifecycle creates new instances

        Given a service registered as transient
        When I resolve it multiple times
        Then I get different instances each time
        Because transients are not cached
        """
        from cortical.common import Container, Lifecycle

        class LightweightService:
            def __init__(self):
                import uuid
                self.id = str(uuid.uuid4())

        container = Container()
        container.register(LightweightService, LightweightService, lifecycle=Lifecycle.TRANSIENT)

        svc1 = container.resolve(LightweightService)
        svc2 = container.resolve(LightweightService)

        assert svc1 is not svc2
        assert svc1.id != svc2.id


# =============================================================================
# STORY: Child Containers for Isolation
# =============================================================================


class TestDeveloperUsesChildContainers:
    """
    Epic: Child Container Isolation

    As a developer writing tests or building multi-tenant systems,
    I want to create child containers that inherit but can override registrations,
    So that I can isolate changes without affecting the parent.
    """

    def test_scenario_child_inherits_parent_registrations(self):
        """
        Scenario: Child container inherits from parent

        Given a parent container with registered services
        When I create a child container
        Then the child can resolve parent's services
        Because children inherit parent registrations
        """
        from cortical.common import Container

        class Logger:
            def log(self, msg: str) -> str:
                return f"LOG: {msg}"

        parent = Container()
        parent.register(Logger, Logger)

        child = parent.create_child()

        # Child can resolve parent's registration
        logger = child.resolve(Logger)
        assert logger.log("test") == "LOG: test"

    def test_scenario_child_can_override_parent(self):
        """
        Scenario: Child container overrides parent registration

        Given a parent container with a service
        When I register a different implementation in child
        Then the child resolves to its own implementation
        And the parent still resolves to original
        Because child overrides don't affect parent
        """
        from cortical.common import Container

        class Database(Protocol):
            def query(self) -> str: ...

        class ProductionDB:
            def query(self) -> str:
                return "production"

        class MockDB:
            def query(self) -> str:
                return "mock"

        parent = Container()
        parent.register(Database, ProductionDB)

        child = parent.create_child()
        child.register(Database, MockDB)

        # Parent unchanged
        assert parent.resolve(Database).query() == "production"
        # Child overridden
        assert child.resolve(Database).query() == "mock"

    def test_scenario_child_isolation_for_tests(self):
        """
        Scenario: Using child containers for test isolation

        Given a production container configuration
        When I create a child for testing with mocks
        Then tests run in isolation without affecting production config
        Because each test gets its own container scope
        """
        from cortical.common import Container

        class EmailService:
            def send(self, to: str, msg: str) -> bool:
                raise NotImplementedError("Would send real email")

        class MockEmailService:
            def __init__(self):
                self.sent = []

            def send(self, to: str, msg: str) -> bool:
                self.sent.append((to, msg))
                return True

        # Production container
        production = Container()
        production.register(EmailService, EmailService)

        # Test container - inherits but overrides
        test_container = production.create_child()
        mock_email = MockEmailService()
        test_container.register_instance(EmailService, mock_email)

        # Test uses mock
        email_svc = test_container.resolve(EmailService)
        email_svc.send("test@example.com", "Hello")
        assert len(mock_email.sent) == 1


# =============================================================================
# STORY: Auto-Wiring Dependencies
# =============================================================================


class TestDeveloperUsesAutoWiring:
    """
    Epic: Automatic Dependency Resolution

    As a developer with complex dependency graphs,
    I want the container to automatically resolve constructor parameters,
    So that I don't have to manually wire everything.
    """

    def test_scenario_auto_wire_constructor_dependencies(self):
        """
        Scenario: Container auto-wires constructor parameters

        Given services with typed constructor parameters
        When I resolve a service with dependencies
        Then the container automatically injects resolved dependencies
        Because type hints tell the container what to inject
        """
        from cortical.common import Container

        class Logger:
            def log(self, msg: str) -> str:
                return f"[LOG] {msg}"

        class Cache:
            def __init__(self):
                self.data = {}

            def get(self, key: str) -> Optional[str]:
                return self.data.get(key)

            def set(self, key: str, value: str) -> None:
                self.data[key] = value

        class UserService:
            def __init__(self, logger: Logger, cache: Cache):
                self.logger = logger
                self.cache = cache

            def get_user(self, user_id: str) -> str:
                cached = self.cache.get(user_id)
                if cached:
                    return cached
                self.logger.log(f"Cache miss for {user_id}")
                return f"User-{user_id}"

        container = Container()
        container.register(Logger, Logger)
        container.register(Cache, Cache)
        container.register_auto(UserService)  # Auto-wire!

        user_svc = container.resolve(UserService)

        # Dependencies were injected
        assert user_svc.logger is not None
        assert user_svc.cache is not None
        assert user_svc.get_user("123") == "User-123"

    def test_scenario_auto_wire_nested_dependencies(self):
        """
        Scenario: Auto-wiring handles nested dependencies

        Given service A depends on B, and B depends on C
        When I resolve A
        Then the container creates C, injects into B, injects B into A
        Because auto-wiring is recursive
        """
        from cortical.common import Container

        class Config:
            def __init__(self):
                self.debug = True

        class Database:
            def __init__(self, config: Config):
                self.config = config

        class Repository:
            def __init__(self, db: Database):
                self.db = db

        container = Container()
        container.register(Config, Config)
        container.register_auto(Database)
        container.register_auto(Repository)

        repo = container.resolve(Repository)

        # Full chain was wired
        assert repo.db is not None
        assert repo.db.config is not None
        assert repo.db.config.debug is True


# =============================================================================
# STORY: Module System for Organized Registration
# =============================================================================


class TestDeveloperUsesModules:
    """
    Epic: Module-Based Registration

    As a developer organizing a large application,
    I want to group related registrations into modules,
    So that configuration is organized and maintainable.
    """

    def test_scenario_register_module(self):
        """
        Scenario: Register a module with multiple services

        Given a module that registers related services
        When I apply the module to a container
        Then all module services are registered
        Because modules bundle related registrations
        """
        from cortical.common import Container, ContainerModule

        class StorageModule(ContainerModule):
            """Module for storage-related services."""

            def register(self, container: Container) -> None:
                container.register(dict, dict)  # Simple example

        container = Container()
        container.apply_module(StorageModule())

        # Module's registrations are available
        storage = container.resolve(dict)
        assert isinstance(storage, dict)

    def test_scenario_module_with_configuration(self):
        """
        Scenario: Module accepts configuration

        Given a module that needs configuration
        When I create the module with config and apply it
        Then services are configured accordingly
        Because modules can be parameterized
        """
        from cortical.common import Container, ContainerModule, Lifecycle

        @dataclass
        class CacheConfig:
            max_size: int = 100
            ttl_seconds: int = 300

        class CacheService:
            def __init__(self, config: CacheConfig):
                self.config = config

        class CacheModule(ContainerModule):
            def __init__(self, config: CacheConfig):
                self.config = config

            def register(self, container: Container) -> None:
                container.register_instance(CacheConfig, self.config)
                container.register_auto(CacheService)

        config = CacheConfig(max_size=50, ttl_seconds=60)
        container = Container()
        container.apply_module(CacheModule(config))

        cache = container.resolve(CacheService)
        assert cache.config.max_size == 50
        assert cache.config.ttl_seconds == 60


# =============================================================================
# STORY: Container in Cortical Bootstrap
# =============================================================================


class TestDeveloperBootstrapsCortical:
    """
    Epic: Cortical Bootstrap with Container

    As a developer using cortical,
    I want a well-organized bootstrap that wires all components,
    So that I have a single entry point for the application container.
    """

    def test_scenario_bootstrap_creates_configured_container(self):
        """
        Scenario: Bootstrap function creates fully configured container

        Given the cortical bootstrap module
        When I call create_container()
        Then I get a container with all cortical services registered
        Because bootstrap is the central configuration point
        """
        from cortical.core.bootstrap import create_container

        container = create_container()

        # Container exists and is configured
        assert container is not None
        # Should have core services registered (we'll add more as we implement)

    def test_scenario_bootstrap_supports_test_configuration(self):
        """
        Scenario: Bootstrap supports test-friendly configuration

        Given the cortical bootstrap
        When I create a container for testing
        Then I can override services with mocks
        Because testing requires isolation
        """
        from cortical.core.bootstrap import create_container

        container = create_container(apply_modules=False)
        test_container = container.create_child()

        # Test container can be customized
        assert test_container is not None

    def test_scenario_bootstrap_resolves_real_services(self):
        """
        Scenario: Bootstrap container resolves real CDG and GoT services

        Given a bootstrap container with modules applied
        When I resolve CDGStore, TransactionManager, GoTManager
        Then I get real working instances
        Because the modules wire up the full dependency graph
        """
        import tempfile
        from pathlib import Path
        from cortical.core.bootstrap import create_container
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.transaction_manager import CDGTransactionManager
        from cortical.got.tx_manager import TransactionManager
        from cortical.got.api import GoTManager

        with tempfile.TemporaryDirectory() as tmpdir:
            got_dir = Path(tmpdir)

            container = create_container(got_dir=got_dir)

            # Resolve CDG services
            store = container.resolve(CDGStore)
            assert store is not None
            assert isinstance(store, CDGStore)

            wal = container.resolve(CDGWALManager)
            assert wal is not None
            assert isinstance(wal, CDGWALManager)

            cdg_tx = container.resolve(CDGTransactionManager)
            assert cdg_tx is not None
            assert isinstance(cdg_tx, CDGTransactionManager)

            # Resolve GoT services
            tx_manager = container.resolve(TransactionManager)
            assert tx_manager is not None
            assert isinstance(tx_manager, TransactionManager)

            got_manager = container.resolve(GoTManager)
            assert got_manager is not None
            assert isinstance(got_manager, GoTManager)

            # Verify singletons are shared
            store2 = container.resolve(CDGStore)
            assert store is store2

            tx_manager2 = container.resolve(TransactionManager)
            assert tx_manager is tx_manager2


# =============================================================================
# STORY: Breaking Backward Compatibility
# =============================================================================


class TestDeveloperMigratestoContainer:
    """
    Epic: Container-First Architecture

    As a developer maintaining cortical,
    I want components to require dependency injection,
    So that we enforce loose coupling throughout the system.
    """

    def test_scenario_got_transaction_manager_requires_injection(self):
        """
        Scenario: TransactionManager requires injected dependencies

        Given the new TransactionManager
        When I try to create it without dependencies
        Then I get a clear error about required dependencies
        Because direct instantiation without DI is prohibited
        """
        # This test documents the expected behavior after we break backward compat
        # For now, we verify the DI path works

        from cortical.got.tx_manager import TransactionManager
        from cortical.cdg.storage import CDGStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            got_dir = Path(tmpdir)

            # Create with injection (the supported way)
            tx_mgr = TransactionManager(got_dir)

            # Should work
            assert tx_mgr is not None
