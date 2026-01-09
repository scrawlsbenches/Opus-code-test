"""
CDG Module - Cortical Distributed Graph Foundation Layer.

Registers storage, WAL, and transaction services for the foundation layer.

Services Provided:
    - CDGStore: Entity storage with checksums
    - CDGWALManager: Write-ahead logging
    - CDGTransactionManager: ACID transactions
    - CDGRecoveryManager: Crash recovery
    - CDGIndexManager: Schema-based index maintenance

Usage:
    from cortical.core.modules import CDGModule

    container = Container()
    container.apply_module(CDGModule(base_dir=Path(".got")))

    store = container.resolve(CDGStore)
    tx_manager = container.resolve(CDGTransactionManager)
"""

from pathlib import Path
from typing import Optional

from cortical.common import Container, ContainerModule, Lifecycle, FileSystem


class CDGModule(ContainerModule):
    """
    Container module for CDG (Cortical Distributed Graph) services.

    The CDG layer provides the foundation storage and transaction
    infrastructure used by GoT and other higher-level systems.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        use_memory: bool = False,
        # Legacy parameter name for backward compatibility during refactor
        got_dir: Optional[Path] = None,
    ):
        """
        Initialize CDG module.

        Args:
            base_dir: Base directory for CDG storage
            use_memory: Use in-memory storage (for testing)
            got_dir: Legacy alias for base_dir (will be removed)
        """
        self.base_dir = base_dir or got_dir or Path(".got")
        self.use_memory = use_memory

    def register(self, container: Container) -> None:
        """Register CDG services with the container."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.transaction_manager import CDGTransactionManager
        from cortical.cdg.recovery import CDGRecoveryManager
        from cortical.cdg.config import CDGConfig
        from cortical.cdg.index_manager import CDGIndexManager
        from cortical.cdg.schema import SchemaRegistry

        # Create CDGConfig (the real one from cortical.cdg.config)
        config = CDGConfig()
        container.register_instance(CDGConfig, config)

        # Resolve FileSystem from container (registered by bootstrap)
        filesystem = container.resolve(FileSystem)

        # Resolve SchemaRegistry (registered by SchemaModule, applied before CDGModule)
        schema_registry = container.resolve(SchemaRegistry)

        # Create CDGIndexManager for schema-based indexes
        entities_dir = self.base_dir / "entities"
        if not self.use_memory:
            entities_dir.mkdir(parents=True, exist_ok=True)

        index_manager = CDGIndexManager(
            store_dir=entities_dir,
            schema_registry=schema_registry,
            config=config,
            filesystem=filesystem,
        )
        container.register_instance(CDGIndexManager, index_manager)

        # Register factory for store
        def create_store() -> CDGStore:
            if not self.use_memory:
                entities_dir.mkdir(parents=True, exist_ok=True)
            return CDGStore(
                entities_dir,
                config=config,
                filesystem=filesystem,
                schema_registry=schema_registry,
                index_manager=index_manager,
            )

        container.register_factory("cdg_store", create_store)

        # Register CDGStore as singleton using factory
        container.register(
            CDGStore,
            lambda: container.create("cdg_store"),
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register factory for WAL (skip in memory mode)
        if not self.use_memory:
            def create_wal() -> CDGWALManager:
                wal_dir = self.base_dir / "wal"
                wal_dir.mkdir(parents=True, exist_ok=True)
                return CDGWALManager(wal_dir, config=config)

            container.register_factory("cdg_wal", create_wal)

            container.register(
                CDGWALManager,
                lambda: container.create("cdg_wal"),
                lifecycle=Lifecycle.SINGLETON,
            )

        # Register transaction manager factory
        def create_tx_manager() -> CDGTransactionManager:
            # Resolve filesystem from container - it's already configured with entities_dir
            filesystem = container.resolve(FileSystem)
            return CDGTransactionManager(
                filesystem=filesystem,
                config=config,
            )

        container.register(
            CDGTransactionManager,
            create_tx_manager,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register recovery manager factory
        def create_recovery() -> CDGRecoveryManager:
            return CDGRecoveryManager(
                store_dir=self.base_dir,
                config=config,
                index_manager=index_manager,
            )

        container.register(
            CDGRecoveryManager,
            create_recovery,
            lifecycle=Lifecycle.SINGLETON,
        )
