"""
CDG Module - Cortical Distributed Graph Foundation Layer.

Registers storage, WAL, and transaction services for the foundation layer.

Services Provided:
    - CDGStore: Entity storage with checksums
    - CDGWALManager: Write-ahead logging
    - CDGTransactionManager: ACID transactions
    - CDGRecoveryManager: Crash recovery

Usage:
    from cortical.core.modules import CDGModule

    container = Container()
    container.apply_module(CDGModule(got_dir=Path(".got")))

    store = container.resolve(CDGStore)
    tx_manager = container.resolve(CDGTransactionManager)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cortical.common import Container, ContainerModule, Lifecycle


@dataclass
class CDGConfig:
    """Configuration for CDG services."""

    base_dir: Path
    """Base directory for CDG storage."""

    wal_enabled: bool = True
    """Enable write-ahead logging."""

    fsync_on_commit: bool = True
    """Force fsync on commit for durability."""


class CDGModule(ContainerModule):
    """
    Container module for CDG (Cortical Distributed Graph) services.

    The CDG layer provides the foundation storage and transaction
    infrastructure used by GoT and other higher-level systems.
    """

    def __init__(self, config: Optional[CDGConfig] = None, got_dir: Optional[Path] = None):
        """
        Initialize CDG module.

        Args:
            config: CDG configuration (preferred)
            got_dir: Shorthand for base_dir (creates default config)
        """
        if config is not None:
            self.config = config
        elif got_dir is not None:
            self.config = CDGConfig(base_dir=got_dir)
        else:
            # Default to .got in current directory
            self.config = CDGConfig(base_dir=Path(".got"))

    def register(self, container: Container) -> None:
        """Register CDG services with the container."""
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.transaction_manager import CDGTransactionManager
        from cortical.cdg.recovery import CDGRecoveryManager
        from cortical.cdg.config import CDGConfig as CDGInternalConfig

        # Register configuration
        container.register_instance(CDGConfig, self.config)

        # Create internal CDG config from our config
        internal_config = CDGInternalConfig()
        container.register_instance(CDGInternalConfig, internal_config)

        # Register factory for CDGStore (needs path)
        def create_store() -> CDGStore:
            entities_dir = self.config.base_dir / "entities"
            entities_dir.mkdir(parents=True, exist_ok=True)
            return CDGStore(entities_dir, config=internal_config)

        container.register_factory("cdg_store", create_store)

        # Register factory for WAL
        def create_wal() -> CDGWALManager:
            wal_dir = self.config.base_dir / "wal"
            wal_dir.mkdir(parents=True, exist_ok=True)
            return CDGWALManager(wal_dir, config=internal_config)

        container.register_factory("cdg_wal", create_wal)

        # Register CDGStore as singleton using factory
        container.register(
            CDGStore,
            lambda: container.create("cdg_store"),
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register WAL as singleton using factory
        container.register(
            CDGWALManager,
            lambda: container.create("cdg_wal"),
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register transaction manager factory
        def create_tx_manager() -> CDGTransactionManager:
            # CDGTransactionManager creates its own store and wal internally
            # when given store_dir. Pass store_dir, not individual components.
            store_dir = self.config.base_dir / "entities"
            store_dir.mkdir(parents=True, exist_ok=True)
            return CDGTransactionManager(
                store_dir=store_dir,
                config=internal_config,
            )

        container.register(
            CDGTransactionManager,
            create_tx_manager,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register recovery manager factory
        def create_recovery() -> CDGRecoveryManager:
            return CDGRecoveryManager(
                store_dir=self.config.base_dir,
                config=internal_config,
            )

        container.register(
            CDGRecoveryManager,
            create_recovery,
            lifecycle=Lifecycle.SINGLETON,
        )
