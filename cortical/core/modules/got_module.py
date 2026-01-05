"""
GoT Module - Graph of Thought Task Management.

Registers GoT services for task, decision, and knowledge management.

Services Provided:
    - TransactionManager: ACID transactions for GoT entities
    - GoTManager: High-level API for tasks, decisions, edges
    - QueryIndexManager: Search and query indexing

Usage:
    from cortical.core.modules import GoTModule

    container = Container()
    container.apply_module(CDGModule(got_dir=Path(".got")))  # CDG first
    container.apply_module(GoTModule(got_dir=Path(".got")))

    tx_manager = container.resolve(TransactionManager)
    got = container.resolve(GoTManager)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cortical.common import Container, ContainerModule, Lifecycle, RealFileSystem, InMemoryFileSystem


@dataclass
class GoTConfig:
    """Configuration for GoT services."""

    got_dir: Path
    """Directory for GoT data (.got)."""

    enable_indexing: bool = True
    """Enable query indexing."""

    use_memory: bool = False
    """Use in-memory storage instead of disk (for testing)."""


class GoTModule(ContainerModule):
    """
    Container module for GoT (Graph of Thought) services.

    GoT provides task management, decision tracking, and knowledge
    organization built on top of CDG's transactional storage.

    Note: CDGModule should be applied first as GoT depends on CDG services.
    """

    def __init__(
        self,
        config: Optional[GoTConfig] = None,
        got_dir: Optional[Path] = None,
        use_memory: bool = False,
    ):
        """
        Initialize GoT module.

        Args:
            config: GoT configuration (preferred)
            got_dir: Shorthand for got_dir (creates default config)
            use_memory: Use in-memory storage (for testing)
        """
        if config is not None:
            self.config = config
        elif got_dir is not None:
            self.config = GoTConfig(got_dir=got_dir, use_memory=use_memory)
        else:
            # Default to .got in current directory
            self.config = GoTConfig(got_dir=Path(".got"), use_memory=use_memory)

    def register(self, container: Container) -> None:
        """Register GoT services with the container."""
        from cortical.got.tx_manager import TransactionManager
        from cortical.got.api import GoTManager
        from cortical.got.indexer import QueryIndexManager
        from cortical.got.config import DurabilityMode
        from cortical.got.versioned_store import _got_entity_factory
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.cdg.config import CDGConfig
        from cortical.utils.locking import ProcessLock

        # Register configuration
        container.register_instance(GoTConfig, self.config)

        # Register TransactionManager with injected dependencies
        def create_tx_manager() -> TransactionManager:
            # Create GoT-specific CDG config
            cdg_config = CDGConfig.for_got()

            # Select filesystem based on use_memory flag
            filesystem = InMemoryFileSystem() if self.config.use_memory else RealFileSystem()

            if self.config.use_memory:
                # In-memory storage for fast testing
                store = CDGStore(
                    self.config.got_dir / "entities",
                    config=cdg_config,
                    entity_factory=_got_entity_factory,
                    filesystem=filesystem,
                )
                wal = None
                # Create a no-op lock for in-memory mode (context manager protocol)
                class NoOpLock:
                    """No-op lock for in-memory storage."""
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def acquire(self, blocking=True, timeout=-1): return True
                    def release(self): pass
                lock = NoOpLock()
            else:
                # Create GoT-specific CDGStore with entity factory for type dispatch
                # This is separate from CDGModule's generic store
                entities_dir = self.config.got_dir / "entities"
                entities_dir.mkdir(parents=True, exist_ok=True)
                store = CDGStore(
                    entities_dir,
                    config=cdg_config,
                    entity_factory=_got_entity_factory,
                    filesystem=filesystem,
                )

                # Create WAL if enabled
                if cdg_config.enable_wal:
                    wal_dir = self.config.got_dir / "wal"
                    wal_dir.mkdir(parents=True, exist_ok=True)
                    wal = CDGWALManager(wal_dir, cdg_config)
                else:
                    wal = None

                # Create lock
                lock_path = self.config.got_dir / ".got.lock"
                lock = ProcessLock(lock_path)

            return TransactionManager(
                got_dir=self.config.got_dir,
                durability=DurabilityMode.BALANCED,
                store=store,
                wal=wal,
                lock=lock,
            )

        container.register(
            TransactionManager,
            create_tx_manager,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register GoTManager with injected TransactionManager
        def create_got_manager() -> GoTManager:
            tx_manager = container.resolve(TransactionManager)
            return GoTManager(
                self.config.got_dir,
                tx_manager=tx_manager,
            )

        container.register(
            GoTManager,
            create_got_manager,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register QueryIndexManager
        def create_index_manager() -> QueryIndexManager:
            tx_manager = container.resolve(TransactionManager)
            return QueryIndexManager(tx_manager)

        container.register(
            QueryIndexManager,
            create_index_manager,
            lifecycle=Lifecycle.SINGLETON,
        )
