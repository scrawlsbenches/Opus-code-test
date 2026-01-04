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

from cortical.common import Container, ContainerModule, Lifecycle


@dataclass
class GoTConfig:
    """Configuration for GoT services."""

    got_dir: Path
    """Directory for GoT data (.got)."""

    enable_indexing: bool = True
    """Enable query indexing."""


class GoTModule(ContainerModule):
    """
    Container module for GoT (Graph of Thought) services.

    GoT provides task management, decision tracking, and knowledge
    organization built on top of CDG's transactional storage.

    Note: CDGModule should be applied first as GoT depends on CDG services.
    """

    def __init__(self, config: Optional[GoTConfig] = None, got_dir: Optional[Path] = None):
        """
        Initialize GoT module.

        Args:
            config: GoT configuration (preferred)
            got_dir: Shorthand for got_dir (creates default config)
        """
        if config is not None:
            self.config = config
        elif got_dir is not None:
            self.config = GoTConfig(got_dir=got_dir)
        else:
            # Default to .got in current directory
            self.config = GoTConfig(got_dir=Path(".got"))

    def register(self, container: Container) -> None:
        """Register GoT services with the container."""
        from cortical.got.tx_manager import TransactionManager
        from cortical.got.api import GoTManager
        from cortical.got.indexer import QueryIndexManager
        from cortical.got.config import DurabilityMode
        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager
        from cortical.utils.locking import ProcessLock

        # Register configuration
        container.register_instance(GoTConfig, self.config)

        # Register TransactionManager with injected dependencies
        def create_tx_manager() -> TransactionManager:
            # Try to get CDG services from container (if CDGModule was applied)
            store = container.resolve_optional(CDGStore)
            wal = container.resolve_optional(CDGWALManager)

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

        # Register GoTManager
        def create_got_manager() -> GoTManager:
            return GoTManager(self.config.got_dir)

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
