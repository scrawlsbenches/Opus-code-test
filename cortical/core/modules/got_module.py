"""
GoT Module - Graph of Thought Task Management.

Registers GoT services for task, decision, and knowledge management.

Services Provided:
    - TransactionManager: ACID transactions for GoT entities
    - GoTManager: High-level API for tasks, decisions, edges

Note: Index management is now handled by CDGIndexManager in the CDG layer.

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

from cortical.common import Container, ContainerModule, Lifecycle, FileSystem
from cortical.cdg.schema import SchemaRegistry


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
        from cortical.cdg.transaction_manager import CDGTransactionManager
        from cortical.got.api import GoTManager
        from cortical.got.types import create_entity_from_dict
        from cortical.cdg.config import CDGConfig

        # Register configuration
        container.register_instance(GoTConfig, self.config)

        # Register CDGTransactionManager directly (no GoT wrapper)
        def create_tx_manager() -> CDGTransactionManager:
            # Create GoT-specific CDG config
            cdg_config = CDGConfig.for_got()

            # For in-memory testing, disable WAL
            if self.config.use_memory:
                cdg_config.enable_wal = False

            return CDGTransactionManager(
                store_dir=self.config.got_dir / "entities",
                config=cdg_config,
                entity_factory=create_entity_from_dict,
            )

        container.register(
            CDGTransactionManager,
            create_tx_manager,
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register GoTManager with injected CDGTransactionManager and SchemaRegistry
        def create_got_manager() -> GoTManager:
            tx_manager = container.resolve(CDGTransactionManager)
            registry = container.resolve(SchemaRegistry)
            return GoTManager(
                self.config.got_dir,
                tx_manager=tx_manager,
                schema_registry=registry,
            )

        container.register(
            GoTManager,
            create_got_manager,
            lifecycle=Lifecycle.SINGLETON,
        )
