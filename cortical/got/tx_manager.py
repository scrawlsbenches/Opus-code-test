"""
TransactionManager for GoT with ACID guarantees.

Orchestrates begin/commit/rollback with:
- Atomicity: All writes succeed or all fail
- Consistency: Checksums verify data integrity
- Isolation: Snapshot isolation via versioning
- Durability: WAL + fsync before commit

This module now delegates to CDGTransactionManager (Cortical Distributed Graph)
for core transaction operations, while maintaining GoT's API for backward
compatibility.

Migration Note (2025-12-31):
    TransactionManager now wraps CDGTransactionManager. All transaction operations
    are delegated to CDG, with GoT providing the entity factory for proper type
    dispatch (Task, Decision, Sprint, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .types import Entity
from .transaction import Transaction
from .config import DurabilityMode
from .versioned_store import _got_entity_factory
from .errors import TransactionError as GoTTransactionError

# Import CDG transaction infrastructure
from cortical.cdg.transaction_manager import CDGTransactionManager
from cortical.cdg.config import CDGConfig, DurabilityMode as CDGDurabilityMode
from cortical.cdg.errors import TransactionError as CDGTransactionError

# Import ProcessLock for backward compatibility (re-exported by __init__.py)
from cortical.utils.locking import ProcessLock

logger = logging.getLogger(__name__)


# Re-export CDG types for API compatibility
from cortical.cdg.transaction_manager import Conflict, CommitResult

# Also import RecoveryResult for recover() method
from cortical.cdg.recovery import RecoveryResult


def _convert_durability(durability: DurabilityMode) -> CDGDurabilityMode:
    """Convert GoT DurabilityMode to CDG DurabilityMode."""
    mapping = {
        DurabilityMode.RELAXED: CDGDurabilityMode.FAST,
        DurabilityMode.BALANCED: CDGDurabilityMode.BALANCED,
        DurabilityMode.PARANOID: CDGDurabilityMode.PARANOID,
    }
    return mapping.get(durability, CDGDurabilityMode.BALANCED)




class TransactionManager:
    """
    Manages transactions with ACID guarantees for GoT.

    - Atomicity: All writes in a TX succeed or all fail
    - Consistency: Checksums verify data integrity
    - Isolation: Snapshot isolation via versioning
    - Durability: WAL + fsync before commit

    Implementation Note:
        This class now wraps CDGTransactionManager, delegating transaction
        operations while maintaining GoT's API for backward compatibility.
        The entity factory (_got_entity_factory) ensures proper type dispatch
        so that reads return Task, Decision, Sprint, etc., not base Entity.
    """

    def __init__(self, got_dir: Path, durability: DurabilityMode = DurabilityMode.BALANCED):
        """
        Initialize transaction manager.

        Creates directories if needed:
        - {got_dir}/entities/
        - {got_dir}/wal/

        Runs recovery on startup.

        Args:
            got_dir: Base directory for GoT storage
            durability: Durability mode controlling fsync behavior
        """
        self.got_dir = Path(got_dir)
        self.got_dir.mkdir(parents=True, exist_ok=True)
        self.durability = durability

        # Create CDG configuration for GoT workloads
        cdg_config = CDGConfig.for_got()

        # Override durability mode from GoT config
        cdg_config.durability = _convert_durability(durability)

        # Manually set up CDG components to match GoT's directory structure:
        #   {got_dir}/entities/       # Entity storage
        #   {got_dir}/wal/            # Write-ahead log
        # We can't use CDGTransactionManager.__init__ directly because it hardcodes
        # wal_dir as store_dir/wal, which would create got_dir/entities/wal.
        # Instead, we manually create the components.

        from cortical.cdg.storage import CDGStore
        from cortical.cdg.wal import CDGWALManager

        # Create store in entities/ subdirectory
        self.store = CDGStore(
            self.got_dir / "entities",
            config=cdg_config,
            entity_factory=_got_entity_factory
        )

        # Create WAL at same level as entities/ (not inside it)
        self.wal = None
        if cdg_config.enable_wal:
            self.wal = CDGWALManager(self.got_dir / "wal", cdg_config)

        # Create lock
        self.lock = ProcessLock(self.got_dir / ".got.lock", reentrant=True)

        # Active transactions (in-memory only)
        self._active_tx = {}

        # Create a minimal CDG transaction manager wrapper
        # We create a CDGTransactionManager but override its store, wal, and lock
        # with our manually configured ones
        self._cdg_tx = CDGTransactionManager.__new__(CDGTransactionManager)
        self._cdg_tx.store_dir = self.got_dir
        self._cdg_tx.config = cdg_config
        self._cdg_tx.store = self.store
        self._cdg_tx.wal = self.wal
        self._cdg_tx.lock = self.lock
        self._cdg_tx._active_tx = self._active_tx

        # Run recovery on startup
        if cdg_config.auto_recover_on_startup:
            self._cdg_tx.recover()

    def begin(self) -> Transaction:
        """
        Start a new transaction.

        Returns:
            New Transaction object in ACTIVE state
        """
        return self._cdg_tx.begin()

    def read(self, tx: Transaction, entity_id: str) -> Optional[Entity]:
        """
        Read entity within transaction.

        Provides snapshot isolation:
        - First checks tx.write_set (see own writes)
        - Then reads from store at tx.snapshot_version
        - Records in tx.read_set for conflict detection

        Args:
            tx: Transaction context
            entity_id: Entity identifier

        Returns:
            Entity instance (Task, Decision, etc.) or None if not found
        """
        return self._cdg_tx.read(tx, entity_id)

    def write(self, tx: Transaction, entity: Entity) -> None:
        """
        Buffer a write within transaction.

        Logs to WAL and adds to tx.write_set.
        Does NOT apply to store until commit.

        Args:
            tx: Transaction context
            entity: Entity to write

        Raises:
            TransactionError: If transaction is not active
        """
        try:
            self._cdg_tx.write(tx, entity)
        except CDGTransactionError as e:
            # Re-raise as GoT TransactionError for API compatibility
            raise GoTTransactionError(str(e)) from e

    def commit(self, tx: Transaction) -> CommitResult:
        """
        Commit transaction.

        Steps:
        1. Acquire lock
        2. Set state to PREPARING
        3. Log TX_PREPARE to WAL
        4. Detect conflicts (version mismatch)
        5. If conflict: abort, return failure
        6. Apply writes atomically via store.apply_writes()
        7. Set state to COMMITTED
        8. Log TX_COMMIT to WAL
        9. Release lock

        Args:
            tx: Transaction to commit

        Returns:
            CommitResult with success, version, conflicts
        """
        return self._cdg_tx.commit(tx)

    def rollback(self, tx: Transaction, reason: str = "explicit") -> None:
        """
        Rollback transaction.

        Discards write_set, sets state to ROLLED_BACK, logs to WAL.

        Args:
            tx: Transaction to rollback
            reason: Reason for rollback

        Raises:
            TransactionError: If transaction cannot be rolled back
        """
        try:
            self._cdg_tx.rollback(tx, reason)
        except CDGTransactionError as e:
            # Re-raise as GoT TransactionError for API compatibility
            raise GoTTransactionError(str(e)) from e

    def recover(self) -> RecoveryResult:
        """
        Recover from crash.

        Finds incomplete transactions from WAL and rolls them back.
        Uses CDGRecoveryManager for comprehensive recovery.

        Returns:
            RecoveryResult with detailed recovery information
        """
        return self._cdg_tx.recover()

