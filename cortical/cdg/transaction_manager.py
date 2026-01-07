"""
CDGTransactionManager for distributed graph with ACID guarantees.

Orchestrates begin/commit/rollback with:
- Atomicity: All writes succeed or all fail
- Consistency: Checksums verify data integrity
- Isolation: Snapshot isolation via versioning
- Durability: WAL + fsync before commit

Lifted from GoT's TransactionManager with CDG extensions for:
- CDGConfig-based configuration
- Optional WAL (controlled by config.enable_wal)
- Recovery mode configuration (NONE, CHECKSUM, FULL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from cortical.utils.locking import ProcessLock
from cortical.common.filesystem import FileSystem
from .types import Entity
from .errors import TransactionError, ConflictError
from .storage import CDGStore, EntityFactory
from .wal import CDGWALManager
from .transaction import Transaction, TransactionState, generate_transaction_id
from .config import CDGConfig, RecoveryMode, DurabilityMode

logger = logging.getLogger(__name__)


@dataclass
class Conflict:
    """Represents a version conflict during commit."""

    entity_id: str
    expected_version: int
    actual_version: int
    conflict_type: str  # "version_mismatch", "create_exists"
    message: str


@dataclass
class CommitResult:
    """Result of transaction commit operation."""

    success: bool
    version: Optional[int] = None  # New version if success
    conflicts: List[Conflict] = field(default_factory=list)  # Conflicts if failure
    reason: Optional[str] = None  # Failure reason


class CDGTransactionManager:
    """
    Manages transactions with ACID guarantees for CDG.

    - Atomicity: All writes in a TX succeed or all fail
    - Consistency: Checksums verify data integrity
    - Isolation: Snapshot isolation via versioning
    - Durability: WAL + fsync before commit (when enabled)

    Lifted from GoT's TransactionManager with CDG extensions:
    - Pluggable entity factory for domain-specific types
    - Optional WAL (controlled by config.enable_wal)
    - Configurable recovery modes (NONE, CHECKSUM, FULL)

    Example:
        config = CDGConfig.for_got()
        manager = CDGTransactionManager(Path("./data"), config)

        # Begin transaction
        tx = manager.begin()

        # Read and write
        entity = manager.read(tx, "E-001")
        entity.properties["updated"] = True
        manager.write(tx, entity)

        # Commit
        result = manager.commit(tx)
        if result.success:
            print(f"Committed at version {result.version}")
        else:
            print(f"Conflict: {result.conflicts}")
    """

    def __init__(
        self,
        store_dir: Path,
        config: Optional[CDGConfig] = None,
        entity_factory: Optional[EntityFactory] = None,
        filesystem: Optional[FileSystem] = None,
    ):
        """
        Initialize transaction manager.

        Creates directories if needed:
        - {store_dir}/
        - {store_dir}/wal/ (if WAL enabled)

        Runs recovery on startup if config.auto_recover_on_startup is True.

        Args:
            store_dir: Base directory for CDG storage
            config: CDG configuration (uses defaults if not provided)
            entity_factory: Optional factory for creating domain-specific entities
            filesystem: FileSystem implementation (defaults to RealFileSystem).
                       Pass InMemoryFileSystem for test isolation.
        """
        self.store_dir = Path(store_dir)

        # Use provided config or create default
        self.config = config or CDGConfig()

        # Initialize storage with filesystem abstraction
        self.store = CDGStore(
            self.store_dir,
            config=self.config,
            entity_factory=entity_factory,
            filesystem=filesystem,
        )

        # Initialize WAL if enabled
        self.wal: Optional[CDGWALManager] = None
        if self.config.enable_wal:
            wal_dir = self.store_dir / "wal"
            self.wal = CDGWALManager(wal_dir, self.config)

        # Process lock for mutual exclusion
        self.lock = ProcessLock(self.store_dir / ".cdg.lock", reentrant=True)

        # Active transactions (in-memory only)
        self._active_tx: Dict[str, Transaction] = {}

        # Run recovery on startup if configured
        if self.config.auto_recover_on_startup:
            self.recover()

    def begin(self) -> Transaction:
        """
        Start a new transaction.

        Returns:
            New Transaction object in ACTIVE state
        """
        tx_id = generate_transaction_id()
        snapshot_version = self.store.current_version()

        tx = Transaction(
            id=tx_id,
            state=TransactionState.ACTIVE,
            started_at="",  # Will be set by Transaction.begin()
            snapshot_version=snapshot_version,
            write_set={},
            read_set={}
        )

        # Log to WAL if enabled (survives crash)
        if self.wal:
            self.wal.log_tx_begin(tx_id, snapshot_version)

        # Track in-memory
        self._active_tx[tx_id] = tx

        return tx

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
            Entity instance or None if not found
        """
        # Check write set first (read own writes)
        if entity_id in tx.write_set:
            return tx.write_set[entity_id]

        # Read from snapshot version
        entity = self.store.read_at_version(entity_id, tx.snapshot_version)

        # Track read for conflict detection
        if entity:
            tx.add_read(entity_id, entity.version)

        return entity

    def write(self, tx: Transaction, entity: Entity) -> None:
        """
        Buffer a write within transaction.

        Logs to WAL (if enabled) and adds to tx.write_set.
        Does NOT apply to store until commit.

        The entity's full state is logged to WAL to enable crash recovery
        reconstruction if a crash occurs after TX_COMMIT but before entity
        files are written to disk.

        Args:
            tx: Transaction context
            entity: Entity to write

        Raises:
            TransactionError: If transaction is not active
        """
        if not tx.is_active():
            raise TransactionError(
                f"Transaction {tx.id} is not active (state: {tx.state.value})"
            )

        # Get old version for WAL
        old_entity = self.read(tx, entity.id)
        old_version = old_entity.version if old_entity else 0

        # Log to WAL before buffering (if enabled)
        # Include full entity data for crash recovery reconstruction
        if self.wal:
            self.wal.log_write(
                tx.id,
                entity.id,
                old_version,
                entity.version,
                entity_data=entity.to_dict()
            )

        # Add to write set
        tx.add_write(entity)

    def delete(self, tx: Transaction, entity_id: str) -> None:
        """
        Mark an entity for deletion within transaction.

        Logs to WAL (if enabled) and adds to tx.delete_set.
        Does NOT apply to store until commit.

        Args:
            tx: Transaction context
            entity_id: Entity ID to delete

        Raises:
            TransactionError: If transaction is not active
        """
        if not tx.is_active():
            raise TransactionError(
                f"Transaction {tx.id} is not active (state: {tx.state.value})"
            )

        # Get entity version for conflict detection
        entity = self.read(tx, entity_id)
        if entity:
            # Track read for conflict detection
            tx.add_read(entity_id, entity.version)

        # Log to WAL before buffering (if enabled)
        if self.wal:
            version = entity.version if entity else 0
            self.wal.log_write(tx.id, entity_id, version, -1)  # -1 indicates deletion

        # Add to delete set
        tx.add_delete(entity_id)

    def commit(self, tx: Transaction) -> CommitResult:
        """
        Commit transaction with WAL-first durability.

        WAL-First Protocol (ACID-compliant):
        1. Acquire lock
        2. Set state to PREPARING, log TX_PREPARE
        3. Detect conflicts (version mismatch)
        4. If conflict: abort, return failure
        5. Log TX_COMMIT to WAL (commit decision is now durable)
        6. Fsync WAL (ensures commit survives crash)
        7. Apply writes to entity files (can be redone from WAL on crash)
        8. Set state to COMMITTED
        9. Release lock

        Key insight: Once TX_COMMIT is in WAL and fsynced, the transaction
        IS committed. Entity files are a materialized view that can be
        reconstructed from WAL on recovery.

        Args:
            tx: Transaction to commit

        Returns:
            CommitResult with success, version, conflicts
        """
        if not tx.can_commit():
            return CommitResult(
                success=False,
                reason=f"Transaction {tx.id} cannot commit (state: {tx.state.value})"
            )

        with self.lock:
            # Step 1: Set state to PREPARING
            tx.state = TransactionState.PREPARING
            if self.wal:
                self.wal.log_tx_prepare(tx.id)

            # Step 2: Detect conflicts before committing
            conflicts = self._detect_conflicts(tx)
            if conflicts:
                # Abort transaction
                tx.state = TransactionState.ABORTED
                if self.wal:
                    self.wal.log_tx_abort(tx.id, "version_conflict")
                self._active_tx.pop(tx.id, None)

                return CommitResult(
                    success=False,
                    conflicts=conflicts,
                    reason="version_conflict"
                )

            # Calculate the version that will result from this commit
            # (store.current_version() + 1 for writes, possibly +1 more for deletes)
            expected_version = self.store.current_version() + 1
            if tx.delete_set:
                expected_version += 1

            # Step 3: Log TX_COMMIT to WAL BEFORE applying writes
            # This is the commit point - once this is durable, the tx IS committed
            if self.wal:
                self.wal.log_tx_commit(tx.id, expected_version)

            # Step 4: Fsync WAL to ensure commit is durable BEFORE modifying entities
            # This is critical: WAL must be durable before we change entity files
            # Skip for RELAXED mode (no durability guarantees)
            if self.wal and self.config.durability != DurabilityMode.RELAXED:
                self.wal.fsync_now()

            # Step 5: Apply writes and deletes to entity files
            # If crash after this point, recovery will see TX_COMMIT in WAL
            # and can verify/redo the writes
            try:
                new_version = self.store.apply_writes(tx.write_set)
                # Apply deletes after writes (both within same lock)
                if tx.delete_set:
                    new_version = self.store.apply_deletes(tx.delete_set)
            except Exception as e:
                # Write failed AFTER commit was logged to WAL
                # This is a serious error - WAL says committed but writes failed
                # Log the failure but don't abort (WAL is source of truth)
                # Recovery will need to redo these writes
                logger.error(
                    "Write failed after WAL commit for TX %s: %s. "
                    "Recovery will need to redo writes from WAL.",
                    tx.id, e
                )
                # Still mark as committed since WAL has the commit
                tx.state = TransactionState.COMMITTED
                self._active_tx.pop(tx.id, None)
                # Return failure so caller knows writes didn't apply
                # But the transaction IS committed in WAL
                return CommitResult(
                    success=False,
                    reason=f"commit_logged_but_write_failed: {e}",
                    version=expected_version
                )

            # Step 6: Mark committed in memory
            tx.state = TransactionState.COMMITTED

            # Step 7: Optionally fsync entity files for extra durability
            if self.config.durability == DurabilityMode.BALANCED:
                self.store.fsync_all()

            # Remove from active transactions
            self._active_tx.pop(tx.id, None)

            return CommitResult(success=True, version=new_version)

    def rollback(self, tx: Transaction, reason: str = "explicit") -> None:
        """
        Rollback transaction.

        Discards write_set, sets state to ROLLED_BACK, logs to WAL.

        Args:
            tx: Transaction to rollback
            reason: Reason for rollback
        """
        if not tx.can_rollback():
            raise TransactionError(
                f"Transaction {tx.id} cannot rollback (state: {tx.state.value})"
            )

        # Discard writes
        tx.write_set.clear()

        # Update state
        tx.state = TransactionState.ROLLED_BACK
        if self.wal:
            self.wal.log_tx_rollback(tx.id, reason)

        # Remove from active
        self._active_tx.pop(tx.id, None)

    def recover(self):
        """
        Recover from crash using CDGRecoveryManager.

        Behavior depends on config.recovery_mode:
        - NONE: Skip recovery, return immediately
        - CHECKSUM: Verify entity checksums only
        - FULL: WAL replay + checksum verification + orphan repair

        Returns:
            RecoveryResult with recovery information

        Note:
            Import RecoveryResult from cortical.cdg.recovery to use result
        """
        # Import here to avoid circular dependency
        from .recovery import CDGRecoveryManager

        # Create recovery manager (shares config and entity factory)
        recovery_manager = CDGRecoveryManager(
            self.store_dir,
            self.config,
            entity_factory=self.store.entity_factory
        )

        # Delegate to recovery manager
        return recovery_manager.recover()

    def has_active_transactions(self) -> bool:
        """
        Check if there are any active transactions.

        This is used by sync operations to ensure that sync only happens
        when there are no in-flight transactions.

        Returns:
            True if there are active transactions, False otherwise
        """
        return len(self._active_tx) > 0

    def get_active_transaction_ids(self) -> List[str]:
        """
        Get list of active transaction IDs.

        Returns:
            List of transaction IDs currently active
        """
        return list(self._active_tx.keys())

    def _detect_conflicts(self, tx: Transaction) -> List[Conflict]:
        """
        Detect version conflicts between transaction and current store.

        Uses optimistic locking: for each entity in write_set that was
        also in read_set, verify that the version hasn't changed since
        the read.

        Args:
            tx: Transaction to check

        Returns:
            List of conflicts (empty if none)
        """
        conflicts = []

        for entity_id in tx.write_set:
            # Check if entity was read (optimistic locking)
            if entity_id in tx.read_set:
                expected_version = tx.read_set[entity_id]

                # Get current version from store
                current_entity = self.store.read(entity_id)
                actual_version = current_entity.version if current_entity else 0

                if expected_version != actual_version:
                    conflicts.append(Conflict(
                        entity_id=entity_id,
                        expected_version=expected_version,
                        actual_version=actual_version,
                        conflict_type="version_mismatch",
                        message=f"Expected version {expected_version}, got {actual_version}"
                    ))

        return conflicts
