"""
TransactionManager for GoT with ACID guarantees.

Orchestrates begin/commit/rollback with:
- Atomicity: All writes succeed or all fail
- Consistency: Checksums verify data integrity
- Isolation: Snapshot isolation via versioning
- Durability: WAL + fsync before commit
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .types import Entity
from .errors import TransactionError, ConflictError
from .versioned_store import VersionedStore
from .wal import WALManager
from .transaction import Transaction, TransactionState, generate_transaction_id
from .config import DurabilityMode

logger = logging.getLogger(__name__)


# Platform detection for file locking
if sys.platform != 'win32':
    import fcntl


class ProcessLock:
    """
    Simple file-based lock for process safety.

    Uses fcntl.flock() on POSIX systems, graceful no-op on Windows.
    Provides context manager interface for safe lock management.
    """

    def __init__(self, lock_path: Path, reentrant: bool = True):
        """
        Initialize process lock.

        Args:
            lock_path: Path to lock file
            reentrant: If True, same process can acquire multiple times
        """
        self.lock_path = Path(lock_path)
        self.reentrant = reentrant
        self._fd = None
        self._lock_count = 0
        self._thread_lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock with timeout and stale lock recovery.

        Args:
            timeout: Timeout in seconds. If None, single non-blocking attempt.
                    If provided, retry with exponential backoff until timeout.

        Returns:
            True if lock acquired, False otherwise
        """
        with self._thread_lock:
            # Reentrant check
            if self.reentrant and self._lock_count > 0:
                self._lock_count += 1
                return True

            # Create lock file if needed
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)

            # Timeout=None: single attempt (backward compatible)
            if timeout is None:
                return self._try_acquire_once()

            # Timeout specified: retry with exponential backoff
            start_time = time.time()
            backoff_ms = 10  # Start with 10ms
            max_backoff_ms = 500  # Cap at 500ms

            while True:
                # Try to acquire
                if self._try_acquire_once():
                    return True

                # Check if timeout exceeded
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

                # Exponential backoff, capped at max_backoff
                sleep_time = min(backoff_ms / 1000.0, timeout - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Double backoff for next iteration, cap at max
                backoff_ms = min(backoff_ms * 2, max_backoff_ms)

                # Check again if we've exceeded timeout after sleep
                if time.time() - start_time >= timeout:
                    return False

    def _try_acquire_once(self) -> bool:
        """
        Try to acquire lock once (non-blocking).

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            # Open file for locking
            self._fd = open(self.lock_path, 'r+' if self.lock_path.exists() else 'w+')

            # Platform-specific locking
            if sys.platform != 'win32':
                try:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (IOError, OSError):
                    # Lock held by another process - check if stale
                    self._fd.close()
                    self._fd = None

                    if self._is_stale_lock():
                        logger.warning(
                            f"Detected stale lock at {self.lock_path}, recovering..."
                        )
                        # Remove stale lock file and retry
                        try:
                            self.lock_path.unlink(missing_ok=True)
                        except Exception as e:
                            logger.error(f"Failed to remove stale lock: {e}")
                            return False

                        # Retry acquisition after removing stale lock
                        try:
                            self._fd = open(self.lock_path, 'w+')
                            if sys.platform != 'win32':
                                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except (IOError, OSError):
                            if self._fd:
                                self._fd.close()
                                self._fd = None
                            return False
                    else:
                        return False

            # Successfully acquired - write holder info
            try:
                holder_info = {
                    "pid": os.getpid(),
                    "acquired_at": time.time()
                }
                self._fd.seek(0)
                self._fd.truncate()
                json.dump(holder_info, self._fd)
                self._fd.flush()
                os.fsync(self._fd.fileno())
            except Exception as e:
                logger.error(f"Failed to write lock holder info: {e}")
                # Don't fail the lock acquisition if we can't write metadata

            self._lock_count = 1
            return True

        except Exception as e:
            # Catch all other exceptions and return False
            logger.error(f"Unexpected error in lock acquisition: {e}")
            if self._fd:
                try:
                    self._fd.close()
                except Exception:
                    pass
                self._fd = None
            return False

    def _is_stale_lock(self) -> bool:
        """
        Check if lock file is stale (held by dead process).

        Returns:
            True if lock is stale and can be stolen, False otherwise
        """
        try:
            # Read lock file to get holder PID
            if not self.lock_path.exists():
                return False

            with open(self.lock_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    # Empty lock file - consider stale
                    return True

                holder_info = json.loads(content)
                holder_pid = holder_info.get("pid")

                if holder_pid is None:
                    # No PID in lock file - consider stale
                    return True

                # Check if process is still alive
                try:
                    # os.kill(pid, 0) doesn't send signal, just checks if process exists
                    os.kill(holder_pid, 0)
                    # Process exists - lock is not stale
                    return False
                except OSError:
                    # Process doesn't exist - lock is stale
                    return True

        except json.JSONDecodeError:
            # Invalid JSON - consider stale
            logger.warning(f"Lock file {self.lock_path} has invalid JSON")
            return True
        except Exception as e:
            # Any other error - don't assume stale
            logger.error(f"Error checking stale lock: {e}")
            return False

    def release(self) -> None:
        """Release the lock."""
        with self._thread_lock:
            if self._lock_count == 0:
                return

            if self.reentrant and self._lock_count > 1:
                self._lock_count -= 1
                return

            if self._fd:
                if sys.platform != 'win32':
                    try:
                        fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                    except (IOError, OSError):
                        pass
                self._fd.close()
                self._fd = None

            self._lock_count = 0

    def __enter__(self) -> ProcessLock:
        """Context manager entry."""
        if not self.acquire():
            raise TransactionError("Failed to acquire lock")
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.release()


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




class TransactionManager:
    """
    Manages transactions with ACID guarantees.

    - Atomicity: All writes in a TX succeed or all fail
    - Consistency: Checksums verify data integrity
    - Isolation: Snapshot isolation via versioning
    - Durability: WAL + fsync before commit
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

        # Initialize storage and WAL with durability mode
        self.store = VersionedStore(self.got_dir / "entities", durability=durability)
        self.wal = WALManager(self.got_dir / "wal", durability=durability)

        # Process lock for mutual exclusion
        self.lock = ProcessLock(self.got_dir / ".got.lock", reentrant=True)

        # Active transactions (in-memory only)
        self._active_tx: Dict[str, Transaction] = {}

        # Run recovery on startup
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
            started_at="",  # Will be set by transaction
            snapshot_version=snapshot_version,
            write_set={},
            read_set={}
        )

        # Log to WAL (survives crash)
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

        Logs to WAL and adds to tx.write_set.
        Does NOT apply to store until commit.

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

        # Log to WAL before buffering
        self.wal.log_write(tx.id, entity.id, old_version, entity.version)

        # Add to write set
        tx.add_write(entity)

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
        if not tx.can_commit():
            return CommitResult(
                success=False,
                reason=f"Transaction {tx.id} cannot commit (state: {tx.state.value})"
            )

        with self.lock:
            # Set state to PREPARING
            tx.state = TransactionState.PREPARING
            self.wal.log_tx_prepare(tx.id)

            # Detect conflicts
            conflicts = self._detect_conflicts(tx)
            if conflicts:
                # Abort transaction
                tx.state = TransactionState.ABORTED
                self.wal.log_tx_abort(tx.id, "version_conflict")
                self._active_tx.pop(tx.id, None)

                return CommitResult(
                    success=False,
                    conflicts=conflicts,
                    reason="version_conflict"
                )

            # Apply writes atomically
            try:
                new_version = self.store.apply_writes(tx.write_set)
            except Exception as e:
                # Abort on any error
                tx.state = TransactionState.ABORTED
                self.wal.log_tx_abort(tx.id, f"write_failed: {e}")
                self._active_tx.pop(tx.id, None)

                return CommitResult(
                    success=False,
                    reason=f"write_failed: {e}"
                )

            # Mark committed
            tx.state = TransactionState.COMMITTED
            self.wal.log_tx_commit(tx.id, new_version)

            # For BALANCED mode, fsync WAL and store now
            if self.durability == DurabilityMode.BALANCED:
                self.wal.fsync_now()
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
        self.wal.log_tx_rollback(tx.id, reason)

        # Remove from active
        self._active_tx.pop(tx.id, None)

    def recover(self):
        """
        Recover from crash.

        Finds incomplete transactions from WAL and rolls them back.
        Uses RecoveryManager for comprehensive recovery.

        Returns:
            RecoveryResult with detailed recovery information
        """
        from .recovery import RecoveryManager

        recovery_mgr = RecoveryManager(self.got_dir)
        return recovery_mgr.recover()

    def _detect_conflicts(self, tx: Transaction) -> List[Conflict]:
        """
        Detect version conflicts between transaction and current store.

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
