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
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .types import Entity, KnowledgeTransfer, Edge
from .transaction import Transaction
from .config import DurabilityMode
from .versioned_store import _got_entity_factory
from .errors import TransactionError as GoTTransactionError

# Import CDG transaction infrastructure
from cortical.cdg.transaction_manager import CDGTransactionManager
from cortical.cdg.config import CDGConfig, DurabilityMode as CDGDurabilityMode
from cortical.cdg.errors import TransactionError as CDGTransactionError
from cortical.cdg.storage import CDGStore
from cortical.cdg.wal import CDGWALManager

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

    def __init__(
        self,
        got_dir: Path,
        durability: DurabilityMode = DurabilityMode.BALANCED,
        *,
        store: CDGStore,
        wal: Optional[CDGWALManager] = None,
        lock: ProcessLock,
    ):
        """
        Initialize transaction manager.

        BREAKING CHANGE (2026-01-04):
            Dependencies must now be injected. Use create_container() from
            cortical.core.bootstrap to get a properly configured TransactionManager.

        Args:
            got_dir: Base directory for GoT storage
            durability: Durability mode controlling fsync behavior
            store: REQUIRED - Injected CDGStore instance
            wal: Optional CDGWALManager (None for disabled WAL)
            lock: REQUIRED - Injected ProcessLock instance

        Raises:
            TypeError: If required dependencies are missing or wrong type

        Example:
            # The only supported way to get a TransactionManager:
            from cortical.core.bootstrap import create_container

            container = create_container(got_dir=Path(".got"))
            tx_manager = container.resolve(TransactionManager)
        """
        # Validate required dependencies (no defaults - DI is mandatory)
        if not isinstance(store, CDGStore):
            raise TypeError(
                f"store is required and must be CDGStore instance, got {type(store).__name__}"
            )
        # Lock must support context manager protocol (ProcessLock or similar)
        # For in-memory storage, a no-op lock is acceptable
        if lock is not None and not hasattr(lock, '__enter__'):
            raise TypeError(
                f"lock must support context manager protocol, got {type(lock).__name__}"
            )
        if wal is not None and not isinstance(wal, CDGWALManager):
            raise TypeError(
                f"wal must be CDGWALManager instance or None, got {type(wal).__name__}"
            )

        self.got_dir = Path(got_dir)
        self.got_dir.mkdir(parents=True, exist_ok=True)
        self.durability = durability

        # Create CDG configuration for GoT workloads
        cdg_config = CDGConfig.for_got()

        # Override durability mode from GoT config
        cdg_config.durability = _convert_durability(durability)

        # Use injected dependencies (no defaults - container provides all)
        self.store = store
        self.wal = wal
        self.lock = lock

        # Active transactions (in-memory only)
        self._active_tx = {}

        # Create a minimal CDG transaction manager wrapper
        # We create a CDGTransactionManager but override its store, wal, and lock
        # with our configured ones (either injected or default)
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

    def delete(self, tx: Transaction, entity_id: str) -> None:
        """
        Mark an entity for deletion within transaction.

        Logs to WAL and adds to tx.delete_set.
        Does NOT apply to store until commit.

        Args:
            tx: Transaction context
            entity_id: Entity ID to delete

        Raises:
            TransactionError: If transaction is not active
        """
        try:
            self._cdg_tx.delete(tx, entity_id)
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

    # ==================== KnowledgeTransfer Methods ====================

    def create_knowledge_transfer(
        self,
        title: str,
        summary: str = "",
        session_id: str = "",
        session_date: str = "",
        sections: Optional[Dict[str, str]] = None,
        code_refs: Optional[List[str]] = None,
        related_handoffs: Optional[List[str]] = None,
        related_tasks: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        kt_id: Optional[str] = None,
    ) -> KnowledgeTransfer:
        """
        Create a new knowledge transfer entity.

        Args:
            title: Knowledge transfer title
            summary: Executive summary
            session_id: Session identifier
            session_date: Session date (ISO format)
            sections: Dictionary mapping section headings to content
            code_refs: List of code references (file:line format)
            related_handoffs: List of related handoff IDs
            related_tasks: List of related task IDs
            tags: Classification tags
            properties: Additional properties
            kt_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Created KnowledgeTransfer entity

        Raises:
            TransactionError: If transaction fails
        """
        # Generate ID if not provided
        if kt_id is None:
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y%m%d-%H%M%S")
            suffix = secrets.token_hex(4)  # 8 hex chars
            kt_id = f"KT-{timestamp}-{suffix}"

        # Create KnowledgeTransfer instance
        kt = KnowledgeTransfer(
            id=kt_id,
            title=title,
            summary=summary,
            session_id=session_id,
            session_date=session_date,
            sections=sections or {},
            code_refs=code_refs or [],
            related_handoffs=related_handoffs or [],
            related_tasks=related_tasks or [],
            tags=tags or [],
            properties=properties or {},
        )

        # Store via transaction
        tx = self.begin()
        try:
            self.write(tx, kt)
            result = self.commit(tx)
            if not result.success:
                raise GoTTransactionError(f"Failed to create knowledge transfer: {result.reason}")
        except Exception as e:
            self.rollback(tx, reason="create_knowledge_transfer_failed")
            raise GoTTransactionError(f"Failed to create knowledge transfer: {e}") from e

        return kt

    def get_knowledge_transfer(self, kt_id: str) -> Optional[KnowledgeTransfer]:
        """
        Get a knowledge transfer by ID.

        Args:
            kt_id: Knowledge transfer identifier

        Returns:
            KnowledgeTransfer entity or None if not found
        """
        tx = self.begin()
        try:
            entity = self.read(tx, kt_id)
            self.rollback(tx, reason="read_only")

            if entity is None:
                return None
            if not isinstance(entity, KnowledgeTransfer):
                return None
            return entity
        except Exception:
            self.rollback(tx, reason="get_knowledge_transfer_failed")
            return None

    def list_knowledge_transfers(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KnowledgeTransfer]:
        """
        List knowledge transfers with optional filtering.

        Args:
            status: Filter by status (draft, published, archived)
            tags: Filter by tags (must have all specified tags)

        Returns:
            List of matching KnowledgeTransfer entities
        """
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        transfers = []
        for entity_file in entities_dir.glob("KT-*.json"):
            try:
                tx = self.begin()
                entity = self.read(tx, entity_file.stem)
                self.rollback(tx, reason="read_only")

                if entity is None or not isinstance(entity, KnowledgeTransfer):
                    continue

                # Apply filters
                if status is not None and entity.status != status:
                    continue
                if tags is not None:
                    if not all(tag in entity.tags for tag in tags):
                        continue

                transfers.append(entity)
            except Exception as e:
                logger.warning(f"Skipping corrupted KT file {entity_file}: {e}")
                continue

        return transfers

    def append_to_knowledge_transfer(
        self,
        kt_id: str,
        section_name: str,
        content: str,
    ) -> KnowledgeTransfer:
        """
        Append content to a section of a knowledge transfer.

        If the section doesn't exist, it will be created.
        If it exists, content will be appended with double newline separator.

        Args:
            kt_id: Knowledge transfer identifier
            section_name: Section heading/name
            content: Content to append

        Returns:
            Updated KnowledgeTransfer entity

        Raises:
            TransactionError: If knowledge transfer not found or transaction fails
        """
        tx = self.begin()
        try:
            # Read existing entity
            entity = self.read(tx, kt_id)
            if entity is None or not isinstance(entity, KnowledgeTransfer):
                self.rollback(tx, reason="kt_not_found")
                raise GoTTransactionError(f"Knowledge transfer not found: {kt_id}")

            # Append to section
            if section_name in entity.sections:
                entity.sections[section_name] += f"\n\n{content}"
            else:
                entity.sections[section_name] = content

            # Update entity
            entity.bump_version()
            self.write(tx, entity)

            result = self.commit(tx)
            if not result.success:
                raise GoTTransactionError(f"Failed to update knowledge transfer: {result.reason}")

            return entity
        except GoTTransactionError:
            raise
        except Exception as e:
            self.rollback(tx, reason="append_failed")
            raise GoTTransactionError(f"Failed to append to knowledge transfer: {e}") from e

    def link_knowledge_transfer(
        self,
        kt_id: str,
        target_id: str,
        link_type: str = "DOCUMENTS",
    ) -> bool:
        """
        Link a knowledge transfer to another entity.

        Creates an edge from the knowledge transfer to the target entity.

        Args:
            kt_id: Knowledge transfer identifier
            target_id: Target entity identifier
            link_type: Edge type (DOCUMENTS, CONTINUES, REFERENCES, etc.)

        Returns:
            True if link created successfully

        Raises:
            TransactionError: If transaction fails
        """
        tx = self.begin()
        try:
            # Verify KT exists
            kt_entity = self.read(tx, kt_id)
            if kt_entity is None or not isinstance(kt_entity, KnowledgeTransfer):
                self.rollback(tx, reason="kt_not_found")
                raise GoTTransactionError(f"Knowledge transfer not found: {kt_id}")

            # Verify target exists
            target_entity = self.read(tx, target_id)
            if target_entity is None:
                self.rollback(tx, reason="target_not_found")
                raise GoTTransactionError(f"Target entity not found: {target_id}")

            # Create edge
            edge = Edge(
                id="",  # Auto-generated in __post_init__
                source_id=kt_id,
                target_id=target_id,
                edge_type=link_type,
                weight=1.0,
                confidence=1.0,
            )

            self.write(tx, edge)
            result = self.commit(tx)

            if not result.success:
                raise GoTTransactionError(f"Failed to create link: {result.reason}")

            return True
        except GoTTransactionError:
            raise
        except Exception as e:
            self.rollback(tx, reason="link_failed")
            raise GoTTransactionError(f"Failed to link knowledge transfer: {e}") from e

