"""
Recovery module for GoT transactional system.

This is a thin wrapper around CDGRecoveryManager that adds GoT-specific
index recovery functionality. All core recovery (WAL replay, orphan repair,
entity reconstruction) is delegated to CDG.

Architecture:
    GoT Recovery = CDG Recovery + Index Management

    CDG provides:
    - WAL replay and rollback
    - Entity checksum verification
    - Orphan detection and repair (with proper WAL logging)
    - Entity reconstruction from WAL
    - Recovery modes (NONE, CHECKSUM, FULL)

    GoT adds:
    - Index staleness detection (needs_index_recovery)
    - Index rebuilding from entities (rebuild_indexes)

Logging:
    This module uses Python's standard logging. Configure via:

        import logging
        logging.getLogger('cortical.got.recovery').setLevel(logging.DEBUG)

    Log levels:
    - DEBUG: Index comparison details
    - INFO: Index rebuild actions
    - WARNING: Skipped entities during rebuild
    - ERROR: Recovery failures
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

# Module-level logger
logger = logging.getLogger(__name__)

from cortical.cdg.recovery import CDGRecoveryManager
from cortical.cdg.config import CDGConfig
from cortical.common.recovery_types import RecoveryResult, RepairResult
from cortical.got.entity_schemas import get_valid_statuses
from .indexer import QueryIndexManager
from .types import Task, create_entity_from_dict
from .errors import CorruptionError

# Re-export for backward compatibility
__all__ = ['RecoveryManager', 'RecoveryResult', 'RepairResult']


class RecoveryManager:
    """
    Handles crash recovery and index rebuilding for GoT.

    This class delegates core recovery operations to CDGRecoveryManager
    and adds GoT-specific index management.

    Recovery cascade (via CDG):
    1. Check WAL for incomplete transactions -> rollback
    2. Reconstruct entities from committed but unwritten transactions
    3. Detect and repair orphaned entities
    4. Verify entity checksums -> flag corrupted
    5. Verify WAL integrity -> skip corrupted entries

    Additional (GoT-specific):
    6. Check if indexes are stale
    7. Rebuild indexes if needed

    Example:
        >>> manager = RecoveryManager(Path(".got"))
        >>> if manager.needs_recovery():
        ...     result = manager.recover()
        ...     for action in result.actions_taken:
        ...         print(action)
    """

    def __init__(self, got_dir: Path, config: Optional[CDGConfig] = None):
        """
        Initialize recovery manager.

        Args:
            got_dir: Base directory for GoT storage
            config: Optional CDG configuration (defaults to CDGConfig.for_got())
        """
        self.got_dir = Path(got_dir)
        self.got_dir.mkdir(parents=True, exist_ok=True)

        # Use provided config or default GoT config
        self.config = config or CDGConfig.for_got()

        # Set up index rebuild callback for CDG
        self.config.index_rebuild_callback = self._index_rebuild_callback

        # Initialize CDG recovery manager for core recovery operations
        self._cdg_recovery = CDGRecoveryManager(
            store_dir=self.got_dir / "entities",
            config=self.config,
            entity_factory=create_entity_from_dict
        )

    def _index_rebuild_callback(self, store_dir: Path) -> int:
        """
        Callback for CDG to trigger index rebuild.

        Args:
            store_dir: Entity store directory

        Returns:
            Number of tasks indexed
        """
        return self.rebuild_indexes()

    def needs_recovery(self) -> bool:
        """
        Check if recovery is needed.

        Delegates to CDG for core recovery checks.

        Returns:
            True if recovery should be performed
        """
        return self._cdg_recovery.needs_recovery()

    def needs_index_recovery(self) -> bool:
        """
        Check if indexes need to be rebuilt.

        Indexes need recovery if:
        - Index directory EXISTS but files are corrupt or stale
        - Indexes were created but are now incomplete

        Note: If index directory doesn't exist, this returns False.
        Missing indexes don't need "recovery" - they were never created.
        Index initialization is the responsibility of GoTManager, not recovery.

        Returns:
            True if index recovery is needed
        """
        # Check if index directory exists - if not, no recovery needed
        index_dir = self.got_dir / "indexes"
        if not index_dir.exists():
            return False

        # Index directory exists - check if it has any files
        index_files = list(index_dir.glob("*.json"))
        if not index_files:
            return False

        # Get all task IDs from disk
        entity_dir = self.got_dir / "entities"
        entity_files = list(entity_dir.glob("T-*.json"))
        disk_task_ids = set()

        for entity_file in entity_files:
            if entity_file.name.startswith("_") or entity_file.suffix == ".tmp":
                continue
            disk_task_ids.add(entity_file.stem)

        if not disk_task_ids:
            return False

        # Check if indexes are stale by comparing with entities
        index_manager = QueryIndexManager(self.got_dir)

        # Get all task IDs from index
        indexed_task_ids = set()
        for status in get_valid_statuses('task'):
            indexed_task_ids.update(index_manager.lookup("status", status))

        # Check if there are tasks on disk not in the index
        missing_from_index = disk_task_ids - indexed_task_ids
        if missing_from_index:
            logger.debug(
                "Index recovery needed: %d task(s) not indexed: %s",
                len(missing_from_index),
                list(missing_from_index)[:5]
            )
            return True

        return False

    def rebuild_indexes(self) -> int:
        """
        Rebuild all indexes from current entities.

        Returns:
            Number of tasks indexed
        """
        index_manager = QueryIndexManager(self.got_dir)

        # Get all tasks from entity store
        tasks = []
        entity_dir = self.got_dir / "entities"
        entity_files = list(entity_dir.glob("T-*.json"))

        for entity_file in entity_files:
            if entity_file.name.startswith("_") or entity_file.suffix == ".tmp":
                continue

            try:
                data = self._cdg_recovery.store._read_and_verify(entity_file)
                if data.get("entity_type") == "task":
                    task = Task(
                        id=data["id"],
                        title=data.get("title", ""),
                        status=data.get("status", "pending"),
                        priority=data.get("priority", "medium"),
                        description=data.get("description", ""),
                        properties=data.get("properties", {}),
                    )
                    tasks.append(task)
            except (CorruptionError, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logger.warning(
                    "Skipping entity %s during index rebuild: %s: %s",
                    entity_file.name, type(e).__name__, e
                )
                continue

        # Rebuild indexes
        edges = []  # No edges for now - just tasks
        index_manager.rebuild_all(tasks, edges)
        index_manager.save()

        logger.info("Rebuilt indexes: %d tasks indexed", len(tasks))
        return len(tasks)

    def recover(self) -> RecoveryResult:
        """
        Perform full recovery procedure.

        Delegates core recovery to CDG, then handles GoT-specific index recovery.

        Steps (via CDG):
        1. Find incomplete transactions in WAL
        2. Roll back any ACTIVE or PREPARING transactions
        3. Reconstruct entities from committed transactions in WAL
        4. Repair orphaned entities (files without WAL records)
        5. Verify all entity checksums
        6. Report any corrupted entities
        7. Verify WAL integrity

        Additional (GoT-specific):
        8. Rebuild indexes if needed

        Returns:
            RecoveryResult with detailed diagnostics
        """
        # Delegate core recovery to CDG
        # Note: CDG will call our index_rebuild_callback if configured
        result = self._cdg_recovery.recover()

        # If CDG didn't rebuild indexes (e.g., callback wasn't triggered),
        # check if we need to do it ourselves
        if not result.indexes_rebuilt and self.needs_index_recovery():
            task_count = self.rebuild_indexes()
            result.indexes_rebuilt = True
            result.add_action(f"Rebuilt indexes: {task_count} task(s) indexed")

        return result

    # =========================================================================
    # DELEGATED METHODS
    # These are exposed for backward compatibility and testing, but delegate
    # to CDGRecoveryManager for the actual implementation.
    # =========================================================================

    def verify_store_integrity(self):
        """Verify all entities have valid checksums. Delegates to CDG."""
        return self._cdg_recovery.verify_store_integrity()

    def verify_wal_integrity(self):
        """Verify WAL entries have valid checksums. Delegates to CDG."""
        return self._cdg_recovery.verify_wal_integrity()

    def rollback_incomplete_transactions(self):
        """Find and rollback incomplete transactions. Delegates to CDG."""
        return self._cdg_recovery.rollback_incomplete_transactions()

    def detect_orphaned_entities(self):
        """Detect entities without WAL records. Delegates to CDG."""
        return self._cdg_recovery.detect_orphaned_entities()

    def repair_orphans(self, strategy: str = 'adopt') -> RepairResult:
        """
        Repair orphaned entities. Delegates to CDG.

        Args:
            strategy: 'delete' or 'adopt' (default: 'adopt' to preserve git-tracked files)

        Returns:
            RepairResult with repair details
        """
        # Map string strategies to CDG's OrphanStrategy enum
        from cortical.cdg.config import OrphanStrategy

        strategy_map = {
            'delete': OrphanStrategy.DELETE,
            'adopt': OrphanStrategy.REPAIR,
        }

        if strategy not in strategy_map:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'delete' or 'adopt'")

        return self._cdg_recovery.repair_orphans(strategy_map[strategy])

    # =========================================================================
    # PROPERTIES FOR ACCESS TO UNDERLYING CDG COMPONENTS
    # =========================================================================

    @property
    def store(self):
        """Access to underlying CDG store."""
        return self._cdg_recovery.store

    @property
    def wal(self):
        """Access to underlying CDG WAL manager."""
        return self._cdg_recovery.wal
