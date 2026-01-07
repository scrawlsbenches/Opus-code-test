"""
Recovery module for CDG transactional system.

Handles crash recovery and data integrity verification through:
- WAL replay to rollback incomplete transactions
- Entity checksum verification
- WAL entry checksum verification
- Comprehensive recovery reporting

Lifted from GoT's RecoveryManager with CDG adaptations:
- Uses CDGStore instead of VersionedStore
- Uses CDGWALManager instead of WALManager
- Configurable via CDGConfig (recovery_mode, orphan_strategy)
- Optional index rebuilding via callback

Logging:
    This module uses Python's standard logging. Configure via:

        import logging
        logging.getLogger('cortical.cdg.recovery').setLevel(logging.DEBUG)

    Log levels:
    - DEBUG: Race conditions, skipped files, detailed operations
    - INFO: Recovery actions, orphan repairs
    - WARNING: Corrupted entries, integrity issues
    - ERROR: Recovery failures
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

# Module-level logger - configure via logging.getLogger('cortical.cdg.recovery')
logger = logging.getLogger(__name__)

from .storage import CDGStore, EntityFactory
from .wal import CDGWALManager
from .config import CDGConfig, RecoveryMode, OrphanStrategy
from .errors import CorruptionError
from cortical.utils.checksums import compute_checksum
from cortical.common.recovery_types import RecoveryResult, RepairResult

# Re-export for backward compatibility
__all__ = ['CDGRecoveryManager', 'RecoveryResult', 'RepairResult']


class CDGRecoveryManager:
    """
    Handles crash recovery and data integrity verification.

    Recovery cascade:
    1. Check WAL for incomplete transactions → rollback
    2. Detect and repair orphaned entities
    3. Verify entity checksums → flag corrupted
    4. Verify WAL integrity → skip corrupted entries
    5. Rebuild indexes if callback provided

    Example:
        >>> config = CDGConfig.for_got()
        >>> manager = CDGRecoveryManager(Path(".cdg"), config)
        >>> if manager.needs_recovery():
        ...     result = manager.recover()
        ...     for action in result.actions_taken:
        ...         print(action)
    """

    def __init__(
        self,
        store_dir: Path,
        config: CDGConfig,
        entity_factory: Optional[EntityFactory] = None
    ):
        """
        Initialize recovery manager.

        Args:
            store_dir: Base directory for CDG storage
            config: CDG configuration controlling recovery behavior
            entity_factory: Optional factory for creating domain-specific entities
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Initialize storage (without triggering transaction manager recovery)
        self.store = CDGStore(
            self.store_dir,
            config=config,
            entity_factory=entity_factory
        )

        # Initialize WAL if enabled
        self.wal: Optional[CDGWALManager] = None
        if self.config.enable_wal:
            wal_dir = self.store_dir / "wal"
            self.wal = CDGWALManager(wal_dir, config)

    def needs_recovery(self) -> bool:
        """
        Check if recovery is needed.

        Recovery is needed if:
        - There are incomplete transactions in the WAL
        - Entity checksums are invalid
        - WAL entries have corrupted checksums

        Returns:
            True if recovery should be performed
        """
        # Recovery mode NONE never needs recovery
        if self.config.recovery_mode == RecoveryMode.NONE:
            return False

        # Check for incomplete transactions (if WAL enabled)
        if self.wal:
            incomplete = self.wal.get_incomplete_transactions()
            if incomplete:
                return True

        # Check for corrupted entities
        corrupted_entities = self.verify_store_integrity()
        if corrupted_entities:
            return True

        # Check for corrupted WAL entries (if WAL enabled)
        if self.wal:
            corrupted_count = self.verify_wal_integrity()
            if corrupted_count > 0:
                return True

        # Note: Index recovery is NOT part of needs_recovery() check.
        # Indexes are an optional feature, not a core part of the transaction
        # system. Missing indexes don't indicate corruption - they may simply
        # not have been created yet (e.g., when using CDGTransactionManager
        # directly without index support).
        # Index recovery is still performed during recover() if callback provided.

        return False

    def needs_index_recovery(self) -> bool:
        """
        Check if indexes need to be rebuilt.

        This is a placeholder that always returns False since index
        recovery is handled via the optional index_rebuild_callback.

        Returns:
            False (indexes handled externally)
        """
        # Index rebuilding is delegated to the callback if provided
        # We don't have direct knowledge of index structure here
        return False

    def recover(self) -> RecoveryResult:
        """
        Perform full recovery procedure.

        Behavior depends on config.recovery_mode:
        - NONE: Skip recovery entirely
        - CHECKSUM: Verify checksums only
        - FULL: Complete recovery cascade

        Steps (FULL mode):
        1. Find incomplete transactions in WAL
        2. Roll back any ACTIVE or PREPARING transactions
        3. Repair orphaned entities (files without WAL records)
        4. Verify all entity checksums
        5. Report any corrupted entities
        6. Verify WAL integrity
        7. Rebuild indexes (if callback provided)

        Returns:
            RecoveryResult with detailed diagnostics
        """
        result = RecoveryResult(success=True, recovered_transactions=0)

        # Handle recovery modes
        if self.config.recovery_mode == RecoveryMode.NONE:
            result.add_action("Recovery skipped (mode=NONE)")
            return result

        if self.config.recovery_mode == RecoveryMode.CHECKSUM:
            # CHECKSUM mode: Only verify integrity
            corrupted_entities = self.verify_store_integrity()
            result.corrupted_entities = corrupted_entities

            if corrupted_entities:
                result.success = False
                result.add_action(f"Found {len(corrupted_entities)} corrupted entity/entities")
                for entity_id in corrupted_entities:
                    result.add_action(f"  - Entity {entity_id}: checksum mismatch")
            else:
                result.add_action("Store integrity verified (no corruption)")

            return result

        # FULL mode: Complete recovery cascade
        # Step 1-2: Rollback incomplete transactions
        rolled_back = self.rollback_incomplete_transactions()
        result.rolled_back = rolled_back
        result.recovered_transactions = len(rolled_back)

        if rolled_back:
            result.add_action(f"Rolled back {len(rolled_back)} incomplete transaction(s)")
            for tx_id in rolled_back:
                result.add_action(f"  - TX {tx_id}: rolled back due to incomplete state")

        # Step 2.5: Reconstruct entities from WAL for committed transactions
        # This handles crashes that occurred after TX_COMMIT but before entity write
        reconstructed = self.reconstruct_entities_from_wal()
        result.reconstructed_entities = reconstructed

        if reconstructed:
            result.add_action(
                f"Reconstructed {len(reconstructed)} entity/entities from WAL"
            )
            for entity_id in reconstructed:
                result.add_action(f"  - Entity {entity_id}: reconstructed from WAL")

        # Step 3: Repair orphaned entities
        # First detect orphans to populate orphans_detected
        orphans = self.detect_orphaned_entities()
        result.orphans_detected = orphans

        # Repair using configured strategy
        repair_result = self.repair_orphans()
        result.orphans_repaired = repair_result.repaired_count

        if repair_result.repaired_count > 0:
            strategy_name = self.config.orphan_strategy.value
            result.add_action(
                f"Repaired {repair_result.repaired_count} orphaned entity/entities "
                f"(strategy={strategy_name})"
            )
            for entity_id in repair_result.repaired_entities:
                result.add_action(f"  - Entity {entity_id}: {strategy_name}")

        if repair_result.errors:
            result.success = False
            for error in repair_result.errors:
                result.add_action(f"  - Error: {error}")

        # Step 4-5: Verify entity checksums
        corrupted_entities = self.verify_store_integrity()
        result.corrupted_entities = corrupted_entities

        if corrupted_entities:
            result.success = False
            result.add_action(f"Found {len(corrupted_entities)} corrupted entity/entities")
            for entity_id in corrupted_entities:
                result.add_action(f"  - Entity {entity_id}: checksum mismatch")

        # Step 6: Verify WAL integrity
        corrupted_wal_count = self.verify_wal_integrity()
        result.corrupted_wal_entries = corrupted_wal_count

        if corrupted_wal_count > 0:
            result.add_action(f"Found {corrupted_wal_count} corrupted WAL entry/entries")

        # Step 7: Rebuild indexes if callback provided
        if self.config.index_rebuild_callback:
            try:
                task_count = self.config.index_rebuild_callback(self.store_dir)
                result.indexes_rebuilt = True
                result.add_action(f"Rebuilt indexes: {task_count} task(s) indexed")
            except Exception as e:
                result.success = False
                result.add_action(f"Index rebuild failed: {e}")
                logger.error("Index rebuild failed: %s: %s", type(e).__name__, e)

        # Final status
        if not result.actions_taken:
            result.add_action("No recovery needed - system is clean")

        return result

    # Minimum valid entity file size in bytes
    # Even minimal JSON like {"data":{},"_checksum":"..."} is ~50 bytes
    MIN_ENTITY_FILE_SIZE = 20

    def verify_store_integrity(self) -> List[str]:
        """
        Verify all entities have valid checksums and are not truncated.

        Checks:
        1. File size > MIN_ENTITY_FILE_SIZE (catches partial/truncated writes)
        2. Valid JSON structure (catches malformed files)
        3. Checksum matches content (catches corruption)

        Returns:
            List of corrupted entity IDs (empty if all valid)
        """
        corrupted = []

        # Find all entity files
        entity_files = list(self.store.store_dir.glob("*.json"))

        for entity_file in entity_files:
            # Skip temporary and special files
            if entity_file.name.startswith("_") or entity_file.suffix == ".tmp":
                continue

            entity_id = entity_file.stem

            try:
                # Check for suspiciously small files (partial writes)
                file_size = entity_file.stat().st_size
                if file_size < self.MIN_ENTITY_FILE_SIZE:
                    corrupted.append(entity_id)
                    logger.warning(
                        "Partial/truncated entity detected: %s - file size %d bytes < minimum %d",
                        entity_id, file_size, self.MIN_ENTITY_FILE_SIZE
                    )
                    continue

                # Read and verify checksum
                self.store._read_and_verify(entity_file)

            except FileNotFoundError:
                # File was deleted between glob and read (race condition)
                # This is fine - another process may have cleaned it up
                logger.debug(
                    "Entity file %s vanished during integrity check (race condition)",
                    entity_file.name
                )
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                # CorruptionError: checksum mismatch
                # JSONDecodeError: truncated or malformed JSON file
                # KeyError: missing required fields (_checksum, data, etc.)
                corrupted.append(entity_id)
                logger.warning(
                    "Corrupted entity detected: %s - %s: %s",
                    entity_id, type(e).__name__, e
                )

        return corrupted

    def verify_wal_integrity(self) -> int:
        """
        Verify WAL entries have valid checksums.

        Reads all WAL entries and validates their checksums.

        Returns:
            Count of corrupted entries (0 if all valid)
        """
        if not self.wal:
            return 0

        if not self.wal.wal_file.exists():
            return 0

        corrupted_count = 0
        total_entries = 0

        with open(self.wal.wal_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_entries += 1

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    corrupted_count += 1
                    continue

                # Check if checksum field exists
                if 'checksum' not in entry:
                    corrupted_count += 1
                    continue

                # Verify checksum
                expected_checksum = entry['checksum']
                entry_without_checksum = {k: v for k, v in entry.items() if k != 'checksum'}
                actual_checksum = compute_checksum(entry_without_checksum)

                if actual_checksum != expected_checksum:
                    corrupted_count += 1

        return corrupted_count

    def rollback_incomplete_transactions(self) -> List[str]:
        """
        Find and rollback incomplete transactions.

        Identifies transactions in ACTIVE or PREPARING state
        and logs rollback entries to the WAL.

        Returns:
            List of rolled back transaction IDs
        """
        if not self.wal:
            return []

        incomplete = self.wal.get_incomplete_transactions()
        rolled_back = []

        for tx_info in incomplete:
            tx_id = tx_info["tx_id"]

            # Log rollback to WAL
            self.wal.log_tx_rollback(tx_id, "crash_recovery")
            rolled_back.append(tx_id)

        return rolled_back

    def reconstruct_entities_from_wal(self) -> List[str]:
        """
        Reconstruct missing/corrupted entities from WAL for committed transactions.

        This handles the case where a crash occurred after TX_COMMIT was written
        to WAL (transaction is committed) but before entity files were written
        to disk. The entity data stored in WAL WRITE entries is used to
        reconstruct the missing entities.

        Returns:
            List of entity IDs that were reconstructed
        """
        if not self.wal:
            return []

        if not self.wal.wal_file.exists():
            return []

        reconstructed = []

        # Parse WAL to find committed transactions and their WRITE entries
        entries = self.wal.replay()

        # Track transaction states and their write operations
        tx_writes: dict = {}  # tx_id -> [write_entries...]
        committed_txs: set = set()

        for entry in entries:
            tx_id = entry.get('tx')
            op = entry.get('op')
            data = entry.get('data', {})

            if op == 'WRITE' and 'entity_data' in data:
                # Store WRITE entry with entity_data for potential reconstruction
                if tx_id not in tx_writes:
                    tx_writes[tx_id] = []
                tx_writes[tx_id].append(data)

            elif op == 'TX_COMMIT':
                committed_txs.add(tx_id)

            elif op in ('TX_ABORT', 'TX_ROLLBACK'):
                # Transaction was aborted/rolled back, discard its writes
                tx_writes.pop(tx_id, None)

        # For each committed transaction, check if entities need reconstruction
        for tx_id in committed_txs:
            write_entries = tx_writes.get(tx_id, [])

            for write_entry in write_entries:
                entity_id = write_entry.get('entity_id')
                entity_data = write_entry.get('entity_data')
                expected_version = write_entry.get('new_version')

                if not entity_id or not entity_data:
                    continue

                # Skip deletions (new_version == -1)
                if expected_version == -1:
                    continue

                # Check if entity needs reconstruction
                needs_reconstruction = False
                entity_path = self.store.store_dir / f"{entity_id}.json"

                if not entity_path.exists():
                    # Entity file missing
                    needs_reconstruction = True
                    logger.info(
                        "Entity %s missing after committed TX %s - reconstructing from WAL",
                        entity_id, tx_id
                    )
                else:
                    # Entity file exists - verify it's not corrupted
                    try:
                        existing_entity = self.store.read(entity_id)
                        if existing_entity is None:
                            needs_reconstruction = True
                            logger.info(
                                "Entity %s unreadable after committed TX %s - reconstructing from WAL",
                                entity_id, tx_id
                            )
                    except (CorruptionError, json.JSONDecodeError) as e:
                        needs_reconstruction = True
                        logger.info(
                            "Entity %s corrupted after committed TX %s - reconstructing from WAL: %s",
                            entity_id, tx_id, e
                        )

                if needs_reconstruction:
                    try:
                        # Reconstruct entity from WAL data
                        entity = self.store.entity_factory(entity_data)

                        # Write entity to disk using store's write mechanism
                        # Use direct file write to avoid triggering WAL again
                        from datetime import datetime, timezone
                        wrapper = {
                            "_checksum": compute_checksum(entity_data),
                            "_written_at": datetime.now(timezone.utc).isoformat(),
                            "_reconstructed_from_wal": True,
                            "data": entity_data
                        }

                        with open(entity_path, 'w', encoding='utf-8') as f:
                            json.dump(wrapper, f, indent=2, sort_keys=True)
                            f.flush()

                        reconstructed.append(entity_id)
                        logger.info(
                            "Reconstructed entity %s from WAL (TX %s)",
                            entity_id, tx_id
                        )

                    except Exception as e:
                        logger.error(
                            "Failed to reconstruct entity %s from WAL: %s: %s",
                            entity_id, type(e).__name__, e
                        )

        return reconstructed

    def detect_orphaned_entities(self) -> List[str]:
        """
        Detect entities that exist on disk but have no WAL record.

        An orphaned entity is a file that exists in the entity store
        but has no corresponding entry in the WAL. This can happen
        when a crash occurs after writing the entity file but before
        writing the WAL entry, or when dealing with pre-WAL data.

        Returns:
            List of orphaned entity IDs
        """
        # If WAL is disabled, no orphans possible
        if not self.wal:
            return []

        orphaned = []

        # Get all entity IDs from disk
        entity_files = list(self.store.store_dir.glob("*.json"))
        disk_entity_ids = set()

        for entity_file in entity_files:
            # Skip temporary and special files
            if entity_file.name.startswith("_") or entity_file.suffix == ".tmp":
                continue
            disk_entity_ids.add(entity_file.stem)

        # Get all entity IDs from WAL
        wal_entity_ids = set()
        try:
            if self.wal.wal_file.exists():
                with open(self.wal.wal_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = json.loads(line)
                            op = entry.get('op')
                            data = entry.get('data', {})

                            # Look for WRITE operations which track entity modifications
                            if op == 'WRITE':
                                if isinstance(data, dict) and 'entity_id' in data:
                                    wal_entity_ids.add(data['entity_id'])

                            # Check for ADOPTED operations (from recovery)
                            # Supports both legacy format (entity_id at root) and
                            # new TransactionWALEntry format (entity_id in data)
                            elif op == 'ADOPTED':
                                # New format: entity_id in data dict
                                if isinstance(data, dict) and 'entity_id' in data:
                                    wal_entity_ids.add(data['entity_id'])
                                # Legacy format: entity_id at root level
                                elif 'entity_id' in entry:
                                    wal_entity_ids.add(entry['entity_id'])

                        except (json.JSONDecodeError, KeyError) as e:
                            # Skip malformed entries
                            logger.debug(
                                "Skipping malformed WAL entry in orphan detection: %s: %s",
                                type(e).__name__, e
                            )
                            continue
        except FileNotFoundError:
            # WAL file was deleted by another process (race condition)
            logger.debug(
                "WAL file vanished during orphan detection (race condition)"
            )

        # Find entities on disk but not in WAL
        orphaned = list(disk_entity_ids - wal_entity_ids)
        return orphaned

    def repair_orphans(self, strategy: Optional[OrphanStrategy] = None) -> RepairResult:
        """
        Repair orphaned entities found during integrity check.

        An orphaned entity is one that exists on disk but has no WAL record.
        This can happen when a crash occurs between file write and WAL entry,
        or when dealing with pre-WAL data.

        Args:
            strategy: Repair strategy (uses config.orphan_strategy if not specified):
                - FAIL: Raise error and refuse to continue
                - DELETE: Remove orphaned files (safest for clean slate)
                - REPAIR: Add synthetic WAL entries to track orphans (preserve data)

        Returns:
            RepairResult with list of repaired entities and any errors

        Raises:
            ValueError: If strategy is FAIL and orphans exist
        """
        # Use provided strategy or fall back to config
        strategy = strategy or self.config.orphan_strategy

        result = RepairResult(success=True, repaired_count=0)

        # Detect orphaned entities
        orphaned_ids = self.detect_orphaned_entities()

        if not orphaned_ids:
            return result

        # Handle FAIL strategy
        if strategy == OrphanStrategy.FAIL:
            result.success = False
            error_msg = f"Found {len(orphaned_ids)} orphaned entities (strategy=FAIL)"
            result.errors.append(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)

        for entity_id in orphaned_ids:
            entity_file = self.store.store_dir / f"{entity_id}.json"

            # Skip if file no longer exists (race condition with another recovery)
            if not entity_file.exists():
                continue

            try:
                if strategy == OrphanStrategy.DELETE:
                    # Delete the orphaned file
                    try:
                        entity_file.unlink()
                        result.repaired_entities.append(entity_id)
                        result.repaired_count += 1
                        logger.info("Deleted orphaned entity: %s", entity_id)
                    except FileNotFoundError:
                        # Another process already deleted it
                        logger.debug(
                            "Orphan %s already deleted by another process (race condition)",
                            entity_id
                        )

                elif strategy == OrphanStrategy.REPAIR:
                    # Verify the entity is valid before adopting
                    try:
                        self.store._read_and_verify(entity_file)
                    except FileNotFoundError:
                        # File was deleted by another process
                        logger.debug(
                            "Orphan %s vanished before adoption (race condition)",
                            entity_id
                        )
                        continue
                    except (CorruptionError, Exception) as e:
                        # If corrupted, delete it instead of adopting
                        error_msg = f"Entity {entity_id} is corrupted, deleting: {str(e)}"
                        result.errors.append(error_msg)
                        logger.warning(
                            "Cannot adopt corrupted orphan %s, deleting: %s: %s",
                            entity_id, type(e).__name__, e
                        )
                        try:
                            entity_file.unlink()
                        except FileNotFoundError:
                            logger.debug(
                                "Corrupted orphan %s already deleted (race condition)",
                                entity_id
                            )
                        result.repaired_entities.append(entity_id)
                        result.repaired_count += 1
                        continue

                    # Add synthetic WAL entry to adopt the orphan
                    # Use proper WAL logging for durability (includes fsync and sequence)
                    if self.wal:
                        self.wal.log(
                            tx_id="RECOVERY",
                            operation="ADOPTED",
                            data={
                                "entity_id": entity_id,
                                "reason": "orphan_recovery",
                            }
                        )

                        result.repaired_entities.append(entity_id)
                        result.repaired_count += 1
                        logger.info("Adopted orphaned entity: %s", entity_id)

            except Exception as e:
                # Handle any unexpected errors
                error_msg = f"Failed to repair {entity_id}: {str(e)}"
                result.errors.append(error_msg)
                result.success = False
                logger.error(
                    "Unexpected error repairing orphan %s: %s: %s",
                    entity_id, type(e).__name__, e
                )

        return result
