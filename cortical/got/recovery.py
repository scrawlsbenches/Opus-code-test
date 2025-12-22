"""
Recovery module for GoT transactional system.

Handles crash recovery and data integrity verification through:
- WAL replay to rollback incomplete transactions
- Entity checksum verification
- WAL entry checksum verification
- Comprehensive recovery reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .tx_manager import TransactionManager
from .versioned_store import VersionedStore
from .wal import WALManager
from cortical.utils.checksums import compute_checksum, verify_checksum
from .errors import CorruptionError


@dataclass
class RecoveryResult:
    """
    Result of crash recovery operation with detailed diagnostics.

    Attributes:
        success: True if recovery completed without corruption
        recovered_transactions: Number of incomplete transactions recovered
        rolled_back: List of transaction IDs that were rolled back
        corrupted_entities: List of entity IDs with checksum mismatches
        corrupted_wal_entries: Count of WAL entries with invalid checksums
        actions_taken: Human-readable log of recovery actions
    """

    success: bool
    recovered_transactions: int
    rolled_back: List[str] = field(default_factory=list)
    corrupted_entities: List[str] = field(default_factory=list)
    corrupted_wal_entries: int = 0
    actions_taken: List[str] = field(default_factory=list)

    def add_action(self, action: str) -> None:
        """
        Log a recovery action.

        Args:
            action: Human-readable description of action taken
        """
        self.actions_taken.append(action)


@dataclass
class RepairResult:
    """
    Result of orphan entity repair operation.

    Attributes:
        success: True if repair completed without errors
        repaired_count: Number of orphaned entities repaired
        repaired_entities: List of entity IDs that were repaired
        errors: List of error messages encountered during repair
    """

    success: bool
    repaired_count: int
    repaired_entities: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RecoveryManager:
    """
    Handles crash recovery and data integrity verification.

    Recovery cascade:
    1. Check WAL for incomplete transactions → rollback
    2. Verify entity checksums → flag corrupted
    3. Verify WAL integrity → skip corrupted entries
    4. (Future) Recover from git history if needed

    Example:
        >>> manager = RecoveryManager(Path(".got"))
        >>> if manager.needs_recovery():
        ...     result = manager.recover()
        ...     for action in result.actions_taken:
        ...         print(action)
    """

    def __init__(self, got_dir: Path):
        """
        Initialize recovery manager.

        Args:
            got_dir: Base directory for GoT storage
        """
        self.got_dir = Path(got_dir)
        self.got_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage and WAL (without triggering TX manager recovery)
        self.store = VersionedStore(self.got_dir / "entities")
        self.wal = WALManager(self.got_dir / "wal")

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
        # Check for incomplete transactions
        incomplete = self.wal.get_incomplete_transactions()
        if incomplete:
            return True

        # Check for corrupted entities
        corrupted_entities = self.verify_store_integrity()
        if corrupted_entities:
            return True

        # Check for corrupted WAL entries
        corrupted_count = self.verify_wal_integrity()
        if corrupted_count > 0:
            return True

        return False

    def recover(self) -> RecoveryResult:
        """
        Perform full recovery procedure.

        Steps:
        1. Find incomplete transactions in WAL
        2. Roll back any ACTIVE or PREPARING transactions
        3. Repair orphaned entities (files without WAL records)
        4. Verify all entity checksums
        5. Report any corrupted entities
        6. Verify WAL integrity

        Returns:
            RecoveryResult with detailed diagnostics
        """
        result = RecoveryResult(success=True, recovered_transactions=0)

        # Step 1-2: Rollback incomplete transactions
        rolled_back = self.rollback_incomplete_transactions()
        result.rolled_back = rolled_back
        result.recovered_transactions = len(rolled_back)

        if rolled_back:
            result.add_action(f"Rolled back {len(rolled_back)} incomplete transaction(s)")
            for tx_id in rolled_back:
                result.add_action(f"  - TX {tx_id}: rolled back due to incomplete state")

        # Step 3: Repair orphaned entities
        # Use 'adopt' strategy to preserve git-tracked files that lack WAL entries
        # (files committed to git don't have WAL records, so 'delete' would wipe them)
        repair_result = self.repair_orphans(strategy='adopt')
        if repair_result.repaired_count > 0:
            result.add_action(f"Adopted {repair_result.repaired_count} orphaned entity/entities")
            for entity_id in repair_result.repaired_entities:
                result.add_action(f"  - Entity {entity_id}: adopted into WAL")

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

        # Final status
        if not result.actions_taken:
            result.add_action("No recovery needed - system is clean")

        return result

    def verify_store_integrity(self) -> List[str]:
        """
        Verify all entities have valid checksums.

        Reads all entity files and validates their embedded checksums.

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

            try:
                # Read and verify checksum
                self.store._read_and_verify(entity_file)
            except CorruptionError:
                # Extract entity ID from filename
                entity_id = entity_file.stem
                corrupted.append(entity_id)

        return corrupted

    def verify_wal_integrity(self) -> int:
        """
        Verify WAL entries have valid checksums.

        Reads all WAL entries and validates their checksums.

        Returns:
            Count of corrupted entries (0 if all valid)
        """
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
                    import json
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
        incomplete = self.wal.get_incomplete_transactions()
        rolled_back = []

        for tx_info in incomplete:
            tx_id = tx_info["tx_id"]

            # Log rollback to WAL
            self.wal.log_tx_rollback(tx_id, "crash_recovery")
            rolled_back.append(tx_id)

        return rolled_back

    def detect_orphaned_entities(self) -> List[str]:
        """
        Detect entities that exist on disk but have no WAL record.

        An orphaned entity is a file that exists in the entity store
        but has no corresponding entry in the WAL. This can happen
        when a crash occurs after writing the entity file but before
        writing the WAL entry.

        Returns:
            List of orphaned entity IDs
        """
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
        if self.wal.wal_file.exists():
            import json
            with open(self.wal.wal_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        # Look for WRITE operations which track entity modifications
                        if entry.get('op') == 'WRITE':
                            if 'data' in entry and isinstance(entry['data'], dict):
                                if 'entity_id' in entry['data']:
                                    wal_entity_ids.add(entry['data']['entity_id'])
                        # Also check for ADOPTED operations (from recovery)
                        elif entry.get('op') == 'ADOPTED':
                            if 'entity_id' in entry:
                                wal_entity_ids.add(entry['entity_id'])
                    except (json.JSONDecodeError, KeyError):
                        # Skip malformed entries
                        continue

        # Find entities on disk but not in WAL
        orphaned = list(disk_entity_ids - wal_entity_ids)
        return orphaned

    def repair_orphans(self, strategy: str = 'delete') -> RepairResult:
        """
        Repair orphaned entities found during integrity check.

        An orphaned entity is one that exists on disk but has no WAL record.
        This can happen when a crash occurs between file write and WAL entry.

        Args:
            strategy: Repair strategy:
                - 'delete': Remove orphaned files (safest, default)
                - 'adopt': Add synthetic WAL entries to track orphans

        Returns:
            RepairResult with list of repaired entities and any errors

        Raises:
            ValueError: If strategy is not 'delete' or 'adopt'
        """
        if strategy not in ('delete', 'adopt'):
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'delete' or 'adopt'")

        result = RepairResult(success=True, repaired_count=0)

        # Detect orphaned entities
        orphaned_ids = self.detect_orphaned_entities()

        if not orphaned_ids:
            return result

        import json

        for entity_id in orphaned_ids:
            entity_file = self.store.store_dir / f"{entity_id}.json"

            try:
                if strategy == 'delete':
                    # Delete the orphaned file
                    entity_file.unlink()
                    result.repaired_entities.append(entity_id)
                    result.repaired_count += 1

                elif strategy == 'adopt':
                    # Verify the entity is valid before adopting
                    try:
                        self.store._read_and_verify(entity_file)
                    except (CorruptionError, Exception) as e:
                        # If corrupted, delete it instead of adopting
                        error_msg = f"Entity {entity_id} is corrupted, deleting: {str(e)}"
                        result.errors.append(error_msg)
                        entity_file.unlink()
                        result.repaired_entities.append(entity_id)
                        result.repaired_count += 1
                        continue

                    # Add synthetic WAL entry to adopt the orphan
                    import time
                    synthetic_entry = {
                        "op": "ADOPTED",
                        "entity_id": entity_id,
                        "reason": "orphan_recovery",
                        "timestamp": time.time()
                    }
                    # Compute checksum for the entry
                    checksum = compute_checksum(synthetic_entry)
                    synthetic_entry["checksum"] = checksum

                    # Append to WAL
                    with open(self.wal.wal_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(synthetic_entry) + '\n')

                    result.repaired_entities.append(entity_id)
                    result.repaired_count += 1

            except Exception as e:
                # Handle any unexpected errors
                error_msg = f"Failed to repair {entity_id}: {str(e)}"
                result.errors.append(error_msg)
                result.success = False

        return result
