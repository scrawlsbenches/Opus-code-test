"""
Recovery result types shared between CDG and GoT.

These dataclasses provide structured results from recovery operations,
enabling consistent reporting and diagnostics across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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
        orphans_detected: List of entity IDs found without WAL records
        orphans_repaired: Number of orphans that were repaired (adopted)
        reconstructed_entities: List of entity IDs reconstructed from WAL
        indexes_rebuilt: True if indexes were rebuilt during recovery
        actions_taken: Human-readable log of recovery actions
    """

    success: bool
    recovered_transactions: int
    rolled_back: List[str] = field(default_factory=list)
    corrupted_entities: List[str] = field(default_factory=list)
    corrupted_wal_entries: int = 0
    orphans_detected: List[str] = field(default_factory=list)
    orphans_repaired: int = 0
    reconstructed_entities: List[str] = field(default_factory=list)
    indexes_rebuilt: bool = False
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
