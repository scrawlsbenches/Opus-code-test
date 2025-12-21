"""
Transaction management for Graph of Thought ACID-compliant storage.

Provides Transaction object with snapshot isolation and optimistic locking.
"""

from __future__ import annotations
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Any

from .types import Entity


class TransactionState(Enum):
    """State machine for transactions."""
    ACTIVE = "active"           # Transaction in progress
    PREPARING = "preparing"     # Entering commit phase
    COMMITTED = "committed"     # Successfully committed
    ABORTED = "aborted"         # Failed during commit
    ROLLED_BACK = "rolled_back" # Explicitly rolled back


def generate_transaction_id() -> str:
    """
    Generate unique transaction ID.

    Format: TX-YYYYMMDD-HHMMSS-XXXX where XXXX is random hex.

    Returns:
        Transaction ID string
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(2)  # 4 hex chars
    return f"TX-{timestamp}-{random_suffix}"


@dataclass
class Transaction:
    """
    Represents a database transaction with ACID guarantees.

    Provides snapshot isolation via version tracking and
    optimistic locking via read_set tracking.

    Attributes:
        id: Unique transaction identifier (TX-YYYYMMDD-HHMMSS-XXXX)
        state: Current transaction state
        started_at: ISO 8601 timestamp when transaction began
        snapshot_version: Store version at transaction start
        write_set: Buffered writes (entity_id → Entity)
        read_set: Read tracking for conflict detection (entity_id → version)
    """
    id: str
    state: TransactionState
    started_at: str
    snapshot_version: int
    write_set: Dict[str, Entity] = field(default_factory=dict)
    read_set: Dict[str, int] = field(default_factory=dict)

    def is_active(self) -> bool:
        """
        Check if transaction is active.

        Returns:
            True if transaction is in ACTIVE state
        """
        return self.state == TransactionState.ACTIVE

    def can_commit(self) -> bool:
        """
        Check if transaction can be committed.

        Returns:
            True if transaction is in ACTIVE state
        """
        return self.state == TransactionState.ACTIVE

    def can_rollback(self) -> bool:
        """
        Check if transaction can be rolled back.

        Returns:
            True if transaction is in ACTIVE or PREPARING state
        """
        return self.state in (TransactionState.ACTIVE, TransactionState.PREPARING)

    def add_read(self, entity_id: str, version: int) -> None:
        """
        Track a read operation for conflict detection.

        Args:
            entity_id: Entity that was read
            version: Version of entity when read
        """
        self.read_set[entity_id] = version

    def add_write(self, entity: Entity) -> None:
        """
        Add a write to the write set.

        Writes are buffered until commit.

        Args:
            entity: Entity to write
        """
        self.write_set[entity.id] = entity

    def get_write(self, entity_id: str) -> Optional[Entity]:
        """
        Get a pending write from the write set.

        This allows reads to see own writes within the transaction.

        Args:
            entity_id: Entity ID to look up

        Returns:
            Entity if found in write set, None otherwise
        """
        return self.write_set.get(entity_id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize transaction to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "state": self.state.value,
            "started_at": self.started_at,
            "snapshot_version": self.snapshot_version,
            "write_set": {
                entity_id: entity.to_dict()
                for entity_id, entity in self.write_set.items()
            },
            "read_set": self.read_set
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transaction:
        """
        Deserialize transaction from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Transaction instance
        """
        return cls(
            id=data["id"],
            state=TransactionState(data["state"]),
            started_at=data["started_at"],
            snapshot_version=data["snapshot_version"],
            write_set={
                entity_id: Entity.from_dict(entity_data)
                for entity_id, entity_data in data.get("write_set", {}).items()
            },
            read_set=data.get("read_set", {})
        )
