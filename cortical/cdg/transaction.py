"""
Transaction management for CDG ACID-compliant storage.

Provides Transaction object with snapshot isolation and optimistic locking,
lifted from GoT with CDG extensions for partition tracking.

Transaction Lifecycle:
    ACTIVE → PREPARING → COMMITTED
                      → ABORTED
           → ROLLED_BACK

Example:
    tx = Transaction(
        id=generate_transaction_id(),
        state=TransactionState.ACTIVE,
        started_at=datetime.now(timezone.utc).isoformat(),
        snapshot_version=store.current_version()
    )

    # Track reads for conflict detection
    entity = store.read_in_tx(tx, "E-001")
    tx.add_read(entity.id, entity.version)

    # Buffer writes
    entity.properties["status"] = "updated"
    tx.add_write(entity)

    # Commit (checks for conflicts)
    store.commit(tx)
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Any, Set

from .types import Entity


class TransactionState(Enum):
    """
    State machine for transaction lifecycle.

    Valid transitions:
        ACTIVE → PREPARING → COMMITTED
        ACTIVE → PREPARING → ABORTED
        ACTIVE → ROLLED_BACK
        PREPARING → ROLLED_BACK
    """
    ACTIVE = "active"           # Transaction in progress
    PREPARING = "preparing"     # Entering commit phase
    COMMITTED = "committed"     # Successfully committed
    ABORTED = "aborted"         # Failed during commit
    ROLLED_BACK = "rolled_back" # Explicitly rolled back


def generate_transaction_id() -> str:
    """
    Generate unique transaction ID.

    Format: TX-YYYYMMDD-HHMMSS-XXXXXXXX where XXXXXXXX is random hex.
    This format is human-readable, sortable by time, and unique.

    Returns:
        Transaction ID string

    Example:
        >>> generate_transaction_id()
        'TX-20251231-120000-a1b2c3d4'
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4)  # 8 hex chars (~4 billion values)
    return f"TX-{timestamp}-{random_suffix}"


@dataclass
class Transaction:
    """
    Represents a database transaction with ACID guarantees.

    Provides snapshot isolation via version tracking and
    optimistic locking via read_set tracking.

    Lifted from GoT with CDG extensions:
    - touched_partitions: Tracks which partitions were accessed
    - metadata: Flexible transaction metadata

    Attributes:
        id: Unique transaction identifier (TX-YYYYMMDD-HHMMSS-XXXX)
        state: Current transaction state
        started_at: ISO 8601 timestamp when transaction began
        snapshot_version: Store version at transaction start
        write_set: Buffered writes (entity_id → Entity)
        read_set: Read tracking for conflict detection (entity_id → version)
        touched_partitions: Set of partition IDs accessed
        metadata: Optional transaction metadata

    Example:
        tx = Transaction(
            id=generate_transaction_id(),
            state=TransactionState.ACTIVE,
            started_at=datetime.now(timezone.utc).isoformat(),
            snapshot_version=100
        )

        # Read entity and track for conflict detection
        tx.add_read("E-001", 5)

        # Buffer write
        tx.add_write(updated_entity)

        # Check if ready to commit
        if tx.can_commit():
            # ... perform commit
    """
    id: str
    state: TransactionState
    started_at: str
    snapshot_version: int
    write_set: Dict[str, Entity] = field(default_factory=dict)
    read_set: Dict[str, int] = field(default_factory=dict)

    # CDG extension: partition tracking
    touched_partitions: Set[int] = field(default_factory=set)

    # CDG extension: transaction metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

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

    def is_complete(self) -> bool:
        """
        Check if transaction has completed (success or failure).

        Returns:
            True if transaction is COMMITTED, ABORTED, or ROLLED_BACK
        """
        return self.state in (
            TransactionState.COMMITTED,
            TransactionState.ABORTED,
            TransactionState.ROLLED_BACK
        )

    def add_read(self, entity_id: str, version: int, partition_id: int = 0) -> None:
        """
        Track a read operation for conflict detection.

        Args:
            entity_id: Entity that was read
            version: Version of entity when read
            partition_id: Partition the entity was read from (CDG extension)
        """
        self.read_set[entity_id] = version
        self.touched_partitions.add(partition_id)

    def add_write(self, entity: Entity, partition_id: int = 0) -> None:
        """
        Add a write to the write set.

        Writes are buffered until commit.

        Args:
            entity: Entity to write
            partition_id: Partition to write to (CDG extension)
        """
        self.write_set[entity.id] = entity
        self.touched_partitions.add(partition_id)

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

    def has_write(self, entity_id: str) -> bool:
        """
        Check if entity is in write set.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if entity is pending write
        """
        return entity_id in self.write_set

    def has_read(self, entity_id: str) -> bool:
        """
        Check if entity was read in this transaction.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if entity was read
        """
        return entity_id in self.read_set

    def is_cross_partition(self) -> bool:
        """
        Check if transaction spans multiple partitions.

        Cross-partition transactions may require 2PC coordination.

        Returns:
            True if more than one partition was touched
        """
        return len(self.touched_partitions) > 1

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
            "read_set": self.read_set,
            "touched_partitions": list(self.touched_partitions),
            "metadata": self.metadata,
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
            read_set=data.get("read_set", {}),
            touched_partitions=set(data.get("touched_partitions", [])),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def begin(cls, snapshot_version: int, **metadata: Any) -> "Transaction":
        """
        Factory method to create a new active transaction.

        Args:
            snapshot_version: Current store version for snapshot isolation
            **metadata: Optional transaction metadata

        Returns:
            New Transaction in ACTIVE state

        Example:
            tx = Transaction.begin(store.current_version(), user="agent-001")
        """
        return cls(
            id=generate_transaction_id(),
            state=TransactionState.ACTIVE,
            started_at=datetime.now(timezone.utc).isoformat(),
            snapshot_version=snapshot_version,
            metadata=metadata,
        )
