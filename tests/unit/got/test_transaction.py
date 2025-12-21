"""Tests for Transaction module."""

import pytest
from datetime import datetime, timezone

from cortical.got.transaction import (
    Transaction,
    TransactionState,
    generate_transaction_id
)
from cortical.got.types import Entity


class TestTransactionState:
    """Tests for TransactionState enum."""

    def test_transaction_state_values(self):
        """Test all state values are defined."""
        assert TransactionState.ACTIVE.value == "active"
        assert TransactionState.PREPARING.value == "preparing"
        assert TransactionState.COMMITTED.value == "committed"
        assert TransactionState.ABORTED.value == "aborted"
        assert TransactionState.ROLLED_BACK.value == "rolled_back"


class TestGenerateTransactionId:
    """Tests for generate_transaction_id function."""

    def test_generate_transaction_id_format(self):
        """Test transaction ID follows format TX-YYYYMMDD-HHMMSS-XXXX."""
        tx_id = generate_transaction_id()

        # Should start with TX-
        assert tx_id.startswith("TX-")

        # Should have correct structure
        parts = tx_id.split("-")
        assert len(parts) == 4  # TX, YYYYMMDD, HHMMSS, XXXX

        # Date part should be 8 digits
        assert len(parts[1]) == 8
        assert parts[1].isdigit()

        # Time part should be 6 digits
        assert len(parts[2]) == 6
        assert parts[2].isdigit()

        # Random suffix should be 4 hex chars
        assert len(parts[3]) == 4
        int(parts[3], 16)  # Should parse as hex

    def test_generate_transaction_id_unique(self):
        """Test generated IDs are unique."""
        ids = [generate_transaction_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestTransaction:
    """Tests for Transaction class."""

    @pytest.fixture
    def transaction(self):
        """Create a test transaction."""
        return Transaction(
            id="TX-20251221-120000-abcd",
            state=TransactionState.ACTIVE,
            started_at=datetime.now(timezone.utc).isoformat(),
            snapshot_version=42
        )

    @pytest.fixture
    def sample_entity(self):
        """Create a sample entity for testing."""
        return Entity(
            id="task-123",
            entity_type="task",
            version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            modified_at=datetime.now(timezone.utc).isoformat()
        )

    def test_transaction_state_machine(self, transaction):
        """Test state transitions follow rules."""
        # Start in ACTIVE
        assert transaction.state == TransactionState.ACTIVE

        # Can transition to PREPARING
        transaction.state = TransactionState.PREPARING
        assert transaction.state == TransactionState.PREPARING

        # Can transition to COMMITTED
        transaction.state = TransactionState.COMMITTED
        assert transaction.state == TransactionState.COMMITTED

        # Can transition from ACTIVE to ROLLED_BACK
        transaction.state = TransactionState.ACTIVE
        transaction.state = TransactionState.ROLLED_BACK
        assert transaction.state == TransactionState.ROLLED_BACK

        # Can transition from PREPARING to ABORTED
        transaction.state = TransactionState.PREPARING
        transaction.state = TransactionState.ABORTED
        assert transaction.state == TransactionState.ABORTED

    def test_write_buffered_until_commit(self, transaction, sample_entity):
        """Test writes go to write_set."""
        # Initially empty
        assert len(transaction.write_set) == 0

        # Add write
        transaction.add_write(sample_entity)

        # Should be in write_set
        assert len(transaction.write_set) == 1
        assert "task-123" in transaction.write_set
        assert transaction.write_set["task-123"] == sample_entity

    def test_read_sees_own_writes(self, transaction, sample_entity):
        """Test get_write returns from write_set."""
        # Add to write set
        transaction.add_write(sample_entity)

        # Should be able to read it
        result = transaction.get_write("task-123")
        assert result is not None
        assert result == sample_entity

        # Non-existent entity returns None
        assert transaction.get_write("nonexistent") is None

    def test_read_sees_snapshot_version(self, transaction):
        """Test read_set tracks version."""
        # Add read tracking
        transaction.add_read("task-123", 5)

        # Should be in read_set
        assert "task-123" in transaction.read_set
        assert transaction.read_set["task-123"] == 5

        # Add another read
        transaction.add_read("task-456", 10)
        assert len(transaction.read_set) == 2
        assert transaction.read_set["task-456"] == 10

    def test_is_active_true_when_active(self, transaction):
        """Test is_active returns True when ACTIVE."""
        transaction.state = TransactionState.ACTIVE
        assert transaction.is_active() is True

    def test_is_active_false_when_committed(self, transaction):
        """Test is_active returns False when COMMITTED."""
        transaction.state = TransactionState.COMMITTED
        assert transaction.is_active() is False

        transaction.state = TransactionState.ABORTED
        assert transaction.is_active() is False

        transaction.state = TransactionState.ROLLED_BACK
        assert transaction.is_active() is False

    def test_can_commit_true_when_active(self, transaction):
        """Test can_commit returns True when ACTIVE."""
        transaction.state = TransactionState.ACTIVE
        assert transaction.can_commit() is True

    def test_can_commit_false_when_rolled_back(self, transaction):
        """Test can_commit returns False when ROLLED_BACK."""
        transaction.state = TransactionState.ROLLED_BACK
        assert transaction.can_commit() is False

        transaction.state = TransactionState.COMMITTED
        assert transaction.can_commit() is False

        transaction.state = TransactionState.ABORTED
        assert transaction.can_commit() is False

    def test_can_rollback_true_when_active(self, transaction):
        """Test can_rollback returns True when ACTIVE."""
        transaction.state = TransactionState.ACTIVE
        assert transaction.can_rollback() is True

        transaction.state = TransactionState.PREPARING
        assert transaction.can_rollback() is True

    def test_can_rollback_false_when_committed(self, transaction):
        """Test can_rollback returns False when COMMITTED."""
        transaction.state = TransactionState.COMMITTED
        assert transaction.can_rollback() is False

        transaction.state = TransactionState.ABORTED
        assert transaction.can_rollback() is False

        transaction.state = TransactionState.ROLLED_BACK
        assert transaction.can_rollback() is False

    def test_to_dict_serialization(self, transaction, sample_entity):
        """Test to_dict serialization."""
        transaction.add_write(sample_entity)
        transaction.add_read("task-456", 10)

        data = transaction.to_dict()

        assert data["id"] == "TX-20251221-120000-abcd"
        assert data["state"] == "active"
        assert data["snapshot_version"] == 42
        assert "started_at" in data

        # Write set serialized
        assert "task-123" in data["write_set"]
        assert data["write_set"]["task-123"]["id"] == "task-123"

        # Read set serialized
        assert data["read_set"]["task-456"] == 10

    def test_from_dict_deserialization(self, sample_entity):
        """Test from_dict deserialization."""
        data = {
            "id": "TX-20251221-120000-abcd",
            "state": "active",
            "started_at": "2025-12-21T12:00:00Z",
            "snapshot_version": 42,
            "write_set": {
                "task-123": sample_entity.to_dict()
            },
            "read_set": {
                "task-456": 10
            }
        }

        tx = Transaction.from_dict(data)

        assert tx.id == "TX-20251221-120000-abcd"
        assert tx.state == TransactionState.ACTIVE
        assert tx.started_at == "2025-12-21T12:00:00Z"
        assert tx.snapshot_version == 42

        # Write set deserialized
        assert "task-123" in tx.write_set
        assert tx.write_set["task-123"].id == "task-123"

        # Read set deserialized
        assert tx.read_set["task-456"] == 10
