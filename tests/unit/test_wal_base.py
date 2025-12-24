"""
Unit tests for WAL base classes.

Tests cover:
- BaseWALEntry base class methods
- TransactionWALEntry transaction-specific methods
- SnapshotInfo, WALIndex, RecoveryResult data classes
- Serialization and deserialization
- Checksum computation and verification
- Edge cases for both entry types
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from cortical.wal import (
    BaseWALEntry,
    TransactionWALEntry,
    SnapshotInfo,
    WALIndex,
    RecoveryResult,
)


class TestBaseWALEntry:
    """Tests for BaseWALEntry base class."""

    def test_creation_with_defaults(self):
        """Test creating entry with default values."""
        entry = BaseWALEntry(operation="TEST_OP")

        assert entry.operation == "TEST_OP"
        assert entry.timestamp  # Should be auto-generated
        assert entry.payload == {}  # Default empty dict
        assert entry.checksum == ""  # Not computed by BaseWALEntry

    def test_creation_with_custom_values(self):
        """Test creating entry with custom values."""
        timestamp = "2025-12-23T10:00:00"
        payload = {"key": "value", "count": 42}

        entry = BaseWALEntry(
            operation="CUSTOM_OP",
            timestamp=timestamp,
            payload=payload,
            checksum="abc123"
        )

        assert entry.operation == "CUSTOM_OP"
        assert entry.timestamp == timestamp
        assert entry.payload == payload
        assert entry.checksum == "abc123"

    def test_get_checksum_data(self):
        """Test _get_checksum_data returns correct fields."""
        timestamp = "2025-12-23T10:00:00"
        payload = {"test": "data"}

        entry = BaseWALEntry(
            operation="TEST",
            timestamp=timestamp,
            payload=payload
        )

        checksum_data = entry._get_checksum_data()

        assert checksum_data["operation"] == "TEST"
        assert checksum_data["timestamp"] == timestamp
        assert checksum_data["payload"] == payload
        assert len(checksum_data) == 3  # Only these 3 fields

    def test_compute_checksum(self):
        """Test _compute_checksum generates valid checksum."""
        entry = BaseWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"data": "test"}
        )

        checksum = entry._compute_checksum()

        # Should be 16-character hex string (truncated SHA256)
        assert isinstance(checksum, str)
        assert len(checksum) == 16
        assert all(c in '0123456789abcdef' for c in checksum)

    def test_compute_checksum_deterministic(self):
        """Test checksum is deterministic for same content."""
        entry1 = BaseWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"data": "test"}
        )
        entry2 = BaseWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"data": "test"}
        )

        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 == checksum2

    def test_compute_checksum_changes_with_operation(self):
        """Test checksum changes when operation changes."""
        entry1 = BaseWALEntry(operation="OP1", timestamp="2025-12-23T10:00:00")
        entry2 = BaseWALEntry(operation="OP2", timestamp="2025-12-23T10:00:00")

        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 != checksum2

    def test_compute_checksum_changes_with_timestamp(self):
        """Test checksum changes when timestamp changes."""
        entry1 = BaseWALEntry(operation="TEST", timestamp="2025-12-23T10:00:00")
        entry2 = BaseWALEntry(operation="TEST", timestamp="2025-12-23T11:00:00")

        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 != checksum2

    def test_compute_checksum_changes_with_payload(self):
        """Test checksum changes when payload changes."""
        entry1 = BaseWALEntry(operation="TEST", payload={"key": "value1"})
        entry2 = BaseWALEntry(operation="TEST", payload={"key": "value2"})

        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 != checksum2

    def test_to_json(self):
        """Test JSON serialization."""
        entry = BaseWALEntry(
            operation="TEST_OP",
            timestamp="2025-12-23T10:00:00",
            payload={"key": "value"},
            checksum="abc123"
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["operation"] == "TEST_OP"
        assert data["timestamp"] == "2025-12-23T10:00:00"
        assert data["payload"] == {"key": "value"}
        assert data["checksum"] == "abc123"

    def test_to_json_empty_payload(self):
        """Test JSON serialization with empty payload."""
        entry = BaseWALEntry(operation="TEST")

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["payload"] == {}

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps({
            "operation": "TEST_OP",
            "timestamp": "2025-12-23T10:00:00",
            "payload": {"key": "value"},
            "checksum": "abc123"
        })

        entry = BaseWALEntry.from_json(json_str)

        assert entry.operation == "TEST_OP"
        assert entry.timestamp == "2025-12-23T10:00:00"
        assert entry.payload == {"key": "value"}
        assert entry.checksum == "abc123"

    def test_from_json_minimal(self):
        """Test JSON deserialization with minimal fields."""
        json_str = json.dumps({
            "operation": "TEST",
            "timestamp": "2025-12-23T10:00:00",
            "payload": {},
            "checksum": ""
        })

        entry = BaseWALEntry.from_json(json_str)

        assert entry.operation == "TEST"
        assert entry.payload == {}
        assert entry.checksum == ""

    def test_roundtrip_serialization(self):
        """Test that serialization roundtrip preserves data."""
        original = BaseWALEntry(
            operation="ROUNDTRIP",
            timestamp="2025-12-23T10:00:00",
            payload={"nested": {"data": [1, 2, 3]}},
            checksum="test123"
        )

        json_str = original.to_json()
        restored = BaseWALEntry.from_json(json_str)

        assert restored.operation == original.operation
        assert restored.timestamp == original.timestamp
        assert restored.payload == original.payload
        assert restored.checksum == original.checksum

    def test_verify_valid_checksum(self):
        """Test verify() returns True for valid checksum."""
        entry = BaseWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"data": "test"}
        )
        # Manually set correct checksum
        entry.checksum = entry._compute_checksum()

        assert entry.verify() is True

    def test_verify_invalid_checksum(self):
        """Test verify() returns False for invalid checksum."""
        entry = BaseWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"data": "test"}
        )
        entry.checksum = "invalid_checksum"

        assert entry.verify() is False

    def test_verify_empty_checksum(self):
        """Test verify() returns False for empty checksum."""
        entry = BaseWALEntry(operation="TEST")
        # checksum is "" by default

        assert entry.verify() is False

    def test_verify_detects_tampering(self):
        """Test verify() detects when data is modified after checksum."""
        entry = BaseWALEntry(
            operation="ORIGINAL",
            timestamp="2025-12-23T10:00:00"
        )
        entry.checksum = entry._compute_checksum()

        # Tamper with data
        entry.operation = "TAMPERED"

        assert entry.verify() is False


class TestTransactionWALEntry:
    """Tests for TransactionWALEntry transaction-specific class."""

    def test_creation_with_defaults(self):
        """Test creating entry with default values."""
        entry = TransactionWALEntry(operation="TX_BEGIN")

        assert entry.operation == "TX_BEGIN"
        assert entry.seq == 0
        assert entry.tx_id == ""
        assert entry.timestamp  # Auto-generated
        assert entry.payload == {}
        assert entry.checksum  # Should be auto-computed in __post_init__

    def test_creation_with_custom_values(self):
        """Test creating entry with custom values."""
        entry = TransactionWALEntry(
            operation="TX_COMMIT",
            seq=42,
            tx_id="TX-12345",
            timestamp="2025-12-23T10:00:00",
            payload={"version": 10}
        )

        assert entry.operation == "TX_COMMIT"
        assert entry.seq == 42
        assert entry.tx_id == "TX-12345"
        assert entry.timestamp == "2025-12-23T10:00:00"
        assert entry.payload == {"version": 10}
        assert entry.checksum  # Auto-computed

    def test_post_init_computes_checksum(self):
        """Test __post_init__ auto-computes checksum."""
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001"
        )

        # Checksum should be computed automatically
        assert entry.checksum
        assert len(entry.checksum) == 16
        assert entry.verify()

    def test_post_init_preserves_existing_checksum(self):
        """Test __post_init__ doesn't overwrite existing checksum."""
        existing_checksum = "abc123def456"
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            checksum=existing_checksum
        )

        # Should preserve the existing checksum
        assert entry.checksum == existing_checksum

    def test_get_checksum_data(self):
        """Test _get_checksum_data returns transaction-specific fields."""
        entry = TransactionWALEntry(
            operation="WRITE",
            seq=42,
            tx_id="TX-001",
            timestamp="2025-12-23T10:00:00",
            payload={"entity_id": "task-1"}
        )

        checksum_data = entry._get_checksum_data()

        # Uses abbreviated field names
        assert checksum_data["seq"] == 42
        assert checksum_data["ts"] == "2025-12-23T10:00:00"
        assert checksum_data["tx"] == "TX-001"
        assert checksum_data["op"] == "WRITE"
        assert checksum_data["data"] == {"entity_id": "task-1"}
        assert len(checksum_data) == 5  # Only these 5 fields

    def test_compute_checksum_deterministic(self):
        """Test checksum is deterministic for same transaction data."""
        entry1 = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001",
            timestamp="2025-12-23T10:00:00"
        )
        entry2 = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001",
            timestamp="2025-12-23T10:00:00"
        )

        # Remove auto-computed checksums and recompute
        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 == checksum2

    def test_compute_checksum_changes_with_seq(self):
        """Test checksum changes when sequence number changes."""
        entry1 = TransactionWALEntry(operation="TEST", seq=1, tx_id="TX-001")
        entry2 = TransactionWALEntry(operation="TEST", seq=2, tx_id="TX-001")

        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 != checksum2

    def test_compute_checksum_changes_with_tx_id(self):
        """Test checksum changes when transaction ID changes."""
        entry1 = TransactionWALEntry(operation="TEST", seq=1, tx_id="TX-001")
        entry2 = TransactionWALEntry(operation="TEST", seq=1, tx_id="TX-002")

        checksum1 = entry1._compute_checksum()
        checksum2 = entry2._compute_checksum()

        assert checksum1 != checksum2

    def test_to_dict(self):
        """Test to_dict() serialization."""
        entry = TransactionWALEntry(
            operation="TX_COMMIT",
            seq=42,
            tx_id="TX-12345",
            timestamp="2025-12-23T10:00:00",
            payload={"version": 10},
            checksum="test123"
        )

        data = entry.to_dict()

        # Uses abbreviated field names
        assert data["seq"] == 42
        assert data["ts"] == "2025-12-23T10:00:00"
        assert data["tx"] == "TX-12345"
        assert data["op"] == "TX_COMMIT"
        assert data["data"] == {"version": 10}
        assert data["checksum"] == "test123"
        assert len(data) == 6

    def test_to_dict_empty_payload(self):
        """Test to_dict() with empty payload."""
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001"
        )

        data = entry.to_dict()

        assert data["data"] == {}

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {
            "seq": 42,
            "ts": "2025-12-23T10:00:00",
            "tx": "TX-12345",
            "op": "TX_COMMIT",
            "data": {"version": 10},
            "checksum": "test123"
        }

        entry = TransactionWALEntry.from_dict(data)

        assert entry.seq == 42
        assert entry.timestamp == "2025-12-23T10:00:00"
        assert entry.tx_id == "TX-12345"
        assert entry.operation == "TX_COMMIT"
        assert entry.payload == {"version": 10}
        assert entry.checksum == "test123"

    def test_from_dict_minimal(self):
        """Test from_dict() with minimal fields."""
        data = {
            "op": "TEST"
        }

        entry = TransactionWALEntry.from_dict(data)

        assert entry.operation == "TEST"
        assert entry.seq == 0  # Default
        assert entry.timestamp == ""  # Default
        assert entry.tx_id == ""  # Default
        assert entry.payload == {}  # Default
        # Checksum is auto-computed in __post_init__ even when empty
        assert entry.checksum  # Should have a checksum

    def test_from_dict_missing_fields_use_defaults(self):
        """Test from_dict() uses defaults for missing fields."""
        data = {
            "seq": 5,
            "op": "WRITE"
        }

        entry = TransactionWALEntry.from_dict(data)

        assert entry.seq == 5
        assert entry.operation == "WRITE"
        assert entry.timestamp == ""
        assert entry.tx_id == ""
        assert entry.payload == {}

    def test_roundtrip_dict_serialization(self):
        """Test that dict serialization roundtrip preserves data."""
        original = TransactionWALEntry(
            operation="ROUNDTRIP",
            seq=99,
            tx_id="TX-ROUND",
            timestamp="2025-12-23T10:00:00",
            payload={"complex": {"nested": [1, 2, 3]}}
        )

        data = original.to_dict()
        restored = TransactionWALEntry.from_dict(data)

        assert restored.seq == original.seq
        assert restored.timestamp == original.timestamp
        assert restored.tx_id == original.tx_id
        assert restored.operation == original.operation
        assert restored.payload == original.payload
        assert restored.checksum == original.checksum

    def test_verify_valid_transaction_entry(self):
        """Test verify() works correctly for transaction entries."""
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001",
            timestamp="2025-12-23T10:00:00"
        )
        # Checksum auto-computed in __post_init__

        assert entry.verify() is True

    def test_verify_detects_tampering_in_seq(self):
        """Test verify() detects tampering with sequence number."""
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001"
        )
        original_checksum = entry.checksum

        # Tamper with seq
        entry.seq = 999

        assert entry.verify() is False

    def test_verify_detects_tampering_in_tx_id(self):
        """Test verify() detects tampering with transaction ID."""
        entry = TransactionWALEntry(
            operation="TX_BEGIN",
            seq=1,
            tx_id="TX-001"
        )
        original_checksum = entry.checksum

        # Tamper with tx_id
        entry.tx_id = "TX-HACKED"

        assert entry.verify() is False

    def test_verify_detects_tampering_in_payload(self):
        """Test verify() detects tampering with payload."""
        entry = TransactionWALEntry(
            operation="WRITE",
            seq=1,
            tx_id="TX-001",
            payload={"entity_id": "task-1"}
        )

        # Tamper with payload
        entry.payload["entity_id"] = "task-999"

        assert entry.verify() is False

    def test_transaction_types(self):
        """Test different transaction operation types."""
        operations = [
            "TX_BEGIN",
            "TX_PREPARE",
            "TX_COMMIT",
            "TX_ABORT",
            "TX_ROLLBACK",
            "WRITE"
        ]

        for op in operations:
            entry = TransactionWALEntry(
                operation=op,
                seq=1,
                tx_id="TX-001"
            )
            assert entry.operation == op
            assert entry.verify()

    def test_large_sequence_number(self):
        """Test handling of large sequence numbers."""
        entry = TransactionWALEntry(
            operation="WRITE",
            seq=999999999,
            tx_id="TX-001"
        )

        assert entry.seq == 999999999
        assert entry.verify()

    def test_complex_payload(self):
        """Test handling of complex nested payload."""
        complex_payload = {
            "entity_id": "task-123",
            "old_version": 5,
            "new_version": 6,
            "changes": {
                "status": {"old": "pending", "new": "in_progress"},
                "assignee": {"old": None, "new": "alice"},
                "metadata": {
                    "tags": ["urgent", "backend"],
                    "priority": "high"
                }
            }
        }

        entry = TransactionWALEntry(
            operation="WRITE",
            seq=1,
            tx_id="TX-001",
            payload=complex_payload
        )

        # Should handle complex nested structures
        assert entry.payload == complex_payload
        assert entry.verify()

        # Roundtrip should preserve
        data = entry.to_dict()
        restored = TransactionWALEntry.from_dict(data)
        assert restored.payload == complex_payload


class TestBaseVsTransactionEntry:
    """Tests comparing BaseWALEntry and TransactionWALEntry behavior."""

    def test_different_checksum_algorithms(self):
        """Test that Base and Transaction entries use different checksum data."""
        # BaseWALEntry uses: operation, timestamp, payload
        base = BaseWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"key": "value"}
        )

        # TransactionWALEntry uses: seq, ts, tx, op, data
        txn = TransactionWALEntry(
            operation="TEST",
            timestamp="2025-12-23T10:00:00",
            payload={"key": "value"},
            seq=0,
            tx_id=""
        )

        # Even with similar data, checksums will differ due to different algorithms
        base.checksum = base._compute_checksum()
        txn_checksum = txn._compute_checksum()

        # Different checksum implementations
        assert base.checksum != txn_checksum

    def test_inheritance_relationship(self):
        """Test that TransactionWALEntry inherits from BaseWALEntry."""
        entry = TransactionWALEntry(operation="TEST")

        assert isinstance(entry, BaseWALEntry)
        assert isinstance(entry, TransactionWALEntry)

    def test_base_methods_available_in_transaction(self):
        """Test that transaction entries can use base class methods."""
        entry = TransactionWALEntry(
            operation="TEST",
            seq=1,
            tx_id="TX-001"
        )

        # Can use base class methods
        json_str = entry.to_json()
        assert json_str

        # Can verify using base class verify()
        assert entry.verify()


class TestSnapshotInfo:
    """Tests for SnapshotInfo dataclass."""

    def test_creation(self):
        """Test creating SnapshotInfo."""
        info = SnapshotInfo(
            snapshot_id="snap_20251223_100000",
            timestamp="2025-12-23T10:00:00",
            document_count=42,
            size_bytes=1024,
            operations_since_last=10,
            wal_file="wal_001.jsonl",
            wal_offset=5,
            path=Path("/tmp/snap.json.gz")
        )

        assert info.snapshot_id == "snap_20251223_100000"
        assert info.timestamp == "2025-12-23T10:00:00"
        assert info.document_count == 42
        assert info.size_bytes == 1024
        assert info.operations_since_last == 10
        assert info.wal_file == "wal_001.jsonl"
        assert info.wal_offset == 5
        assert info.path == Path("/tmp/snap.json.gz")

    def test_to_dict(self):
        """Test to_dict() serialization."""
        info = SnapshotInfo(
            snapshot_id="snap_001",
            timestamp="2025-12-23T10:00:00",
            document_count=10,
            size_bytes=512,
            operations_since_last=5,
            wal_file="wal.jsonl",
            wal_offset=0,
            path=Path("/tmp/snap.json")
        )

        data = info.to_dict()

        assert data["snapshot_id"] == "snap_001"
        assert data["timestamp"] == "2025-12-23T10:00:00"
        assert data["document_count"] == 10
        assert data["size_bytes"] == 512
        assert data["operations_since_last"] == 5
        assert data["wal_file"] == "wal.jsonl"
        assert data["wal_offset"] == 0
        assert data["path"] == "/tmp/snap.json"  # Converted to string

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {
            "snapshot_id": "snap_002",
            "timestamp": "2025-12-23T11:00:00",
            "document_count": 20,
            "size_bytes": 2048,
            "operations_since_last": 15,
            "wal_file": "wal_002.jsonl",
            "wal_offset": 10,
            "path": "/tmp/snapshots/snap_002.json.gz"
        }

        info = SnapshotInfo.from_dict(data)

        assert info.snapshot_id == "snap_002"
        assert info.timestamp == "2025-12-23T11:00:00"
        assert info.document_count == 20
        assert info.size_bytes == 2048
        assert info.operations_since_last == 15
        assert info.wal_file == "wal_002.jsonl"
        assert info.wal_offset == 10
        assert info.path == Path("/tmp/snapshots/snap_002.json.gz")

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization preserves data."""
        original = SnapshotInfo(
            snapshot_id="snap_roundtrip",
            timestamp="2025-12-23T12:00:00",
            document_count=100,
            size_bytes=10240,
            operations_since_last=50,
            wal_file="wal_roundtrip.jsonl",
            wal_offset=25,
            path=Path("/var/wal/snapshots/snap.json.gz")
        )

        data = original.to_dict()
        restored = SnapshotInfo.from_dict(data)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.timestamp == original.timestamp
        assert restored.document_count == original.document_count
        assert restored.size_bytes == original.size_bytes
        assert restored.operations_since_last == original.operations_since_last
        assert restored.wal_file == original.wal_file
        assert restored.wal_offset == original.wal_offset
        assert restored.path == original.path


class TestWALIndex:
    """Tests for WALIndex dataclass."""

    def test_creation_with_defaults(self):
        """Test creating WALIndex with default values."""
        index = WALIndex()

        assert index.latest_snapshot_id is None
        assert index.current_wal_file == ""
        assert index.wal_entry_count == 0
        assert index.last_compaction is None
        assert index.snapshots == []
        assert index.version == 1

    def test_creation_with_custom_values(self):
        """Test creating WALIndex with custom values."""
        index = WALIndex(
            latest_snapshot_id="snap_001",
            current_wal_file="wal_001.jsonl",
            wal_entry_count=42,
            last_compaction="2025-12-23T10:00:00",
            snapshots=["snap_001", "snap_002"],
            version=2
        )

        assert index.latest_snapshot_id == "snap_001"
        assert index.current_wal_file == "wal_001.jsonl"
        assert index.wal_entry_count == 42
        assert index.last_compaction == "2025-12-23T10:00:00"
        assert index.snapshots == ["snap_001", "snap_002"]
        assert index.version == 2

    def test_to_dict(self):
        """Test to_dict() serialization."""
        index = WALIndex(
            latest_snapshot_id="snap_latest",
            current_wal_file="current.wal",
            wal_entry_count=100,
            last_compaction="2025-12-23T09:00:00",
            snapshots=["snap_1", "snap_2", "snap_3"],
            version=1
        )

        data = index.to_dict()

        assert data["latest_snapshot_id"] == "snap_latest"
        assert data["current_wal_file"] == "current.wal"
        assert data["wal_entry_count"] == 100
        assert data["last_compaction"] == "2025-12-23T09:00:00"
        assert data["snapshots"] == ["snap_1", "snap_2", "snap_3"]
        assert data["version"] == 1

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {
            "latest_snapshot_id": "snap_from_dict",
            "current_wal_file": "wal_from_dict.jsonl",
            "wal_entry_count": 200,
            "last_compaction": "2025-12-23T08:00:00",
            "snapshots": ["snap_a", "snap_b"],
            "version": 2
        }

        index = WALIndex.from_dict(data)

        assert index.latest_snapshot_id == "snap_from_dict"
        assert index.current_wal_file == "wal_from_dict.jsonl"
        assert index.wal_entry_count == 200
        assert index.last_compaction == "2025-12-23T08:00:00"
        assert index.snapshots == ["snap_a", "snap_b"]
        assert index.version == 2

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization preserves data."""
        original = WALIndex(
            latest_snapshot_id="snap_rt",
            current_wal_file="wal_rt.jsonl",
            wal_entry_count=999,
            last_compaction="2025-12-23T07:00:00",
            snapshots=["s1", "s2", "s3", "s4"],
            version=3
        )

        data = original.to_dict()
        restored = WALIndex.from_dict(data)

        assert restored.latest_snapshot_id == original.latest_snapshot_id
        assert restored.current_wal_file == original.current_wal_file
        assert restored.wal_entry_count == original.wal_entry_count
        assert restored.last_compaction == original.last_compaction
        assert restored.snapshots == original.snapshots
        assert restored.version == original.version


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_creation_minimal(self):
        """Test creating RecoveryResult with minimal fields."""
        result = RecoveryResult(success=True)

        assert result.success is True
        assert result.snapshot_id is None
        assert result.wal_entries_replayed == 0
        assert result.documents_recovered == 0
        assert result.errors == []
        assert result.state is None

    def test_creation_full(self):
        """Test creating RecoveryResult with all fields."""
        state = {"documents": {"doc1": "content"}}
        errors = ["Error 1", "Error 2"]

        result = RecoveryResult(
            success=True,
            snapshot_id="snap_recovery",
            wal_entries_replayed=50,
            documents_recovered=10,
            errors=errors,
            state=state
        )

        assert result.success is True
        assert result.snapshot_id == "snap_recovery"
        assert result.wal_entries_replayed == 50
        assert result.documents_recovered == 10
        assert result.errors == errors
        assert result.state == state

    def test_failed_recovery(self):
        """Test creating a failed recovery result."""
        result = RecoveryResult(
            success=False,
            errors=["No snapshot found", "WAL corrupted"]
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert "No snapshot found" in result.errors
        assert result.state is None

    def test_partial_recovery(self):
        """Test recovery with some errors but overall success."""
        result = RecoveryResult(
            success=True,
            snapshot_id="snap_001",
            wal_entries_replayed=100,
            documents_recovered=95,
            errors=["Failed to replay 5 corrupted entries"],
            state={"documents": {}}
        )

        assert result.success is True
        assert result.wal_entries_replayed == 100
        assert result.documents_recovered == 95
        assert len(result.errors) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
