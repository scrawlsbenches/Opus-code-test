"""
Tests for CDG Write-Ahead Log.

Tests cover:
- WAL initialization and configuration
- Transaction logging operations
- WAL replay and recovery
- Sequence number management
- Corruption handling
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cortical.cdg.wal import CDGWALManager, _is_legacy_entry
from cortical.cdg.config import CDGConfig, DurabilityMode
from cortical.cdg.errors import CorruptionError


class TestIsLegacyEntry:
    """Test _is_legacy_entry helper function."""

    def test_legacy_entry_with_entity_id(self):
        """Test that entries with entity_id are detected as legacy."""
        data = {
            "op": "ADOPTED",
            "entity_id": "E-001",
            "timestamp": 1234567890.0,
            "checksum": "abc123"
        }
        assert _is_legacy_entry(data) is True

    def test_legacy_entry_with_numeric_timestamp(self):
        """Test that entries with numeric timestamp are detected as legacy."""
        data = {
            "op": "WRITE",
            "timestamp": 1234567890.0
        }
        assert _is_legacy_entry(data) is True

    def test_legacy_entry_without_seq_or_tx(self):
        """Test that entries without seq or tx fields are detected as legacy."""
        data = {
            "op": "WRITE",
            "data": {"key": "value"}
        }
        assert _is_legacy_entry(data) is True

    def test_current_entry_not_legacy(self):
        """Test that current format entries are not detected as legacy."""
        data = {
            "seq": 1,
            "ts": "2025-01-01T12:00:00+00:00",
            "tx": "TX-001",
            "op": "WRITE",
            "data": {}
        }
        assert _is_legacy_entry(data) is False


class TestCDGWALManager:
    """Test CDGWALManager class."""

    @pytest.fixture
    def wal_dir(self, tmp_path):
        """Create temporary WAL directory."""
        wal_path = tmp_path / "wal"
        return wal_path

    @pytest.fixture
    def config(self):
        """Create default CDG configuration."""
        return CDGConfig.for_simple_storage()

    @pytest.fixture
    def wal_manager(self, wal_dir, config):
        """Create WAL manager instance."""
        return CDGWALManager(wal_dir, config)

    def test_init_creates_directories(self, wal_dir, config):
        """Test that __init__ creates necessary directories."""
        manager = CDGWALManager(wal_dir, config)

        assert wal_dir.exists()
        assert (wal_dir / "archived").exists()

    def test_init_creates_wal_file_paths(self, wal_manager, wal_dir):
        """Test that WAL file paths are set correctly."""
        assert wal_manager.wal_file == wal_dir / "current.wal"
        assert wal_manager.seq_file == wal_dir / "_sequence.json"

    def test_load_sequence_empty(self, wal_manager):
        """Test loading sequence counter when file doesn't exist."""
        assert wal_manager._sequence == 0

    def test_load_sequence_existing(self, wal_dir, config):
        """Test loading sequence counter from existing file."""
        wal_dir.mkdir(parents=True, exist_ok=True)
        seq_file = wal_dir / "_sequence.json"
        seq_file.write_text('{"seq": 42}')

        manager = CDGWALManager(wal_dir, config)
        assert manager._sequence == 42

    def test_load_sequence_corrupted(self, wal_dir, config):
        """Test loading sequence counter from corrupted file."""
        wal_dir.mkdir(parents=True, exist_ok=True)
        seq_file = wal_dir / "_sequence.json"
        seq_file.write_text('not valid json')

        # Should not raise, should start from 0
        manager = CDGWALManager(wal_dir, config)
        assert manager._sequence == 0

    def test_save_sequence(self, wal_manager):
        """Test saving sequence counter to disk."""
        wal_manager._sequence = 100
        wal_manager._save_sequence()

        with open(wal_manager.seq_file, 'r') as f:
            data = json.load(f)
        assert data['seq'] == 100

    def test_next_seq_peeks_without_incrementing(self, wal_manager):
        """Test that _next_seq returns next value without incrementing."""
        # _next_seq now peeks at next value WITHOUT incrementing
        # This is intentional for crash safety - sequence only commits after write
        seq1 = wal_manager._next_seq()
        seq2 = wal_manager._next_seq()
        seq3 = wal_manager._next_seq()

        # All should return same value since we're only peeking
        assert seq1 == 1
        assert seq2 == 1
        assert seq3 == 1

    def test_commit_seq_increments_and_persists(self, wal_manager):
        """Test that _commit_seq increments and persists sequence."""
        # Get next sequence (peek)
        seq = wal_manager._next_seq()
        assert seq == 1

        # Commit it (this increments internal counter)
        wal_manager._commit_seq(seq)

        # Now next should be 2
        seq2 = wal_manager._next_seq()
        assert seq2 == 2

        # Commit again
        wal_manager._commit_seq(seq2)

        # Next should be 3
        seq3 = wal_manager._next_seq()
        assert seq3 == 3

    def test_log_properly_increments_sequence(self, wal_manager):
        """Test that log() properly increments sequence via commit."""
        # log() should use _next_seq() and _commit_seq() internally
        seq1 = wal_manager.log("TX-001", "TEST_OP1", {})
        seq2 = wal_manager.log("TX-001", "TEST_OP2", {})
        seq3 = wal_manager.log("TX-001", "TEST_OP3", {})

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    def test_log_creates_entry(self, wal_manager):
        """Test that log() creates WAL entry."""
        seq = wal_manager.log("TX-001", "TEST_OP", {"key": "value"})

        assert seq == 1
        assert wal_manager.wal_file.exists()

        # Verify content
        with open(wal_manager.wal_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry['seq'] == 1
        assert entry['tx'] == "TX-001"
        assert entry['op'] == "TEST_OP"
        assert entry['data']['key'] == "value"
        # Should have checksum
        assert 'checksum' in entry

    def test_log_tx_begin(self, wal_manager):
        """Test logging transaction begin."""
        seq = wal_manager.log_tx_begin("TX-001", snapshot_version=5)

        with open(wal_manager.wal_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry['op'] == "TX_BEGIN"
        assert entry['data']['snapshot'] == 5

    def test_log_write(self, wal_manager):
        """Test logging write operation."""
        seq = wal_manager.log_write("TX-001", "E-001", old_version=1, new_version=2)

        with open(wal_manager.wal_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry['op'] == "WRITE"
        assert entry['data']['entity_id'] == "E-001"
        assert entry['data']['old_version'] == 1
        assert entry['data']['new_version'] == 2

    def test_log_tx_prepare(self, wal_manager):
        """Test logging transaction prepare phase."""
        seq = wal_manager.log_tx_prepare("TX-001")

        with open(wal_manager.wal_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry['op'] == "TX_PREPARE"

    def test_log_tx_commit(self, wal_manager):
        """Test logging transaction commit."""
        seq = wal_manager.log_tx_commit("TX-001", version=10)

        with open(wal_manager.wal_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry['op'] == "TX_COMMIT"
        assert entry['data']['version'] == 10

    def test_log_tx_abort(self, wal_manager):
        """Test logging transaction abort."""
        seq = wal_manager.log_tx_abort("TX-001", reason="conflict detected")

        with open(wal_manager.wal_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry['op'] == "TX_ABORT"
        assert entry['data']['reason'] == "conflict detected"

    def test_log_tx_rollback(self, wal_manager):
        """Test logging transaction rollback."""
        seq = wal_manager.log_tx_rollback("TX-001", reason="test rollback")

        with open(wal_manager.wal_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry['op'] == "TX_ROLLBACK"
        assert entry['data']['reason'] == "test rollback"

    def test_multiple_entries(self, wal_manager):
        """Test multiple WAL entries are appended."""
        wal_manager.log_tx_begin("TX-001", 0)
        wal_manager.log_write("TX-001", "E-001", 0, 1)
        wal_manager.log_tx_commit("TX-001", 1)

        with open(wal_manager.wal_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_replay_empty_wal(self, wal_manager):
        """Test replaying empty WAL returns empty list."""
        entries = wal_manager.replay()
        assert entries == []

    def test_replay_returns_entries(self, wal_manager):
        """Test replay returns logged entries."""
        # Use proper logging to get correct checksums
        wal_manager.log_tx_begin("TX-001", 0)
        wal_manager.log_tx_commit("TX-001", 1)

        entries = wal_manager.replay()

        assert len(entries) == 2
        assert entries[0]['op'] == "TX_BEGIN"
        assert entries[1]['op'] == "TX_COMMIT"

    def test_replay_entries_returns_typed(self, wal_manager):
        """Test replay_entries returns typed TransactionWALEntry objects."""
        wal_manager.log_tx_begin("TX-001", 0)
        wal_manager.log_tx_commit("TX-001", 1)

        entries = wal_manager.replay_entries()

        assert len(entries) == 2
        assert entries[0].operation == "TX_BEGIN"
        assert entries[1].operation == "TX_COMMIT"

    def test_replay_skips_legacy_entries(self, wal_manager):
        """Test that replay skips legacy format entries."""
        # Write a legacy entry directly
        with open(wal_manager.wal_file, 'w') as f:
            f.write(json.dumps({
                "op": "ADOPTED",
                "entity_id": "E-001",
                "timestamp": 1234567890.0,
                "checksum": "abc"
            }) + "\n")

        # Write current format entry via proper API
        wal_manager._sequence = 0  # Reset sequence
        wal_manager.log_tx_begin("TX-001", 0)

        entries = wal_manager.replay()

        # Should only return the non-legacy entry (second one)
        assert len(entries) == 1
        assert entries[0]['op'] == "TX_BEGIN"

    def test_truncate_removes_wal(self, wal_manager):
        """Test truncate removes WAL file."""
        wal_manager.log_tx_begin("TX-001", 0)
        assert wal_manager.wal_file.exists()

        # Truncate with no archive (quick cleanup)
        wal_manager.truncate(archive=False)

        assert not wal_manager.wal_file.exists()

    def test_truncate_with_archive(self, wal_manager):
        """Test truncate with archiving."""
        wal_manager.log_tx_begin("TX-001", 0)
        wal_manager.log_tx_commit("TX-001", 1)

        # Truncate with archive
        archive_path = wal_manager.truncate(archive=True)

        assert archive_path is not None
        assert archive_path.exists()
        assert archive_path.parent == wal_manager.archive_dir
        assert not wal_manager.wal_file.exists()

    def test_truncate_empty_wal(self, wal_manager):
        """Test truncating non-existent WAL returns None."""
        result = wal_manager.truncate(archive=True)
        assert result is None

    def test_get_incomplete_transactions(self, wal_manager):
        """Test finding incomplete transactions."""
        # Complete transaction
        wal_manager.log_tx_begin("TX-001", 0)
        wal_manager.log_tx_commit("TX-001", 1)

        # Incomplete transaction
        wal_manager.log_tx_begin("TX-002", 1)
        wal_manager.log_write("TX-002", "E-001", 1, 2)
        # No commit for TX-002

        incomplete = wal_manager.get_incomplete_transactions()

        # Should contain TX-002
        incomplete_ids = [tx['tx_id'] for tx in incomplete]
        assert "TX-002" in incomplete_ids
        assert "TX-001" not in incomplete_ids

    def test_get_incomplete_transactions_empty(self, wal_manager):
        """Test get_incomplete_transactions on empty WAL."""
        incomplete = wal_manager.get_incomplete_transactions()
        assert incomplete == []

    def test_fsync_now(self, wal_manager):
        """Test fsync_now method."""
        wal_manager.log_tx_begin("TX-001", 0)
        # Should not raise
        wal_manager.fsync_now()

    def test_durability_mode_paranoid(self, wal_dir):
        """Test WAL with PARANOID durability mode."""
        config = CDGConfig.for_simple_storage()
        config.durability = DurabilityMode.PARANOID

        manager = CDGWALManager(wal_dir, config)
        manager.log_tx_begin("TX-001", 0)

        # Should complete without error (fsync is called)
        assert manager.wal_file.exists()

    def test_durability_mode_relaxed(self, wal_dir):
        """Test WAL with RELAXED durability mode (no fsync)."""
        config = CDGConfig.for_simple_storage()
        config.durability = DurabilityMode.RELAXED

        manager = CDGWALManager(wal_dir, config)
        manager.log_tx_begin("TX-001", 0)

        # Should complete without error (no fsync)
        assert manager.wal_file.exists()


class TestCDGWALManagerReplayCorruption:
    """Test WAL replay corruption handling."""

    @pytest.fixture
    def wal_dir(self, tmp_path):
        """Create temporary WAL directory."""
        return tmp_path / "wal"

    @pytest.fixture
    def config(self):
        """Create default CDG configuration."""
        return CDGConfig.for_simple_storage()

    def test_replay_skips_corrupted_json_lines(self, wal_dir, config):
        """Test that replay skips corrupted/invalid JSON lines."""
        wal_dir.mkdir(parents=True, exist_ok=True)

        # Create manager to get proper entries with checksums
        manager = CDGWALManager(wal_dir, config)
        manager.log_tx_begin("TX-001", 0)

        # Append corrupted line directly
        with open(manager.wal_file, 'a') as f:
            f.write("this is not valid json\n")

        # Add another valid entry
        manager.log_tx_commit("TX-001", 1)

        entries = manager.replay()

        # Should return both valid entries, skipping corrupted one
        assert len(entries) == 2
        assert entries[0]['op'] == "TX_BEGIN"
        assert entries[1]['op'] == "TX_COMMIT"

    def test_replay_handles_empty_lines(self, wal_dir, config):
        """Test that replay handles empty lines gracefully."""
        wal_dir.mkdir(parents=True, exist_ok=True)

        manager = CDGWALManager(wal_dir, config)

        # Write empty line, then valid entry, then empty line
        with open(manager.wal_file, 'w') as f:
            f.write("\n")  # Empty line

        manager.log_tx_begin("TX-001", 0)

        with open(manager.wal_file, 'a') as f:
            f.write("\n")  # Another empty line

        entries = manager.replay()

        assert len(entries) == 1
        assert entries[0]['op'] == "TX_BEGIN"

    def test_replay_skips_invalid_checksums(self, wal_dir, config):
        """Test that replay skips entries with invalid checksums."""
        wal_dir.mkdir(parents=True, exist_ok=True)

        manager = CDGWALManager(wal_dir, config)

        # Write valid entry
        manager.log_tx_begin("TX-001", 0)

        # Write entry with corrupted checksum
        with open(manager.wal_file, 'a') as f:
            f.write(json.dumps({
                "seq": 2,
                "ts": "2025-01-01T12:00:00+00:00",
                "tx": "TX-001",
                "op": "TX_COMMIT",
                "data": {"version": 1},
                "checksum": "invalid_checksum_value"
            }) + "\n")

        entries = manager.replay()

        # Should only return the valid entry
        assert len(entries) == 1
        assert entries[0]['op'] == "TX_BEGIN"
