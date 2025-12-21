"""
Unit tests for WAL (Write-Ahead Log) module.

Tests crash recovery, checksums, fsync, and transaction tracking.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from cortical.got.wal import WALManager
from cortical.got.checksums import compute_checksum


class TestWALManager:
    """Test suite for WALManager class."""

    def test_log_appends_with_checksum(self, tmp_path):
        """Verify that log entries have correct checksums."""
        wal = WALManager(tmp_path)

        # Log an entry
        seq = wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})

        # Read the WAL file
        wal_file = tmp_path / "current.wal"
        assert wal_file.exists()

        with open(wal_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        # Verify checksum field exists
        assert 'checksum' in entry

        # Verify checksum is correct
        expected_checksum = entry.pop('checksum')
        actual_checksum = compute_checksum(entry)
        assert actual_checksum == expected_checksum

        # Verify sequence number
        assert entry['seq'] == seq

    def test_fsync_called_on_every_log(self, tmp_path):
        """Verify that os.fsync is called on every log operation."""
        wal = WALManager(tmp_path)

        with patch('os.fsync') as mock_fsync:
            wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})

            # fsync should be called at least once (for WAL write)
            # Note: Also called for sequence file save
            assert mock_fsync.call_count >= 1

    def test_corrupted_entry_detected(self, tmp_path):
        """Verify that corrupted checksums are detected and skipped in replay."""
        wal = WALManager(tmp_path)

        # Log valid entry
        wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})

        # Manually corrupt the checksum in the WAL file
        wal_file = tmp_path / "current.wal"
        with open(wal_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        # Corrupt the checksum
        entry['checksum'] = 'CORRUPTED_CHECKSUM'

        with open(wal_file, 'w') as f:
            f.write(json.dumps(entry) + '\n')

        # Replay should skip corrupted entry
        entries = wal.replay()
        assert len(entries) == 0

    def test_incomplete_tx_detected(self, tmp_path):
        """Verify that incomplete transactions are detected."""
        wal = WALManager(tmp_path)

        # Start transaction but don't commit
        wal.log_tx_begin("TX-001", snapshot_version=5)
        wal.log_write("TX-001", "task-1", old_version=1, new_version=2)

        # Check for incomplete transactions
        incomplete = wal.get_incomplete_transactions()

        assert len(incomplete) == 1
        assert incomplete[0]['tx_id'] == "TX-001"
        assert incomplete[0]['state'] == "ACTIVE"
        assert incomplete[0]['snapshot'] == 5

    def test_truncate_archives_old_wal(self, tmp_path):
        """Verify that truncate moves WAL to archive directory."""
        wal = WALManager(tmp_path)

        # Log some entries
        wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})
        wal.log("TX-001", "TX_COMMIT", {"version": 6})

        # Truncate with archive
        archive_path = wal.truncate(archive=True)

        # Verify archived file exists
        assert archive_path is not None
        assert archive_path.exists()
        assert archive_path.parent == tmp_path / "archived"

        # Verify current.wal is gone
        wal_file = tmp_path / "current.wal"
        assert not wal_file.exists()

    def test_recovery_finds_incomplete_transactions(self, tmp_path):
        """Verify recovery finds multiple incomplete transactions."""
        wal = WALManager(tmp_path)

        # TX-001: Complete
        wal.log_tx_begin("TX-001", snapshot_version=5)
        wal.log_tx_commit("TX-001", version=6)

        # TX-002: Incomplete (ACTIVE)
        wal.log_tx_begin("TX-002", snapshot_version=6)

        # TX-003: Incomplete (PREPARING)
        wal.log_tx_begin("TX-003", snapshot_version=6)
        wal.log_tx_prepare("TX-003")

        # TX-004: Aborted (should not be incomplete)
        wal.log_tx_begin("TX-004", snapshot_version=6)
        wal.log_tx_abort("TX-004", reason="Conflict")

        # Find incomplete transactions
        incomplete = wal.get_incomplete_transactions()

        assert len(incomplete) == 2

        # Check TX-002
        tx2 = next(tx for tx in incomplete if tx['tx_id'] == 'TX-002')
        assert tx2['state'] == 'ACTIVE'
        assert tx2['snapshot'] == 6

        # Check TX-003
        tx3 = next(tx for tx in incomplete if tx['tx_id'] == 'TX-003')
        assert tx3['state'] == 'PREPARING'
        assert tx3['snapshot'] == 6

    def test_log_tx_begin_records_snapshot(self, tmp_path):
        """Verify TX_BEGIN records snapshot version."""
        wal = WALManager(tmp_path)

        wal.log_tx_begin("TX-001", snapshot_version=42)

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]['op'] == 'TX_BEGIN'
        assert entries[0]['data']['snapshot'] == 42

    def test_log_write_records_versions(self, tmp_path):
        """Verify WRITE records entity and versions."""
        wal = WALManager(tmp_path)

        wal.log_write("TX-001", entity_id="task-1", old_version=5, new_version=6)

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]['op'] == 'WRITE'
        assert entries[0]['data']['entity_id'] == 'task-1'
        assert entries[0]['data']['old_version'] == 5
        assert entries[0]['data']['new_version'] == 6

    def test_log_tx_commit_records_version(self, tmp_path):
        """Verify TX_COMMIT records final version."""
        wal = WALManager(tmp_path)

        wal.log_tx_commit("TX-001", version=10)

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]['op'] == 'TX_COMMIT'
        assert entries[0]['data']['version'] == 10

    def test_log_tx_abort_records_reason(self, tmp_path):
        """Verify TX_ABORT records reason."""
        wal = WALManager(tmp_path)

        wal.log_tx_abort("TX-001", reason="Conflict detected")

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]['op'] == 'TX_ABORT'
        assert entries[0]['data']['reason'] == "Conflict detected"

    def test_replay_returns_entries_in_order(self, tmp_path):
        """Verify replay returns entries in sequence order."""
        wal = WALManager(tmp_path)

        # Log multiple entries
        wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})
        wal.log("TX-001", "WRITE", {"entity_id": "task-1", "old_version": 1, "new_version": 2})
        wal.log("TX-001", "TX_COMMIT", {"version": 6})

        entries = wal.replay()
        assert len(entries) == 3

        # Verify sequence order
        assert entries[0]['seq'] == 1
        assert entries[1]['seq'] == 2
        assert entries[2]['seq'] == 3

        # Verify operation order
        assert entries[0]['op'] == 'TX_BEGIN'
        assert entries[1]['op'] == 'WRITE'
        assert entries[2]['op'] == 'TX_COMMIT'

    def test_sequence_persists_across_restarts(self, tmp_path):
        """Verify sequence counter persists across WALManager restarts."""
        # Create first WAL instance
        wal1 = WALManager(tmp_path)
        seq1 = wal1.log("TX-001", "TX_BEGIN", {"snapshot": 5})
        assert seq1 == 1

        # Create second WAL instance (simulates restart)
        wal2 = WALManager(tmp_path)
        seq2 = wal2.log("TX-002", "TX_BEGIN", {"snapshot": 6})
        assert seq2 == 2

        # Verify sequence continues from previous value
        assert seq2 > seq1

    def test_truncate_with_no_archive_deletes_wal(self, tmp_path):
        """Verify truncate without archive deletes WAL file."""
        wal = WALManager(tmp_path)

        # Log some entries
        wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})

        # Truncate without archive
        result = wal.truncate(archive=False)

        # Verify current.wal is deleted
        wal_file = tmp_path / "current.wal"
        assert not wal_file.exists()
        assert result is None

    def test_replay_skips_empty_lines(self, tmp_path):
        """Verify replay skips empty lines in WAL file."""
        wal = WALManager(tmp_path)

        # Log entry
        wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})

        # Manually add empty lines
        wal_file = tmp_path / "current.wal"
        with open(wal_file, 'a') as f:
            f.write('\n\n')

        # Replay should skip empty lines
        entries = wal.replay()
        assert len(entries) == 1

    def test_replay_skips_invalid_json(self, tmp_path):
        """Verify replay skips lines with invalid JSON."""
        wal = WALManager(tmp_path)

        # Log valid entry
        wal.log("TX-001", "TX_BEGIN", {"snapshot": 5})

        # Manually add invalid JSON
        wal_file = tmp_path / "current.wal"
        with open(wal_file, 'a') as f:
            f.write('INVALID JSON LINE\n')

        # Replay should skip invalid JSON
        entries = wal.replay()
        assert len(entries) == 1

    def test_log_tx_prepare_records_empty_data(self, tmp_path):
        """Verify TX_PREPARE records empty data dict."""
        wal = WALManager(tmp_path)

        wal.log_tx_prepare("TX-001")

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]['op'] == 'TX_PREPARE'
        assert entries[0]['data'] == {}

    def test_log_tx_rollback_records_reason(self, tmp_path):
        """Verify TX_ROLLBACK records reason."""
        wal = WALManager(tmp_path)

        wal.log_tx_rollback("TX-001", reason="Manual rollback")

        entries = wal.replay()
        assert len(entries) == 1
        assert entries[0]['op'] == 'TX_ROLLBACK'
        assert entries[0]['data']['reason'] == "Manual rollback"

    def test_replay_returns_empty_list_for_missing_wal(self, tmp_path):
        """Verify replay returns empty list when WAL file doesn't exist."""
        wal = WALManager(tmp_path)

        # Don't log anything, WAL file won't exist
        entries = wal.replay()
        assert entries == []

    def test_truncate_returns_none_for_missing_wal(self, tmp_path):
        """Verify truncate returns None when WAL file doesn't exist."""
        wal = WALManager(tmp_path)

        # Don't log anything, WAL file won't exist
        result = wal.truncate(archive=True)
        assert result is None

    def test_transaction_state_transitions(self, tmp_path):
        """Verify correct state tracking through transaction lifecycle."""
        wal = WALManager(tmp_path)

        # Begin transaction
        wal.log_tx_begin("TX-001", snapshot_version=5)
        incomplete = wal.get_incomplete_transactions()
        assert len(incomplete) == 1
        assert incomplete[0]['state'] == 'ACTIVE'

        # Prepare transaction
        wal.log_tx_prepare("TX-001")
        incomplete = wal.get_incomplete_transactions()
        assert len(incomplete) == 1
        assert incomplete[0]['state'] == 'PREPARING'

        # Commit transaction
        wal.log_tx_commit("TX-001", version=6)
        incomplete = wal.get_incomplete_transactions()
        assert len(incomplete) == 0

    def test_multiple_transactions_interleaved(self, tmp_path):
        """Verify correct tracking of multiple interleaved transactions."""
        wal = WALManager(tmp_path)

        # Start TX-001
        wal.log_tx_begin("TX-001", snapshot_version=5)

        # Start TX-002
        wal.log_tx_begin("TX-002", snapshot_version=5)

        # Commit TX-001
        wal.log_tx_commit("TX-001", version=6)

        # Check incomplete transactions
        incomplete = wal.get_incomplete_transactions()
        assert len(incomplete) == 1
        assert incomplete[0]['tx_id'] == 'TX-002'

        # Abort TX-002
        wal.log_tx_abort("TX-002", reason="Conflict")

        # No incomplete transactions
        incomplete = wal.get_incomplete_transactions()
        assert len(incomplete) == 0
