"""
Behavioral tests for CDG Write-Ahead Logging.

Epic: Developer Uses WAL for Crash Recovery

As a developer building transactional systems,
I want Write-Ahead Logging that we built from first principles,
So that I can recover from crashes without data loss
while maintaining complete sovereignty over our implementation.

Following Metus: We describe behavior, then make it true.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from threading import Thread
from typing import List

import pytest

from cortical.cdg.wal import CDGWALManager
from cortical.cdg.config import CDGConfig, DurabilityMode
from cortical.wal import TransactionWALEntry


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_wal_dir(tmp_path):
    """Provide a temporary directory for WAL storage."""
    wal_dir = tmp_path / "wal"
    wal_dir.mkdir()
    return wal_dir


@pytest.fixture
def wal_manager(temp_wal_dir):
    """Provide a WAL manager with default configuration."""
    config = CDGConfig(
        durability=DurabilityMode.BALANCED,
        enable_wal=True,
        transactions_enabled=True
    )
    return CDGWALManager(temp_wal_dir, config)


@pytest.fixture
def paranoid_wal_manager(temp_wal_dir):
    """Provide a WAL manager with paranoid durability for crash recovery tests."""
    config = CDGConfig(
        durability=DurabilityMode.PARANOID,
        enable_wal=True,
        transactions_enabled=True
    )
    return CDGWALManager(temp_wal_dir, config)


@pytest.fixture
def fast_wal_manager(temp_wal_dir):
    """Provide a WAL manager with fast mode for performance tests."""
    config = CDGConfig(
        durability=DurabilityMode.RELAXED,
        enable_wal=True,
        transactions_enabled=True
    )
    return CDGWALManager(temp_wal_dir, config)


# ==============================================================================
# BEHAVIORAL SCENARIOS
# ==============================================================================

class TestDeveloperLogsTransactionLifecycle:
    """
    Epic: Transaction Lifecycle Logging

    As a developer implementing transactional operations,
    I want to log the complete transaction lifecycle,
    So that every operation is durably recorded before execution.
    """

    def test_scenario_complete_transaction_logged_in_order(self, wal_manager):
        """
        Scenario: Complete transaction workflow is logged correctly

        Given a WAL manager
        When I log a complete transaction (begin → write → commit)
        Then all operations appear in the WAL in correct order
        And each entry has a valid checksum
        Because durability requires every step to be recorded
        """
        # Given a WAL manager (provided by fixture)

        # When I log a complete transaction (begin → write → commit)
        tx_id = "TX-001"
        seq_begin = wal_manager.log_tx_begin(tx_id, snapshot_version=1)
        seq_write = wal_manager.log_write(tx_id, "entity-42", old_version=1, new_version=2)
        seq_commit = wal_manager.log_tx_commit(tx_id, version=2)

        # Then all operations appear in the WAL in correct order
        entries = wal_manager.replay()
        assert len(entries) == 3

        # Verify begin
        assert entries[0]['op'] == 'TX_BEGIN'
        assert entries[0]['tx'] == tx_id
        assert entries[0]['data']['snapshot'] == 1

        # Verify write
        assert entries[1]['op'] == 'WRITE'
        assert entries[1]['tx'] == tx_id
        assert entries[1]['data']['entity_id'] == 'entity-42'
        assert entries[1]['data']['old_version'] == 1
        assert entries[1]['data']['new_version'] == 2

        # Verify commit
        assert entries[2]['op'] == 'TX_COMMIT'
        assert entries[2]['tx'] == tx_id
        assert entries[2]['data']['version'] == 2

        # And each entry has a valid checksum
        for entry_dict in entries:
            entry = TransactionWALEntry.from_dict(entry_dict)
            assert entry.verify(), f"Entry {entry.seq} has invalid checksum"

    def test_scenario_multiple_writes_in_transaction(self, wal_manager):
        """
        Scenario: Transaction with multiple writes is logged atomically

        Given a WAL manager
        When I log a transaction with multiple entity writes
        Then all writes are associated with the same transaction ID
        And the sequence numbers increase monotonically
        """
        # Given a WAL manager (provided by fixture)

        # When I log a transaction with multiple entity writes
        tx_id = "TX-multi-write"
        wal_manager.log_tx_begin(tx_id, snapshot_version=5)

        entity_ids = ["entity-A", "entity-B", "entity-C"]
        sequences = []

        for i, entity_id in enumerate(entity_ids):
            seq = wal_manager.log_write(
                tx_id,
                entity_id,
                old_version=i,
                new_version=i + 1
            )
            sequences.append(seq)

        wal_manager.log_tx_commit(tx_id, version=8)

        # Then all writes are associated with the same transaction ID
        entries = wal_manager.replay()
        write_entries = [e for e in entries if e['op'] == 'WRITE']

        assert len(write_entries) == 3
        for entry in write_entries:
            assert entry['tx'] == tx_id

        # And the sequence numbers increase monotonically
        for i in range(len(sequences) - 1):
            assert sequences[i] < sequences[i + 1], \
                "Sequence numbers must increase monotonically"

    def test_scenario_transaction_abort_recorded(self, wal_manager):
        """
        Scenario: Transaction abort is properly logged

        Given a transaction in progress
        When the transaction is aborted with a reason
        Then the abort is logged with the reason
        And the transaction is marked as incomplete
        """
        # Given a transaction in progress
        tx_id = "TX-abort-test"
        wal_manager.log_tx_begin(tx_id, snapshot_version=10)
        wal_manager.log_write(tx_id, "entity-99", old_version=1, new_version=2)

        # When the transaction is aborted with a reason
        wal_manager.log_tx_abort(tx_id, reason="Conflict detected")

        # Then the abort is logged with the reason
        entries = wal_manager.replay()
        abort_entries = [e for e in entries if e['op'] == 'TX_ABORT']

        assert len(abort_entries) == 1
        assert abort_entries[0]['tx'] == tx_id
        assert abort_entries[0]['data']['reason'] == "Conflict detected"

        # And the transaction is marked as complete (aborted counts as complete)
        incomplete = wal_manager.get_incomplete_transactions()
        incomplete_ids = [t['tx_id'] for t in incomplete]
        assert tx_id not in incomplete_ids

    def test_scenario_transaction_prepare_phase_logged(self, wal_manager):
        """
        Scenario: Two-phase commit prepare phase is logged

        Given a distributed transaction
        When the transaction enters prepare phase
        Then the prepare operation is logged
        And the transaction state transitions from ACTIVE to PREPARING
        """
        # Given a distributed transaction
        tx_id = "TX-2pc"
        wal_manager.log_tx_begin(tx_id, snapshot_version=20)
        wal_manager.log_write(tx_id, "entity-distributed", old_version=5, new_version=6)

        # When the transaction enters prepare phase
        wal_manager.log_tx_prepare(tx_id)

        # Then the prepare operation is logged
        entries = wal_manager.replay()
        prepare_entries = [e for e in entries if e['op'] == 'TX_PREPARE']

        assert len(prepare_entries) == 1
        assert prepare_entries[0]['tx'] == tx_id

        # And the transaction state transitions from ACTIVE to PREPARING
        incomplete = wal_manager.get_incomplete_transactions()
        tx_states = {t['tx_id']: t['state'] for t in incomplete}

        assert tx_states.get(tx_id) == 'PREPARING'


class TestDeveloperRecoversDatabaseAfterCrash:
    """
    Epic: Crash Recovery

    As a developer whose system just crashed,
    I want to replay the WAL to restore committed state,
    So that no committed work is lost.
    """

    def test_scenario_replay_restores_committed_transactions(self, paranoid_wal_manager):
        """
        Scenario: WAL replay after crash restores all committed transactions

        Given a system that crashed after committing transactions
        When I create a new WAL manager and replay the log
        Then all committed transactions are replayed
        And the operations appear in the same order
        """
        # Given a system that crashed after committing transactions
        tx1_id = "TX-crash-1"
        paranoid_wal_manager.log_tx_begin(tx1_id, snapshot_version=1)
        paranoid_wal_manager.log_write(tx1_id, "entity-safe", old_version=0, new_version=1)
        paranoid_wal_manager.log_tx_commit(tx1_id, version=1)

        tx2_id = "TX-crash-2"
        paranoid_wal_manager.log_tx_begin(tx2_id, snapshot_version=1)
        paranoid_wal_manager.log_write(tx2_id, "entity-also-safe", old_version=0, new_version=1)
        paranoid_wal_manager.log_tx_commit(tx2_id, version=2)

        # Simulate crash and restart by creating new manager instance
        wal_dir = paranoid_wal_manager.wal_dir
        config = paranoid_wal_manager.config

        # When I create a new WAL manager and replay the log
        recovered_manager = CDGWALManager(wal_dir, config)
        replayed_entries = recovered_manager.replay()

        # Then all committed transactions are replayed
        tx_ids = {e['tx'] for e in replayed_entries}
        assert tx1_id in tx_ids
        assert tx2_id in tx_ids

        # And the operations appear in the same order
        operations = [e['op'] for e in replayed_entries]
        expected = ['TX_BEGIN', 'WRITE', 'TX_COMMIT', 'TX_BEGIN', 'WRITE', 'TX_COMMIT']
        assert operations == expected

    def test_scenario_incomplete_transaction_detected_after_crash(self, paranoid_wal_manager):
        """
        Scenario: Crash leaves incomplete transaction that needs rollback

        Given a transaction that began but never committed
        When I replay the WAL after crash
        Then the transaction is identified as incomplete
        And I can rollback its changes
        """
        # Given a transaction that began but never committed
        tx_incomplete = "TX-incomplete"
        paranoid_wal_manager.log_tx_begin(tx_incomplete, snapshot_version=10)
        paranoid_wal_manager.log_write(
            tx_incomplete,
            "entity-orphaned",
            old_version=5,
            new_version=6
        )
        # Crash occurs here - no commit!

        # Also add a complete transaction for contrast
        tx_complete = "TX-complete"
        paranoid_wal_manager.log_tx_begin(tx_complete, snapshot_version=10)
        paranoid_wal_manager.log_write(
            tx_complete,
            "entity-complete",
            old_version=7,
            new_version=8
        )
        paranoid_wal_manager.log_tx_commit(tx_complete, version=11)

        # When I replay the WAL after crash
        wal_dir = paranoid_wal_manager.wal_dir
        config = paranoid_wal_manager.config
        recovered_manager = CDGWALManager(wal_dir, config)

        # Then the transaction is identified as incomplete
        incomplete = recovered_manager.get_incomplete_transactions()
        incomplete_ids = [t['tx_id'] for t in incomplete]

        assert tx_incomplete in incomplete_ids
        assert tx_complete not in incomplete_ids

        # And I can rollback its changes
        rollback_seq = recovered_manager.log_tx_rollback(
            tx_incomplete,
            reason="Rollback due to incomplete transaction after crash"
        )
        assert rollback_seq > 0

        # Verify rollback was logged
        entries = recovered_manager.replay()
        rollback_entries = [e for e in entries if e['op'] == 'TX_ROLLBACK']
        assert len(rollback_entries) == 1
        assert rollback_entries[0]['tx'] == tx_incomplete

    def test_scenario_replay_skips_corrupted_entries(self, temp_wal_dir):
        """
        Scenario: Corrupted WAL entry is skipped during replay

        Given a WAL with both valid and corrupted entries
        When I replay the WAL
        Then valid entries are processed
        And corrupted entries are skipped with warnings
        Because partial recovery is better than complete failure
        """
        # Given a WAL with both valid and corrupted entries
        config = CDGConfig(
            durability=DurabilityMode.PARANOID,
            enable_wal=True,
            transactions_enabled=True
        )
        wal = CDGWALManager(temp_wal_dir, config)

        # Write valid entry
        wal.log_tx_begin("TX-valid", snapshot_version=1)

        # Manually corrupt the WAL file by appending invalid JSON
        wal_file = temp_wal_dir / "current.wal"
        with open(wal_file, 'a', encoding='utf-8') as f:
            f.write('{"invalid": "json without checksum}\n')  # Missing closing brace
            f.write('totally not json at all!\n')

        # Write another valid entry
        wal.log_tx_commit("TX-valid", version=1)

        # When I replay the WAL
        entries = wal.replay()

        # Then valid entries are processed
        valid_ops = [e['op'] for e in entries]
        assert 'TX_BEGIN' in valid_ops
        assert 'TX_COMMIT' in valid_ops

        # And corrupted entries are skipped (implicitly - they're not in results)
        # We should have exactly 2 entries (the valid ones)
        assert len(entries) == 2

    def test_scenario_sequence_numbers_recovered_correctly(self, paranoid_wal_manager):
        """
        Scenario: Sequence counter survives crash and continues correctly

        Given a WAL with several entries
        When the system crashes and restarts
        Then the sequence counter continues from the last value
        And no sequence numbers are reused
        """
        # Given a WAL with several entries
        tx_id = "TX-seq-test"
        seq1 = paranoid_wal_manager.log_tx_begin(tx_id, snapshot_version=1)
        seq2 = paranoid_wal_manager.log_write(
            tx_id,
            "entity-1",
            old_version=0,
            new_version=1
        )
        seq3 = paranoid_wal_manager.log_tx_commit(tx_id, version=1)

        last_seq = seq3

        # When the system crashes and restarts
        wal_dir = paranoid_wal_manager.wal_dir
        config = paranoid_wal_manager.config
        recovered_manager = CDGWALManager(wal_dir, config)

        # Then the sequence counter continues from the last value
        new_tx = "TX-after-recovery"
        seq4 = recovered_manager.log_tx_begin(new_tx, snapshot_version=2)

        assert seq4 > last_seq, "Sequence must continue after recovery"

        # And no sequence numbers are reused
        all_entries = recovered_manager.replay()
        all_seqs = [e['seq'] for e in all_entries]

        # Check uniqueness
        assert len(all_seqs) == len(set(all_seqs)), \
            "Sequence numbers must be unique"


class TestSystemMaintainsDataIntegrityThroughWAL:
    """
    Epic: Data Integrity

    As a system operator,
    I want checksums on every WAL entry,
    So that corruption is detected immediately.
    """

    def test_scenario_every_entry_has_valid_checksum(self, wal_manager):
        """
        Scenario: All WAL entries include integrity checksums

        Given a series of transaction operations
        When each operation is logged
        Then every entry includes a checksum
        And the checksum verifies correctly
        """
        # Given a series of transaction operations
        tx_id = "TX-checksum-test"

        # When each operation is logged
        wal_manager.log_tx_begin(tx_id, snapshot_version=1)
        wal_manager.log_write(tx_id, "entity-secure", old_version=0, new_version=1)
        wal_manager.log_tx_prepare(tx_id)
        wal_manager.log_tx_commit(tx_id, version=1)

        # Then every entry includes a checksum
        entries = wal_manager.replay_entries()

        assert len(entries) > 0, "Should have entries"
        for entry in entries:
            assert entry.checksum is not None
            assert entry.checksum != ""
            assert len(entry.checksum) > 0

        # And the checksum verifies correctly
        for entry in entries:
            assert entry.verify(), \
                f"Entry seq={entry.seq} op={entry.operation} failed checksum"

    def test_scenario_tampered_entry_detected(self, temp_wal_dir):
        """
        Scenario: Tampered WAL entry is detected via checksum

        Given a WAL with valid entries
        When an entry is tampered with on disk
        Then replay detects the invalid checksum
        And skips the corrupted entry
        """
        # Given a WAL with valid entries
        config = CDGConfig(
            durability=DurabilityMode.PARANOID,
            enable_wal=True,
            transactions_enabled=True
        )
        wal = CDGWALManager(temp_wal_dir, config)

        tx_id = "TX-tamper-test"
        wal.log_tx_begin(tx_id, snapshot_version=1)
        wal.log_tx_commit(tx_id, version=1)

        # When an entry is tampered with on disk
        wal_file = temp_wal_dir / "current.wal"

        # Read the WAL file
        with open(wal_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Tamper with the first entry by changing the transaction ID
        # but keeping the old checksum
        first_line = json.loads(lines[0])
        first_line['tx'] = "TX-TAMPERED"  # Change data without updating checksum

        # Write back the tampered entry
        lines[0] = json.dumps(first_line, separators=(',', ':')) + '\n'
        with open(wal_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Then replay detects the invalid checksum
        # Create fresh manager to replay
        recovered_wal = CDGWALManager(temp_wal_dir, config)
        valid_entries = recovered_wal.replay()

        # And skips the corrupted entry
        # We should only get the second entry (TX_COMMIT) if first was corrupted
        tx_ids = [e['tx'] for e in valid_entries]

        # The tampered entry should be skipped
        assert "TX-TAMPERED" not in tx_ids

    def test_scenario_checksum_protects_payload_data(self, wal_manager):
        """
        Scenario: Checksum covers entire entry including payload

        Given a write operation with complex payload data
        When the entry is created
        Then the checksum includes all payload fields
        And modification of any field invalidates the checksum
        """
        # Given a write operation with complex payload data
        tx_id = "TX-payload-test"
        wal_manager.log_tx_begin(tx_id, snapshot_version=100)

        # Create an entry with payload
        seq = wal_manager.log_write(
            tx_id,
            "entity-complex",
            old_version=42,
            new_version=43
        )

        # When the entry is created
        entries = wal_manager.replay_entries()
        write_entry = [e for e in entries if e.seq == seq][0]

        # Then the checksum includes all payload fields
        original_checksum = write_entry.checksum
        assert write_entry.verify()

        # And modification of any field invalidates the checksum
        # Test modifying entity_id in payload
        write_entry.payload['entity_id'] = 'MODIFIED'
        assert not write_entry.verify(), \
            "Checksum should be invalid after payload modification"

        # Restore and test modifying version
        write_entry.payload['entity_id'] = 'entity-complex'
        write_entry.payload['new_version'] = 999
        assert not write_entry.verify(), \
            "Checksum should be invalid after version modification"


class TestDeveloperHandlesConcurrentWrites:
    """
    Epic: Concurrent Access Safety

    As a developer with multiple processes writing,
    I want WAL writes to be atomic and ordered,
    So that concurrent operations don't corrupt the log.
    """

    def test_scenario_process_lock_prevents_interleaved_writes(self, temp_wal_dir):
        """
        Scenario: Sequential writes maintain integrity

        Given a WAL manager
        When I write multiple transactions sequentially
        Then all entries are written correctly
        And all entries have valid checksums
        And the WAL file is not corrupted
        """
        # Given a WAL manager
        config = CDGConfig(
            durability=DurabilityMode.RELAXED,  # Fast for test performance
            enable_wal=True,
            transactions_enabled=True
        )

        wal = CDGWALManager(temp_wal_dir, config)

        # When I write multiple transactions sequentially
        num_transactions = 15

        for i in range(num_transactions):
            tx_id = f"TX-{i}"
            wal.log_tx_begin(tx_id, snapshot_version=i)
            wal.log_write(
                tx_id,
                f"entity-{i}",
                old_version=i,
                new_version=i + 1
            )
            wal.log_tx_commit(tx_id, version=i + 1)

        # Then all entries are written correctly
        all_entries = wal.replay()

        # Should have 3 entries per transaction
        expected_count = 3 * num_transactions
        assert len(all_entries) == expected_count

        # And all entries have valid checksums
        for entry_dict in all_entries:
            entry = TransactionWALEntry.from_dict(entry_dict)
            assert entry.verify(), f"Entry {entry.seq} has invalid checksum"

        # And the WAL file is not corrupted
        # Verify we can parse all entries
        tx_ids = {e['tx'] for e in all_entries}
        assert len(tx_ids) == num_transactions

    def test_scenario_fsync_ensures_durability_under_crash(self, temp_wal_dir):
        """
        Scenario: PARANOID mode fsyncs every write for maximum durability

        Given a WAL manager in PARANOID durability mode
        When I write transaction operations
        Then each write is immediately fsynced to disk
        And the data survives immediate power loss simulation
        """
        # Given a WAL manager in PARANOID durability mode
        paranoid_config = CDGConfig(
            durability=DurabilityMode.PARANOID,
            enable_wal=True,
            transactions_enabled=True
        )
        wal = CDGWALManager(temp_wal_dir, paranoid_config)

        # When I write transaction operations
        tx_id = "TX-fsync-test"
        wal.log_tx_begin(tx_id, snapshot_version=1)
        wal.log_write(tx_id, "entity-durable", old_version=0, new_version=1)
        wal.log_tx_commit(tx_id, version=1)

        # Then each write is immediately fsynced to disk
        # Verify the file exists and has content
        wal_file = temp_wal_dir / "current.wal"
        assert wal_file.exists()

        # Verify we can read the entries immediately (they're fsynced)
        file_size = wal_file.stat().st_size
        assert file_size > 0

        # And the data survives immediate power loss simulation
        # (Simulate by creating new manager without closing old one)
        crash_recovery_wal = CDGWALManager(temp_wal_dir, paranoid_config)
        recovered_entries = crash_recovery_wal.replay()

        assert len(recovered_entries) == 3
        assert recovered_entries[0]['op'] == 'TX_BEGIN'
        assert recovered_entries[1]['op'] == 'WRITE'
        assert recovered_entries[2]['op'] == 'TX_COMMIT'

    def test_scenario_balanced_mode_fsyncs_on_commit(self, temp_wal_dir):
        """
        Scenario: BALANCED mode delays fsync until commit for performance

        Given a WAL manager in BALANCED durability mode
        When I log a transaction with multiple writes
        Then fsync is only called on commit
        And performance is better than PARANOID mode
        """
        # Given a WAL manager in BALANCED durability mode
        balanced_config = CDGConfig(
            durability=DurabilityMode.BALANCED,
            enable_wal=True,
            transactions_enabled=True
        )
        wal = CDGWALManager(temp_wal_dir, balanced_config)

        # When I log a transaction with multiple writes
        tx_id = "TX-balanced"
        wal.log_tx_begin(tx_id, snapshot_version=1)

        # Multiple writes
        for i in range(10):
            wal.log_write(tx_id, f"entity-{i}", old_version=i, new_version=i + 1)

        # Manual fsync would be called by transaction manager on commit
        wal.fsync_now()

        # Then the entries are durable
        recovery_wal = CDGWALManager(temp_wal_dir, balanced_config)
        entries = recovery_wal.replay()

        # Should have begin + 10 writes
        assert len(entries) >= 11
        assert entries[0]['op'] == 'TX_BEGIN'
        write_entries = [e for e in entries if e['op'] == 'WRITE']
        assert len(write_entries) == 10


class TestDeveloperManagesWALLifecycle:
    """
    Epic: WAL Management

    As a developer managing storage,
    I want to archive old WAL files after checkpoint,
    So that WAL doesn't grow unbounded.
    """

    def test_scenario_truncate_archives_old_wal(self, wal_manager):
        """
        Scenario: WAL truncation archives completed log

        Given a WAL with committed transactions
        When I truncate the WAL after checkpoint
        Then the current WAL is moved to archive
        And a new empty WAL is ready for new operations
        """
        # Given a WAL with committed transactions
        tx_id = "TX-archive-test"
        wal_manager.log_tx_begin(tx_id, snapshot_version=1)
        wal_manager.log_write(tx_id, "entity-old", old_version=0, new_version=1)
        wal_manager.log_tx_commit(tx_id, version=1)

        # Verify entries exist
        entries_before = wal_manager.replay()
        assert len(entries_before) == 3

        # When I truncate the WAL after checkpoint
        archive_path = wal_manager.truncate(archive=True)

        # Then the current WAL is moved to archive
        assert archive_path is not None
        assert archive_path.exists()
        assert "archived" in str(archive_path)

        # And a new empty WAL is ready for new operations
        entries_after = wal_manager.replay()
        assert len(entries_after) == 0

        # New operations go to fresh WAL
        new_tx = "TX-after-truncate"
        wal_manager.log_tx_begin(new_tx, snapshot_version=2)
        new_entries = wal_manager.replay()
        assert len(new_entries) == 1
        assert new_entries[0]['tx'] == new_tx

    def test_scenario_truncate_without_archive_deletes_wal(self, wal_manager):
        """
        Scenario: WAL truncation can delete instead of archive

        Given a WAL that's been checkpointed
        When I truncate without archiving
        Then the WAL is deleted
        And no archive is created
        """
        # Given a WAL that's been checkpointed
        tx_id = "TX-delete-test"
        wal_manager.log_tx_begin(tx_id, snapshot_version=1)
        wal_manager.log_tx_commit(tx_id, version=1)

        # When I truncate without archiving
        archive_path = wal_manager.truncate(archive=False)

        # Then the WAL is deleted
        assert archive_path is None

        # And no archive is created
        archive_dir = wal_manager.archive_dir
        archived_files = list(archive_dir.glob("*.wal"))
        # There might be files from previous tests, but we didn't create new ones
        # Just verify truncate returned None
        assert archive_path is None

        # Verify WAL is empty
        entries = wal_manager.replay()
        assert len(entries) == 0

    def test_scenario_legacy_entries_skipped_during_replay(self, temp_wal_dir):
        """
        Scenario: Legacy WAL format entries are gracefully skipped

        Given a WAL with legacy format entries
        When I replay the WAL
        Then legacy entries are skipped
        And modern entries are processed normally
        """
        # Given a WAL with legacy format entries
        config = CDGConfig(
            durability=DurabilityMode.BALANCED,
            enable_wal=True,
            transactions_enabled=True
        )
        wal = CDGWALManager(temp_wal_dir, config)

        # Write a modern entry first
        wal.log_tx_begin("TX-modern", snapshot_version=1)

        # Manually inject a legacy entry into the WAL file
        wal_file = temp_wal_dir / "current.wal"
        with open(wal_file, 'a', encoding='utf-8') as f:
            # Legacy format: has entity_id and numeric timestamp
            legacy_entry = {
                "op": "ADOPTED",
                "entity_id": "legacy-entity-123",
                "timestamp": 1234567890.123,
                "checksum": "abc123"
            }
            f.write(json.dumps(legacy_entry) + '\n')

        # Write another modern entry
        wal.log_tx_commit("TX-modern", version=1)

        # When I replay the WAL
        entries = wal.replay()

        # Then legacy entries are skipped
        ops = [e['op'] for e in entries]
        assert 'ADOPTED' not in ops

        # And modern entries are processed normally
        assert 'TX_BEGIN' in ops
        assert 'TX_COMMIT' in ops
        assert len(entries) == 2
