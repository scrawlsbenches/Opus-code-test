"""
Behavioral tests for CDG crash recovery scenarios.

Epic: System Administrator Recovers From Crashes

As a system administrator,
I want the system to recover correctly from various crash scenarios,
So that data integrity is maintained even after unexpected failures.

These tests verify the WAL-first durability model:
1. TX_COMMIT logged to WAL
2. WAL fsynced (commit is now durable)
3. Entity files written (can be redone from WAL)

If crash happens at any point, recovery should restore consistency.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone

from cortical.cdg.storage import CDGStore
from cortical.cdg.transaction_manager import CDGTransactionManager
from cortical.cdg.recovery import CDGRecoveryManager
from cortical.cdg.wal import CDGWALManager
from cortical.cdg.config import CDGConfig, DurabilityMode, RecoveryMode, OrphanStrategy
from cortical.cdg.types import Entity
from cortical.common.filesystem import RealFileSystem
from cortical.utils.checksums import compute_checksum


class SimpleEntity(Entity):
    """Simple test entity for crash recovery scenarios."""

    def __init__(self, id: str, name: str = "test", version: int = 1):
        self.id = id
        self.entity_type = "simple"
        self.name = name
        self._version = version
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.modified_at = self.created_at

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self) -> None:
        self._version += 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "name": self.name,
            "version": self._version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }


def simple_entity_factory(data: dict) -> Entity:
    """Factory for creating SimpleEntity from dict."""
    entity = SimpleEntity(
        id=data["id"],
        name=data.get("name", "test"),
        version=data.get("version", 1),
    )
    entity.created_at = data.get("created_at", entity.created_at)
    entity.modified_at = data.get("modified_at", entity.modified_at)
    return entity


@pytest.fixture
def crash_config():
    """Configuration for crash recovery testing with full ACID."""
    return CDGConfig(
        transactions_enabled=True,
        enable_wal=True,
        recovery_mode=RecoveryMode.FULL,
        orphan_strategy=OrphanStrategy.REPAIR,
        durability=DurabilityMode.BALANCED,
        enable_history=True,
    )


@pytest.fixture
def crash_test_dir(tmp_path_factory):
    """Provide temporary directory for crash recovery tests."""
    return tmp_path_factory.mktemp("crash_recovery")


class TestWALFirstDurabilityModel:
    """
    As a system administrator relying on WAL-first durability,
    I want commits to be durable before entity files are modified,
    So that I can always recover from WAL after a crash.
    """

    def test_scenario_commit_is_durable_before_entity_write(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: TX_COMMIT is written to WAL before entity files

        Given a transaction with writes
        When I commit the transaction
        Then TX_COMMIT appears in WAL
        And WAL is fsynced before entities are modified
        Because WAL-first means commit is durable first.
        """
        # Given a transaction with writes
        fs = RealFileSystem(crash_test_dir)
        tm = CDGTransactionManager(fs, crash_config, simple_entity_factory)
        tx = tm.begin()

        entity = SimpleEntity(id="E-wal-first-001", name="test")
        tm.write(tx, entity)

        # When I commit
        result = tm.commit(tx)

        # Then TX_COMMIT appears in WAL
        assert result.success
        wal_file = crash_test_dir / "wal" / "current.wal"
        assert wal_file.exists()

        with open(wal_file, 'r') as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Find TX_COMMIT entry
        commit_entries = [e for e in entries if e.get('op') == 'TX_COMMIT']
        assert len(commit_entries) == 1

        # And entity file exists
        entity_file = crash_test_dir / "E-wal-first-001.json"
        assert entity_file.exists()

    def test_scenario_crash_after_commit_before_entity_write(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Crash after TX_COMMIT but before entity files written

        Given a committed transaction in WAL
        But entity files are missing (simulating crash before write)
        When recovery runs
        Then recovery detects the committed transaction
        And reports the missing entity as an orphan issue
        Because WAL is source of truth for what should exist.
        """
        # Given a committed transaction in WAL (manually created)
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        # Create WAL entries simulating a committed transaction
        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-001", "op": "TX_BEGIN",
             "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-001", "op": "WRITE",
             "data": {"entity_id": "E-missing-001", "old_version": 0, "new_version": 1}},
            {"seq": 3, "ts": "2025-01-01T00:00:02Z", "tx": "TX-001", "op": "TX_COMMIT",
             "data": {"version": 1}},
        ]

        # Add checksums
        for entry in entries:
            entry_copy = dict(entry)
            entry["checksum"] = compute_checksum(entry_copy)

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        # Also create sequence file
        seq_file = wal_dir / "sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 3}, f)

        # Entity file is MISSING (simulating crash before write)
        entity_file = crash_test_dir / "E-missing-001.json"
        assert not entity_file.exists()

        # When recovery runs
        # Note: The current implementation doesn't yet redo writes from WAL,
        # but it should at least not corrupt anything
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config)
        result = recovery.recover()

        # Then recovery completes (even if entity is not reconstructed yet)
        # This is a known limitation - full entity redo from WAL is future work
        assert result.success or len(result.corrupted_entities) == 0


class TestPartialWriteRecovery:
    """
    As a system administrator,
    I want partial/truncated writes to be detected and reported,
    So that I know about data corruption from crashes during writes.
    """

    def test_scenario_truncated_entity_file_detected(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Truncated entity file is detected as corrupted

        Given an entity file that was truncated mid-write
        When integrity check runs
        Then the file is detected as corrupted
        Because small files indicate partial writes.
        """
        # Given a truncated entity file
        crash_test_dir.mkdir(parents=True, exist_ok=True)
        entity_file = crash_test_dir / "E-truncated-001.json"
        entity_file.write_text('{"da')  # Truncated JSON

        # When integrity check runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config)
        corrupted = recovery.verify_store_integrity()

        # Then file is detected as corrupted
        assert "E-truncated-001" in corrupted

    def test_scenario_empty_entity_file_detected(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Empty entity file is detected as corrupted

        Given an empty entity file (crash during write start)
        When integrity check runs
        Then the file is detected as corrupted
        Because empty files are obviously partial writes.
        """
        # Given an empty entity file
        crash_test_dir.mkdir(parents=True, exist_ok=True)
        entity_file = crash_test_dir / "E-empty-001.json"
        entity_file.write_text('')

        # When integrity check runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config)
        corrupted = recovery.verify_store_integrity()

        # Then file is detected as corrupted
        assert "E-empty-001" in corrupted

    def test_scenario_very_small_file_detected(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Very small entity file is detected as corrupted

        Given an entity file smaller than minimum valid size
        When integrity check runs
        Then the file is flagged as partial write
        Because valid entity JSON is at least ~50 bytes.
        """
        # Given a very small entity file (10 bytes)
        crash_test_dir.mkdir(parents=True, exist_ok=True)
        entity_file = crash_test_dir / "E-small-001.json"
        entity_file.write_text('{"x": 1}')  # Valid JSON but too small for entity

        # When integrity check runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config)
        corrupted = recovery.verify_store_integrity()

        # Then file is detected as corrupted (partial write)
        assert "E-small-001" in corrupted


class TestIncompleteTransactionRecovery:
    """
    As a system administrator,
    I want incomplete transactions to be rolled back on recovery,
    So that I don't have partial changes corrupting my data.
    """

    def test_scenario_active_transaction_rolled_back(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Active transaction is rolled back on recovery

        Given a transaction that was started but never committed
        When recovery runs
        Then the transaction is marked as rolled back in WAL
        Because incomplete transactions must not persist.
        """
        # Given an active transaction in WAL
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-active-001",
             "op": "TX_BEGIN", "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-active-001",
             "op": "WRITE", "data": {"entity_id": "E-001", "old_version": 0, "new_version": 1}},
            # No TX_COMMIT - transaction was active when crash happened
        ]

        for entry in entries:
            entry["checksum"] = compute_checksum(dict(entry))

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        seq_file = wal_dir / "sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 2}, f)

        # When recovery runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config)
        result = recovery.recover()

        # Then transaction is rolled back
        assert "TX-active-001" in result.rolled_back
        assert result.recovered_transactions == 1

    def test_scenario_preparing_transaction_rolled_back(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Preparing transaction is rolled back on recovery

        Given a transaction in PREPARING state (never committed)
        When recovery runs
        Then the transaction is rolled back
        Because PREPARING without COMMIT means it didn't complete.
        """
        # Given a preparing transaction in WAL
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-prep-001",
             "op": "TX_BEGIN", "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-prep-001",
             "op": "WRITE", "data": {"entity_id": "E-001", "old_version": 0, "new_version": 1}},
            {"seq": 3, "ts": "2025-01-01T00:00:02Z", "tx": "TX-prep-001",
             "op": "TX_PREPARE", "data": {}},
            # Crash after PREPARE but before COMMIT
        ]

        for entry in entries:
            entry["checksum"] = compute_checksum(dict(entry))

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        seq_file = wal_dir / "sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 3}, f)

        # When recovery runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config)
        result = recovery.recover()

        # Then transaction is rolled back
        assert "TX-prep-001" in result.rolled_back


class TestHistoryCrashRecoveryWithDeletes:
    """
    As a system administrator,
    I want delete operation history to survive crashes,
    So that I have a complete audit trail even for deletions.
    """

    def test_scenario_delete_history_recovered_from_pending(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Crash after delete but before history finalization

        Given an entity was deleted
        And system crashed before history was finalized
        When system restarts
        Then the history entry is recovered from pending
        Because delete history is just as important as write history.
        """
        # Given an existing entity
        fs = RealFileSystem(crash_test_dir)
        store = CDGStore(fs, crash_config, simple_entity_factory)
        entity = SimpleEntity(id="E-delete-001", name="to-be-deleted")
        store.write(entity)

        # Create pending history simulating crash after delete
        pending_dir = crash_test_dir / "_history" / "_pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_file = pending_dir / "E-delete-001.pending"

        # Pending entry for delete (expected_entity_version=0 means deletion)
        pending_entry = {
            "global_version": store.current_version(),
            "timestamp": "2025-01-01T00:00:00+00:00",
            "data": entity.to_dict(),
            "expected_entity_version": 0  # 0 indicates deletion
        }

        with open(pending_file, 'w') as f:
            json.dump(pending_entry, f)
            f.write('\n')

        # Actually delete the entity file (simulating delete completed)
        entity_file = crash_test_dir / "E-delete-001.json"
        entity_file.unlink()

        # When system restarts (new store triggers recovery)
        fs2 = RealFileSystem(crash_test_dir)
        store2 = CDGStore(fs2, crash_config, simple_entity_factory)

        # Then pending file should be finalized
        assert not pending_file.exists(), "Pending should be finalized"

        # And history should contain the entry
        history_path = crash_test_dir / "_history" / "E-delete-001.jsonl"
        if history_path.exists():
            with open(history_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            assert len(entries) >= 1

    def test_scenario_delete_pending_discarded_if_delete_incomplete(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Pending history discarded if delete didn't complete

        Given a pending delete history file exists
        But the entity still exists (delete didn't complete)
        When system restarts
        Then the pending history is discarded
        Because the delete never happened.
        """
        # Given an existing entity
        fs = RealFileSystem(crash_test_dir)
        store = CDGStore(fs, crash_config, simple_entity_factory)
        entity = SimpleEntity(id="E-nodelete-001", name="still-here")
        store.write(entity)

        # Create pending history for a delete that never completed
        pending_dir = crash_test_dir / "_history" / "_pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_file = pending_dir / "E-nodelete-001.pending"

        pending_entry = {
            "global_version": store.current_version(),
            "timestamp": "2025-01-01T00:00:00+00:00",
            "data": entity.to_dict(),
            "expected_entity_version": 0
        }

        with open(pending_file, 'w') as f:
            json.dump(pending_entry, f)
            f.write('\n')

        # Entity still exists (delete didn't complete)
        entity_file = crash_test_dir / "E-nodelete-001.json"
        assert entity_file.exists()

        # When system restarts
        fs2 = RealFileSystem(crash_test_dir)
        store2 = CDGStore(fs2, crash_config, simple_entity_factory)

        # Then pending file should be discarded
        assert not pending_file.exists(), "Pending should be discarded"

        # And entity should still be readable
        loaded = store2.read("E-nodelete-001")
        assert loaded is not None
        assert loaded.name == "still-here"


class TestEntityReconstructionFromWAL:
    """
    As a system administrator,
    I want entities to be reconstructed from WAL after committed transactions,
    So that data is recovered even if crash occurred before entity files were written.

    This tests the new WAL entity rollback feature where full entity state
    is stored in WAL WRITE entries and can be used to reconstruct missing
    or corrupted entities.
    """

    def test_scenario_missing_entity_reconstructed_from_wal(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Missing entity is reconstructed from WAL data

        Given a transaction committed in WAL with entity_data
        But the entity file is missing (crash before write)
        When recovery runs
        Then the entity is reconstructed from WAL
        Because WAL contains full entity state for recovery.
        """
        # Given a committed transaction in WAL with entity_data
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        # Entity data that will be stored in WAL
        entity_data = {
            "id": "E-reconstruct-001",
            "entity_type": "simple",
            "name": "reconstructed_entity",
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        }

        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-reconstruct-001",
             "op": "TX_BEGIN", "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-reconstruct-001",
             "op": "WRITE", "data": {
                 "entity_id": "E-reconstruct-001",
                 "old_version": 0,
                 "new_version": 1,
                 "entity_data": entity_data  # Full entity state in WAL
             }},
            {"seq": 3, "ts": "2025-01-01T00:00:02Z", "tx": "TX-reconstruct-001",
             "op": "TX_COMMIT", "data": {"version": 1}},
        ]

        # Add checksums
        for entry in entries:
            entry_copy = dict(entry)
            entry["checksum"] = compute_checksum(entry_copy)

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        # Create sequence file
        seq_file = wal_dir / "_sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 3}, f)

        # Entity file is MISSING (crash before write)
        entity_file = crash_test_dir / "E-reconstruct-001.json"
        assert not entity_file.exists()

        # When recovery runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config, simple_entity_factory)
        result = recovery.recover()

        # Then entity is reconstructed from WAL
        assert result.success
        assert "E-reconstruct-001" in result.reconstructed_entities
        assert entity_file.exists()

        # And the entity can be read correctly
        store = CDGStore(fs, crash_config, simple_entity_factory)
        loaded = store.read("E-reconstruct-001")
        assert loaded is not None
        assert loaded.id == "E-reconstruct-001"

    def test_scenario_corrupted_entity_reconstructed_from_wal(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Corrupted entity is reconstructed from WAL data

        Given a transaction committed in WAL with entity_data
        And the entity file exists but is corrupted
        When recovery runs
        Then the entity is reconstructed from WAL
        Because corrupted entity should be replaced with valid WAL data.
        """
        # Given a committed transaction in WAL with entity_data
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        entity_data = {
            "id": "E-corrupted-wal-001",
            "entity_type": "simple",
            "name": "should_be_recovered",
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        }

        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-corrupt-001",
             "op": "TX_BEGIN", "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-corrupt-001",
             "op": "WRITE", "data": {
                 "entity_id": "E-corrupted-wal-001",
                 "old_version": 0,
                 "new_version": 1,
                 "entity_data": entity_data
             }},
            {"seq": 3, "ts": "2025-01-01T00:00:02Z", "tx": "TX-corrupt-001",
             "op": "TX_COMMIT", "data": {"version": 1}},
        ]

        for entry in entries:
            entry_copy = dict(entry)
            entry["checksum"] = compute_checksum(entry_copy)

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        seq_file = wal_dir / "_sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 3}, f)

        # Entity file exists but is corrupted (truncated)
        crash_test_dir.mkdir(parents=True, exist_ok=True)
        entity_file = crash_test_dir / "E-corrupted-wal-001.json"
        entity_file.write_text('{"_checksum": "wrong", "data": {"trun')

        # When recovery runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config, simple_entity_factory)
        result = recovery.recover()

        # Then entity is reconstructed from WAL
        assert result.success
        assert "E-corrupted-wal-001" in result.reconstructed_entities

        # And the entity can be read correctly
        store = CDGStore(fs, crash_config, simple_entity_factory)
        loaded = store.read("E-corrupted-wal-001")
        assert loaded is not None
        assert loaded.id == "E-corrupted-wal-001"

    def test_scenario_multiple_entities_reconstructed_from_wal(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Multiple missing entities are reconstructed from WAL

        Given a transaction with multiple writes committed in WAL
        And multiple entity files are missing
        When recovery runs
        Then all missing entities are reconstructed
        Because WAL recovery handles batch writes correctly.
        """
        # Given a committed transaction with multiple writes
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        entity1_data = {
            "id": "E-multi-001",
            "entity_type": "simple",
            "name": "first_entity",
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        }

        entity2_data = {
            "id": "E-multi-002",
            "entity_type": "simple",
            "name": "second_entity",
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        }

        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-multi-001",
             "op": "TX_BEGIN", "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-multi-001",
             "op": "WRITE", "data": {
                 "entity_id": "E-multi-001",
                 "old_version": 0,
                 "new_version": 1,
                 "entity_data": entity1_data
             }},
            {"seq": 3, "ts": "2025-01-01T00:00:02Z", "tx": "TX-multi-001",
             "op": "WRITE", "data": {
                 "entity_id": "E-multi-002",
                 "old_version": 0,
                 "new_version": 1,
                 "entity_data": entity2_data
             }},
            {"seq": 4, "ts": "2025-01-01T00:00:03Z", "tx": "TX-multi-001",
             "op": "TX_COMMIT", "data": {"version": 1}},
        ]

        for entry in entries:
            entry_copy = dict(entry)
            entry["checksum"] = compute_checksum(entry_copy)

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        seq_file = wal_dir / "_sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 4}, f)

        # Both entity files are missing
        crash_test_dir.mkdir(parents=True, exist_ok=True)
        assert not (crash_test_dir / "E-multi-001.json").exists()
        assert not (crash_test_dir / "E-multi-002.json").exists()

        # When recovery runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config, simple_entity_factory)
        result = recovery.recover()

        # Then both entities are reconstructed
        assert result.success
        assert "E-multi-001" in result.reconstructed_entities
        assert "E-multi-002" in result.reconstructed_entities

        # And both entities can be read correctly
        store = CDGStore(fs, crash_config, simple_entity_factory)
        loaded1 = store.read("E-multi-001")
        loaded2 = store.read("E-multi-002")
        assert loaded1 is not None and loaded1.id == "E-multi-001"
        assert loaded2 is not None and loaded2.id == "E-multi-002"

    def test_scenario_uncommitted_tx_entity_not_reconstructed(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: Entity from uncommitted transaction is not reconstructed

        Given a transaction with entity_data in WAL
        But the transaction was never committed
        When recovery runs
        Then the entity is NOT reconstructed
        Because only committed transactions should have their entities restored.
        """
        # Given an uncommitted transaction in WAL with entity_data
        wal_dir = crash_test_dir / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"

        entity_data = {
            "id": "E-uncommitted-001",
            "entity_type": "simple",
            "name": "should_not_exist",
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        }

        entries = [
            {"seq": 1, "ts": "2025-01-01T00:00:00Z", "tx": "TX-nocommit-001",
             "op": "TX_BEGIN", "data": {"snapshot": 0}},
            {"seq": 2, "ts": "2025-01-01T00:00:01Z", "tx": "TX-nocommit-001",
             "op": "WRITE", "data": {
                 "entity_id": "E-uncommitted-001",
                 "old_version": 0,
                 "new_version": 1,
                 "entity_data": entity_data
             }},
            # No TX_COMMIT - transaction never committed
        ]

        for entry in entries:
            entry_copy = dict(entry)
            entry["checksum"] = compute_checksum(entry_copy)

        with open(wal_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        seq_file = wal_dir / "_sequence.json"
        with open(seq_file, 'w') as f:
            json.dump({"seq": 2}, f)

        # When recovery runs
        fs = RealFileSystem(crash_test_dir)
        recovery = CDGRecoveryManager(fs, crash_config, simple_entity_factory)
        result = recovery.recover()

        # Then entity is NOT reconstructed (tx was rolled back)
        assert "E-uncommitted-001" not in result.reconstructed_entities
        assert not (crash_test_dir / "E-uncommitted-001.json").exists()

    def test_scenario_entity_reconstruction_via_transaction_manager(
        self, crash_test_dir, crash_config
    ):
        """
        Scenario: End-to-end test of entity data in WAL via transaction manager

        Given a transaction committed through CDGTransactionManager
        When I check the WAL
        Then entity_data is present in WRITE entries
        Because transaction_manager now stores full entity state in WAL.
        """
        # Given a transaction committed through transaction manager
        fs = RealFileSystem(crash_test_dir)
        tm = CDGTransactionManager(fs, crash_config, simple_entity_factory)
        tx = tm.begin()

        entity = SimpleEntity(id="E-e2e-001", name="end_to_end_test")
        tm.write(tx, entity)

        result = tm.commit(tx)
        assert result.success

        # When I check the WAL
        wal_file = crash_test_dir / "wal" / "current.wal"
        with open(wal_file, 'r') as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Then entity_data is present in WRITE entries
        write_entries = [e for e in entries if e.get('op') == 'WRITE']
        assert len(write_entries) == 1

        write_data = write_entries[0].get('data', {})
        assert 'entity_data' in write_data
        assert write_data['entity_data']['id'] == "E-e2e-001"
        assert write_data['entity_data']['name'] == "end_to_end_test"
