"""
Behavioral tests for CDG storage history integrity.

Epic: System Administrator Expects Reliable History

As a system administrator,
I want history entries to accurately reflect actual changes,
So that I can trust the audit trail for recovery and compliance.

BUG REFERENCES:
- History-before-write: storage.py lines 260-269, 312-323
- History missing fsync: storage.py lines 631-634
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from cortical.cdg.storage import CDGStore
from cortical.cdg.types import Entity
from cortical.cdg.config import CDGConfig


class SimpleEntity(Entity):
    """Simple test entity."""

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


class TestHistoryAccuracyOnWriteFailure:
    """
    As a system administrator relying on history for audit,
    I want history to only contain entries for successful writes,
    So that I can trust the audit trail.

    BUG: History is saved BEFORE entity write, so if write fails,
    we have phantom history entries for changes that never happened.
    """

    def test_scenario_failed_write_should_not_create_history_entry(self, tmp_path):
        """
        Scenario: Write failure should not create phantom history

        Given an existing entity
        When I attempt to update it but the write fails
        Then no new history entry should be created
        Because phantom history entries corrupt the audit trail.

        BUG: Currently history is saved BEFORE write, creating phantom entries.
        """
        # Given an existing entity
        config = CDGConfig()
        config.enable_wal = False
        store = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        entity = SimpleEntity(id="E-test-001", name="original")
        store.write(entity)

        # Get initial history state
        history_path = tmp_path / "_history" / "E-test-001.jsonl"

        # First update should create history
        entity.name = "updated-once"
        store.write(entity)

        # Now we should have 1 history entry
        assert history_path.exists()
        with open(history_path) as f:
            initial_history = [json.loads(line) for line in f if line.strip()]
        initial_count = len(initial_history)
        assert initial_count == 1, "Should have exactly 1 history entry"

        # When I attempt update but write fails
        entity.name = "failed-update"

        # Simulate write failure by patching _write_with_checksum
        original_write = store._write_with_checksum

        def failing_write(*args, **kwargs):
            raise IOError("Simulated disk failure")

        # Attempt the write (should fail)
        with patch.object(store, '_write_with_checksum', failing_write):
            with pytest.raises(IOError):
                store.write(entity)

        # Then - history should NOT have a new entry
        with open(history_path) as f:
            final_history = [json.loads(line) for line in f if line.strip()]

        # BUG DEMONSTRATION: History has a phantom entry for the failed write
        assert len(final_history) == initial_count, (
            f"BUG: History has {len(final_history)} entries, expected {initial_count}. "
            f"Failed write should not create phantom history entry."
        )

    def test_scenario_apply_writes_failure_should_not_create_history(self, tmp_path):
        """
        Scenario: Batch write failure should not create phantom history

        Given multiple existing entities
        When I batch update them but the operation fails partway
        Then no new history entries should be created for any entity
        Because all-or-nothing applies to history too.

        BUG: apply_writes saves history for each entity before writing temp files.
        """
        # Given multiple existing entities
        config = CDGConfig()
        config.enable_wal = False
        store = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        entity1 = SimpleEntity(id="E-batch-001", name="original1")
        entity2 = SimpleEntity(id="E-batch-002", name="original2")
        store.write(entity1)
        store.write(entity2)

        # Update both once to establish history files
        entity1.name = "updated1"
        entity2.name = "updated2"
        store.apply_writes({entity1.id: entity1, entity2.id: entity2})

        # Get initial history counts
        history1 = tmp_path / "_history" / "E-batch-001.jsonl"
        history2 = tmp_path / "_history" / "E-batch-002.jsonl"

        with open(history1) as f:
            initial_count1 = len([l for l in f if l.strip()])
        with open(history2) as f:
            initial_count2 = len([l for l in f if l.strip()])

        # When I batch update but it fails
        entity1.name = "failed1"
        entity2.name = "failed2"

        write_count = [0]
        original_write = store._write_with_checksum

        def failing_on_second(*args, **kwargs):
            write_count[0] += 1
            if write_count[0] == 2:  # Fail on second entity
                raise IOError("Simulated disk failure")
            return original_write(*args, **kwargs)

        with patch.object(store, '_write_with_checksum', failing_on_second):
            with pytest.raises(IOError):
                store.apply_writes({entity1.id: entity1, entity2.id: entity2})

        # Then - no new history entries for either entity
        with open(history1) as f:
            final_count1 = len([l for l in f if l.strip()])
        with open(history2) as f:
            final_count2 = len([l for l in f if l.strip()])

        # BUG DEMONSTRATION: Both entities have phantom history entries
        assert final_count1 == initial_count1, (
            f"BUG: Entity1 has {final_count1} history entries, expected {initial_count1}. "
            "Failed batch write should not create phantom history."
        )
        assert final_count2 == initial_count2, (
            f"BUG: Entity2 has {final_count2} history entries, expected {initial_count2}. "
            "Failed batch write should not create phantom history."
        )


class TestHistoryOrderingCorrectness:
    """
    As a system administrator reviewing audit logs,
    I want history to reflect the actual state at each point in time,
    So that I can accurately reconstruct what happened.
    """

    def test_scenario_history_should_match_entity_state_before_change(self, tmp_path):
        """
        Scenario: History entry contains the state BEFORE the change

        Given an entity with a known state
        When I update it
        Then the history entry should contain the PREVIOUS state
        Because we want to know what was changed FROM, not TO.
        """
        # Given
        config = CDGConfig()
        config.enable_wal = False
        store = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        entity = SimpleEntity(id="E-history-001", name="state-A")
        store.write(entity)

        # When I update
        entity.name = "state-B"
        store.write(entity)

        # Then history should show state-A (the previous state)
        history_path = tmp_path / "_history" / "E-history-001.jsonl"
        with open(history_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == 1
        assert entries[0]["data"]["name"] == "state-A", (
            "History should capture the state BEFORE the change"
        )


class TestHistoryPersistsAcrossRestart:
    """
    As a system administrator recovering from issues,
    I want history to survive system restarts,
    So that I can review past states after recovery.
    """

    def test_scenario_history_readable_after_store_restart(self, tmp_path):
        """
        Scenario: History survives store restart

        Given an entity that was modified
        When I restart the store
        Then the history should still be readable
        Because history must persist for audit purposes.
        """
        # Given an entity that was modified
        config = CDGConfig()
        config.enable_wal = False

        store1 = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)
        entity = SimpleEntity(id="E-persist-001", name="original")
        store1.write(entity)
        entity.name = "modified"
        store1.write(entity)

        # Verify history was created
        history_path = tmp_path / "_history" / "E-persist-001.jsonl"
        assert history_path.exists()

        # When I restart the store (create new instance)
        store2 = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        # Then history should be readable
        with open(history_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == 1
        assert entries[0]["data"]["name"] == "original"

        # And entity should have current state
        loaded = store2.read("E-persist-001")
        assert loaded.name == "modified"


class TestHistoryCrashRecovery:
    """
    Epic: Crash Recovery for History Integrity

    As a system administrator responsible for data integrity,
    I need history to survive crashes during write operations,
    So that audit trails are never lost due to unexpected failures.
    """

    def test_scenario_pending_history_recovered_after_crash(self, tmp_path):
        """
        Scenario: Crash after entity write but before history finalization

        Given an entity that was updated
        And the system crashed after entity write but before history finalization
        When the system restarts
        Then the history entry should be recovered from pending
        Because history must never be lost.
        """
        # Given an existing entity
        config = CDGConfig()
        config.enable_wal = False
        store = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        entity = SimpleEntity(id="E-crash-001", name="original")
        store.write(entity)

        # Simulate: Update entity and create pending history, but "crash" before finalization
        # We manually create the pending state that would exist after crash

        # Read current state
        current = store.read("E-crash-001")
        expected_version = current.version + 1

        # Create pending history file manually (simulating state after entity write but before finalize)
        pending_dir = tmp_path / "_history" / "_pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_file = pending_dir / "E-crash-001.pending"

        pending_entry = {
            "global_version": store.current_version(),
            "timestamp": "2025-01-01T00:00:00+00:00",
            "data": {"id": "E-crash-001", "name": "original", "entity_type": "simple",
                     "version": current.version, "created_at": "2025-01-01T00:00:00+00:00",
                     "modified_at": "2025-01-01T00:00:00+00:00"},
            "expected_entity_version": expected_version
        }

        with open(pending_file, 'w') as f:
            json.dump(pending_entry, f)
            f.write('\n')

        # Also update the entity file to simulate the entity write completed
        entity.name = "modified"
        entity._version = expected_version
        store._write_with_checksum(store._entity_path("E-crash-001"), entity.to_dict())

        assert pending_file.exists(), "Pending file should exist (simulating crash state)"

        # When: System restarts (new store instance triggers recovery)
        store2 = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        # Then: Pending file should be gone (finalized)
        assert not pending_file.exists(), "Pending file should be finalized during recovery"

        # And: History should contain the recovered entry
        history_path = tmp_path / "_history" / "E-crash-001.jsonl"
        with open(history_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == 1, f"Should have 1 history entry (recovered from pending), got {len(entries)}"
        assert entries[0]["data"]["name"] == "original"

    def test_scenario_pending_history_discarded_if_write_incomplete(self, tmp_path):
        """
        Scenario: Crash before entity write completes

        Given a pending history file exists
        But the entity write did not complete (version mismatch)
        When the system restarts
        Then the pending history should be discarded
        Because the entity change never happened.
        """
        # Given an existing entity
        config = CDGConfig()
        config.enable_wal = False
        store = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        entity = SimpleEntity(id="E-crash-002", name="original")
        store.write(entity)

        # Read current state
        current = store.read("E-crash-002")

        # Create pending history with WRONG expected version (simulating crash before entity write)
        pending_dir = tmp_path / "_history" / "_pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_file = pending_dir / "E-crash-002.pending"

        pending_entry = {
            "global_version": store.current_version(),
            "timestamp": "2025-01-01T00:00:00+00:00",
            "data": {"id": "E-crash-002", "name": "original", "entity_type": "simple"},
            "expected_entity_version": current.version + 999  # Wrong version - entity was never updated
        }

        with open(pending_file, 'w') as f:
            json.dump(pending_entry, f)
            f.write('\n')

        assert pending_file.exists()

        # When: System restarts
        store2 = CDGStore(tmp_path, config=config, entity_factory=simple_entity_factory)

        # Then: Pending file should be discarded (entity write never completed)
        assert not pending_file.exists(), "Pending should be discarded (version mismatch)"

        # And: History should be empty (no phantom entries)
        history_path = tmp_path / "_history" / "E-crash-002.jsonl"
        if history_path.exists():
            with open(history_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            assert len(entries) == 0, "Should have no history entries"


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("cdg_history_test")
