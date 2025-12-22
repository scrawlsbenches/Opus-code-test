"""
Tests for VersionedStore module.

Validates ACID properties, checksums, versioning, and atomic operations.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cortical.got.versioned_store import VersionedStore
from cortical.got.types import Task, Decision, Edge, Entity
from cortical.got.errors import CorruptionError
from cortical.utils.checksums import compute_checksum


class TestVersionedStoreBasics:
    """Test basic VersionedStore operations."""

    def test_write_creates_file_with_checksum(self, tmp_path):
        """Write entity, verify file has checksum wrapper."""
        store = VersionedStore(tmp_path)
        task = Task(id="T1", title="Test Task", status="pending", priority="high")

        store.write(task)

        # Verify file exists
        task_file = tmp_path / "T1.json"
        assert task_file.exists()

        # Read raw JSON and verify wrapper structure
        with open(task_file, 'r') as f:
            wrapper = json.load(f)

        assert "_checksum" in wrapper
        assert "_written_at" in wrapper
        assert "data" in wrapper
        assert wrapper["data"]["id"] == "T1"
        assert wrapper["data"]["title"] == "Test Task"

        # Verify checksum is correct
        expected_checksum = compute_checksum(wrapper["data"])
        assert wrapper["_checksum"] == expected_checksum

    def test_read_verifies_checksum(self, tmp_path):
        """Read entity, verify checksum is verified (no error raised)."""
        store = VersionedStore(tmp_path)
        task = Task(id="T2", title="Valid Task", status="in_progress", priority="medium")

        store.write(task)
        loaded_task = store.read("T2")

        assert loaded_task is not None
        assert loaded_task.id == "T2"
        assert loaded_task.title == "Valid Task"
        assert isinstance(loaded_task, Task)

    def test_corrupted_checksum_raises_error(self, tmp_path):
        """Corrupt checksum, verify CorruptionError is raised."""
        store = VersionedStore(tmp_path)
        task = Task(id="T3", title="Corrupt Task", status="pending", priority="low")

        store.write(task)

        # Manually corrupt the checksum
        task_file = tmp_path / "T3.json"
        with open(task_file, 'r') as f:
            wrapper = json.load(f)

        wrapper["_checksum"] = "0000000000000000"  # Invalid checksum

        with open(task_file, 'w') as f:
            json.dump(wrapper, f)

        # Attempt to read should raise CorruptionError
        with pytest.raises(CorruptionError) as exc_info:
            store.read("T3")

        assert "Checksum mismatch" in str(exc_info.value)

    def test_read_nonexistent_returns_none(self, tmp_path):
        """Read non-existent entity returns None."""
        store = VersionedStore(tmp_path)
        result = store.read("NONEXISTENT")
        assert result is None

    def test_exists_true_for_existing(self, tmp_path):
        """exists() returns True for existing entity."""
        store = VersionedStore(tmp_path)
        task = Task(id="T4", title="Exists Test", status="pending", priority="medium")

        store.write(task)
        assert store.exists("T4") is True

    def test_exists_false_for_nonexistent(self, tmp_path):
        """exists() returns False for non-existent entity."""
        store = VersionedStore(tmp_path)
        assert store.exists("NONEXISTENT") is False

    def test_delete_removes_file(self, tmp_path):
        """Delete removes file and returns True."""
        store = VersionedStore(tmp_path)
        task = Task(id="T5", title="Delete Test", status="pending", priority="medium")

        store.write(task)
        assert store.exists("T5") is True

        # Delete
        result = store.delete("T5")
        assert result is True
        assert store.exists("T5") is False

        # Delete non-existent returns False
        result = store.delete("T5")
        assert result is False


class TestVersionedStoreVersioning:
    """Test versioning and history functionality."""

    def test_version_increments_on_write(self, tmp_path):
        """Write entity, verify version increments."""
        store = VersionedStore(tmp_path)
        assert store.current_version() == 0

        task = Task(id="T6", title="Version Test", status="pending", priority="medium")
        store.write(task)
        assert store.current_version() == 1

        task.title = "Updated Title"
        store.write(task)
        assert store.current_version() == 2

    def test_read_at_version_returns_historical(self, tmp_path):
        """Write multiple versions, read old version from history."""
        store = VersionedStore(tmp_path)

        # Create task
        task = Task(id="T7", title="Version 1", status="pending", priority="medium")
        store.write(task)
        version1 = store.current_version()

        # Update task
        task.title = "Version 2"
        task.status = "in_progress"
        store.write(task)
        version2 = store.current_version()

        # Update again
        task.title = "Version 3"
        task.status = "completed"
        store.write(task)
        version3 = store.current_version()

        # Read at version 1 should return old state
        old_task = store.read_at_version("T7", version1)
        assert old_task is not None
        assert old_task.title == "Version 1"
        assert old_task.status == "pending"

        # Read at version 2 should return middle state
        mid_task = store.read_at_version("T7", version2)
        assert mid_task is not None
        assert mid_task.title == "Version 2"
        assert mid_task.status == "in_progress"

        # Read at version 3 should return current state
        current_task = store.read_at_version("T7", version3)
        assert current_task is not None
        assert current_task.title == "Version 3"
        assert current_task.status == "completed"

    def test_read_at_version_nonexistent(self, tmp_path):
        """Read at version before entity existed returns None."""
        store = VersionedStore(tmp_path)

        # Create task at version 0->1
        task = Task(id="T8", title="Created Later", status="pending", priority="medium")
        store.write(task)

        # Read at version 0 (before creation) should return None
        result = store.read_at_version("T8", 0)
        assert result is None


class TestVersionedStoreAtomicity:
    """Test atomic operations and crash resistance."""

    def test_atomic_write_survives_crash(self, tmp_path):
        """Mock crash during write, verify no partial state."""
        store = VersionedStore(tmp_path)

        task1 = Task(id="T9", title="Task 1", status="pending", priority="medium")
        task2 = Task(id="T10", title="Task 2", status="pending", priority="medium")

        # Simulate crash during rename by making it fail
        original_rename = Path.rename

        def failing_rename(self, target):
            if "T10.json" in str(target):
                raise OSError("Simulated crash during rename")
            return original_rename(self, target)

        with patch.object(Path, 'rename', failing_rename):
            with pytest.raises(OSError, match="Simulated crash"):
                store.apply_writes({"T9": task1, "T10": task2})

        # Verify no partial state - neither task should exist
        assert store.exists("T9") is False
        assert store.exists("T10") is False

        # Temp files should be cleaned up
        assert not (tmp_path / "T9.json.tmp").exists()
        assert not (tmp_path / "T10.json.tmp").exists()

    def test_apply_writes_atomic_batch(self, tmp_path):
        """Apply multiple writes atomically."""
        store = VersionedStore(tmp_path)

        task1 = Task(id="T11", title="Task 11", status="pending", priority="high")
        task2 = Task(id="T12", title="Task 12", status="in_progress", priority="medium")
        decision = Decision(id="D1", title="Decision 1", rationale="Testing")

        write_set = {
            "T11": task1,
            "T12": task2,
            "D1": decision,
        }

        new_version = store.apply_writes(write_set)

        # All entities should exist
        assert store.exists("T11") is True
        assert store.exists("T12") is True
        assert store.exists("D1") is True

        # Verify version incremented once (atomic batch)
        assert new_version == 1
        assert store.current_version() == 1

        # Verify all entities readable
        loaded_task1 = store.read("T11")
        assert loaded_task1.title == "Task 11"

        loaded_decision = store.read("D1")
        assert isinstance(loaded_decision, Decision)
        assert loaded_decision.rationale == "Testing"

    def test_apply_writes_increments_global_version(self, tmp_path):
        """Apply writes increments global version once per batch."""
        store = VersionedStore(tmp_path)

        # First write
        task1 = Task(id="T13", title="First", status="pending", priority="medium")
        store.write(task1)
        assert store.current_version() == 1

        # Batch write
        task2 = Task(id="T14", title="Second", status="pending", priority="medium")
        task3 = Task(id="T15", title="Third", status="pending", priority="medium")

        store.apply_writes({"T14": task2, "T15": task3})
        assert store.current_version() == 2  # Only incremented once for batch

    def test_fsync_called_on_write(self, tmp_path):
        """Mock os.fsync, verify it's called during write in PARANOID mode."""
        from cortical.got.config import DurabilityMode
        store = VersionedStore(tmp_path, durability=DurabilityMode.PARANOID)
        task = Task(id="T16", title="Fsync Test", status="pending", priority="medium")

        with patch('os.fsync') as mock_fsync:
            store.write(task)

            # Fsync should be called for version file
            assert mock_fsync.called
            # Should be called at least once (for version file)
            assert mock_fsync.call_count >= 1


class TestVersionedStoreEntityTypes:
    """Test correct entity type deserialization."""

    def test_read_task_returns_task_instance(self, tmp_path):
        """Read task entity returns Task instance."""
        store = VersionedStore(tmp_path)
        task = Task(id="T17", title="Task Type", status="pending", priority="medium")

        store.write(task)
        loaded = store.read("T17")

        assert isinstance(loaded, Task)
        assert loaded.title == "Task Type"
        assert loaded.status == "pending"

    def test_read_decision_returns_decision_instance(self, tmp_path):
        """Read decision entity returns Decision instance."""
        store = VersionedStore(tmp_path)
        decision = Decision(
            id="D2",
            title="Decision Type",
            rationale="Testing types",
            affects=["T1", "T2"]
        )

        store.write(decision)
        loaded = store.read("D2")

        assert isinstance(loaded, Decision)
        assert loaded.title == "Decision Type"
        assert loaded.rationale == "Testing types"
        assert loaded.affects == ["T1", "T2"]

    def test_read_edge_returns_edge_instance(self, tmp_path):
        """Read edge entity returns Edge instance."""
        store = VersionedStore(tmp_path)
        edge = Edge(
            id="E1",
            source_id="T1",
            target_id="T2",
            edge_type="DEPENDS_ON",
            weight=0.8,
            confidence=0.9
        )

        store.write(edge)
        loaded = store.read("E1")

        assert isinstance(loaded, Edge)
        assert loaded.source_id == "T1"
        assert loaded.target_id == "T2"
        assert loaded.edge_type == "DEPENDS_ON"
        assert loaded.weight == 0.8


class TestVersionedStoreHistory:
    """Test history file operations."""

    def test_history_file_created_on_update(self, tmp_path):
        """History file is created when entity is updated."""
        store = VersionedStore(tmp_path)

        # Create task
        task = Task(id="T18", title="Original", status="pending", priority="medium")
        store.write(task)

        # No history yet (first write)
        history_path = tmp_path / "_history" / "T18.jsonl"
        assert not history_path.exists()

        # Update task - this should create history
        task.title = "Updated"
        store.write(task)

        # History should now exist
        assert history_path.exists()

        # Read history file
        with open(history_path, 'r') as f:
            lines = f.readlines()

        # Should have one entry (original state)
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["data"]["title"] == "Original"

    def test_history_appends_on_multiple_updates(self, tmp_path):
        """History file appends entries on multiple updates."""
        store = VersionedStore(tmp_path)

        task = Task(id="T19", title="V1", status="pending", priority="medium")
        store.write(task)

        task.title = "V2"
        store.write(task)

        task.title = "V3"
        store.write(task)

        task.title = "V4"
        store.write(task)

        # Read history file
        history_path = tmp_path / "_history" / "T19.jsonl"
        with open(history_path, 'r') as f:
            lines = f.readlines()

        # Should have 3 entries (V1, V2, V3 saved before overwrites)
        assert len(lines) == 3

        entries = [json.loads(line) for line in lines]
        assert entries[0]["data"]["title"] == "V1"
        assert entries[1]["data"]["title"] == "V2"
        assert entries[2]["data"]["title"] == "V3"

    def test_delete_saves_to_history(self, tmp_path):
        """Delete operation saves entity to history before removing."""
        store = VersionedStore(tmp_path)

        task = Task(id="T20", title="To Delete", status="pending", priority="medium")
        store.write(task)
        store.delete("T20")

        # History should exist with deleted entity
        history_path = tmp_path / "_history" / "T20.jsonl"
        assert history_path.exists()

        with open(history_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["data"]["title"] == "To Delete"


class TestVersionedStoreInitialization:
    """Test store initialization and directory creation."""

    def test_creates_directory_structure(self, tmp_path):
        """Store initialization creates directory structure."""
        store_dir = tmp_path / "new_store"
        assert not store_dir.exists()

        store = VersionedStore(store_dir)

        assert store_dir.exists()
        assert (store_dir / "_history").exists()

    def test_loads_existing_version(self, tmp_path):
        """Store loads existing version on initialization."""
        # Create version file manually
        version_file = tmp_path / "_version.json"
        with open(version_file, 'w') as f:
            json.dump({"version": 42}, f)

        store = VersionedStore(tmp_path)
        assert store.current_version() == 42

    def test_starts_at_version_zero_if_no_file(self, tmp_path):
        """Store starts at version 0 if no version file exists."""
        store = VersionedStore(tmp_path)
        assert store.current_version() == 0
