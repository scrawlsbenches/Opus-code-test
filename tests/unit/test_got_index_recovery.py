"""
Tests for index recovery in RecoveryManager.

These tests verify that:
1. Recovery detects missing/corrupted indexes
2. Recovery rebuilds indexes from entities
3. Index recovery is part of the standard recovery flow

TDD: These tests are written FIRST before implementation.
"""

import json
import pytest
import tempfile
from pathlib import Path

from cortical.got.api import GoTManager
from cortical.got.recovery import RecoveryManager
from cortical.got.indexer import QueryIndexManager
from cortical.got.config import DurabilityMode


class TestIndexRecoveryDetection:
    """Test that recovery detects when indexes need rebuilding."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_needs_index_recovery_false_when_indexes_never_created(self, got_dir):
        """needs_index_recovery should return False when indexes were never created.

        Missing indexes don't need "recovery" - they were never created.
        Index initialization is the responsibility of GoTManager, not recovery.
        """
        # Create manager and add tasks WITHOUT accessing index_manager
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")
        manager.create_task("Task 2", status="completed")

        # Don't access index_manager - indexes should not exist
        index_dir = got_dir / "indexes"
        assert not index_dir.exists(), "Indexes should not exist yet"

        # Recovery should NOT detect missing indexes (they were never created)
        recovery = RecoveryManager(got_dir)
        assert not recovery.needs_index_recovery()

    def test_needs_index_recovery_false_when_empty_index_dir(self, got_dir):
        """needs_index_recovery should return False when index dir is empty.

        An empty index directory (leftover from aborted init) is not a
        recovery situation - there's nothing to recover.
        """
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")

        # Force index creation
        _ = manager.index_manager

        # Delete all index files but keep directory
        index_dir = got_dir / "indexes"
        for f in index_dir.glob("*.json"):
            f.unlink()

        # Empty directory - not a recovery situation
        recovery = RecoveryManager(got_dir)
        assert not recovery.needs_index_recovery()

    def test_needs_recovery_when_indexes_stale(self, got_dir):
        """needs_recovery should return True when indexes are stale."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending")

        # Force index creation
        _ = manager.index_manager

        # Manually add a task entity file without updating the index
        # (simulating crash between entity write and index update)
        import json
        from datetime import datetime, timezone
        from cortical.utils.checksums import compute_checksum

        entities_dir = got_dir / "entities"
        fake_task_id = "T-FAKE-12345678-abcd1234"
        fake_task_data = {
            "id": fake_task_id,
            "entity_type": "task",
            "title": "Orphan Task",
            "status": "pending",
            "priority": "medium",
            "description": "",
            "properties": {},
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.now(timezone.utc).isoformat()
        }
        # Wrap in expected format
        entity_wrapper = {"data": fake_task_data}
        entity_wrapper["_checksum"] = compute_checksum(fake_task_data)
        entity_wrapper["_written_at"] = datetime.now(timezone.utc).isoformat()

        with open(entities_dir / f"{fake_task_id}.json", "w") as f:
            json.dump(entity_wrapper, f)

        # Recovery should detect stale indexes
        recovery = RecoveryManager(got_dir)
        assert recovery.needs_index_recovery()


class TestIndexRecoveryExecution:
    """Test that recovery rebuilds indexes correctly."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_recover_rebuilds_stale_indexes(self, got_dir):
        """recover() should rebuild indexes when they become stale."""
        from datetime import datetime, timezone
        from cortical.utils.checksums import compute_checksum

        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending", priority="high")

        # Force index creation
        _ = manager.index_manager

        # Add a task directly to disk without updating the index (stale scenario)
        entities_dir = got_dir / "entities"
        stale_task_id = "T-STALE-12345678-abcd1234"
        stale_task_data = {
            "id": stale_task_id,
            "entity_type": "task",
            "title": "Stale Task",
            "status": "completed",
            "priority": "low",
            "description": "",
            "properties": {},
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.now(timezone.utc).isoformat()
        }
        entity_wrapper = {"data": stale_task_data}
        entity_wrapper["_checksum"] = compute_checksum(stale_task_data)
        entity_wrapper["_written_at"] = datetime.now(timezone.utc).isoformat()

        with open(entities_dir / f"{stale_task_id}.json", "w") as f:
            json.dump(entity_wrapper, f)

        # Run recovery
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Create new manager to access rebuilt indexes
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        index = manager2.index_manager

        # Verify indexes were rebuilt correctly (both tasks should be indexed)
        pending_ids = index.lookup("status", "pending")
        completed_ids = index.lookup("status", "completed")
        high_ids = index.lookup("priority", "high")
        low_ids = index.lookup("priority", "low")

        assert task1.id in pending_ids
        assert stale_task_id in completed_ids
        assert task1.id in high_ids
        assert stale_task_id in low_ids

    def test_recover_includes_index_action(self, got_dir):
        """Recovery result should include index rebuild action when indexes are stale."""
        from datetime import datetime, timezone
        from cortical.utils.checksums import compute_checksum

        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")

        # Force index creation
        _ = manager.index_manager

        # Add a task directly to disk without updating the index (stale scenario)
        entities_dir = got_dir / "entities"
        stale_task_id = "T-STALE-12345678-efgh5678"
        stale_task_data = {
            "id": stale_task_id,
            "entity_type": "task",
            "title": "Stale Task",
            "status": "pending",
            "priority": "medium",
            "description": "",
            "properties": {},
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.now(timezone.utc).isoformat()
        }
        entity_wrapper = {"data": stale_task_data}
        entity_wrapper["_checksum"] = compute_checksum(stale_task_data)
        entity_wrapper["_written_at"] = datetime.now(timezone.utc).isoformat()

        with open(entities_dir / f"{stale_task_id}.json", "w") as f:
            json.dump(entity_wrapper, f)

        # Run recovery
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Check for index rebuild action
        index_actions = [a for a in result.actions_taken if "index" in a.lower()]
        assert len(index_actions) > 0, "Recovery should report index rebuild action"


class TestIndexRecoveryResult:
    """Test RecoveryResult includes index information."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_recovery_result_has_indexes_rebuilt_flag(self, got_dir):
        """RecoveryResult should indicate if indexes were rebuilt due to stale state."""
        from datetime import datetime, timezone
        from cortical.utils.checksums import compute_checksum

        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")

        # Force index creation
        _ = manager.index_manager

        # Add a task directly to disk without updating the index (stale scenario)
        entities_dir = got_dir / "entities"
        stale_task_id = "T-STALE-12345678-ijkl9012"
        stale_task_data = {
            "id": stale_task_id,
            "entity_type": "task",
            "title": "Stale Task",
            "status": "pending",
            "priority": "medium",
            "description": "",
            "properties": {},
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.now(timezone.utc).isoformat()
        }
        entity_wrapper = {"data": stale_task_data}
        entity_wrapper["_checksum"] = compute_checksum(stale_task_data)
        entity_wrapper["_written_at"] = datetime.now(timezone.utc).isoformat()

        with open(entities_dir / f"{stale_task_id}.json", "w") as f:
            json.dump(entity_wrapper, f)

        # Run recovery
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Check indexes_rebuilt field
        assert hasattr(result, 'indexes_rebuilt')
        assert result.indexes_rebuilt == True

    def test_recovery_result_indexes_rebuilt_false_when_not_needed(self, got_dir):
        """RecoveryResult.indexes_rebuilt should be False when indexes are intact."""
        # Create manager and add tasks with indexes
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")
        _ = manager.index_manager  # Create indexes

        # Run recovery (should not need to rebuild)
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Indexes should not be rebuilt
        assert hasattr(result, 'indexes_rebuilt')
        assert result.indexes_rebuilt == False


class TestIndexErrorRecovery:
    """Test index recovery from various error conditions.

    These tests verify that the index system handles corruption,
    partial writes, and other error conditions gracefully.

    Task: T-20251226-112817-d9027d49
    """

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_recovery_from_corrupted_index_file(self, got_dir):
        """Recovery should rebuild indexes when index file is corrupted."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending", priority="high")
        _ = manager.index_manager

        # Corrupt the index file by writing garbage
        index_dir = got_dir / "indexes"
        index_dir.mkdir(exist_ok=True)
        status_index = index_dir / "by_status.json"
        with open(status_index, "w") as f:
            f.write("not valid json {{{")

        # Create new manager - should recover
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        index = manager2.index_manager

        # Index should still work after recovery
        pending_ids = index.lookup("status", "pending")
        assert task1.id in pending_ids

    def test_recovery_from_empty_index_file(self, got_dir):
        """Recovery should rebuild indexes when index file is empty."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="completed", priority="low")
        _ = manager.index_manager

        # Make index file empty
        index_dir = got_dir / "indexes"
        index_dir.mkdir(exist_ok=True)
        status_index = index_dir / "by_status.json"
        with open(status_index, "w") as f:
            f.write("")

        # Create new manager - should recover
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        index = manager2.index_manager

        # Index should still work after recovery
        completed_ids = index.lookup("status", "completed")
        assert task1.id in completed_ids

    def test_recovery_from_partial_index_data(self, got_dir):
        """Recovery should rebuild indexes when index has partial/truncated data."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending")
        task2 = manager.create_task("Task 2", status="in_progress")
        _ = manager.index_manager

        # Write partial JSON (truncated during write)
        index_dir = got_dir / "indexes"
        index_dir.mkdir(exist_ok=True)
        status_index = index_dir / "by_status.json"
        with open(status_index, "w") as f:
            f.write('{"pending": ["T-incomplete-data"')  # Truncated JSON

        # Create new manager - should recover
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        index = manager2.index_manager

        # Index should be rebuilt with correct data
        pending_ids = index.lookup("status", "pending")
        in_progress_ids = index.lookup("status", "in_progress")
        assert task1.id in pending_ids
        assert task2.id in in_progress_ids

    def test_index_recovered_after_entity_with_corrupted_checksum(self, got_dir):
        """Index should skip entities with corrupted checksums during rebuild."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending")
        task2 = manager.create_task("Task 2", status="pending")
        _ = manager.index_manager

        # Corrupt task1's checksum in entity file
        entity_file = got_dir / "entities" / f"{task1.id}.json"
        with open(entity_file, "r") as f:
            content = json.load(f)
        content["_checksum"] = "corrupted123"
        with open(entity_file, "w") as f:
            json.dump(content, f)

        # Delete indexes to force rebuild
        index_dir = got_dir / "indexes"
        if index_dir.exists():
            for f in index_dir.glob("*.json"):
                f.unlink()

        # Run recovery
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # task2 should be in index, task1 may be skipped due to corruption
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        index = manager2.index_manager
        pending_ids = index.lookup("status", "pending")

        # At minimum, task2 should be indexed
        assert task2.id in pending_ids

    def test_recovery_does_not_create_indexes_when_never_existed(self, got_dir):
        """Recovery should NOT create indexes if they never existed.

        Missing indexes don't need "recovery" - they were never created.
        Index initialization is the responsibility of GoTManager, not recovery.
        """
        # Create manager and add tasks (without accessing index_manager)
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending")

        # Ensure no index directory (don't access index_manager)
        index_dir = got_dir / "indexes"
        assert not index_dir.exists(), "Index directory should not exist yet"

        # Run recovery
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Indexes should NOT be rebuilt (they never existed)
        assert not result.indexes_rebuilt
        assert "clean" in result.actions_taken[0].lower() or "no recovery" in result.actions_taken[0].lower()

    def test_got_manager_creates_indexes_on_first_access(self, got_dir):
        """GoTManager should create indexes on first access to index_manager.

        This is the correct way to initialize indexes - not through recovery.
        """
        # Create manager and add tasks (without accessing index_manager)
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending")

        # Ensure no index directory
        index_dir = got_dir / "indexes"
        assert not index_dir.exists(), "Index directory should not exist yet"

        # Access index_manager - this should create indexes
        index = manager.index_manager

        # Indexes should now exist and contain the task
        pending_ids = index.lookup("status", "pending")
        assert task1.id in pending_ids
