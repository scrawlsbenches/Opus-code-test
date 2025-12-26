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

    def test_needs_recovery_when_indexes_missing(self, got_dir):
        """needs_recovery should return True when index files are missing."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")
        manager.create_task("Task 2", status="completed")

        # Force index creation
        _ = manager.index_manager

        # Delete index files
        index_dir = got_dir / "indexes"
        if index_dir.exists():
            for f in index_dir.glob("*.json"):
                f.unlink()

        # Recovery should detect missing indexes
        recovery = RecoveryManager(got_dir)
        assert recovery.needs_index_recovery()

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

    def test_recover_rebuilds_missing_indexes(self, got_dir):
        """recover() should rebuild indexes from entities."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager.create_task("Task 1", status="pending", priority="high")
        task2 = manager.create_task("Task 2", status="completed", priority="low")

        # Force index creation
        _ = manager.index_manager

        # Delete index files
        index_dir = got_dir / "indexes"
        if index_dir.exists():
            for f in index_dir.glob("*.json"):
                f.unlink()

        # Run recovery
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Create new manager to access rebuilt indexes
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        index = manager2.index_manager

        # Verify indexes were rebuilt correctly
        pending_ids = index.lookup("status", "pending")
        completed_ids = index.lookup("status", "completed")
        high_ids = index.lookup("priority", "high")
        low_ids = index.lookup("priority", "low")

        assert task1.id in pending_ids
        assert task2.id in completed_ids
        assert task1.id in high_ids
        assert task2.id in low_ids

    def test_recover_includes_index_action(self, got_dir):
        """Recovery result should include index rebuild action."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")

        # Force index creation then delete
        _ = manager.index_manager

        index_dir = got_dir / "indexes"
        if index_dir.exists():
            for f in index_dir.glob("*.json"):
                f.unlink()

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
        """RecoveryResult should indicate if indexes were rebuilt."""
        # Create manager and add tasks
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        manager.create_task("Task 1", status="pending")

        # Force index creation then delete
        _ = manager.index_manager

        index_dir = got_dir / "indexes"
        if index_dir.exists():
            for f in index_dir.glob("*.json"):
                f.unlink()

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
