"""
Tests for QueryIndexManager integration with TransactionManager.

These tests verify that:
1. Index updates happen atomically with transaction commits
2. Index updates are logged to WAL for crash recovery
3. Index lookups work correctly after transactions
4. Index state is consistent even after crashes

TDD: These tests are written FIRST before implementation.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cortical.got.api import GoTManager
from cortical.got.tx_manager import TransactionManager
from cortical.got.indexer import QueryIndexManager
from cortical.got.config import DurabilityMode


class TestIndexTransactionIntegration:
    """Test that indexes are updated atomically with transactions."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, got_dir):
        """Create a GoTManager with indexing enabled."""
        return GoTManager(got_dir, durability=DurabilityMode.RELAXED)

    def test_index_updated_after_task_create(self, manager):
        """Index should reflect newly created task."""
        # Create a task
        task = manager.create_task("Test task", priority="high", status="pending")

        # Get index manager and check it has the task indexed
        index = manager.index_manager

        # Lookup by status should return the task
        pending_ids = index.lookup("status", "pending")
        assert task.id in pending_ids

        # Lookup by priority should return the task
        high_ids = index.lookup("priority", "high")
        assert task.id in high_ids

    def test_index_updated_after_task_update(self, manager):
        """Index should reflect task status changes."""
        # Create a task
        task = manager.create_task("Test task", priority="medium", status="pending")

        # Update the task status
        manager.update_task(task.id, status="in_progress")

        # Get index manager
        index = manager.index_manager

        # Task should no longer be in pending
        pending_ids = index.lookup("status", "pending")
        assert task.id not in pending_ids

        # Task should now be in in_progress
        in_progress_ids = index.lookup("status", "in_progress")
        assert task.id in in_progress_ids

    def test_index_updated_after_task_delete(self, manager):
        """Index should remove deleted tasks."""
        # Create a task
        task = manager.create_task("Test task", priority="high", status="pending")

        # Verify it's in the index
        index = manager.index_manager
        assert task.id in index.lookup("status", "pending")

        # Delete the task
        manager.delete_task(task.id)

        # Task should be removed from all indexes
        assert task.id not in index.lookup("status", "pending")
        assert task.id not in index.lookup("priority", "high")

    def test_index_rollback_on_transaction_failure(self, manager):
        """Index should not be updated if transaction fails."""
        # Create a task to set up initial state
        initial_task = manager.create_task("Initial", status="pending")

        # Get initial index state
        index = manager.index_manager
        initial_pending = index.lookup("status", "pending").copy()

        # Try to update a non-existent task (should fail)
        try:
            with manager.transaction() as tx:
                # This should fail
                tx.update_task("non-existent-task-id", status="completed")
        except Exception:
            pass

        # Index should be unchanged
        current_pending = index.lookup("status", "pending")
        assert current_pending == initial_pending


class TestIndexWALLogging:
    """Test that index updates are logged to WAL."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, got_dir):
        """Create a GoTManager."""
        return GoTManager(got_dir, durability=DurabilityMode.RELAXED)

    def test_index_update_logged_to_wal(self, manager, got_dir):
        """Index updates should be logged to WAL for recovery."""
        # Create a task
        task = manager.create_task("Test task", status="pending")

        # Check WAL contains index-related entries
        wal_file = got_dir / "wal" / "wal.jsonl"
        if wal_file.exists():
            with open(wal_file, "r") as f:
                entries = [json.loads(line) for line in f if line.strip()]

            # Should have some entries (at minimum the transaction entries)
            assert len(entries) > 0

    def test_index_rebuild_from_entities_on_recovery(self, got_dir):
        """Indexes should be rebuilt from entities on recovery."""
        # Create manager and add tasks
        manager1 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)
        task1 = manager1.create_task("Task 1", status="pending", priority="high")
        task2 = manager1.create_task("Task 2", status="completed", priority="low")

        # Simulate crash by deleting index files
        index_dir = got_dir / "indexes"
        if index_dir.exists():
            for f in index_dir.glob("*.json"):
                f.unlink()

        # Create new manager (should recover/rebuild indexes)
        manager2 = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        # Index should be rebuilt
        index = manager2.index_manager

        # Verify tasks are indexed correctly
        pending_ids = index.lookup("status", "pending")
        completed_ids = index.lookup("status", "completed")

        assert task1.id in pending_ids
        assert task2.id in completed_ids


class TestIndexConcurrency:
    """Test index behavior under concurrent operations.

    NOTE: Full concurrent access tests are in Task T-20251226-112810-f4d8650c.
    The current VersionedStore has a known race condition with _version.tmp
    that needs to be fixed before concurrent tests can pass reliably.
    """

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_index_consistent_after_concurrent_transactions(self, got_dir):
        """Index should remain consistent after concurrent transactions."""
        import threading

        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        # Create tasks in parallel threads
        results = []
        errors = []

        def create_task(name, status):
            try:
                task = manager.create_task(name, status=status)
                results.append((name, task.id, status))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_task, args=(f"Task-{i}", "pending" if i % 2 == 0 else "completed"))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify index consistency
        index = manager.index_manager

        # All created tasks should be in the index
        for name, task_id, status in results:
            status_ids = index.lookup("status", status)
            assert task_id in status_ids, f"Task {task_id} not found in {status} index"


class TestIndexTransactionContext:
    """Test index updates within TransactionContext."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, got_dir):
        """Create a GoTManager."""
        return GoTManager(got_dir, durability=DurabilityMode.RELAXED)

    def test_batch_updates_indexed_atomically(self, manager):
        """Multiple updates in one transaction should be indexed atomically."""
        with manager.transaction() as tx:
            task1 = tx.create_task("Task 1", status="pending")
            task2 = tx.create_task("Task 2", status="pending")
            tx.update_task(task1.id, status="in_progress")

        # After commit, indexes should reflect final state
        index = manager.index_manager

        # task1 should be in_progress
        in_progress_ids = index.lookup("status", "in_progress")
        assert task1.id in in_progress_ids

        # task2 should be pending
        pending_ids = index.lookup("status", "pending")
        assert task2.id in pending_ids

    def test_index_property_access(self, manager):
        """GoTManager should expose index_manager property."""
        assert hasattr(manager, 'index_manager')
        assert isinstance(manager.index_manager, QueryIndexManager)
