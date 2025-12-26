"""
Tests for concurrent access to QueryIndexManager.

These tests verify that the index system handles concurrent operations
correctly, including:
- Parallel task creation
- Parallel task updates
- Parallel task deletions
- Mixed concurrent operations

The race condition in VersionedStore._version.tmp has been fixed by
adding ProcessLock around version file operations. See:
- cortical/got/versioned_store.py: _save_version() now uses _version_lock
- Task: T-20251226-132353-68a469de (Fix VersionedStore race condition)
"""

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Tuple

from cortical.got.api import GoTManager
from cortical.got.indexer import QueryIndexManager
from cortical.got.config import DurabilityMode




class TestConcurrentTaskCreation:
    """Test index behavior when tasks are created concurrently."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_parallel_task_creation_all_indexed(self, got_dir):
        """All tasks created in parallel should be indexed."""
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        results: List[Tuple[str, str, str]] = []  # (name, task_id, status)
        errors: List[Exception] = []

        def create_task(name: str, status: str):
            try:
                task = manager.create_task(name, status=status)
                results.append((name, task.id, status))
            except Exception as e:
                errors.append(e)

        # Create 20 tasks in parallel
        threads = [
            threading.Thread(
                target=create_task,
                args=(f"Task-{i}", "pending" if i % 2 == 0 else "completed")
            )
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All tasks should be in the index
        index = manager.index_manager
        for name, task_id, status in results:
            status_ids = index.lookup("status", status)
            assert task_id in status_ids, f"Task {task_id} not found in {status} index"

    def test_parallel_task_creation_index_counts_match(self, got_dir):
        """Index counts should match number of tasks created."""
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        created_count = [0]  # Use list for mutable reference
        lock = threading.Lock()

        def create_task(i: int):
            try:
                manager.create_task(f"Task-{i}", status="pending")
                with lock:
                    created_count[0] += 1
            except Exception:
                pass

        threads = [
            threading.Thread(target=create_task, args=(i,))
            for i in range(15)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Index should have all created tasks
        index = manager.index_manager
        pending_ids = index.lookup("status", "pending")
        assert len(pending_ids) == created_count[0]


class TestConcurrentTaskUpdates:
    """Test index behavior when tasks are updated concurrently."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_parallel_status_updates_indexed_correctly(self, got_dir):
        """Index should reflect final status after parallel updates."""
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        # Create initial tasks
        tasks = [
            manager.create_task(f"Task-{i}", status="pending")
            for i in range(10)
        ]

        errors: List[Exception] = []

        def update_to_completed(task_id: str):
            try:
                manager.update_task(task_id, status="completed")
            except Exception as e:
                errors.append(e)

        # Update all tasks to completed in parallel
        threads = [
            threading.Thread(target=update_to_completed, args=(t.id,))
            for t in tasks
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All tasks should be completed in index
        index = manager.index_manager
        completed_ids = index.lookup("status", "completed")
        for task in tasks:
            assert task.id in completed_ids

    def test_rapid_status_toggles(self, got_dir):
        """Index should handle rapid status changes on same task."""
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        task = manager.create_task("Toggle Task", status="pending")
        errors: List[Exception] = []

        def toggle_status(to_status: str):
            try:
                manager.update_task(task.id, status=to_status)
            except Exception as e:
                errors.append(e)

        # Rapidly toggle between statuses
        threads = []
        for i in range(10):
            status = "completed" if i % 2 == 0 else "pending"
            threads.append(threading.Thread(target=toggle_status, args=(status,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Task should be in exactly one status bucket
        index = manager.index_manager
        pending_ids = index.lookup("status", "pending")
        completed_ids = index.lookup("status", "completed")

        # Task should be in one or the other, not both
        in_pending = task.id in pending_ids
        in_completed = task.id in completed_ids
        assert in_pending != in_completed, "Task should be in exactly one status"


class TestConcurrentMixedOperations:
    """Test index behavior with mixed concurrent operations."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_create_update_delete_in_parallel(self, got_dir):
        """Index should handle create, update, delete all in parallel."""
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        # Pre-create some tasks for updates and deletes
        update_tasks = [
            manager.create_task(f"Update-{i}", status="pending")
            for i in range(5)
        ]
        delete_tasks = [
            manager.create_task(f"Delete-{i}", status="pending")
            for i in range(5)
        ]

        errors: List[Exception] = []
        created_ids: List[str] = []
        lock = threading.Lock()

        def create_task(name: str):
            try:
                task = manager.create_task(name, status="pending")
                with lock:
                    created_ids.append(task.id)
            except Exception as e:
                errors.append(e)

        def update_task(task_id: str):
            try:
                manager.update_task(task_id, status="in_progress")
            except Exception as e:
                errors.append(e)

        def delete_task(task_id: str):
            try:
                manager.delete_task(task_id, force=True)
            except Exception as e:
                errors.append(e)

        threads = []

        # Add create threads
        for i in range(5):
            threads.append(threading.Thread(target=create_task, args=(f"New-{i}",)))

        # Add update threads
        for task in update_tasks:
            threads.append(threading.Thread(target=update_task, args=(task.id,)))

        # Add delete threads
        for task in delete_tasks:
            threads.append(threading.Thread(target=delete_task, args=(task.id,)))

        # Run all in parallel
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify final index state
        index = manager.index_manager

        # Created tasks should be in pending
        pending_ids = index.lookup("status", "pending")
        for task_id in created_ids:
            assert task_id in pending_ids

        # Updated tasks should be in in_progress
        in_progress_ids = index.lookup("status", "in_progress")
        for task in update_tasks:
            assert task.id in in_progress_ids

        # Deleted tasks should not be in any index
        all_indexed = (
            set(index.lookup("status", "pending")) |
            set(index.lookup("status", "in_progress")) |
            set(index.lookup("status", "completed"))
        )
        for task in delete_tasks:
            assert task.id not in all_indexed


class TestIndexThreadSafety:
    """Test that index operations themselves are thread-safe."""

    @pytest.fixture
    def got_dir(self):
        """Create a temporary GoT directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_parallel_index_reads_during_writes(self, got_dir):
        """Index reads should be consistent during concurrent writes."""
        manager = GoTManager(got_dir, durability=DurabilityMode.RELAXED)

        # Pre-create some tasks
        for i in range(10):
            manager.create_task(f"Task-{i}", status="pending")

        errors: List[Exception] = []
        read_results: List[int] = []
        lock = threading.Lock()

        def create_tasks():
            try:
                for i in range(5):
                    manager.create_task(f"New-{i}", status="pending")
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)

        def read_index():
            try:
                for _ in range(10):
                    index = manager.index_manager
                    count = len(index.lookup("status", "pending"))
                    with lock:
                        read_results.append(count)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        # Run writer and multiple readers in parallel
        threads = [
            threading.Thread(target=create_tasks),
            threading.Thread(target=read_index),
            threading.Thread(target=read_index),
            threading.Thread(target=read_index),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors during concurrent access
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All reads should return valid counts (monotonically increasing or equal)
        assert len(read_results) > 0
        assert all(c >= 10 for c in read_results)  # At least initial count
