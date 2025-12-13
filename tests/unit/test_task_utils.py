"""Unit tests for merge-friendly task ID utilities."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from task_utils import (
    generate_task_id,
    generate_short_task_id,
    generate_session_id,
    Task,
    TaskSession,
    load_all_tasks,
    get_task_by_id,
    consolidate_tasks,
)


class TestTaskIdGeneration(unittest.TestCase):
    """Tests for task ID generation."""

    def test_generate_task_id_format(self):
        """Task ID should have correct format."""
        task_id = generate_task_id()
        # Format: T-YYYYMMDD-HHMMSS-XXXX
        self.assertTrue(task_id.startswith("T-"))
        parts = task_id.split("-")
        self.assertEqual(len(parts), 4)
        self.assertEqual(len(parts[1]), 8)  # YYYYMMDD
        self.assertEqual(len(parts[2]), 6)  # HHMMSS
        self.assertEqual(len(parts[3]), 4)  # session suffix

    def test_generate_task_id_with_session(self):
        """Task ID should use provided session suffix."""
        task_id = generate_task_id("test")
        self.assertTrue(task_id.endswith("-test"))

    def test_generate_short_task_id(self):
        """Short task ID should be 10 characters."""
        task_id = generate_short_task_id()
        # Format: T-XXXXXXXX
        self.assertTrue(task_id.startswith("T-"))
        self.assertEqual(len(task_id), 10)

    def test_generate_session_id(self):
        """Session ID should be 4 hex characters."""
        session_id = generate_session_id()
        self.assertEqual(len(session_id), 4)
        # Should be valid hex
        int(session_id, 16)

    def test_unique_task_ids(self):
        """Generated task IDs should be unique."""
        ids = {generate_task_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestTask(unittest.TestCase):
    """Tests for Task dataclass."""

    def test_task_creation(self):
        """Task should be created with required fields."""
        task = Task(id="T-test", title="Test task")
        self.assertEqual(task.id, "T-test")
        self.assertEqual(task.title, "Test task")
        self.assertEqual(task.status, "pending")

    def test_task_to_dict(self):
        """Task should serialize to dict."""
        task = Task(
            id="T-test",
            title="Test task",
            priority="high",
            category="arch"
        )
        d = task.to_dict()
        self.assertEqual(d["id"], "T-test")
        self.assertEqual(d["title"], "Test task")
        self.assertEqual(d["priority"], "high")

    def test_task_from_dict(self):
        """Task should deserialize from dict."""
        d = {
            "id": "T-test",
            "title": "Test task",
            "status": "completed",
            "priority": "low",
            "category": "test",
            "description": "",
            "depends_on": [],
            "effort": "small",
            "created_at": "2025-12-13T00:00:00",
            "updated_at": None,
            "completed_at": None,
            "context": {}
        }
        task = Task.from_dict(d)
        self.assertEqual(task.id, "T-test")
        self.assertEqual(task.status, "completed")

    def test_mark_complete(self):
        """mark_complete should update status and timestamp."""
        task = Task(id="T-test", title="Test task")
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.completed_at)

        task.mark_complete()
        self.assertEqual(task.status, "completed")
        self.assertIsNotNone(task.completed_at)

    def test_mark_in_progress(self):
        """mark_in_progress should update status and timestamp."""
        task = Task(id="T-test", title="Test task")
        task.mark_in_progress()
        self.assertEqual(task.status, "in_progress")
        self.assertIsNotNone(task.updated_at)


class TestTaskSession(unittest.TestCase):
    """Tests for TaskSession."""

    def setUp(self):
        """Create temporary directory for task files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_session_id_consistency(self):
        """All tasks in session should share same suffix."""
        session = TaskSession()
        id1 = session.new_task_id()
        id2 = session.new_task_id()

        suffix1 = id1.split("-")[-1]
        suffix2 = id2.split("-")[-1]

        self.assertEqual(suffix1, suffix2)
        self.assertEqual(suffix1, session.session_id)

    def test_create_task(self):
        """create_task should add task to session."""
        session = TaskSession()
        task = session.create_task(
            title="Test task",
            priority="high"
        )

        self.assertEqual(len(session.tasks), 1)
        self.assertEqual(task.title, "Test task")
        self.assertEqual(task.priority, "high")

    def test_save_and_load(self):
        """Session should save and load correctly."""
        session = TaskSession()
        session.create_task(title="Task 1")
        session.create_task(title="Task 2")

        filepath = session.save(self.temp_dir)
        self.assertTrue(filepath.exists())

        loaded = TaskSession.load(filepath)
        self.assertEqual(len(loaded.tasks), 2)
        self.assertEqual(loaded.tasks[0].title, "Task 1")

    def test_session_filename_format(self):
        """Session filename should have correct format."""
        session = TaskSession()
        filename = session.get_filename()
        # Format: YYYY-MM-DD_HH-MM-SS_XXXX.json
        self.assertTrue(filename.endswith(".json"))
        parts = filename[:-5].split("_")  # Remove .json
        self.assertEqual(len(parts), 3)


class TestTaskLoading(unittest.TestCase):
    """Tests for loading tasks from multiple files."""

    def setUp(self):
        """Create temporary directory with task files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create two sessions
        session1 = TaskSession()
        session1.create_task(title="Session 1 Task 1")
        session1.create_task(title="Session 1 Task 2")
        session1.save(self.temp_dir)

        session2 = TaskSession()
        session2.create_task(title="Session 2 Task 1")
        session2.save(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_all_tasks(self):
        """load_all_tasks should load from all session files."""
        tasks = load_all_tasks(self.temp_dir)
        self.assertEqual(len(tasks), 3)

    def test_load_empty_directory(self):
        """load_all_tasks should return empty list for missing directory."""
        tasks = load_all_tasks("/nonexistent")
        self.assertEqual(tasks, [])

    def test_get_task_by_id(self):
        """get_task_by_id should find task across sessions."""
        tasks = load_all_tasks(self.temp_dir)
        target_id = tasks[0].id

        found = get_task_by_id(target_id, self.temp_dir)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, target_id)

    def test_get_task_by_id_not_found(self):
        """get_task_by_id should return None for missing ID."""
        found = get_task_by_id("T-nonexistent", self.temp_dir)
        self.assertIsNone(found)


class TestConsolidation(unittest.TestCase):
    """Tests for task consolidation."""

    def setUp(self):
        """Create temporary directory with task files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_consolidate_groups_by_status(self):
        """consolidate_tasks should group by status."""
        session = TaskSession()
        task1 = session.create_task(title="Pending task")
        task2 = session.create_task(title="In progress task")
        task2.mark_in_progress()
        task3 = session.create_task(title="Completed task")
        task3.mark_complete()
        session.save(self.temp_dir)

        grouped = consolidate_tasks(self.temp_dir)

        self.assertEqual(len(grouped["pending"]), 1)
        self.assertEqual(len(grouped["in_progress"]), 1)
        self.assertEqual(len(grouped["completed"]), 1)


if __name__ == "__main__":
    unittest.main()
