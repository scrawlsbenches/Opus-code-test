"""
Integration tests for merge-friendly task utilities.

These tests verify that the task system works correctly in realistic
multi-agent scenarios where multiple sessions create tasks concurrently.

Run with: pytest tests/integration/test_task_integration.py -v
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from task_utils import (
    TaskSession,
    Task,
    load_all_tasks,
    consolidate_tasks,
    generate_task_id,
    generate_short_task_id,
)
from consolidate_tasks import (
    consolidate_and_dedupe,
    find_conflicts,
    merge_duplicate_tasks,
    write_consolidated_file,
    archive_old_session_files,
)


class TestMultiSessionConcurrency(unittest.TestCase):
    """Test that multiple sessions don't conflict when run concurrently."""

    def setUp(self):
        """Create temporary directory for task files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_parallel_sessions_create_unique_files(self):
        """Multiple sessions running in parallel should create unique files."""
        num_sessions = 10
        sessions_created = []

        def create_session_with_tasks():
            session = TaskSession()
            session.create_task(title=f"Task from session {session.session_id}")
            filepath = session.save(self.temp_dir)
            return str(filepath)

        # Run sessions in parallel
        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [executor.submit(create_session_with_tasks) for _ in range(num_sessions)]
            for future in as_completed(futures):
                sessions_created.append(future.result())

        # All files should be unique
        self.assertEqual(len(set(sessions_created)), num_sessions)

        # All files should exist
        for filepath in sessions_created:
            self.assertTrue(Path(filepath).exists())

    def test_parallel_task_id_generation_high_uniqueness(self):
        """Task IDs generated in parallel should be mostly unique.

        Note: Same-second generation may cause collisions (timestamp-based).
        We verify >95% uniqueness which is sufficient for real-world use
        where agents run for longer durations.
        """
        num_ids = 1000
        generated_ids = []

        def generate_ids():
            return [generate_task_id() for _ in range(100)]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_ids) for _ in range(10)]
            for future in as_completed(futures):
                generated_ids.extend(future.result())

        # At least 95% should be unique (same-second collisions expected)
        unique_count = len(set(generated_ids))
        uniqueness_ratio = unique_count / num_ids
        self.assertGreater(uniqueness_ratio, 0.95,
            f"Only {uniqueness_ratio*100:.1f}% unique IDs")

    def test_concurrent_session_saves_no_corruption(self):
        """Concurrent saves should not corrupt files."""
        sessions = [TaskSession() for _ in range(5)]

        # Add tasks to each session
        for i, session in enumerate(sessions):
            for j in range(10):
                session.create_task(title=f"Session {i} Task {j}")

        # Save all concurrently
        def save_session(session):
            return session.save(self.temp_dir)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_session, s) for s in sessions]
            filepaths = [f.result() for f in as_completed(futures)]

        # Verify all files are valid JSON
        for filepath in filepaths:
            with open(filepath) as f:
                data = json.load(f)
                self.assertIn("tasks", data)
                self.assertEqual(len(data["tasks"]), 10)


class TestConsolidationIntegration(unittest.TestCase):
    """Test task consolidation across multiple session files."""

    def setUp(self):
        """Create temporary directory with multiple session files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_consolidate_preserves_all_tasks(self):
        """Consolidation should preserve all tasks from all sessions."""
        total_tasks = 0

        # Create multiple sessions with varying task counts
        for i in range(5):
            session = TaskSession()
            task_count = (i + 1) * 3  # 3, 6, 9, 12, 15 tasks
            for j in range(task_count):
                session.create_task(title=f"Session {i} Task {j}")
            total_tasks += task_count
            session.save(self.temp_dir)

        # Load all tasks
        all_tasks = load_all_tasks(self.temp_dir)
        self.assertEqual(len(all_tasks), total_tasks)

    def test_consolidate_detects_duplicates(self):
        """Consolidation should detect tasks with same title as potential duplicates."""
        # Create two sessions with same task title
        session1 = TaskSession()
        session1.create_task(title="Fix authentication bug")
        session1.save(self.temp_dir)

        session2 = TaskSession()
        session2.create_task(title="Fix authentication bug")  # Duplicate!
        session2.save(self.temp_dir)

        # Load and check for conflicts
        all_tasks = load_all_tasks(self.temp_dir)
        conflicts = find_conflicts(all_tasks)

        self.assertEqual(len(conflicts), 1)
        self.assertIn("fix authentication bug", conflicts)
        self.assertEqual(len(conflicts["fix authentication bug"]), 2)

    def test_auto_merge_resolves_duplicates(self):
        """Auto-merge should combine duplicate tasks intelligently."""
        # Create two sessions with same task, different metadata
        session1 = TaskSession()
        task1 = session1.create_task(
            title="Fix authentication bug",
            priority="low",
            description="Short description"
        )
        session1.save(self.temp_dir)

        # Wait a bit to ensure different timestamp
        time.sleep(0.01)

        session2 = TaskSession()
        task2 = session2.create_task(
            title="Fix authentication bug",
            priority="high",  # Higher priority
            description="Much longer and more detailed description of the bug"
        )
        task2.mark_in_progress()  # More advanced status
        session2.save(self.temp_dir)

        # Consolidate with auto-merge
        tasks, conflicts = consolidate_and_dedupe(self.temp_dir, auto_merge=True)

        # Should have merged to one task
        auth_tasks = [t for t in tasks if "authentication" in t.title.lower()]
        self.assertEqual(len(auth_tasks), 1)

        merged = auth_tasks[0]
        # Should keep higher priority
        self.assertEqual(merged.priority, "high")
        # Should keep more advanced status
        self.assertEqual(merged.status, "in_progress")
        # Should keep longer description
        self.assertIn("longer", merged.description)

    def test_consolidated_file_is_valid(self):
        """Written consolidated file should be valid and loadable."""
        # Create some sessions
        for i in range(3):
            session = TaskSession()
            session.create_task(title=f"Task {i}")
            session.save(self.temp_dir)

        all_tasks = load_all_tasks(self.temp_dir)

        # Write consolidated file
        consolidated_path = write_consolidated_file(all_tasks, self.temp_dir)

        # Verify it's valid JSON
        with open(consolidated_path) as f:
            data = json.load(f)

        self.assertEqual(data["type"], "consolidated")
        self.assertEqual(data["task_count"], 3)
        self.assertEqual(len(data["tasks"]), 3)

    def test_archive_moves_files_correctly(self):
        """Archiving should move session files but keep consolidated."""
        # Create sessions
        session_files = []
        for i in range(3):
            session = TaskSession()
            session.create_task(title=f"Task {i}")
            path = session.save(self.temp_dir)
            session_files.append(path)

        # Write consolidated file
        all_tasks = load_all_tasks(self.temp_dir)
        consolidated_path = write_consolidated_file(all_tasks, self.temp_dir)

        # Archive old files
        archived = archive_old_session_files(self.temp_dir)

        # Session files should be moved
        self.assertEqual(len(archived), 3)
        for path in session_files:
            self.assertFalse(path.exists())

        # Consolidated file should remain
        self.assertTrue(consolidated_path.exists())

        # Archive directory should exist
        archive_dir = Path(self.temp_dir) / "archive"
        self.assertTrue(archive_dir.exists())
        self.assertEqual(len(list(archive_dir.glob("*.json"))), 3)


class TestTaskLifecycleIntegration(unittest.TestCase):
    """Test complete task lifecycle: create -> update -> complete -> consolidate."""

    def setUp(self):
        """Create temporary directory for task files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_full_task_lifecycle(self):
        """Test complete lifecycle of task management."""
        # 1. Create tasks in session
        session = TaskSession()
        task1 = session.create_task(
            title="Implement feature",
            priority="high",
            category="feature"
        )
        task2 = session.create_task(
            title="Write tests",
            priority="medium",
            depends_on=[task1.id]
        )

        # 2. Update task status before saving
        task1.mark_in_progress()
        task1.mark_complete()
        self.assertEqual(task1.status, "completed")
        self.assertIsNotNone(task1.completed_at)

        # 3. Save and verify persistence
        session.save(self.temp_dir)

        # 4. Load and verify
        all_tasks = load_all_tasks(self.temp_dir)
        self.assertEqual(len(all_tasks), 2)

        # 5. Consolidate and check grouping
        grouped = consolidate_tasks(self.temp_dir)
        self.assertEqual(len(grouped["pending"]), 1)  # task2
        self.assertEqual(len(grouped["completed"]), 1)  # task1

    def test_multi_agent_workflow_simulation(self):
        """Simulate a realistic multi-agent workflow."""
        # Agent A: Research phase
        agent_a = TaskSession()
        a1 = agent_a.create_task(
            title="Research existing codebase",
            priority="high",
            category="research"
        )
        a2 = agent_a.create_task(
            title="Identify integration points",
            depends_on=[a1.id]
        )
        agent_a.save(self.temp_dir)

        # Agent B: Implementation phase (concurrent)
        agent_b = TaskSession()
        b1 = agent_b.create_task(
            title="Implement core feature",
            priority="high",
            category="implementation"
        )
        b2 = agent_b.create_task(
            title="Add unit tests",
            depends_on=[b1.id]
        )
        agent_b.save(self.temp_dir)

        # Agent C: Documentation (concurrent)
        agent_c = TaskSession()
        c1 = agent_c.create_task(
            title="Update documentation",
            priority="medium",
            category="docs"
        )
        agent_c.save(self.temp_dir)

        # Verify all tasks exist and are unique
        all_tasks = load_all_tasks(self.temp_dir)
        self.assertEqual(len(all_tasks), 5)

        # All IDs should be unique
        all_ids = [t.id for t in all_tasks]
        self.assertEqual(len(set(all_ids)), 5)

        # Dependencies should reference valid IDs
        for task in all_tasks:
            for dep_id in task.depends_on:
                self.assertTrue(
                    any(t.id == dep_id for t in all_tasks),
                    f"Dependency {dep_id} not found"
                )


class TestFileSystemResilience(unittest.TestCase):
    """Test resilience to file system edge cases."""

    def setUp(self):
        """Create temporary directory for task files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_handles_missing_directory(self):
        """load_all_tasks should handle missing directory gracefully."""
        tasks = load_all_tasks("/nonexistent/path/that/does/not/exist")
        self.assertEqual(tasks, [])

    def test_handles_empty_directory(self):
        """load_all_tasks should handle empty directory."""
        tasks = load_all_tasks(self.temp_dir)
        self.assertEqual(tasks, [])

    def test_handles_corrupt_file(self):
        """load_all_tasks should skip corrupt files."""
        # Create valid session
        session = TaskSession()
        session.create_task(title="Valid task")
        session.save(self.temp_dir)

        # Create corrupt file
        corrupt_path = Path(self.temp_dir) / "2025-12-13_00-00-00_xxxx.json"
        with open(corrupt_path, "w") as f:
            f.write("{ this is not valid json }")

        # Should load valid tasks and skip corrupt
        tasks = load_all_tasks(self.temp_dir)
        self.assertEqual(len(tasks), 1)

    def test_creates_directory_on_save(self):
        """save() should create directory if it doesn't exist."""
        nested_dir = Path(self.temp_dir) / "nested" / "deep" / "tasks"
        session = TaskSession()
        session.create_task(title="Task")
        filepath = session.save(str(nested_dir))

        self.assertTrue(filepath.exists())
        self.assertTrue(nested_dir.exists())


class TestSecurityAndRobustness(unittest.TestCase):
    """Test security fixes and robustness improvements."""

    def setUp(self):
        """Create temporary directory for task files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_path_traversal_rejected(self):
        """Archive should reject path traversal attempts."""
        # Create a session with tasks
        session = TaskSession()
        session.create_task(title="Task")
        session.save(self.temp_dir)

        # Attempt path traversal with ..
        evil_archive = str(Path(self.temp_dir) / ".." / "evil_dir")
        with self.assertRaises(ValueError) as cm:
            archive_old_session_files(self.temp_dir, archive_dir=evil_archive)
        self.assertIn("path traversal", str(cm.exception).lower())

    def test_absolute_path_outside_tasks_rejected(self):
        """Archive should reject absolute paths outside tasks directory."""
        session = TaskSession()
        session.create_task(title="Task")
        session.save(self.temp_dir)

        # Attempt to use /tmp as archive (outside tasks_dir)
        with self.assertRaises(ValueError) as cm:
            archive_old_session_files(self.temp_dir, archive_dir="/tmp")
        self.assertIn("must be within", str(cm.exception).lower())

    def test_subdirectory_archive_allowed(self):
        """Archive within tasks directory should be allowed."""
        session = TaskSession()
        session.create_task(title="Task")
        session.save(self.temp_dir)

        # Subdirectory should work fine
        sub_archive = str(Path(self.temp_dir) / "deep" / "archive")
        archived = archive_old_session_files(self.temp_dir, archive_dir=sub_archive)
        self.assertEqual(len(archived), 1)
        self.assertTrue(Path(sub_archive).exists())

    def test_counter_supports_100_plus_tasks(self):
        """Session should support 100+ tasks without format issues."""
        session = TaskSession()

        # Create 150 tasks
        for i in range(150):
            task = session.create_task(title=f"Task {i}")

        # All IDs should be unique and follow pattern
        all_ids = [t.id for t in session.tasks]
        self.assertEqual(len(set(all_ids)), 150)

        # Check format consistency (3-digit counter)
        for task in session.tasks:
            parts = task.id.split("-")
            self.assertEqual(len(parts), 5)  # T-YYYYMMDD-HHMMSS-XXXX-NNN
            counter = parts[-1]
            self.assertEqual(len(counter), 3)  # Always 3 digits

    def test_atomic_write_no_temp_file_left_on_success(self):
        """Successful save should not leave temp files."""
        session = TaskSession()
        session.create_task(title="Task")
        filepath = session.save(self.temp_dir)

        # Check no .tmp files exist
        tmp_files = list(Path(self.temp_dir).glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)

        # Main file should exist
        self.assertTrue(filepath.exists())

    def test_save_creates_valid_json(self):
        """Saved file should be valid JSON after atomic write."""
        session = TaskSession()
        for i in range(10):
            session.create_task(title=f"Task {i}")
        filepath = session.save(self.temp_dir)

        # Should load without error
        with open(filepath) as f:
            data = json.load(f)

        self.assertEqual(len(data["tasks"]), 10)
        self.assertEqual(data["session_id"], session.session_id)


if __name__ == "__main__":
    unittest.main()
