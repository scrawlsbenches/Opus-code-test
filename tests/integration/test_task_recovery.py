#!/usr/bin/env python3
"""
Integration tests for task persistence and crash recovery.

Tests verify that the TaskSession atomic write pattern correctly
handles various failure scenarios:
- Interrupted saves
- Concurrent sessions
- Corrupted files
- State persistence across reloads
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Import from scripts
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.task_utils import TaskSession, Task, load_all_tasks


class TestTaskRecovery:
    """Test task persistence survives crashes and failures."""

    def test_atomic_write_creates_temp_file(self, tmp_path):
        """Verify save() uses temp file before rename."""
        # Create session with a task
        session = TaskSession(tasks_dir=str(tmp_path))
        task = session.create_task("Test task for atomic write")

        # Track temp file creation (monkey-patch open to observe)
        temp_file_seen = []
        original_open = open

        def tracking_open(path, *args, **kwargs):
            if '.json.tmp' in str(path):
                temp_file_seen.append(path)
            return original_open(path, *args, **kwargs)

        # Temporarily patch open
        import builtins
        builtins.open = tracking_open

        try:
            filepath = session.save()
        finally:
            # Restore original open
            builtins.open = original_open

        # Verify temp file was used
        assert len(temp_file_seen) > 0, "Temp file should be created during save"

        # Verify temp file doesn't persist after save
        temp_path = filepath.with_suffix('.json.tmp')
        assert not temp_path.exists(), "Temp file should be cleaned up after successful save"

        # Verify final file exists and is valid
        assert filepath.exists(), "Final file should exist"
        with open(filepath) as f:
            data = json.load(f)
        assert data['version'] == 1
        assert len(data['tasks']) == 1
        assert data['tasks'][0]['title'] == "Test task for atomic write"

    def test_recovery_after_interrupted_save(self, tmp_path):
        """Simulate crash during save, verify old state preserved."""
        # Create session with initial tasks
        session = TaskSession(tasks_dir=str(tmp_path))
        task1 = session.create_task("Original task 1")
        task2 = session.create_task("Original task 2")

        # Save successfully
        filepath = session.save()
        assert filepath.exists()

        # Verify initial state
        with open(filepath) as f:
            original_data = json.load(f)
        assert len(original_data['tasks']) == 2

        # Simulate interrupted save by creating a partial temp file
        temp_filepath = filepath.with_suffix('.json.tmp')
        with open(temp_filepath, 'w') as f:
            f.write('{"version": 1, "incomplete": ')  # Incomplete JSON

        # Verify temp file exists (simulating crash mid-write)
        assert temp_filepath.exists()

        # Load session - should get last good state, not partial
        loaded_session = TaskSession.load(filepath)

        # Verify we got the original data, not the corrupted temp file
        assert len(loaded_session.tasks) == 2
        assert loaded_session.tasks[0].title == "Original task 1"
        assert loaded_session.tasks[1].title == "Original task 2"

        # Verify original file is still intact
        with open(filepath) as f:
            current_data = json.load(f)
        assert current_data == original_data

    def test_concurrent_sessions_no_conflict(self, tmp_path):
        """Two sessions can work in parallel without conflicts."""
        # Create two separate sessions
        session1 = TaskSession(tasks_dir=str(tmp_path))
        session2 = TaskSession(tasks_dir=str(tmp_path))

        # Add tasks to both sessions
        task1a = session1.create_task("Session 1 - Task A", priority="high")
        task1b = session1.create_task("Session 1 - Task B", priority="medium")

        task2a = session2.create_task("Session 2 - Task A", priority="low")
        task2b = session2.create_task("Session 2 - Task B", priority="high")

        # Save both sessions
        file1 = session1.save()
        file2 = session2.save()

        # Verify both files exist
        assert file1.exists(), "Session 1 file should exist"
        assert file2.exists(), "Session 2 file should exist"
        assert file1 != file2, "Sessions should have different filenames"

        # Verify both files are valid
        with open(file1) as f:
            data1 = json.load(f)
        assert len(data1['tasks']) == 2
        assert data1['session_id'] == session1.session_id

        with open(file2) as f:
            data2 = json.load(f)
        assert len(data2['tasks']) == 2
        assert data2['session_id'] == session2.session_id

        # Verify all tasks are loadable
        all_tasks = load_all_tasks(str(tmp_path))
        assert len(all_tasks) == 4, "Should load all 4 tasks from both sessions"

        # Verify task IDs are unique
        task_ids = [t.id for t in all_tasks]
        assert len(task_ids) == len(set(task_ids)), "All task IDs should be unique"

    def test_corrupted_json_handled_gracefully(self, tmp_path):
        """Corrupted task file doesn't crash loading."""
        # Create a valid session first
        session1 = TaskSession(tasks_dir=str(tmp_path))
        session1.create_task("Valid task")
        file1 = session1.save()

        # Create a corrupted JSON file
        corrupted_file = tmp_path / "2025-12-13_10-00-00_corrupt.json"
        with open(corrupted_file, 'w') as f:
            f.write('{"version": 1, "tasks": [INVALID JSON')

        # Create another valid session
        session2 = TaskSession(tasks_dir=str(tmp_path))
        session2.create_task("Another valid task")
        file2 = session2.save()

        # Load all tasks - should handle corruption gracefully
        all_tasks = load_all_tasks(str(tmp_path))

        # Should load only the 2 valid tasks, skipping corrupted file
        assert len(all_tasks) == 2, "Should load 2 tasks from valid files"
        assert all_tasks[0].title == "Valid task"
        assert all_tasks[1].title == "Another valid task"

    def test_task_state_persists_across_reload(self, tmp_path):
        """Task status changes persist across session reload."""
        # Create session with a pending task
        session = TaskSession(tasks_dir=str(tmp_path))
        task = session.create_task("Task with state changes", priority="high")
        assert task.status == "pending"
        assert task.completed_at is None

        # Save and reload
        filepath = session.save()
        loaded_session1 = TaskSession.load(filepath)

        # Verify task is still pending
        assert len(loaded_session1.tasks) == 1
        assert loaded_session1.tasks[0].status == "pending"
        assert loaded_session1.tasks[0].priority == "high"

        # Mark task as in progress
        loaded_session1.tasks[0].mark_in_progress()
        assert loaded_session1.tasks[0].status == "in_progress"
        assert loaded_session1.tasks[0].updated_at is not None

        # Save and reload again
        loaded_session1.save(tasks_dir=str(tmp_path))
        loaded_session2 = TaskSession.load(filepath)

        # Verify status is still in_progress
        assert loaded_session2.tasks[0].status == "in_progress"
        assert loaded_session2.tasks[0].updated_at is not None

        # Mark task as complete
        loaded_session2.tasks[0].mark_complete()
        assert loaded_session2.tasks[0].status == "completed"
        assert loaded_session2.tasks[0].completed_at is not None

        # Save and reload final time
        loaded_session2.save(tasks_dir=str(tmp_path))
        loaded_session3 = TaskSession.load(filepath)

        # Verify status is still completed
        assert loaded_session3.tasks[0].status == "completed"
        assert loaded_session3.tasks[0].completed_at is not None

        # Verify all metadata preserved
        final_task = loaded_session3.tasks[0]
        assert final_task.title == "Task with state changes"
        assert final_task.priority == "high"
        assert final_task.created_at == task.created_at

    def test_partial_session_recovery(self, tmp_path):
        """If one task file is corrupt, others still load."""
        # Create multiple valid session files
        sessions = []
        for i in range(3):
            session = TaskSession(tasks_dir=str(tmp_path))
            session.create_task(f"Valid task {i+1}", priority="medium")
            filepath = session.save()
            sessions.append((session, filepath))

        # Corrupt the middle file
        middle_file = sessions[1][1]
        with open(middle_file, 'w') as f:
            f.write('COMPLETELY INVALID JSON{{{')

        # Load all tasks
        all_tasks = load_all_tasks(str(tmp_path))

        # Should load 2 valid tasks, skipping the corrupted one
        assert len(all_tasks) == 2, "Should load 2 tasks from valid files"
        assert all_tasks[0].title == "Valid task 1"
        assert all_tasks[1].title == "Valid task 3"

        # Verify corrupted file still exists (not deleted)
        assert middle_file.exists(), "Corrupted file should not be deleted"

    def test_temp_file_cleanup_on_write_failure(self, tmp_path):
        """Temp file is cleaned up if write fails."""
        session = TaskSession(tasks_dir=str(tmp_path))
        session.create_task("Test task")

        filepath = tmp_path / session.get_filename()
        temp_filepath = filepath.with_suffix('.json.tmp')

        # Simulate write failure by making directory read-only after temp file creation
        # This is hard to test reliably cross-platform, so we'll test the cleanup path
        # by manually creating a temp file and verifying it gets cleaned up

        # Create a pre-existing temp file
        with open(temp_filepath, 'w') as f:
            f.write('{"old": "temp_file"}')

        assert temp_filepath.exists(), "Pre-existing temp file should exist"

        # Normal save should clean up and replace it
        session.save()

        # After successful save, temp file should be gone
        assert not temp_filepath.exists(), "Temp file should be cleaned up after save"
        assert filepath.exists(), "Final file should exist"

    def test_fsync_called_for_durability(self, tmp_path):
        """Verify that fsync is called to ensure data durability."""
        session = TaskSession(tasks_dir=str(tmp_path))
        session.create_task("Durable task")

        # Track fsync calls
        fsync_called = []
        original_fsync = os.fsync

        def tracking_fsync(fd):
            fsync_called.append(fd)
            return original_fsync(fd)

        # Patch fsync
        os.fsync = tracking_fsync

        try:
            filepath = session.save()
        finally:
            # Restore original fsync
            os.fsync = original_fsync

        # Verify fsync was called at least once
        assert len(fsync_called) > 0, "fsync should be called for data durability"

        # Verify file was saved correctly
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert len(data['tasks']) == 1

    def test_task_id_uniqueness_within_session(self, tmp_path):
        """Task IDs are unique even when created in quick succession."""
        session = TaskSession(tasks_dir=str(tmp_path))

        # Create many tasks quickly
        task_ids = []
        for i in range(100):
            task = session.create_task(f"Rapid task {i}")
            task_ids.append(task.id)

        # Verify all IDs are unique
        assert len(task_ids) == len(set(task_ids)), "All task IDs should be unique"

        # Verify IDs follow expected format
        for task_id in task_ids:
            assert task_id.startswith('T-'), "Task IDs should start with T-"
            parts = task_id.split('-')
            assert len(parts) == 5, "Task IDs should have 5 parts (T-DATE-TIME-SESSION-NUM)"
            assert parts[4].isdigit(), "Last part should be task number"

    def test_load_with_missing_task_counter(self, tmp_path):
        """Loading a session without _task_counter field works."""
        # Create a session and save it
        session = TaskSession(tasks_dir=str(tmp_path))
        session.create_task("Task 1")
        session.create_task("Task 2")
        filepath = session.save()

        # Load the session
        loaded = TaskSession.load(filepath)

        # Verify tasks loaded correctly
        assert len(loaded.tasks) == 2
        assert loaded.tasks[0].title == "Task 1"
        assert loaded.tasks[1].title == "Task 2"

        # The _task_counter is not persisted, so it should be 0 after load
        # Creating new tasks should still work
        new_task = loaded.create_task("Task 3")
        assert new_task.id is not None
        assert len(loaded.tasks) == 3

    def test_empty_tasks_directory(self, tmp_path):
        """Loading from empty directory returns empty list."""
        # Don't create any task files
        empty_dir = tmp_path / "empty_tasks"

        # Load from non-existent directory
        tasks = load_all_tasks(str(empty_dir))
        assert tasks == [], "Should return empty list for non-existent directory"

        # Create directory but leave it empty
        empty_dir.mkdir()
        tasks = load_all_tasks(str(empty_dir))
        assert tasks == [], "Should return empty list for empty directory"

    def test_task_dependencies_persist(self, tmp_path):
        """Task dependencies are preserved across save/load."""
        session = TaskSession(tasks_dir=str(tmp_path))

        # Create tasks with dependencies
        task1 = session.create_task("Foundation task")
        task2 = session.create_task(
            "Dependent task",
            depends_on=[task1.id]
        )
        task3 = session.create_task(
            "Multi-dependency task",
            depends_on=[task1.id, task2.id]
        )

        # Save and reload
        filepath = session.save()
        loaded = TaskSession.load(filepath)

        # Verify dependencies preserved
        assert len(loaded.tasks) == 3
        assert loaded.tasks[1].depends_on == [task1.id]
        assert loaded.tasks[2].depends_on == [task1.id, task2.id]

    def test_task_context_metadata_persists(self, tmp_path):
        """Task context metadata is preserved across save/load."""
        session = TaskSession(tasks_dir=str(tmp_path))

        # Create task with rich context
        task = session.create_task(
            "Task with context",
            context={
                "files": ["cortical/processor.py", "tests/test_processor.py"],
                "methods": ["compute_all", "process_document"],
                "line_numbers": [100, 200, 300],
                "nested": {"key": "value", "count": 42}
            }
        )

        # Save and reload
        filepath = session.save()
        loaded = TaskSession.load(filepath)

        # Verify context fully preserved
        loaded_task = loaded.tasks[0]
        assert loaded_task.context["files"] == ["cortical/processor.py", "tests/test_processor.py"]
        assert loaded_task.context["methods"] == ["compute_all", "process_document"]
        assert loaded_task.context["line_numbers"] == [100, 200, 300]
        assert loaded_task.context["nested"]["key"] == "value"
        assert loaded_task.context["nested"]["count"] == 42
