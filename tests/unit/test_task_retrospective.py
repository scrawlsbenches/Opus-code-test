#!/usr/bin/env python3
"""
Tests for task retrospective metadata capture.

This module tests the retrospective tracking features added to the task
management system, which capture metadata about completed tasks such as
files touched, duration, and tests added.
"""

import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scripts.task_utils import Task, TaskSession


class TestTaskRetrospective(unittest.TestCase):
    """Tests for retrospective metadata capture on tasks."""

    def setUp(self):
        """Set up test session."""
        self.session = TaskSession()

    def test_task_has_retrospective_field(self):
        """Task dataclass includes retrospective field."""
        task = Task(
            id="T-test-001",
            title="Test task"
        )
        self.assertIsNone(task.retrospective)
        self.assertTrue(hasattr(task, 'retrospective'))

    def test_capture_retrospective_basic(self):
        """capture_retrospective stores data correctly."""
        task = self.session.create_task("Implement feature X")
        task_id = task.id

        # Capture retrospective data
        self.session.capture_retrospective(
            task_id=task_id,
            files_touched=['cortical/processor.py', 'tests/test_processor.py'],
            tests_added=3,
            commits=['abc1234', 'def5678'],
            notes='Added new search algorithm'
        )

        # Verify data was stored
        task = self.session.get_task(task_id)
        self.assertIsNotNone(task.retrospective)
        self.assertEqual(
            task.retrospective['files_touched'],
            ['cortical/processor.py', 'tests/test_processor.py']
        )
        self.assertEqual(task.retrospective['tests_added'], 3)
        self.assertEqual(task.retrospective['commits'], ['abc1234', 'def5678'])
        self.assertEqual(task.retrospective['notes'], 'Added new search algorithm')
        self.assertIsInstance(task.retrospective['duration_minutes'], int)
        self.assertIn('captured_at', task.retrospective)

    def test_capture_retrospective_with_defaults(self):
        """capture_retrospective works with minimal parameters."""
        task = self.session.create_task("Simple task")
        task_id = task.id

        # Capture with only task_id
        self.session.capture_retrospective(task_id=task_id)

        task = self.session.get_task(task_id)
        self.assertIsNotNone(task.retrospective)
        self.assertEqual(task.retrospective['files_touched'], [])
        self.assertEqual(task.retrospective['tests_added'], 0)
        self.assertEqual(task.retrospective['commits'], [])
        self.assertIsNone(task.retrospective['notes'])
        self.assertGreaterEqual(task.retrospective['duration_minutes'], 0)

    def test_capture_retrospective_invalid_task_id(self):
        """capture_retrospective raises ValueError for unknown task."""
        with self.assertRaises(ValueError) as ctx:
            self.session.capture_retrospective(task_id="T-nonexistent-001")

        self.assertIn("Task not found", str(ctx.exception))

    def test_duration_calculation(self):
        """Duration is calculated correctly from task creation."""
        task = self.session.create_task("Time test")
        task_id = task.id

        # Simulate some time passing (small delay for testing)
        time.sleep(0.1)

        self.session.capture_retrospective(task_id=task_id)

        task = self.session.get_task(task_id)
        # Duration should be >= 0 minutes (rounds down for sub-minute tasks)
        self.assertGreaterEqual(task.retrospective['duration_minutes'], 0)
        self.assertIsInstance(task.retrospective['duration_minutes'], int)

    def test_duration_calculation_edge_case(self):
        """Duration calculation with manually set old created_at."""
        # Create task with old timestamp
        task = Task(
            id="T-old-001",
            title="Old task",
            created_at=(datetime.now() - timedelta(hours=2, minutes=30)).isoformat()
        )
        self.session.tasks.append(task)

        self.session.capture_retrospective(task_id=task.id)

        task = self.session.get_task(task.id)
        # Should be approximately 150 minutes (2.5 hours)
        duration = task.retrospective['duration_minutes']
        self.assertGreater(duration, 145)  # Allow some margin
        self.assertLess(duration, 155)

    def test_get_retrospective_summary_empty(self):
        """get_retrospective_summary returns correct defaults for empty session."""
        summary = self.session.get_retrospective_summary()

        self.assertEqual(summary['total_completed'], 0)
        self.assertEqual(summary['avg_duration_minutes'], 0)
        self.assertEqual(summary['total_duration_minutes'], 0)
        self.assertEqual(summary['total_tests_added'], 0)
        self.assertEqual(summary['most_touched_files'], [])
        self.assertEqual(summary['tasks_with_retrospective'], [])

    def test_get_retrospective_summary_single_task(self):
        """get_retrospective_summary aggregates single task correctly."""
        task = self.session.create_task("Task 1")
        task.status = "completed"

        self.session.capture_retrospective(
            task_id=task.id,
            files_touched=['file1.py', 'file2.py'],
            tests_added=5,
            commits=['abc123']
        )

        summary = self.session.get_retrospective_summary()

        self.assertEqual(summary['total_completed'], 1)
        self.assertGreaterEqual(summary['avg_duration_minutes'], 0)
        self.assertEqual(summary['total_tests_added'], 5)
        self.assertEqual(len(summary['most_touched_files']), 2)
        self.assertEqual(summary['tasks_with_retrospective'], [task.id])

    def test_get_retrospective_summary_multiple_tasks(self):
        """get_retrospective_summary aggregates multiple tasks correctly."""
        # Create and complete multiple tasks
        task1 = self.session.create_task("Task 1")
        task1.status = "completed"
        self.session.capture_retrospective(
            task_id=task1.id,
            files_touched=['file1.py', 'file2.py'],
            tests_added=3
        )

        task2 = self.session.create_task("Task 2")
        task2.status = "completed"
        self.session.capture_retrospective(
            task_id=task2.id,
            files_touched=['file1.py', 'file3.py'],
            tests_added=7
        )

        task3 = self.session.create_task("Task 3")
        task3.status = "completed"
        self.session.capture_retrospective(
            task_id=task3.id,
            files_touched=['file2.py'],
            tests_added=2
        )

        summary = self.session.get_retrospective_summary()

        self.assertEqual(summary['total_completed'], 3)
        self.assertEqual(summary['total_tests_added'], 12)  # 3 + 7 + 2

        # Check file counts
        file_counts = dict(summary['most_touched_files'])
        self.assertEqual(file_counts['file1.py'], 2)  # in task1 and task2
        self.assertEqual(file_counts['file2.py'], 2)  # in task1 and task3
        self.assertEqual(file_counts['file3.py'], 1)  # only in task2

        self.assertEqual(len(summary['tasks_with_retrospective']), 3)

    def test_get_retrospective_summary_ignores_incomplete_tasks(self):
        """get_retrospective_summary only includes completed tasks."""
        # Create pending task with retrospective (shouldn't happen in practice)
        task1 = self.session.create_task("Pending task")
        task1.status = "pending"
        self.session.capture_retrospective(task_id=task1.id, tests_added=5)

        # Create completed task without retrospective
        task2 = self.session.create_task("Completed without retro")
        task2.status = "completed"

        # Create completed task with retrospective
        task3 = self.session.create_task("Completed with retro")
        task3.status = "completed"
        self.session.capture_retrospective(task_id=task3.id, tests_added=3)

        summary = self.session.get_retrospective_summary()

        # Only task3 should be counted
        self.assertEqual(summary['total_completed'], 1)
        self.assertEqual(summary['total_tests_added'], 3)
        self.assertEqual(summary['tasks_with_retrospective'], [task3.id])

    def test_retrospective_persists_across_save_load(self):
        """Retrospective data survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task with retrospective
            task = self.session.create_task("Persistent task")
            task.status = "completed"
            self.session.capture_retrospective(
                task_id=task.id,
                files_touched=['file1.py', 'file2.py'],
                tests_added=5,
                commits=['abc123'],
                notes='Test persistence'
            )

            # Save session
            saved_path = self.session.save(tasks_dir=tmpdir)
            self.assertTrue(saved_path.exists())

            # Load session
            loaded_session = TaskSession.load(saved_path)
            loaded_task = loaded_session.get_task(task.id)

            # Verify retrospective data
            self.assertIsNotNone(loaded_task.retrospective)
            self.assertEqual(
                loaded_task.retrospective['files_touched'],
                ['file1.py', 'file2.py']
            )
            self.assertEqual(loaded_task.retrospective['tests_added'], 5)
            self.assertEqual(loaded_task.retrospective['commits'], ['abc123'])
            self.assertEqual(loaded_task.retrospective['notes'], 'Test persistence')

    def test_backward_compatibility_tasks_without_retrospective(self):
        """Tasks without retrospective field load correctly (backward compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually create a task file without retrospective field
            task_data = {
                "version": 1,
                "session_id": "test",
                "started_at": datetime.now().isoformat(),
                "saved_at": datetime.now().isoformat(),
                "tasks": [
                    {
                        "id": "T-old-001",
                        "title": "Old task without retrospective",
                        "status": "completed",
                        "priority": "medium",
                        "category": "general",
                        "description": "",
                        "depends_on": [],
                        "effort": "medium",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": None,
                        "completed_at": None,
                        "context": {}
                        # Note: no retrospective field
                    }
                ]
            }

            filepath = Path(tmpdir) / "old_session.json"
            with open(filepath, 'w') as f:
                json.dump(task_data, f, indent=2)

            # Load and verify
            loaded_session = TaskSession.load(filepath)
            loaded_task = loaded_session.tasks[0]

            self.assertEqual(loaded_task.id, "T-old-001")
            self.assertIsNone(loaded_task.retrospective)  # Should default to None

    def test_to_dict_includes_retrospective(self):
        """Task.to_dict() includes retrospective field."""
        task = Task(
            id="T-test-001",
            title="Test task"
        )

        # Without retrospective
        task_dict = task.to_dict()
        self.assertIn('retrospective', task_dict)
        self.assertIsNone(task_dict['retrospective'])

        # With retrospective
        task.retrospective = {
            'files_touched': ['file1.py'],
            'duration_minutes': 30,
            'tests_added': 2,
            'commits': ['abc123'],
            'notes': 'Done',
            'captured_at': datetime.now().isoformat()
        }
        task_dict = task.to_dict()
        self.assertIsNotNone(task_dict['retrospective'])
        self.assertEqual(task_dict['retrospective']['tests_added'], 2)

    def test_from_dict_handles_retrospective(self):
        """Task.from_dict() correctly reconstructs retrospective."""
        task_dict = {
            'id': 'T-test-001',
            'title': 'Test task',
            'status': 'completed',
            'priority': 'high',
            'category': 'feature',
            'description': 'Test description',
            'depends_on': [],
            'effort': 'medium',
            'created_at': datetime.now().isoformat(),
            'updated_at': None,
            'completed_at': None,
            'context': {},
            'retrospective': {
                'files_touched': ['file1.py', 'file2.py'],
                'duration_minutes': 45,
                'tests_added': 5,
                'commits': ['abc123', 'def456'],
                'notes': 'All done',
                'captured_at': datetime.now().isoformat()
            }
        }

        task = Task.from_dict(task_dict)

        self.assertEqual(task.id, 'T-test-001')
        self.assertIsNotNone(task.retrospective)
        self.assertEqual(task.retrospective['files_touched'], ['file1.py', 'file2.py'])
        self.assertEqual(task.retrospective['duration_minutes'], 45)
        self.assertEqual(task.retrospective['tests_added'], 5)

    def test_most_touched_files_ordering(self):
        """most_touched_files returns files in descending order by count."""
        # Create tasks touching files different amounts
        for i in range(5):
            task = self.session.create_task(f"Task {i}")
            task.status = "completed"
            # file1.py touched 5 times, file2.py 4 times, etc.
            files = ['file1.py'] * (5 - i)
            if i < 4:
                files.append('file2.py')
            if i < 3:
                files.append('file3.py')
            self.session.capture_retrospective(
                task_id=task.id,
                files_touched=files
            )

        summary = self.session.get_retrospective_summary()
        most_touched = summary['most_touched_files']

        # Should be ordered by count descending
        self.assertGreater(len(most_touched), 0)
        counts = [count for _, count in most_touched]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_get_task_returns_none_for_unknown_id(self):
        """get_task returns None for unknown task ID."""
        result = self.session.get_task("T-nonexistent-001")
        self.assertIsNone(result)

    def test_get_task_finds_existing_task(self):
        """get_task finds task by ID."""
        task = self.session.create_task("Findable task")
        found = self.session.get_task(task.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, task.id)
        self.assertEqual(found.title, "Findable task")


class TestRetrospectiveDurationCalculation(unittest.TestCase):
    """Focused tests for duration calculation logic."""

    def test_calculate_duration_basic(self):
        """_calculate_duration computes correct minutes."""
        session = TaskSession()

        # 1 hour ago
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        duration = session._calculate_duration(one_hour_ago)
        self.assertGreater(duration, 55)  # Allow some margin
        self.assertLess(duration, 65)

    def test_calculate_duration_rounds(self):
        """_calculate_duration rounds to nearest minute."""
        session = TaskSession()

        # 90 seconds ago (1.5 minutes)
        ninety_sec_ago = (datetime.now() - timedelta(seconds=90)).isoformat()
        duration = session._calculate_duration(ninety_sec_ago)
        # Should round to 2 minutes
        self.assertIn(duration, [1, 2])  # Could be 1 or 2 depending on rounding

    def test_calculate_duration_zero(self):
        """_calculate_duration handles immediate calculation."""
        session = TaskSession()

        # Right now
        now = datetime.now().isoformat()
        duration = session._calculate_duration(now)
        self.assertEqual(duration, 0)


if __name__ == '__main__':
    unittest.main()
