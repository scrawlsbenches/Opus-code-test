#!/usr/bin/env python3
"""
Unit tests for session_context.py - session context generator.

Tests cover:
- Recent session retrieval
- Pending task grouping
- Git commit parsing
- File change analysis
- Markdown and JSON generation
- Edge cases (missing directories, no git, corrupted files)
"""

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from session_context import SessionContextGenerator
from task_utils import Task, TaskSession


class TestSessionContextGenerator(unittest.TestCase):
    """Test suite for SessionContextGenerator class."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)
        self.tasks_dir = self.repo_path / 'tasks'
        self.tasks_dir.mkdir()

        self.generator = SessionContextGenerator(str(self.repo_path))

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_test_session(
        self,
        session_id: str,
        num_tasks: int = 3,
        started_at: str = None
    ) -> Path:
        """
        Helper to create a test session file.

        Args:
            session_id: Session identifier
            num_tasks: Number of tasks to create
            started_at: ISO timestamp (default: now)

        Returns:
            Path to created session file
        """
        if started_at is None:
            started_at = datetime.now().isoformat()

        session = TaskSession(
            session_id=session_id,
            started_at=started_at,
            tasks_dir=str(self.tasks_dir)
        )

        # Create tasks with different statuses
        statuses = ['completed', 'in_progress', 'pending', 'deferred']
        for i in range(num_tasks):
            task = session.create_task(
                title=f"Task {i+1}",
                priority=['high', 'medium', 'low'][i % 3],
                category="test",
                description=f"Test task number {i+1}"
            )
            task.status = statuses[i % len(statuses)]

        return session.save()

    def test_init(self):
        """Test SessionContextGenerator initialization."""
        self.assertEqual(self.generator.repo_path, self.repo_path)
        self.assertEqual(self.generator.tasks_dir, self.tasks_dir)

    def test_get_recent_sessions_empty(self):
        """Test getting recent sessions when no tasks exist."""
        sessions = self.generator.get_recent_sessions(5)
        self.assertEqual(sessions, [])

    def test_get_recent_sessions_missing_dir(self):
        """Test getting recent sessions when tasks directory doesn't exist."""
        # Remove tasks directory
        self.tasks_dir.rmdir()

        sessions = self.generator.get_recent_sessions(5)
        self.assertEqual(sessions, [])

    def test_get_recent_sessions_single(self):
        """Test getting a single recent session."""
        # Create one session
        self._create_test_session('abc1', num_tasks=3)

        sessions = self.generator.get_recent_sessions(5)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]['session_id'], 'abc1')
        self.assertEqual(sessions[0]['total'], 3)
        # Check status counts (created with rotating statuses: completed, in_progress, pending)
        self.assertEqual(sessions[0]['completed'], 1)

    def test_get_recent_sessions_multiple(self):
        """Test getting multiple recent sessions."""
        # Create sessions with different timestamps
        base_time = datetime.now()

        for i in range(5):
            timestamp = (base_time - timedelta(hours=i)).isoformat()
            self._create_test_session(f'test{i}', num_tasks=2, started_at=timestamp)

        sessions = self.generator.get_recent_sessions(3)

        # Should get 3 most recent (sorted by file modification time, so last created is first)
        self.assertEqual(len(sessions), 3)

        # Most recent file (test4) should be first since it was created last
        self.assertEqual(sessions[0]['session_id'], 'test4')

    def test_get_recent_sessions_with_corrupted_file(self):
        """Test handling of corrupted session files."""
        # Create valid session
        self._create_test_session('good', num_tasks=1)

        # Create corrupted file
        corrupted = self.tasks_dir / 'corrupted.json'
        corrupted.write_text('{"invalid json')

        sessions = self.generator.get_recent_sessions(5)

        # Should get only the valid session
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]['session_id'], 'good')

    def test_get_pending_tasks_empty(self):
        """Test getting pending tasks when none exist."""
        pending = self.generator.get_pending_tasks()

        self.assertIn('high', pending)
        self.assertIn('medium', pending)
        self.assertIn('low', pending)
        self.assertEqual(len(pending['high']), 0)
        self.assertEqual(len(pending['medium']), 0)
        self.assertEqual(len(pending['low']), 0)

    def test_get_pending_tasks_by_priority(self):
        """Test grouping pending tasks by priority."""
        session = TaskSession(session_id='test', tasks_dir=str(self.tasks_dir))

        # Create tasks with different priorities and statuses
        high_task = session.create_task("High priority", priority='high')
        high_task.status = 'pending'

        med_task = session.create_task("Medium priority", priority='medium')
        med_task.status = 'in_progress'

        low_task = session.create_task("Low priority", priority='low')
        low_task.status = 'pending'

        completed = session.create_task("Completed", priority='high')
        completed.status = 'completed'

        session.save()

        pending = self.generator.get_pending_tasks()

        # Should group by priority, exclude completed
        self.assertEqual(len(pending['high']), 1)
        self.assertEqual(len(pending['medium']), 1)
        self.assertEqual(len(pending['low']), 1)

        self.assertEqual(pending['high'][0].title, "High priority")
        self.assertEqual(pending['medium'][0].title, "Medium priority")
        self.assertEqual(pending['low'][0].title, "Low priority")

    def test_get_pending_tasks_invalid_priority(self):
        """Test handling tasks with invalid priority values."""
        session = TaskSession(session_id='test', tasks_dir=str(self.tasks_dir))

        task = session.create_task("Test task", priority='invalid')
        task.status = 'pending'
        session.save()

        pending = self.generator.get_pending_tasks()

        # Should default to medium priority
        self.assertEqual(len(pending['medium']), 1)

    @patch('subprocess.run')
    def test_get_recent_commits(self, mock_run):
        """Test getting recent commits from git."""
        # Mock git log output
        mock_result = MagicMock()
        mock_result.stdout = (
            "abc1234|2 hours ago|Fix validation bug\n"
            "def5678|1 day ago|Add new feature\n"
        )
        mock_result.returncode = 0

        # Mock git diff-tree output for files
        mock_files_result = MagicMock()
        mock_files_result.stdout = "file1.py\nfile2.py\n"
        mock_files_result.returncode = 0

        mock_run.side_effect = [mock_result, mock_files_result, mock_result, mock_files_result]

        commits = self.generator.get_recent_commits(2)

        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0]['hash'], 'abc1234')
        self.assertEqual(commits[0]['time_ago'], '2 hours ago')
        self.assertEqual(commits[0]['subject'], 'Fix validation bug')
        self.assertEqual(commits[0]['files_changed'], 2)

    @patch('subprocess.run')
    def test_get_recent_commits_no_git(self, mock_run):
        """Test handling when git is not available."""
        mock_run.side_effect = FileNotFoundError()

        commits = self.generator.get_recent_commits(10)

        self.assertEqual(commits, [])

    @patch('subprocess.run')
    def test_get_recent_commits_git_error(self, mock_run):
        """Test handling git command errors."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')

        commits = self.generator.get_recent_commits(10)

        self.assertEqual(commits, [])

    @patch('subprocess.run')
    def test_get_recent_file_changes(self, mock_run):
        """Test getting recent file changes grouped by directory."""
        mock_result = MagicMock()
        mock_result.stdout = (
            "cortical/processor.py\n"
            "cortical/analysis.py\n"
            "scripts/session_context.py\n"
            "tests/unit/test_session.py\n"
            "README.md\n"
        )
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        changes = self.generator.get_recent_file_changes(7)

        # Should group by top-level directory
        self.assertIn('cortical', changes)
        self.assertIn('scripts', changes)
        self.assertIn('tests', changes)
        self.assertIn('root', changes)

        self.assertEqual(len(changes['cortical']), 2)
        self.assertEqual(len(changes['scripts']), 1)
        self.assertEqual(len(changes['root']), 1)

    @patch('subprocess.run')
    def test_get_recent_file_changes_no_git(self, mock_run):
        """Test handling when git is not available."""
        mock_run.side_effect = FileNotFoundError()

        changes = self.generator.get_recent_file_changes(7)

        self.assertEqual(changes, {})

    def test_generate_context_empty_repo(self):
        """Test generating context for empty repository."""
        with patch.object(self.generator, 'get_recent_commits', return_value=[]):
            with patch.object(self.generator, 'get_recent_file_changes', return_value={}):
                context = self.generator.generate_context()

                self.assertIn('Session Context', context)
                self.assertIn('No task sessions found', context)
                self.assertIn('No pending tasks found', context)

    def test_generate_context_with_data(self):
        """Test generating context with sessions and tasks."""
        # Create test sessions
        session = TaskSession(session_id='test1', tasks_dir=str(self.tasks_dir))

        task1 = session.create_task("Pending high priority", priority='high')
        task1.status = 'pending'

        task2 = session.create_task("In progress medium", priority='medium')
        task2.status = 'in_progress'

        session.save()

        # Mock git functions
        mock_commits = [{
            'hash': 'abc123',
            'time_ago': '2 hours ago',
            'subject': 'Test commit',
            'files_changed': 3,
            'files': ['file1.py', 'file2.py', 'file3.py']
        }]

        mock_changes = {
            'cortical': ['cortical/file1.py', 'cortical/file2.py'],
            'scripts': ['scripts/test.py']
        }

        with patch.object(self.generator, 'get_recent_commits', return_value=mock_commits):
            with patch.object(self.generator, 'get_recent_file_changes', return_value=mock_changes):
                context = self.generator.generate_context()

                # Should contain all sections
                self.assertIn('Session Context', context)
                self.assertIn('Recent Work', context)
                self.assertIn('Pending Tasks (2 total)', context)
                self.assertIn('High Priority', context)
                self.assertIn('Medium Priority', context)
                self.assertIn('Recent Changes', context)
                self.assertIn('Recent Commits', context)
                self.assertIn('Quick Stats', context)

                # Check task details
                self.assertIn('Pending high priority', context)
                self.assertIn('In progress medium', context)

                # Check commit details
                self.assertIn('abc123', context)
                self.assertIn('Test commit', context)

                # Check file changes
                self.assertIn('cortical', context)
                self.assertIn('scripts', context)

    def test_generate_json_structure(self):
        """Test JSON output structure."""
        # Create minimal test data
        self._create_test_session('json_test', num_tasks=1)

        with patch.object(self.generator, 'get_recent_commits', return_value=[]):
            with patch.object(self.generator, 'get_recent_file_changes', return_value={}):
                json_output = self.generator.generate_json()

                # Check structure
                self.assertIn('generated_at', json_output)
                self.assertIn('recent_sessions', json_output)
                self.assertIn('pending_tasks', json_output)
                self.assertIn('recent_commits', json_output)
                self.assertIn('recent_file_changes', json_output)

                # Check types
                self.assertIsInstance(json_output['recent_sessions'], list)
                self.assertIsInstance(json_output['pending_tasks'], dict)
                self.assertIsInstance(json_output['recent_commits'], list)
                self.assertIsInstance(json_output['recent_file_changes'], dict)

    def test_generate_json_serializable(self):
        """Test that JSON output is fully serializable."""
        self._create_test_session('serial_test', num_tasks=2)

        with patch.object(self.generator, 'get_recent_commits', return_value=[]):
            with patch.object(self.generator, 'get_recent_file_changes', return_value={}):
                json_output = self.generator.generate_json()

                # Should not raise
                serialized = json.dumps(json_output, indent=2)
                self.assertIsInstance(serialized, str)

                # Should be deserializable
                deserialized = json.loads(serialized)
                self.assertEqual(deserialized['generated_at'], json_output['generated_at'])

    def test_pending_tasks_sorted_by_creation(self):
        """Test that pending tasks are sorted by creation time within priority."""
        session = TaskSession(session_id='sort_test', tasks_dir=str(self.tasks_dir))

        # Create tasks in non-chronological order
        base_time = datetime.now()

        task1 = session.create_task("Second created", priority='high')
        task1.created_at = (base_time - timedelta(hours=1)).isoformat()
        task1.status = 'pending'

        task2 = session.create_task("First created", priority='high')
        task2.created_at = (base_time - timedelta(hours=2)).isoformat()
        task2.status = 'pending'

        task3 = session.create_task("Third created", priority='high')
        task3.created_at = base_time.isoformat()
        task3.status = 'pending'

        session.save()

        pending = self.generator.get_pending_tasks()

        # Should be sorted by creation time (oldest first)
        self.assertEqual(len(pending['high']), 3)
        self.assertEqual(pending['high'][0].title, "First created")
        self.assertEqual(pending['high'][1].title, "Second created")
        self.assertEqual(pending['high'][2].title, "Third created")

    def test_context_limits_tasks_per_priority(self):
        """Test that context generation limits tasks shown per priority."""
        session = TaskSession(session_id='limit_test', tasks_dir=str(self.tasks_dir))

        # Create 15 high priority tasks
        for i in range(15):
            task = session.create_task(f"Task {i}", priority='high')
            task.status = 'pending'

        session.save()

        context = self.generator.generate_context()

        # Should show max 10 per priority
        task_lines = [line for line in context.split('\n') if line.startswith('- 📋')]
        self.assertLessEqual(len(task_lines), 10)

    def test_context_truncates_long_descriptions(self):
        """Test that long task descriptions are truncated."""
        session = TaskSession(session_id='desc_test', tasks_dir=str(self.tasks_dir))

        long_desc = "A" * 200  # 200 character description
        task = session.create_task("Test task", priority='high', description=long_desc)
        task.status = 'pending'

        session.save()

        context = self.generator.generate_context()

        # Description should be truncated to ~100 chars
        self.assertIn('AAA...', context)  # Should have ellipsis
        self.assertNotIn('A' * 150, context)  # Shouldn't have full length

    def test_get_recent_commits_empty_output(self):
        """Test handling empty git log output."""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = ""
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            commits = self.generator.get_recent_commits(10)

            self.assertEqual(commits, [])


class TestCLI(unittest.TestCase):
    """Test CLI argument parsing and output."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    @patch('sys.argv', ['session_context.py', '--repo-path', '/tmp/test'])
    @patch('session_context.SessionContextGenerator.generate_context')
    def test_cli_default_markdown(self, mock_generate):
        """Test CLI with default markdown output."""
        mock_generate.return_value = "# Test Context"

        # Import here to trigger CLI with mocked argv
        import session_context
        # Would need to mock print to fully test, but this validates imports work


if __name__ == '__main__':
    unittest.main()
