#!/usr/bin/env python3
"""
Unit tests for session memory generator.

Tests Sprint-2.3 (SessionEnd auto-memory generation) and Sprint-2.4 (post-commit task linking).
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Import the module we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.session_memory_generator import (
    SessionMemoryGenerator,
    SessionData,
    CommitInfo,
    extract_task_ids,
    link_commit_to_tasks,
)


class TestExtractTaskIds(unittest.TestCase):
    """Test task ID extraction from text."""

    def test_extract_old_format(self):
        """Test extraction of old format task IDs (without microseconds)."""
        text = "Fix bug in task T-20251213-143052-a1b2"
        task_ids = extract_task_ids(text)
        self.assertEqual(task_ids, ['T-20251213-143052-a1b2'])

    def test_extract_new_format(self):
        """Test extraction of new format task IDs (with microseconds)."""
        text = "Complete T-20251213-143052123456-a1b2"
        task_ids = extract_task_ids(text)
        self.assertEqual(task_ids, ['T-20251213-143052123456-a1b2'])

    def test_extract_session_task(self):
        """Test extraction of session task IDs (with -NNN suffix)."""
        text = "Working on T-20251213-143052123456-a1b2-001"
        task_ids = extract_task_ids(text)
        self.assertEqual(task_ids, ['T-20251213-143052123456-a1b2-001'])

    def test_extract_multiple_tasks(self):
        """Test extraction of multiple task IDs."""
        text = """
        Complete T-20251213-143052-a1b2 and T-20251214-153045-b2c3
        Also related to T-20251215-163038-c3d4-001
        """
        task_ids = extract_task_ids(text)
        self.assertEqual(len(task_ids), 3)
        self.assertIn('T-20251213-143052-a1b2', task_ids)
        self.assertIn('T-20251214-153045-b2c3', task_ids)
        self.assertIn('T-20251215-163038-c3d4-001', task_ids)

    def test_extract_no_tasks(self):
        """Test extraction when no task IDs present."""
        text = "This is a commit message with no task IDs"
        task_ids = extract_task_ids(text)
        self.assertEqual(task_ids, [])

    def test_extract_case_insensitive(self):
        """Test extraction is case insensitive."""
        text = "Fix t-20251213-143052-a1b2"
        task_ids = extract_task_ids(text)
        self.assertEqual(len(task_ids), 1)

    def test_extract_duplicates_removed(self):
        """Test that duplicate task IDs are removed."""
        text = "T-20251213-143052-a1b2 and T-20251213-143052-a1b2 again"
        task_ids = extract_task_ids(text)
        self.assertEqual(len(task_ids), 1)


class TestCommitInfo(unittest.TestCase):
    """Test CommitInfo data class and methods."""

    @patch('subprocess.run')
    def test_from_git_log_success(self, mock_run):
        """Test creating CommitInfo from git log."""
        # Mock git show output
        show_output = """abc123def456
abc123d
John Doe
2025-12-17T10:30:00-08:00
feat: Add new feature
This is the commit body.
Task: T-20251217-103000-a1b2"""

        stat_output = """cortical/processor.py | 10 +++++-----
tests/test_processor.py | 5 +++--
"""

        mock_run.side_effect = [
            MagicMock(stdout=show_output, returncode=0),
            MagicMock(stdout=stat_output, returncode=0),
        ]

        commit = CommitInfo.from_git_log('abc123')

        self.assertIsNotNone(commit)
        self.assertEqual(commit.sha, 'abc123def456')
        self.assertEqual(commit.short_sha, 'abc123d')
        self.assertEqual(commit.author, 'John Doe')
        self.assertIn('feat: Add new feature', commit.message)
        self.assertEqual(len(commit.files_changed), 2)
        self.assertIn('cortical/processor.py', commit.files_changed)
        self.assertEqual(len(commit.task_ids), 1)
        self.assertIn('T-20251217-103000-a1b2', commit.task_ids)

    @patch('subprocess.run')
    def test_from_git_log_failure(self, mock_run):
        """Test handling of git command failure."""
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, 'git')
        commit = CommitInfo.from_git_log('invalid_sha')
        self.assertIsNone(commit)


class TestSessionMemoryGenerator(unittest.TestCase):
    """Test SessionMemoryGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.git_ml_dir = Path(self.temp_dir) / '.git-ml'
        self.sessions_dir = self.git_ml_dir / 'sessions'
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def test_init(self):
        """Test generator initialization."""
        gen = SessionMemoryGenerator(session_id='test123', git_ml_dir=str(self.git_ml_dir))
        self.assertEqual(gen.session_id, 'test123')
        self.assertEqual(gen.git_ml_dir, self.git_ml_dir)

    def test_load_session_metadata(self):
        """Test loading session metadata from .git-ml/sessions/."""
        # Create a session file
        session_data = {
            'id': 'test123',
            'started_at': '2025-12-17T10:00:00',
            'ended_at': '2025-12-17T11:00:00',
            'chat_ids': ['chat1', 'chat2'],
            'action_ids': ['action1'],
            'summary': 'Test session'
        }
        session_file = self.sessions_dir / '2025-12-17_test123.json'
        with open(session_file, 'w') as f:
            json.dump(session_data, f)

        gen = SessionMemoryGenerator(session_id='test123', git_ml_dir=str(self.git_ml_dir))
        result = SessionData(session_id='test123')
        result = gen._load_session_metadata(result)

        self.assertEqual(result.started_at, '2025-12-17T10:00:00')
        self.assertEqual(result.ended_at, '2025-12-17T11:00:00')
        self.assertEqual(result.chat_ids, ['chat1', 'chat2'])
        self.assertEqual(result.summary, 'Test session')

    def test_extract_topic(self):
        """Test topic extraction from session data."""
        gen = SessionMemoryGenerator(session_id='test')

        # Test with summary
        session_data = SessionData(session_id='test', summary='Session: Refactoring processor')
        topic = gen._extract_topic(session_data)
        self.assertIn('Refactoring', topic)

        # Test with commits
        commit = CommitInfo(
            sha='abc123',
            short_sha='abc',
            message='feat: Add memory generation feature',
            author='Test',
            date='2025-12-17'
        )
        session_data = SessionData(session_id='test', commits=[commit])
        topic = gen._extract_topic(session_data)
        self.assertTrue(len(topic) > 0)

    def test_generate_tags(self):
        """Test tag generation from session data."""
        gen = SessionMemoryGenerator(session_id='test')

        commit = CommitInfo(
            sha='abc123',
            short_sha='abc',
            message='fix: Fix bug in processor',
            author='Test',
            date='2025-12-17'
        )
        session_data = SessionData(
            session_id='test',
            commits=[commit],
            all_files_modified={'cortical/processor.py', 'tests/test_processor.py', 'docs/README.md'}
        )

        tags = gen._generate_tags(session_data)

        self.assertIn('session', tags)
        self.assertIn('python', tags)
        self.assertIn('docs', tags)
        self.assertIn('testing', tags)
        self.assertIn('bugfix', tags)

    def test_categorize_files(self):
        """Test file categorization."""
        gen = SessionMemoryGenerator(session_id='test')

        files = {
            'cortical/processor.py',
            'tests/test_processor.py',
            'scripts/session_memory_generator.py',
            'docs/README.md',
            'samples/memories/test.md',
            '.gitignore',
            'README.md'
        }

        categorized = gen._categorize_files(files)

        self.assertIn('Core Library', categorized)
        self.assertIn('Tests', categorized)
        self.assertIn('Scripts', categorized)
        self.assertIn('Documentation', categorized)
        self.assertIn('Samples', categorized)
        self.assertIn('Configuration', categorized)

        self.assertIn('cortical/processor.py', categorized['Core Library'])
        self.assertIn('tests/test_processor.py', categorized['Tests'])

    def test_generate_draft_memory(self):
        """Test draft memory generation."""
        gen = SessionMemoryGenerator(session_id='test123')

        commit = CommitInfo(
            sha='abc123def456',
            short_sha='abc123d',
            message='feat: Add session memory generator',
            author='Test User',
            date='2025-12-17T10:00:00',
            files_changed=['scripts/session_memory_generator.py'],
            task_ids=['T-20251217-100000-a1b2']
        )

        session_data = SessionData(
            session_id='test123',
            started_at='2025-12-17T10:00:00',
            commits=[commit],
            all_files_modified={'scripts/session_memory_generator.py'},
            task_ids_referenced={'T-20251217-100000-a1b2'}
        )

        memory = gen.generate_draft_memory(session_data)

        # Verify structure
        self.assertIn('# Memory Entry:', memory)
        self.assertIn('Session ID:', memory)
        self.assertIn('test123', memory)
        self.assertIn('## What Happened', memory)
        self.assertIn('## Key Insights', memory)
        self.assertIn('## Files Modified', memory)
        self.assertIn('## Tasks Updated', memory)
        self.assertIn('T-20251217-100000-a1b2', memory)
        self.assertIn('abc123d', memory)

    def test_save_draft(self):
        """Test saving draft memory."""
        gen = SessionMemoryGenerator(session_id='test123')
        output_dir = Path(self.temp_dir) / 'memories'

        content = "# Test Memory\n\nThis is a test."
        filepath = gen.save_draft(content, output_dir=str(output_dir))

        self.assertTrue(filepath.exists())
        self.assertIn('[DRAFT]', filepath.name)
        self.assertIn('session-test123', filepath.name)

        with open(filepath) as f:
            saved_content = f.read()
        self.assertEqual(saved_content, content)


class TestLinkCommitToTasks(unittest.TestCase):
    """Test task linking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tasks_dir = Path(self.temp_dir) / 'tasks'
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def test_link_commit_to_task(self):
        """Test linking a commit to a task."""
        # Create a task file
        task_data = {
            'version': 1,
            'session_id': 'test',
            'tasks': [
                {
                    'id': 'T-20251217-100000-a1b2',
                    'title': 'Test task',
                    'status': 'in_progress',
                    'context': {}
                }
            ]
        }
        task_file = self.tasks_dir / 'test_session.json'
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        # Link commit
        commit_sha = 'abc123def456'
        commit_message = 'feat: Complete T-20251217-100000-a1b2'

        updated = link_commit_to_tasks(commit_sha, commit_message, tasks_dir=str(self.tasks_dir))

        self.assertEqual(len(updated), 1)
        self.assertIn('T-20251217-100000-a1b2', updated)

        # Verify task was updated
        with open(task_file, 'r') as f:
            updated_data = json.load(f)

        task = updated_data['tasks'][0]
        self.assertIn('commits', task['context'])
        self.assertIn(commit_sha, task['context']['commits'])

    def test_link_commit_multiple_tasks(self):
        """Test linking a commit to multiple tasks."""
        # Create a task file with multiple tasks
        task_data = {
            'version': 1,
            'session_id': 'test',
            'tasks': [
                {'id': 'T-20251217-100000-a1b2', 'title': 'Task 1', 'status': 'in_progress', 'context': {}},
                {'id': 'T-20251217-100100-b2c3', 'title': 'Task 2', 'status': 'in_progress', 'context': {}},
            ]
        }
        task_file = self.tasks_dir / 'test_session.json'
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        # Link commit referencing both tasks
        commit_message = 'feat: Complete T-20251217-100000-a1b2 and T-20251217-100100-b2c3'
        updated = link_commit_to_tasks('abc123', commit_message, tasks_dir=str(self.tasks_dir))

        self.assertEqual(len(updated), 2)

    def test_link_commit_no_tasks(self):
        """Test linking commit with no task references."""
        commit_message = 'feat: Add feature with no task reference'
        updated = link_commit_to_tasks('abc123', commit_message, tasks_dir=str(self.tasks_dir))

        self.assertEqual(len(updated), 0)

    def test_link_commit_nonexistent_task(self):
        """Test linking commit to nonexistent task."""
        # Create empty task file
        task_data = {'version': 1, 'session_id': 'test', 'tasks': []}
        task_file = self.tasks_dir / 'test_session.json'
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        commit_message = 'feat: Complete T-20251217-999999-zzzz'
        updated = link_commit_to_tasks('abc123', commit_message, tasks_dir=str(self.tasks_dir))

        self.assertEqual(len(updated), 0)

    def test_link_commit_duplicate_prevention(self):
        """Test that duplicate commits are not added to task."""
        # Create a task file with existing commit
        task_data = {
            'version': 1,
            'session_id': 'test',
            'tasks': [
                {
                    'id': 'T-20251217-100000-a1b2',
                    'title': 'Test task',
                    'status': 'in_progress',
                    'context': {'commits': ['abc123']}
                }
            ]
        }
        task_file = self.tasks_dir / 'test_session.json'
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        # Try to link the same commit again
        commit_message = 'feat: Update T-20251217-100000-a1b2'
        updated = link_commit_to_tasks('abc123', commit_message, tasks_dir=str(self.tasks_dir))

        # Should not update (commit already present)
        self.assertEqual(len(updated), 0)

        # Verify only one commit in context
        with open(task_file, 'r') as f:
            updated_data = json.load(f)

        task = updated_data['tasks'][0]
        self.assertEqual(len(task['context']['commits']), 1)


if __name__ == '__main__':
    unittest.main()
