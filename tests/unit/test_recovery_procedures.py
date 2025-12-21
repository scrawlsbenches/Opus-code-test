"""
Unit Tests for RecoveryProcedures in cortical/reasoning/crisis_manager.py
=========================================================================

Tests for the RecoveryProcedures class which provides automated recovery
procedures for different crisis types including git operations, memory
document generation, and stash management.

Test Categories:
1. Initialization and Git Availability
2. Full Rollback Operations
3. Partial Recovery Operations
4. Memory Document Generation
5. Git Stash Operations
6. Last Good Commit Detection
7. Failure Analysis Generation
8. Git-Not-Available Scenarios
"""

import os
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

from cortical.reasoning.crisis_manager import (
    CrisisEvent,
    CrisisLevel,
    RecoveryProcedures,
)


class TestRecoveryProceduresInit(unittest.TestCase):
    """Test RecoveryProcedures initialization."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_init_with_git_available(self, mock_which):
        """Test initialization when git is available."""
        mock_which.return_value = '/usr/bin/git'

        recovery = RecoveryProcedures()

        self.assertTrue(recovery._git_available)
        self.assertEqual(recovery._recovery_log, [])
        self.assertEqual(recovery._memory_path, 'samples/memories/')
        mock_which.assert_called_once_with('git')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_init_without_git(self, mock_which):
        """Test initialization when git is not available."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()

        self.assertFalse(recovery._git_available)
        self.assertEqual(recovery._recovery_log, [])

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_init_with_custom_memory_path(self, mock_which):
        """Test initialization with custom memory path."""
        mock_which.return_value = '/usr/bin/git'
        custom_path = '/tmp/memories/'

        recovery = RecoveryProcedures(memory_path=custom_path)

        self.assertEqual(recovery._memory_path, custom_path)


class TestRunGit(unittest.TestCase):
    """Test the _run_git helper method."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_run_git_success(self, mock_run, mock_which):
        """Test successful git command execution."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.return_value = Mock(stdout='success', stderr='', returncode=0)

        recovery = RecoveryProcedures()
        result = recovery._run_git('status')

        mock_run.assert_called_once_with(
            ['git', 'status'],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_run_git_without_git_available(self, mock_which):
        """Test _run_git raises error when git is not available."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()

        with self.assertRaises(RuntimeError) as ctx:
            recovery._run_git('status')

        self.assertEqual(str(ctx.exception), "Git is not available")


class TestFullRollback(unittest.TestCase):
    """Test full_rollback functionality."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_full_rollback_success(self, mock_run, mock_which):
        """Test successful full rollback."""
        mock_which.return_value = '/usr/bin/git'

        # Mock subprocess responses
        mock_run.side_effect = [
            Mock(stdout='', stderr='', returncode=0),  # stash save
            Mock(stdout='stash@{0}: recovery-...', stderr='', returncode=0),  # stash list
            Mock(stdout='', stderr='', returncode=0),  # checkout
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            recovery = RecoveryProcedures(memory_path=temp_dir)
            result = recovery.full_rollback('abc123', 'Test crisis')

            self.assertTrue(result['success'])
            self.assertEqual(result['stash_ref'], 'stash@{0}')
            self.assertEqual(result['restored_to'], 'abc123')
            self.assertIn(temp_dir, result['memory_file'])
            self.assertIn('[DRAFT]-recovery-', result['memory_file'])
            self.assertNotIn('error', result)

            # Verify memory file was created
            self.assertTrue(os.path.exists(result['memory_file']))

            # Verify git commands were called
            calls = mock_run.call_args_list
            self.assertEqual(calls[0][0][0][:3], ['git', 'stash', 'save'])
            self.assertEqual(calls[2][0][0], ['git', 'checkout', 'abc123'])

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_full_rollback_without_git(self, mock_which):
        """Test full rollback fails gracefully without git."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()
        result = recovery.full_rollback('abc123')

        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Git is not available')
        self.assertEqual(result['stash_ref'], '')
        self.assertEqual(result['restored_to'], 'abc123')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_full_rollback_with_no_changes_to_stash(self, mock_run, mock_which):
        """Test rollback when there are no changes to stash."""
        mock_which.return_value = '/usr/bin/git'

        # Stash save fails (no changes), checkout succeeds
        def side_effect(*args, **kwargs):
            if 'stash' in args[0] and 'save' in args[0]:
                raise subprocess.CalledProcessError(1, args[0], stderr='No local changes')
            return Mock(stdout='', stderr='', returncode=0)

        mock_run.side_effect = side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            recovery = RecoveryProcedures(memory_path=temp_dir)
            result = recovery.full_rollback('abc123', 'Test crisis')

            self.assertTrue(result['success'])
            self.assertEqual(result['stash_ref'], 'none (no changes)')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_full_rollback_checkout_failure(self, mock_run, mock_which):
        """Test rollback handles checkout failure."""
        mock_which.return_value = '/usr/bin/git'

        # Stash succeeds, checkout fails
        def side_effect(*args, **kwargs):
            if 'checkout' in args[0]:
                raise subprocess.CalledProcessError(
                    1, args[0], stderr='error: pathspec \'abc123\' did not match'
                )
            return Mock(stdout='', stderr='', returncode=0)

        mock_run.side_effect = side_effect

        recovery = RecoveryProcedures()
        result = recovery.full_rollback('invalid-commit')

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('Git command failed', result['error'])


class TestPartialRecovery(unittest.TestCase):
    """Test partial_recovery functionality."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_partial_recovery_success(self, mock_run, mock_which):
        """Test successful partial recovery."""
        mock_which.return_value = '/usr/bin/git'

        working_files = ['file1.py', 'file2.py']
        broken_files = ['broken.py']

        # Mock git operations
        mock_run.side_effect = [
            Mock(stdout='', returncode=0),  # add file1
            Mock(stdout='', returncode=0),  # add file2
            Mock(stdout='', returncode=0),  # commit
            Mock(stdout='def456\n', returncode=0),  # rev-parse HEAD
            Mock(stdout='', returncode=0),  # add broken.py
            Mock(stdout='', returncode=0),  # stash save
        ]

        recovery = RecoveryProcedures()
        result = recovery.partial_recovery(working_files, broken_files)

        self.assertTrue(result['success'])
        self.assertEqual(result['committed_files'], working_files)
        self.assertEqual(result['commit_hash'], 'def456')
        self.assertEqual(result['stashed_files'], broken_files)
        self.assertEqual(result['stash_ref'], 'stash@{0}')
        self.assertIn('Follow-up: Fix Broken Files', result['task_description'])
        self.assertNotIn('error', result)

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_partial_recovery_without_git(self, mock_which):
        """Test partial recovery fails gracefully without git."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()
        result = recovery.partial_recovery(['file1.py'], ['broken.py'])

        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Git is not available')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_partial_recovery_with_custom_message(self, mock_run, mock_which):
        """Test partial recovery with custom commit message."""
        mock_which.return_value = '/usr/bin/git'

        working_files = ['file1.py']
        custom_msg = 'fix: salvaged authentication module'

        mock_run.side_effect = [
            Mock(stdout='', returncode=0),  # add
            Mock(stdout='', returncode=0),  # commit
            Mock(stdout='abc123\n', returncode=0),  # rev-parse
        ]

        recovery = RecoveryProcedures()
        result = recovery.partial_recovery(working_files, [], commit_message=custom_msg)

        self.assertTrue(result['success'])

        # Verify custom message was used
        commit_call = mock_run.call_args_list[1]
        self.assertIn(custom_msg, commit_call[0][0])

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_partial_recovery_only_working_files(self, mock_run, mock_which):
        """Test partial recovery with only working files (no broken files)."""
        mock_which.return_value = '/usr/bin/git'

        working_files = ['file1.py']

        mock_run.side_effect = [
            Mock(stdout='', returncode=0),  # add
            Mock(stdout='', returncode=0),  # commit
            Mock(stdout='abc123\n', returncode=0),  # rev-parse
        ]

        recovery = RecoveryProcedures()
        result = recovery.partial_recovery(working_files, [])

        self.assertTrue(result['success'])
        self.assertEqual(result['stashed_files'], [])
        self.assertEqual(result['stash_ref'], '')


class TestCreateRecoveryMemory(unittest.TestCase):
    """Test create_recovery_memory functionality."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_create_recovery_memory(self, mock_which):
        """Test recovery memory document generation."""
        mock_which.return_value = '/usr/bin/git'

        with tempfile.TemporaryDirectory() as temp_dir:
            recovery = RecoveryProcedures(memory_path=temp_dir)

            crisis = "Tests failing after refactor"
            actions = ["Rolled back to last good commit", "Stashed broken changes"]
            lessons = ["Should have run tests before committing", "Refactor was too large"]

            file_path = recovery.create_recovery_memory(crisis, actions, lessons)

            # Verify file exists
            self.assertTrue(os.path.exists(file_path))
            self.assertIn(temp_dir, file_path)
            self.assertIn('[DRAFT]-recovery-', file_path)

            # Verify content
            with open(file_path, 'r') as f:
                content = f.read()

            self.assertIn(crisis, content)
            self.assertIn('Crisis Recovery', content)
            self.assertIn('recovery', content)
            self.assertIn('lessons-learned', content)

            for action in actions:
                self.assertIn(action, content)

            for lesson in lessons:
                self.assertIn(lesson, content)

            # Verify markdown structure
            self.assertIn('# Recovery Memory:', content)
            self.assertIn('## Crisis Description', content)
            self.assertIn('## Recovery Actions Taken', content)
            self.assertIn('## Lessons Learned', content)
            self.assertIn('## Preventive Measures', content)

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_create_recovery_memory_creates_directory(self, mock_which):
        """Test that create_recovery_memory creates directory if missing."""
        mock_which.return_value = '/usr/bin/git'

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = os.path.join(temp_dir, 'nested', 'memories')
            recovery = RecoveryProcedures(memory_path=memory_path)

            file_path = recovery.create_recovery_memory(
                "Test crisis", ["Action 1"], ["Lesson 1"]
            )

            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(os.path.isdir(memory_path))


class TestGetLastGoodCommit(unittest.TestCase):
    """Test get_last_good_commit functionality."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_get_last_good_commit_with_test_commit(self, mock_run, mock_which):
        """Test finding commit with 'test' in message."""
        mock_which.return_value = '/usr/bin/git'

        log_output = """abc123|WIP: broken feature
def456|test: add new tests
ghi789|feat: new feature"""

        mock_run.return_value = Mock(stdout=log_output, returncode=0)

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        self.assertEqual(commit, 'def456')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_get_last_good_commit_avoids_bad_indicators(self, mock_run, mock_which):
        """Test that commits with bad indicators are skipped."""
        mock_which.return_value = '/usr/bin/git'

        log_output = """abc123|WIP: in progress
def456|broken: something failing
ghi789|fix: bug fix
jkl012|feat: new feature"""

        mock_run.return_value = Mock(stdout=log_output, returncode=0)

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        self.assertEqual(commit, 'ghi789')  # First with 'fix'

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_get_last_good_commit_fallback_to_5th(self, mock_run, mock_which):
        """Test fallback to 5th commit when no good indicators found."""
        mock_which.return_value = '/usr/bin/git'

        log_output = """commit1|message1
commit2|message2
commit3|message3
commit4|message4
commit5|message5
commit6|message6"""

        mock_run.return_value = Mock(stdout=log_output, returncode=0)

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        self.assertEqual(commit, 'commit5')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_get_last_good_commit_without_git(self, mock_which):
        """Test get_last_good_commit returns empty string without git."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        self.assertEqual(commit, '')


class TestStashOperations(unittest.TestCase):
    """Test stash-related operations."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_stash_current_state_success(self, mock_run, mock_which):
        """Test successful stash of current state."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.return_value = Mock(stdout='', returncode=0)

        recovery = RecoveryProcedures()
        stash_ref = recovery.stash_current_state('my-label')

        self.assertEqual(stash_ref, 'stash@{0}')
        mock_run.assert_called_once_with(
            ['git', 'stash', 'save', 'my-label'],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_stash_current_state_without_git(self, mock_which):
        """Test stash returns empty string without git."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()
        stash_ref = recovery.stash_current_state('label')

        self.assertEqual(stash_ref, '')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_list_stashes_success(self, mock_run, mock_which):
        """Test listing stashes."""
        mock_which.return_value = '/usr/bin/git'

        stash_output = """stash@{0}: WIP on main: abc123 Latest work
stash@{1}: On main: recovery-20251219
stash@{2}: On feature: broken-files"""

        mock_run.return_value = Mock(stdout=stash_output, returncode=0)

        recovery = RecoveryProcedures()
        stashes = recovery.list_stashes()

        self.assertEqual(len(stashes), 3)
        self.assertEqual(stashes[0]['ref'], 'stash@{0}')
        self.assertEqual(stashes[0]['index'], 0)
        self.assertIn('WIP on main', stashes[0]['message'])

        self.assertEqual(stashes[1]['ref'], 'stash@{1}')
        self.assertEqual(stashes[1]['index'], 1)

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_list_stashes_without_git(self, mock_which):
        """Test list_stashes returns empty list without git."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()
        stashes = recovery.list_stashes()

        self.assertEqual(stashes, [])

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_restore_from_stash_success(self, mock_run, mock_which):
        """Test successful stash restore."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.return_value = Mock(stdout='', returncode=0)

        recovery = RecoveryProcedures()
        success = recovery.restore_from_stash('stash@{0}')

        self.assertTrue(success)
        mock_run.assert_called_once_with(
            ['git', 'stash', 'apply', 'stash@{0}'],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_restore_from_stash_without_git(self, mock_which):
        """Test restore returns False without git."""
        mock_which.return_value = None

        recovery = RecoveryProcedures()
        success = recovery.restore_from_stash('stash@{0}')

        self.assertFalse(success)

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_restore_from_stash_failure(self, mock_run, mock_which):
        """Test restore handles failure gracefully."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ['git', 'stash', 'apply'], stderr='Conflicts'
        )

        recovery = RecoveryProcedures()
        success = recovery.restore_from_stash('stash@{99}')

        self.assertFalse(success)


class TestGenerateFailureAnalysis(unittest.TestCase):
    """Test generate_failure_analysis functionality."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_generate_failure_analysis_basic(self, mock_which):
        """Test basic failure analysis generation."""
        mock_which.return_value = '/usr/bin/git'

        event = CrisisEvent(
            level=CrisisLevel.WALL,
            description="Tests keep failing",
            context={
                'goal': 'Fix authentication bug',
                'approach': 'Refactor authentication module',
                'attempts': ['Try 1: failed', 'Try 2: failed', 'Try 3: failed'],
            },
        )

        recovery = RecoveryProcedures()
        analysis = recovery.generate_failure_analysis(event)

        self.assertIn('Tests keep failing', analysis)
        self.assertIn('Fix authentication bug', analysis)
        self.assertIn('Refactor authentication module', analysis)
        self.assertIn('Try 1: failed', analysis)
        self.assertIn('Multiple attempts were made', analysis)

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_generate_failure_analysis_with_lessons(self, mock_which):
        """Test failure analysis includes lessons learned."""
        mock_which.return_value = '/usr/bin/git'

        event = CrisisEvent(
            level=CrisisLevel.OBSTACLE,
            description="Performance degradation",
            context={'goal': 'Optimize query', 'approach': 'Add caching'},
        )
        event.add_lesson("Caching added complexity without benefit")
        event.add_lesson("Should have profiled first")

        recovery = RecoveryProcedures()
        analysis = recovery.generate_failure_analysis(event)

        self.assertIn('Caching added complexity', analysis)
        self.assertIn('Should have profiled first', analysis)

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_generate_failure_analysis_minimal_context(self, mock_which):
        """Test failure analysis with minimal context."""
        mock_which.return_value = '/usr/bin/git'

        event = CrisisEvent(
            level=CrisisLevel.HICCUP,
            description="Simple error",
            context={},
        )

        recovery = RecoveryProcedures()
        analysis = recovery.generate_failure_analysis(event)

        self.assertIn('Simple error', analysis)
        self.assertIn('[Unknown goal]', analysis)
        self.assertIn('[Unknown approach]', analysis)


class TestRecoveryLog(unittest.TestCase):
    """Test recovery log tracking."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_recovery_log_tracks_full_rollback(self, mock_run, mock_which):
        """Test that full_rollback adds to recovery log."""
        mock_which.return_value = '/usr/bin/git'

        mock_run.side_effect = [
            Mock(stdout='', returncode=0),  # stash
            Mock(stdout='', returncode=0),  # stash list
            Mock(stdout='', returncode=0),  # checkout
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            recovery = RecoveryProcedures(memory_path=temp_dir)

            self.assertEqual(len(recovery._recovery_log), 0)

            recovery.full_rollback('abc123', 'Test crisis')

            self.assertEqual(len(recovery._recovery_log), 1)
            log_entry = recovery._recovery_log[0]
            self.assertEqual(log_entry['type'], 'full_rollback')
            self.assertEqual(log_entry['checkpoint'], 'abc123')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_recovery_log_tracks_partial_recovery(self, mock_run, mock_which):
        """Test that partial_recovery adds to recovery log."""
        mock_which.return_value = '/usr/bin/git'

        mock_run.side_effect = [
            Mock(stdout='', returncode=0),  # add
            Mock(stdout='', returncode=0),  # commit
            Mock(stdout='def456\n', returncode=0),  # rev-parse
        ]

        recovery = RecoveryProcedures()
        recovery.partial_recovery(['file1.py'], [])

        self.assertEqual(len(recovery._recovery_log), 1)
        log_entry = recovery._recovery_log[0]
        self.assertEqual(log_entry['type'], 'partial_recovery')
        self.assertEqual(log_entry['working_files'], ['file1.py'])


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete recovery workflows."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_complete_recovery_workflow(self, mock_run, mock_which):
        """Test a complete recovery workflow."""
        mock_which.return_value = '/usr/bin/git'

        # Scenario: Work goes bad, need to find last good commit and rollback

        # Setup mock responses
        log_output = """abc123|WIP: broken
def456|test: all tests passing
ghi789|feat: new feature"""

        mock_run.side_effect = [
            # get_last_good_commit
            Mock(stdout=log_output, returncode=0),
            # full_rollback: stash
            Mock(stdout='', returncode=0),
            Mock(stdout='', returncode=0),
            # checkout
            Mock(stdout='', returncode=0),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            recovery = RecoveryProcedures(memory_path=temp_dir)

            # Find last good commit
            good_commit = recovery.get_last_good_commit()
            self.assertEqual(good_commit, 'def456')

            # Rollback to it
            result = recovery.full_rollback(good_commit, 'Complete recovery test')
            self.assertTrue(result['success'])
            self.assertTrue(os.path.exists(result['memory_file']))


class TestBehavioralRequirements(unittest.TestCase):
    """Test behavioral requirements for recovery procedures."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    def test_memory_documents_contain_useful_information(self, mock_which):
        """Test that generated memory documents are useful and actionable."""
        mock_which.return_value = '/usr/bin/git'

        with tempfile.TemporaryDirectory() as temp_dir:
            recovery = RecoveryProcedures(memory_path=temp_dir)

            crisis = "Authentication module refactor caused cascading test failures"
            actions = [
                "Stashed current changes to preserve work",
                "Rolled back to commit before refactor",
                "Created follow-up task to retry with smaller scope",
            ]
            lessons = [
                "Refactor scope was too large",
                "Should have refactored one function at a time",
                "Need better integration tests before major refactors",
            ]

            file_path = recovery.create_recovery_memory(crisis, actions, lessons)

            with open(file_path, 'r') as f:
                content = f.read()

            # Verify useful sections exist
            self.assertIn('Crisis Description', content)
            self.assertIn('Recovery Actions Taken', content)
            self.assertIn('Lessons Learned', content)
            self.assertIn('Preventive Measures', content)

            # Verify all content is present
            self.assertIn(crisis, content)
            for action in actions:
                self.assertIn(action, content)
            for lesson in lessons:
                self.assertIn(lesson, content)

            # Verify actionable guidance
            self.assertIn('prevent', content.lower())
            self.assertIn('[DRAFT]', file_path)  # Marked as draft for review

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_partial_recovery_generates_actionable_tasks(self, mock_run, mock_which):
        """Test that partial recovery generates clear follow-up tasks."""
        mock_which.return_value = '/usr/bin/git'

        working_files = ['auth/login.py', 'auth/session.py']
        broken_files = ['auth/permissions.py', 'auth/roles.py']

        mock_run.side_effect = [
            Mock(stdout='', returncode=0),  # add x2
            Mock(stdout='', returncode=0),
            Mock(stdout='', returncode=0),  # commit
            Mock(stdout='abc123\n', returncode=0),  # rev-parse
            Mock(stdout='', returncode=0),  # add x2
            Mock(stdout='', returncode=0),
            Mock(stdout='', returncode=0),  # stash
        ]

        recovery = RecoveryProcedures()
        result = recovery.partial_recovery(working_files, broken_files)

        task_desc = result['task_description']

        # Verify task is actionable
        self.assertIn('Next steps:', task_desc)
        self.assertIn('git stash apply', task_desc)
        self.assertIn(result['stash_ref'], task_desc)

        # Verify it lists what's salvaged and what needs work
        for file in working_files:
            self.assertIn(file, task_desc)
        for file in broken_files:
            self.assertIn(file, task_desc)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_full_rollback_generic_exception(self, mock_run, mock_which):
        """Test full_rollback handles generic exceptions."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.side_effect = Exception("Unexpected error")

        recovery = RecoveryProcedures()
        result = recovery.full_rollback('abc123')

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Unexpected error')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_partial_recovery_generic_exception(self, mock_run, mock_which):
        """Test partial_recovery handles generic exceptions."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.side_effect = Exception("Unexpected error")

        recovery = RecoveryProcedures()
        result = recovery.partial_recovery(['file1.py'], [])

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Unexpected error')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_stash_current_state_failure(self, mock_run, mock_which):
        """Test stash_current_state handles failures."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.side_effect = subprocess.CalledProcessError(1, ['git', 'stash'])

        recovery = RecoveryProcedures()
        stash_ref = recovery.stash_current_state('label')

        self.assertEqual(stash_ref, '')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_list_stashes_with_empty_output(self, mock_run, mock_which):
        """Test list_stashes handles empty output."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.return_value = Mock(stdout='', returncode=0)

        recovery = RecoveryProcedures()
        stashes = recovery.list_stashes()

        self.assertEqual(stashes, [])

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_list_stashes_failure(self, mock_run, mock_which):
        """Test list_stashes handles git command failures."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.side_effect = subprocess.CalledProcessError(1, ['git', 'stash', 'list'])

        recovery = RecoveryProcedures()
        stashes = recovery.list_stashes()

        self.assertEqual(stashes, [])

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_get_last_good_commit_with_malformed_log(self, mock_run, mock_which):
        """Test get_last_good_commit handles malformed git log output."""
        mock_which.return_value = '/usr/bin/git'

        # Lines without pipe separator
        log_output = """malformed line 1
malformed line 2"""

        mock_run.side_effect = [
            Mock(stdout=log_output, returncode=0),  # git log
            Mock(stdout='fallback123\n', returncode=0),  # git rev-parse HEAD~1
        ]

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        # Should fall back to HEAD~1
        self.assertEqual(commit, 'fallback123')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_get_last_good_commit_with_few_commits(self, mock_run, mock_which):
        """Test get_last_good_commit with less than 5 commits."""
        mock_which.return_value = '/usr/bin/git'

        log_output = """commit1|message1
commit2|message2
commit3|message3"""

        mock_run.side_effect = [
            Mock(stdout=log_output, returncode=0),  # git log
            Mock(stdout='fallback123\n', returncode=0),  # git rev-parse HEAD~1
        ]

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        # Should fall back to HEAD~1
        self.assertEqual(commit, 'fallback123')

    @patch('cortical.reasoning.crisis_manager.shutil.which')
    @patch('cortical.reasoning.crisis_manager.subprocess.run')
    def test_get_last_good_commit_complete_failure(self, mock_run, mock_which):
        """Test get_last_good_commit when all git commands fail."""
        mock_which.return_value = '/usr/bin/git'
        mock_run.side_effect = subprocess.CalledProcessError(1, ['git'])

        recovery = RecoveryProcedures()
        commit = recovery.get_last_good_commit()

        self.assertEqual(commit, '')


if __name__ == '__main__':
    unittest.main()
