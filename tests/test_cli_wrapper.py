"""
Tests for CLI wrapper framework.

Tests cover:
- ExecutionContext data collection
- GitContext collection
- HookRegistry registration and triggering
- CLIWrapper command execution
- TaskCompletionManager callbacks
- ContextWindowManager tracking
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.cli_wrapper import (
    ExecutionContext,
    GitContext,
    HookRegistry,
    HookType,
    CLIWrapper,
    TaskCompletionManager,
    ContextWindowManager,
    create_wrapper_with_completion_manager,
    run_with_context,
)


class TestGitContext(unittest.TestCase):
    """Tests for GitContext collection."""

    def test_collect_non_repo(self):
        """Test collection outside git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = GitContext.collect(tmpdir)
            self.assertFalse(ctx.is_repo)
            self.assertEqual(ctx.branch, "")

    def test_collect_in_repo(self):
        """Test collection inside current repo."""
        # Current directory should be a git repo
        ctx = GitContext.collect()
        self.assertTrue(ctx.is_repo)
        self.assertTrue(len(ctx.branch) > 0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ctx = GitContext(
            is_repo=True,
            branch="main",
            commit_hash="abc123",
            is_dirty=True,
            staged_files=["file1.py"],
            modified_files=["file2.py"],
            untracked_files=["file3.py"],
        )
        d = ctx.to_dict()
        self.assertEqual(d['branch'], "main")
        self.assertEqual(d['commit_hash'], "abc123")
        self.assertTrue(d['is_dirty'])
        self.assertEqual(len(d['staged_files']), 1)


class TestExecutionContext(unittest.TestCase):
    """Tests for ExecutionContext."""

    def test_default_values(self):
        """Test default context values."""
        ctx = ExecutionContext()
        self.assertEqual(ctx.exit_code, 0)
        self.assertEqual(ctx.command, [])
        self.assertFalse(ctx.success)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ctx = ExecutionContext(
            command=['echo', 'hello'],
            command_str='echo hello',
            exit_code=0,
            success=True,
            stdout='hello\n',
        )
        d = ctx.to_dict()
        self.assertEqual(d['command'], ['echo', 'hello'])
        self.assertEqual(d['exit_code'], 0)
        self.assertTrue(d['success'])

    def test_to_json(self):
        """Test JSON serialization."""
        ctx = ExecutionContext(
            command=['test'],
            success=True,
        )
        json_str = ctx.to_json()
        self.assertIn('"success": true', json_str)
        self.assertIn('"command":', json_str)
        self.assertIn('"test"', json_str)

    def test_summary(self):
        """Test summary generation."""
        ctx = ExecutionContext(
            command_str='pytest tests/',
            success=True,
            exit_code=0,
            duration=1.5,
        )
        summary = ctx.summary()
        self.assertIn('pytest tests/', summary)
        self.assertIn('1.50s', summary)
        self.assertIn('✓', summary)

    def test_summary_failure(self):
        """Test summary for failed command."""
        ctx = ExecutionContext(
            command_str='pytest tests/',
            success=False,
            exit_code=1,
            duration=0.5,
        )
        summary = ctx.summary()
        self.assertIn('✗', summary)
        self.assertIn('exit=1', summary)


class TestHookRegistry(unittest.TestCase):
    """Tests for HookRegistry."""

    def test_register_global_hook(self):
        """Test registering a global hook."""
        registry = HookRegistry()
        callback = Mock()

        registry.register(HookType.POST_EXEC, callback)

        hooks = registry.get_hooks(HookType.POST_EXEC, ['any', 'command'])
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0], callback)

    def test_register_pattern_hook(self):
        """Test registering a pattern-specific hook."""
        registry = HookRegistry()
        git_callback = Mock()
        pytest_callback = Mock()

        registry.register(HookType.POST_EXEC, git_callback, pattern='git')
        registry.register(HookType.POST_EXEC, pytest_callback, pattern='pytest')

        # Git command should match git hook
        git_hooks = registry.get_hooks(HookType.POST_EXEC, ['git', 'status'])
        self.assertEqual(len(git_hooks), 1)
        self.assertEqual(git_hooks[0], git_callback)

        # Pytest command should match pytest hook
        pytest_hooks = registry.get_hooks(HookType.POST_EXEC, ['pytest', 'tests/'])
        self.assertEqual(len(pytest_hooks), 1)
        self.assertEqual(pytest_hooks[0], pytest_callback)

    def test_convenience_methods(self):
        """Test convenience registration methods."""
        registry = HookRegistry()
        pre_cb = Mock()
        post_cb = Mock()
        success_cb = Mock()
        error_cb = Mock()

        registry.register_pre(None, pre_cb)
        registry.register_post(None, post_cb)
        registry.register_success('git', success_cb)
        registry.register_error('pytest', error_cb)

        self.assertEqual(len(registry.get_hooks(HookType.PRE_EXEC, ['any'])), 1)
        self.assertEqual(len(registry.get_hooks(HookType.POST_EXEC, ['any'])), 1)
        self.assertEqual(len(registry.get_hooks(HookType.ON_SUCCESS, ['git'])), 1)
        self.assertEqual(len(registry.get_hooks(HookType.ON_ERROR, ['pytest'])), 1)

    def test_trigger_hooks(self):
        """Test triggering hooks."""
        registry = HookRegistry()
        callback = Mock()
        registry.register(HookType.POST_EXEC, callback)

        ctx = ExecutionContext(command=['test'])
        registry.trigger(HookType.POST_EXEC, ctx)

        callback.assert_called_once_with(ctx)

    def test_hook_error_handling(self):
        """Test that hook errors don't crash execution."""
        registry = HookRegistry()

        def bad_callback(ctx):
            raise ValueError("Hook error!")

        registry.register(HookType.POST_EXEC, bad_callback)

        ctx = ExecutionContext(command=['test'])
        # Should not raise
        registry.trigger(HookType.POST_EXEC, ctx)

        # Error should be recorded
        self.assertIn('hook_errors', ctx.metadata)
        self.assertTrue(any('Hook error!' in e for e in ctx.metadata['hook_errors']))


class TestCLIWrapper(unittest.TestCase):
    """Tests for CLIWrapper."""

    def test_run_simple_command(self):
        """Test running a simple command."""
        wrapper = CLIWrapper(collect_git_context=False)
        result = wrapper.run(['echo', 'hello'])

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.success)
        self.assertIn('hello', result.stdout)
        self.assertGreater(result.duration, 0)

    def test_run_with_string_command(self):
        """Test running command as string."""
        wrapper = CLIWrapper(collect_git_context=False)
        result = wrapper.run('echo hello')

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.command, ['echo', 'hello'])

    def test_run_failing_command(self):
        """Test handling failed command."""
        wrapper = CLIWrapper(collect_git_context=False)
        result = wrapper.run(['python', '-c', 'import sys; sys.exit(1)'])

        self.assertEqual(result.exit_code, 1)
        self.assertFalse(result.success)

    def test_run_with_timeout(self):
        """Test command timeout handling."""
        wrapper = CLIWrapper(collect_git_context=False)
        result = wrapper.run(
            ['python', '-c', 'import time; time.sleep(10)'],
            timeout=0.1
        )

        self.assertFalse(result.success)
        self.assertIn('timeout', result.metadata)

    def test_run_nonexistent_command(self):
        """Test handling nonexistent command."""
        wrapper = CLIWrapper(collect_git_context=False)
        result = wrapper.run(['nonexistent_command_xyz'])

        self.assertEqual(result.exit_code, 127)
        self.assertFalse(result.success)
        self.assertIn('error', result.metadata)

    def test_task_classification(self):
        """Test task type classification."""
        wrapper = CLIWrapper(collect_git_context=False)

        # Test task
        result = wrapper.run(['echo', 'test'])  # not a test command
        self.assertEqual(result.task_type, 'other')

        # Simulate pytest classification
        ctx = wrapper._build_context(['pytest', 'tests/'])
        self.assertEqual(ctx.task_type, 'test')

        # Git classification
        ctx = wrapper._build_context(['git', 'commit', '-m', 'test'])
        self.assertEqual(ctx.task_type, 'commit')

    def test_hooks_triggered(self):
        """Test that hooks are triggered during execution."""
        wrapper = CLIWrapper(collect_git_context=False)

        pre_called = []
        post_called = []

        def pre_hook(ctx):
            pre_called.append(ctx.command_str)

        def post_hook(ctx):
            post_called.append((ctx.command_str, ctx.success))

        wrapper.hooks.register_pre(None, pre_hook)
        wrapper.hooks.register_post(None, post_hook)

        wrapper.run(['echo', 'hello'])

        self.assertEqual(len(pre_called), 1)
        self.assertEqual(len(post_called), 1)
        self.assertTrue(post_called[0][1])  # success = True

    def test_success_error_hooks(self):
        """Test ON_SUCCESS and ON_ERROR hooks."""
        wrapper = CLIWrapper(collect_git_context=False)

        success_calls = []
        error_calls = []

        wrapper.hooks.register_success(None, lambda ctx: success_calls.append(1))
        wrapper.hooks.register_error(None, lambda ctx: error_calls.append(1))

        # Successful command
        wrapper.run(['echo', 'hello'])
        self.assertEqual(len(success_calls), 1)
        self.assertEqual(len(error_calls), 0)

        # Failed command
        wrapper.run(['python', '-c', 'import sys; sys.exit(1)'])
        self.assertEqual(len(success_calls), 1)  # unchanged
        self.assertEqual(len(error_calls), 1)

    def test_git_context_collection(self):
        """Test git context is collected when enabled."""
        wrapper = CLIWrapper(collect_git_context=True)
        result = wrapper.run(['echo', 'test'])

        # Should have git context (we're in a git repo)
        self.assertTrue(result.git.is_repo)
        self.assertTrue(len(result.git.branch) > 0)

    def test_output_capture(self):
        """Test stdout/stderr capture."""
        wrapper = CLIWrapper(collect_git_context=False)

        # Stdout
        result = wrapper.run(['echo', 'hello'])
        self.assertIn('hello', result.stdout)
        self.assertEqual(result.output_lines, 1)

        # Stderr
        result = wrapper.run(['python', '-c', 'import sys; print("error", file=sys.stderr)'])
        self.assertIn('error', result.stderr)
        self.assertEqual(result.error_lines, 1)


class TestTaskCompletionManager(unittest.TestCase):
    """Tests for TaskCompletionManager."""

    def test_register_task_handler(self):
        """Test registering task-specific handlers."""
        manager = TaskCompletionManager()
        handler_calls = []

        manager.on_task_complete('test', lambda ctx: handler_calls.append(ctx))

        # Simulate completion
        ctx = ExecutionContext(task_type='test', success=True)
        manager.handle_completion(ctx)

        self.assertEqual(len(handler_calls), 1)
        self.assertEqual(handler_calls[0], ctx)

    def test_register_any_handler(self):
        """Test registering global completion handler."""
        manager = TaskCompletionManager()
        handler_calls = []

        manager.on_any_complete(lambda ctx: handler_calls.append(ctx.task_type))

        manager.handle_completion(ExecutionContext(task_type='test'))
        manager.handle_completion(ExecutionContext(task_type='commit'))
        manager.handle_completion(ExecutionContext(task_type='build'))

        self.assertEqual(handler_calls, ['test', 'commit', 'build'])

    def test_session_summary(self):
        """Test session summary generation."""
        manager = TaskCompletionManager()

        # Empty summary
        summary = manager.get_session_summary()
        self.assertEqual(summary['task_count'], 0)

        # Add some completions
        manager.handle_completion(ExecutionContext(
            task_type='test',
            success=True,
            duration=1.0
        ))
        manager.handle_completion(ExecutionContext(
            task_type='test',
            success=False,
            duration=0.5
        ))
        manager.handle_completion(ExecutionContext(
            task_type='commit',
            success=True,
            duration=0.2
        ))

        summary = manager.get_session_summary()
        self.assertEqual(summary['task_count'], 3)
        self.assertAlmostEqual(summary['success_rate'], 2/3, places=2)
        self.assertAlmostEqual(summary['total_duration'], 1.7, places=2)
        self.assertEqual(summary['tasks_by_type']['test']['count'], 2)
        self.assertEqual(summary['tasks_by_type']['commit']['successes'], 1)

    def test_should_trigger_reindex_on_commit(self):
        """Test reindex recommendation after commit."""
        manager = TaskCompletionManager()

        # No activity
        self.assertFalse(manager.should_trigger_reindex())

        # Commit success should trigger
        manager.handle_completion(ExecutionContext(
            task_type='commit',
            success=True,
        ))
        self.assertTrue(manager.should_trigger_reindex())

    def test_should_trigger_reindex_on_file_changes(self):
        """Test reindex recommendation on file changes."""
        manager = TaskCompletionManager()

        ctx = ExecutionContext(task_type='other')
        ctx.git.modified_files = ['test.py']
        manager.handle_completion(ctx)

        self.assertTrue(manager.should_trigger_reindex())


class TestContextWindowManager(unittest.TestCase):
    """Tests for ContextWindowManager."""

    def test_add_execution(self):
        """Test adding executions to context."""
        manager = ContextWindowManager()

        ctx = ExecutionContext(
            task_type='test',
            command_str='pytest tests/',
            success=True,
            duration=1.0,
        )
        manager.add_execution(ctx)

        summary = manager.get_context_summary()
        self.assertEqual(summary['executions'], 1)
        self.assertEqual(summary['task_types']['test'], 1)

    def test_add_file_read(self):
        """Test tracking file reads."""
        manager = ContextWindowManager()

        manager.add_file_read('test.py')
        manager.add_file_read('another.py')

        summary = manager.get_context_summary()
        self.assertEqual(summary['file_reads'], 2)
        self.assertEqual(summary['unique_files_accessed'], 2)

    def test_recent_files(self):
        """Test getting recently accessed files."""
        manager = ContextWindowManager()

        manager.add_file_read('old.py')
        time.sleep(0.01)  # Small delay
        manager.add_file_read('new.py')

        recent = manager.get_recent_files(limit=2)
        self.assertEqual(recent[0], 'new.py')  # Most recent first

    def test_context_pruning_suggestion(self):
        """Test pruning suggestions for stale files."""
        manager = ContextWindowManager(max_context_items=10)

        # Add some files
        for i in range(20):
            manager.add_file_read(f'file{i}.py')

        # With default 5-minute staleness, nothing should be stale yet
        suggestions = manager.suggest_pruning()
        # All files accessed just now, so no suggestions expected
        # (unless we mock time)
        self.assertIsInstance(suggestions, list)

    def test_max_context_pruning(self):
        """Test automatic pruning when max context exceeded."""
        manager = ContextWindowManager(max_context_items=5)

        # Add more items than max
        for i in range(10):
            ctx = ExecutionContext(task_type='test', command_str=f'cmd{i}')
            manager.add_execution(ctx)

        summary = manager.get_context_summary()
        self.assertEqual(summary['total_items'], 5)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_wrapper_with_manager(self):
        """Test creating wrapper with completion manager."""
        wrapper, manager = create_wrapper_with_completion_manager()

        self.assertIsInstance(wrapper, CLIWrapper)
        self.assertIsInstance(manager, TaskCompletionManager)

        # Run a command and verify manager receives it
        wrapper.run(['echo', 'test'])
        summary = manager.get_session_summary()
        self.assertEqual(summary['task_count'], 1)

    def test_run_with_context(self):
        """Test run_with_context convenience function."""
        result = run_with_context(['echo', 'hello'])

        self.assertIsInstance(result, ExecutionContext)
        self.assertTrue(result.success)
        self.assertIn('hello', result.stdout)


class TestIntegration(unittest.TestCase):
    """Integration tests for the wrapper system."""

    def test_full_workflow(self):
        """Test complete workflow with hooks and completion tracking."""
        wrapper, manager = create_wrapper_with_completion_manager()

        # Track hook calls
        hook_calls = []
        wrapper.hooks.register_pre(None, lambda ctx: hook_calls.append('pre'))
        wrapper.hooks.register_post(None, lambda ctx: hook_calls.append('post'))

        # Track completion
        completion_data = []
        manager.on_any_complete(lambda ctx: completion_data.append({
            'cmd': ctx.command_str,
            'success': ctx.success,
        }))

        # Run commands
        wrapper.run(['echo', 'first'])
        wrapper.run(['echo', 'second'])
        wrapper.run(['python', '-c', 'import sys; sys.exit(1)'])

        # Verify hooks
        self.assertEqual(hook_calls.count('pre'), 3)
        self.assertEqual(hook_calls.count('post'), 3)

        # Verify completions
        self.assertEqual(len(completion_data), 3)
        self.assertTrue(completion_data[0]['success'])
        self.assertTrue(completion_data[1]['success'])
        self.assertFalse(completion_data[2]['success'])

        # Verify session summary
        summary = manager.get_session_summary()
        self.assertEqual(summary['task_count'], 3)
        self.assertAlmostEqual(summary['success_rate'], 2/3, places=2)


if __name__ == '__main__':
    unittest.main()
