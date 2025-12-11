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
    run,
    Session,
    test_then_commit,
    commit_and_push,
    sync_with_main,
    TaskCheckpoint,
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

    def test_default_values(self):
        """Test default GitContext values."""
        ctx = GitContext()
        self.assertFalse(ctx.is_repo)
        self.assertEqual(ctx.branch, "")
        self.assertEqual(ctx.commit_hash, "")
        self.assertFalse(ctx.is_dirty)
        self.assertEqual(ctx.staged_files, [])
        self.assertEqual(ctx.modified_files, [])
        self.assertEqual(ctx.untracked_files, [])


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

    def test_run_with_env(self):
        """Test running command with custom environment."""
        wrapper = CLIWrapper(collect_git_context=False)
        result = wrapper.run(
            ['python', '-c', 'import os; print(os.environ.get("TEST_VAR", ""))'],
            env={'TEST_VAR': 'hello_test'}
        )

        self.assertTrue(result.success)
        self.assertIn('hello_test', result.stdout)

    def test_default_timeout(self):
        """Test wrapper with default timeout."""
        wrapper = CLIWrapper(
            collect_git_context=False,
            default_timeout=0.1
        )
        result = wrapper.run(['python', '-c', 'import time; time.sleep(10)'])

        self.assertFalse(result.success)
        self.assertIn('timeout', result.metadata)

    def test_timeout_hook_triggered(self):
        """Test ON_TIMEOUT hook is triggered."""
        wrapper = CLIWrapper(collect_git_context=False)
        timeout_calls = []

        wrapper.hooks.register(
            HookType.ON_TIMEOUT,
            lambda ctx: timeout_calls.append(ctx.command_str)
        )

        wrapper.run(['python', '-c', 'import time; time.sleep(10)'], timeout=0.1)

        self.assertEqual(len(timeout_calls), 1)

    def test_capture_output_disabled(self):
        """Test wrapper with capture_output disabled."""
        wrapper = CLIWrapper(
            collect_git_context=False,
            capture_output=False
        )
        result = wrapper.run(['echo', 'test'])

        self.assertTrue(result.success)
        # stdout/stderr not captured
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


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

    def test_run_with_context_string_command(self):
        """Test run_with_context with string command."""
        result = run_with_context("echo test")

        self.assertTrue(result.success)
        self.assertIn("test", result.stdout)

    def test_run_with_cwd(self):
        """Test run() with custom working directory."""
        result = run("pwd", cwd="/tmp")

        self.assertTrue(result.success)
        self.assertIn("/tmp", result.stdout)


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


class TestSimpleRunAPI(unittest.TestCase):
    """Tests for the simple run() API."""

    def test_run_basic(self):
        """Test basic run() usage."""
        result = run("echo hello")
        self.assertTrue(result.success)
        self.assertIn("hello", result.stdout)

    def test_run_no_git_by_default(self):
        """Test that git context is not collected by default."""
        result = run("echo test")
        # Git context should be empty/default when git=False
        self.assertFalse(result.git.is_repo)

    def test_run_with_git(self):
        """Test run with git context."""
        result = run("echo test", git=True)
        # We're in a git repo, so this should be True
        self.assertTrue(result.git.is_repo)
        self.assertTrue(len(result.git.branch) > 0)

    def test_run_with_timeout(self):
        """Test run with timeout."""
        result = run("python -c 'import time; time.sleep(10)'", timeout=0.1)
        self.assertFalse(result.success)

    def test_run_failure(self):
        """Test run with failing command."""
        result = run(["python", "-c", "import sys; sys.exit(42)"])
        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 42)


class TestSession(unittest.TestCase):
    """Tests for Session context manager."""

    def test_session_basic(self):
        """Test basic session usage."""
        with Session(git=False) as s:
            s.run("echo first")
            s.run("echo second")

        self.assertEqual(len(s.results), 2)
        self.assertTrue(s.all_passed)

    def test_session_tracks_failures(self):
        """Test session tracks failures correctly."""
        with Session(git=False) as s:
            s.run("echo ok")
            s.run("python -c 'import sys; sys.exit(1)'")
            s.run("echo also ok")

        self.assertEqual(len(s.results), 3)
        self.assertFalse(s.all_passed)
        self.assertAlmostEqual(s.success_rate, 2/3, places=2)

    def test_session_summary(self):
        """Test session summary."""
        with Session(git=False) as s:
            s.run("echo test")
            summary = s.summary()

        self.assertEqual(summary['task_count'], 1)
        self.assertEqual(summary['success_rate'], 1.0)

    def test_session_should_reindex(self):
        """Test should_reindex detection."""
        with Session(git=True) as s:
            # Just an echo - no code changes
            s.run("echo test")

        # No commits or file changes, shouldn't need reindex
        # (Unless the test repo is dirty)
        # Just verify it returns a boolean
        self.assertIsInstance(s.should_reindex(), bool)

    def test_session_context_manager(self):
        """Test session works as context manager."""
        results_outside = []

        with Session(git=False) as s:
            result = s.run("echo inside")
            results_outside.append(result)

        # Can still access results after exiting
        self.assertEqual(len(s.results), 1)
        self.assertTrue(s.results[0].success)

    def test_session_success_rate_empty(self):
        """Test success_rate with no commands."""
        with Session(git=False) as s:
            pass  # No commands run

        self.assertEqual(s.success_rate, 1.0)  # Default to 1.0

    def test_session_modified_files(self):
        """Test modified_files property."""
        with Session(git=True) as s:
            s.run("echo test")

        # Should return a list (possibly empty)
        self.assertIsInstance(s.modified_files, list)

    def test_session_with_git_context(self):
        """Test session collects git context when enabled."""
        with Session(git=True) as s:
            result = s.run("echo test")

        # Git context should be collected
        self.assertTrue(result.git.is_repo)


class TestDecoratorHooks(unittest.TestCase):
    """Tests for decorator-style hook registration."""

    def test_on_success_decorator(self):
        """Test @wrapper.on_success decorator."""
        wrapper = CLIWrapper(collect_git_context=False)
        calls = []

        @wrapper.on_success()
        def track_success(ctx):
            calls.append(('success', ctx.command_str))

        wrapper.run("echo hello")
        wrapper.run("python -c 'import sys; sys.exit(1)'")

        # Only the successful command should trigger
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'success')

    def test_on_error_decorator(self):
        """Test @wrapper.on_error decorator."""
        wrapper = CLIWrapper(collect_git_context=False)
        errors = []

        @wrapper.on_error()
        def track_error(ctx):
            errors.append(ctx.exit_code)

        wrapper.run(["echo", "hello"])  # success
        wrapper.run(["python", "-c", "import sys; sys.exit(1)"])  # fail
        wrapper.run(["python", "-c", "import sys; sys.exit(2)"])  # fail

        self.assertEqual(len(errors), 2)
        self.assertEqual(errors, [1, 2])

    def test_on_complete_decorator(self):
        """Test @wrapper.on_complete decorator."""
        wrapper = CLIWrapper(collect_git_context=False)
        completions = []

        @wrapper.on_complete()
        def track_all(ctx):
            completions.append(ctx.success)

        wrapper.run("echo hello")
        wrapper.run("python -c 'import sys; sys.exit(1)'")

        self.assertEqual(completions, [True, False])

    def test_pattern_decorator(self):
        """Test decorator with pattern matching."""
        wrapper = CLIWrapper(collect_git_context=False)
        echo_count = [0]

        @wrapper.on_success("echo")
        def on_echo_success(ctx):
            echo_count[0] += 1

        wrapper.run("echo one")
        wrapper.run("echo two")
        wrapper.run("python -c 'print(1)'")  # Not echo

        self.assertEqual(echo_count[0], 2)


class TestCompoundCommands(unittest.TestCase):
    """Tests for compound command functions."""

    def test_test_then_commit_fails_on_test_failure(self):
        """Test that test_then_commit stops if tests fail."""
        # Use a test command that fails
        ok, results = test_then_commit(
            test_cmd=["python", "-c", "import sys; sys.exit(1)"],
            message="Should not commit"
        )

        self.assertFalse(ok)
        self.assertEqual(len(results), 1)  # Only test ran
        self.assertFalse(results[0].success)

    def test_test_then_commit_returns_results(self):
        """Test that results are returned correctly."""
        # Use a test that passes
        ok, results = test_then_commit(
            test_cmd=["echo", "tests pass"],
            message="Test commit",
            add_all=False  # Don't actually add files
        )

        # Test passed, so we get test result + commit attempt
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(results[0].success)  # echo succeeded

    def test_sync_with_main_returns_tuple(self):
        """Test sync_with_main returns proper structure."""
        # This will likely fail (no remote) but should return proper tuple
        ok, results = sync_with_main(main_branch="nonexistent-branch-xyz")

        self.assertIsInstance(ok, bool)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_commit_and_push_returns_tuple(self):
        """Test commit_and_push returns proper structure."""
        # Will fail (nothing to commit) but should return proper tuple
        ok, results = commit_and_push(
            message="Test message",
            add_all=False  # Don't add files
        )

        self.assertIsInstance(ok, bool)
        self.assertIsInstance(results, list)

    def test_commit_and_push_with_add_all_false(self):
        """Test commit_and_push without adding files."""
        ok, results = commit_and_push(
            message="No add",
            add_all=False
        )

        # Should have at least the commit attempt
        self.assertGreaterEqual(len(results), 1)

    def test_commit_and_push_explicit_branch(self):
        """Test commit_and_push with explicit branch name."""
        ok, results = commit_and_push(
            message="Test",
            add_all=False,
            branch="test-branch-xyz"
        )

        # Should return results regardless of success
        self.assertIsInstance(results, list)

    def test_test_then_commit_with_string_command(self):
        """Test test_then_commit with string test command."""
        ok, results = test_then_commit(
            test_cmd="echo passing",
            message="String cmd test",
            add_all=False
        )

        self.assertTrue(results[0].success)


class TestTaskCheckpoint(unittest.TestCase):
    """Tests for TaskCheckpoint context saving."""

    def setUp(self):
        """Create a temporary checkpoint directory."""
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint = TaskCheckpoint(checkpoint_dir=self.tmpdir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        """Test saving and loading a checkpoint."""
        context = {
            'branch': 'feature/test',
            'notes': 'Working on tests',
            'files': ['test.py'],
        }

        self.checkpoint.save("my-task", context)
        loaded = self.checkpoint.load("my-task")

        self.assertEqual(loaded['branch'], 'feature/test')
        self.assertEqual(loaded['notes'], 'Working on tests')
        self.assertEqual(loaded['files'], ['test.py'])

    def test_load_nonexistent(self):
        """Test loading a nonexistent checkpoint returns None."""
        result = self.checkpoint.load("does-not-exist")
        self.assertIsNone(result)

    def test_list_tasks(self):
        """Test listing saved tasks."""
        self.checkpoint.save("task-a", {'note': 'a'})
        self.checkpoint.save("task-b", {'note': 'b'})

        tasks = self.checkpoint.list_tasks()

        self.assertIn("task-a", tasks)
        self.assertIn("task-b", tasks)

    def test_delete(self):
        """Test deleting a checkpoint."""
        self.checkpoint.save("to-delete", {'temp': True})

        # Should exist
        self.assertIsNotNone(self.checkpoint.load("to-delete"))

        # Delete it
        deleted = self.checkpoint.delete("to-delete")
        self.assertTrue(deleted)

        # Should be gone
        self.assertIsNone(self.checkpoint.load("to-delete"))

    def test_delete_nonexistent(self):
        """Test deleting nonexistent checkpoint returns False."""
        deleted = self.checkpoint.delete("never-existed")
        self.assertFalse(deleted)

    def test_summarize(self):
        """Test one-line summary generation."""
        self.checkpoint.save("feature-x", {
            'branch': 'feature/x',
            'notes': 'Need to add validation',
        })

        summary = self.checkpoint.summarize("feature-x")

        self.assertIn("feature-x", summary)
        self.assertIn("[feature/x]", summary)
        self.assertIn("validation", summary)

    def test_summarize_truncates_long_notes(self):
        """Test that long notes are truncated."""
        long_notes = "A" * 100  # 100 chars

        self.checkpoint.save("verbose-task", {
            'notes': long_notes,
        })

        summary = self.checkpoint.summarize("verbose-task")

        # Should be truncated
        self.assertLess(len(summary), 100)
        self.assertIn("...", summary)

    def test_summarize_without_notes(self):
        """Test summarize with no notes field."""
        self.checkpoint.save("no-notes", {
            'branch': 'feature/x',
        })

        summary = self.checkpoint.summarize("no-notes")

        self.assertIn("no-notes", summary)
        self.assertIn("[feature/x]", summary)

    def test_summarize_nonexistent(self):
        """Test summarize for nonexistent task."""
        result = self.checkpoint.summarize("does-not-exist")
        self.assertIsNone(result)

    def test_summarize_minimal(self):
        """Test summarize with minimal context."""
        self.checkpoint.save("minimal", {})

        summary = self.checkpoint.summarize("minimal")

        self.assertEqual(summary, "minimal")


class TestContextWindowManagerEdgeCases(unittest.TestCase):
    """Additional tests for ContextWindowManager edge cases."""

    def test_get_recent_files_empty(self):
        """Test get_recent_files with no files."""
        manager = ContextWindowManager()
        recent = manager.get_recent_files()
        self.assertEqual(recent, [])

    def test_suggest_pruning_below_threshold(self):
        """Test suggest_pruning when below threshold."""
        manager = ContextWindowManager(max_context_items=100)

        # Add a few files (well below threshold)
        manager.add_file_read("file1.py")
        manager.add_file_read("file2.py")

        # Should return empty - not enough items to suggest pruning
        suggestions = manager.suggest_pruning()
        self.assertEqual(suggestions, [])

    def test_context_summary_empty(self):
        """Test context summary with no items."""
        manager = ContextWindowManager()
        summary = manager.get_context_summary()

        self.assertEqual(summary['total_items'], 0)
        self.assertEqual(summary['executions'], 0)
        self.assertEqual(summary['file_reads'], 0)
        self.assertEqual(summary['task_types'], {})
        self.assertEqual(summary['recent_files'], [])
        self.assertEqual(summary['unique_files_accessed'], 0)


class TestHookRegistryEdgeCases(unittest.TestCase):
    """Additional tests for HookRegistry edge cases."""

    def test_get_hooks_empty_command(self):
        """Test get_hooks with empty command list."""
        registry = HookRegistry()
        callback = Mock()
        registry.register(HookType.POST_EXEC, callback)

        hooks = registry.get_hooks(HookType.POST_EXEC, [])
        self.assertEqual(len(hooks), 1)  # Global hook still returned

    def test_multiple_patterns_same_hook_type(self):
        """Test multiple pattern hooks for same type."""
        registry = HookRegistry()
        git_cb = Mock()
        pytest_cb = Mock()

        registry.register(HookType.POST_EXEC, git_cb, pattern='git')
        registry.register(HookType.POST_EXEC, pytest_cb, pattern='pytest')

        # Neither should match 'echo'
        hooks = registry.get_hooks(HookType.POST_EXEC, ['echo', 'hello'])
        self.assertEqual(len(hooks), 0)


class TestExecutionContextEdgeCases(unittest.TestCase):
    """Additional tests for ExecutionContext edge cases."""

    def test_metadata_field(self):
        """Test that metadata field works correctly."""
        ctx = ExecutionContext()
        ctx.metadata['custom_key'] = 'custom_value'

        self.assertEqual(ctx.metadata['custom_key'], 'custom_value')

        d = ctx.to_dict()
        self.assertEqual(d['metadata']['custom_key'], 'custom_value')


if __name__ == '__main__':
    unittest.main()
