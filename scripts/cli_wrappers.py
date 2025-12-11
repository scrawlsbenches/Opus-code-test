#!/usr/bin/env python3
"""
Example CLI wrappers for common development tasks.

These wrappers collect context and trigger actions automatically:
- git_wrapper: Track commits, trigger re-indexing
- test_wrapper: Run tests, log results
- dev_wrapper: General development command wrapper

Usage:
    # As a module
    from scripts.cli_wrappers import git_wrapper, test_wrapper

    result = git_wrapper.run(['git', 'status'])
    result = test_wrapper.run(['pytest', 'tests/', '-v'])

    # As CLI
    python scripts/cli_wrappers.py git status
    python scripts/cli_wrappers.py test pytest tests/ -v
    python scripts/cli_wrappers.py run echo hello
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.cli_wrapper import (
    CLIWrapper,
    ExecutionContext,
    TaskCompletionManager,
    ContextWindowManager,
    HookType,
    create_wrapper_with_completion_manager,
)


# =============================================================================
# Context-Aware Wrapper Configuration
# =============================================================================

class DevWrapper:
    """
    Development-focused CLI wrapper with smart defaults.

    Features:
    - Automatic re-index triggering after code changes
    - Test result logging
    - Session activity tracking
    - Context window management hints
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        auto_reindex: bool = True,
        verbose: bool = False
    ):
        self.wrapper, self.completion_manager = create_wrapper_with_completion_manager()
        self.context_manager = ContextWindowManager()
        self.verbose = verbose
        self.auto_reindex = auto_reindex

        # Set up logging directory
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = Path('.cli_wrapper_logs')
            self.log_dir.mkdir(exist_ok=True)

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Set up default hooks for development workflow."""

        # Pre-execution: log command start
        def log_start(ctx: ExecutionContext):
            if self.verbose:
                print(f"→ {ctx.command_str}")

        self.wrapper.hooks.register_pre(None, log_start)

        # Post-execution: track in context window
        def track_context(ctx: ExecutionContext):
            self.context_manager.add_execution(ctx)

        self.wrapper.hooks.register_post(None, track_context)

        # Git commit success: suggest re-indexing
        def on_commit_success(ctx: ExecutionContext):
            if self.auto_reindex and ctx.task_type == 'commit' and ctx.success:
                print("\n💡 Code committed. Consider re-indexing:")
                print("   python scripts/index_codebase.py --incremental")

        self.wrapper.hooks.register_success('git commit', on_commit_success)

        # Test completion: log results
        def on_test_complete(ctx: ExecutionContext):
            if ctx.task_type == 'test':
                status = "✓ PASS" if ctx.success else "✗ FAIL"
                print(f"\n{status} ({ctx.duration:.2f}s)")

                # Log detailed results
                self._log_test_result(ctx)

        self.wrapper.hooks.register_post('pytest', on_test_complete)
        self.wrapper.hooks.register_post('python -m unittest', on_test_complete)
        self.wrapper.hooks.register_post('python -m pytest', on_test_complete)

    def _log_test_result(self, ctx: ExecutionContext):
        """Log test results to file."""
        log_file = self.log_dir / 'test_results.jsonl'

        entry = {
            'timestamp': ctx.timestamp,
            'command': ctx.command_str,
            'success': ctx.success,
            'duration': ctx.duration,
            'exit_code': ctx.exit_code,
            'output_lines': ctx.output_lines,
            'error_lines': ctx.error_lines,
            'git_branch': ctx.git.branch,
            'git_commit': ctx.git.commit_hash,
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def run(self, command: List[str], **kwargs) -> ExecutionContext:
        """Execute a command with full context collection."""
        return self.wrapper.run(command, **kwargs)

    def get_session_summary(self) -> dict:
        """Get summary of session activity."""
        return self.completion_manager.get_session_summary()

    def get_context_summary(self) -> dict:
        """Get context window state summary."""
        return self.context_manager.get_context_summary()

    def should_reindex(self) -> bool:
        """Check if re-indexing is recommended."""
        return self.completion_manager.should_trigger_reindex()


# =============================================================================
# Specialized Wrappers
# =============================================================================

class GitWrapper(DevWrapper):
    """Git-specific wrapper with additional context collection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track files changed per commit
        def track_commit_files(ctx: ExecutionContext):
            if 'commit' in ctx.command_str and ctx.success:
                ctx.metadata['committed_files'] = ctx.git.staged_files.copy()

        self.wrapper.hooks.register_success('git commit', track_commit_files)

    def status(self) -> ExecutionContext:
        """Run git status."""
        return self.run(['git', 'status'])

    def diff(self, staged: bool = False) -> ExecutionContext:
        """Run git diff."""
        cmd = ['git', 'diff']
        if staged:
            cmd.append('--staged')
        return self.run(cmd)

    def log(self, count: int = 5) -> ExecutionContext:
        """Run git log."""
        return self.run(['git', 'log', f'-{count}', '--oneline'])

    def add(self, *paths: str) -> ExecutionContext:
        """Run git add."""
        return self.run(['git', 'add'] + list(paths))

    def commit(self, message: str) -> ExecutionContext:
        """Run git commit."""
        return self.run(['git', 'commit', '-m', message])


class TestWrapper(DevWrapper):
    """Test runner wrapper with result tracking."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_history: List[ExecutionContext] = []

        # Track test results
        def track_test(ctx: ExecutionContext):
            if ctx.task_type == 'test':
                self.test_history.append(ctx)

        self.wrapper.hooks.register_post(None, track_test)

    def pytest(self, *args: str, **kwargs) -> ExecutionContext:
        """Run pytest with given arguments."""
        cmd = ['pytest'] + list(args)
        return self.run(cmd, **kwargs)

    def unittest(self, pattern: str = 'tests/', **kwargs) -> ExecutionContext:
        """Run unittest discover."""
        cmd = ['python', '-m', 'unittest', 'discover', '-s', pattern, '-v']
        return self.run(cmd, **kwargs)

    def get_test_summary(self) -> dict:
        """Get summary of test runs in this session."""
        if not self.test_history:
            return {'runs': 0, 'passes': 0, 'failures': 0}

        return {
            'runs': len(self.test_history),
            'passes': sum(1 for t in self.test_history if t.success),
            'failures': sum(1 for t in self.test_history if not t.success),
            'total_duration': sum(t.duration for t in self.test_history),
            'last_run': self.test_history[-1].timestamp if self.test_history else None,
        }


# =============================================================================
# Singleton Instances
# =============================================================================

# Global wrapper instances for convenience
_git_wrapper: Optional[GitWrapper] = None
_test_wrapper: Optional[TestWrapper] = None
_dev_wrapper: Optional[DevWrapper] = None


def get_git_wrapper(**kwargs) -> GitWrapper:
    """Get or create the global git wrapper."""
    global _git_wrapper
    if _git_wrapper is None:
        _git_wrapper = GitWrapper(**kwargs)
    return _git_wrapper


def get_test_wrapper(**kwargs) -> TestWrapper:
    """Get or create the global test wrapper."""
    global _test_wrapper
    if _test_wrapper is None:
        _test_wrapper = TestWrapper(**kwargs)
    return _test_wrapper


def get_dev_wrapper(**kwargs) -> DevWrapper:
    """Get or create the global development wrapper."""
    global _dev_wrapper
    if _dev_wrapper is None:
        _dev_wrapper = DevWrapper(**kwargs)
    return _dev_wrapper


# Convenience aliases
git_wrapper = get_git_wrapper
test_wrapper = get_test_wrapper
dev_wrapper = get_dev_wrapper


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point for wrapped commands."""
    parser = argparse.ArgumentParser(
        description='Context-aware CLI wrapper for development commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s git status
    %(prog)s git log -5 --oneline
    %(prog)s test pytest tests/ -v
    %(prog)s run echo "hello world"
    %(prog)s --summary
        """
    )

    parser.add_argument(
        'wrapper_type',
        nargs='?',
        choices=['git', 'test', 'run'],
        help='Type of wrapper to use'
    )
    parser.add_argument(
        'command',
        nargs='*',
        help='Command to execute'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show session summary'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Handle summary request
    if args.summary:
        wrapper = get_dev_wrapper(verbose=args.verbose)
        summary = {
            'session': wrapper.get_session_summary(),
            'context': wrapper.get_context_summary(),
            'should_reindex': wrapper.should_reindex(),
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Session Summary:")
            print(f"  Tasks: {summary['session']['task_count']}")
            print(f"  Success Rate: {summary['session']['success_rate']:.1%}")
            print(f"  Total Duration: {summary['session']['total_duration']:.2f}s")
            print(f"\nContext Window:")
            print(f"  Items: {summary['context']['total_items']}")
            print(f"  Files Accessed: {summary['context']['unique_files_accessed']}")
            if summary['should_reindex']:
                print("\n💡 Re-indexing recommended")
        return 0

    # Need wrapper type and command
    if not args.wrapper_type:
        parser.print_help()
        return 1

    if not args.command:
        print(f"Error: No command specified for '{args.wrapper_type}' wrapper")
        return 1

    # Select wrapper
    if args.wrapper_type == 'git':
        wrapper = get_git_wrapper(verbose=args.verbose)
        cmd = ['git'] + args.command
    elif args.wrapper_type == 'test':
        wrapper = get_test_wrapper(verbose=args.verbose)
        cmd = args.command
    else:  # 'run'
        wrapper = get_dev_wrapper(verbose=args.verbose)
        cmd = args.command

    # Execute
    result = wrapper.run(cmd)

    # Output
    if args.json:
        print(result.to_json())
    else:
        if result.stdout:
            print(result.stdout, end='')
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)

    return result.exit_code


if __name__ == '__main__':
    sys.exit(main())
