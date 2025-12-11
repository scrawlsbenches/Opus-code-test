"""
CLI wrapper framework for collecting context and triggering actions.

Design philosophy: QUIET BY DEFAULT, POWERFUL WHEN NEEDED.

Most of the time you just want to run a command and check if it worked.
The fancy stuff (hooks, tracking, context management) is there when you
need it, invisible when you don't.

Simple usage (90% of cases):
    from cortical.cli_wrapper import run

    result = run("pytest tests/")
    if result.success:
        print("Tests passed")
    else:
        print(result.stderr)

With git context (when you need it):
    result = run("git status", git=True)
    print(result.git.modified_files)

With session tracking (for complex workflows):
    with Session() as s:
        s.run("pytest tests/")
        s.run("git add -A")
        s.run("git commit -m 'fix tests'")

        if s.should_reindex():
            s.run("python scripts/index_codebase.py --incremental")

        print(s.summary())

Advanced (hooks for automation):
    wrapper = CLIWrapper()

    @wrapper.on_success("pytest")
    def after_tests(result):
        # Auto-update coverage badge, etc.
        pass
"""

import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Protocol
)


# =============================================================================
# Execution Context
# =============================================================================

@dataclass
class GitContext:
    """Git repository context information."""
    is_repo: bool = False
    branch: str = ""
    commit_hash: str = ""
    is_dirty: bool = False
    staged_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)

    @classmethod
    def collect(cls, cwd: Optional[str] = None) -> 'GitContext':
        """Collect git context from current directory."""
        ctx = cls()
        try:
            # Check if in git repo
            result = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5
            )
            if result.returncode != 0:
                return ctx
            ctx.is_repo = True

            # Get branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5
            )
            if result.returncode == 0:
                ctx.branch = result.stdout.strip()

            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5
            )
            if result.returncode == 0:
                ctx.commit_hash = result.stdout.strip()

            # Get status (porcelain for parsing)
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    status = line[:2]
                    filepath = line[3:]
                    if status[0] in ('A', 'M', 'D', 'R'):
                        ctx.staged_files.append(filepath)
                    if status[1] in ('M', 'D'):
                        ctx.modified_files.append(filepath)
                    if status == '??':
                        ctx.untracked_files.append(filepath)
                ctx.is_dirty = bool(
                    ctx.staged_files or ctx.modified_files or ctx.untracked_files
                )

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return ctx

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExecutionContext:
    """
    Complete context for a CLI command execution.

    Captures everything needed for:
    - Logging and debugging
    - Context window management decisions
    - Task completion triggers
    """
    # Execution metadata
    command: List[str] = field(default_factory=list)
    command_str: str = ""
    exit_code: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # Output capture
    stdout: str = ""
    stderr: str = ""
    output_lines: int = 0
    error_lines: int = 0

    # Environment context
    working_directory: str = ""
    session_id: str = ""
    timestamp: str = ""
    platform: str = ""
    python_version: str = ""

    # Git context
    git: GitContext = field(default_factory=GitContext)

    # Task classification
    task_type: str = ""  # 'test', 'build', 'commit', 'search', etc.
    success: bool = False

    # Custom metadata from hooks
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['git'] = self.git.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Return a concise summary string."""
        status = "✓" if self.success else "✗"
        return (
            f"{status} {self.command_str} "
            f"[{self.duration:.2f}s, exit={self.exit_code}]"
        )


# =============================================================================
# Hook System
# =============================================================================

class HookType(Enum):
    """Types of hooks that can be registered."""
    PRE_EXEC = "pre_exec"       # Before command execution
    POST_EXEC = "post_exec"     # After command execution (success or failure)
    ON_SUCCESS = "on_success"   # Only on successful execution
    ON_ERROR = "on_error"       # Only on failed execution
    ON_TIMEOUT = "on_timeout"   # When command times out


# Hook callback signature
HookCallback = Callable[[ExecutionContext], None]


class HookRegistry:
    """
    Registry for CLI execution hooks.

    Hooks can be registered globally or for specific command patterns.
    """

    def __init__(self):
        # Global hooks (apply to all commands)
        self._global_hooks: Dict[HookType, List[HookCallback]] = {
            hook_type: [] for hook_type in HookType
        }
        # Pattern-specific hooks (command prefix matching)
        self._pattern_hooks: Dict[str, Dict[HookType, List[HookCallback]]] = {}

    def register(
        self,
        hook_type: HookType,
        callback: HookCallback,
        pattern: Optional[str] = None
    ) -> None:
        """
        Register a hook callback.

        Args:
            hook_type: When to trigger the hook
            callback: Function to call with ExecutionContext
            pattern: Optional command pattern (e.g., 'git', 'pytest')
                    If None, applies to all commands
        """
        if pattern is None:
            self._global_hooks[hook_type].append(callback)
        else:
            if pattern not in self._pattern_hooks:
                self._pattern_hooks[pattern] = {
                    hook_type: [] for hook_type in HookType
                }
            self._pattern_hooks[pattern][hook_type].append(callback)

    def register_pre(
        self,
        pattern: Optional[str],
        callback: HookCallback
    ) -> None:
        """Convenience method for pre-execution hooks."""
        self.register(HookType.PRE_EXEC, callback, pattern)

    def register_post(
        self,
        pattern: Optional[str],
        callback: HookCallback
    ) -> None:
        """Convenience method for post-execution hooks."""
        self.register(HookType.POST_EXEC, callback, pattern)

    def register_success(
        self,
        pattern: Optional[str],
        callback: HookCallback
    ) -> None:
        """Convenience method for success hooks."""
        self.register(HookType.ON_SUCCESS, callback, pattern)

    def register_error(
        self,
        pattern: Optional[str],
        callback: HookCallback
    ) -> None:
        """Convenience method for error hooks."""
        self.register(HookType.ON_ERROR, callback, pattern)

    def get_hooks(
        self,
        hook_type: HookType,
        command: List[str]
    ) -> List[HookCallback]:
        """
        Get all hooks that should be triggered for a command.

        Args:
            hook_type: Type of hook
            command: Command being executed

        Returns:
            List of callbacks to execute
        """
        callbacks = list(self._global_hooks[hook_type])

        # Match patterns against command
        if command:
            cmd_str = ' '.join(command)
            for pattern, hooks in self._pattern_hooks.items():
                if cmd_str.startswith(pattern) or command[0] == pattern:
                    callbacks.extend(hooks[hook_type])

        return callbacks

    def trigger(
        self,
        hook_type: HookType,
        context: ExecutionContext
    ) -> None:
        """Trigger all matching hooks."""
        for callback in self.get_hooks(hook_type, context.command):
            try:
                callback(context)
            except Exception as e:
                # Log but don't fail on hook errors
                context.metadata.setdefault('hook_errors', []).append(
                    f"{hook_type.value}: {str(e)}"
                )


# =============================================================================
# CLI Wrapper
# =============================================================================

class CLIWrapper:
    """
    Wrapper for CLI command execution with context collection and hooks.

    Features:
    - Automatic context collection (timing, git status, environment)
    - Pre/post execution hooks
    - Task type classification
    - Timeout handling
    - Output capture
    """

    # Command patterns for task type classification
    TASK_PATTERNS = {
        'test': ['pytest', 'python -m pytest', 'python -m unittest', 'npm test'],
        'build': ['python -m build', 'npm run build', 'make', 'cargo build'],
        'commit': ['git commit', 'git add', 'git push'],
        'search': ['grep', 'rg', 'find', 'ag'],
        'install': ['pip install', 'npm install', 'cargo install'],
        'lint': ['flake8', 'pylint', 'mypy', 'eslint', 'ruff'],
        'format': ['black', 'prettier', 'rustfmt'],
    }

    def __init__(
        self,
        collect_git_context: bool = True,
        capture_output: bool = True,
        default_timeout: Optional[float] = None
    ):
        """
        Initialize CLI wrapper.

        Args:
            collect_git_context: Whether to collect git info before execution
            capture_output: Whether to capture stdout/stderr
            default_timeout: Default timeout in seconds (None = no timeout)
        """
        self.hooks = HookRegistry()
        self.collect_git_context = collect_git_context
        self.capture_output = capture_output
        self.default_timeout = default_timeout
        self.session_id = uuid.uuid4().hex[:16]

    def _classify_task(self, command: List[str]) -> str:
        """Classify command into a task type."""
        if not command:
            return 'unknown'

        cmd_str = ' '.join(command)
        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if cmd_str.startswith(pattern) or command[0] == pattern.split()[0]:
                    return task_type

        return 'other'

    def _build_context(
        self,
        command: List[str],
        cwd: Optional[str] = None
    ) -> ExecutionContext:
        """Build initial execution context."""
        cwd = cwd or os.getcwd()

        ctx = ExecutionContext(
            command=command,
            command_str=' '.join(command),
            working_directory=cwd,
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(timespec='seconds'),
            platform=platform.system(),
            python_version=platform.python_version(),
            task_type=self._classify_task(command),
        )

        if self.collect_git_context:
            ctx.git = GitContext.collect(cwd)

        return ctx

    def run(
        self,
        command: Union[str, List[str]],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ExecutionContext:
        """
        Execute a command with context collection and hooks.

        Args:
            command: Command to execute (string or list)
            cwd: Working directory
            timeout: Timeout in seconds (overrides default)
            env: Environment variables (merged with current env)
            **kwargs: Additional args passed to subprocess.run

        Returns:
            ExecutionContext with all collected metadata
        """
        # Normalize command
        if isinstance(command, str):
            command = command.split()

        # Build initial context
        ctx = self._build_context(command, cwd)
        ctx.start_time = time.time()

        # Merge environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Trigger pre-execution hooks
        self.hooks.trigger(HookType.PRE_EXEC, ctx)

        # Execute command
        effective_timeout = timeout if timeout is not None else self.default_timeout

        try:
            result = subprocess.run(
                command,
                capture_output=self.capture_output,
                text=True,
                cwd=cwd,
                env=run_env,
                timeout=effective_timeout,
                **kwargs
            )

            ctx.exit_code = result.returncode
            ctx.success = result.returncode == 0

            if self.capture_output:
                ctx.stdout = result.stdout or ""
                ctx.stderr = result.stderr or ""
                ctx.output_lines = len(ctx.stdout.splitlines())
                ctx.error_lines = len(ctx.stderr.splitlines())

        except subprocess.TimeoutExpired as e:
            ctx.exit_code = -1
            ctx.success = False
            ctx.metadata['timeout'] = effective_timeout
            ctx.metadata['timeout_error'] = str(e)
            if e.stdout:
                ctx.stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
            if e.stderr:
                ctx.stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            self.hooks.trigger(HookType.ON_TIMEOUT, ctx)

        except FileNotFoundError:
            ctx.exit_code = 127
            ctx.success = False
            ctx.metadata['error'] = f"Command not found: {command[0]}"

        except Exception as e:
            ctx.exit_code = -1
            ctx.success = False
            ctx.metadata['error'] = str(e)

        # Finalize timing
        ctx.end_time = time.time()
        ctx.duration = ctx.end_time - ctx.start_time

        # Trigger post-execution hooks
        self.hooks.trigger(HookType.POST_EXEC, ctx)

        # Trigger success/error hooks
        if ctx.success:
            self.hooks.trigger(HookType.ON_SUCCESS, ctx)
        else:
            self.hooks.trigger(HookType.ON_ERROR, ctx)

        return ctx

    # -------------------------------------------------------------------------
    # Decorator-style hook registration (cleaner API)
    # -------------------------------------------------------------------------

    def on_success(self, pattern: Optional[str] = None):
        """
        Decorator to register a success hook.

        Example:
            wrapper = CLIWrapper()

            @wrapper.on_success("pytest")
            def after_tests(result):
                print(f"Tests passed in {result.duration:.1f}s")
        """
        def decorator(func: HookCallback) -> HookCallback:
            self.hooks.register_success(pattern, func)
            return func
        return decorator

    def on_error(self, pattern: Optional[str] = None):
        """Decorator to register an error hook."""
        def decorator(func: HookCallback) -> HookCallback:
            self.hooks.register_error(pattern, func)
            return func
        return decorator

    def on_complete(self, pattern: Optional[str] = None):
        """Decorator to register a completion hook (success or failure)."""
        def decorator(func: HookCallback) -> HookCallback:
            self.hooks.register_post(pattern, func)
            return func
        return decorator


# =============================================================================
# Task Completion Manager
# =============================================================================

class TaskCompletionManager:
    """
    Manager for task completion triggers and context window management.

    Provides high-level task completion callbacks that can:
    - Trigger corpus re-indexing after code changes
    - Update context window summaries
    - Log task completions for session analysis
    - Chain multiple actions on task completion
    """

    def __init__(self):
        self._task_handlers: Dict[str, List[HookCallback]] = {}
        self._completion_log: List[ExecutionContext] = []
        self._completion_callbacks: List[HookCallback] = []

    def on_task_complete(
        self,
        task_type: str,
        callback: HookCallback
    ) -> None:
        """
        Register a callback for when a specific task type completes.

        Args:
            task_type: Task type to match ('test', 'commit', 'build', etc.)
            callback: Function called with ExecutionContext on completion
        """
        if task_type not in self._task_handlers:
            self._task_handlers[task_type] = []
        self._task_handlers[task_type].append(callback)

    def on_any_complete(self, callback: HookCallback) -> None:
        """Register a callback for any task completion."""
        self._completion_callbacks.append(callback)

    def handle_completion(self, context: ExecutionContext) -> None:
        """
        Handle task completion and trigger appropriate callbacks.

        Should be called from CLIWrapper post-execution hook.
        """
        # Log completion
        self._completion_log.append(context)

        # Trigger task-specific handlers
        if context.task_type in self._task_handlers:
            for callback in self._task_handlers[context.task_type]:
                try:
                    callback(context)
                except Exception as e:
                    context.metadata.setdefault('completion_errors', []).append(str(e))

        # Trigger global completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(context)
            except Exception as e:
                context.metadata.setdefault('completion_errors', []).append(str(e))

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tasks completed in this session.

        Useful for context window management decisions.
        """
        if not self._completion_log:
            return {
                'task_count': 0,
                'success_rate': 0.0,
                'total_duration': 0.0,
                'tasks_by_type': {},
                'files_modified': [],
            }

        tasks_by_type: Dict[str, Dict[str, Any]] = {}
        all_modified_files: List[str] = []

        for ctx in self._completion_log:
            task_type = ctx.task_type or 'unknown'
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = {
                    'count': 0,
                    'successes': 0,
                    'failures': 0,
                    'total_duration': 0.0,
                }

            tasks_by_type[task_type]['count'] += 1
            tasks_by_type[task_type]['total_duration'] += ctx.duration
            if ctx.success:
                tasks_by_type[task_type]['successes'] += 1
            else:
                tasks_by_type[task_type]['failures'] += 1

            # Collect modified files from git context
            all_modified_files.extend(ctx.git.modified_files)
            all_modified_files.extend(ctx.git.staged_files)

        total_tasks = len(self._completion_log)
        successes = sum(1 for ctx in self._completion_log if ctx.success)

        return {
            'task_count': total_tasks,
            'success_rate': successes / total_tasks if total_tasks > 0 else 0.0,
            'total_duration': sum(ctx.duration for ctx in self._completion_log),
            'tasks_by_type': tasks_by_type,
            'files_modified': list(set(all_modified_files)),
        }

    def should_trigger_reindex(self) -> bool:
        """
        Determine if corpus should be re-indexed based on session activity.

        Returns True if:
        - Code files were modified
        - Tests were run (may indicate code changes)
        - Git commits were made
        """
        summary = self.get_session_summary()

        # Check if relevant files were modified
        code_extensions = {'.py', '.js', '.ts', '.md', '.rst'}
        for filepath in summary['files_modified']:
            if any(filepath.endswith(ext) for ext in code_extensions):
                return True

        # Check if commits were made
        tasks_by_type = summary.get('tasks_by_type', {})
        if 'commit' in tasks_by_type and tasks_by_type['commit']['successes'] > 0:
            return True

        return False


# =============================================================================
# Context Window Integration
# =============================================================================

class ContextWindowManager:
    """
    Manages context window state based on CLI execution history.

    Tracks what information is "in context" and provides utilities for:
    - Summarizing recent activity
    - Identifying relevant files for the current task
    - Suggesting context pruning
    """

    def __init__(self, max_context_items: int = 50):
        self.max_context_items = max_context_items
        self._context_items: List[Dict[str, Any]] = []
        self._file_access_log: Dict[str, float] = {}  # filepath -> last access time

    def add_execution(self, context: ExecutionContext) -> None:
        """Add an execution to the context window."""
        item = {
            'type': 'execution',
            'task_type': context.task_type,
            'command': context.command_str,
            'success': context.success,
            'duration': context.duration,
            'timestamp': context.timestamp,
            'files': context.git.modified_files + context.git.staged_files,
        }
        self._context_items.append(item)

        # Track file access
        now = time.time()
        for filepath in item['files']:
            self._file_access_log[filepath] = now

        # Prune if needed
        if len(self._context_items) > self.max_context_items:
            self._context_items = self._context_items[-self.max_context_items:]

    def add_file_read(self, filepath: str) -> None:
        """Track that a file was read."""
        self._file_access_log[filepath] = time.time()
        self._context_items.append({
            'type': 'file_read',
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        })

    def get_recent_files(self, limit: int = 10) -> List[str]:
        """Get most recently accessed files."""
        sorted_files = sorted(
            self._file_access_log.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [f for f, _ in sorted_files[:limit]]

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current context window state.

        Useful for context window management decisions.
        """
        executions = [i for i in self._context_items if i['type'] == 'execution']
        file_reads = [i for i in self._context_items if i['type'] == 'file_read']

        task_types = {}
        for ex in executions:
            task_type = ex.get('task_type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1

        return {
            'total_items': len(self._context_items),
            'executions': len(executions),
            'file_reads': len(file_reads),
            'task_types': task_types,
            'recent_files': self.get_recent_files(5),
            'unique_files_accessed': len(self._file_access_log),
        }

    def suggest_pruning(self) -> List[str]:
        """
        Suggest files that could be pruned from context.

        Returns files that:
        - Haven't been accessed recently
        - Aren't related to recent task types
        """
        if len(self._file_access_log) < self.max_context_items // 2:
            return []

        now = time.time()
        stale_threshold = 300  # 5 minutes

        stale_files = [
            filepath
            for filepath, last_access in self._file_access_log.items()
            if now - last_access > stale_threshold
        ]

        return stale_files[:10]  # Suggest up to 10 files


# =============================================================================
# Convenience Functions
# =============================================================================

def create_wrapper_with_completion_manager() -> Tuple[CLIWrapper, TaskCompletionManager]:
    """
    Create a CLIWrapper with an attached TaskCompletionManager.

    Returns:
        Tuple of (wrapper, completion_manager) configured together
    """
    wrapper = CLIWrapper()
    manager = TaskCompletionManager()

    # Connect wrapper to completion manager
    wrapper.hooks.register_post(None, manager.handle_completion)

    return wrapper, manager


def run_with_context(
    command: Union[str, List[str]],
    **kwargs
) -> ExecutionContext:
    """
    Convenience function to run a command with full context collection.

    Args:
        command: Command to execute
        **kwargs: Additional arguments for CLIWrapper.run()

    Returns:
        ExecutionContext with all metadata
    """
    wrapper = CLIWrapper()
    return wrapper.run(command, **kwargs)


# =============================================================================
# Simple API (the 90% use case)
# =============================================================================

def run(
    command: Union[str, List[str]],
    git: bool = False,
    timeout: Optional[float] = None,
    cwd: Optional[str] = None,
) -> ExecutionContext:
    """
    Run a command. That's it.

    This is the simple API for the 90% use case. No hooks, no tracking,
    no noise. Just run and get results.

    Args:
        command: Command to run (string or list)
        git: If True, collect git context (branch, modified files, etc.)
        timeout: Timeout in seconds (None = no timeout)
        cwd: Working directory

    Returns:
        ExecutionContext with:
        - .success: bool - did it work?
        - .stdout: str - standard output
        - .stderr: str - standard error
        - .exit_code: int - exit code
        - .duration: float - how long it took
        - .git: GitContext - if git=True

    Example:
        result = run("pytest tests/")
        if result.success:
            print("All tests passed")
        else:
            print(f"Failed: {result.stderr}")
    """
    wrapper = CLIWrapper(
        collect_git_context=git,
        capture_output=True,
        default_timeout=timeout,
    )
    return wrapper.run(command, cwd=cwd)


# =============================================================================
# Session Context Manager
# =============================================================================

class Session:
    """
    Track a sequence of commands as a session.

    Use this when you want to:
    - Track multiple related commands together
    - Know if you should re-index after changes
    - Get a summary of what happened

    Example:
        with Session() as s:
            s.run("pytest tests/")
            s.run("git add -A")
            s.run("git commit -m 'fix'")

            if s.should_reindex():
                s.run("python scripts/index_codebase.py -i")

            print(s.summary())
    """

    def __init__(self, git: bool = True):
        """
        Start a session.

        Args:
            git: Whether to collect git context for commands (default True)
        """
        self._wrapper = CLIWrapper(collect_git_context=git)
        self._manager = TaskCompletionManager()
        self._context_manager = ContextWindowManager()
        self._results: List[ExecutionContext] = []

        # Wire up tracking (silent - no hooks that print anything)
        self._wrapper.hooks.register_post(None, self._track)

    def _track(self, ctx: ExecutionContext) -> None:
        """Internal: track command completion."""
        self._results.append(ctx)
        self._manager.handle_completion(ctx)
        self._context_manager.add_execution(ctx)

    def __enter__(self) -> 'Session':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # Nothing to clean up

    def run(
        self,
        command: Union[str, List[str]],
        **kwargs
    ) -> ExecutionContext:
        """Run a command within this session."""
        return self._wrapper.run(command, **kwargs)

    def should_reindex(self) -> bool:
        """Check if corpus re-indexing is recommended based on session activity."""
        return self._manager.should_trigger_reindex()

    def summary(self) -> Dict[str, Any]:
        """Get a summary of this session's activity."""
        return self._manager.get_session_summary()

    @property
    def results(self) -> List[ExecutionContext]:
        """All command results from this session."""
        return self._results.copy()

    @property
    def success_rate(self) -> float:
        """Fraction of commands that succeeded (0.0 to 1.0)."""
        if not self._results:
            return 1.0
        return sum(1 for r in self._results if r.success) / len(self._results)

    @property
    def all_passed(self) -> bool:
        """True if all commands in this session succeeded."""
        return all(r.success for r in self._results)

    @property
    def modified_files(self) -> List[str]:
        """List of files modified during this session (from git context)."""
        files = set()
        for r in self._results:
            files.update(r.git.modified_files)
            files.update(r.git.staged_files)
        return list(files)
