"""
ClaudeCodeSpawner - Production spawner for Claude Code sub-agents.

This module provides two implementations of AgentSpawner:

1. ClaudeCodeSpawner: Generates Task tool configurations (non-subprocess)
   - For use within Claude Code sessions
   - Generates configurations for Task tool calls
   - Requires manual result recording after Task completion

2. SubprocessClaudeCodeSpawner: Spawns actual Claude Code CLI processes
   - For production automation and parallel workflows
   - Spawns real subprocesses with isolation
   - Handles timeouts, output capture, and process lifecycle
   - Tracks performance metrics

Architecture:
    Unlike SequentialSpawner which runs tasks inline, ClaudeCodeSpawner
    generates configurations that must be executed via the Task tool.

    The workflow is:
    1. Call prepare_agents() to generate Task tool configurations
    2. The orchestrator (Claude) calls Task tools in parallel
    3. Call record_results() with the agent outputs
    4. Use get_result() to access parsed results

Example usage in a Claude Code session:

    spawner = ClaudeCodeSpawner()
    coordinator = ParallelCoordinator(spawner)

    # Prepare agents
    configs = spawner.prepare_agents([
        ("Implement auth", boundary1),
        ("Write tests", boundary2),
    ])

    # configs now contains Task tool parameters for parallel execution
    # The orchestrator calls Task tools with these configs

    # After Task tools return, record results:
    spawner.record_result("agent-001", task_output_1)
    spawner.record_result("agent-002", task_output_2)

    # Collect via coordinator
    results = coordinator.collect_results(["agent-001", "agent-002"])

Example usage for subprocess spawning:

    spawner = SubprocessClaudeCodeSpawner(
        max_concurrent=5,
        default_timeout=300.0,
        working_dir=Path("/path/to/repo")
    )

    # Spawn a subprocess
    agent_id = spawner.spawn("Implement auth feature", boundary)

    # Wait for completion
    result = spawner.wait_for(agent_id, timeout_seconds=300)

    # Check metrics
    metrics = spawner.get_metrics()
"""

import asyncio
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from .collaboration import (
    AgentSpawner,
    AgentStatus,
    AgentResult,
    ParallelWorkBoundary,
)


@dataclass
class TaskToolConfig:
    """Configuration for a Task tool invocation."""
    agent_id: str
    description: str  # Short 3-5 word description
    prompt: str  # Full task prompt
    subagent_type: str = "general-purpose"
    boundary: Optional[ParallelWorkBoundary] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for Task tool parameters."""
        return {
            "description": self.description,
            "prompt": self.prompt,
            "subagent_type": self.subagent_type,
        }


class ClaudeCodeSpawner(AgentSpawner):
    """
    Production spawner that generates Task tool configurations.

    This spawner is designed for use within Claude Code sessions where
    the Task tool is the mechanism for spawning sub-agents.

    Key differences from SequentialSpawner:
    - Does not execute tasks directly
    - Generates configurations for Task tool calls
    - Requires explicit result recording after Task completion
    - Supports parallel execution when multiple Task tools are called together

    Workflow:
        1. spawn() or prepare_agents() generates configurations
        2. Orchestrator executes Task tools (outside Python)
        3. record_result() captures outputs
        4. get_result() returns parsed AgentResult
    """

    # Template for generating sub-agent prompts
    PROMPT_TEMPLATE = '''## Task: {task_title}

You are working on branch `{branch}` in the repository.

### Your Boundary
**Files you OWN (can create/modify/delete):**
{files_owned}

**Files you can READ (but not modify):**
{files_read_only}

### Task Description
{task_description}

### Requirements
{requirements}

### Deliverables
1. Implement the requested changes within your boundary
2. Add tests for new functionality
3. Verify tests pass: `python -m pytest {test_pattern} -v`
4. Report what files you modified/created and test results

**IMPORTANT:**
- DO NOT modify any files outside your boundary
- DO NOT commit changes - just implement and report results
- Include a summary of files changed in your final output

### Output Format
At the end of your response, include:
```
FILES_MODIFIED: file1.py, file2.py
FILES_CREATED: new_file.py
FILES_DELETED: old_file.py
TESTS_PASSED: 15/15
STATUS: SUCCESS or FAILURE
```
'''

    def __init__(self, branch: str = "main"):
        """
        Initialize the spawner.

        Args:
            branch: Git branch agents should work on
        """
        self.branch = branch
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, AgentResult] = {}
        self._configs: Dict[str, TaskToolConfig] = {}
        self._counter = 0

    def _generate_agent_id(self) -> str:
        """Generate a unique agent ID."""
        self._counter += 1
        short_uuid = str(uuid.uuid4())[:8]
        return f"agent-{self._counter:03d}-{short_uuid}"

    def _format_file_list(self, files: set, indent: str = "- ") -> str:
        """Format a set of files as a bulleted list."""
        if not files:
            return f"{indent}(none)"
        return "\n".join(f"{indent}`{f}`" for f in sorted(files))

    def _generate_prompt(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
        requirements: Optional[str] = None,
    ) -> str:
        """Generate a detailed prompt for a sub-agent."""
        # Extract a short title from the task
        task_title = task.split('\n')[0][:60]
        if len(task.split('\n')[0]) > 60:
            task_title += "..."

        # Determine test pattern based on owned files
        test_files = [f for f in boundary.files_owned if 'test' in f.lower()]
        if test_files:
            test_pattern = ' '.join(test_files)
        else:
            # Guess test location from source files
            test_pattern = "tests/"

        return self.PROMPT_TEMPLATE.format(
            task_title=task_title,
            branch=self.branch,
            files_owned=self._format_file_list(boundary.files_owned),
            files_read_only=self._format_file_list(boundary.files_read_only),
            task_description=task,
            requirements=requirements or "Follow existing code patterns and conventions.",
            test_pattern=test_pattern,
        )

    def _extract_short_description(self, task: str) -> str:
        """Extract a 3-5 word description from a task."""
        # Get first line, first few words
        first_line = task.split('\n')[0].strip()
        words = first_line.split()[:5]
        desc = ' '.join(words)
        if len(desc) > 40:
            desc = desc[:37] + "..."
        return desc

    def spawn(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
        timeout_seconds: int = 300
    ) -> str:
        """
        Prepare a spawn configuration for a task.

        Note: This doesn't actually spawn the agent - it prepares the
        configuration. Use get_config() to get the Task tool parameters,
        then record_result() after execution.

        Args:
            task: Description of what the agent should do
            boundary: Work boundary defining files the agent can modify
            timeout_seconds: Maximum time for agent execution (stored for reference)

        Returns:
            Agent ID for tracking
        """
        agent_id = self._generate_agent_id()

        prompt = self._generate_prompt(task, boundary)
        description = self._extract_short_description(task)

        config = TaskToolConfig(
            agent_id=agent_id,
            description=description,
            prompt=prompt,
            boundary=boundary,
        )

        self._configs[agent_id] = config
        self._agents[agent_id] = {
            "task": task,
            "boundary": boundary,
            "timeout_seconds": timeout_seconds,
            "status": AgentStatus.PENDING,
            "created_at": datetime.now(),
        }

        return agent_id

    def prepare_agents(
        self,
        tasks_and_boundaries: List[Tuple[str, ParallelWorkBoundary]],
        requirements: Optional[str] = None,
    ) -> List[TaskToolConfig]:
        """
        Prepare multiple agents for parallel execution.

        This is a convenience method that generates all configurations
        at once, making it easy to call multiple Task tools in parallel.

        Args:
            tasks_and_boundaries: List of (task, boundary) tuples
            requirements: Optional shared requirements for all agents

        Returns:
            List of TaskToolConfig ready for Task tool invocation
        """
        configs = []
        for task, boundary in tasks_and_boundaries:
            agent_id = self.spawn(task, boundary)
            config = self._configs[agent_id]

            # Override prompt with custom requirements if provided
            if requirements:
                config.prompt = self._generate_prompt(task, boundary, requirements)

            configs.append(config)

        return configs

    def get_config(self, agent_id: str) -> Optional[TaskToolConfig]:
        """Get the Task tool configuration for an agent."""
        return self._configs.get(agent_id)

    def get_all_configs(self) -> Dict[str, TaskToolConfig]:
        """Get all pending agent configurations."""
        return {
            agent_id: config
            for agent_id, config in self._configs.items()
            if self._agents.get(agent_id, {}).get("status") == AgentStatus.PENDING
        }

    def get_status(self, agent_id: str) -> AgentStatus:
        """Get current status of an agent."""
        if agent_id in self._results:
            return self._results[agent_id].status
        if agent_id in self._agents:
            return self._agents[agent_id].get("status", AgentStatus.PENDING)
        return AgentStatus.PENDING

    def get_result(self, agent_id: str) -> Optional[AgentResult]:
        """Get result from a completed agent."""
        return self._results.get(agent_id)

    def wait_for(self, agent_id: str, timeout_seconds: int = 300) -> AgentResult:
        """
        Wait for an agent to complete.

        Note: In ClaudeCodeSpawner, this is a no-op since Task tool calls
        are synchronous from the orchestrator's perspective. The result
        must be recorded via record_result() first.

        Raises:
            ValueError: If result hasn't been recorded yet
        """
        if agent_id not in self._results:
            raise ValueError(
                f"Agent {agent_id} has no recorded result. "
                "Call record_result() after Task tool completes."
            )
        return self._results[agent_id]

    def record_result(
        self,
        agent_id: str,
        output: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> AgentResult:
        """
        Record the result from a completed Task tool call.

        Parses the agent output to extract file changes and status.

        Args:
            agent_id: ID of the agent
            output: Full output from the Task tool
            success: Whether the task completed successfully
            error: Error message if failed

        Returns:
            Parsed AgentResult
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        agent_info = self._agents[agent_id]
        started_at = agent_info.get("created_at", datetime.now())
        completed_at = datetime.now()

        # Parse output for file changes
        files_modified = self._parse_file_list(output, "FILES_MODIFIED")
        files_created = self._parse_file_list(output, "FILES_CREATED")
        files_deleted = self._parse_file_list(output, "FILES_DELETED")

        # Check for status in output
        status_match = re.search(r"STATUS:\s*(SUCCESS|FAILURE)", output, re.IGNORECASE)
        if status_match:
            success = status_match.group(1).upper() == "SUCCESS"

        result = AgentResult(
            agent_id=agent_id,
            status=AgentStatus.COMPLETED if success else AgentStatus.FAILED,
            task_description=agent_info["task"],
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            output=output,
            error=error,
            duration_seconds=(completed_at - started_at).total_seconds(),
            started_at=started_at,
            completed_at=completed_at,
        )

        # Validate boundary compliance
        boundary = agent_info.get("boundary")
        if boundary:
            violations = self._check_boundary_violations(result, boundary)
            if violations:
                result.error = f"Boundary violations: {', '.join(violations)}"

        self._results[agent_id] = result
        self._agents[agent_id]["status"] = result.status

        return result

    def _parse_file_list(self, output: str, marker: str) -> List[str]:
        """Parse a comma-separated file list from agent output."""
        pattern = rf"{marker}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, output, re.IGNORECASE)
        if not match:
            return []

        files_str = match.group(1).strip()
        if files_str.lower() in ("none", "(none)", ""):
            return []

        # Split by comma, clean up whitespace
        files = [f.strip() for f in files_str.split(",")]
        return [f for f in files if f and f.lower() != "none"]

    def _check_boundary_violations(
        self,
        result: AgentResult,
        boundary: ParallelWorkBoundary,
    ) -> List[str]:
        """Check if agent modified files outside its boundary."""
        violations = []
        for f in result.all_modified_files():
            if not boundary.can_modify(f):
                violations.append(f)
        return violations

    def mark_running(self, agent_id: str) -> None:
        """Mark an agent as running (Task tool invoked)."""
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = AgentStatus.RUNNING

    def mark_timed_out(self, agent_id: str) -> None:
        """Mark an agent as timed out."""
        if agent_id not in self._results:
            agent_info = self._agents.get(agent_id, {})
            self._results[agent_id] = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.TIMED_OUT,
                task_description=agent_info.get("task", "Unknown"),
                error="Agent timed out",
            )
            self._agents[agent_id]["status"] = AgentStatus.TIMED_OUT

    def get_pending_agents(self) -> List[str]:
        """Get list of agents that haven't been executed yet."""
        return [
            agent_id
            for agent_id, info in self._agents.items()
            if info.get("status") == AgentStatus.PENDING
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all agents and their statuses."""
        statuses = {}
        for agent_id, info in self._agents.items():
            status = info.get("status", AgentStatus.PENDING)
            statuses[status.name] = statuses.get(status.name, 0) + 1

        return {
            "total_agents": len(self._agents),
            "by_status": statuses,
            "pending": self.get_pending_agents(),
            "completed": [
                agent_id
                for agent_id, result in self._results.items()
                if result.status == AgentStatus.COMPLETED
            ],
            "failed": [
                agent_id
                for agent_id, result in self._results.items()
                if result.status == AgentStatus.FAILED
            ],
        }


def generate_parallel_task_calls(
    spawner: ClaudeCodeSpawner,
    tasks_and_boundaries: List[Tuple[str, ParallelWorkBoundary]],
) -> str:
    """
    Generate markdown documentation for Task tool calls.

    This helper generates the Task tool configurations in a format
    that can be easily copied into parallel Task tool invocations.

    Args:
        spawner: ClaudeCodeSpawner instance
        tasks_and_boundaries: List of (task, boundary) tuples

    Returns:
        Markdown string with Task tool configurations
    """
    configs = spawner.prepare_agents(tasks_and_boundaries)

    lines = ["## Parallel Task Tool Configurations\n"]
    lines.append("Call these Task tools in parallel (single message with multiple tool uses):\n")

    for i, config in enumerate(configs, 1):
        lines.append(f"### Agent {i}: {config.agent_id}")
        lines.append(f"- **Description**: {config.description}")
        lines.append(f"- **Type**: {config.subagent_type}")
        lines.append("\n**Prompt preview** (first 200 chars):")
        lines.append(f"```\n{config.prompt[:200]}...\n```\n")

    return "\n".join(lines)


# =============================================================================
# SUBPROCESS SPAWNING - PRODUCTION IMPLEMENTATION
# =============================================================================


@dataclass
class SpawnResult:
    """
    Result from a spawned subprocess execution.

    This is the return type for synchronous spawn operations.
    """
    success: bool
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    exit_code: int = 0


@dataclass
class SpawnHandle:
    """
    Handle for an asynchronously spawned agent.

    Allows polling and waiting for completion without blocking.
    """
    agent_id: str
    process: subprocess.Popen
    started_at: datetime
    prompt_file: Optional[Path] = None
    timeout_seconds: float = 300.0
    _output_lines: List[str] = field(default_factory=list)
    _error_lines: List[str] = field(default_factory=list)
    _completed: bool = False
    _result: Optional[SpawnResult] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_running(self) -> bool:
        """Check if the process is still running."""
        with self._lock:
            if self._completed:
                return False
            return self.process.poll() is None

    def get_result(self) -> Optional[SpawnResult]:
        """Get result if completed, None otherwise."""
        with self._lock:
            return self._result

    def poll(self) -> Optional[SpawnResult]:
        """
        Poll for completion without blocking.

        Returns:
            SpawnResult if completed, None if still running
        """
        with self._lock:
            if self._completed:
                return self._result

            retcode = self.process.poll()
            if retcode is None:
                return None  # Still running

            # Process completed
            self._complete(retcode)
            return self._result

    def _complete(self, exit_code: int) -> None:
        """Mark as completed and store result (called with lock held)."""
        if self._completed:
            return

        duration = (datetime.now() - self.started_at).total_seconds()

        # Collect any remaining output
        try:
            stdout, stderr = self.process.communicate(timeout=1.0)
            if stdout:
                self._output_lines.extend(stdout.decode('utf-8', errors='replace').splitlines())
            if stderr:
                self._error_lines.extend(stderr.decode('utf-8', errors='replace').splitlines())
        except subprocess.TimeoutExpired:
            pass

        output = '\n'.join(self._output_lines)
        error_output = '\n'.join(self._error_lines) if self._error_lines else None

        self._result = SpawnResult(
            success=(exit_code == 0),
            output=output,
            error=error_output,
            duration_seconds=duration,
            exit_code=exit_code,
        )
        self._completed = True

        # Clean up temp file
        if self.prompt_file and self.prompt_file.exists():
            try:
                self.prompt_file.unlink()
            except OSError:
                pass

    def wait(self, timeout_seconds: Optional[float] = None) -> SpawnResult:
        """
        Wait for completion with optional timeout.

        Args:
            timeout_seconds: Override timeout, or None to use default

        Returns:
            SpawnResult

        Raises:
            subprocess.TimeoutExpired: If timeout exceeded
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds

        try:
            retcode = self.process.wait(timeout=timeout)
            with self._lock:
                self._complete(retcode)
                return self._result
        except subprocess.TimeoutExpired:
            # Kill the process
            self.terminate()
            raise

    def terminate(self, grace_period: float = 5.0) -> None:
        """
        Gracefully terminate the process.

        Args:
            grace_period: Seconds to wait before SIGKILL
        """
        with self._lock:
            if self._completed:
                return

            # Try graceful shutdown first
            try:
                self.process.terminate()
                self.process.wait(timeout=grace_period)
            except subprocess.TimeoutExpired:
                # Force kill
                self.process.kill()
                self.process.wait()

            self._complete(self.process.returncode)


@dataclass
class SpawnMetrics:
    """Metrics for spawned agents."""
    total_spawned: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    total_duration_seconds: float = 0.0
    success_rate: float = 0.0
    avg_duration_seconds: float = 0.0
    peak_concurrent: int = 0
    current_active: int = 0

    def update_from_result(self, result: SpawnResult, timed_out: bool = False) -> None:
        """Update metrics from a result."""
        self.completed += 1
        if timed_out:
            self.timed_out += 1
            self.failed += 1
        elif not result.success:
            self.failed += 1

        self.total_duration_seconds += result.duration_seconds
        self.avg_duration_seconds = self.total_duration_seconds / self.completed
        self.success_rate = (self.completed - self.failed) / self.completed if self.completed > 0 else 0.0


class SubprocessClaudeCodeSpawner(AgentSpawner):
    """
    Production spawner that spawns actual Claude Code CLI subprocesses.

    This spawner provides:
    - Actual subprocess spawning of claude-code CLI
    - Proper process isolation and cleanup
    - Timeout handling and graceful termination
    - Output capture (stdout/stderr)
    - Concurrency limiting via semaphore
    - Performance metrics tracking
    - Context passing via temp files or stdin

    Usage:
        spawner = SubprocessClaudeCodeSpawner(
            max_concurrent=5,
            default_timeout=300.0,
            working_dir=Path("/path/to/repo")
        )

        # Synchronous spawn
        agent_id = spawner.spawn("Implement feature", boundary)
        result = spawner.wait_for(agent_id, timeout_seconds=300)

        # Async spawn
        agent_id, handle = spawner.spawn_async("Implement feature", boundary)
        # ... do other work ...
        result = handle.wait()

        # Check metrics
        metrics = spawner.get_metrics()
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        default_timeout: float = 300.0,
        working_dir: Optional[Path] = None,
        claude_code_path: Optional[str] = None,
        branch: str = "main",
    ):
        """
        Initialize the subprocess spawner.

        Args:
            max_concurrent: Maximum number of concurrent subprocesses
            default_timeout: Default timeout in seconds
            working_dir: Working directory for spawned processes
            claude_code_path: Path to claude-code CLI (auto-detected if None)
            branch: Git branch agents should work on
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.branch = branch

        # Auto-detect claude-code CLI path
        if claude_code_path:
            self.claude_code_path = claude_code_path
        else:
            # Try to find in PATH
            self.claude_code_path = shutil.which("claude-code") or shutil.which("claude")
            if not self.claude_code_path:
                raise RuntimeError(
                    "Could not find claude-code CLI in PATH. "
                    "Please specify claude_code_path explicitly."
                )

        # State tracking
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._handles: Dict[str, SpawnHandle] = {}
        self._results: Dict[str, AgentResult] = {}
        self._counter = 0
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()

        # Metrics
        self._metrics = SpawnMetrics()

        # Cleanup tracking
        self._temp_files: Set[Path] = set()

    def _generate_agent_id(self) -> str:
        """Generate a unique agent ID."""
        with self._lock:
            self._counter += 1
            short_uuid = str(uuid.uuid4())[:8]
            return f"subprocess-{self._counter:03d}-{short_uuid}"

    def _create_prompt_file(self, prompt: str) -> Path:
        """
        Create a temporary file containing the prompt.

        Args:
            prompt: The full prompt text

        Returns:
            Path to the temp file
        """
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="claude_prompt_")
        path_obj = Path(path)

        try:
            with os.fdopen(fd, 'w') as f:
                f.write(prompt)
            self._temp_files.add(path_obj)
            return path_obj
        except (OSError, UnicodeEncodeError) as e:
            # Clean up temp file on failure, then re-raise
            path_obj.unlink(missing_ok=True)
            raise

    def _build_command(
        self,
        prompt_file: Path,
        context_files: Optional[List[Path]] = None
    ) -> List[str]:
        """
        Build the claude-code CLI command.

        Args:
            prompt_file: Path to file containing the prompt
            context_files: Optional list of context files to include

        Returns:
            Command as list of strings
        """
        cmd = [self.claude_code_path]

        # Add prompt from file
        cmd.extend(["--prompt-file", str(prompt_file)])

        # Add context files if provided
        if context_files:
            for ctx_file in context_files:
                if ctx_file.exists():
                    cmd.extend(["--file", str(ctx_file)])

        # Set working directory
        cmd.extend(["--cwd", str(self.working_dir)])

        return cmd

    def _generate_prompt(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
    ) -> str:
        """Generate a detailed prompt for a sub-agent (reuses ClaudeCodeSpawner template)."""
        # Reuse the existing template from ClaudeCodeSpawner
        task_title = task.split('\n')[0][:60]
        if len(task.split('\n')[0]) > 60:
            task_title += "..."

        test_files = [f for f in boundary.files_owned if 'test' in f.lower()]
        test_pattern = ' '.join(test_files) if test_files else "tests/"

        files_owned = "\n".join(f"- `{f}`" for f in sorted(boundary.files_owned)) or "- (none)"
        files_read_only = "\n".join(f"- `{f}`" for f in sorted(boundary.files_read_only)) or "- (none)"

        return f'''## Task: {task_title}

You are working on branch `{self.branch}` in the repository.

### Your Boundary
**Files you OWN (can create/modify/delete):**
{files_owned}

**Files you can READ (but not modify):**
{files_read_only}

### Task Description
{task}

### Requirements
Follow existing code patterns and conventions.

### Deliverables
1. Implement the requested changes within your boundary
2. Add tests for new functionality
3. Verify tests pass: `python -m pytest {test_pattern} -v`
4. Report what files you modified/created and test results

**IMPORTANT:**
- DO NOT modify any files outside your boundary
- DO NOT commit changes - just implement and report results
- Include a summary of files changed in your final output

### Output Format
At the end of your response, include:
```
FILES_MODIFIED: file1.py, file2.py
FILES_CREATED: new_file.py
FILES_DELETED: old_file.py
TESTS_PASSED: 15/15
STATUS: SUCCESS or FAILURE
```
'''

    def spawn(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
        timeout_seconds: int = 300,
        context_files: Optional[List[Path]] = None,
    ) -> str:
        """
        Spawn an agent subprocess (synchronous - waits for completion).

        Args:
            task: Description of what the agent should do
            boundary: Work boundary defining files the agent can modify
            timeout_seconds: Maximum time for agent execution
            context_files: Optional list of context files to include

        Returns:
            Agent ID for tracking

        Note:
            This is the synchronous version - it blocks until completion.
            For async spawning, use spawn_async().
        """
        agent_id, handle = self.spawn_async(task, boundary, timeout_seconds, context_files)

        # Wait for completion
        try:
            handle.wait(timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired:
            self.mark_timed_out(agent_id)

        return agent_id

    def spawn_async(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
        timeout_seconds: int = 300,
        context_files: Optional[List[Path]] = None,
    ) -> Tuple[str, SpawnHandle]:
        """
        Spawn an agent subprocess asynchronously.

        Args:
            task: Description of what the agent should do
            boundary: Work boundary defining files the agent can modify
            timeout_seconds: Maximum time for agent execution
            context_files: Optional list of context files to include

        Returns:
            Tuple of (agent_id, SpawnHandle)

        Usage:
            agent_id, handle = spawner.spawn_async("Implement auth", boundary)
            # ... do other work ...
            result = handle.wait()
        """
        agent_id = self._generate_agent_id()

        # Acquire semaphore to limit concurrency
        self._semaphore.acquire()

        try:
            # Generate prompt
            prompt = self._generate_prompt(task, boundary)
            prompt_file = self._create_prompt_file(prompt)

            # Build command
            cmd = self._build_command(prompt_file, context_files)

            # Start subprocess
            started_at = datetime.now()
            process = subprocess.Popen(
                cmd,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # We'll decode manually
            )

            # Create handle
            handle = SpawnHandle(
                agent_id=agent_id,
                process=process,
                started_at=started_at,
                prompt_file=prompt_file,
                timeout_seconds=timeout_seconds,
            )

            # Track state
            with self._lock:
                self._agents[agent_id] = {
                    "task": task,
                    "boundary": boundary,
                    "timeout_seconds": timeout_seconds,
                    "status": AgentStatus.RUNNING,
                    "started_at": started_at,
                }
                self._handles[agent_id] = handle

                # Update metrics
                self._metrics.total_spawned += 1
                self._metrics.current_active += 1
                if self._metrics.current_active > self._metrics.peak_concurrent:
                    self._metrics.peak_concurrent = self._metrics.current_active

            return agent_id, handle

        except Exception as e:
            # Release semaphore on failure
            self._semaphore.release()
            raise RuntimeError(f"Failed to spawn agent: {e}") from e

    def get_status(self, agent_id: str) -> AgentStatus:
        """Get current status of an agent."""
        with self._lock:
            if agent_id in self._results:
                return self._results[agent_id].status
            if agent_id in self._agents:
                return self._agents[agent_id].get("status", AgentStatus.PENDING)
            return AgentStatus.PENDING

    def get_result(self, agent_id: str) -> Optional[AgentResult]:
        """
        Get result from a completed agent.

        Returns None if agent is still running or doesn't exist.
        """
        with self._lock:
            return self._results.get(agent_id)

    def wait_for(self, agent_id: str, timeout_seconds: int = 300) -> AgentResult:
        """
        Wait for an agent to complete and return its result.

        Args:
            agent_id: The agent ID
            timeout_seconds: Maximum time to wait

        Returns:
            AgentResult

        Raises:
            ValueError: If agent doesn't exist
            subprocess.TimeoutExpired: If timeout exceeded
        """
        handle = self._handles.get(agent_id)
        if not handle:
            raise ValueError(f"Unknown agent: {agent_id}")

        try:
            # Wait for subprocess completion
            spawn_result = handle.wait(timeout_seconds=timeout_seconds)

            # Convert to AgentResult
            result = self._convert_spawn_result(agent_id, spawn_result)

            # Store result
            with self._lock:
                self._results[agent_id] = result
                self._agents[agent_id]["status"] = result.status
                self._metrics.current_active -= 1
                self._metrics.update_from_result(spawn_result, timed_out=False)

            return result

        except subprocess.TimeoutExpired:
            # Mark as timed out
            self.mark_timed_out(agent_id)
            raise

        finally:
            # Release semaphore
            self._semaphore.release()

    def _convert_spawn_result(
        self,
        agent_id: str,
        spawn_result: SpawnResult
    ) -> AgentResult:
        """Convert a SpawnResult to AgentResult with file parsing."""
        agent_info = self._agents.get(agent_id, {})

        # Parse output for file changes
        files_modified = self._parse_file_list(spawn_result.output, "FILES_MODIFIED")
        files_created = self._parse_file_list(spawn_result.output, "FILES_CREATED")
        files_deleted = self._parse_file_list(spawn_result.output, "FILES_DELETED")

        # Check for status in output
        status_match = re.search(r"STATUS:\s*(SUCCESS|FAILURE)", spawn_result.output, re.IGNORECASE)
        if status_match:
            success = status_match.group(1).upper() == "SUCCESS"
        else:
            success = spawn_result.success

        status = AgentStatus.COMPLETED if success else AgentStatus.FAILED

        result = AgentResult(
            agent_id=agent_id,
            status=status,
            task_description=agent_info.get("task", "Unknown"),
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            output=spawn_result.output,
            error=spawn_result.error,
            duration_seconds=spawn_result.duration_seconds,
            started_at=agent_info.get("started_at"),
            completed_at=datetime.now(),
        )

        # Validate boundary compliance
        boundary = agent_info.get("boundary")
        if boundary:
            violations = self._check_boundary_violations(result, boundary)
            if violations:
                result.error = f"Boundary violations: {', '.join(violations)}"

        return result

    def _parse_file_list(self, output: str, marker: str) -> List[str]:
        """Parse a comma-separated file list from agent output."""
        pattern = rf"{marker}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, output, re.IGNORECASE)
        if not match:
            return []

        files_str = match.group(1).strip()
        if files_str.lower() in ("none", "(none)", ""):
            return []

        files = [f.strip() for f in files_str.split(",")]
        return [f for f in files if f and f.lower() != "none"]

    def _check_boundary_violations(
        self,
        result: AgentResult,
        boundary: ParallelWorkBoundary,
    ) -> List[str]:
        """Check if agent modified files outside its boundary."""
        violations = []
        for f in result.all_modified_files():
            if not boundary.can_modify(f):
                violations.append(f)
        return violations

    def mark_timed_out(self, agent_id: str) -> None:
        """Mark an agent as timed out and terminate its process."""
        handle = self._handles.get(agent_id)
        if handle:
            handle.terminate()

            spawn_result = handle.get_result()
            if spawn_result:
                # Create timed out result
                agent_info = self._agents.get(agent_id, {})
                result = AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.TIMED_OUT,
                    task_description=agent_info.get("task", "Unknown"),
                    error=f"Agent timed out after {spawn_result.duration_seconds:.1f}s",
                    output=spawn_result.output,
                    duration_seconds=spawn_result.duration_seconds,
                    started_at=agent_info.get("started_at"),
                    completed_at=datetime.now(),
                )

                with self._lock:
                    self._results[agent_id] = result
                    self._agents[agent_id]["status"] = AgentStatus.TIMED_OUT
                    self._metrics.current_active -= 1
                    self._metrics.update_from_result(spawn_result, timed_out=True)

            # Release semaphore
            self._semaphore.release()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get spawn success rates, durations, etc.

        Returns:
            Dictionary with metrics:
            - total_spawned: Total agents spawned
            - completed: Successfully completed agents
            - failed: Failed agents
            - timed_out: Timed out agents
            - success_rate: Success rate (0.0-1.0)
            - avg_duration_seconds: Average duration
            - peak_concurrent: Peak concurrent agents
            - current_active: Currently running agents
        """
        with self._lock:
            return {
                "total_spawned": self._metrics.total_spawned,
                "completed": self._metrics.completed,
                "failed": self._metrics.failed,
                "timed_out": self._metrics.timed_out,
                "success_rate": self._metrics.success_rate,
                "avg_duration_seconds": self._metrics.avg_duration_seconds,
                "total_duration_seconds": self._metrics.total_duration_seconds,
                "peak_concurrent": self._metrics.peak_concurrent,
                "current_active": self._metrics.current_active,
            }

    def cleanup(self) -> None:
        """Clean up all temp files and terminate any running processes."""
        # Terminate all running processes
        with self._lock:
            for handle in self._handles.values():
                if handle.is_running():
                    handle.terminate()

        # Clean up temp files
        for temp_file in self._temp_files:
            temp_file.unlink(missing_ok=True)
        self._temp_files.clear()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except OSError:
            # Ignore file/process errors during cleanup - can occur during
            # interpreter shutdown when resources may already be released
            pass
