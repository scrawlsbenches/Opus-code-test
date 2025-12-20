"""
ClaudeCodeSpawner - Production spawner for Claude Code sub-agents.

This module provides the production implementation of AgentSpawner that
generates Task tool configurations for spawning real Claude Code sub-agents.

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
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

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
