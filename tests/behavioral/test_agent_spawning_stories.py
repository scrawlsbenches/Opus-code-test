"""
Behavioral Tests for Agent Spawning and Coordination.

This module tests the agent spawning infrastructure we built ourselves
for parallel work coordination and subprocess management.

Epic: Orchestrator manages parallel agent workforce
Story: As an orchestrator coordinating parallel work,
       I want agent spawning we implemented ourselves,
       So that I control distributed execution completely.
"""

import pytest
from pathlib import Path
from cortical.reasoning.claude_code_spawner import (
    ClaudeCodeSpawner,
    TaskToolConfig,
)
from cortical.reasoning.collaboration import (
    ParallelWorkBoundary,
    AgentStatus,
)


class OrchestratorManagesParallelAgents:
    """
    Epic: Orchestrator Manages Parallel Agent Workforce

    As an orchestrator building parallel execution,
    I want spawning infrastructure we built ourselves,
    So that I control agent lifecycle completely.
    """

    def test_scenario_spawner_prepares_task_configurations(self):
        """
        Scenario: Spawner generates agent task configurations

        Given work to distribute we're coordinating
        When I prepare agent tasks
        Then configurations are ready for execution
        Because we built task generation ourselves
        """
        # Given work to distribute
        spawner = ClaudeCodeSpawner(branch="main")
        boundary = ParallelWorkBoundary(
            files_owned={"custom/indexer.py", "tests/test_indexer.py"},
            files_read_only={"custom/base.py"}
        )

        # When preparing task
        agent_id = spawner.spawn(
            "Implement custom inverted index we're building",
            boundary,
            timeout_seconds=300
        )

        # Then configuration ready
        assert agent_id is not None
        config = spawner.get_config(agent_id)
        assert isinstance(config, TaskToolConfig)
        assert "inverted index" in config.prompt.lower()
        assert "custom/indexer.py" in config.prompt

    def test_scenario_spawner_enforces_work_boundaries(self):
        """
        Scenario: Agent boundaries prevent file conflicts

        Given agents with defined boundaries we set
        When work is assigned
        Then boundaries are encoded in prompts
        Because we prevent conflicts ourselves
        """
        # Given defined boundaries
        spawner = ClaudeCodeSpawner()
        boundary = ParallelWorkBoundary(
            files_owned={"module_a/impl.py"},
            files_read_only={"shared/types.py"},
            files_forbidden={"module_b/impl.py"}
        )

        # When assigning work
        agent_id = spawner.spawn(
            "Build custom module A implementation",
            boundary
        )

        # Then boundaries encoded
        config = spawner.get_config(agent_id)
        assert "module_a/impl.py" in config.prompt
        assert "shared/types.py" in config.prompt
        assert "can READ (but not modify)" in config.prompt

    def test_scenario_spawner_tracks_agent_status(self):
        """
        Scenario: Agent status is tracked throughout lifecycle

        Given agents we're managing
        When their status changes
        Then transitions are tracked
        Because we monitor lifecycle ourselves
        """
        # Given managed agents
        spawner = ClaudeCodeSpawner()
        boundary = ParallelWorkBoundary(files_owned={"task.py"})

        agent_id = spawner.spawn("Implement task", boundary)

        # When status changes
        initial_status = spawner.get_status(agent_id)
        spawner.mark_running(agent_id)
        running_status = spawner.get_status(agent_id)

        # Then tracked
        assert initial_status == AgentStatus.PENDING
        assert running_status == AgentStatus.RUNNING

    def test_scenario_spawner_records_execution_results(self):
        """
        Scenario: Agent results are parsed and stored

        Given completed agent work
        When results are recorded
        Then outputs are parsed for artifacts
        Because we built result parsing ourselves
        """
        # Given completed work
        spawner = ClaudeCodeSpawner()
        boundary = ParallelWorkBoundary(files_owned={"output.py"})
        agent_id = spawner.spawn("Generate file", boundary)

        # When recording result
        agent_output = """
        I implemented the feature.

        FILES_MODIFIED: output.py, config.py
        FILES_CREATED: new_module.py
        FILES_DELETED: old_file.py
        TESTS_PASSED: 10/10
        STATUS: SUCCESS
        """

        result = spawner.record_result(agent_id, agent_output, success=True)

        # Then parsed correctly
        assert result.status == AgentStatus.COMPLETED
        assert "output.py" in result.files_modified
        assert "new_module.py" in result.files_created
        assert "old_file.py" in result.files_deleted

    def test_scenario_spawner_detects_boundary_violations(self):
        """
        Scenario: Boundary violations are detected and flagged

        Given agents with strict boundaries we enforced
        When an agent modifies unauthorized files
        Then violation is detected
        Because we validate boundaries ourselves
        """
        # Given strict boundary
        spawner = ClaudeCodeSpawner()
        boundary = ParallelWorkBoundary(
            files_owned={"allowed/file.py"}
        )
        agent_id = spawner.spawn("Respect boundaries", boundary)

        # When agent violates boundary
        output = """
        FILES_MODIFIED: allowed/file.py, forbidden/other.py
        STATUS: SUCCESS
        """
        result = spawner.record_result(agent_id, output)

        # Then violation detected
        assert result.error is not None
        assert "boundary violations" in result.error.lower()
        assert "forbidden/other.py" in result.error


class ParallelCoordinatorDistributesWork:
    """
    Epic: Parallel Coordinator Distributes Complex Work

    As a coordinator managing parallel execution,
    I want work decomposition and coordination,
    So that I maximize throughput we control.
    """

    def test_scenario_coordinator_decomposes_work_into_tasks(self):
        """
        Scenario: Complex work is decomposed into parallel tasks

        Given a complex task we're breaking down
        When I decompose into parallel work
        Then boundaries are non-overlapping
        Because we designed decomposition ourselves
        """
        # Given complex task
        spawner = ClaudeCodeSpawner()

        # When decomposing
        tasks_and_boundaries = [
            ("Build custom indexer", ParallelWorkBoundary(
                files_owned={"indexing/core.py"}
            )),
            ("Build custom ranker", ParallelWorkBoundary(
                files_owned={"ranking/core.py"}
            )),
            ("Build custom parser", ParallelWorkBoundary(
                files_owned={"parsing/core.py"}
            )),
        ]

        configs = spawner.prepare_agents(tasks_and_boundaries)

        # Then non-overlapping boundaries
        assert len(configs) == 3
        all_files = set()
        for config in configs:
            boundary = spawner.get_config(config.agent_id).boundary
            # Check no overlap
            assert not (all_files & boundary.files_owned)
            all_files.update(boundary.files_owned)

    def test_scenario_coordinator_aggregates_results(self):
        """
        Scenario: Agent results are aggregated into summary

        Given multiple completed agents
        When I collect results
        Then aggregate view is available
        Because we built aggregation ourselves
        """
        # Given completed agents
        spawner = ClaudeCodeSpawner()

        boundary1 = ParallelWorkBoundary(files_owned={"task1.py"})
        boundary2 = ParallelWorkBoundary(files_owned={"task2.py"})

        id1 = spawner.spawn("Task 1", boundary1)
        id2 = spawner.spawn("Task 2", boundary2)

        spawner.record_result(id1, "FILES_CREATED: task1.py\nSTATUS: SUCCESS", True)
        spawner.record_result(id2, "FILES_CREATED: task2.py\nSTATUS: SUCCESS", True)

        # When collecting results
        summary = spawner.get_summary()

        # Then aggregate available
        assert summary['total_agents'] == 2
        assert len(summary['completed']) == 2
        assert summary['by_status']['COMPLETED'] == 2


class SubprocessSpawnerManagesRealProcesses:
    """
    Epic: Subprocess Spawner Manages Real Process Execution

    As a system spawning actual subprocesses,
    I want process lifecycle management,
    So that I handle real execution safely.

    Note: These tests describe the subprocess spawner behavior
    but may not execute actual subprocesses in the test environment.
    """

    def test_scenario_spawner_builds_subprocess_commands(self):
        """
        Scenario: Spawner constructs proper subprocess commands

        Given subprocess spawning configuration
        When building command for execution
        Then proper CLI invocation is generated
        Because we built command generation ourselves
        """
        # Given subprocess configuration
        # Note: SubprocessClaudeCodeSpawner requires claude-code in PATH
        # This test just verifies the interface exists

        from cortical.reasoning.claude_code_spawner import SubprocessClaudeCodeSpawner

        # Verify the class and methods exist
        assert hasattr(SubprocessClaudeCodeSpawner, 'spawn')
        assert hasattr(SubprocessClaudeCodeSpawner, 'spawn_async')
        assert hasattr(SubprocessClaudeCodeSpawner, 'wait_for')
        assert hasattr(SubprocessClaudeCodeSpawner, 'get_metrics')

    def test_scenario_spawner_enforces_concurrency_limits(self):
        """
        Scenario: Concurrent subprocess limits prevent overload

        Given a max concurrency limit we set
        When spawning many agents
        Then limit is enforced
        Because we built throttling ourselves
        """
        # Given concurrency limit
        from cortical.reasoning.claude_code_spawner import SubprocessClaudeCodeSpawner

        # Verify concurrency control exists in API
        # (actual subprocess spawning not tested here)
        assert hasattr(SubprocessClaudeCodeSpawner, '__init__')

        # API signature includes max_concurrent parameter
        import inspect
        sig = inspect.signature(SubprocessClaudeCodeSpawner.__init__)
        assert 'max_concurrent' in sig.parameters

    def test_scenario_spawner_handles_timeouts_gracefully(self):
        """
        Scenario: Timed out processes are terminated gracefully

        Given agents with timeout limits we set
        When timeout is exceeded
        Then process is terminated gracefully
        Because we handle timeouts ourselves
        """
        # Given timeout handling
        from cortical.reasoning.claude_code_spawner import SubprocessClaudeCodeSpawner

        # Verify timeout handling in API
        assert hasattr(SubprocessClaudeCodeSpawner, 'mark_timed_out')

        # spawn() includes timeout_seconds parameter
        import inspect
        sig = inspect.signature(SubprocessClaudeCodeSpawner.spawn)
        assert 'timeout_seconds' in sig.parameters

    def test_scenario_spawner_tracks_performance_metrics(self):
        """
        Scenario: Spawner tracks execution metrics

        Given agents completing work
        When I query metrics
        Then performance data is available
        Because we built metrics tracking ourselves
        """
        # Given metrics tracking
        from cortical.reasoning.claude_code_spawner import SubprocessClaudeCodeSpawner

        # Verify metrics API exists
        assert hasattr(SubprocessClaudeCodeSpawner, 'get_metrics')

        # Check what metrics should be available
        import inspect
        sig = inspect.signature(SubprocessClaudeCodeSpawner.get_metrics)
        # Should return Dict with metrics
        assert sig.return_annotation != inspect.Signature.empty


class SpawningSystemEnablesCodeGeneration:
    """
    Epic: Spawning System Enables Parallel Code Generation

    As a developer generating code in parallel,
    I want structured output from agents,
    So that I can integrate generated code.
    """

    def test_scenario_agents_follow_structured_output_format(self):
        """
        Scenario: Agents report results in structured format

        Given agents we instruct
        When they complete work
        Then output follows our format
        Because we designed the output schema ourselves
        """
        # Given agent instructions
        spawner = ClaudeCodeSpawner()
        boundary = ParallelWorkBoundary(files_owned={"gen.py"})

        agent_id = spawner.spawn("Generate code", boundary)
        config = spawner.get_config(agent_id)

        # Then prompt includes output format
        assert "FILES_MODIFIED:" in config.prompt
        assert "FILES_CREATED:" in config.prompt
        assert "FILES_DELETED:" in config.prompt
        assert "STATUS:" in config.prompt

    def test_scenario_result_parsing_handles_edge_cases(self):
        """
        Scenario: Parser handles varied output formats

        Given agents with imperfect output
        When parsing results
        Then parser is robust
        Because we built robust parsing ourselves
        """
        # Given various output formats
        spawner = ClaudeCodeSpawner()
        boundary = ParallelWorkBoundary(files_owned={"test.py"})

        # Test with minimal output
        id1 = spawner.spawn("Task 1", boundary)
        spawner.record_result(id1, "STATUS: SUCCESS")
        result1 = spawner.get_result(id1)
        assert result1.status == AgentStatus.COMPLETED

        # Test with no status
        id2 = spawner.spawn("Task 2", boundary)
        spawner.record_result(id2, "Some output without status marker")
        result2 = spawner.get_result(id2)
        assert result2 is not None

        # Test with "none" files
        id3 = spawner.spawn("Task 3", boundary)
        spawner.record_result(id3, "FILES_MODIFIED: none\nSTATUS: SUCCESS")
        result3 = spawner.get_result(id3)
        assert len(result3.files_modified) == 0

    def test_scenario_spawner_maintains_agent_context(self):
        """
        Scenario: Agent context preserved throughout lifecycle

        Given agents with task context
        When retrieving agent info
        Then full context is available
        Because we track everything ourselves
        """
        # Given agent with context
        spawner = ClaudeCodeSpawner(branch="feature/custom-impl")
        boundary = ParallelWorkBoundary(
            files_owned={"feature/impl.py"},
            files_read_only={"core/base.py"}
        )

        agent_id = spawner.spawn(
            "Implement custom feature we're building",
            boundary,
            timeout_seconds=600
        )

        # When retrieving context
        config = spawner.get_config(agent_id)

        # Then context preserved
        assert "feature/custom-impl" in config.prompt
        assert "custom feature" in config.prompt
        assert "feature/impl.py" in config.prompt
        assert "core/base.py" in config.prompt
