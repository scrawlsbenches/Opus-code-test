"""
Tests for ClaudeCodeSpawner - production spawner for Claude Code sub-agents.
"""

import unittest
from datetime import datetime

from cortical.reasoning.claude_code_spawner import (
    ClaudeCodeSpawner,
    TaskToolConfig,
    generate_parallel_task_calls,
)
from cortical.reasoning.collaboration import (
    AgentStatus,
    AgentResult,
    ParallelWorkBoundary,
)


class TestTaskToolConfig(unittest.TestCase):
    """Tests for TaskToolConfig dataclass."""

    def test_basic_config(self):
        """Test creating a basic config."""
        config = TaskToolConfig(
            agent_id="agent-001",
            description="Implement auth",
            prompt="Full prompt here",
        )
        self.assertEqual(config.agent_id, "agent-001")
        self.assertEqual(config.description, "Implement auth")
        self.assertEqual(config.subagent_type, "general-purpose")

    def test_to_dict(self):
        """Test converting config to dict."""
        config = TaskToolConfig(
            agent_id="agent-001",
            description="Test task",
            prompt="Do the thing",
            subagent_type="general-purpose",
        )
        d = config.to_dict()
        self.assertEqual(d["description"], "Test task")
        self.assertEqual(d["prompt"], "Do the thing")
        self.assertEqual(d["subagent_type"], "general-purpose")
        # agent_id not in dict (it's for internal tracking)
        self.assertNotIn("agent_id", d)


class TestClaudeCodeSpawner(unittest.TestCase):
    """Tests for ClaudeCodeSpawner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.spawner = ClaudeCodeSpawner(branch="test-branch")
        self.boundary = ParallelWorkBoundary(
            agent_id="test-agent",
            scope_description="Test implementation",
            files_owned={"src/auth.py", "src/middleware.py"},
            files_read_only={"config.py"},
        )

    def test_spawn_returns_agent_id(self):
        """Test that spawn returns a unique agent ID."""
        agent_id = self.spawner.spawn("Implement feature", self.boundary)
        self.assertIsNotNone(agent_id)
        self.assertTrue(agent_id.startswith("agent-"))

    def test_spawn_generates_unique_ids(self):
        """Test that multiple spawns generate unique IDs."""
        ids = set()
        for i in range(10):
            agent_id = self.spawner.spawn(f"Task {i}", self.boundary)
            self.assertNotIn(agent_id, ids)
            ids.add(agent_id)
        self.assertEqual(len(ids), 10)

    def test_spawn_creates_config(self):
        """Test that spawn creates a TaskToolConfig."""
        agent_id = self.spawner.spawn("Implement auth", self.boundary)
        config = self.spawner.get_config(agent_id)

        self.assertIsNotNone(config)
        self.assertEqual(config.agent_id, agent_id)
        self.assertIn("Implement auth", config.prompt)
        self.assertIn("test-branch", config.prompt)

    def test_spawn_includes_boundary_in_prompt(self):
        """Test that prompt includes file boundaries."""
        agent_id = self.spawner.spawn("Implement auth", self.boundary)
        config = self.spawner.get_config(agent_id)

        # Should list owned files
        self.assertIn("src/auth.py", config.prompt)
        self.assertIn("src/middleware.py", config.prompt)
        # Should list read-only files
        self.assertIn("config.py", config.prompt)

    def test_get_status_pending(self):
        """Test that new agents have PENDING status."""
        agent_id = self.spawner.spawn("Task", self.boundary)
        status = self.spawner.get_status(agent_id)
        self.assertEqual(status, AgentStatus.PENDING)

    def test_get_status_unknown_agent(self):
        """Test status of unknown agent."""
        status = self.spawner.get_status("nonexistent")
        self.assertEqual(status, AgentStatus.PENDING)

    def test_get_result_before_recording(self):
        """Test that get_result returns None before recording."""
        agent_id = self.spawner.spawn("Task", self.boundary)
        result = self.spawner.get_result(agent_id)
        self.assertIsNone(result)

    def test_record_result_success(self):
        """Test recording a successful result."""
        agent_id = self.spawner.spawn("Implement feature", self.boundary)

        output = """
        Task completed successfully.
        FILES_MODIFIED: src/auth.py
        FILES_CREATED: tests/test_auth.py
        STATUS: SUCCESS
        """

        result = self.spawner.record_result(agent_id, output)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.files_modified, ["src/auth.py"])
        self.assertEqual(result.files_created, ["tests/test_auth.py"])
        self.assertIn("Task completed", result.output)

    def test_record_result_failure(self):
        """Test recording a failed result."""
        agent_id = self.spawner.spawn("Task", self.boundary)

        output = """
        Error occurred during implementation.
        STATUS: FAILURE
        """

        result = self.spawner.record_result(agent_id, output, success=False)

        self.assertEqual(result.status, AgentStatus.FAILED)

    def test_record_result_parses_status_from_output(self):
        """Test that STATUS in output overrides success param."""
        agent_id = self.spawner.spawn("Task", self.boundary)

        # Pass success=True but output says FAILURE
        output = "STATUS: FAILURE"
        result = self.spawner.record_result(agent_id, output, success=True)

        self.assertEqual(result.status, AgentStatus.FAILED)

    def test_record_result_detects_boundary_violations(self):
        """Test that boundary violations are detected."""
        agent_id = self.spawner.spawn("Task", self.boundary)

        # Agent modified file outside its boundary
        output = """
        FILES_MODIFIED: src/auth.py, other/file.py
        STATUS: SUCCESS
        """

        result = self.spawner.record_result(agent_id, output)

        self.assertIsNotNone(result.error)
        self.assertIn("Boundary violations", result.error)
        self.assertIn("other/file.py", result.error)

    def test_record_result_unknown_agent_raises(self):
        """Test that recording for unknown agent raises."""
        with self.assertRaises(ValueError):
            self.spawner.record_result("nonexistent", "output")

    def test_wait_for_without_result_raises(self):
        """Test that wait_for raises if result not recorded."""
        agent_id = self.spawner.spawn("Task", self.boundary)

        with self.assertRaises(ValueError):
            self.spawner.wait_for(agent_id)

    def test_wait_for_with_result(self):
        """Test wait_for returns recorded result."""
        agent_id = self.spawner.spawn("Task", self.boundary)
        self.spawner.record_result(agent_id, "Done\nSTATUS: SUCCESS")

        result = self.spawner.wait_for(agent_id)
        self.assertEqual(result.status, AgentStatus.COMPLETED)

    def test_prepare_agents(self):
        """Test preparing multiple agents at once."""
        boundary1 = ParallelWorkBoundary("a1", "Auth", {"auth.py"})
        boundary2 = ParallelWorkBoundary("a2", "API", {"api.py"})

        configs = self.spawner.prepare_agents([
            ("Implement auth", boundary1),
            ("Implement API", boundary2),
        ])

        self.assertEqual(len(configs), 2)
        self.assertIn("auth.py", configs[0].prompt)
        self.assertIn("api.py", configs[1].prompt)

    def test_prepare_agents_with_requirements(self):
        """Test prepare_agents with custom requirements."""
        boundary = ParallelWorkBoundary("a1", "Test", {"test.py"})

        configs = self.spawner.prepare_agents(
            [("Task", boundary)],
            requirements="Use pytest for testing"
        )

        self.assertIn("Use pytest for testing", configs[0].prompt)

    def test_get_all_configs(self):
        """Test getting all pending configs."""
        self.spawner.spawn("Task 1", self.boundary)
        self.spawner.spawn("Task 2", self.boundary)

        configs = self.spawner.get_all_configs()
        self.assertEqual(len(configs), 2)

    def test_mark_running(self):
        """Test marking agent as running."""
        agent_id = self.spawner.spawn("Task", self.boundary)
        self.spawner.mark_running(agent_id)

        # Status still shows from internal state
        self.assertEqual(
            self.spawner._agents[agent_id]["status"],
            AgentStatus.RUNNING
        )

    def test_mark_timed_out(self):
        """Test marking agent as timed out."""
        agent_id = self.spawner.spawn("Task", self.boundary)
        self.spawner.mark_timed_out(agent_id)

        result = self.spawner.get_result(agent_id)
        self.assertEqual(result.status, AgentStatus.TIMED_OUT)

    def test_get_pending_agents(self):
        """Test getting list of pending agents."""
        id1 = self.spawner.spawn("Task 1", self.boundary)
        id2 = self.spawner.spawn("Task 2", self.boundary)

        pending = self.spawner.get_pending_agents()
        self.assertIn(id1, pending)
        self.assertIn(id2, pending)

        # Record one result
        self.spawner.record_result(id1, "Done\nSTATUS: SUCCESS")

        pending = self.spawner.get_pending_agents()
        self.assertNotIn(id1, pending)
        self.assertIn(id2, pending)

    def test_get_summary(self):
        """Test getting spawner summary."""
        id1 = self.spawner.spawn("Task 1", self.boundary)
        id2 = self.spawner.spawn("Task 2", self.boundary)
        id3 = self.spawner.spawn("Task 3", self.boundary)

        self.spawner.record_result(id1, "STATUS: SUCCESS")
        self.spawner.record_result(id2, "STATUS: FAILURE", success=False)

        summary = self.spawner.get_summary()

        self.assertEqual(summary["total_agents"], 3)
        self.assertEqual(len(summary["completed"]), 1)
        self.assertEqual(len(summary["failed"]), 1)
        self.assertEqual(len(summary["pending"]), 1)


class TestParseFileList(unittest.TestCase):
    """Tests for file list parsing."""

    def setUp(self):
        self.spawner = ClaudeCodeSpawner()
        self.boundary = ParallelWorkBoundary("a", "Test", {"test.py"})

    def test_parse_single_file(self):
        """Test parsing single file."""
        output = "FILES_MODIFIED: src/main.py"
        agent_id = self.spawner.spawn("Task", self.boundary)
        result = self.spawner.record_result(agent_id, output)

        self.assertEqual(result.files_modified, ["src/main.py"])

    def test_parse_multiple_files(self):
        """Test parsing multiple files."""
        output = "FILES_MODIFIED: a.py, b.py, c.py"
        agent_id = self.spawner.spawn("Task", self.boundary)
        result = self.spawner.record_result(agent_id, output)

        self.assertEqual(result.files_modified, ["a.py", "b.py", "c.py"])

    def test_parse_none_value(self):
        """Test parsing 'none' value."""
        output = "FILES_MODIFIED: none"
        agent_id = self.spawner.spawn("Task", self.boundary)
        result = self.spawner.record_result(agent_id, output)

        self.assertEqual(result.files_modified, [])

    def test_parse_missing_marker(self):
        """Test parsing when marker is missing."""
        output = "Task completed successfully"
        agent_id = self.spawner.spawn("Task", self.boundary)
        result = self.spawner.record_result(agent_id, output)

        self.assertEqual(result.files_modified, [])

    def test_parse_all_file_types(self):
        """Test parsing all file types."""
        output = """
        FILES_MODIFIED: mod.py
        FILES_CREATED: new.py
        FILES_DELETED: old.py
        """
        agent_id = self.spawner.spawn("Task", self.boundary)
        result = self.spawner.record_result(agent_id, output)

        self.assertEqual(result.files_modified, ["mod.py"])
        self.assertEqual(result.files_created, ["new.py"])
        self.assertEqual(result.files_deleted, ["old.py"])


class TestGenerateParallelTaskCalls(unittest.TestCase):
    """Tests for generate_parallel_task_calls helper."""

    def test_generates_markdown(self):
        """Test that helper generates markdown output."""
        spawner = ClaudeCodeSpawner()

        md = generate_parallel_task_calls(spawner, [
            ("Task 1", ParallelWorkBoundary("a1", "Test1", {"a.py"})),
            ("Task 2", ParallelWorkBoundary("a2", "Test2", {"b.py"})),
        ])

        self.assertIn("## Parallel Task Tool Configurations", md)
        self.assertIn("Agent 1:", md)
        self.assertIn("Agent 2:", md)

    def test_includes_agent_ids(self):
        """Test that output includes agent IDs."""
        spawner = ClaudeCodeSpawner()

        md = generate_parallel_task_calls(spawner, [
            ("Task", ParallelWorkBoundary("a", "Test", {"a.py"})),
        ])

        self.assertIn("agent-", md)


class TestIntegrationWithParallelCoordinator(unittest.TestCase):
    """Integration tests with ParallelCoordinator."""

    def test_full_workflow(self):
        """Test complete spawn -> record -> collect workflow."""
        from cortical.reasoning.collaboration import ParallelCoordinator

        spawner = ClaudeCodeSpawner(branch="feature-branch")
        coordinator = ParallelCoordinator(spawner)

        # Define boundaries
        boundaries = [
            ParallelWorkBoundary("a1", "Auth module", {"auth.py"}),
            ParallelWorkBoundary("a2", "API module", {"api.py"}),
        ]

        # Spawn agents
        agent_ids = coordinator.spawn_agents(
            ["Implement auth", "Implement API"],
            boundaries
        )

        self.assertEqual(len(agent_ids), 2)

        # Simulate Task tool execution and record results
        for agent_id in agent_ids:
            output = f"""
            Completed task for {agent_id}
            FILES_MODIFIED: {spawner._configs[agent_id].boundary.files_owned.pop()}
            STATUS: SUCCESS
            """
            spawner.record_result(agent_id, output)

        # Collect results via coordinator
        results = coordinator.collect_results(agent_ids)

        self.assertEqual(len(results), 2)
        for result in results.values():
            self.assertEqual(result.status, AgentStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
