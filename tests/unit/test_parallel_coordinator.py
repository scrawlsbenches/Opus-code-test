"""
Unit tests for ParallelCoordinator and related classes.

Tests the boundary-based isolation approach for parallel agent coordination,
including the AgentSpawner interface and SequentialSpawner implementation.
"""

import unittest
from datetime import datetime, timedelta

from cortical.reasoning.collaboration import (
    AgentResult,
    AgentSpawner,
    AgentStatus,
    ConflictDetail,
    ConflictType,
    ParallelCoordinator,
    ParallelWorkBoundary,
    SequentialSpawner,
)


class TestAgentStatus(unittest.TestCase):
    """Tests for AgentStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses exist."""
        statuses = [AgentStatus.PENDING, AgentStatus.RUNNING,
                    AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TIMED_OUT]
        self.assertEqual(len(statuses), 5)

    def test_statuses_are_distinct(self):
        """Each status is unique."""
        statuses = list(AgentStatus)
        self.assertEqual(len(statuses), len(set(statuses)))


class TestAgentResult(unittest.TestCase):
    """Tests for AgentResult dataclass."""

    def test_basic_creation(self):
        """Create result with minimal fields."""
        result = AgentResult(
            agent_id="test-001",
            status=AgentStatus.COMPLETED,
            task_description="Test task"
        )
        self.assertEqual(result.agent_id, "test-001")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.success())

    def test_success_returns_true_for_completed(self):
        """success() returns True only for COMPLETED status."""
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.COMPLETED,
            task_description="Test"
        )
        self.assertTrue(result.success())

    def test_success_returns_false_for_failed(self):
        """success() returns False for FAILED status."""
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.FAILED,
            task_description="Test",
            error="Something went wrong"
        )
        self.assertFalse(result.success())

    def test_success_returns_false_for_timed_out(self):
        """success() returns False for TIMED_OUT status."""
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.TIMED_OUT,
            task_description="Test"
        )
        self.assertFalse(result.success())

    def test_all_modified_files_combines_all_file_types(self):
        """all_modified_files() returns union of modified, created, deleted."""
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.COMPLETED,
            task_description="Test",
            files_modified=["a.py", "b.py"],
            files_created=["c.py"],
            files_deleted=["d.py"]
        )
        all_files = result.all_modified_files()
        self.assertEqual(all_files, {"a.py", "b.py", "c.py", "d.py"})

    def test_all_modified_files_empty_when_no_changes(self):
        """all_modified_files() returns empty set when no files changed."""
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.COMPLETED,
            task_description="Test"
        )
        self.assertEqual(result.all_modified_files(), set())

    def test_all_modified_files_handles_duplicates(self):
        """all_modified_files() handles same file in multiple lists."""
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.COMPLETED,
            task_description="Test",
            files_modified=["a.py"],
            files_deleted=["a.py"]  # Same file modified then deleted
        )
        all_files = result.all_modified_files()
        self.assertEqual(all_files, {"a.py"})

    def test_timestamps_and_duration(self):
        """Test timestamp and duration fields."""
        start = datetime.now()
        end = start + timedelta(seconds=5)
        result = AgentResult(
            agent_id="test",
            status=AgentStatus.COMPLETED,
            task_description="Test",
            started_at=start,
            completed_at=end,
            duration_seconds=5.0
        )
        self.assertEqual(result.started_at, start)
        self.assertEqual(result.completed_at, end)
        self.assertEqual(result.duration_seconds, 5.0)


class TestSequentialSpawner(unittest.TestCase):
    """Tests for SequentialSpawner implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.boundary = ParallelWorkBoundary(
            agent_id="test-agent",
            scope_description="Test scope",
            files_owned={"a.py", "b.py"}
        )

    def test_default_handler_returns_completed(self):
        """Default handler returns successful result."""
        spawner = SequentialSpawner()
        agent_id = spawner.spawn("Test task", self.boundary)

        result = spawner.get_result(agent_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIn("Test task", result.output)

    def test_spawn_returns_unique_agent_ids(self):
        """Each spawn returns a unique agent ID."""
        spawner = SequentialSpawner()
        ids = []
        for i in range(5):
            agent_id = spawner.spawn(f"Task {i}", self.boundary)
            ids.append(agent_id)

        self.assertEqual(len(ids), len(set(ids)))

    def test_agent_id_format(self):
        """Agent IDs follow expected format."""
        spawner = SequentialSpawner()
        agent_id = spawner.spawn("Test", self.boundary)
        self.assertTrue(agent_id.startswith("seq-agent-"))

    def test_custom_handler_is_used(self):
        """Custom handler function is called."""
        handler_called = []

        def custom_handler(task, boundary):
            handler_called.append(task)
            return AgentResult(
                agent_id="",  # Will be overwritten
                status=AgentStatus.COMPLETED,
                task_description=task,
                output="Custom output",
                files_modified=["custom.py"]
            )

        spawner = SequentialSpawner(handler=custom_handler)
        agent_id = spawner.spawn("My task", self.boundary)

        self.assertEqual(handler_called, ["My task"])
        result = spawner.get_result(agent_id)
        self.assertEqual(result.output, "Custom output")
        self.assertEqual(result.files_modified, ["custom.py"])

    def test_handler_exception_creates_failed_result(self):
        """Exception in handler creates FAILED result."""
        def failing_handler(task, boundary):
            raise ValueError("Handler error")

        spawner = SequentialSpawner(handler=failing_handler)
        agent_id = spawner.spawn("Task", self.boundary)

        result = spawner.get_result(agent_id)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Handler error", result.error)

    def test_get_status_returns_correct_status(self):
        """get_status returns the agent's status."""
        spawner = SequentialSpawner()
        agent_id = spawner.spawn("Test", self.boundary)

        status = spawner.get_status(agent_id)
        self.assertEqual(status, AgentStatus.COMPLETED)

    def test_get_status_unknown_agent(self):
        """get_status returns PENDING for unknown agent."""
        spawner = SequentialSpawner()
        status = spawner.get_status("unknown-agent")
        self.assertEqual(status, AgentStatus.PENDING)

    def test_get_result_unknown_agent(self):
        """get_result returns None for unknown agent."""
        spawner = SequentialSpawner()
        result = spawner.get_result("unknown-agent")
        self.assertIsNone(result)

    def test_wait_for_returns_result(self):
        """wait_for returns the agent result."""
        spawner = SequentialSpawner()
        agent_id = spawner.spawn("Test", self.boundary)

        result = spawner.wait_for(agent_id)
        self.assertEqual(result.agent_id, agent_id)
        self.assertEqual(result.status, AgentStatus.COMPLETED)

    def test_wait_for_unknown_agent(self):
        """wait_for returns failed result for unknown agent."""
        spawner = SequentialSpawner()
        result = spawner.wait_for("unknown-agent")

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("not found", result.error)

    def test_boundary_violation_detection(self):
        """Detect when handler modifies files outside boundary."""
        def violating_handler(task, boundary):
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=["outside.py"]  # Not in boundary
            )

        spawner = SequentialSpawner(handler=violating_handler)
        agent_id = spawner.spawn("Task", self.boundary)

        result = spawner.get_result(agent_id)
        self.assertIsNotNone(result.error)
        self.assertIn("Boundary violation", result.error)

    def test_no_violation_for_owned_files(self):
        """No violation when modifying owned files."""
        def good_handler(task, boundary):
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=["a.py"]  # In boundary
            )

        spawner = SequentialSpawner(handler=good_handler)
        agent_id = spawner.spawn("Task", self.boundary)

        result = spawner.get_result(agent_id)
        self.assertIsNone(result.error)

    def test_timestamps_are_set(self):
        """Spawn sets timestamps on result."""
        spawner = SequentialSpawner()
        before = datetime.now()
        agent_id = spawner.spawn("Test", self.boundary)
        after = datetime.now()

        result = spawner.get_result(agent_id)
        self.assertIsNotNone(result.started_at)
        self.assertIsNotNone(result.completed_at)
        self.assertGreaterEqual(result.started_at, before)
        self.assertLessEqual(result.completed_at, after)

    def test_duration_is_calculated(self):
        """Duration is calculated from timestamps."""
        spawner = SequentialSpawner()
        agent_id = spawner.spawn("Test", self.boundary)

        result = spawner.get_result(agent_id)
        self.assertIsNotNone(result.duration_seconds)
        self.assertGreaterEqual(result.duration_seconds, 0.0)


class TestConflictDetail(unittest.TestCase):
    """Tests for ConflictDetail dataclass."""

    def test_basic_creation(self):
        """Create conflict with required fields."""
        conflict = ConflictDetail(
            conflict_type=ConflictType.FILE_CONFLICT,
            agents_involved=["agent1", "agent2"],
            files_affected=["file.py"],
            description="Both modified file.py"
        )
        self.assertEqual(conflict.conflict_type, ConflictType.FILE_CONFLICT)
        self.assertEqual(len(conflict.agents_involved), 2)
        self.assertEqual(conflict.files_affected, ["file.py"])

    def test_optional_resolution_suggestion(self):
        """Resolution suggestion is optional."""
        conflict = ConflictDetail(
            conflict_type=ConflictType.SCOPE_OVERLAP,
            agents_involved=["agent1"],
            files_affected=["a.py", "b.py"],
            description="Scope issue",
            resolution_suggestion="Review boundaries"
        )
        self.assertEqual(conflict.resolution_suggestion, "Review boundaries")


class TestParallelCoordinator(unittest.TestCase):
    """Tests for ParallelCoordinator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.spawner = SequentialSpawner()
        self.coordinator = ParallelCoordinator(self.spawner)

    def create_boundary(self, agent_id, files_owned=None, files_read=None):
        """Helper to create boundaries."""
        return ParallelWorkBoundary(
            agent_id=agent_id,
            scope_description=f"Scope for {agent_id}",
            files_owned=set(files_owned or []),
            files_read_only=set(files_read or [])
        )

    def test_can_spawn_no_conflicts(self):
        """can_spawn returns True when no conflicts."""
        b1 = self.create_boundary("agent1", ["a.py", "b.py"])
        b2 = self.create_boundary("agent2", ["c.py", "d.py"])

        can_spawn, issues = self.coordinator.can_spawn([b1, b2])
        self.assertTrue(can_spawn)
        self.assertEqual(issues, [])

    def test_can_spawn_detects_file_ownership_conflict(self):
        """can_spawn detects when two boundaries own same file."""
        b1 = self.create_boundary("agent1", ["a.py", "shared.py"])
        b2 = self.create_boundary("agent2", ["shared.py", "c.py"])

        can_spawn, issues = self.coordinator.can_spawn([b1, b2])
        self.assertFalse(can_spawn)
        self.assertEqual(len(issues), 1)
        self.assertIn("shared.py", issues[0])

    def test_can_spawn_detects_read_write_conflict(self):
        """can_spawn detects when one reads what another writes."""
        b1 = self.create_boundary("agent1", ["a.py"], files_read=["shared.py"])
        b2 = self.create_boundary("agent2", ["shared.py"])  # Writes what b1 reads

        can_spawn, issues = self.coordinator.can_spawn([b1, b2])
        self.assertFalse(can_spawn)
        self.assertIn("race", issues[0].lower())

    def test_can_spawn_multiple_conflicts(self):
        """can_spawn reports all conflicts."""
        b1 = self.create_boundary("agent1", ["a.py", "shared1.py", "shared2.py"])
        b2 = self.create_boundary("agent2", ["shared1.py", "b.py"])
        b3 = self.create_boundary("agent3", ["shared2.py", "c.py"])

        can_spawn, issues = self.coordinator.can_spawn([b1, b2, b3])
        self.assertFalse(can_spawn)
        self.assertGreaterEqual(len(issues), 2)

    def test_spawn_agents_returns_agent_ids(self):
        """spawn_agents returns list of agent IDs."""
        b1 = self.create_boundary("agent1", ["a.py"])
        b2 = self.create_boundary("agent2", ["b.py"])

        agent_ids = self.coordinator.spawn_agents(
            ["Task 1", "Task 2"],
            [b1, b2]
        )

        self.assertEqual(len(agent_ids), 2)
        self.assertNotEqual(agent_ids[0], agent_ids[1])

    def test_spawn_agents_validates_list_lengths(self):
        """spawn_agents raises ValueError for mismatched lengths."""
        b1 = self.create_boundary("agent1", ["a.py"])

        with self.assertRaises(ValueError) as ctx:
            self.coordinator.spawn_agents(
                ["Task 1", "Task 2"],  # 2 tasks
                [b1]  # 1 boundary
            )

        self.assertIn("same number", str(ctx.exception))

    def test_collect_results_returns_all_results(self):
        """collect_results waits for and returns all agent results."""
        b1 = self.create_boundary("agent1", ["a.py"])
        b2 = self.create_boundary("agent2", ["b.py"])

        agent_ids = self.coordinator.spawn_agents(
            ["Task 1", "Task 2"],
            [b1, b2]
        )

        results = self.coordinator.collect_results(agent_ids)

        self.assertEqual(len(results), 2)
        for agent_id in agent_ids:
            self.assertIn(agent_id, results)
            self.assertEqual(results[agent_id].status, AgentStatus.COMPLETED)

    def test_detect_conflicts_no_overlapping_files(self):
        """detect_conflicts returns empty when no overlapping modifications."""
        results = {
            "agent1": AgentResult(
                agent_id="agent1",
                status=AgentStatus.COMPLETED,
                task_description="Task 1",
                files_modified=["a.py"]
            ),
            "agent2": AgentResult(
                agent_id="agent2",
                status=AgentStatus.COMPLETED,
                task_description="Task 2",
                files_modified=["b.py"]
            ),
        }

        conflicts = self.coordinator.detect_conflicts(results)
        self.assertEqual(conflicts, [])

    def test_detect_conflicts_finds_overlapping_modifications(self):
        """detect_conflicts finds when two agents modified same file."""
        results = {
            "agent1": AgentResult(
                agent_id="agent1",
                status=AgentStatus.COMPLETED,
                task_description="Task 1",
                files_modified=["shared.py", "a.py"]
            ),
            "agent2": AgentResult(
                agent_id="agent2",
                status=AgentStatus.COMPLETED,
                task_description="Task 2",
                files_modified=["shared.py", "b.py"]
            ),
        }

        conflicts = self.coordinator.detect_conflicts(results)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.FILE_CONFLICT)
        self.assertIn("shared.py", conflicts[0].files_affected)
        self.assertIn("agent1", conflicts[0].agents_involved)
        self.assertIn("agent2", conflicts[0].agents_involved)

    def test_detect_conflicts_finds_boundary_violations(self):
        """detect_conflicts finds boundary violation errors."""
        results = {
            "agent1": AgentResult(
                agent_id="agent1",
                status=AgentStatus.COMPLETED,
                task_description="Task 1",
                files_modified=["a.py"],
                error="Boundary violation: modified outside.py outside owned files"
            ),
        }

        conflicts = self.coordinator.detect_conflicts(results)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.SCOPE_OVERLAP)

    def test_detect_conflicts_multiple_agents_same_file(self):
        """detect_conflicts handles three agents modifying same file."""
        results = {
            "agent1": AgentResult(
                agent_id="agent1",
                status=AgentStatus.COMPLETED,
                task_description="Task 1",
                files_modified=["shared.py"]
            ),
            "agent2": AgentResult(
                agent_id="agent2",
                status=AgentStatus.COMPLETED,
                task_description="Task 2",
                files_modified=["shared.py"]
            ),
            "agent3": AgentResult(
                agent_id="agent3",
                status=AgentStatus.COMPLETED,
                task_description="Task 3",
                files_modified=["shared.py"]
            ),
        }

        conflicts = self.coordinator.detect_conflicts(results)

        # Should find 3 pairwise conflicts: (1,2), (1,3), (2,3)
        self.assertEqual(len(conflicts), 3)

    def test_get_summary(self):
        """get_summary returns coordination state."""
        b1 = self.create_boundary("agent1", ["a.py"])

        agent_ids = self.coordinator.spawn_agents(["Task 1"], [b1])
        self.coordinator.collect_results(agent_ids)

        summary = self.coordinator.get_summary()

        self.assertEqual(summary['active_agents'], 0)
        self.assertEqual(summary['completed_agents'], 1)
        self.assertEqual(summary['failed_agents'], 0)
        self.assertEqual(summary['conflicts_detected'], 0)

    def test_get_summary_with_failures(self):
        """get_summary counts failures."""
        def failing_handler(task, boundary):
            raise ValueError("Intentional failure")

        spawner = SequentialSpawner(handler=failing_handler)
        coordinator = ParallelCoordinator(spawner)

        b1 = self.create_boundary("agent1", ["a.py"])
        agent_ids = coordinator.spawn_agents(["Task 1"], [b1])
        coordinator.collect_results(agent_ids)

        summary = coordinator.get_summary()
        self.assertEqual(summary['failed_agents'], 1)
        self.assertEqual(summary['completed_agents'], 0)

    def test_get_active_agent_ids(self):
        """get_active_agent_ids tracks active agents."""
        b1 = self.create_boundary("agent1", ["a.py"])

        # Before spawn, no active agents
        self.assertEqual(self.coordinator.get_active_agent_ids(), [])

        # After spawn, agent is tracked as active
        agent_ids = self.coordinator.spawn_agents(["Task 1"], [b1])
        self.assertEqual(len(self.coordinator.get_active_agent_ids()), 1)
        self.assertIn(agent_ids[0], self.coordinator.get_active_agent_ids())

        # After collecting results, agent is no longer active
        self.coordinator.collect_results(agent_ids)
        self.assertEqual(self.coordinator.get_active_agent_ids(), [])

    def test_get_completed_results(self):
        """get_completed_results returns all completed results."""
        b1 = self.create_boundary("agent1", ["a.py"])
        b2 = self.create_boundary("agent2", ["b.py"])

        agent_ids = self.coordinator.spawn_agents(
            ["Task 1", "Task 2"],
            [b1, b2]
        )
        self.coordinator.collect_results(agent_ids)

        completed = self.coordinator.get_completed_results()
        self.assertEqual(len(completed), 2)

    def test_get_conflicts_accumulates(self):
        """get_conflicts returns all detected conflicts."""
        results1 = {
            "a1": AgentResult(
                agent_id="a1",
                status=AgentStatus.COMPLETED,
                task_description="T1",
                files_modified=["x.py"]
            ),
            "a2": AgentResult(
                agent_id="a2",
                status=AgentStatus.COMPLETED,
                task_description="T2",
                files_modified=["x.py"]
            ),
        }

        results2 = {
            "a3": AgentResult(
                agent_id="a3",
                status=AgentStatus.COMPLETED,
                task_description="T3",
                files_modified=["y.py"]
            ),
            "a4": AgentResult(
                agent_id="a4",
                status=AgentStatus.COMPLETED,
                task_description="T4",
                files_modified=["y.py"]
            ),
        }

        self.coordinator.detect_conflicts(results1)
        self.coordinator.detect_conflicts(results2)

        all_conflicts = self.coordinator.get_conflicts()
        self.assertEqual(len(all_conflicts), 2)

    def test_reset_clears_state(self):
        """reset clears all coordinator state."""
        b1 = self.create_boundary("agent1", ["a.py"])

        agent_ids = self.coordinator.spawn_agents(["Task 1"], [b1])
        self.coordinator.collect_results(agent_ids)

        # Verify state exists
        self.assertGreater(len(self.coordinator.get_completed_results()), 0)

        self.coordinator.reset()

        # Verify state is cleared
        self.assertEqual(len(self.coordinator.get_completed_results()), 0)
        self.assertEqual(len(self.coordinator.get_conflicts()), 0)
        self.assertEqual(len(self.coordinator.get_active_agent_ids()), 0)

    def test_full_workflow(self):
        """Test complete coordinator workflow."""
        # 1. Create boundaries
        b1 = self.create_boundary("frontend", ["ui.py", "components.py"])
        b2 = self.create_boundary("backend", ["api.py", "db.py"])
        b3 = self.create_boundary("tests", ["test_ui.py", "test_api.py"])

        # 2. Check if can spawn
        can_spawn, issues = self.coordinator.can_spawn([b1, b2, b3])
        self.assertTrue(can_spawn)

        # 3. Spawn agents
        tasks = [
            "Implement UI components",
            "Build API endpoints",
            "Write comprehensive tests"
        ]
        agent_ids = self.coordinator.spawn_agents(tasks, [b1, b2, b3])
        self.assertEqual(len(agent_ids), 3)

        # 4. Collect results
        results = self.coordinator.collect_results(agent_ids)
        self.assertEqual(len(results), 3)

        # 5. Check for conflicts
        conflicts = self.coordinator.detect_conflicts(results)
        self.assertEqual(conflicts, [])

        # 6. Get summary
        summary = self.coordinator.get_summary()
        self.assertEqual(summary['completed_agents'], 3)
        self.assertEqual(summary['failed_agents'], 0)

    def test_workflow_with_conflict(self):
        """Test workflow where agents produce conflicting results."""
        # Custom handler that simulates overlapping file modifications
        modifications = {
            0: ["shared.py", "a.py"],
            1: ["shared.py", "b.py"],
        }
        call_count = [0]

        def overlapping_handler(task, boundary):
            idx = call_count[0]
            call_count[0] += 1
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=modifications.get(idx, [])
            )

        spawner = SequentialSpawner(handler=overlapping_handler)
        coordinator = ParallelCoordinator(spawner)

        b1 = self.create_boundary("agent1", ["shared.py", "a.py"])
        b2 = self.create_boundary("agent2", ["shared.py", "b.py"])

        # Note: can_spawn would detect this, but we proceed anyway for testing
        agent_ids = coordinator.spawn_agents(["Task 1", "Task 2"], [b1, b2])
        results = coordinator.collect_results(agent_ids)
        conflicts = coordinator.detect_conflicts(results)

        self.assertEqual(len(conflicts), 1)
        self.assertIn("shared.py", conflicts[0].files_affected)


class TestParallelCoordinatorEdgeCases(unittest.TestCase):
    """Edge case tests for ParallelCoordinator."""

    def setUp(self):
        """Set up test fixtures."""
        self.spawner = SequentialSpawner()
        self.coordinator = ParallelCoordinator(self.spawner)

    def create_boundary(self, agent_id, files_owned=None, files_read=None):
        """Helper to create boundaries."""
        return ParallelWorkBoundary(
            agent_id=agent_id,
            scope_description=f"Scope for {agent_id}",
            files_owned=set(files_owned or []),
            files_read_only=set(files_read or [])
        )

    def test_empty_boundaries(self):
        """Handle empty boundary list."""
        can_spawn, issues = self.coordinator.can_spawn([])
        self.assertTrue(can_spawn)
        self.assertEqual(issues, [])

    def test_single_boundary(self):
        """Single boundary always succeeds can_spawn."""
        b1 = self.create_boundary("agent1", ["a.py"])
        can_spawn, issues = self.coordinator.can_spawn([b1])
        self.assertTrue(can_spawn)

    def test_empty_tasks_and_boundaries(self):
        """spawn_agents handles empty lists."""
        agent_ids = self.coordinator.spawn_agents([], [])
        self.assertEqual(agent_ids, [])

    def test_detect_conflicts_empty_results(self):
        """detect_conflicts handles empty results."""
        conflicts = self.coordinator.detect_conflicts({})
        self.assertEqual(conflicts, [])

    def test_detect_conflicts_single_result(self):
        """detect_conflicts handles single result (no conflicts possible)."""
        results = {
            "agent1": AgentResult(
                agent_id="agent1",
                status=AgentStatus.COMPLETED,
                task_description="Task 1",
                files_modified=["a.py"]
            ),
        }
        conflicts = self.coordinator.detect_conflicts(results)
        self.assertEqual(conflicts, [])

    def test_many_files_conflict_truncation(self):
        """can_spawn truncates long conflict file lists."""
        many_files = {f"file{i}.py" for i in range(10)}
        b1 = self.create_boundary("agent1", many_files)
        b2 = self.create_boundary("agent2", many_files)

        can_spawn, issues = self.coordinator.can_spawn([b1, b2])
        self.assertFalse(can_spawn)
        # Should truncate to 3 files + "and X more"
        self.assertIn("more", issues[0])

    def test_files_modified_includes_created_and_deleted(self):
        """detect_conflicts considers all file change types."""
        results = {
            "agent1": AgentResult(
                agent_id="agent1",
                status=AgentStatus.COMPLETED,
                task_description="Task 1",
                files_created=["new.py"]  # Created
            ),
            "agent2": AgentResult(
                agent_id="agent2",
                status=AgentStatus.COMPLETED,
                task_description="Task 2",
                files_deleted=["new.py"]  # Deleted same file!
            ),
        }

        conflicts = self.coordinator.detect_conflicts(results)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("new.py", conflicts[0].files_affected)

    def test_summary_limits_files_list(self):
        """get_summary limits files_modified to first 20."""
        def many_files_handler(task, boundary):
            return AgentResult(
                agent_id="",
                status=AgentStatus.COMPLETED,
                task_description=task,
                files_modified=[f"file{i}.py" for i in range(50)]
            )

        spawner = SequentialSpawner(handler=many_files_handler)
        coordinator = ParallelCoordinator(spawner)

        b1 = self.create_boundary("agent1", [f"file{i}.py" for i in range(50)])
        agent_ids = coordinator.spawn_agents(["Task 1"], [b1])
        coordinator.collect_results(agent_ids)

        summary = coordinator.get_summary()
        self.assertLessEqual(len(summary['files_modified']), 20)
        self.assertEqual(summary['total_files_modified'], 50)


if __name__ == '__main__':
    unittest.main()
