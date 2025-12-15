"""Unit tests for director orchestration utilities."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from orchestration_utils import (
    generate_plan_id,
    generate_execution_id,
    Agent,
    Batch,
    OrchestrationPlan,
    AgentResult,
    BatchVerification,
    ReplanEvent,
    ExecutionTracker,
    MetricsEvent,
    OrchestrationMetrics,
    load_all_plans,
    get_plan_by_id,
    PLANS_DIR,
    EXECUTIONS_DIR,
)


class TestIDGeneration(unittest.TestCase):
    """Tests for plan and execution ID generation."""

    def test_plan_id_format(self):
        """Plan ID should match OP-YYYYMMDD-HHMMSS-XXXXXXXX format."""
        plan_id = generate_plan_id()
        self.assertTrue(plan_id.startswith("OP-"), "Plan ID should start with 'OP-'")
        parts = plan_id.split("-")
        self.assertEqual(len(parts), 4, "Plan ID should have 4 parts separated by hyphens")
        self.assertEqual(len(parts[1]), 8, "Date part should be 8 characters (YYYYMMDD)")
        self.assertEqual(len(parts[2]), 6, "Time part should be 6 characters (HHMMSS)")
        self.assertEqual(len(parts[3]), 8, "Suffix should be 8 characters")

    def test_execution_id_format(self):
        """Execution ID should match EX-YYYYMMDD-HHMMSS-XXXXXXXX format."""
        exec_id = generate_execution_id()
        self.assertTrue(exec_id.startswith("EX-"), "Execution ID should start with 'EX-'")
        parts = exec_id.split("-")
        self.assertEqual(len(parts), 4, "Execution ID should have 4 parts separated by hyphens")
        self.assertEqual(len(parts[1]), 8, "Date part should be 8 characters (YYYYMMDD)")
        self.assertEqual(len(parts[2]), 6, "Time part should be 6 characters (HHMMSS)")
        self.assertEqual(len(parts[3]), 8, "Suffix should be 8 characters")

    def test_ids_are_unique(self):
        """Generated IDs should be unique."""
        plan_ids = {generate_plan_id() for _ in range(50)}
        exec_ids = {generate_execution_id() for _ in range(50)}

        self.assertEqual(len(plan_ids), 50, "All generated plan IDs should be unique")
        self.assertEqual(len(exec_ids), 50, "All generated execution IDs should be unique")


class TestAgent(unittest.TestCase):
    """Tests for Agent dataclass."""

    def test_create_agent(self):
        """Test basic agent creation."""
        agent = Agent(
            agent_id="A1",
            task_type="research",
            description="Research implementation patterns",
            scope={"files_read": ["src/**/*.py"]}
        )

        self.assertEqual(agent.agent_id, "A1")
        self.assertEqual(agent.task_type, "research")
        self.assertEqual(agent.description, "Research implementation patterns")
        self.assertEqual(agent.status, "pending", "Default status should be 'pending'")
        self.assertIsNone(agent.result, "Default result should be None")
        self.assertIn("files_read", agent.scope)

    def test_agent_status_transitions(self):
        """Test mark_in_progress, mark_completed, mark_failed."""
        agent = Agent(
            agent_id="A1",
            task_type="implement",
            description="Implement feature",
            scope={}
        )

        # Test mark_in_progress
        agent.mark_in_progress()
        self.assertEqual(agent.status, "in_progress")

        # Test mark_completed
        result = {"files_modified": ["src/test.py"]}
        agent.mark_completed(result)
        self.assertEqual(agent.status, "completed")
        self.assertEqual(agent.result, result)

        # Test mark_failed
        agent2 = Agent(agent_id="A2", task_type="test", description="Run tests", scope={})
        agent2.mark_failed("Tests failed")
        self.assertEqual(agent2.status, "failed")
        self.assertIsNotNone(agent2.result)
        self.assertIn("error", agent2.result)

    def test_agent_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        agent = Agent(
            agent_id="A1",
            task_type="verify",
            description="Verify implementation",
            scope={"files_read": ["test.py"]},
            status="completed",
            result={"success": True}
        )

        # Serialize to dict
        d = agent.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["agent_id"], "A1")
        self.assertEqual(d["task_type"], "verify")
        self.assertEqual(d["status"], "completed")

        # Deserialize from dict
        agent2 = Agent.from_dict(d)
        self.assertEqual(agent2.agent_id, agent.agent_id)
        self.assertEqual(agent2.task_type, agent.task_type)
        self.assertEqual(agent2.description, agent.description)
        self.assertEqual(agent2.status, agent.status)
        self.assertEqual(agent2.result, agent.result)


class TestBatch(unittest.TestCase):
    """Tests for Batch dataclass."""

    def test_create_batch(self):
        """Test batch creation with agents."""
        agents = [
            Agent(agent_id="A1", task_type="research", description="Research", scope={}),
            Agent(agent_id="A2", task_type="implement", description="Implement", scope={})
        ]

        batch = Batch(
            batch_id="B1",
            name="Research phase",
            batch_type="parallel",
            agents=agents
        )

        self.assertEqual(batch.batch_id, "B1")
        self.assertEqual(batch.name, "Research phase")
        self.assertEqual(batch.batch_type, "parallel")
        self.assertEqual(len(batch.agents), 2)
        self.assertEqual(batch.status, "pending")
        self.assertEqual(batch.depends_on, [])

    def test_get_agent(self):
        """Test finding agent by ID."""
        agents = [
            Agent(agent_id="A1", task_type="research", description="Research", scope={}),
            Agent(agent_id="A2", task_type="implement", description="Implement", scope={})
        ]
        batch = Batch(batch_id="B1", name="Test", batch_type="parallel", agents=agents)

        found = batch.get_agent("A1")
        self.assertIsNotNone(found)
        self.assertEqual(found.agent_id, "A1")

        not_found = batch.get_agent("A999")
        self.assertIsNone(not_found)

    def test_all_agents_completed(self):
        """Test completion detection."""
        agents = [
            Agent(agent_id="A1", task_type="test", description="Test", scope={}),
            Agent(agent_id="A2", task_type="test", description="Test", scope={})
        ]
        batch = Batch(batch_id="B1", name="Test", batch_type="parallel", agents=agents)

        # Initially not all completed
        self.assertFalse(batch.all_agents_completed())

        # Mark one as completed
        agents[0].mark_completed()
        self.assertFalse(batch.all_agents_completed())

        # Mark all as completed
        agents[1].mark_completed()
        self.assertTrue(batch.all_agents_completed())

    def test_any_agent_failed(self):
        """Test failure detection."""
        agents = [
            Agent(agent_id="A1", task_type="test", description="Test", scope={}),
            Agent(agent_id="A2", task_type="test", description="Test", scope={})
        ]
        batch = Batch(batch_id="B1", name="Test", batch_type="parallel", agents=agents)

        # Initially no failures
        self.assertFalse(batch.any_agent_failed())

        # Mark one as failed
        agents[0].mark_failed("Error")
        self.assertTrue(batch.any_agent_failed())

    def test_batch_status_transitions(self):
        """Test batch status methods."""
        batch = Batch(
            batch_id="B1",
            name="Test",
            batch_type="sequential",
            agents=[]
        )

        self.assertEqual(batch.status, "pending")

        batch.mark_in_progress()
        self.assertEqual(batch.status, "in_progress")

        batch.mark_completed()
        self.assertEqual(batch.status, "completed")

        batch2 = Batch(batch_id="B2", name="Test2", batch_type="parallel", agents=[])
        batch2.mark_failed()
        self.assertEqual(batch2.status, "failed")

    def test_batch_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        agents = [
            Agent(agent_id="A1", task_type="test", description="Test", scope={})
        ]
        batch = Batch(
            batch_id="B1",
            name="Test batch",
            batch_type="parallel",
            agents=agents,
            depends_on=["B0"],
            status="in_progress"
        )

        # Serialize to dict
        d = batch.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["batch_id"], "B1")
        self.assertEqual(len(d["agents"]), 1)
        self.assertEqual(d["depends_on"], ["B0"])

        # Deserialize from dict
        batch2 = Batch.from_dict(d)
        self.assertEqual(batch2.batch_id, batch.batch_id)
        self.assertEqual(batch2.name, batch.name)
        self.assertEqual(batch2.batch_type, batch.batch_type)
        self.assertEqual(len(batch2.agents), len(batch.agents))
        self.assertEqual(batch2.depends_on, batch.depends_on)


class TestOrchestrationPlan(unittest.TestCase):
    """Tests for OrchestrationPlan."""

    def test_create_plan(self):
        """Test plan creation."""
        plan = OrchestrationPlan.create(
            title="Implement feature X",
            goal={
                "summary": "Add new search feature",
                "success_criteria": ["Tests pass", "Coverage >90%"]
            }
        )

        self.assertTrue(plan.plan_id.startswith("OP-"))
        self.assertEqual(plan.title, "Implement feature X")
        self.assertIn("summary", plan.goal)
        self.assertEqual(len(plan.batches), 0)
        self.assertIsNotNone(plan.created_at)

    def test_create_plan_with_parent_task(self):
        """Test plan creation with parent task link."""
        plan = OrchestrationPlan.create(
            title="Test plan",
            goal={"summary": "Test"},
            parent_task_id="T-12345"
        )

        self.assertEqual(plan.task_links.get("parent_task_id"), "T-12345")
        self.assertEqual(plan.task_links.get("child_task_ids"), [])

    def test_add_batch(self):
        """Test adding batches (auto-incrementing IDs)."""
        plan = OrchestrationPlan.create(
            title="Test plan",
            goal={"summary": "Test"}
        )

        agents1 = [Agent(agent_id="A1", task_type="research", description="Research", scope={})]
        batch1 = plan.add_batch(
            name="Research phase",
            batch_type="parallel",
            agents=agents1
        )

        self.assertEqual(batch1.batch_id, "B1")
        self.assertEqual(len(plan.batches), 1)

        agents2 = [Agent(agent_id="A2", task_type="implement", description="Implement", scope={})]
        batch2 = plan.add_batch(
            name="Implementation phase",
            batch_type="sequential",
            agents=agents2,
            depends_on=["B1"]
        )

        self.assertEqual(batch2.batch_id, "B2")
        self.assertEqual(len(plan.batches), 2)
        self.assertEqual(batch2.depends_on, ["B1"])

    def test_get_batch(self):
        """Test finding batch by ID."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        agents = [Agent(agent_id="A1", task_type="test", description="Test", scope={})]
        plan.add_batch(name="Batch 1", batch_type="parallel", agents=agents)

        found = plan.get_batch("B1")
        self.assertIsNotNone(found)
        self.assertEqual(found.batch_id, "B1")

        not_found = plan.get_batch("B999")
        self.assertIsNone(not_found)

    def test_get_agent(self):
        """Test finding agent across all batches."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})

        agents1 = [Agent(agent_id="A1", task_type="test", description="Test", scope={})]
        plan.add_batch(name="Batch 1", batch_type="parallel", agents=agents1)

        agents2 = [Agent(agent_id="A2", task_type="test", description="Test", scope={})]
        plan.add_batch(name="Batch 2", batch_type="parallel", agents=agents2)

        found1 = plan.get_agent("A1")
        self.assertIsNotNone(found1)
        self.assertEqual(found1.agent_id, "A1")

        found2 = plan.get_agent("A2")
        self.assertIsNotNone(found2)
        self.assertEqual(found2.agent_id, "A2")

        not_found = plan.get_agent("A999")
        self.assertIsNone(not_found)

    def test_get_ready_batches(self):
        """Test dependency-aware batch readiness."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})

        # Add three batches with dependencies
        agents = [Agent(agent_id=f"A{i}", task_type="test", description="Test", scope={}) for i in range(3)]
        plan.add_batch(name="B1", batch_type="parallel", agents=[agents[0]])
        plan.add_batch(name="B2", batch_type="parallel", agents=[agents[1]], depends_on=["B1"])
        plan.add_batch(name="B3", batch_type="parallel", agents=[agents[2]], depends_on=["B1", "B2"])

        # Initially only B1 is ready (no dependencies)
        ready = plan.get_ready_batches()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].batch_id, "B1")

        # Mark B1 as completed
        plan.get_batch("B1").mark_completed()
        ready = plan.get_ready_batches()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].batch_id, "B2")

        # Mark B2 as completed
        plan.get_batch("B2").mark_completed()
        ready = plan.get_ready_batches()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].batch_id, "B3")

    def test_plan_serialization(self):
        """Test to_dict/from_dict."""
        plan = OrchestrationPlan.create(
            title="Test plan",
            goal={"summary": "Test goal"}
        )
        agents = [Agent(agent_id="A1", task_type="test", description="Test", scope={})]
        plan.add_batch(name="Batch 1", batch_type="parallel", agents=agents)

        # Serialize to dict
        d = plan.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["title"], "Test plan")
        self.assertEqual(len(d["batches"]), 1)

        # Deserialize from dict
        plan2 = OrchestrationPlan.from_dict(d)
        self.assertEqual(plan2.plan_id, plan.plan_id)
        self.assertEqual(plan2.title, plan.title)
        self.assertEqual(len(plan2.batches), len(plan.batches))

    def test_save_load_roundtrip(self):
        """Test atomic save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = OrchestrationPlan.create(
                title="Test plan",
                goal={"summary": "Test goal"}
            )
            agents = [Agent(agent_id="A1", task_type="test", description="Test", scope={})]
            plan.add_batch(name="Batch 1", batch_type="parallel", agents=agents)

            # Save to custom path
            path = Path(temp_dir) / "test_plan.json"
            saved_path = plan.save(path)
            self.assertTrue(saved_path.exists())

            # Load and verify
            loaded = OrchestrationPlan.load(saved_path)
            self.assertEqual(loaded.plan_id, plan.plan_id)
            self.assertEqual(loaded.title, plan.title)
            self.assertEqual(len(loaded.batches), 1)
            self.assertEqual(loaded.batches[0].batch_id, "B1")

    def test_save_creates_temp_file(self):
        """Test that save uses atomic write pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
            path = Path(temp_dir) / "test.json"

            plan.save(path)

            # Temp file should be cleaned up
            temp_path = path.with_suffix('.json.tmp')
            self.assertFalse(temp_path.exists())


class TestAgentResult(unittest.TestCase):
    """Tests for AgentResult dataclass."""

    def test_agent_result_creation(self):
        """Test creating agent result."""
        result = AgentResult(
            status="completed",
            started_at="2025-12-15T10:00:00",
            completed_at="2025-12-15T10:05:00",
            duration_ms=300000,
            output_summary="Task completed successfully",
            files_modified=["src/test.py"],
            errors=[]
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.duration_ms, 300000)
        self.assertEqual(len(result.files_modified), 1)
        self.assertEqual(len(result.errors), 0)

    def test_agent_result_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        result = AgentResult(
            status="failed",
            started_at="2025-12-15T10:00:00",
            completed_at="2025-12-15T10:01:00",
            duration_ms=60000,
            output_summary="Task failed",
            errors=["Connection timeout"]
        )

        d = result.to_dict()
        self.assertEqual(d["status"], "failed")
        self.assertEqual(len(d["errors"]), 1)

        result2 = AgentResult.from_dict(d)
        self.assertEqual(result2.status, result.status)
        self.assertEqual(result2.errors, result.errors)


class TestBatchVerification(unittest.TestCase):
    """Tests for BatchVerification dataclass."""

    def test_batch_verification_creation(self):
        """Test creating batch verification."""
        verification = BatchVerification(
            batch_id="B1",
            verified_at="2025-12-15T10:00:00",
            checks={"tests_pass": True, "no_conflicts": True, "git_clean": True},
            verdict="pass"
        )

        self.assertEqual(verification.batch_id, "B1")
        self.assertEqual(verification.verdict, "pass")
        self.assertTrue(verification.checks["tests_pass"])

    def test_batch_verification_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        verification = BatchVerification(
            batch_id="B1",
            verified_at="2025-12-15T10:00:00",
            checks={"tests_pass": False, "no_conflicts": True},
            verdict="fail",
            notes="Tests failed on agent A1"
        )

        d = verification.to_dict()
        self.assertEqual(d["verdict"], "fail")
        self.assertIn("notes", d)

        verification2 = BatchVerification.from_dict(d)
        self.assertEqual(verification2.verdict, verification.verdict)
        self.assertEqual(verification2.notes, verification.notes)


class TestReplanEvent(unittest.TestCase):
    """Tests for ReplanEvent dataclass."""

    def test_replan_event_creation(self):
        """Test creating replan event."""
        event = ReplanEvent(
            at="2025-12-15T10:00:00",
            trigger="verification_fail",
            reason="Tests failed, need to adjust scope",
            old_plan_summary="3 batches, 5 agents",
            new_plan_summary="2 batches, 4 agents"
        )

        self.assertEqual(event.trigger, "verification_fail")
        self.assertIn("Tests failed", event.reason)

    def test_replan_event_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        event = ReplanEvent(
            at="2025-12-15T10:00:00",
            trigger="agent_blocker",
            reason="Agent A1 blocked on dependency",
            old_plan_summary="Old plan",
            new_plan_summary="New plan"
        )

        d = event.to_dict()
        self.assertEqual(d["trigger"], "agent_blocker")

        event2 = ReplanEvent.from_dict(d)
        self.assertEqual(event2.trigger, event.trigger)
        self.assertEqual(event2.reason, event.reason)


class TestExecutionTracker(unittest.TestCase):
    """Tests for ExecutionTracker."""

    def test_create_from_plan(self):
        """Test tracker creation from plan."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        agents = [Agent(agent_id="A1", task_type="test", description="Test", scope={})]
        plan.add_batch(name="Batch 1", batch_type="parallel", agents=agents)
        plan.add_batch(name="Batch 2", batch_type="parallel", agents=agents)

        tracker = ExecutionTracker.create(plan)

        self.assertTrue(tracker.execution_id.startswith("EX-"))
        self.assertEqual(tracker.plan_id, plan.plan_id)
        self.assertEqual(tracker.status, "in_progress")
        self.assertIsNone(tracker.current_batch)
        self.assertEqual(len(tracker.batches_remaining), 2)
        self.assertEqual(len(tracker.batches_completed), 0)

    def test_start_batch(self):
        """Test batch start tracking."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        tracker = ExecutionTracker.create(plan)

        tracker.start_batch("B1")
        self.assertEqual(tracker.current_batch, "B1")

    def test_record_agent_result(self):
        """Test agent result recording."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        tracker = ExecutionTracker.create(plan)

        result = AgentResult(
            status="completed",
            started_at="2025-12-15T10:00:00",
            completed_at="2025-12-15T10:05:00",
            duration_ms=300000,
            output_summary="Done"
        )

        tracker.record_agent_result("A1", result)
        self.assertIn("A1", tracker.agent_results)
        self.assertEqual(tracker.agent_results["A1"].status, "completed")

    def test_complete_batch(self):
        """Test batch completion with verification."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        agents = [Agent(agent_id="A1", task_type="test", description="Test", scope={})]
        plan.add_batch(name="Batch 1", batch_type="parallel", agents=agents)

        tracker = ExecutionTracker.create(plan)
        tracker.start_batch("B1")

        verification = BatchVerification(
            batch_id="B1",
            verified_at="2025-12-15T10:00:00",
            checks={"tests_pass": True},
            verdict="pass"
        )

        tracker.complete_batch("B1", verification)

        self.assertIn("B1", tracker.batches_completed)
        self.assertNotIn("B1", tracker.batches_remaining)
        self.assertEqual(len(tracker.verifications), 1)
        self.assertIsNone(tracker.current_batch)

    def test_record_replan(self):
        """Test replanning event recording."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        tracker = ExecutionTracker.create(plan)

        tracker.record_replan(
            trigger="verification_fail",
            reason="Tests failed",
            old_summary="3 batches",
            new_summary="2 batches"
        )

        self.assertEqual(len(tracker.replanning_events), 1)
        self.assertEqual(tracker.replanning_events[0].trigger, "verification_fail")

    def test_get_batch_duration(self):
        """Test batch duration calculation."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        tracker = ExecutionTracker.create(plan)

        # Record two agent results
        result1 = AgentResult(
            status="completed",
            started_at="2025-12-15T10:00:00",
            completed_at="2025-12-15T10:05:00",
            duration_ms=300000,
            output_summary="Done"
        )
        result2 = AgentResult(
            status="completed",
            started_at="2025-12-15T10:01:00",
            completed_at="2025-12-15T10:06:00",
            duration_ms=300000,
            output_summary="Done"
        )

        tracker.record_agent_result("A1", result1)
        tracker.record_agent_result("A2", result2)
        tracker.batches_completed.append("B1")

        duration = tracker.get_batch_duration("B1")
        # Should be from earliest start (10:00:00) to latest end (10:06:00) = 6 minutes = 360000 ms
        self.assertIsNotNone(duration)
        self.assertEqual(duration, 360000)

    def test_tracker_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
        tracker = ExecutionTracker.create(plan)

        result = AgentResult(
            status="completed",
            started_at="2025-12-15T10:00:00",
            completed_at="2025-12-15T10:05:00",
            duration_ms=300000,
            output_summary="Done"
        )
        tracker.record_agent_result("A1", result)

        # Serialize to dict
        d = tracker.to_dict()
        self.assertEqual(d["plan_id"], plan.plan_id)
        self.assertIn("A1", d["agent_results"])

        # Deserialize from dict
        tracker2 = ExecutionTracker.from_dict(d)
        self.assertEqual(tracker2.plan_id, tracker.plan_id)
        self.assertEqual(tracker2.execution_id, tracker.execution_id)
        self.assertIn("A1", tracker2.agent_results)

    def test_tracker_save_load(self):
        """Test persistence roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = OrchestrationPlan.create(title="Test", goal={"summary": "Test"})
            tracker = ExecutionTracker.create(plan)

            result = AgentResult(
                status="completed",
                started_at="2025-12-15T10:00:00",
                completed_at="2025-12-15T10:05:00",
                duration_ms=300000,
                output_summary="Done"
            )
            tracker.record_agent_result("A1", result)

            # Save to custom path
            path = Path(temp_dir) / "test_execution.json"
            saved_path = tracker.save(path)
            self.assertTrue(saved_path.exists())

            # Load and verify
            loaded = ExecutionTracker.load(saved_path)
            self.assertEqual(loaded.plan_id, tracker.plan_id)
            self.assertEqual(loaded.execution_id, tracker.execution_id)
            self.assertIn("A1", loaded.agent_results)


class TestMetricsEvent(unittest.TestCase):
    """Tests for MetricsEvent."""

    def test_metrics_event_creation(self):
        """Test creating metrics event."""
        event = MetricsEvent(
            timestamp="2025-12-15T10:00:00",
            plan_id="OP-test",
            event_type="batch_start",
            batch_id="B1"
        )

        self.assertEqual(event.event_type, "batch_start")
        self.assertEqual(event.batch_id, "B1")

    def test_to_json_line(self):
        """Test JSONL serialization."""
        event = MetricsEvent(
            timestamp="2025-12-15T10:00:00",
            plan_id="OP-test",
            event_type="agent_complete",
            batch_id="B1",
            agent_id="A1",
            duration_ms=5000,
            success=True
        )

        json_line = event.to_json_line()
        self.assertIsInstance(json_line, str)
        self.assertNotIn("\n", json_line)

        # Should be valid JSON
        parsed = json.loads(json_line)
        self.assertEqual(parsed["event_type"], "agent_complete")
        self.assertEqual(parsed["duration_ms"], 5000)

    def test_from_json_line(self):
        """Test JSONL deserialization."""
        json_line = '{"timestamp":"2025-12-15T10:00:00","plan_id":"OP-test","event_type":"batch_complete","batch_id":"B1","agent_id":null,"duration_ms":300000,"success":true,"metadata":{}}'

        event = MetricsEvent.from_json_line(json_line)
        self.assertEqual(event.event_type, "batch_complete")
        self.assertEqual(event.duration_ms, 300000)
        self.assertTrue(event.success)


class TestOrchestrationMetrics(unittest.TestCase):
    """Tests for OrchestrationMetrics."""

    def setUp(self):
        """Create temporary directory for metrics file."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = Path(self.temp_dir) / "metrics.jsonl"

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_record_events(self):
        """Test JSONL append."""
        metrics = OrchestrationMetrics(self.metrics_file)

        metrics.record_batch_start("OP-test", "B1")
        metrics.record_agent_complete("OP-test", "B1", "A1", duration_ms=5000, success=True)

        # File should exist with 2 lines
        self.assertTrue(self.metrics_file.exists())
        lines = self.metrics_file.read_text().strip().split('\n')
        self.assertEqual(len(lines), 2)

    def test_get_events(self):
        """Test loading events."""
        metrics = OrchestrationMetrics(self.metrics_file)

        metrics.record_batch_start("OP-test1", "B1")
        metrics.record_batch_start("OP-test2", "B1")

        # Get all events
        all_events = metrics.get_events()
        self.assertEqual(len(all_events), 2)

        # Get events for specific plan
        filtered = metrics.get_events(plan_id="OP-test1")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].plan_id, "OP-test1")

    def test_get_events_filtered(self):
        """Test filtering by plan_id."""
        metrics = OrchestrationMetrics(self.metrics_file)

        metrics.record("batch_start", "OP-plan1", batch_id="B1")
        metrics.record("batch_start", "OP-plan2", batch_id="B1")
        metrics.record("agent_complete", "OP-plan1", batch_id="B1", agent_id="A1", duration_ms=1000, success=True)

        events = metrics.get_events(plan_id="OP-plan1")
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.plan_id == "OP-plan1" for e in events))

    def test_get_summary(self):
        """Test aggregate statistics."""
        metrics = OrchestrationMetrics(self.metrics_file)

        # Record some events
        metrics.record_batch_complete("OP-plan1", "B1", duration_ms=300000, success=True)
        metrics.record_batch_complete("OP-plan1", "B2", duration_ms=400000, success=False)
        metrics.record_agent_complete("OP-plan1", "B1", "A1", duration_ms=100000, success=True)
        metrics.record_agent_complete("OP-plan1", "B1", "A2", duration_ms=200000, success=True)
        metrics.record_replan("OP-plan1", trigger="verification_fail", reason="Tests failed")

        summary = metrics.get_summary()

        self.assertEqual(summary["total_plans"], 1)
        self.assertEqual(summary["total_batches"], 2)
        self.assertEqual(summary["total_agents"], 2)
        self.assertEqual(summary["batch_success_rate"], 50.0)  # 1 out of 2 succeeded
        self.assertEqual(summary["agent_success_rate"], 100.0)  # Both agents succeeded
        self.assertEqual(summary["avg_batch_duration_ms"], 350000)  # (300000 + 400000) / 2
        self.assertEqual(summary["avg_agent_duration_ms"], 150000)  # (100000 + 200000) / 2
        self.assertEqual(summary["total_replans"], 1)

    def test_get_summary_empty(self):
        """Test summary with no events."""
        metrics = OrchestrationMetrics(self.metrics_file)

        summary = metrics.get_summary()

        self.assertEqual(summary["total_plans"], 0)
        self.assertEqual(summary["total_batches"], 0)
        self.assertEqual(summary["batch_success_rate"], 0.0)

    def test_get_failure_patterns(self):
        """Test failure analysis."""
        metrics = OrchestrationMetrics(self.metrics_file)

        # Record some failures
        metrics.record_batch_complete("OP-plan1", "B1", duration_ms=1000, success=False)
        metrics.record_batch_complete("OP-plan2", "B1", duration_ms=1000, success=False)
        metrics.record_agent_complete("OP-plan1", "B1", "A1", duration_ms=1000, success=False)
        metrics.record_replan("OP-plan1", trigger="verification_fail", reason="Tests failed")
        metrics.record_replan("OP-plan2", trigger="verification_fail", reason="Tests failed")

        patterns = metrics.get_failure_patterns()

        # Should have 3 failure types
        self.assertGreater(len(patterns), 0)

        # Most common should be listed first
        self.assertEqual(patterns[0]["failure_type"], "batch_complete")
        self.assertEqual(patterns[0]["count"], 2)

        # Check for replan trigger
        replan_pattern = next((p for p in patterns if p["failure_type"] == "replan:verification_fail"), None)
        self.assertIsNotNone(replan_pattern)
        self.assertEqual(replan_pattern["count"], 2)

    def test_record_verification(self):
        """Test verification recording."""
        metrics = OrchestrationMetrics(self.metrics_file)

        metrics.record_verification(
            "OP-test",
            "B1",
            passed=True,
            checks={"tests_pass": True, "no_conflicts": True}
        )

        events = metrics.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "verification")
        self.assertTrue(events[0].success)
        self.assertIn("checks", events[0].metadata)


class TestHelperFunctions(unittest.TestCase):
    """Tests for load_all_plans and get_plan_by_id."""

    def setUp(self):
        """Create temporary directory for plans."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_all_plans(self):
        """Test loading all plans from directory."""
        plans_dir = Path(self.temp_dir) / "plans"
        plans_dir.mkdir()

        # Create two plans
        plan1 = OrchestrationPlan.create(title="Plan 1", goal={"summary": "Test 1"})
        plan2 = OrchestrationPlan.create(title="Plan 2", goal={"summary": "Test 2"})

        plan1.save(plans_dir / f"{plan1.plan_id}.json")
        plan2.save(plans_dir / f"{plan2.plan_id}.json")

        # Load all plans
        loaded = load_all_plans(plans_dir)
        self.assertEqual(len(loaded), 2)

        titles = {p.title for p in loaded}
        self.assertIn("Plan 1", titles)
        self.assertIn("Plan 2", titles)

    def test_load_all_plans_empty_dir(self):
        """Test loading from empty directory."""
        plans_dir = Path(self.temp_dir) / "empty"

        loaded = load_all_plans(plans_dir)
        self.assertEqual(loaded, [])

    def test_get_plan_by_id(self):
        """Test finding plan by ID."""
        plans_dir = Path(self.temp_dir) / "plans"
        plans_dir.mkdir()

        plan = OrchestrationPlan.create(title="Test Plan", goal={"summary": "Test"})
        plan.save(plans_dir / f"{plan.plan_id}.json")

        # Find by ID
        found = get_plan_by_id(plan.plan_id, plans_dir)
        self.assertIsNotNone(found)
        self.assertEqual(found.plan_id, plan.plan_id)
        self.assertEqual(found.title, "Test Plan")

    def test_get_plan_by_id_not_found(self):
        """Test finding non-existent plan."""
        plans_dir = Path(self.temp_dir) / "plans"
        plans_dir.mkdir()

        found = get_plan_by_id("OP-nonexistent", plans_dir)
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
