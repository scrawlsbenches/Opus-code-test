"""
Tests for TransactionalGoTAdapter sprint methods.

Tests sprint creation, retrieval, listing, task association, and progress tracking.
"""

import pytest
from pathlib import Path

# Import from scripts since TransactionalGoTAdapter is in got_utils.py
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.got_utils import TransactionalGoTAdapter
from cortical.reasoning.graph_of_thought import NodeType


class TestTransactionalGoTAdapterSprint:
    """Tests for sprint-related methods in TransactionalGoTAdapter."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create adapter with temporary directory."""
        got_dir = tmp_path / ".got"
        return TransactionalGoTAdapter(got_dir)

    def test_create_sprint_basic(self, adapter):
        """Create sprint with title only."""
        sprint_id = adapter.create_sprint("Sprint 1")

        assert sprint_id is not None
        assert sprint_id.startswith("S-")

        # Verify sprint was created
        sprint = adapter.get_sprint(sprint_id)
        assert sprint is not None
        assert sprint.content == "Sprint 1"
        assert sprint.properties.get("status") == "available"
        assert sprint.node_type == NodeType.GOAL

    def test_create_sprint_with_number(self, adapter):
        """Create sprint with explicit number."""
        sprint_id = adapter.create_sprint("Sprint 5", number=5)

        assert sprint_id is not None
        # Number should be in properties
        sprint = adapter.get_sprint(sprint_id)
        assert sprint is not None
        assert sprint.content == "Sprint 5"
        assert sprint.properties.get("number") == 5

    def test_create_sprint_with_epic(self, adapter):
        """Create sprint linked to an epic."""
        # First create an epic (this will create it in the TX backend)
        epic = adapter._manager.create_epic("Epic 1", epic_id="epic:test-epic-1")

        # Create sprint linked to epic
        sprint_id = adapter.create_sprint("Sprint 1", epic_id="epic:test-epic-1")

        assert sprint_id is not None
        sprint = adapter.get_sprint(sprint_id)
        assert sprint is not None
        assert sprint.properties.get("epic_id") == "epic:test-epic-1"

    def test_get_sprint_existing(self, adapter):
        """Get an existing sprint."""
        sprint_id = adapter.create_sprint("Test Sprint")

        # Get the sprint
        sprint = adapter.get_sprint(sprint_id)

        assert sprint is not None
        assert sprint.id == sprint_id
        assert sprint.content == "Test Sprint"
        assert sprint.node_type == NodeType.GOAL

    def test_get_sprint_not_found(self, adapter):
        """Get non-existent sprint returns None."""
        sprint = adapter.get_sprint("S-999999")
        assert sprint is None

    def test_list_sprints_empty(self, adapter):
        """List sprints when none exist."""
        sprints = adapter.list_sprints()
        assert sprints == []

    def test_list_sprints_all(self, adapter):
        """List all sprints."""
        # Create multiple sprints
        sprint1_id = adapter.create_sprint("Sprint 1", number=1)
        sprint2_id = adapter.create_sprint("Sprint 2", number=2)
        sprint3_id = adapter.create_sprint("Sprint 3", number=3)

        # List all
        sprints = adapter.list_sprints()

        assert len(sprints) == 3
        sprint_ids = [s.id for s in sprints]
        assert sprint1_id in sprint_ids
        assert sprint2_id in sprint_ids
        assert sprint3_id in sprint_ids

    def test_list_sprints_by_status(self, adapter):
        """Filter sprints by status."""
        # Create sprints with different statuses - use explicit numbers to avoid ID collision
        adapter.create_sprint("Available Sprint 1", number=101)
        adapter.create_sprint("Available Sprint 2", number=102)

        # Create a sprint and update its status using TX backend
        sprint_in_progress = adapter._manager.create_sprint("InProgress Sprint", number=103)
        adapter._manager.update_sprint(sprint_in_progress.id, status="in_progress")

        # List by status
        available = adapter.list_sprints(status="available")
        in_progress = adapter.list_sprints(status="in_progress")

        assert len(available) == 2
        assert len(in_progress) == 1
        assert in_progress[0].properties.get("status") == "in_progress"

    def test_get_sprint_tasks_empty(self, adapter):
        """Get tasks for sprint with no tasks."""
        sprint_id = adapter.create_sprint("Empty Sprint")

        tasks = adapter.get_sprint_tasks(sprint_id)
        assert tasks == []

    def test_get_sprint_tasks_with_tasks(self, adapter):
        """Get tasks for sprint with linked tasks."""
        # Create sprint
        sprint_id = adapter.create_sprint("Sprint with Tasks")

        # Create tasks
        task1_id = adapter.create_task("Task 1", priority="high")
        task2_id = adapter.create_task("Task 2", priority="medium")
        task3_id = adapter.create_task("Task 3", priority="low")

        # Link tasks to sprint using TX backend
        adapter._manager.add_task_to_sprint(task1_id, sprint_id)
        adapter._manager.add_task_to_sprint(task2_id, sprint_id)
        adapter._manager.add_task_to_sprint(task3_id, sprint_id)

        # Get tasks
        tasks = adapter.get_sprint_tasks(sprint_id)

        assert len(tasks) == 3
        task_ids = [t.id for t in tasks]
        assert task1_id in task_ids
        assert task2_id in task_ids
        assert task3_id in task_ids

    def test_get_sprint_progress_empty(self, adapter):
        """Get progress for sprint with no tasks."""
        sprint_id = adapter.create_sprint("Empty Sprint")

        progress = adapter.get_sprint_progress(sprint_id)

        assert progress["total_tasks"] == 0
        assert progress["completed"] == 0
        assert progress["progress_percent"] == 0.0

    def test_get_sprint_progress_with_tasks(self, adapter):
        """Get progress for sprint with various task statuses."""
        # Create sprint
        sprint_id = adapter.create_sprint("Active Sprint")

        # Create tasks with pending status (default)
        task1_id = adapter.create_task("Task 1")
        task2_id = adapter.create_task("Task 2")
        task3_id = adapter.create_task("Task 3")
        task4_id = adapter.create_task("Task 4")
        task5_id = adapter.create_task("Task 5")

        # Link tasks to sprint
        adapter._manager.add_task_to_sprint(task1_id, sprint_id)
        adapter._manager.add_task_to_sprint(task2_id, sprint_id)
        adapter._manager.add_task_to_sprint(task3_id, sprint_id)
        adapter._manager.add_task_to_sprint(task4_id, sprint_id)
        adapter._manager.add_task_to_sprint(task5_id, sprint_id)

        # Update some task statuses
        adapter._manager.update_task(task1_id, status="completed")
        adapter._manager.update_task(task2_id, status="completed")
        adapter._manager.update_task(task3_id, status="in_progress")

        # Get progress
        progress = adapter.get_sprint_progress(sprint_id)

        assert progress["total_tasks"] == 5
        assert progress["completed"] == 2
        assert progress["progress_percent"] == 40.0  # 2/5 = 40%
        assert progress["by_status"]["completed"] == 2
        assert progress["by_status"]["in_progress"] == 1
        assert progress["by_status"]["pending"] == 2

    def test_get_sprint_progress_all_completed(self, adapter):
        """Get progress for fully completed sprint."""
        sprint_id = adapter.create_sprint("Completed Sprint")

        # Create tasks and link to sprint
        for i in range(3):
            task_id = adapter.create_task(f"Task {i+1}")
            adapter._manager.add_task_to_sprint(task_id, sprint_id)
            adapter._manager.update_task(task_id, status="completed")

        progress = adapter.get_sprint_progress(sprint_id)

        assert progress["total_tasks"] == 3
        assert progress["completed"] == 3
        assert progress["progress_percent"] == 100.0

    def test_sprint_properties(self, adapter):
        """Verify sprint properties are correctly set."""
        sprint_id = adapter.create_sprint("Test Sprint", number=10)
        sprint = adapter.get_sprint(sprint_id)

        # Check basic properties
        assert sprint.properties["name"] == "Test Sprint"
        assert sprint.properties["status"] == "available"
        assert sprint.properties["number"] == 10

        # Check metadata
        assert "created_at" in sprint.metadata
        assert "modified_at" in sprint.metadata

    def test_get_current_sprint(self, adapter):
        """Get the currently active sprint."""
        # Initially no sprint is active
        current = adapter.get_current_sprint()
        assert current is None

        # Create and start a sprint
        sprint = adapter._manager.create_sprint("Current Sprint")
        adapter._manager.update_sprint(sprint.id, status="in_progress")

        # Now should return the in-progress sprint
        current = adapter.get_current_sprint()
        assert current is not None
        assert current.content == "Current Sprint"
        assert current.properties.get("status") == "in_progress"


class TestTransactionalGoTAdapterDecisions:
    """Tests for decision-related methods in TransactionalGoTAdapter."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create adapter with temporary directory."""
        got_dir = tmp_path / ".got"
        return TransactionalGoTAdapter(got_dir)

    def test_log_decision_basic(self, adapter):
        """Log a basic decision."""
        decision_id = adapter.log_decision(
            decision="Use TX backend",
            rationale="ACID guarantees needed",
        )

        assert decision_id is not None
        assert decision_id.startswith("D-")

    def test_log_decision_with_affects(self, adapter):
        """Log decision with affected tasks."""
        # Create a task first
        task_id = adapter.create_task("Test Task")

        # Log decision affecting that task
        decision_id = adapter.log_decision(
            decision="Prioritize this task",
            rationale="Customer request",
            affects=[task_id],
        )

        assert decision_id is not None

        # Verify the decision is linked via why()
        reasons = adapter.why(task_id)
        assert len(reasons) == 1
        assert reasons[0]["decision_id"] == decision_id
        assert reasons[0]["decision"] == "Prioritize this task"
        assert reasons[0]["rationale"] == "Customer request"

    def test_log_decision_with_alternatives(self, adapter):
        """Log decision with alternatives considered."""
        decision_id = adapter.log_decision(
            decision="Use JSON storage",
            rationale="Human-readable and git-friendly",
            alternatives=["SQLite", "Pickle", "YAML"],
        )

        assert decision_id is not None

        # Verify alternatives are stored
        decisions = adapter.list_decisions()
        decision = next(d for d in decisions if d.id == decision_id)
        assert "SQLite" in decision.properties.get("alternatives", [])

    def test_log_decision_with_context(self, adapter):
        """Log decision with context metadata."""
        decision_id = adapter.log_decision(
            decision="Add validation",
            rationale="Prevent invalid data",
            context={"file": "api.py", "line": 123},
        )

        assert decision_id is not None

    def test_list_decisions_empty(self, adapter):
        """List decisions when none exist."""
        decisions = adapter.list_decisions()
        assert decisions == []

    def test_list_decisions_multiple(self, adapter):
        """List multiple decisions."""
        id1 = adapter.log_decision("Decision 1", "Reason 1")
        id2 = adapter.log_decision("Decision 2", "Reason 2")
        id3 = adapter.log_decision("Decision 3", "Reason 3")

        decisions = adapter.list_decisions()

        assert len(decisions) == 3
        decision_ids = [d.id for d in decisions]
        assert id1 in decision_ids
        assert id2 in decision_ids
        assert id3 in decision_ids

    def test_get_decisions_for_task(self, adapter):
        """Get only decisions affecting a specific task."""
        # Create two tasks
        task1_id = adapter.create_task("Task 1")
        task2_id = adapter.create_task("Task 2")

        # Log decisions affecting different tasks
        adapter.log_decision("For task 1", "Reason", affects=[task1_id])
        adapter.log_decision("For task 2", "Reason", affects=[task2_id])
        adapter.log_decision("For both", "Reason", affects=[task1_id, task2_id])

        # Get decisions for task 1
        task1_decisions = adapter.get_decisions_for_task(task1_id)
        assert len(task1_decisions) == 2
        contents = [d.content for d in task1_decisions]
        assert "For task 1" in contents
        assert "For both" in contents

        # Get decisions for task 2
        task2_decisions = adapter.get_decisions_for_task(task2_id)
        assert len(task2_decisions) == 2
        contents = [d.content for d in task2_decisions]
        assert "For task 2" in contents
        assert "For both" in contents

    def test_why_no_decisions(self, adapter):
        """Query why for task with no decisions."""
        task_id = adapter.create_task("New Task")
        reasons = adapter.why(task_id)
        assert reasons == []

    def test_why_multiple_decisions(self, adapter):
        """Query why for task with multiple decisions."""
        task_id = adapter.create_task("Important Task")

        # Log multiple decisions
        adapter.log_decision(
            decision="Created for customer X",
            rationale="Customer request",
            affects=[task_id],
        )
        adapter.log_decision(
            decision="Set to high priority",
            rationale="Revenue impact",
            affects=[task_id],
            alternatives=["medium", "low"],
        )

        # Query why
        reasons = adapter.why(task_id)

        assert len(reasons) == 2
        decisions = [r["decision"] for r in reasons]
        assert "Created for customer X" in decisions
        assert "Set to high priority" in decisions

        # Check structure of why response
        for reason in reasons:
            assert "decision_id" in reason
            assert "decision" in reason
            assert "rationale" in reason
            assert "alternatives" in reason
            assert "created_at" in reason

    def test_why_returns_rationale(self, adapter):
        """Verify why returns full rationale."""
        task_id = adapter.create_task("Task with rationale")

        adapter.log_decision(
            decision="Complex decision",
            rationale="This is a detailed rationale explaining why this decision was made.",
            affects=[task_id],
        )

        reasons = adapter.why(task_id)
        assert len(reasons) == 1
        assert "detailed rationale" in reasons[0]["rationale"]

    def test_why_returns_alternatives(self, adapter):
        """Verify why returns alternatives considered."""
        task_id = adapter.create_task("Task with alternatives")

        adapter.log_decision(
            decision="Chose option A",
            rationale="Best fit",
            affects=[task_id],
            alternatives=["Option B", "Option C", "Option D"],
        )

        reasons = adapter.why(task_id)
        assert len(reasons) == 1
        assert len(reasons[0]["alternatives"]) == 3
        assert "Option B" in reasons[0]["alternatives"]
