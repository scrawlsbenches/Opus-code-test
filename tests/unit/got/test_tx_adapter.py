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
