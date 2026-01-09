"""
Tests for GoT high-level API (GoTManager and TransactionContext).

Tests context manager behavior, single-operation methods, and error handling.
"""

import tempfile
import pytest
from pathlib import Path

from cortical.got.api import GoTManager, generate_task_id, generate_decision_id
from cortical.got.types import Task, Decision, Edge
from cortical.got.errors import TransactionError
from cortical.core.bootstrap import create_container


class TestGoTManager:
    """Tests for GoTManager high-level API."""

    @pytest.fixture
    def got_dir(self, tmp_path):
        """Provide temporary GoT directory."""
        return tmp_path / ".got"

    @pytest.fixture
    def manager(self, got_dir):
        """Provide GoTManager instance via container."""
        container = create_container(got_dir=got_dir)
        return container.resolve(GoTManager)

    def test_get_task_returns_none_for_missing(self, manager):
        """get_task returns None for non-existent task."""
        result = manager.get_task("T-20251221-000000-abcd")
        assert result is None

    def test_single_operation_create_task(self, manager):
        """manager.create_task() single-operation method works."""
        task = manager.create_task(
            "Single-op task",
            priority="high",
            description="Created in one call"
        )

        assert task.title == "Single-op task"
        assert task.priority == "high"
        assert task.description == "Created in one call"

        # Verify persistence
        retrieved = manager.get_task(task.id)
        assert retrieved is not None
        assert retrieved.title == "Single-op task"

    def test_single_operation_update_task(self, manager):
        """manager.update_task() single-operation method works."""
        # Create task
        task = manager.create_task("Original", status="pending")

        # Update using single-operation method
        updated = manager.update_task(
            task.id,
            title="Modified",
            status="completed"
        )

        assert updated.title == "Modified"
        assert updated.status == "completed"

        # Verify persistence
        retrieved = manager.get_task(task.id)
        assert retrieved.title == "Modified"
        assert retrieved.status == "completed"

    def test_add_edge_creates_relationship(self, manager):
        """Edge creation between tasks works correctly."""
        # Create two tasks
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")

        # Create edge
        edge = manager.add_edge(
            task1.id,
            task2.id,
            "DEPENDS_ON",
            weight=0.8
        )

        assert edge.source_id == task1.id
        assert edge.target_id == task2.id
        assert edge.edge_type == "DEPENDS_ON"
        assert edge.weight == 0.8
        assert edge.id == f"E-{task1.id}-{task2.id}-DEPENDS_ON"

class TestHelperFunctions:
    """Tests for ID generation helper functions."""

    def test_generate_task_id_format(self):
        """Task ID has correct format T-YYYYMMDD-HHMMSS-XXXX."""
        task_id = generate_task_id()
        assert task_id.startswith("T-")
        parts = task_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # XXXXXXXX hex

    def test_generate_decision_id_format(self):
        """Decision ID has correct format D-YYYYMMDD-HHMMSS-XXXX."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("D-")
        parts = decision_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # XXXXXXXX hex

    def test_generate_task_id_uniqueness(self):
        """Generated task IDs are unique."""
        ids = [generate_task_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique


class TestDecisionOperations:
    """Tests for decision operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Provide GoTManager instance via container."""
        container = create_container(got_dir=tmp_path / ".got")
        return container.resolve(GoTManager)

    def test_create_decision(self, manager):
        """Decision creation works correctly."""
        decision = manager.create_decision(
            "Use PostgreSQL",
            rationale="Better ACID guarantees",
            affects=["T-001", "T-002"]
        )

        assert decision.title == "Use PostgreSQL"
        assert decision.rationale == "Better ACID guarantees"
        assert decision.affects == ["T-001", "T-002"]
        assert decision.id.startswith("D-")

class TestQueryAPI:
    """Tests for GoTManager query API methods."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Provide GoTManager instance via container."""
        container = create_container(got_dir=tmp_path / ".got")
        return container.resolve(GoTManager)

    def test_find_tasks_by_status(self, manager):
        """find_tasks filters by status correctly."""
        # Create tasks with different statuses
        task1 = manager.create_task("Task 1", status="pending")
        task2 = manager.create_task("Task 2", status="in_progress")
        task3 = manager.create_task("Task 3", status="pending")

        # Find pending tasks
        pending = manager.find_tasks(status="pending")
        assert len(pending) == 2
        pending_ids = {t.id for t in pending}
        assert task1.id in pending_ids
        assert task3.id in pending_ids

        # Find in_progress tasks
        in_progress = manager.find_tasks(status="in_progress")
        assert len(in_progress) == 1
        assert in_progress[0].id == task2.id

    def test_find_tasks_by_priority(self, manager):
        """find_tasks filters by priority correctly."""
        # Create tasks with different priorities
        task1 = manager.create_task("Task 1", priority="low")
        task2 = manager.create_task("Task 2", priority="high")
        task3 = manager.create_task("Task 3", priority="high")

        # Find high priority tasks
        high = manager.find_tasks(priority="high")
        assert len(high) == 2
        high_ids = {t.id for t in high}
        assert task2.id in high_ids
        assert task3.id in high_ids

        # Find low priority tasks
        low = manager.find_tasks(priority="low")
        assert len(low) == 1
        assert low[0].id == task1.id

    def test_find_tasks_by_title(self, manager):
        """find_tasks filters by title substring (case-insensitive)."""
        # Create tasks with different titles
        task1 = manager.create_task("Implement authentication")
        task2 = manager.create_task("Fix bug in database")
        task3 = manager.create_task("Implement authorization")

        # Find tasks with "implement" in title
        implement_tasks = manager.find_tasks(title_contains="implement")
        assert len(implement_tasks) == 2
        implement_ids = {t.id for t in implement_tasks}
        assert task1.id in implement_ids
        assert task3.id in implement_ids

        # Case-insensitive search
        auth_tasks = manager.find_tasks(title_contains="AUTH")
        assert len(auth_tasks) == 2

    def test_find_tasks_combined_filters(self, manager):
        """find_tasks with multiple filters works correctly."""
        # Create tasks
        task1 = manager.create_task("Fix auth bug", status="pending", priority="high")
        task2 = manager.create_task("Fix database bug", status="pending", priority="low")
        task3 = manager.create_task("Implement auth", status="in_progress", priority="high")

        # Find high priority, pending, with "auth" in title
        results = manager.find_tasks(status="pending", priority="high", title_contains="auth")
        assert len(results) == 1
        assert results[0].id == task1.id

    def test_find_tasks_empty_results(self, manager):
        """find_tasks returns empty list when no matches."""
        manager.create_task("Task 1", status="pending")

        results = manager.find_tasks(status="completed")
        assert results == []

    def test_find_tasks_no_filters(self, manager):
        """find_tasks with no filters returns all tasks."""
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")

        all_tasks = manager.find_tasks()
        assert len(all_tasks) == 2
        task_ids = {t.id for t in all_tasks}
        assert task1.id in task_ids
        assert task2.id in task_ids

    def test_get_blockers(self, manager):
        """get_blockers returns tasks that block the given task."""
        # Create tasks
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")
        task3 = manager.create_task("Task 3")
        blocked_task = manager.create_task("Blocked task")

        # Create BLOCKS edges
        manager.add_edge(task1.id, blocked_task.id, "BLOCKS")
        manager.add_edge(task2.id, blocked_task.id, "BLOCKS")

        # Get blockers
        blockers = manager.get_blockers(blocked_task.id)
        assert len(blockers) == 2
        blocker_ids = {t.id for t in blockers}
        assert task1.id in blocker_ids
        assert task2.id in blocker_ids
        assert task3.id not in blocker_ids

    def test_get_blockers_empty(self, manager):
        """get_blockers returns empty list when no blockers."""
        task = manager.create_task("Task 1")

        blockers = manager.get_blockers(task.id)
        assert blockers == []

    def test_get_dependents(self, manager):
        """get_dependents returns tasks that depend on the given task."""
        # Create tasks
        dependency = manager.create_task("Dependency task")
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")
        task3 = manager.create_task("Task 3")

        # Create DEPENDS_ON edges
        manager.add_edge(task1.id, dependency.id, "DEPENDS_ON")
        manager.add_edge(task2.id, dependency.id, "DEPENDS_ON")

        # Get dependents
        dependents = manager.get_dependents(dependency.id)
        assert len(dependents) == 2
        dependent_ids = {t.id for t in dependents}
        assert task1.id in dependent_ids
        assert task2.id in dependent_ids
        assert task3.id not in dependent_ids

    def test_get_dependents_empty(self, manager):
        """get_dependents returns empty list when no dependents."""
        task = manager.create_task("Task 1")

        dependents = manager.get_dependents(task.id)
        assert dependents == []

    def test_list_all_tasks(self, manager):
        """list_all_tasks returns all tasks in the store."""
        task1 = manager.create_task("Task 1", status="pending")
        task2 = manager.create_task("Task 2", status="completed")
        task3 = manager.create_task("Task 3", status="in_progress")

        all_tasks = manager.list_all_tasks()
        assert len(all_tasks) == 3
        task_ids = {t.id for t in all_tasks}
        assert task1.id in task_ids
        assert task2.id in task_ids
        assert task3.id in task_ids

    def test_list_all_tasks_empty(self, manager):
        """list_all_tasks returns empty list when no tasks."""
        all_tasks = manager.list_all_tasks()
        assert all_tasks == []

    def test_get_edges_for_task(self, manager):
        """get_edges_for_task returns outgoing and incoming edges."""
        # Create tasks
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")
        task3 = manager.create_task("Task 3")

        # Create edges
        edge1 = manager.add_edge(task1.id, task2.id, "DEPENDS_ON")
        edge2 = manager.add_edge(task3.id, task1.id, "BLOCKS")

        # Get edges for task1
        outgoing, incoming = manager.get_edges_for_task(task1.id)

        # Verify outgoing
        assert len(outgoing) == 1
        assert outgoing[0].id == edge1.id
        assert outgoing[0].source_id == task1.id
        assert outgoing[0].target_id == task2.id

        # Verify incoming
        assert len(incoming) == 1
        assert incoming[0].id == edge2.id
        assert incoming[0].source_id == task3.id
        assert incoming[0].target_id == task1.id

    def test_get_edges_for_task_empty(self, manager):
        """get_edges_for_task returns empty tuples when no edges."""
        task = manager.create_task("Task 1")

        outgoing, incoming = manager.get_edges_for_task(task.id)
        assert outgoing == []
        assert incoming == []

    def test_query_handles_missing_entities_dir(self, manager):
        """Query methods handle missing entities directory gracefully."""
        # Don't create any tasks - entities dir won't exist
        assert manager.find_tasks() == []
        assert manager.get_blockers("T-test") == []
        assert manager.get_dependents("T-test") == []
        assert manager.list_all_tasks() == []
        assert manager.get_edges_for_task("T-test") == ([], [])
