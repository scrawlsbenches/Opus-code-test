"""
Behavioral tests for atomic delete operations in Graph of Thought.

Epic: Developer Expects Atomic Delete Operations

As a developer working with GoT,
I want delete operations to be atomic (all-or-nothing),
So that a crash during deletion doesn't leave orphaned edges or corrupt state.

BUG FIX REFERENCE: T-20251229-123957-0affebee
These tests verify the fix for non-atomic delete operations.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from cortical.got.api import GoTManager
from cortical.got.errors import TransactionError
from tests.conftest import _create_tx_manager, _create_got_manager


class TestDeleteOperationsAreAtomic:
    """
    As a developer relying on data integrity,
    I want delete operations to be atomic,
    So that crashes don't corrupt my task graph.
    """

    def test_scenario_delete_task_removes_all_connected_edges(self, tmp_path):
        """
        Scenario: Delete task removes task AND all connected edges atomically

        Given a task with multiple connected edges
        When I delete the task using delete_task
        Then the task is deleted
        And all connected edges are also deleted
        Because delete operations are now atomic.
        """
        # Given a task with multiple connected edges
        manager = _create_got_manager(tmp_path / ".got")

        task_a = manager.create_task(title="Task A - will be deleted", priority="high")
        task_b = manager.create_task(title="Task B - dependent")
        task_c = manager.create_task(title="Task C - dependency")

        # Create edges: B depends on A, A depends on C
        edge_ba = manager.add_dependency(task_b.id, task_a.id)
        edge_ac = manager.add_dependency(task_a.id, task_c.id)

        task_a_id = task_a.id
        edge_ba_id = edge_ba.id
        edge_ac_id = edge_ac.id
        entities_dir = tmp_path / ".got" / "entities"

        # Verify initial state
        assert (entities_dir / f"{task_a_id}.json").exists()
        assert (entities_dir / f"{edge_ba_id}.json").exists()
        assert (entities_dir / f"{edge_ac_id}.json").exists()

        # When I delete the task using delete_task API
        manager.delete_task(task_a_id, force=True)

        # Then the task is deleted
        assert not (entities_dir / f"{task_a_id}.json").exists()
        assert manager.get_task(task_a_id) is None

        # And all connected edges are also deleted
        assert not (entities_dir / f"{edge_ba_id}.json").exists()
        assert not (entities_dir / f"{edge_ac_id}.json").exists()

        # Verify no orphaned edges in a fresh manager
        manager2 = _create_got_manager(tmp_path / ".got")
        orphaned_edges = []
        for edge in manager2.list_edges():
            if edge.source_id == task_a_id or edge.target_id == task_a_id:
                orphaned_edges.append(edge)

        assert len(orphaned_edges) == 0, (
            f"Found {len(orphaned_edges)} orphaned edges after delete. "
            "Delete should remove task and all connected edges atomically."
        )

    def test_scenario_force_delete_with_dependents_is_atomic(self, tmp_path):
        """
        Scenario: Force delete with many edges removes all atomically

        Given a task with 5+ connected edges
        When I force delete the task
        Then all edges are also deleted
        Because delete operations are now atomic.
        """
        # Given a task with many edges
        manager = _create_got_manager(tmp_path / ".got")

        central_task = manager.create_task(title="Central task", priority="critical")

        # Create 5 tasks that depend on central task
        dependent_tasks = []
        for i in range(5):
            dep = manager.create_task(title=f"Dependent {i}")
            manager.add_dependency(dep.id, central_task.id)
            dependent_tasks.append(dep)

        central_id = central_task.id
        entities_dir = tmp_path / ".got" / "entities"

        # Count edges before
        edges_before = list(entities_dir.glob("E-*.json"))
        assert len(edges_before) == 5, "Should have 5 DEPENDS_ON edges"

        # When I force delete - should be atomic
        manager.delete_task(central_id, force=True)

        # Then task is deleted
        assert manager.get_task(central_id) is None
        assert not (entities_dir / f"{central_id}.json").exists()

        # And all edges connected to it are deleted
        manager2 = _create_got_manager(tmp_path / ".got")
        orphaned = []
        for edge in manager2.list_edges():
            if edge.source_id == central_id or edge.target_id == central_id:
                orphaned.append(edge)

        assert len(orphaned) == 0, (
            f"Found {len(orphaned)} orphaned edges after force delete. "
            "Atomic delete should remove all connected edges."
        )

    def test_scenario_delete_decision_removes_connected_edges(self, tmp_path):
        """
        Scenario: Decision deletion also removes connected edges atomically

        Given a decision with JUSTIFIES edges to tasks
        When I delete the decision
        Then the decision is deleted
        And all connected edges are deleted
        Because delete operations are atomic.
        """
        # Given a decision with edges
        manager = _create_got_manager(tmp_path / ".got")

        task1 = manager.create_task(title="Task 1")
        task2 = manager.create_task(title="Task 2")

        decision = manager.log_decision(
            title="Use approach A",
            rationale="Better performance",
        )

        decision_id = decision.id
        entities_dir = tmp_path / ".got" / "entities"

        # Create JUSTIFIES edges
        edge1 = manager.add_edge(decision_id, task1.id, "JUSTIFIES")
        edge2 = manager.add_edge(decision_id, task2.id, "JUSTIFIES")

        # Verify setup
        assert (entities_dir / f"{decision_id}.json").exists()
        assert (entities_dir / f"{edge1.id}.json").exists()
        assert (entities_dir / f"{edge2.id}.json").exists()

        # When I delete the decision
        manager.delete_decision(decision_id, force=True)

        # Then decision is deleted
        assert manager.get_decision(decision_id) is None
        assert not (entities_dir / f"{decision_id}.json").exists()

        # And edges are deleted
        manager2 = _create_got_manager(tmp_path / ".got")
        orphaned = []
        for edge in manager2.list_edges():
            if edge.source_id == decision_id:
                orphaned.append(edge)

        assert len(orphaned) == 0, (
            f"Found {len(orphaned)} orphaned JUSTIFIES edges after decision delete. "
            "Atomic delete should remove all connected edges."
        )


class TestDeletePreservesUnrelatedEntities:
    """
    As a developer expecting transactional behavior,
    I want deletes to only affect related entities,
    So that unrelated data remains intact.
    """

    def test_scenario_delete_task_preserves_unrelated_tasks(self, tmp_path):
        """
        Scenario: Deleting one task doesn't affect unrelated tasks

        Given multiple independent tasks
        When I delete one task
        Then the other tasks remain unchanged
        Because delete only affects the target and connected edges.
        """
        # Given multiple independent tasks
        manager = _create_got_manager(tmp_path / ".got")
        task1 = manager.create_task(title="Task 1", priority="high")
        task2 = manager.create_task(title="Task 2", priority="medium")
        task3 = manager.create_task(title="Task 3", priority="low")

        task1_id = task1.id
        task2_id = task2.id
        task3_id = task3.id

        # When I delete task1
        manager.delete_task(task1_id, force=True)

        # Then task1 is deleted
        assert manager.get_task(task1_id) is None

        # And task2 and task3 remain
        assert manager.get_task(task2_id) is not None
        assert manager.get_task(task3_id) is not None
        assert manager.get_task(task2_id).title == "Task 2"
        assert manager.get_task(task3_id).title == "Task 3"

    def test_scenario_delete_task_preserves_unconnected_edges(self, tmp_path):
        """
        Scenario: Deleting a task only removes edges connected to it

        Given multiple tasks with various edges
        When I delete one task
        Then only edges connected to that task are deleted
        And other edges remain intact.
        """
        # Given multiple tasks with various edges
        manager = _create_got_manager(tmp_path / ".got")

        task_a = manager.create_task(title="Task A")
        task_b = manager.create_task(title="Task B")
        task_c = manager.create_task(title="Task C")
        task_d = manager.create_task(title="Task D")

        # A -> B (will be deleted when A is deleted)
        edge_ab = manager.add_dependency(task_a.id, task_b.id)
        # C -> D (should remain)
        edge_cd = manager.add_dependency(task_c.id, task_d.id)

        # When I delete task A
        manager.delete_task(task_a.id, force=True)

        # Then A and edge A->B are deleted
        assert manager.get_task(task_a.id) is None

        # And C, D, and edge C->D remain
        assert manager.get_task(task_c.id) is not None
        assert manager.get_task(task_d.id) is not None

        # Check the edge C->D still exists
        edges = manager.list_edges()
        cd_edges = [e for e in edges if e.source_id == task_c.id and e.target_id == task_d.id]
        assert len(cd_edges) == 1, "Edge C->D should remain intact"


class TestDeleteErrorHandling:
    """
    As a developer expecting clear error handling,
    I want delete operations to fail cleanly,
    So that I know when something goes wrong.
    """

    def test_scenario_delete_nonexistent_task_raises_error(self, tmp_path):
        """
        Scenario: Deleting a non-existent task raises TransactionError

        Given a manager with no tasks
        When I try to delete a non-existent task ID
        Then a TransactionError is raised
        Because we can't delete what doesn't exist.
        """
        manager = GoTManager(tmp_path / ".got")

        with pytest.raises(TransactionError, match="Task not found"):
            manager.delete_task("T-nonexistent-task", force=True)

    def test_scenario_delete_task_with_dependents_without_force_fails(self, tmp_path):
        """
        Scenario: Deleting a task with dependents requires force flag

        Given a task that other tasks depend on
        When I try to delete it without force=True
        Then a TransactionError is raised
        And the task remains intact
        Because we protect graph integrity by default.
        """
        manager = _create_got_manager(tmp_path / ".got")

        foundation = manager.create_task(title="Foundation task")
        dependent = manager.create_task(title="Dependent task")
        manager.add_dependency(dependent.id, foundation.id)

        # Without force, should fail
        with pytest.raises(TransactionError, match="has dependents"):
            manager.delete_task(foundation.id, force=False)

        # Task should still exist
        assert manager.get_task(foundation.id) is not None


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_delete_test")
