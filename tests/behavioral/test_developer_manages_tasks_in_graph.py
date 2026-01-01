"""
Behavioral tests for task management in Graph of Thought.

Epic: Developer Manages Tasks in Graph of Thought

As a developer working with a task graph system built from first principles,
I want to create, update, and organize tasks with relationships,
So that I can track dependencies and manage complex work structures
using a system we built ourselves and control completely.
"""

import pytest
from pathlib import Path
from cortical.got.api import GoTManager
from cortical.got.errors import TransactionError


class TestDeveloperCreatesAndUpdatesTasksInGraph:
    """
    As a developer managing a complex project,
    I want to create tasks and update their properties,
    So that I can track work progress in a custom-built graph system.
    """

    def test_scenario_create_task_with_basic_properties(self, tmp_path):
        """
        Scenario: Creating a task with title and priority

        Given a GoT manager instance
        When I create a task with a title and priority
        Then the task is persisted with correct properties
        And I can retrieve it by its generated ID
        """
        # Given a GoT manager instance
        manager = GoTManager(tmp_path / ".got")

        # When I create a task with a title and priority
        task = manager.create_task(
            title="Implement custom search algorithm",
            priority="high",
            description="Build our own search from scratch"
        )

        # Then the task is persisted with correct properties
        assert task.id.startswith("T-")
        assert task.title == "Implement custom search algorithm"
        assert task.priority == "high"
        assert task.description == "Build our own search from scratch"
        assert task.status == "pending"  # default

        # And I can retrieve it by its generated ID
        retrieved = manager.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.title == task.title

    def test_scenario_update_task_status_and_priority(self, tmp_path):
        """
        Scenario: Updating a task's status and priority

        Given an existing task
        When I update its status to in_progress and priority to critical
        Then the changes are persisted
        And the task version is incremented
        """
        # Given an existing task
        manager = GoTManager(tmp_path / ".got")
        task = manager.create_task(
            title="Build custom caching layer",
            priority="medium",
            status="pending"
        )
        original_version = task.version

        # When I update its status to in_progress and priority to critical
        updated = manager.update_task(
            task.id,
            status="in_progress",
            priority="critical"
        )

        # Then the changes are persisted
        assert updated.status == "in_progress"
        assert updated.priority == "critical"

        # And the task version is incremented
        assert updated.version == original_version + 1

        # Verify persistence
        retrieved = manager.get_task(task.id)
        assert retrieved.status == "in_progress"
        assert retrieved.priority == "critical"

    def test_scenario_create_task_with_custom_properties(self, tmp_path):
        """
        Scenario: Creating a task with custom metadata

        Given a GoT manager
        When I create a task with custom properties dictionary
        Then the custom properties are stored
        And I can access them later
        """
        # Given a GoT manager
        manager = GoTManager(tmp_path / ".got")

        # When I create a task with custom properties dictionary
        task = manager.create_task(
            title="Optimize hand-built query engine",
            priority="high",
            properties={
                "component": "query_engine",
                "estimated_hours": 8,
                "tags": ["performance", "optimization"]
            }
        )

        # Then the custom properties are stored
        assert task.properties["component"] == "query_engine"
        assert task.properties["estimated_hours"] == 8
        assert "performance" in task.properties["tags"]

        # And I can access them later
        retrieved = manager.get_task(task.id)
        assert retrieved.properties["component"] == "query_engine"
        assert retrieved.properties["estimated_hours"] == 8


class TestDeveloperEstablishesTaskRelationships:
    """
    As a developer managing task dependencies,
    I want to create relationships between tasks,
    So that I can model complex dependency chains in our custom graph system.
    """

    def test_scenario_create_dependency_between_tasks(self, tmp_path):
        """
        Scenario: Creating a dependency relationship

        Given two tasks A and B
        When I create a DEPENDS_ON edge from A to B
        Then the edge is created with correct properties
        And I can query the relationship
        """
        # Given two tasks A and B
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(
            title="Implement custom parser",
            priority="high"
        )
        task_b = manager.create_task(
            title="Design custom syntax",
            priority="high"
        )

        # When I create a DEPENDS_ON edge from A to B
        edge = manager.add_dependency(task_a.id, task_b.id)

        # Then the edge is created with correct properties
        assert edge.source_id == task_a.id
        assert edge.target_id == task_b.id
        assert edge.edge_type == "DEPENDS_ON"

        # And I can query the relationship
        outgoing, incoming = manager.get_edges_for_task(task_a.id)
        assert len(outgoing) == 1
        assert outgoing[0].target_id == task_b.id
        assert outgoing[0].edge_type == "DEPENDS_ON"

    def test_scenario_create_blocking_relationship(self, tmp_path):
        """
        Scenario: One task blocks another

        Given a blocker task and a blocked task
        When I create a BLOCKS edge
        Then the blocking relationship is established
        And I can find blockers for the blocked task
        """
        # Given a blocker task and a blocked task
        manager = GoTManager(tmp_path / ".got")
        blocker = manager.create_task(
            title="Build custom auth system",
            priority="critical"
        )
        blocked = manager.create_task(
            title="Implement user profile features",
            priority="high"
        )

        # When I create a BLOCKS edge
        edge = manager.add_blocks(blocker.id, blocked.id)

        # Then the blocking relationship is established
        assert edge.edge_type == "BLOCKS"
        assert edge.source_id == blocker.id
        assert edge.target_id == blocked.id

        # And I can find blockers for the blocked task
        blockers = manager.get_blockers(blocked.id)
        assert len(blockers) == 1
        assert blockers[0].id == blocker.id

    def test_scenario_query_task_dependents(self, tmp_path):
        """
        Scenario: Finding tasks that depend on a foundational task

        Given a foundational task with multiple dependents
        When I query for dependents
        Then I get all tasks that depend on it
        """
        # Given a foundational task with multiple dependents
        manager = GoTManager(tmp_path / ".got")
        foundation = manager.create_task(
            title="Build core data structures from scratch",
            priority="critical"
        )
        dependent1 = manager.create_task(title="Implement algorithm A")
        dependent2 = manager.create_task(title="Implement algorithm B")
        dependent3 = manager.create_task(title="Implement algorithm C")

        manager.add_dependency(dependent1.id, foundation.id)
        manager.add_dependency(dependent2.id, foundation.id)
        manager.add_dependency(dependent3.id, foundation.id)

        # When I query for dependents
        dependents = manager.get_dependents(foundation.id)

        # Then I get all tasks that depend on it
        assert len(dependents) == 3
        dependent_ids = {d.id for d in dependents}
        assert dependent1.id in dependent_ids
        assert dependent2.id in dependent_ids
        assert dependent3.id in dependent_ids


class TestDeveloperDeletesTasksWithSafetyChecks:
    """
    As a developer maintaining graph integrity,
    I want to safely delete tasks with dependency checks,
    So that I don't accidentally break our custom-built dependency graph.
    """

    def test_scenario_cannot_delete_task_with_dependents_without_force(self, tmp_path):
        """
        Scenario: Attempting to delete a task that has dependents

        Given a task with dependents
        When I attempt to delete it without force flag
        Then the deletion fails with an error
        And the task still exists
        Because we protect our graph integrity
        """
        # Given a task with dependents
        manager = GoTManager(tmp_path / ".got")
        foundation = manager.create_task(title="Core module we built")
        dependent = manager.create_task(title="Feature using core module")
        manager.add_dependency(dependent.id, foundation.id)

        # When I attempt to delete it without force flag
        # Then the deletion fails with an error
        with pytest.raises(TransactionError, match="has dependents"):
            manager.delete_task(foundation.id, force=False)

        # And the task still exists
        assert manager.get_task(foundation.id) is not None

    def test_scenario_force_delete_removes_task_and_edges(self, tmp_path):
        """
        Scenario: Force deleting a task with relationships

        Given a task with incoming and outgoing edges
        When I force delete the task
        Then the task is removed
        And all connected edges are removed
        Because force deletion cleans up the entire subgraph
        """
        # Given a task with incoming and outgoing edges
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(title="Task A")
        task_b = manager.create_task(title="Task B")
        task_c = manager.create_task(title="Task C")

        manager.add_dependency(task_b.id, task_a.id)
        manager.add_dependency(task_b.id, task_c.id)

        # When I force delete the task
        manager.delete_task(task_b.id, force=True)

        # Then the task is removed
        assert manager.get_task(task_b.id) is None

        # And all connected edges are removed
        outgoing_a, incoming_a = manager.get_edges_for_task(task_a.id)
        assert len(incoming_a) == 0  # Edge from B to A removed

        outgoing_c, incoming_c = manager.get_edges_for_task(task_c.id)
        assert len(incoming_c) == 0  # Edge from B to C removed


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
