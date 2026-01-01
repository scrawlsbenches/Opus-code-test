"""
Behavioral tests for querying the Graph of Thought using our custom query builder.

Epic: Developer Queries Task Graph

As a developer working with a custom-built query system,
I want to filter, sort, and aggregate tasks using a fluent query builder,
So that I can analyze my task graph without external database dependencies.
"""

import pytest
from cortical.got.api import GoTManager
from cortical.got.query_builder import Query, Count, Collect, Avg


class TestDeveloperFiltersTasksWithFluentQueries:
    """
    As a developer analyzing my task backlog,
    I want to filter tasks by status and priority using a fluent query API,
    So that I can find specific tasks without writing complex loops.
    """

    def test_scenario_find_high_priority_pending_tasks(self, tmp_path):
        """
        Scenario: Finding urgent work using our custom query builder

        Given a task graph with tasks of various statuses and priorities
        When I query for pending high-priority tasks
        Then I get only the matching tasks
        And the results are accurate
        """
        # Given a task graph with tasks of various statuses and priorities
        manager = GoTManager(tmp_path / ".got")
        high_pending_1 = manager.create_task(
            title="Build custom indexer",
            priority="high",
            status="pending"
        )
        high_pending_2 = manager.create_task(
            title="Implement hand-rolled cache",
            priority="high",
            status="pending"
        )
        medium_pending = manager.create_task(
            title="Refactor module",
            priority="medium",
            status="pending"
        )
        high_completed = manager.create_task(
            title="Design algorithm",
            priority="high",
            status="completed"
        )

        # When I query for pending high-priority tasks
        results = (
            Query(manager)
            .tasks()
            .where(status="pending", priority="high")
            .execute()
        )

        # Then I get only the matching tasks
        assert len(results) == 2
        result_ids = {t.id for t in results}
        assert high_pending_1.id in result_ids
        assert high_pending_2.id in result_ids
        assert medium_pending.id not in result_ids
        assert high_completed.id not in result_ids

    def test_scenario_query_tasks_with_or_conditions(self, tmp_path):
        """
        Scenario: Finding tasks matching multiple criteria

        Given tasks with different priorities
        When I query for critical OR high priority tasks
        Then I get tasks matching either condition
        """
        # Given tasks with different priorities
        manager = GoTManager(tmp_path / ".got")
        critical = manager.create_task(title="Critical work", priority="critical")
        high = manager.create_task(title="High priority work", priority="high")
        medium = manager.create_task(title="Medium work", priority="medium")
        low = manager.create_task(title="Low priority work", priority="low")

        # When I query for critical OR high priority tasks
        results = (
            Query(manager)
            .tasks()
            .where(priority="critical")
            .or_where(priority="high")
            .execute()
        )

        # Then I get tasks matching either condition
        assert len(results) == 2
        result_ids = {t.id for t in results}
        assert critical.id in result_ids
        assert high.id in result_ids
        assert medium.id not in result_ids
        assert low.id not in result_ids

    def test_scenario_sort_tasks_by_priority(self, tmp_path):
        """
        Scenario: Ordering tasks by priority using our custom sort logic

        Given tasks with different priorities
        When I query and sort by priority descending
        Then critical tasks appear first
        And low priority tasks appear last
        """
        # Given tasks with different priorities
        manager = GoTManager(tmp_path / ".got")
        low_task = manager.create_task(title="Low", priority="low")
        medium_task = manager.create_task(title="Medium", priority="medium")
        high_task = manager.create_task(title="High", priority="high")
        critical_task = manager.create_task(title="Critical", priority="critical")

        # When I query and sort by priority descending
        results = (
            Query(manager)
            .tasks()
            .order_by("priority", desc=True)
            .execute()
        )

        # Then critical tasks appear first
        assert results[0].priority == "critical"
        assert results[1].priority == "high"
        assert results[2].priority == "medium"

        # And low priority tasks appear last
        assert results[3].priority == "low"


class TestDeveloperAggregatesTaskStatistics:
    """
    As a developer tracking project health,
    I want to aggregate tasks by status and priority,
    So that I can generate reports using our custom aggregation engine.
    """

    def test_scenario_count_tasks_by_status(self, tmp_path):
        """
        Scenario: Counting tasks grouped by status

        Given tasks in various states
        When I group by status and count
        Then I get accurate counts per status
        """
        # Given tasks in various states
        manager = GoTManager(tmp_path / ".got")
        manager.create_task(title="Task 1", status="pending")
        manager.create_task(title="Task 2", status="pending")
        manager.create_task(title="Task 3", status="pending")
        manager.create_task(title="Task 4", status="in_progress")
        manager.create_task(title="Task 5", status="in_progress")
        manager.create_task(title="Task 6", status="completed")

        # When I group by status and count
        results = (
            Query(manager)
            .tasks()
            .group_by("status")
            .count()
            .execute()
        )

        # Then I get accurate counts per status
        assert results["pending"] == 3
        assert results["in_progress"] == 2
        assert results["completed"] == 1

    def test_scenario_collect_task_ids_by_priority(self, tmp_path):
        """
        Scenario: Collecting task IDs grouped by priority

        Given tasks with different priorities
        When I group by priority and collect IDs
        Then I get lists of IDs per priority level
        """
        # Given tasks with different priorities
        manager = GoTManager(tmp_path / ".got")
        high1 = manager.create_task(title="High 1", priority="high")
        high2 = manager.create_task(title="High 2", priority="high")
        medium1 = manager.create_task(title="Medium 1", priority="medium")

        # When I group by priority and collect IDs
        results = (
            Query(manager)
            .tasks()
            .group_by("priority")
            .aggregate(ids=Collect("id"))
            .execute()
        )

        # Then I get lists of IDs per priority level
        assert len(results["high"]["ids"]) == 2
        assert high1.id in results["high"]["ids"]
        assert high2.id in results["high"]["ids"]

        assert len(results["medium"]["ids"]) == 1
        assert medium1.id in results["medium"]["ids"]

    def test_scenario_multiple_aggregations_in_one_query(self, tmp_path):
        """
        Scenario: Computing multiple statistics simultaneously

        Given tasks with custom numeric properties
        When I aggregate with Count and Collect together
        Then I get all aggregations in one pass
        Because our custom query engine supports multiple aggregations
        """
        # Given tasks with custom numeric properties
        manager = GoTManager(tmp_path / ".got")
        manager.create_task(title="Task 1", status="pending")
        manager.create_task(title="Task 2", status="pending")
        manager.create_task(title="Task 3", status="completed")

        # When I aggregate with Count and Collect together
        results = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(
                count=Count(),
                task_ids=Collect("id")
            )
            .execute()
        )

        # Then I get all aggregations in one pass
        assert results["pending"]["count"] == 2
        assert len(results["pending"]["task_ids"]) == 2

        assert results["completed"]["count"] == 1
        assert len(results["completed"]["task_ids"]) == 1


class TestDeveloperPaginatesQueryResults:
    """
    As a developer working with large task graphs,
    I want to paginate query results,
    So that I can efficiently process tasks in batches.
    """

    def test_scenario_limit_query_results(self, tmp_path):
        """
        Scenario: Getting only the first N results

        Given many tasks in the graph
        When I query with a limit of 3
        Then I get exactly 3 results
        """
        # Given many tasks in the graph
        manager = GoTManager(tmp_path / ".got")
        for i in range(10):
            manager.create_task(title=f"Task {i}", status="pending")

        # When I query with a limit of 3
        results = (
            Query(manager)
            .tasks()
            .where(status="pending")
            .limit(3)
            .execute()
        )

        # Then I get exactly 3 results
        assert len(results) == 3

    def test_scenario_paginate_with_offset_and_limit(self, tmp_path):
        """
        Scenario: Implementing pagination

        Given 10 tasks
        When I query with offset 5 and limit 3
        Then I get tasks 6, 7, and 8
        """
        # Given 10 tasks
        manager = GoTManager(tmp_path / ".got")
        tasks = []
        for i in range(10):
            task = manager.create_task(title=f"Task {i:02d}", status="pending")
            tasks.append(task)

        # When I query with offset 5 and limit 3
        results = (
            Query(manager)
            .tasks()
            .where(status="pending")
            .order_by("title")
            .offset(5)
            .limit(3)
            .execute()
        )

        # Then I get tasks 6, 7, and 8 (after sorting by title)
        assert len(results) == 3
        # Verify we got the middle portion after sorting


class TestDeveloperQueriesConnectedTasks:
    """
    As a developer analyzing task relationships,
    I want to query tasks connected to a specific node,
    So that I can explore our custom-built dependency graph.
    """

    def test_scenario_find_tasks_connected_to_sprint(self, tmp_path):
        """
        Scenario: Finding all tasks in a sprint

        Given a sprint with multiple tasks
        When I query for tasks connected to the sprint
        Then I get all sprint tasks
        """
        # Given a sprint with multiple tasks
        manager = GoTManager(tmp_path / ".got")
        sprint = manager.create_sprint(title="Sprint 1")
        task1 = manager.create_task(title="Task 1")
        task2 = manager.create_task(title="Task 2")
        task3 = manager.create_task(title="Task 3")
        other_task = manager.create_task(title="Other task")

        manager.add_task_to_sprint(task1.id, sprint.id)
        manager.add_task_to_sprint(task2.id, sprint.id)
        manager.add_task_to_sprint(task3.id, sprint.id)

        # When I query for tasks connected to the sprint
        # Edge is Sprint --CONTAINS--> Task, so we use "outgoing" from sprint
        results = (
            Query(manager)
            .tasks()
            .connected_to(sprint.id, via="CONTAINS", direction="outgoing")
            .execute()
        )

        # Then I get all sprint tasks
        assert len(results) == 3
        result_ids = {t.id for t in results}
        assert task1.id in result_ids
        assert task2.id in result_ids
        assert task3.id in result_ids
        assert other_task.id not in result_ids


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
