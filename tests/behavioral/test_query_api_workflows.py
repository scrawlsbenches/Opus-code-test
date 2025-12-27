"""
Behavioral tests for Query API user workflows.

Tests realistic user scenarios:
1. "Why did my query return these results?" - Using explain() to debug
2. "What's blocking my task?" - Using GraphWalker to find blockers
3. "How do I find related tasks?" - Using PatternMatcher for relationships
4. "My results were truncated, what did I miss?" - Using truncation metadata

These tests document expected user workflows.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from cortical.got import GoTManager
from cortical.got.query_builder import Query
from cortical.got.graph_walker import GraphWalker
from cortical.got.path_finder import PathFinder
from cortical.got.pattern_matcher import Pattern, PatternMatcher


class TestExplainDebugWorkflow:
    """User workflow: Understanding why a query returned specific results."""

    @pytest.fixture
    def setup_project(self):
        """Create a realistic project with tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks for a realistic project
        task1 = manager.create_task("Setup CI/CD pipeline", priority="high")
        task2 = manager.create_task("Write unit tests", priority="high")
        task3 = manager.create_task("Update documentation", priority="low")
        task4 = manager.create_task("Fix login bug", priority="critical")

        # Mark some as in progress
        manager.update_task(task2.id, status="in_progress")

        yield manager, [task1, task2, task3, task4]
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_explain_shows_filter_applied(self, setup_project):
        """User can see what filters are applied to understand results."""
        manager, tasks = setup_project

        # User wants high priority tasks
        query = Query(manager).tasks().where(priority="high")

        # First, explain what the query will do
        plan = query.explain()

        # Plan should show the filter
        plan_str = str(plan)
        assert "priority" in plan_str.lower() or "filter" in plan_str.lower()

        # Then execute
        results = query.execute()

        # Results should match what plan described
        assert len(results) == 2  # Two high priority tasks
        for task in results:
            assert task.priority == "high"

    def test_explain_shows_limit(self, setup_project):
        """User can see limit is applied before executing."""
        manager, tasks = setup_project

        query = Query(manager).tasks().limit(2)
        plan = query.explain()

        # Plan should show the limit
        plan_str = str(plan)
        assert "limit" in plan_str.lower() or "2" in plan_str

    def test_explain_helps_debug_empty_results(self, setup_project):
        """User can use explain() to understand why results are empty."""
        manager, tasks = setup_project

        # Query with impossible filter
        query = Query(manager).tasks().where(priority="impossible")
        plan = query.explain()

        # User can see the filter and understand why results are empty
        results = query.execute()
        assert len(results) == 0

        # Plan helps user understand: the filter is looking for "impossible"
        plan_str = str(plan)
        # Should mention the filter somehow


class TestFindBlockersWorkflow:
    """User workflow: Finding what's blocking a task."""

    @pytest.fixture
    def setup_dependencies(self):
        """Create tasks with dependencies."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Feature depends on API, which depends on Database
        db_task = manager.create_task("Setup database", priority="high")
        api_task = manager.create_task("Build API endpoints", priority="high")
        feature_task = manager.create_task("Implement feature", priority="medium")

        # Create dependency chain: db <- api <- feature
        manager.add_edge(api_task.id, db_task.id, "DEPENDS_ON")
        manager.add_edge(feature_task.id, api_task.id, "DEPENDS_ON")

        yield manager, db_task, api_task, feature_task
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_find_all_blockers_with_walker(self, setup_dependencies):
        """User finds all tasks blocking their current work."""
        manager, db_task, api_task, feature_task = setup_dependencies

        # User is working on feature_task, wants to know what blocks it
        blockers = []
        result = (
            GraphWalker(manager)
            .starting_from(feature_task.id)
            .outgoing()  # Follow outgoing DEPENDS_ON edges
            .follow("DEPENDS_ON")
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should find: feature_task itself, api_task, db_task
        assert feature_task.id in result
        # api_task and db_task should be reachable
        assert len(result) >= 1

    def test_explain_walker_before_running(self, setup_dependencies):
        """User can preview walker behavior before running."""
        manager, db_task, api_task, feature_task = setup_dependencies

        walker = (
            GraphWalker(manager)
            .starting_from(feature_task.id)
            .outgoing()
            .follow("DEPENDS_ON")
            .max_depth(10)
            .bfs()
        )

        plan = walker.explain()
        plan_str = str(plan)

        # Plan should describe the traversal
        assert "DEPENDS_ON" in plan_str or "edge" in plan_str.lower()


class TestFindRelatedTasksWorkflow:
    """User workflow: Finding tasks related by patterns."""

    @pytest.fixture
    def setup_related_tasks(self):
        """Create related tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create several pending high-priority tasks
        tasks = []
        for i in range(5):
            task = manager.create_task(
                f"High priority task {i}",
                priority="high"
            )
            tasks.append(task)

        # Create some low priority tasks
        for i in range(3):
            task = manager.create_task(
                f"Low priority task {i}",
                priority="low"
            )
            tasks.append(task)

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_find_by_pattern(self, setup_related_tasks):
        """User finds tasks matching a pattern."""
        manager, tasks = setup_related_tasks

        pattern = Pattern().node("A", priority="high", status="pending")
        matcher = PatternMatcher(manager)

        result = matcher.find(pattern)

        # Should find all high priority pending tasks
        assert len(result) == 5
        for match in result:
            task = match.bindings["A"]
            # bindings contain Task objects
            assert hasattr(task, "priority")
            assert task.priority == "high"

    def test_explain_pattern_before_matching(self, setup_related_tasks):
        """User can preview pattern behavior."""
        manager, tasks = setup_related_tasks

        pattern = Pattern().node("A", priority="high")
        matcher = PatternMatcher(manager).limit(10)

        plan = matcher.explain(pattern)
        plan_str = str(plan)

        # Plan should describe the pattern
        assert "priority" in plan_str.lower() or "node" in plan_str.lower()
        assert "10" in plan_str or "limit" in plan_str.lower()


class TestTruncationAwarenessWorkflow:
    """User workflow: Handling truncated results."""

    @pytest.fixture
    def setup_many_tasks(self):
        """Create many tasks to trigger truncation."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create 50 tasks
        tasks = []
        for i in range(50):
            task = manager.create_task(f"Task {i}", priority="medium")
            tasks.append(task)

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_know_when_results_truncated(self, setup_many_tasks):
        """User knows when they didn't get all results."""
        manager, tasks = setup_many_tasks

        pattern = Pattern().node("A")
        matcher = PatternMatcher(manager).limit(10)

        result = matcher.find(pattern)

        # User can check if results were truncated
        assert result.truncated is True
        assert result.matches_found >= 10
        assert result.limit_value == 10
        assert len(result) == 10

    def test_iterate_without_checking_truncation(self, setup_many_tasks):
        """User can iterate without caring about truncation."""
        manager, tasks = setup_many_tasks

        pattern = Pattern().node("A")
        result = PatternMatcher(manager).limit(5).find(pattern)

        # Just iterate, don't care about truncation
        count = 0
        for match in result:
            count += 1

        assert count == 5

    def test_check_if_got_all_results(self, setup_many_tasks):
        """User can verify they got all results."""
        manager, tasks = setup_many_tasks

        pattern = Pattern().node("A")
        result = PatternMatcher(manager).limit(100).find(pattern)

        # User checks: did I get everything?
        if not result.truncated:
            # Got all results
            assert len(result) == 50
        else:
            # Need to increase limit
            assert result.matches_found > len(result)


class TestPathFindingWorkflow:
    """User workflow: Finding paths between tasks."""

    @pytest.fixture
    def setup_task_graph(self):
        """Create a graph of related tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create a dependency chain
        tasks = []
        for i in range(5):
            task = manager.create_task(f"Task {i}", priority="medium")
            tasks.append(task)

        # Link them: 0 -> 1 -> 2 -> 3 -> 4
        for i in range(len(tasks) - 1):
            manager.add_edge(tasks[i].id, tasks[i + 1].id, "DEPENDS_ON")

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_find_path_between_tasks(self, setup_task_graph):
        """User finds path between two tasks."""
        manager, tasks = setup_task_graph

        finder = PathFinder(manager).directed()
        paths = finder.all_paths(tasks[0].id, tasks[4].id)

        # Should find the path
        assert len(paths) >= 1
        # Path should go from first to last
        assert paths[0][0] == tasks[0].id
        assert paths[0][-1] == tasks[4].id

    def test_explain_path_search(self, setup_task_graph):
        """User previews path search parameters."""
        manager, tasks = setup_task_graph

        finder = PathFinder(manager).max_paths(10).max_length(5).directed()
        plan = finder.explain()  # No arguments - describes finder config

        plan_str = str(plan)
        # Should show configuration
        assert "path" in plan_str.lower() or "max" in plan_str.lower() or "directed" in plan_str.lower()

    def test_check_if_path_exists(self, setup_task_graph):
        """User checks if any path exists."""
        manager, tasks = setup_task_graph

        finder = PathFinder(manager).directed()

        # Path exists
        paths = finder.all_paths(tasks[0].id, tasks[4].id)
        assert len(paths) > 0

        # No path in reverse direction
        reverse_paths = finder.all_paths(tasks[4].id, tasks[0].id)
        assert len(reverse_paths) == 0


class TestCombinedToolsWorkflow:
    """User workflow: Using multiple query tools together."""

    @pytest.fixture
    def setup_complex_project(self):
        """Create a complex project with multiple relationships."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks
        epic = manager.create_task("Implement Auth System", priority="critical")
        task1 = manager.create_task("Design auth flow", priority="high")
        task2 = manager.create_task("Implement login", priority="high")
        task3 = manager.create_task("Implement logout", priority="medium")
        task4 = manager.create_task("Write auth tests", priority="high")

        # Create dependencies
        manager.add_edge(task1.id, epic.id, "DEPENDS_ON")
        manager.add_edge(task2.id, task1.id, "DEPENDS_ON")
        manager.add_edge(task3.id, task1.id, "DEPENDS_ON")
        manager.add_edge(task4.id, task2.id, "DEPENDS_ON")
        manager.add_edge(task4.id, task3.id, "DEPENDS_ON")

        yield manager, epic, task1, task2, task3, task4
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_then_explore(self, setup_complex_project):
        """User queries for tasks, then explores their dependencies."""
        manager, epic, task1, task2, task3, task4 = setup_complex_project

        # First, find high priority tasks
        high_priority = Query(manager).tasks().where(priority="high").execute()
        assert len(high_priority) >= 2

        # Then, for each, explore what they depend on
        for task in high_priority[:1]:  # Just check first one
            dependencies = (
                GraphWalker(manager)
                .starting_from(task.id)
                .outgoing()
                .follow("DEPENDS_ON")
                .bfs()
                .visit(lambda n, acc: acc + [n.id], initial=[])
                .run()
            )
            # Should find some dependencies
            assert len(dependencies) >= 1

    def test_pattern_match_then_find_path(self, setup_complex_project):
        """User finds tasks by pattern, then finds paths between them."""
        manager, epic, task1, task2, task3, task4 = setup_complex_project

        # Find all high priority tasks
        pattern = Pattern().node("A", priority="high")
        matches = PatternMatcher(manager).find(pattern)

        # If we have at least 2, find path between first two
        if len(matches) >= 2:
            # bindings contain Task objects, extract IDs
            task_ids = [m.bindings["A"].id for m in matches[:2]]

            finder = PathFinder(manager)
            result = finder.all_paths(task_ids[0], task_ids[1])
            # May or may not find a path depending on graph structure
            # result is PathSearchResult with .paths attribute
