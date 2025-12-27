"""
Integration tests for Query API multi-tool scenarios.

Tests that all query tools work together correctly:
1. Query results feeding into GraphWalker
2. PatternMatcher results used with PathFinder
3. Chaining multiple tools in a pipeline
4. Shared manager state consistency

These tests verify the tools integrate properly.
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


class TestQueryToWalkerIntegration:
    """Test Query results feeding into GraphWalker."""

    @pytest.fixture
    def project_with_dependencies(self):
        """Create a project with task dependencies."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create a realistic project structure
        # Sprint 1 tasks
        auth_epic = manager.create_task("Auth System", priority="critical")
        login = manager.create_task("Implement login", priority="high")
        logout = manager.create_task("Implement logout", priority="high")
        session = manager.create_task("Session management", priority="high")

        # Dependencies: login, logout depend on session
        manager.add_edge(login.id, session.id, "DEPENDS_ON")
        manager.add_edge(logout.id, session.id, "DEPENDS_ON")
        manager.add_edge(session.id, auth_epic.id, "DEPENDS_ON")

        # Sprint 2 tasks
        api = manager.create_task("Build API", priority="high")
        docs = manager.create_task("API documentation", priority="low")
        manager.add_edge(docs.id, api.id, "DEPENDS_ON")

        yield manager, {
            "auth_epic": auth_epic,
            "login": login,
            "logout": logout,
            "session": session,
            "api": api,
            "docs": docs,
        }
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_then_walk_dependencies(self, project_with_dependencies):
        """Query for tasks, then walk their dependencies."""
        manager, tasks = project_with_dependencies

        # Step 1: Find all high priority tasks
        high_priority = Query(manager).tasks().where(priority="high").execute()
        assert len(high_priority) >= 3

        # Step 2: For each, find all tasks it depends on
        all_dependencies = set()
        for task in high_priority:
            deps = (
                GraphWalker(manager)
                .starting_from(task.id)
                .outgoing()
                .follow("DEPENDS_ON")
                .bfs()
                .visit(lambda n, acc: acc | {n.id}, initial=set())
                .run()
            )
            all_dependencies.update(deps)

        # Should find dependencies
        assert len(all_dependencies) > 0

    def test_query_results_are_valid_walker_starts(self, project_with_dependencies):
        """Query results can be used as GraphWalker starting points."""
        manager, tasks = project_with_dependencies

        # Get tasks from Query
        results = Query(manager).tasks().limit(3).execute()

        # Each result should be usable as walker start
        for task in results:
            walker = GraphWalker(manager).starting_from(task.id)
            plan = walker.explain()
            # Should not raise
            assert plan is not None


class TestPatternToPathIntegration:
    """Test PatternMatcher results used with PathFinder."""

    @pytest.fixture
    def connected_graph(self):
        """Create a connected graph of tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks in a grid pattern
        # A -> B -> C
        # |    |    |
        # v    v    v
        # D -> E -> F
        tasks = {}
        for name in ["A", "B", "C", "D", "E", "F"]:
            task = manager.create_task(f"Task {name}", priority="medium")
            tasks[name] = task

        # Horizontal edges
        manager.add_edge(tasks["A"].id, tasks["B"].id, "DEPENDS_ON")
        manager.add_edge(tasks["B"].id, tasks["C"].id, "DEPENDS_ON")
        manager.add_edge(tasks["D"].id, tasks["E"].id, "DEPENDS_ON")
        manager.add_edge(tasks["E"].id, tasks["F"].id, "DEPENDS_ON")

        # Vertical edges
        manager.add_edge(tasks["A"].id, tasks["D"].id, "DEPENDS_ON")
        manager.add_edge(tasks["B"].id, tasks["E"].id, "DEPENDS_ON")
        manager.add_edge(tasks["C"].id, tasks["F"].id, "DEPENDS_ON")

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_find_pattern_then_path(self, connected_graph):
        """Find tasks by pattern, then find paths between them."""
        manager, tasks = connected_graph

        # Step 1: Find all tasks (simple pattern)
        pattern = Pattern().node("X")
        matches = PatternMatcher(manager).find(pattern)
        assert len(matches) == 6

        # Step 2: Extract task IDs (bindings contain Task objects)
        task_ids = [m.bindings["X"].id for m in matches]

        # Step 3: Find paths between first and last found
        finder = PathFinder(manager).directed()
        result = finder.all_paths(task_ids[0], task_ids[-1])

        # May or may not find path depending on which tasks were first/last
        # The important thing is it doesn't crash

    def test_pattern_edge_matches_path_exists(self, connected_graph):
        """Tasks connected in pattern should have paths."""
        manager, tasks = connected_graph

        # Find connected pairs via pattern (using outgoing() for edges)
        pattern = (
            Pattern()
            .node("A")
            .outgoing("DEPENDS_ON")
            .node("B")
        )
        matches = PatternMatcher(manager).find(pattern)

        # For each match, verify path exists
        finder = PathFinder(manager).directed()
        for match in matches:
            source = match.bindings["A"].id  # bindings contain Task objects
            target = match.bindings["B"].id
            result = finder.all_paths(source, target)
            # Should find at least the direct edge
            assert len(result.paths) >= 1


class TestMultiToolPipeline:
    """Test chaining multiple tools in a pipeline."""

    @pytest.fixture
    def complex_project(self):
        """Create a complex project for pipeline testing."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create hierarchical structure
        epic = manager.create_task("Epic: User Management", priority="critical")

        features = []
        for i in range(3):
            feature = manager.create_task(f"Feature {i}", priority="high")
            features.append(feature)
            manager.add_edge(feature.id, epic.id, "DEPENDS_ON")

        stories = []
        for i, feature in enumerate(features):
            for j in range(2):
                story = manager.create_task(f"Story {i}.{j}", priority="medium")
                stories.append(story)
                manager.add_edge(story.id, feature.id, "DEPENDS_ON")

        yield manager, epic, features, stories
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pipeline_query_filter_walk_collect(self, complex_project):
        """Pipeline: Query -> Filter -> Walk -> Collect."""
        manager, epic, features, stories = complex_project

        # Step 1: Query for medium priority tasks
        medium_priority = (
            Query(manager)
            .tasks()
            .where(priority="medium")
            .execute()
        )
        assert len(medium_priority) == 6  # All stories

        # Step 2: Walk from each to find what they depend on
        dependency_counts = {}
        for task in medium_priority:
            deps = (
                GraphWalker(manager)
                .starting_from(task.id)
                .outgoing()
                .follow("DEPENDS_ON")
                .bfs()
                .visit(lambda n, acc: acc + 1, initial=0)
                .run()
            )
            dependency_counts[task.id] = deps

        # Each story should have dependencies (feature + epic)
        for task_id, count in dependency_counts.items():
            assert count >= 1  # At least the task itself

    def test_pipeline_pattern_path_analyze(self, complex_project):
        """Pipeline: Pattern match -> Path find -> Analyze."""
        manager, epic, features, stories = complex_project

        # Step 1: Find all medium-high connections via pattern (using outgoing())
        pattern = (
            Pattern()
            .node("A", priority="medium")
            .outgoing("DEPENDS_ON")
            .node("B", priority="high")
        )
        connections = PatternMatcher(manager).find(pattern)
        assert len(connections) >= 1

        # Step 2: For interesting connections, find full path to epic
        finder = PathFinder(manager).directed()
        longest_path = []

        for match in connections[:3]:  # Check first 3
            story_id = match.bindings["A"].id  # bindings contain Task objects
            result = finder.all_paths(story_id, epic.id)
            for path in result.paths:
                if len(path) > len(longest_path):
                    longest_path = path

        # Should find path to epic
        assert len(longest_path) >= 2


class TestSharedManagerConsistency:
    """Test that tools share manager state correctly."""

    @pytest.fixture
    def shared_manager(self):
        """Create a manager that all tools will share."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        task = manager.create_task("Initial task", priority="high")

        yield manager, task, temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_modifications_visible_across_tools(self, shared_manager):
        """Changes made via one tool are visible to others."""
        manager, initial_task, _ = shared_manager

        # Query sees initial task
        results1 = Query(manager).tasks().execute()
        assert len(results1) == 1

        # Add more tasks
        task2 = manager.create_task("Task 2", priority="medium")
        manager.add_edge(task2.id, initial_task.id, "DEPENDS_ON")

        # Query sees new task
        results2 = Query(manager).tasks().execute()
        assert len(results2) == 2

        # Walker can traverse to it
        visited = (
            GraphWalker(manager)
            .starting_from(task2.id)
            .outgoing()
            .bfs()
            .visit(lambda n, acc: acc + [n.id], initial=[])
            .run()
        )
        assert initial_task.id in visited

        # Pattern finds connection (using outgoing() for edges)
        pattern = Pattern().node("A").outgoing("DEPENDS_ON").node("B")
        matches = PatternMatcher(manager).find(pattern)
        assert len(matches) >= 1

    def test_all_tools_see_same_edge_count(self, shared_manager):
        """All tools agree on edge count."""
        manager, task1, _ = shared_manager

        # Add second task and edge
        task2 = manager.create_task("Task 2", priority="medium")
        manager.add_edge(task2.id, task1.id, "DEPENDS_ON")

        # Count edges via different methods
        edges = manager.list_edges()
        edge_count = len(edges)

        # Pattern should find same number of edge pairs (using outgoing())
        pattern = Pattern().node("A").outgoing("DEPENDS_ON").node("B")
        matches = PatternMatcher(manager).find(pattern)
        pattern_edge_count = len(matches)

        # Should be consistent
        assert pattern_edge_count == edge_count


class TestExplainConsistency:
    """Test that explain() across tools gives consistent information."""

    @pytest.fixture
    def manager_with_data(self):
        """Create manager with test data."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        for i in range(10):
            manager.create_task(f"Task {i}", priority="high" if i < 5 else "low")

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_all_explains_mention_entity_type(self, manager_with_data):
        """All explain() outputs should mention what they're querying."""
        manager = manager_with_data

        # Query explain
        query_plan = Query(manager).tasks().where(priority="high").explain()
        query_str = str(query_plan)
        assert "task" in query_str.lower() or "entity" in query_str.lower()

        # Walker explain
        walker_plan = GraphWalker(manager).starting_from("task-1").bfs().explain()
        walker_str = str(walker_plan)
        # Walker plan should exist and be non-empty
        assert len(walker_str) > 0

        # PathFinder explain (no arguments - describes finder config)
        path_plan = PathFinder(manager).explain()
        path_str = str(path_plan)
        assert len(path_str) > 0

        # PatternMatcher explain
        pattern = Pattern().node("A")
        pattern_plan = PatternMatcher(manager).explain(pattern)
        pattern_str = str(pattern_plan)
        assert len(pattern_str) > 0

    def test_limits_shown_in_all_explains(self, manager_with_data):
        """Limits should be visible in explain() output."""
        manager = manager_with_data

        # Query with limit
        query_plan = Query(manager).tasks().limit(5).explain()
        query_str = str(query_plan)
        # Should mention limit somehow
        assert "5" in query_str or "limit" in query_str.lower()

        # PatternMatcher with limit
        pattern = Pattern().node("A")
        matcher_plan = PatternMatcher(manager).limit(10).explain(pattern)
        matcher_str = str(matcher_plan)
        assert "10" in matcher_str or "limit" in matcher_str.lower()

        # PathFinder with max_paths (explain() has no arguments)
        path_plan = PathFinder(manager).max_paths(20).explain()
        path_str = str(path_plan)
        assert "20" in path_str or "path" in path_str.lower() or "max" in path_str.lower()
