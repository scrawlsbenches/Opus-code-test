"""
Contract tests for Query API consistency.

Tests that query classes honor consistent API contracts.

Note: Direction methods are handled differently by each class:
- GraphWalker: outgoing()/incoming()/both() on the Walker (global setting)
- PatternMatcher: outgoing()/incoming()/both() on the Pattern (per-edge)
- PathFinder: directed() on the Finder

Contracts tested:
1. All classes have explain() returning a Plan object
2. Plan objects can be stringified without error
3. Direction methods exist where appropriate
4. Fluent API returns self for chaining
5. Result types support iteration

These tests prevent API drift between classes.
"""

import pytest
from unittest.mock import MagicMock
from typing import Any

from cortical.got.query_builder import Query, QueryPlan
from cortical.got.graph_walker import GraphWalker, WalkerPlan
from cortical.got.path_finder import PathFinder, PathPlan, PathSearchResult
from cortical.got.pattern_matcher import (
    Pattern,
    PatternMatcher,
    PatternPlan,
    PatternSearchResult,
    PatternMatch,
)
from cortical.got.types import Task, Edge


class TestExplainContract:
    """All query classes must have explain() that returns a Plan."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager for all tests."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id="task-1", title="Task 1", status="pending", priority="high"),
            Task(id="task-2", title="Task 2", status="in_progress", priority="medium"),
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = [
            Edge(id="edge-1", source_id="task-1", target_id="task-2", edge_type="DEPENDS_ON"),
        ]
        return manager

    def test_query_has_explain(self, mock_manager):
        """Query has explain() method."""
        query = Query(mock_manager).tasks()
        assert hasattr(query, "explain")

    def test_graph_walker_has_explain(self, mock_manager):
        """GraphWalker has explain() method."""
        walker = GraphWalker(mock_manager)
        assert hasattr(walker, "explain")

    def test_path_finder_has_explain(self, mock_manager):
        """PathFinder has explain() method."""
        finder = PathFinder(mock_manager)
        assert hasattr(finder, "explain")

    def test_pattern_matcher_has_explain(self, mock_manager):
        """PatternMatcher has explain() method."""
        matcher = PatternMatcher(mock_manager)
        assert hasattr(matcher, "explain")

    def test_query_explain_returns_plan(self, mock_manager):
        """Query.explain() returns QueryPlan."""
        query = Query(mock_manager).tasks().where(status="pending")
        plan = query.explain()
        assert isinstance(plan, QueryPlan)

    def test_graph_walker_explain_returns_plan(self, mock_manager):
        """GraphWalker.explain() returns WalkerPlan."""
        walker = GraphWalker(mock_manager).starting_from("task-1").bfs()
        plan = walker.explain()
        assert isinstance(plan, WalkerPlan)

    def test_path_finder_explain_returns_plan(self, mock_manager):
        """PathFinder.explain() returns PathPlan."""
        finder = PathFinder(mock_manager)
        plan = finder.explain()  # No arguments - plan is for the finder config
        assert isinstance(plan, PathPlan)

    def test_pattern_matcher_explain_returns_plan(self, mock_manager):
        """PatternMatcher.explain() returns PatternPlan."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)
        plan = matcher.explain(pattern)
        assert isinstance(plan, PatternPlan)


class TestPlanStringContract:
    """All Plan objects must be safely convertible to string."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id="task-1", title="Task 1", status="pending", priority="high"),
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_plan_str(self, mock_manager):
        """QueryPlan.__str__() doesn't raise."""
        query = Query(mock_manager).tasks().where(status="pending")
        plan = query.explain()

        # Should not raise
        output = str(plan)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_walker_plan_str(self, mock_manager):
        """WalkerPlan.__str__() doesn't raise."""
        walker = GraphWalker(mock_manager).starting_from("task-1").bfs()
        plan = walker.explain()

        output = str(plan)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_path_plan_str(self, mock_manager):
        """PathPlan.__str__() doesn't raise."""
        finder = PathFinder(mock_manager)
        plan = finder.explain()

        output = str(plan)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_pattern_plan_str(self, mock_manager):
        """PatternPlan.__str__() doesn't raise."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager)
        plan = matcher.explain(pattern)

        output = str(plan)
        assert isinstance(output, str)
        assert len(output) > 0


class TestDirectionMethodsContract:
    """Direction methods should exist where appropriate."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id="task-1", title="Task 1", status="pending", priority="high"),
            Task(id="task-2", title="Task 2", status="pending", priority="medium"),
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = [
            Edge(id="edge-1", source_id="task-1", target_id="task-2", edge_type="DEPENDS_ON"),
        ]
        return manager

    # GraphWalker direction methods
    def test_graph_walker_has_outgoing(self, mock_manager):
        """GraphWalker has outgoing() method."""
        walker = GraphWalker(mock_manager)
        assert hasattr(walker, "outgoing")

    def test_graph_walker_has_incoming(self, mock_manager):
        """GraphWalker has incoming() method."""
        walker = GraphWalker(mock_manager)
        assert hasattr(walker, "incoming")

    def test_graph_walker_has_both(self, mock_manager):
        """GraphWalker has both() method."""
        walker = GraphWalker(mock_manager)
        assert hasattr(walker, "both")

    def test_graph_walker_outgoing_returns_self(self, mock_manager):
        """GraphWalker.outgoing() returns self for chaining."""
        walker = GraphWalker(mock_manager)
        result = walker.outgoing()
        assert result is walker

    def test_graph_walker_incoming_returns_self(self, mock_manager):
        """GraphWalker.incoming() returns self for chaining."""
        walker = GraphWalker(mock_manager)
        result = walker.incoming()
        assert result is walker

    def test_graph_walker_both_returns_self(self, mock_manager):
        """GraphWalker.both() returns self for chaining."""
        walker = GraphWalker(mock_manager)
        result = walker.both()
        assert result is walker

    # Pattern direction methods (on Pattern, not PatternMatcher)
    def test_pattern_has_outgoing(self):
        """Pattern has outgoing() method."""
        pattern = Pattern().node("A")
        assert hasattr(pattern, "outgoing")

    def test_pattern_has_incoming(self):
        """Pattern has incoming() method."""
        pattern = Pattern().node("A")
        assert hasattr(pattern, "incoming")

    def test_pattern_has_both(self):
        """Pattern has both() method."""
        pattern = Pattern().node("A")
        assert hasattr(pattern, "both")

    def test_pattern_outgoing_returns_pattern(self):
        """Pattern.outgoing() returns Pattern for chaining."""
        pattern = Pattern().node("A")
        result = pattern.outgoing("DEPENDS_ON")
        assert isinstance(result, Pattern)

    def test_pattern_incoming_returns_pattern(self):
        """Pattern.incoming() returns Pattern for chaining."""
        pattern = Pattern().node("A")
        result = pattern.incoming("DEPENDS_ON")
        assert isinstance(result, Pattern)

    # PathFinder direction methods
    def test_path_finder_has_directed(self, mock_manager):
        """PathFinder has directed() method."""
        finder = PathFinder(mock_manager)
        assert hasattr(finder, "directed")

    def test_path_finder_directed_returns_self(self, mock_manager):
        """PathFinder.directed() returns self for chaining."""
        finder = PathFinder(mock_manager)
        result = finder.directed()
        assert result is finder


class TestFluentAPIContract:
    """All query classes must support fluent API (return self)."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_fluent_chain(self, mock_manager):
        """Query methods can be chained."""
        query = (
            Query(mock_manager)
            .tasks()
            .where(status="pending")
            .order_by("priority")
            .limit(10)
        )
        # Should not raise, and query is usable
        assert query is not None

    def test_graph_walker_fluent_chain(self, mock_manager):
        """GraphWalker methods can be chained."""
        walker = (
            GraphWalker(mock_manager)
            .starting_from("task-1")
            .bfs()
            .max_depth(5)
            .outgoing()
            .filter(lambda n: True)
        )
        assert walker is not None

    def test_path_finder_fluent_chain(self, mock_manager):
        """PathFinder methods can be chained."""
        finder = (
            PathFinder(mock_manager)
            .max_paths(10)
            .max_length(5)
            .directed()
        )
        assert finder is not None

    def test_pattern_matcher_fluent_chain(self, mock_manager):
        """PatternMatcher methods can be chained."""
        matcher = (
            PatternMatcher(mock_manager)
            .limit(10)
        )
        assert matcher is not None

    def test_pattern_fluent_chain(self):
        """Pattern methods can be chained."""
        pattern = (
            Pattern()
            .node("A", status="pending")
            .outgoing("DEPENDS_ON")
            .node("B", priority="high")
        )
        assert pattern is not None


class TestResultIterationContract:
    """Result types must support iteration."""

    def test_pattern_search_result_iterable(self):
        """PatternSearchResult is iterable."""
        matches = [PatternMatch(bindings={"A": "task-1"})]
        result = PatternSearchResult(matches=matches)

        # Should be iterable
        collected = list(result)
        assert len(collected) == 1

    def test_path_search_result_has_paths(self):
        """PathSearchResult has paths attribute."""
        result = PathSearchResult(
            paths=[["task-1", "task-2"]],
            truncated=False,
            paths_found=1,
        )

        # paths should be iterable
        assert hasattr(result, "paths")
        collected = list(result.paths)
        assert len(collected) == 1


class TestLimitContract:
    """Limit functionality should work consistently."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager with multiple tasks."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
            for i in range(20)
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_limit(self, mock_manager):
        """Query.limit() restricts results."""
        query = Query(mock_manager).tasks().limit(5)
        results = query.execute()
        assert len(results) <= 5

    def test_pattern_matcher_limit(self, mock_manager):
        """PatternMatcher.limit() restricts results."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager).limit(5)
        results = matcher.find(pattern)
        assert len(results) <= 5

    def test_path_finder_max_paths(self, mock_manager):
        """PathFinder.max_paths() restricts results."""
        finder = PathFinder(mock_manager).max_paths(5)
        # Method exists and is chainable
        assert finder is not None


class TestTruncationMetadataContract:
    """Results should report truncation consistently."""

    def test_pattern_search_result_has_truncation(self):
        """PatternSearchResult has truncation fields."""
        result = PatternSearchResult(
            matches=[],
            truncated=True,
            matches_found=100,
            limit_value=10,
        )

        assert hasattr(result, "truncated")
        assert hasattr(result, "matches_found")
        assert hasattr(result, "limit_value")

    def test_path_search_result_has_truncation(self):
        """PathSearchResult has truncation fields."""
        result = PathSearchResult(
            paths=[],
            truncated=True,
            paths_found=50,
        )

        assert hasattr(result, "truncated")
        assert hasattr(result, "paths_found")
