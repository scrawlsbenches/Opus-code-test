"""
Property-based tests for Query API invariants.

Tests invariants that must ALWAYS hold regardless of input:
1. PatternSearchResult.matches_found >= len(matches) when truncated
2. explain() never raises for valid input
3. Direction methods are idempotent
4. Empty input doesn't crash
5. Limit of 0 returns empty results

Uses hypothesis for property-based testing if available,
falls back to comprehensive manual testing otherwise.
"""

import pytest
from unittest.mock import MagicMock
from typing import List

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

# Try to import hypothesis, make tests optional if not available
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

    # Dummy decorators for when hypothesis isn't available
    def given(*args, **kwargs):
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)
        return decorator

    class st:
        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

    def settings(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def assume(condition):
        pass


class TestPatternSearchResultInvariants:
    """Invariants for PatternSearchResult."""

    def test_truncated_implies_matches_found_gte_len(self):
        """When truncated, matches_found >= len(matches)."""
        # Create truncated result
        matches = [PatternMatch(bindings={"A": f"task-{i}"}) for i in range(5)]
        result = PatternSearchResult(
            matches=matches,
            truncated=True,
            matches_found=100,
            limit_value=5
        )

        assert result.truncated
        assert result.matches_found >= len(result.matches)

    def test_not_truncated_matches_found_can_be_zero(self):
        """When not truncated, matches_found may be 0 (not counted)."""
        matches = [PatternMatch(bindings={"A": "task-1"})]
        result = PatternSearchResult(
            matches=matches,
            truncated=False,
            matches_found=0  # Didn't bother counting
        )

        assert not result.truncated
        # matches_found=0 is valid when not truncated

    def test_empty_result_is_falsy(self):
        """Empty PatternSearchResult is always falsy."""
        result = PatternSearchResult(matches=[])
        assert not result
        assert bool(result) is False

    def test_nonempty_result_is_truthy(self):
        """Non-empty PatternSearchResult is always truthy."""
        matches = [PatternMatch(bindings={"A": "task-1"})]
        result = PatternSearchResult(matches=matches)
        assert result
        assert bool(result) is True

    def test_len_matches_iteration_count(self):
        """len(result) always equals iteration count."""
        for count in [0, 1, 5, 10, 100]:
            matches = [
                PatternMatch(bindings={"A": f"task-{i}"})
                for i in range(count)
            ]
            result = PatternSearchResult(matches=matches)

            assert len(result) == count
            assert len(list(result)) == count


class TestExplainNeverRaisesInvariant:
    """explain() should never raise for valid input."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_explain_empty_graph(self, mock_manager):
        """Query.explain() works on empty graph."""
        query = Query(mock_manager).tasks()
        plan = query.explain()  # Should not raise
        assert plan is not None

    def test_graph_walker_explain_no_start(self, mock_manager):
        """GraphWalker.explain() works without starting node."""
        walker = GraphWalker(mock_manager).bfs()
        plan = walker.explain()  # Should not raise
        assert plan is not None

    def test_path_finder_explain_works(self, mock_manager):
        """PathFinder.explain() works."""
        finder = PathFinder(mock_manager)
        plan = finder.explain()  # No arguments - describes finder config
        assert plan is not None

    def test_pattern_matcher_explain_empty_pattern(self, mock_manager):
        """PatternMatcher.explain() works with minimal pattern."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)
        plan = matcher.explain(pattern)  # Should not raise
        assert plan is not None


class TestDirectionIdempotenceInvariant:
    """Direction methods should be idempotent."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_graph_walker_outgoing_idempotent(self, mock_manager):
        """outgoing().outgoing() equals outgoing()."""
        walker1 = GraphWalker(mock_manager).outgoing()
        walker2 = GraphWalker(mock_manager).outgoing().outgoing()

        # Both should have same direction state
        assert walker1._bidirectional == walker2._bidirectional
        assert walker1._reverse_direction == walker2._reverse_direction

    def test_graph_walker_incoming_idempotent(self, mock_manager):
        """incoming().incoming() equals incoming()."""
        walker1 = GraphWalker(mock_manager).incoming()
        walker2 = GraphWalker(mock_manager).incoming().incoming()

        assert walker1._bidirectional == walker2._bidirectional
        assert walker1._reverse_direction == walker2._reverse_direction

    def test_graph_walker_both_idempotent(self, mock_manager):
        """both().both() equals both()."""
        walker1 = GraphWalker(mock_manager).both()
        walker2 = GraphWalker(mock_manager).both().both()

        assert walker1._bidirectional == walker2._bidirectional
        assert walker1._reverse_direction == walker2._reverse_direction

    def test_graph_walker_both_resets_direction(self, mock_manager):
        """both() after incoming() resets to bidirectional."""
        walker = GraphWalker(mock_manager).incoming().both()

        assert walker._bidirectional is True
        assert walker._reverse_direction is False


class TestEmptyInputInvariant:
    """Empty input should never crash, always return empty/default."""

    @pytest.fixture
    def mock_manager_empty(self):
        """Create empty mock manager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_empty_graph(self, mock_manager_empty):
        """Query on empty graph returns empty list."""
        results = Query(mock_manager_empty).tasks().execute()
        assert results == []

    def test_graph_walker_empty_graph(self, mock_manager_empty):
        """GraphWalker on empty graph returns initial value."""
        result = (
            GraphWalker(mock_manager_empty)
            .starting_from("nonexistent")
            .bfs()
            .visit(lambda n, acc: acc + 1, initial=0)
            .run()
        )
        assert result == 0

    def test_path_finder_empty_graph(self, mock_manager_empty):
        """PathFinder on empty graph returns empty PathSearchResult."""
        result = PathFinder(mock_manager_empty).all_paths("a", "b")
        # Returns PathSearchResult with empty paths
        assert hasattr(result, "paths")
        assert result.paths == []

    def test_pattern_matcher_empty_graph(self, mock_manager_empty):
        """PatternMatcher on empty graph returns empty result."""
        pattern = Pattern().node("A")
        result = PatternMatcher(mock_manager_empty).find(pattern)
        assert len(result) == 0
        assert not result


class TestLimitZeroInvariant:
    """Limit of 0 should return empty results."""

    @pytest.fixture
    def mock_manager_with_data(self):
        """Create mock manager with data."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
            for i in range(10)
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_limit_zero(self, mock_manager_with_data):
        """Query.limit(0) returns empty list."""
        results = Query(mock_manager_with_data).tasks().limit(0).execute()
        assert results == []

    def test_pattern_matcher_limit_works(self, mock_manager_with_data):
        """PatternMatcher.limit() restricts results."""
        pattern = Pattern().node("A")
        result = PatternMatcher(mock_manager_with_data).limit(5).find(pattern)
        # Should have at most 5 results
        assert len(result) <= 5


class TestLimitPositiveInvariant:
    """Positive limits should respect the limit."""

    @pytest.fixture
    def mock_manager_with_many_tasks(self):
        """Create mock manager with many tasks."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
            for i in range(100)
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    @pytest.mark.parametrize("limit", [1, 5, 10, 50])
    def test_query_respects_limit(self, mock_manager_with_many_tasks, limit):
        """Query.limit(n) returns at most n results."""
        results = Query(mock_manager_with_many_tasks).tasks().limit(limit).execute()
        assert len(results) <= limit

    @pytest.mark.parametrize("limit", [1, 5, 10, 50])
    def test_pattern_matcher_respects_limit(self, mock_manager_with_many_tasks, limit):
        """PatternMatcher.limit(n) returns at most n matches."""
        pattern = Pattern().node("A")
        result = PatternMatcher(mock_manager_with_many_tasks).limit(limit).find(pattern)
        assert len(result) <= limit


# Hypothesis-based property tests (only run if hypothesis is available)
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestHypothesisProperties:
    """Property-based tests using hypothesis."""

    @given(st.integers(min_value=0, max_value=1000))
    @settings(max_examples=50)
    def test_pattern_search_result_len_equals_matches_count(self, count):
        """len(result) always equals number of matches for any count."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"})
            for i in range(count)
        ]
        result = PatternSearchResult(matches=matches)

        assert len(result) == count
        assert len(list(result)) == count

    @given(st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_indexing_in_bounds_never_raises(self, count):
        """Indexing within bounds never raises."""
        assume(count > 0)  # Skip empty case

        matches = [
            PatternMatch(bindings={"A": f"task-{i}"})
            for i in range(count)
        ]
        result = PatternSearchResult(matches=matches)

        # All valid indices should work
        for i in range(count):
            _ = result[i]  # Should not raise

        # Negative indices should work
        _ = result[-1]  # Should not raise


class TestOrderingInvariant:
    """Query ordering should be stable and predictable."""

    @pytest.fixture
    def mock_manager_with_tasks(self):
        """Create mock manager with varied tasks."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id="task-3", title="C", status="pending", priority="low"),
            Task(id="task-1", title="A", status="pending", priority="high"),
            Task(id="task-2", title="B", status="pending", priority="medium"),
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_order_by_is_deterministic(self, mock_manager_with_tasks):
        """Same query with order_by returns same order."""
        query = Query(mock_manager_with_tasks).tasks().order_by("title")

        results1 = query.execute()
        results2 = query.execute()

        assert [t.id for t in results1] == [t.id for t in results2]

    def test_query_order_by_actually_orders(self, mock_manager_with_tasks):
        """order_by() actually changes the order."""
        results = Query(mock_manager_with_tasks).tasks().order_by("title").execute()
        titles = [t.title for t in results]

        assert titles == sorted(titles)
