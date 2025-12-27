"""
Tests for PatternMatcher new features.

This module tests:
- PatternSearchResult with truncation metadata
- PatternPlan and explain() introspection
- Backwards compatibility (iteration, indexing, bool)

All tests use mocked GoTManager to avoid creating real data.
"""

import pytest
from unittest.mock import MagicMock
from typing import List

from cortical.got.pattern_matcher import (
    Pattern,
    PatternMatcher,
    PatternMatch,
    PatternSearchResult,
    PatternPlan,
)
from cortical.got.types import Task, Edge


class TestPatternSearchResult:
    """Test PatternSearchResult dataclass and its backwards-compatible interface."""

    def test_basic_creation(self):
        """PatternSearchResult can be created with matches."""
        matches = [
            PatternMatch(bindings={"A": "task-1"}),
            PatternMatch(bindings={"A": "task-2"}),
        ]
        result = PatternSearchResult(matches=matches)

        assert len(result.matches) == 2
        assert result.truncated is False
        assert result.matches_found == 0
        assert result.limit_value is None

    def test_truncation_metadata(self):
        """PatternSearchResult tracks truncation."""
        matches = [PatternMatch(bindings={"A": "task-1"})]
        result = PatternSearchResult(
            matches=matches,
            truncated=True,
            matches_found=100,
            limit_value=1
        )

        assert result.truncated is True
        assert result.matches_found == 100
        assert result.limit_value == 1

    def test_iteration(self):
        """PatternSearchResult is iterable like a list."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"})
            for i in range(3)
        ]
        result = PatternSearchResult(matches=matches)

        collected = list(result)
        assert len(collected) == 3
        assert collected[0].bindings["A"] == "task-0"

    def test_len(self):
        """len() returns number of matches."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"})
            for i in range(5)
        ]
        result = PatternSearchResult(matches=matches)

        assert len(result) == 5

    def test_bool_true(self):
        """bool() is True when matches exist."""
        matches = [PatternMatch(bindings={"A": "task-1"})]
        result = PatternSearchResult(matches=matches)

        assert bool(result) is True
        assert result  # Truthy

    def test_bool_false(self):
        """bool() is False when no matches."""
        result = PatternSearchResult(matches=[])

        assert bool(result) is False
        assert not result  # Falsy

    def test_indexing(self):
        """Result supports index access like a list."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"})
            for i in range(3)
        ]
        result = PatternSearchResult(matches=matches)

        assert result[0].bindings["A"] == "task-0"
        assert result[1].bindings["A"] == "task-1"
        assert result[-1].bindings["A"] == "task-2"

    def test_slicing(self):
        """Result supports slicing like a list."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"})
            for i in range(5)
        ]
        result = PatternSearchResult(matches=matches)

        sliced = result[1:3]
        assert len(sliced) == 2
        assert sliced[0].bindings["A"] == "task-1"


class TestPatternPlan:
    """Test PatternPlan introspection."""

    def test_basic_creation(self):
        """PatternPlan can be created with pattern info."""
        plan = PatternPlan(
            pattern_nodes=2,
            pattern_edges=1,
            node_constraints=["A: status=pending"],
            edge_constraints=["A->B: DEPENDS_ON"],
        )

        assert plan.pattern_nodes == 2
        assert plan.pattern_edges == 1
        assert len(plan.node_constraints) == 1
        assert len(plan.edge_constraints) == 1

    def test_with_limit(self):
        """PatternPlan tracks limit."""
        plan = PatternPlan(
            pattern_nodes=1,
            pattern_edges=0,
            node_constraints=[],
            edge_constraints=[],
            limit=10
        )

        assert plan.limit == 10

    def test_with_estimation(self):
        """PatternPlan includes graph size estimation."""
        plan = PatternPlan(
            pattern_nodes=2,
            pattern_edges=1,
            node_constraints=[],
            edge_constraints=[],
            estimated_graph_nodes=1000,
            estimated_complexity="O(n^2)"
        )

        assert plan.estimated_graph_nodes == 1000
        assert plan.estimated_complexity == "O(n^2)"

    def test_str_output(self):
        """PatternPlan has human-readable string representation."""
        plan = PatternPlan(
            pattern_nodes=2,
            pattern_edges=1,
            node_constraints=["A: status=pending"],
            edge_constraints=["A->B: DEPENDS_ON"],
            limit=5,
            estimated_graph_nodes=100
        )

        output = str(plan)
        assert "Pattern:" in output
        assert "2 nodes" in output
        assert "1 edges" in output
        assert "Node constraints" in output
        assert "Edge constraints" in output
        assert "Limit: 5" in output


class TestPatternMatcherExplain:
    """Test PatternMatcher.explain() method."""

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

    def test_explain_simple_pattern(self, mock_manager):
        """explain() returns PatternPlan for simple pattern."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)

        plan = matcher.explain(pattern)

        assert isinstance(plan, PatternPlan)
        assert plan.pattern_nodes == 1
        assert plan.pattern_edges == 0

    def test_explain_pattern_with_edge(self, mock_manager):
        """explain() includes edge constraints."""
        pattern = Pattern().node("A").outgoing("DEPENDS_ON").node("B")
        matcher = PatternMatcher(mock_manager)

        plan = matcher.explain(pattern)

        assert plan.pattern_nodes == 2
        assert plan.pattern_edges == 1
        assert any("DEPENDS_ON" in c for c in plan.edge_constraints)

    def test_explain_pattern_with_node_constraint(self, mock_manager):
        """explain() includes node constraints."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager)

        plan = matcher.explain(pattern)

        assert any("status" in c for c in plan.node_constraints)

    def test_explain_with_limit(self, mock_manager):
        """explain() includes limit from matcher."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager).limit(10)

        plan = matcher.explain(pattern)

        assert plan.limit == 10

    def test_explain_estimates_graph_size(self, mock_manager):
        """explain() estimates graph size."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)

        plan = matcher.explain(pattern)

        # Should count tasks from mock
        assert plan.estimated_graph_nodes == 2


class TestPatternMatcherFind:
    """Test PatternMatcher.find() returns PatternSearchResult."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager with test data."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = [
            Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
            for i in range(10)
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        # Create chain: task-0 -> task-1 -> ... -> task-9
        edges = [
            Edge(id=f"edge-{i}", source_id=f"task-{i}", target_id=f"task-{i+1}", edge_type="DEPENDS_ON")
            for i in range(9)
        ]
        manager.list_edges.return_value = edges
        return manager

    def test_find_returns_pattern_search_result(self, mock_manager):
        """find() returns PatternSearchResult instead of list."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager)

        result = matcher.find(pattern)

        assert isinstance(result, PatternSearchResult)

    def test_find_result_is_iterable(self, mock_manager):
        """find() result can be iterated."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager)

        result = matcher.find(pattern)
        matches = list(result)

        assert len(matches) == 10  # All 10 tasks match

    def test_find_with_limit_tracks_truncation(self, mock_manager):
        """find() tracks truncation when limit applied."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager).limit(3)

        result = matcher.find(pattern)

        assert len(result) == 3
        assert result.truncated is True
        assert result.matches_found >= 3
        assert result.limit_value == 3

    def test_find_without_truncation(self, mock_manager):
        """find() shows truncated=False when all results returned."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager).limit(100)  # Higher than total

        result = matcher.find(pattern)

        assert len(result) == 10
        assert result.truncated is False

    def test_find_first_still_works(self, mock_manager):
        """find_first() still returns single match."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager)

        match = matcher.find_first(pattern)

        assert isinstance(match, PatternMatch)
        assert "A" in match.bindings

    def test_count_still_works(self, mock_manager):
        """count() still returns integer count."""
        pattern = Pattern().node("A", status="pending")
        matcher = PatternMatcher(mock_manager)

        count = matcher.count(pattern)

        assert isinstance(count, int)
        assert count == 10


class TestPatternDirectionMethods:
    """Test direction methods on Pattern (not PatternMatcher)."""

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


class TestPatternMatcherEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_manager_empty(self):
        """Create empty mock manager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_find_no_matches(self, mock_manager_empty):
        """find() returns empty result when no matches."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager_empty)

        result = matcher.find(pattern)

        assert isinstance(result, PatternSearchResult)
        assert len(result) == 0
        assert result.truncated is False
        assert not result  # Falsy

    def test_find_first_no_matches(self, mock_manager_empty):
        """find_first() returns None when no matches."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager_empty)

        match = matcher.find_first(pattern)

        assert match is None

    def test_count_no_matches(self, mock_manager_empty):
        """count() returns 0 when no matches."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager_empty)

        count = matcher.count(pattern)

        assert count == 0

    def test_explain_empty_graph(self, mock_manager_empty):
        """explain() works on empty graph."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager_empty)

        plan = matcher.explain(pattern)

        assert isinstance(plan, PatternPlan)
        assert plan.estimated_graph_nodes == 0


class TestPatternMatcherBackwardsCompatibility:
    """Ensure backwards compatibility with existing code."""

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
        manager.list_edges.return_value = []
        return manager

    def test_can_iterate_directly(self, mock_manager):
        """Can iterate over find() result directly."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)

        for match in matcher.find(pattern):
            assert isinstance(match, PatternMatch)

    def test_can_check_if_matches(self, mock_manager):
        """Can use 'if result:' idiom."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)

        result = matcher.find(pattern)
        if result:
            assert len(result) > 0
        else:
            pytest.fail("Should have matches")

    def test_can_access_first_match(self, mock_manager):
        """Can access result[0] for first match."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)

        result = matcher.find(pattern)
        first = result[0]

        assert isinstance(first, PatternMatch)

    def test_can_use_in_list_comprehension(self, mock_manager):
        """Can use result in list comprehension."""
        pattern = Pattern().node("A")
        matcher = PatternMatcher(mock_manager)

        # Get the Task objects from bindings, then get their ids
        task_ids = [m.bindings["A"].id for m in matcher.find(pattern)]

        assert "task-1" in task_ids
        assert "task-2" in task_ids
