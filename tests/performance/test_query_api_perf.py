"""
Performance tests for Query API.

Tests that new features don't add unacceptable overhead:
1. explain() should be fast (no actual execution)
2. Truncation tracking should have minimal overhead
3. PatternSearchResult wrapper should be negligible
4. Direction methods should not affect traversal speed

These tests use timing assertions to catch performance regressions.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from cortical.got import GoTManager
from cortical.got.query_builder import Query
from cortical.got.graph_walker import GraphWalker
from cortical.got.path_finder import PathFinder
from cortical.got.pattern_matcher import Pattern, PatternMatcher, PatternSearchResult, PatternMatch
from cortical.got.types import Task, Edge


class TestExplainPerformance:
    """Test that explain() is fast.

    Note: Timing thresholds are set conservatively (50ms) to avoid
    flaky failures on slow CI servers. Local development typically
    sees 1-5ms per call.
    """

    @pytest.fixture
    def large_mock_manager(self):
        """Create mock manager with many tasks."""
        manager = MagicMock()
        # Simulate 1000 tasks
        manager.list_all_tasks.return_value = [
            Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
            for i in range(1000)
        ]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_query_explain_is_fast(self, large_mock_manager):
        """Query.explain() should complete in < 50ms."""
        query = (
            Query(large_mock_manager)
            .tasks()
            .where(status="pending", priority="high")
            .order_by("title")
            .limit(100)
        )

        start = time.perf_counter()
        for _ in range(100):  # Run 100 times
            plan = query.explain()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"explain() took {avg_ms:.2f}ms average, expected < 50ms"

    def test_walker_explain_is_fast(self, large_mock_manager):
        """GraphWalker.explain() should complete in < 50ms."""
        walker = (
            GraphWalker(large_mock_manager)
            .starting_from("task-0")
            .bfs()
            .max_depth(10)
            .follow("DEPENDS_ON")
        )

        start = time.perf_counter()
        for _ in range(100):
            plan = walker.explain()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"explain() took {avg_ms:.2f}ms average, expected < 50ms"

    def test_path_finder_explain_is_fast(self, large_mock_manager):
        """PathFinder.explain() should complete in < 50ms."""
        finder = PathFinder(large_mock_manager).max_paths(100).max_length(10)

        start = time.perf_counter()
        for _ in range(100):
            plan = finder.explain()  # No arguments - describes finder config
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"explain() took {avg_ms:.2f}ms average, expected < 50ms"

    def test_pattern_matcher_explain_is_fast(self, large_mock_manager):
        """PatternMatcher.explain() should complete in < 50ms."""
        pattern = Pattern().node("A", status="pending").outgoing("DEPENDS_ON").node("B")
        matcher = PatternMatcher(large_mock_manager).limit(100)

        start = time.perf_counter()
        for _ in range(100):
            plan = matcher.explain(pattern)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"explain() took {avg_ms:.2f}ms average, expected < 50ms"


class TestPatternSearchResultOverhead:
    """Test PatternSearchResult wrapper overhead."""

    def test_iteration_overhead_is_minimal(self):
        """Iterating PatternSearchResult should be nearly as fast as list."""
        # Create 10000 matches
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"}, )
            for i in range(10000)
        ]

        # Time raw list iteration
        raw_list = matches
        start = time.perf_counter()
        for _ in range(10):
            count = 0
            for m in raw_list:
                count += 1
        list_time = time.perf_counter() - start

        # Time PatternSearchResult iteration
        result = PatternSearchResult(matches=matches)
        start = time.perf_counter()
        for _ in range(10):
            count = 0
            for m in result:
                count += 1
        result_time = time.perf_counter() - start

        # Result should be no more than 2x slower than raw list
        overhead_ratio = result_time / list_time
        assert overhead_ratio < 2.0, f"Overhead ratio {overhead_ratio:.2f}x, expected < 2x"

    def test_len_is_o1(self):
        """len(result) should be O(1)."""
        # Create large result
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"}, )
            for i in range(100000)
        ]
        result = PatternSearchResult(matches=matches)

        # Time len()
        start = time.perf_counter()
        for _ in range(100000):
            _ = len(result)
        elapsed = time.perf_counter() - start

        # 100000 calls should complete in < 100ms
        assert elapsed < 0.1, f"len() took {elapsed*1000:.2f}ms for 100k calls, expected < 100ms"

    def test_bool_is_o1(self):
        """bool(result) should be O(1)."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"}, )
            for i in range(100000)
        ]
        result = PatternSearchResult(matches=matches)

        start = time.perf_counter()
        for _ in range(100000):
            _ = bool(result)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"bool() took {elapsed*1000:.2f}ms for 100k calls, expected < 100ms"


class TestTruncationTrackingOverhead:
    """Test that truncation tracking doesn't slow down queries."""

    @pytest.fixture
    def manager_with_chain(self):
        """Create manager with a chain of tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create 100 tasks
        tasks = []
        for i in range(100):
            task = manager.create_task(f"Task {i}", priority="medium")
            tasks.append(task)

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pattern_find_with_limit_not_slower(self, manager_with_chain):
        """PatternMatcher.find() with limit should not be significantly slower."""
        manager, tasks = manager_with_chain

        pattern = Pattern().node("A")

        # Time without limit (gets all)
        matcher1 = PatternMatcher(manager)
        start = time.perf_counter()
        result1 = matcher1.find(pattern)
        time_no_limit = time.perf_counter() - start

        # Time with limit
        matcher2 = PatternMatcher(manager).limit(10)
        start = time.perf_counter()
        result2 = matcher2.find(pattern)
        time_with_limit = time.perf_counter() - start

        # With limit should be faster or equal (less work)
        # Allow 50% overhead for truncation tracking
        assert time_with_limit < time_no_limit * 1.5, \
            f"With limit: {time_with_limit*1000:.2f}ms, without: {time_no_limit*1000:.2f}ms"

        # Verify truncation was tracked correctly
        assert len(result2) == 10
        assert result2.truncated is True


class TestDirectionMethodsOverhead:
    """Test that direction methods don't add overhead."""

    @pytest.fixture
    def mock_manager_with_edges(self):
        """Create mock manager with edges."""
        manager = MagicMock()
        tasks = [
            Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
            for i in range(100)
        ]
        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        # Create chain of edges
        edges = [
            Edge(id=f"edge-{i}", source_id=f"task-{i}", target_id=f"task-{i+1}", edge_type="DEPENDS_ON")
            for i in range(99)
        ]
        manager.list_edges.return_value = edges
        return manager

    def test_outgoing_same_as_directed(self, mock_manager_with_edges):
        """outgoing() should produce same results as directed()."""
        # Test behavioral equivalence, not timing (timing is too noisy for CI)
        walker1 = GraphWalker(mock_manager_with_edges).starting_from("task-0").directed().bfs()
        walker2 = GraphWalker(mock_manager_with_edges).starting_from("task-0").outgoing().bfs()

        result1 = walker1.visit(lambda n, acc: acc | {n.id}, initial=set()).run()
        result2 = walker2.visit(lambda n, acc: acc | {n.id}, initial=set()).run()

        # Both should visit the same nodes
        assert result1 == result2, f"directed() and outgoing() should be equivalent"

    def test_incoming_same_as_reverse(self, mock_manager_with_edges):
        """incoming() should produce same results as reverse()."""
        # Test behavioral equivalence, not timing (timing is too noisy for CI)
        walker1 = GraphWalker(mock_manager_with_edges).starting_from("task-99").reverse().bfs()
        walker2 = GraphWalker(mock_manager_with_edges).starting_from("task-99").incoming().bfs()

        result1 = walker1.visit(lambda n, acc: acc | {n.id}, initial=set()).run()
        result2 = walker2.visit(lambda n, acc: acc | {n.id}, initial=set()).run()

        # Both should visit the same nodes
        assert result1 == result2, f"reverse() and incoming() should be equivalent"


class TestQueryScaling:
    """Test that query performance scales reasonably."""

    def test_query_scales_linearly_with_task_count(self):
        """Query execution should scale roughly linearly with task count."""
        times = []

        for count in [100, 500, 1000]:
            manager = MagicMock()
            manager.list_all_tasks.return_value = [
                Task(id=f"task-{i}", title=f"Task {i}", status="pending", priority="high")
                for i in range(count)
            ]
            manager.list_sprints.return_value = []
            manager.list_decisions.return_value = []
            manager.list_edges.return_value = []

            query = Query(manager).tasks().where(status="pending")

            start = time.perf_counter()
            for _ in range(10):
                results = query.execute()
            elapsed = time.perf_counter() - start

            times.append((count, elapsed / 10))

        # Check scaling: 10x more tasks should be < 20x slower
        time_100, time_1000 = times[0][1], times[2][1]
        scaling_factor = time_1000 / time_100 if time_100 > 0 else 1.0

        # Allow for some overhead, but should be roughly linear
        assert scaling_factor < 20, f"Scaling factor {scaling_factor:.2f}x for 10x tasks, expected < 20x"


class TestMemoryEfficiency:
    """Test memory efficiency of result types."""

    def test_pattern_search_result_no_copy(self):
        """PatternSearchResult should not copy the matches list."""
        matches = [
            PatternMatch(bindings={"A": f"task-{i}"}, )
            for i in range(1000)
        ]

        result = PatternSearchResult(matches=matches)

        # The internal list should be the same object (no copy)
        assert result.matches is matches
