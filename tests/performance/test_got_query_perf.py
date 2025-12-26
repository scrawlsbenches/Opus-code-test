"""
Performance tests for GoT Query API.

KPI Targets (based on profiling with 50 tasks, 10 decisions, ~90 edges):
- Query API: <10ms (current: 7ms)
- GraphWalker: <30ms (current: 20ms)
- PathFinder: <20ms (current: 10-20ms)
- PatternMatcher: <30ms (current: 23ms)

These tests verify that Query API operations stay within acceptable bounds.
Run with: python -m pytest tests/performance/test_got_query_perf.py -v -s
"""

import time
import tempfile
import shutil
from pathlib import Path
import pytest

from cortical.got import GoTManager
from cortical.got.query_builder import Query
from cortical.got.graph_walker import GraphWalker
from cortical.got.path_finder import PathFinder
from cortical.got.pattern_matcher import PatternMatcher, Pattern


# =============================================================================
# KPI TARGETS (in milliseconds)
# =============================================================================
# These are the performance targets based on user experience research:
# - < 100ms = Instant (no perceived delay)
# - 100-300ms = Fast
# - 300-1000ms = Noticeable delay
# - > 1000ms = Slow

KPI_QUERY_MS = 10        # Query API operations
KPI_WALKER_MS = 30       # GraphWalker traversals
KPI_PATH_FINDER_MS = 20  # PathFinder operations
KPI_PATTERN_MS = 30      # PatternMatcher operations

# Safety margin for CI variance (1.5x)
CI_VARIANCE_FACTOR = 1.5


class TestGoTQueryPerformance:
    """Performance regression tests for GoT Query API."""

    @pytest.fixture
    def got_manager(self):
        """Create a GoT manager with representative test data."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create 50 tasks (representative workload)
        tasks = []
        for i in range(50):
            priority = ["high", "medium", "low"][i % 3]
            status = ["pending", "in_progress", "completed"][i % 3]
            task = manager.create_task(
                f"Task {i}: Performance test task",
                priority=priority
            )
            if status != "pending":
                if status == "in_progress":
                    manager.update_task(task.id, status="in_progress")
                else:
                    manager.update_task(task.id, status="completed")
            tasks.append(task)

        # Create 10 decisions
        decisions = []
        for i in range(10):
            decision = manager.create_decision(
                f"Decision {i}: Performance test decision",
                rationale=f"Rationale for decision {i}"
            )
            decisions.append(decision)

        # Create ~90 edges (dependencies between tasks)
        for i in range(len(tasks) - 1):
            if i % 2 == 0:  # ~25 DEPENDS_ON edges
                manager.add_edge(tasks[i].id, tasks[i + 1].id, "DEPENDS_ON")
            if i % 3 == 0:  # ~17 BLOCKS edges
                manager.add_edge(tasks[i].id, tasks[(i + 3) % len(tasks)].id, "BLOCKS")
            if i % 4 == 0:  # ~12 RELATES_TO edges
                manager.add_edge(tasks[i].id, decisions[i % 10].id, "RELATES_TO")

        # Clear cache to ensure fair timing
        manager.cache_clear()

        yield manager, tasks, decisions
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _measure(self, operation_name: str, fn, iterations: int = 5) -> float:
        """Measure average execution time over multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        print(f"\n  {operation_name}: {avg_ms:.2f}ms avg (min={min(times):.2f}, max={max(times):.2f})")
        return avg_ms

    # =========================================================================
    # QUERY API TESTS
    # =========================================================================

    def test_query_all_tasks(self, got_manager):
        """Query all tasks should be < KPI target."""
        manager, tasks, _ = got_manager

        avg_ms = self._measure(
            "Query: All tasks",
            lambda: Query(manager).tasks().execute()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query all tasks took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    def test_query_filter_by_status(self, got_manager):
        """Query with status filter should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "Query: Filter by status",
            lambda: Query(manager).tasks().where(status="pending").execute()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query filter by status took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    def test_query_filter_by_priority(self, got_manager):
        """Query with priority filter should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "Query: Filter by priority",
            lambda: Query(manager).tasks().where(priority="high").execute()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query filter by priority took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    def test_query_order_by(self, got_manager):
        """Query with ordering should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "Query: Order by priority",
            lambda: Query(manager).tasks().order_by("priority").execute()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query order by took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    def test_query_pagination(self, got_manager):
        """Query with pagination should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "Query: Pagination (limit 10)",
            lambda: Query(manager).tasks().limit(10).execute()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query pagination took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    def test_query_count(self, got_manager):
        """Query count should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "Query: Count",
            lambda: Query(manager).tasks().count()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query count took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    def test_query_group_by(self, got_manager):
        """Query group by with count should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "Query: Group by + count",
            lambda: Query(manager).tasks().group_by("priority").count().execute()
        )

        assert avg_ms < KPI_QUERY_MS * CI_VARIANCE_FACTOR, \
            f"Query group by took {avg_ms:.2f}ms, target is <{KPI_QUERY_MS}ms"

    # =========================================================================
    # GRAPH WALKER TESTS
    # =========================================================================

    def test_walker_bfs_traversal(self, got_manager):
        """GraphWalker BFS should be < KPI target."""
        manager, tasks, _ = got_manager
        start_id = tasks[0].id

        def walker_bfs():
            def visitor(node, acc):
                acc.append(node)
                return acc
            return GraphWalker(manager).starting_from(start_id).bfs().visit(visitor, initial=[]).run()

        avg_ms = self._measure("Walker: BFS traversal", walker_bfs)

        assert avg_ms < KPI_WALKER_MS * CI_VARIANCE_FACTOR, \
            f"Walker BFS took {avg_ms:.2f}ms, target is <{KPI_WALKER_MS}ms"

    def test_walker_dfs_traversal(self, got_manager):
        """GraphWalker DFS should be < KPI target."""
        manager, tasks, _ = got_manager
        start_id = tasks[0].id

        def walker_dfs():
            def visitor(node, acc):
                acc.append(node)
                return acc
            return GraphWalker(manager).starting_from(start_id).dfs().visit(visitor, initial=[]).run()

        avg_ms = self._measure("Walker: DFS traversal", walker_dfs)

        assert avg_ms < KPI_WALKER_MS * CI_VARIANCE_FACTOR, \
            f"Walker DFS took {avg_ms:.2f}ms, target is <{KPI_WALKER_MS}ms"

    def test_walker_with_max_depth(self, got_manager):
        """GraphWalker with max depth should be < KPI target."""
        manager, tasks, _ = got_manager
        start_id = tasks[0].id

        def walker_depth():
            def visitor(node, acc):
                acc.append(node)
                return acc
            return GraphWalker(manager).starting_from(start_id).max_depth(2).bfs().visit(visitor, initial=[]).run()

        avg_ms = self._measure("Walker: Max depth 2", walker_depth)

        assert avg_ms < KPI_WALKER_MS * CI_VARIANCE_FACTOR, \
            f"Walker max depth took {avg_ms:.2f}ms, target is <{KPI_WALKER_MS}ms"

    # =========================================================================
    # PATH FINDER TESTS
    # =========================================================================

    def test_path_finder_shortest_path(self, got_manager):
        """PathFinder shortest path should be < KPI target."""
        manager, tasks, _ = got_manager
        from_id = tasks[0].id
        to_id = tasks[10].id

        avg_ms = self._measure(
            "PathFinder: Shortest path",
            lambda: PathFinder(manager).shortest_path(from_id, to_id)
        )

        assert avg_ms < KPI_PATH_FINDER_MS * CI_VARIANCE_FACTOR, \
            f"PathFinder shortest path took {avg_ms:.2f}ms, target is <{KPI_PATH_FINDER_MS}ms"

    def test_path_finder_reachable_from(self, got_manager):
        """PathFinder reachable_from should be < KPI target."""
        manager, tasks, _ = got_manager
        start_id = tasks[0].id

        avg_ms = self._measure(
            "PathFinder: Reachable from",
            lambda: PathFinder(manager).reachable_from(start_id)
        )

        assert avg_ms < KPI_PATH_FINDER_MS * CI_VARIANCE_FACTOR, \
            f"PathFinder reachable_from took {avg_ms:.2f}ms, target is <{KPI_PATH_FINDER_MS}ms"

    def test_path_finder_connected_components(self, got_manager):
        """PathFinder connected_components should be < KPI target."""
        manager, _, _ = got_manager

        avg_ms = self._measure(
            "PathFinder: Connected components",
            lambda: PathFinder(manager).connected_components()
        )

        # Connected components is more expensive, allow 2x target
        assert avg_ms < KPI_PATH_FINDER_MS * 2 * CI_VARIANCE_FACTOR, \
            f"PathFinder connected_components took {avg_ms:.2f}ms, target is <{KPI_PATH_FINDER_MS * 2}ms"

    def test_path_finder_all_paths_with_limits(self, got_manager):
        """PathFinder all_paths with limits should be < KPI target."""
        manager, tasks, _ = got_manager
        from_id = tasks[0].id
        to_id = tasks[5].id

        avg_ms = self._measure(
            "PathFinder: All paths (limited)",
            lambda: PathFinder(manager).max_paths(10).max_length(5).all_paths(from_id, to_id)
        )

        assert avg_ms < KPI_PATH_FINDER_MS * CI_VARIANCE_FACTOR, \
            f"PathFinder all_paths took {avg_ms:.2f}ms, target is <{KPI_PATH_FINDER_MS}ms"

    # =========================================================================
    # PATTERN MATCHER TESTS
    # =========================================================================

    def test_pattern_matcher_2_node(self, got_manager):
        """PatternMatcher 2-node pattern should be < KPI target."""
        manager, _, _ = got_manager

        # Simple 2-node dependency pattern using fluent API
        pattern = Pattern().node("a", type="task").edge("DEPENDS_ON").node("b", type="task")

        avg_ms = self._measure(
            "Pattern: 2-node dependency",
            lambda: PatternMatcher(manager).find(pattern)
        )

        assert avg_ms < KPI_PATTERN_MS * CI_VARIANCE_FACTOR, \
            f"PatternMatcher 2-node took {avg_ms:.2f}ms, target is <{KPI_PATTERN_MS}ms"

    def test_pattern_matcher_with_constraints(self, got_manager):
        """PatternMatcher with constraints should be < KPI target."""
        manager, _, _ = got_manager

        # Pattern with status constraint using fluent API
        pattern = (
            Pattern()
            .node("a", type="task", status="pending")
            .edge("DEPENDS_ON")
            .node("b", type="task")
        )

        avg_ms = self._measure(
            "Pattern: With constraints",
            lambda: PatternMatcher(manager).find(pattern)
        )

        assert avg_ms < KPI_PATTERN_MS * CI_VARIANCE_FACTOR, \
            f"PatternMatcher with constraints took {avg_ms:.2f}ms, target is <{KPI_PATTERN_MS}ms"


class TestCachePerformance:
    """Test that caching provides expected speedup."""

    @pytest.fixture
    def got_manager_with_tasks(self):
        """Create a GoT manager with tasks for cache testing."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create 50 tasks
        for i in range(50):
            manager.create_task(f"Task {i}", priority="medium")

        manager.cache_clear()

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_provides_speedup(self, got_manager_with_tasks):
        """Second query should be faster due to caching."""
        manager = got_manager_with_tasks

        # First query (cold cache)
        start = time.perf_counter()
        Query(manager).tasks().execute()
        cold_ms = (time.perf_counter() - start) * 1000

        # Second query (warm cache)
        start = time.perf_counter()
        Query(manager).tasks().execute()
        warm_ms = (time.perf_counter() - start) * 1000

        print(f"\n  Cold cache: {cold_ms:.2f}ms")
        print(f"  Warm cache: {warm_ms:.2f}ms")
        print(f"  Speedup: {cold_ms / warm_ms:.1f}x")

        stats = manager.cache_stats()
        print(f"  Cache hits: {stats['hits']}, misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")

        # Warm cache should be significantly faster
        assert warm_ms < cold_ms, "Warm cache should be faster than cold cache"
        assert stats['hits'] > 0, "Cache should have hits on second query"

    def test_cache_hit_rate(self, got_manager_with_tasks):
        """After warm-up, hit rate should be high."""
        manager = got_manager_with_tasks

        # Warm up cache
        Query(manager).tasks().execute()

        # Run several queries
        for _ in range(5):
            Query(manager).tasks().execute()
            Query(manager).tasks().where(priority="medium").execute()

        stats = manager.cache_stats()
        print(f"\n  Cache hit rate: {stats['hit_rate']:.1%}")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")

        # After warm-up, hit rate should be at least 50%
        assert stats['hit_rate'] >= 0.5, \
            f"Hit rate {stats['hit_rate']:.1%} is below 50% threshold"
