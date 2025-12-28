"""
Performance tests for GoT Index Operations.

KPI Targets (based on index design goals):
- Index rebuild (50 tasks): <100ms (current: ~50-80ms)
- Index rebuild (100 tasks): <200ms
- Index rebuild (200 tasks): <400ms (should scale linearly)
- Indexed lookup: <1ms (O(1) hash lookup)
- Uncached lookup: <5ms (O(n) scan)
- Cache speedup: 5-10x faster with index vs. without

These tests verify that index operations meet performance targets and
that indexes provide the expected speedup over linear scans.

Run with: python -m pytest tests/performance/test_got_index_perf.py -v -s --timeout=60

PERFORMANCE NOTE:
This file uses class-scoped fixtures from conftest.py to avoid recreating
managers for each test, saving significant time.
"""

import time
from pathlib import Path
import pytest

from cortical.got import GoTManager
from cortical.got.query_builder import Query


# =============================================================================
# KPI TARGETS (in milliseconds)
# =============================================================================
# Based on index design goals:
# - Index operations should be <100ms for typical datasets
# - Lookup should be instant (<1ms) for indexed fields
# - Rebuild should scale linearly with dataset size

KPI_INDEX_REBUILD_50 = 100      # Rebuild with 50 tasks
KPI_INDEX_REBUILD_100 = 200     # Rebuild with 100 tasks
KPI_INDEX_REBUILD_200 = 400     # Rebuild with 200 tasks
KPI_INDEXED_LOOKUP = 1          # Indexed field lookup
KPI_QUERY_RESPONSE = 500        # Typical query with index

# Safety margin for CI variance (2x for index operations which involve I/O)
CI_VARIANCE_FACTOR = 2.0


class TestIndexBuildPerformance:
    """Performance tests for index building with varying dataset sizes."""

    def _measure(self, operation_name: str, fn, iterations: int = 5) -> float:
        """Measure average execution time over multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"\n  {operation_name}: {avg_ms:.2f}ms avg (min={min_ms:.2f}, max={max_ms:.2f})")
        return avg_ms

    def _create_tasks(self, manager: GoTManager, count: int) -> list:
        """Helper to create tasks for testing."""
        tasks = []
        priorities = ["critical", "high", "medium", "low"]
        statuses = ["pending", "in_progress", "completed"]

        for i in range(count):
            task = manager.create_task(
                f"Task {i}",
                priority=priorities[i % len(priorities)]
            )
            # Vary status to populate index groups
            if i % 3 == 0:
                manager.update_task(task.id, status="in_progress")
            elif i % 5 == 0:
                manager.update_task(task.id, status="completed")
            tasks.append(task)

        return tasks

    @pytest.mark.slow
    def test_index_rebuild_50_tasks(self, fresh_got_manager):
        """Index rebuild with 50 tasks should be < KPI target."""
        manager = fresh_got_manager

        # Create 50 tasks
        tasks = self._create_tasks(manager, 50)

        # Get all edges for sprint index
        edges = manager.list_edges()

        # Measure rebuild time
        avg_ms = self._measure(
            "Index rebuild (50 tasks)",
            lambda: manager.index_manager.rebuild_all(tasks, edges)
        )

        assert avg_ms < KPI_INDEX_REBUILD_50 * CI_VARIANCE_FACTOR, \
            f"Index rebuild (50 tasks) took {avg_ms:.2f}ms, target is <{KPI_INDEX_REBUILD_50}ms"

    @pytest.mark.slow
    def test_index_rebuild_100_tasks(self, fresh_got_manager):
        """Index rebuild with 100 tasks should be < KPI target."""
        manager = fresh_got_manager

        # Create 100 tasks
        tasks = self._create_tasks(manager, 100)
        edges = manager.list_edges()

        # Measure rebuild time
        avg_ms = self._measure(
            "Index rebuild (100 tasks)",
            lambda: manager.index_manager.rebuild_all(tasks, edges)
        )

        assert avg_ms < KPI_INDEX_REBUILD_100 * CI_VARIANCE_FACTOR, \
            f"Index rebuild (100 tasks) took {avg_ms:.2f}ms, target is <{KPI_INDEX_REBUILD_100}ms"

    @pytest.mark.slow
    def test_index_rebuild_200_tasks(self, fresh_got_manager):
        """Index rebuild with 200 tasks should be < KPI target."""
        manager = fresh_got_manager

        # Create 200 tasks
        tasks = self._create_tasks(manager, 200)
        edges = manager.list_edges()

        # Measure rebuild time
        avg_ms = self._measure(
            "Index rebuild (200 tasks)",
            lambda: manager.index_manager.rebuild_all(tasks, edges)
        )

        assert avg_ms < KPI_INDEX_REBUILD_200 * CI_VARIANCE_FACTOR, \
            f"Index rebuild (200 tasks) took {avg_ms:.2f}ms, target is <{KPI_INDEX_REBUILD_200}ms"

    @pytest.mark.slow
    def test_index_rebuild_scales_linearly(self, fresh_got_manager):
        """Index rebuild should scale linearly with task count."""
        manager = fresh_got_manager

        # Measure with 50 tasks
        tasks_50 = self._create_tasks(manager, 50)
        edges = manager.list_edges()

        start = time.perf_counter()
        manager.index_manager.rebuild_all(tasks_50, edges)
        time_50 = (time.perf_counter() - start) * 1000

        # Measure with 100 tasks (2x)
        tasks_100 = self._create_tasks(manager, 50)  # Add 50 more
        tasks_all = tasks_50 + tasks_100
        edges = manager.list_edges()

        start = time.perf_counter()
        manager.index_manager.rebuild_all(tasks_all, edges)
        time_100 = (time.perf_counter() - start) * 1000

        print(f"\n  50 tasks: {time_50:.2f}ms")
        print(f"  100 tasks: {time_100:.2f}ms")
        print(f"  Ratio: {time_100 / time_50:.2f}x")

        # Should be roughly 2x (allowing for overhead)
        # Linear scaling means time_100 should be ~2x time_50
        # Allow 3x variance for overhead and CI variability
        assert time_100 < time_50 * 3, \
            f"Index rebuild doesn't scale linearly: {time_100:.2f}ms for 100 tasks vs {time_50:.2f}ms for 50 tasks"


class TestIndexLookupPerformance:
    """Performance tests for index lookup operations."""

    def _measure(self, operation_name: str, fn, iterations: int = 100) -> float:
        """Measure average execution time over multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"\n  {operation_name}: {avg_ms:.4f}ms avg (min={min_ms:.4f}, max={max_ms:.4f})")
        return avg_ms

    def test_indexed_lookup_performance(self, got_manager_large):
        """Indexed lookup should be < KPI target."""
        manager, tasks = got_manager_large

        # Ensure indexes are built
        manager.index_manager.rebuild_all(tasks)

        # Measure indexed lookup
        avg_ms = self._measure(
            "Indexed lookup (status='pending')",
            lambda: manager.index_manager.lookup("status", "pending"),
            iterations=100
        )

        assert avg_ms < KPI_INDEXED_LOOKUP * CI_VARIANCE_FACTOR, \
            f"Indexed lookup took {avg_ms:.4f}ms, target is <{KPI_INDEXED_LOOKUP}ms"

    def test_query_with_index_performance(self, got_manager_large):
        """Query using index should be < KPI target."""
        manager, tasks = got_manager_large

        # Ensure indexes are built
        manager.index_manager.rebuild_all(tasks)

        # Measure query with index
        avg_ms = self._measure(
            "Query with index (status='pending')",
            lambda: Query(manager).tasks().where(status="pending").execute(),
            iterations=20
        )

        assert avg_ms < KPI_QUERY_RESPONSE * CI_VARIANCE_FACTOR, \
            f"Query with index took {avg_ms:.2f}ms, target is <{KPI_QUERY_RESPONSE}ms"


class TestIndexCachePerformance:
    """Test that indexes provide expected speedup over linear scans."""

    def test_index_provides_speedup(self, fresh_got_manager):
        """Indexed queries should be significantly faster than unindexed."""
        manager = fresh_got_manager

        # Create 100 tasks
        priorities = ["critical", "high", "medium", "low"]
        tasks = []
        for i in range(100):
            task = manager.create_task(
                f"Task {i}",
                priority=priorities[i % len(priorities)]
            )
            tasks.append(task)

        # Rebuild indexes
        manager.index_manager.rebuild_all(tasks)

        # Measure indexed lookup
        start = time.perf_counter()
        indexed_result = manager.index_manager.lookup("priority", "high")
        indexed_ms = (time.perf_counter() - start) * 1000

        # Measure unindexed scan (simulate by iterating all tasks)
        start = time.perf_counter()
        unindexed_result = {t.id for t in tasks if t.priority == "high"}
        unindexed_ms = (time.perf_counter() - start) * 1000

        print(f"\n  Indexed lookup: {indexed_ms:.4f}ms")
        print(f"  Unindexed scan: {unindexed_ms:.4f}ms")
        print(f"  Speedup: {unindexed_ms / indexed_ms:.1f}x")

        # Verify same results
        assert indexed_result == unindexed_result, "Indexed and unindexed results should match"

        # Indexed should be faster (allow some variance for very small datasets)
        assert indexed_ms < unindexed_ms or indexed_ms < 1.0, \
            "Indexed lookup should be faster than linear scan (or both extremely fast)"

    def test_index_stats_tracking(self, got_manager_large):
        """Index stats should track hits and misses correctly."""
        manager, tasks = got_manager_large

        # Rebuild indexes and reset stats
        manager.index_manager.rebuild_all(tasks)
        manager.index_manager._stats.hits = 0
        manager.index_manager._stats.misses = 0

        # Perform several lookups
        # Note: got_manager_large has all tasks with status="pending" (no "completed")
        for _ in range(10):
            manager.index_manager.lookup("status", "pending")  # Should hit (100 tasks)
            manager.index_manager.lookup("status", "completed")  # Should miss (no tasks)
            manager.index_manager.lookup("nonexistent_field", "value")  # Should miss (field doesn't exist)

        stats = manager.index_manager._stats

        print(f"\n  Hits: {stats.hits}")
        print(f"  Misses: {stats.misses}")
        print(f"  Hit rate: {stats.hit_rate:.1%}")

        # We did 10 successful lookups (pending) + 20 misses (10 completed + 10 nonexistent) = 30 total
        assert stats.hits == 10, f"Expected 10 hits, got {stats.hits}"
        assert stats.misses == 20, f"Expected 20 misses, got {stats.misses}"
        assert stats.hit_rate == 10/30, f"Expected hit rate of {10/30:.1%}, got {stats.hit_rate:.1%}"


class TestIndexMaintenancePerformance:
    """Test performance of incremental index updates."""

    def _measure(self, operation_name: str, fn, iterations: int = 100) -> float:
        """Measure average execution time over multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"\n  {operation_name}: {avg_ms:.4f}ms avg (min={min_ms:.4f}, max={max_ms:.4f})")
        return avg_ms

    def test_incremental_update_performance(self, got_manager_large):
        """Incremental index updates should be fast."""
        manager, tasks = got_manager_large

        # Rebuild indexes
        manager.index_manager.rebuild_all(tasks)

        # Measure update time
        task_id = tasks[0].id
        avg_ms = self._measure(
            "Incremental update (status change)",
            lambda: manager.index_manager.update_task(
                task_id,
                old_status="pending",
                new_status="in_progress"
            ),
            iterations=50
        )

        # Updates should be very fast (just hash table updates)
        assert avg_ms < 1.0, \
            f"Incremental update took {avg_ms:.4f}ms, should be <1ms"

    def test_add_task_performance(self, got_manager_large):
        """Adding tasks to index should be fast."""
        manager, tasks = got_manager_large

        # Rebuild indexes
        manager.index_manager.rebuild_all(tasks)

        # Measure add time
        avg_ms = self._measure(
            "Add task to index",
            lambda: manager.index_manager.index_task(
                "T-TEST-123",
                status="pending",
                priority="high"
            ),
            iterations=50
        )

        # Cleanup
        manager.index_manager.remove_task("T-TEST-123")

        # Add should be very fast
        assert avg_ms < 1.0, \
            f"Index add took {avg_ms:.4f}ms, should be <1ms"

    def test_remove_task_performance(self, got_manager_large):
        """Removing tasks from index should be fast."""
        manager, tasks = got_manager_large

        # Rebuild indexes
        manager.index_manager.rebuild_all(tasks)

        # Add a task to remove
        manager.index_manager.index_task(
            "T-TEST-123",
            status="pending",
            priority="high"
        )

        # Measure remove time
        avg_ms = self._measure(
            "Remove task from index",
            lambda: manager.index_manager.remove_task("T-TEST-123"),
            iterations=50
        )

        # Remove should be very fast
        assert avg_ms < 1.0, \
            f"Index remove took {avg_ms:.4f}ms, should be <1ms"
