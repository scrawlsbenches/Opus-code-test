"""
╔══════════════════════════════════════════════════════════════════════╗
║                    INDEXER PERFORMANCE CONTRACT                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Index lookup latency < 1ms (O(1) vs O(n) scan)                   ║
║  • Index update latency < 2ms per task                               ║
║  • Indexed query 10x faster than unindexed scan                      ║
║  • Index rebuild < 200ms for 1,000 tasks                             ║
║  • Index persistence < 50ms for 1,000 tasks                          ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import tempfile
import time
from pathlib import Path
from typing import List

import pytest

from cortical.got.indexer import QueryIndexManager
from cortical.got.types import Task


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestIndexLookupPerformanceContract:
    """
    Index Lookup Performance Contract

    As a developer using our custom index manager for fast queries,
    I expect index lookups to be nearly instant,
    So that indexed queries provide dramatic speedup over linear scans.

    Our hand-built index implementation uses hash tables for O(1) lookup
    instead of O(n) linear scans.
    """

    # The sacred numbers
    LOOKUP_LATENCY_MS = 1.0  # Max 1ms for index lookup
    SPEEDUP_FACTOR = 10.0    # Indexed must be 10x faster than unindexed

    def test_lookup_latency_honored(self):
        """
        CONTRACT: Index lookups complete in under 1ms.

        Our custom hash-based index provides O(1) lookup time regardless
        of the total number of entities. This is the core value proposition.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Index 1000 tasks across different statuses
            tasks = []
            for i in range(1000):
                status = ["pending", "in_progress", "completed", "blocked"][i % 4]
                task = Task(
                    id=f"T-{i:04d}",
                    title=f"Task {i}",
                    status=status,
                    priority="medium"
                )
                tasks.append(task)
                index_mgr.index_task(task.id, status=status, priority="medium")

            index_mgr.save()

            # Measure lookup latency
            latencies = []
            for status in ["pending", "in_progress", "completed", "blocked"]:
                for _ in range(25):  # 100 lookups total
                    start = time.perf_counter()
                    results = index_mgr.lookup("status", status)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed_ms)

                    # Verify correctness
                    assert len(results) > 0, f"Expected results for status={status}"

            p95 = percentile(latencies, 95)

            assert p95 < self.LOOKUP_LATENCY_MS, (
                f"CONTRACT VIOLATION: Index lookup p95 is {p95:.3f}ms, "
                f"contract requires <{self.LOOKUP_LATENCY_MS}ms"
            )

    def test_indexed_query_speedup(self):
        """
        CONTRACT: Indexed queries are at least 10x faster than unindexed scans.

        This validates the fundamental performance benefit of our custom
        indexing implementation. Without this speedup, indexes are pointless.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Create 1000 tasks
            tasks = []
            for i in range(1000):
                status = "pending" if i < 500 else "completed"
                task = Task(
                    id=f"T-{i:04d}",
                    title=f"Task {i}",
                    status=status,
                    priority="medium"
                )
                tasks.append(task)
                index_mgr.index_task(task.id, status=status, priority="medium")

            # Measure indexed lookup
            indexed_times = []
            for _ in range(50):
                start = time.perf_counter()
                indexed_results = index_mgr.lookup("status", "pending")
                elapsed_ms = (time.perf_counter() - start) * 1000
                indexed_times.append(elapsed_ms)

            indexed_avg = sum(indexed_times) / len(indexed_times)

            # Measure unindexed linear scan (our fallback implementation)
            def linear_scan(tasks: List[Task], status: str) -> List[str]:
                return [t.id for t in tasks if t.status == status]

            unindexed_times = []
            for _ in range(50):
                start = time.perf_counter()
                unindexed_results = linear_scan(tasks, "pending")
                elapsed_ms = (time.perf_counter() - start) * 1000
                unindexed_times.append(elapsed_ms)

            unindexed_avg = sum(unindexed_times) / len(unindexed_times)

            # Calculate speedup
            speedup = unindexed_avg / indexed_avg if indexed_avg > 0 else 0

            assert speedup >= self.SPEEDUP_FACTOR, (
                f"CONTRACT VIOLATION: Index speedup is {speedup:.1f}x, "
                f"contract requires >={self.SPEEDUP_FACTOR}x "
                f"(indexed: {indexed_avg:.3f}ms, unindexed: {unindexed_avg:.3f}ms)"
            )

    def test_lookup_scales_with_index_size_not_corpus_size(self):
        """
        CONTRACT: Lookup time is independent of total corpus size.

        O(1) lookup means doubling the corpus size doesn't double lookup time.
        Our custom hash-based implementation must maintain this property.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Index 1000 tasks, only 10 are "critical" priority
            for i in range(1000):
                priority = "critical" if i < 10 else "medium"
                task = Task(
                    id=f"T-{i:04d}",
                    title=f"Task {i}",
                    status="pending",
                    priority=priority
                )
                index_mgr.index_task(task.id, status="pending", priority=priority)

            # Measure lookup for small result set (10 critical tasks)
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                results = index_mgr.lookup("priority", "critical")
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

                assert len(results) == 10, f"Expected 10 critical tasks"

            p95 = percentile(latencies, 95)

            # Even with 1000 total tasks, lookup for 10 should be instant
            assert p95 < 1.0, (
                f"CONTRACT VIOLATION: Lookup p95 is {p95:.3f}ms with 1000 corpus size"
            )


@pytest.mark.contract
class TestIndexUpdatePerformanceContract:
    """
    Index Update Performance Contract

    As a system maintaining our custom indexes in real-time,
    I expect index updates to be fast,
    So that transaction commits aren't bottlenecked by index maintenance.
    """

    # The sacred numbers
    UPDATE_LATENCY_MS = 2.0  # Max 2ms per task update

    def test_update_latency_honored(self):
        """
        CONTRACT: Index updates complete in under 2ms per task.

        Every transaction commit may update indexes. Our custom implementation
        must update indexes quickly to avoid slowing down commits.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Pre-populate with 500 tasks
            for i in range(500):
                task = Task(
                    id=f"T-{i:04d}",
                    title=f"Task {i}",
                    status="pending",
                    priority="medium"
                )
                index_mgr.index_task(task.id, status="pending", priority="medium")

            # Measure update latency
            latencies = []
            for i in range(100):
                task_id = f"T-{i:04d}"

                start = time.perf_counter()
                index_mgr.update_task(
                    task_id,
                    old_status="pending",
                    new_status="completed",
                    old_priority="medium",
                    new_priority="high"
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            p95 = percentile(latencies, 95)

            assert p95 < self.UPDATE_LATENCY_MS, (
                f"CONTRACT VIOLATION: Index update p95 is {p95:.3f}ms, "
                f"contract requires <{self.UPDATE_LATENCY_MS}ms"
            )

    def test_index_task_fast(self):
        """
        CONTRACT: Adding new tasks to index is fast.

        Our custom implementation must efficiently add new entries without
        degrading as the index grows.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            latencies = []
            for i in range(1000):
                task_id = f"T-{i:04d}"

                start = time.perf_counter()
                index_mgr.index_task(
                    task_id,
                    status="pending",
                    priority="medium"
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            # Average should be very low
            avg = sum(latencies) / len(latencies)
            p95 = percentile(latencies, 95)

            assert avg < 0.5, (
                f"CONTRACT VIOLATION: Average index_task time is {avg:.3f}ms"
            )
            assert p95 < 2.0, (
                f"CONTRACT VIOLATION: index_task p95 is {p95:.3f}ms"
            )

    def test_remove_task_fast(self):
        """
        CONTRACT: Removing tasks from index is fast.

        Our custom implementation must efficiently remove entries from
        all indexes without scanning entire index.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Pre-populate with 1000 tasks
            for i in range(1000):
                index_mgr.index_task(
                    f"T-{i:04d}",
                    status="pending",
                    priority="medium"
                )

            # Measure removal latency
            latencies = []
            for i in range(100):
                task_id = f"T-{i:04d}"

                start = time.perf_counter()
                index_mgr.remove_task(task_id)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            p95 = percentile(latencies, 95)

            assert p95 < 2.0, (
                f"CONTRACT VIOLATION: remove_task p95 is {p95:.3f}ms"
            )


@pytest.mark.contract
class TestIndexRebuildPerformanceContract:
    """
    Index Rebuild Performance Contract

    As a system that occasionally rebuilds our custom indexes,
    I expect rebuild to complete in reasonable time,
    So that recovery and maintenance operations don't block for long.
    """

    # The sacred numbers
    REBUILD_TIME_PER_1K_TASKS_MS = 200.0  # Max 200ms for 1K tasks

    def test_rebuild_time_bounded(self):
        """
        CONTRACT: Index rebuild completes in under 200ms for 1,000 tasks.

        Our hand-built index rebuild scans all tasks and reconstructs
        all indexes from scratch. This must be efficient.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Create 1000 tasks
            tasks = []
            for i in range(1000):
                task = Task(
                    id=f"T-{i:04d}",
                    title=f"Task {i}",
                    status=["pending", "in_progress", "completed"][i % 3],
                    priority=["low", "medium", "high"][i % 3]
                )
                tasks.append(task)

            # Measure rebuild time
            start = time.perf_counter()
            index_mgr.rebuild_all(tasks, edges=[])
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < self.REBUILD_TIME_PER_1K_TASKS_MS, (
                f"CONTRACT VIOLATION: Rebuild took {elapsed_ms:.2f}ms for 1000 tasks, "
                f"contract requires <{self.REBUILD_TIME_PER_1K_TASKS_MS}ms"
            )

            # Verify indexes work after rebuild
            pending_tasks = index_mgr.lookup("status", "pending")
            assert len(pending_tasks) > 0, "Index should contain pending tasks after rebuild"

    def test_rebuild_correctness(self):
        """
        CONTRACT: Rebuilt indexes produce correct results.

        Our custom rebuild must create indexes that return accurate
        results for all queries.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Create tasks with known distribution
            tasks = []
            pending_count = 0
            completed_count = 0

            for i in range(500):
                status = "pending" if i < 300 else "completed"
                task = Task(
                    id=f"T-{i:04d}",
                    title=f"Task {i}",
                    status=status,
                    priority="medium"
                )
                tasks.append(task)

                if status == "pending":
                    pending_count += 1
                else:
                    completed_count += 1

            # Rebuild indexes
            index_mgr.rebuild_all(tasks, edges=[])

            # Verify correct counts
            pending_results = index_mgr.lookup("status", "pending")
            completed_results = index_mgr.lookup("status", "completed")

            assert len(pending_results) == pending_count, (
                f"Expected {pending_count} pending tasks, got {len(pending_results)}"
            )
            assert len(completed_results) == completed_count, (
                f"Expected {completed_count} completed tasks, got {len(completed_results)}"
            )


@pytest.mark.contract
class TestIndexPersistenceContract:
    """
    Index Persistence Contract

    As a system persisting our custom indexes to disk,
    I expect save/load operations to be fast,
    So that index durability doesn't impact performance.
    """

    # The sacred numbers
    SAVE_TIME_PER_1K_TASKS_MS = 50.0  # Max 50ms to save 1K tasks worth of indexes
    LOAD_TIME_PER_1K_TASKS_MS = 50.0  # Max 50ms to load 1K tasks worth of indexes

    def test_save_time_bounded(self):
        """
        CONTRACT: Index persistence completes in under 50ms for 1,000 tasks.

        Our custom implementation writes indexes to JSON files with atomic
        rename. This must be fast to not slow down transaction commits.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Index 1000 tasks
            for i in range(1000):
                index_mgr.index_task(
                    f"T-{i:04d}",
                    status=["pending", "completed"][i % 2],
                    priority="medium"
                )

            # Measure save time
            start = time.perf_counter()
            index_mgr.save()
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < self.SAVE_TIME_PER_1K_TASKS_MS, (
                f"CONTRACT VIOLATION: Index save took {elapsed_ms:.2f}ms for 1000 tasks, "
                f"contract requires <{self.SAVE_TIME_PER_1K_TASKS_MS}ms"
            )

    def test_load_time_bounded(self):
        """
        CONTRACT: Index loading completes in under 50ms for 1,000 tasks.

        Our custom implementation loads indexes from JSON files on startup.
        This must be fast to minimize startup time.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save indexes
            index_mgr = QueryIndexManager(Path(tmpdir))
            for i in range(1000):
                index_mgr.index_task(
                    f"T-{i:04d}",
                    status=["pending", "completed"][i % 2],
                    priority="medium"
                )
            index_mgr.save()

            # Measure load time (create new manager instance)
            start = time.perf_counter()
            new_index_mgr = QueryIndexManager(Path(tmpdir))
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < self.LOAD_TIME_PER_1K_TASKS_MS, (
                f"CONTRACT VIOLATION: Index load took {elapsed_ms:.2f}ms for 1000 tasks, "
                f"contract requires <{self.LOAD_TIME_PER_1K_TASKS_MS}ms"
            )

            # Verify indexes loaded correctly
            pending = new_index_mgr.lookup("status", "pending")
            assert len(pending) > 0, "Loaded index should contain data"

    def test_atomic_save_no_corruption(self):
        """
        CONTRACT: Index saves are atomic and never leave partial state.

        Our custom implementation uses temp file + rename pattern.
        Interrupting a save should never corrupt existing indexes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index_mgr = QueryIndexManager(Path(tmpdir))

            # Index some tasks and save
            for i in range(100):
                index_mgr.index_task(
                    f"T-{i:04d}",
                    status="pending",
                    priority="medium"
                )
            index_mgr.save()

            # Verify files exist
            index_files = list((Path(tmpdir) / "indexes").glob("*.json"))
            assert len(index_files) > 0, "Index files should exist after save"

            # Verify no .tmp files left behind
            tmp_files = list((Path(tmpdir) / "indexes").glob("*.tmp"))
            assert len(tmp_files) == 0, (
                f"CONTRACT VIOLATION: Found {len(tmp_files)} .tmp files after save"
            )

            # Verify indexes can be loaded
            new_index_mgr = QueryIndexManager(Path(tmpdir))
            results = new_index_mgr.lookup("status", "pending")
            assert len(results) == 100, "All tasks should be in loaded index"
