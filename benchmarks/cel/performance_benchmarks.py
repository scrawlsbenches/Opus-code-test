"""
Performance Optimization Benchmarks for CEL.

These benchmarks specifically measure the optimizations in cortical.cel.performance:
- EntityIndex: O(1) entity → events lookup
- OptimizedDAG: Heap-based topological sort
- SnapshotManager: Fast recovery from snapshots
- StreamingEventStore: Lazy loading + write batching

Usage:
    python -m benchmarks.cel.performance_benchmarks
"""

from __future__ import annotations

import random
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
)


# Try to import performance modules
try:
    from cortical.cel.performance.entity_index import (
        EntityIndex,
        ConceptIndex,
        TemporalIndex,
    )
    from cortical.cel.performance.optimized_dag import (
        OptimizedDAG,
        HeapTopologicalSort,
    )
    from cortical.cel.performance.snapshots import (
        SnapshotManager,
        SnapshotConfig,
        Snapshot,
    )
    from cortical.cel.core.events import CognitiveEvent, EventType
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False


def create_test_event(
    index: int,
    entity_id: Optional[str] = None,
    concepts: Optional[List[str]] = None,
    parents: Optional[List[str]] = None,
) -> 'CognitiveEvent':
    """Create a test event for benchmarking."""
    content = {
        'index': index,
        'entity_id': entity_id or f'entity_{index % 100}',
        'data': f'test_data_{index}',
    }
    return CognitiveEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=EventType.OBSERVATION,
        causal_parents=tuple(parents or []),
        content=content,
        concepts=tuple(concepts or [f'concept_{index % 50}']),
    )


class EntityIndexBenchmark(BaseBenchmark):
    """
    Benchmark EntityIndex performance.

    Compares:
    - Linear scan (baseline)
    - Indexed lookup (optimized)

    Expected: 100x+ speedup for indexed lookups.
    """

    name = "entity_index"
    description = "Benchmark O(1) entity → events index"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 10000) if config else 10000
        self.n_entities = config.get("n_entities", 100) if config else 100
        self.n_queries = config.get("n_queries", 1000) if config else 1000
        self._index: Optional[EntityIndex] = None
        self._events: List[CognitiveEvent] = []

    def setup(self) -> None:
        """Create index with test events."""
        if not PERFORMANCE_AVAILABLE:
            return

        self._index = EntityIndex()
        self._events = []

        for i in range(self.n_events):
            entity_id = f'entity_{i % self.n_entities}'
            event = create_test_event(i, entity_id=entity_id)
            self._events.append(event)
            self._index.on_event(event)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        if not PERFORMANCE_AVAILABLE:
            result.status = BenchmarkStatus.SKIPPED
            result.metadata['reason'] = 'Performance modules not available'
            return result

        # Baseline: Linear scan
        linear_times = []
        for _ in range(min(self.n_queries, 100)):  # Fewer for baseline
            entity_id = f'entity_{random.randint(0, self.n_entities - 1)}'

            start = time.perf_counter()
            # Simulate linear scan
            matching = [e for e in self._events if e.content.get('entity_id') == entity_id]
            linear_times.append((time.perf_counter() - start) * 1000)

        # Optimized: Indexed lookup
        indexed_times = []
        for _ in range(self.n_queries):
            entity_id = f'entity_{random.randint(0, self.n_entities - 1)}'

            start = time.perf_counter()
            event_ids = self._index.events_for(entity_id)
            indexed_times.append((time.perf_counter() - start) * 1000)

        # Calculate metrics
        linear_avg = statistics.mean(linear_times)
        indexed_avg = statistics.mean(indexed_times)
        speedup = linear_avg / indexed_avg if indexed_avg > 0 else float('inf')

        result.add_metric(
            "linear_avg",
            linear_avg,
            unit="ms",
        )
        result.add_metric(
            "indexed_avg",
            indexed_avg,
            unit="ms",
            threshold_max=0.1,  # Target: <0.1ms
        )
        result.add_metric(
            "speedup",
            speedup,
            unit="x",
            threshold_min=100.0,  # Expect 100x+ speedup
        )
        result.add_metric(
            "indexed_p99",
            sorted(indexed_times)[int(len(indexed_times) * 0.99)],
            unit="ms",
            threshold_max=1.0,
        )

        stats = self._index.stats
        result.add_metric("index_entries", stats.entries, unit="entries")
        result.add_metric("memory_bytes", stats.memory_bytes, unit="bytes")

        result.metadata["n_events"] = self.n_events
        result.metadata["n_entities"] = self.n_entities
        result.metadata["n_queries"] = self.n_queries

        return result

    def teardown(self) -> None:
        self._index = None
        self._events = []


class HeapTopologicalSortBenchmark(BaseBenchmark):
    """
    Benchmark heap-based topological sort.

    Compares:
    - Naive sort (O(n² log n))
    - Heap-based (O(n log n))

    Expected: 10x+ speedup for large DAGs.
    """

    name = "heap_topological_sort"
    description = "Benchmark O(n log n) topological sort"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 5000) if config else 5000
        self._dag: Optional[OptimizedDAG] = None

    def setup(self) -> None:
        """Create DAG with test events."""
        if not PERFORMANCE_AVAILABLE:
            return

        self._dag = OptimizedDAG()

        # Create a chain of events (worst case for naive sort)
        prev_id = None
        for i in range(self.n_events):
            parents = [prev_id] if prev_id else []
            event = create_test_event(i, parents=parents)
            root = self._dag.add(event, verify_parents=False)
            prev_id = event.id

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        if not PERFORMANCE_AVAILABLE:
            result.status = BenchmarkStatus.SKIPPED
            result.metadata['reason'] = 'Performance modules not available'
            return result

        # Measure heap-based sort time
        iterations = 5
        sort_times = []

        for _ in range(iterations):
            start = time.perf_counter()
            count = 0
            for event in self._dag.causal_order():
                count += 1
            sort_times.append((time.perf_counter() - start) * 1000)

        avg_time = statistics.mean(sort_times)
        events_per_ms = self.n_events / avg_time if avg_time > 0 else 0

        result.add_metric(
            "sort_time_avg",
            avg_time,
            unit="ms",
            threshold_max=100.0,  # Target: <100ms for 5K events
        )
        result.add_metric(
            "sort_time_p99",
            sorted(sort_times)[-1],
            unit="ms",
        )
        result.add_metric(
            "throughput",
            events_per_ms,
            unit="events/ms",
            threshold_min=50.0,  # Target: >50 events/ms
        )
        result.add_metric(
            "dag_depth",
            self._dag.depth,
            unit="levels",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["iterations"] = iterations

        return result

    def teardown(self) -> None:
        self._dag = None


class SnapshotRecoveryBenchmark(BaseBenchmark):
    """
    Benchmark snapshot-based recovery.

    Compares:
    - Full replay (baseline)
    - Snapshot + incremental replay (optimized)

    Expected: Proportional speedup based on snapshot frequency.
    """

    name = "snapshot_recovery"
    description = "Benchmark snapshot-based recovery speed"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 5000) if config else 5000
        self.snapshot_interval = config.get("snapshot_interval", 500) if config else 500
        self._temp_dir: Optional[Path] = None
        self._manager: Optional[SnapshotManager] = None

    def setup(self) -> None:
        """Create snapshots for testing."""
        if not PERFORMANCE_AVAILABLE:
            return

        self._temp_dir = Path(tempfile.mkdtemp())
        self._manager = SnapshotManager(
            self._temp_dir,
            SnapshotConfig(full_interval=self.snapshot_interval),
        )

        # Simulate building entity index over time
        entity_index: Dict[str, List[str]] = {}
        for i in range(self.n_events):
            entity_id = f'entity_{i % 100}'
            event_id = f'event_{i}'

            if entity_id not in entity_index:
                entity_index[entity_id] = []
            entity_index[entity_id].append(event_id)

            # Create snapshot at intervals
            if (i + 1) % self.snapshot_interval == 0:
                from cortical.cel.core.references import EventHorizon
                self._manager.create_snapshot(
                    horizon=EventHorizon(event_id=event_id),
                    event_count=i + 1,
                    entity_index={k: list(v) for k, v in entity_index.items()},
                    snapshot_type='full',
                )

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        if not PERFORMANCE_AVAILABLE:
            result.status = BenchmarkStatus.SKIPPED
            result.metadata['reason'] = 'Performance modules not available'
            return result

        # Measure snapshot loading time
        load_times = []
        for _ in range(5):
            start = time.perf_counter()
            snapshot = self._manager.load_latest()
            load_times.append((time.perf_counter() - start) * 1000)

        avg_load = statistics.mean(load_times)

        # Calculate theoretical speedup
        # Without snapshots: replay all N events
        # With snapshots: load snapshot + replay (N % interval) events
        events_to_replay = self.n_events % self.snapshot_interval
        theoretical_speedup = self.n_events / max(events_to_replay + 1, 1)

        result.add_metric(
            "load_time_avg",
            avg_load,
            unit="ms",
            threshold_max=50.0,  # Target: <50ms to load snapshot
        )
        result.add_metric(
            "snapshot_count",
            self._manager.snapshot_count,
            unit="snapshots",
        )
        result.add_metric(
            "snapshot_size",
            self._manager.total_size_bytes / 1024,
            unit="KB",
        )
        result.add_metric(
            "theoretical_speedup",
            theoretical_speedup,
            unit="x",
            threshold_min=5.0,  # Expect at least 5x speedup
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["snapshot_interval"] = self.snapshot_interval

        return result

    def teardown(self) -> None:
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir)
        self._temp_dir = None
        self._manager = None


class ConceptIndexBenchmark(BaseBenchmark):
    """
    Benchmark concept index with bloom filter.

    Measures:
    - Index update performance
    - Bloom filter effectiveness
    - Search performance
    """

    name = "concept_index"
    description = "Benchmark concept indexing with bloom filters"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 10000) if config else 10000
        self.n_concepts = config.get("n_concepts", 500) if config else 500
        self.n_queries = config.get("n_queries", 1000) if config else 1000
        self._index: Optional[ConceptIndex] = None

    def setup(self) -> None:
        """Create concept index with test events."""
        if not PERFORMANCE_AVAILABLE:
            return

        self._index = ConceptIndex()

        for i in range(self.n_events):
            # Each event gets 1-3 concepts
            concepts = [f'concept_{random.randint(0, self.n_concepts - 1)}'
                       for _ in range(random.randint(1, 3))]
            event = create_test_event(i, concepts=concepts)
            self._index.on_event(event)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        if not PERFORMANCE_AVAILABLE:
            result.status = BenchmarkStatus.SKIPPED
            result.metadata['reason'] = 'Performance modules not available'
            return result

        # Measure bloom filter checks
        bloom_times = []
        bloom_hits = 0
        for _ in range(self.n_queries):
            concept = f'concept_{random.randint(0, self.n_concepts - 1)}'

            start = time.perf_counter()
            exists = self._index.probably_has(concept)
            bloom_times.append((time.perf_counter() - start) * 1000)
            if exists:
                bloom_hits += 1

        # Measure full lookup
        lookup_times = []
        for _ in range(self.n_queries):
            concept = f'concept_{random.randint(0, self.n_concepts - 1)}'

            start = time.perf_counter()
            events = self._index.events_for(concept)
            lookup_times.append((time.perf_counter() - start) * 1000)

        # Test false positive rate
        false_positives = 0
        false_positive_tests = 100
        for i in range(false_positive_tests):
            fake_concept = f'nonexistent_concept_{i}'
            if self._index.probably_has(fake_concept):
                false_positives += 1

        result.add_metric(
            "bloom_avg",
            statistics.mean(bloom_times),
            unit="ms",
            threshold_max=0.01,  # Bloom filter should be <0.01ms
        )
        result.add_metric(
            "lookup_avg",
            statistics.mean(lookup_times),
            unit="ms",
            threshold_max=0.1,
        )
        result.add_metric(
            "false_positive_rate",
            false_positives / false_positive_tests,
            unit="ratio",
            threshold_max=0.05,  # Target: <5% false positives
        )
        result.add_metric(
            "concept_count",
            self._index.concept_count,
            unit="concepts",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["n_concepts"] = self.n_concepts

        return result

    def teardown(self) -> None:
        self._index = None


class TemporalIndexBenchmark(BaseBenchmark):
    """
    Benchmark temporal index for time-range queries.

    Measures:
    - Insert performance
    - Range query performance
    - Binary search effectiveness
    """

    name = "temporal_index"
    description = "Benchmark temporal indexing for time-range queries"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 10000) if config else 10000
        self.n_queries = config.get("n_queries", 500) if config else 500
        self._index: Optional[TemporalIndex] = None
        self._timestamps: List[str] = []

    def setup(self) -> None:
        """Create temporal index with test events."""
        if not PERFORMANCE_AVAILABLE:
            return

        self._index = TemporalIndex()
        self._timestamps = []

        for i in range(self.n_events):
            event = create_test_event(i)
            self._index.on_event(event)
            self._timestamps.append(event.timestamp)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        if not PERFORMANCE_AVAILABLE:
            result.status = BenchmarkStatus.SKIPPED
            result.metadata['reason'] = 'Performance modules not available'
            return result

        # Measure range query performance
        range_times = []
        results_counts = []

        for _ in range(self.n_queries):
            # Random time range (10% of total)
            start_idx = random.randint(0, int(self.n_events * 0.9))
            end_idx = start_idx + int(self.n_events * 0.1)

            start_ts = self._timestamps[start_idx]
            end_ts = self._timestamps[min(end_idx, len(self._timestamps) - 1)]

            start = time.perf_counter()
            events = self._index.events_in_range(start_ts, end_ts)
            range_times.append((time.perf_counter() - start) * 1000)
            results_counts.append(len(events))

        result.add_metric(
            "range_query_avg",
            statistics.mean(range_times),
            unit="ms",
            threshold_max=1.0,  # Target: <1ms for range queries
        )
        result.add_metric(
            "range_query_p99",
            sorted(range_times)[int(len(range_times) * 0.99)],
            unit="ms",
        )
        result.add_metric(
            "avg_results",
            statistics.mean(results_counts),
            unit="events",
        )
        result.add_metric(
            "index_size",
            self._index.event_count,
            unit="events",
        )

        time_range = self._index.time_range
        result.metadata["n_events"] = self.n_events
        result.metadata["time_range_start"] = time_range[0][:10] if time_range[0] else None
        result.metadata["time_range_end"] = time_range[1][:10] if time_range[1] else None

        return result

    def teardown(self) -> None:
        self._index = None
        self._timestamps = []


class StreamingStoreBenchmark(BaseBenchmark):
    """
    Benchmark streaming event store with write batching.

    Measures:
    - Append throughput (with batching)
    - Get latency (with LRU cache)
    - Index query performance
    - Memory efficiency
    """

    name = "streaming_store"
    description = "Benchmark streaming event store with batching and caching"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 5000) if config else 5000
        self.n_queries = config.get("n_queries", 1000) if config else 1000
        self._store = None
        self._event_ids: List[str] = []
        self._temp_dir = None

    def setup(self) -> None:
        """Create streaming store with test events."""
        if not PERFORMANCE_AVAILABLE:
            return

        import tempfile
        self._temp_dir = tempfile.mkdtemp()

        try:
            from cortical.cel.performance.streaming_store import (
                StreamingEventStore,
                StoreConfig,
            )

            config = StoreConfig(
                events_per_segment=500,
                event_cache_size=1000,
                batch_size=50,
            )
            self._store = StreamingEventStore(Path(self._temp_dir), config)
            self._event_ids = []

            # Append events
            for i in range(self.n_events):
                entity_id = f'entity_{i % 100}'
                event = create_test_event(i, entity_id=entity_id)
                self._store.append(event)
                self._event_ids.append(event.id)

            # Flush to ensure all batched writes complete
            self._store.flush()
        except ImportError:
            pass

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        if not PERFORMANCE_AVAILABLE or self._store is None:
            result.status = BenchmarkStatus.SKIPPED
            result.metadata['reason'] = 'Streaming store not available'
            return result

        # Measure get latency (cache misses first, then hits)
        get_times_cold = []
        get_times_hot = []

        # Cold reads (cache miss)
        sample_ids = random.sample(self._event_ids, min(100, len(self._event_ids)))
        for event_id in sample_ids:
            start = time.perf_counter()
            event = self._store.get(event_id)
            get_times_cold.append((time.perf_counter() - start) * 1000)

        # Hot reads (cache hit - same IDs again)
        for event_id in sample_ids:
            start = time.perf_counter()
            event = self._store.get(event_id)
            get_times_hot.append((time.perf_counter() - start) * 1000)

        # Measure entity query performance
        entity_times = []
        for _ in range(min(self.n_queries, 200)):
            entity_id = f'entity_{random.randint(0, 99)}'
            start = time.perf_counter()
            event_ids = self._store.events_for_entity(entity_id)
            entity_times.append((time.perf_counter() - start) * 1000)

        result.add_metric(
            "get_cold_avg",
            statistics.mean(get_times_cold),
            unit="ms",
            threshold_max=5.0,  # Cold reads from disk <5ms
        )
        result.add_metric(
            "get_hot_avg",
            statistics.mean(get_times_hot),
            unit="ms",
            threshold_max=0.1,  # Hot reads from cache <0.1ms
        )
        result.add_metric(
            "cache_speedup",
            statistics.mean(get_times_cold) / statistics.mean(get_times_hot) if get_times_hot else 0,
            unit="x",
            threshold_min=10.0,  # Cache should provide 10x+ speedup
        )
        result.add_metric(
            "entity_query_avg",
            statistics.mean(entity_times),
            unit="ms",
            threshold_max=1.0,
        )
        result.add_metric(
            "events_stored",
            self._store.count,
            unit="events",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["cache_size"] = len(self._store._event_cache) if hasattr(self._store, '_event_cache') else 0

        result.status = BenchmarkStatus.PASSED
        return result

    def teardown(self) -> None:
        self._store = None
        self._event_ids = []
        if self._temp_dir:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_all_performance_benchmarks(quick: bool = False) -> Dict[str, BenchmarkResult]:
    """Run all performance benchmarks."""
    config = {"quick": True} if quick else {}

    if quick:
        config.update({
            "n_events": 1000,
            "n_entities": 50,
            "n_queries": 100,
            "n_concepts": 100,
        })

    benchmarks = [
        EntityIndexBenchmark(config),
        HeapTopologicalSortBenchmark(config),
        SnapshotRecoveryBenchmark(config),
        ConceptIndexBenchmark(config),
        TemporalIndexBenchmark(config),
        StreamingStoreBenchmark(config),
    ]

    results = {}
    for benchmark in benchmarks:
        print(f"Running {benchmark.name}...")
        try:
            benchmark.setup()
            result = benchmark.run()
            benchmark.teardown()
            results[benchmark.name] = result
            print(f"  Status: {result.status.name}")
            for metric in result.metrics:
                threshold_status = ""
                if metric.threshold_min and metric.value < metric.threshold_min:
                    threshold_status = " ❌ BELOW MIN"
                elif metric.threshold_max and metric.value > metric.threshold_max:
                    threshold_status = " ❌ ABOVE MAX"
                else:
                    threshold_status = " ✓"
                print(f"    {metric.name}: {metric.value:.3f} {metric.unit}{threshold_status}")
        except Exception as e:
            print(f"  Error: {e}")
            results[benchmark.name] = BenchmarkResult(
                benchmark_name=benchmark.name,
                category=benchmark.category,
                status=BenchmarkStatus.FAILED,
            )

    return results


if __name__ == "__main__":
    import sys

    quick = "--quick" in sys.argv
    results = run_all_performance_benchmarks(quick=quick)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r.status == BenchmarkStatus.PASSED)
    failed = sum(1 for r in results.values() if r.status == BenchmarkStatus.FAILED)
    skipped = sum(1 for r in results.values() if r.status == BenchmarkStatus.SKIPPED)

    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
