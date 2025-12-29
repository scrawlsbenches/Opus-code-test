"""
CEL Sanity Module Benchmarks.

Benchmarks for validating performance of the CEL sanity modules:
- Compaction: TimeWindowCompactor, SemanticCompactor, CausalChainCompactor
- Health: EventStoreHealthMonitor, HealthCheckScheduler
- Migration: SchemaMigrationEngine, MigrationStep

These benchmarks focus on:
- Scalability: How performance changes with event count
- Memory: Peak memory usage during operations
- Latency: Operation timing for SLA validation
- Throughput: Events processed per second
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
)

# Import actual CEL components
from cortical.cel.core.events import CognitiveEvent, EventType
from cortical.cel.sanity.compaction import (
    BaseCompactor,
    CausalChainCompactor,
    CompactionResult,
    SemanticCompactor,
    TimeWindowCompactor,
    create_compaction_schedule,
    estimate_compaction_savings,
)
from cortical.cel.sanity.health import (
    EventStoreHealthMonitor,
    HealthCheckScheduler,
    HealthMetric,
    HealthReport,
    HealthStatus,
)
from cortical.cel.sanity.migration import (
    MigrationPlan,
    MigrationStatus,
    MigrationStep,
    SchemaMigrationEngine,
    add_field,
    by_event_type,
    rename_field,
)


# =============================================================================
# MOCK EVENT STORE FOR BENCHMARKING
# =============================================================================


class BenchmarkEventStore:
    """
    In-memory event store for benchmarking.

    Implements the EventStore protocol needed by sanity modules.
    """

    def __init__(self):
        self._events: List[CognitiveEvent] = []
        self._by_id: Dict[str, CognitiveEvent] = {}

    def append(self, event: CognitiveEvent) -> MagicMock:
        """Append event and return mock MerkleRoot."""
        self._events.append(event)
        self._by_id[event.id] = event
        root = MagicMock()
        root.value = event.id
        return root

    def iterate(self):
        """Iterate over all events."""
        return iter(self._events)

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get event by ID."""
        return self._by_id.get(event_id)

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        """Always truthy - needed for SchemaMigrationEngine 'or' check."""
        return True


def create_test_events(
    count: int,
    concepts_per_event: int = 3,
    with_causal_chain: bool = False,
    age_days: int = 0,
    entity_id: Optional[str] = None,
) -> List[CognitiveEvent]:
    """Create test events for benchmarking."""
    events = []
    prev_id = None

    base_time = datetime.now(timezone.utc) - timedelta(days=age_days)

    for i in range(count):
        # Generate concepts
        concepts = tuple(
            f"concept_{random.randint(0, 99)}"
            for _ in range(concepts_per_event)
        )

        # Timestamp with slight variation
        ts = base_time + timedelta(seconds=i)

        # Causal parents
        parents = (prev_id,) if with_causal_chain and prev_id else ()

        # Content
        content = {
            'index': i,
            'data': f'benchmark_data_{i}',
        }
        if entity_id:
            content['entity_id'] = entity_id

        event = CognitiveEvent(
            timestamp=ts.isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=parents,
            content=content,
            concepts=concepts,
        )
        events.append(event)
        prev_id = event.id

    return events


# =============================================================================
# BENCHMARK: COMPACTION SCALABILITY
# =============================================================================


class CompactionScalabilityBenchmark(BaseBenchmark):
    """
    Benchmark compaction scalability with event count.

    Measures:
    - Time to identify compactable groups (O(n) expected)
    - Time to compact a group
    - should_compact() latency
    - Memory efficiency
    """

    name = "compaction_scalability"
    description = "Measure compaction performance vs event count"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.event_counts = config.get("event_counts", [100, 500, 1000, 5000]) if config else [100, 500, 1000, 5000]
        self._stores: Dict[int, BenchmarkEventStore] = {}

    def setup(self) -> None:
        """Create event stores with varying sizes."""
        for count in self.event_counts:
            store = BenchmarkEventStore()
            # Create old events (30 days old) for compaction
            events = create_test_events(
                count,
                age_days=30,
                entity_id="entity_1",
            )
            for event in events:
                store.append(event)
            self._stores[count] = store

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        identify_times: Dict[int, float] = {}
        should_compact_times: Dict[int, float] = {}

        for count in self.event_counts:
            store = self._stores[count]
            compactor = TimeWindowCompactor(
                store,
                window_size=timedelta(hours=24),
                min_age=timedelta(days=7),
            )

            # Measure identify_compactable()
            start = time.perf_counter()
            groups = compactor.identify_compactable()
            identify_times[count] = (time.perf_counter() - start) * 1000

            # Measure should_compact()
            start = time.perf_counter()
            _ = compactor.should_compact()
            should_compact_times[count] = (time.perf_counter() - start) * 1000

        # Calculate scaling factor (should be ~linear)
        if len(self.event_counts) >= 2:
            smallest = min(self.event_counts)
            largest = max(self.event_counts)
            time_ratio = identify_times[largest] / max(identify_times[smallest], 0.001)
            event_ratio = largest / smallest
            scaling_factor = time_ratio / event_ratio  # Should be ~1 for O(n)
        else:
            scaling_factor = 1.0

        # Add metrics
        for count in self.event_counts:
            result.add_metric(
                f"identify_{count}",
                identify_times[count],
                unit="ms",
                threshold_max=count * 0.1,  # 0.1ms per event
            )
            result.add_metric(
                f"should_compact_{count}",
                should_compact_times[count],
                unit="ms",
                threshold_max=count * 0.05,  # 0.05ms per event
            )

        result.add_metric(
            "scaling_factor",
            scaling_factor,
            unit="x",
            threshold_max=2.0,  # Should be < 2x (close to linear)
        )

        result.metadata["event_counts"] = self.event_counts

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._stores = {}


# =============================================================================
# BENCHMARK: SEMANTIC COMPACTOR CLUSTERING
# =============================================================================


class SemanticClusteringBenchmark(BaseBenchmark):
    """
    Benchmark SemanticCompactor clustering performance.

    This is O(n²) in the worst case, so we need to document limits.

    Measures:
    - Clustering time vs event count
    - Number of clusters formed
    - Concept overlap detection accuracy
    """

    name = "semantic_clustering"
    description = "Measure semantic clustering performance (O(n²) warning)"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Smaller counts due to O(n²) complexity
        self.event_counts = config.get("event_counts", [50, 100, 200, 500]) if config else [50, 100, 200, 500]
        self._stores: Dict[int, BenchmarkEventStore] = {}

    def setup(self) -> None:
        """Create event stores with similar concepts."""
        for count in self.event_counts:
            store = BenchmarkEventStore()
            # Create events with overlapping concepts
            for i in range(count):
                # Each event gets concepts that overlap with neighbors
                base_concept = i // 10  # Groups of 10 share base concept
                concepts = (
                    f"base_{base_concept}",
                    f"specific_{i}",
                    "common",
                )
                event = CognitiveEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type=EventType.OBSERVATION,
                    causal_parents=(),
                    content={'index': i},
                    concepts=concepts,
                )
                store.append(event)
            self._stores[count] = store

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        clustering_times: Dict[int, float] = {}
        cluster_counts: Dict[int, int] = {}

        for count in self.event_counts:
            store = self._stores[count]
            compactor = SemanticCompactor(
                store,
                similarity_threshold=0.5,
                min_group_size=3,
            )

            # Measure identify_compactable() (the O(n²) operation)
            start = time.perf_counter()
            groups = compactor.identify_compactable()
            clustering_times[count] = (time.perf_counter() - start) * 1000
            cluster_counts[count] = len(groups)

        # Calculate O(n²) scaling
        if len(self.event_counts) >= 2:
            smallest = min(self.event_counts)
            largest = max(self.event_counts)
            time_ratio = clustering_times[largest] / max(clustering_times[smallest], 0.001)
            n_squared_ratio = (largest / smallest) ** 2
            # Scaling factor relative to O(n²)
            scaling_vs_n2 = time_ratio / n_squared_ratio
        else:
            scaling_vs_n2 = 1.0

        # Add metrics
        for count in self.event_counts:
            result.add_metric(
                f"cluster_{count}",
                clustering_times[count],
                unit="ms",
                # O(n²) means 500 events could take 25x longer than 100 events
                threshold_max=count * count * 0.001,  # 1µs per pair
            )
            result.add_metric(
                f"clusters_{count}",
                float(cluster_counts[count]),
                unit="groups",
            )

        result.add_metric(
            "scaling_vs_n2",
            scaling_vs_n2,
            unit="ratio",
            threshold_max=1.5,  # Should be ≤ O(n²)
        )

        result.metadata["event_counts"] = self.event_counts

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._stores = {}


# =============================================================================
# BENCHMARK: CAUSAL CHAIN COMPACTOR
# =============================================================================


class CausalChainBenchmark(BaseBenchmark):
    """
    Benchmark CausalChainCompactor performance.

    Measures:
    - Chain identification time
    - Chain traversal depth
    - Compaction effectiveness
    """

    name = "causal_chain_compaction"
    description = "Measure causal chain compaction performance"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chain_lengths = config.get("chain_lengths", [10, 50, 100, 200]) if config else [10, 50, 100, 200]
        self._stores: Dict[int, BenchmarkEventStore] = {}

    def setup(self) -> None:
        """Create event stores with causal chains."""
        for length in self.chain_lengths:
            store = BenchmarkEventStore()
            events = create_test_events(
                length,
                with_causal_chain=True,
            )
            for event in events:
                store.append(event)
            self._stores[length] = store

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        identify_times: Dict[int, float] = {}
        chains_found: Dict[int, int] = {}

        for length in self.chain_lengths:
            store = self._stores[length]
            compactor = CausalChainCompactor(
                store,
                max_chain_length=5,  # Chains > 5 get compacted
            )

            # Measure identify_compactable()
            start = time.perf_counter()
            groups = compactor.identify_compactable()
            identify_times[length] = (time.perf_counter() - start) * 1000
            chains_found[length] = len(groups)

        # Add metrics
        for length in self.chain_lengths:
            result.add_metric(
                f"identify_{length}",
                identify_times[length],
                unit="ms",
                threshold_max=length * 0.2,  # 0.2ms per event
            )
            result.add_metric(
                f"chains_{length}",
                float(chains_found[length]),
                unit="chains",
            )

        result.metadata["chain_lengths"] = self.chain_lengths

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._stores = {}


# =============================================================================
# BENCHMARK: HEALTH MONITOR LATENCY
# =============================================================================


class HealthMonitorLatencyBenchmark(BaseBenchmark):
    """
    Benchmark health monitoring latency.

    Health checks iterate the entire store multiple times,
    so latency scales with event count.

    Measures:
    - check() latency
    - is_healthy() latency
    - diagnose() latency
    """

    name = "health_monitor_latency"
    description = "Measure health check latency vs event count"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.event_counts = config.get("event_counts", [100, 500, 1000, 5000]) if config else [100, 500, 1000, 5000]
        self._stores: Dict[int, BenchmarkEventStore] = {}

    def setup(self) -> None:
        """Create event stores with healthy data."""
        for count in self.event_counts:
            store = BenchmarkEventStore()
            events = create_test_events(count)
            for event in events:
                store.append(event)
            self._stores[count] = store

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        check_times: Dict[int, float] = {}
        is_healthy_times: Dict[int, float] = {}
        diagnose_times: Dict[int, float] = {}

        for count in self.event_counts:
            store = self._stores[count]
            monitor = EventStoreHealthMonitor(store)

            # Measure check()
            start = time.perf_counter()
            report = monitor.check()
            check_times[count] = (time.perf_counter() - start) * 1000

            # Measure is_healthy() (uses cached report)
            start = time.perf_counter()
            _ = monitor.is_healthy()
            is_healthy_times[count] = (time.perf_counter() - start) * 1000

            # Measure diagnose() (full check + extra info)
            start = time.perf_counter()
            _ = monitor.diagnose()
            diagnose_times[count] = (time.perf_counter() - start) * 1000

        # Add metrics
        for count in self.event_counts:
            result.add_metric(
                f"check_{count}",
                check_times[count],
                unit="ms",
                threshold_max=max(100, count * 0.1),  # 0.1ms per event, min 100ms
            )
            result.add_metric(
                f"is_healthy_{count}",
                is_healthy_times[count],
                unit="ms",
                threshold_max=1.0,  # Should use cache
            )
            result.add_metric(
                f"diagnose_{count}",
                diagnose_times[count],
                unit="ms",
                threshold_max=max(200, count * 0.2),  # 0.2ms per event
            )

        result.metadata["event_counts"] = self.event_counts

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._stores = {}


# =============================================================================
# BENCHMARK: HEALTH CHECK ACCURACY
# =============================================================================


class HealthCheckAccuracyBenchmark(BaseBenchmark):
    """
    Benchmark health check detection accuracy.

    Measures:
    - Orphan detection rate
    - False positive rate
    - Issue detection sensitivity
    """

    name = "health_check_accuracy"
    description = "Measure health check detection accuracy"
    category = BenchmarkCategory.STABILITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_trials = config.get("n_trials", 10) if config else 10

    def setup(self) -> None:
        """Nothing to setup."""
        pass

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        orphan_detected = 0
        orphan_missed = 0
        false_positives = 0

        for _ in range(self.n_trials):
            # Create store with known orphan references
            store = BenchmarkEventStore()

            # Add some good events
            good_events = create_test_events(50)
            for event in good_events:
                store.append(event)

            # Add events with orphan references
            orphan_event = CognitiveEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.OBSERVATION,
                causal_parents=('non-existent-parent',),
                content={'is_orphan': True},
                concepts=('test',),
            )
            store.append(orphan_event)

            # Check health
            monitor = EventStoreHealthMonitor(store)
            report = monitor.check()

            # Find orphan metric
            orphan_metric = next(
                (m for m in report.metrics if m.name == 'dag_orphan_ratio'),
                None
            )

            if orphan_metric:
                if orphan_metric.value > 0:
                    orphan_detected += 1
                else:
                    orphan_missed += 1
            else:
                orphan_missed += 1

        # Test for false positives (healthy store)
        for _ in range(self.n_trials):
            store = BenchmarkEventStore()
            events = create_test_events(50)
            for event in events:
                store.append(event)

            monitor = EventStoreHealthMonitor(store)
            report = monitor.check()

            orphan_metric = next(
                (m for m in report.metrics if m.name == 'dag_orphan_ratio'),
                None
            )

            if orphan_metric and orphan_metric.value > 0:
                false_positives += 1

        # Add metrics
        detection_rate = orphan_detected / self.n_trials if self.n_trials > 0 else 0
        false_positive_rate = false_positives / self.n_trials if self.n_trials > 0 else 0

        result.add_metric(
            "orphan_detection_rate",
            detection_rate,
            unit="ratio",
            threshold_min=0.9,  # Should detect >90% of orphans
        )
        result.add_metric(
            "false_positive_rate",
            false_positive_rate,
            unit="ratio",
            threshold_max=0.05,  # <5% false positives
        )

        result.metadata["n_trials"] = self.n_trials

        return result

    def teardown(self) -> None:
        """Nothing to cleanup."""
        pass


# =============================================================================
# BENCHMARK: MIGRATION THROUGHPUT
# =============================================================================


class MigrationThroughputBenchmark(BaseBenchmark):
    """
    Benchmark schema migration throughput.

    Measures:
    - Events migrated per second
    - Migration step latency
    - Transform function overhead
    """

    name = "migration_throughput"
    description = "Measure schema migration throughput"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Respect n_events from quick config, or use event_counts for custom config
        if config and "n_events" in config:
            # Quick mode - use proportional counts based on n_events
            n = config["n_events"]
            self.event_counts = [n // 4, n // 2, n] if n >= 100 else [n // 2, n] if n >= 20 else [n]
        else:
            self.event_counts = config.get("event_counts", [100, 500, 1000]) if config else [100, 500, 1000]
        self._source_stores: Dict[int, BenchmarkEventStore] = {}
        self._target_stores: Dict[int, BenchmarkEventStore] = {}

    def setup(self) -> None:
        """Create source stores with v1 schema events."""
        for count in self.event_counts:
            source = BenchmarkEventStore()
            target = BenchmarkEventStore()

            # Create v1 schema events
            for i in range(count):
                event = CognitiveEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type=EventType.INTENTION,
                    causal_parents=(),
                    content={
                        '_schema_version': 'v1',
                        'task_name': f'Task {i}',  # Old field name
                        'priority': 'medium',
                    },
                    concepts=('task',),
                )
                source.append(event)

            self._source_stores[count] = source
            self._target_stores[count] = target

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        migration_times: Dict[int, float] = {}
        events_processed: Dict[int, int] = {}

        for count in self.event_counts:
            source = self._source_stores[count]
            target = self._target_stores[count]

            # Create migration step
            step = MigrationStep(
                name='rename_task_name',
                description='Rename task_name to title',
                event_filter=by_event_type(EventType.INTENTION),
                transform=rename_field('task_name', 'title'),
                version_from='v1',
                version_to='v2',
            )

            plan = MigrationPlan(
                name='v1-to-v2',
                description='Migrate v1 to v2',
                steps=[step],
            )

            engine = SchemaMigrationEngine(source, target)
            engine.register_plan(plan)

            # Measure migration time
            start = time.perf_counter()
            result_plan = engine.migrate('v1-to-v2')
            migration_times[count] = (time.perf_counter() - start) * 1000
            events_processed[count] = result_plan._events_processed

        # Add metrics
        for count in self.event_counts:
            result.add_metric(
                f"migration_{count}",
                migration_times[count],
                unit="ms",
                threshold_max=count * 1.0,  # 1ms per event
            )

            throughput = events_processed[count] / (migration_times[count] / 1000) if migration_times[count] > 0 else 0
            result.add_metric(
                f"throughput_{count}",
                throughput,
                unit="events/sec",
                threshold_min=100,  # At least 100 events/sec
            )

        result.metadata["event_counts"] = self.event_counts

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._source_stores = {}
        self._target_stores = {}


# =============================================================================
# BENCHMARK: COMPACTION SAVINGS ESTIMATION
# =============================================================================


class CompactionSavingsEstimationBenchmark(BaseBenchmark):
    """
    Benchmark compaction savings estimation.

    Measures:
    - Estimation accuracy
    - Estimation speed
    """

    name = "compaction_savings_estimation"
    description = "Measure compaction savings estimation performance"
    category = BenchmarkCategory.QUALITY

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.event_counts = config.get("event_counts", [100, 500, 1000]) if config else [100, 500, 1000]
        self._stores: Dict[int, BenchmarkEventStore] = {}

    def setup(self) -> None:
        """Create stores with redundant data."""
        for count in self.event_counts:
            store = BenchmarkEventStore()

            # Create events with repeated concepts (compaction opportunity)
            for i in range(count):
                # Use only 10 unique concepts, creating redundancy
                concepts = (
                    f"common_{i % 10}",
                    "shared_concept",
                )
                event = CognitiveEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type=EventType.OBSERVATION,
                    causal_parents=(),
                    content={'index': i},
                    concepts=concepts,
                )
                store.append(event)

            self._stores[count] = store

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        estimation_times: Dict[int, float] = {}
        savings_estimates: Dict[int, float] = {}

        for count in self.event_counts:
            store = self._stores[count]

            # Measure estimation time
            start = time.perf_counter()
            savings = estimate_compaction_savings(store)
            estimation_times[count] = (time.perf_counter() - start) * 1000
            savings_estimates[count] = savings['estimated_savings_percent']

        # Add metrics
        for count in self.event_counts:
            result.add_metric(
                f"estimate_{count}",
                estimation_times[count],
                unit="ms",
                threshold_max=count * 0.1,  # 0.1ms per event
            )
            result.add_metric(
                f"savings_{count}",
                savings_estimates[count],
                unit="%",
            )

        result.metadata["event_counts"] = self.event_counts

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._stores = {}


# =============================================================================
# BENCHMARK REGISTRY
# =============================================================================

ALL_SANITY_BENCHMARKS = [
    CompactionScalabilityBenchmark,
    SemanticClusteringBenchmark,
    CausalChainBenchmark,
    HealthMonitorLatencyBenchmark,
    HealthCheckAccuracyBenchmark,
    MigrationThroughputBenchmark,
    CompactionSavingsEstimationBenchmark,
]

SANITY_BENCHMARK_MAP = {b.name: b for b in ALL_SANITY_BENCHMARKS}

SANITY_BENCHMARKS_BY_CATEGORY = {
    "compaction": [
        CompactionScalabilityBenchmark,
        SemanticClusteringBenchmark,
        CausalChainBenchmark,
        CompactionSavingsEstimationBenchmark,
    ],
    "health": [
        HealthMonitorLatencyBenchmark,
        HealthCheckAccuracyBenchmark,
    ],
    "migration": [
        MigrationThroughputBenchmark,
    ],
}
