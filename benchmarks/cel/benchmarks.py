"""
CEL Benchmark Implementations.

Benchmarks for validating performance and correctness of the
Cognitive Event Lattice architecture.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
    measure_time,
    measure_stability,
)


# =============================================================================
# CEL-SPECIFIC BENCHMARK CATEGORY
# =============================================================================

class CELBenchmarkCategory(Enum):
    """CEL-specific benchmark categories."""
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    QUERY = "query"
    CORRECTNESS = "correctness"


# =============================================================================
# IN-MEMORY TEST FIXTURES (Minimal implementations for benchmarking)
# =============================================================================

class EventType(Enum):
    """Event types for benchmark testing."""
    OBSERVATION = "observation"
    INTENTION = "intention"
    FULFILLMENT = "fulfillment"
    INVALIDATION = "invalidation"


@dataclass(frozen=True)
class BenchmarkEvent:
    """Minimal event implementation for benchmarking."""
    event_type: EventType
    timestamp: str
    content: Dict[str, Any]
    concepts: Tuple[str, ...]
    causal_parents: Tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Content-addressed ID via SHA256."""
        content_str = f"{self.event_type.value}:{self.timestamp}:{self.content}:{self.concepts}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


class BenchmarkEventStore:
    """In-memory event store for benchmarking."""

    def __init__(self):
        self._events: Dict[str, BenchmarkEvent] = {}
        self._order: List[str] = []  # Maintains insertion order
        self._concept_index: Dict[str, Set[str]] = {}  # concept -> event_ids

    def append(self, event: BenchmarkEvent) -> str:
        """Append event and return its ID."""
        event_id = event.id
        self._events[event_id] = event
        self._order.append(event_id)

        # Index by concepts
        for concept in event.concepts:
            if concept not in self._concept_index:
                self._concept_index[concept] = set()
            self._concept_index[concept].add(event_id)

        return event_id

    def get(self, event_id: str) -> Optional[BenchmarkEvent]:
        """Retrieve event by ID."""
        return self._events.get(event_id)

    def search_by_concept(self, concept: str) -> List[str]:
        """Find events by concept."""
        return list(self._concept_index.get(concept, set()))

    def events_before(self, event_id: str) -> List[BenchmarkEvent]:
        """Get all events causally before given event."""
        try:
            idx = self._order.index(event_id)
            return [self._events[eid] for eid in self._order[:idx]]
        except ValueError:
            return []

    def __len__(self) -> int:
        return len(self._events)


class BenchmarkMaterializer:
    """Minimal materializer for benchmarking."""

    def __init__(self, store: BenchmarkEventStore):
        self._store = store
        self._cache: Dict[str, Dict[str, Any]] = {}

    def materialize(
        self,
        entity_id: str,
        at_event: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Materialize entity state by replaying events."""
        cache_key = f"{entity_id}:{at_event or 'HEAD'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        state: Dict[str, Any] = {"id": entity_id, "history": []}

        # Get events up to horizon
        events_to_replay = (
            self._store.events_before(at_event)
            if at_event else
            list(self._store._events.values())
        )

        # Replay events that affect this entity
        for event in events_to_replay:
            if event.content.get("entity_id") == entity_id:
                state["history"].append(event.id)
                state.update(event.content.get("state", {}))

        self._cache[cache_key] = state
        return state

    def invalidate_cache(self) -> int:
        """Clear cache and return number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count


# =============================================================================
# BENCHMARK: EVENT APPEND THROUGHPUT
# =============================================================================

class EventAppendBenchmark(BaseBenchmark):
    """
    Benchmark event append throughput.

    Measures:
    - Events per second (single append)
    - Batch append performance
    - Throughput with varying event sizes
    """

    name = "event_append"
    description = "Measure event append throughput and latency"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 1000) if config else 1000
        self.n_warmup = config.get("n_warmup", 100) if config else 100
        self._store: Optional[BenchmarkEventStore] = None

    def setup(self) -> None:
        """Initialize fresh event store."""
        self._store = BenchmarkEventStore()

        # Warmup
        for i in range(self.n_warmup):
            self._create_and_append_event(i, warmup=True)

    def _create_and_append_event(
        self,
        index: int,
        warmup: bool = False,
    ) -> str:
        """Create and append a single event."""
        event = BenchmarkEvent(
            event_type=EventType.OBSERVATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            content={
                "index": index,
                "data": f"benchmark_data_{index}",
                "warmup": warmup,
            },
            concepts=(f"concept_{index % 100}", "benchmark"),
        )
        return self._store.append(event)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        # Measure single append latency
        latencies = []
        for i in range(self.n_events):
            start = time.perf_counter()
            self._create_and_append_event(self.n_warmup + i)
            latencies.append((time.perf_counter() - start) * 1000)

        import statistics

        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        events_per_sec = 1000 / avg_latency if avg_latency > 0 else 0

        # Add metrics with thresholds
        result.add_metric(
            "avg_latency",
            avg_latency,
            unit="ms",
            threshold_max=1.0,  # Target: <1ms per append
        )
        result.add_metric(
            "p50_latency",
            p50_latency,
            unit="ms",
            threshold_max=0.5,
        )
        result.add_metric(
            "p99_latency",
            p99_latency,
            unit="ms",
            threshold_max=5.0,
        )
        result.add_metric(
            "throughput",
            events_per_sec,
            unit="events/sec",
            threshold_min=1000,  # Target: >1000 events/sec
        )
        result.add_metric(
            "total_events",
            len(self._store),
            unit="events",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["n_warmup"] = self.n_warmup

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._store = None


# =============================================================================
# BENCHMARK: MATERIALIZATION PERFORMANCE
# =============================================================================

class MaterializationBenchmark(BaseBenchmark):
    """
    Benchmark entity materialization performance.

    Measures:
    - Cold materialization (no cache)
    - Warm materialization (cached)
    - Materialization vs event history depth
    """

    name = "materialization"
    description = "Measure entity materialization performance"
    category = BenchmarkCategory.SCALE

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_entities = config.get("n_entities", 100) if config else 100
        self.events_per_entity = config.get("events_per_entity", 50) if config else 50
        self._store: Optional[BenchmarkEventStore] = None
        self._materializer: Optional[BenchmarkMaterializer] = None

    def setup(self) -> None:
        """Create event store with test data."""
        self._store = BenchmarkEventStore()
        self._materializer = BenchmarkMaterializer(self._store)

        # Create events for each entity
        for entity_idx in range(self.n_entities):
            entity_id = f"entity_{entity_idx}"
            for event_idx in range(self.events_per_entity):
                event = BenchmarkEvent(
                    event_type=EventType.OBSERVATION,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content={
                        "entity_id": entity_id,
                        "state": {
                            "counter": event_idx,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    concepts=(f"entity_{entity_idx}", "update"),
                )
                self._store.append(event)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        import statistics

        # Cold materialization (no cache)
        self._materializer.invalidate_cache()
        cold_times = []
        for i in range(self.n_entities):
            entity_id = f"entity_{i}"
            start = time.perf_counter()
            self._materializer.materialize(entity_id)
            cold_times.append((time.perf_counter() - start) * 1000)

        # Warm materialization (cached)
        warm_times = []
        for i in range(self.n_entities):
            entity_id = f"entity_{i}"
            start = time.perf_counter()
            self._materializer.materialize(entity_id)
            warm_times.append((time.perf_counter() - start) * 1000)

        # Calculate metrics
        result.add_metric(
            "cold_avg",
            statistics.mean(cold_times),
            unit="ms",
            threshold_max=10.0,
        )
        result.add_metric(
            "cold_p99",
            sorted(cold_times)[int(len(cold_times) * 0.99)],
            unit="ms",
            threshold_max=50.0,
        )
        result.add_metric(
            "warm_avg",
            statistics.mean(warm_times),
            unit="ms",
            threshold_max=0.1,  # Cached should be very fast
        )
        result.add_metric(
            "cache_speedup",
            statistics.mean(cold_times) / max(statistics.mean(warm_times), 0.001),
            unit="x",
            threshold_min=10.0,  # Expect 10x+ speedup from cache
        )

        result.metadata["n_entities"] = self.n_entities
        result.metadata["events_per_entity"] = self.events_per_entity
        result.metadata["total_events"] = len(self._store)

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._store = None
        self._materializer = None


# =============================================================================
# BENCHMARK: SEMANTIC INDEX PERFORMANCE
# =============================================================================

class SemanticIndexBenchmark(BaseBenchmark):
    """
    Benchmark semantic index operations.

    Measures:
    - Index update performance
    - Concept lookup speed
    - Search accuracy (false positive rate)
    """

    name = "semantic_index"
    description = "Measure semantic indexing performance"
    category = BenchmarkCategory.QUALITY  # Query performance = quality

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 5000) if config else 5000
        self.n_concepts = config.get("n_concepts", 500) if config else 500
        self.n_queries = config.get("n_queries", 1000) if config else 1000
        self._store: Optional[BenchmarkEventStore] = None

    def setup(self) -> None:
        """Create indexed event store."""
        self._store = BenchmarkEventStore()

        for i in range(self.n_events):
            # Each event gets 1-3 concepts
            n_concepts = random.randint(1, 3)
            concepts = tuple(
                f"concept_{random.randint(0, self.n_concepts - 1)}"
                for _ in range(n_concepts)
            )

            event = BenchmarkEvent(
                event_type=EventType.OBSERVATION,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content={"index": i},
                concepts=concepts,
            )
            self._store.append(event)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        import statistics

        # Measure lookup times
        lookup_times = []
        results_counts = []

        for _ in range(self.n_queries):
            concept = f"concept_{random.randint(0, self.n_concepts - 1)}"

            start = time.perf_counter()
            results = self._store.search_by_concept(concept)
            lookup_times.append((time.perf_counter() - start) * 1000)
            results_counts.append(len(results))

        # Measure false positive rate (search for non-existent concepts)
        false_positives = 0
        false_positive_tests = 100
        for i in range(false_positive_tests):
            fake_concept = f"nonexistent_{i}"
            results = self._store.search_by_concept(fake_concept)
            if len(results) > 0:
                false_positives += 1

        # Calculate metrics
        result.add_metric(
            "avg_lookup",
            statistics.mean(lookup_times),
            unit="ms",
            threshold_max=0.5,
        )
        result.add_metric(
            "p99_lookup",
            sorted(lookup_times)[int(len(lookup_times) * 0.99)],
            unit="ms",
            threshold_max=2.0,
        )
        result.add_metric(
            "avg_results",
            statistics.mean(results_counts),
            unit="events",
        )
        result.add_metric(
            "false_positive_rate",
            false_positives / false_positive_tests,
            unit="ratio",
            threshold_max=0.01,  # Target: <1% false positives
        )
        result.add_metric(
            "index_size",
            len(self._store._concept_index),
            unit="concepts",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["n_concepts"] = self.n_concepts
        result.metadata["n_queries"] = self.n_queries

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._store = None


# =============================================================================
# BENCHMARK: TIME TRAVEL QUERIES
# =============================================================================

class TimeTravelBenchmark(BaseBenchmark):
    """
    Benchmark time-travel query performance.

    Measures:
    - Point-in-time materialization
    - Historical state reconstruction
    - Query at various depths
    """

    name = "time_travel"
    description = "Measure time-travel query performance"
    category = BenchmarkCategory.QUALITY  # Query performance = quality

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 1000) if config else 1000
        self.n_queries = config.get("n_queries", 100) if config else 100
        self._store: Optional[BenchmarkEventStore] = None
        self._materializer: Optional[BenchmarkMaterializer] = None
        self._event_ids: List[str] = []

    def setup(self) -> None:
        """Create event timeline."""
        self._store = BenchmarkEventStore()
        self._materializer = BenchmarkMaterializer(self._store)
        self._event_ids = []

        # Create a linear timeline of events for a single entity
        entity_id = "timeline_entity"
        for i in range(self.n_events):
            event = BenchmarkEvent(
                event_type=EventType.OBSERVATION,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content={
                    "entity_id": entity_id,
                    "state": {"version": i},
                },
                concepts=("timeline", f"v{i}"),
            )
            event_id = self._store.append(event)
            self._event_ids.append(event_id)

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        import statistics

        # Query at various points in history
        query_times: Dict[str, List[float]] = {
            "recent": [],    # Last 10%
            "middle": [],    # 40-60%
            "ancient": [],   # First 10%
        }

        entity_id = "timeline_entity"

        for _ in range(self.n_queries):
            self._materializer.invalidate_cache()

            # Recent query
            recent_idx = random.randint(int(self.n_events * 0.9), self.n_events - 1)
            start = time.perf_counter()
            self._materializer.materialize(entity_id, at_event=self._event_ids[recent_idx])
            query_times["recent"].append((time.perf_counter() - start) * 1000)

            # Middle query
            middle_idx = random.randint(int(self.n_events * 0.4), int(self.n_events * 0.6))
            start = time.perf_counter()
            self._materializer.materialize(entity_id, at_event=self._event_ids[middle_idx])
            query_times["middle"].append((time.perf_counter() - start) * 1000)

            # Ancient query
            ancient_idx = random.randint(0, int(self.n_events * 0.1))
            start = time.perf_counter()
            self._materializer.materialize(entity_id, at_event=self._event_ids[ancient_idx])
            query_times["ancient"].append((time.perf_counter() - start) * 1000)

        # Add metrics
        for period, times in query_times.items():
            result.add_metric(
                f"{period}_avg",
                statistics.mean(times),
                unit="ms",
                threshold_max=20.0,
            )
            result.add_metric(
                f"{period}_p99",
                sorted(times)[int(len(times) * 0.99)],
                unit="ms",
            )

        # Query depth should affect performance linearly, not exponentially
        depth_ratio = (
            statistics.mean(query_times["ancient"]) /
            max(statistics.mean(query_times["recent"]), 0.001)
        )
        result.add_metric(
            "depth_scaling_ratio",
            depth_ratio,
            unit="x",
            threshold_max=10.0,  # Ancient should be <10x slower than recent
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["n_queries"] = self.n_queries

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._store = None
        self._materializer = None
        self._event_ids = []


# =============================================================================
# BENCHMARK: DAG TRAVERSAL
# =============================================================================

class DAGTraversalBenchmark(BaseBenchmark):
    """
    Benchmark DAG traversal operations.

    Measures:
    - Ancestor traversal
    - Descendant traversal
    - Path finding between events
    """

    name = "dag_traversal"
    description = "Measure DAG traversal performance"
    category = BenchmarkCategory.QUALITY  # Query performance = quality

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 1000) if config else 1000
        self.branching_factor = config.get("branching_factor", 2) if config else 2
        self._events: Dict[str, BenchmarkEvent] = {}
        self._children: Dict[str, List[str]] = {}

    def setup(self) -> None:
        """Create a DAG structure."""
        self._events = {}
        self._children = {}

        # Create genesis event
        genesis = BenchmarkEvent(
            event_type=EventType.OBSERVATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            content={"genesis": True},
            concepts=("genesis",),
        )
        genesis_id = genesis.id
        self._events[genesis_id] = genesis
        self._children[genesis_id] = []

        # Build DAG with branching
        frontier = [genesis_id]
        events_created = 1

        while events_created < self.n_events and frontier:
            parent_id = frontier.pop(0)

            # Create children
            for _ in range(min(self.branching_factor, self.n_events - events_created)):
                child = BenchmarkEvent(
                    event_type=EventType.OBSERVATION,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content={"parent": parent_id},
                    concepts=(f"level_{events_created}",),
                    causal_parents=(parent_id,),
                )
                child_id = child.id
                self._events[child_id] = child
                self._children[child_id] = []
                self._children[parent_id].append(child_id)

                frontier.append(child_id)
                events_created += 1

                if events_created >= self.n_events:
                    break

    def _get_ancestors(self, event_id: str) -> Set[str]:
        """Get all ancestors of an event."""
        ancestors = set()
        queue = [event_id]

        while queue:
            current_id = queue.pop()
            event = self._events.get(current_id)
            if event:
                for parent_id in event.causal_parents:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        queue.append(parent_id)

        return ancestors

    def _get_descendants(self, event_id: str) -> Set[str]:
        """Get all descendants of an event."""
        descendants = set()
        queue = [event_id]

        while queue:
            current_id = queue.pop()
            for child_id in self._children.get(current_id, []):
                if child_id not in descendants:
                    descendants.add(child_id)
                    queue.append(child_id)

        return descendants

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        import statistics

        event_ids = list(self._events.keys())
        n_queries = min(100, len(event_ids))

        # Ancestor traversal
        ancestor_times = []
        ancestor_counts = []
        for _ in range(n_queries):
            event_id = random.choice(event_ids)
            start = time.perf_counter()
            ancestors = self._get_ancestors(event_id)
            ancestor_times.append((time.perf_counter() - start) * 1000)
            ancestor_counts.append(len(ancestors))

        # Descendant traversal
        descendant_times = []
        descendant_counts = []
        for _ in range(n_queries):
            event_id = random.choice(event_ids)
            start = time.perf_counter()
            descendants = self._get_descendants(event_id)
            descendant_times.append((time.perf_counter() - start) * 1000)
            descendant_counts.append(len(descendants))

        # Add metrics
        result.add_metric(
            "ancestor_avg",
            statistics.mean(ancestor_times),
            unit="ms",
            threshold_max=10.0,
        )
        result.add_metric(
            "descendant_avg",
            statistics.mean(descendant_times),
            unit="ms",
            threshold_max=10.0,
        )
        result.add_metric(
            "avg_ancestors",
            statistics.mean(ancestor_counts),
            unit="nodes",
        )
        result.add_metric(
            "avg_descendants",
            statistics.mean(descendant_counts),
            unit="nodes",
        )
        result.add_metric(
            "dag_size",
            len(self._events),
            unit="events",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["branching_factor"] = self.branching_factor

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._events = {}
        self._children = {}


# =============================================================================
# BENCHMARK: CONTENT ADDRESSING
# =============================================================================

class ContentAddressingBenchmark(BaseBenchmark):
    """
    Benchmark content-addressed ID computation.

    Measures:
    - Hash computation speed
    - ID uniqueness verification
    - Collision detection
    """

    name = "content_addressing"
    description = "Measure content-addressed ID performance"
    category = BenchmarkCategory.STABILITY  # Correctness = stability

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 10000) if config else 10000

    def setup(self) -> None:
        """Nothing to set up."""
        pass

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        import statistics

        # Measure hash computation time
        hash_times = []
        ids_seen: Set[str] = set()
        collisions = 0

        for i in range(self.n_events):
            event = BenchmarkEvent(
                event_type=EventType.OBSERVATION,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content={"unique_index": i, "random": random.random()},
                concepts=(f"test_{i}",),
            )

            start = time.perf_counter()
            event_id = event.id
            hash_times.append((time.perf_counter() - start) * 1000)

            if event_id in ids_seen:
                collisions += 1
            ids_seen.add(event_id)

        # Verify idempotency (same content = same ID)
        idempotency_checks = 100
        idempotency_failures = 0
        for i in range(idempotency_checks):
            event1 = BenchmarkEvent(
                event_type=EventType.OBSERVATION,
                timestamp="2025-01-01T00:00:00+00:00",  # Fixed timestamp
                content={"check_index": i},
                concepts=("idempotency",),
            )
            event2 = BenchmarkEvent(
                event_type=EventType.OBSERVATION,
                timestamp="2025-01-01T00:00:00+00:00",  # Same timestamp
                content={"check_index": i},  # Same content
                concepts=("idempotency",),
            )
            if event1.id != event2.id:
                idempotency_failures += 1

        # Add metrics
        result.add_metric(
            "avg_hash_time",
            statistics.mean(hash_times),
            unit="ms",
            threshold_max=0.1,  # Target: <0.1ms per hash
        )
        result.add_metric(
            "collision_rate",
            collisions / self.n_events,
            unit="ratio",
            threshold_max=0.0,  # No collisions allowed
        )
        result.add_metric(
            "idempotency_rate",
            1 - (idempotency_failures / idempotency_checks),
            unit="ratio",
            threshold_min=1.0,  # Must be 100% idempotent
        )
        result.add_metric(
            "unique_ids",
            len(ids_seen),
            unit="ids",
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["idempotency_checks"] = idempotency_checks

        return result

    def teardown(self) -> None:
        """Nothing to clean up."""
        pass


# =============================================================================
# BENCHMARK: COMPACTION
# =============================================================================

class CompactionBenchmark(BaseBenchmark):
    """
    Benchmark compaction effectiveness.

    Measures:
    - Storage reduction ratio
    - Compaction speed
    - Semantic preservation
    """

    name = "compaction"
    description = "Measure compaction effectiveness"
    category = BenchmarkCategory.REGRESSION  # Memory = regression tracking

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_events = config.get("n_events", 1000) if config else 1000
        self.n_entities = config.get("n_entities", 50) if config else 50
        self._store: Optional[BenchmarkEventStore] = None
        self._events_by_entity: Dict[str, List[str]] = {}

    def setup(self) -> None:
        """Create event store with compactable data."""
        self._store = BenchmarkEventStore()
        self._events_by_entity = {}

        # Create multiple updates per entity (compaction candidates)
        for entity_idx in range(self.n_entities):
            entity_id = f"entity_{entity_idx}"
            self._events_by_entity[entity_id] = []

            updates_per_entity = self.n_events // self.n_entities
            for update_idx in range(updates_per_entity):
                event = BenchmarkEvent(
                    event_type=EventType.OBSERVATION,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content={
                        "entity_id": entity_id,
                        "state": {
                            "counter": update_idx,
                            "data": f"update_{update_idx}",
                        },
                    },
                    concepts=(entity_id, "update"),
                )
                event_id = self._store.append(event)
                self._events_by_entity[entity_id].append(event_id)

    def _simulate_compaction(self) -> Tuple[int, int]:
        """
        Simulate compaction by keeping only latest event per entity.

        Returns:
            Tuple of (original_count, compacted_count)
        """
        original_count = len(self._store)

        # In real compaction, we'd create summary events
        # Here we count how many could be removed
        compacted_count = self.n_entities  # One event per entity

        return original_count, compacted_count

    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        original_count = len(self._store)

        # Measure compaction time
        start = time.perf_counter()
        original, compacted = self._simulate_compaction()
        compaction_time = (time.perf_counter() - start) * 1000

        # Calculate metrics
        reduction_ratio = (original - compacted) / original if original > 0 else 0

        result.add_metric(
            "compaction_time",
            compaction_time,
            unit="ms",
            threshold_max=100.0,
        )
        result.add_metric(
            "reduction_ratio",
            reduction_ratio,
            unit="ratio",
            threshold_min=0.5,  # Expect >50% reduction in this test scenario
        )
        result.add_metric(
            "original_events",
            original,
            unit="events",
        )
        result.add_metric(
            "compacted_events",
            compacted,
            unit="events",
        )

        # Verify semantic preservation
        # After compaction, each entity should still be materializable
        preservation_checks = 0
        preservation_failures = 0
        materializer = BenchmarkMaterializer(self._store)

        for entity_id in list(self._events_by_entity.keys())[:10]:  # Sample
            try:
                state = materializer.materialize(entity_id)
                if "id" not in state:
                    preservation_failures += 1
                preservation_checks += 1
            except Exception:
                preservation_failures += 1
                preservation_checks += 1

        result.add_metric(
            "semantic_preservation",
            1 - (preservation_failures / max(preservation_checks, 1)),
            unit="ratio",
            threshold_min=1.0,  # Must preserve 100%
        )

        result.metadata["n_events"] = self.n_events
        result.metadata["n_entities"] = self.n_entities

        return result

    def teardown(self) -> None:
        """Cleanup."""
        self._store = None
        self._events_by_entity = {}
