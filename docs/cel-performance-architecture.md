# CEL Performance Architecture: Aiming for the Moon

## Executive Summary

This document outlines a performance-optimized architecture for the Cognitive Event Lattice (CEL) that can scale to millions of events while maintaining sub-millisecond read latencies.

## Current State Analysis

### Bottlenecks Identified

| Component | Current Complexity | Target | Bottleneck |
|-----------|-------------------|--------|------------|
| Materialization | O(all_events) | O(entity_events) | Scans every event |
| Startup | O(all_events) | O(1) | Loads entire history |
| Causal ordering | O(n² log n) | O(n log n) | Sort per iteration |
| Event append | O(parents) | O(1) amortized | Parent verification |
| Entity lookup | O(1) cached / O(n) miss | O(1) always | Cache miss penalty |

### Scale Targets

```
Level 1 (Current):   1K-10K events    - Works today
Level 2 (Target):    100K events      - This optimization
Level 3 (Moon):      1M+ events       - Architectural limit
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CEL Performance Stack                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   API       │  │  Batch      │  │  Async      │  │  Parallel   │    │
│  │   Layer     │──│  Writer     │──│  Indexer    │──│  Materializer│   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│  ┌──────┴────────────────┴────────────────┴────────────────┴──────┐    │
│  │                     Index Layer                                  │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │    │
│  │  │ Entity     │  │ Concept    │  │ Temporal   │  │ Causal   │  │    │
│  │  │ Index      │  │ Index      │  │ Index      │  │ Index    │  │    │
│  │  │ id→events  │  │ concept→id │  │ time→id   │  │ parent→  │  │    │
│  │  └────────────┘  └────────────┘  └────────────┘  │ children │  │    │
│  └────────────────────────────────────────────────────┴──────────┴─┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Snapshot Layer                                 │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │   │
│  │  │ Full Snapshots │  │ Delta Snapshots│  │ Materialized Cache │  │   │
│  │  │ (periodic)     │  │ (incremental)  │  │ (hot entities)     │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Storage Layer                                  │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │   │
│  │  │ WAL (writes)   │  │ Segments       │  │ Memory-Mapped      │  │   │
│  │  │ append-only    │  │ (compacted)    │  │ Index Files        │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Designs

### 1. Entity Index: O(1) Materialization Lookups

**Problem:** Current materialization scans ALL events to find ones affecting an entity.

**Solution:** Maintain inverted index: `entity_id → [event_id, ...]`

```python
@dataclass
class EntityIndex:
    """Index mapping entity IDs to their events."""

    # Primary index: entity_id → ordered list of event IDs
    _entity_events: Dict[str, List[str]]

    # Reverse index: event_id → set of affected entities
    _event_entities: Dict[str, Set[str]]

    def events_for(self, entity_id: str) -> List[str]:
        """O(1) lookup of all events for an entity."""
        return self._entity_events.get(entity_id, [])

    def on_event_append(self, event: CognitiveEvent) -> None:
        """Update index when new event appended."""
        entity_id = event.content.get('entity_id')
        if entity_id:
            self._entity_events.setdefault(entity_id, []).append(event.id)
            self._event_entities.setdefault(event.id, set()).add(entity_id)
```

**Benefit:** Materialization goes from O(all_events) to O(entity_events).

---

### 2. Snapshot-Based Recovery: O(1) Startup

**Problem:** Loading 100K events at startup takes seconds.

**Solution:** Periodic snapshots + incremental WAL replay.

```python
@dataclass
class SnapshotStrategy:
    """Configurable snapshot strategy."""

    # Snapshot every N events
    interval: int = 1000

    # Keep last N snapshots
    retention: int = 5

    # Compress snapshots
    compress: bool = True

class SnapshotMaterializer:
    """Materializer that uses snapshots for fast recovery."""

    def materialize(self, entity_id: str, at: EventHorizon = None) -> T:
        # 1. Find most recent snapshot before horizon
        snapshot = self._find_snapshot_before(at)

        # 2. Load entity state from snapshot
        state = snapshot.get_entity(entity_id)

        # 3. Replay only events AFTER snapshot
        for event in self._events_since(snapshot.horizon, at):
            if self._affects_entity(event, entity_id):
                state = self._reduce(state, event)

        return state
```

**Benefit:** Startup becomes O(events_since_snapshot) instead of O(all_events).

---

### 3. Heap-Based Topological Sort: O(n log n)

**Problem:** Current causal_order() sorts on every iteration.

**Solution:** Use heapq for proper topological sort.

```python
import heapq

def causal_order_optimized(self) -> Iterator[CognitiveEvent]:
    """O(n log n) topological sort using heap."""

    # Compute in-degrees once
    in_degree = defaultdict(int)
    for event in self.events.values():
        for parent in event.causal_parents:
            if parent in self.events:
                in_degree[event.id] += 1

    # Priority queue: (timestamp, event_id)
    # Events with no dependencies go first
    heap = [
        (self.events[eid].timestamp, eid)
        for eid in self.events
        if in_degree[eid] == 0
    ]
    heapq.heapify(heap)

    visited = set()

    while heap:
        _, event_id = heapq.heappop(heap)

        if event_id in visited:
            continue
        visited.add(event_id)

        event = self.events[event_id]
        yield event

        # Decrease in-degree of children, add to heap if ready
        for child_id in self.children.get(event_id, []):
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                child = self.events[child_id]
                heapq.heappush(heap, (child.timestamp, child_id))
```

**Benefit:** Proper O(n log n) vs O(n² log n).

---

### 4. Streaming Event Store: Lazy Loading

**Problem:** `_ensure_loaded()` loads entire event history into memory.

**Solution:** Memory-mapped files with lazy loading.

```python
class StreamingEventStore:
    """Event store with lazy, streaming access."""

    def __init__(self, base_path: Path):
        self._base_path = base_path

        # Only load index on startup (small)
        self._index = self._load_index()

        # Events loaded on demand
        self._event_cache = LRUCache(max_size=10000)

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Load single event on demand."""
        if event_id in self._event_cache:
            return self._event_cache[event_id]

        # Find event file from index
        file_path = self._index.get_path(event_id)
        if file_path is None:
            return None

        # Load just this event
        event = self._load_event(file_path)
        self._event_cache[event_id] = event
        return event

    def iterate(self, start: str = None, end: str = None) -> Iterator[CognitiveEvent]:
        """Stream events without loading all into memory."""
        for event_id in self._index.range(start, end):
            yield self.get(event_id)
```

**Benefit:** Startup becomes O(index_size), not O(all_events).

---

### 5. Write Batching: Amortized O(1) Appends

**Problem:** Each append does I/O and index updates.

**Solution:** Batch writes with WAL.

```python
class BatchingEventStore:
    """Event store with write batching."""

    def __init__(self, base_path: Path, batch_size: int = 100):
        self._wal = WAL(base_path / "wal")
        self._batch: List[CognitiveEvent] = []
        self._batch_size = batch_size
        self._lock = threading.Lock()

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """Batch append - writes to WAL immediately, flushes periodically."""
        with self._lock:
            # Write to WAL (fast, sequential)
            self._wal.append(event)

            # Add to in-memory batch
            self._batch.append(event)

            # Flush if batch full
            if len(self._batch) >= self._batch_size:
                self._flush_batch()

            return MerkleRoot(event.id)

    def _flush_batch(self) -> None:
        """Flush batch to main storage and update indexes."""
        # Batch write all events
        self._storage.write_many(self._batch)

        # Batch update indexes
        for event in self._batch:
            self._entity_index.on_event_append(event)
            self._concept_index.on_event_append(event)

        self._batch.clear()
```

**Benefit:** Amortized O(1) per append instead of O(1) with high constant.

---

### 6. Parallel Materialization

**Problem:** Materializing many entities is sequential.

**Solution:** Parallel materialization with thread pool.

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelMaterializer:
    """Materializer with parallel entity resolution."""

    def __init__(self, base_materializer: Materializer, workers: int = 4):
        self._base = base_materializer
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def materialize_many(
        self,
        entity_ids: List[str],
        at: EventHorizon = None,
    ) -> Dict[str, T]:
        """Materialize multiple entities in parallel."""
        futures = {
            self._executor.submit(self._base.materialize, eid, at): eid
            for eid in entity_ids
        }

        results = {}
        for future in as_completed(futures):
            entity_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results[entity_id] = result
            except Exception:
                pass  # Handle errors

        return results
```

---

## Implementation Priority

| Phase | Component | Impact | Effort | Priority |
|-------|-----------|--------|--------|----------|
| 1 | Entity Index | 10x materialization | Low | **HIGH** |
| 1 | Heap-based sort | 100x causal queries | Low | **HIGH** |
| 2 | Snapshot recovery | 10x startup | Medium | **MEDIUM** |
| 2 | Write batching | 5x append throughput | Medium | **MEDIUM** |
| 3 | Streaming store | Memory efficiency | High | **LOW** |
| 3 | Parallel materialization | CPU utilization | Low | **LOW** |

---

## Benchmarks to Implement

```python
class CELBenchmarks:
    """Performance benchmarks for CEL."""

    def bench_append_latency(self, n: int = 10000):
        """Measure event append latency at scale."""

    def bench_materialize_cold(self, n_entities: int = 100):
        """Measure cold materialization (no cache)."""

    def bench_materialize_hot(self, n_entities: int = 100):
        """Measure hot materialization (cached)."""

    def bench_causal_order(self, n_events: int = 10000):
        """Measure topological sort performance."""

    def bench_startup_time(self, n_events: int = 100000):
        """Measure cold start time with N events."""

    def bench_memory_usage(self, n_events: int = 100000):
        """Measure memory footprint at scale."""
```

---

## Success Metrics

| Metric | Current | Target | Moon |
|--------|---------|--------|------|
| Append latency (p99) | ~1ms | <100μs | <10μs |
| Materialize cold (p99) | ~100ms | <10ms | <1ms |
| Materialize hot (p99) | ~1ms | <100μs | <10μs |
| Startup (100K events) | ~10s | <1s | <100ms |
| Memory (100K events) | ~500MB | <100MB | <50MB |

---

## Next Steps

1. **Create benchmark suite** - Establish baseline measurements
2. **Implement Entity Index** - Biggest immediate win
3. **Implement heap-based sort** - Low effort, high impact
4. **Add snapshot layer** - For startup performance
5. **Profile and iterate** - Measure, don't guess
