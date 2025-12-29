"""
Streaming Event Store with Write Batching.

Solves two problems:
1. Startup: Don't load all events into memory at once
2. Writes: Batch writes for amortized O(1) appends

Key Design Decisions:
- Events are stored in segment files (configurable size)
- Index is loaded at startup (small, memory-mapped possible)
- Events loaded on-demand with LRU cache
- Writes go to WAL first, then batched to segments
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from ..core.events import CognitiveEvent, EventType
from ..core.references import MerkleRoot, EventHorizon
from .entity_index import EntityIndex, ConceptIndex, TemporalIndex
from .optimized_dag import OptimizedDAG
from .snapshots import SnapshotManager, SnapshotConfig


@dataclass
class StoreConfig:
    """Configuration for streaming event store."""

    # Segment configuration
    events_per_segment: int = 1000
    max_segment_size_mb: int = 10

    # Cache configuration
    event_cache_size: int = 10000
    segment_cache_size: int = 10

    # Write batching
    batch_size: int = 100
    batch_timeout_ms: int = 100

    # Indexes to maintain
    enable_entity_index: bool = True
    enable_concept_index: bool = True
    enable_temporal_index: bool = True

    # Snapshots
    snapshot_interval: int = 1000
    snapshot_retention: int = 5


class LRUCache:
    """Simple LRU cache with configurable max size."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # Remove oldest
                    self._cache.popitem(last=False)
            self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


@dataclass
class EventIndex:
    """
    Lightweight index for locating events.

    Maps event_id → (segment_id, offset)
    Loaded entirely at startup (small - just IDs and offsets).
    """

    # event_id → (segment_id, offset_in_segment)
    locations: Dict[str, Tuple[str, int]] = field(default_factory=dict)

    # segment_id → list of event_ids (in order)
    segments: Dict[str, List[str]] = field(default_factory=dict)

    # Ordered list of all event IDs (by insertion order)
    event_order: List[str] = field(default_factory=list)

    # Current segment being written to
    current_segment: str = ""
    current_segment_count: int = 0

    def add(self, event_id: str, segment_id: str, offset: int) -> None:
        """Record event location."""
        self.locations[event_id] = (segment_id, offset)
        if segment_id not in self.segments:
            self.segments[segment_id] = []
        self.segments[segment_id].append(event_id)
        self.event_order.append(event_id)

    def get_location(self, event_id: str) -> Optional[Tuple[str, int]]:
        """Get segment and offset for an event."""
        return self.locations.get(event_id)

    def get_segment_events(self, segment_id: str) -> List[str]:
        """Get all event IDs in a segment."""
        return self.segments.get(segment_id, [])

    def contains(self, event_id: str) -> bool:
        """Check if event is indexed."""
        return event_id in self.locations

    def events_after(self, event_id: Optional[str]) -> Iterator[str]:
        """Iterate event IDs after a given event."""
        if event_id is None:
            yield from self.event_order
            return

        started = False
        for eid in self.event_order:
            if started:
                yield eid
            elif eid == event_id:
                started = True

    @property
    def count(self) -> int:
        return len(self.event_order)

    def save(self, path: Path) -> None:
        """Persist index to disk."""
        data = {
            'locations': {k: list(v) for k, v in self.locations.items()},
            'segments': self.segments,
            'event_order': self.event_order,
            'current_segment': self.current_segment,
            'current_segment_count': self.current_segment_count,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'EventIndex':
        """Load index from disk."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        index = cls()
        index.locations = {k: tuple(v) for k, v in data.get('locations', {}).items()}
        index.segments = data.get('segments', {})
        index.event_order = data.get('event_order', [])
        index.current_segment = data.get('current_segment', '')
        index.current_segment_count = data.get('current_segment_count', 0)
        return index


class WriteAheadLog:
    """
    Simple WAL for durability before batch writes.

    Events are appended to WAL immediately, then periodically
    flushed to segments in batches.
    """

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: CognitiveEvent) -> None:
        """Append event to WAL."""
        with self._lock:
            with open(self._path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')

    def read_all(self) -> List[CognitiveEvent]:
        """Read all events from WAL."""
        if not self._path.exists():
            return []

        events = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(CognitiveEvent.from_dict(json.loads(line)))
        return events

    def truncate(self) -> None:
        """Clear the WAL after successful flush."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    @property
    def size(self) -> int:
        """Size of WAL in bytes."""
        if self._path.exists():
            return self._path.stat().st_size
        return 0


class BatchingWriter:
    """
    Batches writes for efficiency.

    Instead of writing each event immediately:
    1. Write to WAL for durability
    2. Accumulate in memory batch
    3. Flush batch to segment when full or on timeout
    """

    def __init__(
        self,
        base_path: Path,
        config: StoreConfig,
        on_flush: Callable[[List[CognitiveEvent]], None],
    ):
        """
        Initialize batching writer.

        Args:
            base_path: Directory for storage
            config: Store configuration
            on_flush: Callback when batch is flushed
        """
        self._base_path = base_path
        self._config = config
        self._on_flush = on_flush

        self._wal = WriteAheadLog(base_path / "wal" / "current.wal")
        self._batch: List[CognitiveEvent] = []
        self._lock = threading.Lock()

        # Timeout-based flushing
        self._last_write = time.time()
        self._flush_timer: Optional[threading.Timer] = None

    def write(self, event: CognitiveEvent) -> None:
        """
        Write an event (batched).

        Event is immediately durable (WAL) but may not be
        visible in segments until batch flush.
        """
        with self._lock:
            # Write to WAL for durability
            self._wal.append(event)

            # Add to batch
            self._batch.append(event)
            self._last_write = time.time()

            # Flush if batch full
            if len(self._batch) >= self._config.batch_size:
                self._flush_locked()
            else:
                # Schedule timeout flush
                self._schedule_timeout_flush()

    def _schedule_timeout_flush(self) -> None:
        """Schedule a flush after timeout if no more writes."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()

        timeout = self._config.batch_timeout_ms / 1000.0
        self._flush_timer = threading.Timer(timeout, self._timeout_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _timeout_flush(self) -> None:
        """Flush triggered by timeout."""
        with self._lock:
            if self._batch:
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush batch (must hold lock)."""
        if not self._batch:
            return

        # Copy and clear batch
        events = self._batch.copy()
        self._batch.clear()

        # Notify callback
        try:
            self._on_flush(events)
        except Exception as e:
            # Re-add events on failure
            self._batch = events + self._batch
            raise

        # Clear WAL after successful flush
        self._wal.truncate()

        # Cancel any pending timer
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def flush(self) -> None:
        """Force immediate flush."""
        with self._lock:
            self._flush_locked()

    def recover(self) -> List[CognitiveEvent]:
        """Recover unflushed events from WAL."""
        return self._wal.read_all()

    @property
    def pending_count(self) -> int:
        """Number of events waiting to be flushed."""
        with self._lock:
            return len(self._batch)


class StreamingEventStore:
    """
    High-performance event store with streaming and batching.

    Features:
    1. Lazy loading: Only loads index at startup, events on demand
    2. Write batching: Amortized O(1) appends
    3. LRU caching: Hot events stay in memory
    4. Multiple indexes: Entity, concept, temporal
    5. Snapshots: Fast recovery

    Usage:
        store = StreamingEventStore(Path(".cel"))

        # Write events (batched)
        store.append(event1)
        store.append(event2)

        # Read events (lazy loaded)
        event = store.get(event_id)

        # Query by entity (uses index)
        events = store.events_for_entity("T-001")

        # Iterate in causal order (streaming)
        for event in store.iterate():
            process(event)
    """

    def __init__(
        self,
        base_path: Path,
        config: Optional[StoreConfig] = None,
    ):
        """
        Initialize streaming event store.

        Args:
            base_path: Directory for all storage
            config: Store configuration
        """
        self._base_path = Path(base_path)
        self._config = config or StoreConfig()

        # Ensure directories exist
        self._base_path.mkdir(parents=True, exist_ok=True)
        (self._base_path / "segments").mkdir(exist_ok=True)
        (self._base_path / "indexes").mkdir(exist_ok=True)

        # Load lightweight index (fast)
        self._event_index = EventIndex.load(self._base_path / "indexes" / "events.json")

        # Initialize secondary indexes
        self._entity_index: Optional[EntityIndex] = None
        self._concept_index: Optional[ConceptIndex] = None
        self._temporal_index: Optional[TemporalIndex] = None

        if self._config.enable_entity_index:
            self._entity_index = EntityIndex.load(
                self._base_path / "indexes" / "entities.json"
            )
        if self._config.enable_concept_index:
            self._concept_index = ConceptIndex()
        if self._config.enable_temporal_index:
            self._temporal_index = TemporalIndex()

        # Initialize caches
        self._event_cache = LRUCache(self._config.event_cache_size)
        self._segment_cache = LRUCache(self._config.segment_cache_size)

        # Initialize DAG for causal operations
        self._dag = OptimizedDAG()

        # Initialize batching writer
        self._writer = BatchingWriter(
            self._base_path,
            self._config,
            self._flush_to_segment,
        )

        # Snapshot manager
        self._snapshots = SnapshotManager(
            self._base_path / "snapshots",
            SnapshotConfig(
                full_interval=self._config.snapshot_interval,
                retention_count=self._config.snapshot_retention,
            ),
        )

        # Recover from WAL if needed
        self._recover()

        self._lock = threading.RLock()

    def _recover(self) -> None:
        """Recover from WAL after crash."""
        wal_events = self._writer.recover()
        if wal_events:
            # Re-process WAL events
            for event in wal_events:
                self._index_event(event)
                self._dag.add(event, verify_parents=False)
            # Flush to segment
            self._writer.flush()

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """
        Append an event to the store.

        Event is immediately durable and indexed, but may not
        be in a segment until batch flush.

        Args:
            event: Event to append

        Returns:
            Merkle root of appended event
        """
        with self._lock:
            # Add to DAG
            root = self._dag.add(event)

            # Index immediately for queries
            self._index_event(event)

            # Cache
            self._event_cache.put(event.id, event)

            # Write (batched)
            self._writer.write(event)

            # Check if snapshot needed
            snapshot_type = self._snapshots.should_snapshot(self._event_index.count)
            if snapshot_type != 'none':
                self._create_snapshot(snapshot_type)

            return root

    def _index_event(self, event: CognitiveEvent) -> None:
        """Update all indexes for an event."""
        if self._entity_index:
            self._entity_index.on_event(event)
        if self._concept_index:
            self._concept_index.on_event(event)
        if self._temporal_index:
            self._temporal_index.on_event(event)

    def _flush_to_segment(self, events: List[CognitiveEvent]) -> None:
        """Flush a batch of events to a segment file."""
        if not events:
            return

        # Determine segment
        segment_id = self._event_index.current_segment
        if (
            not segment_id or
            self._event_index.current_segment_count >= self._config.events_per_segment
        ):
            # Create new segment
            segment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._event_index.current_segment = segment_id
            self._event_index.current_segment_count = 0

        # Write events to segment
        segment_path = self._base_path / "segments" / f"{segment_id}.jsonl"
        with open(segment_path, 'a') as f:
            for i, event in enumerate(events):
                offset = self._event_index.current_segment_count + i
                f.write(json.dumps(event.to_dict()) + '\n')
                self._event_index.add(event.id, segment_id, offset)

        self._event_index.current_segment_count += len(events)

        # Persist index
        self._event_index.save(self._base_path / "indexes" / "events.json")

        # Persist entity index
        if self._entity_index:
            self._entity_index.save(self._base_path / "indexes" / "entities.json")

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """
        Get an event by ID.

        Checks cache first, then loads from segment if needed.

        Args:
            event_id: Event to retrieve

        Returns:
            Event or None if not found
        """
        # Check cache
        cached = self._event_cache.get(event_id)
        if cached is not None:
            return cached

        # Check DAG (for recently added events)
        dag_event = self._dag.get(event_id)
        if dag_event is not None:
            self._event_cache.put(event_id, dag_event)
            return dag_event

        # Load from segment
        location = self._event_index.get_location(event_id)
        if location is None:
            return None

        segment_id, offset = location
        event = self._load_from_segment(segment_id, offset)
        if event:
            self._event_cache.put(event_id, event)
        return event

    def _load_from_segment(self, segment_id: str, offset: int) -> Optional[CognitiveEvent]:
        """Load a specific event from a segment."""
        # Check segment cache
        segment_events = self._segment_cache.get(segment_id)
        if segment_events is not None:
            if offset < len(segment_events):
                return segment_events[offset]

        # Load segment
        segment_path = self._base_path / "segments" / f"{segment_id}.jsonl"
        if not segment_path.exists():
            return None

        events = []
        with open(segment_path) as f:
            for line in f:
                events.append(CognitiveEvent.from_dict(json.loads(line.strip())))

        # Cache segment
        self._segment_cache.put(segment_id, events)

        if offset < len(events):
            return events[offset]
        return None

    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
        event_types: Optional[Set[EventType]] = None,
    ) -> Iterator[CognitiveEvent]:
        """
        Iterate events in causal order.

        Uses streaming to avoid loading all events at once.

        Args:
            from_event: Start after this event (exclusive)
            to_event: Stop at this event (inclusive)
            event_types: Filter by event types

        Yields:
            Events in causal order
        """
        for event in self._dag.causal_order(from_event, to_event):
            if event_types is None or event.event_type in event_types:
                yield event

    def events_for_entity(self, entity_id: str) -> List[CognitiveEvent]:
        """
        Get all events affecting an entity.

        Uses entity index for O(1) lookup of event IDs,
        then loads events (from cache or disk).

        Args:
            entity_id: Entity to query

        Returns:
            List of events in chronological order
        """
        if self._entity_index is None:
            raise RuntimeError("Entity index not enabled")

        event_ids = self._entity_index.events_for(entity_id)
        return [self.get(eid) for eid in event_ids if self.get(eid) is not None]

    def search_concepts(self, concepts: List[str], match_all: bool = True) -> List[str]:
        """
        Search for events by concepts.

        Args:
            concepts: Concepts to search for
            match_all: True for AND, False for OR

        Returns:
            List of matching event IDs
        """
        if self._concept_index is None:
            raise RuntimeError("Concept index not enabled")

        if match_all:
            return list(self._concept_index.events_for_all(concepts))
        else:
            return list(self._concept_index.events_for_any(concepts))

    def heads(self) -> List[MerkleRoot]:
        """Get current branch heads."""
        return self._dag.get_heads()

    def latest(self) -> Optional[MerkleRoot]:
        """Get most recent event."""
        return self._dag.get_latest()

    def horizon(self) -> EventHorizon:
        """Get current event horizon."""
        latest = self.latest()
        if latest is None:
            return EventHorizon(event_id="GENESIS", is_head=True)
        return EventHorizon(event_id=latest.value, is_head=True)

    def _create_snapshot(self, snapshot_type: str) -> None:
        """Create a snapshot of current state."""
        entity_index_data = {}
        if self._entity_index:
            # Convert to simple dict format
            for entity_id in self._entity_index._known_entities:
                event_ids = self._entity_index.events_for(entity_id)
                entity_index_data[entity_id] = event_ids

        self._snapshots.create_snapshot(
            horizon=self.horizon(),
            event_count=self._event_index.count,
            entity_index=entity_index_data,
            event_ids=self._event_index.event_order.copy(),
            snapshot_type=snapshot_type,
        )

    @property
    def count(self) -> int:
        """Total number of events."""
        return self._event_index.count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            'event_count': self._event_index.count,
            'segment_count': len(self._event_index.segments),
            'cache_size': len(self._event_cache),
            'pending_writes': self._writer.pending_count,
            'entity_count': self._entity_index.entity_count() if self._entity_index else 0,
            'concept_count': self._concept_index.concept_count if self._concept_index else 0,
            'snapshot_count': self._snapshots.snapshot_count,
        }

    def flush(self) -> None:
        """Force flush pending writes."""
        self._writer.flush()

    def close(self) -> None:
        """Clean shutdown - flush and persist state."""
        self.flush()
        self._event_index.save(self._base_path / "indexes" / "events.json")
        if self._entity_index:
            self._entity_index.save(self._base_path / "indexes" / "entities.json")
