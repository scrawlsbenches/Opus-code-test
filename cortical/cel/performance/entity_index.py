"""
High-performance indexes for the Cognitive Event Lattice.

These indexes provide O(1) lookups for common access patterns,
eliminating the need to scan all events for entity materialization.

Key insight: Maintain inverted indexes updated on each append.
Trade space for time - indexes are cheap, scanning is expensive.

Thread Safety: All indexes are thread-safe with fine-grained locking.
"""

from __future__ import annotations

import bisect
import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from ..core.events import CognitiveEvent, EventType


@dataclass
class IndexStats:
    """Statistics for an index."""

    entries: int
    memory_bytes: int
    last_updated: datetime
    lookups: int = 0
    hits: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups > 0 else 0.0


class EntityIndex:
    """
    Inverted index mapping entity IDs to their events.

    This is the primary optimization for materialization.
    Instead of scanning all events, we can directly retrieve
    only the events that affect a specific entity.

    Complexity:
        - Lookup: O(1)
        - Insert: O(1) amortized
        - Memory: O(entities × avg_events_per_entity)

    Example:
        index = EntityIndex()

        # Index an event
        index.on_event(event)

        # Get all events for an entity - O(1)!
        event_ids = index.events_for("T-20251229-001")

        # Materialize only those events instead of scanning all
        for event_id in event_ids:
            event = store.get(event_id)
            state = reducer(state, event)
    """

    def __init__(self):
        """Initialize empty entity index."""
        # Primary index: entity_id → ordered list of (timestamp, event_id)
        # Ordered by timestamp for temporal queries
        self._entity_events: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # Reverse index: event_id → set of affected entity IDs
        self._event_entities: Dict[str, Set[str]] = defaultdict(set)

        # Entity existence tracking (for fast "does entity exist" checks)
        self._known_entities: Set[str] = set()

        # Thread safety
        self._lock = threading.RLock()

        # Stats
        self._lookups = 0
        self._inserts = 0

    def on_event(self, event: CognitiveEvent) -> None:
        """
        Update index when a new event is appended.

        Extracts entity references from the event and updates
        all relevant index structures.

        Args:
            event: The newly appended event
        """
        entity_ids = self._extract_entity_ids(event)
        if not entity_ids:
            return

        with self._lock:
            for entity_id in entity_ids:
                # Add to entity → events index (maintain order by timestamp)
                entry = (event.timestamp, event.id)
                events_list = self._entity_events[entity_id]

                # Binary search insert to maintain sorted order
                bisect.insort(events_list, entry)

                # Add to reverse index
                self._event_entities[event.id].add(entity_id)

                # Track entity existence
                self._known_entities.add(entity_id)

            self._inserts += 1

    def events_for(
        self,
        entity_id: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[str]:
        """
        Get all event IDs affecting an entity.

        Args:
            entity_id: Entity to look up
            since: Only events after this timestamp (ISO format)
            until: Only events before this timestamp (ISO format)

        Returns:
            List of event IDs in chronological order

        Complexity: O(1) for full list, O(log n) for filtered
        """
        with self._lock:
            self._lookups += 1

            events = self._entity_events.get(entity_id, [])
            if not events:
                return []

            # Fast path: no filtering
            if since is None and until is None:
                return [event_id for _, event_id in events]

            # Filtered path: use binary search
            result = []
            for timestamp, event_id in events:
                if since is not None and timestamp < since:
                    continue
                if until is not None and timestamp > until:
                    break
                result.append(event_id)

            return result

    def entities_affected_by(self, event_id: str) -> Set[str]:
        """
        Get all entities affected by an event.

        Useful for cache invalidation.

        Args:
            event_id: Event to look up

        Returns:
            Set of affected entity IDs
        """
        with self._lock:
            return self._event_entities.get(event_id, set()).copy()

    def entity_exists(self, entity_id: str) -> bool:
        """
        Fast check if an entity has any events.

        Complexity: O(1)
        """
        with self._lock:
            return entity_id in self._known_entities

    def entity_count(self) -> int:
        """Total number of unique entities indexed."""
        with self._lock:
            return len(self._known_entities)

    def event_count(self, entity_id: str) -> int:
        """Number of events for a specific entity."""
        with self._lock:
            return len(self._entity_events.get(entity_id, []))

    def _extract_entity_ids(self, event: CognitiveEvent) -> Set[str]:
        """
        Extract all entity IDs referenced by an event.

        Handles various event structures:
        - Direct entity_id field
        - Intention/Fulfillment references
        - Invalidation targets
        - Edge endpoints
        """
        entity_ids = set()
        content = event.content

        # Direct entity reference
        if 'entity_id' in content:
            entity_ids.add(content['entity_id'])

        # Task/intention ID
        if 'id' in content and content.get('entity_type') in (
            'task', 'decision', 'sprint', 'epic', 'handoff', 'document'
        ):
            entity_ids.add(content['id'])

        # Fulfillment references
        if event.event_type == EventType.FULFILLMENT:
            if 'intention_id' in content:
                entity_ids.add(content['intention_id'])

        # Invalidation target
        if event.event_type == EventType.INVALIDATION:
            if 'target_id' in content:
                entity_ids.add(content['target_id'])

        # Edge endpoints
        if content.get('entity_type') == 'edge':
            if 'from_id' in content:
                entity_ids.add(content['from_id'])
            if 'to_id' in content:
                entity_ids.add(content['to_id'])

        # Affects list (for decisions)
        if 'affects' in content:
            for affected in content['affects']:
                if isinstance(affected, str):
                    entity_ids.add(affected)

        return entity_ids

    def clear(self) -> None:
        """Clear all index data."""
        with self._lock:
            self._entity_events.clear()
            self._event_entities.clear()
            self._known_entities.clear()

    @property
    def stats(self) -> IndexStats:
        """Get index statistics."""
        with self._lock:
            # Rough memory estimate
            memory = (
                len(self._entity_events) * 64 +  # dict overhead
                sum(len(v) * 48 for v in self._entity_events.values()) +  # entries
                len(self._event_entities) * 64 +
                len(self._known_entities) * 32
            )
            return IndexStats(
                entries=sum(len(v) for v in self._entity_events.values()),
                memory_bytes=memory,
                last_updated=datetime.now(),
                lookups=self._lookups,
                hits=self._lookups,  # All lookups are hits for this index
            )

    def save(self, path: Path) -> None:
        """Persist index to disk."""
        with self._lock:
            data = {
                'entity_events': {
                    k: list(v) for k, v in self._entity_events.items()
                },
                'event_entities': {
                    k: list(v) for k, v in self._event_entities.items()
                },
                'known_entities': list(self._known_entities),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'EntityIndex':
        """Load index from disk."""
        index = cls()
        if not path.exists():
            return index

        with open(path) as f:
            data = json.load(f)

        with index._lock:
            index._entity_events = defaultdict(
                list,
                {k: [tuple(e) for e in v] for k, v in data.get('entity_events', {}).items()}
            )
            index._event_entities = defaultdict(
                set,
                {k: set(v) for k, v in data.get('event_entities', {}).items()}
            )
            index._known_entities = set(data.get('known_entities', []))

        return index


class ConceptIndex:
    """
    Inverted index mapping concepts to events.

    Optimizes semantic search by pre-indexing concept terms.
    Uses bloom filter for fast "probably has" checks.

    Complexity:
        - Lookup: O(1)
        - Insert: O(concepts_per_event)
    """

    def __init__(self, bloom_size: int = 10000, bloom_hashes: int = 3):
        """
        Initialize concept index.

        Args:
            bloom_size: Size of bloom filter (larger = fewer false positives)
            bloom_hashes: Number of hash functions
        """
        # Concept → set of event IDs
        self._concept_events: Dict[str, Set[str]] = defaultdict(set)

        # Bloom filter for fast "probably contains" checks
        self._bloom_bits = [False] * bloom_size
        self._bloom_size = bloom_size
        self._bloom_hashes = bloom_hashes

        self._lock = threading.RLock()

    def on_event(self, event: CognitiveEvent) -> None:
        """Update index when event is appended."""
        with self._lock:
            for concept in event.concepts:
                # Add to inverted index
                self._concept_events[concept].add(event.id)

                # Update bloom filter
                for pos in self._bloom_positions(concept):
                    self._bloom_bits[pos] = True

    def events_for(self, concept: str) -> Set[str]:
        """Get all events with a concept."""
        with self._lock:
            return self._concept_events.get(concept, set()).copy()

    def events_for_all(self, concepts: List[str]) -> Set[str]:
        """Get events matching ALL concepts (intersection)."""
        with self._lock:
            if not concepts:
                return set()

            result = self._concept_events.get(concepts[0], set()).copy()
            for concept in concepts[1:]:
                result &= self._concept_events.get(concept, set())
            return result

    def events_for_any(self, concepts: List[str]) -> Set[str]:
        """Get events matching ANY concept (union)."""
        with self._lock:
            result = set()
            for concept in concepts:
                result |= self._concept_events.get(concept, set())
            return result

    def probably_has(self, concept: str) -> bool:
        """
        Fast probabilistic check if concept exists.

        May return false positives, but never false negatives.
        Use before expensive exact lookup.
        """
        return all(
            self._bloom_bits[pos]
            for pos in self._bloom_positions(concept)
        )

    def _bloom_positions(self, item: str) -> List[int]:
        """Generate bloom filter bit positions."""
        import hashlib
        positions = []
        for i in range(self._bloom_hashes):
            h = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            positions.append(int(h, 16) % self._bloom_size)
        return positions

    @property
    def concept_count(self) -> int:
        """Number of unique concepts indexed."""
        with self._lock:
            return len(self._concept_events)


class TemporalIndex:
    """
    Index for efficient time-range queries.

    Maintains events in sorted order by timestamp for efficient
    range scans and "as of" queries.

    Complexity:
        - Range query: O(log n + results)
        - Insert: O(log n)
    """

    def __init__(self):
        """Initialize temporal index."""
        # Sorted list of (timestamp, event_id)
        self._timeline: List[Tuple[str, str]] = []
        self._lock = threading.RLock()

    def on_event(self, event: CognitiveEvent) -> None:
        """Update index when event is appended."""
        with self._lock:
            entry = (event.timestamp, event.id)
            bisect.insort(self._timeline, entry)

    def events_in_range(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[str]:
        """
        Get events in a time range.

        Args:
            start: Start timestamp (inclusive), None for beginning
            end: End timestamp (inclusive), None for end

        Returns:
            Event IDs in chronological order
        """
        with self._lock:
            if not self._timeline:
                return []

            # Find start position
            if start is None:
                start_idx = 0
            else:
                start_idx = bisect.bisect_left(
                    self._timeline,
                    (start, '')
                )

            # Find end position
            if end is None:
                end_idx = len(self._timeline)
            else:
                end_idx = bisect.bisect_right(
                    self._timeline,
                    (end, '\xff' * 64)  # Max event ID
                )

            return [
                event_id
                for _, event_id in self._timeline[start_idx:end_idx]
            ]

    def events_before(self, timestamp: str, limit: int = 100) -> List[str]:
        """Get most recent events before a timestamp."""
        with self._lock:
            end_idx = bisect.bisect_left(self._timeline, (timestamp, ''))
            start_idx = max(0, end_idx - limit)
            return [
                event_id
                for _, event_id in self._timeline[start_idx:end_idx]
            ]

    def events_after(self, timestamp: str, limit: int = 100) -> List[str]:
        """Get events after a timestamp."""
        with self._lock:
            start_idx = bisect.bisect_right(self._timeline, (timestamp, '\xff' * 64))
            end_idx = min(len(self._timeline), start_idx + limit)
            return [
                event_id
                for _, event_id in self._timeline[start_idx:end_idx]
            ]

    @property
    def event_count(self) -> int:
        """Total events indexed."""
        with self._lock:
            return len(self._timeline)

    @property
    def time_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Get (earliest, latest) timestamps."""
        with self._lock:
            if not self._timeline:
                return (None, None)
            return (self._timeline[0][0], self._timeline[-1][0])
