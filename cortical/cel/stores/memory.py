"""
In-Memory Event Store for CEL.

A lightweight EventStore implementation for testing, demos, and development.
For production use, see StreamingEventStore in cortical.cel.performance.

Features:
- Implements full EventStore protocol
- Append-only semantics (immutable events)
- Content-addressed IDs (via CognitiveEvent.id)
- Causal ordering preserved
- Temporal query support (iterate with from_event/to_event)
- Ancestor/descendant traversal

Limitations:
- All events in memory (not suitable for large event streams)
- No persistence (events lost on restart)
- No compaction (events accumulate indefinitely)

Usage:
    from cortical.cel.stores import MemoryEventStore
    from cortical.cel.core.events import Observation

    store = MemoryEventStore()

    # Append events
    event = Observation(content={'type': 'test'}, concepts=('test',))
    root = store.append(event)

    # Query
    event = store.get(root.value)
    for e in store.iterate():
        process(e)

    # Temporal queries
    for e in store.iterate(from_event=start_id, to_event=end_id):
        process(e)
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Set

from ..core.events import CognitiveEvent, EventType
from ..core.references import MerkleRoot, EventHorizon


class MemoryEventStore:
    """
    In-memory implementation of CEL EventStore protocol.

    This is a lightweight store for testing, demonstrations, and development.
    In production, use StreamingEventStore for persistence and performance.

    The store maintains:
    - Ordered dict of events (preserves insertion order)
    - Parent-child relationships for causal traversal
    - No persistence - all state is in memory
    """

    def __init__(self):
        """Initialize empty event store."""
        self._events: OrderedDict[str, CognitiveEvent] = OrderedDict()
        self._children: Dict[str, Set[str]] = {}  # parent_id -> child_ids

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """
        Append event to store.

        Args:
            event: CognitiveEvent to append

        Returns:
            MerkleRoot containing the event's content-addressed ID

        Note:
            Idempotent - appending the same event twice returns same root.
        """
        event_id = event.id

        # Idempotent - don't add duplicates
        if event_id in self._events:
            return MerkleRoot(event_id)

        # Track causal relationships
        for parent_id in event.causal_parents:
            if parent_id not in self._children:
                self._children[parent_id] = set()
            self._children[parent_id].add(event_id)

        self._events[event_id] = event
        return MerkleRoot(event_id)

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """
        Retrieve event by ID.

        Args:
            event_id: Content-addressed event ID

        Returns:
            CognitiveEvent if found, None otherwise
        """
        return self._events.get(event_id)

    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
        event_types: Optional[Sequence[EventType]] = None,
    ) -> Iterator[CognitiveEvent]:
        """
        Iterate events in causal order.

        Args:
            from_event: Start after this event ID (exclusive)
            to_event: Stop at this event ID (inclusive)
            event_types: Filter to only these event types

        Yields:
            CognitiveEvent instances in insertion (causal) order
        """
        started = from_event is None

        for event_id, event in self._events.items():
            if not started:
                if event_id == from_event:
                    started = True
                continue

            if event_types is None or event.event_type in event_types:
                yield event

            if to_event and event_id == to_event:
                break

    def heads(self) -> List[MerkleRoot]:
        """
        Get events with no children (branch heads).

        Returns:
            List of MerkleRoot for all head events
        """
        all_ids = set(self._events.keys())
        children = set()
        for child_set in self._children.values():
            children.update(child_set)

        head_ids = all_ids - children
        return [MerkleRoot(eid) for eid in head_ids]

    def latest(self) -> Optional[MerkleRoot]:
        """
        Get most recent event.

        Returns:
            MerkleRoot of the last appended event, or None if empty
        """
        if not self._events:
            return None
        # OrderedDict preserves insertion order
        last_id = list(self._events.keys())[-1]
        return MerkleRoot(last_id)

    def horizon(self) -> EventHorizon:
        """
        Get current event horizon.

        Returns:
            EventHorizon pointing to latest event or GENESIS if empty
        """
        latest = self.latest()
        if latest is None:
            return EventHorizon(event_id="GENESIS", is_head=True)
        return EventHorizon(event_id=latest.value, is_head=True)

    def ancestors(self, event_id: str, depth: int = -1) -> Iterator[CognitiveEvent]:
        """
        Iterate ancestors in reverse causal order (breadth-first).

        Args:
            event_id: Starting event ID
            depth: Maximum depth to traverse (-1 for unlimited)

        Yields:
            Ancestor events, closest first
        """
        visited: Set[str] = set()
        to_visit = [event_id]
        current_depth = 0

        while to_visit and (depth == -1 or current_depth < depth):
            next_level = []
            for eid in to_visit:
                if eid in visited:
                    continue
                visited.add(eid)

                event = self.get(eid)
                if event and eid != event_id:  # Don't yield start event
                    yield event

                if event:
                    for parent_id in event.causal_parents:
                        if parent_id not in visited:
                            next_level.append(parent_id)

            to_visit = next_level
            current_depth += 1

    def descendants(self, event_id: str) -> Iterator[CognitiveEvent]:
        """
        Iterate descendants in causal order (breadth-first).

        Args:
            event_id: Starting event ID

        Yields:
            Descendant events in causal order
        """
        visited: Set[str] = set()
        to_visit = [event_id]

        while to_visit:
            next_level = []
            for eid in to_visit:
                if eid in visited:
                    continue
                visited.add(eid)

                if eid != event_id:  # Don't yield start event
                    event = self.get(eid)
                    if event:
                        yield event

                for child_id in self._children.get(eid, set()):
                    if child_id not in visited:
                        next_level.append(child_id)

            to_visit = next_level

    def events_in_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Iterator[CognitiveEvent]:
        """
        Iterate events within a time range.

        Args:
            start_time: Include events at or after this time
            end_time: Include events at or before this time

        Yields:
            Events within the specified time range
        """
        for event in self._events.values():
            event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            if start_time and event_time < start_time:
                continue
            if end_time and event_time > end_time:
                continue

            yield event

    def events_for_entity(self, entity_id: str) -> List[CognitiveEvent]:
        """
        Get all events that reference an entity.

        Args:
            entity_id: Entity ID to search for

        Returns:
            List of events referencing this entity (in causal order)

        Note:
            This is O(n) - for production use, enable entity indexing
            in StreamingEventStore.
        """
        result = []
        for event in self._events.values():
            # Check if entity_id appears in event content
            content = event.content
            if isinstance(content, dict):
                # Check common fields
                if content.get('entity_id') == entity_id:
                    result.append(event)
                elif content.get('task_id') == entity_id:
                    result.append(event)
                elif entity_id in str(content.get('method', '')):
                    result.append(event)
        return result

    @property
    def count(self) -> int:
        """Total number of events in store."""
        return len(self._events)

    @property
    def stats(self) -> Dict[str, any]:
        """
        Get store statistics.

        Returns:
            Dict with count, heads, and type breakdown
        """
        type_counts: Dict[str, int] = {}
        for event in self._events.values():
            type_name = event.event_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            'event_count': len(self._events),
            'head_count': len(self.heads()),
            'events_by_type': type_counts,
        }

    def clear(self) -> int:
        """
        Clear all events from store.

        Returns:
            Number of events cleared

        Warning:
            This destroys all event history. Use with caution.
        """
        count = len(self._events)
        self._events.clear()
        self._children.clear()
        return count

    def __len__(self) -> int:
        """Return number of events."""
        return len(self._events)

    def __contains__(self, event_id: str) -> bool:
        """Check if event exists in store."""
        return event_id in self._events

    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryEventStore(events={len(self._events)}, heads={len(self.heads())})"
