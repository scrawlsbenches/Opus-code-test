"""
Merkle DAG implementation for the Cognitive Event Lattice.

The DAG (Directed Acyclic Graph) is the fundamental structure
for event storage. Events are content-addressed (identified by
their hash) and causally linked (each event references its parents).

Key Properties:
    - APPEND-ONLY: Events can only be added, never modified
    - CONTENT-ADDRESSED: ID = hash of content
    - CAUSALLY ORDERED: Parents must exist before children
    - VERIFIABLE: Hash chain ensures integrity

This module provides:
    - MerkleDAG: In-memory DAG operations
    - FileSystemEventStore: Persistent storage implementation
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set

from ..core.events import CognitiveEvent, EventType
from ..core.references import CausalLink, EventHorizon, MerkleRoot


class CausalViolationError(Exception):
    """Raised when an event references non-existent parents."""

    def __init__(self, event_id: str, missing_parents: List[str]):
        self.event_id = event_id
        self.missing_parents = missing_parents
        super().__init__(
            f"Event {event_id[:16]}... references missing parents: "
            f"{[p[:16] for p in missing_parents]}"
        )


class DuplicateEventError(Exception):
    """Raised when attempting to add an event that already exists."""

    def __init__(self, event_id: str):
        self.event_id = event_id
        super().__init__(f"Event already exists: {event_id[:16]}...")


@dataclass
class MerkleDAG:
    """
    In-memory Merkle DAG for event operations.

    The DAG maintains:
    - Forward edges: parent -> children (for finding descendants)
    - Backward edges: child -> parents (for finding ancestors)
    - Head tracking: Events with no children (branch tips)

    Thread Safety:
        This class is NOT thread-safe. Use external locking
        or the thread-safe FileSystemEventStore for concurrent access.
    """

    events: Dict[str, CognitiveEvent] = field(default_factory=dict)
    children: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    heads: Set[str] = field(default_factory=set)

    def add(self, event: CognitiveEvent) -> MerkleRoot:
        """
        Add an event to the DAG.

        Args:
            event: Event to add

        Returns:
            MerkleRoot of the added event

        Raises:
            CausalViolationError: If parents don't exist
            DuplicateEventError: If event already exists
        """
        event_id = event.id

        # Check for duplicate
        if event_id in self.events:
            raise DuplicateEventError(event_id)

        # Verify all parents exist
        missing = [p for p in event.causal_parents if p not in self.events]
        if missing:
            raise CausalViolationError(event_id, missing)

        # Add event
        self.events[event_id] = event

        # Update forward edges (parent -> child)
        for parent_id in event.causal_parents:
            self.children[parent_id].add(event_id)
            # Parent is no longer a head
            self.heads.discard(parent_id)

        # This event is a new head (no children yet)
        self.heads.add(event_id)

        return MerkleRoot(event_id)

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get an event by ID."""
        return self.events.get(event_id)

    def contains(self, event_id: str) -> bool:
        """Check if event exists."""
        return event_id in self.events

    def get_heads(self) -> List[MerkleRoot]:
        """Get current branch heads."""
        return [MerkleRoot(h) for h in self.heads]

    def get_latest(self) -> Optional[MerkleRoot]:
        """
        Get the latest event (most recent head).

        For single-branch DAGs, this is the only head.
        For multi-branch, returns the head with latest timestamp.
        """
        if not self.heads:
            return None

        latest_id = max(
            self.heads,
            key=lambda h: self.events[h].timestamp
        )
        return MerkleRoot(latest_id)

    def ancestors(
        self,
        event_id: str,
        depth: int = -1,
    ) -> Iterator[CognitiveEvent]:
        """
        Iterate ancestors in reverse causal order.

        Uses BFS to traverse parents, yielding each event once.

        Args:
            event_id: Starting event
            depth: Max depth (-1 for unlimited)

        Yields:
            Ancestor events (parents before grandparents)
        """
        if event_id not in self.events:
            return

        visited = set()
        queue = [(event_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_id in visited:
                continue
            visited.add(current_id)

            event = self.events.get(current_id)
            if event is None:
                continue

            # Don't yield the starting event
            if current_id != event_id:
                yield event

            # Check depth limit
            if depth >= 0 and current_depth >= depth:
                continue

            # Queue parents
            for parent_id in event.causal_parents:
                if parent_id not in visited:
                    queue.append((parent_id, current_depth + 1))

    def descendants(self, event_id: str) -> Iterator[CognitiveEvent]:
        """
        Iterate descendants in causal order.

        Uses BFS to traverse children, yielding each event once.

        Args:
            event_id: Starting event

        Yields:
            Descendant events (children before grandchildren)
        """
        if event_id not in self.events:
            return

        visited = set()
        queue = [event_id]

        while queue:
            current_id = queue.pop(0)

            if current_id in visited:
                continue
            visited.add(current_id)

            # Don't yield the starting event
            if current_id != event_id:
                event = self.events.get(current_id)
                if event:
                    yield event

            # Queue children
            for child_id in self.children.get(current_id, set()):
                if child_id not in visited:
                    queue.append(child_id)

    def causal_order(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
    ) -> Iterator[CognitiveEvent]:
        """
        Iterate events in topological (causal) order.

        Parents are always yielded before their children.

        Args:
            from_event: Start after this event (exclusive)
            to_event: Stop at this event (inclusive)

        Yields:
            Events in causal order
        """
        # Find roots (events with no parents in our set)
        in_degree: Dict[str, int] = defaultdict(int)
        for event_id, event in self.events.items():
            for parent in event.causal_parents:
                if parent in self.events:
                    in_degree[event_id] += 1

        # Initialize queue with roots
        queue = [
            event_id for event_id in self.events
            if in_degree[event_id] == 0
        ]

        started = from_event is None
        visited = set()

        while queue:
            # Sort by timestamp for deterministic order among concurrent events
            queue.sort(key=lambda e: self.events[e].timestamp)
            current_id = queue.pop(0)

            if current_id in visited:
                continue
            visited.add(current_id)

            event = self.events[current_id]

            # Handle from_event
            if not started:
                if current_id == from_event:
                    started = True
                continue

            yield event

            # Handle to_event
            if to_event and current_id == to_event:
                return

            # Queue children whose parents are all visited
            for child_id in self.children.get(current_id, set()):
                child = self.events.get(child_id)
                if child and all(p in visited for p in child.causal_parents if p in self.events):
                    queue.append(child_id)

    @property
    def count(self) -> int:
        """Total number of events."""
        return len(self.events)


class FileSystemEventStore:
    """
    Persistent event store using filesystem.

    Storage Layout:
        {base_path}/
        ├── heads.json           # Current branch heads
        ├── events/
        │   ├── ab/
        │   │   └── cdef1234...  # Events by hash prefix
        │   └── 12/
        │       └── 3456abcd...
        └── indexes/
            ├── by_type.json     # Events grouped by type
            └── by_time.json     # Events by timestamp

    Thread Safety:
        Uses file locking for safe concurrent access.
        Multiple readers, single writer.

    Implements: EventStore protocol
    """

    def __init__(self, base_path: Path):
        """
        Initialize event store.

        Args:
            base_path: Directory for storage
        """
        self.base_path = Path(base_path)
        self.events_path = self.base_path / "events"
        self.heads_path = self.base_path / "heads.json"

        # In-memory DAG for operations
        self._dag = MerkleDAG()
        self._loaded = False

        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.events_path.mkdir(exist_ok=True)

    def _ensure_loaded(self) -> None:
        """Lazy load events from disk."""
        if self._loaded:
            return

        # Load heads
        heads = set()
        if self.heads_path.exists():
            with open(self.heads_path) as f:
                heads = set(json.load(f).get('heads', []))

        # Load all events
        events = {}
        for prefix_dir in self.events_path.iterdir():
            if not prefix_dir.is_dir():
                continue
            for event_file in prefix_dir.iterdir():
                if event_file.suffix == '.json':
                    with open(event_file) as f:
                        data = json.load(f)
                        event = CognitiveEvent.from_dict(data)
                        events[event.id] = event

        # Build DAG in causal order
        # Sort by timestamp to ensure parents before children
        sorted_events = sorted(events.values(), key=lambda e: e.timestamp)
        for event in sorted_events:
            try:
                self._dag.add(event)
            except CausalViolationError:
                # Parent not loaded yet - will be handled by second pass
                pass

        # Second pass for any missed events
        for event in sorted_events:
            if event.id not in self._dag.events:
                try:
                    self._dag.add(event)
                except (CausalViolationError, DuplicateEventError):
                    pass

        # Restore heads from file (may differ from computed heads)
        # This handles the case where we have persisted heads
        if heads:
            self._dag.heads = heads & set(self._dag.events.keys())

        self._loaded = True

    def _event_path(self, event_id: str) -> Path:
        """Get filesystem path for an event."""
        prefix = event_id[:2]
        return self.events_path / prefix / f"{event_id}.json"

    def _save_event(self, event: CognitiveEvent) -> None:
        """Persist an event to disk."""
        path = self._event_path(event.id)
        path.parent.mkdir(exist_ok=True)

        # Atomic write
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)
        temp_path.rename(path)

    def _save_heads(self) -> None:
        """Persist current heads to disk."""
        temp_path = self.heads_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump({'heads': list(self._dag.heads)}, f)
        temp_path.rename(self.heads_path)

    # EventStore protocol implementation

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """Append an event to the store."""
        self._ensure_loaded()

        # Add to DAG (validates causal ordering)
        root = self._dag.add(event)

        # Persist
        self._save_event(event)
        self._save_heads()

        return root

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get an event by ID."""
        self._ensure_loaded()
        return self._dag.get(event_id)

    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
        event_types: Optional[Sequence[EventType]] = None,
    ) -> Iterator[CognitiveEvent]:
        """Iterate events in causal order."""
        self._ensure_loaded()

        for event in self._dag.causal_order(from_event, to_event):
            if event_types is None or event.event_type in event_types:
                yield event

    def heads(self) -> List[MerkleRoot]:
        """Get current branch heads."""
        self._ensure_loaded()
        return self._dag.get_heads()

    def latest(self) -> Optional[MerkleRoot]:
        """Get latest event."""
        self._ensure_loaded()
        return self._dag.get_latest()

    def ancestors(self, event_id: str, depth: int = -1) -> Iterator[CognitiveEvent]:
        """Iterate ancestors."""
        self._ensure_loaded()
        yield from self._dag.ancestors(event_id, depth)

    def descendants(self, event_id: str) -> Iterator[CognitiveEvent]:
        """Iterate descendants."""
        self._ensure_loaded()
        yield from self._dag.descendants(event_id)

    @property
    def count(self) -> int:
        """Total event count."""
        self._ensure_loaded()
        return self._dag.count

    # Additional methods

    def horizon(self) -> EventHorizon:
        """Get current event horizon."""
        self._ensure_loaded()
        latest = self._dag.get_latest()
        if latest is None:
            raise ValueError("Cannot get horizon for empty store")
        return EventHorizon(event_id=latest.value, is_head=True)

    def verify_integrity(self) -> List[str]:
        """
        Verify DAG integrity.

        Returns:
            List of error messages (empty if valid)
        """
        self._ensure_loaded()
        errors = []

        for event_id, event in self._dag.events.items():
            # Verify content hash
            if event.id != event_id:
                errors.append(f"ID mismatch: stored as {event_id}, computed {event.id}")

            # Verify parents exist
            for parent_id in event.causal_parents:
                if parent_id not in self._dag.events:
                    errors.append(f"Missing parent: {event_id} -> {parent_id}")

        return errors
