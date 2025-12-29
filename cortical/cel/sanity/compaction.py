"""
Compaction strategies for the Cognitive Event Lattice.

Compaction reduces storage while preserving semantic meaning.
Unlike traditional compression (bit-level), compaction works
at the semantic level - understanding what information matters.

Key Insight:
    Events are immutable, but their information can be
    summarized. A sequence of "update" events can be
    compacted to the final state plus a summary.

Compaction Strategies:
    - Time Window: Compress events older than threshold
    - Semantic: Merge semantically similar events
    - Causal: Flatten redundant causal chains
    - Archival: Move old events to cold storage

Trade-offs:
    Compaction loses detail for storage efficiency.
    The system must decide: What history matters?

This module implements Level 6 of the CEL architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from ..core.events import CognitiveEvent, Compaction, EventType
from ..core.protocols import CompactionStrategy, EventStore
from ..core.references import MerkleRoot


@dataclass
class CompactionResult:
    """
    Result of a compaction operation.

    Tracks what was compacted and the compression achieved.

    Attributes:
        original_count: Events before compaction
        compacted_count: Events after compaction
        events_removed: IDs of removed events
        events_created: IDs of new compaction events
        bytes_saved: Estimated storage savings
        started_at: When compaction started
        completed_at: When compaction finished
    """

    original_count: int
    compacted_count: int
    events_removed: List[str]
    events_created: List[str]
    bytes_saved: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_count == 0:
            return 1.0
        return self.compacted_count / self.original_count

    @property
    def duration(self) -> Optional[timedelta]:
        """Get compaction duration."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            'original_count': self.original_count,
            'compacted_count': self.compacted_count,
            'compression_ratio': self.compression_ratio,
            'events_removed': len(self.events_removed),
            'events_created': len(self.events_created),
            'bytes_saved': self.bytes_saved,
            'started_at': self.started_at.isoformat(),
            'completed_at': (
                self.completed_at.isoformat()
                if self.completed_at else None
            ),
            'duration_seconds': (
                self.duration.total_seconds()
                if self.duration else None
            ),
        }


class BaseCompactor(ABC):
    """
    Base class for compaction strategies.

    Provides common infrastructure for event iteration,
    result tracking, and safety checks.
    """

    def __init__(self, event_store: EventStore):
        """Initialize with event store."""
        self._store = event_store
        self._preserve_ids: Set[str] = set()

    def preserve(self, event_id: str) -> None:
        """Mark an event ID as non-compactable."""
        self._preserve_ids.add(event_id)

    def is_preserved(self, event_id: str) -> bool:
        """Check if an event is preserved."""
        return event_id in self._preserve_ids

    @abstractmethod
    def identify_compactable(self) -> List[List[CognitiveEvent]]:
        """
        Identify groups of events that can be compacted.

        Returns:
            List of event groups, each group will be compacted together
        """
        pass

    @abstractmethod
    def compact_group(
        self,
        events: List[CognitiveEvent],
    ) -> Tuple[CognitiveEvent, List[str]]:
        """
        Compact a group of events into a single event.

        Args:
            events: Events to compact

        Returns:
            (compacted_event, removed_event_ids)
        """
        pass

    def compact(self) -> CompactionResult:
        """
        Execute compaction.

        Returns:
            CompactionResult with statistics
        """
        result = CompactionResult(
            original_count=sum(1 for _ in self._store.iterate()),
            compacted_count=0,
            events_removed=[],
            events_created=[],
        )

        # Identify compactable groups
        groups = self.identify_compactable()

        # Process each group
        for group in groups:
            # Filter out preserved events
            compactable = [
                e for e in group
                if not self.is_preserved(e.id)
            ]

            if len(compactable) < 2:
                continue

            # Create compacted event
            compacted, removed_ids = self.compact_group(compactable)

            # Append compacted event
            root = self._store.append(compacted)
            result.events_created.append(root.value)
            result.events_removed.extend(removed_ids)

        result.compacted_count = (
            result.original_count - len(result.events_removed) +
            len(result.events_created)
        )
        result.completed_at = datetime.now()

        return result


class TimeWindowCompactor(BaseCompactor):
    """
    Compacts events based on time windows.

    Events within the same time window that affect the same
    entity are merged into summary events.

    Strategy:
        1. Group events by entity and time window
        2. For each group, keep first and last state
        3. Create summary event with intermediate changes

    Implements: CompactionStrategy protocol
    """

    def __init__(
        self,
        event_store: EventStore,
        window_size: timedelta = timedelta(hours=24),
        min_age: timedelta = timedelta(days=7),
    ):
        """
        Initialize time window compactor.

        Args:
            event_store: The event store
            window_size: Size of time windows
            min_age: Minimum age before compaction
        """
        super().__init__(event_store)
        self._window_size = window_size
        self._min_age = min_age

    def identify_compactable(self) -> List[List[CognitiveEvent]]:
        """Identify time-window groups for compaction."""
        cutoff = datetime.now(timezone.utc) - self._min_age

        # Group by entity_id and time window
        groups: Dict[Tuple[str, int], List[CognitiveEvent]] = {}

        for event in self._store.iterate():
            # Parse timestamp
            try:
                ts = datetime.fromisoformat(event.timestamp)
                # Ensure timezone-aware for comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Skip recent events
            if ts > cutoff:
                continue

            # Extract entity ID if present
            entity_id = event.content.get('entity_id', 'default')

            # Calculate window index
            window_idx = int(
                (ts.timestamp() // self._window_size.total_seconds())
            )

            key = (entity_id, window_idx)
            if key not in groups:
                groups[key] = []
            groups[key].append(event)

        # Return only groups with multiple events
        return [g for g in groups.values() if len(g) >= 2]

    def compact_group(
        self,
        events: List[CognitiveEvent],
    ) -> Tuple[CognitiveEvent, List[str]]:
        """Compact a time-window group."""
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        first = sorted_events[0]
        last = sorted_events[-1]

        # Collect all concepts
        all_concepts = set()
        for e in sorted_events:
            all_concepts.update(e.concepts)

        # Create summary content
        summary = {
            'compaction_type': 'time_window',
            'first_event_id': first.id,
            'last_event_id': last.id,
            'event_count': len(sorted_events),
            'time_span': {
                'start': first.timestamp,
                'end': last.timestamp,
            },
            'final_state': last.content,
            'entity_id': first.content.get('entity_id'),
        }

        # Create compaction event
        compacted = Compaction(
            compressed_events=tuple(e.id for e in sorted_events),
            snapshot=summary,
            preserved_merkle_root=last.id,  # Use last event's ID as merkle reference
        )

        removed_ids = [e.id for e in sorted_events[:-1]]  # Keep last event
        return compacted, removed_ids

    def should_compact(self) -> bool:
        """Check if compaction is recommended."""
        cutoff = datetime.now(timezone.utc) - self._min_age
        old_count = 0

        for event in self._store.iterate():
            try:
                ts = datetime.fromisoformat(event.timestamp)
                # Ensure timezone-aware for comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    old_count += 1
            except ValueError:
                continue

        # Recommend compaction if >100 old events
        return old_count > 100


class SemanticCompactor(BaseCompactor):
    """
    Compacts events based on semantic similarity.

    Events with similar meaning are merged, preserving
    the essential information while removing redundancy.

    Strategy:
        1. Group events by semantic similarity
        2. Extract key information from each group
        3. Create summary that preserves meaning

    This is more sophisticated than time-window compaction
    because it understands content, not just timing.

    Implements: CompactionStrategy protocol
    """

    def __init__(
        self,
        event_store: EventStore,
        similarity_threshold: float = 0.8,
        min_group_size: int = 3,
    ):
        """
        Initialize semantic compactor.

        Args:
            event_store: The event store
            similarity_threshold: Minimum similarity to group
            min_group_size: Minimum events to form a group
        """
        super().__init__(event_store)
        self._threshold = similarity_threshold
        self._min_group = min_group_size

    def identify_compactable(self) -> List[List[CognitiveEvent]]:
        """Identify semantically similar event groups."""
        # Group by event type first (coarse filter)
        by_type: Dict[EventType, List[CognitiveEvent]] = {}

        for event in self._store.iterate():
            if event.event_type not in by_type:
                by_type[event.event_type] = []
            by_type[event.event_type].append(event)

        groups = []

        for event_type, events in by_type.items():
            # Within each type, cluster by concept overlap
            clusters = self._cluster_by_concepts(events)
            groups.extend(clusters)

        return groups

    def _cluster_by_concepts(
        self,
        events: List[CognitiveEvent],
    ) -> List[List[CognitiveEvent]]:
        """Cluster events by concept overlap."""
        if len(events) < self._min_group:
            return []

        # Simple greedy clustering
        clusters: List[List[CognitiveEvent]] = []
        used: Set[str] = set()

        for event in events:
            if event.id in used:
                continue

            cluster = [event]
            used.add(event.id)
            event_concepts = set(event.concepts)

            for other in events:
                if other.id in used:
                    continue

                other_concepts = set(other.concepts)
                similarity = self._jaccard_similarity(
                    event_concepts, other_concepts
                )

                if similarity >= self._threshold:
                    cluster.append(other)
                    used.add(other.id)

            if len(cluster) >= self._min_group:
                clusters.append(cluster)

        return clusters

    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def compact_group(
        self,
        events: List[CognitiveEvent],
    ) -> Tuple[CognitiveEvent, List[str]]:
        """Compact a semantic group."""
        # Collect all unique concepts
        all_concepts = set()
        for e in events:
            all_concepts.update(e.concepts)

        # Find the "best" representative event (most concepts)
        representative = max(events, key=lambda e: len(e.concepts))

        # Create semantic summary
        summary = {
            'compaction_type': 'semantic',
            'representative_id': representative.id,
            'event_count': len(events),
            'shared_concepts': list(
                set.intersection(*[set(e.concepts) for e in events])
            ),
            'all_concepts': list(all_concepts),
            'representative_content': representative.content,
        }

        # Create compaction event
        compacted = Compaction(
            compressed_events=tuple(e.id for e in events),
            snapshot=summary,
            preserved_merkle_root=representative.id,  # Use representative as merkle reference
        )

        # Remove all but representative
        removed_ids = [e.id for e in events if e.id != representative.id]
        return compacted, removed_ids

    def should_compact(self) -> bool:
        """Check if compaction is recommended."""
        # Count total events
        total = sum(1 for _ in self._store.iterate())

        # Check for high duplication (many events with shared concepts)
        concept_counts: Dict[str, int] = {}
        for event in self._store.iterate():
            for concept in event.concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

        if not concept_counts:
            return False

        # If average concept appears in >10% of events, recommend compaction
        avg_count = sum(concept_counts.values()) / len(concept_counts)
        return avg_count > total * 0.1


class CausalChainCompactor(BaseCompactor):
    """
    Compacts redundant causal chains.

    When A -> B -> C and we only need A -> C, the
    intermediate event B can be summarized.

    Strategy:
        1. Find long causal chains
        2. Identify intermediate events with no branches
        3. Compact to direct links with summaries

    This preserves causal structure while reducing depth.
    """

    def __init__(
        self,
        event_store: EventStore,
        max_chain_length: int = 5,
    ):
        """
        Initialize causal chain compactor.

        Args:
            event_store: The event store
            max_chain_length: Chains longer than this get compacted
        """
        super().__init__(event_store)
        self._max_chain = max_chain_length

    def identify_compactable(self) -> List[List[CognitiveEvent]]:
        """Identify long causal chains."""
        # Build event lookup
        events_by_id: Dict[str, CognitiveEvent] = {}
        children: Dict[str, List[str]] = {}

        for event in self._store.iterate():
            events_by_id[event.id] = event
            for parent_id in event.causal_parents:
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(event.id)

        # Find chain starts (events with no parents or multiple children)
        chain_starts = []
        for event in events_by_id.values():
            if not event.causal_parents:
                chain_starts.append(event.id)
            elif event.id in children and len(children[event.id]) > 1:
                # Branch point - each child starts a potential chain
                chain_starts.extend(children[event.id])

        # Follow chains
        chains = []
        for start_id in chain_starts:
            chain = self._follow_chain(
                start_id, events_by_id, children
            )
            if len(chain) > self._max_chain:
                chains.append([events_by_id[eid] for eid in chain])

        return chains

    def _follow_chain(
        self,
        start_id: str,
        events: Dict[str, CognitiveEvent],
        children: Dict[str, List[str]],
    ) -> List[str]:
        """Follow a single causal chain."""
        chain = [start_id]
        current = start_id

        while current in children:
            kids = children[current]
            if len(kids) != 1:
                break  # Branch or end
            current = kids[0]
            chain.append(current)

        return chain

    def compact_group(
        self,
        events: List[CognitiveEvent],
    ) -> Tuple[CognitiveEvent, List[str]]:
        """Compact a causal chain."""
        first = events[0]
        last = events[-1]

        # Create chain summary
        summary = {
            'compaction_type': 'causal_chain',
            'chain_start': first.id,
            'chain_end': last.id,
            'chain_length': len(events),
            'intermediate_events': [e.id for e in events[1:-1]],
            'start_content': first.content,
            'end_content': last.content,
        }

        # Create compaction event
        compacted = Compaction(
            compressed_events=tuple(e.id for e in events),
            snapshot=summary,
            preserved_merkle_root=last.id,  # Use last event as merkle reference
        )

        # Remove intermediate events
        removed_ids = [e.id for e in events[1:-1]]
        return compacted, removed_ids

    def should_compact(self) -> bool:
        """Check if chain compaction is recommended."""
        # Look for any chain longer than threshold
        groups = self.identify_compactable()
        return len(groups) > 0


# =============================================================================
# COMPACTION UTILITIES
# =============================================================================


def create_compaction_schedule(
    store: EventStore,
) -> List[Tuple[str, BaseCompactor]]:
    """
    Create a recommended compaction schedule.

    Returns compactors in recommended order of execution.
    """
    schedule = []

    # 1. Time-based compaction first (oldest events)
    time_compactor = TimeWindowCompactor(
        store,
        window_size=timedelta(hours=24),
        min_age=timedelta(days=30),
    )
    if time_compactor.should_compact():
        schedule.append(('time_window', time_compactor))

    # 2. Semantic compaction for redundancy
    semantic_compactor = SemanticCompactor(
        store,
        similarity_threshold=0.85,
        min_group_size=3,
    )
    if semantic_compactor.should_compact():
        schedule.append(('semantic', semantic_compactor))

    # 3. Causal chain compaction
    causal_compactor = CausalChainCompactor(
        store,
        max_chain_length=10,
    )
    if causal_compactor.should_compact():
        schedule.append(('causal_chain', causal_compactor))

    return schedule


def estimate_compaction_savings(store: EventStore) -> Dict[str, Any]:
    """
    Estimate potential savings from compaction.

    Useful for deciding when to run compaction.
    """
    total_events = 0
    total_concepts = 0
    concept_reuse: Dict[str, int] = {}
    events_by_type: Dict[str, int] = {}

    for event in store.iterate():
        total_events += 1
        total_concepts += len(event.concepts)

        et = event.event_type.name
        events_by_type[et] = events_by_type.get(et, 0) + 1

        for concept in event.concepts:
            concept_reuse[concept] = concept_reuse.get(concept, 0) + 1

    # Calculate potential savings
    duplicate_concepts = sum(
        count - 1 for count in concept_reuse.values() if count > 1
    )

    return {
        'total_events': total_events,
        'events_by_type': events_by_type,
        'unique_concepts': len(concept_reuse),
        'total_concept_refs': total_concepts,
        'duplicate_concept_refs': duplicate_concepts,
        'estimated_savings_percent': (
            (duplicate_concepts / max(total_concepts, 1)) * 100
        ),
    }
