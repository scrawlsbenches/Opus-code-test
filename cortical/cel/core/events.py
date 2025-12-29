"""
Event type definitions for the Cognitive Event Lattice.

Events are the fundamental unit of the CEL. They are:
- IMMUTABLE: Once created, never changed
- CONTENT-ADDRESSED: ID is a hash of content
- CAUSALLY LINKED: Each event references its causal parents
- TYPED: Different event types for different purposes

The Event Hierarchy:
    CognitiveEvent (base)
    ├── Observation - Something happened (passive)
    ├── Intention - Something should happen (active/task)
    ├── Fulfillment - An intention was completed
    ├── Invalidation - Something is no longer true
    ├── Compaction - Events were compressed
    └── MetaCognition - System observing itself

Design Philosophy:
    Events capture WHAT HAPPENED or WHAT IS INTENDED.
    They do NOT capture current state - that's materialized from events.
    This separation enables temporal queries and audit trails.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence

from .references import MerkleRoot, EventHorizon


class EventType(Enum):
    """Types of cognitive events."""

    # Wisdom events
    OBSERVATION = auto()     # External event observed
    INTENTION = auto()       # Task/goal created
    FULFILLMENT = auto()     # Intention completed
    INVALIDATION = auto()    # Entity/fact invalidated

    # Sanity events
    COMPACTION = auto()      # Events compressed
    MIGRATION = auto()       # Schema migrated
    REPAIR = auto()          # Self-healing action

    # Meta-cognitive events
    METACOGNITION = auto()   # System self-observation
    HEALTH_CHECK = auto()    # Health metrics recorded
    MAINTENANCE = auto()     # Self-maintenance action


@dataclass(frozen=True)
class CognitiveEvent:
    """
    Base class for all cognitive events.

    Events are immutable (frozen=True) and content-addressed.
    The ID is computed from the content, ensuring deterministic hashing.

    Attributes:
        id: Merkle root (content hash) - computed, not stored
        timestamp: When the event occurred (ISO 8601)
        event_type: Type of event
        causal_parents: Events this event causally depends on
        content: Event-specific payload
        concepts: Extracted concepts for semantic indexing
        metadata: Additional metadata (branch, session, etc.)
    """

    timestamp: str
    event_type: EventType
    causal_parents: tuple[str, ...]  # Immutable tuple for hashing
    content: Dict[str, Any]
    concepts: tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed field - not part of content hash
    _id: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def id(self) -> str:
        """
        Content-addressed ID (Merkle root).

        Computed lazily and cached. The hash includes:
        - timestamp
        - event_type
        - causal_parents (in order)
        - content (sorted keys)

        This ensures the same logical event always has the same ID.
        """
        if self._id is not None:
            return self._id

        # Compute deterministic hash
        hash_input = {
            'timestamp': self.timestamp,
            'event_type': self.event_type.name,
            'causal_parents': list(self.causal_parents),
            'content': self.content,
        }
        hash_bytes = json.dumps(hash_input, sort_keys=True).encode('utf-8')
        hash_value = hashlib.sha256(hash_bytes).hexdigest()

        # Store in mutable wrapper (frozen dataclass workaround)
        object.__setattr__(self, '_id', hash_value)
        return hash_value

    @property
    def merkle_root(self) -> MerkleRoot:
        """Get as MerkleRoot type."""
        return MerkleRoot(self.id)

    @property
    def horizon(self) -> EventHorizon:
        """Get as EventHorizon type."""
        return EventHorizon(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for storage)."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'event_type': self.event_type.name,
            'causal_parents': list(self.causal_parents),
            'content': self.content,
            'concepts': list(self.concepts),
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitiveEvent:
        """Deserialize from dictionary."""
        event = cls(
            timestamp=data['timestamp'],
            event_type=EventType[data['event_type']],
            causal_parents=tuple(data.get('causal_parents', [])),
            content=data.get('content', {}),
            concepts=tuple(data.get('concepts', [])),
            metadata=data.get('metadata', {}),
        )
        # Verify ID matches
        if 'id' in data and event.id != data['id']:
            raise ValueError(
                f"Event ID mismatch: computed {event.id}, stored {data['id']}"
            )
        return event

    def with_parent(self, parent_id: str) -> CognitiveEvent:
        """Create new event with additional causal parent."""
        return CognitiveEvent(
            timestamp=self.timestamp,
            event_type=self.event_type,
            causal_parents=self.causal_parents + (parent_id,),
            content=self.content,
            concepts=self.concepts,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class Observation(CognitiveEvent):
    """
    An observation event - something happened in the external world.

    Observations are passive recordings of external events:
    - File changes
    - User actions
    - Test results
    - Git commits

    Example:
        Observation(
            content={
                'type': 'file_modified',
                'path': 'cortical/cel/core/events.py',
                'lines_added': 150,
            }
        )
    """

    def __init__(
        self,
        content: Dict[str, Any],
        causal_parents: Sequence[str] = (),
        concepts: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        super().__init__(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=tuple(causal_parents),
            content=content,
            concepts=tuple(concepts),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class Intention(CognitiveEvent):
    """
    An intention event - something that should happen (a task/goal).

    Intentions are active desires for state changes:
    - Tasks to complete
    - Goals to achieve
    - Changes to make

    Intentions can reference the system state at creation time,
    enabling temporal queries even after the system changes.

    Content fields:
        title: str - What should happen
        priority: str - low/medium/high/critical
        category: str - feature/bugfix/refactor/etc.
        references: List[TemporalReference] - What this references
        invalidates: List[str] - What this will change
        after: List[str] - Intentions that must complete first

    Example:
        Intention(
            content={
                'title': 'Optimize GoT storage',
                'priority': 'high',
                'category': 'refactor',
                'references': [{'entity_id': 'got_storage', 'at': 'abc123'}],
                'invalidates': ['storage_format', 'edge_structure'],
            }
        )
    """

    def __init__(
        self,
        title: str,
        causal_parents: Sequence[str] = (),
        priority: str = 'medium',
        category: str = 'feature',
        description: str = '',
        references: Optional[List[Dict[str, Any]]] = None,
        invalidates: Optional[List[str]] = None,
        after: Optional[List[str]] = None,
        concepts: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        content = {
            'title': title,
            'priority': priority,
            'category': category,
            'description': description,
            'references': references or [],
            'invalidates': invalidates or [],
            'after': after or [],
        }
        super().__init__(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=EventType.INTENTION,
            causal_parents=tuple(causal_parents),
            content=content,
            concepts=tuple(concepts) or self._extract_concepts(title),
            metadata=metadata or {},
        )

    @staticmethod
    def _extract_concepts(title: str) -> tuple[str, ...]:
        """Extract concepts from title for indexing."""
        # Simple word extraction - could be more sophisticated
        words = title.lower().split()
        # Filter common words
        stop_words = {'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at'}
        return tuple(w for w in words if w not in stop_words and len(w) > 2)

    @property
    def title(self) -> str:
        return self.content['title']

    @property
    def priority(self) -> str:
        return self.content['priority']

    @property
    def category(self) -> str:
        return self.content['category']

    @property
    def references_entities(self) -> List[str]:
        """Get entity IDs this intention references."""
        return [ref['entity_id'] for ref in self.content.get('references', [])]

    @property
    def will_invalidate(self) -> List[str]:
        """Get what this intention will invalidate when fulfilled."""
        return self.content.get('invalidates', [])

    @property
    def depends_on_intentions(self) -> List[str]:
        """Get intentions that must complete before this one."""
        return self.content.get('after', [])


@dataclass(frozen=True)
class Fulfillment(CognitiveEvent):
    """
    A fulfillment event - an intention was completed.

    Fulfillments close the loop on intentions, recording:
    - Which intention was fulfilled
    - What the result was
    - What changed as a result

    Content fields:
        intention_id: str - The intention that was fulfilled
        result: Dict - Outcome of the intention
        invalidated: List[str] - What was actually invalidated
        artifacts: List[str] - Created artifacts (files, entities)

    Example:
        Fulfillment(
            intention_id='abc123...',
            result={'success': True, 'files_modified': 5},
            invalidated=['storage_format'],
            artifacts=['cortical/cel/core/events.py'],
        )
    """

    def __init__(
        self,
        intention_id: str,
        result: Optional[Dict[str, Any]] = None,
        invalidated: Optional[List[str]] = None,
        artifacts: Optional[List[str]] = None,
        causal_parents: Sequence[str] = (),
        concepts: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        # The fulfilled intention is always a causal parent
        parents = tuple(causal_parents)
        if intention_id not in parents:
            parents = (intention_id,) + parents

        content = {
            'intention_id': intention_id,
            'result': result or {},
            'invalidated': invalidated or [],
            'artifacts': artifacts or [],
        }
        super().__init__(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=EventType.FULFILLMENT,
            causal_parents=parents,
            content=content,
            concepts=tuple(concepts),
            metadata=metadata or {},
        )

    @property
    def intention_id(self) -> str:
        return self.content['intention_id']

    @property
    def result(self) -> Dict[str, Any]:
        return self.content['result']

    @property
    def was_successful(self) -> bool:
        return self.content.get('result', {}).get('success', True)


@dataclass(frozen=True)
class Invalidation(CognitiveEvent):
    """
    An invalidation event - something is no longer true.

    Invalidations explicitly mark entities or facts as invalid.
    This is different from deletion - the history remains, but
    future materializations will not include the invalidated entity.

    Content fields:
        entity_id: str - What is being invalidated
        reason: str - Why it's being invalidated
        superseded_by: Optional[str] - Replacement entity (if any)

    Example:
        Invalidation(
            entity_id='T-20251228-...',
            reason='Duplicate of T-20251227-...',
            superseded_by='T-20251227-...',
        )
    """

    def __init__(
        self,
        entity_id: str,
        reason: str = '',
        superseded_by: Optional[str] = None,
        causal_parents: Sequence[str] = (),
        concepts: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        content = {
            'entity_id': entity_id,
            'reason': reason,
            'superseded_by': superseded_by,
        }
        super().__init__(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=EventType.INVALIDATION,
            causal_parents=tuple(causal_parents),
            content=content,
            concepts=tuple(concepts),
            metadata=metadata or {},
        )

    @property
    def entity_id(self) -> str:
        return self.content['entity_id']

    @property
    def superseded_by(self) -> Optional[str]:
        return self.content.get('superseded_by')


@dataclass(frozen=True)
class Compaction(CognitiveEvent):
    """
    A compaction event - events were semantically compressed.

    Compaction creates a new event that summarizes a range of
    previous events while preserving the same materialized state.

    Content fields:
        compressed_events: List[str] - Event IDs that were compressed
        snapshot: Dict - Materialized state snapshot
        preserved_merkle_root: str - Original root for verification

    Invariant:
        materialize(compressed_events) == snapshot
    """

    def __init__(
        self,
        compressed_events: Sequence[str],
        snapshot: Dict[str, Any],
        preserved_merkle_root: str,
        causal_parents: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        content = {
            'compressed_events': list(compressed_events),
            'snapshot': snapshot,
            'preserved_merkle_root': preserved_merkle_root,
        }
        super().__init__(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=EventType.COMPACTION,
            causal_parents=tuple(causal_parents),
            content=content,
            concepts=(),
            metadata=metadata or {},
        )

    @property
    def compressed_count(self) -> int:
        return len(self.content['compressed_events'])


@dataclass(frozen=True)
class MetaCognition(CognitiveEvent):
    """
    A meta-cognitive event - the system observing itself.

    MetaCognition events capture the system's self-awareness:
    - Health check results
    - Performance observations
    - Self-modification decisions

    This is the "thinking about thinking" layer.

    Content fields:
        observation_type: str - What kind of self-observation
        metrics: Dict - Observed metrics
        conclusions: List[str] - Conclusions drawn
        actions_triggered: List[str] - Actions taken as result

    Example:
        MetaCognition(
            observation_type='health_check',
            metrics={'cache_hit_rate': 0.85, 'event_count': 1000},
            conclusions=['Cache is healthy', 'May need compaction soon'],
            actions_triggered=['schedule_compaction_check'],
        )
    """

    def __init__(
        self,
        observation_type: str,
        metrics: Optional[Dict[str, Any]] = None,
        conclusions: Optional[List[str]] = None,
        actions_triggered: Optional[List[str]] = None,
        causal_parents: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        content = {
            'observation_type': observation_type,
            'metrics': metrics or {},
            'conclusions': conclusions or [],
            'actions_triggered': actions_triggered or [],
        }
        super().__init__(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=EventType.METACOGNITION,
            causal_parents=tuple(causal_parents),
            content=content,
            concepts=('metacognition', observation_type),
            metadata=metadata or {},
        )

    @property
    def observation_type(self) -> str:
        return self.content['observation_type']

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.content['metrics']

    @property
    def triggered_actions(self) -> List[str]:
        return self.content.get('actions_triggered', [])
