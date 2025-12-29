"""
Bridge adapter between the Cognitive Event Lattice and Graph of Thought.

This adapter enables gradual migration from GoT to CEL by:
1. Reading GoT entities as CEL events
2. Writing CEL events back to GoT format
3. Maintaining both systems in parallel during transition

Key Design:
    GoT entities are mutable with versioning.
    CEL events are immutable with causal links.

    The bridge reconciles these by:
    - Treating each GoT entity version as a CEL event
    - Using GoT entity IDs as CEL entity references
    - Mapping GoT edges to CEL causal links

Migration Strategy:
    Phase 1: Read GoT, write both (current adapter)
    Phase 2: Read CEL, write both
    Phase 3: Read/write CEL only, retire GoT

This module provides the Phase 1 bridge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from ..core.events import CognitiveEvent, EventType, Observation, Intention
from ..core.protocols import EventStore
from ..core.references import MerkleRoot, CausalLink


# Import GoT types if available
try:
    from cortical.got.types import (
        Entity,
        Task,
        Decision,
        Sprint,
        Epic,
        Edge,
        Handoff,
        Document,
        VALID_ENTITY_TYPES,
    )
    GOT_AVAILABLE = True
except ImportError:
    GOT_AVAILABLE = False
    Entity = None
    Task = None
    Decision = None
    Sprint = None
    Epic = None
    Edge = None
    Handoff = None
    Document = None
    VALID_ENTITY_TYPES = set()


@dataclass
class GoTEventAdapter:
    """
    Adapts GoT entities to CEL events.

    Transforms GoT's mutable entity model to CEL's immutable
    event model. Each GoT entity version becomes a CEL event.

    Mapping:
        GoT Task (pending)     -> CEL Intention
        GoT Task (completed)   -> CEL Intention + Fulfillment
        GoT Decision           -> CEL Observation (category=decision)
        GoT Sprint             -> CEL Observation (category=sprint)
        GoT Edge               -> CEL CausalLink (metadata)
    """

    @staticmethod
    def entity_to_event(entity: 'Entity') -> CognitiveEvent:
        """
        Convert a GoT entity to a CEL event.

        Args:
            entity: GoT entity to convert

        Returns:
            Corresponding CEL event
        """
        if not GOT_AVAILABLE:
            raise ImportError("GoT types not available")

        entity_type = entity.entity_type

        if entity_type == 'task':
            return GoTEventAdapter._task_to_event(entity)
        elif entity_type == 'decision':
            return GoTEventAdapter._decision_to_event(entity)
        elif entity_type == 'sprint':
            return GoTEventAdapter._sprint_to_event(entity)
        elif entity_type == 'epic':
            return GoTEventAdapter._epic_to_event(entity)
        elif entity_type == 'handoff':
            return GoTEventAdapter._handoff_to_event(entity)
        elif entity_type == 'document':
            return GoTEventAdapter._document_to_event(entity)
        else:
            # Generic observation for unknown types
            return GoTEventAdapter._generic_to_event(entity)

    @staticmethod
    def _task_to_event(task: 'Task') -> CognitiveEvent:
        """Convert Task to Intention event."""
        # Determine event type based on status
        if task.status == 'completed':
            event_type = EventType.FULFILLMENT
        elif task.status == 'blocked':
            event_type = EventType.OBSERVATION
        else:
            event_type = EventType.INTENTION

        # Extract concepts from task
        concepts = []
        if task.title:
            concepts.extend(task.title.lower().split()[:5])
        if task.properties.get('category'):
            concepts.append(task.properties['category'])
        if task.properties.get('tags'):
            concepts.extend(task.properties['tags'])

        content = {
            'entity_id': task.id,
            'entity_type': 'task',
            'title': task.title,
            'description': task.description,
            'status': task.status,
            'priority': task.priority,
            'properties': task.properties,
            'metadata': task.metadata,
            'got_version': task.version,
        }

        return CognitiveEvent(
            timestamp=task.modified_at,
            event_type=event_type,
            causal_parents=(),  # Will be filled by store
            content=content,
            concepts=tuple(set(concepts)),
        )

    @staticmethod
    def _decision_to_event(decision: 'Decision') -> CognitiveEvent:
        """Convert Decision to Observation event."""
        concepts = []
        if decision.title:
            concepts.extend(decision.title.lower().split()[:5])
        concepts.append('decision')

        content = {
            'entity_id': decision.id,
            'entity_type': 'decision',
            'title': decision.title,
            'rationale': decision.rationale,
            'affects': decision.affects,
            'properties': decision.properties,
            'got_version': decision.version,
            'category': 'decision',
        }

        return CognitiveEvent(
            timestamp=decision.modified_at,
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content=content,
            concepts=tuple(set(concepts)),
        )

    @staticmethod
    def _sprint_to_event(sprint: 'Sprint') -> CognitiveEvent:
        """Convert Sprint to Observation event."""
        concepts = ['sprint']
        if sprint.title:
            concepts.extend(sprint.title.lower().split()[:3])
        if sprint.epic_id:
            concepts.append('epic')

        content = {
            'entity_id': sprint.id,
            'entity_type': 'sprint',
            'title': sprint.title,
            'status': sprint.status,
            'epic_id': sprint.epic_id,
            'number': sprint.number,
            'goals': sprint.goals,
            'notes': sprint.notes,
            'properties': sprint.properties,
            'metadata': sprint.metadata,
            'got_version': sprint.version,
            'category': 'sprint',
        }

        return CognitiveEvent(
            timestamp=sprint.modified_at,
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content=content,
            concepts=tuple(set(concepts)),
        )

    @staticmethod
    def _epic_to_event(epic: 'Epic') -> CognitiveEvent:
        """Convert Epic to Observation event."""
        concepts = ['epic']
        if epic.title:
            concepts.extend(epic.title.lower().split()[:3])

        content = {
            'entity_id': epic.id,
            'entity_type': 'epic',
            'title': epic.title,
            'status': epic.status,
            'phase': epic.phase,
            'phases': epic.phases,
            'properties': epic.properties,
            'metadata': epic.metadata,
            'got_version': epic.version,
            'category': 'epic',
        }

        return CognitiveEvent(
            timestamp=epic.modified_at,
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content=content,
            concepts=tuple(set(concepts)),
        )

    @staticmethod
    def _handoff_to_event(handoff: 'Handoff') -> CognitiveEvent:
        """Convert Handoff to Observation event."""
        concepts = ['handoff']
        if handoff.status == 'completed':
            concepts.append('completed')
        concepts.append('agent-coordination')

        content = {
            'entity_id': handoff.id,
            'entity_type': 'handoff',
            'source_agent': handoff.source_agent,
            'target_agent': handoff.target_agent,
            'task_id': handoff.task_id,
            'status': handoff.status,
            'instructions': handoff.instructions,
            'context': handoff.context,
            'result': handoff.result,
            'properties': handoff.properties,
            'got_version': handoff.version,
            'category': 'handoff',
        }

        return CognitiveEvent(
            timestamp=handoff.modified_at,
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content=content,
            concepts=tuple(set(concepts)),
        )

    @staticmethod
    def _document_to_event(doc: 'Document') -> CognitiveEvent:
        """Convert Document to Observation event."""
        concepts = ['document', doc.doc_type]
        concepts.extend(doc.tags[:3])

        content = {
            'entity_id': doc.id,
            'entity_type': 'document',
            'path': doc.path,
            'title': doc.title,
            'doc_type': doc.doc_type,
            'tags': doc.tags,
            'category': doc.category,
            'properties': doc.properties,
            'metadata': doc.metadata,
            'got_version': doc.version,
        }

        return CognitiveEvent(
            timestamp=doc.modified_at,
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content=content,
            concepts=tuple(set(c for c in concepts if c)),
        )

    @staticmethod
    def _generic_to_event(entity: 'Entity') -> CognitiveEvent:
        """Convert unknown entity type to generic Observation."""
        content = {
            'entity_id': entity.id,
            'entity_type': entity.entity_type,
            'got_version': entity.version,
            'raw_data': entity.to_dict(),
        }

        return CognitiveEvent(
            timestamp=entity.modified_at,
            event_type=EventType.OBSERVATION,
            causal_parents=(),
            content=content,
            concepts=(entity.entity_type,),
        )


@dataclass
class GoTEntityAdapter:
    """
    Adapts CEL events back to GoT entities.

    Enables writing to GoT from CEL events, maintaining
    both systems during migration.
    """

    @staticmethod
    def event_to_entity(event: CognitiveEvent) -> Optional['Entity']:
        """
        Convert a CEL event to a GoT entity.

        Args:
            event: CEL event to convert

        Returns:
            GoT entity, or None if not mappable
        """
        if not GOT_AVAILABLE:
            raise ImportError("GoT types not available")

        entity_type = event.content.get('entity_type')
        if not entity_type:
            return None

        if entity_type == 'task':
            return GoTEntityAdapter._event_to_task(event)
        elif entity_type == 'decision':
            return GoTEntityAdapter._event_to_decision(event)
        elif entity_type == 'sprint':
            return GoTEntityAdapter._event_to_sprint(event)
        elif entity_type == 'epic':
            return GoTEntityAdapter._event_to_epic(event)
        elif entity_type == 'handoff':
            return GoTEntityAdapter._event_to_handoff(event)
        elif entity_type == 'document':
            return GoTEntityAdapter._event_to_document(event)
        else:
            return None

    @staticmethod
    def _event_to_task(event: CognitiveEvent) -> 'Task':
        """Convert event back to Task."""
        content = event.content
        return Task(
            id=content.get('entity_id', ''),
            title=content.get('title', ''),
            status=content.get('status', 'pending'),
            priority=content.get('priority', 'medium'),
            description=content.get('description', ''),
            properties=content.get('properties', {}),
            metadata=content.get('metadata', {}),
            version=content.get('got_version', 1),
            created_at=content.get('created_at', event.timestamp),
            modified_at=event.timestamp,
        )

    @staticmethod
    def _event_to_decision(event: CognitiveEvent) -> 'Decision':
        """Convert event back to Decision."""
        content = event.content
        return Decision(
            id=content.get('entity_id', ''),
            title=content.get('title', ''),
            rationale=content.get('rationale', ''),
            affects=content.get('affects', []),
            properties=content.get('properties', {}),
            version=content.get('got_version', 1),
            created_at=content.get('created_at', event.timestamp),
            modified_at=event.timestamp,
        )

    @staticmethod
    def _event_to_sprint(event: CognitiveEvent) -> 'Sprint':
        """Convert event back to Sprint."""
        content = event.content
        return Sprint(
            id=content.get('entity_id', ''),
            title=content.get('title', ''),
            status=content.get('status', 'available'),
            epic_id=content.get('epic_id', ''),
            number=content.get('number', 0),
            goals=content.get('goals', []),
            notes=content.get('notes', []),
            properties=content.get('properties', {}),
            metadata=content.get('metadata', {}),
            version=content.get('got_version', 1),
            created_at=content.get('created_at', event.timestamp),
            modified_at=event.timestamp,
        )

    @staticmethod
    def _event_to_epic(event: CognitiveEvent) -> 'Epic':
        """Convert event back to Epic."""
        content = event.content
        return Epic(
            id=content.get('entity_id', ''),
            title=content.get('title', ''),
            status=content.get('status', 'active'),
            phase=content.get('phase', 1),
            phases=content.get('phases', []),
            properties=content.get('properties', {}),
            metadata=content.get('metadata', {}),
            version=content.get('got_version', 1),
            created_at=content.get('created_at', event.timestamp),
            modified_at=event.timestamp,
        )

    @staticmethod
    def _event_to_handoff(event: CognitiveEvent) -> 'Handoff':
        """Convert event back to Handoff."""
        content = event.content
        return Handoff(
            id=content.get('entity_id', ''),
            source_agent=content.get('source_agent', ''),
            target_agent=content.get('target_agent', ''),
            task_id=content.get('task_id', ''),
            status=content.get('status', 'initiated'),
            instructions=content.get('instructions', ''),
            context=content.get('context', {}),
            result=content.get('result', {}),
            properties=content.get('properties', {}),
            version=content.get('got_version', 1),
            created_at=content.get('created_at', event.timestamp),
            modified_at=event.timestamp,
        )

    @staticmethod
    def _event_to_document(event: CognitiveEvent) -> 'Document':
        """Convert event back to Document."""
        content = event.content
        return Document(
            id=content.get('entity_id', ''),
            path=content.get('path', ''),
            title=content.get('title', ''),
            doc_type=content.get('doc_type', 'general'),
            tags=content.get('tags', []),
            category=content.get('category', ''),
            properties=content.get('properties', {}),
            metadata=content.get('metadata', {}),
            version=content.get('got_version', 1),
            created_at=content.get('created_at', event.timestamp),
            modified_at=event.timestamp,
        )


class GotBridgeEventStore:
    """
    Event store that reads from GoT and writes to both systems.

    This is the main bridge component for Phase 1 migration.
    It maintains compatibility with existing GoT while enabling
    CEL event-based access.

    Usage:
        bridge = GotBridgeEventStore(got_path=Path(".got"))

        # Read GoT entities as events
        for event in bridge.iterate():
            print(event.content['entity_type'])

        # Write new events (goes to both systems)
        bridge.append(new_event)

    Implements: EventStore protocol
    """

    def __init__(
        self,
        got_path: Path,
        cel_path: Optional[Path] = None,
        write_to_got: bool = True,
        write_to_cel: bool = True,
    ):
        """
        Initialize the bridge.

        Args:
            got_path: Path to .got directory
            cel_path: Path for CEL storage (None = don't persist CEL)
            write_to_got: Whether to write back to GoT
            write_to_cel: Whether to write to CEL storage
        """
        self._got_path = Path(got_path)
        self._cel_path = Path(cel_path) if cel_path else None
        self._write_got = write_to_got
        self._write_cel = write_to_cel

        # Cache for loaded events
        self._events: Dict[str, CognitiveEvent] = {}
        self._head_ids: List[str] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load GoT entities into event cache."""
        if self._loaded:
            return

        entities_path = self._got_path / "entities"
        if not entities_path.exists():
            self._loaded = True
            return

        # Load all entity files
        for entity_file in entities_path.glob("*.json"):
            try:
                with open(entity_file) as f:
                    wrapper = json.load(f)

                # Extract entity data from wrapper
                data = wrapper.get('data', wrapper)

                # Determine entity type
                entity_type = data.get('entity_type', '')
                if not entity_type or entity_type not in VALID_ENTITY_TYPES:
                    continue

                # Create appropriate entity
                entity = self._load_entity(data)
                if entity is None:
                    continue

                # Convert to event
                event = GoTEventAdapter.entity_to_event(entity)
                self._events[event.id] = event
                self._head_ids.append(event.id)

            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        self._loaded = True

    def _load_entity(self, data: Dict[str, Any]) -> Optional['Entity']:
        """Load entity from dictionary."""
        if not GOT_AVAILABLE:
            return None

        entity_type = data.get('entity_type', '')

        type_map = {
            'task': Task,
            'decision': Decision,
            'sprint': Sprint,
            'epic': Epic,
            'edge': Edge,
            'handoff': Handoff,
            'document': Document,
        }

        entity_class = type_map.get(entity_type)
        if entity_class is None:
            return None

        try:
            return entity_class.from_dict(data)
        except Exception:
            return None

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """
        Append an event to the store.

        Writes to both GoT and CEL if configured.

        Args:
            event: Event to append

        Returns:
            Merkle root of the appended event
        """
        self._ensure_loaded()

        # Store in memory
        self._events[event.id] = event
        self._head_ids.append(event.id)

        # Write back to GoT if configured
        if self._write_got and GOT_AVAILABLE:
            entity = GoTEntityAdapter.event_to_entity(event)
            if entity is not None:
                self._write_to_got(entity)

        # Write to CEL if configured
        if self._write_cel and self._cel_path is not None:
            self._write_to_cel(event)

        return MerkleRoot(event.id)

    def _write_to_got(self, entity: 'Entity') -> None:
        """Write entity to GoT storage."""
        from cortical.utils.checksums import compute_checksum

        entities_path = self._got_path / "entities"
        entities_path.mkdir(parents=True, exist_ok=True)

        entity_data = entity.to_dict()
        checksum = compute_checksum(entity_data)

        wrapper = {
            '_checksum': checksum,
            '_written_at': datetime.utcnow().isoformat(),
            'data': entity_data,
        }

        file_path = entities_path / f"{entity.id}.json"
        with open(file_path, 'w') as f:
            json.dump(wrapper, f, indent=2)

    def _write_to_cel(self, event: CognitiveEvent) -> None:
        """Write event to CEL storage."""
        if self._cel_path is None:
            return

        events_path = self._cel_path / "events"
        events_path.mkdir(parents=True, exist_ok=True)

        event_file = events_path / f"{event.id}.json"
        with open(event_file, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get event by ID."""
        self._ensure_loaded()
        return self._events.get(event_id)

    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
    ) -> Iterator[CognitiveEvent]:
        """Iterate over events."""
        self._ensure_loaded()

        # Simple iteration (no causal ordering for bridge)
        started = from_event is None
        for event_id in self._head_ids:
            if not started:
                if event_id == from_event:
                    started = True
                continue

            event = self._events.get(event_id)
            if event is not None:
                yield event

            if to_event is not None and event_id == to_event:
                break

    def heads(self) -> List[MerkleRoot]:
        """Get current head events."""
        self._ensure_loaded()
        return [MerkleRoot(eid) for eid in self._head_ids[-10:]]

    def latest(self) -> Optional[MerkleRoot]:
        """Get most recent event."""
        self._ensure_loaded()
        if self._head_ids:
            return MerkleRoot(self._head_ids[-1])
        return None

    def contains(self, event_id: str) -> bool:
        """Check if event exists."""
        self._ensure_loaded()
        return event_id in self._events

    @property
    def stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        self._ensure_loaded()

        entity_types: Dict[str, int] = {}
        for event in self._events.values():
            et = event.content.get('entity_type', 'unknown')
            entity_types[et] = entity_types.get(et, 0) + 1

        return {
            'total_events': len(self._events),
            'entity_types': entity_types,
            'got_path': str(self._got_path),
            'cel_path': str(self._cel_path) if self._cel_path else None,
            'write_to_got': self._write_got,
            'write_to_cel': self._write_cel,
        }


def create_got_bridge(
    got_path: Path = Path(".got"),
    cel_path: Optional[Path] = None,
    write_mode: str = 'both',
) -> GotBridgeEventStore:
    """
    Create a GoT bridge with common configurations.

    Args:
        got_path: Path to .got directory
        cel_path: Path for CEL storage (optional)
        write_mode: 'both', 'got_only', 'cel_only', or 'read_only'

    Returns:
        Configured GotBridgeEventStore
    """
    write_got = write_mode in ('both', 'got_only')
    write_cel = write_mode in ('both', 'cel_only')

    return GotBridgeEventStore(
        got_path=got_path,
        cel_path=cel_path,
        write_to_got=write_got,
        write_to_cel=write_cel,
    )
