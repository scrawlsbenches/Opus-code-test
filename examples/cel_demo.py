#!/usr/bin/env python3
"""
Cognitive Event Lattice (CEL) - Comprehensive Demo

This demo showcases how CEL enables self-referential, self-maintaining
cognitive systems. Run it to see the architecture in action.

Usage:
    python examples/cel_demo.py

What this demonstrates:
    1. Event sourcing - Events are truth, entities are derived
    2. Temporal references - Reference "state at time T" without paradox
    3. Content-addressed storage - Same content = same ID, no conflicts
    4. Causal DAG - Events form a directed acyclic graph
    5. Materialization - Compute entities from event history
    6. Semantic indexing - Fast concept-based lookup
    7. Self-monitoring - System watches its own health
    8. Self-maintenance - Compaction for storage efficiency
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from enum import Enum, auto


# =============================================================================
# SECTION 1: IN-MEMORY EVENT INFRASTRUCTURE
# =============================================================================

class EventType(Enum):
    """Types of cognitive events."""
    OBSERVATION = auto()   # External fact observed
    INTENTION = auto()     # Task/goal to accomplish
    FULFILLMENT = auto()   # Intention completed
    INVALIDATION = auto()  # Entity invalidated
    COMPACTION = auto()    # Events compressed
    META_COGNITION = auto() # System self-observation


@dataclass(frozen=True)
class CognitiveEvent:
    """
    Immutable record of something that happened.

    Key insight: The ID is a hash of the content, making it content-addressed.
    Same content anywhere = same ID = natural deduplication and no conflicts.
    """
    timestamp: str
    event_type: EventType
    causal_parents: Tuple[str, ...]  # IDs of events this depends on
    content: Dict[str, Any]
    concepts: Tuple[str, ...]  # Semantic tags for indexing

    @property
    def id(self) -> str:
        """Content-addressed ID (Merkle root)."""
        data = {
            'timestamp': self.timestamp,
            'event_type': self.event_type.name,
            'causal_parents': list(self.causal_parents),
            'content': self.content,
            'concepts': list(self.concepts),
        }
        content_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'event_type': self.event_type.name,
            'causal_parents': list(self.causal_parents),
            'content': self.content,
            'concepts': list(self.concepts),
        }


@dataclass
class EventHorizon:
    """
    A point in the event DAG.

    Used for temporal queries: "materialize entity X as of horizon H"
    """
    event_id: str
    is_head: bool = False

    def __repr__(self):
        suffix = " (HEAD)" if self.is_head else ""
        return f"Horizon({self.event_id[:8]}...{suffix})"


@dataclass
class TemporalReference:
    """
    Reference to an entity AT A SPECIFIC POINT IN TIME.

    This is the key to self-reference without paradox:
    - "the task" is ambiguous (which version?)
    - "the task at horizon H" is concrete and stable
    """
    entity_id: str
    horizon: EventHorizon

    def __repr__(self):
        return f"Ref({self.entity_id[:12]}... @ {self.horizon})"


# =============================================================================
# SECTION 2: IN-MEMORY EVENT STORE
# =============================================================================

class InMemoryEventStore:
    """
    In-memory event store for demonstration.

    In production, this would be FileSystemEventStore or similar.
    """

    def __init__(self):
        self._events: Dict[str, CognitiveEvent] = {}
        self._order: List[str] = []  # Append order
        self._children: Dict[str, List[str]] = defaultdict(list)  # Parent -> children

    def append(self, event: CognitiveEvent) -> str:
        """Append event, return its content-addressed ID."""
        event_id = event.id

        if event_id in self._events:
            # Already exists (content-addressed deduplication)
            return event_id

        self._events[event_id] = event
        self._order.append(event_id)

        # Track parent-child relationships
        for parent_id in event.causal_parents:
            self._children[parent_id].append(event_id)

        return event_id

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        return self._events.get(event_id)

    def iterate(self) -> Iterator[CognitiveEvent]:
        """Iterate in append order."""
        for eid in self._order:
            yield self._events[eid]

    def iterate_until(self, horizon: EventHorizon) -> Iterator[CognitiveEvent]:
        """Iterate events up to (and including) the horizon."""
        for eid in self._order:
            yield self._events[eid]
            if eid == horizon.event_id:
                break

    def heads(self) -> List[str]:
        """Get events with no children (current heads)."""
        all_ids = set(self._events.keys())
        has_children = set(self._children.keys())
        return [eid for eid in self._order if eid not in has_children or eid == self._order[-1]]

    def latest(self) -> Optional[str]:
        """Get most recent event ID."""
        return self._order[-1] if self._order else None

    @property
    def count(self) -> int:
        return len(self._events)


# =============================================================================
# SECTION 3: ENTITY MATERIALIZATION
# =============================================================================

@dataclass
class MaterializedTask:
    """Task entity materialized from events."""
    id: str
    title: str
    status: str  # pending, in_progress, completed
    description: str = ""
    priority: str = "medium"
    created_at: str = ""
    completed_at: str = ""
    version: int = 0

    def __repr__(self):
        return f"Task({self.id[:12]}... '{self.title}' [{self.status}])"


@dataclass
class MaterializedDecision:
    """Decision entity materialized from events."""
    id: str
    title: str
    rationale: str
    affects: List[str] = field(default_factory=list)
    created_at: str = ""

    def __repr__(self):
        return f"Decision({self.id[:12]}... '{self.title}')"


class EntityMaterializer:
    """
    Materializes entities from event history.

    Key insight: Entities don't exist in storage. They are COMPUTED
    from events on demand. This enables time-travel queries.
    """

    def __init__(self, store: InMemoryEventStore):
        self._store = store
        self._cache: Dict[Tuple[str, str], Any] = {}  # (entity_id, horizon_id) -> entity

    def materialize_task(
        self,
        entity_id: str,
        at: Optional[EventHorizon] = None
    ) -> Optional[MaterializedTask]:
        """
        Materialize a task as of a specific horizon.

        If at=None, uses current head (latest state).
        """
        # Build task state by folding events
        state = None

        events = (
            self._store.iterate_until(at) if at
            else self._store.iterate()
        )

        for event in events:
            if event.content.get('entity_id') != entity_id:
                continue
            if event.content.get('entity_type') != 'task':
                continue

            if state is None:
                # First event creates the task
                state = MaterializedTask(
                    id=entity_id,
                    title=event.content.get('title', ''),
                    status=event.content.get('status', 'pending'),
                    description=event.content.get('description', ''),
                    priority=event.content.get('priority', 'medium'),
                    created_at=event.timestamp,
                    version=1,
                )
            else:
                # Subsequent events update it
                if 'title' in event.content:
                    state.title = event.content['title']
                if 'status' in event.content:
                    state.status = event.content['status']
                if 'description' in event.content:
                    state.description = event.content['description']
                if state.status == 'completed' and not state.completed_at:
                    state.completed_at = event.timestamp
                state.version += 1

        return state

    def materialize_decision(
        self,
        entity_id: str,
        at: Optional[EventHorizon] = None
    ) -> Optional[MaterializedDecision]:
        """Materialize a decision as of a specific horizon."""
        events = (
            self._store.iterate_until(at) if at
            else self._store.iterate()
        )

        for event in events:
            if event.content.get('entity_id') != entity_id:
                continue
            if event.content.get('entity_type') != 'decision':
                continue

            return MaterializedDecision(
                id=entity_id,
                title=event.content.get('title', ''),
                rationale=event.content.get('rationale', ''),
                affects=event.content.get('affects', []),
                created_at=event.timestamp,
            )

        return None


# =============================================================================
# SECTION 4: SEMANTIC INDEXING
# =============================================================================

class BloomFilter:
    """Simple bloom filter for probabilistic set membership."""

    def __init__(self, size: int = 1000, hash_count: int = 3):
        self._bits = [False] * size
        self._size = size
        self._hash_count = hash_count

    def _hashes(self, item: str) -> List[int]:
        """Generate multiple hash positions."""
        positions = []
        for i in range(self._hash_count):
            h = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            positions.append(int(h, 16) % self._size)
        return positions

    def add(self, item: str) -> None:
        for pos in self._hashes(item):
            self._bits[pos] = True

    def __contains__(self, item: str) -> bool:
        return all(self._bits[pos] for pos in self._hashes(item))


class SemanticIndex:
    """
    Index for fast concept-based lookup.

    Uses bloom filter for O(1) "probably exists" checks,
    inverted index for actual retrieval.
    """

    def __init__(self):
        self._bloom = BloomFilter()
        self._concept_to_events: Dict[str, Set[str]] = defaultdict(set)
        self._event_concepts: Dict[str, Set[str]] = defaultdict(set)

    def index_event(self, event: CognitiveEvent) -> None:
        """Index an event's concepts."""
        event_id = event.id
        for concept in event.concepts:
            self._bloom.add(concept)
            self._concept_to_events[concept].add(event_id)
            self._event_concepts[event_id].add(concept)

    def probably_has(self, concept: str) -> bool:
        """Fast probabilistic check (may have false positives)."""
        return concept in self._bloom

    def find_events(self, concept: str) -> Set[str]:
        """Find all events with a concept."""
        return self._concept_to_events.get(concept, set())

    def search(self, query: str) -> List[str]:
        """Search for events matching query terms."""
        terms = query.lower().split()
        if not terms:
            return []

        # Intersection of all term matches
        result = None
        for term in terms:
            matches = self._concept_to_events.get(term, set())
            if result is None:
                result = matches.copy()
            else:
                result &= matches

        return list(result) if result else []


# =============================================================================
# SECTION 5: HEALTH MONITORING
# =============================================================================

class HealthStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()


@dataclass
class HealthReport:
    status: HealthStatus
    event_count: int
    concept_count: int
    dag_depth: int
    issues: List[str]
    recommendations: List[str]


class HealthMonitor:
    """
    Self-monitoring for the cognitive lattice.

    The system watches itself and reports on its health.
    Health checks are themselves events (meta-cognition).
    """

    def __init__(self, store: InMemoryEventStore, index: SemanticIndex):
        self._store = store
        self._index = index

    def check(self) -> HealthReport:
        """Perform comprehensive health check."""
        issues = []
        recommendations = []

        event_count = self._store.count
        concept_count = len(self._index._concept_to_events)

        # Check event count
        if event_count > 1000:
            issues.append(f"High event count: {event_count}")
            recommendations.append("Consider running compaction")

        # Check for orphan events (no parents, not genesis)
        orphans = 0
        for event in self._store.iterate():
            if not event.causal_parents and event_count > 1:
                # First event is OK to have no parents
                if event.id != self._store._order[0]:
                    orphans += 1

        if orphans > 0:
            issues.append(f"Found {orphans} orphan events")
            recommendations.append("Investigate DAG consistency")

        # Determine overall status
        if issues:
            status = HealthStatus.DEGRADED if len(issues) < 3 else HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.HEALTHY

        # Calculate DAG depth (simplified)
        dag_depth = self._calculate_depth()

        return HealthReport(
            status=status,
            event_count=event_count,
            concept_count=concept_count,
            dag_depth=dag_depth,
            issues=issues,
            recommendations=recommendations,
        )

    def _calculate_depth(self) -> int:
        """Calculate maximum DAG depth."""
        if not self._store._order:
            return 0

        depths = {}
        max_depth = 0

        for event in self._store.iterate():
            if not event.causal_parents:
                depths[event.id] = 1
            else:
                parent_depths = [depths.get(p, 0) for p in event.causal_parents]
                depths[event.id] = max(parent_depths) + 1
            max_depth = max(max_depth, depths[event.id])

        return max_depth


# =============================================================================
# SECTION 6: THE COGNITIVE LATTICE (UNIFIED INTERFACE)
# =============================================================================

class CognitiveLattice:
    """
    The unified cognitive substrate.

    Combines all components into a coherent system that can:
    - Store immutable events
    - Materialize entities at any point in time
    - Search by concepts
    - Monitor its own health
    """

    def __init__(self):
        self.store = InMemoryEventStore()
        self.materializer = EntityMaterializer(self.store)
        self.index = SemanticIndex()
        self.health = HealthMonitor(self.store, self.index)
        self._genesis_created = False

    @property
    def current_horizon(self) -> EventHorizon:
        """Get the current head of the event DAG."""
        latest = self.store.latest()
        if latest:
            return EventHorizon(event_id=latest, is_head=True)
        return EventHorizon(event_id="GENESIS", is_head=True)

    def observe(
        self,
        content: Dict[str, Any],
        concepts: List[str] = None,
        parents: List[str] = None,
    ) -> str:
        """Record an observation event."""
        event = CognitiveEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=tuple(parents or [self.store.latest()] if self.store.latest() else []),
            content=content,
            concepts=tuple(concepts or []),
        )
        event_id = self.store.append(event)
        self.index.index_event(event)
        return event_id

    def intend(
        self,
        task_id: str,
        title: str,
        description: str = "",
        priority: str = "medium",
        references_at: EventHorizon = None,
        concepts: List[str] = None,
    ) -> Tuple[str, TemporalReference]:
        """
        Create an intention (task) with optional temporal reference.

        The temporal reference captures "the system state when this was created",
        enabling the task to reason about past state even as the system evolves.
        """
        # Capture temporal reference if requested
        ref_horizon = references_at or self.current_horizon
        temporal_ref = TemporalReference(
            entity_id=task_id,
            horizon=ref_horizon,
        )

        content = {
            'entity_id': task_id,
            'entity_type': 'task',
            'title': title,
            'description': description,
            'priority': priority,
            'status': 'pending',
            'temporal_reference': {
                'entity_id': temporal_ref.entity_id,
                'horizon_id': temporal_ref.horizon.event_id,
            }
        }

        event = CognitiveEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.INTENTION,
            causal_parents=tuple([self.store.latest()] if self.store.latest() else []),
            content=content,
            concepts=tuple(concepts or ['task', priority]),
        )

        event_id = self.store.append(event)
        self.index.index_event(event)

        return event_id, temporal_ref

    def fulfill(self, task_id: str, notes: str = "") -> str:
        """Mark an intention as fulfilled (task completed)."""
        content = {
            'entity_id': task_id,
            'entity_type': 'task',
            'status': 'completed',
            'completion_notes': notes,
        }

        event = CognitiveEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.FULFILLMENT,
            causal_parents=tuple([self.store.latest()] if self.store.latest() else []),
            content=content,
            concepts=tuple(['task', 'completed']),
        )

        event_id = self.store.append(event)
        self.index.index_event(event)
        return event_id

    def decide(
        self,
        decision_id: str,
        title: str,
        rationale: str,
        affects: List[str] = None,
        concepts: List[str] = None,
    ) -> str:
        """Record a decision."""
        content = {
            'entity_id': decision_id,
            'entity_type': 'decision',
            'title': title,
            'rationale': rationale,
            'affects': affects or [],
        }

        event = CognitiveEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.OBSERVATION,
            causal_parents=tuple([self.store.latest()] if self.store.latest() else []),
            content=content,
            concepts=tuple(concepts or ['decision']),
        )

        event_id = self.store.append(event)
        self.index.index_event(event)
        return event_id

    def meta_observe(self, observation: str, conclusions: List[str] = None) -> str:
        """
        Record a meta-cognition event (system observing itself).

        This is how the system reasons about its own state.
        """
        health_report = self.health.check()

        content = {
            'observation_type': 'self_check',
            'observation': observation,
            'conclusions': conclusions or [],
            'health_status': health_report.status.name,
            'event_count': health_report.event_count,
        }

        event = CognitiveEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.META_COGNITION,
            causal_parents=tuple([self.store.latest()] if self.store.latest() else []),
            content=content,
            concepts=tuple(['meta', 'self-observation']),
        )

        event_id = self.store.append(event)
        self.index.index_event(event)
        return event_id

    def get_task(self, task_id: str, at: EventHorizon = None) -> Optional[MaterializedTask]:
        """Get a task, optionally at a specific point in time."""
        return self.materializer.materialize_task(task_id, at)

    def get_decision(self, decision_id: str, at: EventHorizon = None) -> Optional[MaterializedDecision]:
        """Get a decision, optionally at a specific point in time."""
        return self.materializer.materialize_decision(decision_id, at)

    def search(self, query: str) -> List[CognitiveEvent]:
        """Search events by concept."""
        event_ids = self.index.search(query)
        return [self.store.get(eid) for eid in event_ids if self.store.get(eid)]


# =============================================================================
# SECTION 7: DEMONSTRATION
# =============================================================================

def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_event(event: CognitiveEvent, prefix: str = ""):
    """Pretty print an event."""
    print(f"{prefix}Event: {event.id[:12]}...")
    print(f"{prefix}  Type: {event.event_type.name}")
    print(f"{prefix}  Time: {event.timestamp}")
    print(f"{prefix}  Parents: {[p[:8] + '...' for p in event.causal_parents]}")
    print(f"{prefix}  Concepts: {list(event.concepts)}")
    if 'title' in event.content:
        print(f"{prefix}  Title: {event.content['title']}")


def run_demo():
    """Run the comprehensive demonstration."""

    print("\n" + "🧠" * 35)
    print("    COGNITIVE EVENT LATTICE - COMPREHENSIVE DEMO")
    print("🧠" * 35)

    # Create the lattice
    lattice = CognitiveLattice()

    # =========================================================================
    print_header("1. EVENT SOURCING: Creating Events")
    # =========================================================================

    print("""
    Key Insight: We store EVENTS (what happened), not entities.
    Entities are computed from events on demand.
    """)

    # Create a decision
    decision_id = lattice.decide(
        decision_id="D-001",
        title="Use event sourcing for GoT storage",
        rationale="Enables time travel queries and avoids merge conflicts",
        affects=["storage", "query", "persistence"],
        concepts=['decision', 'architecture', 'storage'],
    )
    print(f"Created decision: {decision_id[:12]}...")

    # Create a task that references the current system state
    print("\nCapturing horizon BEFORE creating task...")
    horizon_before_task = lattice.current_horizon
    print(f"  Horizon: {horizon_before_task}")

    task_id, task_ref = lattice.intend(
        task_id="T-001",
        title="Implement CEL event store",
        description="Create the core event storage infrastructure",
        priority="high",
        references_at=horizon_before_task,
        concepts=['task', 'implementation', 'storage'],
    )
    print(f"\nCreated task: {task_id[:12]}...")
    print(f"  Temporal reference: {task_ref}")

    # =========================================================================
    print_header("2. TEMPORAL REFERENCES: The Key to Self-Reference")
    # =========================================================================

    print("""
    Problem: A task that references "the system" is ambiguous.
             Which version of the system? Past? Present? Future?

    Solution: Reference "the system AT event E" - a specific, stable point.
    """)

    # The task was created with a reference to the system state
    # BEFORE it existed. This is not a paradox because we reference
    # a specific point in time.

    print(f"\nTask T-001 references system state at: {task_ref.horizon}")
    print("This horizon was captured BEFORE the task was created.")
    print("Even as the system evolves, this reference remains stable.")

    # Create more events to show the system evolving
    task2_id, _ = lattice.intend(
        task_id="T-002",
        title="Add semantic indexing",
        priority="medium",
        concepts=['task', 'indexing', 'semantic'],
    )

    task3_id, _ = lattice.intend(
        task_id="T-003",
        title="Implement health monitoring",
        priority="medium",
        concepts=['task', 'health', 'monitoring'],
    )

    print(f"\nCreated additional tasks: T-002, T-003")
    print(f"Current horizon: {lattice.current_horizon}")
    print(f"Task T-001's reference still points to: {task_ref.horizon}")

    # =========================================================================
    print_header("3. CONTENT-ADDRESSED STORAGE: No Merge Conflicts")
    # =========================================================================

    print("""
    Key Insight: Event IDs are SHA256 hashes of their content.
    Same content ANYWHERE = Same ID = Natural deduplication.

    This means:
    - No conflicts when merging branches
    - Events can be created independently and still interleave correctly
    - Integrity verification is built-in
    """)

    # Demonstrate that same content = same ID
    event1 = CognitiveEvent(
        timestamp="2024-01-15T10:00:00",
        event_type=EventType.OBSERVATION,
        causal_parents=(),
        content={"message": "Hello, world!"},
        concepts=("greeting",),
    )

    event2 = CognitiveEvent(
        timestamp="2024-01-15T10:00:00",
        event_type=EventType.OBSERVATION,
        causal_parents=(),
        content={"message": "Hello, world!"},
        concepts=("greeting",),
    )

    print(f"\nEvent 1 ID: {event1.id}")
    print(f"Event 2 ID: {event2.id}")
    print(f"Same content = Same ID: {event1.id == event2.id}")

    # =========================================================================
    print_header("4. MATERIALIZATION: Entities from Events")
    # =========================================================================

    print("""
    Entities don't exist in storage - they are COMPUTED from events.
    This enables "time travel" queries.
    """)

    # Get current state of task
    task = lattice.get_task("T-001")
    print(f"\nTask T-001 current state:")
    print(f"  {task}")
    print(f"  Version: {task.version}")
    print(f"  Status: {task.status}")

    # Update the task
    lattice.store.append(CognitiveEvent(
        timestamp=datetime.now().isoformat(),
        event_type=EventType.OBSERVATION,
        causal_parents=(lattice.store.latest(),),
        content={
            'entity_id': 'T-001',
            'entity_type': 'task',
            'status': 'in_progress',
            'description': 'Working on event store implementation',
        },
        concepts=('task', 'update'),
    ))

    # Get updated state
    task_updated = lattice.get_task("T-001")
    print(f"\nAfter update:")
    print(f"  {task_updated}")
    print(f"  Version: {task_updated.version}")
    print(f"  Status: {task_updated.status}")

    # =========================================================================
    print_header("5. TIME TRAVEL: Query Past States")
    # =========================================================================

    print("""
    The killer feature: Query what an entity looked like at ANY point in time.
    """)

    # Remember the current horizon
    horizon_in_progress = lattice.current_horizon

    # Complete the task
    lattice.fulfill("T-001", notes="Event store implemented successfully")

    # Get current state (completed)
    task_now = lattice.get_task("T-001")
    print(f"\nTask T-001 NOW (completed):")
    print(f"  Status: {task_now.status}")
    print(f"  Completed at: {task_now.completed_at}")

    # Get state at earlier horizon (in progress)
    task_then = lattice.get_task("T-001", at=horizon_in_progress)
    print(f"\nTask T-001 at horizon {horizon_in_progress.event_id[:8]}... (in progress):")
    print(f"  Status: {task_then.status}")
    print(f"  Completed at: {task_then.completed_at}")

    print("\n✨ Same entity, different points in time, different states!")

    # =========================================================================
    print_header("6. SEMANTIC INDEXING: Fast Concept Lookup")
    # =========================================================================

    print("""
    Events are tagged with concepts for fast retrieval.
    Bloom filters provide O(1) probabilistic "exists?" checks.
    """)

    # Search by concept
    print("\nSearching for 'task' events:")
    task_events = lattice.search("task")
    print(f"  Found {len(task_events)} events")

    print("\nSearching for 'storage' events:")
    storage_events = lattice.search("storage")
    print(f"  Found {len(storage_events)} events")
    for event in storage_events[:3]:
        print(f"    - {event.content.get('title', event.content.get('message', 'N/A'))}")

    # Bloom filter check
    print("\nBloom filter probabilistic checks:")
    print(f"  'task' probably exists: {lattice.index.probably_has('task')}")
    print(f"  'quantum' probably exists: {lattice.index.probably_has('quantum')}")

    # =========================================================================
    print_header("7. CAUSAL DAG: Event Relationships")
    # =========================================================================

    print("""
    Events form a Directed Acyclic Graph (DAG) through causal_parents.
    This captures "what caused what" - essential for reasoning.
    """)

    print("\nEvent DAG structure:")
    print("─" * 50)

    for i, event in enumerate(lattice.store.iterate()):
        indent = "  " * min(i, 4)
        parent_info = ""
        if event.causal_parents:
            parent_info = f" ← {event.causal_parents[0][:8]}..."

        title = event.content.get('title', event.content.get('observation', event.event_type.name))
        if len(title) > 30:
            title = title[:27] + "..."

        print(f"{indent}[{event.id[:8]}] {title}{parent_info}")

    print("─" * 50)

    # =========================================================================
    print_header("8. SELF-MONITORING: Health Checks")
    # =========================================================================

    print("""
    The system monitors its own health. Health checks are themselves
    events, creating recursive self-awareness.
    """)

    report = lattice.health.check()
    print(f"\nHealth Status: {report.status.name}")
    print(f"Event Count: {report.event_count}")
    print(f"Concept Count: {report.concept_count}")
    print(f"DAG Depth: {report.dag_depth}")

    if report.issues:
        print(f"\nIssues:")
        for issue in report.issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ No issues detected")

    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  💡 {rec}")

    # Record a meta-cognition event
    meta_id = lattice.meta_observe(
        observation="Performed self-health check",
        conclusions=[
            f"System is {report.status.name}",
            f"DAG depth of {report.dag_depth} is manageable",
        ]
    )
    print(f"\nRecorded meta-cognition event: {meta_id[:12]}...")

    # =========================================================================
    print_header("9. BENEFITS FOR COGNITIVE SOFTWARE DEVELOPMENT")
    # =========================================================================

    print("""
    Why CEL matters for intelligent, evolving systems:

    1. SELF-REFERENCE WITHOUT PARADOX
       Tasks can reference "the system when I was created" without
       race conditions or inconsistent views.

    2. COMPLETE AUDIT TRAIL
       Every change is an event. You can always ask:
       "What did the system know when it made decision X?"

    3. NO MERGE CONFLICTS
       Content-addressed storage means same content = same ID.
       Parallel agents can work independently.

    4. GRADUAL EVOLUTION
       Schema changes don't require migration - just add new event
       types and update materialization logic.

    5. SELF-MAINTENANCE
       The system can observe its own health and request compaction
       or other maintenance operations.

    6. DEPENDENCY INJECTION
       All components are protocol-based. Swap implementations
       without changing consumers.

    7. TIME TRAVEL DEBUGGING
       Reproduce any past state by materializing at that horizon.
       Essential for debugging complex agent interactions.
    """)

    # =========================================================================
    print_header("10. SUMMARY: THE DOUBLE HELIX")
    # =========================================================================

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │     WISDOM STRAND              SANITY STRAND                │
    │     (What we know)             (Keeping us coherent)        │
    │                                                             │
    │     ◉ Events                   ◉ Health Monitoring          │
    │     ◉ Merkle DAG               ◉ Schema Migration           │
    │     ◉ Materialization          ◉ Compaction                 │
    │     ◉ Semantic Index           ◉ Self-Repair                │
    │                                                             │
    │         ╲                    ╱                              │
    │          ╲                  ╱                               │
    │           ╲                ╱                                │
    │            ╲══════════════╱                                 │
    │             COGNITIVE                                       │
    │              LATTICE                                        │
    │                                                             │
    │     The two strands are intertwined:                        │
    │     - Wisdom without sanity leads to corruption             │
    │     - Sanity without wisdom leads to empty process          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print(f"\nFinal Statistics:")
    print(f"  Total Events: {lattice.store.count}")
    print(f"  Indexed Concepts: {len(lattice.index._concept_to_events)}")
    print(f"  Current Horizon: {lattice.current_horizon}")

    print("\n" + "🧠" * 35)
    print("    DEMO COMPLETE")
    print("🧠" * 35 + "\n")


if __name__ == "__main__":
    run_demo()
