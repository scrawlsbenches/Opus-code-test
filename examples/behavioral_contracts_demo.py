#!/usr/bin/env python3
"""
Behavioral Contracts - End-to-End Demonstration with CEL Integration

This example demonstrates behavioral contracts in the context CEL was
designed for: cognitive operations, task management, and knowledge transfer.

The contracts express INTENT about:
- Task state machine transitions (pending -> in_progress -> completed)
- Knowledge transfer lifecycle (draft -> finalized)
- Session integrity (active before work, clean handoff)
- Cognitive invariants (WAL entries >= committed entities)

NEW: Full CEL integration showing:
- In-memory EventStore for contract events
- Temporal queries ("What was contract state at event X?")
- Compaction demonstration (semantic compression of contract history)
- Recovery flow (handling violations gracefully)

Run with:
    python examples/behavioral_contracts_demo.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.contracts import (
    requires,
    ensures,
    invariant,
    ContractViolation,
    ContractRegistry,
    ContractEventEmitter,
    ContractMaterializer,
    ContractState,
)

# CEL imports for integration
from cortical.cel.core.events import (
    CognitiveEvent,
    EventType,
    Observation,
    MetaCognition,
    Compaction,
)
from cortical.cel.core.references import MerkleRoot, EventHorizon


# =============================================================================
# STEP 0: In-Memory EventStore (implements CEL EventStore protocol)
# =============================================================================

class MemoryEventStore:
    """
    In-memory implementation of CEL EventStore protocol.

    This is a lightweight store for demonstrations and testing.
    In production, use StreamingEventStore or SQLiteEventStore.

    Features:
    - Append-only semantics (immutable events)
    - Content-addressed IDs (via CognitiveEvent.id)
    - Causal ordering preserved
    - Temporal query support (iterate with from_event/to_event)
    """

    def __init__(self):
        self._events: OrderedDict[str, CognitiveEvent] = OrderedDict()
        self._children: Dict[str, Set[str]] = {}  # parent_id -> child_ids

    def append(self, event: CognitiveEvent) -> MerkleRoot:
        """Append event to store. Returns Merkle root (content hash)."""
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
        """Retrieve event by ID."""
        return self._events.get(event_id)

    def iterate(
        self,
        from_event: Optional[str] = None,
        to_event: Optional[str] = None,
        event_types: Optional[Sequence[EventType]] = None,
    ) -> Iterator[CognitiveEvent]:
        """Iterate events in causal order."""
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
        """Get events with no children (branch heads)."""
        all_ids = set(self._events.keys())
        children = set()
        for child_set in self._children.values():
            children.update(child_set)

        head_ids = all_ids - children
        return [MerkleRoot(eid) for eid in head_ids]

    def latest(self) -> Optional[MerkleRoot]:
        """Get most recent event."""
        if not self._events:
            return None
        # OrderedDict preserves insertion order
        last_id = list(self._events.keys())[-1]
        return MerkleRoot(last_id)

    def ancestors(self, event_id: str, depth: int = -1) -> Iterator[CognitiveEvent]:
        """Iterate ancestors in reverse causal order."""
        visited = set()
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
        """Iterate descendants in causal order."""
        visited = set()
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

    @property
    def count(self) -> int:
        """Total number of events."""
        return len(self._events)

    def events_in_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Iterator[CognitiveEvent]:
        """Iterate events within a time range."""
        for event in self._events.values():
            event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            if start_time and event_time < start_time:
                continue
            if end_time and event_time > end_time:
                continue

            yield event


# =============================================================================
# STEP 1: Create Registry, Emitter, and EventStore
# =============================================================================

print("=" * 70)
print("BEHAVIORAL CONTRACTS - CEL INTEGRATED DEMO")
print("=" * 70)
print()

# Create in-memory CEL event store
event_store = MemoryEventStore()
print("[1] Created in-memory CEL EventStore")

# Create event emitter WITH CEL store
emitter = ContractEventEmitter(
    event_store=event_store,
    emit_all_checks=True,
)
print("    - Connected ContractEventEmitter to EventStore")

# Create registry with emitter
registry = ContractRegistry(emitter=emitter)
print("    - Created ContractRegistry")

# Create materializer with store
materializer = ContractMaterializer(
    event_store=event_store,
    emitter=emitter,
)
print("    - Created ContractMaterializer with EventStore")
print()


# =============================================================================
# STEP 2: Define Cognitive Classes with Contracts
# =============================================================================

class TaskStatus(Enum):
    """Valid task states in the state machine."""
    PENDING = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class KTStatus(Enum):
    """Knowledge Transfer document states."""
    DRAFT = auto()
    FINALIZED = auto()


class Task:
    """
    A task in the Graph of Thought system.

    Contracts enforce the state machine:
    - Cannot complete a task that isn't in progress
    - Cannot start a task that's already completed
    - Task ID must be valid format
    """

    VALID_TRANSITIONS = {
        TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
        TaskStatus.IN_PROGRESS: {TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
        TaskStatus.BLOCKED: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
        TaskStatus.COMPLETED: set(),  # Terminal state
        TaskStatus.CANCELLED: set(),  # Terminal state
    }

    def __init__(self, task_id: str, title: str):
        if not task_id.startswith("T-"):
            raise ValueError(f"Invalid task ID format: {task_id}")
        self._id = task_id
        self._title = title
        self._status = TaskStatus.PENDING
        self._history: List[tuple] = [(datetime.now(), TaskStatus.PENDING)]

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def id(self) -> str:
        return self._id

    def _can_transition_to(self, new_status: TaskStatus) -> bool:
        """Check if transition is valid per state machine."""
        return new_status in self.VALID_TRANSITIONS.get(self._status, set())

    @registry.track
    @requires(lambda self: self._status == TaskStatus.PENDING,
              "Can only start a PENDING task")
    @ensures(lambda self, result: self._status == TaskStatus.IN_PROGRESS,
             "Task must be IN_PROGRESS after start")
    def start(self) -> 'Task':
        """Begin work on this task."""
        self._status = TaskStatus.IN_PROGRESS
        self._history.append((datetime.now(), TaskStatus.IN_PROGRESS))
        return self

    @registry.track
    @requires(lambda self: self._status == TaskStatus.IN_PROGRESS,
              "Can only complete an IN_PROGRESS task")
    @requires(lambda self, retrospective: len(retrospective) > 0,
              "Retrospective cannot be empty")
    @ensures(lambda self, result: self._status == TaskStatus.COMPLETED,
             "Task must be COMPLETED after completion")
    def complete(self, retrospective: str) -> 'Task':
        """Mark task as completed with retrospective."""
        self._status = TaskStatus.COMPLETED
        self._retrospective = retrospective
        self._history.append((datetime.now(), TaskStatus.COMPLETED))
        return self

    @registry.track
    @requires(lambda self: self._status == TaskStatus.IN_PROGRESS,
              "Can only block an IN_PROGRESS task")
    @requires(lambda self, reason: len(reason) > 0,
              "Block reason cannot be empty")
    def block(self, reason: str) -> 'Task':
        """Mark task as blocked."""
        self._status = TaskStatus.BLOCKED
        self._block_reason = reason
        self._history.append((datetime.now(), TaskStatus.BLOCKED))
        return self

    @registry.track
    @requires(lambda self: self._status not in {TaskStatus.COMPLETED, TaskStatus.CANCELLED},
              "Cannot cancel a terminal task")
    def cancel(self) -> 'Task':
        """Cancel the task."""
        self._status = TaskStatus.CANCELLED
        self._history.append((datetime.now(), TaskStatus.CANCELLED))
        return self


class KnowledgeTransfer:
    """
    Knowledge Transfer document for preserving session learnings.

    Contracts enforce:
    - Cannot finalize an empty document
    - Cannot append to a finalized document
    - Must have summary before finalizing
    """

    def __init__(self, kt_id: str, title: str):
        if not kt_id.startswith("KT-"):
            raise ValueError(f"Invalid KT ID format: {kt_id}")
        self._id = kt_id
        self._title = title
        self._status = KTStatus.DRAFT
        self._sections: Dict[str, str] = {}
        self._summary: Optional[str] = None

    @property
    def status(self) -> KTStatus:
        return self._status

    @property
    def is_empty(self) -> bool:
        return len(self._sections) == 0 and self._summary is None

    @registry.track
    @requires(lambda self: self._status == KTStatus.DRAFT,
              "Cannot append to a finalized document")
    @requires(lambda self, section_name, content: len(section_name) > 0,
              "Section name cannot be empty")
    @requires(lambda self, section_name, content: len(content) > 0,
              "Section content cannot be empty")
    @ensures(lambda self, result: not self.is_empty,
             "Document must not be empty after append")
    def append(self, section_name: str, content: str) -> 'KnowledgeTransfer':
        """Append a section to the document."""
        self._sections[section_name] = content
        return self

    @registry.track
    @requires(lambda self: self._status == KTStatus.DRAFT,
              "Cannot set summary on a finalized document")
    @requires(lambda self, summary: len(summary) >= 10,
              "Summary must be at least 10 characters")
    def set_summary(self, summary: str) -> 'KnowledgeTransfer':
        """Set the document summary."""
        self._summary = summary
        return self

    @registry.track
    @requires(lambda self: self._status == KTStatus.DRAFT,
              "Document is already finalized")
    @requires(lambda self: not self.is_empty,
              "Cannot finalize an empty document")
    @requires(lambda self: self._summary is not None,
              "Cannot finalize without a summary")
    @ensures(lambda self, result: self._status == KTStatus.FINALIZED,
             "Document must be FINALIZED after finalize")
    def finalize(self) -> 'KnowledgeTransfer':
        """Finalize the document (no more changes allowed)."""
        self._status = KTStatus.FINALIZED
        return self


class CognitiveSession:
    """
    An AI agent session with cognitive state management.

    Contracts enforce:
    - Session must be active before performing work
    - WAL entries must be >= committed operations (durability invariant)
    - Clean handoff requires saving state first
    """

    def __init__(self, session_id: str):
        self._id = session_id
        self._active = False
        self._tasks: List[Task] = []
        self._wal_entries = 0  # Write-ahead log entries
        self._committed = 0    # Committed operations
        self._state_saved = False

    @property
    def is_active(self) -> bool:
        return self._active

    @registry.track
    @requires(lambda self: not self._active,
              "Session is already active")
    @ensures(lambda self, result: self._active,
             "Session must be active after begin")
    def begin(self) -> 'CognitiveSession':
        """Begin the cognitive session."""
        self._active = True
        self._wal_entries += 1
        return self

    @registry.track
    @requires(lambda self: self._active,
              "Session must be active to create tasks")
    @invariant(lambda self: self._wal_entries >= self._committed,
               "WAL entries must be >= committed operations")
    def create_task(self, task_id: str, title: str) -> Task:
        """Create a new task in this session."""
        self._wal_entries += 1  # WAL first
        task = Task(task_id, title)
        self._tasks.append(task)
        self._committed += 1
        return task

    @registry.track
    @requires(lambda self: self._active,
              "Session must be active to save state")
    @ensures(lambda self, result: self._state_saved,
             "State must be marked as saved after save")
    @invariant(lambda self: self._wal_entries >= self._committed,
               "WAL entries must be >= committed operations")
    def save_state(self) -> 'CognitiveSession':
        """Save current session state."""
        self._wal_entries += 1
        self._state_saved = True
        self._committed += 1
        return self

    @registry.track
    @requires(lambda self: self._active,
              "Session must be active to handoff")
    @requires(lambda self: self._state_saved,
              "Must save state before handoff")
    @ensures(lambda self, result: not self._active,
             "Session must not be active after handoff")
    def handoff(self, target_agent: str) -> dict:
        """Handoff session to another agent."""
        self._active = False
        return {
            'session_id': self._id,
            'target': target_agent,
            'tasks': len(self._tasks),
            'wal_entries': self._wal_entries,
            'committed': self._committed,
        }

    @registry.track
    @requires(lambda self: self._active,
              "Session must be active to end")
    @invariant(lambda self: self._wal_entries >= self._committed,
               "WAL entries must be >= committed operations")
    def end(self) -> 'CognitiveSession':
        """End the session."""
        self._active = False
        return self


print("[2] Defined cognitive classes with contracts:")
print()
print("    Task: start(), complete(), block(), cancel()")
print("    KnowledgeTransfer: append(), set_summary(), finalize()")
print("    CognitiveSession: begin(), create_task(), save_state(), handoff()")
print()


# =============================================================================
# STEP 3: Execute Operations - Capture Horizon Markers
# =============================================================================

print("[3] Executing cognitive operations with horizon markers...")
print()

# Capture initial horizon
initial_horizon = event_store.latest()
print(f"    Initial horizon: {initial_horizon}")

# Create a session
session = CognitiveSession("S-001")
session.begin()
print("    [OK] Session S-001 began")

# Capture horizon after session start
post_begin_horizon = event_store.latest()

# Create and work on tasks
task1 = session.create_task("T-001", "Implement behavioral contracts")
task1.start()
task1.complete("Implemented @requires, @ensures, @invariant with CEL integration")
print("    [OK] Task T-001: PENDING -> IN_PROGRESS -> COMPLETED")

# Capture horizon after task completion
post_task_horizon = event_store.latest()
print(f"    Post-task horizon: {post_task_horizon.value[:12] if post_task_horizon else 'None'}...")

# Create a knowledge transfer
kt = KnowledgeTransfer("KT-001", "Session Learnings")
kt.append("What Worked", "Contract decorators capture intent effectively")
kt.append("Challenges", "Decorator ordering matters - track must be outermost")
kt.set_summary("Behavioral contracts enable executable documentation")
kt.finalize()
print("    [OK] Knowledge transfer KT-001 created and finalized")

# Capture horizon after KT
post_kt_horizon = event_store.latest()

print()
print("    Now triggering contract violations...")
print()

# Violation 1: Complete a task that isn't in progress
task2 = session.create_task("T-002", "Another task")
try:
    task2.complete("This should fail")
except ContractViolation as e:
    print(f"    [VIOLATION] {e.contract_type}: {e.description}")

# Capture violation horizon
violation1_horizon = event_store.latest()

# Violation 2: Append to finalized KT
try:
    kt.append("New Section", "This should fail")
except ContractViolation as e:
    print(f"    [VIOLATION] {e.contract_type}: {e.description}")

# Violation 3: Finalize empty KT
kt2 = KnowledgeTransfer("KT-002", "Empty KT")
try:
    kt2.finalize()
except ContractViolation as e:
    print(f"    [VIOLATION] {e.contract_type}: {e.description}")

# Violation 4: Handoff without saving state
try:
    session.handoff("agent-2")
except ContractViolation as e:
    print(f"    [VIOLATION] {e.contract_type}: {e.description}")

print()

# Proper handoff sequence
session.save_state()
result = session.handoff("agent-2")
print(f"    [OK] Session handed off to {result['target']}")
print(f"         WAL: {result['wal_entries']}, Committed: {result['committed']}")

# Final horizon
final_horizon = event_store.latest()
print()
print(f"    Final horizon: {final_horizon.value[:12] if final_horizon else 'None'}...")
print(f"    Total events in store: {event_store.count}")
print()


# =============================================================================
# STEP 4: Temporal Queries - "What was the state at horizon X?"
# =============================================================================

print("[4] Temporal Queries - Querying contract state at different horizons")
print()

def state_at_horizon(horizon: Optional[MerkleRoot]) -> ContractState:
    """Compute contract state up to a specific horizon."""
    state = ContractState()

    if horizon is None:
        return state

    # Iterate events up to horizon
    for event in event_store.iterate(to_event=horizon.value):
        content = event.content
        timestamp = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))

        if content.get('type') == 'contract_check':
            state.total_checks += 1
            state.last_check = timestamp
            if state.first_check is None:
                state.first_check = timestamp

        if content.get('observation_type') == 'contract_violation':
            state.total_violations += 1
            state.last_violation = timestamp

            method = content.get('violation_detail', {}).get('method', 'unknown')
            if method not in state.violations_by_method:
                state.violations_by_method[method] = 0
            state.violations_by_method[method] += 1

    return state

# Query state at different points in time
print("    State after task completion (before violations):")
state_pre_violation = state_at_horizon(post_task_horizon)
print(f"      Checks: {state_pre_violation.total_checks}, Violations: {state_pre_violation.total_violations}")
print(f"      Healthy: {state_pre_violation.is_healthy}")
print()

print("    State after first violation:")
state_post_violation = state_at_horizon(violation1_horizon)
print(f"      Checks: {state_post_violation.total_checks}, Violations: {state_post_violation.total_violations}")
print(f"      Healthy: {state_post_violation.is_healthy}")
print()

print("    Final state:")
final_state = materializer.current_state()
print(f"      Checks: {final_state.total_checks}, Violations: {final_state.total_violations}")
print(f"      Violation rate: {final_state.violation_rate:.2f}%")
print(f"      Healthy: {final_state.is_healthy}")
print()

# Show temporal comparison
print("    Temporal comparison:")
print(f"      Pre-violation: {state_pre_violation.violation_rate:.1f}% violation rate")
print(f"      Post-violation: {final_state.violation_rate:.1f}% violation rate")
print(f"      Delta: +{final_state.total_violations - state_pre_violation.total_violations} violations")
print()


# =============================================================================
# STEP 5: Compaction Demonstration
# =============================================================================

print("[5] Compaction - Semantic compression of contract history")
print()

def demonstrate_compaction():
    """
    Demonstrate how contract events could be compacted.

    Compaction preserves the same materialized state while reducing
    event count. For contracts:
    - Multiple passing checks for same method -> single summary
    - Violation count preserved (never lose violation history)
    """
    print("    Before compaction:")
    print(f"      Total events: {event_store.count}")

    # Count events by type
    check_events = 0
    violation_events = 0
    other_events = 0

    for event in event_store.iterate():
        if event.event_type == EventType.OBSERVATION:
            check_events += 1
        elif event.event_type == EventType.METACOGNITION:
            violation_events += 1
        else:
            other_events += 1

    print(f"      Contract checks: {check_events}")
    print(f"      MetaCognition (violations): {violation_events}")
    print(f"      Other events: {other_events}")
    print()

    # Calculate compaction potential
    # Contract checks for same method could be compacted to counts
    method_check_counts: Dict[str, int] = {}
    for event in event_store.iterate():
        if event.event_type == EventType.OBSERVATION:
            method = event.content.get('method', 'unknown')
            if method not in method_check_counts:
                method_check_counts[method] = 0
            method_check_counts[method] += 1

    compactable = sum(count - 1 for count in method_check_counts.values() if count > 1)

    print("    Compaction analysis:")
    print(f"      Methods with multiple checks: {sum(1 for c in method_check_counts.values() if c > 1)}")
    print(f"      Events that could be compacted: {compactable}")
    print(f"      Compaction ratio: {(event_store.count - compactable) / event_store.count:.1%}")
    print()

    # Show what a compaction event would look like
    print("    Example compaction event structure:")
    print("      {")
    print("        'type': 'COMPACTION',")
    print("        'compressed_events': [<event_ids>],")
    print("        'snapshot': {")
    print("          'total_checks': %d," % final_state.total_checks)
    print("          'total_violations': %d," % final_state.total_violations)
    print("          'checks_by_method': {...}")
    print("        },")
    print("        'preserved_merkle_root': '%s...'" % (final_horizon.value[:12] if final_horizon else 'None'))
    print("      }")
    print()
    print("    Key invariant: materialize(events) == materialize(compact(events))")
    print()

demonstrate_compaction()


# =============================================================================
# STEP 6: Recovery Flow - Handling violations gracefully
# =============================================================================

print("[6] Recovery Flow - Handling contract violations gracefully")
print()

class RecoverableSession:
    """
    A session that can recover from contract violations.

    Instead of crashing on violation, captures the violation event
    and suggests corrective actions.
    """

    def __init__(self, session_id: str, registry: ContractRegistry):
        self._inner = CognitiveSession(session_id)
        self._registry = registry
        self._recovery_log: List[Dict[str, Any]] = []

    def safe_handoff(self, target: str) -> Optional[dict]:
        """Attempt handoff with automatic recovery."""
        try:
            return self._inner.handoff(target)
        except ContractViolation as e:
            # Log violation for analysis
            recovery_entry = {
                'operation': 'handoff',
                'violation': e.description,
                'suggestion': 'Save state before handoff',
                'timestamp': datetime.now().isoformat(),
            }
            self._recovery_log.append(recovery_entry)

            # Attempt automatic recovery
            if 'save state' in e.description.lower():
                print(f"      [RECOVERY] Auto-saving state...")
                self._inner.save_state()
                return self._inner.handoff(target)

            return None

    @property
    def recovery_log(self) -> List[Dict[str, Any]]:
        return self._recovery_log

# Demonstrate recovery
print("    Demonstrating automatic recovery from violation...")
print()

# Create a new session for recovery demo
recovery_session = RecoverableSession("S-002", registry)
recovery_session._inner.begin()
recovery_session._inner.create_task("T-003", "Recovery demo task")
print("    Created new session S-002 with task T-003")

# This would normally fail - state not saved
result = recovery_session.safe_handoff("agent-3")
if result:
    print(f"    [OK] Recovery succeeded! Handed off to {result['target']}")
    print()
    print("    Recovery log:")
    for entry in recovery_session.recovery_log:
        print(f"      - {entry['operation']}: {entry['violation']}")
        print(f"        Suggestion: {entry['suggestion']}")
print()


# =============================================================================
# STEP 7: CEL Event Store Statistics
# =============================================================================

print("[7] CEL Event Store Statistics")
print()

print(f"    Total events: {event_store.count}")
print(f"    Branch heads: {len(event_store.heads())}")
print(f"    Latest horizon: {event_store.latest().value[:16] if event_store.latest() else 'None'}...")
print()

# Count by event type
type_counts: Dict[str, int] = {}
for event in event_store.iterate():
    type_name = event.event_type.name
    if type_name not in type_counts:
        type_counts[type_name] = 0
    type_counts[type_name] += 1

print("    Events by type:")
for type_name, count in sorted(type_counts.items()):
    print(f"      {type_name}: {count}")
print()


# =============================================================================
# STEP 8: Health Report
# =============================================================================

print("[8] Contract Health Report")
print()

report = materializer.health_report()

print(f"    Status: {report['status']}")
print(f"    Recent violations (24h): {report['recent_violations_24h']}")
print()

if report['recommendations']:
    print("    Recommendations:")
    for rec in report['recommendations']:
        print(f"      - {rec}")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY - CEL Integration Features Demonstrated")
print("=" * 70)
print()
print("  1. In-Memory EventStore:")
print("     - Implements CEL EventStore protocol")
print("     - Content-addressed events with Merkle roots")
print("     - Causal ordering preserved")
print()
print("  2. Temporal Queries:")
print("     - 'What was contract state at horizon X?'")
print("     - Query state before/after violations")
print("     - Track contract health over time")
print()
print("  3. Compaction Awareness:")
print("     - Multiple checks -> summary stats")
print("     - Preserves violation history")
print("     - Maintains materialized state invariant")
print()
print("  4. Recovery Flow:")
print("     - Catch violations gracefully")
print("     - Suggest corrective actions")
print("     - Automatic recovery when possible")
print()
print("  5. Integration Points:")
print("     - Contract checks -> OBSERVATION events")
print("     - Violations -> METACOGNITION events (self-awareness)")
print("     - Full audit trail in event store")
print()
print("=" * 70)
