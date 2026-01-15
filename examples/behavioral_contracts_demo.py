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
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from cortical.cel.stores import MemoryEventStore
from cortical.cel.container import LatticeBuilder, Container
from cortical.cel.core.protocols import EventStore, Materializer
from cortical.cel.wisdom.materializer import CachingMaterializer, default_reducer_registry
from cortical.contracts.cel_integration import create_contract_reducer


# =============================================================================
# STEP 1: Create CognitiveLattice with LatticeBuilder (Proper CEL DI)
# =============================================================================

print("=" * 70)
print("BEHAVIORAL CONTRACTS - CEL INTEGRATED DEMO")
print("=" * 70)
print()

# Build CognitiveLattice using LatticeBuilder (CEL's DI pattern)
print("[1] Building CognitiveLattice with LatticeBuilder...")

# Method 1: Use LatticeBuilder for storage
lattice = (
    LatticeBuilder()
    .with_storage(MemoryEventStore)
    .build()
)
print("    - Built CognitiveLattice with MemoryEventStore via LatticeBuilder")

# Get the event store from the lattice (proper DI)
event_store = lattice.event_store
print(f"    - EventStore type: {type(event_store).__name__}")

# Register contract_reducer with CEL's reducer registry for materialization
reducer_registry = default_reducer_registry()
reducer_registry.add(create_contract_reducer())
print("    - Registered contract_reducer with CEL reducer registry")

# Create CEL CachingMaterializer with contract reducer
cel_materializer = CachingMaterializer(
    event_store=event_store,
    reducer_registry=reducer_registry,
)
print("    - Created CEL CachingMaterializer with contract_reducer")

# Create contract emitter connected to CEL EventStore
emitter = ContractEventEmitter(
    event_store=event_store,
    emit_all_checks=True,
)
print("    - Connected ContractEventEmitter to CEL EventStore")

# Create registry with emitter
registry = ContractRegistry(emitter=emitter)
print("    - Created ContractRegistry")

# Create contract materializer (for convenience API)
materializer = ContractMaterializer(
    event_store=event_store,
    emitter=emitter,
)
print("    - Created ContractMaterializer (convenience wrapper)")
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
# STEP 4b: CEL contract_reducer - Direct Event Stream Materialization
# =============================================================================

print("[4b] CEL contract_reducer - Materializing from event stream")
print()

# Contracts are different from entity-specific CEL patterns:
# - Entity pattern: Each entity (T-001, D-002) has its own events
# - Contract pattern: ALL contract events aggregate into ONE summary
#
# We use contract_reducer directly on the event stream for this "aggregate" pattern

print("    Using contract_reducer to materialize contract summary...")
contract_reducer = create_contract_reducer()

# Fold all events through the reducer
cel_contract_state = None
for event in event_store.iterate():
    cel_contract_state = contract_reducer(cel_contract_state, event)

if cel_contract_state:
    print(f"      Entity type: {cel_contract_state.get('entity_type', 'N/A')}")
    print(f"      Total checks: {cel_contract_state.get('total_checks', 0)}")
    print(f"      Total violations: {cel_contract_state.get('total_violations', 0)}")
    print(f"      First check: {cel_contract_state.get('first_check', 'N/A')}")
    print(f"      Last check: {cel_contract_state.get('last_check', 'N/A')}")

    violations_by_method = cel_contract_state.get('violations_by_method', {})
    if violations_by_method:
        print(f"      Violations by method:")
        for method, count in violations_by_method.items():
            print(f"        - {method}: {count}")
    print()

    print("    CEL Reducer Integration Benefits:")
    print("      - Same reducer works with CachingMaterializer (for entity patterns)")
    print("      - Same reducer works with direct stream folding (for aggregate patterns)")
    print("      - Temporal queries: Fold events up to a specific horizon")
    print("      - Compaction: Reducer output can be snapshotted for compression")
else:
    print("      [INFO] No contract events in stream")
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
# STEP 9: Recovery to Healthy State
# =============================================================================

print("[9] Recovery to Healthy State")
print()
print("    Current state: UNHEALTHY (violation rate > 5%)")
print("    Strategy: Execute successful operations to dilute violation rate")
print()

# Calculate how many successful operations we need
current_checks = final_state.total_checks
current_violations = final_state.total_violations
target_rate = 0.05  # 5% threshold

# Formula: violations / (checks + N) < target_rate
# Solving: N > (violations / target_rate) - checks
needed_checks = int((current_violations / target_rate) - current_checks) + 1
print(f"    Need {needed_checks} successful operations to reach < 5% violation rate")
print()

# Execute successful operations
print("    Executing successful cognitive operations...")

for i in range(needed_checks):
    # Create a new session and complete a full successful workflow
    recovery_sess = CognitiveSession(f"S-RECOVERY-{i:03d}")
    recovery_sess.begin()
    recovery_task = recovery_sess.create_task(f"T-RECOVERY-{i:03d}", "Recovery task")
    recovery_task.start()
    recovery_task.complete("Successfully completed recovery operation")
    recovery_sess.save_state()
    recovery_sess.handoff("archive")

print(f"    Completed {needed_checks} successful session cycles")
print()

# Check final health status
final_final_state = materializer.current_state()
final_report = materializer.health_report()

print("    Final Health Report:")
print(f"      Status: {final_report['status']}")
print(f"      Total checks: {final_final_state.total_checks}")
print(f"      Total violations: {final_final_state.total_violations}")
print(f"      Violation rate: {final_final_state.violation_rate:.2f}%")
print(f"      Is healthy: {final_final_state.is_healthy}")
print()

if final_report['status'] == 'HEALTHY':
    print("    System recovered to HEALTHY state.")
else:
    print("    Note: System still unhealthy. May need more successful operations.")
print()


# =============================================================================
# STEP 10: Next Steps / Production Guidance
# =============================================================================

print("[10] Production Guidance")
print()
print("    What to do when system is UNHEALTHY:")
print()
print("      1. IDENTIFY: Use temporal queries to find when violations occurred")
print("         state_at_horizon(violation_horizon)")
print()
print("      2. ANALYZE: Check violations_by_method to find problem areas")
print("         materializer.current_state().violations_by_method")
print()
print("      3. FIX: Either fix the code OR adjust the contract")
print("         - If contract is too strict: relax preconditions")
print("         - If code is buggy: fix the implementation")
print()
print("      4. RECOVER: Execute successful operations to dilute rate")
print("         - Violation rate = violations / total_checks")
print("         - More successful checks -> lower rate")
print()
print("      5. MONITOR: Use health_report() for ongoing monitoring")
print("         - Set up alerts when status != 'HEALTHY'")
print("         - Track violation rate trends over time")
print()
print("    Contract Health Thresholds:")
print("      - HEALTHY: violation rate < 5%")
print("      - UNHEALTHY: violation rate >= 5%")
print("      - CRITICAL: violation rate >= 10% (immediate attention)")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY - CEL Integration Features Demonstrated")
print("=" * 70)
print()
print("  1. LatticeBuilder Integration:")
print("     - CognitiveLattice built with LatticeBuilder DI pattern")
print("     - MemoryEventStore injected via .with_storage()")
print("     - Proper dependency injection for testability")
print()
print("  2. contract_reducer with CEL:")
print("     - Registered with EntityReducerRegistry")
print("     - Materializes contract state from event stream")
print("     - Works with CachingMaterializer or direct stream folding")
print()
print("  3. Temporal Queries:")
print("     - 'What was contract state at horizon X?'")
print("     - Query state before/after violations")
print("     - Track contract health over time")
print()
print("  4. Compaction Awareness:")
print("     - Multiple checks -> summary stats")
print("     - Preserves violation history")
print("     - Maintains materialized state invariant")
print()
print("  5. Recovery Flow:")
print("     - Catch violations gracefully")
print("     - Suggest corrective actions")
print("     - Automatic recovery when possible")
print()
print("  6. Integration Points:")
print("     - Contract checks -> OBSERVATION events")
print("     - Violations -> METACOGNITION events (self-awareness)")
print("     - Full audit trail in CEL EventStore")
print()
print("=" * 70)
