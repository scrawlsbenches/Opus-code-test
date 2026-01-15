#!/usr/bin/env python3
"""
Behavioral Contracts - End-to-End Demonstration

This example demonstrates behavioral contracts in the context CEL was
designed for: cognitive operations, task management, and knowledge transfer.

The contracts express INTENT about:
- Task state machine transitions (pending → in_progress → completed)
- Knowledge transfer lifecycle (draft → finalized)
- Session integrity (active before work, clean handoff)
- Cognitive invariants (WAL entries >= committed entities)

Run with:
    python examples/behavioral_contracts_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

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


# =============================================================================
# STEP 1: Create Registry and Emitter
# =============================================================================

print("=" * 70)
print("BEHAVIORAL CONTRACTS - COGNITIVE OPERATIONS DEMO")
print("=" * 70)
print()

# Create event emitter (standalone mode - no CEL store)
# In production, you'd pass a CEL EventStore here
emitter = ContractEventEmitter(emit_all_checks=True)

# Create registry with emitter
registry = ContractRegistry(emitter=emitter)

print("[1] Created ContractRegistry and ContractEventEmitter")
print("    - emit_all_checks: True (for demonstration)")
print("    - CEL store: None (using in-memory buffer)")
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
print("    Task (state machine):")
print("      - start(): @requires PENDING → @ensures IN_PROGRESS")
print("      - complete(): @requires IN_PROGRESS + retrospective → @ensures COMPLETED")
print("      - block(): @requires IN_PROGRESS + reason")
print("      - cancel(): @requires not terminal state")
print()
print("    KnowledgeTransfer (document lifecycle):")
print("      - append(): @requires DRAFT, non-empty content")
print("      - set_summary(): @requires DRAFT, min 10 chars")
print("      - finalize(): @requires DRAFT, not empty, has summary")
print()
print("    CognitiveSession (session integrity):")
print("      - begin(): @requires not active → @ensures active")
print("      - create_task(): @requires active, @invariant WAL >= committed")
print("      - save_state(): @requires active, @invariant WAL >= committed")
print("      - handoff(): @requires active + state saved → @ensures not active")
print()


# =============================================================================
# STEP 3: Execute Operations (Some Will Pass, Some Will Fail)
# =============================================================================

print("[3] Executing cognitive operations...")
print()

# Create a session
session = CognitiveSession("S-001")
print("    [OK] Session S-001 created")

session.begin()
print("    [OK] Session began")

# Create and work on tasks
task1 = session.create_task("T-001", "Implement behavioral contracts")
print("    [OK] Created task T-001")

task1.start()
print("    [OK] Task T-001 started (PENDING → IN_PROGRESS)")

task1.complete("Implemented @requires, @ensures, @invariant with CEL integration")
print("    [OK] Task T-001 completed with retrospective")

# Create a knowledge transfer
kt = KnowledgeTransfer("KT-001", "Session Learnings")
kt.append("What Worked", "Contract decorators capture intent effectively")
kt.append("Challenges", "Decorator ordering matters - track must be outermost")
kt.set_summary("Behavioral contracts enable executable documentation")
kt.finalize()
print("    [OK] Knowledge transfer KT-001 created and finalized")

print()
print("    Now attempting operations that violate contracts...")
print()

# Violation 1: Complete a task that isn't in progress
task2 = session.create_task("T-002", "Another task")
try:
    print("    [FAIL] Complete T-002 without starting it")
    task2.complete("This should fail")
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Violation 2: Append to finalized KT
try:
    print("    [FAIL] Append to finalized KT-001")
    kt.append("New Section", "This should fail")
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Violation 3: Finalize empty KT
kt2 = KnowledgeTransfer("KT-002", "Empty KT")
try:
    print("    [FAIL] Finalize empty KT-002")
    kt2.finalize()
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Violation 4: Handoff without saving state
try:
    print("    [FAIL] Handoff without saving state")
    session.handoff("agent-2")
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()

# Proper handoff sequence
session.save_state()
print("    [OK] Session state saved")

result = session.handoff("agent-2")
print(f"    [OK] Session handed off to {result['target']}")
print(f"         Tasks: {result['tasks']}, WAL: {result['wal_entries']}, Committed: {result['committed']}")
print()

# Violation 5: Create task on inactive session
try:
    print("    [FAIL] Create task on ended session")
    session.create_task("T-003", "This should fail")
except ContractViolation as e:
    print(f"           Caught: {e.contract_type} - {e.description}")
print()


# =============================================================================
# STEP 4: Query Contract State via Materializer
# =============================================================================

print("[4] Materializing contract state from events...")
print()

materializer = ContractMaterializer(emitter=emitter)
state = materializer.current_state()

print(f"    {state}")
print(f"    - Total checks: {state.total_checks}")
print(f"    - Total violations: {state.total_violations}")
print(f"    - Violation rate: {state.violation_rate:.2f}%")
print(f"    - Is healthy: {state.is_healthy}")
print()

if state.violations_by_method:
    print("    Violations by method:")
    for method, count in state.violations_by_method.items():
        print(f"      - {method}: {count}")
    print()


# =============================================================================
# STEP 5: View CEL Events (Buffered)
# =============================================================================

print("[5] CEL Events emitted (in-memory buffer):")
print()

events = emitter.buffered_events
print(f"    Total events: {len(events)}")
print()

# Show last 5 events
print("    Last 5 events:")
for event in events[-5:]:
    event_type = event.get('event_type', 'UNKNOWN')
    content = event.get('content', {})

    if event_type == 'OBSERVATION':
        check_type = content.get('contract_type', 'unknown')
        method = content.get('method', 'unknown')
        passed = content.get('passed', False)
        status = "PASS" if passed else "FAIL"
        print(f"      [{event_type}] {status} {check_type} on {method}")
    elif event_type == 'METACOGNITION':
        obs_type = content.get('observation_type', 'unknown')
        conclusions = content.get('conclusions', [])
        print(f"      [{event_type}] {obs_type}: {conclusions[0] if conclusions else 'N/A'}")
print()


# =============================================================================
# STEP 6: Generate Health Report
# =============================================================================

print("[6] Health Report:")
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
# STEP 7: Query Recent Violations
# =============================================================================

print("[7] Recent Violations:")
print()

violations = materializer.violations_since(hours=1)
print(f"    Found {len(violations)} violations in last hour:")
print()

for v in violations:
    print(f"      {v.timestamp.strftime('%H:%M:%S')} | {v.method}")
    print(f"        {v.description}")
print()


# =============================================================================
# STEP 8: Registry Statistics
# =============================================================================

print("[8] Contract Registry Statistics:")
print()

stats = registry.stats()
print(f"    Total contracts: {stats['total_contracts']}")
print(f"    Contracts with violations: {stats['contracts_with_violations']}")
print(f"    Total checks: {stats['total_checks']}")
print(f"    Total violations: {stats['total_violations']}")
print(f"    Violation rate: {stats['violation_rate']:.2%}")
print()

print("    By type:")
for type_name, count in stats['by_type'].items():
    print(f"      - {type_name}: {count}")
print()


# =============================================================================
# STEP 9: Export Documentation
# =============================================================================

print("[9] Exported Contract Documentation:")
print()

docs = registry.export_documentation()
# Show first 40 lines
lines = docs.split('\n')[:40]
for line in lines:
    print(f"    {line}")
print("    ...")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("This demonstration showed behavioral contracts in cognitive operations:")
print()
print("  Task State Machine:")
print("    - Contracts enforce valid state transitions")
print("    - Cannot complete without starting first")
print("    - Terminal states cannot be exited")
print()
print("  Knowledge Transfer Lifecycle:")
print("    - Cannot finalize empty documents")
print("    - Cannot modify finalized documents")
print("    - Must have summary before finalizing")
print()
print("  Cognitive Session Integrity:")
print("    - WAL entries >= committed (durability invariant)")
print("    - Must save state before handoff")
print("    - Cannot work on inactive session")
print()
print("Integration with CEL enables:")
print("  - Contract violations → MetaCognition events (self-awareness)")
print("  - Temporal queries: 'What contracts were violated during task T-001?'")
print("  - Compaction: Summarize contract history into statistics")
print("  - Audit trail: Full history of cognitive operations")
print()
print("=" * 70)
