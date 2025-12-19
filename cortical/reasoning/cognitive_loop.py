"""
QAPV Cognitive Loop: The Core Reasoning Cycle.

This module implements the Question-Answer-Produce-Verify (QAPV) cognitive loop
from docs/complex-reasoning-workflow.md. This is the fundamental unit of
collaborative work between humans and AI agents.

The loop structure:
    QUESTION → ANSWER → PRODUCE → VERIFY → (loop with new knowledge)

Each phase contains nested sub-phases (fractal structure), and the loop can be
entered at any phase depending on context (debugging starts at VERIFY,
greenfield starts at QUESTION, spiking starts at PRODUCE).

Key Design Decisions:
    - Loops are hierarchical: each phase can spawn child loops
    - State is explicit: observers can see exactly where we are
    - Transitions are logged: builds audit trail for knowledge transfer
    - Termination is explicit: clear exit conditions prevent infinite loops

Implementation Notes:
    - This is a state machine with push/pop semantics for nested loops
    - Time boxing is enforced per phase (see time_boxing.py)
    - Crisis detection hooks into loop iteration count (see crisis_manager.py)
    - All transitions emit events for metrics collection (see metrics.py)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid


class LoopPhase(Enum):
    """
    The four phases of the QAPV cognitive loop.

    Each phase has distinct characteristics:
    - QUESTION: Reduce ambiguity, discover constraints, clarify intent
    - ANSWER: Research, analyze, propose solutions, evaluate options
    - PRODUCE: Create artifacts (code, docs, tests), implement solutions
    - VERIFY: Confirm correctness, run tests, validate against requirements
    """
    QUESTION = "question"
    ANSWER = "answer"
    PRODUCE = "produce"
    VERIFY = "verify"


class LoopStatus(Enum):
    """Status of a cognitive loop instance."""
    NOT_STARTED = auto()
    ACTIVE = auto()
    PAUSED = auto()  # Waiting for external input
    BLOCKED = auto()  # Cannot proceed without resolution
    COMPLETED = auto()
    ABANDONED = auto()  # Consciously stopped


class TerminationReason(Enum):
    """Why a loop was terminated."""
    SUCCESS = "success"  # All acceptance criteria passed
    USER_APPROVED = "user_approved"  # Explicit human approval
    BUDGET_EXHAUSTED = "budget_exhausted"  # Time/resource limit reached
    QUESTION_INVALID = "question_invalid"  # Discovered the question was wrong
    ESCALATED = "escalated"  # Handed off to human
    CRISIS = "crisis"  # Critical failure detected


@dataclass
class PhaseContext:
    """
    Context for a single phase execution.

    Tracks what's happening within a phase, enabling:
    - Time boxing and progress monitoring
    - Crisis detection (iteration counting)
    - Knowledge capture for handoffs
    """
    phase: LoopPhase
    started_at: datetime
    iteration: int = 1  # Which iteration of this phase
    notes: List[str] = field(default_factory=list)
    questions_raised: List[str] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    artifacts_produced: List[str] = field(default_factory=list)
    time_box_minutes: int = 30  # Default time box

    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def add_note(self, note: str) -> None:
        """Add a note to this phase's context."""
        self.notes.append(f"[{datetime.now().isoformat()}] {note}")

    def record_question(self, question: str) -> None:
        """Record a question raised during this phase."""
        self.questions_raised.append(question)

    def record_decision(self, decision: str, rationale: str = "") -> None:
        """Record a decision made during this phase."""
        self.decisions_made.append({
            'decision': decision,
            'rationale': rationale,
            'timestamp': datetime.now().isoformat()
        })

    def end_phase(self) -> None:
        """Mark this phase as ended and calculate duration."""
        self.ended_at = datetime.now()
        self.duration_seconds = (self.ended_at - self.started_at).total_seconds()

    def elapsed_minutes(self) -> float:
        """Return elapsed time in minutes."""
        return (datetime.now() - self.started_at).total_seconds() / 60


@dataclass
class LoopTransition:
    """
    Record of a phase transition within a loop.

    Used for:
    - Audit trail (what happened when)
    - Metrics collection (how long each phase took)
    - Knowledge transfer (understanding the reasoning process)
    """
    from_phase: Optional[LoopPhase]
    to_phase: LoopPhase
    timestamp: datetime
    reason: str
    context_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveLoop:
    """
    A single instance of the QAPV cognitive loop.

    This represents one cycle of Question→Answer→Produce→Verify, potentially
    with nested child loops for complex sub-tasks.

    Attributes:
        id: Unique identifier for this loop instance
        goal: What this loop is trying to achieve
        status: Current status of the loop
        current_phase: Which phase we're in
        phase_contexts: History of phase executions
        transitions: Log of all phase transitions
        parent_id: ID of parent loop if this is nested
        child_ids: IDs of any child loops spawned
        created_at: When this loop was created
        completed_at: When this loop finished (if applicable)
        termination_reason: Why the loop ended (if applicable)

    Example:
        >>> loop = CognitiveLoop(goal="Implement user authentication")
        >>> loop.start(LoopPhase.QUESTION)
        >>> loop.add_note("Need to clarify: OAuth vs session-based?")
        >>> loop.transition(LoopPhase.ANSWER, reason="User clarified: use OAuth")
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    status: LoopStatus = LoopStatus.NOT_STARTED
    current_phase: Optional[LoopPhase] = None
    phase_contexts: List[PhaseContext] = field(default_factory=list)
    transitions: List[LoopTransition] = field(default_factory=list)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    termination_reason: Optional[TerminationReason] = None

    # Callbacks for extensibility (see STUB comments below)
    _on_transition: Optional[Callable[['CognitiveLoop', LoopTransition], None]] = None
    _on_complete: Optional[Callable[['CognitiveLoop', TerminationReason], None]] = None

    def start(self, initial_phase: LoopPhase = LoopPhase.QUESTION) -> PhaseContext:
        """
        Start the cognitive loop at a given phase.

        Args:
            initial_phase: Which phase to start at (default: QUESTION)

        Returns:
            PhaseContext for the initial phase

        Raises:
            ValueError: If loop is already started
        """
        if self.status != LoopStatus.NOT_STARTED:
            raise ValueError(f"Loop already in status {self.status}")

        self.status = LoopStatus.ACTIVE
        self.current_phase = initial_phase

        context = PhaseContext(phase=initial_phase, started_at=datetime.now())
        self.phase_contexts.append(context)

        transition = LoopTransition(
            from_phase=None,
            to_phase=initial_phase,
            timestamp=datetime.now(),
            reason="Loop started"
        )
        self.transitions.append(transition)

        if self._on_transition:
            self._on_transition(self, transition)

        return context

    def transition(self, to_phase: LoopPhase, reason: str) -> PhaseContext:
        """
        Transition to a new phase.

        Args:
            to_phase: Phase to transition to
            reason: Why we're making this transition

        Returns:
            PhaseContext for the new phase

        Raises:
            ValueError: If loop is not active
        """
        if self.status != LoopStatus.ACTIVE:
            raise ValueError(f"Cannot transition in status {self.status}")

        from_phase = self.current_phase
        self.current_phase = to_phase

        # Calculate iteration (how many times we've entered this phase)
        iteration = sum(1 for ctx in self.phase_contexts if ctx.phase == to_phase) + 1

        context = PhaseContext(
            phase=to_phase,
            started_at=datetime.now(),
            iteration=iteration
        )
        self.phase_contexts.append(context)

        transition = LoopTransition(
            from_phase=from_phase,
            to_phase=to_phase,
            timestamp=datetime.now(),
            reason=reason,
            context_snapshot=self._snapshot_context()
        )
        self.transitions.append(transition)

        if self._on_transition:
            self._on_transition(self, transition)

        return context

    def complete(self, reason: TerminationReason) -> None:
        """
        Mark this loop as complete.

        Args:
            reason: Why the loop is ending
        """
        self.status = LoopStatus.COMPLETED
        self.completed_at = datetime.now()
        self.termination_reason = reason

        if self._on_complete:
            self._on_complete(self, reason)

    def pause(self, reason: str) -> None:
        """Pause the loop (waiting for external input)."""
        self.status = LoopStatus.PAUSED
        self.current_context().add_note(f"PAUSED: {reason}")

    def resume(self) -> None:
        """Resume a paused loop."""
        if self.status != LoopStatus.PAUSED:
            raise ValueError(f"Cannot resume from status {self.status}")
        self.status = LoopStatus.ACTIVE
        self.current_context().add_note("RESUMED")

    _block_reason: Optional[str] = None

    @property
    def block_reason(self) -> Optional[str]:
        """Get the reason why this loop is blocked."""
        return self._block_reason

    def block(self, blocker: str) -> None:
        """Mark loop as blocked on a dependency."""
        self.status = LoopStatus.BLOCKED
        self._block_reason = blocker
        self.current_context().add_note(f"BLOCKED: {blocker}")

    def abandon(self, reason: str) -> None:
        """Consciously stop the loop without completing."""
        self.status = LoopStatus.ABANDONED
        self.completed_at = datetime.now()
        self.current_context().add_note(f"ABANDONED: {reason}")

    def current_context(self) -> PhaseContext:
        """Get the context for the current phase."""
        if not self.phase_contexts:
            raise ValueError("Loop has no phase contexts")
        return self.phase_contexts[-1]

    def add_note(self, note: str) -> None:
        """Add a note to the current phase."""
        self.current_context().add_note(note)

    def get_all_contexts(self) -> List[PhaseContext]:
        """Get all phase contexts for this loop."""
        return self.phase_contexts.copy()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this loop's state.

        Returns:
            Dictionary with id, goal, status, phase info, etc.
        """
        return {
            'id': self.id,
            'goal': self.goal,
            'status': self.status.name,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'phase_count': len(self.phase_contexts),
            'transition_count': len(self.transitions),
            'elapsed_minutes': self.total_elapsed_minutes(),
            'parent_id': self.parent_id,
            'child_count': len(self.child_ids),
        }

    def spawn_child(self, goal: str) -> 'CognitiveLoop':
        """
        Spawn a nested child loop for a sub-task.

        Args:
            goal: What the child loop should achieve

        Returns:
            New CognitiveLoop instance linked to this parent
        """
        child = CognitiveLoop(
            goal=goal,
            parent_id=self.id,
            _on_transition=self._on_transition,
            _on_complete=self._on_complete
        )
        self.child_ids.append(child.id)
        return child

    def get_iteration_count(self, phase: LoopPhase) -> int:
        """
        Get how many times we've been in a specific phase.

        Used for crisis detection: 3+ iterations without progress is a warning sign.
        """
        return sum(1 for ctx in self.phase_contexts if ctx.phase == phase)

    def total_elapsed_minutes(self) -> float:
        """Total time spent in this loop."""
        end = self.completed_at or datetime.now()
        return (end - self.created_at).total_seconds() / 60

    def _snapshot_context(self) -> Dict[str, Any]:
        """Create a snapshot of current context for transition log."""
        ctx = self.current_context()
        return {
            'phase': ctx.phase.value,
            'iteration': ctx.iteration,
            'elapsed_minutes': ctx.elapsed_minutes(),
            'notes_count': len(ctx.notes),
            'questions_raised': len(ctx.questions_raised),
            'decisions_made': len(ctx.decisions_made),
            'artifacts_produced': len(ctx.artifacts_produced),
        }


class CognitiveLoopManager:
    """
    Manager for multiple concurrent cognitive loops.

    Handles:
    - Creating and tracking loop instances
    - Detecting loops that need attention (stuck, blocked, overdue)
    - Aggregating metrics across loops
    - Supporting parallel exploration with coordination

    STUB: Full implementation would include:
    - Integration with external task systems
    - Persistent storage for loop state
    - Real-time monitoring and alerting
    - ML-based suggestions for loop optimization
    """

    def __init__(self):
        """Initialize the loop manager."""
        self._loops: Dict[str, CognitiveLoop] = {}
        self._active_loops: Dict[str, CognitiveLoop] = {}

        # Event handlers (extensibility points)
        self._transition_handlers: List[Callable[[CognitiveLoop, LoopTransition], None]] = []
        self._completion_handlers: List[Callable[[CognitiveLoop, TerminationReason], None]] = []

    def create_loop(self, goal: str, parent_id: Optional[str] = None) -> CognitiveLoop:
        """
        Create a new cognitive loop.

        Args:
            goal: What this loop is trying to achieve
            parent_id: Optional parent loop ID for nested loops

        Returns:
            New CognitiveLoop instance
        """
        loop = CognitiveLoop(
            goal=goal,
            parent_id=parent_id,
            _on_transition=self._handle_transition,
            _on_complete=self._handle_completion
        )
        self._loops[loop.id] = loop
        return loop

    def get_loop(self, loop_id: str) -> Optional[CognitiveLoop]:
        """Get a loop by ID."""
        return self._loops.get(loop_id)

    def get_active_loops(self) -> List[CognitiveLoop]:
        """Get all currently active loops."""
        return [
            loop for loop in self._loops.values()
            if loop.status == LoopStatus.ACTIVE
        ]

    def get_blocked_loops(self) -> List[CognitiveLoop]:
        """Get all blocked loops that need attention."""
        return [
            loop for loop in self._loops.values()
            if loop.status == LoopStatus.BLOCKED
        ]

    def get_stuck_loops(self, iteration_threshold: int = 3) -> List[CognitiveLoop]:
        """
        Find loops that appear to be stuck (many iterations without progress).

        This implements the "danger sign" from the workflow doc:
        "Looping without progress. If you've done 3 iterations without
        measurable improvement, STOP and escalate."

        Args:
            iteration_threshold: Number of iterations before considering stuck

        Returns:
            List of loops that may be stuck
        """
        stuck = []
        for loop in self._loops.values():
            if loop.status != LoopStatus.ACTIVE:
                continue

            # Check if any phase has exceeded threshold
            for phase in LoopPhase:
                if loop.get_iteration_count(phase) >= iteration_threshold:
                    stuck.append(loop)
                    break

        return stuck

    def get_overdue_loops(self, max_minutes: float = 120) -> List[CognitiveLoop]:
        """
        Find loops that have exceeded their time budget.

        Args:
            max_minutes: Maximum expected loop duration

        Returns:
            List of loops that may be taking too long
        """
        return [
            loop for loop in self._loops.values()
            if loop.status == LoopStatus.ACTIVE
            and loop.total_elapsed_minutes() > max_minutes
        ]

    def register_transition_handler(
        self,
        handler: Callable[[CognitiveLoop, LoopTransition], None]
    ) -> None:
        """Register a handler for loop transitions (for metrics, logging, etc.)."""
        self._transition_handlers.append(handler)

    def register_completion_handler(
        self,
        handler: Callable[[CognitiveLoop, TerminationReason], None]
    ) -> None:
        """Register a handler for loop completions."""
        self._completion_handlers.append(handler)

    def _handle_transition(self, loop: CognitiveLoop, transition: LoopTransition) -> None:
        """Internal handler called on any loop transition."""
        for handler in self._transition_handlers:
            try:
                handler(loop, transition)
            except Exception as e:
                # Log but don't fail on handler errors
                loop.add_note(f"Transition handler error: {e}")

    def _handle_completion(self, loop: CognitiveLoop, reason: TerminationReason) -> None:
        """Internal handler called on loop completion."""
        for handler in self._completion_handlers:
            try:
                handler(loop, reason)
            except Exception as e:
                # Log but don't fail on handler errors
                pass

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loops managed by this instance.

        STUB: Full implementation would include:
        - Detailed metrics by phase
        - Success/failure rates
        - Average duration statistics
        - Blocked dependency analysis
        """
        loops_by_status = {}
        for status in LoopStatus:
            loops_by_status[status.name] = sum(
                1 for loop in self._loops.values()
                if loop.status == status
            )

        return {
            'total_loops': len(self._loops),
            'by_status': loops_by_status,
            'stuck_count': len(self.get_stuck_loops()),
            'blocked_count': len(self.get_blocked_loops()),
            'active_count': len(self.get_active_loops()),
        }


# =============================================================================
# STUB CLASSES FOR COMPLEX IMPLEMENTATIONS
# =============================================================================
#
# The following classes represent complex functionality that would require
# significant implementation effort. They are stubbed with detailed comments
# explaining what the full implementation would do.
# =============================================================================


class NestedLoopExecutor:
    """
    STUB: Executor for nested cognitive loops with automatic child management.

    Full Implementation Would:
    --------------------------
    1. Automatically detect when current work should spawn a child loop
       - Pattern: When a phase's complexity exceeds threshold
       - Pattern: When a QUESTION raises multiple independent sub-questions
       - Pattern: When PRODUCE needs multiple independent artifacts

    2. Manage child loop lifecycles
       - Start children when parent phase begins
       - Wait for children to complete before parent transitions
       - Aggregate child results into parent context
       - Handle child failures (retry, escalate, or partial success)

    3. Support parallel child execution
       - Spawn multiple children simultaneously for independent work
       - Coordinate shared resources (files, APIs, etc.)
       - Merge results when all children complete

    4. Provide observability
       - Visualize loop hierarchy (tree view)
       - Track time spent at each nesting level
       - Identify bottleneck children

    Key Algorithms Needed:
    ----------------------
    - Dependency detection: Which children depend on others?
    - Parallelization strategy: What can run concurrently?
    - Result aggregation: How to merge child outputs?
    - Failure propagation: When does child failure fail parent?

    Integration Points:
    -------------------
    - CognitiveLoopManager: For loop lifecycle
    - CrisisManager: For failure escalation
    - MetricsCollector: For timing and efficiency data
    - CollaborationManager: For parallel coordination
    """

    def __init__(self, manager: CognitiveLoopManager):
        """
        Initialize the nested loop executor.

        Args:
            manager: The CognitiveLoopManager to use for loop operations
        """
        self._manager = manager
        self._execution_stack: List[str] = []  # Stack of active loop IDs

    def execute_with_nesting(
        self,
        loop: CognitiveLoop,
        max_depth: int = 5
    ) -> TerminationReason:
        """
        STUB: Execute a loop with automatic nested loop handling.

        Args:
            loop: The loop to execute
            max_depth: Maximum nesting depth (prevents infinite recursion)

        Returns:
            Final termination reason

        Raises:
            RecursionError: If max_depth exceeded
        """
        # STUB: Would implement full execution logic here
        # For now, just record that execution was attempted
        loop.add_note(f"NestedLoopExecutor.execute_with_nesting called (max_depth={max_depth})")
        return TerminationReason.SUCCESS

    def should_spawn_child(self, context: PhaseContext) -> bool:
        """
        STUB: Determine if current context warrants a child loop.

        Full Implementation Would:
        - Analyze phase complexity (questions raised, decisions needed)
        - Check if work is decomposable into independent parts
        - Consider time remaining vs work estimated
        - Use ML model trained on historical data

        Returns:
            True if a child loop should be spawned
        """
        # STUB: Simple heuristic - too many questions raised
        return len(context.questions_raised) > 3

    def suggest_child_goals(self, context: PhaseContext) -> List[str]:
        """
        STUB: Suggest goals for potential child loops.

        Full Implementation Would:
        - Parse questions raised to identify themes
        - Group related work items
        - Prioritize by dependency order
        - Generate clear, actionable goal statements

        Returns:
            List of suggested goal strings for child loops
        """
        # STUB: Return questions as potential child goals
        return context.questions_raised[:3]


class LoopStateSerializer:
    """
    Serializer for persisting and restoring loop state.

    Supports full serialization of:
    - All loop attributes including contexts and transitions
    - Parent/child relationships
    - Complete phase history with notes, questions, decisions, artifacts
    - Full transition history with context snapshots
    - Proper datetime handling with ISO format
    - Enum serialization (status, phase, termination reason)

    File Format Design:
    -------------------
    ```json
    {
      "id": "abc123",
      "goal": "Implement authentication",
      "status": "ACTIVE",
      "current_phase": "PRODUCE",
      "created_at": "2025-12-19T10:00:00",
      "parent_id": null,
      "children": ["def456", "ghi789"],
      "phase_contexts": [...],
      "transitions": [...]
    }
    ```

    Integration Points:
    -------------------
    - KnowledgeTransferManager: For handoff generation
    - ML data collection: Training data for loop optimization
    - Session memory: Context for future sessions
    """

    def serialize_phase_context(self, context: PhaseContext) -> Dict[str, Any]:
        """
        Serialize a PhaseContext to dictionary.

        Args:
            context: The PhaseContext to serialize

        Returns:
            Dictionary representation
        """
        return {
            'phase': context.phase.value,
            'started_at': context.started_at.isoformat(),
            'iteration': context.iteration,
            'notes': context.notes.copy(),
            'questions_raised': context.questions_raised.copy(),
            'decisions_made': context.decisions_made.copy(),
            'artifacts_produced': context.artifacts_produced.copy(),
            'time_box_minutes': context.time_box_minutes,
            'ended_at': context.ended_at.isoformat() if context.ended_at else None,
            'duration_seconds': context.duration_seconds,
        }

    def deserialize_phase_context(self, data: Dict[str, Any]) -> PhaseContext:
        """
        Deserialize a PhaseContext from dictionary.

        Args:
            data: Dictionary with PhaseContext data

        Returns:
            Reconstructed PhaseContext
        """
        from datetime import datetime

        context = PhaseContext(
            phase=LoopPhase(data['phase']),
            started_at=datetime.fromisoformat(data['started_at']),
            iteration=data['iteration'],
            notes=data['notes'].copy(),
            questions_raised=data['questions_raised'].copy(),
            decisions_made=data['decisions_made'].copy(),
            artifacts_produced=data['artifacts_produced'].copy(),
            time_box_minutes=data['time_box_minutes'],
            ended_at=datetime.fromisoformat(data['ended_at']) if data.get('ended_at') else None,
            duration_seconds=data.get('duration_seconds'),
        )
        return context

    def serialize_loop_transition(self, transition: LoopTransition) -> Dict[str, Any]:
        """
        Serialize a LoopTransition to dictionary.

        Args:
            transition: The LoopTransition to serialize

        Returns:
            Dictionary representation
        """
        return {
            'from_phase': transition.from_phase.value if transition.from_phase else None,
            'to_phase': transition.to_phase.value,
            'timestamp': transition.timestamp.isoformat(),
            'reason': transition.reason,
            'context_snapshot': transition.context_snapshot.copy(),
        }

    def deserialize_loop_transition(self, data: Dict[str, Any]) -> LoopTransition:
        """
        Deserialize a LoopTransition from dictionary.

        Args:
            data: Dictionary with LoopTransition data

        Returns:
            Reconstructed LoopTransition
        """
        from datetime import datetime

        transition = LoopTransition(
            from_phase=LoopPhase(data['from_phase']) if data.get('from_phase') else None,
            to_phase=LoopPhase(data['to_phase']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            reason=data['reason'],
            context_snapshot=data['context_snapshot'].copy(),
        )
        return transition

    def serialize(self, loop: CognitiveLoop) -> str:
        """
        Serialize a loop to JSON string.

        Handles all loop attributes including:
        - Basic attributes (id, goal, status, phase)
        - Timestamps (created_at, completed_at)
        - Parent/child relationships
        - Full phase contexts with all data
        - Full transitions with context snapshots
        - Block reason and termination reason

        Args:
            loop: The loop to serialize

        Returns:
            JSON string representation
        """
        import json

        data = {
            'id': loop.id,
            'goal': loop.goal,
            'status': loop.status.name,
            'current_phase': loop.current_phase.value if loop.current_phase else None,
            'created_at': loop.created_at.isoformat(),
            'completed_at': loop.completed_at.isoformat() if loop.completed_at else None,
            'termination_reason': loop.termination_reason.value if loop.termination_reason else None,
            'parent_id': loop.parent_id,
            'child_ids': loop.child_ids.copy(),
            'block_reason': loop.block_reason,
            'phase_contexts': [
                self.serialize_phase_context(ctx) for ctx in loop.phase_contexts
            ],
            'transitions': [
                self.serialize_loop_transition(trans) for trans in loop.transitions
            ],
        }

        return json.dumps(data, indent=2)

    def deserialize(self, data: str) -> CognitiveLoop:
        """
        Deserialize a loop from JSON string.

        Reconstructs all loop state including:
        - All basic attributes
        - Full phase contexts with notes, questions, decisions, artifacts
        - Full transitions with context snapshots
        - Parent/child relationships
        - Block reason and termination reason

        Args:
            data: JSON string to deserialize

        Returns:
            Reconstructed CognitiveLoop with full state
        """
        import json
        from datetime import datetime

        obj = json.loads(data)

        # Create loop with basic attributes
        loop = CognitiveLoop(
            id=obj['id'],
            goal=obj['goal'],
            status=LoopStatus[obj['status']],
            parent_id=obj.get('parent_id'),
            child_ids=obj.get('child_ids', []).copy(),
        )

        # Restore timestamps
        loop.created_at = datetime.fromisoformat(obj['created_at'])
        if obj.get('completed_at'):
            loop.completed_at = datetime.fromisoformat(obj['completed_at'])

        # Restore current phase
        if obj.get('current_phase'):
            loop.current_phase = LoopPhase(obj['current_phase'])

        # Restore termination reason
        if obj.get('termination_reason'):
            loop.termination_reason = TerminationReason(obj['termination_reason'])

        # Restore block reason
        if obj.get('block_reason'):
            loop._block_reason = obj['block_reason']

        # Restore phase contexts
        loop.phase_contexts = [
            self.deserialize_phase_context(ctx_data)
            for ctx_data in obj.get('phase_contexts', [])
        ]

        # Restore transitions
        loop.transitions = [
            self.deserialize_loop_transition(trans_data)
            for trans_data in obj.get('transitions', [])
        ]

        return loop

    def serialize_manager(self, manager: CognitiveLoopManager) -> str:
        """
        Serialize entire CognitiveLoopManager state to JSON string.

        Args:
            manager: The CognitiveLoopManager to serialize

        Returns:
            JSON string representation of all loops
        """
        import json

        data = {
            'loops': {
                loop_id: json.loads(self.serialize(loop))
                for loop_id, loop in manager._loops.items()
            }
        }

        return json.dumps(data, indent=2)

    def deserialize_manager(self, data: str) -> CognitiveLoopManager:
        """
        Deserialize CognitiveLoopManager state from JSON string.

        Restores all loops with their complete state and rebuilds
        the manager's internal tracking structures.

        Args:
            data: JSON string with manager state

        Returns:
            Reconstructed CognitiveLoopManager with all loops
        """
        import json

        obj = json.loads(data)
        manager = CognitiveLoopManager()

        # Restore all loops
        for loop_id, loop_data in obj.get('loops', {}).items():
            loop = self.deserialize(json.dumps(loop_data))
            manager._loops[loop_id] = loop

        return manager

    def generate_handoff(self, loop: CognitiveLoop) -> str:
        """
        STUB: Generate a human-readable handoff document for a loop.

        Full Implementation Would:
        - Summarize current state and progress
        - List key decisions made and why
        - Identify open questions and blockers
        - Provide clear next steps
        - Include context that might get lost

        Returns:
            Markdown-formatted handoff document
        """
        # STUB: Generate basic handoff
        lines = [
            f"# Loop Handoff: {loop.goal}",
            "",
            f"**ID:** {loop.id}",
            f"**Status:** {loop.status.name}",
            f"**Current Phase:** {loop.current_phase.value if loop.current_phase else 'N/A'}",
            "",
            "## Progress Summary",
            f"- Phases completed: {len(loop.phase_contexts)}",
            f"- Transitions made: {len(loop.transitions)}",
            f"- Time elapsed: {loop.total_elapsed_minutes():.1f} minutes",
            "",
            "## Next Steps",
            "1. Review current phase context",
            "2. Check for blocking issues",
            "3. Continue with appropriate action",
        ]

        return "\n".join(lines)
