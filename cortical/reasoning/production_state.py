"""
Production State Machine: Managing Artifact Creation Lifecycle.

This module implements the production state machine from Part 5 of
docs/complex-reasoning-workflow.md. It tracks the lifecycle of
artifact creation during the PRODUCE phase of the QAPV loop.

State transitions:
    PLANNING → DRAFTING → REFINING → FINALIZING → COMPLETE
                  ↓           ↓           ↓
              BLOCKED ← → REWORK ← ──────┘
                  ↓           ↓
              ABANDONED ←─────┘

Design Philosophy:
    Production isn't just "writing code" - it's a structured process with
    checkpoints. Each state has clear entry/exit criteria and time expectations.
    This enables:
    - Progress visibility (where are we in the process?)
    - Quality gates (did we verify before finalizing?)
    - Recovery paths (how do we handle failures?)
    - Knowledge capture (what happened at each stage?)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid


class ProductionState(Enum):
    """
    States in the production lifecycle.

    From docs/complex-reasoning-workflow.md Part 5:
    - PLANNING: Designing what to build, not yet writing
    - DRAFTING: First pass, getting something working
    - REFINING: Improving quality, handling edge cases
    - FINALIZING: Documentation, cleanup, verification
    - COMPLETE: Ready for merge/deployment
    - BLOCKED: Waiting on external input/decision
    - REWORK: Verification failed, fixing issues
    - ABANDONED: Consciously stopped (with documentation)
    """
    PLANNING = auto()
    DRAFTING = auto()
    REFINING = auto()
    FINALIZING = auto()
    COMPLETE = auto()
    BLOCKED = auto()
    REWORK = auto()
    ABANDONED = auto()


# Valid state transitions (from_state -> [valid_to_states])
VALID_TRANSITIONS: Dict[ProductionState, Set[ProductionState]] = {
    ProductionState.PLANNING: {ProductionState.DRAFTING, ProductionState.ABANDONED},
    ProductionState.DRAFTING: {ProductionState.REFINING, ProductionState.BLOCKED, ProductionState.ABANDONED},
    ProductionState.REFINING: {ProductionState.FINALIZING, ProductionState.REWORK, ProductionState.BLOCKED},
    ProductionState.FINALIZING: {ProductionState.COMPLETE, ProductionState.REWORK, ProductionState.BLOCKED},
    ProductionState.COMPLETE: set(),  # Terminal state
    ProductionState.BLOCKED: {ProductionState.DRAFTING, ProductionState.REFINING, ProductionState.FINALIZING, ProductionState.ABANDONED},
    ProductionState.REWORK: {ProductionState.REFINING, ProductionState.BLOCKED, ProductionState.ABANDONED},
    ProductionState.ABANDONED: set(),  # Terminal state
}


@dataclass
class ProductionChunk:
    """
    A chunk of work within a production task.

    From docs/complex-reasoning-workflow.md:
    "Ideal chunk characteristics:
    - Takes 15-45 minutes of focused work
    - Produces a testable artifact
    - Has clear success criteria
    - Can be committed independently"

    Attributes:
        id: Unique identifier
        name: Short name for the chunk
        goal: What this chunk accomplishes
        inputs: Prior chunks/information this depends on
        outputs: Files/artifacts this will create
        verification: Checklist of verification steps
        time_estimate_minutes: Expected duration
        actual_minutes: Actual time spent (if complete)
        status: Current state
        notes: Developer notes during work
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    goal: str = ""
    inputs: List[str] = field(default_factory=list)  # Dependencies
    outputs: List[str] = field(default_factory=list)  # Files to create/modify
    verification: List[str] = field(default_factory=list)  # Checks to perform
    time_estimate_minutes: int = 30
    actual_minutes: float = 0.0
    status: ProductionState = ProductionState.PLANNING
    notes: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def start(self) -> None:
        """Mark chunk as started."""
        self.started_at = datetime.now()
        self.status = ProductionState.DRAFTING

    def complete(self) -> None:
        """Mark chunk as complete."""
        self.completed_at = datetime.now()
        self.status = ProductionState.COMPLETE
        if self.started_at:
            self.actual_minutes = (self.completed_at - self.started_at).total_seconds() / 60


@dataclass
class CommentMarker:
    """
    An in-progress comment marker for knowledge capture.

    From docs/complex-reasoning-workflow.md Part 5.3:
    "Types of in-progress comments:
    - THINKING: Why I'm doing it this way...
    - TODO: Need to handle edge case X
    - QUESTION: Is this the right abstraction?
    - NOTE: This pattern matches what we do in module Y
    - PERF: This is O(n²), acceptable for n < 1000
    - HACK: Workaround for issue #123, remove when fixed"

    These markers capture reasoning as it happens, not after the fact.
    """
    marker_type: str  # THINKING, TODO, QUESTION, NOTE, PERF, HACK
    content: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None

    @classmethod
    def thinking(cls, content: str, file_path: str = None, line: int = None) -> 'CommentMarker':
        """Create a THINKING marker (captures decision reasoning)."""
        return cls("THINKING", content, file_path, line)

    @classmethod
    def todo(cls, content: str, file_path: str = None, line: int = None) -> 'CommentMarker':
        """Create a TODO marker (captures known gaps)."""
        return cls("TODO", content, file_path, line)

    @classmethod
    def question(cls, content: str, file_path: str = None, line: int = None) -> 'CommentMarker':
        """Create a QUESTION marker (captures uncertainty)."""
        return cls("QUESTION", content, file_path, line)

    @classmethod
    def note(cls, content: str, file_path: str = None, line: int = None) -> 'CommentMarker':
        """Create a NOTE marker (captures cross-references)."""
        return cls("NOTE", content, file_path, line)

    @classmethod
    def perf(cls, content: str, file_path: str = None, line: int = None) -> 'CommentMarker':
        """Create a PERF marker (captures performance decisions)."""
        return cls("PERF", content, file_path, line)

    @classmethod
    def hack(cls, content: str, file_path: str = None, line: int = None) -> 'CommentMarker':
        """Create a HACK marker (captures technical debt)."""
        return cls("HACK", content, file_path, line)


@dataclass
class ProductionTask:
    """
    A production task tracking the full lifecycle of artifact creation.

    This is the main class for managing production state. It tracks:
    - Current state in the production lifecycle
    - Chunks of work within the task
    - Comment markers for knowledge capture
    - Transition history for audit trail
    - Files modified during production

    Example:
        >>> task = ProductionTask(goal="Implement user authentication")
        >>> task.transition_to(ProductionState.DRAFTING, "Design approved")
        >>> task.add_chunk(ProductionChunk(name="Auth service", goal="Create auth.py"))
        >>> task.add_marker(CommentMarker.thinking("Using OAuth for flexibility"))
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    description: str = ""
    state: ProductionState = ProductionState.PLANNING
    chunks: List[ProductionChunk] = field(default_factory=list)
    markers: List[CommentMarker] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Callbacks
    _on_state_change: Optional[Callable[['ProductionTask', ProductionState, ProductionState], None]] = None

    def transition_to(self, new_state: ProductionState, reason: str) -> None:
        """
        Transition to a new production state.

        Args:
            new_state: Target state
            reason: Why we're making this transition

        Raises:
            ValueError: If transition is not valid
        """
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            raise ValueError(
                f"Invalid transition: {self.state.name} -> {new_state.name}. "
                f"Valid targets: {[s.name for s in VALID_TRANSITIONS.get(self.state, set())]}"
            )

        old_state = self.state
        self.state = new_state

        self.transitions.append({
            'from': old_state.name,
            'to': new_state.name,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })

        if new_state in (ProductionState.COMPLETE, ProductionState.ABANDONED):
            self.completed_at = datetime.now()

        if self._on_state_change:
            self._on_state_change(self, old_state, new_state)

    def add_chunk(self, chunk: ProductionChunk) -> None:
        """Add a work chunk to this task."""
        self.chunks.append(chunk)

    def add_marker(self, marker: CommentMarker) -> None:
        """Add a comment marker."""
        self.markers.append(marker)

    def add_file(self, file_path: str) -> None:
        """Record a file as modified by this task."""
        if file_path not in self.files_modified:
            self.files_modified.append(file_path)

    def get_unresolved_markers(self) -> List[CommentMarker]:
        """Get all unresolved comment markers."""
        return [m for m in self.markers if not m.resolved]

    def get_markers_by_type(self, marker_type: str) -> List[CommentMarker]:
        """Get markers of a specific type."""
        return [m for m in self.markers if m.marker_type == marker_type]

    def can_finalize(self) -> Tuple[bool, List[str]]:
        """
        Check if task can move to FINALIZING state.

        Returns:
            (can_finalize, list of blocking issues)
        """
        issues = []

        # Check for blocking markers
        todos = self.get_markers_by_type("TODO")
        unresolved_todos = [t for t in todos if not t.resolved]
        if unresolved_todos:
            issues.append(f"{len(unresolved_todos)} unresolved TODOs")

        questions = self.get_markers_by_type("QUESTION")
        unresolved_questions = [q for q in questions if not q.resolved]
        if unresolved_questions:
            issues.append(f"{len(unresolved_questions)} unresolved QUESTIONs")

        # Check chunk completion
        incomplete_chunks = [c for c in self.chunks if c.status != ProductionState.COMPLETE]
        if incomplete_chunks:
            issues.append(f"{len(incomplete_chunks)} incomplete chunks")

        return len(issues) == 0, issues

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this production task."""
        return {
            'id': self.id,
            'goal': self.goal,
            'state': self.state.name,
            'chunks_total': len(self.chunks),
            'chunks_complete': sum(1 for c in self.chunks if c.status == ProductionState.COMPLETE),
            'markers_total': len(self.markers),
            'markers_unresolved': len(self.get_unresolved_markers()),
            'files_modified': len(self.files_modified),
            'transitions': len(self.transitions),
            'duration_minutes': self._get_duration_minutes(),
        }

    def _get_duration_minutes(self) -> float:
        """Calculate total duration in minutes."""
        end = self.completed_at or datetime.now()
        return (end - self.created_at).total_seconds() / 60


class ProductionManager:
    """
    Manager for production tasks across a session.

    Provides:
    - Task lifecycle management
    - Aggregate metrics across tasks
    - Pattern detection (common issues, bottlenecks)
    - Integration with verification system
    """

    def __init__(self):
        """Initialize the production manager."""
        self._tasks: Dict[str, ProductionTask] = {}
        self._state_change_handlers: List[Callable[[ProductionTask, ProductionState, ProductionState], None]] = []

    def create_task(self, goal: str, description: str = "") -> ProductionTask:
        """
        Create a new production task.

        Args:
            goal: What this task accomplishes
            description: Detailed description

        Returns:
            New ProductionTask instance
        """
        task = ProductionTask(
            goal=goal,
            description=description,
            _on_state_change=self._handle_state_change,
        )
        self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Optional[ProductionTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_tasks_in_state(self, state: ProductionState) -> List[ProductionTask]:
        """Get all tasks in a specific state."""
        return [t for t in self._tasks.values() if t.state == state]

    def get_blocked_tasks(self) -> List[ProductionTask]:
        """Get all blocked tasks."""
        return self.get_tasks_in_state(ProductionState.BLOCKED)

    def get_in_progress_tasks(self) -> List[ProductionTask]:
        """Get all tasks that are in progress (not complete/abandoned)."""
        terminal_states = {ProductionState.COMPLETE, ProductionState.ABANDONED}
        return [t for t in self._tasks.values() if t.state not in terminal_states]

    def register_state_change_handler(
        self,
        handler: Callable[[ProductionTask, ProductionState, ProductionState], None]
    ) -> None:
        """Register a handler for state changes."""
        self._state_change_handlers.append(handler)

    def _handle_state_change(
        self,
        task: ProductionTask,
        old_state: ProductionState,
        new_state: ProductionState
    ) -> None:
        """Internal handler for state changes."""
        for handler in self._state_change_handlers:
            try:
                handler(task, old_state, new_state)
            except Exception:
                pass  # Don't fail on handler errors

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary of all tasks."""
        by_state = {}
        for state in ProductionState:
            by_state[state.name] = len(self.get_tasks_in_state(state))

        total_markers = sum(len(t.markers) for t in self._tasks.values())
        unresolved_markers = sum(len(t.get_unresolved_markers()) for t in self._tasks.values())

        return {
            'total_tasks': len(self._tasks),
            'by_state': by_state,
            'in_progress': len(self.get_in_progress_tasks()),
            'blocked': len(self.get_blocked_tasks()),
            'total_markers': total_markers,
            'unresolved_markers': unresolved_markers,
        }


# =============================================================================
# STUB CLASSES FOR COMPLEX IMPLEMENTATIONS
# =============================================================================


class ChunkPlanner:
    """
    STUB: Automatic chunk planning based on task complexity.

    Full Implementation Would:
    --------------------------
    1. Analyze task description to identify logical work units
       - Use NLP to extract action items
       - Identify dependencies between items
       - Group related items into chunks

    2. Estimate time for each chunk
       - Use historical data from similar tasks
       - Consider complexity indicators (files touched, LOC)
       - Adjust for developer experience level

    3. Suggest optimal chunk ordering
       - Dependency-aware topological sort
       - Front-load risky/uncertain chunks
       - Parallelize where possible

    4. Adapt chunks as work progresses
       - Detect when estimates are off
       - Suggest re-chunking for too-large chunks
       - Merge too-small chunks

    Key Algorithms:
    ---------------
    - Task decomposition (dependency graph construction)
    - Time estimation (historical similarity matching)
    - Topological sorting with risk weighting
    - Dynamic re-planning based on actual progress
    """

    def plan_chunks(self, task: ProductionTask) -> List[ProductionChunk]:
        """
        STUB: Plan chunks for a production task.

        Args:
            task: The task to plan chunks for

        Returns:
            List of suggested ProductionChunk instances
        """
        # STUB: Generate placeholder chunks based on goal length
        # Full impl would use NLP and historical data
        chunks = []
        for i in range(3):  # Default to 3 chunks
            chunks.append(ProductionChunk(
                name=f"Chunk {i+1}",
                goal=f"Part {i+1} of: {task.goal[:50]}...",
                time_estimate_minutes=30,
            ))
        return chunks

    def replan(self, task: ProductionTask) -> List[ProductionChunk]:
        """
        STUB: Re-plan chunks based on actual progress.

        Full Implementation Would:
        - Analyze completed chunks vs estimates
        - Identify remaining work
        - Generate new chunk plan for remaining work
        - Preserve completed chunk history
        """
        # STUB: Return existing chunks
        return task.chunks


class CommentCleaner:
    """
    STUB: Automated comment cleanup after production.

    Full Implementation Would:
    --------------------------
    From docs/complex-reasoning-workflow.md Part 5.3:
    "Post-production comment cleanup:
    - THINKING → Usually remove or convert to doc comment
    - TODO → Convert to task if not addressed, or remove if done
    - QUESTION → Resolve and remove, or convert to doc explaining decision
    - NOTE → Keep if adds value, remove if obvious
    - PERF → Keep, these are valuable
    - HACK → Keep until resolved, reference task"

    1. Scan files for in-progress comment markers
    2. For each marker, suggest appropriate action:
       - Remove (resolved, no longer relevant)
       - Keep (valuable context)
       - Convert (change to standard doc comment)
       - Escalate (create task for unresolved issues)

    3. Apply suggestions (with user approval)
    4. Track which comments were cleaned for audit

    Integration Points:
    -------------------
    - Task system: Create tasks for unresolved TODOs
    - Documentation: Convert valuable comments to docs
    - Git: Track comment changes in commits
    """

    def scan_file(self, file_path: str) -> List[CommentMarker]:
        """
        STUB: Scan a file for comment markers.

        Args:
            file_path: Path to file to scan

        Returns:
            List of CommentMarker instances found
        """
        # STUB: Would parse file and extract markers
        return []

    def suggest_cleanup(self, marker: CommentMarker) -> Dict[str, Any]:
        """
        STUB: Suggest cleanup action for a marker.

        Returns:
            {'action': 'remove'|'keep'|'convert'|'escalate', 'reason': str}
        """
        # STUB: Simple heuristic by marker type
        cleanup_rules = {
            'THINKING': {'action': 'remove', 'reason': 'Reasoning captured elsewhere'},
            'TODO': {'action': 'escalate', 'reason': 'Create task for unresolved TODO'},
            'QUESTION': {'action': 'convert', 'reason': 'Document the decision made'},
            'NOTE': {'action': 'keep', 'reason': 'Cross-reference is valuable'},
            'PERF': {'action': 'keep', 'reason': 'Performance context is valuable'},
            'HACK': {'action': 'keep', 'reason': 'Technical debt must be tracked'},
        }
        return cleanup_rules.get(marker.marker_type, {'action': 'keep', 'reason': 'Unknown type'})


class ProductionMetrics:
    """
    STUB: Metrics collection for production analysis.

    Full Implementation Would:
    --------------------------
    1. Track timing metrics:
       - Time in each state
       - Time per chunk
       - Actual vs estimated time
       - Rework cycles

    2. Track quality metrics:
       - Markers per task
       - Resolution rate
       - Chunk completion rate
       - Files per task

    3. Provide analytics:
       - Average production time by task type
       - Common bottleneck states
       - Estimation accuracy over time
       - Correlation: markers vs rework

    4. Generate reports:
       - Session summary
       - Trend analysis
       - Recommendations for improvement
    """

    def record_state_transition(
        self,
        task: ProductionTask,
        from_state: ProductionState,
        to_state: ProductionState
    ) -> None:
        """STUB: Record a state transition for metrics."""
        pass  # Would store in time-series database

    def get_average_time_in_state(self, state: ProductionState) -> float:
        """STUB: Get average time spent in a state (minutes)."""
        # Would calculate from historical data
        default_times = {
            ProductionState.PLANNING: 15,
            ProductionState.DRAFTING: 45,
            ProductionState.REFINING: 30,
            ProductionState.FINALIZING: 20,
        }
        return default_times.get(state, 0)

    def get_estimation_accuracy(self) -> float:
        """STUB: Get accuracy of time estimates (actual/estimated ratio)."""
        # Would calculate from completed chunks
        return 0.85  # Placeholder
