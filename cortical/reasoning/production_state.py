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
import os
import re
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
            except Exception:  # noqa: S110
                # Broad exception catch is intentional: handlers are user-provided
                # callbacks and we don't know what they might raise. We must not
                # let a failing handler break state transition tracking.
                pass

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
    Automatic chunk planning based on task complexity.

    Analyzes task descriptions to:
    1. Extract action items from goal text
    2. Create focused work chunks (15-45 minutes each)
    3. Estimate time based on complexity indicators
    4. Detect dependencies between chunks
    5. Suggest parallel execution opportunities

    Key Algorithms:
    ---------------
    - Pattern-based action extraction (regex + verb detection)
    - Complexity-based time estimation
    - Dependency detection via keyword matching
    - Progress-aware re-planning
    """

    def plan_chunks(self, task: ProductionTask) -> List[ProductionChunk]:
        """
        Plan chunks for a production task.

        Analyzes the task goal to identify:
        - Numbered lists (1., 2., etc.)
        - Bullet points (-, *, •)
        - Verb phrases (implement X, test Y, refactor Z)

        Args:
            task: The task to plan chunks for

        Returns:
            List of ProductionChunk instances with time estimates and dependencies
        """
        import re

        # Extract action items from goal
        action_items = self._extract_action_items(task.goal)

        # If no action items found, create a single chunk
        if not action_items:
            return [ProductionChunk(
                name="Main task",
                goal=task.goal,
                time_estimate_minutes=30,
            )]

        # Create chunks from action items
        chunks = []
        for i, action in enumerate(action_items):
            # Extract files mentioned in action (look for .py, .md, etc.)
            files = re.findall(r'[\w/]+\.(?:py|md|txt|json|yaml|yml|js|ts|jsx|tsx)', action)

            chunk = ProductionChunk(
                name=f"Step {i+1}: {action[:30]}..." if len(action) > 30 else f"Step {i+1}: {action}",
                goal=action,
                outputs=files,
                time_estimate_minutes=self._estimate_chunk_time(action, files),
            )
            chunks.append(chunk)

        # Detect dependencies between chunks
        self._detect_dependencies(chunks)

        return chunks

    def _extract_action_items(self, goal: str) -> List[str]:
        """
        Extract action items from goal text.

        Looks for:
        - Numbered lists: "1. Do X", "2. Do Y"
        - Bullet points: "- Do X", "* Do Y"
        - Verb phrases: "implement X", "test Y", "refactor Z"

        Args:
            goal: The task goal text

        Returns:
            List of action item strings
        """
        import re

        items = []

        # Pattern 1: Numbered lists (1., 2., etc.)
        numbered = re.findall(r'(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.]|\n\s*[*•-]|$)', goal, re.MULTILINE | re.DOTALL)
        items.extend([item.strip() for item in numbered if item.strip()])

        # Pattern 2: Bullet points (-, *, •)
        bullets = re.findall(r'(?:^|\n)\s*[*•-]\s*(.+?)(?=\n\s*[*•-]|\n\s*\d+[.]|\n\n|$)', goal, re.MULTILINE | re.DOTALL)
        items.extend([item.strip() for item in bullets if item.strip()])

        # Pattern 3: Verb phrases (if no lists found)
        if not items:
            # Common action verbs in programming tasks
            verbs = [
                'implement', 'add', 'create', 'build', 'write', 'design',
                'test', 'verify', 'validate', 'check',
                'refactor', 'redesign', 'restructure', 'optimize',
                'fix', 'debug', 'resolve', 'handle',
                'update', 'modify', 'change', 'edit',
                'document', 'explain', 'describe',
                'remove', 'delete', 'clean'
            ]

            # Split on sentence boundaries and look for verb phrases
            sentences = re.split(r'[.!?]+', goal)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check if sentence starts with a verb
                lower = sentence.lower()
                for verb in verbs:
                    if lower.startswith(verb) or f' {verb} ' in lower:
                        items.append(sentence)
                        break

        # Clean up items (remove newlines, extra spaces)
        items = [re.sub(r'\s+', ' ', item).strip() for item in items]

        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                unique_items.append(item)

        return unique_items

    def _estimate_chunk_time(self, action: str, files: List[str]) -> int:
        """
        Estimate time for a chunk based on complexity indicators.

        Heuristics:
        - Base time: 15 minutes per action item
        - +10 minutes per file mentioned
        - +20 minutes for refactor/redesign tasks
        - +15 minutes for test/verify tasks
        - +10 minutes for multiple related operations
        - Cap at 60 minutes per chunk

        Args:
            action: The action item text
            files: List of files mentioned in action

        Returns:
            Estimated time in minutes (15-60)
        """
        base_time = 15

        # Add time per file
        base_time += len(files) * 10

        # Check for complexity keywords
        lower = action.lower()

        # Refactoring/redesign is complex
        if any(word in lower for word in ['refactor', 'redesign', 'restructure']):
            base_time += 20

        # Testing requires setup and verification
        if any(word in lower for word in ['test', 'verify', 'validate', 'check']):
            base_time += 15

        # Multiple operations (and, then, also)
        if any(word in lower for word in [' and ', ' then ', ' also ']):
            base_time += 10

        # Long descriptions suggest complexity
        if len(action) > 100:
            base_time += 10

        # Cap at 60 minutes (if longer, should be broken down)
        return min(base_time, 60)

    def _detect_dependencies(self, chunks: List[ProductionChunk]) -> None:
        """
        Detect dependencies between chunks.

        Rules:
        - Tests depend on implementations
        - Documentation depends on implementation
        - If chunk B mentions output of chunk A, B depends on A
        - Refactoring depends on having tests

        Modifies chunks in place by setting the 'inputs' field.

        Args:
            chunks: List of chunks to analyze
        """
        import re

        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.goal.lower()

            # Check previous chunks for dependencies
            for j, prev_chunk in enumerate(chunks[:i]):
                prev_lower = prev_chunk.goal.lower()

                # Rule 1: Tests depend on implementations
                if 'test' in chunk_lower and 'implement' in prev_lower:
                    chunk.inputs.append(f"{prev_chunk.name}")

                # Rule 2: Documentation depends on implementation
                if any(word in chunk_lower for word in ['document', 'doc', 'readme']) and \
                   any(word in prev_lower for word in ['implement', 'add', 'create', 'build']):
                    chunk.inputs.append(f"{prev_chunk.name}")

                # Rule 3: Refactoring depends on having tests
                if 'refactor' in chunk_lower and 'test' in prev_lower:
                    chunk.inputs.append(f"{prev_chunk.name}")

                # Rule 4: If chunk mentions files from previous chunk
                if prev_chunk.outputs:
                    for output_file in prev_chunk.outputs:
                        if output_file in chunk.goal or output_file in chunk.outputs:
                            chunk.inputs.append(f"{prev_chunk.name}")
                            break

                # Rule 5: Sequential numbered steps have implicit dependency
                # (Step 2 depends on Step 1)
                if i == j + 1 and 'step' in chunk.name.lower() and 'step' in prev_chunk.name.lower():
                    # Only add if no other dependencies found (avoid redundant deps)
                    if not chunk.inputs:
                        chunk.inputs.append(f"{prev_chunk.name}")

    def replan(self, task: ProductionTask) -> List[ProductionChunk]:
        """
        Re-plan chunks based on actual progress.

        Strategy:
        1. Keep all completed chunks (preserve history)
        2. Identify incomplete chunks
        3. Re-estimate remaining time based on progress
        4. Suggest new chunks if scope has changed

        Args:
            task: The task to re-plan

        Returns:
            Updated list of chunks
        """
        completed = [c for c in task.chunks if c.status == ProductionState.COMPLETE]
        incomplete = [c for c in task.chunks if c.status != ProductionState.COMPLETE]

        # If no incomplete chunks, we're done
        if not incomplete:
            return task.chunks

        # Recalculate estimates for incomplete chunks
        for chunk in incomplete:
            # If chunk has been started, adjust estimate based on progress
            if chunk.started_at and chunk.status == ProductionState.DRAFTING:
                elapsed = (datetime.now() - chunk.started_at).total_seconds() / 60

                # If we've spent more than the estimate, increase it
                if elapsed > chunk.time_estimate_minutes:
                    chunk.time_estimate_minutes = int(elapsed * 1.3)  # Add 30% buffer

        # Return updated chunk list
        return completed + incomplete

    def suggest_parallel_chunks(self, chunks: List[ProductionChunk]) -> List[List[ProductionChunk]]:
        """
        Suggest which chunks can be executed in parallel.

        Chunks can run in parallel if:
        - They have no dependencies on each other
        - They don't modify the same files

        Args:
            chunks: List of chunks to analyze

        Returns:
            List of groups, where each group contains chunks that can run in parallel
        """
        # Build dependency graph
        parallel_groups = []
        processed = set()

        for chunk in chunks:
            if chunk.id in processed:
                continue

            # Start a new parallel group with this chunk
            group = [chunk]
            processed.add(chunk.id)

            # Find other chunks that can run in parallel with this one
            for other in chunks:
                if other.id in processed:
                    continue

                # Check if they can run in parallel
                can_parallelize = True

                # Check 1: No dependency on each other
                if chunk.name in other.inputs or other.name in chunk.inputs:
                    can_parallelize = False

                # Check 2: Don't modify same files
                chunk_files = set(chunk.outputs)
                other_files = set(other.outputs)
                if chunk_files & other_files:  # Intersection
                    can_parallelize = False

                # Check 3: Other doesn't depend on anything in current group
                for group_chunk in group:
                    if group_chunk.name in other.inputs:
                        can_parallelize = False
                        break

                if can_parallelize:
                    group.append(other)
                    processed.add(other.id)

            parallel_groups.append(group)

        return parallel_groups



class CommentCleaner:
    """
    Automated comment cleanup after production.

    From docs/complex-reasoning-workflow.md Part 5.3:
    "Post-production comment cleanup:
    - THINKING → Usually remove or convert to doc comment
    - TODO → Convert to task if not addressed, or remove if done
    - QUESTION → Resolve and remove, or convert to doc explaining decision
    - NOTE → Keep if adds value, remove if obvious
    - PERF → Keep, these are valuable
    - HACK → Keep until resolved, reference task"

    Scans files for in-progress comment markers and suggests appropriate actions:
    - Remove (resolved, no longer relevant)
    - Keep (valuable context)
    - Convert (change to standard doc comment)
    - Escalate (create task for unresolved issues)

    Integration Points:
    -------------------
    - Task system: Create tasks for unresolved TODOs
    - Documentation: Convert valuable comments to docs
    - Git: Track comment changes in commits
    """

    # Marker patterns: # MARKER: content or # MARKER content
    MARKER_PATTERN = re.compile(
        r'^\s*#\s*(THINKING|TODO|QUESTION|NOTE|PERF|HACK)\s*:?\s*(.*)$',
        re.IGNORECASE
    )

    # Keywords that suggest a comment should be kept
    KEEP_KEYWORDS = {
        'cross-reference', 'see also', 'related to', 'matches',
        'performance', 'complexity', 'O(', 'edge case', 'workaround',
        'issue', 'bug', 'ticket', 'task'
    }

    # Keywords that suggest a comment can be removed
    REMOVE_KEYWORDS = {
        'obvious', 'self-explanatory', 'debug', 'temporary',
        'testing', 'experiment', 'try'
    }

    def scan_file(self, file_path: str) -> List[CommentMarker]:
        """
        Scan a file for comment markers.

        Args:
            file_path: Path to file to scan

        Returns:
            List of CommentMarker instances found

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        markers = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    marker = self._parse_marker(line, line_num)
                    if marker:
                        marker.file_path = file_path
                        markers.append(marker)
        except IOError as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        return markers

    def _parse_marker(self, line: str, line_num: int) -> Optional[CommentMarker]:
        """
        Parse a line to extract comment marker.

        Args:
            line: Line of text to parse
            line_num: Line number in file

        Returns:
            CommentMarker if found, None otherwise
        """
        match = self.MARKER_PATTERN.match(line)
        if not match:
            return None

        marker_type = match.group(1).upper()
        content = match.group(2).strip()

        return CommentMarker(
            marker_type=marker_type,
            content=content,
            line_number=line_num
        )

    def suggest_cleanup(self, marker: CommentMarker) -> Dict[str, Any]:
        """
        Suggest cleanup action for a marker with smart rules.

        Args:
            marker: The comment marker to analyze

        Returns:
            Dict with 'action' ('remove'|'keep'|'convert'|'escalate') and 'reason'
        """
        content_lower = marker.content.lower()

        # THINKING: remove unless it explains non-obvious logic
        if marker.marker_type == 'THINKING':
            if any(kw in content_lower for kw in self.KEEP_KEYWORDS):
                return {
                    'action': 'keep',
                    'reason': 'Explains non-obvious logic or design decision'
                }
            return {
                'action': 'remove',
                'reason': 'Reasoning should be captured in docs or commit messages'
            }

        # TODO: escalate if not addressed, check if task exists
        elif marker.marker_type == 'TODO':
            # Check if it references a task/issue
            if re.search(r'(task|issue|ticket|#)\s*\d+', content_lower):
                return {
                    'action': 'keep',
                    'reason': 'References existing task, keep for tracking'
                }
            # Otherwise escalate to create a task
            return {
                'action': 'escalate',
                'reason': 'Create task for unresolved TODO'
            }

        # QUESTION: convert to doc if resolved, escalate if not
        elif marker.marker_type == 'QUESTION':
            # Check if it contains resolution indicators
            if any(word in content_lower for word in ['resolved', 'answered', 'yes', 'no', 'decided']):
                return {
                    'action': 'convert',
                    'reason': 'Convert resolved question to documentation'
                }
            return {
                'action': 'escalate',
                'reason': 'Unresolved question needs attention'
            }

        # NOTE: keep if cross-reference, remove if obvious
        elif marker.marker_type == 'NOTE':
            if any(kw in content_lower for kw in self.KEEP_KEYWORDS):
                return {
                    'action': 'keep',
                    'reason': 'Cross-reference or valuable context'
                }
            if any(kw in content_lower for kw in self.REMOVE_KEYWORDS):
                return {
                    'action': 'remove',
                    'reason': 'Obvious or temporary note'
                }
            return {
                'action': 'keep',
                'reason': 'NOTE provides context, keep by default'
            }

        # PERF: always keep
        elif marker.marker_type == 'PERF':
            return {
                'action': 'keep',
                'reason': 'Performance context is valuable'
            }

        # HACK: always keep, suggest task creation
        elif marker.marker_type == 'HACK':
            if re.search(r'(task|issue|ticket|#)\s*\d+', content_lower):
                return {
                    'action': 'keep',
                    'reason': 'Technical debt tracked via task reference'
                }
            return {
                'action': 'escalate',
                'reason': 'Technical debt should be tracked as a task'
            }

        # Unknown marker type
        return {
            'action': 'keep',
            'reason': 'Unknown marker type, keep for manual review'
        }

    def apply_cleanup(self, file_path: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply cleanup actions to a file.

        Args:
            file_path: Path to file to modify
            actions: List of dicts with 'line_number', 'action', and optional 'replacement'

        Returns:
            Dict with 'modified' (bool), 'removed' (int), 'converted' (int), 'kept' (int)

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read/written
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except IOError as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        # Build action map by line number
        action_map = {a['line_number']: a for a in actions}

        # Apply actions
        new_lines = []
        stats = {'removed': 0, 'converted': 0, 'kept': 0}

        for line_num, line in enumerate(lines, 1):
            if line_num in action_map:
                action = action_map[line_num]['action']

                if action == 'remove':
                    stats['removed'] += 1
                    continue  # Skip this line

                elif action == 'convert':
                    # Convert to standard comment (remove marker prefix)
                    replacement = action_map[line_num].get('replacement')
                    if replacement:
                        new_lines.append(replacement)
                    else:
                        # Default: just remove the marker prefix
                        match = self.MARKER_PATTERN.match(line)
                        if match:
                            content = match.group(2).strip()
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(' ' * indent + f"# {content}\n")
                        else:
                            new_lines.append(line)
                    stats['converted'] += 1

                elif action == 'keep':
                    new_lines.append(line)
                    stats['kept'] += 1

                else:  # escalate or unknown
                    new_lines.append(line)
                    stats['kept'] += 1
            else:
                new_lines.append(line)

        # Check if file was modified
        modified = new_lines != lines

        # Write file if modified
        if modified:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
            except IOError as e:
                raise IOError(f"Failed to write file {file_path}: {e}")

        return {'modified': modified, **stats}

    def scan_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, List[CommentMarker]]:
        """
        Recursively scan directory for comment markers.

        Args:
            dir_path: Directory to scan
            extensions: List of file extensions to scan (default: ['.py'])

        Returns:
            Dict mapping file paths to lists of markers found

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not os.path.isdir(dir_path):
            raise ValueError(f"Not a directory: {dir_path}")

        if extensions is None:
            extensions = ['.py']

        results = {}

        for root, dirs, files in os.walk(dir_path):
            # Skip hidden directories and common excludes
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]

            for filename in files:
                # Check extension
                if not any(filename.endswith(ext) for ext in extensions):
                    continue

                file_path = os.path.join(root, filename)
                try:
                    markers = self.scan_file(file_path)
                    if markers:
                        results[file_path] = markers
                except (IOError, UnicodeDecodeError):
                    # Skip files that can't be read
                    continue

        return results

    def generate_cleanup_report(self, markers: Dict[str, List[CommentMarker]]) -> str:
        """
        Generate a markdown report of all markers found.

        Args:
            markers: Dict mapping file paths to lists of markers

        Returns:
            Markdown-formatted report string
        """
        if not markers:
            return "# Comment Cleanup Report\n\nNo markers found.\n"

        # Count markers by type
        type_counts = {}
        total_count = 0

        for file_markers in markers.values():
            for marker in file_markers:
                type_counts[marker.marker_type] = type_counts.get(marker.marker_type, 0) + 1
                total_count += 1

        # Build report
        lines = [
            "# Comment Cleanup Report",
            "",
            f"**Total markers found:** {total_count}",
            "",
            "## Summary by Type",
            ""
        ]

        for marker_type in sorted(type_counts.keys()):
            count = type_counts[marker_type]
            lines.append(f"- **{marker_type}:** {count}")

        lines.extend(["", "## Markers by File", ""])

        # Group by file
        for file_path in sorted(markers.keys()):
            file_markers = markers[file_path]
            lines.append(f"### `{file_path}`")
            lines.append("")

            # Group by marker type within file
            by_type = {}
            for marker in file_markers:
                if marker.marker_type not in by_type:
                    by_type[marker.marker_type] = []
                by_type[marker.marker_type].append(marker)

            for marker_type in sorted(by_type.keys()):
                lines.append(f"**{marker_type}** ({len(by_type[marker_type])} found):")
                lines.append("")

                for marker in by_type[marker_type]:
                    suggestion = self.suggest_cleanup(marker)
                    action = suggestion['action']
                    reason = suggestion['reason']

                    lines.append(f"- Line {marker.line_number}: `{marker.content}`")
                    lines.append(f"  - **Suggested action:** {action}")
                    lines.append(f"  - **Reason:** {reason}")
                    lines.append("")

        return "\n".join(lines)


class ProductionMetrics:
    """
    Metrics collection for production analysis.

    Tracks timing and quality metrics across production tasks:
    - Time in each state per task
    - Chunk completion times vs estimates
    - State transition patterns
    - Estimation accuracy over time

    Example:
        >>> metrics = ProductionMetrics()
        >>> metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        >>> metrics.record_chunk_start(chunk, estimated_minutes=30)
        >>> metrics.record_chunk_complete(chunk)
        >>> accuracy = metrics.get_estimation_accuracy()
    """

    def __init__(self):
        """Initialize metrics tracking."""
        # State transitions: {task_id, from_state, to_state, timestamp, duration_in_previous_state}
        self._state_transitions: List[Dict[str, Any]] = []

        # Chunk timings: chunk_id -> {start, end, estimated, actual, status}
        self._chunk_timings: Dict[str, Dict[str, Any]] = {}

        # Task timings: task_id -> {state -> [durations_in_minutes]}
        self._task_timings: Dict[str, Dict[str, List[float]]] = {}

    def record_state_transition(
        self,
        task: ProductionTask,
        from_state: ProductionState,
        to_state: ProductionState
    ) -> None:
        """
        Record a state transition for metrics.

        Args:
            task: The task that transitioned
            from_state: Previous state
            to_state: New state
        """
        timestamp = datetime.now()

        # Calculate duration in previous state
        duration_minutes = 0.0
        if self._state_transitions:
            # Find last transition for this task
            last_transition = None
            for trans in reversed(self._state_transitions):
                if trans['task_id'] == task.id:
                    last_transition = trans
                    break

            if last_transition:
                # Calculate time since last transition
                last_timestamp = datetime.fromisoformat(last_transition['timestamp'])
                duration_minutes = (timestamp - last_timestamp).total_seconds() / 60

        # Record transition
        self._state_transitions.append({
            'task_id': task.id,
            'from_state': from_state.name,
            'to_state': to_state.name,
            'timestamp': timestamp.isoformat(),
            'duration_in_previous_state': duration_minutes,
        })

        # Update task timings
        if task.id not in self._task_timings:
            self._task_timings[task.id] = {}

        state_name = from_state.name
        if state_name not in self._task_timings[task.id]:
            self._task_timings[task.id][state_name] = []

        if duration_minutes > 0:  # Only record if we have a valid duration
            self._task_timings[task.id][state_name].append(duration_minutes)

    def record_chunk_start(self, chunk: ProductionChunk, estimated_minutes: int = None) -> None:
        """
        Record the start of a chunk.

        Args:
            chunk: The chunk being started
            estimated_minutes: Estimated time (uses chunk.time_estimate_minutes if None)
        """
        if estimated_minutes is None:
            estimated_minutes = chunk.time_estimate_minutes

        self._chunk_timings[chunk.id] = {
            'start': datetime.now().isoformat(),
            'end': None,
            'estimated_minutes': estimated_minutes,
            'actual_minutes': None,
            'status': 'in_progress',
        }

    def record_chunk_complete(self, chunk: ProductionChunk) -> None:
        """
        Record the completion of a chunk.

        Args:
            chunk: The chunk being completed
        """
        if chunk.id not in self._chunk_timings:
            # Chunk was never started via record_chunk_start, create entry
            if chunk.started_at:
                self._chunk_timings[chunk.id] = {
                    'start': chunk.started_at.isoformat(),
                    'end': None,
                    'estimated_minutes': chunk.time_estimate_minutes,
                    'actual_minutes': None,
                    'status': 'in_progress',
                }
            else:
                # No timing data available
                return

        timing = self._chunk_timings[chunk.id]
        timing['end'] = datetime.now().isoformat()
        timing['status'] = 'complete'

        # Calculate actual duration
        start_time = datetime.fromisoformat(timing['start'])
        end_time = datetime.fromisoformat(timing['end'])
        timing['actual_minutes'] = (end_time - start_time).total_seconds() / 60

    def get_average_time_in_state(self, state: ProductionState) -> float:
        """
        Get average time spent in a state across all tasks (minutes).

        Args:
            state: The state to calculate average for

        Returns:
            Average minutes spent in state, or 0 if no data
        """
        state_name = state.name
        all_durations = []

        for task_id, states in self._task_timings.items():
            if state_name in states:
                all_durations.extend(states[state_name])

        if not all_durations:
            return 0.0

        return sum(all_durations) / len(all_durations)

    def get_estimation_accuracy(self) -> float:
        """
        Get accuracy of time estimates (actual/estimated ratio).

        Returns:
            Ratio of actual to estimated time:
            - 1.0 = perfect accuracy
            - < 1.0 = faster than estimated
            - > 1.0 = slower than estimated
            - 0.0 = no completed chunks
        """
        completed_chunks = [
            timing for timing in self._chunk_timings.values()
            if timing['status'] == 'complete' and timing['actual_minutes'] is not None
        ]

        if not completed_chunks:
            return 0.0

        total_actual = sum(c['actual_minutes'] for c in completed_chunks)
        total_estimated = sum(c['estimated_minutes'] for c in completed_chunks)

        if total_estimated == 0:
            return 0.0

        return total_actual / total_estimated

    def get_time_in_state_distribution(self) -> Dict[str, float]:
        """
        Get average time distribution across all states.

        Returns:
            Dictionary mapping state name to average minutes
        """
        distribution = {}

        for state in ProductionState:
            avg_time = self.get_average_time_in_state(state)
            if avg_time > 0:
                distribution[state.name] = avg_time

        return distribution

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary with:
            - total_transitions: Number of state transitions recorded
            - average_accuracy: Estimation accuracy ratio
            - time_distribution: Average time per state
            - chunks_completed: Number of completed chunks
            - chunks_in_progress: Number of in-progress chunks
            - tasks_tracked: Number of unique tasks tracked
        """
        completed = sum(1 for t in self._chunk_timings.values() if t['status'] == 'complete')
        in_progress = sum(1 for t in self._chunk_timings.values() if t['status'] == 'in_progress')

        return {
            'total_transitions': len(self._state_transitions),
            'average_accuracy': self.get_estimation_accuracy(),
            'time_distribution': self.get_time_in_state_distribution(),
            'chunks_completed': completed,
            'chunks_in_progress': in_progress,
            'tasks_tracked': len(self._task_timings),
        }
