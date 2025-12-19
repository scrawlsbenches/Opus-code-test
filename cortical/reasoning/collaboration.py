"""
Collaboration and Coordination: Working Together Effectively.

This module implements the collaboration protocols from Part 14 of
docs/complex-reasoning-workflow.md. It provides:

- Collaboration modes (synchronous, asynchronous, semi-synchronous)
- Disagreement resolution protocols
- Status communication standards
- Parallel work coordination
- Handoff management during active work

Design Philosophy:
    Real work involves coordination with others (humans and AIs).
    This module covers the dynamics of working together, ensuring
    that parallel efforts don't conflict and communication is clear.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import uuid


class CollaborationMode(Enum):
    """
    Modes of collaboration between human and AI.

    From docs/complex-reasoning-workflow.md Part 14.1:
    - SYNCHRONOUS: Real-time collaboration, immediate feedback
    - ASYNCHRONOUS: Batch mode, human provides task and returns for results
    - SEMI_SYNCHRONOUS: Hybrid, human available but not actively watching
    """
    SYNCHRONOUS = auto()  # Human actively present
    ASYNCHRONOUS = auto()  # Human provides task, returns for results
    SEMI_SYNCHRONOUS = auto()  # Human available but not watching


class BlockerType(Enum):
    """
    Type of blocker requiring attention.

    From docs/complex-reasoning-workflow.md Part 14.3:
    - HARD: Cannot proceed at all
    - SOFT: Can workaround but suboptimal
    - INFO: Would help but not blocking
    """
    HARD = auto()  # Work stopped, ASAP response needed
    SOFT = auto()  # Workaround available, response before completion
    INFO = auto()  # Nice to have, respond when convenient


class ConflictType(Enum):
    """Types of conflicts that can arise in parallel work."""
    FILE_CONFLICT = auto()  # Same file modified
    LOGIC_CONFLICT = auto()  # Incompatible changes
    DEPENDENCY_CONFLICT = auto()  # Circular or broken dependencies
    SCOPE_OVERLAP = auto()  # Work boundaries unclear


@dataclass
class StatusUpdate:
    """
    A status update during work.

    From docs/complex-reasoning-workflow.md Part 14.3:
    Provides visibility into progress, blockers, and needs.
    """
    task_name: str
    progress_percent: int = 0
    current_phase: str = ""
    eta_minutes: Optional[int] = None
    completed_items: List[str] = field(default_factory=list)
    in_progress_items: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    needs_from_human: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Generate markdown-formatted status update."""
        lines = [
            f"## Status: {self.task_name}",
            "",
            f"**Progress:** {self.progress_percent}% complete",
            f"**Current phase:** {self.current_phase}",
        ]

        if self.eta_minutes:
            lines.append(f"**ETA:** {self.eta_minutes} minutes")

        lines.append("")

        if self.completed_items:
            lines.append("**Completed:**")
            for item in self.completed_items:
                lines.append(f"- [x] {item}")
            lines.append("")

        if self.in_progress_items:
            lines.append("**In progress:**")
            for item in self.in_progress_items:
                lines.append(f"- [ ] {item}")
            lines.append("")

        lines.append(f"**Blockers:** {', '.join(self.blockers) if self.blockers else 'None'}")
        lines.append(f"**Concerns:** {', '.join(self.concerns) if self.concerns else 'None'}")
        lines.append(f"**Need from you:** {', '.join(self.needs_from_human) if self.needs_from_human else 'Nothing'}")

        return "\n".join(lines)


@dataclass
class Blocker:
    """
    A blocker preventing progress.

    Tracks what's blocked, why, and what's needed to unblock.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    blocker_type: BlockerType = BlockerType.SOFT
    raised_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    resolution_needed: str = ""  # What would unblock
    workaround: Optional[str] = None  # If SOFT, what's the workaround
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    def resolve(self, resolution: str) -> None:
        """Mark blocker as resolved."""
        self.resolved = True
        self.resolution = resolution
        self.resolved_at = datetime.now()


@dataclass
class DisagreementRecord:
    """
    Record of a disagreement between human and AI (or between agents).

    From docs/complex-reasoning-workflow.md Part 14.2:
    "Sometimes the AI has information or perspective the human lacks.
     Respectfully surfacing disagreement is a feature, not a bug."
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    instruction_given: str = ""  # What human asked
    concern_raised: str = ""  # Why AI disagrees
    evidence: List[str] = field(default_factory=list)  # Supporting evidence
    risk_if_proceed: str = ""  # What could go wrong
    alternative_suggested: str = ""  # Different approach
    human_decision: Optional[str] = None  # What human decided
    outcome: Optional[str] = None  # How it turned out
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None

    def to_markdown(self) -> str:
        """Generate the Respectful Disagreement template."""
        lines = [
            "## Respectful Disagreement",
            "",
            "**Your instruction:**",
            self.instruction_given,
            "",
            "**My concern:**",
            self.concern_raised,
            "",
            "**Evidence:**",
        ]

        for evidence in self.evidence:
            lines.append(f"- {evidence}")

        lines.extend([
            "",
            "**Risk if we proceed as instructed:**",
            self.risk_if_proceed,
            "",
            "**Alternative I'd suggest:**",
            self.alternative_suggested,
            "",
            "**However:**",
            "You have context I may lack. If you still want to proceed",
            "with the original instruction, I will do so and document",
            "my concern for future reference.",
            "",
            "**Your call:** Proceed as instructed / Try alternative / Discuss further",
        ])

        return "\n".join(lines)


@dataclass
class ParallelWorkBoundary:
    """
    Defines boundaries for parallel work to prevent conflicts.

    From docs/complex-reasoning-workflow.md Part 14.4:
    Pre-parallel work requires clear boundaries.
    """
    agent_id: str
    scope_description: str
    files_owned: Set[str] = field(default_factory=set)  # Exclusive access
    files_read_only: Set[str] = field(default_factory=set)  # Can read, not write
    dependencies: List[str] = field(default_factory=list)  # Other agent work needed
    created_at: datetime = field(default_factory=datetime.now)

    def add_file(self, file_path: str, write_access: bool = True) -> None:
        """Add a file to this boundary."""
        if write_access:
            self.files_owned.add(file_path)
        else:
            self.files_read_only.add(file_path)

    def can_modify(self, file_path: str) -> bool:
        """Check if this boundary allows modifying a file."""
        return file_path in self.files_owned

    def conflicts_with(self, other: 'ParallelWorkBoundary') -> List[str]:
        """Find files that conflict with another boundary."""
        return list(self.files_owned & other.files_owned)


@dataclass
class ConflictEvent:
    """
    Record of a conflict detected between parallel workers.

    When conflicts are detected:
    1. DETECT: Agent discovers conflict
    2. STOP: Don't proceed with conflicting changes
    3. DOCUMENT: What the conflict is
    4. ESCALATE: Notify coordinator
    5. WAIT: Don't resolve unilaterally
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conflict_type: ConflictType = ConflictType.FILE_CONFLICT
    agents_involved: List[str] = field(default_factory=list)
    description: str = ""
    files_affected: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    escalated: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class ActiveWorkHandoff:
    """
    Handoff document for when work must transfer mid-stream.

    From docs/complex-reasoning-workflow.md Part 14.5:
    "Context limits, session ends, or work must transfer.
     Make handoff seamless."
    """
    task_description: str
    status: str  # Where we are in the process
    urgency: str  # How time-sensitive

    # Current state
    files_working: List[str] = field(default_factory=list)  # In good state
    files_in_progress: Dict[str, str] = field(default_factory=dict)  # file -> what remains
    known_issues: List[str] = field(default_factory=list)

    # Context
    key_decisions: Dict[str, str] = field(default_factory=dict)  # decision -> rationale
    gotchas: List[str] = field(default_factory=list)
    files_to_read_first: List[str] = field(default_factory=list)

    # Next steps
    immediate_next_steps: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    verification_command: str = ""  # How to verify you understand

    created_at: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Generate the Active Work Handoff template."""
        lines = [
            "## Active Work Handoff",
            "",
            f"**Task:** {self.task_description}",
            f"**Status:** {self.status}",
            f"**Urgency:** {self.urgency}",
            "",
            "### Current State",
            "",
            "**What's working:**",
        ]

        for file in self.files_working:
            lines.append(f"- {file}")

        lines.extend(["", "**What's in progress:**"])
        for file, remaining in self.files_in_progress.items():
            lines.append(f"- {file}: {remaining}")

        lines.extend(["", "**What's broken:**"])
        for issue in self.known_issues:
            lines.append(f"- {issue}")

        lines.extend([
            "",
            "### Context You Need",
            "",
            "**Why we're doing it this way:**",
        ])
        for decision, rationale in self.key_decisions.items():
            lines.append(f"- {decision}: {rationale}")

        lines.extend(["", "**Gotchas discovered:**"])
        for gotcha in self.gotchas:
            lines.append(f"- {gotcha}")

        lines.extend(["", "**Files to read first:**"])
        for i, file in enumerate(self.files_to_read_first, 1):
            lines.append(f"{i}. {file}")

        lines.extend([
            "",
            "### Immediate Next Steps",
            "",
        ])
        for i, step in enumerate(self.immediate_next_steps, 1):
            lines.append(f"{i}. {step}")

        lines.extend(["", "### Questions Still Open", ""])
        for question in self.open_questions:
            lines.append(f"- {question}")

        lines.extend([
            "",
            "### How to Verify You're On Track",
            "",
            self.verification_command,
        ])

        return "\n".join(lines)


class CollaborationManager:
    """
    Central manager for collaboration activities.

    Manages:
    - Collaboration mode detection and handling
    - Status update scheduling
    - Blocker tracking and resolution
    - Disagreement recording
    - Parallel work coordination
    - Handoff generation
    """

    def __init__(self, mode: CollaborationMode = CollaborationMode.SEMI_SYNCHRONOUS):
        """Initialize the collaboration manager."""
        self._mode = mode
        self._status_updates: List[StatusUpdate] = []
        self._blockers: Dict[str, Blocker] = {}
        self._disagreements: List[DisagreementRecord] = []
        self._boundaries: Dict[str, ParallelWorkBoundary] = {}
        self._conflicts: List[ConflictEvent] = []
        self._handoffs: List[ActiveWorkHandoff] = []

        # Update frequency by mode (in minutes)
        self._update_intervals = {
            CollaborationMode.SYNCHRONOUS: 5,  # Very frequent
            CollaborationMode.ASYNCHRONOUS: 60,  # At milestones
            CollaborationMode.SEMI_SYNCHRONOUS: 15,  # Regular
        }

    @property
    def mode(self) -> CollaborationMode:
        """Get current collaboration mode."""
        return self._mode

    @mode.setter
    def mode(self, value: CollaborationMode) -> None:
        """Set collaboration mode."""
        self._mode = value

    def get_update_interval(self) -> int:
        """Get recommended status update interval in minutes."""
        return self._update_intervals[self._mode]

    def post_status(self, update: StatusUpdate) -> None:
        """Post a status update."""
        self._status_updates.append(update)

    def raise_blocker(
        self,
        description: str,
        blocker_type: BlockerType = BlockerType.SOFT,
        resolution_needed: str = "",
        workaround: str = None,
        context: Dict[str, Any] = None
    ) -> Blocker:
        """
        Raise a new blocker.

        Args:
            description: What's blocked
            blocker_type: Severity of blocker
            resolution_needed: What's needed to unblock
            workaround: Alternative approach if SOFT blocker
            context: Additional context

        Returns:
            Blocker instance
        """
        blocker = Blocker(
            description=description,
            blocker_type=blocker_type,
            resolution_needed=resolution_needed,
            workaround=workaround,
            context=context or {},
        )
        self._blockers[blocker.id] = blocker
        return blocker

    def resolve_blocker(self, blocker_id: str, resolution: str) -> None:
        """Resolve a blocker."""
        if blocker_id in self._blockers:
            self._blockers[blocker_id].resolve(resolution)

    def get_active_blockers(self) -> List[Blocker]:
        """Get all unresolved blockers."""
        return [b for b in self._blockers.values() if not b.resolved]

    def get_hard_blockers(self) -> List[Blocker]:
        """Get all unresolved HARD blockers."""
        return [
            b for b in self._blockers.values()
            if not b.resolved and b.blocker_type == BlockerType.HARD
        ]

    def record_disagreement(
        self,
        instruction: str,
        concern: str,
        evidence: List[str],
        risk: str,
        alternative: str
    ) -> DisagreementRecord:
        """
        Record a disagreement with human guidance.

        From Part 14.2: Respectfully surfacing disagreement.

        Returns:
            DisagreementRecord for tracking
        """
        record = DisagreementRecord(
            instruction_given=instruction,
            concern_raised=concern,
            evidence=evidence,
            risk_if_proceed=risk,
            alternative_suggested=alternative,
        )
        self._disagreements.append(record)
        return record

    def create_boundary(
        self,
        agent_id: str,
        scope: str,
        files: Set[str] = None
    ) -> ParallelWorkBoundary:
        """
        Create a work boundary for parallel coordination.

        Args:
            agent_id: Identifier for the agent
            scope: Description of work scope
            files: Files owned by this boundary

        Returns:
            ParallelWorkBoundary instance
        """
        boundary = ParallelWorkBoundary(
            agent_id=agent_id,
            scope_description=scope,
            files_owned=files or set(),
        )
        self._boundaries[agent_id] = boundary
        return boundary

    def check_conflicts(self) -> List[ConflictEvent]:
        """
        Check for conflicts between all registered boundaries.

        Returns:
            List of detected conflicts
        """
        conflicts = []
        boundaries = list(self._boundaries.values())

        for i, b1 in enumerate(boundaries):
            for b2 in boundaries[i + 1:]:
                conflicting_files = b1.conflicts_with(b2)
                if conflicting_files:
                    conflict = ConflictEvent(
                        conflict_type=ConflictType.FILE_CONFLICT,
                        agents_involved=[b1.agent_id, b2.agent_id],
                        description=f"File ownership conflict between {b1.agent_id} and {b2.agent_id}",
                        files_affected=conflicting_files,
                    )
                    conflicts.append(conflict)
                    self._conflicts.append(conflict)

        return conflicts

    def create_handoff(
        self,
        task: str,
        status: str,
        urgency: str
    ) -> ActiveWorkHandoff:
        """
        Create a handoff document for mid-work transfer.

        Args:
            task: What's being worked on
            status: Where we are
            urgency: How time-sensitive

        Returns:
            ActiveWorkHandoff to fill in and share
        """
        handoff = ActiveWorkHandoff(
            task_description=task,
            status=status,
            urgency=urgency,
        )
        self._handoffs.append(handoff)
        return handoff

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collaboration state."""
        return {
            'mode': self._mode.name,
            'update_interval_minutes': self.get_update_interval(),
            'status_updates': len(self._status_updates),
            'active_blockers': len(self.get_active_blockers()),
            'hard_blockers': len(self.get_hard_blockers()),
            'disagreements': len(self._disagreements),
            'boundaries': len(self._boundaries),
            'conflicts': len([c for c in self._conflicts if not c.resolved_at]),
            'handoffs': len(self._handoffs),
        }


# =============================================================================
# STUB CLASSES FOR COMPLEX IMPLEMENTATIONS
# =============================================================================


class ParallelCoordinator:
    """
    STUB: Coordinator for parallel agent execution.

    Full Implementation Would:
    --------------------------
    1. Agent orchestration
       - Spawn parallel agents with defined boundaries
       - Monitor progress across agents
       - Coordinate shared resources
       - Handle agent failures

    2. Conflict prevention
       - Pre-check boundaries before starting
       - Lock files during modification
       - Queue conflicting operations
       - Automatic conflict resolution where possible

    3. Result merging
       - Collect outputs from all agents
       - Merge non-conflicting changes
       - Flag conflicts for human resolution
       - Generate unified commit

    4. Communication
       - Cross-agent messaging
       - Shared context updates
       - Progress aggregation
       - Dependency notification
    """

    def spawn_parallel(
        self,
        tasks: List[Dict[str, Any]],
        boundaries: List[ParallelWorkBoundary]
    ) -> Dict[str, Any]:
        """
        STUB: Spawn parallel agents for tasks.

        Args:
            tasks: List of task definitions
            boundaries: Work boundaries for each task

        Returns:
            {'agents': list of agent IDs, 'conflicts': list of pre-detected conflicts}
        """
        return {
            'agents': [f"agent_{i}" for i in range(len(tasks))],
            'conflicts': [],
            'note': 'STUB: Would spawn actual parallel agents',
        }

    def wait_for_completion(self, agent_ids: List[str], timeout_minutes: int = 60) -> Dict[str, Any]:
        """
        STUB: Wait for parallel agents to complete.

        Returns:
            {'completed': list, 'failed': list, 'timed_out': list}
        """
        return {
            'completed': agent_ids,
            'failed': [],
            'timed_out': [],
            'note': 'STUB: Would wait for actual agent completion',
        }

    def merge_results(self, agent_ids: List[str]) -> Dict[str, Any]:
        """
        STUB: Merge results from parallel agents.

        Returns:
            {'merged_files': list, 'conflicts': list, 'commit_ready': bool}
        """
        return {
            'merged_files': [],
            'conflicts': [],
            'commit_ready': True,
            'note': 'STUB: Would perform actual merge',
        }


class QuestionBatcher:
    """
    STUB: Intelligent question batching for async communication.

    Full Implementation Would:
    --------------------------
    From docs/complex-reasoning-workflow.md Part 4.3:
    "Before asking the human questions:
    - Batch them
    - Explain why you're asking
    - Respect their time and attention"

    1. Question collection
       - Collect questions as they arise
       - Categorize by urgency and topic
       - Identify dependencies between questions

    2. Batch optimization
       - Group related questions
       - Order by importance
       - Include context for batch
       - Provide defaults if no response

    3. Response handling
       - Parse responses
       - Distribute answers to waiting contexts
       - Handle partial responses
       - Track unanswered questions
    """

    def __init__(self):
        self._pending_questions: List[Dict[str, Any]] = []

    def add_question(
        self,
        question: str,
        context: str = "",
        default: str = None,
        urgency: str = "medium"
    ) -> str:
        """
        STUB: Add a question to the batch.

        Returns:
            Question ID for tracking
        """
        q_id = f"Q-{len(self._pending_questions):03d}"
        self._pending_questions.append({
            'id': q_id,
            'question': question,
            'context': context,
            'default': default,
            'urgency': urgency,
        })
        return q_id

    def generate_batch(self) -> str:
        """
        STUB: Generate a batched question request.

        Returns:
            Markdown-formatted question batch
        """
        if not self._pending_questions:
            return "No pending questions."

        lines = [
            "## Question Request",
            "",
            f"I need to ask {len(self._pending_questions)} questions before proceeding.",
            "",
            "**Questions:**",
        ]

        for i, q in enumerate(self._pending_questions, 1):
            lines.append(f"{i}. {q['question']}")
            if q['default']:
                lines.append(f"   (Default if no response: {q['default']})")

        return "\n".join(lines)
