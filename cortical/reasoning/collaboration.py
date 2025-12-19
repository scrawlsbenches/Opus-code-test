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


@dataclass
class BatchedQuestion:
    """
    A question in the batch with metadata for categorization and tracking.

    From docs/complex-reasoning-workflow.md Part 4.3:
    "Before asking the human questions:
    - Batch them
    - Explain why you're asking
    - Respect their time and attention"
    """
    id: str
    question: str
    context: str = ""
    default: Optional[str] = None
    urgency: str = "medium"  # critical, high, medium, low
    category: str = "general"  # technical, clarification, approval, design, general
    blocking: bool = False  # Does work stop until answered?
    related_ids: List[str] = field(default_factory=list)  # Related question IDs
    answered: bool = False
    response: Optional[str] = None


class QuestionBatcher:
    """
    Intelligent question batching for async communication.

    From docs/complex-reasoning-workflow.md Part 4.3:
    "Before asking the human questions:
    - Batch them
    - Explain why you're asking
    - Respect their time and attention"

    Features:
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
        """Initialize the question batcher."""
        self._questions: Dict[str, BatchedQuestion] = {}
        self._question_counter = 0

    def add_question(
        self,
        question: str,
        context: str = "",
        default: Optional[str] = None,
        urgency: str = "medium",
        category: str = "general",
        blocking: bool = False,
        related_ids: Optional[List[str]] = None
    ) -> str:
        """
        Add a question to the batch.

        Args:
            question: The question text
            context: Context explaining why we're asking
            default: Default value if no response
            urgency: Priority level (critical, high, medium, low)
            category: Question category (technical, clarification, approval, design, general)
            blocking: Whether work stops until answered
            related_ids: IDs of related questions

        Returns:
            Question ID for tracking
        """
        q_id = f"Q-{self._question_counter:03d}"
        self._question_counter += 1

        batched_q = BatchedQuestion(
            id=q_id,
            question=question,
            context=context,
            default=default,
            urgency=urgency,
            category=category,
            blocking=blocking,
            related_ids=related_ids or [],
        )

        self._questions[q_id] = batched_q
        return q_id

    def categorize_questions(self) -> Dict[str, List[BatchedQuestion]]:
        """
        Group questions by category and sort by priority.

        Returns:
            Dictionary mapping category to sorted list of questions
        """
        # Priority order for urgency
        urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        # Group by category
        categorized: Dict[str, List[BatchedQuestion]] = {}
        for question in self._questions.values():
            if not question.answered:
                category = question.category
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(question)

        # Sort within each category by urgency (critical first) and blocking status
        for category in categorized:
            categorized[category].sort(
                key=lambda q: (
                    not q.blocking,  # Blocking questions first
                    urgency_order.get(q.urgency, 99),  # Then by urgency
                    q.id  # Then by ID for stable sort
                )
            )

        return categorized

    def generate_batch(self) -> str:
        """
        Generate a well-formatted markdown batch of questions.

        Returns:
            Markdown-formatted question batch with sections by category
        """
        unanswered = [q for q in self._questions.values() if not q.answered]

        if not unanswered:
            return "No pending questions."

        categorized = self.categorize_questions()
        blocking_questions = [q for q in unanswered if q.blocking]

        lines = [
            "## Question Request",
            "",
        ]

        # Summary
        if blocking_questions:
            lines.extend([
                f"**URGENT:** {len(blocking_questions)} blocking question(s) - work cannot proceed until answered.",
                "",
            ])

        lines.extend([
            f"I need to ask {len(unanswered)} question(s) to ensure I proceed correctly.",
            "",
        ])

        # Questions by category
        category_names = {
            'technical': 'Technical Questions',
            'clarification': 'Clarification Needed',
            'approval': 'Approval Required',
            'design': 'Design Decisions',
            'general': 'General Questions',
        }

        for category in sorted(categorized.keys()):
            questions = categorized[category]
            category_title = category_names.get(category, category.title())

            lines.extend([
                f"### {category_title}",
                "",
            ])

            for q in questions:
                # Question with ID
                prefix = "🔴 **[BLOCKING]**" if q.blocking else ""
                urgency_marker = ""
                if q.urgency == "critical":
                    urgency_marker = " ⚠️"
                elif q.urgency == "high":
                    urgency_marker = " ⬆️"

                lines.append(f"**{q.id}:**{urgency_marker} {prefix} {q.question}")

                # Context if provided
                if q.context:
                    lines.append(f"   *Context:* {q.context}")

                # Default if provided
                if q.default:
                    lines.append(f"   *Default if no response:* `{q.default}`")

                # Related questions
                if q.related_ids:
                    related_str = ", ".join(q.related_ids)
                    lines.append(f"   *Related to:* {related_str}")

                lines.append("")

        # Instructions for responding
        lines.extend([
            "---",
            "",
            "### How to Respond",
            "",
            "Please answer using the question IDs:",
            "```",
            "Q-001: Your answer here",
            "Q-002: Another answer",
            "```",
            "",
            "Or in any clear format that references the question IDs.",
        ])

        return "\n".join(lines)

    def process_responses(self, response_text: str) -> Dict[str, Any]:
        """
        Parse structured responses and match to question IDs.

        Supports formats:
        - Q-001: Answer text
        - Q-001 Answer text
        - 1: Answer text (maps to Q-000, i.e., first question)
        - 2: Answer text (maps to Q-001, i.e., second question)
        - Question ID followed by answer on next line

        Args:
            response_text: The human's response text

        Returns:
            Dictionary with parsed results:
            {
                'matched': {question_id: answer},
                'unmatched_questions': [question_ids],
                'unparsed_lines': [lines we couldn't parse]
            }
        """
        matched: Dict[str, str] = {}
        unparsed: List[str] = []

        lines = response_text.strip().split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and markdown artifacts
            if not line or line.startswith('```') or line.startswith('#'):
                i += 1
                continue

            # Try to parse Q-NNN: Answer format
            if line.startswith('Q-'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    q_id = parts[0].strip()
                    answer = parts[1].strip()

                    # Check if this is a valid question ID
                    if q_id in self._questions:
                        matched[q_id] = answer
                        self._questions[q_id].answered = True
                        self._questions[q_id].response = answer
                    else:
                        unparsed.append(line)
                else:
                    # Q-NNN on its own line, answer on next line
                    q_id = parts[0].strip()
                    if q_id in self._questions and i + 1 < len(lines):
                        answer = lines[i + 1].strip()
                        matched[q_id] = answer
                        self._questions[q_id].answered = True
                        self._questions[q_id].response = answer
                        i += 1  # Skip the answer line
                    else:
                        unparsed.append(line)
            # Try to parse numeric format (1: Answer -> Q-000, 2: Answer -> Q-001)
            elif line[0].isdigit():
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        num = int(parts[0].strip())
                        # Convert 1-indexed (human-friendly) to 0-indexed (our IDs)
                        q_id = f"Q-{num-1:03d}"
                        answer = parts[1].strip()

                        if q_id in self._questions:
                            matched[q_id] = answer
                            self._questions[q_id].answered = True
                            self._questions[q_id].response = answer
                        else:
                            unparsed.append(line)
                    except ValueError:
                        unparsed.append(line)
                else:
                    unparsed.append(line)
            else:
                # Couldn't parse this line
                if line:  # Only track non-empty unparsed lines
                    unparsed.append(line)

            i += 1

        # Find unanswered questions
        unanswered = [q.id for q in self._questions.values() if not q.answered]

        return {
            'matched': matched,
            'unanswered_questions': unanswered,
            'unparsed_lines': unparsed,
        }

    def get_pending_blockers(self) -> List[BatchedQuestion]:
        """
        Get all unanswered blocking questions.

        Returns:
            List of blocking questions that haven't been answered
        """
        return [
            q for q in self._questions.values()
            if q.blocking and not q.answered
        ]

    def get_question(self, question_id: str) -> Optional[BatchedQuestion]:
        """Get a question by ID."""
        return self._questions.get(question_id)

    def get_all_questions(self) -> List[BatchedQuestion]:
        """Get all questions."""
        return list(self._questions.values())

    def get_unanswered_questions(self) -> List[BatchedQuestion]:
        """Get all unanswered questions."""
        return [q for q in self._questions.values() if not q.answered]

    def mark_answered(self, question_id: str, response: str) -> bool:
        """
        Mark a question as answered.

        Args:
            question_id: The question ID
            response: The answer

        Returns:
            True if question was found and marked, False otherwise
        """
        if question_id in self._questions:
            self._questions[question_id].answered = True
            self._questions[question_id].response = response
            return True
        return False
