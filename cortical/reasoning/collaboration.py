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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
# PARALLEL AGENT COORDINATION - BOUNDARY-BASED ISOLATION
# =============================================================================
#
# Design Philosophy (Phase 1):
# - Clear boundaries prevent conflicts WITHOUT inter-agent communication
# - Agents work independently within their assigned file sets
# - Conflicts detected at merge time, not prevented at runtime
# - Simple spawner interface allows testing without actual subprocess overhead
#
# This follows the principle: "The 8 sprint agents worked WITHOUT communication
# due to clear boundaries" - proven in practice during Sprints 1-3.
# =============================================================================


class AgentStatus(Enum):
    """Status of a spawned agent."""
    PENDING = auto()  # Created but not started
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Finished successfully
    FAILED = auto()  # Finished with error
    TIMED_OUT = auto()  # Exceeded time budget


@dataclass
class AgentResult:
    """
    Result from a completed agent execution.

    Contains all information needed to assess agent output and merge changes.
    """
    agent_id: str
    status: AgentStatus
    task_description: str
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    output: str = ""  # Agent's final output/summary
    error: Optional[str] = None
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def success(self) -> bool:
        """Check if agent completed successfully."""
        return self.status == AgentStatus.COMPLETED

    def all_modified_files(self) -> Set[str]:
        """Get all files that were changed in any way."""
        return set(self.files_modified) | set(self.files_created) | set(self.files_deleted)


class AgentSpawner(ABC):
    """
    Abstract interface for spawning agents.

    This abstraction allows:
    - SequentialSpawner: For testing, runs tasks inline without subprocesses
    - ClaudeCodeSpawner: For production, uses Task tool to spawn real agents
    - MockSpawner: For unit tests, returns predefined results

    Why this design:
    - Tests run fast without subprocess overhead
    - Same ParallelCoordinator logic works in all environments
    - Easy to add new spawner types (e.g., ThreadPoolSpawner)
    """

    @abstractmethod
    def spawn(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
        timeout_seconds: int = 300
    ) -> str:
        """
        Spawn an agent to work on a task.

        Args:
            task: Description of what the agent should do
            boundary: Work boundary defining files the agent can modify
            timeout_seconds: Maximum time for agent execution

        Returns:
            Agent ID for tracking
        """
        pass

    @abstractmethod
    def get_status(self, agent_id: str) -> AgentStatus:
        """Get current status of an agent."""
        pass

    @abstractmethod
    def get_result(self, agent_id: str) -> Optional[AgentResult]:
        """
        Get result from a completed agent.

        Returns None if agent is still running or doesn't exist.
        """
        pass

    @abstractmethod
    def wait_for(self, agent_id: str, timeout_seconds: int = 300) -> AgentResult:
        """Wait for an agent to complete and return its result."""
        pass


class SequentialSpawner(AgentSpawner):
    """
    Sequential spawner for testing - runs tasks inline.

    Instead of spawning subprocesses, executes tasks sequentially using
    a provided task handler function. This allows fast testing without
    subprocess overhead.

    Usage:
        def my_handler(task: str, boundary: ParallelWorkBoundary) -> AgentResult:
            # Do the work synchronously
            return AgentResult(...)

        spawner = SequentialSpawner(handler=my_handler)
        coordinator = ParallelCoordinator(spawner)
    """

    def __init__(self, handler: Optional[Callable[[str, ParallelWorkBoundary], AgentResult]] = None):
        """
        Initialize with optional task handler.

        Args:
            handler: Function that executes tasks and returns results.
                     If None, uses a default that returns empty success.
        """
        self._handler = handler or self._default_handler
        self._agents: Dict[str, AgentResult] = {}
        self._boundaries: Dict[str, ParallelWorkBoundary] = {}
        self._counter = 0

    def _default_handler(self, task: str, boundary: ParallelWorkBoundary) -> AgentResult:
        """Default handler returns successful empty result."""
        return AgentResult(
            agent_id="",  # Will be set by spawn()
            status=AgentStatus.COMPLETED,
            task_description=task,
            output=f"Completed task: {task[:50]}...",
        )

    def spawn(
        self,
        task: str,
        boundary: ParallelWorkBoundary,
        timeout_seconds: int = 300
    ) -> str:
        """Spawn runs task synchronously and stores result."""
        agent_id = f"seq-agent-{self._counter:03d}"
        self._counter += 1

        self._boundaries[agent_id] = boundary

        start_time = datetime.now()
        try:
            result = self._handler(task, boundary)
            result.agent_id = agent_id
            result.started_at = start_time
            result.completed_at = datetime.now()
            result.duration_seconds = (result.completed_at - start_time).total_seconds()

            # Validate result respects boundary
            for f in result.all_modified_files():
                if not boundary.can_modify(f) and f not in boundary.files_read_only:
                    # Agent modified file outside boundary - mark as conflict
                    result.error = f"Boundary violation: modified {f} outside owned files"

        except Exception as e:
            result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                task_description=task,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.now(),
            )
            result.duration_seconds = (result.completed_at - start_time).total_seconds()

        self._agents[agent_id] = result
        return agent_id

    def get_status(self, agent_id: str) -> AgentStatus:
        """Get status - always COMPLETED since we run synchronously."""
        if agent_id in self._agents:
            return self._agents[agent_id].status
        return AgentStatus.PENDING  # Unknown agent

    def get_result(self, agent_id: str) -> Optional[AgentResult]:
        """Get result for completed agent."""
        return self._agents.get(agent_id)

    def wait_for(self, agent_id: str, timeout_seconds: int = 300) -> AgentResult:
        """Return result immediately since tasks run synchronously."""
        result = self._agents.get(agent_id)
        if result is None:
            return AgentResult(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                task_description="Unknown agent",
                error=f"Agent {agent_id} not found",
            )
        return result


@dataclass
class ConflictDetail:
    """Detailed information about a conflict between agent results."""
    conflict_type: ConflictType
    agents_involved: List[str]
    files_affected: List[str]
    description: str
    resolution_suggestion: str = ""


class ParallelCoordinator:
    """
    Coordinator for parallel agent execution using boundary-based isolation.

    Phase 1 Design (No Inter-Agent Communication):
    - Agents work independently within assigned file boundaries
    - Conflicts detected at result collection, not prevented at runtime
    - Simple spawner abstraction allows flexible execution strategies

    The coordinator:
    1. Validates boundaries don't overlap before spawning
    2. Spawns agents to work within their boundaries
    3. Collects results and detects any conflicts
    4. Reports which files changed and any issues

    Usage:
        spawner = SequentialSpawner(handler=my_task_handler)
        coordinator = ParallelCoordinator(spawner)

        # Check if tasks can run in parallel
        can_spawn, issues = coordinator.can_spawn([boundary1, boundary2])

        if can_spawn:
            agent_ids = coordinator.spawn_agents(tasks, boundaries)
            results = coordinator.collect_results(agent_ids)
            conflicts = coordinator.detect_conflicts(results)
    """

    def __init__(self, spawner: AgentSpawner):
        """
        Initialize coordinator with an agent spawner.

        Args:
            spawner: The spawner implementation to use for agent execution
        """
        self._spawner = spawner
        self._active_agents: Dict[str, ParallelWorkBoundary] = {}
        self._completed_results: Dict[str, AgentResult] = {}
        self._detected_conflicts: List[ConflictDetail] = []

    def can_spawn(self, boundaries: List[ParallelWorkBoundary]) -> Tuple[bool, List[str]]:
        """
        Check if tasks with given boundaries can run in parallel.

        Args:
            boundaries: List of work boundaries for proposed parallel tasks

        Returns:
            Tuple of (can_spawn: bool, issues: list of conflict descriptions)
        """
        issues = []

        # Check for file ownership conflicts between boundaries
        for i, b1 in enumerate(boundaries):
            for b2 in boundaries[i + 1:]:
                conflicts = b1.conflicts_with(b2)
                if conflicts:
                    issues.append(
                        f"Conflict: {b1.agent_id} and {b2.agent_id} both claim "
                        f"ownership of: {', '.join(conflicts[:3])}"
                        + (f" (and {len(conflicts) - 3} more)" if len(conflicts) > 3 else "")
                    )

        # Check for read-write conflicts (one reads what another writes)
        for i, b1 in enumerate(boundaries):
            for b2 in boundaries[i + 1:]:
                # Does b1 read files that b2 writes?
                read_write_conflict = b1.files_read_only & b2.files_owned
                if read_write_conflict:
                    issues.append(
                        f"Potential race: {b1.agent_id} reads files that "
                        f"{b2.agent_id} may modify: {', '.join(list(read_write_conflict)[:3])}"
                    )

                # Does b2 read files that b1 writes?
                read_write_conflict = b2.files_read_only & b1.files_owned
                if read_write_conflict:
                    issues.append(
                        f"Potential race: {b2.agent_id} reads files that "
                        f"{b1.agent_id} may modify: {', '.join(list(read_write_conflict)[:3])}"
                    )

        return len(issues) == 0, issues

    def spawn_agents(
        self,
        tasks: List[str],
        boundaries: List[ParallelWorkBoundary],
        timeout_seconds: int = 300
    ) -> List[str]:
        """
        Spawn agents for parallel task execution.

        Args:
            tasks: List of task descriptions
            boundaries: Matching list of work boundaries
            timeout_seconds: Maximum time per agent

        Returns:
            List of agent IDs

        Raises:
            ValueError: If tasks and boundaries lists have different lengths
        """
        if len(tasks) != len(boundaries):
            raise ValueError(
                f"Must have same number of tasks ({len(tasks)}) "
                f"and boundaries ({len(boundaries)})"
            )

        agent_ids = []
        for task, boundary in zip(tasks, boundaries):
            agent_id = self._spawner.spawn(task, boundary, timeout_seconds)
            self._active_agents[agent_id] = boundary
            agent_ids.append(agent_id)

        return agent_ids

    def collect_results(
        self,
        agent_ids: List[str],
        timeout_seconds: int = 300
    ) -> Dict[str, AgentResult]:
        """
        Wait for agents and collect their results.

        Args:
            agent_ids: List of agent IDs to wait for
            timeout_seconds: Maximum total time to wait

        Returns:
            Dictionary mapping agent_id to AgentResult
        """
        results = {}
        for agent_id in agent_ids:
            result = self._spawner.wait_for(agent_id, timeout_seconds)
            results[agent_id] = result
            self._completed_results[agent_id] = result

            # Clean up active tracking
            if agent_id in self._active_agents:
                del self._active_agents[agent_id]

        return results

    def detect_conflicts(self, results: Dict[str, AgentResult]) -> List[ConflictDetail]:
        """
        Detect conflicts in completed agent results.

        Checks for:
        - File conflicts: Multiple agents modified same file
        - Boundary violations: Agent modified files outside its boundary
        - Dependency conflicts: Agent needed files another modified

        Args:
            results: Dictionary of agent results

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Collect all modified files by agent
        agent_files: Dict[str, Set[str]] = {}
        for agent_id, result in results.items():
            agent_files[agent_id] = result.all_modified_files()

        # Check for overlapping modifications
        agent_ids = list(results.keys())
        for i, a1 in enumerate(agent_ids):
            for a2 in agent_ids[i + 1:]:
                overlap = agent_files[a1] & agent_files[a2]
                if overlap:
                    conflict = ConflictDetail(
                        conflict_type=ConflictType.FILE_CONFLICT,
                        agents_involved=[a1, a2],
                        files_affected=list(overlap),
                        description=f"Both {a1} and {a2} modified: {', '.join(list(overlap)[:5])}",
                        resolution_suggestion="Review changes in both agents and merge manually",
                    )
                    conflicts.append(conflict)

        # Check for boundary violations
        for agent_id, result in results.items():
            if result.error and "Boundary violation" in result.error:
                conflict = ConflictDetail(
                    conflict_type=ConflictType.SCOPE_OVERLAP,
                    agents_involved=[agent_id],
                    files_affected=list(result.all_modified_files()),
                    description=result.error,
                    resolution_suggestion="Agent worked outside assigned boundary - review all changes",
                )
                conflicts.append(conflict)

        self._detected_conflicts.extend(conflicts)
        return conflicts

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of coordination state."""
        completed = [r for r in self._completed_results.values() if r.success()]
        failed = [r for r in self._completed_results.values() if not r.success()]

        all_files = set()
        for result in self._completed_results.values():
            all_files.update(result.all_modified_files())

        return {
            'active_agents': len(self._active_agents),
            'completed_agents': len(completed),
            'failed_agents': len(failed),
            'total_files_modified': len(all_files),
            'conflicts_detected': len(self._detected_conflicts),
            'files_modified': list(all_files)[:20],  # First 20 for summary
        }

    def get_active_agent_ids(self) -> List[str]:
        """Get IDs of currently active agents."""
        return list(self._active_agents.keys())

    def get_completed_results(self) -> Dict[str, AgentResult]:
        """Get all completed agent results."""
        return dict(self._completed_results)

    def get_conflicts(self) -> List[ConflictDetail]:
        """Get all detected conflicts."""
        return list(self._detected_conflicts)

    def reset(self) -> None:
        """Reset coordinator state for reuse."""
        self._active_agents.clear()
        self._completed_results.clear()
        self._detected_conflicts.clear()


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
