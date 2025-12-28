"""
Core data types for the LLM Orchestration Framework.

This module defines the fundamental data structures used throughout
the orchestration system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Iterator, Literal


# =============================================================================
# ENUMS
# =============================================================================


class AgentRole(Enum):
    """Roles an agent can take in the hierarchy."""
    ORCHESTRATOR = auto()
    DIRECTOR = auto()
    WORKER = auto()


class TaskStatus(Enum):
    """Status of a task or goal."""
    PENDING = auto()
    READY = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()


class EventType(Enum):
    """Types of events in the pub/sub system."""
    # Lifecycle
    AGENT_SPAWNED = "agent.spawned"
    AGENT_TERMINATED = "agent.terminated"

    # Progress
    TASK_STARTED = "task.started"
    TASK_PROGRESS = "task.progress"
    TASK_CHECKPOINT = "task.checkpoint"
    TASK_COMPLETED = "task.completed"

    # Issues
    BLOCKER_RAISED = "blocker.raised"
    BLOCKER_RESOLVED = "blocker.resolved"
    ERROR_OCCURRED = "error.occurred"

    # Coordination
    DEPENDENCY_READY = "dependency.ready"
    RESOURCE_ACQUIRED = "resource.acquired"
    RESOURCE_RELEASED = "resource.released"

    # Knowledge
    DISCOVERY_MADE = "discovery.made"
    DECISION_MADE = "decision.made"
    ASSUMPTION_INVALID = "assumption.invalid"

    # Evolution
    RETROSPECTIVE = "worker.retrospective"
    INCREMENT_READY = "worker.increment_ready"
    GENERATION_COMPLETE = "evolution.generation_complete"


# =============================================================================
# CORE TYPES
# =============================================================================


@dataclass
class Goal:
    """A high-level goal to be achieved."""

    id: str
    description: str
    constraints: list[Constraint] = field(default_factory=list)
    priority: int = 0
    urgency: float = 0.0
    value: float = 1.0
    cost: float = 1.0

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    entered_ready_at: datetime | None = None
    deadline: datetime | None = None

    # State
    status: TaskStatus = TaskStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A unit of work assigned to a worker."""

    id: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    estimate_points: int = 1

    status: TaskStatus = TaskStatus.PENDING
    actual_points: int | None = None

    # Tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class SprintTask(Task):
    """A task within an agile sprint."""
    pass


@dataclass
class Constraint:
    """A constraint that must be respected."""

    name: str
    description: str
    validator: Callable[[Any], bool] | None = None
    priority: Literal["must", "should", "could"] = "must"


@dataclass
class Scope:
    """Defines the boundaries of an agent's responsibility."""

    includes: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    max_depth: int = 5
    max_duration: timedelta = field(default_factory=lambda: timedelta(minutes=30))


@dataclass
class Result:
    """The outcome of executing a goal or task."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration(self) -> timedelta | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


# =============================================================================
# CONTEXTS
# =============================================================================


@dataclass
class DirectorContext:
    """Context provided to a Director agent."""

    # Identity
    role: str
    goal: str
    scope: Scope

    # Authority
    can_spawn: list[str] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)
    escalation_path: str = ""

    # Context
    relevant_knowledge: dict[str, Any] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    # Communication
    event_bus: EventBus | None = None
    upstream_channel: Channel | None = None

    # State
    checkpoint: Checkpoint | None = None


@dataclass
class WorkerContext:
    """Context provided to a Worker agent."""

    task: str
    tools: list[str] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    constraints: list[Constraint] = field(default_factory=list)

    # Communication
    event_bus: EventBus | None = None
    result_channel: Channel | None = None

    # Sprint context
    timebox: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    velocity_benchmark: float = 5.0


@dataclass
class Checkpoint:
    """Serializable state for resume capability."""

    agent_id: str
    role: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Progress
    completed_steps: list[str] = field(default_factory=list)
    current_step: str = ""
    pending_steps: list[str] = field(default_factory=list)

    # Knowledge
    discoveries: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    # Partial outputs
    draft_outputs: dict[str, Any] = field(default_factory=dict)

    # Resume
    resume_from: str = ""
    context_needed: list[str] = field(default_factory=list)


# =============================================================================
# SEARCH RESULTS
# =============================================================================


@dataclass
class SearchResult:
    """A single search result."""

    file_path: str
    relevance_score: float
    matched_content: str
    line_range: tuple[int, int] = (0, 0)
    context_before: str = ""
    context_after: str = ""
    match_type: Literal["exact", "semantic", "expanded"] = "exact"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Complete response from a search operation."""

    results: list[SearchResult] = field(default_factory=list)
    query_interpreted: str = ""
    expansions_used: list[str] = field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    search_time_ms: float = 0.0
    corpus_stats: dict[str, Any] = field(default_factory=dict)

    # Status
    status: Literal["success", "partial", "degraded", "failed"] = "success"
    errors: list[ToolError] = field(default_factory=list)
    fallback_used: bool = False
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ToolError:
    """Structured error from a tool."""

    code: str
    message: str
    recoverable: bool = True
    suggestion: str = ""


# =============================================================================
# INCREMENTS AND DELIVERABLES
# =============================================================================


@dataclass
class Increment:
    """A deliverable increment from a sprint."""

    sprint_id: str
    goal: str
    outputs: dict[str, Any] = field(default_factory=dict)
    acceptance_met: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Retrospective:
    """Learnings from a sprint."""

    sprint_id: str
    went_well: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    # Metrics
    velocity_actual: int = 0
    velocity_planned: int = 0
    estimation_accuracy: float = 0.0
    impediment_count: int = 0


# =============================================================================
# EVENTS AND COMMUNICATION
# =============================================================================


@dataclass
class Event:
    """An event in the pub/sub system."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking
    event_id: str = ""
    trace_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source_agent_id: str = ""


class EventBus:
    """Pub/sub event bus for agent communication."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._event_log: list[Event] = []

    def subscribe(
        self,
        pattern: str,
        handler: Callable[[Event], None],
    ) -> None:
        """Subscribe to events matching pattern."""
        if pattern not in self._subscribers:
            self._subscribers[pattern] = []
        self._subscribers[pattern].append(handler)

    def unsubscribe(
        self,
        pattern: str,
        handler: Callable[[Event], None],
    ) -> None:
        """Unsubscribe from events."""
        if pattern in self._subscribers:
            self._subscribers[pattern].remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers."""
        self._event_log.append(event)

        for pattern, handlers in self._subscribers.items():
            if self._matches(event.type, pattern):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        # Log but don't fail
                        pass

    def _matches(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches subscription pattern."""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix)
        return event_type == pattern

    async def wait_for(
        self,
        patterns: list[str],
        timeout: float = 300.0,
    ) -> Event | None:
        """Wait for an event matching any of the patterns."""
        # In a real implementation, this would use async waiting
        # For now, returns None (placeholder)
        return None


@dataclass
class Channel:
    """Point-to-point communication channel."""

    channel_id: str
    from_agent: str
    to_agent: str

    async def send(self, message: Any) -> None:
        """Send a message through the channel."""
        pass

    async def receive(self, timeout: float = 60.0) -> Any:
        """Receive a message from the channel."""
        pass


# =============================================================================
# DELEGATION
# =============================================================================


@dataclass
class Delegation:
    """Standard format for delegating work to a sub-agent."""

    # What to do
    task: str
    success_looks_like: str

    # Boundaries
    scope: Scope = field(default_factory=Scope)
    constraints: list[Constraint] = field(default_factory=list)
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=15))

    # Resources
    tools: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    # Communication
    report_progress_every: timedelta = field(
        default_factory=lambda: timedelta(minutes=1)
    )
    escalate_if: list[str] = field(default_factory=list)

    # Output
    output_schema: dict[str, Any] | None = None


# =============================================================================
# IMPEDIMENTS
# =============================================================================


@dataclass
class Impediment:
    """A blocker preventing progress."""

    task_id: str
    description: str
    raised_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolution: str | None = None


@dataclass
class Blocked:
    """Exception raised when a worker is blocked."""

    reason: str
    what_i_need: str
    context: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AGENT TREE
# =============================================================================


@dataclass
class AgentNode:
    """A node in the agent hierarchy tree."""

    agent_id: str
    role: AgentRole
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)

    status: TaskStatus = TaskStatus.PENDING
    spawned_at: datetime = field(default_factory=datetime.now)
    terminated_at: datetime | None = None


class AgentTree:
    """Tree structure of agent hierarchy."""

    def __init__(self):
        self._nodes: dict[str, AgentNode] = {}
        self._root_id: str | None = None

    def add_agent(
        self,
        agent_id: str,
        role: AgentRole,
        parent_id: str | None = None,
    ) -> AgentNode:
        """Add an agent to the tree."""
        node = AgentNode(
            agent_id=agent_id,
            role=role,
            parent_id=parent_id,
        )
        self._nodes[agent_id] = node

        if parent_id is None:
            self._root_id = agent_id
        elif parent_id in self._nodes:
            self._nodes[parent_id].children.append(agent_id)

        return node

    def get_agent(self, agent_id: str) -> AgentNode | None:
        """Get an agent node by ID."""
        return self._nodes.get(agent_id)

    def get_children(self, agent_id: str) -> list[AgentNode]:
        """Get all children of an agent."""
        node = self._nodes.get(agent_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children if cid in self._nodes]

    def traverse_depth_first(self) -> Iterator[AgentNode]:
        """Traverse the tree depth-first."""
        if not self._root_id:
            return

        stack = [self._root_id]
        while stack:
            agent_id = stack.pop()
            node = self._nodes.get(agent_id)
            if node:
                yield node
                stack.extend(reversed(node.children))
