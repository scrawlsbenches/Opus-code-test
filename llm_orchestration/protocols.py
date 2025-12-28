"""
Protocols: Abstract Interfaces for the Cognitive Architecture

This module defines the contracts that implementations must fulfill.
Using Python's Protocol (structural subtyping) so implementations
don't need to inherit—they just need to implement the methods.

These protocols serve as:
1. Documentation of expected behavior
2. Type checking support
3. Guidance for implementors
4. Boundaries between components

Design Note: I use protocols rather than ABCs because they allow
structural typing—any object that has the right methods works,
regardless of inheritance. This is more flexible and Pythonic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .types import (
    Checkpoint,
    Event,
    Goal,
    Increment,
    Result,
    Retrospective,
    SprintTask,
    Task,
    TaskStatus,
)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
NodeT = TypeVar("NodeT")
EdgeT = TypeVar("EdgeT")


# =============================================================================
# CORE PROTOCOLS
# =============================================================================


@runtime_checkable
class Identifiable(Protocol):
    """
    Anything that has a unique identifier.

    This is fundamental—almost everything in the system is identifiable
    so we can reference it, track it, and persist it.
    """

    @property
    def id(self) -> str:
        """Unique identifier for this entity."""
        ...


@runtime_checkable
class Timestamped(Protocol):
    """
    Anything that tracks when it was created/modified.

    Time is crucial for:
    - Ordering events
    - Detecting staleness
    - Computing metrics (cycle time, etc.)
    """

    @property
    def created_at(self) -> datetime:
        """When this entity was created."""
        ...

    @property
    def modified_at(self) -> datetime | None:
        """When this entity was last modified, if ever."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """
    Anything that can be serialized to/from a dict.

    Critical for:
    - Persistence to disk
    - Transmission between agents
    - Checkpointing
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Deserialize from a dictionary."""
        ...


# =============================================================================
# GRAPH OF THOUGHT PROTOCOLS
# =============================================================================


@runtime_checkable
class ThoughtNode(Protocol):
    """
    A node in the Graph of Thought.

    Nodes represent discrete thoughts: questions, hypotheses,
    decisions, observations, reflections.

    Implementation Requirements:
    - Must be identifiable (has unique id)
    - Must be serializable (can persist)
    - Must track its type and content
    """

    @property
    def id(self) -> str:
        """Unique identifier for this thought."""
        ...

    @property
    def node_type(self) -> str:
        """Type of thought: QUESTION, HYPOTHESIS, DECISION, etc."""
        ...

    @property
    def content(self) -> str:
        """The actual thought content."""
        ...

    @property
    def created_at(self) -> datetime:
        """When this thought was created."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Additional metadata (confidence, source, etc.)."""
        ...


@runtime_checkable
class ThoughtEdge(Protocol):
    """
    An edge in the Graph of Thought.

    Edges represent relationships between thoughts:
    LEADS_TO, SUPPORTS, CONTRADICTS, DEPENDS_ON, etc.

    Implementation Requirements:
    - Must connect two nodes (from_id, to_id)
    - Must specify relationship type
    - May carry weight or metadata
    """

    @property
    def from_id(self) -> str:
        """ID of the source node."""
        ...

    @property
    def to_id(self) -> str:
        """ID of the target node."""
        ...

    @property
    def edge_type(self) -> str:
        """Type of relationship."""
        ...

    @property
    def weight(self) -> float:
        """Strength of relationship (default 1.0)."""
        ...


@runtime_checkable
class ThoughtGraph(Protocol):
    """
    The Graph of Thought itself.

    This is the core cognitive data structure—a graph of thoughts
    connected by typed relationships.

    Implementation Requirements:
    - Add/remove nodes
    - Add/remove edges
    - Query by node type, edge type
    - Traverse from any node
    - Persist to disk
    """

    def add_node(self, node: ThoughtNode) -> None:
        """Add a thought to the graph."""
        ...

    def add_edge(self, edge: ThoughtEdge) -> None:
        """Add a relationship between thoughts."""
        ...

    def get_node(self, node_id: str) -> ThoughtNode | None:
        """Get a node by ID."""
        ...

    def get_edges_from(self, node_id: str) -> list[ThoughtEdge]:
        """Get all edges originating from a node."""
        ...

    def get_edges_to(self, node_id: str) -> list[ThoughtEdge]:
        """Get all edges pointing to a node."""
        ...

    def query_nodes(
        self,
        node_type: str | None = None,
        **filters: Any,
    ) -> Iterator[ThoughtNode]:
        """Query nodes by type and/or filters."""
        ...

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 10,
    ) -> Iterator[tuple[ThoughtNode, int]]:
        """
        Traverse the graph from a starting node.

        Yields (node, depth) pairs.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the graph to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> "ThoughtGraph":
        """Load a graph from disk."""
        ...


# =============================================================================
# COGNITIVE STATE PROTOCOLS
# =============================================================================


@runtime_checkable
class CognitiveState(Protocol):
    """
    My externalized cognitive state.

    This captures "where I am" in my thinking—what I'm focused on,
    what questions are open, what decisions I've made.

    Implementation Requirements:
    - Track current focus
    - Track open questions
    - Track decisions with rationale
    - Support checkpointing
    - Support recovery from checkpoint
    """

    @property
    def current_focus(self) -> str | None:
        """What am I currently focused on?"""
        ...

    @property
    def open_questions(self) -> list[str]:
        """What questions remain unanswered?"""
        ...

    @property
    def decisions(self) -> list[dict[str, Any]]:
        """What decisions have I made? (with rationale)"""
        ...

    def set_focus(self, focus: str) -> None:
        """Set what I'm currently focusing on."""
        ...

    def add_question(self, question: str) -> str:
        """Add an open question. Returns question ID."""
        ...

    def answer_question(self, question_id: str, answer: str) -> None:
        """Mark a question as answered."""
        ...

    def record_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
    ) -> str:
        """Record a decision with its rationale. Returns decision ID."""
        ...

    def checkpoint(self) -> Checkpoint:
        """Create a checkpoint of current state."""
        ...

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore state from a checkpoint."""
        ...


@runtime_checkable
class ConfusionDetector(Protocol):
    """
    Detects when I'm confused or in a bad state.

    Confusion signals:
    - Repeating same failed approach
    - Contradicting earlier statements
    - Making changes without reading
    - Generating placeholder content

    Implementation Requirements:
    - Analyze recent actions for patterns
    - Return confidence level and type of confusion
    """

    def detect(
        self,
        recent_actions: list[Event],
        cognitive_state: CognitiveState,
    ) -> tuple[bool, str | None, float]:
        """
        Detect confusion.

        Returns:
            (is_confused, confusion_type, confidence)

        confusion_type is one of:
            - "loop": repeating failed approach
            - "contradiction": conflicting statements
            - "premature_action": acting without understanding
            - "placeholder": generating fake content
            - None if not confused
        """
        ...


@runtime_checkable
class RecoveryProtocol(Protocol):
    """
    Recovers from confusion or bad state.

    Recovery steps:
    1. Stop current action
    2. Load checkpoint
    3. Verify against reality
    4. Reconcile differences
    5. Resume or escalate

    Implementation Requirements:
    - Access to checkpoints
    - Access to verification tools
    - Ability to reconcile state
    """

    def recover(
        self,
        confusion_type: str,
        cognitive_state: CognitiveState,
        available_checkpoints: list[Checkpoint],
    ) -> tuple[bool, str]:
        """
        Attempt recovery.

        Returns:
            (success, message)

        If recovery fails, message explains what went wrong
        and suggests escalation.
        """
        ...


# =============================================================================
# AGENT PROTOCOLS
# =============================================================================


@runtime_checkable
class Agent(Protocol):
    """
    An agent in the hierarchy.

    Agents are the execution units—they receive work, execute it,
    and report results.

    Implementation Requirements:
    - Has identity and role
    - Can run to completion
    - Can checkpoint and resume
    - Reports status
    """

    @property
    def agent_id(self) -> str:
        """Unique identifier."""
        ...

    @property
    def role(self) -> str:
        """Role in hierarchy: orchestrator, director, worker."""
        ...

    @property
    def status(self) -> TaskStatus:
        """Current status."""
        ...

    async def run(self) -> Result:
        """Execute the agent's work."""
        ...

    async def checkpoint(self) -> Checkpoint:
        """Create resumable checkpoint."""
        ...

    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from checkpoint."""
        ...


@runtime_checkable
class Director(Agent, Protocol):
    """
    A director agent that orchestrates workers.

    Directors sit between orchestrator and workers. They:
    - Decompose goals into tasks
    - Spawn and manage workers
    - Handle blockers and coordination
    - Synthesize results

    Implementation Requirements:
    - All Agent requirements
    - Can spawn workers
    - Can decompose goals
    - Can synthesize outputs
    """

    async def decompose(self, goal: str) -> list[Task]:
        """Decompose a goal into tasks."""
        ...

    async def spawn_worker(self, task: Task) -> Agent:
        """Spawn a worker for a task."""
        ...

    async def synthesize(self, outputs: dict[str, Any]) -> Any:
        """Synthesize worker outputs into result."""
        ...


@runtime_checkable
class Worker(Agent, Protocol):
    """
    A worker agent that executes tasks.

    Workers are leaf nodes. They:
    - Execute focused tasks
    - Report progress
    - Deliver increments

    Implementation Requirements:
    - All Agent requirements
    - Reports progress
    - Delivers increments
    - Supports retrospectives
    """

    async def report_progress(self, progress: float, step: str) -> None:
        """Report current progress."""
        ...

    async def deliver_increment(self) -> Increment:
        """Deliver completed work increment."""
        ...

    async def retrospective(self) -> Retrospective:
        """Reflect on completed work."""
        ...


# =============================================================================
# ORCHESTRATION PROTOCOLS
# =============================================================================


@runtime_checkable
class Orchestrator(Protocol):
    """
    The top-level orchestrator.

    Manages the flow of goals through the system using
    kanban principles (WIP limits, pull-based assignment).

    Implementation Requirements:
    - Manages kanban board
    - Enforces WIP limits
    - Assigns directors to goals
    - Detects bottlenecks
    """

    async def submit_goal(self, goal: Goal) -> bool:
        """Submit a goal to the backlog."""
        ...

    async def pull_next_goal(self) -> Goal | None:
        """Pull next goal for execution (respects WIP)."""
        ...

    async def assign_director(self, goal: Goal) -> Director:
        """Assign a director to execute a goal."""
        ...

    def detect_bottlenecks(self) -> list[dict[str, Any]]:
        """Detect flow bottlenecks."""
        ...

    def get_board_state(self) -> dict[str, Any]:
        """Get current kanban board state."""
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """
    Pub/sub event bus for agent coordination.

    Events flow through the system:
    - Agents publish events
    - Interested parties subscribe
    - Events are logged for replay

    Implementation Requirements:
    - Subscribe with patterns
    - Publish events
    - Wait for events (async)
    """

    def subscribe(
        self,
        pattern: str,
        handler: Callable[[Event], None],
    ) -> None:
        """Subscribe to events matching pattern."""
        ...

    def unsubscribe(
        self,
        pattern: str,
        handler: Callable[[Event], None],
    ) -> None:
        """Unsubscribe from events."""
        ...

    async def publish(self, event: Event) -> None:
        """Publish an event."""
        ...

    async def wait_for(
        self,
        patterns: list[str],
        timeout: float = 300.0,
    ) -> Event | None:
        """Wait for an event matching any pattern."""
        ...


# =============================================================================
# EVOLUTION PROTOCOLS
# =============================================================================


@runtime_checkable
class Genome(Protocol):
    """
    A strategy genome.

    Genomes encode how agents behave—decomposition patterns,
    delegation strategies, failure handling, etc.

    Implementation Requirements:
    - Identifiable
    - Has fitness history
    - Can be copied for mutation
    - Iterable over genes
    """

    @property
    def genome_id(self) -> str:
        """Unique identifier."""
        ...

    @property
    def fitness_history(self) -> list[float]:
        """History of fitness scores."""
        ...

    def copy(self) -> "Genome":
        """Create a copy for mutation."""
        ...

    def genes(self) -> Iterator[tuple[str, Any]]:
        """Iterate over (gene_name, gene_value) pairs."""
        ...


@runtime_checkable
class FitnessEvaluator(Protocol):
    """
    Evaluates fitness of a genome based on execution traces.

    Fitness is multi-objective:
    - Success: did it work?
    - Efficiency: resource usage
    - Quality: output correctness
    - Stability: error rate
    - Elegance: solution simplicity

    Implementation Requirements:
    - Evaluate trace to fitness score
    - Attribute fitness to specific genes
    """

    def evaluate(self, trace: Any) -> dict[str, float]:
        """
        Evaluate execution trace.

        Returns dict with fitness dimensions:
            {"success": 0.9, "efficiency": 0.7, ...}
        """
        ...

    def attribute(
        self,
        trace: Any,
        fitness: dict[str, float],
    ) -> dict[str, float]:
        """
        Attribute fitness to genes.

        Returns dict mapping gene names to contribution:
            {"decomposition": 0.3, "delegation": -0.1, ...}

        Negative means that gene hurt fitness.
        """
        ...


@runtime_checkable
class Evolver(Protocol):
    """
    Evolves the strategy population.

    Evolution cycle:
    1. Select high-fitness genomes
    2. Crossover to create offspring
    3. Mutate for variation
    4. Validate no regression
    5. Update population

    Implementation Requirements:
    - Selection (tournament, etc.)
    - Crossover (combine genes)
    - Mutation (vary genes)
    - Population management
    """

    def select(
        self,
        population: list[Genome],
        fitness_scores: dict[str, float],
    ) -> list[Genome]:
        """Select genomes for reproduction."""
        ...

    def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
    ) -> Genome:
        """Create offspring from two parents."""
        ...

    def mutate(
        self,
        genome: Genome,
        mutation_rate: float = 0.1,
    ) -> Genome:
        """Mutate a genome."""
        ...

    def evolve_generation(
        self,
        population: list[Genome],
        fitness_scores: dict[str, float],
    ) -> list[Genome]:
        """Complete one evolution cycle."""
        ...


# =============================================================================
# TOOL PROTOCOLS
# =============================================================================


@runtime_checkable
class Tool(Protocol):
    """
    A tool I can use to interact with the world.

    Tools extend my capabilities—they let me:
    - Search and read files
    - Execute commands
    - Verify state
    - Persist data

    Implementation Requirements:
    - Has a name
    - Has a description
    - Defines parameters
    - Executes with structured result
    """

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def description(self) -> str:
        """What this tool does."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """Parameter schema (JSON schema format)."""
        ...

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool.

        Returns structured result with at least:
            {"success": bool, "result": Any, "error": str | None}
        """
        ...


@runtime_checkable
class SearchTool(Tool, Protocol):
    """
    A tool for searching content.

    Specialized search tools provide:
    - Query interpretation
    - Query expansion
    - Ranked results
    - Metadata for verification

    Implementation Requirements:
    - All Tool requirements
    - Returns ranked results
    - Provides query interpretation
    """

    async def search(
        self,
        query: str,
        top_n: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute search.

        Returns:
            {
                "success": bool,
                "results": [{"path", "snippet", "score", ...}],
                "query_interpreted": str,
                "total_found": int,
            }
        """
        ...


# =============================================================================
# METRICS PROTOCOLS
# =============================================================================


@runtime_checkable
class MetricsCollector(Protocol):
    """
    Collects metrics from all system layers.

    Metrics are essential for:
    - Fitness evaluation
    - Bottleneck detection
    - Evolution targeting
    - Dashboard visualization

    Implementation Requirements:
    - Record data points
    - Compute aggregates
    - Provide time series
    """

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric data point."""
        ...

    def get_aggregate(
        self,
        name: str,
        aggregation: Literal["sum", "avg", "min", "max", "count"],
        since: datetime | None = None,
    ) -> float:
        """Get aggregate of a metric."""
        ...

    def get_time_series(
        self,
        name: str,
        since: datetime | None = None,
    ) -> list[tuple[datetime, float]]:
        """Get time series for a metric."""
        ...


# =============================================================================
# PERSISTENCE PROTOCOLS
# =============================================================================


@runtime_checkable
class Persister(Protocol):
    """
    Persists data to durable storage.

    Everything important gets persisted:
    - Graph of Thought
    - Cognitive state
    - Strategy population
    - Execution traces

    Implementation Requirements:
    - Save and load
    - Support versioning
    - Handle corruption gracefully
    """

    def save(self, key: str, data: Serializable) -> None:
        """Save data with key."""
        ...

    def load(self, key: str) -> Serializable | None:
        """Load data by key. Returns None if not found."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    def delete(self, key: str) -> bool:
        """Delete by key. Returns True if deleted."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with prefix."""
        ...


@runtime_checkable
class WriteAheadLog(Protocol):
    """
    Write-ahead log for durability.

    Before any mutation, log the intent. If crash occurs,
    replay the log to recover.

    Implementation Requirements:
    - Append log entries
    - Replay from position
    - Truncate old entries
    """

    def append(self, entry: dict[str, Any]) -> int:
        """Append entry. Returns position."""
        ...

    def replay(
        self,
        from_position: int = 0,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Replay log from position. Yields (position, entry)."""
        ...

    def truncate(self, before_position: int) -> int:
        """Truncate entries before position. Returns count removed."""
        ...

    @property
    def current_position(self) -> int:
        """Current log position."""
        ...


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_implements(obj: Any, protocol: type) -> list[str]:
    """
    Check if an object implements a protocol.

    Returns list of missing methods/properties.
    Useful for debugging implementation issues.
    """
    missing = []

    # Get protocol methods
    for name in dir(protocol):
        if name.startswith("_"):
            continue

        protocol_attr = getattr(protocol, name, None)
        if protocol_attr is None:
            continue

        # Check if it's a method or property
        if callable(protocol_attr) or isinstance(protocol_attr, property):
            if not hasattr(obj, name):
                missing.append(name)

    return missing
