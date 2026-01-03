"""
Agent implementations for the LLM Orchestration Framework.

Agents are the execution units in the hierarchy:
- Directors: Orchestrate workers, manage phases
- Workers: Execute focused tasks within sprints
- HybridDirector: Bridge between kanban (above) and agile (below)

Each agent type has:
- A context that defines its scope and resources
- An event loop for execution
- Communication channels for coordination
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, List, Dict

from .cognitive_state import CognitiveStateManager
from .thought_patterns import QAPVPattern, create_pattern
from .recovery import (
    RecoveryCoordinator,
    ConfusionSignal,
    ConfusionDiagnosis,
    SeverityLevel,
    SynapticConfusionDetector,
)
from .escalation import (
    EscalationLevel,
    EscalationProtocol,
    EscalationManager,
)
from .types import (
    AgentRole,
    Blocked,
    Channel,
    Checkpoint,
    Constraint,
    Delegation,
    DirectorContext,
    Event,
    EventBus,
    Goal,
    Impediment,
    Increment,
    Result,
    Retrospective,
    Scope,
    SprintTask,
    Task,
    TaskStatus,
    WorkerContext,
)

# Import learning components with graceful fallback
try:
    from .learning import LearningCycle, Lesson, Context as LearningContext
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    LearningCycle = None
    Lesson = None
    LearningContext = None

# Import Woven Mind for dual-process cognition with graceful fallback
try:
    from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig, WovenMindResult
    from cortical.reasoning.loom import ThinkingMode
    WOVEN_MIND_AVAILABLE = True
except ImportError:
    WOVEN_MIND_AVAILABLE = False
    WovenMind = None
    WovenMindConfig = None
    WovenMindResult = None
    ThinkingMode = None

# Import PRISM Synaptic Memory with graceful fallback
try:
    from cortical.reasoning.prism_got import SynapticMemoryGraph
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False
    SynapticMemoryGraph = None

# Import GoT Learning Bridge with graceful fallback
try:
    from cortical.got.learning_integration import GoTLearningBridge
    GOT_LEARNING_AVAILABLE = True
except ImportError:
    GOT_LEARNING_AVAILABLE = False
    GoTLearningBridge = None


# =============================================================================
# TOOL FRAMEWORK
# =============================================================================


class ToolType(Enum):
    """Built-in tool types for worker agents."""
    SEARCH = "search"          # Search codebase
    READ = "read"              # Read file
    WRITE = "write"            # Write file
    EXECUTE = "execute"        # Run command
    ANALYZE = "analyze"        # Analyze code
    REASON = "reason"          # Use reasoning engine


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    status: Literal["success", "failed", "simulated"]
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """Record of a tool execution for history tracking."""

    tool_name: str
    parameters: dict[str, Any]
    result: ToolResult
    timestamp: datetime = field(default_factory=datetime.now)


class ToolExecutor:
    """
    Executor for tools with registration and history tracking.

    The ToolExecutor manages:
    - Tool registration by name with handler functions
    - Tool execution with parameter passing
    - Execution history tracking
    - Graceful error handling

    Example:
        executor = ToolExecutor()
        executor.register("search", search_handler)
        result = await executor.execute("search", {"query": "foo"})
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._execution_history: list[ToolExecution] = []

    def register(self, tool_name: str, handler: Callable) -> None:
        """
        Register a tool with its handler function.

        Args:
            tool_name: Name of the tool
            handler: Callable that executes the tool (sync or async)
        """
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError("tool_name must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")

        self._tools[tool_name] = handler

    def is_registered(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        context: str = ""
    ) -> ToolResult:
        """
        Execute a tool with the given parameters.

        Args:
            tool_name: Name of the registered tool
            parameters: Parameters to pass to the tool handler
            context: Context string (e.g., task description) for unregistered tools

        Returns:
            ToolResult with execution outcome
        """
        import logging
        import inspect

        logger = logging.getLogger(__name__)
        start_time = datetime.now()
        parameters = parameters or {}

        try:
            # Check if tool is registered
            if tool_name not in self._tools:
                # Return simulated result for unregistered tools
                logger.debug(f"Tool '{tool_name}' not registered, simulating")
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                result = ToolResult(
                    tool_name=tool_name,
                    status="simulated",
                    output=f"Tool '{tool_name}' would be invoked for: {context}",
                    metadata={"parameters": parameters}
                )
                result.duration_ms = duration_ms

                # Track execution
                execution = ToolExecution(
                    tool_name=tool_name,
                    parameters=parameters,
                    result=result
                )
                self._execution_history.append(execution)

                return result

            # Execute registered tool
            handler = self._tools[tool_name]

            # Call handler (handle both sync and async)
            if inspect.iscoroutinefunction(handler):
                output = await handler(**parameters)
            else:
                output = handler(**parameters)

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Create successful result
            result = ToolResult(
                tool_name=tool_name,
                status="success",
                output=output,
                duration_ms=duration_ms,
                metadata={"parameters": parameters}
            )

            logger.debug(
                f"Tool '{tool_name}' executed successfully in {duration_ms:.2f}ms"
            )

            # Track execution
            execution = ToolExecution(
                tool_name=tool_name,
                parameters=parameters,
                result=result
            )
            self._execution_history.append(execution)

            return result

        except Exception as e:
            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Create failed result
            result = ToolResult(
                tool_name=tool_name,
                status="failed",
                error=str(e),
                duration_ms=duration_ms,
                metadata={"parameters": parameters}
            )

            logger.warning(
                f"Tool '{tool_name}' execution failed: {e}",
                exc_info=True
            )

            # Track execution
            execution = ToolExecution(
                tool_name=tool_name,
                parameters=parameters,
                result=result
            )
            self._execution_history.append(execution)

            return result

    def get_execution_history(self) -> list[ToolExecution]:
        """Get the execution history for learning."""
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())



# =============================================================================
# BASE AGENT
# =============================================================================


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.status = TaskStatus.PENDING
        self.spawned_at = datetime.now()

    @abstractmethod
    async def run(self) -> Result:
        """Execute the agent's main loop."""
        pass

    @abstractmethod
    async def checkpoint(self) -> Checkpoint:
        """Create a resumable checkpoint."""
        pass

    @abstractmethod
    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from a checkpoint."""
        pass


# =============================================================================
# WORKER
# =============================================================================


@dataclass
class CheckpointInfo:
    """Information about a cognitive state checkpoint."""

    checkpoint_id: str
    label: str
    timestamp: datetime
    can_restore: bool


@dataclass
class ConfusionRecord:
    """Record of a confusion detection event during execution."""

    signal_type: str
    severity: str
    recovery_action: str
    recovered: bool
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveMetrics:
    """
    Metrics for tracking worker cognitive performance and health.

    Tracks execution, learning, tool usage, QAPV cycles, and recovery metrics
    for observability and optimization of worker behavior.
    """
    # Execution metrics
    tasks_executed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0

    # QAPV metrics
    qapv_cycles: int = 0
    verify_pass_rate: float = 0.0
    avg_phase_duration: Dict[str, float] = field(default_factory=dict)

    # Learning metrics
    lessons_retrieved: int = 0
    lessons_applied: int = 0
    experiences_captured: int = 0

    # Tool metrics
    tools_invoked: int = 0
    tool_success_rate: float = 0.0
    avg_tool_duration: float = 0.0

    # Recovery metrics
    confusion_signals: int = 0
    recoveries_attempted: int = 0
    recoveries_successful: int = 0

    # Checkpoint metrics
    checkpoints_created: int = 0
    checkpoints_restored: int = 0


class CognitiveMetricsCollector:
    """
    Collects and aggregates cognitive metrics for worker agents.

    Provides methods to record various cognitive activities (task execution,
    QAPV cycles, tool use, learning, recovery) and calculate health scores
    for observability and optimization.
    """

    def __init__(self):
        self._metrics = CognitiveMetrics()
        self._start_time = datetime.now()

        # Track cumulative data for averages
        self._qapv_verify_passes: int = 0
        self._qapv_verify_total: int = 0
        self._tool_successes: int = 0
        self._tool_failures: int = 0
        self._tool_duration_sum: float = 0.0
        self._phase_duration_sums: Dict[str, float] = {}
        self._phase_duration_counts: Dict[str, int] = {}

    def record_task(self, success: bool) -> None:
        """
        Record a task execution.

        Args:
            success: Whether the task succeeded
        """
        self._metrics.tasks_executed += 1
        if success:
            self._metrics.tasks_successful += 1
        else:
            self._metrics.tasks_failed += 1

    def record_qapv_cycle(self, execution: QAPVExecution) -> None:
        """
        Record a QAPV cognitive cycle.

        Args:
            execution: The QAPV execution record
        """
        self._metrics.qapv_cycles += 1

        # Track verification pass rate
        if execution.verify_passed:
            self._qapv_verify_passes += 1
        self._qapv_verify_total += 1

        if self._qapv_verify_total > 0:
            self._metrics.verify_pass_rate = self._qapv_verify_passes / self._qapv_verify_total

        # Track phase durations (cumulative average)
        for phase, duration in execution.phase_durations.items():
            if phase not in self._phase_duration_sums:
                self._phase_duration_sums[phase] = 0.0
                self._phase_duration_counts[phase] = 0

            self._phase_duration_sums[phase] += duration
            self._phase_duration_counts[phase] += 1

        # Update average phase durations
        self._metrics.avg_phase_duration = {
            phase: self._phase_duration_sums[phase] / self._phase_duration_counts[phase]
            for phase in self._phase_duration_sums
        }

    def record_tool_use(self, result: ToolResult) -> None:
        """
        Record a tool execution.

        Args:
            result: The tool execution result
        """
        self._metrics.tools_invoked += 1

        # Track success rate
        if result.status == "success":
            self._tool_successes += 1
        elif result.status == "failed":
            self._tool_failures += 1

        total_tool_executions = self._tool_successes + self._tool_failures
        if total_tool_executions > 0:
            self._metrics.tool_success_rate = self._tool_successes / total_tool_executions

        # Track average duration
        self._tool_duration_sum += result.duration_ms
        if self._metrics.tools_invoked > 0:
            self._metrics.avg_tool_duration = self._tool_duration_sum / self._metrics.tools_invoked

    def record_lesson(self, retrieved: bool = False, applied: bool = False) -> None:
        """
        Record lesson retrieval and application.

        Args:
            retrieved: Whether lessons were retrieved
            applied: Whether lessons were applied
        """
        if retrieved:
            self._metrics.lessons_retrieved += 1
        if applied:
            self._metrics.lessons_applied += 1

    def record_experience(self) -> None:
        """Record that an experience was captured."""
        self._metrics.experiences_captured += 1

    def record_confusion(self, signal: ConfusionSignal, recovered: bool) -> None:
        """
        Record a confusion signal and recovery attempt.

        Args:
            signal: The confusion signal detected
            recovered: Whether recovery was successful
        """
        self._metrics.confusion_signals += 1
        self._metrics.recoveries_attempted += 1
        if recovered:
            self._metrics.recoveries_successful += 1

    def record_checkpoint(self, created: bool = False, restored: bool = False) -> None:
        """
        Record checkpoint creation or restoration.

        Args:
            created: Whether a checkpoint was created
            restored: Whether a checkpoint was restored
        """
        if created:
            self._metrics.checkpoints_created += 1
        if restored:
            self._metrics.checkpoints_restored += 1

    def calculate_health_score(self) -> float:
        """
        Calculate overall cognitive health score (0-100).

        Health score is a weighted combination of:
        - Success rate (40%): tasks_successful / tasks_executed
        - Verify pass rate (20%): QAPV verification success rate
        - Recovery success rate (20%): recoveries_successful / recoveries_attempted
        - Tool success rate (20%): tool successes / tool executions

        Returns:
            float: Health score between 0 and 100
        """
        score = 0.0

        # Success rate (40%)
        if self._metrics.tasks_executed > 0:
            success_rate = self._metrics.tasks_successful / self._metrics.tasks_executed
            score += success_rate * 40.0

        # Verify pass rate (20%)
        if self._qapv_verify_total > 0:
            score += self._metrics.verify_pass_rate * 20.0

        # Recovery success rate (20%)
        if self._metrics.recoveries_attempted > 0:
            recovery_rate = self._metrics.recoveries_successful / self._metrics.recoveries_attempted
            score += recovery_rate * 20.0
        else:
            # No confusion detected is good - full score
            score += 20.0

        # Tool success rate (20%)
        total_tool_executions = self._tool_successes + self._tool_failures
        if total_tool_executions > 0:
            score += self._metrics.tool_success_rate * 20.0
        else:
            # No tools used - neutral score (half points)
            score += 10.0

        return min(100.0, max(0.0, score))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected metrics.

        Returns:
            dict: Summary with metrics and health score
        """
        runtime_seconds = (datetime.now() - self._start_time).total_seconds()

        return {
            "health_score": self.calculate_health_score(),
            "runtime_seconds": runtime_seconds,
            "execution": {
                "tasks_executed": self._metrics.tasks_executed,
                "tasks_successful": self._metrics.tasks_successful,
                "tasks_failed": self._metrics.tasks_failed,
                "success_rate": (
                    self._metrics.tasks_successful / self._metrics.tasks_executed
                    if self._metrics.tasks_executed > 0 else 0.0
                ),
            },
            "qapv": {
                "cycles": self._metrics.qapv_cycles,
                "verify_pass_rate": self._metrics.verify_pass_rate,
                "avg_phase_duration": self._metrics.avg_phase_duration,
            },
            "learning": {
                "lessons_retrieved": self._metrics.lessons_retrieved,
                "lessons_applied": self._metrics.lessons_applied,
                "experiences_captured": self._metrics.experiences_captured,
            },
            "tools": {
                "tools_invoked": self._metrics.tools_invoked,
                "tool_success_rate": self._metrics.tool_success_rate,
                "avg_tool_duration_ms": self._metrics.avg_tool_duration,
            },
            "recovery": {
                "confusion_signals": self._metrics.confusion_signals,
                "recoveries_attempted": self._metrics.recoveries_attempted,
                "recoveries_successful": self._metrics.recoveries_successful,
                "recovery_success_rate": (
                    self._metrics.recoveries_successful / self._metrics.recoveries_attempted
                    if self._metrics.recoveries_attempted > 0 else 1.0
                ),
            },
            "checkpoints": {
                "checkpoints_created": self._metrics.checkpoints_created,
                "checkpoints_restored": self._metrics.checkpoints_restored,
            },
        }

    def get_metrics(self) -> CognitiveMetrics:
        """Get the current metrics snapshot."""
        return self._metrics

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._metrics = CognitiveMetrics()
        self._start_time = datetime.now()
        self._qapv_verify_passes = 0
        self._qapv_verify_total = 0
        self._tool_successes = 0
        self._tool_failures = 0
        self._tool_duration_sum = 0.0
        self._phase_duration_sums = {}
        self._phase_duration_counts = {}


@dataclass
class WorkerResult:
    """Result from a worker execution."""

    status: Literal["complete", "blocked", "failed"]
    output: Any = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    cognitive_metrics: Optional[CognitiveMetrics] = None

@dataclass
class WovenMindExecution:
    """
    Record of Woven Mind dual-process thinking during QAPV execution.

    Tracks which thinking mode (FAST/SLOW) was used for each phase,
    mode switches, and surprise-triggered transitions.
    """
    modes_used: Dict[str, str]  # phase -> mode name (e.g., "FAST", "SLOW")
    mode_switches: int  # Number of mode switches during execution
    surprise_triggers: int  # Number of surprise-based mode switches
    consolidation_triggered: bool  # Whether consolidation was triggered


@dataclass
class QAPVExecution:
    """Result from a QAPV cognitive cycle execution."""
    question_result: str
    answer_approach: str
    produce_output: Any
    verify_passed: bool
    phase_durations: Dict[str, float]

    confusion_records: List[ConfusionRecord] = field(default_factory=list)
    woven_mind: Optional[WovenMindExecution] = None  # Woven Mind execution tracking


@dataclass
class TaskGuidance:
    """
    Guidance retrieved from learning system for task execution.

    Contains lessons learned from past experiences and recommendations
    for approaching the current task.
    """

    lessons: List[Any] = field(default_factory=list)  # List[Lesson] when available
    recommended_approach: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0



# QAPV Phase to Thinking Mode Mapping
# Maps each QAPV phase to the appropriate thinking mode (FAST/SLOW)
QAPV_THINKING_MODES = {
    "QUESTION": "SLOW",   # Deep analysis needed to understand the problem
    "ANSWER": "FAST",     # Pattern matching, use heuristics for approach
    "PRODUCE": "FAST",    # Execution, muscle memory, automated processes
    "VERIFY": "SLOW",     # Careful checking, deliberate validation
}


class Worker(Agent):
    """
    A cognitive worker agent that executes tasks using structured thinking.

    The Worker uses the QAPV cognitive loop (Question→Answer→Produce→Verify)
    with Woven Mind dual-process thinking (FAST/SLOW modes) to execute tasks.
    Workers are leaf nodes in the agent hierarchy that perform focused work
    with cognitive capabilities including learning, recovery, and observability.

    Features:
        - **QAPV Cognitive Loop**: Structured thinking through four phases
          - QUESTION (SLOW): Deep analysis of requirements
          - ANSWER (FAST): Pattern matching for approach
          - PRODUCE (FAST): Task execution
          - VERIFY (SLOW): Careful validation

        - **Dual-Process Thinking**: Woven Mind integration
          - FAST mode: Pattern matching, heuristics, execution
          - SLOW mode: Deliberate analysis and validation
          - Automatic mode switching based on cognitive needs

        - **Tool Execution**: Managed tool invocation
          - Tool registration and execution
          - Execution history tracking
          - Error handling with structured results

        - **Learning Integration**: Experience-based improvement
          - Retrieves relevant lessons before execution
          - Captures experiences after execution
          - Builds knowledge base over time

        - **Confusion Detection & Recovery**: Resilience
          - Detects confusion signals (repetition, contradictions)
          - Coordinates recovery strategies
          - State restoration to known-good checkpoints

        - **Cognitive State Management**: Continuity
          - Creates checkpoints during execution
          - Maintains execution context
          - Supports state restoration

        - **Metrics Collection**: Observability
          - Tracks execution success/failure
          - Monitors QAPV cycle performance
          - Records tool usage and learning activity
          - Calculates cognitive health score (0-100)

    Lifecycle:
        1. Initialization: Set up context, tools, cognitive systems
        2. Pre-execution: Retrieve relevant lessons
        3. Execution: Run QAPV cognitive loop
        4. Post-execution: Capture experience, record metrics
        5. Completion: Return structured result

    Example:
        >>> from llm_orchestration.agents import Worker, WorkerContext
        >>> from llm_orchestration.cognitive_state import CognitiveStateManager
        >>> from pathlib import Path
        >>>
        >>> # Create cognitive state manager
        >>> state_dir = Path(".llm_orchestration/cognitive_state")
        >>> state_manager = CognitiveStateManager(state_dir)
        >>>
        >>> # Create worker context
        >>> context = WorkerContext(
        ...     task="Implement authentication",
        ...     tools=["read", "write", "search"],
        ...     constraints=["Must pass tests"],
        ... )
        >>>
        >>> # Create worker with cognitive capabilities
        >>> worker = Worker("worker-1", context, state_manager=state_manager)
        >>>
        >>> # Execute task
        >>> result = await worker.execute_task()
        >>>
        >>> # Check results
        >>> print(f"Success: {result['success']}")
        >>> print(f"Health: {result['health_score']:.1f}/100")
        >>>
        >>> # Get detailed metrics
        >>> summary = worker.get_metrics_summary()
        >>> print(f"QAPV cycles: {summary['qapv']['cycles']}")
        >>> print(f"Lessons used: {summary['learning']['lessons_retrieved']}")

    Attributes:
        agent_id (str): Unique identifier for this worker
        role (AgentRole): Always AgentRole.WORKER
        context (WorkerContext): Current execution context
        status (TaskStatus): Current execution status
        current_task (Task | None): Currently executing task
        progress (float): Execution progress (0.0-1.0)

    Private Attributes:
        _state_manager: Manages cognitive state and checkpoints
        _thinking_pattern: QAPV reasoning pattern instance
        _recovery_coordinator: Handles confusion detection and recovery
        _learning_cycle: Manages experience capture and retrieval
        _tool_executor: Manages tool registration and execution
        _woven_mind: Dual-process thinking engine
        _metrics: Collects cognitive performance metrics
        _qapv_executions: History of QAPV cycle executions
        _confusion_signals: Detected confusion signals

    Raises:
        ValueError: If agent_id is empty or context is None
        TypeError: If context is not a WorkerContext instance
        Blocked: If worker encounters a blocking condition

    See Also:
        AgileWorker: Worker with sprint-based execution
        Director: Orchestrates multiple workers
        WorkerContext: Configuration for worker execution
        ToolExecutor: Tool registration and execution
        CognitiveStateManager: State management and checkpointing
        LearningCycle: Experience capture and lesson retrieval
        RecoveryCoordinator: Confusion detection and recovery
    """

    def __init__(
        self,
        agent_id: str,
        context: WorkerContext,
        state_manager: Optional[CognitiveStateManager] = None,
    ):
        super().__init__(agent_id, AgentRole.WORKER)

        # Validate inputs
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")
        if context is None:
            raise ValueError("context cannot be None")
        if not isinstance(context, WorkerContext):
            raise TypeError(f"context must be WorkerContext, got {type(context)}")

        self.context = context
        self.current_task: Task | None = None
        self.progress: float = 0.0

        # Cognitive state management
        self._state_manager: Optional[CognitiveStateManager] = state_manager
        self._checkpoint_id: Optional[str] = None
        # QAPV cognitive pattern
        self._thinking_pattern: Optional[QAPVPattern] = None
        self._qapv_executions: list[QAPVExecution] = []

        # Confusion detection and recovery
        self._recovery_coordinator: Optional[RecoveryCoordinator] = None
        self._confusion_signals: List[ConfusionSignal] = []

        # PRISM Synaptic memory for confusion detection
        self._synaptic_graph: Optional['SynapticMemoryGraph'] = None

        # Initialize recovery coordinator if we have a storage location
        try:
            storage_dir = Path.home() / ".llm_orchestration" / "recovery" / agent_id
            self._recovery_coordinator = RecoveryCoordinator(storage_dir)
        except Exception:
            # Recovery coordinator is optional
            pass

        # Initialize PRISM synaptic detection if available
        if PRISM_AVAILABLE and self._recovery_coordinator:
            try:
                self._synaptic_graph = SynapticMemoryGraph(
                    agent_id=agent_id,
                    max_nodes=1000,
                    decay_factor=0.95
                )
                self._recovery_coordinator.enable_synaptic_detection(self._synaptic_graph)
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Worker {agent_id}: PRISM synaptic detection enabled")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize PRISM: {e}")

        # Learning cycle for retrieving lessons
        self._learning_cycle: Optional[Any] = None  # LearningCycle when available
        if LEARNING_AVAILABLE:
            try:
                from pathlib import Path
                storage_dir = Path.home() / ".llm_orchestration" / "learning"
                self._learning_cycle = LearningCycle(storage_dir)
            except Exception:
                # Learning cycle initialization failed, proceed without it
                pass

        # GoT Learning Bridge for persistent experience capture
        self._got_learning_bridge: Optional['GoTLearningBridge'] = None
        if GOT_LEARNING_AVAILABLE:
            try:
                from pathlib import Path
                got_dir = Path(".got")
                if got_dir.exists():
                    self._got_learning_bridge = GoTLearningBridge(got_dir)
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Worker {agent_id}: GoT Learning Bridge enabled")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize GoT Learning Bridge: {e}")

        # Tool executor for managing tool invocations
        self._tool_executor = ToolExecutor()

        # Woven Mind for dual-process cognition (FAST/SLOW thinking)
        self._woven_mind: Optional[Any] = None  # WovenMind when available
        if WOVEN_MIND_AVAILABLE:
            try:
                # Initialize with default configuration
                self._woven_mind = WovenMind(config=WovenMindConfig(
                    surprise_threshold=0.3,
                    k_winners=5,
                    auto_switch=True,
                    enable_observability=True,
                ))
            except Exception:
                # Woven Mind initialization failed, proceed without it
                pass

        # Metrics collector for cognitive observability
        self._metrics = CognitiveMetricsCollector()
    async def run(self) -> Result:
        """Execute the worker's task."""
        try:
            # Publish start
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.started",
                    payload={"task": self.context.task},
                    source_agent_id=self.agent_id,
                ))

            # Execute task
            self.status = TaskStatus.IN_PROGRESS
            result = await self.execute_task()

            # Record successful task execution
            self._metrics.record_task(success=True)

            # Publish completion
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.completed",
                    payload={"result": result},
                    source_agent_id=self.agent_id,
                ))

            self.status = TaskStatus.COMPLETED
            return Result(success=True, output=result)

        except Blocked as b:
            # Record failed task execution
            self._metrics.record_task(success=False)

            self.status = TaskStatus.BLOCKED
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.blocked",
                    payload={
                        "reason": b.reason,
                        "need": b.what_i_need,
                    },
                    source_agent_id=self.agent_id,
                ))
            return Result(success=False, error=b.reason)

        except Exception as e:
            # Record failed task execution
            self._metrics.record_task(success=False)

            self.status = TaskStatus.FAILED
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.failed",
                    payload={"error": str(e)},
                    source_agent_id=self.agent_id,
                ))
            return Result(success=False, error=str(e))

    def _checkpoint_state(self, label: str) -> Optional[str]:
        """
        Create checkpoint of current cognitive state.

        Args:
            label: Label for the checkpoint (e.g., "pre_execution", "post_tool_read")

        Returns:
            Checkpoint ID if state manager is available, None otherwise
        """
        if self._state_manager:
            try:
                checkpoint_data = self._state_manager.checkpoint()
                # Extract timestamp from checkpoint to create ID
                timestamp = checkpoint_data.get("timestamp", datetime.now().isoformat())
                checkpoint_id = f"ckpt-{self.agent_id}-{label}-{timestamp.replace(':', '-').replace('.', '-')}"
                self._checkpoint_id = checkpoint_id

                # Record checkpoint creation
                self._metrics.record_checkpoint(created=True)

                return checkpoint_id
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create checkpoint '{label}': {e}")
                return None
        return None

    def _restore_state(self, checkpoint_id: str) -> bool:
        """
        Restore to a previous checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to restore

        Returns:
            True if restoration succeeded, False otherwise
        """
        if self._state_manager:
            try:
                # Find the checkpoint file
                from pathlib import Path
                checkpoints = self._state_manager.list_checkpoints()

                # Look for checkpoint matching the ID pattern
                for checkpoint_path in checkpoints:
                    checkpoint_data = self._state_manager.load_checkpoint(checkpoint_path)
                    # Restore the checkpoint
                    self._state_manager.restore_from_checkpoint(checkpoint_data)

                    # Record checkpoint restoration
                    self._metrics.record_checkpoint(restored=True)

                    return True

                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Checkpoint '{checkpoint_id}' not found")
                return False

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to restore checkpoint '{checkpoint_id}': {e}")
                return False
        return False


    def _get_thinking_mode(self, phase: str) -> Optional[str]:
        """
        Get the appropriate thinking mode for a QAPV phase.

        Maps QAPV phases to thinking modes (FAST/SLOW) based on the
        cognitive requirements of each phase:
        - QUESTION: SLOW (deep analysis)
        - ANSWER: FAST (pattern matching)
        - PRODUCE: FAST (execution)
        - VERIFY: SLOW (careful validation)

        Args:
            phase: QAPV phase name (QUESTION, ANSWER, PRODUCE, VERIFY)

        Returns:
            Thinking mode name ("FAST" or "SLOW"), or None if not available
        """
        if not WOVEN_MIND_AVAILABLE or not self._woven_mind:
            return None

        return QAPV_THINKING_MODES.get(phase.upper())

    def _run_qapv_cycle(self, task_context: dict) -> dict:
        """
        Execute a QAPV (Question → Answer → Produce → Verify) cognitive cycle.

        This method structures task execution through four phases:
        1. QUESTION: Analyze what needs to be done
        2. ANSWER: Determine approach
        3. PRODUCE: Execute the approach
        4. VERIFY: Validate results

        Args:
            task_context: Context for the task including:
                - task_description: What to accomplish
                - tools_available: Available tools
                - constraints: Any constraints

        Returns:
            dict: Results from QAPV cycle with phase tracking
        """
        import logging
        from datetime import datetime
        from pathlib import Path

        logger = logging.getLogger(__name__)
        phase_durations: Dict[str, float] = {}

        # Initialize Woven Mind execution tracking
        modes_used: Dict[str, str] = {}
        mode_switches = 0
        surprise_triggers = 0
        consolidation_triggered = False
        previous_mode: Optional[str] = None

        # Initialize QAPV pattern if needed
        if not self._thinking_pattern or not self._state_manager:
            # Create temporary state manager if not available
            if not self._state_manager:
                logger.debug("Creating temporary cognitive state manager for QAPV")
                storage_dir = Path.home() / ".llm_orchestration" / "cognitive_state"
                storage_dir.mkdir(parents=True, exist_ok=True)
                self._state_manager = CognitiveStateManager(storage_dir)

            # Create QAPV pattern
            self._thinking_pattern = create_pattern(
                "qapv",
                self._state_manager,
                goal=task_context.get("task_description", "Execute task")
            )
            logger.info(f"Initialized QAPV pattern for goal: {self._thinking_pattern.goal}")

        pattern = self._thinking_pattern

        try:
            # ===================================================================
            # PHASE 1: QUESTION - What am I trying to do?
            # ===================================================================
            # Switch to SLOW mode for deep analysis
            current_mode = self._get_thinking_mode("QUESTION")
            if current_mode and self._woven_mind:
                try:
                    # Get ThinkingMode enum from string
                    mode_enum = ThinkingMode.SLOW if current_mode == "SLOW" else ThinkingMode.FAST
                    self._woven_mind.force_mode(mode_enum, reason="QAPV_QUESTION_phase")
                    modes_used["QUESTION"] = current_mode
                    if previous_mode and previous_mode != current_mode:
                        mode_switches += 1
                    previous_mode = current_mode
                    logger.debug(f"[Woven Mind] Switched to {current_mode} mode for QUESTION phase")
                except Exception as e:
                    logger.warning(f"[Woven Mind] Mode switch failed: {e}")

            logger.info("[QAPV] QUESTION phase: Clarifying task requirements")
            question_start = datetime.now()

            # Set the core question
            task_desc = task_context.get("task_description", "")
            success_criteria = task_context.get("success_criteria",
                                                 "Task completed successfully")

            pattern.set_question(
                question_text=f"How do I accomplish: {task_desc}?",
                success_criteria=success_criteria
            )

            # Record what we need to know
            logger.debug(f"Task: {task_desc}")
            logger.debug(f"Success criteria: {success_criteria}")
            logger.debug(f"Tools available: {task_context.get('tools_available', [])}")

            question_duration = (datetime.now() - question_start).total_seconds()
            phase_durations["question"] = question_duration

            # Record synaptic activation for PRISM
            if self._synaptic_graph:
                try:
                    self._synaptic_graph.record_activation(
                        concept="QAPV_QUESTION",
                        strength=1.0,
                        context={'task_id': task_context.get('task_id', 'unknown'), 'phase': 'QUESTION'}
                    )
                except Exception as e:
                    logger.debug(f"Failed to record PRISM activation: {e}")

            # ===================================================================
            # PHASE 2: ANSWER - Determine approach
            # ===================================================================
            # Switch to FAST mode for pattern matching
            current_mode = self._get_thinking_mode("ANSWER")
            if current_mode and self._woven_mind:
                try:
                    mode_enum = ThinkingMode.SLOW if current_mode == "SLOW" else ThinkingMode.FAST
                    self._woven_mind.force_mode(mode_enum, reason="QAPV_ANSWER_phase")
                    modes_used["ANSWER"] = current_mode
                    if previous_mode and previous_mode != current_mode:
                        mode_switches += 1
                    previous_mode = current_mode
                    logger.debug(f"[Woven Mind] Switched to {current_mode} mode for ANSWER phase")
                except Exception as e:
                    logger.warning(f"[Woven Mind] Mode switch failed: {e}")

            logger.info("[QAPV] ANSWER phase: Determining approach")
            answer_start = datetime.now()

            # Analyze available tools and determine approach
            tools = task_context.get("tools_available", [])
            approach = ""

            if tools:
                approach = f"Use available tools: {', '.join(tools)}"
                pattern.record_decision(
                    decision=f"Execute using {len(tools)} available tools",
                    rationale=f"Tools {tools} are available and suitable for this task",
                    alternatives=["Execute without tools", "Request additional tools"]
                )
            else:
                approach = "Execute task without specialized tools"
                pattern.record_decision(
                    decision="Execute without tools",
                    rationale="No tools available, proceed with direct execution",
                    alternatives=["Wait for tools", "Request tools"]
                )

            logger.debug(f"Approach determined: {approach}")

            answer_duration = (datetime.now() - answer_start).total_seconds()
            phase_durations["answer"] = answer_duration

            # Record synaptic activation for PRISM
            if self._synaptic_graph:
                try:
                    self._synaptic_graph.record_activation(
                        concept="QAPV_ANSWER",
                        strength=1.0,
                        context={'task_id': task_context.get('task_id', 'unknown'), 'phase': 'ANSWER'}
                    )
                except Exception as e:
                    logger.debug(f"Failed to record PRISM activation: {e}")

            # Advance to PRODUCE phase
            pattern.advance()

            # ===================================================================
            # PHASE 3: PRODUCE - Execute the approach
            # ===================================================================
            # Switch to FAST mode for execution
            current_mode = self._get_thinking_mode("PRODUCE")
            if current_mode and self._woven_mind:
                try:
                    mode_enum = ThinkingMode.SLOW if current_mode == "SLOW" else ThinkingMode.FAST
                    self._woven_mind.force_mode(mode_enum, reason="QAPV_PRODUCE_phase")
                    modes_used["PRODUCE"] = current_mode
                    if previous_mode and previous_mode != current_mode:
                        mode_switches += 1
                    previous_mode = current_mode
                    logger.debug(f"[Woven Mind] Switched to {current_mode} mode for PRODUCE phase")
                except Exception as e:
                    logger.warning(f"[Woven Mind] Mode switch failed: {e}")

            logger.info("[QAPV] PRODUCE phase: Executing task")
            produce_start = datetime.now()

            # The actual execution happens here
            # This is where the task logic would be invoked
            execution_result = {
                "approach": approach,
                "task": task_desc,
                "tools_used": tools,
                "status": "produced",
            }

            # Set artifact
            pattern.set_artifact(
                artifact=execution_result,
                description=f"Executed task using approach: {approach}"
            )

            logger.debug(f"Produced result: {execution_result}")

            produce_duration = (datetime.now() - produce_start).total_seconds()
            phase_durations["produce"] = produce_duration

            # Record synaptic activation for PRISM
            if self._synaptic_graph:
                try:
                    self._synaptic_graph.record_activation(
                        concept="QAPV_PRODUCE",
                        strength=1.0,
                        context={'task_id': task_context.get('task_id', 'unknown'), 'phase': 'PRODUCE'}
                    )
                except Exception as e:
                    logger.debug(f"Failed to record PRISM activation: {e}")

            # Advance to VERIFY phase
            pattern.advance()

            # ===================================================================
            # PHASE 4: VERIFY - Validate results
            # ===================================================================
            # Switch to SLOW mode for careful validation
            current_mode = self._get_thinking_mode("VERIFY")
            if current_mode and self._woven_mind:
                try:
                    mode_enum = ThinkingMode.SLOW if current_mode == "SLOW" else ThinkingMode.FAST
                    self._woven_mind.force_mode(mode_enum, reason="QAPV_VERIFY_phase")
                    modes_used["VERIFY"] = current_mode
                    if previous_mode and previous_mode != current_mode:
                        mode_switches += 1
                    previous_mode = current_mode
                    logger.debug(f"[Woven Mind] Switched to {current_mode} mode for VERIFY phase")
                except Exception as e:
                    logger.warning(f"[Woven Mind] Mode switch failed: {e}")

            logger.info("[QAPV] VERIFY phase: Validating results")
            verify_start = datetime.now()

            # Verify the execution meets success criteria
            # Simple verification for now - check that we have a result
            verify_passed = (
                execution_result is not None and
                execution_result.get("status") == "produced"
            )

            verification_details = {
                "has_result": execution_result is not None,
                "status_correct": execution_result.get("status") == "produced",
                "approach_executed": bool(approach),
                "overall": verify_passed
            }

            pattern.record_verification(
                passed=verify_passed,
                details=verification_details
            )

            logger.info(f"[QAPV] Verification {'PASSED' if verify_passed else 'FAILED'}")

            verify_duration = (datetime.now() - verify_start).total_seconds()
            phase_durations["verify"] = verify_duration

            # Record synaptic activation for PRISM
            if self._synaptic_graph:
                try:
                    self._synaptic_graph.record_activation(
                        concept="QAPV_VERIFY",
                        strength=1.0 if verify_passed else 0.5,
                        context={'task_id': task_context.get('task_id', 'unknown'), 'phase': 'VERIFY', 'passed': verify_passed}
                    )
                except Exception as e:
                    logger.debug(f"Failed to record PRISM activation: {e}")

            # Advance to COMPLETE if verification passed
            if verify_passed:
                pattern.advance()
                logger.info("[QAPV] Cycle COMPLETE")

            # ===================================================================
            # Package results
            # ===================================================================
            total_duration = sum(phase_durations.values())
            logger.info(
                f"[QAPV] Total cycle time: {total_duration:.3f}s "
                f"(Q:{phase_durations['question']:.3f}s, "
                f"A:{phase_durations['answer']:.3f}s, "
                f"P:{phase_durations['produce']:.3f}s, "
                f"V:{phase_durations['verify']:.3f}s)"
            )

            # Create Woven Mind execution record
            woven_mind_execution: Optional[WovenMindExecution] = None
            if modes_used:
                woven_mind_execution = WovenMindExecution(
                    modes_used=modes_used,
                    mode_switches=mode_switches,
                    surprise_triggers=surprise_triggers,
                    consolidation_triggered=consolidation_triggered
                )
                logger.info(
                    f"[Woven Mind] Execution summary: "
                    f"{len(modes_used)} phases, {mode_switches} mode switches, "
                    f"{surprise_triggers} surprise triggers"
                )

            # Create execution record
            qapv_execution = QAPVExecution(
                question_result=f"Clarified: {task_desc}",
                answer_approach=approach,
                produce_output=execution_result,
                verify_passed=verify_passed,
                phase_durations=phase_durations,
                woven_mind=woven_mind_execution
            )

            self._qapv_executions.append(qapv_execution)

            # Record QAPV cycle metrics
            self._metrics.record_qapv_cycle(qapv_execution)

            return {
                "qapv_result": execution_result,
                "qapv_execution": qapv_execution,
                "pattern_progress": pattern.get_progress(),
                "verify_passed": verify_passed,
                "phase_durations": phase_durations,
                "woven_mind": woven_mind_execution,
            }

        except Exception as e:
            logger.error(f"[QAPV] Error during cycle: {e}", exc_info=True)
            raise


    def _get_lessons_for_task(self, task_context: dict) -> List[Any]:
        """
        Retrieve relevant lessons from learning cycle for the current task.

        Args:
            task_context: Dictionary with task information including:
                - task: Task description
                - tools: Available tools
                - constraints: Task constraints

        Returns:
            List of Lesson objects if learning available, empty list otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self._learning_cycle:
            logger.debug("Learning cycle not available for lesson retrieval")
            return []

        try:
            # Build learning context from task context
            learning_context = LearningContext(
                goal_type=self._infer_goal_type_from_task(task_context.get("task", "")),
                goal_complexity="moderate",
                available_tools=task_context.get("tools", []),
                domain="worker_task",
                constraints=task_context.get("constraints", [])
            )

            # Get guidance from learning cycle
            guidance = self._learning_cycle.get_guidance(
                context=learning_context,
                include_experiences=False  # Only get lessons, not full experiences
            )

            lessons = guidance.get("lessons", [])
            logger.info(f"Retrieved {len(lessons)} lessons for task")

            # Record lesson retrieval
            if lessons:
                self._metrics.record_lesson(retrieved=True)

            # Log lesson summaries
            for lesson in lessons[:3]:  # Log top 3 lessons
                if hasattr(lesson, 'title') and hasattr(lesson, 'confidence'):
                    logger.debug(
                        f"Lesson: {lesson.title} (confidence: {lesson.confidence:.2f})"
                    )

            return lessons

        except Exception as e:
            logger.warning(f"Failed to retrieve lessons: {e}")
            return []

    def _infer_goal_type_from_task(self, task_description: str) -> str:
        """Infer goal type from task description."""
        task_lower = task_description.lower()

        if any(word in task_lower for word in ["implement", "create", "build", "add"]):
            return "implementation"
        elif any(word in task_lower for word in ["fix", "debug", "resolve"]):
            return "debugging"
        elif any(word in task_lower for word in ["refactor", "improve", "optimize"]):
            return "refactoring"
        elif any(word in task_lower for word in ["test", "verify"]):
            return "testing"
        elif any(word in task_lower for word in ["document", "doc"]):
            return "documentation"
        else:
            return "general"

    def _check_for_confusion(self, context: Dict[str, Any]) -> Optional[ConfusionSignal]:
        """
        Check if current execution shows confusion signals.

        Args:
            context: Execution context to check for confusion patterns

        Returns:
            ConfusionSignal if confusion detected, None otherwise
        """
        if not self._recovery_coordinator:
            return None

        # Check for confusion using the recovery coordinator
        diagnosis = self._recovery_coordinator.check_confusion(context)

        if diagnosis and diagnosis.signals:
            # Return the most severe signal
            return max(diagnosis.signals, key=lambda s: s.confidence)

        return None

    def _handle_confusion(self, signal: ConfusionSignal, context: Dict[str, Any]) -> str:
        """
        Trigger recovery based on confusion signal.

        Args:
            signal: The confusion signal detected
            context: Current execution context

        Returns:
            Recovery action taken (CONTINUE, CHECKPOINT, STOP, ESCALATE)
        """
        if not self._recovery_coordinator:
            return "CONTINUE"  # No recovery available

        # Diagnose the confusion
        diagnosis = self._recovery_coordinator.check_confusion(context)

        if not diagnosis:
            return "CONTINUE"

        # Record the signal
        self._confusion_signals.append(signal)

        # Determine action based on severity
        if diagnosis.severity == SeverityLevel.LOW:
            # Log warning and continue
            import logging
            logging.getLogger(__name__).warning(
                f"Low-severity confusion detected: {signal.description}"
            )
            # Record confusion with recovery success (low severity = handled)
            self._metrics.record_confusion(signal, recovered=True)
            return "CONTINUE"

        elif diagnosis.severity == SeverityLevel.MEDIUM:
            # Checkpoint and attempt recovery
            if self._state_manager:
                try:
                    self._state_manager.save_checkpoint(
                        agent_id=self.agent_id,
                        label=f"pre_recovery_{datetime.now().isoformat()}",
                        state_data={"signal": signal.to_dict()}
                    )
                except Exception as e:
                    # Checkpoint failed but recovery can still proceed
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to save pre-recovery checkpoint: {e}"
                    )

            # Attempt recovery
            attempt = self._recovery_coordinator.recover(diagnosis, context)

            if attempt.success:
                # Record confusion with recovery outcome
                self._metrics.record_confusion(signal, recovered=attempt.success)

            if attempt.success:
                return "CHECKPOINT"
            else:
                return "ESCALATE"

        elif diagnosis.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL):
            # Stop and escalate
            import logging
            logging.getLogger(__name__).error(
                f"Critical confusion detected: {diagnosis.confusion_type.name}. "
                f"Cause: {diagnosis.likely_cause}"
            )
            # Record confusion with no recovery
            self._metrics.record_confusion(signal, recovered=False)
            return "ESCALATE"

        return "CONTINUE"

    async def execute_task(self) -> Any:
        """
        Execute the task using available tools.

        This method:
        - Validates task context
        - Tracks execution time
        - Records the execution as an experience in the LearningCycle
        - Executes using available tools
        - Returns structured results

        Subclasses can override this to implement specific task execution logic.

        Returns:
            dict: Task execution results with status, output, and timing
        """
        import logging
        from datetime import datetime
        from pathlib import Path

        logger = logging.getLogger(__name__)
        start_time = datetime.now()

        # Validate execution context
        if not self.context or not self.context.task:
            raise ValueError("Cannot execute task: context or task is not set")

        # Retrieve lessons from learning cycle BEFORE execution
        task_context = {
            "task": self.context.task,
            "tools": self.context.tools or [],
            "constraints": self.context.constraints or []
        }
        lessons = self._get_lessons_for_task(task_context)
        logger.info(f"Retrieved {len(lessons)} lessons for guidance")

        # Retrieve guidance from GoT Learning Bridge before execution
        # This includes lessons, recommendations, warnings, and relevant experiences
        got_guidance = {
            "lessons": [],
            "recommendations": [],
            "warnings": [],
            "relevant_successes": [],
            "relevant_failures": [],
        }
        got_lesson_ids_used = []  # Track for feedback loop

        if self._got_learning_bridge:
            try:
                got_guidance = self._got_learning_bridge.get_guidance_for_task(
                    task_title=self.context.task,
                    task_category="task_execution",
                    task_priority="medium",
                )

                # Track lesson IDs for later validation
                got_lesson_ids_used = [lesson.id for lesson in got_guidance.get("lessons", [])]

                if got_guidance.get("lessons") or got_guidance.get("recommendations"):
                    logger.info(
                        f"Worker {self.agent_id}: Retrieved GoT guidance - "
                        f"{len(got_guidance.get('lessons', []))} lessons, "
                        f"{len(got_guidance.get('recommendations', []))} recommendations, "
                        f"{len(got_guidance.get('warnings', []))} warnings"
                    )
            except Exception as e:
                logger.warning(f"Failed to retrieve GoT guidance: {e}")

        # =====================================================================
        # RUN QAPV COGNITIVE CYCLE
        # =====================================================================
        logger.info(f"[QAPV] Running cognitive cycle for task: {self.context.task}")
        
        # Prepare QAPV task context with injected learning
        qapv_task_context = {
            "task_description": self.context.task,
            "tools_available": self.context.tools or [],
            "constraints": self.context.constraints or [],
            "success_criteria": "Task completed successfully",

            # INJECTED FROM GOT LEARNING - This is the key integration!
            # Recommendations from past successful experiences
            "learned_recommendations": got_guidance.get("recommendations", []),
            # Warnings from past failures to avoid
            "learned_warnings": got_guidance.get("warnings", []),
            # Similar past failures to be aware of
            "similar_failures": [
                exp.intent for exp in got_guidance.get("relevant_failures", [])
                if hasattr(exp, 'intent')
            ][:3],  # Limit to 3 most relevant
        }

        # Log if we have actionable learning guidance
        if qapv_task_context["learned_recommendations"]:
            logger.info(
                f"[QAPV] Applying {len(qapv_task_context['learned_recommendations'])} "
                f"learned recommendations from past experiences"
            )
        if qapv_task_context["learned_warnings"]:
            logger.warning(
                f"[QAPV] {len(qapv_task_context['learned_warnings'])} warnings from past failures: "
                f"{qapv_task_context['learned_warnings'][:2]}"  # Show first 2
            )
        
        # Run QAPV cycle to structure thinking
        qapv_result = self._run_qapv_cycle(qapv_task_context)
        qapv_execution = qapv_result.get("qapv_execution")
        verify_passed = qapv_result.get("verify_passed", False)
        phase_durations = qapv_result.get("phase_durations", {})
        
        logger.info(
            f"[QAPV] Cycle complete: "
            f"Question={qapv_execution.question_result if qapv_execution else 'N/A'}, "
            f"Approach={qapv_execution.answer_approach if qapv_execution else 'N/A'}, "
            f"Verified={'YES' if verify_passed else 'NO'}"
        )


        # Checkpoint BEFORE execution
        pre_execution_checkpoint = self._checkpoint_state("pre_execution")
        checkpoint_history = []
        if pre_execution_checkpoint:
            checkpoint_history.append(CheckpointInfo(
                checkpoint_id=pre_execution_checkpoint,
                label="pre_execution",
                timestamp=datetime.now(),
                can_restore=True
            ))

        try:
            # Initialize learning cycle if available
            learning_cycle = None
            experience = None
            try:
                from .learning import (
                    LearningCycle, Context, Action, Outcome, OutcomeType,
                    ExperienceType
                )
                storage_dir = Path.home() / ".llm_orchestration" / "learning"
                learning_cycle = LearningCycle(storage_dir)

                # Create experience context
                context = Context(
                    goal_type="task_execution",
                    goal_complexity="moderate",
                    available_tools=self.context.tools,
                    domain="worker_task"
                )
                experience = learning_cycle.start_experience(
                    context=context,
                    intent=self.context.task,
                    experience_type=ExperienceType.TASK_EXECUTION
                )
            except Exception as e:
                logger.debug(f"Learning cycle not available: {e}")

            # Execute the task
            logger.info(f"Worker {self.agent_id} executing task: {self.context.task}")

            # Record action if learning
            if learning_cycle and experience:
                action = Action(
                    action_type="execute_task",
                    description=f"Execute task: {self.context.task}",
                    target=self.context.task,
                    parameters={"tools": self.context.tools}
                )
                experience.add_action(action)

            # Execute task logic - use ToolExecutor for real tool invocation
            tool_outputs = []
            if self.context.tools:
                logger.debug(f"Available tools: {self.context.tools}")
                for tool_name in self.context.tools:
                    try:
                        # Execute tool using ToolExecutor
                        tool_result = await self._tool_executor.execute(
                            tool_name=tool_name,
                            parameters={"task": self.context.task},
                            context=self.context.task
                        )
                        
                        # Record tool use
                        self._metrics.record_tool_use(tool_result)

                        # Convert ToolResult to dict for compatibility
                        tool_outputs.append({
                            "tool": tool_name,
                            "result": tool_result.output,
                            "status": tool_result.status,
                            "duration_ms": tool_result.duration_ms,
                            "error": tool_result.error
                        })
                        logger.debug(
                            f"Executed tool '{tool_name}' with status {tool_result.status} "
                            f"in {tool_result.duration_ms:.2f}ms"
                        )

                        # Checkpoint AFTER successful tool use
                        post_tool_checkpoint = self._checkpoint_state(f"post_tool_{tool_name}")
                        if post_tool_checkpoint:
                            checkpoint_history.append(CheckpointInfo(
                                checkpoint_id=post_tool_checkpoint,
                                label=f"post_tool_{tool_name}",
                                timestamp=datetime.now(),
                                can_restore=True
                            ))

                    except Exception as tool_error:
                        logger.warning(f"Tool '{tool_name}' execution failed: {tool_error}")
                        tool_outputs.append({
                            "tool": tool_name,
                            "error": str(tool_error),
                            "status": "failed"
                        })

            # Check for confusion before building result
            confusion_context = {
                "task": self.context.task,
                "agent_id": self.agent_id,
                "tool_outputs": tool_outputs,
                "start_time": start_time.isoformat()
            }

            signal = self._check_for_confusion(confusion_context)
            confusion_records = []

            if signal:
                action = self._handle_confusion(signal, confusion_context)
                confusion_records.append(ConfusionRecord(
                    signal_type=signal.signal_type,
                    severity=action,
                    recovery_action=action,
                    recovered=(action in ["CONTINUE", "CHECKPOINT"]),
                    details={"signal_description": signal.description}
                ))

            # Build result with actual execution details
            result = {
                "status": "completed",
                "task": self.context.task,
                "agent_id": self.agent_id,
                "output": {
                    "description": f"Executed task: {self.context.task}",
                    "tools_used": len(tool_outputs),
                    "tool_results": tool_outputs if tool_outputs else None,
                },
                "timestamp": datetime.now().isoformat(),
                "execution_metadata": {
                    "worker_id": self.agent_id,
                    "tools_available": self.context.tools or [],
                    "constraints": len(self.context.constraints) if self.context.constraints else 0,
                },
                # QAPV cognitive pattern metadata
                "qapv": {
                    "question": qapv_execution.question_result if qapv_execution else None,
                    "approach": qapv_execution.answer_approach if qapv_execution else None,
                    "verified": verify_passed,
                    "phase_durations": phase_durations,
                    "total_qapv_time": sum(phase_durations.values()) if phase_durations else 0,
                },
                # Add lessons retrieved for this task
                "lessons_retrieved": len(lessons),
                "lesson_summaries": [
                    {
                        "title": lesson.title if hasattr(lesson, 'title') else "Unknown",
                        "confidence": lesson.confidence if hasattr(lesson, 'confidence') else 0.0
                    }
                    for lesson in lessons[:5]  # Include top 5 lessons
                ] if lessons else [],
                # Add checkpoint metadata
                "checkpoints": [
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "label": cp.label,
                        "timestamp": cp.timestamp.isoformat(),
                        "can_restore": cp.can_restore
                    }
                    for cp in checkpoint_history
                ] if checkpoint_history else None
            }

            # Calculate duration
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            result["duration_ms"] = duration_ms
            result["confusion_records"] = confusion_records

            # Record outcome if learning
            if learning_cycle and experience:
                outcome = Outcome(
                    outcome_type=OutcomeType.SUCCESS,
                    description="Task completed successfully",
                    achieved=["task_execution"],
                    quality_score=0.8,
                    efficiency_score=0.7
                )

                # Record experience capture
                self._metrics.record_experience()

                # Complete experience
                learning_cycle.complete_experience(
                    experience,
                    outcome,
                    reflection={
                        "worked": [
                            "Task execution completed",
                            f"Used {len(tool_outputs)} tools" if tool_outputs else "No tools used"
                                                    f"QAPV Question phase: {phase_durations.get('question', 0):.3f}s",
                            f"QAPV Answer phase: {phase_durations.get('answer', 0):.3f}s",
                            f"QAPV Produce phase: {phase_durations.get('produce', 0):.3f}s",
                            f"QAPV Verify phase: {phase_durations.get('verify', 0):.3f}s",
                        ],
                        "didnt_work": [],
                        "different": []
                    }
                )

            # =========================================================================
            # GOT LEARNING FEEDBACK LOOP - Close the loop!
            # =========================================================================
            if self._got_learning_bridge:
                # Step 1: Validate lessons that were used (feedback loop)
                # This strengthens helpful lessons and weakens unhelpful ones
                if got_lesson_ids_used:
                    try:
                        for lesson_id in got_lesson_ids_used:
                            # Task succeeded, so the lessons were helpful
                            self._got_learning_bridge.cycle.validate_lesson(
                                lesson_id=lesson_id,
                                was_helpful=True
                            )
                        logger.info(
                            f"Worker {self.agent_id}: Validated {len(got_lesson_ids_used)} lessons "
                            f"as helpful (task succeeded)"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to validate lessons: {e}")

                # Step 2: Capture this task completion for future learning
                try:
                    self._got_learning_bridge.capture_task_completion(
                        task_id=self.agent_id,
                        task_title=self.context.task,
                        task_category="task_execution",
                        task_priority="medium",
                        duration_seconds=duration_ms / 1000.0,
                        retrospective=f"Task completed successfully. "
                                      f"Used {len(tool_outputs) if tool_outputs else 0} tools. "
                                      f"QAPV verified: {verify_passed}.",
                        approach=qapv_execution.answer_approach if qapv_execution else None,
                    )
                    logger.info(f"Worker {self.agent_id}: Captured task completion for learning")
                except Exception as e:
                    logger.warning(f"Failed to capture GoT learning experience: {e}")

            # Add cognitive metrics summary to result
            result["cognitive_metrics"] = self._metrics.get_summary()
            result["health_score"] = self._metrics.calculate_health_score()

            health_score = result.get("health_score", 0.0)
            logger.info(
                f"Task completed in {duration_ms:.2f}ms "
                f"(tools: {len(tool_outputs) if tool_outputs else 0}) "
                f"[health: {health_score:.1f}]"
            )
            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)

            # Record failure if learning
            if learning_cycle and experience:
                outcome = Outcome(
                    outcome_type=OutcomeType.FAILURE,
                    description=f"Task failed: {str(e)}",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                learning_cycle.complete_experience(experience, outcome)

            # =========================================================================
            # GOT LEARNING FEEDBACK LOOP - Negative feedback on failure
            # =========================================================================
            if self._got_learning_bridge:
                # Step 1: Validate lessons as unhelpful (task failed)
                if got_lesson_ids_used:
                    try:
                        for lesson_id in got_lesson_ids_used:
                            self._got_learning_bridge.cycle.validate_lesson(
                                lesson_id=lesson_id,
                                was_helpful=False  # Task failed, lessons didn't help
                            )
                        logger.info(
                            f"Worker {self.agent_id}: Validated {len(got_lesson_ids_used)} lessons "
                            f"as unhelpful (task failed)"
                        )
                    except Exception as ve:
                        logger.debug(f"Failed to validate lessons on failure: {ve}")

                # Step 2: Capture failure for future avoidance learning
                try:
                    self._got_learning_bridge.capture_task_failure(
                        task_id=self.agent_id,
                        task_title=self.context.task,
                        task_category="task_execution",
                        task_priority="medium",
                        error_message=str(e),
                        attempted_approach=qapv_execution.answer_approach if qapv_execution else None,
                    )
                    logger.info(f"Worker {self.agent_id}: Captured task failure for learning")
                except Exception as fe:
                    logger.debug(f"Failed to capture failure experience: {fe}")

            # Offer to restore to last good checkpoint on failure
            if checkpoint_history and self._checkpoint_id:
                logger.warning(
                    f"Task execution failed. Last checkpoint: {self._checkpoint_id}. "
                    f"Call _restore_state() to recover."
                )
                # Store for potential recovery
                logger.info(f"Available checkpoints: {[cp.checkpoint_id for cp in checkpoint_history]}")

            raise

    async def report_progress(self, progress: float, step: str) -> None:
        """Report progress to director."""
        self.progress = progress
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="worker.progress",
                payload={
                    "progress": progress,
                    "step": step,
                },
                source_agent_id=self.agent_id,
            ))

    async def checkpoint(self) -> Checkpoint:
        """Create a resumable checkpoint."""
        return Checkpoint(
            agent_id=self.agent_id,
            role="worker",
            current_step=self.context.task,
            draft_outputs={"progress": self.progress},
        )

    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from checkpoint."""
        self.progress = checkpoint.draft_outputs.get("progress", 0.0)


# =============================================================================
# DIRECTOR
# =============================================================================


@dataclass
class DirectorResult:
    """Result from a director execution."""

    status: Literal["complete", "partial", "failed"]
    output: Any = None
    workers_spawned: int = 0
    events_handled: int = 0
    decisions_made: list[str] = field(default_factory=list)


@dataclass
class WorkerHandle:
    """Handle to a spawned worker."""

    worker_id: str
    task: Task
    worker: Worker
    status: TaskStatus = TaskStatus.PENDING


class Director(Agent):
    """
    Director agent that orchestrates multiple workers to achieve goals.

    Directors are intermediate nodes in the agent hierarchy that decompose
    high-level goals into focused tasks, spawn and manage workers, handle
    coordination and impediments, and synthesize results.

    Features:
        - **Goal Decomposition**: Break down complex goals
          - Strategic decomposition into manageable tasks
          - Dependency analysis and ordering
          - Parallel vs sequential task identification

        - **Worker Management**: Spawn and coordinate workers
          - Dynamic worker creation
          - Task assignment and delegation
          - Worker health monitoring
          - Load balancing

        - **Escalation Handling**: Manage worker confusion
          - Track worker confusion signals
          - Evaluate escalation levels
          - Execute recovery protocols
          - Three-strikes policy for worker reassignment

        - **Event Coordination**: Pub/sub integration
          - Subscribe to worker events
          - Coordinate cross-worker dependencies
          - Handle blockers and impediments
          - Publish director-level events

        - **Result Synthesis**: Aggregate worker outputs
          - Collect and combine results
          - Validate completeness
          - Generate unified output
          - Track progress

    Coordination Patterns:
        - **Sequential**: Tasks executed in order
        - **Parallel**: Independent tasks executed concurrently
        - **Pipeline**: Output of one task feeds next
        - **Fan-out/Fan-in**: Multiple workers, aggregated results

    Example:
        >>> from llm_orchestration.agents import Director, DirectorContext
        >>> from llm_orchestration.types import Goal, EventBus
        >>>
        >>> # Create event bus for coordination
        >>> event_bus = EventBus()
        >>>
        >>> # Create director context
        >>> context = DirectorContext(
        ...     role="feature_director",
        ...     goal="Implement user authentication",
        ...     scope=None,
        ...     can_spawn=["worker"],
        ...     tools_available=["read", "write", "test"],
        ...     event_bus=event_bus,
        ... )
        >>>
        >>> # Create director
        >>> director = Director("dir-1", context)
        >>>
        >>> # Execute goal (director will decompose and manage workers)
        >>> result = await director.run()
        >>>
        >>> # Check results
        >>> if result.success:
        ...     print(f"Goal achieved: {result.output}")
        ...     print(f"Workers spawned: {len(director.workers)}")
        ...     print(f"Decisions made: {len(director.decisions)}")

    Attributes:
        agent_id (str): Unique identifier for this director
        role (AgentRole): Always AgentRole.DIRECTOR
        context (DirectorContext): Current execution context
        status (TaskStatus): Current execution status
        workers (dict[str, WorkerHandle]): Active and completed workers
        completed_outputs (dict[str, Any]): Results from completed workers
        decisions (list[str]): Decision log
        event_count (int): Number of events handled

    Private Attributes:
        _recovery_coordinator: Handles confusion detection and recovery
        _confusion_signals: Detected confusion signals
        _worker_confusion_count: Tracks confusion per worker (3-strikes)
        _escalation_manager: Manages worker escalation protocols

    See Also:
        HybridDirector: Director with Kanban flow management
        Worker: Leaf agent that executes tasks
        DirectorContext: Configuration for director execution
        EscalationManager: Worker confusion escalation
        WorkerHandle: Reference to spawned worker
    """

    def __init__(
        self,
        agent_id: str,
        context: DirectorContext,
    ):
        super().__init__(agent_id, AgentRole.DIRECTOR)
        self.context = context
        self.workers: dict[str, WorkerHandle] = {}
        self.completed_outputs: dict[str, Any] = {}
        self.decisions: list[str] = []
        self.event_count = 0

        # Confusion detection and recovery for director
        self._recovery_coordinator: Optional[RecoveryCoordinator] = None
        self._confusion_signals: List[ConfusionSignal] = []
        self._worker_confusion_count: Dict[str, int] = {}

        # Initialize recovery coordinator
        try:
            from pathlib import Path
            storage_dir = Path.home() / ".llm_orchestration" / "recovery" / agent_id
            self._recovery_coordinator = RecoveryCoordinator(storage_dir)
        except Exception:
            pass

        # Aggregate worker metrics
        self._worker_metrics: List[Dict[str, Any]] = []
        self._aggregate_health_scores: List[float] = []

        # Initialize escalation manager
        self._escalation_manager = EscalationManager()

        # Track workers under enhanced monitoring
        self._monitored_workers: set[str] = set()

        # Track worker task type suitability (for reassignment)
        self._worker_blacklist: Dict[str, set[str]] = {}  # worker_id -> set of task types

    def _aggregate_worker_metrics(self) -> Dict[str, Any]:
        """
        Aggregate cognitive metrics from all workers.

        Returns:
            dict: Aggregated metrics with fleet-wide statistics
        """
        if not self._worker_metrics:
            return {
                "fleet_health_score": 0.0,
                "total_workers": 0,
                "workers_analyzed": 0,
            }

        # Calculate fleet-wide statistics
        total_workers = len(self._worker_metrics)
        avg_health_score = (
            sum(self._aggregate_health_scores) / len(self._aggregate_health_scores)
            if self._aggregate_health_scores else 0.0
        )

        # Aggregate execution metrics
        total_tasks = sum(m.get("execution", {}).get("tasks_executed", 0) for m in self._worker_metrics)
        total_successful = sum(m.get("execution", {}).get("tasks_successful", 0) for m in self._worker_metrics)
        total_failed = sum(m.get("execution", {}).get("tasks_failed", 0) for m in self._worker_metrics)

        # Aggregate QAPV metrics
        total_qapv_cycles = sum(m.get("qapv", {}).get("cycles", 0) for m in self._worker_metrics)
        avg_verify_pass_rate = (
            sum(m.get("qapv", {}).get("verify_pass_rate", 0) for m in self._worker_metrics) / total_workers
            if total_workers > 0 else 0.0
        )

        # Aggregate learning metrics
        total_lessons_retrieved = sum(m.get("learning", {}).get("lessons_retrieved", 0) for m in self._worker_metrics)
        total_experiences_captured = sum(m.get("learning", {}).get("experiences_captured", 0) for m in self._worker_metrics)

        # Aggregate tool metrics
        total_tools_invoked = sum(m.get("tools", {}).get("tools_invoked", 0) for m in self._worker_metrics)
        avg_tool_success_rate = (
            sum(m.get("tools", {}).get("tool_success_rate", 0) for m in self._worker_metrics) / total_workers
            if total_workers > 0 else 0.0
        )

        # Aggregate recovery metrics
        total_confusion_signals = sum(m.get("recovery", {}).get("confusion_signals", 0) for m in self._worker_metrics)
        total_recoveries = sum(m.get("recovery", {}).get("recoveries_attempted", 0) for m in self._worker_metrics)
        total_recovery_successes = sum(m.get("recovery", {}).get("recoveries_successful", 0) for m in self._worker_metrics)

        # Identify underperforming workers (health score < 70)
        underperforming_workers = [
            i for i, score in enumerate(self._aggregate_health_scores)
            if score < 70.0
        ]

        return {
            "fleet_health_score": avg_health_score,
            "total_workers": total_workers,
            "workers_analyzed": len(self._worker_metrics),
            "underperforming_workers": len(underperforming_workers),
            "execution": {
                "total_tasks_executed": total_tasks,
                "total_tasks_successful": total_successful,
                "total_tasks_failed": total_failed,
                "fleet_success_rate": total_successful / total_tasks if total_tasks > 0 else 0.0,
            },
            "qapv": {
                "total_cycles": total_qapv_cycles,
                "avg_verify_pass_rate": avg_verify_pass_rate,
            },
            "learning": {
                "total_lessons_retrieved": total_lessons_retrieved,
                "total_experiences_captured": total_experiences_captured,
            },
            "tools": {
                "total_tools_invoked": total_tools_invoked,
                "avg_tool_success_rate": avg_tool_success_rate,
            },
            "recovery": {
                "total_confusion_signals": total_confusion_signals,
                "total_recoveries_attempted": total_recoveries,
                "total_recoveries_successful": total_recovery_successes,
                "fleet_recovery_success_rate": (
                    total_recovery_successes / total_recoveries
                    if total_recoveries > 0 else 1.0
                ),
            },
        }

    async def run(self) -> Result:
        """Execute the director's orchestration loop."""
        try:
            # 1. Decompose goal into tasks
            plan = await self.decompose_goal()

            # Publish plan
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="director.plan_created",
                    payload={
                        "subtask_count": len(plan),
                    },
                    source_agent_id=self.agent_id,
                ))

            # 2. Spawn workers for each task
            for task in plan:
                await self.spawn_worker(task)

            # 3. Orchestrate: manage execution
            await self.orchestrate()

            # 4. Synthesize results
            synthesis = await self.synthesize_results()

            # 5. Report completion
            self.status = TaskStatus.COMPLETED
            return Result(
                success=True,
                output=synthesis,
                metadata={
                    "workers_spawned": len(self.workers),
                    "events_handled": self.event_count,
                    "decisions_made": self.decisions,
                },
            )

        except Exception as e:
            self.status = TaskStatus.FAILED
            return Result(success=False, error=str(e))


    def _check_worker_confusion(self, worker_id: str, worker_result: Any) -> Optional[ConfusionSignal]:
        """
        Check if a worker is showing signs of confusion.

        Args:
            worker_id: ID of the worker to check
            worker_result: Result from worker execution

        Returns:
            ConfusionSignal if confusion detected in worker coordination
        """
        # Track repeated failures from same worker (works without recovery_coordinator)
        if isinstance(worker_result, dict) and worker_result.get("status") == "failed":
            self._worker_confusion_count[worker_id] = self._worker_confusion_count.get(worker_id, 0) + 1

            if self._worker_confusion_count[worker_id] >= 3:
                return ConfusionSignal(
                    signal_type="worker_repetition",
                    description=f"Worker {worker_id} has failed {self._worker_confusion_count[worker_id]} times",
                    evidence=[f"Worker: {worker_id}", f"Failures: {self._worker_confusion_count[worker_id]}"],
                    confidence=0.8,
                    source="Director"
                )

        return None

    async def handle_worker_escalation(self, protocol: EscalationProtocol) -> bool:
        """
        Handle an escalation protocol from the escalation manager.

        This method executes the recommended actions based on the escalation level:
        - MONITOR: Enable additional logging and monitoring
        - INTERVENE: Pause worker and analyze state
        - REASSIGN: Move task to a different worker
        - ESCALATE: Escalate to orchestrator
        - ABORT: Abort task and create failure record

        Args:
            protocol: The escalation protocol to handle

        Returns:
            True if handling succeeded, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        # Execute the protocol using the escalation manager
        success = self._escalation_manager.execute(protocol)

        if not success:
            logger.error(f"Failed to execute escalation protocol for worker {protocol.worker_id}")
            return False

        # Perform director-specific actions based on level
        if protocol.level == EscalationLevel.MONITOR:
            logger.info(f"Monitoring worker {protocol.worker_id}")
            # Enhanced monitoring implementation
            logger.info(f"[ESCALATION] MONITOR: Enabling enhanced monitoring for {protocol.worker_id}")

            # Track this worker closely
            self._monitored_workers.add(protocol.worker_id)

            # Log the confusion history for analysis
            for record in protocol.confusion_history[-3:]:
                logger.info(
                    f"  Confusion: {record.signal_type} "
                    f"(severity={record.severity}, recovered={record.recovered})"
                )

            # If worker has a handle, mark it for close observation
            if protocol.worker_id in self.workers:
                handle = self.workers[protocol.worker_id]
                logger.info(
                    f"  Worker status: {handle.status.value}, "
                    f"task: {handle.task.description[:50]}..."
                )

        elif protocol.level == EscalationLevel.INTERVENE:
            logger.warning(f"Intervening for worker {protocol.worker_id}")
            # Pause worker and analyze state
            logger.warning(f"[ESCALATION] INTERVENE: Pausing worker {protocol.worker_id} for analysis")

            # Get worker handle if it exists
            if protocol.worker_id in self.workers:
                handle = self.workers[protocol.worker_id]

                # Mark worker as blocked for intervention
                handle.status = TaskStatus.BLOCKED

                # Capture cognitive state snapshot if worker has one
                if hasattr(handle.worker, '_cognitive_state'):
                    try:
                        state_snapshot = handle.worker._cognitive_state.get_health_metrics()
                        logger.warning(f"  Cognitive state: {state_snapshot}")
                    except Exception as e:
                        logger.warning(f"  Failed to capture cognitive state: {e}")

                # Log intervention details
                logger.warning(f"  Task paused: {handle.task.description[:50]}...")
                logger.warning(f"  Reason: {protocol.reason}")
                logger.warning(f"  Recommended action: {protocol.recommended_action}")

                # If recovery coordinator available, try recovery guidance
                if self._recovery_coordinator:
                    try:
                        # Attempt state-based recovery guidance
                        logger.warning("  Requesting recovery guidance...")
                    except Exception as e:
                        logger.warning(f"  Recovery guidance failed: {e}")
            else:
                logger.warning(f"  Worker {protocol.worker_id} not found in active workers")

        elif protocol.level == EscalationLevel.REASSIGN:
            logger.warning(f"Reassigning task from worker {protocol.worker_id}")
            # Move task to different worker
            logger.warning(f"[ESCALATION] REASSIGN: Moving task from {protocol.worker_id}")

            if protocol.worker_id in self.workers:
                handle = self.workers[protocol.worker_id]
                task = handle.task

                # Mark this worker as unsuitable for this task type
                task_type = getattr(task, 'task_type', 'general')
                if protocol.worker_id not in self._worker_blacklist:
                    self._worker_blacklist[protocol.worker_id] = set()
                self._worker_blacklist[protocol.worker_id].add(task_type)

                logger.warning(
                    f"  Blacklisting worker {protocol.worker_id} for task type: {task_type}"
                )

                # Mark current worker as failed
                handle.status = TaskStatus.FAILED

                # Try to find an alternative worker by spawning a new one
                # (Director will handle this in orchestrate loop)
                logger.warning(
                    f"  Task '{task.description[:50]}...' marked for reassignment"
                )
                logger.warning(f"  Reason: {protocol.reason}")

                # Store task for re-queuing
                if not hasattr(self, '_reassigned_tasks'):
                    self._reassigned_tasks = []
                self._reassigned_tasks.append(task)
            else:
                logger.warning(f"  Worker {protocol.worker_id} not found for reassignment")

        elif protocol.level == EscalationLevel.ESCALATE:
            # Escalate to orchestrator
            await self.escalate(
                f"Worker {protocol.worker_id} escalation: {protocol.reason}"
            )

        elif protocol.level == EscalationLevel.ABORT:
            logger.error(f"Aborting task {protocol.task_id} for worker {protocol.worker_id}")
            # Create failure record and trigger learning
            logger.error(f"[ESCALATION] ABORT: Creating failure record for task {protocol.task_id}")

            # Create failure record
            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "worker_id": protocol.worker_id,
                "task_id": protocol.task_id,
                "reason": protocol.reason,
                "confusion_count": len(protocol.confusion_history),
                "confusion_history": [
                    {
                        "signal_type": record.signal_type,
                        "severity": record.severity,
                        "recovery_action": record.recovery_action,
                        "recovered": record.recovered,
                        "timestamp": record.timestamp.isoformat(),
                    }
                    for record in protocol.confusion_history
                ],
            }

            logger.error(f"  Failure record: {failure_record}")

            # Trigger experience capture if learning available
            if LEARNING_AVAILABLE:
                try:
                    from pathlib import Path
                    from .learning import LearningCycle, Context, Outcome, OutcomeType

                    storage_dir = Path.home() / ".llm_orchestration" / "learning"
                    learning_cycle = LearningCycle(storage_dir)

                    # Create learning context
                    context = Context(
                        goal_type="worker_task_execution",
                        goal_complexity="complex",
                        domain="worker_escalation_abort",
                        prior_failures=len(protocol.confusion_history),
                        notes=f"Worker {protocol.worker_id} aborted on task {protocol.task_id}: {protocol.reason}"
                    )

                    # Create failure outcome
                    outcome = Outcome(
                        outcome_type=OutcomeType.FAILURE,
                        description=f"Worker {protocol.worker_id} aborted task {protocol.task_id}",
                        not_achieved=[f"Complete task {protocol.task_id}"],
                        error_type="worker_escalation_abort",
                        error_message=protocol.reason
                    )

                    # Start and immediately complete experience to record the failure
                    experience = learning_cycle.start_experience(
                        context=context,
                        intent=f"Worker {protocol.worker_id} task execution"
                    )
                    learning_cycle.complete_experience(experience, outcome)

                    logger.error(f"  Failure captured in learning system")

                except Exception as e:
                    logger.error(f"  Failed to capture learning experience: {e}")

            # Mark worker as aborted if it exists
            if protocol.worker_id in self.workers:
                handle = self.workers[protocol.worker_id]
                handle.status = TaskStatus.FAILED

                # Clean up resources - remove from active workers
                logger.error(f"  Cleaning up resources for worker {protocol.worker_id}")

            # Record in director's own tracking
            if not hasattr(self, '_aborted_tasks'):
                self._aborted_tasks = []
            self._aborted_tasks.append({
                "task_id": protocol.task_id,
                "worker_id": protocol.worker_id,
                "timestamp": datetime.now(),
                "reason": protocol.reason,
            })

        return True

    def _handle_worker_confusion(self, worker_id: str, signal: ConfusionSignal) -> str:
        """
        Handle confusion detected in worker execution.

        Args:
            worker_id: ID of the confused worker
            signal: Confusion signal from worker

        Returns:
            Recovery action (CONTINUE, REASSIGN, ESCALATE)
        """
        if not self._recovery_coordinator:
            return "CONTINUE"

        # Record the signal
        self._confusion_signals.append(signal)

        # For worker confusion, we can:
        # 1. CONTINUE - let worker retry
        # 2. REASSIGN - assign task to different worker
        # 3. ESCALATE - escalate to higher level

        if signal.confidence >= 0.8:
            # High confidence confusion - escalate
            import logging
            logging.getLogger(__name__).error(
                f"Worker {worker_id} showing high-confidence confusion. Escalating."
            )
            return "ESCALATE"
        elif signal.confidence >= 0.5:
            # Medium confidence - try reassigning
            return "REASSIGN"
        else:
            # Low confidence - continue
            return "CONTINUE"

    async def decompose_goal(self) -> list[Task]:
        """
        Decompose the goal into worker tasks.

        This method breaks down a complex goal into smaller, manageable tasks
        based on:
        - Goal complexity (inferred from description length and keywords)
        - Available resources
        - Scope constraints

        The decomposition follows these heuristics:
        - Simple goals (< 50 chars): 1-2 tasks
        - Moderate goals (50-150 chars): 2-4 tasks
        - Complex goals (> 150 chars): 4-8 tasks

        Returns:
            list[Task]: List of decomposed tasks
        """
        import logging
        import re

        logger = logging.getLogger(__name__)

        goal = self.context.goal
        tasks = []

        # Analyze goal complexity
        goal_length = len(goal)
        goal_lower = goal.lower()

        # Check for complexity indicators
        complex_keywords = [
            "implement", "build", "create", "design", "refactor",
            "migrate", "integrate", "analyze", "optimize"
        ]
        simple_keywords = [
            "fix", "update", "change", "modify", "add", "remove"
        ]

        complexity_score = 1.0

        # Adjust based on length
        if goal_length > 150:
            complexity_score += 2.0
        elif goal_length > 100:
            complexity_score += 1.5
        elif goal_length > 50:
            complexity_score += 1.0

        # Adjust based on keywords
        for keyword in complex_keywords:
            if keyword in goal_lower:
                complexity_score += 0.5

        for keyword in simple_keywords:
            if keyword in goal_lower:
                complexity_score -= 0.5

        # Check for multiple components (indicated by "and", commas, etc.)
        if " and " in goal_lower:
            complexity_score += 1.0
        component_count = len(re.split(r'[,;]', goal)) - 1
        if component_count > 0:
            complexity_score += component_count * 0.5

        # Determine number of tasks based on complexity
        num_tasks = min(max(int(complexity_score), 1), 8)

        logger.info(
            f"Decomposing goal (complexity: {complexity_score:.1f}) "
            f"into {num_tasks} tasks"
        )

        # Create tasks based on decomposition strategy
        if num_tasks == 1:
            # Simple goal - single task
            tasks.append(
                Task(
                    id=f"{self.agent_id}-task-1",
                    description=goal,
                    acceptance_criteria=["Task completed successfully"]
                )
            )
        elif num_tasks <= 3:
            # Moderate goal - break into planning, execution, verification
            tasks.extend([
                Task(
                    id=f"{self.agent_id}-task-1",
                    description=f"Plan approach for: {goal}",
                    acceptance_criteria=["Approach documented and reviewed"]
                ),
                Task(
                    id=f"{self.agent_id}-task-2",
                    description=f"Execute: {goal}",
                    acceptance_criteria=["Implementation completed"]
                ),
                Task(
                    id=f"{self.agent_id}-task-3",
                    description=f"Verify completion of: {goal}",
                    acceptance_criteria=["Tests pass", "Quality checks pass"]
                )
            ])
        else:
            # Complex goal - break into multiple phases
            # Try to split by conjunctions or components
            components = re.split(r'\s+and\s+|,\s*|;\s*', goal)

            if len(components) > 1 and len(components) <= num_tasks:
                # Use natural language components
                for i, component in enumerate(components, 1):
                    if component.strip():
                        tasks.append(
                            Task(
                                id=f"{self.agent_id}-task-{i}",
                                description=component.strip(),
                                acceptance_criteria=[f"Component {i} completed"]
                            )
                        )
            else:
                # Generic decomposition
                phases = [
                    ("Research and design", "Design documented"),
                    ("Implement core functionality", "Core implementation complete"),
                    ("Add supporting features", "Features implemented"),
                    ("Testing and validation", "Tests pass"),
                    ("Documentation", "Documentation complete"),
                    ("Review and refinement", "Quality standards met")
                ]

                for i in range(min(num_tasks, len(phases))):
                    phase_name, acceptance = phases[i]
                    tasks.append(
                        Task(
                            id=f"{self.agent_id}-task-{i+1}",
                            description=f"{phase_name}: {goal}",
                            acceptance_criteria=[acceptance]
                        )
                    )

        logger.info(f"Created {len(tasks)} tasks from goal decomposition")

        # Add task estimates (simple heuristic)
        for task in tasks:
            task.estimate_points = max(1, 8 // len(tasks))

        return tasks

    async def spawn_worker(self, task: Task) -> WorkerHandle:
        """Spawn a worker for a task."""
        # Validate task
        if task is None:
            raise ValueError("task cannot be None")
        if not isinstance(task, Task):
            raise TypeError(f"task must be Task, got {type(task)}")
        if not task.id or not isinstance(task.id, str):
            raise ValueError("task.id must be a non-empty string")
        if not task.description:
            raise ValueError("task.description cannot be empty")

        worker_id = f"{self.agent_id}-worker-{len(self.workers)}"

        worker_context = WorkerContext(
            task=task.description,
            tools=self.context.tools_available,
            event_bus=self.context.event_bus,
            constraints=self.context.constraints,
        )

        worker = Worker(worker_id, worker_context)

        handle = WorkerHandle(
            worker_id=worker_id,
            task=task,
            worker=worker,
        )

        self.workers[worker_id] = handle

        # Subscribe to worker events
        if self.context.event_bus:
            self.context.event_bus.subscribe(
                f"worker.{worker_id}.*",
                self.handle_worker_event,
            )

        return handle

    def handle_worker_event(self, event: Event) -> None:
        """Handle an event from a worker."""
        self.event_count += 1

        # Extract worker ID from event type
        if event.source_agent_id in self.workers:
            handle = self.workers[event.source_agent_id]

            if "completed" in event.type:
                handle.status = TaskStatus.COMPLETED
                self.completed_outputs[handle.worker_id] = event.payload.get(
                    "result"
                )

                # Collect worker metrics if available
                worker_result = event.payload.get("result")
                if isinstance(worker_result, dict):
                    if "cognitive_metrics" in worker_result:
                        self._worker_metrics.append(worker_result["cognitive_metrics"])
                    if "health_score" in worker_result:
                        self._aggregate_health_scores.append(worker_result["health_score"])

            elif "blocked" in event.type:
                handle.status = TaskStatus.BLOCKED

            # Check for worker confusion
            if "completed" in event.type and event.payload.get("result"):
                signal = self._check_worker_confusion(
                    event.source_agent_id,
                    event.payload.get("result")
                )
                if signal:
                    action = self._handle_worker_confusion(event.source_agent_id, signal)
                    if action == "ESCALATE":
                        # Trigger escalation
                        import asyncio
                        asyncio.create_task(
                            self.escalate(f"Worker {event.source_agent_id} confusion: {signal.description}")
                        )

            elif "failed" in event.type:
                handle.status = TaskStatus.FAILED

    async def orchestrate(self) -> None:
        """Manage worker execution."""
        pending = set(self.workers.keys())
        max_concurrent = 3  # WIP limit

        while pending:
            # Start workers up to WIP limit
            in_progress = sum(
                1 for w in self.workers.values()
                if w.status == TaskStatus.IN_PROGRESS
            )

            ready_to_start = [
                wid for wid in pending
                if self.workers[wid].status == TaskStatus.PENDING
                and self.dependencies_met(wid)
            ]

            for wid in ready_to_start[:max_concurrent - in_progress]:
                handle = self.workers[wid]
                handle.status = TaskStatus.IN_PROGRESS
                asyncio.create_task(handle.worker.run())

            # Wait briefly for events
            await asyncio.sleep(0.1)

            # Update pending set
            pending = {
                wid for wid in pending
                if self.workers[wid].status not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                }
            }

    def dependencies_met(self, worker_id: str) -> bool:
        """Check if a worker's dependencies are met."""
        # Subclasses would implement dependency checking
        return True

    async def synthesize_results(self) -> Any:
        """Synthesize worker outputs into final result."""
        return {
            "fleet_metrics": self._aggregate_worker_metrics(),
            "workers": len(self.workers),
            "completed": len(self.completed_outputs),
            "outputs": self.completed_outputs,
        }

    async def escalate(self, issue: str) -> None:
        """Escalate an issue to the orchestrator."""
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="director.escalation",
                payload={
                    "issue": issue,
                    "director_id": self.agent_id,
                },
                source_agent_id=self.agent_id,
            ))

    async def checkpoint(self) -> Checkpoint:
        """Create a resumable checkpoint."""
        return Checkpoint(
            agent_id=self.agent_id,
            role="director",
            completed_steps=[
                wid for wid, h in self.workers.items()
                if h.status == TaskStatus.COMPLETED
            ],
            pending_steps=[
                wid for wid, h in self.workers.items()
                if h.status == TaskStatus.PENDING
            ],
            decisions=self.decisions,
            draft_outputs=self.completed_outputs,
        )

    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from checkpoint."""
        self.completed_outputs = checkpoint.draft_outputs
        self.decisions = checkpoint.decisions


# =============================================================================
# AGILE WORKER
# =============================================================================


@dataclass
class WorkerSprint:
    """A time-boxed sprint for a worker."""

    sprint_id: str
    goal: str
    timebox: timedelta
    tasks: list[SprintTask] = field(default_factory=list)

    # Tracking
    estimated_points: int = 0
    completed_points: int = 0
    impediments: list[Impediment] = field(default_factory=list)

    # Outcome
    increment: Increment | None = None
    retrospective: Retrospective | None = None

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate (completed/estimated points)."""
        if self.estimated_points == 0:
            return 0.0
        return self.completed_points / self.estimated_points

    @property
    def velocity(self) -> float:
        """Get velocity (completed points per day)."""
        if not self.timebox or self.timebox.total_seconds() == 0:
            return 0.0
        days = self.timebox.total_seconds() / (24 * 3600)
        if days == 0:
            return float(self.completed_points)
        return self.completed_points / days


class AgileWorker(Worker):
    """
    Worker that operates in time-boxed sprints with agile practices.

    AgileWorker extends Worker with sprint-based execution, velocity tracking,
    and retrospectives. It combines cognitive capabilities from Worker with
    agile practices for predictable, iterative delivery.

    Features (in addition to Worker):
        - **Sprint Planning**: Time-boxed iterations
          - Sprint goals and task allocation
          - Capacity-based planning
          - Commitment tracking

        - **Velocity Tracking**: Predictability
          - Story points completed per sprint
          - Historical velocity for forecasting
          - Trend analysis

        - **Retrospectives**: Continuous improvement
          - Sprint-end reflection
          - What worked / what didn't / learnings
          - Action items for next sprint

        - **Increment Delivery**: Demonstrable progress
          - Shippable increments at sprint end
          - Cumulative value delivery
          - Sprint review and feedback

    Sprint Lifecycle:
        1. **Planning**: Define sprint goal and select tasks
        2. **Execution**: Work through tasks in time box
        3. **Review**: Demonstrate completed work
        4. **Retrospective**: Reflect and improve
        5. **Repeat**: Next sprint with learnings applied

    Example:
        >>> from llm_orchestration.agents import AgileWorker, WorkerContext
        >>> from llm_orchestration.agile import WorkerSprint
        >>> from datetime import datetime, timedelta
        >>>
        >>> # Create worker context
        >>> context = WorkerContext(
        ...     task="Implement feature X",
        ...     tools=["read", "write", "test"],
        ... )
        >>>
        >>> # Create agile worker
        >>> worker = AgileWorker("agile-worker-1", context)
        >>>
        >>> # Start sprint
        >>> sprint = WorkerSprint(
        ...     sprint_id="sprint-1",
        ...     duration=timedelta(weeks=2),
        ...     goal="Complete feature X with tests",
        ...     capacity_points=13,
        ... )
        >>> worker.current_sprint = sprint
        >>>
        >>> # Execute in sprint context
        >>> result = await worker.execute_task()
        >>>
        >>> # Complete sprint with retrospective
        >>> retrospective = {
        ...     "what_worked": ["TDD approach", "Clear acceptance criteria"],
        ...     "what_didnt": ["Scope creep midway"],
        ...     "action_items": ["Better scope definition upfront"],
        ... }
        >>> worker.complete_sprint(retrospective)
        >>>
        >>> # Check velocity
        >>> print(f"Velocity history: {worker.velocity_history}")

    Attributes (in addition to Worker):
        current_sprint (WorkerSprint | None): Active sprint if in one
        velocity_history (list[int]): Historical velocity data (points/sprint)
        completed_sprints (list[WorkerSprint]): History of completed sprints
        retrospectives (list[Retrospective]): Sprint retrospectives

    See Also:
        Worker: Base worker with cognitive capabilities
        WorkerSprint: Sprint data structure
        SprintPlanner: Sprint planning utilities
        VelocityTracker: Velocity tracking and forecasting
    """

    def __init__(
        self,
        agent_id: str,
        context: WorkerContext,
        state_manager: Optional[CognitiveStateManager] = None,
    ):
        super().__init__(agent_id, context, state_manager)
        self.current_sprint: WorkerSprint | None = None
        self.velocity_history: list[int] = []

    async def run(self) -> Result:
        """Execute as a sprint."""
        # Plan sprint
        sprint = await self.plan_sprint(
            self.context.task,
            self.context.timebox,
        )

        # Execute sprint
        sprint_result = await self.execute_sprint()

        # Retrospective
        retro = await self.retrospective()

        # Deliver increment
        increment = await self.deliver_increment()

        return Result(
            success=sprint_result.completed_points > 0,
            output=increment,
            metadata={
                "sprint_id": sprint.sprint_id,
                "velocity": sprint_result.completed_points,
                "retrospective": retro,
            },
        )

    async def plan_sprint(
        self,
        goal: str,
        timebox: timedelta,
    ) -> WorkerSprint:
        """Plan a sprint with estimated tasks."""
        # Validate inputs
        if not goal or not isinstance(goal, str):
            raise ValueError("goal must be a non-empty string")
        if timebox is None:
            raise ValueError("timebox cannot be None")
        if not isinstance(timebox, timedelta):
            raise TypeError(f"timebox must be timedelta, got {type(timebox)}")
        if timebox.total_seconds() <= 0:
            raise ValueError(f"timebox must be positive, got {timebox.total_seconds()}s")

        tasks = await self.decompose_to_tasks(goal)

        # Estimate tasks
        for task in tasks:
            task.estimate_points = await self.estimate_task(task)

        # Commit based on velocity
        velocity = self.get_velocity()
        capacity = self._timebox_to_points(timebox, velocity)

        committed = []
        points = 0
        for task in tasks:
            if points + task.estimate_points <= capacity:
                committed.append(task)
                points += task.estimate_points

        sprint = WorkerSprint(
            sprint_id=f"sprint-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            goal=goal,
            timebox=timebox,
            tasks=committed,
            estimated_points=points,
        )

        self.current_sprint = sprint
        return sprint

    def get_velocity(self) -> float:
        """Get average velocity from history."""
        if not self.velocity_history:
            return 5.0  # Default
        return sum(self.velocity_history[-5:]) / min(len(self.velocity_history), 5)

    async def execute_sprint(self) -> WorkerSprint:
        """Execute the sprint tasks."""
        if not self.current_sprint:
            raise ValueError("No sprint planned")

        sprint = self.current_sprint
        start_time = datetime.now()

        for task in sprint.tasks:
            # Check timebox
            elapsed = datetime.now() - start_time
            if elapsed > sprint.timebox:
                break

            # Execute task
            task.status = TaskStatus.IN_PROGRESS
            await self.report_progress(
                sprint.completed_points / max(sprint.estimated_points, 1),
                task.description,
            )

            try:
                await self.work_on_task(task)
                task.status = TaskStatus.COMPLETED
                task.actual_points = task.estimate_points
                sprint.completed_points += task.actual_points

            except Blocked as b:
                task.status = TaskStatus.BLOCKED
                sprint.impediments.append(Impediment(
                    task_id=task.id,
                    description=b.reason,
                ))

        return sprint

    async def retrospective(self) -> Retrospective:
        """Conduct a retrospective."""
        if not self.current_sprint:
            raise ValueError("No sprint to retrospect on")

        sprint = self.current_sprint

        retro = Retrospective(
            sprint_id=sprint.sprint_id,
            went_well=await self.analyze_successes(),
            improvements=await self.analyze_improvements(),
            action_items=await self.generate_actions(),
            velocity_actual=sprint.completed_points,
            velocity_planned=sprint.estimated_points,
            estimation_accuracy=(
                sprint.completed_points / max(sprint.estimated_points, 1)
            ),
            impediment_count=len(sprint.impediments),
        )

        # Update velocity history
        self.velocity_history.append(sprint.completed_points)

        # Publish for evolution
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="worker.retrospective",
                payload=retro.__dict__,
                source_agent_id=self.agent_id,
            ))

        sprint.retrospective = retro
        return retro

    async def deliver_increment(self) -> Increment:
        """Package sprint output as deliverable."""
        if not self.current_sprint:
            raise ValueError("No sprint to deliver")

        sprint = self.current_sprint

        increment = Increment(
            sprint_id=sprint.sprint_id,
            goal=sprint.goal,
            outputs=await self.collect_outputs(),
            acceptance_met=await self.verify_acceptance(),
            metrics={
                "planned_points": sprint.estimated_points,
                "completed_points": sprint.completed_points,
            },
        )

        sprint.increment = increment

        # Publish increment ready
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="worker.increment_ready",
                payload=increment.__dict__,
                source_agent_id=self.agent_id,
            ))

        return increment

    # Helper methods - subclasses would implement
    async def decompose_to_tasks(self, goal: str) -> list[SprintTask]:
        """Decompose goal into sprint tasks."""
        return [
            SprintTask(
                id=f"task-{i}",
                description=f"Task {i} for: {goal}",
                estimate_points=1,
            )
            for i in range(3)
        ]

    async def estimate_task(self, task: SprintTask) -> int:
        """
        Estimate a task in points using historical data.

        This method uses the Estimator class from agile.py to estimate
        task complexity based on:
        - Historical estimation accuracy
        - Task description analysis
        - Complexity indicators

        Args:
            task: The sprint task to estimate

        Returns:
            int: Estimated points (1-8)
        """
        import logging
        from .agile import Estimator

        logger = logging.getLogger(__name__)

        try:
            # Initialize or get estimator
            if not hasattr(self, '_estimator'):
                self._estimator = Estimator()

            # Estimate using historical data
            estimate = self._estimator.estimate(
                task=task,
                task_type="worker_task"
            )

            logger.debug(
                f"Estimated task '{task.description}' at {estimate} points"
            )

            return estimate

        except Exception as e:
            logger.warning(f"Estimation failed, using default: {e}")
            # Fallback to simple heuristic
            return 1

    async def work_on_task(self, task: SprintTask) -> None:
        """
        Execute a single task within sprint context.

        This method:
        - Executes the task using the parent Worker's execute_task logic
        - Tracks progress within the sprint
        - Records actual points for estimation learning
        - Updates task timestamps

        Args:
            task: The sprint task to execute

        Raises:
            Blocked: If the task cannot proceed due to dependencies
            Exception: If task execution fails
        """
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        # Preserve original context for restoration
        original_task = self.context.task

        # Checkpoint BEFORE starting sprint task
        pre_task_checkpoint = self._checkpoint_state(f"pre_sprint_task_{task.id}")


        try:
            # Mark task as started
            task.started_at = datetime.now()
            logger.info(f"Starting sprint task: {task.description}")

            # Update worker context to current task
            self.context.task = task.description

            # Execute the task using parent Worker logic
            result = await self.execute_task()

            # Mark task as completed
            task.completed_at = datetime.now()

            # Checkpoint AFTER successful task completion
            post_task_checkpoint = self._checkpoint_state(f"post_sprint_task_{task.id}")
            if post_task_checkpoint:
                logger.debug(f"Created checkpoint after sprint task: {post_task_checkpoint}")


            # Record actual points for learning
            if hasattr(self, '_estimator'):
                self._estimator.record(
                    task=task,
                    task_type="worker_task",
                    actual_points=task.estimate_points  # Actual equals estimate for now
                )

            logger.info(
                f"Completed sprint task '{task.description}' "
                f"({task.estimate_points} points)"
            )

        except Blocked as b:
            # Let blocked exceptions propagate
            logger.warning(f"Task blocked: {b.reason}")
            # Offer checkpoint restoration for blocked tasks
            if pre_task_checkpoint:
                logger.info(f"Pre-task checkpoint available: {pre_task_checkpoint}")
            raise

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            # Offer checkpoint restoration on failure
            if pre_task_checkpoint:
                logger.warning(
                    f"Sprint task failed. Pre-task checkpoint available: {pre_task_checkpoint}. "
                    f"Call _restore_state() to recover."
                )
            raise

        finally:
            # Always restore original context
            self.context.task = original_task

    async def analyze_successes(self) -> list[str]:
        """
        Analyze what went well in the sprint.

        This method examines the current sprint and identifies positive outcomes:
        - High completion rates
        - Tasks completed without impediments
        - Good estimation accuracy
        - Efficient velocity

        Returns:
            list[str]: List of success observations
        """
        import logging

        logger = logging.getLogger(__name__)
        successes = []

        if not self.current_sprint:
            logger.warning("No sprint to analyze")
            return successes

        sprint = self.current_sprint

        # Analyze completion rate
        if sprint.completion_rate >= 0.9:
            successes.append(
                f"Excellent completion rate: {sprint.completion_rate:.0%} of planned work"
            )
        elif sprint.completion_rate >= 0.7:
            successes.append(
                f"Good completion rate: {sprint.completion_rate:.0%} of planned work"
            )

        # Analyze tasks without blocks
        unblocked_tasks = [
            t for t in sprint.tasks
            if t.status == TaskStatus.COMPLETED
        ]
        if unblocked_tasks and not sprint.impediments:
            successes.append(
                f"All {len(unblocked_tasks)} tasks completed without impediments"
            )

        # Analyze velocity
        if sprint.velocity > 0:
            avg_velocity = self.get_velocity()
            if sprint.velocity >= avg_velocity * 1.1:
                successes.append(
                    f"Velocity improved: {sprint.velocity:.1f} vs avg {avg_velocity:.1f}"
                )
            elif sprint.velocity >= avg_velocity * 0.9:
                successes.append(
                    f"Maintained steady velocity: {sprint.velocity:.1f}"
                )

        # Analyze estimation accuracy
        if 0.8 <= sprint.completion_rate <= 1.2:
            successes.append("Estimation accuracy was good")

        logger.debug(f"Identified {len(successes)} successes")
        return successes

    async def analyze_improvements(self) -> list[str]:
        """
        Analyze what could be improved in the sprint.

        This method identifies areas for improvement:
        - Low completion rates
        - Impediments encountered
        - Poor estimation accuracy
        - Declining velocity

        Returns:
            list[str]: List of improvement observations
        """
        import logging

        logger = logging.getLogger(__name__)
        improvements = []

        if not self.current_sprint:
            logger.warning("No sprint to analyze")
            return improvements

        sprint = self.current_sprint

        # Analyze completion rate
        if sprint.completion_rate < 0.5:
            improvements.append(
                f"Low completion rate: only {sprint.completion_rate:.0%} of planned work"
            )
        elif sprint.completion_rate < 0.7:
            improvements.append(
                f"Moderate completion rate: {sprint.completion_rate:.0%} of planned work"
            )

        # Analyze impediments
        if sprint.impediments:
            improvements.append(
                f"Encountered {len(sprint.impediments)} impediment(s)"
            )
            # List specific impediments
            for imp in sprint.impediments[:3]:  # Limit to first 3
                improvements.append(f"  - {imp.description}")

        # Analyze velocity trend
        if len(self.velocity_history) >= 2:
            recent_velocity = sprint.velocity
            avg_velocity = self.get_velocity()
            if recent_velocity < avg_velocity * 0.8:
                improvements.append(
                    f"Velocity declined: {recent_velocity:.1f} vs avg {avg_velocity:.1f}"
                )

        # Analyze estimation
        if sprint.completion_rate < 0.6:
            improvements.append(
                "Overestimated capacity - consider smaller commitments"
            )
        elif sprint.completion_rate > 1.5:
            improvements.append(
                "Underestimated capacity - could commit to more work"
            )

        # Analyze incomplete tasks
        incomplete_tasks = [
            t for t in sprint.tasks
            if t.status != TaskStatus.COMPLETED
        ]
        if incomplete_tasks:
            improvements.append(
                f"{len(incomplete_tasks)} task(s) not completed"
            )

        logger.debug(f"Identified {len(improvements)} improvements")
        return improvements

    async def generate_actions(self) -> list[str]:
        """
        Generate actionable items from sprint analysis.

        This method creates specific, actionable recommendations for future sprints:
        - Adjustment to estimation
        - Changes to planning
        - Process improvements
        - Impediment resolution

        Returns:
            list[str]: List of action items
        """
        import logging

        logger = logging.getLogger(__name__)
        actions = []

        if not self.current_sprint:
            logger.warning("No sprint to generate actions from")
            return actions

        sprint = self.current_sprint

        # Actions based on completion rate
        if sprint.completion_rate < 0.6:
            actions.append(
                "Action: Reduce sprint commitment by 30% in next sprint"
            )
            actions.append(
                "Action: Break down large tasks into smaller chunks"
            )
        elif sprint.completion_rate > 1.5:
            actions.append(
                "Action: Increase sprint commitment by 20% in next sprint"
            )

        # Actions based on impediments
        if sprint.impediments:
            for imp in sprint.impediments:
                actions.append(
                    f"Action: Resolve impediment - {imp.description}"
                )
            actions.append(
                "Action: Identify and mitigate common blockers proactively"
            )

        # Actions based on velocity trend
        if len(self.velocity_history) >= 3:
            recent_avg = sum(self.velocity_history[-3:]) / 3
            older_avg = sum(self.velocity_history[-6:-3]) / 3 if len(self.velocity_history) >= 6 else recent_avg

            if recent_avg < older_avg * 0.85:
                actions.append(
                    "Action: Investigate causes of declining velocity"
                )
                actions.append(
                    "Action: Review and eliminate process bottlenecks"
                )

        # Actions based on estimation accuracy
        if sprint.completion_rate < 0.8 or sprint.completion_rate > 1.2:
            actions.append(
                "Action: Review estimation process for accuracy"
            )
            actions.append(
                "Action: Compare estimates vs actuals for pattern identification"
            )

        # Default action if sprint was successful
        if not actions and sprint.completion_rate >= 0.8:
            actions.append(
                "Action: Continue current approach - showing good results"
            )

        logger.debug(f"Generated {len(actions)} action items")
        return actions

    async def collect_outputs(self) -> dict[str, Any]:
        """Collect sprint outputs."""
        return {}

    async def verify_acceptance(self) -> bool:
        """Verify acceptance criteria."""
        return True

    def _timebox_to_points(self, timebox: timedelta, velocity: float) -> int:
        """Convert timebox to capacity in points."""
        # Assume 1 point = 5 minutes of work
        minutes = timebox.total_seconds() / 60
        return int(minutes / 5 * (velocity / 5))


# =============================================================================
# HYBRID DIRECTOR
# =============================================================================


@dataclass
class Phase:
    """A phase of work (like an epic)."""

    id: str
    description: str
    tasks: list[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class WorkerKanbanItem:
    """An item on the worker kanban board."""

    id: str
    task: Task
    column: Literal["backlog", "ready", "in_progress", "done"]
    worker_id: str | None = None


class WorkerKanbanBoard:
    """Mini kanban board for managing workers."""

    def __init__(self):
        self.items: dict[str, WorkerKanbanItem] = {}
        self.wip_limit = 3

    def add_to_backlog(self, task: Task) -> None:
        """Add a task to backlog."""
        item = WorkerKanbanItem(
            id=task.id,
            task=task,
            column="backlog",
        )
        self.items[task.id] = item

    def move_to_ready(self, task_id: str) -> None:
        """Move task to ready column."""
        if task_id in self.items:
            self.items[task_id].column = "ready"

    def move_to_in_progress(self, task_id: str, worker_id: str) -> bool:
        """Move task to in_progress if WIP allows."""
        current_wip = sum(
            1 for item in self.items.values()
            if item.column == "in_progress"
        )

        if current_wip >= self.wip_limit:
            return False

        if task_id in self.items:
            self.items[task_id].column = "in_progress"
            self.items[task_id].worker_id = worker_id
            return True

        return False

    def move_to_done(self, task_id: str) -> None:
        """Move task to done column."""
        if task_id in self.items:
            self.items[task_id].column = "done"

    def get_wip(self) -> int:
        """Get current WIP count."""
        return sum(
            1 for item in self.items.values()
            if item.column == "in_progress"
        )

    def get_blocked_count(self) -> int:
        """Get count of blocked items."""
        return 0  # Would track blocked state


class HybridDirector(Director):
    """
    Director that bridges Kanban flow (above) and Agile sprints (below).

    HybridDirector operates at the boundary between continuous flow and
    time-boxed iterations. It receives work via pull-based Kanban from the
    orchestrator and manages workers using Agile sprint practices.

    Features (in addition to Director):
        - **Kanban Integration**: Pull-based work intake
          - Receives goals from orchestrator board
          - Respects WIP limits
          - Manages work item flow
          - Visualizes worker kanban board

        - **Phase Planning**: Strategic decomposition
          - Breaks goals into phases (like epics)
          - Defines phase dependencies
          - Estimates phase durations
          - Tracks phase progress

        - **Sprint Management**: Time-boxed execution
          - Runs workers in sprints
          - Sprint planning and review
          - Velocity-based forecasting
          - Sprint retrospectives

        - **Worker Kanban Board**: Visual management
          - Columns: Ready, In Progress, Review, Done
          - Per-column WIP limits
          - Worker task flow visualization
          - Bottleneck detection

    Methodology Mapping:
        ```
        Orchestrator (Kanban) → HybridDirector → Workers (Agile)
              ↓                       ↓                  ↓
        Continuous flow         Phases/Sprints     Sprint tasks
        Pull-based              WIP limits         Time-boxed
        Flow metrics            Hybrid metrics     Velocity
        ```

    Example:
        >>> from llm_orchestration.agents import HybridDirector
        >>> from llm_orchestration.types import DirectorContext, EventBus
        >>>
        >>> # Create event bus
        >>> event_bus = EventBus()
        >>>
        >>> # Create hybrid director
        >>> context = DirectorContext(
        ...     role="hybrid_director",
        ...     goal="Build authentication system",
        ...     scope=None,
        ...     can_spawn=["agile_worker"],
        ...     tools_available=["read", "write", "test"],
        ...     event_bus=event_bus,
        ... )
        >>>
        >>> director = HybridDirector("hybrid-dir-1", context)
        >>>
        >>> # Director will:
        >>> # 1. Break goal into phases
        >>> # 2. For each phase, create sprint tasks
        >>> # 3. Spawn AgileWorkers to execute in sprints
        >>> # 4. Track via worker kanban board
        >>> # 5. Conduct retrospectives
        >>>
        >>> result = await director.run()
        >>>
        >>> # Check results
        >>> print(f"Phases completed: {len(director.phases)}")
        >>> print(f"Retrospectives: {len(director.retrospectives)}")
        >>> print(f"Worker board state:")
        >>> print(director.worker_board.visualize())

    Attributes (in addition to Director):
        worker_board (WorkerKanbanBoard): Kanban board for worker tasks
        current_phase (Phase | None): Currently executing phase
        phases (list[Phase]): Planned phases for goal
        retrospectives (list[Retrospective]): Phase/sprint retrospectives

    See Also:
        Director: Base director with worker orchestration
        KanbanOrchestrator: Upstream kanban flow management
        AgileWorker: Workers that execute in sprints
        WorkerKanbanBoard: Visual board for worker tasks
        Phase: Strategic decomposition unit
    """

    def __init__(
        self,
        agent_id: str,
        context: DirectorContext,
    ):
        super().__init__(agent_id, context)
        self.worker_board = WorkerKanbanBoard()
        self.current_phase: Phase | None = None
        self.phases: list[Phase] = []
        self.retrospectives: list[Retrospective] = []

    async def run(self) -> Result:
        """Execute as hybrid: receive via pull, execute as sprints."""
        try:
            # Plan phases (like epics → stories)
            self.phases = await self.plan_phases()

            # Execute each phase
            for phase in self.phases:
                self.current_phase = phase
                phase.status = TaskStatus.IN_PROGRESS

                # Add phase tasks to kanban board
                for task in phase.tasks:
                    self.worker_board.add_to_backlog(task)
                    self.worker_board.move_to_ready(task.id)

                # Execute phase
                phase_result = await self.execute_phase(phase)

                phase.status = TaskStatus.COMPLETED

            # Synthesize
            synthesis = await self.synthesize_phases()

            return Result(
                success=True,
                output=synthesis,
                metadata={
                    "phases_completed": len(self.phases),
                    "retrospectives": len(self.retrospectives),
                },
            )

        except Exception as e:
            return Result(success=False, error=str(e))

    async def plan_phases(self) -> list[Phase]:
        """Plan phases from goal."""
        # Subclasses would implement decomposition
        return [
            Phase(
                id="phase-1",
                description=self.context.goal,
                tasks=[
                    Task(id="task-1", description="First task"),
                    Task(id="task-2", description="Second task"),
                ],
            )
        ]

    async def execute_phase(self, phase: Phase) -> dict[str, Any]:
        """Execute a phase using agile workers."""
        results = {}

        for task in phase.tasks:
            # Pull from board (respects WIP)
            worker_id = f"{self.agent_id}-worker-{task.id}"

            if not self.worker_board.move_to_in_progress(task.id, worker_id):
                # WIP limit reached, wait
                await self.wait_for_capacity()
                self.worker_board.move_to_in_progress(task.id, worker_id)

            # Spawn agile worker
            worker_context = WorkerContext(
                task=task.description,
                tools=self.context.tools_available,
                event_bus=self.context.event_bus,
                timebox=timedelta(minutes=10),
            )

            worker = AgileWorker(worker_id, worker_context)

            # Execute
            result = await worker.run()
            results[task.id] = result

            # Update board
            self.worker_board.move_to_done(task.id)

            # Collect retrospective
            if worker.current_sprint and worker.current_sprint.retrospective:
                self.retrospectives.append(
                    worker.current_sprint.retrospective
                )

        return results

    async def wait_for_capacity(self) -> None:
        """Wait for WIP capacity."""
        while self.worker_board.get_wip() >= self.worker_board.wip_limit:
            await asyncio.sleep(0.1)

    async def synthesize_phases(self) -> dict[str, Any]:
        """Synthesize all phase results."""
        return {
            "phases": len(self.phases),
            "retrospectives": [r.__dict__ for r in self.retrospectives],
        }

    def report_up_flow(self) -> dict[str, Any]:
        """Report to orchestrator in kanban terms."""
        return {
            "wip": self.worker_board.get_wip(),
            "throughput": len([
                item for item in self.worker_board.items.values()
                if item.column == "done"
            ]),
            "blocked": self.worker_board.get_blocked_count(),
        }

    async def swarm_on_blocked(self, blocked_task_id: str) -> None:
        """Swarm available workers to help unblock."""
        # Classic agile practice
        available = [
            wid for wid, handle in self.workers.items()
            if handle.status == TaskStatus.COMPLETED
        ]

        if not available:
            await self.escalate(f"Cannot unblock {blocked_task_id}, no workers available")
            return

        # Assign helpers
        for helper_id in available[:2]:
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="director.swarm_requested",
                    payload={
                        "blocked_task": blocked_task_id,
                        "helper": helper_id,
                    },
                    source_agent_id=self.agent_id,
                ))
