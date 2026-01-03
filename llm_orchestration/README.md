# LLM Orchestration Framework

A hierarchical agent orchestration system designed for LLM-based information science tasks. This framework enables structured delegation, coordination, and self-improvement of LLM agents.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER INTENT                                │
│            "Implement secure authentication system"                  │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION (Kanban)                       │
│  • Continuous flow of goals                                          │
│  • WIP limits for system stability                                   │
│  • Pull-based work assignment                                        │
│  • Bottleneck detection & relief                                     │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │  DIRECTOR   │   │  DIRECTOR   │   │  DIRECTOR   │
            │  (Hybrid)   │   │  (Hybrid)   │   │  (Hybrid)   │
            └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                   │                 │                 │
            ┌──────┴──────┐         │          ┌──────┴──────┐
            ▼             ▼         ▼          ▼             ▼
        ┌───────┐    ┌───────┐  ┌───────┐  ┌───────┐    ┌───────┐
        │Worker │    │Worker │  │Worker │  │Worker │    │Worker │
        │(Agile)│    │(Agile)│  │(Agile)│  │(Agile)│    │(Agile)│
        └───────┘    └───────┘  └───────┘  └───────┘    └───────┘

─ ─ ─ ─ ─ ─ ─ ─ ─ ─ EVENT BUS ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EVOLUTION LAYER                              │
│  • Survey: Instrument and observe executions                        │
│  • Study: Analyze traces, attribute outcomes                        │
│  • Evolve: Select, crossover, mutate, propagate                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Cognitive Worker Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WORKER AGENT                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         QAPV COGNITIVE LOOP (Thinking Pattern)               │  │
│  │   QUESTION → ANSWER → PRODUCE → VERIFY                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │      WOVEN MIND (Dual-Process Cognition)                     │  │
│  │   FAST: Pattern matching, execution, heuristics              │  │
│  │   SLOW: Deep analysis, careful validation                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│           ┌───────────────┼───────────────┐                         │
│           ▼               ▼               ▼                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │   TOOLS     │  │  LEARNING   │  │  RECOVERY   │                │
│  │  Executor   │  │   Cycle     │  │ Coordinator │                │
│  │             │  │             │  │             │                │
│  │ • Read      │  │ • Lessons   │  │ • Confusion │                │
│  │ • Write     │  │ • Patterns  │  │   Detection │                │
│  │ • Execute   │  │ • Experience│  │ • State     │                │
│  │ • Search    │  │             │  │   Restore   │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         COGNITIVE STATE MANAGER                              │  │
│  │   • Snapshots  • Checkpoints  • Context tracking             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         METRICS COLLECTOR                                    │  │
│  │   • Health score  • QAPV cycles  • Tool use  • Recovery      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
USER GOAL → Orchestrator → Director → Worker
                                        │
                                        ├─► Retrieve Lessons (Learning)
                                        │
                                        ├─► QAPV Cycle
                                        │   ├─ QUESTION (SLOW thinking)
                                        │   ├─ ANSWER (FAST thinking)
                                        │   ├─ PRODUCE (FAST thinking)
                                        │   └─ VERIFY (SLOW thinking)
                                        │
                                        ├─► Execute Tools
                                        │   ├─ Search codebase
                                        │   ├─ Read/Write files
                                        │   └─ Run commands
                                        │
                                        ├─► Detect Confusion
                                        │   └─ Trigger Recovery
                                        │
                                        ├─► Create Checkpoint
                                        │
                                        └─► Capture Experience
                                            └─► Store for future learning
```

## Key Concepts

### Hybrid Methodology

The framework combines two proven methodologies at different levels:

| Level | Methodology | Why |
|-------|-------------|-----|
| **Orchestration** | Kanban | Continuous flow, variable arrival, optimize throughput |
| **Directors** | Hybrid | Bridge between flow and sprints |
| **Workers** | Agile/Scrum | Time-boxed, predictable, learning via retrospectives |

### Strategy Genome

Agents operate according to a "strategy genome" that can evolve:

```python
@dataclass
class StrategyGenome:
    # How to break down problems
    decomposition_patterns: list[DecompositionPattern]

    # How to assign work
    delegation_strategies: list[DelegationStrategy]

    # How to compress context
    context_compression_methods: list[CompressionMethod]

    # How to coordinate workers
    coordination_protocols: list[CoordinationProtocol]

    # How to handle failures
    failure_strategies: list[FailureStrategy]

    # Meta-genes
    exploration_rate: float
    confidence_threshold: float
```

### Evolutionary Loop

The system improves itself through continuous evolution:

```
EXECUTE → SURVEY → STUDY → SELECT → CROSSOVER → MUTATE → PROPAGATE → EXECUTE
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Core Modules

#### `types.py` - Data Types
Core data types: Goals, Tasks, Events, Contexts, Results

**Key types:**
- `WorkerContext`: Configuration for worker agents
- `DirectorContext`: Configuration for director agents
- `Goal`: High-level objective
- `Task`: Concrete work unit
- `Event`: Pub/sub events for coordination
- `Result`: Execution outcomes

#### `tools.py` - Tool Framework
LLM-optimized tool designs with structured I/O, error handling, and composability

**Key classes:**
- `ToolExecutor`: Manages tool registration and execution
- `ToolResult`: Structured tool execution results
- Built-in tools: search, read, write, execute, analyze

#### `agents.py` - Agent Implementations
Agent implementations: Director, Worker, AgileWorker, HybridDirector

**Key classes:**
- `Worker`: Executes focused tasks with QAPV cognitive loop
- `AgileWorker`: Worker operating in time-boxed sprints
- `Director`: Orchestrates multiple workers
- `HybridDirector`: Bridges Kanban flow and Agile sprints
- `ToolExecutor`: Tool management and execution
- `CognitiveMetricsCollector`: Tracks cognitive performance

#### `orchestration.py` - Kanban Orchestration
Kanban orchestration: Board, Columns, WIP limits, Flow metrics

**Key classes:**
- `KanbanOrchestrator`: Pull-based goal orchestration
- `OrchestrationBoard`: Manages goal flow through columns
- `BottleneckDetector`: Identifies flow bottlenecks
- `FlowOptimizer`: Suggests flow improvements
- `FlowMetrics`: Tracks throughput and cycle time

#### `evolution.py` - Evolutionary System
Evolutionary algorithm: Genome, Selection, Crossover, Mutation, Fitness

**Key classes:**
- `StrategyGenome`: Defines agent behavior strategy
- `StrategyPool`: Population of strategies
- `StrategyEvolver`: Evolves strategies based on fitness
- `FitnessEvaluator`: Multi-objective fitness scoring

#### `agile.py` - Agile Practices
Agile practices: Sprints, Velocity, Estimation, Retrospectives

**Key classes:**
- `WorkerSprint`: Time-boxed iteration for workers
- `SprintPlanner`: Plans sprint capacity and tasks
- `VelocityTracker`: Tracks team velocity over time
- `SprintMetrics`: Sprint performance metrics

#### `metrics.py` - Unified Metrics
Unified metrics combining Kanban flow + Agile sprint + Evolution fitness

**Key classes:**
- `CognitiveMetricsCollector`: Tracks worker cognitive health
- `MetricsCollector`: Aggregates system-wide metrics
- `HybridMetrics`: Combined Kanban + Agile + Evolution metrics

#### `cognitive_state.py` - State Management
State management: CognitiveStateManager, StateSnapshot, state persistence

**Key classes:**
- `CognitiveStateManager`: Manages agent cognitive state
- `StateSnapshot`: Point-in-time state capture
- Supports checkpointing and restoration

#### `learning.py` - Experience Learning
Experience capture: LearningCycle, Experience, Pattern extraction, Lesson retrieval

**Key classes:**
- `LearningCycle`: Captures and retrieves experiences
- `Experience`: Structured experience record
- `Pattern`: Extracted behavior patterns
- `Lesson`: Actionable guidance from experiences

#### `recovery.py` - Confusion Recovery
Recovery coordination: ConfusionDetector, RecoveryCoordinator, recovery strategies

**Key classes:**
- `ConfusionDetector`: Detects confusion signals
- `RecoveryCoordinator`: Orchestrates recovery strategies
- `ConfusionSignal`: Indicators of cognitive confusion
- Recovery strategies: state restoration, goal simplification

#### `escalation.py` - Escalation Protocol
Worker confusion escalation for Directors

**Key classes:**
- `EscalationManager`: Manages worker confusion escalation
- `EscalationProtocol`: Defines escalation responses
- `EscalationLevel`: Severity levels (Monitor → Abort)

#### `thought_patterns.py` - Reasoning Patterns
Thought patterns: QAPVPattern, MCTSPattern, ChainOfThought, ReasoningEngine

**Key classes:**
- `QAPVPattern`: Question → Answer → Produce → Verify loop
- `MCTSPattern`: Monte Carlo Tree Search reasoning
- `ChainOfThought`: Sequential reasoning chain
- `ReasoningEngine`: Executes reasoning patterns

#### `protocols.py` - Communication
Communication protocols: Message types, Protocol validators, Agent coordination

**Key classes:**
- Protocol definitions for agent communication
- Message validation and serialization
- Event bus integration

## Quick Start Guide

### Installation

```python
# The framework is part of the Cortical repository
# No separate installation needed - it's already integrated

from llm_orchestration import Worker, WorkerContext
from llm_orchestration.cognitive_state import CognitiveStateManager
from llm_orchestration.learning import LearningCycle
```

### Basic Worker Usage

```python
from llm_orchestration.agents import Worker, WorkerContext
from llm_orchestration.types import EventBus
from pathlib import Path

# Create worker context
context = WorkerContext(
    task="Implement user authentication",
    tools=["read", "write", "search"],
    constraints=["Must pass tests", "Follow TDD"],
)

# Create event bus for coordination
event_bus = EventBus()

# Create worker with cognitive capabilities
worker = Worker("worker-1", context)

# Execute task
result = await worker.execute_task()

# Check results
if result["success"]:
    print(f"Task completed successfully")
    print(f"Health score: {result['health_score']:.1f}/100")
    print(f"QAPV cycles: {result['metrics']['qapv_cycles']}")
else:
    print(f"Task failed: {result['error']}")
```

### Using Cognitive Features

```python
from llm_orchestration.agents import Worker
from llm_orchestration.cognitive_state import CognitiveStateManager
from llm_orchestration.learning import LearningCycle
from pathlib import Path

# Set up cognitive state management
state_dir = Path(".llm_orchestration/cognitive_state")
state_manager = CognitiveStateManager(state_dir)

# Set up learning
learning_dir = Path(".llm_orchestration/learning")
learning_cycle = LearningCycle(learning_dir)

# Create worker with cognitive capabilities
worker = Worker(
    "worker-1",
    WorkerContext(task="Add OAuth support"),
    state_manager=state_manager,
)

# Worker will automatically:
# - Retrieve relevant lessons before execution
# - Use QAPV cognitive loop (Question → Answer → Produce → Verify)
# - Switch between FAST/SLOW thinking modes
# - Create checkpoints during execution
# - Detect confusion and trigger recovery
# - Capture experiences for future learning

result = await worker.execute_task()

# Get cognitive metrics
summary = worker.get_metrics_summary()
print(f"Tasks executed: {summary['execution']['tasks_executed']}")
print(f"QAPV cycles: {summary['qapv']['cycles']}")
print(f"Lessons retrieved: {summary['learning']['lessons_retrieved']}")
print(f"Confusion signals: {summary['recovery']['confusion_signals']}")
print(f"Health score: {summary['health_score']:.1f}/100")
```

### Using ToolExecutor

```python
from llm_orchestration.agents import ToolExecutor

# Create tool executor
executor = ToolExecutor()

# Register custom tools
async def search_codebase(query: str) -> list[str]:
    # Your search implementation
    return ["file1.py", "file2.py"]

executor.register("search", search_codebase)

# Execute tool
result = await executor.execute("search", {"query": "authentication"})

if result.status == "success":
    print(f"Found files: {result.output}")
    print(f"Duration: {result.duration_ms:.2f}ms")
else:
    print(f"Tool failed: {result.error}")

# Get execution history
history = executor.get_execution_history()
for execution in history:
    print(f"{execution.tool_name}: {execution.result.status}")
```

### Using Bottleneck Detection

```python
from llm_orchestration.orchestration import (
    KanbanOrchestrator,
    BottleneckDetector,
    FlowOptimizer,
)

# Create orchestrator
orchestrator = KanbanOrchestrator()

# Detect bottlenecks
bottlenecks = orchestrator.detect_bottlenecks()

for bottleneck in bottlenecks:
    print(f"Bottleneck in {bottleneck.location}:")
    print(f"  Type: {bottleneck.type}")
    print(f"  Severity: {bottleneck.severity:.2f}")
    print(f"  Queue depth: {bottleneck.queue_depth}")
    print(f"  Recommendation: {bottleneck.recommendation}")

# Get optimization suggestions
optimizations = orchestrator.get_optimizations(bottlenecks)

for opt in optimizations:
    print(f"\n{opt.type.upper()} Optimization:")
    print(f"  Target: {opt.target}")
    print(f"  Action: {opt.action}")
    print(f"  Priority: {opt.priority}/5")
    print(f"  Estimated impact: {opt.estimated_impact:.1%}")

# Apply high-priority optimizations
if optimizations and optimizations[0].priority >= 4:
    success = orchestrator.apply_optimization(optimizations[0])
    if success:
        print("Optimization applied successfully")
```

### Using EscalationManager

```python
from llm_orchestration.escalation import (
    EscalationManager,
    EscalationLevel,
)
from llm_orchestration.recovery import ConfusionSignal

# Create escalation manager
escalation_manager = EscalationManager()

# Worker reports confusion
confusion_signal = ConfusionSignal(
    signal_type="repetition_loop",
    description="Attempting same failed approach repeatedly",
    evidence=["read_file", "read_file", "read_file"],
    confidence=0.9,
    source="worker-1",
)

# Evaluate escalation
protocol = escalation_manager.evaluate_escalation(
    worker_id="worker-1",
    task_id="T-123",
    confusion_signal=confusion_signal,
)

print(f"Escalation level: {protocol.level.name}")
print(f"Reason: {protocol.reason}")
print(f"Recommended action: {protocol.recommended_action}")

# Execute escalation if needed
if protocol.level.value >= EscalationLevel.INTERVENE.value:
    result = escalation_manager.execute_protocol(protocol)
    print(f"Escalation executed: {result}")
```

### Using CognitiveMetricsCollector

```python
from llm_orchestration.agents import (
    CognitiveMetricsCollector,
    QAPVExecution,
    ToolResult,
)

# Create metrics collector
metrics = CognitiveMetricsCollector()

# Record task execution
metrics.record_task(success=True)

# Record QAPV cycle
qapv_execution = QAPVExecution(
    cycle_id="cycle-1",
    verify_passed=True,
    phase_durations={
        "question": 0.5,
        "answer": 0.3,
        "produce": 2.0,
        "verify": 0.8,
    }
)
metrics.record_qapv_cycle(qapv_execution)

# Record tool use
tool_result = ToolResult(
    tool_name="search",
    status="success",
    output=["file1.py"],
    duration_ms=150.0,
)
metrics.record_tool_use(tool_result)

# Record learning
metrics.record_lesson(retrieved=True, applied=True)

# Calculate health score
health = metrics.calculate_health_score()
print(f"Cognitive health: {health:.1f}/100")

# Get summary
summary = metrics.get_summary()
print(f"Tasks executed: {summary['execution']['tasks_executed']}")
print(f"Success rate: {summary['execution']['success_rate']:.1%}")
print(f"QAPV verify pass rate: {summary['qapv']['verify_pass_rate']:.1%}")
print(f"Tool success rate: {summary['tools']['tool_success_rate']:.1%}")
```

## Implementation Status

### ✅ READY (Production-Ready Modules)

These modules are fully implemented and tested:

- **`cognitive_state.py`** - State management with persistence and versioning
- **`learning.py`** - Experience capture, pattern extraction, lesson retrieval
- **`recovery.py`** - Confusion detection and recovery coordination
- **`thought_patterns.py`** - QAPV, MCTS, and other reasoning patterns
- **`agile.py`** - Sprint planning, velocity tracking, retrospectives
- **`metrics.py`** - Unified metrics collection and dashboards
- **`protocols.py`** - Protocol definitions and validators
- **`types.py`** - Core data types and schemas

### ⚠️ NEEDS WORK (Partial Implementation)

These modules have stubs or incomplete implementations:

- **`agents.py`** - Abstract base classes complete; concrete `work_on_task()` methods are stubs
- **`evolution.py`** - Core structure present; some analysis methods need implementation
- **`tools.py`** - Tool interfaces complete; search logic has placeholder implementations
- **`orchestration.py`** - Basic Kanban flow works; strategy selection needs wiring to GoT

## GoT Integration

The framework integrates deeply with the Graph of Thought (GoT) transactional task system:

### GoTLearningBridge

**Location:** `cortical/got/learning_integration.py`

Bidirectional bridge connecting:
- **Task → Experience**: Completed tasks become learning experiences
- **Experience → Lesson**: Patterns extracted for future guidance
- **Failure → Learning**: Task failures tracked and analyzed

**Features:**
- Automatic experience capture from task completions
- Retrospective-to-reflection mapping
- Tag-based experience categorization (by category, priority, approach)
- Lesson retrieval for similar task contexts
- Failure pattern tracking and analysis

**Storage:** `.got/learning/` subdirectory
- `experiences/` - Captured task experiences
- `patterns/` - Extracted patterns
- `lessons/` - Distilled lessons

### LearningCycle Integration

The `LearningCycle` automatically connects to task completions via the bridge:

1. **Task Completes** → `bridge.capture_task_completion()`
2. **Experience Created** → Stored in `.got/learning/experiences/`
3. **Patterns Extracted** → Similar experiences clustered
4. **Lessons Distilled** → Actionable guidance generated
5. **New Task Starts** → Relevant lessons retrieved and applied

### Failure Tracking

When tasks fail or encounter errors:

```python
# Capture failure with context
bridge.capture_task_failure(
    task_id="T-123",
    error_type="test_failure",
    error_message="Assertion failed in test_auth",
    context={"test": "test_login", "assertion": "status_code == 200"},
    attempted_fix="Added input validation",
)

# Retrieve similar failure patterns
similar = bridge.get_failure_patterns(
    error_type="test_failure",
    context={"test": "test_*"},
)
```

This creates a feedback loop where failures become learning opportunities.

## Usage

### Working Examples (READY Modules)

#### CognitiveStateManager - State Persistence

```python
from llm_orchestration.cognitive_state import CognitiveStateManager
from pathlib import Path

# Initialize state manager
state_dir = Path(".got/cognitive_state")
manager = CognitiveStateManager(state_dir)

# Update cognitive state
manager.update_context({
    "current_task": "T-123",
    "approach": "test-first",
    "blockers": [],
})

# Save snapshot
snapshot = manager.save_snapshot(
    label="pre-implementation",
    metadata={"phase": "planning"},
)

# Later: restore if confused
manager.restore_snapshot(snapshot.snapshot_id)
```

#### LearningCycle - Experience Capture

```python
from llm_orchestration.learning import (
    LearningCycle,
    Experience,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)
from pathlib import Path

# Initialize learning cycle
cycle = LearningCycle(storage_dir=Path(".got/learning"))

# Capture an experience
exp = cycle.capture_experience(
    context=Context(
        situation="Implementing authentication API",
        constraints=["Must pass existing tests", "Cannot break backward compatibility"],
        goals=["Add JWT support"],
    ),
    action=Action(
        approach="Test-first development",
        reasoning="Tests protect against regressions",
        alternatives_considered=["Implementation-first", "Spike-then-test"],
    ),
    outcome=Outcome(
        result="All tests pass, JWT support added",
        outcome_type=OutcomeType.SUCCESS,
        metrics={"tests_written": 5, "time_minutes": 45},
    ),
    experience_type=ExperienceType.TASK_COMPLETION,
    tags=["api", "authentication", "tdd"],
)

# Extract patterns from similar experiences
patterns = cycle.extract_patterns(
    experience_ids=[exp.experience_id],
    min_similarity=0.7,
)

# Get guidance for new situation
lessons = cycle.get_relevant_lessons(
    context=Context(
        situation="Adding OAuth2 support to API",
        constraints=["Must maintain JWT support"],
        goals=["Support Google/GitHub login"],
    ),
    limit=3,
)

for lesson in lessons:
    print(f"Lesson: {lesson.title}")
    print(f"Guidance: {lesson.guidance}")
```

#### RecoveryCoordinator - Confusion Detection

```python
from llm_orchestration.recovery import (
    RecoveryCoordinator,
    ConfusionDetector,
    RecoveryStrategy,
)
from llm_orchestration.cognitive_state import CognitiveStateManager
from pathlib import Path

# Initialize recovery system
state_manager = CognitiveStateManager(Path(".got/cognitive_state"))
coordinator = RecoveryCoordinator(state_manager)

# Detect confusion signals
detector = ConfusionDetector()
confusion = detector.detect_confusion(
    recent_actions=["read_file", "read_file", "read_file"],  # Repetition
    error_count=3,  # Elevated errors
    state=state_manager.get_current_state(),
)

if confusion.is_confused:
    print(f"Confusion detected: {confusion.signals}")

    # Coordinate recovery
    recovery_plan = await coordinator.coordinate_recovery(
        confusion_signals=confusion.signals,
        available_strategies=[
            RecoveryStrategy.RESTORE_SNAPSHOT,
            RecoveryStrategy.SIMPLIFY_GOAL,
        ],
    )

    print(f"Recovery strategy: {recovery_plan.strategy}")
    print(f"Steps: {recovery_plan.steps}")
```

#### QAPVPattern - Structured Reasoning

```python
from llm_orchestration.thought_patterns import QAPVPattern, ReasoningEngine

# Initialize QAPV reasoning
qapv = QAPVPattern()

# Question phase
questions = qapv.question(
    problem="How should we implement user authentication?",
    context={"requirements": "JWT-based, secure, scalable"},
)

# Answer phase
answers = qapv.answer(
    questions=questions,
    knowledge_sources=["past_experiences", "best_practices"],
)

# Produce phase
plan = qapv.produce(
    answers=answers,
    constraints=["Must not break existing tests"],
)

# Verify phase
verification = qapv.verify(
    plan=plan,
    acceptance_criteria=["Tests pass", "Security audit clean"],
)

print(f"Plan verified: {verification.is_valid}")
print(f"Concerns: {verification.concerns}")
```

#### GoT Learning Integration

```python
from cortical.got.learning_integration import GoTLearningBridge
from pathlib import Path

# Initialize bridge
bridge = GoTLearningBridge(Path(".got"))

# When task completes successfully
exp = bridge.capture_task_completion(
    task_id="T-123",
    task_title="Implement user authentication",
    task_category="feature",
    task_priority="high",
    approach="test-first",
    retrospective="TDD worked well. Tests caught edge cases early.",
    files_changed=["api/auth.py", "tests/test_auth.py"],
    metrics={"tests_written": 5, "lines_changed": 120},
)

# When starting similar task, get lessons
lessons = bridge.get_lessons_for_task(
    task_category="feature",
    task_context="API implementation",
    limit=3,
)

for lesson in lessons:
    print(f"From {lesson.source_count} similar tasks:")
    print(f"  {lesson.guidance}")

# Track failure for learning
bridge.capture_task_failure(
    task_id="T-124",
    task_title="Add OAuth support",
    error_type="test_failure",
    error_message="Assertion failed: status_code == 200",
    context={"test": "test_google_login"},
    attempted_fix="Added token validation",
)
```

### Partial Implementation (Use with Caution)

#### Basic Orchestration (Strategy Selection Stubbed)

```python
from llm_orchestration import (
    KanbanOrchestrator,
    Goal,
    EventBus,
)

# Create orchestrator
event_bus = EventBus()
orchestrator = KanbanOrchestrator(event_bus=event_bus)

# Submit a goal
goal = Goal(
    id="goal-1",
    description="Implement user authentication",
    priority=1,
)
await orchestrator.submit_goal(goal)

# Run orchestration loop
# Note: Strategy selection not yet wired to GoT
await orchestrator.run()
```

## Design Principles

### For LLM Limitations

1. **Externalize Memory** - Tools provide persistent state
2. **Concise Defaults** - Return minimal data, expand on request
3. **Combined Operations** - Reduce round-trips with multi-step tools
4. **Structured Errors** - Enable reasoning about failures
5. **Progressive Disclosure** - Summary first, details available

### For Self-Improvement

1. **Instrument Everything** - All executions are traced
2. **Attribute Outcomes** - Success/failure attributed to specific genes
3. **Preserve Golden Strategies** - Never lose proven approaches
4. **Regression Tests** - Ensure new generations don't regress
5. **Diversity Preservation** - Avoid premature convergence

## Event Types

The pub/sub event system uses typed events:

| Category | Events |
|----------|--------|
| Lifecycle | `agent.spawned`, `agent.terminated` |
| Progress | `task.started`, `task.progress`, `task.completed` |
| Issues | `blocker.raised`, `blocker.resolved`, `error.occurred` |
| Coordination | `dependency.ready`, `resource.acquired` |
| Knowledge | `discovery.made`, `decision.made` |
| Evolution | `worker.retrospective`, `evolution.generation_complete` |

## Fitness Dimensions

Multi-objective fitness evaluation:

| Dimension | Measures |
|-----------|----------|
| Success | Goal achievement rate |
| Efficiency | Time, resources, context usage |
| Quality | Completeness, correctness |
| Stability | Error rate, recovery success |
| Elegance | Solution simplicity |

## Safeguards

The evolution system includes safeguards:

- **Elitism**: Best strategy always preserved
- **Golden Strategies**: Critical approaches never lost
- **Regression Tests**: Automated validation
- **Diversity Monitoring**: Prevent population collapse
- **WIP Limits**: Prevent system overload

## Examples

The `llm_orchestration/examples/` directory contains working demonstrations:

### Core Framework Examples

- **`basic_workflow.py`** - QAPV reasoning pattern demonstration
  - Question → Answer → Produce → Verify cycle
  - Structured problem-solving
  - Verification with acceptance criteria

- **`learning_demo.py`** - Experience capture and pattern extraction
  - Capturing experiences from task completions
  - Extracting patterns from similar experiences
  - Retrieving relevant lessons for new situations

- **`recovery_demo.py`** - Confusion detection and recovery
  - Detecting confusion signals (repetition, errors, contradictions)
  - Recovery strategy selection
  - State restoration

- **`multi_session.py`** - State persistence across sessions
  - Saving cognitive state snapshots
  - Restoring state after interruption
  - Maintaining continuity

### Integration Examples

- **`examples/got_learning_demo.py`** - GoT learning integration (NEW)
  - Task completion → experience capture
  - Lesson retrieval for similar tasks
  - Failure pattern tracking
  - Located in project root `examples/` directory

### Running Examples

```bash
# Core framework examples
cd llm_orchestration/examples
python basic_workflow.py
python learning_demo.py
python recovery_demo.py
python multi_session.py

# GoT integration example
cd /home/user/Opus-code-test
python examples/got_learning_demo.py
```

## Future Work

### Completed ✅

- [x] Integrate with GoT learning (via `GoTLearningBridge`)
- [x] Add failure tracking (task failures → learning experiences)
- [x] Cognitive state management (snapshots, persistence)
- [x] Recovery coordination (confusion detection, recovery strategies)
- [x] Thought patterns (QAPV, MCTS, Chain-of-Thought)

### In Progress 🔄

- [ ] Complete agent execution stubs in `agents.py`
  - `work_on_task()` implementations for Workers
  - Concrete execution logic for Directors

- [ ] Implement search logic in `tools.py`
  - Wire search tools to actual codebase analysis
  - Connect to Cortical text processor

- [ ] Wire orchestration to GoT task system
  - Strategy selection based on GoT task context
  - Automatic task creation for goals
  - Bidirectional sync: orchestration ↔ GoT

### Planned 📋

- [ ] Distributed orchestration across multiple hosts
- [ ] Persistent strategy pool with versioning
- [ ] A/B testing framework for strategies
- [ ] Visual dashboard for real-time monitoring
- [ ] Integration with external tool ecosystems
- [ ] Performance benchmarks for all reasoning patterns
- [ ] Automated regression tests for evolved strategies

## License

MIT License - See LICENSE file for details.
