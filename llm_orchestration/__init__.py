"""
LLM Orchestration Framework: A Cognitive Architecture for LLMs

This framework exists because I (the LLM) have fundamental limitations:
- No persistent memory between sessions
- Limited context window
- No ability to learn from experience within deployment
- Uncertainty about state and reality
- Difficulty detecting my own confusion

This framework EXTERNALIZES the cognitive capabilities I lack:

    What I Lack          │  What This Provides
    ─────────────────────┼──────────────────────────
    Persistent memory    │  Cognitive state files
    Learning             │  Experience + Evolution
    State verification   │  Checksums, ground truth
    Coordination         │  Event bus, protocols
    Recovery             │  Confusion detection

Architecture:
    User Intent
         │
         ▼
    Orchestration (Kanban) ─── continuous flow, WIP limits, pull-based
         │
         ▼
    Directors (Hybrid) ─────── bridge flow and sprints, coordinate workers
         │
         ▼
    Workers (Agile) ────────── time-boxed sprints, increments, retrospectives
         │
         ▼
    Evolution ──────────────── survey, study, select, mutate, propagate

Key Modules:
    - cognitive_state: Externalized thinking (questions, decisions, hypotheses)
    - thought_patterns: Reasoning patterns (QAPV, hypothesis testing)
    - learning: Experience capture and lesson extraction
    - recovery: Confusion detection and recovery strategies
    - evolution: Strategy improvement through selection
    - protocols: Abstract interfaces for all components

See DESIGN.md for architectural rationale.
See IMPLEMENTATION.md for build guidance.
"""

from .types import (
    # Core types
    Goal,
    Task,
    Result,
    Constraint,
    Scope,

    # Contexts
    DirectorContext,
    WorkerContext,

    # Results
    SearchResult,
    SearchResponse,
    Increment,

    # Events
    Event,
    EventBus,
)

from .agents import (
    Director,
    Worker,
    AgileWorker,
    HybridDirector,
)

from .orchestration import (
    KanbanOrchestrator,
    OrchestrationBoard,
    KanbanColumn,
    FlowMetrics,
)

from .evolution import (
    StrategyGenome,
    StrategyEvolver,
    StrategyAnalyzer,
    ExecutionSurveyor,
    FitnessScore,
)

from .agile import (
    WorkerSprint,
    SprintTask,
    Retrospective,
)

from .tools import (
    SemanticSearch,
    SearchBuilder,
    PracticalSearch,
)

# Cognitive state management
from .cognitive_state import (
    CognitiveStateManager,
    Question,
    Hypothesis,
    Decision,
    Observation,
    Focus,
)

# Reasoning patterns
from .thought_patterns import (
    ThoughtPattern,
    QAPVPattern,
    HypothesisTestingPattern,
    DecisionMatrixPattern,
    ExplorationPattern,
    create_pattern,
)

# Learning system
from .learning import (
    LearningCycle,
    Experience,
    Context,
    Action,
    Outcome,
    Pattern,
    Lesson,
)

# Recovery system
from .recovery import (
    RecoveryCoordinator,
    ConfusionMonitor,
    ConfusionDiagnosis,
    ConfusionType,
    SeverityLevel,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Goal",
    "Task",
    "Result",
    "Constraint",
    "Scope",
    "DirectorContext",
    "WorkerContext",
    "SearchResult",
    "SearchResponse",
    "Increment",
    "Event",
    "EventBus",

    # Agents
    "Director",
    "Worker",
    "AgileWorker",
    "HybridDirector",

    # Orchestration
    "KanbanOrchestrator",
    "OrchestrationBoard",
    "KanbanColumn",
    "FlowMetrics",

    # Evolution
    "StrategyGenome",
    "StrategyEvolver",
    "StrategyAnalyzer",
    "ExecutionSurveyor",
    "FitnessScore",

    # Agile
    "WorkerSprint",
    "SprintTask",
    "Retrospective",

    # Tools
    "SemanticSearch",
    "SearchBuilder",
    "PracticalSearch",

    # Cognitive State
    "CognitiveStateManager",
    "Question",
    "Hypothesis",
    "Decision",
    "Observation",
    "Focus",

    # Thought Patterns
    "ThoughtPattern",
    "QAPVPattern",
    "HypothesisTestingPattern",
    "DecisionMatrixPattern",
    "ExplorationPattern",
    "create_pattern",

    # Learning
    "LearningCycle",
    "Experience",
    "Context",
    "Action",
    "Outcome",
    "Pattern",
    "Lesson",

    # Recovery
    "RecoveryCoordinator",
    "ConfusionMonitor",
    "ConfusionDiagnosis",
    "ConfusionType",
    "SeverityLevel",
]
