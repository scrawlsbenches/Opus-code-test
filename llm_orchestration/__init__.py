"""
LLM Orchestration Framework

A hierarchical agent orchestration system designed for LLM-based information
science tasks. Combines:
- Kanban flow management at orchestration level
- Agile sprint practices at worker level
- Evolutionary algorithms for continuous improvement

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
]
