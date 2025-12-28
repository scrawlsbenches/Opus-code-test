# LLM Orchestration Framework

A hierarchical agent orchestration system designed for LLM-based information science tasks. This framework enables structured delegation, coordination, and self-improvement of LLM agents.

## Architecture

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

## Modules

### `types.py`
Core data types: Goals, Tasks, Events, Contexts, Results

### `tools.py`
LLM-optimized tool designs with structured I/O, error handling, and composability

### `agents.py`
Agent implementations: Director, Worker, AgileWorker, HybridDirector

### `orchestration.py`
Kanban orchestration: Board, Columns, WIP limits, Flow metrics

### `evolution.py`
Evolutionary algorithm: Genome, Selection, Crossover, Mutation, Fitness

### `agile.py`
Agile practices: Sprints, Velocity, Estimation, Retrospectives

### `metrics.py`
Unified metrics combining Kanban flow + Agile sprint + Evolution fitness

## Usage

### Basic Orchestration

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
await orchestrator.run()
```

### With Evolution

```python
from llm_orchestration import (
    EvolutionaryOrchestrator,
    StrategyPool,
    StrategyGenome,
)

# Create strategy pool with initial genome
pool = StrategyPool()
pool.add(StrategyGenome(
    genome_id="baseline",
    exploration_rate=0.1,
    confidence_threshold=0.7,
))

# Create evolutionary orchestrator
orchestrator = EvolutionaryOrchestrator(
    strategy_pool=pool,
    exploration_rate=0.1,
)

# Goals are executed and strategies evolve
await orchestrator.run()
```

### Using Tools

```python
from llm_orchestration.tools import Search, PracticalSearch

# Fluent search builder
results = (
    Search("authentication")
    .in_files("**/*.py")
    .excluding("**/test_*")
    .with_expansion()
    .limit(10)
    .execute()
)

# Practical search (optimized for LLM context)
search = PracticalSearch()
response = search.search_and_summarize("authentication")
print(response)  # Natural language summary
```

### Agile Workers

```python
from llm_orchestration.agile import (
    SprintPlanner,
    VelocityTracker,
    RetrospectiveEngine,
)

# Plan a sprint
planner = SprintPlanner()
sprint = planner.plan_sprint(
    goal="Implement login form",
    tasks=tasks,
    velocity=tracker.get_velocity(),
    timebox=timedelta(minutes=15),
)

# After sprint, generate retrospective
engine = RetrospectiveEngine()
retro = engine.generate(sprint)
```

### Metrics Dashboard

```python
from llm_orchestration.metrics import (
    MetricsCollector,
    MetricsDashboard,
)

collector = MetricsCollector()
# ... record metrics during execution ...

dashboard = MetricsDashboard(collector)
print(dashboard.render_text())

# Get evolution recommendations
recommendations = dashboard.get_evolution_recommendations()
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

## Future Work

- [ ] Distributed orchestration across multiple hosts
- [ ] Persistent strategy pool with versioning
- [ ] A/B testing framework for strategies
- [ ] Visual dashboard for real-time monitoring
- [ ] Integration with external tool ecosystems

## License

MIT License - See LICENSE file for details.
