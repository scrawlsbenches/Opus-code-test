# Implementation Guide

This document provides practical guidance for implementing the LLM Cognitive Architecture. It's written for developers (human or AI) who will be building the actual running system.

## Implementation Philosophy

### Build in Layers

Each layer should be **useful on its own** even if higher layers aren't complete. This allows:
- Incremental development
- Early testing with real usage
- Graceful degradation if parts fail

### Test with Real LLM Usage

The best test is actual usage. Each component should be tested with:
1. Unit tests (correctness)
2. Integration tests (compatibility)
3. Real LLM sessions (actual utility)

### Maintain Simplicity

Complex systems fail in complex ways. Prefer:
- Clear data structures over clever algorithms
- Explicit state over implicit inference
- Logging over silence

---

## Implementation Order

Build in this order. Each phase creates value before the next begins.

### Phase 1: Foundation (Week 1-2)

**Goal:** Core data types and event system.

```
types.py (done)
├── Basic data classes (Goal, Task, Result)
├── Event and EventBus
└── Basic enums

protocols.py (done)
├── Core protocols (Identifiable, Timestamped)
├── Agent hierarchy protocols
└── Tool protocols
```

**Validation:**
- [ ] All types can be serialized/deserialized to JSON
- [ ] EventBus can publish and subscribe
- [ ] Protocols can be satisfied by simple implementations

**Integration Test:**
```python
# Create event bus, publish/subscribe, verify delivery
bus = EventBus()
received = []
bus.subscribe("test.event", lambda e: received.append(e))
bus.publish(Event(event_type="test.event", payload={"msg": "hello"}))
assert len(received) == 1
assert received[0].payload["msg"] == "hello"
```

---

### Phase 2: Cognitive State (Week 2-3)

**Goal:** External thinking that persists.

```
cognitive_state.py (done)
├── Question, Hypothesis, Decision
├── CognitiveStateManager
└── Checkpoint/restore

thought_patterns.py (done)
├── QAPVPattern
├── HypothesisTesting
└── Pattern factories
```

**Validation:**
- [ ] State can be saved to and loaded from disk
- [ ] Questions track their full lifecycle
- [ ] Decisions preserve rationale and alternatives
- [ ] Checkpoints can be restored correctly

**Integration Test:**
```python
# Full QAPV cycle with persistence
manager = CognitiveStateManager(Path("./test_state"))

q = manager.add_question("How should authentication work?")
h = manager.add_hypothesis(q.id, "Use JWT tokens")
d = manager.add_decision(
    question_id=q.id,
    choice="JWT with refresh tokens",
    rationale="Stateless, works with microservices"
)

manager.save_checkpoint()
# Later...
manager2 = CognitiveStateManager(Path("./test_state"))
manager2.load_latest_checkpoint()
assert manager2.get_question(q.id).status == QuestionStatus.ANSWERED
```

---

### Phase 3: Basic Agents (Week 3-4)

**Goal:** Worker agents that can execute tasks.

```
agents.py
├── Base Agent class
├── Worker with task execution
└── Basic Director
```

**Key Implementation Details:**

1. **Workers are stateless between tasks:**
   ```python
   class Worker:
       def execute(self, task: Task, context: WorkerContext) -> Result:
           # Each call is independent
           # State comes from context
           pass
   ```

2. **Context compression happens at handoff:**
   ```python
   def prepare_worker_context(director_context: DirectorContext, task: Task) -> WorkerContext:
       # Compress relevant parts only
       return WorkerContext(
           task=task,
           relevant_decisions=[...],  # Only decisions affecting this task
           constraints=task.constraints,
           # NOT full history
       )
   ```

3. **Results include metadata:**
   ```python
   @dataclass
   class Result:
       status: ResultStatus
       output: Any
       artifacts: List[Artifact]
       metrics: Dict[str, float]
       notes: str  # For learning
   ```

**Validation:**
- [ ] Worker can execute simple task and return result
- [ ] Director can decompose goal into tasks
- [ ] Context flows correctly through hierarchy

---

### Phase 4: Orchestration (Week 4-5)

**Goal:** Kanban-style work management.

```
orchestration.py
├── KanbanBoard with columns
├── WIP limits enforcement
├── Pull-based assignment
└── Basic metrics (lead time, throughput)
```

**Key Implementation Details:**

1. **WIP limits are enforced at pull:**
   ```python
   def pull_work(self, agent_id: str) -> Optional[Goal]:
       column = self.get_in_progress_column()
       if column.count() >= column.wip_limit:
           return None  # Cannot pull more work
       return self.backlog.pop_highest_priority()
   ```

2. **State transitions are events:**
   ```python
   def move_to_column(self, goal: Goal, column: str):
       old_column = goal.current_column
       goal.current_column = column
       self.bus.publish(Event(
           event_type="goal.moved",
           payload={"goal_id": goal.id, "from": old_column, "to": column}
       ))
   ```

**Validation:**
- [ ] WIP limits prevent overwork
- [ ] Pull semantics work correctly
- [ ] Lead time is calculated accurately

---

### Phase 5: Agile Practices (Week 5-6)

**Goal:** Sprints for workers.

```
agile.py
├── Sprint planning
├── Velocity tracking
├── Retrospectives
└── Sprint metrics
```

**Key Implementation Details:**

1. **Velocity is running average:**
   ```python
   def update_velocity(self, sprint_result: SprintResult):
       points = sprint_result.completed_points
       self.velocity_history.append(points)
       if len(self.velocity_history) > 5:
           self.velocity_history = self.velocity_history[-5:]
       self.average_velocity = sum(self.velocity_history) / len(self.velocity_history)
   ```

2. **Retrospectives generate insights:**
   ```python
   def run_retrospective(self, sprint: Sprint) -> Retrospective:
       return Retrospective(
           went_well=[...],  # From successful tasks
           needs_improvement=[...],  # From failed/slow tasks
           action_items=[...]  # Concrete next steps
       )
   ```

**Validation:**
- [ ] Sprints are time-boxed correctly
- [ ] Velocity stabilizes after ~3 sprints
- [ ] Retrospectives produce actionable insights

---

### Phase 6: Learning System (Week 6-7)

**Goal:** Capture and apply experience.

```
learning.py (done)
├── Experience capture
├── Pattern extraction
├── Lesson distillation
└── LearningCycle
```

**Key Implementation Details:**

1. **Experiences are captured automatically:**
   ```python
   # Wrap task execution
   def execute_with_learning(task: Task, context: Context) -> Result:
       experience = learning_cycle.start_experience(context, task.intent)
       try:
           result = worker.execute(task, context)
           experience.complete(Outcome(
               outcome_type=OutcomeType.SUCCESS,
               achieved=[task.goal]
           ))
           return result
       except Exception as e:
           experience.complete(Outcome(
               outcome_type=OutcomeType.FAILURE,
               error_message=str(e)
           ))
           raise
       finally:
           learning_cycle.complete_experience(experience)
   ```

2. **Patterns require minimum evidence:**
   ```python
   # Don't create patterns from single occurrences
   MIN_OCCURRENCES = 3
   if pattern.occurrence_count < MIN_OCCURRENCES:
       continue  # Wait for more evidence
   ```

**Validation:**
- [ ] Experiences are captured for all executions
- [ ] Patterns emerge from repeated structures
- [ ] Lessons can be retrieved for similar contexts

---

### Phase 7: Recovery System (Week 7-8)

**Goal:** Detect and recover from confusion.

```
recovery.py (done)
├── Signal detectors
├── Confusion diagnoser
├── Recovery strategies
└── RecoveryCoordinator
```

**Key Implementation Details:**

1. **Detection is continuous:**
   ```python
   class ConfusionMonitor:
       def wrap_action(self, fn):
           def wrapped(*args, **kwargs):
               result = fn(*args, **kwargs)
               self.check()  # Check after every action
               return result
           return wrapped
   ```

2. **Recovery is layered:**
   ```python
   # Try strategies in order of severity
   strategies = [
       StopAndAnalyzeStrategy(),     # Mild: pause and think
       CheckpointRestoreStrategy(),  # Medium: rollback state
       EscalationStrategy(),         # Severe: ask for help
       UserInterventionStrategy(),   # Critical: involve user
   ]
   ```

**Validation:**
- [ ] Repetition loops are detected within 3 repetitions
- [ ] State mismatches are caught when verifiers exist
- [ ] Recovery strategies execute in order

---

### Phase 8: Evolution (Week 8-10)

**Goal:** Self-improvement through selection.

```
evolution.py (done)
├── StrategyGenome
├── ExecutionSurveyor
├── StrategyAnalyzer
├── StrategyEvolver
└── EvolutionSafeguards
```

**Key Implementation Details:**

1. **Fitness is multi-objective:**
   ```python
   def compute_fitness(metrics: ExecutionMetrics) -> FitnessScore:
       return FitnessScore(
           success_rate=metrics.success_count / metrics.total_count,
           efficiency=1.0 / metrics.average_duration,
           quality=metrics.average_quality,
           predictability=1.0 - metrics.variance
       )
   ```

2. **Elitism preserves best:**
   ```python
   def select(self, population: List[Genome]) -> List[Genome]:
       sorted_pop = sorted(population, key=lambda g: g.fitness)
       elite = sorted_pop[-self.elite_count:]  # Best survive unchanged
       remaining = self.tournament_select(sorted_pop[:-self.elite_count])
       return elite + remaining
   ```

3. **Safeguards prevent regression:**
   ```python
   def validate_new_generation(self, new_gen: List[Genome]) -> bool:
       for test_case in self.golden_tests:
           for genome in new_gen:
               if not self.passes(genome, test_case):
                   return False  # Reject entire generation
       return True
   ```

**Validation:**
- [ ] Fitness improves over generations
- [ ] Diversity is maintained (no collapse to single strategy)
- [ ] Safeguards prevent performance regression

---

## Integration Points

### Event Bus Integration

All components publish/subscribe through central EventBus:

```python
# Component A publishes
bus.publish(Event("task.completed", {"task_id": "T-123", "result": result}))

# Component B subscribes
bus.subscribe("task.completed", self.on_task_completed)
```

**Standard Events:**

| Event Type | Publisher | Subscribers |
|------------|-----------|-------------|
| `goal.submitted` | User/External | Orchestrator |
| `goal.started` | Orchestrator | Director, Metrics |
| `task.started` | Director | Worker, Surveyor |
| `task.completed` | Worker | Director, Surveyor, Learning |
| `task.failed` | Worker | Director, Recovery |
| `confusion.detected` | Recovery | All Agents |
| `decision.made` | Any | Learning, State Manager |

### State Persistence

All stateful components use consistent persistence:

```python
class Persistable(Protocol):
    def save(self, path: Path): ...
    def load(self, path: Path): ...
    def get_state_hash(self) -> str: ...
```

**Persistence Layout:**
```
.cognitive_state/
├── current/
│   ├── focus.json
│   ├── questions.json
│   ├── decisions.json
│   └── hypotheses.json
├── checkpoints/
│   └── 2025-12-28T10-30-00.json
├── experiences/
│   └── exp_20251228_103000_0001.json
├── patterns/
│   └── seq_a1b2c3d4e5f6.json
└── lessons/
    └── lesson_strat_xyz.json
```

### Tool Integration

Tools follow consistent interface:

```python
class Tool(Protocol):
    name: str
    description: str

    def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolContext
    ) -> ToolResult:
        ...
```

**Result standardization:**
```python
@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str]
    metadata: Dict[str, Any]  # timing, resource usage, etc.
```

---

## Testing Strategy

### Unit Tests

Test individual components in isolation:

```python
class TestCognitiveState(unittest.TestCase):
    def test_question_lifecycle(self):
        manager = CognitiveStateManager(Path("/tmp/test"))
        q = manager.add_question("Test?")
        self.assertEqual(q.status, QuestionStatus.OPEN)

        manager.answer_question(q.id, "Answer")
        self.assertEqual(q.status, QuestionStatus.ANSWERED)
```

### Integration Tests

Test component interactions:

```python
class TestAgentHierarchy(unittest.TestCase):
    def test_director_delegates_to_worker(self):
        director = Director()
        workers = [Worker() for _ in range(3)]

        goal = Goal(description="Build feature")
        result = director.execute(goal, workers)

        self.assertTrue(result.success)
        self.assertGreater(len(result.artifacts), 0)
```

### Real-World Tests

Test with actual LLM sessions:

```python
class TestRealUsage(unittest.TestCase):
    def test_multi_session_continuity(self):
        # Session 1: Start work
        session1 = create_session()
        session1.start_goal("Implement authentication")
        session1.make_progress()
        session1.checkpoint()

        # Session 2: Continue work
        session2 = create_session()
        session2.restore_checkpoint()
        self.assertEqual(
            session2.current_goal.description,
            "Implement authentication"
        )
        session2.complete_goal()
```

---

## Common Pitfalls

### 1. Circular Dependencies

**Problem:** Component A imports B, B imports A.

**Solution:** Use protocols for interfaces, implementations for concrete types.

```python
# protocols.py - no circular deps
class WorkerProtocol(Protocol):
    def execute(self, task: Task) -> Result: ...

# director.py - imports protocol only
from .protocols import WorkerProtocol

class Director:
    def __init__(self, workers: List[WorkerProtocol]): ...
```

### 2. Unbounded State Growth

**Problem:** Experiences/patterns/events grow forever.

**Solution:** Implement retention policies.

```python
class ExperienceStore:
    MAX_EXPERIENCES = 10000
    RETENTION_DAYS = 30

    def cleanup(self):
        cutoff = datetime.now() - timedelta(days=self.RETENTION_DAYS)
        self.experiences = [
            e for e in self.experiences
            if e.timestamp > cutoff
        ][:self.MAX_EXPERIENCES]
```

### 3. Detection False Positives

**Problem:** Confusion detected when none exists.

**Solution:** Require multiple signals and confidence thresholds.

```python
def diagnose(self, signals: List[Signal]) -> Optional[Diagnosis]:
    if not signals:
        return None

    # Require multiple signals OR very high confidence
    if len(signals) < 2:
        if all(s.confidence < 0.9 for s in signals):
            return None  # Not enough evidence

    return self._create_diagnosis(signals)
```

### 4. Recovery Loops

**Problem:** Recovery itself triggers confusion detection.

**Solution:** Mark recovery mode, suspend detection during recovery.

```python
class RecoveryCoordinator:
    def recover(self, diagnosis: Diagnosis):
        self.in_recovery = True
        try:
            # Detection is suspended here
            self._execute_recovery(diagnosis)
        finally:
            self.in_recovery = False
```

### 5. Evolution Overfitting

**Problem:** Strategies optimize for recent history, fail on new situations.

**Solution:** Maintain diversity, use holdout validation.

```python
class StrategyEvolver:
    def select(self, population: List[Genome]) -> List[Genome]:
        # Never let diversity fall below threshold
        if self.compute_diversity(population) < self.MIN_DIVERSITY:
            return self.diversity_preserving_select(population)
        return self.fitness_based_select(population)
```

---

## Success Criteria

The system is working when:

1. **State survives sessions:**
   - Open questions remain open after session restart
   - Decisions persist with rationale
   - Checkpoints can be restored

2. **Work flows through hierarchy:**
   - Goals enter backlog
   - Directors decompose to tasks
   - Workers execute tasks
   - Results flow back up

3. **Learning accumulates:**
   - Experiences capture what happened
   - Patterns emerge from repetition
   - Lessons inform future decisions

4. **Confusion is caught:**
   - Repetition loops are detected
   - State mismatches are caught
   - Recovery restores functionality

5. **Performance improves:**
   - Fitness increases over generations
   - Good strategies are selected
   - Bad strategies are eliminated

---

## Next Steps After Implementation

Once the core system is working:

1. **Build domain-specific tools**
   - Code search and modification
   - File system operations
   - External API integrations

2. **Create specialized workers**
   - Coding worker
   - Research worker
   - Documentation worker

3. **Develop dashboards**
   - Kanban board visualization
   - Evolution metrics
   - Recovery statistics

4. **Scale testing**
   - Multi-agent coordination
   - Long-running sessions
   - High-volume goal processing

5. **Production hardening**
   - Error handling
   - Monitoring and alerting
   - Backup and recovery
