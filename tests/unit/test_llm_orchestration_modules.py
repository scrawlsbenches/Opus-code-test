"""
Unit tests for LLM orchestration modules.

Tests for types.py, orchestration.py, agile.py, metrics.py, and evolution.py.

DISABLED: llm_orchestration module scheduled for removal.
"""

import pytest

# Skip entire module - llm_orchestration scheduled for removal
pytestmark = pytest.mark.skip(reason="DISABLED: llm_orchestration module scheduled for removal")

import asyncio
from datetime import datetime, timedelta

from llm_orchestration.types import (
    AgentRole,
    AgentTree,
    Channel,
    Constraint,
    Event,
    EventBus,
    Goal,
    Result,
    SprintTask,
    TaskStatus,
)
from llm_orchestration.orchestration import (
    Bottleneck,
    KanbanColumn,
    KanbanOrchestrator,
    OrchestrationBoard,
    WIPViolation,
)
from llm_orchestration.agile import (
    Estimator,
    EstimationHistory,
    IncrementBuilder,
    RetrospectiveEngine,
    SprintConfig,
    SprintMetricsCollector,
    SprintPlanner,
    VelocityTracker,
    WorkerSprint,
)
from llm_orchestration.metrics import (
    EvolutionTargetIdentifier,
    FitnessCalculator,
    HybridMetrics,
    MetricsCollector,
    MetricsDashboard,
)
from llm_orchestration.evolution import (
    DecompositionPattern,
    DelegationStrategy,
    ExecutionMetrics,
    ExecutionSurveyor,
    ExecutionTrace,
    EvolutionSafeguards,
    FitnessScore,
    StrategyAnalyzer,
    StrategyEvolver,
    StrategyGenome,
    StrategyPool,
)


# =============================================================================
# TYPES.PY TESTS
# =============================================================================


class TestEventBus:
    """Test EventBus subscribe/publish functionality."""

    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""
        bus = EventBus()
        events_received = []

        def handler(event: Event):
            events_received.append(event)

        bus.subscribe("test.event", handler)

        event = Event(type="test.event", payload={"data": "test"})
        asyncio.run(bus.publish(event))

        assert len(events_received) == 1
        assert events_received[0].type == "test.event"
        assert events_received[0].payload["data"] == "test"

    def test_pattern_matching_wildcard(self):
        """Test wildcard pattern matching."""
        bus = EventBus()
        events_received = []

        def handler(event: Event):
            events_received.append(event)

        bus.subscribe("*", handler)

        asyncio.run(bus.publish(Event(type="any.event")))
        asyncio.run(bus.publish(Event(type="another.event")))

        assert len(events_received) == 2

    def test_pattern_matching_prefix(self):
        """Test prefix pattern matching."""
        bus = EventBus()
        events_received = []

        def handler(event: Event):
            events_received.append(event)

        bus.subscribe("task.*", handler)

        asyncio.run(bus.publish(Event(type="task.started")))
        asyncio.run(bus.publish(Event(type="task.completed")))
        asyncio.run(bus.publish(Event(type="agent.spawned")))

        assert len(events_received) == 2

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        events_received = []

        def handler(event: Event):
            events_received.append(event)

        bus.subscribe("test.event", handler)
        bus.unsubscribe("test.event", handler)

        # This should not crash or call handler
        asyncio.run(bus.publish(Event(type="test.event")))
        assert len(events_received) == 0


class TestChannel:
    """Test Channel communication."""

    def test_channel_creation(self):
        """Test creating a channel."""
        channel = Channel(
            channel_id="test-channel",
            from_agent="agent-1",
            to_agent="agent-2",
        )

        assert channel.channel_id == "test-channel"
        assert channel.from_agent == "agent-1"
        assert channel.to_agent == "agent-2"

    def test_send_receive_placeholder(self):
        """Test send/receive placeholders don't crash."""
        channel = Channel(
            channel_id="test",
            from_agent="a",
            to_agent="b",
        )

        # These are placeholders, should not crash
        asyncio.run(channel.send({"message": "test"}))
        result = asyncio.run(channel.receive(timeout=0.1))
        # Result is None in placeholder implementation
        assert result is None


class TestAgentTree:
    """Test AgentTree hierarchy management."""

    def test_add_root_agent(self):
        """Test adding root agent."""
        tree = AgentTree()

        node = tree.add_agent("root", AgentRole.ORCHESTRATOR)

        assert node.agent_id == "root"
        assert node.role == AgentRole.ORCHESTRATOR
        assert node.parent_id is None
        assert tree._root_id == "root"

    def test_add_child_agent(self):
        """Test adding child agents."""
        tree = AgentTree()

        root = tree.add_agent("root", AgentRole.ORCHESTRATOR)
        child = tree.add_agent("child-1", AgentRole.WORKER, parent_id="root")

        assert child.parent_id == "root"
        assert "child-1" in root.children

    def test_get_children(self):
        """Test getting children of an agent."""
        tree = AgentTree()

        tree.add_agent("root", AgentRole.ORCHESTRATOR)
        tree.add_agent("child-1", AgentRole.WORKER, parent_id="root")
        tree.add_agent("child-2", AgentRole.WORKER, parent_id="root")

        children = tree.get_children("root")

        assert len(children) == 2
        assert {c.agent_id for c in children} == {"child-1", "child-2"}

    def test_traverse_depth_first(self):
        """Test depth-first traversal."""
        tree = AgentTree()

        tree.add_agent("root", AgentRole.ORCHESTRATOR)
        tree.add_agent("child-1", AgentRole.DIRECTOR, parent_id="root")
        tree.add_agent("child-2", AgentRole.DIRECTOR, parent_id="root")
        tree.add_agent("grandchild", AgentRole.WORKER, parent_id="child-1")

        nodes = list(tree.traverse_depth_first())

        assert len(nodes) == 4
        assert nodes[0].agent_id == "root"


class TestGoalTaskResult:
    """Test Goal, Task, and Result data classes."""

    def test_goal_creation(self):
        """Test creating a Goal."""
        goal = Goal(
            id="G-001",
            description="Test goal",
            priority=1,
            urgency=0.8,
        )

        assert goal.id == "G-001"
        assert goal.description == "Test goal"
        assert goal.status == TaskStatus.PENDING

    def test_task_creation(self):
        """Test creating a Task."""
        task = SprintTask(
            id="T-001",
            description="Test task",
            estimate_points=3,
        )

        assert task.id == "T-001"
        assert task.estimate_points == 3
        assert task.status == TaskStatus.PENDING

    def test_result_duration(self):
        """Test Result duration calculation."""
        start = datetime.now()
        end = start + timedelta(minutes=5)

        result = Result(
            success=True,
            started_at=start,
            completed_at=end,
        )

        assert result.duration == timedelta(minutes=5)

    def test_result_no_duration(self):
        """Test Result with no duration."""
        result = Result(success=True)

        assert result.duration is None


class TestConstraint:
    """Test Constraint validation."""

    def test_constraint_with_validator(self):
        """Test constraint with custom validator."""
        def validate_positive(value):
            return value > 0

        constraint = Constraint(
            name="positive_value",
            description="Must be positive",
            validator=validate_positive,
        )

        assert constraint.validator(5) is True
        assert constraint.validator(-1) is False


# =============================================================================
# ORCHESTRATION.PY TESTS
# =============================================================================


class TestKanbanColumn:
    """Test KanbanColumn functionality."""

    def test_column_creation(self):
        """Test creating a column."""
        column = KanbanColumn(name="ready", wip_limit=5)

        assert column.name == "ready"
        assert column.wip_limit == 5
        assert column.count == 0

    def test_can_accept_within_limit(self):
        """Test can_accept when under WIP limit."""
        column = KanbanColumn(name="ready", wip_limit=3)
        column.items.append(Goal(id="G-1", description="Test"))

        assert column.can_accept() is True

    def test_can_accept_at_limit(self):
        """Test can_accept when at WIP limit."""
        column = KanbanColumn(name="ready", wip_limit=2)
        column.items.append(Goal(id="G-1", description="Test 1"))
        column.items.append(Goal(id="G-2", description="Test 2"))

        assert column.can_accept() is False

    def test_can_accept_no_limit(self):
        """Test can_accept with no WIP limit."""
        column = KanbanColumn(name="backlog", wip_limit=None)
        for i in range(100):
            column.items.append(Goal(id=f"G-{i}", description=f"Test {i}"))

        assert column.can_accept() is True


class TestOrchestrationBoard:
    """Test OrchestrationBoard operations."""

    def test_get_column(self):
        """Test getting a column by name."""
        board = OrchestrationBoard()

        column = board.get_column("backlog")

        assert column is not None
        assert column.name == "backlog"

    def test_get_nonexistent_column(self):
        """Test getting non-existent column."""
        board = OrchestrationBoard()

        column = board.get_column("nonexistent")

        assert column is None

    def test_add_to_column(self):
        """Test adding a goal to a column."""
        board = OrchestrationBoard()
        goal = Goal(id="G-1", description="Test")

        success = board.add_to_column(goal, "backlog")

        assert success is True
        assert goal in board.get_column("backlog").items

    def test_add_to_full_column(self):
        """Test adding to a column at WIP limit."""
        board = OrchestrationBoard()

        # Fill ready column to limit
        ready = board.get_column("ready")
        for i in range(ready.wip_limit):
            ready.items.append(Goal(id=f"G-{i}", description=f"Test {i}"))

        # Try to add one more
        goal = Goal(id="G-overflow", description="Overflow")
        success = board.add_to_column(goal, "ready")

        assert success is False

    def test_move_between_columns(self):
        """Test moving a goal between columns."""
        board = OrchestrationBoard()
        goal = Goal(id="G-1", description="Test")

        board.add_to_column(goal, "backlog")
        success = board.move(goal, "backlog", "ready")

        assert success is True
        assert goal not in board.get_column("backlog").items
        assert goal in board.get_column("ready").items

    def test_move_respects_wip_limit(self):
        """Test move respects target WIP limit."""
        board = OrchestrationBoard()
        goal = Goal(id="G-overflow", description="Test")

        # Fill target column
        in_progress = board.get_column("in_progress")
        for i in range(in_progress.wip_limit):
            in_progress.items.append(Goal(id=f"G-{i}", description=f"Test {i}"))

        board.add_to_column(goal, "ready")
        success = board.move(goal, "ready", "in_progress")

        assert success is False


class TestKanbanOrchestrator:
    """Test KanbanOrchestrator functionality."""

    def test_submit_goal(self):
        """Test submitting a goal."""
        orchestrator = KanbanOrchestrator()
        goal = Goal(id="G-1", description="Test goal")

        success = asyncio.run(orchestrator.submit_goal(goal))

        assert success is True
        # Goal may be automatically advanced to READY if it meets criteria
        assert goal.status in (TaskStatus.PENDING, TaskStatus.READY)

    def test_wip_limit_enforcement(self):
        """Test WIP limit enforcement."""
        orchestrator = KanbanOrchestrator()

        # Violate WIP limit
        in_progress = orchestrator.board.get_column("in_progress")
        for i in range(in_progress.wip_limit + 2):
            in_progress.items.append(Goal(id=f"G-{i}", description=f"Test {i}"))

        actions = orchestrator.enforce_wip_limits()

        assert len(actions) > 0
        assert any("block" in str(a) or "alert" in str(a) for a in actions)

    def test_detect_bottlenecks(self):
        """Test bottleneck detection."""
        orchestrator = KanbanOrchestrator()

        # Create bottleneck: full in_progress, items waiting in ready
        in_progress = orchestrator.board.get_column("in_progress")
        ready = orchestrator.board.get_column("ready")

        for i in range(in_progress.wip_limit):
            in_progress.items.append(Goal(id=f"G-ip-{i}", description=f"In progress {i}"))

        for i in range(5):
            ready.items.append(Goal(id=f"G-ready-{i}", description=f"Ready {i}"))

        bottlenecks = orchestrator.detect_bottlenecks()

        assert len(bottlenecks) > 0
        assert any(b.location == "in_progress" for b in bottlenecks)

    def test_goal_prioritization(self):
        """Test goal prioritization."""
        orchestrator = KanbanOrchestrator()

        goals = [
            Goal(id="G-1", description="Low priority", urgency=0.2, value=1.0),
            Goal(id="G-2", description="High priority", urgency=0.9, value=10.0),
            Goal(id="G-3", description="Medium priority", urgency=0.5, value=5.0),
        ]

        prioritized = orchestrator._prioritize(goals)

        # High urgency should be first
        assert prioritized[0].id == "G-2"


# =============================================================================
# AGILE.PY TESTS
# =============================================================================


class TestSprintPlanner:
    """Test SprintPlanner capacity calculation."""

    def test_plan_sprint_basic(self):
        """Test basic sprint planning."""
        planner = SprintPlanner()
        tasks = [
            SprintTask(id="T-1", description="Task 1", estimate_points=2),
            SprintTask(id="T-2", description="Task 2", estimate_points=3),
            SprintTask(id="T-3", description="Task 3", estimate_points=5),
        ]

        sprint = planner.plan_sprint(
            goal="Test sprint",
            tasks=tasks,
            velocity=5.0,
            timebox=timedelta(minutes=15),
        )

        assert sprint.estimated_points <= 5
        assert len(sprint.tasks) > 0

    def test_calculate_capacity(self):
        """Test capacity calculation."""
        planner = SprintPlanner()

        capacity = planner._calculate_capacity(
            timebox=timedelta(minutes=30),
            velocity=5.0,
        )

        # 30 minutes = 2x the 15-minute baseline, so capacity should be ~10
        assert capacity == 10

    def test_task_estimation(self):
        """Test task estimation."""
        planner = SprintPlanner()
        task = SprintTask(id="T-1", description="Complex refactoring task")

        estimate = planner.estimate_task(task)

        assert estimate >= 1
        assert estimate <= planner.config.max_points_per_task


class TestVelocityTracker:
    """Test VelocityTracker averaging."""

    def test_record_and_get_velocity(self):
        """Test recording sprints and getting velocity."""
        tracker = VelocityTracker()

        sprints = [
            WorkerSprint(
                sprint_id="S-1",
                goal="Test",
                timebox=timedelta(minutes=15),
                estimated_points=5,
                completed_points=4,
                started_at=datetime.now() - timedelta(minutes=15),
                completed_at=datetime.now(),
            ),
            WorkerSprint(
                sprint_id="S-2",
                goal="Test",
                timebox=timedelta(minutes=15),
                estimated_points=5,
                completed_points=6,
                started_at=datetime.now() - timedelta(minutes=15),
                completed_at=datetime.now(),
            ),
        ]

        for sprint in sprints:
            tracker.record_sprint(sprint)

        velocity = tracker.get_velocity()
        assert velocity > 0

    def test_velocity_trend_improving(self):
        """Test detecting improving velocity trend."""
        tracker = VelocityTracker()

        # Add sprints with increasing velocity
        for i in range(6):
            sprint = WorkerSprint(
                sprint_id=f"S-{i}",
                goal="Test",
                timebox=timedelta(minutes=15),
                estimated_points=5,
                completed_points=i + 1,
                started_at=datetime.now() - timedelta(minutes=15),
                completed_at=datetime.now(),
            )
            tracker.record_sprint(sprint)

        trend = tracker.get_velocity_trend()
        assert trend == "improving"

    def test_predict_completion(self):
        """Test predicting completion time."""
        tracker = VelocityTracker()

        sprint = WorkerSprint(
            sprint_id="S-1",
            goal="Test",
            timebox=timedelta(minutes=15),
            estimated_points=5,
            completed_points=5,
            started_at=datetime.now() - timedelta(minutes=15),
            completed_at=datetime.now(),
        )
        tracker.record_sprint(sprint)

        prediction = tracker.predict_completion(remaining_points=10)

        assert prediction.total_seconds() > 0


class TestEstimator:
    """Test Estimator with historical data."""

    def test_estimate_without_history(self):
        """Test estimation without historical data."""
        estimator = Estimator()
        task = SprintTask(id="T-1", description="New task type")

        estimate = estimator.estimate(task, task_type="new_type")

        assert estimate >= 1

    def test_estimate_with_history(self):
        """Test estimation using historical data."""
        estimator = Estimator()

        # Record some history
        task = SprintTask(id="T-1", description="Test", estimate_points=3)
        estimator.record(task, task_type="feature", actual_points=5)

        # Estimate new task
        new_task = SprintTask(id="T-2", description="Another feature", estimate_points=3)
        estimate = estimator.estimate(new_task, task_type="feature")

        # Should be adjusted based on historical accuracy
        assert estimate >= 1

    def test_heuristic_estimate_complexity(self):
        """Test heuristic estimation based on complexity."""
        estimator = Estimator()

        simple_task = SprintTask(id="T-1", description="Simple small fix")
        complex_task = SprintTask(
            id="T-2",
            description="Complex large refactor with major redesign",
        )

        simple_estimate = estimator._heuristic_estimate(simple_task)
        complex_estimate = estimator._heuristic_estimate(complex_task)

        assert complex_estimate > simple_estimate


class TestRetrospectiveEngine:
    """Test RetrospectiveEngine insights."""

    def test_generate_retrospective_success(self):
        """Test generating retrospective for successful sprint."""
        engine = RetrospectiveEngine()

        sprint = WorkerSprint(
            sprint_id="S-1",
            goal="Test goal",
            timebox=timedelta(minutes=15),
            estimated_points=5,
            completed_points=5,
        )

        retro = engine.generate(sprint)

        assert len(retro.went_well) > 0
        assert "Completed most planned work" in retro.went_well

    def test_generate_retrospective_with_impediments(self):
        """Test retrospective with impediments."""
        engine = RetrospectiveEngine()
        from llm_orchestration.types import Impediment

        sprint = WorkerSprint(
            sprint_id="S-1",
            goal="Test goal",
            timebox=timedelta(minutes=15),
            estimated_points=5,
            completed_points=3,
            impediments=[
                Impediment(task_id="T-1", description="Blocked on external API")
            ],
        )

        retro = engine.generate(sprint)

        assert len(retro.improvements) > 0
        assert retro.impediment_count == 1


class TestSprintMetricsCollector:
    """Test SprintMetricsCollector aggregation."""

    def test_collect_metrics(self):
        """Test collecting and aggregating metrics."""
        collector = SprintMetricsCollector()

        sprints = [
            WorkerSprint(
                sprint_id=f"S-{i}",
                goal="Test",
                timebox=timedelta(minutes=15),
                estimated_points=5,
                completed_points=4,
                started_at=datetime.now() - timedelta(minutes=15),
                completed_at=datetime.now(),
            )
            for i in range(3)
        ]

        for sprint in sprints:
            collector.record(sprint)

        metrics = collector.get_metrics()

        assert metrics.sprint_count == 3
        assert metrics.avg_velocity > 0


# =============================================================================
# METRICS.PY TESTS
# =============================================================================


class TestMetricsCollector:
    """Test MetricsCollector recording."""

    def test_record_data_point(self):
        """Test recording a metric data point."""
        collector = MetricsCollector()

        collector.record("test_metric", 42.0, labels={"env": "test"})

        assert len(collector.data_points) == 1
        assert collector.data_points[0].name == "test_metric"
        assert collector.data_points[0].value == 42.0

    def test_record_goal_completion(self):
        """Test recording goal completion."""
        collector = MetricsCollector()

        start = datetime.now() - timedelta(minutes=10)
        end = datetime.now()

        collector.record_goal_completion(start, end)

        assert len(collector._goal_times) == 1

    def test_get_time_series(self):
        """Test getting time series data."""
        collector = MetricsCollector()

        collector.record("cpu_usage", 50.0)
        collector.record("cpu_usage", 60.0)
        collector.record("memory_usage", 70.0)

        series = collector.get_time_series("cpu_usage")

        assert len(series) == 2


class TestHybridMetrics:
    """Test HybridMetrics aggregation."""

    def test_to_fitness_score(self):
        """Test converting to fitness score."""
        metrics = HybridMetrics(
            throughput=5.0,
            flow_efficiency=0.8,
            wip_stability=0.9,
            sprint_completion_rate=0.85,
            predictability=0.7,
        )

        fitness = metrics.to_fitness_score()

        assert fitness.success == 0.85
        assert fitness.efficiency == 0.8


class TestFitnessCalculator:
    """Test FitnessCalculator scoring."""

    def test_calculate_fitness(self):
        """Test calculating fitness from metrics."""
        calculator = FitnessCalculator()

        metrics = HybridMetrics(
            throughput=3.0,
            flow_efficiency=0.7,
            wip_stability=0.85,
            velocity_stability=0.8,
            estimation_accuracy=0.9,
            sprint_completion_rate=0.88,
            predictability=0.75,
            responsiveness=0.82,
        )

        fitness = calculator.calculate(metrics)

        assert 0 <= fitness.efficiency <= 1
        assert fitness.success == metrics.sprint_completion_rate


class TestMetricsDashboard:
    """Test MetricsDashboard rendering."""

    def test_render_text(self):
        """Test rendering text dashboard."""
        collector = MetricsCollector()
        dashboard = MetricsDashboard(collector)

        text = dashboard.render_text()

        assert "ORCHESTRATION METRICS DASHBOARD" in text
        assert "FLOW METRICS" in text

    def test_get_evolution_recommendations(self):
        """Test getting evolution recommendations."""
        collector = MetricsCollector()
        dashboard = MetricsDashboard(collector)

        recommendations = dashboard.get_evolution_recommendations()

        assert isinstance(recommendations, list)


class TestEvolutionTargetIdentifier:
    """Test EvolutionTargetIdentifier."""

    def test_identify_low_flow_efficiency(self):
        """Test identifying low flow efficiency."""
        identifier = EvolutionTargetIdentifier()

        metrics = HybridMetrics(flow_efficiency=0.3)

        targets = identifier.identify(metrics)

        assert len(targets) > 0
        assert any(t.gene == "coordination_protocols" for t in targets)

    def test_identify_poor_estimation(self):
        """Test identifying poor estimation accuracy."""
        identifier = EvolutionTargetIdentifier()

        metrics = HybridMetrics(estimation_accuracy=0.5)

        targets = identifier.identify(metrics)

        assert any(t.gene == "decomposition_patterns" for t in targets)


# =============================================================================
# EVOLUTION.PY TESTS
# =============================================================================


class TestStrategyGenome:
    """Test StrategyGenome copy and genes."""

    def test_genome_creation(self):
        """Test creating a genome."""
        genome = StrategyGenome(
            genome_id="test-genome",
            exploration_rate=0.1,
            confidence_threshold=0.7,
        )

        assert genome.genome_id == "test-genome"
        assert genome.exploration_rate == 0.1

    def test_genome_copy(self):
        """Test copying a genome."""
        original = StrategyGenome(
            genome_id="original",
            exploration_rate=0.2,
            confidence_threshold=0.8,
        )

        copy = original.copy()

        assert copy.genome_id == "original-copy"
        assert copy.exploration_rate == 0.2
        assert copy.confidence_threshold == 0.8

    def test_genome_genes_iterator(self):
        """Test iterating over genes."""
        genome = StrategyGenome(genome_id="test")

        genes = list(genome.genes())

        assert len(genes) > 0
        gene_names = [name for name, _ in genes]
        assert "exploration_rate" in gene_names


class TestStrategyPool:
    """Test StrategyPool management."""

    def test_add_genome(self):
        """Test adding a genome to the pool."""
        pool = StrategyPool()
        genome = StrategyGenome(genome_id="test-1")

        pool.add(genome)

        assert pool.get("test-1") is not None

    def test_get_random(self):
        """Test getting a random genome."""
        pool = StrategyPool()

        for i in range(5):
            pool.add(StrategyGenome(genome_id=f"genome-{i}"))

        random_genome = pool.get_random()

        assert random_genome is not None

    def test_update_fitness(self):
        """Test updating fitness for a genome."""
        pool = StrategyPool()
        genome = StrategyGenome(genome_id="test-1")
        pool.add(genome)

        pool.update_fitness("test-1", 0.85)

        assert pool._fitness["test-1"] == 0.85
        assert 0.85 in genome.fitness_history


class TestStrategyEvolver:
    """Test StrategyEvolver selection and evolution."""

    def test_select_parents(self):
        """Test selecting parents for reproduction."""
        pool = StrategyPool()
        evolver = StrategyEvolver(pool)

        # Create population with different fitness
        population = [
            StrategyGenome(genome_id=f"genome-{i}")
            for i in range(10)
        ]

        fitness_scores = {
            g.genome_id: FitnessScore(success=i / 10.0)
            for i, g in enumerate(population)
        }

        parents = evolver.select_parents(population, fitness_scores)

        assert len(parents) > 0
        assert len(parents) <= len(population)

    def test_crossover_produces_valid_genome(self):
        """Test crossover produces valid offspring."""
        pool = StrategyPool()
        evolver = StrategyEvolver(pool)

        parent_a = StrategyGenome(
            genome_id="parent-a",
            exploration_rate=0.1,
            confidence_threshold=0.7,
        )
        parent_b = StrategyGenome(
            genome_id="parent-b",
            exploration_rate=0.2,
            confidence_threshold=0.9,
        )

        child = evolver.crossover(parent_a, parent_b)

        assert child.genome_id.startswith("genome-")
        assert 0 <= child.exploration_rate <= 1
        assert 0 <= child.confidence_threshold <= 1

    def test_mutation_within_bounds(self):
        """Test mutation keeps values within valid bounds."""
        pool = StrategyPool()
        evolver = StrategyEvolver(pool)

        genome = StrategyGenome(
            genome_id="test",
            exploration_rate=0.5,
            confidence_threshold=0.5,
            parallelism_preference=0.5,
        )

        mutated = evolver.mutate(genome, mutation_rate=1.0)

        assert 0 <= mutated.exploration_rate <= 1
        assert 0 <= mutated.confidence_threshold <= 1
        assert 0 <= mutated.parallelism_preference <= 1


class TestExecutionSurveyor:
    """Test ExecutionSurveyor tracing."""

    def test_start_trace(self):
        """Test starting an execution trace."""
        surveyor = ExecutionSurveyor()

        trace_id = surveyor.start_trace("Test goal", "genome-1")

        assert trace_id in surveyor.traces
        assert surveyor.traces[trace_id].goal == "Test goal"

    def test_record_event(self):
        """Test recording events to a trace."""
        bus = EventBus()
        surveyor = ExecutionSurveyor(event_bus=bus)

        trace_id = surveyor.start_trace("Test goal", "genome-1")

        event = Event(
            type="agent.spawned",
            trace_id=trace_id,
            payload={"agent_id": "agent-1"},
        )

        asyncio.run(bus.publish(event))

        trace = surveyor.traces[trace_id]
        assert len(trace.event_log) > 0

    def test_finalize_trace(self):
        """Test finalizing a trace with outcome."""
        surveyor = ExecutionSurveyor()

        trace_id = surveyor.start_trace("Test goal", "genome-1")
        result = Result(success=True, output="Done")

        trace = surveyor.finalize_trace(trace_id, result)

        assert trace.result == result
        assert trace.metrics.goal_achieved is True


class TestStrategyAnalyzer:
    """Test StrategyAnalyzer fitness computation."""

    def test_compute_fitness_success(self):
        """Test computing fitness for successful execution."""
        analyzer = StrategyAnalyzer()

        trace = ExecutionTrace(
            trace_id="test",
            goal="Test goal",
            strategy_genome_id="genome-1",
        )
        trace.metrics.goal_achieved = True
        trace.metrics.total_duration_ms = 10000
        trace.metrics.agent_count = 3

        fitness = analyzer.compute_fitness(trace)

        assert fitness.success == 1.0
        assert 0 <= fitness.efficiency <= 1

    def test_compute_fitness_failure(self):
        """Test computing fitness for failed execution."""
        analyzer = StrategyAnalyzer()

        trace = ExecutionTrace(
            trace_id="test",
            goal="Test goal",
            strategy_genome_id="genome-1",
        )
        trace.metrics.goal_achieved = False

        fitness = analyzer.compute_fitness(trace)

        assert fitness.success == 0.0


class TestEvolutionSafeguards:
    """Test EvolutionSafeguards validation."""

    def test_validate_new_generation(self):
        """Test validating a new generation."""
        safeguards = EvolutionSafeguards()

        old_gen = [
            StrategyGenome(genome_id="old-1", exploration_rate=0.1),
            StrategyGenome(genome_id="old-2", exploration_rate=0.2),
        ]

        new_gen = [
            StrategyGenome(genome_id="new-1", exploration_rate=0.15),
            StrategyGenome(genome_id="new-2", exploration_rate=0.25),
        ]

        result = safeguards.validate_new_generation(new_gen, old_gen)

        # Should have some validation (may have issues but shouldn't crash)
        assert isinstance(result.valid, bool)
        assert isinstance(result.issues, list)

    def test_define_golden_strategy(self):
        """Test defining a golden strategy."""
        safeguards = EvolutionSafeguards()
        genome = StrategyGenome(genome_id="golden-1")

        safeguards.define_golden_strategy(genome, "Baseline strategy")

        assert len(safeguards.golden_strategies) == 1
        assert safeguards.golden_strategies[0].genome.genome_id == "golden-1"


class TestFitnessScore:
    """Test FitnessScore aggregation."""

    def test_aggregate_default_weights(self):
        """Test aggregating fitness with default weights."""
        fitness = FitnessScore(
            success=0.9,
            efficiency=0.8,
            quality=0.85,
            stability=0.9,
            elegance=0.7,
            user_satisfaction=0.8,
        )

        aggregate = fitness.aggregate()

        assert 0 <= aggregate <= 1

    def test_aggregate_custom_weights(self):
        """Test aggregating fitness with custom weights."""
        fitness = FitnessScore(
            success=1.0,
            efficiency=0.5,
        )

        weights = {
            "success": 1.0,
            "efficiency": 0.0,
            "quality": 0.0,
            "stability": 0.0,
            "elegance": 0.0,
            "user_satisfaction": 0.0,
        }

        aggregate = fitness.aggregate(weights)

        assert aggregate == 1.0


# =============================================================================
# EDGE CASES AND INTEGRATION
# =============================================================================


class TestEdgeCases:
    """Test edge cases across modules."""

    def test_empty_agent_tree_traversal(self):
        """Test traversing an empty agent tree."""
        tree = AgentTree()

        nodes = list(tree.traverse_depth_first())

        assert len(nodes) == 0

    def test_velocity_tracker_no_sprints(self):
        """Test velocity tracker with no sprint history."""
        tracker = VelocityTracker()

        velocity = tracker.get_velocity()

        # Should return default velocity
        assert velocity == tracker.config.default_velocity

    def test_estimation_history_accuracy_zero_estimate(self):
        """Test estimation accuracy with zero estimates."""
        history = EstimationHistory(task_type="test")
        history.estimates = [0, 0]
        history.actuals = [5, 10]

        accuracy = history.accuracy

        # Should handle gracefully
        assert accuracy == 1.0

    def test_orchestrator_empty_ready_column(self):
        """Test pulling from empty ready column."""
        orchestrator = KanbanOrchestrator()

        goal = asyncio.run(orchestrator.pull_next_goal())

        assert goal is None

    def test_strategy_pool_empty_get_best(self):
        """Test getting best from empty pool."""
        pool = StrategyPool()

        best = pool.get_best_for("any_type")

        assert best is None
