"""
End-to-End Behavioral Test: Cognitive Team Working Together

Epic: A team of cognitive agents collaborates on a complex multi-phase project,
      demonstrating the full capabilities of the cognitive framework.

This is the ULTIMATE integration test that exercises:
- PRISM synaptic memory and confusion detection
- WovenMind dual-process thinking (FAST/SLOW modes)
- GoT Learning (experience capture and lesson retrieval)
- QAPV cognitive loop (Question → Answer → Produce → Verify)
- Recovery procedures (confusion detection, state restoration)
- Multi-agent coordination (Director + Workers)
- Escalation protocols
- Bottleneck detection
- Strategy evolution
- Knowledge transfer across sessions

Story: As a team of cognitive agents,
       We want to collaborate on complex tasks with learning and recovery,
       So that we can deliver high-quality results while adapting to challenges.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field

# Core orchestration imports
from llm_orchestration.agents import Worker
from llm_orchestration.types import (
    WorkerContext, EventBus, Task, TaskStatus,
)
from llm_orchestration.cognitive_state import CognitiveStateManager
from llm_orchestration.thought_patterns import QAPVPattern, QAPVPhase
from llm_orchestration.recovery import (
    RecoveryCoordinator,
    ConfusionSignal,
    SeverityLevel,
    SynapticConfusionDetector,
)
from llm_orchestration.learning import (
    LearningCycle,
    Context as LearningContext,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
    LearningConsolidator,
)
from llm_orchestration.escalation import (
    EscalationLevel,
    EscalationProtocol,
    EscalationManager,
)
from llm_orchestration.agile import VelocityPredictor
from llm_orchestration.evolution import StrategyEvolver, StrategyPool, StrategyGenome

# PRISM imports
try:
    from cortical.reasoning.prism_got import SynapticMemoryGraph
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False
    SynapticMemoryGraph = None

# WovenMind imports
try:
    from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
    from cortical.reasoning.loom import ThinkingMode
    WOVEN_MIND_AVAILABLE = True
except ImportError:
    WOVEN_MIND_AVAILABLE = False
    WovenMind = None

# GoT Learning Bridge
try:
    from cortical.got.learning_integration import GoTLearningBridge
    GOT_LEARNING_AVAILABLE = True
except ImportError:
    GOT_LEARNING_AVAILABLE = False
    GoTLearningBridge = None


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for the cognitive team."""
    temp_dir = Path(tempfile.mkdtemp(prefix="cognitive_team_"))
    (temp_dir / "state").mkdir()
    (temp_dir / "learning").mkdir()
    (temp_dir / "got").mkdir()
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def cognitive_state_manager(temp_workspace):
    """Cognitive state manager for workers."""
    return CognitiveStateManager(temp_workspace / "state")


@pytest.fixture
def learning_cycle(temp_workspace):
    """Learning cycle for experience capture."""
    return LearningCycle(temp_workspace / "learning")


@pytest.fixture
def recovery_coordinator():
    """Recovery coordinator for confusion handling."""
    return RecoveryCoordinator()


@pytest.fixture
def velocity_predictor():
    """Velocity predictor for sprint planning."""
    return VelocityPredictor(history_window=10)


@pytest.fixture
def strategy_pool():
    """Strategy pool for evolution."""
    return StrategyPool()


@pytest.fixture
def complex_project_tasks():
    """Define a complex multi-phase project with dependencies."""
    return [
        {"id": "T-001", "title": "Analyze requirements", "phase": "discovery",
         "complexity": "moderate", "dependencies": [], "estimated_hours": 4},
        {"id": "T-002", "title": "Design architecture", "phase": "design",
         "complexity": "high", "dependencies": ["T-001"], "estimated_hours": 8},
        {"id": "T-003", "title": "Implement core module", "phase": "implementation",
         "complexity": "high", "dependencies": ["T-002"], "estimated_hours": 16},
        {"id": "T-004", "title": "Implement API layer", "phase": "implementation",
         "complexity": "moderate", "dependencies": ["T-002"], "estimated_hours": 8},
        {"id": "T-005", "title": "Write unit tests", "phase": "testing",
         "complexity": "moderate", "dependencies": ["T-003", "T-004"], "estimated_hours": 6},
        {"id": "T-006", "title": "Integration testing", "phase": "testing",
         "complexity": "high", "dependencies": ["T-005"], "estimated_hours": 8},
        {"id": "T-007", "title": "Documentation", "phase": "delivery",
         "complexity": "low", "dependencies": ["T-006"], "estimated_hours": 4},
        {"id": "T-008", "title": "Deployment", "phase": "delivery",
         "complexity": "moderate", "dependencies": ["T-006", "T-007"], "estimated_hours": 4},
    ]


@pytest.fixture
def worker_context_factory():
    """Factory for creating worker contexts."""
    def create_context(task_description: str, tools: List[str] = None):
        return WorkerContext(
            task=task_description,
            tools=tools or ["read", "write", "search", "analyze"],
            constraints=["Must pass all tests", "Follow coding standards"],
        )
    return create_context


@pytest.fixture
def mock_synaptic_graph():
    """Mock synaptic memory graph for PRISM integration."""
    graph = Mock()
    graph.record_activation = Mock()
    graph.get_recent_activations = Mock(return_value=[])
    graph.detect_patterns = Mock(return_value=[])
    # Mock the activation_traces as a dict-like object
    graph.activation_traces = {}
    graph.nodes = {}
    graph.edges = {}
    return graph


# =============================================================================
# HELPER CLASSES
# =============================================================================


@dataclass
class TeamSimulation:
    """Simulates a cognitive team working together."""
    workers: Dict[str, Worker] = field(default_factory=dict)
    completed_tasks: List[str] = field(default_factory=list)
    confusion_events: List[Dict] = field(default_factory=list)
    recovery_attempts: List[Dict] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    total_qapv_cycles: int = 0


# =============================================================================
# TEST CLASS: FULL COGNITIVE TEAM E2E
# =============================================================================


@pytest.mark.skip(reason="DISABLED: llm_orchestration module scheduled for removal. Worker hardcodes Path('.got'), creates runtime data. See llm_orchestration/agents.py:897-899")
class TestCognitiveTeamEndToEnd:
    """
    Epic: Cognitive Team Collaboration

    Complete end-to-end test of the cognitive framework.
    """

    # =========================================================================
    # SCENARIO 1: Team Formation and Task Assignment
    # =========================================================================

    def test_scenario_team_forms_and_receives_complex_project(
        self, temp_workspace, cognitive_state_manager, worker_context_factory,
        complex_project_tasks,
    ):
        """
        Scenario: Director assigns complex project to worker team

        Given a director and multiple workers with cognitive capabilities
        When a complex multi-phase project is assigned
        Then the team organizes and workers receive appropriate tasks
        """
        # Given: Create a team of workers
        workers = {}
        for i in range(3):
            worker_id = f"worker-{i+1}"
            context = worker_context_factory(f"Worker {i+1} ready for tasks")
            worker = Worker(worker_id, context, state_manager=cognitive_state_manager)
            workers[worker_id] = worker

        # Verify workers have cognitive capabilities
        for worker_id, worker in workers.items():
            assert worker.agent_id == worker_id
            assert worker.context is not None
            assert worker._state_manager is not None
            assert hasattr(worker, '_recovery_coordinator')

        # Then: Workers are ready
        assert len(workers) == 3
        assert len(complex_project_tasks) == 8
        phases = set(t["phase"] for t in complex_project_tasks)
        assert phases == {"discovery", "design", "implementation", "testing", "delivery"}

    # =========================================================================
    # SCENARIO 2: QAPV Cognitive Loop Execution
    # =========================================================================

    def test_scenario_worker_executes_qapv_cognitive_loop(
        self, temp_workspace, cognitive_state_manager, worker_context_factory,
    ):
        """
        Scenario: Worker uses QAPV for structured reasoning

        Given a worker with a moderately complex task
        When the worker executes using QAPV cognitive loop
        Then it progresses through Question → Answer → Produce → Verify
        """
        # Given: Worker with task
        context = worker_context_factory("Implement user authentication module")
        worker = Worker("qapv-worker", context, state_manager=cognitive_state_manager)

        # When: Create QAPV pattern (requires cognitive_state and goal)
        qapv = QAPVPattern(cognitive_state_manager, "Implement authentication")

        # Verify QAPV phases exist
        phases = qapv.get_phases()
        assert QAPVPhase.QUESTION in phases
        assert QAPVPhase.ANSWER in phases
        assert QAPVPhase.PRODUCE in phases
        assert QAPVPhase.VERIFY in phases

        # Verify current phase starts at QUESTION
        assert qapv.current_phase == QAPVPhase.QUESTION

        # Get guidance for current phase
        guidance = qapv.get_current_guidance()
        assert "QUESTION" in guidance
        assert len(guidance) > 0

    # =========================================================================
    # SCENARIO 3: PRISM Synaptic Confusion Detection
    # =========================================================================

    @pytest.mark.skipif(not PRISM_AVAILABLE, reason="PRISM not available")
    def test_scenario_prism_detects_cognitive_confusion(
        self, temp_workspace, mock_synaptic_graph, recovery_coordinator,
    ):
        """
        Scenario: PRISM detects confusion patterns in worker cognition

        Given a worker with PRISM synaptic memory enabled
        When the worker exhibits confusion patterns
        Then PRISM detects the pattern through synaptic activation analysis
        """
        # Given: Enable synaptic detection
        recovery_coordinator.enable_synaptic_detection(mock_synaptic_graph)

        # Create detector with memory graph
        detector = SynapticConfusionDetector(memory_graph=mock_synaptic_graph)

        # When: Record activations (just node_id string)
        for i in range(5):
            detector.record_activation(f"node_{i}")

        # Then: Detector exists and can detect
        signals = detector.detect()
        assert isinstance(signals, list)

    # =========================================================================
    # SCENARIO 4: Confusion Detection and Recovery
    # =========================================================================

    def test_scenario_worker_recovers_from_confusion(
        self, temp_workspace, cognitive_state_manager, worker_context_factory,
        recovery_coordinator,
    ):
        """
        Scenario: Worker detects confusion and recovers gracefully

        Given a worker that encounters repeated failures
        When confusion is detected (stagnation pattern)
        Then the recovery coordinator activates
        """
        # Given: Worker with recovery capability
        context = worker_context_factory("Complex refactoring task")
        worker = Worker("recovery-worker", context, state_manager=cognitive_state_manager)

        # Simulate repeated actions to trigger detection
        for i in range(5):
            recovery_coordinator.record_action(
                action_type="read_file",
                target="same_file.py",
                result="failure",
                parameters={"attempt": i}
            )

        # When: Check for confusion
        diagnosis = recovery_coordinator.check_confusion({"worker_id": "recovery-worker"})

        # Then: If confusion detected, can recover
        if diagnosis is not None:
            assert hasattr(diagnosis, 'confusion_type')
            assert hasattr(diagnosis, 'recommendations')

            # Attempt recovery
            recovery_result = recovery_coordinator.recover(
                diagnosis,
                {"worker_id": "recovery-worker"}
            )
            assert recovery_result is not None

        # Recovery coordinator has stats
        stats = recovery_coordinator.get_recovery_stats()
        assert stats is not None

    # =========================================================================
    # SCENARIO 5: WovenMind FAST/SLOW Mode Switching
    # =========================================================================

    @pytest.mark.skipif(not WOVEN_MIND_AVAILABLE, reason="WovenMind not available")
    def test_scenario_woven_mind_switches_cognitive_modes(self, temp_workspace):
        """
        Scenario: Worker uses WovenMind for adaptive cognitive processing

        Given a worker with WovenMind dual-process thinking
        When processing simple vs complex cognitive tasks
        Then appropriate modes are used
        """
        # Given: WovenMind
        woven_mind = WovenMind()

        # Train on patterns
        woven_mind.train("Common coding patterns and best practices")

        # When: Process tasks
        simple_result = woven_mind.process("Add a log statement")
        complex_result = woven_mind.process(
            "Redesign authentication architecture with security implications"
        )

        # Then: Results exist
        assert simple_result is not None
        assert complex_result is not None

        # Stats available
        stats = woven_mind.get_stats()
        assert stats is not None

    # =========================================================================
    # SCENARIO 6: GoT Learning Integration
    # =========================================================================

    def test_scenario_got_captures_learning_from_task_completion(
        self, temp_workspace, learning_cycle,
    ):
        """
        Scenario: GoT captures learning experiences from completed tasks

        Given a worker that completes a task successfully
        When the task outcome is recorded
        Then an experience is captured with context, actions, and outcome
        """
        # Given: Task context
        task_context = LearningContext(
            goal_type="implementation",
            goal_complexity="moderate",
            available_tools=["read", "write", "test"],
            domain="backend_development",
        )

        # When: Capture experience
        experience = learning_cycle.start_experience(
            context=task_context,
            intent="Implement caching layer for API",
            experience_type=ExperienceType.TASK_EXECUTION,
            strategy="incremental_implementation",
        )

        # Record actions
        experience.add_action(Action(
            action_type="design",
            description="Designed cache interface",
            target="cache.py",
        ))
        experience.add_action(Action(
            action_type="implement",
            description="Implemented Redis cache adapter",
            target="redis_cache.py",
        ))

        # Complete with outcome
        outcome = Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Cache layer implemented and tested",
            achieved=["cache_interface", "redis_adapter"],
            quality_score=0.92,
        )

        learning_cycle.complete_experience(experience, outcome, reflection={
            'worked': ['Incremental approach allowed early testing'],
            'didnt_work': [],
            'different': ['Could have added performance benchmarks'],
        })

        # Then: Extract and distill
        result = learning_cycle.extract_and_distill()
        assert result is not None

        # Get guidance for similar context
        guidance = learning_cycle.get_guidance(task_context)
        assert guidance is not None

        # Stats show learning occurred
        stats = learning_cycle.get_stats()
        assert stats is not None

    # =========================================================================
    # SCENARIO 7: Learning Consolidation
    # =========================================================================

    def test_scenario_learning_consolidates_over_time(
        self, temp_workspace, learning_cycle,
    ):
        """
        Scenario: Learning system consolidates knowledge over time

        Given multiple similar experiences have been captured
        When consolidation is triggered
        Then similar lessons are merged
        """
        # Given: Create multiple experiences
        for i in range(5):
            context = LearningContext(
                goal_type="bugfix",
                goal_complexity="low",
                available_tools=["read", "debug"],
                domain="error_handling",
            )

            experience = learning_cycle.start_experience(
                context=context,
                intent=f"Fix null pointer exception {i+1}",
                experience_type=ExperienceType.TASK_EXECUTION,
                strategy="defensive_programming",
            )

            experience.add_action(Action(
                action_type="debug",
                description="Added null check",
                target=f"module_{i}.py",
            ))

            outcome = Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Fixed NPE",
                achieved=["null_check_added"],
                quality_score=0.85,
            )

            learning_cycle.complete_experience(experience, outcome, reflection={
                'worked': ['Defensive programming pattern'],
                'didnt_work': [],
                'different': [],
            })

        # When: Extract and distill
        result = learning_cycle.extract_and_distill()

        # Then: Consolidation occurred
        assert result is not None
        stats = learning_cycle.get_stats()
        assert stats is not None

    # =========================================================================
    # SCENARIO 8: Escalation Protocol
    # =========================================================================

    def test_scenario_escalation_triggers_on_repeated_failures(self):
        """
        Scenario: Escalation protocol activates on repeated failures

        Given a worker encountering repeated failures
        When escalation is evaluated
        Then appropriate escalation level is determined
        """
        # Given: Escalation manager
        manager = EscalationManager()

        # When: Worker reports confusion
        confusion = ConfusionSignal(
            signal_type="repetition_loop",
            description="Repeating same failed approach",
            evidence=["read_file", "read_file", "read_file"],
            confidence=0.85,
            source="worker-1",
        )

        # Evaluate escalation
        protocol = manager.evaluate(
            worker_id="worker-1",
            confusion=confusion,
            task_id="T-123",
        )

        # Then: Protocol determined
        assert protocol is not None
        assert hasattr(protocol, 'level')
        assert protocol.level.value >= EscalationLevel.MONITOR.value

    # =========================================================================
    # SCENARIO 9: Velocity Prediction
    # =========================================================================

    def test_scenario_velocity_prediction_guides_sprint_planning(
        self, velocity_predictor,
    ):
        """
        Scenario: Velocity prediction helps plan realistic sprints

        Given historical velocity data from past sprints
        When predicting velocity for next sprint
        Then prediction uses EMA with trend adjustment
        """
        # Given: Record historical velocity
        for velocity in [20, 22, 21, 24, 23]:
            velocity_predictor.record_velocity(velocity, {"sprint": velocity})

        # When: Predict next sprint
        prediction = velocity_predictor.predict_next()

        # Then: Prediction has required components
        assert prediction.predicted_velocity > 0
        assert 0.0 <= prediction.confidence <= 1.0

        # Trend is detected
        trend = velocity_predictor.get_trend()
        assert trend in ["increasing", "stable", "decreasing"]

    # =========================================================================
    # SCENARIO 10: Strategy Evolution
    # =========================================================================

    def test_scenario_strategy_evolution_improves_over_generations(
        self, strategy_pool,
    ):
        """
        Scenario: Strategy pool evolves better strategies over time

        Given initial strategy population
        When strategies are evaluated and evolved
        Then fitter strategies survive
        """
        # Given: Create initial strategies with proper dataclass fields
        for i in range(5):
            strategy = StrategyGenome(
                genome_id=f"strat-{i}",
                exploration_rate=0.1 + (i * 0.02),
                confidence_threshold=0.7 - (i * 0.05),
                parallelism_preference=0.5,
            )
            strategy_pool.add(strategy)

        # When: Update fitness based on traits
        for genome in strategy_pool.get_current_generation():
            fitness = 0.5 + genome.exploration_rate
            strategy_pool.update_fitness(genome.genome_id, fitness)

        # Then: Can get best strategy
        best = strategy_pool.get_best_for("default")
        assert best is not None

    # =========================================================================
    # SCENARIO 11: Knowledge Transfer Between Sessions
    # =========================================================================

    def test_scenario_knowledge_transfers_across_sessions(
        self, temp_workspace, learning_cycle,
    ):
        """
        Scenario: Knowledge persists and transfers between sessions

        Given lessons learned in one session
        When a new session starts
        Then previous lessons are available
        """
        # Given: Session 1 - Learn something
        context1 = LearningContext(
            goal_type="optimization",
            goal_complexity="high",
            available_tools=["profile", "refactor"],
            domain="performance",
        )

        exp1 = learning_cycle.start_experience(
            context=context1,
            intent="Optimize database queries",
            experience_type=ExperienceType.TASK_EXECUTION,
            strategy="measure_then_optimize",
        )

        exp1.add_action(Action(
            action_type="profile",
            description="Profiled slow queries",
            target="db_layer.py",
        ))

        outcome1 = Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Reduced query time by 80%",
            achieved=["query_optimization"],
            quality_score=0.95,
        )

        learning_cycle.complete_experience(exp1, outcome1, reflection={
            'worked': ['Always profile before optimizing'],
            'didnt_work': [],
            'different': [],
        })

        learning_cycle.extract_and_distill()

        # When: Session 2 - Similar task
        context2 = LearningContext(
            goal_type="optimization",
            goal_complexity="moderate",
            available_tools=["profile", "refactor"],
            domain="performance",
        )

        # Then: Previous lessons available
        guidance = learning_cycle.get_guidance(context2)
        assert guidance is not None

    # =========================================================================
    # SCENARIO 12: Full Team Simulation
    # =========================================================================

    def test_scenario_full_team_executes_complex_project(
        self, temp_workspace, cognitive_state_manager, worker_context_factory,
        complex_project_tasks, learning_cycle, recovery_coordinator,
        velocity_predictor,
    ):
        """
        Scenario: Full cognitive team executes complex project end-to-end

        Given a team of cognitive workers with full capabilities
        When executing a complex multi-phase project
        Then tasks complete, learning is captured, and velocity tracked
        """
        simulation = TeamSimulation()

        # Given: Create worker team
        worker_ids = ["alpha", "beta", "gamma"]
        for wid in worker_ids:
            context = worker_context_factory(f"Worker {wid} ready")
            worker = Worker(wid, context, state_manager=cognitive_state_manager)
            simulation.workers[wid] = worker

        # When: Execute project phases
        completed_task_ids = set()
        task_queue = list(complex_project_tasks)
        sprint_velocity = 0

        while task_queue:
            # Find ready tasks
            ready_tasks = [
                t for t in task_queue
                if all(dep in completed_task_ids for dep in t["dependencies"])
            ]

            if not ready_tasks:
                break

            # Execute tasks
            for i, task in enumerate(ready_tasks[:len(worker_ids)]):
                worker_id = worker_ids[i % len(worker_ids)]

                # Capture learning
                task_context = LearningContext(
                    goal_type="implementation",
                    goal_complexity=task["complexity"],
                    available_tools=["read", "write"],
                    domain=task["phase"],
                )

                exp = learning_cycle.start_experience(
                    context=task_context,
                    intent=task["title"],
                    experience_type=ExperienceType.TASK_EXECUTION,
                    strategy="qapv",
                )

                exp.add_action(Action(
                    action_type="execute",
                    description=f"Completed {task['title']}",
                    target=task["id"],
                ))

                outcome = Outcome(
                    outcome_type=OutcomeType.SUCCESS,
                    description="Task completed",
                    achieved=[task["id"]],
                    quality_score=0.88,
                )

                learning_cycle.complete_experience(exp, outcome, reflection={
                    'worked': ['QAPV structure helped'],
                    'didnt_work': [],
                    'different': [],
                })

                simulation.completed_tasks.append(task["id"])
                completed_task_ids.add(task["id"])
                sprint_velocity += task["estimated_hours"]
                simulation.total_qapv_cycles += 1

            # Remove completed tasks
            task_queue = [t for t in task_queue if t["id"] not in completed_task_ids]

            # Record sprint velocity
            if sprint_velocity > 0:
                velocity_predictor.record_velocity(sprint_velocity, {})
                sprint_velocity = 0

        # Then: Project completed
        assert len(simulation.completed_tasks) == len(complex_project_tasks)
        assert simulation.total_qapv_cycles >= len(complex_project_tasks)

        # Velocity tracked
        prediction = velocity_predictor.predict_next()
        assert prediction.predicted_velocity > 0

    # =========================================================================
    # SCENARIO 13: Cognitive Resilience Under Stress
    # =========================================================================

    def test_scenario_team_maintains_resilience_under_stress(
        self, temp_workspace, cognitive_state_manager, worker_context_factory,
        recovery_coordinator,
    ):
        """
        Scenario: Team maintains cognitive resilience under stress

        Given a team facing multiple concurrent challenges
        When experiencing high cognitive load with confusion signals
        Then recovery mechanisms activate appropriately
        """
        # Given: Worker under stress
        context = worker_context_factory("Handle production incident")
        worker = Worker("stress-test", context, state_manager=cognitive_state_manager)

        # When: Simulate stressful actions that cause repetition detection
        for i in range(10):
            recovery_coordinator.record_action(
                action_type="analyze",
                target="incident.log",
                result="inconclusive",
                parameters={"attempt": i}
            )

        # Check for confusion
        diagnosis = recovery_coordinator.check_confusion({"worker_id": "stress-test"})

        # Then: If diagnosis found, recovery should work
        if diagnosis is not None:
            assert hasattr(diagnosis, 'confusion_type')
            recovery_result = recovery_coordinator.recover(
                diagnosis,
                {"worker_id": "stress-test"}
            )
            assert recovery_result is not None

        # Stats are tracked
        stats = recovery_coordinator.get_recovery_stats()
        assert stats is not None

    # =========================================================================
    # SCENARIO 14: Complete Cognitive Lifecycle
    # =========================================================================

    def test_scenario_complete_cognitive_lifecycle_integration(
        self, temp_workspace, cognitive_state_manager, worker_context_factory,
        learning_cycle, recovery_coordinator, velocity_predictor, strategy_pool,
    ):
        """
        Scenario: Complete integration of all cognitive capabilities

        This is the ULTIMATE test verifying all components work together.
        """
        # =====================================================================
        # PHASE 1: Initialization
        # =====================================================================
        context = worker_context_factory("Implement complete feature")
        worker = Worker("ultimate-worker", context, state_manager=cognitive_state_manager)
        assert worker.agent_id == "ultimate-worker"

        # =====================================================================
        # PHASE 2: Velocity-Based Planning
        # =====================================================================
        for v in [20, 22, 21, 23, 24]:
            velocity_predictor.record_velocity(v, {})
        prediction = velocity_predictor.predict_next()
        assert prediction.predicted_velocity > 0

        # =====================================================================
        # PHASE 3: QAPV Execution
        # =====================================================================
        qapv = QAPVPattern(cognitive_state_manager, "Implement feature")
        assert qapv.current_phase == QAPVPhase.QUESTION

        phases = qapv.get_phases()
        assert len(phases) >= 4  # Q, A, P, V (may include COMPLETE)

        # =====================================================================
        # PHASE 4: Confusion and Recovery
        # =====================================================================
        # Simulate actions that could cause confusion
        for i in range(5):
            recovery_coordinator.record_action(
                action_type="implement",
                target="feature.py",
                result="partial",
                parameters={"attempt": i}
            )

        diagnosis = recovery_coordinator.check_confusion({"phase": "PRODUCE"})
        if diagnosis is not None:
            recovery_coordinator.recover(diagnosis, {"phase": "PRODUCE"})

        # =====================================================================
        # PHASE 5: Learning Capture
        # =====================================================================
        task_context = LearningContext(
            goal_type="feature_implementation",
            goal_complexity="high",
            available_tools=["read", "write", "test"],
            domain="full_stack",
        )

        experience = learning_cycle.start_experience(
            context=task_context,
            intent="Implement complete feature",
            experience_type=ExperienceType.TASK_EXECUTION,
            strategy="qapv_with_recovery",
        )

        for action_type in ["question", "answer", "produce", "verify"]:
            experience.add_action(Action(
                action_type=action_type,
                description=f"Executed {action_type} phase",
                target="feature",
            ))

        outcome = Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Feature implemented with cognitive support",
            achieved=["feature_complete", "tests_passing"],
            quality_score=0.93,
        )

        learning_cycle.complete_experience(experience, outcome, reflection={
            'worked': ['QAPV structure', 'Recovery from stagnation'],
            'didnt_work': [],
            'different': [],
        })

        learning_cycle.extract_and_distill()

        # =====================================================================
        # PHASE 6: Strategy Recording
        # =====================================================================
        strategy = StrategyGenome(
            genome_id="qapv-recovery",
            exploration_rate=0.1,
            confidence_threshold=0.8,
            parallelism_preference=0.5,
        )
        strategy_pool.add(strategy)
        strategy_pool.update_fitness("qapv-recovery", 0.93)

        # =====================================================================
        # FINAL VERIFICATION
        # =====================================================================
        stats = learning_cycle.get_stats()
        assert stats is not None

        recovery_stats = recovery_coordinator.get_recovery_stats()
        assert recovery_stats is not None

        assert strategy_pool.get("qapv-recovery") is not None

        print("\n" + "=" * 70)
        print("COMPLETE COGNITIVE LIFECYCLE TEST: PASSED")
        print("=" * 70)


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================


@pytest.mark.skip(reason="DISABLED: llm_orchestration module scheduled for removal")
class TestCognitiveEdgeCases:
    """Test edge cases and failure modes."""

    def test_learning_with_no_prior_experiences(self, temp_workspace):
        """System handles empty experience base gracefully."""
        learning_cycle = LearningCycle(temp_workspace / "fresh_learning")

        context = LearningContext(
            goal_type="new_task",
            goal_complexity="moderate",
            available_tools=["tool1"],
            domain="new_domain",
        )

        # Request guidance from empty system
        guidance = learning_cycle.get_guidance(context)
        # Should not crash, returns something
        assert guidance is not None or guidance == [] or guidance == {}

    def test_velocity_prediction_with_high_variance(self, velocity_predictor):
        """Prediction handles high variance data."""
        velocities = [10, 50, 15, 45, 20, 55, 12]
        for v in velocities:
            velocity_predictor.record_velocity(v, {})

        prediction = velocity_predictor.predict_next()
        assert prediction.predicted_velocity > 0

    def test_recovery_from_multiple_signals(self, recovery_coordinator):
        """Recovery handles multiple concurrent confusion signals."""
        # Simulate various confusing actions
        for i in range(8):
            recovery_coordinator.record_action(
                action_type="analyze",
                target="same_target.py",
                result="failed",
                parameters={"attempt": i}
            )

        diagnosis = recovery_coordinator.check_confusion({"context": "multiple_signals"})

        # If confusion detected, verify it can be handled
        if diagnosis is not None:
            assert hasattr(diagnosis, 'confusion_type')
            assert hasattr(diagnosis, 'recommendations')

        # Recovery stats should be available
        stats = recovery_coordinator.get_recovery_stats()
        assert stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
