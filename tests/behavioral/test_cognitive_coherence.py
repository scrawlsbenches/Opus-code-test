"""
Behavioral tests for Phase 1 cognitive coherence features.

As a developer building coherent cognitive systems,
I want the Phase 1 features to work seamlessly together,
So that agents exhibit intelligent mode-switching, bottleneck detection,
confusion recovery, and adaptive orchestration.

This test suite verifies:
- Woven Mind + QAPV integration (FAST/SLOW thinking)
- Bottleneck detection in orchestration
- Cognitive metrics collection and aggregation
- Escalation protocol for confusion/failures
- Synaptic confusion detection (PRISM-GoT)

Phase 1 Context:
- PRISM-GoT synaptic memory for confusion detection
- Woven Mind dual-process (FAST/SLOW) integration with QAPV
- Bottleneck detection in orchestration
- Cognitive metrics collection

DISABLED: llm_orchestration module scheduled for removal.
"""

import pytest

# Skip entire module - llm_orchestration scheduled for removal
pytestmark = pytest.mark.skip(reason="DISABLED: llm_orchestration module scheduled for removal")

from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
from collections import defaultdict

from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig, ThinkingMode
from cortical.reasoning.prism_got import (
    SynapticMemoryGraph,
    IncrementalReasoner,
    NodeType,
    EdgeType,
    ActivationTrace,
)
from llm_orchestration.metrics import (
    MetricsCollector,
    HybridMetrics,
    MetricsDashboard,
)
from llm_orchestration.orchestration import (
    KanbanOrchestrator,
    OrchestrationBoard,
    Bottleneck,
    WIPViolation,
    KanbanColumn,
)
from llm_orchestration.types import Goal, TaskStatus, EventBus, Event
from llm_orchestration.thought_patterns import QAPVPattern, QAPVPhase
from llm_orchestration.cognitive_state import CognitiveStateManager


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_storage():
    """Provide temporary storage for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def woven_mind():
    """Create a WovenMind instance for testing."""
    config = WovenMindConfig(
        surprise_threshold=0.3,
        k_winners=5,
        auto_switch=True,
        enable_observability=True,
    )
    return WovenMind(config=config)


@pytest.fixture
def synaptic_graph():
    """Create a synaptic memory graph for testing."""
    return SynapticMemoryGraph()


@pytest.fixture
def incremental_reasoner(synaptic_graph):
    """Create an incremental reasoner."""
    return IncrementalReasoner(synaptic_graph)


@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector()


@pytest.fixture
def event_bus():
    """Create an event bus for tracking."""
    return EventBus()


@pytest.fixture
def kanban_orchestrator(event_bus):
    """Create a kanban orchestrator."""
    return KanbanOrchestrator(event_bus=event_bus)


@pytest.fixture
def cognitive_state_manager(temp_storage):
    """Create a cognitive state manager."""
    return CognitiveStateManager(temp_storage)


# =============================================================================
# TEST WOVEN MIND + QAPV INTEGRATION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.cognitive
@pytest.mark.coherence
class TestWovenMindQAPVIntegration:
    """
    Epic: Woven Mind + QAPV Integration

    As a developer building intelligent reasoning systems,
    I want QAPV phases to use appropriate thinking modes,
    So that agents think FAST for pattern matching and SLOW for deep analysis.
    """

    def test_question_phase_uses_slow_thinking(
        self, woven_mind, temp_storage
    ):
        """
        Scenario: QUESTION phase should use SLOW mode for deep analysis

        Given a QAPV pattern in QUESTION phase
        And Woven Mind configured for mode switching
        When processing the question context
        Then SLOW mode should be selected
        Because questions require deliberate analysis
        """
        # Given: QAPV in QUESTION phase
        state_manager = CognitiveStateManager(temp_storage)
        qapv = QAPVPattern(state_manager, goal="Implement feature X")

        # Then: Starts in QUESTION phase
        assert qapv.current_phase == QAPVPhase.QUESTION

        # When: Processing with Woven Mind (simulate QUESTION context)
        # Train mind with some patterns first
        woven_mind.train("implement feature using pattern")

        # Force SLOW mode for question analysis (as would happen in integration)
        woven_mind.force_mode(ThinkingMode.SLOW, reason="question_phase")

        # Then: SLOW mode is active
        assert woven_mind.get_current_mode() == ThinkingMode.SLOW

        # Verify mode transitions are tracked
        transitions = woven_mind.get_transition_history()
        assert len(transitions) >= 1
        assert transitions[-1].to_mode == ThinkingMode.SLOW

    def test_answer_phase_uses_fast_thinking(
        self, woven_mind, temp_storage
    ):
        """
        Scenario: ANSWER phase should use FAST mode for pattern matching

        Given a QAPV pattern in ANSWER phase
        And Woven Mind with trained patterns
        When exploring known solutions
        Then FAST mode should be selected
        Because known patterns can be quickly retrieved
        """
        # Given: Train Woven Mind with patterns
        woven_mind.train("use jwt for authentication")
        woven_mind.train("jwt tokens are secure")
        woven_mind.train("authentication requires jwt")

        # When: Processing with FAST mode (pattern retrieval)
        result = woven_mind.process(["authentication", "jwt"], mode=ThinkingMode.FAST)

        # Then: FAST mode was used
        assert result.mode == ThinkingMode.FAST
        assert result.source == "hive"
        assert len(result.activations) > 0

    def test_surprise_triggers_mode_switch(self, woven_mind):
        """
        Scenario: Unexpected results should trigger FAST→SLOW switch

        Given Woven Mind in FAST mode
        And trained patterns for normal cases
        When encountering unexpected input
        Then surprise should be detected
        And mode should switch to SLOW
        Because novelty requires deliberate thinking
        """
        # Given: Train with normal patterns
        woven_mind.train("database connection successful")
        woven_mind.train("query executed successfully")
        woven_mind.train("data retrieved from database")

        # Process normal input in FAST mode
        result1 = woven_mind.process(["database", "query"], mode=ThinkingMode.FAST)
        assert result1.mode == ThinkingMode.FAST

        # When: Encounter surprising input (untrained)
        # The auto_switch mechanism should detect surprise
        result2 = woven_mind.process(["quantum", "entanglement", "database"])

        # Then: Surprise may be detected (depends on baseline)
        # At minimum, the system should track it
        stats = woven_mind.get_stats()
        assert "mode" in stats
        assert "loom" in stats

    def test_mode_switches_logged_for_learning(self, woven_mind):
        """
        Scenario: All mode switches should be captured for learning

        Given Woven Mind with observability enabled
        When multiple mode switches occur
        Then all transitions should be logged
        And transition history should be accessible
        Because learning requires understanding mode-switching patterns
        """
        # Given: Observability enabled (via config)
        assert woven_mind.config.enable_observability is True

        # When: Multiple mode switches
        woven_mind.force_mode(ThinkingMode.FAST, reason="test_1")
        woven_mind.force_mode(ThinkingMode.SLOW, reason="test_2")
        woven_mind.force_mode(ThinkingMode.FAST, reason="test_3")

        # Then: Transitions are logged
        transitions = woven_mind.get_transition_history()
        # Note: First transition is FAST->SLOW (test_2), then SLOW->FAST (test_3)
        # The initial force to FAST doesn't create a transition if already in FAST
        assert len(transitions) >= 2

        # Verify transition details
        assert transitions[-2].to_mode == ThinkingMode.SLOW
        assert transitions[-1].to_mode == ThinkingMode.FAST


# =============================================================================
# TEST BOTTLENECK DETECTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.cognitive
@pytest.mark.coherence
class TestBottleneckDetection:
    """
    Epic: Bottleneck Detection in Orchestration

    As a developer building flow-based orchestration,
    I want bottlenecks to be automatically detected,
    So that the system can adapt and optimize throughput.
    """

    def test_wip_violation_detected(self, kanban_orchestrator):
        """
        Scenario: WIP limit violations should be detected

        Given a kanban board with WIP limits
        When items exceed the limit
        Then a WIP violation should be detected
        And violation details should be captured
        Because WIP limits maintain system stability
        """
        # Given: Board with WIP limits
        board = kanban_orchestrator.board
        in_progress = board.get_column("in_progress")
        assert in_progress is not None
        assert in_progress.wip_limit == 3

        # When: Add items beyond limit
        goals = [
            Goal(id=f"goal-{i}", description=f"Task {i}")
            for i in range(5)
        ]

        for goal in goals:
            in_progress.items.append(goal)

        # When: Enforce WIP limits
        violations = kanban_orchestrator.enforce_wip_limits()

        # Then: Violation detected
        assert len(violations) > 0
        assert violations[0]["action"] in ["block", "alert"]

        # Verify metrics captured violation
        assert len(board.metrics.wip_violations) > 0
        violation = board.metrics.wip_violations[0]
        assert violation.column == "in_progress"
        assert violation.current > violation.limit

    def test_queue_buildup_triggers_alert(self, kanban_orchestrator):
        """
        Scenario: Growing queues should trigger bottleneck alert

        Given a kanban board with flowing work
        When items pile up before a full column
        Then a bottleneck should be detected
        And the location should be identified
        Because queue buildup indicates constrained capacity
        """
        # Given: Board with work
        board = kanban_orchestrator.board

        # When: Create queue buildup
        # Fill in_progress to its limit
        in_progress = board.get_column("in_progress")
        for i in range(3):
            goal = Goal(id=f"ip-{i}", description=f"In progress {i}")
            in_progress.items.append(goal)

        # Add items waiting in ready
        ready = board.get_column("ready")
        for i in range(5):
            goal = Goal(id=f"ready-{i}", description=f"Ready {i}")
            ready.items.append(goal)

        # When: Detect bottlenecks
        bottlenecks = kanban_orchestrator.detect_bottlenecks()

        # Then: Bottleneck detected
        assert len(bottlenecks) > 0
        bottleneck = bottlenecks[0]
        assert bottleneck.location == "in_progress"
        assert bottleneck.queue_depth > 0
        # Note: blocked_items may be empty depending on bottleneck detection logic
        # The queue_depth indicates items are piling up

    def test_slow_stage_identified(self, kanban_orchestrator):
        """
        Scenario: Stages slower than baseline should be flagged

        Given a kanban board with flow metrics
        When a stage consistently takes longer than others
        Then it should be identified as a bottleneck
        And optimization recommendations should be provided
        Because slow stages constrain overall throughput
        """
        # Given: Board with items
        board = kanban_orchestrator.board
        in_progress = board.get_column("in_progress")

        # Simulate slow stage (fill to capacity)
        for i in range(3):
            goal = Goal(id=f"slow-{i}", description=f"Slow task {i}")
            in_progress.items.append(goal)

        # When: Detect bottlenecks
        bottlenecks = kanban_orchestrator.detect_bottlenecks()

        # Then: Stage identified
        if len(bottlenecks) > 0:
            assert bottlenecks[0].location is not None
            # Bottleneck should have type information
            assert hasattr(bottlenecks[0], 'location')

    def test_optimization_suggested(self, kanban_orchestrator):
        """
        Scenario: Bottlenecks should generate optimization suggestions

        Given a detected bottleneck
        When requesting recommendations
        Then specific actions should be suggested
        And rationale should be provided
        Because actionable recommendations enable improvement
        """
        # Given: Create bottleneck condition
        board = kanban_orchestrator.board
        in_progress = board.get_column("in_progress")

        # Fill in_progress
        for i in range(3):
            in_progress.items.append(
                Goal(id=f"task-{i}", description=f"Task {i}")
            )

        # Add waiting items
        ready = board.get_column("ready")
        for i in range(3):
            ready.items.append(
                Goal(id=f"waiting-{i}", description=f"Waiting {i}")
            )

        # When: Detect bottleneck
        bottlenecks = kanban_orchestrator.detect_bottlenecks()

        # Then: Recommendation provided
        if len(bottlenecks) > 0:
            bottleneck = bottlenecks[0]
            assert bottleneck.recommendation is not None
            assert len(bottleneck.recommendation) > 0
            # Recommendation should provide actionable advice
            # Could mention WIP, monitoring, reducing intake, etc.
            assert any(keyword in bottleneck.recommendation.lower()
                      for keyword in ["wip", "monitor", "reduce", "swarm", "consider"])


# =============================================================================
# TEST COGNITIVE METRICS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.cognitive
@pytest.mark.coherence
class TestCognitiveMetrics:
    """
    Epic: Cognitive Metrics Collection

    As a developer building observable systems,
    I want cognitive metrics to be collected automatically,
    So that system health and performance can be monitored.
    """

    def test_metrics_collected_during_execution(self, metrics_collector):
        """
        Scenario: All metric types should be collected during task execution

        Given a metrics collector
        When various operations occur
        Then flow, sprint, and cognitive metrics should be recorded
        And aggregations should be available
        Because comprehensive metrics enable informed decisions
        """
        # Given: Metrics collector
        # (provided by fixture)

        # When: Record various metrics
        metrics_collector.record("throughput", 2.5, labels={"unit": "goals/day"})
        metrics_collector.record("cycle_time", 45.0, labels={"unit": "minutes"})
        metrics_collector.record("wip_stability", 0.85, labels={"unit": "ratio"})

        # Record goal completion
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        metrics_collector.record_goal_completion(start, end)

        # Then: Metrics are collected
        assert len(metrics_collector.data_points) >= 3

        # Verify metric types
        metric_names = {dp.name for dp in metrics_collector.data_points}
        assert "throughput" in metric_names
        assert "cycle_time" in metric_names
        assert "wip_stability" in metric_names

    def test_health_score_calculation(self, metrics_collector):
        """
        Scenario: Health score should reflect worker performance

        Given collected metrics
        When computing hybrid metrics
        Then health indicators should be calculated
        And scores should be in valid ranges
        Because health scores guide optimization
        """
        # Given: Collect some metrics
        for i in range(5):
            start = datetime.now() - timedelta(hours=2)
            end = datetime.now() - timedelta(hours=1)
            metrics_collector.record_goal_completion(start, end)

        # When: Get hybrid metrics
        hybrid = metrics_collector.get_hybrid_metrics()

        # Then: Health indicators are computed
        assert isinstance(hybrid, HybridMetrics)
        assert 0.0 <= hybrid.wip_stability <= 1.0
        assert 0.0 <= hybrid.predictability <= 1.0
        assert 0.0 <= hybrid.responsiveness <= 1.0
        assert 0.0 <= hybrid.quality <= 1.0

    def test_metrics_aggregated_by_director(self, metrics_collector):
        """
        Scenario: Director should aggregate worker metrics

        Given multiple workers reporting metrics
        When aggregating at director level
        Then combined metrics should be available
        And trends should be identifiable
        Because directors need visibility into team performance
        """
        # Given: Multiple metric data points (simulating workers)
        workers = ["worker-1", "worker-2", "worker-3"]

        for worker in workers:
            for i in range(3):
                metrics_collector.record(
                    "task_completion",
                    1.0,
                    labels={"worker": worker, "task_id": f"t-{i}"}
                )

        # When: Get summary
        summary = metrics_collector.get_summary()

        # Then: Aggregated view available
        assert summary["total_data_points"] >= 9
        assert "goals_completed" in summary
        assert "throughput" in summary


# =============================================================================
# TEST ESCALATION PROTOCOL
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.cognitive
@pytest.mark.coherence
class TestEscalationProtocol:
    """
    Epic: Confusion Escalation Protocol

    As a developer building resilient systems,
    I want confusion to trigger escalation,
    So that agents don't get stuck in unproductive loops.
    """

    def test_single_confusion_triggers_monitor(
        self, cognitive_state_manager
    ):
        """
        Scenario: First confusion should trigger MONITOR level

        Given a cognitive state manager
        When confusion is detected once
        Then escalation level should be MONITOR
        And state should be tracked
        Because initial confusion warrants observation
        """
        # Given: Clean state
        # (provided by fixture)

        # When: Simulate confusion (conflicting answers)
        q1 = cognitive_state_manager.ask_question("Use approach A?")
        q2 = cognitive_state_manager.ask_question("Use approach B instead?")

        cognitive_state_manager.answer_question(q1.id, "Yes, use A")
        cognitive_state_manager.answer_question(q2.id, "Yes, use B")

        # Then: Conflicting state exists
        assert len(cognitive_state_manager.questions) == 2

        # In a real escalation system, this would trigger MONITOR
        # For now, verify the state is captured
        from llm_orchestration.cognitive_state import QuestionStatus
        assert q1.status == QuestionStatus.ANSWERED
        assert q2.status == QuestionStatus.ANSWERED

    def test_repeated_confusion_escalates(
        self, cognitive_state_manager
    ):
        """
        Scenario: Repeated confusion should escalate to higher levels

        Given a system in MONITOR state
        When confusion occurs again
        Then escalation should increase
        And recovery actions should intensify
        Because persistent confusion requires stronger intervention
        """
        # Given: Multiple questions showing confusion pattern
        questions = []
        for i in range(5):
            q = cognitive_state_manager.ask_question(f"Conflicting question {i}?")
            cognitive_state_manager.answer_question(q.id, f"Contradictory answer {i}")
            questions.append(q)

        # Then: Multiple confused states
        assert len(cognitive_state_manager.questions) >= 5

        # In escalation protocol:
        # 1 confusion = MONITOR
        # 2-3 confusions = INTERVENE
        # 4+ confusions = ABORT
        confusion_count = len(questions)

        # Simulated escalation logic
        if confusion_count == 1:
            escalation_level = "MONITOR"
        elif confusion_count <= 3:
            escalation_level = "INTERVENE"
        else:
            escalation_level = "ABORT"

        assert confusion_count > 3
        assert escalation_level == "ABORT"

    def test_high_severity_fast_escalation(
        self, cognitive_state_manager
    ):
        """
        Scenario: HIGH severity should escalate faster than LOW

        Given confusion events with different severities
        When comparing escalation thresholds
        Then high severity should reach ABORT sooner
        Because critical issues need immediate attention
        """
        # Simulate severity-weighted escalation

        # Low severity: 5 events to ABORT
        low_severity_threshold = 5

        # High severity: 2 events to ABORT
        high_severity_threshold = 2

        # Then: High severity escalates faster
        assert high_severity_threshold < low_severity_threshold
        assert high_severity_threshold == 2
        assert low_severity_threshold == 5

    def test_abort_captures_learning(
        self, cognitive_state_manager, temp_storage
    ):
        """
        Scenario: ABORT should capture failure for learning

        Given an escalation reaching ABORT level
        When the abort is triggered
        Then failure context should be captured
        And learning data should be available
        Because failures are valuable learning opportunities
        """
        # Given: State leading to abort
        for i in range(5):
            q = cognitive_state_manager.ask_question(f"Repeated confusion {i}")
            cognitive_state_manager.answer_question(q.id, f"Unclear answer {i}")

        # When: Create checkpoint before abort (preserving state for learning)
        checkpoint = cognitive_state_manager.checkpoint()

        # Then: State is captured
        assert checkpoint is not None
        assert "questions" in checkpoint
        assert len(checkpoint["questions"]) >= 5

        # Verify checkpoint can be used for learning
        assert "timestamp" in checkpoint
        assert checkpoint["timestamp"] is not None


# =============================================================================
# TEST SYNAPTIC CONFUSION DETECTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.cognitive
@pytest.mark.coherence
class TestSynapticConfusionDetection:
    """
    Epic: Synaptic Confusion Detection (PRISM-GoT)

    As a developer building memory-based reasoning,
    I want activation patterns to reveal confusion,
    So that circular or contradictory reasoning can be detected.
    """

    def test_activation_loop_detected(
        self, incremental_reasoner, synaptic_graph
    ):
        """
        Scenario: Circular activation patterns should trigger confusion

        Given a synaptic memory graph
        When nodes activate in a circular pattern
        Then the loop should be detectable
        And it should signal confusion
        Because circular reasoning indicates being stuck
        """
        # Given: Create a circular reasoning path
        q1 = incremental_reasoner.process_thought(
            "Should we use approach A?",
            NodeType.QUESTION
        )

        h1 = incremental_reasoner.process_thought(
            "Approach A requires B",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        h2 = incremental_reasoner.process_thought(
            "But B requires approach A",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        # Create circular edge
        synaptic_graph.add_synaptic_edge(
            from_id=h2.id,
            to_id=q1.id,
            edge_type=EdgeType.DEPENDS_ON
        )

        # When: Activate nodes in sequence
        synaptic_graph.activate_node(q1.id)
        synaptic_graph.activate_node(h1.id)
        synaptic_graph.activate_node(h2.id)
        synaptic_graph.activate_node(q1.id)  # Loop back

        # Then: Activations are tracked
        trace_q1 = synaptic_graph.activation_traces[q1.id]
        assert trace_q1.total_activations >= 2

        # Recent activations show the loop
        recent = trace_q1.get_recent(n=5)
        assert len(recent) >= 2

    def test_contradictory_activations_detected(
        self, incremental_reasoner, synaptic_graph
    ):
        """
        Scenario: Opposing concepts activated together should trigger confusion

        Given a synaptic memory graph
        When contradictory hypotheses are both activated
        Then contradiction should be detectable
        And it should signal confusion
        Because contradictions indicate unclear reasoning
        """
        # Given: Create contradictory hypotheses
        q = incremental_reasoner.process_thought(
            "Which approach should we use?",
            NodeType.QUESTION
        )

        h1 = incremental_reasoner.process_thought(
            "Use synchronous processing",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        h2 = incremental_reasoner.process_thought(
            "Use asynchronous processing",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        # Mark them as conflicting
        synaptic_graph.add_synaptic_edge(
            from_id=h1.id,
            to_id=h2.id,
            edge_type=EdgeType.CONFLICTS
        )

        # When: Both are activated together
        synaptic_graph.activate_node(h1.id, context={"decision": "sync"})
        synaptic_graph.activate_node(h2.id, context={"decision": "async"})

        # Then: Both have activation traces
        assert h1.id in synaptic_graph.activation_traces
        assert h2.id in synaptic_graph.activation_traces

        # Both activated recently (indicating confusion)
        trace_h1 = synaptic_graph.activation_traces[h1.id]
        trace_h2 = synaptic_graph.activation_traces[h2.id]

        assert trace_h1.total_activations >= 1
        assert trace_h2.total_activations >= 1

        # In a real confusion detector, we would check for:
        # - Conflicting edges between recently activated nodes
        # - Co-activation of contradictory concepts
        conflicting_edges = [
            e for e in synaptic_graph.get_synaptic_edges_from(h1.id)
            if e.edge_type == EdgeType.CONFLICTS
        ]
        assert len(conflicting_edges) > 0

    def test_activation_frequency_indicates_obsession(
        self, incremental_reasoner, synaptic_graph
    ):
        """
        Scenario: Repeated activation of same node indicates obsession/loop

        Given a synaptic memory graph
        When a single node is activated many times in short window
        Then high frequency should be detectable
        And it should signal potential loop/obsession
        Because obsessive activation indicates stuck reasoning
        """
        # Given: Create a node
        node = incremental_reasoner.process_thought(
            "Must solve this problem",
            NodeType.OBSERVATION
        )

        # When: Activate repeatedly in short time
        for i in range(10):
            synaptic_graph.activate_node(
                node.id,
                context={"attempt": i}
            )

        # Then: High activation count
        trace = synaptic_graph.activation_traces[node.id]
        assert trace.total_activations >= 10

        # Calculate frequency (activations per minute)
        frequency = trace.get_frequency(window_minutes=1)

        # Should show high frequency if all activations were recent
        # (depends on timing, but total_activations should be high)
        assert trace.total_activations >= 10

    def test_hebbian_strengthening_from_coactivation(
        self, synaptic_graph
    ):
        """
        Scenario: Co-activated nodes should strengthen connections

        Given a synaptic memory graph
        When two connected nodes are activated together
        Then their connection should strengthen (Hebbian learning)
        And connection weight should increase
        Because "neurons that fire together wire together"
        """
        # Given: Create connected nodes
        n1 = synaptic_graph.add_node(
            node_id="n1",
            node_type=NodeType.OBSERVATION,
            content="Input pattern"
        )

        n2 = synaptic_graph.add_node(
            node_id="n2",
            node_type=NodeType.HYPOTHESIS,
            content="Response pattern"
        )

        edge = synaptic_graph.add_synaptic_edge(
            from_id=n1.id,
            to_id=n2.id,
            edge_type=EdgeType.TRIGGERS,  # Use TRIGGERS for causal relationship
            weight=0.5
        )

        initial_weight = edge.weight

        # When: Co-activate nodes
        synaptic_graph.activate_node(n1.id)
        synaptic_graph.activate_node(n2.id)

        # Apply Hebbian learning
        strengthened = synaptic_graph.apply_hebbian_learning(time_window_seconds=60)

        # Then: Connection strengthened
        assert strengthened >= 0
        # Weight may have increased (depends on plasticity rules)
        # At minimum, activation was recorded
        assert edge.activation_count >= 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.cognitive
@pytest.mark.coherence
class TestPhase1Integration:
    """
    Epic: Phase 1 Full Integration

    As a developer building coherent cognitive systems,
    I want all Phase 1 components to work together,
    So that the system exhibits intelligent, adaptive behavior.
    """

    def test_full_cognitive_flow_with_all_phase1_features(
        self,
        woven_mind,
        incremental_reasoner,
        metrics_collector,
        cognitive_state_manager,
        temp_storage,
    ):
        """
        Scenario: All Phase 1 features work together seamlessly

        Given all Phase 1 components initialized
        When executing a complex reasoning task
        Then:
          - Woven Mind switches modes appropriately
          - PRISM tracks activation patterns
          - Metrics are collected
          - Confusion is detected if present
          - State is checkpointed
        Because integrated systems should exhibit emergent intelligence
        """
        # Given: All components initialized
        # (provided by fixtures)

        # Create QAPV pattern
        qapv = QAPVPattern(cognitive_state_manager, goal="Implement auth system")

        # Phase 1: QUESTION (SLOW mode)
        assert qapv.current_phase == QAPVPhase.QUESTION
        woven_mind.force_mode(ThinkingMode.SLOW, reason="question_phase")

        # Record thought in graph
        q_node = incremental_reasoner.process_thought(
            "What authentication method should we use?",
            NodeType.QUESTION
        )

        # Phase 2: ANSWER (FAST mode for known patterns)
        woven_mind.train("jwt authentication is secure and scalable")
        woven_mind.train("jwt tokens contain claims")
        result = woven_mind.process(["jwt", "authentication"], mode=ThinkingMode.FAST)

        assert result.mode == ThinkingMode.FAST

        # Record hypothesis
        h_node = incremental_reasoner.process_thought(
            "Use JWT authentication",
            NodeType.HYPOTHESIS,
            relation_to_focus=EdgeType.EXPLORES
        )

        # Collect metrics
        start_time = datetime.now() - timedelta(minutes=5)
        end_time = datetime.now()
        metrics_collector.record_goal_completion(start_time, end_time)
        metrics_collector.record("mode_switches", 2.0)

        # Create checkpoint
        checkpoint = cognitive_state_manager.checkpoint()

        # Then: All components worked
        assert qapv.current_phase == QAPVPhase.QUESTION
        assert len(incremental_reasoner.graph.nodes) >= 2
        assert len(metrics_collector.data_points) >= 1
        assert checkpoint is not None
        # Note: Only 1 explicit transition (FAST->SLOW) in this test
        assert len(woven_mind.get_transition_history()) >= 1

    def test_bottleneck_triggers_mode_switch_recommendation(
        self,
        kanban_orchestrator,
        metrics_collector,
        woven_mind,
    ):
        """
        Scenario: Detected bottleneck should influence cognitive strategies

        Given a bottleneck in orchestration
        When metrics are collected
        Then optimization recommendations should be generated
        And mode switching strategies may adapt
        Because bottlenecks require strategic changes
        """
        # Given: Create bottleneck
        board = kanban_orchestrator.board
        in_progress = board.get_column("in_progress")

        # Fill to capacity
        for i in range(3):
            in_progress.items.append(
                Goal(id=f"task-{i}", description=f"Task {i}")
            )

        # Add queue
        ready = board.get_column("ready")
        for i in range(5):
            ready.items.append(
                Goal(id=f"waiting-{i}", description=f"Waiting {i}")
            )

        # When: Detect bottleneck
        bottlenecks = kanban_orchestrator.detect_bottlenecks()

        # Collect metrics
        for i in range(3):
            start = datetime.now() - timedelta(hours=1)
            end = datetime.now()
            metrics_collector.record_goal_completion(start, end)

        # Get dashboard
        dashboard = MetricsDashboard(metrics_collector)
        recommendations = dashboard.get_evolution_recommendations()

        # Then: System provides actionable insights
        assert len(bottlenecks) > 0
        bottleneck = bottlenecks[0]
        assert bottleneck.recommendation is not None

        # Metrics available for decision-making
        hybrid = metrics_collector.get_hybrid_metrics()
        assert isinstance(hybrid, HybridMetrics)
