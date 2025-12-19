"""
Integration tests for the reasoning framework.

Tests cover interactions between:
- Workflow orchestrator and cognitive loops
- Production tasks and verification
- Crisis management and recovery
- Collaboration modes and handoffs
"""

import pytest
from cortical.reasoning import (
    # Core types
    NodeType,
    EdgeType,
    ThoughtGraph,
    # Cognitive loop
    LoopPhase,
    LoopStatus,
    CognitiveLoop,
    CognitiveLoopManager,
    # Production state
    ProductionState,
    ProductionTask,
    ProductionManager,
    ProductionChunk,
    # Crisis management
    CrisisLevel,
    RecoveryAction,
    CrisisManager,
    CrisisEvent,
    # Verification
    VerificationLevel,
    VerificationPhase,
    VerificationStatus,
    VerificationManager,
    VerificationCheck,
    create_drafting_checklist,
    # Collaboration
    CollaborationMode,
    CollaborationManager,
    StatusUpdate,
    # Workflow
    WorkflowContext,
    ReasoningWorkflow,
    # Patterns
    create_investigation_graph,
    create_feature_graph,
)


class TestWorkflowWithCognitiveLoop:
    """Test workflow orchestrator with cognitive loop integration."""

    def test_workflow_creates_context(self):
        """Test workflow creates proper context."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Implement feature X")

        assert ctx.session_id is not None
        assert ctx.goal == "Implement feature X"
        assert ctx.thought_graph is not None
        assert ctx.current_loop_id is not None

    def test_workflow_question_phase(self):
        """Test workflow question phase."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Add authentication")

        workflow.begin_question_phase(ctx)

        # Record questions
        q1 = workflow.record_question(ctx, "What authentication method?")
        q2 = workflow.record_question(ctx, "What are the security requirements?")

        assert q1 is not None
        assert q2 is not None
        # Questions should be in thought graph
        assert ctx.thought_graph.get_node(q1) is not None
        assert ctx.thought_graph.get_node(q2) is not None

    def test_workflow_answer_phase(self):
        """Test workflow answer phase with findings."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Investigate bug")

        workflow.begin_question_phase(ctx)
        q_id = workflow.record_question(ctx, "What causes the crash?")
        workflow.begin_answer_phase(ctx)

        # Record answer
        a_id = workflow.record_answer(ctx, q_id, "Stack overflow in recursion")
        assert a_id is not None

        # Check answer is linked to question
        edges = ctx.thought_graph.get_edges_from(a_id)
        assert any(e.target_id == q_id for e in edges)

    def test_workflow_produce_phase(self):
        """Test workflow produce phase creates artifacts."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Write utility function")

        # Move through phases
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What should it do?")
        workflow.begin_answer_phase(ctx)
        workflow.begin_production_phase(ctx)

        # Record artifact
        workflow.record_artifact(ctx, "utils.py", "file")
        assert "utils.py" in ctx.artifacts_produced

    def test_workflow_verify_phase(self):
        """Test workflow verify phase checks artifacts."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Fix validation bug")

        # Move through all phases
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What validation fails?")
        workflow.begin_answer_phase(ctx)
        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "validator.py")
        workflow.begin_verify_phase(ctx)

        assert ctx.current_verification_suite is not None

    def test_workflow_full_qapv_cycle(self):
        """Test complete QAPV cycle."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Add logging")

        # Q: Question
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "Where to add logging?")
        workflow.record_question(ctx, "What log level?")

        # A: Answer
        workflow.begin_answer_phase(ctx)

        # P: Produce
        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "logger.py")

        # V: Verify
        workflow.begin_verify_phase(ctx)
        results = workflow.verify(ctx)

        # Complete
        summary = workflow.complete_session(ctx)
        assert summary['session_id'] == ctx.session_id

    def test_workflow_decisions_tracked(self):
        """Test that decisions are tracked in workflow."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Choose database")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What database to use?")
        workflow.begin_answer_phase(ctx)

        # Record decision
        decision_id = workflow.record_decision(
            ctx,
            "Use PostgreSQL",
            "Better support for complex queries",
            ["PostgreSQL", "MySQL", "MongoDB"]
        )

        assert len(ctx.decisions_made) == 1
        assert ctx.decisions_made[0]['decision'] == "Use PostgreSQL"


class TestProductionWithVerification:
    """Test production state management with verification integration."""

    def test_production_task_lifecycle(self):
        """Test production task through state transitions."""
        manager = ProductionManager()
        task = manager.create_task(goal="Implement OAuth", description="OAuth flow")

        assert task.state == ProductionState.PLANNING

        # Test state transitions via task methods (with reason argument)
        task.transition_to(ProductionState.DRAFTING, "Starting implementation")
        assert task.state == ProductionState.DRAFTING

        task.transition_to(ProductionState.REFINING, "First draft complete")
        assert task.state == ProductionState.REFINING

        task.transition_to(ProductionState.FINALIZING, "Refinements done")
        assert task.state == ProductionState.FINALIZING

        task.transition_to(ProductionState.COMPLETE, "All checks passed")
        assert task.state == ProductionState.COMPLETE

    def test_production_chunk_creation(self):
        """Test production with chunk-based progress."""
        manager = ProductionManager()
        task = manager.create_task(goal="Large refactor", description="Refactor module")

        # Add chunks
        chunk1 = ProductionChunk(name="imports", goal="Update imports")
        chunk2 = ProductionChunk(name="classes", goal="Refactor classes")
        chunk3 = ProductionChunk(name="tests", goal="Update tests")

        task.add_chunk(chunk1)
        task.add_chunk(chunk2)
        task.add_chunk(chunk3)

        assert len(task.chunks) == 3

    def test_verification_manager_basics(self):
        """Test verification manager basic operations."""
        manager = VerificationManager()

        # Create a standard suite
        suite = manager.create_standard_suite("test_suite", "Test verification")

        assert suite is not None
        assert suite.name == "test_suite"


class TestCrisisManagement:
    """Test crisis management and recovery integration."""

    def test_crisis_manager_creation(self):
        """Test crisis manager creates events."""
        manager = CrisisManager()

        event = manager.record_crisis(
            CrisisLevel.HICCUP,
            "Minor issue encountered",
            context={"task": "test"}
        )

        assert event is not None
        assert event.level == CrisisLevel.HICCUP

    def test_crisis_levels(self):
        """Test different crisis levels."""
        assert CrisisLevel.HICCUP.value < CrisisLevel.OBSTACLE.value
        assert CrisisLevel.OBSTACLE.value < CrisisLevel.WALL.value
        assert CrisisLevel.WALL.value < CrisisLevel.CRISIS.value

    def test_recovery_actions_defined(self):
        """Test recovery actions are defined."""
        assert RecoveryAction.CONTINUE is not None
        assert RecoveryAction.ADAPT is not None
        assert RecoveryAction.ROLLBACK is not None
        assert RecoveryAction.PARTIAL_RECOVER is not None
        assert RecoveryAction.ESCALATE is not None
        assert RecoveryAction.STOP is not None


class TestCollaborationIntegration:
    """Test collaboration modes and handoff integration."""

    def test_collaboration_modes(self):
        """Test collaboration mode initialization."""
        sync_manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)
        async_manager = CollaborationManager(mode=CollaborationMode.ASYNCHRONOUS)
        semi_manager = CollaborationManager(mode=CollaborationMode.SEMI_SYNCHRONOUS)

        assert sync_manager.mode == CollaborationMode.SYNCHRONOUS
        assert async_manager.mode == CollaborationMode.ASYNCHRONOUS
        assert semi_manager.mode == CollaborationMode.SEMI_SYNCHRONOUS

    def test_status_update_creation(self):
        """Test creating status updates."""
        update = StatusUpdate(
            task_name="Implement feature",
            progress_percent=50,
            current_phase="production",
        )

        assert update.task_name == "Implement feature"
        assert update.progress_percent == 50

    def test_collaboration_manager_handoff(self):
        """Test creating handoff documents."""
        manager = CollaborationManager(mode=CollaborationMode.SYNCHRONOUS)

        handoff = manager.create_handoff(
            task="Complex feature",
            status="In progress",
            urgency="medium",
        )

        assert handoff is not None
        assert handoff.task_description == "Complex feature"


class TestWorkflowWithCrisis:
    """Test workflow handling crisis situations."""

    def test_workflow_crisis_reporting(self):
        """Test workflow can report crisis."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Implement feature")

        event = workflow.report_crisis(
            ctx,
            CrisisLevel.OBSTACLE,
            "Tests failing repeatedly"
        )

        assert event is not None
        assert event.level == CrisisLevel.OBSTACLE


class TestGraphIntegration:
    """Test thought graph integration with workflow."""

    def test_workflow_builds_graph(self):
        """Test workflow populates thought graph."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Design API")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What endpoints needed?")
        workflow.record_question(ctx, "What authentication?")

        workflow.begin_answer_phase(ctx)

        # Graph should have nodes for goal, phases, and questions
        assert ctx.thought_graph.node_count() >= 3

    def test_workflow_with_pattern_graph(self):
        """Test merging pattern graphs into workflow."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Investigate performance issue")

        # Create investigation pattern
        investigation = create_investigation_graph("Why is the API slow?")

        # Merge into workflow graph
        for node_id, node in investigation.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id,
                    node.node_type,
                    node.content,
                    properties=node.properties,
                )

        # Should have nodes from both
        assert ctx.thought_graph.node_count() >= 4

    def test_graph_connections_preserved(self):
        """Test graph edges are preserved."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Feature implementation")

        workflow.begin_question_phase(ctx)
        q_id = workflow.record_question(ctx, "What to build?")

        workflow.begin_answer_phase(ctx)
        a_id = workflow.record_answer(ctx, q_id, "User dashboard")

        # Check edge exists
        edges = ctx.thought_graph.get_edges_from(a_id)
        assert len(edges) > 0


class TestVerificationLevels:
    """Test verification level integration."""

    def test_verification_levels_hierarchy(self):
        """Test verification levels are properly ordered."""
        levels = [
            VerificationLevel.UNIT,
            VerificationLevel.INTEGRATION,
            VerificationLevel.E2E,
            VerificationLevel.ACCEPTANCE,
        ]
        # Should be in ascending order of scope
        assert levels[0].value < levels[1].value
        assert levels[1].value < levels[2].value
        assert levels[2].value < levels[3].value

    def test_verification_phases(self):
        """Test verification phases exist."""
        assert VerificationPhase.DRAFTING is not None
        assert VerificationPhase.REFINING is not None
        assert VerificationPhase.FINALIZING is not None

    def test_verification_status_values(self):
        """Test verification status values."""
        assert VerificationStatus.PENDING is not None
        assert VerificationStatus.PASSED is not None
        assert VerificationStatus.FAILED is not None


class TestCognitiveLoopManager:
    """Test cognitive loop manager operations."""

    def test_loop_manager_creates_loops(self):
        """Test creating cognitive loops."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Implement feature")

        assert loop is not None
        assert loop.goal == "Implement feature"
        assert loop.status == LoopStatus.NOT_STARTED

    def test_loop_lifecycle(self):
        """Test loop state transitions."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Test loop")

        loop.start(LoopPhase.QUESTION)
        assert loop.status == LoopStatus.ACTIVE
        assert loop.current_phase == LoopPhase.QUESTION

        loop.transition(LoopPhase.ANSWER, "Moving to answers")
        assert loop.current_phase == LoopPhase.ANSWER

    def test_loop_iteration_tracking(self):
        """Test tracking loop iterations."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Iterative loop")

        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "First pass")
        loop.transition(LoopPhase.QUESTION, "Need more info")

        # Should track iterations
        q_count = loop.get_iteration_count(LoopPhase.QUESTION)
        assert q_count >= 1


class TestEndToEndWorkflow:
    """End-to-end workflow integration tests."""

    def test_complete_feature_implementation(self):
        """Test complete feature implementation workflow."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Add user registration feature")

        # Phase 1: Planning with questions
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What fields are required?")
        workflow.record_question(ctx, "How to validate email?")

        # Phase 2: Research and answers
        workflow.begin_answer_phase(ctx)
        workflow.record_insight(ctx, "Use standard email regex")
        workflow.record_decision(
            ctx,
            "Email and password required",
            "Minimum viable auth",
            ["Email only", "Email+password", "Social auth"]
        )

        # Phase 3: Implementation
        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "user_model.py")
        workflow.record_artifact(ctx, "validators.py")
        workflow.record_artifact(ctx, "register_endpoint.py")

        # Phase 4: Verification
        workflow.begin_verify_phase(ctx)
        results = workflow.verify(ctx)

        # Complete
        summary = workflow.complete_session(ctx)

        assert len(ctx.artifacts_produced) == 3
        assert len(ctx.decisions_made) >= 1
        assert summary['artifacts_produced'] == 3

    def test_workflow_with_status_updates(self):
        """Test workflow with status posting."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Implement logging")

        workflow.begin_question_phase(ctx)
        update = workflow.post_status(ctx, 25, "Gathering requirements")

        assert update.progress_percent == 25
        assert "requirements" in update.in_progress_items[0].lower()

    def test_workflow_session_abandonment(self):
        """Test abandoning a workflow session."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Complex feature")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "Is this feasible?")

        record = workflow.abandon_session(ctx, "Requirements too vague")

        assert record['reason'] == "Requirements too vague"
        assert "abandoned" in record['lessons_learned'][0].lower()

    def test_workflow_handoff_creation(self):
        """Test creating handoff documents."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Multi-day feature")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What's the scope?")
        workflow.begin_answer_phase(ctx)
        workflow.record_decision(ctx, "Scope is X", "Agreed with stakeholder")

        handoff = workflow.create_handoff(ctx)

        assert handoff is not None
        assert ctx.goal in handoff.task_description or "multi-day" in handoff.task_description.lower()

    def test_workflow_insights_tracking(self):
        """Test tracking insights and lessons learned."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Learn something")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)

        insight_id = workflow.record_insight(
            ctx,
            "Caching significantly improves performance",
            source="profiling"
        )

        assert insight_id is not None
        assert "Caching" in ctx.lessons_learned[0]
