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


# =============================================================================
# NEW COMPREHENSIVE INTEGRATION TESTS
# =============================================================================


class TestFullQAPVCycleIntegration:
    """
    Test full QAPV cycle with all components integrated.

    Covers: CognitiveLoop + ProductionManager + VerificationManager
    """

    def test_qapv_with_cognitive_loop_manager(self):
        """Test QAPV cycle using CognitiveLoopManager directly."""
        from cortical.reasoning import CognitiveLoopManager, LoopPhase, LoopStatus

        loop_mgr = CognitiveLoopManager()
        loop = loop_mgr.create_loop("Implement authentication")

        # Start loop in QUESTION phase
        loop.start(LoopPhase.QUESTION)
        assert loop.status == LoopStatus.ACTIVE
        assert loop.current_phase == LoopPhase.QUESTION

        # Transition through phases
        loop.transition(LoopPhase.ANSWER, "Questions clarified")
        assert loop.current_phase == LoopPhase.ANSWER

        loop.transition(LoopPhase.PRODUCE, "Solution designed")
        assert loop.current_phase == LoopPhase.PRODUCE

        loop.transition(LoopPhase.VERIFY, "Implementation complete")
        assert loop.current_phase == LoopPhase.VERIFY

        # Complete loop
        from cortical.reasoning import TerminationReason
        loop.complete(TerminationReason.SUCCESS)
        assert loop.status == LoopStatus.COMPLETED

    def test_production_manager_with_verification(self):
        """Test ProductionManager integrated with VerificationManager."""
        from cortical.reasoning import (
            ProductionManager, ProductionState,
            VerificationManager, VerificationLevel, VerificationPhase,
            create_drafting_checklist, create_refining_checklist
        )

        prod_mgr = ProductionManager()
        ver_mgr = VerificationManager()

        # Create production task
        task = prod_mgr.create_task(
            goal="Implement login endpoint",
            description="REST API for user login"
        )

        # Transition through states with verification at each stage
        task.transition_to(ProductionState.DRAFTING, "Starting implementation")

        # DRAFTING verification
        suite = ver_mgr.create_standard_suite(
            "drafting_checks",
            "Quick sanity checks"
        )
        drafting_checks = create_drafting_checklist()
        for check in drafting_checks:
            suite.add_check(check)

        # Mark checks as passed
        for check in suite.checks:
            check.mark_passed("Looks good")

        task.transition_to(ProductionState.REFINING, "Initial draft complete")

        # REFINING verification
        refining_suite = ver_mgr.create_standard_suite(
            "refining_checks",
            "Thorough validation"
        )
        refining_checks = create_refining_checklist()
        for check in refining_checks:
            refining_suite.add_check(check)

        task.transition_to(ProductionState.FINALIZING, "Refinements complete")
        task.transition_to(ProductionState.COMPLETE, "All checks passed")

        assert task.state == ProductionState.COMPLETE
        # COMPLETE is a terminal state
        assert task.state in [ProductionState.COMPLETE, ProductionState.ABANDONED]

    def test_workflow_coordinates_all_qapv_components(self):
        """Test that ReasoningWorkflow properly coordinates all QAPV components."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Complex QAPV integration test")

        # Verify context has all components initialized
        assert ctx.thought_graph is not None
        assert ctx.current_loop_id is not None

        # Go through full cycle
        workflow.begin_question_phase(ctx)
        q1 = workflow.record_question(ctx, "What is the requirement?")
        q2 = workflow.record_question(ctx, "What are the constraints?")

        workflow.begin_answer_phase(ctx)
        workflow.record_answer(ctx, q1, "Build a REST API")
        workflow.record_answer(ctx, q2, "Must be stateless")
        workflow.record_decision(ctx, "Use JWT tokens", "Standard for stateless auth")

        workflow.begin_production_phase(ctx)

        # Create production chunks
        chunk1 = workflow.create_production_chunk(
            ctx,
            name="auth_endpoint",
            goal="Create authentication endpoint",
            files=["auth.py"]
        )
        chunk2 = workflow.create_production_chunk(
            ctx,
            name="tests",
            goal="Write integration tests",
            files=["test_auth.py"]
        )

        workflow.record_artifact(ctx, "auth.py")
        workflow.record_artifact(ctx, "test_auth.py")

        workflow.begin_verify_phase(ctx)
        results = workflow.verify(ctx)

        summary = workflow.complete_session(ctx)

        # Verify complete integration
        assert len(ctx.questions_answered) >= 2
        assert len(ctx.decisions_made) >= 1
        assert len(ctx.artifacts_produced) >= 2
        assert summary['session_id'] == ctx.session_id


class TestCrisisAndRecoveryIntegration:
    """
    Test crisis management integrated with recovery procedures.

    Covers: CrisisManager + RecoveryProcedures activation
    """

    def test_crisis_triggers_recovery_suggestions(self):
        """Test that crisis events trigger appropriate recovery suggestions."""
        from cortical.reasoning import (
            CrisisManager, CrisisLevel, RecoveryAction, RecoveryProcedures
        )

        crisis_mgr = CrisisManager()
        recovery = RecoveryProcedures()

        # Test HICCUP level
        hiccup = crisis_mgr.record_crisis(
            CrisisLevel.HICCUP,
            "Minor test failure",
            context={'test': 'test_login'}
        )
        suggestions = recovery.suggest_recovery(hiccup)
        assert RecoveryAction.CONTINUE in suggestions

        # Test OBSTACLE level
        obstacle = crisis_mgr.record_crisis(
            CrisisLevel.OBSTACLE,
            "Tests failing repeatedly",
            context={'attempts': 3}
        )
        suggestions = recovery.suggest_recovery(obstacle)
        assert RecoveryAction.ADAPT in suggestions or RecoveryAction.ROLLBACK in suggestions

        # Test WALL level
        wall = crisis_mgr.record_crisis(
            CrisisLevel.WALL,
            "Fundamental assumption proven false",
            context={'assumption': 'API supports feature X'}
        )
        suggestions = recovery.suggest_recovery(wall)
        assert RecoveryAction.ESCALATE in suggestions

    def test_workflow_crisis_activates_recovery(self):
        """Test workflow crisis reporting activates recovery procedures."""
        from cortical.reasoning import CrisisLevel

        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Task with crisis")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)
        workflow.begin_production_phase(ctx)

        # Report a crisis
        crisis = workflow.report_crisis(
            ctx,
            CrisisLevel.OBSTACLE,
            "Implementation blocked by dependency"
        )

        assert crisis is not None
        assert crisis.level == CrisisLevel.OBSTACLE

        # Workflow should track crisis
        workflow_summary = workflow.get_workflow_summary()
        assert workflow_summary['crises']['unresolved'] >= 1

    def test_repeated_failure_escalation(self):
        """Test that repeated failures escalate crisis level."""
        from cortical.reasoning import (
            CrisisManager, CrisisLevel, RepeatedFailureTracker
        )

        crisis_mgr = CrisisManager()

        # Create failure tracker
        tracker = RepeatedFailureTracker(issue_description="Tests failing")

        # Record multiple attempts
        tracker.record_attempt(
            "Wrong import path",
            "Fixed import",
            "Still failing"
        )
        assert len(tracker.attempts) == 1

        tracker.record_attempt(
            "Missing dependency",
            "Installed package",
            "Still failing"
        )
        assert len(tracker.attempts) == 2

        tracker.record_attempt(
            "API changed",
            "Updated calls",
            "Still failing"
        )
        assert len(tracker.attempts) == 3

        # After 3 attempts, should recommend escalation
        assert tracker.should_escalate()


class TestParallelCoordinationIntegration:
    """
    Test parallel coordination with boundary enforcement.

    Covers: ClaudeCodeSpawner + ParallelCoordinator
    """

    def test_parallel_coordinator_boundary_validation(self):
        """Test ParallelCoordinator validates boundaries before spawning."""
        from cortical.reasoning import (
            ParallelCoordinator, ParallelWorkBoundary, SequentialSpawner
        )

        spawner = SequentialSpawner()
        coordinator = ParallelCoordinator(spawner)

        # Create non-conflicting boundaries
        boundary1 = ParallelWorkBoundary(
            agent_id="agent-1",
            scope_description="Implement auth module"
        )
        boundary1.add_file("auth.py", write_access=True)
        boundary1.add_file("utils.py", write_access=False)

        boundary2 = ParallelWorkBoundary(
            agent_id="agent-2",
            scope_description="Implement user module"
        )
        boundary2.add_file("user.py", write_access=True)
        boundary2.add_file("utils.py", write_access=False)

        # Should be able to spawn (different write files)
        can_spawn, issues = coordinator.can_spawn([boundary1, boundary2])
        assert can_spawn
        assert len(issues) == 0

    def test_parallel_coordinator_detects_conflicts(self):
        """Test ParallelCoordinator detects file conflicts."""
        from cortical.reasoning import (
            ParallelCoordinator, ParallelWorkBoundary, SequentialSpawner
        )

        spawner = SequentialSpawner()
        coordinator = ParallelCoordinator(spawner)

        # Create conflicting boundaries
        boundary1 = ParallelWorkBoundary(
            agent_id="agent-1",
            scope_description="Refactor module A"
        )
        boundary1.add_file("shared.py", write_access=True)

        boundary2 = ParallelWorkBoundary(
            agent_id="agent-2",
            scope_description="Refactor module B"
        )
        boundary2.add_file("shared.py", write_access=True)  # Conflict!

        # Should detect conflict
        can_spawn, issues = coordinator.can_spawn([boundary1, boundary2])
        assert not can_spawn
        assert len(issues) >= 1
        assert "conflict" in issues[0].lower()

    def test_claude_code_spawner_task_generation(self):
        """Test ClaudeCodeSpawner generates proper task configs."""
        from cortical.reasoning import (
            ClaudeCodeSpawner, ParallelWorkBoundary
        )

        spawner = ClaudeCodeSpawner()

        # Create boundary
        boundary = ParallelWorkBoundary(
            agent_id="test-agent",
            scope_description="Implement feature X"
        )
        boundary.add_file("feature.py", write_access=True)
        boundary.add_file("base.py", write_access=False)

        # Prepare agent
        configs = spawner.prepare_agents([
            ("Implement feature X", boundary)
        ])

        assert len(configs) == 1
        config = configs[0]

        # Verify config structure
        assert config.agent_id is not None
        assert "feature x" in config.description.lower()
        assert "feature.py" in config.prompt
        assert "base.py" in config.prompt


class TestVerificationFailureAnalysisIntegration:
    """
    Test verification integrated with failure analysis.

    Covers: VerificationSuite + FailureAnalyzer + RegressionDetector
    """

    def test_verification_suite_with_failure_analyzer(self):
        """Test VerificationSuite failures are analyzed by FailureAnalyzer."""
        from cortical.reasoning import (
            VerificationManager, VerificationCheck, VerificationLevel,
            VerificationPhase, VerificationFailure, FailureAnalyzer
        )

        ver_mgr = VerificationManager()
        analyzer = FailureAnalyzer()

        # Create suite with checks
        suite = ver_mgr.create_standard_suite("test_suite", "Test verification")

        check = VerificationCheck(
            name="import_test",
            description="Test imports work",
            level=VerificationLevel.UNIT,
            phase=VerificationPhase.DRAFTING
        )
        suite.add_check(check)

        # Simulate failure
        check.mark_failed("ImportError: No module named 'nonexistent'")

        # Create failure record
        failure = VerificationFailure(
            check=check,
            observed="ImportError: No module named 'nonexistent'",
            expected_vs_actual="Expected: import succeeds, Actual: ImportError"
        )

        # Analyze failure
        analysis = analyzer.analyze_failure(failure)

        assert 'likely_cause' in analysis
        assert 'matched_patterns' in analysis
        assert len(analysis['matched_patterns']) >= 1
        assert 'import_error' in analysis['matched_patterns']

    def test_regression_detector_tracks_history(self):
        """Test RegressionDetector tracks verification history."""
        from cortical.reasoning import (
            RegressionDetector, VerificationStatus
        )

        detector = RegressionDetector()

        # Save baseline
        baseline = {
            'test_auth': VerificationStatus.PASSED,
            'test_user': VerificationStatus.PASSED,
            'test_api': VerificationStatus.PASSED,
        }
        detector.save_baseline('main', baseline)

        # Current results with regression
        current = {
            'test_auth': VerificationStatus.PASSED,
            'test_user': VerificationStatus.FAILED,  # Regression!
            'test_api': VerificationStatus.PASSED,
        }

        # Detect regressions
        regressions = detector.detect_regression(current, baseline_name='main')

        assert len(regressions) == 1
        assert regressions[0]['test_name'] == 'test_user'
        assert regressions[0]['baseline_status'] == VerificationStatus.PASSED
        assert regressions[0]['current_status'] == VerificationStatus.FAILED

    def test_workflow_verification_uses_failure_analysis(self):
        """Test workflow verification integrates failure analysis."""
        from cortical.reasoning import VerificationLevel

        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Test with verification")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)
        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "implementation.py")

        # Begin verify phase
        workflow.begin_verify_phase(ctx)

        # Verify phase should create verification suite
        assert ctx.current_verification_suite is not None


class TestThoughtGraphCognitiveLoopIntegration:
    """
    Test thought graph integration with cognitive loop.

    Covers: ThoughtGraph captures reasoning flow during QAPV phases
    """

    def test_graph_captures_qapv_flow(self):
        """Test ThoughtGraph captures complete QAPV reasoning flow."""
        from cortical.reasoning import NodeType, EdgeType

        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Graph integration test")

        # Question phase
        workflow.begin_question_phase(ctx)
        q1 = workflow.record_question(ctx, "What is the goal?")
        q2 = workflow.record_question(ctx, "What are the constraints?")

        # Answer phase
        workflow.begin_answer_phase(ctx)
        a1 = workflow.record_answer(ctx, q1, "Build a search feature")
        a2 = workflow.record_answer(ctx, q2, "Must be fast (<100ms)")

        # Verify graph has nodes
        question_nodes = ctx.thought_graph.nodes_of_type(NodeType.QUESTION)
        answer_nodes = ctx.thought_graph.nodes_of_type(NodeType.FACT)  # Answers are stored as FACT nodes

        assert len(question_nodes) >= 2
        assert len(answer_nodes) >= 2

        # Verify edges exist
        edges_from_a1 = ctx.thought_graph.get_edges_from(a1)
        assert len(edges_from_a1) > 0

    def test_graph_visualization_exports(self):
        """Test ThoughtGraph can export for visualization."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Visualization test")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "How to implement?")
        workflow.begin_answer_phase(ctx)
        workflow.record_decision(ctx, "Use algorithm X", "Most efficient")

        # Export graph to DOT format
        dot = ctx.thought_graph.to_dot()
        assert dot is not None
        assert len(dot) > 0
        assert "digraph" in dot  # DOT format starts with digraph

    def test_graph_merges_pattern_graphs(self):
        """Test merging pattern graphs into workflow graph."""
        from cortical.reasoning import create_investigation_graph

        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Pattern merge test")

        # Create investigation pattern
        investigation = create_investigation_graph(
            "Why is performance slow?",
            initial_hypotheses=[
                "Database queries too slow",
                "Frontend rendering bottleneck",
                "Network latency"
            ]
        )

        # Merge into workflow graph
        initial_count = ctx.thought_graph.node_count()

        for node_id, node in investigation.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id,
                    node.node_type,
                    node.content,
                    properties=node.properties
                )

        # Graph should have more nodes
        assert ctx.thought_graph.node_count() > initial_count

    def test_graph_tracks_decision_dependencies(self):
        """Test graph tracks dependencies between decisions."""
        from cortical.reasoning import NodeType

        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Decision tracking test")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)

        # Record decisions
        d1 = workflow.record_decision(
            ctx,
            "Use PostgreSQL for database",
            "Need ACID guarantees"
        )

        d2 = workflow.record_decision(
            ctx,
            "Use SQLAlchemy ORM",
            "Works well with PostgreSQL"
        )

        # Verify decisions exist in graph
        decision_nodes = ctx.thought_graph.nodes_of_type(NodeType.DECISION)
        assert len(decision_nodes) >= 2

        # Verify both decisions are in context
        assert len(ctx.decisions_made) >= 2
