"""
Behavioral tests for the reasoning framework.

Tests cover real-world usage scenarios:
- Bug investigation workflows
- Feature implementation workflows
- Complex multi-phase reasoning
- Human-AI collaboration scenarios
- Session management and handoffs
"""

import pytest
from cortical.reasoning import (
    # Core types
    NodeType,
    EdgeType,
    ThoughtGraph,
    # Workflow
    WorkflowContext,
    ReasoningWorkflow,
    # Cognitive loop
    LoopPhase,
    LoopStatus,
    TerminationReason,
    # Production
    ProductionState,
    ProductionChunk,
    # Crisis
    CrisisLevel,
    RecoveryAction,
    # Verification
    VerificationLevel,
    VerificationStatus,
    # Collaboration
    CollaborationMode,
    BlockerType,
    # Patterns
    create_investigation_graph,
    create_decision_graph,
    create_debug_graph,
    create_feature_graph,
    create_requirements_graph,
)


class TestBugInvestigationWorkflow:
    """Test realistic bug investigation scenarios."""

    def test_investigate_crash_bug(self):
        """
        Scenario: User reports app crashes on login.
        Expected: AI investigates systematically, identifies root cause.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Investigate login crash")

        # Start with investigation pattern
        investigation = create_investigation_graph(
            "Why does the app crash on login?",
            initial_hypotheses=[
                "Invalid user input",
                "Database connection timeout",
                "Authentication service failure"
            ]
        )

        # Merge pattern into workflow graph
        for node_id, node in investigation.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id, node.node_type, node.content,
                    properties=node.properties
                )

        # Question phase - gather information
        workflow.begin_question_phase(ctx)
        q1 = workflow.record_question(ctx, "What error message appears?")
        q2 = workflow.record_question(ctx, "Does it happen for all users?")
        q3 = workflow.record_question(ctx, "What changed recently?")

        # Answer phase - record findings
        workflow.begin_answer_phase(ctx)
        workflow.record_answer(ctx, q1, "NullPointerException in AuthService.java:45")
        workflow.record_answer(ctx, q2, "Only affects new users")
        workflow.record_answer(ctx, q3, "Deployed new auth module yesterday")

        # Record insight
        workflow.record_insight(ctx, "Bug is in new auth module for new users")

        # Produce phase - implement fix
        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "AuthService.java")

        # Verify phase - test fix
        workflow.begin_verify_phase(ctx)
        workflow.verify(ctx)

        # Complete
        summary = workflow.complete_session(ctx)

        # Verify investigation was thorough
        assert ctx.thought_graph.node_count() >= 6  # questions, answers, insights
        assert len(ctx.questions_answered) >= 3

    def test_investigate_performance_degradation(self):
        """
        Scenario: Users report slow page loads.
        Expected: AI profiles, identifies bottleneck, proposes optimization.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Performance investigation")

        # Debug pattern for symptoms
        debug = create_debug_graph("Pages loading slowly (>5s)")
        for node_id, node in debug.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id, node.node_type, node.content
                )

        workflow.begin_question_phase(ctx)
        q1 = workflow.record_question(ctx, "Which pages are slow?")
        q2 = workflow.record_question(ctx, "When did it start?")
        q3 = workflow.record_question(ctx, "Are there any timeout errors?")

        workflow.begin_answer_phase(ctx)
        workflow.record_answer(ctx, q1, "Dashboard and reports pages affected")
        workflow.record_answer(ctx, q2, "Started after adding new analytics feature")
        workflow.record_answer(ctx, q3, "Database queries timing out")

        # Decision
        workflow.record_decision(
            ctx,
            "Add database index",
            "Queries on analytics table are slow",
            ["Add index", "Add caching", "Optimize query"]
        )

        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "migration.sql")

        summary = workflow.complete_session(ctx)
        assert len(ctx.decisions_made) >= 1


class TestFeatureImplementationWorkflow:
    """Test realistic feature implementation scenarios."""

    def test_implement_user_preferences(self):
        """
        Scenario: Implement user preferences feature.
        Expected: AI plans, implements incrementally, verifies each step.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Add user preferences")

        # Use feature pattern
        feature = create_feature_graph(
            "User Preferences",
            "As a user, I want to customize my dashboard layout"
        )
        for node_id, node in feature.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id, node.node_type, node.content
                )

        # Planning
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What preferences should be configurable?")
        workflow.record_question(ctx, "Where to store preferences?")

        workflow.begin_answer_phase(ctx)
        workflow.record_decision(
            ctx,
            "Store in database with Redis cache",
            "Need persistence and fast access",
            ["Database only", "Redis only", "Database + Redis"]
        )

        # Implementation in chunks
        workflow.begin_production_phase(ctx)

        # Create production chunks
        chunk1 = workflow.create_production_chunk(
            ctx,
            name="data_model",
            goal="Define preferences data model",
            files=["preferences_model.py"]
        )

        chunk2 = workflow.create_production_chunk(
            ctx,
            name="api",
            goal="Create API endpoints",
            files=["preferences_api.py"]
        )

        chunk3 = workflow.create_production_chunk(
            ctx,
            name="frontend",
            goal="Build preferences UI",
            files=["PreferencesPanel.tsx"]
        )

        workflow.record_artifact(ctx, "preferences_model.py")
        workflow.record_artifact(ctx, "preferences_api.py")
        workflow.record_artifact(ctx, "PreferencesPanel.tsx")

        # Final verification
        workflow.begin_verify_phase(ctx)
        workflow.verify(ctx)

        summary = workflow.complete_session(ctx)

        assert len(ctx.artifacts_produced) == 3
        assert len(ctx.decisions_made) >= 1

    def test_implement_with_architectural_decision(self):
        """
        Scenario: Feature requires architectural decision.
        Expected: AI presents options, documents decision, implements.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Add real-time notifications")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What notification mechanisms to support?")
        workflow.record_question(ctx, "How to handle offline users?")

        workflow.begin_answer_phase(ctx)

        # Create decision graph for architecture choice
        decision = create_decision_graph(
            "Choose real-time communication method",
            ["WebSocket", "Server-Sent Events", "Long Polling"]
        )
        for node_id, node in decision.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id, node.node_type, node.content
                )

        # Record decision with rationale
        workflow.record_decision(
            ctx,
            "Use Server-Sent Events",
            "Simple implementation, good browser support, sufficient for one-way notifications",
            ["WebSocket", "SSE", "Long Polling"]
        )

        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "notifications.py")

        summary = workflow.complete_session(ctx)

        # Verify decision is documented
        assert len(ctx.decisions_made) >= 1
        assert "SSE" in ctx.decisions_made[0]['decision'] or "Server-Sent" in ctx.decisions_made[0]['decision']


class TestComplexMultiPhaseReasoning:
    """Test complex reasoning requiring multiple iterations."""

    def test_iterative_refinement(self):
        """
        Scenario: Solution needs refinement across multiple iterations.
        Expected: AI iterates and improves.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Implement search algorithm")

        # First iteration - basic implementation
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What search approach to use?")

        workflow.begin_answer_phase(ctx)
        workflow.record_decision(
            ctx, "Start with basic keyword matching",
            "Simple baseline to iterate from"
        )
        workflow.record_insight(ctx, "Basic matching won't handle synonyms")

        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "search_v1.py")

        # Second iteration - add improvements
        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "How to improve relevance?")

        workflow.begin_answer_phase(ctx)
        workflow.record_decision(
            ctx, "Add TF-IDF weighting",
            "Better relevance scoring"
        )

        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "search_v2.py")

        workflow.begin_verify_phase(ctx)
        workflow.verify(ctx)

        summary = workflow.complete_session(ctx)

        assert len(ctx.artifacts_produced) >= 2
        assert len(ctx.decisions_made) >= 2
        assert len(ctx.lessons_learned) >= 1


class TestHumanAICollaboration:
    """Test human-AI collaboration scenarios."""

    def test_status_updates(self):
        """
        Scenario: AI provides regular status updates.
        Expected: Updates are clear and informative.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Long running task")

        workflow.begin_question_phase(ctx)
        update1 = workflow.post_status(ctx, 10, "Gathering requirements")

        workflow.begin_answer_phase(ctx)
        update2 = workflow.post_status(ctx, 30, "Analyzing options")

        workflow.begin_production_phase(ctx)
        update3 = workflow.post_status(ctx, 60, "Implementing solution")

        assert update1.progress_percent == 10
        assert update2.progress_percent == 30
        assert update3.progress_percent == 60

    def test_disagreement_handling(self):
        """
        Scenario: AI disagrees with instruction.
        Expected: Disagreement is documented properly.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Implementation with constraints")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)

        # Record disagreement
        disagreement = workflow.raise_disagreement(
            ctx,
            instruction="Use eval() for dynamic code",
            concern="Security vulnerability",
            evidence=["OWASP Top 10", "Security audit findings"],
            risk="Remote code execution possible",
            alternative="Use safe parser or AST evaluation"
        )

        assert disagreement is not None

    def test_handoff_creation(self):
        """
        Scenario: AI creates handoff for session pause.
        Expected: Handoff captures context.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Feature requiring handoff")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What's the scope?")

        workflow.begin_answer_phase(ctx)
        workflow.record_decision(ctx, "Scope includes A, B, C", "Per requirements doc")

        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "implementation.py")

        # Create handoff
        handoff = workflow.create_handoff(ctx)

        assert handoff is not None
        assert handoff.task_description is not None


class TestCrisisAndRecoveryScenarios:
    """Test handling of crisis situations."""

    def test_crisis_levels_escalation(self):
        """
        Scenario: Issues escalate through crisis levels.
        Expected: Proper crisis handling.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Problematic task")

        # Need to start a phase before reporting crises that block the loop
        workflow.begin_question_phase(ctx)

        # Report escalating crises
        event1 = workflow.report_crisis(ctx, CrisisLevel.HICCUP, "Minor compilation error")
        assert event1.level == CrisisLevel.HICCUP

        event2 = workflow.report_crisis(ctx, CrisisLevel.OBSTACLE, "Tests failing consistently")
        assert event2.level == CrisisLevel.OBSTACLE

        event3 = workflow.report_crisis(ctx, CrisisLevel.WALL, "Blocked on external dependency")
        assert event3.level == CrisisLevel.WALL

    def test_recovery_with_insights(self):
        """
        Scenario: Learn from crisis and record insight.
        Expected: Lessons learned are captured.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Task with learning")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)

        # Hit an obstacle
        workflow.report_crisis(ctx, CrisisLevel.OBSTACLE, "Unexpected API rate limiting")

        # Record what was learned
        workflow.record_insight(ctx, "Always check API rate limits before integration")
        workflow.record_insight(ctx, "Need to implement exponential backoff")

        assert len(ctx.lessons_learned) >= 2


class TestSessionManagement:
    """Test session lifecycle management."""

    def test_complete_session_summary(self):
        """
        Scenario: Complete a session and get summary.
        Expected: Summary includes all relevant data.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Complete workflow test")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "What to build?")

        workflow.begin_answer_phase(ctx)
        workflow.record_decision(ctx, "Build X", "Because Y")

        workflow.begin_production_phase(ctx)
        workflow.record_artifact(ctx, "output.py")

        workflow.begin_verify_phase(ctx)
        workflow.verify(ctx)

        summary = workflow.complete_session(ctx)

        assert 'session_id' in summary
        assert 'goal' in summary
        assert 'decisions_made' in summary
        assert 'artifacts_produced' in summary

    def test_abandon_session(self):
        """
        Scenario: Abandon a session that can't continue.
        Expected: Proper cleanup and record.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Doomed task")

        workflow.begin_question_phase(ctx)
        workflow.record_question(ctx, "Is this feasible?")

        record = workflow.abandon_session(ctx, "Requirements impossible to meet")

        assert record['reason'] == "Requirements impossible to meet"
        assert 'lessons_learned' in record

    def test_workflow_summary(self):
        """
        Scenario: Get overall workflow status.
        Expected: Summary of all components.
        """
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Test session")

        summary = workflow.get_workflow_summary()

        assert 'active_sessions' in summary
        assert 'loops' in summary
        assert 'production' in summary
        assert 'crises' in summary


class TestPatternIntegration:
    """Test thought pattern integration with workflow."""

    def test_investigation_pattern_in_workflow(self):
        """Test using investigation pattern."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Debug issue")

        investigation = create_investigation_graph("Why is X failing?")

        # Merge pattern
        for node_id, node in investigation.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id, node.node_type, node.content
                )

        # Pattern should add structure
        questions = ctx.thought_graph.nodes_of_type(NodeType.QUESTION)
        assert len(questions) >= 1

    def test_requirements_pattern_in_workflow(self):
        """Test using requirements pattern."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Gather requirements")

        requirements = create_requirements_graph("User needs feature X")

        for node_id, node in requirements.nodes.items():
            if not ctx.thought_graph.get_node(node_id):
                ctx.thought_graph.add_node(
                    node_id, node.node_type, node.content
                )

        # Should have requirement-related nodes
        assert ctx.thought_graph.node_count() >= 5


class TestCommentMarkers:
    """Test production comment markers."""

    def test_add_comment_markers(self):
        """Test adding comment markers during production."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Implementation with markers")

        workflow.begin_question_phase(ctx)
        workflow.begin_answer_phase(ctx)
        workflow.begin_production_phase(ctx)

        # Add various markers
        workflow.add_comment_marker(ctx, "THINKING", "Considering approach X")
        workflow.add_comment_marker(ctx, "TODO", "Need to add error handling")
        workflow.add_comment_marker(ctx, "QUESTION", "Should we use library Y?")

        # Markers should be recorded in the production task
        # (actual assertion depends on production task implementation)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_session(self):
        """Test handling empty session."""
        workflow = ReasoningWorkflow()
        ctx = workflow.start_session("Empty test")

        summary = workflow.complete_session(ctx)

        assert summary['decisions_made'] == 0
        assert summary['artifacts_produced'] == 0

    def test_multiple_sessions(self):
        """Test managing multiple concurrent sessions."""
        workflow = ReasoningWorkflow()

        ctx1 = workflow.start_session("Session 1")
        ctx2 = workflow.start_session("Session 2")

        assert ctx1.session_id != ctx2.session_id

        # Each session must start from question phase
        workflow.begin_question_phase(ctx1)
        workflow.begin_question_phase(ctx2)
        workflow.begin_answer_phase(ctx2)
        workflow.begin_production_phase(ctx2)

        # Sessions should be independent
        summary = workflow.get_workflow_summary()
        assert summary['active_sessions'] >= 2
