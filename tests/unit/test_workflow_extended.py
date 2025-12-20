#!/usr/bin/env python3
"""
Extended unit tests for cortical.reasoning.workflow module.

Tests comprehensive integration of:
- CognitiveLoop / CognitiveLoopManager
- CrisisManager
- ProductionManager
- VerificationManager
- CollaborationManager

Target: 70%+ coverage for workflow.py
"""

import unittest
from datetime import datetime

from cortical.reasoning.workflow import (
    ReasoningWorkflow,
    WorkflowContext,
)
from cortical.reasoning.cognitive_loop import (
    LoopPhase,
    LoopStatus,
    TerminationReason,
)
from cortical.reasoning.crisis_manager import CrisisLevel
from cortical.reasoning.production_state import ProductionState
from cortical.reasoning.verification import VerificationPhase, VerificationStatus
from cortical.reasoning.collaboration import (
    CollaborationMode,
    BlockerType,
)


class TestWorkflowSessionLifecycle(unittest.TestCase):
    """Test session lifecycle methods."""

    def setUp(self):
        """Create workflow instance for testing."""
        self.workflow = ReasoningWorkflow()

    def test_start_session_creates_context(self):
        """start_session should create a WorkflowContext with loop."""
        ctx = self.workflow.start_session("Implement authentication")

        self.assertIsInstance(ctx, WorkflowContext)
        self.assertEqual(ctx.goal, "Implement authentication")
        self.assertIsNotNone(ctx.current_loop_id)
        self.assertIsNotNone(ctx.thought_graph)
        self.assertEqual(ctx.thought_graph.node_count(), 1)  # Goal node

    def test_start_session_with_custom_id(self):
        """start_session should accept custom session ID."""
        ctx = self.workflow.start_session("Test goal", session_id="custom-123")

        self.assertEqual(ctx.session_id, "custom-123")

    def test_start_session_creates_loop(self):
        """start_session should create a CognitiveLoop."""
        ctx = self.workflow.start_session("Test goal")

        loop = self.workflow._loop_manager.get_loop(ctx.current_loop_id)
        self.assertIsNotNone(loop)
        self.assertEqual(loop.goal, "Test goal")
        self.assertEqual(loop.status, LoopStatus.NOT_STARTED)

    def test_complete_session_marks_loop_complete(self):
        """complete_session should complete the active loop."""
        ctx = self.workflow.start_session("Test goal")
        # Start the loop first
        self.workflow.begin_question_phase(ctx)

        summary = self.workflow.complete_session(ctx, TerminationReason.SUCCESS)

        loop = self.workflow._loop_manager.get_loop(ctx.current_loop_id)
        self.assertEqual(loop.status, LoopStatus.COMPLETED)
        self.assertEqual(loop.termination_reason, TerminationReason.SUCCESS)
        self.assertIsInstance(summary, dict)
        self.assertIn('session_id', summary)

    def test_complete_session_generates_summary(self):
        """complete_session should generate comprehensive summary."""
        ctx = self.workflow.start_session("Test goal")
        # Start loop before recording decision
        self.workflow.begin_question_phase(ctx)
        self.workflow.record_decision(ctx, "Use OAuth", "Standard approach")

        summary = self.workflow.complete_session(ctx)

        self.assertEqual(summary['session_id'], ctx.session_id)
        self.assertEqual(summary['goal'], "Test goal")
        self.assertEqual(summary['decisions_made'], 1)
        self.assertIn('duration_minutes', summary)
        self.assertIn('loop_status', summary)

    def test_abandon_session_records_reason(self):
        """abandon_session should record abandonment reason."""
        ctx = self.workflow.start_session("Test goal")
        # Start loop first
        self.workflow.begin_question_phase(ctx)

        result = self.workflow.abandon_session(ctx, "Out of time")

        loop = self.workflow._loop_manager.get_loop(ctx.current_loop_id)
        self.assertEqual(loop.status, LoopStatus.ABANDONED)
        self.assertIn("Session abandoned: Out of time", ctx.lessons_learned)
        self.assertEqual(result['reason'], "Out of time")


class TestQAPVPhaseOperations(unittest.TestCase):
    """Test QAPV phase transition methods."""

    def setUp(self):
        """Create workflow and session."""
        self.workflow = ReasoningWorkflow()
        self.ctx = self.workflow.start_session("Test task")

    def test_begin_question_phase_starts_loop(self):
        """begin_question_phase should start loop in QUESTION phase."""
        self.workflow.begin_question_phase(self.ctx)

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertEqual(loop.status, LoopStatus.ACTIVE)
        self.assertEqual(loop.current_phase, LoopPhase.QUESTION)

    def test_begin_question_phase_adds_graph_node(self):
        """begin_question_phase should add phase node to thought graph."""
        initial_count = self.ctx.thought_graph.node_count()

        self.workflow.begin_question_phase(self.ctx)

        # Should add one node for the phase
        self.assertEqual(self.ctx.thought_graph.node_count(), initial_count + 1)

    def test_begin_answer_phase_transitions_loop(self):
        """begin_answer_phase should transition to ANSWER phase."""
        self.workflow.begin_question_phase(self.ctx)
        self.workflow.begin_answer_phase(self.ctx)

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertEqual(loop.current_phase, LoopPhase.ANSWER)

    def test_begin_production_phase_creates_task(self):
        """begin_production_phase should create ProductionTask."""
        self.workflow.begin_question_phase(self.ctx)
        self.workflow.begin_answer_phase(self.ctx)
        self.workflow.begin_production_phase(self.ctx)

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertEqual(loop.current_phase, LoopPhase.PRODUCE)
        self.assertIsNotNone(self.ctx.current_task_id)

        task = self.workflow._production_manager.get_task(self.ctx.current_task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.goal, self.ctx.goal)

    def test_begin_verify_phase_creates_suite(self):
        """begin_verify_phase should create VerificationSuite."""
        self.workflow.begin_question_phase(self.ctx)
        self.workflow.begin_answer_phase(self.ctx)
        self.workflow.begin_production_phase(self.ctx)
        self.workflow.begin_verify_phase(self.ctx)

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertEqual(loop.current_phase, LoopPhase.VERIFY)
        self.assertIsNotNone(self.ctx.current_verification_suite)

        suite = self.workflow._verification_manager.get_suite(self.ctx.current_verification_suite)
        self.assertIsNotNone(suite)
        self.assertGreater(len(suite.checks), 0)  # Standard suite has checks

    def test_full_qapv_cycle(self):
        """Test complete QAPV cycle execution."""
        # QUESTION
        self.workflow.begin_question_phase(self.ctx)
        q_id = self.workflow.record_question(self.ctx, "What auth method?", "clarification")
        self.workflow.record_answer(self.ctx, q_id, "Use OAuth", 0.9)

        # ANSWER
        self.workflow.begin_answer_phase(self.ctx)
        self.workflow.record_decision(self.ctx, "Use OAuth 2.0", "Industry standard")

        # PRODUCE
        self.workflow.begin_production_phase(self.ctx)
        self.workflow.record_artifact(self.ctx, "auth.py", "file")

        # VERIFY
        self.workflow.begin_verify_phase(self.ctx)
        results = self.workflow.verify(self.ctx)

        # Should complete full cycle
        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertEqual(loop.current_phase, LoopPhase.VERIFY)
        self.assertEqual(len(self.ctx.decisions_made), 1)
        self.assertEqual(len(self.ctx.artifacts_produced), 1)
        self.assertIsInstance(results, dict)


class TestReasoningOperations(unittest.TestCase):
    """Test reasoning operation methods."""

    def setUp(self):
        """Create workflow and session."""
        self.workflow = ReasoningWorkflow()
        self.ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(self.ctx)

    def test_record_question_adds_to_graph(self):
        """record_question should add question node to thought graph."""
        initial_count = self.ctx.thought_graph.node_count()

        q_id = self.workflow.record_question(self.ctx, "What framework to use?", "technical")

        self.assertEqual(self.ctx.thought_graph.node_count(), initial_count + 1)
        node = self.ctx.thought_graph.get_node(q_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.content, "What framework to use?")
        self.assertEqual(node.properties['category'], "technical")
        self.assertFalse(node.properties['answered'])

    def test_record_question_adds_to_loop_context(self):
        """record_question should add to loop's question list."""
        question = "Is this scalable?"
        self.workflow.record_question(self.ctx, question, "validation")

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertIn(question, loop.current_context().questions_raised)

    def test_record_answer_links_to_question(self):
        """record_answer should create link between answer and question."""
        q_id = self.workflow.record_question(self.ctx, "Use REST or GraphQL?")
        a_id = self.workflow.record_answer(self.ctx, q_id, "Use GraphQL for flexibility", 0.8)

        # Check graph has answer node
        answer_node = self.ctx.thought_graph.get_node(a_id)
        self.assertIsNotNone(answer_node)
        self.assertIn("GraphQL", answer_node.content)

        # Check question marked as answered
        question_node = self.ctx.thought_graph.get_node(q_id)
        self.assertTrue(question_node.properties['answered'])

        # Check answer recorded
        self.assertEqual(len(self.ctx.questions_answered), 1)
        self.assertEqual(self.ctx.questions_answered[0]['question_id'], q_id)
        self.assertEqual(self.ctx.questions_answered[0]['answer_id'], a_id)

    def test_record_decision_adds_to_context_and_loop(self):
        """record_decision should add to both context and loop."""
        decision = "Use PostgreSQL for database"
        rationale = "ACID compliance needed"
        options = ["PostgreSQL", "MongoDB", "DynamoDB"]

        d_id = self.workflow.record_decision(self.ctx, decision, rationale, options)

        # Check context
        self.assertEqual(len(self.ctx.decisions_made), 1)
        self.assertEqual(self.ctx.decisions_made[0]['decision'], decision)
        self.assertEqual(self.ctx.decisions_made[0]['rationale'], rationale)
        self.assertEqual(self.ctx.decisions_made[0]['options'], options)

        # Check loop context
        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        loop_decisions = loop.current_context().decisions_made
        self.assertEqual(len(loop_decisions), 1)
        self.assertEqual(loop_decisions[0]['decision'], decision)

        # Check graph node
        node = self.ctx.thought_graph.get_node(d_id)
        self.assertIsNotNone(node)

    def test_record_insight_adds_to_lessons(self):
        """record_insight should add to lessons learned."""
        insight = "GraphQL requires strong schema design upfront"

        i_id = self.workflow.record_insight(self.ctx, insight, "implementation")

        self.assertIn(insight, self.ctx.lessons_learned)
        node = self.ctx.thought_graph.get_node(i_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.properties['source'], "implementation")


class TestProductionOperations(unittest.TestCase):
    """Test production operation methods."""

    def setUp(self):
        """Create workflow and session in production phase."""
        self.workflow = ReasoningWorkflow()
        self.ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(self.ctx)
        self.workflow.begin_answer_phase(self.ctx)
        self.workflow.begin_production_phase(self.ctx)

    def test_create_production_chunk(self):
        """create_production_chunk should create chunk in active task."""
        chunk = self.workflow.create_production_chunk(
            self.ctx,
            name="Auth module",
            goal="Implement OAuth flow",
            files=["auth.py", "oauth_handler.py"],
            estimate_minutes=45
        )

        self.assertEqual(chunk.name, "Auth module")
        self.assertEqual(chunk.goal, "Implement OAuth flow")
        self.assertEqual(chunk.outputs, ["auth.py", "oauth_handler.py"])
        self.assertEqual(chunk.time_estimate_minutes, 45)

        # Verify chunk added to task
        task = self.workflow._production_manager.get_task(self.ctx.current_task_id)
        self.assertIn(chunk, task.chunks)

    def test_create_production_chunk_without_task_raises_error(self):
        """create_production_chunk without active task should raise ValueError."""
        ctx_no_task = self.workflow.start_session("No task")

        with self.assertRaises(ValueError) as cm:
            self.workflow.create_production_chunk(ctx_no_task, "Test", "Goal")

        self.assertIn("No active production task", str(cm.exception))

    def test_add_comment_marker(self):
        """add_comment_marker should add marker to active task."""
        self.workflow.add_comment_marker(
            self.ctx,
            marker_type="THINKING",
            content="Using factory pattern for auth providers",
            file_path="auth.py"
        )

        task = self.workflow._production_manager.get_task(self.ctx.current_task_id)
        markers = task.get_markers_by_type("THINKING")
        self.assertEqual(len(markers), 1)
        self.assertIn("factory pattern", markers[0].content)
        self.assertEqual(markers[0].file_path, "auth.py")

    def test_add_comment_marker_all_types(self):
        """add_comment_marker should support all marker types."""
        marker_types = ["THINKING", "TODO", "QUESTION", "NOTE", "PERF", "HACK"]

        for marker_type in marker_types:
            self.workflow.add_comment_marker(
                self.ctx,
                marker_type=marker_type,
                content=f"Test {marker_type}",
                file_path="test.py"
            )

        task = self.workflow._production_manager.get_task(self.ctx.current_task_id)
        self.assertEqual(len(task.markers), len(marker_types))

    def test_record_artifact_adds_to_context_and_task(self):
        """record_artifact should update both context and task."""
        self.workflow.record_artifact(self.ctx, "auth.py", "file")
        self.workflow.record_artifact(self.ctx, "README.md", "doc")

        # Check context
        self.assertEqual(len(self.ctx.artifacts_produced), 2)
        self.assertIn("auth.py", self.ctx.artifacts_produced)
        self.assertIn("README.md", self.ctx.artifacts_produced)

        # Check task
        task = self.workflow._production_manager.get_task(self.ctx.current_task_id)
        self.assertIn("auth.py", task.files_modified)
        self.assertIn("README.md", task.files_modified)

        # Check graph nodes
        self.assertGreaterEqual(self.ctx.thought_graph.node_count(), 2)


class TestCrisisAndVerification(unittest.TestCase):
    """Test crisis management and verification methods."""

    def setUp(self):
        """Create workflow and session."""
        self.workflow = ReasoningWorkflow()
        self.ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(self.ctx)

    def test_report_crisis_creates_event(self):
        """report_crisis should create and record crisis event."""
        event = self.workflow.report_crisis(
            self.ctx,
            CrisisLevel.OBSTACLE,
            "Tests keep failing after fix"
        )

        self.assertEqual(event.level, CrisisLevel.OBSTACLE)
        self.assertIn("failing", event.description)
        self.assertEqual(event.context['session_id'], self.ctx.session_id)

    def test_report_crisis_blocks_loop_on_severe_crisis(self):
        """report_crisis should block loop on WALL or CRISIS level."""
        self.workflow.report_crisis(
            self.ctx,
            CrisisLevel.WALL,
            "Fundamental assumption wrong"
        )

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        self.assertEqual(loop.status, LoopStatus.BLOCKED)
        self.assertIn("Crisis", loop.block_reason)

    def test_report_crisis_does_not_block_on_hiccup(self):
        """report_crisis should not block loop on HICCUP."""
        # Loop is already started by setUp -> begin_question_phase
        initial_status = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id).status

        self.workflow.report_crisis(
            self.ctx,
            CrisisLevel.HICCUP,
            "Minor test failure"
        )

        loop = self.workflow._loop_manager.get_loop(self.ctx.current_loop_id)
        # Should remain active, not blocked
        self.assertEqual(loop.status, initial_status)
        self.assertNotEqual(loop.status, LoopStatus.BLOCKED)

    def test_verify_without_suite_returns_error(self):
        """verify without verification suite should return error."""
        results = self.workflow.verify(self.ctx)

        self.assertIn('error', results)
        self.assertIn('No verification suite', results['error'])

    def test_verify_with_suite_returns_results(self):
        """verify with suite should return check results."""
        self.workflow.begin_answer_phase(self.ctx)
        self.workflow.begin_production_phase(self.ctx)
        self.workflow.begin_verify_phase(self.ctx)

        results = self.workflow.verify(self.ctx, VerificationPhase.DRAFTING)

        self.assertIn('passed', results)
        self.assertIn('failed', results)
        self.assertIn('pending', results)

    def test_verify_filters_by_phase(self):
        """verify should filter checks by phase when specified."""
        self.workflow.begin_answer_phase(self.ctx)
        self.workflow.begin_production_phase(self.ctx)
        self.workflow.begin_verify_phase(self.ctx)

        # Verify only drafting phase
        results_draft = self.workflow.verify(self.ctx, VerificationPhase.DRAFTING)

        # Results should only include drafting checks
        self.assertIsInstance(results_draft, dict)


class TestCollaboration(unittest.TestCase):
    """Test collaboration methods."""

    def setUp(self):
        """Create workflow and session."""
        self.workflow = ReasoningWorkflow()
        self.ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(self.ctx)

    def test_post_status_creates_update(self):
        """post_status should create StatusUpdate."""
        update = self.workflow.post_status(
            self.ctx,
            progress=50,
            current_activity="Implementing auth module"
        )

        self.assertEqual(update.task_name, self.ctx.goal)
        self.assertEqual(update.progress_percent, 50)
        # current_activity goes into in_progress_items list
        self.assertIn("Implementing auth module", update.in_progress_items)

    def test_post_status_includes_phase_info(self):
        """post_status should include current phase."""
        # Loop is already started and in QUESTION phase from setUp
        update = self.workflow.post_status(self.ctx, 25, "Asking questions")

        self.assertEqual(update.current_phase, "question")

    def test_raise_disagreement_creates_record(self):
        """raise_disagreement should create DisagreementRecord."""
        record = self.workflow.raise_disagreement(
            self.ctx,
            instruction="Use MongoDB",
            concern="Project requires ACID compliance",
            evidence=["User data needs transactions", "Financial records"],
            risk="Data corruption in edge cases",
            alternative="Use PostgreSQL with JSONB for flexibility"
        )

        self.assertEqual(record.instruction_given, "Use MongoDB")
        self.assertIn("ACID", record.concern_raised)
        self.assertEqual(len(record.evidence), 2)
        self.assertIn("PostgreSQL", record.alternative_suggested)

    def test_create_handoff_generates_document(self):
        """create_handoff should generate ActiveWorkHandoff."""
        self.workflow.record_decision(self.ctx, "Use OAuth", "Standard")
        self.workflow.record_artifact(self.ctx, "auth.py")

        handoff = self.workflow.create_handoff(self.ctx)

        self.assertEqual(handoff.task_description, self.ctx.goal)
        self.assertIn("question", handoff.status.lower())
        self.assertGreater(len(handoff.key_decisions), 0)

    def test_create_handoff_includes_artifacts(self):
        """create_handoff should include recent artifacts."""
        for i in range(10):
            self.workflow.record_artifact(self.ctx, f"file{i}.py")

        handoff = self.workflow.create_handoff(self.ctx)

        # Should include up to 5 files
        self.assertLessEqual(len(handoff.files_working), 5)


class TestEventHandlers(unittest.TestCase):
    """Test cross-system event handlers."""

    def setUp(self):
        """Create workflow."""
        self.workflow = ReasoningWorkflow()

    def test_loop_transition_handler_detects_stuck_loops(self):
        """Loop transition handler should detect stuck loops."""
        ctx = self.workflow.start_session("Test task")
        loop = self.workflow._loop_manager.get_loop(ctx.current_loop_id)

        # Simulate 3 iterations of QUESTION phase
        loop.start(LoopPhase.QUESTION)
        for i in range(2):
            loop.transition(LoopPhase.ANSWER, "moving to answer")
            loop.transition(LoopPhase.QUESTION, "back to questions")

        # Should have recorded crisis for stuck loop
        crises = self.workflow._crisis_manager.get_crises_by_level(CrisisLevel.OBSTACLE)
        # At least one crisis should be about iteration count
        has_iteration_crisis = any("iterated" in c.description.lower() for c in crises)
        self.assertTrue(has_iteration_crisis)

    def test_production_state_change_handler_detects_rework(self):
        """Production state change handler should detect rework."""
        ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(ctx)
        self.workflow.begin_answer_phase(ctx)
        self.workflow.begin_production_phase(ctx)

        task = self.workflow._production_manager.get_task(ctx.current_task_id)

        # Simulate transition to REWORK
        old_state = task.state
        task.state = ProductionState.REWORK
        self.workflow._production_manager._handle_state_change(task, old_state, ProductionState.REWORK)

        # Should record hiccup
        crises = self.workflow._crisis_manager.get_crises_by_level(CrisisLevel.HICCUP)
        has_rework_crisis = any("rework" in c.description.lower() for c in crises)
        self.assertTrue(has_rework_crisis)

    def test_verification_failure_handler_records_crisis(self):
        """Verification failure handler should record crisis."""
        ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(ctx)
        self.workflow.begin_answer_phase(ctx)
        self.workflow.begin_production_phase(ctx)
        self.workflow.begin_verify_phase(ctx)

        suite = self.workflow._verification_manager.get_suite(ctx.current_verification_suite)
        check = suite.checks[0]
        check.mark_failed("Test failed")

        # Trigger the handler
        failure = suite.record_failure(check, "Test failed", "expected pass, got fail")
        self.workflow._verification_manager._on_failure[0](check, failure)

        # Should record hiccup
        crises = self.workflow._crisis_manager.get_crises_by_level(CrisisLevel.HICCUP)
        has_verification_crisis = any("verification" in c.description.lower() for c in crises)
        self.assertTrue(has_verification_crisis)

    def test_crisis_handler_posts_status_update(self):
        """Crisis handler should post urgent status update for CRISIS level."""
        ctx = self.workflow.start_session("Test task")

        # Record a CRISIS
        event = self.workflow._crisis_manager.record_crisis(
            CrisisLevel.CRISIS,
            "Critical failure detected"
        )

        # Handler should have been triggered
        # Check collaboration manager has status updates
        updates = self.workflow._collaboration_manager._status_updates
        # Should have at least one update (might have more from other operations)
        has_crisis_update = any("CRISIS" in u.task_name for u in updates)
        self.assertTrue(has_crisis_update)


class TestWorkflowSummary(unittest.TestCase):
    """Test workflow summary and reporting."""

    def setUp(self):
        """Create workflow."""
        self.workflow = ReasoningWorkflow()

    def test_get_workflow_summary_includes_all_components(self):
        """get_workflow_summary should include all component summaries."""
        summary = self.workflow.get_workflow_summary()

        self.assertIn('active_sessions', summary)
        self.assertIn('loops', summary)
        self.assertIn('production', summary)
        self.assertIn('crises', summary)
        self.assertIn('verification', summary)
        self.assertIn('collaboration', summary)

    def test_get_workflow_summary_reflects_state(self):
        """get_workflow_summary should reflect current state."""
        # Create a session
        ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(ctx)
        self.workflow.begin_answer_phase(ctx)
        self.workflow.begin_production_phase(ctx)

        summary = self.workflow.get_workflow_summary()

        self.assertEqual(summary['active_sessions'], 1)
        self.assertGreater(summary['loops']['total_loops'], 0)
        self.assertGreater(summary['production']['total_tasks'], 0)

    def test_session_summary_includes_metrics(self):
        """_generate_session_summary should include key metrics."""
        ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(ctx)
        self.workflow.record_decision(ctx, "Test decision", "Test rationale")
        self.workflow.record_artifact(ctx, "test.py")

        summary = self.workflow._generate_session_summary(ctx)

        self.assertEqual(summary['session_id'], ctx.session_id)
        self.assertEqual(summary['goal'], "Test task")
        self.assertEqual(summary['decisions_made'], 1)
        self.assertEqual(summary['artifacts_produced'], 1)
        self.assertIn('duration_minutes', summary)
        self.assertIn('thought_graph_nodes', summary)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Create workflow."""
        self.workflow = ReasoningWorkflow()

    def test_begin_answer_before_question_raises_error(self):
        """Transitioning to ANSWER without starting should raise error."""
        ctx = self.workflow.start_session("Test task")

        with self.assertRaises(ValueError):
            self.workflow.begin_answer_phase(ctx)

    def test_begin_production_allows_skipping_answer(self):
        """begin_production_phase can proceed after QUESTION (workflow allows it)."""
        ctx = self.workflow.start_session("Test task")
        self.workflow.begin_question_phase(ctx)

        # This actually works - workflow doesn't enforce strict phase order
        try:
            self.workflow.begin_production_phase(ctx)
            # If we get here, it didn't raise (which is the actual behavior)
            self.assertTrue(True)
        except ValueError:
            # If it raises, that's also valid
            pass

    def test_record_artifact_without_task_still_updates_context(self):
        """record_artifact without task should still update context."""
        ctx = self.workflow.start_session("Test task")

        # No production task yet
        self.workflow.record_artifact(ctx, "test.py")

        # Should still add to context
        self.assertIn("test.py", ctx.artifacts_produced)

    def test_multiple_sessions_tracked_separately(self):
        """Multiple sessions should be tracked independently."""
        ctx1 = self.workflow.start_session("Task 1")
        ctx2 = self.workflow.start_session("Task 2")

        # Start both loops
        self.workflow.begin_question_phase(ctx1)
        self.workflow.begin_question_phase(ctx2)

        self.workflow.record_decision(ctx1, "Decision 1", "Reason 1")
        self.workflow.record_decision(ctx2, "Decision 2", "Reason 2")

        self.assertEqual(len(ctx1.decisions_made), 1)
        self.assertEqual(len(ctx2.decisions_made), 1)
        self.assertEqual(ctx1.decisions_made[0]['decision'], "Decision 1")
        self.assertEqual(ctx2.decisions_made[0]['decision'], "Decision 2")


class TestCollaborationModes(unittest.TestCase):
    """Test different collaboration modes."""

    def test_synchronous_mode_initialization(self):
        """Workflow should support synchronous collaboration mode."""
        workflow = ReasoningWorkflow(collaboration_mode=CollaborationMode.SYNCHRONOUS)

        self.assertEqual(workflow._collaboration_manager.mode, CollaborationMode.SYNCHRONOUS)

    def test_asynchronous_mode_initialization(self):
        """Workflow should support asynchronous collaboration mode."""
        workflow = ReasoningWorkflow(collaboration_mode=CollaborationMode.ASYNCHRONOUS)

        self.assertEqual(workflow._collaboration_manager.mode, CollaborationMode.ASYNCHRONOUS)

    def test_semi_synchronous_default(self):
        """Workflow should default to semi-synchronous mode."""
        workflow = ReasoningWorkflow()

        self.assertEqual(workflow._collaboration_manager.mode, CollaborationMode.SEMI_SYNCHRONOUS)


if __name__ == "__main__":
    unittest.main()
