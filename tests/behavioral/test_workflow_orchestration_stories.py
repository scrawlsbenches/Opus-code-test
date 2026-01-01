"""
Behavioral Tests for Reasoning Workflow Orchestration.

This module tests the unified workflow orchestrator that coordinates
all reasoning components we built ourselves.

Epic: System architect orchestrates complex multi-phase workflows
Story: As a system architect building orchestration,
       I want unified workflow coordination we implemented,
       So that I can manage complex reasoning we control completely.
"""

import pytest
from cortical.reasoning.workflow import ReasoningWorkflow, WorkflowContext
from cortical.reasoning.cognitive_loop import LoopPhase, TerminationReason
from cortical.reasoning.collaboration import CollaborationMode


class TestSystemArchitectOrchestratesWorkflows:
    """
    Epic: System Architect Orchestrates Complex Workflows

    As a system architect building orchestration systems,
    I want workflow coordination we implemented ourselves,
    So that I control complex multi-component reasoning.
    """

    def test_scenario_workflow_session_coordinates_full_lifecycle(self):
        """
        Scenario: Session manages complete reasoning lifecycle

        Given a workflow orchestrator we built
        When I start and complete a reasoning session
        Then full lifecycle is coordinated
        Because we implemented session management ourselves
        """
        # Given workflow orchestrator
        workflow = ReasoningWorkflow()

        # When managing lifecycle
        context = workflow.start_session("Build custom query optimizer")
        workflow.begin_question_phase(context)
        workflow.begin_answer_phase(context)
        workflow.begin_production_phase(context)
        summary = workflow.complete_session(context, TerminationReason.SUCCESS)

        # Then lifecycle coordinated
        assert summary['session_id'] == context.session_id
        assert summary['goal'] == "Build custom query optimizer"
        assert summary['loop_status'] == 'COMPLETED'

    def test_scenario_workflow_captures_reasoning_decisions(self):
        """
        Scenario: Workflow records all reasoning decisions

        Given an active reasoning session we're managing
        When I record decisions and insights
        Then knowledge is captured in our system
        Because we built knowledge capture ourselves
        """
        # Given active session
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Design custom indexing strategy")
        workflow.begin_answer_phase(context)

        # When recording decisions
        decision_id = workflow.record_decision(
            context,
            "Use inverted index we implemented",
            "Full control over implementation details",
            ["B-tree approach", "Hash table approach"]
        )
        insight_id = workflow.record_insight(
            context,
            "Custom implementation gives us complete flexibility",
            source="analysis"
        )

        # Then knowledge captured
        assert len(context.decisions_made) == 1
        assert len(context.lessons_learned) == 1
        assert "inverted index" in context.decisions_made[0]['decision']

    def test_scenario_workflow_tracks_artifacts_produced(self):
        """
        Scenario: Workflow tracks all produced artifacts

        Given a production phase we're orchestrating
        When artifacts are created
        Then they're tracked in our system
        Because we monitor production ourselves
        """
        # Given production phase
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Implement custom search module")
        workflow.begin_production_phase(context)

        # When creating artifacts
        workflow.record_artifact(context, "/custom/indexer.py", "file")
        workflow.record_artifact(context, "/custom/ranker.py", "file")
        workflow.record_artifact(context, "/tests/test_custom_search.py", "test")

        # Then artifacts tracked
        assert len(context.artifacts_produced) == 3
        assert "/custom/indexer.py" in context.artifacts_produced

    def test_scenario_workflow_integrates_thought_graph(self):
        """
        Scenario: Workflow maintains reasoning graph

        Given a workflow session we're managing
        When I record questions and answers
        Then thought graph captures relationships
        Because we built graph integration ourselves
        """
        # Given workflow session
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Optimize custom query performance")
        workflow.begin_question_phase(context)

        # When recording Q&A
        q_id = workflow.record_question(context, "What bottleneck exists?", "exploration")
        a_id = workflow.record_answer(context, q_id, "Index lookup is slow", confidence=0.8)

        # Then graph captures structure
        assert context.thought_graph is not None
        assert context.thought_graph.node_count() >= 3  # goal + question + answer
        question_node = context.thought_graph.get_node(q_id)
        assert question_node is not None

    def test_scenario_workflow_coordinates_verification(self):
        """
        Scenario: Verification phase validates work

        Given completed production we're validating
        When I run verification checks
        Then workflow coordinates validation
        Because we built verification ourselves
        """
        # Given completed production
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Validate custom implementation")
        workflow.begin_production_phase(context)
        workflow.begin_verify_phase(context)

        # When running verification
        results = workflow.verify(context)

        # Then validation coordinated
        assert results is not None
        assert 'passed' in results or 'pending' in results


class TestWorkflowManagesCrisisAndRecovery:
    """
    Epic: Workflow Manages Crisis and Recovery

    As a system managing complex work,
    I want crisis detection and recovery,
    So that failures are handled gracefully.
    """

    def test_scenario_workflow_detects_crisis_situations(self):
        """
        Scenario: System detects and reports crises

        Given a workflow monitoring our work
        When a crisis situation occurs
        Then it's detected and reported
        Because we built crisis management ourselves
        """
        # Given monitoring workflow
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Handle crisis detection")

        # When crisis occurs
        from cortical.reasoning.crisis_manager import CrisisLevel
        event = workflow.report_crisis(
            context,
            CrisisLevel.OBSTACLE,
            "Custom module integration failing repeatedly"
        )

        # Then crisis reported
        assert event is not None
        assert event.level == CrisisLevel.OBSTACLE

    def test_scenario_workflow_can_abandon_failed_sessions(self):
        """
        Scenario: Failed sessions can be abandoned gracefully

        Given a session that cannot proceed
        When I abandon it with reason
        Then abandonment is recorded
        Because we handle failures ourselves
        """
        # Given failing session
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Attempt impossible task")
        workflow.begin_production_phase(context)

        # When abandoning
        record = workflow.abandon_session(
            context,
            "Discovered fundamental blocker in custom approach"
        )

        # Then abandonment recorded
        assert 'reason' in record
        assert "fundamental blocker" in record['reason']
        assert 'lessons_learned' in record


class TestCollaborationWorkflowCoordinatesHumanAI:
    """
    Epic: Collaboration Workflow Coordinates Human-AI Work

    As a system coordinating human-AI collaboration,
    I want status updates and handoffs,
    So that work transitions smoothly.
    """

    def test_scenario_workflow_posts_progress_updates(self):
        """
        Scenario: Progress updates keep collaborators informed

        Given ongoing work we're coordinating
        When I post status updates
        Then collaborators are informed
        Because we built status tracking ourselves
        """
        # Given ongoing work
        workflow = ReasoningWorkflow(CollaborationMode.SEMI_SYNCHRONOUS)
        context = workflow.start_session("Build custom feature")
        workflow.begin_production_phase(context)

        # When posting status
        update = workflow.post_status(
            context,
            progress=60,
            current_activity="Implementing custom indexing logic"
        )

        # Then status communicated
        assert update.progress_percent == 60
        assert "indexing" in update.current_activity.lower()

    def test_scenario_workflow_generates_handoff_documentation(self):
        """
        Scenario: Handoff documents preserve context

        Given work we need to hand off
        When I create handoff documentation
        Then context is preserved for continuation
        Because we built handoff generation ourselves
        """
        # Given work to hand off
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Complex multi-phase work")
        workflow.begin_production_phase(context)
        workflow.record_decision(context, "Use custom approach", "Best fit")
        workflow.record_artifact(context, "/module.py", "file")

        # When creating handoff
        handoff = workflow.create_handoff(context)

        # Then context preserved
        assert handoff.task == context.goal
        assert len(handoff.key_decisions) > 0
        assert handoff.files_working is not None

    def test_scenario_workflow_raises_disagreements_when_needed(self):
        """
        Scenario: System can raise concerns about direction

        Given instructions that seem problematic
        When I raise a disagreement
        Then concerns are formally documented
        Because we built disagreement tracking ourselves
        """
        # Given problematic instruction
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Implement questionable approach")

        # When raising disagreement
        disagreement = workflow.raise_disagreement(
            context,
            instruction="Adopt third-party library",
            concern="Violates our build-everything-ourselves principle",
            evidence=["Past dependency issues", "Loss of control"],
            risk="Cannot maintain or debug external code",
            alternative="Implement the functionality ourselves from scratch"
        )

        # Then concern documented
        assert disagreement is not None
        assert "build-everything-ourselves" in disagreement.concern.lower()


class TestWorkflowProvidesComprehensiveReporting:
    """
    Epic: Workflow Provides Comprehensive Reporting

    As a system operator monitoring reasoning,
    I want aggregate statistics and reporting,
    So that I understand system health.
    """

    def test_scenario_workflow_reports_aggregate_statistics(self):
        """
        Scenario: Summary reports show system health

        Given multiple active workflows
        When I query aggregate statistics
        Then comprehensive health metrics are shown
        Because we built monitoring ourselves
        """
        # Given active workflows
        workflow = ReasoningWorkflow()
        ctx1 = workflow.start_session("Task 1")
        workflow.begin_production_phase(ctx1)
        ctx2 = workflow.start_session("Task 2")

        # When querying statistics
        summary = workflow.get_workflow_summary()

        # Then health metrics shown
        assert 'active_sessions' in summary
        assert 'loops' in summary
        assert 'production' in summary
        assert summary['active_sessions'] == 2

    def test_scenario_session_summary_captures_complete_history(self):
        """
        Scenario: Session summaries preserve reasoning history

        Given a completed session
        When I generate summary
        Then complete history is captured
        Because we track everything ourselves
        """
        # Given completed session
        workflow = ReasoningWorkflow()
        context = workflow.start_session("Complete workflow")
        workflow.begin_question_phase(context)
        workflow.record_question(context, "What approach to use?")
        workflow.begin_answer_phase(context)
        workflow.record_decision(context, "Custom implementation", "We control it")
        workflow.begin_production_phase(context)
        workflow.record_artifact(context, "/output.py")

        summary = workflow.complete_session(context)

        # Then history captured
        assert summary['decisions_made'] >= 1
        assert summary['artifacts_produced'] >= 1
        assert summary['thought_graph_nodes'] > 0
