"""
Behavioral tests for the agent task rejection protocol.

As a director agent managing sub-agents,
I want to validate task rejections with evidence,
So that I can distinguish legitimate blockers from lazy rejections.

Based on: examples/rejection_protocol_demo.py
"""

import pytest
from cortical.reasoning.rejection_protocol import (
    RejectionReason,
    TaskRejection,
    RejectionValidator,
    DecisionType,
    RejectionDecision,
    log_rejection_to_got,
    analyze_rejection_patterns,
)
from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import NodeType


class TestAgentRejectsTasksDueToScopeCreep:
    """
    Epic: Scope Creep Detection

    As a sub-agent attempting a task,
    I want to reject tasks when actual scope exceeds original estimate,
    So that tasks can be properly decomposed.
    """

    def test_scenario_agent_rejects_when_scope_grows_significantly(self):
        """
        Scenario: Agent detects and reports scope creep

        Given a task with original 2-hour estimate
        When investigation reveals 18-hour actual scope
        Then agent creates rejection with scope growth factor
        Because massive scope growth requires decomposition
        """
        # Given: task with small original scope
        task_id = "task:T-test-001"
        handoff_id = "handoff:H-test-001"
        agent_id = "test-agent"

        # When: agent discovers large actual scope
        rejection = TaskRejection(
            task_id=task_id,
            handoff_id=handoff_id,
            agent_id=agent_id,
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Task requires auth system redesign",
            reason_detail="Original scope: 2 hours. Actual scope: 18 hours.",
            what_attempted=["Analyzed codebase", "Estimated impact"],
            blocking_factor="Cannot complete without major refactoring",
            suggested_alternative="Decompose into 3 sequential tasks",
            task_original_scope="Add OAuth button (~2 hours)",
            scope_growth_factor=9.0,
        )

        # Then: rejection captures scope growth
        assert rejection.reason_type == RejectionReason.SCOPE_CREEP
        assert rejection.scope_growth_factor == 9.0
        assert rejection.scope_growth_factor > 5.0  # Significant growth

    def test_scenario_validator_accepts_well_documented_scope_creep(self):
        """
        Scenario: Well-documented scope creep passes validation

        Given a rejection with detailed evidence and alternatives
        When validator checks the rejection
        Then validation passes
        Because agent demonstrated due diligence
        """
        # Given: well-documented rejection
        rejection = TaskRejection(
            task_id="task:T-test-002",
            handoff_id="handoff:H-test-002",
            agent_id="test-agent",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Scope increased 8x",
            reason_detail="Detailed analysis shows 16-hour scope vs 2-hour estimate",
            what_attempted=[
                "Analyzed auth/ module - 47 files",
                "Reviewed tests - 89 files to update",
                "Estimated effort - 16 hours",
            ],
            blocking_factor="Auth module lacks plugin architecture",
            evidence=[
                {"type": "analysis", "data": {"files": 47}, "source": "static analysis"},
                {"type": "estimate", "data": {"hours": 16}, "source": "engineering"},
            ],
            suggested_alternative="Decompose into 3 tasks",
            alternative_tasks=[
                {"title": "Refactor auth", "estimate": "8 hours"},
                {"title": "Add OAuth plugin", "estimate": "4 hours"},
            ],
            task_original_scope="Add OAuth button",
            scope_growth_factor=8.0,
        )

        # When: validating
        validator = RejectionValidator()
        task_context = {"title": "Add OAuth", "scope": "Add OAuth login button"}
        is_valid, issues = validator.validate(rejection, task_context)

        # Then: validation passes
        assert is_valid
        assert len(issues) == 0


class TestAgentProvidesEvidenceForRejections:
    """
    Epic: Evidence-Based Rejection

    As a validation system,
    I want rejections to include concrete evidence,
    So that I can verify rejection legitimacy.
    """

    def test_scenario_rejection_includes_what_was_attempted(self):
        """
        Scenario: Agent documents investigation attempts

        Given an agent investigating a task
        When creating a rejection
        Then rejection lists specific attempts made
        Because validators need proof of effort
        """
        # Given & When: creating rejection with attempts
        rejection = TaskRejection(
            task_id="task:T-test-003",
            handoff_id="handoff:H-test-003",
            agent_id="test-agent",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="API credentials unavailable",
            reason_detail="Cannot deploy without credentials",
            what_attempted=[
                "Checked environment variables",
                "Checked config files",
                "Contacted ops team",
            ],
            blocking_factor="Credentials not provisioned",
        )

        # Then: attempts are documented
        assert len(rejection.what_attempted) >= 2  # Validator requires 2+
        assert all(isinstance(attempt, str) for attempt in rejection.what_attempted)

    def test_scenario_rejection_includes_supporting_evidence(self):
        """
        Scenario: Agent provides measurable evidence

        Given an agent analyzing task complexity
        When creating a rejection
        Then rejection includes quantitative evidence
        Because subjective claims are not sufficient
        """
        # Given & When: creating rejection with evidence
        rejection = TaskRejection(
            task_id="task:T-test-004",
            handoff_id="handoff:H-test-004",
            agent_id="test-agent",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Complex refactoring required",
            reason_detail="Analysis shows large impact",
            what_attempted=["Analyzed codebase", "Measured impact"],
            blocking_factor="Too complex for time available",
            evidence=[
                {"type": "metrics", "data": {"files_affected": 47}, "source": "analysis"},
                {"type": "estimate", "data": {"hours": 20}, "source": "estimation"},
            ],
        )

        # Then: evidence is quantitative
        assert len(rejection.evidence) > 0
        assert all(isinstance(e, dict) for e in rejection.evidence)
        assert all("data" in e for e in rejection.evidence)


class TestValidatorDetectsLazyRejections:
    """
    Epic: Lazy Rejection Prevention

    As a validation system,
    I want to detect insufficient rejection justifications,
    So that agents cannot avoid work through lazy rejections.
    """

    def test_scenario_validator_rejects_insufficient_attempts(self):
        """
        Scenario: Validator catches minimal effort rejections

        Given a rejection with only one vague attempt
        When validator checks the rejection
        Then validation fails with specific issues
        Because agent did not demonstrate sufficient effort
        """
        # Given: lazy rejection
        lazy_rejection = TaskRejection(
            task_id="task:T-test-005",
            handoff_id="handoff:H-test-005",
            agent_id="test-agent",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Too complex",
            reason_detail="This is too hard",
            what_attempted=["Looked at the code"],
            blocking_factor="It's confusing",
        )

        # When: validating
        validator = RejectionValidator()
        task_context = {"title": "Implement feature"}
        is_valid, issues = validator.validate(lazy_rejection, task_context)

        # Then: validation fails
        assert not is_valid
        assert len(issues) > 0

    def test_scenario_validator_requires_specific_blocking_factors(self):
        """
        Scenario: Validator demands measurable blocking factors

        Given a rejection with vague blocking factor
        When validator checks the rejection
        Then validation fails
        Because "too complex" is not actionable
        """
        # Given: rejection with vague blocker
        rejection = TaskRejection(
            task_id="task:T-test-006",
            handoff_id="handoff:H-test-006",
            agent_id="test-agent",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Task blocked",
            reason_detail="Cannot proceed",
            what_attempted=["Tried to do it", "Gave up"],
            blocking_factor="Too hard",  # Vague!
        )

        # When: validating
        validator = RejectionValidator()
        task_context = {"title": "Implement auth"}
        is_valid, issues = validator.validate(rejection, task_context)

        # Then: validation identifies vagueness
        assert not is_valid


class TestDirectorHandlesRejectionDecisions:
    """
    Epic: Rejection Decision Making

    As a director agent,
    I want to make decisions about valid rejections,
    So that work can continue productively.
    """

    def test_scenario_director_decomposes_scope_creep_rejection(self):
        """
        Scenario: Director decomposes task on scope creep

        Given a valid scope creep rejection with alternative tasks
        When director processes the rejection
        Then director creates decomposed sub-tasks
        Because scope creep requires task breakdown
        """
        # Given: valid scope creep rejection
        rejection = TaskRejection(
            task_id="task:T-test-007",
            handoff_id="handoff:H-test-007",
            agent_id="test-agent",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Scope grew 9x",
            reason_detail="Detailed evidence...",
            what_attempted=["Analysis 1", "Analysis 2"],
            blocking_factor="Auth refactor required",
            suggested_alternative="Decompose into 3 tasks",
            alternative_tasks=[
                {"title": "Refactor auth", "estimate": "8h"},
                {"title": "Add OAuth", "estimate": "4h"},
            ],
            scope_growth_factor=9.0,
        )

        # When: director decides
        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
            rejection=rejection,
            rationale="Valid scope creep, decomposing",
            created_tasks=["task:new-1", "task:new-2"],
        )

        # Then: tasks are decomposed
        assert decision.decision_type == DecisionType.ACCEPT_AND_DECOMPOSE
        assert len(decision.created_tasks) > 0

    def test_scenario_director_defers_task_on_external_blocker(self):
        """
        Scenario: Director defers task when external blocker exists

        Given a valid blocker rejection (external dependency)
        When director processes the rejection
        Then director defers the task
        Because external blockers require waiting
        """
        # Given: external blocker rejection
        rejection = TaskRejection(
            task_id="task:T-test-008",
            handoff_id="handoff:H-test-008",
            agent_id="test-agent",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="API credentials not provisioned",
            reason_detail="Ops team has not provided credentials",
            what_attempted=[
                "Checked environment",
                "Contacted ops team",
                "Verified ticket OPS-1234",
            ],
            blocking_factor="External dependency on ops team",
            evidence=[
                {"type": "ticket", "data": {"ticket": "OPS-1234"}, "source": "ops"},
            ],
        )

        # When: director decides
        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DEFER,
            rejection=rejection,
            rationale="Valid external blocker, deferring",
            deferred_task="task:T-test-008",
        )

        # Then: task is deferred
        assert decision.decision_type == DecisionType.ACCEPT_AND_DEFER
        assert decision.deferred_task is not None

    def test_scenario_director_overrides_lazy_rejection(self):
        """
        Scenario: Director overrides invalid rejection

        Given a rejection that fails validation
        When director processes the rejection
        Then director overrides with explanation
        Because lazy rejections must not be accepted
        """
        # Given: lazy rejection
        rejection = TaskRejection(
            task_id="task:T-test-009",
            handoff_id="handoff:H-test-009",
            agent_id="test-agent",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Too complex",
            reason_detail="This is hard",
            what_attempted=["Looked at it"],
            blocking_factor="Confusing",
        )

        # When: director overrides
        decision = RejectionDecision(
            decision_type=DecisionType.OVERRIDE,
            rejection=rejection,
            rationale="Insufficient evidence",
            override_message="Please retry with concrete evidence",
            reassign_to="test-agent",
        )

        # Then: override is recorded
        assert decision.decision_type == DecisionType.OVERRIDE
        assert decision.override_message is not None
        assert decision.reassign_to == "test-agent"


class TestSystemLearnsFromRejectionPatterns:
    """
    Epic: Pattern Learning

    As a learning system,
    I want to log rejections to the graph of thought,
    So that patterns can be analyzed over time.
    """

    def test_scenario_rejections_logged_to_thought_graph(self):
        """
        Scenario: Rejection logged as graph node

        Given a rejection and decision
        When logging to thought graph
        Then rejection node is created with metadata
        Because historical patterns inform future planning
        """
        # Given: rejection and decision
        graph = ThoughtGraph()

        task_id = "task:T-test-010"
        graph.add_node(
            task_id,
            NodeType.TASK,
            "Test task",
            properties={"status": "pending"},
        )

        rejection = TaskRejection(
            task_id=task_id,
            handoff_id="handoff:H-test-010",
            agent_id="test-agent",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Scope grew",
            reason_detail="Details",
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="Blocker",
            scope_growth_factor=5.0,
        )

        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
            rejection=rejection,
            rationale="Valid",
        )

        # When: logging to graph
        rejection_node_id = log_rejection_to_got(graph, rejection, decision)

        # Then: node exists with metadata
        assert rejection_node_id is not None
        assert rejection_node_id in graph.nodes
        node = graph.nodes[rejection_node_id]
        # Node properties include the rejection details
        assert "reason_type" in node.properties

    def test_scenario_system_analyzes_rejection_patterns(self):
        """
        Scenario: System identifies rejection patterns

        Given multiple rejections logged to graph
        When analyzing patterns
        Then statistics reveal trends
        Because patterns help improve task planning
        """
        # Given: multiple rejections in graph
        graph = ThoughtGraph()
        from datetime import datetime, timedelta
        base_time = datetime.now()

        for i in range(5):
            task_id = f"task:T-test-{100+i}"
            graph.add_node(
                task_id,
                NodeType.TASK,
                f"Task {i}",
                properties={"status": "pending"},
            )

            # Use unique timestamps to avoid ID collisions
            rejection = TaskRejection(
                task_id=task_id,
                handoff_id=f"handoff:H-test-{100+i}",
                agent_id="test-agent",
                reason_type=RejectionReason.SCOPE_CREEP if i % 2 == 0 else RejectionReason.BLOCKER,
                reason_summary=f"Rejection {i}",
                reason_detail="Details",
                what_attempted=["Attempt 1", "Attempt 2"],
                blocking_factor="Blocker",
                rejected_at=base_time + timedelta(seconds=i),  # Unique timestamp
            )

            decision = RejectionDecision(
                decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
                rejection=rejection,
                rationale="Valid",
            )

            log_rejection_to_got(graph, rejection, decision)

        # When: analyzing patterns
        patterns = analyze_rejection_patterns(graph)

        # Then: statistics exist
        assert "total_rejections" in patterns
        assert patterns["total_rejections"] > 0
        assert "by_reason" in patterns
        assert "by_agent" in patterns
