"""
Tests for Agent Rejection Protocol

Tests the structured rejection system that enables agents to reject tasks
with validated reasons while preventing lazy rejections.
"""

import unittest
from datetime import datetime, timedelta

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
from cortical.reasoning.graph_of_thought import NodeType, EdgeType


class TestRejectionReason(unittest.TestCase):
    """Test RejectionReason enum."""

    def test_rejection_reasons_exist(self):
        """Test all expected rejection reasons are defined."""
        self.assertTrue(hasattr(RejectionReason, 'BLOCKER'))
        self.assertTrue(hasattr(RejectionReason, 'SCOPE_CREEP'))
        self.assertTrue(hasattr(RejectionReason, 'MISSING_DEPENDENCY'))
        self.assertTrue(hasattr(RejectionReason, 'INFEASIBLE'))
        self.assertTrue(hasattr(RejectionReason, 'UNCLEAR_REQUIREMENTS'))


class TestTaskRejection(unittest.TestCase):
    """Test TaskRejection dataclass."""

    def test_create_basic_rejection(self):
        """Test creating a basic rejection."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Test blocker",
            reason_detail="Detailed explanation",
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="API unavailable",
            suggested_alternative="Wait for API",
        )

        self.assertEqual(rejection.task_id, "task:T-test")
        self.assertEqual(rejection.agent_id, "agent-1")
        self.assertEqual(rejection.reason_type, RejectionReason.BLOCKER)
        self.assertEqual(len(rejection.what_attempted), 2)

    def test_rejection_to_dict(self):
        """Test converting rejection to dict."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Scope grew",
            what_attempted=["Analysis"],
            blocking_factor="Too big",
            suggested_alternative="Break down",
            scope_growth_factor=5.0,
        )

        data = rejection.to_dict()

        self.assertEqual(data["task_id"], "task:T-test")
        self.assertEqual(data["reason_type"], "SCOPE_CREEP")
        self.assertEqual(data["scope_growth_factor"], 5.0)

    def test_rejection_from_dict(self):
        """Test reconstructing rejection from dict."""
        original = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Test",
            what_attempted=["Attempt"],
            blocking_factor="Blocker",
            suggested_alternative="Alternative",
        )

        data = original.to_dict()
        restored = TaskRejection.from_dict(data)

        self.assertEqual(restored.task_id, original.task_id)
        self.assertEqual(restored.reason_type, original.reason_type)
        self.assertEqual(restored.agent_id, original.agent_id)


class TestRejectionValidator(unittest.TestCase):
    """Test RejectionValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = RejectionValidator()
        self.task_context = {
            "title": "Test task",
            "scope": "Do something",
            "priority": "medium",
        }

    def test_valid_scope_creep_rejection(self):
        """Test that valid scope creep rejection passes validation."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Scope grew significantly",
            reason_detail="Original scope X, actual scope Y",
            what_attempted=[
                "Analyzed codebase - found 47 files need changes",
                "Reviewed requirements - 23 new constraints discovered",
            ],
            blocking_factor="Cannot complete without major refactoring",
            suggested_alternative="Break into 3 sub-tasks",
            alternative_tasks=[
                {"title": "Task 1", "scope": "Part 1", "dependencies": "none", "estimate": "2h"},
                {"title": "Task 2", "scope": "Part 2", "dependencies": "task1", "estimate": "3h"},
            ],
            task_original_scope="Simple change",
            scope_growth_factor=5.0,
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertTrue(is_valid, f"Should be valid but got issues: {issues}")
        self.assertEqual(len(issues), 0)

    def test_lazy_rejection_fails_validation(self):
        """Test that lazy rejection fails validation."""
        lazy_rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Too hard",
            reason_detail="It's complex",
            what_attempted=["Looked at it", "Tried to understand"],
            blocking_factor="Too confusing",
            suggested_alternative="Ask someone else",
        )

        is_valid, issues = self.validator.validate(lazy_rejection, self.task_context)

        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)

    def test_insufficient_attempts_fails(self):
        """Test that fewer than 2 attempts fails validation."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Blocked",
            reason_detail="Details",
            what_attempted=["Only one attempt"],
            blocking_factor="Some blocker",
            suggested_alternative="Alternative",
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertFalse(is_valid)
        self.assertTrue(any("at least 2" in issue.lower() for issue in issues))

    def test_vague_blocking_factor_fails(self):
        """Test that vague blocking factors fail validation."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Blocked",
            reason_detail="Details",
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="Too complex",  # Vague
            suggested_alternative="Alternative",
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertFalse(is_valid)
        self.assertTrue(any("vague" in issue.lower() for issue in issues))

    def test_blocker_requires_evidence(self):
        """Test that BLOCKER rejections require evidence."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="API unavailable",
            reason_detail="Cannot reach API",
            what_attempted=["Pinged API - no response", "Checked status page - down"],
            blocking_factor="External API is down (503 error)",
            evidence=[],  # No evidence
            suggested_alternative="Wait for API to come back online",
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertFalse(is_valid)
        self.assertTrue(any("evidence" in issue.lower() for issue in issues))

    def test_valid_blocker_with_evidence(self):
        """Test that BLOCKER with proper evidence passes."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="API unavailable",
            reason_detail="Cannot reach external API",
            what_attempted=[
                "Attempted API call - received 503 error",
                "Checked status page - API is down for maintenance",
            ],
            blocking_factor="External API returns 503 Service Unavailable",
            evidence=[
                {
                    "type": "error_log",
                    "data": "503 Service Unavailable",
                    "source": "API call attempt",
                }
            ],
            suggested_alternative="Defer task until API is back online (ETA 2 hours)",
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertTrue(is_valid, f"Should be valid but got: {issues}")

    def test_scope_creep_requires_growth_factor(self):
        """Test that SCOPE_CREEP requires 2x+ growth factor."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.SCOPE_CREEP,
            reason_summary="Scope grew",
            reason_detail="Details",
            what_attempted=["Analyzed scope", "Estimated effort"],
            blocking_factor="Scope too large",
            suggested_alternative="Break down",
            alternative_tasks=[
                {"title": "Task 1", "scope": "S1", "dependencies": "none", "estimate": "1h"},
                {"title": "Task 2", "scope": "S2", "dependencies": "t1", "estimate": "1h"},
            ],
            task_original_scope="Original",
            scope_growth_factor=1.5,  # Less than 2x
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertFalse(is_valid)
        self.assertTrue(any("2x" in issue.lower() or "growth" in issue.lower() for issue in issues))

    def test_infeasible_requires_high_burden_of_proof(self):
        """Test that INFEASIBLE has highest burden of proof."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.INFEASIBLE,
            reason_summary="Task impossible",
            reason_detail="Cannot be done",
            what_attempted=["Attempted approach A", "Attempted approach B"],
            blocking_factor="Logically impossible",
            evidence=[],  # Not enough evidence
            suggested_alternative="Short alternative",  # Too short
        )

        is_valid, issues = self.validator.validate(rejection, self.task_context)

        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 1)  # Should have multiple issues


class TestRejectionDecision(unittest.TestCase):
    """Test RejectionDecision dataclass."""

    def test_create_accept_decision(self):
        """Test creating ACCEPT decision."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="Blocker",
            suggested_alternative="Alternative",
        )

        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT,
            rejection=rejection,
            rationale="Valid blocker",
        )

        self.assertEqual(decision.decision_type, DecisionType.ACCEPT)
        self.assertEqual(decision.rationale, "Valid blocker")

    def test_create_override_decision(self):
        """Test creating OVERRIDE decision."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            what_attempted=["Lazy attempt"],
            blocking_factor="Vague",
            suggested_alternative="Vague alternative",
        )

        decision = RejectionDecision(
            decision_type=DecisionType.OVERRIDE,
            rejection=rejection,
            rationale="Rejection invalid",
            override_message="Please provide concrete evidence",
            reassign_to="agent-1",
        )

        self.assertEqual(decision.decision_type, DecisionType.OVERRIDE)
        self.assertIn("concrete", decision.override_message)

    def test_decision_to_dict(self):
        """Test converting decision to dict."""
        rejection = TaskRejection(
            task_id="task:T-test",
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.SCOPE_CREEP,
            what_attempted=["Attempt"],
            blocking_factor="Blocker",
            suggested_alternative="Alternative",
        )

        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
            rejection=rejection,
            rationale="Valid scope creep",
            created_tasks=["task:T-1", "task:T-2"],
        )

        data = decision.to_dict()

        self.assertEqual(data["decision_type"], "ACCEPT_AND_DECOMPOSE")
        self.assertEqual(len(data["created_tasks"]), 2)


class TestGoTIntegration(unittest.TestCase):
    """Test GoT integration functions."""

    def test_log_rejection_to_got(self):
        """Test logging rejection to thought graph."""
        graph = ThoughtGraph()

        # Create task node first
        task_id = "task:T-test"
        graph.add_node(task_id, NodeType.TASK, "Test task", properties={})

        rejection = TaskRejection(
            task_id=task_id,
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Test blocker",
            reason_detail="Details",
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="API down",
            suggested_alternative="Wait for API",
        )

        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DEFER,
            rejection=rejection,
            rationale="Valid blocker",
            deferred_task=task_id,
        )

        rejection_node_id = log_rejection_to_got(graph, rejection, decision)

        # Verify nodes were created
        self.assertIn(rejection_node_id, graph.nodes)

        # Verify rejection node has correct properties
        rejection_node = graph.nodes[rejection_node_id]
        self.assertEqual(rejection_node.node_type, NodeType.OBSERVATION)
        self.assertEqual(rejection_node.properties["reason_type"], "BLOCKER")
        self.assertEqual(rejection_node.properties["agent_id"], "agent-1")

    def test_log_blocker_creates_blocker_node(self):
        """Test that BLOCKER rejection creates blocker constraint node."""
        graph = ThoughtGraph()

        task_id = "task:T-test"
        graph.add_node(task_id, NodeType.TASK, "Test task", properties={})

        rejection = TaskRejection(
            task_id=task_id,
            handoff_id="handoff:H-test",
            agent_id="agent-1",
            reason_type=RejectionReason.BLOCKER,
            reason_summary="Blocker",
            reason_detail="Details",
            what_attempted=["Attempt 1", "Attempt 2"],
            blocking_factor="API unavailable",
            suggested_alternative="Wait",
        )

        decision = RejectionDecision(
            decision_type=DecisionType.ACCEPT,
            rejection=rejection,
            rationale="Valid blocker",
        )

        log_rejection_to_got(graph, rejection, decision)

        # Find blocker node
        blocker_nodes = [
            node for node in graph.nodes.values()
            if node.id.startswith("blocker:")
        ]

        self.assertEqual(len(blocker_nodes), 1)
        self.assertEqual(blocker_nodes[0].node_type, NodeType.CONSTRAINT)

    def test_analyze_rejection_patterns(self):
        """Test analyzing rejection patterns from graph."""
        graph = ThoughtGraph()

        # Create multiple rejections
        base_time = datetime.now()
        for i in range(5):
            task_id = f"task:T-{i}"
            graph.add_node(task_id, NodeType.TASK, f"Task {i}", properties={})

            rejection = TaskRejection(
                task_id=task_id,
                handoff_id=f"handoff:H-{i}",
                agent_id=f"agent-{i % 2}",
                reason_type=RejectionReason.BLOCKER if i % 2 == 0 else RejectionReason.SCOPE_CREEP,
                reason_summary=f"Rejection {i}",
                reason_detail="Details",
                what_attempted=["Attempt 1", "Attempt 2"],
                blocking_factor="Blocker",
                suggested_alternative="Alternative",
                rejected_at=base_time + timedelta(seconds=i),
            )

            decision = RejectionDecision(
                decision_type=DecisionType.ACCEPT,
                rejection=rejection,
                rationale="Valid",
            )

            log_rejection_to_got(graph, rejection, decision)

        # Analyze patterns
        patterns = analyze_rejection_patterns(graph)

        self.assertEqual(patterns["total_rejections"], 5)
        self.assertIn("BLOCKER", patterns["by_reason"])
        self.assertIn("SCOPE_CREEP", patterns["by_reason"])
        self.assertEqual(patterns["by_reason"]["BLOCKER"], 3)  # i=0,2,4
        self.assertEqual(patterns["by_reason"]["SCOPE_CREEP"], 2)  # i=1,3

    def test_pattern_analysis_empty_graph(self):
        """Test pattern analysis on empty graph."""
        graph = ThoughtGraph()

        patterns = analyze_rejection_patterns(graph)

        self.assertEqual(patterns["total_rejections"], 0)
        self.assertEqual(len(patterns["by_reason"]), 0)
        self.assertEqual(len(patterns["by_agent"]), 0)


if __name__ == "__main__":
    unittest.main()
