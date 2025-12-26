"""
Unit tests for Goal Stack.

Tests cover:
- Goal dataclass and progress tracking
- Monotonic progress enforcement
- GoalStack operations (push, pop, update)
- Goal blocking and unblocking
- Statistics and serialization

Part of Sprint 3: Cortex Abstraction (Woven Mind + PRISM Marriage)
"""

import pytest
from datetime import datetime
from cortical.reasoning.goal_stack import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalStack,
)


# ==============================================================================
# GOAL DATACLASS TESTS
# ==============================================================================


class TestGoal:
    """Tests for Goal dataclass."""

    def test_default_values(self):
        """Default goal should have neutral values."""
        goal = Goal(id="G-test", name="Test Goal")

        assert goal.id == "G-test"
        assert goal.name == "Test Goal"
        assert goal.progress == 0.0
        assert goal.status == GoalStatus.PENDING
        assert goal.priority == GoalPriority.MEDIUM

    def test_update_progress_monotonic(self):
        """Progress can only increase (monotonic)."""
        goal = Goal(id="G-test", name="Test")

        # Increase should work
        assert goal.update_progress(0.3) is True
        assert goal.progress == 0.3

        # Decrease should fail
        assert goal.update_progress(0.2) is False
        assert goal.progress == 0.3  # Unchanged

        # Equal should fail
        assert goal.update_progress(0.3) is False
        assert goal.progress == 0.3

        # Further increase should work
        assert goal.update_progress(0.5) is True
        assert goal.progress == 0.5

    def test_update_progress_clamped(self):
        """Progress should be clamped to [0, 1]."""
        goal = Goal(id="G-test", name="Test")

        goal.update_progress(1.5)
        assert goal.progress == 1.0

        goal2 = Goal(id="G-test2", name="Test2")
        goal2.update_progress(-0.5)
        assert goal2.progress == 0.0  # Unchanged from 0

    def test_completion_on_full_progress(self):
        """Goal should be achieved at 100% progress."""
        goal = Goal(id="G-test", name="Test")

        goal.update_progress(1.0)

        assert goal.status == GoalStatus.ACHIEVED
        assert goal.completed_at is not None

    def test_is_blocked(self):
        """is_blocked should check blocking_goals."""
        goal = Goal(id="G-test", name="Test", blocking_goals={"G-other"})
        assert goal.is_blocked() is True

        goal.blocking_goals.clear()
        assert goal.is_blocked() is False

    def test_is_achievable(self):
        """is_achievable should check status."""
        goal = Goal(id="G-test", name="Test")
        assert goal.is_achievable() is True

        goal.status = GoalStatus.ACHIEVED
        assert goal.is_achievable() is False

        goal.status = GoalStatus.ABANDONED
        assert goal.is_achievable() is False

    def test_progress_velocity(self):
        """Velocity should measure rate of progress."""
        goal = Goal(id="G-test", name="Test")

        # Single update - no velocity yet
        goal.update_progress(0.2)
        assert goal.get_progress_velocity() == 0.0

        # More updates
        goal.update_progress(0.4)
        goal.update_progress(0.6)
        goal.update_progress(0.8)

        velocity = goal.get_progress_velocity()
        assert velocity > 0  # Progress is increasing

    def test_serialization_roundtrip(self):
        """Serialize and deserialize should preserve state."""
        original = Goal(
            id="G-test",
            name="Test Goal",
            target_nodes=frozenset(["a", "b"]),
            priority=GoalPriority.HIGH,
            progress=0.5,
        )

        data = original.to_dict()
        restored = Goal.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.target_nodes == original.target_nodes
        assert restored.priority == original.priority
        assert restored.progress == original.progress


# ==============================================================================
# GOAL STACK TESTS
# ==============================================================================


class TestGoalStack:
    """Tests for GoalStack class."""

    @pytest.fixture
    def stack(self):
        """Create a default goal stack."""
        return GoalStack()

    def test_push_goal(self, stack):
        """push_goal should add goal to stack."""
        goal = stack.push_goal("Test Goal")

        assert goal.id in stack.goals
        assert goal.id in stack.goal_order
        assert goal.name == "Test Goal"

    def test_push_auto_activates(self, stack):
        """push_goal should auto-activate if not blocked."""
        goal = stack.push_goal("Test Goal")

        assert goal.status == GoalStatus.ACTIVE
        assert goal.id in stack.active_goals

    def test_push_blocked_goal(self, stack):
        """Blocked goals should not auto-activate."""
        blocker = stack.push_goal("Blocker")
        blocked = stack.push_goal("Blocked", blocking_goals={blocker.id})

        assert blocked.status != GoalStatus.ACTIVE
        assert blocked.id not in stack.active_goals

    def test_pop_goal(self, stack):
        """pop_goal should remove top goal."""
        goal1 = stack.push_goal("First")
        goal2 = stack.push_goal("Second")

        popped = stack.pop_goal()

        assert popped.id == goal2.id  # LIFO
        assert goal2.id not in stack.goals
        assert goal1.id in stack.goals

    def test_pop_empty_stack(self, stack):
        """pop_goal on empty stack returns None."""
        assert stack.pop_goal() is None

    def test_update_progress(self, stack):
        """update_progress should update goal progress."""
        goal = stack.push_goal("Test")

        result = stack.update_progress(goal.id, 0.5)

        assert result is True
        assert stack.get_progress(goal.id) == 0.5

    def test_update_progress_monotonic(self, stack):
        """update_progress enforces monotonicity."""
        goal = stack.push_goal("Test")

        stack.update_progress(goal.id, 0.5)
        result = stack.update_progress(goal.id, 0.3)  # Regression

        assert result is False
        assert stack.get_progress(goal.id) == 0.5

    def test_update_progress_force(self, stack):
        """force=True allows non-monotonic updates."""
        goal = stack.push_goal("Test")

        stack.update_progress(goal.id, 0.5)
        stack.update_progress(goal.id, 0.3, force=True)

        assert stack.get_progress(goal.id) == 0.3

    def test_update_progress_unknown_goal(self, stack):
        """update_progress on unknown goal returns False."""
        result = stack.update_progress("unknown-id", 0.5)
        assert result is False

    def test_completion_unblocks_dependents(self, stack):
        """Completing a goal should unblock dependent goals."""
        blocker = stack.push_goal("Blocker")
        dependent = stack.push_goal("Dependent", blocking_goals={blocker.id})

        assert dependent.is_blocked()

        # Complete the blocker
        stack.update_progress(blocker.id, 1.0)

        assert not dependent.is_blocked()

    def test_get_active_goals(self, stack):
        """get_active_goals returns currently active goals."""
        goal1 = stack.push_goal("Goal 1")
        goal2 = stack.push_goal("Goal 2")

        active = stack.get_active_goals()

        assert len(active) == 2
        assert goal1 in active
        assert goal2 in active

    def test_get_goals_by_status(self, stack):
        """get_goals_by_status filters correctly."""
        goal1 = stack.push_goal("Active")
        goal2 = stack.push_goal("To Complete")
        stack.update_progress(goal2.id, 1.0)

        active = stack.get_goals_by_status(GoalStatus.ACTIVE)
        achieved = stack.get_goals_by_status(GoalStatus.ACHIEVED)

        assert len(active) == 1
        assert len(achieved) == 1
        assert goal1 in active
        assert goal2 in achieved

    def test_get_goals_by_priority(self, stack):
        """get_goals_by_priority sorts by priority."""
        low = stack.push_goal("Low", priority=GoalPriority.LOW)
        high = stack.push_goal("High", priority=GoalPriority.HIGH)
        medium = stack.push_goal("Medium", priority=GoalPriority.MEDIUM)

        goals = stack.get_goals_by_priority(GoalPriority.LOW)

        assert goals[0] == high  # Highest priority first
        assert goals[1] == medium
        assert goals[2] == low

    def test_abandon_goal(self, stack):
        """abandon_goal should mark as abandoned."""
        goal = stack.push_goal("To Abandon")

        result = stack.abandon_goal(goal.id, reason="Testing")

        assert result is True
        assert goal.status == GoalStatus.ABANDONED
        assert goal.id not in stack.active_goals
        assert goal.metadata["abandon_reason"] == "Testing"

    def test_check_achievement(self, stack):
        """check_achievement should update progress based on nodes."""
        goal = stack.push_goal(
            "Learn Concept",
            target_nodes={"neural", "network", "deep"},
        )

        # Partial achievement
        stack.check_achievement(goal.id, frozenset(["neural", "network"]))
        assert stack.get_progress(goal.id) == pytest.approx(2/3)

        # Full achievement
        stack.check_achievement(goal.id, frozenset(["neural", "network", "deep"]))
        assert stack.get_progress(goal.id) == 1.0
        assert goal.status == GoalStatus.ACHIEVED

    def test_get_subgoals(self, stack):
        """get_subgoals should return child goals."""
        parent = stack.push_goal("Parent")
        child1 = stack.push_goal("Child 1", parent_id=parent.id)
        child2 = stack.push_goal("Child 2", parent_id=parent.id)
        other = stack.push_goal("Other")

        subgoals = stack.get_subgoals(parent.id)

        assert len(subgoals) == 2
        assert child1 in subgoals
        assert child2 in subgoals
        assert other not in subgoals

    def test_get_blocked_goals(self, stack):
        """get_blocked_goals should return blocked goals."""
        blocker = stack.push_goal("Blocker")
        blocked1 = stack.push_goal("Blocked 1", blocking_goals={blocker.id})
        blocked2 = stack.push_goal("Blocked 2", blocking_goals={blocker.id})
        unblocked = stack.push_goal("Unblocked")

        blocked = stack.get_blocked_goals()

        assert len(blocked) == 2
        assert blocked1 in blocked
        assert blocked2 in blocked
        assert unblocked not in blocked

    def test_get_statistics(self, stack):
        """get_statistics should return accurate stats."""
        goal1 = stack.push_goal("Active")
        goal2 = stack.push_goal("To Complete")
        goal3 = stack.push_goal("To Abandon")

        stack.update_progress(goal2.id, 1.0)
        stack.abandon_goal(goal3.id)

        stats = stack.get_statistics()

        assert stats["total_goals"] == 3
        assert stats["achieved_count"] == 1
        assert stats["abandoned_count"] == 1

    def test_serialization_roundtrip(self, stack):
        """Serialize and deserialize should preserve state."""
        stack.push_goal("Goal 1", priority=GoalPriority.HIGH)
        stack.push_goal("Goal 2", target_nodes={"a", "b"})

        data = stack.to_dict()
        restored = GoalStack.from_dict(data)

        assert len(restored.goals) == len(stack.goals)
        assert restored.max_active_goals == stack.max_active_goals

    def test_clear(self, stack):
        """clear should remove all goals."""
        stack.push_goal("Goal 1")
        stack.push_goal("Goal 2")

        stack.clear()

        assert len(stack.goals) == 0
        assert len(stack.active_goals) == 0
        assert len(stack.goal_order) == 0

    def test_max_active_goals(self):
        """Should respect max_active_goals limit."""
        stack = GoalStack(max_active_goals=2)

        stack.push_goal("Goal 1")
        stack.push_goal("Goal 2")
        stack.push_goal("Goal 3")

        assert len(stack.active_goals) == 2


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestGoalStackIntegration:
    """Integration tests for goal stack."""

    def test_complex_dependency_chain(self):
        """Test chain of dependent goals."""
        stack = GoalStack()

        # Create dependency chain: A <- B <- C
        goal_a = stack.push_goal("A")
        goal_b = stack.push_goal("B", blocking_goals={goal_a.id})
        goal_c = stack.push_goal("C", blocking_goals={goal_b.id})

        # Initially only A is active
        assert goal_a.status == GoalStatus.ACTIVE
        assert goal_b.status != GoalStatus.ACTIVE
        assert goal_c.status != GoalStatus.ACTIVE

        # Complete A -> B becomes active
        stack.update_progress(goal_a.id, 1.0)
        assert goal_b.status == GoalStatus.ACTIVE
        assert goal_c.status != GoalStatus.ACTIVE

        # Complete B -> C becomes active
        stack.update_progress(goal_b.id, 1.0)
        assert goal_c.status == GoalStatus.ACTIVE

    def test_progress_tracking_workflow(self):
        """Test realistic progress tracking scenario."""
        stack = GoalStack()

        # Create goal to learn a concept
        goal = stack.push_goal(
            "Learn Neural Networks",
            target_nodes={"neural", "network", "backprop", "gradient"},
        )

        # Partial learning
        active_1 = frozenset(["neural"])
        stack.check_achievement(goal.id, active_1)
        assert goal.progress == pytest.approx(0.25)

        # More learning (monotonic increase)
        active_2 = frozenset(["neural", "network"])
        stack.check_achievement(goal.id, active_2)
        assert goal.progress == pytest.approx(0.5)

        # Regression attempt (should not decrease)
        active_3 = frozenset(["neural"])  # Less than before
        stack.check_achievement(goal.id, active_3)
        assert goal.progress == pytest.approx(0.5)  # Unchanged

        # Full learning
        active_4 = frozenset(["neural", "network", "backprop", "gradient"])
        stack.check_achievement(goal.id, active_4)
        assert goal.progress == 1.0
        assert goal.status == GoalStatus.ACHIEVED

    def test_hierarchical_goal_tracking(self):
        """Test parent-child goal relationships."""
        stack = GoalStack()

        # Parent goal
        parent = stack.push_goal("Master Machine Learning")

        # Child goals
        child1 = stack.push_goal("Learn Neural Networks", parent_id=parent.id)
        child2 = stack.push_goal("Learn Decision Trees", parent_id=parent.id)
        child3 = stack.push_goal("Learn Reinforcement Learning", parent_id=parent.id)

        # Verify hierarchy
        subgoals = stack.get_subgoals(parent.id)
        assert len(subgoals) == 3
        assert child1 in subgoals
        assert child2 in subgoals
        assert child3 in subgoals
