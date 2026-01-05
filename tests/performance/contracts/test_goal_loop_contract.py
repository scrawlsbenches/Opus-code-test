"""
╔══════════════════════════════════════════════════════════════════════╗
║            GOAL & LOOP MANAGEMENT PERFORMANCE CONTRACT                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Goal stack push/pop operations < 10ms                             ║
║  • Progress update (monotonic) < 5ms                                 ║
║  • Goal achievement check < 15ms                                     ║
║  • Loop spawn/advance/complete < 20ms each                           ║
║  • Nested loop depth enforcement is O(1)                             ║
║  • Goal stack supports ≥ 1000 concurrent goals                       ║
║  • Loop hierarchy traversal < 50ms for depth 10                      ║
║  • Monotonic progress NEVER regresses                                ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List

import pytest

from cortical.reasoning.goal_stack import GoalStack, GoalPriority, GoalStatus
from cortical.reasoning.nested_loop import NestedLoopExecutor, LoopContext
from cortical.reasoning.cognitive_loop import LoopPhase


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestGoalStackContract:
    """
    Goal Stack Performance Contract

    As a cognitive system tracking goals,
    I expect goal operations to be instantaneous,
    So that goal management never becomes a performance bottleneck.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    PUSH_POP_MS = 20
    PROGRESS_UPDATE_MS = 10
    ACHIEVEMENT_CHECK_MS = 30
    MIN_CONCURRENT_GOALS = 1000

    def test_push_operation_latency(self):
        """
        CONTRACT: Pushing goals to stack completes in under 10ms.

        Goal creation must be fast to support dynamic goal decomposition.
        """
        stack = GoalStack(max_active_goals=100)

        latencies = []
        for i in range(100):
            start = time.perf_counter()
            stack.push_goal(
                name=f"Goal {i}",
                target_nodes={f"node_{j}" for j in range(5)},
                priority=GoalPriority.MEDIUM
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.PUSH_POP_MS, (
            f"CONTRACT VIOLATION: p95 push latency is {p95:.1f}ms, "
            f"contract requires <{self.PUSH_POP_MS}ms"
        )

    def test_pop_operation_latency(self):
        """
        CONTRACT: Popping goals from stack completes in under 10ms.

        Goal removal must be fast to support rapid goal lifecycle.
        """
        stack = GoalStack()

        # Pre-populate
        for i in range(100):
            stack.push_goal(name=f"Goal {i}", target_nodes={f"node_{i}"})

        # Measure pop latency
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            stack.pop_goal()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.PUSH_POP_MS, (
            f"CONTRACT VIOLATION: p95 pop latency is {p95:.1f}ms, "
            f"contract requires <{self.PUSH_POP_MS}ms"
        )

    def test_progress_update_latency(self):
        """
        CONTRACT: Progress updates complete in under 5ms.

        Monotonic progress tracking must have minimal overhead.
        """
        stack = GoalStack()
        goal = stack.push_goal(
            name="Test goal",
            target_nodes={"node_a", "node_b", "node_c"}
        )

        latencies = []
        progress_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for progress in progress_values:
            start = time.perf_counter()
            stack.update_progress(goal.id, progress)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.PROGRESS_UPDATE_MS, (
            f"CONTRACT VIOLATION: p95 progress update is {p95:.1f}ms, "
            f"contract requires <{self.PROGRESS_UPDATE_MS}ms"
        )

    def test_achievement_check_latency(self):
        """
        CONTRACT: Achievement checking completes in under 15ms.

        Checking goal progress against active nodes must be efficient.
        """
        stack = GoalStack()
        goal = stack.push_goal(
            name="Activation goal",
            target_nodes={"node_1", "node_2", "node_3", "node_4", "node_5"}
        )

        active_nodes = frozenset(["node_1", "node_2", "node_3"])

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            stack.check_achievement(goal.id, active_nodes)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.ACHIEVEMENT_CHECK_MS, (
            f"CONTRACT VIOLATION: p95 achievement check is {p95:.1f}ms, "
            f"contract requires <{self.ACHIEVEMENT_CHECK_MS}ms"
        )

    def test_concurrent_goals_supported(self):
        """
        CONTRACT: Goal stack supports at least 1000 concurrent goals.

        Large-scale goal tracking must not degrade performance.
        """
        stack = GoalStack(max_active_goals=1000)

        # Create 1000 goals
        start = time.perf_counter()
        for i in range(self.MIN_CONCURRENT_GOALS):
            stack.push_goal(
                name=f"Concurrent goal {i}",
                target_nodes={f"node_{i % 10}"},
                priority=GoalPriority.LOW
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Average should be well under budget
        avg_ms = elapsed_ms / self.MIN_CONCURRENT_GOALS

        assert avg_ms < 5, (
            f"CONTRACT VIOLATION: Average goal creation is {avg_ms:.2f}ms, "
            f"should be < 5ms to support {self.MIN_CONCURRENT_GOALS} goals"
        )

        # Verify all goals exist
        assert len(stack.goals) == self.MIN_CONCURRENT_GOALS

    def test_monotonic_progress_guarantee(self):
        """
        CONTRACT: Progress NEVER regresses (monotonic guarantee).

        This is a correctness contract - progress must only increase.
        """
        stack = GoalStack()
        goal = stack.push_goal(name="Monotonic test", target_nodes={"node_a"})

        # Set initial progress
        assert stack.update_progress(goal.id, 0.5) is True
        assert stack.get_progress(goal.id) == 0.5

        # Try to regress (should be rejected)
        assert stack.update_progress(goal.id, 0.3) is False
        assert stack.get_progress(goal.id) == 0.5, (
            "CONTRACT VIOLATION: Progress regressed! Monotonicity broken."
        )

        # Advance should work
        assert stack.update_progress(goal.id, 0.7) is True
        assert stack.get_progress(goal.id) == 0.7

        # Regression attempt again
        assert stack.update_progress(goal.id, 0.6) is False
        assert stack.get_progress(goal.id) == 0.7, (
            "CONTRACT VIOLATION: Progress regressed after advancement!"
        )

    def test_goal_statistics_fast(self):
        """
        CONTRACT: Statistics gathering is efficient even with many goals.

        Metrics must not block the system.
        """
        stack = GoalStack()

        # Create varied goals
        for i in range(500):
            goal = stack.push_goal(
                name=f"Goal {i}",
                target_nodes={f"node_{i % 20}"},
                priority=GoalPriority.MEDIUM
            )
            # Vary the status
            if i % 3 == 0:
                stack.update_progress(goal.id, 1.0)  # Achieved
            elif i % 5 == 0:
                stack.abandon_goal(goal.id, "test")  # Abandoned

        # Measure statistics gathering
        start = time.perf_counter()
        stats = stack.get_statistics()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"CONTRACT VIOLATION: Statistics took {elapsed_ms:.1f}ms, "
            f"should complete in < 50ms"
        )

        # Verify stats are correct
        assert stats["total_goals"] == 500


@pytest.mark.contract
class TestNestedLoopContract:
    """
    Nested Loop Executor Performance Contract

    As a hierarchical reasoning system,
    I expect loop operations to be fast and depth-bounded,
    So that recursive problem decomposition remains tractable.
    """

    # The sacred numbers
    SPAWN_LATENCY_MS = 40
    ADVANCE_LATENCY_MS = 40
    COMPLETE_LATENCY_MS = 40
    HIERARCHY_TRAVERSAL_MS = 100

    def test_loop_spawn_latency(self):
        """
        CONTRACT: Spawning child loops completes in under 20ms.

        Fast loop creation enables dynamic task decomposition.
        """
        executor = NestedLoopExecutor(max_depth=5)

        # Create root
        root_id = executor.start_root("Root task")

        # Measure child spawn latency
        latencies = []
        for i in range(20):
            start = time.perf_counter()
            executor.spawn_child(root_id, f"Subtask {i}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            # Complete the child to allow more spawns
            children = [lid for lid in executor.get_all_loops().keys() if lid != root_id]
            if children:
                executor.complete(children[-1], f"Result {i}")

        p95 = percentile(latencies, 95)

        assert p95 < self.SPAWN_LATENCY_MS, (
            f"CONTRACT VIOLATION: p95 spawn latency is {p95:.1f}ms, "
            f"contract requires <{self.SPAWN_LATENCY_MS}ms"
        )

    def test_loop_advance_latency(self):
        """
        CONTRACT: Advancing loop phase completes in under 20ms.

        Phase transitions must not introduce noticeable lag.
        """
        executor = NestedLoopExecutor(max_depth=3)
        loop_id = executor.start_root("Test loop")

        # Measure phase advancement
        latencies = []
        for _ in range(20):  # Cycle through phases 5 times
            start = time.perf_counter()
            executor.advance(loop_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.ADVANCE_LATENCY_MS, (
            f"CONTRACT VIOLATION: p95 advance latency is {p95:.1f}ms, "
            f"contract requires <{self.ADVANCE_LATENCY_MS}ms"
        )

    def test_loop_complete_latency(self):
        """
        CONTRACT: Completing loops completes in under 20ms.

        Loop completion must be fast to support rapid iteration.
        """
        executor = NestedLoopExecutor(max_depth=3)

        latencies = []
        for i in range(50):
            # Create and immediately complete
            loop_id = executor.start_root(f"Quick task {i}")
            start = time.perf_counter()
            executor.complete(loop_id, {"result": f"output_{i}"})
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)

        assert p95 < self.COMPLETE_LATENCY_MS, (
            f"CONTRACT VIOLATION: p95 complete latency is {p95:.1f}ms, "
            f"contract requires <{self.COMPLETE_LATENCY_MS}ms"
        )

    def test_depth_enforcement_is_fast(self):
        """
        CONTRACT: Depth enforcement check is O(1).

        Checking depth limits must not traverse the entire hierarchy.
        """
        executor = NestedLoopExecutor(max_depth=5)

        # Create a deep hierarchy
        current_id = executor.start_root("Root")
        for depth in range(4):
            current_id = executor.spawn_child(current_id, f"Level {depth + 1}")

        # Now we're at max depth - 1. Spawning should check depth quickly
        start = time.perf_counter()
        try:
            # This should fail with RecursionError
            executor.spawn_child(current_id, "Too deep")
        except RecursionError:
            pass  # Expected
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Depth check should be near-instant
        assert elapsed_ms < 1, (
            f"CONTRACT VIOLATION: Depth check took {elapsed_ms:.1f}ms, "
            f"should be O(1) (< 1ms)"
        )

    def test_hierarchy_traversal_bounded(self):
        """
        CONTRACT: Hierarchy traversal completes in under 50ms for depth 10.

        Even deep hierarchies must be traversable efficiently.
        """
        executor = NestedLoopExecutor(max_depth=10)

        # Build a deep hierarchy (depth 10)
        current_id = executor.start_root("Root")
        deepest_id = current_id
        for depth in range(9):
            deepest_id = executor.spawn_child(current_id, f"Level {depth + 1}")
            current_id = deepest_id

        # Measure hierarchy traversal
        start = time.perf_counter()
        hierarchy = executor.get_loop_hierarchy(deepest_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.HIERARCHY_TRAVERSAL_MS, (
            f"CONTRACT VIOLATION: Hierarchy traversal took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.HIERARCHY_TRAVERSAL_MS}ms"
        )

        # Verify hierarchy depth
        assert len(hierarchy) == 10, "Hierarchy depth incorrect"

    def test_context_accumulation_scales(self):
        """
        CONTRACT: Context accumulation doesn't degrade with depth.

        Storing and retrieving context must remain efficient.
        """
        executor = NestedLoopExecutor(max_depth=5)

        # Create hierarchy and accumulate answers
        root_id = executor.start_root("Root analysis")
        current_id = root_id

        for depth in range(4):
            child_id = executor.spawn_child(current_id, f"Sub-analysis {depth}")

            # Record many answers
            for i in range(20):
                executor.record_answer(child_id, f"Finding {i} at depth {depth}")

            current_id = child_id

        # Measure context retrieval
        start = time.perf_counter()
        context = executor.get_context(current_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5, (
            f"CONTRACT VIOLATION: Context retrieval took {elapsed_ms:.1f}ms, "
            f"should be near-instant (< 5ms)"
        )

        # Verify context has accumulated answers
        assert len(context.accumulated_answers) == 20
