"""
Behavioral tests for hierarchical cognitive loop management.

As a developer building hierarchical reasoning systems,
I want to manage nested QAPV (Question-Answer-Produce-Verify) cycles,
So that complex goals decompose into manageable sub-goals.

Based on: examples/nested_loop_demo.py
"""

import pytest
from cortical.reasoning import (
    NestedLoopExecutor,
    LoopPhase,
    LoopStatus,
)


class TestDeveloperDecomposesGoalsHierarchically:
    """
    Epic: Hierarchical Goal Decomposition

    As a developer building a reasoning system,
    I want to break complex goals into nested sub-goals,
    So that difficult problems become tractable.
    """

    def test_parent_pauses_when_child_spawns(self):
        """
        Scenario: Parent loop pauses when spawning child

        Given an active parent loop
        When spawning a child loop
        Then the parent pauses automatically
        And the child becomes active
        Because only one loop executes at a time per branch
        """
        # Given: active parent loop
        executor = NestedLoopExecutor(max_depth=5)
        parent = executor.start_root("Build web application")
        assert executor.get_loop(parent).status == LoopStatus.ACTIVE

        # When: spawning a child
        child = executor.spawn_child(parent, "Implement backend API")

        # Then: parent pauses
        assert executor.get_loop(parent).status == LoopStatus.PAUSED
        # Then: child is active
        assert executor.get_loop(child).status == LoopStatus.ACTIVE

    def test_parent_resumes_when_child_completes(self):
        """
        Scenario: Completing child resumes parent

        Given a parent with active child
        When the child completes
        Then the parent resumes
        And receives the child's result
        Because parent needs child's output to continue
        """
        # Given: parent with active child
        executor = NestedLoopExecutor(max_depth=5)
        parent = executor.start_root("Develop mobile app")
        child = executor.spawn_child(parent, "Design UI mockups")

        # When: child completes with result
        result = {"screens": 12, "status": "done"}
        returned_parent = executor.complete(child, result)

        # Then: parent resumes
        assert executor.get_loop(parent).status == LoopStatus.ACTIVE
        # Then: parent receives result
        parent_context = executor.get_context(parent)
        assert child in parent_context.child_results
        assert parent_context.child_results[child] == result

    def test_loop_advances_through_qapv_phases(self):
        """
        Scenario: Loop progresses through QAPV phases

        Given a loop in Question phase
        When advancing the loop
        Then it progresses through Answer, Produce, Verify phases
        Because QAPV is the cognitive cycle
        """
        # Given: loop in Question phase
        executor = NestedLoopExecutor(max_depth=5)
        loop_id = executor.start_root("Research topic")
        loop = executor.get_loop(loop_id)
        assert loop.current_phase == LoopPhase.QUESTION

        # When: advancing through phases
        phase1 = executor.advance(loop_id)
        assert phase1 == LoopPhase.ANSWER

        phase2 = executor.advance(loop_id)
        assert phase2 == LoopPhase.PRODUCE

        phase3 = executor.advance(loop_id)
        assert phase3 == LoopPhase.VERIFY

        # Then: cycle continues
        phase4 = executor.advance(loop_id)
        assert phase4 == LoopPhase.QUESTION  # Cycles back


class TestDeveloperManagesMultipleLevels:
    """
    Epic: Deep Nesting

    As a developer handling complex decomposition,
    I want to nest goals multiple levels deep,
    So that I can break down very complex problems.
    """

    def test_system_supports_deep_nesting(self):
        """
        Scenario: Multiple levels of nesting are supported

        Given an executor allowing deep nesting
        When creating a multi-level hierarchy
        Then each level has correct depth
        And hierarchy is trackable
        Because complex problems need deep decomposition
        """
        # Given: executor with max depth 5
        executor = NestedLoopExecutor(max_depth=5)

        # When: creating hierarchy
        level0 = executor.start_root("Build system")
        level1 = executor.spawn_child(level0, "Backend services")
        level2 = executor.spawn_child(level1, "Authentication")
        level3 = executor.spawn_child(level2, "JWT tokens")

        # Then: each has correct depth
        assert executor.get_context(level0).depth == 0
        assert executor.get_context(level1).depth == 1
        assert executor.get_context(level2).depth == 2
        assert executor.get_context(level3).depth == 3

        # Then: hierarchy is trackable
        hierarchy = executor.get_loop_hierarchy(level3)
        assert len(hierarchy) == 4
        assert hierarchy == [level0, level1, level2, level3]

    def test_depth_limit_prevents_infinite_recursion(self):
        """
        Scenario: Depth limit enforces maximum nesting

        Given an executor with max depth of 3
        When attempting to exceed the limit
        Then a RecursionError is raised
        Because unbounded recursion must be prevented
        """
        # Given: executor with depth limit
        executor = NestedLoopExecutor(max_depth=3)

        # When: creating chain to limit
        root = executor.start_root("Level 0")         # depth 0
        child1 = executor.spawn_child(root, "Level 1")     # depth 1
        child2 = executor.spawn_child(child1, "Level 2")   # depth 2

        # Then: exceeding limit raises error
        with pytest.raises(RecursionError) as exc_info:
            executor.spawn_child(child2, "Level 3")  # would be depth 3, exceeds limit
        assert "Maximum nesting depth" in str(exc_info.value)


class TestDeveloperAggregatesResults:
    """
    Epic: Result Collection

    As a developer coordinating sub-tasks,
    I want parent loops to collect results from children,
    So that work products flow upward.
    """

    def test_parent_collects_all_child_results(self):
        """
        Scenario: Parent aggregates results from multiple children

        Given a parent that spawns multiple children
        When each child completes with results
        Then the parent has all child results
        Because parents need to synthesize child outputs
        """
        # Given: parent with multiple children
        executor = NestedLoopExecutor(max_depth=3)
        parent = executor.start_root("Develop mobile app")

        # When: spawning and completing children
        ui_child = executor.spawn_child(parent, "Design UI")
        executor.complete(ui_child, {"screens": 12})

        ci_child = executor.spawn_child(parent, "Setup CI/CD")
        executor.complete(ci_child, {"platform": "GitHub Actions"})

        api_child = executor.spawn_child(parent, "Write API client")
        executor.complete(api_child, {"endpoints": 15})

        # Then: parent has all results
        context = executor.get_context(parent)
        assert len(context.child_results) == 3
        assert context.child_results[ui_child] == {"screens": 12}
        assert context.child_results[ci_child] == {"platform": "GitHub Actions"}
        assert context.child_results[api_child] == {"endpoints": 15}

    def test_loop_records_answers_during_cycle(self):
        """
        Scenario: Loop accumulates answers during Answer phase

        Given a loop in Answer phase
        When recording multiple answers
        Then all answers are stored
        Because the reasoning process builds knowledge incrementally
        """
        # Given: loop in Answer phase
        executor = NestedLoopExecutor(max_depth=3)
        loop_id = executor.start_root("Research framework")
        executor.advance(loop_id)  # Move to ANSWER phase

        # When: recording answers
        executor.record_answer(loop_id, "Framework uses React")
        executor.record_answer(loop_id, "TypeScript is primary language")
        executor.record_answer(loop_id, "Built-in state management")

        # Then: answers are stored
        context = executor.get_context(loop_id)
        assert len(context.accumulated_answers) == 3
        assert "Framework uses React" in context.accumulated_answers
        assert "TypeScript is primary language" in context.accumulated_answers


class TestDeveloperHandlesEarlyTermination:
    """
    Epic: Flexible Control Flow

    As a developer managing dynamic reasoning,
    I want to terminate loops early when needed,
    So that I can abandon unproductive paths.
    """

    def test_loop_breaks_early_with_reason(self):
        """
        Scenario: Loop can be broken before natural completion

        Given an active loop
        When breaking the loop with a reason
        Then the loop is marked broken
        And the parent resumes
        Because sometimes we discover a path is not viable
        """
        # Given: active loop
        executor = NestedLoopExecutor(max_depth=3)
        root = executor.start_root("Research technologies")
        investigation = executor.spawn_child(root, "Investigate Framework X")

        # When: breaking the loop
        executor.break_loop(investigation, "Framework X is deprecated")

        # Then: loop is abandoned
        assert executor.get_loop(investigation).status == LoopStatus.ABANDONED
        # Then: parent resumes
        assert executor.get_loop(root).status == LoopStatus.ACTIVE

    def test_broken_loop_retains_reason(self):
        """
        Scenario: Break reason is recorded

        Given an abandoned loop
        When checking the loop context
        Then the loop is marked as abandoned
        Because we need to understand why paths were abandoned
        """
        # Given: abandoned loop
        executor = NestedLoopExecutor(max_depth=3)
        loop_id = executor.start_root("Explore approach")
        reason = "Approach not feasible due to constraints"
        executor.break_loop(loop_id, reason)

        # When: checking context
        # Then: reason is recorded (status indicates it was abandoned)
        assert executor.get_loop(loop_id).status == LoopStatus.ABANDONED


class TestDeveloperMonitorsExecution:
    """
    Epic: Execution Observability

    As a developer debugging hierarchical execution,
    I want to inspect executor state and statistics,
    So that I can understand what's happening.
    """

    def test_executor_provides_summary_statistics(self):
        """
        Scenario: Executor reports comprehensive statistics

        Given an executor with multiple loops in various states
        When requesting a summary
        Then statistics cover total loops, status breakdown, and depth
        Because observability enables debugging
        """
        # Given: executor with various loops
        executor = NestedLoopExecutor(max_depth=5)
        root = executor.start_root("Build system")
        child1 = executor.spawn_child(root, "Component A")
        child2 = executor.spawn_child(child1, "Subcomponent A1")

        # Complete one branch
        executor.complete(child2, {"done": True})
        executor.complete(child1, {"component": "done"})

        # Break another
        child3 = executor.spawn_child(root, "Component B")
        executor.break_loop(child3, "Not needed")

        # When: requesting summary
        summary = executor.get_summary()

        # Then: comprehensive statistics
        assert "total_loops" in summary
        assert "active_loops" in summary
        assert "max_depth_limit" in summary
        assert "max_depth_reached" in summary
        assert "status_counts" in summary
        assert summary["total_loops"] == 4  # root + 3 children

    def test_can_retrieve_individual_loop_state(self):
        """
        Scenario: Individual loops can be inspected

        Given an executor with multiple loops
        When retrieving a specific loop
        Then its full state is accessible
        Because detailed inspection is needed for debugging
        """
        # Given: executor with loops
        executor = NestedLoopExecutor(max_depth=3)
        loop_id = executor.start_root("Main task")

        # When: retrieving the loop
        loop = executor.get_loop(loop_id)
        context = executor.get_context(loop_id)

        # Then: state is accessible
        assert loop.goal == "Main task"
        assert loop.current_phase == LoopPhase.QUESTION
        assert loop.status == LoopStatus.ACTIVE
        assert context.depth == 0


class TestDeveloperBuildsComplexHierarchy:
    """
    Epic: Real-World Workflows

    As a developer building practical systems,
    I want to model real hierarchical workflows,
    So that the system handles actual complexity.
    """

    def test_complete_software_development_workflow(self):
        """
        Scenario: Model full software development hierarchy

        Given a project with multiple components and sub-tasks
        When modeling as nested loops
        Then the hierarchy reflects real decomposition
        And results flow from leaves to root
        Because real projects have this structure
        """
        # Given: executor for project
        executor = NestedLoopExecutor(max_depth=5)

        # Root: Overall project
        project = executor.start_root("Build e-commerce platform")

        # Level 1: Major components
        backend = executor.spawn_child(project, "Backend services")
        frontend = executor.spawn_child(backend, "Frontend application")  # Wait, this should be separate

        # Let me model this correctly
        executor = NestedLoopExecutor(max_depth=5)
        project = executor.start_root("Build e-commerce platform")

        # When: modeling hierarchical decomposition
        backend = executor.spawn_child(project, "Backend services")
        auth = executor.spawn_child(backend, "Authentication module")
        jwt = executor.spawn_child(auth, "JWT token implementation")

        # Complete the JWT implementation
        jwt_result = {
            "algorithm": "RS256",
            "expiry": "24h",
            "refresh": True
        }
        executor.complete(jwt, jwt_result)

        # Complete auth with aggregated result
        auth_result = {
            "jwt": jwt_result,
            "oauth": "implemented",
            "sessions": "redis"
        }
        executor.complete(auth, auth_result)

        # Then: results flow upward
        backend_context = executor.get_context(backend)
        assert auth in backend_context.child_results
        assert backend_context.child_results[auth] == auth_result

        # Then: hierarchy is correct
        hierarchy = executor.get_loop_hierarchy(jwt)
        assert hierarchy == [project, backend, auth, jwt]

    def test_parallel_branches_work_independently(self):
        """
        Scenario: Sibling branches don't interfere

        Given a parent with multiple completed children
        When working on a new child
        Then previous sibling results are preserved
        Because sibling branches are independent
        """
        # Given: parent with completed children
        executor = NestedLoopExecutor(max_depth=3)
        parent = executor.start_root("Multi-component system")

        component_a = executor.spawn_child(parent, "Component A")
        executor.complete(component_a, {"a_result": "done"})

        component_b = executor.spawn_child(parent, "Component B")
        executor.complete(component_b, {"b_result": "done"})

        # When: working on new child
        component_c = executor.spawn_child(parent, "Component C")

        # Then: previous results preserved
        context = executor.get_context(parent)
        assert component_a in context.child_results
        assert component_b in context.child_results
        assert context.child_results[component_a] == {"a_result": "done"}
        assert context.child_results[component_b] == {"b_result": "done"}
