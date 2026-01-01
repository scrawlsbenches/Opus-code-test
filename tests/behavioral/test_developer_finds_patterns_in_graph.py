"""
Behavioral tests for pattern matching in Graph of Thought.

Epic: Developer Finds Structural Patterns

As a developer analyzing complex dependency structures,
I want to find specific graph patterns using our custom pattern matcher,
So that I can detect anti-patterns and analyze relationships
in a system built entirely from first principles.
"""

import pytest
from cortical.got.api import GoTManager
from cortical.got.pattern_matcher import Pattern, PatternMatcher


class TestDeveloperFindsDependencyChains:
    """
    As a developer analyzing task dependencies,
    I want to find chains of dependent tasks,
    So that I can understand critical paths using our custom pattern engine.
    """

    def test_scenario_find_two_node_dependency_chain(self, tmp_path):
        """
        Scenario: Finding simple A-depends-on-B patterns

        Given tasks with dependency relationships
        When I search for the pattern "A depends on B"
        Then I find all matching pairs
        """
        # Given tasks with dependency relationships
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(title="Build custom parser")
        task_b = manager.create_task(title="Design custom grammar")
        task_c = manager.create_task(title="Implement tokenizer")
        task_d = manager.create_task(title="Unrelated task")

        manager.add_dependency(task_a.id, task_b.id)
        manager.add_dependency(task_c.id, task_b.id)

        # When I search for the pattern "A depends on B"
        pattern = (
            Pattern()
            .node("a", type="task")
            .outgoing("DEPENDS_ON")
            .node("b", type="task")
        )
        matcher = PatternMatcher(manager)
        result = matcher.find(pattern)

        # Then I find all matching pairs
        assert len(result) == 2
        # Should find task_a -> task_b and task_c -> task_b

    def test_scenario_find_three_node_dependency_chain(self, tmp_path):
        """
        Scenario: Finding transitive dependencies A -> B -> C

        Given a chain of three dependent tasks
        When I search for three-node dependency pattern
        Then I find the complete chain
        Because our pattern matcher handles multi-node patterns
        """
        # Given a chain of three dependent tasks
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(title="High-level feature")
        task_b = manager.create_task(title="Mid-level component")
        task_c = manager.create_task(title="Low-level primitive")

        manager.add_dependency(task_a.id, task_b.id)
        manager.add_dependency(task_b.id, task_c.id)

        # When I search for three-node dependency pattern
        pattern = (
            Pattern()
            .node("top", type="task")
            .outgoing("DEPENDS_ON")
            .node("middle", type="task")
            .outgoing("DEPENDS_ON")
            .node("bottom", type="task")
        )
        matcher = PatternMatcher(manager)
        result = matcher.find(pattern)

        # Then I find the complete chain
        assert len(result) == 1
        match = result[0]
        assert match["top"].id == task_a.id
        assert match["middle"].id == task_b.id
        assert match["bottom"].id == task_c.id

    def test_scenario_find_tasks_blocking_high_priority_work(self, tmp_path):
        """
        Scenario: Finding bottlenecks in our task graph

        Given tasks where some block high-priority work
        When I search for "blocker blocks high-priority task"
        Then I find the blocking relationships
        So I can focus on removing blockers
        """
        # Given tasks where some block high-priority work
        manager = GoTManager(tmp_path / ".got")
        blocker1 = manager.create_task(title="Blocker 1", priority="medium")
        blocker2 = manager.create_task(title="Blocker 2", priority="low")
        high_priority = manager.create_task(title="Critical feature", priority="high")
        other_task = manager.create_task(title="Other", priority="medium")

        manager.add_blocks(blocker1.id, high_priority.id)
        manager.add_blocks(blocker2.id, high_priority.id)

        # When I search for "blocker blocks high-priority task"
        pattern = (
            Pattern()
            .node("blocker", type="task")
            .outgoing("BLOCKS")
            .node("blocked", type="task", priority="high")
        )
        matcher = PatternMatcher(manager)
        result = matcher.find(pattern)

        # Then I find the blocking relationships
        assert len(result) == 2
        blocker_ids = {match["blocker"].id for match in result}
        assert blocker1.id in blocker_ids
        assert blocker2.id in blocker_ids


class TestDeveloperFindsPatternsWithLimitedResults:
    """
    As a developer analyzing large graphs,
    I want to limit pattern search results,
    So that I don't get overwhelmed by matches in our custom pattern engine.
    """

    def test_scenario_limit_pattern_matches_to_prevent_explosion(self, tmp_path):
        """
        Scenario: Limiting pattern matches for performance

        Given many tasks with similar patterns
        When I search with a limit of 5
        Then I get exactly 5 matches
        And I'm warned about truncation
        """
        # Given many tasks with similar patterns
        manager = GoTManager(tmp_path / ".got")
        base = manager.create_task(title="Base task")
        for i in range(20):
            dependent = manager.create_task(title=f"Dependent {i}")
            manager.add_dependency(dependent.id, base.id)

        # When I search with a limit of 5
        pattern = (
            Pattern()
            .node("dep", type="task")
            .outgoing("DEPENDS_ON")
            .node("base", type="task")
        )
        matcher = PatternMatcher(manager).limit(5)
        result = matcher.find(pattern)

        # Then I get exactly 5 matches
        assert len(result.matches) == 5

        # And I'm warned about truncation
        assert result.truncated is True
        assert result.limit_value == 5

    def test_scenario_find_first_match_efficiently(self, tmp_path):
        """
        Scenario: Finding just one example quickly

        Given a graph with multiple matches
        When I use find_first()
        Then I get exactly one match
        And the search stops early for efficiency
        """
        # Given a graph with multiple matches
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(title="Task A")
        task_b = manager.create_task(title="Task B")
        task_c = manager.create_task(title="Task C")

        manager.add_dependency(task_a.id, task_b.id)
        manager.add_dependency(task_c.id, task_b.id)

        # When I use find_first()
        pattern = (
            Pattern()
            .node("a", type="task")
            .outgoing("DEPENDS_ON")
            .node("b", type="task")
        )
        matcher = PatternMatcher(manager)
        match = matcher.find_first(pattern)

        # Then I get exactly one match
        assert match is not None
        assert match["a"].id in [task_a.id, task_c.id]
        assert match["b"].id == task_b.id


class TestDeveloperExplainsPatternExecutionPlan:
    """
    As a developer optimizing pattern searches,
    I want to see the execution plan before running,
    So that I understand what our custom pattern engine will do.
    """

    def test_scenario_explain_pattern_without_executing(self, tmp_path):
        """
        Scenario: Getting pattern analysis without execution

        Given a pattern to search
        When I call explain()
        Then I get a plan without executing the search
        And the plan shows node and edge constraints
        """
        # Given a pattern to search
        manager = GoTManager(tmp_path / ".got")
        pattern = (
            Pattern()
            .node("a", type="task", status="pending")
            .outgoing("DEPENDS_ON")
            .node("b", type="task")
        )

        # When I call explain()
        matcher = PatternMatcher(manager).limit(10)
        plan = matcher.explain(pattern)

        # Then I get a plan without executing the search
        assert plan.pattern_nodes == 2
        assert plan.pattern_edges == 1
        assert plan.limit == 10

        # And the plan shows node and edge constraints
        assert len(plan.node_constraints) == 2
        assert "DEPENDS_ON" in plan.edge_constraints[0]


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
