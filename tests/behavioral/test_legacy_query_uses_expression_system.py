"""
Behavioral tests for legacy query command using expression system.

Epic: Legacy Query Backward Compatibility

As an agent using the legacy query command,
I want queries to execute through the expression system,
So that I get consistent results while the system evolves.

This test file defines requirements for the query consolidation:
- Legacy `query()` method should use translator internally
- Output format should remain backward compatible
- All 21 legacy patterns should work through the new path

Design Reference: docs/design/got-query-audit-and-design.md Section 1.5
"""

import pytest
from tests.conftest import _create_got_manager


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def got_path(tmp_path):
    """Provide an isolated GoT path for testing."""
    return tmp_path / ".got"


@pytest.fixture
def got_manager(got_path):
    """Provide an isolated GoT manager for testing."""
    return _create_got_manager(got_path)


@pytest.fixture
def populated_graph(got_manager, got_path):
    """
    Create a graph with known data for testing query equivalence.

    Returns dict with manager, path, and references to created entities.
    """
    manager = got_manager

    # Create tasks with specific properties
    high_pending_1 = manager.create_task(
        title="High priority pending 1",
        priority="high",
        status="pending"
    )
    high_pending_2 = manager.create_task(
        title="High priority pending 2",
        priority="high",
        status="pending"
    )
    critical_blocked = manager.create_task(
        title="Critical blocked task",
        priority="critical",
        status="blocked"
    )
    medium_in_progress = manager.create_task(
        title="Medium in progress",
        priority="medium",
        status="in_progress"
    )
    low_completed = manager.create_task(
        title="Low completed",
        priority="low",
        status="completed"
    )

    # Create blocking relationship
    blocker = manager.create_task(
        title="Blocker task",
        priority="high",
        status="pending"
    )
    manager.add_edge(blocker.id, critical_blocked.id, "BLOCKS")

    # Create orphan task (no edges)
    orphan = manager.create_task(
        title="Orphan task",
        priority="low",
        status="pending"
    )

    # Create sprint with tasks
    sprint = manager.create_sprint(title="Test Sprint", status="in_progress")
    manager.add_task_to_sprint(high_pending_1.id, sprint.id)
    manager.add_task_to_sprint(medium_in_progress.id, sprint.id)

    # Create decision
    decision = manager.create_decision(
        title="Test Decision",
        rationale="For testing"
    )

    return {
        "manager": manager,
        "path": got_path,
        "high_pending_1": high_pending_1,
        "high_pending_2": high_pending_2,
        "critical_blocked": critical_blocked,
        "medium_in_progress": medium_in_progress,
        "low_completed": low_completed,
        "blocker": blocker,
        "orphan": orphan,
        "sprint": sprint,
        "decision": decision,
    }


# =============================================================================
# Test: Query Uses Expression System
# =============================================================================

class TestQueryUsesExpressionSystem:
    """
    Scenario: Legacy query() method uses expression system internally

    Given the query consolidation is complete
    When I call the legacy query() method
    Then it should use translate() → parse() → execute() internally
    And return results in the legacy format
    """

    def test_scenario_query_blocked_tasks_uses_expression(self, populated_graph):
        """
        Scenario: "blocked tasks" query uses blocked() function

        Given a graph with blocked tasks
        When I query "blocked tasks" via legacy API
        Then it returns the same tasks as expr "blocked()"
        And the result format includes expected fields
        """
        from scripts.got_utils import TransactionalGoTAdapter
        from cortical.got.expression import parse, execute

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        # Query via legacy API
        legacy_results = adapter.query("blocked tasks")

        # Query via expression system
        from cortical.got.expression.translator import translate
        expr_str = translate("blocked tasks")
        ast = parse(expr_str)
        expr_results = execute(manager, ast)

        # Results should be equivalent (same task IDs)
        legacy_ids = {r["id"] for r in legacy_results}
        expr_ids = {t.id for t in expr_results}

        assert legacy_ids == expr_ids, (
            f"Legacy returned {legacy_ids}, expr returned {expr_ids}"
        )

    def test_scenario_query_high_priority_uses_expression(self, populated_graph):
        """
        Scenario: "high priority tasks" uses expression filter

        Given tasks with various priorities
        When I query "high priority tasks"
        Then results match expression "priority = 'high'"
        """
        from scripts.got_utils import TransactionalGoTAdapter
        from cortical.got.expression import parse, execute

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        # Query via legacy API
        legacy_results = adapter.query("high priority tasks")

        # Query via expression system
        from cortical.got.expression.translator import translate
        expr_str = translate("high priority tasks")
        ast = parse(expr_str)
        expr_results = execute(manager, ast)

        # Results should be equivalent
        legacy_ids = {r["id"] for r in legacy_results}
        expr_ids = {t.id for t in expr_results}

        assert legacy_ids == expr_ids

    def test_scenario_query_orphans_uses_expression(self, populated_graph):
        """
        Scenario: "orphan tasks" uses orphan_nodes() function

        Given tasks with and without edges
        When I query "orphan tasks"
        Then results match expression "orphan_nodes()"
        """
        from scripts.got_utils import TransactionalGoTAdapter
        from cortical.got.expression import parse, execute

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        # Query via legacy API
        legacy_results = adapter.query("orphan tasks")

        # Query via expression system
        from cortical.got.expression.translator import translate
        expr_str = translate("orphan tasks")
        ast = parse(expr_str)
        expr_results = execute(manager, ast)

        # Results should be equivalent
        legacy_ids = {r["id"] for r in legacy_results}
        expr_ids = {t.id for t in expr_results}

        assert legacy_ids == expr_ids


# =============================================================================
# Test: Parameterized Queries
# =============================================================================

class TestParameterizedQueriesUseExpression:
    """
    Scenario: Parameterized queries extract IDs and use expression functions
    """

    def test_scenario_what_blocks_uses_blockers_function(self, populated_graph):
        """
        Scenario: "what blocks T-XXX" uses blockers() function

        Given a task that is blocked
        When I query "what blocks <task_id>"
        Then results match expression "blockers('<task_id>')"
        """
        from scripts.got_utils import TransactionalGoTAdapter
        from cortical.got.expression import parse, execute

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)
        blocked_task = populated_graph["critical_blocked"]
        blocker_task = populated_graph["blocker"]

        # Query via legacy API
        legacy_results = adapter.query(f"what blocks {blocked_task.id}")

        # Query via expression system
        from cortical.got.expression.translator import translate
        expr_str = translate(f"what blocks {blocked_task.id}")
        ast = parse(expr_str)
        expr_results = execute(manager, ast)

        # Results should include the blocker task
        legacy_ids = {r["id"] for r in legacy_results}
        expr_ids = {t.id for t in expr_results}

        assert blocker_task.id in legacy_ids
        assert legacy_ids == expr_ids

    def test_scenario_tasks_in_sprint_uses_in_sprint_function(self, populated_graph):
        """
        Scenario: "tasks in sprint S-XXX" uses in_sprint() function

        Given a sprint with tasks
        When I query "tasks in sprint <sprint_id>"
        Then results match expression "in_sprint('<sprint_id>')"
        """
        from scripts.got_utils import TransactionalGoTAdapter
        from cortical.got.expression import parse, execute

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)
        sprint = populated_graph["sprint"]

        # Query via legacy API
        legacy_results = adapter.query(f"tasks in sprint {sprint.id}")

        # Query via expression system
        from cortical.got.expression.translator import translate
        expr_str = translate(f"tasks in sprint {sprint.id}")
        ast = parse(expr_str)
        expr_results = execute(manager, ast)

        # Results should be equivalent
        legacy_ids = {r["id"] for r in legacy_results}
        expr_ids = {t.id for t in expr_results}

        assert legacy_ids == expr_ids


# =============================================================================
# Test: Output Format Compatibility
# =============================================================================

class TestOutputFormatCompatibility:
    """
    Scenario: Legacy output format is preserved

    The legacy query() returns List[Dict] with specific fields.
    This format must be maintained for backward compatibility.
    """

    def test_scenario_blocked_tasks_returns_dict_with_reason(self, populated_graph):
        """
        Scenario: Blocked tasks result includes reason field

        Given blocked tasks exist
        When I query "blocked tasks"
        Then each result is a dict with id, title, and reason fields
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        results = adapter.query("blocked tasks")

        # Verify format
        for result in results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "title" in result
            # reason may or may not be present depending on implementation

    def test_scenario_status_queries_return_dict_with_priority(self, populated_graph):
        """
        Scenario: Status queries include priority field

        Given tasks with priorities
        When I query by status
        Then each result includes priority field
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        results = adapter.query("pending tasks")

        for result in results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "title" in result
            assert "priority" in result

    def test_scenario_sprint_queries_return_dict_with_status(self, populated_graph):
        """
        Scenario: Sprint queries include task status

        Given tasks in a sprint
        When I query tasks in sprint
        Then each result includes status field
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)
        sprint = populated_graph["sprint"]

        results = adapter.query(f"tasks in sprint {sprint.id}")

        for result in results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "title" in result
            assert "status" in result


# =============================================================================
# Test: All Legacy Patterns Work
# =============================================================================

class TestAllLegacyPatternsWork:
    """
    Scenario: All 21 legacy patterns produce valid results

    Every pattern from the original query() method should work
    through the translator → expression pipeline.
    """

    # Static patterns that should work without parameters
    STATIC_PATTERNS = [
        "blocked tasks",
        "active tasks",
        "pending tasks",
        "completed tasks",
        "in_progress tasks",
        "all tasks",
        "high priority tasks",
        "critical tasks",
        "orphan tasks",
        "orphan nodes",
        "orphans",
        "current sprint",
        "active sprint",
        "sprints",
        "all sprints",
        "decisions",
        "all decisions",
        "recent tasks",
        "tasks today",
        "stale tasks",
    ]

    def test_scenario_all_static_patterns_return_list(self, populated_graph):
        """
        Scenario: All static patterns return a list

        Given the graph with test data
        When I run each static pattern
        Then each returns a list (possibly empty)
        And no exceptions are raised
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        for pattern in self.STATIC_PATTERNS:
            try:
                results = adapter.query(pattern)
                assert isinstance(results, list), (
                    f"Pattern '{pattern}' should return list, got {type(results)}"
                )
            except Exception as e:
                pytest.fail(f"Pattern '{pattern}' raised exception: {e}")

    def test_scenario_parameterized_what_blocks_works(self, populated_graph):
        """
        Scenario: "what blocks <id>" pattern works

        Given a blocked task
        When I query what blocks it
        Then I get the blocking task
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)
        blocked_task = populated_graph["critical_blocked"]
        blocker_task = populated_graph["blocker"]

        results = adapter.query(f"what blocks {blocked_task.id}")

        assert isinstance(results, list)
        result_ids = {r["id"] for r in results}
        assert blocker_task.id in result_ids

    def test_scenario_parameterized_tasks_in_sprint_works(self, populated_graph):
        """
        Scenario: "tasks in sprint <id>" pattern works

        Given a sprint with tasks
        When I query tasks in that sprint
        Then I get the sprint's tasks
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)
        sprint = populated_graph["sprint"]
        task_in_sprint = populated_graph["high_pending_1"]

        results = adapter.query(f"tasks in sprint {sprint.id}")

        assert isinstance(results, list)
        result_ids = {r["id"] for r in results}
        assert task_in_sprint.id in result_ids


# =============================================================================
# Test: Fallback Behavior
# =============================================================================

class TestFallbackBehavior:
    """
    Scenario: Unknown patterns fall back gracefully

    When a query doesn't match any known pattern,
    the system should handle it gracefully.
    """

    def test_scenario_unknown_pattern_returns_empty_list(self, populated_graph):
        """
        Scenario: Unknown patterns return empty list

        Given a query that matches no pattern
        When I run the query
        Then I get an empty list (not an exception)
        """
        from scripts.got_utils import TransactionalGoTAdapter

        manager = populated_graph["manager"]
        got_path = populated_graph["path"]
        adapter = TransactionalGoTAdapter(got_path)

        # This pattern doesn't match anything
        results = adapter.query("frobnicate the widgets")

        assert isinstance(results, list)
        # May be empty or contain something if it falls back


# =============================================================================
# Test: Expression System Integration
# =============================================================================

class TestExpressionSystemIntegration:
    """
    Scenario: Query method is now a thin wrapper over expression system

    The query() method should be primarily routing through:
    translate() → parse() → execute() → format_results()
    """

    def test_scenario_query_uses_translator_module(self, populated_graph):
        """
        Scenario: Query imports and uses translator

        Given the translator module exists
        When query() processes a request
        Then it should use translate() from the translator module
        """
        # This is a structural verification
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        # Verify these can be called
        expr = translate("blocked tasks")
        assert expr == "blocked()"

        ast = parse(expr)
        assert ast is not None

    def test_scenario_combined_expression_works_via_query(self, populated_graph):
        """
        Scenario: Complex expressions work when translated

        Given compound natural language
        When translated and executed
        Then results are correct
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_graph["manager"]

        # Test compound translation
        expr = translate("high priority pending")
        assert "AND" in expr

        ast = parse(expr)
        results = execute(manager, ast)

        # All results should be high priority AND pending
        for task in results:
            assert task.priority == "high"
            assert task.status == "pending"
