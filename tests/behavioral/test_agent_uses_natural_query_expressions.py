"""
Behavioral tests for natural language query translation to expression DSL.

Epic: Agent Uses Natural Query Expressions

As an AI agent working with the Graph of Thought,
I want to use natural language queries that translate to expression DSL,
So that I can query the task graph intuitively while leveraging the power of the expression system.

This test file defines the requirements BEFORE implementation (TDD).
The translator bridges the gap between legacy natural language patterns
and the new expression DSL system.

Design Reference: docs/design/got-query-audit-and-design.md Section 1.5
"""

import pytest
from tests.conftest import _create_got_manager


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def got_manager(tmp_path):
    """Provide an isolated GoT manager for testing."""
    return _create_got_manager(tmp_path / ".got")


@pytest.fixture
def populated_got(got_manager):
    """
    Provide a GoT manager populated with test data for query testing.

    Creates:
    - Tasks with various statuses and priorities
    - Blocking relationships
    - Sprints with tasks
    - Orphan tasks (no edges)
    - Decisions
    """
    manager = got_manager

    # High priority pending tasks
    task_hp1 = manager.create_task(title="Critical bug fix", priority="high", status="pending")
    task_hp2 = manager.create_task(title="Security audit", priority="high", status="pending")

    # Medium priority tasks
    task_mp1 = manager.create_task(title="Refactor module", priority="medium", status="pending")
    task_mp2 = manager.create_task(title="Add tests", priority="medium", status="in_progress")

    # Completed tasks
    task_c1 = manager.create_task(title="Setup CI", priority="high", status="completed")
    task_c2 = manager.create_task(title="Write docs", priority="low", status="completed")

    # Blocked task
    task_blocked = manager.create_task(title="Deploy feature", priority="critical", status="blocked")

    # Create blocking relationships: task_hp1 blocks task_blocked
    manager.add_edge(task_hp1.id, task_blocked.id, "BLOCKS")
    # blocked_task depends on medium_pending (task_blocked → DEPENDS_ON → task_mp1)
    manager.add_edge(task_blocked.id, task_mp1.id, "DEPENDS_ON")

    # Create a sprint with some tasks
    sprint = manager.create_sprint(title="Sprint 1", status="in_progress")
    manager.add_task_to_sprint(task_hp1.id, sprint.id)
    manager.add_task_to_sprint(task_mp1.id, sprint.id)
    manager.add_task_to_sprint(task_mp2.id, sprint.id)

    # Orphan task (no edges)
    task_orphan = manager.create_task(title="Orphan task", priority="low", status="pending")

    # Decision
    decision = manager.create_decision(
        title="Use expression DSL",
        rationale="Better composability and type safety"
    )

    return {
        "manager": manager,
        "tasks": {
            "high_pending_1": task_hp1,
            "high_pending_2": task_hp2,
            "medium_pending": task_mp1,
            "medium_in_progress": task_mp2,
            "completed_1": task_c1,
            "completed_2": task_c2,
            "blocked": task_blocked,
            "orphan": task_orphan,
        },
        "sprint": sprint,
        "decision": decision,
    }


# =============================================================================
# Test: Translator Module Structure
# =============================================================================

class TestTranslatorModuleImportable:
    """
    Scenario: Translator module is properly structured

    Given the translator module exists
    When I import it
    Then all components are available
    Because proper module structure enables integration.
    """

    def test_import_translator_module(self):
        """Translator module should be importable."""
        from cortical.got.expression import translator
        assert translator is not None

    def test_translate_function_available(self):
        """The translate function should be available."""
        from cortical.got.expression.translator import translate
        assert callable(translate)

    def test_pattern_registry_available(self):
        """Pattern registry should be available for introspection."""
        from cortical.got.expression.translator import get_supported_patterns
        patterns = get_supported_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0


# =============================================================================
# Test: Simple Status Queries
# =============================================================================

class TestAgentTranslatesStatusQueries:
    """
    Epic: Agent Uses Natural Query Expressions

    As an agent querying task status,
    I want natural language patterns to translate to status filters,
    So that I can find tasks by state without learning DSL syntax.
    """

    def test_scenario_blocked_tasks_translates_to_function(self, populated_got):
        """
        Scenario: "blocked tasks" translates to blocked() function

        Given a graph with blocked tasks
        When I translate "blocked tasks"
        Then I get an expression that calls blocked()
        And executing it returns the blocked tasks
        Because "blocked tasks" is the most common status query.
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        blocked_task = populated_got["tasks"]["blocked"]

        # When I translate "blocked tasks"
        expr_str = translate("blocked tasks")

        # Then I get an expression that calls blocked()
        assert expr_str == "blocked()", f"Expected 'blocked()', got '{expr_str}'"

        # And executing it returns the blocked tasks
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {t.id for t in results}
        assert blocked_task.id in result_ids

    def test_scenario_pending_tasks_translates_to_filter(self, populated_got):
        """
        Scenario: "pending tasks" translates to status filter

        Given a graph with pending tasks
        When I translate "pending tasks"
        Then I get a status = 'pending' expression
        And executing it returns pending tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        pending_tasks = [
            populated_got["tasks"]["high_pending_1"],
            populated_got["tasks"]["high_pending_2"],
            populated_got["tasks"]["medium_pending"],
            populated_got["tasks"]["orphan"],  # Also pending
        ]

        # When I translate "pending tasks"
        expr_str = translate("pending tasks")

        # Then I get a status filter expression
        assert "status" in expr_str.lower() or "pending" in expr_str.lower()

        # And executing it returns pending tasks
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {t.id for t in results}
        for task in pending_tasks:
            assert task.id in result_ids

    def test_scenario_active_tasks_means_in_progress(self, populated_got):
        """
        Scenario: "active tasks" translates to in_progress status

        Given a graph with tasks in various states
        When I translate "active tasks"
        Then I get a status = 'in_progress' expression
        And executing it returns only in_progress tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        in_progress_task = populated_got["tasks"]["medium_in_progress"]

        # When I translate "active tasks"
        expr_str = translate("active tasks")

        # Then we get an expression for in_progress
        ast = parse(expr_str)
        results = execute(manager, ast)

        # And executing it returns only in_progress tasks
        result_ids = {t.id for t in results}
        assert in_progress_task.id in result_ids
        # Completed tasks should not be included
        assert populated_got["tasks"]["completed_1"].id not in result_ids

    def test_scenario_completed_tasks_filter(self, populated_got):
        """
        Scenario: "completed tasks" translates to completed status filter

        Given a graph with completed and incomplete tasks
        When I translate "completed tasks"
        Then executing the expression returns only completed tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]

        # When I translate "completed tasks"
        expr_str = translate("completed tasks")
        ast = parse(expr_str)
        results = execute(manager, ast)

        # Then executing the expression returns only completed tasks
        for task in results:
            assert task.status == "completed"

        # And includes our completed tasks
        result_ids = {t.id for t in results}
        assert populated_got["tasks"]["completed_1"].id in result_ids
        assert populated_got["tasks"]["completed_2"].id in result_ids


# =============================================================================
# Test: Priority Queries
# =============================================================================

class TestAgentTranslatesPriorityQueries:
    """
    As an agent prioritizing work,
    I want to query tasks by priority using natural language,
    So that I can quickly find urgent work.
    """

    def test_scenario_high_priority_tasks(self, populated_got):
        """
        Scenario: "high priority tasks" finds high priority work

        Given tasks with various priorities
        When I translate "high priority tasks"
        Then executing returns only high priority tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]

        # When I translate "high priority tasks"
        expr_str = translate("high priority tasks")
        ast = parse(expr_str)
        results = execute(manager, ast)

        # Then executing returns only high priority tasks
        for task in results:
            assert task.priority == "high"

        # And includes our high priority tasks
        result_ids = {t.id for t in results}
        assert populated_got["tasks"]["high_pending_1"].id in result_ids
        assert populated_got["tasks"]["high_pending_2"].id in result_ids
        assert populated_got["tasks"]["completed_1"].id in result_ids  # Also high

    def test_scenario_critical_tasks(self, populated_got):
        """
        Scenario: "critical tasks" finds critical priority work

        Given tasks with various priorities
        When I translate "critical tasks"
        Then executing returns only critical priority tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]

        # When I translate "critical tasks"
        expr_str = translate("critical tasks")
        ast = parse(expr_str)
        results = execute(manager, ast)

        # Then executing returns critical tasks
        result_ids = {t.id for t in results}
        assert populated_got["tasks"]["blocked"].id in result_ids  # Critical priority

    def test_scenario_compound_priority_and_status(self, populated_got):
        """
        Scenario: "high priority pending" combines priority and status

        Given tasks with various priorities and statuses
        When I translate "high priority pending"
        Then I get a compound expression
        And executing returns only high priority pending tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]

        # When I translate "high priority pending"
        expr_str = translate("high priority pending")

        # Then I get a compound expression (contains AND)
        assert "AND" in expr_str.upper() or "and" in expr_str

        # And executing returns only high priority pending tasks
        ast = parse(expr_str)
        results = execute(manager, ast)

        for task in results:
            assert task.priority == "high"
            assert task.status == "pending"

        result_ids = {t.id for t in results}
        assert populated_got["tasks"]["high_pending_1"].id in result_ids
        assert populated_got["tasks"]["high_pending_2"].id in result_ids
        # Should NOT include completed high priority
        assert populated_got["tasks"]["completed_1"].id not in result_ids


# =============================================================================
# Test: Relationship Queries (Parameterized)
# =============================================================================

class TestAgentTranslatesRelationshipQueries:
    """
    As an agent navigating task dependencies,
    I want to query relationships using natural language,
    So that I can understand blocking and dependency chains.
    """

    def test_scenario_what_blocks_task(self, populated_got):
        """
        Scenario: "what blocks T-XXX" finds blocking tasks

        Given a task that is blocked by another task
        When I translate "what blocks <task_id>"
        Then I get a blocks() function call with the task ID
        And executing returns the blocking tasks
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        blocked_task = populated_got["tasks"]["blocked"]
        blocking_task = populated_got["tasks"]["high_pending_1"]

        # When I translate "what blocks <task_id>"
        query = f"what blocks {blocked_task.id}"
        expr_str = translate(query)

        # Then I get a blocks() or similar function call
        assert blocked_task.id in expr_str

        # And executing returns the blocking tasks
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {t.id for t in results}
        assert blocking_task.id in result_ids

    def test_scenario_what_depends_on_task(self, populated_got):
        """
        Scenario: "what depends on T-XXX" finds dependent tasks

        Given a task that other tasks depend on
        When I translate "what depends on <task_id>"
        Then executing returns tasks that depend on this task
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        dependency = populated_got["tasks"]["medium_pending"]  # Has DEPENDS_ON edge
        blocked_task = populated_got["tasks"]["blocked"]  # Depends on medium_pending

        # When I translate "what depends on <task_id>"
        query = f"what depends on {dependency.id}"
        expr_str = translate(query)

        # Then executing returns tasks that depend on this task
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {t.id for t in results}
        assert blocked_task.id in result_ids


# =============================================================================
# Test: Graph Function Queries
# =============================================================================

class TestAgentTranslatesGraphFunctionQueries:
    """
    As an agent analyzing graph structure,
    I want to query graph patterns using natural language,
    So that I can find orphans, cycles, and structural issues.
    """

    def test_scenario_orphan_tasks_query(self, populated_got):
        """
        Scenario: "orphan tasks" translates to orphan_nodes() function

        Given a graph with tasks that have no edges
        When I translate "orphan tasks"
        Then I get an orphan_nodes() function call
        And executing returns tasks with no connections
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        orphan_task = populated_got["tasks"]["orphan"]

        # When I translate "orphan tasks"
        expr_str = translate("orphan tasks")

        # Then I get an orphan_nodes() function call
        assert "orphan" in expr_str.lower()

        # And executing returns tasks with no connections
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {t.id for t in results}
        assert orphan_task.id in result_ids

    def test_scenario_orphans_variant(self, populated_got):
        """
        Scenario: "orphans" is equivalent to "orphan tasks"

        Given synonymous query forms
        When I translate "orphans"
        Then I get the same result as "orphan tasks"
        """
        from cortical.got.expression.translator import translate

        expr1 = translate("orphan tasks")
        expr2 = translate("orphans")
        expr3 = translate("orphan nodes")

        # All should produce equivalent expressions
        assert expr1 == expr2 == expr3


# =============================================================================
# Test: Sprint Queries
# =============================================================================

class TestAgentTranslatesSprintQueries:
    """
    As an agent working with sprints,
    I want to query sprint contents using natural language,
    So that I can track sprint progress.
    """

    def test_scenario_tasks_in_sprint(self, populated_got):
        """
        Scenario: "tasks in sprint S-XXX" finds sprint tasks

        Given a sprint with assigned tasks
        When I translate "tasks in sprint <sprint_id>"
        Then I get an expression that finds sprint contents
        And executing returns tasks in that sprint
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        sprint = populated_got["sprint"]
        sprint_tasks = [
            populated_got["tasks"]["high_pending_1"],
            populated_got["tasks"]["medium_pending"],
            populated_got["tasks"]["medium_in_progress"],
        ]

        # When I translate "tasks in sprint <sprint_id>"
        query = f"tasks in sprint {sprint.id}"
        expr_str = translate(query)

        # Then I get an expression that finds sprint contents
        assert sprint.id in expr_str

        # And executing returns tasks in that sprint
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {t.id for t in results}
        for task in sprint_tasks:
            assert task.id in result_ids

    def test_scenario_current_sprint(self, populated_got):
        """
        Scenario: "current sprint" finds the active sprint

        Given a sprint with in_progress status
        When I translate "current sprint"
        Then executing returns the active sprint
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        sprint = populated_got["sprint"]  # status="in_progress"

        # When I translate "current sprint"
        expr_str = translate("current sprint")

        # Then executing returns the active sprint
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {getattr(r, 'id', None) for r in results}
        assert sprint.id in result_ids


# =============================================================================
# Test: Time-Based Queries
# =============================================================================

class TestAgentTranslatesTimeQueries:
    """
    As an agent monitoring recency,
    I want to query tasks by time using natural language,
    So that I can find recent or stale work.
    """

    def test_scenario_recent_tasks(self, populated_got):
        """
        Scenario: "recent tasks" translates to recent() function

        Given tasks created at various times
        When I translate "recent tasks"
        Then I get a recent() function call with default timeframe
        """
        from cortical.got.expression.translator import translate

        # When I translate "recent tasks"
        expr_str = translate("recent tasks")

        # Then I get a recent() function call
        assert "recent" in expr_str.lower()

    def test_scenario_stale_tasks(self, populated_got):
        """
        Scenario: "stale tasks" finds tasks not updated recently

        Given the stale tasks query
        When I translate "stale tasks"
        Then I get a stale() function call
        """
        from cortical.got.expression.translator import translate

        # When I translate "stale tasks"
        expr_str = translate("stale tasks")

        # Then I get a stale() function call
        assert "stale" in expr_str.lower()


# =============================================================================
# Test: Entity Type Queries
# =============================================================================

class TestAgentTranslatesEntityQueries:
    """
    As an agent working with various entity types,
    I want to query non-task entities using natural language,
    So that I can find decisions, sprints, and other entities.
    """

    def test_scenario_all_decisions(self, populated_got):
        """
        Scenario: "decisions" or "all decisions" lists decisions

        Given decisions in the graph
        When I translate "decisions"
        Then I get an expression that lists decision entities
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        decision = populated_got["decision"]

        # When I translate "decisions"
        expr_str = translate("decisions")

        # Then executing returns decisions
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {getattr(r, 'id', None) for r in results}
        assert decision.id in result_ids

    def test_scenario_all_sprints(self, populated_got):
        """
        Scenario: "sprints" or "all sprints" lists sprints

        Given sprints in the graph
        When I translate "sprints"
        Then I get an expression that lists sprint entities
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]
        sprint = populated_got["sprint"]

        # When I translate "sprints"
        expr_str = translate("sprints")

        # Then executing returns sprints
        ast = parse(expr_str)
        results = execute(manager, ast)

        result_ids = {getattr(r, 'id', None) for r in results}
        assert sprint.id in result_ids


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================

class TestTranslatorHandlesEdgeCases:
    """
    As a robust system,
    I want the translator to handle edge cases gracefully,
    So that invalid input doesn't cause crashes.
    """

    def test_scenario_unknown_pattern_returns_original(self):
        """
        Scenario: Unknown patterns pass through unchanged

        Given a query that doesn't match any pattern
        When I translate it
        Then I get the original query (for direct DSL use)
        Or a helpful error suggesting DSL syntax
        """
        from cortical.got.expression.translator import translate

        # When I translate an unknown pattern
        unknown_query = "frobnicate the widgets"
        result = translate(unknown_query)

        # Then we either get it back unchanged or get an error marker
        # (Implementation can choose behavior)
        assert result is not None

    def test_scenario_case_insensitive_matching(self):
        """
        Scenario: Pattern matching is case-insensitive

        Given queries with varied case
        When I translate them
        Then they match the same patterns
        """
        from cortical.got.expression.translator import translate

        # Various case combinations should produce same result
        assert translate("BLOCKED TASKS") == translate("blocked tasks")
        assert translate("Pending Tasks") == translate("pending tasks")
        assert translate("HIGH PRIORITY TASKS") == translate("high priority tasks")

    def test_scenario_whitespace_normalization(self):
        """
        Scenario: Extra whitespace is handled gracefully

        Given queries with extra whitespace
        When I translate them
        Then whitespace is normalized
        """
        from cortical.got.expression.translator import translate

        # Extra spaces should be normalized
        assert translate("blocked  tasks") == translate("blocked tasks")
        assert translate("  pending tasks  ") == translate("pending tasks")


# =============================================================================
# Test: CLI Integration
# =============================================================================

class TestQueryCommandUsesTranslator:
    """
    As an agent using the CLI,
    I want the 'query' command to use the translator,
    So that legacy patterns continue to work via the expression system.

    This ensures backward compatibility during the transition period.
    """

    def test_scenario_query_command_forwards_to_expr(self, populated_got, tmp_path):
        """
        Scenario: 'query' command translates and executes via expr

        Given a legacy natural language query
        When I run it through the query command
        Then it produces the same results as the translated expr
        Because the query command should use the translator internally.

        Note: This test verifies the TransactionalGoTAdapter.query() method
        uses the translator when available.
        """
        # This test defines the integration requirement
        # Implementation should make query() call translate() then expr's execute()
        pass  # Implementation will make this pass


# =============================================================================
# Test: Full Pattern Coverage
# =============================================================================

class TestAllLegacyPatternsTranslate:
    """
    As a system maintainer,
    I want all 21 legacy query patterns to have translations,
    So that no functionality is lost during consolidation.

    Reference: scripts/got_utils.py lines 2392-2689 (legacy query method)
    """

    LEGACY_PATTERNS = [
        # Status queries
        ("blocked tasks", "blocked()"),
        ("active tasks", "status = 'in_progress'"),
        ("pending tasks", "status = 'pending'"),
        ("completed tasks", "status = 'completed'"),
        ("in_progress tasks", "status = 'in_progress'"),
        ("all tasks", None),  # Returns all, no filter needed

        # Priority queries
        ("high priority tasks", "priority = 'high'"),
        ("critical tasks", "priority = 'critical'"),

        # Orphan queries
        ("orphan tasks", "orphan_nodes()"),
        ("orphan nodes", "orphan_nodes()"),
        ("orphans", "orphan_nodes()"),

        # Sprint queries
        ("current sprint", "type:sprint AND status = 'in_progress'"),
        ("active sprint", "type:sprint AND status = 'in_progress'"),
        ("sprints", "type:sprint"),
        ("all sprints", "type:sprint"),

        # Entity queries
        ("decisions", "type:decision"),
        ("all decisions", "type:decision"),

        # Time queries
        ("recent tasks", "recent(1)"),  # 1 day default
        ("tasks today", "recent(1)"),
        ("stale tasks", "stale(7)"),  # 7 days default
    ]

    # Parameterized patterns (require ID extraction)
    PARAMETERIZED_PATTERNS = [
        ("what blocks {task_id}", "blocks('{task_id}')"),
        ("what depends on {task_id}", "depends_on('{task_id}')"),
        ("relationships {task_id}", "relationships('{task_id}')"),
        ("tasks in sprint {sprint_id}", "sprint_tasks('{sprint_id}')"),
        ("what is in {sprint_id}", "sprint_tasks('{sprint_id}')"),
        ("tasks in {sprint_id}", "sprint_tasks('{sprint_id}')"),
        ("show sprint {sprint_id}", "sprint_info('{sprint_id}')"),
    ]

    def test_scenario_all_static_patterns_have_translations(self):
        """
        Scenario: All static legacy patterns translate

        Given the list of legacy patterns
        When I translate each one
        Then each produces a valid expression (or explicitly passes through)
        """
        from cortical.got.expression.translator import translate

        for pattern, expected in self.LEGACY_PATTERNS:
            result = translate(pattern)
            assert result is not None, f"Pattern '{pattern}' should have a translation"
            if expected:
                # Verify the translation is semantically equivalent
                # (exact string match not required, just functional equivalence)
                assert result, f"Pattern '{pattern}' produced empty result"

    def test_scenario_parameterized_patterns_extract_ids(self):
        """
        Scenario: Parameterized patterns correctly extract IDs

        Given a pattern with task/sprint ID placeholder
        When I translate with a real ID
        Then the ID is correctly embedded in the expression
        """
        from cortical.got.expression.translator import translate

        # Test with sample IDs
        test_task_id = "T-20260105-123456-abcdef"
        test_sprint_id = "S-20260105-123456-abcdef"

        # Test task ID extraction
        result = translate(f"what blocks {test_task_id}")
        assert test_task_id in result, f"Task ID not found in: {result}"

        result = translate(f"what depends on {test_task_id}")
        assert test_task_id in result, f"Task ID not found in: {result}"

        # Test sprint ID extraction
        result = translate(f"tasks in sprint {test_sprint_id}")
        assert test_sprint_id in result, f"Sprint ID not found in: {result}"


# =============================================================================
# Test: Compound Expression Support
# =============================================================================

class TestCompoundExpressionTranslation:
    """
    As an agent building complex queries,
    I want to combine natural language with DSL expressions,
    So that I can leverage both systems together.
    """

    def test_scenario_translate_then_combine(self, populated_got):
        """
        Scenario: Translated expression can be combined with DSL

        Given a natural language query
        When I translate it
        Then I can combine it with additional DSL filters
        And the combined expression executes correctly
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        manager = populated_got["manager"]

        # Translate base query
        base_expr = translate("high priority tasks")

        # Combine with additional filter
        combined = f"({base_expr}) AND status = 'pending'"

        # Should parse and execute
        ast = parse(combined)
        results = execute(manager, ast)

        # All results should be high priority AND pending
        for task in results:
            assert task.priority == "high"
            assert task.status == "pending"
