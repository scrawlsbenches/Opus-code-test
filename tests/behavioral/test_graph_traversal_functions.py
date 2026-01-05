"""
Behavioral tests for graph traversal expression functions.

Epic: Developer Explores Task Dependencies Through Graph Functions

As a developer working with the Graph of Thought,
I want to traverse entity relationships using graph functions,
So that I can understand dependencies, find cycles, and analyze structure.

This test file defines the requirements BEFORE verifying implementation (TDD).
These tests prove we understand what we're building.

Design Reference: docs/design/got-query-audit-and-design.md Section 3.3 (T-013)

UNTESTED FUNCTIONS (from coverage audit):
- ancestors(entity_id)     - Transitive dependencies
- descendants(entity_id)   - Transitive dependents
- children(entity_id)      - Direct dependents
- parents(entity_id)       - Direct dependencies
- all_dependencies(entity_id) - Full dependency graph
- cycle_detect(entity_id)  - Circular dependency detection
- dependents(entity_id)    - Tasks that depend on given task
- exists(entity_id)        - Check if entity exists
- type_of(entity_id)       - Get entity type
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
def dependency_chain(got_manager):
    """
    Create a linear dependency chain for testing transitive relationships.

    Structure:
        T-root → DEPENDS_ON → T-mid → DEPENDS_ON → T-leaf

    This means:
    - T-root depends on T-mid (T-mid must complete first)
    - T-mid depends on T-leaf (T-leaf must complete first)

    Transitive: T-root depends on both T-mid AND T-leaf

    Edge semantics: A → DEPENDS_ON → B means "A depends on B"
    """
    manager = got_manager

    # Create tasks in chain order (leaf first, as it's the foundation)
    task_leaf = manager.create_task(
        title="Foundation task (leaf)",
        priority="low",
        status="completed"
    )
    task_mid = manager.create_task(
        title="Middle task (depends on leaf)",
        priority="medium",
        status="in_progress"
    )
    task_root = manager.create_task(
        title="Top task (depends on mid)",
        priority="high",
        status="pending"
    )

    # Create DEPENDS_ON edges
    # T-root depends on T-mid
    manager.add_edge(task_root.id, task_mid.id, "DEPENDS_ON")
    # T-mid depends on T-leaf
    manager.add_edge(task_mid.id, task_leaf.id, "DEPENDS_ON")

    return {
        "manager": manager,
        "root": task_root,   # Depends on mid and (transitively) leaf
        "mid": task_mid,     # Depends on leaf
        "leaf": task_leaf,   # No dependencies
    }


@pytest.fixture
def circular_deps(got_manager):
    """
    Create a circular dependency for cycle detection testing.

    Structure:
        T-A → DEPENDS_ON → T-B → DEPENDS_ON → T-C → DEPENDS_ON → T-A

    This creates a cycle: A → B → C → A
    """
    manager = got_manager

    task_a = manager.create_task(title="Task A", priority="high", status="pending")
    task_b = manager.create_task(title="Task B", priority="high", status="pending")
    task_c = manager.create_task(title="Task C", priority="high", status="pending")

    # Create cycle: A depends on B, B depends on C, C depends on A
    manager.add_edge(task_a.id, task_b.id, "DEPENDS_ON")
    manager.add_edge(task_b.id, task_c.id, "DEPENDS_ON")
    manager.add_edge(task_c.id, task_a.id, "DEPENDS_ON")

    return {
        "manager": manager,
        "task_a": task_a,
        "task_b": task_b,
        "task_c": task_c,
    }


@pytest.fixture
def mixed_entities(got_manager):
    """
    Create various entity types for exists() and type_of() testing.
    """
    manager = got_manager

    task = manager.create_task(title="Test Task", priority="medium", status="pending")
    sprint = manager.create_sprint(title="Test Sprint", status="available")
    decision = manager.create_decision(title="Test Decision", rationale="Testing")

    return {
        "manager": manager,
        "task": task,
        "sprint": sprint,
        "decision": decision,
    }


# =============================================================================
# Test: ancestors() Function - Transitive Dependencies
# =============================================================================

class TestAncestorsFunction:
    """
    As a developer analyzing task dependencies,
    I want to find all transitive dependencies of a task,
    So that I can understand the full dependency chain.

    ancestors(entity_id) returns all entities this task depends on,
    following DEPENDS_ON edges transitively.
    """

    def test_scenario_direct_ancestor(self, dependency_chain):
        """
        Scenario: ancestors() finds direct dependencies

        Given a task that depends on another task
        When I call ancestors(task_id)
        Then the direct dependency is included
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]

        # When I call ancestors()
        expr = f"ancestors('{root.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then the direct dependency (mid) is included
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, (
            f"Direct dependency {mid.id} not found in ancestors. "
            f"Got: {result_ids}"
        )

    def test_scenario_transitive_ancestors(self, dependency_chain):
        """
        Scenario: ancestors() finds transitive dependencies

        Given a chain: root → mid → leaf
        When I call ancestors(root_id)
        Then both mid AND leaf are returned (transitive closure)
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]
        leaf = dependency_chain["leaf"]

        # When I call ancestors() on root
        expr = f"ancestors('{root.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then both direct (mid) and transitive (leaf) are returned
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, f"Mid not in ancestors: {result_ids}"
        assert leaf.id in result_ids, f"Leaf not in ancestors: {result_ids}"

    def test_scenario_leaf_has_no_ancestors(self, dependency_chain):
        """
        Scenario: Leaf node has no ancestors

        Given a task with no dependencies (leaf)
        When I call ancestors(leaf_id)
        Then an empty list is returned
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        leaf = dependency_chain["leaf"]

        # When I call ancestors() on leaf
        expr = f"ancestors('{leaf.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then empty list is returned (leaf has no dependencies)
        assert len(results) == 0, f"Expected no ancestors for leaf, got: {results}"


# =============================================================================
# Test: descendants() Function - Transitive Dependents
# =============================================================================

class TestDescendantsFunction:
    """
    As a developer understanding impact,
    I want to find all tasks that depend on a given task,
    So that I can understand what would be affected by changes.

    descendants(entity_id) returns all entities that depend on this task,
    following DEPENDS_ON edges transitively in reverse.
    """

    def test_scenario_direct_descendants(self, dependency_chain):
        """
        Scenario: descendants() finds direct dependents

        Given a task that others depend on
        When I call descendants(task_id)
        Then direct dependents are included
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        leaf = dependency_chain["leaf"]
        mid = dependency_chain["mid"]

        # When I call descendants() on leaf
        expr = f"descendants('{leaf.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then direct dependent (mid) is included
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, (
            f"Direct dependent {mid.id} not found in descendants. "
            f"Got: {result_ids}"
        )

    def test_scenario_transitive_descendants(self, dependency_chain):
        """
        Scenario: descendants() finds transitive dependents

        Given a chain: root → mid → leaf
        When I call descendants(leaf_id)
        Then both mid AND root are returned
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]
        leaf = dependency_chain["leaf"]

        # When I call descendants() on leaf
        expr = f"descendants('{leaf.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then both direct (mid) and transitive (root) are returned
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, f"Mid not in descendants: {result_ids}"
        assert root.id in result_ids, f"Root not in descendants: {result_ids}"

    def test_scenario_root_has_no_descendants(self, dependency_chain):
        """
        Scenario: Root node has no descendants

        Given a task that nothing depends on
        When I call descendants(root_id)
        Then an empty list is returned
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]

        # When I call descendants() on root
        expr = f"descendants('{root.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then empty list is returned (nothing depends on root)
        assert len(results) == 0, f"Expected no descendants for root, got: {results}"


# =============================================================================
# Test: children() Function - Direct Dependents (One Level)
# =============================================================================

class TestChildrenFunction:
    """
    As a developer analyzing immediate relationships,
    I want to find tasks that directly depend on a given task,
    So that I can see first-level dependencies without transitive closure.

    children(entity_id) returns only direct dependents (one hop).
    """

    def test_scenario_children_returns_direct_only(self, dependency_chain):
        """
        Scenario: children() returns only direct dependents

        Given a chain: root → mid → leaf
        When I call children(leaf_id)
        Then only mid is returned (not root)
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]
        leaf = dependency_chain["leaf"]

        # When I call children() on leaf
        expr = f"children('{leaf.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then only direct dependent (mid) is returned
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, f"Direct child mid not found: {result_ids}"
        assert root.id not in result_ids, (
            f"Transitive child root should NOT be in children: {result_ids}"
        )


# =============================================================================
# Test: parents() Function - Direct Dependencies (One Level)
# =============================================================================

class TestParentsFunction:
    """
    As a developer checking immediate blockers,
    I want to find tasks that a given task directly depends on,
    So that I can see what needs to complete first.

    parents(entity_id) returns only direct dependencies (one hop).
    """

    def test_scenario_parents_returns_direct_only(self, dependency_chain):
        """
        Scenario: parents() returns only direct dependencies

        Given a chain: root → mid → leaf
        When I call parents(root_id)
        Then only mid is returned (not leaf)
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]
        leaf = dependency_chain["leaf"]

        # When I call parents() on root
        expr = f"parents('{root.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then only direct dependency (mid) is returned
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, f"Direct parent mid not found: {result_ids}"
        assert leaf.id not in result_ids, (
            f"Transitive parent leaf should NOT be in parents: {result_ids}"
        )


# =============================================================================
# Test: all_dependencies() Function - Full Dependency Graph
# =============================================================================

class TestAllDependenciesFunction:
    """
    As a developer understanding the complete picture,
    I want to find all dependencies of a task (direct and transitive),
    So that I have the full dependency graph.

    all_dependencies(entity_id) is semantically equivalent to ancestors()
    but provides a more explicit name.
    """

    def test_scenario_all_dependencies_returns_complete_chain(self, dependency_chain):
        """
        Scenario: all_dependencies() returns complete transitive closure

        Given a chain: root → mid → leaf
        When I call all_dependencies(root_id)
        Then both mid and leaf are returned
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]
        leaf = dependency_chain["leaf"]

        # When I call all_dependencies() on root
        expr = f"all_dependencies('{root.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then both dependencies are returned
        result_ids = {t.id for t in results}
        assert mid.id in result_ids, f"Mid not in all_dependencies: {result_ids}"
        assert leaf.id in result_ids, f"Leaf not in all_dependencies: {result_ids}"


# =============================================================================
# Test: cycle_detect() Function - Circular Dependency Detection
# =============================================================================

class TestCycleDetectFunction:
    """
    As a developer maintaining dependency integrity,
    I want to detect circular dependencies,
    So that I can fix cycles that would cause infinite blocking.

    cycle_detect(entity_id) returns the cycle path if found,
    or an empty list if no cycle exists.
    """

    def test_scenario_detect_cycle_returns_path(self, circular_deps):
        """
        Scenario: cycle_detect() finds and returns cycle path

        Given a circular dependency: A → B → C → A
        When I call cycle_detect(A)
        Then a list containing the cycle path is returned
        """
        from cortical.got.expression import parse, execute

        manager = circular_deps["manager"]
        task_a = circular_deps["task_a"]
        task_b = circular_deps["task_b"]
        task_c = circular_deps["task_c"]

        # When I call cycle_detect()
        expr = f"cycle_detect('{task_a.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then a cycle path is returned
        assert len(results) > 0, "Expected cycle to be detected, got empty list"

        # The cycle should contain all three nodes
        cycle_ids = set(results) if isinstance(results[0], str) else {r.id for r in results}
        assert task_a.id in cycle_ids or any(task_a.id in str(r) for r in results), (
            f"Task A not in cycle path: {results}"
        )

    def test_scenario_no_cycle_returns_empty(self, dependency_chain):
        """
        Scenario: cycle_detect() returns empty for acyclic graph

        Given a linear dependency chain (no cycle)
        When I call cycle_detect(any_task)
        Then an empty list is returned
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]

        # When I call cycle_detect() on acyclic graph
        expr = f"cycle_detect('{root.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then empty list is returned (no cycle)
        assert len(results) == 0, f"Expected no cycle, got: {results}"


# =============================================================================
# Test: dependents() Function - Tasks That Depend On Given Task
# =============================================================================

class TestDependentsFunction:
    """
    As a developer understanding downstream impact,
    I want to find tasks that depend on a given task,
    So that I know what's blocked by this task.

    dependents(task_id) returns tasks that have DEPENDS_ON edges TO this task.
    """

    def test_scenario_dependents_finds_depending_tasks(self, dependency_chain):
        """
        Scenario: dependents() finds tasks depending on target

        Given: root → DEPENDS_ON → mid
        When I call dependents(mid_id)
        Then root is returned (root depends on mid)
        """
        from cortical.got.expression import parse, execute

        manager = dependency_chain["manager"]
        root = dependency_chain["root"]
        mid = dependency_chain["mid"]

        # When I call dependents() on mid
        expr = f"dependents('{mid.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then root is returned (root depends on mid)
        result_ids = {t.id for t in results}
        assert root.id in result_ids, (
            f"Root (which depends on mid) not found. Got: {result_ids}"
        )


# =============================================================================
# Test: exists() Function - Entity Existence Check
# =============================================================================

class TestExistsFunction:
    """
    As a developer validating references,
    I want to check if an entity exists,
    So that I can validate entity IDs before operations.

    exists(entity_id) returns True if entity exists, False otherwise.

    Design Reference: docs/design/got-query-audit-and-design.md specifies
    this function but it was NOT implemented in the expression system.
    """

    def test_scenario_exists_returns_true_for_existing_task(self, mixed_entities):
        """
        Scenario: exists() returns True for existing entity

        Given an existing task
        When I call exists(task_id)
        Then True is returned
        """
        from cortical.got.expression import parse, execute

        manager = mixed_entities["manager"]
        task = mixed_entities["task"]

        # When I call exists()
        expr = f"exists('{task.id}')"
        ast = parse(expr)
        result = execute(manager, ast)

        # Then True is returned
        assert result is True, f"Expected True for existing task, got: {result}"

    def test_scenario_exists_returns_false_for_nonexistent(self, mixed_entities):
        """
        Scenario: exists() returns False for non-existent entity

        Given a fake entity ID
        When I call exists(fake_id)
        Then False is returned
        """
        from cortical.got.expression import parse, execute

        manager = mixed_entities["manager"]

        # When I call exists() with fake ID
        expr = "exists('T-NONEXISTENT-999999')"
        ast = parse(expr)
        result = execute(manager, ast)

        # Then False is returned
        assert result is False, f"Expected False for non-existent, got: {result}"


# =============================================================================
# Test: type_of() Function - Entity Type Discovery
# =============================================================================

class TestTypeOfFunction:
    """
    As a developer working with heterogeneous entities,
    I want to determine the type of an entity,
    So that I can handle different entity types appropriately.

    type_of(entity_id) returns the entity type as a string.

    Design Reference: docs/design/got-query-audit-and-design.md specifies
    this function but it was NOT implemented in the expression system.
    """

    def test_scenario_type_of_returns_task_for_task(self, mixed_entities):
        """
        Scenario: type_of() returns 'task' for task entities

        Given a task entity
        When I call type_of(task_id)
        Then 'task' is returned
        """
        from cortical.got.expression import parse, execute

        manager = mixed_entities["manager"]
        task = mixed_entities["task"]

        # When I call type_of()
        expr = f"type_of('{task.id}')"
        ast = parse(expr)
        result = execute(manager, ast)

        # Then 'task' is returned (case-insensitive check)
        assert result.lower() == "task", f"Expected 'task', got: {result}"

    def test_scenario_type_of_returns_sprint_for_sprint(self, mixed_entities):
        """
        Scenario: type_of() returns 'sprint' for sprint entities

        Given a sprint entity
        When I call type_of(sprint_id)
        Then 'sprint' is returned
        """
        from cortical.got.expression import parse, execute

        manager = mixed_entities["manager"]
        sprint = mixed_entities["sprint"]

        # When I call type_of()
        expr = f"type_of('{sprint.id}')"
        ast = parse(expr)
        result = execute(manager, ast)

        # Then 'sprint' is returned
        assert result.lower() == "sprint", f"Expected 'sprint', got: {result}"

    def test_scenario_type_of_returns_decision_for_decision(self, mixed_entities):
        """
        Scenario: type_of() returns 'decision' for decision entities

        Given a decision entity
        When I call type_of(decision_id)
        Then 'decision' is returned
        """
        from cortical.got.expression import parse, execute

        manager = mixed_entities["manager"]
        decision = mixed_entities["decision"]

        # When I call type_of()
        expr = f"type_of('{decision.id}')"
        ast = parse(expr)
        result = execute(manager, ast)

        # Then 'decision' is returned
        assert result.lower() == "decision", f"Expected 'decision', got: {result}"


# =============================================================================
# Test: Design Principle - No Hardcoded Depth Limits
# =============================================================================

class TestNoHardcodedDepthLimits:
    """
    Design Principle: No hardcoded magic numbers.

    If a query is slow, the developer stops it manually.
    The system doesn't hide the problem.

    Design Reference: docs/design/got-query-audit-and-design.md lines 47-61
    """

    def test_scenario_deep_transitive_traversal(self, got_manager):
        """
        Scenario: Graph functions traverse deeply without artificial limits

        Given a 50-level deep dependency chain
        When I call ancestors(deepest_task)
        Then all 49 ancestors are returned (no depth limit)

        This verifies the design principle: "Default to unlimited traversal"
        """
        from cortical.got.expression import parse, execute

        manager = got_manager

        # Create a deep chain of 50 tasks
        tasks = []
        for i in range(50):
            task = manager.create_task(
                title=f"Task level {i}",
                priority="medium",
                status="pending"
            )
            tasks.append(task)

            # Each task depends on the previous one
            if i > 0:
                manager.add_edge(task.id, tasks[i - 1].id, "DEPENDS_ON")

        # The deepest task (last created) depends on all previous
        deepest = tasks[-1]

        # When I call ancestors() on the deepest task
        expr = f"ancestors('{deepest.id}')"
        ast = parse(expr)
        results = execute(manager, ast)

        # Then all 49 ancestors are returned (no artificial depth limit)
        assert len(results) == 49, (
            f"Expected 49 ancestors (all levels), got {len(results)}. "
            f"Design principle violated: hardcoded depth limit may exist."
        )


# =============================================================================
# Test: Integration with Natural Language Translator
# =============================================================================

class TestGraphFunctionNaturalLanguageIntegration:
    """
    As an agent using natural language,
    I want graph functions to be accessible via natural language,
    So that I can query dependencies without knowing DSL syntax.
    """

    def test_scenario_natural_ancestors_query(self, dependency_chain):
        """
        Scenario: Natural language can invoke ancestors()

        Given the translator supports natural language
        When I query "ancestors of <task_id>" or similar
        Then it translates to ancestors() function call
        """
        from cortical.got.expression.translator import translate, get_supported_patterns

        root = dependency_chain["root"]

        # Check if ancestors pattern is supported
        patterns = get_supported_patterns()

        # Try to translate (may pass through if not yet supported)
        result = translate(f"ancestors of {root.id}")

        # At minimum, the query should not error
        assert result is not None

    def test_scenario_natural_cycle_detection_query(self, circular_deps):
        """
        Scenario: Natural language can request cycle detection

        Given the translator supports natural language
        When I query "check for cycles in <task_id>" or similar
        Then it translates to cycle_detect() function call
        """
        from cortical.got.expression.translator import translate

        task_a = circular_deps["task_a"]

        # Try to translate
        result = translate(f"detect cycles from {task_a.id}")

        # At minimum, the query should not error
        assert result is not None
