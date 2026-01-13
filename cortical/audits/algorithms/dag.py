"""
TaskDAG Implementation for GoT Task Dependencies.

This module provides a TaskDAG class built on top of DAGGraph from the
unified graph architecture. It maintains full backward compatibility
with the original TaskDAG API while leveraging the BaseGraph infrastructure.

Migration: 2026-01-13 - Migrated from standalone implementation to DAGGraph wrapper.
"""

from typing import List, Set

from cortical.graph.implementations import DAGGraph


class TaskDAG:
    """
    Directed Acyclic Graph for task dependencies.

    This class wraps DAGGraph to provide a task-oriented API:
    - add_task() instead of add_node()
    - add_dependency() returns bool instead of raising on cycle
    - has_dependency() for edge queries
    - roots() and leaves() for graph analysis

    Example:
        dag = TaskDAG()
        dag.add_task("design")
        dag.add_task("implement")
        dag.add_dependency("design", "implement")  # design blocks implement

        order = dag.topological_sort()  # ["design", "implement"]
        ready = dag.ready_tasks({"design"})  # {"design", "implement"}
    """

    def __init__(self) -> None:
        """Initialize empty DAG."""
        self._graph = DAGGraph()

    def add_task(self, task_id: str) -> None:
        """
        Add a task to the graph. No-op if already exists.

        Args:
            task_id: Unique identifier for the task
        """
        if not self._graph.has_node(task_id):
            self._graph.add_node(task_id)

    def add_dependency(self, from_task: str, to_task: str) -> bool:
        """
        Add dependency: from_task must complete before to_task can start.

        Edge direction: from_task -> to_task (from_task blocks to_task)

        Auto-creates tasks if they don't exist.
        Returns False if edge would create a cycle or self-loop, True otherwise.
        If False, the edge is NOT added.

        Args:
            from_task: The blocking task (must complete first)
            to_task: The blocked task (waits for from_task)

        Returns:
            True if edge was added, False if rejected (cycle/self-loop)
        """
        # Auto-create tasks if they don't exist
        self.add_task(from_task)
        self.add_task(to_task)

        try:
            self._graph.add_edge(from_task, to_task)
            return True
        except ValueError:
            # DAGGraph raises ValueError for cycles and self-loops
            return False

    def has_dependency(self, from_task: str, to_task: str) -> bool:
        """
        Check if direct dependency edge exists.

        Args:
            from_task: Source task
            to_task: Target task

        Returns:
            True if from_task directly blocks to_task
        """
        return self._graph.get_edge(from_task, to_task) is not None

    def topological_sort(self) -> List[str]:
        """
        Return tasks in topological order (dependencies before dependents).

        If A -> B, then A appears before B in result.
        Handles disconnected components.

        Returns:
            List of task IDs in dependency order

        Raises:
            ValueError: If graph contains a cycle (shouldn't happen)
        """
        return self._graph.topological_sort()

    def blocked_by(self, task_id: str) -> Set[str]:
        """
        Return all tasks that must complete before this one (transitive).

        This is all predecessors in the dependency graph.

        Args:
            task_id: The task to check

        Returns:
            Set of all blocking task IDs
        """
        return self._graph.blocked_by(task_id)

    def blocks(self, task_id: str) -> Set[str]:
        """
        Return all tasks that are waiting on this one (transitive).

        This is all successors in the dependency graph.

        Args:
            task_id: The task to check

        Returns:
            Set of all blocked task IDs
        """
        return self._graph.blocks(task_id)

    def ready_tasks(self, completed: Set[str]) -> Set[str]:
        """
        Given set of completed task IDs, return tasks that are now ready.

        A task is ready if all its dependencies are in completed set.

        Args:
            completed: Set of completed task IDs

        Returns:
            Set of task IDs that can be started
        """
        return self._graph.ready_tasks(completed)

    def roots(self) -> Set[str]:
        """
        Return tasks with no incoming dependencies (can start immediately).

        Returns:
            Set of root task IDs
        """
        result = set()
        for node in self._graph.nodes:
            if not list(self._graph.neighbors(node.id, "in")):
                result.add(node.id)
        return result

    def leaves(self) -> Set[str]:
        """
        Return tasks with no outgoing dependencies (nothing depends on them).

        Returns:
            Set of leaf task IDs
        """
        result = set()
        for node in self._graph.nodes:
            if not list(self._graph.neighbors(node.id, "out")):
                result.add(node.id)
        return result


# Demo/test cases - this module doesn't use DI so standalone execution is safe
if __name__ == "__main__":
    # Run all test cases
    print("Running TaskDAG Test Suite...")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic dependency tracking
    print("\nTest 1: Basic dependency tracking")
    try:
        dag = TaskDAG()
        dag.add_task("T-001")
        dag.add_task("T-002")
        assert dag.add_dependency("T-001", "T-002") == True  # T-001 blocks T-002
        assert dag.has_dependency("T-001", "T-002") == True
        assert dag.has_dependency("T-002", "T-001") == False
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 2: Cycle detection
    print("\nTest 2: Cycle detection")
    try:
        dag = TaskDAG()
        dag.add_dependency("T-001", "T-002")
        dag.add_dependency("T-002", "T-003")
        # This would create cycle: T-001 -> T-002 -> T-003 -> T-001
        assert dag.add_dependency("T-003", "T-001") == False
        assert dag.has_dependency("T-003", "T-001") == False  # Edge rejected
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 3: Self-loop detection
    print("\nTest 3: Self-loop detection")
    try:
        dag = TaskDAG()
        dag.add_task("T-001")
        assert dag.add_dependency("T-001", "T-001") == False  # Self-loop is a cycle
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 4: Topological sort
    print("\nTest 4: Topological sort")
    try:
        dag = TaskDAG()
        # Sprint planning: design -> implement -> test -> deploy
        dag.add_dependency("T-DESIGN", "T-IMPL")
        dag.add_dependency("T-IMPL", "T-TEST")
        dag.add_dependency("T-TEST", "T-DEPLOY")

        order = dag.topological_sort()
        assert order.index("T-DESIGN") < order.index("T-IMPL")
        assert order.index("T-IMPL") < order.index("T-TEST")
        assert order.index("T-TEST") < order.index("T-DEPLOY")
        print(f"  Order: {order}")
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 5: Blocking relationships
    print("\nTest 5: Blocking relationships")
    try:
        dag = TaskDAG()
        dag.add_dependency("T-001", "T-002")
        dag.add_dependency("T-002", "T-003")
        dag.add_dependency("T-001", "T-003")  # Direct + indirect

        assert dag.blocked_by("T-003") == {"T-001", "T-002"}
        assert dag.blocked_by("T-001") == set()
        assert dag.blocks("T-001") == {"T-002", "T-003"}
        assert dag.blocks("T-003") == set()
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 6: Roots and leaves
    print("\nTest 6: Roots and leaves")
    try:
        dag = TaskDAG()
        dag.add_dependency("T-001", "T-002")
        dag.add_dependency("T-001", "T-003")
        dag.add_dependency("T-002", "T-004")
        dag.add_dependency("T-003", "T-004")

        assert dag.roots() == {"T-001"}
        assert dag.leaves() == {"T-004"}
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 7: Ready tasks
    print("\nTest 7: Ready tasks")
    try:
        dag = TaskDAG()
        dag.add_dependency("T-001", "T-003")
        dag.add_dependency("T-002", "T-003")
        dag.add_dependency("T-003", "T-004")

        # Initially, only T-001 and T-002 are ready (no dependencies)
        assert dag.ready_tasks(set()) == {"T-001", "T-002"}

        # After T-001 done, still waiting on T-002
        assert dag.ready_tasks({"T-001"}) == {"T-001", "T-002"}

        # After both done, T-003 is ready
        assert dag.ready_tasks({"T-001", "T-002"}) == {"T-001", "T-002", "T-003"}

        # After T-003 done, T-004 is ready
        assert dag.ready_tasks({"T-001", "T-002", "T-003"}) == {"T-001", "T-002", "T-003", "T-004"}
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 8: Disconnected components
    print("\nTest 8: Disconnected components")
    try:
        dag = TaskDAG()
        dag.add_dependency("T-001", "T-002")  # Component 1
        dag.add_dependency("T-003", "T-004")  # Component 2 (disconnected)

        order = dag.topological_sort()
        assert len(order) == 4
        assert order.index("T-001") < order.index("T-002")
        assert order.index("T-003") < order.index("T-004")
        print(f"  Order: {order}")
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 9: Duplicate edge handling (idempotent)
    print("\nTest 9: Duplicate edge handling (idempotent)")
    try:
        dag = TaskDAG()
        assert dag.add_dependency("T-001", "T-002") == True
        assert dag.add_dependency("T-001", "T-002") == True  # Idempotent, already exists
        assert len(dag.blocks("T-001")) == 1  # Still just one edge
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 10: Empty graph
    print("\nTest 10: Empty graph")
    try:
        dag = TaskDAG()
        assert dag.topological_sort() == []
        assert dag.roots() == set()
        assert dag.ready_tasks(set()) == set()
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    # Test 11: Real GoT scenario - audit task dependencies
    print("\nTest 11: Real GoT scenario - audit task dependencies")
    try:
        dag = TaskDAG()
        # Planning the misleading comments audit
        dag.add_dependency("T-TEMPLATE", "T-AUDIT-CDG")  # Template before audit
        dag.add_dependency("T-TEMPLATE", "T-AUDIT-GOT")
        dag.add_dependency("T-TEMPLATE", "T-AUDIT-CORE")
        dag.add_dependency("T-AUDIT-CDG", "T-REVIEW")     # All audits before review
        dag.add_dependency("T-AUDIT-GOT", "T-REVIEW")
        dag.add_dependency("T-AUDIT-CORE", "T-REVIEW")
        dag.add_dependency("T-REVIEW", "T-FIX")           # Review before fix

        # T-TEMPLATE is the root
        assert dag.roots() == {"T-TEMPLATE"}
        # T-FIX is the leaf
        assert dag.leaves() == {"T-FIX"}
        # After template done, all audits are ready
        assert dag.ready_tasks({"T-TEMPLATE"}) == {"T-TEMPLATE", "T-AUDIT-CDG", "T-AUDIT-GOT", "T-AUDIT-CORE"}
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print(f"Success Rate: {tests_passed}/{tests_passed + tests_failed} ({100 * tests_passed // (tests_passed + tests_failed)}%)")
    print("=" * 70)

    # Additional edge case demonstrations
    print("\n" + "=" * 70)
    print("Edge Case Demonstrations:")
    print("=" * 70)

    print("\nEdge Case 1: Complex cycle detection")
    dag = TaskDAG()
    dag.add_dependency("A", "B")
    dag.add_dependency("B", "C")
    dag.add_dependency("C", "D")
    result = dag.add_dependency("D", "A")  # Would create 4-node cycle
    print(f"  Attempting to create A->B->C->D->A cycle: {'REJECTED' if not result else 'ALLOWED (BUG!)'}")

    print("\nEdge Case 2: Diamond dependency pattern")
    dag = TaskDAG()
    dag.add_dependency("A", "B")
    dag.add_dependency("A", "C")
    dag.add_dependency("B", "D")
    dag.add_dependency("C", "D")
    print(f"  Diamond pattern (A->B,C; B,C->D):")
    print(f"    Topological order: {dag.topological_sort()}")
    print(f"    D is blocked by: {dag.blocked_by('D')}")

    print("\nEdge Case 3: Isolated nodes")
    dag = TaskDAG()
    dag.add_task("ISOLATED-1")
    dag.add_task("ISOLATED-2")
    dag.add_dependency("A", "B")
    print(f"  Graph with isolated nodes:")
    print(f"    Roots (no dependencies): {dag.roots()}")
    print(f"    Leaves (nothing depends on): {dag.leaves()}")
    print(f"    Topological order: {dag.topological_sort()}")

    print("\nEdge Case 4: Long chain")
    dag = TaskDAG()
    for i in range(10):
        dag.add_dependency(f"T-{i}", f"T-{i+1}")
    print(f"  Long chain T-0 -> T-1 -> ... -> T-10:")
    print(f"    T-10 is blocked by {len(dag.blocked_by('T-10'))} tasks")
    print(f"    T-0 blocks {len(dag.blocks('T-0'))} tasks")

    print("\n" + "=" * 70)
