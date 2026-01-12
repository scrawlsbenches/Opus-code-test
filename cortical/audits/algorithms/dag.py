"""
TaskDAG Implementation for GoT Task Dependencies
Implements a Directed Acyclic Graph with cycle detection.
"""

from typing import Dict, List, Set, Optional


class TaskDAG:
    def __init__(self):
        """Initialize empty DAG."""
        self._nodes: Set[str] = set()
        self._edges: Dict[str, Set[str]] = {}  # from -> {to, ...}
        self._reverse: Dict[str, Set[str]] = {}  # to -> {from, ...}

    def add_task(self, task_id: str) -> None:
        """Add a task to the graph. No-op if already exists."""
        if task_id not in self._nodes:
            self._nodes.add(task_id)
            self._edges[task_id] = set()
            self._reverse[task_id] = set()

    def add_dependency(self, from_task: str, to_task: str) -> bool:
        """
        Add dependency: from_task must complete before to_task can start.
        Edge direction: from_task -> to_task (from_task blocks to_task)

        Auto-creates tasks if they don't exist.
        Returns False if edge would create a cycle, True otherwise.
        If False, the edge is NOT added.
        """
        # Auto-create tasks if they don't exist
        self.add_task(from_task)
        self.add_task(to_task)

        # Self-loop detection (a task cannot depend on itself)
        if from_task == to_task:
            return False

        # If edge already exists, it's idempotent - return True
        if to_task in self._edges[from_task]:
            return True

        # Cycle detection: Check if adding this edge would create a cycle
        # If there's already a path from to_task to from_task, adding
        # from_task -> to_task would create a cycle
        if self._has_path(to_task, from_task):
            return False

        # Safe to add the edge
        self._edges[from_task].add(to_task)
        self._reverse[to_task].add(from_task)
        return True

    def _has_path(self, start: str, end: str) -> bool:
        """
        Check if there's a path from start to end using DFS.
        Used for cycle detection.
        """
        if start not in self._nodes or end not in self._nodes:
            return False

        if start == end:
            return True

        visited = set()
        stack = [start]

        while stack:
            current = stack.pop()
            if current == end:
                return True

            if current in visited:
                continue

            visited.add(current)

            # Add all neighbors to stack
            for neighbor in self._edges.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)

        return False

    def has_dependency(self, from_task: str, to_task: str) -> bool:
        """Check if direct dependency edge exists."""
        if from_task not in self._edges:
            return False
        return to_task in self._edges[from_task]

    def topological_sort(self) -> List[str]:
        """
        Return tasks in topological order (dependencies before dependents).
        If A -> B, then A appears before B in result.
        Handles disconnected components.
        Raises ValueError if graph has cycle (shouldn't happen if add_dependency works).

        Uses Kahn's algorithm with a heap for O(V + E log V) complexity
        and deterministic ordering (lexicographically smallest node processed first).
        """
        import heapq

        if not self._nodes:
            return []

        # Kahn's algorithm using in-degree with heap for deterministic ordering
        in_degree = {node: len(self._reverse[node]) for node in self._nodes}
        # Use a min-heap for O(log n) insertion and O(log n) extraction
        # This gives deterministic ordering (smallest node ID first) efficiently
        heap = [node for node in self._nodes if in_degree[node] == 0]
        heapq.heapify(heap)  # O(n)
        result = []

        while heap:
            # Pop smallest node - O(log n) vs O(n) for list.pop(0)
            current = heapq.heappop(heap)
            result.append(current)

            # Reduce in-degree for all neighbors
            for neighbor in self._edges[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, neighbor)  # O(log n) vs O(n log n) for sort

        # If not all nodes processed, there's a cycle
        if len(result) != len(self._nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def blocked_by(self, task_id: str) -> Set[str]:
        """
        Return all tasks that must complete before this one (transitive).
        This is all predecessors in the dependency graph.
        """
        if task_id not in self._nodes:
            return set()

        result = set()
        visited = set()
        stack = list(self._reverse[task_id])

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            result.add(current)

            # Add all tasks that block the current task
            for blocker in self._reverse.get(current, set()):
                if blocker not in visited:
                    stack.append(blocker)

        return result

    def blocks(self, task_id: str) -> Set[str]:
        """
        Return all tasks that are waiting on this one (transitive).
        This is all successors in the dependency graph.
        """
        if task_id not in self._nodes:
            return set()

        result = set()
        visited = set()
        stack = list(self._edges[task_id])

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            result.add(current)

            # Add all tasks that current blocks
            for blocked in self._edges.get(current, set()):
                if blocked not in visited:
                    stack.append(blocked)

        return result

    def ready_tasks(self, completed: Set[str]) -> Set[str]:
        """
        Given set of completed task IDs, return tasks that are now ready.
        A task is ready if all its dependencies are in completed set.
        """
        ready = set()

        for task in self._nodes:
            # A task is ready if:
            # 1. All its dependencies (tasks that block it) are completed
            # 2. It can be completed (or already completed)
            dependencies = self._reverse[task]
            if dependencies.issubset(completed):
                ready.add(task)

        return ready

    def roots(self) -> Set[str]:
        """Return tasks with no incoming dependencies (can start immediately)."""
        return {task for task in self._nodes if len(self._reverse[task]) == 0}

    def leaves(self) -> Set[str]:
        """Return tasks with no outgoing dependencies (nothing depends on them)."""
        return {task for task in self._nodes if len(self._edges[task]) == 0}


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
