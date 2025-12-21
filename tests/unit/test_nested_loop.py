"""
Unit tests for NestedLoopExecutor.

Tests hierarchical QAPV loop management including nesting, result aggregation,
depth limiting, and early termination.
"""

import unittest
from cortical.reasoning.nested_loop import NestedLoopExecutor, LoopContext
from cortical.reasoning import LoopPhase, LoopStatus, TerminationReason


class TestLoopContext(unittest.TestCase):
    """Test LoopContext data structure."""

    def test_context_initialization(self):
        """Test basic context initialization."""
        ctx = LoopContext(
            depth=2,
            parent_id="parent123",
            goal="Test goal"
        )

        self.assertEqual(ctx.depth, 2)
        self.assertEqual(ctx.parent_id, "parent123")
        self.assertEqual(ctx.goal, "Test goal")
        self.assertEqual(ctx.accumulated_answers, [])
        self.assertEqual(ctx.child_results, {})

    def test_add_answer(self):
        """Test adding answers to context."""
        ctx = LoopContext(depth=0, parent_id=None, goal="Test")

        ctx.add_answer("Answer 1")
        ctx.add_answer("Answer 2")

        self.assertEqual(len(ctx.accumulated_answers), 2)
        self.assertIn("Answer 1", ctx.accumulated_answers)
        self.assertIn("Answer 2", ctx.accumulated_answers)

    def test_add_child_result(self):
        """Test recording child results."""
        ctx = LoopContext(depth=0, parent_id=None, goal="Test")

        ctx.add_child_result("child1", {"status": "success"})
        ctx.add_child_result("child2", {"status": "failed"})

        self.assertEqual(len(ctx.child_results), 2)
        self.assertEqual(ctx.child_results["child1"]["status"], "success")
        self.assertEqual(ctx.child_results["child2"]["status"], "failed")

    def test_get_all_answers(self):
        """Test retrieving all answers."""
        ctx = LoopContext(depth=0, parent_id=None, goal="Test")

        ctx.add_answer("A1")
        ctx.add_answer("A2")
        ctx.add_answer("A3")

        answers = ctx.get_all_answers()
        self.assertEqual(len(answers), 3)
        # Verify it returns a copy
        answers.append("A4")
        self.assertEqual(len(ctx.accumulated_answers), 3)


class TestNestedLoopExecutor(unittest.TestCase):
    """Test NestedLoopExecutor functionality."""

    def setUp(self):
        """Create executor for each test."""
        self.executor = NestedLoopExecutor(max_depth=5)

    def test_initialization(self):
        """Test executor initialization."""
        executor = NestedLoopExecutor(max_depth=3)
        self.assertEqual(executor._max_depth, 3)
        self.assertEqual(len(executor._loops), 0)
        self.assertEqual(len(executor._contexts), 0)

    def test_initialization_invalid_depth(self):
        """Test initialization with invalid max_depth."""
        with self.assertRaises(ValueError):
            NestedLoopExecutor(max_depth=0)

        with self.assertRaises(ValueError):
            NestedLoopExecutor(max_depth=-1)

    def test_start_root_loop(self):
        """Test creating a root-level loop."""
        loop_id = self.executor.start_root("Implement feature X")

        self.assertIsNotNone(loop_id)
        self.assertIn(loop_id, self.executor._loops)
        self.assertIn(loop_id, self.executor._contexts)

        # Check loop state
        loop = self.executor.get_loop(loop_id)
        self.assertEqual(loop.goal, "Implement feature X")
        self.assertEqual(loop.status, LoopStatus.ACTIVE)
        self.assertEqual(loop.current_phase, LoopPhase.QUESTION)
        self.assertIsNone(loop.parent_id)

        # Check context
        context = self.executor.get_context(loop_id)
        self.assertEqual(context.depth, 0)
        self.assertIsNone(context.parent_id)
        self.assertEqual(context.goal, "Implement feature X")

    def test_advance_loop_through_phases(self):
        """Test advancing a loop through QAPV phases."""
        loop_id = self.executor.start_root("Test advancement")

        # Should start in QUESTION phase
        loop = self.executor.get_loop(loop_id)
        self.assertEqual(loop.current_phase, LoopPhase.QUESTION)

        # Advance to ANSWER
        phase = self.executor.advance(loop_id)
        self.assertEqual(phase, LoopPhase.ANSWER)
        self.assertEqual(loop.current_phase, LoopPhase.ANSWER)

        # Advance to PRODUCE
        phase = self.executor.advance(loop_id)
        self.assertEqual(phase, LoopPhase.PRODUCE)

        # Advance to VERIFY
        phase = self.executor.advance(loop_id)
        self.assertEqual(phase, LoopPhase.VERIFY)

        # Advance back to QUESTION (cycle)
        phase = self.executor.advance(loop_id)
        self.assertEqual(phase, LoopPhase.QUESTION)

    def test_advance_nonexistent_loop(self):
        """Test advancing a loop that doesn't exist."""
        with self.assertRaises(KeyError):
            self.executor.advance("nonexistent_id")

    def test_spawn_child_loop(self):
        """Test spawning a child loop."""
        parent_id = self.executor.start_root("Parent task")
        child_id = self.executor.spawn_child(parent_id, "Child subtask")

        self.assertIsNotNone(child_id)
        self.assertNotEqual(parent_id, child_id)

        # Check child loop
        child_loop = self.executor.get_loop(child_id)
        self.assertEqual(child_loop.goal, "Child subtask")
        self.assertEqual(child_loop.parent_id, parent_id)
        self.assertEqual(child_loop.status, LoopStatus.ACTIVE)

        # Check child context
        child_context = self.executor.get_context(child_id)
        self.assertEqual(child_context.depth, 1)
        self.assertEqual(child_context.parent_id, parent_id)

        # Check parent is paused
        parent_loop = self.executor.get_loop(parent_id)
        self.assertEqual(parent_loop.status, LoopStatus.PAUSED)

    def test_spawn_child_from_nonexistent_parent(self):
        """Test spawning child from nonexistent parent."""
        with self.assertRaises(KeyError):
            self.executor.spawn_child("nonexistent_parent", "Child task")

    def test_spawn_child_exceeds_max_depth(self):
        """Test that spawning exceeds max depth."""
        executor = NestedLoopExecutor(max_depth=3)

        # Create chain: root -> child1 -> child2
        root = executor.start_root("Root")
        child1 = executor.spawn_child(root, "Child 1")
        child2 = executor.spawn_child(child1, "Child 2")

        # Verify depths
        self.assertEqual(executor.get_context(root).depth, 0)
        self.assertEqual(executor.get_context(child1).depth, 1)
        self.assertEqual(executor.get_context(child2).depth, 2)

        # Attempting to spawn from child2 should fail (would be depth 3, max is 3)
        with self.assertRaises(RecursionError):
            executor.spawn_child(child2, "Child 3")

    def test_record_answer(self):
        """Test recording answers in a loop."""
        loop_id = self.executor.start_root("Research task")

        self.executor.record_answer(loop_id, "Finding 1")
        self.executor.record_answer(loop_id, "Finding 2")

        context = self.executor.get_context(loop_id)
        self.assertEqual(len(context.accumulated_answers), 2)
        self.assertIn("Finding 1", context.accumulated_answers)
        self.assertIn("Finding 2", context.accumulated_answers)

    def test_complete_root_loop(self):
        """Test completing a root loop."""
        loop_id = self.executor.start_root("Task to complete")

        result = {"output": "success"}
        parent_id = self.executor.complete(loop_id, result)

        # Should return None for root loop
        self.assertIsNone(parent_id)

        # Loop should be completed
        loop = self.executor.get_loop(loop_id)
        self.assertEqual(loop.status, LoopStatus.COMPLETED)
        self.assertEqual(loop.termination_reason, TerminationReason.SUCCESS)

    def test_complete_child_loop_and_resume_parent(self):
        """Test completing a child loop resumes parent and aggregates result."""
        parent_id = self.executor.start_root("Parent task")
        child_id = self.executor.spawn_child(parent_id, "Child task")

        # Parent should be paused
        parent_loop = self.executor.get_loop(parent_id)
        self.assertEqual(parent_loop.status, LoopStatus.PAUSED)

        # Complete child
        child_result = {"status": "done", "value": 42}
        returned_parent_id = self.executor.complete(child_id, child_result)

        # Should return parent ID
        self.assertEqual(returned_parent_id, parent_id)

        # Child should be completed
        child_loop = self.executor.get_loop(child_id)
        self.assertEqual(child_loop.status, LoopStatus.COMPLETED)

        # Parent should be resumed
        self.assertEqual(parent_loop.status, LoopStatus.ACTIVE)

        # Result should be in parent's context
        parent_context = self.executor.get_context(parent_id)
        self.assertIn(child_id, parent_context.child_results)
        self.assertEqual(parent_context.child_results[child_id], child_result)

    def test_break_loop(self):
        """Test breaking a loop early."""
        loop_id = self.executor.start_root("Task to break")

        self.executor.break_loop(loop_id, "Requirements changed")

        loop = self.executor.get_loop(loop_id)
        self.assertEqual(loop.status, LoopStatus.ABANDONED)

    def test_break_child_loop_resumes_parent(self):
        """Test breaking a child loop resumes parent."""
        parent_id = self.executor.start_root("Parent task")
        child_id = self.executor.spawn_child(parent_id, "Child task")

        # Parent should be paused
        parent_loop = self.executor.get_loop(parent_id)
        self.assertEqual(parent_loop.status, LoopStatus.PAUSED)

        # Break child
        self.executor.break_loop(child_id, "Not needed anymore")

        # Child should be abandoned
        child_loop = self.executor.get_loop(child_id)
        self.assertEqual(child_loop.status, LoopStatus.ABANDONED)

        # Parent should be resumed
        self.assertEqual(parent_loop.status, LoopStatus.ACTIVE)

    def test_get_loop_hierarchy(self):
        """Test getting loop hierarchy path."""
        root = self.executor.start_root("Root")
        child1 = self.executor.spawn_child(root, "Child 1")
        child2 = self.executor.spawn_child(child1, "Child 2")

        # Check hierarchy for root
        hierarchy = self.executor.get_loop_hierarchy(root)
        self.assertEqual(hierarchy, [root])

        # Check hierarchy for child1
        hierarchy = self.executor.get_loop_hierarchy(child1)
        self.assertEqual(hierarchy, [root, child1])

        # Check hierarchy for child2
        hierarchy = self.executor.get_loop_hierarchy(child2)
        self.assertEqual(hierarchy, [root, child1, child2])

    def test_get_active_loops(self):
        """Test getting active loops."""
        root = self.executor.start_root("Root")
        child1 = self.executor.spawn_child(root, "Child 1")

        active = self.executor.get_active_loops()

        # Child should be active, parent paused
        self.assertIn(child1, active)
        self.assertNotIn(root, active)  # Paused

        # Complete child
        self.executor.complete(child1, {})

        active = self.executor.get_active_loops()

        # Now parent should be active, child completed
        self.assertIn(root, active)
        self.assertNotIn(child1, active)

    def test_get_summary(self):
        """Test getting executor summary."""
        root = self.executor.start_root("Root")
        child1 = self.executor.spawn_child(root, "Child 1")
        child2 = self.executor.spawn_child(child1, "Child 2")

        summary = self.executor.get_summary()

        self.assertEqual(summary['total_loops'], 3)
        self.assertEqual(summary['max_depth_limit'], 5)
        self.assertEqual(summary['max_depth_reached'], 2)
        self.assertIn('ACTIVE', summary['status_counts'])
        self.assertIn('PAUSED', summary['status_counts'])

    def test_multiple_children_aggregation(self):
        """Test spawning multiple children and aggregating results."""
        parent_id = self.executor.start_root("Parent with multiple children")

        # Spawn first child
        child1_id = self.executor.spawn_child(parent_id, "Child 1")
        self.executor.complete(child1_id, {"data": "from_child1"})

        # Parent should be active again
        parent_loop = self.executor.get_loop(parent_id)
        self.assertEqual(parent_loop.status, LoopStatus.ACTIVE)

        # Spawn second child
        child2_id = self.executor.spawn_child(parent_id, "Child 2")
        self.executor.complete(child2_id, {"data": "from_child2"})

        # Check both results are in parent context
        parent_context = self.executor.get_context(parent_id)
        self.assertEqual(len(parent_context.child_results), 2)
        self.assertIn(child1_id, parent_context.child_results)
        self.assertIn(child2_id, parent_context.child_results)
        self.assertEqual(parent_context.child_results[child1_id]["data"], "from_child1")
        self.assertEqual(parent_context.child_results[child2_id]["data"], "from_child2")

    def test_spawn_from_inactive_parent_fails(self):
        """Test that spawning from inactive parent raises error."""
        parent_id = self.executor.start_root("Parent")

        # Complete parent
        self.executor.complete(parent_id, {})

        # Try to spawn from completed parent
        with self.assertRaises(ValueError) as ctx:
            self.executor.spawn_child(parent_id, "Child")

        self.assertIn("not active", str(ctx.exception).lower())

    def test_integration_complex_hierarchy(self):
        """Integration test: complex nested hierarchy with multiple branches."""
        # Create root
        root = self.executor.start_root("Build application")

        # Create two main branches
        backend = self.executor.spawn_child(root, "Build backend")
        self.executor.record_answer(backend, "Using Python/Flask")

        # Backend has subtasks
        backend_child = self.executor.spawn_child(backend, "Setup database")
        self.executor.record_answer(backend_child, "PostgreSQL configured")
        self.executor.complete(backend_child, {"db": "postgres"})

        # Complete backend
        self.executor.complete(backend, {"backend": "ready"})

        # Now work on frontend
        frontend = self.executor.spawn_child(root, "Build frontend")
        self.executor.record_answer(frontend, "Using React")
        self.executor.complete(frontend, {"frontend": "ready"})

        # Verify root has both results
        root_context = self.executor.get_context(root)
        self.assertEqual(len(root_context.child_results), 2)
        self.assertIn("backend", root_context.child_results[backend])
        self.assertIn("frontend", root_context.child_results[frontend])

        # Verify backend has its child result
        backend_context = self.executor.get_context(backend)
        self.assertEqual(len(backend_context.child_results), 1)
        self.assertIn("db", backend_context.child_results[backend_child])


if __name__ == '__main__':
    unittest.main()
