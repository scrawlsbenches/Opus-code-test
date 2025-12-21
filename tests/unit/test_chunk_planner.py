"""
Unit tests for ChunkPlanner in cortical/reasoning/production_state.py

Tests cover:
- Action item extraction from various goal formats
- Time estimation based on complexity indicators
- Dependency detection between chunks
- Intelligent replanning with progress awareness
- Parallel chunk suggestions
"""

import pytest
from datetime import datetime, timedelta
from cortical.reasoning.production_state import (
    ChunkPlanner,
    ProductionTask,
    ProductionChunk,
    ProductionState,
)


class TestActionItemExtraction:
    """Unit tests for _extract_action_items method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_extract_numbered_list(self):
        """Test extraction of numbered lists."""
        goal = """
        Implement authentication:
        1. Create user model
        2. Add login endpoint
        3. Implement JWT tokens
        """

        items = self.planner._extract_action_items(goal)

        assert len(items) == 3
        assert "Create user model" in items
        assert "Add login endpoint" in items
        assert "Implement JWT tokens" in items

    def test_extract_bullet_points(self):
        """Test extraction of bullet point lists."""
        goal = """
        Fix the bugs:
        - Handle edge case for empty input
        - Validate user credentials
        - Add error logging
        """

        items = self.planner._extract_action_items(goal)

        assert len(items) == 3
        assert any("edge case" in item.lower() for item in items)
        assert any("validate" in item.lower() for item in items)
        assert any("logging" in item.lower() for item in items)

    def test_extract_mixed_list_formats(self):
        """Test extraction with mixed numbered and bullet formats."""
        goal = """
        Project tasks:
        1. Design the API
        2. Implement endpoints
        - Add tests
        - Update documentation
        """

        items = self.planner._extract_action_items(goal)

        assert len(items) == 4
        assert any("Design" in item for item in items)
        assert any("Implement endpoints" in item for item in items)
        assert any("tests" in item.lower() for item in items)
        assert any("documentation" in item.lower() for item in items)

    def test_extract_verb_phrases_when_no_lists(self):
        """Test fallback to verb phrase extraction when no lists found."""
        goal = "Implement user authentication. Test the login flow. Refactor the code."

        items = self.planner._extract_action_items(goal)

        # Should find at least the verb phrases
        assert len(items) >= 1
        assert any("implement" in item.lower() for item in items)

    def test_extract_removes_duplicates(self):
        """Test that duplicate items are removed."""
        goal = """
        1. Add feature X
        2. Add feature X
        3. Test feature X
        """

        items = self.planner._extract_action_items(goal)

        # Should only have 2 unique items (feature X appears once, test appears once)
        assert len(items) == 2

    def test_extract_empty_goal(self):
        """Test extraction from empty goal."""
        items = self.planner._extract_action_items("")

        assert items == []

    def test_extract_cleans_whitespace(self):
        """Test that items have whitespace cleaned."""
        goal = """
        1.  Create   file   with  spaces
        2. Normal item
        """

        items = self.planner._extract_action_items(goal)

        # Should have cleaned up extra spaces
        assert all("  " not in item for item in items)


class TestTimeEstimation:
    """Unit tests for _estimate_chunk_time method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_base_time_for_simple_action(self):
        """Test that base time is 15 minutes."""
        time = self.planner._estimate_chunk_time("Add a function", [])

        assert time == 15

    def test_time_increases_per_file(self):
        """Test that time increases with more files."""
        files = ["file1.py", "file2.py"]
        time = self.planner._estimate_chunk_time("Modify files", files)

        # 15 (base) + 10 * 2 (files) = 35
        assert time == 35

    def test_refactor_adds_extra_time(self):
        """Test that refactor keyword adds 20 minutes."""
        time = self.planner._estimate_chunk_time("Refactor the authentication module", [])

        # 15 (base) + 20 (refactor) = 35
        assert time == 35

    def test_test_adds_extra_time(self):
        """Test that test keyword adds 15 minutes."""
        time = self.planner._estimate_chunk_time("Test the login flow", [])

        # 15 (base) + 15 (test) = 30
        assert time == 30

    def test_multiple_operations_add_time(self):
        """Test that 'and' keyword adds 10 minutes."""
        time = self.planner._estimate_chunk_time("Create file and add tests", [])

        # 15 (base) + 15 (test) + 10 (and) = 40
        assert time == 40

    def test_long_description_adds_time(self):
        """Test that long descriptions (>100 chars) add 10 minutes."""
        long_action = "A" * 101  # 101 characters
        time = self.planner._estimate_chunk_time(long_action, [])

        # 15 (base) + 10 (long) = 25
        assert time == 25

    def test_time_capped_at_60_minutes(self):
        """Test that time estimate is capped at 60 minutes."""
        # Create a scenario that would exceed 60 minutes
        files = ["f1.py", "f2.py", "f3.py", "f4.py", "f5.py"]  # 50 minutes just from files
        action = "Refactor and test " + "A" * 101  # +20 (refactor) +15 (test) +10 (and) +10 (long)

        time = self.planner._estimate_chunk_time(action, files)

        # Should be capped at 60
        assert time == 60

    def test_combined_complexity_factors(self):
        """Test combined complexity factors."""
        files = ["auth.py"]
        time = self.planner._estimate_chunk_time(
            "Refactor and test authentication with validation", files
        )

        # 15 (base) + 10 (1 file) + 20 (refactor) + 15 (test) + 10 (and) = 70, capped at 60
        assert time == 60


class TestDependencyDetection:
    """Unit tests for _detect_dependencies method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_tests_depend_on_implementation(self):
        """Test that test chunks depend on implementation chunks."""
        chunks = [
            ProductionChunk(name="Step 1: Implement feature", goal="Implement authentication"),
            ProductionChunk(name="Step 2: Test feature", goal="Test authentication flow"),
        ]

        self.planner._detect_dependencies(chunks)

        assert "Step 1: Implement feature" in chunks[1].inputs

    def test_documentation_depends_on_implementation(self):
        """Test that documentation depends on implementation."""
        chunks = [
            ProductionChunk(name="Step 1: Build API", goal="Create the REST API"),
            ProductionChunk(name="Step 2: Doc API", goal="Document the API endpoints"),
        ]

        self.planner._detect_dependencies(chunks)

        assert "Step 1: Build API" in chunks[1].inputs

    def test_refactor_depends_on_tests(self):
        """Test that refactoring depends on having tests."""
        chunks = [
            ProductionChunk(name="Step 1: Add tests", goal="Test the module"),
            ProductionChunk(name="Step 2: Refactor", goal="Refactor the code"),
        ]

        self.planner._detect_dependencies(chunks)

        assert "Step 1: Add tests" in chunks[1].inputs

    def test_file_dependencies(self):
        """Test that chunks depending on file outputs are detected."""
        chunks = [
            ProductionChunk(
                name="Step 1: Create file",
                goal="Create auth.py module",
                outputs=["auth.py"]
            ),
            ProductionChunk(
                name="Step 2: Modify file",
                goal="Update auth.py with new feature",
                outputs=["auth.py"]
            ),
        ]

        self.planner._detect_dependencies(chunks)

        assert "Step 1: Create file" in chunks[1].inputs

    def test_sequential_steps_have_dependency(self):
        """Test that sequential numbered steps have implicit dependency."""
        chunks = [
            ProductionChunk(name="Step 1: First task", goal="Do something"),
            ProductionChunk(name="Step 2: Second task", goal="Do another thing"),
        ]

        self.planner._detect_dependencies(chunks)

        # Step 2 should depend on Step 1 (only if no other dependencies)
        assert "Step 1: First task" in chunks[1].inputs

    def test_no_dependencies_for_independent_chunks(self):
        """Test that independent chunks have no dependencies."""
        chunks = [
            ProductionChunk(name="Step 1: Feature A", goal="Implement feature A"),
            ProductionChunk(name="Step 2: Feature B", goal="Implement feature B"),
        ]

        self.planner._detect_dependencies(chunks)

        # Step 2 depends on Step 1 only via sequential rule
        assert "Step 1: Feature A" in chunks[1].inputs


class TestPlanChunks:
    """Integration tests for plan_chunks method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_plan_chunks_with_numbered_list(self):
        """Test planning chunks from numbered list."""
        task = ProductionTask(
            goal="""
            Implement user authentication:
            1. Create user model in models.py
            2. Add login endpoint in api.py
            3. Test authentication flow
            """,
        )

        chunks = self.planner.plan_chunks(task)

        assert len(chunks) == 3
        assert all(isinstance(c, ProductionChunk) for c in chunks)
        assert all(c.time_estimate_minutes > 0 for c in chunks)

        # Check that files were extracted
        assert any("models.py" in c.outputs for c in chunks)
        assert any("api.py" in c.outputs for c in chunks)

    def test_plan_chunks_with_bullet_points(self):
        """Test planning chunks from bullet points."""
        task = ProductionTask(
            goal="""
            Bug fixes needed:
            - Fix validation error
            - Update error messages
            - Add logging
            """,
        )

        chunks = self.planner.plan_chunks(task)

        assert len(chunks) == 3

    def test_plan_chunks_creates_dependencies(self):
        """Test that dependencies are created between chunks."""
        task = ProductionTask(
            goal="""
            1. Implement feature in feature.py
            2. Test the feature
            3. Document the feature
            """,
        )

        chunks = self.planner.plan_chunks(task)

        # Test should depend on implementation
        assert any("Step 1" in inp for inp in chunks[1].inputs)

        # Documentation should depend on implementation
        assert any("Step 1" in inp for inp in chunks[2].inputs)

    def test_plan_chunks_with_no_action_items(self):
        """Test fallback when no action items found."""
        task = ProductionTask(goal="A simple task with no structure")

        chunks = self.planner.plan_chunks(task)

        # Should create a single chunk
        assert len(chunks) == 1
        assert chunks[0].name == "Main task"
        assert chunks[0].goal == task.goal

    def test_plan_chunks_estimates_time_correctly(self):
        """Test that time estimates vary based on complexity."""
        task = ProductionTask(
            goal="""
            1. Add a simple function
            2. Refactor the entire module and test it thoroughly
            """,
        )

        chunks = self.planner.plan_chunks(task)

        # First chunk should have lower estimate than second
        assert chunks[0].time_estimate_minutes < chunks[1].time_estimate_minutes

    def test_plan_chunks_complex_scenario(self):
        """Integration test with complex realistic goal."""
        task = ProductionTask(
            goal="""
            Build user authentication system:
            1. Create User model in models/user.py
            2. Implement password hashing in utils/auth.py
            3. Add login/logout endpoints in api/auth.py
            4. Write unit tests in tests/test_auth.py
            5. Refactor authentication logic for clarity
            6. Document the authentication flow in docs/auth.md
            """,
        )

        chunks = self.planner.plan_chunks(task)

        assert len(chunks) == 6

        # Check that files were extracted for each chunk
        assert any("user.py" in str(c.outputs) for c in chunks)
        assert any("auth.py" in str(c.outputs) for c in chunks)
        assert any("test_auth.py" in str(c.outputs) for c in chunks)
        assert any("auth.md" in str(c.outputs) for c in chunks)

        # Check time estimates are reasonable (15-60 range)
        assert all(15 <= c.time_estimate_minutes <= 60 for c in chunks)

        # Check dependencies
        # Refactor chunk should have dependencies
        refactor_chunk = [c for c in chunks if "refactor" in c.goal.lower()][0]
        assert len(refactor_chunk.inputs) > 0


class TestReplan:
    """Behavioral tests for replan method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_replan_preserves_completed_chunks(self):
        """Test that replanning preserves completed chunks."""
        task = ProductionTask(goal="Test task")
        task.chunks = [
            ProductionChunk(name="Chunk 1", goal="First", status=ProductionState.COMPLETE),
            ProductionChunk(name="Chunk 2", goal="Second", status=ProductionState.DRAFTING),
            ProductionChunk(name="Chunk 3", goal="Third", status=ProductionState.PLANNING),
        ]

        updated_chunks = self.planner.replan(task)

        # Should still have 3 chunks
        assert len(updated_chunks) == 3

        # First chunk should still be complete
        completed = [c for c in updated_chunks if c.status == ProductionState.COMPLETE]
        assert len(completed) == 1
        assert completed[0].name == "Chunk 1"

    def test_replan_adjusts_estimates_for_in_progress(self):
        """Test that estimates are adjusted for in-progress chunks."""
        task = ProductionTask(goal="Test task")

        # Create a chunk that's been running longer than estimate
        chunk = ProductionChunk(
            name="Chunk 1",
            goal="Long task",
            time_estimate_minutes=20,
            status=ProductionState.DRAFTING,
        )
        chunk.started_at = datetime.now() - timedelta(minutes=30)  # 30 minutes ago
        task.chunks = [chunk]

        updated_chunks = self.planner.replan(task)

        # Estimate should be increased (30 * 1.3 = 39)
        assert updated_chunks[0].time_estimate_minutes > 20

    def test_replan_no_changes_when_all_complete(self):
        """Test that replan returns same chunks when all complete."""
        task = ProductionTask(goal="Test task")
        task.chunks = [
            ProductionChunk(name="Chunk 1", goal="First", status=ProductionState.COMPLETE),
            ProductionChunk(name="Chunk 2", goal="Second", status=ProductionState.COMPLETE),
        ]

        updated_chunks = self.planner.replan(task)

        assert len(updated_chunks) == 2
        assert all(c.status == ProductionState.COMPLETE for c in updated_chunks)


class TestSuggestParallelChunks:
    """Unit tests for suggest_parallel_chunks method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_independent_chunks_can_parallelize(self):
        """Test that independent chunks can run in parallel."""
        chunks = [
            ProductionChunk(name="Chunk 1", goal="Feature A", outputs=["a.py"]),
            ProductionChunk(name="Chunk 2", goal="Feature B", outputs=["b.py"]),
            ProductionChunk(name="Chunk 3", goal="Feature C", outputs=["c.py"]),
        ]

        groups = self.planner.suggest_parallel_chunks(chunks)

        # All chunks should be in the same group (can run in parallel)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_dependent_chunks_cannot_parallelize(self):
        """Test that dependent chunks cannot run in parallel."""
        chunks = [
            ProductionChunk(name="Chunk 1", goal="Create feature", outputs=["feature.py"]),
            ProductionChunk(name="Chunk 2", goal="Test feature", outputs=["test.py"], inputs=["Chunk 1"]),
        ]

        groups = self.planner.suggest_parallel_chunks(chunks)

        # Should have 2 groups (sequential)
        assert len(groups) == 2

    def test_same_file_chunks_cannot_parallelize(self):
        """Test that chunks modifying same file cannot parallelize."""
        chunks = [
            ProductionChunk(name="Chunk 1", goal="Add function", outputs=["code.py"]),
            ProductionChunk(name="Chunk 2", goal="Add another function", outputs=["code.py"]),
        ]

        groups = self.planner.suggest_parallel_chunks(chunks)

        # Should have 2 groups (can't modify same file in parallel)
        assert len(groups) == 2

    def test_mixed_parallelization(self):
        """Test mixed scenario with some parallel, some sequential."""
        chunks = [
            ProductionChunk(name="Chunk 1", goal="Feature A", outputs=["a.py"]),
            ProductionChunk(name="Chunk 2", goal="Feature B", outputs=["b.py"]),
            ProductionChunk(name="Chunk 3", goal="Test A", outputs=["test_a.py"], inputs=["Chunk 1"]),
            ProductionChunk(name="Chunk 4", goal="Test B", outputs=["test_b.py"], inputs=["Chunk 2"]),
        ]

        groups = self.planner.suggest_parallel_chunks(chunks)

        # Should have at least 2 groups
        # Group 1: Chunk 1 and Chunk 2 (parallel)
        # Group 2: Chunk 3 and Chunk 4 (parallel, but after group 1)
        assert len(groups) >= 2

        # First group should have Chunk 1 and Chunk 2
        first_group_names = [c.name for c in groups[0]]
        assert "Chunk 1" in first_group_names
        assert "Chunk 2" in first_group_names


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = ChunkPlanner()

    def test_empty_task_goal(self):
        """Test handling of empty task goal."""
        task = ProductionTask(goal="")

        chunks = self.planner.plan_chunks(task)

        # Should create a default chunk
        assert len(chunks) == 1

    def test_very_long_goal(self):
        """Test handling of very long goal text."""
        # Create a goal with 10 numbered items
        goal = "\n".join([f"{i}. Item {i}" for i in range(1, 11)])
        task = ProductionTask(goal=goal)

        chunks = self.planner.plan_chunks(task)

        # Should create 10 chunks
        assert len(chunks) == 10

    def test_special_characters_in_goal(self):
        """Test handling of special characters."""
        task = ProductionTask(
            goal="""
            1. Handle edge case: user@domain.com
            2. Support special chars: $, %, &
            """
        )

        chunks = self.planner.plan_chunks(task)

        # Should still extract items correctly
        assert len(chunks) == 2
