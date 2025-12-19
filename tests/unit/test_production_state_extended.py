"""
Extended unit tests for ProductionState module.
"""

import pytest
from cortical.reasoning import (
    ProductionState,
    ProductionChunk,
    CommentMarker,
    ProductionTask,
    ProductionManager,
    ChunkPlanner,
    CommentCleaner,
    ProductionMetrics,
)


class TestProductionState:
    """Tests for ProductionState enum."""

    def test_all_states_exist(self):
        """Test all production states exist."""
        assert ProductionState.PLANNING is not None
        assert ProductionState.DRAFTING is not None
        assert ProductionState.REFINING is not None
        assert ProductionState.FINALIZING is not None
        assert ProductionState.COMPLETE is not None
        assert ProductionState.BLOCKED is not None
        assert ProductionState.REWORK is not None
        assert ProductionState.ABANDONED is not None


class TestProductionChunk:
    """Tests for ProductionChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a production chunk."""
        chunk = ProductionChunk(name="test", goal="Test goal")

        assert chunk.name == "test"
        assert chunk.goal == "Test goal"
        assert chunk.status == ProductionState.PLANNING

    def test_chunk_with_outputs(self):
        """Test chunk with output files."""
        chunk = ProductionChunk(
            name="test",
            goal="Test goal",
            outputs=["file1.py", "file2.py"]
        )

        assert len(chunk.outputs) == 2

    def test_chunk_time_estimate(self):
        """Test chunk time estimate."""
        chunk = ProductionChunk(
            name="test",
            goal="Test goal",
            time_estimate_minutes=30
        )

        assert chunk.time_estimate_minutes == 30

    def test_chunk_start(self):
        """Test starting a chunk."""
        chunk = ProductionChunk(name="test", goal="Test goal")
        chunk.start()

        assert chunk.status == ProductionState.DRAFTING
        assert chunk.started_at is not None

    def test_chunk_complete(self):
        """Test completing a chunk."""
        chunk = ProductionChunk(name="test", goal="Test goal")
        chunk.start()
        chunk.complete()

        assert chunk.status == ProductionState.COMPLETE
        assert chunk.completed_at is not None


class TestCommentMarker:
    """Tests for CommentMarker dataclass."""

    def test_create_marker(self):
        """Test creating a comment marker."""
        marker = CommentMarker(
            marker_type="TODO",
            content="Fix this later"
        )

        assert marker.marker_type == "TODO"
        assert marker.content == "Fix this later"

    def test_marker_with_file_path(self):
        """Test marker with file path."""
        marker = CommentMarker(
            marker_type="THINKING",
            content="Consider approach",
            file_path="src/main.py"
        )

        assert marker.file_path == "src/main.py"

    def test_thinking_factory(self):
        """Test THINKING factory method."""
        marker = CommentMarker.thinking("Why using this approach", "main.py", 42)

        assert marker.marker_type == "THINKING"
        assert "this approach" in marker.content
        assert marker.file_path == "main.py"
        assert marker.line_number == 42

    def test_todo_factory(self):
        """Test TODO factory method."""
        marker = CommentMarker.todo("Add error handling")

        assert marker.marker_type == "TODO"
        assert "error handling" in marker.content

    def test_question_factory(self):
        """Test QUESTION factory method."""
        marker = CommentMarker.question("Is this the right abstraction?")

        assert marker.marker_type == "QUESTION"

    def test_note_factory(self):
        """Test NOTE factory method."""
        marker = CommentMarker.note("Similar to pattern in module X")

        assert marker.marker_type == "NOTE"

    def test_perf_factory(self):
        """Test PERF factory method."""
        marker = CommentMarker.perf("O(n^2) but acceptable for n < 100")

        assert marker.marker_type == "PERF"

    def test_hack_factory(self):
        """Test HACK factory method."""
        marker = CommentMarker.hack("Workaround for issue #123")

        assert marker.marker_type == "HACK"


class TestProductionTask:
    """Tests for ProductionTask class."""

    def test_create_task(self):
        """Test creating a production task."""
        task = ProductionTask(goal="Build feature")

        assert task.id is not None
        assert task.goal == "Build feature"
        assert task.state == ProductionState.PLANNING

    def test_transition_to_drafting(self):
        """Test transitioning to drafting state."""
        task = ProductionTask(goal="Build feature")
        task.transition_to(ProductionState.DRAFTING, "Starting work")

        assert task.state == ProductionState.DRAFTING

    def test_transition_to_refining(self):
        """Test transitioning to refining state."""
        task = ProductionTask(goal="Build feature")
        task.transition_to(ProductionState.DRAFTING, "Start")
        task.transition_to(ProductionState.REFINING, "Draft done")

        assert task.state == ProductionState.REFINING

    def test_transition_to_complete(self):
        """Test transitioning to complete state."""
        task = ProductionTask(goal="Build feature")
        task.transition_to(ProductionState.DRAFTING, "Start")
        task.transition_to(ProductionState.REFINING, "Draft done")
        task.transition_to(ProductionState.FINALIZING, "Refined")
        task.transition_to(ProductionState.COMPLETE, "Done")

        assert task.state == ProductionState.COMPLETE

    def test_invalid_transition_raises(self):
        """Test invalid state transition raises error."""
        task = ProductionTask(goal="Build feature")

        with pytest.raises(ValueError, match="Invalid transition"):
            task.transition_to(ProductionState.COMPLETE, "Skip ahead")

    def test_add_chunk(self):
        """Test adding chunks to task."""
        task = ProductionTask(goal="Build feature")
        chunk = ProductionChunk(name="setup", goal="Setup environment")

        task.add_chunk(chunk)

        assert len(task.chunks) == 1

    def test_add_marker(self):
        """Test adding markers to task."""
        task = ProductionTask(goal="Build feature")
        marker = CommentMarker(marker_type="TODO", content="Fix later")

        task.add_marker(marker)

        assert len(task.markers) == 1

    def test_add_file(self):
        """Test adding files to task."""
        task = ProductionTask(goal="Build feature")
        task.add_file("src/main.py")

        assert "src/main.py" in task.files_modified

    def test_get_summary(self):
        """Test getting task summary."""
        task = ProductionTask(goal="Build feature")
        task.add_file("src/main.py")
        task.add_marker(CommentMarker(marker_type="TODO", content="Fix later"))

        summary = task.get_summary()
        assert summary['goal'] == "Build feature"
        assert summary['files_modified'] == 1
        assert summary['markers_total'] == 1

    def test_can_finalize_success(self):
        """Test can_finalize when ready."""
        task = ProductionTask(goal="Build feature")
        task.transition_to(ProductionState.DRAFTING, "Start")
        task.transition_to(ProductionState.REFINING, "Done draft")

        can_fin, issues = task.can_finalize()
        assert can_fin is True
        assert len(issues) == 0

    def test_can_finalize_with_todos(self):
        """Test can_finalize with unresolved TODOs."""
        task = ProductionTask(goal="Build feature")
        task.add_marker(CommentMarker(marker_type="TODO", content="Fix later"))

        can_fin, issues = task.can_finalize()
        assert can_fin is False
        assert any("TODO" in issue for issue in issues)

    def test_can_finalize_with_questions(self):
        """Test can_finalize with unresolved questions."""
        task = ProductionTask(goal="Build feature")
        task.add_marker(CommentMarker(marker_type="QUESTION", content="Right approach?"))

        can_fin, issues = task.can_finalize()
        assert can_fin is False
        assert any("QUESTION" in issue for issue in issues)

    def test_can_finalize_with_incomplete_chunks(self):
        """Test can_finalize with incomplete chunks."""
        task = ProductionTask(goal="Build feature")
        chunk = ProductionChunk(name="chunk1", goal="First part")
        task.add_chunk(chunk)

        can_fin, issues = task.can_finalize()
        assert can_fin is False
        assert any("incomplete" in issue.lower() for issue in issues)

    def test_get_unresolved_markers(self):
        """Test getting unresolved markers."""
        task = ProductionTask(goal="Build feature")
        marker1 = CommentMarker(marker_type="TODO", content="Fix A")
        marker2 = CommentMarker(marker_type="TODO", content="Fix B")
        marker2.resolved = True
        marker2.resolution = "Fixed"
        task.add_marker(marker1)
        task.add_marker(marker2)

        unresolved = task.get_unresolved_markers()
        assert len(unresolved) == 1
        assert unresolved[0].content == "Fix A"

    def test_get_markers_by_type(self):
        """Test getting markers by type."""
        task = ProductionTask(goal="Build feature")
        task.add_marker(CommentMarker(marker_type="TODO", content="Fix"))
        task.add_marker(CommentMarker(marker_type="THINKING", content="Why"))
        task.add_marker(CommentMarker(marker_type="TODO", content="Another"))

        todos = task.get_markers_by_type("TODO")
        assert len(todos) == 2


class TestProductionManager:
    """Tests for ProductionManager class."""

    def test_create_manager(self):
        """Test creating a production manager."""
        manager = ProductionManager()

        assert manager is not None

    def test_create_task(self):
        """Test creating task via manager."""
        manager = ProductionManager()
        task = manager.create_task(goal="Build feature", description="Feature X")

        assert task is not None
        assert task.goal == "Build feature"

    def test_get_task(self):
        """Test getting task by ID."""
        manager = ProductionManager()
        task = manager.create_task(goal="Build feature", description="Feature X")

        retrieved = manager.get_task(task.id)
        assert retrieved is task

    def test_get_nonexistent_task(self):
        """Test getting nonexistent task returns None."""
        manager = ProductionManager()

        retrieved = manager.get_task("nonexistent")
        assert retrieved is None

    def test_get_tasks_in_state(self):
        """Test getting tasks by state."""
        manager = ProductionManager()
        task1 = manager.create_task(goal="Task 1", description="First")
        task2 = manager.create_task(goal="Task 2", description="Second")

        task1.transition_to(ProductionState.DRAFTING, "Start")

        drafting = manager.get_tasks_in_state(ProductionState.DRAFTING)
        planning = manager.get_tasks_in_state(ProductionState.PLANNING)

        assert len(drafting) == 1
        assert len(planning) == 1

    def test_get_summary(self):
        """Test getting manager summary."""
        manager = ProductionManager()
        manager.create_task(goal="Task 1", description="First")
        manager.create_task(goal="Task 2", description="Second")

        summary = manager.get_summary()

        assert "total_tasks" in summary
        assert summary["total_tasks"] == 2

    def test_register_state_change_handler(self):
        """Test registering state change handler."""
        manager = ProductionManager()
        changes = []

        def handler(task, old_state, new_state):
            changes.append((task.id, old_state, new_state))

        manager.register_state_change_handler(handler)
        task = manager.create_task(goal="Test", description="Test")
        task.transition_to(ProductionState.DRAFTING, "Start")

        # Handler should be called
        assert len(changes) >= 1

    def test_get_blocked_tasks(self):
        """Test getting blocked tasks."""
        manager = ProductionManager()
        task = manager.create_task(goal="Test", description="Test")
        task.transition_to(ProductionState.DRAFTING, "Start")
        task.transition_to(ProductionState.BLOCKED, "Waiting for API")

        blocked = manager.get_blocked_tasks()
        assert len(blocked) == 1

    def test_get_in_progress_tasks(self):
        """Test getting in-progress tasks."""
        manager = ProductionManager()
        task1 = manager.create_task(goal="Task 1", description="First")
        task2 = manager.create_task(goal="Task 2", description="Second")

        task1.transition_to(ProductionState.DRAFTING, "Start")
        task1.transition_to(ProductionState.REFINING, "Done draft")
        task1.transition_to(ProductionState.FINALIZING, "Done refine")
        task1.transition_to(ProductionState.COMPLETE, "Done")

        in_progress = manager.get_in_progress_tasks()
        # Only task2 is still in progress (PLANNING)
        assert len(in_progress) == 1


class TestChunkPlanner:
    """Tests for ChunkPlanner stub class."""

    def test_plan_chunks(self):
        """Test planning chunks for a task."""
        planner = ChunkPlanner()
        task = ProductionTask(goal="Build authentication system")

        chunks = planner.plan_chunks(task)

        assert len(chunks) > 0
        assert all(isinstance(c, ProductionChunk) for c in chunks)

    def test_replan(self):
        """Test replanning chunks."""
        planner = ChunkPlanner()
        task = ProductionTask(goal="Build feature")
        task.add_chunk(ProductionChunk(name="chunk1", goal="First"))

        chunks = planner.replan(task)

        assert chunks is not None


class TestCommentCleaner:
    """Tests for CommentCleaner stub class."""

    def test_scan_file(self):
        """Test scanning a file for markers."""
        cleaner = CommentCleaner()

        markers = cleaner.scan_file("some/file.py")

        # Stub returns empty list
        assert isinstance(markers, list)

    def test_suggest_cleanup_thinking(self):
        """Test cleanup suggestion for THINKING marker."""
        cleaner = CommentCleaner()
        marker = CommentMarker(marker_type="THINKING", content="Why this approach")

        suggestion = cleaner.suggest_cleanup(marker)

        assert suggestion['action'] == 'remove'

    def test_suggest_cleanup_todo(self):
        """Test cleanup suggestion for TODO marker."""
        cleaner = CommentCleaner()
        marker = CommentMarker(marker_type="TODO", content="Fix this")

        suggestion = cleaner.suggest_cleanup(marker)

        assert suggestion['action'] == 'escalate'

    def test_suggest_cleanup_perf(self):
        """Test cleanup suggestion for PERF marker."""
        cleaner = CommentCleaner()
        marker = CommentMarker(marker_type="PERF", content="O(n^2)")

        suggestion = cleaner.suggest_cleanup(marker)

        assert suggestion['action'] == 'keep'


class TestProductionMetrics:
    """Tests for ProductionMetrics stub class."""

    def test_record_state_transition(self):
        """Test recording state transition."""
        metrics = ProductionMetrics()
        task = ProductionTask(goal="Test")

        # Should not raise
        metrics.record_state_transition(
            task,
            ProductionState.PLANNING,
            ProductionState.DRAFTING
        )

    def test_get_average_time_in_state(self):
        """Test getting average time in state."""
        metrics = ProductionMetrics()

        avg = metrics.get_average_time_in_state(ProductionState.DRAFTING)

        assert avg > 0

    def test_get_estimation_accuracy(self):
        """Test getting estimation accuracy."""
        metrics = ProductionMetrics()

        accuracy = metrics.get_estimation_accuracy()

        assert 0 <= accuracy <= 1
