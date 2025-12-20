"""
Extended unit tests for CognitiveLoop covering all operations.
"""

import pytest
from datetime import datetime
from cortical.reasoning import (
    LoopPhase,
    LoopStatus,
    TerminationReason,
    PhaseContext,
    LoopTransition,
    CognitiveLoop,
    CognitiveLoopManager,
    NestedLoopExecutor,
    LoopStateSerializer,
)


class TestLoopPhase:
    """Tests for LoopPhase enum."""

    def test_all_phases_exist(self):
        """Test all QAPV phases exist."""
        assert LoopPhase.QUESTION is not None
        assert LoopPhase.ANSWER is not None
        assert LoopPhase.PRODUCE is not None
        assert LoopPhase.VERIFY is not None


class TestLoopStatus:
    """Tests for LoopStatus enum."""

    def test_all_statuses_exist(self):
        """Test all loop statuses exist."""
        assert LoopStatus.NOT_STARTED is not None
        assert LoopStatus.ACTIVE is not None
        assert LoopStatus.PAUSED is not None
        assert LoopStatus.BLOCKED is not None
        assert LoopStatus.COMPLETED is not None
        assert LoopStatus.ABANDONED is not None


class TestTerminationReason:
    """Tests for TerminationReason enum."""

    def test_all_reasons_exist(self):
        """Test all termination reasons exist."""
        assert TerminationReason.SUCCESS is not None
        assert TerminationReason.USER_APPROVED is not None
        assert TerminationReason.BUDGET_EXHAUSTED is not None
        assert TerminationReason.QUESTION_INVALID is not None
        assert TerminationReason.ESCALATED is not None
        assert TerminationReason.CRISIS is not None


class TestPhaseContext:
    """Tests for PhaseContext dataclass."""

    def test_create_context(self):
        """Test creating phase context."""
        from datetime import datetime
        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())

        assert ctx.phase == LoopPhase.QUESTION
        assert ctx.questions_raised == []
        assert ctx.decisions_made == []

    def test_add_note(self):
        """Test adding notes to context."""
        from datetime import datetime
        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        ctx.add_note("Important finding")

        assert any("Important finding" in note for note in ctx.notes)


class TestCognitiveLoop:
    """Tests for CognitiveLoop class."""

    def test_create_loop(self):
        """Test creating a cognitive loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")

        assert loop.id == "test1"
        assert loop.goal == "Test goal"
        assert loop.status == LoopStatus.NOT_STARTED

    def test_start_loop(self):
        """Test starting a loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)

        assert loop.status == LoopStatus.ACTIVE
        assert loop.current_phase == LoopPhase.QUESTION

    def test_transition_phases(self):
        """Test transitioning between phases."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Questions answered")

        assert loop.current_phase == LoopPhase.ANSWER

    def test_transition_not_started_raises(self):
        """Test transition before start raises error."""
        loop = CognitiveLoop(id="test1", goal="Test goal")

        with pytest.raises(ValueError, match="Cannot transition"):
            loop.transition(LoopPhase.ANSWER, "reason")

    def test_pause_loop(self):
        """Test pausing a loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.pause("Waiting for input")

        assert loop.status == LoopStatus.PAUSED

    def test_resume_loop(self):
        """Test resuming a paused loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.pause("Waiting")
        loop.resume()

        assert loop.status == LoopStatus.ACTIVE

    def test_block_loop(self):
        """Test blocking a loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.block("External dependency")

        assert loop.status == LoopStatus.BLOCKED

    def test_complete_loop(self):
        """Test completing a loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.complete(TerminationReason.SUCCESS)

        assert loop.status == LoopStatus.COMPLETED
        assert loop.termination_reason == TerminationReason.SUCCESS

    def test_abandon_loop(self):
        """Test abandoning a loop."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.abandon("Not feasible")

        assert loop.status == LoopStatus.ABANDONED

    def test_iteration_count(self):
        """Test tracking iteration counts."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Moving on")
        loop.transition(LoopPhase.QUESTION, "Back to questions")

        count = loop.get_iteration_count(LoopPhase.QUESTION)
        assert count >= 1

    def test_current_context(self):
        """Test getting current phase context."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)

        ctx = loop.current_context()
        assert ctx.phase == LoopPhase.QUESTION

    def test_current_context_no_phase_raises(self):
        """Test getting context without phase raises error."""
        loop = CognitiveLoop(id="test1", goal="Test goal")

        with pytest.raises(ValueError, match="no phase"):
            loop.current_context()


class TestCognitiveLoopManager:
    """Tests for CognitiveLoopManager class."""

    def test_create_manager(self):
        """Test creating a loop manager."""
        manager = CognitiveLoopManager()

        assert manager is not None

    def test_create_loop(self):
        """Test creating a loop via manager."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Test goal")

        assert loop is not None
        assert loop.goal == "Test goal"

    def test_get_loop(self):
        """Test getting a loop by ID."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Test goal")
        loop_id = loop.id

        retrieved = manager.get_loop(loop_id)
        assert retrieved is loop

    def test_get_nonexistent_loop(self):
        """Test getting nonexistent loop returns None."""
        manager = CognitiveLoopManager()

        retrieved = manager.get_loop("nonexistent")
        assert retrieved is None

    def test_get_active_loops(self):
        """Test getting active loops."""
        manager = CognitiveLoopManager()
        loop1 = manager.create_loop("Loop 1")
        loop2 = manager.create_loop("Loop 2")

        loop1.start(LoopPhase.QUESTION)

        active = manager.get_active_loops()
        assert len(active) == 1
        assert active[0] is loop1

    def test_get_summary(self):
        """Test getting manager summary."""
        manager = CognitiveLoopManager()
        manager.create_loop("Loop 1")
        manager.create_loop("Loop 2")

        summary = manager.get_summary()
        assert "total_loops" in summary
        assert summary["total_loops"] == 2

    def test_register_transition_handler(self):
        """Test registering transition handler."""
        manager = CognitiveLoopManager()
        handler_called = []

        def handler(loop, transition):
            handler_called.append((loop.id, transition))

        manager.register_transition_handler(handler)
        loop = manager.create_loop("Test")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Test transition")

        # Handler should be called
        assert len(handler_called) >= 1

    def test_register_completion_handler(self):
        """Test registering completion handler."""
        manager = CognitiveLoopManager()
        completion_called = []

        def handler(loop, reason):
            completion_called.append((loop.id, reason))

        manager.register_completion_handler(handler)
        loop = manager.create_loop("Test")
        loop.start(LoopPhase.QUESTION)
        loop.complete(TerminationReason.SUCCESS)

        assert len(completion_called) == 1
        assert completion_called[0][1] == TerminationReason.SUCCESS

    def test_get_blocked_loops(self):
        """Test getting blocked loops."""
        manager = CognitiveLoopManager()
        loop1 = manager.create_loop("Loop 1")
        loop2 = manager.create_loop("Loop 2")

        loop1.start(LoopPhase.QUESTION)
        loop1.block("Waiting for dependency")
        loop2.start(LoopPhase.ANSWER)

        blocked = manager.get_blocked_loops()
        assert len(blocked) == 1
        assert blocked[0] is loop1

    def test_get_stuck_loops(self):
        """Test detecting stuck loops with many iterations."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Loop")
        loop.start(LoopPhase.QUESTION)

        # Transition multiple times to the same phase
        for i in range(4):
            loop.transition(LoopPhase.ANSWER, f"Iteration {i}")
            loop.transition(LoopPhase.QUESTION, f"Back {i}")

        stuck = manager.get_stuck_loops(iteration_threshold=3)
        assert len(stuck) >= 1
        assert stuck[0] is loop

    def test_get_overdue_loops(self):
        """Test detecting overdue loops."""
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Loop")
        loop.start(LoopPhase.QUESTION)

        # Set created_at to make it appear old
        from datetime import timedelta
        loop.created_at = datetime.now() - timedelta(hours=3)

        overdue = manager.get_overdue_loops(max_minutes=120)
        assert len(overdue) == 1
        assert overdue[0] is loop

    def test_transition_handler_error_handling(self):
        """Test that transition handler errors don't crash the system."""
        manager = CognitiveLoopManager()

        def bad_handler(loop, transition):
            raise Exception("Handler error")

        manager.register_transition_handler(bad_handler)
        loop = manager.create_loop("Test")

        # Should not raise, handler errors are caught
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Test")

    def test_completion_handler_error_handling(self):
        """Test that completion handler errors don't crash the system."""
        manager = CognitiveLoopManager()

        def bad_handler(loop, reason):
            raise Exception("Handler error")

        manager.register_completion_handler(bad_handler)
        loop = manager.create_loop("Test")

        # Should not raise, handler errors are caught
        loop.start(LoopPhase.QUESTION)
        loop.complete(TerminationReason.SUCCESS)

    def test_create_loop_with_parent(self):
        """Test creating a nested loop with parent ID."""
        manager = CognitiveLoopManager()
        parent_loop = manager.create_loop("Parent")
        child_loop = manager.create_loop("Child", parent_id=parent_loop.id)

        assert child_loop.parent_id == parent_loop.id


class TestLoopTransition:
    """Tests for LoopTransition dataclass."""

    def test_create_transition(self):
        """Test creating a transition record."""
        transition = LoopTransition(
            from_phase=LoopPhase.QUESTION,
            to_phase=LoopPhase.ANSWER,
            timestamp=datetime.now(),
            reason="Questions answered"
        )

        assert transition.from_phase == LoopPhase.QUESTION
        assert transition.to_phase == LoopPhase.ANSWER
        assert transition.reason == "Questions answered"


class TestPhaseContextAdditional:
    """Additional tests for PhaseContext."""

    def test_record_question(self):
        """Test recording a question."""
        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        ctx.record_question("What is the requirement?")

        assert "What is the requirement?" in ctx.questions_raised

    def test_record_decision(self):
        """Test recording a decision."""
        ctx = PhaseContext(phase=LoopPhase.ANSWER, started_at=datetime.now())
        ctx.record_decision("Use approach A", "Because it's simpler")

        assert len(ctx.decisions_made) == 1

    def test_end_phase(self):
        """Test ending a phase."""
        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        ctx.end_phase()

        assert ctx.ended_at is not None
        assert ctx.duration_seconds is not None

    def test_elapsed_minutes(self):
        """Test calculating elapsed time in minutes."""
        import time
        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        time.sleep(0.01)  # Sleep 10ms

        elapsed = ctx.elapsed_minutes()
        assert elapsed > 0


class TestCognitiveLoopAdditional:
    """Additional tests for CognitiveLoop."""

    def test_get_all_contexts(self):
        """Test getting all phase contexts."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Moving on")

        contexts = loop.get_all_contexts()
        assert len(contexts) >= 1

    def test_get_summary(self):
        """Test getting loop summary."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Moving on")

        summary = loop.get_summary()
        assert 'id' in summary
        assert 'goal' in summary
        assert 'status' in summary

    def test_block_reason(self):
        """Test block reason is stored."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.block("Waiting for external dependency")

        assert loop.block_reason == "Waiting for external dependency"

    def test_start_already_started_raises(self):
        """Test starting an already started loop raises error."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)

        with pytest.raises(ValueError, match="already in status"):
            loop.start(LoopPhase.ANSWER)

    def test_resume_not_paused_raises(self):
        """Test resuming a non-paused loop raises error."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)

        with pytest.raises(ValueError, match="Cannot resume"):
            loop.resume()

    def test_add_note(self):
        """Test adding notes to current phase."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.add_note("Important finding")

        ctx = loop.current_context()
        assert any("Important finding" in note for note in ctx.notes)

    def test_spawn_child(self):
        """Test spawning a child loop."""
        loop = CognitiveLoop(id="parent", goal="Parent goal")
        loop.start(LoopPhase.QUESTION)

        child = loop.spawn_child("Child goal")

        assert child.parent_id == loop.id
        assert child.id in loop.child_ids
        assert child.goal == "Child goal"

    def test_total_elapsed_minutes(self):
        """Test calculating total elapsed time."""
        import time
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        time.sleep(0.01)  # Sleep 10ms

        elapsed = loop.total_elapsed_minutes()
        assert elapsed > 0

    def test_transition_with_callbacks(self):
        """Test that transition callbacks are called."""
        transition_called = []

        def on_transition(loop, transition):
            transition_called.append(transition)

        loop = CognitiveLoop(id="test1", goal="Test goal", _on_transition=on_transition)
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Moving on")

        assert len(transition_called) >= 1

    def test_complete_with_callback(self):
        """Test that completion callbacks are called."""
        completion_called = []

        def on_complete(loop, reason):
            completion_called.append((loop, reason))

        loop = CognitiveLoop(id="test1", goal="Test goal", _on_complete=on_complete)
        loop.start(LoopPhase.QUESTION)
        loop.complete(TerminationReason.SUCCESS)

        assert len(completion_called) == 1
        assert completion_called[0][1] == TerminationReason.SUCCESS

    def test_iteration_count_multiple_visits(self):
        """Test iteration count with multiple visits to same phase."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "First answer")
        loop.transition(LoopPhase.QUESTION, "Back to questions")
        loop.transition(LoopPhase.ANSWER, "Second answer")

        # Should have been in ANSWER phase twice
        assert loop.get_iteration_count(LoopPhase.ANSWER) == 2

    def test_transition_records_context_snapshot(self):
        """Test that transitions record context snapshots."""
        loop = CognitiveLoop(id="test1", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.add_note("Test note")
        loop.transition(LoopPhase.ANSWER, "Moving on")

        # Check that the transition has a context snapshot
        assert len(loop.transitions) >= 2
        last_transition = loop.transitions[-1]
        assert 'phase' in last_transition.context_snapshot


class TestNestedLoopExecutor:
    """Tests for NestedLoopExecutor stub class."""

    def test_create_executor(self):
        """Test creating executor."""
        manager = CognitiveLoopManager()
        executor = NestedLoopExecutor(manager)

        assert executor is not None

    def test_execute_with_nesting(self):
        """Test executing a loop with nesting (stub implementation)."""
        manager = CognitiveLoopManager()
        executor = NestedLoopExecutor(manager)
        loop = manager.create_loop("Test loop")
        loop.start(LoopPhase.QUESTION)

        # Execute (stub just returns SUCCESS)
        result = executor.execute_with_nesting(loop, max_depth=5)

        assert result == TerminationReason.SUCCESS
        # Should have added a note
        ctx = loop.current_context()
        assert any("NestedLoopExecutor" in note for note in ctx.notes)

    def test_should_spawn_child(self):
        """Test heuristic for spawning child loops."""
        manager = CognitiveLoopManager()
        executor = NestedLoopExecutor(manager)

        # Create context with few questions (should not spawn)
        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        ctx.record_question("Q1")
        ctx.record_question("Q2")

        assert not executor.should_spawn_child(ctx)

        # Add more questions (should spawn)
        ctx.record_question("Q3")
        ctx.record_question("Q4")

        assert executor.should_spawn_child(ctx)

    def test_suggest_child_goals(self):
        """Test suggesting child loop goals from context."""
        manager = CognitiveLoopManager()
        executor = NestedLoopExecutor(manager)

        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        ctx.record_question("How to implement auth?")
        ctx.record_question("How to handle errors?")
        ctx.record_question("How to test this?")

        goals = executor.suggest_child_goals(ctx)

        # Should return up to 3 questions
        assert len(goals) <= 3
        assert "How to implement auth?" in goals


class TestLoopStateSerializer:
    """Tests for LoopStateSerializer class."""

    def test_create_serializer(self):
        """Test creating serializer."""
        serializer = LoopStateSerializer()

        assert serializer is not None

    def test_serialize_deserialize_phase_context(self):
        """Test serializing and deserializing PhaseContext."""
        serializer = LoopStateSerializer()

        ctx = PhaseContext(phase=LoopPhase.QUESTION, started_at=datetime.now())
        ctx.record_question("What is the requirement?")
        ctx.record_decision("Use approach A", "Best option")
        ctx.add_note("Important finding")
        ctx.end_phase()

        # Serialize
        data = serializer.serialize_phase_context(ctx)

        assert data['phase'] == LoopPhase.QUESTION.value
        assert len(data['questions_raised']) == 1
        assert len(data['decisions_made']) == 1
        assert len(data['notes']) == 1

        # Deserialize
        restored_ctx = serializer.deserialize_phase_context(data)

        assert restored_ctx.phase == LoopPhase.QUESTION
        assert len(restored_ctx.questions_raised) == 1
        assert len(restored_ctx.decisions_made) == 1
        assert restored_ctx.duration_seconds is not None

    def test_serialize_deserialize_loop_transition(self):
        """Test serializing and deserializing LoopTransition."""
        serializer = LoopStateSerializer()

        transition = LoopTransition(
            from_phase=LoopPhase.QUESTION,
            to_phase=LoopPhase.ANSWER,
            timestamp=datetime.now(),
            reason="Questions answered",
            context_snapshot={'phase': 'question', 'iteration': 1}
        )

        # Serialize
        data = serializer.serialize_loop_transition(transition)

        assert data['from_phase'] == LoopPhase.QUESTION.value
        assert data['to_phase'] == LoopPhase.ANSWER.value
        assert data['reason'] == "Questions answered"

        # Deserialize
        restored_transition = serializer.deserialize_loop_transition(data)

        assert restored_transition.from_phase == LoopPhase.QUESTION
        assert restored_transition.to_phase == LoopPhase.ANSWER
        assert restored_transition.reason == "Questions answered"

    def test_serialize_deserialize_loop_transition_no_from_phase(self):
        """Test serializing transition with None from_phase (loop start)."""
        serializer = LoopStateSerializer()

        transition = LoopTransition(
            from_phase=None,
            to_phase=LoopPhase.QUESTION,
            timestamp=datetime.now(),
            reason="Loop started"
        )

        # Serialize
        data = serializer.serialize_loop_transition(transition)

        assert data['from_phase'] is None

        # Deserialize
        restored_transition = serializer.deserialize_loop_transition(data)

        assert restored_transition.from_phase is None
        assert restored_transition.to_phase == LoopPhase.QUESTION

    def test_serialize_deserialize_loop(self):
        """Test full loop serialization and deserialization."""
        serializer = LoopStateSerializer()

        # Create a complex loop
        loop = CognitiveLoop(id="test123", goal="Test goal")
        loop.start(LoopPhase.QUESTION)
        loop.add_note("First note")
        loop.transition(LoopPhase.ANSWER, "Moving to answer")
        loop.transition(LoopPhase.PRODUCE, "Creating artifacts")
        loop.block("Waiting for dependency")

        # Serialize
        json_str = serializer.serialize(loop)

        assert "test123" in json_str
        assert "Test goal" in json_str

        # Deserialize
        restored_loop = serializer.deserialize(json_str)

        assert restored_loop.id == loop.id
        assert restored_loop.goal == loop.goal
        assert restored_loop.status == LoopStatus.BLOCKED
        assert restored_loop.current_phase == LoopPhase.PRODUCE
        assert len(restored_loop.phase_contexts) == 3
        assert len(restored_loop.transitions) == 3
        assert restored_loop.block_reason == "Waiting for dependency"

    def test_serialize_deserialize_completed_loop(self):
        """Test serializing a completed loop with termination reason."""
        serializer = LoopStateSerializer()

        loop = CognitiveLoop(id="complete1", goal="Completed task")
        loop.start(LoopPhase.QUESTION)
        loop.complete(TerminationReason.SUCCESS)

        # Serialize
        json_str = serializer.serialize(loop)

        # Deserialize
        restored_loop = serializer.deserialize(json_str)

        assert restored_loop.status == LoopStatus.COMPLETED
        assert restored_loop.termination_reason == TerminationReason.SUCCESS
        assert restored_loop.completed_at is not None

    def test_serialize_deserialize_loop_with_children(self):
        """Test serializing a loop with child loops."""
        serializer = LoopStateSerializer()

        parent = CognitiveLoop(id="parent1", goal="Parent task")
        parent.start(LoopPhase.QUESTION)

        child1 = parent.spawn_child("Child 1")
        child2 = parent.spawn_child("Child 2")

        # Serialize
        json_str = serializer.serialize(parent)

        # Deserialize
        restored_parent = serializer.deserialize(json_str)

        assert restored_parent.parent_id is None
        assert len(restored_parent.child_ids) == 2
        assert child1.id in restored_parent.child_ids
        assert child2.id in restored_parent.child_ids

    def test_serialize_deserialize_manager(self):
        """Test serializing entire manager with multiple loops."""
        serializer = LoopStateSerializer()
        manager = CognitiveLoopManager()

        # Create multiple loops
        loop1 = manager.create_loop("Loop 1")
        loop1.start(LoopPhase.QUESTION)

        loop2 = manager.create_loop("Loop 2")
        loop2.start(LoopPhase.ANSWER)
        loop2.complete(TerminationReason.SUCCESS)

        # Serialize
        json_str = serializer.serialize_manager(manager)

        assert "Loop 1" in json_str
        assert "Loop 2" in json_str

        # Deserialize
        restored_manager = serializer.deserialize_manager(json_str)

        assert len(restored_manager._loops) == 2
        restored_loop1 = restored_manager.get_loop(loop1.id)
        restored_loop2 = restored_manager.get_loop(loop2.id)

        assert restored_loop1 is not None
        assert restored_loop2 is not None
        assert restored_loop1.goal == "Loop 1"
        assert restored_loop2.status == LoopStatus.COMPLETED

    def test_generate_handoff(self):
        """Test generating handoff document (stub implementation)."""
        serializer = LoopStateSerializer()

        loop = CognitiveLoop(id="handoff1", goal="Complex task")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Research complete")
        loop.transition(LoopPhase.PRODUCE, "Creating artifacts")

        handoff = serializer.generate_handoff(loop)

        # Should be markdown format
        assert "# Loop Handoff" in handoff
        assert loop.id in handoff
        assert loop.goal in handoff
        assert "Progress Summary" in handoff
        assert "Next Steps" in handoff
