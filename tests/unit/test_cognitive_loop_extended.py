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


class TestNestedLoopExecutor:
    """Tests for NestedLoopExecutor stub class."""

    def test_create_executor(self):
        """Test creating executor."""
        manager = CognitiveLoopManager()
        executor = NestedLoopExecutor(manager)

        assert executor is not None


class TestLoopStateSerializer:
    """Tests for LoopStateSerializer stub class."""

    def test_create_serializer(self):
        """Test creating serializer."""
        serializer = LoopStateSerializer()

        assert serializer is not None
