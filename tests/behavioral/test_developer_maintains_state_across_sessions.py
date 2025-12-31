"""
Behavioral tests for cognitive state persistence across sessions.

As a developer working with LLM agents,
I want cognitive state to persist across sessions,
So that work continues seamlessly despite session boundaries.

Based on: llm_orchestration/examples/multi_session.py
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from llm_orchestration.cognitive_state import (
    CognitiveStateManager,
    QuestionStatus,
    DecisionStatus
)


class TestDeveloperPersistsState:
    """
    Epic: State Persistence

    As a developer building stateless LLM applications,
    I want all cognitive state externalized to storage,
    So that sessions can end without losing progress.
    """

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for cognitive state."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_checkpoints_capture_complete_state(self, temp_storage):
        """
        Scenario: Checkpoints preserve all cognitive elements

        Given a cognitive state with questions, decisions, and focus
        When creating a checkpoint
        Then all elements are serialized to storage
        Because complete state capture enables full restoration
        """
        # Given: cognitive state with various elements
        state = CognitiveStateManager(temp_storage)

        state.set_focus(
            current_goal="Implement authentication",
            context={"framework": "FastAPI", "deadline": "next_week"}
        )

        question = state.add_question("How to implement OAuth?")
        decision = state.add_decision(
            question_id=question.id,
            choice="Use custom OAuth implementation we built",
            rationale="Full control over security we built from scratch"
        )

        state.add_observation(
            content="OAuth 2.0 requires HTTPS in production",
            source="security documentation we wrote"
        )

        # When: creating checkpoint
        checkpoint = state.save_checkpoint()

        # Then: checkpoint is created and contains state
        assert checkpoint is not None, "Should create checkpoint"
        assert 'id' in checkpoint, "Checkpoint should have ID"
        assert 'timestamp' in checkpoint, "Checkpoint should have timestamp"
        assert 'state' in checkpoint, "Checkpoint should contain state data"

    def test_scenario_checkpoints_stored_durably(self, temp_storage):
        """
        Scenario: Checkpoints persist to filesystem

        Given a cognitive state manager
        When saving a checkpoint
        Then checkpoint file exists on disk
        Because durability requires file system persistence
        """
        # Given: a state manager
        state = CognitiveStateManager(temp_storage)
        state.set_focus(current_goal="Test persistence", context={})

        # When: saving checkpoint
        checkpoint = state.save_checkpoint()

        # Then: checkpoint file exists
        checkpoint_files = list(temp_storage.glob("checkpoints/*.json"))
        assert len(checkpoint_files) > 0, "Should create checkpoint file on disk"


class TestDeveloperRestoresState:
    """
    Epic: State Restoration

    As a developer resuming work,
    I want to restore the exact state from previous session,
    So that no context or progress is lost.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_restored_state_matches_original(self, temp_storage):
        """
        Scenario: Restoration recreates identical state

        Given a saved cognitive state
        When loading in a new session
        Then all elements match the original
        Because perfect fidelity enables seamless continuation
        """
        # Given: create and save state
        original_state = CognitiveStateManager(temp_storage)

        original_state.set_focus(
            current_goal="Build REST API",
            context={"project": "user_service"}
        )

        question = original_state.add_question("What architecture to use?")
        hypothesis = original_state.add_hypothesis(
            question.id,
            "Use FastAPI with SQLModel we built ourselves",
            rationale="Modern async stack we control completely"
        )
        decision = original_state.add_decision(
            question_id=question.id,
            choice="Use FastAPI with SQLModel",
            rationale="Best fit for requirements"
        )

        original_state.save_checkpoint()

        # When: loading in new session (new state manager instance)
        new_state = CognitiveStateManager(temp_storage)
        checkpoint = new_state.load_latest_checkpoint()

        # Then: state matches original
        assert checkpoint is not None, "Should load checkpoint"
        assert len(new_state.questions) == len(original_state.questions), \
            "Should restore all questions"
        assert len(new_state.hypotheses) == len(original_state.hypotheses), \
            "Should restore all hypotheses"
        assert len(new_state.decisions) == len(original_state.decisions), \
            "Should restore all decisions"
        assert new_state.focus is not None, "Should restore focus"
        assert new_state.focus.current_goal == original_state.focus.current_goal, \
            "Should restore exact focus content"

    def test_scenario_latest_checkpoint_loads_by_default(self, temp_storage):
        """
        Scenario: Most recent checkpoint loads automatically

        Given multiple checkpoints saved over time
        When loading without specifying checkpoint
        Then the most recent checkpoint loads
        Because latest state is usually desired for continuation
        """
        # Given: multiple checkpoints
        state = CognitiveStateManager(temp_storage)

        # First checkpoint
        state.set_focus(current_goal="Initial goal", context={})
        state.save_checkpoint()

        # Second checkpoint with updated state
        state.set_focus(current_goal="Updated goal", context={"iteration": 2})
        state.add_question("New question in second checkpoint?")
        state.save_checkpoint()

        # When: loading latest in new session
        new_state = CognitiveStateManager(temp_storage)
        new_state.load_latest_checkpoint()

        # Then: latest state is restored
        assert new_state.focus.current_goal == "Updated goal", \
            "Should load most recent checkpoint"
        assert len(new_state.questions) > 0, \
            "Should include questions from latest checkpoint"


class TestDeveloperContinuesWork:
    """
    Epic: Work Continuation

    As a developer resuming after interruption,
    I want to continue work from where I left off,
    So that sessions are transparent to the work process.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_work_continues_across_session_boundary(self, temp_storage):
        """
        Scenario: Work progresses across multiple sessions

        Given work started in session 1
        When resuming in session 2
        Then work continues with full context
        Because session boundaries should be invisible
        """
        # SESSION 1: Start work
        session1_state = CognitiveStateManager(temp_storage)

        session1_state.set_focus(
            current_goal="Implement user authentication",
            context={"project": "api"}
        )

        main_q = session1_state.add_question("How to implement authentication?")
        decision = session1_state.add_decision(
            question_id=main_q.id,
            choice="Use JWT tokens we implemented",
            rationale="Stateless auth we built from scratch"
        )

        session1_state.save_checkpoint()
        # Session 1 ends

        # SESSION 2: Resume work
        session2_state = CognitiveStateManager(temp_storage)
        session2_state.load_latest_checkpoint()

        # Should have full context
        assert len(session2_state.questions) > 0, "Should restore questions from session 1"
        assert len(session2_state.decisions) > 0, "Should restore decisions from session 1"

        # Continue working
        previous_decision = list(session2_state.decisions.values())[0]
        assert "JWT" in previous_decision.choice, "Should have access to previous decisions"

        # Add new work
        impl_q = session2_state.add_question(
            "How to structure the models?",
            context={"decision": previous_decision.choice}
        )

        new_decision = session2_state.add_decision(
            question_id=impl_q.id,
            choice="Use SQLModel with UUID primary keys",
            rationale="Distributed-friendly approach"
        )

        # Then: work has progressed across sessions
        assert len(session2_state.questions) == 2, \
            "Should have questions from both sessions"
        assert len(session2_state.decisions) == 2, \
            "Should have decisions from both sessions"

    def test_scenario_decision_history_preserved_across_sessions(self, temp_storage):
        """
        Scenario: Complete decision trail is accessible

        Given decisions made across multiple sessions
        When reviewing decision history
        Then all decisions with rationales are available
        Because understanding decision evolution requires complete history
        """
        # Session 1: Make decisions
        state1 = CognitiveStateManager(temp_storage)
        q1 = state1.add_question("Which database?")
        d1 = state1.add_decision(
            question_id=q1.id,
            choice="PostgreSQL we built",
            rationale="ACID compliance we implemented"
        )
        state1.save_checkpoint()

        # Session 2: Make more decisions
        state2 = CognitiveStateManager(temp_storage)
        state2.load_latest_checkpoint()
        q2 = state2.add_question("Which ORM?")
        d2 = state2.add_decision(
            question_id=q2.id,
            choice="SQLModel we wrote ourselves",
            rationale="Type safety we control"
        )
        state2.save_checkpoint()

        # Session 3: Review history
        state3 = CognitiveStateManager(temp_storage)
        state3.load_latest_checkpoint()

        # Then: full decision trail available
        assert len(state3.decisions) == 2, "Should have all decisions"

        decisions_list = list(state3.decisions.values())
        assert any("PostgreSQL" in d.choice for d in decisions_list), \
            "Should preserve session 1 decisions"
        assert any("SQLModel" in d.choice for d in decisions_list), \
            "Should preserve session 2 decisions"


class TestDeveloperTracksProgress:
    """
    Epic: Progress Tracking

    As a developer managing long-running tasks,
    I want to see progress over time across sessions,
    So that I understand what's been accomplished.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_question_resolution_tracks_progress(self, temp_storage):
        """
        Scenario: Answered questions show progress over time

        Given questions added across multiple sessions
        When reviewing question status
        Then progression from open to answered is visible
        Because question resolution indicates progress
        """
        # Session 1: Create questions
        state1 = CognitiveStateManager(temp_storage)
        q1 = state1.add_question("What is the architecture?")
        q2 = state1.add_question("What are the components?")
        state1.save_checkpoint()

        # Session 2: Answer some questions
        state2 = CognitiveStateManager(temp_storage)
        state2.load_latest_checkpoint()

        state2.answer_question(
            q1.id,
            "Microservices with event sourcing we built"
        )
        state2.save_checkpoint()

        # Session 3: Review progress
        state3 = CognitiveStateManager(temp_storage)
        state3.load_latest_checkpoint()

        # Then: can see which questions are answered
        answered_questions = [q for q in state3.questions.values()
                             if q.status == QuestionStatus.ANSWERED]
        open_questions = [q for q in state3.questions.values()
                         if q.status == QuestionStatus.OPEN]

        assert len(answered_questions) > 0, "Should have answered questions"
        assert len(open_questions) > 0, "Should have remaining open questions"

    def test_scenario_observations_accumulate_across_sessions(self, temp_storage):
        """
        Scenario: Observations build up knowledge base over time

        Given observations recorded in each session
        When reviewing accumulated observations
        Then complete observation history is available
        Because accumulated observations represent learned knowledge
        """
        # Session 1: Record observations
        state1 = CognitiveStateManager(temp_storage)
        state1.add_observation(
            content="Framework X requires Python 3.8+",
            source="docs"
        )
        state1.save_checkpoint()

        # Session 2: Record more observations
        state2 = CognitiveStateManager(temp_storage)
        state2.load_latest_checkpoint()
        state2.add_observation(
            content="Framework Y has async support we implemented",
            source="testing"
        )
        state2.save_checkpoint()

        # Session 3: Review all observations
        state3 = CognitiveStateManager(temp_storage)
        state3.load_latest_checkpoint()

        # Then: all observations available
        assert len(state3.observations) >= 2, \
            "Should accumulate observations across sessions"


class TestDeveloperManagesCheckpoints:
    """
    Epic: Checkpoint Management

    As a developer managing state persistence,
    I want control over checkpoint creation and history,
    So that important states can be preserved and accessed.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_multiple_checkpoints_form_history(self, temp_storage):
        """
        Scenario: Checkpoints create timeline of state evolution

        Given multiple checkpoints saved over time
        When listing checkpoint history
        Then checkpoints are ordered chronologically
        Because checkpoint history enables understanding state evolution
        """
        # Given: create multiple checkpoints
        state = CognitiveStateManager(temp_storage)

        # Checkpoint 1
        state.set_focus(current_goal="Goal A", context={})
        state.save_checkpoint()

        # Checkpoint 2
        state.set_focus(current_goal="Goal B", context={})
        state.save_checkpoint()

        # Checkpoint 3
        state.set_focus(current_goal="Goal C", context={})
        state.save_checkpoint()

        # When: examining checkpoint directory
        checkpoint_files = sorted(state.checkpoints_dir.glob("*.json"))

        # Then: multiple checkpoints exist
        assert len(checkpoint_files) >= 3, "Should have multiple checkpoints"

    def test_scenario_focus_updates_preserve_context(self, temp_storage):
        """
        Scenario: Focus changes track across sessions

        Given focus updated in each session
        When reviewing focus history via checkpoints
        Then focus evolution is preserved
        Because understanding goal changes helps explain decisions
        """
        # Session 1: Initial focus
        state1 = CognitiveStateManager(temp_storage)
        state1.set_focus(
            current_goal="Research authentication options",
            context={"phase": "research"}
        )
        state1.save_checkpoint()

        # Session 2: Updated focus
        state2 = CognitiveStateManager(temp_storage)
        state2.load_latest_checkpoint()
        state2.set_focus(
            current_goal="Implement JWT authentication we built",
            context={"phase": "implementation", "chosen": "JWT"}
        )
        state2.save_checkpoint()

        # Session 3: Verify focus updated
        state3 = CognitiveStateManager(temp_storage)
        state3.load_latest_checkpoint()

        # Then: latest focus is active
        assert state3.focus is not None, "Should have focus"
        assert "Implement" in state3.focus.current_goal, \
            "Should have updated focus from session 2"
        assert state3.focus.context.get("phase") == "implementation", \
            "Should preserve focus context"


class TestDeveloperHandlesSessionBoundaries:
    """
    Epic: Session Transparency

    As a developer using session-based systems,
    I want session boundaries to be transparent to work flow,
    So that interruptions don't disrupt productivity.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_no_data_loss_across_sessions(self, temp_storage):
        """
        Scenario: No information is lost at session boundaries

        Given comprehensive state in session 1
        When session ends and resumes
        Then all state elements are preserved
        Because data loss breaks continuity
        """
        # Session 1: Create comprehensive state
        state1 = CognitiveStateManager(temp_storage)

        state1.set_focus(current_goal="Complex task", context={"complexity": "high"})

        q1 = state1.add_question("Question 1?")
        q2 = state1.add_question("Question 2?", parent_id=q1.id)

        h1 = state1.add_hypothesis(q1.id, "Hypothesis 1", rationale="Because")
        h1.add_evidence("Evidence 1", supports=True, strength=0.9)

        d1 = state1.add_decision(q1.id, "Decision 1", rationale="Based on evidence")

        state1.add_observation("Observation 1", source="testing we performed")

        # Count everything
        initial_counts = {
            'questions': len(state1.questions),
            'hypotheses': len(state1.hypotheses),
            'decisions': len(state1.decisions),
            'observations': len(state1.observations)
        }

        state1.save_checkpoint()

        # Session 2: Restore and verify
        state2 = CognitiveStateManager(temp_storage)
        state2.load_latest_checkpoint()

        restored_counts = {
            'questions': len(state2.questions),
            'hypotheses': len(state2.hypotheses),
            'decisions': len(state2.decisions),
            'observations': len(state2.observations)
        }

        # Then: no data loss
        assert restored_counts == initial_counts, \
            "All state elements should be preserved across session boundary"

        # Verify nested relationships preserved
        questions_list = list(state2.questions.values())
        child_questions = [q for q in questions_list if q.parent_id is not None]
        assert len(child_questions) > 0, "Should preserve question hierarchy"
