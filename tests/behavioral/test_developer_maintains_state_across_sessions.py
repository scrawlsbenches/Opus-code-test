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

        state.set_focus("Implement authentication")

        question = state.ask_question("How to implement OAuth?")
        decision = state.make_decision(
            decision="Use custom OAuth implementation we built",
            rationale="Full control over security we built from scratch",
            from_question_id=question.id
        )

        state.record_observation(
            observation="OAuth 2.0 requires HTTPS in production",
            source="security documentation we wrote"
        )

        # When: creating checkpoint (state auto-saves on each mutation)
        checkpoint = state.checkpoint()

        # Then: checkpoint is created and contains state
        assert checkpoint is not None, "Should create checkpoint"
        assert 'timestamp' in checkpoint, "Checkpoint should have timestamp"
        assert 'focus' in checkpoint, "Checkpoint should contain focus"
        assert 'questions' in checkpoint, "Checkpoint should contain questions"

    def test_scenario_checkpoints_stored_durably(self, temp_storage):
        """
        Scenario: State persists to filesystem

        Given a cognitive state manager
        When mutating state
        Then state file exists on disk
        Because durability requires file system persistence
        """
        # Given: a state manager
        state = CognitiveStateManager(temp_storage)
        state.set_focus("Test persistence")

        # When: state mutates (auto-saved)
        state.ask_question("Test question?")

        # Then: state file exists on disk
        state_file = temp_storage / "current_state.json"
        assert state_file.exists(), "Should create state file on disk"


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
        # Given: create and save state (auto-saved on each mutation)
        original_state = CognitiveStateManager(temp_storage)

        original_state.set_focus("Build REST API")

        question = original_state.ask_question("What architecture to use?")
        hypothesis = original_state.form_hypothesis(
            "Use FastAPI with SQLModel we built ourselves",
            rationale="Modern async stack we control completely",
            for_question_id=question.id
        )
        decision = original_state.make_decision(
            decision="Use FastAPI with SQLModel",
            rationale="Best fit for requirements",
            from_question_id=question.id
        )

        # State auto-saves on each mutation, capture counts
        original_question_count = len(original_state.questions)
        original_hypothesis_count = len(original_state.hypotheses)
        original_decision_count = len(original_state.decisions)
        original_focus_desc = original_state.focus.description

        # When: loading in new session (new state manager instance auto-loads)
        new_state = CognitiveStateManager(temp_storage)

        # Then: state matches original
        assert len(new_state.questions) == original_question_count, \
            "Should restore all questions"
        assert len(new_state.hypotheses) == original_hypothesis_count, \
            "Should restore all hypotheses"
        assert len(new_state.decisions) == original_decision_count, \
            "Should restore all decisions"
        assert new_state.focus is not None, "Should restore focus"
        assert new_state.focus.description == original_focus_desc, \
            "Should restore exact focus content"

    def test_scenario_latest_state_loads_by_default(self, temp_storage):
        """
        Scenario: Most recent state loads automatically

        Given multiple mutations over time
        When creating a new session
        Then the most recent state loads
        Because latest state is usually desired for continuation
        """
        # Given: state evolves over time
        state = CognitiveStateManager(temp_storage)

        # Initial state
        state.set_focus("Initial goal")

        # Updated state
        state.set_focus("Updated goal")
        state.ask_question("New question in second state?")

        # When: loading latest in new session (auto-loads on init)
        new_state = CognitiveStateManager(temp_storage)

        # Then: latest state is restored
        assert new_state.focus.description == "Updated goal", \
            "Should load most recent state"
        assert len(new_state.questions) > 0, \
            "Should include questions from latest state"


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

        session1_state.set_focus("Implement user authentication")

        main_q = session1_state.ask_question("How to implement authentication?")
        decision = session1_state.make_decision(
            decision="Use JWT tokens we implemented",
            rationale="Stateless auth we built from scratch",
            from_question_id=main_q.id
        )

        # State auto-saves, session 1 ends

        # SESSION 2: Resume work (auto-loads on init)
        session2_state = CognitiveStateManager(temp_storage)

        # Should have full context
        assert len(session2_state.questions) > 0, "Should restore questions from session 1"
        assert len(session2_state.decisions) > 0, "Should restore decisions from session 1"

        # Continue working
        previous_decision = list(session2_state.decisions.values())[0]
        assert "JWT" in previous_decision.decision, "Should have access to previous decisions"

        # Add new work
        impl_q = session2_state.ask_question(
            "How to structure the models?",
            context=previous_decision.decision
        )

        new_decision = session2_state.make_decision(
            decision="Use SQLModel with UUID primary keys",
            rationale="Distributed-friendly approach",
            from_question_id=impl_q.id
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
        q1 = state1.ask_question("Which database?")
        d1 = state1.make_decision(
            decision="PostgreSQL we built",
            rationale="ACID compliance we implemented",
            from_question_id=q1.id
        )

        # Session 2: Make more decisions (auto-loads on init)
        state2 = CognitiveStateManager(temp_storage)
        q2 = state2.ask_question("Which ORM?")
        d2 = state2.make_decision(
            decision="SQLModel we wrote ourselves",
            rationale="Type safety we control",
            from_question_id=q2.id
        )

        # Session 3: Review history (auto-loads on init)
        state3 = CognitiveStateManager(temp_storage)

        # Then: full decision trail available
        assert len(state3.decisions) == 2, "Should have all decisions"

        decisions_list = list(state3.decisions.values())
        assert any("PostgreSQL" in d.decision for d in decisions_list), \
            "Should preserve session 1 decisions"
        assert any("SQLModel" in d.decision for d in decisions_list), \
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
        q1 = state1.ask_question("What is the architecture?")
        q2 = state1.ask_question("What are the components?")
        q1_id = q1.id

        # Session 2: Answer some questions (auto-loads on init)
        state2 = CognitiveStateManager(temp_storage)

        state2.answer_question(
            q1_id,
            "Microservices with event sourcing we built"
        )

        # Session 3: Review progress (auto-loads on init)
        state3 = CognitiveStateManager(temp_storage)

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
        state1.record_observation(
            observation="Framework X requires Python 3.8+",
            source="docs"
        )

        # Session 2: Record more observations (auto-loads on init)
        state2 = CognitiveStateManager(temp_storage)
        state2.record_observation(
            observation="Framework Y has async support we implemented",
            source="testing"
        )

        # Session 3: Review all observations (auto-loads on init)
        state3 = CognitiveStateManager(temp_storage)

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

    def test_scenario_state_evolves_over_time(self, temp_storage):
        """
        Scenario: State evolution is tracked over time

        Given multiple state changes over time
        When reviewing current state
        Then the latest state reflects all changes
        Because state evolution enables understanding progress
        """
        # Given: create multiple state changes
        state = CognitiveStateManager(temp_storage)

        # State changes
        state.set_focus("Goal A")
        state.ask_question("Question A?")

        state.set_focus("Goal B")
        state.ask_question("Question B?")

        state.set_focus("Goal C")
        state.ask_question("Question C?")

        # When: examining final state
        # Then: state reflects all changes
        assert state.focus.description == "Goal C", "Should have latest focus"
        assert len(state.questions) >= 3, "Should have all questions"

    def test_scenario_focus_updates_preserve_context(self, temp_storage):
        """
        Scenario: Focus changes track across sessions

        Given focus updated in each session
        When reviewing focus in new session
        Then latest focus is available
        Because understanding goal changes helps explain decisions
        """
        # Session 1: Initial focus
        state1 = CognitiveStateManager(temp_storage)
        state1.set_focus("Research authentication options")

        # Session 2: Updated focus (auto-loads on init)
        state2 = CognitiveStateManager(temp_storage)
        state2.set_focus("Implement JWT authentication we built")

        # Session 3: Verify focus updated (auto-loads on init)
        state3 = CognitiveStateManager(temp_storage)

        # Then: latest focus is active
        assert state3.focus is not None, "Should have focus"
        assert "Implement" in state3.focus.description, \
            "Should have updated focus from session 2"


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

        state1.set_focus("Complex task")

        q1 = state1.ask_question("Question 1?")
        q2 = state1.ask_question("Question 2?")

        h1 = state1.form_hypothesis("Hypothesis 1", rationale="Because", for_question_id=q1.id)
        state1.add_evidence(h1.id, "Evidence 1", supports=True)

        d1 = state1.make_decision(
            decision="Decision 1",
            rationale="Based on evidence",
            from_question_id=q1.id
        )

        state1.record_observation(observation="Observation 1", source="testing we performed")

        # Count everything
        initial_counts = {
            'questions': len(state1.questions),
            'hypotheses': len(state1.hypotheses),
            'decisions': len(state1.decisions),
            'observations': len(state1.observations)
        }

        # Session 2: Restore and verify (auto-loads on init)
        state2 = CognitiveStateManager(temp_storage)

        restored_counts = {
            'questions': len(state2.questions),
            'hypotheses': len(state2.hypotheses),
            'decisions': len(state2.decisions),
            'observations': len(state2.observations)
        }

        # Then: no data loss
        assert restored_counts == initial_counts, \
            "All state elements should be preserved across session boundary"

        # Verify hypothesis evidence preserved
        hypotheses_list = list(state2.hypotheses.values())
        assert len(hypotheses_list) > 0, "Should restore hypotheses"
        assert len(hypotheses_list[0].supporting_evidence) > 0, "Should preserve evidence"
