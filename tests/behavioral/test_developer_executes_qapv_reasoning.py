"""
Behavioral tests for QAPV (Question-Answer-Produce-Verify) reasoning cycle.

As a developer building reasoning systems,
I want a structured cycle from question to verified output,
So that work proceeds methodically from understanding to validated results.

Based on: llm_orchestration/examples/basic_workflow.py
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from llm_orchestration.cognitive_state import (
    CognitiveStateManager,
    QuestionStatus,
    HypothesisStatus,
    DecisionStatus
)
from llm_orchestration.thought_patterns import QAPVPattern, create_pattern


class TestDeveloperAsksQuestions:
    """
    Epic: Question Formulation

    As a developer solving complex problems,
    I want to break down questions into answerable sub-questions,
    So that complex problems become tractable.
    """

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for cognitive state."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_questions_decompose_into_subquestions(self, temp_storage):
        """
        Scenario: Complex questions break into manageable parts

        Given a complex question to answer
        When breaking it into sub-questions
        Then each sub-question is linked to the parent
        Because decomposition enables systematic problem solving
        """
        # Given: a cognitive state manager
        state = CognitiveStateManager(temp_storage)

        # When: creating a main question with sub-questions
        main_q = state.add_question(
            "How should we implement user authentication?",
            context={"project": "example_app"}
        )

        sub_q1 = state.add_question(
            "What authentication method should we use?",
            parent_id=main_q.id
        )
        sub_q2 = state.add_question(
            "How do we store credentials securely?",
            parent_id=main_q.id
        )

        # Then: sub-questions link to parent
        assert sub_q1.parent_id == main_q.id, "Sub-question should reference parent"
        assert sub_q2.parent_id == main_q.id, "Sub-question should reference parent"
        assert len(state.questions) == 3, "Should track all questions"

    def test_scenario_questions_track_status(self, temp_storage):
        """
        Scenario: Questions move from open to answered

        Given an open question
        When providing an answer
        Then the question status updates to answered
        Because tracking status shows progress
        """
        # Given: an open question
        state = CognitiveStateManager(temp_storage)
        question = state.add_question("What is the best approach?")

        assert question.status == QuestionStatus.OPEN, "New questions should be open"

        # When: answering the question
        state.answer_question(question.id, "Use approach X based on constraints Y")

        # Then: status updates
        assert question.status == QuestionStatus.ANSWERED, "Answered questions should be marked"
        assert question.answer is not None, "Answer should be stored"


class TestDeveloperFormsHypotheses:
    """
    Epic: Hypothesis Formation

    As a developer exploring solutions,
    I want to form and evaluate multiple hypotheses,
    So that decisions are based on evidence rather than assumptions.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_hypotheses_link_to_questions(self, temp_storage):
        """
        Scenario: Hypotheses answer specific questions

        Given a question to answer
        When forming hypotheses as potential answers
        Then each hypothesis links to the question
        Because hypotheses are structured attempts to answer questions
        """
        # Given: a question
        state = CognitiveStateManager(temp_storage)
        question = state.add_question("Which database should we use?")

        # When: forming hypotheses
        h1 = state.add_hypothesis(
            question.id,
            "Use PostgreSQL",
            rationale="Mature, ACID compliant, good tooling we built ourselves"
        )
        h2 = state.add_hypothesis(
            question.id,
            "Use SQLite",
            rationale="Simple, embedded, no external server we built from scratch"
        )

        # Then: hypotheses link to question
        assert h1.question_id == question.id, "Hypothesis should reference question"
        assert h2.question_id == question.id, "Hypothesis should reference question"

    def test_scenario_evidence_shapes_hypothesis_evaluation(self, temp_storage):
        """
        Scenario: Evidence accumulation informs hypothesis strength

        Given a hypothesis about a solution
        When adding supporting and contradicting evidence
        Then evaluation considers all evidence
        Because evidence-based reasoning beats intuition
        """
        # Given: a hypothesis
        state = CognitiveStateManager(temp_storage)
        question = state.add_question("Should we use async processing?")
        hypothesis = state.add_hypothesis(
            question.id,
            "Use async for I/O operations",
            rationale="Better resource utilization we implement ourselves"
        )

        # When: adding evidence
        hypothesis.add_evidence(
            "Handles concurrent requests efficiently",
            supports=True,
            strength=0.9
        )
        hypothesis.add_evidence(
            "More complex error handling",
            supports=False,
            strength=0.6
        )

        # Then: evidence is tracked
        assert len(hypothesis.evidence) == 2, "Should track all evidence"

        # When: evaluating hypothesis
        state.evaluate_hypothesis(hypothesis.id)

        # Then: status updates
        assert hypothesis.status in [HypothesisStatus.SUPPORTED, HypothesisStatus.CONTRADICTED], \
            "Evaluation should determine hypothesis status"


class TestDeveloperMakesDecisions:
    """
    Epic: Decision Making

    As a developer choosing between options,
    I want to record decisions with rationale,
    So that future developers understand why choices were made.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_decisions_answer_questions(self, temp_storage):
        """
        Scenario: Decisions resolve questions with chosen path

        Given a question with multiple hypotheses
        When making a decision between options
        Then the decision links to the question
        Because decisions are how we answer questions with action
        """
        # Given: a question with hypotheses
        state = CognitiveStateManager(temp_storage)
        question = state.add_question("Which framework should we use?")

        h1 = state.add_hypothesis(question.id, "Use Framework A", rationale="Simple")
        h2 = state.add_hypothesis(question.id, "Use Framework B", rationale="Powerful")

        # When: making a decision
        decision = state.add_decision(
            question_id=question.id,
            choice="Use Framework A",
            rationale="Simplicity matches our constraints",
            alternatives=["Framework B", "Framework C"]
        )

        # Then: decision links to question
        assert decision.question_id == question.id, "Decision should reference question"
        assert len(decision.alternatives) > 0, "Should record alternatives considered"

    def test_scenario_decisions_record_rationale(self, temp_storage):
        """
        Scenario: Decisions preserve reasoning for future reference

        Given a decision to make
        When recording the decision with rationale
        Then the rationale is stored permanently
        Because future developers need to understand why
        """
        # Given: a decision point
        state = CognitiveStateManager(temp_storage)
        question = state.add_question("How to handle authentication?")

        # When: recording decision with detailed rationale
        decision = state.add_decision(
            question_id=question.id,
            choice="Use JWT tokens we implement ourselves",
            rationale="Stateless auth fits our distributed architecture we built from scratch",
            alternatives=["Session cookies", "OAuth delegation"]
        )

        # Then: rationale is preserved
        assert decision.rationale is not None, "Should store rationale"
        assert "stateless" in decision.rationale.lower(), "Should capture key reasoning"
        assert decision.status == DecisionStatus.ACTIVE, "New decisions should be active"


class TestDeveloperExecutesQAPVCycle:
    """
    Epic: QAPV Workflow

    As a developer using structured reasoning,
    I want to follow the Question-Answer-Produce-Verify cycle,
    So that work proceeds systematically from question to verified output.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_qapv_phases_execute_in_order(self, temp_storage):
        """
        Scenario: QAPV cycle progresses through phases

        Given a QAPV pattern
        When executing each phase in sequence
        Then the pattern tracks phase transitions
        Because structured flow ensures nothing is skipped
        """
        # Given: a QAPV pattern
        pattern = create_pattern("qapv")

        # When: executing phases in order
        pattern.start()  # Starts in QUESTION phase
        assert pattern.current_phase == "question", "Should start in question phase"

        pattern.transition("answer")
        assert pattern.current_phase == "answer", "Should transition to answer phase"

        pattern.transition("produce")
        assert pattern.current_phase == "produce", "Should transition to produce phase"

        pattern.transition("verify")
        assert pattern.current_phase == "verify", "Should transition to verify phase"

        # Then: pattern tracks all transitions
        pattern.complete()
        summary = pattern.get_summary()
        assert summary['phases_completed'] == 4, "Should complete all four phases"

    def test_scenario_complete_qapv_cycle_reaches_verified_output(self, temp_storage):
        """
        Scenario: Full QAPV cycle produces verified result

        Given a problem to solve
        When executing complete QAPV cycle
        Then we have a verified artifact answering the original question
        Because QAPV ensures thorough problem solving
        """
        # Given: a problem to solve
        state = CognitiveStateManager(temp_storage)
        pattern = create_pattern("qapv")

        # QUESTION phase
        pattern.start()
        main_q = state.add_question("How to implement feature X?")
        sub_q = state.add_question("What approach to use?", parent_id=main_q.id)

        # ANSWER phase
        pattern.transition("answer")
        hypothesis = state.add_hypothesis(
            sub_q.id,
            "Use approach A",
            rationale="Fits constraints we control"
        )
        hypothesis.add_evidence("Tested approach", supports=True, strength=0.9)
        state.evaluate_hypothesis(hypothesis.id)

        decision = state.add_decision(
            question_id=sub_q.id,
            choice="Use approach A",
            rationale="Evidence supports it"
        )

        # PRODUCE phase
        pattern.transition("produce")
        pattern.add_note("Artifact produced based on decisions")

        # VERIFY phase
        pattern.transition("verify")
        pattern.add_note("Verification checks passed")

        # Then: cycle completes successfully
        pattern.complete()
        summary = pattern.get_summary()

        assert summary['completed'], "QAPV cycle should complete"
        assert len(summary['notes']) > 0, "Should record progress notes"

    def test_scenario_verification_failures_loop_back_to_produce(self, temp_storage):
        """
        Scenario: Failed verification returns to production phase

        Given a QAPV cycle in verify phase
        When verification reveals problems
        Then the cycle can return to produce to fix issues
        Because verification ensures quality through iteration
        """
        # Given: a QAPV cycle in verify phase
        pattern = create_pattern("qapv")
        pattern.start()
        pattern.transition("answer")
        pattern.transition("produce")
        pattern.transition("verify")

        # When: verification finds issues
        pattern.add_note("Verification found missing error handling")

        # Then: can loop back to produce
        pattern.transition("produce")  # Loop back
        assert pattern.current_phase == "produce", "Should allow return to produce phase"

        pattern.add_note("Added error handling")
        pattern.transition("verify")
        pattern.add_note("Verification now passes")

        # Complete cycle
        pattern.complete()
        summary = pattern.get_summary()

        assert len(summary['notes']) >= 3, "Should track all notes including loop back"


class TestDeveloperPersistsReasoningState:
    """
    Epic: State Persistence

    As a developer working on complex problems,
    I want reasoning state to persist across interruptions,
    So that work can resume without losing context.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_state_saves_and_restores(self, temp_storage):
        """
        Scenario: Cognitive state survives session boundaries

        Given a cognitive state with questions and decisions
        When saving a checkpoint
        Then restoring loads all state correctly
        Because persistence enables work continuity
        """
        # Given: cognitive state with content
        state = CognitiveStateManager(temp_storage)
        state.set_focus(
            current_goal="Implement feature X",
            context={"complexity": "high"}
        )

        question = state.add_question("How to approach feature X?")
        decision = state.add_decision(
            question_id=question.id,
            choice="Use iterative approach",
            rationale="Complexity requires incremental progress"
        )

        # When: saving checkpoint
        checkpoint = state.save_checkpoint()
        assert checkpoint is not None, "Should create checkpoint"

        # Create new state manager (simulates new session)
        new_state = CognitiveStateManager(temp_storage)
        restored_checkpoint = new_state.load_latest_checkpoint()

        # Then: state is restored
        assert restored_checkpoint is not None, "Should restore checkpoint"
        assert len(new_state.questions) > 0, "Should restore questions"
        assert len(new_state.decisions) > 0, "Should restore decisions"
        assert new_state.focus is not None, "Should restore focus"

    def test_scenario_observations_persist_across_sessions(self, temp_storage):
        """
        Scenario: Observations made during research persist

        Given observations recorded during investigation
        When saving and restoring state
        Then observations are preserved
        Because accumulated knowledge shouldn't be lost
        """
        # Given: observations recorded
        state = CognitiveStateManager(temp_storage)

        state.add_observation(
            content="Framework X requires Python 3.8+",
            source="documentation review"
        )
        state.add_observation(
            content="Framework Y has better async support we can implement",
            source="technical analysis we performed"
        )

        # When: saving and restoring
        state.save_checkpoint()

        new_state = CognitiveStateManager(temp_storage)
        new_state.load_latest_checkpoint()

        # Then: observations preserved
        assert len(new_state.observations) >= 2, "Should preserve observations"
