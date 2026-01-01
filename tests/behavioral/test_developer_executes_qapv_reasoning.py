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
from llm_orchestration.thought_patterns import QAPVPattern, QAPVPhase, create_pattern


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
        main_q = state.ask_question(
            "How should we implement user authentication?",
            context="example_app project"
        )

        sub_q1 = state.ask_question(
            "What authentication method should we use?",
            parent_question_id=main_q.id
        )
        sub_q2 = state.ask_question(
            "How do we store credentials securely?",
            parent_question_id=main_q.id
        )

        # Then: sub-questions are tracked in parent's sub_questions list
        assert sub_q1.id in main_q.sub_questions, "Parent should track sub-question"
        assert sub_q2.id in main_q.sub_questions, "Parent should track sub-question"
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
        question = state.ask_question("What is the best approach?")

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
        question = state.ask_question("Which database should we use?")

        # When: forming hypotheses
        h1 = state.form_hypothesis(
            "Use PostgreSQL",
            rationale="Mature, ACID compliant, good tooling we built ourselves",
            for_question_id=question.id
        )
        h2 = state.form_hypothesis(
            "Use SQLite",
            rationale="Simple, embedded, no external server we built from scratch",
            for_question_id=question.id
        )

        # Then: hypotheses link to question
        assert h1.related_question == question.id, "Hypothesis should reference question"
        assert h2.related_question == question.id, "Hypothesis should reference question"

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
        question = state.ask_question("Should we use async processing?")
        hypothesis = state.form_hypothesis(
            "Use async for I/O operations",
            rationale="Better resource utilization we implement ourselves",
            for_question_id=question.id
        )

        # When: adding evidence via state manager
        state.add_evidence(
            hypothesis.id,
            "Handles concurrent requests efficiently",
            supports=True
        )
        state.add_evidence(
            hypothesis.id,
            "More complex error handling",
            supports=False
        )

        # Then: evidence is tracked in supporting/contradicting lists
        total_evidence = len(hypothesis.supporting_evidence) + len(hypothesis.contradicting_evidence)
        assert total_evidence == 2, "Should track all evidence"

        # When: updating hypothesis confidence based on evidence
        # More supporting than contradicting evidence → higher confidence
        state.update_hypothesis_confidence(hypothesis.id, 0.8, reason="More supporting evidence")

        # Then: status updates based on confidence
        assert hypothesis.status == HypothesisStatus.SUPPORTED, \
            "High confidence should set status to SUPPORTED"


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
        question = state.ask_question("Which framework should we use?")

        h1 = state.form_hypothesis("Use Framework A", for_question_id=question.id, rationale="Simple")
        h2 = state.form_hypothesis("Use Framework B", for_question_id=question.id, rationale="Powerful")

        # When: making a decision
        decision = state.make_decision(
            from_question_id=question.id,
            decision="Use Framework A",
            rationale="Simplicity matches our constraints",
            alternatives=["Framework B", "Framework C"]
        )

        # Then: decision links to question
        assert decision.from_question == question.id, "Decision should reference question"
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
        question = state.ask_question("How to handle authentication?")

        # When: recording decision with detailed rationale
        decision = state.make_decision(
            from_question_id=question.id,
            decision="Use JWT tokens we implement ourselves",
            rationale="Stateless auth fits our distributed architecture we built from scratch",
            alternatives=["Session cookies", "OAuth delegation"]
        )

        # Then: rationale is preserved
        assert decision.rationale is not None, "Should store rationale"
        assert "stateless" in decision.rationale.lower(), "Should capture key reasoning"
        assert decision.status == DecisionStatus.TENTATIVE, "New decisions should be tentative"


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
        # Given: a QAPV pattern with cognitive state
        state = CognitiveStateManager(temp_storage)
        pattern = create_pattern("qapv", state, goal="Test goal")

        # Pattern starts in QUESTION phase automatically
        assert pattern.current_phase == QAPVPhase.QUESTION, "Should start in question phase"

        # Set question to satisfy phase requirements
        pattern.set_question("What should we build?", "Working feature")

        # When: executing phases in order via advance()
        pattern.advance()
        assert pattern.current_phase == QAPVPhase.ANSWER, "Should transition to answer phase"

        # Record a decision to satisfy ANSWER phase requirements
        pattern.record_decision(
            decision="Use approach A",
            rationale="Best fit for requirements we control"
        )

        pattern.advance()
        assert pattern.current_phase == QAPVPhase.PRODUCE, "Should transition to produce phase"

        # Set artifact to satisfy PRODUCE phase requirements
        pattern.set_artifact({"code": "implementation"}, "Feature implemented")

        pattern.advance()
        assert pattern.current_phase == QAPVPhase.VERIFY, "Should transition to verify phase"

        # Then: pattern tracks progress through phases
        progress = pattern.get_progress()
        assert progress['current_phase'] == 'VERIFY', "Should be in verify phase"
        assert progress['steps_taken'] > 0, "Should track steps taken"

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
        pattern = create_pattern("qapv", state, goal="Implement feature X")

        # QUESTION phase - pattern starts here automatically
        assert pattern.current_phase == QAPVPhase.QUESTION
        pattern.set_question("How to implement feature X?", "Working, tested feature")

        # ANSWER phase
        pattern.advance()
        assert pattern.current_phase == QAPVPhase.ANSWER

        hypothesis = state.form_hypothesis(
            "Use approach A",
            rationale="Fits constraints we control",
            for_question_id=pattern.question.id
        )
        state.add_evidence(hypothesis.id, "Tested approach works well", supports=True)
        state.update_hypothesis_confidence(hypothesis.id, 0.9, reason="Evidence supports it")

        pattern.record_decision(
            decision="Use approach A",
            rationale="Evidence supports it"
        )

        # PRODUCE phase
        pattern.advance()
        assert pattern.current_phase == QAPVPhase.PRODUCE
        pattern.set_artifact({"feature": "implemented"}, "Feature X implemented")
        pattern.record_step("implementation", "Code written and ready for verification")

        # VERIFY phase
        pattern.advance()
        assert pattern.current_phase == QAPVPhase.VERIFY
        pattern.record_verification(passed=True, details={"tests": "all passing"})

        # Complete cycle by advancing from VERIFY with passed=True
        pattern.advance()
        assert pattern.current_phase == QAPVPhase.COMPLETE, "Should reach COMPLETE phase"

        # Then: cycle completes successfully
        progress = pattern.get_progress()
        assert progress['current_phase'] == 'COMPLETE', "QAPV cycle should complete"
        assert progress['steps_taken'] > 0, "Should record progress steps"

    def test_scenario_verification_failures_loop_back_to_question(self, temp_storage):
        """
        Scenario: Failed verification returns to question phase for re-evaluation

        Given a QAPV cycle in verify phase
        When verification reveals problems
        Then the cycle loops back to question phase with new understanding
        Because verification failures require re-examining our approach
        """
        # Given: a QAPV cycle progressed to verify phase
        state = CognitiveStateManager(temp_storage)
        pattern = create_pattern("qapv", state, goal="Build feature with quality")

        # Progress through phases to VERIFY
        pattern.set_question("How to build this feature?", "Passing tests")
        pattern.advance()  # -> ANSWER

        pattern.record_decision(
            decision="Use quick approach",
            rationale="Faster implementation we control"
        )
        pattern.advance()  # -> PRODUCE

        pattern.set_artifact({"code": "v1"}, "First implementation")
        pattern.advance()  # -> VERIFY

        # When: verification finds issues (fails)
        pattern.record_verification(
            passed=False,
            details={"error": "Missing error handling"}
        )
        pattern.record_step("verification_issue", "Found missing error handling")

        # Then: advancing loops back to QUESTION phase for re-evaluation
        pattern.advance()
        assert pattern.current_phase == QAPVPhase.QUESTION, \
            "Failed verification should loop back to question phase"

        # Can now start a new iteration with better understanding
        progress = pattern.get_progress()
        assert progress['steps_taken'] > 3, "Should track all steps including loop back"


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
        # Given: cognitive state with content (auto-saves on each mutation)
        state = CognitiveStateManager(temp_storage)
        state.set_focus("Implement feature X")

        question = state.ask_question("How to approach feature X?")
        decision = state.make_decision(
            from_question_id=question.id,
            decision="Use iterative approach",
            rationale="Complexity requires incremental progress"
        )

        # State auto-saves, capture checkpoint data
        checkpoint = state.checkpoint()
        assert checkpoint is not None, "Should create checkpoint"

        # Create new state manager (simulates new session - auto-loads on init)
        new_state = CognitiveStateManager(temp_storage)

        # Then: state is restored automatically
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
        # Given: observations recorded (auto-saves on each mutation)
        state = CognitiveStateManager(temp_storage)

        state.record_observation(
            observation="Framework X requires Python 3.8+",
            source="documentation review"
        )
        state.record_observation(
            observation="Framework Y has better async support we can implement",
            source="technical analysis we performed"
        )

        # When: creating new session (auto-loads on init)
        new_state = CognitiveStateManager(temp_storage)

        # Then: observations preserved
        assert len(new_state.observations) >= 2, "Should preserve observations"
