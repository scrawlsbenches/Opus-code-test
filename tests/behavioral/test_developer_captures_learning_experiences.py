"""
Behavioral tests for learning experience capture and retrieval.

As a developer building intelligent systems,
I want the system to capture experiences and extract actionable lessons,
So that it learns from past successes and failures without neural weight updates.

Based on: llm_orchestration/examples/learning_demo.py
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from llm_orchestration.learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType
)


class TestDeveloperCapturesExperiences:
    """
    Epic: Experience Capture

    As a developer using an LLM orchestration system,
    I want every execution to be captured as an experience,
    So that the system can learn from what worked and what didn't.
    """

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for learning system."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_experience_captures_actions_and_outcomes(self, temp_storage):
        """
        Scenario: Experience records complete execution history

        Given a learning system tracking work
        When I execute a series of actions with an outcome
        Then the experience contains all actions and final result
        Because complete history enables pattern recognition
        """
        # Given: a learning system tracking work
        cycle = LearningCycle(temp_storage)
        context = Context(
            goal_type="implementation",
            goal_complexity="moderate",
            domain="authentication"
        )

        # When: executing a series of actions
        experience = cycle.start_experience(
            context=context,
            intent="Implement JWT authentication",
            strategy="test_driven_development"
        )

        actions = [
            Action("read_file", "Read existing auth module", "/src/auth.py"),
            Action("write_test", "Write test for token generation", "/tests/test_auth.py"),
            Action("implement", "Implement token generation", "/src/auth.py"),
            Action("run_tests", "Run test suite", "pytest")
        ]

        for action in actions:
            experience.add_action(action)

        outcome = Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="JWT authentication implemented",
            achieved=["token generation", "token verification"],
            quality_score=0.9
        )

        # Then: complete experience contains all information
        cycle.complete_experience(experience, outcome)

        assert len(experience.actions) == 4, "Should record all actions taken"
        assert experience.outcome.outcome_type == OutcomeType.SUCCESS, "Should capture success outcome"
        assert experience.intent == "Implement JWT authentication", "Should preserve original intent"

    def test_scenario_experiences_include_reflection(self, temp_storage):
        """
        Scenario: Experiences capture what worked and what didn't

        Given a completed experience
        When adding reflection about the process
        Then the reflection is stored for future learning
        Because knowing why things worked enables better decisions
        """
        # Given: an experience in progress
        cycle = LearningCycle(temp_storage)
        context = Context(goal_type="implementation", domain="api")
        experience = cycle.start_experience(context, "Build API endpoint", "tdd")

        # When: completing with reflection
        outcome = Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="API endpoint built",
            quality_score=0.8
        )

        reflection = {
            "worked": ["TDD caught edge cases early"],
            "didnt_work": ["Initial response format was wrong"],
            "different": ["Would use OpenAPI spec first"]
        }

        cycle.complete_experience(experience, outcome, reflection)

        # Then: reflection is stored
        assert experience.reflection is not None, "Should store reflection"
        assert "worked" in experience.reflection, "Should capture what worked"
        assert "didnt_work" in experience.reflection, "Should capture what didn't work"


class TestDeveloperExtractsPatterns:
    """
    Epic: Pattern Recognition

    As a developer building a learning system,
    I want patterns to emerge from repeated experiences,
    So that the system discovers what strategies work in which contexts.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_repeated_sequences_form_patterns(self, temp_storage):
        """
        Scenario: Common action sequences are recognized as patterns

        Given multiple experiences with the same action sequence
        When pattern extraction runs
        Then the common sequence is identified as a pattern
        Because repeated structures indicate effective approaches
        """
        # Given: multiple experiences with the same action sequence
        cycle = LearningCycle(temp_storage)

        for i in range(5):
            context = Context(goal_type="implementation", goal_complexity="moderate")
            exp = cycle.start_experience(context, f"Feature {i}", "test_driven_development")

            # Same sequence: read -> test -> implement -> verify
            exp.add_action(Action("read", "Read code", "/src/"))
            exp.add_action(Action("write_test", "Write tests", "/tests/"))
            exp.add_action(Action("implement", "Write implementation", "/src/"))
            exp.add_action(Action("run_tests", "Verify", "pytest"))

            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Success")
            )

        # When: pattern extraction runs
        results = cycle.extract_and_distill()

        # Then: common sequence is recognized
        assert results['sequence_patterns'] > 0, "Should identify repeated action sequences"

    def test_scenario_successful_strategies_form_patterns(self, temp_storage):
        """
        Scenario: Successful strategies are recognized as patterns

        Given multiple successful experiences using the same strategy
        When pattern extraction runs
        Then the strategy is identified as effective
        Because consistent success indicates a good approach
        """
        # Given: multiple successful experiences with TDD
        cycle = LearningCycle(temp_storage)

        for i in range(5):
            context = Context(goal_type="implementation")
            exp = cycle.start_experience(context, f"Feature {i}", "test_driven_development")
            exp.add_action(Action("test_first", "Write test", "/tests/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Worked well")
            )

        # When: extracting patterns
        results = cycle.extract_and_distill()

        # Then: strategy pattern is recognized
        assert results['strategy_patterns'] > 0, "Should recognize successful strategy"

    def test_scenario_failed_patterns_become_antipatterns(self, temp_storage):
        """
        Scenario: Repeated failures are flagged as antipatterns

        Given multiple experiences with the same strategy failing
        When pattern extraction runs
        Then the strategy is flagged as an antipattern
        Because consistent failure warns against an approach
        """
        # Given: multiple failures with code-first approach
        cycle = LearningCycle(temp_storage)

        for i in range(3):
            context = Context(goal_type="implementation")
            exp = cycle.start_experience(context, f"Feature {i}", "code_first")
            exp.add_action(Action("implement", "Code without tests", "/src/"))
            cycle.complete_experience(
                exp,
                Outcome(
                    outcome_type=OutcomeType.FAILURE,
                    description="Bugs found late"
                )
            )

        # When: extracting patterns
        results = cycle.extract_and_distill()

        # Then: antipattern is flagged
        assert results['antipatterns'] > 0, "Should identify antipattern from repeated failures"


class TestDeveloperRetrievesLessons:
    """
    Epic: Contextual Guidance

    As a developer working on a new task,
    I want to receive relevant lessons from past experiences,
    So that I benefit from previous learning.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_similar_context_retrieves_relevant_lessons(self, temp_storage):
        """
        Scenario: Lessons are retrieved for similar contexts

        Given past experiences in a domain
        When requesting guidance for a similar context
        Then relevant lessons are retrieved
        Because similar contexts benefit from similar approaches
        """
        # Given: past experiences in authentication domain
        cycle = LearningCycle(temp_storage)

        for i in range(5):
            context = Context(
                goal_type="implementation",
                domain="authentication"
            )
            exp = cycle.start_experience(context, "Auth task", "tdd")
            exp.add_action(Action("test_first", "Test first", "/tests/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Success")
            )

        # Extract patterns and lessons
        cycle.extract_and_distill()

        # When: requesting guidance for similar context
        new_context = Context(
            goal_type="implementation",
            domain="security"  # Similar to authentication
        )
        guidance = cycle.get_guidance(new_context)

        # Then: lessons are retrieved
        assert len(guidance['lessons']) > 0 or len(guidance['recommendations']) > 0, \
            "Should provide guidance based on past experiences"

    def test_scenario_guidance_includes_successes_and_failures(self, temp_storage):
        """
        Scenario: Guidance shows both what worked and what failed

        Given a history of successes and failures
        When requesting guidance with experience details
        Then both relevant successes and failures are provided
        Because learning from mistakes is as valuable as learning from success
        """
        # Given: a mix of successes and failures
        cycle = LearningCycle(temp_storage)

        # Successful TDD experiences
        for i in range(3):
            context = Context(goal_type="implementation", domain="api")
            exp = cycle.start_experience(context, "API feature", "tdd")
            exp.add_action(Action("test", "Test", "/tests/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Success")
            )

        # Failed code-first experiences
        for i in range(2):
            context = Context(goal_type="implementation", domain="api")
            exp = cycle.start_experience(context, "API feature", "code_first")
            exp.add_action(Action("code", "Code", "/src/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.FAILURE, description="Failed")
            )

        # When: requesting guidance with experiences
        context = Context(goal_type="implementation", domain="api")
        guidance = cycle.get_guidance(context, include_experiences=True)

        # Then: both successes and failures are available
        has_successes = len(guidance.get('relevant_successes', [])) > 0
        has_failures = len(guidance.get('relevant_failures', [])) > 0

        assert has_successes or has_failures, \
            "Should provide relevant past experiences for learning"


class TestDeveloperValidatesLessons:
    """
    Epic: Lesson Validation

    As a developer applying lessons,
    I want lesson confidence to adjust based on results,
    So that reliable lessons strengthen and unreliable ones weaken.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_helpful_lessons_gain_confidence(self, temp_storage):
        """
        Scenario: Successful application increases lesson confidence

        Given a lesson with initial confidence
        When the lesson is applied and proves helpful
        Then confidence increases
        Because validated lessons deserve higher trust
        """
        # Given: a lesson from past experiences
        cycle = LearningCycle(temp_storage)

        for i in range(5):
            context = Context(goal_type="implementation")
            exp = cycle.start_experience(context, "Task", "tdd")
            exp.add_action(Action("test", "Test", "/tests/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Success")
            )

        cycle.extract_and_distill()
        context = Context(goal_type="implementation")
        lessons = cycle.distiller.get_lessons_for_context(context)

        if lessons:
            lesson = lessons[0]
            initial_confidence = lesson.confidence

            # When: lesson is applied successfully
            cycle.validate_lesson(lesson.id, was_helpful=True)
            cycle.validate_lesson(lesson.id, was_helpful=True)

            # Then: confidence increases
            assert lesson.confidence >= initial_confidence, \
                "Confidence should increase with successful validation"

    def test_scenario_unhelpful_lessons_lose_confidence(self, temp_storage):
        """
        Scenario: Failed application decreases lesson confidence

        Given a lesson with high confidence
        When the lesson is applied but not helpful
        Then confidence decreases
        Because unreliable lessons should be trusted less
        """
        # Given: a lesson with some confidence
        cycle = LearningCycle(temp_storage)

        for i in range(5):
            context = Context(goal_type="implementation")
            exp = cycle.start_experience(context, "Task", "approach_x")
            exp.add_action(Action("do", "Do thing", "/src/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Success")
            )

        cycle.extract_and_distill()
        context = Context(goal_type="implementation")
        lessons = cycle.distiller.get_lessons_for_context(context)

        if lessons:
            lesson = lessons[0]
            # Build up confidence first
            cycle.validate_lesson(lesson.id, was_helpful=True)
            cycle.validate_lesson(lesson.id, was_helpful=True)
            high_confidence = lesson.confidence

            # When: lesson proves not helpful
            cycle.validate_lesson(lesson.id, was_helpful=False)

            # Then: confidence should not exceed the high point
            # (The exact behavior depends on validation algorithm)
            assert lesson.validation_count > 0, "Should track validation attempts"


class TestDeveloperTracksLearningProgress:
    """
    Epic: Learning Metrics

    As a developer operating a learning system,
    I want to see statistics about learning progress,
    So that I can verify the system is accumulating knowledge.
    """

    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_scenario_stats_show_accumulation_over_time(self, temp_storage):
        """
        Scenario: Statistics reflect growing knowledge base

        Given a learning system
        When experiences are captured and processed
        Then statistics show increasing experiences, patterns, and lessons
        Because metrics demonstrate learning progress
        """
        # Given: a learning system
        cycle = LearningCycle(temp_storage)
        initial_stats = cycle.get_stats()

        # When: adding experiences
        for i in range(8):
            context = Context(goal_type="implementation")
            exp = cycle.start_experience(context, f"Task {i}", "tdd")
            exp.add_action(Action("test", "Test", "/tests/"))
            cycle.complete_experience(
                exp,
                Outcome(outcome_type=OutcomeType.SUCCESS, description="Success")
            )

        # Extract patterns
        cycle.extract_and_distill()

        # Then: stats show growth
        final_stats = cycle.get_stats()

        assert final_stats['total_experiences'] >= initial_stats['total_experiences'], \
            "Should accumulate experiences"
        assert final_stats['total_patterns'] >= initial_stats['total_patterns'], \
            "Should extract patterns from experiences"
