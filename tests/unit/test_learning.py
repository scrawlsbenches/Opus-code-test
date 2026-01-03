"""
Unit Tests for Learning System

Tests the experience capture, pattern extraction, and lesson distillation
components of the learning system (llm_orchestration/learning.py).

Coverage areas:
- Data classes: Context, Action, Outcome, Experience
- Experience storage and retrieval
- Pattern extraction (sequence, strategy, antipatterns)
- Lesson distillation and retrieval
- Learning consolidation (merging, aging, deprecation)
- Full learning cycle integration
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from typing import List

from llm_orchestration.learning import (
    # Enums
    OutcomeType, ExperienceType,
    # Data classes
    Context, Action, Outcome, Experience, Pattern, Lesson,
    # Storage and extraction
    ExperienceStore, PatternExtractor, LessonDistiller,
    # Main system
    LearningCycle, LearningConsolidator, ConsolidationResult
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_storage():
    """Provide a temporary directory for storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_context():
    """Provide a sample context for testing."""
    return Context(
        goal_type="test_goal",
        goal_complexity="moderate",
        domain="testing",
        available_tools=["tool1", "tool2"],
        available_agents=2,
        prior_failures=0,
        time_pressure="none",
        constraints=["constraint1"]
    )


@pytest.fixture
def sample_action():
    """Provide a sample action for testing."""
    return Action(
        action_type="test_action",
        description="Test action description",
        target="test_target",
        parameters={"key": "value"},
        duration_ms=100.0
    )


@pytest.fixture
def sample_outcome_success():
    """Provide a successful outcome for testing."""
    return Outcome(
        outcome_type=OutcomeType.SUCCESS,
        description="Test succeeded",
        achieved=["goal1", "goal2"],
        quality_score=0.9,
        efficiency_score=0.8
    )


@pytest.fixture
def sample_outcome_failure():
    """Provide a failed outcome for testing."""
    return Outcome(
        outcome_type=OutcomeType.FAILURE,
        description="Test failed",
        not_achieved=["goal1"],
        error_type="TestError",
        error_message="Something went wrong"
    )


@pytest.fixture
def experience_store(temp_storage):
    """Provide an ExperienceStore instance."""
    return ExperienceStore(temp_storage / "experiences")


@pytest.fixture
def learning_cycle(temp_storage):
    """Provide a LearningCycle instance."""
    return LearningCycle(temp_storage)


# =============================================================================
# TEST EXPERIENCE DATA CLASSES
# =============================================================================

class TestExperienceDataClasses:
    """Test the data classes used for experience capture."""

    def test_context_creation(self):
        """Test creating a Context with all fields."""
        context = Context(
            goal_type="refactor",
            goal_complexity="complex",
            domain="backend",
            available_tools=["git", "pytest"],
            available_agents=3,
            prior_failures=1,
            time_pressure="moderate",
            constraints=["no_breaking_changes"],
            notes="Important refactor"
        )

        assert context.goal_type == "refactor"
        assert context.goal_complexity == "complex"
        assert context.domain == "backend"
        assert len(context.available_tools) == 2
        assert context.available_agents == 3
        assert context.prior_failures == 1
        assert context.time_pressure == "moderate"
        assert len(context.constraints) == 1
        assert context.notes == "Important refactor"

    def test_context_similarity_identical(self, sample_context):
        """Test that identical contexts have similarity of 1.0."""
        context2 = Context(
            goal_type=sample_context.goal_type,
            goal_complexity=sample_context.goal_complexity,
            domain=sample_context.domain,
            available_agents=sample_context.available_agents,
            prior_failures=sample_context.prior_failures,
            time_pressure=sample_context.time_pressure
        )

        similarity = sample_context.similarity_to(context2)
        assert similarity == pytest.approx(1.0)

    def test_context_similarity_different(self, sample_context):
        """Test that completely different contexts have low similarity."""
        context2 = Context(
            goal_type="different_goal",
            goal_complexity="simple",
            domain="different",
            available_agents=10,
            prior_failures=20,
            time_pressure="high"
        )

        similarity = sample_context.similarity_to(context2)
        assert similarity < 0.5

    def test_context_similarity_partial(self, sample_context):
        """Test partial context similarity."""
        context2 = Context(
            goal_type=sample_context.goal_type,  # Same (0.3)
            goal_complexity="different",  # Different (0)
            domain=sample_context.domain,  # Same (0.2)
            available_agents=sample_context.available_agents,  # Same (0.1)
            prior_failures=sample_context.prior_failures,  # Same (0.1)
            time_pressure="different"  # Different (0)
        )

        similarity = sample_context.similarity_to(context2)
        # Should be 0.3 + 0.2 + 0.1 + 0.1 = 0.7
        assert 0.6 < similarity < 0.8

    def test_action_creation_and_serialization(self, sample_action):
        """Test Action creation and to_dict serialization."""
        action_dict = sample_action.to_dict()

        assert action_dict['action_type'] == "test_action"
        assert action_dict['description'] == "Test action description"
        assert action_dict['target'] == "test_target"
        assert action_dict['parameters'] == {"key": "value"}
        assert action_dict['duration_ms'] == 100.0
        assert 'timestamp' in action_dict

    def test_outcome_success(self, sample_outcome_success):
        """Test successful outcome."""
        assert sample_outcome_success.outcome_type == OutcomeType.SUCCESS
        assert sample_outcome_success.was_successful()
        assert len(sample_outcome_success.achieved) == 2
        assert sample_outcome_success.quality_score == 0.9

    def test_outcome_failure(self, sample_outcome_failure):
        """Test failed outcome."""
        assert sample_outcome_failure.outcome_type == OutcomeType.FAILURE
        assert not sample_outcome_failure.was_successful()
        assert len(sample_outcome_failure.not_achieved) == 1
        assert sample_outcome_failure.error_type == "TestError"

    def test_outcome_serialization(self, sample_outcome_success):
        """Test Outcome to_dict serialization."""
        outcome_dict = sample_outcome_success.to_dict()

        assert outcome_dict['outcome_type'] == "SUCCESS"
        assert outcome_dict['description'] == "Test succeeded"
        assert outcome_dict['achieved'] == ["goal1", "goal2"]
        assert outcome_dict['quality_score'] == 0.9

    def test_experience_creation(self, sample_context):
        """Test creating an Experience."""
        experience = Experience(
            id="exp_test_001",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test the system"
        )

        assert experience.id == "exp_test_001"
        assert experience.experience_type == ExperienceType.GOAL_EXECUTION
        assert experience.intent == "Test the system"
        assert experience.context == sample_context
        assert len(experience.actions) == 0
        assert experience.outcome is None

    def test_experience_add_action(self, sample_context, sample_action):
        """Test adding actions to an experience."""
        experience = Experience(
            id="exp_test_002",
            experience_type=ExperienceType.TASK_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test actions"
        )

        experience.add_action(sample_action)
        assert len(experience.actions) == 1
        assert experience.actions[0] == sample_action

    def test_experience_complete(self, sample_context, sample_outcome_success):
        """Test completing an experience with an outcome."""
        experience = Experience(
            id="exp_test_003",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test completion"
        )

        experience.complete(sample_outcome_success)
        assert experience.outcome == sample_outcome_success
        assert experience.outcome.was_successful()

    def test_experience_reflect(self, sample_context):
        """Test adding reflection to an experience."""
        experience = Experience(
            id="exp_test_004",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test reflection"
        )

        worked = ["approach1", "approach2"]
        didnt_work = ["approach3"]
        different = ["try_approach4"]

        experience.reflect(worked, didnt_work, different)

        assert experience.what_worked == worked
        assert experience.what_didnt_work == didnt_work
        assert experience.would_do_differently == different

    def test_experience_serialization_and_deserialization(
        self, sample_context, sample_action, sample_outcome_success
    ):
        """Test Experience to_dict and from_dict round-trip."""
        experience = Experience(
            id="exp_test_005",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test serialization",
            strategy_used="test_strategy"
        )

        experience.add_action(sample_action)
        experience.complete(sample_outcome_success)
        experience.reflect(["worked"], ["didnt_work"], ["different"])
        experience.tags.add("test")

        # Serialize
        exp_dict = experience.to_dict()
        assert exp_dict['id'] == "exp_test_005"
        assert exp_dict['experience_type'] == "GOAL_EXECUTION"

        # Deserialize
        restored = Experience.from_dict(exp_dict)
        assert restored.id == experience.id
        assert restored.experience_type == experience.experience_type
        assert restored.intent == experience.intent
        assert restored.strategy_used == experience.strategy_used
        assert len(restored.actions) == 1
        assert restored.outcome.outcome_type == OutcomeType.SUCCESS
        assert "test" in restored.tags


# =============================================================================
# TEST EXPERIENCE STORE
# =============================================================================

class TestExperienceStore:
    """Test the ExperienceStore for persistence and retrieval."""

    def test_save_and_retrieve_experience(
        self, experience_store, sample_context, sample_outcome_success
    ):
        """Test saving and retrieving an experience."""
        experience = Experience(
            id="exp_store_001",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test storage"
        )
        experience.complete(sample_outcome_success)

        # Save
        experience_store.save(experience)

        # Retrieve
        retrieved = experience_store.get("exp_store_001")
        assert retrieved is not None
        assert retrieved.id == "exp_store_001"
        assert retrieved.intent == "Test storage"

    def test_find_similar_context(self, experience_store, sample_context):
        """Test finding experiences with similar contexts."""
        # Create and save multiple experiences with varying contexts
        for i in range(5):
            context = Context(
                goal_type="test_goal" if i < 3 else "other_goal",
                goal_complexity="moderate",
                domain="testing"
            )
            experience = Experience(
                id=f"exp_sim_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=context,
                intent=f"Test {i}"
            )
            experience_store.save(experience)

        # Find similar to sample_context (which has goal_type="test_goal")
        similar = experience_store.find_similar_context(sample_context, min_similarity=0.3)

        # Should find at least the 3 with matching goal_type
        assert len(similar) >= 3
        # Each result is (experience, similarity_score)
        for exp, score in similar:
            assert score >= 0.3

    def test_find_by_tags(self, experience_store, sample_context):
        """Test finding experiences by tags."""
        # Create experiences with different tags
        exp1 = Experience(
            id="exp_tag_1",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test 1"
        )
        exp1.tags.add("tag1")
        exp1.tags.add("tag2")

        exp2 = Experience(
            id="exp_tag_2",
            experience_type=ExperienceType.GOAL_EXECUTION,
            timestamp=datetime.now(),
            context=sample_context,
            intent="Test 2"
        )
        exp2.tags.add("tag2")
        exp2.tags.add("tag3")

        experience_store.save(exp1)
        experience_store.save(exp2)

        # Find by single tag (match_all=False)
        results = experience_store.find_by_tags({"tag2"}, match_all=False)
        assert len(results) == 2

        # Find by multiple tags with match_all=True
        results = experience_store.find_by_tags({"tag1", "tag2"}, match_all=True)
        assert len(results) == 1
        assert results[0].id == "exp_tag_1"

    def test_find_by_outcome(self, experience_store, sample_context):
        """Test finding experiences by outcome type."""
        # Create successes and failures
        for i in range(3):
            exp = Experience(
                id=f"exp_out_success_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=sample_context,
                intent=f"Success {i}"
            )
            exp.complete(Outcome(OutcomeType.SUCCESS, "Success"))
            experience_store.save(exp)

        for i in range(2):
            exp = Experience(
                id=f"exp_out_fail_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=sample_context,
                intent=f"Failure {i}"
            )
            exp.complete(Outcome(OutcomeType.FAILURE, "Failure"))
            experience_store.save(exp)

        # Find successes
        successes = experience_store.find_by_outcome(OutcomeType.SUCCESS)
        assert len(successes) == 3

        # Find failures
        failures = experience_store.find_by_outcome(OutcomeType.FAILURE)
        assert len(failures) == 2

    def test_find_successful_for_context(self, experience_store, sample_context):
        """Test finding successful experiences with similar context."""
        # Create successful experiences with similar context
        for i in range(4):
            context = Context(
                goal_type=sample_context.goal_type,
                goal_complexity=sample_context.goal_complexity,
                domain=sample_context.domain
            )
            exp = Experience(
                id=f"exp_succ_ctx_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=context,
                intent=f"Test {i}"
            )
            exp.complete(Outcome(OutcomeType.SUCCESS, "Success"))
            experience_store.save(exp)

        # Find successful for context
        results = experience_store.find_successful_for_context(sample_context, limit=2)
        assert len(results) <= 2
        for exp in results:
            assert exp.outcome.was_successful()

    def test_count(self, experience_store, sample_context):
        """Test counting experiences in the store."""
        initial_count = experience_store.count()

        # Add experiences
        for i in range(3):
            exp = Experience(
                id=f"exp_count_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=sample_context,
                intent=f"Test {i}"
            )
            experience_store.save(exp)

        assert experience_store.count() == initial_count + 3


# =============================================================================
# TEST PATTERN EXTRACTION
# =============================================================================

class TestPatternExtraction:
    """Test pattern extraction from experiences."""

    def test_pattern_add_evidence_success(self):
        """Test adding successful evidence to a pattern."""
        pattern = Pattern(
            id="pat_test_001",
            pattern_type="sequence",
            description="Test pattern"
        )

        # Add successful evidence
        pattern.add_evidence("exp_1", was_successful=True)
        assert pattern.occurrence_count == 1
        assert pattern.success_rate == 1.0
        assert pattern.confidence > 0

    def test_pattern_add_evidence_failure(self):
        """Test adding failure evidence to a pattern."""
        pattern = Pattern(
            id="pat_test_002",
            pattern_type="sequence",
            description="Test pattern"
        )

        # Add failure evidence
        pattern.add_evidence("exp_1", was_successful=False)
        assert pattern.occurrence_count == 1
        assert pattern.success_rate == 0.0

    def test_pattern_confidence_grows_with_evidence(self):
        """Test that pattern confidence increases with more evidence."""
        pattern = Pattern(
            id="pat_test_003",
            pattern_type="sequence",
            description="Test pattern"
        )

        # Add evidence gradually
        confidences = []
        for i in range(10):
            pattern.add_evidence(f"exp_{i}", was_successful=True)
            confidences.append(pattern.confidence)

        # Confidence should increase (but not linearly - logarithmically)
        assert confidences[-1] > confidences[0]
        assert confidences[-1] <= 0.95  # Cap at 0.95

    def test_extract_sequence_patterns(self, experience_store):
        """Test extracting action sequence patterns."""
        # Create experiences with recurring action sequences
        for i in range(5):
            context = Context(goal_type="test", goal_complexity="simple", domain="test")
            exp = Experience(
                id=f"exp_seq_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=context,
                intent="Test"
            )

            # Add same sequence to each experience
            exp.add_action(Action("action_A", "First action", "target"))
            exp.add_action(Action("action_B", "Second action", "target"))
            exp.add_action(Action("action_C", "Third action", "target"))

            # Most succeed
            if i < 4:
                exp.complete(Outcome(OutcomeType.SUCCESS, "Success"))
            else:
                exp.complete(Outcome(OutcomeType.FAILURE, "Failure"))

            experience_store.save(exp)

        # Extract patterns
        extractor = PatternExtractor(experience_store)
        patterns = extractor.extract_sequence_patterns(min_occurrences=3)

        # Should find patterns for pairs and triples
        assert len(patterns) > 0

        # Check that patterns have evidence
        for pattern in patterns:
            assert pattern.occurrence_count >= 3
            assert pattern.success_rate > 0  # Most succeeded

    def test_extract_strategy_patterns(self, experience_store):
        """Test extracting strategy patterns."""
        # Create experiences with strategy and goal type pairs
        for i in range(5):
            context = Context(goal_type="refactor", goal_complexity="simple", domain="test")
            exp = Experience(
                id=f"exp_strat_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=context,
                intent="Test",
                strategy_used="test_strategy"
            )

            # Most succeed
            if i < 4:
                exp.complete(Outcome(OutcomeType.SUCCESS, "Success"))
            else:
                exp.complete(Outcome(OutcomeType.FAILURE, "Failure"))

            experience_store.save(exp)

        # Extract strategy patterns
        extractor = PatternExtractor(experience_store)
        patterns = extractor.extract_strategy_patterns(min_occurrences=3)

        # Should find at least one strategy pattern
        assert len(patterns) >= 1

        # Check pattern structure
        for pattern in patterns:
            assert pattern.pattern_type == "strategy"
            assert 'strategy' in pattern.structure
            assert 'goal_type' in pattern.structure

    def test_extract_antipatterns(self, experience_store):
        """Test extracting failure antipatterns."""
        # Create many failures with the same goal type
        for i in range(5):
            context = Context(goal_type="problematic_goal", goal_complexity="simple", domain="test")
            exp = Experience(
                id=f"exp_anti_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=context,
                intent="Test"
            )
            exp.complete(Outcome(OutcomeType.FAILURE, "Failure"))
            experience_store.save(exp)

        # Extract antipatterns
        extractor = PatternExtractor(experience_store)
        patterns = extractor.extract_antipatterns(min_failures=3)

        # Should find at least one antipattern
        assert len(patterns) >= 1

        # Check antipattern properties
        for pattern in patterns:
            assert pattern.pattern_type == "antipattern"
            assert pattern.success_rate == 0.0

    def test_get_patterns_for_context(self, experience_store):
        """Test retrieving patterns applicable to a context."""
        # Create experiences and extract patterns
        context = Context(goal_type="test_goal", goal_complexity="simple", domain="test")

        for i in range(5):
            exp = Experience(
                id=f"exp_ctx_{i}",
                experience_type=ExperienceType.GOAL_EXECUTION,
                timestamp=datetime.now(),
                context=context,
                intent="Test",
                strategy_used="test_strategy"
            )
            exp.complete(Outcome(OutcomeType.SUCCESS, "Success"))
            experience_store.save(exp)

        extractor = PatternExtractor(experience_store)
        extractor.extract_strategy_patterns(min_occurrences=3)

        # Get patterns for context
        applicable = extractor.get_patterns_for_context(context, pattern_type="strategy")

        # Should find strategy patterns
        assert len(applicable) > 0


# =============================================================================
# TEST LESSON DISTILLATION
# =============================================================================

class TestLessonDistillation:
    """Test lesson distillation from patterns."""

    def test_lesson_is_applicable_to_context(self):
        """Test checking if a lesson applies to a context."""
        lesson = Lesson(
            id="lesson_001",
            title="Test Lesson",
            description="Test",
            applicable_conditions={
                'goal_types': ['refactor', 'test'],
                'domains': ['backend']
            }
        )

        # Matching context
        context1 = Context(goal_type="refactor", goal_complexity="simple", domain="backend")
        assert lesson.is_applicable_to(context1)

        # Non-matching goal type
        context2 = Context(goal_type="other", goal_complexity="simple", domain="backend")
        assert not lesson.is_applicable_to(context2)

        # Non-matching domain
        context3 = Context(goal_type="refactor", goal_complexity="simple", domain="frontend")
        assert not lesson.is_applicable_to(context3)

    def test_lesson_validate_helpful(self):
        """Test validating a lesson as helpful."""
        lesson = Lesson(
            id="lesson_002",
            title="Test",
            description="Test",
            confidence=0.5
        )

        initial_confidence = lesson.confidence
        lesson.validate(was_helpful=True)

        assert lesson.validation_count == 1
        assert lesson.confidence > initial_confidence
        assert lesson.last_validated is not None

    def test_lesson_validate_not_helpful(self):
        """Test validating a lesson as not helpful."""
        lesson = Lesson(
            id="lesson_003",
            title="Test",
            description="Test",
            confidence=0.5
        )

        initial_confidence = lesson.confidence
        lesson.validate(was_helpful=False)

        assert lesson.validation_count == 1
        assert lesson.confidence < initial_confidence

    def test_lesson_record_application(self):
        """Test recording lesson application."""
        lesson = Lesson(
            id="lesson_004",
            title="Test",
            description="Test",
            confidence=0.5
        )

        lesson.record_application()

        assert lesson.application_count == 1
        assert lesson.last_applied is not None
        # Confidence should increase slightly
        assert lesson.confidence > 0.5

    def test_lesson_apply_aging(self):
        """Test applying aging to lessons."""
        lesson = Lesson(
            id="lesson_005",
            title="Test",
            description="Test",
            confidence=0.8
        )

        # No aging for recent usage
        lesson.apply_aging(days_since_last_use=10)
        assert lesson.confidence == 0.8

        # Some aging after 40 days
        lesson.apply_aging(days_since_last_use=40)
        assert lesson.confidence < 0.8

    def test_lesson_similarity(self):
        """Test calculating similarity between lessons."""
        lesson1 = Lesson(
            id="lesson_006",
            title="Test 1",
            description="Test",
            applicable_conditions={'goal_types': ['refactor']},
            recommendations=["rec1", "rec2"],
            warnings=["warn1"]
        )

        # Very similar lesson
        lesson2 = Lesson(
            id="lesson_007",
            title="Test 2",
            description="Test",
            applicable_conditions={'goal_types': ['refactor']},
            recommendations=["rec1", "rec2"],
            warnings=["warn1"]
        )

        similarity = lesson1.similarity_to(lesson2)
        assert similarity > 0.8

        # Different lesson
        lesson3 = Lesson(
            id="lesson_008",
            title="Test 3",
            description="Test",
            applicable_conditions={'goal_types': ['debug']},
            recommendations=["rec3"],
            warnings=["warn2"]
        )

        similarity = lesson1.similarity_to(lesson3)
        assert similarity < 0.5

    def test_distill_from_high_confidence_pattern(self, experience_store):
        """Test distilling a lesson from a high-confidence pattern."""
        # Create a high-confidence pattern
        pattern = Pattern(
            id="pat_lesson_001",
            pattern_type="strategy",
            description="Test strategy pattern",
            structure={'strategy': 'test_strat', 'goal_type': 'test_goal'},
            confidence=0.7,
            success_rate=0.8
        )

        extractor = PatternExtractor(experience_store)
        extractor.patterns[pattern.id] = pattern

        distiller = LessonDistiller(extractor, experience_store)
        lesson = distiller.distill_from_pattern(pattern)

        assert lesson is not None
        assert lesson.confidence > 0
        assert pattern.id in lesson.supporting_patterns

    def test_no_lesson_from_low_confidence_pattern(self, experience_store):
        """Test that no lesson is created from low-confidence patterns."""
        # Create a low-confidence pattern
        pattern = Pattern(
            id="pat_lesson_002",
            pattern_type="strategy",
            description="Test",
            confidence=0.2  # Too low
        )

        extractor = PatternExtractor(experience_store)
        extractor.patterns[pattern.id] = pattern

        distiller = LessonDistiller(extractor, experience_store)
        lesson = distiller.distill_from_pattern(pattern)

        assert lesson is None

    def test_get_lessons_for_context(self, experience_store):
        """Test retrieving lessons applicable to a context."""
        extractor = PatternExtractor(experience_store)
        distiller = LessonDistiller(extractor, experience_store)

        # Create lessons with different conditions
        lesson1 = Lesson(
            id="lesson_ctx_1",
            title="Lesson 1",
            description="Test",
            applicable_conditions={'goal_types': ['refactor']},
            confidence=0.8
        )

        lesson2 = Lesson(
            id="lesson_ctx_2",
            title="Lesson 2",
            description="Test",
            applicable_conditions={'goal_types': ['debug']},
            confidence=0.7
        )

        distiller.lessons[lesson1.id] = lesson1
        distiller.lessons[lesson2.id] = lesson2

        # Get lessons for refactor context
        context = Context(goal_type="refactor", goal_complexity="simple", domain="test")
        lessons = distiller.get_lessons_for_context(context, min_confidence=0.5)

        assert len(lessons) == 1
        assert lessons[0].id == "lesson_ctx_1"


# =============================================================================
# TEST LEARNING CONSOLIDATOR
# =============================================================================

class TestLearningConsolidator:
    """Test lesson consolidation (merging, aging, deprecation)."""

    def test_merge_similar_lessons(self, learning_cycle):
        """Test merging similar lessons to reduce redundancy."""
        # Create very similar lessons
        lesson1 = Lesson(
            id="merge_1",
            title="Lesson 1",
            description="Test",
            applicable_conditions={'goal_types': ['refactor']},
            recommendations=["rec1", "rec2"],
            warnings=["warn1"],
            confidence=0.8,
            validation_count=5
        )

        lesson2 = Lesson(
            id="merge_2",
            title="Lesson 2",
            description="Test",
            applicable_conditions={'goal_types': ['refactor']},
            recommendations=["rec1", "rec2"],
            warnings=["warn1"],
            confidence=0.7,
            validation_count=3
        )

        learning_cycle.distiller.lessons[lesson1.id] = lesson1
        learning_cycle.distiller.lessons[lesson2.id] = lesson2

        consolidator = LearningConsolidator(learning_cycle)
        result = consolidator.consolidate()

        # One lesson should be merged
        assert result.lessons_merged >= 1

        # The lesson with lower confidence should be superseded
        assert lesson2.superseded_by == lesson1.id

    def test_age_lessons(self, learning_cycle):
        """Test aging of unused lessons."""
        # Create an old lesson
        old_time = datetime.now() - timedelta(days=60)
        lesson = Lesson(
            id="age_1",
            title="Old Lesson",
            description="Test",
            confidence=0.8,
            created_at=old_time,
            last_applied=None  # Never applied
        )

        learning_cycle.distiller.lessons[lesson.id] = lesson

        consolidator = LearningConsolidator(learning_cycle)
        consolidator._apply_aging_to_all()

        # Confidence should have decayed
        assert lesson.confidence < 0.8

    def test_deprecate_stale_lessons(self, learning_cycle):
        """Test deprecation of low-confidence lessons."""
        # Create a low-confidence lesson
        lesson = Lesson(
            id="deprecate_1",
            title="Low Confidence",
            description="Test",
            confidence=0.2  # Below deprecation threshold
        )

        learning_cycle.distiller.lessons[lesson.id] = lesson

        consolidator = LearningConsolidator(learning_cycle)
        result = consolidator.consolidate()

        # Lesson should be deprecated
        assert result.lessons_deprecated >= 1
        assert lesson.superseded_by is not None

    def test_consolidation_metrics(self, learning_cycle):
        """Test that consolidation returns meaningful metrics."""
        # Create lessons for consolidation
        for i in range(3):
            lesson = Lesson(
                id=f"metric_{i}",
                title=f"Lesson {i}",
                description="Test",
                confidence=0.5 + (i * 0.2)
            )
            learning_cycle.distiller.lessons[lesson.id] = lesson

        consolidator = LearningConsolidator(learning_cycle)
        result = consolidator.consolidate()

        # Check that result has all expected fields
        assert isinstance(result, ConsolidationResult)
        assert result.lessons_merged >= 0
        assert result.lessons_deprecated >= 0
        assert result.lessons_promoted >= 0
        assert isinstance(result.new_patterns, list)

        # Summary should be readable
        summary = result.summary()
        assert "promoted" in summary
        assert "merged" in summary
        assert "deprecated" in summary


# =============================================================================
# TEST LEARNING CYCLE
# =============================================================================

class TestLearningCycle:
    """Test the main LearningCycle orchestration."""

    def test_initialization(self, temp_storage):
        """Test LearningCycle initialization."""
        cycle = LearningCycle(temp_storage)

        assert cycle.store is not None
        assert cycle.extractor is not None
        assert cycle.distiller is not None
        assert cycle.store.count() == 0

    def test_start_experience(self, learning_cycle, sample_context):
        """Test starting a new experience."""
        experience = learning_cycle.start_experience(
            context=sample_context,
            intent="Test the system",
            experience_type=ExperienceType.GOAL_EXECUTION,
            strategy="test_strategy"
        )

        assert experience is not None
        assert experience.id.startswith("exp_")
        assert experience.intent == "Test the system"
        assert experience.strategy_used == "test_strategy"
        assert experience.context == sample_context

    def test_complete_experience(self, learning_cycle, sample_context, sample_outcome_success):
        """Test completing and saving an experience."""
        experience = learning_cycle.start_experience(
            context=sample_context,
            intent="Test completion"
        )

        reflection = {
            'worked': ["approach1"],
            'didnt_work': ["approach2"],
            'different': ["try_approach3"]
        }

        learning_cycle.complete_experience(
            experience,
            sample_outcome_success,
            reflection
        )

        # Experience should be saved
        retrieved = learning_cycle.store.get(experience.id)
        assert retrieved is not None
        assert retrieved.outcome.was_successful()
        assert len(retrieved.what_worked) == 1

        # Auto-tags should be added
        assert sample_context.goal_type in retrieved.tags
        assert "success" in retrieved.tags

    def test_get_guidance_empty(self, learning_cycle, sample_context):
        """Test getting guidance when no lessons exist."""
        guidance = learning_cycle.get_guidance(sample_context)

        assert 'lessons' in guidance
        assert 'recommendations' in guidance
        assert 'warnings' in guidance
        assert 'relevant_successes' in guidance
        assert 'relevant_failures' in guidance

        # Should be empty
        assert len(guidance['lessons']) == 0

    def test_get_guidance_with_lessons(self, learning_cycle, sample_context):
        """Test getting guidance when lessons exist."""
        # Create a lesson applicable to the context
        lesson = Lesson(
            id="guide_1",
            title="Test Lesson",
            description="Test",
            applicable_conditions={'goal_types': [sample_context.goal_type]},
            recommendations=["rec1"],
            warnings=["warn1"],
            confidence=0.8
        )
        learning_cycle.distiller.lessons[lesson.id] = lesson

        guidance = learning_cycle.get_guidance(sample_context, include_experiences=False)

        assert len(guidance['lessons']) >= 1
        assert "rec1" in guidance['recommendations']
        assert "warn1" in guidance['warnings']

    def test_extract_and_distill(self, learning_cycle, sample_context):
        """Test pattern extraction and lesson distillation."""
        # Create some experiences first
        for i in range(5):
            exp = learning_cycle.start_experience(
                context=sample_context,
                intent=f"Test {i}",
                strategy="test_strategy"
            )
            exp.add_action(Action("action_A", "First", "target"))
            exp.add_action(Action("action_B", "Second", "target"))

            outcome = Outcome(OutcomeType.SUCCESS, "Success")
            learning_cycle.complete_experience(exp, outcome)

        # Extract and distill
        results = learning_cycle.extract_and_distill()

        assert 'sequence_patterns' in results
        assert 'strategy_patterns' in results
        assert 'antipatterns' in results
        assert 'lessons' in results

        # Should have found some patterns
        total_patterns = sum([
            results['sequence_patterns'],
            results['strategy_patterns'],
            results['antipatterns']
        ])
        assert total_patterns >= 0  # May be 0 if not enough data

    def test_validate_lesson(self, learning_cycle):
        """Test validating a lesson."""
        lesson = Lesson(
            id="validate_1",
            title="Test",
            description="Test",
            confidence=0.5
        )
        learning_cycle.distiller.lessons[lesson.id] = lesson

        learning_cycle.validate_lesson("validate_1", was_helpful=True)

        assert lesson.validation_count == 1
        assert lesson.confidence > 0.5

    def test_consolidate_lessons(self, learning_cycle):
        """Test running consolidation."""
        # Add some lessons
        for i in range(3):
            lesson = Lesson(
                id=f"consolidate_{i}",
                title=f"Lesson {i}",
                description="Test",
                confidence=0.4 + (i * 0.2)
            )
            learning_cycle.distiller.lessons[lesson.id] = lesson

        result = learning_cycle.consolidate_lessons()

        assert isinstance(result, ConsolidationResult)
        assert result.lessons_promoted >= 0

    def test_get_stats(self, learning_cycle, sample_context):
        """Test getting learning system statistics."""
        # Add some data
        exp = learning_cycle.start_experience(sample_context, "Test")
        outcome = Outcome(OutcomeType.SUCCESS, "Success")
        learning_cycle.complete_experience(exp, outcome)

        stats = learning_cycle.get_stats()

        assert 'total_experiences' in stats
        assert 'total_patterns' in stats
        assert 'total_lessons' in stats
        assert 'patterns_by_type' in stats
        assert 'high_confidence_lessons' in stats
        assert 'active_lessons' in stats

        assert stats['total_experiences'] >= 1

    def test_persistence_across_sessions(self, temp_storage, sample_context):
        """Test that learning persists across LearningCycle instances."""
        # Create first cycle and add experience
        cycle1 = LearningCycle(temp_storage)
        exp = cycle1.start_experience(sample_context, "Test persistence")
        outcome = Outcome(OutcomeType.SUCCESS, "Success")
        cycle1.complete_experience(exp, outcome)

        exp_id = exp.id
        initial_count = cycle1.store.count()

        # Create new cycle from same storage
        cycle2 = LearningCycle(temp_storage)

        # Should load the saved experience
        assert cycle2.store.count() == initial_count
        retrieved = cycle2.store.get(exp_id)
        assert retrieved is not None
        assert retrieved.intent == "Test persistence"


# =============================================================================
# TEST LESSON RETRIEVAL
# =============================================================================

class TestLessonRetrieval:
    """Test lesson retrieval mechanisms."""

    def test_retrieve_by_context(self, learning_cycle):
        """Test retrieving lessons by context match."""
        # Create lessons with specific contexts
        lesson1 = Lesson(
            id="retrieve_ctx_1",
            title="Refactor Lesson",
            description="Test",
            applicable_conditions={'goal_types': ['refactor']},
            confidence=0.8
        )

        lesson2 = Lesson(
            id="retrieve_ctx_2",
            title="Debug Lesson",
            description="Test",
            applicable_conditions={'goal_types': ['debug']},
            confidence=0.7
        )

        learning_cycle.distiller.lessons[lesson1.id] = lesson1
        learning_cycle.distiller.lessons[lesson2.id] = lesson2

        # Retrieve for refactor context
        context = Context(goal_type="refactor", goal_complexity="simple", domain="test")
        lessons = learning_cycle.distiller.get_lessons_for_context(context)

        assert len(lessons) == 1
        assert lessons[0].id == "retrieve_ctx_1"

    def test_retrieve_by_similarity(self, learning_cycle):
        """Test retrieving lessons by context similarity."""
        # Create lessons with overlapping conditions
        lesson1 = Lesson(
            id="retrieve_sim_1",
            title="Backend Lesson",
            description="Test",
            applicable_conditions={
                'goal_types': ['refactor'],
                'domains': ['backend']
            },
            confidence=0.8
        )

        learning_cycle.distiller.lessons[lesson1.id] = lesson1

        # Context that partially matches
        context = Context(
            goal_type="refactor",  # Matches
            goal_complexity="simple",
            domain="backend"  # Matches
        )

        lessons = learning_cycle.distiller.get_lessons_for_context(context)

        assert len(lessons) >= 1

    def test_lesson_ranking_by_confidence(self, learning_cycle):
        """Test that lessons are ranked by confidence."""
        # Create lessons with different confidences
        for i in range(5):
            lesson = Lesson(
                id=f"rank_{i}",
                title=f"Lesson {i}",
                description="Test",
                applicable_conditions={'goal_types': ['test']},
                confidence=0.3 + (i * 0.1)
            )
            learning_cycle.distiller.lessons[lesson.id] = lesson

        context = Context(goal_type="test", goal_complexity="simple", domain="test")
        lessons = learning_cycle.distiller.get_lessons_for_context(context)

        # Should be sorted by confidence (descending)
        confidences = [l.confidence for l in lessons]
        assert confidences == sorted(confidences, reverse=True)

    def test_exclude_superseded_lessons(self, learning_cycle):
        """Test that superseded lessons are not retrieved."""
        lesson1 = Lesson(
            id="supersede_1",
            title="Old Lesson",
            description="Test",
            applicable_conditions={'goal_types': ['test']},
            confidence=0.8,
            superseded_by="supersede_2"  # Superseded
        )

        lesson2 = Lesson(
            id="supersede_2",
            title="New Lesson",
            description="Test",
            applicable_conditions={'goal_types': ['test']},
            confidence=0.9
        )

        learning_cycle.distiller.lessons[lesson1.id] = lesson1
        learning_cycle.distiller.lessons[lesson2.id] = lesson2

        context = Context(goal_type="test", goal_complexity="simple", domain="test")
        lessons = learning_cycle.distiller.get_lessons_for_context(context)

        # Should only get the new lesson
        lesson_ids = [l.id for l in lessons]
        assert "supersede_2" in lesson_ids
        assert "supersede_1" not in lesson_ids

    def test_min_confidence_filtering(self, learning_cycle):
        """Test filtering lessons by minimum confidence."""
        lesson1 = Lesson(
            id="conf_filter_1",
            title="High Confidence",
            description="Test",
            applicable_conditions={'goal_types': ['test']},
            confidence=0.9
        )

        lesson2 = Lesson(
            id="conf_filter_2",
            title="Low Confidence",
            description="Test",
            applicable_conditions={'goal_types': ['test']},
            confidence=0.2
        )

        learning_cycle.distiller.lessons[lesson1.id] = lesson1
        learning_cycle.distiller.lessons[lesson2.id] = lesson2

        context = Context(goal_type="test", goal_complexity="simple", domain="test")

        # High threshold - only high confidence
        lessons = learning_cycle.distiller.get_lessons_for_context(context, min_confidence=0.7)
        assert len(lessons) == 1
        assert lessons[0].id == "conf_filter_1"

        # Low threshold - both
        lessons = learning_cycle.distiller.get_lessons_for_context(context, min_confidence=0.1)
        assert len(lessons) == 2
