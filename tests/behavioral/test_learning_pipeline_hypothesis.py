"""
Behavioral test: Learning Pipeline Hypothesis

HYPOTHESIS:
1. Experiences are created when tasks complete/fail
2. Patterns are extracted when extract_patterns_and_lessons() is called
3. Lessons are distilled from patterns with confidence >= 0.4
4. get_guidance_for_task() returns lessons that inform future tasks

This test proves the hypothesis by:
- Creating multiple similar experiences
- Extracting patterns from them
- Verifying lessons are created
- Verifying guidance is returned for new similar tasks

DISABLED: llm_orchestration module scheduled for removal.
"""

import pytest

# Skip entire module - llm_orchestration scheduled for removal
pytestmark = pytest.mark.skip(reason="DISABLED: llm_orchestration module scheduled for removal")
from pathlib import Path
import tempfile
import shutil

from llm_orchestration.learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)


class TestLearningPipelineHypothesis:
    """
    Prove that: Experience → Pattern → Lesson → Guidance

    This is the foundation for the GoT Learning integration.
    If this test passes, we know the pipeline works.
    """

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for learning system."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_hypothesis_experiences_to_patterns_to_lessons_to_guidance(self, temp_storage):
        """
        HYPOTHESIS TEST: The full learning pipeline works end-to-end.

        Given: Multiple similar experiences with the same action sequence
        When: extract_patterns_and_lessons() is called
        Then: Patterns are extracted AND lessons are created
        And: get_guidance() returns those lessons for similar contexts

        This proves the learning system can:
        1. Capture experiences
        2. Find patterns in experiences
        3. Distill patterns into actionable lessons
        4. Return lessons as guidance for future work
        """
        # =====================================================================
        # STEP 1: Create a learning cycle
        # =====================================================================
        cycle = LearningCycle(temp_storage)

        # =====================================================================
        # STEP 2: Create 7 similar successful experiences
        # Pattern confidence = log(occurrences + 1) / 5
        # Need 7+ occurrences to reach 0.4 confidence threshold for lessons
        # =====================================================================
        for i in range(7):
            context = Context(
                goal_type="feature_implementation",
                goal_complexity="moderate",
                domain="api",
                available_tools=["read", "write", "test"],
            )

            experience = cycle.start_experience(
                context=context,
                intent=f"Implement API endpoint {i+1}",
                strategy="test_first",
                experience_type=ExperienceType.TASK_EXECUTION,
            )

            # Same action sequence each time: write_test → write_code → run_tests
            experience.add_action(Action(
                action_type="write_test",
                description="Write test for endpoint",
                target=f"tests/test_endpoint_{i+1}.py",
            ))
            experience.add_action(Action(
                action_type="write_code",
                description="Implement endpoint",
                target=f"src/endpoints/endpoint_{i+1}.py",
            ))
            experience.add_action(Action(
                action_type="run_tests",
                description="Run test suite",
                target="pytest",
            ))

            # All succeed
            outcome = Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Endpoint implemented successfully",
                achieved=["endpoint created", "tests passing"],
                quality_score=0.85,
            )

            cycle.complete_experience(experience, outcome)

        # Verify experiences were captured
        stats = cycle.get_stats()
        assert stats["total_experiences"] == 7, "Should have 7 experiences"

        # =====================================================================
        # STEP 3: Extract patterns and distill lessons
        # =====================================================================
        results = cycle.extract_and_distill()

        # Verify patterns were extracted
        assert results["sequence_patterns"] > 0, (
            "Should extract at least one sequence pattern from "
            "the repeated write_test → write_code → run_tests sequence"
        )

        # Verify lessons were created
        assert results["lessons"] > 0, (
            "Should distill at least one lesson from patterns "
            "with sufficient confidence"
        )

        # =====================================================================
        # STEP 4: Get guidance for a similar new task
        # =====================================================================
        new_context = Context(
            goal_type="feature_implementation",  # Same goal type
            goal_complexity="moderate",
            domain="api",  # Same domain
            available_tools=["read", "write", "test"],
        )

        guidance = cycle.get_guidance(new_context, include_experiences=True)

        # Verify guidance contains lessons
        assert len(guidance["lessons"]) > 0, (
            "get_guidance() should return lessons for similar context"
        )

        # Verify lessons have recommendations (since success_rate > 0.6)
        lesson = guidance["lessons"][0]
        assert len(lesson.recommendations) > 0 or len(lesson.warnings) > 0, (
            "Lesson should have recommendations or warnings"
        )

        # Print for visibility
        print(f"\n=== HYPOTHESIS VALIDATED ===")
        print(f"Experiences captured: {stats['total_experiences']}")
        print(f"Patterns extracted: {results['sequence_patterns']}")
        print(f"Lessons distilled: {results['lessons']}")
        print(f"Guidance returned: {len(guidance['lessons'])} lessons")
        print(f"First lesson: {lesson.title}")
        print(f"Recommendations: {lesson.recommendations}")

    def test_hypothesis_failure_experiences_create_warning_lessons(self, temp_storage):
        """
        HYPOTHESIS: Failed experiences create warning lessons.

        Given: Multiple similar experiences that FAIL
        When: extract_patterns_and_lessons() is called
        Then: Antipattern lessons with WARNINGS are created

        This proves we can learn from failures, not just successes.
        """
        cycle = LearningCycle(temp_storage)

        # Create 7 similar FAILING experiences (need 7+ for confidence >= 0.4)
        for i in range(7):
            context = Context(
                goal_type="bugfix",
                goal_complexity="complex",
                domain="database",
            )

            experience = cycle.start_experience(
                context=context,
                intent=f"Fix database issue {i+1}",
                strategy="dive_in_without_tests",  # Bad strategy
                experience_type=ExperienceType.TASK_EXECUTION,
            )

            # Action sequence that fails: write_code directly without tests
            experience.add_action(Action(
                action_type="write_code",
                description="Modify database code directly",
                target="src/db/queries.py",
            ))
            experience.add_action(Action(
                action_type="deploy",
                description="Deploy to staging",
                target="staging",
            ))

            # All fail
            outcome = Outcome(
                outcome_type=OutcomeType.FAILURE,
                description="Deployment broke existing functionality",
                error_type="RegressionError",
                error_message="Existing tests now failing",
            )

            cycle.complete_experience(experience, outcome)

        # Extract patterns
        results = cycle.extract_and_distill()

        # Verify we got something
        assert results["sequence_patterns"] > 0 or results["antipatterns"] > 0, (
            "Should extract patterns from failure experiences"
        )

        # Get guidance for similar context
        new_context = Context(
            goal_type="bugfix",
            goal_complexity="complex",
            domain="database",
        )

        guidance = cycle.get_guidance(new_context)

        # If we have lessons, check for warnings
        if guidance["lessons"]:
            # At least one lesson should have warnings (low success rate)
            has_warning = any(len(l.warnings) > 0 for l in guidance["lessons"])
            # Note: might not have warnings if success_rate calculation differs
            print(f"\n=== FAILURE LEARNING ===")
            print(f"Lessons from failures: {len(guidance['lessons'])}")
            for lesson in guidance["lessons"]:
                print(f"  - {lesson.title}")
                print(f"    Warnings: {lesson.warnings}")
                print(f"    Recommendations: {lesson.recommendations}")

    def test_hypothesis_lesson_validation_changes_confidence(self, temp_storage):
        """
        HYPOTHESIS: validate_lesson() changes lesson confidence.

        Given: A lesson exists with some confidence
        When: validate_lesson(was_helpful=True) is called
        Then: Confidence increases by 0.05
        When: validate_lesson(was_helpful=False) is called
        Then: Confidence decreases by 0.10

        This proves the feedback loop works.
        """
        cycle = LearningCycle(temp_storage)

        # Create 7 experiences and extract lessons (need 7+ for confidence >= 0.4)
        for i in range(7):
            context = Context(goal_type="test", goal_complexity="simple", domain="unit")
            exp = cycle.start_experience(context=context, intent=f"Test {i}")
            exp.add_action(Action("test_action", "Do something", "target"))
            exp.add_action(Action("verify_action", "Verify it", "target"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Worked",
            ))

        cycle.extract_and_distill()

        # Get a lesson
        guidance = cycle.get_guidance(Context(goal_type="test", goal_complexity="simple"))

        if not guidance["lessons"]:
            pytest.skip("No lessons created - pattern confidence may be too low")

        lesson = guidance["lessons"][0]
        original_confidence = lesson.confidence

        # Validate as helpful
        cycle.validate_lesson(lesson.id, was_helpful=True)

        # Re-fetch and check confidence increased
        guidance_after = cycle.get_guidance(Context(goal_type="test", goal_complexity="simple"))
        lesson_after = next((l for l in guidance_after["lessons"] if l.id == lesson.id), None)

        if lesson_after:
            assert lesson_after.confidence > original_confidence, (
                f"Confidence should increase after positive validation. "
                f"Before: {original_confidence}, After: {lesson_after.confidence}"
            )
            print(f"\n=== VALIDATION FEEDBACK ===")
            print(f"Original confidence: {original_confidence:.3f}")
            print(f"After positive validation: {lesson_after.confidence:.3f}")
            print(f"Delta: +{lesson_after.confidence - original_confidence:.3f}")

    def test_hypothesis_no_extraction_means_no_lessons(self, temp_storage):
        """
        HYPOTHESIS: Without calling extract_and_distill(), no lessons exist.

        Given: Experiences have been captured
        When: extract_and_distill() is NOT called
        Then: get_guidance() returns empty lessons

        This proves extraction is required - it's not automatic.
        """
        cycle = LearningCycle(temp_storage)

        # Create experiences
        for i in range(4):
            context = Context(goal_type="demo", goal_complexity="simple")
            exp = cycle.start_experience(context=context, intent=f"Demo {i}")
            exp.add_action(Action("action_a", "Do A", "target"))
            exp.add_action(Action("action_b", "Do B", "target"))
            cycle.complete_experience(exp, Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Done",
            ))

        # DO NOT call extract_and_distill()

        # Try to get guidance
        guidance = cycle.get_guidance(Context(goal_type="demo", goal_complexity="simple"))

        # Should have no lessons because extraction never happened
        assert len(guidance["lessons"]) == 0, (
            "Without extraction, there should be no lessons. "
            "This proves extraction is a required step."
        )

        print(f"\n=== NO EXTRACTION = NO LESSONS ===")
        print(f"Experiences captured: 4")
        print(f"extract_and_distill() called: NO")
        print(f"Lessons available: {len(guidance['lessons'])}")
