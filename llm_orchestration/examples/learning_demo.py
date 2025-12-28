#!/usr/bin/env python3
"""
Learning Demo: Experience Capture and Lesson Extraction

This example demonstrates how the learning system captures experiences,
extracts patterns, and distills actionable lessons.

Key concepts:
1. Every execution creates an experience
2. Patterns emerge from repeated structures
3. Lessons encode what to do (or avoid) in context
4. Retrieval matches lessons to current situation

This is how I learn without weight updates - through accumulated
experiences stored externally and retrieved contextually.
"""

from pathlib import Path
from datetime import datetime
import sys
import shutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType
)


def demonstrate_experience_capture():
    """Show how experiences are captured during execution."""

    print("=" * 60)
    print("Demo 1: Experience Capture")
    print("=" * 60)

    storage_dir = Path("/tmp/learning_demo")
    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    cycle = LearningCycle(storage_dir)

    # Create context for the work
    context = Context(
        goal_type="implementation",
        goal_complexity="moderate",
        domain="authentication",
        available_tools=["code_editor", "terminal", "browser"],
        available_agents=1
    )

    print(f"\nContext: {context.goal_type} in {context.domain} domain")

    # Start tracking an experience
    experience = cycle.start_experience(
        context=context,
        intent="Implement JWT authentication",
        strategy="test_driven_development"
    )

    print(f"Experience started: {experience.id[:20]}...")

    # Record actions taken
    actions = [
        Action(
            action_type="read_file",
            description="Read existing auth module",
            target="/src/auth.py"
        ),
        Action(
            action_type="write_test",
            description="Write test for token generation",
            target="/tests/test_auth.py"
        ),
        Action(
            action_type="implement",
            description="Implement token generation",
            target="/src/auth.py"
        ),
        Action(
            action_type="run_tests",
            description="Run test suite",
            target="pytest"
        )
    ]

    for action in actions:
        experience.add_action(action)
        print(f"  Recorded: {action.action_type} -> {action.target}")

    # Complete with outcome
    outcome = Outcome(
        outcome_type=OutcomeType.SUCCESS,
        description="JWT authentication implemented",
        achieved=["token generation", "token verification"],
        quality_score=0.9,
        efficiency_score=0.8
    )

    reflection = {
        "worked": ["TDD approach caught edge cases early"],
        "didnt_work": ["Initial token expiry was too short"],
        "different": ["Would use a dedicated crypto library"]
    }

    cycle.complete_experience(experience, outcome, reflection)

    print(f"\nExperience completed: {outcome.outcome_type.name}")
    print(f"Quality: {outcome.quality_score}")
    print(f"Reflection: {reflection['worked'][0]}")


def demonstrate_pattern_extraction():
    """Show how patterns are extracted from experiences."""

    print("\n" + "=" * 60)
    print("Demo 2: Pattern Extraction")
    print("=" * 60)

    storage_dir = Path("/tmp/learning_demo")
    cycle = LearningCycle(storage_dir)

    # Create multiple experiences to form patterns
    print("\nSimulating multiple similar experiences...")

    # Several successful TDD experiences
    for i in range(5):
        context = Context(
            goal_type="implementation",
            goal_complexity="moderate",
            domain=["authentication", "api", "database"][i % 3]
        )

        exp = cycle.start_experience(
            context=context,
            intent=f"Implement feature {i}",
            strategy="test_driven_development"
        )

        # Same action pattern: read -> test -> implement -> verify
        exp.add_action(Action("read", "Read existing code", "/src/"))
        exp.add_action(Action("write_test", "Write tests first", "/tests/"))
        exp.add_action(Action("implement", "Write implementation", "/src/"))
        exp.add_action(Action("run_tests", "Verify with tests", "pytest"))

        cycle.complete_experience(
            exp,
            Outcome(
                outcome_type=OutcomeType.SUCCESS,
                description="Feature implemented",
                quality_score=0.85
            )
        )
        print(f"  Experience {i+1}: TDD approach -> SUCCESS")

    # Some failed non-TDD experiences
    for i in range(3):
        context = Context(
            goal_type="implementation",
            goal_complexity="moderate",
            domain="frontend"
        )

        exp = cycle.start_experience(
            context=context,
            intent=f"Implement feature {i}",
            strategy="code_first"  # Different strategy
        )

        exp.add_action(Action("implement", "Write code directly", "/src/"))
        exp.add_action(Action("run_tests", "Test after", "pytest"))

        cycle.complete_experience(
            exp,
            Outcome(
                outcome_type=OutcomeType.FAILURE,
                description="Bugs found late",
                error_message="Tests revealed issues after implementation"
            )
        )
        print(f"  Experience {5+i+1}: code_first approach -> FAILURE")

    # Extract patterns
    print("\nExtracting patterns...")
    results = cycle.extract_and_distill()

    print(f"\nPatterns extracted:")
    print(f"  Sequence patterns: {results['sequence_patterns']}")
    print(f"  Strategy patterns: {results['strategy_patterns']}")
    print(f"  Antipatterns: {results['antipatterns']}")
    print(f"  Lessons distilled: {results['lessons']}")


def demonstrate_lesson_retrieval():
    """Show how lessons are retrieved for new situations."""

    print("\n" + "=" * 60)
    print("Demo 3: Lesson Retrieval")
    print("=" * 60)

    storage_dir = Path("/tmp/learning_demo")
    cycle = LearningCycle(storage_dir)

    # New situation similar to past experiences
    new_context = Context(
        goal_type="implementation",
        goal_complexity="moderate",
        domain="security"  # Similar to authentication
    )

    print(f"\nNew context: {new_context.goal_type} in {new_context.domain}")

    # Get guidance
    guidance = cycle.get_guidance(new_context, include_experiences=True)

    print(f"\nGuidance retrieved:")

    if guidance['lessons']:
        print(f"\n  Lessons ({len(guidance['lessons'])}):")
        for lesson in guidance['lessons'][:3]:
            print(f"    • {lesson.title}")
            print(f"      Confidence: {lesson.confidence:.0%}")

    if guidance['recommendations']:
        print(f"\n  Recommendations:")
        for rec in guidance['recommendations'][:3]:
            print(f"    ✓ {rec}")

    if guidance['warnings']:
        print(f"\n  Warnings:")
        for warn in guidance['warnings'][:3]:
            print(f"    ⚠ {warn}")

    if guidance['relevant_successes']:
        print(f"\n  Relevant past successes ({len(guidance['relevant_successes'])}):")
        for exp in guidance['relevant_successes'][:2]:
            print(f"    • {exp.intent} (strategy: {exp.strategy_used})")

    if guidance['relevant_failures']:
        print(f"\n  Relevant past failures ({len(guidance['relevant_failures'])}):")
        for exp in guidance['relevant_failures'][:2]:
            print(f"    • {exp.intent} (strategy: {exp.strategy_used})")


def demonstrate_lesson_validation():
    """Show how lessons are validated over time."""

    print("\n" + "=" * 60)
    print("Demo 4: Lesson Validation")
    print("=" * 60)

    storage_dir = Path("/tmp/learning_demo")
    cycle = LearningCycle(storage_dir)

    # Get a lesson
    context = Context(goal_type="implementation", goal_complexity="moderate", domain="api")
    lessons = cycle.distiller.get_lessons_for_context(context)

    if not lessons:
        print("\nNo lessons available for validation demo")
        return

    lesson = lessons[0]
    print(f"\nLesson: {lesson.title}")
    print(f"Initial confidence: {lesson.confidence:.0%}")

    # Apply lesson and record outcome
    print("\nSimulating lesson application...")

    # Lesson was helpful
    cycle.validate_lesson(lesson.id, was_helpful=True)
    print(f"  Applied lesson -> was helpful")
    print(f"  Confidence now: {lesson.confidence:.0%}")

    # Apply again
    cycle.validate_lesson(lesson.id, was_helpful=True)
    print(f"  Applied lesson -> was helpful")
    print(f"  Confidence now: {lesson.confidence:.0%}")

    # One time it wasn't helpful
    cycle.validate_lesson(lesson.id, was_helpful=False)
    print(f"  Applied lesson -> was NOT helpful")
    print(f"  Confidence now: {lesson.confidence:.0%}")

    print(f"\nValidation count: {lesson.validation_count}")


def demonstrate_learning_stats():
    """Show overall learning system statistics."""

    print("\n" + "=" * 60)
    print("Demo 5: Learning Statistics")
    print("=" * 60)

    storage_dir = Path("/tmp/learning_demo")
    cycle = LearningCycle(storage_dir)

    stats = cycle.get_stats()

    print(f"\nLearning System Statistics:")
    print(f"  Total experiences: {stats['total_experiences']}")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Total lessons: {stats['total_lessons']}")

    print(f"\nPatterns by type:")
    for ptype, count in stats['patterns_by_type'].items():
        print(f"  {ptype}: {count}")

    print(f"\nHigh-confidence lessons: {stats['high_confidence_lessons']}")


def main():
    """Run all learning demonstrations."""

    demonstrate_experience_capture()
    demonstrate_pattern_extraction()
    demonstrate_lesson_retrieval()
    demonstrate_lesson_validation()
    demonstrate_learning_stats()

    print("\n" + "=" * 60)
    print("LEARNING DEMO COMPLETE")
    print("=" * 60)
    print("""
Key Takeaways:
1. Experiences are captured automatically during execution
2. Patterns emerge from repeated structures (min 3 occurrences)
3. Lessons encode actionable guidance with confidence levels
4. Similar contexts retrieve relevant lessons and past experiences
5. Validation improves lesson confidence over time

This is how I learn without neural updates - through accumulated
experiences that inform future decisions contextually.
""")


if __name__ == "__main__":
    main()
