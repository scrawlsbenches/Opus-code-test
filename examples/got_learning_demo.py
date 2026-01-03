#!/usr/bin/env python3
"""
Demo: GoT Learning Integration

Demonstrates how to integrate the LearningCycle with GoT task management
to capture experiences and retrieve lessons.
"""

from pathlib import Path
import tempfile
from cortical.got.learning_integration import GoTLearningBridge


def main():
    """Run the learning integration demo."""
    # Create temporary GoT directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        got_dir = Path(tmpdir) / ".got"
        got_dir.mkdir()

        print("=" * 70)
        print("GoT Learning Integration Demo")
        print("=" * 70)

        # Initialize the bridge
        bridge = GoTLearningBridge(got_dir)
        print(f"\nInitialized learning bridge at: {bridge.learning_dir}")

        # =====================================================================
        # 1. Capture a successful task completion
        # =====================================================================
        print("\n" + "-" * 70)
        print("1. Capturing successful task completion")
        print("-" * 70)

        exp1 = bridge.capture_task_completion(
            task_id="T-20260103-001",
            task_title="Implement user authentication API",
            task_category="feature",
            task_priority="high",
            approach="test-first",
            retrospective=(
                "TDD approach worked well for this API implementation. "
                "Tests helped catch edge cases early. "
                "Would use the same approach for similar features."
            ),
            files_changed=[
                "api/auth.py",
                "api/middleware.py",
                "tests/test_auth.py"
            ],
            duration_seconds=7200  # 2 hours
        )

        print(f"Created experience: {exp1.id}")
        print(f"  - Type: {exp1.experience_type.name}")
        print(f"  - Context: {exp1.context.goal_type} ({exp1.context.goal_complexity})")
        print(f"  - Outcome: {exp1.outcome.outcome_type.name}")
        print(f"  - Actions: {len(exp1.actions)}")
        print(f"  - Tags: {', '.join(sorted(exp1.tags))}")
        print(f"  - What worked: {len(exp1.what_worked)} items")

        # =====================================================================
        # 2. Capture a task failure
        # =====================================================================
        print("\n" + "-" * 70)
        print("2. Capturing task failure")
        print("-" * 70)

        exp2 = bridge.capture_task_failure(
            task_id="T-20260103-002",
            task_title="Integrate third-party payment API",
            task_category="feature",
            task_priority="critical",
            attempted_approach="direct-integration",
            error_message="API authentication failed - missing sandbox credentials",
            files_attempted=["payment/gateway.py", "payment/config.py"],
            blockers=[
                "Need sandbox API keys from vendor",
                "Documentation incomplete for auth flow"
            ]
        )

        print(f"Created failure experience: {exp2.id}")
        print(f"  - Outcome: {exp2.outcome.outcome_type.name}")
        print(f"  - Error: {exp2.outcome.error_message}")
        print(f"  - What didn't work: {len(exp2.what_didnt_work)} items")
        print(f"  - Context constraints: {exp2.context.constraints}")

        # =====================================================================
        # 3. Capture more successes for pattern building
        # =====================================================================
        print("\n" + "-" * 70)
        print("3. Capturing additional successes")
        print("-" * 70)

        for i in range(3, 6):
            exp = bridge.capture_task_completion(
                task_id=f"T-20260103-00{i}",
                task_title=f"Feature implementation #{i}",
                task_category="feature",
                approach="test-first",
                retrospective="TDD approach continues to work well.",
                files_changed=[f"api/feature{i}.py", f"tests/test_feature{i}.py"]
            )
            print(f"  - Captured: {exp.id}")

        # =====================================================================
        # 4. Request guidance for a new task
        # =====================================================================
        print("\n" + "-" * 70)
        print("4. Requesting guidance for new task")
        print("-" * 70)

        guidance = bridge.get_guidance_for_task(
            task_title="Implement OAuth2 authentication",
            task_category="feature",
            task_priority="high",
            files_to_modify=["api/oauth.py", "api/tokens.py"]
        )

        print(f"Guidance retrieved:")
        print(f"  - Lessons: {len(guidance['lessons'])}")
        print(f"  - Recommendations: {len(guidance['recommendations'])}")
        print(f"  - Warnings: {len(guidance['warnings'])}")
        print(f"  - Relevant successes: {len(guidance['relevant_successes'])}")
        print(f"  - Relevant failures: {len(guidance['relevant_failures'])}")

        if guidance['relevant_successes']:
            print("\n  Recent successes:")
            for exp in guidance['relevant_successes'][:3]:
                print(f"    - {exp.intent} ({exp.outcome.outcome_type.name})")

        if guidance['relevant_failures']:
            print("\n  Recent failures to avoid:")
            for exp in guidance['relevant_failures'][:3]:
                print(f"    - {exp.intent}: {exp.outcome.error_message}")

        # =====================================================================
        # 5. Link a task to related experiences
        # =====================================================================
        print("\n" + "-" * 70)
        print("5. Finding related experiences")
        print("-" * 70)

        related = bridge.link_task_to_experiences(
            task_id="T-20260103-999",
            task_category="feature",
            task_title="New feature development"
        )

        print(f"Found {len(related)} related experiences:")
        for exp in related[:5]:
            print(f"  - {exp.id}: {exp.intent} ({exp.outcome.outcome_type.name})")

        # =====================================================================
        # 6. Extract patterns and lessons
        # =====================================================================
        print("\n" + "-" * 70)
        print("6. Extracting patterns and lessons")
        print("-" * 70)

        results = bridge.extract_patterns_and_lessons()

        print("Pattern extraction results:")
        print(f"  - Sequence patterns: {results['sequence_patterns']}")
        print(f"  - Strategy patterns: {results['strategy_patterns']}")
        print(f"  - Anti-patterns: {results['antipatterns']}")
        print(f"  - Lessons distilled: {results['lessons']}")

        # =====================================================================
        # 7. Get learning statistics
        # =====================================================================
        print("\n" + "-" * 70)
        print("7. Learning system statistics")
        print("-" * 70)

        stats = bridge.get_learning_stats()

        print("Current learning stats:")
        print(f"  - Total experiences: {stats['total_experiences']}")
        print(f"  - Total patterns: {stats['total_patterns']}")
        print(f"  - Total lessons: {stats['total_lessons']}")
        print(f"  - High confidence lessons: {stats['high_confidence_lessons']}")

        if stats['patterns_by_type']:
            print("\n  Patterns by type:")
            for ptype, count in stats['patterns_by_type'].items():
                print(f"    - {ptype}: {count}")

        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
