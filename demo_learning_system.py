#!/usr/bin/env python3
"""
DEMO: Learning System with Semantic Matching and File Risk Tracking

This demonstrates the new capabilities we just built:
1. Semantic intent matching (find similar tasks by meaning)
2. File outcome tracking (know which files are risky)
3. Combined guidance for agents
"""

import tempfile
from pathlib import Path
import shutil

from llm_orchestration.learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)


def create_demo_experiences(cycle: LearningCycle):
    """Populate the learning system with realistic experiences."""

    print("=" * 70)
    print("PHASE 1: Recording Past Experiences")
    print("=" * 70)

    # === SUCCESSFUL JWT Implementation ===
    exp1 = cycle.start_experience(
        context=Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="authentication",
        ),
        intent="Implement JWT token authentication for REST API",
        experience_type=ExperienceType.TASK_EXECUTION,
    )
    exp1.add_action(Action("write_code", "Create JWT validator", "src/auth/jwt.py"))
    exp1.add_action(Action("write_code", "Add token middleware", "src/middleware/auth.py"))
    exp1.add_action(Action("write_test", "JWT unit tests", "tests/test_jwt.py"))
    exp1.reflect(
        what_worked=["Test-first approach caught edge cases early",
                     "Used existing crypto library for signing"],
        what_didnt_work=[],
        would_do_differently=[]
    )
    cycle.complete_experience(exp1, Outcome(
        outcome_type=OutcomeType.SUCCESS,
        description="JWT authentication working in production",
    ))
    print(f"✓ Recorded: {exp1.intent}")

    # === FAILED auth.py modification ===
    exp2 = cycle.start_experience(
        context=Context(
            goal_type="bugfix",
            goal_complexity="complex",
            domain="authentication",
        ),
        intent="Fix token expiry edge case in authentication",
        experience_type=ExperienceType.TASK_EXECUTION,
    )
    exp2.add_action(Action("write_code", "Fix expiry logic", "src/auth/jwt.py"))
    exp2.reflect(
        what_worked=[],
        what_didnt_work=["Changed expiry without updating tests",
                        "Broke backward compatibility"],
        would_do_differently=["Run full test suite before commit"]
    )
    cycle.complete_experience(exp2, Outcome(
        outcome_type=OutcomeType.FAILURE,
        description="Broke existing token validation",
        error_type="RegressionError",
        error_message="Existing tokens rejected after deploy",
    ))
    print(f"✗ Recorded: {exp2.intent} (FAILED)")

    # === Another FAILED auth.py modification ===
    exp3 = cycle.start_experience(
        context=Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="authentication",
        ),
        intent="Add refresh token support to JWT system",
        experience_type=ExperienceType.TASK_EXECUTION,
    )
    exp3.add_action(Action("write_code", "Add refresh logic", "src/auth/jwt.py"))
    exp3.add_action(Action("write_code", "Update token store", "src/auth/store.py"))
    exp3.reflect(
        what_worked=[],
        what_didnt_work=["Circular import introduced",
                        "Token store not thread-safe"],
        would_do_differently=["Check import graph before adding dependencies"]
    )
    cycle.complete_experience(exp3, Outcome(
        outcome_type=OutcomeType.FAILURE,
        description="Circular import broke the auth module",
        error_type="ImportError",
        error_message="circular import detected in auth module",
    ))
    print(f"✗ Recorded: {exp3.intent} (FAILED)")

    # === Successful database work ===
    exp4 = cycle.start_experience(
        context=Context(
            goal_type="feature_implementation",
            goal_complexity="simple",
            domain="database",
        ),
        intent="Add connection pooling to database layer",
        experience_type=ExperienceType.TASK_EXECUTION,
    )
    exp4.add_action(Action("write_code", "Pool implementation", "src/db/pool.py"))
    exp4.add_action(Action("write_test", "Pool tests", "tests/test_pool.py"))
    exp4.reflect(
        what_worked=["Used established pooling pattern",
                    "Comprehensive timeout tests"],
        what_didnt_work=[],
        would_do_differently=[]
    )
    cycle.complete_experience(exp4, Outcome(
        outcome_type=OutcomeType.SUCCESS,
        description="Connection pooling reduces DB load by 60%",
    ))
    print(f"✓ Recorded: {exp4.intent}")

    # === Successful OAuth work (related to auth) ===
    exp5 = cycle.start_experience(
        context=Context(
            goal_type="feature_implementation",
            goal_complexity="complex",
            domain="authentication",
        ),
        intent="Implement OAuth2 provider integration",
        experience_type=ExperienceType.TASK_EXECUTION,
    )
    exp5.add_action(Action("write_code", "OAuth flow", "src/auth/oauth.py"))
    exp5.add_action(Action("write_code", "Token exchange", "src/auth/exchange.py"))
    exp5.add_action(Action("write_test", "OAuth tests", "tests/test_oauth.py"))
    exp5.reflect(
        what_worked=["Followed OAuth2 RFC strictly",
                    "Mock provider for testing"],
        what_didnt_work=[],
        would_do_differently=[]
    )
    cycle.complete_experience(exp5, Outcome(
        outcome_type=OutcomeType.SUCCESS,
        description="OAuth2 working with Google and GitHub",
    ))
    print(f"✓ Recorded: {exp5.intent}")

    print(f"\nTotal experiences recorded: 5")
    print()


def demo_semantic_matching(cycle: LearningCycle):
    """Demonstrate semantic intent matching."""

    print("=" * 70)
    print("PHASE 2: Semantic Intent Matching Demo")
    print("=" * 70)

    # Search for JWT-related tasks
    search_intent = "Add JWT token validation"
    print(f"\nSearching for: \"{search_intent}\"")
    print("-" * 50)

    results = cycle.find_by_intent(search_intent, min_similarity=0.1, limit=5)

    for i, exp in enumerate(results, 1):
        similarity = cycle.intent_similarity(search_intent, exp.intent)
        outcome = "✓" if exp.outcome.was_successful() else "✗"
        print(f"{i}. [{outcome}] {exp.intent}")
        print(f"   Similarity: {similarity:.1%}")
        keywords_search = cycle.extract_keywords(search_intent)
        keywords_exp = cycle.extract_keywords(exp.intent)
        shared = keywords_search & keywords_exp
        print(f"   Shared keywords: {shared}")
        print()

    # Compare to unrelated search
    unrelated_search = "Fix database connection timeout"
    print(f"\nSearching for: \"{unrelated_search}\"")
    print("-" * 50)

    unrelated_results = cycle.find_by_intent(unrelated_search, min_similarity=0.1, limit=5)
    if unrelated_results:
        for exp in unrelated_results:
            similarity = cycle.intent_similarity(unrelated_search, exp.intent)
            print(f"  Found: {exp.intent} (similarity: {similarity:.1%})")
    else:
        print("  No matches found (as expected for unrelated task)")
    print()


def demo_file_risk_tracking(cycle: LearningCycle):
    """Demonstrate file risk tracking."""

    print("=" * 70)
    print("PHASE 3: File Risk Tracking Demo")
    print("=" * 70)

    # Check history for the risky file
    print("\nFile History: src/auth/jwt.py")
    print("-" * 50)
    history = cycle.get_file_history("src/auth/jwt.py")
    print(f"  Total touches: {history['total_experiences']}")
    print(f"  Successes: {history['success_count']}")
    print(f"  Failures: {history['failure_count']}")
    print(f"  Success rate: {history['success_rate']:.0%}")
    if history['error_patterns']:
        print(f"  Error patterns:")
        for error, count in history['error_patterns'].items():
            print(f"    - {error}: {count} occurrences")
    print()

    # Check history for a safe file
    print("File History: src/db/pool.py")
    print("-" * 50)
    safe_history = cycle.get_file_history("src/db/pool.py")
    print(f"  Total touches: {safe_history['total_experiences']}")
    print(f"  Successes: {safe_history['success_count']}")
    print(f"  Failures: {safe_history['failure_count']}")
    print(f"  Success rate: {safe_history['success_rate']:.0%}")
    print()

    # Get risky files
    print("Risky Files (≥2 touches, <50% success rate):")
    print("-" * 50)
    risky = cycle.get_risky_files(min_experiences=2, max_success_rate=0.5)
    if risky:
        for r in risky:
            print(f"  ⚠️  {r['file_path']}")
            print(f"      Success rate: {r['success_rate']:.0%}")
            print(f"      Failures: {r['failure_count']}")
            if r['error_patterns']:
                print(f"      Common errors: {list(r['error_patterns'].keys())}")
    else:
        print("  No risky files found")
    print()


def demo_combined_guidance(cycle: LearningCycle):
    """Demonstrate combined guidance for a new task."""

    print("=" * 70)
    print("PHASE 4: Combined Guidance for New Task")
    print("=" * 70)

    new_task = "Add API key authentication to JWT system"
    files_to_modify = ["src/auth/jwt.py", "src/auth/api_keys.py"]

    print(f"\nNew Task: \"{new_task}\"")
    print(f"Files to modify: {files_to_modify}")
    print("-" * 50)

    guidance = cycle.get_guidance_for_files(
        intent=new_task,
        files_to_modify=files_to_modify,
    )

    print("\n📚 SIMILAR EXPERIENCES:")
    for exp in guidance.get('similar_experiences', [])[:3]:
        outcome = "✓" if exp.outcome and exp.outcome.was_successful() else "✗"
        print(f"  [{outcome}] {exp.intent}")

    print("\n⚠️  WARNINGS:")
    for warning in guidance.get('warnings', []):
        print(f"  • {warning}")

    print("\n💡 RECOMMENDATIONS:")
    for rec in guidance.get('recommendations', []):
        print(f"  • {rec}")

    print("\n📁 FILE RISK ASSESSMENT:")
    for file_path, risk in guidance.get('file_risks', {}).items():
        status = "🔴 RISKY" if risk['is_risky'] else "🟢 SAFE"
        print(f"  {file_path}: {status}")
        print(f"    Success rate: {risk['success_rate']:.0%}")
        print(f"    Past failures: {risk['failure_count']}")
        if risk.get('error_patterns'):
            print(f"    Common errors: {list(risk['error_patterns'].keys())}")
    print()


def main():
    """Run the full demo."""
    print("\n" + "=" * 70)
    print("   LEARNING SYSTEM DEMO")
    print("   Semantic Matching + File Risk Tracking")
    print("=" * 70 + "\n")

    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp())

    try:
        cycle = LearningCycle(temp_dir)

        # Run demo phases
        create_demo_experiences(cycle)
        demo_semantic_matching(cycle)
        demo_file_risk_tracking(cycle)
        demo_combined_guidance(cycle)

        print("=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("""
Key Takeaways:
1. Semantic matching finds related tasks by meaning, not just category
2. File tracking identifies risky files from historical failures
3. Combined guidance helps agents avoid past mistakes
4. Error patterns highlight recurring issues per file
""")

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
