#!/usr/bin/env python3
"""
Multi-Session Example: State Persistence Across Sessions

This example demonstrates how cognitive state persists across sessions,
allowing work to continue where it left off.

Key concepts:
1. State is externalized to files
2. Checkpoints capture complete state
3. New sessions can restore and continue
4. Progress is not lost on session end

This is critical for me (the LLM) because I have no inherent memory
between sessions - all continuity must come from external state.
"""

from pathlib import Path
from datetime import datetime
import sys
import shutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_state import (
    CognitiveStateManager,
    QuestionStatus,
    DecisionStatus
)


def simulate_session_1(state_dir: Path):
    """First session: Start work, make progress, checkpoint."""

    print("=" * 60)
    print("SESSION 1: Starting Work")
    print("=" * 60)

    state = CognitiveStateManager(state_dir)

    # Set up initial focus
    state.set_focus(
        current_goal="Build a REST API for user management",
        context={"project": "user_service", "deadline": "next_week"}
    )
    print(f"\nFocus: {state.focus.current_goal}")

    # Create main question
    main_q = state.add_question(
        "What's the best architecture for the user management API?"
    )
    print(f"Main question: {main_q.content}")

    # Research and form hypotheses
    h1 = state.add_hypothesis(
        main_q.id,
        "Use Flask with SQLAlchemy",
        rationale="Simple, well-documented, good for small services"
    )
    h2 = state.add_hypothesis(
        main_q.id,
        "Use FastAPI with SQLModel",
        rationale="Modern, async-native, automatic OpenAPI docs"
    )

    print(f"\nHypotheses formed: 2")

    # Evaluate one hypothesis
    h2.add_evidence(
        "Built-in request validation via Pydantic",
        supports=True,
        strength=0.9
    )
    h2.add_evidence(
        "Native async support for database operations",
        supports=True,
        strength=0.8
    )
    state.evaluate_hypothesis(h2.id)

    print(f"Evaluated hypothesis: {h2.id[:8]}...")

    # Make a decision
    decision = state.add_decision(
        question_id=main_q.id,
        choice="Use FastAPI with SQLModel",
        rationale="Modern stack, better async support, automatic docs",
        alternatives=["Flask + SQLAlchemy", "Django REST Framework"]
    )

    print(f"\nDecision made: {decision.choice}")

    # Record some observations
    state.add_observation(
        content="FastAPI requires Python 3.7+",
        source="documentation review"
    )
    state.add_observation(
        content="SQLModel is maintained by the FastAPI author",
        source="GitHub research"
    )

    print(f"Observations recorded: 2")

    # Save checkpoint before session ends
    checkpoint = state.save_checkpoint()
    print(f"\n✓ Checkpoint saved: {checkpoint['id'][:20]}...")

    # Session ends here
    print("\n[Session 1 ending...]")
    print("-" * 60)

    return checkpoint


def simulate_session_2(state_dir: Path):
    """Second session: Restore state and continue work."""

    print("\n" + "=" * 60)
    print("SESSION 2: Resuming Work")
    print("=" * 60)

    # Create new state manager (simulates fresh session)
    state = CognitiveStateManager(state_dir)

    # Load the latest checkpoint
    checkpoint = state.load_latest_checkpoint()
    if not checkpoint:
        print("ERROR: No checkpoint found!")
        return

    print(f"\n✓ Restored from checkpoint: {checkpoint['id'][:20]}...")

    # Verify state was restored
    print(f"\nRestored state:")
    print(f"  Focus: {state.focus.current_goal if state.focus else 'None'}")
    print(f"  Questions: {len(state.questions)}")
    print(f"  Hypotheses: {len(state.hypotheses)}")
    print(f"  Decisions: {len(state.decisions)}")
    print(f"  Observations: {len(state.observations)}")

    # Find our previous decision
    previous_decision = list(state.decisions.values())[0]
    print(f"\nPrevious decision: {previous_decision.choice}")
    print(f"Rationale: {previous_decision.rationale}")

    # Continue work - add more questions
    impl_q = state.add_question(
        "How should we structure the database models?",
        context={"framework": "FastAPI", "orm": "SQLModel"}
    )
    print(f"\nNew question: {impl_q.content}")

    # Add more decisions
    model_decision = state.add_decision(
        question_id=impl_q.id,
        choice="Use SQLModel with UUID primary keys",
        rationale="UUIDs are better for distributed systems and security"
    )
    print(f"New decision: {model_decision.choice}")

    # Answer the implementation question
    state.answer_question(
        impl_q.id,
        "SQLModel with UUID PKs, separate models for API response"
    )

    # Answer the original question now that we have full plan
    main_q = list(state.questions.values())[0]
    state.answer_question(
        main_q.id,
        "FastAPI + SQLModel with UUID-based models and Pydantic schemas"
    )

    # Update focus to reflect progress
    state.set_focus(
        current_goal="Implement the user management API models",
        context={
            "decided": "FastAPI + SQLModel",
            "next_step": "Write User model"
        }
    )

    print(f"\nUpdated focus: {state.focus.current_goal}")

    # Save progress
    checkpoint = state.save_checkpoint()
    print(f"\n✓ New checkpoint saved: {checkpoint['id'][:20]}...")

    print("\n[Session 2 ending...]")
    print("-" * 60)


def simulate_session_3(state_dir: Path):
    """Third session: Show full history of work."""

    print("\n" + "=" * 60)
    print("SESSION 3: Reviewing Progress")
    print("=" * 60)

    state = CognitiveStateManager(state_dir)
    state.load_latest_checkpoint()

    print("\n--- Complete Decision History ---")
    for d_id, decision in state.decisions.items():
        print(f"\n[{d_id[:12]}...]")
        print(f"  Choice: {decision.choice}")
        print(f"  Rationale: {decision.rationale}")
        print(f"  Status: {decision.status.name}")
        print(f"  Made at: {decision.made_at}")

    print("\n--- Question Resolution ---")
    for q_id, question in state.questions.items():
        print(f"\n[{q_id[:12]}...]")
        print(f"  Question: {question.content}")
        print(f"  Status: {question.status.name}")
        if question.answer:
            print(f"  Answer: {question.answer}")

    print("\n--- Observations Made ---")
    for obs in state.observations.values():
        print(f"  • {obs.content} (from {obs.source})")

    print("\n--- Current Focus ---")
    if state.focus:
        print(f"  Goal: {state.focus.current_goal}")
        print(f"  Context: {state.focus.context}")

    # Show checkpoint history
    print("\n--- Checkpoint History ---")
    checkpoints = list(state.checkpoints_dir.glob("*.json"))
    for cp in sorted(checkpoints):
        print(f"  {cp.name}")


def main():
    """Run the multi-session demonstration."""

    # Use a consistent directory
    state_dir = Path("/tmp/multi_session_example")

    # Clean up from previous runs
    if state_dir.exists():
        shutil.rmtree(state_dir)

    # Run three sessions
    simulate_session_1(state_dir)
    simulate_session_2(state_dir)
    simulate_session_3(state_dir)

    print("\n" + "=" * 60)
    print("MULTI-SESSION DEMO COMPLETE")
    print("=" * 60)
    print("""
Key Takeaways:
1. State persisted across all three sessions
2. Decisions and rationale were preserved
3. Work could continue without re-explaining context
4. Full history is available for review

This is how I maintain continuity despite having no inherent memory.
""")


if __name__ == "__main__":
    main()
