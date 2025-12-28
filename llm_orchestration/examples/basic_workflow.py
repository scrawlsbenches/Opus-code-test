#!/usr/bin/env python3
"""
Basic Workflow Example: QAPV Reasoning Cycle

This example demonstrates the fundamental Question-Answer-Produce-Verify
cycle that forms the core of structured reasoning.

The QAPV pattern:
1. QUESTION - What are we trying to understand/accomplish?
2. ANSWER - Research, explore, form hypotheses, make decisions
3. PRODUCE - Create the actual artifact (code, docs, etc.)
4. VERIFY - Check that what we produced is correct

This example walks through implementing a simple feature using QAPV.
"""

from pathlib import Path
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_state import (
    CognitiveStateManager,
    QuestionStatus,
    HypothesisStatus,
    DecisionStatus
)
from thought_patterns import QAPVPattern, create_pattern


def main():
    """Run through a complete QAPV cycle."""

    print("=" * 60)
    print("QAPV Reasoning Cycle Example")
    print("=" * 60)

    # Initialize cognitive state
    state_dir = Path("/tmp/qapv_example")
    state = CognitiveStateManager(state_dir)

    # Create the pattern
    pattern = create_pattern("qapv")

    # =========================================================================
    # PHASE 1: QUESTION
    # =========================================================================
    print("\n--- PHASE 1: QUESTION ---")
    pattern.start()

    # The main question we're trying to answer
    main_question = state.add_question(
        content="How should we implement user authentication?",
        context={"project": "example_app", "constraints": ["simple", "secure"]}
    )
    print(f"Main question: {main_question.content}")

    # Break into sub-questions
    sub_questions = [
        state.add_question(
            "What authentication method should we use?",
            parent_id=main_question.id
        ),
        state.add_question(
            "How do we store credentials securely?",
            parent_id=main_question.id
        ),
        state.add_question(
            "What libraries are available?",
            parent_id=main_question.id
        )
    ]

    for sq in sub_questions:
        print(f"  Sub-question: {sq.content}")

    pattern.add_note("Identified main question and 3 sub-questions")

    # =========================================================================
    # PHASE 2: ANSWER
    # =========================================================================
    print("\n--- PHASE 2: ANSWER ---")
    pattern.transition("answer")

    # Research and form hypotheses
    print("\nForming hypotheses...")

    hypotheses = {
        "auth_method": [
            state.add_hypothesis(
                sub_questions[0].id,
                "Use JWT tokens for stateless auth",
                rationale="Works well with microservices, no session storage needed"
            ),
            state.add_hypothesis(
                sub_questions[0].id,
                "Use session-based auth with cookies",
                rationale="Simple, well-understood, browser-native"
            )
        ],
        "storage": [
            state.add_hypothesis(
                sub_questions[1].id,
                "Use bcrypt for password hashing",
                rationale="Industry standard, handles salting automatically"
            )
        ],
        "libraries": [
            state.add_hypothesis(
                sub_questions[2].id,
                "Use passlib + python-jose",
                rationale="Well-maintained, good documentation"
            )
        ]
    }

    for category, hyps in hypotheses.items():
        for h in hyps:
            print(f"  Hypothesis ({category}): {h.statement}")

    # Evaluate hypotheses (simulating evidence gathering)
    print("\nEvaluating hypotheses...")

    # JWT is better for our use case
    hypotheses["auth_method"][0].add_evidence(
        "Works with API clients",
        supports=True,
        strength=0.9
    )
    hypotheses["auth_method"][0].add_evidence(
        "No server-side session storage needed",
        supports=True,
        strength=0.8
    )
    hypotheses["auth_method"][1].add_evidence(
        "Requires session storage infrastructure",
        supports=False,
        strength=0.7
    )

    state.evaluate_hypothesis(hypotheses["auth_method"][0].id)
    state.evaluate_hypothesis(hypotheses["auth_method"][1].id)

    # Make decisions
    print("\nMaking decisions...")

    decisions = [
        state.add_decision(
            question_id=sub_questions[0].id,
            choice="Use JWT tokens",
            rationale="Better for API-first design, no session storage needed",
            alternatives=["Session cookies", "OAuth only"]
        ),
        state.add_decision(
            question_id=sub_questions[1].id,
            choice="Use bcrypt via passlib",
            rationale="Industry standard, battle-tested",
            alternatives=["argon2", "scrypt"]
        ),
        state.add_decision(
            question_id=sub_questions[2].id,
            choice="passlib + python-jose",
            rationale="Good compatibility, active maintenance",
            alternatives=["PyJWT", "authlib"]
        )
    ]

    for d in decisions:
        print(f"  Decision: {d.choice}")
        print(f"    Rationale: {d.rationale}")

    # Answer the sub-questions
    for sq in sub_questions:
        state.answer_question(sq.id, f"See decision for question {sq.id[:8]}...")

    pattern.add_note("Made 3 key decisions based on hypothesis evaluation")

    # =========================================================================
    # PHASE 3: PRODUCE
    # =========================================================================
    print("\n--- PHASE 3: PRODUCE ---")
    pattern.transition("produce")

    # Now we would actually write the code
    # This is a simulation - in real use, this would be actual implementation

    artifact = """
# auth.py - Generated based on decisions above

from passlib.hash import bcrypt
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"  # In production, use env var
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

def hash_password(password: str) -> str:
    '''Hash password using bcrypt.'''
    return bcrypt.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    '''Verify password against hash.'''
    return bcrypt.verify(password, hashed)

def create_token(user_id: str) -> str:
    '''Create JWT token for user.'''
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    '''Verify and decode JWT token.'''
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
"""

    print("Produced artifact: auth.py")
    print("-" * 40)
    print(artifact[:500] + "...")

    pattern.add_note("Generated auth.py implementing JWT authentication")

    # =========================================================================
    # PHASE 4: VERIFY
    # =========================================================================
    print("\n--- PHASE 4: VERIFY ---")
    pattern.transition("verify")

    # Verification checks (simulated)
    verifications = [
        ("Password hashing uses bcrypt", True),
        ("Tokens include expiration", True),
        ("Secret key is configurable", True),
        ("Token verification handles errors", False),  # Found an issue!
    ]

    print("Running verifications...")
    all_passed = True
    for check, passed in verifications:
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\n⚠️  Some verifications failed!")
        print("   Need to add error handling to verify_token()")

        # In QAPV, we would loop back to PRODUCE to fix
        pattern.add_note("Verification found missing error handling")
        pattern.add_note("Would loop back to PRODUCE to fix")
    else:
        print("\n✓ All verifications passed!")

    # Complete the main question
    state.answer_question(
        main_question.id,
        "Implemented JWT-based auth with bcrypt password hashing"
    )

    pattern.complete()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("QAPV CYCLE COMPLETE")
    print("=" * 60)

    summary = pattern.get_summary()
    print(f"\nPhases completed: {summary['phases_completed']}")
    print(f"Total duration: {summary['total_time']}")
    print(f"Notes recorded: {len(summary['notes'])}")

    print(f"\nCognitive State:")
    print(f"  Questions: {len(state.questions)}")
    print(f"  Hypotheses: {len(state.hypotheses)}")
    print(f"  Decisions: {len(state.decisions)}")

    # Save state for potential continuation
    state.save_checkpoint()
    print(f"\nState saved to: {state_dir}")


if __name__ == "__main__":
    main()
