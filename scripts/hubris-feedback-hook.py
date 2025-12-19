#!/usr/bin/env python3
"""
Hubris Feedback Hook - Wire predictions to git commits

This script is called by git hooks to integrate the MoE feedback loop:
- prepare-commit-msg: Records expert prediction before commit
- post-commit: Evaluates prediction accuracy and updates credits

Usage:
    # Called by prepare-commit-msg hook
    python scripts/hubris-feedback-hook.py pre-commit "commit message"

    # Called by post-commit hook
    python scripts/hubris-feedback-hook.py post-commit <commit-hash>

The feedback loop enables experts to learn from actual outcomes:
1. Pre-commit: Expert predicts which files will be modified
2. Post-commit: Compares prediction to actual files
3. Credit update: Rewards accurate predictions, penalizes inaccurate ones
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Add hubris to path
SCRIPT_DIR = Path(__file__).parent
HUBRIS_DIR = SCRIPT_DIR / 'hubris'
sys.path.insert(0, str(HUBRIS_DIR))

from feedback_collector import on_pre_commit, on_post_commit, PredictionRecorder
from credit_account import CreditLedger

# Paths
ML_DATA_DIR = Path('.git-ml')
MODEL_DIR = ML_DATA_DIR / 'models' / 'hubris'
LEDGER_PATH = MODEL_DIR / 'credit_ledger.json'
PREDICTION_STATE = ML_DATA_DIR / 'predictions' / 'pending_prediction.json'

# ANSI colors
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_experimental_banner():
    """Print EXPERIMENTAL warning for any output."""
    print(f"{YELLOW}{'─' * 60}{RESET}")
    print(f"{YELLOW}  EXPERIMENTAL: Hubris MoE Feedback Loop{RESET}")
    print(f"{YELLOW}  Predictions are learning - review before trusting{RESET}")
    print(f"{YELLOW}{'─' * 60}{RESET}")


def get_commit_files(commit_hash: str) -> list:
    """Get files modified in a commit."""
    try:
        result = subprocess.run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
            capture_output=True,
            text=True,
            check=True
        )
        return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
    except subprocess.CalledProcessError:
        return []


def load_ledger() -> CreditLedger:
    """Load or create credit ledger."""
    if LEDGER_PATH.exists():
        return CreditLedger.load(LEDGER_PATH)
    return CreditLedger()


def save_ledger(ledger: CreditLedger):
    """Save credit ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.save(LEDGER_PATH)


def do_pre_commit(message: str) -> int:
    """
    Record a prediction before commit.

    This is called by prepare-commit-msg to record what files
    the expert predicts will be in the commit.
    """
    # Check if Hubris experts are trained
    if not MODEL_DIR.exists():
        # Silently skip if not trained
        return 0

    # Get currently staged files as the "prediction"
    # (In a full implementation, we'd call the expert here)
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True,
            check=True
        )
        staged_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
    except subprocess.CalledProcessError:
        return 0

    if not staged_files:
        return 0

    # Try to get expert prediction for comparison
    # For now, we use the file expert's prediction based on commit message
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from hubris.expert_consolidator import ExpertConsolidator
        from hubris.credit_router import CreditRouter

        consolidator = ExpertConsolidator(model_dir=MODEL_DIR)

        if consolidator.experts and 'file' in consolidator.experts:
            expert = consolidator.experts['file']
            context = {'query': message}
            prediction = expert.predict(context)
            predicted_files = [f for f, _ in prediction.items[:10]]
            confidence = prediction.confidence
            expert_id = 'file_expert'
        else:
            # Fall back to staged files
            predicted_files = staged_files
            confidence = 0.5
            expert_id = 'staged_files'
    except Exception:
        # If anything fails, use staged files
        predicted_files = staged_files
        confidence = 0.5
        expert_id = 'staged_files'

    # Record the prediction
    prediction_id = on_pre_commit(
        task_description=message,
        expert_id=expert_id,
        predicted_files=predicted_files,
        confidence=confidence
    )

    # Store prediction ID for post-commit to find
    PREDICTION_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_STATE, 'w') as f:
        json.dump({
            'prediction_id': prediction_id,
            'expert_id': expert_id,
            'predicted_files': predicted_files,
            'confidence': confidence,
            'message': message
        }, f)

    return 0


def do_post_commit(commit_hash: str) -> int:
    """
    Evaluate prediction accuracy after commit.

    Compares what was predicted to what was actually committed,
    and updates expert credits accordingly.
    """
    # Check if there's a pending prediction
    if not PREDICTION_STATE.exists():
        return 0

    try:
        with open(PREDICTION_STATE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 0

    # Get actual files from commit
    actual_files = get_commit_files(commit_hash)

    if not actual_files:
        # Clean up state and exit
        PREDICTION_STATE.unlink(missing_ok=True)
        return 0

    # Load ledger
    ledger = load_ledger()

    # Process feedback
    credit_updates = on_post_commit(
        commit_hash=commit_hash,
        actual_files=actual_files,
        prediction_id=state.get('prediction_id'),
        ledger=ledger
    )

    # Save updated ledger
    save_ledger(ledger)

    # Clean up state
    PREDICTION_STATE.unlink(missing_ok=True)

    # Report results (only if there were updates)
    if credit_updates:
        print_experimental_banner()
        print(f"\n{CYAN}Hubris Feedback:{RESET}")

        predicted = set(state.get('predicted_files', []))
        actual = set(actual_files)
        correct = predicted & actual
        missed = actual - predicted
        extra = predicted - actual

        accuracy = len(correct) / len(actual) if actual else 0

        print(f"  Accuracy: {accuracy:.1%} ({len(correct)}/{len(actual)} files)")

        for expert_id, amount in credit_updates.items():
            if amount >= 0:
                print(f"  {GREEN}{expert_id}: +{amount:.1f} credits{RESET}")
            else:
                print(f"  {YELLOW}{expert_id}: {amount:.1f} credits{RESET}")

        if missed and len(missed) <= 3:
            print(f"  Missed: {', '.join(list(missed)[:3])}")

        print()

    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: hubris-feedback-hook.py <pre-commit|post-commit> [args]")
        return 1

    command = sys.argv[1]

    if command == 'pre-commit':
        message = sys.argv[2] if len(sys.argv) > 2 else ""
        return do_pre_commit(message)

    elif command == 'post-commit':
        commit_hash = sys.argv[2] if len(sys.argv) > 2 else 'HEAD'
        return do_post_commit(commit_hash)

    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
