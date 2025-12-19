#!/usr/bin/env python3
"""
Hubris MoE CLI - Command-line interface for the Mixture of Experts system

Commands:
    train            - Train all experts from collected data
    predict          - Get predictions for a task
    stats            - Show expert statistics
    leaderboard      - Show expert credit leaderboard
    evaluate         - Evaluate expert accuracy on recent commits
    calibration      - Show calibration analysis for predictions
    suggest-tests    - Suggest tests to run for code changes
    suggest-refactor - Suggest files that may need refactoring

Examples:
    python scripts/hubris_cli.py train --commits 100
    python scripts/hubris_cli.py predict "Add authentication feature"
    python scripts/hubris_cli.py stats --expert file_expert
    python scripts/hubris_cli.py leaderboard
    python scripts/hubris_cli.py evaluate --commits 20
    python scripts/hubris_cli.py calibration --curve
    python scripts/hubris_cli.py suggest-tests --staged
    python scripts/hubris_cli.py suggest-tests --files cortical/query/search.py
    python scripts/hubris_cli.py suggest-refactor --scan
    python scripts/hubris_cli.py suggest-refactor --files cortical/analysis.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add scripts/hubris to path
SCRIPT_DIR = Path(__file__).parent
HUBRIS_DIR = SCRIPT_DIR / 'hubris'
sys.path.insert(0, str(HUBRIS_DIR))

from expert_consolidator import ExpertConsolidator, load_experts
from credit_account import CreditLedger
from credit_router import CreditRouter
from calibration_tracker import CalibrationTracker
from test_calibration_tracker import TestCalibrationTracker

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def color(text: str, color_code: str) -> str:
    """Apply ANSI color to text."""
    return f"{color_code}{text}{Colors.END}"

# Paths
ML_DATA_DIR = Path('.git-ml')
MODEL_DIR = ML_DATA_DIR / 'models' / 'hubris'
LEDGER_PATH = MODEL_DIR / 'credit_ledger.json'
COMMITS_DIR = ML_DATA_DIR / 'commits'
CHATS_DIR = ML_DATA_DIR / 'chats'

def _is_test_file(file_path: str) -> bool:
    """Check if a file is a test file."""
    path_lower = file_path.lower()
    return (
        'test' in path_lower or
        path_lower.startswith('tests/') or
        path_lower.endswith('_test.py')
    )


def transform_commit_for_test_expert(commit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform commit data to include test_results from CI status.

    Uses heuristic approach when CI fails:
    - Changed test files are likely the ones that failed
    - If no test files changed, we can't determine which tests failed

    Args:
        commit: Raw commit data with optional ci_result field

    Returns:
        Commit dict with test_results field added if CI data available
    """
    # Skip if already has test_results
    if 'test_results' in commit:
        return commit

    # Check for CI status
    ci_result = commit.get('ci_result')
    if not ci_result:
        return commit

    # Get changed files
    files = commit.get('files', []) or commit.get('files_changed', [])
    if not files:
        return commit

    # Identify test files
    test_files = [f for f in files if _is_test_file(f)]

    # Build test_results based on CI outcome
    test_results = {}

    if ci_result == 'fail' and test_files:
        # Heuristic: If CI failed and test files changed, those tests likely failed
        test_results['failed'] = test_files
        test_results['passed'] = []
        test_results['source'] = 'ci_heuristic'
    elif ci_result == 'pass' and test_files:
        # CI passed - all changed tests passed
        test_results['failed'] = []
        test_results['passed'] = test_files
        test_results['source'] = 'ci_heuristic'

    # Only add if we have meaningful data
    if test_results:
        commit = commit.copy()
        commit['test_results'] = test_results

    return commit


def load_commit_data(limit: Optional[int] = None, include_ci: bool = False) -> List[Dict[str, Any]]:
    """
    Load commit data from .git-ml/commits/ or .git-ml/tracked/commits.jsonl.

    Args:
        limit: Maximum number of commits to load (most recent first)
        include_ci: Transform CI results into test_results format for TestExpert

    Returns:
        List of commit dictionaries
    """
    commits = []

    # Try JSONL format first (new format in .git-ml/tracked/)
    tracked_commits_file = ML_DATA_DIR / 'tracked' / 'commits.jsonl'
    if tracked_commits_file.exists():
        try:
            with open(tracked_commits_file) as f:
                for line in f:
                    if line.strip():
                        commit = json.loads(line)
                        # Transform CI data to test_results if requested
                        if include_ci:
                            commit = transform_commit_for_test_expert(commit)
                        commits.append(commit)
        except Exception as e:
            print(color(f"Warning: Failed to load {tracked_commits_file}: {e}", Colors.YELLOW))

    # Fall back to individual JSON files (legacy format)
    if not commits and COMMITS_DIR.exists():
        commit_files = sorted(COMMITS_DIR.glob('*.json'), reverse=True)

        if limit:
            commit_files = commit_files[:limit]

        for commit_file in commit_files:
            try:
                with open(commit_file) as f:
                    commit = json.load(f)
                    if include_ci:
                        commit = transform_commit_for_test_expert(commit)
                    commits.append(commit)
            except Exception as e:
                print(color(f"Warning: Failed to load {commit_file}: {e}", Colors.YELLOW))

    if not commits:
        print(color(f"Warning: No commit data found in {tracked_commits_file} or {COMMITS_DIR}", Colors.YELLOW))
        return []

    # Apply limit after loading
    if limit and len(commits) > limit:
        commits = commits[:limit]

    return commits

def load_transcript_data() -> List[Any]:
    """
    Load chat transcript data from .git-ml/chats/.

    Returns:
        List of chat transcripts
    """
    if not CHATS_DIR.exists():
        print(color(f"Warning: {CHATS_DIR} not found", Colors.YELLOW))
        return []

    transcripts = []
    chat_files = sorted(CHATS_DIR.glob('*.json'))

    for chat_file in chat_files:
        try:
            with open(chat_file) as f:
                chat = json.load(f)
                transcripts.append(chat)
        except Exception as e:
            print(color(f"Warning: Failed to load {chat_file}: {e}", Colors.YELLOW))

    return transcripts

def load_ledger() -> CreditLedger:
    """Load credit ledger or create new one."""
    if LEDGER_PATH.exists():
        return CreditLedger.load(LEDGER_PATH)
    else:
        return CreditLedger()

def save_ledger(ledger: CreditLedger) -> None:
    """Save credit ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.save(LEDGER_PATH)

def cmd_train(args) -> int:
    """Train all experts from collected data."""
    print(color("=" * 60, Colors.BOLD))
    print(color("TRAINING HUBRIS MoE EXPERTS", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))

    # Load data
    print("\nLoading training data...")
    include_ci = getattr(args, 'include_ci', False)
    commits = load_commit_data(limit=args.commits, include_ci=include_ci)
    transcripts = load_transcript_data() if args.transcripts else None

    print(f"  Commits: {len(commits)}")
    if include_ci:
        # Count commits with CI data
        ci_commits = sum(1 for c in commits if 'ci_result' in c)
        test_results = sum(1 for c in commits if 'test_results' in c)
        print(f"  CI data: {ci_commits} commits")
        print(f"  Test results: {test_results} commits (for TestExpert)")
    if transcripts:
        print(f"  Transcripts: {len(transcripts)}")

    if not commits and not transcripts:
        print(color("\nError: No training data found", Colors.RED))
        print("Run some commands and commit changes to collect data.")
        return 1

    # Load or create consolidator
    if MODEL_DIR.exists():
        print("\nLoading existing experts...")
        consolidator = ExpertConsolidator(model_dir=MODEL_DIR)
        print(f"  Loaded: {', '.join(consolidator.get_loaded_experts())}")
    else:
        print("\nCreating fresh experts...")
        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()

    # Train experts
    print("\nTraining experts...")
    results = consolidator.consolidate_training(
        commits=commits,
        transcripts=transcripts,
        errors=None  # TODO: Add error data loading
    )

    # Display results
    print(color("\nTraining Results:", Colors.BOLD))
    for expert_type, success in results.items():
        status = color("✓ Success", Colors.GREEN) if success else color("✗ Failed", Colors.RED)
        print(f"  {expert_type:15} {status}")

    # Save experts
    print(f"\nSaving experts to {MODEL_DIR}...")
    consolidator.save_all_experts(MODEL_DIR)

    # Show stats
    stats = consolidator.get_training_stats()
    print(color("\nExpert Statistics:", Colors.BOLD))
    for expert_type, stat in stats.items():
        print(f"  {expert_type}:")
        print(f"    Commits:  {stat['trained_on_commits']}")
        print(f"    Sessions: {stat['trained_on_sessions']}")
        print(f"    Version:  {stat['version']}")

    print(color("\n✓ Training complete!", Colors.GREEN))
    return 0

def print_experimental_banner() -> None:
    """Print the EXPERIMENTAL warning banner."""
    print(color("┏" + "━" * 68 + "┓", Colors.YELLOW))
    print(color("┃  ⚠️  EXPERIMENTAL - Hubris MoE Predictions                          ┃", Colors.YELLOW))
    print(color("┃                                                                      ┃", Colors.YELLOW))
    print(color("┃  These predictions are generated by a learning system with limited  ┃", Colors.YELLOW))
    print(color("┃  training data. Use as suggestions only - always verify results.    ┃", Colors.YELLOW))
    print(color("┃                                                                      ┃", Colors.YELLOW))
    print(color("┃  The system improves over time as it learns from actual outcomes.   ┃", Colors.YELLOW))
    print(color("┗" + "━" * 68 + "┛", Colors.YELLOW))


def is_cold_start(ledger: CreditLedger, threshold_balance: float = 100.0) -> bool:
    """
    Detect if the system is in cold-start mode.

    Cold-start is when all experts have default balance (never received feedback).

    Args:
        ledger: Credit ledger to check
        threshold_balance: Default starting balance (100.0)

    Returns:
        True if all experts have default balance (no learning yet)
    """
    if not ledger.accounts:
        return True

    # Check if all accounts have exactly the default balance
    for account in ledger.accounts.values():
        if abs(account.balance - threshold_balance) > 0.01:
            return False  # At least one expert has learned

    return True


def print_cold_start_info(ledger: CreditLedger) -> None:
    """Print information about cold-start mode with recommendations."""
    print()
    print(color("┏" + "━" * 68 + "┓", Colors.CYAN))
    print(color("┃  ❄️  COLD START MODE - Experts Have Not Learned Yet                 ┃", Colors.CYAN))
    print(color("┃                                                                      ┃", Colors.CYAN))
    print(color("┃  All experts have equal weight (no feedback received yet).          ┃", Colors.CYAN))
    print(color("┃  Predictions will improve after making commits.                     ┃", Colors.CYAN))
    print(color("┃                                                                      ┃", Colors.CYAN))
    print(color("┃  💡 Try: python scripts/ml_file_prediction.py predict \"your task\"   ┃", Colors.CYAN))
    print(color("┃     The ML model is trained on commit history and may work better.  ┃", Colors.CYAN))
    print(color("┗" + "━" * 68 + "┛", Colors.CYAN))


def try_ml_fallback(query: str, top_n: int = 10) -> bool:
    """
    Try to get predictions from ML file prediction model as fallback.

    Returns True if fallback was used successfully.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ml_file_prediction import load_model, predict_files

        model_path = Path(__file__).parent.parent / '.git-ml' / 'models' / 'file_prediction.json'
        if not model_path.exists():
            return False

        model = load_model(model_path)
        if model is None:
            return False

        predictions = predict_files(query, model, top_n=top_n)

        if predictions:
            print(color("\n📁 ML File Prediction Fallback:", Colors.BOLD))
            print(color("   (Trained on commit history - may be more accurate)", Colors.DIM))
            print()
            for i, (filepath, score) in enumerate(predictions[:top_n], 1):
                print(f"  {i:2}. {color(filepath, Colors.GREEN):55} ({score:.3f})")
            return True

    except Exception:
        pass

    return False


def cmd_predict(args) -> int:
    """Get predictions for a task."""
    # Load query from file or use argument
    if args.file:
        try:
            with open(args.file) as f:
                query = f.read().strip()
        except Exception as e:
            print(color(f"Error reading {args.file}: {e}", Colors.RED))
            return 1
    else:
        query = args.query

    # Load experts
    if not MODEL_DIR.exists():
        print(color("Error: No trained models found. Run 'train' first.", Colors.RED))
        return 1

    consolidator = ExpertConsolidator(model_dir=MODEL_DIR)

    if not consolidator.experts:
        print(color("Error: No experts loaded.", Colors.RED))
        return 1

    # Load credit ledger for weights
    ledger = load_ledger()
    router = CreditRouter(ledger)

    # Print EXPERIMENTAL banner before any predictions
    print()
    print_experimental_banner()

    # Check for cold-start mode
    cold_start = is_cold_start(ledger)
    if cold_start:
        print_cold_start_info(ledger)

    # Get prediction from each expert
    print(color(f"\n🎯 MoE Prediction for: \"{query}\"", Colors.BOLD))
    print()

    context = {'query': query}
    predictions = {}

    for expert_type, expert in consolidator.experts.items():
        try:
            pred = expert.predict(context)
            predictions[expert_type] = pred
        except Exception as e:
            print(color(f"Warning: {expert_type} prediction failed: {e}", Colors.YELLOW))

    if not predictions:
        print(color("No predictions available", Colors.RED))
        return 1

    # Aggregate predictions using credit routing
    aggregated = router.aggregate_predictions(predictions)

    # Display top files
    print(color("Top Files (by confidence):", Colors.BOLD))
    for i, (filepath, score) in enumerate(aggregated.items[:args.top], 1):
        # Find which experts contributed
        contributors = []
        for expert_type, pred in predictions.items():
            for item, conf in pred.items:
                if item == filepath:
                    contributors.append(f"{expert_type.title()}Expert: {conf:.2f}")

        contrib_str = ", ".join(contributors) if contributors else "ensemble"
        print(f"  {i:2}. {color(filepath, Colors.CYAN):60} {score:.3f}  [{contrib_str}]")

    # Show expert contributions
    print(color("\nExpert Contributions:", Colors.BOLD))
    weights = aggregated.metadata.get('weights', {})
    boosts = aggregated.metadata.get('boosts', {})

    for expert_id in aggregated.contributing_experts:
        account = ledger.get_or_create_account(expert_id)
        weight = weights.get(expert_id, 0.0)
        boost = boosts.get(expert_id, 1.0)

        print(f"  • {expert_id:15}  weight: {weight:.2f}  balance: {account.balance:6.1f}  boost: {boost:.2f}x")

    # Show confidence and disagreement
    print(f"\n{color('Overall Confidence:', Colors.BOLD)} {aggregated.confidence:.3f}")
    print(f"{color('Expert Disagreement:', Colors.BOLD)} {aggregated.disagreement_score:.3f}")

    # Offer ML fallback in cold-start mode or when confidence is low
    low_confidence = aggregated.confidence < 0.5
    if cold_start or low_confidence:
        reason = "cold-start mode" if cold_start else "low confidence"
        print(color(f"\n💡 Due to {reason}, showing ML file prediction fallback:", Colors.CYAN))
        if not try_ml_fallback(query, top_n=args.top):
            print(color("   (ML model not trained - run: python scripts/ml_file_prediction.py train)", Colors.DIM))

    return 0

def cmd_stats(args) -> int:
    """Show expert statistics."""
    if not MODEL_DIR.exists():
        print(color("Error: No trained models found. Run 'train' first.", Colors.RED))
        return 1

    consolidator = ExpertConsolidator(model_dir=MODEL_DIR)

    if not consolidator.experts:
        print(color("Error: No experts loaded.", Colors.RED))
        return 1

    # Load credit ledger
    ledger = load_ledger()

    # Filter to specific expert if requested
    if args.expert:
        experts_to_show = {args.expert: consolidator.experts.get(args.expert)}
        if not experts_to_show[args.expert]:
            print(color(f"Error: Expert '{args.expert}' not found", Colors.RED))
            print(f"Available: {', '.join(consolidator.experts.keys())}")
            return 1
    else:
        experts_to_show = consolidator.experts

    print(color("=" * 60, Colors.BOLD))
    print(color("HUBRIS MoE EXPERT STATISTICS", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))

    for expert_type, expert in experts_to_show.items():
        print(f"\n{color(expert_type.upper() + ' EXPERT', Colors.BOLD)}")
        print("-" * 60)

        # Basic info
        print(f"  Expert ID:     {expert.expert_id}")
        print(f"  Version:       {expert.version}")
        print(f"  Created:       {expert.created_at}")

        # Training data
        print(f"\n  Training Data:")
        print(f"    Commits:     {expert.trained_on_commits}")
        print(f"    Sessions:    {expert.trained_on_sessions}")

        # Credit account
        account = ledger.get_or_create_account(expert_type)
        print(f"\n  Credit Account:")
        print(f"    Balance:     {account.balance:.2f}")
        print(f"    Transactions: {len(account.transactions)}")

        # Recent transactions
        if account.transactions:
            recent = account.get_recent_transactions(5)
            print(f"\n  Recent Transactions:")
            for tx in recent:
                amount_str = color(f"+{tx.amount:.1f}", Colors.GREEN) if tx.amount > 0 else color(f"{tx.amount:.1f}", Colors.RED)
                print(f"    {amount_str:20} {tx.reason:30} (balance: {tx.balance_after:.1f})")

        # Metrics if available
        if expert.metrics:
            print(f"\n  Performance Metrics:")
            print(f"    MRR:          {expert.metrics.mrr:.4f}")
            if expert.metrics.recall_at_k:
                for k, recall in sorted(expert.metrics.recall_at_k.items()):
                    print(f"    Recall@{k:2}:    {recall:.4f}")
            if expert.metrics.precision_at_k:
                for k, precision in sorted(expert.metrics.precision_at_k.items()):
                    print(f"    Precision@{k:2}: {precision:.4f}")

    return 0

def cmd_leaderboard(args) -> int:
    """Show expert credit leaderboard."""
    ledger = load_ledger()

    if not ledger.accounts:
        print(color("No expert accounts found.", Colors.YELLOW))
        return 0

    print(color("=" * 60, Colors.BOLD))
    print(color("HUBRIS MoE EXPERT LEADERBOARD", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))
    print()

    top_experts = ledger.get_top_experts(n=args.top)

    print(f"{'Rank':<6} {'Expert':<20} {'Balance':>12} {'Transactions':>15}")
    print("-" * 60)

    for i, (expert_id, balance) in enumerate(top_experts, 1):
        account = ledger.accounts[expert_id]

        # Color code by performance
        if balance > 150:
            balance_str = color(f"{balance:>12.2f}", Colors.GREEN)
        elif balance < 80:
            balance_str = color(f"{balance:>12.2f}", Colors.RED)
        else:
            balance_str = f"{balance:>12.2f}"

        print(f"{i:<6} {expert_id:<20} {balance_str} {len(account.transactions):>15}")

    # Summary stats
    total_credits = ledger.get_total_credits()
    print()
    print("-" * 60)
    print(f"Total Credits in System: {total_credits:.2f}")
    print(f"Total Accounts: {len(ledger.accounts)}")

    return 0

def cmd_evaluate(args) -> int:
    """Evaluate expert accuracy on recent commits."""
    print(color("=" * 60, Colors.BOLD))
    print(color("EVALUATING HUBRIS MoE EXPERTS", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))

    # Load experts
    if not MODEL_DIR.exists():
        print(color("\nError: No trained models found. Run 'train' first.", Colors.RED))
        return 1

    consolidator = ExpertConsolidator(model_dir=MODEL_DIR)

    if not consolidator.experts:
        print(color("Error: No experts loaded.", Colors.RED))
        return 1

    # Load evaluation commits
    print(f"\nLoading last {args.commits} commits for evaluation...")
    commits = load_commit_data(limit=args.commits)

    if not commits:
        print(color("Error: No commits found for evaluation", Colors.RED))
        return 1

    print(f"Evaluating on {len(commits)} commits...")

    # Evaluate each expert
    results = {}
    ledger = load_ledger()

    for expert_type, expert in consolidator.experts.items():
        print(f"\nEvaluating {expert_type}...")

        correct_top1 = 0
        correct_top5 = 0
        correct_top10 = 0
        total = 0
        reciprocal_ranks = []

        for commit in commits:
            # Skip if no files or message
            if not commit.get('files') or not commit.get('message'):
                continue

            actual_files = set(commit['files'])
            context = {'query': commit['message']}

            try:
                pred = expert.predict(context)
                predicted_files = [item for item, _ in pred.items]

                # Calculate metrics
                total += 1

                # Check top-k accuracy
                if predicted_files and predicted_files[0] in actual_files:
                    correct_top1 += 1

                top5 = set(predicted_files[:5])
                if top5 & actual_files:
                    correct_top5 += 1

                top10 = set(predicted_files[:10])
                if top10 & actual_files:
                    correct_top10 += 1

                # Calculate reciprocal rank
                for rank, pred_file in enumerate(predicted_files, 1):
                    if pred_file in actual_files:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    reciprocal_ranks.append(0.0)

            except Exception as e:
                print(color(f"  Warning: Prediction failed: {e}", Colors.YELLOW))
                continue

        if total == 0:
            print(color("  No valid predictions", Colors.YELLOW))
            continue

        # Calculate metrics
        precision_at_1 = correct_top1 / total
        recall_at_5 = correct_top5 / total
        recall_at_10 = correct_top10 / total
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        results[expert_type] = {
            'precision_at_1': precision_at_1,
            'recall_at_5': recall_at_5,
            'recall_at_10': recall_at_10,
            'mrr': mrr,
            'total': total
        }

        # Update credit based on performance
        # Award credits for good performance, deduct for poor performance
        account = ledger.get_or_create_account(expert_type)

        # Simple reward: 10 credits per correct top-1, -5 for incorrect
        credits_earned = (correct_top1 * 10) - ((total - correct_top1) * 5)

        if credits_earned > 0:
            account.credit(
                credits_earned,
                "evaluation_performance",
                {'commits': total, 'correct': correct_top1}
            )
        else:
            account.debit(
                abs(credits_earned),
                "evaluation_penalty",
                {'commits': total, 'correct': correct_top1}
            )

        print(f"  Evaluated on {total} commits")
        print(f"  MRR:          {mrr:.4f}")
        print(f"  Precision@1:  {precision_at_1:.4f}")
        print(f"  Recall@5:     {recall_at_5:.4f}")
        print(f"  Recall@10:    {recall_at_10:.4f}")
        print(f"  Credits:      {'+' if credits_earned > 0 else ''}{credits_earned:.1f}")

    # Save updated ledger
    save_ledger(ledger)

    # Summary
    print(color("\n" + "=" * 60, Colors.BOLD))
    print(color("EVALUATION SUMMARY", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))

    for expert_type, metrics in sorted(results.items(), key=lambda x: -x[1]['mrr']):
        print(f"\n{expert_type}:")
        print(f"  MRR:          {metrics['mrr']:.4f}")
        print(f"  Precision@1:  {metrics['precision_at_1']:.4f}")
        print(f"  Recall@5:     {metrics['recall_at_5']:.4f}")
        print(f"  Recall@10:    {metrics['recall_at_10']:.4f}")

    print(color("\n✓ Evaluation complete! Credit balances updated.", Colors.GREEN))

    return 0


def cmd_calibration(args) -> int:
    """Show calibration analysis for expert predictions."""
    # Route to test calibration if --tests flag is set
    if args.tests:
        return cmd_calibration_tests(args)

    # Default: file prediction calibration
    print(color("=" * 60, Colors.BOLD))
    print(color("HUBRIS MoE CALIBRATION ANALYSIS", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))

    tracker = CalibrationTracker()
    loaded = tracker.load_from_resolved()

    if loaded == 0:
        print(color("\nNo calibration data available yet.", Colors.YELLOW))
        print("Calibration data is collected after predictions resolve via commits.")
        print("\nTo generate calibration data:")
        print("  1. Make predictions: hubris predict \"your task\"")
        print("  2. Make commits: git commit -m \"...\"")
        print("  3. Re-run this command")
        return 0

    print(f"\nLoaded {loaded} resolved predictions.")

    # Output format
    if args.json:
        import json
        print(json.dumps(tracker.get_summary(), indent=2))
        return 0

    if args.curve:
        # Show calibration curve
        curve = tracker.get_calibration_curve(args.expert)
        print(color("\nCalibration Curve:", Colors.BOLD))
        print(color("(bin_center, actual_accuracy, sample_count)", Colors.DIM))
        print()
        for conf, acc, count in curve:
            bar_len = int(acc * 30)
            bar = color("█" * bar_len, Colors.GREEN) + color("░" * (30 - bar_len), Colors.DIM)
            deviation = acc - conf
            dev_str = f"+{deviation:.2f}" if deviation > 0 else f"{deviation:.2f}"
            print(f"  {conf:.1f} │{bar}│ {acc:.3f} [{count:3d}] ({dev_str})")
        return 0

    # Default: show report
    metrics = tracker.get_metrics(args.expert)

    if not metrics:
        print(color("\nInsufficient data for metrics.", Colors.YELLOW))
        return 0

    # Header
    target = f"Expert: {args.expert}" if args.expert else "All Experts"
    print(f"\n{color(target, Colors.BOLD)}")
    print("-" * 60)

    # Metrics
    print(f"\n{color('Calibration Metrics:', Colors.BOLD)}")

    # ECE with interpretation
    ece_status = _get_ece_status(metrics.ece)
    print(f"  ECE (Expected Calibration Error): {metrics.ece:.3f} {ece_status}")
    print(f"  MCE (Max Calibration Error):      {metrics.mce:.3f}")
    print(f"  Brier Score:                      {metrics.brier_score:.3f}")

    print(f"\n{color('Confidence vs Accuracy:', Colors.BOLD)}")
    print(f"  Average Confidence: {metrics.confidence_mean:.3f}")
    print(f"  Average Accuracy:   {metrics.accuracy_mean:.3f}")

    # Trend with emoji
    trend_emoji = {"overconfident": "📈", "underconfident": "📉", "well_calibrated": "✓"}
    trend_color = {"overconfident": Colors.YELLOW, "underconfident": Colors.CYAN, "well_calibrated": Colors.GREEN}
    emoji = trend_emoji.get(metrics.trend, "")
    t_color = trend_color.get(metrics.trend, "")
    print(f"  Trend: {color(f'{emoji} {metrics.trend}', t_color)}")

    print(f"\n{color('Sample Size:', Colors.BOLD)} {metrics.sample_count} predictions")

    # Recommendations
    print(color("\nRecommendations:", Colors.BOLD))
    if metrics.trend == 'overconfident' and metrics.ece > 0.10:
        print(color("  ⚠️  System is overconfident - predictions claim higher accuracy than achieved", Colors.YELLOW))
        print("     Consider: reducing base confidence or retraining with more data")
    elif metrics.trend == 'underconfident' and metrics.ece > 0.10:
        print(color("  ℹ️  System is underconfident - predictions are more accurate than claimed", Colors.CYAN))
        print("     Consider: boosting confidence scores")
    elif metrics.ece < 0.10:
        print(color("  ✓ Calibration is good (ECE < 0.10)", Colors.GREEN))
    else:
        print(color("  ⚠️  Calibration needs improvement (ECE >= 0.10)", Colors.YELLOW))

    return 0


def cmd_calibration_tests(args) -> int:
    """Show test selection calibration analysis."""
    print(color("=" * 70, Colors.BOLD))
    print(color("TEST SELECTION CALIBRATION ANALYSIS", Colors.BOLD))
    print(color("=" * 70, Colors.BOLD))

    tracker = TestCalibrationTracker()
    loaded = tracker.load_all()

    if loaded == 0:
        print(color("\nNo test calibration data available yet.", Colors.YELLOW))
        print("Test calibration tracks how well TestExpert predicts which tests to run.")
        print("\nTo generate test calibration data:")
        print("  1. Record test predictions via TestCalibrationTracker.record_prediction()")
        print("  2. Run tests and record outcomes via TestCalibrationTracker.record_outcome()")
        print("  3. Re-run this command")
        return 0

    print(f"\nLoaded {loaded} calibration records.")

    # Output format
    if args.json:
        import json
        print(json.dumps(tracker.get_summary(), indent=2))
        return 0

    # Get metrics
    metrics = tracker.get_metrics()

    if not metrics:
        print(color("\nInsufficient data for metrics.", Colors.YELLOW))
        return 0

    # Show report
    summary = tracker.get_summary()

    print(f"\nPredictions recorded:  {summary['predictions_recorded']}")
    print(f"Outcomes recorded:     {summary['outcomes_recorded']}")
    print(f"Calibration records:   {summary['calibration_records']}")
    print()

    # Metrics with color coding
    print(color("TEST SELECTION METRICS:", Colors.BOLD))
    print()

    # Precision@5
    p5 = metrics.precision_at_5_mean
    p5_color = Colors.GREEN if p5 >= 0.7 else Colors.YELLOW if p5 >= 0.5 else Colors.RED
    print(f"  Precision@5:      {color(f'{p5:.3f}', p5_color)}  (of top 5 suggestions, how many relevant?)")

    # Recall
    recall = metrics.recall_mean
    recall_color = Colors.GREEN if recall >= 0.8 else Colors.YELLOW if recall >= 0.6 else Colors.RED
    print(f"  Recall:           {color(f'{recall:.3f}', recall_color)}  (of failures, what % did we catch?)")

    # Hit rate
    hit_rate = metrics.hit_rate
    hit_color = Colors.GREEN if hit_rate >= 0.85 else Colors.YELLOW if hit_rate >= 0.7 else Colors.RED
    print(f"  Hit Rate:         {color(f'{hit_rate:.3f}', hit_color)}  (% predictions catching at least one failure)")

    # MRR
    mrr = metrics.mrr
    mrr_color = Colors.GREEN if mrr >= 0.5 else Colors.YELLOW if mrr >= 0.3 else Colors.RED
    print(f"  MRR:              {color(f'{mrr:.3f}', mrr_color)}  (rank of first failure in suggestions)")

    # False alarm rate
    far = metrics.false_alarm_rate
    far_color = Colors.GREEN if far <= 0.3 else Colors.YELLOW if far <= 0.5 else Colors.RED
    print(f"  False Alarm Rate: {color(f'{far:.3f}', far_color)}  (suggested tests that didn't fail)")

    # Coverage
    coverage = metrics.coverage
    cov_color = Colors.GREEN if coverage >= 0.9 else Colors.YELLOW if coverage >= 0.7 else Colors.RED
    print(f"  Coverage:         {color(f'{coverage:.3f}', cov_color)}  (% of all failures caught)")

    print()

    # Status
    status = metrics.get_status()
    status_emoji = {
        'excellent': '🌟',
        'good': '✓',
        'acceptable': '⚠️',
        'needs_attention': '⚠️',
        'poor': '❌'
    }
    status_colors = {
        'excellent': Colors.GREEN,
        'good': Colors.GREEN,
        'acceptable': Colors.YELLOW,
        'needs_attention': Colors.YELLOW,
        'poor': Colors.RED
    }
    emoji = status_emoji.get(status, '')
    s_color = status_colors.get(status, '')
    print(f"{color('Status:', Colors.BOLD)} {color(f'{emoji} {status.upper()}', s_color)}")
    print()

    # Recommendations
    if summary['recommendations']:
        print(color("RECOMMENDATIONS:", Colors.BOLD))
        for rec in summary['recommendations']:
            print(f"  {rec}")
        print()

    # Interpretation guide
    print(color("METRIC INTERPRETATION:", Colors.BOLD))
    print("  Good thresholds:")
    print("    • Precision@5 ≥ 0.7  (most suggestions are useful)")
    print("    • Recall ≥ 0.8       (catch most failures)")
    print("    • Hit Rate ≥ 0.85    (rarely miss failures entirely)")
    print("    • MRR ≥ 0.5          (failures ranked in top 2 on average)")
    print("    • Coverage ≥ 0.9     (catch 90%+ of all failures)")

    return 0


def _get_ece_status(ece: float) -> str:
    """Get colored status string for ECE value."""
    if ece < 0.05:
        return color("[excellent]", Colors.GREEN)
    elif ece < 0.10:
        return color("[good]", Colors.GREEN)
    elif ece < 0.15:
        return color("[acceptable]", Colors.YELLOW)
    elif ece < 0.20:
        return color("[needs attention]", Colors.YELLOW)
    else:
        return color("[poor]", Colors.RED)


def cmd_suggest_refactor(args) -> int:
    """Suggest files that may benefit from refactoring."""
    from experts.refactor_expert import RefactorExpert

    print(color("=" * 60, Colors.BOLD))
    print(color("REFACTORING SUGGESTIONS", Colors.BOLD))
    print(color("=" * 60, Colors.BOLD))

    # Load or create RefactorExpert
    refactor_model_path = MODEL_DIR / 'refactor_expert.json'

    if refactor_model_path.exists():
        print("\nLoading trained RefactorExpert...")
        expert = RefactorExpert.load(refactor_model_path)
        print(f"  Trained on {expert.trained_on_commits} refactoring commits")
    else:
        print(color("\nNote: No trained model found. Using heuristics only.", Colors.YELLOW))
        print("  Run 'hubris train' first to learn from commit history.")
        expert = RefactorExpert()

    # Determine files to analyze
    if args.files:
        files_to_analyze = args.files
        print(f"\nAnalyzing {len(files_to_analyze)} specified files...")
    elif args.scan:
        print(f"\nScanning codebase for refactoring candidates...")
        # Use analyze_codebase for full scan
        prediction = expert.analyze_codebase(
            repo_root=args.repo or '.',
            top_n=args.top
        )
        files_to_analyze = None  # Already analyzed
    else:
        # Default: analyze recently changed files
        print("\nAnalyzing recently changed files...")
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~10'],
                capture_output=True, text=True, cwd=args.repo or '.'
            )
            files_to_analyze = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            if not files_to_analyze:
                print(color("  No recent changes found. Use --scan to analyze entire codebase.", Colors.YELLOW))
                return 0
            print(f"  Found {len(files_to_analyze)} recently changed files")
        except Exception as e:
            print(color(f"  Could not get git history: {e}", Colors.YELLOW))
            print("  Use --files or --scan to specify files.")
            return 1

    # Get predictions
    if files_to_analyze is not None:
        prediction = expert.predict({
            'files': files_to_analyze,
            'top_n': args.top,
            'include_heuristics': True,
            'repo_root': args.repo or '.'
        })

    if not prediction.items:
        print(color("\n✓ No significant refactoring candidates found!", Colors.GREEN))
        return 0

    # Display results
    print(color(f"\nTop {len(prediction.items)} Refactoring Candidates:", Colors.BOLD))
    print()

    file_signals = prediction.metadata.get('file_signals', {})
    signal_icons = {
        'extract': '📦',   # Split/extract
        'inline': '🔗',    # Merge/inline
        'rename': '🏷️',    # Rename
        'move': '📁',      # Move
        'dedupe': '🔄',    # Deduplicate
        'simplify': '✨',  # Simplify
        'co_refactor': '🔀',  # Co-refactoring pattern
    }

    for i, (filepath, score) in enumerate(prediction.items, 1):
        signals = file_signals.get(filepath, [])
        signal_str = ' '.join(signal_icons.get(s, '•') for s in signals) if signals else ''

        # Color code by score
        if score > 0.7:
            score_str = color(f"{score:.2f}", Colors.RED)
            priority = color("[HIGH]", Colors.RED)
        elif score > 0.4:
            score_str = color(f"{score:.2f}", Colors.YELLOW)
            priority = color("[MED]", Colors.YELLOW)
        else:
            score_str = f"{score:.2f}"
            priority = color("[LOW]", Colors.DIM)

        print(f"  {i:2}. {color(filepath, Colors.CYAN):55} {score_str} {priority} {signal_str}")

        # Show detailed report if verbose
        if args.verbose and filepath:
            report = expert.get_file_report(filepath, args.repo or '.')
            if report.get('recommendations'):
                for rec in report['recommendations'][:2]:  # Limit to 2 recommendations
                    print(color(f"      → {rec}", Colors.DIM))

    # Signal summary
    signal_counts = prediction.metadata.get('signal_counts', {})
    if signal_counts:
        print(color("\nSignal Summary:", Colors.BOLD))
        for signal, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
            icon = signal_icons.get(signal, '•')
            print(f"  {icon} {signal}: {count} files")

    # Show scoring sources
    sources = prediction.metadata.get('scoring_sources', [])
    if sources:
        print(color(f"\nScoring based on: {', '.join(sources)}", Colors.DIM))

    print()
    return 0


def cmd_suggest_tests(args) -> int:
    """Suggest tests to run for code changes."""
    import subprocess

    # Get changed files from args or git
    changed_files = []

    if args.files:
        changed_files = args.files
    elif args.staged:
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )
            changed_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except subprocess.CalledProcessError as e:
            print(color(f"Error: Failed to get staged files: {e}", Colors.RED))
            return 1
    elif args.modified:
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )
            changed_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except subprocess.CalledProcessError as e:
            print(color(f"Error: Failed to get modified files: {e}", Colors.RED))
            return 1
    else:
        print(color("Error: Must specify --files, --staged, or --modified", Colors.RED))
        return 1

    if not changed_files:
        print(color("No files to analyze.", Colors.YELLOW))
        return 0

    # Load TestExpert
    test_expert_path = MODEL_DIR / 'test_expert.json'
    if not test_expert_path.exists():
        print(color("Error: TestExpert model not found. Run 'train' first.", Colors.RED))
        print(f"Expected: {test_expert_path}")
        return 1

    try:
        sys.path.insert(0, str(HUBRIS_DIR / 'experts'))
        from test_expert import TestExpert
        expert = TestExpert.load(test_expert_path)
    except Exception as e:
        print(color(f"Error: Failed to load TestExpert: {e}", Colors.RED))
        return 1

    # Print EXPERIMENTAL banner
    print()
    print_experimental_banner()

    # Display changed files
    print(color(f"\n🧪 Suggested Tests for Changed Files:", Colors.BOLD))
    print()

    # Filter out test files from changed files
    source_files = [f for f in changed_files if not f.startswith('tests/')]
    test_files = [f for f in changed_files if f.startswith('tests/')]

    print(color(f"Changed files ({len(changed_files)}):", Colors.BOLD))
    for f in source_files:
        print(f"  {color('•', Colors.CYAN)} {f}")
    if test_files:
        print(color(f"\nTest files already modified ({len(test_files)}):", Colors.DIM))
        for f in test_files:
            print(f"  {color('•', Colors.DIM)} {f}")

    # Get predictions
    context = {
        'changed_files': changed_files,
        'top_n': args.top
    }

    try:
        prediction = expert.predict(context)
    except Exception as e:
        print(color(f"\nError: Prediction failed: {e}", Colors.RED))
        return 1

    if not prediction.items:
        print(color("\nNo test suggestions available.", Colors.YELLOW))
        print("This may mean:")
        print("  • These files are new and haven't been tested yet")
        print("  • The expert needs more training data")
        return 0

    # Display suggestions
    print(color(f"\nSuggested tests (by confidence):", Colors.BOLD))
    for i, (test_file, confidence) in enumerate(prediction.items, 1):
        # Get scoring signals
        signals = prediction.metadata.get('scoring_signals', [])
        signals_str = ', '.join(signals) if signals else 'ensemble'

        # Color code by confidence
        if confidence > 0.8:
            conf_str = color(f"{confidence:.3f}", Colors.GREEN)
        elif confidence > 0.5:
            conf_str = color(f"{confidence:.3f}", Colors.CYAN)
        else:
            conf_str = color(f"{confidence:.3f}", Colors.YELLOW)

        print(f"  {i:2}. {test_file:55} {conf_str}  [{color(signals_str, Colors.DIM)}]")

    # Show coverage estimate
    coverage = expert.get_coverage_estimate(source_files if source_files else changed_files)
    coverage_pct = coverage * 100

    print()
    if coverage > 0.8:
        cov_str = color(f"{coverage_pct:.0f}%", Colors.GREEN)
        status = color("✓ Good coverage", Colors.GREEN)
    elif coverage > 0.5:
        cov_str = color(f"{coverage_pct:.0f}%", Colors.YELLOW)
        status = color("⚠ Partial coverage", Colors.YELLOW)
    else:
        cov_str = color(f"{coverage_pct:.0f}%", Colors.RED)
        status = color("⚠ Low coverage", Colors.RED)

    print(f"{color('Coverage estimate:', Colors.BOLD)} {cov_str} of changed files have test mappings")
    print(f"Status: {status}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Hubris MoE CLI - Mixture of Experts command-line interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --commits 100
  %(prog)s predict "Add authentication feature"
  %(prog)s stats --expert file
  %(prog)s leaderboard
  %(prog)s evaluate --commits 20
  %(prog)s calibration --curve
  %(prog)s suggest-tests --staged
  %(prog)s suggest-tests --files cortical/query/search.py cortical/analysis.py
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train all experts from collected data')
    train_parser.add_argument('--commits', type=int, default=None,
                             help='Number of commits to use (default: all)')
    train_parser.add_argument('--transcripts', action='store_true',
                             help='Include transcript data for EpisodeExpert')
    train_parser.add_argument('--errors', action='store_true',
                             help='Include error data for ErrorDiagnosisExpert')
    train_parser.add_argument('--include-ci', action='store_true',
                             help='Transform CI results into test_results for TestExpert training')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Get predictions for a task')
    predict_parser.add_argument('query', nargs='?', type=str,
                               help='Task description')
    predict_parser.add_argument('--file', '-f', type=str,
                               help='Read query from file')
    predict_parser.add_argument('--top', '-n', type=int, default=10,
                               help='Number of predictions to show (default: 10)')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show expert statistics')
    stats_parser.add_argument('--expert', '-e', type=str,
                             help='Show stats for specific expert (file, test, error, episode)')

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser('leaderboard', help='Show expert credit leaderboard')
    leaderboard_parser.add_argument('--top', '-n', type=int, default=10,
                                   help='Number of experts to show (default: 10)')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate expert accuracy on recent commits')
    evaluate_parser.add_argument('--commits', '-n', type=int, default=20,
                                help='Number of recent commits to evaluate on (default: 20)')

    # Calibration command
    calibration_parser = subparsers.add_parser('calibration', help='Show calibration analysis for predictions')
    calibration_parser.add_argument('--expert', '-e', type=str,
                                   help='Show calibration for specific expert')
    calibration_parser.add_argument('--json', action='store_true',
                                   help='Output as JSON')
    calibration_parser.add_argument('--curve', action='store_true',
                                   help='Show calibration curve visualization')
    calibration_parser.add_argument('--tests', action='store_true',
                                   help='Show test selection calibration instead of file prediction calibration')

    # Suggest-tests command
    suggest_tests_parser = subparsers.add_parser('suggest-tests', help='Suggest tests to run for code changes')
    suggest_tests_parser.add_argument('--files', '-f', nargs='+',
                                     help='Files being changed')
    suggest_tests_parser.add_argument('--staged', action='store_true',
                                     help='Use git staged files')
    suggest_tests_parser.add_argument('--modified', action='store_true',
                                     help='Use git modified files')
    suggest_tests_parser.add_argument('--top', '-n', type=int, default=10,
                                     help='Number of suggestions to show (default: 10)')

    # Suggest-refactor command
    refactor_parser = subparsers.add_parser('suggest-refactor', help='Suggest files that may need refactoring')
    refactor_parser.add_argument('--files', '-f', nargs='+', type=str,
                                help='Specific files to analyze')
    refactor_parser.add_argument('--scan', '-s', action='store_true',
                                help='Scan entire codebase for candidates')
    refactor_parser.add_argument('--top', '-n', type=int, default=10,
                                help='Number of suggestions to show (default: 10)')
    refactor_parser.add_argument('--repo', '-r', type=str, default='.',
                                help='Repository root directory (default: current)')
    refactor_parser.add_argument('--verbose', '-v', action='store_true',
                                help='Show detailed recommendations')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handlers
    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'predict':
        if not args.query and not args.file:
            print(color("Error: Must provide query or --file", Colors.RED))
            predict_parser.print_help()
            return 1
        return cmd_predict(args)
    elif args.command == 'stats':
        return cmd_stats(args)
    elif args.command == 'leaderboard':
        return cmd_leaderboard(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    elif args.command == 'calibration':
        return cmd_calibration(args)
    elif args.command == 'suggest-tests':
        return cmd_suggest_tests(args)
    elif args.command == 'suggest-refactor':
        return cmd_suggest_refactor(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
