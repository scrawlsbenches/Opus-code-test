#!/usr/bin/env python3
"""
Hubris MoE CLI - Command-line interface for the Mixture of Experts system

Commands:
    train       - Train all experts from collected data
    predict     - Get predictions for a task
    stats       - Show expert statistics
    leaderboard - Show expert credit leaderboard
    evaluate    - Evaluate expert accuracy on recent commits

Examples:
    python scripts/hubris_cli.py train --commits 100
    python scripts/hubris_cli.py predict "Add authentication feature"
    python scripts/hubris_cli.py stats --expert file_expert
    python scripts/hubris_cli.py leaderboard
    python scripts/hubris_cli.py evaluate --commits 20
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

def load_commit_data(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load commit data from .git-ml/commits/.

    Args:
        limit: Maximum number of commits to load (most recent first)

    Returns:
        List of commit dictionaries
    """
    if not COMMITS_DIR.exists():
        print(color(f"Warning: {COMMITS_DIR} not found", Colors.YELLOW))
        return []

    commits = []
    commit_files = sorted(COMMITS_DIR.glob('*.json'), reverse=True)

    if limit:
        commit_files = commit_files[:limit]

    for commit_file in commit_files:
        try:
            with open(commit_file) as f:
                commit = json.load(f)
                commits.append(commit)
        except Exception as e:
            print(color(f"Warning: Failed to load {commit_file}: {e}", Colors.YELLOW))

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
    commits = load_commit_data(limit=args.commits)
    transcripts = load_transcript_data() if args.transcripts else None

    print(f"  Commits: {len(commits)}")
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
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
