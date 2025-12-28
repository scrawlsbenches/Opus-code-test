#!/usr/bin/env python3
"""
Train SparkSLM models from git commit history.

Usage:
    python scripts/train_spark_from_git.py train [options]
    python scripts/train_spark_from_git.py stats
    python scripts/train_spark_from_git.py evaluate <model_path>
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.spark import NGramModel, GitHistoryTrainer, WeightedCommit


def cmd_train(args):
    """Train a SparkSLM model from git history."""
    print(f"Training SparkSLM from git history...")
    print(f"  Repository: {args.repo}")
    print(f"  Branches: {args.branches or 'all'}")
    print(f"  Half-life: {args.half_life} days")
    print(f"  Output: {args.output}")

    if args.dry_run:
        print("\n[DRY RUN] Would train model with above settings")
        return 0

    # Create trainer
    trainer = GitHistoryTrainer(
        repo_path=args.repo,
        temporal_half_life_days=args.half_life,
        min_weight=args.min_weight
    )

    # Get commits (stub - would need actual git integration)
    commits = list(trainer.iter_commits(
        branches=args.branches.split(',') if args.branches else None,
        max_commits=args.max_commits
    ))

    if not commits:
        print("\nNo commits found. Git integration is stubbed.")
        print("Future implementation will parse actual git log.")

        # Create demo model for testing
        if args.demo:
            print("\n[DEMO MODE] Creating sample model...")
            model = NGramModel(n=args.ngram_size)
            model.train(["demo training data for spark model"])

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
            print(f"Demo model saved to: {output_path}")
        return 0

    # Prepare training data
    documents, weights = trainer.prepare_training_data(commits)

    print(f"\nTraining on {len(documents)} commits...")
    print(f"  Total weight: {sum(weights):.2f}")
    print(f"  Weight range: {min(weights):.3f} - {max(weights):.3f}")

    # Train model
    model = NGramModel(n=args.ngram_size)
    model.train_weighted(documents, weights)
    model.finalize()

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    print(f"\nModel trained successfully!")
    print(f"  Vocabulary size: {len(model.vocab)}")
    print(f"  Output: {output_path}")

    return 0


def cmd_stats(args):
    """Show statistics about available training data."""
    print("Git Training Statistics")
    print("=" * 40)

    # Count commits per branch (would use actual git)
    print("\nBranch weights (configured):")
    for branch, weight in GitHistoryTrainer.BRANCH_WEIGHTS.items():
        print(f"  {branch}: {weight}")

    print("\nQuality multipliers:")
    for signal, mult in GitHistoryTrainer.QUALITY_MULTIPLIERS.items():
        print(f"  {signal}: {mult}x")

    # Check for existing model
    model_path = Path(args.model_dir) / "spark_model.json"
    if model_path.exists():
        model = NGramModel.load(str(model_path))
        print(f"\nExisting model found:")
        print(f"  Vocabulary: {len(model.vocab)} terms")
        print(f"  N-gram size: {model.n}")
    else:
        print(f"\nNo existing model at: {model_path}")

    return 0


def cmd_evaluate(args):
    """Evaluate a trained model."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    model = NGramModel.load(str(model_path))
    print(f"Loaded model: {model_path}")
    print(f"  Vocabulary: {len(model.vocab)} terms")
    print(f"  N-gram size: {model.n}")

    # Sample predictions
    print("\nSample predictions:")
    test_contexts = [
        ["def"],
        ["import"],
        ["class"],
        ["return"],
    ]

    for context in test_contexts:
        predictions = model.predict(context, top_k=5)
        pred_str = ", ".join(f"{p[0]}({p[1]:.3f})" for p in predictions[:3])
        print(f"  {context} -> {pred_str}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Train SparkSLM models from git commit history"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--repo', default='.', help='Repository path')
    train_parser.add_argument('--branches', help='Comma-separated branch list')
    train_parser.add_argument('--half-life', type=float, default=30.0,
                              help='Temporal decay half-life in days')
    train_parser.add_argument('--min-weight', type=float, default=0.1,
                              help='Minimum weight for commits')
    train_parser.add_argument('--max-commits', type=int, help='Max commits to process')
    train_parser.add_argument('--ngram-size', type=int, default=3,
                              help='N-gram size (2=bigram, 3=trigram)')
    train_parser.add_argument('--output', default='.git-ml/spark_model/model.json',
                              help='Output model path')
    train_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be done without training')
    train_parser.add_argument('--demo', action='store_true',
                              help='Create demo model (for testing)')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show training statistics')
    stats_parser.add_argument('--model-dir', default='.git-ml/spark_model',
                              help='Model directory')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('model_path', help='Path to model file')

    args = parser.parse_args()

    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'stats':
        return cmd_stats(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
