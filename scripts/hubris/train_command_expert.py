#!/usr/bin/env python3
"""
Train and evaluate CommandExpert.

Usage:
    python scripts/hubris/train_command_expert.py train
    python scripts/hubris/train_command_expert.py evaluate
    python scripts/hubris/train_command_expert.py predict "run tests"
    python scripts/hubris/train_command_expert.py stats
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experts.command_expert import CommandExpert


def get_actions_dir() -> Path:
    """Get the .git-ml/actions directory."""
    # Try relative to script location
    script_dir = Path(__file__).parent.parent.parent
    actions_dir = script_dir / '.git-ml' / 'actions'
    if actions_dir.exists():
        return actions_dir

    # Try current directory
    actions_dir = Path('.git-ml/actions')
    if actions_dir.exists():
        return actions_dir

    raise FileNotFoundError("Could not find .git-ml/actions directory")


def get_model_path() -> Path:
    """Get path to save/load model."""
    script_dir = Path(__file__).parent.parent.parent
    models_dir = script_dir / '.git-ml' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / 'command_expert.json'


def train(args):
    """Train the CommandExpert."""
    print("Training CommandExpert...")

    actions_dir = get_actions_dir()
    print(f"Actions directory: {actions_dir}")

    expert = CommandExpert()
    stats = expert.train(actions_dir)

    print(f"\nTraining Statistics:")
    print(f"  Total actions:      {stats.get('total_actions', 0)}")
    print(f"  Bash actions:       {stats.get('bash_actions', 0)}")
    print(f"  Python -c commands: {stats.get('python_c_commands', 0)}")
    print(f"  Unique commands:    {stats.get('unique_commands', 0)}")
    print(f"  Sessions:           {stats.get('sessions', 0)}")

    # Save model
    model_path = get_model_path()
    expert.save(model_path)
    print(f"\nModel saved to: {model_path}")

    return expert


def evaluate(args):
    """Evaluate the CommandExpert."""
    model_path = get_model_path()

    if not model_path.exists():
        print("No trained model found. Run 'train' first.")
        return

    expert = CommandExpert.load(model_path)
    print(f"Loaded model: {expert.expert_id} v{expert.version}")
    print(f"Trained on: {expert.trained_on_commits} commands from {expert.trained_on_sessions} sessions")

    # Build test set from recent actions (last 20%)
    actions_dir = get_actions_dir()
    test_actions = []

    all_actions = []
    for date_dir in sorted(actions_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for action_file in sorted(date_dir.glob('A-*.json')):
            try:
                with open(action_file) as f:
                    action = json.load(f)
                ctx = action.get('context', {})
                if ctx.get('tool') == 'Bash':
                    input_data = ctx.get('input', {})
                    if isinstance(input_data, dict):
                        input_data = input_data.get('input', input_data)
                    command = input_data.get('command', '') if isinstance(input_data, dict) else ''
                    description = input_data.get('description', '') if isinstance(input_data, dict) else ''
                    if command and description:
                        all_actions.append({
                            'query': description,
                            'command': command
                        })
            except (json.JSONDecodeError, KeyError):
                continue

    if not all_actions:
        print("No test actions found with descriptions.")
        return

    # Use last 20% as test set
    split_idx = int(len(all_actions) * 0.8)
    test_actions = all_actions[split_idx:]

    print(f"\nEvaluating on {len(test_actions)} test actions...")

    metrics = expert.evaluate(test_actions)

    print(f"\nEvaluation Metrics:")
    print(f"  MRR:          {metrics.mrr:.3f}")
    print(f"  Recall@1:     {metrics.recall_at_k.get(1, 0):.3f}")
    print(f"  Recall@5:     {metrics.recall_at_k.get(5, 0):.3f}")
    print(f"  Recall@10:    {metrics.recall_at_k.get(10, 0):.3f}")
    print(f"  Precision@1:  {metrics.precision_at_k.get(1, 0):.3f}")
    print(f"  Test examples: {metrics.test_examples}")

    # Save updated model with metrics
    expert.save(model_path)
    print(f"\nModel with metrics saved to: {model_path}")


def predict(args):
    """Make a prediction."""
    model_path = get_model_path()

    if not model_path.exists():
        print("No trained model found. Run 'train' first.")
        return

    expert = CommandExpert.load(model_path)
    query = ' '.join(args.query)

    print(f"Query: {query}")
    print(f"Top {args.top_n} predictions:\n")

    prediction = expert.predict({
        'query': query,
        'top_n': args.top_n,
        'command_type': args.type
    })

    for i, (cmd, confidence) in enumerate(prediction.items, 1):
        print(f"  {i}. [{confidence:.2f}] {cmd}")

    if prediction.metadata.get('keywords'):
        print(f"\nKeywords: {', '.join(prediction.metadata['keywords'])}")


def stats(args):
    """Show model statistics."""
    model_path = get_model_path()

    if not model_path.exists():
        print("No trained model found. Run 'train' first.")
        return

    expert = CommandExpert.load(model_path)

    print(f"CommandExpert Statistics")
    print(f"{'=' * 40}")
    print(f"Expert ID:        {expert.expert_id}")
    print(f"Version:          {expert.version}")
    print(f"Created:          {expert.created_at}")
    print(f"Trained on:       {expert.trained_on_commits} commands")
    print(f"Sessions:         {expert.trained_on_sessions}")
    print(f"Total commands:   {expert.model_data.get('total_commands', 0)}")
    print(f"Unique commands:  {len(expert.model_data.get('command_frequency', {}))}")
    print(f"Keywords:         {len(expert.model_data.get('keyword_to_commands', {}))}")
    print(f"Python -c:        {len(expert.model_data.get('python_c_patterns', {}))}")

    if expert.metrics:
        print(f"\nMetrics:")
        print(f"  MRR:        {expert.metrics.mrr:.3f}")
        print(f"  Recall@5:   {expert.metrics.recall_at_k.get(5, 0):.3f}")
        print(f"  Precision@1: {expert.metrics.precision_at_k.get(1, 0):.3f}")

    # Show top python -c patterns
    python_c = expert.model_data.get('python_c_patterns', {})
    if python_c:
        print(f"\nTop Python -c patterns:")
        sorted_patterns = sorted(python_c.items(), key=lambda x: -x[1].get('count', 0))[:5]
        for cmd, data in sorted_patterns:
            print(f"  [{data.get('count', 0)}x] {cmd[:80]}...")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate CommandExpert')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')

    # Predict
    predict_parser = subparsers.add_parser('predict', help='Make a prediction')
    predict_parser.add_argument('query', nargs='+', help='Task description')
    predict_parser.add_argument('--top-n', type=int, default=5, help='Number of predictions')
    predict_parser.add_argument('--type', help='Command type filter (python, git, etc.)')

    # Stats
    stats_parser = subparsers.add_parser('stats', help='Show model statistics')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'stats':
        stats(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
