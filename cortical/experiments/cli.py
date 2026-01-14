"""
Experiment CLI
==============

Command-line interface for running and comparing experiments.

Usage:
    python -m cortical.experiments.cli run --input samples/unix_evolution.txt --name test
    python -m cortical.experiments.cli compare experiments/runs/2026-01-14_test1 experiments/runs/2026-01-14_test2
    python -m cortical.experiments.cli list
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import ExperimentConfig
from .logging import ExperimentLog, ExperimentMetrics, list_experiments


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cortical.experiments.cli",
        description="Run and manage AttentionGraph experiments",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input text file path",
    )
    run_parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        help="Experiment name (used in output directory)",
    )
    run_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension (default: 16)",
    )
    run_parser.add_argument(
        "--num-heads",
        type=int,
        default=2,  # Default to 2 - tests show much better than 1
        help="Number of attention heads (default: 2)",
    )
    run_parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of attention layers (default: 2)",
    )
    run_parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs (default: 500)",
    )
    run_parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Learning rate (default: 0.03)",
    )
    run_parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: 1.0)",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to use from input (default: 50)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print training progress",
    )
    run_parser.add_argument(
        "--position-encoding",
        type=str,
        choices=["none", "learned"],
        default="none",
        help="Position encoding type (default: none)",
    )

    # TODO(agent): Add these arguments when features are implemented
    # run_parser.add_argument("--dropout", type=float, default=0.0)
    # run_parser.add_argument("--use-bias", action="store_true")
    # run_parser.add_argument("--loss-fn", choices=["mse", "cross_entropy"], default="mse")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiment results")
    compare_parser.add_argument(
        "experiments",
        nargs="+",
        type=str,
        help="Paths to experiment directories to compare",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.add_argument(
        "--dir",
        type=str,
        default="experiments/runs",
        help="Base directory for experiments (default: experiments/runs)",
    )

    return parser


def run_experiment(args: argparse.Namespace) -> int:
    """Run a training experiment."""
    # Import here to avoid slow startup for other commands
    from cortical.graph.attention import create_causal_attention_graph
    from cortical.graph.trainable import Adam, MSELoss
    from cortical.experiments.kernel import ExperimentKernel
    from cortical.experiments.tokenizer import tokenize, build_vocab, tokens_to_ids
    from cortical.experiments.position import create_position_encoding

    # Create config
    config = ExperimentConfig.from_args(args)

    print("=" * 60)
    print("EXPERIMENT: " + config.name)
    print("=" * 60)
    print()
    print(config.summary())
    print()

    # Load and tokenize input
    input_path = Path(config.input_path)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    text = input_path.read_text()
    tokens = tokenize(text)[:config.max_tokens]
    vocab, id_to_token = build_vocab(tokens)
    token_ids = tokens_to_ids(tokens, vocab)

    print(f"Loaded {len(tokens)} tokens, vocabulary size: {len(vocab)}")
    print()

    # Set seed for reproducibility
    np.random.seed(config.seed)

    # Create embeddings
    embeddings = np.random.randn(len(vocab), config.embedding_dim) * 0.35

    # Create position encoding if requested
    pos_encoding = create_position_encoding(
        encoding_type=config.position_encoding,
        max_len=len(tokens),
        embedding_dim=config.embedding_dim,
    )
    if pos_encoding:
        print(f"Using {config.position_encoding} position encoding")

    # Create graph
    graph = create_causal_attention_graph(
        seq_len=len(tokens),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        seed=config.seed,
    )

    # Prepare inputs (token embeddings)
    input_nodes = {
        f"pos_{i}": embeddings[token_ids[i]].copy()
        for i in range(len(tokens))
    }

    # Add position encodings to inputs if enabled
    if pos_encoding:
        input_nodes = pos_encoding.add_to_inputs(input_nodes)

    # Targets are next-token embeddings (without position encoding)
    targets = {
        f"pos_{i}": embeddings[token_ids[i + 1]].copy()
        for i in range(len(tokens) - 1)
    }

    # Initialize graph
    _ = graph.forward(num_layers=config.num_layers, input_nodes=input_nodes)

    # Create optimizer with all trainable parameters
    all_params = graph.parameters()
    if pos_encoding:
        all_params = all_params + pos_encoding.parameters()

    optimizer = Adam(all_params, lr=config.lr)
    loss_fn = MSELoss()
    kernel = ExperimentKernel(graph, optimizer, loss_fn, profiling=False)

    # Setup logging
    log = ExperimentLog(config)

    # Training loop
    print("Training...")
    start_time = time.time()

    history = kernel.fit(
        targets=targets,
        epochs=config.epochs,
        num_layers=config.num_layers,
        clip_grad=config.clip_grad,
        input_nodes=input_nodes,
        verbose=args.verbose,
    )

    training_time = time.time() - start_time

    # Evaluate final accuracy
    outputs = graph.forward(num_layers=config.num_layers, input_nodes=input_nodes)
    correct = 0
    total = 0

    for i in range(len(tokens) - 1):
        node_id = f"pos_{i}"
        if node_id in outputs:
            output_vec = outputs[node_id]
            distances = np.linalg.norm(embeddings - output_vec, axis=1)
            predicted_id = np.argmin(distances)
            predicted_token = id_to_token.get(predicted_id, "<UNK>")
            actual_token = tokens[i + 1]
            if predicted_token == actual_token:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    # Log metrics
    for loss in history.train_losses:
        log.log_epoch(loss)

    log.finalize(
        final_loss=history.train_losses[-1],
        final_accuracy=accuracy,
        training_time=training_time,
    )

    # Save results
    run_dir = log.save()

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Final loss: {history.train_losses[-1]:.4f}")
    print(f"  Min loss: {min(history.train_losses):.4f}")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Training time: {training_time:.1f}s")
    print()
    print(f"Results saved to: {run_dir}")
    print()

    return 0


def compare_experiments(args: argparse.Namespace) -> int:
    """Compare multiple experiment results."""
    paths = [Path(p) for p in args.experiments]

    # Validate paths
    for path in paths:
        if not path.exists():
            print(f"ERROR: Experiment not found: {path}")
            return 1
        if not (path / "config.json").exists():
            print(f"ERROR: Not a valid experiment directory: {path}")
            return 1

    # Load experiments
    experiments = []
    for path in paths:
        try:
            log = ExperimentLog.load(path)
            experiments.append(log)
        except Exception as e:
            print(f"ERROR: Failed to load {path}: {e}")
            return 1

    # Print comparison table
    print()
    print("=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print()

    # Header
    print(f"{'Name':<20} {'Embed':>6} {'Heads':>6} {'Layers':>7} {'Accuracy':>10} {'Loss':>10}")
    print("-" * 80)

    # Rows
    for exp in experiments:
        name = exp.config.name[:20]
        embed = exp.config.embedding_dim
        heads = exp.config.num_heads
        layers = exp.config.num_layers
        acc = f"{exp.metrics.final_accuracy:.1%}" if exp.metrics.final_accuracy else "N/A"
        loss = f"{exp.metrics.final_loss:.4f}" if exp.metrics.final_loss else "N/A"

        print(f"{name:<20} {embed:>6} {heads:>6} {layers:>7} {acc:>10} {loss:>10}")

    print()

    # Find best
    best_acc = max(experiments, key=lambda e: e.metrics.final_accuracy or 0)
    best_loss = min(experiments, key=lambda e: e.metrics.final_loss or float("inf"))

    print(f"Best accuracy: {best_acc.config.name} ({best_acc.metrics.final_accuracy:.1%})")
    print(f"Lowest loss: {best_loss.config.name} ({best_loss.metrics.final_loss:.4f})")
    print()

    return 0


def list_all_experiments(args: argparse.Namespace) -> int:
    """List all experiments."""
    base_dir = Path(args.dir)
    experiments = list_experiments(base_dir)

    if not experiments:
        print(f"No experiments found in {base_dir}")
        return 0

    print()
    print(f"Experiments in {base_dir}:")
    print("-" * 60)

    for exp_path in experiments:
        try:
            log = ExperimentLog.load(exp_path)
            acc = f"{log.metrics.final_accuracy:.1%}" if log.metrics.final_accuracy else "N/A"
            loss = f"{log.metrics.final_loss:.4f}" if log.metrics.final_loss else "N/A"
            print(f"  {exp_path.name}: acc={acc}, loss={loss}")
        except Exception:
            print(f"  {exp_path.name}: (failed to load)")

    print()
    print(f"Total: {len(experiments)} experiments")
    print()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return run_experiment(args)
    elif args.command == "compare":
        return compare_experiments(args)
    elif args.command == "list":
        return list_all_experiments(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
