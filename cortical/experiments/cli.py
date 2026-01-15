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
        help="Input text file or directory (loads all .txt files)",
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
        choices=["none", "learned", "sinusoidal"],
        default="none",
        help="Position encoding type: 'none', 'learned' (trainable), or 'sinusoidal' (fixed) (default: none)",
    )
    run_parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (default: 0.0)",
    )
    run_parser.add_argument(
        "--use-bias",
        action="store_true",
        help="Enable bias in attention layers",
    )
    run_parser.add_argument(
        "--residual",
        action="store_true",
        help="Enable residual connections (output = attention + input). "
             "Helps gradient flow in multi-layer networks.",
    )
    run_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization) factor (default: 0.0)",
    )
    run_parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of tokens for validation (0.0-0.5). "
             "If 0.0, no validation is performed. (default: 0.0)",
    )
    run_parser.add_argument(
        "--loss-fn",
        type=str,
        choices=["mse", "cross_entropy"],
        default="mse",
        help="Loss function: 'mse' for embedding matching, 'cross_entropy' for language modeling (default: mse)",
    )

    # ============================================================================
    # EXPERIMENTAL FEATURES (stub - not yet implemented)
    # ============================================================================

    # Resume training from checkpoint
    # TODO(agent): Implement checkpoint loading and optimizer state restoration
    # SESSION_HANDOFF: Need to load parameters, optimizer state, and starting epoch
    run_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="[EXPERIMENTAL] Resume training from checkpoint path (not yet implemented)",
    )

    # Early stopping
    # TODO(agent): Implement early stopping with patience counter and best model tracking
    # CONTEXT: Validation infrastructure already exists (val_split, val_losses)
    run_parser.add_argument(
        "--early-stop",
        type=int,
        default=None,
        metavar="PATIENCE",
        help="[EXPERIMENTAL] Stop training if val loss doesn't improve for N epochs (not yet implemented)",
    )
    run_parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        metavar="DELTA",
        help="[EXPERIMENTAL] Minimum improvement to reset early stop patience (default: 1e-4)",
    )

    # Learning rate scheduling
    # TODO(agent): Implement LR schedulers (StepLR, CosineAnnealing, ReduceLROnPlateau)
    # CONTEXT: Optimizer.lr can be modified dynamically
    run_parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["step", "cosine", "plateau"],
        default=None,
        metavar="TYPE",
        help="[EXPERIMENTAL] LR schedule: 'step', 'cosine', or 'plateau' (not yet implemented)",
    )
    run_parser.add_argument(
        "--lr-step-size",
        type=int,
        default=100,
        metavar="N",
        help="[EXPERIMENTAL] Epochs between LR reductions for 'step' schedule (default: 100)",
    )
    run_parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        metavar="FACTOR",
        help="[EXPERIMENTAL] LR decay factor (default: 0.1)",
    )
    run_parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        metavar="LR",
        help="[EXPERIMENTAL] Minimum learning rate (default: 1e-6)",
    )

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


def _check_experimental_features(args: argparse.Namespace) -> None:
    """
    Check for experimental features and raise NotImplementedError.

    This function validates that experimental CLI options are properly
    handled before they're implemented.

    TODO(agent): Remove NotImplementedError once each feature is implemented
    SESSION_HANDOFF: Three features need implementation:
        1. --resume: Checkpoint loading with optimizer state
        2. --early-stop: Patience-based early stopping with best model tracking
        3. --lr-schedule: Learning rate scheduling (step, cosine, plateau)
    """
    experimental_features = []

    # Check --resume
    if args.resume is not None:
        experimental_features.append(f"--resume={args.resume}")
        # TODO(agent): Implement checkpoint loading
        # Steps:
        #   1. Load checkpoint with ExperimentLog.load_checkpoint()
        #   2. Restore parameters with ExperimentLog.restore_parameters()
        #   3. Restore optimizer state with optimizer.load_state_dict()
        #   4. Get starting epoch from checkpoint
        raise NotImplementedError(
            "Resume training (--resume) is not yet implemented.\n"
            "Checkpoint loading infrastructure exists but needs integration.\n"
            "See: cortical/experiments/logging.py:save_checkpoint()"
        )

    # Check --early-stop
    if args.early_stop is not None:
        experimental_features.append(f"--early-stop={args.early_stop}")
        # TODO(agent): Implement early stopping
        # Steps:
        #   1. Track best_val_loss and patience_counter in training loop
        #   2. If val_loss improves by min_delta, reset counter
        #   3. If counter exceeds patience, stop training
        #   4. Save best model checkpoint when val_loss improves
        raise NotImplementedError(
            "Early stopping (--early-stop) is not yet implemented.\n"
            "Validation infrastructure exists (--val-split) but early stopping logic is needed.\n"
            "Requires: patience counter, best model tracking, min_delta comparison"
        )

    # Check --lr-schedule
    if args.lr_schedule is not None:
        experimental_features.append(f"--lr-schedule={args.lr_schedule}")
        # TODO(agent): Implement LR scheduling
        # Steps:
        #   1. Create scheduler.py with LRScheduler base class
        #   2. Implement StepLR: decay every N epochs
        #   3. Implement CosineAnnealingLR: smooth decay to lr_min
        #   4. Implement ReduceLROnPlateau: decay when val_loss stalls
        #   5. Call scheduler.step() in training loop
        raise NotImplementedError(
            f"LR scheduling (--lr-schedule={args.lr_schedule}) is not yet implemented.\n"
            "Optimizer.lr can be modified dynamically but scheduler classes are needed.\n"
            "Types to implement: step, cosine, plateau"
        )

    # Print warning if any experimental features are used (before NotImplementedError)
    if experimental_features:
        print()
        print("=" * 60)
        print("WARNING: EXPERIMENTAL FEATURES DETECTED")
        print("=" * 60)
        for feature in experimental_features:
            print(f"  • {feature}")
        print()
        print("These features are stubbed out and not yet functional.")
        print("See cortical/experiments/scheduler.py for implementation stubs.")
        print("=" * 60)
        print()


def run_experiment(args: argparse.Namespace) -> int:
    """Run a training experiment."""
    # Import here to avoid slow startup for other commands
    from cortical.graph.attention import create_causal_attention_graph
    from cortical.graph.trainable import Adam, MSELoss
    from cortical.experiments.kernel import ExperimentKernel
    from cortical.experiments.tokenizer import tokenize, build_vocab, tokens_to_ids, load_text
    from cortical.experiments.position import create_position_encoding
    from cortical.experiments.projection import VocabProjection, CrossEntropyWithLogits

    # ============================================================================
    # Check for experimental features and warn/fail
    # ============================================================================
    _check_experimental_features(args)

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
        print(f"ERROR: Input path not found: {input_path}")
        return 1

    text = load_text(input_path)
    tokens = tokenize(text)[:config.max_tokens]

    is_dir = input_path.is_dir()
    if is_dir:
        file_count = len(list(input_path.glob("*.txt")))
        print(f"Loaded {file_count} files from {input_path}")
    else:
        print(f"Loaded from {input_path}")
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
        dropout=config.dropout,
        use_bias=config.use_bias,
        use_residual=config.residual,
    )

    # Prepare inputs (token embeddings)
    input_nodes = {
        f"pos_{i}": embeddings[token_ids[i]].copy()
        for i in range(len(tokens))
    }

    # Add position encodings to inputs if enabled
    if pos_encoding:
        input_nodes = pos_encoding.add_to_inputs(input_nodes)

    # Initialize graph
    _ = graph.forward(num_layers=config.num_layers, input_nodes=input_nodes)

    # Create optimizer with all trainable parameters
    all_params = graph.parameters()
    if pos_encoding:
        all_params = all_params + pos_encoding.parameters()

    # Setup loss function and targets based on config
    vocab_proj = None
    if config.loss_fn == "cross_entropy":
        # Cross-entropy mode: use vocab projection and token indices as targets
        vocab_proj = VocabProjection(
            embedding_dim=config.embedding_dim,
            vocab_size=len(vocab),
        )
        all_params = all_params + vocab_proj.parameters()
        loss_fn = CrossEntropyWithLogits()

        # Targets are next-token indices (as one-hot for compatibility)
        all_targets = {
            f"pos_{i}": np.eye(len(vocab))[token_ids[i + 1]]
            for i in range(len(tokens) - 1)
        }
        print(f"Using cross-entropy loss with vocabulary projection")
    else:
        # MSE mode: targets are next-token embeddings
        loss_fn = MSELoss()
        all_targets = {
            f"pos_{i}": embeddings[token_ids[i + 1]].copy()
            for i in range(len(tokens) - 1)
        }

    # Split targets into train/val if val_split > 0
    # We split the prediction positions, not the tokens themselves
    train_targets = all_targets
    val_targets = {}
    val_positions = []

    if config.val_split > 0:
        # Get all position indices for prediction (0 to len-2)
        all_positions = list(range(len(tokens) - 1))
        n_val = int(len(all_positions) * config.val_split)

        if n_val < 1:
            print(f"Warning: val_split too small, no validation positions. Using all for training.")
        else:
            # Use last n_val positions for validation (more realistic for sequence data)
            # This tests generalization to later positions
            val_positions = all_positions[-n_val:]
            train_positions = all_positions[:-n_val]

            train_targets = {f"pos_{i}": all_targets[f"pos_{i}"] for i in train_positions}
            val_targets = {f"pos_{i}": all_targets[f"pos_{i}"] for i in val_positions}

            print(f"Train/val split: {len(train_targets)} train, {len(val_targets)} val positions")

    targets = train_targets  # Use train_targets for training loop

    optimizer = Adam(all_params, lr=config.lr, weight_decay=config.weight_decay)
    kernel = ExperimentKernel(
        graph, optimizer, loss_fn,
        profiling=False,
        position_encoding=pos_encoding,
        vocab_projection=vocab_proj,
    )

    # Setup logging
    log = ExperimentLog(config)

    # Helper function to compute loss on a set of targets
    def compute_loss(target_dict):
        """Compute loss on given targets without gradient updates."""
        graph.eval()  # Disable dropout for evaluation
        outputs = graph.forward(num_layers=config.num_layers, input_nodes=input_nodes)

        total_loss = 0.0
        count = 0
        for node_id, target in target_dict.items():
            if node_id in outputs:
                if vocab_proj is not None:
                    logits = vocab_proj.forward({node_id: outputs[node_id]}, apply_softmax=False)
                    output = logits[node_id]
                else:
                    output = outputs[node_id]
                total_loss += loss_fn(output, target)
                count += 1

        graph.train()  # Re-enable dropout
        return total_loss / count if count > 0 else 0.0

    # Helper function to compute accuracy on positions
    def compute_accuracy(positions):
        """Compute accuracy on given positions."""
        graph.eval()
        outputs = graph.forward(num_layers=config.num_layers, input_nodes=input_nodes)
        correct = 0
        total = 0

        if config.loss_fn == "cross_entropy" and vocab_proj is not None:
            logits = vocab_proj.forward(outputs, apply_softmax=False)
            for i in positions:
                node_id = f"pos_{i}"
                if node_id in logits:
                    predicted_id = int(np.argmax(logits[node_id]))
                    predicted_token = id_to_token.get(predicted_id, "<UNK>")
                    actual_token = tokens[i + 1]
                    if predicted_token == actual_token:
                        correct += 1
                    total += 1
        else:
            for i in positions:
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

        graph.train()
        return correct / total if total > 0 else 0.0, correct, total

    # Training loop with optional validation
    print("Training...")
    start_time = time.time()

    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        # Training step - extract loss value from StepMetrics
        step_metrics = kernel.train_step(
            targets=targets,
            num_layers=config.num_layers,
            clip_grad=config.clip_grad,
            input_nodes=input_nodes,
        )
        train_loss = step_metrics.loss
        train_losses.append(train_loss)

        # Compute validation loss if we have validation data
        val_loss = None
        if val_targets:
            val_loss = compute_loss(val_targets)
            val_losses.append(val_loss)

        # Log epoch
        log.log_epoch(train_loss, val_loss=val_loss)

        # Verbose output
        if args.verbose and (epoch + 1) % max(1, config.epochs // 20) == 0:
            msg = f"Epoch {epoch + 1}/{config.epochs}: train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            print(msg)

    training_time = time.time() - start_time

    # Evaluate final accuracy
    all_positions = list(range(len(tokens) - 1))
    train_positions = [i for i in all_positions if i not in val_positions] if val_positions else all_positions

    train_accuracy, train_correct, train_total = compute_accuracy(train_positions)
    accuracy = train_accuracy  # Default accuracy is train accuracy

    val_accuracy = None
    val_correct = 0
    val_total = 0
    final_val_loss = None
    if val_targets:
        val_accuracy, val_correct, val_total = compute_accuracy(val_positions)
        final_val_loss = compute_loss(val_targets)

    # Log final metrics
    log.finalize(
        final_loss=train_losses[-1],
        final_accuracy=train_accuracy,
        training_time=training_time,
        final_val_loss=final_val_loss,
        final_val_accuracy=val_accuracy,
    )

    # Save results
    run_dir = log.save()

    # Save model checkpoint
    # TODO(agent): Pass scheduler when LR scheduling is implemented
    checkpoint_path = log.save_checkpoint(
        all_params,
        optimizer=optimizer,
        epoch=config.epochs,  # Final epoch
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Min train loss: {min(train_losses):.4f}")
    print(f"  Train accuracy: {train_accuracy:.1%} ({train_correct}/{train_total})")
    if val_targets:
        print(f"  Final val loss: {final_val_loss:.4f}")
        print(f"  Min val loss: {min(val_losses):.4f}")
        print(f"  Val accuracy: {val_accuracy:.1%} ({val_correct}/{val_total})")
    print(f"  Training time: {training_time:.1f}s")
    print()
    print(f"Results saved to: {run_dir}")
    print(f"Model checkpoint: {checkpoint_path}")
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
