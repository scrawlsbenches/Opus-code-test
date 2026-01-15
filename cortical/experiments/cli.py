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

# Token embedding initialization scale
# This value is used to scale random embeddings for stable training.
# Must match between training (cli.py) and inference (predict.py).
EMBEDDING_INIT_SCALE = 0.35


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
    # ADVANCED TRAINING FEATURES
    # ============================================================================

    # Resume training from checkpoint
    run_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Resume training from checkpoint.pkl path (restores parameters, optimizer, scheduler)",
    )

    # Vocabulary file
    run_parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        metavar="VOCAB_FILE",
        help="Use pre-built vocabulary JSON file instead of building from input",
    )

    # Early stopping
    run_parser.add_argument(
        "--early-stop",
        type=int,
        default=None,
        metavar="PATIENCE",
        help="Stop training if val loss doesn't improve for N epochs. "
             "Requires --val-split > 0. Restores best model parameters at end.",
    )
    run_parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        metavar="DELTA",
        help="Minimum improvement to reset early stop patience (default: 1e-4)",
    )

    # Learning rate scheduling
    run_parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["step", "cosine", "plateau"],
        default=None,
        metavar="TYPE",
        help="LR schedule: 'step' (decay every N epochs), 'cosine' (smooth decay), "
             "or 'plateau' (reduce on val loss stall)",
    )
    run_parser.add_argument(
        "--lr-step-size",
        type=int,
        default=100,
        metavar="N",
        help="Epochs between LR reductions for 'step' schedule (default: 100)",
    )
    run_parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        metavar="FACTOR",
        help="LR decay factor for 'step' and 'plateau' schedules (default: 0.1)",
    )
    run_parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        metavar="LR",
        help="Minimum learning rate for 'cosine' and 'plateau' (default: 1e-6)",
    )
    run_parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="Number of epochs for linear LR warmup (default: 0, disabled)",
    )
    run_parser.add_argument(
        "--warmup-start-lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Starting learning rate for warmup (default: 0)",
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

    # Vocab command (with subcommands)
    vocab_parser = subparsers.add_parser("vocab", help="Vocabulary management")
    vocab_subparsers = vocab_parser.add_subparsers(dest="vocab_command", help="Vocab commands")

    # vocab create
    vocab_create_parser = vocab_subparsers.add_parser("create", help="Create vocabulary from corpus")
    vocab_create_parser.add_argument(
        "--from",
        dest="from_path",
        type=str,
        required=True,
        help="Input file or directory (loads all .txt files from directory)",
    )
    vocab_create_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output vocabulary JSON file",
    )
    vocab_create_parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum token frequency for inclusion (default: 1)",
    )
    vocab_create_parser.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="Maximum vocabulary size (default: unlimited)",
    )

    # vocab inspect
    vocab_inspect_parser = vocab_subparsers.add_parser("inspect", help="Inspect vocabulary file")
    vocab_inspect_parser.add_argument(
        "vocab_path",
        type=str,
        help="Path to vocabulary JSON file",
    )
    vocab_inspect_parser.add_argument(
        "--show-tokens",
        type=int,
        default=10,
        help="Number of sample tokens to show (default: 10)",
    )

    return parser


def _validate_feature_requirements(args: argparse.Namespace) -> None:
    """
    Validate feature requirements and dependencies.

    This function validates that CLI options are properly configured
    and have required dependencies.

    Features:
        - --resume: Checkpoint loading with optimizer/scheduler state
        - --lr-schedule: Learning rate scheduling (step, cosine, plateau)
        - --warmup-epochs: Linear LR warmup to prevent gradient explosion
        - --early-stop: Patience-based early stopping with best model tracking
    """
    # --early-stop requires --val-split > 0
    if args.early_stop is not None and args.val_split <= 0:
        raise ValueError(
            "Early stopping (--early-stop) requires validation split.\n"
            "Please add --val-split 0.1 (or similar) to enable validation loss monitoring."
        )


def run_experiment(args: argparse.Namespace) -> int:
    """Run a training experiment."""
    # Import here to avoid slow startup for other commands
    from cortical.graph.attention import create_causal_attention_graph
    from cortical.graph.trainable import Adam, MSELoss
    from cortical.experiments.kernel import ExperimentKernel
    from cortical.experiments.tokenizer import tokenize, build_vocab, tokens_to_ids, load_text
    from cortical.experiments.position import create_position_encoding
    from cortical.experiments.projection import VocabProjection, CrossEntropyWithLogits
    from cortical.experiments.scheduler import create_scheduler
    from cortical.experiments.early_stopping import EarlyStopper

    # ============================================================================
    # Validate feature requirements
    # ============================================================================
    _validate_feature_requirements(args)

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

    # Use provided vocabulary or build from input
    if args.vocab:
        from .vocabulary import Vocabulary
        vocab_obj = Vocabulary.load(args.vocab)
        vocab = vocab_obj.get_token_to_id()
        id_to_token = vocab_obj.get_id_to_token()
        print(f"Using vocabulary from: {args.vocab}")
    else:
        vocab, id_to_token = build_vocab(tokens)

    token_ids = tokens_to_ids(tokens, vocab)

    print(f"Loaded {len(tokens)} tokens, vocabulary size: {len(vocab)} tokens")
    print()

    # Set seed for reproducibility
    np.random.seed(config.seed)

    # Create embeddings
    embeddings = np.random.randn(len(vocab), config.embedding_dim) * EMBEDDING_INIT_SCALE

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
    # TODO: Add --profile CLI flag to enable profiling with memory tracking.
    # Simplest option: always track memory when profiling is enabled.
    kernel = ExperimentKernel(
        graph, optimizer, loss_fn,
        profiling=False,
        position_encoding=pos_encoding,
        vocab_projection=vocab_proj,
    )

    # Setup LR scheduler if requested
    scheduler = None
    if config.lr_schedule is not None:
        scheduler = create_scheduler(
            optimizer,
            schedule_type=config.lr_schedule,
            epochs=config.epochs,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
            lr_min=config.lr_min,
            warmup_epochs=config.warmup_epochs,
            warmup_start_lr=config.warmup_start_lr,
        )
        schedule_info = f"Using {config.lr_schedule} LR schedule (gamma={config.lr_gamma})"
        if config.warmup_epochs > 0:
            schedule_info += f" with {config.warmup_epochs} warmup epochs"
        print(schedule_info)

    # Setup early stopping if requested
    early_stopper = None
    best_param_snapshot = None
    if args.early_stop is not None:
        early_stopper = EarlyStopper(
            patience=args.early_stop,
            min_delta=args.early_stop_min_delta,
            mode="min",  # Lower val_loss is better
        )
        print(f"Using early stopping (patience={args.early_stop}, min_delta={args.early_stop_min_delta})")

    # Helper functions for parameter snapshots (for early stopping)
    def save_param_snapshot(params):
        """Save a snapshot of parameter values."""
        return {p.name: p.data.copy() for p in params}

    def restore_param_snapshot(params, snapshot):
        """Restore parameters from snapshot."""
        for p in params:
            if p.name in snapshot:
                p.data[:] = snapshot[p.name]

    # ========================================================================
    # Resume from checkpoint if requested
    # ========================================================================
    start_epoch = 0
    if args.resume is not None:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return 1

        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = ExperimentLog.load_checkpoint(checkpoint_path)

        # Restore parameters
        restored_count = ExperimentLog.restore_parameters(all_params, checkpoint)
        print(f"  Restored {restored_count}/{len(all_params)} parameters")

        # Restore optimizer state
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(f"  Restored optimizer state (step={optimizer.t}, lr={optimizer.lr:.2e})")

        # Restore scheduler state if present and scheduler is configured
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"  Restored scheduler state (last_epoch={scheduler.last_epoch})")

        # Get starting epoch from checkpoint
        if checkpoint.get("epoch") is not None:
            start_epoch = checkpoint["epoch"]
            print(f"  Resuming from epoch {start_epoch}")

        print()

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
    if start_epoch > 0:
        print(f"Continuing training from epoch {start_epoch} to {config.epochs}...")
    else:
        print("Training...")
    start_time = time.time()

    train_losses = []
    val_losses = []
    stopped_early = False
    final_epoch = config.epochs

    for epoch in range(start_epoch, config.epochs):
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

        # Early stopping check
        if early_stopper is not None and val_loss is not None:
            result = early_stopper.step(val_loss, epoch=epoch)

            # Save best parameters when val_loss improves
            if result.is_best:
                best_param_snapshot = save_param_snapshot(all_params)

            # Check if we should stop
            if result.should_stop:
                stopped_early = True
                final_epoch = epoch + 1
                if args.verbose:
                    print(f"Early stopping at epoch {epoch + 1} (patience={args.early_stop} exceeded)")
                    print(f"  Best val_loss: {early_stopper.best:.4f}")
                break

        # Update learning rate schedule
        if scheduler is not None:
            if config.lr_schedule == "plateau":
                # ReduceLROnPlateau needs metric (use val_loss or train_loss)
                metric = val_loss if val_loss is not None else train_loss
                scheduler.step(metric)
            else:
                # StepLR and CosineAnnealing use epoch
                scheduler.step(epoch=epoch)

        # Verbose output
        if args.verbose and (epoch + 1) % max(1, config.epochs // 20) == 0:
            msg = f"Epoch {epoch + 1}/{config.epochs}: train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            if scheduler is not None:
                msg += f", lr={optimizer.lr:.2e}"
            if early_stopper is not None:
                msg += f", patience={early_stopper.patience - early_stopper.counter}"
            print(msg)

    training_time = time.time() - start_time

    # Restore best parameters if early stopped
    if stopped_early and best_param_snapshot is not None:
        restore_param_snapshot(all_params, best_param_snapshot)
        if args.verbose:
            print(f"Restored best parameters (val_loss={early_stopper.best:.4f})")

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

    # Save model checkpoint (includes optimizer and scheduler state for resume)
    checkpoint_path = log.save_checkpoint(
        all_params,
        optimizer=optimizer,
        epoch=final_epoch,  # Actual final epoch (may differ if early stopped)
        scheduler=scheduler,  # May be None if no LR schedule
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    if stopped_early:
        print(f"  Early stopped at epoch {final_epoch}/{config.epochs}")
        print(f"  Best val_loss: {early_stopper.best:.4f}")
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


def vocab_create(args: argparse.Namespace) -> int:
    """Create vocabulary from corpus."""
    import json
    from .vocabulary import Vocabulary

    print(f"Creating vocabulary from: {args.from_path}")

    try:
        vocab = Vocabulary.from_file(
            args.from_path,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    vocab.save(args.output)

    print(f"Vocabulary created: {vocab.size} tokens")
    print(f"  - Special tokens: 4")
    print(f"  - Regular tokens: {vocab.size - 4}")
    print(f"  - Min frequency: {args.min_freq}")
    print(f"  - Source files: {len(vocab.source_files)}")
    print(f"Saved to: {args.output}")

    return 0


def vocab_inspect(args: argparse.Namespace) -> int:
    """Inspect vocabulary file."""
    import json
    from .vocabulary import Vocabulary

    try:
        vocab = Vocabulary.load(args.vocab_path)
    except FileNotFoundError:
        print(f"ERROR: Vocabulary file not found: {args.vocab_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in vocabulary file: {e}")
        return 1

    print(f"\nVocabulary: {args.vocab_path}")
    print("-" * 50)
    print(f"Size: {vocab.size} tokens")
    print(f"Hash: {vocab.hash()}")
    print(f"Source files: {vocab.source_files}")

    # Show sample tokens
    id_to_token = vocab.get_id_to_token()
    print(f"\nSample tokens (first {args.show_tokens}):")
    for i in range(min(args.show_tokens, vocab.size)):
        token = id_to_token.get(i, "?")
        print(f"  {i}: {token!r}")

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
    elif args.command == "vocab":
        if args.vocab_command == "create":
            return vocab_create(args)
        elif args.vocab_command == "inspect":
            return vocab_inspect(args)
        else:
            parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
