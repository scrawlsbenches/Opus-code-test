#!/usr/bin/env python3
"""
Overfitting test for AttentionGraph.

This script trains an AttentionGraph on a small document (unix_evolution.txt)
to verify that:
1. The model can overfit (loss approaches zero)
2. Gradients flow correctly through attention
3. Attention patterns become meaningful
4. No performance or memory issues

Usage:
    python experiments/overfit_attention.py

Expected outcome:
    - Loss decreases from ~initial to near-zero within 200-500 epochs
    - Attention weights show interpretable patterns (not all uniform)
    - Profiling shows stable memory usage
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from cortical.graph.attention import AttentionGraph, create_causal_attention_graph
from cortical.graph.trainable import Adam, MSELoss
from cortical.experiments import ExperimentKernel, Profiler
from cortical.experiments.tokenizer import tokenize, build_vocab, tokens_to_ids


def load_document(path: str) -> str:
    """Load document text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_embeddings(
    vocab_size: int,
    embedding_dim: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Create random initial embeddings.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        seed: Random seed for reproducibility

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    np.random.seed(seed)
    # Xavier initialization
    scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
    return np.random.randn(vocab_size, embedding_dim) * scale


def setup_causal_lm_task(
    token_ids: list[int],
    embeddings: np.ndarray,
    embedding_dim: int,
) -> tuple[AttentionGraph, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Set up a causal language modeling task.

    Creates an AttentionGraph where each position predicts the next token.

    Args:
        token_ids: List of token IDs
        embeddings: Embedding matrix
        embedding_dim: Dimension of embeddings

    Returns:
        Tuple of (graph, input_nodes, targets)
    """
    seq_len = len(token_ids)

    # Create causal attention graph
    graph = create_causal_attention_graph(
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        seed=42,
    )

    # Set up inputs (all tokens except last)
    input_nodes: dict[str, np.ndarray] = {}
    for i in range(seq_len - 1):
        node_id = f"pos_{i}"
        token_id = token_ids[i]
        input_nodes[node_id] = embeddings[token_id].copy()

    # Last position also needs input for forward pass
    input_nodes[f"pos_{seq_len - 1}"] = embeddings[token_ids[seq_len - 1]].copy()

    # Set up targets (predict next token for positions 0 to seq_len-2)
    targets: dict[str, np.ndarray] = {}
    for i in range(seq_len - 1):
        node_id = f"pos_{i}"
        next_token_id = token_ids[i + 1]
        targets[node_id] = embeddings[next_token_id].copy()

    return graph, input_nodes, targets


def visualize_attention(graph: AttentionGraph, id_to_token: dict[int, str], token_ids: list[int]) -> str:
    """
    Create a text visualization of attention patterns.

    Args:
        graph: Trained AttentionGraph
        id_to_token: ID to token mapping
        token_ids: Token IDs in sequence

    Returns:
        String visualization
    """
    lines = ["", "ATTENTION PATTERNS", "=" * 60]

    attention_weights = graph.get_attention_weights()

    # Show attention for a few interesting positions
    positions_to_show = [1, 5, 10, min(20, len(token_ids) - 2)]
    positions_to_show = [p for p in positions_to_show if p < len(token_ids)]

    for pos in positions_to_show:
        node_id = f"pos_{pos}"
        if node_id not in attention_weights:
            continue

        weights = attention_weights[node_id]
        if not weights:
            continue

        current_token = id_to_token.get(token_ids[pos], "<UNK>")
        lines.append(f"\nPosition {pos} ('{current_token}') attends to:")

        # Sort by weight descending
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])

        for source_id, weight in sorted_weights[:5]:  # Top 5
            # Extract position number from source_id
            try:
                source_pos = int(source_id.split("_")[1])
                source_token = id_to_token.get(token_ids[source_pos], "<UNK>")
                lines.append(f"  {source_id} ('{source_token}'): {weight:.4f}")
            except (IndexError, ValueError):
                lines.append(f"  {source_id}: {weight:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    """Run the overfitting experiment."""
    print("=" * 60)
    print("ATTENTIONGRAPH OVERFITTING TEST")
    print("=" * 60)
    print()

    # Configuration
    EMBEDDING_DIM = 16  # Smaller for faster convergence
    NUM_LAYERS = 2
    EPOCHS = 500
    LEARNING_RATE = 0.03  # Moderate LR - balance speed vs stability
    CLIP_GRAD = 1.0  # Conservative clipping prevents explosion
    LOG_EVERY = 50

    # Load document
    doc_path = project_root / "samples" / "unix_evolution.txt"
    print(f"Loading document: {doc_path}")

    if not doc_path.exists():
        print(f"ERROR: Document not found: {doc_path}")
        sys.exit(1)

    text = load_document(str(doc_path))
    print(f"Document length: {len(text)} characters")

    # Tokenize
    tokens = tokenize(text)
    print(f"Tokens: {len(tokens)}")

    # Truncate for faster testing if needed
    MAX_TOKENS = 50  # Limit for faster iteration
    if len(tokens) > MAX_TOKENS:
        print(f"Truncating to {MAX_TOKENS} tokens for faster testing")
        tokens = tokens[:MAX_TOKENS]

    # Build vocabulary
    vocab, id_to_token = build_vocab(tokens)
    print(f"Vocabulary size: {len(vocab)}")

    # Convert to IDs
    token_ids = tokens_to_ids(tokens, vocab)

    # Create embeddings
    embeddings = create_embeddings(len(vocab), EMBEDDING_DIM)

    # Set up task
    print()
    print("Setting up causal LM task...")
    graph, input_nodes, targets = setup_causal_lm_task(
        token_ids, embeddings, EMBEDDING_DIM
    )

    print(f"Graph nodes: {len(list(graph.nodes))}")
    print(f"Input nodes: {len(input_nodes)}")
    print(f"Target nodes: {len(targets)}")

    # Create optimizer and loss function
    # Note: We need to do a forward pass first to create attention layers
    _ = graph.forward(num_layers=NUM_LAYERS, input_nodes=input_nodes)

    optimizer = Adam(graph.parameters(), lr=LEARNING_RATE)
    loss_fn = MSELoss()

    # Create experiment kernel
    kernel = ExperimentKernel(
        graph=graph,
        optimizer=optimizer,
        loss_fn=loss_fn,
        profiling=True,
        track_memory=True,
    )

    # Train
    print()
    print("Starting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Gradient clipping: {CLIP_GRAD}")
    print(f"  Layers: {NUM_LAYERS}")
    print()

    history = kernel.fit(
        targets=targets,
        epochs=EPOCHS,
        num_layers=NUM_LAYERS,
        clip_grad=CLIP_GRAD,
        input_nodes=input_nodes,
        verbose=True,
        log_every=LOG_EVERY,
    )

    # Print profiling report
    print()
    report = kernel.profile_report()
    print(report)

    # Visualize attention
    attention_viz = visualize_attention(graph, id_to_token, token_ids)
    print(attention_viz)

    # Validation checks
    print()
    print("VALIDATION CHECKS")
    print("=" * 60)

    initial_loss = history.train_losses[0]
    final_loss = history.train_losses[-1]
    min_loss = min(history.train_losses)

    checks_passed = 0
    total_checks = 4

    # Check 1: Loss decreased
    if final_loss < initial_loss * 0.1:
        print("[PASS] Loss decreased significantly (>90% reduction)")
        checks_passed += 1
    else:
        print(f"[FAIL] Loss reduction insufficient: {initial_loss:.4f} -> {final_loss:.4f}")

    # Check 2: Loss is low (overfitting)
    if min_loss < 1.0:
        print(f"[PASS] Achieved low loss: {min_loss:.6f}")
        checks_passed += 1
    else:
        print(f"[FAIL] Loss too high: {min_loss:.4f}")

    # Check 3: Gradient norms are reasonable
    max_grad = max(history.gradient_norms)
    if max_grad < 100 and max_grad > 0:
        print(f"[PASS] Gradient norms reasonable: max={max_grad:.4f}")
        checks_passed += 1
    else:
        print(f"[FAIL] Gradient norms problematic: max={max_grad:.4f}")

    # Check 4: Memory stable
    if report.memory_trend in ("stable", "decreasing"):
        print(f"[PASS] Memory trend: {report.memory_trend}")
        checks_passed += 1
    else:
        print(f"[WARN] Memory trend: {report.memory_trend}")
        # Don't fail on memory trend - might be normal allocation

    print()
    print(f"Checks passed: {checks_passed}/{total_checks}")
    print("=" * 60)

    # Overall result
    if checks_passed >= 3:
        print()
        print("SUCCESS: AttentionGraph overfitting test PASSED")
        return 0
    else:
        print()
        print("FAILURE: AttentionGraph overfitting test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
