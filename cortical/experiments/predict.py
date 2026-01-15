#!/usr/bin/env python3
"""
Load a trained model checkpoint and make predictions.

Usage:
    python -m cortical.experiments.predict experiments/runs/2026-01-15_overfit-cosine-lr/checkpoint.pkl
"""

import argparse
import numpy as np
from pathlib import Path

from cortical.graph.attention import create_causal_attention_graph
from cortical.experiments.tokenizer import tokenize, build_vocab, load_text
from cortical.experiments.logging import ExperimentLog


def load_model(checkpoint_path: Path):
    """Load model from checkpoint and return graph, embeddings, and token mappings."""
    checkpoint = ExperimentLog.load_checkpoint(checkpoint_path)
    config = checkpoint["config"]

    # Load and tokenize input (same as training)
    input_path = Path(config["input_path"])
    text = load_text(input_path)
    tokens = tokenize(text)[:config["max_tokens"]]
    token_to_id, id_to_token = build_vocab(tokens)
    token_ids = [token_to_id[t] for t in tokens]

    # Create embeddings (same seed and scale as training)
    np.random.seed(config["seed"])
    embeddings = np.random.randn(len(token_to_id), config["embedding_dim"]) * 0.35

    # Create graph (same architecture as training)
    graph = create_causal_attention_graph(
        seq_len=len(tokens),
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        seed=config["seed"],
        dropout=config.get("dropout", 0.0),
        use_bias=config.get("use_bias", False),
        use_residual=config.get("residual", False),
    )

    # Prepare inputs
    input_nodes = {
        f"pos_{i}": embeddings[token_ids[i]].copy()
        for i in range(len(tokens))
    }

    # Initialize layers by running forward
    _ = graph.forward(num_layers=config["num_layers"], input_nodes=input_nodes)

    # Restore trained parameters
    all_params = graph.parameters()
    restored = ExperimentLog.restore_parameters(all_params, checkpoint)

    return {
        "graph": graph,
        "config": config,
        "embeddings": embeddings,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "input_nodes": input_nodes,
        "checkpoint": checkpoint,
        "restored_params": restored,
    }


def predict(model_data: dict) -> dict:
    """Run predictions and return results."""
    graph = model_data["graph"]
    config = model_data["config"]
    embeddings = model_data["embeddings"]
    tokens = model_data["tokens"]
    token_ids = model_data["token_ids"]
    id_to_token = model_data["id_to_token"]
    input_nodes = model_data["input_nodes"]

    # Run forward in eval mode
    graph.eval()
    outputs = graph.forward(num_layers=config["num_layers"], input_nodes=input_nodes)

    # Compute predictions
    predictions = []
    correct = 0
    total = 0

    for i in range(len(tokens) - 1):
        node_id = f"pos_{i}"
        if node_id in outputs:
            output_vec = outputs[node_id]
            # Find closest embedding (nearest neighbor)
            distances = np.linalg.norm(embeddings - output_vec, axis=1)
            predicted_id = int(np.argmin(distances))
            actual_id = token_ids[i + 1]

            is_correct = predicted_id == actual_id
            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                "position": i,
                "context": " ".join(tokens[max(0, i-2):i+1]),
                "predicted": id_to_token[predicted_id],
                "actual": tokens[i + 1],
                "correct": is_correct,
                "distance": float(distances[predicted_id]),
            })

    return {
        "predictions": predictions,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Load model and make predictions")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint.pkl")
    parser.add_argument("--limit", type=int, default=None, help="Limit predictions shown")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    model_data = load_model(checkpoint_path)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trained for {model_data['checkpoint']['epoch']} epochs")
    print(f"Vocabulary size: {len(model_data['token_to_id'])}")
    print(f"Sequence length: {len(model_data['tokens'])}")
    print(f"Restored {model_data['restored_params']}/{len(model_data['graph'].parameters())} parameters")
    print()

    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    results = predict(model_data)

    limit = args.limit or len(results["predictions"])
    for pred in results["predictions"][:limit]:
        match = "✓" if pred["correct"] else "✗"
        print(f"pos_{pred['position']:2d}: '{pred['context']}'")
        print(f"       predicted: {pred['predicted']:10s}  actual: {pred['actual']:10s} {match}")

    if limit < len(results["predictions"]):
        print(f"... ({len(results['predictions']) - limit} more predictions)")

    print()
    print("=" * 60)
    print(f"ACCURACY: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
