#!/usr/bin/env python3
"""
Bigram Language Model using TrainableGraph.

This example trains word embeddings using bigram co-occurrence data
from the samples/ corpus. Similar to Word2Vec but using graph structure.

The approach:
1. Extract bigrams (word pairs) from documents
2. Build a graph where nodes are words, edges are bigram connections
3. Train: given word A, predict word B if (A, B) is a bigram
4. Result: words that appear in similar contexts have similar embeddings

Usage:
    PYTHONPATH=. python examples/trainable_graph_bigram_model.py

    # With more vocabulary:
    PYTHONPATH=. python examples/trainable_graph_bigram_model.py --vocab-size 500 --embedding-dim 32
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

import numpy as np

from cortical.graph import (
    TrainableGraph,
    Adam,
    MSELoss,
    ContrastiveLoss,
    Activation,
    Aggregation,
)


# =============================================================================
# Text Processing
# =============================================================================

STOP_WORDS = {
    "the", "and", "for", "that", "with", "are", "this", "from",
    "have", "has", "was", "were", "been", "being", "which", "their",
    "can", "more", "into", "such", "when", "also", "than", "these",
    "other", "may", "its", "but", "not", "they", "would", "could",
    "about", "through", "between", "while", "each", "most", "some",
    "will", "only", "over", "where", "after", "before", "just",
    "your", "what", "there", "then", "how", "all", "any", "both",
}


def tokenize(text: str) -> List[str]:
    """Extract lowercase words."""
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,15}\b", text)
    return [w for w in words if w not in STOP_WORDS]


def extract_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    """Extract bigram pairs from token list."""
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i], tokens[i + 1]))
    return bigrams


def load_corpus(samples_dir: str, max_docs: int = 100) -> str:
    """Load text from samples directory."""
    samples_path = Path(samples_dir)
    texts = []

    for i, txt_file in enumerate(samples_path.glob("**/*.txt")):
        if i >= max_docs:
            break
        try:
            texts.append(txt_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return " ".join(texts)


# =============================================================================
# Bigram Graph Builder
# =============================================================================


def build_bigram_graph(
    corpus: str,
    vocab_size: int = 200,
    embedding_dim: int = 16,
    min_bigram_count: int = 2,
) -> Tuple[TrainableGraph, Dict[str, int], Dict[Tuple[str, str], int]]:
    """
    Build a trainable graph from bigram statistics.

    Args:
        corpus: Text corpus
        vocab_size: Number of words to include
        embedding_dim: Dimension of word embeddings
        min_bigram_count: Minimum count to include a bigram edge

    Returns:
        Tuple of (graph, word_to_idx, bigram_counts)
    """
    # Tokenize and count words
    tokens = tokenize(corpus)
    word_counts = Counter(tokens)

    # Build vocabulary from most common words
    vocab = [word for word, _ in word_counts.most_common(vocab_size)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_set = set(vocab)

    print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Top words: {vocab[:10]}")

    # Count bigrams
    bigrams = extract_bigrams(tokens)
    bigram_counts = Counter(bigrams)

    # Filter to vocabulary and minimum count
    valid_bigrams = {
        bg: count for bg, count in bigram_counts.items()
        if bg[0] in vocab_set and bg[1] in vocab_set and count >= min_bigram_count
    }

    print(f"  Valid bigrams: {len(valid_bigrams)}")

    # Create graph
    graph = TrainableGraph(
        embedding_dim=embedding_dim,
        activation=Activation.TANH,
        aggregation=Aggregation.MEAN,
        seed=42,
    )

    # Add word nodes with random initialization
    for word in vocab:
        graph.add_node(word)

    # Add bigram edges with weight based on count
    max_count = max(valid_bigrams.values()) if valid_bigrams else 1

    for (word1, word2), count in valid_bigrams.items():
        # Weight based on normalized frequency
        weight = min(0.95, 0.3 + 0.65 * (count / max_count))
        graph.add_edge(word1, word2, weight=weight)

    print(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")

    return graph, word_to_idx, valid_bigrams


# =============================================================================
# Training
# =============================================================================


def create_training_pairs(
    bigram_counts: Dict[Tuple[str, str], int],
    vocab: Set[str],
    negative_samples: int = 3,
) -> List[Tuple[str, str, float]]:
    """
    Create training pairs: positive (bigram) and negative (random) samples.

    Returns list of (word1, word2, target) where target=1 for bigram, 0 for random.
    """
    pairs = []
    vocab_list = list(vocab)

    # Positive samples (actual bigrams)
    for (w1, w2), count in bigram_counts.items():
        # Add multiple times based on frequency
        for _ in range(min(count, 5)):
            pairs.append((w1, w2, 1.0))

    # Negative samples (random word pairs)
    num_negative = len(pairs) * negative_samples
    for _ in range(num_negative):
        w1 = random.choice(vocab_list)
        w2 = random.choice(vocab_list)
        if (w1, w2) not in bigram_counts and w1 != w2:
            pairs.append((w1, w2, 0.0))

    random.shuffle(pairs)
    return pairs


def train_bigram_model(
    graph: TrainableGraph,
    bigram_counts: Dict[Tuple[str, str], int],
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.01,
) -> List[float]:
    """
    Train the graph to predict bigram relationships.

    Training objective: words that form bigrams should have similar embeddings.
    """
    vocab = set(node.id for node in graph.nodes)
    optimizer = Adam(graph.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        # Create training pairs for this epoch
        pairs = create_training_pairs(bigram_counts, vocab, negative_samples=2)

        graph.train()
        epoch_loss = 0.0
        num_batches = 0

        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            # Forward pass
            outputs = graph.forward(num_layers=1)

            # Compute loss: similar embeddings for bigrams, different for non-bigrams
            batch_loss = 0.0
            output_grads = defaultdict(lambda: np.zeros(graph.embedding_dim))

            for w1, w2, target in batch:
                emb1 = outputs[w1]
                emb2 = outputs[w2]

                # Cosine similarity
                dot = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1) + 1e-8
                norm2 = np.linalg.norm(emb2) + 1e-8
                sim = dot / (norm1 * norm2)

                # Target: 1 for bigrams (similar), 0 for random (different)
                # Loss: (sim - target)^2
                loss = (sim - target) ** 2
                batch_loss += loss

                # Gradient of similarity w.r.t embeddings
                grad_sim = 2 * (sim - target)

                # d(sim)/d(emb1) = emb2 / (norm1 * norm2) - sim * emb1 / norm1^2
                grad_emb1 = grad_sim * (emb2 / (norm1 * norm2) - sim * emb1 / (norm1 ** 2))
                grad_emb2 = grad_sim * (emb1 / (norm1 * norm2) - sim * emb2 / (norm2 ** 2))

                output_grads[w1] += grad_emb1
                output_grads[w2] += grad_emb2

            # Backward pass
            graph.backward(dict(output_grads), num_layers=1)
            graph.clip_gradients(1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += batch_loss
            num_batches += 1

        avg_loss = epoch_loss / max(1, len(pairs))
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    return losses


# =============================================================================
# Evaluation
# =============================================================================


def get_similar_words(
    graph: TrainableGraph,
    word: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Find words most similar to the given word."""
    graph.eval()
    outputs = graph.forward(num_layers=1)

    if word not in outputs:
        return []

    target_emb = outputs[word]
    target_norm = np.linalg.norm(target_emb) + 1e-8

    similarities = []
    for node in graph.nodes:
        if node.id == word:
            continue

        emb = outputs[node.id]
        sim = np.dot(target_emb, emb) / (target_norm * (np.linalg.norm(emb) + 1e-8))
        similarities.append((node.id, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def get_bigram_predictions(
    graph: TrainableGraph,
    word: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Predict most likely next words (bigram completion)."""
    # Get words connected by outgoing edges
    edges = graph.edges_from(word)
    if not edges:
        return get_similar_words(graph, word, top_k)

    graph.eval()
    outputs = graph.forward(num_layers=1)

    if word not in outputs:
        return []

    source_emb = outputs[word]
    source_norm = np.linalg.norm(source_emb) + 1e-8

    predictions = []
    for edge in edges:
        target_word = edge.target_id
        target_emb = outputs[target_word]
        sim = np.dot(source_emb, target_emb) / (source_norm * (np.linalg.norm(target_emb) + 1e-8))
        predictions.append((target_word, sim, edge.weight))

    # Sort by similarity
    predictions.sort(key=lambda x: -x[1])
    return [(w, s) for w, s, _ in predictions[:top_k]]


def analogy(
    graph: TrainableGraph,
    word_a: str,
    word_b: str,
    word_c: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Word analogy: A is to B as C is to ?

    Computes: embedding(B) - embedding(A) + embedding(C)
    Returns words closest to this vector.
    """
    graph.eval()
    outputs = graph.forward(num_layers=1)

    if word_a not in outputs or word_b not in outputs or word_c not in outputs:
        return []

    # Compute analogy vector
    vec = outputs[word_b] - outputs[word_a] + outputs[word_c]
    vec_norm = np.linalg.norm(vec) + 1e-8

    exclude = {word_a, word_b, word_c}

    similarities = []
    for node in graph.nodes:
        if node.id in exclude:
            continue

        emb = outputs[node.id]
        sim = np.dot(vec, emb) / (vec_norm * (np.linalg.norm(emb) + 1e-8))
        similarities.append((node.id, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Bigram language model with TrainableGraph")
    parser.add_argument("--samples-dir", default="samples", help="Path to samples directory")
    parser.add_argument("--vocab-size", type=int, default=300, help="Vocabulary size")
    parser.add_argument("--embedding-dim", type=int, default=24, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate")
    parser.add_argument("--min-bigram-count", type=int, default=3, help="Minimum bigram frequency")
    parser.add_argument("--max-docs", type=int, default=150, help="Maximum documents to load")

    args = parser.parse_args()

    print("=" * 60)
    print("Bigram Language Model with TrainableGraph")
    print("=" * 60)

    # Load corpus
    print(f"\n1. Loading corpus from {args.samples_dir}/...")
    corpus = load_corpus(args.samples_dir, max_docs=args.max_docs)
    tokens = tokenize(corpus)
    print(f"   Total tokens: {len(tokens)}")

    # Build graph
    print(f"\n2. Building bigram graph...")
    graph, word_to_idx, bigram_counts = build_bigram_graph(
        corpus,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        min_bigram_count=args.min_bigram_count,
    )

    # Train
    print(f"\n3. Training (epochs={args.epochs}, lr={args.lr})...")
    losses = train_bigram_model(
        graph,
        bigram_counts,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    print(f"   Final loss: {losses[-1]:.4f}")

    # Evaluate
    print("\n4. Evaluation...")
    print("\n" + "=" * 60)

    # Test words - pick some that likely appear in the corpus
    test_words = ["model", "learning", "data", "system", "process", "art", "design", "code"]
    test_words = [w for w in test_words if w in word_to_idx][:5]

    if test_words:
        print("\n   Similar words (by learned embedding):")
        print("   " + "-" * 50)
        for word in test_words:
            similar = get_similar_words(graph, word, top_k=5)
            if similar:
                similar_str = ", ".join(f"{w}({s:.2f})" for w, s in similar)
                print(f"   {word:12} -> {similar_str}")

        print("\n   Bigram predictions (next word):")
        print("   " + "-" * 50)
        for word in test_words:
            predictions = get_bigram_predictions(graph, word, top_k=5)
            if predictions:
                pred_str = ", ".join(f"{w}({s:.2f})" for w, s in predictions)
                print(f"   {word:12} -> {pred_str}")

    # Show some actual bigrams learned
    print("\n   Sample bigrams from corpus:")
    print("   " + "-" * 50)
    sample_bigrams = list(bigram_counts.items())[:15]
    for (w1, w2), count in sorted(sample_bigrams, key=lambda x: -x[1])[:10]:
        print(f"   '{w1} {w2}' (count: {count})")

    # Word analogies (if vocabulary allows)
    print("\n   Word analogies (A:B :: C:?):")
    print("   " + "-" * 50)

    analogy_tests = [
        ("learning", "machine", "neural"),
        ("art", "painting", "music"),
        ("model", "training", "data"),
    ]

    for a, b, c in analogy_tests:
        if a in word_to_idx and b in word_to_idx and c in word_to_idx:
            results = analogy(graph, a, b, c, top_k=3)
            if results:
                result_str = ", ".join(f"{w}({s:.2f})" for w, s in results)
                print(f"   {a}:{b} :: {c}:? -> {result_str}")

    # Save model
    print("\n5. Model saved.")
    state = graph.save_state()
    print(f"   Parameters: {len(graph.parameters())}")
    print(f"   Embedding dim: {args.embedding_dim}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return graph, state, word_to_idx


if __name__ == "__main__":
    main()
