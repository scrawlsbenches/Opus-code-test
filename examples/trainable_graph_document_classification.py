#!/usr/bin/env python3
"""
Document Classification using TrainableGraph.

This example demonstrates how to use the TrainableGraph for document
classification using the samples/ directory as a dataset.

The workflow:
1. Load documents from samples/ directory (organized by category folders)
2. Extract bag-of-words embeddings for each document
3. Build a graph connecting similar documents
4. Train the graph to classify documents by category
5. Evaluate on held-out test documents

Usage:
    python examples/trainable_graph_document_classification.py

    # Or with custom categories:
    python examples/trainable_graph_document_classification.py --categories art_history religious_studies
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cortical.graph import (
    TrainableGraph,
    Adam,
    MSELoss,
    Activation,
    EarlyStopping,
)


# =============================================================================
# Document Loading
# =============================================================================


def load_documents(
    samples_dir: str,
    target_categories: List[str] = None,
    max_content_length: int = 3000,
) -> List[dict]:
    """
    Load documents from samples directory.

    Args:
        samples_dir: Path to samples directory
        target_categories: List of category folders to load (None = all)
        max_content_length: Maximum characters to read per document

    Returns:
        List of document dicts with 'id', 'category', 'content' keys
    """
    docs = []
    samples_path = Path(samples_dir)

    if target_categories:
        # Load only specified categories
        for category in target_categories:
            cat_path = samples_path / category
            if cat_path.exists():
                for txt_file in cat_path.glob("*.txt"):
                    try:
                        content = txt_file.read_text(encoding="utf-8")[
                            :max_content_length
                        ]
                        docs.append(
                            {
                                "id": txt_file.stem,
                                "category": category,
                                "content": content,
                            }
                        )
                    except Exception:
                        pass
    else:
        # Load all categories
        for txt_file in samples_path.glob("**/*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8")[:max_content_length]
                # Category is parent folder name
                if txt_file.parent.name == "samples":
                    category = "misc"
                else:
                    category = txt_file.parent.name
                docs.append(
                    {
                        "id": txt_file.stem,
                        "category": category,
                        "content": content,
                    }
                )
            except Exception:
                pass

    return docs


# =============================================================================
# Text Processing
# =============================================================================

# Common English stop words to exclude
STOP_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "are",
    "this",
    "from",
    "have",
    "has",
    "was",
    "were",
    "been",
    "being",
    "which",
    "their",
    "can",
    "more",
    "into",
    "such",
    "when",
    "also",
    "than",
    "these",
    "other",
    "may",
    "its",
    "but",
    "not",
    "they",
    "would",
    "could",
    "about",
    "through",
    "between",
    "while",
    "each",
    "most",
    "some",
    "will",
    "only",
    "over",
    "where",
    "after",
    "before",
    "just",
    "your",
    "what",
    "there",
}


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization: extract lowercase words of 4-15 characters.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    text = text.lower()
    return re.findall(r"\b[a-z]{4,15}\b", text)


def build_vocabulary(docs: List[dict], vocab_size: int = 100) -> Dict[str, int]:
    """
    Build vocabulary from most frequent words across documents.

    Args:
        docs: List of document dicts
        vocab_size: Maximum vocabulary size

    Returns:
        Dict mapping word to index
    """
    word_counts = Counter()

    for doc in docs:
        for word in tokenize(doc["content"]):
            if word not in STOP_WORDS and len(word) > 3:
                word_counts[word] += 1

    return {word: idx for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}


def doc_to_embedding(doc: dict, vocab: Dict[str, int]) -> np.ndarray:
    """
    Convert document to embedding vector using TF-like weighting.

    Args:
        doc: Document dict with 'content' key
        vocab: Vocabulary mapping word to index

    Returns:
        Normalized embedding vector
    """
    embedding = np.zeros(len(vocab), dtype=np.float64)
    words = tokenize(doc["content"])
    word_counts = Counter(words)

    for word, count in word_counts.items():
        if word in vocab:
            # Log-scaled term frequency
            embedding[vocab[word]] = np.log1p(count)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================================
# Graph Building
# =============================================================================


def build_document_graph(
    docs: List[dict],
    embeddings: Dict[str, np.ndarray],
    vocab_size: int,
    similarity_threshold: float = 0.2,
) -> TrainableGraph:
    """
    Build a trainable graph from documents.

    Nodes are documents, edges connect similar documents.

    Args:
        docs: List of document dicts
        embeddings: Dict mapping doc_id to embedding
        vocab_size: Embedding dimension
        similarity_threshold: Minimum similarity to create edge

    Returns:
        TrainableGraph instance
    """
    graph = TrainableGraph(
        embedding_dim=vocab_size,
        activation=Activation.TANH,
        seed=42,
    )

    # Add document nodes
    for doc in docs:
        graph.add_node(doc["id"], embedding=embeddings[doc["id"]])

    # Connect similar documents
    doc_ids = [d["id"] for d in docs]
    edge_count = 0

    for i, id_i in enumerate(doc_ids):
        for j, id_j in enumerate(doc_ids):
            if i >= j:
                continue

            sim = cosine_similarity(embeddings[id_i], embeddings[id_j])

            if sim > similarity_threshold:
                # Bidirectional edges with similarity as weight
                weight = min(sim, 0.95)  # Cap weight
                graph.add_edge(id_i, id_j, weight=weight)
                graph.add_edge(id_j, id_i, weight=weight)
                edge_count += 1

    print(f"  Created graph with {graph.node_count} nodes and {edge_count} edge pairs")
    return graph


# =============================================================================
# Training
# =============================================================================


def train_document_classifier(
    graph: TrainableGraph,
    train_docs: List[dict],
    categories: List[str],
    vocab_size: int,
    epochs: int = 100,
    learning_rate: float = 0.03,
    patience: int = 20,
) -> Tuple[Dict[str, int], List[float]]:
    """
    Train the graph for document classification.

    Args:
        graph: TrainableGraph instance
        train_docs: Training documents
        categories: List of category names
        vocab_size: Embedding dimension
        epochs: Maximum training epochs
        learning_rate: Adam learning rate
        patience: Early stopping patience

    Returns:
        Tuple of (category_to_index mapping, training losses)
    """
    # Create category index mapping
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # Create training targets (one-hot encoded categories)
    train_targets = {}
    for doc in train_docs:
        target = np.zeros(vocab_size, dtype=np.float64)
        target[cat_to_idx[doc["category"]]] = 1.0
        train_targets[doc["id"]] = target

    # Setup optimizer and loss
    optimizer = Adam(graph.parameters(), lr=learning_rate)
    loss_fn = MSELoss()
    early_stopping = EarlyStopping(patience=patience, restore_best=True)

    losses = []

    for epoch in range(epochs):
        graph.train()

        # Forward pass
        outputs = graph.forward(num_layers=2)

        # Compute loss and gradients
        total_loss = 0.0
        output_grads = {}

        for doc_id, target in train_targets.items():
            loss = loss_fn(outputs[doc_id], target)
            total_loss += loss
            output_grads[doc_id] = loss_fn.gradient(outputs[doc_id], target)

        losses.append(total_loss)

        # Backward pass
        graph.backward(output_grads, num_layers=2)
        graph.clip_gradients(1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Early stopping check
        if early_stopping(total_loss, graph):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        # Progress
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    return cat_to_idx, losses


def evaluate(
    graph: TrainableGraph,
    test_docs: List[dict],
    cat_to_idx: Dict[str, int],
) -> Tuple[float, Dict[str, dict]]:
    """
    Evaluate the trained graph on test documents.

    Args:
        graph: Trained TrainableGraph
        test_docs: Test documents
        cat_to_idx: Category to index mapping

    Returns:
        Tuple of (accuracy, per-category results)
    """
    idx_to_cat = {idx: cat for cat, idx in cat_to_idx.items()}
    num_categories = len(cat_to_idx)

    graph.eval()
    outputs = graph.forward(num_layers=2)

    correct = 0
    results_by_cat = {cat: {"correct": 0, "total": 0} for cat in cat_to_idx}

    print("\n  Predictions:")
    print("  " + "-" * 70)

    for doc in test_docs:
        # Predict category from output scores
        scores = outputs[doc["id"]][:num_categories]
        pred_cat = idx_to_cat[int(np.argmax(scores))]
        true_cat = doc["category"]

        is_correct = pred_cat == true_cat
        correct += int(is_correct)
        results_by_cat[true_cat]["total"] += 1
        results_by_cat[true_cat]["correct"] += int(is_correct)

        status = "OK" if is_correct else "X"
        score_str = " ".join(f"{s:+.2f}" for s in scores)
        print(
            f"  [{status}] {doc['id'][:30]:30} | {true_cat[:12]:12} -> {pred_cat[:12]:12} | [{score_str}]"
        )

    print("  " + "-" * 70)

    accuracy = correct / len(test_docs) if test_docs else 0.0
    return accuracy, results_by_cat


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Document classification using TrainableGraph"
    )
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Path to samples directory",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["art_history", "religious_studies", "ai_market_prediction", "hvac_engineering"],
        help="Categories to use (folder names in samples/)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=80,
        help="Vocabulary size for embeddings",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Learning rate",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.2,
        help="Minimum cosine similarity to create edge",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.75,
        help="Fraction of documents for training",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Document Classification with TrainableGraph")
    print("=" * 60)

    # Load documents
    print(f"\n1. Loading documents from {args.samples_dir}/...")
    print(f"   Target categories: {args.categories}")

    docs = load_documents(args.samples_dir, args.categories)
    print(f"   Loaded {len(docs)} documents")

    # Count per category
    cat_counts = Counter(doc["category"] for doc in docs)
    for cat, count in sorted(cat_counts.items()):
        print(f"     {cat}: {count} docs")

    if len(docs) < 10:
        print("\n   WARNING: Very few documents loaded. Results may be poor.")

    # Build vocabulary and embeddings
    print(f"\n2. Building vocabulary (size={args.vocab_size})...")
    vocab = build_vocabulary(docs, args.vocab_size)
    print(f"   Vocabulary size: {len(vocab)}")

    embeddings = {doc["id"]: doc_to_embedding(doc, vocab) for doc in docs}

    # Split train/test (stratified by category)
    print(f"\n3. Splitting data ({args.train_split:.0%} train)...")
    np.random.seed(42)
    train_docs, test_docs = [], []

    for category in args.categories:
        cat_docs = [d for d in docs if d["category"] == category]
        np.random.shuffle(cat_docs)
        split = max(1, int(len(cat_docs) * args.train_split))
        train_docs.extend(cat_docs[:split])
        test_docs.extend(cat_docs[split:])

    print(f"   Train: {len(train_docs)}, Test: {len(test_docs)}")

    # Build graph
    print(f"\n4. Building document graph (threshold={args.similarity_threshold})...")
    graph = build_document_graph(
        docs, embeddings, len(vocab), args.similarity_threshold
    )

    # Train
    print(f"\n5. Training (epochs={args.epochs}, lr={args.lr})...")
    cat_to_idx, losses = train_document_classifier(
        graph,
        train_docs,
        args.categories,
        len(vocab),
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    print(f"   Final loss: {losses[-1]:.4f}")

    # Evaluate
    print("\n6. Evaluating on test set...")
    accuracy, results_by_cat = evaluate(graph, test_docs, cat_to_idx)

    print(f"\n   Overall Accuracy: {accuracy:.1%}")
    print("\n   Per-category accuracy:")
    for cat in args.categories:
        r = results_by_cat[cat]
        if r["total"] > 0:
            acc = r["correct"] / r["total"]
            print(f"     {cat:25}: {r['correct']}/{r['total']} = {acc:.0%}")

    # Save trained model state
    print("\n7. Saving model state...")
    state = graph.save_state()
    print(f"   Model has {len(graph.parameters())} trainable parameters")
    print("   State can be restored with graph.load_state(state)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return graph, state


if __name__ == "__main__":
    main()
