#!/usr/bin/env python3
"""
Graph-based Language Model using TrainableGraph.

A character-level language model that learns to predict the next character
given a context window. Can generate text autoregressively.

Architecture:
- Each character position in the context window is a node
- Edges connect sequential positions (position i -> position i+1)
- Message passing aggregates context information
- Final node predicts next character distribution

Usage:
    PYTHONPATH=. python examples/trainable_graph_language_model.py

    # Generate text after training:
    PYTHONPATH=. python examples/trainable_graph_language_model.py --generate "machine learning"
"""

import argparse
import re
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from cortical.graph import (
    TrainableGraph,
    Adam,
    SGD,
    Activation,
    Aggregation,
    Parameter,
)


# =============================================================================
# Character Vocabulary
# =============================================================================


class CharVocab:
    """Character vocabulary with encoding/decoding."""

    def __init__(self, chars: str = None):
        if chars is None:
            # Default: lowercase + space + punctuation
            chars = "abcdefghijklmnopqrstuvwxyz .,;:!?'-\n"

        self.chars = list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.size = len(self.chars)

        # Special tokens
        self.unk_idx = self.size  # Unknown character

    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        return [self.char_to_idx.get(c, self.unk_idx) for c in text.lower()]

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        return "".join(self.idx_to_char.get(i, "?") for i in indices)

    def one_hot(self, idx: int) -> np.ndarray:
        """Create one-hot vector for character."""
        vec = np.zeros(self.size, dtype=np.float64)
        if 0 <= idx < self.size:
            vec[idx] = 1.0
        return vec

    def from_probs(self, probs: np.ndarray, temperature: float = 1.0) -> int:
        """Sample character index from probability distribution."""
        if temperature <= 0:
            return int(np.argmax(probs[:self.size]))

        # Apply temperature
        logits = np.log(probs[:self.size] + 1e-10) / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return int(np.random.choice(self.size, p=probs))


# =============================================================================
# Language Model Graph
# =============================================================================


class GraphLanguageModel:
    """
    Character-level language model using TrainableGraph.

    Architecture:
    - Context window of N positions
    - Each position is a graph node
    - Sequential edges: pos_0 -> pos_1 -> ... -> pos_N
    - Character embedding at each position
    - Output layer predicts next character
    """

    def __init__(
        self,
        vocab: CharVocab,
        context_size: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        self.vocab = vocab
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character embeddings (learnable)
        self.char_embeddings = Parameter(
            data=np.random.randn(vocab.size, hidden_dim) * 0.1,
            name="char_embeddings",
        )

        # Position embeddings (learnable)
        self.pos_embeddings = Parameter(
            data=np.random.randn(context_size, hidden_dim) * 0.1,
            name="pos_embeddings",
        )

        # Output projection: hidden_dim -> vocab_size
        self.output_proj = Parameter(
            data=np.random.randn(hidden_dim, vocab.size) * 0.1,
            name="output_proj",
        )
        self.output_bias = Parameter(
            data=np.zeros(vocab.size),
            name="output_bias",
        )

        # Build the graph structure
        self._build_graph()

    def _build_graph(self):
        """Build the context window graph."""
        self.graph = TrainableGraph(
            embedding_dim=self.hidden_dim,
            activation=Activation.TANH,
            aggregation=Aggregation.SUM,
            seed=42,
        )

        # Add position nodes
        for i in range(self.context_size):
            self.graph.add_node(
                f"pos_{i}",
                embedding=np.zeros(self.hidden_dim),
            )

        # Add sequential edges (causal: only look at previous positions)
        for i in range(self.context_size - 1):
            self.graph.add_edge(f"pos_{i}", f"pos_{i+1}", weight=0.8)

        # Add skip connections (for longer range dependencies)
        for i in range(self.context_size - 2):
            self.graph.add_edge(f"pos_{i}", f"pos_{i+2}", weight=0.4)

    def parameters(self) -> List[Parameter]:
        """Get all learnable parameters."""
        return [
            self.char_embeddings,
            self.pos_embeddings,
            self.output_proj,
            self.output_bias,
        ] + self.graph.parameters()

    def zero_grad(self):
        """Reset all gradients."""
        for param in self.parameters():
            param.zero_grad()

    def forward(self, char_indices: List[int]) -> np.ndarray:
        """
        Forward pass: predict next character distribution.

        Args:
            char_indices: List of character indices (context window)

        Returns:
            Probability distribution over next character
        """
        # Pad or truncate to context size
        if len(char_indices) < self.context_size:
            # Pad with zeros at the beginning
            padding = [0] * (self.context_size - len(char_indices))
            char_indices = padding + char_indices
        elif len(char_indices) > self.context_size:
            # Take last context_size characters
            char_indices = char_indices[-self.context_size:]

        # Set node embeddings: char_embedding + pos_embedding
        for i, char_idx in enumerate(char_indices):
            if 0 <= char_idx < self.vocab.size:
                char_emb = self.char_embeddings.data[char_idx]
            else:
                char_emb = np.zeros(self.hidden_dim)

            pos_emb = self.pos_embeddings.data[i]
            combined = char_emb + pos_emb

            node = self.graph.get_node(f"pos_{i}")
            if node and node.embedding:
                node.embedding.data = combined

        # Forward through graph
        self.graph.train()
        outputs = self.graph.forward(num_layers=self.num_layers)

        # Get final position output
        final_hidden = outputs[f"pos_{self.context_size - 1}"]

        # Project to vocabulary
        logits = final_hidden @ self.output_proj.data + self.output_bias.data

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def compute_loss(
        self,
        char_indices: List[int],
        target_idx: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss and gradient.

        Args:
            char_indices: Context character indices
            target_idx: Target character index

        Returns:
            Tuple of (loss, gradient w.r.t. logits)
        """
        probs = self.forward(char_indices)

        # Cross-entropy loss
        target_prob = probs[target_idx] + 1e-10
        loss = -np.log(target_prob)

        # Gradient of cross-entropy w.r.t. logits (softmax + CE)
        grad = probs.copy()
        grad[target_idx] -= 1.0

        return float(loss), grad

    def backward(self, output_grad: np.ndarray):
        """
        Backward pass through the model.

        Args:
            output_grad: Gradient w.r.t. output logits
        """
        # Gradient through output projection
        final_node = f"pos_{self.context_size - 1}"
        outputs = self.graph.forward(num_layers=self.num_layers)
        final_hidden = outputs[final_node]

        # d(loss)/d(output_proj) = outer(final_hidden, output_grad)
        self.output_proj.add_grad(np.outer(final_hidden, output_grad))
        self.output_bias.add_grad(output_grad)

        # d(loss)/d(final_hidden) = output_grad @ output_proj.T
        hidden_grad = output_grad @ self.output_proj.data.T

        # Backward through graph
        self.graph.backward({final_node: hidden_grad}, num_layers=self.num_layers)

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate text starting from a prompt.

        Args:
            prompt: Starting text
            max_length: Maximum characters to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text
        """
        # Encode prompt
        indices = self.vocab.encode(prompt)

        generated = list(prompt.lower())

        for _ in range(max_length):
            # Get context window
            context = indices[-self.context_size:] if len(indices) > self.context_size else indices

            # Predict next character
            probs = self.forward(context)

            # Sample
            next_idx = self.vocab.from_probs(probs, temperature)
            next_char = self.vocab.idx_to_char.get(next_idx, "?")

            generated.append(next_char)
            indices.append(next_idx)

            # Stop at newline or max length
            if next_char == "\n":
                break

        return "".join(generated)


# =============================================================================
# Training
# =============================================================================


def create_training_sequences(
    text: str,
    vocab: CharVocab,
    context_size: int,
    max_sequences: int = 10000,
) -> List[Tuple[List[int], int]]:
    """
    Create training sequences from text.

    Returns list of (context_indices, target_idx) tuples.
    """
    indices = vocab.encode(text)
    sequences = []

    for i in range(len(indices) - context_size):
        context = indices[i:i + context_size]
        target = indices[i + context_size]

        if target < vocab.size:  # Valid target
            sequences.append((context, target))

        if len(sequences) >= max_sequences:
            break

    return sequences


def train_language_model(
    model: GraphLanguageModel,
    text: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    checkpoint_path: Optional[str] = None,
    save_every: int = 10,
) -> Tuple[List[float], Optional[dict]]:
    """
    Train the language model with checkpoint support.

    Args:
        model: The language model to train
        text: Training text corpus
        epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        checkpoint_path: Path to save/load checkpoints (None to disable)
        save_every: Save checkpoint every N epochs

    Returns:
        Tuple of (loss history, final checkpoint dict)
    """
    # Create training data
    sequences = create_training_sequences(
        text,
        model.vocab,
        model.context_size,
        max_sequences=5000,
    )
    print(f"  Training sequences: {len(sequences)}")

    optimizer = Adam(model.parameters(), lr=learning_rate)
    losses = []
    start_epoch = 0

    # Try to load checkpoint if path provided
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            import pickle
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            resume_info = model.graph.load_checkpoint(checkpoint, optimizer)
            start_epoch = resume_info["epoch"]
            losses = resume_info.get("extra", {}).get("losses", [])
            print(f"  Resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        random.shuffle(sequences)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_loss = 0.0

            model.zero_grad()

            for context, target in batch:
                loss, grad = model.compute_loss(context, target)
                model.backward(grad)
                batch_loss += loss

            # Clip gradients
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += np.sum(param.grad ** 2)
            total_norm = np.sqrt(total_norm)

            if total_norm > 5.0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad *= 5.0 / total_norm

            # Update
            optimizer.step()

            epoch_loss += batch_loss
            num_batches += 1

        avg_loss = epoch_loss / len(sequences)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Save checkpoint periodically
        if checkpoint_path and (epoch + 1) % save_every == 0:
            checkpoint = model.graph.save_checkpoint(
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                extra={"losses": losses},
            )
            import pickle
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"  Checkpoint saved at epoch {epoch + 1}")

    # Final checkpoint
    final_checkpoint = None
    if checkpoint_path:
        final_checkpoint = model.graph.save_checkpoint(
            optimizer=optimizer,
            epoch=epochs,
            loss=losses[-1] if losses else 0.0,
            extra={"losses": losses},
        )
        import pickle
        with open(checkpoint_path, "wb") as f:
            pickle.dump(final_checkpoint, f)

    return losses, final_checkpoint


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Graph-based language model")
    parser.add_argument("--samples-dir", default="samples", help="Samples directory")
    parser.add_argument("--context-size", type=int, default=12, help="Context window size")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.008, help="Learning rate")
    parser.add_argument("--generate", type=str, default=None, help="Generate from prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max-docs", type=int, default=30, help="Max documents to load")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file for save/resume")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    print("=" * 60)
    print("Graph-based Character Language Model")
    print("=" * 60)

    # Load corpus
    print(f"\n1. Loading corpus from {args.samples_dir}/...")
    samples_path = Path(args.samples_dir)
    texts = []

    for i, txt_file in enumerate(samples_path.glob("**/*.txt")):
        if i >= args.max_docs:
            break
        try:
            texts.append(txt_file.read_text(encoding="utf-8")[:2000])
        except Exception:
            pass

    corpus = "\n\n".join(texts)
    print(f"   Loaded {len(texts)} documents")
    print(f"   Total characters: {len(corpus)}")

    # Build vocabulary from corpus
    print("\n2. Building vocabulary...")
    char_counts = Counter(corpus.lower())
    common_chars = "".join(c for c, _ in char_counts.most_common(40) if c.isprintable())
    vocab = CharVocab(common_chars)
    print(f"   Vocabulary size: {vocab.size}")
    print(f"   Characters: {repr(common_chars[:30])}...")

    # Create model
    print(f"\n3. Creating model (context={args.context_size}, hidden={args.hidden_dim})...")
    model = GraphLanguageModel(
        vocab=vocab,
        context_size=args.context_size,
        hidden_dim=args.hidden_dim,
        num_layers=2,
    )
    print(f"   Parameters: {sum(p.data.size for p in model.parameters())}")

    # Train
    print(f"\n4. Training (epochs={args.epochs}, lr={args.lr})...")
    if args.checkpoint:
        print(f"   Checkpoint: {args.checkpoint}")
    losses, checkpoint = train_language_model(
        model,
        corpus,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
    )
    print(f"   Final loss: {losses[-1]:.4f}")

    # Generate samples
    print("\n5. Generating text samples...")
    print("=" * 60)

    prompts = ["the ", "machine ", "learning ", "data ", "system "]

    if args.generate:
        prompts = [args.generate]

    for prompt in prompts:
        generated = model.generate(
            prompt,
            max_length=60,
            temperature=args.temperature,
        )
        print(f"\n   Prompt: '{prompt}'")
        print(f"   Output: '{generated}'")

    # Interactive generation
    print("\n" + "=" * 60)
    print("\n6. Sample completions at different temperatures:")

    test_prompt = "the model "
    for temp in [0.3, 0.7, 1.0]:
        output = model.generate(test_prompt, max_length=40, temperature=temp)
        print(f"   T={temp}: '{output}'")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    main()
