#!/usr/bin/env python3
"""
Training and Evaluation Benchmark Harness for TrainableGraph Language Model.

Collects metrics over time for monitoring model performance:
- Training loss and perplexity
- Validation loss and perplexity
- Generation quality samples
- Timing benchmarks
- Character-level accuracy

Usage:
    # Run a training benchmark
    PYTHONPATH=. python examples/trainable_graph_benchmark.py --mode train --epochs 20

    # Evaluate existing checkpoint
    PYTHONPATH=. python examples/trainable_graph_benchmark.py --mode eval --checkpoint model.pkl

    # View benchmark history
    PYTHONPATH=. python examples/trainable_graph_benchmark.py --mode report

    # Continuous training with periodic benchmarks
    PYTHONPATH=. python examples/trainable_graph_benchmark.py --mode continuous --epochs 100
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import pickle

import numpy as np

from cortical.graph import TrainableGraph, Adam, SGD, Activation, Aggregation, Parameter


# =============================================================================
# Benchmark Data Structures
# =============================================================================


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run."""

    # Identification
    run_id: str
    timestamp: str
    mode: str  # 'train', 'eval', 'continuous'

    # Model configuration
    context_size: int
    hidden_dim: int
    num_layers: int
    vocab_size: int
    total_parameters: int

    # Training metrics
    epoch: int
    train_loss: float
    train_perplexity: float

    # Validation metrics
    val_loss: float
    val_perplexity: float

    # Timing
    train_time_sec: float
    eval_time_sec: float
    samples_per_sec: float

    # Generation samples
    generation_samples: List[Dict[str, str]]

    # Prediction accuracy
    top1_accuracy: float
    top5_accuracy: float

    # Additional info
    corpus_size: int
    train_sequences: int
    val_sequences: int
    checkpoint_path: Optional[str] = None
    notes: Optional[str] = None

    # Word quality metrics (for overfit experiments)
    word_quality_rate: Optional[float] = None
    real_words_found: Optional[List[str]] = None


class BenchmarkLogger:
    """Logs benchmark results to a JSON file."""

    def __init__(self, log_path: str = "benchmarks/trainable_graph_benchmarks.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load existing benchmark history."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {"runs": [], "metadata": {"created": datetime.now().isoformat()}}

    def log(self, metrics: BenchmarkMetrics):
        """Log a benchmark run."""
        self.history["runs"].append(asdict(metrics))
        self.history["metadata"]["last_updated"] = datetime.now().isoformat()
        self.history["metadata"]["total_runs"] = len(self.history["runs"])

        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Benchmark logged to {self.log_path}")

    def get_history(self) -> List[Dict]:
        """Get all benchmark runs."""
        return self.history["runs"]

    def get_latest(self, n: int = 5) -> List[Dict]:
        """Get latest N benchmark runs."""
        return self.history["runs"][-n:]

    def summary(self) -> str:
        """Generate a summary report of benchmark history."""
        if not self.history["runs"]:
            return "No benchmark runs recorded yet."

        lines = []
        lines.append("=" * 70)
        lines.append("TRAINABLE GRAPH BENCHMARK HISTORY")
        lines.append("=" * 70)
        lines.append(f"Total runs: {len(self.history['runs'])}")
        lines.append(f"Log file: {self.log_path}")
        lines.append("")

        # Recent runs table
        lines.append("Recent Runs:")
        lines.append("-" * 70)
        lines.append(f"{'Timestamp':<20} {'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} {'PPL':>8} {'Acc':>6}")
        lines.append("-" * 70)

        for run in self.history["runs"][-10:]:
            ts = run["timestamp"][:16]
            lines.append(
                f"{ts:<20} {run['epoch']:>6} {run['train_loss']:>11.4f} "
                f"{run['val_loss']:>10.4f} {run['val_perplexity']:>8.2f} {run['top1_accuracy']*100:>5.1f}%"
            )

        lines.append("-" * 70)

        # Best runs
        if len(self.history["runs"]) > 1:
            best_loss = min(self.history["runs"], key=lambda x: x["val_loss"])
            best_acc = max(self.history["runs"], key=lambda x: x["top1_accuracy"])

            lines.append("")
            lines.append("Best Results:")
            lines.append(f"  Lowest Val Loss: {best_loss['val_loss']:.4f} (run {best_loss['run_id']})")
            lines.append(f"  Best Accuracy:   {best_acc['top1_accuracy']*100:.1f}% (run {best_acc['run_id']})")

        # Performance trend
        if len(self.history["runs"]) >= 3:
            recent = self.history["runs"][-3:]
            avg_speed = np.mean([r["samples_per_sec"] for r in recent])
            lines.append(f"  Avg Speed:       {avg_speed:.0f} samples/sec (last 3 runs)")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Character Vocabulary (copied from language model)
# =============================================================================


class CharVocab:
    """Character vocabulary with encoding/decoding."""

    def __init__(self, chars: str = None):
        if chars is None:
            chars = "abcdefghijklmnopqrstuvwxyz .,;:!?'-\n"

        self.chars = list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.size = len(self.chars)
        self.unk_idx = self.size

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.unk_idx) for c in text.lower()]

    def decode(self, indices: List[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in indices)

    def one_hot(self, idx: int) -> np.ndarray:
        vec = np.zeros(self.size, dtype=np.float64)
        if 0 <= idx < self.size:
            vec[idx] = 1.0
        return vec

    def from_probs(self, probs: np.ndarray, temperature: float = 1.0) -> int:
        if temperature <= 0:
            return int(np.argmax(probs[:self.size]))
        logits = np.log(probs[:self.size] + 1e-10) / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return int(np.random.choice(self.size, p=probs))


# =============================================================================
# Language Model (copied from language model example)
# =============================================================================


class GraphLanguageModel:
    """Character-level language model using TrainableGraph."""

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

        self.char_embeddings = Parameter(
            data=np.random.randn(vocab.size, hidden_dim) * 0.1,
            name="char_embeddings",
        )
        self.pos_embeddings = Parameter(
            data=np.random.randn(context_size, hidden_dim) * 0.1,
            name="pos_embeddings",
        )
        self.output_proj = Parameter(
            data=np.random.randn(hidden_dim, vocab.size) * 0.1,
            name="output_proj",
        )
        self.output_bias = Parameter(
            data=np.zeros(vocab.size),
            name="output_bias",
        )

        self._build_graph()

    def _build_graph(self):
        self.graph = TrainableGraph(
            embedding_dim=self.hidden_dim,
            activation=Activation.TANH,
            aggregation=Aggregation.SUM,
            seed=42,
        )

        for i in range(self.context_size):
            self.graph.add_node(f"pos_{i}", embedding=np.zeros(self.hidden_dim))

        for i in range(self.context_size - 1):
            self.graph.add_edge(f"pos_{i}", f"pos_{i+1}", weight=0.8)

        for i in range(self.context_size - 2):
            self.graph.add_edge(f"pos_{i}", f"pos_{i+2}", weight=0.4)

    def parameters(self) -> List[Parameter]:
        return [
            self.char_embeddings,
            self.pos_embeddings,
            self.output_proj,
            self.output_bias,
        ] + self.graph.parameters()

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def forward(self, char_indices: List[int]) -> np.ndarray:
        if len(char_indices) < self.context_size:
            padding = [0] * (self.context_size - len(char_indices))
            char_indices = padding + char_indices
        elif len(char_indices) > self.context_size:
            char_indices = char_indices[-self.context_size:]

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

        self.graph.train()
        outputs = self.graph.forward(num_layers=self.num_layers)
        final_hidden = outputs[f"pos_{self.context_size - 1}"]
        logits = final_hidden @ self.output_proj.data + self.output_bias.data
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def compute_loss(self, char_indices: List[int], target_idx: int) -> Tuple[float, np.ndarray]:
        probs = self.forward(char_indices)
        target_prob = probs[target_idx] + 1e-10
        loss = -np.log(target_prob)
        grad = probs.copy()
        grad[target_idx] -= 1.0
        return float(loss), grad

    def backward(self, output_grad: np.ndarray):
        final_node = f"pos_{self.context_size - 1}"
        outputs = self.graph.forward(num_layers=self.num_layers)
        final_hidden = outputs[final_node]
        self.output_proj.add_grad(np.outer(final_hidden, output_grad))
        self.output_bias.add_grad(output_grad)
        hidden_grad = output_grad @ self.output_proj.data.T
        self.graph.backward({final_node: hidden_grad}, num_layers=self.num_layers)

    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        indices = self.vocab.encode(prompt)
        generated = list(prompt.lower())

        for _ in range(max_length):
            context = indices[-self.context_size:] if len(indices) > self.context_size else indices
            probs = self.forward(context)
            next_idx = self.vocab.from_probs(probs, temperature)
            next_char = self.vocab.idx_to_char.get(next_idx, "?")
            generated.append(next_char)
            indices.append(next_idx)
            if next_char == "\n":
                break

        return "".join(generated)


# =============================================================================
# Benchmark Harness
# =============================================================================


class BenchmarkHarness:
    """Training and evaluation benchmark harness."""

    def __init__(
        self,
        context_size: int = 12,
        hidden_dim: int = 48,
        num_layers: int = 2,
        log_path: str = "benchmarks/trainable_graph_benchmarks.json",
    ):
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.logger = BenchmarkLogger(log_path)

        self.model: Optional[GraphLanguageModel] = None
        self.vocab: Optional[CharVocab] = None
        self.train_sequences: List[Tuple[List[int], int]] = []
        self.val_sequences: List[Tuple[List[int], int]] = []
        self.corpus_size = 0

    def load_corpus(
        self,
        samples_dir: str = "samples",
        max_docs: int = 100,
        max_sequences: int = 5000,
    ) -> str:
        """Load and prepare the training corpus."""
        samples_path = Path(samples_dir)
        texts = []

        for i, txt_file in enumerate(samples_path.glob("**/*.txt")):
            if i >= max_docs:
                break
            try:
                texts.append(txt_file.read_text(encoding="utf-8")[:3000])
            except Exception:
                pass

        corpus = "\n\n".join(texts)
        self.corpus_size = len(corpus)

        # Build vocabulary
        char_counts = Counter(corpus.lower())
        common_chars = "".join(c for c, _ in char_counts.most_common(45) if c.isprintable())
        self.vocab = CharVocab(common_chars)

        # Create sequences
        indices = self.vocab.encode(corpus)
        all_sequences = []

        for i in range(len(indices) - self.context_size):
            context = indices[i:i + self.context_size]
            target = indices[i + self.context_size]
            if target < self.vocab.size:
                all_sequences.append((context, target))
            if len(all_sequences) >= max_sequences * 2:  # Get enough for train/val split
                break

        # Split train/val (90/10)
        random.shuffle(all_sequences)
        split_idx = min(max_sequences, int(len(all_sequences) * 0.9))
        val_size = min(max_sequences // 5, len(all_sequences) - split_idx)
        self.train_sequences = all_sequences[:split_idx]
        self.val_sequences = all_sequences[split_idx:split_idx + val_size]

        print(f"Corpus: {len(texts)} docs, {len(corpus)} chars")
        print(f"Vocab: {self.vocab.size} chars")
        print(f"Sequences: {len(self.train_sequences)} train, {len(self.val_sequences)} val")

        return corpus

    def create_model(self) -> GraphLanguageModel:
        """Create a new model instance."""
        self.model = GraphLanguageModel(
            vocab=self.vocab,
            context_size=self.context_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        return self.model

    def load_checkpoint(self, checkpoint_path: str) -> GraphLanguageModel:
        """Load model from checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.model = GraphLanguageModel(
            vocab=self.vocab,
            context_size=self.context_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.model.graph.load_checkpoint(checkpoint)
        return self.model

    def train_epoch(
        self,
        optimizer,
        batch_size: int = 32,
    ) -> Tuple[float, float]:
        """Train for one epoch, return loss and time."""
        random.shuffle(self.train_sequences)

        start_time = time.time()
        total_loss = 0.0

        for i in range(0, len(self.train_sequences), batch_size):
            batch = self.train_sequences[i:i + batch_size]
            self.model.zero_grad()

            for context, target in batch:
                loss, grad = self.model.compute_loss(context, target)
                self.model.backward(grad)
                total_loss += loss

            # Gradient clipping
            total_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_norm += np.sum(param.grad ** 2)
            total_norm = np.sqrt(total_norm)

            if total_norm > 5.0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad *= 5.0 / total_norm

            optimizer.step()

        elapsed = time.time() - start_time
        avg_loss = total_loss / len(self.train_sequences)

        return avg_loss, elapsed

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        start_time = time.time()

        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0

        for context, target in self.val_sequences:
            probs = self.model.forward(context)

            # Loss
            target_prob = probs[target] + 1e-10
            total_loss += -np.log(target_prob)

            # Accuracy
            top5_indices = np.argsort(probs)[-5:]
            if target == np.argmax(probs):
                correct_top1 += 1
            if target in top5_indices:
                correct_top5 += 1

        elapsed = time.time() - start_time
        n = len(self.val_sequences)
        avg_loss = total_loss / n

        return {
            "loss": avg_loss,
            "perplexity": np.exp(avg_loss),
            "top1_accuracy": correct_top1 / n,
            "top5_accuracy": correct_top5 / n,
            "eval_time": elapsed,
        }

    def generate_samples(self, prompts: List[str], temperature: float = 0.7) -> List[Dict[str, str]]:
        """Generate text samples for quality evaluation."""
        samples = []
        for prompt in prompts:
            output = self.model.generate(prompt, max_length=50, temperature=temperature)
            samples.append({
                "prompt": prompt,
                "output": output,
                "temperature": temperature,
            })
        return samples

    def extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        return re.findall(r'[a-z]+', text.lower())

    def evaluate_word_quality(
        self,
        reference_words: Set[str],
        prompts: List[str],
        temperature: float = 0.5,
    ) -> Tuple[float, List[str]]:
        """
        Evaluate how many real words appear in generated text.

        Returns:
            Tuple of (word_quality_rate, list of real words found)
        """
        all_real_words = []
        total_words = 0

        for prompt in prompts:
            output = self.model.generate(prompt, max_length=50, temperature=temperature)
            gen_words = self.extract_words(output)
            total_words += len(gen_words)

            for word in gen_words:
                if word in reference_words and len(word) > 2:
                    all_real_words.append(word)

        word_rate = len(all_real_words) / max(total_words, 1)
        unique_words = list(dict.fromkeys(all_real_words))[:20]  # Dedupe, keep order
        return word_rate, unique_words

    def run_benchmark(
        self,
        mode: str = "train",
        epochs: int = 20,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        checkpoint_path: Optional[str] = None,
        save_checkpoint: bool = True,
        notes: Optional[str] = None,
        log_every: int = 10,
    ) -> BenchmarkMetrics:
        """Run a complete benchmark."""

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*60}")
        print(f"BENCHMARK RUN: {run_id}")
        print(f"Mode: {mode}, Epochs: {epochs}")
        print(f"{'='*60}\n")

        # Load or create model
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        else:
            print("Creating new model...")
            self.create_model()

        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        # Training
        train_losses = []
        total_train_time = 0.0
        word_quality_rate = None
        real_words_found = None

        # Build reference words for word quality evaluation
        reference_words: Set[str] = set()
        if mode == "overfit":
            # For overfit mode, get words from training corpus
            corpus_text = self.vocab.decode([
                self.vocab.char_to_idx.get(c, 0)
                for seq, _ in self.train_sequences[:100]
                for c in self.vocab.decode(seq)
            ])
            reference_words = set(self.extract_words(corpus_text))
            # Also add common English words for reference
            reference_words.update(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'for', 'that',
                                    'on', 'are', 'as', 'with', 'be', 'at', 'this', 'have', 'from'])

        if mode in ("train", "continuous", "overfit"):
            print(f"\nTraining for {epochs} epochs...")
            if mode == "overfit":
                print(f"Reference vocabulary: {len(reference_words)} words")
                print(f"Logging every {log_every} epochs\n")

            prompts = ["the ", "and ", "to ", "of ", "a "]

            for epoch in range(epochs):
                loss, elapsed = self.train_epoch(optimizer, batch_size)
                train_losses.append(loss)
                total_train_time += elapsed

                # Standard logging every 5 epochs
                if mode != "overfit" and (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}: loss={loss:.4f}, time={elapsed:.1f}s")

                # Overfit mode: detailed logging with word quality
                if mode == "overfit" and ((epoch + 1) % log_every == 0 or epoch < 5):
                    word_rate, words = self.evaluate_word_quality(reference_words, prompts)
                    word_quality_rate = word_rate
                    real_words_found = words

                    print(f"Epoch {epoch+1:4d}: loss={loss:.4f}, ppl={np.exp(loss):.1f}, words={word_rate*100:.0f}%")

                    # Show sample generation
                    sample = self.model.generate("the ", max_length=50, temperature=0.5)
                    print(f"  Sample: \"{sample}\"")
                    if words:
                        print(f"  Real words: {words[:10]}")
                    print()

        # Evaluation
        print("\nEvaluating...")
        eval_results = self.evaluate()

        # Generation samples
        prompts = ["the ", "model ", "learning ", "data ", "system "]
        samples = self.generate_samples(prompts)

        print("\nGeneration Samples:")
        for s in samples[:3]:
            print(f"  '{s['prompt']}' -> '{s['output'][:40]}...'")

        # Calculate metrics
        final_train_loss = train_losses[-1] if train_losses else 0.0
        samples_per_sec = (len(self.train_sequences) * epochs) / total_train_time if total_train_time > 0 else 0

        metrics = BenchmarkMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            mode=mode,
            context_size=self.context_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            vocab_size=self.vocab.size,
            total_parameters=sum(p.data.size for p in self.model.parameters()),
            epoch=epochs,
            train_loss=final_train_loss,
            train_perplexity=np.exp(final_train_loss) if final_train_loss > 0 else 0,
            val_loss=eval_results["loss"],
            val_perplexity=eval_results["perplexity"],
            train_time_sec=total_train_time,
            eval_time_sec=eval_results["eval_time"],
            samples_per_sec=samples_per_sec,
            generation_samples=samples,
            top1_accuracy=eval_results["top1_accuracy"],
            top5_accuracy=eval_results["top5_accuracy"],
            corpus_size=self.corpus_size,
            train_sequences=len(self.train_sequences),
            val_sequences=len(self.val_sequences),
            checkpoint_path=checkpoint_path,
            notes=notes,
            word_quality_rate=word_quality_rate,
            real_words_found=real_words_found,
        )

        # Log results
        self.logger.log(metrics)

        # Save checkpoint
        if save_checkpoint:
            ckpt_path = f"benchmarks/checkpoint_{run_id}.pkl"
            Path("benchmarks").mkdir(exist_ok=True)
            checkpoint = self.model.graph.save_checkpoint(
                optimizer=optimizer,
                epoch=epochs,
                loss=final_train_loss,
            )
            with open(ckpt_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"\nCheckpoint saved: {ckpt_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  Train Loss:      {final_train_loss:.4f}")
        print(f"  Val Loss:        {eval_results['loss']:.4f}")
        print(f"  Val Perplexity:  {eval_results['perplexity']:.2f}")
        print(f"  Top-1 Accuracy:  {eval_results['top1_accuracy']*100:.1f}%")
        print(f"  Top-5 Accuracy:  {eval_results['top5_accuracy']*100:.1f}%")
        print(f"  Train Time:      {total_train_time:.1f}s")
        print(f"  Throughput:      {samples_per_sec:.0f} samples/sec")
        if word_quality_rate is not None:
            print(f"  Word Quality:    {word_quality_rate*100:.1f}%")
            if real_words_found:
                print(f"  Words Found:     {real_words_found[:10]}")
        print(f"{'='*60}\n")

        return metrics


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="TrainableGraph Benchmark Harness")
    parser.add_argument("--mode", choices=["train", "eval", "continuous", "report", "overfit"],
                        default="train", help="Benchmark mode (overfit: single-doc word learning test)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--context-size", type=int, default=12, help="Context window size")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load/save")
    parser.add_argument("--samples-dir", type=str, default="samples", help="Training corpus directory")
    parser.add_argument("--max-docs", type=int, default=100, help="Max documents to load")
    parser.add_argument("--max-sequences", type=int, default=5000, help="Max training sequences")
    parser.add_argument("--log-path", type=str, default="benchmarks/trainable_graph_benchmarks.json",
                        help="Benchmark log file")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N epochs (overfit mode)")
    parser.add_argument("--notes", type=str, default=None, help="Notes for this run")

    args = parser.parse_args()

    # Report mode - just show history
    if args.mode == "report":
        logger = BenchmarkLogger(args.log_path)
        print(logger.summary())
        return

    # Create harness
    harness = BenchmarkHarness(
        context_size=args.context_size,
        hidden_dim=args.hidden_dim,
        log_path=args.log_path,
    )

    # Load corpus
    print("Loading corpus...")
    harness.load_corpus(args.samples_dir, args.max_docs, args.max_sequences)

    # Run benchmark
    harness.run_benchmark(
        mode=args.mode,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        notes=args.notes,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
