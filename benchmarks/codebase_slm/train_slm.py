#!/usr/bin/env python3
"""
Train PRISM-SLM on the generated repository corpus.

This script demonstrates training a domain-specific SLM that understands
the repository's code, documentation, and structure.

Usage:
    # Quick training (sample corpus)
    python -m benchmarks.codebase_slm.train_slm --quick

    # Full training
    python -m benchmarks.codebase_slm.train_slm --full

    # Interactive mode after training
    python -m benchmarks.codebase_slm.train_slm --interactive
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortical.reasoning.prism_slm import PRISMLanguageModel


def load_training_corpus(corpus_path: Path, limit: int = None) -> List[str]:
    """Load training patterns from corpus file."""
    patterns = []

    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}")
        print("Run: python -m benchmarks.codebase_slm.generate_corpus --full")
        return patterns

    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line.strip())
                # Combine input and target for training
                text = f"{data['input_text']} {data['target_text']}"
                patterns.append(text)
            except json.JSONDecodeError:
                continue

    return patterns


def train_model(
    patterns: List[str],
    context_size: int = 3,
    progress_callback=None
) -> PRISMLanguageModel:
    """Train PRISM-SLM on patterns."""
    model = PRISMLanguageModel(context_size=context_size)

    for i, text in enumerate(patterns):
        model.train(text)
        if progress_callback and (i + 1) % 1000 == 0:
            progress_callback(i + 1, len(patterns))

    return model


def evaluate_model(model: PRISMLanguageModel, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate model on test queries."""
    results = {
        'queries': [],
        'avg_perplexity': 0.0,
        'completion_quality': 0.0,
    }

    total_ppl = 0.0
    for query in test_queries:
        prompt = query['prompt']
        expected = query.get('expected', '')

        # Generate
        generated = model.generate(prompt, max_tokens=15, temperature=0.5)

        # Calculate perplexity if expected is provided
        ppl = None
        if expected:
            ppl = model.perplexity(expected)
            total_ppl += ppl

        results['queries'].append({
            'prompt': prompt,
            'generated': generated,
            'expected': expected,
            'perplexity': ppl,
        })

    if test_queries:
        results['avg_perplexity'] = total_ppl / len(test_queries)

    return results


def interactive_mode(model: PRISMLanguageModel):
    """Interactive Q&A with trained model."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Ask questions about the codebase. Type 'quit' to exit.")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue

            # Generate response
            response = model.generate(prompt, max_tokens=30, temperature=0.7)
            print(f"SLM: {response}")
            print()

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Train PRISM-SLM on repository corpus')
    parser.add_argument('--quick', action='store_true', help='Quick training (1000 patterns)')
    parser.add_argument('--full', action='store_true', help='Full training (all patterns)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode after training')
    parser.add_argument('--corpus', type=str,
                        default='benchmarks/codebase_slm/corpus/training_patterns.jsonl',
                        help='Path to training corpus')
    parser.add_argument('--context-size', type=int, default=3, help='Context window size')
    args = parser.parse_args()

    corpus_path = Path(args.corpus)

    print("=" * 60)
    print("Repository-Native SLM Training")
    print("=" * 60)
    print()

    # Determine pattern limit
    limit = 1000 if args.quick else None

    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    start = time.time()
    patterns = load_training_corpus(corpus_path, limit=limit)

    if not patterns:
        print("No patterns loaded. Generate corpus first.")
        return 1

    print(f"Loaded {len(patterns)} patterns in {time.time() - start:.1f}s")

    # Train
    print(f"\nTraining PRISM-SLM (context_size={args.context_size})...")
    start = time.time()

    def progress(done, total):
        print(f"  Trained {done}/{total} patterns...", end='\r')

    model = train_model(patterns, context_size=args.context_size, progress_callback=progress)
    train_time = time.time() - start

    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"  Vocabulary size: {model.vocab_size}")
    print(f"  Transitions: {sum(len(t) for t in model.graph._transitions.values())}")

    # Test queries
    test_queries = [
        {'prompt': 'Where is PageRank', 'expected': 'cortical/analysis/pagerank.py'},
        {'prompt': 'What does compute_all', 'expected': 'computes all analysis phases'},
        {'prompt': 'from cortical.got import', 'expected': 'GoTManager'},
        {'prompt': 'How to create a task', 'expected': 'python scripts/got_utils.py task create'},
        {'prompt': 'What is PRISM', 'expected': 'Statistical Language Model'},
    ]

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for q in test_queries:
        generated = model.generate(q['prompt'], max_tokens=15, temperature=0.5)
        print(f"\nPrompt: {q['prompt']}")
        print(f"Generated: {generated}")
        if q.get('expected'):
            print(f"Expected: {q['expected']}")

    # Interactive mode
    if args.interactive:
        interactive_mode(model)

    return 0


if __name__ == '__main__':
    sys.exit(main())
