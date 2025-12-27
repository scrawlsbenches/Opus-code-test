#!/usr/bin/env python3
"""
Train PRISM-SLM on the generated repository corpus.

This script trains a domain-specific SLM that understands
the repository's code, documentation, and structure.

IMPORTANT: Use --dry-run to evaluate without saving, or --output to
specify a custom output path.

Usage:
    # Quick training (sample corpus) - evaluate only
    python -m benchmarks.codebase_slm.train_slm --quick --dry-run

    # Full training with model save
    python -m benchmarks.codebase_slm.train_slm --full --output prism_full.json

    # Interactive mode after training
    python -m benchmarks.codebase_slm.train_slm --quick --interactive
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.reasoning.prism_slm import PRISMLanguageModel

# Default paths
DEFAULT_MODEL_PATH = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "prism_slm.json"
BACKUP_DIR = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "backups"


def load_training_corpus(corpus_path: Path, limit: int = None) -> List[str]:
    """Load training patterns from corpus file.

    Uses Q: / A: format to create clear separation between
    questions and answers for n-gram learning.
    """
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
                pattern_type = data.get('pattern_type', 'unknown')

                # Format based on pattern type
                if pattern_type == 'qa':
                    # Q&A format for clear separation
                    text = f"Q: {data['input_text']} A: {data['target_text']}"
                elif pattern_type == 'completion':
                    # Code completion: input → target
                    text = f"{data['input_text']} → {data['target_text']}"
                elif pattern_type == 'association':
                    # Association: input relates to target
                    text = f"{data['input_text']} relates to {data['target_text']}"
                elif pattern_type == 'explanation':
                    # Explanation format
                    text = f"{data['input_text']}: {data['target_text']}"
                else:
                    # Default: simple concatenation
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


def backup_existing_model(model_path: Path) -> Optional[Path]:
    """Create a backup of the existing model before overwriting."""
    if not model_path.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{model_path.stem}_{timestamp}.json"
    backup_path = BACKUP_DIR / backup_name

    shutil.copy2(model_path, backup_path)
    print(f"⚠️  Backed up existing model to: {backup_path}")

    # Keep only last 5 backups per model type
    pattern = f"{model_path.stem}_*.json"
    backups = sorted(BACKUP_DIR.glob(pattern))
    if len(backups) > 5:
        for old_backup in backups[:-5]:
            old_backup.unlink()
            print(f"   Removed old backup: {old_backup.name}")

    return backup_path


def compute_corpus_hash(patterns: List[str]) -> str:
    """Compute a hash of the training corpus for provenance."""
    content = "\n".join(sorted(set(patterns)))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_model(model: PRISMLanguageModel, path: Path, provenance: dict):
    """Save trained model with provenance metadata."""
    # Get model internal state
    model_data = {
        # Provenance metadata
        '_provenance': {
            'trained_at': datetime.now().isoformat(),
            'corpus_hash': provenance.get('corpus_hash', 'unknown'),
            'corpus_size': provenance.get('corpus_size', 0),
            'corpus_path': provenance.get('corpus_path', 'unknown'),
            'script': 'train_slm.py',
            'model_type': 'PRISMLanguageModel',
            'context_size': model.context_size,
        },
        # Model data
        'vocab_size': model.vocab_size,
        'context_size': model.context_size,
        'transitions': {
            ' '.join(ctx): dict(trans)
            for ctx, trans in model.graph._transitions.items()
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"\n✓ Model saved to {path}")
    print(f"  Corpus hash: {provenance.get('corpus_hash', 'unknown')}")
    print(f"  Corpus size: {provenance.get('corpus_size', 0)} patterns")


def main():
    parser = argparse.ArgumentParser(
        description='Train PRISM-SLM on repository corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick --dry-run           # Quick eval, no save
  %(prog)s --full --output my_model.json  # Full training, save to file
  %(prog)s --quick --interactive       # Quick train + interactive mode
        """
    )
    parser.add_argument('--quick', action='store_true', help='Quick training (1000 patterns)')
    parser.add_argument('--full', action='store_true', help='Full training (all patterns)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode after training')
    parser.add_argument('--corpus', type=str,
                        default='benchmarks/codebase_slm/corpus/training_patterns.jsonl',
                        help='Path to training corpus')
    parser.add_argument('--context-size', type=int, default=3, help='Context window size')
    parser.add_argument('--output', '-o', type=str,
                        help='Save trained model to this path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Evaluate only, do not save model')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing model without backup')
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path

    print("=" * 60)
    print("Repository-Native SLM Training")
    print("=" * 60)

    if args.dry_run:
        print("⚠️  DRY RUN MODE - Model will NOT be saved")
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

    # Test queries - use Q: format to trigger trained pattern
    test_queries = [
        {'prompt': 'Q: Where is PageRank implemented? A:', 'expected': 'cortical analysis pagerank', 'category': 'file_location'},
        {'prompt': 'Q: What does compute_all do? A:', 'expected': 'computes all analysis', 'category': 'function'},
        {'prompt': 'from cortical.got import', 'expected': 'GoTManager', 'category': 'completion'},
        {'prompt': 'Q: How to create a task? A:', 'expected': 'python scripts got_utils', 'category': 'how_to'},
        {'prompt': 'Q: What is PRISM? A:', 'expected': 'Statistical Language Model', 'category': 'concept'},
        {'prompt': 'Q: Where is GoTManager defined? A:', 'expected': 'cortical got api', 'category': 'file_location'},
        {'prompt': 'Q: What does Hebbian learning mean? A:', 'expected': 'neurons that fire together', 'category': 'concept'},
        {'prompt': 'Q: What is the work priority order? A:', 'expected': 'Security Bugs Features', 'category': 'process'},
        {'prompt': 'Q: Where is TF-IDF implemented? A:', 'expected': 'cortical analysis tfidf', 'category': 'file_location'},
    ]

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print("(Note: Slashes become spaces due to tokenization)")

    correct = 0
    for q in test_queries:
        generated = model.generate(q['prompt'], max_tokens=20, temperature=0.3)
        # Check if key terms appear in output
        expected_terms = q['expected'].lower().split()
        generated_lower = generated.lower()
        matches = sum(1 for term in expected_terms if term in generated_lower)
        match_pct = matches / len(expected_terms) * 100 if expected_terms else 0
        is_match = match_pct >= 50

        if is_match:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"\n[{q['category']}] {status}")
        print(f"  Prompt: {q['prompt']}")
        print(f"  Generated: {generated}")
        print(f"  Expected terms: {q['expected']}")
        print(f"  Match: {matches}/{len(expected_terms)} terms ({match_pct:.0f}%)")

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {correct}/{len(test_queries)} queries matched (≥50% of terms)")
    print(f"{'=' * 60}")

    # Save model if --output specified (and not --dry-run)
    if args.output and not args.dry_run:
        model_path = Path(args.output)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / model_path

        # Backup existing model (unless --force)
        if model_path.exists() and not args.force:
            backup_existing_model(model_path)
        elif model_path.exists() and args.force:
            print("⚠️  --force specified, skipping backup")

        provenance = {
            'corpus_hash': compute_corpus_hash(patterns),
            'corpus_size': len(patterns),
            'corpus_path': str(corpus_path),
        }

        save_model(model, model_path, provenance)
    elif args.output and args.dry_run:
        print("\n⚠️  DRY RUN - Model was NOT saved (would save to: {})".format(args.output))
    elif not args.output and not args.dry_run:
        print("\n💡 Tip: Use --output <path> to save the trained model")

    # Interactive mode
    if args.interactive:
        interactive_mode(model)

    return 0


if __name__ == '__main__':
    sys.exit(main())
