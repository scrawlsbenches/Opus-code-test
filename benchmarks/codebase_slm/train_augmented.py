#!/usr/bin/env python3
"""
Train PRISM-SLM with augmented corpus and run benchmarks.

================================================================================
TRAINING PIPELINE OVERVIEW
================================================================================

This script is STEP 2 of a two-step process:

  STEP 1: Generate corpus from codebase (REQUIRED FIRST!)
  --------------------------------------------------------
  python -m benchmarks.codebase_slm.generate_corpus --full

  This extracts training patterns from:
    - 149 Python files in cortical/ → functions, classes, imports
    - 254 Markdown files in docs/ + samples/ → sections, Q&A pairs
    - GoT entities → tasks, decisions, metadata

  Output: corpus/training_patterns.jsonl (~35,000 patterns)


  STEP 2: Train model (THIS SCRIPT)
  ----------------------------------
  python -m benchmarks.codebase_slm.train_augmented --dry-run

  This combines:
    - corpus/training_patterns.jsonl (~35,000 patterns) ← from Step 1
    - data/augmented_corpus.txt (~2,000 lines) ← curated definitions

  Output: models/prism_augmented.json (~13MB, 37K docs, 15K vocab)


================================================================================
WHAT WENT WRONG (Dec 27, 2025 Incident)
================================================================================

An agent ran this script WITHOUT running generate_corpus.py first:
  - corpus/training_patterns.jsonl didn't exist (gitignored, not tracked)
  - Script printed "No existing patterns found" and continued
  - Trained on only 2,094 lines instead of 37,676
  - Result: 329 vocab model replaced 15,814 vocab model (98% loss!)

The model was restored from git, and this warning was added.

See: samples/memories/2025-12-27-knowledge-transfer-prism-model-incident.md


================================================================================
DATA SOURCES
================================================================================

1. corpus/training_patterns.jsonl (GENERATED, gitignored)
   - Created by: generate_corpus.py --full
   - Contains: ~35,000 Q&A and completion patterns
   - Format: {"pattern_type": "qa", "input_text": "...", "target_text": "..."}
   - MUST BE REGENERATED after codebase changes

2. data/augmented_corpus.txt (TRACKED in git)
   - Contains: ~2,000 curated concept definitions
   - Format: Plain text, one pattern per line
   - Example: "PageRank is a graph algorithm for computing node importance"

3. samples/knowledge-base/*.md (TRACKED in git)
   - Contains: 3,144 lines of curated Q&A pairs
   - Used by generate_corpus.py to create patterns


================================================================================
USAGE
================================================================================

# ALWAYS run generate_corpus.py first!
python -m benchmarks.codebase_slm.generate_corpus --full

# Then train with dry-run to verify (RECOMMENDED)
python -m benchmarks.codebase_slm.train_augmented --dry-run

# Save to custom path (safest)
python -m benchmarks.codebase_slm.train_augmented --output models/my_model.json

# Save to default path (creates timestamped backup first)
python -m benchmarks.codebase_slm.train_augmented

# Force overwrite without backup (use with caution!)
python -m benchmarks.codebase_slm.train_augmented --force


================================================================================
SAFEGUARDS
================================================================================

1. --dry-run: Evaluate without saving anything
2. --output: Explicit path prevents accidental overwrites
3. Auto-backup: Creates models/backups/prism_augmented_TIMESTAMP.json
4. Provenance: Saves corpus hash and metadata in model's _provenance field
5. Loud warning: Shows error if corpus/training_patterns.jsonl is missing
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.spark import NGramModel

# Default output path
DEFAULT_MODEL_PATH = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "prism_augmented.json"
BACKUP_DIR = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "backups"


def load_augmented_corpus():
    """Load the augmented training corpus."""
    corpus_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "data" / "augmented_corpus.txt"

    if not corpus_path.exists():
        print("ERROR: Augmented corpus not found. Run data_augmentation.py first.")
        return [], None

    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} augmented training lines")
    return lines, str(corpus_path)


def load_existing_patterns():
    """Load existing training patterns."""
    patterns_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "corpus" / "training_patterns.jsonl"

    if not patterns_path.exists():
        print()
        print("=" * 60)
        print("⚠️  WARNING: No training corpus found!")
        print("=" * 60)
        print()
        print("The corpus/training_patterns.jsonl file is missing.")
        print("This means training will only use augmented_corpus.txt (~2K lines)")
        print("instead of the full corpus (~35K patterns).")
        print()
        print("To generate the corpus, run:")
        print("  python -m benchmarks.codebase_slm.generate_corpus --full")
        print()
        print("Then re-run this training script.")
        print()
        print("Proceeding with limited data (NOT RECOMMENDED)...")
        print("=" * 60)
        print()
        return [], None

    patterns = []
    with open(patterns_path) as f:
        for line in f:
            try:
                p = json.loads(line)
                # Format as training text
                ptype = p.get('pattern_type', '')
                input_text = p.get('input_text', '')
                target = p.get('target_text', '')

                if ptype == 'qa':
                    text = f"Q: {input_text} A: {target}"
                else:
                    text = f"{input_text} {target}"

                patterns.append(text)
            except:
                continue

    print(f"Loaded {len(patterns)} existing patterns")
    return patterns, str(patterns_path)


def train_model(corpus):
    """Train NGramModel on combined corpus."""
    print(f"\nTraining on {len(corpus)} patterns...")

    model = NGramModel(n=3)
    model.train(corpus)

    print(f"  Vocabulary size: {len(model.vocab)}")
    print(f"  Context count: {len(model.counts)}")
    print(f"  Total tokens: {model.total_tokens}")

    return model


def test_model(model):
    """Test the trained model on key queries."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    test_cases = [
        # Concept explanations (previously 0%)
        {
            'prompt': ['what', 'is', 'pagerank'],
            'expected_terms': ['graph', 'algorithm', 'importance', 'scores'],
            'category': 'concept'
        },
        {
            'prompt': ['what', 'is', 'hebbian'],
            'expected_terms': ['neurons', 'fire', 'together', 'wire'],
            'category': 'concept'
        },
        {
            'prompt': ['what', 'is', 'tfidf'],
            'expected_terms': ['term', 'frequency', 'document', 'relevance'],
            'category': 'concept'
        },

        # File locations (previously 87.5%)
        {
            'prompt': ['where', 'is', 'pagerank'],
            'expected_terms': ['cortical', 'analysis', 'pagerank'],
            'category': 'file_location'
        },
        {
            'prompt': ['where', 'is', 'gotmanager'],
            'expected_terms': ['cortical', 'got', 'api'],
            'category': 'file_location'
        },

        # Hierarchical
        {
            'prompt': ['what', 'type', 'is', 'pagerank'],
            'expected_terms': ['algorithm', 'type'],
            'category': 'hierarchical'
        },
    ]

    results = {'concept': [], 'file_location': [], 'hierarchical': []}

    for tc in test_cases:
        prompt = tc['prompt']
        expected = tc['expected_terms']
        category = tc['category']

        # Generate completion
        generated = model.predict_sequence(prompt, length=10)
        generated_text = ' '.join(generated).lower()

        # Score
        matches = sum(1 for term in expected if term in generated_text)
        score = matches / len(expected) if expected else 0

        results[category].append(score)

        print(f"\n[{category}] Prompt: {' '.join(prompt)}")
        print(f"  Generated: {generated_text}")
        print(f"  Expected terms: {expected}")
        print(f"  Match score: {score:.0%}")

    # Summary
    print("\n" + "=" * 60)
    print("CATEGORY SCORES")
    print("=" * 60)

    for category, scores in results.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {category}: {avg:.0%}")

    overall = sum(sum(s) for s in results.values()) / sum(len(s) for s in results.values())
    print(f"\n  OVERALL: {overall:.0%}")

    return results


def backup_existing_model(model_path: Path) -> Path | None:
    """Create a backup of the existing model before overwriting."""
    if not model_path.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"prism_augmented_{timestamp}.json"
    backup_path = BACKUP_DIR / backup_name

    shutil.copy2(model_path, backup_path)
    print(f"⚠️  Backed up existing model to: {backup_path}")

    # Also keep only last 5 backups to avoid bloat
    backups = sorted(BACKUP_DIR.glob("prism_augmented_*.json"))
    if len(backups) > 5:
        for old_backup in backups[:-5]:
            old_backup.unlink()
            print(f"   Removed old backup: {old_backup.name}")

    return backup_path


def compute_corpus_hash(corpus: list) -> str:
    """Compute a hash of the training corpus for provenance."""
    content = "\n".join(sorted(set(corpus)))  # Dedupe and sort for consistency
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_model(model, path: Path, provenance: dict):
    """Save trained model with provenance metadata."""
    model_data = {
        # Provenance metadata (new!)
        '_provenance': {
            'trained_at': datetime.now().isoformat(),
            'corpus_hash': provenance.get('corpus_hash', 'unknown'),
            'corpus_size': provenance.get('corpus_size', 0),
            'sources': provenance.get('sources', []),
            'script': 'train_augmented.py',
            'model_type': 'NGramModel',
        },
        # Model data
        'vocab': list(model.vocab),
        'counts': {
            ' '.join(ctx): dict(counter)
            for ctx, counter in model.counts.items()
        },
        'context_totals': {
            ' '.join(ctx): total
            for ctx, total in model.context_totals.items()
        },
        'total_tokens': model.total_tokens,
        'total_documents': model.total_documents,
        'n': model.n,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"\n✓ Model saved to {path}")
    print(f"  Corpus hash: {provenance.get('corpus_hash', 'unknown')}")
    print(f"  Corpus size: {provenance.get('corpus_size', 0)} patterns")


def main():
    parser = argparse.ArgumentParser(
        description='Train PRISM-SLM with augmented corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run              # Evaluate only, don't save
  %(prog)s --output my_model.json # Save to custom path
  %(prog)s                        # Save to default (with backup)
  %(prog)s --force                # Overwrite without backup
        """
    )
    parser.add_argument('--output', '-o', type=str,
                        help='Output path for trained model (default: models/prism_augmented.json)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Evaluate only, do not save model')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing model without backup')
    parser.add_argument('--no-existing', action='store_true',
                        help='Train only on augmented corpus, skip existing patterns')
    args = parser.parse_args()

    print("=" * 60)
    print("PRISM-SLM AUGMENTED TRAINING")
    print("=" * 60)

    if args.dry_run:
        print("⚠️  DRY RUN MODE - Model will NOT be saved")

    # Load corpora
    augmented, aug_source = load_augmented_corpus()

    if args.no_existing:
        existing, exist_source = [], None
        print("Skipping existing patterns (--no-existing)")
    else:
        existing, exist_source = load_existing_patterns()

    # Combine (augmented has higher weight due to oversampling)
    combined = augmented + existing
    print(f"\nTotal training corpus: {len(combined)} patterns")

    if not combined:
        print("ERROR: No training data available")
        return 1

    # Train
    model = train_model(combined)

    # Test
    results = test_model(model)

    # Save (unless dry-run)
    if not args.dry_run:
        # Determine output path
        if args.output:
            model_path = Path(args.output)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / model_path
        else:
            model_path = DEFAULT_MODEL_PATH

        # Backup existing model (unless --force)
        if model_path.exists() and not args.force:
            backup_existing_model(model_path)
        elif model_path.exists() and args.force:
            print("⚠️  --force specified, skipping backup")

        # Build provenance
        sources = []
        if aug_source:
            sources.append(aug_source)
        if exist_source:
            sources.append(exist_source)

        provenance = {
            'corpus_hash': compute_corpus_hash(combined),
            'corpus_size': len(combined),
            'sources': sources,
        }

        save_model(model, model_path, provenance)
    else:
        print("\n⚠️  DRY RUN - Model was NOT saved")

    # Compare with baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)

    baseline_results = {
        'file_location': 0.875,
        'concept': 0.0,
        'how_to': 0.50,
        'completion': 0.50,
        'process': 0.0,
    }

    print("\n| Category      | Baseline | Augmented | Change |")
    print("|---------------|----------|-----------|--------|")

    for category in ['concept', 'file_location']:
        baseline = baseline_results.get(category, 0)
        current = sum(results[category]) / len(results[category]) if results.get(category) else 0
        change = current - baseline
        change_str = f"+{change:.0%}" if change >= 0 else f"{change:.0%}"
        print(f"| {category:13} | {baseline:.0%}      | {current:.0%}       | {change_str:6} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
