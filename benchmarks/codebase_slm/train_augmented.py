#!/usr/bin/env python3
"""
Train PRISM-SLM with augmented corpus and run benchmarks.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.spark import NGramModel


def load_augmented_corpus():
    """Load the augmented training corpus."""
    corpus_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "data" / "augmented_corpus.txt"

    if not corpus_path.exists():
        print("ERROR: Augmented corpus not found. Run data_augmentation.py first.")
        return []

    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} augmented training lines")
    return lines


def load_existing_patterns():
    """Load existing training patterns."""
    patterns_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "corpus" / "training_patterns.jsonl"

    if not patterns_path.exists():
        print("No existing patterns found")
        return []

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
    return patterns


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


def save_model(model, path):
    """Save trained model."""
    model_data = {
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

    with open(path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"\nModel saved to {path}")


def main():
    print("=" * 60)
    print("PRISM-SLM AUGMENTED TRAINING")
    print("=" * 60)

    # Load corpora
    augmented = load_augmented_corpus()
    existing = load_existing_patterns()

    # Combine (augmented has higher weight due to oversampling)
    combined = augmented + existing
    print(f"\nTotal training corpus: {len(combined)} patterns")

    # Train
    model = train_model(combined)

    # Test
    results = test_model(model)

    # Save
    model_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "prism_augmented.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)

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


if __name__ == "__main__":
    main()
