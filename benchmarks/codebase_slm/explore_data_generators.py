#!/usr/bin/env python3
"""
Explore micro models as training data generators.

This explores:
1. SparkSLM: N-gram sequence generation
2. Woven Mind: Semantic expansion via activation spreading
3. Combined: Using both for synthetic data augmentation
"""

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.spark import NGramModel
from cortical.reasoning.woven_mind import WovenMind


def explore_spark_generation():
    """Explore SparkSLM's data generation capabilities."""
    print("\n" + "=" * 70)
    print("SPARKSML AS DATA GENERATOR")
    print("=" * 70)

    # Train on repository knowledge
    training_texts = [
        "PageRank is implemented in cortical analysis pagerank.py",
        "TF-IDF is implemented in cortical analysis tfidf.py",
        "GoTManager is the main API for Graph of Thought",
        "Import GoTManager from cortical.got",
        "Hebbian learning means neurons that fire together wire together",
        "The tokenizer splits text into tokens and bigrams",
        "Louvain clustering detects community structure",
        "Persistence saves and loads processor state",
        "Query expansion adds related terms to searches",
        "BM25 is the default scoring algorithm",
        "Use python scripts got_utils.py to manage tasks",
        "Tests are in tests directory organized by category",
        "Configuration is in cortical config.py",
        "Minicolumns store lateral and typed connections",
    ]

    # Create and train NGramModel directly
    print("\n1. TRAINING SPARK ON REPOSITORY KNOWLEDGE")
    print("-" * 40)
    ngram = NGramModel(n=3)
    ngram.train(training_texts)
    print(f"Trained on {len(training_texts)} texts")

    # Test completion generation
    print("\n2. COMPLETION GENERATION")
    print("-" * 40)

    prompts = [
        ["pagerank", "is"],
        ["import", "gotmanager"],
        ["hebbian", "learning"],
        ["the", "tokenizer"],
        ["cortical", "analysis"],
    ]

    for prompt in prompts:
        predictions = ngram.predict(prompt, top_k=3)
        sequence = ngram.predict_sequence(prompt, length=5)
        print(f"\nPrompt: {prompt}")
        print(f"  Top predictions: {predictions}")
        print(f"  Full sequence: {sequence}")

    # Generate synthetic training data
    print("\n3. SYNTHETIC DATA GENERATION")
    print("-" * 40)

    # Generate variations of existing patterns
    seed_patterns = [
        (["where", "is"], ["pagerank", "tfidf", "gotmanager", "tokenizer"]),
        (["import"], ["gotmanager", "ngram", "wovenmind"]),
        (["the"], ["tokenizer", "processor", "config", "query"]),
    ]

    synthetic_data = []
    for prefix, subjects in seed_patterns:
        for subject in subjects:
            prompt = prefix + [subject]
            sequence = ngram.predict_sequence(prompt, length=6)
            synthetic_data.append({
                'input': ' '.join(prompt),
                'generated': ' '.join(sequence),
                'type': 'spark_completion'
            })

    print(f"Generated {len(synthetic_data)} synthetic patterns:")
    for item in synthetic_data[:5]:
        print(f"  '{item['input']}' → '{item['generated']}'")

    return synthetic_data


def explore_woven_activation():
    """Explore Woven Mind's activation spreading for semantic expansion."""
    print("\n" + "=" * 70)
    print("WOVEN MIND AS SEMANTIC EXPANDER")
    print("=" * 70)

    # Train Woven Mind
    print("\n1. TRAINING WOVEN MIND")
    print("-" * 40)

    training_patterns = [
        "pagerank algorithm importance scoring graph",
        "tfidf term frequency inverse document",
        "gotmanager task decision sprint epic",
        "tokenizer split text tokens bigrams",
        "clustering louvain community detection",
        "persistence save load state json pickle",
        "query search expansion retrieval ranking",
        "hebbian learning neurons fire wire",
        "minicolumn lateral connections typed edges",
        "config configuration settings parameters",
    ]

    mind = WovenMind()
    for pattern in training_patterns:
        mind.train(pattern)
    print(f"Trained on {len(training_patterns)} patterns")

    # Test activation spreading
    print("\n2. ACTIVATION SPREADING (Semantic Expansion)")
    print("-" * 40)

    seed_terms = ["pagerank", "query", "task", "neurons"]

    expansions = []
    for term in seed_terms:
        result = mind.process([term])
        activated = list(result.activations)[:5]  # Top 5 activated nodes
        print(f"\nSeed: '{term}'")
        print(f"  Mode: {result.mode.name}")
        print(f"  Activations: {len(result.activations)} nodes")
        print(f"  Top activated: {activated}")
        if result.predictions:
            top_preds = sorted(result.predictions.items(), key=lambda x: -x[1])[:5]
            print(f"  Predictions: {top_preds}")
            expansions.append({
                'seed': term,
                'expansions': [p[0] for p in top_preds],
                'type': 'woven_activation'
            })

    # Generate semantic pairs
    print("\n3. SEMANTIC PAIR GENERATION")
    print("-" * 40)

    # Use activation to find related concepts
    concept_seeds = ["algorithm", "search", "learning", "connection"]

    semantic_pairs = []
    for seed in concept_seeds:
        result = mind.process([seed])
        if result.predictions:
            for related, score in sorted(result.predictions.items(), key=lambda x: -x[1])[:3]:
                pair = {
                    'concept1': seed,
                    'concept2': related,
                    'score': score,
                    'type': 'semantic_pair'
                }
                semantic_pairs.append(pair)
                print(f"  {seed} ↔ {related} (score: {score:.3f})")

    return expansions, semantic_pairs


def explore_combined_generation():
    """Combine SparkSLM generation with Woven Mind semantics."""
    print("\n" + "=" * 70)
    print("COMBINED DATA GENERATION PIPELINE")
    print("=" * 70)

    # Initialize both models
    ngram = NGramModel(n=3)
    mind = WovenMind()

    # Shared training corpus
    corpus = [
        "PageRank computes importance scores using graph structure",
        "TF-IDF measures term relevance across documents",
        "GoTManager provides task and decision tracking",
        "The tokenizer processes text into analyzable units",
        "Query expansion improves search recall",
        "Hebbian learning strengthens co-activated connections",
    ]

    print("\n1. TRAINING BOTH MODELS")
    print("-" * 40)
    ngram.train(corpus)
    for text in corpus:
        mind.train(text)
    print(f"Both models trained on {len(corpus)} texts")

    # Pipeline: Woven Mind identifies related concepts, Spark generates
    print("\n2. SEMANTIC-GUIDED GENERATION")
    print("-" * 40)

    query_seeds = ["pagerank", "tokenizer", "query"]

    augmented_data = []
    for seed in query_seeds:
        # Step 1: Woven Mind finds related concepts
        result = mind.process([seed.lower()])
        related = []
        if result.predictions:
            related = [p[0] for p in sorted(result.predictions.items(), key=lambda x: -x[1])[:3]]

        # Step 2: NGram generates completions for seed + related
        prompt = [seed, "is"]
        sequence = ngram.predict_sequence(prompt, length=5)

        augmented_data.append({
            'seed': seed,
            'related_concepts': related,
            'generated': ' '.join(sequence),
            'type': 'semantic_guided'
        })

        print(f"\nSeed: '{seed}'")
        print(f"  Related (Woven Mind): {related}")
        print(f"  Generated (NGram): '{' '.join(sequence)}'")

    # Generate Q&A pairs using both
    print("\n3. Q&A PAIR AUGMENTATION")
    print("-" * 40)

    qa_templates = [
        ("Where is {concept} implemented?", ["{concept}", "is", "implemented", "in"]),
        ("What does {concept} do?", ["{concept}"]),
        ("How do I use {concept}?", ["use", "{concept}"]),
    ]

    concepts = ["pagerank", "gotmanager", "tokenizer"]

    qa_pairs = []
    for concept in concepts:
        # Get semantic context from Woven Mind
        result = mind.process([concept.lower()])
        surprise = result.surprise.magnitude if result.surprise else 0.5

        # If low surprise (familiar), generate confidently
        # If high surprise (novel), mark for review
        for q_template, a_prefix_template in qa_templates:
            question = q_template.format(concept=concept)
            a_prefix = [t.format(concept=concept) for t in a_prefix_template]
            answer_tokens = ngram.predict_sequence(a_prefix, length=5)
            answer = ' '.join(answer_tokens)

            qa_pairs.append({
                'question': question,
                'answer': answer,
                'confidence': 1.0 - surprise,  # Low surprise = high confidence
                'type': 'generated_qa'
            })

    print(f"Generated {len(qa_pairs)} Q&A pairs:")
    for qa in qa_pairs[:6]:
        conf_str = "HIGH" if qa['confidence'] > 0.5 else "LOW"
        print(f"  [{conf_str}] Q: {qa['question']}")
        print(f"       A: {qa['answer']}")

    return augmented_data, qa_pairs


def main():
    """Run all explorations."""
    print("=" * 70)
    print("MICRO MODELS AS TRAINING DATA GENERATORS")
    print("=" * 70)

    # Explore each approach
    spark_data = explore_spark_generation()
    woven_expansions, semantic_pairs = explore_woven_activation()
    augmented, qa_pairs = explore_combined_generation()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: DATA GENERATION CAPABILITIES")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION COMPARISON                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  SPARKSML (N-gram Generator)                                         │
│  ├── Strengths:                                                      │
│  │   • Generates coherent sequences from prefixes                    │
│  │   • Can complete partial patterns                                 │
│  │   • Fast and deterministic                                        │
│  │                                                                   │
│  └── Best for:                                                       │
│      • Completion-style training data                                │
│      • Variations of existing patterns                               │
│      • "X is implemented in Y" style facts                           │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  WOVEN MIND (Semantic Expander)                                      │
│  ├── Strengths:                                                      │
│  │   • Identifies semantically related concepts                      │
│  │   • Detects novelty (what needs more training data)               │
│  │   • Builds abstractions for concept grouping                      │
│  │                                                                   │
│  └── Best for:                                                       │
│      • Semantic pair generation                                      │
│      • Concept clustering for training organization                  │
│      • Confidence scoring (familiar vs novel)                        │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  COMBINED PIPELINE (Recommended)                                     │
│  ├── Step 1: Woven Mind identifies related concepts                  │
│  ├── Step 2: Woven Mind scores confidence (surprise-based)           │
│  ├── Step 3: SparkSLM generates completions                          │
│  └── Step 4: Filter by confidence, augment training corpus           │
│                                                                       │
│  Benefits:                                                           │
│  • Semantic coherence from Woven Mind                                │
│  • Fluent generation from SparkSLM                                   │
│  • Quality filtering via surprise detection                          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
    """)

    print("\nData generated in this exploration:")
    print(f"  • SparkSLM completions: {len(spark_data)}")
    print(f"  • Woven Mind expansions: {len(woven_expansions)}")
    print(f"  • Semantic pairs: {len(semantic_pairs)}")
    print(f"  • Combined Q&A pairs: {len(qa_pairs)}")

    print("\nRECOMMENDED NEXT STEPS:")
    print("  1. Build data augmentation pipeline using combined approach")
    print("  2. Generate 10x synthetic variations of successful patterns")
    print("  3. Use Woven Mind surprise to weight training data quality")
    print("  4. Re-train PRISM-SLM with augmented corpus")
    print("  5. Re-run benchmarks to measure improvement")


if __name__ == "__main__":
    main()
